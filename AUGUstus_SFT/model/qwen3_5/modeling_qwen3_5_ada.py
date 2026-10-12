"""
Qwen3.5 with DiT-style Adaptive RMSNorm (scale/shift).

Extends ``modeling_qwen3_5.py`` with:
  - ``AdaRMSNorm``: projects the conditioning vector to (scale, shift)
    via a single Linear(cond_dim -> hidden_size * 2).
  - ``Qwen3_5AdaDecoderLayer``: decoder layer with adaptive residuals +
    action_FFN / action_QKV routing (full_attention layers only).
  - ``Qwen3_5AdaTextModel`` / ``Qwen3_5AdaModel`` /
    ``Qwen3_5AdaForConditionalGeneration``: full model wrappers that
    forward ``adarms_cond`` through all layers.

For linear_attention (GatedDeltaNet) layers:
  - AdaRMSNorm conditioning is applied normally.
  - action_FFN routing applies to the MLP.
  - action_QKV is NOT applied (GatedDeltaNet has a different QKV structure).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5MLP,
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5Model,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ModelOutputWithPast,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Adaptive RMSNorm (same interface as Qwen2.5-VL Ada variant)
# ======================================================================

class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with scale and shift (DiT / SD3 style).

    When ``cond=None``, degrades to standard RMSNorm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        # Qwen3.5 RMSNorm uses (1 + weight) scaling (zeros init)
        self.weight = nn.Parameter(torch.zeros(hidden_size))

        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, hidden_size * 2)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
        else:
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states: torch.Tensor,
                cond: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dtype = hidden_states.dtype
        normed = self._norm(hidden_states.float())

        if self.dense is None or cond is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.to(dtype), None

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)

        scale, shift = torch.chunk(modulation, 2, dim=-1)

        normed = normed * (1.0 + self.weight.float()) * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), None


def _action_span(spec, seq_len: int):
    """Resolve an action-routing ``spec`` to a contiguous ``(start, end)`` span.

    ``spec`` selects which tokens are *action* tokens (routed through the
    action MLP / Q-K-V-O projections):

      * ``-1``            : every token is an action token (cached inference).
      * ``N`` (int > 0)   : the LAST ``N`` tokens are action tokens (self-CoT
                            layout ``[ctx | BOT | cot | EOT | actions]``).
      * ``(start, length)``: the contiguous span ``[start, start+length)`` are
                            action tokens (action-first layout ``[ctx | actions | BOT |
                            cot | EOT]`` where the actions sit in the middle).

    Returns ``(start, end)`` with ``0 <= start <= end <= seq_len``.
    """
    if isinstance(spec, (tuple, list)):
        start = int(spec[0])
        end = start + int(spec[1])
    elif spec == -1:
        start, end = 0, seq_len
    else:
        start, end = seq_len - int(spec), seq_len
    return start, end


def _route_action_segments(x, spec, text_fn, action_fn):
    """Apply ``action_fn`` to the action span of ``x`` and ``text_fn`` elsewhere.

    Generalises the legacy "last N tokens" routing to also support an action
    span in the MIDDLE of the sequence (action-first). When ``action_fn is text_fn`` the
    result is identical to ``text_fn(x)`` (per-token Linear), preserving the
    shared-projection fallback used for ``QO``-only / ``KV``-only configs.
    """
    start, end = _action_span(spec, x.shape[1])
    parts = []
    if start > 0:
        parts.append(text_fn(x[:, :start]))
    parts.append(action_fn(x[:, start:end]))
    if end < x.shape[1]:
        parts.append(text_fn(x[:, end:]))
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


# ======================================================================
# Attention with MoT-style action Q/K/V/O routing
# ======================================================================

class Qwen3_5AdaAttention(Qwen3_5Attention):
    """Qwen3.5 attention with optional MoT-style action Q/K/V/O routing.

    Identical to the stock ``Qwen3_5Attention`` when ``action_QKV == 0``. When
    ``action_QKV`` is set, the action tokens are routed through the separate
    ``action_{q,k,v,o}_proj`` projections (created on this module when
    ``special_QKVO`` is enabled in the model); a sub-projection that does not
    exist falls back to the shared text projection (so ``QO``-only / ``KV``-only
    configurations work).

      * ``action_QKV == -1`` (cached inference): EVERY current token is an action
        token (the text/CoT prefix lives in the KV cache, already projected by
        the text K/V), so all tokens use the action projections.
      * ``action_QKV == N`` (training): the LAST ``N`` tokens are action tokens
        and the leading ``q_len - N`` are the text/CoT prefix.

    Qwen3.5 specifics preserved: the ``q_proj`` output carries a gate that is
    ``torch.sigmoid``-applied after attention, and ``q_norm`` / ``k_norm`` are
    applied per head. ``action_q_proj`` is a deep copy of ``q_proj`` so it
    produces the action gate as well; ``q_norm`` / ``k_norm`` stay shared.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        action_QKV: int = 0,
        detach_prefix_len: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # ---- Q/K/V projections (optionally MoT-routed) ----
        # ``action_QKV`` may be -1 (all tokens), an int N (last N tokens), or a
        # ``(start, length)`` tuple (action-first: actions in the middle). A missing
        # action sub-projection falls back to the shared text projection.
        if action_QKV:
            has_q = hasattr(self, "action_q_proj")
            has_k = hasattr(self, "action_k_proj")
            has_v = hasattr(self, "action_v_proj")
            q_out = _route_action_segments(
                hidden_states, action_QKV, self.q_proj,
                self.action_q_proj if has_q else self.q_proj,
            )
            k_out = _route_action_segments(
                hidden_states, action_QKV, self.k_proj,
                self.action_k_proj if has_k else self.k_proj,
            )
            v_out = _route_action_segments(
                hidden_states, action_QKV, self.v_proj,
                self.action_v_proj if has_v else self.v_proj,
            )
        else:
            q_out = self.q_proj(hidden_states)
            k_out = self.k_proj(hidden_states)
            v_out = self.v_proj(hidden_states)

        # is_bp_combine=False (option A): the action tokens attend to the prefix
        # keys/values, so the projection that PRODUCES the prefix K/V would
        # otherwise receive a diffusion gradient even when the prefix INPUT is
        # detached. We detach the prefix K/V OUTPUT to cut that gradient — but
        # ONLY when a SEPARATE action K/V projection exists (special_KV). In that
        # case the prefix uses the text k/v_proj and we want it trained by the LM
        # loss ONLY. When K/V are SHARED (no action_k_proj / action_v_proj), the
        # SAME projection serves both prefix and action tokens, so it must stay
        # trained by BOTH losses: we do NOT detach, letting the diffusion gradient
        # flow through the prefix K/V exactly like the single joint pass. Detach
        # never changes forward values. Prefix Q/O never need a detach: the prefix
        # attention output flows only into the (discarded) prefix residual, so a
        # shared q/o_proj still gets the diffusion gradient from the action tokens.
        if detach_prefix_len:
            n = detach_prefix_len
            if hasattr(self, "action_k_proj"):
                k_out = torch.cat([k_out[:, :n].detach(), k_out[:, n:]], dim=1)
            if hasattr(self, "action_v_proj"):
                v_out = torch.cat([v_out[:, :n].detach(), v_out[:, n:]], dim=1)

        # ---- gate split + per-head norms (Qwen3.5) ----
        query_states, gate = torch.chunk(
            q_out.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(k_out.view(hidden_shape)).transpose(1, 2)
        value_states = v_out.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        # ---- O projection (optionally MoT-routed) ----
        if action_QKV and hasattr(self, "action_o_proj"):
            attn_output = _route_action_segments(
                attn_output, action_QKV, self.o_proj, self.action_o_proj,
            )
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# ======================================================================
# Decoder Layer with AdaRMSNorm + action_FFN / action_QKV
# ======================================================================

class Qwen3_5AdaDecoderLayer(nn.Module):
    """Decoder layer with adaptive normalization + action routing.

    Per-norm AdaRMSNorm with scale + shift (DiT style, no gate).

    For ``full_attention`` layers: action_QKV routes last N tokens through
    separate Q/O projections.

    For ``linear_attention`` (GatedDeltaNet) layers: action_QKV is ignored.
    action_FFN routing works for both layer types.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int,
                 cond_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5AdaAttention(config, layer_idx)

        self.mlp = Qwen3_5MLP(config, config.intermediate_size)

        self.input_layernorm = AdaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            cond_dim=cond_dim,
        )
        self.post_attention_layernorm = AdaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        action_FFN: int = 0,
        action_QKV: int = 0,
        detach_prefix_len: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        # Forward the prefix-detach length to the attention (via kwargs) so the
        # text K/V projections get no diffusion gradient (is_bp_combine=False).
        if detach_prefix_len:
            kwargs["detach_prefix_len"] = detach_prefix_len
        return self._forward_adanorm(
            hidden_states, position_embeddings, attention_mask, position_ids,
            past_key_values, adarms_cond, action_FFN, action_QKV, **kwargs,
        )

    def _do_attention(self, hidden_states, position_embeddings, attention_mask,
                      position_ids, past_key_values, action_QKV, **kwargs):
        """Dispatch to full_attention or linear_attention."""
        if self.layer_type == "linear_attention":
            # GatedDeltaNet doesn't accept use_cache or other full-attention kwargs
            return self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            out, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                action_QKV=action_QKV,
                **kwargs,
            )
            return out

    def _apply_mlp(self, hidden_states, action_FFN):
        # ``action_FFN`` may be -1 (all tokens), an int N (last N tokens), or a
        # ``(start, length)`` tuple (action-first: actions in the middle of the sequence).
        if action_FFN and hasattr(self, 'action_mlp'):
            return _route_action_segments(
                hidden_states, action_FFN, self.mlp, self.action_mlp,
            )
        return self.mlp(hidden_states)

    def _forward_adanorm(self, hidden_states, position_embeddings, attention_mask,
                         position_ids, past_key_values, adarms_cond,
                         action_FFN, action_QKV, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.input_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self._do_attention(
            hidden_states, position_embeddings, attention_mask,
            position_ids, past_key_values, action_QKV, **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states, _ = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self._apply_mlp(hidden_states, action_FFN)
        hidden_states = residual + hidden_states
        return hidden_states


# ======================================================================
# Text Model with AdaRMSNorm
# ======================================================================

class Qwen3_5AdaTextModel(nn.Module):
    """Qwen3.5 text decoder with AdaRMSNorm decoder layers.

    Replaces standard layers with ``Qwen3_5AdaDecoderLayer``.
    Forwards ``adarms_cond``, ``action_FFN``, ``action_QKV`` through all layers.
    """

    def __init__(self, config: Qwen3_5TextConfig, cond_dim: Optional[int] = None,
                 ada_layer_mode: str = "full"):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        self.ada_layer_mode = ada_layer_mode

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)

        layers = []
        self._ada_layer_flags = []
        for layer_idx in range(config.num_hidden_layers):
            if ada_layer_mode == "half" and layer_idx % 2 == 1:
                layer_cond_dim = None
                self._ada_layer_flags.append(False)
            else:
                layer_cond_dim = cond_dim
                self._ada_layer_flags.append(True)
            layers.append(Qwen3_5AdaDecoderLayer(
                config, layer_idx, cond_dim=layer_cond_dim,
            ))
        self.layers = nn.ModuleList(layers)

        self.norm = AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                               cond_dim=cond_dim)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        action_FFN: int = 0,
        action_QKV: int = 0,
        output_hidden_states: Optional[bool] = None,
        linear_attn_padding_mask: Optional[torch.Tensor] = None,
        detach_prefix_len: int = 0,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Position IDs handling (matches original Qwen3_5TextModel logic):
        # - When position_ids=None: create 4-tuple [text, temporal, height, width]
        # - When position_ids is 3D with shape[0]==4: extract text dim, use rest for RoPE
        # - When position_ids is 3D with shape[0]==3 (from get_rope_index): text=None
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]  # [3, B, S] for temporal/height/width
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        # Causal mask for full_attention layers
        if create_causal_mask is not None:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
        else:
            causal_mask = attention_mask

        # Linear-attention (GatedDeltaNet) layers can ONLY consume a 2D padding
        # mask [B, L] (see apply_mask_to_padding_states, which requires
        # attention_mask.shape[1] > 1). The 4D Transfusion mask [B, 1, L, L] is
        # silently ignored there, so left-padding would pollute the recurrent
        # state. When the caller supplies an explicit 2D padding mask we use it;
        # otherwise we fall back to the original behaviour, guarding against a
        # 4D mask being mis-used (which would disable padding masking).
        if linear_attn_padding_mask is not None:
            linear_attn_mask = self._update_linear_attn_mask(
                linear_attn_padding_mask, past_key_values,
            )
        elif attention_mask is not None and attention_mask.dim() == 4:
            # No explicit 2D mask given but a 4D mask was passed: the linear
            # layers cannot use it. Derive validity (a key token is visible if
            # any query can attend to it) to recover a 2D padding mask.
            with torch.no_grad():
                valid = (attention_mask[:, 0] > torch.finfo(attention_mask.dtype).min).any(dim=1)
            linear_attn_mask = self._update_linear_attn_mask(valid.to(inputs_embeds.dtype), past_key_values)
        else:
            linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        all_hidden_states = () if output_hidden_states else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            # is_bp_combine=False (option A): detach the first ``detach_prefix_len``
            # positions (the text/CoT prefix) at the input of EVERY layer. The
            # action suffix keeps its gradient, so the diffusion loss can reach
            # the action FFN / projections but NOT the prefix (text) parameters:
            # prefix K/V seen by the action attention are detached, and the text
            # MLP runs on a detached prefix whose output is re-detached before the
            # next layer. RMSNorm is per-token, so the final-layer prefix never
            # leaks into the action slice either. Equivalent to attending to a
            # frozen prefix KV cache, but recomputed fresh each pass so it is
            # safe for both full- and linear-attention (GatedDeltaNet) layers.
            if detach_prefix_len:
                hidden_states = torch.cat(
                    [hidden_states[:, :detach_prefix_len].detach(),
                     hidden_states[:, detach_prefix_len:]],
                    dim=1,
                )

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_type = self.config.layer_types[layer_idx]
            layer_mask = linear_attn_mask if layer_type == "linear_attention" else causal_mask
            layer_cond = adarms_cond if self._ada_layer_flags[layer_idx] else None

            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    position_embeddings,
                    layer_mask,
                    text_position_ids,
                    past_key_values,
                    layer_cond,
                    action_FFN,
                    action_QKV,
                    detach_prefix_len,
                    use_reentrant=False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    adarms_cond=layer_cond,
                    action_FFN=action_FFN,
                    action_QKV=action_QKV,
                    detach_prefix_len=detach_prefix_len,
                    use_cache=use_cache,
                    **kwargs,
                )

        hidden_states, _ = self.norm(hidden_states, cond=adarms_cond)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


# ======================================================================
# Full Model wrappers
# ======================================================================

class Qwen3_5AdaModel(Qwen3_5Model):
    """Qwen3.5 multimodal model with AdaRMSNorm text backbone."""

    def __init__(self, config: Qwen3_5Config, cond_dim: Optional[int] = None,
                 ada_layer_mode: str = "full"):
        # Skip Qwen3_5Model.__init__ to avoid creating standard TextModel
        super(Qwen3_5Model, self).__init__(config)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5AdaTextModel(
            config.text_config, cond_dim=cond_dim,
            ada_layer_mode=ada_layer_mode,
        )
        self.rope_deltas = None
        self.post_init()


class Qwen3_5AdaForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 with Ada variant for Transfusion integration."""

    def __init__(self, config: Qwen3_5Config, cond_dim: Optional[int] = None,
                 ada_layer_mode: str = "full"):
        super(Qwen3_5ForConditionalGeneration, self).__init__(config)
        self.model = Qwen3_5AdaModel(
            config, cond_dim=cond_dim,
            ada_layer_mode=ada_layer_mode,
        )
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False,
        )
        self._cond_dim = cond_dim
        self._ada_layer_mode = ada_layer_mode
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args,
                        cond_dim=None, ada_layer_mode="full", **kwargs):
        """Load HF checkpoint, then build local Ada model."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as _HF

        hf_model = _HF.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        config = hf_model.config

        if cond_dim is None:
            cond_dim = config.text_config.hidden_size

        local_model = cls(config, cond_dim=cond_dim,
                          ada_layer_mode=ada_layer_mode)

        hf_sd = hf_model.state_dict()
        local_sd = local_model.state_dict()

        loaded, skipped = 0, 0
        for key in hf_sd:
            if key in local_sd and hf_sd[key].shape == local_sd[key].shape:
                local_sd[key] = hf_sd[key]
                loaded += 1
            else:
                skipped += 1

        local_model.load_state_dict(local_sd, strict=False)
        logger.info(
            f"Loaded {loaded} params from HF, skipped {skipped} "
            f"(AdaRMSNorm dense layers are zero-initialized)."
        )

        del hf_model
        return local_model
