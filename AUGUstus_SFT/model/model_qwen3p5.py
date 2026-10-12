"""
Transfusion Model: Qwen3.5 backbone + Diffusion/FlowMatching Trajectory Prediction.

Standalone version for Qwen3.5, independent from the Qwen2.5-VL model.py.

Faithful to the Transfusion paper (Zhou et al., 2024), Figures 1/3/4,
with extensions for Chain-of-Thought (CoT) and Flow Matching.

Sequence layout during forward pass (with optional CoT):

  +---------------+-------+----------+-------+----------------+
  | text+vision   | <BOT> | CoT text | <EOT> |  actions (T)   |
  +---------------+-------+----------+-------+----------------+
  |     causal    |          causal          |  bidirectional mask  |
                                    
Without CoT (empty CoT):

    +---------------+-------+-------+---------+--------------+
    | text+vision   | <BOT> | <EOT> | noised actions (T)     |
    +---------------+-------+-------+--------------------------+

Action/time embeddings follow openpi/pi0_pytorch.py:
  - Single-Linear action projection
  - Sine-cosine positional time encoding with concat/add MLP fusion

Inference:
  1. Autoregress text tokens from <BOT> until <EOT> is generated (optional CoT)
  2. After <EOT>, embed noise(traj_token) directly (bidirectional actions)
  3. Iteratively denoise actions via Flow Matching Euler ODE (or its SDE variant)
"""
from __future__ import annotations

import contextlib
import json
import math
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

from .diffusion import FlowMatching


# ---------------------------------------------------------------------------
# Grad-safe DynamicCache for the RL denoiser re-scoring
# ---------------------------------------------------------------------------
class _GradSafeDynamicCache(DynamicCache):
    """``DynamicCache`` whose linear-attention state write-backs REASSIGN instead
    of ``copy_`` in place.

    The stock ``LinearAttentionLayer.update_{conv,recurrent}_state`` copies the
    new state INTO the same tensor a fused FLA (GatedDeltaNet) kernel saved as its
    ``initial_state`` for backward ("copy instead of assigning, to preserve the
    static address for cuda graphs"). That is fine for the ``@torch.no_grad()``
    inference denoise, but the Transfusion GRPO recipe re-scores the denoise steps
    WITH gradients on a *cloned* prefix cache; the in-place copy then mutates a
    tensor that is part of the autograd graph ("one of the variables needed for
    gradient computation has been modified by an inplace operation ...
    CopyBackwards ... version 1"). Reassigning stores the (discarded, single-step)
    final state in a fresh tensor and leaves the saved initial state at version 0,
    so gradients flow correctly through the frozen prefix state.
    """

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        layer = self.layers[layer_idx]
        layer.conv_states = conv_states
        layer.is_conv_states_initialized = True
        layer.has_previous_state = True
        return conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        layer = self.layers[layer_idx]
        layer.recurrent_states = recurrent_states
        layer.is_recurrent_states_initialized = True
        return recurrent_states


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class TransfusionOutput:
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    diffusion_loss: Optional[torch.Tensor] = None
    per_dim_loss: Optional[Dict[str, float]] = None  # {"x": ..., "y": ..., "heading": ...}
    noise_pred: Optional[torch.Tensor] = None
    cot_ids: Optional[torch.Tensor] = None
    debug_info: Optional[Dict[str, Any]] = None  # first sample debug info


# ---------------------------------------------------------------------------
# Timestep embedding  (pi0 style)
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int,
    min_period: float = 4e-3, max_period: float = 4.0,
) -> torch.Tensor:
    """
    Sine-cosine positional embedding for continuous scalar positions (pi0 style).

    Following openpi/pi0_pytorch.py: uses log-spaced periods between
    (min_period, max_period) to build frequency bands tuned for t ->[0, 1].

"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).float()


class TimeEncoder(nn.Module):
    """Sine-cosine timestep -> MLP -> time conditioning signal [B, out_dim].

    Produces the diffusion-time AdaRMSNorm conditioning vector.
    This *is* the full ``adarms_cond``.

    The submodule is named ``time_mlp`` so its parameters
    (``time_embed.time_mlp.{0,2}.*``) match the time-only baseline checkpoint.
    """

    def __init__(self, sincos_dim: int, out_dim: int):
        super().__init__()
        self.sincos_dim = sincos_dim  # must be even (sin/cos pairs)
        self.out_dim = out_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(sincos_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, t_norm: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            t_norm, self.sincos_dim, min_period=4e-3, max_period=4.0,
        ).to(dtype)
        with torch.amp.autocast('cuda', dtype=dtype):
            return self.time_mlp(time_emb)  # [B, out_dim]


# ---------------------------------------------------------------------------
# ActionEncoder  (recogdrive-style 3-layer MLP with time fusion)
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    """
    3-layer MLP action encoder with OPTIONAL sinusoidal time fusion.

    When ``use_time`` is True (default) the diffusion timestep is fused at the
    input level for richer, time-aware action representations:
      1. fc1: action_dim → hidden_size
      2. Concatenate with sinusoidal time embedding
      3. fc2: hidden_size*2 → hidden_size (with SiLU)
      4. fc3: hidden_size → hidden_size

    When ``use_time`` is False the timestep is NOT injected here (``fc2`` takes
    ``hidden_size`` instead of ``hidden_size*2``) and the encoder becomes a
    plain time-agnostic action MLP. This path is INDEPENDENT of the AdaRMSNorm
    time-modulation path (see docs/time_conditioning.md).
    """

    def __init__(self, action_dim: int, hidden_size: int, use_time: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_time = use_time
        self.fc1 = nn.Linear(action_dim, hidden_size)
        fc2_in = hidden_size * 2 if use_time else hidden_size
        self.fc2 = nn.Linear(fc2_in, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x_t: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, T, action_dim] noised actions
            t_norm: [B] normalized timestep in [0, 1] (ignored when use_time=False)
        Returns:
            [B, T, hidden_size]
        """
        B, T, _ = x_t.shape
        action_emb = self.fc1(x_t.to(self.fc1.weight.dtype))  # [B, T, H]

        if self.use_time:
            time_emb = create_sinusoidal_pos_embedding(
                t_norm, self.hidden_size, min_period=4e-3, max_period=4.0,
            )  # [B, H]
            time_emb = time_emb.to(dtype=action_emb.dtype)
            time_emb = time_emb[:, None, :].expand(-1, T, -1)  # [B, T, H]
            x = torch.cat([action_emb, time_emb], dim=-1)  # [B, T, 2H]
        else:
            x = action_emb  # [B, T, H] — time-agnostic (no diffusion time here)

        x = self.act(self.fc2(x))  # [B, T, H]
        x = self.fc3(x)  # [B, T, H]
        return x


# ---------------------------------------------------------------------------
# ActionDecoder  (backbone hidden states -> action space)
# ---------------------------------------------------------------------------

class ActionDecoder(nn.Sequential):
    """MLP decoder mapping backbone hidden states -> action space.

    ``LayerNorm -> Linear -> SiLU -> LayerNorm -> Linear(action_dim)``.

    Subclasses ``nn.Sequential`` so the parameter names stay
    ``action_decoder.0..4`` (checkpoint-compatible with the time-only baseline).
    The decoder does NOT use time information (time is handled by the
    ActionEncoder input fusion + AdaRMSNorm modulation). The input LayerNorm
    makes it robust to the distribution shift between clean training features
    and iteratively-denoised inference features; the output layer is near-zero
    init (DiT best practice) so initial predictions start neutral.
    """

    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, action_dim),
        )
        nn.init.normal_(self[-1].weight, std=1e-4)
        nn.init.zeros_(self[-1].bias)


# ---------------------------------------------------------------------------
# short_prediction compact CoT tokens
# Keep IN SYNC with the data side:
#   NAVSIM/data_process/4_3_v2_short_prediction.py
# Per object:
#   <|sp_bbox|><|sp_x_{x1}|><|sp_y_{y1}|><|sp_x_{x2}|><|sp_y_{y2}|>[<|sp_fp_{fp}|>]<|/sp_bbox|>
#   (fp token omitted when the object has no future position; objects concatenated)
# No object at all -> <|object_none|>.
# With WIDTH x HEIGHT = 2048 x 512 there are 2048 x-tokens + 512 y-tokens
# + 3 markers (object_none, bbox open/close) + 10 fp = 2573 tokens.
# Token lengths dropped further: test avg 14.22, train avg 14.83 (min 1 for object_none).
# ---------------------------------------------------------------------------
SP_DEFAULT_WIDTH = 2048
SP_DEFAULT_HEIGHT = 512
SP_NONE_TOKEN = "<|object_none|>"   # emitted when the prediction list is empty
SP_BBOX_OPEN = "<|sp_bbox|>"        # opens one object
SP_BBOX_CLOSE = "<|/sp_bbox|>"      # closes one object
SP_X_FMT = "<|sp_x_{}|>"
SP_Y_FMT = "<|sp_y_{}|>"
SP_FP_FMT = "<|sp_fp_{}|>"
# Exactly 10 future actions: behind + {near,middle,far} x {front,left,right}.
SP_FPOS_ORDER = [
    "behind",
    "near_front", "near_left", "near_right",
    "middle_front", "middle_left", "middle_right",
    "far_front", "far_left", "far_right",
]


def build_short_prediction_tokens(width: int = SP_DEFAULT_WIDTH,
                                  height: int = SP_DEFAULT_HEIGHT) -> List[str]:
    """Full ordered list of short_prediction special tokens for a resolution."""
    tokens = [SP_NONE_TOKEN, SP_BBOX_OPEN, SP_BBOX_CLOSE]
    tokens += [SP_X_FMT.format(i) for i in range(width)]
    tokens += [SP_Y_FMT.format(i) for i in range(height)]
    tokens += [SP_FP_FMT.format(s) for s in SP_FPOS_ORDER]
    return tokens


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class TransfusionQwen3p5(nn.Module):
    """
    Transfusion = shared transformer (Qwen3.5) processing both discrete
    text tokens and noised continuous action tokens in a SINGLE sequence.

    Standalone version for Qwen3.5 backbone.

    Training:
        Noise ground-truth actions ->embed into sequence ->forward through
        transformer ->predict noise at action positions ->MSE loss.

    Inference:
        Start with pure-noise actions ->iteratively denoise through the
        transformer ->output clean trajectory.
    """

    BOT_TOKEN = "<|bot|>"
    EOT_TOKEN = "<|eot|>"

    # RL / SFT switch for the KV-cache clone used by the denoiser (``_clone_cache``).
    # DEFAULT False = SFT / inference behaviour (stock in-place DynamicCache, unchanged).
    # The RL recipe sets this True on its model so the WITH-GRAD denoiser re-scoring
    # gets the grad-safe (reassign) cache. Never touched by supervised training.
    rl_grad_safe_cache: bool = False

    # ---- Action normalization statistics  (computed from NAVSIM training set) ----
    # Following openpi/pi0: quantile-based normalization using q01/q99.
    # Actions are mapped to [-1, 1] via:  (x - q01) / (q99 - q01 + eps) * 2 - 1
    # This is more robust to outliers than mean/std normalization.
    #
    # When action_norm is enabled, actions are normalised during training and
    # denormalised at inference.  The statistics can be overridden via config.
    DEFAULT_ACTION_STATS = {
        # Per-dimension (x, y, heading) q0.1/q99.9 from NAVSIM training set
        "q01": [-0.02, -12.03, -1.30],
        "q99": [50.02, 16.03, 1.31],
    }

    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen3.5-2B",
        traj_token: int = 12,
        action_dim: int = 2,
        num_inference_steps: int = 50,
        timestep_shift: float = 1.0,
        lambda_diffusion: float = 5.0,
        lambda_lm: float = 1.0,
        cfg_scale: float = 3.0,
        freeze_vision_encoder: bool = True,
        freeze_llm: bool = False,
        max_cot_tokens: int = 256,
        action_norm: bool = True,
        action_stats: Optional[Dict] = None,
        dim_weights: Optional[List[float]] = None,
        action_repr: str = "absolute",
        action_offset: Optional[Dict] = None,
        # ----- fixed design options (the only supported values) -----
        ada_variant: str = "dit",          # AdaRMSNorm scale+shift (DiT style)
        ada_layer_mode: str = "full",      # AdaRMSNorm on every layer
        ada_use_weight: bool = True,       # multiply the pretrained RMSNorm weight
        action_time_fusion: bool = True,   # fuse t into the ActionEncoder input
        diffusion_type: str = "flow_matching",
        prediction_type: str = "x0",
        # ----- backbone weight-source switch (RL warm-start fast path) -----
        # False (default): stock behaviour — load the HF backbone weights via
        # ``Qwen3_5AdaForConditionalGeneration.from_pretrained`` (reads the whole
        # checkpoint file, random-inits, then copies the 618 matching params).
        # True: build the backbone from its CONFIG ONLY under ``no_init_weights``
        # and zero-init — valid ONLY when a full Transfusion checkpoint load
        # (``_load_state_dict_into``) is guaranteed to follow, since that load
        # supplies every backbone weight (a partial/legacy ckpt would leave
        # zeros — the recipe logs missing/unexpected and warns in that case).
        skip_backbone_weight_load: bool = False,
        action_loss_weight: float = 5.0,
        recon_loss_weight: float = 1.0,
        goal_point_weight: float = 0.0,
        clip_sample_range: Optional[float] = None,
        special_FFN=False,
        special_QKVO: Optional[dict] = None,
        # ----- short_prediction CoT (compact <|sp_...|> tokens) -----
        # ``cot_components`` is the yaml cot.cot_components list. When it equals
        # ["short_prediction"], the compact prediction tokens are registered on
        # the tokenizer (see _add_special_tokens). ``short_prediction_vocab`` is
        # the path to short_prediction_vocab.json written by
        # 4_3_v2_short_prediction.py; if missing, the tokens are reconstructed
        # deterministically from defaults.
        cot_components: Optional[List[str]] = None,
        short_prediction_vocab: Optional[str] = None,
        # ----- action-first -----
        # When True, switch the CoT/action ordering to the action-first scheme: the
        # action (generation) tokens come FIRST and the CoT (understanding)
        # tokens come AFTER, so the CoT cross-entropy loss back-propagates
        # THROUGH the noised action representations — i.e. "understanding
        # supervises generation". When False (default)
        # the model keeps the original self-CoT order (CoT -> action).
        use_action_first: bool = False,
        # ----- action-first eval-time CoT switch -----
        # Only effective when ``use_action_first`` is True. Controls what the action-first
        # inference path produces at TEST time:
        #   True  (default) -> generate the trajectory AND the post-hoc CoT
        #                      (action + CoT).
        #   False           -> generate the trajectory ONLY and skip the slow
        #                      autoregressive CoT decode (action only).
        action_first_generate_cot: bool = True,
    ):
        super().__init__()
        # ----- fixed design options: reject anything but the supported value -----
        if ada_variant != "dit":
            raise ValueError(f"ada_variant must be 'dit', got {ada_variant!r}")
        if not ada_use_weight:
            raise ValueError("ada_use_weight must be True")
        if not action_time_fusion:
            raise ValueError("action_time_fusion must be True")
        if diffusion_type != "flow_matching":
            raise ValueError(f"diffusion_type must be 'flow_matching', got {diffusion_type!r}")
        if prediction_type != "x0":
            raise ValueError(f"prediction_type must be 'x0', got {prediction_type!r}")
        self.traj_token = traj_token
        self.action_dim = action_dim
        self.lambda_diffusion = lambda_diffusion
        self.lambda_lm = lambda_lm
        self.cfg_scale = cfg_scale
        self.max_cot_tokens = max_cot_tokens
        # ----- short_prediction CoT setup -----
        _cot_components = [str(c).strip().lower() for c in (cot_components or []) if str(c).strip()]
        self.cot_components = _cot_components
        self.use_short_prediction = "short_prediction" in _cot_components
        if self.use_short_prediction and _cot_components != ["short_prediction"]:
            raise ValueError(
                "cot_components: 'short_prediction' must be the ONLY component when used; "
                f"got {cot_components!r}"
            )
        self.short_prediction_vocab = short_prediction_vocab
        # ----- action-first ordering -----
        # use_action_first=True  -> [ctx | actions(T) | <BOT> | cot | <EOT>]  (action -> CoT)
        # use_action_first=False -> [ctx | <BOT> | cot | <EOT> | actions(T)]  (CoT -> action)
        self.use_action_first = bool(use_action_first)
        # action-first eval-time CoT switch (only used on the action-first inference path):
        # True => action + post-hoc CoT (default); False => action only.
        self.action_first_generate_cot = bool(action_first_generate_cot)
        self.action_repr = action_repr  # "absolute" or "delta"
        # noise_space is fixed to "raw" (latent path removed); time fusion is
        # fixed to AdaRMSNorm ("adanorm").
        self.noise_space = "raw"
        # ----- special_FFN: bool OR dict {enabled, dim, is_bp_combine} -----
        #   enabled : use a separate per-layer action FFN (else original FFN only)
        #   dim     : action-FFN intermediate size. None => use the ORIGINAL
        #             intermediate_size AND copy the pretrained FFN weights. Any
        #             OTHER int (e.g. 4096) builds a fresh action FFN of that size
        #             (randomly initialised). A custom dim IS fully supported — it
        #             is only the pretrained weight-copy that needs the orig dim.
        #   is_bp_combine : True  => one COMBINED loss backprops through the whole
        #                      sequence in a single joint forward (text + action
        #                      tokens share one graph; action tokens -> action_mlp).
        #             False => SPLIT backprop matching the forward routing: the
        #                      TEXT tokens go through the text FFN and the text-or-
        #                      shared QKVO, so they are updated by the LM loss; the
        #                      ACTION tokens go through the action FFN and the
        #                      action-or-shared QKVO, so they are updated by the
        #                      diffusion (action) loss. Implemented as a two-pass
        #                      forward where the action stream attends to a
        #                      DETACHED prefix. A SHARED FFN/QKVO (no separate
        #                      action module) is updated by BOTH losses, since the
        #                      same params serve text AND action; isolation applies
        #                      only to a SEPARATE action FFN / projection.
        # The plain-bool form is only a shorthand and does NOT restrict the dim
        # (use the dict to choose one): ``special_FFN: true`` == {enabled: true,
        # dim: null, is_bp_combine: true}; ``special_FFN: false`` disables it.
        if isinstance(special_FFN, dict):
            self.special_FFN = bool(special_FFN.get("enabled", False))
            _ffn_dim = special_FFN.get("dim", None)
            self.special_FFN_dim = int(_ffn_dim) if _ffn_dim is not None else None
            self.is_bp_combine = bool(special_FFN.get("is_bp_combine", True))
        else:
            self.special_FFN = bool(special_FFN)
            self.special_FFN_dim = None
            self.is_bp_combine = True
        _qkvo = special_QKVO or {}
        self.special_KV = bool(_qkvo.get("KV", False))
        self.special_QO = bool(_qkvo.get("QO", False))
        # special_QKVO is effective only if at least one sub-flag is active
        self.special_QKV = self.special_KV or self.special_QO
        # NOTE: action-first is fully compatible with the MoT-style action routing
        # (special_FFN / special_QKVO). Under action-first the action tokens sit in the
        # middle of the sequence ([ctx | actions | BOT | cot | EOT]); the backbone
        # routes them via a ``(start, length)`` span instead of "last N tokens",
        # so both can be enabled together with no conflict.
        self.action_loss_weight = action_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.goal_point_weight = goal_point_weight

        # ----- backbone (Qwen3.5) -----
        # ----- backbone (Qwen3.5) — always AdaRMSNorm (adanorm) -----
        from .qwen3_5.modeling_qwen3_5_ada import Qwen3_5AdaForConditionalGeneration
        if skip_backbone_weight_load:
            # Build from configuration when a full Transfusion checkpoint will
            # supply the backbone weights immediately after construction.
            from transformers.initialization import no_init_weights
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

            backbone_config = Qwen3_5Config.from_pretrained(qwen_model_name)
            # Use SDPA for the language and vision models to avoid materializing
            # the quadratic attention score matrix.
            for _cfg in (
                backbone_config,
                getattr(backbone_config, "text_config", None),
                getattr(backbone_config, "vision_config", None),
            ):
                if _cfg is not None:
                    _cfg._attn_implementation = "sdpa"
            del _cfg
            with no_init_weights():
                self.qwen_model = Qwen3_5AdaForConditionalGeneration(
                    backbone_config,
                    cond_dim=backbone_config.text_config.hidden_size,
                    ada_layer_mode=ada_layer_mode,
                )
            # CRITICAL dtype decision (RL fast path): the config-only build above is
            # ALL fp32. For RL on 76GB cards we build the backbone ENTIRELY in
            # bfloat16. This cuts the
            # 2.3B-parameter LLM body's weights/gradients/activations in half
            # (~9.2GB -> ~4.6GB of weights, and matching cuts to the forward
            # activations and FLA backward scratch), which is what lets the actor
            # update fit on the 76GB cards -- at ppo_micro_batch_size_per_gpu=16 the
            # all-fp32 build peaked ~59.7GB and then OOM'd trying to Triton-autotune
            # the FLA l2norm_bwd (do_bench needs ~+12.6GB) during the backward.
            # bf16 is the standard RL/LLM training precision and matches what the
            # supervised (SFT) path loads via ``from_pretrained(torch_dtype=bf16)``.
            # The Transfusion ckpt later overwrites all values (it is bf16-stored),
            # so this only changes the working dtype, not the loaded weights.
            with torch.no_grad():
                self.qwen_model = self.qwen_model.to(torch.bfloat16)
            with torch.no_grad():
                for p in self.qwen_model.parameters():
                    p.zero_()
        else:
            # attn_implementation made EXPLICIT (was left to HF's auto-detection):
            # the `flash_attn` package isn't installed in this env, so auto-detect
            # already resolves to "sdpa" (both Qwen3_5AdaModel/-ForConditionalGeneration
            # declare _supports_sdpa=True, inherited from the stock Qwen3_5 classes;
            # PyTorch's own fused SDPA kernels are enabled). Pinning it here just
            # removes the dependency on that auto-detection (and the installed
            # flash-attn/transformers versions) silently changing the backend later.
            self.qwen_model = Qwen3_5AdaForConditionalGeneration.from_pretrained(
                qwen_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                cond_dim=None,
                ada_layer_mode=ada_layer_mode,
            )
        self.hidden_size = self._config.text_config.hidden_size
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        self._add_special_tokens()

        if freeze_vision_encoder:
            for p in self._visual.parameters():
                p.requires_grad = False
        if freeze_llm:
            for name, p in self.qwen_model.named_parameters():
                if "visual" not in name:
                    p.requires_grad = False

        # ----- action encoder / decoder -----
        # ActionEncoder: 3-layer MLP with input-level diffusion-time fusion
        # (recogdrive style): the denoising timestep t enters through this
        # token-input path; it is INDEPENDENT of the AdaRMSNorm time-modulation
        # path.
        self.action_encoder = ActionEncoder(action_dim, self.hidden_size)
        # ActionDecoder: hidden -> action space (no time info; time is handled
        # by the ActionEncoder input fusion + AdaRMSNorm modulation).
        self.action_decoder = ActionDecoder(self.hidden_size, action_dim)

        # ===================================================================
        # AdaRMSNorm time-modulation conditioning
        # -------------------------------------------------------------------
        #   adarms_cond(action) = time_embed(t)
        #   adarms_cond(prefix) = time_embed(0)
        # ===================================================================
        self.cond_dim = self.hidden_size

        # time — the entire ``adarms_cond``.
        self.time_embed = TimeEncoder(self.hidden_size, self.cond_dim)

        # ----- learnable position embedding for each action token -----
        self.traj_pos_embed = nn.Embedding(traj_token, self.hidden_size)

        # ----- Flow Matching schedule -----
        # Flow Matching Euler ODE requires timestep_shift=1.0 (uniform steps).
        # With shift != 1.0, the query t values are non-uniform but the step
        # size dt = -1/N is fixed, causing (x_t, t) mismatch and wrong velocity
        # predictions throughout inference.  Auto-correct to 1.0.
        import logging as _logging
        if timestep_shift != 1.0:
            _logging.warning(
                f"[TransfusionQwen3p5] Flow Matching requires "
                f"timestep_shift=1.0 for correct Euler integration "
                f"(got {timestep_shift}). Overriding to 1.0."
            )
            timestep_shift = 1.0
        fm_clip = clip_sample_range if clip_sample_range is not None else 0.0
        self.diffusion = FlowMatching(
            num_inference_steps=num_inference_steps,
            timestep_shift=timestep_shift,
            clip_sample_range=fm_clip,
        )

        # ----- action normalization -----
        self.action_norm = action_norm
        if action_norm:
            stats = action_stats or self.DEFAULT_ACTION_STATS
            # Per-timestep normalization: each step has its own q01/q99
            # This ensures all 8 timesteps use the full [-1, 1] range.
            # Without this, step-0 x (range ~[0, 7]) is squished into [-1, -0.72]
            # while step-7 x (range ~[0, 55]) spans the full range.
            # Broadcast global stats to [T, D]
            q01 = torch.tensor(stats["q01"][:action_dim], dtype=torch.float32)
            q99 = torch.tensor(stats["q99"][:action_dim], dtype=torch.float32)
            q01 = q01.unsqueeze(0).expand(traj_token, -1).clone()  # [T, D]
            q99 = q99.unsqueeze(0).expand(traj_token, -1).clone()  # [T, D]
            self.register_buffer("action_q01", q01)  # [T, D]
            self.register_buffer("action_q99", q99)  # [T, D]

        # ----- per-dimension loss weights -----
        if dim_weights is not None:
            w = torch.tensor(dim_weights[:action_dim], dtype=torch.float32)
            self.register_buffer("dim_weights", w)  # [action_dim]
        else:
            self.dim_weights = None

        # ----- constant post-denorm trajectory offset (default OFF) -----
        # Added to the FINAL denormalized, absolute ego trajectory that
        # generate_trajectory() returns, so every downstream consumer (the
        # training-time val metrics and test_*.py) sees the shifted waypoints.
        # The training LOSS is deliberately NOT offset -- otherwise the model
        # would simply learn to cancel it out.
        # Config: trajectory.action_offset = {enabled: bool, offset: [lat, lon]}
        # with lat = LATERAL, right positive and lon = LONGITUDINAL, forward
        # positive (the nuScenes eval/collision report's convention). The action
        # space is (x = forward, y = LEFT, heading), hence x += lon, y -= lat.
        # Kept as a plain attribute (NOT a registered buffer) so the state_dict
        # is unchanged and existing checkpoints still load.
        self.action_offset = None
        if action_offset and action_offset.get("enabled"):
            off = action_offset.get("offset") or [0.0, 0.0]
            if len(off) != 2:
                raise ValueError(
                    f"trajectory.action_offset.offset must be [lateral, longitudinal], "
                    f"got {off!r}"
                )
            lat, lon = float(off[0]), float(off[1])
            vec = torch.zeros(action_dim, dtype=torch.float32)
            vec[0] = lon              # x: forward
            if action_dim > 1:
                vec[1] = -lat         # y: left  (offset is right-positive)
            self.action_offset = vec  # [action_dim]
            print(f"[action_offset] ENABLED: lateral(right+)={lat:+.4f} m, "
                  f"longitudinal(forward+)={lon:+.4f} m "
                  f"-> added to the denormalized (x, y, heading) as {vec.tolist()}")

        # ----- zero-initialization (DiT best practice) -----
        # Following recogdrive/LightningDiT: zero-init the output projection
        # so initial noise predictions are zero. This means the model starts
        # from a neutral state and gradually learns to predict noise/velocity,
        # leading to much more stable early training and faster convergence.
        self._zero_init_diffusion_outputs()

        # ----- special_FFN: per-layer action FFN -----
        # Default dim (None): deep-copy the original FFN so the action FFN starts
        # from the pretrained weights. A custom dim builds a fresh Qwen3_5MLP with
        # the requested intermediate size (weights cannot be inherited, so it is
        # randomly initialized).
        if self.special_FFN:
            orig_dim = int(self._config.text_config.intermediate_size)
            tgt_dim = orig_dim if self.special_FFN_dim is None else int(self.special_FFN_dim)
            for layer in self._layers:
                if tgt_dim == orig_dim:
                    layer.action_mlp = copy.deepcopy(layer.mlp)
                else:
                    ref = next(layer.mlp.parameters())
                    layer.action_mlp = type(layer.mlp)(
                        self._config.text_config, tgt_dim,
                    ).to(dtype=ref.dtype, device=ref.device)

        # ----- special_QKV: MoT-style separate projections -----
        # sub-flags: special_QO → action Q/O projections, special_KV → action K/V projections.
        # Both False → original model. Both True → full separate QKVO.
        if self.special_QKV:
            for layer in self._layers:
                if not hasattr(layer, 'self_attn'):
                    continue
                attn = layer.self_attn
                if self.special_QO:
                    attn.action_q_proj = copy.deepcopy(attn.q_proj)
                    attn.action_o_proj = copy.deepcopy(attn.o_proj)
                if self.special_KV:
                    attn.action_k_proj = copy.deepcopy(attn.k_proj)
                    attn.action_v_proj = copy.deepcopy(attn.v_proj)

    # ------------------------------------------------------------------
    # Model-agnostic accessors  (overridden in model_qwen3_5.py)
    # ------------------------------------------------------------------

    @property
    def _visual(self):
        """Vision encoder module (Qwen3.5 path)."""
        return self.qwen_model.model.visual

    @property
    def _text_model(self):
        """Text backbone (Qwen3.5 path: language_model)."""
        return self.qwen_model.model.language_model

    @property
    def _layers(self):
        """Decoder layers list (Qwen3.5 path)."""
        return self.qwen_model.model.language_model.layers

    @property
    def _embed_tokens(self):
        """Token embedding layer (Qwen3.5 path)."""
        return self.qwen_model.model.language_model.embed_tokens

    @property
    def _lm_head(self):
        """LM head for CoT logits (Qwen3.5 path)."""
        return self.qwen_model.lm_head

    @property
    def _config(self):
        """Model config (Qwen3.5)."""
        return self.qwen_model.config

    @property
    def config(self):
        """Public HF-style config alias (Qwen3.5 backbone's config).

        Needed because this model is a plain ``nn.Module`` (not a
        ``PreTrainedModel``): verl's FSDP checkpoint manager
        (``fsdp_checkpoint_manager.py::save_checkpoint``) unwraps the FSDP
        module and reads ``unwrap_model.config`` to dump ``config.json`` next
        to the sharded checkpoint. Without this, saving ANY checkpoint
        (including step 0) raises
        ``AttributeError: 'TransfusionQwen3p5' object has no attribute 'config'``.
        """
        return self._config

    def can_generate(self) -> bool:
        """No HF-style ``.generate()`` support (see ``generate_trajectory`` for
        this model's own diffusion trajectory sampling). Checkpoint saving
        (verl's ``fsdp_checkpoint_manager.py``) calls this to decide whether to
        also dump a ``generation_config.json``; returning False just skips
        that (irrelevant for a diffusion planner, not a partial/degraded save).
        """
        return False

    @property
    def _mrope_dims(self):
        """Number of RoPE dimensions (3 for Qwen3.5 external API)."""
        return 3

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _zero_init_diffusion_outputs(self):
        """Zero-initialize diffusion-related output layers (DiT best practice)."""
        # Small-random init the last layer of ActionEncoder (fc3) so initial
        # action embeddings are near-zero (neutral start) while gradients still
        # flow back to fc1/fc2. The AdaRMSNorm identity-start is handled by the
        # zero-init ``dense`` projection inside the backbone's AdaRMSNorm layers
        # (so the time signal is non-zero but the modulation starts as identity).
        nn.init.normal_(self.action_encoder.fc3.weight, std=1e-4)
        if self.action_encoder.fc3.bias is not None:
            nn.init.zeros_(self.action_encoder.fc3.bias)

    def _time_cond(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Time-only AdaRMSNorm signal ``time_embed(t)`` [B, cond_dim]."""
        t_norm = t.float()
        return self.time_embed(t_norm, dtype)

    def _build_action_cond(
        self,
        t: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Action AdaRMSNorm conditioning = ``time_embed(t)``."""
        return self._time_cond(t, dtype)

    def _add_special_tokens(self):
        tok = self.processor.tokenizer
        special_tokens = [self.BOT_TOKEN, self.EOT_TOKEN]
        num_added = tok.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        # short_prediction: register the compact <|sp_...|> tokens as ATOMIC
        # added tokens. We use AddedToken(normalized=False, special=False):
        #   - normalized=False + atomic added-token matching => each <|sp_...|>
        #     is guaranteed to encode to exactly ONE id.
        #   - special=False => they SURVIVE ``decode(skip_special_tokens=True)``
        #     so eval/visualised CoT stays human-readable.
        if self.use_short_prediction:
            from transformers import AddedToken
            sp_tokens = [
                AddedToken(t, normalized=False, special=False)
                for t in self._short_prediction_token_list()
            ]
            num_added += tok.add_tokens(sp_tokens, special_tokens=False)
        if num_added > 0:
            self.qwen_model.resize_token_embeddings(len(tok))
        self.bot_token_id = tok.convert_tokens_to_ids(self.BOT_TOKEN)
        self.eot_token_id = tok.convert_tokens_to_ids(self.EOT_TOKEN)

    def _short_prediction_token_list(self) -> List[str]:
        """Return the ordered list of short_prediction special tokens.

        Prefers the vocab JSON written by 4_3_v2_short_prediction.py (single
        source of truth); falls back to a deterministic reconstruction from
        defaults if the file is unavailable.
        """
        path = self.short_prediction_vocab
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                toks = meta.get("all_special_tokens")
                if toks:
                    return list(toks)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[short_prediction] failed to read vocab {path!r}: {exc}; "
                      f"falling back to default construction.")
        return build_short_prediction_tokens(SP_DEFAULT_WIDTH, SP_DEFAULT_HEIGHT)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to [-1, 1] using per-timestep q01/q99 quantiles.

        Args:
            actions: [B, T, D] or [T, D]
        """
        if not self.action_norm:
            return actions
        q01 = self.action_q01.to(actions.device, actions.dtype)  # [T, D]
        q99 = self.action_q99.to(actions.device, actions.dtype)  # [T, D]
        return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions from [-1, 1] back to original scale (per-timestep).

        Args:
            actions: [B, T, D] or [T, D]
        """
        if not self.action_norm:
            return actions
        q01 = self.action_q01.to(actions.device, actions.dtype)  # [T, D]
        q99 = self.action_q99.to(actions.device, actions.dtype)  # [T, D]
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def abs_to_delta(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert absolute trajectory to per-step deltas (openpi style).

        Input:  [B, T, D]  absolute waypoints  [(x0,y0,h0), (x1,y1,h1), ...]
        Output: [B, T, D]  per-step increments  [(x0,y0,h0), (x1-x0,y1-y0,h1-h0), ...]

        Since ego starts at origin (0,0,0), delta_0 = abs_0.
        For i>=1: delta_i = abs_i - abs_{i-1}.

        This makes all steps have similar magnitude ranges, dramatically
        simplifying the flow matching target distribution.
        """
        # delta[:, 0, :] = actions[:, 0, :]  (first step stays the same)
        # delta[:, i, :] = actions[:, i, :] - actions[:, i-1, :]  for i >= 1
        delta = actions.clone()
        delta[:, 1:, :] = actions[:, 1:, :] - actions[:, :-1, :]
        return delta

    def apply_action_offset(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Add the configured constant offset to a final ABSOLUTE trajectory.

        Must be called AFTER ``denormalize_actions`` (and after ``delta_to_abs``
        when action_repr="delta"), i.e. on absolute ego-frame metres, so the
        whole trajectory is rigidly translated instead of the offset
        accumulating across waypoints.  No-op when disabled.

        Args:
            trajectory: [..., T, D]
        """
        if self.action_offset is None:
            return trajectory
        off = self.action_offset.to(trajectory.device, trajectory.dtype)
        return trajectory + off

    def delta_to_abs(self, delta: torch.Tensor) -> torch.Tensor:
        """Convert per-step deltas back to absolute trajectory (cumulative sum).

        Input:  [B, T, D]  per-step increments
        Output: [B, T, D]  absolute waypoints

        Inverse of abs_to_delta: abs = cumsum(delta, dim=1).
        """
        return torch.cumsum(delta, dim=1)

    # ------------------------------------------------------------------
    # Embedding construction (Qwen3.5 API)
    # ------------------------------------------------------------------

    def _get_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ):
        """Build merged text+vision embeddings using Qwen3.5 API."""
        embed_tokens = self._embed_tokens
        visual = self._visual
        cfg = self._config

        inputs_embeds = embed_tokens(input_ids)

        # Frozen vision tower (RL default: TRANSFUSION_FREEZE_VIT / SFT's
        # freeze_vision_encoder): run the visual forward under ``no_grad`` — no
        # parameter in ``self._visual`` receives a gradient, and nothing upstream
        # of it is trainable, so the autograd graph through the ViT is pure
        # waste (activation memory + bookkeeping). The visual outputs become
        # autograd constants; ``masked_scatter`` below still keeps the graph for
        # the (trainable) text-embedding path. When the ViT is NOT frozen this
        # is a no-op and gradients flow exactly as before.
        _visual_frozen = not any(p.requires_grad for p in visual.parameters())
        _vis_ctx = torch.no_grad() if _visual_frozen else contextlib.nullcontext()

        with _vis_ctx:
            if pixel_values is not None:
                # NOTE: was ``pixel_values.type(visual.dtype)`` — under FSDP (actor
                # engine) ``visual.dtype`` (``next(p.dtype for p in self.parameters()
                # if p.is_floating_point())``) can raise ``StopIteration`` if this
                # rank's local view of ``visual``'s parameters is empty for a given
                # sharding. The whole model uses one uniform compute dtype, so the
                # already-computed text embedding dtype is an equivalent, robust
                # substitute that never depends on a submodule's own parameter shard.
                pixel_values = pixel_values.type(inputs_embeds.dtype)
                image_output = visual(pixel_values, grid_thw=image_grid_thw)
                if hasattr(image_output, 'pooler_output'):
                    image_embeds = image_output.pooler_output
                    if isinstance(image_embeds, (list, tuple)):
                        image_embeds = torch.cat(image_embeds, dim=0)
                else:
                    image_embeds = image_output
                mask = (input_ids == cfg.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(inputs_embeds.dtype)
                video_output = visual(pixel_values_videos, grid_thw=video_grid_thw)
                if hasattr(video_output, 'pooler_output'):
                    video_embeds = video_output.pooler_output
                    if isinstance(video_embeds, (list, tuple)):
                        video_embeds = torch.cat(video_embeds, dim=0)
                else:
                    video_embeds = video_output
                mask = (input_ids == cfg.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(mask, video_embeds)

        return inputs_embeds

    # ------------------------------------------------------------------
    # Position IDs  (M-RoPE aware)
    # ------------------------------------------------------------------

    def _get_extended_position_ids(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        extra_len: int,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute Qwen3.5 M-RoPE position IDs for text+vision, then
        append sequential positions for action tokens.

        Qwen3.5 requires ``mm_token_type_ids`` for ``get_rope_index``.
        """
        mm_token_type_ids = self._build_mm_token_type_ids(input_ids)

        position_ids, rope_deltas = self.qwen_model.model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        device = position_ids.device

        if position_ids.dim() == 3:
            last = position_ids[:, :, -1:] + 1
            offsets = torch.arange(extra_len, device=device).view(1, 1, -1)
            position_ids = torch.cat([position_ids, last + offsets], dim=-1)
        else:
            last = position_ids[:, -1:] + 1
            offsets = torch.arange(extra_len, device=device).view(1, -1)
            position_ids = torch.cat([position_ids, last + offsets], dim=-1)

        return position_ids, rope_deltas

    def _build_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build mm_token_type_ids: 0=text, 1=image, 2=video."""
        cfg = self._config
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == cfg.image_token_id] = 1
        if hasattr(cfg, 'video_token_id') and cfg.video_token_id is not None:
            mm_token_type_ids[input_ids == cfg.video_token_id] = 2
        return mm_token_type_ids

    def _next_position_ids(self, position_ids: torch.Tensor, extra_len: int) -> torch.Tensor:
        """Return the next ``extra_len`` M-RoPE positions after ``position_ids``.

        This must continue from the last position returned by Qwen's
        ``get_rope_index`` rather than from the padded sequence length.  With
        LEFT padding, shorter samples have different real position cursors.
        """
        device = position_ids.device
        if position_ids.dim() == 3:
            last = position_ids[:, :, -1:] + 1
            offsets = torch.arange(extra_len, device=device).view(1, 1, -1)
            return last + offsets

        last = position_ids[:, -1:] + 1
        offsets = torch.arange(extra_len, device=device).view(1, -1)
        return last + offsets

    # ------------------------------------------------------------------
    # Transfusion attention mask  (Figure 4, pi0-style cumsum)
    # ------------------------------------------------------------------

    @staticmethod
    def make_attn_mask(
        input_mask: torch.Tensor,
        mask_ar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pi0-style attention mask via cumulative sum of ar_mask.

        Tokens can attend to valid input tokens whose cumulative mask_ar
        value is **smaller or equal** to theirs.

        Following openpi/pi0.py ``make_attn_mask``:
          - mask_ar = False (0): token shares block with previous (bidirectional)
          - mask_ar = True  (1): token starts a new causal block boundary

        Examples (mask_ar patterns):
          [1,1,1,1,1,1]:         pure causal attention
          [0,0,0,1,0,0]:         prefix bidirectional, suffix bidirectional,
                                 suffix can see prefix but not vice-versa
          [1,1,1,1,1, 1,0,0]:   causal prefix, then bidirectional block for
                                 last 3 tokens (Transfusion-style)

        Args:
            input_mask: [B, L] bool — True for valid tokens, False for padding.
            mask_ar:    [L] or [B, L] bool — block boundary indicators.

        Returns:
            [B, L, L] bool attention mask.
        """
        mask_ar = mask_ar.broadcast_to(input_mask.shape)
        cumsum = torch.cumsum(mask_ar.int(), dim=1)            # [B, L]
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]   # [B, L, L]
        valid_mask = input_mask[:, None, :] & input_mask[:, :, None]
        return attn_mask & valid_mask

    @staticmethod
    def bool_mask_to_4d(
        mask: torch.Tensor, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert [B, L, L] bool mask → [B, 1, L, L] float mask for HuggingFace."""
        min_val = torch.finfo(dtype).min
        return torch.where(mask[:, None, :, :], 0.0, min_val).to(dtype)

    def _build_transfusion_attn_mask(
        self,
        ctx_len: int,
        B: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        cot_len: int = 0,
        action_first: bool = False,
    ) -> torch.Tensor:
        """
        4-D attention mask implementing the Transfusion pattern with optional CoT.

        Two orderings are supported:

        * ``action_first=False`` (default, self-CoT) — sequence
          ``[ctx | <BOT> | cot | <EOT> | actions(T)]``:
            ctx / <BOT> / cot / <EOT>: ar = 1  (causal text attention)
            actions:                   [1, 0, …, 0]  (one bidirectional block,
                                       causal to everything before)

        * ``action_first=True`` (action-first: action -> CoT) — sequence
          ``[ctx | actions(T) | <BOT> | cot | <EOT>]``:
            ctx:                       ar = 1  (causal)
            actions:                   [1, 0, …, 0]  (bidirectional block, sees ctx)
            <BOT> / cot / <EOT>:       ar = 1  (causal; ATTEND to the noised
                                       actions so the CoT LM loss back-props into
                                       the action representations)
          Note actions do NOT attend to the CoT (cumsum ordering), so the
          generation pathway stays "action-first".
        """
        T = self.traj_token

        if action_first:
            # [ctx | actions(T) | BOT | cot | EOT]
            ar_mask = torch.ones(ctx_len + T + 1 + cot_len + 1, device=device, dtype=torch.bool)
            if T > 1:
                ar_mask[ctx_len + 1:ctx_len + T] = False  # actions[1:] share the block
            if attention_mask is not None:
                extra = T + 1 + cot_len + 1
                extra_mask = torch.ones(B, extra, device=device, dtype=torch.bool)
                input_mask = torch.cat([attention_mask.bool(), extra_mask], dim=1)
            else:
                input_mask = torch.ones(
                    B, ctx_len + T + 1 + cot_len + 1, device=device, dtype=torch.bool,
                )
            bool_mask = self.make_attn_mask(input_mask, ar_mask)
            return self.bool_mask_to_4d(bool_mask, dtype)

        causal_len = ctx_len + 1 + cot_len + 1  # ctx + BOT + cot + EOT
        bidir_len = T  # future action tokens in the bidirectional block

        ar_mask = torch.ones(causal_len + bidir_len, device=device, dtype=torch.bool)
        if bidir_len > 1:
            ar_mask[causal_len + 1:] = False  # actions[1:] share block with actions[0]

        # input_mask: incorporate padding from attention_mask for ctx tokens
        if attention_mask is not None:
            extra = 1 + cot_len + 1 + bidir_len
            extra_mask = torch.ones(B, extra, device=device, dtype=torch.bool)
            input_mask = torch.cat(
                [attention_mask.bool(), extra_mask], dim=1,
            )  # [B, total]
        else:
            input_mask = torch.ones(
                B, causal_len + bidir_len, device=device, dtype=torch.bool,
            )

        bool_mask = self.make_attn_mask(input_mask, ar_mask)  # [B, L, L]
        return self.bool_mask_to_4d(bool_mask, dtype)

    # ------------------------------------------------------------------
    # Encode noised actions  (Noising + Linear, cf. Figure 3 bottom half)
    # ------------------------------------------------------------------

    def _encode_noised_actions(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode noised actions into action token embeddings (input-level time fusion).

        Pipeline:
          1. ActionEncoder: fc1(action) ⊕ sin_time → SiLU(fc2) → fc3  [B, T, H]
          2. Learnable per-token position embedding added.

        The AdaRMSNorm modulation conditioning is built separately by
        ``_build_action_cond`` (time_embed(t)).

        Returns:
            action_embeds [B, T, H]
        """
        # 3-layer MLP with input-level time fusion (recogdrive style)
        h = self.action_encoder(x_t, t)                       # [B, T, H]

        # Learnable per-token position embedding
        pos = torch.arange(self.traj_token, device=x_t.device)
        h = h + self.traj_pos_embed(pos)                      # [B, T, H]

        # Cast back to backbone dtype (action_encoder is float32, backbone is bf16)
        backbone_dtype = self._embed_tokens.weight.dtype
        return h.to(backbone_dtype)

    def _get_special_embed(self, token_id: int, B: int, device: torch.device):
        ids = torch.full((B, 1), token_id, device=device, dtype=torch.long)
        return self._embed_tokens(ids)

    def _compute_adarms_cond(
        self, t: torch.Tensor, dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Time-only AdaRMSNorm conditioning [B, cond_dim] = ``time_embed(t)``.

        Used for the prefix (ctx + BOT + CoT + EOT) where only the diffusion
        time participates in the modulation. ``t`` is a [B] tensor.
        """
        return self._time_cond(t, dtype)

    # ------------------------------------------------------------------
    # Noise / time sampling  (pi0 style)
    # ------------------------------------------------------------------

    def sample_noise(self, shape, device: torch.device) -> torch.Tensor:
        """Sample noise ~ N(0, I).  Matches pi0_pytorch.py ``sample_noise``."""
        return torch.normal(
            mean=0.0, std=1.0, size=shape,
            dtype=torch.float32, device=device,
        )

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample diffusion time.

        Flow Matching: Beta(1.5, 1.0) * 0.999 + 0.001  (pi0 style, t ->[0.001, 1])
        """
        return self.diffusion.sample_time(batch_size, device)

    # ------------------------------------------------------------------
    # Forward  (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        cot_ids: Optional[torch.Tensor] = None,
        cot_labels: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        rl_score_fn: Optional[Any] = None,
        **kwargs,
    ) -> TransfusionOutput:
        """
        Transfusion forward pass with optional Chain-of-Thought (CoT).

        Sequence layout:
            [text+vision | <BOT> | cot_tokens | <EOT> | noised_actions(T)]

        If cot_ids is None, the CoT section is empty:
            [text+vision | <BOT> | <EOT> | noised_actions(T)]

        Loss = λ_lm * L_CoT + λ_diff * L_diff
        (Prompt = input context, no LM loss; only CoT has LM loss.)

        Args:
            input_ids:  [B, L]  text + image/video placeholder tokens (prompt, no loss)
            actions:    [B, traj_token, action_dim]  GT trajectory
            cot_ids:    [B, C]  CoT token IDs (optional) -> Do not add special tokens; use CoT text tokens only.  If None, no CoT is used.
            cot_labels: [B, C]  CoT LM targets (optional; -100 = ignore)
            rl_score_fn: optional FSDP-safe scoring hook. When given, the forward
                immediately returns ``rl_score_fn(self)`` — running the caller's
                function while this module's parameters are gathered by the
                (FSDP-wrapped) root forward. Submodule access (denoiser steps /
                ``cot_logprob``) outside the root forward would otherwise see
                *sharded* FSDP-1 params; this hook lets the Transfusion GRPO recipe
                (``recipe/transfusion``) compute the differentiable closed-form
                action x0 predictions + per-token CoT log-probs inside one forward.
        """
        if rl_score_fn is not None:
            return rl_score_fn(self)

        B = input_ids.shape[0]
        device = input_ids.device

        # ---- 1. multimodal context embeddings ----
        ctx_embeds = self._get_multimodal_embeddings(
            input_ids, pixel_values, image_grid_thw,
            pixel_values_videos, video_grid_thw,
        )
        L_ctx = ctx_embeds.shape[1]

        # ---- 2. CoT embeddings ----
        bot = self._get_special_embed(self.bot_token_id, B, device)
        eot = self._get_special_embed(self.eot_token_id, B, device)
        embed_tokens = self._embed_tokens

        if cot_ids is not None:
            cot_embeds = embed_tokens(cot_ids)  # [B, C, H]
            cot_len = cot_ids.shape[1]
        else:
            cot_embeds = None
            cot_len = 0

        # ---- 3. sample timestep & build noised actions (raw space) ----
        actions_f = actions.float()
        original_actions = actions_f.clone()
        if self.action_repr == "delta":
            actions_f = self.abs_to_delta(actions_f)
        actions_f = self.normalize_actions(actions_f)  # normalize to [-1, 1] via q01/q99
        t = self.sample_time(B, device)
        noise = self.sample_noise(actions_f.shape, device)
        x_t, noise = self.diffusion.q_sample(actions_f, t, noise=noise)
        act_embeds = self._encode_noised_actions(x_t.to(ctx_embeds.dtype), t)  # [B, T, H]

        # ---- 5. Build sequence (layout depends on the action-first ordering) ----
        # self-CoT (default): [ctx | <BOT> | cot | <EOT> | actions(T)]  (CoT -> action)
        # action-first  (use_action_first=True): [ctx | actions(T) | <BOT> | cot | <EOT>]  (action -> CoT)
        # Layout indices into the assembled sequence (used for hidden slicing):
        #   act_start     : first action token
        #   bot_pos       : <BOT> position (CoT LM loss reads from here)
        T = self.traj_token
        cot_parts = [bot]
        if cot_embeds is not None:
            cot_parts.append(cot_embeds)
        cot_parts.append(eot)
        cot_block = torch.cat(cot_parts, dim=1)  # [B, 1+cot_len+1, H]  (BOT|cot|EOT)

        if self.use_action_first:
            # action -> CoT : the CoT attends to the noised actions so its LM
            # loss back-props into the action (generation) representations.
            full_embeds = torch.cat([ctx_embeds, act_embeds, cot_block], dim=1)
            act_start = L_ctx
            bot_pos = L_ctx + T
            P = L_ctx  # prefix-before-actions (no separate prefix pass under action-first)
        else:
            # CoT -> action : the original self-CoT layout.
            full_embeds = torch.cat([ctx_embeds, cot_block, act_embeds], dim=1)
            act_start = L_ctx + 1 + cot_len + 1
            bot_pos = L_ctx
            P = act_start  # ctx + BOT + cot + EOT
        # Prefix embeddings for the non-action-first split-backprop path.
        prefix_embeds = full_embeds[:, :P, :]

        full_mask = self._build_transfusion_attn_mask(
            L_ctx, B, attention_mask, device, full_embeds.dtype,
            cot_len=cot_len, action_first=self.use_action_first,
        )

        # 2D padding mask for linear-attention (GatedDeltaNet) layers, which
        # cannot consume the 4D Transfusion mask. Prefix may be left-padded.
        linear_pad = None
        if attention_mask is not None:
            extra_valid = full_embeds.shape[1] - L_ctx
            linear_pad = torch.cat([
                attention_mask.to(full_embeds.dtype),
                torch.ones(B, extra_valid, device=device, dtype=full_embeds.dtype),
            ], dim=1)

        prefix_extra = 1 + cot_len + 1 + self.traj_token  # same total for both orderings
        full_pos, _ = self._get_extended_position_ids(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
            prefix_extra, second_per_grid_ts=second_per_grid_ts,
        )

        # ---- Per-position AdaRMSNorm conditioning ----
        # ctx + BOT/cot/EOT : time(0) ; actions : time(t).
        t_prefix = torch.zeros(B, device=device, dtype=torch.float32)
        prefix_cond = self._compute_adarms_cond(t_prefix, ctx_embeds.dtype)
        action_cond = self._build_action_cond(t, ctx_embeds.dtype)
        if self.use_action_first:
            text_block_len = 1 + cot_len + 1  # BOT + cot + EOT
            full_adarms_cond = torch.cat([
                prefix_cond.unsqueeze(1).expand(-1, L_ctx, -1),            # ctx @ t=0
                action_cond.unsqueeze(1).expand(-1, T, -1),               # actions @ t
                prefix_cond.unsqueeze(1).expand(-1, text_block_len, -1),  # BOT|cot|EOT @ t=0
            ], dim=1)
        else:
            full_adarms_cond = torch.cat([
                prefix_cond.unsqueeze(1).expand(-1, P, -1),               # ctx|BOT|cot|EOT @ t=0
                action_cond.unsqueeze(1).expand(-1, self.traj_token, -1), # actions @ t
            ], dim=1)

        # is_bp_combine=False (option A): split the backprop so it matches the
        # forward routing. The TEXT tokens flow through the text FFN and the
        # text-or-shared QKVO, so those params are updated by the LM loss; the
        # ACTION tokens flow through the action FFN and the action-or-shared
        # QKVO, so those params are updated by the diffusion (action) loss.
        #   (1) prefix-only pass (grad) -> prefix_hidden (LM loss). The prefix is
        #       causal and never attends to actions, so this matches the joint
        #       prefix exactly while training every prefix param (text FFN and
        #       text Q/K/V/O).
        #   (2) full pass with ``detach_prefix_len=P`` (grad) -> act_hidden
        #       (diffusion loss). The backbone detaches the prefix at every layer,
        #       so the action loss reaches the action FFN / Q/K/V/O projections but
        #       NOT the text-only params. ``action_FFN`` / ``action_QKV`` route the
        #       trailing action tokens through the action modules. A SHARED FFN /
        #       QKVO (no separate action module) is still updated by BOTH losses
        #       because the same params serve text AND action.
        # ``is_bp_combine`` lives under special_FFN but governs BOTH the action
        # FFN and the action QKV stream, so the split fires whenever either is
        # enabled. Forward values equal the joint pass (only the gradient routing
        # differs), so the split runs ONLY during training; eval/inference use the
        # joint pass below. No KV cache is reused, so this is safe with gradient
        # checkpointing and with the hybrid full-/linear-attention backbone.
        #
        # action-first exception: its whole purpose is to let the CoT (understanding) LM
        # loss flow INTO the action (generation) representations, so the action
        # tokens must NOT be detached from the LM-loss graph. The split-backprop
        # path deliberately isolates the two losses, which is antithetical to action-first;
        # under action-first we therefore always take the JOINT pass (the action span is
        # routed by ``(act_start, T)`` so MoT FFN/QKV still apply to the actions).
        if (self.special_FFN or self.special_QKV) and not self.is_bp_combine \
                and self.training and not self.use_action_first:
            # (1) prefix-only pass -> prefix_hidden (LM loss)
            prefix_kwargs = dict(
                inputs_embeds=prefix_embeds,
                attention_mask=full_mask[:, :, :P, :P],
                position_ids=full_pos[..., :P],
                use_cache=False,
                adarms_cond=(None if full_adarms_cond is None else full_adarms_cond[:, :P, :]),
            )
            if linear_pad is not None:
                prefix_kwargs["linear_attn_padding_mask"] = linear_pad[:, :P]
            prefix_out = self._text_model(**prefix_kwargs)
            prefix_hidden = prefix_out.last_hidden_state  # [B, P, H]

            # (2) full pass with per-layer prefix detach -> act_hidden (diff loss)
            action_kwargs = dict(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                position_ids=full_pos,
                use_cache=False,
                adarms_cond=full_adarms_cond,
                detach_prefix_len=P,
            )
            if linear_pad is not None:
                action_kwargs["linear_attn_padding_mask"] = linear_pad
            # action tokens are the trailing traj_token positions
            if self.special_FFN:
                action_kwargs["action_FFN"] = self.traj_token
            if self.special_QKV:
                action_kwargs["action_QKV"] = self.traj_token
            action_out = self._text_model(**action_kwargs)
            act_hidden = action_out.last_hidden_state[:, P:, :]  # [B, T, H]
            full_hidden = torch.cat([prefix_hidden, act_hidden], dim=1)  # [B, P+T, H]
        else:
            full_kwargs = dict(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                position_ids=full_pos,
                use_cache=False,
                adarms_cond=full_adarms_cond,
            )
            if linear_pad is not None:
                full_kwargs["linear_attn_padding_mask"] = linear_pad
            # special_FFN / special_QKV: route the action tokens through the
            # action-specific MLP / QKV projections. self-CoT puts the actions
            # last ("last N" = traj_token); action-first puts them in the middle, so pass
            # the explicit ``(act_start, T)`` span instead.
            action_route = (act_start, T) if self.use_action_first else self.traj_token
            if self.special_FFN:
                full_kwargs["action_FFN"] = action_route
            if self.special_QKV:
                full_kwargs["action_QKV"] = action_route

            backbone_out = self._text_model(**full_kwargs)
            full_hidden = backbone_out.last_hidden_state  # [B, P+T, H]

        # Slice the action and CoT hidden states using the layout indices so
        # both the self-CoT ([ctx|BOT|cot|EOT|actions]) and action-first
        # ([ctx|actions|BOT|cot|EOT]) orderings are handled uniformly.
        prefix_hidden = full_hidden[:, :P, :]  # [B, P, H]
        act_hidden = full_hidden[:, act_start:act_start + T, :]  # [B, T, H]

        # ---- 10. diffusion loss (raw action space) ----
        # action_decoder maps hidden → action space (no time info)
        # NOTE: was ``next(self.action_decoder.parameters()).dtype`` -- under
        # FSDP (actor engine) that can raise ``StopIteration`` if this rank's
        # local view of action_decoder's parameters is empty for a given
        # sharding (same hazard already fixed for ``visual.dtype`` in
        # ``_get_multimodal_embeddings``). The whole model uses one uniform
        # compute dtype, so the already-computed hidden-state dtype is an
        # equivalent, robust substitute (this ``decoder_dtype`` is reused below
        # for the encoder-decoder reconstruction loss too).
        decoder_dtype = self._action_decoder_dtype()
        noise_pred = self.action_decoder(act_hidden.to(decoder_dtype)).float()
        # Target (x0 prediction): the decoder output IS the (normalized) action prediction
        target = actions_f
        sq_err = (noise_pred - target) ** 2  # [B, T, D]

        # Per-dimension MSE for monitoring
        dim_names = ["x", "y", "heading"]
        per_dim_mse = sq_err.mean(dim=(0, 1))  # [D]
        per_dim_loss = {
            dim_names[i] if i < len(dim_names) else f"dim{i}": per_dim_mse[i].item()
            for i in range(per_dim_mse.shape[0])
        }

        ## Compute errors between denormalized actions ["x", "y", "heading"] and ground truth
        pred_x0_norm = torch.clamp(noise_pred, -1.0, 1.0)
        pred_actions = self.denormalize_actions(pred_x0_norm)
        gt_denorm_actions = self.denormalize_actions(actions_f)
        action_errors = pred_actions - original_actions  # [B, T, D]
        # Compute the mean absolute error for each dimension
        abs_errors = torch.abs(action_errors)
        mean_abs_error = abs_errors.mean(dim=(0, 1))  # [D]
        for i in range(mean_abs_error.shape[0]):
            dim_name = dim_names[i] if i < len(dim_names) else f"dim{i}"
            per_dim_loss[f"{dim_name}_mae"] = mean_abs_error[i].item()

        if self.dim_weights is not None:
            w = self.dim_weights.to(sq_err.device, sq_err.dtype)  # [D]
            diff_loss = (sq_err * w).mean()
        else:
            diff_loss = sq_err.mean()

        # ---- Action-space auxiliary loss (denormalized L1) ----
        # Directly penalize the gap between predicted x0 and GT in action space,
        # providing a stronger gradient signal for trajectory accuracy.
        action_space_loss = F.l1_loss(pred_x0_norm, actions_f)
        diff_loss = diff_loss + self.action_loss_weight * action_space_loss
        per_dim_loss["action_l1"] = action_space_loss.item()

        # ---- Goal point loss (last waypoint L2) ----
        # Supervise the endpoint (4s) with an additional L2 penalty.
        if self.goal_point_weight > 0:
            goal_pred = pred_actions[:, -1, :]     # [B, action_dim]
            goal_gt = original_actions[:, -1, :]   # [B, action_dim]
            goal_l2 = F.mse_loss(goal_pred, goal_gt)
            diff_loss = diff_loss + self.goal_point_weight * goal_l2
            per_dim_loss["goal_point_l2"] = goal_l2.item()
            # Per-dim error for the goal point
            goal_err = torch.abs(goal_pred - goal_gt).mean(dim=0)
            for i, dn in enumerate(["x", "y", "heading"][:goal_err.shape[0]]):
                per_dim_loss[f"goal_{dn}_mae"] = goal_err[i].item()

        # ---- Encoder-Decoder reconstruction loss (autoencoder round-trip) ----
        # Encode clean actions at t=0 through the action_encoder→action_decoder
        # round-trip.
        recon_raw = None
        recon_loss_raw = None
        if self.recon_loss_weight > 0:
            t_zero = torch.zeros(B, device=device)
            with torch.no_grad() if not self.training else torch.enable_grad():
                clean_embeds_raw = self._encode_noised_actions(
                    actions_f.to(ctx_embeds.dtype), t_zero
                )
            recon_raw = self.action_decoder(clean_embeds_raw.to(decoder_dtype)).float()
            recon_loss_raw = F.l1_loss(recon_raw, actions_f)
            diff_loss = diff_loss + self.recon_loss_weight * recon_loss_raw
            per_dim_loss["norm_recon_l1"] = recon_loss_raw.item()
            recon_denorm = self.denormalize_actions(recon_raw.clamp(-1, 1))
            gt_denorm_recon = self.denormalize_actions(actions_f.clamp(-1, 1))
            recon_loss_denorm = F.l1_loss(recon_denorm, gt_denorm_recon)
            per_dim_loss["denorm_recon_l1"] = recon_loss_denorm.item()
            recon_err = torch.abs(recon_denorm - gt_denorm_recon).mean(dim=(0, 1))
            for i, dn in enumerate(["x", "y", "heading"][:recon_err.shape[0]]):
                per_dim_loss[f"recon_{dn}_err"] = recon_err[i].item()

        # Cap loss to prevent gradient explosions (NaN protection)
        diff_loss = torch.clamp(diff_loss, max=20.0)

        # ---- 11. LM loss (CoT cross-entropy) ----
        # Read the [BOT|cot] hidden states from ``full_hidden`` at ``bot_pos`` so
        # the action-first ordering (CoT after the actions) and the self-CoT ordering (CoT
        # in the prefix) are both handled. Under action-first this loss back-propagates
        # THROUGH the noised action representations (understanding supervises
        # generation). Always predict at least BOT -> EOT to align with inference.
        if cot_labels is not None and cot_len > 0:
            # Full CoT LM loss: [<BOT>, cot_0, ..., cot_{C-1}] -> [cot_0, ..., cot_{C-1}, <EOT>]
            logits_cot = self._lm_head(full_hidden[:, bot_pos:bot_pos + 1 + cot_len, :])  # [B, 1+C, V]
            eot_ids = torch.full((B, 1), self.eot_token_id, device=device, dtype=torch.long)
            cot_targets = torch.cat([cot_labels, eot_ids], dim=1)  # [B, C+1]
            lm_loss = F.cross_entropy(
                logits_cot.view(-1, logits_cot.size(-1)),
                cot_targets.view(-1),
                ignore_index=-100,
            )
        else:
            # No CoT data: predict BOT -> EOT (single token) to stay aligned with inference
            logits_bot = self._lm_head(full_hidden[:, bot_pos:bot_pos + 1, :])  # [B, 1, V]
            eot_target = torch.full((B,), self.eot_token_id, device=device, dtype=torch.long)
            lm_loss = F.cross_entropy(logits_bot.squeeze(1), eot_target)

        # ---- combined loss ----
        loss = self.lambda_diffusion * diff_loss
        if self.lambda_diffusion == 0:
            # lambda_diffusion=0 gates the WHOLE diff_loss group (the action /
            # recon aux losses are folded into diff_loss above), which would
            # silently zero the action head's training signal. Re-add the aux
            # losses directly (decoupled from lambda_diffusion) so the action
            # head still trains when the master diffusion gate is off.
            # NOTE: goal_point terms remain gated by lambda_diffusion.
            loss = self.action_loss_weight * action_space_loss
            if self.recon_loss_weight > 0 and recon_loss_raw is not None:
                loss = loss + self.recon_loss_weight * recon_loss_raw
        if lm_loss is not None:
            loss = loss + self.lambda_lm * lm_loss

        # ---- Debug info: first sample in batch ----
        debug_info = {}
        with torch.no_grad():
            debug_info["mode"] = self.noise_space
            debug_info["t"] = t[0].item()
            debug_info["gt_actions_norm"] = actions_f[0].cpu().numpy().tolist()
            debug_info["gt_actions_denorm"] = self.denormalize_actions(actions_f[0:1]).cpu().numpy()[0].tolist()
            debug_info["gt_actions"] = original_actions[0].cpu().numpy().tolist()
            debug_info["pred_actions_denorm"] = pred_actions[0].cpu().numpy().tolist()
            debug_info["denoised_actions"] = pred_actions.cpu().numpy().tolist()

            debug_info["noise_pred_0"] = noise_pred[0].cpu().numpy().tolist()
            debug_info["target_0"] = target[0].cpu().numpy().tolist()
            debug_info["pred_x0_norm"] = pred_x0_norm[0].cpu().numpy().tolist()
            debug_info["x_t_0"] = x_t[0].cpu().numpy().tolist()
            if recon_raw is not None:
                debug_info["recon_actions_denorm"] = self.denormalize_actions(
                    recon_raw[0:1].clamp(-1, 1)).cpu().numpy()[0].tolist()

            # Goal point info
            if "goal_point_l2" in per_dim_loss:
                debug_info["goal_point_l2"] = per_dim_loss["goal_point_l2"]
                debug_info["goal_pred"] = pred_actions[0, -1].cpu().numpy().tolist()
                debug_info["goal_gt"] = original_actions[0, -1].cpu().numpy().tolist()

        return TransfusionOutput(
            loss=loss,
            lm_loss=lm_loss,
            diffusion_loss=diff_loss,
            per_dim_loss=per_dim_loss,
            noise_pred=noise_pred,
            debug_info=debug_info,
        )

    # ------------------------------------------------------------------
    # RL: teacher-forced per-token CoT log-probability + entropy
    # ------------------------------------------------------------------
    def cot_logprob(
        self,
        input_ids: torch.Tensor,
        actions_normed: torch.Tensor,
        cot_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        ctx_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token CoT log-prob and entropy for a *given* CoT under a *given* action.

        Teacher-forced twin of :meth:`_generate_cot_after_action` (action-first layout
        ``[ctx | clean_actions | <BOT> | cot]``): it reproduces that method's
        exact conditioning (clean actions encoded at ``t=0``, uniform ``t=0``
        AdaRMSNorm, the same MoT action routing and bidirectional-action mask) so
        the returned log-prob is the density of the SAME autoregressive CoT
        policy the rollout sampled from — the token-probability signal used by the
        Transfusion GRPO recipe (``recipe/transfusion``).

        Args:
            input_ids:      ``[B, L]`` prompt tokens (text + image/video placeholders).
            actions_normed: ``[B, T, D]`` clean trajectory in NORMALISED ``[-1, 1]``
                space (the diffusion output ``x0``, exactly what
                ``_generate_cot_after_action`` conditions on).
            cot_ids:        ``[B, C]`` CoT token ids to score (right-padded is fine;
                score only the valid positions via a mask on the returned tensor).
            ctx_embeds:     optional precomputed multimodal context embeddings
                ``[B, L, H]`` (e.g. reused from the diffusion adapter's context) to
                skip a second vision-tower pass; recomputed from the inputs when
                ``None``.

        Returns:
            ``(token_logp, token_entropy)`` each ``[B, C]``: ``token_logp[:, k]`` is
            ``log pi(cot_ids[:, k] | ctx, actions, cot_{<k})`` and
            ``token_entropy[:, k]`` is the entropy of that next-token distribution.
            Gradients flow (call under ``torch.enable_grad()`` for the actor update;
            wrap in ``torch.no_grad()`` for the behaviour-policy old-log-prob pass).
        """
        B = input_ids.shape[0]
        device = input_ids.device
        T = self.traj_token
        C = int(cot_ids.shape[1])

        # ---- 1. multimodal context embeddings (reuse if supplied) ----
        if ctx_embeds is None:
            ctx_embeds = self._get_multimodal_embeddings(
                input_ids, pixel_values, image_grid_thw,
                pixel_values_videos, video_grid_thw,
            )
        dtype = ctx_embeds.dtype
        L_ctx = ctx_embeds.shape[1]

        # ---- 2. clean action + <BOT> + cot embeddings (action-first order) ----
        t_zero = torch.zeros(B, device=device, dtype=torch.float32)
        act_embeds = self._encode_noised_actions(actions_normed.to(dtype), t_zero)  # [B, T, H]
        bot = self._get_special_embed(self.bot_token_id, B, device)  # [B, 1, H]
        cot_embeds = self._embed_tokens(cot_ids)  # [B, C, H]
        seq_embeds = torch.cat([ctx_embeds, act_embeds, bot, cot_embeds], dim=1)  # [B, L_ctx+T+1+C, H]
        seq_len = L_ctx + T + 1 + C

        # ---- 3. attention mask: ctx causal, action bidirectional block, BOT|cot causal ----
        ar_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        if T > 1:
            ar_mask[L_ctx + 1:L_ctx + T] = False
        if attention_mask is not None:
            extra_mask = torch.ones(B, T + 1 + C, device=device, dtype=torch.bool)
            input_mask = torch.cat([attention_mask.bool(), extra_mask], dim=1)
        else:
            input_mask = torch.ones(B, seq_len, device=device, dtype=torch.bool)
        full_mask = self.bool_mask_to_4d(self.make_attn_mask(input_mask, ar_mask), dtype)

        # ---- 4. positions: ctx M-RoPE, then sequential for actions + <BOT> + cot ----
        prefix_pos, _ = self._get_extended_position_ids(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
            extra_len=0, second_per_grid_ts=second_per_grid_ts,
        )
        ext_pos = self._next_position_ids(prefix_pos, T + 1 + C)  # [.., B, T+1+C]
        full_pos = torch.cat([prefix_pos, ext_pos], dim=-1)

        # ---- 5. backbone forward (uniform t=0 AdaRMSNorm, MoT action routing) ----
        prefix_adarms_cond = self._compute_adarms_cond(t_zero, dtype=dtype)
        kwargs = dict(
            inputs_embeds=seq_embeds,
            attention_mask=full_mask,
            position_ids=full_pos,
            use_cache=False,
        )
        if prefix_adarms_cond is not None:
            kwargs["adarms_cond"] = prefix_adarms_cond.unsqueeze(1).expand(-1, seq_len, -1)
        if attention_mask is not None:
            kwargs["linear_attn_padding_mask"] = torch.cat([
                attention_mask.to(dtype),
                torch.ones(B, T + 1 + C, device=device, dtype=dtype),
            ], dim=1)
        if self.special_FFN:
            kwargs["action_FFN"] = (L_ctx, T)
        if self.special_QKV:
            kwargs["action_QKV"] = (L_ctx, T)
        hidden = self._text_model(**kwargs).last_hidden_state  # [B, seq_len, H]

        # ---- 6. per-token log-prob + entropy of the sampled CoT ----
        # Hidden at [<BOT>, cot_0, ..., cot_{C-1}] predicts [cot_0, ..., cot_{C-1}, <EOT>].
        # The first C predictions score the sampled tokens cot_0..cot_{C-1}.
        # The vocab projection runs in 64-token CHECKPOINTED chunks: the
        # [B, C, V] fp32 logits + log_softmax + exp chain is the single biggest
        # retained activation in the RL actor update (~15 GB at B=32, V≈152k).
        # Plain chunking only trims the transient peak — autograd still retains
        # every chunk's intermediates; wrapping each chunk in
        # ``torch.utils.checkpoint`` makes the backward RECOMPUTE them from the
        # tiny [B, 64, H] hidden slice instead, so the full-vocab tensors are
        # never kept (same maths, ~4x less retained memory).
        bot_pos = L_ctx + T
        h_cot = hidden[:, bot_pos:bot_pos + C, :]  # [B, C, H]
        cot_ids_l = cot_ids.long()

        def _vocab_score(h_chunk, ids_chunk):
            _lg = self._lm_head(h_chunk).float()  # [B, c, V]
            _lp = torch.log_softmax(_lg, dim=-1)
            _logp = _lp.gather(-1, ids_chunk.unsqueeze(-1)).squeeze(-1)
            _ent = -(_lp.exp() * _lp).sum(dim=-1)
            return _logp, _ent

        logp_chunks, ent_chunks = [], []
        _CHUNK = 64
        for _s in range(0, C, _CHUNK):
            _h = h_cot[:, _s:_s + _CHUNK, :]
            _ids = cot_ids_l[:, _s:_s + _CHUNK]
            if torch.is_grad_enabled():
                _logp, _ent = torch.utils.checkpoint.checkpoint(
                    _vocab_score, _h, _ids, use_reentrant=False,
                )
            else:
                _logp, _ent = _vocab_score(_h, _ids)
            logp_chunks.append(_logp)
            ent_chunks.append(_ent)
        token_logp = torch.cat(logp_chunks, dim=1)  # [B, C]
        token_entropy = torch.cat(ent_chunks, dim=1)  # [B, C]
        return token_logp, token_entropy

    # ------------------------------------------------------------------
    # Inference  (iterative denoising with KV-cache)
    # ------------------------------------------------------------------
    #
    # Optimisation strategy (following pi0):
    #   1. Forward the *prefix* [text+vision | <BOT> (| CoT) | <EOT>] ONCE
    #      with ``use_cache=True`` ->obtain ``past_key_values``.
    #   2. Each denoising step only forwards the *suffix*
    #      [noised_actions(T)] with ``past_key_values``
    #      and ``use_cache=False`` ->the prefix KV cache is never grown
    #      during denoising, only reused.  Actions attend bidirectionally.
    #   3. For CFG a second (unconditional) KV cache is built from
    #      zeroed-out prefix embeddings.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
        generate_cot: bool = False,
        cot_temperature: float = 0.9,
        cot_top_p: float = 0.95,
        cot_top_k: int = 100,
        return_normed: bool = False,
        fm_sde: bool = False,
        sde_noise_scale: float = 1.0,
        sde_kernel: str = "sit",
        sde_noise_level=None,
        sde_smooth_noise: Optional[dict] = None,
        step_recorder: Optional[List] = None,
        max_cot_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate trajectory (and optional CoT) with KV-cache optimisation.

        Returns:
            trajectory: [B, traj_token, action_dim]
            cot_ids:    [B, C] generated CoT token IDs, or None
            (if return_normed) normed_trajectory: [B, traj_token, action_dim] in [-1,1]

        Args (sampling controls):
            fm_sde:            flow matching only — use a stochastic SDE sampler
                               instead of the deterministic Euler ODE.
            sde_noise_scale:   stochasticity multiplier for the SiT SDE (0 -> ODE);
                               ignored when ``sde_kernel='flow_grpo'``.
            sde_kernel:        ``"sit"`` (SiT / stochastic-interpolants SDE) or
                               ``"flow_grpo"`` (the GRPO TRAINING rollout kernel).
            sde_noise_level:   flow_grpo kernel only — the g(t)=a*sqrt(t)
                               schedule's std-scale hyper-parameter (scalar or
                               per-dim ``[D]`` list; ``0`` -> deterministic ODE).
            sde_smooth_noise:  flow_grpo kernel only — optional kwargs dict for
                               ``smooth_noise_like`` (waypoint-correlated noise).
            step_recorder:   optional list; when provided, each denoising step's
                             inputs/outputs are appended for offline inspection.
            max_cot_tokens:  optionally cap the number of generated CoT CONTENT
                             tokens (default: ``self.max_cot_tokens``). Lets the
                             caller keep the VALIDATION CoT budget identical to
                             the RL training rollout (``response_length - 1``),
                             so val cannot blow up to the model's native budget.
        """
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # action-first (action -> CoT): denoise the trajectory FIRST conditioned on the
        # context, THEN optionally generate the CoT as a post-hoc explanation.
        if self.use_action_first:
            return self._generate_trajectory_action_first(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                cfg_scale=cfg_scale,
                generate_cot=generate_cot,
                cot_temperature=cot_temperature,
                cot_top_p=cot_top_p,
                cot_top_k=cot_top_k,
                return_normed=return_normed,
                fm_sde=fm_sde,
                sde_noise_scale=sde_noise_scale,
                sde_kernel=sde_kernel,
                sde_noise_level=sde_noise_level,
                sde_smooth_noise=sde_smooth_noise,
                step_recorder=step_recorder,
                max_cot_tokens=max_cot_tokens,
            )

        B = input_ids.shape[0]
        device = input_ids.device
        H = self._config.text_config.hidden_size

        # -------- 1. Prefix forward: [ctx | <BOT>] ->KV cache ----------
        ctx_embeds = self._get_multimodal_embeddings(
            input_ids, pixel_values, image_grid_thw,
            pixel_values_videos, video_grid_thw,
        )
        bot = self._get_special_embed(self.bot_token_id, B, device)
        prefix_embeds = torch.cat([ctx_embeds, bot], dim=1)  # [B, P, H]
        P = prefix_embeds.shape[1]

        prefix_mask = self._make_causal_mask(P, B, device, prefix_embeds.dtype)

        # Use M-RoPE position IDs for prefix (matching training)
        # extra_len=1 accounts for the <BOT> token appended to ctx.
        prefix_pos, _ = self._get_extended_position_ids(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
            extra_len=1,
            second_per_grid_ts=second_per_grid_ts,
        )
        next_position_ids = self._next_position_ids(prefix_pos, 1)

        # Prefix is processed WITH adarms_cond at FIXED t=0 (matching training
        # phase 1). Using a fixed t means the prefix KV cache is computed
        # once and reused across every denoising step without recomputation,
        # and CoT generation also receives consistent AdaRMSNorm modulation.
        # Each denoising step recomputes its own adarms_cond from the
        # current (action) t.
        t_prefix = torch.zeros(B, device=device, dtype=torch.float32)
        prefix_adarms_cond = self._compute_adarms_cond(
            t_prefix, dtype=prefix_embeds.dtype,
        )
        prefix_kwargs = dict(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            position_ids=prefix_pos,
            use_cache=True,
        )
        if prefix_adarms_cond is not None:
            prefix_kwargs["adarms_cond"] = prefix_adarms_cond
        # 2D padding mask for linear-attention layers so left-padding in the
        # prompt does not corrupt the GatedDeltaNet recurrence (Task 3).
        if attention_mask is not None:
            prefix_kwargs["linear_attn_padding_mask"] = torch.cat([
                attention_mask.to(prefix_embeds.dtype),
                torch.ones(B, P - attention_mask.shape[1], device=device, dtype=prefix_embeds.dtype),
            ], dim=1)
        out = self._text_model(**prefix_kwargs)
        past_kv = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:, :]  # for first CoT prediction
        cur_pos = P

        # -------- 2. Autoregressive CoT, then <EOT>, then optional state ---
        cot_ids_out = None
        if generate_cot:
            cot_ids_raw, cot_lengths = self._generate_cot_tokens(
                last_hidden, past_kv, cur_pos, B, device,
                next_position_ids=next_position_ids,
                temperature=cot_temperature,
                top_p=cot_top_p,
                top_k=cot_top_k,
                adarms_cond=prefix_adarms_cond,
                max_cot_tokens=max_cot_tokens,
            )
            cot_ids_out = cot_ids_raw

            # Rebuild full sequence: [extra_pads | prefix | cot_real | EOT]
            # so that pads are ALL on the left, matching training layout.
            max_cot = int(cot_lengths.max().item()) if cot_ids_raw is not None else 0
            embed_tokens = self._embed_tokens
            eot_emb = self._get_special_embed(self.eot_token_id, 1, device)  # [1, 1, H]

            # Per-sample original pad count (in input_ids space = ctx_embeds space)
            if attention_mask is not None:
                orig_pad_counts = (input_ids.shape[1] - attention_mask.sum(dim=1)).long()  # [B]
            else:
                orig_pad_counts = torch.zeros(B, dtype=torch.long, device=device)

            # BOT is already in prefix_embeds (last position). Per-sample real prefix:
            #   real_prefix_len_i = P - orig_pad_counts[i]
            # Full real content per sample = real_prefix_len_i + cot_len_i + 1 (EOT)
            real_lens = (P - orig_pad_counts) + cot_lengths + 1  # [B]
            max_full_len = int(real_lens.max().item())

            # Build full embeddings with left-padding
            pad_id = self.processor.tokenizer.pad_token_id or 0
            pad_emb_single = embed_tokens(
                torch.tensor([pad_id], device=device)
            )  # [1, H]
            full_embeds = pad_emb_single.unsqueeze(0).expand(B, max_full_len, -1).clone()

            for i in range(B):
                real_prefix_len_i = P - int(orig_pad_counts[i].item())
                cot_len_i = int(cot_lengths[i].item())
                real_total_i = real_prefix_len_i + cot_len_i + 1
                offset = max_full_len - real_total_i

                # Copy real prefix (without original left-pads)
                full_embeds[i, offset:offset + real_prefix_len_i] = \
                    prefix_embeds[i, int(orig_pad_counts[i].item()):]

                # Copy real COT tokens
                if cot_len_i > 0:
                    cot_embs_i = embed_tokens(
                        cot_ids_raw[i, :cot_len_i].unsqueeze(0)
                    ).squeeze(0)
                    full_embeds[i, offset + real_prefix_len_i:offset + real_prefix_len_i + cot_len_i] = cot_embs_i

                # EOT is always the last token
                full_embeds[i, -1] = eot_emb.squeeze()

            # Build causal attention mask [B, 1, max_full_len, max_full_len]
            neg_inf = torch.finfo(full_embeds.dtype).min
            full_mask = torch.triu(
                torch.full((max_full_len, max_full_len), neg_inf, device=device, dtype=full_embeds.dtype),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).clone()

            # Block attention to left-pad positions
            for i in range(B):
                real_total_i = int(real_lens[i].item())
                pad_count_i = max_full_len - real_total_i
                if pad_count_i > 0:
                    full_mask[i, :, :, :pad_count_i] = neg_inf
                    full_mask[i, :, :pad_count_i, :] = neg_inf

            # Position IDs: build per-sample to preserve original M-RoPE
            # spatial positions for visual tokens while correctly placing
            # COT + EOT positions right after each sample's real prefix.
            full_pos = torch.zeros(
                3, B, max_full_len, device=device, dtype=torch.long,
            )
            for i in range(B):
                pad_i = int(orig_pad_counts[i].item())
                cot_len_i = int(cot_lengths[i].item())
                real_total_i = int(real_lens[i].item())
                new_pad_i = max_full_len - real_total_i

                # Original M-RoPE positions for the real prefix content
                real_prefix_pos_i = prefix_pos[:, i, pad_i:]  # [3, P - pad_i]

                # Sequential extension for COT + EOT after last prefix pos
                last_pos_i = real_prefix_pos_i[:, -1:]  # [3, 1]
                ext_offsets = torch.arange(
                    1, cot_len_i + 2, device=device,
                ).view(1, -1)  # [1, cot_len_i + 1]
                ext_pos_i = last_pos_i + ext_offsets  # [3, cot_len_i + 1]

                real_pos_i = torch.cat(
                    [real_prefix_pos_i, ext_pos_i], dim=-1,
                )  # [3, real_total_i]
                full_pos[:, i, new_pad_i:] = real_pos_i

            # Forward full sequence from scratch
            full_kwargs = dict(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                position_ids=full_pos,
                past_key_values=None,
                use_cache=True,
            )
            if prefix_adarms_cond is not None:
                full_kwargs["adarms_cond"] = prefix_adarms_cond
            out = self._text_model(**full_kwargs)
            past_kv = out.past_key_values
            cur_pos = max_full_len
            next_position_ids = full_pos[:, :, -1:] + 1  # per-sample next pos after EOT
        else:
            # No CoT: forward <EOT> directly to mark boundary.
            eot = self._get_special_embed(self.eot_token_id, B, device)
            out = self._forward_single_token(
                eot, past_kv, cur_pos, B, device,
                position_ids=next_position_ids,
                adarms_cond=prefix_adarms_cond,
            )
            past_kv = out.past_key_values
            cur_pos += 1
            next_position_ids = next_position_ids + 1

        prefix_total_len = cur_pos  # total cached length
        cached_extra_len = prefix_total_len - P
        if generate_cot:
            # COT case: position IDs already built per-sample in full_pos.
            full_prefix_pos = full_pos
        elif cached_extra_len > 0:
            cached_extra_pos = self._next_position_ids(prefix_pos, cached_extra_len)
            full_prefix_pos = torch.cat([prefix_pos, cached_extra_pos], dim=-1)
        else:
            full_prefix_pos = prefix_pos
        action_position_ids = next_position_ids + torch.arange(
            self.traj_token, device=device,
        ).view(1, 1, -1)

        # -------- 3. Unconditional KV cache for CFG --------------------
        uncond_past_kv = None
        if cfg_scale > 1.0:
            uncond_past_kv = self._build_uncond_cache(
                prefix_embeds.shape, cot_ids_out, cur_pos,
                B, H, device, prefix_embeds.dtype,
                adarms_cond=prefix_adarms_cond,
                position_ids=full_prefix_pos,
            )

        # -------- 4. Iterative denoising with frozen prefix cache ------
        # Pass prefix KV cache to denoising steps so actions attend to the
        # full prefix context (matching single-pass training where actions
        # see the prefix through the Transfusion mask).
        denoise_uncond_kv = uncond_past_kv if past_kv is not None else None
        denoise_prefix_len = prefix_total_len if past_kv is not None else 0

        x_t = self.sample_noise((B, self.traj_token, self.action_dim), device)
        denoised = self._denoise_loop_fm(
            x_t, past_kv, denoise_uncond_kv, denoise_prefix_len,
            B, device, cfg_scale, prefix_embeds.dtype,
            action_position_ids,
            fm_sde=fm_sde, sde_noise_scale=sde_noise_scale,
            sde_kernel=sde_kernel, sde_noise_level=sde_noise_level,
            sde_smooth_noise=sde_smooth_noise,
            step_recorder=step_recorder,
        )

        trajectory = denoised

        # Save normalized trajectory before denormalization
        normed_trajectory = trajectory.clone() if return_normed else None

        # Denormalize back to original action space
        trajectory = self.denormalize_actions(trajectory)
        # Convert from delta back to absolute if configured
        if self.action_repr == "delta":
            trajectory = self.delta_to_abs(trajectory)
        # Constant post-denorm offset (trajectory.action_offset; no-op if off)
        trajectory = self.apply_action_offset(trajectory)

        if return_normed:
            return (trajectory, cot_ids_out, normed_trajectory, denoised)
        return (trajectory, cot_ids_out)

    # ------------------------------------------------------------------
    # action-first inference  (action -> CoT)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_trajectory_action_first(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        second_per_grid_ts: Optional[torch.Tensor],
        cfg_scale: float,
        generate_cot: bool,
        cot_temperature: float,
        cot_top_p: float,
        cot_top_k: int,
        return_normed: bool,
        fm_sde: bool = False,
        sde_noise_scale: float = 1.0,
        sde_kernel: str = "sit",
        sde_noise_level=None,
        sde_smooth_noise: Optional[dict] = None,
        step_recorder: Optional[List] = None,
        max_cot_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """action-first inference: generate the trajectory FIRST, then the CoT.

        Mirrors the action-first training layout ``[ctx | actions | <BOT> | cot | <EOT>]``:

          1. Forward the context ``[ctx]`` once -> a frozen prefix KV cache.
          2. Denoise the trajectory conditioned ONLY on the context (the actions
             attend to ``ctx`` via the cache; the CoT is absent from the
             generation pathway, exactly as in training).
          3. (optional) Generate the CoT as a post-hoc explanation, conditioned
             on ``[ctx | clean_actions | <BOT>]`` -> autoregress cot -> ``<EOT>``.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        H = self._config.text_config.hidden_size

        # -------- 1. Context prefix forward [ctx] -> KV cache ----------
        ctx_embeds = self._get_multimodal_embeddings(
            input_ids, pixel_values, image_grid_thw,
            pixel_values_videos, video_grid_thw,
        )
        L_ctx = ctx_embeds.shape[1]
        P = L_ctx  # prefix = context only (no <BOT>/CoT before the actions)

        prefix_mask = self._make_causal_mask(P, B, device, ctx_embeds.dtype)
        # Context M-RoPE positions only (extra_len=0: nothing appended yet).
        prefix_pos, _ = self._get_extended_position_ids(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
            extra_len=0, second_per_grid_ts=second_per_grid_ts,
        )
        t_prefix = torch.zeros(B, device=device, dtype=torch.float32)
        prefix_adarms_cond = self._compute_adarms_cond(t_prefix, dtype=ctx_embeds.dtype)
        prefix_kwargs = dict(
            inputs_embeds=ctx_embeds,
            attention_mask=prefix_mask,
            position_ids=prefix_pos,
            use_cache=True,
        )
        if prefix_adarms_cond is not None:
            prefix_kwargs["adarms_cond"] = prefix_adarms_cond
        if attention_mask is not None:
            prefix_kwargs["linear_attn_padding_mask"] = attention_mask.to(ctx_embeds.dtype)
        out = self._text_model(**prefix_kwargs)
        past_kv = out.past_key_values

        # -------- 2. Action position IDs (immediately after ctx) -------
        action_position_ids = self._next_position_ids(prefix_pos, self.traj_token)  # [3, B, T]

        # -------- 3. Unconditional KV cache for CFG (zeroed ctx) --------
        uncond_past_kv = None
        if cfg_scale > 1.0:
            zeros = torch.zeros(B, P, H, device=device, dtype=ctx_embeds.dtype)
            uncond_kwargs = dict(
                inputs_embeds=zeros,
                attention_mask=self._make_causal_mask(P, B, device, ctx_embeds.dtype),
                position_ids=prefix_pos,
                use_cache=True,
            )
            if prefix_adarms_cond is not None:
                uncond_kwargs["adarms_cond"] = prefix_adarms_cond
            if attention_mask is not None:
                uncond_kwargs["linear_attn_padding_mask"] = attention_mask.to(ctx_embeds.dtype)
            uncond_past_kv = self._text_model(**uncond_kwargs).past_key_values

        # -------- 4. Iterative denoising (actions attend to ctx) -------
        denoise_uncond_kv = uncond_past_kv if past_kv is not None else None
        denoise_prefix_len = P if past_kv is not None else 0

        x_t = self.sample_noise((B, self.traj_token, self.action_dim), device)
        denoised = self._denoise_loop_fm(
            x_t, past_kv, denoise_uncond_kv, denoise_prefix_len,
            B, device, cfg_scale, ctx_embeds.dtype,
            action_position_ids,
            fm_sde=fm_sde, sde_noise_scale=sde_noise_scale,
            sde_kernel=sde_kernel, sde_noise_level=sde_noise_level,
            sde_smooth_noise=sde_smooth_noise,
            step_recorder=step_recorder,
        )

        normed_trajectory = denoised.clone() if return_normed else None

        # -------- 6. Optional post-hoc CoT (conditioned on the action) -
        # Gated by BOTH the per-call ``generate_cot`` flag and the model-level
        # ``action_first_generate_cot`` config switch. When ``action_first_generate_cot`` is False
        # we emit the trajectory ONLY (action-only eval) and skip the slow
        # autoregressive CoT decode, regardless of ``generate_cot``.
        cot_ids_out = None
        if generate_cot and self.action_first_generate_cot:
            cot_ids_out = self._generate_cot_after_action(
                ctx_embeds, denoised, prefix_pos, attention_mask,
                B, device, prefix_adarms_cond,
                temperature=cot_temperature, top_p=cot_top_p, top_k=cot_top_k,
                max_cot_tokens=max_cot_tokens,
            )

        # -------- 7. Denormalize and return ----------------------------
        trajectory = self.denormalize_actions(denoised)
        if self.action_repr == "delta":
            trajectory = self.delta_to_abs(trajectory)
        # Constant post-denorm offset (trajectory.action_offset; no-op if off)
        trajectory = self.apply_action_offset(trajectory)

        if return_normed:
            return (trajectory, cot_ids_out, normed_trajectory, denoised)
        return (trajectory, cot_ids_out)

    @torch.no_grad()
    def _generate_cot_after_action(
        self,
        ctx_embeds: torch.Tensor,
        denoised_actions: torch.Tensor,
        prefix_pos: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        B: int,
        device: torch.device,
        prefix_adarms_cond: Optional[torch.Tensor],
        temperature: float,
        top_p: float,
        top_k: int,
        record_logp: bool = False,
        max_cot_tokens: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """action-first post-hoc CoT: condition on ``[ctx | clean_actions | <BOT>]`` and
        autoregress the CoT that explains the generated trajectory.

        The clean (denoised) actions are encoded at ``t=0`` so the CoT reads a
        stable action representation. All non-action positions use the ``t=0``
        AdaRMSNorm conditioning, matching the prefix/text handling in training.

        ``record_logp=True`` returns ``(cot_ids, cot_logps, cot_entropies)`` —
        the sampler's own per-token raw-distribution log-probs / entropies (see
        ``_generate_cot_tokens``); default keeps returning just ``cot_ids``.
        """
        L_ctx = ctx_embeds.shape[1]
        T = self.traj_token
        dtype = ctx_embeds.dtype

        t_zero = torch.zeros(B, device=device, dtype=torch.float32)
        act_embeds = self._encode_noised_actions(denoised_actions.to(dtype), t_zero)  # [B, T, H]
        bot = self._get_special_embed(self.bot_token_id, B, device)
        seq_embeds = torch.cat([ctx_embeds, act_embeds, bot], dim=1)  # [B, L_ctx+T+1, H]
        seq_len = seq_embeds.shape[1]

        # ar_mask: ctx causal, actions one bidirectional block, <BOT> causal.
        ar_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        if T > 1:
            ar_mask[L_ctx + 1:L_ctx + T] = False
        if attention_mask is not None:
            extra_mask = torch.ones(B, T + 1, device=device, dtype=torch.bool)
            input_mask = torch.cat([attention_mask.bool(), extra_mask], dim=1)
        else:
            input_mask = torch.ones(B, seq_len, device=device, dtype=torch.bool)
        full_mask = self.bool_mask_to_4d(self.make_attn_mask(input_mask, ar_mask), dtype)

        # Positions: ctx M-RoPE, then sequential for actions + <BOT>.
        ext_pos = self._next_position_ids(prefix_pos, T + 1)  # [3, B, T+1]
        full_pos = torch.cat([prefix_pos, ext_pos], dim=-1)   # [3, B, seq_len]

        kwargs = dict(
            inputs_embeds=seq_embeds,
            attention_mask=full_mask,
            position_ids=full_pos,
            use_cache=True,
        )
        if prefix_adarms_cond is not None:
            # Uniform t=0 conditioning across ctx | actions | <BOT>.
            kwargs["adarms_cond"] = prefix_adarms_cond.unsqueeze(1).expand(-1, seq_len, -1)
        if attention_mask is not None:
            kwargs["linear_attn_padding_mask"] = torch.cat([
                attention_mask.to(dtype),
                torch.ones(B, T + 1, device=device, dtype=dtype),
            ], dim=1)
        # MoT routing: the actions sit at [L_ctx : L_ctx+T], route that span
        # through the action MLP / QKV so this matches the action-first training forward.
        if self.special_FFN:
            kwargs["action_FFN"] = (L_ctx, T)
        if self.special_QKV:
            kwargs["action_QKV"] = (L_ctx, T)
        out = self._text_model(**kwargs)
        past_kv = out.past_key_values
        last_hidden = out.last_hidden_state[:, -1:, :]  # hidden at <BOT>
        cur_pos = seq_len
        next_position_ids = full_pos[:, :, -1:] + 1

        # Additive per-sample block for the cached prompt positions so the
        # autoregressive decode below CANNOT attend to the prompt's left-pad
        # KV entries — exactly what the teacher-forced training forward
        # (``cot_logprob``) enforces via its 4D input mask. Without this the
        # decode-time token distributions (whose per-token log-probs become
        # the behaviour-policy ``old_log_probs``) drift from the update-side
        # recompute, and the resulting PPO importance ratio explodes on
        # sharpened policies (see ``_forward_single_token``).
        prefix_pad_block = None
        if attention_mask is not None:
            keep = attention_mask.to(device=device).bool()
            prefix_pad_block = torch.where(
                keep,
                torch.zeros((), device=device, dtype=dtype),
                torch.full((), torch.finfo(dtype).min, device=device, dtype=dtype),
            )  # [B, L_ctx]; 0 = attend, dtype-min = block

        if record_logp:
            cot_ids_raw, _, cot_logps, cot_ents = self._generate_cot_tokens(
                last_hidden, past_kv, cur_pos, B, device,
                next_position_ids=next_position_ids,
                temperature=temperature, top_p=top_p, top_k=top_k,
                adarms_cond=prefix_adarms_cond, record_logp=True,
                max_cot_tokens=max_cot_tokens,
                prefix_pad_block=prefix_pad_block,
            )
            return cot_ids_raw, cot_logps, cot_ents
        cot_ids_raw, _ = self._generate_cot_tokens(
            last_hidden, past_kv, cur_pos, B, device,
            next_position_ids=next_position_ids,
            temperature=temperature, top_p=top_p, top_k=top_k,
            adarms_cond=prefix_adarms_cond,
            max_cot_tokens=max_cot_tokens,
            prefix_pad_block=prefix_pad_block,
        )
        return cot_ids_raw

    # ---- low-level cache helpers (pi0-style cumsum) ------------------

    @staticmethod
    def _make_causal_mask(
        seq_len: int, B: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Standard [B, 1, L, L] causal attention mask (pi0-style).

        Equivalent to ar_mask = [1, 1, 1, …, 1] (every token is its own block).
        """
        ar_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        input_mask = torch.ones(B, seq_len, device=device, dtype=torch.bool)
        bool_mask = TransfusionQwen3p5.make_attn_mask(input_mask, ar_mask)
        return TransfusionQwen3p5.bool_mask_to_4d(bool_mask, dtype)

    def _make_denoise_suffix_mask(
        self,
        prefix_len: int,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
        suffix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Attention mask for the denoising suffix [history?(H) | actions(T)] with KV cache.

        Shape: [B, 1, suffix_len, prefix_len + suffix_len]

        Built using pi0-style ar_mask over [cached_prefix | suffix]:
          cached_prefix: ar_mask = [0, 0, …, 0]  (all visible, already computed)
          suffix:        ar_mask = [1, 0, 0, …, 0]  (one bidirectional block)

        Actions (and history tokens) attend to the full prefix (via KV cache)
        and to each other **bidirectionally**.
        """
        T = suffix_len if suffix_len is not None else self.traj_token
        total = prefix_len + T

        # Suffix queries attend the FULL cached prefix (cumsum block 0) plus the
        # whole bidirectional suffix block (block 1): with an all-ones input
        # mask every key is visible, so the additive 4D mask is ALL ZEROS —
        # identical to the previous [B, total, total] cumsum construction
        # sliced to the suffix rows, without materialising it (that cost real
        # time at RL batch sizes: [32, ~2.3k, ~2.3k] bool per denoise call).
        return torch.zeros(B, 1, T, total, device=device, dtype=dtype)

    def _forward_single_token(
        self, embeds, past_kv, cur_pos, B, device,
        position_ids: Optional[torch.Tensor] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        prefix_pad_block: Optional[torch.Tensor] = None,
    ):
        """Forward a single [B, 1, H] token using KV cache, growing the cache.

        ``prefix_pad_block`` — optional ``[B, L_prefix]`` ADDITIVE attention
        values for the cached prefix positions (0 = attend, dtype-min = block).
        The RL CoT decode MUST pass the prompt's left-padding block so the
        pad KV entries stay masked, EXACTLY like the teacher-forced training
        forward (``cot_logprob``) masks them via ``input_mask``. Without it
        every decode step attends to the left-pad tokens, so the per-token
        log-probs recorded here (the behaviour-policy ``old_log_probs``)
        drift from the update-side recompute and the PPO/GRPO importance
        ratio explodes (observed: cot_ratio up to 1e31 -> NaN grads ->
        "CUDA error: device-side assert" in the next rollout).
        """
        if position_ids is None:
            pos = torch.full((self._mrope_dims, B, 1), cur_pos, device=device, dtype=torch.long)  # [3, B, 1] for M-RoPE
        else:
            pos = position_ids.to(device=device, dtype=torch.long)
        total = cur_pos + 1
        # Single token attends to all cached + itself
        mask = torch.zeros(B, 1, 1, total, device=device, dtype=embeds.dtype)
        if prefix_pad_block is not None:
            # Block the cached prompt-pad positions (matching the training-side
            # full-sequence forward). Additive values: 0 = attend, min = block.
            mask[:, 0, 0, : prefix_pad_block.shape[-1]] = prefix_pad_block.to(embeds.dtype)
        kwargs = dict(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=past_kv,
            use_cache=True,
        )
        if adarms_cond is not None:
            kwargs["adarms_cond"] = adarms_cond
        return self._text_model(**kwargs)

    # ---- CoT with KV cache ------------------------------------------
    @torch.no_grad()
    def _generate_cot_tokens(
        self,
        last_hidden: torch.Tensor,
        past_kv,
        cur_pos: int,
        B: int,
        device: torch.device,
        next_position_ids: Optional[torch.Tensor] = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 100,
        adarms_cond: Optional[torch.Tensor] = None,
        record_logp: bool = False,
        max_cot_tokens: Optional[int] = None,
        prefix_pad_block: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Autoregress CoT tokens with per-sample EOT tracking.

        ``max_cot_tokens`` optionally caps the number of generated CONTENT
        tokens (default: ``self.max_cot_tokens``). In the RL path the caller
        passes the rollout response budget minus one so the forced final EOT
        column (see below) fits inside the response window kept by verl.

        The SFT-designated stop token is ``<|eot|>`` (the SFT LM loss predicts
        ``[cot_0, ..., cot_{C-1}, <EOT>]``): sampling stops when EOT is
        sampled. In the RL path (``record_logp=True``) the sampled EOT is KEPT
        in the returned ids so ``cot_logprob`` / the token GRPO loss score
        P(EOT | prefix) — without that, EOT is never optimised, its
        probability decays, and the model fills the whole response with junk
        after the prediction block (the observed CoT collapse).

        Only performs autoregressive generation — does NOT rebuild KV cache.
        The caller (generate_trajectory) rebuilds the full sequence with
        proper layout matching training.

        ``record_logp=True`` additionally records, for every generated position,
        the per-token log-prob and entropy of the RAW (pre-temperature /
        pre-top-k/top-p) next-token distribution — exactly the quantities the
        teacher-forced ``cot_logprob`` recompute would produce, captured for
        free from the sampler's own logits (RL behaviour-policy ``old_log_probs``).

        Returns (when ``record_logp``):
            cot_ids: [B, max_gen_steps] generated token IDs (left-aligned,
                     with pad tokens after each sample's EOT), or None.
            cot_lengths: [B] actual COT length per sample (excluding EOT).
            cot_logps: [B, max_gen_steps] per-token log-prob (raw distribution).
            cot_entropies: [B, max_gen_steps] per-token entropy (raw distribution).
        """
        lm_head = self._lm_head
        embed_tokens = self._embed_tokens
        pad_id = self.processor.tokenizer.pad_token_id or 0
        cap = int(max_cot_tokens) if max_cot_tokens else int(self.max_cot_tokens)

        generated_ids: list[torch.Tensor] = []
        rec_logps: list[torch.Tensor] = []
        rec_ents: list[torch.Tensor] = []
        hidden = last_hidden  # [B, 1, H]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cot_lengths = torch.zeros(B, dtype=torch.long, device=device)
        if next_position_ids is None:
            next_position_ids = torch.full((self._mrope_dims, B, 1), cur_pos, device=device, dtype=torch.long)

        for step in range(cap):
            raw_logits = lm_head(hidden)[:, 0, :]  # [B, V]
            logits = raw_logits / max(temperature, 1e-6)

            if top_k > 0:
                # FAST top-k + top-p: work entirely on the top-k set ([B, K]
                # instead of [B, V]). EXACTLY the same sampling distribution as
                # mask-to--inf + full-vocab sort + softmax: entries outside the
                # top-k have zero probability anyway, so nucleus/cumsum over the
                # sorted top-k values is identical — just ~V/K times cheaper
                # (no per-step [B, 151k] sort/softmax).
                k = min(top_k, logits.size(-1))
                top_vals, top_idx = torch.topk(logits, k, dim=-1)  # sorted desc
                top_probs = torch.softmax(top_vals.float(), dim=-1)
                if 0.0 < top_p < 1.0:
                    cumulative_probs = torch.cumsum(top_probs, dim=-1)
                    # Keep at least one token (same rule as the reference).
                    keep = (cumulative_probs - top_probs) < top_p
                    top_probs = top_probs * keep
                pick = torch.multinomial(top_probs, num_samples=1)  # [B, 1]
                next_id = top_idx.gather(-1, pick).squeeze(-1)  # [B]
            else:
                # Top-p (nucleus) filtering: keep smallest set with cumulative prob >= top_p
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens with cumulative prob above top_p (keep at least one)
                    remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[remove_mask] = float('-inf')
                    # Scatter back to original ordering
                    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

            # Record actual COT length when sample first generates EOT.
            newly_finished = (next_id == self.eot_token_id) & ~finished
            cot_lengths[newly_finished] = step
            finished = finished | (next_id == self.eot_token_id)

            # Pad samples that finished EARLIER (their EOT is already recorded).
            # In the RL path (record_logp=True) the just-sampled EOT is KEPT as
            # a real column so cot_logprob / the token GRPO loss can score
            # P(EOT | prefix) — this is what stops the observed collapse into
            # repeated ``</prediction>`` junk: the model can no longer shift
            # probability mass from EOT to filler tokens because EOT itself is
            # optimised with the same advantage. Inference (record_logp=False)
            # keeps the legacy behaviour (EOT dropped from the returned ids).
            next_id = next_id.clone()
            if record_logp:
                next_id[finished & ~newly_finished] = pad_id
            else:
                next_id[finished] = pad_id
            generated_ids.append(next_id)

            if record_logp:
                # Raw (temperature=1) distribution log-prob / entropy of the
                # token ACTUALLY appended at this position (pad for finished
                # samples — those positions are masked out downstream, exactly
                # like the teacher-forced recompute scores whatever sits there;
                # for newly-finished samples this is log P(EOT | prefix)).
                lp = torch.log_softmax(raw_logits.float(), dim=-1)
                rec_logps.append(lp.gather(-1, next_id.long().unsqueeze(-1)).squeeze(-1))
                rec_ents.append(-(lp.exp() * lp).sum(dim=-1))

            if finished.all():
                break

            next_emb = embed_tokens(next_id.unsqueeze(1))  # [B, 1, H]
            out = self._forward_single_token(
                next_emb, past_kv, cur_pos, B, device,
                position_ids=next_position_ids,
                adarms_cond=adarms_cond,
                prefix_pad_block=prefix_pad_block,
            )
            past_kv = out.past_key_values
            hidden = out.last_hidden_state
            cur_pos += 1
            next_position_ids = next_position_ids + 1

        # Samples that never finished: their length = number of generated tokens
        cot_lengths[~finished] = len(generated_ids)

        if record_logp and not finished.all() and len(generated_ids) > 0:
            # Force an EOT column for samples that hit the cap without ever
            # sampling EOT, so P(EOT | last generated token) still receives a
            # gradient in the RL update (otherwise the degenerate all-cap rows
            # never learn to stop). Already-finished samples get pad here (their
            # own EOT column is already inside the sequence).
            forced = torch.full((B,), self.eot_token_id, dtype=torch.long, device=device)
            forced[finished] = pad_id
            generated_ids.append(forced)
            # hidden is the last generated token's hidden state; its logits are
            # the distribution over the token AFTER that last content token.
            last_logits = lm_head(hidden)[:, 0, :]  # [B, V]
            lp = torch.log_softmax(last_logits.float(), dim=-1)
            rec_logps.append(lp.gather(-1, forced.long().unsqueeze(-1)).squeeze(-1))
            rec_ents.append(-(lp.exp() * lp).sum(dim=-1))

        if len(generated_ids) == 0:
            if record_logp:
                return None, cot_lengths, None, None
            return None, cot_lengths

        cot_ids = torch.stack(generated_ids, dim=1)  # [B, max_gen_steps]
        if record_logp:
            cot_logps = torch.stack(rec_logps, dim=1)      # [B, max_gen_steps]
            cot_ents = torch.stack(rec_ents, dim=1)        # [B, max_gen_steps]
            return cot_ids, cot_lengths, cot_logps, cot_ents
        return cot_ids, cot_lengths

    # ---- Unconditional cache for CFG --------------------------------
    def _build_uncond_cache(
        self,
        prefix_shape: torch.Size,
        cot_ids: Optional[torch.Tensor],
        target_len: int,
        B: int,
        H: int,
        device: torch.device,
        dtype: torch.dtype,
        adarms_cond: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Build an unconditional KV cache for classifier-free guidance.

        All embeddings are zeroed out so the backbone sees no context.
        The cache is constructed to have exactly ``target_len`` entries
        (matching the conditional cache).
        """
        # Forward entire zeroed prefix + CoT + <EOT> in one shot
        cot_len = cot_ids.shape[1] if cot_ids is not None else 0
        # prefix_shape = [B, P, H]  (ctx + <BOT>)
        P = prefix_shape[1]
        total = P + cot_len + 1  # prefix + CoT + <EOT>
        assert total == target_len, f"uncond cache length {total} != target {target_len}"

        zeros = torch.zeros(B, total, H, device=device, dtype=dtype)
        mask = self._make_causal_mask(total, B, device, dtype)
        if position_ids is not None:
            pos = position_ids
        else:
            pos = torch.arange(total, device=device).view(1, 1, total).expand(self._mrope_dims, B, total)  # [3, B, L] for M-RoPE

        uncond_kwargs = dict(
            inputs_embeds=zeros,
            attention_mask=mask,
            position_ids=pos,
            use_cache=True,
        )
        if adarms_cond is not None:
            uncond_kwargs["adarms_cond"] = adarms_cond
        out = self._text_model(**uncond_kwargs)
        return out.past_key_values

    def _clone_cache(self, cache):
        """Clone a DynamicCache so the original is not mutated by attention updates.

        Qwen3.5 has hybrid attention (full + linear layers), so both
        DynamicLayer (keys/values) and LinearAttentionLayer (conv_states/
        recurrent_states) must be handled.

        RL / SFT switch (``self.rl_grad_safe_cache``, DEFAULT ``False`` = SFT):
        only when the flag is set AND gradients are enabled (the Transfusion GRPO
        recipe re-scores the denoise steps with grad) is the clone a
        :class:`_GradSafeDynamicCache`, whose linear-attention state write-back
        REASSIGNS instead of ``copy_``-ing in place — the stock in-place copy (for
        CUDA-graph static addresses) would otherwise mutate the fused FLA kernel's
        saved ``initial_state`` and break backward ("... modified by an inplace
        operation ... CopyBackwards ... version 1"). SFT training / all inference
        keep the flag ``False`` and therefore the ORIGINAL stock ``DynamicCache``
        (in-place, CUDA-graph friendly) — so this change cannot affect SFT.
        """
        if cache is None:
            return None
        use_grad_safe = getattr(self, "rl_grad_safe_cache", False) and torch.is_grad_enabled()
        new_cache = _GradSafeDynamicCache() if use_grad_safe else DynamicCache()
        for layer_idx in range(len(cache.layers)):
            layer = cache.layers[layer_idx]
            # Full attention layers have .keys and .values
            if hasattr(layer, 'keys') and hasattr(layer, 'values') and hasattr(layer, 'is_initialized') and layer.is_initialized:
                new_cache.update(
                    layer.keys.clone(),
                    layer.values.clone(),
                    layer_idx,
                )
            else:
                # Linear attention layers or uninitialized layers:
                # Ensure the new cache has a matching placeholder layer
                while len(new_cache.layers) <= layer_idx:
                    new_cache.layers.append(type(layer)())
                new_layer = new_cache.layers[layer_idx]
                # Clone conv_states / recurrent_states if present
                if hasattr(layer, 'conv_states') and layer.conv_states is not None:
                    new_layer.conv_states = layer.conv_states.clone()
                    new_layer.is_conv_states_initialized = True
                    new_layer.dtype = layer.dtype
                    new_layer.device = layer.device
                    if hasattr(layer, 'max_batch_size'):
                        new_layer.max_batch_size = layer.max_batch_size
                    if hasattr(layer, 'conv_kernel_size'):
                        new_layer.conv_kernel_size = layer.conv_kernel_size
                if hasattr(layer, 'recurrent_states') and layer.recurrent_states is not None:
                    new_layer.recurrent_states = layer.recurrent_states.clone()
                    new_layer.is_recurrent_states_initialized = True
                if hasattr(layer, 'has_previous_state'):
                    new_layer.has_previous_state = layer.has_previous_state
        return new_cache

    # ---- Denoising loops --------------------------------------------
    def _action_decoder_dtype(self) -> torch.dtype:
        """Dtype the ``action_decoder`` expects for its input.

        The backbone is bf16 (RL fast path) while ``action_decoder`` is fp32, so
        hidden states must be re-cast to fp32 before entering it. Read the dtype
        from a concrete submodule weight via index access — this is safe under
        FSDP sharding, where ``next(self.action_decoder.parameters())`` raises
        ``StopIteration`` when the rank's shard has no parameters.
        """
        mdl = self.action_decoder
        while isinstance(mdl, torch.nn.Sequential) and len(mdl) > 0:
            mdl = mdl[0]
        return mdl.weight.dtype

    def _denoise_step_cached(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        past_kv,
        prefix_len: int,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
        action_position_ids: torch.Tensor,
        skip_cache_clone: bool = False,
    ) -> torch.Tensor:
        """
        One denoising step (raw action space) using a *frozen* prefix KV cache.

        Only the action suffix [act_embeds(T)] is forwarded through the backbone.
        When ``past_kv`` is provided, actions attend to the full prefix and
        bidirectionally to each other; when None they only attend to each other.

        ``skip_cache_clone=True`` passes ``past_kv`` to the suffix forward
        DIRECTLY (no protective clone): the attention layers then append the
        action K/V to the ORIGINAL cache in place. Only safe when nothing reads
        that cache afterwards (e.g. a single-step SDE rollout) — cloning per
        step costs seconds at RL batch sizes.
        """
        act_embeds = self._encode_noised_actions(x_t.to(dtype), t)  # [B, T, H]

        # AdaRMSNorm conditioning = time_embed(t)
        adarms_cond = self._build_action_cond(t, dtype)

        suffix_embeds = act_embeds
        T = suffix_embeds.shape[1]

        # Attention mask: depends on whether prefix KV cache is available
        if past_kv is not None:
            attn_mask = self._make_denoise_suffix_mask(prefix_len, B, device, dtype, suffix_len=T)
            cloned_kv = past_kv if skip_cache_clone else self._clone_cache(past_kv)
        else:
            attn_mask = self._make_denoise_suffix_mask(0, B, device, dtype, suffix_len=T)
            cloned_kv = None

        pos_ids = action_position_ids[:, :, :T].to(device=device, dtype=torch.long)

        model_kwargs = dict(
            inputs_embeds=suffix_embeds,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=cloned_kv,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        # During inference denoising, ALL suffix tokens are action tokens
        if self.special_FFN:
            model_kwargs["action_FFN"] = -1
        if self.special_QKV:
            model_kwargs["action_QKV"] = -1

        # Gradient checkpointing must NOT wrap this suffix forward: the layers
        # mutate the (cloned) KV cache in place, and a checkpoint recompute
        # during backward would re-append the same action K/V onto the mutated
        # cache, corrupting the attention shapes (the tiny T-token suffix costs
        # ~nothing to keep uncheckpointed anyway).
        hidden = self._run_text_model_no_ckpt(model_kwargs).last_hidden_state

        # Raw mode: action_decoder maps hidden → action space (no time info)
        # See forward()'s note: avoid next(self.action_decoder.parameters())
        # (StopIteration risk under FSDP sharding) -- reuse the already-computed
        # hidden-state dtype instead.
        decoder_dtype = self._action_decoder_dtype()
        return self.action_decoder(hidden.to(decoder_dtype)).float()

    def _run_text_model_no_ckpt(self, model_kwargs):
        """Call ``self._text_model`` with gradient checkpointing temporarily off.

        See ``_denoise_step_cached``: checkpoint recompute is incompatible with
        the in-place KV-cache mutation the attention layers perform. No-op when
        checkpointing is disabled (SFT / inference).
        """
        tm = self._text_model
        gc_flag = getattr(tm, "gradient_checkpointing", False)
        if not gc_flag:
            return tm(**model_kwargs)
        tm.gradient_checkpointing = False
        try:
            return tm(**model_kwargs)
        finally:
            tm.gradient_checkpointing = gc_flag

    def _denoise_fullseq_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx_embeds: torch.Tensor,
        prefix_pos: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prefix_adarms_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cache-free twin of ``_denoise_step_cached`` for the WITH-GRAD RL path.

        Forwards ``[ctx | noised_actions(T)]`` as ONE sequence (``use_cache=False``)
        and reads the action span's raw network output. Mathematically identical
        to ``_denoise_step_cached`` attending to a frozen prefix KV cache (same
        embeddings / causal-ctx + bidirectional-action mask / M-RoPE positions /
        t=0-prefix + t-action AdaRMSNorm conditioning / MoT action routing), but
        mutates NO cache, so per-layer gradient checkpointing (which recomputes
        the forward during backward) is safe. Costs one full-sequence forward
        per denoising step — for the RL recipe's single trained step this equals
        the cached path's prefix forward in FLOPs.
        """
        B, L_ctx = ctx_embeds.shape[0], ctx_embeds.shape[1]
        device, dtype = ctx_embeds.device, ctx_embeds.dtype

        act_embeds = self._encode_noised_actions(x_t.to(dtype), t)  # [B, T, H]
        action_cond = self._build_action_cond(t, dtype)
        T = act_embeds.shape[1]
        seq_embeds = torch.cat([ctx_embeds, act_embeds], dim=1)
        seq_len = L_ctx + T

        # ctx causal, actions one bidirectional block — same as the action rows
        # of ``_make_denoise_suffix_mask`` used by the cached path.
        ar_mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        if T > 1:
            ar_mask[L_ctx + 1:L_ctx + T] = False
        # NOTE: the 4D full-attention mask deliberately IGNORES left-padding
        # (all-ones input mask) — this EXACTLY mirrors the cached rollout path,
        # whose prefix forward (``_make_causal_mask``) and suffix mask
        # (``_make_denoise_suffix_mask``) both attend to padded positions too,
        # so the with-grad re-scoring stays numerically identical to the
        # behaviour policy (ratio == 1 at init). Padding is still honoured by
        # the linear-attention layers via ``linear_attn_padding_mask`` below
        # (again matching the cached path).
        input_mask = torch.ones(B, seq_len, device=device, dtype=torch.bool)
        full_mask = self.bool_mask_to_4d(self.make_attn_mask(input_mask, ar_mask), dtype)

        ext_pos = self._next_position_ids(prefix_pos, T)
        full_pos = torch.cat([prefix_pos, ext_pos], dim=-1)

        kwargs = dict(
            inputs_embeds=seq_embeds,
            attention_mask=full_mask,
            position_ids=full_pos,
            use_cache=False,
        )
        if prefix_adarms_cond is not None:
            kwargs["adarms_cond"] = torch.cat([
                prefix_adarms_cond.unsqueeze(1).expand(-1, L_ctx, -1),  # ctx @ t=0
                action_cond.unsqueeze(1).expand(-1, T, -1),             # actions @ t
            ], dim=1)
        if attention_mask is not None:
            kwargs["linear_attn_padding_mask"] = torch.cat([
                attention_mask.to(dtype),
                torch.ones(B, T, device=device, dtype=dtype),
            ], dim=1)
        if self.special_FFN:
            kwargs["action_FFN"] = (L_ctx, T)
        if self.special_QKV:
            kwargs["action_QKV"] = (L_ctx, T)

        hidden = self._text_model(**kwargs).last_hidden_state
        act_hidden = hidden[:, L_ctx:L_ctx + T, :]
        decoder_dtype = self._action_decoder_dtype()
        return self.action_decoder(act_hidden.to(decoder_dtype)).float()

    @torch.no_grad()
    def _denoise_loop_fm(
        self,
        x_t: torch.Tensor,
        past_kv,
        uncond_past_kv,
        prefix_len: int,
        B: int,
        device: torch.device,
        cfg_scale: float,
        dtype: torch.dtype,
        action_position_ids: torch.Tensor,
        fm_sde: bool = False,
        sde_noise_scale: float = 1.0,
        sde_kernel: str = "sit",
        sde_noise_level=None,
        sde_smooth_noise: Optional[dict] = None,
        step_recorder: Optional[List] = None,
    ) -> torch.Tensor:
        """Flow Matching reverse integration from t=1 -> t=0 with KV-cached prefix.

        Deterministic Euler ODE by default. Set ``fm_sde=True`` to use a
        stochastic SDE sampler instead, selected by ``sde_kernel``:

        * ``"sit"`` (default): the marginal-preserving SiT / stochastic-
          interpolants SDE; ``sde_noise_scale`` controls the stochasticity
          (0 -> ODE); the final step is kept noise-free.
        * ``"flow_grpo"``: the GRPO TRAINING rollout kernel
          (``FlowMatching.flow_grpo_step``): ``sde_noise_level`` is the
          reverse-SDE g(t)=a*sqrt(t) schedule hyper-parameter (scalar or
          per-dim [D], e.g. ``[0.05, 0.05, 0.02]``) and
          ``sde_smooth_noise`` optionally colours the drawn noise across
          waypoints. Noise is injected at EVERY step (exactly like the RL
          rollouts) and the chain output is clamped to [-1, 1], mirroring
          ``sample_grpo``'s ``x0_normed.clamp(-1, 1)`` before de-normalising.

        When ``step_recorder`` is a list, each step's
        ``{step, t, x_in, model_out, v_pred, x_out}`` is appended (on-device
        float tensors) for offline inspection / statistics.
        """
        timesteps = self.diffusion.get_inference_timesteps().to(device)
        n_steps = len(timesteps)
        dt = -1.0 / n_steps

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            model_out = self._denoise_step_cached(
                x_t, t, past_kv, prefix_len, B, device, dtype, action_position_ids,
            )
            if cfg_scale > 1.0:
                uncond_out = self._denoise_step_cached(
                    x_t, t, uncond_past_kv, prefix_len, B, device, dtype, action_position_ids,
                )
                model_out = uncond_out + cfg_scale * (model_out - uncond_out)

            # Convert the x0 prediction to the velocity field
            v_pred = self.diffusion.output_to_velocity(model_out, x_t, t)

            x_in = x_t
            if fm_sde and sde_kernel == "flow_grpo":
                # Training-rollout kernel: sigma_prev is the next grid point
                # (0.0 on the last step); noise is injected at EVERY step,
                # including the final one (parity with the GRPO rollouts).
                sig_prev_val = timesteps[i + 1] if i < n_steps - 1 else 0.0
                sig_prev = torch.full(
                    (B,), float(sig_prev_val), device=device, dtype=torch.float32,
                )
                x_t = self.diffusion.flow_grpo_step(
                    model_out, x_t, t, sig_prev,
                    noise_level=sde_noise_level,
                    add_noise=True,
                    smooth_noise=sde_smooth_noise,
                )
            elif fm_sde:
                # Keep the final step deterministic so we land on the clean estimate.
                add_noise = i < n_steps - 1
                x_t = self.diffusion.sde_step(
                    v_pred, x_t, t, dt,
                    noise_scale=sde_noise_scale, add_noise=add_noise,
                )
            else:
                x_t = self.diffusion.euler_step(v_pred, x_t, dt)

            if step_recorder is not None:
                step_recorder.append({
                    "step": i,
                    "t": t.detach().float(),
                    "x_in": x_in.detach().float(),
                    "model_out": model_out.detach().float(),
                    "v_pred": v_pred.detach().float(),
                    "x_out": x_t.detach().float(),
                })

        if fm_sde and sde_kernel == "flow_grpo":
            # Parity with the GRPO rollout's post-processing: the raw SDE chain
            # output is clamped to the normalised [-1, 1] action range before
            # de-normalising (sample_grpo: x0_normed.clamp(-1.0, 1.0)).
            x_t = x_t.clamp(-1.0, 1.0)
        return x_t

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_groups(
        self, weight_decay: float = 0.1, backbone_lr_scale: float = 1.0,
        base_lr: Optional[float] = None,
    ) -> List[Dict]:
        """Create parameter groups with optional differential learning rate.

        New action-head modules (action_encoder, action_decoder, time_embed,
        traj_pos_embed) use the full learning rate.  Pretrained backbone
        (qwen_model) uses ``lr * backbone_lr_scale`` to avoid corrupting
        pretrained features with aggressive updates.

        When ``base_lr`` is provided, each group gets an explicit ``lr`` key
        so PyTorch optimizers actually apply the differential rate.
        """
        # Module prefixes that belong to the new action head
        head_prefixes = ("action_encoder", "action_decoder", "time_embed", "traj_pos_embed")

        head_decay, head_nodecay = [], []
        back_decay, back_nodecay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head = any(name.startswith(hp) for hp in head_prefixes) or "action_mlp" in name or "action_q_proj" in name or "action_k_proj" in name or "action_v_proj" in name or "action_o_proj" in name
            is_nodecay = any(nd in name for nd in ("bias", "norm", "embed"))
            if is_head:
                (head_nodecay if is_nodecay else head_decay).append(p)
            else:
                (back_nodecay if is_nodecay else back_decay).append(p)

        def _make_group(params, wd, lr_scale):
            g = {"params": params, "weight_decay": wd}
            if base_lr is not None:
                g["lr"] = base_lr * lr_scale
            return g

        groups = [
            _make_group(head_decay, weight_decay, 1.0),
            _make_group(head_nodecay, 0.0, 1.0),
        ]
        if back_decay:
            groups.append(_make_group(back_decay, weight_decay, backbone_lr_scale))
        if back_nodecay:
            groups.append(_make_group(back_nodecay, 0.0, backbone_lr_scale))
        return groups
