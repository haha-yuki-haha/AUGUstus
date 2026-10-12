# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bind the framework-light diffusion-RL maths to a concrete Transfusion model.

:class:`TransfusionDiffusionAdapter` exposes exactly what Flow-GRPO and
DiffusionNFT need on top of a built ``TransfusionQwen3p5`` model, **reusing the
model's own building-block methods** (multimodal prefix encoding, the cached
per-step denoiser, the flow-matching helpers) so no Transfusion core file is
edited:

* :meth:`prepare_context` — encode the multimodal prompt once into a frozen
  prefix KV cache + conditioning vectors + action position ids (mirrors the
  setup phase of ``TransfusionQwen3p5._generate_trajectory_action_first``).  The expensive
  part runs once and is reused across every denoising step.
* :meth:`denoise_velocity` — the **differentiable** per-step velocity
  ``v_theta(x_t, t)`` via ``model._denoise_step_cached`` + ``output_to_velocity``.
* :meth:`sample_sde_with_logprob` — Flow-GRPO data collection: roll out the
  reverse-time SDE (:mod:`.fm_sde_logprob`) recording per-step states / times /
  log-probabilities.
* :meth:`recompute_sde_logprobs` — Flow-GRPO training: re-score the recorded
  transitions under the current policy (grad) to form the importance ratio.
* :meth:`nft_velocity` — DiffusionNFT training: velocity of an externally
  re-noised ``x_t`` at time ``t`` (grad on the trainable policy).

All tensor maths matches Transfusion's ``x_t = (1 - t) x_0 + t * noise`` flow
path. The Flow-GRPO SDE transition is dispatched on the network's
``prediction_type`` (see ``docs/X0_PREDICTION_GRPO.md``): an ``x0`` network
(Transfusion's checkpoint) uses the x0-parameterised closed-form Gaussian
kernel :func:`.fm_sde_logprob.fm_sde_step_with_logprob_x0` directly on the raw
model output; any other prediction type is first converted to a velocity with
``FlowMatching.output_to_velocity`` and uses Flow-GRPO's original kernel
:func:`.fm_sde_logprob.fm_sde_step_with_logprob`. The two forms are the same
transition distribution (unit-tested equivalence).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class TransfusionDiffusionAdapter:
    """Differentiable diffusion-RL view over a built ``TransfusionQwen3p5`` model."""

    def __init__(self, model, cfg_scale: float = 1.0):
        self.model = model
        # No CFG during RL by default (Flow-GRPO: the RL process performs CFG
        # distillation); cfg_scale > 1 builds an unconditional prefix too.
        self.cfg_scale = float(cfg_scale)
        if getattr(model, "diffusion_type", "flow_matching") != "flow_matching":
            raise ValueError(
                "TransfusionDiffusionAdapter requires diffusion_type='flow_matching'; "
                f"got {getattr(model, 'diffusion_type', None)!r}."
            )

    # -- convenience accessors ---------------------------------------------
    @property
    def diffusion(self):
        return self.model.diffusion

    @property
    def traj_token(self) -> int:
        return int(self.model.traj_token)

    @property
    def action_dim(self) -> int:
        return int(self.model.action_dim)

    # -- (1) multimodal prefix / conditioning ------------------------------
    def prepare_context(self, batch: Dict[str, Any], *, grad: bool = False) -> Dict[str, Any]:
        """Encode the prompt into a reusable denoising context.

        ``batch`` is a collated Transfusion model-input dict (the output of
        ``transfusion.dataset.TransfusionCollator``): ``input_ids``,
        ``attention_mask``, ``pixel_values_videos`` / ``video_grid_thw`` (or
        image equivalents), ``second_per_grid_ts``, ``state_features`` ...

        Set ``grad=True`` at training time so gradients flow through the prefix
        encoder; leave ``False`` for rollout/data-collection.
        """
        model = self.model
        with torch.set_grad_enabled(grad):
            return self._prepare_context_inner(batch)

    def _prepare_context_inner(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        import time as _time

        model = self.model
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        pixel_values_videos = batch.get("pixel_values_videos")
        video_grid_thw = batch.get("video_grid_thw")
        second_per_grid_ts = batch.get("second_per_grid_ts")
        state_features = batch.get("state_features")

        B = input_ids.shape[0]
        device = input_ids.device

        # ---- context embeddings (+ optional ViT capture) ----
        _t0 = _time.time()
        if model.vit_feature_enabled:
            ctx_embeds, vit_captured = model._get_multimodal_embeddings(
                input_ids, pixel_values, image_grid_thw,
                pixel_values_videos, video_grid_thw, capture_vit=True,
            )
        else:
            ctx_embeds = model._get_multimodal_embeddings(
                input_ids, pixel_values, image_grid_thw,
                pixel_values_videos, video_grid_thw,
            )
            vit_captured = None
        _t_mm = _time.time() - _t0
        dtype = ctx_embeds.dtype
        P = ctx_embeds.shape[1]  # action_first prefix = context only (actions follow)

        prefix_mask = model._make_causal_mask(P, B, device, dtype)
        prefix_pos, _ = model._get_extended_position_ids(
            input_ids, image_grid_thw, video_grid_thw, attention_mask,
            extra_len=0, second_per_grid_ts=second_per_grid_ts,
        )
        t_prefix = torch.zeros(B, device=device, dtype=torch.float32)
        prefix_adarms_cond = model._compute_adarms_cond(t_prefix, dtype=dtype)
        # GRAD path (RL actor update): skip the prefix KV-cache build entirely.
        # The cache is an inference-time compute optimisation; in the grad path
        # the attention layers mutate it IN PLACE, which is incompatible with
        # per-layer gradient checkpointing (the backward recompute re-appends
        # and corrupts shapes). The denoise re-scoring then uses the cache-free
        # full-sequence twin (``_denoise_fullseq_step``, identical maths) — for
        # the recipe's single trained step this is the SAME total FLOPs, and it
        # also frees the KV-cache memory. NO-GRAD paths (rollout / KL reference
        # precompute) keep the cache for fast multi-step sampling.
        grad_mode = torch.is_grad_enabled()
        if grad_mode:
            past_kv = None
        else:
            prefix_kwargs = dict(
                inputs_embeds=ctx_embeds,
                attention_mask=prefix_mask,
                position_ids=prefix_pos,
                use_cache=True,
            )
            if prefix_adarms_cond is not None:
                prefix_kwargs["adarms_cond"] = prefix_adarms_cond
            if attention_mask is not None:
                prefix_kwargs["linear_attn_padding_mask"] = attention_mask.to(dtype)
            past_kv = model._text_model(**prefix_kwargs).past_key_values

        action_position_ids = model._next_position_ids(prefix_pos, self.traj_token)

        # ---- optional unconditional prefix for CFG ----
        uncond_past_kv = None
        if self.cfg_scale > 1.0 and not grad_mode:
            H = ctx_embeds.shape[-1]
            zeros = torch.zeros(B, P, H, device=device, dtype=dtype)
            uncond_kwargs = dict(
                inputs_embeds=zeros,
                attention_mask=model._make_causal_mask(P, B, device, dtype),
                position_ids=prefix_pos,
                use_cache=True,
            )
            if prefix_adarms_cond is not None:
                uncond_kwargs["adarms_cond"] = prefix_adarms_cond
            if attention_mask is not None:
                uncond_kwargs["linear_attn_padding_mask"] = attention_mask.to(dtype)
            uncond_past_kv = model._text_model(**uncond_kwargs).past_key_values

        # ---- conditioning feature vectors (computed once, reused per step) ----
        _t1 = _time.time()
        cond_vecs: Dict[str, Optional[torch.Tensor]] = {}
        if model.scene_cond_enabled:
            cond_vecs["scene"] = model._compute_scene_cond(ctx_embeds, attention_mask, dtype)
        if model.state_cond_enabled:
            cond_vecs["state"] = model._compute_ego_cond(state_features, B, device, dtype)
        if model.vit_feature_enabled:
            cond_vecs["vit"] = model._compute_vit_cond(
                vit_captured, image_grid_thw, video_grid_thw, B, device, dtype,
            )
        _t_cond = _time.time() - _t1
        _t_prefix = (_time.time() - _t0) - _t_mm - _t_cond
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[time] prepare_context: mm_embeds=%.1fs prefix_llm=%.1fs cond=%.1fs (B=%d, grad=%s)",
            _t_mm, _t_prefix, _t_cond, B, torch.is_grad_enabled(),
        )

        return {
            "past_kv": past_kv,
            "uncond_past_kv": uncond_past_kv,
            "prefix_len": P,
            "action_position_ids": action_position_ids,
            "cond_vecs": cond_vecs,
            "B": B,
            "device": device,
            "dtype": dtype,
            # Extra handles needed to generate / score the post-hoc CoT
            # conditioned on the sampled action (action_first ``_generate_cot_after_action``
            # / ``cot_logprob``): the raw context embeddings, the context M-RoPE
            # positions (``extra_len=0``) and the ``t=0`` AdaRMSNorm conditioning.
            "ctx_embeds": ctx_embeds,
            "prefix_pos": prefix_pos,
            "prefix_adarms_cond": prefix_adarms_cond,
            "attention_mask": attention_mask,
        }

    # -- (2) differentiable per-step velocity ------------------------------
    @property
    def prediction_type(self) -> str:
        """The network's prediction parameterisation (``x0`` / ``v_prediction`` / ...).

        Drives which closed form of the Flow-GRPO Gaussian transition kernel is
        used (see ``docs/X0_PREDICTION_GRPO.md``): the SDE step is taken directly
        on the raw network output for ``x0`` networks
        (:func:`fm_sde_step_with_logprob_x0`) and on the recovered velocity
        otherwise (:func:`fm_sde_step_with_logprob`).
        """
        return str(getattr(self.diffusion, "prediction_type", "v_prediction"))

    def denoise_model_output(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx: Dict[str, Any],
        *,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Raw network output ``[B, T, D]`` (carries grad; CFG applied if enabled).

        For Transfusion's ``prediction_type='x0'`` checkpoint this IS the clean
        trajectory prediction ``x0_hat_theta(x_t, t)`` — the quantity the
        x0-parameterised Flow-GRPO Gaussian kernel is written in.
        """
        model = self.model
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        # Model-vintage compatibility: the ORIGINAL Code/Transfusion denoise
        # entry points accepted `cond_vecs` (the feature-conditioning vectors of
        # the scene/state/vit experiments); the cleaned AUGUstus_SFT model
        # removed the whole feature system. With every cond_* disabled the dict
        # is empty, so dropping it is semantics-preserving.
        import inspect as _inspect

        _drop_cond_vecs = "cond_vecs" not in _inspect.signature(
            model._denoise_step_cached
        ).parameters
        _cond = {} if _drop_cond_vecs else {"cond_vecs": ctx["cond_vecs"]}
        if ctx.get("past_kv") is None:
            # Grad path (cache-free): full-sequence denoise, gradient-
            # checkpointing safe (no cache mutation). Identical maths to the
            # cached suffix forward (see ``_denoise_fullseq_step``).
            if cfg_scale > 1.0:
                raise NotImplementedError(
                    "cfg_scale > 1 is not supported on the cache-free grad path."
                )
            return model._denoise_fullseq_step(
                x_t, t, ctx["ctx_embeds"], ctx["prefix_pos"], ctx.get("attention_mask"),
                ctx["prefix_adarms_cond"], **_cond,
            )
        model_out = model._denoise_step_cached(
            x_t, t, ctx["past_kv"], ctx["prefix_len"], ctx["B"], ctx["device"],
            ctx["dtype"], ctx["action_position_ids"], **_cond,
            skip_cache_clone=bool(ctx.get("_skip_cache_clone", False)),
        )
        if cfg_scale > 1.0 and ctx.get("uncond_past_kv") is not None:
            uncond_out = model._denoise_step_cached(
                x_t, t, ctx["uncond_past_kv"], ctx["prefix_len"], ctx["B"], ctx["device"],
                ctx["dtype"], ctx["action_position_ids"], **_cond,
            )
            model_out = uncond_out + cfg_scale * (model_out - uncond_out)
        return model_out

    def denoise_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx: Dict[str, Any],
        *,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Velocity ``v_theta(x_t, t)`` ``[B, T, D]`` (carries grad).

        ``t`` is a ``[B]`` float tensor in ``(0, 1]``. The raw model output is
        converted to a velocity with ``FlowMatching.output_to_velocity`` (an
        exact re-parameterisation for every ``prediction_type``).
        """
        model_out = self.denoise_model_output(x_t, t, ctx, cfg_scale=cfg_scale)
        return self.diffusion.output_to_velocity(model_out, x_t, t)

    def _sde_step(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        ctx: Dict[str, Any],
        *,
        noise_level: float,
        sigma_max: float,
        noise_schedule: str = "flow_grpo",
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """One reverse-SDE transition under the current policy (grad).

        Dispatches on :attr:`prediction_type`: an ``x0`` network feeds its raw
        output to the x0-parameterised Gaussian kernel (the modified closed
        form of ``docs/X0_PREDICTION_GRPO.md``); any other prediction type is
        converted to a velocity first and uses Flow-GRPO's original kernel.
        Both forms are the SAME distribution — the dispatch only changes which
        network quantity the mean is written (and differentiated) through.
        """
        from .fm_sde_logprob import fm_sde_step_with_logprob, fm_sde_step_with_logprob_x0

        if self.prediction_type == "x0":
            x0_hat = self.denoise_model_output(x_t, sigma, ctx)
            prev, log_prob, mean, std = fm_sde_step_with_logprob_x0(
                x0_hat, x_t, sigma, sigma_prev,
                noise_level=noise_level, sigma_max=sigma_max, schedule=noise_schedule,
                prev_sample=prev_sample, generator=generator,
            )
            return prev, log_prob, mean, std, x0_hat
        v = self.denoise_velocity(x_t, sigma, ctx)
        prev, log_prob, mean, std = fm_sde_step_with_logprob(
            v, x_t, sigma, sigma_prev,
            noise_level=noise_level, sigma_max=sigma_max, schedule=noise_schedule,
            prev_sample=prev_sample, generator=generator,
        )
        # Implied clean-trajectory prediction x0 = x_t - t * v (JiT record for
        # non-x0 networks; exact re-parameterisation of the velocity output).
        t_b = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma)
        t_b = t_b.to(device=x_t.device, dtype=torch.float32).reshape(-1)
        while t_b.ndim < x_t.ndim:
            t_b = t_b.unsqueeze(-1)
        x0_hat = x_t.float() - t_b * v.float()
        return prev, log_prob, mean, std, x0_hat

    # -- time schedule (mirrors FlowMatching.get_inference_timesteps) -------
    def sde_time_schedule(self, num_steps: int, device) -> List[Dict[str, torch.Tensor]]:
        """List of ``{sigma, sigma_prev}`` pairs for an ``num_steps`` SDE rollout.

        ``sigma[i] = 1 - i/N`` (optionally ``timestep_shift``-warped), the final
        ``sigma_prev`` is ``0`` so the last step lands on the clean estimate —
        identical spacing to ``FlowMatching.get_inference_timesteps`` with a
        fixed ``dt = -1/N``.
        """
        N = int(num_steps)
        shift = float(getattr(self.diffusion, "timestep_shift", 1.0) or 1.0)
        idx = torch.arange(N, dtype=torch.float32)
        progress = idx / N
        if shift != 1.0:
            progress = progress.pow(shift)
        times = (1.0 - progress).clamp(min=0.0)
        dt = 1.0 / N
        sched = []
        for i in range(N):
            sigma = times[i]
            sigma_prev = times[i + 1] if i + 1 < N else torch.tensor(0.0)
            if shift == 1.0:
                sigma_prev = torch.clamp(sigma - dt, min=0.0)
            sched.append({
                "sigma": sigma.to(device),
                "sigma_prev": sigma_prev.to(device),
            })
        return sched

    # -- (3) Flow-GRPO data collection: SDE rollout w/ per-step log-prob ----
    @torch.no_grad()
    def sample_sde_with_logprob(
        self,
        ctx: Dict[str, Any],
        *,
        num_steps: int = 10,
        noise_level=0.7,
        noise_schedule: str = "flow_grpo",
        x_init: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Any]:
        """Roll out the reverse-time SDE, recording the per-step MDP transitions.

        Returns a dict with the normalised clean trajectory ``x0_normed`` and the
        stacked per-step tensors needed for the Flow-GRPO update:
        ``x_in`` ``[S, B, T, D]``, ``next_sample`` ``[S, B, T, D]``,
        ``sigma`` / ``sigma_prev`` ``[S, B]``, ``log_probs`` ``[S, B]``.
        """
        B, device, dtype = ctx["B"], ctx["device"], ctx["dtype"]
        T, D = self.traj_token, self.action_dim
        # Single-step SDE: the suffix denoise may append its action K/V directly
        # to the ORIGINAL prefix cache — nothing reads that cache afterwards
        # (the CoT generation rebuilds its own), so the protective per-step
        # cache clone is pure overhead (seconds at RL batch sizes).
        if int(num_steps) == 1:
            ctx["_skip_cache_clone"] = True
        if x_init is None:
            x_t = self.model.sample_noise((B, T, D), device)
        else:
            x_t = x_init.to(device=device, dtype=torch.float32)

        sched = self.sde_time_schedule(num_steps, device)
        # Guard value substituted for sigma when sigma == 1 (Flow-GRPO uses
        # sigmas[1], i.e. the second-largest time ~ 1 - 1/N) so the first step's
        # std_dev_t = sqrt(sigma / (1 - sigma_max)) stays finite.
        sigma_max = float(sched[1]["sigma"].item()) if len(sched) > 1 else 1.0

        rec_x_in, rec_next, rec_sigma, rec_sigma_prev, rec_logp, rec_x0 = [], [], [], [], [], []
        for step in sched:
            sigma = step["sigma"].expand(B)
            sigma_prev = step["sigma_prev"].expand(B)
            next_sample, log_prob, _mean, _std, x0_hat = self._sde_step(
                x_t, sigma, sigma_prev, ctx,
                noise_level=noise_level, sigma_max=sigma_max,
                noise_schedule=noise_schedule, generator=generator,
            )
            rec_x_in.append(x_t)
            rec_next.append(next_sample)
            rec_sigma.append(sigma)
            rec_sigma_prev.append(sigma_prev)
            rec_logp.append(log_prob)
            rec_x0.append(x0_hat)
            x_t = next_sample

        x0_normed = x_t  # last next_sample is the (near-)clean estimate
        return {
            "x0_normed": x0_normed,
            "x_in": torch.stack(rec_x_in, dim=0),
            "next_sample": torch.stack(rec_next, dim=0),
            "sigma": torch.stack(rec_sigma, dim=0),
            "sigma_prev": torch.stack(rec_sigma_prev, dim=0),
            "log_probs": torch.stack(rec_logp, dim=0),
            # Behaviour-policy per-step x0 prediction (JiT-GRPO closed-form ratio).
            "x0_old": torch.stack(rec_x0, dim=0),
            "noise_level": noise_level,
            "noise_schedule": noise_schedule,
            "sigma_max": sigma_max,
        }

    # -- (4) Flow-GRPO training: re-score recorded transitions (grad) ------
    def recompute_sde_logprobs(
        self,
        ctx: Dict[str, Any],
        x_in: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        next_sample: torch.Tensor,
        *,
        noise_level: float = 0.7,
        sigma_max: float = 1.0,
        noise_schedule: str = "flow_grpo",
        step_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Per-step log-prob of the recorded ``next_sample`` under the current policy.

        ``x_in`` / ``next_sample`` are ``[S, B, T, D]``; ``sigma`` / ``sigma_prev``
        are ``[S, B]``.  Returns ``[S', B]`` log-probs (grad), where ``S'`` is the
        number of (optionally sub-selected) trained steps.
        """
        S = x_in.shape[0]
        steps = range(S) if step_indices is None else step_indices
        out = []
        for s in steps:
            _ps, log_prob, _mean, _std, _x0 = self._sde_step(
                x_in[s], sigma[s], sigma_prev[s], ctx,
                noise_level=noise_level, sigma_max=sigma_max,
                noise_schedule=noise_schedule, prev_sample=next_sample[s],
            )
            out.append(log_prob)
        return torch.stack(out, dim=0)

    # -- (4b) JiT-GRPO training: re-predict x0 per step (grad) -------------
    def recompute_x0_preds(
        self,
        ctx: Dict[str, Any],
        x_in: torch.Tensor,
        sigma: torch.Tensor,
        *,
        step_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Per-step clean-trajectory prediction ``x0_hat_theta(x_t, t)`` (grad).

        ``x_in`` is ``[S, B, T, D]``; ``sigma`` is ``[S, B]``.  Returns
        ``[S', B, T, D]`` x0 predictions under the current policy for the
        (optionally sub-selected) trained steps — the differentiable quantity
        the JiT-GRPO closed-form ratio/KL are written in (see
        :func:`.jit_grpo.jit_grpo_step_logratio_kl`).
        """
        S = x_in.shape[0]
        steps = range(S) if step_indices is None else step_indices
        steps = list(steps)
        if len(steps) == 1:
            # Single re-scored step: no later step re-reads the prefix cache,
            # so the per-call cache clone is unnecessary (see sample_sde_with_logprob).
            ctx["_skip_cache_clone"] = True
        out = []
        for s in steps:
            out.append(self.denoise_model_output(x_in[s], sigma[s], ctx))
        return torch.stack(out, dim=0)

    # -- (5) DiffusionNFT training: forward velocity of a re-noised x_t -----
    def nft_velocity(
        self,
        ctx: Dict[str, Any],
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Velocity at an externally re-noised ``x_t`` (DiffusionNFT forward pass)."""
        return self.denoise_velocity(x_t, t, ctx, cfg_scale=cfg_scale)

    # -- helpers -----------------------------------------------------------
    def renoise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """Forward flow-matching interpolation ``x_t = (1-t) x_0 + t * noise``."""
        return self.diffusion.q_sample(x_0.float(), t.float(), noise=noise)

    def denormalize(self, x_normed: torch.Tensor) -> torch.Tensor:
        """Map a normalised trajectory in ``[-1, 1]`` back to metric space."""
        traj = self.model.denormalize_actions(x_normed)
        if getattr(self.model, "action_repr", "absolute") == "delta":
            traj = self.model.delta_to_abs(traj)
        return traj


__all__ = ["TransfusionDiffusionAdapter"]
