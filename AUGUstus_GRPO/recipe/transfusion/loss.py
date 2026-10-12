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

"""Training-time RL loss for the Transfusion trajectory diffusion policy.

The actor update uses :func:`hybrid_grpo_loss_from_rollout`: the ACTION head is
a closed-form x0-prediction GRPO (exact per-step log-ratio + Gaussian KL from
:mod:`.algorithms.jit_grpo` on the FM-SDE rollout record) and the CoT head is a
per-token probability-ratio GRPO (:func:`.algorithms.cot_grpo.cot_token_grpo_loss`).

The function takes a :class:`~recipe.transfusion.algorithms.diffusion_adapter.TransfusionDiffusionAdapter`
(wrapping the current trainable policy), the collated model-input ``batch`` used
to rebuild the denoising context with gradients, the rollout data produced during
collection, and per-rollout ``advantages``; it returns ``(loss, metrics)``.

It is deliberately backend-neutral (plain ``torch``); the verl FSDP/Ray actor
update only needs to (1) build the adapter from its wrapped module, (2) hand over
the rollout tensors + advantages, and (3) call ``backward`` on the returned loss.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .algorithms.cot_grpo import cot_token_grpo_loss
from .algorithms.diffusion_adapter import TransfusionDiffusionAdapter
from .algorithms.fm_sde_logprob import _sde_kernel_coefficients
from .algorithms.jit_grpo import jit_grpo_policy_loss, jit_grpo_step_logratio_kl

import logging as _logging

_mem_logger = _logging.getLogger(__name__)


def _mem_log(tag: str) -> None:
    """Lightweight CUDA memory probe for RL update memory debugging."""
    if torch.cuda.is_available():
        _mem_logger.warning(
            "[mem] %s: alloc=%.2fGB peak=%.2fGB", tag,
            torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9,
        )


def collate_flow_grpo_records(records, device=None) -> Dict[str, Any]:
    """Re-batch per-sample Flow-GRPO records (from the rollout) into one dict.

    Inverse of ``TransfusionRollout._attach_flow_grpo_record``: stacks a list of
    ``B`` per-sample dicts (each holding ``[S, ...]`` slices) back into batched
    ``x_in``/``next_sample`` ``[S, B, T, D]`` and ``sigma``/``sigma_prev``/
    ``log_probs`` ``[S, B]`` tensors expected by the actor-loss re-score.
    """
    if not records:
        raise ValueError("collate_flow_grpo_records got an empty record list.")

    def _stack(key):
        t = torch.stack([torch.as_tensor(r[key]) for r in records], dim=1)
        return t.to(device) if device is not None else t

    keys = ["x_in", "next_sample", "sigma", "sigma_prev", "log_probs"]
    if "x0_old" in records[0]:  # JiT-GRPO closed-form ratio needs the old x0 preds
        keys.append("x0_old")
    out = {k: _stack(k) for k in keys}
    out["noise_level"] = _stack_noise_levels([_noise_level_from_record(r.get("noise_level", 0.7)) for r in records])
    out["noise_schedule"] = str(records[0].get("noise_schedule", "flow_grpo"))
    out["sigma_max"] = float(records[0].get("sigma_max", 1.0))
    return out


# Scalar/tensor equality + collation helpers for the per-sample noise-level
# plans: a batch of records may now carry EITHER the legacy batch-shared value
# OR per-sample slices of a per-group plan tensor; both handled uniformly.


def _noise_levels_equal(a, b) -> bool:
    """Scalar/tensor equality for two reconstructed noise levels."""
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and bool(torch.equal(a, b))
    if torch.is_tensor(a) or torch.is_tensor(b):
        return False
    return a == b


def _stack_noise_levels(nls):
    """Collate per-sample noise levels back into the training-time form.

    Scalar / batch-shared per-dim values are returned as-is (legacy behaviour);
    PER-SAMPLE plans (the NESTED per-group ``rl_noise_level`` form, where each
    record carries its own row slice of the plan tensor) are stacked back into a
    ``[B, T?, D]`` tensor — rank-3, exactly the broadcast shape the SDE kernels
    consume at re-score time.
    """
    if not nls:
        return 0.7
    first = nls[0]
    if all(_noise_levels_equal(v, first) for v in nls[1:]):
        return first
    if all(torch.is_tensor(v) for v in nls):
        return torch.stack(nls, dim=0)
    out = []
    for v in nls:
        if torch.is_tensor(v):
            out.append(v)
        else:
            out.append(torch.as_tensor([float(v)], dtype=torch.float32))
    return torch.stack(out, dim=0)


# Rebuild ONE record's stored noise level (inverse of the rollout-side stash).
def _noise_level_from_record(value):
    """Rebuild a record's noise level: scalar float OR per-dim/per-step or
    per-sample-sliced tensor.

    The rollout stores tensor noise levels ([D] per-dim, [T, D] time-scaled, or
    a sample's own plan slice ``[T?, D]`` from the NESTED per-group form) as
    numpy arrays inside the per-sample object record; scalars stay plain floats.
    """
    if torch.is_tensor(value):
        return value.detach().float().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, dtype=np.float32))
    if isinstance(value, (list, tuple)):
        return torch.as_tensor([float(v) for v in value], dtype=torch.float32)
    return float(value)



def _action_gaussian_entropy(
    x_in: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    step_indices,
    *,
    noise_level,
    sigma_max: float,
    noise_schedule: str,
) -> float:
    """Mean per-element Gaussian entropy of the reverse-SDE transitions (diagnostic).

    The transition std ``std = g(t) sqrt(-dt)`` is policy-independent, so this is
    a constant given the noise schedule (no gradient) — reported as
    ``actor/action_entropy`` to make the continuous-action exploration level
    visible alongside the discrete CoT entropy.
    """
    const = 0.5 * (1.0 + math.log(2.0 * math.pi))
    ents = []
    for s in step_indices:
        _sig, _dt, _sdt, std = _sde_kernel_coefficients(
            x_in[s], sigma[s], sigma_prev[s],
            noise_level=noise_level, sigma_max=sigma_max, eps=1e-6, schedule=noise_schedule,
        )
        ent = const + torch.log(std.clamp(min=1e-6))
        ents.append(ent.mean(dim=tuple(range(1, ent.ndim))))  # [B]
    if not ents:
        return 0.0
    return float(torch.stack(ents, dim=1).mean().detach())


def hybrid_grpo_loss_from_rollout(
    model: Any,
    batch: Dict[str, Any],
    rollout: Dict[str, Any],
    advantages: torch.Tensor,
    *,
    cot_ids: torch.Tensor,
    cot_old_logp: torch.Tensor,
    cot_mask: torch.Tensor,
    x0_normed: Optional[torch.Tensor] = None,
    cfg: Optional[Dict[str, Any]] = None,
    cfg_scale: float = 1.0,
    ref_model: Optional[Any] = None,
    x0_ref: Optional[torch.Tensor] = None,
    cot_ref_logp: Optional[torch.Tensor] = None,
    action_advantages: Optional[torch.Tensor] = None,
    cot_advantages: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined actor loss: closed-form action GRPO + token-level CoT GRPO.

    The trajectory action uses the JiT-GRPO closed-form x0 log-ratio
    (:func:`jit_grpo.jit_grpo_step_logratio_kl`); the CoT uses the plain per-token
    probability ratio (:func:`cot_token_grpo_loss`). Both share the SAME
    group-relative ``advantages``.

    When ``ref_model`` is given and either ``cfg['action_kl_coef']`` or
    ``cfg['cot_kl_coef']`` is > 0 (both fall back to the shared
    ``cfg['kl_coef']`` when unset), a KL-to-REFERENCE penalty is added to the
    corresponding head with the SAME machinery as the ratios:

    * action head — the exact closed-form Gaussian KL between the two policies'
      transition kernels, ``0.5 * ||c_0 (x0_policy - x0_ref)||^2 / std^2``
      (equal covariances, so it is symmetric), via
      :func:`jit_grpo.jit_grpo_step_logratio_kl` with the reference x0
      predictions in place of the behaviour ones;
    * CoT head — the k3 estimator from the plain per-token probability ratio
      ``pi_ref / pi_policy`` (:func:`cot_token_grpo_loss` ``ref_logp``/``kl_coef``).

    ``model`` is the (FSDP-wrapped) **actor** module. The differentiable model
    computation — the per-step x0 re-prediction and the per-token CoT log-prob —
    is run INSIDE one ``model(rl_score_fn=...)`` call so FSDP-1 gathers the (root)
    parameters that the denoiser / ``cot_logprob`` submodules need (they would see
    sharded params if called outside the root forward). The pure-tensor GRPO loss
    math (importance ratios, clipping) then runs on the returned quantities.

    Args:
        model:      FSDP-wrapped actor model exposing the ``rl_score_fn`` forward
            hook (see ``TransfusionQwen3p5.forward``) and ``cot_logprob``.
        batch:      collated Transfusion model inputs (already on device).
        rollout:    collated behaviour-policy SDE record (``x_in`` /
            ``next_sample`` ``[S,B,T,D]``, ``sigma`` / ``sigma_prev`` ``[S,B]``,
            ``x0_old`` ``[S,B,T,D]``, ``noise_level`` / ``sigma_max`` / ...).
        advantages: ``[B]`` group-relative advantage per rollout.
        cot_ids:    ``[B, R]`` sampled CoT token ids (right-padded).
        cot_old_logp: ``[B, R]`` behaviour-policy per-token CoT log-prob.
        cot_mask:   ``[B, R]`` 1/0 valid-CoT-token mask.
        x0_normed:  ``[B, T, D]`` clean sampled trajectory the CoT was conditioned
            on; defaults to the record's last ``next_sample``.
        cfg:        hyper-parameters (``action_clip_low`` / ``action_clip_high`` /
            ``cot_clip_low`` / ``cot_clip_high`` / ``entropy_coef`` /
            ``timestep_fraction`` / ``action_coef`` / ``cot_coef`` / ``kl_coef``
            / ``action_kl_coef`` / ``cot_kl_coef`` / ``cot_kl_estimator``).
        cfg_scale:  classifier-free-guidance scale for the denoiser (1.0 = off).
        ref_model:  optional frozen reference policy (plain eager module, NOT
            FSDP-wrapped) for the KL penalty; only used when either
            ``action_kl_coef`` or ``cot_kl_coef`` (or the shared ``kl_coef``
            they fall back to) is > 0.
        action_advantages / cot_advantages: optional SEPARATE per-head
            group-relative advantages (``[B]``) for the action and CoT heads.
            ``None`` -> both use the shared ``advantages`` (legacy single-scalar
            behaviour). Supplying them lets each head optimize ONLY the reward it
            can influence (planning vs. prediction), so the higher-variance
            component no longer dominates the other head.
    """
    from .algorithms.diffusion_adapter import TransfusionDiffusionAdapter

    cfg = cfg or {}
    action_clip_low = float(cfg.get("action_clip_low", cfg.get("clip_range", 0.2)))
    action_clip_high = cfg.get("action_clip_high", cfg.get("clip_range_high"))
    cot_clip_low = float(cfg.get("cot_clip_low", cfg.get("cot_clip_range", action_clip_low)))
    cot_clip_high = cfg.get("cot_clip_high", cfg.get("cot_clip_range_high"))
    if cot_clip_high is None:
        cot_clip_high = action_clip_high
    entropy_coef = float(cfg.get("entropy_coef", 0.0))
    timestep_fraction = float(cfg.get("timestep_fraction", 1.0))
    action_coef = float(cfg.get("action_coef", 1.0))
    cot_coef = float(cfg.get("cot_coef", 1.0))
    kl_coef = float(cfg.get("kl_coef", 0.0))
    action_kl_coef = float(cfg.get("action_kl_coef", kl_coef))
    cot_kl_coef = float(cfg.get("cot_kl_coef", kl_coef))
    # Diagnostic CoT new-vs-old approx-KL estimator ("k1"/"k3"); an unknown
    # value must fail loudly here (a typo'd yaml key otherwise silently gets k1).
    cot_kl_estimator = str(cfg.get("cot_kl_estimator", "k1"))
    if cot_kl_estimator not in ("k1", "k3"):
        raise ValueError(
            f"hybrid_grpo_loss_from_rollout: cot_kl_estimator must be 'k1' or 'k3', got {cot_kl_estimator!r}"
        )
    # Ablation switch: skip the CoT head loss entirely (rollout still records
    # the CoT ratios for diagnostics; they just contribute no gradient).
    train_only_action = bool(cfg.get("train_only_action", False))

    device = advantages.device
    advantages = advantages.float()

    x_in = rollout["x_in"]
    sigma = rollout["sigma"]
    sigma_prev = rollout["sigma_prev"]
    next_sample = rollout["next_sample"]
    if "x0_old" not in rollout:
        raise ValueError(
            "hybrid_grpo loss requires the recorded behaviour-policy x0 predictions "
            "'x0_old'; re-run the SDE rollout to populate it."
        )
    x0_old = rollout["x0_old"].to(device).float()
    noise_level = _noise_level_from_record(rollout.get("noise_level", 0.7))
    noise_schedule = str(rollout.get("noise_schedule", "flow_grpo"))
    sigma_max = float(rollout.get("sigma_max", 1.0))

    S = x_in.shape[0]
    n_train = max(1, int(round(S * timestep_fraction)))
    step_indices = list(range(n_train))  # train the first (highest-noise) steps

    if x0_normed is None:
        # Final clean trajectory (last recorded SDE step) the CoT was conditioned on.
        x0_normed = next_sample[-1]
    x0_normed = x0_normed.to(device).float()

    # ---- frozen-reference quantities for the KL penalty (no grad) ----
    # PREFER precomputed reference quantities: the worker runs a verl-style
    # batched ref-logprob phase ONCE per batch (chunked at
    # log_prob_micro_batch_size_per_gpu) and passes ``x0_ref`` / ``cot_ref_logp``
    # in, so the expensive full-context reference forward is NOT repeated inside
    # every actor micro-batch. Falls back to recomputing them here from
    # ``ref_model`` when they are not supplied (backward compatible). Reuses the
    # exact same computation as the policy below: per-step x0 re-predictions for
    # the action head, per-token CoT log-probs for the token head.
    if x0_ref is not None:
        x0_ref = x0_ref.to(device).float()
    if cot_ref_logp is not None:
        cot_ref_logp = cot_ref_logp.to(device).float()
    if (
        x0_ref is None
        and cot_ref_logp is None
        and ref_model is not None
        and max(action_kl_coef, cot_kl_coef) > 0.0
    ):
        with torch.no_grad():
            ref_adapter = TransfusionDiffusionAdapter(ref_model, cfg_scale=cfg_scale)
            ref_ctx = ref_adapter.prepare_context(batch, grad=False)
            x0_ref = ref_adapter.recompute_x0_preds(
                ref_ctx, x_in, sigma, step_indices=step_indices,
            ).detach()
            if not train_only_action:
                # Pass state_features only when cot_logprob accepts it.
                import inspect as _inspect

                _ref_kwargs = dict(
                    input_ids=batch["input_ids"],
                    actions_normed=x0_normed,
                    cot_ids=cot_ids.to(device),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    pixel_values_videos=batch.get("pixel_values_videos"),
                    video_grid_thw=batch.get("video_grid_thw"),
                    second_per_grid_ts=batch.get("second_per_grid_ts"),
                    ctx_embeds=ref_ctx.get("ctx_embeds"),
                )
                if "state_features" in _inspect.signature(ref_model.cot_logprob).parameters:
                    _ref_kwargs["state_features"] = batch.get("state_features")
                cot_ref_logp, _ = ref_model.cot_logprob(**_ref_kwargs)
                cot_ref_logp = cot_ref_logp.detach()

    # ---- differentiable model computation, run INSIDE the root forward so FSDP
    #      gathers the (root) params the denoiser / cot_logprob submodules use.
    #      With train_only_action the CoT forward is skipped entirely (the model
    #      call still runs so FSDP sees the same communication pattern, but the
    #      expensive CoT logits never materialise and _score returns Nones).
    #
    def _score(m):
        adapter = TransfusionDiffusionAdapter(m, cfg_scale=cfg_scale)
        ctx = adapter.prepare_context(batch, grad=True)
        _mem_log("hybrid_loss after prepare_context")
        x0_policy = adapter.recompute_x0_preds(ctx, x_in, sigma, step_indices=step_indices)
        _mem_log("hybrid_loss after recompute_x0_preds")
        if train_only_action:
            return x0_policy, None, None
        # Model-vintage compatibility: drop `state_features` when the installed
        # cot_logprob does not accept it (removed in the cleaned AUGUstus_SFT).
        import inspect as _inspect

        _cot_kwargs = dict(
            input_ids=batch["input_ids"],
            actions_normed=x0_normed,
            cot_ids=cot_ids.to(device),
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
            second_per_grid_ts=batch.get("second_per_grid_ts"),
            ctx_embeds=ctx.get("ctx_embeds"),
        )
        if "state_features" in _inspect.signature(m.cot_logprob).parameters:
            _cot_kwargs["state_features"] = batch.get("state_features")
        new_cot_logp, cot_entropy = m.cot_logprob(**_cot_kwargs)
        return x0_policy, new_cot_logp, cot_entropy

    x0_policy, new_cot_logp, cot_entropy = model(
        input_ids=batch["input_ids"], rl_score_fn=_score,
    )
    _mem_log("hybrid_loss after model forward")

    # ---- (1) action head: closed-form x0 log-ratio (pure tensor math) ----
    log_ratios, ref_kls = [], []
    for j, s in enumerate(step_indices):
        lr, _kl = jit_grpo_step_logratio_kl(
            x0_policy[j], x0_old[s], x_in[s], next_sample[s], sigma[s], sigma_prev[s],
            noise_level=noise_level, sigma_max=sigma_max, schedule=noise_schedule,
        )
        log_ratios.append(lr)
        if x0_ref is not None:
            # Exact Gaussian KL(policy || ref) in the SAME closed form: equal
            # covariances make it symmetric, and only x0_policy carries grad.
            _lr_ref, kl_ref = jit_grpo_step_logratio_kl(
                x0_policy[j], x0_ref[j], x_in[s], next_sample[s], sigma[s], sigma_prev[s],
                noise_level=noise_level, sigma_max=sigma_max, schedule=noise_schedule,
            )
            ref_kls.append(kl_ref)
    log_ratio = torch.stack(log_ratios, dim=1)  # [B, n_train]
    kl_stack = torch.stack(ref_kls, dim=1) if ref_kls else None
    # SPLIT GRPO: use per-head advantages if given, else the shared scalar.
    action_adv = action_advantages if action_advantages is not None else advantages
    cot_adv = cot_advantages if cot_advantages is not None else advantages
    action_adv = action_adv.to(device).float()
    cot_adv = cot_adv.to(device).float()
    action_loss, action_metrics = jit_grpo_policy_loss(
        log_ratio, action_adv,
        kl=kl_stack,
        clip_range=action_clip_low, clip_range_high=action_clip_high,
        kl_coef=action_kl_coef if kl_stack is not None else 0.0,
    )

    # ---- (2) CoT head: per-token probability ratio ----
    # (max_log_ratio clamps the per-token importance |log(pi_new/pi_old)|; 0 or
    # None disables the clamp, keeping the legacy unbounded behaviour)
    if train_only_action:
        cot_loss = torch.zeros((), device=device, dtype=action_loss.dtype)
        cot_metrics = {}
    else:
        _cot_max_lr = cfg.get("cot_max_log_ratio", 5.0)
        cot_loss, cot_metrics = cot_token_grpo_loss(
            new_cot_logp, cot_old_logp.to(device), cot_adv, cot_mask.to(device),
            entropy=cot_entropy, clip_range=cot_clip_low, clip_range_high=cot_clip_high,
            entropy_coef=entropy_coef,
            kl_coef=cot_kl_coef if cot_ref_logp is not None else 0.0,
            ref_logp=cot_ref_logp,
            kl_estimator=cot_kl_estimator,
            max_log_ratio=float(_cot_max_lr) if _cot_max_lr is not None and float(_cot_max_lr) > 0 else None,
        )

    # ---- combine ----
    # With train_only_action the CoT head is a zero tensor (and an empty metrics
    # dict), so the combined loss degenerates to the pure action head while the
    # rollout-side CoT ratios stay visible in the driver's diagnostics. The
    # per-head coefficients still apply, so a zeroed head never changes the
    # action head's gradient scale relative to the legacy single-head runs
    # (cot_coef * 0 == 0, and the metrics dict from the clamped head is kept
    # separate so nothing downstream has to special-case the ablation). The
    # same applies to the KL coefs above.
    loss = action_coef * action_loss + cot_coef * cot_loss

    action_entropy = _action_gaussian_entropy(
        x_in, sigma, sigma_prev, step_indices,
        noise_level=noise_level, sigma_max=sigma_max, noise_schedule=noise_schedule,
    )

    metrics: Dict[str, float] = {}
    metrics.update(action_metrics)
    metrics.update(cot_metrics)
    metrics["hybrid/loss"] = float(loss.detach())
    metrics["hybrid/action_loss"] = float(action_loss.detach())
    metrics["hybrid/cot_loss"] = float(cot_loss.detach())
    metrics["hybrid/adv_action_mean"] = float(action_adv.mean().detach())
    metrics["hybrid/adv_cot_mean"] = float(cot_adv.mean().detach()) if not train_only_action else 0.0
    metrics["hybrid/num_train_steps"] = float(n_train)
    metrics["hybrid/action_entropy"] = action_entropy
    metrics["hybrid/cot_entropy"] = cot_metrics.get("cot_grpo/entropy", 0.0)
    metrics["hybrid/action_kl"] = action_metrics.get("jit_grpo/kl", 0.0)
    metrics["hybrid/cot_kl"] = cot_metrics.get("cot_grpo/kl", 0.0)
    return loss, metrics


__all__ = [
    "collate_flow_grpo_records",
    "hybrid_grpo_loss_from_rollout",
]
