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

"""JiT-GRPO: closed-form x0-prediction GRPO for the flow-matching trajectory head.

"JiT" = the network directly predicts the clean trajectory ``x0`` (JiT-style
x0 prediction, ``prediction_type='x0'``). This module shares the same reverse-time
SDE Gaussian transition kernel as :mod:`.fm_sde_logprob`
(``fm_sde_step_with_logprob_x0`` mean/std) but computes the per-step importance
ratio and KL **in closed form directly from the two policies' x0 predictions**,
following the derivation in ``docs/SINGLE_STEP_X0_GRPO.md`` (§ "jit-grpo").

Given the recorded rollout transition ``x_t -> x_{t-1}`` sampled under the old
policy (``x_{t-1} = mu_old + std * eps``), with

    a   = noise_level,     t = sigma,      dt = t - sigma_prev  (> 0)
    m   = (x0_policy - x0_old) / t                 (= v_old - v_policy)
    c   = (1/a + a/2) * sqrt((1 - t) / t) * sqrt(dt)
    cm  = c * m = (mu_policy - mu_old) / std       (per element)

the log importance ratio and (exact, Gaussian) KL are

    log_ratio = c*m*eps - 0.5*(c*m)^2             (== Gaussian re-scoring, proven)
    KL(old || policy) = 0.5*(c*m)^2

Both are averaged over the non-batch dims (Flow-GRPO convention). The quantity
``std`` is policy-independent, so it cancels in the ratio; only the x0-prediction
difference ``x0_policy - x0_old`` carries gradient. See the unit tests for the
exact numerical equivalence with the (log-prob based) PPO-clip surrogate in
:mod:`.fm_sde_logprob`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .fm_sde_logprob import DEFAULT_NOISE_SCHEDULE, _sde_kernel_coefficients


def jit_grpo_step_logratio_kl(
    x0_policy: torch.Tensor,
    x0_old: torch.Tensor,
    sample: torch.Tensor,
    prev_sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    *,
    noise_level: float = 0.7,
    sigma_max: float = 1.0,
    schedule: str = DEFAULT_NOISE_SCHEDULE,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form per-step log importance ratio + exact KL for one SDE transition.

    Args:
        x0_policy:  current-policy clean-trajectory prediction ``x0_hat_theta``
            ``[B, T, D]`` (carries grad).
        x0_old:     behaviour-policy prediction recorded during rollout ``[B, T, D]``
            (no grad).
        sample:     the state ``x_t`` the step was taken from ``[B, T, D]``.
        prev_sample: the recorded next state ``x_{t-1}`` sampled under the old
            policy ``[B, T, D]``.
        sigma / sigma_prev: current / next flow-matching time ``[B]`` (or scalar).
        noise_level: the ``schedule``'s scalar hyper-parameter (``a`` / ``eta`` /
            ``lambda``); for the default ``flow_grpo`` schedule this is Flow-GRPO's
            ``a`` (``std_dev_t = sqrt(t/(1-t))*a``).
        sigma_max:  value substituted for ``sigma`` when ``sigma == 1`` (finite
            first-step std), matching :func:`fm_sde_step_with_logprob_x0`
            (only used by the ``flow_grpo`` schedule).
        schedule:   reverse-SDE diffusion-coefficient ``g(t)`` schedule (one of
            :data:`~recipe.transfusion.algorithms.fm_sde_logprob.NOISE_SCHEDULES`).
            The closed-form ratio/KL below hold for every schedule (they only use
            the shared ``std_dev_t`` / ``std``); only the ``c`` coefficient value
            changes with the schedule.
        eps:        numerical floor.

    Returns:
        ``(log_ratio, kl)`` each ``[B]`` (averaged over the non-batch dims).
        ``log_ratio = log pi_policy(x_{t-1}) - log pi_old(x_{t-1})`` and
        ``kl = KL(pi_old || pi_policy)`` in closed form.
    """
    x0_policy = x0_policy.float()
    x0_old = x0_old.float()
    sample = sample.float()
    prev_sample = prev_sample.float()

    # Shared (policy-independent) kernel coefficients: t, dt (<0, code convention),
    # std_dev_t = sigma_t (guarded at t==1), std = sigma_t * sqrt(-dt).
    t, dt, std_dev_t, std = _sde_kernel_coefficients(
        sample, sigma, sigma_prev, noise_level=noise_level, sigma_max=sigma_max, eps=eps,
        schedule=schedule,
    )

    # x0-form mean coefficients — identical to fm_sde_step_with_logprob_x0.
    vel_coef = (1.0 + std_dev_t**2 * (1.0 - t) / (2.0 * t)) * dt
    c_x = 1.0 + std_dev_t**2 / (2.0 * t) * dt + vel_coef / t
    c_0 = -vel_coef / t

    mean_old = c_x * sample + c_0 * x0_old
    mean_policy = c_x * sample + c_0 * x0_policy

    # cm = c * m = (mu_policy - mu_old) / std  (per element); eps = recorded noise.
    cm = (mean_policy - mean_old) / std
    noise = (prev_sample - mean_old) / std

    reduce_dims = tuple(range(1, sample.ndim))
    # log_ratio = c*m*eps - 0.5*(c*m)^2  (== Gaussian re-scoring, see docs/tests).
    log_ratio = (cm * noise - 0.5 * cm**2).mean(dim=reduce_dims)
    # Exact Gaussian KL(old || policy) = 0.5 * (c*m)^2 (equal covariances).
    kl = (0.5 * cm**2).mean(dim=reduce_dims)
    return log_ratio, kl


def jit_grpo_policy_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    *,
    kl: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    clip_range: float = 1e-4,
    clip_range_high: Optional[float] = None,
    kl_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PPO-clipped JiT-GRPO surrogate from the closed-form per-step log-ratio + KL.

    Args:
        log_ratio: ``[B, S]`` per-step closed-form log importance ratio (grad).
        advantages: ``[B]`` (broadcast over steps) or ``[B, S]`` group-relative
            advantage.
        kl:        ``[B, S]`` per-step exact KL(old || policy) for the penalty.
        mask:      ``[B, S]`` 1/0 mask of valid (trained) steps.
        clip_range: lower PPO clip ``1 - clip_range``.
        clip_range_high: optional asymmetric upper clip ``1 + clip_range_high``.
        kl_coef:   weight of the (exact) KL penalty.

    Returns:
        ``(loss, metrics)``.
    """
    log_ratio = log_ratio.float()
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    advantages = advantages.float().to(log_ratio.device)
    if advantages.shape[1] == 1 and log_ratio.shape[1] > 1:
        advantages = advantages.expand(-1, log_ratio.shape[1])

    if mask is None:
        mask = torch.ones_like(log_ratio)
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)

    hi = clip_range if clip_range_high is None else clip_range_high
    ratio = torch.exp(log_ratio)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + hi)
    per_step = torch.maximum(unclipped, clipped)
    policy_loss = (per_step * mask).sum() / denom

    clipfrac = (((unclipped < clipped).float()) * mask).sum() / denom

    loss = policy_loss
    kl_val = 0.0
    if kl is not None and kl_coef != 0.0:
        kl_term = (kl.float() * mask).sum() / denom
        loss = loss + kl_coef * kl_term
        kl_val = float(kl_term.detach())

    metrics = {
        "jit_grpo/policy_loss": float(policy_loss.detach()),
        "jit_grpo/loss": float(loss.detach()),
        "jit_grpo/ratio_mean": float(((ratio * mask).sum() / denom).detach()),
        "jit_grpo/clipfrac": float(clipfrac.detach()),
        "jit_grpo/kl": kl_val,
        "jit_grpo/adv_mean": float(((advantages * mask).sum() / denom).detach()),
        "jit_grpo/adv_abs_mean": float(((advantages.abs() * mask).sum() / denom).detach()),
    }
    return loss, metrics


__all__ = [
    "jit_grpo_step_logratio_kl",
    "jit_grpo_policy_loss",
]
