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

"""Flow-matching reverse-time SDE step with tractable log-probability.

Faithful port of Flow-GRPO's
``flow_grpo/diffusers_patch/sd3_sde_with_logprob.py::sde_step_with_logprob``
(https://github.com/yifan123/flow_grpo), specialised to Transfusion's trajectory
flow-matching path.

Both code-bases use the *same* linear interpolation convention

    x_t = (1 - t) * x_0 + t * noise ,    t in [0, 1]   (t = 0 clean, t = 1 noise)

so Flow-GRPO's ``sigma`` is exactly Transfusion's flow-matching time ``t``.

Two parameterisations of the SAME Gaussian transition kernel are provided (see
``docs/X0_PREDICTION_GRPO.md`` for the full derivation):

* :func:`fm_sde_step_with_logprob` — the **velocity** form (Flow-GRPO's
  original, for ``prediction_type='v_prediction'`` networks):

      dt        = sigma_prev - sigma                       (< 0, denoising)
      std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level
      mean      = x_t * (1 + std_dev_t**2 / (2*sigma) * dt)
                  +  v  * (1 + std_dev_t**2 * (1 - sigma) / (2*sigma)) * dt
      std       = std_dev_t * sqrt(-dt)

* :func:`fm_sde_step_with_logprob_x0` — the **x0** form (for Transfusion's
  ``prediction_type='x0'`` network). Flow-GRPO / DiffusionNFT both assume a
  velocity-predicting network; with ``x_t = (1-t) x_0 + t eps`` the velocity is
  ``v = (x_t - x0_hat) / t``, and substituting this into the velocity-form mean
  gives the closed form directly in the network output ``x0_hat``:

      b    = (1 + std_dev_t**2 * (1 - sigma) / (2*sigma)) * dt   (velocity coef)
      mean = (1 + std_dev_t**2/(2*sigma)*dt + b/sigma) * x_t  -  (b/sigma) * x0_hat

  i.e. ``mean = c_x * x_t + c_0 * x0_hat`` with ``c_0 = -b/sigma``. This is
  algebraically identical to composing ``FlowMatching.output_to_velocity`` with
  the velocity form (unit-tested), but avoids the explicit ``1/t`` division and
  makes the x0-prediction closed form the literal code path.

``noise_level = 0`` recovers the deterministic Euler ODE step ``x_t + v*dt``.
Crucially ``std`` is **policy-independent** (it only depends on ``sigma`` and
``noise_level``), so the Flow-GRPO importance ratio depends only on how the mean
shifts between the behaviour and current policies — numerically well-behaved.

The reverse-SDE diffusion coefficient ``g(t)`` (``std_dev_t``) is selectable via
``schedule`` (see :data:`NOISE_SCHEDULES`): ``flow_grpo`` (``a*sqrt(t/(1-t))``),
``dancegrpo`` (constant ``eta``) or ``lambda_t`` (``a*sqrt(t)``). ``noise_level``
scales the per-step std LINEARLY in every schedule. The mean closed form is
identical for all three (it depends only on ``std_dev_t**2``); see
``docs/DANCEGRPO_CONST_NOISE_LOGRATIO.md`` §9 for the equivalence.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

# A noise level is either one scalar (all dims share it) or a tensor/list that
# broadcasts against the [B, T, D] action: [D] (per-dimension, e.g. x/y/heading)
# or [T, D] (per-waypoint AND per-dimension, e.g. the accel*t^2 time scaling of
# :func:`resolve_noise_level`). All kernel maths below is elementwise/diagonal,
# so every closed form (mean, std, log-prob, ratio, KL) stays exact.
NoiseLevel = Union[float, Sequence[float], torch.Tensor]

# ---------------------------------------------------------------------------
# Reverse-time SDE diffusion-coefficient g(t) schedules.
#
# The Gaussian transition kernel mean is the SAME closed form for every
# schedule (it depends only on ``std_dev_t**2 = g(t)**2`` -- see
# ``docs/SINGLE_STEP_X0_GRPO.md`` §1 and ``docs/DANCEGRPO_CONST_NOISE_LOGRATIO.md``);
# ONLY the diffusion coefficient ``std_dev_t = g(t)`` (hence the per-step std
# ``std = g(t) * sqrt(-dt)``) changes. ``noise_level`` is each schedule's single
# scalar hyper-parameter (a / eta / lambda); ``noise_level == 0`` collapses g to
# 0 -> deterministic Euler ODE for ALL schedules.
#
#   flow_grpo : g(t) = a * sqrt(t / (1 - t))   (Flow-GRPO; diverges at t=1 ->
#               needs the ``sigma_max`` guard)
#   dancegrpo : g(t) = eta                     (DanceGRPO const; finite everywhere)
#   lambda_t  : g(t) = a * sqrt(t)             (proposed; finite at t=1, ->0 at t=0;
#               g^2 = a^2 t, i.e. docs' lambda = a^2 -- so noise_level scales the
#               std linearly here too, and a 1-step rollout has std = a exactly)
# ---------------------------------------------------------------------------
NOISE_SCHEDULE_FLOW_GRPO = "flow_grpo"
NOISE_SCHEDULE_DANCEGRPO = "dancegrpo"
NOISE_SCHEDULE_LAMBDA_T = "lambda_t"
NOISE_SCHEDULES = (
    NOISE_SCHEDULE_FLOW_GRPO,
    NOISE_SCHEDULE_DANCEGRPO,
    NOISE_SCHEDULE_LAMBDA_T,
)
DEFAULT_NOISE_SCHEDULE = NOISE_SCHEDULE_FLOW_GRPO

# Accept a few friendly aliases so config / env strings are forgiving.
_NOISE_SCHEDULE_ALIASES = {
    "flow_grpo": NOISE_SCHEDULE_FLOW_GRPO,
    "flow-grpo": NOISE_SCHEDULE_FLOW_GRPO,
    "flowgrpo": NOISE_SCHEDULE_FLOW_GRPO,
    "dancegrpo": NOISE_SCHEDULE_DANCEGRPO,
    "dance_grpo": NOISE_SCHEDULE_DANCEGRPO,
    "dance-grpo": NOISE_SCHEDULE_DANCEGRPO,
    "const": NOISE_SCHEDULE_DANCEGRPO,
    "constant": NOISE_SCHEDULE_DANCEGRPO,
    "lambda_t": NOISE_SCHEDULE_LAMBDA_T,
    "lambda-t": NOISE_SCHEDULE_LAMBDA_T,
    "lambdat": NOISE_SCHEDULE_LAMBDA_T,
    "lambda": NOISE_SCHEDULE_LAMBDA_T,
}


def normalize_noise_schedule(schedule: Optional[str]) -> str:
    """Canonicalise a schedule name (aliases -> one of :data:`NOISE_SCHEDULES`).

    ``None`` / empty -> :data:`DEFAULT_NOISE_SCHEDULE`. Raises ``ValueError`` on
    an unrecognised name so a typo fails loudly instead of silently defaulting.
    """
    if schedule is None:
        return DEFAULT_NOISE_SCHEDULE
    key = str(schedule).strip().lower()
    if key in ("", "none", "null"):
        return DEFAULT_NOISE_SCHEDULE
    if key not in _NOISE_SCHEDULE_ALIASES:
        raise ValueError(
            f"Unknown noise schedule {schedule!r}; expected one of {NOISE_SCHEDULES}."
        )
    return _NOISE_SCHEDULE_ALIASES[key]


def _diffusion_std_dev_t(
    sigma_b: torch.Tensor,
    *,
    schedule: str,
    noise_level: NoiseLevel,
    sigma_max: float,
    eps: float,
) -> torch.Tensor:
    """Reverse-SDE diffusion coefficient ``g(t)`` (== ``std_dev_t``) for ``schedule``.

    ``sigma_b`` is the (broadcast, clamped) flow-matching time ``t``. See the
    module-level schedule table for the three closed forms. ``noise_level`` is
    the schedule's std-scale hyper-parameter (``a`` / ``eta`` / ``a``) — a scalar
    OR a tensor broadcastable against the trailing action dims ([D] or [T, D]),
    in which case every formula is applied elementwise (diagonal SDE).
    """
    if torch.is_tensor(noise_level):
        a = noise_level.to(device=sigma_b.device, dtype=sigma_b.dtype)
    elif isinstance(noise_level, (list, tuple)):
        a = torch.as_tensor(noise_level, device=sigma_b.device, dtype=sigma_b.dtype)
    else:
        a = float(noise_level)
    sched = normalize_noise_schedule(schedule)
    if sched == NOISE_SCHEDULE_FLOW_GRPO:
        # g = a * sqrt(t / (1 - t)); substitute sigma_max for the 1 in the denom
        # when sigma == 1 (Flow-GRPO reference guard) to keep the first step finite.
        denom = torch.where(
            sigma_b >= 1.0,
            torch.as_tensor(sigma_max, dtype=sigma_b.dtype, device=sigma_b.device),
            sigma_b,
        )
        return torch.sqrt(sigma_b / (1.0 - denom).clamp(min=eps)) * a
    if sched == NOISE_SCHEDULE_DANCEGRPO:
        # g = eta (constant); finite for every t including t == 1.
        return torch.ones_like(sigma_b) * a
    if sched == NOISE_SCHEDULE_LAMBDA_T:
        # g = a * sqrt(t)  (i.e. g^2 = a^2 * t -- the docs' lambda = a^2); finite at
        # t == 1, ->0 as t -> 0 (clean terminal). Parameterised through the STD
        # multiplier ``a`` so noise_level scales the per-step std LINEARLY, exactly
        # like the flow_grpo / dancegrpo schedules: a SINGLE-step rollout
        # (sigma 1 -> 0) then has std = a -- e.g. a = 0.1 keeps the sampled action
        # within ~0.1 of the deterministic ODE output in the normalised [-1, 1]
        # action space (a = sqrt(lambda) of docs/SINGLE_STEP_X0_GRPO.md).
        return torch.sqrt(sigma_b.clamp(min=0.0)) * a
    raise ValueError(  # pragma: no cover - normalize_noise_schedule already guards
        f"Unknown noise schedule {schedule!r}; expected one of {NOISE_SCHEDULES}."
    )


def _as_bcast(value, like: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as a tensor broadcastable against ``like`` ([B, ...])."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    value = value.to(device=like.device, dtype=like.dtype)
    if value.ndim == 0:
        value = value.reshape(1)
    # [B] -> [B, 1, 1, ...] to broadcast over the trailing (T, D) dims.
    while value.ndim < like.ndim:
        value = value.unsqueeze(-1)
    return value


def _sde_kernel_coefficients(
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    *,
    noise_level: float,
    sigma_max: float,
    eps: float,
    schedule: str = DEFAULT_NOISE_SCHEDULE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared coefficients of the reverse-SDE Gaussian transition kernel.

    Returns ``(sigma_b, dt, std_dev_t, std)`` (all broadcast against ``sample``):

    * ``dt = sigma_prev - sigma`` (negative when denoising),
    * ``std_dev_t = g(t)`` from the selected ``schedule`` (see
      :func:`_diffusion_std_dev_t`),
    * ``std = std_dev_t * sqrt(-dt)`` — the (policy-independent) per-step std.

    The mean coefficients built downstream depend only on ``std_dev_t**2``, so a
    single formula serves all three schedules; only ``std_dev_t`` differs here.
    """
    sigma_b = _as_bcast(sigma, sample).clamp(min=eps, max=1.0)
    sigma_prev_b = _as_bcast(sigma_prev, sample).clamp(min=0.0, max=1.0)
    dt = sigma_prev_b - sigma_b  # negative when denoising (sigma decreasing)

    std_dev_t = _diffusion_std_dev_t(
        sigma_b, schedule=schedule, noise_level=noise_level, sigma_max=sigma_max, eps=eps,
    )

    # Per-step standard deviation of the Gaussian transition.
    std = std_dev_t * torch.sqrt((-dt).clamp(min=0.0))
    std = std.clamp(min=eps)
    return sigma_b, dt, std_dev_t, std


def _finish_step(
    sample: torch.Tensor,
    prev_sample_mean: torch.Tensor,
    std: torch.Tensor,
    *,
    prev_sample: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (or fix) ``x_prev`` and evaluate the Gaussian log-prob.

    Diagonal-Gaussian log density, then mean over all non-batch dims (matches
    Flow-GRPO's ``log_prob.mean(dim=tuple(range(1, ndim)))``).
    """
    if prev_sample is None:
        noise = torch.empty_like(sample).normal_(generator=generator)
        prev_sample = prev_sample_mean + std * noise
    else:
        prev_sample = prev_sample.float()

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * std**2)
        - torch.log(std)
        - math.log(math.sqrt(2.0 * math.pi))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, std


def fm_sde_step_with_logprob(
    velocity: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    *,
    noise_level: float = 0.7,
    sigma_max: float = 1.0,
    schedule: str = DEFAULT_NOISE_SCHEDULE,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One reverse-time SDE step + Gaussian log-prob of the transition.

    Args:
        velocity:    predicted flow-matching velocity ``v(x_t, t)`` ``[B, T, D]``.
        sample:      current state ``x_t`` ``[B, T, D]``.
        sigma:       current flow-matching time ``t`` ``[B]`` (or scalar).
        sigma_prev:  next (smaller) time ``[B]`` (or scalar); ``0`` lands on ``x_0``.
        noise_level: SDE stochasticity multiplier (Flow-GRPO ``noise_level``).
            ``0`` -> deterministic Euler step.
        sigma_max:   value substituted for ``sigma`` when ``sigma == 1`` to avoid
            division by zero (Flow-GRPO uses ``sigmas[1]``; ``~1`` here).
        prev_sample: if given, evaluate the log-prob of this *fixed* next state
            (used at training time to re-score the behaviour-policy sample).
            If ``None`` a fresh ``x_{prev}`` is sampled.
        generator:   optional RNG for reproducible sampling.
        eps:         numerical floor.

    Returns:
        ``(prev_sample, log_prob, mean, std)`` where ``log_prob`` is ``[B]``
        (averaged over all non-batch dims, matching Flow-GRPO) and ``mean`` /
        ``std`` describe the Gaussian transition kernel.
    """
    velocity = velocity.float()
    sample = sample.float()

    sigma_b, dt, std_dev_t, std = _sde_kernel_coefficients(
        sample, sigma, sigma_prev,
        noise_level=noise_level, sigma_max=sigma_max, eps=eps, schedule=schedule,
    )

    prev_sample_mean = sample * (
        1.0 + std_dev_t**2 / (2.0 * sigma_b) * dt
    ) + velocity * (
        1.0 + std_dev_t**2 * (1.0 - sigma_b) / (2.0 * sigma_b)
    ) * dt

    return _finish_step(
        sample, prev_sample_mean, std, prev_sample=prev_sample, generator=generator,
    )


def fm_sde_step_with_logprob_x0(
    x0_pred: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    *,
    noise_level: float = 0.7,
    sigma_max: float = 1.0,
    schedule: str = DEFAULT_NOISE_SCHEDULE,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x0-parameterised SDE step + log-prob (Transfusion ``prediction_type='x0'``).

    Same Gaussian transition kernel as :func:`fm_sde_step_with_logprob` but with
    the mean written **directly in the network's x0 prediction** — the closed
    form derived in ``docs/X0_PREDICTION_GRPO.md``: with
    ``v = (x_t - x0_hat) / sigma`` substituted into the velocity-form mean,

        b    = (1 + std_dev_t**2 * (1 - sigma) / (2*sigma)) * dt
        mean = c_x * x_t + c_0 * x0_hat ,
        c_x  = 1 + std_dev_t**2 * dt / (2*sigma) + b / sigma ,
        c_0  = -b / sigma .

    The gradient flows into the network through ``x0_hat`` only (``c_x`` /
    ``c_0`` / ``std`` are policy-independent), so the Flow-GRPO importance ratio
    between two policies is exactly the Gaussian ratio of two means that differ
    by ``c_0 * (x0_hat_new - x0_hat_old)``.

    Args / returns: identical to :func:`fm_sde_step_with_logprob` except the
    first argument is the raw network output ``x0_hat(x_t, sigma)`` ``[B, T, D]``
    instead of the velocity.
    """
    x0_pred = x0_pred.float()
    sample = sample.float()

    sigma_b, dt, std_dev_t, std = _sde_kernel_coefficients(
        sample, sigma, sigma_prev,
        noise_level=noise_level, sigma_max=sigma_max, eps=eps, schedule=schedule,
    )

    # Velocity coefficient b and the x0-form mean coefficients c_x / c_0.
    vel_coef = (1.0 + std_dev_t**2 * (1.0 - sigma_b) / (2.0 * sigma_b)) * dt
    c_x = 1.0 + std_dev_t**2 / (2.0 * sigma_b) * dt + vel_coef / sigma_b
    c_0 = -vel_coef / sigma_b
    prev_sample_mean = c_x * sample + c_0 * x0_pred

    return _finish_step(
        sample, prev_sample_mean, std, prev_sample=prev_sample, generator=generator,
    )


def resolve_noise_level(noise_level: NoiseLevel) -> NoiseLevel:
    """Resolve the configured noise level into the kernel's scalar / tensor form.

    Input forms of ``noise_level``:
      * scalar          -> all action dims share it;
      * list of ``D``   -> per-dimension, e.g. ``[nl_x, nl_y, nl_heading]``.

    Returns the input unchanged (scalar) or as a ``[D]`` tensor — every
    waypoint gets the same std.
    """
    if torch.is_tensor(noise_level):
        return noise_level.detach().float().cpu()
    if isinstance(noise_level, (list, tuple)):
        return torch.as_tensor([float(v) for v in noise_level], dtype=torch.float32)
    return float(noise_level)


def fm_sde_logprob_from_mean_std(
    prev_sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Gaussian log-prob of ``prev_sample`` under ``N(mean, std^2)`` ([B])."""
    prev_sample = prev_sample.float()
    mean = mean.float()
    std = std.float().clamp(min=1e-6)
    log_prob = (
        -((prev_sample - mean) ** 2) / (2.0 * std**2)
        - torch.log(std)
        - math.log(math.sqrt(2.0 * math.pi))
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


# ---------------------------------------------------------------------------
# Per-GROUP noise levels: [[nl_x, nl_y, nl_h], [nl_x, nl_y, nl_h]] mixes
# different exploration strengths WITHIN one rollout.n group.
# ---------------------------------------------------------------------------
def is_group_noise_plan(noise_level) -> bool:
    """True iff ``noise_level`` is the NESTED (per-group) form: a list whose own
    elements are sequences (``[[...], [...]]``). Flat scalars / per-dim lists are
    NOT group plans."""
    if isinstance(noise_level, torch.Tensor):
        return False
    if not isinstance(noise_level, (list, tuple)) or not noise_level:
        return False
    def _is_seq(x) -> bool:
        return isinstance(x, (list, tuple)) or type(x).__name__ == "ListConfig"
    first = [
        v for v in noise_level
        if _is_seq(v) or isinstance(v, (int, float))
    ]
    return len(first) == len(noise_level) and any(_is_seq(v) for v in first)


def group_noise_slot_mapping(group_size: int, num_levels: int) -> List[int]:
    """Which inner noise list each WITHIN-GROUP slot (0..G-1) maps to.

    The G samples of one GRPO group are split into ``num_levels`` contiguous
    chunks: every non-final list gets ``ceil(G / num_levels)`` samples, and the
    LAST list gets the remainder (fewer when G is not divisible). Lists beyond
    what the slots cover are simply unused (warned upstream)."""
    import math as _math
    base = max(1, _math.ceil(group_size / max(1, num_levels)))
    return [min(slot // base, num_levels - 1) for slot in range(group_size)]


def build_group_noise_plan(
    noise_level,
    *,
    num_rows: int,
    group_size: int,
    action_dim: int,
) -> torch.Tensor:
    """Resolve a NESTED ``rl_noise_level`` into a per-sample noise tensor.

    ``noise_level`` is a list of ``L`` inner entries (each a scalar or a
    ``[D]``-per-dim list). The ``group_size`` samples of EVERY group (rows are
    laid out as consecutive groups: row ``i`` is slot ``i % group_size``) are
    split into ``L`` contiguous chunks — every non-final inner list covers
    ``ceil(group_size / L)`` samples and the LAST inner list the remainder
    (``[0.05,...], [0.01,...]`` with G=8 -> 4 + 4; G=8, L=3 -> 3 + 3 + 2).

    Returns a CPU float32 tensor ``[num_rows, 1, action_dim]`` (constant across
    waypoints; broadcast safely against the SDE kernel's ``[B, ...]``
    elementwise maths).
    """
    inners = []
    for i, inner in enumerate(noise_level):
        if isinstance(inner, (int, float)):
            inner_res = float(inner)
        else:
            inner_res = [float(v) for v in inner]
        row = resolve_noise_level(inner_res)
        if not torch.is_tensor(row):
            row = torch.full((1, action_dim), float(row), dtype=torch.float32)
        else:
            # [D]-per-dim -> [1, D] (broadcastable against the waypoint axis
            # by the kernel's elementwise maths).
            if row.numel() == 1:
                row = row.expand(action_dim)
            if row.numel() != action_dim:
                raise ValueError(
                    f"rl_noise_level inner list #{i} has {row.numel()} entries "
                    f"but the action has D={action_dim} dims."
                )
            row = row.detach().float().cpu().reshape(1, -1)
        inners.append(row)

    slots = group_noise_slot_mapping(group_size, len(inners))
    used = sorted(set(slots))
    per_row = [inners[slots[i % group_size]] for i in range(num_rows)]
    plan = torch.stack(per_row, dim=0)  # [num_rows, 1, D]
    import logging as _logging

    if len(used) < len(inners):
        _logging.getLogger(__name__).warning(
            "rl_noise_level declares %d lists but group_size=%d only covers %d "
            "of them (lists beyond #%d get NO samples).",
            len(inners), group_size, len(used), used[-1],
        )
    return plan


__all__ = [
    "fm_sde_step_with_logprob",
    "fm_sde_step_with_logprob_x0",
    "fm_sde_logprob_from_mean_std",
    "normalize_noise_schedule",
    "resolve_noise_level",
    "is_group_noise_plan",
    "group_noise_slot_mapping",
    "build_group_noise_plan",
    "NoiseLevel",
    "NOISE_SCHEDULE_FLOW_GRPO",
    "NOISE_SCHEDULE_DANCEGRPO",
    "NOISE_SCHEDULE_LAMBDA_T",
    "NOISE_SCHEDULES",
    "DEFAULT_NOISE_SCHEDULE",
]
