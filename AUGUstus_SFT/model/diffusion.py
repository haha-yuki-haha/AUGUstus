"""
Flow Matching for Transfusion trajectory prediction.

Lipman et al. 2023 / pi0 style, with x0 prediction and ODE/SDE-based sampling.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Flow Matching  (Lipman et al. 2023, pi0 style)
# ---------------------------------------------------------------------------


def sample_beta_distribution(
    alpha: float, beta: float, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Sample from Beta distribution for Flow Matching time scheduling (pi0 style)."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((batch_size,))


# ---------------------------------------------------------------------------
# Flow-GRPO reverse-SDE kernel (training-rollout parity port)
#
# Exact port of the Training/RL side kernel
#   verl_transfusion/recipe/transfusion/algorithms/fm_sde_logprob.py
# (``fm_sde_step_with_logprob`` / ``fm_sde_step_with_logprob_x0`` /
# ``smooth_noise_like``) so the test-time multi-step sampler can reproduce the
# GRPO rollout distribution EXACTLY. Only torch is used; the maths is copied
# one-to-one (same clamping, same broadcast, same closed forms):
#
#     x_t = (1 - t) * x_0 + t * noise          (t = 0 clean, t = 1 noise)
#     dt        = sigma_prev - sigma           (< 0, denoising)
#     std_dev_t = g(sigma) = a*sqrt(sigma)      (lambda_t schedule)
#     std       = std_dev_t * sqrt(-dt)
#     vel_coef  = (1 + std_dev_t^2*(1-sigma)/(2sigma)) * dt
#     v-form    : mean = x_t*(1 + std_dev_t^2*dt/(2sigma)) + v*vel_coef
#     x0-form   : mean = c_x*x_t + c_0*x0_hat,
#                 c_x = 1 + std_dev_t^2*dt/(2sigma) + vel_coef/sigma,
#                 c_0 = -vel_coef/sigma
#     x_prev    = mean + std * z
#
# g(t) schedule (``noise_level`` is the scalar hyper-parameter a and may be a
# scalar or a per-action-dim [D] list):
#   lambda_t  : g = a*sqrt(t)          (finite at t=1, clean terminal)
#
# ``noise_level = 0`` collapses the SDE step to the deterministic Euler step.
# ---------------------------------------------------------------------------


class FlowMatching(nn.Module):
    """
    Flow Matching for continuous trajectory tokens (pi0 / Lipman et al. 2023).

    Flow Matching defines a probability path from data x_0 to noise x_1 via
    linear interpolation:

        x_t = (1 - t) * x_0 + t * noise,   t ∈ [0, 1]

    Prediction type: "x0" — the network predicts the clean data x_0 directly
    (JiT style, with velocity-space loss).

    Inference uses Euler ODE integration from t=1 (pure noise) to t=0 (clean data).

    Time sampling follows pi0's Beta(1.5, 1.0) distribution scaled to [0.001, 1.0].
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        timestep_shift: float = 1.0,
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        time_min: float = 0.001,
        time_max: float = 1.0,
        clip_sample_range: float = 0.0,
    ):
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.timestep_shift = timestep_shift
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.time_min = time_min
        self.time_max = time_max
        self.clip_sample_range = clip_sample_range

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample training timesteps t ∈ [time_min, time_max] from Beta distribution.

        Following pi0: time ~ Beta(1.5, 1.0) * 0.999 + 0.001
        """
        t_beta = sample_beta_distribution(
            self.time_beta_alpha, self.time_beta_beta, batch_size, device
        )
        t = t_beta * (self.time_max - self.time_min) + self.time_min
        return t.to(dtype=torch.float32, device=device)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flow Matching forward interpolation: x_t = (1 - t) * x_0 + t * noise.

        Args:
            x_0: clean data [B, traj_token, traj_dim]
            t:   timesteps  [B] in [0, 1]
            noise: optional pre-sampled noise

        Returns:
            x_t:   interpolated data [B, traj_token, traj_dim]
            noise: the noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        t_expanded = t[:, None, None]  # [B, 1, 1]
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * noise
        return x_t, noise

    def training_losses(
        self,
        model_output: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Flow Matching training loss (x0 prediction: target = x_0).

        Args:
            model_output: predicted output [B, traj_token, traj_dim]
            x_0: clean data [B, traj_token, traj_dim]
            x_t: interpolated data [B, traj_token, traj_dim]
            t:   timesteps [B] in [0, 1]
            noise: noise [B, traj_token, traj_dim]

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(model_output, x_0, reduction="mean")

    def output_to_velocity(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the x0 prediction to the velocity field for Euler integration.

        velocity = (x_t - x_0) / t.
        """
        t_expanded = t.clone()
        while t_expanded.dim() < x_t.dim():
            t_expanded = t_expanded.unsqueeze(-1)
        x0 = model_output
        if self.clip_sample_range > 0:
            x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
        return (x_t - x0) / t_expanded.clamp(min=1e-5)

    @torch.no_grad()
    def euler_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Single Euler integration step: x_{t+dt} = x_t + dt * v_θ(x_t, t).

        For inference we integrate from t=1 → t=0, so dt is negative.
        """
        return x_t + dt * model_output

    @torch.no_grad()
    def sde_step(
        self,
        velocity: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        noise_scale: float = 1.0,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Single reverse-time SDE step for the linear Flow-Matching path.

        The probability-flow ODE drift is the velocity ``v``.  A marginal-
        preserving reverse-time SDE (integrating t: 1 -> 0, so ``dt < 0``) is

            dx = [v - 0.5 * w(t) * s(x, t)] dt + sqrt(w(t)) dW̄,

        where the score of the linear path ``x_t = (1 - t) x_0 + t·ε`` is
        ``s = -ε̂ / t`` with the noise estimate ``ε̂ = x_t + (1 - t)·v``
        (SiT / stochastic-interpolants formulation).  The diffusion coefficient
        uses the ``sigma`` form ``w(t) = noise_scale · t`` so that the ``1/t``
        in the score cancels (drift = ``v + 0.5·noise_scale·ε̂``) and the
        injected noise vanishes as ``t -> 0``.  ``noise_scale = 0`` (or
        ``add_noise=False``) recovers the deterministic Euler ODE step.

        Args:
            velocity:    velocity field v_θ(x_t, t)            [B, T, D]
            x_t:         current state                          [B, T, D]
            t:           current time in (0, 1]                 [B]
            dt:          (negative) integration step, e.g. -1/N
            noise_scale: stochasticity multiplier (>= 0); 0 -> ODE
            add_noise:   inject Brownian noise this step (set False on the
                         final step to land on the deterministic estimate)

        Returns:
            x at time ``t + dt``.
        """
        t_exp = t.clone()
        while t_exp.dim() < x_t.dim():
            t_exp = t_exp.unsqueeze(-1)
        t_exp = t_exp.clamp(min=self.time_min)

        eps_hat = x_t + (1.0 - t_exp) * velocity      # noise estimate ε̂
        score = -eps_hat / t_exp                       # ∇ log p_t(x_t)
        w_t = noise_scale * t_exp                       # diffusion coeff (sigma form)
        drift = velocity - 0.5 * w_t * score            # = v + 0.5·noise_scale·ε̂
        x_next = x_t + drift * dt
        if add_noise and noise_scale > 0.0:
            z = torch.randn_like(x_t)
            x_next = x_next + torch.sqrt(w_t * abs(dt)) * z
        return x_next

    # ---- Flow-GRPO SDE kernel (training-rollout parity) -------------------
    @staticmethod
    def _flow_grpo_diffusion_g(
        sigma_b: torch.Tensor,
        noise_level,
        eps: float,
    ) -> torch.Tensor:
        """Reverse-SDE diffusion coefficient g(t) for one step (lambda_t schedule).

        ``sigma_b`` is the broadcast / clamped FM time (any shape); ``noise_level``
        is the std-scale hyper-parameter — a scalar OR a tensor / list
        broadcastable against the trailing action dims ([D]). Elementwise port of
        the training side's ``_diffusion_std_dev_t``.
        """
        if torch.is_tensor(noise_level):
            a = noise_level.to(device=sigma_b.device, dtype=sigma_b.dtype)
        elif isinstance(noise_level, (list, tuple)):
            a = torch.as_tensor(noise_level, device=sigma_b.device, dtype=sigma_b.dtype)
        else:
            a = float(noise_level)
        return torch.sqrt(sigma_b.clamp(min=0.0)) * a

    @torch.no_grad()
    def flow_grpo_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prev: torch.Tensor,
        noise_level=0.7,
        add_noise: bool = True,
        generator: Optional[torch.Generator] = None,
        smooth_noise: Optional[dict] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """One reverse-SDE step with the TRAINING (GRPO rollout) kernel.

        Faithful port of ``fm_sde_logprob.fm_sde_step_with_logprob_x0``: the
        Gaussian transition ``x_prev = mean + std * z`` whose mean is written
        directly in the network's x0 prediction. ``noise_level`` may be a scalar
        or a per-dim ``[D]`` list (e.g. ``[0.05, 0.05, 0.02]``); the diffusion
        coefficient is g(t) = a*sqrt(t) (lambda_t schedule).

        Args:
            model_output: raw network output ``[B, T, D]`` (the x0_hat prediction).
            x_t:          current state ``[B, T, D]``.
            sigma:        current FM time ``[B]`` (or scalar).
            sigma_prev:   next (smaller) FM time ``[B]`` (or scalar); ``0``
                lands on x0.
            noise_level:  the g(t) std-scale hyper-parameter; ``0`` ->
                deterministic Euler step.
            add_noise:    False falls back to the deterministic mean step
                (training rollouts always sample, i.e. True).
            generator:    optional RNG for reproducible sampling.
            smooth_noise: kwargs dict for :func:`smooth_noise_like` (waypoint-
                correlated exploration noise) or None for i.i.d. white noise.
            eps:          numerical floor.

        Returns:
            ``x_prev`` ``[B, T, D]`` at time ``sigma_prev`` (float32, like the
            training kernel which works in full precision).
        """
        sample = x_t.float()
        # sigma_b / sigma_prev_b broadcast to [B, 1, ...]; dt < 0 when denoising.
        sigma_b = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
        sigma_prev_b = torch.as_tensor(sigma_prev, device=sample.device, dtype=torch.float32)
        if sigma_b.dim() == 0:
            sigma_b = sigma_b.expand(sample.shape[0])
        if sigma_prev_b.dim() == 0:
            sigma_prev_b = sigma_prev_b.expand(sample.shape[0])
        while sigma_b.dim() < sample.dim():
            sigma_b = sigma_b.unsqueeze(-1)
        while sigma_prev_b.dim() < sample.dim():
            sigma_prev_b = sigma_prev_b.unsqueeze(-1)
        sigma_b = sigma_b.clamp(min=eps, max=1.0)
        sigma_prev_b = sigma_prev_b.clamp(min=0.0, max=1.0)
        dt = sigma_prev_b - sigma_b

        # Policy-independent per-step std = g(t) * sqrt(-dt).
        std_dev_t = self._flow_grpo_diffusion_g(
            sigma_b, noise_level, eps,
        )
        std = (std_dev_t * torch.sqrt((-dt).clamp(min=0.0))).clamp(min=eps)

        vel_coef = (1.0 + std_dev_t**2 * (1.0 - sigma_b) / (2.0 * sigma_b)) * dt
        # x0-form: mean written directly in the x0 prediction.
        x0_pred = model_output.float()
        c_x = 1.0 + std_dev_t**2 / (2.0 * sigma_b) * dt + vel_coef / sigma_b
        c_0 = -vel_coef / sigma_b
        mean = c_x * sample + c_0 * x0_pred

        if not add_noise:
            return mean
        if smooth_noise is not None:
            z = smooth_noise_like(sample, generator=generator, **smooth_noise)
        else:
            z = torch.empty_like(sample).normal_(generator=generator)
        return mean + std * z


    def get_inference_timesteps(self) -> torch.Tensor:
        """
        Return shifted timesteps from 1.0 to near 0.0 for Euler integration.

        Returns tensor of shape [num_inference_steps] with values [1.0, ..., dt].
        """
        if self.num_inference_steps <= 1:
            return torch.tensor([1.0], dtype=torch.float32)

        N = self.num_inference_steps
        # Left endpoints of N equal intervals spanning [0, 1].
        # times[i] = 1 - i/N  for i = 0, 1, ..., N-1
        # Result: [1.0, 1-1/N, ..., 1/N]  — does NOT include t=0.
        #
        # Rationale: Euler integration uses N steps of dt = -1/N starting at
        # t=1 and arriving at t=0.  Each step queries the velocity at the LEFT
        # endpoint of its interval.  Including t=0 would add an extra step
        # beyond the terminal point, and t=0 is also outside the training
        # distribution (time_min=0.001), leading to unreliable predictions.
        progress = torch.arange(N, dtype=torch.float32) / N
        if self.timestep_shift != 1.0:
            progress = progress.pow(self.timestep_shift)
        times = 1.0 - progress
        # Enforce strictly decreasing times
        eps = 1e-6
        for i in range(1, N):
            if times[i] >= times[i - 1]:
                times[i] = times[i - 1] - eps
        times = torch.clamp(times, min=0.0)
        return times


# ---------------------------------------------------------------------------
# Smooth (waypoint-correlated) exploration noise.
#
# One-to-one port of the training side's ``fm_sde_logprob.smooth_noise_like``
# (an RBF/GP Cholesky colouring of the sampled SDE noise along the waypoint
# axis): unit marginal variance is preserved, so the Flow-GRPO Gaussian
# transition marginals are unchanged and only the drawn trajectory is smooth.
# ---------------------------------------------------------------------------
_SMOOTH_L_CACHE: dict = {}


def _smooth_correlation_cholesky(
    num_waypoints: int,
    length_scale: float,
    *,
    anchor: Optional[float],
    eps: float,
    device,
    dtype,
) -> Optional[torch.Tensor]:
    """Lower-Cholesky ``L`` of an RBF CORRELATION matrix over the waypoint axis.

    ``R[i, j] = exp(-(i - j)^2 / (2 l^2))`` (RBF, unit diagonal), optionally
    conditioned on a GP ``anchor`` waypoint. Returns ``None`` when
    ``length_scale <= 0`` (caller keeps i.i.d. white noise for that channel).
    Cached per ``(T, length_scale, anchor, device, dtype)``.
    """
    if length_scale is None or float(length_scale) <= 0.0:
        return None
    key = (
        int(num_waypoints), float(length_scale),
        None if anchor is None else float(anchor),
        float(eps), str(device), str(dtype),
    )
    cached = _SMOOTH_L_CACHE.get(key)
    if cached is not None:
        return cached
    t = torch.arange(num_waypoints, dtype=torch.float32)
    l2 = 2.0 * float(length_scale) ** 2
    cov = torch.exp(-((t.unsqueeze(1) - t.unsqueeze(0)) ** 2) / l2)  # RBF, unit diag
    if anchor is not None:
        k_a = torch.exp(-((t - float(anchor)) ** 2) / l2)
        cov = cov - torch.outer(k_a, k_a)
    # Rescale to a correlation matrix -> exact unit marginal variance per waypoint.
    d = torch.sqrt(cov.diagonal().clamp(min=eps))
    cov = cov / torch.outer(d, d)
    cov = cov + eps * torch.eye(num_waypoints)  # numerical PD guard
    L = torch.linalg.cholesky(cov).to(device=device, dtype=dtype)
    _SMOOTH_L_CACHE[key] = L
    return L


def smooth_noise_like(
    sample: torch.Tensor,
    *,
    length_scale: float,
    heading_length_scale: Optional[float] = None,
    n_position_dims: int = 2,
    anchor: Optional[float] = None,
    eps: float = 1e-4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Smooth (waypoint-correlated) UNIT-marginal-variance noise ``like`` ``[B, T, D]``.

    Correlates the noise along the T (waypoint) axis INDEPENDENTLY per action
    dim via an RBF-Cholesky factor. The first ``n_position_dims`` dims (x, y)
    share ``length_scale``; the remaining dims (heading, ...) use
    ``heading_length_scale`` (``None`` -> reuse ``length_scale``; ``<= 0`` ->
    keep i.i.d. white noise for those dims).
    """
    B, T, D = sample.shape
    device, dtype = sample.device, sample.dtype
    white = torch.empty((B, T, D), device=device, dtype=dtype).normal_(generator=generator)
    out = white.clone()
    if heading_length_scale is None:
        heading_length_scale = length_scale
    n_pos = max(0, min(int(n_position_dims), D))
    # Position channels (x, y): shared correlation length -> a smooth path.
    if n_pos > 0:
        L_pos = _smooth_correlation_cholesky(
            T, length_scale, anchor=anchor, eps=eps, device=device, dtype=dtype,
        )
        if L_pos is not None:
            out[:, :, :n_pos] = torch.einsum("ij,bjd->bid", L_pos, white[:, :, :n_pos])
    # Remaining channels (heading, ...): own correlation length (may be i.i.d.).
    if D > n_pos:
        L_head = _smooth_correlation_cholesky(
            T, heading_length_scale, anchor=anchor, eps=eps, device=device, dtype=dtype,
        )
        if L_head is not None:
            out[:, :, n_pos:] = torch.einsum("ij,bjd->bid", L_head, white[:, :, n_pos:])
    return out
