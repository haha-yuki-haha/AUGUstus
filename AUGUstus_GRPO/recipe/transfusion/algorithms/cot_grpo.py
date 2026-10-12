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

"""Token-level PPO/GRPO surrogate for the Chain-of-Thought half of the policy.

The Transfusion action is a continuous diffusion trajectory whose importance
ratio is the closed-form JiT-GRPO Gaussian log-ratio
(:mod:`recipe.transfusion.algorithms.jit_grpo`); the CoT is an ordinary
autoregressive token sequence, so its ratio is the plain **per-token
probability ratio** ``exp(logp_new - logp_old)``. This module implements that
token-level clipped surrogate (plus an entropy bonus and optional KL penalty),
matching the reduction conventions of :func:`jit_grpo.jit_grpo_policy_loss` so
the two halves compose into one GRPO objective sharing the same group-relative
advantage.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def cot_token_grpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    entropy: Optional[torch.Tensor] = None,
    clip_range: float = 0.2,
    clip_range_high: Optional[float] = None,
    entropy_coef: float = 0.0,
    kl_coef: float = 0.0,
    ref_logp: Optional[torch.Tensor] = None,
    kl_estimator: str = "k1",
    max_log_ratio: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PPO-clipped token-level GRPO loss for the CoT response tokens.

    Args:
        new_logp:   ``[B, R]`` current-policy per-token log-prob (grad).
        old_logp:   ``[B, R]`` behaviour-policy per-token log-prob (no grad).
        advantages: ``[B]`` (broadcast over tokens) or ``[B, R]`` group-relative
            advantage.
        mask:       ``[B, R]`` 1/0 mask of valid CoT tokens (right-padding = 0).
        entropy:    optional ``[B, R]`` per-token entropy for the entropy bonus.
        clip_range: lower PPO clip ``1 - clip_range``.
        clip_range_high: optional asymmetric upper clip ``1 + clip_range_high``.
        entropy_coef: weight of the (maximise-entropy) bonus.
        kl_coef:    weight of the ``KL(pi_new || pi_ref)`` penalty.
        ref_logp:   optional ``[B, R]`` reference-policy per-token log-prob.
        kl_estimator: estimator for the *diagnostic* new-vs-old approximate KL
            (``cot_grpo/approx_kl``). ``"k1"`` — the PPO-convention
            ``mean(old - new)`` (cheap, can go NEGATIVE under finite-sample
            fluctuation; that is not a bug, just estimator noise). ``"k3"`` —
            verl's ``(logr.neg()).exp() - 1 + logr`` with ``logr = new - old``,
            i.e. ``exp(gap) - 1 - gap`` with ``gap = old - new``: pointwise
            NON-negative and lower-variance. Both are metrics only — the loss
            itself is identical. Anything else raises ``ValueError``.

    Returns:
        ``(loss, metrics)``.
    """
    new_logp = new_logp.float()
    old_logp = old_logp.float().to(new_logp.device)
    mask = mask.float().to(new_logp.device)
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    advantages = advantages.float().to(new_logp.device)
    if advantages.shape[1] == 1 and new_logp.shape[1] > 1:
        advantages = advantages.expand(-1, new_logp.shape[1])

    denom = mask.sum().clamp(min=1.0)
    hi = clip_range if clip_range_high is None else clip_range_high

    # Padding positions carry meaningless logits; zero them BEFORE any arithmetic
    # so a stray inf/nan there can never poison the (masked-out) loss.
    keep = mask > 0
    # Non-finite old/new log-probs (NaN/inf weights upstream, or a corrupted
    # behaviour-policy record) must be dropped from the loss entirely: a single
    # NaN token would NaN the whole gradient (clip_grad_norm_ does NOT guard
    # NaN) and the run dies in the next rollout with a CUDA device-side assert.
    finite = torch.isfinite(new_logp) & torch.isfinite(old_logp)
    n_nonfinite = int((keep & ~finite).sum().item())
    keep = keep & finite
    new_logp = torch.where(keep, new_logp, torch.zeros_like(new_logp))
    old_logp = torch.where(keep, old_logp, torch.zeros_like(old_logp))

    log_ratio = new_logp - old_logp
    # Hard saturation guard on the importance log-ratio. On-policy with one
    # optimizer step per rollout, legitimate ratios live in ~[0.8, 1.2]
    # (|log_ratio| <= ~0.3); PPO's torch.maximum(unclipped, clipped) does NOT
    # bound a too-large ratio on negative-advantage tokens, so a broken
    # behaviour-policy record (old_logp catastrophically wrong, e.g. from a
    # KV-cache decode vs teacher-forced recompute mismatch on a sharpened
    # policy) detonates the loss (observed: ratio 1e31 -> loss 5.6e10 ->
    # NaN grads). Clamping saturates such tokens instead; the clamp also
    # zeroes their gradient, acting like a per-token mask.
    if max_log_ratio is not None and max_log_ratio > 0:
        n_clamped = int(((log_ratio.abs() > float(max_log_ratio)) & keep).sum().item())
        log_ratio = torch.clamp(log_ratio, -float(max_log_ratio), float(max_log_ratio))
    else:
        n_clamped = 0
    ratio = torch.exp(log_ratio)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + hi)
    per_token = torch.maximum(unclipped, clipped)
    policy_loss = torch.where(keep, per_token, torch.zeros_like(per_token)).sum() / denom
    clipfrac = (((unclipped < clipped) & keep).float()).sum() / denom

    loss = policy_loss
    entropy_val = 0.0
    if entropy is not None:
        entropy = torch.where(keep, entropy.float(), torch.zeros_like(entropy.float()))
        ent = entropy.sum() / denom
        entropy_val = float(ent.detach())
        if entropy_coef != 0.0:
            loss = loss - entropy_coef * ent

    kl_val = 0.0
    if ref_logp is not None and kl_coef != 0.0:
        # k3 estimator of KL(pi_new || pi_ref): E[r - 1 - log r], r = pi_ref/pi_new.
        # Clamped exactly like verl's `low_var_kl` (core_algos.kl_penalty): a single
        # rare sampled token with a large |logp gap| otherwise explodes exp(gap)
        # and lets one token dominate the whole loss (observed: cot_ref_kl 0.004
        # -> 74 within 3 steps on the first RL run of this recipe).
        ref_logp = ref_logp.float().to(new_logp.device)
        ref_logp = torch.where(keep, ref_logp, new_logp)  # -> 0 contribution at pad
        kl_gap = torch.clamp(ref_logp - new_logp, min=-20.0, max=20.0)
        kl_tok = (torch.exp(kl_gap) - 1.0) - kl_gap
        kl_tok = torch.clamp(kl_tok, min=-10.0, max=10.0)
        kl_term = torch.where(keep, kl_tok, torch.zeros_like(kl_tok)).sum() / denom
        loss = loss + kl_coef * kl_term
        kl_val = float(kl_term.detach())

    with torch.no_grad():
        if kl_estimator == "k1":
            approx_kl = (torch.where(keep, old_logp - new_logp, torch.zeros_like(new_logp)).sum() / denom)
        elif kl_estimator == "k3":
            # Pointwise non-negative k3 estimator, clamped exactly like the
            # ref-KL above: a single rare token with a large |logp gap| must not
            # let one exp(gap) dominate the diagnostic.
            gap = torch.clamp(old_logp - new_logp, min=-20.0, max=20.0)
            k3_tok = torch.clamp((torch.exp(gap) - 1.0) - gap, min=-10.0, max=10.0)
            approx_kl = torch.where(keep, k3_tok, torch.zeros_like(k3_tok)).sum() / denom
        else:
            raise ValueError(
                f"cot_token_grpo_loss: unknown kl_estimator {kl_estimator!r} (expected 'k1' or 'k3')"
            )

    metrics = {
        "cot_grpo/policy_loss": float(policy_loss.detach()),
        "cot_grpo/loss": float(loss.detach()),
        "cot_grpo/ratio_mean": float(((ratio * mask).sum() / denom).detach()),
        "cot_grpo/clipfrac": float(clipfrac.detach()),
        "cot_grpo/approx_kl": float(approx_kl.detach()),
        "cot_grpo/entropy": entropy_val,
        "cot_grpo/kl": kl_val,
        "cot_grpo/num_tokens": float(denom.detach()),
        # Guards: fraction of valid tokens whose raw |log_ratio| saturated the
        # hard clamp, and the count of valid tokens dropped for non-finite
        # log-probs. Non-zero clamped_frac means old/new disagree beyond any
        # legitimate on-policy range (record-vs-recompute mismatch).
        "cot_grpo/clamped_frac": float(n_clamped) / max(float(denom.detach()), 1.0),
        "cot_grpo/nonfinite_masked": float(n_nonfinite),
    }
    return loss, metrics


__all__ = ["cot_token_grpo_loss"]
