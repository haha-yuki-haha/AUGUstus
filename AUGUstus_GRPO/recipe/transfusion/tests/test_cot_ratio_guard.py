# Copyright 2024 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for CoT importance-ratio guards.

Checks saturation of log(new/old) before PPO clipping and masking of
non-finite log-probability tokens, including gradients on saturated tokens.

Run::

    python -m pytest recipe/transfusion/tests/test_cot_ratio_guard.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recipe.transfusion.algorithms.cot_grpo import cot_token_grpo_loss  # noqa: E402

torch.manual_seed(0)


def _toy_inputs(b=2, r=8, bad_col=3):
    """new/old logp tensors whose ratio is ~1 everywhere except one column."""
    new_logp = torch.full((b, r), -1.0)
    old_logp = torch.full((b, r), -1.0)
    mask = torch.ones(b, r)
    adv = torch.tensor([-1.0, -1.0])  # NEGATIVE advantage for every sample
    # broken record: old says the token was ~-70 nats, recompute says ~0
    old_logp[:, bad_col] = -70.0
    return new_logp, old_logp, mask, adv, bad_col


def test_exploding_ratio_is_saturated_by_max_log_ratio():
    """Large ratios on negative-advantage tokens must produce a bounded loss."""
    new_logp, old_logp, mask, adv, bad_col = _toy_inputs()

    # guard OFF (legacy behaviour): loss must be astronomically large
    loss_raw, m_raw = cot_token_grpo_loss(
        new_logp, old_logp, adv, mask, clip_range=0.2, max_log_ratio=None,
    )
    assert loss_raw.item() > 1e25, loss_raw.item()

    # guard ON (production default): loss bounded by exp(5) * |adv| / tokens
    loss, m = cot_token_grpo_loss(
        new_logp, old_logp, adv, mask, clip_range=0.2, max_log_ratio=5.0,
    )
    assert torch.isfinite(loss), loss
    assert loss.item() < 1e3, loss.item()
    # exactly one token per row was clamped
    assert m["cot_grpo/clamped_frac"] > 0.0
    assert m["cot_grpo/nonfinite_masked"] == 0.0


def test_clamped_token_has_zero_gradient():
    """The clamp acts like a mask: no gradient flows through saturated tokens."""
    new_logp = torch.zeros(1, 4, requires_grad=True)
    old_logp = torch.tensor([[0.0, -70.0, 0.0, 0.0]])
    mask = torch.ones(1, 4)
    adv = torch.tensor([-1.0])
    loss, _ = cot_token_grpo_loss(
        new_logp, old_logp, adv, mask, clip_range=0.2, max_log_ratio=5.0,
    )
    loss.backward()
    # the clamped column (index 1) must receive NO gradient
    assert new_logp.grad is not None
    assert new_logp.grad[0, 1].abs().item() == 0.0
    assert new_logp.grad[0, 0].abs().item() > 0.0


def test_nonfinite_logp_tokens_are_dropped():
    """NaN/inf old_logp tokens must be masked out, not NaN the loss."""
    new_logp = torch.tensor([[0.0, float("nan"), 0.0, float("inf")]])
    old_logp = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4)
    adv = torch.tensor([1.0])
    loss, m = cot_token_grpo_loss(
        new_logp, old_logp, adv, mask, clip_range=0.2, max_log_ratio=5.0,
    )
    assert torch.isfinite(loss), loss
    # one NaN + one inf token dropped
    assert m["cot_grpo/nonfinite_masked"] == 2.0


def test_healthy_onpolicy_ratios_unchanged_by_guard():
    """With matched old/new (ratio==1) the guarded loss equals the plain loss."""
    torch.manual_seed(3)
    r = 16
    new_logp = torch.randn(4, r) - 1.0
    old_logp = new_logp.clone()
    mask = torch.ones(4, r)
    adv = torch.tensor([1.0, -0.5, 0.25, -1.0])

    loss_plain, m_plain = cot_token_grpo_loss(
        new_logp, old_logp, adv, mask, clip_range=0.2, max_log_ratio=None,
    )
    loss_guard, m_guard = cot_token_grpo_loss(
        new_logp, old_logp, adv, mask, clip_range=0.2, max_log_ratio=5.0,
    )
    assert torch.allclose(loss_plain, loss_guard)
    assert m_guard["cot_grpo/clamped_frac"] == 0.0
    assert m_guard["cot_grpo/ratio_mean"] == m_plain["cot_grpo/ratio_mean"]
