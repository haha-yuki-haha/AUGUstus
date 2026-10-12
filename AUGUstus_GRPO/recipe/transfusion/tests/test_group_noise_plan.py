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

"""Unit tests for the NESTED per-group ``rl_noise_level`` form.

``rl_noise_level: [0.05, 0.05, 0.02]`` stays the flat per-DIMENSION noise (x,
y, heading) used by EVERY sample of every group (unchanged legacy behaviour).
``rl_noise_level: [[0.05, 0.05, 0.02], [0.01, 0.01, 0.01]]`` mixes different
exploration strengths WITHIN one ``rollout.n`` group: the G samples are split
into L contiguous chunks — every non-final inner list covers ``ceil(G / L)``
samples and the LAST inner list gets the remainder (fewer when G is not
divisible). The plan tensor (per-sample noise ``[B, 1, D]``) sails through the
rollout record and is re-attached as the training-time re-score noise so the
Flow-GRPO / JiT-GRPO importance ratio uses EACH SAMPLE's own std.

Run::

    conda activate augustus_grpo
    cd AUGUstus_GRPO
    python -m pytest recipe/transfusion/tests/test_group_noise_plan.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from verl import DataProto  # noqa: E402

from recipe.transfusion.algorithms.fm_sde_logprob import (  # noqa: E402
    _diffusion_std_dev_t,
    _sde_kernel_coefficients,
    build_group_noise_plan,
    fm_sde_step_with_logprob,
    group_noise_slot_mapping,
    is_group_noise_plan,
)
from recipe.transfusion.loss import collate_flow_grpo_records  # noqa: E402
from recipe.transfusion.rollout import TransfusionRollout  # noqa: E402


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_slot_mapping_divisible():
    assert group_noise_slot_mapping(8, 2) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert group_noise_slot_mapping(8, 4) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_slot_mapping_indivisible_last_list_smaller():
    # G=8, L=3 -> 3 + 3 + 2 (the LAST list gets the remainder).
    assert group_noise_slot_mapping(8, 3) == [0, 0, 0, 1, 1, 1, 2, 2]
    assert group_noise_slot_mapping(10, 3) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]


def test_slot_mapping_more_lists_than_samples():
    # G=2, L=4 -> lists #0/#1 get one sample each; #2/#3 are unused.
    assert group_noise_slot_mapping(2, 4) == [0, 1]
    assert group_noise_slot_mapping(1, 3) == [0]


def test_is_group_noise_plan():
    assert is_group_noise_plan([[0.05, 0.05, 0.02], [0.01, 0.01, 0.01]])
    assert is_group_noise_plan([[0.05, 0.05, 0.02]])
    assert not is_group_noise_plan([0.05, 0.05, 0.02])
    assert not is_group_noise_plan(0.7)
    assert not is_group_noise_plan(torch.zeros(3))


def test_build_group_noise_plan_values():
    plan = build_group_noise_plan(
        [[0.05, 0.05, 0.02], [0.01, 0.01, 0.01]],
        num_rows=16, group_size=8, action_dim=3,
    )
    assert plan.shape == (16, 1, 3)
    exp_hi = pytest.approx([0.05, 0.05, 0.02], abs=1e-7)
    exp_lo = pytest.approx([0.01, 0.01, 0.01], abs=1e-7)
    for row in (0, 1, 2, 3, 8, 9, 10, 11):  # first chunk of EACH group
        assert plan[row, 0].tolist() == pytest.approx(exp_hi, abs=1e-7)
    for row in (4, 5, 6, 7, 12, 13, 14, 15):  # second chunk of EACH group
        assert plan[row, 0].tolist() == pytest.approx(exp_lo, abs=1e-7)


def test_build_group_noise_plan_three_lists():
    plan = build_group_noise_plan(
        [[0.3], [0.2], [0.1]],
        num_rows=8, group_size=8, action_dim=3,
    )
    assert plan.shape == (8, 1, 3)
    vals = plan[:, 0, 0].tolist()
    assert vals == pytest.approx([0.3] * 3 + [0.2] * 3 + [0.1] * 2, abs=1e-7)


def test_build_group_noise_plan_dim_mismatch_raises():
    with pytest.raises(ValueError, match="has 2 entries"):
        build_group_noise_plan([[0.05, 0.05]], num_rows=8, group_size=8, action_dim=3)


def test_build_group_noise_plan_scalar_inners():
    # Scalars inside the nested form broadcast over all D dims.
    plan = build_group_noise_plan([[0.3], 0.1], num_rows=8, group_size=8, action_dim=3)
    assert plan.shape == (8, 1, 3)
    assert plan[0, 0].tolist() == pytest.approx([0.3, 0.3, 0.3], abs=1e-7)
    assert plan[7, 0].tolist() == pytest.approx([0.1, 0.1, 0.1], abs=1e-7)


# ---------------------------------------------------------------------------
# rollout wiring (parse + plan computation + record attach)
# ---------------------------------------------------------------------------

class _FakeAdapter:
    action_dim = 3


class _FakePolicy:
    """Policy stub recording the noise_level it was called with.

    Mimics the real ``sample_grpo`` contract: it passes the resolved
    ``noise_level`` (scalar / [D] list / per-sample plan tensor) through into
    the per-step SDE record, exactly like
    ``TransfusionPolicy.sample_grpo -> resolve_noise_level`` (tensor
    passthrough).
    """

    processor = None
    adapter = _FakeAdapter()
    model = None

    def __init__(self):
        self.seen_noise_level = []

    def sample_grpo(self, transfusion_inputs, *, num_steps=None, noise_level=0.7,
                    noise_schedule="flow_grpo", generate_cot=True, **kwargs):
        import torch as _t

        if not _t.is_tensor(noise_level):
            resolved = _t.as_tensor(noise_level, dtype=_t.float32) if isinstance(noise_level, (list, tuple)) else float(noise_level)
        else:
            resolved = noise_level.detach().float().cpu()
        self.seen_noise_level.append(resolved)
        b = len(transfusion_inputs)
        S, T, D = int(num_steps or 4), 8, 3
        record = {
            "x_in": _t.zeros(S, b, T, D), "next_sample": _t.zeros(S, b, T, D),
            "sigma": _t.zeros(S, b), "sigma_prev": _t.zeros(S, b),
            "log_probs": _t.zeros(S, b), "x0_old": _t.zeros(S, b, T, D),
            "noise_level": resolved, "noise_schedule": noise_schedule, "sigma_max": 1.0,
        }
        return {
            "trajectories": np.zeros((b, T, D), dtype=np.float32),
            "x0_normed": _t.zeros(b, T, D),
            "record": record,
            "cot_texts": [""] * b,
            "cot_ids": [[1, 2, 3]] * b,
        }


def _gen_batch(rows: int) -> DataProto:
    return DataProto(
        non_tensor_batch={
            "transfusion_inputs": np.array([{"dummy": i} for i in range(rows)], dtype=object),
        },
        meta_info={},
    )


def _make_rollout(policy, n: int = 8, **kwargs):
    import types

    config = types.SimpleNamespace(n=n)
    return TransfusionRollout(config=config, model_config=None, device_mesh=None,
                              policy=policy, **kwargs)


def test_parse_nested_env_string():
    os.environ["TRANSFUSION_RL_NOISE_LEVEL"] = "[[0.05, 0.05, 0.02], [0.01, 0.01, 0.01]]"
    try:
        rollout = _make_rollout(_FakePolicy(), rl_noise_level=None)
    finally:
        del os.environ["TRANSFUSION_RL_NOISE_LEVEL"]
    assert rollout._rl_noise_level == [[0.05, 0.05, 0.02], [0.01, 0.01, 0.01]]


def test_parse_nested_direct_list():
    rollout = _make_rollout(_FakePolicy(), rl_noise_level=[[0.05, 0.05, 0.02], [0.01]])
    assert rollout._rl_noise_level == [[0.05, 0.05, 0.02], [0.01]]


def test_nested_noise_reaches_policy_as_per_sample_tensor():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, n=8, rl_num_steps=2,
                            rl_noise_level=[[0.05, 0.05, 0.02], [0.01, 0.01, 0.01]])
    packed = rollout.generate_sequences(_gen_batch(16))
    nl = policy.seen_noise_level[0]
    assert torch.is_tensor(nl) and nl.shape == (16, 1, 3)
    assert nl[0, 0].tolist() == pytest.approx([0.05, 0.05, 0.02], abs=1e-7)
    assert nl[4, 0].tolist() == pytest.approx([0.01, 0.01, 0.01], abs=1e-7)
    assert nl[8, 0].tolist() == pytest.approx([0.05, 0.05, 0.02], abs=1e-7)

    recs = list(packed.non_tensor_batch["flow_grpo_record"])
    assert len(recs) == 16
    # Each sample's record carries ITS OWN noise row (per-sample re-score fidelity).
    assert np.allclose(recs[0]["noise_level"], [[0.05, 0.05, 0.02]], atol=1e-7)
    assert np.allclose(recs[4]["noise_level"], [[0.01, 0.01, 0.01]], atol=1e-7)
    assert np.allclose(recs[8]["noise_level"], [[0.05, 0.05, 0.02]], atol=1e-7)

    # The training-side collate re-stacks them into the [B, 1, 3] plan tensor.
    out = collate_flow_grpo_records(recs)
    assert out["noise_level"].shape == (16, 1, 3)
    assert torch.allclose(out["noise_level"][4], torch.tensor([[0.01, 0.01, 0.01]]))


def test_flat_list_keeps_legacy_batch_wide_behaviour():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, n=8, rl_num_steps=2, rl_noise_level=[0.05, 0.05, 0.02])
    packed = rollout.generate_sequences(_gen_batch(16))
    nl = policy.seen_noise_level[0]
    # Flat per-dim -> unchanged: the batch gets ONE shared [3] noise tensor.
    assert torch.is_tensor(nl) and nl.shape == (3,)
    recs = list(packed.non_tensor_batch["flow_grpo_record"])
    assert np.allclose(recs[0]["noise_level"], [0.05, 0.05, 0.02], atol=1e-7)
    out = collate_flow_grpo_records(recs)
    # All samples share -> legacy singular [3] form (NOT stacked).
    assert out["noise_level"].shape == (3,)


def test_scalar_noise_keeps_legacy_scalar_behaviour():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, n=8, rl_num_steps=2, rl_noise_level=0.7)
    rollout.generate_sequences(_gen_batch(16))
    assert policy.seen_noise_level[0] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# SDE-kernel broadcasting with a per-sample [B, T?, D] noise tensor
# ---------------------------------------------------------------------------

def test_per_sample_noise_broadcasts_in_std_formula():
    B, T, D = 4, 8, 3
    sigma_b = torch.full((B, 1, 1), 0.5)
    a = torch.zeros(B, 1, D)
    a[1] = 1.0
    g = _diffusion_std_dev_t(sigma_b, schedule="dancegrpo", noise_level=a, sigma_max=0.99, eps=1e-6)
    assert g.shape == (B, 1, D)
    assert g[0].abs().max() < 1e-7 and (g[1] - 1.0).abs().max() < 1e-7

    a_td = torch.zeros(B, T, D)
    g = _diffusion_std_dev_t(sigma_b, schedule="dancegrpo", noise_level=a_td, sigma_max=0.99, eps=1e-6)
    assert g.shape == (B, T, D)


def test_sde_kernel_coefficients_per_sample_std():
    B, T, D = 4, 2, 3
    sample = torch.zeros(B, T, D)
    sigma = torch.full((B,), 0.75)
    sigma_prev = torch.full((B,), 0.5)  # dt = -0.25 -> sqrt(-dt) = 0.5
    a = torch.zeros(B, T, D)
    a[1] = 2.0   # row 1 explores with eta=2
    a[2] = 1.0   # row 2 with eta=1
    _, _, std_dev_t, std = _sde_kernel_coefficients(
        sample, sigma, sigma_prev, noise_level=a, sigma_max=1.0, eps=1e-6, schedule="dancegrpo",
    )
    assert std.shape == (B, T, D)
    # dancegrpo: g = eta (constant) -> std = eta * sqrt(-dt) = 0.5 * eta per row.
    assert abs(float(std[1, 0, 0]) - 1.0) < 1e-6
    assert abs(float(std[2, 0, 0]) - 0.5) < 1e-6
    # a=0 rows collapse to the eps floor (deterministic up to eps).
    assert float(std[0].max()) < 1e-3 and float(std[0].min()) > 0.0


def test_lambda_t_step_with_per_sample_noise_shapes():
    torch.manual_seed(0)
    B, T, D = 6, 8, 3
    x = torch.randn(B, T, D)
    v = torch.zeros(B, T, D)
    sigma = torch.full((B,), 1.0)
    sigma_prev = torch.zeros(B)
    a = torch.full((B, 1, D), 0.01)
    a[3:] = 0.9  # half the group explores much harder
    prev, log_prob, mean, std = fm_sde_step_with_logprob(
        v, x, sigma, sigma_prev, noise_level=a, schedule="lambda_t", sigma_max=1.0,
    )
    assert prev.shape == (B, T, D)
    assert log_prob.shape == (B,)
    # per-sample std rows respect the per-sample noise multiplier.
    assert float(std[0].max()) < 0.05 < float(std[4].max()) < 2.0
    assert abs(float(std[0].mean()) - 0.01) < 1e-4
    assert abs(float(std[4].mean()) - 0.9) < 1e-4
    # re-score path (prev_sample given) also runs and is finite.
    prev2, logp2, _, _ = fm_sde_step_with_logprob(
        v, x, sigma, sigma_prev, noise_level=a, schedule="lambda_t",
        sigma_max=1.0, prev_sample=prev,
    )
    assert torch.isfinite(logp2).all() and torch.allclose(prev2, prev)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
