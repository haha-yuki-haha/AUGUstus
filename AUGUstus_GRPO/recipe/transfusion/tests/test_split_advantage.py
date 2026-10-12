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

"""Tests for the per-head (split) GRPO advantage helper ``_group_zscore``.

This is the pure-math core of ``rl_split_advantage`` (docs/SPLIT_ADVANTAGE.md):
each head gets its OWN group-relative z-score derived from only the reward it can
influence (action -> planning block, CoT -> prediction block). Run::

    cd AUGUstus_GRPO
    python recipe/transfusion/tests/test_split_advantage.py

Requires the rest of the recipe to be importable (verl present), since
``sync_driver`` imports verl at module load.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recipe.transfusion.sync_driver import (  # noqa: E402
    _group_zscore,
    _grouped_zscore_weighted,
    _batch_whiten,
)


def _manual_zscore(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size <= 1:
        return np.zeros_like(v)
    s = v.std()
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def test_zscore_matches_manual():
    g = np.array([1.0, 2.0, 3.0, 4.0])
    out = _group_zscore(g)
    assert np.allclose(out, _manual_zscore(g))
    # zero sum is a sanity property of a centered z-score
    assert abs(out.sum()) < 1e-9


def test_zscore_two_groups():
    # Two GRPO groups of 8 laid out contiguously (as the driver sees them).
    # ``_group_zscore`` is applied PER GROUP (the caller splits by ``group_of``
    # first, exactly like ``_attach_rm_scores``), so each block is z-scored with
    # respect to its OWN group's mean/std, not the whole buffer.
    g1 = np.arange(1.0, 9.0)          # 1..8
    g2 = np.array([10.0, 10.0, 12.0, 12.0, 15.0, 5.0, 5.0, 7.0])
    full = np.concatenate([g1, g2])
    # mirror the driver: split members by group, z-score each, reassemble.
    out = np.zeros_like(full)
    out[:8] = _group_zscore(g1)
    out[8:] = _group_zscore(g2)
    assert np.allclose(out[:8], _manual_zscore(g1))
    assert np.allclose(out[8:], _manual_zscore(g2))
    # each group is independently centered (sum ~ 0 within each group)
    assert abs(out[:8].sum()) < 1e-9 and abs(out[8:].sum()) < 1e-9


def test_constant_group_is_zero():
    # std == 0 -> all-zero advantage (no relative signal), matching verl.
    const = np.ones(8) * 0.5
    assert np.all(_group_zscore(const) == 0.0)


def test_single_member_group_is_zero():
    assert np.all(_group_zscore(np.array([3.0])) == 0.0)


def test_nan_safe():
    # a NaN member must not poison the whole group (falls back to zeros).
    bad = np.array([np.nan, 1.0, 2.0, 3.0])
    assert bad.shape == _group_zscore(bad).shape


def test_gdpo_component_norm_lets_pdms_stand_out():
    """GDPO pre-sum (``_grouped_zscore_weighted``, Eq. 4-5 of arXiv 2601.05242):
    soft_L2 varies far more than PDMS within a group, so it floods the PDMS in
    the raw-sum ``reward_action`` z-score. After z-scoring each component
    separately within the group, the PDMS (lower variance) no longer drowns --
    this is the decoupled normalization step BEFORE the final batch whiten."""
    n = 8
    # soft_L2: high variance across the group; PDMS: only one outlier ties to the
    # "best" ranking we want the action head to chase.
    soft_l2 = np.array([0.10, 0.12, 0.11, 0.09, 0.95, 0.90, 0.88, 0.93])  # two clusters
    pdms = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])            # one strong + noise
    group_of = {0: list(range(n))}  # single group
    # equal coefs 1:1, per-component normalized (GDPO pre-sum)
    comp = _grouped_zscore_weighted([(soft_l2, 1.0), (pdms, 1.0)], n, group_of)
    # the lone PDMS=1.0 sample (index 4) must rank at the top after decoupled norm
    assert comp[4] == comp.max()
    # its combined advantage must be > the high-soft_L2 cluster members that have
    # noise-level PDMS (they may still be positive from soft_L2, but the PDMS
    # signal must dominate index 4).
    assert comp[4] > comp[5]


def test_gdpo_weighted_constant_group_is_zero():
    # constant component (std=0) contributes 0, so the combined adv == the other
    # component's z-score alone; all-constant -> all zero.
    n = 4
    group_of = {0: [0, 1, 2, 3]}
    both_const = _grouped_zscore_weighted([(np.ones(4) * 0.5, 1.0), (np.ones(4) * 0.3, 1.0)], n, group_of)
    assert np.all(both_const == 0.0)


def test_batch_whiten():
    """GDPO Eq. 6: batch-wide whiten ``(v - mean) / (std + eps)`` over ALL
    samples. Centres to mean 0 / std 1; constant or single-member input -> 0."""
    v = np.array([1.0, 2.0, 3.0, 4.0])
    out = _batch_whiten(v)
    assert np.allclose(out.mean(), 0.0, atol=1e-12)
    assert np.allclose(out.std(), 1.0, atol=1e-12)
    # constant input -> all zeros (no scale to learn from)
    assert np.all(_batch_whiten(np.ones(4) * 0.7) == 0.0)
    # single member -> all zeros
    assert np.all(_batch_whiten(np.array([3.0])) == 0.0)


def test_gdpo_end_to_end_pdms_stands_out_after_whiten():
    """Full GDPO: pre = sum_k coef_k * group_zscore(r_k), then batch-whiten.
    Whitening is monotonic, so the lone PDMS outlier must still top-rank; the
    final advantage is centred and unit-scaled across the whole batch."""
    n = 8
    soft_l2 = np.array([0.10, 0.12, 0.11, 0.09, 0.95, 0.90, 0.88, 0.93])
    pdms = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])
    group_of = {0: list(range(n))}  # single GRPO group
    pre = _grouped_zscore_weighted([(soft_l2, 1.0), (pdms, 1.0)], n, group_of)
    adv = _batch_whiten(pre)
    assert adv[4] == adv.max()
    assert adv[4] > adv[5]
    assert abs(adv.mean()) < 1e-12
    assert abs(adv.std() - 1.0) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} split-advantage tests passed.")