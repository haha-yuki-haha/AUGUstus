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

"""Unit tests for the two-frame Extended Comfort (EC) wiring:

* ``_build_prev_map_from_meta`` — the official ``infer_start_adjacent_mapping``
  rule (same log, ORIGINAL frames, <=0.55s gap), pure parent-side;
* ``_inject_ec_score`` / ``per_frame_components`` — EC-injected per-frame
  EPDMS maths, identical to the official ``compute_final_scores``;
* ``epdms_score_per_frame(with_gt_prev_ec=True)`` — training reward path adds
  EC for chained frames and degrades to one-stage when the map is missing;
* ``aggregate_shard_results`` — the driver-side OFFICIAL aggregation over all
  shards' buffered rows rewrites epdms/epdms_reward to the final numbers and
  surfaces ``epdms_EC`` (skipna mean over chained frames).

Run::

    cd AUGUstus_GRPO
    python -m pytest recipe/transfusion/tests/test_epdms_ec.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recipe.transfusion import epdms_eval  # noqa: E402
from recipe.transfusion import validation as valmod  # noqa: E402


# ---------------------------------------------------------------------------
# prev-token mapping (official infer_start_adjacent_mapping rule)
# ---------------------------------------------------------------------------

def test_build_prev_map_chains_adjacent_original_frames():
    metas = [
        ("tokC", "log1", 1.0, "ORIGINAL"),
        ("tokA", "log1", 0.0, "ORIGINAL"),
        ("tokB", "log1", 0.5, "ORIGINAL"),
        ("tokS", "log1", 1.5, "SYNTHETIC"),     # synthetic frames never chain
        ("tokX", "log2", 0.0, "ORIGINAL"),       # first frame of a log: no prev
        ("tokZ", "log1", 3.0, "ORIGINAL"),       # gap 1.0s > 0.55s: no prev
        ("tokB2", "log1", 1.5001, "ORIGINAL"),   # rounds back onto the chain
        ("bad", None, None, None),               # unloadable: excluded
    ]
    m = epdms_eval._build_prev_map_from_meta(metas)
    assert m == {
        "tokB": "tokA",
        "tokC": "tokB",
        "tokB2": "tokC",   # 0.5001s <= 0.55s still adjacent
    }
    assert "tokZ" not in m and "tokA" not in m and "tokX" not in m
    assert "tokS" not in m and "tokS" not in m.values()


def test_build_prev_map_gap_threshold_inclusive():
    metas = [
        ("a", "log", 0.0, "ORIGINAL"),
        ("b", "log", 0.55, "ORIGINAL"),   # exactly the official threshold
        ("c", "log", 1.11, "ORIGINAL"),   # 0.56s -> no chain
    ]
    m = epdms_eval._build_prev_map_from_meta(metas)
    assert m == {"b": "a"}


# ---------------------------------------------------------------------------
# EC injection maths (same as the official compute_final_scores)
# ---------------------------------------------------------------------------

def _row(*, ec=None, wm=(0.8, 1.0, 1.0, 1.0, 0.0), wa=(5.0, 5.0, 2.0, 2.0, 2.0), mult=1.0):
    row = {
        "token": "t",
        "valid": True,
        "multiplicative_metrics_prod": mult,
        "weighted_metrics": list(wm),
        "weighted_metrics_array": list(wa),
        "pdm_score": mult * (5 * wm[0] + 5 * wm[1] + 2 * wm[2] + 2 * wm[3]) / 14.0,
        "no_at_fault_collisions": 1.0,
        "drivable_area_compliance": 1.0,
        "driving_direction_compliance": 1.0,
        "traffic_light_compliance": 1.0,
        "ego_progress": wm[0],
        "time_to_collision_within_bound": wm[1],
        "lane_keeping": wm[2],
        "history_comfort": wm[3],
    }
    if ec is not None:
        row["EC"] = ec
    return row


def test_inject_ec_score_matches_official_weighting():
    row = _row(ec=1.0)
    # official chained-frame score: mult * (5*EP + 5*TTC + 2*LK + 2*HC + 2*EC)/16
    expect = (5 * 0.8 + 5 * 1.0 + 2 * 1.0 + 2 * 1.0 + 2 * 1.0) / 16.0
    assert epdms_eval._inject_ec_score(row) == pytest.approx(expect, abs=1e-12)


def test_inject_ec_score_zero_weight_slot_gets_official_weight():
    row = _row(ec=0.5, wa=(5.0, 5.0, 2.0, 2.0, 0.0))
    expect = (5 * 0.8 + 5 * 1.0 + 2 * 1.0 + 2 * 1.0 + 2.0 * 0.5) / 16.0
    assert epdms_eval._inject_ec_score(row) == pytest.approx(expect, abs=1e-12)


def test_inject_ec_score_nan_returns_none():
    assert epdms_eval._inject_ec_score(_row(ec=float("nan"))) is None
    assert epdms_eval._inject_ec_score({"token": "x"}) is None


def test_per_frame_components_one_stage_and_ec_injected():
    row = _row(ec=None)
    d = epdms_eval.per_frame_components(row)
    assert "EC" not in d
    assert d["EPDMS"] == pytest.approx(row["pdm_score"], abs=1e-12)
    assert d["NC"] == 1.0 and d["LK"] == 1.0

    row_ec = _row(ec=0.0)  # chain exists but EC fails -> EPDMS drops
    d_ec = epdms_eval.per_frame_components(row_ec)
    assert d_ec["EC"] == 0.0
    expect = (5 * 0.8 + 5 * 1.0 + 2 * 1.0 + 2 * 1.0 + 2 * 0.0) / 16.0
    assert d_ec["EPDMS"] == pytest.approx(expect, abs=1e-12)
    assert d_ec["EPDMS"] < d["EPDMS"]


# ---------------------------------------------------------------------------
# epdms_score_per_frame(with_gt_prev_ec=True) — pool + map mocked
# ---------------------------------------------------------------------------


class _FakePool:
    def __init__(self, rows):
        self._rows = rows
        self.jobs = None

    def map(self, fn, jobs, chunksize=1):
        self.jobs = jobs
        return list(self._rows)[:len(jobs)]


def test_per_frame_with_gt_prev_ec(monkeypatch):
    row_plain = _row()
    row_chained = _row(ec=1.0)
    pool = _FakePool([dict(row_plain), dict(row_chained)])
    monkeypatch.setattr(epdms_eval, "epdms_available", lambda *a, **k: True)
    monkeypatch.setattr(epdms_eval, "_get_pool", lambda *a, **k: pool)
    monkeypatch.setattr(
        epdms_eval, "_ensure_prev_map", lambda *a, **k: {"tokA": "tokPrev"}
    )
    trajs = np.zeros((2, 8, 3), dtype=np.float64)
    out = epdms_eval.epdms_score_per_frame(
        trajs, ["tokX", "tokA"], "/cache", with_gt_prev_ec=True
    )
    # the map only chained tokA -> tokPrev, and only that job carries prev_token
    assert "prev_token" not in pool.jobs[0]
    assert pool.jobs[1]["prev_token"] == "tokPrev"
    assert out[0]["EPDMS"] == pytest.approx(row_plain["pdm_score"], abs=1e-12)
    assert "EC" not in out[0]
    expect = (5 * 0.8 + 5 * 1.0 + 2 * 1.0 + 2 * 1.0 + 2 * 1.0) / 16.0
    assert out[1]["EPDMS"] == pytest.approx(expect, abs=1e-12)
    assert out[1]["EC"] == 1.0


def test_per_frame_map_failure_degrades_to_one_stage(monkeypatch):
    pool = _FakePool([dict(_row())])
    monkeypatch.setattr(epdms_eval, "epdms_available", lambda *a, **k: True)
    monkeypatch.setattr(epdms_eval, "_get_pool", lambda *a, **k: pool)
    def _boom(*a, **k):
        raise RuntimeError("no map")
    monkeypatch.setattr(epdms_eval, "_ensure_prev_map", _boom)
    trajs = np.zeros((1, 8, 3), dtype=np.float64)
    out = epdms_eval.epdms_score_per_frame(
        trajs, ["tokX"], "/cache", with_gt_prev_ec=True
    )
    assert "prev_token" not in pool.jobs[0]          # EC machinery skipped
    assert out[0]["EPDMS"] == pytest.approx(_row()["pdm_score"], abs=1e-12)


def test_per_frame_default_is_one_stage(monkeypatch):
    pool = _FakePool([dict(_row())])
    monkeypatch.setattr(epdms_eval, "epdms_available", lambda *a, **k: True)
    monkeypatch.setattr(epdms_eval, "_get_pool", lambda *a, **k: pool)
    def _forbidden(*a, **k):  # must not even be attempted without the flag
        raise AssertionError("prev map must not be built when off")
    monkeypatch.setattr(epdms_eval, "_ensure_prev_map", _forbidden)
    out = epdms_eval.epdms_score_per_frame(
        np.zeros((1, 8, 3)), ["tokX"], "/cache"
    )
    assert out[0]["EPDMS"] == pytest.approx(_row()["pdm_score"], abs=1e-12)


# ---------------------------------------------------------------------------
# aggregate_shard_results — driver-side official two-frame EC aggregation
# ---------------------------------------------------------------------------


def _shard(epdms_sum, epdms_n, rows, cache="/v2/test"):
    return {
        "rank": 0,
        "count": epdms_n,
        "num_batches": 1,
        "seconds": {},
        "total_seconds": 1.0,
        "metric_sums": {"epdms": epdms_sum, "epdms_reward": epdms_sum, "pdms_reward": 1.6},
        "metric_counts": {"epdms": epdms_n, "epdms_reward": epdms_n, "pdms_reward": 2},
        "pdms_counts": {"candidate": 2, "scored": 2},
        "epdms_rows": rows,
        "epdms_metric_cache_v2": cache,
    }


def test_aggregate_rewrites_epdms_with_official_chaining(monkeypatch):
    rows_a = [{"token": "f0", "valid": True}]
    rows_b = [{"token": "f1", "valid": True}]
    shards = [
        _shard(0.6, 1, rows_a),
        _shard(0.9, 1, rows_b),
    ]
    seen = {}

    def fake_finalize(rows, cache, **kw):
        seen["rows"] = list(rows)
        seen["cache"] = cache
        return {
            "per_frame": [
                # unchained: final == one-stage -> delta 0, EC NaN (skipped)
                {"token": "f0", "EPDMS": 0.6, "EPDMS_one_stage": 0.6, "EC": float("nan")},
                # chained: EC injected into the final score
                {"token": "f1", "EPDMS": 0.85, "EPDMS_one_stage": 0.9, "EC": 1.0},
            ],
            "summary": {},
        }

    monkeypatch.setattr(epdms_eval, "epdms_finalize_rows", fake_finalize)
    metrics = valmod.aggregate_shard_results(shards, data_source="navsim_transfusion")
    # rows from BOTH shards reached one global aggregation call (cross-rank chains)
    assert seen["cache"] == "/v2/test"
    assert len(seen["rows"]) == 2
    # epdms mean = (0.6 + 0.85) / 2 — the OFFICIAL final per-frame scores
    assert metrics["val-aux/navsim_transfusion/epdms/mean"] == pytest.approx(0.725)
    assert metrics["val-aux/navsim_transfusion/epdms_reward/mean"] == pytest.approx(0.725)
    # epdms_EC is the skipna mean over chained frames only
    assert metrics["val-aux/navsim_transfusion/epdms_EC/mean"] == pytest.approx(1.0)


def test_aggregate_keeps_one_stage_when_finalize_fails(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("pool dead")
    monkeypatch.setattr(epdms_eval, "epdms_finalize_rows", _boom)
    metrics = valmod.aggregate_shard_results([_shard(1.5, 2, [{"token": "f"}])],
                                             data_source="navsim_transfusion")
    assert metrics["val-aux/navsim_transfusion/epdms/mean"] == pytest.approx(0.75)
    assert "val-aux/navsim_transfusion/epdms_EC/mean" not in metrics


def test_aggregate_without_rows_is_unchanged():
    shard = {
        "rank": 0, "count": 2, "num_batches": 1, "seconds": {}, "total_seconds": 1.0,
        "metric_sums": {"epdms": 1.4, "epdms_reward": 1.4},
        "metric_counts": {"epdms": 2, "epdms_reward": 2},
        "pdms_counts": {"candidate": 2, "scored": 2},
    }
    metrics = valmod.aggregate_shard_results([shard], data_source="navsim_transfusion")
    assert metrics["val-aux/navsim_transfusion/epdms/mean"] == pytest.approx(0.7)
    assert "val-aux/navsim_transfusion/epdms_EC/mean" not in metrics


def test_ec_flows_to_swanlab_val_key(monkeypatch):
    """epdms_EC reaches the Transfusion-style val/EPDMS_EC key (best_ckpt map)."""
    from recipe.transfusion import best_ckpt as bc

    def fake_finalize(rows, cache, **kw):
        return {
            "per_frame": [{"token": "f1", "EPDMS": 0.8, "EPDMS_one_stage": 0.9, "EC": 0.5}],
            "summary": {},
        }

    monkeypatch.setattr(epdms_eval, "epdms_finalize_rows", fake_finalize)
    metrics = valmod.aggregate_shard_results([_shard(0.9, 1, [{"token": "f1", "valid": True}])],
                                             data_source="navsim_transfusion")
    metrics = bc._add_transfusion_val_metrics(metrics)
    assert metrics["val/EPDMS"] == pytest.approx(0.8)
    assert metrics["val/EPDMS_EC"] == pytest.approx(0.5)
