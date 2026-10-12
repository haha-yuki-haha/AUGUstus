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

"""Integration tests: official v2 EPDMS gating in ``_attach_rm_scores``.

Verifies the end-to-end wiring (sync driver batch precompute -> extra_info ->
compute_score -> rm_scores / reward-extra metrics) with the heavy scorer pools
mocked out:

* train with ``epdms: 0.0``            -> the v2 EPDMS scorer is NEVER called
  (the v2 caches are never loaded — plain-PDMS runs waste no time);
* train with ``epdms: 1.0``            -> per-frame EPDMS batch-scored against
  the TRAIN v2 cache, the term enters the score AND the metric is logged;
* val with ``val_epdms: false``        -> scorer never called (PDMS only);
* val with ``val_epdms: true``         -> EPDMS batch-scored against the TEST
  v2 cache, logged as a report-only metric (the val score keeps the standard
  composition).
* PDMS is batch-computed in EVERY case.
* ``pdm_ep_weight`` reweights the PDMS term and never leaks into EPDMS.

Run::

    cd AUGUstus_GRPO
    python -m pytest recipe/transfusion/tests/test_epdms_sync_integration.py -q
"""

import os
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from omegaconf import OmegaConf  # noqa: E402

from recipe.transfusion import reward_function as rf  # noqa: E402
from recipe.transfusion import sync_driver as sd  # noqa: E402
from recipe.transfusion import transfusion_reward as tr  # noqa: E402
from recipe.transfusion import epdms_eval  # noqa: E402

_N = 4
_V1_PDMS = {"PDMS": 0.8, "NC": 1.0, "DAC": 1.0, "DDC": 1.0, "EP": 0.6, "TTC": 0.8, "C": 0.5}
_V2_ROW = {"EPDMS": 0.5, "NC": 1.0, "DAC": 1.0, "DDC": 1.0, "TLC": 1.0,
           "EP": 0.5, "TTC": 1.0, "LK": 1.0, "HC": 1.0}
# reweighted v1 PDMS for pdm_ep_weight=10 on _V1_PDMS: 11/17
_REW10 = (10.0 * 0.6 + 5.0 * 0.8 + 2.0 * 0.5) / 17.0


def _make_batch(*, is_validate: bool, v2_stamp: str):
    trajs = [
        np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (8, 1)) * (i + 1)
        for i in range(_N)
    ]
    reward_model = [
        {"ground_truth": {
            "gt_trajectory": [[0.0, 0.0, 0.0]] * 8,
            "gt_cot_text": "",
            "frame_id": f"tok{i}",
            "metric_cache_path": "/v1/cache",
            "metric_cache_v2_path": v2_stamp,
        }}
        for i in range(_N)
    ]
    return SimpleNamespace(
        batch={
            "responses": torch.zeros((_N, 4), dtype=torch.long),
            "response_mask": torch.ones((_N, 4), dtype=torch.long),
        },
        non_tensor_batch={
            "pred_trajectory": np.array(trajs, dtype=object),
            "reward_model": np.array(reward_model, dtype=object),
            "extra_info": np.array([{} for _ in range(_N)], dtype=object),
            "data_source": np.array(["navsim_transfusion"] * _N, dtype=object),
        },
        meta_info={"validate": is_validate, "global_steps": 0},
    )


def _make_manager(weights):
    cfg = OmegaConf.create({
        "reward": {"custom_reward_function": {"reward_kwargs": {
            "weights": weights,
            "metric_cache_train_v2": "/v2/train",
            "metric_cache_test_v2": "/v2/test",
        }}},
        "actor_rollout_ref": {"rollout": {"n": 2}},
    })
    mgr = sd._SyncRolloutManager(SimpleNamespace(config=cfg, tokenizer=None))
    mgr._reward_fn_loaded = True
    rk = OmegaConf.to_container(cfg.reward.custom_reward_function.reward_kwargs, resolve=True)
    mgr._reward_fn = partial(rf.compute_score, **rk)
    return mgr


class _FakeScorers:
    """Mock the two heavy scorer entry points; records calls + returns fixed rows."""

    def __init__(self):
        self.epdms_calls = []
        self.pdms_calls = 0
        self._orig_pdms = tr.pdms_trajectory_reward
        self._orig_ep = epdms_eval.epdms_score_per_frame

    def __enter__(self):
        def fake_pdms(trajs, fids, cache, **kw):
            self.pdms_calls += 1
            return [dict(_V1_PDMS) for _ in fids]

        def fake_ep(trajs, tokens, cache, **kw):
            self.epdms_calls.append({"n": len(tokens), "cache": cache})
            return [dict(_V2_ROW) for _ in tokens]

        tr.pdms_trajectory_reward = fake_pdms
        epdms_eval.epdms_score_per_frame = fake_ep
        return self

    def __exit__(self, *exc):
        tr.pdms_trajectory_reward = self._orig_pdms
        epdms_eval.epdms_score_per_frame = self._orig_ep
        return False


def _scores(batch):
    rm = batch.batch["rm_scores"]
    return rm[:, -1].tolist()  # score sits on the last valid token (mask = all ones)


def test_train_epdms_off_never_touches_v2():
    w = {"prediction": 0.0, "planning": 1.0, "soft_l2": 0.0, "endpoint": 0.0,
         "pdms": 1.0, "pdm_ep_weight": 10.0, "epdms": 0.0, "val_epdms": True}
    mgr = _make_manager(w)
    batch = _make_batch(is_validate=False, v2_stamp="/v2/train")
    with _FakeScorers() as sc:
        mgr._attach_rm_scores(batch)
    assert sc.pdms_calls == 1            # PDMS ALWAYS computed
    assert sc.epdms_calls == []          # EPDMS not in training -> v2 never loaded
    assert np.allclose(_scores(batch), [_REW10] * _N, atol=1e-6)
    assert "epdms" not in batch.non_tensor_batch


def test_train_epdms_on_scores_against_train_v2():
    w = {"prediction": 0.0, "planning": 1.0, "soft_l2": 0.0, "endpoint": 0.0,
         "pdms": 1.0, "pdm_ep_weight": 10.0, "epdms": 1.0, "val_epdms": True}
    mgr = _make_manager(w)
    batch = _make_batch(is_validate=False, v2_stamp="/v2/train")
    with _FakeScorers() as sc:
        mgr._attach_rm_scores(batch)
    assert sc.pdms_calls == 1
    assert len(sc.epdms_calls) == 1
    assert sc.epdms_calls[0]["cache"] == "/v2/train"   # train split -> TRAIN v2 cache
    assert sc.epdms_calls[0]["n"] == _N
    # score = pdms_term(rew10) + epdms_term(0.5); pdm_ep_weight never leaks into EPDMS
    assert np.allclose(_scores(batch), [_REW10 + 0.5] * _N, atol=1e-6)
    assert np.allclose(np.asarray(batch.non_tensor_batch["epdms"], dtype=np.float64), 0.5)
    assert np.allclose(np.asarray(batch.non_tensor_batch["epdms_reward"], dtype=np.float64), 0.5)
    assert np.allclose(np.asarray(batch.non_tensor_batch["epdms_TLC"], dtype=np.float64), 1.0)
    # the reweighted PDMS term is reported, the raw standard score stays visible
    assert np.allclose(np.asarray(batch.non_tensor_batch["pdms_reward"], dtype=np.float64), _REW10)
    assert np.allclose(np.asarray(batch.non_tensor_batch["pdms_PDMS"], dtype=np.float64), 0.8)


def test_val_default_is_pdms_only():
    w = {"prediction": 0.0, "planning": 1.0, "soft_l2": 0.0, "endpoint": 0.0,
         "pdms": 1.0, "pdm_ep_weight": 10.0, "epdms": 1.0}  # epdms weight set but val switch off
    mgr = _make_manager(w)
    batch = _make_batch(is_validate=True, v2_stamp="/v2/test")
    with _FakeScorers() as sc:
        mgr._attach_rm_scores(batch)
    assert sc.pdms_calls == 1            # PDMS ALWAYS computed
    assert sc.epdms_calls == []          # val_epdms unset/false -> test v2 cache never loaded
    assert np.allclose(_scores(batch), [_REW10] * _N, atol=1e-6)
    assert "epdms" not in batch.non_tensor_batch


def test_val_epdms_true_reports_epdms_without_scoring_it():
    w = {"prediction": 0.0, "planning": 1.0, "soft_l2": 0.0, "endpoint": 0.0,
         "pdms": 1.0, "pdm_ep_weight": 10.0, "epdms": 1.0, "val_epdms": True}
    mgr = _make_manager(w)
    batch = _make_batch(is_validate=True, v2_stamp="/v2/test")
    with _FakeScorers() as sc:
        mgr._attach_rm_scores(batch)
    assert sc.pdms_calls == 1
    assert len(sc.epdms_calls) == 1
    assert sc.epdms_calls[0]["cache"] == "/v2/test"    # val split -> TEST v2 cache
    # report-only: the val score keeps the standard composition (EPDMS term stripped)
    assert np.allclose(_scores(batch), [_REW10] * _N, atol=1e-6)
    # ... while the official EPDMS metric IS logged for inspection
    assert np.allclose(np.asarray(batch.non_tensor_batch["epdms"], dtype=np.float64), 0.5)


def test_val_metric_flow_reaches_swanlab_keys():
    """EPDMS survives the whole validation metric pipeline: shard metric_sums
    -> aggregate_shard_results (val-aux/.../mean) -> _add_transfusion_val_metrics
    (Transfusion-style val/ keys). Mirrors exactly what validate_shard emits for
    one sample whose epdms_components were computed."""
    from recipe.transfusion import best_ckpt as bc
    from recipe.transfusion import validation as valmod

    shard = {
        "rank": 0,
        "count": 2,
        "num_batches": 1,
        "seconds": {},
        "total_seconds": 1.0,
        # what _add_numeric accumulated from two score_sample results
        "metric_sums": {"pdms_reward": 1.6, "pdms_PDMS": 1.66, "epdms": 1.4,
                        "epdms_reward": 1.4, "epdms_TLC": 2.0, "epdms_LK": 1.9,
                        "score": 3.0},
        "metric_counts": {"pdms_reward": 2, "pdms_PDMS": 2, "epdms": 2,
                          "epdms_reward": 2, "epdms_TLC": 2, "epdms_LK": 2,
                          "score": 2},
        "pdms_counts": {"candidate": 2, "scored": 2},
    }
    metrics = valmod.aggregate_shard_results([shard], data_source="navsim_transfusion")
    assert metrics["val-aux/navsim_transfusion/epdms/mean"] == 0.7
    assert metrics["val-aux/navsim_transfusion/epdms_reward/mean"] == 0.7
    assert metrics["val-aux/navsim_transfusion/epdms_TLC/mean"] == 1.0
    assert metrics["val-aux/navsim_transfusion/epdms_LK/mean"] == 0.95
    metrics = bc._add_transfusion_val_metrics(metrics)
    # the Transfusion-style val/ group in SwanLab now carries EPDMS next to PDMS
    assert metrics["val/EPDMS"] == 0.7
    assert metrics["val/EPDMS_TLC"] == 1.0
    assert metrics["val/EPDMS_LK"] == 0.95
    assert metrics["val/pdms_reward"] == 0.8
    assert metrics["val/epdms_reward"] == 0.7
    # ... and trainer.best_metrics can resolve the short name `epdms_reward`
    assert bc.find_metric_value(metrics, "epdms_reward") == 0.7
    assert bc.find_metric_value(metrics, "pdms_reward") == 0.8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} EPDMS sync-integration tests passed.")
