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

"""Unit tests for best-checkpoint-by-metric selection logic.

Run::

    python recipe/transfusion/tests/test_best_ckpt.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recipe.transfusion import best_ckpt as bc  # noqa: E402


def test_guess_direction():
    assert bc.guess_higher_is_better("pdms_reward") is True
    assert bc.guess_higher_is_better("PDMS") is True
    assert bc.guess_higher_is_better("L2") is False
    assert bc.guess_higher_is_better("ADE") is False
    assert bc.guess_higher_is_better("cot_iou_score") is True


def test_find_metric_value_resolves_verl_keys():
    metrics = {
        "val-core/navsim_transfusion/pdms_reward/mean": 0.83,
        "val-core/navsim_transfusion/L2/mean": 1.42,
        "training/global_step": 100,
    }
    assert bc.find_metric_value(metrics, "pdms_reward") == 0.83
    assert bc.find_metric_value(metrics, "L2") == 1.42
    assert bc.find_metric_value(metrics, "missing") is None
    # exact key also works
    assert bc.find_metric_value({"score": 2.5}, "score") == 2.5


def test_tracker_higher_is_better():
    t = bc.BestMetricTracker("pdms_reward", higher_is_better=True)
    assert t.update({"val-core/x/pdms_reward/mean": 0.5}, 10) is True
    assert t.best_value == 0.5 and t.best_step == 10
    assert t.update({"val-core/x/pdms_reward/mean": 0.4}, 20) is False  # worse
    assert t.update({"val-core/x/pdms_reward/mean": 0.7}, 30) is True   # better
    assert t.best_value == 0.7 and t.best_step == 30


def test_tracker_lower_is_better():
    t = bc.BestMetricTracker("L2", higher_is_better=False)
    assert t.update({"val-core/x/L2/mean": 2.0}, 10) is True
    assert t.update({"val-core/x/L2/mean": 1.5}, 20) is True            # lower -> better
    assert t.update({"val-core/x/L2/mean": 1.8}, 30) is False
    assert t.best_value == 1.5 and t.best_step == 20


def test_tracker_ignores_missing_metric():
    t = bc.BestMetricTracker("pdms_reward")
    assert t.update({"some/other/metric": 1.0}, 5) is False
    assert t.best_value is None


def test_parse_best_metrics_forms():
    assert bc.parse_best_metrics(None) == []
    one = bc.parse_best_metrics("L2")
    assert len(one) == 1 and one[0].name == "L2" and one[0].higher_is_better is False
    many = bc.parse_best_metrics([
        {"name": "pdms_reward", "higher_is_better": True},
        {"name": "L2", "higher_is_better": False},
    ])
    assert [m.name for m in many] == ["pdms_reward", "L2"]
    assert many[0].higher_is_better is True and many[1].higher_is_better is False


def test_snapshot_writes_pointer_and_copies():
    with tempfile.TemporaryDirectory() as tmp:
        step_dir = os.path.join(tmp, "global_step_50")
        os.makedirs(os.path.join(step_dir, "actor"))
        with open(os.path.join(step_dir, "actor", "model.bin"), "w") as f:
            f.write("weights")
        best_dir = os.path.join(tmp, "best_pdms_reward")
        bc.snapshot_checkpoint(
            step_dir, best_dir,
            info={"metric": "pdms_reward", "value": 0.9, "global_step": 50},
            background=False,  # synchronous for the test
        )
        # pointer written immediately
        assert os.path.exists(os.path.join(best_dir, "best.json"))
        # full copy landed at <best_dir>.ckpt
        assert os.path.exists(os.path.join(best_dir + ".ckpt", "actor", "model.bin"))


def test_seed_tracker_from_disk_resume():
    """A resumed run must NOT lose the historical best: the tracker is re-seeded
    from the persisted best_<metric>.ckpt/best.json, so the first post-resume
    validation can only overwrite it when it is genuinely better."""
    import json as _json

    with tempfile.TemporaryDirectory() as tmp:
        # simulate a prior segment: best_pdms_reward.ckpt/best.json @ step 6000
        best_dir = os.path.join(tmp, "best_pdms_reward.ckpt")
        os.makedirs(best_dir)
        with open(os.path.join(best_dir, "best.json"), "w") as f:
            _json.dump({"metric": "pdms_reward", "higher_is_better": True,
                        "value": 0.9051, "global_step": 6000,
                        "source": os.path.join(tmp, "global_step_6000")}, f)

        # fresh tracker (as at every launch), then seed from disk
        t = bc.BestMetricTracker("pdms_reward", higher_is_better=True)
        assert bc.seed_tracker_from_disk(t, tmp) is True
        assert abs(t.best_value - 0.9051) < 1e-12 and t.best_step == 6000

        # a slightly WORSE post-resume validation must NOT count as an improvement
        # (without seeding, best_value=None would have made it one!)
        assert t.update({"val-core/x/pdms_reward/mean": 0.90}, 6500) is False
        assert t.best_step == 6000
        # ... and a genuinely better one must pass
        assert t.update({"val-core/x/pdms_reward/mean": 0.906}, 6500) is True
        assert t.best_step == 6500


def test_seed_tracker_from_disk_skips_unusable():
    import json as _json

    with tempfile.TemporaryDirectory() as tmp:
        # no pointer at all
        t = bc.BestMetricTracker("pdms_reward", higher_is_better=True)
        assert bc.seed_tracker_from_disk(t, tmp) is False
        assert t.best_value is None

        # unparsable json
        bd = os.path.join(tmp, "best_pdms_reward.ckpt")
        os.makedirs(bd)
        with open(os.path.join(bd, "best.json"), "w") as f:
            f.write("{not json")
        assert bc.seed_tracker_from_disk(t, tmp) is False and t.best_value is None

        # direction mismatch -> must NOT trust the old baseline
        with open(os.path.join(bd, "best.json"), "w") as f:
            _json.dump({"metric": "pdms_reward", "higher_is_better": False,
                        "value": 0.9, "global_step": 10}, f)
        assert bc.seed_tracker_from_disk(t, tmp) is False and t.best_value is None

        # metric name mismatch
        with open(os.path.join(bd, "best.json"), "w") as f:
            _json.dump({"metric": "score", "higher_is_better": True,
                        "value": 0.9, "global_step": 10}, f)
        assert bc.seed_tracker_from_disk(t, tmp) is False and t.best_value is None

        # non-numeric value
        with open(os.path.join(bd, "best.json"), "w") as f:
            _json.dump({"metric": "pdms_reward", "higher_is_better": True,
                        "value": "nan", "global_step": 10}, f)
        assert bc.seed_tracker_from_disk(t, tmp) is False and t.best_value is None


def test_add_transfusion_val_metrics_maps_pdms_and_epdms():
    """val-aux/val-core means are re-published under the Transfusion SFT val/
    convention — including the OFFICIAL EPDMS family (epdms + epdms_* subs),
    so SwanLab shows val/EPDMS next to val/PDMS whenever val_epdms computed it."""
    metrics = {
        "val-aux/navsim_transfusion/pdms_PDMS/mean": 0.83,
        "val-aux/navsim_transfusion/pdms_reward/mean": 0.79,   # EP-reweighted term
        "val-aux/navsim_transfusion/pdms_NC/mean": 1.0,
        "val-core/navsim_transfusion/pdms_reward/mean": 0.79,  # core copy, first wins
        "val-aux/navsim_transfusion/epdms/mean": 0.71,
        "val-aux/navsim_transfusion/epdms_reward/mean": 0.71,
        "val-aux/navsim_transfusion/epdms_TLC/mean": 0.99,
        "val-aux/navsim_transfusion/epdms_LK/mean": 0.95,
        "val-aux/navsim_transfusion/epdms/mean@10": 0.5,       # not the plain mean -> ignored
    }
    out = bc._add_transfusion_val_metrics(dict(metrics))
    assert out["val/PDMS"] == 0.83
    assert out["val/pdms_reward"] == 0.79
    assert out["val/NC"] == 1.0
    assert out["val/EPDMS"] == 0.71
    assert out["val/epdms_reward"] == 0.71
    assert out["val/EPDMS_TLC"] == 0.99
    assert out["val/EPDMS_LK"] == 0.95
    # trainer.best_metrics can track the official EPDMS by its short name
    t = bc.BestMetricTracker("epdms_reward", higher_is_better=True)
    assert t.update(out, 10) is True and t.best_value == 0.71 and t.best_step == 10
    # a val run without EPDMS (val_epdms: false) simply yields no EPDMS keys
    out2 = bc._add_transfusion_val_metrics(
        {"val-aux/navsim_transfusion/pdms_PDMS/mean": 0.83}
    )
    assert "val/PDMS" in out2 and "val/EPDMS" not in out2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} best-ckpt tests passed.")
