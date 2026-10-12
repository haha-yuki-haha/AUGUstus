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

"""Unit tests for the framework-free Transfusion reward maths.

Run (no GPU / verl / torch needed)::

    conda activate transfusion_pro_qwen3p5
    cd AUGUstus_GRPO
    python -m pytest recipe/transfusion/tests/test_transfusion_reward.py -q
    # or simply:  python recipe/transfusion/tests/test_transfusion_reward.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recipe.transfusion import transfusion_reward as tr  # noqa: E402

# A realistic GT prediction CoT (exact format from hf_cache_*_CoT_Solution).
GT_COT = (
    "<prediction>["
    '{"bbox_2d":[0,180,552,509],"future_position":"far, front"},'
    '{"bbox_2d":[149,235,792,509],"future_position":"far, front"},'
    '{"bbox_2d":[1122,268,1283,372],"future_position":"middle, front"},'
    '{"bbox_2d":[1563,238,1640,347],"future_position":"near, right"}'
    "]</prediction>"
)


def test_bbox_iou_basic():
    assert tr.bbox_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert tr.bbox_iou([0, 0, 2, 2], [2, 2, 4, 4]) == 0.0
    # half-overlap: inter=2, union=4+4-2=6 -> 1/3
    assert abs(tr.bbox_iou([0, 0, 2, 2], [1, 0, 3, 2]) - (2.0 / 6.0)) < 1e-9


def test_parse_cot_boxes():
    boxes = tr.parse_cot_boxes(GT_COT)
    assert len(boxes) == 4
    assert boxes[0].box == (0.0, 180.0, 552.0, 509.0)
    assert boxes[0].label == "far, front"
    assert boxes[3].label == "near, right"
    # malformed / inverted boxes are dropped
    assert tr.parse_cot_boxes('{"bbox_2d":[5,5,1,1],"future_position":"x"}') == []
    assert tr.parse_cot_boxes("not json at all") == []


def test_cot_reward_perfect():
    r = tr.cot_perception_prediction_reward(GT_COT, GT_COT, iou_threshold=0.5)
    assert abs(r["cot_iou"] - 1.0) < 1e-9
    assert abs(r["cot_pred_acc"] - 1.0) < 1e-9
    assert abs(r["cot_reward"] - 2.0) < 1e-9  # w_iou=1 + w_pred=1
    assert r["n_matched"] == 4


def test_cot_reward_wrong_future_position():
    # Same boxes, but all future_position labels flipped -> detection perfect,
    # prediction accuracy zero.
    pred = GT_COT.replace("far, front", "behind").replace("middle, front", "behind").replace(
        "near, right", "behind"
    )
    r = tr.cot_perception_prediction_reward(pred, GT_COT, iou_threshold=0.5)
    assert abs(r["cot_iou"] - 1.0) < 1e-9       # boxes still localise perfectly
    assert r["cot_pred_acc"] < 1e-9             # none of the categories match


def test_cot_reward_partial_detection():
    # Predict only the first two boxes correctly -> recall 2/4.
    pred = (
        "<prediction>["
        '{"bbox_2d":[0,180,552,509],"future_position":"far, front"},'
        '{"bbox_2d":[149,235,792,509],"future_position":"far, front"}'
        "]</prediction>"
    )
    r = tr.cot_perception_prediction_reward(pred, GT_COT, iou_threshold=0.5)
    # iou_score = (1+1) / max(4 gt, 2 pred) = 0.5 ; pred_acc = 2 correct / 4 gt = 0.5
    assert abs(r["cot_iou"] - 0.5) < 1e-9
    assert abs(r["cot_pred_acc"] - 0.5) < 1e-9


def test_cot_reward_spurious_boxes_penalised():
    # Correct 2 GT boxes but emit 2 extra bogus boxes -> denom = max(4,4)=4.
    pred = (
        "<prediction>["
        '{"bbox_2d":[0,180,552,509],"future_position":"far, front"},'
        '{"bbox_2d":[149,235,792,509],"future_position":"far, front"},'
        '{"bbox_2d":[5,5,10,10],"future_position":"near, left"},'
        '{"bbox_2d":[20,20,30,30],"future_position":"near, left"}'
        "]</prediction>"
    )
    r = tr.cot_perception_prediction_reward(pred, GT_COT, iou_threshold=0.5)
    assert r["n_matched"] == 2
    assert abs(r["cot_iou"] - 0.5) < 1e-9  # 2 matched / max(4,4)


def test_l2_reward():
    gt = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    assert tr.l2_trajectory_reward(gt, gt)["l2_reward"] == 0.0  # perfect -> 0
    pred = gt + np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = tr.l2_trajectory_reward(pred, gt)
    assert abs(out["ADE"] - 1.0) < 1e-6
    assert abs(out["l2_reward"] - (-1.0)) < 1e-6
    out_exp = tr.l2_trajectory_reward(pred, gt, use_exp=True)
    assert abs(out_exp["l2_reward"] - np.exp(-1.0)) < 1e-6


def test_compute_total_reward_combines():
    gt = np.zeros((8, 3), dtype=np.float32)
    pred = gt.copy()
    w = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=1.0, endpoint=0.0, pdms=1.0)
    out = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=w,
    )
    # prediction=1.0 (perfect match) + planning(soft_l2=1.0 + pdms=0.9) -> 1.0 + 1.9 = 2.9
    assert abs(out["score"] - 2.9) < 1e-6
    assert out["pdms_reward"] == 0.9
    assert abs(out["prediction_reward"] - 1.0) < 1e-6
    assert abs(out["planning_reward"] - 1.9) < 1e-6
    assert out["pdms_PDMS"] == 0.9  # sub-metric surfaced for logging
    assert out["coef_prediction"] == 1.0  # coefficient surfaced for logging


def test_split_advantage_components():
    """compute_total_reward must surface per-head reward components."""
    gt = np.zeros((8, 3), dtype=np.float32)
    pred = gt.copy()
    w = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=1.0, endpoint=0.0, pdms=1.0)
    out = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=w,
    )
    # action head: planning block = soft_l2(1.0) + pdms(0.9) = 1.9
    assert abs(out["reward_action"] - 1.9) < 1e-6
    # CoT head: prediction(1.0) + format(0.0) + strict_verbosity(0.0) = 1.0
    assert abs(out["reward_cot"] - 1.0) < 1e-6
    # the two components must sum to score: score = prediction + planning.
    assert abs(out["score"] - (out["reward_action"] + out["reward_cot"])) < 1e-6

    # multi-pred gate zeroes ONLY the CoT reward; the action/planning head is
    # unaffected (CoT and action are independent). With strict_format_gate off,
    # the repeated-tag input trips the dedicated `multi_pred` flag.
    wm = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=1.0, endpoint=0.0,
                          pdms=1.0, strict_format_gate=False)
    bad = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt,
        pred_cot_text="<prediction>a</prediction>\n<prediction>b</prediction>",
        gt_cot_text=GT_COT, pdms_components={"PDMS": 0.9}, weights=wm,
    )
    assert bad["multi_pred"] == 1.0
    assert bad["reward_cot"] == wm.format_penalty  # CoT contribution = format penalty only
    assert bad["reward_action"] == 1.9             # action planning preserved
    assert bad["score"] == bad["reward_action"] + bad["reward_cot"]

    # When soft_l2 is disabled (the nol2 runs), reward_action == pdms only.
    wn = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=0.0, pdms=1.0)
    outn = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=wn,
    )
    assert abs(outn["reward_action"] - 0.9) < 1e-6


def test_prediction_reward_format_and_state():
    # Perfect prediction -> 1.0; wrong format (no <prediction>) -> 0.0.
    perfect = tr.prediction_reward(GT_COT, GT_COT, iou_threshold=0.5)
    assert abs(perfect["prediction_reward"] - 1.0) < 1e-6 and perfect["pred_format"] == 1.0
    bad = tr.prediction_reward("no prediction block here", GT_COT)
    assert bad["prediction_reward"] == 0.0 and bad["pred_format"] == 0.0


def test_strict_format_gate_v2_whitespace_tail():
    """strict_format_gate_v2: zero-whitespace tail after </prediction>.

    A valid block followed by whitespace filler (the RL overrun failure mode,
    docs/COT_COLLAPSE_ANALYSIS.md §2④) passes v1 (reward 1.0) but must be
    rejected by v2 (prediction reward 0 + format_penalty). A cleanly-terminated
    CoT (tail == "") must pass BOTH.
    """
    clean = GT_COT                       # ends exactly at </prediction>
    padded = GT_COT + "  \n\n   \n"       # valid block + whitespace filler

    # gate-shell view (parse=False isolates the tail rule)
    assert tr._strict_cot_format_ok(clean, strict_tail=False) is True
    assert tr._strict_cot_format_ok(padded, strict_tail=False) is True   # v1 loophole
    assert tr._strict_cot_format_ok(clean, strict_tail=True) is True     # clean still ok
    assert tr._strict_cot_format_ok(padded, strict_tail=True) is False   # v2 closes it

    gt = np.zeros((8, 3), dtype=np.float32)
    # v1 (default): whitespace tail scores full prediction reward.
    w1 = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=0.0,
                          pdms=1.0, strict_format_gate=True, strict_format_gate_v2=False)
    o1 = tr.compute_total_reward(pred_traj=gt, gt_traj=gt, pred_cot_text=padded,
                                 gt_cot_text=GT_COT, pdms_components={"PDMS": 0.9}, weights=w1)
    assert o1.get("strict_format", 0.0) == 0.0
    assert abs(o1["reward_cot"] - 1.0) < 1e-6

    # v2: identical padded text now trips the gate -> prediction 0 + penalty.
    w2 = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=0.0,
                          pdms=1.0, strict_format_gate=True, strict_format_gate_v2=True)
    o2 = tr.compute_total_reward(pred_traj=gt, gt_traj=gt, pred_cot_text=padded,
                                 gt_cot_text=GT_COT, pdms_components={"PDMS": 0.9}, weights=w2)
    assert o2["strict_format"] == 1.0
    assert o2["reward_cot"] == w2.format_penalty        # prediction zeroed, only penalty
    assert abs(o2["reward_action"] - 0.9) < 1e-6        # action head untouched

    # v2 must NOT penalise a cleanly-terminated CoT.
    o2c = tr.compute_total_reward(pred_traj=gt, gt_traj=gt, pred_cot_text=clean,
                                  gt_cot_text=GT_COT, pdms_components={"PDMS": 0.9}, weights=w2)
    assert o2c.get("strict_format", 0.0) == 0.0
    assert abs(o2c["reward_cot"] - 1.0) < 1e-6
    # v2 is inert when the base gate is off (it is a refinement of it).
    assert w2.as_coef_dict()["coef_strict_format_gate_v2"] == 1.0


def test_soft_l2_reward_bounds():
    gt = np.zeros((8, 3), dtype=np.float32)
    assert abs(tr.soft_l2_reward(gt, gt)["soft_l2_reward"] - 1.0) < 1e-6  # perfect -> 1
    far = gt.copy(); far[:, 0] = 100.0
    r = tr.soft_l2_reward(far, gt)["soft_l2_reward"]
    assert 0.0 <= r < 1e-3  # large error -> ~0, still bounded


def test_endpoint_reward_default_tight():
    # Default tightest schedule [2,4,6,10,15] -> [1,.8,.6,.4,.2], scored on the
    # LAST-waypoint L2 (x/y) only.
    gt = np.zeros((8, 3), dtype=np.float32)
    # perfect -> 1.0
    assert abs(tr.endpoint_reward(gt, gt)["endpoint_reward"] - 1.0) < 1e-9
    assert abs(tr.endpoint_reward(gt, gt)["FDE"]) < 1e-9
    # final waypoint 1.0 m off (x) -> just under threshold[0]=2.0 -> 1.0
    p = gt.copy(); p[-1, 0] = 1.0
    assert abs(tr.endpoint_reward(p, gt)["endpoint_reward"] - 1.0) < 1e-9
    # 3.0 m off -> under 4.0 -> 0.8
    p = gt.copy(); p[-1, 0] = 3.0
    assert abs(tr.endpoint_reward(p, gt)["endpoint_reward"] - 0.8) < 1e-9
    # 5.0 m off -> under 6.0 -> 0.6
    p = gt.copy(); p[-1, 0] = 5.0
    assert abs(tr.endpoint_reward(p, gt)["endpoint_reward"] - 0.6) < 1e-9
    # 50.0 m off -> past the last threshold -> 0.0
    p = gt.copy(); p[-1, 0] = 50.0
    assert tr.endpoint_reward(p, gt)["endpoint_reward"] == 0.0
    # only the LAST waypoint counts (early waypoints are ignored): large error
    # on t=0 but tight final point -> still full score.
    p = gt.copy(); p[0, 0] = 100.0
    assert abs(tr.endpoint_reward(p, gt)["endpoint_reward"] - 1.0) < 1e-9


def test_endpoint_reward_loose_schedule():
    # Looser schedule (scratch2_softerrrrr2_* family): [24,36,48,60,72].
    gt = np.zeros((8, 3), dtype=np.float32)
    loose = (24.0, 36.0, 48.0, 60.0, 72.0)
    w = tr.RewardWeights(
        endpoint_thresholds=loose,
        endpoint_scores=(1.0, 0.8, 0.6, 0.4, 0.2),
    )
    # from_dict normalises the config's list-style thresholds into tuples.
    wd = tr.RewardWeights.from_dict({"endpoint_thresholds": [24, 36, 48, 60, 72],
                                     "endpoint_scores": [1.0, 0.8, 0.6, 0.4, 0.2]})
    assert wd.endpoint_thresholds == loose and isinstance(wd.endpoint_thresholds, tuple)
    p = gt.copy(); p[-1, 0] = 10.0   # 10 m is "perfect" under the loose schedule
    assert abs(tr.endpoint_reward(p, gt, thresholds=w.endpoint_thresholds,
                                  scores=w.endpoint_scores)["endpoint_reward"] - 1.0) < 1e-9
    p = gt.copy(); p[-1, 0] = 30.0   # <36 -> 0.8
    assert abs(tr.endpoint_reward(p, gt, thresholds=w.endpoint_thresholds,
                                  scores=w.endpoint_scores)["endpoint_reward"] - 0.8) < 1e-9
    p = gt.copy(); p[-1, 0] = 80.0   # past 72 -> 0.0
    assert tr.endpoint_reward(p, gt, thresholds=w.endpoint_thresholds,
                              scores=w.endpoint_scores)["endpoint_reward"] == 0.0


def test_endpoint_reward_scratch2_l1_metric():
    """endpoint_metric="l1" reproduces scratch2_format_softerrrrr2_dac_ttc_noCoT.py's
    endpoint_reward EXACTLY: distance = |dx| + |dy| (Manhattan), NOT the L2 norm.
    The metrics diverge off-axis: (20, 20) is L2 ~28.28 (<36 -> 0.8) but L1 40
    (in [36, 48) -> 0.6)."""
    gt = np.zeros((8, 3), dtype=np.float32)
    loose = (24.0, 36.0, 48.0, 60.0, 72.0)
    scores = (1.0, 0.8, 0.6, 0.4, 0.2)

    # Off-axis divergence point at the final waypoint.
    p = gt.copy(); p[-1, 0] = 20.0; p[-1, 1] = 20.0
    r_l2 = tr.endpoint_reward(p, gt, thresholds=loose, scores=scores, metric="l2")
    r_l1 = tr.endpoint_reward(p, gt, thresholds=loose, scores=scores, metric="l1")
    assert abs(r_l2["FDE"] - float(np.sqrt(800.0))) < 1e-6   # ~28.28 <36 -> 0.8
    assert abs(r_l2["endpoint_reward"] - 0.8) < 1e-9
    assert abs(r_l1["FDE"] - 40.0) < 1e-9                    # |20| + |20| = 40
    assert abs(r_l1["endpoint_reward"] - 0.6) < 1e-9         # 40 in [36, 48) -> 0.6

    # Reference: the scratch2_softerrrrr2 endpoint_reward transcribed inline (L1).
    def scratch2_endpoint_reward(pred_xy, gt_xy):
        d = float(np.sum(np.abs(np.asarray(pred_xy[-1]) - np.asarray(gt_xy[-1]))))
        for t, s in ((24.0, 1.0), (36.0, 0.8), (48.0, 0.6), (60.0, 0.4), (72.0, 0.2)):
            if d < t:
                return s
        return 0.0

    rng = np.random.RandomState(0)
    for _ in range(50):
        a = rng.uniform(-80, 80, size=gt.shape).astype(np.float32)
        b = rng.uniform(-80, 80, size=gt.shape).astype(np.float32)
        got = tr.endpoint_reward(a, b, thresholds=loose, scores=scores, metric="l1")["endpoint_reward"]
        want = scratch2_endpoint_reward(a[:, :2], b[:, :2])
        assert got == want

    # Wiring: endpoint_metric flows from config through RewardWeights into
    # compute_total_reward; the value is normalised; the used metric is surfaced
    # as a coef_* diagnostic; invalid values are rejected at config-parse time.
    w = tr.RewardWeights.from_dict({"endpoint_metric": "L1",
                                    "endpoint_thresholds": [24.0, 36.0, 48.0, 60.0, 72.0]})
    assert w.endpoint_metric == "l1"
    out = tr.compute_total_reward(pred_traj=p, gt_traj=gt, pred_cot_text=GT_COT,
                                  gt_cot_text=GT_COT, weights=w)
    assert abs(out["FDE"] - 40.0) < 1e-9               # L1 distance flowed through
    assert abs(out["endpoint_reward"] - 0.6) < 1e-9
    assert out["coef_endpoint_metric_l1"] == 1.0
    assert tr.RewardWeights().endpoint_metric == "l2"  # default stays L2 (legacy)
    assert tr.RewardWeights().as_coef_dict()["coef_endpoint_metric_l1"] == 0.0
    for bad_ctor in (lambda: tr.RewardWeights(endpoint_metric="l_inf"),
                     lambda: tr.endpoint_reward(p, gt, metric="chebyshev")):
        try:
            bad_ctor()
            raise AssertionError("expected ValueError for an unknown endpoint metric")
        except ValueError:
            pass


def test_compute_total_reward_planning_includes_endpoint():
    gt = np.zeros((8, 3), dtype=np.float32)
    pred = gt.copy(); pred[-1, 0] = 3.0  # FDE 3.0 -> endpoint_reward 0.8 (default tight)
    w = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=1.0, pdms=1.0)
    out = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=w,
    )
    # planning = endpoint(0.8) + pdms(0.9) = 1.7 ; prediction = 1.0 ; score = 2.7
    assert abs(out["endpoint_reward"] - 0.8) < 1e-9
    assert abs(out["planning_reward"] - 1.7) < 1e-9
    assert abs(out["reward_action"] - 1.7) < 1e-9
    assert abs(out["score"] - 2.7) < 1e-9
    assert out["coef_endpoint"] == 1.0
    # endpoint sub-weight 0.0 keeps it out of planning (coef still surfaced).
    w0 = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=0.0, pdms=1.0)
    out0 = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=w0,
    )
    assert abs(out0["endpoint_reward"] - 0.8) < 1e-9  # still surfaced for logging
    assert abs(out0["planning_reward"] - 0.9) < 1e-9  # but not added to planning


def test_safe_reward():
    """DAC/TTC safety reward mirrors scratch3's planning-reward gating:
    safe_reward = TTC + DAC (NO endpoint), hard-gated to 0 when NC/DAC/TTC/EP is 0.
    The ``safe`` sub-weight defaults to 0.0 (off by default)."""
    gt = np.zeros((8, 3), dtype=np.float32)
    pred = gt.copy()  # perfect trajectory -> endpoint_reward = 1.0
    full = {"PDMS": 0.9, "NC": 1.0, "DAC": 1.0, "DDC": 1.0, "EP": 1.0, "TTC": 1.0, "C": 1.0}
    w = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=1.0,
                         pdms=0.0, safe=1.0)

    # default is OFF (0.0).
    assert tr.RewardWeights().safe == 0.0

    # unit-level: gate keys missing / None components -> None (skipped).
    assert tr.safe_reward(None) is None
    assert tr.safe_reward({"PDMS": 0.9}) is None
    assert abs(tr.safe_reward(full) - 2.0) < 1e-9  # TTC + DAC, no endpoint
    assert tr.safe_reward({**full, "TTC": 0.0}) == 0.0

    # wired into the planning block: safe = TTC(1) + DAC(1) = 2.0
    out = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components=full, weights=w,
    )
    assert abs(out["safe_reward"] - 2.0) < 1e-9
    assert abs(out["planning_reward"] - (1.0 + 2.0)) < 1e-9  # endpoint + safe
    assert out["coef_safe"] == 1.0

    # the gate: NC / DAC / TTC / EP == 0 -> safe_reward = 0 (planning keeps the
    # endpoint component only; the hard gate zeroes ONLY the safe term).
    for key in ("NC", "DAC", "TTC", "EP"):
        comps = dict(full)
        comps[key] = 0.0
        o = tr.compute_total_reward(
            pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
            pdms_components=comps, weights=w,
        )
        assert o["safe_reward"] == 0.0, key
        assert abs(o["planning_reward"] - 1.0) < 1e-9, key  # endpoint only

    # missing PDM sub-metrics -> safe component skipped entirely (no key, no
    # contribution), so PDMS-only runs are unaffected.
    o = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=w,
    )
    assert "safe_reward" not in o
    assert abs(o["planning_reward"] - 1.0) < 1e-9

    # sub-weight 0.0 (the default) -> still surfaced for logging, but not added.
    w0 = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=1.0,
                          pdms=0.0)
    assert w0.safe == 0.0
    o0 = tr.compute_total_reward(
        pred_traj=pred, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components=full, weights=w0,
    )
    assert abs(o0["safe_reward"] - 2.0) < 1e-9   # surfaced
    assert abs(o0["planning_reward"] - 1.0) < 1e-9  # but not added


def test_reward_weights_from_dict_ignores_unknown():
    w = tr.RewardWeights.from_dict({"prediction": 2.0, "pdms": 0.0, "bogus": 5})
    assert w.prediction == 2.0 and w.pdms == 0.0


def test_reverse_travel_metric():
    # forward-monotone path -> 0; a dead-zone (1 cm) ignores tiny jitter.
    fwd = np.array([[0.0, 0], [1.0, 0], [2.0, 0], [3.0, 0]])
    assert tr.reverse_travel(fwd) == 0.0
    assert tr.reverse_travel(fwd + np.array([[0.0, 0], [0.005, 0], [0.0, 0], [0.0, 0]])) == 0.0
    # surge forward then reverse 0.9 m (the metric-farming hook shape).
    hook = np.array([[0.0, 0], [1.6, 0], [1.2, 0], [0.7, 0]])
    assert abs(tr.reverse_travel(hook) - 0.9) < 1e-9
    assert tr.reverse_travel(None) == 0.0 and tr.reverse_travel(np.zeros((1, 2))) == 0.0

    # ORIGIN = waypoint 0 = [0,0]: the FIRST waypoint's x already counts.
    # starts backing up 0.9 m total, then grows forward:
    #   x = [0, -0.3, -0.9, -0.2, 0.5] -> dx = [-0.3, -0.6, +0.7, +0.7] -> 0.9
    backstart = np.array([[-0.3, 0], [-0.9, 0], [-0.2, 0], [0.5, 0]])
    assert abs(tr.reverse_travel(backstart) - 0.9) < 1e-9
    # parked backwards for its whole length (all x < 0, then monotone up):
    #   x = [0, -0.5, -0.4, -0.3] -> dx = [-0.5, +0.1, +0.1] -> 0.5 (was 0
    #   before the origin rule — the blind spot this spec targets).
    parked = np.array([[-0.5, 0], [-0.4, 0], [-0.3, 0]])
    assert abs(tr.reverse_travel(parked) - 0.5) < 1e-9
    # sub-centimetre jitter (< deadzone) stays silent.
    tiny = np.array([[-0.005, 0], [0.0, 0], [0.05, 0]])
    assert tr.reverse_travel(tiny) == 0.0


def test_reverse_gate_default_off_and_logging():
    # Default: gate OFF (False) -> reverse_travel_m is LOGGED but nothing changes.
    assert tr.RewardWeights().reverse_gate is False
    gt = np.array([[x, 0.0, 0.0] for x in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)])
    # "surge forward then reverse" hook: dx = [1.8,-0.4,-0.5,0.1,0.3,0.6,1.5]
    # -> reverse travel 0.9 m; last point == GT's -> FDE 0 -> endpoint = 1.0.
    hook = np.array([[x, 0.0, 0.0] for x in (0.6, 2.4, 2.0, 1.5, 1.6, 1.9, 2.5, 4.0)])
    full = {"PDMS": .9, "NC": 1., "DAC": 1., "DDC": 1., "EP": 1., "TTC": 1., "C": 1.}
    w0 = tr.RewardWeights(soft_l2=0.0, endpoint=1.0, pdms=0.0, safe=1.0)
    out0 = tr.compute_total_reward(pred_traj=hook, gt_traj=gt, pdms_components=full, weights=w0)
    assert abs(out0["reverse_travel_m"] - 0.9) < 1e-9        # always logged
    assert "reverse_gated" not in out0                        # gate off
    assert abs(out0["safe_reward"] - 2.0) < 1e-9              # untouched
    assert abs(out0["planning_reward"] - 3.0) < 1e-9          # untouched

    # gate ON (reverse_gate=True): the hook has backward segments -> the WHOLE
    # planning reward is hard-zeroed, EVEN with safe_gates_planning off;
    # safe_reward itself is still logged as usual (the gate zeroes the BASE
    # planning block, not the safe metric readout).
    w = tr.RewardWeights(soft_l2=0.0, endpoint=1.0, pdms=0.0, safe=1.0,
                          reverse_gate=True)
    out = tr.compute_total_reward(pred_traj=hook, gt_traj=gt, pdms_components=full, weights=w)
    assert out["reverse_gated"] == 1.0
    assert abs(out["safe_reward"] - 2.0) < 1e-9               # still logged
    assert out["planning_reward"] == 0.0                      # whole block zeroed
    # a clean forward trajectory of the SAME scene is untouched (planning 3.0).
    clean = tr.compute_total_reward(pred_traj=gt, gt_traj=gt, pdms_components=full, weights=w)
    assert "reverse_gated" not in clean and abs(clean["planning_reward"] - 3.0) < 1e-9
    # NO GT exemption any more (user spec): even though the GT ITSELF reverses,
    # a reversing prediction is still hard-zeroed.
    gt_rev = np.array([[2.0, 0, 0], [1.5, 0, 0], [1.0, 0, 0], [0.5, 0, 0],
                       [0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]])
    out_ex = tr.compute_total_reward(pred_traj=hook, gt_traj=gt_rev,
                                     pdms_components=full, weights=w)
    assert out_ex.get("reverse_gated") == 1.0 and out_ex["planning_reward"] == 0.0
    # no GT available -> the gate still applies.
    out_nogt = tr.compute_total_reward(pred_traj=hook, gt_traj=None, weights=w)
    assert out_nogt.get("reverse_gated") == 1.0 and out_nogt["planning_reward"] == 0.0

    # ORIGIN = waypoint 0: a trajectory whose FIRST waypoint already sits
    # backwards is gated even if it then grows forward (the stated rule:
    # "if the first x is negative -> 0 score").
    backstart8 = np.array([[-0.4, 0, 0], [-0.9, 0, 0], [-0.2, 0, 0],
                           [0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0],
                           [2.0, 0, 0], [4.0, 0, 0]])
    out_bs = tr.compute_total_reward(pred_traj=backstart8, gt_traj=gt,
                                     pdms_components=full, weights=w)
    assert out_bs.get("reverse_gated") == 1.0 and out_bs["planning_reward"] == 0.0

    # from_dict wiring: bool accepted; missing key defaults to False; non-bool
    # value caught. The LEGACY metre-threshold knob stays accepted (resume
    # compat) with its OLD semantics: threshold + GT exemption.
    wd = tr.RewardWeights.from_dict({"reverse_gate": True})
    assert wd.reverse_gate is True
    assert tr.RewardWeights.from_dict({}).reverse_gate is False
    import pytest
    with pytest.raises(ValueError):
        tr.RewardWeights(reverse_gate=0.3)  # type: ignore[arg-type]
    wd_m = tr.RewardWeights.from_dict({"reverse_gate_m": 0.3})
    assert wd_m.reverse_gate_m == 0.3 and wd_m.reverse_gate is False
    assert tr.RewardWeights.from_dict({}).reverse_gate_m is None

    # legacy threshold knob semantics (resume compat): > 0.3 m reverses ->
    # zeroed; the hook's 0.9 m reverses -> gate fires.
    wm = tr.RewardWeights(soft_l2=0.0, endpoint=1.0, pdms=0.0, safe=1.0,
                          reverse_gate_m=0.3)
    out_m = tr.compute_total_reward(pred_traj=hook, gt_traj=gt, pdms_components=full, weights=wm)
    assert out_m["reverse_gated"] == 1.0 and out_m["planning_reward"] == 0.0
    # ...below the threshold it stays silent: 0.9 < 1.0 -> untouched.
    wm2 = tr.RewardWeights(soft_l2=0.0, endpoint=1.0, pdms=0.0, safe=1.0,
                           reverse_gate_m=1.0)
    out_m2 = tr.compute_total_reward(pred_traj=hook, gt_traj=gt, pdms_components=full, weights=wm2)
    assert "reverse_gated" not in out_m2 and abs(out_m2["planning_reward"] - 3.0) < 1e-9
    # ...and the GT-exemption STILL applies to the legacy knob: GT itself
    # reverses 2.0 m past the threshold -> valid backing-up scene -> silent.
    out_ex2 = tr.compute_total_reward(pred_traj=hook, gt_traj=gt_rev,
                                      pdms_components=full, weights=wm)
    assert "reverse_gated" not in out_ex2 and out_ex2["planning_reward"] > 0.0


def test_safe_reward_configurable_metrics_and_gates():
    """``safe_metrics`` (summed) and ``safe_gates`` (zeroed when any is 0) are
    configurable; the defaults reproduce the original TTC+DAC / NC,DAC,TTC,EP."""
    full = {"PDMS": 0.9, "NC": 1.0, "DAC": 1.0, "DDC": 1.0, "EP": 1.0, "TTC": 0.8, "C": 0.6}

    # defaults == original behaviour (TTC + DAC, gated by NC/DAC/TTC/EP).
    assert abs(tr.safe_reward(full) - 1.8) < 1e-9
    assert tr.safe_reward({**full, "EP": 0.0}) == 0.0

    # custom SUM: add C (and EP) into the reward.
    assert abs(tr.safe_reward(full, metrics=("TTC", "DAC", "C")) - 2.4) < 1e-9
    assert abs(tr.safe_reward(full, metrics=("PDMS",)) - 0.9) < 1e-9

    # custom GATE: C==0 now zeroes the reward; EP no longer gates.
    assert abs(tr.safe_reward(full, gates=("NC", "DAC", "TTC", "C")) - 1.8) < 1e-9
    assert tr.safe_reward({**full, "C": 0.0}, gates=("NC", "DAC", "TTC", "C")) == 0.0
    assert abs(tr.safe_reward({**full, "EP": 0.0}, gates=("NC", "DAC", "TTC")) - 1.8) < 1e-9

    # ANY required key missing (metrics OR gates) -> None (component skipped);
    # keys NOT required by the config do not matter.
    assert tr.safe_reward({"PDMS": 0.9}, metrics=("PDMS",), gates=()) == 0.9
    assert tr.safe_reward({"NC": 1.0}, metrics=("TTC",), gates=()) is None
    assert tr.safe_reward({"TTC": 1.0}, metrics=("TTC",), gates=("DAC",)) is None

    # empty gate -> never zeroed; empty metrics -> sums to 0 when the gate passes.
    assert tr.safe_reward(full, metrics=("TTC",), gates=()) == 0.8
    assert tr.safe_reward(full, metrics=(), gates=("NC",)) == 0.0

    # wired through RewardWeights / compute_total_reward (normalised case,
    # list->tuple, aliases rejected). planning = endpoint(1.0) + safe(2.4).
    gt = np.zeros((8, 3), dtype=np.float32)
    w = tr.RewardWeights.from_dict({
        "prediction": 1.0, "planning": 1.0, "soft_l2": 0.0, "endpoint": 1.0,
        "pdms": 0.0, "safe": 1.0,
        "safe_metrics": ["ttc", "DAC", "c"],      # case-insensitive, deduped
        "safe_gates": ["NC", "DAC", "TTC"],        # EP no longer gates
    })
    assert w.safe_metrics == ("TTC", "DAC", "C")
    assert w.safe_gates == ("NC", "DAC", "TTC")
    out = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components=full, weights=w,
    )
    assert abs(out["safe_reward"] - 2.4) < 1e-9
    assert abs(out["planning_reward"] - (1.0 + 2.4)) < 1e-9
    out_ep0 = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={**full, "EP": 0.0}, weights=w,
    )
    assert abs(out_ep0["safe_reward"] - 2.4) < 1e-9  # EP==0 no longer gates

    # unknown metric names raise loudly at config-parse time.
    for bad in ("safe_metrics", "safe_gates"):
        try:
            tr.RewardWeights(**{bad: ["bogus"]})
            raise AssertionError(f"expected ValueError for {bad}=['bogus']")
        except ValueError:
            pass


def test_compute_total_reward_multi_prediction_zero_cot_only():
    gt_traj = np.zeros((8, 3), dtype=np.float32)
    pred_traj = np.array([[0.1, 0, 0]] * 8, dtype=np.float32)
    gt_cot = (
        "<prediction>"
        '[{"bbox_2d": [1, 2, 3, 4], "future_position": "near, left"}]'
        "</prediction>"
    )
    # Isolate the multi-prediction gate: strict_format_gate off (it already
    # subsumes repeated-tag detection and returns earlier with strict_format).
    w = tr.RewardWeights(strict_format_gate=False)
    # Single valid block -> normal reward, gate off.
    ok = tr.compute_total_reward(
        pred_traj=pred_traj, gt_traj=gt_traj,
        pred_cot_text=gt_cot, gt_cot_text=gt_cot,
        pdms_components={"PDMS": 0.8},
        weights=w,
    )
    assert ok["score"] > 0.0 and ok.get("multi_pred", 0.0) == 0.0
    # The observed collapse pattern: one block + repeated closing tags. The CoT
    # term is zeroed, but the planning/action reward is preserved.
    bad = tr.compute_total_reward(
        pred_traj=pred_traj, gt_traj=gt_traj,
        pred_cot_text=gt_cot + "</prediction>",
        gt_cot_text=gt_cot,
        pdms_components={"PDMS": 0.8},
        weights=w,
    )
    assert bad["multi_pred"] == 1.0
    assert bad["prediction_reward"] == 0.0       # CoT term zeroed
    assert bad["planning_reward"] > 0.0          # action/planning preserved
    assert bad["score"] == bad["reward_action"] + bad["reward_cot"]
    # Multiple opening tags -> prediction zeroed, planning preserved.
    bad2 = tr.compute_total_reward(
        pred_traj=pred_traj, gt_traj=gt_traj,
        pred_cot_text="<prediction>a</prediction>\n<prediction>b</prediction>",
        gt_cot_text=gt_cot,
        pdms_components={"PDMS": 0.8},
        weights=w,
    )
    assert bad2["multi_pred"] == 1.0
    assert bad2["prediction_reward"] == 0.0 and bad2["planning_reward"] > 0.0


def test_strict_format_gate_ok():
    # GT_COT is exactly one clean <prediction>[...]</prediction> -> passes.
    gt = np.zeros((8, 3), dtype=np.float32)
    w = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=1.0, pdms=1.0)
    out = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=w,
    )
    assert out.get("strict_format", 0.0) == 0.0
    assert out["score"] > 0.0


def test_strict_format_gate_zeros_cot_only_on_deviation():
    gt = np.zeros((8, 3), dtype=np.float32)
    w = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=1.0, pdms=1.0)

    def _run(pred_cot):
        out = tr.compute_total_reward(
            pred_traj=gt, gt_traj=gt, pred_cot_text=pred_cot, gt_cot_text=GT_COT,
            pdms_components={"PDMS": 0.9}, weights=w,
        )
        return out

    # 1. non-whitespace content AFTER </prediction> (the "not yet ended" tail).
    out_tail = _run(GT_COT + ",,,,,,")
    assert out_tail["strict_format"] == 1.0
    assert out_tail["prediction_reward"] == 0.0       # CoT zeroed
    assert out_tail["planning_reward"] > 0.0          # action/planning preserved
    # 2. a code-fence-wrapped block is the SFT model's NATIVE format
    #    (<prediction>```json[...]```</prediction>) -- it must now PASS the gate.
    inner = GT_COT[len("<prediction>"):-len("</prediction>")]  # the bare JSON array
    out_fence_ok = _run("<prediction>```json\n" + inner + "\n```</prediction>")
    assert out_fence_ok.get("strict_format", 0.0) == 0.0 and out_fence_ok["score"] > 0.0
    assert out_fence_ok["planning_reward"] > 0.0  # trajectory reward survives
    # 2b. a fence that is NOT a clean ```json/``` wrapper still fails
    #     (malformed fence -> deviation, CoT zero but planning preserved).
    out_fence_bad = _run("<prediction>```NOPE\n" + inner + "\n```</prediction>")
    assert out_fence_bad["strict_format"] == 1.0
    assert out_fence_bad["prediction_reward"] == 0.0 and out_fence_bad["planning_reward"] > 0.0
    # 3. non-whitespace content BEFORE <prediction>.
    out_head = _run("thinking...\n" + GT_COT)
    assert out_head["strict_format"] == 1.0
    assert out_head["prediction_reward"] == 0.0 and out_head["planning_reward"] > 0.0
    # 4. repeated / extra tags.
    out_repeat = _run(GT_COT + "</prediction>")
    assert out_repeat["strict_format"] == 1.0
    assert out_repeat["prediction_reward"] == 0.0 and out_repeat["planning_reward"] > 0.0
    # 5. inner content is not a JSON array.
    out_not_array = _run("<prediction>not-json</prediction>")
    assert out_not_array["strict_format"] == 1.0
    assert out_not_array["prediction_reward"] == 0.0 and out_not_array["planning_reward"] > 0.0


def test_default_trajectory_invalid_action_to_zeros():
    """Invalid/absent action must default to [0,0,0]*8 and still produce a defined
    planning reward, in both training and testing (compute_total_reward is the
    single shared entry point)."""
    gt = np.zeros((8, 3), dtype=np.float32)
    gt[:, 0] = 5.0
    w = tr.RewardWeights(prediction=1.0, planning=1.0, soft_l2=1.0, endpoint=1.0, pdms=0.0)

    # _default_trajectory unit checks.
    assert tr._default_trajectory(None).shape == (8, 3)
    assert np.allclose(tr._default_trajectory(None), 0.0)
    assert np.allclose(tr._default_trajectory(np.full((8, 3), np.nan)), 0.0)
    assert np.allclose(tr._default_trajectory(np.array([1.0, 2.0, 3.0])), 0.0)  # 1 frame != 8 => invalid
    assert np.allclose(tr._default_trajectory(np.zeros((8, 3))), 0.0)  # valid zero passes through
    # a valid 8x3 trajectory passes through unchanged.
    valid = np.arange(24, dtype=np.float64).reshape(8, 3)
    assert np.allclose(tr._default_trajectory(valid), valid)
    # a flattened [8*dim] action (still 8 frames) passes through as [8,3].
    flat = np.arange(24, dtype=np.float64)
    assert np.allclose(tr._default_trajectory(flat), flat.reshape(8, 3))

    # None/NaN/wrong-shape action -> planning computed against default zeros,
    # never a crash and always a defined (non-NaN) planning reward.
    for bad_traj in (None, np.full((8, 3), np.nan), np.array([1.0, 2.0, 3.0])):
        out = tr.compute_total_reward(
            pred_traj=bad_traj, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
            pdms_components=None, weights=w,
        )
        assert np.isfinite(out["planning_reward"])
        # the default zero trajectory is far from the GT (x=5) -> low, defined score.
        assert out["planning_reward"] < 1.0

    # a VALID action keeps its full (near-perfect) planning reward.
    good = gt.copy()
    out_good = tr.compute_total_reward(
        pred_traj=good, gt_traj=gt, pred_cot_text=GT_COT, gt_cot_text=GT_COT,
        pdms_components=None, weights=w,
    )
    assert out_good["planning_reward"] > 1.0


def test_strict_format_gate_disable_restores_soft():
    # With strict_format_gate=False the hard-zero is lifted: a tail of filler is
    # NOT zeroed (it has no dedicated soft penalty anymore), and the score still
    # comes from prediction + planning.
    gt = np.zeros((8, 3), dtype=np.float32)
    w = tr.RewardWeights(
        prediction=1.0, planning=1.0, soft_l2=0.0, endpoint=1.0, pdms=1.0,
        strict_format_gate=False,
    )
    out = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text=GT_COT + ",,,,,,", gt_cot_text=GT_COT,
        pdms_components={"PDMS": 0.9}, weights=w,
    )
    assert out.get("strict_format", 0.0) == 0.0
    assert out["score"] > 0.0  # not zeroed when the gate is off


def test_reweigh_pdms_ep():
    """Custom EP weight recomposes the PDM Score with a re-derived denominator."""
    comps = {"PDMS": 0.0, "NC": 1.0, "DAC": 1.0, "DDC": 1.0,
             "EP": 0.6, "TTC": 0.8, "C": 0.5}

    # ep_weight = 5.0 must reproduce the standard NAVSIM score exactly:
    # NC*DAC*(5*EP + 5*TTC + 2*C) / 12
    std = 1.0 * 1.0 * (5.0 * 0.6 + 5.0 * 0.8 + 2.0 * 0.5) / 12.0
    assert abs(tr.reweigh_pdms_ep(comps, ep_weight=5.0) - std) < 1e-12

    # ep_weight = 10.0: numerator AND denominator change -> / (10 + 5 + 2) = 17
    rew = 1.0 * 1.0 * (10.0 * 0.6 + 5.0 * 0.8 + 2.0 * 0.5) / 17.0
    assert abs(tr.reweigh_pdms_ep(comps, ep_weight=10.0) - rew) < 1e-12
    # raising the EP weight pulls PDMS toward EP (0.6 < the other terms'
    # weighted average (5*0.8+2*0.5)/7 ~= 0.714, so the score dips here)
    assert rew < std < 1.0

    # ... and when EP beats the rest, the same higher weight pays MORE:
    comps_hi = dict(comps, EP=0.9)
    assert tr.reweigh_pdms_ep(comps_hi, ep_weight=10.0) > tr.reweigh_pdms_ep(comps_hi, ep_weight=5.0)

    # multiplicative terms zero the score regardless of the EP weight
    blocked = dict(comps, NC=0.0)
    assert tr.reweigh_pdms_ep(blocked, ep_weight=10.0) == 0.0

    # missing sub-metrics / degenerate inputs -> None (caller keeps the scorer PDMS)
    assert tr.reweigh_pdms_ep(None, ep_weight=10.0) is None
    assert tr.reweigh_pdms_ep({"PDMS": 0.9}, ep_weight=10.0) is None
    assert tr.reweigh_pdms_ep(comps, ep_weight=-5.0,
                              ttc_weight=4.0, comfort_weight=1.0) is None  # denom 0


def test_pdm_ep_weight_reweights_pdms_term():
    """pdm_ep_weight is the EP parameter OF THE PDMS score: with a non-standard
    value the ``pdms`` term / ``pdms_reward`` metric recompose PDMS from its
    sub-metrics with that EP weight (train AND val); the raw standard score
    stays logged as ``pdms_PDMS``. It never defines/touches EPDMS."""
    gt = np.zeros((8, 3), dtype=np.float32)
    comps = {"PDMS": 0.8, "NC": 1.0, "DAC": 1.0, "DDC": 1.0,
             "EP": 0.6, "TTC": 0.8, "C": 0.5}
    rew = (10.0 * 0.6 + 5.0 * 0.8 + 2.0 * 0.5) / 17.0  # / (10+5+2)

    # default: standard PDMS everywhere, no EPDMS.
    w0 = tr.RewardWeights(prediction=0.0, planning=1.0, soft_l2=0.0, endpoint=0.0, pdms=1.0)
    out0 = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text="", gt_cot_text=None,
        pdms_components=comps, weights=w0,
    )
    assert out0["pdms_reward"] == 0.8
    assert "epdms" not in out0
    assert out0["coef_pdm_ep_weight"] == 5.0  # logged effective weight = standard
    assert out0["coef_epdms"] == 0.0

    # pdm_ep_weight=10: the `pdms` TERM itself is recomposed with EP weight 10
    # (the raw standard score stays as `pdms_PDMS`).
    w10 = tr.RewardWeights(prediction=0.0, planning=1.0, soft_l2=0.0, endpoint=0.0,
                           pdms=1.0, pdm_ep_weight=10.0)
    out10 = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text="", gt_cot_text=None,
        pdms_components=comps, weights=w10,
    )
    assert abs(out10["pdms_reward"] - rew) < 1e-12          # EP-reweighted PDMS
    assert out10["pdms_PDMS"] == 0.8                        # raw std still logged
    assert out10["coef_pdm_ep_weight"] == 10.0
    assert abs(out10["planning_reward"] - rew) < 1e-12
    assert abs(out10["score"] - rew) < 1e-12

    # missing sub-metrics -> recomposition impossible -> fall back to the raw
    # scorer PDMS (nothing is silently dropped).
    out_fb = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text="", gt_cot_text=None,
        pdms_components={"PDMS": 0.9}, weights=w10,
    )
    assert out_fb["pdms_reward"] == 0.9
    assert "epdms" not in out_fb

    # knob validation unchanged.
    w_fd = tr.RewardWeights.from_dict({"pdm_ep_weight": 10.0, "epdms": 0.5,
                                       "val_epdms": True})
    assert w_fd.pdm_ep_weight == 10.0 and w_fd.epdms == 0.5 and w_fd.val_epdms is True
    for bad in ({"pdm_ep_weight": -1.0}, {"val_epdms": "yes"}):
        try:
            tr.RewardWeights(**bad)
            raise AssertionError(f"{bad} must be rejected")
        except ValueError:
            pass


def test_official_epdms_in_total_reward():
    """The OFFICIAL NAVSIM v2 EPDMS (the ``epdms`` term/metric) is consumed from
    precomputed ``epdms_components`` — it is logged with its v2 sub-metrics and
    added to the planning block via the ``epdms`` weight; ``pdm_ep_weight`` has
    NO effect on it."""
    gt = np.zeros((8, 3), dtype=np.float32)
    pdms_comps = {"PDMS": 0.8, "NC": 1.0, "DAC": 1.0, "DDC": 1.0,
                  "EP": 0.6, "TTC": 0.8, "C": 0.5}
    ep_comp = {"EPDMS": 0.7, "NC": 1.0, "DAC": 1.0, "DDC": 1.0, "TLC": 1.0,
               "EP": 0.6, "TTC": 0.9, "LK": 0.8, "HC": 0.5}

    # epdms=0 (default): components alone do NOT add the term to the score
    # (the metric is still logged when supplied — e.g. the val reporting path).
    w0 = tr.RewardWeights(prediction=0.0, planning=1.0, soft_l2=0.0, endpoint=0.0,
                          pdms=1.0)
    out0 = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text="", gt_cot_text=None,
        pdms_components=pdms_comps, epdms_components=ep_comp, weights=w0,
    )
    assert out0["epdms"] == 0.7
    assert out0["epdms_reward"] == 0.7                      # reward-facing alias (best_metrics)
    assert out0["epdms_TLC"] == 1.0 and out0["epdms_LK"] == 0.8
    assert abs(out0["score"] - 0.8) < 1e-12                 # no EPDMS in score

    # epdms=1.0 (pdms=0): reward EPDMS INSTEAD of the standard PDMS.
    we = tr.RewardWeights(prediction=0.0, planning=1.0, soft_l2=0.0, endpoint=0.0,
                          pdms=0.0, epdms=1.0)
    oute = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text="", gt_cot_text=None,
        pdms_components=pdms_comps, epdms_components=ep_comp, weights=we,
    )
    assert oute["pdms_reward"] == 0.8                       # PDMS metric still there
    assert abs(oute["planning_reward"] - 0.7) < 1e-12
    assert abs(oute["score"] - 0.7) < 1e-12

    # pdm_ep_weight NEVER affects EPDMS: same components, EP-weighted PDMS on
    # the pdms side, identical EPDMS value/term.
    w10 = tr.RewardWeights(prediction=0.0, planning=1.0, soft_l2=0.0, endpoint=0.0,
                           pdms=1.0, pdm_ep_weight=10.0, epdms=1.0)
    out10 = tr.compute_total_reward(
        pred_traj=gt, gt_traj=gt, pred_cot_text="", gt_cot_text=None,
        pdms_components=pdms_comps, epdms_components=ep_comp, weights=w10,
    )
    rew = (10.0 * 0.6 + 5.0 * 0.8 + 2.0 * 0.5) / 17.0
    assert out10["epdms"] == 0.7                            # EPDMS unchanged
    assert out10["epdms_reward"] == 0.7
    assert abs(out10["pdms_reward"] - rew) < 1e-12          # pdms side reweighted
    assert abs(out10["score"] - (rew + 0.7)) < 1e-12


def test_epdms_enabled_rule():
    """The ONE shared 'compute EPDMS?' rule: train needs an active epdms
    weight, validation needs val_epdms=true; PDMS is irrelevant to it."""
    w_off = tr.RewardWeights()                              # epdms=0, val_epdms=False
    assert tr.epdms_enabled(w_off, is_validate=False) is False
    assert tr.epdms_enabled(w_off, is_validate=True) is False
    w_train = tr.RewardWeights(epdms=1.0)
    assert tr.epdms_enabled(w_train, is_validate=False) is True
    assert tr.epdms_enabled(w_train, is_validate=True) is False  # term is train-only
    w_val = tr.RewardWeights(val_epdms=True)
    assert tr.epdms_enabled(w_val, is_validate=True) is True
    assert tr.epdms_enabled(w_val, is_validate=False) is False   # val switch inert in train


def test_epdms_train_vs_validation():
    """compute_score split semantics with the OFFICIAL EPDMS: the epdms TERM is
    train-only (the val score keeps the standard composition); the epdms METRIC
    is logged in val only when val_epdms=true; pdm_ep_weight keeps reweighting
    the PDMS term in BOTH splits (it is PDMS's EP parameter, never EPDMS's);
    and runs that do NOT compute EPDMS drop any attached components."""
    from recipe.transfusion import reward_function as rf

    gt_traj = [[0.0, 0.0, 0.0]] * 8
    comps = {"PDMS": 0.8, "NC": 1.0, "DAC": 1.0, "DDC": 1.0,
             "EP": 0.6, "TTC": 0.8, "C": 0.5}
    ep_comp = {"EPDMS": 0.7, "NC": 1.0, "DAC": 1.0, "DDC": 1.0, "TLC": 1.0,
               "EP": 0.6, "TTC": 0.9, "LK": 0.8, "HC": 0.5}
    rew = (10.0 * 0.6 + 5.0 * 0.8 + 2.0 * 0.5) / 17.0
    cfg_weights = {"prediction": 0.0, "planning": 1.0, "soft_l2": 0.0,
                   "endpoint": 0.0, "pdms": 1.0, "pdm_ep_weight": 10.0,
                   "epdms": 1.0, "val_epdms": True}

    # TRAINING: pdms term = EP-reweighted rew, PLUS the official EPDMS term;
    # val_epdms is inert.
    r_train = rf.compute_score(
        solution_str="",
        ground_truth={"gt_trajectory": gt_traj, "frame_id": ""},
        extra_info={"pred_trajectory": gt_traj, "pdms_components": comps,
                    "epdms_components": ep_comp},
        weights=cfg_weights,
    )
    assert abs(r_train["pdms_reward"] - rew) < 1e-12
    assert r_train["pdms_PDMS"] == 0.8
    assert r_train["epdms"] == 0.7
    assert r_train["epdms_reward"] == 0.7
    assert abs(r_train["score"] - (rew + 0.7)) < 1e-12

    # TRAINING with epdms=0: EPDMS is NOT in the training -> no v2 computation,
    # and any attached components are dropped (no epdms metric at all).
    r_train_off = rf.compute_score(
        solution_str="",
        ground_truth={"gt_trajectory": gt_traj, "frame_id": ""},
        extra_info={"pred_trajectory": gt_traj, "pdms_components": comps,
                    "epdms_components": ep_comp},
        weights={**cfg_weights, "epdms": 0.0},
    )
    assert "epdms" not in r_train_off
    assert "epdms_reward" not in r_train_off
    assert abs(r_train_off["score"] - rew) < 1e-12

    # VALIDATION default (val_epdms not set): exactly the PDMS pipeline —
    # pdm_ep_weight STILL reweights the pdms term (it belongs to PDMS), and NO
    # epdms metric at all (attached components dropped; v2 cache never loaded).
    r_val_off = rf.compute_score(
        solution_str="",
        ground_truth={"gt_trajectory": gt_traj, "frame_id": ""},
        extra_info={"pred_trajectory": gt_traj, "pdms_components": comps,
                    "epdms_components": ep_comp,
                    "is_validate": True},
        weights={k: v for k, v in cfg_weights.items() if k != "val_epdms"},
    )
    assert abs(r_val_off["pdms_reward"] - rew) < 1e-12
    assert "epdms" not in r_val_off
    assert "epdms_reward" not in r_val_off
    assert abs(r_val_off["score"] - rew) < 1e-12

    # VALIDATION with val_epdms=true: the val score is untouched (epdms term
    # stripped), the pdms term stays EP-weighted, and the epdms METRIC is
    # additionally logged — both PDM results visible.
    r_val = rf.compute_score(
        solution_str="",
        ground_truth={"gt_trajectory": gt_traj, "frame_id": ""},
        extra_info={"pred_trajectory": gt_traj, "pdms_components": comps,
                    "epdms_components": ep_comp,
                    "is_validate": True},
        weights=cfg_weights,
    )
    assert abs(r_val["pdms_reward"] - rew) < 1e-12
    assert r_val["epdms"] == 0.7
    assert r_val["epdms_reward"] == 0.7
    assert abs(r_val["score"] - rew) < 1e-12


def test_resolve_metric_cache_v2_path():
    """v2 (official EPDMS) cache resolution mirrors the v1 rule: the per-sample
    split-correct stamp wins; the reward_kwargs v2 caches are the fallback."""
    from recipe.transfusion import reward_function as rf

    assert rf.resolve_metric_cache_v2_path("/per/sample_v2", {}) == "/per/sample_v2"
    assert rf.resolve_metric_cache_v2_path(None, {"metric_cache_train_v2": "/train_v2"}) == "/train_v2"
    assert rf.resolve_metric_cache_v2_path(
        None, {"metric_cache_test_v2": "/test_v2"}) == "/test_v2"
    assert rf.resolve_metric_cache_v2_path(None, {}) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} reward tests passed.")
