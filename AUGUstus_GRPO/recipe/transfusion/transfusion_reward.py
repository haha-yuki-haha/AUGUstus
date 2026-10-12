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

"""Reward functions for RL fine-tuning of the Transfusion driving planner.

The module is intentionally **framework-free**: it depends only on ``numpy`` and
the Python standard library, so the reward maths can be unit-tested in isolation
(``tests/test_transfusion_reward.py``) without importing ``verl`` or ``torch``.

Three reward signals are provided, matching the user request:

1. ``l2_trajectory_reward``      — negative average L2 (xy) displacement vs. GT.
2. ``pdms_trajectory_reward``    — NAVSIM closed-loop PDM Score (lazy-imported
                                   from ``transfusion.utils``; needs the metric
                                   cache + the ``navsim`` package).
3. ``cot_perception_prediction_reward`` — Chain-of-Thought reward = bounding-box
                                   IoU (perception) + future-position
                                   prediction correctness. Implemented here from
                                   scratch (Transfusion only ships a parser/visualiser).
4. ``safe_reward``               — DAC/TTC closed-loop safety reward
                                   (``TTC + DAC``), hard-gated to 0
                                   when NC / DAC / TTC / EP is 0 (mirrors the
                                   planning reward of
                                   ``eval_tmp/reward_function/scratch3_format_softerrrrrr_dac_ttc_noCoT.py``).
                                   The summed metrics (``RewardWeights.safe_metrics``)
                                   and the gated metrics (``.safe_gates``) are
                                   configurable; defaults reproduce the above.

``compute_total_reward`` linearly combines the three with configurable weights.

Coordinate / unit conventions (must match Transfusion):
    * Trajectories are ``[T, 3]`` = (x, y, heading_radian) in **metres / radians**,
      already de-normalised — exactly what ``generate_trajectory`` returns and
      what the dataset stores in ``future_x_y_angle``.
    * CoT bounding boxes are pixel-space ``[x1, y1, x2, y2]`` in the model's image
      frame (e.g. 2048x512), parsed from the ``<prediction>[...]</prediction>``
      JSON blocks, identical format for prediction and ground-truth.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Default reward weights (overridable from config / call site).
# These are the "for now" defaults requested: L2 + PDMS + CoT(IoU + prediction).
# ---------------------------------------------------------------------------

# Official NAVSIM PDM Score aggregation weights (EP / TTC / comfort) — used by
# :func:`reweigh_pdms_ep` to recompose PDMS with a CUSTOM EP weight
# (``RewardWeights.pdm_ep_weight``); the standard values reproduce the
# scorer's PDMS to floating-point precision.
PDMS_STD_EP_WEIGHT = 5.0
PDMS_STD_TTC_WEIGHT = 5.0
PDMS_STD_COMFORT_WEIGHT = 2.0

# Canonical NAVSIM PDM sub-metric names accepted by the safe_reward knobs
# (:func:`safe_reward`, ``RewardWeights.safe_metrics`` / ``.safe_gates``).
# All keys are matched case-insensitively.
_PDM_METRIC_ALIASES = {
    "pdms": "PDMS",
    "nc": "NC",
    "dac": "DAC",
    "ddc": "DDC",
    "ep": "EP",
    "ttc": "TTC",
    "c": "C",
}

# Human-readable metric list for error messages / comments (C = comfortable).
_PDM_METRIC_NAMES_HELP = "PDMS / NC / DAC / DDC / EP / TTC / C(comfortable)"


def _canon_pdm_metrics(names: Sequence[str], *, what: str) -> Tuple[str, ...]:
    """Normalise PDM sub-metric names (case-insensitive, de-duplicated, order
    preserved). Raises ``ValueError`` on an unknown name."""
    out: List[str] = []
    for n in names:
        key = _PDM_METRIC_ALIASES.get(str(n).strip().lower())
        if key is None:
            raise ValueError(
                f"{what}: unknown PDM metric {n!r}; expected one of "
                f"{_PDM_METRIC_NAMES_HELP}"
            )
        if key not in out:
            out.append(key)
    return tuple(out)


@dataclass
class RewardWeights:
    """Hyper-parameters for the two-part Transfusion RL reward.

    ``score = prediction * prediction_reward + planning * planning_reward`` where
    ``planning_reward = soft_l2 * soft_l2_reward + endpoint * endpoint_reward
    + pdms * pdms_reward + epdms * EPDMS + safe * safe_reward``.

    * **prediction** — from the CoT text: bbox detection (IoU precision/recall)
      times future-position accuracy, gated by the ``<prediction>`` FORMAT
      (mirrors ``eval_tmp/reward_function/scratch2_format_softerrrrr2_dac_ttc_noCoT.py``).
    * **planning** — from the diffusion trajectory: a bounded *soft* L2 reward,
      the endpoint (FDE) reward, the NAVSIM PDM Score (closed-loop), and the
      DAC/TTC safety reward (closed-loop, hard-gated, see :func:`safe_reward`).
    """

    # ---- top-level parts ----
    prediction: float = 1.0   # weight of the CoT prediction reward
    planning: float = 1.0     # weight of the whole planning block
    # ---- planning sub-weights ----
    soft_l2: float = 1.0      # sub-weight of the soft-L2 trajectory reward
    endpoint: float = 1.0     # sub-weight of the (FDE) endpoint reward
    pdms: float = 1.0         # sub-weight of the PDM Score
    # EP weight INSIDE the PDMS aggregation (None = the official NAVSIM 5.0);
    # see :func:`reweigh_pdms_ep`. Belongs to PDMS (ALWAYS computed), never to
    # EPDMS.
    pdm_ep_weight: Optional[float] = None
    # OFFICIAL NAVSIM v2 EPDMS planning term (TRAINING only — validation
    # strips it via ``strip_weights_for_split`` and reports EPDMS read-only
    # when ``val_epdms`` is true; see :func:`epdms_enabled`).
    epdms: float = 0.0
    # VALIDATION-only switch: compute + log the official EPDMS (report-only,
    # never enters the val score). ``False`` keeps validation exactly the
    # PDMS pipeline (the v2 metric caches are never loaded).
    val_epdms: bool = False
    safe: float = 0.0         # sub-weight of the DAC/TTC safety reward (off by default)
    # ---- safe_reward composition (which PDM sub-metrics are gated / added) ----
    # ``safe_reward`` = the SUM (addition) of ``safe_metrics``, HARD-gated to 0
    # when ANY of ``safe_gates`` is 0 (see :func:`safe_reward`). Each name is a
    # NAVSIM PDM sub-metric: PDMS / NC / DAC / DDC / EP / TTC / C(comfortable)
    # (case-insensitive). The defaults are EXACTLY the original behaviour
    # (gate: any of NC/DAC/TTC/EP == 0 -> 0; reward = TTC + DAC).
    safe_metrics: Tuple[str, ...] = ("TTC", "DAC")
    safe_gates: Tuple[str, ...] = ("NC", "DAC", "TTC", "EP")
    # ---- ABNORMAL REVERSE (backing-up) master switch ----
    # Hard gate over the whole planning block on ANY backward travel
    # (``reverse_travel_m`` > 0); see the ``reverse_gate_m`` threshold below
    # for the metered variant. OFF by default.
    reverse_gate: bool = False
    # ---- ABNORMAL REVERSE (backing-up) trajectory gate ----
    # PDMS is blind to motion plausibility: on near-stationary scenes a
    # trajectory that surges forward then REVERSES still scores NC/DAC/TTC/EP
    # = 1 (backing up 0.3-1 m never collides, never leaves the drivable area,
    # and barely lowers EP) — the observed "reverse to farm the metrics" hack.
    # ``reverse_travel_m`` = total NEGATIVE longitudinal (x) travel of the
    # predicted path in metres (sum of the per-step backward displacements,
    # with a small dead-zone); it is ALWAYS logged alongside the trajectory
    # metrics.
    #   reverse_gate_m: OFF by default (None) — identical to the current
    #     experiments. Set a threshold in metres (e.g. 0.3) to enable: a
    #     rollout whose predicted trajectory reverses MORE than this many
    #     metres has its WHOLE planning reward hard-zeroed (as if a safety
    #     gate had failed). Samples whose GT trajectory itself reverses past
    #     the threshold are EXEMPT (legitimate backing-up / parking scenes).
    reverse_gate_m: Optional[float] = None
    # ---- scratch2-style safety gate over the WHOLE planning block ----
    # When true, a FAILED DAC/TTC safety gate (NC / DAC / TTC / EP == 0) zeroes
    # the ENTIRE planning block (endpoint / soft-L2 / PDMS included), not just the
    # safe term — exactly the planning reward of
    # ``eval_tmp/reward_function/scratch2_format_softerrrrr2_dac_ttc_noCoT.py``::
    #
    #     planning = 0                                  if any(NC,DAC,TTC,EP) == 0
    #     planning = TTC + DAC + endpoint_reward        otherwise
    #
    # (reproduce it with soft_l2=0, pdms=0, safe=1, endpoint=1, the loose
    # endpoint_thresholds=[24,36,48,60,72], endpoint_metric="l1" and this switch
    # on). Default false keeps the legacy behaviour where the gate only shapes
    # the safe term.
    safe_gates_planning: bool = False

    # ---- endpoint tightness (metres) ----
    # Ascending FDE thresholds with their matching [0,1] scores. A distance below
    # endpoint_thresholds[0] scores endpoint_scores[0], below [1] -> scores[1], ...
    # The default is the TIGHTEST schedule from the eval_tmp endpoint_reward loop;
    # loosen it by scaling the thresholds up (e.g. 24/36/48/60/72 for the loose
    # scratch2_softerrrrr2 variants). Pass as a list from config.
    endpoint_thresholds: Tuple[float, ...] = (2.0, 4.0, 6.0, 10.0, 15.0)
    endpoint_scores: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
    # Distance metric for the final-waypoint error that feeds the endpoint
    # schedule (x/y plane only):
    #   "l2" -> Euclidean  sqrt(dx^2 + dy^2)   (legacy default, backward-compat)
    #   "l1" -> Manhattan  |dx| + |dy|         (EXACTLY the
    #          scratch2_softerrrrr2_dac_ttc_noCoT.py endpoint_reward metric; L1 >= L2,
    #          so the same thresholds gate strictly harder). The logged "FDE"
    #          mirrors the metric actually used for the reward.
    endpoint_metric: str = "l2"

    # ---- prediction matching ----
    iou_threshold: float = 0.5

    # ---- soft-L2 shaping ----
    # "exp"       -> reward = exp(-dist / soft_l2_scale)  in (0, 1]  (smooth)
    # "piecewise" -> reward = generous metre-threshold schedule       (reference style)
    soft_l2_mode: str = "exp"
    soft_l2_scale: float = 2.0        # metres (exp mode)
    soft_l2_endpoint: bool = False    # use endpoint (FDE) instead of ADE for the distance

    # ---- CoT-format handling (DECOUPLED from planning) ----
    # Small additive penalty applied to the TOTAL score when the CoT
    # <prediction> block is missing/malformed. It deliberately does NOT zero the
    # planning reward: the trajectory head must keep receiving its own
    # (soft-L2 / PDMS) gradient even when the CoT format fails, otherwise a CoT
    # collapse destroys the closed-loop signal and the group advantage becomes
    # format noise (observed: pred_format dropped to 0.40 in the offpolicy run
    # and the whole score signal collapsed).
    format_penalty: float = -0.5
    # ---- STRICT FORMAT GATE (hard zero on any format deviation) ----
    # When true, the ENTIRE reward (prediction + planning) is zeroed if the CoT
    # is not EXACTLY one clean ``<prediction>[...]</prediction>`` block:
    #   * non-whitespace content BEFORE <prediction>,
    #   * <prediction> / </prediction> not appearing EXACTLY once,
    #   * the block's inner content is not a BARE JSON array (no ```json / ```
    #     fences, no extra wrapper tags/characters),
    #   * non-whitespace content AFTER </prediction> (the "not yet ended" tail).
    strict_format_gate: bool = True
    # V2 of the strict gate: also hard-fail ANY non-empty tail after
    # </prediction> (not just non-whitespace), passed to
    # :func:`_strict_cot_format_ok` as ``strict_tail``. OFF by default keeps
    # the whitespace-tail tolerance.
    strict_format_gate_v2: bool = False

    def __post_init__(self) -> None:
        m = str(self.endpoint_metric).lower()
        if m not in ("l1", "l2"):
            raise ValueError(f"endpoint_metric must be 'l1' or 'l2', got {self.endpoint_metric!r}")
        self.endpoint_metric = m
        self.safe_metrics = _canon_pdm_metrics(self.safe_metrics, what="safe_metrics")
        self.safe_gates = _canon_pdm_metrics(self.safe_gates, what="safe_gates")
        if not isinstance(self.reverse_gate, bool):
            raise ValueError(f"reverse_gate must be a bool (true/false), got {self.reverse_gate!r}.")
        if self.reverse_gate_m is not None:
            self.reverse_gate_m = float(self.reverse_gate_m)
        if self.pdm_ep_weight is not None:
            self.pdm_ep_weight = float(self.pdm_ep_weight)
            if not np.isfinite(self.pdm_ep_weight) or self.pdm_ep_weight < 0.0:
                raise ValueError(f"pdm_ep_weight must be a non-negative finite float or None, got {self.pdm_ep_weight!r}")
        self.epdms = float(self.epdms)
        if not isinstance(self.val_epdms, bool):
            raise ValueError(f"val_epdms must be a bool (true/false), got {self.val_epdms!r}.")

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RewardWeights":
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        # tolerate legacy keys (l2/cot/cot_iou/...) by ignoring the unknown ones,
        # and normalise the list-style thresholds/scores into tuples.
        d = dict(d)
        for k in ("endpoint_thresholds", "endpoint_scores", "safe_metrics", "safe_gates"):
            v = d.get(k)
            if v is not None and not isinstance(v, tuple):
                d[k] = tuple(v)
        return cls(**{k: v for k, v in d.items() if k in known})

    def as_coef_dict(self) -> Dict[str, float]:
        """The scalar coefficients, surfaced as ``coef_*`` reward diagnostics so
        SwanLab shows exactly which weights a run used (train + val)."""
        return {
            "coef_prediction": float(self.prediction),
            "coef_planning": float(self.planning),
            "coef_soft_l2": float(self.soft_l2),
            "coef_endpoint": float(self.endpoint),
            "coef_pdms": float(self.pdms),
            "coef_pdm_ep_weight": float(
                PDMS_STD_EP_WEIGHT if self.pdm_ep_weight is None else self.pdm_ep_weight
            ),
            "coef_epdms": float(self.epdms),
            "coef_safe": float(self.safe),
            "coef_safe_gates_planning": float(self.safe_gates_planning),
            "coef_endpoint_metric_l1": float(self.endpoint_metric == "l1"),
            "coef_format_penalty": float(self.format_penalty),
            "coef_strict_format_gate": float(self.strict_format_gate),
            "coef_strict_format_gate_v2": float(self.strict_format_gate_v2),
        }


def epdms_enabled(weights: 'RewardWeights', *, is_validate: bool) -> bool:
    """Whether the OFFICIAL NAVSIM v2 EPDMS should be computed for this split.

    The ONE shared rule (used by the batch precompute in sync_driver /
    validate_shard AND by the per-sample entry points), so "does this run
    compute EPDMS?" is decided identically everywhere:

    * TRAINING — only when the ``epdms`` planning sub-weight is active
      (``!= 0.0``), i.e. "EPDMS joined the training". The per-frame EPDMS then
      enters both the planning block and the logged metrics.
    * VALIDATION — only when ``val_epdms`` is true. EPDMS is then computed +
      logged as a report-only metric (the val score keeps its standard
      composition; the ``epdms`` weight is stripped upstream anyway).

    When False, the v2 metric caches (``metric_cache_*_v2``) are NEVER loaded
    and no v2 scorer pool is spawned — plain PDMS runs waste no time on EPDMS.
    PDMS (v1) is ALWAYS computed regardless of this switch, in both splits.
    """
    if is_validate:
        return bool(weights.val_epdms)
    return float(weights.epdms) != 0.0


# ===========================================================================
# 1. L2 trajectory reward
# ===========================================================================


def _as_xy(traj: np.ndarray) -> np.ndarray:
    """Return the ``[T, 2]`` xy slice of a trajectory array."""
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj.reshape(-1, traj.shape[-1] if traj.shape[-1] in (2, 3) else 2)
    return traj[:, :2]


def trajectory_l2(pred_traj: np.ndarray, gt_traj: np.ndarray) -> Dict[str, float]:
    """Average displacement metrics between predicted and GT trajectories.

    Mirrors ``transfusion.utils.compute_action_metrics`` (xy-only, per-step L2).

    Args:
        pred_traj: ``[T, >=2]`` predicted waypoints (metres).
        gt_traj:   ``[T, >=2]`` ground-truth waypoints (metres).

    Returns:
        ``{"ADE", "FDE", "L2"}`` floats. ADE == L2 (mean over steps); FDE is the
        last-waypoint error.
    """
    p = _as_xy(pred_traj)
    g = _as_xy(gt_traj)
    t = min(len(p), len(g))
    if t == 0:
        return {"ADE": float("nan"), "FDE": float("nan"), "L2": float("nan")}
    d = np.linalg.norm(p[:t] - g[:t], axis=-1)  # [T]
    return {"ADE": float(d.mean()), "FDE": float(d[-1]), "L2": float(d.mean())}


def l2_trajectory_reward(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    *,
    scale: float = 1.0,
    use_exp: bool = False,
) -> Dict[str, float]:
    """L2-based trajectory reward (higher == better).

    By default ``reward = -scale * ADE`` so that a perfect match scores ``0`` and
    larger errors are increasingly negative — a well-behaved, monotonic signal for
    GRPO's group-relative advantage. With ``use_exp`` the reward is the bounded
    ``exp(-ADE / scale) in (0, 1]`` instead.

    Returns ``{"l2_reward", "ADE", "FDE", "L2"}``.
    """
    m = trajectory_l2(pred_traj, gt_traj)
    ade = m["ADE"]
    if np.isnan(ade):
        reward = 0.0
    elif use_exp:
        reward = float(np.exp(-ade / max(scale, 1e-6)))
    else:
        reward = float(-scale * ade)
    return {"l2_reward": reward, **m}


def _piecewise_soft_l2(dist: float) -> float:
    """Generous metre-threshold soft-L2 schedule (reference "softer" style, in metres)."""
    for thr, sc in ((1.0, 1.0), (2.0, 0.8), (3.0, 0.6), (5.0, 0.4), (8.0, 0.2)):
        if dist < thr:
            return sc
    return 0.0


def soft_l2_reward(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    *,
    mode: str = "exp",
    scale: float = 2.0,
    use_endpoint: bool = False,
) -> Dict[str, float]:
    """Bounded *soft* L2 trajectory reward in ``[0, 1]`` (higher == better).

    Unlike :func:`l2_trajectory_reward` (unbounded ``-scale*ADE``), this maps the
    displacement error through a smooth/bounded shaping so it composes cleanly
    with the ``[0, 1]`` PDM Score in the planning block:

    * ``mode="exp"``       -> ``exp(-dist / scale)`` (smooth, no thresholds),
    * ``mode="piecewise"`` -> the generous metre-threshold schedule above.

    ``dist`` is the average displacement (ADE) or, with ``use_endpoint``, the
    final-waypoint error (FDE). Returns ``{"soft_l2_reward", "ADE", "FDE", "L2"}``.
    """
    m = trajectory_l2(pred_traj, gt_traj)
    dist = m["FDE"] if use_endpoint else m["ADE"]
    if dist is None or np.isnan(dist):
        r = 0.0
    elif str(mode) == "piecewise":
        r = _piecewise_soft_l2(float(dist))
    else:  # "exp" (default)
        r = float(np.exp(-float(dist) / max(scale, 1e-6)))
    return {"soft_l2_reward": float(r), **m}


def reverse_travel(traj: Optional[np.ndarray], *, deadzone: float = 0.01) -> float:
    """Total NEGATIVE longitudinal (x) travel of a trajectory, in metres.

    The ego frame's x axis points FORWARD, so a per-step displacement
    ``dx < -deadzone`` is backwards driving. The ORIGIN counts as waypoint 0
    (``[0, 0]``): the first waypoint already sitting behind the ego (``x < 0``)
    starts the trip in reverse and MUST be captured — without it a trajectory
    parked backwards for its whole length registers ZERO travel. Returns the
    sum of those backward displacements' magnitudes (0 for a forward-monotone
    path). The dead-zone ignores sub-centimetre jitter. ``None`` / malformed
    input -> 0.0.
    """
    if traj is None:
        return 0.0
    try:
        p = _as_xy(traj)
    except Exception:
        return 0.0
    if len(p) < 1:
        return 0.0
    # Anchor the path at the ego origin (x = 0) so the FIRST waypoint's
    # backward offset from the start also counts as reverse travel.
    x = np.concatenate([[0.0], p[:, 0]])
    dx = np.diff(x)
    return float(-dx[dx < -deadzone].sum())


def endpoint_reward(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    *,
    thresholds: Sequence[float] = (2.0, 4.0, 6.0, 10.0, 15.0),
    scores: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
    metric: str = "l2",
) -> Dict[str, float]:
    """Bounded endpoint reward in ``[0, 1]`` from the FDE (last-waypoint) error.

    Unlike :func:`soft_l2_reward` (which uses the mean / ADE), this scores only the
    FINAL waypoint on the x/y plane, mirroring the ``endpoint_reward`` loop from
    the eval_tmp reward scripts. The **distance metric** is selectable via
    ``metric``: ``"l2"`` = Euclidean ``sqrt(dx^2+dy^2)`` (legacy default); ``"l1"``
    = Manhattan ``|dx| + |dy|`` -- EXACTLY the metric used by
    ``scratch2_format_softerrrrr2_dac_ttc_noCoT.py`` (``np.sum(np.abs(...))``).
    The **tightness** is a first-class knob: ``thresholds`` (metres, ascending)
    pair with ``scores`` (descending, in ``[0, 1]``) -- a distance below
    ``thresholds[0]`` scores ``scores[0]``, below ``thresholds[1]`` scores
    ``scores[1]``, and so on; anything past the last threshold scores ``0.0``.

    Examples (observed schedule families):
    * tightest ``[2,4,6,10,15]`` -> ``[1.,.8,.6,.4,.2]``  (scratch1_format_pdms_*)
    * tight      ``[6,9,12,15,20]`` -> ``[1.,.8,.6,.4,.2]`` (scratch1_endpoint / softerr)
    * very loose ``[24,36,48,60,72]`` -> ``[1.,.8,.6,.4,.2]`` (scratch2_softerrrrr2_*)

    Returns ``{"endpoint_reward", "FDE"}``; ``FDE`` is the distance in the
    ``metric`` actually used for the reward.
    """
    p = _as_xy(pred_traj)
    g = _as_xy(gt_traj)
    if len(p) == 0 or len(g) == 0:
        return {"endpoint_reward": 0.0, "FDE": float("nan")}
    m = str(metric).lower()
    if m == "l1":
        fde = float(np.sum(np.abs(p[-1] - g[-1])))
    elif m == "l2":
        fde = float(np.linalg.norm(p[-1] - g[-1]))
    else:
        raise ValueError(f"endpoint_reward: metric must be 'l1' or 'l2', got {metric!r}")
    thr = tuple(float(t) for t in thresholds)
    sc = tuple(float(s) for s in scores)
    r = 0.0
    for t, s in zip(thr, sc):
        if fde < t:
            r = s
            break
    return {"endpoint_reward": float(r), "FDE": fde}


# ===========================================================================
# 2. PDMS (NAVSIM closed-loop PDM Score) reward
# ===========================================================================


def _ensure_transfusion_on_path(transfusion_root: Optional[str] = None) -> None:
    """Make the ``transfusion`` package importable for PDMS computation.

    Resolution order: explicit arg -> ``$TRANSFUSION_ROOT`` -> the repo-relative
    sibling ``AUGUstus_SFT`` directory inferred from this file's location.
    """
    candidates: List[str] = []
    if transfusion_root:
        candidates.append(transfusion_root)
    env_root = os.environ.get("TRANSFUSION_ROOT")
    if env_root:
        candidates.append(env_root)
    # .../Code/verl_transfusion/recipe/transfusion/transfusion_reward.py
    #  -> .../AUGUstus_SFT
    here = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(here, "..", "..", ".."))
    candidates.append(os.path.join(code_dir, "Transfusion"))

    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "transfusion")) and root not in sys.path:
            sys.path.insert(0, root)
            return


# NAVSIM PDM sub-metric key aliases (raw dataclass field <-> short display name).
_PDM_KEYMAP = {
    "PDMS": ("PDMS", "score", "pdm_score"),
    "NC": ("NC", "no_at_fault_collisions"),
    "DAC": ("DAC", "drivable_area_compliance"),
    "DDC": ("DDC", "driving_direction_compliance"),
    "EP": ("EP", "ego_progress"),
    "TTC": ("TTC", "time_to_collision_within_bound"),
    "C": ("C", "comfort"),
}


def _normalize_pdm(ps: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Map a per-sample PDM result to ``{PDMS, NC, DAC, DDC, EP, TTC, C}`` floats,
    tolerating either the NAVSIM raw keys or the short display names."""
    if not ps:
        return None
    out: Dict[str, float] = {}
    for short, cands in _PDM_KEYMAP.items():
        for c in cands:
            if c in ps and ps[c] is not None:
                try:
                    out[short] = float(ps[c])
                except (TypeError, ValueError):
                    pass
                break
    return out if "PDMS" in out else None


def pdms_trajectory_reward(
    pred_trajs: np.ndarray,
    frame_ids: Sequence[str],
    metric_cache_path: Optional[str],
    *,
    num_workers: int = 8,
    transfusion_root: Optional[str] = None,
) -> List[Optional[Dict[str, float]]]:
    """Batch PDM Score reward, one dict per sample (higher better).

    Returns per sample ``{"PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C"}`` (all in
    ``[0, 1]``) — the overall closed-loop score PLUS every NAVSIM sub-metric, so
    the reward can both use ``PDMS`` and surface the sub-metrics for logging.
    ``None`` for samples whose PDM Score could not be computed (missing
    ``frame_id`` / metric cache entry), or ``[None] * N`` when the metric cache or
    the ``navsim`` package is unavailable.

    Args:
        pred_trajs: ``[N, 8, 3]`` predicted trajectories (x, y, heading).
        frame_ids:  ``N`` NAVSIM frame tokens (``batch["_frame_id"]``).
        metric_cache_path: NAVSIM metric cache dir (``data.metric_cache_test``).
    """
    pred_trajs = np.asarray(pred_trajs, dtype=np.float32)
    if pred_trajs.ndim == 2:
        pred_trajs = pred_trajs[None]
    n = len(pred_trajs)
    if not metric_cache_path:
        return [None] * n

    _ensure_transfusion_on_path(transfusion_root)
    try:
        from model.utils import compute_pdm_scores_batch
    except Exception:  # pragma: no cover - environment dependent
        return [None] * n

    result = compute_pdm_scores_batch(
        pred_trajs,
        list(frame_ids),
        num_workers=num_workers,
        metric_cache_path=metric_cache_path,
        return_per_sample=True,
    )
    per_sample = result.get("per_sample") if isinstance(result, dict) else None
    if not per_sample:
        return [None] * n
    return [_normalize_pdm(ps) for ps in per_sample]


# ---------------------------------------------------------------------------
# 2b. PDMS recomposition with a CUSTOM EP weight
# ---------------------------------------------------------------------------


def reweigh_pdms_ep(
    pdms_components: Optional[Dict[str, float]],
    *,
    ep_weight: float,
    ttc_weight: float = PDMS_STD_TTC_WEIGHT,
    comfort_weight: float = PDMS_STD_COMFORT_WEIGHT,
) -> Optional[float]:
    """Recompose the PDM Score from its sub-metrics with a CUSTOM EP weight.

    Applies the exact NAVSIM aggregation (see the constants above)::

        PDMS' = NC * DAC * (ep_weight*EP + ttc_weight*TTC + comfort_weight*C)
                          / (ep_weight + ttc_weight + comfort_weight)

    With ``ep_weight = 5.0`` (and the standard TTC/comfort weights) this
    reproduces the scorer's PDMS to floating-point precision, so it is only
    worth calling for a DIFFERENT weight. Requires the full sub-metric dict
    (``NC/DAC/EP/TTC/C``; ``DDC`` is not needed — its NAVSIM weight is 0);
    returns ``None`` when any of them is missing (nothing to reweight — the
    caller keeps the scorer's PDMS).
    """
    if not pdms_components:
        return None
    try:
        nc = float(pdms_components["NC"])
        dac = float(pdms_components["DAC"])
        ep = float(pdms_components["EP"])
        ttc = float(pdms_components["TTC"])
        c = float(pdms_components["C"])
        ep_w, ttc_w, c_w = float(ep_weight), float(ttc_weight), float(comfort_weight)
        denom = ep_w + ttc_w + c_w
        if denom <= 0.0:
            return None
        return float(nc * dac * (ep_w * ep + ttc_w * ttc + c_w * c) / denom)
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 2c. Safety reward (closed-loop, hard-gated; composition configurable)
# ---------------------------------------------------------------------------


def safe_reward(
    pdms_components: Optional[Dict[str, float]],
    *,
    metrics: Sequence[str] = ("TTC", "DAC"),
    gates: Sequence[str] = ("NC", "DAC", "TTC", "EP"),
) -> Optional[float]:
    """Safety reward = the SUM (addition) of ``metrics``, HARD-gated to 0 when
    ANY of the ``gates`` metrics is 0. With the defaults this mirrors EXACTLY
    the safety gating of the planning reward in
    ``eval_tmp/reward_function/scratch3_format_softerrrrrr_dac_ttc_noCoT.py``::

        safe_reward = TTC + DAC                           (range [0, 2])
        safe_reward = 0   if NC == 0 or DAC == 0 or TTC == 0 or EP == 0

    i.e. time-to-collision compliance plus drivable-area compliance --
    HARD-gated to zero whenever any closed-loop safety metric fails: an
    at-fault collision (``NC``), leaving the drivable area (``DAC``), a
    time-to-collision violation (``TTC``), or no ego progress (``EP``).

    Both lists are config-settable via ``RewardWeights.safe_metrics`` /
    ``.safe_gates`` (yaml ``reward.weights.safe_metrics`` / ``safe_gates``;
    see :func:`_canon_pdm_metrics` for the accepted names). OMITTING them
    keeps the (default) behaviour above byte-identical.

    Args:
        pdms_components: per-sample NAVSIM sub-metrics ``{PDMS, NC, DAC, DDC,
            EP, TTC, C}`` (as returned by :func:`pdms_trajectory_reward`).
        metrics: the PDM sub-metric names to ADD into the reward sum
            (default ``("TTC", "DAC")``; canonical names, see
            :func:`_canon_pdm_metrics`).
        gates: the PDM sub-metric names that must ALL be non-zero, otherwise
            the reward is 0 (default ``("NC", "DAC", "TTC", "EP")``).

    Returns:
        The gated reward as a float, or ``None`` when ANY required PDM
        sub-metric (from ``metrics`` or ``gates``) is unavailable (the caller
        then skips this component, so runs without full PDM sub-metrics are
        unaffected).
    """
    if not pdms_components:
        return None
    vals: Dict[str, float] = {}
    for name in tuple(metrics) + tuple(gates):
        if name in vals:
            continue
        try:
            vals[name] = float(pdms_components[name])
        except (KeyError, TypeError, ValueError):
            return None
    if any(vals[g] == 0.0 for g in gates):
        return 0.0
    return float(sum(vals[m] for m in metrics))


# ===========================================================================
# 3. CoT reward: bbox IoU (perception) + future-position prediction correctness
# ===========================================================================

# Strict bbox regex, dependency-free re-implementation of
# ``transfusion.logger.parse_pred_bboxes`` (which pulls in matplotlib). Captures
# the four box coordinates and the optional trailing ``future_position`` / ``label``.
_JSON_BBOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]'
    r'(?:\s*,\s*"(?:future_position|label)"\s*:\s*"([^"]*)")?'
)


@dataclass
class CoTBox:
    """A parsed CoT bounding box with its semantic label (future_position or class)."""

    box: Tuple[float, float, float, float]
    label: str = ""


def parse_cot_boxes(text: Optional[str]) -> List[CoTBox]:
    """Parse strictly-conforming ``bbox_2d`` entries from a CoT string.

    Works for both predicted and ground-truth CoT (identical JSON format). Boxes
    with non-finite or inverted coordinates are dropped, mirroring the strictness
    of the Transfusion visualiser so malformed early-training output scores 0.
    """
    if not text:
        return []
    out: List[CoTBox] = []
    for m in _JSON_BBOX_RE.finditer(text):
        try:
            box = tuple(float(m.group(k)) for k in range(1, 5))
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in box):
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:  # degenerate / inverted -> not conforming
            continue
        out.append(CoTBox(box=box, label=_normalize_label(m.group(5))))
    return out


def _normalize_label(label: Optional[str]) -> str:
    """Canonicalise a future_position / class label for equality comparison."""
    if not label:
        return ""
    # Collapse internal whitespace, lowercase; keep the comma structure
    # ("far, front") which is semantically ordered (distance, direction).
    parts = [p.strip().lower() for p in label.split(",")]
    return ", ".join(p for p in parts if p)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Axis-aligned intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def match_boxes(
    pred: List[CoTBox], gt: List[CoTBox], iou_threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching of predicted to GT boxes by descending IoU.

    Returns a list of ``(pred_idx, gt_idx, iou)`` for pairs with ``iou >=
    iou_threshold``. Greedy matching is a standard, deterministic detection-style
    assignment and needs no SciPy dependency.
    """
    pairs: List[Tuple[float, int, int]] = []
    for pi, pb in enumerate(pred):
        for gi, gb in enumerate(gt):
            iou = bbox_iou(pb.box, gb.box)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)  # highest IoU first
    used_p, used_g = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))
    return matches


def cot_perception_prediction_reward(
    pred_text: Optional[str],
    gt_text: Optional[str],
    *,
    iou_threshold: float = 0.5,
    w_iou: float = 1.0,
    w_pred: float = 1.0,
) -> Dict[str, float]:
    """CoT reward = bbox-IoU detection score + future-position prediction accuracy.

    Detection (``iou_score``): greedily match predicted boxes to GT boxes and sum
    the matched IoUs, normalised by ``max(#GT, #Pred)``. This is a soft F1 — it
    rewards localisation while penalising both missed GT boxes (false negatives)
    and spurious predicted boxes (false positives), so the model cannot game the
    score by emitting many boxes.

    Prediction (``pred_acc``): fraction of GT boxes whose matched prediction has
    the **same** ``future_position`` label, normalised by ``#GT`` so that an
    unmatched (missed) GT box counts as an incorrect prediction.

    ``cot_reward = w_iou * iou_score + w_pred * pred_acc``.

    Returns a dict with the scalar reward plus diagnostics (counts, raw scores).
    """
    pred_boxes = parse_cot_boxes(pred_text)
    gt_boxes = parse_cot_boxes(gt_text)
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)

    if n_gt == 0:
        # No GT boxes to localise. Reward perfectly only when nothing was predicted;
        # otherwise treat spurious boxes as fully wrong (iou_score 0).
        iou_score = 1.0 if n_pred == 0 else 0.0
        pred_acc = 1.0 if n_pred == 0 else 0.0
        reward = w_iou * iou_score + w_pred * pred_acc
        return {
            "cot_reward": float(reward),
            "cot_iou": float(iou_score),
            "cot_pred_acc": float(pred_acc),
            "n_pred": float(n_pred),
            "n_gt": float(n_gt),
            "n_matched": 0.0,
        }

    matches = match_boxes(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
    denom = max(n_gt, n_pred)
    iou_sum = sum(iou for _, _, iou in matches)
    iou_score = float(iou_sum / denom) if denom > 0 else 0.0

    correct = 0
    for pi, gi, _ in matches:
        if pred_boxes[pi].label and pred_boxes[pi].label == gt_boxes[gi].label:
            correct += 1
    pred_acc = float(correct / n_gt)

    reward = w_iou * iou_score + w_pred * pred_acc
    return {
        "cot_reward": float(reward),
        "cot_iou": float(iou_score),
        "cot_pred_acc": float(pred_acc),
        "n_pred": float(n_pred),
        "n_gt": float(n_gt),
        "n_matched": float(len(matches)),
    }


# ===========================================================================
# Prediction reward (CoT ``<prediction>``): detection x future-position accuracy
# ===========================================================================
# Ported from eval_tmp/reward_function/scratch2_format_softerrrrr2_dac_ttc_noCoT.py
# (``prediction_reward`` + its ``<prediction>`` parser), adapted so the ground
# truth is parsed from the dataset's CoT reasoning trace (``gt_cot_text``).


def parse_json(text: Optional[str]) -> Any:
    """Best-effort JSON parse tolerating ```json fences (reference ``parse_json``)."""
    if not text:
        return None
    try:
        s = text.strip()
        lines = s.splitlines()
        if lines and lines[0].startswith("```json"):
            s = "\n".join(lines[1:])
        if "```" in s:
            s = s.split("```")[0]
        return json.loads(s)
    except Exception:
        return None


def _parse_prediction_block(text: Optional[str]) -> Tuple[List[List[float]], List[str], bool]:
    """Parse the ``<prediction>`` block into (boxes, future_positions, format_ok).

    Mirrors the reference ``parse_prediction_answer``: requires each item to be a
    ``{"bbox_2d": [x1,y1,x2,y2], "future_position": "..."}`` dict; a single empty
    ``bbox_2d`` means "no objects" (valid). ``format_ok`` is False when the block
    is missing or malformed (so the caller can penalise bad prediction format).
    """
    m = re.search(r"<prediction>(.*?)</prediction>", text or "", re.DOTALL)
    if not m:
        return [], [], False
    data = parse_json(m.group(1).strip())
    if not isinstance(data, list):
        return [], [], False
    boxes: List[List[float]] = []
    states: List[str] = []
    flag = True
    for item in data:
        if isinstance(item, dict) and set(item.keys()) == {"bbox_2d", "future_position"}:
            bb = item["bbox_2d"]
            if not isinstance(bb, list):
                flag = False
                continue
            if len(bb) == 0 and len(data) == 1:
                return [], [], True  # explicitly "no objects"
            if len(bb) != 4:
                flag = False
                continue
            try:
                boxes.append([float(c) for c in bb])
                states.append(str(item.get("future_position", "unknown")))
            except (ValueError, TypeError):
                flag = False
        else:
            flag = False
    return boxes, states, flag


def prediction_reward(
    pred_text: Optional[str],
    gt_text: Optional[str],
    *,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """CoT prediction reward = future-position accuracy x bbox detection score.

    ``detection = avg_iou * (0.5*precision + 0.5*recall)`` over greedily matched
    boxes; ``state_acc`` = IoU-weighted fraction of matched boxes whose
    ``future_position`` label is correct; ``reward = state_acc * detection``
    (reference ``prediction_reward``). Gated by the ``<prediction>`` FORMAT: a
    missing/malformed block scores 0. A correctly-formatted empty prediction that
    matches an empty ground truth scores 1.
    """
    pred_boxes, pred_states, flag = _parse_prediction_block(pred_text)
    gt_boxes, gt_states, _gt_flag = _parse_prediction_block(gt_text)
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)

    base = {
        "pred_format": 1.0 if flag else 0.0,
        "pred_detection": 0.0,
        "pred_state_acc": 0.0,
        "n_pred": float(n_pred),
        "n_gt": float(n_gt),
        "n_matched": 0.0,
    }

    if not flag:  # bad prediction format -> full penalty
        return {"prediction_reward": 0.0, **base}

    if gt_boxes and gt_states:
        matches = match_boxes(
            [CoTBox(box=tuple(b)) for b in pred_boxes],
            [CoTBox(box=tuple(b)) for b in gt_boxes],
            iou_threshold=iou_threshold,
        )
        tp = len(matches)
        fp, fn = n_pred - tp, n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        avg_iou = (sum(iou for _, _, iou in matches) / tp) if tp > 0 else 0.0
        detection = avg_iou * (0.5 * precision + 0.5 * recall)

        state_score, total_w = 0.0, 0.0
        for pi, gi, iou in matches:
            if pred_states[pi].strip().lower() == gt_states[gi].strip().lower():
                state_score += iou
            total_w += iou
        state_acc = state_score / total_w if total_w > 0 else 0.0

        reward = state_acc * detection
        base.update({
            "pred_detection": float(detection),
            "pred_state_acc": float(state_acc),
            "n_matched": float(tp),
        })
        return {"prediction_reward": float(reward), **base}

    if not gt_boxes and not pred_boxes:  # both empty + valid format -> perfect
        base.update({"pred_detection": 1.0, "pred_state_acc": 1.0})
        return {"prediction_reward": 1.0, **base}

    # GT empty but predicted boxes (or vice-versa) -> wrong
    return {"prediction_reward": 0.0, **base}


def _strict_cot_format_ok(text: Optional[str], *, parse: bool = True, strict_tail: bool = False) -> bool:
    """True only when ``text`` is EXACTLY one clean ``<prediction>[...]</prediction>``.

    Enforces the "strict format gate" (no extra characters anywhere):

    1. ``<prediction>`` and ``</prediction>`` each appear EXACTLY once;
    2. nothing but whitespace BEFORE ``<prediction>``;
    3. the inner content is a SINGLE JSON ARRAY -- the array may optionally be
       wrapped in one matching ```json / ``` code-fence pair (the SFT model's
       native ``<prediction>```json[...]```</prediction> layout), but nothing
       else: no extra wrapping tags, no stray characters;
    4. AFTER ``</prediction>``: v1 (``strict_tail=False``, default) allows any
       amount of WHITESPACE (the CoT is "ended"); v2 (``strict_tail=True``)
       requires the tail to be EXACTLY empty -- not even whitespace. v1's
       whitespace tolerance is the loophole that lets the RL policy pad a valid
       block with reward-neutral spaces/newlines up to the token cap
       (docs/COT_COLLAPSE_ANALYSIS.md §2④); v2 closes it. Naturally-stopped
       (EOT) CoTs end exactly at ``</prediction>`` (tail == "" empirically), so
       ``strict_tail`` rejects only the whitespace-filler overrun.

    With ``parse=False`` the JSON-content check is skipped (only tag/count/head/
    tail strictness), which is useful when only the format shell matters.
    """
    t = text or ""
    o = "<prediction>"
    c = "</prediction>"
    if t.count(o) != 1 or t.count(c) != 1:
        return False
    open_i = t.find(o)
    close_i = t.find(c)
    # 2. head: only whitespace before <prediction>
    if t[:open_i].strip():
        return False
    # 4. tail: only whitespace after </prediction> (the "not yet ended" CoT).
    #    ``strict_tail`` hard-fails ANY non-empty tail (v2 gate); the default
    #    tolerates a whitespace-only tail.
    tail = t[close_i + len(c):]
    if strict_tail:
        if tail != "":
            return False
    elif tail.strip():
        return False
    # 3. inner must be ONE JSON list, optionally code-fence wrapped.
    if parse:
        inner = t[open_i + len(o):close_i].strip()
        # Tolerate a single matching ```json / ``` code-fence pair wrapping the
        # list (the format the SFT checkpoint actually emits). Strip it so the
        # body below is checked as a bare list. Any OTHER extra text between the
        # tags still fails json.loads / the list check below.
        if "```" in inner:
            m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", inner, re.DOTALL)
            if m is None:
                return False  # malformed / unmatched fence -> deviation
            inner = m.group(1).strip()
        if not inner.startswith("["):
            return False
        try:
            data = json.loads(inner)
        except Exception:
            return False
        if not isinstance(data, list) or isinstance(data, str):
            return False
    return True


# ===========================================================================
# Combined reward: prediction + planning
# ===========================================================================


def _default_trajectory(
    pred_traj: Optional[np.ndarray],
    n_steps: int = 8,
    dim: int = 3,
) -> np.ndarray:
    """Return a concrete ``[n_steps, dim]`` trajectory, defaulting to an all-zero
    ``[0, 0, 0] * n_steps`` (8 frames by default) whenever the predicted action is
    invalid: ``None``, non-finite (NaN/inf), or a wrong shape.

    The default is used as the final result for BOTH training and testing so a
    missing/collapsed action never crashes or silently drops the planning reward,
    and CoT outcomes remain fully independent of the action reward.
    """
    zeros = np.zeros((n_steps, dim), dtype=np.float64)
    if pred_traj is None:
        return zeros
    try:
        arr = np.asarray(pred_traj, dtype=np.float64)
    except Exception:
        return zeros
    if arr.ndim == 1:
        # [T*dim] -> [T, dim]; only valid when it forms exactly n_steps rows
        # (an 8-frame action). Anything else is a collapsed/invalid action.
        if arr.size != n_steps * dim:
            return zeros
        arr = arr.reshape(n_steps, dim)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0:
        return zeros
    if not np.all(np.isfinite(arr)):
        return zeros
    if arr.shape[0] < n_steps:
        pad = np.zeros((n_steps, arr.shape[1]), dtype=np.float64)
        pad[:arr.shape[0]] = arr
        return pad
    return arr[:n_steps]


def compute_total_reward(
    *,
    pred_traj: Optional[np.ndarray] = None,
    gt_traj: Optional[np.ndarray] = None,
    pred_cot_text: Optional[str] = None,
    gt_cot_text: Optional[str] = None,
    pdms: Optional[float] = None,
    pdms_components: Optional[Dict[str, float]] = None,
    epdms_components: Optional[Dict[str, float]] = None,
    weights: Optional[RewardWeights] = None,
) -> Dict[str, float]:
    """Two-part Transfusion RL reward: PREDICTION (CoT) + PLANNING (trajectory).

    ``score = w.prediction * prediction_reward
              + w.planning * (w.soft_l2 * soft_l2_reward
                              + w.endpoint * endpoint_reward
                              + w.pdms * PDMS
                              + w.epdms * EPDMS
                              + w.safe * safe_reward)``.

    * PREDICTION comes from the CoT text (``pred_cot_text`` vs. the GT reasoning
      trace ``gt_cot_text``).
    * PLANNING comes from the diffusion **trajectory** (``pred_traj`` vs.
      ``gt_traj`` for the soft L2 and the FDE endpoint reward, and the
      pre-computed NAVSIM ``pdms`` / ``pdms_components`` for the closed-loop
      score and the DAC/TTC safety reward, cf. :func:`safe_reward`). PDMS is
      ALWAYS computed (train and val); ``weights.pdm_ep_weight`` only sets the
      EP weight INSIDE that PDMS aggregation (None/5.0 = standard) and never
      touches EPDMS. EPDMS is the OFFICIAL NAVSIM v2 metric, supplied
      pre-computed as ``epdms_components`` (``{"EPDMS", "NC", ..., "HC"}``
      plus ``"EC"`` for frames chained to a previous frame — see
      ``recipe.transfusion.epdms_eval``; against the v2 metric cache) only when
      the run asks for it (:func:`epdms_enabled`); it is logged as ``epdms``
      (plus ``epdms_<sub>``) and enters the (training) planning block via
      ``weights.epdms``. Validation reports it read-only — the val score /
      ``pdms_reward`` keep the standard composition either way.

    The output dict is flat and surfaces EVERY component + PDM sub-metric (``PDMS``
    / ``NC`` / ``DAC`` / ``DDC`` / ``EP`` / ``TTC`` / ``C``) and the run's
    coefficients (``coef_*``), so all of them appear in the SwanLab reward panel
    (train and val).

    COT AND ACTION ARE INDEPENDENT. The CoT format (``<prediction>[...]```) only
    ever affects the PREDICTION (CoT) term and its ``format_penalty``; it NEVER
    zeroes the PLANNING (action/trajectory) reward. A malformed CoT -- strict-format
    deviation or the multi-prediction "copy & pad" hack -- simply yields
    ``prediction_reward = 0`` (plus ``format_penalty``), while the planning signal
    is still computed from the trajectory as usual.

    INVALID ACTION DEFAULTS TO ZEROS. Whatever ``pred_traj`` arrives (``None`` /
    NaN / wrong shape, e.g. a collapsed generation), it is replaced by a default
    all-zero trajectory ``[0, 0, 0] * 8`` (8 frames) via :func:`_default_trajectory`
    before any planning sub-reward is computed -- identical behaviour in training
    and testing, so a missing action never crashes or silently drops the planning
    signal.
    """
    w = weights or RewardWeights()
    out: Dict[str, float] = {}

    # ---- PLANNING (trajectory) : soft L2 + endpoint (FDE) + PDMS ----
    # Computed FIRST and ALWAYS, INDEPENDENT of the CoT. Any invalid/absent action
    # is defaulted to an all-zero [0,0,0]*8 trajectory so a planning score is
    # always produced for every sample (train + val).
    pred_traj_eff = _default_trajectory(pred_traj)
    planning_r = 0.0
    if gt_traj is not None:
        sl2 = soft_l2_reward(
            pred_traj_eff, gt_traj,
            mode=w.soft_l2_mode, scale=w.soft_l2_scale, use_endpoint=w.soft_l2_endpoint,
        )
        out.update(sl2)
        planning_r += w.soft_l2 * sl2["soft_l2_reward"]

        # Endpoint (FDE) reward — same trajectory source, scores only the final
        # waypoint; tightness driven by w.endpoint_thresholds / _scores, and the
        # distance metric by w.endpoint_metric ("l1" = scratch2_softerrrrr2).
        ep = endpoint_reward(
            pred_traj_eff, gt_traj,
            thresholds=w.endpoint_thresholds, scores=w.endpoint_scores,
            metric=w.endpoint_metric,
        )
        out.update(ep)
        planning_r += w.endpoint * ep["endpoint_reward"]

    pdms_val: Optional[float] = None
    if pdms_components and "PDMS" in pdms_components:
        # The scorer's PDMS with the OFFICIAL EP weight …
        std_pdms = float(pdms_components["PDMS"])
        # … unless the run overrides the EP weight INSIDE the PDMS aggregation
        # (RewardWeights.pdm_ep_weight), in which case recompose it from the
        # sub-metrics and fall back to the scorer value when they are missing.
        if w.pdm_ep_weight is not None and float(w.pdm_ep_weight) != PDMS_STD_EP_WEIGHT:
            pdms_val = reweigh_pdms_ep(pdms_components, ep_weight=float(w.pdm_ep_weight))
        if pdms_val is None:
            pdms_val = std_pdms
    elif pdms is not None and np.isfinite(pdms):
        pdms_val = float(pdms)

    # OFFICIAL NAVSIM v2 EPDMS (pre-computed against the v2 metric cache and
    # passed in as ``epdms_components``; see recipe.transfusion.epdms_eval).
    # Logged always; joins the (training) planning block via ``w.epdms``.
    epdms_val: Optional[float] = None
    if epdms_components:
        try:
            epdms_val = float(epdms_components.get("EPDMS"))
        except (TypeError, ValueError):
            epdms_val = None
    if epdms_val is not None:
        out["epdms"] = epdms_val
        out["epdms_reward"] = epdms_val
        planning_r += w.epdms * epdms_val
        # surface every EPDMS sub-metric (EPDMS/NC/DAC/DDC/TLC/EP/TTC/LK/HC/EC)
        # for logging (the aggregate itself is logged separately above).
        for k, v in epdms_components.items():
            if str(k).upper() == "EPDMS":
                continue
            try:
                out[f"epdms_{k}"] = float(v)
            except (TypeError, ValueError):
                pass

    if pdms_val is not None:
        out["pdms_reward"] = pdms_val
        planning_r += w.pdms * pdms_val
    # surface every PDM sub-metric (PDMS/NC/DAC/DDC/EP/TTC/C) for logging.
    if pdms_components:
        for k, v in pdms_components.items():
            try:
                out[f"pdms_{k}"] = float(v)
            except (TypeError, ValueError):
                pass

    # ---- ABNORMAL REVERSE (backing-up) trajectory gate ----
    # ``reverse_travel_m`` = total backward (negative-x) travel of the planned
    # trajectory in metres; ALWAYS logged (pred is defaulted above, so it is
    # always computable). The GT reverse travel EXEMPTS scenes where backing
    # up is legitimate (parking / reverse-gear). The gate itself is applied
    # to the WHOLE planning block below, after all sub-terms are accumulated.
    rev_travel = reverse_travel(pred_traj_eff)
    out["reverse_travel_m"] = rev_travel
    has_reverse = rev_travel > 0.0
    rev_gt = reverse_travel(gt_traj) if gt_traj is not None else 0.0

    # DAC/TTC safety reward (scratch3-style gating): with the defaults
    # ``TTC + DAC``, hard gated to 0 when NC / DAC / TTC / EP is 0; the summed
    # (``safe_metrics``) and gated (``safe_gates``) PDM sub-metrics are
    # configurable (see ``safe_reward``). Always surfaced for logging;
    # contributes to planning only via the ``safe`` sub-weight.
    safe_r = safe_reward(pdms_components, metrics=w.safe_metrics, gates=w.safe_gates)
    if safe_r is not None:
        out["safe_reward"] = safe_r
        planning_r += w.safe * safe_r

    # scratch2-style whole-planning safety gate: a failed gate (safe_r == 0.0 can
    # only happen via the gate — a passing gate means TTC > 0 and DAC > 0) zeroes
    # the ENTIRE planning block, endpoint / soft-L2 / PDMS included.
    if w.safe_gates_planning and safe_r is not None and safe_r == 0.0:
        planning_r = 0.0

    # ABNORMAL REVERSE master gate (``reverse_gate``, OFF by default): ANY
    # backward travel (``has_reverse``) hard-zeroes the WHOLE planning block.
    if w.reverse_gate and has_reverse:
        planning_r = 0.0
        out["reverse_gated"] = 1.0

    # ABNORMAL REVERSE gate (OFF by default — None keeps current experiments
    # identical): enable with a threshold in metres (e.g. 0.3); a rollout whose
    # predicted trajectory reverses MORE than this has its WHOLE planning
    # reward hard-zeroed (as if a safety gate had failed). Samples whose GT
    # itself reverses past the threshold are exempt (legitimate backing-up).
    if (
        w.reverse_gate_m is not None
        and rev_travel > float(w.reverse_gate_m)
        and rev_gt <= float(w.reverse_gate_m)
    ):
        planning_r = 0.0
        out["reverse_gated"] = 1.0

    out["planning_reward"] = float(planning_r)

    # ---- PREDICTION (CoT) ----
    # When no CoT was generated (train_only_action or empty rollout), skip the
    # entire prediction block: no format gate, no format_penalty — the score is
    # purely the planning (action) reward.
    _pred_text = pred_cot_text or ""
    pred: Dict[str, float] = {"prediction_reward": 0.0, "pred_format": 0.0}
    if not _pred_text:
        # No CoT text -> no prediction reward, no format penalty.
        prediction_r = 0.0
        format_penalty = 0.0
    elif w.strict_format_gate and not _strict_cot_format_ok(
        pred_cot_text, strict_tail=w.strict_format_gate_v2
    ):
        out["strict_format"] = 1.0
        out.update(pred)
        prediction_r = 0.0
        format_penalty = float(w.format_penalty)
        out["format_penalty"] = format_penalty
    elif _pred_text.count("<prediction>") > 1 or _pred_text.count("</prediction>") > 1:
        out["multi_pred"] = 1.0
        out.update(pred)
        prediction_r = 0.0
        format_penalty = float(w.format_penalty)
        out["format_penalty"] = format_penalty
    else:
        pred = prediction_reward(pred_cot_text, gt_cot_text, iou_threshold=w.iou_threshold)
        out.update(pred)
        prediction_r = pred["prediction_reward"]
        format_penalty = 0.0
        if pred.get("pred_format", 1.0) == 0.0:
            format_penalty = float(w.format_penalty)
            out["format_penalty"] = format_penalty

    # ---- per-head reward components for split GRPO advantage ----
    # The score mixes a PLANNING (trajectory) signal and a PREDICTION (CoT)
    # signal. Naively feeding this single scalar to every head lets the noisier / 
    # higher-variance component dominate BOTH heads' group-relative advantage (the
    # action head chases imitation of the GT, the CoT head drifts with trajectory
    # outcomes). Splitting here lets each head normalize ONLY the reward it can
    # actually influence:
    #   * action head -> the planning block (soft_L2 + endpoint + PDMS), as
    #     ``reward_action``;
    #   * CoT head    -> the prediction reward + format_penalty, as
    #     ``reward_cot``. The format penalty is a consequence of the CoT TEXT, so
    #     it belongs to the CoT signal.
    reward_action = float(w.planning * planning_r)
    reward_cot = float(w.prediction * prediction_r + format_penalty)
    out["reward_action"] = reward_action
    out["reward_cot"] = reward_cot

    out["score"] = float(
        w.prediction * prediction_r + w.planning * planning_r
        + format_penalty
    )
    out.update(w.as_coef_dict())
    return out


__all__ = [
    "RewardWeights",
    "epdms_enabled",
    "trajectory_l2",
    "l2_trajectory_reward",
    "soft_l2_reward",
    "endpoint_reward",
    "reverse_travel",
    "pdms_trajectory_reward",
    "reweigh_pdms_ep",
    "safe_reward",
    "parse_cot_boxes",
    "parse_json",
    "bbox_iou",
    "match_boxes",
    "cot_perception_prediction_reward",
    "prediction_reward",
    "compute_total_reward",
    "CoTBox",
]
