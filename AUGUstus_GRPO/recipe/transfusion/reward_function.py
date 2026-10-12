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

"""verl ``custom_reward_function`` hook for Transfusion.

Wire it in via the recipe config (kept out of the untouched RL section)::

    reward:
      custom_reward_function:
        path: pkg://recipe.transfusion.reward_function
        name: compute_score

``compute_score`` has the exact signature verl expects
(``data_source, solution_str, ground_truth, extra_info=None, **kwargs``) and
returns a dict whose ``"score"`` key is the scalar reward. The remaining keys are
surfaced by verl as ``reward_extra_info`` for logging.

Because the predicted **trajectory** is a continuous tensor (not part of the
decoded ``solution_str`` token string), it must be carried in ``extra_info``
under ``"pred_trajectory"`` (the :mod:`recipe.transfusion.rollout` populates it).
When it is absent, only the CoT reward (computed from ``solution_str`` vs. the
ground-truth CoT text) contributes — so this hook degrades gracefully.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# --- robust import of the framework-free reward maths (works for pkg:// and file paths) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # verl_transfusion/
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:  # preferred: imported as part of the recipe package
    from recipe.transfusion import transfusion_reward as tr
except Exception:  # fallback: module loaded standalone by file path
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    import transfusion_reward as tr  # type: ignore


def _to_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            return None
    arr = np.asarray(x, dtype=np.float32)
    return arr if arr.size else None


def _coerce_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    """Ground-truth payload may arrive as a dict or a JSON string."""
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
            if isinstance(obj, dict):
                return obj
            return {"gt_cot_text": ground_truth}
        except json.JSONDecodeError:
            return {"gt_cot_text": ground_truth}
    return {}


def strip_weights_for_split(weight_cfg: Dict[str, Any], *, is_validate: bool) -> Dict[str, Any]:
    """Return the split-correct view of a raw weights dict (config / GT-carried).

    Used by EVERY reward entry point (this scorer, the optional reward manager,
    and the SFT-style validator) so the train/val split rule lives in ONE place:

    * TRAINING — the validation-only key (``val_epdms``) is stripped; the
      ``epdms`` (official NAVSIM v2 EPDMS) term applies normally, and
      ``pdm_ep_weight`` (the EP weight INSIDE the PDMS aggregation) reshapes
      the ``pdms`` term.
    * VALIDATION — the EPDMS planning TERM (``epdms``) is stripped so the val
      score keeps the standard composition. ``pdm_ep_weight`` stays — it
      belongs to PDMS (ALWAYS computed), never to EPDMS. EPDMS itself is only
      computed + logged when ``val_epdms: true`` (report-only; see
      ``tr.epdms_enabled``); with ``val_epdms: false`` validation is exactly
      the PDMS pipeline and the v2 metric caches are never loaded.
    """
    cfg = dict(weight_cfg)
    if is_validate:
        cfg.pop("epdms", None)
        return cfg
    cfg.pop("val_epdms", None)
    return cfg


def resolve_metric_cache_path(per_sample: Optional[str], reward_kwargs: Any) -> Optional[str]:
    """The NAVSIM PDMS metric cache for one sample — resolved the SAME way in both
    reward paths (this scorer and the reward manager), so the rule lives in ONE place.

    The dataset stamps every sample with its OWN split-correct cache
    (``data.metric_cache_train`` in training, ``data.metric_cache_test`` in
    validation), so ``per_sample`` is authoritative. Only when it is absent do we
    fall back to the reward_kwargs caches — train first, since PDMS during training
    drives the reward.
    """
    def _get(key: str) -> Optional[str]:
        getter = getattr(reward_kwargs, "get", None)
        return getter(key) if callable(getter) else None

    return per_sample or _get("metric_cache_train") or _get("metric_cache_test")


def resolve_metric_cache_v2_path(per_sample: Optional[str], reward_kwargs: Any) -> Optional[str]:
    """The NAVSIM **v2** metric cache used for the official EPDMS — same rule as
    :func:`resolve_metric_cache_path`, one level up (``*_v2`` keys).

    The dataset stamps every sample with its split-correct v2 cache
    (``data.metric_cache_train_v2`` in training, ``data.metric_cache_test_v2``
    in validation), so ``per_sample`` is authoritative; the reward_kwargs v2
    caches are only the config-level fallback. Used ONLY when EPDMS is actually
    computed (``tr.epdms_enabled``) — otherwise the v2 caches are never touched.
    """
    def _get(key: str) -> Optional[str]:
        getter = getattr(reward_kwargs, "get", None)
        if callable(getter):
            return getter(key)
        return None

    return per_sample or _get("metric_cache_train_v2") or _get("metric_cache_test_v2")


def score_sample(
    *,
    pred_traj: Optional[np.ndarray],
    gt_traj: Optional[np.ndarray],
    pred_cot_text: Optional[str],
    gt_cot_text: Optional[str],
    frame_id: Optional[str],
    metric_cache_path: Optional[str],
    weights: tr.RewardWeights,
    pdms_components: Optional[Dict[str, float]] = None,
    epdms_components: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute the combined reward for one sample.

    Shared by :func:`compute_score` and
    :class:`recipe.transfusion.reward_manager.TransfusionRewardManager`. If
    ``pdms_components`` (the per-sample NAVSIM PDM sub-metric dict) is not supplied
    and the PDMS weight is active, it is computed on-demand for this single sample
    (slow path); batched PDMS is preferred and handled by the reward manager.
    ``epdms_components`` (the per-sample OFFICIAL NAVSIM v2 EPDMS dict) has NO
    on-demand path: it must be batch-precomputed by the caller (spawning the v2
    scorer pool for a single sample would be far too slow), and only when
    ``tr.epdms_enabled`` says this split computes EPDMS at all.
    """
    if (
        pdms_components is None
        and weights.planning != 0.0
        and weights.pdms != 0.0
        and pred_traj is not None
        and frame_id
        and metric_cache_path
    ):
        scored = tr.pdms_trajectory_reward(
            pred_traj[None] if pred_traj.ndim == 2 else pred_traj,
            [frame_id],
            metric_cache_path,
        )
        pdms_components = scored[0] if scored else None

    return tr.compute_total_reward(
        pred_traj=pred_traj,
        gt_traj=gt_traj,
        pred_cot_text=pred_cot_text,
        gt_cot_text=gt_cot_text,
        pdms_components=pdms_components,
        epdms_components=epdms_components,
        weights=weights,
    )


def compute_score(
    data_source: Optional[str] = None,
    solution_str: Optional[str] = None,
    ground_truth: Any = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """verl reward entry point — returns ``{"score": float, ...diagnostics}``.

    Args:
        data_source: dataset tag (unused; single reward for all Transfusion data).
        solution_str: decoded model response == generated CoT text.
        ground_truth: dict (or JSON) with ``gt_trajectory``, ``gt_cot_text``,
            ``frame_id``, ``metric_cache_path`` and optional ``weights``.
        extra_info: per-sample dict; may carry ``pred_trajectory`` (from rollout),
            ``metric_cache_path``, a precomputed ``pdms`` score /
            ``pdms_components``, and a batch-precomputed ``epdms_components``
            (official NAVSIM v2 EPDMS; honoured only when ``tr.epdms_enabled``
            says this split computes EPDMS).
        **kwargs: optional ``reward_kwargs`` from the config (e.g. global weights).
    """
    extra_info = extra_info or {}
    gt = _coerce_ground_truth(ground_truth)

    is_validate = bool(extra_info.get("is_validate", False))
    weight_cfg = strip_weights_for_split(
        {**(kwargs.get("weights") or {}), **(gt.get("weights") or {})},
        is_validate=is_validate,
    )
    weights = tr.RewardWeights.from_dict(weight_cfg)
    epdms_components = extra_info.get("epdms_components")
    if not tr.epdms_enabled(weights, is_validate=is_validate):
        epdms_components = None

    pred_traj = _to_array(extra_info.get("pred_trajectory"))
    gt_traj = _to_array(gt.get("gt_trajectory"))
    pred_cot_text = solution_str
    gt_cot_text = gt.get("gt_cot_text")
    frame_id = gt.get("frame_id") or extra_info.get("frame_id")
    metric_cache_path = resolve_metric_cache_path(
        gt.get("metric_cache_path") or extra_info.get("metric_cache_path"), kwargs
    )
    # Per-sample NAVSIM PDM sub-metric dict (PDMS/NC/DAC/DDC/EP/TTC/C), batch-
    # precomputed by the sync reward manager; fall back to a bare float pdms.
    pdms_components = extra_info.get("pdms_components")
    if pdms_components is None and extra_info.get("pdms") is not None:
        pdms_components = {"PDMS": float(extra_info["pdms"])}

    result = score_sample(
        pred_traj=pred_traj,
        gt_traj=gt_traj,
        pred_cot_text=pred_cot_text,
        gt_cot_text=gt_cot_text,
        frame_id=frame_id,
        metric_cache_path=metric_cache_path,
        weights=weights,
        pdms_components=pdms_components,
        epdms_components=epdms_components,
    )

    # Best-effort Transfusion-style per-sample logging (disk + SwanLab). Logs the
    # leading samples of each window; recipe.transfusion.sync_driver flushes the
    # accumulated component-reward means once per whole batch (one training step,
    # or one full validation pass) — see docs/VAL_LOGGING.md. Never raises —
    # logging must not affect the reward.
    try:
        from recipe.transfusion.sample_logging import log_sample_best_effort

        log_sample_best_effort({
            "text": extra_info.get("prompt_text"),
            "gt_traj": gt_traj,
            "pred_traj": pred_traj,
            "cot": pred_cot_text,
            "gt_cot": gt_cot_text,
            "metrics": {k: v for k, v in result.items() if isinstance(v, (int, float))},
            # Only present for validation samples (TransfusionRLDataset drops it for
            # train to save RAM); lets the logger show "what the model is looking at"
            # (last video frame [+ bbox overlay]), mirroring supervised Transfusion.
            "raw_frames": extra_info.get("raw_frames"),
            # GRPO group id (sync_driver._attach_rm_scores): buffers the rollout.n
            # siblings per prompt so flush_metric_means can render the in-group
            # trajectory overlay plots (sample_logging).
            "group_key": extra_info.get("group_key"),
        }, is_validate=bool(extra_info.get("is_validate", False)), step=extra_info.get("global_step"))
    except Exception:
        pass

    return result


__all__ = [
    "compute_score",
    "score_sample",
    "strip_weights_for_split",
    "resolve_metric_cache_path",
    "resolve_metric_cache_v2_path",
]
