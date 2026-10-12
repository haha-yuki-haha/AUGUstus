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

"""Trajectory-aware reward manager for Transfusion RL.

This is an *optional* alternative to the native ``custom_reward_function`` path
(:mod:`recipe.transfusion.reward_function`). It is useful because, unlike the
generic scorer signature ``(data_source, solution_str, ground_truth, ...)``, a
reward manager receives the full :class:`~verl.DataProto` and can therefore read
the continuous predicted **trajectory** directly from ``non_tensor_batch`` (as
emitted by :mod:`recipe.transfusion.rollout`), rather than relying on it being
threaded through ``extra_info``.

Wire it in (kept out of the untouched RL section)::

    reward:
      reward_manager:
        source: importlib
        name: TransfusionRewardManager
        module:
          path: pkg://recipe.transfusion.reward_manager
          name: TransfusionRewardManager
      custom_reward_function:
        # weights / metric cache are read from here (shared with the scorer path)
        reward_kwargs:
          weights: {l2: 1.0, pdms: 1.0, cot: 1.0, cot_iou: 1.0, cot_pred: 1.0}
          metric_cache_path: /path/to/navsim/metric_cache_test

Every RL hyper-parameter is untouched; this only changes how the scalar reward
is produced.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

import logging

logger = logging.getLogger(__name__)

# --- robust import of recipe-local helpers (works for pkg:// and file paths) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # verl_transfusion/
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from recipe.transfusion import transfusion_reward as tr
    from recipe.transfusion.reward_function import resolve_metric_cache_path, score_sample, strip_weights_for_split
except Exception:  # pragma: no cover - fallback for file-path loading
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    import transfusion_reward as tr  # type: ignore
    from reward_function import resolve_metric_cache_path, score_sample, strip_weights_for_split  # type: ignore

# _cfg_get: tiny OmegaConf/dict-safe walker. Reward managers receive the FULL
# verl config; recipe keys may be plain dicts OR OmegaConf nodes, so a
# getattr("get")-based walk with a hard ``default`` never raises.


def _cfg_get(node: Any, *keys: str, default: Any = None) -> Any:
    """Safely walk a (possibly ``DictConfig``) tree, returning ``default`` if absent."""
    cur = node
    for k in keys:
        if cur is None:
            return default
        getter = getattr(cur, "get", None)
        cur = getter(k, None) if callable(getter) else None
    return default if cur is None else cur


@register("TransfusionRewardManager")
class TransfusionRewardManager(RewardManagerBase):
    """Combine L2 + PDMS + CoT(IoU + future-position) into the per-sample reward."""

    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        # Shared config block (also used by the custom_reward_function path).
        reward_kwargs = _cfg_get(config, "reward", "custom_reward_function", "reward_kwargs", default={}) or {}
        self._weights = tr.RewardWeights.from_dict(
            reward_kwargs.get("weights") if hasattr(reward_kwargs, "get") else None
        )
        # PDMS cache is resolved PER SAMPLE in run_single via the shared
        # resolve_metric_cache_path (dataset-stamped split-correct path first, then
        # the reward_kwargs train/test caches).
        self._reward_kwargs = reward_kwargs

        # ---- Transfusion-style per-sample logging (disk + best-effort SwanLab) ----
        # The actual logging lives in recipe.transfusion.sample_logging (a
        # process-wide singleton configured by the TRANSFUSION_* env vars set in
        # the recipe config's ray_kwargs). As a fallback for runs that don't export
        # them, publish the output dir from the config if it isn't already set.
        output_dir = _cfg_get(config, "trainer", "default_local_dir", default=None)
        if output_dir and not os.environ.get("TRANSFUSION_OUTPUT_DIR"):
            os.environ["TRANSFUSION_OUTPUT_DIR"] = str(output_dir)


    @staticmethod
    def _get(nt: Dict[str, Any], *names: str, default: Any = None) -> Any:
        for n in names:
            if n in nt and nt[n] is not None:
                return nt[n]
        return default

    @staticmethod
    def _as_array(x: Any) -> Optional[np.ndarray]:
        if x is None:
            return None
        try:
            arr = np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        return arr if arr.size else None

    async def run_single(self, data: DataProto) -> dict:
        # verl stamps the batch with these in ray_trainer.py's fit()/_validate();
        # capture BEFORE slicing so the sample logger can tell a training-step
        # rollout apart from an actual validation pass (docs/VAL_LOGGING.md).
        meta_info = getattr(data, "meta_info", None) or {}
        is_validate = bool(meta_info.get("validate", False))
        global_step = meta_info.get("global_steps")
        # For multi-sequence outputs, score the last sequence (verl convention).
        data = data[-1:]
        data_item = data[0]
        nt = data_item.non_tensor_batch

        # Decode the generated CoT text from the response tokens.
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_len = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_ids = response_ids[:valid_len]
        pred_cot_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_ids, skip_special_tokens=True)
        )

        ground_truth = nt.get("reward_model", {}) or {}
        if isinstance(ground_truth, dict):
            gt = ground_truth.get("ground_truth", ground_truth)
        else:
            gt = {}
        gt = gt if isinstance(gt, dict) else {}
        extra_info = nt.get("extra_info", {}) or {}

        pred_traj = self._as_array(self._get(nt, "pred_trajectory") or extra_info.get("pred_trajectory"))
        gt_traj = self._as_array(gt.get("gt_trajectory"))
        gt_cot_text = gt.get("gt_cot_text")
        frame_id = gt.get("frame_id") or self._get(nt, "frame_id", "_frame_id")
        metric_cache_path = resolve_metric_cache_path(gt.get("metric_cache_path"), self._reward_kwargs)
        pdms_components = extra_info.get("pdms_components")
        if pdms_components is None and extra_info.get("pdms") is not None:
            pdms_components = {"PDMS": float(extra_info["pdms"])}

        # Per-split weights: the manager's base weights are the TRAIN-side
        # knobs; validation samples get the *_val / epdms variants stripped
        # in (see strip_weights_for_split), so val rollouts are scored with
        # the same weighting the SFT-style validation path uses (the shared
        # scorer reads the identical RewardWeights from both paths, keeping
        # val_* and train_* curves directly comparable).
        weights = tr.RewardWeights.from_dict(
            strip_weights_for_split(
                {f.name: getattr(self._weights, f.name) for f in dataclasses.fields(self._weights)},
                is_validate=is_validate,
            )
        )
        epdms_components = extra_info.get("epdms_components")
        if not tr.epdms_enabled(weights, is_validate=is_validate):
            epdms_components = None

        result = await self.loop.run_in_executor(
            None,
            lambda: score_sample(
                pred_traj=pred_traj,
                gt_traj=gt_traj,
                pred_cot_text=pred_cot_text,
                gt_cot_text=gt_cot_text,
                frame_id=frame_id,
                metric_cache_path=metric_cache_path,
                weights=weights,
                pdms_components=pdms_components,
                epdms_components=epdms_components,
            ),
        )

        # Best-effort Transfusion-style logging (never breaks reward computation).
        try:
            self._maybe_log_sample(
                nt=nt, gt=gt, pred_traj=pred_traj, gt_traj=gt_traj,
                pred_cot_text=pred_cot_text, gt_cot_text=gt_cot_text, result=result,
                is_validate=is_validate, step=global_step,
            )
        except Exception as e:  # pragma: no cover - logging must never break training
            logger.warning("Transfusion sample logging failed: %s", e)

        return {"reward_score": float(result["score"]), "reward_extra_info": result}

    # ---- logging helpers (faithful to Transfusion's per-sample logging) -------
    def _maybe_log_sample(self, *, nt, gt, pred_traj, gt_traj, pred_cot_text, gt_cot_text, result, is_validate=False, step=None):
        """Feed one sample to the process-wide Transfusion sample logger."""
        from recipe.transfusion.sample_logging import log_sample_best_effort

        raw_frames = self._get(nt, "_raw_video_frames")
        ti = self._get(nt, "transfusion_inputs")
        if raw_frames is None and isinstance(ti, dict):
            raw_frames = ti.get("_raw_video_frames")
        log_sample_best_effort({
            "text": self._decode_prompt(nt),
            "gt_traj": gt_traj,
            "pred_traj": pred_traj,
            "cot": pred_cot_text,
            "gt_cot": gt_cot_text,
            "metrics": {k: v for k, v in result.items() if isinstance(v, (int, float))},
            "raw_frames": raw_frames,
        }, is_validate=is_validate, step=step)

    def _decode_prompt(self, nt: Dict[str, Any]) -> str:
        """Decode the prompt text from transfusion_inputs (best-effort)."""
        ti = self._get(nt, "transfusion_inputs")
        try:
            if isinstance(ti, dict) and ti.get("input_ids") is not None:
                ids = ti["input_ids"]
                ids = ids[0] if hasattr(ids, "ndim") and getattr(ids, "ndim", 1) > 1 else ids
                return self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            pass
        return ""


__all__ = ["TransfusionRewardManager"]
