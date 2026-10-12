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

"""Best-checkpoint-by-metric for the Transfusion RL recipe.

verl saves checkpoints periodically (``trainer.save_freq``) and rotates them
(``trainer.max_actor_ckpt_to_keep``) but has **no native "keep the best by a
validation metric"** facility. This module adds it the same way supervised
Transfusion does (``AUGUstus_SFT/train_qwen3p5.py``: a ``best_metric`` such as
``PDMS`` higher-is-better / ``L2`` lower-is-better that is monitored every
validation):

* :class:`BestMetricTracker` — framework-light, unit-tested selection logic:
  resolve a short metric name against verl's ``val-core/.../<name>/mean`` keys and
  decide whether the latest validation improved it.
* :class:`TransfusionRayPPOTrainer` — a thin :class:`RayPPOTrainer` subclass that
  overrides ``_validate`` to, after each validation, snapshot the current
  checkpoint into ``<output>/best_<metric>/`` whenever a tracked metric improves
  (plus a ``best.json`` pointer). Injected without editing verl core via
  :mod:`recipe.transfusion.train` (``run_ppo(task_runner_class=...)``).

All of this is **opt-in**: with ``trainer.best_metrics`` empty/unset the subclass
behaves exactly like the stock trainer.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framework-light selection logic (unit-tested)
# ---------------------------------------------------------------------------
_HIGHER_HINTS = ("pdms", "reward", "acc", "score", "iou", "ep", "ttc", "dac", "nc")
_LOWER_HINTS = ("l2", "ade", "fde", "loss", "error", "dist")


def guess_higher_is_better(name: str) -> bool:
    """Heuristic direction for a metric name (PDMS↑, L2↓), as in Transfusion."""
    n = name.lower()
    for h in _LOWER_HINTS:
        if h in n:
            return False
    for h in _HIGHER_HINTS:
        if h in n:
            return True
    return True  # default: higher is better


def find_metric_value(metrics: Dict[str, Any], name: str) -> Optional[float]:
    """Resolve a short metric ``name`` against a verl validation-metrics dict.

    verl logs validation metrics as ``val-core/<data_source>/<name>/<agg>`` (e.g.
    ``val-core/navsim_transfusion/pdms_reward/mean``). This matches, in order:
    exact key, ``.../<name>/mean`` / ``.../<name>`` suffix, then case-insensitive
    substring — so a user can simply write ``best_metric: pdms_reward`` (or
    ``PDMS``).
    """
    if name in metrics:
        return _as_float(metrics[name])
    low = name.lower()
    for k, v in metrics.items():
        kl = k.lower()
        if kl.endswith(f"/{low}/mean") or kl.endswith(f"/{low}"):
            return _as_float(v)
    for k, v in metrics.items():
        if low in k.lower():
            return _as_float(v)
    return None


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


class BestMetricTracker:
    """Track the best value of a single validation metric across steps."""

    def __init__(self, name: str, higher_is_better: Optional[bool] = None):
        self.name = name
        self.higher_is_better = (
            guess_higher_is_better(name) if higher_is_better is None else bool(higher_is_better)
        )
        self.best_value: Optional[float] = None
        self.best_step: int = -1

    def update(self, metrics: Dict[str, Any], step: int) -> bool:
        """Return ``True`` (and record) if ``metrics`` improves the tracked metric."""
        cur = find_metric_value(metrics, self.name)
        if cur is None:
            return False
        if self.best_value is None:
            improved = True
        elif self.higher_is_better:
            improved = cur > self.best_value
        else:
            improved = cur < self.best_value
        if improved:
            self.best_value = cur
            self.best_step = step
        return improved

    @property
    def safe_name(self) -> str:
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in self.name)


def parse_best_metrics(spec: Any) -> List[BestMetricTracker]:
    """Build trackers from ``trainer.best_metrics`` config.

    Accepts: ``None`` -> []; a string ``"pdms_reward"``; a list of strings; or a
    list of ``{name, higher_is_better}`` dicts.
    """
    if not spec:
        return []
    if isinstance(spec, str):
        spec = [spec]
    out: List[BestMetricTracker] = []
    for item in spec:
        if isinstance(item, str):
            out.append(BestMetricTracker(item))
        elif isinstance(item, dict):
            name = item.get("name")
            if name:
                out.append(BestMetricTracker(name, item.get("higher_is_better")))
        else:  # OmegaConf node
            name = getattr(item, "name", None) or (item.get("name") if hasattr(item, "get") else None)
            if name:
                hib = getattr(item, "higher_is_better", None)
                if hib is None and hasattr(item, "get"):
                    hib = item.get("higher_is_better")
                out.append(BestMetricTracker(name, hib))
    return out


def seed_tracker_from_disk(tracker: BestMetricTracker, out_dir: str) -> bool:
    """Re-seed a tracker from a PERSISTED best checkpoint (resume-safe).

    On a resume the run dir is reused in place, so the historical best lives at
    ``<out_dir>/best_<metric>.ckpt/best.json`` (falling back to the un-suffixed
    ``best_<metric>/best.json``). Without this, a fresh tracker starts with
    ``best_value=None`` and the FIRST post-resume validation always counts as an
    "improvement" — wiping the true cross-resume best checkpoint (and its
    ``best.json`` pointer) with whatever the new weights happen to score.

    Returns ``True`` when the tracker was seeded. Skips (returns ``False``)
    silently when no pointer exists and with a warning when the pointer is
    unusable (unreadable / no numeric value / metric-name or direction
    mismatch) — in those cases the tracker must NOT trust the old baseline.
    """
    base = os.path.join(out_dir, f"best_{tracker.safe_name}")
    pointer = None
    for cand in (base + ".ckpt", base):
        p = os.path.join(cand, "best.json")
        if os.path.isfile(p):
            pointer = p
            break
    if pointer is None:
        return False
    try:
        with open(pointer, encoding="utf-8") as f:
            info = json.load(f)
    except Exception as e:
        logger.warning("best-ckpt resume: unparsable %s (%s); tracker starts fresh", pointer, e)
        return False
    val = _as_float(info.get("value"))
    if val is None:
        logger.warning("best-ckpt resume: %s has no numeric 'value'; tracker starts fresh", pointer)
        return False
    metric = info.get("metric")
    if metric is not None and str(metric) != tracker.name:
        logger.warning(
            "best-ckpt resume: %s is for metric %r (tracker tracks %r); tracker starts fresh",
            pointer, metric, tracker.name,
        )
        return False
    hib = info.get("higher_is_better")
    if hib is not None and bool(hib) != tracker.higher_is_better:
        logger.warning(
            "best-ckpt resume: %s direction (higher_is_better=%s) mismatches the config (%s); "
            "tracker starts fresh", pointer, hib, tracker.higher_is_better,
        )
        return False
    try:
        step = int(info.get("global_step"))
    except (TypeError, ValueError):
        step = -1
    tracker.best_value = val
    tracker.best_step = step
    return True


def snapshot_checkpoint(step_dir: str, best_dir: str, info: Dict[str, Any], background: bool = True) -> None:
    """Copy ``step_dir`` -> ``best_dir`` atomically; write ``best.json`` first.

    The JSON pointer (step + metric + value + source) is written synchronously so
    the best step is always recorded even if the (potentially large) copy is
    skipped/fails; the copy itself runs in a daemon thread by default so it never
    stalls the training loop.
    """
    os.makedirs(best_dir, exist_ok=True)
    try:
        with open(os.path.join(best_dir, "best.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    except Exception as e:  # pragma: no cover
        logger.warning("best-ckpt: failed to write best.json in %s: %s", best_dir, e)

    def _copy():
        try:
            if not os.path.isdir(step_dir):
                logger.warning("best-ckpt: source %s missing, only pointer kept", step_dir)
                return
            tmp = best_dir.rstrip("/") + ".tmp"
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(step_dir, tmp, dirs_exist_ok=True)
            # Preserve the pointer json inside the freshly copied tree too.
            shutil.copyfile(os.path.join(best_dir, "best.json"), os.path.join(tmp, "best.json"))
            final = best_dir.rstrip("/") + ".ckpt"
            if os.path.isdir(final):
                shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
            logger.info("best-ckpt: snapshotted %s -> %s", step_dir, final)
        except Exception as e:  # pragma: no cover - filesystem dependent
            logger.warning("best-ckpt: snapshot copy failed (%s); pointer kept", e)

    if background:
        threading.Thread(target=_copy, name=f"best-ckpt-{os.path.basename(best_dir)}", daemon=True).start()
    else:
        _copy()


# ---------------------------------------------------------------------------
# RayPPOTrainer subclass (imported lazily so the maths above stays light)
# ---------------------------------------------------------------------------

# Map: verl reward-extra var_name -> Transfusion-style val/ key for SwanLab.
# verl logs validation metrics as val-aux/<data_source>/<var_name>/mean@N
# (or val-core/.../reward/mean@N). The SFT Transfusion convention is val/PDMS,
# val/NC, val/DAC, etc. — this map bridges the two so SwanLab shows the same
# keys the user expects from supervised training.
_VAL_METRIC_MAP = {
    "pdms_reward": "val/pdms_reward",
    "pdms_PDMS": "val/PDMS",
    "pdms_NC": "val/NC",
    "pdms_DAC": "val/DAC",
    "pdms_DDC": "val/DDC",
    "pdms_EP": "val/EP",
    "pdms_TTC": "val/TTC",
    "pdms_C": "val/C",
    "epdms": "val/EPDMS",
    "epdms_reward": "val/epdms_reward",
    "epdms_NC": "val/EPDMS_NC",
    "epdms_DAC": "val/EPDMS_DAC",
    "epdms_DDC": "val/EPDMS_DDC",
    "epdms_TLC": "val/EPDMS_TLC",
    "epdms_EP": "val/EPDMS_EP",
    "epdms_TTC": "val/EPDMS_TTC",
    "epdms_LK": "val/EPDMS_LK",
    **{
        "epdms_HC": "val/EPDMS_HC",
        "epdms_EC": "val/EPDMS_EC",
        "soft_l2_reward": "val/soft_l2_reward",
        "planning_reward": "val/planning_reward",
        "prediction_reward": "val/prediction_reward",
        "pred_detection": "val/pred_detection",
        "pred_state_acc": "val/pred_state_acc",
        "pred_format": "val/pred_format",
        "cot_reward": "val/cot_reward",
        "cot_iou": "val/cot_iou",
        "cot_pred_acc": "val/cot_pred_acc",
        "ADE": "val/ADE",
        "FDE": "val/FDE",
        "L2": "val/L2",
        "l2_reward": "val/l2_reward",
        "reward": "val/reward",
        "score": "val/score",
    },
    **{
        "cot_len": "val/cot_len",
        "prompt_len": "val/prompt_len",
    },
}


def _add_transfusion_val_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PDMS/EPDMS sub-metrics + trajectory metrics from verl's val-aux/
    val-core format and add val/ prefixed keys, matching the Transfusion SFT
    convention (val/PDMS, val/NC, val/DAC, val/DDC, val/EP, val/TTC, val/C, the
    official val/EPDMS family when computed, val/L2, val/ADE, val/FDE). Only
    the "mean" aggregation is mapped (not std/best/worst).
    """
    for k, v in list(metrics.items()):
        parts = k.split("/")
        if len(parts) < 4 or not parts[0].startswith("val-"):
            continue
        if "mean" not in parts[-1]:
            continue
        var_name = parts[-2]
        if var_name in _VAL_METRIC_MAP:
            val_key = _VAL_METRIC_MAP[var_name]
            if val_key not in metrics:
                try:
                    metrics[val_key] = float(v)
                except (TypeError, ValueError):
                    pass
    return metrics


def build_trainer_class():
    """Return the ``TransfusionRayPPOTrainer`` subclass (imports verl on demand)."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    from recipe.transfusion.sync_driver import SyncRolloutTrainerMixin

    class TransfusionRayPPOTrainer(SyncRolloutTrainerMixin, RayPPOTrainer):
        """``RayPPOTrainer`` that runs the synchronous Transfusion rollout (via
        :class:`SyncRolloutTrainerMixin`) and keeps the best checkpoint(s) by metric."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tcfg = self.config.trainer
            spec = tcfg.get("best_metrics", None) if hasattr(tcfg, "get") else None
            self._best_trackers = parse_best_metrics(spec)
            self._best_background = bool(tcfg.get("best_ckpt_background_copy", True)) if hasattr(tcfg, "get") else True
            if self._best_trackers:
                logger.info(
                    "best-ckpt enabled for: %s",
                    ", ".join(f"{t.name}({'↑' if t.higher_is_better else '↓'})" for t in self._best_trackers),
                )
                # Resume-safe: restore each tracker's historical best from the
                # run dir's persisted best_<metric>.ckpt/best.json. On an
                # in-place resume default_local_dir already points at the
                # EXISTING run dir, so a fresh tracker would otherwise treat the
                # first post-resume validation as "the best ever" and overwrite
                # the true best checkpoint from the previous segment.
                out_dir = str(tcfg.get("default_local_dir", ""))
                for t in self._best_trackers:
                    if seed_tracker_from_disk(t, out_dir):
                        logger.info(
                            "best-ckpt resume: %s seeded from disk (best=%.6f @ step %d)",
                            t.name, t.best_value, t.best_step,
                        )
            # Ray injects runtime_env.env_vars into the WORKERS only — the driver
            # (where the train-split sample logger and rollout_data dump gates
            # read them) would otherwise never see the sample-logging knobs.
            # setdefault: a shell export still wins.
            try:
                ev = (
                    self.config.get("ray_kwargs", {}).get("ray_init", {}).get("runtime_env", {}).get("env_vars", {})
                    if hasattr(self.config, "get") else {}
                ) or {}
                for k in ("TRANSFUSION_TEST_FREQ", "TRANSFUSION_LOG_FIRST_STEPS", "TRANSFUSION_LOG_NUM_GROUPS"):
                    v = ev.get(k)
                    if v is not None:
                        os.environ.setdefault(k, str(v))
            except Exception:  # pragma: no cover - logging knobs must not break training
                pass

        def _validate(self, merged: bool = False):  # noqa: D401 - fast SFT path override
            if not merged and self._use_sft_style_validation():
                metrics = self._validate_sft_style()
            else:
                metrics = super()._validate(merged)
            # Extract PDMS sub-metrics + trajectory metrics from verl's
            # val-aux/val-core format and add Transfusion-style val/ keys
            # (val/PDMS, val/NC, val/DAC, val/DDC, val/EP, val/TTC, val/C,
            # val/L2, val/ADE, val/FDE, ...) so SwanLab shows the same keys
            # the user expects from supervised training (AUGUstus_SFT/train_qwen3p5.py
            # logs val/{k} for k in [PDMS, NC, DAC, DDC, EP, TTC, C, ADE, FDE, L2]).
            try:
                metrics = _add_transfusion_val_metrics(metrics)
            except Exception as e:  # pragma: no cover
                logger.warning("val metric extraction failed: %s", e)
            # Flush the accumulated val_samples window + val_reward/* means EXACTLY
            # once here, now that the FULL validation pass (every data.val_batch_size
            # chunk) has completed -- recipe.transfusion.sync_driver._attach_rm_scores
            # deliberately does NOT flush per-chunk (that used to repeatedly
            # overwrite the same step's val_samples/* files + val_reward/* scalars
            # with just the LAST chunk's data once val_batch_size started chunking
            # the split instead of sending it as one giant batch, 2026-07-07).
            # The SFT-style path logs its leading samples on rank 0 directly; the
            # driver-side flush remains for the legacy DataProto path.
            try:
                from recipe.transfusion.sample_logging import flush_metric_means

                flush_metric_means("val")
            except Exception as e:  # pragma: no cover - logging must never break training
                logger.warning("val sample-logger flush failed: %s", e)
            # [val_samples -> SwanLab] The val_samples artefacts were written by
            # the rank-0 RAY WORKER (validation.py), which has no active SwanLab
            # run — they never reached SwanLab. The driver owns the SwanLab run
            # and reads the same shared output dir, so mirror them here.
            #
            # Mirror ALL validation-step dirs that have artefacts on disk (not
            # just the current ``self.global_steps``): this way every validation
            # pass is surfaced in SwanLab under its real step, and each step's
            # samples are tagged with their step (see
            # TransfusionSampleLogger.mirror_step_dir_to_swanlab). That fixes the
            # symptom where only ONE (unidentifiable) validation step ever shows
            # up (docs/VAL_LOGGING.md).
            try:
                from recipe.transfusion.sample_logging import get_global_sample_logger

                get_global_sample_logger().mirror_all_val_steps_to_swanlab()
            except Exception as e:  # pragma: no cover - never break training
                logger.warning("val_samples SwanLab mirror failed: %s", e)
            try:
                self._maybe_save_best(metrics)
            except Exception as e:  # pragma: no cover - never break training
                logger.warning("best-ckpt update failed: %s", e)
            return metrics

        def _use_sft_style_validation(self) -> bool:
            """Whether to use worker-local validation instead of central DataProto/Ray."""
            try:
                tsub = self.config.data.get("transfusion", {}) or {}
                return bool(tsub.get("use_sft_style_validation", False))
            except Exception:
                return False

        def _validate_sft_style(self) -> Dict[str, float]:
            """Validate on the ray workers' local shards, mirroring SFT evaluate()."""
            from omegaconf import OmegaConf

            from recipe.transfusion.validation import aggregate_shard_results

            cfg = OmegaConf.to_container(self.config, resolve=True)
            data = cfg.get("data", {}) or {}
            tsub = data.get("transfusion", {}) or {}
            rollout = (cfg.get("actor_rollout_ref", {}) or {}).get("rollout", {}) or {}
            reward_fn = ((cfg.get("reward", {}) or {}).get("custom_reward_function", {}) or {})
            trainer_cfg = cfg.get("trainer", {}) or {}

            val_files = data.get("val_files")
            if isinstance(val_files, str):
                val_files = [val_files]
            train_files = data.get("train_files")
            if isinstance(train_files, str):
                train_files = [train_files]

            n_gpus = int(trainer_cfg.get("nnodes", 1)) * int(trainer_cfg.get("n_gpus_per_node", 8))
            val_global_bs = int(data.get("val_batch_size", 0) or 0)
            batch_size = int(tsub.get("sft_val_batch_size_per_gpu",
                                      val_global_bs // n_gpus if val_global_bs and n_gpus else 32))
            num_workers = int(tsub.get("sft_val_num_workers_per_gpu", 8))
            pdms_workers = int(tsub.get("sft_val_pdms_workers_per_gpu", 8))
            data_source = str(tsub.get("data_source", "navsim_transfusion"))

            # Validation CoT sampling: rollout.rl_val_cot_* override the
            # TRAINING-rollout sampling knobs (rollout.temperature/top_p/top_k);
            # only the resolved values enter the payload so the worker-side
            # fallback ($TRANSFUSION_CONFIG's cot.cot_*) never silently kicks in.
            def _val_cot(rl_kwargs, val_key, train_key, default):
                v = rl_kwargs.get(val_key)
                if v is None:
                    v = rl_kwargs.get(train_key)
                return default if v is None else v

            cot_temperature = float(_val_cot(rollout, "rl_val_cot_temperature", "temperature", 1.0))
            cot_top_p = float(_val_cot(rollout, "rl_val_cot_top_p", "top_p", 0.95))
            cot_top_k = int(float(_val_cot(rollout, "rl_val_cot_top_k", "top_k", 100)))

            # Validation CoT generation cap: ``rl_val_cot_max_len`` (when set) else
            # the RL response budget ``rollout.response_length`` (default 256). The
            # model generates ``max_cot_tokens - 1`` CONTENT tokens so the forced
            # EOT column fits inside the window, mirroring the training rollout
            # (rollout.py ``_cot_max_tokens = response_length - 1``). Capping this
            # is what keeps the validation CoT from exploding to the model's native
            # ``max_cot_tokens`` (512) and padding the whole window with junk.
            rl_val_cot_max_len = rollout.get("rl_val_cot_max_len")
            rl_resp = rollout.get("response_length")
            val_cot_budget = int(data.get("max_response_length", 256) or 256)
            if rl_resp is not None:
                val_cot_budget = int(rl_resp)
            if rl_val_cot_max_len is not None:
                val_cot_budget = int(rl_val_cot_max_len)
            val_cot_max_tokens = max(1, val_cot_budget - 1)

            payload = {
                "train_files": train_files,
                "val_files": val_files,
                "metric_cache_train": data.get("metric_cache_train"),
                "metric_cache_test": data.get("metric_cache_test"),
                "metric_cache_train_v2": data.get("metric_cache_train_v2"),
                "metric_cache_test_v2": data.get("metric_cache_test_v2"),
                "seed": int(data.get("seed", 42) or 42),
                "max_samples": int(data.get("val_max_samples", -1)),
                "batch_size_per_gpu": max(1, batch_size),
                "num_workers_per_gpu": max(0, num_workers),
                "pdms_workers_per_gpu": max(1, pdms_workers),
                "global_step": int(self.global_steps),
                "log_num_samples": int(trainer_cfg.get("log_val_generations", 10) or 0),
                "num_steps": rollout.get("rl_val_num_steps", rollout.get("rl_num_steps")),
                # nocot (rollout.train_only_action): do NOT generate CoT in
                # validation either — the autoregressive CoT pass over the full
                # val split is pure wasted time. The reward skips the whole
                # prediction block on empty CoT (transfusion_reward), so val
                # metrics stay well-defined (planning/PDMS only).
                "generate_cot": not bool(rollout.get("train_only_action", False)),
                "cot_max_tokens": val_cot_max_tokens,
                "cot_temperature": cot_temperature,
                "cot_top_p": cot_top_p,
                "cot_top_k": cot_top_k,
                "reward_kwargs": reward_fn.get("reward_kwargs", {}) or {},
                "transfusion": tsub,
            }

            t0 = time.time()
            results = self.actor_rollout_wg.validate_transfusion_sft(payload)
            elapsed = time.time() - t0
            metrics = aggregate_shard_results(results, data_source=data_source)
            metrics.setdefault(f"val-core/{data_source}/score/mean", 0.0)
            metrics[f"val/testing_time"] = elapsed
            _aux = f"val-aux/{data_source}"
            _pdms_scored = int(metrics.get(f"{_aux}/pdms_scored/num", 0))
            _pdms_cand = int(metrics.get(f"{_aux}/pdms_candidate/num", 0))
            _pdms_inv = int(metrics.get(f"{_aux}/pdms_invalid/num", 0))
            logger.warning(
                "[time] fast_sft_validate: %.1fs (%d samples, %d ranks, B/gpu=%d, workers/gpu=%d, "
                "cot_sampler: temperature=%.3g top_p=%.3g top_k=%d num_steps=%s)",
                elapsed,
                int(metrics.get("val/num_samples", 0)),
                len(results),
                payload["batch_size_per_gpu"],
                payload["num_workers_per_gpu"],
                cot_temperature,
                cot_top_p,
                cot_top_k,
                str(payload["num_steps"]),
            )
            logger.warning(
                "[pdms-counts] validation: eligible=%d  valid/scored=%d  invalid=%d   "
                "(pdms_reward mean denominator = valid/scored)",
                _pdms_cand, _pdms_scored, _pdms_inv,
            )
            return metrics

        def _maybe_save_best(self, metrics: Dict[str, Any]):
            if not self._best_trackers or not metrics:
                return
            if self.global_steps == 0:
                # step 0 = val_before_train on the freshly-loaded baseline, not a
                # trained step -- skip best-ckpt tracking/saving entirely so it
                # never force-creates a global_step_0 checkpoint (no training has
                # happened yet; nothing worth keeping).
                return
            improved = [t for t in self._best_trackers if t.update(metrics, self.global_steps)]
            if not improved:
                return
            out_dir = self.config.trainer.default_local_dir
            step_dir = os.path.join(out_dir, f"global_step_{self.global_steps}")
            if not os.path.isdir(step_dir):
                # Validation can run before this step's periodic save; create it.
                self._save_checkpoint()
            for t in improved:
                best_dir = os.path.join(out_dir, f"best_{t.safe_name}")
                snapshot_checkpoint(
                    step_dir, best_dir,
                    info={
                        "metric": t.name,
                        "higher_is_better": t.higher_is_better,
                        "value": t.best_value,
                        "global_step": t.best_step,
                        "source": step_dir,
                    },
                    background=self._best_background,
                )

    return TransfusionRayPPOTrainer


__all__ = [
    "BestMetricTracker",
    "find_metric_value",
    "guess_higher_is_better",
    "parse_best_metrics",
    "seed_tracker_from_disk",
    "snapshot_checkpoint",
    "build_trainer_class",
    "_add_transfusion_val_metrics",
]
