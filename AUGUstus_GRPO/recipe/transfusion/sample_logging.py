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

"""Transfusion-style per-sample / per-eval logging for the verl RL recipe.

Mirrors ``AUGUstus_SFT/model/logger.py`` (``log_data_samples`` /
``log_eval_samples`` / ``plot_trajectory_comparison``) so the RL runs surface the
same rich per-sample information as supervised training:

* training samples — prompt text, media-shape info, GT trajectory (values + plot),
  CoT labels;
* eval samples — prompt text, GT-vs-pred trajectory (values + plot), per-step L2,
  predicted CoT, predicted/GT bbox visualisation, and per-sample metrics
  (L2 / PDMS / CoT IoU / future-position accuracy).

Robustness first: media artefacts (txt/png) are **always mirrored to the run's
output dir** (``<default_local_dir>/samples/step_XXXX/...``) so they survive
regardless of which process produces them, and are *additionally* pushed to
SwanLab on a best-effort basis (a no-op if SwanLab is not initialised in the
current process).  Scalar series (``train_reward/*`` means etc.) are NOT written
per step — they append ONE JSON row per flush to a single
``<default_local_dir>/samples/scalars.jsonl``, mirroring how SwanLab accumulates
them (a scalar joins a series; no new blob per step).  Net effect: a
``step_XXXX/`` dir only exists for steps that actually carry uploaded media
(``val_samples`` / thinned ``train_samples``) — the local mirror now keeps
exactly the set of artefacts that get uploaded.  matplotlib / SwanLab / PIL
imports are all guarded so logging can never crash training.

The numeric helpers (:func:`per_step_l2`, :func:`plot_trajectory_comparison`,
:func:`format_sample_values`, :func:`plot_group_trajectories`,
:func:`should_log_train_samples`) are framework-light and unit-tested.

Volume control (2026-07-30): the per-step TRAIN-sample artefacts
(``train_samples/*`` + ``rollout_data/{step}.jsonl``) follow
:func:`should_log_train_samples` — by default every step (old behaviour); with
``TRANSFUSION_LOG_FIRST_STEPS=N`` only the first N steps plus the step right
after each validation pass. The ``train_reward/*`` scalar means are NOT
affected (flushed every step — to ``samples/scalars.jsonl``, see above; since
2026-07-31 no longer one ``step_N/metrics.json`` per step). GRPO groups
(``rollout.n`` siblings per prompt) can additionally be rendered as overlay
plots (``TRANSFUSION_LOG_NUM_GROUPS``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framework-light numeric / formatting helpers (unit-tested)
# ---------------------------------------------------------------------------
def per_step_l2(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-timestep xy L2 distance ``[T]`` (matches Transfusion's eval logging)."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    d = min(pred.shape[-1], gt.shape[-1], 2)
    return np.sqrt(((pred[:, :d] - gt[:, :d]) ** 2).sum(axis=-1))


def format_sample_values(pred: Optional[np.ndarray], gt: np.ndarray) -> str:
    """Human-readable GT(/Pred) per-step table, mirroring Transfusion's text log."""
    gt = np.asarray(gt, dtype=np.float32)
    lines: List[str] = []
    if pred is not None:
        pred = np.asarray(pred, dtype=np.float32)
        l2 = per_step_l2(pred, gt)
        lines.append(f"L2 (mean over {gt.shape[0]} steps): {float(l2.mean()):.4f}")
        lines.append("per-step L2: " + ", ".join(f"{v:.4f}" for v in l2.tolist()))
        lines.append("GT vs Pred:")
        for j in range(gt.shape[0]):
            gt_str = ", ".join(f"{gt[j, k]:.4f}" for k in range(gt.shape[-1]))
            pr_str = ", ".join(f"{pred[j, k]:.4f}" for k in range(pred.shape[-1]))
            lines.append(f"  step {j}: GT=({gt_str})  Pred=({pr_str})")
    else:
        lines.append("GT:")
        for j in range(gt.shape[0]):
            gt_str = ", ".join(f"{gt[j, k]:.4f}" for k in range(gt.shape[-1]))
            lines.append(f"  step {j}: ({gt_str})")
    return "\n".join(lines)


def plot_trajectory_comparison(pred: Optional[np.ndarray], gt: np.ndarray, title: str = "Trajectory"):
    """Predicted-vs-GT trajectory figure (faithful to Transfusion). Returns a
    matplotlib ``Figure`` or ``None`` if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib always present in this env
        return None

    gt = np.asarray(gt, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 6))
    dd = min(gt.shape[-1], 2)
    ax.plot(gt[:, 0], gt[:, 1] if dd > 1 else range(len(gt)), "r--o", ms=4, lw=1.5, label="GT")
    if pred is not None:
        pred = np.asarray(pred, dtype=np.float32)
        dp = min(pred.shape[-1], 2)
        ax.plot(pred[:, 0], pred[:, 1] if dp > 1 else range(len(pred)), "b-o", ms=4, lw=1.5, label="Pred")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_group_trajectories(
    preds: Sequence[np.ndarray],
    gt: np.ndarray,
    scores: Optional[Sequence[float]] = None,
    title: str = "Group trajectories",
):
    """Overlay of ONE GRPO group's sampled trajectories vs the shared GT.

    Training rollouts sample ``rollout.n`` trajectories per prompt (one GRPO
    group); the per-sample plots only ever show single trajectories, so the
    in-group spread is invisible. ``preds`` is that group's list of ``[T, D]``
    trajectories (tab10 colours, legend carries each trajectory's reward);
    ``gt`` is the shared ground truth (red dashed). Returns a matplotlib
    ``Figure`` or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib always present in this env
        return None

    gt = np.asarray(gt, dtype=np.float32)
    preds = [np.asarray(p, dtype=np.float32) for p in (preds or [])]
    fig, ax = plt.subplots(figsize=(6, 6))
    dd = min(gt.shape[-1], 2)
    ax.plot(gt[:, 0], gt[:, 1] if dd > 1 else range(len(gt)), "r--o", ms=4, lw=1.5, label="GT")
    cmap = plt.get_cmap("tab10")
    for j, p in enumerate(preds):
        dp = min(p.shape[-1], 2)
        label = f"traj {j}"
        if scores is not None and j < len(scores):
            label += f" (r={float(scores[j]):.3f})"
        ax.plot(
            p[:, 0], p[:, 1] if dp > 1 else range(len(p)), "-o", ms=3, lw=1.0,
            color=cmap(j % 10), label=label,
        )
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Logger: local-dir mirror (always) + best-effort SwanLab
# ---------------------------------------------------------------------------
class TransfusionSampleLogger:
    """Writes faithful per-sample / per-eval artefacts to disk + best-effort SwanLab.

    Args:
        output_dir: run output dir (``trainer.default_local_dir``). Artefacts go
            under ``<output_dir>/samples/step_XXXXXXXX/``.
        num_samples: how many leading samples to log per call (default 10).
        enable_swanlab: also push to SwanLab when a run is active (best-effort).
    """

    def __init__(self, output_dir: Optional[str], num_samples: int = 10, enable_swanlab: bool = True):
        self.output_dir = output_dir
        self.num_samples = int(num_samples)
        self.enable_swanlab = bool(enable_swanlab)
        self.samples_dir = Path(output_dir) / "samples" if output_dir else None
        if self.samples_dir is not None:
            try:
                self.samples_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:  # pragma: no cover - filesystem dependent
                logger.warning("Sample logger could not create %s: %s", self.samples_dir, e)
                self.samples_dir = None

    # -- low-level sinks ---------------------------------------------------
    def _swanlab(self):
        if not self.enable_swanlab:
            return None
        try:
            import swanlab  # noqa: F401

            # Only log if a run is active (verl already called swanlab.init()).
            if getattr(swanlab, "get_run", None) is not None and swanlab.get_run() is None:
                return None
            return swanlab
        except Exception:
            return None

    def _step_dir(self, step: int) -> Optional[Path]:
        if self.samples_dir is None:
            return None
        d = self.samples_dir / f"step_{int(step):08d}"
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _safe(key: str) -> str:
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in key)

    def _artifact_path(self, step_dir: Path, key: str, suffix: str) -> Path:
        """Resolve ``key`` (e.g. ``"val_samples/sample_0_text"``) to a real path
        under ``step_dir`` -- the part before the first ``/`` becomes an actual
        subdirectory (``val_samples/`` / ``train_samples/``, created as needed),
        matching the SwanLab grouping (see log_text/log_figure) instead of the
        old flattened ``val_samples_sample_0_text.txt`` naming.
        """
        group, _, rest = key.partition("/")
        if rest:
            sub = step_dir / self._safe(group)
            sub.mkdir(parents=True, exist_ok=True)
            return sub / (self._safe(rest) + suffix)
        return step_dir / (self._safe(key) + suffix)

    def log_scalars(self, data: Dict[str, float], step: int):
        sl = self._swanlab()
        if sl is not None:
            try:
                sl.log({k: float(v) for k, v in data.items()}, step=int(step))
            except Exception:
                pass
        # Local mirror: append ONE JSON row to the single scalars.jsonl instead
        # of creating samples/step_XXXXXXXX/metrics.json for every step (that
        # used to leave a step dir behind for EVERY training step even when the
        # step carries no uploaded artefacts). Appends are process-safe enough
        # here: the only concurrent writers are driver flushes and the rank-0
        # val worker, and lines are short; worst case is an interleaved line.
        if self.samples_dir is not None:
            try:
                import json

                rec = {"step": int(step), **{k: float(v) for k, v in data.items()}}
                with open(self.samples_dir / "scalars.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def log_text(self, key: str, text: str, step: int):
        d = self._step_dir(step)
        if d is not None:
            try:
                out_path = self._artifact_path(d, key, ".txt")
                out_path.write_text(text, encoding="utf-8")
            except Exception:
                pass
        sl = self._swanlab()
        if sl is not None:
            try:
                # NOTE: ``key`` already carries its own top-level namespace
                # ("val_samples/..." or "train_samples/...", set by the
                # caller's ``prefix``) -- do NOT prepend another "samples/"
                # here, that used to nest everything under one shared
                # "samples" group in SwanLab ("samples/val_samples/..."),
                # merging train+val together instead of keeping val_samples
                # as its own top-level group (2026-07-07 fix).
                sl.log({key: sl.Text(text)}, step=int(step))
            except Exception:
                pass

    def log_figure(self, key: str, fig, step: int, caption: str = ""):
        if fig is None:
            return
        d = self._step_dir(step)
        saved_path = None
        if d is not None:
            try:
                saved_path = self._artifact_path(d, key, ".png")
                fig.savefig(saved_path, dpi=100, bbox_inches="tight")
            except Exception:
                saved_path = None
        sl = self._swanlab()
        if sl is not None:
            try:
                # See log_text() above -- ``key`` is already fully namespaced.
                sl.log({key: sl.Image(fig, caption=caption)}, step=int(step))
            except Exception:
                pass
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass

    # -- driver-side mirror of worker-written artefacts -----------------------
    def mirror_step_dir_to_swanlab(self, step: int, prefix: str = "val_samples") -> int:
        """Re-publish one step's on-disk artefacts to SwanLab from THIS process.

        The SFT-style validation renders ``<output_dir>/samples/step_XXXXXXXX/
        val_samples/*.txt|png`` on the rank-0 RAY WORKER, which has no active
        SwanLab run — the live ``sl.log`` inside ``log_text`` / ``log_figure``
        is a no-op there, so those artefacts never reached SwanLab. The output
        dir sits on the shared filesystem, so the DRIVER (which owns the
        SwanLab run) re-reads the files and logs them here with the SAME keys
        (``val_samples/<stem>``) the live path would have used. No-op when
        SwanLab is inactive or the directory is missing; returns the number of
        mirrored artefacts.

        The artefacts are uploaded with the SwanLab ``step`` set to the step
        dir's REAL step number, using the SAME keys for every step — so all the
        validation passes form ONE series per sample key (``val_samples/
        sample_0_traj`` ...) and SwanLab renders a *step slider* you can drag to
        view the result of each distinct validation step, exactly like the
        training samples (previously only the current ``global_steps`` was
        mirrored, so intermediate validation steps never showed up). Each
        artefact's text/plot already carries its ``step N`` tag (added at log
        time by ``log_eval_sample``).
        """
        sl = self._swanlab()
        if sl is None or self.samples_dir is None:
            return 0
        d = self.samples_dir / f"step_{int(step):08d}" / self._safe(prefix)
        if not d.is_dir():
            return 0
        payload: Dict[str, Any] = {}
        try:
            for p in sorted(d.iterdir()):
                key = f"{prefix}/{p.stem}"
                if p.suffix == ".txt":
                    payload[key] = sl.Text(p.read_text(encoding="utf-8", errors="replace"))
                elif p.suffix == ".png":
                    from PIL import Image

                    payload[key] = sl.Image(Image.open(p))
        except Exception as e:  # pragma: no cover - never break training
            logger.warning("[%s] mirror to SwanLab failed (step=%d): %s", prefix, int(step), e)
            return 0
        if not payload:
            return 0
        try:
            sl.log(payload, step=int(step))
            logger.warning(
                "[%s] mirrored %d artifacts (step=%d) to SwanLab", prefix, len(payload), int(step)
            )
        except Exception as e:  # pragma: no cover - never break training
            logger.warning("[%s] mirror to SwanLab failed (step=%d): %s", prefix, int(step), e)
            return 0
        return len(payload)

    def mirror_all_val_steps_to_swanlab(self) -> int:
        """Re-publish EVERY validation step's on-disk ``val_samples`` artefacts.

        Mirrors each distinct ``<output_dir>/samples/step_XXXXXXXX/val_samples``
        directory under ITS own step, so no validation pass that ever wrote
        artefacts to disk is silently dropped (previously only the CURRENT
        ``self.global_steps`` was mirrored, so if an intermediate validation's
        driver-side mirror was missed — e.g. a worker/driver hiccup or a resumed
        run — that validation's samples never surfaced in SwanLab and the user
        only ever saw one, unidentified step). Each step's artefacts are re-tagged
        with their real step via :meth:`mirror_step_dir_to_swanlab`. Returns the
        total number of mirrored artefacts.
        """
        if self.samples_dir is None:
            return 0
        total = 0
        try:
            # sort so deterministic (lowest step first)
            for sub in sorted(self.samples_dir.glob(f"step_*")):
                if not sub.is_dir():
                    continue
                if not (sub / self._safe("val_samples")).is_dir():
                    continue
                try:
                    step = int(sub.name.replace("step_", ""))
                except ValueError:
                    continue
                total += self.mirror_step_dir_to_swanlab(step, prefix="val_samples")
            return total
        except Exception as e:  # pragma: no cover - never break training
            logger.warning("[val_samples] mirror-all failed: %s", e)
            return total

    # -- high-level: training samples (mirrors log_data_samples) ------------
    def log_train_sample(self, s: Dict[str, Any], index: int, step: int):
        if s.get("text"):
            self.log_text(f"train_data/sample_{index}_text", str(s["text"]), step)
        if s.get("media_info"):
            self.log_text(f"train_data/sample_{index}_media_info", str(s["media_info"]), step)
        gt = s.get("gt_traj")
        if gt is not None:
            gt = np.asarray(gt, dtype=np.float32)
            self.log_text(f"train_data/sample_{index}_gt_values", format_sample_values(None, gt), step)
            self.log_figure(
                f"train_data/sample_{index}_gt_traj",
                plot_trajectory_comparison(None, gt, title=f"GT Trajectory (sample {index})"),
                step,
            )
        if s.get("cot"):
            self.log_text(f"train_data/sample_{index}_cot", str(s["cot"]), step)

    def log_train_samples(self, samples: Sequence[Dict[str, Any]], step: int):
        for i in range(min(self.num_samples, len(samples))):
            self.log_train_sample(samples[i], i, step)

    # -- high-level: eval samples (mirrors log_eval_samples) ---------------
    def log_eval_sample(self, s: Dict[str, Any], index: int, step: int, prefix: str = "val_samples"):
        """``s``: ``{text?, gt_traj, pred_traj?, cot?, gt_cot?, metrics?, raw_frames?}``.

        ``prefix`` picks the SwanLab/disk key namespace -- ``"val_samples"`` for an
        actual validation pass, ``"train_samples"`` for a training-rollout preview
        (same renderer either way; only the label differs, see docs/VAL_LOGGING.md).
        """
        gt = s.get("gt_traj")
        pred = s.get("pred_traj")
        # Tag every artefact with the step so the user can always identify WHICH
        # validation step a sample belongs to (docs/VAL_LOGGING.md).
        step_tag = f" (step {int(step)})" if step is not None else ""
        if s.get("text"):
            self.log_text(
                f"{prefix}/sample_{index}_text",
                f"===== step {int(step)} =====\n{str(s['text'])}" if step is not None else str(s["text"]),
                step,
            )
        if gt is not None:
            gt = np.asarray(gt, dtype=np.float32)
            pred_arr = np.asarray(pred, dtype=np.float32) if pred is not None else None
            self.log_text(
                f"{prefix}/sample_{index}_values",
                f"===== step {int(step)} =====\n{format_sample_values(pred_arr, gt)}" if step is not None else format_sample_values(pred_arr, gt),
                step,
            )
            self.log_figure(
                f"{prefix}/sample_{index}_traj",
                plot_trajectory_comparison(pred_arr, gt, title=f"{prefix} sample {index}{step_tag}"),
                step,
                caption=f"step {step} — sample {index}",
            )
            if pred_arr is not None:
                self.log_scalars(
                    {f"{prefix}/sample_{index}_L2": float(per_step_l2(pred_arr, gt).mean())}, step
                )
        self.log_text(
            f"{prefix}/sample_{index}_cot",
            f"===== step {int(step)} =====\n{str(s.get('cot') or '(no CoT generated)')}" if step is not None else str(s.get("cot") or "(no CoT generated)"),
            step,
        )
        # Always show the last video frame (independent of whether metrics/CoT bbox
        # parse) so you can see what the model is currently looking at, exactly like
        # supervised Transfusion's log_eval_samples.
        self._log_last_frame(index, s, step, prefix=prefix)
        metrics = s.get("metrics")
        if metrics:
            self.log_text(
                f"{prefix}/sample_{index}_metrics",
                f"===== step {int(step)} =====\n" + "\n".join(f"{k}: {float(v):.4f}" for k, v in metrics.items() if isinstance(v, (int, float))) if step is not None else "\n".join(f"{k}: {float(v):.4f}" for k, v in metrics.items() if isinstance(v, (int, float))),
                step,
            )

    def log_eval_samples(self, samples: Sequence[Dict[str, Any]], step: int, prefix: str = "val_samples"):
        for i in range(min(self.num_samples, len(samples))):
            self.log_eval_sample(samples[i], i, step, prefix=prefix)

    def _log_last_frame(self, i: int, s: Dict[str, Any], step: int, prefix: str = "val_samples"):
        """Show the LAST raw video frame, mirroring Transfusion's eval logging.

        Always logged (plain) when ``raw_frames`` is available, regardless of
        whether the generated/GT CoT contains a parseable bbox, so you can always
        see the current scene. When a bbox DOES parse, it is additionally drawn as
        a second, overlaid artefact (faithful to Transfusion's ``log_eval_samples``).
        """
        raw_frames = s.get("raw_frames")
        if not raw_frames:
            return
        last_frame = raw_frames[-1]

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(np.array(last_frame))
            ax.set_title(f"Last frame — sample {i}")
            ax.axis("off")
            fig.tight_layout()
            self.log_figure(
                f"{prefix}/sample_{i}_last_frame", fig, step,
                caption=f"sample {i} — last frame",
            )
        except Exception:
            pass

        try:
            from model.logger import draw_pred_bboxes_on_frame, parse_pred_bboxes
        except Exception:
            return
        try:
            pred_boxes = parse_pred_bboxes(s.get("cot") or "")
            gt_boxes = parse_pred_bboxes(s.get("gt_cot") or "")
            if not (pred_boxes or gt_boxes):
                return
            bbox_fig = draw_pred_bboxes_on_frame(
                last_frame, pred_boxes, gt_boxes,
                title=f"Pred(red)/GT(green) bbox — sample {i}",
            )
            self.log_figure(
                f"{prefix}/sample_{i}_pred_bbox", bbox_fig, step,
                caption=f"pred={len(pred_boxes)} gt={len(gt_boxes)} (last frame)",
            )
        except Exception:
            return


# ---------------------------------------------------------------------------
# Process-wide singleton + best-effort entry points (used by the reward path)
# ---------------------------------------------------------------------------
_GLOBAL_LOGGER: Optional[TransfusionSampleLogger] = None
# Per-split accumulation state, keyed by "train" / "val". Kept SEPARATE (unlike
# the old single shared counter) so a validation pass's samples/means never mix
# with training rollout samples -- see docs/VAL_LOGGING.md. Each split is
# flushed EXPLICITLY once per rollout batch (one training step, or one full
# validation pass over data.val_max_samples) by recipe.transfusion.sync_driver,
# rather than an arbitrary sample-count threshold.
_SPLIT_STATE: Dict[str, Dict[str, Any]] = {
    "train": {"step": 0, "window": 0, "sums": {}, "count": 0, "groups": {}},
    "val": {"step": 0, "window": 0, "sums": {}, "count": 0, "groups": {}},
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def should_log_train_samples(step: Optional[int]) -> bool:
    """Whether the per-sample TRAIN artefacts are written for training ``step``.

    Covers BOTH volumes: ``samples/step_N/train_samples/*`` (per-sample txt/png)
    and ``rollout_data/{step}.jsonl`` (the callers gate on this too). The
    ``train_reward/*`` scalar means are NOT affected — they flush every step.

    * ``TRANSFUSION_LOG_FIRST_STEPS`` unset (default): log EVERY step — the
      original behaviour, so plain verl usage is untouched.
    * ``TRANSFUSION_LOG_FIRST_STEPS=N``: log the first ``N`` steps plus the ONE
      step right after each validation pass, i.e. ``(step - 1) %
      TRANSFUSION_TEST_FREQ == 0`` (an approximation: verl's extra is_last_step
      validation is not modelled).
    """
    if step is None:
        return True
    raw = (os.environ.get("TRANSFUSION_LOG_FIRST_STEPS") or "").strip()
    if not raw:
        return True
    first_n = _env_int("TRANSFUSION_LOG_FIRST_STEPS", 0)
    if int(step) <= first_n:
        return True
    tf = _env_int("TRANSFUSION_TEST_FREQ", 0)
    return tf > 0 and (int(step) - 1) % tf == 0


def get_global_sample_logger() -> TransfusionSampleLogger:
    """Lazily build the process-wide sample logger from the environment.

    Output dir comes from ``TRANSFUSION_OUTPUT_DIR`` (the run scripts export it =
    ``trainer.default_local_dir``); ``TRANSFUSION_LOG_NUM_SAMPLES`` controls the
    leading-sample count (default 10).
    """
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        out = os.environ.get("TRANSFUSION_OUTPUT_DIR") or os.environ.get("TRANSFUSION_SAMPLE_LOG_DIR")
        _GLOBAL_LOGGER = TransfusionSampleLogger(out, num_samples=_env_int("TRANSFUSION_LOG_NUM_SAMPLES", 10))
    return _GLOBAL_LOGGER


def log_sample_best_effort(
    sample: Dict[str, Any], *, is_validate: bool = False, step: Optional[int] = None,
) -> None:
    """Log one reward-stage sample, faithful to Transfusion, never raising.

    ``is_validate`` / ``step`` should come straight from the rollout batch's own
    ``meta_info["validate"]`` / ``meta_info["global_steps"]`` (verl sets both;
    see ``recipe.transfusion.sync_driver._attach_rm_scores``), so training and
    validation samples land under DISTINCT keys (``train_samples/*`` vs.
    ``val_samples/*``) tagged with the ACTUAL step -- see docs/VAL_LOGGING.md.
    Accumulates this split's running metric sums/leading-sample window; call
    :func:`flush_metric_means` once the whole batch (train step or validation
    pass) has been scored to average + emit them (never on a sample-count
    heuristic, which used to conflate a validation pass with training steps).
    """
    try:
        split = "val" if is_validate else "train"
        lg = get_global_sample_logger()
        st = _SPLIT_STATE[split]
        if step is not None:
            new_step = int(step)
            if new_step != st["step"]:
                # A NEW step/pass begins -> reset the per-pass artefact window.
                # The val window used to only ever reset via
                # ``flush_metric_means("val")``, which runs in the DRIVER — the
                # rank-0 validation WORKER (where val_samples are actually
                # written) never got the reset, so after the first validation
                # filled the window (= ``TRANSFUSION_LOG_NUM_SAMPLES``), every
                # later validation silently logged zero samples, even though
                # ``trainer.log_val_generations`` was set. Keying the window to
                # the step makes EVERY validation log its leading samples
                # again (same for the group overlay plots, which share this
                # gating). The means/flushing semantics are untouched.
                st["window"] = 0
                st["groups"] = {}
            st["step"] = new_step

        metrics = sample.get("metrics") or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                st["sums"][k] = st["sums"].get(k, 0.0) + float(v)
        st["count"] += 1

        # Volume control: train-split per-sample artefacts follow
        # should_log_train_samples (first N steps + after-validation steps);
        # the means above and the VAL split are never thinned.
        log_this_step = split != "train" or should_log_train_samples(st["step"])
        if log_this_step and st["window"] < lg.num_samples:
            lg.log_eval_sample(sample, st["window"], st["step"], prefix=f"{split}_samples")
            st["window"] += 1
        # GRPO in-group overlay plots: buffer (pred/gt/score/metrics) per
        # group_key; flush_metric_means renders them once the whole step is
        # scored. Bounded to the first `num_groups` DISTINCT groups. Buffered
        # for BOTH splits so the per-rollout trajectory spread is also visible
        # on the validation samples (val_samples/group_*_trajs), not only on
        # training steps.
        num_groups = _env_int("TRANSFUSION_LOG_NUM_GROUPS", 0)
        if log_this_step and num_groups > 0:
            gkey = sample.get("group_key")
            if gkey is not None:
                groups = st["groups"]
                g = None
                if gkey in groups or len(groups) < num_groups:
                    g = groups.setdefault(gkey, {"preds": [], "scores": [], "metrics": [], "gt": None})
                    if g["gt"] is None:
                        g["gt"] = sample.get("gt_traj")
                if g is not None:
                    g["preds"].append(sample.get("pred_traj"))
                    score = (sample.get("metrics") or {}).get("score")
                    g["scores"].append(float(score) if isinstance(score, (int, float)) else 0.0)
                    g["metrics"].append(sample.get("metrics") or {})
    except Exception:  # pragma: no cover - logging must never break training
        pass


def flush_metric_means(split: Optional[str] = None) -> None:
    """Average + flush ONE split's (or both, if ``split`` is ``None``)
    accumulated per-component reward means, then reset that split's window.

    Call this ONCE at the end of scoring a whole rollout batch (one training
    step, or one full validation pass) -- see
    ``recipe.transfusion.sync_driver._attach_rm_scores`` -- so "the" number is a
    true average over everything just scored, not a running window that
    happens to close after an arbitrary sample count.
    """
    try:
        lg = get_global_sample_logger()
        for s in ([split] if split else ["train", "val"]):
            st = _SPLIT_STATE[s]
            if st["count"] > 0:
                prefix = "val_reward" if s == "val" else "train_reward"
                means = {f"{prefix}/{k}": v / st["count"] for k, v in st["sums"].items()}
                lg.log_scalars(means, step=st["step"])
            if s == "train" and st.get("groups") and should_log_train_samples(st["step"]):
                _render_group_plots(lg, st["groups"], st["step"], prefix="train_samples")
            elif s == "val" and st.get("groups"):
                # Per-rollout trajectory spread on validation: always render (the
                # val split is never throttled), under the val_samples namespace
                # so it does not collide with the train group overlays.
                _render_group_plots(lg, st["groups"], st["step"], prefix="val_samples")
            st["sums"] = {}
            st["count"] = 0
            st["window"] = 0
            st["groups"] = {}
    except Exception:  # pragma: no cover
        pass


def _render_group_plots(lg: TransfusionSampleLogger, groups: Dict[str, Dict[str, Any]], step: int,
                        prefix: str = "train_samples") -> None:
    """Render the buffered GRPO in-group overlay plots.

    One figure per group: shared GT (red dashed) + the group's ``rollout.n``
    sampled trajectories (tab10 colours, legend carries each trajectory's scalar
    reward) — the per-sample plots only ever show single trajectories, so the
    in-group spread is otherwise invisible. A companion text artefact lists each
    trajectory's full reward-component breakdown. ``prefix`` picks the SwanLab /
    disk key namespace (``train_samples`` or ``val_samples``) so a training step
    and a validation pass never overwrite each other's group overlays.
    """
    for g_idx, (_gkey, g) in enumerate(groups.items()):
        preds = [p for p in g["preds"] if p is not None]
        scores = [s for p, s in zip(g["preds"], g["scores"]) if p is not None]
        gt = g["gt"]
        split = "val" if prefix == "val_samples" else "train"
        mean_r = float(np.mean(scores)) if scores else 0.0
        title = f"{split} group {g_idx} ({len(preds)} samples, mean r={mean_r:.3f})"
        if gt is not None:
            lg.log_figure(
                f"{prefix}/group_{g_idx}_trajs",
                plot_group_trajectories(preds, gt, scores=scores, title=title),
                step,
                caption=title,
            )
        lines = []
        for j, m in enumerate(g["metrics"]):
            kv = ", ".join(f"{k}={float(v):.4f}" for k, v in m.items() if isinstance(v, (int, float)))
            lines.append(f"traj {j}: {kv}")
        lg.log_text(f"{prefix}/group_{g_idx}_rewards", "\n".join(lines), step)


__all__ = [
    "per_step_l2",
    "format_sample_values",
    "plot_trajectory_comparison",
    "plot_group_trajectories",
    "should_log_train_samples",
    "TransfusionSampleLogger",
    "log_sample_best_effort",
    "flush_metric_means",
]
