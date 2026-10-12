"""
Multi-backend training logger (verl-style dispatch).

Supports: wandb, swanlab, tensorboard, console.

Config format (EasyR1 style):
    logger: ["console", "wandb"]
"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================
# Multi-backend Logger
# ============================================================


class TrainingLogger:
    """
    Unified logger supporting wandb / swanlab / tensorboard / console.

    Inspired by verl's ``Tracking`` class — initialises each backend
    and fans out ``.log()`` calls to all active backends.
    """

    def __init__(
        self,
        backends: List[str],
        project: str = "transfusion",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        output_dir: str = "./logs",
        token: Optional[str] = None,  # for backends that require auth (e.g. wandb, swanlab)
    ):
        self.loggers: Dict[str, Any] = {}
        self.backends = backends
        self.output_dir = output_dir

        # Always mirror images / texts to local disk, regardless of remote backends.
        # Files are written under  <output_dir>/samples/step_<step>/<safe_key>.{png,txt}
        self.samples_dir = Path(output_dir) / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        for backend in backends:
            if backend == "wandb":
                try:
                    import wandb
                    wandb.init(project=project, name=run_name, config=config, dir=output_dir)
                    self.loggers["wandb"] = wandb
                    logger.info(f"  [wandb]       run_dir = {wandb.run.dir}")
                except ImportError:
                    logger.warning("wandb not installed, skipping")
            elif backend == "swanlab":
                try:
                    import swanlab
                    sl_dir = str(Path(output_dir) / "swanlab")
                    Path(sl_dir).mkdir(parents=True, exist_ok=True)
                    # If this output_dir already has a swanlab experiment (e.g. we are
                    # resuming training into the SAME <output_dir>/<timestamp> run
                    # directory), continue logging into that SAME experiment instead of
                    # starting a brand-new one -- pass its persisted id back with
                    # resume="allow". On a genuinely fresh run no id file exists yet, so
                    # a new experiment is created and its id is persisted for next time.
                    run_id_file = Path(sl_dir) / "run_id.txt"
                    existing_id = run_id_file.read_text().strip() if run_id_file.exists() else None
                    init_kwargs = dict(project=project, experiment_name=run_name, config=config, logdir=sl_dir)
                    if existing_id:
                        init_kwargs.update(id=existing_id, resume="allow")
                    # SWANLAB_MODE: "offline" (default) keeps a local backup only —
                    # no account/token needed; set to "cloud" to sync to swanlab.cn.
                    init_kwargs["mode"] = os.environ.get("SWANLAB_MODE", "offline")
                    try:
                        swanlab.login(api_key=token)
                    except Exception:
                        pass  # offline mode works without a token
                    run = swanlab.init(**init_kwargs)
                    run_id = getattr(run, "id", None)
                    if run_id:
                        run_id_file.write_text(str(run_id))
                    self.loggers["swanlab"] = swanlab
                    logger.info(
                        f"  [swanlab]     logdir  = {sl_dir}"
                        + (f" (resumed experiment id={existing_id})" if existing_id else "")
                    )
                except ImportError:
                    logger.warning("swanlab not installed, skipping")
                except Exception as e:
                    logger.warning(f"swanlab init failed ({e}); continuing without swanlab")
            elif backend == "tensorboard":
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    tb_dir = str(Path(output_dir) / "tensorboard")
                    self.loggers["tensorboard"] = SummaryWriter(log_dir=tb_dir)
                    logger.info(f"  [tensorboard] logdir  = {tb_dir}")
                except ImportError:
                    logger.warning("tensorboard not installed, skipping")
            elif backend == "console":
                self.loggers["console"] = True
                # Also set up file logging
                log_file = str(Path(output_dir) / "training.log")
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_file, mode="a")
                fh.setLevel(logging.INFO)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                logging.getLogger().addHandler(fh)
                self.log_file = log_file
                logger.info(f"  [console]     enabled (also logging to {log_file})")

        logger.info(f"  [checkpoints] output  = {output_dir}")

    def log(self, data: Dict[str, Any], step: int):
        """Log scalar metrics to all active backends."""
        for name, lg in self.loggers.items():
            if name == "wandb":
                lg.log(data, step=step)
            elif name == "swanlab":
                lg.log(data, step=step)
            elif name == "tensorboard":
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        lg.add_scalar(k, v, global_step=step)
            elif name == "console":
                parts = [f"[step {step}]"]
                for k, v in data.items():
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.4f}")
                    else:
                        parts.append(f"{k}={v}")
                logger.info("  ".join(parts))

    def log_image(self, key: str, fig_or_path, step: int, caption: str = ""):
        """Log a matplotlib figure or image path. Always also mirrors to local disk."""
        # 1) Always save a local copy so eval visualisations survive even
        #    when no remote backend (wandb/swanlab) is enabled.
        self._save_image_local(key, fig_or_path, step, caption)

        for name, lg in self.loggers.items():
            if name == "wandb":
                import wandb
                if isinstance(fig_or_path, plt.Figure):
                    lg.log({key: wandb.Image(fig_or_path, caption=caption)}, step=step)
                else:
                    lg.log({key: wandb.Image(str(fig_or_path), caption=caption)}, step=step)
            elif name == "swanlab":
                import swanlab
                if isinstance(fig_or_path, plt.Figure):
                    buf = io.BytesIO()
                    fig_or_path.savefig(buf, format="png", dpi=100)
                    buf.seek(0)
                    from PIL import Image as PILImage
                    img = PILImage.open(buf)
                    lg.log({key: swanlab.Image(img, caption=caption)}, step=step)
                else:
                    lg.log({key: swanlab.Image(str(fig_or_path), caption=caption)}, step=step)
            elif name == "tensorboard":
                if isinstance(fig_or_path, plt.Figure):
                    lg.add_figure(key, fig_or_path, global_step=step)
            elif name == "console":
                logger.info(f"[step {step}] {key}: (image logged) {caption}")

    def log_text(self, key: str, text: str, step: int):
        """Log text content. Always also mirrors to local disk."""
        self._save_text_local(key, text, step)

        for name, lg in self.loggers.items():
            if name == "wandb":
                import wandb
                lg.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)
            elif name == "swanlab":
                import swanlab
                lg.log({key: swanlab.Text(text)}, step=step)
            elif name == "tensorboard":
                lg.add_text(key, text, global_step=step)
            elif name == "console":
                # display = text[:200] + "..." if len(text) > 200 else text
                display = text
                logger.info(f"[step {step}] {key}: {display}")

    def _step_dir(self, step: int) -> Path:
        d = self.samples_dir / f"step_{step:08d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _safe_key(key: str) -> str:
        # Preserve "group/sample" hierarchy as sub-folders, but sanitise the
        # individual segments so they are valid filenames.
        parts = [p.strip().replace(" ", "_") for p in key.split("/") if p.strip()]
        safe = []
        for p in parts:
            safe.append("".join(c if (c.isalnum() or c in "._-") else "_" for c in p))
        return "/".join(safe) if safe else "item"

    def _save_image_local(self, key: str, fig_or_path, step: int, caption: str = ""):
        try:
            out_path = self._step_dir(step) / (self._safe_key(key) + ".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(fig_or_path, plt.Figure):
                fig_or_path.savefig(out_path, dpi=100, bbox_inches="tight")
            else:
                # Path / str → copy bytes
                src = Path(str(fig_or_path))
                if src.exists():
                    import shutil
                    shutil.copyfile(src, out_path)
            if caption:
                cap_path = out_path.with_suffix(".caption.txt")
                cap_path.write_text(caption, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save local image for key={key} step={step}: {e}")

    def _save_text_local(self, key: str, text: str, step: int):
        try:
            out_path = self._step_dir(step) / (self._safe_key(key) + ".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save local text for key={key} step={step}: {e}")

    def finish(self):
        for name, lg in self.loggers.items():
            if name == "wandb":
                lg.finish()
            elif name == "swanlab":
                lg.finish()
            elif name == "tensorboard":
                lg.close()


# ============================================================
# Visualisation helpers
# ============================================================


def plot_trajectory_comparison(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str = "Trajectory",
) -> plt.Figure:
    """Plot predicted vs GT trajectory, return figure."""
    fig, ax = plt.subplots(figsize=(6, 6))
    d = min(pred.shape[-1], 2)
    ax.plot(gt[:, 0], gt[:, 1] if d > 1 else range(len(gt)), "r--o", ms=4, lw=1.5, label="GT")
    ax.plot(pred[:, 0], pred[:, 1] if d > 1 else range(len(pred)), "b-o", ms=4, lw=1.5, label="Pred")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _make_video_montage(frames, fps: float = 2.0) -> Optional[plt.Figure]:
    """Create a montage figure from a list of PIL Image frames.

    Each frame is titled "t-Xs" (seconds before the current/last frame,
    assuming ``fps``) so the logged montage can be visually cross-checked
    against the video frames actually fed to the model.
    """
    if not frames:
        return None
    n = len(frames)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < n:
            ax.imshow(np.array(frames[idx]))
            offset = (n - idx) / fps
            ax.set_title(f"t-{offset:g}s (frame {idx})", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prediction bbox parsing + visualisation
# ---------------------------------------------------------------------------
# Bboxes can appear in the CoT in two formats:
#   1. "prediction"       -> JSON: {"bbox_2d": [x1, y1, x2, y2], "future_position": "..."}
#   2. "short_prediction" -> tokens: <|sp_bbox|><|sp_x_x1|><|sp_y_y1|><|sp_x_x2|>
#                            <|sp_y_y2|>[<|sp_fp_*|>]<|/sp_bbox|>  (no object: <|object_none|>)
# Coordinates are in the SAME pixel space as the raw video frames (e.g. 2048x512),
# so they are drawn directly on the last frame.
#
# IMPORTANT: generated CoT (especially early in training) may be MALFORMED.
# Only STRICTLY conforming bboxes are returned/drawn; anything that does not
# match the exact expected structure is silently dropped (never drawn).
_SP_BBOX_RE = re.compile(r"<\|sp_bbox\|>(.*?)<\|/sp_bbox\|>", re.S)
# A conforming short-prediction object is EXACTLY: x, y, x, y, optional single fp.
# Anchored (^...$) so any extra/missing/out-of-order token => no match => skip.
_SP_CHUNK_RE = re.compile(
    r"^<\|sp_x_(\d+)\|><\|sp_y_(\d+)\|><\|sp_x_(\d+)\|><\|sp_y_(\d+)\|>"
    r"(?:<\|sp_fp_([a-z_]+)\|>)?$"
)
_JSON_BBOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]'
    r'(?:\s*,\s*"(?:future_position|label)"\s*:\s*"([^"]*)")?'
)


def parse_pred_bboxes(text: Optional[str]) -> List[Tuple[List[float], str]]:
    """Parse STRICTLY-conforming predicted bboxes from a CoT string (either format).

    Returns a list of ``([x1, y1, x2, y2], label)``. Only well-formed bboxes are
    returned; malformed / partial / out-of-order content (common in generated
    CoT early in training, e.g. ``<|object_none|>`` or truncated tokens) yields
    no entry so it is never drawn. Geometry/bounds are validated later, against
    the actual frame, in ``draw_pred_bboxes_on_frame``.
    """
    if not text:
        return []
    out: List[Tuple[List[float], str]] = []
    # --- short_prediction tokens (strict, anchored per object) ---
    if "<|sp_bbox|>" in text:
        for chunk in _SP_BBOX_RE.findall(text):
            m = _SP_CHUNK_RE.match(chunk.strip())
            if not m:
                continue  # not conforming -> skip
            x1, y1, x2, y2 = (float(m.group(k)) for k in range(1, 5))
            label = m.group(5).replace("_", ", ") if m.group(5) else ""
            out.append(([x1, y1, x2, y2], label))
        return out
    # --- prediction JSON ---
    for m in _JSON_BBOX_RE.finditer(text):
        try:
            box = [float(m.group(k)) for k in range(1, 5)]
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in box):
            continue
        label = (m.group(5) or "").strip()
        out.append((box, label))
    return out


def _valid_box(box: List[float], W: int, H: int) -> bool:
    """True only when the box conforms: finite, ordered, inside frame bounds."""
    if box is None or len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box):
        return False
    if x2 <= x1 or y2 <= y1:           # degenerate / inverted
        return False
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:  # outside the frame -> not conforming
        return False
    return True


def draw_pred_bboxes_on_frame(
    frame,
    pred_bboxes: List[Tuple[List[float], str]],
    gt_bboxes: Optional[List[Tuple[List[float], str]]] = None,
    title: str = "",
) -> Optional[plt.Figure]:
    """Draw predicted (red) and optional GT (lime) bboxes on a PIL frame.

    Coordinates are in the frame's own pixel space. Boxes that do NOT conform
    (non-finite, inverted/degenerate, or outside the frame) are silently skipped
    — they are never clamped or drawn. Returns a Figure, or ``None`` when the
    frame is unusable or no conforming box remains to draw.
    """
    import matplotlib.patches as patches

    try:
        img = np.array(frame.convert("RGB"))
    except Exception:
        img = np.array(frame)
    if img.ndim != 3:
        return None
    H, W = img.shape[:2]

    def _draw(ax, boxes, color, tag) -> int:
        drawn = 0
        for box, label in (boxes or []):
            if not _valid_box(box, W, H):
                continue  # not conforming -> do NOT draw
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            txt = f"{tag}: {label}" if label else tag
            ax.text(
                x1, max(0, y1 - 3), txt, color="white", fontsize=7,
                bbox=dict(facecolor=color, alpha=0.6, pad=0.5, edgecolor="none"),
            )
            drawn += 1
        return drawn

    # Pre-check: if nothing conforms, skip the figure entirely.
    n_valid = sum(_valid_box(b, W, H) for b, _ in (pred_bboxes or [])) \
        + sum(_valid_box(b, W, H) for b, _ in (gt_bboxes or []))
    if n_valid == 0:
        return None

    fig, ax = plt.subplots(figsize=(max(W / 100.0, 4), max(H / 100.0, 2)))
    ax.imshow(img)
    _draw(ax, gt_bboxes, "lime", "GT")
    _draw(ax, pred_bboxes, "red", "Pred")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    return fig


def log_data_samples(
    tlogger: TrainingLogger,
    batch: Dict[str, torch.Tensor],
    processor,
    step: int,
    num_samples: int = 3,
    prefix: str = "train_data",
):
    """Log a few training data samples (text, video/image info, GT traj, CoT labels)."""
    B = batch["input_ids"].shape[0]
    n = min(num_samples, B)

    for i in range(n):
        text = processor.tokenizer.decode(
            batch["input_ids"][i], skip_special_tokens=False,
        )
        tlogger.log_text(f"{prefix}/sample_{i}_text", text, step)

        # Log raw video/image frames if available
        raw_frames = batch.get("_raw_video_frames")
        if raw_frames is not None and i < len(raw_frames):
            fig = _make_video_montage(raw_frames[i])
            if fig is not None:
                tlogger.log_image(f"{prefix}/sample_{i}_video_frames", fig, step, caption=f"Video frames (sample {i})")
                plt.close(fig)

        # Log video/image shape info
        info_parts = []
        if "pixel_values_videos" in batch and batch["pixel_values_videos"] is not None:
            info_parts.append(f"pixel_values_videos shape: {list(batch['pixel_values_videos'].shape)}")
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            info_parts.append(f"pixel_values shape: {list(batch['pixel_values'].shape)}")
        if "video_grid_thw" in batch and batch["video_grid_thw"] is not None:
            info_parts.append(f"video_grid_thw: {batch['video_grid_thw'].tolist()}")
        if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
            info_parts.append(f"image_grid_thw: {batch['image_grid_thw'].tolist()}")
        if "second_per_grid_ts" in batch and batch["second_per_grid_ts"] is not None:
            v = batch["second_per_grid_ts"]
            info_parts.append(f"second_per_grid_ts: {v.tolist() if isinstance(v, torch.Tensor) else v}")
        if info_parts:
            tlogger.log_text(f"{prefix}/sample_{i}_media_info", "\n".join(info_parts), step)

        if "actions" in batch and batch["actions"] is not None:
            gt = batch["actions"][i].cpu().numpy()
            gt_str = "\n".join([f"  step {j}: x={gt[j,0]:.4f}, y={gt[j,1]:.4f}" + (f", heading={gt[j,2]:.4f}" if gt.shape[-1] > 2 else "") for j in range(gt.shape[0])])
            tlogger.log_text(f"{prefix}/sample_{i}_gt_values", gt_str, step)

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            d = min(gt.shape[-1], 2)
            ax.plot(gt[:, 0], gt[:, 1] if d > 1 else range(len(gt)), "r-o", ms=4)
            ax.set_title(f"GT Trajectory (sample {i})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            tlogger.log_image(f"{prefix}/sample_{i}_gt_traj", fig, step)
            plt.close(fig)

        if "cot_labels" in batch and batch["cot_labels"] is not None:
            cot = processor.tokenizer.decode(
                batch["cot_labels"][i][batch["cot_labels"][i] != -100],
                skip_special_tokens=False,
            )
            tlogger.log_text(f"{prefix}/sample_{i}_cot", cot, step)


def log_eval_samples(
    tlogger: TrainingLogger,
    batch: Dict[str, torch.Tensor],
    preds: torch.Tensor,
    cot_ids: Optional[torch.Tensor],
    processor,
    step: int,
    num_samples: int = 5,
):
    """Log evaluation samples: text, GT traj, pred traj, pred values, CoT."""
    B = batch["input_ids"].shape[0]
    n = min(num_samples, B)

    for i in range(n):
        text = processor.tokenizer.decode(
            batch["input_ids"][i], skip_special_tokens=False,
        )
        tlogger.log_text(f"val_samples/sample_{i}_text", text, step)

        # Log raw video/image frames if available
        raw_frames = batch.get("_raw_video_frames")
        if raw_frames is not None and i < len(raw_frames):
            fig = _make_video_montage(raw_frames[i])
            if fig is not None:
                tlogger.log_image(f"val_samples/sample_{i}_video_frames", fig, step, caption=f"Video frames (sample {i})")
                plt.close(fig)

        gt = batch["actions"][i].cpu().numpy()
        pred = preds[i].cpu().numpy()

        # Per-sample L2 (xy only): per-timestep distance + average over future horizon.
        # This mirrors the aggregate "L2" metric ( = self.L2 / self.total_data ) but
        # restricted to a single sample, so it is meaningful when only one sample is
        # being logged.
        d = min(gt.shape[-1], pred.shape[-1], 2)
        l2_per_step = np.sqrt(((pred[:, :d] - gt[:, :d]) ** 2).sum(axis=-1))  # [T]
        sample_l2 = float(l2_per_step.mean())
        tlogger.log({f"val_samples/sample_{i}_L2": sample_l2}, step=step)

        # Log numeric values
        val_lines = [f"L2 (mean over {gt.shape[0]} steps): {sample_l2:.4f}",
                     "per-step L2: " + ", ".join(f"{v:.4f}" for v in l2_per_step.tolist()),
                     "GT vs Pred:"]
        for j in range(gt.shape[0]):
            gt_str = ", ".join([f"{gt[j, k]:.4f}" for k in range(gt.shape[-1])])
            pred_str = ", ".join([f"{pred[j, k]:.4f}" for k in range(pred.shape[-1])])
            val_lines.append(f"  step {j}: GT=({gt_str})  Pred=({pred_str})")
        tlogger.log_text(f"val_samples/sample_{i}_values", "\n".join(val_lines), step)

        fig = plot_trajectory_comparison(pred, gt, title=f"Val sample {i}")
        tlogger.log_image(f"val_samples/sample_{i}_traj", fig, step, caption=f"sample {i}")
        plt.close(fig)

        # Log CoT output (strip right-padding to show only real generated tokens)
        pred_cot_text = ""
        if cot_ids is not None and cot_ids.shape[0] > i and cot_ids.shape[1] > 0:
            sample_cot = cot_ids[i]
            # Remove right-padding: find last non-pad token
            pad_id = processor.tokenizer.pad_token_id or 0
            non_pad = (sample_cot != pad_id).nonzero(as_tuple=True)[0]
            if len(non_pad) > 0:
                real_cot = sample_cot[:non_pad[-1].item() + 1]
            else:
                real_cot = sample_cot[:0]  # all padding
            pred_cot_text = processor.tokenizer.decode(real_cot, skip_special_tokens=False)
            tlogger.log_text(f"val_samples/sample_{i}_cot", pred_cot_text, step)
        else:
            tlogger.log_text(f"val_samples/sample_{i}_cot", "(no CoT generated)", step)

        # Visualise predicted (and GT) bboxes on the LAST video frame. Works for
        # both "prediction" (JSON) and "short_prediction" (token) CoT formats.
        if raw_frames is not None and i < len(raw_frames) and raw_frames[i]:
            pred_boxes = parse_pred_bboxes(pred_cot_text)
            gt_cot_text = ""
            gt_traces = batch.get("_reasoning_trace")
            if isinstance(gt_traces, (list, tuple)) and i < len(gt_traces):
                gt_cot_text = gt_traces[i]
            gt_boxes = parse_pred_bboxes(gt_cot_text)
            if pred_boxes or gt_boxes:
                last_frame = raw_frames[i][-1]
                bbox_fig = draw_pred_bboxes_on_frame(
                    last_frame, pred_boxes, gt_boxes,
                    title=f"Pred(red)/GT(green) bbox — sample {i}",
                )
                if bbox_fig is not None:
                    tlogger.log_image(
                        f"val_samples/sample_{i}_pred_bbox", bbox_fig, step,
                        caption=f"pred={len(pred_boxes)} gt={len(gt_boxes)} (last frame)",
                    )
                    plt.close(bbox_fig)
