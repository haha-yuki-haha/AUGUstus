"""Utility helpers for Transfusion training and evaluation."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) for model parameters
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model parameters.

    Shadow copies are stored on **CPU** to avoid doubling GPU memory usage.
    To avoid the overhead of GPU→CPU transfer every step, the update is
    performed only every ``update_every`` steps with an adjusted decay
    ``effective_decay = decay ** update_every``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        update_every: int = 10,
    ):
        self.model = model
        self.decay = decay
        self.update_every = max(update_every, 1)
        self.effective_decay = decay ** self.update_every
        self.step_counter = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self):
        """Copy all trainable parameters to CPU as EMA shadow."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().float().cpu()

    @torch.no_grad()
    def update(self):
        """Update EMA shadow every ``update_every`` steps.

        Uses warm-up: effective decay ramps from 0 to target decay via
        min(decay, 1 - 1/(num_updates + 1)).  This ensures EMA tracks
        recent weights closely during early training, avoiding the problem
        where shadow is dominated by (poor) initial weights.
        """
        self.step_counter += 1
        if self.step_counter % self.update_every != 0:
            return  # skip — no GPU→CPU transfer
        num_updates = self.step_counter // self.update_every
        # Warm-up decay: starts near 0, ramps to self.effective_decay
        decay = min(self.effective_decay, 1.0 - 1.0 / (num_updates + 1))
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_val = param.data.detach().float().cpu()
                self.shadow[name].mul_(decay).add_(
                    new_val, alpha=1.0 - decay
                )

    def apply_shadow(self):
        """Swap model params with EMA shadow (for eval). Backs up current params on CPU."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.detach().cpu()
                param.data.copy_(self.shadow[name].to(param.device, param.dtype))

    def restore(self):
        """Restore original params (from CPU backup) after eval."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].to(param.device, param.dtype))
        self.backup = {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "shadow": self.shadow,
            "decay": self.decay,
            "update_every": self.update_every,
            "step_counter": self.step_counter,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.shadow = {k: v.cpu() for k, v in state["shadow"].items()}
        self.decay = state.get("decay", self.decay)
        self.update_every = state.get("update_every", self.update_every)
        self.effective_decay = self.decay ** self.update_every
        self.step_counter = state.get("step_counter", 0)


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``$VAR`` environment variables in strings.

    Lets config files stay machine-independent: paths are written as e.g.
    ``${AUGUSTUS_DATA_ROOT}/4_data_2048_512/...`` and resolved from the
    environment defined by ``env.sh`` (the single place holding real paths).
    Unknown variables expand to an empty string (like bash ``${VAR:-}``).
    """
    if isinstance(value, str):
        if "$" in value:
            import re as _re
            value = _re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
                lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""),
                value,
            )
            value = os.path.expandvars(value)
        return os.path.expanduser(value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env_vars(cfg)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Cosine LR with linear warmup (§4.1 of the paper)."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_L2(trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
    """
    Per-step L2 displacement between predicted and GT trajectories (xy only).

    Args:
        trajs:    (B, n_future, >=2) tensor — predicted waypoints
        gt_trajs: (B, n_future, >=2) tensor — ground-truth waypoints

    Returns:
        (B, n_future) tensor of L2 distances per timestep.
    """
    diff = trajs[:, :, :2] - gt_trajs[:, :, :2]
    return torch.sqrt((diff ** 2).sum(dim=-1))


def compute_action_metrics(
    pred: torch.Tensor, gt: torch.Tensor, action_dim: int = 2,
) -> Dict[str, float]:
    """
    Compute trajectory metrics.

    If action_dim >= 2 the first two channels are treated as spatial (x, y)
    or (accel, angular_vel).  ADE / FDE use L2 over the first two dims.
    """
    d = min(action_dim, 2)
    disp = torch.norm(pred[:, :, :d] - gt[:, :, :d], dim=-1)  # [B, T]
    l2 = compute_L2(pred, gt)  # [B, T] — sum_xy(diff^2) -> sqrt
    return {
        "ADE": disp.mean().item(),
        "FDE": disp[:, -1].mean().item(),
        # Average L2 over all samples and all future steps (== self.L2 / self.total_data).
        "L2": l2.mean().item(),
    }


# ---------------------------------------------------------------------------
# PDMS (PDM Score) evaluation via NAVSIM metric cache
# ---------------------------------------------------------------------------


_pdm_initialized = False
_pdm_simulator = None
_pdm_scorer = None
_pdm_cache_loader = None

# (metric cache paths are passed in from the config, e.g.
#  data.metric_cache_test = ${NAVSIM_EXP_ROOT}/metric_cache_test)


# Per-cache-path PDM state (simulator / scorer / MetricCacheLoader), created
# lazily on first use. MUST be keyed by the cache path: the RL recipe computes
# PDMS against the TRAIN cache for training rollouts and the TEST cache for
# validation rollouts in the SAME (fork-pooled) processes -- a single global
# loader pinned to whichever cache was initialised first silently misses every
# frame token of the other split (validation PDMS never fired, 2026-07-28).
_pdm_states: Dict[str, Dict[str, Any]] = {}


def _init_pdm(metric_cache_path: Optional[str] = None):
    """Lazily initialize NAVSIM PDM evaluation components FOR ONE cache path.

    Returns True when the state for ``metric_cache_path`` is ready; separate
    states are kept per path so train/test caches coexist in one process.
    """
    if not metric_cache_path:
        return False
    state = _pdm_states.get(metric_cache_path)
    if state is not None:
        return bool(state.get("ok"))
    state = {"ok": False}
    _pdm_states[metric_cache_path] = state
    if not Path(metric_cache_path).exists():
        logger.warning(f"PDMS metric cache not found: {metric_cache_path}")
        return False
    try:
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.common.dataloader import MetricCacheLoader
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig

        proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        state["simulator"] = PDMSimulator(proposal_sampling=proposal_sampling)

        scorer_config = PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=5e-03,
            progress_distance_threshold=5.0,
        )
        state["scorer"] = PDMScorer(proposal_sampling=proposal_sampling, config=scorer_config)
        state["loader"] = MetricCacheLoader(Path(metric_cache_path))
        state["ok"] = True
        return True
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"PDMS evaluation unavailable: {e}")
        return False


def compute_pdm_score_single(
    traj_xy: np.ndarray, heading: np.ndarray, frame_id: str,
    metric_cache_path: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Compute PDM score for a single sample.

    Args:
        traj_xy: [8, 2] array of (x, y) waypoints
        heading: [8] array of heading angles
        frame_id: NAVSIM frame token for metric cache lookup
        metric_cache_path: the split-correct NAVSIM metric cache dir (train vs
            test) -- REQUIRED; selects that cache's loader/simulator/scorer.
    """
    if not _init_pdm(metric_cache_path):
        return None
    state = _pdm_states[metric_cache_path]
    try:
        from navsim.evaluate.pdm_score import pdm_score
        from navsim.planning.metric_caching.metric_cache import MetricCache
        from navsim.common.dataclasses import Trajectory
        from dataclasses import asdict
        import lzma
        import pickle

        cache_file = state["loader"].metric_cache_paths.get(frame_id)
        if cache_file is None:
            return None
        with lzma.open(cache_file, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        trajectory_data = np.hstack([traj_xy, heading.reshape(-1, 1)])
        trajectory = Trajectory(trajectory_data)
        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=state["simulator"].proposal_sampling,
            simulator=state["simulator"],
            scorer=state["scorer"],
        )
        return asdict(pdm_result)
    except Exception as e:
        logger.debug(f"PDM score failed for frame {frame_id}: {e}")
        return None


_PDM_POOL = None
_PDM_POOL_WORKERS = 0


def _get_pdm_pool(num_workers: int):
    """Persistent ProcessPoolExecutor for PDMS scoring (see compute_pdm_scores_batch).

    Recreated only when the requested worker count changes; never shut down on
    the per-step path (process exit cleans it up).
    """
    global _PDM_POOL, _PDM_POOL_WORKERS
    from concurrent.futures import ProcessPoolExecutor

    if _PDM_POOL is None or _PDM_POOL_WORKERS != int(num_workers):
        if _PDM_POOL is not None:
            _PDM_POOL.shutdown(wait=False)
        _PDM_POOL = ProcessPoolExecutor(max_workers=int(num_workers))
        _PDM_POOL_WORKERS = int(num_workers)
    return _PDM_POOL


def compute_pdm_scores_batch(
    preds: np.ndarray,
    frame_ids: List[str],
    num_workers: int = 8,
    metric_cache_path: Optional[str] = None,
    return_per_sample: bool = False,
    return_count: bool = False,
    show_progress: bool = True,
) -> Dict[str, float]:
    """Compute PDMS metrics for a batch of predictions.

    Uses multiprocessing to parallelize the expensive per-sample PDM computation
    (each sample requires lzma decompression + simulation + scoring).

    Args:
        preds: [N, 8, 3] array of (x, y, heading)
        frame_ids: list of N frame_id tokens
        num_workers: number of parallel worker processes
        metric_cache_path: path to NAVSIM metric cache directory (None=default)
        return_per_sample: if True, include "per_sample" key with list of per-sample dicts
        return_count: if True, include "count" key = number of SUCCESSFULLY scored
            samples. Needed to reduce this call's averages across ranks when the
            batch is a per-rank shard (final avg = sum_ranks(avg*count)/sum_ranks(count)).
        show_progress: draw the tqdm progress bar. Set False on non-main ranks so
            only one bar is printed when several ranks score disjoint shards.

    Returns:
        Dict with PDMS, NC, DAC, DDC, EP, TTC, C averages.
        If return_count, also includes "count": int (successfully scored samples).
        If return_per_sample, also includes "per_sample": list of N dicts (None for failed samples).
    """
    if not _init_pdm(metric_cache_path):
        return {}

    from concurrent.futures import as_completed
    from tqdm.auto import tqdm

    keys = ["score", "no_at_fault_collisions", "drivable_area_compliance",
            "driving_direction_compliance", "ego_progress",
            "time_to_collision_within_bound", "comfort"]
    totals = {k: 0.0 for k in keys}
    count = 0
    per_sample = [None] * len(preds) if return_per_sample else None

    # Build argument list with original index
    args_list = []
    for i in range(len(preds)):
        fid = frame_ids[i] if i < len(frame_ids) else ""
        if not fid:
            continue
        traj_xy = preds[i, :, :2]
        heading = preds[i, :, 2] if preds.shape[-1] > 2 else np.zeros(preds.shape[1])
        args_list.append((i, traj_xy, heading, fid))

    # Use multiprocessing for speed. The pool is PERSISTENT (module-level): a
    # per-call 128-process pool pays several seconds of fork+import startup on
    # EVERY RL step; the warm pool is reused across calls.
    if num_workers > 1 and len(args_list) > 100:
        executor = _get_pdm_pool(num_workers)
        futures = {
            executor.submit(compute_pdm_score_single, xy, h, fid, metric_cache_path): idx
            for idx, xy, h, fid in args_list
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Computing PDMS", leave=False, disable=not show_progress,
        ):
            result = future.result()
            orig_idx = futures[future]
            if result is not None:
                for k in keys:
                    totals[k] += result.get(k, 0.0)
                count += 1
                if per_sample is not None:
                    per_sample[orig_idx] = result
    else:
        # Sequential fallback (with progress bar)
        for idx, traj_xy, heading, fid in tqdm(
            args_list, desc="Computing PDMS", leave=False, disable=not show_progress,
        ):
            result = compute_pdm_score_single(traj_xy, heading, fid, metric_cache_path)
            if result is not None:
                for k in keys:
                    totals[k] += result.get(k, 0.0)
                count += 1
                if per_sample is not None:
                    per_sample[idx] = result

    if count == 0:
        return {}

    short_names = {
        "score": "PDMS",
        "no_at_fault_collisions": "NC",
        "drivable_area_compliance": "DAC",
        "driving_direction_compliance": "DDC",
        "ego_progress": "EP",
        "time_to_collision_within_bound": "TTC",
        "comfort": "C",
    }
    result = {short_names[k]: totals[k] / count for k in keys}
    if return_count:
        result["count"] = count
    if return_per_sample:
        # Convert per-sample results to short names
        per_sample_short = []
        for ps in per_sample:
            if ps is not None:
                per_sample_short.append({short_names[k]: ps.get(k, 0.0) for k in keys})
            else:
                per_sample_short.append(None)
        result["per_sample"] = per_sample_short
    return result


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0,
    }


def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Action normalization statistics
# ---------------------------------------------------------------------------


def compute_action_norm_stats(
    cache_dir: str,
    action_dim: int = 3,
    action_repr: str = "absolute",
) -> Dict[str, List[float]]:
    """Compute per-dimension q001/q999 quantile statistics from an HF cache dataset.

    Following openpi (pi0), we use percentiles to define the normalization
    range. We use q0.1/q99.9 (0.1th/99.9th percentile) for tighter bounds,
    which is important for scenarios with stopped vehicles (x near 0).
    Values are mapped to [-1, 1] via:
        normalized = (x - q01) / (q99 - q01 + eps) * 2 - 1

    Also computes per-timestep statistics so each of the T waypoints
    uses the full [-1, 1] range, avoiding dynamic range waste when
    early steps have small magnitudes and later steps have large ones.

    Args:
        cache_dir: Path to HF cache dataset directory.
        action_dim: Number of action dimensions.
        action_repr: "absolute" or "delta". If "delta", converts absolute
            actions to per-step increments before computing statistics.

    Returns:
        {"q01": [...], "q99": [...], "mean": [...], "std": [...],
         "q01_per_t": [[...], ...], "q99_per_t": [[...], ...],
         "action_repr": str}
    """
    from datasets import load_from_disk

    cache_path = Path(cache_dir)
    batch_dirs = sorted(cache_path.glob("batch_*"))
    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found in {cache_dir}")

    # Only load the column we need to avoid loading images into memory
    all_actions = []
    for bd in batch_dirs:
        ds = load_from_disk(str(bd))
        col = ds["future_x_y_angle"]  # list of JSON strings
        batch_actions = [json.loads(s) for s in col]
        all_actions.extend(batch_actions)
        del ds, col, batch_actions

    all_actions = np.array(all_actions, dtype=np.float32)  # [N, T, action_dim]

    # Convert to delta representation if requested
    if action_repr == "delta":
        delta = all_actions.copy()
        delta[:, 1:, :] = all_actions[:, 1:, :] - all_actions[:, :-1, :]
        all_actions = delta
        logger.info("Computing stats on delta (per-step increment) actions")

    # Per-timestep statistics
    T = all_actions.shape[1]
    q01_per_t = []
    q99_per_t = []
    for t_idx in range(T):
        step_data = all_actions[:, t_idx, :]  # [N, D]
        q01_per_t.append(np.percentile(step_data, 0.1, axis=0).tolist()[:action_dim])
        q99_per_t.append(np.percentile(step_data, 99.9, axis=0).tolist()[:action_dim])

    # Global statistics (flattened across all timesteps)
    flat = all_actions.reshape(-1, all_actions.shape[-1])  # [N*T, D]
    q01 = np.percentile(flat, 0.1, axis=0).tolist()
    q99 = np.percentile(flat, 99.9, axis=0).tolist()
    mean = flat.mean(axis=0).tolist()
    std = flat.std(axis=0).tolist()

    return {
        "q01": q01[:action_dim],
        "q99": q99[:action_dim],
        "mean": mean[:action_dim],
        "std": std[:action_dim],
        "q01_per_t": q01_per_t,
        "q99_per_t": q99_per_t,
        "action_repr": action_repr,
    }


def load_or_compute_action_norm_stats(
    stats_file: Optional[str],
    train_data_path: str,
    action_dim: int = 3,
    action_repr: str = "absolute",
) -> Dict[str, List[float]]:
    """Load action normalization stats from JSON, or compute and save them.

    If the loaded file is missing per-timestep stats (q01_per_t, q99_per_t),
    or the action_repr doesn't match, recompute from training data.

    Args:
        stats_file: Path to JSON file with precomputed stats. If None or
            file doesn't exist, stats are computed from training data.
        train_data_path: Path to HF cache training data directory.
        action_dim: Number of action dimensions.
        action_repr: "absolute" or "delta". Stats are computed in this space.

    Returns:
        {"q01": [...], "q99": [...], "mean": [...], "std": [...],
         "q01_per_t": [[...], ...], "q99_per_t": [[...], ...],
         "action_repr": str}
    """
    need_recompute = False
    if stats_file and Path(stats_file).exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
        # Recompute if: missing per-timestep stats, or action_repr changed
        saved_repr = stats.get("action_repr", "absolute")
        if "q01_per_t" not in stats or saved_repr != action_repr:
            reason = "per-timestep stats missing" if "q01_per_t" not in stats else f"action_repr changed ({saved_repr} -> {action_repr})"
            logger.info(f"Recomputing stats: {reason}")
            need_recompute = True
        else:
            logger.info(f"Loaded action norm stats from {stats_file}: "
                         f"q01={stats['q01']}, q99={stats['q99']}, repr={saved_repr}")
            return stats
    else:
        need_recompute = True

    logger.info(f"Computing action normalization statistics (repr={action_repr})...")
    stats = compute_action_norm_stats(train_data_path, action_dim, action_repr=action_repr)
    logger.info(f"Computed stats: q01={stats['q01']}, q99={stats['q99']}, "
                f"mean={stats['mean']}, std={stats['std']}")

    if stats_file:
        Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved action norm stats to {stats_file}")

    return stats
