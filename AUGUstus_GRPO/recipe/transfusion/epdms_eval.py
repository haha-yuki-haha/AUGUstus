"""Official NAVSIM v2 EPDMS (Extended PDM Score) for the Transfusion RL recipe.

The recipe's PDMS reward uses the **v1** ``navsim`` package (imported in the very
same process); the official EPDMS pipeline lives in the **v2** repo
(``Code/eval_tmp/navsim_v2``) — a second, incompatible ``navsim`` package. Both
cannot coexist in one interpreter, so ALL v2 work runs in a dedicated
spawn-context process pool whose ``sys.path`` is prepended with the v2 root
BEFORE any ``navsim`` import (v1 is installed via a plain ``.pth`` path entry,
so prepending wins — the same trick as ``Transfusion/test_EPDMS.py``).

Everything (scorer config, log-replay traffic, human penalty filter ON,
``PDMSimulator`` / ``PDMScorer`` / one-stage aggregation) mirrors
``Transfusion/test_EPDMS.py`` VERBATIM — EPDMS is the official NAVSIM v2 metric
with fixed official weights (EP=5.0 inside); it NEVER reads any recipe knob.

Entry points:

* :func:`epdms_score_per_frame` — official **PER-FRAME** EPDMS for the
  TRAINING reward. With ``with_gt_prev_ec=True`` (the training default) each
  frame that has an adjacent predecessor (same log, <=0.55s, ORIGINAL frames
  only — the official ``infer_start_adjacent_mapping`` rule) additionally gets
  the two-frame Extended Comfort computed against the **GT trajectory of the
  previous frame** (log-replay keeps the reward independent of other rollouts)
  and the EC-aware per-frame score is returned together with an ``EC``
  sub-metric. Frames without a predecessor keep the one-stage score (the
  official handling — EC excluded, weight removed).

* :func:`epdms_batch_rows` + :func:`epdms_finalize_rows` — the two halves of
  the FULL official one-stage evaluation (per-frame scoring, then
  ``SceneAggregator`` two-frame EC chaining + ``compute_final_scores`` over
  ALL frames, model predictions on both sides). Used for VALIDATION: shards
  score their frames incrementally and buffer the raw rows; the driver merges
  every shard's rows and finalizes ONCE, so cross-rank chains stay intact.

* :func:`epdms_score_official` — convenience = batch_rows + finalize (offline
  full-set number, same as ``Transfusion/test_EPDMS.py``).

No verl / torch dependency; everything heavy (nuplan/navsim v2/pandas) is
imported lazily inside the pool children.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as _mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SUBSCORE_COLS = ["no_at_fault_collisions", "drivable_area_compliance",
                 "driving_direction_compliance", "traffic_light_compliance",
                 "ego_progress", "time_to_collision_within_bound", "lane_keeping",
                 "history_comfort"]
SHORT_NAMES = {
    "pdm_score": "EPDMS",
    "score": "EPDMS",
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "driving_direction_compliance": "DDC",
    "traffic_light_compliance": "TLC",
    "ego_progress": "EP",
    "time_to_collision_within_bound": "TTC",
    "lane_keeping": "LK",
    "history_comfort": "HC",
    "two_frame_extended_comfort": "EC",
}
SUMMARY_COLS = SUBSCORE_COLS + ["two_frame_extended_comfort", "score"]
DEFAULT_NAVSIM_V2_ROOT = os.environ.get(
    "NAVSIM_V2_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "AUGUstus_SFT", "third_party", "navsim_v2")),
)
# Official one-stage column layout: weighted_metrics[4] is the two-frame
# Extended Comfort slot (WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT == 4)
# and its official weight inside the one-stage aggregation.
_EC_INDEX = 4
_EC_WEIGHT = 2.0
_ADJACENCY_GAP_S = 0.55

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_STATE: Dict[str, Any] = {}


def _v2_worker_init(repo_root: str, navsim_v2_root: str, metric_cache_path: str, human_penalty_filter: bool) -> None:
    """Pool initializer: make THE MODULE importable + prepend the v2 root BEFORE
    any ``navsim`` import, then build the official one-stage evaluation
    components (verbatim test_EPDMS.py)."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if navsim_v2_root not in sys.path:
        sys.path.insert(0, navsim_v2_root)
    from pathlib import Path
    import navsim
    assert "navsim_v2" in str(Path(navsim.__file__).resolve()), (
        f"navsim resolved to {navsim.__file__}, expected the v2 repo ({navsim_v2_root})"
    )
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    assert int(WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT) == _EC_INDEX
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    scorer_config = PDMScorerConfig(
        progress_weight=5.0,
        ttc_weight=5.0,
        lane_keeping_weight=2.0,
        history_comfort_weight=2.0,
        two_frame_extended_comfort_weight=2.0,
        driving_direction_horizon=1.0,
        driving_direction_compliance_threshold=2.0,
        driving_direction_violation_threshold=6.0,
        stopped_speed_threshold=0.005,
        future_collision_horizon_window=1.0,
        progress_distance_threshold=5.0,
        lane_keeping_deviation_limit=0.5,
        lane_keeping_horizon_window=2.0,
        human_penalty_filter=human_penalty_filter,
    )
    _STATE.clear()
    _STATE.update({
        "sampling": proposal_sampling,
        "simulator": PDMSimulator(proposal_sampling=proposal_sampling),
        "scorer": PDMScorer(proposal_sampling=proposal_sampling, config=scorer_config),
        "traffic": LogReplayTrafficAgents(proposal_sampling),
        "loader": MetricCacheLoader(Path(metric_cache_path)),
        "gt_prev_states": {},
        "metric_cache_path": metric_cache_path,
    })


def _gt_prev_frame_states(prev_token: str) -> Optional[Tuple[np.ndarray, float]]:
    """(worker) GT ego-simulated states + start time of a previous frame.

    The GT future trajectory of the previous frame is rolled out with the same
    one-stage components as the model trajectory (``pdm_score`` with
    ``pred = metric_cache.trajectory``); results are memoized in ``_STATE`` —
    a previous frame's GT rollout never changes, and every frame chains to it
    at most once per epoch per worker... so this cache does the heavy lifting.
    """
    cache = _STATE["gt_prev_states"]
    entry = cache.get(prev_token)
    if entry is not None:
        return entry
    from navsim.evaluate.pdm_score import pdm_score_from_interpolated_trajectory
    metric_cache = _STATE["loader"].get_from_token(prev_token)
    _, gt_states = pdm_score_from_interpolated_trajectory(
        metric_cache=metric_cache,
        pred_trajectory=metric_cache.trajectory,
        future_sampling=_STATE["sampling"],
        simulator=_STATE["simulator"],
        scorer=_STATE["scorer"],
        traffic_agents_policy=_STATE["traffic"],
    )
    entry = (np.asarray(gt_states, dtype=np.float64), float(metric_cache.timepoint.time_s))
    cache[prev_token] = entry
    return entry


def _two_frame_ec(current_states: np.ndarray, prev_states: np.ndarray, observation_interval_s: float) -> Optional[float]:
    """(worker) Two-frame Extended Comfort — verbatim
    ``SceneAggregator._compute_two_frame_comfort`` (one-stage branch)."""
    if not 0.0 < observation_interval_s < _ADJACENCY_GAP_S:
        return None
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort
    interval_length = _STATE["sampling"].interval_length
    overlap_start = round(observation_interval_s / interval_length)
    if overlap_start <= 0 or overlap_start >= prev_states.shape[0]:
        return None
    current_overlap = current_states[:-overlap_start]
    prev_overlap = prev_states[overlap_start:]
    time_point_s = np.arange(current_overlap.shape[0]) * interval_length
    ec = ego_is_two_frame_extended_comfort(current_overlap[:], prev_overlap[:], time_point_s)[0]
    return float(ec)


def _collect_frame_meta(tokens: Sequence[str]) -> List[Tuple[str, Optional[str], Optional[float], Optional[str]]]:
    """(worker) (token, log_name, start_time_s, frame_type) per token — the
    metadata needed to build the previous-token mapping (one-time scan of the
    whole v2 metric cache, parallelized over the pool)."""
    loader = _STATE["loader"]
    out = []
    for token in tokens:
        try:
            mc = loader.get_from_token(token)
            out.append((str(token), str(mc.log_name), float(mc.timepoint.time_s), str(mc.scene_type.name)))
        except Exception:
            out.append((str(token), None, None, None))
    return out


def _score_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Score one predicted trajectory with the official v2 ``pdm_score``.

    Runs inside a pool child (components in _STATE). Hard rule: return ONLY
    primitives / numpy arrays — nothing owned by the v2 ``navsim`` package may
    cross the process boundary (unpickling it in the v1 parent would import the
    v2 modules there and collide with v1).
    """
    token = job["token"]
    row = {"token": token, "valid": False}
    try:
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        metric_cache = _STATE["loader"].get_from_token(token)
        score_row, ego_simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(np.asarray(job["poses"], dtype=np.float64)),
            future_sampling=_STATE["sampling"],
            simulator=_STATE["simulator"],
            scorer=_STATE["scorer"],
            traffic_agents_policy=_STATE["traffic"],
        )
        rec = score_row.iloc[0]
        for col in SUBSCORE_COLS + ["multiplicative_metrics_prod", "pdm_score"]:
            row[col] = float(rec[col])
        row["weighted_metrics"] = np.asarray(rec["weighted_metrics"], dtype=np.float64).tolist()
        row["weighted_metrics_array"] = np.asarray(rec["weighted_metrics_array"], dtype=np.float64).tolist()
        row["valid"] = True
        row["log_name"] = str(metric_cache.log_name)
        row["frame_type"] = str(metric_cache.scene_type.name)
        row["start_time"] = float(metric_cache.timepoint.time_s)
        row["ego_simulated_states"] = np.asarray(ego_simulated_states, dtype=np.float64)
        prev_token = job.get("prev_token")
        if row.get("valid") and prev_token:
            try:
                prev = _gt_prev_frame_states(str(prev_token))
                if prev is not None:
                    ec = _two_frame_ec(
                        row["ego_simulated_states"], prev[0],
                        row["start_time"] - prev[1],
                    )
                    if ec is not None:
                        row["EC"] = ec
            except Exception as e:
                row["ec_error"] = f"{type(e).__name__}: {e}"
        return row
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row


def _finalize_official(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inside a pool child: official two-frame EC chaining + final scores.

    Verbatim the official flow reimplemented by Transfusion/test_EPDMS.py
    (``infer_start_adjacent_mapping`` + ``add_two_frame_extended_comfort`` +
    ``compute_final_scores``), minus the pieces already done on scoring.
    """
    import pandas as pd
    from navsim.common.enums import SceneFrameType
    from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
    df = pd.DataFrame(rows)
    df_valid = df[df["valid"]].copy()
    if df_valid.empty:
        return {"per_frame": [], "summary": {}}
    df_valid["weighted_metrics"] = df_valid["weighted_metrics"].apply(np.asarray)
    df_valid["weighted_metrics_array"] = df_valid["weighted_metrics_array"].apply(np.asarray)
    df_valid["frame_type"] = df_valid["frame_type"].map(
        lambda s: getattr(SceneFrameType, str(s).rsplit(".", 1)[-1], None)
    )
    adjacent_mapping = {}
    original = df_valid[df_valid["frame_type"] == SceneFrameType.ORIGINAL]
    for _log_name, group_df in original.groupby("log_name"):
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)
        for i in range(1, len(group_df)):
            if abs(float(group_df.iloc[i]["start_time"]) - float(group_df.iloc[i - 1]["start_time"])) <= 0.55:
                adjacent_mapping[group_df.iloc[i]["token"]] = group_df.iloc[i - 1]["token"]
    full_df = df_valid.copy()
    full_df["two_frame_extended_comfort"] = np.nan
    full_df = full_df.set_index("token")
    updates = []
    for now_frame, previous_frame in adjacent_mapping.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            score_df=full_df,
            proposal_sampling=_STATE["sampling"],
        )
        updates.append(aggregator.aggregate_scores(one_stage_only=True))
    if updates:
        full_df.update(pd.concat(updates, ignore_index=True).set_index("token"))
    full_df.reset_index(inplace=True)
    full_df = full_df.drop(columns=["ego_simulated_states"])
    tf_scores = full_df["two_frame_extended_comfort"].to_numpy()
    wm = np.stack(full_df["weighted_metrics"].to_numpy())
    wa = np.stack(full_df["weighted_metrics_array"].to_numpy())
    mask = np.isnan(tf_scores)
    ec_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    wm[mask, ec_idx] = 0.0
    wa[mask, ec_idx] = 0.0
    wm[~mask, ec_idx] = tf_scores[~mask]
    total_weight = wa.sum(axis=1)
    total_weight[total_weight == 0.0] = np.nan
    full_df["score"] = full_df["multiplicative_metrics_prod"].to_numpy() * ((wm * wa).sum(axis=1) / total_weight)
    per_frame = []
    for _, r in full_df.iterrows():
        per_frame.append({
            "token": r["token"],
            "valid": True,
            **{SHORT_NAMES[c]: float(r[c]) for c in SUBSCORE_COLS + ["two_frame_extended_comfort"]},
            **{"EPDMS": float(r["score"]), "EPDMS_one_stage": float(r["pdm_score"])},
        })
    summary = full_df[SUBSCORE_COLS + ["two_frame_extended_comfort", "score"]].mean(skipna=True)
    return {
        "per_frame": per_frame,
        "summary": {SHORT_NAMES[c]: float(summary[c]) for c in SUBSCORE_COLS + ["two_frame_extended_comfort", "score"]},
    }


_POOL: Optional[ProcessPoolExecutor] = None
_POOL_KEY = None


def _get_pool(navsim_v2_root: str, metric_cache_path: str, human_penalty: bool, workers: int):
    global _POOL, _POOL_KEY
    key = (navsim_v2_root, metric_cache_path, bool(human_penalty), int(workers))
    if _POOL is not None and _POOL_KEY == key:
        return _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False)
    _POOL = ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=_mp.get_context("spawn"),
        initializer=_v2_worker_init,
        initargs=(_REPO_ROOT, navsim_v2_root, metric_cache_path, bool(human_penalty)),
    )
    _POOL_KEY = key
    return _POOL


def _make_jobs(trajs: np.ndarray, tokens: Sequence[str], prev_map: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    jobs = []
    for i, (poses, token) in enumerate(zip(trajs, tokens)):
        if not token:
            continue
        arr = np.asarray(poses, dtype=np.float64)
        if arr.ndim != 2:
            continue
        if arr.shape[1] != 3:
            continue
        if not np.isfinite(arr).all():
            continue
        job = {"token": str(token), "poses": arr, "_idx": i}
        if prev_map:
            prev = prev_map.get(str(token))
            if prev:
                job["prev_token"] = prev
        jobs.append(job)
    return jobs


_PREV_MAPS: Dict[str, Dict[str, str]] = {}


def _prev_map_path(metric_cache_path: str) -> str:
    return os.path.join(metric_cache_path, "epdms_prev_token_map.json")


def _metric_cache_tokens(metric_cache_path: str) -> List[str]:
    """All tokens of a v2 metric cache — the metadata CSV's token column,
    exactly the rule ``MetricCacheLoader._load_metric_cache_paths`` applies."""
    meta_dir = os.path.join(metric_cache_path, "metadata")
    csvs = sorted(f for f in os.listdir(meta_dir) if f.endswith(".csv"))
    tokens = []
    for fn in csvs:
        with open(os.path.join(meta_dir, fn)) as f:
            lines = f.read().splitlines()[1:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            tokens.append(line.rstrip("/").split("/")[-2])
    return tokens


def _build_prev_map_from_meta(
    metas: Iterable[Tuple[str, Optional[str], Optional[float], Optional[str]]],
    gap_threshold_s: float = _ADJACENCY_GAP_S,
) -> Dict[str, str]:
    """Official ``infer_start_adjacent_mapping``: group ORIGINAL frames by
    log, sort by start time, chain each frame to its predecessor when they are
    at most ``gap_threshold_s`` apart. Pure — unit-testable without v2."""
    by_log = {}
    for token, log_name, time_s, frame_type in metas:
        if not log_name or time_s is None or frame_type != "ORIGINAL":
            continue
        by_log.setdefault(log_name, []).append((float(time_s), str(token)))
    mapping = {}
    for frames in by_log.values():
        frames.sort()
        for i in range(1, len(frames)):
            dt = frames[i][0] - frames[i - 1][0]
            if not 0.0 < dt <= gap_threshold_s:
                continue
            mapping[frames[i][1]] = frames[i - 1][1]
    return mapping


def _ensure_prev_map(navsim_v2_root: str, metric_cache_path: str, human_penalty: bool, workers: int) -> Dict[str, str]:
    """Load (or one-time build + persist) the previous-token mapping.

    The full-cache metadata scan is RESUMABLE: every scanned chunk is
    appended to a ``<map>.meta.jsonl`` sidecar first, so a crashed scan (e.g.
    the pool dying under host memory pressure) continues where it stopped. The
    sidecar is deleted once the final map JSON is written."""
    cached = _PREV_MAPS.get(metric_cache_path)
    if cached is not None:
        return cached
    path = _prev_map_path(metric_cache_path)
    if not os.path.exists(path):
        tokens = _metric_cache_tokens(metric_cache_path)
        sidecar = f"{path}.meta.jsonl"
        metas = []
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                try:
                    for line in f:
                        metas.append(tuple(json.loads(line)))
                except (ValueError, TypeError):
                    pass
        done = {m[0] for m in metas}
        todo = [t for t in tokens if t not in done]
        if todo:
            pool = _get_pool(navsim_v2_root, metric_cache_path, human_penalty, workers)
            logger.warning(
                "[epdms] building previous-token map for %s (%d tokens, %d already scanned, one-time scan)...",
                metric_cache_path, len(tokens), len(done),
            )
            chunks = [todo[i:i + 512] for i in range(0, len(todo), 512)]
            with open(sidecar, "a") as fh:
                for batch in pool.map(_collect_frame_meta, chunks):
                    for rec in batch:
                        fh.write(json.dumps(rec) + "\n")
                        metas.append(tuple(rec))
                    fh.flush()
        mapping = _build_prev_map_from_meta(metas)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(mapping, f)
        os.replace(tmp, path)
        try:
            os.remove(sidecar)
        except OSError:
            pass
        logger.warning("[epdms] previous-token map: %d chained frames -> %s", len(mapping), path)
    with open(path) as f:
        mapping = {str(k): str(v) for k, v in json.load(f).items()}
    _PREV_MAPS[metric_cache_path] = mapping
    return mapping


def _inject_ec_score(row: Dict[str, Any]) -> Optional[float]:
    """Recompute a per-frame EPDMS with the two-frame EC slot filled —
    identical maths to ``compute_final_scores`` for chained frames
    (``weighted_metrics[EC] = ec``; the EC weight stays at its official 2.0,
    i.e. the weighted average denominator becomes 5+5+2+2+2). Parent-side and
    v2-free; unit-testable."""
    try:
        ec = float(row["EC"])
        wm = np.asarray(row["weighted_metrics"], dtype=np.float64).copy()
        wa = np.asarray(row["weighted_metrics_array"], dtype=np.float64).copy()
        mult = float(row["multiplicative_metrics_prod"])
        if not np.isfinite(ec) or wm.shape != wa.shape or wm.size <= _EC_INDEX:
            return None
        wm[_EC_INDEX] = ec
        if wa[_EC_INDEX] <= 0.0:
            wa[_EC_INDEX] = _EC_WEIGHT
        total = float(wa.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None
        return float(mult * ((wm * wa).sum() / total))
    except (KeyError, TypeError, ValueError):
        return None


def per_frame_components(row: Dict[str, Any]) -> Dict[str, float]:
    """Short-name per-frame dict from a raw ``_score_one`` row
    (``{"EPDMS", "NC", ..., "HC"}`` + ``"EC"`` when the job chained a previous
    frame, in which case EPDMS is the EC-injected score — same semantics as
    the official final per-frame score)."""
    out = {SHORT_NAMES[c]: float(row[c]) for c in SUBSCORE_COLS}
    epdms = float(row["pdm_score"])
    if row.get("EC") is not None:
        final = _inject_ec_score(row)
        if final is not None:
            epdms = final
            out["EC"] = float(row["EC"])
    out["EPDMS"] = epdms
    return out


def epdms_available(navsim_v2_root: Optional[str] = None, metric_cache_path: Optional[str] = None) -> bool:
    """Cheap pre-flight check (no navsim import in the parent)."""
    root = navsim_v2_root or DEFAULT_NAVSIM_V2_ROOT
    return bool(metric_cache_path) and os.path.isdir(os.path.join(root, "navsim")) and os.path.isdir(metric_cache_path)


def epdms_score_per_frame(
    trajs: np.ndarray,
    tokens: Sequence[str],
    metric_cache_path: str,
    *,
    navsim_v2_root: Optional[str] = None,
    num_workers: int = 64,
    human_penalty_filter: bool = True,
    with_gt_prev_ec: bool = False,
) -> List[Optional[Dict[str, float]]]:
    """Official PER-FRAME EPDMS for a batch (training reward path).

    Returns (aligned to the inputs) one dict per sample with
    ``{"EPDMS", "NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC"}`` — the
    official v2 one-stage per-frame score — plus, with
    ``with_gt_prev_ec=True``, an ``"EC"`` entry and the EC-injected EPDMS for
    every frame that has an adjacent ORIGINAL predecessor (two-frame EC
    against the GT rollout of that previous frame; frames without a
    predecessor keep the one-stage number, the official handling). Any failure
    of the EC machinery degrades to the plain one-stage result. ``None`` for
    samples that could not be scored.
    """
    trajs = np.asarray(trajs, dtype=np.float64)
    n = len(trajs)
    out = [None] * n
    root = navsim_v2_root or DEFAULT_NAVSIM_V2_ROOT
    if not epdms_available(root, metric_cache_path):
        return out
    prev_map = None
    if with_gt_prev_ec:
        try:
            prev_map = _ensure_prev_map(root, metric_cache_path, human_penalty_filter, num_workers)
        except Exception as e:
            logger.warning("[epdms] previous-token map unavailable; EC skipped: %s", e)
            prev_map = None
    jobs = _make_jobs(trajs, tokens, prev_map)
    if not jobs:
        return out
    pool = _get_pool(root, metric_cache_path, human_penalty_filter, num_workers)
    rows = pool.map(_score_one, jobs, chunksize=8)
    for job, row in zip(jobs, rows):
        if not row.get("valid"):
            continue
        out[job["_idx"]] = per_frame_components(row)
    return out


def epdms_batch_rows(
    trajs: np.ndarray,
    tokens: Sequence[str],
    metric_cache_path: str,
    *,
    navsim_v2_root: Optional[str] = None,
    num_workers: int = 64,
    human_penalty_filter: bool = True,
) -> List[Optional[Dict[str, Any]]]:
    """Per-frame scoring half of the official evaluation: raw ``_score_one``
    rows (aligned to the inputs, ``None`` for unscorable samples), keeping the
    full primitives (weighted metrics, log/time, ``ego_simulated_states``) so
    the caller can buffer them and run :func:`epdms_finalize_rows` later over
    the COMPLETE set — validation shards stream batches but two-frame EC must
    chain across batch (and rank) boundaries."""
    trajs = np.asarray(trajs, dtype=np.float64)
    out = [None] * len(trajs)
    root = navsim_v2_root or DEFAULT_NAVSIM_V2_ROOT
    if not epdms_available(root, metric_cache_path):
        return out
    jobs = _make_jobs(trajs, tokens)
    if not jobs:
        return out
    pool = _get_pool(root, metric_cache_path, human_penalty_filter, num_workers)
    for job, row in zip(jobs, pool.map(_score_one, jobs, chunksize=8)):
        if not row.get("valid"):
            continue
        out[job["_idx"]] = row
    return out


def epdms_finalize_rows(
    rows: Iterable[Dict[str, Any]],
    metric_cache_path: str,
    *,
    navsim_v2_root: Optional[str] = None,
    num_workers: int = 64,
    human_penalty_filter: bool = True,
) -> Dict[str, Any]:
    """Two-frame EC aggregation half of the official evaluation, over the
    union of every shard's raw rows: ``infer_start_adjacent_mapping`` +
    ``SceneAggregator`` + ``compute_final_scores`` (official). Returns
    ``{"per_frame": [...], "summary": {...}}`` like :func:`epdms_score_official`."""
    valid = [r for r in rows if r and r.get("valid")]
    if not valid:
        return {"per_frame": [], "summary": {}}
    root = navsim_v2_root or DEFAULT_NAVSIM_V2_ROOT
    if not epdms_available(root, metric_cache_path):
        return {"per_frame": [], "summary": {}}
    pool = _get_pool(root, metric_cache_path, human_penalty_filter, num_workers)
    return pool.submit(_finalize_official, valid).result()


def epdms_score_official(
    trajs: np.ndarray,
    tokens: Sequence[str],
    metric_cache_path: str,
    *,
    navsim_v2_root: Optional[str] = None,
    num_workers: int = 64,
    human_penalty_filter: bool = True,
) -> Dict[str, Any]:
    """FULL official one-stage EPDMS in one call (offline; mirrors
    ``Transfusion/test_EPDMS.py``): :func:`epdms_batch_rows` +
    :func:`epdms_finalize_rows`. Returns
    ``{"per_frame": [...], "summary": {...}}`` with the complete official
    column set (NC/DAC/DDC/TLC/EP/TTC/LK/HC/EC/EPDMS)."""
    root = navsim_v2_root or DEFAULT_NAVSIM_V2_ROOT
    rows = epdms_batch_rows(
        trajs, tokens, metric_cache_path,
        navsim_v2_root=root, num_workers=num_workers, human_penalty_filter=human_penalty_filter,
    )
    return epdms_finalize_rows(
        (r for r in rows if r is not None),
        metric_cache_path,
        navsim_v2_root=root, num_workers=num_workers, human_penalty_filter=human_penalty_filter,
    )


__all__ = [
    "epdms_available",
    "epdms_score_per_frame",
    "epdms_score_official",
    "epdms_batch_rows",
    "epdms_finalize_rows",
    "per_frame_components",
    "DEFAULT_NAVSIM_V2_ROOT",
    "SUMMARY_COLS",
    "SHORT_NAMES",
]
