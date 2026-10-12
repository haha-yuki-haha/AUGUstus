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

"""SFT-style worker-local validation for the Transfusion RL recipe.

verl's stock validation datapath decodes/encodes every multimodal sample on the
driver and then ships the resulting tensors through ``DataProto``/Ray. That is
correct for APIs like vLLM, but it is several times slower than supervised
Transfusion evaluation, where each GPU rank owns a local DataLoader and only the
small metric results leave the rank.

This module implements that SFT layout inside the existing actor-rollout
workers: every worker materialises only its own contiguous validation shard,
runs the already-synced rollout policy, computes rewards/PDMS locally, and
returns small per-metric sums to the driver. No verl core file is edited.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _identity_collate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep recipe-dataset rows unstacked; ``TransfusionPolicy.collate`` stacks them."""
    return rows


class _Subset(Dataset):
    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.base[self.indices[idx]]


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    getter = getattr(obj, "get", None)
    if callable(getter):
        val = getter(key, default)
        return default if val is None else val
    return getattr(obj, key, default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_path(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value) if value else None


def _cot_sampler_from_transfusion_config() -> Dict[str, float]:
    """Read the CoT sampling defaults used by SFT eval from ``$TRANSFUSION_CONFIG``."""
    path = os.environ.get("TRANSFUSION_CONFIG")
    if not path or not os.path.isfile(path):
        return {}
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        cot = cfg.get("cot", {}) or {}
        return {
            "cot_temperature": float(cot.get("cot_temperature", 0.7)),
            "cot_top_p": float(cot.get("cot_top_p", 0)),
            "cot_top_k": int(cot.get("cot_top_k", 0)),
        }
    except Exception:
        return {}


def _contiguous_shard_indices(num_samples: int, num_ranks: int, rank: int) -> List[int]:
    """Rank-contiguous partition so rank 0 owns the leading logged samples."""
    base, rem = divmod(num_samples, num_ranks)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


def _add_numeric(dst_sums: Dict[str, float], dst_counts: Dict[str, int],
                 key: str, value: Any) -> None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(val):
        return
    dst_sums[key] = dst_sums.get(key, 0) + val
    dst_counts[key] = dst_counts.get(key, 0) + 1


def _log_first_samples(
    *,
    rank: int,
    log_num_samples: int,
    local_seen: int,
    row: Dict[str, Any],
    pred_traj: np.ndarray,
    pred_cot: str,
    metrics: Dict[str, float],
    processor: Any,
    global_step: int,
) -> None:
    """Mirror the existing Transfusion sample logger on rank 0 only."""
    if rank != 0 or local_seen >= log_num_samples:
        return
    try:
        from recipe.transfusion.sample_logging import log_sample_best_effort

        gt = row.get("reward_model", {}).get("ground_truth", {}) or {}
        gt_traj = np.asarray(gt.get("gt_trajectory"), dtype=np.float32)
        prompt_text = None
        ti = row.get("transfusion_inputs") or {}
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and ti.get("input_ids") is not None:
            ids = ti["input_ids"]
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            try:
                prompt_text = tokenizer.decode(ids, skip_special_tokens=False)
            except Exception:
                prompt_text = None
        log_sample_best_effort(
            {
                "text": prompt_text,
                "gt_traj": gt_traj,
                "pred_traj": pred_traj,
                "cot": pred_cot,
                "gt_cot": gt.get("gt_cot_text"),
                "metrics": metrics,
                "raw_frames": row.get("_raw_video_frames"),
            },
            is_validate=True,
            step=global_step,
        )
    except Exception:  # pragma: no cover - logging must never break validation
        pass


def validate_shard(worker: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the worker-local shard and return small metric/count payloads.

    The ``payload`` is intentionally tiny (paths + a few scalars). All video
    decode/processor work happens in this worker's DataLoader, matching the SFT
    eval layout.
    """
    from recipe.transfusion.dataset import TransfusionRLDataset
    from recipe.transfusion.transfusion_reward import RewardWeights, pdms_trajectory_reward
    from recipe.transfusion.reward_function import score_sample

    rank = _as_int(getattr(worker, "rank", 0), 0)
    world_size = _as_int(getattr(worker, "world_size", 1), 1)
    policy = worker.rollout.policy
    processor = policy.processor

    # Match SFT's cached-video handling: the mp4 bytes are already decoded at
    # the target fps; do not let the Qwen video processor silently resample.
    try:
        if getattr(processor, "video_processor", None) is not None:
            processor.video_processor.do_sample_frames = False
        import transformers.models.qwen3_vl.processing_qwen3_vl as _qwen3_proc_mod

        _qwen3_proc_mod.logger.warning_once = lambda *a, **kw: None
    except Exception:
        pass

    data_cfg = {
        "train_files": payload.get("train_files"),
        "metric_cache_train": payload.get("metric_cache_train"),
        "metric_cache_test": payload.get("metric_cache_test"),
        "metric_cache_train_v2": payload.get("metric_cache_train_v2"),
        "metric_cache_test_v2": payload.get("metric_cache_test_v2"),
        "seed": payload.get("seed", 42),
        "transfusion": payload.get("transfusion", {}),
    }
    dataset = TransfusionRLDataset(
        payload["val_files"],
        tokenizer=getattr(processor, "tokenizer", None),
        config=data_cfg,
        processor=processor,
        max_samples=-1,
    )

    max_samples = _as_int(payload.get("max_samples", -1), -1)
    n_total = len(dataset)
    if 0 < max_samples < n_total:
        n_total = max_samples
    indices = _contiguous_shard_indices(n_total, world_size, rank)
    local_n = len(indices)

    batch_size = max(1, _as_int(payload.get("batch_size_per_gpu", 32), 32))
    num_workers = max(0, _as_int(payload.get("num_workers_per_gpu", 8), 8))
    loader = DataLoader(
        _Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        pin_memory=False,
    )

    cot_sampler = {
        "cot_temperature": float(payload.get("cot_temperature", 0.7)),
        "cot_top_p": float(payload.get("cot_top_p", 0)),
        "cot_top_k": int(payload.get("cot_top_k", 0)),
    }
    defaults = _cot_sampler_from_transfusion_config()
    cot_sampler = {k: payload.get(k, defaults.get(k, cot_sampler[k])) for k in cot_sampler}

    num_steps = payload.get("num_steps")
    num_steps = None if num_steps is None else int(num_steps)
    # Optional CoT generation cap for validation, matching the RL training rollout
    # budget (``response_length - 1``). When absent the model falls back to its
    # native ``max_cot_tokens`` (e.g. 512), which is why the old validation CoT
    # could blow up to ~511 tokens of post-``</prediction>`` filler.
    cot_max_tokens = payload.get("cot_max_tokens")
    if cot_max_tokens is not None:
        cot_max_tokens = max(1, int(cot_max_tokens))
    else:
        cot_max_tokens = None
    global_step = _as_int(payload.get("global_step", 0), 0)
    log_num_samples = max(0, _as_int(payload.get("log_num_samples", 10), 10))
    reward_kwargs = dict(payload.get("reward_kwargs", {}) or {})
    from recipe.transfusion.reward_function import resolve_metric_cache_v2_path, strip_weights_for_split
    from recipe.transfusion.transfusion_reward import epdms_enabled

    default_weights = strip_weights_for_split(reward_kwargs.get("weights", {}) or {}, is_validate=True)
    pdms_workers = max(1, _as_int(payload.get("pdms_workers_per_gpu", 8), 8))

    metric_sums: Dict[str, float] = defaultdict(float)
    metric_counts: Dict[str, int] = defaultdict(int)
    seconds: Dict[str, float] = defaultdict(float)
    # per-sample dump (one JSONL line per evaluated sample). Opens lazily under
    # TRANSFUSION_OUTPUT_DIR/samples/per_step_<global_step>/val_rank_<r>.jsonl.
    _dump_fh = None
    try:
        import json as _json

        out_dir = os.environ.get("TRANSFUSION_OUTPUT_DIR") or "/tmp"
        out_dir = os.path.join(out_dir, "samples", f"step_{int(global_step):07d}")
        os.makedirs(out_dir, exist_ok=True)
        _dump_fh = open(os.path.join(out_dir, f"val_rank{rank}.jsonl"), "a", encoding="utf-8")
    except Exception:
        _dump_fh = None
    # PDMS diagnostics per shard: "candidate" = samples eligible for PDMS scoring
    # (valid trajectory + frame_id), "scored" = samples that actually produced a
    # pdms_reward value (the denominator used for the pdms_reward mean).
    pdms_counts: Dict[str, int] = {"candidate": 0, "scored": 0}
    epdms_rows: List[Dict[str, Any]] = []
    epdms_cache_v2: Optional[Dict[str, Any]] = None
    t_total = time.time()
    local_seen = 0

    policy.model.eval()

    # [probe] deterministic param-load fingerprint for "which ckpt is THIS?":
    # seed the RNG, sample ONE trajectory for the shard's first sample with the
    # 1-step ODE sampler, hash it. Deterministically reproducible OFFLINE with
    # the same input + a monolithic copy of candidate weights.
    try:
        if rank == 0:
            import hashlib

            first_row = dataset[indices[0]]
            torch.manual_seed(0)
            _probe = policy.rollout(
                [first_row["transfusion_inputs"]],
                generate_cot=False,
                return_normed=False,
                num_steps=num_steps,
                device=next(policy.model.parameters()).device,
            )
            _pa = np.asarray(_probe["trajectories"][0], dtype=np.float32)
            _h = hashlib.sha1(np.round(_pa, 3).tobytes()).hexdigest()[:12]
            logger.warning("[probe] fingerprint=%s pred=%s", _h, np.round(_pa, 3).tolist())
            torch.manual_seed(time.time_ns() % (2**31))  # restore unseeded behavior
    except Exception as _e:  # pragma: no cover - probe must never break val
        logger.warning("[probe] failed: %s", _e)

    for batch_i, rows in enumerate(loader):
        t_gen = time.time()
        inputs = [row["transfusion_inputs"] for row in rows]
        out = policy.rollout(
            inputs,
            generate_cot=bool(payload.get("generate_cot", True)),
            return_normed=False,
            num_steps=num_steps,
            device=next(policy.model.parameters()).device,
            cot_max_tokens=cot_max_tokens,
            **cot_sampler,
        )
        seconds["generate"] += time.time() - t_gen
        pred_trajs = np.asarray(out["trajectories"], dtype=np.float32)

        t_reward = time.time()
        gts = [row.get("reward_model", {}).get("ground_truth", {}) or {} for row in rows]
        frame_ids = [gt.get("frame_id") or "" for gt in gts]
        metric_cache_path = gts[0].get("metric_cache_path") if gts else None
        precomputed_pdms: List[Optional[Dict[str, float]]] = [None] * len(rows)
        precomputed_epdms: List[Optional[Dict[str, float]]] = [None] * len(rows)
        valid_idx, valid_trajs, valid_fids = [], [], []
        for i, (pt, fid) in enumerate(zip(pred_trajs, frame_ids)):
            if pt is not None and fid:
                valid_idx.append(i)
                valid_trajs.append(np.asarray(pt, dtype=np.float32))
                valid_fids.append(fid)
        # pdms_candidate: number of samples actually sent to the metric-cache
        # scorer this batch (valid trajectory + frame_id).
        pdms_candidate = len(valid_trajs)
        if valid_trajs and metric_cache_path:
            precomputed = pdms_trajectory_reward(
                np.stack(valid_trajs),
                valid_fids,
                metric_cache_path,
                num_workers=min(pdms_workers, max(1, len(valid_trajs))),
            )
            for i, components in zip(valid_idx, precomputed):
                precomputed_pdms[i] = components

        # Official two-frame EPDMS scoring (val split): enabled per-sample by the
        # reward weights (epdms_enabled) and requires the v2 metric cache path
        # (per-sample override -> reward_kwargs -> data.metric_cache_*_v2).
        if valid_trajs:
            _w0 = RewardWeights.from_dict(
                strip_weights_for_split({**default_weights, **(gts[0].get("weights") or {})}, is_validate=True)
            )
            if epdms_enabled(_w0, is_validate=True):
                metric_cache_v2 = resolve_metric_cache_v2_path(
                    gts[0].get("metric_cache_v2_path") if gts else None, reward_kwargs
                )
                if metric_cache_v2:
                    from recipe.transfusion import epdms_eval

                    epdms_cache_v2 = metric_cache_v2
                    _t_ep = time.time()
                    ep_rows = epdms_eval.epdms_batch_rows(
                        np.stack(valid_trajs),
                        valid_fids,
                        metric_cache_v2,
                        num_workers=min(pdms_workers, max(1, len(valid_trajs))),
                    )
                    logger.warning(
                        "[time] sft_val rank=%d epdms batch=%d: %.1fs (%d samples, cache=%s)",
                        rank, batch_i + 1, time.time() - _t_ep, len(valid_trajs), metric_cache_v2,
                    )
                    for i, row in zip(valid_idx, ep_rows):
                        if row is None:
                            continue
                        precomputed_epdms[i] = epdms_eval.per_frame_components(row)
                        epdms_rows.append(row)
                else:
                    logger.warning(
                        "sft_val rank=%d: val_epdms enabled but no v2 metric cache (data.metric_cache_*_v2 / reward_kwargs); skipping EPDMS.",
                        rank,
                    )

        cot_ids = out.get("cot_ids") or []
        pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0) or 0
        pdms_scored = 0
        for i, row in enumerate(rows):
            gt = gts[i]
            weights = RewardWeights.from_dict(strip_weights_for_split({**default_weights, **(gt.get("weights") or {})}, is_validate=True))
            pred_cot = out.get("cot_texts", [""] * len(rows))[i]
            result = score_sample(
                pred_traj=pred_trajs[i],
                gt_traj=np.asarray(gt.get("gt_trajectory"), dtype=np.float32),
                pred_cot_text=pred_cot,
                gt_cot_text=gt.get("gt_cot_text"),
                frame_id=frame_ids[i],
                metric_cache_path=metric_cache_path,
                weights=weights,
                pdms_components=precomputed_pdms[i],
                epdms_components=precomputed_epdms[i],
            )
            # A sample "participates" in the pdms_reward mean iff it actually
            # produced a pdms_reward value (that is the denominator basis used by
            # aggregate_shard_results for val-core/.../pdms_reward/mean).
            if "pdms_reward" in result:
                pdms_scored += 1
            # per-sample dump: one JSONL line per evaluated sample with full
            # numeric metrics + trajectory + frame_id so config-sensitivity
            # can be debugged per-sample (the aggregate means hide which
            # subset of samples shifted).
            try:
                _dump_fh.write(_json.dumps({
                    "frame_id": frame_ids[i] or "",
                    "dataset_index": int(batch_i),  # loader batch index
                    "local_seq": local_seen,
                    "rank": rank,
                    "global_step": int(global_step),
                    **{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)[:100]) for k, v in result.items()},
                    "traj": np.round(np.asarray(pred_trajs[i], dtype=np.float32), 3).tolist(),
                    "gt_traj": np.round(np.asarray(gt.get("gt_trajectory"), dtype=np.float32), 3).tolist(),
                    "cot_text": str(pred_cot),
                }) + "\n")
            except Exception:
                pass
            for key, value in result.items():
                _add_numeric(metric_sums, metric_counts, key, value)

            # Keep the same prompt/response length diagnostics as the old
            # DataProto path, but compute them locally.
            ti = row.get("transfusion_inputs") or {}
            input_ids = ti.get("input_ids")
            if input_ids is not None:
                _add_numeric(metric_sums, metric_counts, "prompt_len", input_ids.shape[-1])
            if i < len(cot_ids) and cot_ids[i]:
                ids = np.asarray(cot_ids[i])
                cot_len = int(np.sum(ids != pad_id)) if ids.size else 0
            else:
                cot_len = 0
            _add_numeric(metric_sums, metric_counts, "cot_len", cot_len)
            _add_numeric(metric_sums, metric_counts, "reward", result.get("score", 0))

            _log_first_samples(
                rank=rank,
                log_num_samples=log_num_samples,
                local_seen=local_seen,
                row=row,
                pred_traj=pred_trajs[i],
                pred_cot=pred_cot,
                metrics=result,
                processor=processor,
                global_step=global_step,
            )
            local_seen += 1
        seconds["reward"] += time.time() - t_reward
        pdms_counts["candidate"] += pdms_candidate
        pdms_counts["scored"] += pdms_scored

        if (batch_i + 1) % 10 == 0 or batch_i == len(loader) - 1:
            logger.warning(
                "[time] sft_val rank=%d batch=%d/%d count=%d gen=%.1fs reward=%.1fs elapsed=%.1fs",
                rank, batch_i + 1, len(loader), local_seen,
                seconds["generate"], seconds["reward"], time.time() - t_total,
            )

    total_seconds = time.time() - t_total
    logger.warning(
        "[time] sft_val rank=%d done: %d samples, %.1fs total (gen=%.1fs reward=%.1fs, B=%d, workers=%d)",
        rank, local_seen, total_seconds, seconds["generate"], seconds["reward"], batch_size,
        num_workers,
    )

    return {
        "rank": rank,
        "count": local_n,
        "num_batches": len(loader),
        "seconds": dict(seconds),
        "total_seconds": total_seconds,
        "metric_sums": dict(metric_sums),
        "metric_counts": dict(metric_counts),
        "pdms_counts": dict(pdms_counts),
        "epdms_rows": epdms_rows,
        "epdms_metric_cache_v2": epdms_cache_v2,
    }


# -----------------------------------------------------------------------------
# Driver-side aggregation notes (validate_shard -> aggregate_shard_results):
#
# Each worker's validate_shard returns a SMALL payload dict (see the return
# statement above): metric sums/counts, PDMS eligibility counts, and -- when
# the official two-frame EPDMS path is enabled -- the raw per-frame `epdms_rows`
# plus the `epdms_metric_cache_v2` cache handle they were scored against. The
# rows are NOT aggregated worker-locally because the official EPDMS metric
# chains CONSECUTIVE frames of the same log across the whole validation set;
# a per-worker partial chain would truncate the horizon at every shard
# boundary, so the rows are shipped to the driver and only finalized there
# (via recipe.transfusion.epdms_eval).
#
# Failure policy: EPDMS finalization is strictly best-effort. If the chained
# finalization raises for ANY reason (missing cache entries, corrupt rows,
# import problems), the driver logs a warning and keeps the one-stage EPDMS
# sums exactly as the workers produced them -- validation results are never
# lost, and the two-frame numbers can be recomputed offline if needed.
#
# The one-stage / two-frame pairing: every per-frame record carries BOTH the
# one-stage EPDMS value (what the metric sums above already contain) and the
# final chained value, so the driver-side delta (final - one-stage) can be
# folded back into the `epdms` / `epdms_reward` sums, keeping a SINGLE mean
# per metric instead of two near-duplicate curves.


def aggregate_shard_results(
    results: Iterable[Dict[str, Any]],
    *,
    data_source: str,
    core_metrics: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Reduce worker-local metric sums to exact full-set means on the driver."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    total_count = 0
    seconds: Dict[str, float] = defaultdict(float)
    pdms_candidate = 0
    pdms_scored = 0
    epdms_rows = []
    epdms_cache_v2 = None
    for result in results:
        total_count += int(result.get("count", 0))
        for key, value in (result.get("seconds") or {}).items():
            seconds[key] += float(value)
        for key, value in (result.get("metric_sums") or {}).items():
            sums[key] += float(value)
        for key, value in (result.get("metric_counts") or {}).items():
            counts[key] += int(value)
        pc = result.get("pdms_counts") or {}
        pdms_candidate += int(pc.get("candidate", 0))
        pdms_scored += int(pc.get("scored", 0))
        rows = result.get("epdms_rows") or []
        if rows:
            epdms_rows.extend(rows)
        cache_v2 = result.get("epdms_metric_cache_v2")
        if not cache_v2:
            continue
        if epdms_cache_v2:
            continue
        epdms_cache_v2 = cache_v2

    # Official two-frame EPDMS: chain the per-frame rows across the shards and
    # fold the final-vs-one-stage delta back into the EPDMS metric sums.
    if epdms_rows and epdms_cache_v2:
        try:
            from recipe.transfusion import epdms_eval

            final = epdms_eval.epdms_finalize_rows(epdms_rows, epdms_cache_v2)
            per_frame = final.get("per_frame") or []
            delta = 0.0
            ec_sum = 0.0
            ec_n = 0
            for rec in per_frame:
                one, fin = rec.get("EPDMS_one_stage"), rec.get("EPDMS")
                if one is not None and fin is not None and np.isfinite(one) and np.isfinite(fin):
                    delta += float(fin) - float(one)
                ec = rec.get("EC")
                if ec is None:
                    continue
                if not np.isfinite(ec):
                    continue
                ec_sum += float(ec)
                ec_n += 1
            if "epdms" in sums:
                sums["epdms"] += delta
            if "epdms_reward" in sums:
                sums["epdms_reward"] += delta
            if ec_n:
                sums["epdms_EC"] = ec_sum
                counts["epdms_EC"] = ec_n
                logger.warning(
                    "[val] official two-frame EC: %d/%d frames chained, mean EC=%.4f",
                    ec_n, len(per_frame), ec_sum / ec_n,
                )
        except Exception as e:
            logger.warning("official two-frame EC aggregation failed (keeping one-stage EPDMS): %s", e)

    metrics: Dict[str, float] = {"val/num_samples": float(total_count)}

    # PDMS count diagnostics (absolute numbers, one scalar per metric so the count
    # is directly visible in SwanLab):
    #   pdms_candidate/num -> samples eligible (valid trajectory + frame_id)
    #   pdms_scored/num    -> samples that produced a real pdms_reward (== the
    #                         denominator of the pdms_reward mean below = "valid")
    #   pdms_invalid/num   -> the REST of the full set that did NOT get a PDMS
    #                         value (= total - scored), incl. missing frame_id.
    metrics[f"val-aux/{data_source}/pdms_scored/num"] = float(pdms_scored)
    metrics[f"val-aux/{data_source}/pdms_invalid/num"] = float(
        max(0, total_count - pdms_scored)
    )
    if pdms_candidate > 0:
        metrics[f"val-aux/{data_source}/pdms_candidate/num"] = float(pdms_candidate)
    if total_count > 0:
        metrics[f"val-aux/{data_source}/pdms_scored_ratio/mean"] = pdms_scored / total_count

    for key, value in sums.items():
        n = counts.get(key, 0)
        if n <= 0:
            continue
        metrics[f"val-aux/{data_source}/{key}/mean"] = value / n

    core = set(core_metrics or ("score", "reward", "pdms_reward"))
    aux_prefix = f"val-aux/{data_source}"
    for key in core:
        aux_key = f"{aux_prefix}/{key}/mean"
        if aux_key in metrics:
            metrics[f"val-core/{data_source}/{key}/mean"] = metrics[aux_key]

    # Wall-clock/timings are diagnostics, not optimization metrics.
    for key, value in seconds.items():
        metrics[f"val-aux/{data_source}/time_{key}_sum"] = value
    return metrics


__all__ = ["validate_shard", "aggregate_shard_results"]
