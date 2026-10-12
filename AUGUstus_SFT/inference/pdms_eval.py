#!/usr/bin/env python
"""Full bs=1 action inference + PDMS evaluation with the fast (CUDA-graph) engine.

Runs the optimized no-CoT single-step action decoder over the whole test cache
(one sample at a time), then computes NAVSIM PDMS for every sample and dumps a
single JSON with:

  * number of inferences,
  * average / total *processing* time (CUDA-graph capture excluded),
  * average PDMS + sub-metrics (NC/DAC/DDC/EP/TTC/C) and ADE/FDE/L2,
  * per-sample values (frame_id, inference time, trajectory, per-sample metrics).

Timing only covers the model forward pass; PDMS scoring and data loading are
measured separately and excluded from the per-sample average.

Usage
-----
python inference/pdms_eval.py \
    --config <run_dir>/config.yaml \
    --checkpoint <run_dir>/best/pytorch_model \
    --test_data <hf_cache_test> \
    --output_dir inference/pdms_eval_out \
    [--max_samples N] [--num_workers 8] [--no_graph]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_qwen3p5 import load_model  # noqa: E402
from model.dataset import TransfusionCollator, TransfusionHFCacheDataset  # noqa: E402
from model.utils import (  # noqa: E402
    compute_action_metrics,
    compute_pdm_scores_batch,
    load_config,
    set_seed,
)
from fast_inference import FastActionInference, _sync, resolve_config_path  # noqa: E402


PDM_KEYS = ["PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C"]


def build_dataloader(model, cfg, test_data, num_workers: int, max_samples):
    raw = getattr(model, "module", model)
    tc, dc, cot_cfg = cfg["trajectory"], cfg["data"], cfg.get("cot", {})
    if getattr(raw.processor, "video_processor", None) is not None:
        raw.processor.video_processor.do_sample_frames = False
    ds = TransfusionHFCacheDataset(
        cache_dir=test_data, processor=raw.processor, prompt_file=dc["prompt_file"],
        traj_token=tc["traj_token"], action_dim=tc["action_dim"],
        is_training=False,
        cot_components=cot_cfg.get("cot_components", None),
        video_fps=dc.get("video_fps", 2.0),
    )
    if max_samples and max_samples < len(ds):
        ds.hf_dataset = ds.hf_dataset.select(range(max_samples))
    collator = TransfusionCollator(raw.processor, tc["traj_token"], tc["action_dim"])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                    collate_fn=collator, pin_memory=True,
                    prefetch_factor=(4 if num_workers > 0 else None))
    return ds, dl


def move_to_device(batch, device):
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "pdms_eval_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of samples (default: full test cache).")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--pdm_workers", type=int, default=16, help="PDMS process pool workers.")
    p.add_argument("--warmup", type=int, default=3, help="Untimed warmup iterations.")
    p.add_argument("--graph_warmup", type=int, default=5)
    p.add_argument("--no_graph", action="store_true", help="Disable CUDA graphs.")
    p.add_argument("--no_pdms", action="store_true", help="Skip PDMS (action metrics only).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    config_path = resolve_config_path(args.config, args.checkpoint)
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("  bs=1 action inference + PDMS evaluation (fast CUDA-graph engine)")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  config     : {config_path}")
    print(f"  test_data  : {args.test_data}")
    print(f"  device     : {device} | graph={not args.no_graph} | workers={args.num_workers}")
    print("=" * 78)

    t_load = time.perf_counter()
    model = load_model(cfg, args.checkpoint, str(device))
    _sync(device)
    load_s = time.perf_counter() - t_load
    print(f"[load] model loaded in {load_s:.1f} s")

    engine = FastActionInference(model, cfg, device, graph_warmup=args.graph_warmup,
                                 deterministic_noise=True)

    ds, dl = build_dataloader(model, cfg, args.test_data, args.num_workers, args.max_samples)
    n = len(ds)
    print(f"[data] {n} samples, {len(dl)} batches")

    # ---- untimed warmup -----------------------------------------------------
    it = iter(dl)
    for _ in range(args.warmup):
        try:
            batch = move_to_device(next(it), device)
        except StopIteration:
            it = iter(dl)
            batch = move_to_device(next(it), device)
        engine.infer(batch, use_graph=not args.no_graph)
    engine.capture_time_s = 0.0
    engine.num_captures = 0

    # ---- inference loop (timed) --------------------------------------------
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    frame_ids: List[str] = []
    per_sample_ms: List[float] = []
    infer_total_s = 0.0
    n_done = 0
    loop_t0 = time.perf_counter()
    for batch in dl:
        batch = move_to_device(batch, device)
        cap_before = engine.capture_time_s
        t0 = time.perf_counter()
        pred = engine.infer(batch, use_graph=not args.no_graph)
        _sync(device)
        dt = time.perf_counter() - t0
        # exclude one-time CUDA-graph capture from "processing" time
        dt -= (engine.capture_time_s - cap_before)
        dt = max(dt, 0.0)
        infer_total_s += dt
        per_sample_ms.append(dt * 1e3)
        preds.append(pred)
        gts.append(batch["actions"][0].detach().float().cpu().numpy())
        fid = batch.get("_frame_id", [""])
        frame_ids.append(fid[0] if len(fid) else "")
        n_done += 1
        if n_done % 200 == 0 or n_done == n:
            avg = infer_total_s / n_done * 1e3
            print(f"  [{n_done}/{n}] avg {avg:.2f} ms/sample "
                  f"(captures={engine.num_captures})")
    loop_s = time.perf_counter() - loop_t0
    print(f"[infer] {n_done} samples | total processing {infer_total_s:.2f} s | "
          f"avg {infer_total_s / max(n_done, 1) * 1e3:.2f} ms | "
          f"captures {engine.num_captures} ({engine.capture_time_s:.1f} s, excluded)")

    preds_np = np.stack(preds)  # [N, T, A]
    gts_np = np.stack(gts)

    # ---- action metrics -----------------------------------------------------
    action_metrics = compute_action_metrics(
        torch.from_numpy(preds_np), torch.from_numpy(gts_np),
        action_dim=cfg["trajectory"]["action_dim"])
    print(f"[action] ADE {action_metrics['ADE']:.4f} | FDE {action_metrics['FDE']:.4f} "
          f"| L2 {action_metrics['L2']:.4f}")

    # ---- PDMS ---------------------------------------------------------------
    pdms_avg = {}
    pdms_per_sample = [None] * n_done
    pdms_s = 0.0
    if not args.no_pdms:
        metric_cache = cfg.get("data", {}).get("metric_cache_test")
        t_pdm = time.perf_counter()
        pdms = compute_pdm_scores_batch(
            preds_np, frame_ids, num_workers=args.pdm_workers,
            metric_cache_path=metric_cache, return_per_sample=True,
            return_count=True, show_progress=True)
        pdms_s = time.perf_counter() - t_pdm
        pdms_per_sample = pdms.pop("per_sample", [None] * n_done)
        count = pdms.pop("count", n_done)
        pdms_avg = pdms
        print(f"[pdms] scored {count}/{n_done} in {pdms_s:.1f} s | " +
              " | ".join(f"{k} {v:.4f}" for k, v in pdms_avg.items()))

    # ---- per-sample records -------------------------------------------------
    per_sample = []
    for i in range(n_done):
        ps = pdms_per_sample[i] if i < len(pdms_per_sample) else None
        disp = np.linalg.norm(preds_np[i][:, :2] - gts_np[i][:, :2], axis=-1)
        rec = {
            "index": i,
            "frame_id": frame_ids[i],
            "inference_ms": round(per_sample_ms[i], 3),
            "ADE": round(float(disp.mean()), 4),
            "FDE": round(float(disp[-1]), 4),
            "pred": np.round(preds_np[i], 2).tolist(),
        }
        if ps is not None:
            rec.update({k: (round(float(ps[k]), 4) if k in ps else None) for k in PDM_KEYS})
        per_sample.append(rec)

    # ---- timing summary -----------------------------------------------------
    ms = np.array(per_sample_ms)
    timing = {
        "num_inferences": n_done,
        "inference_total_s": round(infer_total_s, 4),
        "avg_ms_per_sample": round(float(ms.mean()), 4) if n_done else None,
        "median_ms_per_sample": round(float(np.median(ms)), 4) if n_done else None,
        "min_ms": round(float(ms.min()), 4) if n_done else None,
        "max_ms": round(float(ms.max()), 4) if n_done else None,
        "std_ms": round(float(ms.std()), 4) if n_done else None,
        "graph_captures": engine.num_captures,
        "graph_capture_time_s": round(engine.capture_time_s, 4),
        "inference_loop_wall_s": round(loop_s, 4),
        "pdms_scoring_s": round(pdms_s, 4),
        "model_load_s": round(load_s, 4),
    }

    metrics_avg = {k: round(float(v), 4) for k, v in action_metrics.items()}
    metrics_avg.update({k: round(float(v), 4) for k, v in pdms_avg.items()})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "checkpoint": args.checkpoint,
            "config": config_path,
            "test_data": args.test_data,
            "num_samples": n_done,
            "use_graph": not args.no_graph,
            "max_samples": args.max_samples,
            "device": str(device),
            "seed": args.seed,
        },
        "timing": timing,
        "metrics_avg": metrics_avg,
        "per_sample": per_sample,
    }
    result_path = out / "pdms_eval.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"  num_inferences   : {n_done}")
    print(f"  avg processing   : {timing['avg_ms_per_sample']} ms/sample")
    print(f"  total processing : {timing['inference_total_s']} s")
    print(f"  metrics (avg)    : " +
          " | ".join(f"{k} {v:.4f}" for k, v in metrics_avg.items()))
    print(f"  saved            : {result_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
