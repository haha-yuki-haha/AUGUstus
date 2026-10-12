#!/usr/bin/env python
"""Single-sample (batch_size=1) inference benchmark with per-module timing.

Loads a Transfusion checkpoint, feeds ONE sample through
``TransfusionQwen3p5.generate_trajectory`` and records how long each internal
module takes (vision encoder, LLM prefix forward, diffusion denoising, CoT
autoregressive decode, ...).

Timing is collected by wrapping the model's own methods with a lightweight
context-manager timer (``ModuleTimer``) -- no model code is modified, so the
numbers reflect the production inference path exactly.

Example
-------
python inference/benchmark_inference.py \
    --config <run_dir>/config.yaml \
    --checkpoint <run_dir>/best/pytorch_model \
    --test_data <hf_cache_test> \
    --runs 5 --warmup 2 \
    --output_dir inference/bench_out

The script prints a per-module table and writes ``timing_results.json`` and
``inference_timing.md`` (the latter documents every module).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_qwen3p5 import load_model  # noqa: E402
from model.dataset import TransfusionCollator, TransfusionHFCacheDataset  # noqa: E402
from model.utils import load_config, set_seed  # noqa: E402


# ----------------------------------------------------------------------------
# Lightweight per-module timer
# ----------------------------------------------------------------------------
class ModuleTimer:
    """Accumulate elapsed wall-clock time per named module.

    CUDA is synchronised before/after each region so GPU work is attributed
    correctly. Nested regions are permitted: a region records its own
    inclusive time (parent) while a nested child records its share too, so the
    two can be reported together (e.g. CoT prefill vs. CoT autoregressive).
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.stack: List[str] = []
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def _sync(self) -> None:
        if self.device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextmanager
    def record(self, name: str):
        self.stack.append(name)
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            self.times[name] = self.times.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            self.stack.pop()

    def reset(self) -> None:
        self.times.clear()
        self.counts.clear()
        self.stack.clear()


def _wrap_method(model, method_name: str, label: str, timer: ModuleTimer) -> None:
    """Shadow ``model.<method_name>`` with a timing wrapper.

    The original is already a *bound* method, so the wrapper (stored on the
    instance) is called without ``self`` and simply forwards the arguments.
    """
    orig = getattr(model, method_name)

    def wrapper(*args, **kwargs):
        with timer.record(label):
            return orig(*args, **kwargs)

    setattr(model, method_name, wrapper)


def instrument_model(model, timer: ModuleTimer) -> Dict[str, str]:
    """Attach timing wrappers to the modules we want to break out.

    Returns a mapping {label: human description} for the report.
    """
    wrapped = {
        "_get_multimodal_embeddings": "vision_encoder",
        "_denoise_loop_fm": "diffusion_denoise",
        "_generate_cot_after_action": "cot_generation",
        "_generate_cot_tokens": "cot_autoregressive",
        "_build_uncond_cache": "cfg_uncond_cache",
        "denormalize_actions": "postprocess_denorm",
    }
    for method_name, label in wrapped.items():
        if hasattr(model, method_name):
            _wrap_method(model, method_name, label, timer)
    return wrapped


# ----------------------------------------------------------------------------
# Data / model helpers
# ----------------------------------------------------------------------------
def resolve_config_path(config_arg: Optional[str], checkpoint: str) -> str:
    if config_arg:
        return config_arg
    ckpt = Path(checkpoint)
    for cand in (
        ckpt / "config.yaml",
        ckpt.parent / "config.yaml",
        ckpt.parent.parent / "config.yaml",
    ):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"No --config given and no config.yaml found next to checkpoint {checkpoint}"
    )


def build_dataset(model, cfg, test_data: str):
    raw = getattr(model, "module", model)
    tc, dc = cfg["trajectory"], cfg["data"]
    cot_cfg = cfg.get("cot", {})
    data_format = dc.get("data_format", "json")
    if data_format != "hf_cache":
        raise NotImplementedError(
            "This benchmark expects data.data_format='hf_cache' (as used by the "
            "NAVSIM eval caches)."
        )
    # Match test_qwen3p5.batch_evaluate: do not re-sample the cached video frames.
    if getattr(raw.processor, "video_processor", None) is not None:
        raw.processor.video_processor.do_sample_frames = False
    ds = TransfusionHFCacheDataset(
        cache_dir=test_data,
        processor=raw.processor,
        prompt_file=dc["prompt_file"],
        traj_token=tc["traj_token"],
        action_dim=tc["action_dim"],
        is_training=False,
        cot_components=cot_cfg.get("cot_components", None),
        video_fps=dc.get("video_fps", 2.0),
    )
    return ds


def prepare_batch(model, cfg, ds, collator):
    """Fetch sample 0, collate it (bs=1) and move tensors to the model device."""
    t0 = time.perf_counter()
    sample = ds[0]
    t1 = time.perf_counter()
    batch = collator([dict(sample)])
    t2 = time.perf_counter()
    device = next(getattr(model, "module", model).parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    batch_load = {
        "dataset_getitem_s": t1 - t0,
        "collate_s": t2 - t1,
        "host_to_device_s": t3 - t2,
    }
    return batch, batch_load


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def run_once(model, batch, cfg, timer: ModuleTimer, generate_cot: bool,
             max_cot_tokens: Optional[int] = None):
    raw = getattr(model, "module", model)
    cot_cfg = cfg.get("cot", {})

    timer.reset()
    with timer.record("generate_total"):
        result = raw.generate_trajectory(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
            return_normed=True,
            cfg_scale=cfg["diffusion"]["cfg_scale"],
            generate_cot=generate_cot,
            cot_temperature=cot_cfg.get("cot_temperature", 0.7),
            cot_top_p=cot_cfg.get("cot_top_p", 0.9),
            cot_top_k=cot_cfg.get("cot_top_k", 50),
            max_cot_tokens=max_cot_tokens,
        )
    return result, dict(timer.times)


def benchmark_mode(model, batch, cfg, timer: ModuleTimer, generate_cot: bool,
                   max_cot_tokens: Optional[int], warmup: int, runs: int,
                   label: str):
    """Warm up then time ``runs`` samples for one CoT mode.

    Returns (per_run, summary, final_result, peak_mem_gb).
    """
    device = timer.device
    for i in range(warmup):
        run_once(model, batch, cfg, timer, generate_cot, max_cot_tokens)
        print(f"[{label}][warmup] {i + 1}/{warmup} done")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    per_run: List[Dict[str, float]] = []
    final_result = None
    for i in range(runs):
        result, module_times = run_once(model, batch, cfg, timer, generate_cot,
                                        max_cot_tokens)
        per_run.append(module_times)
        final_result = result
        print(f"[{label}][run {i + 1}/{runs}] "
              f"total={module_times.get('generate_total', 0):.4f}s "
              f"vision={module_times.get('vision_encoder', 0):.4f}s "
              f"diff={module_times.get('diffusion_denoise', 0):.4f}s "
              f"cot={module_times.get('cot_generation', 0):.4f}s")
    summary = summarise(per_run)
    peak = (torch.cuda.max_memory_allocated(device) / 1e9
            if torch.cuda.is_available() else 0.0)
    return per_run, summary, final_result, peak


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
TOP_LEVEL_ORDER = [
    "vision_encoder",
    "context_prefix_llm",
    "diffusion_denoise",
    "cot_prefill",
    "cot_autoregressive",
    "cfg_uncond_cache",
    "postprocess_denorm",
]

MODULE_DOC = {
    "vision_encoder": (
        "`_get_multimodal_embeddings` — Qwen3.5 vision tower forward over the "
        "cached history frames + merge of visual tokens into the text embedding "
        "sequence (includes M-RoPE / image-token scatter)."
    ),
    "context_prefix_llm": (
        "Residual: the LLM context/prefix forward `[ctx] -> KV cache` in "
        "`_generate_trajectory_action_first` (plus any small bookkeeping not covered by "
        "the other timers). Isolated as `generate_total - Σ(other modules)`."
    ),
    "diffusion_denoise": (
        "`_denoise_loop_fm` — flow-matching reverse integration. Each step runs "
        "`_denoise_step_cached` (LLM suffix forward over the T action tokens + "
        "`action_decoder`) with the frozen prefix KV cache; ODE (Euler) by "
        "default, or the SDE kernels when `fm_sde` is set."
    ),
    "cot_prefill": (
        "`_generate_cot_after_action` MINUS `_generate_cot_tokens` — the "
        "one-shot prefill of `[ctx | clean_actions | <BOT>]` that builds the "
        "KV cache used by the CoT decode."
    ),
    "cot_autoregressive": (
        "`_generate_cot_tokens` — autoregressive post-hoc CoT decode, one "
        "LLM forward + `lm_head` + sampling per token (up to "
        "`cot.max_cot_tokens`, stops at `<|eot|>`)."
    ),
    "cfg_uncond_cache": (
        "`_build_uncond_cache` — unconditional (zeroed-context) KV cache for "
        "classifier-free guidance. Only runs when `diffusion.cfg_scale > 1.0`."
    ),
    "postprocess_denorm": (
        "`denormalize_actions` (+ delta/offset) — map the normalised "
        "trajectory back to the ego `(x, y, heading)` action space."
    ),
}


def _stat(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarise(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Collapse per-run module dicts into per-module statistics.

    ``context_prefix_llm`` and ``cot_prefill`` are derived residuals.
    """
    # Derived per run first.
    for rec in records:
        total = rec.get("generate_total", 0.0)
        rec["cot_prefill"] = max(
            0.0, rec.get("cot_generation", 0.0) - rec.get("cot_autoregressive", 0.0)
        )
        covered = (
            rec.get("vision_encoder", 0.0)
            + rec.get("diffusion_denoise", 0.0)
            + rec.get("cot_generation", 0.0)
            + rec.get("cfg_uncond_cache", 0.0)
            + rec.get("postprocess_denorm", 0.0)
        )
        rec["context_prefix_llm"] = max(0.0, total - covered)

    keys = set()
    for rec in records:
        keys.update(rec.keys())
    return {k: _stat([r[k] for r in records if k in r]) for k in sorted(keys)}


def print_table(summary: Dict[str, Dict[str, float]], total_s: Dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print(f"{'module':<24}{'mean(s)':>10}{'std':>9}{'min':>9}{'max':>9}{'% total':>10}")
    print("-" * 78)
    for key in TOP_LEVEL_ORDER:
        if key not in summary:
            continue
        s = summary[key]
        pct = (s["mean"] / total_s["mean"] * 100.0) if total_s.get("mean") else 0.0
        print(f"{key:<24}{s['mean']:>10.4f}{s['std']:>9.4f}"
              f"{s['min']:>9.4f}{s['max']:>9.4f}{pct:>9.1f}%")
    print("-" * 78)
    s = total_s
    print(f"{'TOTAL (generate)':<24}{s['mean']:>10.4f}{s['std']:>9.4f}"
          f"{s['min']:>9.4f}{s['max']:>9.4f}{100.0:>9.1f}%")
    print("=" * 78)


def _table(summary: Dict[str, Dict[str, float]], total_s: Dict[str, float]) -> List[str]:
    out = ["| module | mean (s) | std | min | max | % of total |",
           "|--------|---------:|----:|----:|----:|-----------:|"]
    for key in TOP_LEVEL_ORDER:
        if key not in summary:
            continue
        s = summary[key]
        pct = (s["mean"] / total_s["mean"] * 100.0) if total_s.get("mean") else 0.0
        out.append(f"| `{key}` | {s['mean']:.4f} | {s['std']:.4f} | "
                   f"{s['min']:.4f} | {s['max']:.4f} | {pct:.1f}% |")
    out.append(f"| **TOTAL** | **{total_s['mean']:.4f}** | **{total_s['std']:.4f}** | "
               f"**{total_s['min']:.4f}** | **{total_s['max']:.4f}** | **100%** |")
    return out


def write_markdown(path: Path, meta: Dict, modes: Dict[str, Dict],
                   batch_load: Dict[str, float]) -> None:
    """``modes`` maps a label (e.g. 'with_cot' / 'action_only') to
    {'per_run', 'summary', 'total', 'cot_tokens', 'trajectory_shape'}."""
    lines: List[str] = []
    lines.append("# Transfusion single-sample (bs=1) inference timing\n")
    lines.append(f"Generated by `inference/benchmark_inference.py` on "
                 f"{meta['timestamp']}.\n")

    lines.append("## Setup\n")
    lines.append(f"- Checkpoint: `{meta['checkpoint']}`")
    lines.append(f"- Config: `{meta['config']}`")
    lines.append(f"- Test data: `{meta['test_data']}`")
    lines.append(f"- Model type: `{meta['model_type']}`, use_action_first=`{meta['use_action_first']}`, "
                 f"diffusion=`{meta['diffusion_type']}`")
    lines.append(f"- num_inference_steps: `{meta['num_inference_steps']}`, "
                 f"cfg_scale: `{meta['cfg_scale']}`")
    lines.append(f"- device: `{meta['device']}`, dtype: bf16")
    lines.append(f"- model load: {meta['model_load_s']:.1f} s, "
                 f"peak CUDA mem: {meta['peak_mem_gb']:.2f} GB")
    lines.append(f"- warmup runs: {meta['warmup']}, timed runs: {meta['runs']}")
    lines.append(f"- torch {meta['torch']}, transformers {meta['transformers']}, "
                 f"CUDA {meta['cuda']}\n")

    lines.append("## Commands\n")
    lines.append("```bash")
    lines.append(meta["command"])
    lines.append("```\n")

    # ---- headline ----------------------------------------------------------
    lines.append("## Summary\n")
    for label, m in modes.items():
        lines.append(f"- **{label}**: {m['total']['mean']:.4f} s / sample "
                     f"(CoT tokens: {m.get('cot_tokens', 0)}, "
                     f"trajectory {m.get('trajectory_shape')})")
    lines.append("")

    lines.append("## Non-model (data) timing\n")
    lines.append("| stage | time (s) |")
    lines.append("|-------|---------:|")
    lines.append(f"| dataset `__getitem__` (video decode + processor) | "
                 f"{batch_load.get('dataset_getitem_s', 0.0):.4f} |")
    lines.append(f"| collate (bs=1) | {batch_load.get('collate_s', 0.0):.4f} |")
    lines.append(f"| host -> device | {batch_load.get('host_to_device_s', 0.0):.4f} |")
    lines.append("")
    lines.append("> These are one-off prep costs; they are **not** part of the "
                 "per-sample steady-state latency below.\n")

    for label, m in modes.items():
        summary, total_s = m["summary"], m["total"]
        lines.append(f"## Module breakdown — {label}\n")
        lines.append(f"Total measured `generate_trajectory`: **{total_s['mean']:.4f} s** "
                     f"(mean over {meta['runs']} runs, std {total_s['std']:.4f}).\n")
        lines.extend(_table(summary, total_s))
        lines.append("")
        if "cot_generation" in summary:
            s = summary["cot_generation"]
            lines.append(f"CoT generation total (`_generate_cot_after_action`, "
                         f"prefill + autoregressive): **{s['mean']:.4f} s** "
                         f"({s['mean'] / total_s['mean'] * 100.0:.1f}% of total).\n")
        lines.append("Per-run totals:\n")
        lines.append("| run | total (s) |")
        lines.append("|----:|----------:|")
        for i, rec in enumerate(m["per_run"]):
            lines.append(f"| {i} | {rec.get('generate_total', 0.0):.4f} |")
        lines.append("")

    lines.append("## What each module covers\n")
    for key in TOP_LEVEL_ORDER:
        if key not in MODULE_DOC:
            continue
        lines.append(f"### `{key}`")
        lines.append(MODULE_DOC[key] + "\n")

    lines.append("## Notes\n")
    lines.append("- Timing wraps the model's own methods (`ModuleTimer`), so the "
                 "measured path is exactly production inference.")
    lines.append("- `vision_encoder` runs once per sample; `diffusion_denoise` "
                 "runs `num_inference_steps` times; `cot_autoregressive` runs once "
                 "per generated CoT token.")
    lines.append("- Every region is CUDA-synchronised before/after, so numbers "
                 "include GPU stalls but no cross-region overlap.")
    lines.append("- `context_prefix_llm` is a residual and absorbs any code not "
                 "captured by a dedicated timer.\n")

    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="Path to the run config.yaml (default: auto from ckpt).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "bench_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=5, help="Timed runs (default 5).")
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs (default 2).")
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="Override diffusion.num_inference_steps.")
    p.add_argument("--max_cot_tokens", type=int, default=None,
                   help="Cap generated CoT content tokens.")
    p.add_argument("--generate_cot", choices=["auto", "1", "0"], default="auto",
                   help="auto: use config cot.enable_cot/eval_generate_cot.")
    p.add_argument("--skip_action_only", action="store_true",
                   help="When CoT is enabled, do NOT also benchmark the "
                        "action-only path (saves time).")
    p.add_argument("--action_only", action="store_true",
                   help="Benchmark ONLY the action-only path (no CoT).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = resolve_config_path(args.config, args.checkpoint)
    cfg = load_config(config_path)
    if args.num_inference_steps is not None:
        cfg["diffusion"]["num_inference_steps"] = args.num_inference_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  Transfusion bs=1 inference benchmark")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  config     : {config_path}")
    print(f"  test_data  : {args.test_data}")
    print(f"  device     : {device}")
    print("=" * 78)

    # ---- model load (timed) -------------------------------------------------
    t0 = time.perf_counter()
    model = load_model(cfg, args.checkpoint, str(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    load_s = time.perf_counter() - t0
    print(f"[load] model loaded in {load_s:.2f} s")

    # ---- data ---------------------------------------------------------------
    ds = build_dataset(model, cfg, args.test_data)
    collator = TransfusionCollator(
        getattr(model, "module", model).processor,
        cfg["trajectory"]["traj_token"], cfg["trajectory"]["action_dim"],
    )
    batch, batch_load = prepare_batch(model, cfg, ds, collator)
    print(f"[data] getitem={batch_load['dataset_getitem_s']*1000:.1f} ms, "
          f"collate={batch_load['collate_s']*1000:.1f} ms, "
          f"h2d={batch_load['host_to_device_s']*1000:.1f} ms")

    # ---- CoT switch ---------------------------------------------------------
    cot_cfg = cfg.get("cot", {})
    if args.action_only:
        generate_cot = False
        args.skip_action_only = True
    elif args.generate_cot == "auto":
        generate_cot = bool(cot_cfg.get("enable_cot", False)
                            and cot_cfg.get("eval_generate_cot", True))
    else:
        generate_cot = args.generate_cot == "1"
    print(f"[cot] generate_cot={generate_cot} (action_first_generate_cot="
          f"{cot_cfg.get('action_first_generate_cot', True)})")

    # ---- instrument ---------------------------------------------------------
    timer = ModuleTimer(device)
    instrument_model(getattr(model, "module", model), timer)

    def _out_info(result):
        traj = result[0] if isinstance(result, (tuple, list)) else result
        cot_ids = (result[1] if isinstance(result, (tuple, list)) and len(result) > 1
                   else None)
        arr = (traj.detach().float().cpu().numpy() if torch.is_tensor(traj)
               else np.asarray(traj))
        cot_len = int((cot_ids > 0).sum().item()) if torch.is_tensor(cot_ids) else 0
        return list(arr.shape), cot_len

    modes: Dict[str, Dict] = {}
    peak_mem_gb = 0.0

    # ---- primary mode (as configured) --------------------------------------
    per_run, summary, final_result, peak = benchmark_mode(
        model, batch, cfg, timer, generate_cot, args.max_cot_tokens,
        args.warmup, args.runs, "with_cot" if generate_cot else "action_only")
    peak_mem_gb = max(peak_mem_gb, peak)
    shape, cot_len = _out_info(final_result)
    modes["with_cot" if generate_cot else "action_only"] = {
        "per_run": per_run, "summary": summary,
        "total": summary["generate_total"],
        "trajectory_shape": shape, "cot_tokens": cot_len,
    }
    print_table(summary, summary["generate_total"])

    # ---- action-only counterpart (skip slow CoT) ---------------------------
    if generate_cot and not args.skip_action_only:
        print("\n[info] Also benchmarking the action-only path (generate_cot=False) ...")
        per_run_a, summary_a, result_a, peak_a = benchmark_mode(
            model, batch, cfg, timer, False, args.max_cot_tokens,
            args.warmup, args.runs, "action_only")
        peak_mem_gb = max(peak_mem_gb, peak_a)
        shape_a, cot_len_a = _out_info(result_a)
        modes["action_only"] = {
            "per_run": per_run_a, "summary": summary_a,
            "total": summary_a["generate_total"],
            "trajectory_shape": shape_a, "cot_tokens": cot_len_a,
        }
        print_table(summary_a, summary_a["generate_total"])

    print(f"[mem] peak cuda allocated: {peak_mem_gb:.2f} GB")

    # ---- persist ------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "config": config_path,
        "test_data": args.test_data,
        "model_type": cfg.get("model", {}).get("model_type"),
        "use_action_first": cot_cfg.get("use_action_first", False),
        "diffusion_type": "flow_matching",
        "num_inference_steps": cfg["diffusion"].get("num_inference_steps"),
        "cfg_scale": cfg["diffusion"].get("cfg_scale"),
        "max_cot_tokens": args.max_cot_tokens or cot_cfg.get("max_cot_tokens"),
        "device": str(device),
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "model_load_s": load_s,
        "peak_mem_gb": peak_mem_gb,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    results = {"meta": meta, "batch_load": batch_load, "modes": modes}
    (out / "timing_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    md_path = out / "inference_timing.md"
    write_markdown(md_path, meta, modes, batch_load)
    # Keep a copy at inference/inference_timing.md for quick reading.
    if out.resolve() != Path(__file__).resolve().parent:
        (Path(__file__).resolve().parent / "inference_timing.md").write_text(
            md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[done] wrote {out / 'timing_results.json'} and {md_path}")


if __name__ == "__main__":
    main()
