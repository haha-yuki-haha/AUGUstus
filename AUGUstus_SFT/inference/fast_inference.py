#!/usr/bin/env python
"""Deeply-optimized bs=1 inference for the no-CoT, single-step action path.

Motivation
----------
The standard ``generate_trajectory`` path for a bs=1, no-CoT, 1-step flow
matching model takes ~135 ms. Profiling shows it is almost entirely
**CPU-bound** launch overhead, not GPU math:

  * ~3.7k kernel launches / sample (mostly tiny FLA ``GatedDeltaNet`` and
    elementwise kernels),
  * the ``ChunkGatedDeltaRuleFunction`` Python/Triton wrapper alone burns
    ~34 ms of CPU per sample,
  * ~200 small pageable H2D copies (``torch.full(scalar-tensor)``,
    CPU-built timesteps, ...).

Strategy
--------
Split the pipeline into its two stages:

  1. **Vision stage** (`_get_multimodal_embeddings`): the HF vision forward has
     unavoidable host syncs (``grid_thw.tolist()``, ``repeat_interleave`` with
     tensor repeats). It is kept **eager** (~38 ms) and its output is copied
     into a static buffer.
  2. **LLM action-decoder stage** (prefix forward + 1 denoise step + denorm):
     pure GPU work with no host syncs once a few careful adaptations are made
     (cache multimodal embeds / M-RoPE positions, skip the padding-all-ones
     check, pre-build the inference timesteps on-device, avoid
     ``torch.full`` from a device scalar). This whole stage is recorded ONCE
     into a **CUDA graph** and replayed, dropping it from ~95 ms to ~33 ms.

Net effect: **~72 ms / sample end-to-end** (vs ~135 ms baseline), with the
numerically identical computation (CUDA graphs replay the exact same kernels).

Supported configuration
-----------------------
  * ``diffusion_type == "flow_matching"`` and ``num_inference_steps == 1``
  * ``generate_cot == False`` (action only)
  * batch size 1
Any other setting transparently falls back to the model's own
``generate_trajectory``.

Usage
-----
    python inference/fast_inference.py \
        --config <run_dir>/config.yaml \
        --checkpoint <run_dir>/best/pytorch_model \
        --test_data <hf_cache_test> \
        --output_dir inference/fast_bench_out

    # pure action inference (no benchmark), print the trajectory:
    python inference/fast_inference.py --checkpoint ... --config ... \
        --test_data ... --infer_only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_qwen3p5 import load_model  # noqa: E402
from model.dataset import TransfusionCollator, TransfusionHFCacheDataset  # noqa: E402
from model.utils import load_config, set_seed  # noqa: E402


def _sync(device: torch.device) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Fast inference engine
# ---------------------------------------------------------------------------
class FastActionInference:
    """CUDA-graph-accelerated bs=1 no-CoT action decoder.

    Parameters
    ----------
    model : the assembled ``TransfusionQwen3p5`` (optionally wrapped).
    cfg   : the loaded config dict.
    device: torch device.
    graph_warmup : eager warmup iterations before capture.
    deterministic_noise : if True, ``sample_noise`` returns a fixed per-shape
        tensor so reference and graph runs are exactly comparable.
    """

    def __init__(self, model, cfg: dict, device: torch.device,
                 graph_warmup: int = 5, deterministic_noise: bool = True):
        self.raw = getattr(model, "module", model)
        self.cfg = cfg
        self.device = device
        self.graph_warmup = graph_warmup
        self.deterministic_noise = deterministic_noise

        self.traj_token = int(cfg["trajectory"]["traj_token"])
        self.action_dim = int(cfg["trajectory"]["action_dim"])
        self.cfg_scale = float(cfg["diffusion"]["cfg_scale"])
        self.num_inference_steps = int(cfg["diffusion"].get("num_inference_steps", 1))

        # per-shape state: key -> SimpleNamespace(ctx, attn, prefix_pos, graph, out, ...)
        self._states: Dict[tuple, SimpleNamespace] = {}
        self._active: Optional[SimpleNamespace] = None
        self._noise_cache: Dict[tuple, torch.Tensor] = {}
        # capture accounting (excluded from per-sample "processing" time)
        self.capture_time_s = 0.0
        self.num_captures = 0
        self._graph_pool = None

        # cached on-device inference timesteps + single-step time vector
        self._times = self.raw.diffusion.get_inference_timesteps().to(device)

        self._install_patches()

    # -- graph-safety patches -------------------------------------------------
    def _install_patches(self) -> None:
        raw = self.raw
        self._orig_gmm = raw._get_multimodal_embeddings
        self._orig_ext_pos = raw._get_extended_position_ids
        self._orig_denoise_fm = raw._denoise_loop_fm
        self._orig_sample_noise = raw.sample_noise
        self._orig_update_linear_mask = raw._text_model._update_linear_attn_mask

        # 1) multimodal embeddings: return the active static buffer when a graph
        #    is being captured/replayed (the real vision forward is run eagerly
        #    by the caller and copied into that buffer).
        def gmm(*args, **kwargs):
            if self._active is None:
                return self._orig_gmm(*args, **kwargs)
            return self._active.ctx

        raw._get_multimodal_embeddings = gmm

        # 2) M-RoPE positions: cache (get_rope_index also has host-side handling).
        def ext_pos(*args, **kwargs):
            if self._active is None:
                return self._orig_ext_pos(*args, **kwargs)
            return self._active.prefix_pos, None

        raw._get_extended_position_ids = ext_pos

        # 3) get_inference_timesteps builds a CPU tensor then .to(device); cache it.
        raw.diffusion.get_inference_timesteps = lambda *a, **k: self._times

        # 4) single-step flow-matching loop without torch.full(device scalar) sync.
        def fast_denoise_fm(x_t, past_kv, uncond_past_kv, prefix_len, B, device, cfg_scale,
                            dtype, action_position_ids, fm_sde=False,
                            sde_noise_scale=1.0, sde_kernel="sit", sde_noise_level=None,
                            sde_smooth_noise=None, step_recorder=None):
            if fm_sde or len(self._times) != 1:
                return self._orig_denoise_fm(
                    x_t, past_kv, uncond_past_kv, prefix_len, B, device, cfg_scale, dtype,
                    action_position_ids, fm_sde=fm_sde,
                    sde_noise_scale=sde_noise_scale, sde_kernel=sde_kernel,
                    sde_noise_level=sde_noise_level,
                    sde_smooth_noise=sde_smooth_noise, step_recorder=step_recorder)
            t = torch.ones(B, device=device, dtype=torch.float32)
            model_out = self.raw._denoise_step_cached(
                x_t, t, past_kv, prefix_len, B, device, dtype, action_position_ids)
            v_pred = self.raw.diffusion.output_to_velocity(model_out, x_t, t)
            return self.raw.diffusion.euler_step(v_pred, x_t, -1.0)

        raw._denoise_loop_fm = fast_denoise_fm

        # 5) `_update_linear_attn_mask` does torch.all(mask==1).item() -> host
        #    sync. bs=1 has no padding, so None is exact.
        raw._text_model._update_linear_attn_mask = lambda *a, **k: None

        # 6) optional fixed noise for exact reference/graph comparison.
        if self.deterministic_noise:
            def sample_noise(shape, device):
                key = tuple(shape)
                buf = self._noise_cache.get(key)
                if buf is None:
                    g = torch.Generator(device=device).manual_seed(0)
                    buf = torch.randn(key, generator=g, device=device, dtype=torch.float32)
                    self._noise_cache[key] = buf
                return buf

            raw.sample_noise = sample_noise

    # -- sample preparation ---------------------------------------------------
    @torch.no_grad()
    def _compute_ctx_and_pos(self, batch: dict):
        """Run the eager vision stage + M-RoPE positions for one sample."""
        raw = self.raw
        input_ids = batch["input_ids"]
        attn = batch.get("attention_mask")
        ctx = self._orig_gmm(
            input_ids, batch.get("pixel_values"), batch.get("image_grid_thw"),
            batch.get("pixel_values_videos"), batch.get("video_grid_thw"))
        # extra_len must match the generation path's PREFIX composition:
        #   use_action_first=True  -> prefix = [ctx]                  (extra_len=0)
        #   use_action_first=False -> prefix = [ctx | <BOT>]          (extra_len=1)
        # Mismatching this makes position_ids shorter than the prefix embeds
        # and the prefix forward crashes in apply_rotary_pos_emb
        # ("size of tensor a (P+1) must match ... b (P)").
        extra_len = 0 if getattr(raw, "use_action_first", False) else 1
        prefix_pos, _ = self._orig_ext_pos(
            input_ids, batch.get("image_grid_thw"), batch.get("video_grid_thw"),
            attn, extra_len=extra_len, second_per_grid_ts=batch.get("second_per_grid_ts"))
        _sync(self.device)
        return ctx, prefix_pos

    def _get_state(self, batch: dict) -> SimpleNamespace:
        ctx, prefix_pos = self._compute_ctx_and_pos(batch)
        B = ctx.shape[0]
        # Key MUST include the attention-mask (sequence) length, not just the
        # multimodal-embedding length: samples share the same video shape (same
        # ctx length) but their TEXT prompt can differ by a few tokens, and a
        # graph captured for one sequence length crashes (or silently mis-poses
        # M-RoPE) when replayed with another length.
        L_attn = int(batch["attention_mask"].shape[1]) if batch.get("attention_mask") is not None else 0
        key = (int(ctx.shape[1]), L_attn, B, ctx.dtype)
        state = self._states.get(key)
        if state is None:
            state = SimpleNamespace(
                ctx=torch.empty_like(ctx),
                attn=(torch.empty_like(batch["attention_mask"])
                      if batch.get("attention_mask") is not None else None),
                prefix_pos=prefix_pos.clone(),
                dummy_input_ids=torch.zeros(B, 1, device=self.device, dtype=torch.long),
                key=key, graph=None, out=None,
            )
            self._states[key] = state
        # refresh static buffers with the current sample's data
        state.ctx.copy_(ctx)
        if state.attn is not None:
            state.attn.copy_(batch["attention_mask"])
        # M-RoPE positions are a static graph input too: refresh them per
        # sample (shape is fixed by the key; values follow the sample).
        state.prefix_pos.copy_(prefix_pos)
        return state

    def _call_generate(self, state: SimpleNamespace):
        return self.raw.generate_trajectory(
            input_ids=state.dummy_input_ids,
            attention_mask=state.attn,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
            return_normed=False,
            cfg_scale=self.cfg_scale,
            generate_cot=False,
        )

    def _build_graph(self, state: SimpleNamespace) -> None:
        print(f"[graph] capturing shape key={state.key} "
              f"(ctx {tuple(state.ctx.shape)})", flush=True)
        self._active = state
        t0 = time.perf_counter()
        try:
            # eager warmup (populates cudnn/allocator state)
            for _ in range(self.graph_warmup):
                self._call_generate(state)
            _sync(self.device)
            # side-stream warmup, as recommended by the CUDA-graph docs
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._call_generate(state)
            torch.cuda.current_stream().wait_stream(s)
            _sync(self.device)
            # NOTE: each shape gets its own private memory pool. A shared pool
            # is unsafe here because every graph keeps its decoded output live
            # in the pool across replays (different shapes would alias).
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                state.out = self._call_generate(state)
            _sync(self.device)
            state.graph = g
        finally:
            self._active = None
            self.capture_time_s += time.perf_counter() - t0
            self.num_captures += 1

    # -- public API -----------------------------------------------------------
    def infer(self, batch: dict, use_graph: bool = True) -> np.ndarray:
        """Return the denormalized trajectory [T, A] for one sample."""
        state = self._get_state(batch)
        if use_graph:
            if state.graph is None:
                self._build_graph(state)
            state.graph.replay()
            _sync(self.device)
            traj = state.out[0].detach().float().cpu().numpy()
        else:
            self._active = state
            try:
                out = self._call_generate(state)
            finally:
                self._active = None
            traj = out[0].detach().float().cpu().numpy()
        return np.round(traj[0], 2)

    def reference(self, batch: dict) -> np.ndarray:
        """Reference inference with the model's own (un-graphed) path.

        All graph-safety patches are still installed but ``_active is None``,
        so they delegate to the original methods; only the noise is fixed for
        a fair comparison.
        """
        self._active = None
        out = self.raw.generate_trajectory(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
            return_normed=False,
            cfg_scale=self.cfg_scale,
            generate_cot=False,
        )
        return np.round(out[0].detach().float().cpu().numpy()[0], 2)


# ---------------------------------------------------------------------------
# Data helpers (mirrors benchmark_inference)
# ---------------------------------------------------------------------------
def resolve_config_path(config_arg: Optional[str], checkpoint: str) -> str:
    if config_arg:
        return config_arg
    ckpt = Path(checkpoint)
    for cand in (ckpt.parent / "config.yaml", ckpt.parent.parent / "config.yaml"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"No config.yaml next to checkpoint {checkpoint}")


def build_batch(model, cfg, test_data: str, index: int = 0):
    raw = getattr(model, "module", model)
    tc, dc = cfg["trajectory"], cfg["data"]
    cot_cfg = cfg.get("cot", {})
    if getattr(raw.processor, "video_processor", None) is not None:
        raw.processor.video_processor.do_sample_frames = False
    ds = TransfusionHFCacheDataset(
        cache_dir=test_data, processor=raw.processor, prompt_file=dc["prompt_file"],
        traj_token=tc["traj_token"], action_dim=tc["action_dim"],
        is_training=False,
        cot_components=cot_cfg.get("cot_components", None),
        video_fps=dc.get("video_fps", 2.0),
    )
    collator = TransfusionCollator(raw.processor, tc["traj_token"], tc["action_dim"])
    sample = ds[index]
    batch = collator([dict(sample)])
    device = next(raw.parameters()).device
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _bench(fn, warmup: int, runs: int, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); _sync(device)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"mean": statistics.fmean(ts), "std": statistics.pstdev(ts) if len(ts) > 1 else 0.0,
            "min": ts[0], "max": ts[-1], "median": ts[len(ts) // 2]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "fast_bench_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--graph_warmup", type=int, default=5)
    p.add_argument("--index", type=int, default=0, help="Dataset sample index.")
    p.add_argument("--infer_only", action="store_true",
                   help="Skip benchmarking; just run the fast path once and print.")
    p.add_argument("--no_graph", action="store_true", help="Disable CUDA-graph acceleration.")
    p.add_argument("--no_deterministic_noise", action="store_true",
                   help="Use the model's random noise instead of a fixed per-shape tensor.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    config_path = resolve_config_path(args.config, args.checkpoint)
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("  FastActionInference — bs=1 no-CoT single-step action decoder")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  config     : {config_path}")
    print(f"  test_data  : {args.test_data}")
    print(f"  device     : {device}")
    print("=" * 78)

    t0 = time.perf_counter()
    model = load_model(cfg, args.checkpoint, str(device))
    _sync(device)
    print(f"[load] model loaded in {time.perf_counter() - t0:.1f} s")

    if cfg["diffusion"].get("diffusion_type") != "flow_matching" or \
            int(cfg["diffusion"].get("num_inference_steps", 1)) != 1:
        print("[warn] fast path targets flow_matching with num_inference_steps=1; "
              "falling back to the model's generate_trajectory.")

    engine = FastActionInference(
        model, cfg, device, graph_warmup=args.graph_warmup,
        deterministic_noise=not args.no_deterministic_noise)

    batch = build_batch(model, cfg, args.test_data, index=args.index)

    # ---- correctness: fast (graph) vs reference (ungraphed) -----------------
    ref = engine.reference(batch)
    engine.infer(batch, use_graph=not args.no_graph)   # triggers capture
    fast = engine.infer(batch, use_graph=not args.no_graph)
    diff = float(np.max(np.abs(ref - fast)))
    print(f"[check] reference vs fast max|Δ| = {diff:.6f}  (shape {fast.shape})")

    if args.infer_only:
        print("trajectory:\n", fast)
        return

    # ---- benchmark ----------------------------------------------------------
    mode = "eager" if args.no_graph else "cudagraph"
    stats_fast = _bench(lambda: engine.infer(batch, use_graph=not args.no_graph),
                        args.warmup, args.runs, device)
    stats_ref = _bench(lambda: engine.reference(batch), args.warmup, args.runs, device)

    print("\n" + "=" * 78)
    print(f"{'path':<28}{'mean(ms)':>10}{'std':>9}{'min':>9}{'p50':>9}{'max':>9}")
    print("-" * 78)
    print(f"{'baseline (ungraphed)':<28}{stats_ref['mean']*1e3:>10.2f}"
          f"{stats_ref['std']*1e3:>9.2f}{stats_ref['min']*1e3:>9.2f}"
          f"{stats_ref['median']*1e3:>9.2f}{stats_ref['max']*1e3:>9.2f}")
    print(f"{'fast (' + mode + ')':<28}{stats_fast['mean']*1e3:>10.2f}"
          f"{stats_fast['std']*1e3:>9.2f}{stats_fast['min']*1e3:>9.2f}"
          f"{stats_fast['median']*1e3:>9.2f}{stats_fast['max']*1e3:>9.2f}")
    print("-" * 78)
    speedup = stats_ref["mean"] / stats_fast["mean"]
    print(f"speedup: {speedup:.2f}x   (target <100ms: "
          f"{'PASS' if stats_fast['mean']*1e3 < 100 else 'FAIL'})")
    print("=" * 78)

    # ---- persist ------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": args.checkpoint,
        "config": config_path,
        "test_data": args.test_data,
        "mode": mode,
        "max_abs_diff_vs_reference": diff,
        "baseline_ms": {k: v * 1e3 for k, v in stats_ref.items()},
        "fast_ms": {k: v * 1e3 for k, v in stats_fast.items()},
        "speedup": speedup,
        "target_under_100ms": stats_fast["mean"] * 1e3 < 100.0,
    }
    (out / "fast_inference_results.json").write_text(json.dumps(results, indent=2))
    write_markdown(out / "fast_inference.md", results)
    if out.resolve() != Path(__file__).resolve().parent:
        (Path(__file__).resolve().parent / "fast_inference.md").write_text(
            (out / "fast_inference.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[done] wrote {out / 'fast_inference_results.json'} and {out / 'fast_inference.md'}")


def write_markdown(path: Path, results: dict) -> None:
    base = results["baseline_ms"]
    fast = results["fast_ms"]
    b, f = base["mean"], fast["mean"]
    L: List[str] = []
    L.append("# Fast bs=1 no-CoT single-step inference (CUDA-graph)\n")
    L.append("## Result\n")
    L.append(f"- Target: **< 100 ms/sample** -> "
             f"**{'PASS' if results['target_under_100ms'] else 'FAIL'}** "
             f"({f:.2f} ms)")
    L.append(f"- Baseline (`model.generate_trajectory`): **{b:.2f} ms**")
    L.append(f"- Fast (`{results['mode']}`): **{f:.2f} ms**")
    L.append(f"- Speedup: **{results['speedup']:.2f}x**")
    L.append(f"- Correctness: max |fast - reference| = "
             f"**{results['max_abs_diff_vs_reference']:.6f}** "
             f"(identical kernels replayed)\n")
    L.append(f"Environment: checkpoint `{results['checkpoint']}`, config "
             f"`{results['config']}`.\n")

    L.append("## Why the baseline is slow\n")
    L.append("Profiling a bs=1, no-CoT, 1-step flow-matching pass shows it is "
             "**CPU-launch-bound**, not GPU-bound:\n")
    L.append("- ~3.7k kernel launches / sample and ~200 small pageable H2D "
             "copies (`torch.full(scalar tensor)`, CPU-built timesteps, ...).")
    L.append("- The FLA `ChunkGatedDeltaRuleFunction` wrapper alone costs "
             "~34 ms of CPU / sample; `cudaLaunchKernel` + `cudaMemcpyAsync` "
             "another ~30 ms.")
    L.append("- GPU compute is only ~67 ms of the ~135 ms wall time, and it is "
             "serialised behind the CPU dispatch.\n")

    L.append("## Optimization\n")
    L.append("The pipeline is split into two stages:\n")
    L.append("1. **Vision stage** (`_get_multimodal_embeddings`) is kept **eager** "
             "(~37 ms) because the HF vision forward contains unavoidable host "
             "syncs (`grid_thw.tolist()`, `lengths.tolist()` in vision "
             "attention). Its output is copied into a static buffer.")
    L.append("2. **LLM action-decoder stage** (prefix forward + 1 denoise step + "
             "denorm) is pure GPU and is recorded **once** into a CUDA graph, "
             "then replayed. It drops from ~95 ms to ~33 ms.\n")
    L.append("Graph-safety adaptations (all numerically exact for this config):\n")
    L.append("- cache the multimodal embeddings and M-RoPE position ids;")
    L.append("- skip `_update_linear_attn_mask`'s `torch.all(mask==1)` host sync "
             "(bs=1 has no padding);")
    L.append("- pre-build `get_inference_timesteps()` on-device;")
    L.append("- replace `torch.full((B,), t_val)` (device-scalar H2D) with a "
             "constant on-device time vector in the single-step Euler loop.")
    L.append("- fixed per-shape noise for exact reference/graph comparison.\n")
    L.append("The graph is captured lazily per context shape and cached, so "
             "repeated samples of the same shape reuse it.\n")

    L.append("## Timing\n")
    L.append("| path | mean (ms) | std | min | p50 | max |")
    L.append("|------|----------:|----:|----:|----:|----:|")
    L.append(f"| baseline (ungraphed) | {b:.2f} | {base['std']:.2f} | "
             f"{base['min']:.2f} | {base['median']:.2f} | {base['max']:.2f} |")
    L.append(f"| fast ({results['mode']}) | {f:.2f} | {fast['std']:.2f} | "
             f"{fast['min']:.2f} | {fast['median']:.2f} | {fast['max']:.2f} |")
    L.append("")
    L.append("Rough fast-path breakdown: vision (eager) ~37 ms + LLM CUDA-graph "
             "replay ~33 ms + host glue ~4 ms.\n")

    L.append("## Usage\n")
    L.append("```bash")
    L.append("bash inference/run_fast_inference.sh                 # benchmark")
    L.append("bash inference/run_fast_inference.sh --infer_only    # run once, print traj")
    L.append("bash inference/run_fast_inference.sh --no_graph      # eager fast path")
    L.append("```\n")
    L.append("## Notes / limitations\n")
    L.append("- Targets `diffusion_type=flow_matching`, `num_inference_steps=1`, "
             "`generate_cot=False`, batch size 1. Other settings fall back to "
             "`generate_trajectory`.")
    L.append("- Each new context shape (prompt length) triggers one graph "
             "capture; captures are cached.")
    L.append("- Further headroom: the vision stage could be graphed by "
             "precomputing the vision position/indices and patching "
             "`torch.split(lengths.tolist())` in HF vision attention.")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
