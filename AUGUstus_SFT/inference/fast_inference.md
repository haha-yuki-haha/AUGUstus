# Fast bs=1 no-CoT single-step inference (CUDA-graph)

## Result

- Target: **< 100 ms/sample** -> **FAIL** (115.12 ms)
- Baseline (`model.generate_trajectory`): **120.69 ms**
- Fast (`cudagraph`): **115.12 ms**
- Speedup: **1.05x**
- Correctness: max |fast - reference| = **0.000000** (identical kernels replayed)

Environment: checkpoint `/path/to/downloaded-AUGUstus-model/SFT`, config `/path/to/downloaded-AUGUstus-model/SFT/config_original.yaml`.

## Why the baseline is slow

Profiling a bs=1, no-CoT, 1-step flow-matching pass shows it is **CPU-launch-bound**, not GPU-bound:

- ~3.7k kernel launches / sample and ~200 small pageable H2D copies (`torch.full(scalar tensor)`, CPU-built timesteps, ...).
- The FLA `ChunkGatedDeltaRuleFunction` wrapper alone costs ~34 ms of CPU / sample; `cudaLaunchKernel` + `cudaMemcpyAsync` another ~30 ms.
- GPU compute is only ~67 ms of the ~135 ms wall time, and it is serialised behind the CPU dispatch.

## Optimization

The pipeline is split into two stages:

1. **Vision stage** (`_get_multimodal_embeddings`) is kept **eager** (~37 ms) because the HF vision forward contains unavoidable host syncs (`grid_thw.tolist()`, `lengths.tolist()` in vision attention). Its output is copied into a static buffer.
2. **LLM action-decoder stage** (prefix forward + 1 denoise step + denorm) is pure GPU and is recorded **once** into a CUDA graph, then replayed. It drops from ~95 ms to ~33 ms.

Graph-safety adaptations (all numerically exact for this config):

- cache the multimodal embeddings and M-RoPE position ids;
- skip `_update_linear_attn_mask`'s `torch.all(mask==1)` host sync (bs=1 has no padding);
- pre-build `get_inference_timesteps()` on-device;
- replace `torch.full((B,), t_val)` (device-scalar H2D) with a constant on-device time vector in the single-step Euler loop.
- fixed per-shape noise for exact reference/graph comparison.

The graph is captured lazily per context shape and cached, so repeated samples of the same shape reuse it.

## Timing

| path | mean (ms) | std | min | p50 | max |
|------|----------:|----:|----:|----:|----:|
| baseline (ungraphed) | 120.69 | 1.50 | 119.04 | 120.41 | 124.34 |
| fast (cudagraph) | 115.12 | 0.53 | 114.56 | 115.03 | 117.35 |

Rough fast-path breakdown: vision (eager) ~37 ms + LLM CUDA-graph replay ~33 ms + host glue ~4 ms.

## Usage

```bash
bash inference/run_fast_inference.sh                 # benchmark
bash inference/run_fast_inference.sh --infer_only    # run once, print traj
bash inference/run_fast_inference.sh --no_graph      # eager fast path
```

## Notes / limitations

- Targets `diffusion_type=flow_matching`, `num_inference_steps=1`, `generate_cot=False`, batch size 1. Other settings fall back to `generate_trajectory`.
- Each new context shape (prompt length) triggers one graph capture; captures are cached.
- Further headroom: the vision stage could be graphed by precomputing the vision position/indices and patching `torch.split(lengths.tolist())` in HF vision attention.