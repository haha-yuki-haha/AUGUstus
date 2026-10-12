# AUGUstus_SFT — Inference & Evaluation Guide

How to evaluate a checkpoint: **multi-GPU batch eval** (recommended), plus the
single-GPU benchmark tools. All scripts source `env.sh` and activate the env.

---

## 1. Multi-GPU evaluation (`scripts/test_checkpoint.sh`)

Sharded eval over the NAVSIM test cache with accelerate: each GPU evaluates its
shard, metrics (ADE / FDE / PDMS + sub-metrics) are gathered and reported once,
with per-sample dumps + visualizations under the output dir.

```bash
cd "$ADT_ROOT/AUGUstus_SFT"
conda activate augustus_sft          # or: source env.sh && _activate_augustus_env

# ---- 4-GPU eval of the released SFT best (defaults) ----
GPUS=0,1,2,3 bash scripts/test_checkpoint.sh

# ---- adjust the DENOISING STEPS (flow-matching integration steps) ----
NUM_INFERENCE_STEPS=10 GPUS=0,1,2,3 bash scripts/test_checkpoint.sh   # 10-step ODE
NUM_INFERENCE_STEPS=1  GPUS=0,1,2,3 bash scripts/test_checkpoint.sh   # 1-step (the SFT default)

# ---- toggle CoT GENERATION ----
GENERATE_COT=0 GPUS=0,1,2,3 bash scripts/test_checkpoint.sh   # action-only (fast; no CoT decode)
GENERATE_COT=1 GPUS=0,1,2,3 bash scripts/test_checkpoint.sh   # force CoT on (CoT-trained ckpts)
                                                              # (default: auto = config's
                                                              #  cot.enable_cot AND cot.eval_generate_cot)

# ---- other knobs ----
BATCH_SIZE=8 MAX_SAMPLES=500 bash scripts/test_checkpoint.sh   # subset / batch size
CHECKPOINT=/path/to/best/pytorch_model \
  CONFIG=/path/to/config_original.yaml \
  TEST_DATA=/path/to/hf_cache_test_CoT_Solution \
  bash scripts/test_checkpoint.sh                              # explicit paths

# ---- stochastic SDE sampler instead of the deterministic ODE ----
FM_SDE=1 SDE_NOISE_SCALE=1.0 SEED=42 bash scripts/test_checkpoint.sh

# ---- dump every denoising step's inputs/outputs (debug/analysis) ----
SAVE_STEPS=1 NUM_INFERENCE_STEPS=1 bash scripts/test_checkpoint.sh
```

| Env var | Default | Meaning |
|---------|---------|---------|
| `GPUS` | `0,1,...` | CUDA devices to shard the eval across (accelerate launches one process each) |
| `NUM_INFERENCE_STEPS` | config | denoising steps for the flow-matching sampler (1 = one-step SFT model) |
| `GENERATE_COT` | `auto` | `auto` = config (`cot.enable_cot` + `cot.eval_generate_cot`) · `1` = force CoT on · `0` = action-only |
| `BATCH_SIZE` | 16 | per-GPU eval batch |
| `MAX_SAMPLES` | full split | cap the number of eval samples |
| `FM_SDE` / `SDE_NOISE_SCALE` / `SDE_*` | ODE | stochastic sampler (see the header of `test_checkpoint.sh`) |
| `SAVE_STEPS` | 0 | dump per-step denoising tensors to `<output>/denoise_steps/` |

Outputs (under `${CHECKPOINT}/eval_outputs_<sampler-tag>/` by default):
`eval_results.json` (summary metrics), `predictions*.npy` /
`ground_truths.npy`, per-sample CoT text + trajectory visualizations,
optionally `denoise_steps/`.

> Multi-GPU semantics: samples are SHARDED (`batch_size` per GPU per step), so
> total wall-time ≈ full-split time / num GPUs. Metrics are gathered across
> ranks before reporting — the numbers match a single-GPU run (same seed).

## 2. Single-GPU tools (`inference/`)

| Script | What it does | Key knobs |
|--------|--------------|-----------|
| `inference/run_benchmark.sh` | per-module latency breakdown (vision / prefix / denoising / CoT decode) | `RUNS`, `NUM_INFERENCE_STEPS`, `GENERATE_COT=auto\|1\|0`, `MAX_COT_TOKENS` |
| `inference/run_fast_inference.sh` | CUDA-graph bs=1 fast path (~72 ms/sample) | `RUNS`, `--no_graph`, `--infer_only` |
| `inference/run_pdms_eval.sh` | bs=1 loop + full PDMS per sample (json report) | `MAX_SAMPLES`, `NUM_WORKERS`, `PDM_WORKERS`, `--no_pdms` |

All three take `CHECKPOINT=... CONFIG=... TEST_DATA=...` like the multi-GPU
script. Use them for timing/profiling; use `scripts/test_checkpoint.sh` for
dataset-scale metric evaluation.
