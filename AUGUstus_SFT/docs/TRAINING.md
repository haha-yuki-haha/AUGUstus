# AUGUstus_SFT — Training Guide (NAVSIM / nuPlan environment & runs)

This guide covers everything needed to **train** the Transfusion Qwen3.5 model
on NAVSIM, including how the **NAVSIM / nuPlan devkit environments** work and
how to handle them. For installation see [INSTALL.md](INSTALL.md); for data
preparation see [DATA_PREPARATION.md](DATA_PREPARATION.md).

---

## 1. The NAVSIM / nuPlan environment — what it is and how it is handled

### 1.1 Two devkits, one env

The model depends on two research devkits:

| Devkit | Version | Role in this repo |
|--------|---------|-------------------|
| **navsim** | 1.1.0 | dataset split definitions (navtrain/navtest), metric caching, PDMS scoring |
| **nuplan-devkit** | 1.2.0 | maps + simulation primitives used by navsim under the hood |

**We ship both inside `third_party/`** — the exact copies the model was trained
and evaluated with. Do **not** replace them with upstream clones: the shipped
navsim contains local modifications (e.g. `PDMScorerConfig`, used by
`model/utils.py::compute_pdm_score_single`); upstream navsim will fail at PDMS
scoring time. Both are Apache-2.0 licensed.

They are installed **editable, without their own dependencies**
(`pip install -e --no-deps`): every dependency they need (shapely, pyquaternion,
pandas, scikit-learn, hydra-core, omegaconf, ray, rasterio, geopandas, ...) is
already pinned in `requirements.txt`. `scripts/setup_env.sh` does this for you.

### 1.2 Environment variables (all set by `env.sh`)

nuplan/navsim read configuration from environment variables. `env.sh` sets the
authoritative values for this repo — **do not rely on machine-level
`NAVSIM_*` / `NUPLAN_*` variables**:

| Variable | Default (from `env.sh`) | Used for |
|----------|------------------------|----------|
| `NAVSIM_ROOT` | `$ADT_ROOT/Data/navsim` | raw logs / sensor blobs / maps |
| `NAVSIM_DEVKIT_ROOT` | `$AUGUSTUS_CODE_ROOT/third_party/navsim` | devkit import path |
| `NUPLAN_DEVKIT_ROOT` | `$AUGUSTUS_CODE_ROOT/third_party/nuplan-devkit` | devkit import path |
| `NUPLAN_MAPS_ROOT` | `$NAVSIM_ROOT/maps` | nuPlan maps v1.0 (metric caching) |
| `NUPLAN_MAP_VERSION` | `nuplan-maps-v1.0` | maps package name |
| `NAVSIM_EXP_ROOT` | `$AUGUSTUS_DATA_ROOT/navsim_exp` | metric-cache output dir |
| `OPENSCENE_DATA_ROOT` | `$NAVSIM_ROOT` | OpenScene data layout |

> **Gotcha (shared machines):** cluster profiles sometimes inject stale
> `NAVSIM_DEVKIT_ROOT` / `NAVSIM_EXP_ROOT` / `NUPLAN_MAPS_ROOT` variables left
> over from *other* projects (e.g. pointing at a different repo checkout or a
> non-existent `/data/navsim`). Because `env.sh` uses `${VAR:-default}`, those
> injected values would silently win and break imports / PDMS. `env.sh`
> therefore **overrides** them with the repo layout; to change a location use
> the `AUGUSTUS_*` variables (`AUGUSTUS_EXP_ROOT`, `AUGUSTUS_MAPS_ROOT`,
> `AUGUSTUS_NAVSIM_DEVKIT_ROOT`, ...) instead of the plain `NAVSIM_*` names.
> Quick sanity check:
>
> ```bash
> source env.sh && _activate_augustus_env
> python -c "import navsim, nuplan; print(navsim.__file__); print(nuplan.__file__)"
> # both must point into $AUGUSTUS_CODE_ROOT/third_party/
> ```

### 1.3 PDMS metric caches

Validation PDMS needs the **navtest metric cache** (per-frame closed-loop
simulation contexts, ~137 lzma pickles). Build it once from the raw NAVSIM
data:

```bash
bash NAVSIM_Cache.sh                        # navtest -> $NAVSIM_EXP_ROOT/metric_cache_test
# optional: training-set cache
TRAIN_TEST_SPLIT=trainval bash NAVSIM_Cache.sh
```

This runs `navsim/planning/script/run_metric_caching.py` with
`NUPLAN_MAPS_ROOT` pointing at the maps shipped in the raw dataset. It takes
roughly 1-2 h for navtest (one-time cost; the cache is only a few GB).

If `data.metric_cache_test` does not exist, training still runs — eval just
skips PDMS and reports ADE/FDE only.

---

## 2. Training

### 2.1 Quick start (8 GPUs, DeepSpeed ZeRO-2 — the released recipe)

```bash
cd "$AUGUSTUS_CODE_ROOT"
source env.sh && _activate_augustus_env
bash scripts/run_deepspeed.sh               # = accelerate launch, configs/navsim_qwen3p5.yaml
```

- 8 GPUs × 32 per-device batch = **256 effective batch** → ~403 steps/epoch on
  the 103 287-sample train cache.
- On machines with a different GPU count, keep the effective batch at 256 by
  setting `per_device_train_batch_size: 256 / <num_gpus>` (e.g. 64 on 4 GPUs).
- Outputs land in
  `$AUGUSTUS_MODEL_ROOT/<run_name>/<timestamp>/` (`checkpoint-<N>/`,
  `best/pytorch_model`, `config.yaml`, swanlab logs).
- `best/` is selected by `training.best_metric: PDMS` (higher is better).

### 2.2 Experiment tracking (swanlab)

`configs/navsim_qwen3p5.yaml` logs to console and SwanLab. Set
`AUGUSTUS_SWANLAB_TOKEN` to your own API key for cloud logging, or use
offline logging without an account.

```bash
SWANLAB_MODE=offline bash scripts/run_deepspeed.sh   # local only (default, no account needed)
SWANLAB_MODE=cloud   bash scripts/run_deepspeed.sh   # sync to swanlab.cn
```

Useful variants:

```bash
GPUS=4 bash scripts/run_deepspeed.sh        # fewer GPUs (set per_device_train_batch_size 256/4 = 64 in the config to keep the effective batch)
```

### 2.3 Plain accelerate (no DeepSpeed)

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
    train_qwen3p5.py --config configs/navsim_qwen3p5.yaml
```

Single GPU debugging:

```bash
CUDA_VISIBLE_DEVICES=0 python train_qwen3p5.py --config configs/navsim_qwen3p5.yaml
```

### 2.4 What one training step does

1. Video history (4 frames @ 2 Hz, 2048×512 stitched 3-camera view) is encoded
   by the Qwen3.5-VL vision tower.
2. GT waypoints (8 × [x, y, heading]) are noised by the flow-matching forward
   process and fused into the token sequence after `<BOT>` (action-first:
   actions → CoT → `<EOT>`).
3. Losses: `lambda_diffusion=10` (x0 MSE on action positions) +
   `lambda_lm=1` (CoT LM loss) + auxiliary L1 / recon terms.
4. Eval every `eval_steps=400`: full val set, ADE/FDE + PDMS against the
   navtest metric cache (sharded across ranks).

---

## 3. Reproducing the released best run

Released run:
`navsim-qwen3p5-2b-vit-AdaW-cot_prediction_action_first-fm-x0-adanorm-raw-dit-lr1e-5_clip0.0_diff10_bs256-2048_512`
(timestamp `20260718_131023`).

### Full training run (100 epochs)

```bash
bash scripts/run_deepspeed.sh               # configs/navsim_qwen3p5.yaml
```

The released best checkpoint was the `best/` snapshot (highest val/PDMS) of
this run.

---

## 4. Evaluation

See **[`inference/README.md`](../inference/README.md)** for the full guide
(multi-GPU sharded eval, denoising-step and CoT toggles, sampler options).
Quick reference:

```bash
# multi-GPU sharded eval (ADE/FDE/PDMS + visualizations):
GPUS=0,1,2,3 bash scripts/test_checkpoint.sh
# adjust denoising steps / CoT generation:
NUM_INFERENCE_STEPS=10 GPUS=0,1,2,3 bash scripts/test_checkpoint.sh
GENERATE_COT=0        GPUS=0,1,2,3 bash scripts/test_checkpoint.sh   # action-only
# or directly:
accelerate launch --num_processes 4 test_qwen3p5.py \
    --config configs/navsim_qwen3p5.yaml \
    --checkpoint <run>/<timestamp>/best/pytorch_model \
    --test_data ${AUGUSTUS_DATA_ROOT}/4_data_2048_512/hf_cache_test_CoT_Solution \
    --batch_eval --generate_cot auto --visualize
```

Latency benchmarks (bs=1, CUDA-graph fast path): `inference/run_benchmark.sh`,
`inference/run_fast_inference.sh`; standalone PDMS scoring:
`inference/run_pdms_eval.sh`.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `no kernel image is available for execution on the device` | torch build predates the GPU architecture — pro card: use the pinned `torch==2.13.0 + cu130` wheels from `requirements.txt`; H800 (sm_90) on driver 550.x: the cu130 wheels can't run (driver too old) — use the cu126 env `augustus_sft` (see INSTALL.md §6) |
| `import navsim` resolves outside `third_party/` | stale injected `NAVSIM_DEVKIT_ROOT` — re-source `env.sh`; see §1.2 |
| PDMS = 0 / skipped in eval | `data.metric_cache_test` missing → build with `NAVSIM_Cache.sh`; check `NAVSIM_EXP_ROOT` |
| metric caching fails on maps | `NUPLAN_MAPS_ROOT` must point at `$NAVSIM_ROOT/maps` (nuPlan maps v1.0) |
| pip installs wrong versions | global `/etc/pip/constraint.txt` interference — `scripts/setup_env.sh` unsets `PIP_CONSTRAINT`; install via the script |
| pip resolves a broken version mix | install through `scripts/setup_env.sh` with the matching GPU profile and requirements file |
| `Unsupported .version 9.4` while building causal-conv1d | nvcc/nvvm mismatch — handled by `setup_env.sh` (pins nvcc & nvvm to 13.3.73) |
| DataLoader NFS stalls | keep `dataloader_num_workers: 8` per rank; the HF cache reads whole mp4 blobs (better on local disk) |
| DeepSpeed checkpoint resume fails with FileNotFoundError | incomplete multi-node checkpoint (shards scattered across timestamp dirs) — the trainer detects and refuses; point `resume_path` at a complete `checkpoint-<N>` |

### 5.1 Sanity check the env before a long run

```bash
python scripts/smoke_test.py
```

Builds the model from `$QWEN35_2B_PATH` on GPU, runs one training forward +
backward and one KV-cache generation. It checks the model interface and
reports whether the GatedDeltaNet fast path is available.

If `GatedDeltaNet fast path bound: False`, causal-conv1d was not built (the
model still trains, just slower — re-run `scripts/setup_env.sh`).
