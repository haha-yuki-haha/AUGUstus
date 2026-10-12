# AUGUstus_GRPO — Training Guide (GRPO RL, NAVSIM / nuPlan environment & runs)

This guide covers everything needed to **launch GRPO RL training** of the
Transfusion Qwen3.5-2B planner, including how the **NAVSIM / nuPlan devkit
environments** are handled and how to **align a run with the released
checkpoint**. For installation see [INSTALL.md](INSTALL.md); for data see
[DATA_PREPARATION.md](DATA_PREPARATION.md).

---

## 1. The NAVSIM / nuPlan environment — what it is and how it is handled

### 1.1 Two devkits, one env

RL depends on two research devkits (shipped in the sibling AUGUstus_SFT repo):

| Devkit | Version | Role in this repo |
|--------|---------|-------------------|
| **navsim** | 1.1.0 | dataset split definitions (navtrain/navtest), **metric caching, PDMS scoring inside the RL reward** |
| **nuplan-devkit** | 1.2.0 | maps + simulation primitives used by navsim under the hood |

**Do not replace them with upstream clones**: the shipped navsim contains local
modifications (e.g. `PDMScorerConfig`, used by
`model/utils.py::compute_pdm_scores_batch` — the PDMS term of the RL reward);
upstream navsim fails at scoring time. Both are installed **editable, without
their own dependencies** (`pip install -e --no-deps`) — every dependency they
need is pinned in the SFT `requirements.txt` (`scripts/setup_env.sh` does
this).

They are located via `env.sh` (`NAVSIM_DEVKIT_ROOT` / `NUPLAN_DEVKIT_ROOT` →
`$AUGUSTUS_SFT_ROOT/third_party/...`) and are ALSO on `PYTHONPATH` inside the
Ray workers (the recipe config's `ray_kwargs` block forwards `PYTHONPATH`), so
imports work in the driver and in every worker.

### 1.2 Environment variables (all set by `env.sh`)

nuplan/navsim read configuration from environment variables. `env.sh` sets the
authoritative values for this repo — **do not rely on machine-level
`NAVSIM_*` / `NUPLAN_*` variables**:

| Variable | Default (from `env.sh`) | Used for |
|----------|-------------------------|----------|
| `NAVSIM_ROOT` | `$ADT_ROOT/Data/navsim` | raw logs / sensor blobs / maps |
| `NAVSIM_DEVKIT_ROOT` | `$AUGUSTUS_SFT_ROOT/third_party/navsim` | devkit import path |
| `NUPLAN_DEVKIT_ROOT` | `$AUGUSTUS_SFT_ROOT/third_party/nuplan-devkit` | devkit import path |
| `NUPLAN_MAPS_ROOT` | `$NAVSIM_ROOT/maps` | nuPlan maps v1.0 (metric caching) |
| `NUPLAN_MAP_VERSION` | `nuplan-maps-v1.0` | maps package name |
| `NAVSIM_EXP_ROOT` | `$AUGUSTUS_DATA_ROOT/navsim_exp` | **PDMS metric caches** (train + test) |
| `OPENSCENE_DATA_ROOT` | `$NAVSIM_ROOT` | OpenScene data layout |

And the `TRANSFUSION_*` variables consumed by the recipe / every Ray worker:

| Variable | Default | Role |
|----------|---------|------|
| `TRANSFUSION_ROOT` | `$AUGUSTUS_SFT_ROOT` | model + dataset code (`model/`, `train_qwen3p5.py`) |
| `TRANSFUSION_CONFIG` | `$AUGUSTUS_SFT_RUN/config_original.yaml` | model-architecture config (must match the checkpoint) |
| `TRANSFUSION_CKPT` | `$AUGUSTUS_SFT_RUN/best` | SFT checkpoint the RL policy starts from ("" = cold start) |
| `TRANSFUSION_QWEN_MODEL_NAME` | `$AUGUSTUS_SFT_RUN/best/pytorch_model` | backbone arch + tokenizer + image/video processor |
| `TRANSFUSION_STATS_FILE` | recipe-bundled `action_norm_stats.json` | accessible stats file (values come from the checkpoint) |
| `TRANSFUSION_FREEZE_VIT` | `true` | freeze the visual tower during the actor update |
| `PYTHONPATH` | `$AUGUSTUS_GRPO_ROOT:$AUGUSTUS_SFT_ROOT` | `recipe.*` + `model.*` imports in driver + workers |

> **Gotcha (shared machines):** cluster profiles sometimes inject stale
> `NAVSIM_DEVKIT_ROOT` / `NAVSIM_EXP_ROOT` / `NUPLAN_MAPS_ROOT` variables left
> over from *other* projects. `env.sh` therefore **overrides** them with the
> repo layout; to change a location use the `AUGUSTUS_*` variables
> (`AUGUSTUS_EXP_ROOT`, `AUGUSTUS_MAPS_ROOT`, `AUGUSTUS_NAVSIM_DEVKIT_ROOT`,
> ...) instead of the plain `NAVSIM_*` names. Quick sanity check:
>
> ```bash
> source env.sh && _activate_augustus_env
> python -c "import navsim, nuplan; print(navsim.__file__); print(nuplan.__file__)"
> # both must point into $AUGUSTUS_SFT_ROOT/third_party/
> ```

### 1.3 PDMS metric caches (RL reward inputs)

The RL reward scores every rollout with the closed-loop NAVSIM PDM Score —
**per train sample during training** and per val sample during validation:

```bash
# both commands below must run with the GRPO env active (source env.sh first):
bash "$AUGUSTUS_SFT_ROOT/NAVSIM_Cache.sh"                  # navtest  -> $NAVSIM_EXP_ROOT/metric_cache_test
TRAIN_TEST_SPLIT=navtrain bash "$AUGUSTUS_SFT_ROOT/NAVSIM_Cache.sh"   # navtrain -> $NAVSIM_EXP_ROOT/metric_cache_train
```

- `metric_cache_train` is **REQUIRED for training** (the PDMS reward term); a
  missing cache degrades to no-PDMS (only prediction reward remains — the run
  would NOT align with the released one).
- `metric_cache_test` is **required for validation PDMS** (and best-checkpoint
  selection by `pdms_reward`).
- The maps (`NUPLAN_MAPS_ROOT`) must be present when building the caches.
- These are **REAL builds from the raw NAVSIM logs + maps** — do not symlink
  caches from other checkouts. The build procedure is documented in: [`DATA_PREPARATION.md`](DATA_PREPARATION.md) §5.

---

## 2. What a training step looks like

```
 TransfusionRLDataset (HF cache + metric_cache_train)
   └─> TransfusionRollout (sync): reverse-SDE trajectory (rl_num_steps=3,
        λ_t noise schedule [0.05,0.05,0.02])
        + autoregressive CoT (T=1.0, top-p 0.95, top-k 100)
   └─> reward (recipe.transfusion.reward_function):
        score = 1.0 * prediction(CoT bbox-IoU × future-position)
              + 5.0 * planning(1.0 * PDMS [+0*softL2 +0*endpoint], TTC+DAC
                              safe terms, reverse-trajectory gate)
   └─> hybrid GRPO actor update (loss.py):
        action head  — closed-form x0-prediction GRPO (jit_grpo.py)
        CoT head     — token-level PPO ratio, k3 KL estimator (cot_grpo.py)
        split advantages (action←planning block, CoT←prediction block)
        + KL-to-frozen-reference (action 1e-3, CoT 1e-2)
```

All knobs live in the recipe config
(`recipe/transfusion/config/navsim-qwen3p5_2b-grpo_g8-noise5e-2_step3-…yaml`,
commented inline; verl's inherited defaults are in
`verl/trainer/config/ppo_trainer.yaml`).

## 3. Launch

### 3.1 The released stage-1 run (this is the aligned config)

```bash
cd "$ADT_ROOT/AUGUstus_GRPO"
source env.sh && _activate_augustus_env

# single node, all detected GPUs (the released run: 1 node x 8 GPUs):
bash recipe/transfusion/scripts/run_grpo_transfusion.sh
```

The launcher:
- sources `env.sh`, activates the env, exports all `TRANSFUSION_*` variables,
- brings up the Ray head (multi-node: submit the SAME script per node —
  `NUM_NODES`/`NODE_RANK`/`MASTER_ADDR` are auto-detected from the cluster launcher/SLURM
  env, mirroring the SFT `run_deepspeed` launcher contract),
- snapshots the launched configs into
  `<output>/launch_snapshot/<ts>/` and mirrors the full console log to
  `<output>/logs/run_<ts>.log`,
- runs `python -m recipe.transfusion.train --config-dir
  recipe/transfusion/config --config-name <CONFIG_NAME>` with
  `trainer.nnodes=N trainer.n_gpus_per_node=G` auto-injected.

Useful environment overrides (all optional):

```bash
CONFIG_NAME=stage1 \
EXP_NAME=my-smoke \
bash recipe/transfusion/scripts/run_grpo_transfusion.sh
```

| Env var | Meaning |
|---------|---------|
| `CONFIG_NAME` | recipe yaml under `recipe/transfusion/config/` (default: the released stage-1 config) |
| `EXP_NAME` | overrides `trainer.experiment_name` (and thus the output dir) — use for smoke runs |
| `MODEL_DIR` | SFT run dir (default `$AUGUSTUS_SFT_RUN`) |
| `RL_CKPT` | RL warm-start checkpoint (stage-2 usage; default = SFT best) |
| `GPUS_PER_NODE` / `NUM_NODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` | cluster topology (auto-detected) |

### 3.2 4-GPU variant (small batch)

The released runs used 1 node × 8 GPUs (`train_batch_size=32` → 32 rollout
samples/GPU). On a 4-GPU node keep the SAME per-GPU batches by halving the
global batch:

```bash
GPUS_PER_NODE=4 bash recipe/transfusion/scripts/run_grpo_transfusion.sh \
  data.train_batch_size=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
```

- per-GPU rollout batch stays `train_batch_size × n / GPUs = 16×8/4 = 32`
  (identical to the released 8-GPU run);
- `ppo_mini_batch_size = train_batch_size` keeps the update strictly on-policy;
- `ppo_micro_batch_size_per_gpu = 16` = 2 grad-accum chunks/GPU, the released
  chunking;
- GRPO group size `n=8` (the advantage grouping) is unchanged, so the algorithm
  semantics are identical — only 16 prompts are sampled per step instead of 32.

### 3.3 Stage 2 (warm start from a stage-1 best checkpoint)

```bash
CONFIG_NAME=stage2 \
RL_CKPT="$RL_OUTPUT_ROOT/<stage1-exp>/<ts>/best_pdms_reward.ckpt/actor/hf_model" \
  bash recipe/transfusion/scripts/run_grpo_transfusion_stage2.sh
```

(`convert_actor_to_hf.sh` converts verl FSDP shards into that HF dir; the
script's built-in `RL_CKPT` default already points at the RELEASED stage-1
run's best `hf_model`, so stage 2 launches out of the box.)

### 3.4 Direct verl launch (no launcher script)

```bash
python -m recipe.transfusion.train \
  --config-dir recipe/transfusion/config \
  --config-name stage1 \
  trainer.nnodes=1 trainer.n_gpus_per_node=8
```

All `TRANSFUSION_*` / `PYTHONPATH` vars must be exported in the shell (or rely
on the config's `ray_kwargs.ray_init.runtime_env.env_vars` block, which
forwards them to the Ray workers).

## 4. Checkpoints, logging, resume

- Output root: `trainer.default_local_dir` = `$RL_OUTPUT_ROOT/<exp_name>`;
  `recipe/transfusion/ckpt_layout` appends a run timestamp
  (`<exp>/<YYYYMMDD_HHMMSS>/…`), Transfusion-SFT-style:
  `global_step_N/`, `best_pdms_reward.ckpt/`, `swanlab/`, `samples/`,
  `rollout_data/`, `validation_data/`.
- Periodic ckpts carry model + optimizer + extra (`save_optimizer: true`) and
  are verbatim-resumable: set `trainer.resume.enabled: true` +
  `trainer.resume.resume_path: <run-ts-dir>` (or `.../global_step_N`).
- `best_pdms_reward.ckpt/` is updated whenever full-split validation improves
  the mean `val-core/navsim_transfusion/pdms_reward` (best-checkpoint tracking
  via `trainer.best_metrics`).
- Tracking: SwanLab (cloud; `SWANLAB_API_KEY` from `env.sh`) + console. The
  local SwanLab cache is written into the run dir; resuming the same run dir
  reuses the same cloud experiment.

## 5. Smoke test (single GPU, minutes)

```bash
cd "$ADT_ROOT/AUGUstus_GRPO"
source env.sh && _activate_augustus_env

# Pass training overrides on the command line:
python -m recipe.transfusion.train \
  --config-dir recipe/transfusion/config \
  --config-name stage1 \
  trainer.experiment_name=grpo-smoke \
  trainer.total_training_steps=2 trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.val_before_train=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  data.train_batch_size=8 data.dataloader_num_workers=8 \
  data.transfusion.sft_val_batch_size_per_gpu=8
```

## 6. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `no kernel image is available for execution on the device` | torch build predates the GPU architecture — pro card: `constraints_rl.txt` pins the default 2.13+cu130 stack; H800 (sm_90) on driver 550.x: the cu130 wheels can't run (driver too old) — use the cu126 env `augustus_grpo` (see INSTALL.md §5) |
| `CUDA driver version is insufficient for CUDA runtime version` / `libcudart.so.13: cannot open shared object file` | torch cu130 wheels on a driver < 580 (H800 server) — activate the cu126 env (`conda activate augustus_grpo`, `/opt/augustus_grpo/env`), never upgrade torch there (INSTALL.md §5 pitfall) |
| `Could not locate the 'model' package` | `TRANSFUSION_ROOT` wrong / sibling SFT repo missing — `source env.sh` |
| `recipe.transfusion` not importable in Ray workers | `PYTHONPATH` not forwarded — keep `ray_kwargs...env_vars.PYTHONPATH` (config) or `VERL_USE_EXTERNAL_MODULES=recipe.transfusion.registrations` |
| PDMS always 0 / missing | `metric_cache_train`/`_test` missing or empty — DATA_PREPARATION.md §5 |
| Ray `Duplicate GPU detected` | stale raylet from an earlier job — `ray stop -f` on every node, `rm -rf /dev/shm/ray_*` (the launcher does this) |
| Actor-update OOM | lower `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` (floor 1), then `param_offload`/`optimizer_offload` |
| Rollout-gen OOM | lower `actor_rollout_ref.rollout.rl_gen_micro_batch_size` or `data.train_batch_size` × `rollout.n` |
| SwanLab upload fails | `SWANLAB_API_KEY` unset/invalid — export it or log offline |
