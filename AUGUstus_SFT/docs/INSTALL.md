# AUGUstus_SFT — Installation Guide (from scratch)

This guide builds everything from zero on a fresh Linux server with NVIDIA GPUs.
Configure your dataset, annotation, model, and environment paths through
[`env.sh`](../env.sh) and the `AUGUSTUS_*` environment variables.

## 0. Prerequisites

- Linux x86_64 with NVIDIA GPUs and a driver compatible with the selected
  CUDA wheel stack. See the GPU profile section below.
- `wget`, `git`, `bash`, and enough disk space for the environment and data.
- Raw NAVSIM data and CoT annotations; see
  [`DATA_PREPARATION.md`](DATA_PREPARATION.md).

## 1. Get the repository

```bash
git clone https://github.com/haha-yuki-haha/AUGUstus.git /path/to/AUGUstus-code
cd /path/to/AUGUstus-code/AUGUstus_SFT
```

## 2. Configure the root path (the ONLY edit)

```bash
# env.sh — edit the ADT_ROOT default to your project root
export ADT_ROOT="/path/to/AUGUstus-code"
```

Expected layout (created by you / the data pipeline):

```
$ADT_ROOT/
├── AUGUstus_SFT/          this repo (third_party/ ships inside)
├── Data/
│   ├── debug/                  data pipeline output (created automatically)
│   ├── data_annotation/        CoT annotations (transfusion_outputs_{train,test})
│   └── navsim/                 RAW NAVSIM dataset (symlink or real dir)
│       ├── navsim_logs/{trainval,test}
│       ├── sensor_blobs/{trainval,test}
│       └── maps/
└── Model/
    ├── Qwen3.5-2B/             backbone weights
    └── <run_name>/             training outputs (created by training)
```

## 3. One-command environment install

```bash
bash scripts/setup_env.sh
```

This script:

1. Downloads Miniconda to `/opt/augustus_sft/miniconda3` (bootstrapper only; an
   existing `/opt/miniconda3` is reused when present).
2. Creates the Python 3.12 env at `/opt/augustus_sft/env` (override with
   `AUGUSTUS_ENV_ROOT=... bash scripts/setup_env.sh`) and links it as the named
   environment `augustus_sft` (`conda activate augustus_sft`).
3. `pip install -r requirements.txt` — pinned to the pinned training stack
   (torch 2.13.0 + CUDA 13.0 wheels — the pinned stack, transformers 5.9.0
   with native Qwen3.5 support, flash-linear-attention
   for the GatedDeltaNet kernels, causal-conv1d compiled from source, ...).
4. Installs the NAVSIM + nuplan-devkit sources shipped in `third_party/`
   as editable packages (`pip install -e --no-deps`).

**On an H800 machine (driver 550.x)** run the same script with the `h800`
profile — it installs the cu12.6 stack from `requirements-h800.txt` instead
(torch 2.7.0 + cu126, transformers 5.9.0, flash-linear-attention 0.5.1,
causal-conv1d 1.6.2.post1 from a prebuilt wheel, deepspeed 0.18.9 — the stack
the released model was trained with on H800):

```bash
AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh     # H800 / driver 550.x
AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh --check
```

Everything else (env location, named env, activation) is identical. See
[§6 GPU-card-specific environments](#6-gpu-card-specific-environments-pro-card-vs-h800)
for why the wheel stack must differ.

> **Why ship the devkits?** The repo carries the exact navsim 1.1.0 /
> nuplan-devkit 1.2.0 copies the model was trained and evaluated with. The
> navsim copy contains local modifications (e.g. `PDMScorerConfig` used by
> `model/utils.py::compute_pdm_score_single`); the upstream devkits would not
> work unmodified. Both are Apache-2.0 licensed.
> If you prefer, you can `git clone` the official repos instead, but you must
> re-apply the modifications.

Verify (optional):

```bash
bash scripts/setup_env.sh --check
```

> **GPU note:** the default install targets the **pro-card / cu130 stack**
> (driver ≥ 580). On the **H800 GPU with driver 550.x** (driver 550.x, CUDA 12.4) those
> wheels cannot run — use the `h800` profile there
> (`AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh`), see
> [§6 GPU-card-specific environments](#6-gpu-card-specific-environments-pro-card-vs-h800).

## 4. Activate the environment

`scripts/setup_env.sh` installs the env at `$AUGUSTUS_ENV_ROOT` (default
`/opt/augustus_sft/env`, a conda PREFIX env on the fast local disk) and links
it into the conda installation's `envs/` dir under the name **`augustus_sft`**,
so any of these works after setup:

```bash
# (a) by NAME — shown by `conda env list` right after setup_env.sh:
conda activate augustus_sft

# (b) by PATH (no name lookup needed):
conda activate /opt/augustus_sft/env

# (c) the repo's own helper — also exports every NAVSIM_* / NUPLAN_* path
#     variable the pipeline needs (recommended before training / data runs):
source env.sh && _activate_augustus_env
```

`conda env list` should show:

```
augustus_sft             /opt/augustus_sft/env
```

Verify:

```bash
conda activate augustus_sft
python -c "import torch, transformers, navsim, nuplan; print('env OK')"
```

All launch scripts (`scripts/*.sh`, `inference/*.sh`, `data_process/data_process.sh`,
`NAVSIM_Cache.sh`) source `env.sh` and activate the env automatically.

## 5. Backbone weights

Download the Qwen3.5-2B model (HuggingFace) to `$ADT_ROOT/Model/Qwen3.5-2B`
(or point `QWEN35_2B_PATH` in `env.sh` at any local copy):

```bash
# env.sh already sets: QWEN35_2B_PATH=$AUGUSTUS_MODEL_ROOT/Qwen3.5-2B
huggingface-cli download <Qwen3.5-2B-repo-id> --local-dir "$ADT_ROOT/Model/Qwen3.5-2B"
```

## 6. GPU-card-specific environments (pro card vs H800)

The pip wheels' CUDA major version must be supported by the machine's
**NVIDIA driver**; the profiles use the following stacks:

| | Default profile | h800 profile |
|---|---|---|
| GPU | e.g. RTX Pro 6000 (sm_120) | NVIDIA H800 (sm_90) |
| Driver / CUDA | ≥ 580 (CUDA-13 capable) | 550.x (CUDA 12.4) |
| Wheel stack | **torch 2.13.0 + cu130**, triton 3.7.1 | **torch 2.7.0 + cu126**, triton 3.3.0 |
| Requirements file | `requirements.txt` | `requirements-h800.txt` |
| Install command | `bash scripts/setup_env.sh` | `AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh` |
| Env for this repo | `/opt/augustus_sft/env`, name `augustus_sft` | `/opt/augustus_sft/env`, name `augustus_sft` |
| Activate | `conda activate augustus_sft` | `conda activate augustus_sft` |

- **pro card:** the default `bash scripts/setup_env.sh` installs the pinned
  cu130 stack — nothing to change.
- **H800:** the cu130 wheels CANNOT run there. The CUDA-13 runtime needs
  driver ≥ 580; on driver 550.x it fails with
  `CUDA driver version is insufficient for CUDA runtime version` or
  `libcudart.so.13: cannot open shared object file`. Note H800 (sm_90) itself
  is supported by BOTH stacks — the **driver is the binding constraint**, not
  the GPU. Use the `h800` profile:

  ```bash
  # fresh install of the cu126 stack (torch 2.7.0) from requirements-h800.txt:
  AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh
  # verify (expect torch 2.7.0+cu126, cuda True):
  AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh --check
  /opt/augustus_sft/env/bin/python -c \
      "import torch, transformers, navsim, nuplan, causal_conv1d; \
       print(torch.__version__, torch.cuda.is_available())"
  ```

  Compared with the pro-card install (§3), the `h800` profile differs only in:

  1. **Requirements file** — `requirements-h800.txt` pins the cu126 wheel
     stack (torch 2.7.0 / torchvision 0.22.0 / triton 3.3.0, deepspeed 0.18.9,
     safetensors 0.8.0rc0, scikit-learn 1.2.2, rasterio 1.4.4).
  2. **causal-conv1d** — no source build: the upstream v1.6.2.post1 release
     ships a prebuilt cu12/torch2.7/cp312 wheel that matches the official
     torch 2.7 wheels; `setup_env.sh` downloads and installs it
     (needs github.com access; on failure it warns and the model falls back
     to the slower torch path — see the script's message).

  Everything else (env location, named env `augustus_sft`, editable devkits,
  verification) is identical to §3/§4.

  Dependency differences between the supplied profiles:

  | package | pro-card pin (`requirements.txt`) | H800 (`requirements-h800.txt`) |
  |---|---|---|
  | torch / torchvision | 2.13.0 / 0.28.0 (cu130) | 2.7.0 / 0.22.0 (cu126) |
  | triton | 3.7.1 | 3.3.0 |
  | deepspeed | 0.19.0 | 0.18.9 |
  | safetensors | 0.8.0 | 0.8.0rc0 |
  | scikit-learn | 1.3.2 | 1.2.2 |
  | rasterio | 1.3.11 | 1.4.4 |

## 7. Experiment tracking token

Set `AUGUSTUS_SWANLAB_TOKEN` to your own SwanLab API key when using
`SWANLAB_MODE=cloud`. Leave the token empty and use `SWANLAB_MODE=offline`
for offline logging.

## 8. Installation notes

The setup script clears pip index and constraint overrides before installing
the pinned dependencies. Choose the GPU profile that matches your driver.
The `h800` profile builds scikit-learn from source and requires a C toolchain.
`flash-attn` is optional; the model supports SDPA attention without it.

```bash
pip install flash-attn --no-build-isolation
```

## 9. Next steps

- Data preparation from the raw NAVSIM dataset: [`DATA_PREPARATION.md`](DATA_PREPARATION.md)
- Training / evaluation: [`README.md`](../README.md)
