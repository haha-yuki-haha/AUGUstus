# AUGUstus_GRPO — Installation Guide (from scratch)

GRPO post-training (RL) of the **AUGUstus** trajectory planner, built on
[verl](https://github.com/volcengine/verl) (vendored under `verl/`). The policy
is the Qwen3.5-2B Transfusion **SFT** model released in the sibling repo
**AUGUstus_SFT** — this repo only adds the verl RL stack on top of it.

This guide builds everything from zero on a fresh Linux server with NVIDIA GPUs.
Configure your dataset, annotation, model, and environment paths through
[`env.sh`](../env.sh) and the `AUGUSTUS_*` environment variables.

## 0. Prerequisites

- Linux x86_64 with NVIDIA GPUs and a driver compatible with the selected
  CUDA wheel stack. See the GPU profile section below.
- `wget`, `git`, `bash`, and enough disk space for the environment and data.
- Raw NAVSIM data and CoT annotations; see
  [`DATA_PREPARATION.md`](DATA_PREPARATION.md).
- The sibling `AUGUstus_SFT` folder and an SFT checkpoint for RL training.

## 1. Get the repositories

```bash
git clone https://github.com/haha-yuki-haha/AUGUstus.git /path/to/AUGUstus-code
cd /path/to/AUGUstus-code/AUGUstus_GRPO
```

## 2. Configure the root path (the ONLY edit)

```bash
# env.sh — edit the ADT_ROOT default to your project root
export ADT_ROOT="/path/to/AUGUstus-code"
```

`env.sh` derives everything else (both repo roots, the env location, the data
and model roots, the `TRANSFUSION_*` variables consumed by verl workers, the
NAVSIM/nuPlan devkit variables). See its header for the full expected layout.

## 3. One-command environment install

```bash
bash scripts/setup_env.sh
```

This script performs **two stages** in one go:

1. **Stage A — the AUGUstus_SFT base env** (identical to
   `AUGUstus_SFT/scripts/setup_env.sh`):
   - Miniconda at `/opt/augustus_grpo/miniconda3` (bootstrapper only; an
     existing `/opt/miniconda3` is reused when present).
   - Creates the Python 3.12 env at `/opt/augustus_grpo/env` (override with
     `AUGUSTUS_ENV_ROOT=... bash scripts/setup_env.sh`) and links it as the
     NAMED environment `augustus_grpo` (`conda activate augustus_grpo`).
   - `pip install -r $AUGUSTUS_SFT_ROOT/requirements.txt` installs the pinned
     training stack (torch 2.13.0 + CUDA 13.0 wheels, transformers 5.9.0
     with native Qwen3.5 support, flash-linear-attention for the GatedDeltaNet kernels,
     causal-conv1d compiled from source, ...).
   - Installs the NAVSIM + nuplan-devkit sources shipped in the SFT repo's
     `third_party/` as editable packages (`pip install -e --no-deps`).
   - Builds `causal-conv1d 1.6.2.post1` from source against the pip-installed
     nvcc toolchain (no prebuilt wheel exists for the CUDA-13 wheel stack; the
     script applies the nvcc/ptxas compatibility fixes — see
     `AUGUstus_SFT/scripts/setup_env.sh` for the build steps).
2. **Stage B — the verl RL dependencies** (the delta added by this repo):
   - `pip install -c recipe/transfusion/constraints_rl.txt -r
     recipe/transfusion/requirements_rl.txt` — `tensordict 0.10.0`,
     `ray 2.55.1`, `pyarrow 24.0.0`, `codetiming`, `torchdata`, `dill`,
     `math-verify`, `latex2sympy2_extended`, `pylatexenc`, `uvicorn`,
     `fastapi`, `pyvers`, `orjson`, `TransferQueue 0.1.7`, **with the
     constraints file protecting** `torch / transformers / numpy / protobuf /
     tensordict / nccl / ray / pyarrow` from being upgraded (the version
     rationale is in the requirements header).
   - `pip install -e . --no-deps` (this repo — the vendored `verl/` framework +
     the `recipe.transfusion` package).

**On an H800 machine (driver 550.x)** run the same script with the `h800`
profile — stage A then installs the SFT base from `requirements-h800.txt`
(torch 2.7.0 + cu126, transformers 5.9.0, flash-linear-attention 0.5.1,
causal-conv1d 1.6.2.post1 from a prebuilt wheel, deepspeed 0.18.9) and stage B
installs the RL deps under `constraints_rl_h800.txt`, with TransferQueue
installed `--no-deps` first (the pitfall below is handled by the script):

```bash
AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh     # H800 / driver 550.x
AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh --check
```

See [§5 GPU stack notes](#5-gpu-stack-notes-pro-card-vs-h800) for why the
wheel stack must differ.

> **Why ship the devkits?** The SFT repo carries the exact navsim 1.1.0 /
> nuplan-devkit 1.2.0 copies the model was trained and evaluated with. The
> navsim copy contains local modifications (e.g. `PDMScorerConfig` used by
> `model/utils.py::compute_pdm_score_single`); upstream devkits would not work
> unmodified. Both are Apache-2.0 licensed.

Verify (optional):

```bash
bash scripts/setup_env.sh --check
```

> **GPU note:** the install above targets the **pro-card / cu130 stack**
> (torch 2.13.0 wheels, driver ≥ 580). On the **H800 GPU with driver 550.x** (driver
> 550.x, CUDA 12.4) those wheels cannot run — the env there must be
> based on the cu126 stack; use the h800 profile there
> (`AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh`) — see §4 and §5.

## 4. Activate the environment

`scripts/setup_env.sh` installs the env at `$AUGUSTUS_ENV_ROOT` (default
`/opt/augustus_grpo/env`, a conda PREFIX env on the fast local disk) and links
it into the conda installation's `envs/` dir under the name **`augustus_grpo`**,
so any of these works after setup:

```bash
# (a) by NAME — shown by `conda env list` right after setup_env.sh:
conda activate augustus_grpo

# (b) by PATH (no name lookup needed):
conda activate /opt/augustus_grpo/env

# (c) the repo's own helper — also exports every TRANSFUSION_* / NAVSIM_* /
#     NUPLAN_* path variable the recipe needs (recommended before training):
source env.sh && _activate_augustus_env
```

`conda env list` should show:

```
augustus_grpo            /opt/augustus_grpo/env
```

Verify:

```bash
conda activate augustus_grpo
python -c "import torch, transformers, navsim, nuplan, verl, ray; print('env OK')"
```

The launch scripts (`recipe/transfusion/scripts/*.sh`) source `env.sh` and
activate the env automatically.

## 5. GPU stack notes (pro card vs H800)

The supplied dependency profiles use different CUDA wheel stacks. Select
the profile that matches your NVIDIA driver; `constraints_rl.txt` and
`constraints_rl_h800.txt` preserve the matching dependency versions.

### Per-card wheel stacks

The pip wheels' CUDA major version must be supported by the machine's
**NVIDIA driver**; the profiles use the following stacks:

| | Default profile | h800 profile |
|---|---|---|
| GPU | e.g. RTX Pro 6000 (sm_120) | NVIDIA H800 (sm_90) |
| Driver / CUDA | ≥ 580 (CUDA-13 capable) | 550.x (CUDA 12.4) |
| Wheel stack | **torch 2.13.0 + cu130**, triton 3.7.1 | **torch 2.7.0 + cu126**, triton 3.3.0 |
| SFT base file | `AUGUstus_SFT/requirements.txt` | `AUGUstus_SFT/requirements-h800.txt` |
| RL constraints file | `constraints_rl.txt` | `constraints_rl_h800.txt` |
| Install command | `bash scripts/setup_env.sh` | `AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh` |
| Env for this repo | `/opt/augustus_grpo/env`, name `augustus_grpo` | `/opt/augustus_grpo/env`, name `augustus_grpo` |
| Activate | `conda activate augustus_grpo` | `conda activate augustus_grpo` |

- **pro card:** default `bash scripts/setup_env.sh` installs the pinned
  cu130 stack — nothing to change.
- **H800:** the cu130 wheels CANNOT run there — the CUDA-13 runtime needs
  driver ≥ 580, and on driver 550.x torch dies with `CUDA driver version is
  insufficient for CUDA runtime version` (or `libcudart.so.13: cannot open
  shared object file`). H800 (sm_90) itself is supported by BOTH stacks —
  the **driver is the binding constraint**, not the GPU. Use the `h800` profile:

  ```bash
  # fresh install of the cu126 stack from requirements-h800.txt
  # + RL deps under constraints_rl_h800.txt (both stages, one command):
  AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh
  # verify (expect torch 2.7.0+cu126, cuda True, verl from this repo):
  AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh --check
  /opt/augustus_grpo/env/bin/python -c \
      "import torch, verl, recipe.transfusion, ray, causal_conv1d; \
       print(torch.__version__, torch.cuda.is_available(), verl.__file__)"
  ```

  Compared with the pro-card install (§3), the `h800` profile differs only in:

  1. **Stage A requirements** — `AUGUstus_SFT/requirements-h800.txt` pins the
     cu126 wheel stack (torch 2.7.0 / torchvision 0.22.0 / triton 3.3.0,
     deepspeed 0.18.9, safetensors 0.8.0rc0, scikit-learn 1.2.2,
     rasterio 1.4.4);
     causal-conv1d comes from the prebuilt cu12/torch2.7 wheel (no source
     build; needs github.com access — on failure the script warns and the
     model falls back to the slower torch path).
  2. **Stage B constraints** — `constraints_rl_h800.txt` pins
     `torch==2.7.0 / numpy==1.26.4 / nvidia-nccl-cu12==2.26.2` (instead of the
     cu130 pins) so pip can never upgrade torch into an unrunnable cu130
     build.
  3. **TransferQueue pitfall guard** — the script installs
     `TransferQueue==0.1.7` (+ msgspec/pyzmq/prometheus_client) **before** the
     RL requirements, with `--no-deps` (see the pitfall below).

  Everything else (env location, named env `augustus_grpo`, editable devkits,
  verification) is identical to §3/§4.

  **PITFALL (H800 only): do NOT `pip install TransferQueue` with dependencies.**
  On the pro card the plain `pip install -c constraints_rl.txt -r
  requirements_rl.txt` is safe because torch is already 2.13.0. On the H800
  stack (torch 2.7.0) the same command lets pip "satisfy" the
  `torch==2.13.0` constraint by UPGRADING torch to the cu130 build — which
  then cannot run on the 550 driver and leaves a dead env
  (`libcudart.so.13: cannot open shared object file`). Hence `--no-deps`
  above: verl guards the `import transfer_queue` with a try/except + mock
  (`verl/utils/transferqueue_utils.py`), and the Transfusion recipe uses
  `verl.trainer.main_ppo` (NOT `main_ppo_sync`), so TransferQueue is never on
  the critical training path.

  Dependency differences between the supplied profiles:

  | package | pro-card pin | H800 (`h800` profile) |
  |---|---|---|
  | torch / torchvision | 2.13.0 / 0.28.0 (cu130) | 2.7.0 / 0.22.0 (cu126) |
  | triton | 3.7.1 | 3.3.0 |
  | deepspeed | 0.19.0 | 0.18.9 |
  | safetensors | 0.8.0 | 0.8.0rc0 |
  | scikit-learn | 1.3.2 | 1.2.2 |
  | rasterio | 1.3.11 | 1.4.4 |
  | nvidia-nccl | cu13 2.29.7 | cu12 2.26.2 |

## 6. Backbone weights

Nothing to download here: RL starts from the TRAINED SFT checkpoint
(`$AUGUSTUS_SFT_RUN/best`, set via `TRANSFUSION_CKPT`), which is self-contained
— the backbone architecture + tokenizer + image/video processor all come from
its `best/pytorch_model/` HF dir (`TRANSFUSION_QWEN_MODEL_NAME`), and the model
architecture config from its `config_original.yaml` (`TRANSFUSION_CONFIG`).
The original Qwen3.5-2B backbone is never read.

## 7. Installation notes

The setup script clears pip index and constraint overrides before installing
the pinned dependencies. Choose the GPU profile that matches your driver.
The `h800` profile builds scikit-learn from source and requires a C toolchain.
`flash-attn` is optional; the model supports SDPA attention without it.

```bash
pip install flash-attn --no-build-isolation
```

## 8. Next steps

- Data (HF caches + the RL-only PDMS metric caches): [`DATA_PREPARATION.md`](DATA_PREPARATION.md)
- Launching GRPO + aligning with the released run: [`TRAINING.md`](TRAINING.md)
