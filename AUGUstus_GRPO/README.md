# AUGUstus_GRPO: GRPO Post-Training for the Transfusion Trajectory Planner

GRPO post-training (RL) of the **AUGUstus** trajectory planner — the verl-based
RL repo for the Transfusion diffusion-RL workflow, built on [verl](https://github.com/volcengine/verl)
(its framework components are vendored under `verl/`). The policy is the
Qwen3.5-2B Transfusion SFT model released in **AUGUstus_SFT**; this repo
fine-tunes it on NAVSIM with PDM-score rewards.

## Algorithm — Hybrid GRPO (action + CoT)

The rollout samples the trajectory with a reverse-time SDE over the flow-matching
path (`x_t = (1 - t) * x_0 + t * noise`, lambda_t g(t) schedule) and the CoT
autoregressively; the actor update (`recipe/transfusion/loss.py::hybrid_grpo_loss_from_rollout`)
combines two heads:

- **Action head** — closed-form x0-prediction GRPO: the per-step importance ratio
  and exact Gaussian KL are computed in closed form directly from the two
  policies' x0 predictions (`recipe/transfusion/algorithms/jit_grpo.py`).
- **CoT head** — per-token probability-ratio GRPO with k1/k3 KL estimators
  (`recipe/transfusion/algorithms/cot_grpo.py`).

Per-head advantages (`rl_split_advantage`) weight the two heads from disjoint
reward terms (prediction / planning IoU + PDMS); the reward function lives in
`recipe/transfusion/reward_function.py` (NAVSIM metric-cache based).

The diffusion-specialised algorithms (Flow-GRPO, DiffusionNFT) that existed in
this recipe were removed — the released runs use the hybrid GRPO path only.

## Project Structure

```
AUGUstus_GRPO/
├── verl/                      # verl framework (vendored; FSDP/Ray trainer, workers, utils)
├── recipe/
│   └── transfusion/           # the Transfusion RL recipe (loaded by verl as an external module)
│       ├── rollout.py         # TransfusionRollout (SDE trajectory + CoT sampling)
│       ├── rollout_config.py  # TransfusionRolloutConfig (rl_num_steps / noise knobs)
│       ├── transfusion_policy.py  # policy wrapper (builds the model from AUGUstus_SFT)
│       ├── loss.py            # hybrid_grpo_loss_from_rollout (actor update)
│       ├── sync_driver.py     # DP sync driver (collect -> collate -> update)
│       ├── model_engine.py    # model/optimizer engine glue
│       ├── dataset.py         # TransfusionRLDataset (reuses AUGUstus_SFT's dataset)
│       ├── reward_function.py # NAVSIM reward (PDMS + prediction/planning IoU)
│       ├── transfusion_reward.py  # reward internals (metric-cache scoring)
│       ├── reward_manager.py / validation.py / best_ckpt.py / ckpt_layout.py /
│       │   sample_logging.py / epdms_eval.py / registrations.py / train.py
│       ├── algorithms/        # fm_sde_logprob / jit_grpo / cot_grpo / diffusion_adapter
│       ├── config/            # the two released run configs + action_norm_stats.json
│       ├── scripts/           # launch + HF-conversion scripts
│       ├── requirements_rl.txt / constraints_rl.txt
│       └── tests/             # recipe unit tests
├── pyproject.toml
├── README.md
└── LICENSE                    # Apache-2.0 (verl-derived code)
```

## Configuration

The two released runs (hybrid GRPO, on-policy k3, split advantage) are launched
from the configs:

| Launch config (`recipe/transfusion/config/`) | Run | Base checkpoint |
|--------|-----|-----------------|
| `stage1.yaml` | stage-1 | AUGUstus_SFT best |
| `stage2.yaml` | stage-2 | stage-1 run's best |

**Every machine-specific path lives in ONE place: [`env.sh`](env.sh)** — the
configs read them strictly via `${oc.env:VAR}` (NO inline fallbacks): a missing
variable fails the launch loudly instead of silently resolving to some other
machine's path. `source env.sh` (or your own exports) is therefore REQUIRED
before any launch. The config files themselves contain no absolute paths.

All configs read the SFT model/dataset code from the sibling `AUGUstus_SFT`
repo via `TRANSFUSION_ROOT` / `PYTHONPATH` (see the `ray_kwargs` env_vars
block in each config; machine-specific paths are `${oc.env:...}` interpolations
resolved from `env.sh`) and point `data.train_files` / `data.val_files` at the
`hf_cache_{train,test}_CoT_Solution` caches.

## Launch

```bash
conda activate augustus_grpo        # = `source env.sh && _activate_augustus_env`
bash recipe/transfusion/scripts/run_grpo_transfusion.sh             # stage-1 (from the SFT best)
RL_CKPT=<stage1-best>/actor/hf_model \
  bash recipe/transfusion/scripts/run_grpo_transfusion_stage2.sh    # stage-2 (from the stage-1 best)
```

(The launch scripts source `env.sh` themselves; activating first is only for
interactive inspection. See `docs/INSTALL.md` §4 for all activation options.)

## Installation

```bash
# one command from scratch: AUGUstus_SFT base env + verl RL deps + editable install
bash scripts/setup_env.sh
```

or manually: build the sibling AUGUstus_SFT environment (its
`docs/INSTALL.md`), then

```bash
conda activate augustus_grpo    # or: source env.sh && _activate_augustus_env
pip install -c recipe/transfusion/constraints_rl.txt -r recipe/transfusion/requirements_rl.txt
pip install -e . --no-deps   # from this repo root (installs the vendored verl + recipe)
```

See `recipe/transfusion/requirements_rl.txt` for the dependency pins
(torch 2.13+cu130 — the default GPU profile / transformers
5.9 / protobuf 4.25 / tensordict 0.10 pins; with the h800 GPU profile the cu126
stack is required instead — see `docs/INSTALL.md` §5) and the docs:

- [`docs/INSTALL.md`](docs/INSTALL.md) — environment from scratch (incl. the NAVSIM/nuPlan devkits)
- [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) — HF caches + the RL PDMS metric caches
- [`docs/TRAINING.md`](docs/TRAINING.md) — launching GRPO + aligning with the released run

## Acknowledgments

This repo is built on [verl](https://github.com/volcengine/verl) (volcengine) —
the RL training framework (`verl/` here is a vendored fork of it). Many thanks to
the verl team. The verl-derived portions of this codebase carry the Apache-2.0
license (`LICENSE`).

## License

Apache-2.0 (see `LICENSE`).
