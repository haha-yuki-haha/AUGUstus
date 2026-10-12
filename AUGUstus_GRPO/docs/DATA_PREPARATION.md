# AUGUstus_GRPO — Data Preparation Guide (from raw NAVSIM data)

Builds the training / validation HF-cache datasets **from scratch**, starting
only from the raw NAVSIM dataset. All outputs go to
`$AUGUSTUS_DATA_ROOT` (default `$ADT_ROOT/Data/debug`).

> **RL reuses the SFT data.** The RL stage consumes **exactly** the same
> `hf_cache_{train,test}_CoT_Solution` caches the supervised stage
> ([AUGUstus_SFT](../../AUGUstus_SFT)) produced. If you already ran the SFT
> data pipeline, skip to [§5](#5-rl-only-data—the-pdms-metric-caches) — the
> only RL-specific artifacts are the **train-split PDMS metric cache**
> (required) and the **test-split cache** (validation reward).

## 0. Activate the environment

```bash
cd "$ADT_ROOT/AUGUstus_GRPO"
source env.sh && _activate_augustus_env
```

## 1. Raw inputs

### 1.1 Downloading the raw NAVSIM dataset (from scratch)

Official download scripts are shipped with the navsim devkit
(`$AUGUSTUS_SFT_ROOT/third_party/navsim/download/`): `download_trainval.sh`,
`download_test.sh`, `download_navtrain.sh` (sensor blobs), `download_maps.sh`
(nuPlan maps). Run them inside a scratch dir, then arrange into the layout
below:

```bash
mkdir -p "$NAVSIM_ROOT" && cd "$NAVSIM_ROOT"
bash "$NAVSIM_DEVKIT_ROOT/download/download_trainval.sh"   # navsim_logs/trainval (meta_datas)
bash "$NAVSIM_DEVKIT_ROOT/download/download_navtrain.sh"   # sensor_blobs/trainval
bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh"       # navsim_logs/test + sensor_blobs/test
bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh"       # maps/
```

Resulting layout consumed by the pipeline:

```
$NAVSIM_ROOT/
├── navsim_logs/{trainval,test}/   # log pickles
├── sensor_blobs/{trainval,test}/  # camera images
└── maps/                          # nuplan-maps-v1.0
```

### 1.2 What the pipeline consumes

| What | Where (`env.sh` variable) | Notes |
|------|---------------------------|-------|
| NAVSIM log pickles | `$NAVSIM_ROOT/navsim_logs/{trainval,test}` | OpenScene / NAVSIM "navsim logs" download |
| Camera images | `$NAVSIM_ROOT/sensor_blobs/{trainval,test}` | OpenScene / NAVSIM "sensor blobs" download |
| NAVSIM maps | `$NAVSIM_ROOT/maps` | nuPlan maps v1.0 (needed for PDMS metric caching) |
| CoT annotations | `$AUGUSTUS_ANNOTATION_ROOT/transfusion_outputs_{train,test}/<scene>/<frame>/reasoning_trace.json` | reasoning traces used for the CoT LM loss |
| GT trajectories + bbox solutions | `$AUGUSTUS_SFT_ROOT/data_process/AutoDrive_P3/1_RL_{train,test}_2048_512.json` | Download both files as described below. Format: ego-local `[x, y, heading_rad]`, 4 history points and 8 future points, plus perception/prediction/planning CoT solutions |

The `navtrain` / `navtest` splits are defined by
`$AUGUSTUS_SFT_ROOT/data_process/navtrain.yaml` / `navtest.yaml` (official
NAVSIM log/token lists).

### 1.3 Download AutoDrive-P3 annotations

Download both annotation files from the [Hugging Face Data directory](https://huggingface.co/yuki-hahaha/AutoDrive-P3_with_Run-then-walk/tree/main/Data) and save them in `${AUGUSTUS_SFT_ROOT}/data_process/AutoDrive_P3`:

- [1_RL_test_2048_512.json](https://huggingface.co/yuki-hahaha/AutoDrive-P3_with_Run-then-walk/resolve/main/Data/1_RL_test_2048_512.json?download=true)
- [1_RL_train_2048_512.json](https://huggingface.co/yuki-hahaha/AutoDrive-P3_with_Run-then-walk/resolve/main/Data/1_RL_train_2048_512.json?download=true)

```bash
wget -c -O "${AUGUSTUS_SFT_ROOT}/data_process/AutoDrive_P3/1_RL_test_2048_512.json" \
  "https://huggingface.co/yuki-hahaha/AutoDrive-P3_with_Run-then-walk/resolve/main/Data/1_RL_test_2048_512.json?download=true"
wget -c -O "${AUGUSTUS_SFT_ROOT}/data_process/AutoDrive_P3/1_RL_train_2048_512.json" \
  "https://huggingface.co/yuki-hahaha/AutoDrive-P3_with_Run-then-walk/resolve/main/Data/1_RL_train_2048_512.json?download=true"
```

These files provide ground-truth trajectories and bounding-box solutions for
pipeline steps `3_0_get_traj_heading.py` and `4_3_add_solution.py`.

## 2. Run the pipeline (one command)

```bash
bash "$AUGUSTUS_SFT_ROOT/data_process/data_process.sh"
```

Stages (each script also takes `--item train|test` individually):

```
0_get_useful_scenes.py    filter scenes per split yaml -> 0_get_useful_{item}_scenes.pkl
1_get_json.py             per-frame JSONs (tokens, ego status, annotations) -> 1_get_json_{item}/
2_0_get_all_frames.py     stitch CAM_L0|CAM_F0|CAM_R0 -> 1024x4096, resize 2048x512
2_1_get_all_videos.py     encode 4 history frames @ 2 Hz -> mp4 clips
3_0_get_traj_heading.py   GT trajectory (from your AutoDrive_P3 JSON, see §1.2)
3_1_get_v.py              append ego velocity (vx, vy) per waypoint
4_0_merge_json.py         merge into 4_data_2048_512/0_{item}.json
4_1_process_json.py       clean/format fields -> 1_{item}_processed.json
4_2_merge_cot.py          attach reasoning_trace -> 2_cot_{item}.json
4_3_add_solution.py       attach bbox solutions -> 3_cot_solution_{item}.json
4_4_build_hf_cache.py     write Arrow HF caches -> hf_cache_{item}_CoT_Solution
```

Intermediate + final artifacts (all under `$AUGUSTUS_DATA_ROOT`):

```
$AUGUSTUS_DATA_ROOT/
├── 0_get_useful_{train,test}_scenes.pkl
├── 1_get_json_{train,test}/
├── 2_0_get_all_frames_{train,test}_2048_512/
├── 2_1_get_all_videos_{train,test}_2048_512/
├── 3_0_get_traj_heading_{train,test}_2048_512/
├── 3_1_get_v_{train,test}_2048_512/
└── 4_data_2048_512/
    ├── 0_{item}.json / 1_{item}_processed.json / 2_cot_{item}.json / 3_cot_solution_{item}.json
    ├── action_norm_stats.json          # written by training (q01/q99 per dim)
    └── hf_cache_{train,test}_CoT_Solution/   # ← the RL train_files / val_files
```

Expected sample counts: **train 103 288**, **test 12 146** (before the
no-CoT filter in step 4_4; the final caches hold 103 287 / 12 146 rows).

## 3. Action normalization stats

SFT training computes q01/q99 stats on first use
(`model/utils.py::load_or_compute_action_norm_stats`) and writes them to
`normalization.stats_file` (`${AUGUSTUS_DATA_ROOT}/4_data_2048_512/action_norm_stats.json`).
Reference values (navtrain, 2048_512):

```
q01 = [-0.02, -12.03, -1.30]     q99 = [50.02, 16.03, 1.31]
```

**The RL stage does NOT read this file for its VALUES**: the recipe ships a
bundled copy (`recipe/transfusion/config/action_norm_stats.json`) and the
checkpoint itself carries the real `action_q01/q99` — the stats-file env var
(`TRANSFUSION_STATS_FILE`) only needs to point at an ACCESSIBLE per-timestep
stats JSON so `build_model` does not try to recompute from the train cache.

## 4. Sanity checks (HF caches)

```bash
# sample counts per cache
python - <<'EOF'
from datasets import load_from_disk
import glob, os
root = os.environ["AUGUSTUS_DATA_ROOT"] + "/4_data_2048_512"
for split in ["train", "test"]:
    d = f"{root}/hf_cache_{split}_CoT_Solution"
    n = sum(load_from_disk(b).num_rows for b in sorted(glob.glob(d + "/batch_*")))
    print(split, n, "samples")
EOF

# decode one video + check trajectory shape
python - <<'EOF'
import glob, io, os, json, av, numpy as np
from datasets import load_from_disk
d = sorted(glob.glob(os.environ["AUGUSTUS_DATA_ROOT"] + "/4_data_2048_512/hf_cache_test_CoT_Solution/batch_*"))[0]
ds = load_from_disk(d); row = ds[0]
containers = av.open(io.BytesIO(row["video"]))
# NOTE: use to_image() (the same path training uses), NOT to_ndarray(fmt=...)
# — some PyAV builds mis-route the `fmt` kwarg to reformat().
frames = [np.asarray(f.to_image()) for f in containers.decode(video=0)]
print("frames:", len(frames), "shape:", frames[0].shape)
print("future traj:", json.loads(row["future_x_y_angle"]))
EOF
```

Expected: 4 frames of shape **(512, 2048, 3)** (H×W×C), 8 future
waypoints `[x, y, heading]`.

## 5. RL-only data — the PDMS metric caches

The GRPO reward is dominated by the **closed-loop NAVSIM PDM Score**, computed
per ROLLOUT sample — for the TRAIN split during training, not just at
validation. So the RL stage needs **both** metric caches (SFT needed only the
test one):

| Cache | Split | RL role | SFT role |
|-------|-------|---------|----------|
| `metric_cache_train` | navtrain (trainval logs) | **training reward** (PDMS + TTC/DAC safety terms, per train sample) | optional (train PDMS stats) |
| `metric_cache_test` | navtest | **validation reward** (val PDMS metric + best-checkpoint selection by `pdms_reward`) | eval PDMS |

Both are keyed by scene token and referenced per sample via
`data.metric_cache_train` / `data.metric_cache_test` (the recipe config points
them at `${NAVSIM_EXP_ROOT}/metric_cache_{train,test}`, i.e.
`$AUGUSTUS_DATA_ROOT/navsim_exp/metric_cache_{train,test}` by default). A
missing split cache degrades gracefully — that split simply gets no PDMS
reward — but the released runs need both.

### 5.1 Generate them (the ONLY source of these caches)

Both caches are generated from the **raw NAVSIM logs + maps** (NOT the sensor
blobs — metric caching does not read images) by the devkit's
`run_metric_caching.py`, wrapped in the SFT repo's `NAVSIM_Cache.sh`:

```bash
# ---- navtest -> $NAVSIM_EXP_ROOT/metric_cache_test  (validation reward) ----
AUGUSTUS_ENV_ROOT="$AUGUSTUS_ENV_ROOT" \
  bash "$AUGUSTUS_SFT_ROOT/NAVSIM_Cache.sh"

# ---- navtrain -> $NAVSIM_EXP_ROOT/metric_cache_train  (TRAINING reward) ----
# IMPORTANT: the TRAIN cache must be the **navtrain** benchmark split — its
# scene_filter carries the explicit ~104k navtrain token list with
# frame_interval: 1, which is exactly the frame structure of the training HF
# cache. Do NOT build it from `train_test_split=trainval` (all_scenes /
# frame_interval: null): different frame windows per scene token change the
# cached MetricCache and thus the PDMS of the same trajectory, and most
# navtrain tokens are missing entirely .
AUGUSTUS_ENV_ROOT="$AUGUSTUS_ENV_ROOT" TRAIN_TEST_SPLIT=navtrain \
  bash "$AUGUSTUS_SFT_ROOT/NAVSIM_Cache.sh"
```

What the wrapper actually runs (so you can call the devkit directly if you
prefer) — `navsim/planning/script/run_metric_caching.py` is Hydra-based:

```bash
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtest  cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache_test"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtrain cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache_train"
```

Details that matter:

- **Environment**: `NAVSIM_Cache.sh` sources `$AUGUSTUS_SFT_ROOT/env.sh` and
  activates `$AUGUSTUS_ENV_ROOT` — when running it from the GRPO checkout,
  `source "$ADT_ROOT/AUGUstus_GRPO/env.sh"` first (or export
  `AUGUSTUS_ENV_ROOT=/opt/augustus_grpo/env`) so the cache is built with the
  SAME env that trains (navsim/nuplan come from `$AUGUSTUS_SFT_ROOT/third_party/`
  in both cases).
- **Raw-data prerequisite**: `NUPLAN_MAPS_ROOT` (`$NAVSIM_ROOT/maps`) and the
  navsim logs (`$NAVSIM_ROOT/navsim_logs/{trainval,test}`) must exist; the
  build does not touch `sensor_blobs/`.
- **Parallelism**: the default worker is `ray_distributed_no_torch`
  (`use_distributed: false` → local Ray, all cores).
- **Output layout** (per log dir → per scene token):
  `<cache>/<log_name>/unknown/<scene_token>/metric_cache.pkl` — xz-compressed
  pickles of `navsim.planning.metric_caching.metric_cache.MetricCache`.

Build the caches with the same vendored NAVSIM version and split used for
training.
