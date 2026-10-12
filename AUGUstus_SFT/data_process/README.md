# AUGUstus_SFT — NAVSIM Data Process

Pipeline that builds the training/val HF-cache datasets consumed by
`configs/navsim_qwen3p5.yaml` (`data.data_format: "hf_cache"`, 2048x512, Qwen3.5-VL).

All inputs/outputs are resolved from the environment defined by `env.sh`
(repo root) — see `docs/DATA_PREPARATION.md` for the full walkthrough.

```
RAW INPUTS (env.sh):
  $NAVSIM_ROOT/navsim_logs/{trainval,test}      NAVSIM log pickles
  $NAVSIM_ROOT/sensor_blobs/{trainval,test}     camera images
  $AUGUSTUS_ANNOTATION_ROOT/                    CoT reasoning_trace annotations
      transfusion_outputs_{train,test}/
  AutoDrive_P3/1_RL_{item}_2048_512.json        annotation inputs:
                                                GT trajectories + bbox solutions

PIPELINE (this directory):
  0_get_useful_scenes.py     scene filtering (navtrain.yaml / navtest.yaml)
                             -> $AUGUSTUS_DATA_ROOT/0_get_useful_{item}_scenes.pkl
  1_get_json.py              per-frame JSONs -> 1_get_json_{item}/
  2_0_get_all_frames.py      3-camera stitch + resize to 2048x512 -> 2_0_get_all_frames_{item}_2048_512/
  2_1_get_all_videos.py      4-frame history clips @ 2 Hz (mp4) -> 2_1_get_all_videos_{item}_2048_512/
  3_0_get_traj_heading.py    GT trajectory from the AutoDrive_P3 JSON -> 3_0_get_traj_heading_{item}_2048_512/
  3_1_get_v.py               GT historical velocity -> 3_1_get_v_{item}_2048_512/
  4_0_merge_json.py          -> 4_data_2048_512/0_{item}.json
  4_1_process_json.py        -> 4_data_2048_512/1_{item}_processed.json
  4_2_merge_cot.py           -> 4_data_2048_512/2_cot_{item}.json      (reasoning_trace)
  4_3_add_solution.py        -> 4_data_2048_512/3_cot_solution_{item}.json (bbox CoT)
  4_4_build_hf_cache.py      -> 4_data_2048_512/hf_cache_{train,test}_CoT_Solution

Run everything:
  bash data_process.sh
(or run the scripts individually; each takes --item train|test)
```

## Notes

- **GT trajectories** come from `AutoDrive_P3/1_RL_{item}_{x}_{y}.json`
  (step `3_0`): `historical_trajectory` (4 pts) + `solution.planning`
  (8 future pts), ego-local `[x, y, heading_rad]` at 0.5 s spacing. No external
  trajectory files are needed.
- **PDMS metric caches** (`data.metric_cache_train/test`) are built separately
  by `NAVSIM_Cache.sh` at the repo root.
- Final dataset locations consumed by `configs/navsim_qwen3p5.yaml`:
  - `train_data_path`: `${AUGUSTUS_DATA_ROOT}/4_data_2048_512/hf_cache_train_CoT_Solution`
  - `val_data_path`:   `${AUGUSTUS_DATA_ROOT}/4_data_2048_512/hf_cache_test_CoT_Solution`

## AutoDrive_P3/

`convert_bbox.py` / `check_bbox_visualization.py` — convert the AutoDrive_P3
bbox-solution JSONs between resolutions (e.g. 672x168 → 2048x512) and
visualize them. The 2048x512 conversions (`1_RL_{item}_2048_512.json`) are
the inputs to steps `3_0` and `4_3`; see the [data preparation guide](../docs/DATA_PREPARATION.md) for
download instructions. Re-run the converter only if you need a different resolution.

## Dataset columns (hf_cache_{item}_CoT_Solution)

`video` (mp4 bytes), `frame_id`, `scene_token`, `driving_command`,
`ego_dynamic_state`, `historical_x_y_angle`, `historical_vx_vy`,
`future_x_y_angle`, `solution_perception`, `solution_prediction`,
`solution_planning`, `reasoning_trace`. The released config uses
`cot.cot_components: [prediction]` → the CoT is built from
`solution_prediction` (`model/dataset.py::TransfusionHFCacheDataset`).
