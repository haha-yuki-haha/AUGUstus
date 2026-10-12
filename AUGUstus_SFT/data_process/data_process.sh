#!/bin/bash
# ==============================================================================
# AUGUstus_SFT — NAVSIM data preparation pipeline (2048x512, Qwen3.5-VL)
#
# Builds the training/val HF-cache datasets used by the released config
# (data.train_data_path / data.val_data_path = hf_cache_{train,test}_CoT_Solution):
#
#   0_get_useful_scenes.py     scene filtering (navtrain.yaml / navtest.yaml)
#                              -> 0_get_useful_{train,test}_scenes.pkl
#   1_get_json.py              per-frame JSONs from the scene pickles -> 1_get_json_{item}/
#   2_0_get_all_frames.py      stitch 3 cameras + resize to 2048x512 -> 2_0_get_all_frames_{item}_2048_512/
#   2_1_get_all_videos.py      encode the 4-frame history clips (2 Hz, mp4) -> 2_1_get_all_videos_{item}_2048_512/
#   3_0_get_traj_heading.py    GT trajectory from the shipped AutoDrive_P3 JSON -> 3_0_get_traj_heading_{item}_2048_512/
#   3_1_get_v.py               GT historical velocity -> 3_1_get_v_{item}_2048_512/
#   4_0_merge_json.py          merge 3_1 outputs -> 4_data_2048_512/0_{item}.json
#   4_1_process_json.py        fields cleanup -> 4_data_2048_512/1_{item}_processed.json
#   4_2_merge_cot.py           merge reasoning_trace annotations -> 4_data_2048_512/2_cot_{item}.json
#   4_3_add_solution.py        merge AutoDrive_P3 bbox solutions -> 4_data_2048_512/3_cot_solution_{item}.json
#   4_4_build_hf_cache.py      build the HF cache -> 4_data_2048_512/hf_cache_{item}_CoT_Solution
#
# All outputs go to $AUGUSTUS_DATA_ROOT (default: $ADT_ROOT/Data/debug) —
# paths come from env.sh at the repo root, the single place to configure.
#
# The PDMS metric caches (data.metric_cache_train / data.metric_cache_test)
# are built separately by NAVSIM_Cache.sh at the repo root.
#
# Usage:  bash data_process.sh          # full pipeline, both splits
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

source ../env.sh
_activate_augustus_env

echo "[data_process.sh] NAVSIM_ROOT        = $NAVSIM_ROOT"
echo "[data_process.sh] ANNOTATION_ROOT    = $AUGUSTUS_ANNOTATION_ROOT"
echo "[data_process.sh] DATA_OUTPUT_ROOT   = $AUGUSTUS_DATA_ROOT"
echo "[data_process.sh] python             = $(which python)"

# ---- stage 0-1: scene filtering + per-frame JSONs ----------------------------
python 0_get_useful_scenes.py
python 1_get_json.py --item train
python 1_get_json.py --item test

# ---- stage 2: frames + history videos ----------------------------------------
python 2_0_get_all_frames.py --item train
python 2_0_get_all_frames.py --item test
python 2_1_get_all_videos.py --item train
python 2_1_get_all_videos.py --item test

# ---- stage 3: trajectory + velocity ------------------------------------------
python 3_0_get_traj_heading.py --item train
python 3_0_get_traj_heading.py --item test
python 3_1_get_v.py --item train
python 3_1_get_v.py --item test

# ---- optional: AutoDrive_P3 bbox-solution resolution conversion ---------------
# (the 2048x512 JSONs are shipped with the repo; only needed for other resolutions)
# python AutoDrive_P3/convert_bbox.py --item train --src-x 672 --src-y 168 --dst-x 2048 --dst-y 512
# python AutoDrive_P3/convert_bbox.py --item test --src-x 672 --src-y 168 --dst-x 2048 --dst-y 512
# python AutoDrive_P3/check_bbox_visualization.py --item test --src-x 672 --src-y 168 --dst-x 2048 --dst-y 512

# ---- stage 4: merge / CoT / solutions / HF cache ------------------------------
python 4_0_merge_json.py --item train
python 4_0_merge_json.py --item test
python 4_1_process_json.py --item train
python 4_1_process_json.py --item test
python 4_2_merge_cot.py --item train
python 4_2_merge_cot.py --item test
python 4_3_add_solution.py --x 2048 --y 512 --item train
python 4_3_add_solution.py --x 2048 --y 512 --item test
python 4_4_build_hf_cache.py --item train
python 4_4_build_hf_cache.py --item test

echo "[data_process.sh] DONE. Outputs in: $AUGUSTUS_DATA_ROOT/4_data_2048_512/"
