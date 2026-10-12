"""
Build HuggingFace Dataset cache for Transfusion (NAVSIM).

Reads the processed JSON (3_cot_solution_{item}.json), loads videos as bytes,
builds prompts at training time, and saves as HuggingFace Dataset (Arrow
format) in batches.

Usage:
    python 4_4_build_hf_cache.py --item train
    python 4_4_build_hf_cache.py --item test
"""

import argparse
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

from _paths import data_4_dir, videos_dir

# ---- Configuration ----
# ── Resolution for VLM patch alignment ──
# Qwen2.5-VL patch_size=28 | Qwen3.5-VL patch_size=32
# See 2_0_get_all_frames.py for full resolution mapping table.
#
# -- Qwen2.5-VL (patch_size=28) --
# x, y = 672, 168

# -- Qwen3.5-VL (patch_size=32) --
# x, y = 768, 192
# x, y = 1024, 256
DEFAULT_X, DEFAULT_Y = 2048, 512

BATCH_SIZE = 5000
MAX_WORKERS = 48


def parse_str_list(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(s)



# ---- Video loading ----

def load_video_bytes(video_path: str) -> bytes:
    """Read video file as raw bytes (no decoding — much faster).
    Decoding is deferred to training time in TransfusionHFCacheDataset.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    with open(video_path, "rb") as f:
        return f.read()


# ---- Processing ----

def process_single_item(item_id: str, item: Dict) -> Dict:
    """Process a single data item (for parallel execution)."""
    try:
        # Skip items without COT
        reasoning_trace = item.get("reasoning_trace")
        if reasoning_trace is None:
            print(f"[SKIP] No COT for: {item_id}")
            return None

        video_path = os.path.join(VIDEO_ROOT, item["video_path"])
        video_bytes = load_video_bytes(video_path)

        future = parse_str_list(item["future_x_y_angle"])
        future_flat = json.dumps(future)
        solution_perception = item["solution"]["perception"]
        solution_prediction = item["solution"]["prediction"]
        solution_planning = item["solution"]["planning"]

        return {
            "id": item_id,
            "video": video_bytes,
            "scene_token": item["scene_token"],
            "frame_id": item["frame_id"],
            "driving_command": item["driving_command"],
            "ego_dynamic_state": item["ego_dynamic_state"],
            "historical_x_y_angle": item["historical_x_y_angle"],
            "historical_vx_vy": item["historical_vx_vy"],
            "future_x_y_angle": future_flat,
            "reasoning_trace": reasoning_trace,
            "solution_perception": solution_perception,
            "solution_prediction": solution_prediction,
            "solution_planning": solution_planning,
        }
    except Exception as e:
        print(f"Error processing item {item_id}: {e}")
        return None


def load_json_in_batches(json_path: str, batch_size: int):
    """Yield batches of (item_id, item) from a JSON dict."""
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    items = list(raw_data.items())
    for i in range(0, len(items), batch_size):
        yield {k: v for k, v in items[i : i + batch_size]}


def process_batch(batch: Dict[str, Any], max_workers: int) -> List[Dict]:
    """Process a batch in parallel using threads (I/O bound)."""
    processed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_item, item_id, item)
            for item_id, item in batch.items()
        ]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing batch", leave=False):
            result = future.result()
            if result is not None:
                processed.append(result)
    return processed


def create_dataset(processed_data: List[Dict]) -> Dataset:
    """Create a HuggingFace Dataset from processed items."""
    features = Features({
        "id": Value("string"),
        "video": Value("binary"),  # raw mp4 bytes
        "scene_token": Value("string"),
        "frame_id": Value("string"),
        "driving_command": Value("string"),
        "ego_dynamic_state": Value("string"),
        "historical_x_y_angle": Value("string"),
        "historical_vx_vy": Value("string"),
        "future_x_y_angle": Value("string"),
        "reasoning_trace": Value("string"),
        "solution_perception": Value("string"),
        "solution_prediction": Value("string"),
        "solution_planning": Value("string"),
    })

    # video is already bytes, no conversion needed
    dataset_dict = {
        key: [item[key] for item in processed_data]
        for key in features.keys()
    }
    return Dataset.from_dict(dataset_dict, features=features)


def save_batch(batch_data: List[Dict], output_dir: str, batch_idx: int):
    """Save one batch as HuggingFace Dataset."""
    try:
        dataset = create_dataset(batch_data)
        batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
        dataset.save_to_disk(batch_dir)
        print(f"Batch {batch_idx}: saved {len(batch_data)} samples to {batch_dir}")
    except Exception as e:
        print(f"Error saving batch {batch_idx}: {e}")


def main():
    global ITEM, INPUT_JSON, VIDEO_ROOT, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train", dest="ITEM")
    parser.add_argument("--x", type=int, default=DEFAULT_X)
    parser.add_argument("--y", type=int, default=DEFAULT_Y)
    args = parser.parse_args()
    ITEM, x, y = args.ITEM, args.x, args.y

    DATA_DIR = data_4_dir(x, y)
    VIDEO_ROOT = videos_dir(ITEM, x, y)
    INPUT_JSON = str(DATA_DIR / f"3_cot_solution_{ITEM}.json")
    OUTPUT_DIR = str(DATA_DIR / f"hf_cache_{ITEM}_CoT_Solution")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Processing {ITEM} data...")
    print(f"Input:  {INPUT_JSON}")
    print(f"Videos: {VIDEO_ROOT}")
    print(f"Output: {OUTPUT_DIR}")

    for batch_idx, batch in enumerate(load_json_in_batches(INPUT_JSON, BATCH_SIZE)):
        print(f"\nBatch {batch_idx} ({len(batch)} items)...")
        processed_batch = process_batch(batch, MAX_WORKERS)
        save_batch(processed_batch, OUTPUT_DIR, batch_idx)

    print("\nAll batches saved.")


if __name__ == "__main__":
    main()
