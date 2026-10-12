"""Merge reasoning_trace from annotation JSONs into 1_{item}_processed.json → 2_cot_{item}.json."""

import argparse
import json
import os
import shutil

from _paths import annotation_outputs, data_4_dir

# -- Qwen2.5-VL (patch_size=28) --
# x, y = 672, 168

# -- Qwen3.5-VL (patch_size=32) --
# x, y = 768, 192
# x, y = 1024, 256
DEFAULT_X, DEFAULT_Y = 2048, 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train")
    parser.add_argument("--x", type=int, default=DEFAULT_X)
    parser.add_argument("--y", type=int, default=DEFAULT_Y)
    parser.add_argument("--annotation-dir", type=str, default=None,
                        help="reasoning_trace annotation dir; defaults to "
                             "$AUGUSTUS_ANNOTATION_ROOT/transfusion_outputs_{item}")
    args = parser.parse_args()
    item, x, y = args.item, args.x, args.y

    INPUT_JSON = str(data_4_dir(x, y) / f"1_{item}_processed.json")
    ANNOTATION_DIRS = [args.annotation_dir or str(annotation_outputs(item))]
    OUTPUT_JSON = str(data_4_dir(x, y) / f"2_cot_{item}.json")

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    total = len(data)
    success = 0
    failed_keys = []
    moved_count = 0
    primary_dir = ANNOTATION_DIRS[0]

    for key, item in data.items():
        scene_token = item["scene_token"]
        frame_id = item["frame_id"]

        found = False
        for ann_dir in ANNOTATION_DIRS:
            ann_path = os.path.join(ann_dir, scene_token, frame_id, "reasoning_trace.json")
            if os.path.isfile(ann_path):
                try:
                    with open(ann_path, "r") as af:
                        ann = json.load(af)
                    reasoning_trace = ann.get("reasoning_trace")
                    if reasoning_trace is not None:
                        item["reasoning_trace"] = reasoning_trace
                        success += 1
                        found = True
                        # Move from fallback dir to primary dir
                        if ann_dir != primary_dir:
                            dst_dir = os.path.join(primary_dir, scene_token, frame_id)
                            os.makedirs(dst_dir, exist_ok=True)
                            shutil.copy2(ann_path, os.path.join(dst_dir, "reasoning_trace.json"))
                            moved_count += 1
                        break
                except (json.JSONDecodeError, KeyError):
                    continue
        if not found:
            failed_keys.append(key)
            item["reasoning_trace"] = None

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Total items:    {total}")
    print(f"Success (COT):  {success}")
    print(f"Failed:         {len(failed_keys)}")
    print(f"Copied to primary dir: {moved_count}")
    if failed_keys:
        print(f"\nFailed items ({len(failed_keys)}):")
        for k in failed_keys:
            print(f"  - {k}")
    print(f"\nOutput saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
