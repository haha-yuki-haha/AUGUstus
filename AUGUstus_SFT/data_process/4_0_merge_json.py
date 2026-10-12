import os
import json
import argparse

from _paths import data_4_dir, get_v_dir

# ── Resolution for VLM patch alignment ──
# Qwen2.5-VL patch_size=28 | Qwen3.5-VL patch_size=32
# See 2_0_get_all_frames.py for full resolution mapping table.

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
    args = parser.parse_args()
    item, x, y = args.item, args.x, args.y

    origin_root = str(get_v_dir(item, x, y))
    save_path = str(data_4_dir(x, y) / f"0_{item}.json")

    ## Skip processing when save_path already exists
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping processing.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {}
    for scene_name in os.listdir(origin_root):
        for frame_id in os.listdir(os.path.join(origin_root, scene_name)):
            with open(os.path.join(origin_root, scene_name, frame_id), 'r') as f:
                tmp_data = json.load(f)
            tmp_data["video_path"] = f"{scene_name}/{frame_id.replace('.json', '.mp4')}"
            x_y_heading = tmp_data["x_y_heading"]
            # Headings are already radians (AutoDrive_P3 annotations / step 3_0).
            # Normalize -0.0 -> 0.0 for clean JSON output.
            for key in ["history", "future"]:
                for pt in x_y_heading[key]:
                    if pt[2] == 0.0:
                        pt[2] = 0.0
            tmp_data["x_y_heading"] = x_y_heading
            data[f"{scene_name}_{frame_id.replace('.json', '')}"] = tmp_data

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
