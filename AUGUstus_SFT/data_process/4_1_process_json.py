import os
import json
import argparse

from _paths import data_4_dir

# ── Resolution for VLM patch alignment ──
# Qwen2.5-VL patch_size=28 | Qwen3.5-VL patch_size=32
# See 2_0_get_all_frames.py for full resolution mapping table.

# -- Qwen2.5-VL (patch_size=28) --
# x, y = 672, 168

# -- Qwen3.5-VL (patch_size=32) --
# x, y = 768, 192
# x, y = 1024, 256
DEFAULT_X, DEFAULT_Y = 2048, 512

# Paper: https://danieldauner.github.io/assets/pdf/Dauner2024NIPS_supplementary.pdf
# Code: https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/process_data/helpers/driving_command.py
# Correct route_roadblock_ids to handle route departures and disconnected roadblocks
# Find the lane containing the ego vehicle on the route
# Extract the route centerline from the current lane using Dijkstra graph search
# Project 20 m forward along the centerline to find the target point
# Transform the target from global coordinates to the local frame of the current ego pose
# Classify the command by lateral offset
DRIVING_COMMAND_MAP = {
    (1, 0, 0, 0): "TURN LEFT",
    (0, 1, 0, 0): "GO FORWARD",
    (0, 0, 1, 0): "TURN RIGHT",
}


def decode_driving_command(one_hot, key_name=""):
    """Convert one-hot driving command to string."""
    key = tuple(one_hot)
    if key == (0, 0, 0, 1):
        raise ValueError(f"Invalid driving_command [0,0,0,1] (UNKNOWN/route unavailable) found in '{key_name}'")
    return DRIVING_COMMAND_MAP.get(key, "UNKNOWN")


def decode_ego_dynamic_state(state):
    """Convert ego_dynamic_state [vx, vy, ax, ay] to descriptive string."""
    vx, vy, ax, ay = state
    return (
        f"vx: {vx:.2f} m/s, "
        f"vy: {vy:.2f} m/s, "
        f"ax: {ax:.2f} m/s², "
        f"ay: {ay:.2f} m/s²"
    )


def split_trajectory(traj_list):
    """Split trajectory into x_y_angle (first 3 dims) and vx_vy (last 2 dims)."""
    x_y_angle = [point[:3] for point in traj_list]
    vx_vy = [point[3:] for point in traj_list]
    return x_y_angle, vx_vy


HIST_LABELS = ["-1.5s", "-1.0s", "-0.5s", "0.0s"]
FUTURE_LABELS = ["0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "3.5s", "4.0s"]


def format_traj_json_block(points, labels):
    """Format trajectory as ```json [...] ``` code block with x_y_radian + label."""
    items = []
    for pt, label in zip(points, labels):
        rounded = [round(v, 2) for v in pt]
        items.append(f'\t{{"x_y_radian": {rounded}, "label": "{label}"}}')
    return "```json\n[\n" + ",\n".join(items) + "\n]\n```"


def format_vel_json_block(points, labels):
    """Format velocity as ```json [...] ``` code block with vx_vy + label."""
    items = []
    for pt, label in zip(points, labels):
        rounded = [round(v, 2) for v in pt]
        items.append(f'\t{{"vx_vy": {rounded}, "label": "{label}"}}')
    return "```json\n[\n" + ",\n".join(items) + "\n]\n```"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train")
    parser.add_argument("--x", type=int, default=DEFAULT_X)
    parser.add_argument("--y", type=int, default=DEFAULT_Y)
    args = parser.parse_args()
    item, x, y = args.item, args.x, args.y

    input_path = str(data_4_dir(x, y) / f"0_{item}.json")
    output_path = str(data_4_dir(x, y) / f"1_{item}_processed.json")

    with open(input_path, "r") as f:
        data = json.load(f)

    processed_data = {}
    for key, value in data.items():
        hist_x_y_angle, hist_vx_vy = split_trajectory(value["x_y_heading"]["history"])
        fut_x_y_angle, fut_vx_vy = split_trajectory(value["x_y_heading"]["future"])
        processed_data[key] = {
            "scene_token": value["scene_token"],
            "frame_id": value["frame_id"],
            "video_path": value["video_path"],
            "driving_command": decode_driving_command(value["driving_command"], key_name=key),
            "ego_dynamic_state": decode_ego_dynamic_state(value["ego_dynamic_state"]),
            "historical_x_y_angle": str(hist_x_y_angle),
            "future_x_y_angle": str(fut_x_y_angle),
            "historical_vx_vy": str(hist_vx_vy),
            "future_vx_vy": str(fut_vx_vy),
        }

    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=4)

    print(f"Processed {len(processed_data)} entries.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
