import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import math

from _paths import get_json_dir, get_v_dir, traj_heading_dir

# ── Resolution for VLM patch alignment ──
# Qwen2.5-VL patch_size=28 | Qwen3.5-VL patch_size=32
# See 2_0_get_all_frames.py for full resolution mapping table.

# -- Qwen2.5-VL (patch_size=28) --
# x, y = 672, 168

# -- Qwen3.5-VL (patch_size=32) --
# x, y = 768, 192
# x, y = 1024, 256
DEFAULT_X, DEFAULT_Y = 2048, 512


# Globals bound in main(); used by the worker function.
root_dir = None
traj_root = None
save_dir = None


def format_speed(vx, vy):
    speed = math.sqrt(vx ** 2 + vy ** 2)
    speed = round(speed + 1e-8, 2)  # Round to two decimal places and avoid -0.00
    if abs(speed) < 1e-6:
        return 0.0
    return speed

def process_json(args):
    file_path, traj_path, save_path = args
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(file_path, 'r') as f:
        data = json.load(f)
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    traj_tmp_history = traj_data['x_y_heading']["history"]
    traj_tmp_future = traj_data['x_y_heading']["future"]

    # Get historical and future vx and vy
    v_history, v_future = [], []
    for i in range(4):
        vx, vy = data[i]["ego_dynamic_state"][0:2]
        v_history.append([
            round(vx, 2) if abs(vx) > 1e-6 else 0.0,
            round(vy, 2) if abs(vy) > 1e-6 else 0.0
        ])
    for i in range(4, 4+8):
        vx, vy = data[i]["ego_dynamic_state"][0:2]
        v_future.append([
            round(vx, 2) if abs(vx) > 1e-6 else 0.0,
            round(vy, 2) if abs(vy) > 1e-6 else 0.0
        ])

    # Add vx and vy to the trajectories
    for idx, v in enumerate(v_history):
        traj_tmp_history[idx].extend(v)
    for idx, v in enumerate(v_future):
        traj_tmp_future[idx].extend(v)

    traj_data['x_y_heading']["history"] = traj_tmp_history
    traj_data['x_y_heading']["future"] = traj_tmp_future
    with open(save_path, 'w') as f:
        json.dump(traj_data, f, indent=2)

def collect_all_tasks():
    tasks = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                traj_path = os.path.join(traj_root, os.path.relpath(file_path, root_dir))
                save_path = os.path.join(save_dir, os.path.relpath(file_path, root_dir))
                tasks.append((file_path, traj_path, save_path))
    return tasks

def main():
    global root_dir, traj_root, save_dir, x, y, item

    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train")
    parser.add_argument("--x", type=int, default=DEFAULT_X)
    parser.add_argument("--y", type=int, default=DEFAULT_Y)
    args = parser.parse_args()
    item, x, y = args.item, args.x, args.y

    root_dir = str(get_json_dir(item))
    traj_root = str(traj_heading_dir(item, x, y))
    save_dir = str(get_v_dir(item, x, y))
    os.makedirs(save_dir, exist_ok=True)

    tasks = collect_all_tasks()
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_json, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()



