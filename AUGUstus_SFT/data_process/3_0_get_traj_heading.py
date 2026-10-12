import os
import json
import re
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from _paths import autodrive_p3_json, data_4_dir, get_json_dir, traj_heading_dir

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
json_root = None
solution_data = None
save_dir = None


JSON_BLOCK_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.DOTALL)


def parse_json_block(text: str):
    """Parse the ```json [...]``` block used by the AutoDrive_P3 annotations."""
    match = JSON_BLOCK_RE.search(text)
    if match is None:
        raise ValueError("no ```json``` block found")
    return json.loads(match.group(1))


def process_json(args):
    item_id, save_path = args
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    entry = solution_data[item_id]
    scene_token = entry["scene_name"]
    frame_id = entry["frame_id"]
    data = {
        "scene_token": scene_token,
        "frame_id": frame_id,
    }
    data['video_path'] = f"{data['scene_token']}/{data['frame_id']}.mp4"

    # GT trajectory from the shipped AutoDrive_P3 annotations (headings already
    # in radians, ego-local frame, 0.5 s spacing):
    #   historical_trajectory -> 4 history points (-1.5 s ... 0.0 s)
    #   solution.planning     -> 8 future points (0.5 s ... 4.0 s)
    history = parse_json_block(entry["historical_trajectory"])
    future = parse_json_block(entry["solution"]["planning"])
    assert len(history) == 4, f"{item_id}: expected 4 history points, got {len(history)}"
    assert len(future) == 8, f"{item_id}: expected 8 future points, got {len(future)}"
    data['x_y_heading'] = {
        "history": [pt["x_y_radian"] for pt in history],
        "future": [pt["x_y_radian"] for pt in future],
    }

    # Load ego_dynamic_state, traffic_lights, driving_command from 1_get_json
    json_path = os.path.join(json_root, scene_token, f"{frame_id}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            frame_list = json.load(f)
        # Find the key frame (matching token = frame_id)
        for frame in frame_list:
            if frame['token'] == frame_id:
                data['ego_dynamic_state'] = frame.get('ego_dynamic_state')
                data['traffic_lights'] = frame.get('traffic_lights')
                data['driving_command'] = frame.get('driving_command')
                break

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)


def collect_all_tasks():
    tasks = []
    for item_id, entry in solution_data.items():
        save_path = os.path.join(save_dir, entry["scene_name"], f"{entry['frame_id']}.json")
        tasks.append((item_id, save_path))
    return tasks


def main():
    global json_root, solution_data, save_dir, x, y, item

    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train")
    parser.add_argument("--x", type=int, default=DEFAULT_X)
    parser.add_argument("--y", type=int, default=DEFAULT_Y)
    parser.add_argument("--solution-json", type=str, default=None,
                        help="AutoDrive_P3 JSON; defaults to the shipped "
                             "AutoDrive_P3/1_RL_{item}_{x}_{y}.json")
    args = parser.parse_args()
    item, x, y = args.item, args.x, args.y

    solution_path = args.solution_json or str(autodrive_p3_json(item, x, y))
    json_root = str(get_json_dir(item))
    save_dir = str(traj_heading_dir(item, x, y))
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading AutoDrive_P3 solutions from {solution_path}")
    with open(solution_path, 'r') as f:
        solution_data = json.load(f)

    tasks = collect_all_tasks()
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_json, tasks), total=len(tasks)))


if __name__ == "__main__":
    main()
