import pickle
import json
import os
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import tqdm

from _paths import get_json_dir, useful_scenes_pkl


def numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj


def process_scene(args):
    """Process data for one scene"""
    scene_datas, save_dir = args

    for scene_data in scene_datas:
        scene_name = scene_data[3]['scene_name']  # e.g. "log-0001-scene-0003"
        frame_id = scene_data[3]['token']

        # Split scene name into log and scene components
        log_name = scene_name.split('-scene-')[0]  # e.g. "log-0001"
        scene_id = scene_name.split('-scene-')[1]  # e.g. "0003"

        # Create directory structure: save_dir/log-XXXX/scene-XXXX/
        # output_dir = os.path.join(save_dir, log_name, f"scene-{scene_id}")
        # os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, scene_data[3]['log_name'])
        os.makedirs(output_dir, exist_ok=True)

        # Convert each frame to JSON-serializable format and save
        # Convert numpy arrays to lists
        frame_json = numpy_to_list(scene_data)
        output_path = os.path.join(output_dir, f"{frame_id}.json")
        with open(output_path, 'w') as f:
            json.dump(frame_json, f, indent=2)

    return scene_name

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train")
    args = parser.parse_args()
    item = args.item

    pkl_path = str(useful_scenes_pkl(item))
    save_dir = str(get_json_dir(item))
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Group frames by scene
    scenes = defaultdict(list)
    for frame_id, frames in data.items():
        scenes[frames[3]['scene_name']].append(frames)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Process scenes in parallel
    with Pool(processes=48) as pool:
        scene_data = list(scenes.values())
        # Create argument tuples for each scene
        args = [(scene, save_dir) for scene in scene_data]
        for _ in tqdm.tqdm(pool.imap_unordered(process_scene, args), 
                          total=len(scene_data),
                          desc="Processing scenes"):
            pass

if __name__ == "__main__":
    main()

## train: 1192 logs and 103288 key frames
## test: 136 logs and 12146 key frames