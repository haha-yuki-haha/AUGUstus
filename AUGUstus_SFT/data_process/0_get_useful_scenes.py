import yaml
from pathlib import Path
import pickle
import os
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Union
import json

from _paths import navsim_logs, useful_scenes_pkl


class SceneFilter:
    num_history_frames: int = 4
    num_future_frames: int = 10
    frame_interval: Optional[int] = None
    has_route: bool = True

    max_scenes: Optional[int] = None
    log_names: Optional[List[str]] = None
    tokens: Optional[List[str]] = None
    # TODO: expand filter options

    def __post_init__(self):

        if self.frame_interval is None:
            self.frame_interval = self.num_frames

        assert (
            self.num_history_frames >= 1
        ), "SceneFilter: num_history_frames must greater equal one."
        assert (
            self.num_future_frames >= 0
        ), "SceneFilter: num_future_frames must greater equal zero."
        assert self.frame_interval >= 1, "SceneFilter: frame_interval must greater equal one."

    @property
    def num_frames(self) -> int:
        return self.num_history_frames + self.num_future_frames


def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Dict[str, List[Dict[str, Any]]]:

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    # filtered_scenes: Dict[str, Scene] = {}
    filtered_scenes: Dict = {}
    stop_loading: bool = False

    # filter logs
    log_files = list(data_path.iterdir())
    if scene_filter.log_names is not None:
        log_files = [
            log_file
            for log_file in log_files
            if log_file.name.replace(".pkl", "") in scene_filter.log_names
        ]
        
    # filter tokens
    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)
    else:
        filter_tokens = False

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        try:
            scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        except Exception as e:
            print(f"\nWarning: Failed to load {log_pickle_path}: {e}, skipping.")
            continue
        for frame_list in split_list(
            scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval
        ):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if (
                scene_filter.has_route
                and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0
            ):
                continue

            # Filter by token
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            if filter_tokens and token not in tokens:
                continue

            filtered_scenes[token] = frame_list

            if (scene_filter.max_scenes is not None) and (
                len(filtered_scenes) >= scene_filter.max_scenes
            ):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes


_script_dir = Path(__file__).resolve().parent

# Define data_path and output_path for each config
# ("train" reads the trainval logs, "test" reads the test logs — see _paths.py)
CONFIG_PATHS = {
    "navtest.yaml": {
        "item": "test",
    },
    "navtrain.yaml": {
        "item": "train",
    },
}

# Process both test and train
for config_name, paths in CONFIG_PATHS.items():
    config_path = _script_dir / config_name
    print(f"\n{'='*50}")
    print(f"Processing {config_name}")
    print(f"{'='*50}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Create scene filter from config
    scene_filter = SceneFilter()
    scene_filter.num_history_frames = cfg['num_history_frames']
    scene_filter.num_future_frames = cfg['num_future_frames']
    scene_filter.frame_interval = cfg['frame_interval']
    scene_filter.has_route = cfg['has_route']
    scene_filter.max_scenes = cfg['max_scenes']
    scene_filter.log_names = cfg['log_names']
    scene_filter.tokens = cfg.get('tokens')

    # Get data path and output path from Python config (not YAML)
    item = paths['item']
    data_path = navsim_logs(item)
    output_path = useful_scenes_pkl(item)

    # Filter scenes and save to file
    print("Begin filtering scenes...")
    filtered_scenes = filter_scenes(data_path, scene_filter)
    print(f"Finished filtering scenes. Total scenes: {len(filtered_scenes)}")

    # Save filtered scenes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_scenes, f)

    print(f"Saved {len(filtered_scenes)} scenes to {output_path}")

print("\nAll done!")

## trainval: 103288 frame IDs
## test: 12146 frame IDs