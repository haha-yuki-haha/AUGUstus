import os
import argparse
import json
import shutil
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from PIL import Image
from moviepy.editor import ImageSequenceClip

from _paths import frames_dir, get_json_dir, videos_dir


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
item = "train"
info_dir = None
image_dir = None
save_dir = None

HISTORY_LEN = 3
FPS = 2.0
VIDEO_EXT = '.mp4'

def process_json_file(args):
    scene_name, json_file = args
    scene_save_dir = os.path.join(save_dir, scene_name)
    os.makedirs(scene_save_dir, exist_ok=True)
    scene_dir = os.path.join(info_dir, scene_name)
    file_path = os.path.join(scene_dir, json_file)
    try:
        # Resume support: skip frames whose clip was already written.
        output_path = os.path.join(scene_save_dir, f"{json_file.replace('.json', '')}{VIDEO_EXT}")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return f"Skipped (exists): {scene_name}/{json_file}"
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert json_file.replace(".json", "") == data[HISTORY_LEN]['token'], "Frame ID mismatch"
        image_paths = [
            os.path.join(image_dir, scene_name, f"{data[i]['token']}.jpg")
            for i in range(HISTORY_LEN + 1)
        ]
        # Check whether all images exist
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Skip {file_path}: missing image {img_path}")
                return f"Skipped: {scene_name}/{json_file} (missing image)"
        # Read and process image frames while preserving RGB channel order
        frames = []
        for img_path in image_paths:
            img = Image.open(img_path)
            frames.append(np.array(img))  # Use RGB format directly
        assert len(frames) == HISTORY_LEN + 1, "Incorrect number of frames"
        # Generate the video
        clip = ImageSequenceClip(frames, fps=FPS)
        clip.write_videofile(
            output_path,
            codec='libx264',
            audio=False,
            logger=None,
            threads=2
        )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return f"Processed: {scene_name}/{json_file}"

def main():
    global item, info_dir, image_dir, save_dir, x, y

    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train")
    parser.add_argument("--x", type=int, default=DEFAULT_X)
    parser.add_argument("--y", type=int, default=DEFAULT_Y)
    args = parser.parse_args()
    item, x, y = args.item, args.x, args.y

    info_dir = str(get_json_dir(item))
    image_dir = str(frames_dir(item, x, y))
    save_dir = str(videos_dir(item, x, y))
    os.makedirs(save_dir, exist_ok=True)

    # Collect JSON files from all scenes
    tasks = []
    scenes = [scene for scene in os.listdir(info_dir)
             if os.path.isdir(os.path.join(info_dir, scene))]
    for scene in scenes:
        scene_dir = os.path.join(info_dir, scene)
        json_files = [f for f in os.listdir(scene_dir) if f.endswith('.json')]
        for json_file in json_files:
            tasks.append((scene, json_file))

    num_processes = min(96, os.cpu_count() or 8)
    print(f"Using {num_processes} processes")
    # Process all JSON files in parallel
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_json_file, tasks),
            total=len(tasks),
            desc="Processing json files",
            ncols=100
        ))

if __name__ == '__main__':
    main()