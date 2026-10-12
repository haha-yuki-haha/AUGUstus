import os
import argparse
import json
import shutil
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from _paths import frames_dir, get_json_dir, sensor_blobs

# Resize stitched image to model-compatible resolution.
# The stitched image (3 cameras concatenated horizontally) must have dimensions
# that are divisible by the VLM's visual patch size so that the patch grid is integer.
#
# ── Qwen2.5-VL ──  patch_size = 28  (image tokens = token * 28 * 28)
#    672 x 168  →  patch grid 24 x 6   (default)
#
# ── Qwen3.5-VL ──  patch_size = 32  (image tokens = token * 32 * 32)
#    768 x 192  →  patch grid 24 x 6   (equivalent to Qwen2.5 672x168)
#
# >>> To switch to Qwen3.5-VL, comment the Qwen2.5 lines and uncomment the Qwen3.5 lines below. <<<

# -- Qwen2.5-VL (patch_size=28) --
# x, y = 672, 168

# -- Qwen3.5-VL (patch_size=32) --
# x, y = 768, 192
# x, y = 1024, 256
DEFAULT_X, DEFAULT_Y = 2048, 512

# item = "test"  /  item = "train"  (--item CLI flag)
# Globals bound in main(); used by the worker function.
item = "train"
img_root = None
root_dir = None
save_dir = None

HISTORY_LEN = 3

def get_key_frame(scene_name):
    scene_save_dir = os.path.join(save_dir, scene_name)
    os.makedirs(scene_save_dir, exist_ok=True)

    scene_dir = os.path.join(root_dir, scene_name)
    json_files = [f for f in os.listdir(scene_dir) if f.endswith('.json')]

    # Resume support: a scene is complete when EVERY key-frame JSON's 4
    # history frames exist as stitched jpgs (adjacent scenes share frames, so
    # a simple file-count check is not valid here).
    try:
        complete = True
        for json_file in json_files:
            with open(os.path.join(scene_dir, json_file), 'r') as f:
                data = json.load(f)
            if not all(os.path.exists(os.path.join(scene_save_dir, f"{data[i]['token']}.jpg"))
                       for i in range(HISTORY_LEN + 1)):
                complete = False
                break
        if complete:
            return f"Skipped (complete): {scene_name}"
    except Exception:
        complete = False

    for json_file in json_files:
        file_path = os.path.join(root_dir, scene_name, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert json_file.replace(".json", "") == data[3]['token'], "Frame ID mismatch"

        skip_json = False
        for i in range(HISTORY_LEN + 1):
            l0_path = os.path.join(img_root, data[i]["cams"]["CAM_L0"]["data_path"])
            f0_path = os.path.join(img_root, data[i]["cams"]["CAM_F0"]["data_path"])
            r0_path = os.path.join(img_root, data[i]["cams"]["CAM_R0"]["data_path"])

            # Check whether the image exists
            if not (os.path.exists(l0_path) and os.path.exists(f0_path) and os.path.exists(r0_path)):
                print(f"Missing image for {data[i]['token']} in scene {scene_name}, skipping this json.")
                skip_json = True
                break

        if skip_json:
            continue  # Skip the current JSON file

        for i in range(HISTORY_LEN + 1):
            l0_path = os.path.join(img_root, data[i]["cams"]["CAM_L0"]["data_path"])
            f0_path = os.path.join(img_root, data[i]["cams"]["CAM_F0"]["data_path"])
            r0_path = os.path.join(img_root, data[i]["cams"]["CAM_R0"]["data_path"])

            l0 = cv2.imread(l0_path)
            f0 = cv2.imread(f0_path)
            r0 = cv2.imread(r0_path)

            # ── Image Cropping (same for both Qwen2.5-VL and Qwen3.5-VL) ──
            # The crop parameters are determined by the original NAVSIM camera image layout
            # (sensor_blobs), NOT by the target model's patch size.
            # They remove sensor-specific borders / irrelevant regions before stitching.
            #
            # NAVSIM camera images (H x W):
            #   CAM_L0 / CAM_R0: 1080 x 1920  (side cameras, with black border padding)
            #   CAM_F0:          1080 x 1920  (front camera)
            #
            # After crop:
            #   L0/R0: [28:-28, 416:-416]  → 1024 x 1088
            #   F0:    [:-56]              → 1024 x 1920
            #
            # Stitched (horizontal concat):  1024 x (1088 x 2 + 1920) = 1024 x 4096
            #
            # >>> The only thing that changes between Qwen2.5 and Qwen3.5 is the resize target below. <<<
            # Qwen2.5-VL: resize to 672 x 168   (patch_size=28, grid 24x6)
            # Qwen3.5-VL: resize to 768 x 192   (patch_size=32, grid 24x6)
            l0 = l0[28:-28, 416:-416]   # side cam: crop black bars & overlap
            f0 = f0[:-56]               # front cam: crop ego-vehicle hood at bottom
            r0 = r0[28:-28, 416:-416]   # side cam: same as L0

            stitched_image = np.concatenate([l0, f0, r0], axis=1)  # 200 x 1408
            resized_image = cv2.resize(stitched_image, (x, y))     # → (x, y) per model
            save_path = os.path.join(scene_save_dir, f"{data[i]['token']}.jpg")
            cv2.imwrite(save_path, resized_image)

    return f"Processed scene: {scene_name}"


def main():
    global item, img_root, root_dir, save_dir, x, y

    parser = argparse.ArgumentParser()
    parser.add_argument("--item", choices=["train", "test"], default="train")
    parser.add_argument("--x", type=int, default=DEFAULT_X)
    parser.add_argument("--y", type=int, default=DEFAULT_Y)
    args = parser.parse_args()
    item, x, y = args.item, args.x, args.y

    img_root = str(sensor_blobs(item))
    root_dir = str(get_json_dir(item))
    save_dir = str(frames_dir(item, x, y))
    os.makedirs(save_dir, exist_ok=True)

    # Get all scene directories
    scenes = [scene for scene in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, scene))]

    num_processes = min(int(os.environ.get("AUGUSTUS_NUM_WORKERS", 192)), os.cpu_count() or 8)
    print(f"Using {num_processes} processes")

    # Process scenes in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(get_key_frame, scenes),
            total=len(scenes),
            desc="Processing scenes",
            ncols=100
        ))

if __name__ == '__main__':
    main()