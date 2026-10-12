"""
Check & visualize bbox_2d annotations after resolution conversion.

Draws bounding boxes on the corresponding frame images for both the source
(Qwen2.5, e.g. 672x168) and target (Qwen3.5, e.g. 768x192) resolutions,
and saves two separate images for each sample.

Usage:
    # Basic usage — pick first 5 entries that have non-empty bboxes
    python check_bbox_visualization.py --item test

    # Specify custom parameters
    python check_bbox_visualization.py --item test --max-samples 10 --save-dir ./check_output

    # Custom resolutions
    python check_bbox_visualization.py --item test --src-x 672 --src-y 168 --dst-x 768 --dst-y 192
    python check_bbox_visualization.py --item test --src-x 672 --src-y 168 --dst-x 1024 --dst-y 256
    python check_bbox_visualization.py --item test --src-x 672 --src-y 168 --dst-x 2048 --dst-y 512
"""


import argparse
import ast
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


# ── Resolution mapping (same patch grid, different patch sizes) ──
# Qwen2.5-VL (patch_size=28):  672 x 168,  784 x 196,  1008 x 252
# Qwen3.5-VL (patch_size=32):  768 x 192,  896 x 224,  1152 x 288

# Color palette for different object classes (BGR)
CLASS_COLORS = {
    "vehicle":     (0, 255, 0),    # green
    "pedestrian":  (0, 0, 255),    # red
    "bicycle":     (255, 128, 0),  # blue-ish
    "default":     (255, 255, 0),  # cyan
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize bbox_2d on frames for Qwen2.5 vs Qwen3.5 comparison."
    )
    parser.add_argument("--item", choices=["train", "test"], default="test")
    parser.add_argument("--src-x", type=int, default=672,
                        help="Qwen2.5 image width  (default: 672)")
    parser.add_argument("--src-y", type=int, default=168,
                        help="Qwen2.5 image height (default: 168)")
    parser.add_argument("--dst-x", type=int, default=768,
                        help="Qwen3.5 image width  (default: 768)")
    parser.add_argument("--dst-y", type=int, default=192,
                        help="Qwen3.5 image height (default: 192)")
    parser.add_argument("--max-samples", type=int, default=5,
                        help="Max number of samples to visualize.")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Output directory for comparison images.")
    parser.add_argument("--src-json", type=Path, default=None,
                        help="Override source JSON path (672x168).")
    parser.add_argument("--dst-json", type=Path, default=None,
                        help="Override target JSON path (768x192).")
    parser.add_argument("--src-frame-root", type=Path, default=None,
                        help="Override frame image root for source resolution.")
    parser.add_argument("--dst-frame-root", type=Path, default=None,
                        help="Override frame image root for target resolution.")
    return parser.parse_args()


def extract_code_block(text):
    """Extract the JSON list from a ```json ... ``` code block string."""
    if not text:
        return None
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return None


def draw_bboxes(image, bboxes, labels, prefix=""):
    """Draw bounding boxes and labels on an image (in-place). Returns the image."""
    for bbox, label in zip(bboxes, labels):
        color = CLASS_COLORS.get(label, CLASS_COLORS["default"])
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{prefix}{label}"
        # Draw label background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def find_entries_with_bboxes(data):
    """Return list of (key, entry) that have at least one non-empty bbox_2d."""
    result = []
    for key, entry in data.items():
        sol = entry.get("solution", {})
        perception = extract_code_block(sol.get("perception", ""))
        if perception:
            for item in perception:
                if item.get("bbox_2d"):
                    result.append((key, entry))
                    break
    return result


def main():
    args = parse_args()

    data_process_dir = Path(__file__).resolve().parent.parent
    autodrive_dir = data_process_dir / "AutoDrive_P3"
    frames_base = data_process_dir

    src_json_path = args.src_json or (
        autodrive_dir / f"1_RL_{args.item}_{args.src_x}_{args.src_y}.json"
    )
    dst_json_path = args.dst_json or (
        autodrive_dir / f"1_RL_{args.item}_{args.dst_x}_{args.dst_y}.json"
    )

    src_frame_root = args.src_frame_root or (
        frames_base / f"2_0_get_all_frames_{args.item}_{args.src_x}_{args.src_y}"
    )
    dst_frame_root = args.dst_frame_root or (
        frames_base / f"2_0_get_all_frames_{args.item}_{args.dst_x}_{args.dst_y}"
    )

    save_dir = args.save_dir or (
        autodrive_dir / f"check_bbox_{args.src_x}_{args.src_y}_vs_{args.dst_x}_{args.dst_y}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load JSONs ──
    print(f"Loading source JSON: {src_json_path}")
    with open(src_json_path, "r", encoding="utf-8") as f:
        src_data = json.load(f)

    dst_data = None
    if dst_json_path.exists():
        print(f"Loading target JSON: {dst_json_path}")
        with open(dst_json_path, "r", encoding="utf-8") as f:
            dst_data = json.load(f)
    else:
        print(f"[WARN] Target JSON not found: {dst_json_path}")
        print("       Will only draw source bboxes. Run convert_bbox script first for comparison.")

    # ── Find entries with bboxes ──
    bbox_entries = find_entries_with_bboxes(src_data)
    print(f"Found {len(bbox_entries)} entries with non-empty bbox_2d annotations.")

    if not bbox_entries:
        print("Nothing to visualize. Exiting.")
        return

    samples = bbox_entries[:args.max_samples]
    print(f"Visualizing {len(samples)} samples...")

    for idx, (key, src_entry) in enumerate(samples):
        scene_token = src_entry.get("scene_name") or src_entry.get("scene_token")
        frame_id = src_entry["frame_id"]
        print(f"\n[{idx+1}/{len(samples)}] {key}")

        # ── Extract source bboxes ──
        src_perception = extract_code_block(
            src_entry.get("solution", {}).get("perception", "")
        )
        src_bboxes = []
        src_labels = []
        if src_perception:
            for item in src_perception:
                if item.get("bbox_2d"):
                    src_bboxes.append(item["bbox_2d"])
                    src_labels.append(item.get("label", "object"))

        # ── Load source frame image ──
        src_img_path = src_frame_root / scene_token / f"{frame_id}.jpg"
        src_img = None
        if src_img_path.exists():
            src_img = cv2.imread(str(src_img_path))
            draw_bboxes(src_img, src_bboxes, src_labels, prefix="")
            h, w = src_img.shape[:2]
            cv2.putText(src_img, f"Qwen2.5 ({w}x{h})", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            print(f"  [WARN] Source frame not found: {src_img_path}")

        # ── Extract target bboxes & load target frame ──
        dst_img = None
        dst_bboxes = []
        dst_labels = []
        if dst_data and key in dst_data:
            dst_entry = dst_data[key]
            dst_perception = extract_code_block(
                dst_entry.get("solution", {}).get("perception", "")
            )
            if dst_perception:
                for item in dst_perception:
                    if item.get("bbox_2d"):
                        dst_bboxes.append(item["bbox_2d"])
                        dst_labels.append(item.get("label", "object"))

            dst_img_path = dst_frame_root / scene_token / f"{frame_id}.jpg"
            if dst_img_path.exists():
                dst_img = cv2.imread(str(dst_img_path))
                draw_bboxes(dst_img, dst_bboxes, dst_labels, prefix="")
                h, w = dst_img.shape[:2]
                cv2.putText(dst_img, f"Qwen3.5 ({w}x{h})", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                print(f"  [WARN] Target frame not found: {dst_img_path}")
                print(f"         Run 2_0_get_all_frames.py with Qwen3.5 resolution ({args.dst_x}x{args.dst_y}) first.")

        # ── Save two separate images ──
        if src_img is not None:
            save_path = save_dir / f"{idx+1:02d}_{key[:40]}_qwen2.5.jpg"
            cv2.imwrite(str(save_path), src_img)
            print(f"  Saved Qwen2.5 image: {save_path}")
            print(f"  Qwen2.5 bboxes ({args.src_x}x{args.src_y}):")
            for b, l in zip(src_bboxes, src_labels):
                print(f"    {l}: {b}")

        if dst_img is not None:
            save_path = save_dir / f"{idx+1:02d}_{key[:40]}_qwen3.5.jpg"
            cv2.imwrite(str(save_path), dst_img)
            print(f"  Saved Qwen3.5 image: {save_path}")
            print(f"  Qwen3.5 bboxes ({args.dst_x}x{args.dst_y}):")
            for b, l in zip(dst_bboxes, dst_labels):
                print(f"    {l}: {b}")

    print(f"\nDone. Qwen2.5 & Qwen3.5 images saved to: {save_dir}")


if __name__ == "__main__":
    main()
