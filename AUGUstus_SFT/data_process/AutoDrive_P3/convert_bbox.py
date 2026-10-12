"""
Convert bbox_2d coordinates from Qwen2.5-VL (672x168) to Qwen3.5-VL (768x192).

The AutoDrive_P3 solution JSON stores bbox_2d as pixel coordinates in the resized
frame image.  When switching from Qwen2.5 (672x168) to Qwen3.5 (768x192), all
bbox pixel coordinates must be scaled proportionally:

    new_x = old_x * (768 / 672)   = old_x * (8/7)
    new_y = old_y * (192 / 168)   = old_y * (8/7)

Both resolutions share the same 24x6 patch grid, so the scale factor is uniform.

Usage:
    # Convert test set
    python convert_bbox.py --item test

    # Convert train set
    python convert_bbox.py --item train

    # Custom resolutions
    python convert_bbox.py --item test --src-x 672 --src-y 168 --dst-x 768 --dst-y 192
    python convert_bbox.py --item test --src-x 672 --src-y 168 --dst-x 1024 --dst-y 256
    python convert_bbox.py --item test --src-x 672 --src-y 168 --dst-x 2048 --dst-y 512
"""

import argparse
import ast
import json
import re
from pathlib import Path


# ── Resolution mapping (same patch grid, different patch sizes) ──
# Qwen2.5-VL (patch_size=28):  672 x 168
# Qwen3.5-VL (patch_size=32):  768 x 192


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale bbox_2d coordinates from one resolution to another."
    )
    parser.add_argument("--item", choices=["train", "test"], default="test")
    parser.add_argument("--src-x", type=int, default=672,
                        help="Source image width  (Qwen2.5 default: 672)")
    parser.add_argument("--src-y", type=int, default=168,
                        help="Source image height (Qwen2.5 default: 168)")
    parser.add_argument("--dst-x", type=int, default=768,
                        help="Target image width  (Qwen3.5 default: 768)")
    parser.add_argument("--dst-y", type=int, default=192,
                        help="Target image height (Qwen3.5 default: 192)")
    parser.add_argument("--input-json", type=Path, default=None,
                        help="Override input JSON path.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Override output JSON path.")
    return parser.parse_args()


def scale_bbox(bbox, scale_x, scale_y):
    """Scale a single bbox [x1, y1, x2, y2] by the given factors, rounding to int."""
    if not bbox:
        return bbox
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y),
    ]


def extract_code_block(text):
    """Extract the JSON list from a ```json ... ``` code block string."""
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if not match:
        return None
    return ast.literal_eval(match.group(1))


def rebuild_code_block(data):
    """Rebuild a ```json ... ``` code block string from a Python list."""
    items = []
    for entry in data:
        items.append("\t" + json.dumps(entry, ensure_ascii=False))
    return "```json\n[\n" + ",\n".join(items) + "\n]\n```"


def convert_entry(entry, scale_x, scale_y):
    """Scale all bbox_2d values in a single data entry's solution."""
    solution = entry.get("solution")
    if solution is None:
        return entry

    converted = dict(entry)
    converted_solution = dict(solution)

    for field in ("perception", "prediction"):
        raw = solution.get(field)
        if raw is None:
            continue
        parsed = extract_code_block(raw)
        if parsed is None:
            continue
        for item in parsed:
            if "bbox_2d" in item and item["bbox_2d"]:
                item["bbox_2d"] = scale_bbox(item["bbox_2d"], scale_x, scale_y)
        converted_solution[field] = rebuild_code_block(parsed)

    converted["solution"] = converted_solution
    return converted


def main():
    args = parse_args()

    data_process_dir = Path(__file__).resolve().parent.parent
    autodrive_dir = data_process_dir / "AutoDrive_P3"

    input_json = args.input_json or (
        autodrive_dir / f"1_RL_{args.item}_{args.src_x}_{args.src_y}.json"
    )
    output_json = args.output_json or (
        autodrive_dir / f"1_RL_{args.item}_{args.dst_x}_{args.dst_y}.json"
    )

    scale_x = args.dst_x / args.src_x
    scale_y = args.dst_y / args.src_y

    print(f"Input:   {input_json}")
    print(f"Output:  {output_json}")
    print(f"Scale:   x={scale_x:.6f} ({args.dst_x}/{args.src_x}), "
          f"y={scale_y:.6f} ({args.dst_y}/{args.src_y})")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_data = {}
    total_bbox = 0
    for key, entry in data.items():
        converted_data[key] = convert_entry(entry, scale_x, scale_y)
        # Count scaled bboxes for reporting
        sol = entry.get("solution", {})
        for field in ("perception", "prediction"):
            parsed = extract_code_block(sol.get(field, ""))
            if parsed:
                for item in parsed:
                    if item.get("bbox_2d"):
                        total_bbox += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    print(f"Converted {len(converted_data)} entries ({total_bbox} bbox annotations).")
    print(f"Output saved to: {output_json}")


if __name__ == "__main__":
    main()
