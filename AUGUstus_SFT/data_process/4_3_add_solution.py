"""
Merge the AutoDrive_P3 solution JSONs into 2_cot_{item}.json → 3_cot_solution_{item}.json.

Usage (from this directory):
    python 4_3_add_solution.py --item test
    python 4_3_add_solution.py --item train
"""

import argparse
import json
from pathlib import Path

from _paths import autodrive_p3_json, data_4_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge AutoDrive_P3 solution into processed NAVSIM JSON by item id."
    )
    # ── Resolution for VLM patch alignment ──
    # Qwen2.5-VL patch_size=28: 672x168 (24x6)
    # Qwen3.5-VL patch_size=32: 768x192 (24x6)
    parser.add_argument("--x", type=int, default=2048,  # Qwen2.5 use 672; for Qwen3.5 use 768
                        help="Image width (must be divisible by patch_size). "
                             "Qwen2.5-VL patch_size=28 → 672; Qwen3.5-VL patch_size=32 → 768.")
    parser.add_argument("--y", type=int, default=512,  # Qwen2.5 use 168; for Qwen3.5 use 192
                        help="Image height (must be divisible by patch_size). "
                             "Qwen2.5-VL patch_size=28 → 168; Qwen3.5-VL patch_size=32 → 192.")
    parser.add_argument("--item", choices=["train", "test"], default="train")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Processed NAVSIM JSON path. Defaults to data_process/4_data_{x}_{y}/2_cot_{item}.json",
    )
    parser.add_argument(
        "--solution-json",
        type=Path,
        default=None,
        help="AutoDrive_P3 JSON path. Defaults to data_process/AutoDrive_P3/1_RL_{item}_{x}_{y}.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to data_process/4_data_{x}_{y}/3_cot_solution_{item}.json",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep samples without solution instead of failing.",
    )
    return parser.parse_args()


def load_json_dict(path: Path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict JSON in {path}, got {type(data).__name__}")
    return data


def resolve_paths(args):
    # Pipeline outputs live under the configured data root (see _paths.py /
    # env.sh), NOT inside the repository checkout.
    data_dir = data_4_dir(args.x, args.y)

    input_json = Path(args.input_json or data_dir / f"2_cot_{args.item}.json")
    solution_json = Path(
        args.solution_json
        or autodrive_p3_json(args.item, args.x, args.y)
    )
    output_json = Path(args.output_json or data_dir / f"3_cot_solution_{args.item}.json")

    return input_json, solution_json, output_json


def merge_solution(processed_data, solution_data, allow_missing=False):
    merged_data = {}
    missing_ids = []
    missing_solution_ids = []

    for item_id, item_value in processed_data.items():
        solution_item = solution_data.get(item_id)
        if solution_item is None:
            missing_ids.append(item_id)
            if allow_missing:
                merged_data[item_id] = item_value
                continue
            continue

        if "solution" not in solution_item:
            missing_solution_ids.append(item_id)
            if allow_missing:
                merged_data[item_id] = item_value
                continue
            continue

        merged_item = dict(item_value)
        merged_item["solution"] = solution_item["solution"]
        merged_data[item_id] = merged_item

    if (missing_ids or missing_solution_ids) and not allow_missing:
        preview = ", ".join((missing_ids + missing_solution_ids)[:5])
        raise KeyError(
            "Failed to merge all solutions. "
            f"Missing ids: {len(missing_ids)}, missing solution field: {len(missing_solution_ids)}. "
            f"Examples: {preview}"
        )

    extra_ids = sorted(set(solution_data) - set(processed_data))
    return merged_data, missing_ids, missing_solution_ids, extra_ids


def main():
    args = parse_args()
    input_json, solution_json, output_json = resolve_paths(args)

    if not input_json.exists():
        raise FileNotFoundError(f"Processed JSON not found: {input_json}")
    if not solution_json.exists():
        raise FileNotFoundError(f"AutoDrive_P3 JSON not found: {solution_json}")

    processed_data = load_json_dict(input_json)
    solution_data = load_json_dict(solution_json)

    merged_data, missing_ids, missing_solution_ids, extra_ids = merge_solution(
        processed_data,
        solution_data,
        allow_missing=args.allow_missing,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(merged_data, file, indent=4, ensure_ascii=False)

    print(f"Processed items: {len(processed_data)}")
    print(f"Merged items: {len(merged_data)}")
    print(f"Missing ids: {len(missing_ids)}")
    print(f"Missing solution field: {len(missing_solution_ids)}")
    print(f"Extra AutoDrive_P3 ids: {len(extra_ids)}")
    print(f"Output saved to: {output_json}")


if __name__ == "__main__":
    main()