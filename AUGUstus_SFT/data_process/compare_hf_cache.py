"""Compare two HF-cache datasets (hf_cache_{train,test}_CoT_Solution) for
content-level alignment.

Typical use — verify that a freshly regenerated pipeline output matches the
cache a reference run was trained on (see docs/DATA_PREPARATION.md §6):

    python compare_hf_cache.py <REF_CACHE_DIR> <NEW_CACHE_DIR> \
        --video-samples 200 --video-decode-samples 10

Checks:
  * row count and id-set equality,
  * every text field per id (trajectories, CoT, solutions, ego status, ...),
  * video mp4 bytes on a sampled subset (--video-samples, md5),
  * if bytes differ (e.g. different ffmpeg build), decodes the first frames
    of --video-decode-samples rows and compares pixels.

Exit code 0 = aligned, 1 = mismatch. Row ORDER may differ between runs (the
cache writer uses a thread pool); this is expected and not reported as a
mismatch.
"""

import argparse
import glob
import hashlib
import io
import os
import sys
from collections import defaultdict

import numpy as np


TEXT_FIELDS = [
    "scene_token", "frame_id", "driving_command", "ego_dynamic_state",
    "historical_x_y_angle", "historical_vx_vy", "future_x_y_angle",
    "reasoning_trace", "solution_perception", "solution_prediction",
    "solution_planning",
]


def load_all(root: str):
    from datasets import concatenate_datasets, load_from_disk
    batches = sorted(glob.glob(os.path.join(root, "batch_*")))
    assert batches, f"no batch_* dirs under {root}"
    return concatenate_datasets([load_from_disk(b) for b in batches])


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def decode_first_frame(video_bytes: bytes) -> np.ndarray:
    import av
    container = av.open(io.BytesIO(video_bytes))
    frame = next(container.decode(video=0))
    return np.asarray(frame.to_image())  # same decode path as training


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", help="reference cache dir (e.g. the released run's)")
    ap.add_argument("new", help="freshly generated cache dir to verify")
    ap.add_argument("--video-samples", type=int, default=100,
                    help="rows sampled for the video-bytes md5 check")
    ap.add_argument("--video-decode-samples", type=int, default=10,
                    help="rows pixel-decoded when byte hashes differ")
    args = ap.parse_args()

    print(f"[load] ref = {args.ref}")
    ref = load_all(args.ref)
    print(f"[load] ref rows={len(ref)}")
    print(f"[load] new = {args.new}")
    new = load_all(args.new)
    print(f"[load] new rows={len(new)}")

    ok = True
    if len(ref) != len(new):
        print(f"MISMATCH row count: ref={len(ref)} new={len(new)}")
        ok = False

    ref_ids, new_ids = list(ref["id"]), list(new["id"])
    only_ref = set(ref_ids) - set(new_ids)
    only_new = set(new_ids) - set(ref_ids)
    if only_ref or only_new:
        ok = False
        print(f"MISMATCH ids: only_in_ref={len(only_ref)} only_in_new={len(only_new)}")
        for i in sorted(only_ref)[:5]:
            print(f"  ref-only: {i}")
        for i in sorted(only_new)[:5]:
            print(f"  new-only: {i}")
    else:
        print(f"id sets identical: {len(ref_ids)} ids")

    ref_idx = {rid: i for i, rid in enumerate(ref_ids)}
    new_idx = {rid: i for i, rid in enumerate(new_ids)}
    common = [rid for rid in ref_ids if rid in new_idx]

    field_mismatch = defaultdict(int)
    for rid in common:
        r, n = ref[ref_idx[rid]], new[new_idx[rid]]
        for f in TEXT_FIELDS:
            if r[f] != n[f]:
                field_mismatch[f] += 1
    print(f"compared {len(common)} rows x {len(TEXT_FIELDS)} text fields")
    if field_mismatch:
        ok = False
        for f, c in sorted(field_mismatch.items()):
            print(f"  FIELD MISMATCH {f}: {c} rows")
    else:
        print("all text fields identical")

    step = max(1, len(common) // args.video_samples)
    subset = common[::step][: args.video_samples]
    same_bytes = 0
    bytes_diff_rows = []
    for rid in subset:
        r, n = ref[ref_idx[rid]], new[new_idx[rid]]
        if md5(r["video"]) == md5(n["video"]):
            same_bytes += 1
        else:
            bytes_diff_rows.append(rid)
    print(f"video md5 identical for {same_bytes}/{len(subset)} sampled rows")

    if bytes_diff_rows:
        decode_n = min(len(bytes_diff_rows), args.video_decode_samples)
        pixel_diff = 0
        for rid in bytes_diff_rows[:decode_n]:
            r, n = ref[ref_idx[rid]], new[new_idx[rid]]
            fa, fb = decode_first_frame(r["video"]), decode_first_frame(n["video"])
            if fa.shape != fb.shape:
                pixel_diff += 1
                print(f"  DECODE SHAPE MISMATCH {rid}: {fa.shape} vs {fb.shape}")
                continue
            d = np.abs(fa.astype(int) - fb.astype(int))
            if d.max() > 0:
                pixel_diff += 1
                print(f"  DECODE PIXEL DIFF {rid}: max={d.max()} mean={d.mean():.4f}")
        if pixel_diff == 0:
            print(f"video content (decoded pixels) identical for the "
                  f"{decode_n} rows checked despite byte diffs "
                  "(encoder metadata only)")
        else:
            ok = False

    print("RESULT:", "ALIGNED" if ok else "MISALIGNED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
