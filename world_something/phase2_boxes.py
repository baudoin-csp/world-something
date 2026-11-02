"""Phase 2: labelled, gravity-aligned 7-DoF 3D boxes with Boxer, run on Phase 1's output.

Reused tool: Boxer (facebookresearch/boxer) end to end. Its runner detects objects in every photo
with the open-vocabulary OWLv2 detector, lifts each 2D box to a metric 3D box with BoxerNet (using
Phase 1's cameras and depth), and its fusion step merges the per-photo boxes across views.

Phase 1 already writes its data in the CA-1M layout that Boxer's CA-1M loader reads, so the only
glue here is (1) pointing Boxer at that folder and (2) letting the loader run without the
ground-truth file a real CA-1M capture would have.

Outputs in --out:
  ca1m-phase1/boxer_3dbbs.csv        per-photo 3D boxes (Boxer)
  ca1m-phase1/boxer_3dbbs_fused.csv  fused boxes (Boxer)
  ca1m-phase1/boxer_viz_*            overlays of the boxes on the photos (Boxer, unless --skip-viz)
  boxes.json                         fused boxes: label, score, centre, size, yaw (world of Phase 1)
"""

import argparse
import json
import math
import os
import sys

from world_something import add_boxer_to_path
from world_something.phase1_geometry import BOXER_SEQ


def _allow_missing_ground_truth():
    """Boxer's CA-1M loader builds GT boxes from world.gt/instances.json and fails on an empty
    list; Phase 1 output has no GT, so return an empty set instead."""
    import torch
    import loaders.ca_loader as ca_loader
    from utils.tw.obb import ObbTW

    original = ca_loader._load_world_obbs

    def load_world_obbs(data_dir):
        with open(os.path.join(data_dir, "world.gt", "instances.json")) as f:
            if json.load(f):
                return original(data_dir)
        return ObbTW(torch.zeros(0, 165)), {}, {}

    ca_loader._load_world_obbs = load_world_obbs


def run_boxer(phase1_dir, out_dir, labels, thresh2d, thresh3d, skip_viz, force_cpu):
    add_boxer_to_path()
    import utils.demo_utils as demo_utils

    demo_utils.SAMPLE_DATA_PATH = os.path.join(os.path.abspath(phase1_dir), "boxer_input")
    _allow_missing_ground_truth()
    import run_boxer

    argv = ["run_boxer.py", "--input", BOXER_SEQ, "--output_dir", os.path.abspath(out_dir),
            "--start_n", "0", f"--labels={labels}",
            "--thresh2d", str(thresh2d), "--thresh3d", str(thresh3d)]
    argv += ["--skip_viz"] * skip_viz + ["--force_cpu"] * force_cpu
    sys.argv = argv
    try:
        run_boxer.main()
    except OSError as e:  # the final overlay video needs ffmpeg; the boxes are already written
        if "ffmpeg" not in str(e):
            raise
        print(f"WARNING: {e} (overlay images were written, the mp4 was not)")
    return os.path.join(out_dir, BOXER_SEQ, "boxer_3dbbs.csv")


def fuse(csv_path, min_views, conf_threshold):
    add_boxer_to_path()
    from utils.fuse_3d_boxes import fuse_obbs_from_csv

    base, ext = os.path.splitext(csv_path)
    fused = f"{base}_fused{ext}"
    if os.path.exists(fused):  # Boxer writes nothing when no box survives; don't read a stale file
        os.remove(fused)
    fuse_obbs_from_csv(csv_path, min_detections=min_views, conf_threshold=conf_threshold)
    return fused


def boxes_to_json(fused_csv):
    """Boxer CSV -> list of {label, score, center, size, yaw}; size is (x, y, z) in the box frame,
    whose z axis is the world's (Boxer boxes are gravity-aligned)."""
    add_boxer_to_path()
    from utils.file_io import read_obb_csv

    boxes = []
    if not os.path.exists(fused_csv):
        return boxes
    for obbs in read_obb_csv(fused_csv).values():
        R = obbs.T_world_object.R
        centers, sizes = obbs.T_world_object.t, obbs.bb3_diagonal
        for i, label in enumerate(obbs.text_string()):
            boxes.append({
                "label": label, "score": round(float(obbs.prob[i]), 3),
                "center": [round(float(x), 4) for x in centers[i]],
                "size": [round(float(x), 4) for x in sizes[i]],
                "yaw": round(math.atan2(float(R[i, 1, 0]), float(R[i, 0, 0])), 4),
            })
    return sorted(boxes, key=lambda b: -b["score"])


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 2: 3D boxes with Boxer on Phase 1 output.")
    p.add_argument("--phase1", required=True, help="Phase 1 output folder")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", default="lvisplus",
                   help="Boxer taxonomy name or comma-separated prompts, e.g. bed,chair,table")
    p.add_argument("--thresh2d", type=float, default=0.25, help="OWLv2 confidence (Boxer default)")
    p.add_argument("--thresh3d", type=float, default=0.5, help="BoxerNet confidence (Boxer default)")
    p.add_argument("--min-views", type=int, default=2,
                   help="photos a box must be detected in to survive fusion (Boxer uses 4 for "
                        "dense video; 1-12 photos need fewer)")
    p.add_argument("--fuse-conf", type=float, default=0.55, help="fusion confidence (Boxer default)")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    n_photos = len([d for d in os.listdir(os.path.join(args.phase1, "boxer_input", BOXER_SEQ))
                    if d.endswith(".wide")])
    min_views = min(args.min_views, n_photos)  # a single photo can't need 2 views
    csv_path = run_boxer(args.phase1, args.out, args.labels, args.thresh2d, args.thresh3d,
                         args.skip_viz, args.force_cpu)
    boxes = boxes_to_json(fuse(csv_path, min_views, args.fuse_conf))
    with open(os.path.join(args.out, "boxes.json"), "w") as f:
        json.dump(boxes, f, indent=1)
    print(f"Phase 2: {len(boxes)} fused boxes -> {os.path.join(args.out, 'boxes.json')}")
    for b in boxes:
        print(f"  {b['label']:<24} {b['score']:.2f}  size {b['size']}")


if __name__ == "__main__":
    main()
