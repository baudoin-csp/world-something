"""Example 1: CA-1M capture val-42898570 (Apple's Cubify Anything; data licence CC BY-NC-ND 4.0).

  fetch   download and extract the capture with Boxer's own script (asks you to accept the licence)
  select  copy N evenly spaced frames into a plain photo folder: the pipeline's only input

Also holds the ground-truth readers used by evaluate.py. The pipeline itself never reads GT.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from world_something import BOXER_DIR, REPO_ROOT

VIDEO_ID, SPLIT = "42898570", "val"
CAPTURE = f"ca1m-{SPLIT}-{VIDEO_ID}"


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def fetch(data_root, video_id=VIDEO_ID, split=SPLIT, accept_license=False):
    script = os.path.join(BOXER_DIR, "scripts", "download_ca1m_sample.py")
    cmd = [sys.executable, script, "--video-id", video_id, "--split", split, "--output-dir", data_root]
    subprocess.run(cmd, input="y\n" if accept_license else None, text=True, check=True)
    return os.path.join(data_root, f"ca1m-{split}-{video_id}")


def find_data_dir(capture_dir):
    hits = glob.glob(os.path.join(capture_dir, "**", "world.gt", "instances.json"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no CA-1M capture under {capture_dir}; run: python -m "
                                f"world_something.ca1m fetch")
    return os.path.dirname(os.path.dirname(hits[0]))


def frame_tags(data_dir):
    tags = {os.path.basename(d).split(".")[0] for d in glob.glob(os.path.join(data_dir, "*.wide"))}
    return sorted(tags, key=int)


def select(capture_dir, out_dir, num_frames):
    data_dir = find_data_dir(capture_dir)
    tags = frame_tags(data_dir)
    idx = np.unique(np.linspace(0, len(tags) - 1, num_frames).round().astype(int))
    chosen = [tags[i] for i in idx]
    img_dir = os.path.join(out_dir, "images")
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)
    for tag in chosen:
        shutil.copy(os.path.join(data_dir, f"{tag}.wide", "image.png"), os.path.join(img_dir, f"{tag}.png"))
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump({"data_dir": os.path.abspath(data_dir), "tags": chosen}, f, indent=1)
    print(f"selected {len(chosen)} of {len(tags)} frames -> {img_dir}")
    return img_dir


# --- Ground truth (evaluation only) --------------------------------------------------------


def load_gt_frame(data_dir, tag):
    """GT intrinsics, camera-to-world pose (laser-scanner world, z up) and laser depth (m)."""
    wide, gt = os.path.join(data_dir, f"{tag}.wide"), os.path.join(data_dir, f"{tag}.gt")
    with Image.open(os.path.join(wide, "image.png")) as im:
        width, height = im.size
    depth = np.asarray(Image.open(os.path.join(gt, "depth.png")), dtype=np.float32) / 1000.0
    K = np.array(_read_json(os.path.join(wide, "image", "K.json")), dtype=np.float64)
    # GT depth covers the same field of view as the image at a lower resolution.
    K_depth = np.diag([depth.shape[1] / width, depth.shape[0] / height, 1.0]) @ K
    return {
        "K": K, "size": (width, height),
        "c2w": np.array(_read_json(os.path.join(gt, "RT.json")), dtype=np.float64),
        "depth": depth, "K_depth": K_depth,
        "visible_ids": [b["id"] for b in _read_json(os.path.join(wide, "instances.json"))],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download + extract the capture")
    f.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    f.add_argument("--accept-license", action="store_true",
                   help="accept CA-1M's CC BY-NC-ND 4.0 licence without the interactive prompt")
    s = sub.add_parser("select", help="copy N evenly spaced frames into <out>/images")
    s.add_argument("--capture", default=os.path.join(REPO_ROOT, "data", CAPTURE))
    s.add_argument("--out", required=True)
    s.add_argument("--num-frames", type=int, default=12)
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.data_root, accept_license=args.accept_license)
    else:
        select(args.capture, args.out, args.num_frames)


if __name__ == "__main__":
    main()
