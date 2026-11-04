"""Example 1: CA-1M capture val-42898570 (Apple's Cubify Anything; data licence CC BY-NC-ND 4.0).

  fetch   download and extract the capture with Boxer's own script (asks you to accept the licence)
  select  copy N evenly spaced frames into a plain photo folder: the pipeline's only input
  oracle  write ground truth for the selected frames in Phase 1 and Phase 2's output formats, so
          Phases 3-8 can be developed and debugged without a GPU (or with perfect inputs)

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


# --- Oracle Phase 1-2 outputs (development only) ---------------------------------------------


def _gravity_box(position, scale, R):
    """CA-1M OBB -> gravity-aligned (center, size, yaw): the box axis closest to world z is up."""
    R, scale = np.asarray(R, float), np.asarray(scale, float)
    up = int(np.argmax(np.abs(R[2])))
    x = [i for i in range(3) if i != up][0]
    y = [i for i in range(3) if i not in (up, x)][0]
    yaw = float(np.arctan2(R[1, x], R[0, x]))
    return [float(v) for v in position], [float(scale[x]), float(scale[y]), float(scale[up])], yaw


def _floor_height(z):
    """Height of the lowest dense horizontal band of points (the floor)."""
    lo = np.percentile(z, 0.5)
    band = z[(z > lo - 0.05) & (z < lo + 0.10)]
    hist, edges = np.histogram(band, bins=np.arange(lo - 0.05, lo + 0.101, 0.01))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2.0)


# Small wall fixtures and annotations Boxer's taxonomy does not produce as objects.
ORACLE_SKIP = {"wall", "object", "electrical outlet", "light switch", "plug", "motion sensor",
               "door frame"}


def oracle(capture_dir, frames_json, out_dir, max_depth=10.0):
    """phase1/ (cameras.json, geometry.npz) and phase2/ (boxes.json) from CA-1M ground truth.

    The world is shifted so that the floor is z = 0, as Phase 1 does; poses and depth are GT.
    """
    sel = _read_json(frames_json)
    data_dir, tags = sel.get("data_dir") or find_data_dir(capture_dir), sel["tags"]
    img_dir = os.path.join(os.path.dirname(os.path.abspath(frames_json)), "images")
    frames = [load_gt_frame(data_dir, t) for t in tags]
    pts = []
    for fr in frames:
        d = fr["depth"]
        v, u = np.nonzero((d > 0) & (d < max_depth))
        rays = np.linalg.inv(fr["K_depth"]) @ np.stack([u, v, np.ones_like(u)]).astype(float)
        pts.append((fr["c2w"][:3, :3] @ (rays * d[v, u]) + fr["c2w"][:3, 3:4]).T)
    floor = _floor_height(np.concatenate(pts)[:, 2])
    shift = np.eye(4)
    shift[2, 3] = -floor
    p1, p2 = os.path.join(out_dir, "phase1"), os.path.join(out_dir, "phase2")
    os.makedirs(p1, exist_ok=True)
    os.makedirs(p2, exist_ok=True)
    cams = {"world": {"units": "m", "up": [0, 0, 1], "floor_z": 0.0,
                      "camera_convention": "OpenCV, camera-to-world"},
            "model": "ca1m-ground-truth", "is_metric": True, "metric_scale_factor": 1.0,
            "gravity": {"source": "CA-1M laser-scanner world (z up)", "floor_offset_m": floor},
            "frames": []}
    depth, Ks, c2ws, Kf = [], [], [], []
    for tag, fr in zip(tags, frames):
        c2w = shift @ fr["c2w"]
        w, h = fr["size"]
        cams["frames"].append({"name": tag, "file": os.path.join(img_dir, f"{tag}.png"),
                               "width": w, "height": h, "K": fr["K"].tolist(), "c2w": c2w.tolist()})
        d = np.where(fr["depth"] < max_depth, fr["depth"], 0.0)
        depth.append(d)
        Ks.append(fr["K_depth"])
        c2ws.append(c2w)
        Kf.append(fr["K"])
    with open(os.path.join(p1, "cameras.json"), "w") as f:
        json.dump(cams, f, indent=1)
    np.savez_compressed(os.path.join(p1, "geometry.npz"), names=np.array(tags),
                        depth=np.stack(depth).astype(np.float32), K=np.stack(Ks),
                        c2w=np.stack(c2ws), K_full=np.stack(Kf))
    visible = set()
    for fr in frames:
        visible.update(fr["visible_ids"])
    boxes = []
    for inst in _read_json(os.path.join(data_dir, "world.gt", "instances.json")):
        if inst["id"] not in visible or inst["category"] in ORACLE_SKIP:
            continue
        center, size, yaw = _gravity_box(inst["position"], inst["scale"], inst["R"])
        center[2] -= floor
        boxes.append({"label": inst["category"], "score": 1.0, "center": [round(c, 4) for c in center],
                      "size": [round(s, 4) for s in size], "yaw": round(yaw, 4), "gt_id": inst["id"]})
    with open(os.path.join(p2, "boxes.json"), "w") as f:
        json.dump(boxes, f, indent=1)
    print(f"oracle: {len(tags)} frames, floor at {floor:.3f} m, {len(boxes)} boxes -> {out_dir}")
    return p1, p2


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
    o = sub.add_parser("oracle", help="ground truth as Phase 1/2 outputs (<out>/phase1, <out>/phase2)")
    o.add_argument("--capture", default=os.path.join(REPO_ROOT, "data", CAPTURE))
    o.add_argument("--frames", required=True, help="frames.json written by `select`")
    o.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.data_root, accept_license=args.accept_license)
    elif args.cmd == "select":
        select(args.capture, args.out, args.num_frames)
    else:
        oracle(args.capture, args.frames, args.out)


if __name__ == "__main__":
    main()
