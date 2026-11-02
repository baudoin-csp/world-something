"""A synthetic room with exact cameras and depth, to test everything except the networks."""

import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

ROOM_MIN, ROOM_MAX = np.array([0.0, 0.0, 0.0]), np.array([4.0, 5.0, 2.6])  # floor at z = 0
FULL_W, FULL_H, FOCAL = 640, 480, 500.0
PROC_W, PROC_H = 504, 378  # what DA3 would process a 640x480 photo at


def look_at(eye, target):
    """OpenCV camera-to-world looking from eye to target in a z-up world."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    c2w[:3, 3] = eye
    return c2w


def cameras():
    eyes = [(0.6, 0.6, 1.5), (3.4, 0.7, 1.4), (3.3, 4.3, 1.6), (0.7, 4.2, 1.5), (2.0, 1.0, 1.3), (2.2, 4.0, 1.7)]
    targets = [(2.5, 3.0, 0.5), (1.0, 3.5, 0.6), (1.2, 1.0, 0.4), (3.0, 1.5, 0.7), (2.0, 4.5, 0.8), (2.0, 0.5, 0.6)]
    return np.stack([look_at(np.array(e), t) for e, t in zip(eyes, targets)])


def intrinsics(w, h):
    return np.array([[FOCAL * w / FULL_W, 0, w / 2], [0, FOCAL * h / FULL_H, h / 2], [0, 0, 1.0]])


def render_depth(c2w, K, w, h):
    """z-depth of the room's inner walls, floor and ceiling (ray / box-exit intersection)."""
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])  # z = 1
    d = c2w[:3, :3] @ rays
    o = c2w[:3, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, (ROOM_MAX[:, None] - o) / d, (ROOM_MIN[:, None] - o) / d)
    t = np.where(np.isfinite(t) & (t > 0), t, np.inf).min(axis=0)
    return t.reshape(h, w).astype(np.float32)  # rays have z = 1, so t is the z-depth


def fake_da3_prediction():
    """What DA3 returns: poses in the first camera's (tilted) frame, data at processing res."""
    c2w_gt = cameras()
    T_da3_world = np.linalg.inv(c2w_gt[0])
    K = intrinsics(PROC_W, PROC_H)
    rng = np.random.default_rng(0)
    photos = [rng.integers(0, 255, (FULL_H, FULL_W, 3), dtype=np.uint8) for _ in c2w_gt]
    pred = SimpleNamespace(
        depth=np.stack([render_depth(c, K, PROC_W, PROC_H) for c in c2w_gt]),
        conf=np.ones((len(c2w_gt), PROC_H, PROC_W), np.float32),
        intrinsics=np.repeat(K[None], len(c2w_gt), 0),
        extrinsics=np.linalg.inv(T_da3_world @ c2w_gt)[:, :3, :4],  # w2c, (N, 3, 4)
        processed_images=np.stack([cv2.resize(p, (PROC_W, PROC_H)) for p in photos]),
        is_metric=1, scale_factor=1.0,
    )
    return pred, photos, c2w_gt


# Ground-truth boxes of the synthetic room: (category, centre, size xyz, yaw)
GT_BOXES = [
    ("table", (2.0, 2.5, 0.375), (1.2, 0.8, 0.75), 0.3),
    ("chair", (1.2, 2.0, 0.45), (0.5, 0.5, 0.9), -0.4),
    ("cup", (2.1, 2.6, 0.8), (0.08, 0.08, 0.1), 0.0),
    ("floor", (2.0, 2.5, -0.01), (4.0, 5.0, 0.02), 0.0),
]


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def write_fake_ca1m(root, tags, gt_w=320, gt_h=240):
    """A capture in CA-1M's on-disk layout with the synthetic cameras, laser depth and boxes."""
    os.makedirs(os.path.join(root, "world.gt"), exist_ok=True)
    boxes = [{"id": f"id{i}", "category": c, "position": list(p), "scale": list(s),
              "R": yaw_matrix(y).tolist()} for i, (c, p, s, y) in enumerate(GT_BOXES)]
    with open(os.path.join(root, "world.gt", "instances.json"), "w") as f:
        json.dump(boxes, f)
    K_full = intrinsics(FULL_W, FULL_H)
    for tag, c2w in zip(tags, cameras()):
        wide, gt = os.path.join(root, f"{tag}.wide"), os.path.join(root, f"{tag}.gt")
        os.makedirs(os.path.join(wide, "image"), exist_ok=True)
        os.makedirs(gt, exist_ok=True)
        Image.fromarray(np.zeros((FULL_H, FULL_W, 3), np.uint8)).save(os.path.join(wide, "image.png"))
        depth = render_depth(c2w, intrinsics(gt_w, gt_h), gt_w, gt_h)
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(os.path.join(gt, "depth.png"))
        for path, obj in [(os.path.join(wide, "image", "K.json"), K_full.tolist()),
                          (os.path.join(gt, "RT.json"), c2w.tolist()),
                          (os.path.join(wide, "instances.json"), [{"id": b["id"]} for b in boxes])]:
            with open(path, "w") as f:
                json.dump(obj, f)
    return root
