"""Phase 6: proxy videos. Flat-colour renders of the simulated world along a camera path, cut into
the chunks the renderer (Phase 7) generates, each with its context photos.

Mundo's conventions, used here:
  * proxy palette per pixel (mjscene.PALETTE): what is where, not what it looks like
  * camera poses re-expressed relative to the first context view of each chunk
  * chunks of 124 frames that overlap by a 34-frame prefix (the renderer continues from it)
  * at most 9 context photos per chunk (MiniMax-H3 Ref2VA takes up to 9 reference images)
"""

import json
import math
import os

import numpy as np

from world_something.cameras import align_to_first, as4, interpolate_poses, look_at, plucker_rays, project, scale_K

CHUNK = 124      # frames per renderer chunk (17n + 5 with n = 7: 37 latent frames)
PREFIX = 34      # frames each chunk shares with the previous one
MAX_CONTEXT = 9  # reference photos per chunk


def render_frames(world, qpos_frames, K, c2ws, width, height, depth=False):
    """Proxy frames (and object ids) of recorded states; c2ws: one pose, or one per frame."""
    import mujoco

    cs, m, d = world.cs, world.m, world.d
    if not isinstance(c2ws, (list, tuple)):
        c2ws = [c2ws] * len(qpos_frames)
    out = {"proxy": [], "obj": [], "depth": []}
    saved = d.qpos.copy()
    for q, c2w in zip(qpos_frames, c2ws):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        r = cs.render(K, c2w, width, height, depth=depth)
        out["proxy"].append(r["proxy"])
        out["obj"].append(r["obj"].astype(np.int16))
        if depth:
            out["depth"].append(r["depth"])
    d.qpos[:] = saved
    mujoco.mj_kinematics(m, d)
    return out


def moved_objects(traj, tol=0.05):
    """Ids of objects whose root moved more than tol (m) during the recording."""
    out = []
    for oid, p in traj["poses"].items():
        pos = p[:, :3]
        if np.any(np.abs(pos) > 500):  # spawned objects waiting off-stage
            pos = pos[np.all(np.abs(pos) < 500, axis=1)]
        if len(pos) and np.linalg.norm(pos - pos[0], axis=1).max() > tol:
            out.append(oid)
    return out


def best_photo_camera(scene, traj, objects):
    """Index of the photo camera that sees the most of the objects' trajectories."""
    best, best_score = 0, -1
    pts = []
    for oid in objects:
        p = traj["poses"][oid][:, :3]
        pts.append(p[np.all(np.abs(p) < 500, axis=1)])
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    for i, cam in enumerate(scene.cameras):
        if not len(pts):
            return 0
        uv, z = project(cam.K, cam.c2w, pts)
        inside = (z > 1.0) & (uv[:, 0] > 0) & (uv[:, 0] < cam.width) & (uv[:, 1] > 0) & (uv[:, 1] < cam.height)
        score = inside.mean() + 0.05 * float(np.median(np.clip(z, 0, 6)))  # prefer seeing it whole
        if score > best_score:
            best, best_score = i, score
    return best


def photo_view(cam, long_side=640):
    s = long_side / max(cam.width, cam.height)
    w, h = int(round(cam.width * s)), int(round(cam.height * s))
    return scale_K(cam.K, w / cam.width, h / cam.height), as4(cam.c2w), w, h
