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


# --- camera paths ---------------------------------------------------------------------------------


def action_centroid(traj, objects):
    pts = []
    for oid in objects:
        p = traj["poses"][oid][:, :3]
        pts.append(p[np.all(np.abs(p) < 500, axis=1)])
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    return pts.mean(axis=0) if len(pts) else None


def camera_path(spec, scene, traj, n, long_side=640, aspect=None):
    """(K, [c2w per frame], width, height) for a path spec:

      auto                  a 50-degree orbit around the action (landscape), inside the room and
                            clear of every box; else the photo camera that sees it best,
                            dollying back with the gaze on the action
      static:I              photo I's camera, fixed
      dolly:I:D             photo I's camera moving D metres forward (negative = back)
      photos:I,J,K          a smooth path through photo cameras I, J, K
      orbit[:R[:H]]         circle the action at radius R (m) and height H (m)
      file:PATH             JSON {"K": 3x3, "width": w, "height": h, "c2w": [4x4 per key]}
    """
    from world_something.cameras import dolly_path, orbit_path

    moved = moved_objects(traj)
    kind, _, arg = spec.partition(":")
    if kind == "file":
        with open(arg) as f:
            d = json.load(f)
        keys = [np.asarray(c, float) for c in d["c2w"]]
        poses = keys if len(keys) == n else interpolate_poses(keys, n)
        return np.asarray(d["K"], float), poses, int(d["width"]), int(d["height"])
    if kind == "orbit":
        c = action_centroid(traj, moved) if moved else np.asarray(scene.get("floor").center)
        parts = [float(x) for x in arg.split(":") if x] if arg else []
        radius = parts[0] if parts else 2.0
        height = parts[1] if len(parts) > 1 else 1.6
        cam = scene.cameras[0]
        K, _, w, h = photo_view(cam, long_side)
        return K, orbit_path(c, radius, height, n, sweep_deg=120.0, look_height=max(c[2], 0.5)), w, h
    if kind == "photos":
        idx = [int(i) for i in arg.split(",")]
        K, _, w, h = photo_view(scene.cameras[idx[0]], long_side)
        return K, interpolate_poses([as4(scene.cameras[i].c2w) for i in idx], n), w, h
    if kind == "auto":
        center = action_centroid(traj, moved)
        if center is not None:
            K, _, w, h = photo_view(scene.cameras[0], long_side)
            if w < h:  # a moving camera reads better in landscape
                K, w, h = np.array([[K[1, 1], 0, h / 2.0], [0, K[1, 1], w / 2.0], [0, 0, 1]]), h, w
            path = safe_orbit(scene, center, n)
            if path is not None:
                return K, path, w, h
    i = int(arg.split(":")[0]) if kind in ("static", "dolly") else best_photo_camera(scene, traj, moved or list(traj["poses"])[:1])
    K, c2w, w, h = photo_view(scene.cameras[i], long_side)
    if kind == "static":
        return K, [c2w] * n, w, h
    if kind == "dolly":
        return K, dolly_path(c2w, float(arg.split(":")[1]), n), w, h
    # auto: dolly back 0.6 m while looking at the action
    target = action_centroid(traj, moved)
    poses = dolly_path(c2w, -0.6, n)
    if target is not None:
        start_gaze = c2w[:3, 3] + c2w[:3, 2] * max(np.linalg.norm(target - c2w[:3, 3]), 1.0)
        poses = [look_at(p[:3, 3], (1 - s) * start_gaze + s * target)
                 for p, s in zip(poses, np.linspace(0.0, 1.0, n) ** 0.5)]
    return K, poses, w, h


def safe_orbit(scene, center, n, radii=(2.6, 2.2, 1.8, 1.5), height=1.55, sweep_deg=50.0, margin=0.25):
    """An orbit around `center` whose every position is inside the room and outside every box."""
    from world_something.cameras import orbit_path

    floor = scene.get("floor") if "floor" in scene else None
    blockers = [o for o in scene.objects if o.proxy_class not in ("floor", "ceiling")]
    look = max(float(center[2]), 0.6)
    for r in radii:
        for start in np.arange(0.0, 360.0, 15.0):
            path = orbit_path(center, r, height, n, start_deg=start, sweep_deg=sweep_deg, look_height=look)
            pos = np.array([p[:3, 3] for p in path])
            if floor is not None:  # inside the floor's footprint, `margin` from its edges
                from world_something.scene import yaw_matrix
                local = (pos[:, :2] - np.asarray(floor.center[:2])) @ yaw_matrix(floor.yaw)[:2, :2]
                if not np.all(np.abs(local) <= np.asarray(floor.size[:2]) / 2.0 - margin):
                    continue
            if any(np.any(o.contains(pos, margin=margin)) for o in blockers):
                continue
            return path
    return None


# --- chunks and context photos --------------------------------------------------------------------


def chunk_frames(n_frames, chunk=CHUNK, prefix=PREFIX):
    """Smallest chunk-aligned frame count (chunk + k * (chunk - prefix)) >= n_frames."""
    k = max(0, math.ceil((n_frames - chunk) / (chunk - prefix)))
    return chunk + k * (chunk - prefix)


def chunk_ranges(n_frames, chunk=CHUNK, prefix=PREFIX):
    """[(start, end)] frame ranges: chunk frames each, consecutive chunks share `prefix` frames."""
    step = chunk - prefix
    starts = [0]
    while starts[-1] + chunk < n_frames:
        starts.append(starts[-1] + step)
    return [(s, s + chunk) for s in starts]


def visible_points(depth_frames, obj_frames, K, c2ws, stride=8):
    """World points of the proxy surfaces seen in the frames (subsampled)."""
    pts = []
    Kinv = np.linalg.inv(K)
    for d, o, c2w in zip(depth_frames, obj_frames, c2ws):
        v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[v, u]
        ok = (z > 0) & (o[v, u] >= 0)
        rays = Kinv @ np.stack([u[ok] + 0.5, v[ok] + 0.5, np.ones(ok.sum())])
        pts.append((c2w[:3, :3] @ (rays * z[ok]) + c2w[:3, 3:4]).T)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def select_context(world, scene, points, max_views=MAX_CONTEXT, long_side=320, min_gain=0.01, min_views=3):
    """Greedy set cover: photos that see the most of the chunk's surfaces (with occlusion)."""
    if not len(points):
        return [0]
    seen = []
    for cam in scene.cameras:
        K, c2w, w, h = photo_view(cam, long_side)
        d = world.cs.render(K, c2w, w, h, depth=True)["depth"]
        uv, z = project(K, c2w, points)
        ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        vis = np.zeros(len(points), bool)
        ui, vi = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        vis[np.nonzero(ok)[0]] = np.abs(d[vi, ui] - z[ok]) < np.maximum(0.05 * z[ok], 0.05)
        seen.append(vis)
    covered = np.zeros(len(points), bool)
    chosen = []
    for _ in range(min(max_views, len(scene.cameras))):
        gains = [(~covered & s).mean() if i not in chosen else -1 for i, s in enumerate(seen)]
        best = int(np.argmax(gains))
        if gains[best] < min_gain and len(chosen) >= min_views:
            break
        chosen.append(best)
        covered |= seen[best]
    return chosen


def write_video(path, frames, fps):
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=8,
                            pixelformat="yuv420p", ffmpeg_params=["-crf", "8"]) as w:
        for f in frames:
            w.append_data(np.asarray(f, np.uint8))
