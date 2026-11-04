"""Camera math shared by Phases 3-7: OpenCV <-> MuJoCo, projection, paths, alignment, Plücker rays.

Convention everywhere: K is 3x3 at the image's own resolution, poses are camera-to-world (c2w)
with OpenCV axes (x right, y down, z forward), world z up, metres.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CV_TO_GL = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> MuJoCo/OpenGL camera axes


def as4(m):
    m = np.asarray(m, float)
    if m.shape == (4, 4):
        return m.copy()
    out = np.eye(4)
    out[:3, :4] = m[:3, :4]
    return out


def scale_K(K, sx, sy):
    K = np.asarray(K, float).copy()
    K[0] *= sx
    K[1] *= sy
    return K


def project(K, c2w, points):
    """World points (N, 3) -> pixels (N, 2) and camera-frame depth (N,)."""
    w2c = np.linalg.inv(as4(c2w))
    pc = np.asarray(points, float) @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = (pc @ np.asarray(K, float).T)[:, :2] / np.where(np.abs(z) > 1e-9, z, 1e-9)[:, None]
    return uv, z


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """OpenCV camera-to-world looking from eye to target in a z-up world."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight up or down
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = eye
    return c2w


def interpolate_poses(keys, n):
    """n camera-to-world poses through the key poses (slerp rotations, cubic-ish positions)."""
    keys = [as4(k) for k in keys]
    if len(keys) == 1:
        return [keys[0].copy() for _ in range(n)]
    t_keys = np.linspace(0.0, 1.0, len(keys))
    rots = Rotation.from_matrix(np.stack([k[:3, :3] for k in keys]))
    slerp = Slerp(t_keys, rots)
    pos = np.stack([k[:3, 3] for k in keys])
    ts = np.linspace(0.0, 1.0, n)
    out_pos = np.stack([np.interp(ts, t_keys, pos[:, d]) for d in range(3)], axis=1)
    # Catmull-Rom smoothing of the position track when there are enough keys
    if len(keys) >= 3:
        from scipy.interpolate import CubicSpline
        out_pos = CubicSpline(t_keys, pos, bc_type="natural")(ts)
    out = []
    for t, p in zip(ts, out_pos):
        m = np.eye(4)
        m[:3, :3] = slerp([t]).as_matrix()[0]
        m[:3, 3] = p
        out.append(m)
    return out


def align_to_first(c2w_list, first_c2w):
    """Re-express poses in the first context view's camera frame: T' = T1^-1 T."""
    inv = np.linalg.inv(as4(first_c2w))
    return [inv @ as4(c) for c in c2w_list]


def plucker_rays(K, c2w, height, width):
    """(6, H, W) Plücker coordinates (o x d, d) of every pixel's ray, d normalised, world frame."""
    K, c2w = np.asarray(K, float), as4(c2w)
    u, v = np.meshgrid(np.arange(width) + 0.0, np.arange(height) + 0.0)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(K).T
    d = rays @ c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(c2w[:3, 3], d.shape)
    m = np.cross(o, d)
    return np.concatenate([m, d], axis=-1).transpose(2, 0, 1).astype(np.float32)


def orbit_path(center, radius, height, n, start_deg=0.0, sweep_deg=360.0, look_height=None):
    look = np.array([center[0], center[1], height if look_height is None else look_height])
    out = []
    for a in np.linspace(math.radians(start_deg), math.radians(start_deg + sweep_deg), n):
        eye = [center[0] + radius * math.cos(a), center[1] + radius * math.sin(a), height]
        out.append(look_at(eye, look))
    return out


def dolly_path(c2w, distance, n, target=None):
    """Move forward along the camera's optical axis by `distance` metres (negative = back)."""
    c2w = as4(c2w)
    fwd = c2w[:3, 2]
    out = []
    for s in np.linspace(0.0, distance, n):
        m = c2w.copy()
        m[:3, 3] = c2w[:3, 3] + s * fwd
        if target is not None:
            m = look_at(m[:3, 3], target)
        out.append(m)
    return out
