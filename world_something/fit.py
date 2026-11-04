"""Numerical box fitting for Phase 3's agent: the VLM says what to fix, the optimizer places it.

Two fits, both gravity-aligned 7-DoF boxes (centre, size, yaw):
  * fit_to_marks: match the box's projected rectangle to rectangles the agent marked on the
    object in several photos; the fit is kept only if it also matches a held-out photo.
  * fit_to_points: tightest box around the Phase 1 points near the object.
"""

import math

import numpy as np
from scipy.optimize import least_squares

from world_something.cameras import project
from world_something.scene import Obj, yaw_matrix


def rect_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def box_corners(center, size, yaw):
    o = Obj(id="_", label="_", category="_", center=list(center), size=list(size), yaw=float(yaw))
    return o.corners()


def projected_rect(cam, center, size, yaw):
    """Image rectangle [u0, v0, u1, v1] of a box, clipped to the photo; None if behind."""
    uv, z = project(cam.K, cam.c2w, box_corners(center, size, yaw))
    if np.all(z <= 0.05):
        return None
    uv = uv[z > 0.05]
    u0, v0 = uv.min(axis=0)
    u1, v1 = uv.max(axis=0)
    return [float(np.clip(u0, 0, cam.width)), float(np.clip(v0, 0, cam.height)),
            float(np.clip(u1, 0, cam.width)), float(np.clip(v1, 0, cam.height))]


def fit_to_marks(obj, marks, cameras, holdout=None, keep_bottom=True, points=None,
                 accept_iou=0.5, max_scale_change=3.0):
    """Fit obj's box to marked rectangles. marks: {view_index: [u0, v0, u1, v1]} in photo pixels.

    One marked view is held out (the last one, unless `holdout` names it); the fit is accepted
    only if its projection overlaps that view's mark by at least accept_iou.
    """
    views = sorted(marks)
    if holdout is None and len(views) >= 3:
        holdout = views[-1]
    fit_views = [v for v in views if v != holdout]
    if not fit_views:
        return {"accepted": False, "reason": "need at least one marked view besides the held-out one"}
    bottom0, yaw0 = obj.bottom, obj.yaw
    size0 = np.maximum(np.asarray(obj.size, float), 0.02)

    def unpack(p):
        cx, cy, cz, lx, ly, lz, yaw = p
        size = size0 * np.exp([lx, ly, lz])
        if keep_bottom:
            cz = bottom0 + size[2] / 2.0
        return np.array([cx, cy, cz]), size, yaw

    evidence = None
    if points is not None and len(points):
        near = Obj(id="_", label="_", category="_", center=obj.center, size=list(size0 * 1.6 + 0.2),
                   yaw=obj.yaw).contains(points)
        evidence = points[near]
        for v in fit_views:  # drop points a marked view sees outside its mark (objects may be cut
            cam = cameras[v]  # by the image border, so points outside the image are kept)
            uv, z = project(cam.K, cam.c2w, evidence)
            r = marks[v]
            pad = 0.05 * max(r[2] - r[0], r[3] - r[1])
            in_img = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
            inside = (uv[:, 0] >= r[0] - pad) & (uv[:, 0] <= r[2] + pad) & (uv[:, 1] >= r[1] - pad) & (uv[:, 1] <= r[3] + pad)
            evidence = evidence[~(in_img & ~inside)]
        if len(evidence) > 4000:
            evidence = evidence[np.random.default_rng(0).choice(len(evidence), 4000, replace=False)]

    def residuals(p):
        center, size, yaw = unpack(p)
        res = []
        for v in fit_views:
            cam = cameras[v]
            r = projected_rect(cam, center, size, yaw)
            diag = math.hypot(cam.width, cam.height)
            if r is None:
                res += [1.0] * 4
            else:
                res += [(a - b) / diag for a, b in zip(r, marks[v])]
        if evidence is not None and len(evidence) > 20:
            local = (evidence - center) @ yaw_matrix(yaw)
            outside = np.maximum(np.abs(local) - size / 2.0, 0.0)
            res.append(0.5 * float(np.sqrt(np.mean(np.sum(outside ** 2, axis=1)))))
        res.append(0.02 * (yaw - yaw0))                  # stay near the agent's orientation
        res += list(0.01 * np.asarray(p[3:6]))            # mild preference for the old size
        return np.asarray(res, float)

    p0 = np.array(list(obj.center) + [0.0, 0.0, 0.0, obj.yaw], float)
    bound = math.log(max_scale_change)
    lb = [-np.inf, -np.inf, -np.inf, -bound, -bound, -bound, -np.inf]
    ub = [np.inf, np.inf, np.inf, bound, bound, bound, np.inf]
    sol = least_squares(residuals, p0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01)
    center, size, yaw = unpack(sol.x)

    def mean_iou(c, s, y, vs):
        ious = []
        for v in vs:
            r = projected_rect(cameras[v], c, s, y)
            ious.append(0.0 if r is None else rect_iou(r, marks[v]))
        return float(np.mean(ious)) if ious else None

    before = mean_iou(obj.center, obj.size, obj.yaw, fit_views)
    after = mean_iou(center, size, yaw, fit_views)
    held = mean_iou(center, size, yaw, [holdout]) if holdout is not None else None
    accepted = after is not None and (held is None or held >= accept_iou)
    if accepted:
        obj.center, obj.size, obj.yaw = [float(x) for x in center], [float(x) for x in size], float(yaw)
    return {"accepted": bool(accepted), "fit_views": fit_views, "holdout_view": holdout,
            "fit_iou_before": before, "fit_iou_after": after, "holdout_iou": held,
            "evidence_points": 0 if evidence is None else int(len(evidence)),
            "center": [round(float(x), 3) for x in center], "size": [round(float(x), 3) for x in size],
            "yaw_deg": round(math.degrees(yaw), 1),
            "reason": None if accepted else f"held-out IoU {held:.2f} < {accept_iou}" if held is not None else "fit failed"}


def box_around_points(pts, yaw0=0.0, sweep_deg=45.0, step_deg=1.5, pct=(2, 98), z_pct=(2, 99)):
    """Tightest gravity-aligned box (center, size, yaw) around points: yaw sweep, robust extents."""
    best = None
    for deg in np.arange(-sweep_deg, sweep_deg, step_deg):
        yaw = yaw0 + math.radians(deg)
        local = pts[:, :2] @ yaw_matrix(yaw)[:2, :2]
        lo, hi = np.percentile(local, pct, axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, yaw, lo, hi)
    _, yaw, lo, hi = best
    z_lo, z_hi = np.percentile(pts[:, 2], z_pct)
    cxy = yaw_matrix(yaw)[:2, :2] @ ((lo + hi) / 2.0)
    center = [float(cxy[0]), float(cxy[1]), float((z_lo + z_hi) / 2.0)]
    size = [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(max(z_hi - z_lo, 0.02))]
    return center, size, float(yaw)


def fit_to_points(obj, points, margin=0.15, keep_bottom=True, min_points=50):
    """Tightest gravity-aligned box around the Phase 1 points near obj."""
    region = Obj(id="_", label="_", category="_", center=obj.center,
                 size=[s + 2 * margin for s in obj.size], yaw=obj.yaw)
    pts = points[region.contains(points)]
    if len(pts) < min_points:
        return {"accepted": False, "reason": f"only {len(pts)} points near the box"}
    center, size, yaw = box_around_points(pts, obj.yaw)
    if keep_bottom:
        top = center[2] + size[2] / 2.0
        size[2] = max(top - obj.bottom, 0.02)
        center[2] = obj.bottom + size[2] / 2.0
    obj.center, obj.size, obj.yaw = center, size, yaw
    return {"accepted": True, "points": int(len(pts)), "center": [round(x, 3) for x in obj.center],
            "size": [round(x, 3) for x in obj.size], "yaw_deg": round(math.degrees(yaw), 1)}
