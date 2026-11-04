"""Phase 3 agent: Claude Opus 5.5 edits the proxy world until its renders match the photos.

Analysis by synthesis, as in Mundo: render the proxy from every photo's camera, overlay it on the
photo, compare, edit, repeat. Each round runs two roles, each in a fresh conversation:
  * reviewer (read-only): inspects overlays and depth-error maps, then accepts or lists findings
  * editor: fixes the findings with scene tools. Numeric placement is done by fit.py: the VLM
    marks where an object is in a few photos, least squares places the box, and a held-out photo
    has to agree before the change is kept.
The loop stops when the reviewer accepts, depth agreement stops improving, or the round, edit or
time budget runs out. Every request, tool call and result (without images) is logged to disk.

Requests go to Claude Opus 5.5 with server-side refusal fallbacks on (fallbacks="default"): if a
request is declined by a safety classifier, the API re-runs it on Anthropic's fallback model.
"""

import base64
import copy
import io
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageOps

from world_something.cameras import project, scale_K
from world_something.fit import box_around_points, fit_to_marks, fit_to_points, projected_rect
from world_something.mjscene import CompiledScene, draw_ruler, instance_colors, overlay
from world_something.scene import (STRUCTURE, Camera, Obj, Scene, make_object, normalize_category,
                                   proxy_class_for)

MODEL = os.environ.get("WS_AGENT_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


# --- workspace: the scene plus everything needed to render and score it -------------------------


class Workspace:
    def __init__(self, scene, points, phase1_dir, max_side=768, rel_tol=0.1, abs_tol=0.05):
        self.scene, self.points, self.rel_tol, self.abs_tol = scene, points, rel_tol, abs_tol
        geo = np.load(os.path.join(phase1_dir, "geometry.npz"))
        self.depth, self.Kd = geo["depth"], geo["K"]
        self.views, self.photos = [], []
        for cam in scene.cameras:
            s = max_side / max(cam.width, cam.height)
            w, h = int(round(cam.width * s)), int(round(cam.height * s))
            K = scale_K(cam.K, w / cam.width, h / cam.height)
            self.views.append(Camera(cam.name, cam.image, w, h, K.tolist(), cam.c2w))
            img = ImageOps.exif_transpose(Image.open(cam.image)).convert("RGB")
            self.photos.append(np.asarray(img.resize((w, h), Image.LANCZOS)))
        self._cs = None
        self.edits = 0

    # rendering --------------------------------------------------------------------------------
    def compiled(self):
        if self._cs is None:
            self._cs = CompiledScene(self.scene)
        return self._cs

    def changed(self):
        if self._cs is not None:
            self._cs.close()
        self._cs = None
        self.edits += 1

    def depth_check(self, i):
        """Proxy depth vs Phase 1 depth in view i: agreement and the two kinds of error."""
        d = self.depth[i]
        h, w = d.shape
        r = self.compiled().render(self.Kd[i], self.scene.cameras[i].c2w, w, h, depth=True)
        dp, obj = r["depth"], r["obj"]
        valid = d > 0
        tol = np.maximum(self.rel_tol * d, self.abs_tol)
        agree = valid & (dp > 0) & (np.abs(dp - d) <= tol)
        front = valid & (dp > 0) & (dp < d - tol)                  # proxy where space is empty
        behind = valid & ~agree & ~front                           # real surface missing in proxy
        n = max(int(valid.sum()), 1)
        offenders = []
        for k in np.unique(obj[front]):
            if k < 0:
                continue
            pix = valid & (obj == k)
            frac = float((front & pix).sum() / max(pix.sum(), 1))
            if frac > 0.25 and pix.sum() > 0.003 * n:
                offenders.append((frac, self.scene.objects[k].id))
        stats = {"view": i, "depth_agreement": round(float(agree.sum() / n), 3),
                 "proxy_in_front_of_photo": round(float(front.sum() / n), 3),
                 "photo_surface_missing_in_proxy": round(float(behind.sum() / n), 3),
                 "objects_mostly_in_front": [f"{oid} ({f:.0%})" for f, oid in sorted(offenders)[::-1][:5]]}
        return stats, (agree, front, behind)

    def score(self):
        stats = [self.depth_check(i)[0] for i in range(len(self.views))]
        return float(np.mean([s["depth_agreement"] for s in stats])), stats

    def error_map(self, i, masks):
        """Photo (dimmed) with red = surface missing in the proxy, blue = proxy in empty space."""
        v = self.views[i]
        agree, front, behind = masks
        img = self.photos[i].astype(np.float32) * 0.45
        for mask, color in ((agree, (60, 200, 60)), (behind, (255, 40, 40)), (front, (40, 120, 255))):
            m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((v.width, v.height),
                                                                                 Image.NEAREST)) > 0
            img[m] = 0.45 * img[m] + 0.55 * np.array(color, np.float32)
        return img.clip(0, 255).astype(np.uint8)

    def view_images(self, i, show_photo=True, depth_error=True, highlight=None):
        v = self.views[i]
        r = self.compiled().render(v.K, v.c2w, v.width, v.height, depth=False)
        ov = overlay(self.photos[i], r, self.scene, v.K, v.c2w, alpha=0.5,
                     colors=instance_colors(self.scene), highlight=highlight)
        out = []
        if show_photo:
            out.append(("photo", draw_ruler(self.photos[i])))
        out.append(("proxy overlay (outlines + ids; each object its own colour)", draw_ruler(ov)))
        stats, masks = self.depth_check(i)
        if depth_error:
            out.append(("depth check (green agrees, red = real surface missing from proxy, blue = "
                        "proxy box where the photo sees through)", draw_ruler(self.error_map(i, masks))))
        counts = np.bincount(r["obj"][r["obj"] >= 0].ravel(), minlength=len(self.scene.objects))
        stats["visible_objects"] = [self.scene.objects[k].id for k in np.argsort(-counts)
                                    if counts[k] > 0.002 * r["obj"].size]
        return out, stats

    # geometry helpers -------------------------------------------------------------------------
    def in_view(self, obj):
        seen = []
        for i, v in enumerate(self.views):
            rect = projected_rect(v, obj.center, obj.size, obj.yaw)
            if rect and rect[2] - rect[0] > 2 and rect[3] - rect[1] > 2:
                seen.append(i)
        return seen

    def free_points(self, points=None, exclude=None, margin=0.02):
        """Points not explained by the floor, walls, ceiling or another object's box."""
        pts = self.points if points is None else points
        pts = pts[pts[:, 2] > 0.03]
        for o in self.scene.objects:
            if o.id == exclude or o.proxy_class == "floor" or not len(pts):
                continue
            pts = pts[~o.contains(pts, margin=0.05 if o.proxy_class in STRUCTURE else margin)]
        return pts

    def points_in_marks(self, marks):
        """3D points of an object marked in several photos.

        Each marked view contributes the Phase 1 surface points inside its rectangle (union: the
        views may see different sides). A point is dropped when it projects into a marked photo
        outside that photo's rectangle (marks are the object's full extent), when the floor, a
        wall or an existing box explains it, or when it is not in the largest connected cluster.
        """
        H = []
        for m in marks:
            i, b = m["view"], m["box"]
            v, d = self.views[i], self.depth[i]
            sx, sy = d.shape[1] / v.width, d.shape[0] / v.height
            u0, v0, u1, v1 = int(b[0] * sx), int(b[1] * sy), int(math.ceil(b[2] * sx)), int(math.ceil(b[3] * sy))
            vv, uu = np.nonzero(d[v0:v1, u0:u1] > 0)
            uu, vv = uu + u0, vv + v0
            z = d[vv, uu]
            rays = np.linalg.inv(self.Kd[i]) @ np.stack([uu, vv, np.ones_like(uu)]).astype(float)
            c2w = np.asarray(self.scene.cameras[i].c2w, float)
            H.append((c2w[:3, :3] @ (rays * z) + c2w[:3, 3:4]).T)
        pts = np.concatenate(H) if H else np.zeros((0, 3))
        for m in marks:  # visual hull: inside every marked view's rectangle, unless out of frame
            if not len(pts):
                break
            i, b = m["view"], m["box"]
            v = self.views[i]
            uv, z = project(v.K, v.c2w, pts)
            in_img = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < v.width) & (uv[:, 1] >= 0) & (uv[:, 1] < v.height)
            pad = 0.05 * max(b[2] - b[0], b[3] - b[1])
            inside = (uv[:, 0] >= b[0] - pad) & (uv[:, 0] <= b[2] + pad) & (uv[:, 1] >= b[1] - pad) & (uv[:, 1] <= b[3] + pad)
            pts = pts[~(in_img & ~inside)]
        pts = self.free_points(pts)
        return largest_cluster(pts)


def largest_cluster(pts, voxel=0.05, min_points=30):
    """Points of the largest 26-connected voxel cluster."""
    if len(pts) < min_points:
        return pts
    from scipy import ndimage

    idx = np.floor((pts - pts.min(axis=0)) / voxel).astype(int)
    grid = np.zeros(idx.max(axis=0) + 1, dtype=bool)
    grid[tuple(idx.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return pts
    lab = labels[tuple(idx.T)]
    counts = np.bincount(lab)
    counts[0] = 0
    return pts[lab == int(np.argmax(counts))]


# --- tools ---------------------------------------------------------------------------------------

_V3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_MARKS = {
    "type": "array", "minItems": 1, "maxItems": 6,
    "description": "Rectangles around the object's full extent (include parts hidden behind other things) in individual photos, in the pixel "
                   "coordinates of the images you were shown (use the magenta rulers).",
    "items": {"type": "object", "additionalProperties": False, "required": ["view", "box"],
              "properties": {"view": {"type": "integer"},
                             "box": {"type": "array", "items": {"type": "number"}, "minItems": 4,
                                     "maxItems": 4, "description": "[u0, v0, u1, v1]"}}},
}

READ_TOOLS = [
    {"name": "list_objects",
     "description": "Every object in the proxy world: id, label, proxy class, center/size (m), yaw "
                    "(deg), bottom/top height, physics, and the views it projects into.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "render_views",
     "description": "Render the current proxy world into up to 4 photos. Returns, per view, the photo, "
                    "the proxy overlay with object ids, a depth-check image, and depth statistics.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["views"],
                      "properties": {
                          "views": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 4},
                          "show_photo": {"type": "boolean", "description": "include the raw photo (default true)"},
                          "depth_error": {"type": "boolean", "description": "include the depth check (default true)"},
                          "highlight": {"type": "string", "description": "object id to outline in red"}}}},
]

EDIT_TOOLS = READ_TOOLS + [
    {"name": "add_from_marks",
     "description": "Add a missing object: mark it in 1-4 photos and a box is fitted to the Phase 1 "
                    "3D points inside all marks (then refined to the marks). Best tool for missing objects.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["label", "marks"],
                      "properties": {"label": {"type": "string"}, "marks": _MARKS,
                                     "supported": {"type": "boolean", "description": "extend the box down to the floor or the surface it stands on (default true; false for wall-mounted or hanging things)"}}}},
    {"name": "add_object",
     "description": "Add an object with an explicit box (world metres, z up, floor z=0).",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["label", "center", "size", "yaw_deg"],
                      "properties": {"label": {"type": "string"}, "center": _V3, "size": _V3,
                                     "yaw_deg": {"type": "number"},
                                     "movable": {"type": "boolean"}, "mass": {"type": "number"}}}},
    {"name": "update_object",
     "description": "Change any field of an object. Omitted fields keep their value.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["id"],
                      "properties": {"id": {"type": "string"}, "label": {"type": "string"},
                                     "center": _V3, "size": _V3, "yaw_deg": {"type": "number"},
                                     "proxy_class": {"type": "string", "enum": ["object", "floor", "wall", "ceiling", "window", "door"]},
                                     "movable": {"type": "boolean"}, "mass": {"type": "number"},
                                     "friction": {"type": "number"}, "restitution": {"type": "number"},
                                     "notes": {"type": "string"}}}},
    {"name": "remove_object",
     "description": "Delete an object that does not exist in the photos (or is a duplicate).",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["id", "reason"],
                      "properties": {"id": {"type": "string"}, "reason": {"type": "string"}}}},
    {"name": "split_object",
     "description": "Split one box into two along its own x, y or z axis (e.g. one box covering two pieces of furniture).",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["id", "axis", "fraction", "labels"],
                      "properties": {"id": {"type": "string"}, "axis": {"type": "string", "enum": ["x", "y", "z"]},
                                     "fraction": {"type": "number", "description": "share of the size given to the first part, measured from the negative side"},
                                     "labels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}}}},
    {"name": "mark_and_fit",
     "description": "Align an existing object: mark its full extent in 2-6 photos; the optimizer fits "
                    "its center, size and yaw. With 3+ marks one is held out and must agree (IoU>=0.5).",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["id", "marks"],
                      "properties": {"id": {"type": "string"}, "marks": _MARKS,
                                     "keep_bottom": {"type": "boolean", "description": "keep the bottom height (default true)"}}}},
    {"name": "fit_to_points",
     "description": "Shrink-wrap an object's box around the Phase 1 3D points near it.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["id"],
                      "properties": {"id": {"type": "string"}}}},
    {"name": "finish",
     "description": "End this editing round with a short summary of what changed and what is left.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["summary"],
                      "properties": {"summary": {"type": "string"}}}},
]

REVIEW_TOOLS = READ_TOOLS + [
    {"name": "submit_review",
     "description": "Your verdict on the proxy world. Call exactly once, at the end.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["accept", "quality", "findings"],
                      "properties": {
                          "accept": {"type": "boolean"},
                          "quality": {"type": "integer", "description": "0-10"},
                          "findings": {"type": "array", "items": {
                              "type": "object", "additionalProperties": False, "required": ["problem", "fix"],
                              "properties": {"object": {"type": "string", "description": "object id, or 'missing'"},
                                             "views": {"type": "array", "items": {"type": "integer"}},
                                             "problem": {"type": "string"}, "fix": {"type": "string"}}}}}}},
]

MUTATING = {"add_from_marks", "add_object", "update_object", "remove_object", "split_object",
            "mark_and_fit", "fit_to_points"}

WORLD_RULES = """World frame: metres, z up, floor at z = 0. Every object is a gravity-aligned box: center
[x, y, z], size [sx, sy, sz] = full extents along the box's own axes, yaw = rotation about +z.
bottom = center_z - sz/2. Floor, walls and ceiling are thin boxes too. Images are shown at the
resolution given with them; magenta rulers on the top/left borders mark every 100 pixels, and all
pixel coordinates you give are in that image's pixel grid. View indices start at 0."""

REVIEWER_SYSTEM = f"""You review a 3D proxy world reconstructed from photos of a room. It will drive a physics
simulator and a video renderer, so every object a robot or a person could touch must be present,
in the right place and at the right size. You can only look; you cannot edit.

{WORLD_RULES}

Check every view. Look for: missing objects (red regions in the depth check usually mean a real
surface has no box), boxes where there is nothing (blue regions), boxes too big/small/shifted or
rotated, duplicates, one box covering several objects, wrong labels, walls/floor in the wrong
place. Ignore objects smaller than ~10 cm unless they are on a surface someone would interact with.
Accept only if every object larger than ~30 cm is present and aligned to within ~10 cm and there
are no phantom boxes. Then call submit_review once with concrete findings (object id, the views
where it shows, what is wrong, and what fix you suggest), most important first."""

EDITOR_SYSTEM = f"""You edit a 3D proxy world reconstructed from photos of a room so that its renders match the
photos. A reviewer has listed problems; fix them, most important first.

{WORLD_RULES}

How to work:
- Missing object: add_from_marks with tight rectangles in 2-3 photos that show it well.
- Misplaced or mis-sized object: mark_and_fit with rectangles around its full extent in 2-4
  photos (3+ lets the optimizer check itself on a held-out photo), or fit_to_points.
- Use update_object for labels, physics (movable/mass/friction/restitution) and small manual moves;
  remove_object for phantom boxes and duplicates; split_object when one box covers two things.
- After a few edits, call render_views on the affected views to check the result; undo a change
  that made things worse.
- Objects rest on the floor or on another object unless they are mounted (pictures, wall lamps).
Call finish when the listed problems are fixed or cannot be fixed."""


def _b64png(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _image_blocks(images):
    blocks = []
    for caption, img in images:
        blocks.append({"type": "text", "text": f"{caption} ({img.shape[1]}x{img.shape[0]} px):"})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                   "data": _b64png(img)}})
    return blocks


class ToolError(Exception):
    pass


def _vec(x, n, name):
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise ToolError(f"{name} must be a list of {n} numbers")
    try:
        return [float(v) for v in x]
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a list of {n} numbers")


class SceneTools:
    """Executes the agent's tool calls against a Workspace."""

    def __init__(self, ws, max_edits):
        self.ws, self.max_edits = ws, max_edits
        self.finished, self.review = None, None

    def __call__(self, name, args):
        if name in MUTATING and self.ws.edits >= self.max_edits:
            raise ToolError("edit budget exhausted for this run; call finish")
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            raise ToolError(f"unknown tool {name}")
        out = fn(**args)
        if name in MUTATING:
            self.ws.changed()
        return out

    # helpers
    def _obj(self, oid):
        try:
            return self.ws.scene.get(oid)
        except KeyError as e:
            raise ToolError(str(e.args[0]))

    def _marks(self, marks):
        out = []
        for m in marks:
            i = int(m["view"])
            if not 0 <= i < len(self.ws.views):
                raise ToolError(f"view {i} does not exist (0-{len(self.ws.views) - 1})")
            v = self.ws.views[i]
            b = _vec(m["box"], 4, "box")
            b = [min(max(b[0], 0), v.width), min(max(b[1], 0), v.height),
                 min(max(b[2], 0), v.width), min(max(b[3], 0), v.height)]
            if b[2] - b[0] < 3 or b[3] - b[1] < 3:
                raise ToolError(f"box {m['box']} in view {i} is empty; use [u0, v0, u1, v1] with u0<u1, v0<v1")
            out.append({"view": i, "box": b})
        return out

    def _describe(self, o):
        return {"id": o.id, "label": o.label, "class": o.proxy_class,
                "center": [round(c, 3) for c in o.center], "size": [round(s, 3) for s in o.size],
                "yaw_deg": round(math.degrees(o.yaw), 1), "bottom": round(o.bottom, 3),
                "top": round(o.top, 3), "movable": o.movable, "mass": round(o.mass, 2),
                "source": o.source, "notes": o.notes}

    # read-only
    def t_list_objects(self):
        rows = []
        for o in self.ws.scene.objects:
            d = self._describe(o)
            d["in_views"] = self.ws.in_view(o)
            rows.append(d)
        return json.dumps(rows)

    def t_render_views(self, views, show_photo=True, depth_error=True, highlight=None):
        blocks, stats = [], []
        for i in views[:4]:
            if not 0 <= int(i) < len(self.ws.views):
                raise ToolError(f"view {i} does not exist (0-{len(self.ws.views) - 1})")
            imgs, st = self.ws.view_images(int(i), show_photo, depth_error, highlight)
            blocks.append({"type": "text", "text": f"--- view {i}: {json.dumps(st)}"})
            blocks += _image_blocks(imgs)
            stats.append(st)
        return blocks

    # editing
    def t_add_from_marks(self, label, marks, supported=True):
        marks = self._marks(marks)
        pts = self.ws.points_in_marks(marks)
        if len(pts) < 30:
            raise ToolError(f"only {len(pts)} unexplained 3D points fall inside the marks; mark the "
                            "object in other views, or use add_object")
        center, size, yaw = box_around_points(pts, 0.0, sweep_deg=90.0, pct=(3, 97), z_pct=(2, 99))
        if supported:  # lower sides are usually hidden: extend the box down to what supports it
            probe = Obj(id="_", label="_", category="_", center=center, size=size, yaw=yaw)
            base = max([0.0] + [o.top for o in self.ws.scene.objects
                                if o.proxy_class not in ("wall", "ceiling", "floor")
                                and o.top <= probe.bottom + 0.05 and o.contains(np.array([center]), margin=0.3)[0]])
            top = center[2] + size[2] / 2.0
            size[2], center[2] = top - base, (top + base) / 2.0
        obj = make_object(self.ws.scene, label, center, size, yaw, source="agent")
        res = {"added": self._describe(obj), "points_used": int(len(pts))}
        if len(marks) >= 2:
            res["refined_to_marks"] = fit_to_marks(obj, {m["view"]: m["box"] for m in marks},
                                                   self.ws.views, points=pts)
            res["final"] = self._describe(obj)
        return json.dumps(res)

    def t_add_object(self, label, center, size, yaw_deg, movable=None, mass=None):
        extra = {}
        if movable is not None:
            extra["movable"] = bool(movable)
        if mass is not None:
            extra["mass"] = float(mass)
        obj = make_object(self.ws.scene, label, _vec(center, 3, "center"), _vec(size, 3, "size"),
                          math.radians(float(yaw_deg)), source="agent", **extra)
        return json.dumps({"added": self._describe(obj)})

    def t_update_object(self, id, **fields):
        o = self._obj(id)
        if "label" in fields:
            o.label = fields["label"]
            o.category = normalize_category(fields["label"])
            if "proxy_class" not in fields and o.proxy_class not in STRUCTURE:
                o.proxy_class = proxy_class_for(o.category)
        if "center" in fields:
            o.center = _vec(fields["center"], 3, "center")
        if "size" in fields:
            o.size = [max(s, 0.01) for s in _vec(fields["size"], 3, "size")]
        if "yaw_deg" in fields:
            o.yaw = math.radians(float(fields["yaw_deg"]))
        for k in ("proxy_class", "movable", "mass", "friction", "restitution", "notes"):
            if k in fields:
                setattr(o, k, fields[k])
        o.source = o.source if o.source == "agent" else f"{o.source}+agent"
        return json.dumps({"updated": self._describe(o)})

    def t_remove_object(self, id, reason):
        if id == "floor":
            raise ToolError("the floor is needed by the simulator; move or resize it instead")
        self._obj(id)
        self.ws.scene.remove(id)
        return json.dumps({"removed": id, "reason": reason})

    def t_split_object(self, id, axis, fraction, labels):
        o = self._obj(id)
        f = float(fraction)
        if not 0.05 <= f <= 0.95:
            raise ToolError("fraction must be in [0.05, 0.95]")
        a = "xyz".index(axis)
        R = np.array([[math.cos(o.yaw), -math.sin(o.yaw), 0], [math.sin(o.yaw), math.cos(o.yaw), 0], [0, 0, 1]])
        parts = []
        for share, offset_sign, label in ((f, -1, labels[0]), (1 - f, 1, labels[1])):
            size = list(o.size)
            size[a] = o.size[a] * share
            local = np.zeros(3)
            local[a] = offset_sign * (o.size[a] - size[a]) / 2.0
            center = np.asarray(o.center) + R @ local
            parts.append(make_object(self.ws.scene, label, center, size, o.yaw, source="agent"))
        self.ws.scene.remove(id)
        return json.dumps({"removed": id, "added": [self._describe(p) for p in parts]})

    def t_mark_and_fit(self, id, marks, keep_bottom=True):
        o = self._obj(id)
        marks = self._marks(marks)
        before = copy.deepcopy(o)
        res = fit_to_marks(o, {m["view"]: m["box"] for m in marks}, self.ws.views,
                           keep_bottom=keep_bottom, points=self.ws.free_points(exclude=o.id))
        if not res["accepted"]:
            o.center, o.size, o.yaw = before.center, before.size, before.yaw
        res["object"] = self._describe(o)
        return json.dumps(res)

    def t_fit_to_points(self, id):
        o = self._obj(id)
        res = fit_to_points(o, self.ws.free_points(exclude=o.id))
        res["object"] = self._describe(o)
        return json.dumps(res)

    def t_finish(self, summary):
        self.finished = summary
        return "ok"

    def t_submit_review(self, accept, quality, findings):
        self.review = {"accept": bool(accept), "quality": int(quality), "findings": findings}
        return "review recorded"


# --- conversation loop ---------------------------------------------------------------------------


class Agent:
    def __init__(self, ws, out_dir, effort="high", max_edits=60, time_limit_s=3600, max_tokens=32000,
                 overview_views=12, client=None):
        import anthropic

        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.ws, self.out_dir, self.effort = ws, out_dir, effort
        self.max_edits, self.deadline, self.max_tokens = max_edits, time.time() + time_limit_s, max_tokens
        self.overview_views = overview_views
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "requests": 0, "fallbacks": 0, "refusals": 0}
        os.makedirs(os.path.join(out_dir, "agent_log"), exist_ok=True)

    def _request(self, system, tools, messages):
        for attempt in range(5):
            try:
                with self.client.beta.messages.stream(
                        model=MODEL, max_tokens=self.max_tokens, system=system, tools=tools,
                        messages=messages, output_config={"effort": self.effort},
                        cache_control={"type": "ephemeral"},
                        betas=[FALLBACK_BETA], fallbacks="default") as stream:
                    msg = stream.get_final_message()
                break
            except (self.anthropic.RateLimitError, self.anthropic.APIConnectionError,
                    self.anthropic.InternalServerError) as e:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt * 5)
                print(f"  retrying after {type(e).__name__}")
        u = msg.usage
        self.usage["requests"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.usage[k] += int(getattr(u, k, 0) or 0)
        if any(getattr(b, "type", "") == "fallback" for b in msg.content):
            self.usage["fallbacks"] += 1
        return msg

    def overview(self, tools):
        n = len(self.ws.views)
        views = sorted(set(np.linspace(0, n - 1, min(n, self.overview_views)).round().astype(int).tolist()))
        blocks = [{"type": "text", "text": f"Objects:\n{tools.t_list_objects()}\n\nThere are {n} views. "
                                           f"Overview of views {views} (overlay + depth check):"}]
        for i in views:
            imgs, st = self.ws.view_images(i, show_photo=False, depth_error=True)
            blocks.append({"type": "text", "text": f"--- view {i}: {json.dumps(st)}"})
            blocks += _image_blocks(imgs)
        return blocks

    def converse(self, role, system, tool_defs, first_message, tools, max_turns, log_name):
        messages = [{"role": "user", "content": first_message}]
        log = []
        nudged = False
        for turn in range(max_turns):
            if time.time() > self.deadline:
                log.append({"stop": "time limit"})
                break
            msg = self._request(system, tool_defs, messages)
            texts = [b.text for b in msg.content if b.type == "text"]
            calls = [b for b in msg.content if b.type == "tool_use"]
            log.append({"turn": turn, "stop_reason": msg.stop_reason, "text": texts,
                        "calls": [{"name": c.name, "input": c.input} for c in calls]})
            if msg.stop_reason == "refusal":
                self.usage["refusals"] += 1
                log.append({"stop": "refusal"})
                break
            messages.append({"role": "assistant", "content": msg.content})
            if not calls:
                done = tools.finished is not None or tools.review is not None
                if done or nudged:
                    break
                nudged = True
                need = "submit_review" if role == "reviewer" else "finish"
                messages.append({"role": "user", "content": f"Please call {need} now."})
                continue
            results = []
            for c in calls:
                if msg.stop_reason == "max_tokens":
                    results.append({"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                                    "content": "your output was cut off; issue the call again"})
                    continue
                try:
                    out = tools(c.name, dict(c.input))
                    results.append({"type": "tool_result", "tool_use_id": c.id, "content": out})
                    log[-1].setdefault("results", []).append(out if isinstance(out, str) else
                                                             [b["text"] for b in out if b["type"] == "text"])
                except ToolError as e:
                    results.append({"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                                    "content": f"Error: {e}"})
                    log[-1].setdefault("results", []).append(f"Error: {e}")
                except Exception as e:  # a bug in a tool must not kill the round
                    results.append({"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                                    "content": f"Error: {type(e).__name__}: {e}"})
                    log[-1].setdefault("results", []).append(f"Exception: {type(e).__name__}: {e}")
            messages.append({"role": "user", "content": results})
            if tools.finished is not None or tools.review is not None:
                break
        with open(os.path.join(self.out_dir, "agent_log", log_name), "w") as f:
            json.dump(log, f, indent=1, default=str)
        return log

    def review(self, round_idx):
        tools = SceneTools(self.ws, self.max_edits)
        first = self.overview(tools) + [{"type": "text", "text": "Review this proxy world and call submit_review."}]
        self.converse("reviewer", REVIEWER_SYSTEM, REVIEW_TOOLS, first, tools, max_turns=12,
                      log_name=f"round{round_idx}_reviewer.json")
        return tools.review or {"accept": False, "quality": 0, "findings": [],
                                "note": "reviewer did not submit a review"}

    def edit(self, round_idx, review):
        tools = SceneTools(self.ws, self.max_edits)
        findings = json.dumps(review.get("findings", []), indent=1)
        first = [{"type": "text", "text": f"Reviewer findings (quality {review.get('quality')}/10):\n{findings}"}]
        first += self.overview(tools)
        first.append({"type": "text", "text": "Fix the findings, then call finish."})
        self.converse("editor", EDITOR_SYSTEM, EDIT_TOOLS, first, tools, max_turns=40,
                      log_name=f"round{round_idx}_editor.json")
        return tools.finished


def run(scene, points, phase1_dir, out_dir, max_rounds=4, effort="high", max_edits=60,
        time_limit_s=3600, min_gain=0.005, max_side=768, client=None):
    """Reviewer/editor rounds on `scene` (edited in place). Returns (scene, history)."""
    ws = Workspace(scene, points, phase1_dir, max_side=max_side)
    agent = Agent(ws, out_dir, effort=effort, max_edits=max_edits, time_limit_s=time_limit_s,
                  client=client)
    best_score, _ = ws.score()
    best = copy.deepcopy(scene.objects)
    history = [{"round": 0, "depth_agreement": round(best_score, 4), "objects": len(scene.objects)}]
    print(f"Phase 3 agent: start, depth agreement {best_score:.3f}, {len(scene.objects)} objects")
    stalled = 0
    for r in range(1, max_rounds + 1):
        if time.time() > agent.deadline or ws.edits >= max_edits:
            history.append({"round": r, "stop": "budget"})
            break
        review = agent.review(r)
        print(f"  round {r}: reviewer accept={review['accept']} quality={review.get('quality')} "
              f"findings={len(review.get('findings', []))}")
        if review["accept"]:
            history.append({"round": r, "review": review, "stop": "accepted"})
            break
        summary = agent.edit(r, review)
        score, _ = ws.score()
        entry = {"round": r, "review": review, "editor_summary": summary,
                 "depth_agreement": round(score, 4), "objects": len(scene.objects), "edits": ws.edits}
        if score < best_score - 0.03:  # the edits broke geometry: keep the best world instead
            scene.objects = copy.deepcopy(best)
            ws.changed()
            entry["reverted"] = True
            stalled += 1
        else:
            stalled = stalled + 1 if score < best_score + min_gain else 0
            if score >= best_score:
                best_score, best = score, copy.deepcopy(scene.objects)
        history.append(entry)
        print(f"  round {r}: depth agreement {score:.3f} ({ws.edits} edits)")
        if stalled >= 2:
            history.append({"round": r, "stop": "geometry stopped improving"})
            break
    history.append({"usage": agent.usage})
    with open(os.path.join(out_dir, "agent_log", "history.json"), "w") as f:
        json.dump(history, f, indent=1, default=str)
    return scene, history
