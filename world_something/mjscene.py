"""Compile a Scene to MuJoCo and render flat-colour proxy frames from any OpenCV camera.

MuJoCo is both the physics engine (Phase 5) and the proxy render engine (Phases 3 and 6): its
segmentation renderer gives the object id of every pixel, which maps to the proxy palette.
Headless rendering needs MUJOCO_GL=egl (GPU nodes) or MUJOCO_GL=osmesa (CPU only).
"""

import math
import os

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"
if os.environ.get("MUJOCO_GL") in ("egl", "osmesa"):
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from world_something.cameras import CV_TO_GL, as4, project
from world_something.scene import PALETTE, yaw_matrix, yaw_quat

PIXEL_PITCH = 1e-5   # metres per pixel on MuJoCo's virtual sensor
MAX_RENDER = 2048    # offscreen buffer size


def restitution_to_solref(restitution, timeconst=0.005):
    """MuJoCo has no restitution coefficient; a damped contact spring gives the same bounce.

    For a spring-damper, e = exp(-zeta*pi/sqrt(1-zeta^2)), so zeta = -ln e / sqrt(pi^2 + ln^2 e).
    """
    e = float(np.clip(restitution, 0.01, 0.95))
    zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
    return [timeconst, zeta]


class CompiledScene:
    """A MuJoCo model of the scene plus the maps from geoms back to scene objects."""

    def __init__(self, scene, dynamic=False, timestep=1.0 / 960.0, gravity=(0.0, 0.0, -9.81),
                 asset_specs=None, extra=None):
        self.scene = scene
        self.dynamic = dynamic
        spec = mujoco.MjSpec()
        spec.option.timestep = timestep
        spec.option.gravity = list(gravity)
        spec.visual.global_.offwidth = MAX_RENDER
        spec.visual.global_.offheight = MAX_RENDER
        spec.worldbody.add_light(name="sun", pos=[0, 0, 6], dir=[0.2, 0.3, -1.0],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        # Menagerie's recommended settings for its robots (stable grasping and contacts).
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.option.noslip_iterations = 3
        asset_specs = asset_specs or {}
        self.asset_specs = asset_specs
        self.obj_index = {o.id: i for i, o in enumerate(scene.objects)}
        self.body_names = {}
        for o in scene.objects:
            if o.id in asset_specs:
                a = asset_specs[o.id]
                offset = np.asarray(getattr(a, "offset", [0.0, 0.0, 0.0]), float)
                pos = np.asarray(o.center, float) + yaw_matrix(o.yaw) @ offset
                frame = spec.worldbody.add_frame(pos=list(pos), quat=yaw_quat(o.yaw))
                spec.attach(getattr(a, "spec", a), prefix=f"{o.id}/", frame=frame)
                continue
            geom_kw = dict(name=f"obj:{o.id}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=list(np.maximum(o.half, 0.005)),
                           rgba=[c / 255.0 for c in PALETTE.get(o.proxy_class, PALETTE["object"])] + [1.0],
                           friction=[o.friction, 0.005, 0.0001],
                           solref=restitution_to_solref(o.restitution))
            if dynamic and o.movable:
                body = spec.worldbody.add_body(name=f"body:{o.id}", pos=list(o.center),
                                               quat=yaw_quat(o.yaw))
                body.add_freejoint(name=f"free:{o.id}")
                body.add_geom(mass=max(o.mass, 1e-3), **geom_kw)
                self.body_names[o.id] = f"body:{o.id}"
            else:
                spec.worldbody.add_geom(pos=list(o.center), quat=yaw_quat(o.yaw), **geom_kw)
        self.camera_name = "view"
        spec.worldbody.add_camera(name=self.camera_name, pos=[0, 0, 1], quat=[1, 0, 0, 0],
                                  resolution=[640, 480], sensor_size=[640 * PIXEL_PITCH, 480 * PIXEL_PITCH],
                                  focal_pixel=[500.0, 500.0], principal_pixel=[0.0, 0.0])
        if extra is not None:
            extra(spec, self)
        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._apply_start_values(asset_specs)
        mujoco.mj_forward(self.model, self.data)
        self.qpos_init, self.ctrl_init = self.data.qpos.copy(), self.data.ctrl.copy()
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self._build_maps(asset_specs)
        self._renderers = {}

    def _apply_start_values(self, asset_specs):
        """Joint positions and actuator targets from each asset's keyframe (e.g. a robot's home)."""
        m, d = self.model, self.data
        for oid, a in asset_specs.items():
            for name, val in getattr(a, "qpos", {}).items():
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{oid}/{name}")
                if j >= 0:
                    d.qpos[m.jnt_qposadr[j]] = val
            for name, val in getattr(a, "ctrl", {}).items():
                u = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{oid}/{name}")
                if u >= 0:
                    d.ctrl[u] = val

    def _build_maps(self, asset_specs):
        m = self.model
        geom_obj = np.full(m.ngeom, -1, dtype=np.int64)
        body_obj = np.full(m.nbody, -1, dtype=np.int64)
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            for oid in asset_specs:
                if name.startswith(f"{oid}/"):
                    body_obj[b] = self.obj_index[oid]
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("obj:"):
                geom_obj[g] = self.obj_index.get(name[4:], -1)
            else:
                b = m.geom_bodyid[g]
                while b > 0 and body_obj[b] < 0:
                    b = m.body_parentid[b]
                geom_obj[g] = body_obj[b] if b > 0 else -1
        self.geom_obj = geom_obj
        colors = [PALETTE.get(o.proxy_class, PALETTE["object"]) for o in self.scene.objects]
        self.obj_rgb = np.array(colors + [PALETTE["unknown"]], dtype=np.uint8)  # last = background

    # --- cameras -------------------------------------------------------------------------------
    def set_camera(self, K, c2w, width, height):
        m, cid = self.model, self.cam_id
        K = np.asarray(K, float)
        m.cam_resolution[cid] = [width, height]
        m.cam_sensorsize[cid] = [width * PIXEL_PITCH, height * PIXEL_PITCH]
        # MuJoCo's principal offset runs opposite to OpenCV's and uses pixel corners (+0.5).
        m.cam_intrinsic[cid] = [K[0, 0] * PIXEL_PITCH, K[1, 1] * PIXEL_PITCH,
                                (width / 2.0 - K[0, 2] - 0.5) * PIXEL_PITCH,
                                (height / 2.0 - K[1, 2] - 0.5) * PIXEL_PITCH]
        c2w = as4(c2w)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (c2w[:3, :3] @ CV_TO_GL).flatten())
        m.cam_pos[cid] = c2w[:3, 3]
        m.cam_quat[cid] = quat
        mujoco.mj_kinematics(m, self.data)
        mujoco.mj_camlight(m, self.data)

    def _renderer(self, width, height):
        key = (int(width), int(height))
        if key not in self._renderers:
            if width > MAX_RENDER or height > MAX_RENDER:
                raise ValueError(f"render size {key} exceeds {MAX_RENDER}")
            self._renderers[key] = mujoco.Renderer(self.model, height=int(height), width=int(width))
        return self._renderers[key]

    # --- rendering -----------------------------------------------------------------------------
    def render(self, K, c2w, width, height, depth=True, rgb=False):
        """Object index per pixel (-1 = nothing), flat proxy colours, and optionally depth (m)."""
        self.set_camera(K, c2w, width, height)
        r = self._renderer(width, height)
        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=self.cam_id)
        seg = r.render()
        r.disable_segmentation_rendering()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        obj = np.where(geom >= 0, self.geom_obj[np.clip(geom, 0, None)], -1)
        out = {"obj": obj, "proxy": self.obj_rgb[obj]}  # obj = -1 picks the background colour
        if depth:
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=self.cam_id)
            d = r.render()
            r.disable_depth_rendering()
            out["depth"] = np.where(obj >= 0, d, 0.0).astype(np.float32)
        if rgb:
            r.update_scene(self.data, camera=self.cam_id)
            out["rgb"] = r.render()
        return out

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}


# --- overlays for people and agents ----------------------------------------------------------


def boundaries(obj):
    edge = np.zeros(obj.shape, bool)
    edge[:, 1:] |= obj[:, 1:] != obj[:, :-1]
    edge[1:, :] |= obj[1:, :] != obj[:-1, :]
    return edge


def instance_colors(scene):
    """Distinct colour per object (architecture keeps its palette colour); last row = nothing."""
    out = []
    for i, o in enumerate(scene.objects):
        if o.proxy_class in ("floor", "wall", "ceiling"):
            out.append(PALETTE[o.proxy_class])
            continue
        h = (i * 0.618034) % 1.0
        r, g, b = [int(255 * c) for c in _hsv(h, 0.75, 0.95)]
        out.append((r, g, b))
    return np.array(out + [PALETTE["unknown"]], dtype=np.uint8)


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def draw_ruler(img, step=100):
    """Pixel ticks on the top and left borders, so a VLM can read coordinates off the image."""
    img = Image.fromarray(np.asarray(img, np.uint8)) if not isinstance(img, Image.Image) else img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in range(step, w, step):
        draw.line([x, 0, x, 9], fill=(255, 0, 255), width=2)
        draw.text((x + 2, 9), str(x), fill=(255, 0, 255))
    for y in range(step, h, step):
        draw.line([0, y, 9, y], fill=(255, 0, 255), width=2)
        draw.text((11, y - 5), str(y), fill=(255, 0, 255))
    return np.asarray(img)


def overlay(photo, rendered, scene, K, c2w, alpha=0.45, labels=True, label_filter=None,
            colors=None, highlight=None):
    """Photo with the proxy blended on top, object outlines, and each object's id at its centre.

    colors: optional (n_objects + 1, 3) table replacing the proxy palette (e.g. instance_colors);
    highlight: optional object id drawn with a thick outline.
    """
    photo = np.asarray(photo, np.uint8)
    h, w = photo.shape[:2]
    obj = rendered["obj"]
    if obj.shape != (h, w):
        raise ValueError("render the proxy at the photo's resolution before overlaying")
    blend = photo.astype(np.float32)
    mask = obj >= 0
    fill = rendered["proxy"] if colors is None else colors[obj]
    blend[mask] = (1 - alpha) * blend[mask] + alpha * fill[mask]
    if highlight is not None and highlight in scene:
        idx = [o.id for o in scene.objects].index(highlight)
        sel = obj == idx
        ring = np.zeros_like(sel)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                ring |= np.roll(np.roll(sel, dy, 0), dx, 1)
        blend[ring & ~sel] = (255, 0, 0)
    blend[boundaries(obj)] = (20, 20, 20)
    img = Image.fromarray(blend.clip(0, 255).astype(np.uint8))
    if labels:
        draw = ImageDraw.Draw(img)
        centers = np.array([o.center for o in scene.objects])
        uv, z = project(K, c2w, centers)
        visible = set(np.unique(obj[obj >= 0]).tolist())
        for i, o in enumerate(scene.objects):
            if i not in visible or z[i] <= 0 or o.proxy_class in ("floor", "ceiling"):
                continue
            if label_filter is not None and not label_filter(o):
                continue
            u, v = np.clip(uv[i], [4, 4], [w - 60, h - 14])
            draw.rectangle([u - 2, v - 1, u + 7 * len(o.id) + 2, v + 12], fill=(0, 0, 0))
            draw.text((u, v), o.id, fill=(255, 230, 90))
    return np.asarray(img)


def side_by_side(*images):
    h = max(i.shape[0] for i in images)
    padded = [np.pad(i, ((0, h - i.shape[0]), (0, 4), (0, 0)), constant_values=255) for i in images]
    return np.concatenate(padded, axis=1)


def save_png(path, array):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(np.asarray(array, np.uint8)).save(path)
