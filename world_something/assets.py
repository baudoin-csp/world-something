"""Phase 4: asset library. Turns proxy boxes into more detailed MuJoCo models, and adds robots.

An object's `asset` field picks its model; objects without one stay boxes.
  menagerie:<robot>  a robot from MuJoCo Menagerie: panda, google_robot, g1, stretch3.
                     (Galaxea R1 Pro, used in some Mundo demos, is not in Menagerie.)
  rough:<kind>       a rough procedural model fitted into the object's box: table, desk, chair,
                     stool, bed, sofa, shelf, cabinet (hinged door), drawers (sliding drawer),
                     door (hinged), lamp, ball, cylinder, box. Rough models face +x of the box.
  mesh:<path>        a mesh (.obj or .stl, e.g. a TRELLIS image-to-3D output converted from .glb)
                     scaled into the box; MuJoCo collides with its convex hull.

Every geom of an asset belongs to a body named "<object id>/...", so the proxy renderer can colour
it with the object's palette class and the simulator can report its pose.
"""

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from world_something import THIRD_PARTY
from world_something.scene import PALETTE, Obj, make_object, normalize_category

MENAGERIE_DIR = os.environ.get("MENAGERIE_DIR", os.path.join(THIRD_PARTY, "mujoco_menagerie"))

# xml, keyframe to start from, whether the base is free (mobile/legged) and the base footprint (m)
ROBOTS = {
    "panda": dict(dir="franka_emika_panda", xml="panda.xml", key="home", free=False,
                  footprint=(0.3, 0.3), height=1.0),
    "google_robot": dict(dir="google_robot", xml="robot.xml", key=None, free=False,
                         footprint=(0.6, 0.6), height=1.3),
    "g1": dict(dir="unitree_g1", xml="g1.xml", key="stand", free=True,
               footprint=(0.45, 0.7), height=1.35),
    "stretch3": dict(dir="hello_robot_stretch_3", xml="stretch.xml", key="home", free=True,
                     footprint=(0.35, 0.35), height=1.4),
}

# category -> rough model kind
ROUGH_KIND = {
    "table": "table", "dining table": "table", "coffee table": "table", "side table": "table",
    "desk": "table", "chair": "chair", "office chair": "chair", "desk chair": "chair",
    "armchair": "sofa", "stool": "stool", "bed": "bed", "sofa": "sofa", "couch": "sofa",
    "shelf": "shelf", "bookshelf": "shelf", "cabinet": "drawers", "dresser": "drawers",
    "night stand": "drawers", "nightstand": "drawers", "bedside table": "drawers",
    "wardrobe": "cabinet", "closet": "cabinet", "refrigerator": "cabinet", "door": "door",
    "lamp": "lamp", "floor lamp": "lamp", "ball": "ball", "bottle": "cylinder", "cup": "cylinder",
    "mug": "cylinder", "vase": "cylinder", "trash can": "cylinder", "trash bin": "cylinder",
    "bowl": "cylinder", "makeup vanity": "table",
}


@dataclass
class AssetSpec:
    """A MuJoCo spec to attach for one object, the offset of its root from the box centre (box
    frame), and joint/actuator start values by unprefixed name."""
    spec: object
    offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    qpos: dict = field(default_factory=dict)
    ctrl: dict = field(default_factory=dict)
    kind: str = "rough"


# --- robots --------------------------------------------------------------------------------------


def robot_spec(name):
    info = ROBOTS[name]
    path = os.path.join(MENAGERIE_DIR, info["dir"], info["xml"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run ./setup.sh (MuJoCo Menagerie sparse checkout)")
    spec = mujoco.MjSpec.from_file(path)
    base = os.path.dirname(path)
    for coll, attr_dir in ((spec.meshes, spec.meshdir), (spec.textures, spec.texturedir)):
        for a in coll:  # absolute asset paths: the parent model compiles from another directory
            if a.file and not os.path.isabs(a.file):
                a.file = os.path.join(base, attr_dir or "", a.file)
    spec.meshdir, spec.texturedir = "", ""
    qpos, ctrl = {}, {}
    if info["key"]:
        m = spec.compile()
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, info["key"])
        if k >= 0:
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                a = m.jnt_qposadr[j]
                qpos[jn] = float(m.key_qpos[k, a])
            for u in range(m.nu):
                an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, u)
                if an:
                    ctrl[an] = float(m.key_ctrl[k, u])
    for key in list(spec.keys):  # keyframes are re-applied by name after attaching
        spec.delete(key)
    return spec, qpos, ctrl


def place_robot(scene, name, xy, yaw_deg=0.0, on=None, label=None):
    """Add a robot to the scene, standing on the floor or on top of object `on` (e.g. a table)."""
    info = ROBOTS[name]
    base = 0.0 if on is None else scene.get(on).top
    fx, fy = info["footprint"]
    obj = make_object(scene, label or name, [xy[0], xy[1], base + info["height"] / 2.0],
                      [fx, fy, info["height"]], math.radians(yaw_deg), category="robot",
                      source="asset", proxy_class="new_object", movable=False, mass=0.0)
    obj.id = scene.new_id(label or name)
    obj.asset = f"menagerie:{name}"
    obj.notes = f"{name} from MuJoCo Menagerie" + (f", mounted on {on}" if on else "")
    return obj


# --- rough procedural models ----------------------------------------------------------------------


class _Builder:
    def __init__(self, obj, dynamic):
        self.obj, self.dynamic = obj, dynamic
        self.spec = mujoco.MjSpec()
        self.hx, self.hy, self.hz = np.maximum(obj.half, 0.005)
        self.root = self.spec.worldbody.add_body(name="root")
        if dynamic and obj.movable:
            self.root.add_freejoint(name="free")
        self.geoms = []
        self.color = [c / 255.0 for c in PALETTE.get(obj.proxy_class, PALETTE["object"])] + [1.0]

    def box(self, center, half, body=None, name=None):
        g = (body or self.root).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center),
                                          size=list(np.maximum(half, 0.003)), rgba=self.color)
        self.geoms.append((g, 8 * float(np.prod(np.maximum(half, 0.003)))))
        return g

    def cylinder(self, center, radius, half_h, body=None):
        g = (body or self.root).add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=list(center),
                                          size=[max(radius, 0.003), max(half_h, 0.003), 0], rgba=self.color)
        self.geoms.append((g, math.pi * radius ** 2 * 2 * half_h))
        return g

    def sphere(self, center, radius, body=None):
        g = (body or self.root).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=list(center),
                                          size=[max(radius, 0.003), 0, 0], rgba=self.color)
        self.geoms.append((g, 4.0 / 3.0 * math.pi * radius ** 3))
        return g

    def finish(self):
        from world_something.mjscene import restitution_to_solref

        total = sum(v for _, v in self.geoms) or 1.0
        for g, v in self.geoms:
            g.friction = [self.obj.friction, 0.005, 0.0001]
            g.solref = restitution_to_solref(self.obj.restitution)
            g.mass = max(self.obj.mass, 1e-3) * v / total
        return self.spec


def _legs(b, top_z, inset=0.06, radius=None):
    hx, hy, hz = b.hx, b.hy, b.hz
    r = radius or max(0.015, 0.04 * min(hx, hy))
    leg_h = (top_z + hz) / 2.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.box([sx * (hx - inset * hx - r), sy * (hy - inset * hy - r), -hz + leg_h], [r, r, leg_h])


def rough_spec(obj, kind, dynamic=False):
    b = _Builder(obj, dynamic)
    hx, hy, hz = b.hx, b.hy, b.hz
    if kind == "table":
        t = min(0.03, 0.1 * hz)
        b.box([0, 0, hz - t], [hx, hy, t])
        _legs(b, hz - 2 * t)
    elif kind == "stool":
        t = min(0.03, 0.1 * hz)
        b.box([0, 0, hz - t], [hx, hy, t])
        _legs(b, hz - 2 * t, inset=0.1)
    elif kind == "chair":
        seat = -hz + 0.48 * 2 * hz if hz > 0.3 else 0.0
        t = 0.025
        b.box([0, 0, seat], [hx, hy, t])
        _legs(b, seat - t)
        back_t = min(0.03, 0.15 * hx)
        b.box([-hx + back_t, 0, (seat + hz) / 2.0], [back_t, hy, (hz - seat) / 2.0])
    elif kind == "bed":
        base_h = min(0.25, 0.4 * hz)
        b.box([0, 0, -hz + base_h], [hx, hy, base_h])                       # frame / box spring
        top = min(hz, -hz + 2 * base_h + 0.25)
        b.box([0, 0, (-hz + 2 * base_h + top) / 2.0], [hx * 0.98, hy * 0.98, (top + hz - 2 * base_h) / 2.0])
        if top < hz - 0.1:                                                  # headboard at -x
            b.box([-hx + 0.03, 0, 0], [0.03, hy, hz])
    elif kind == "sofa":
        seat_top = -hz + min(0.45 * 2 * hz, 0.45)
        b.box([0, 0, (-hz + seat_top) / 2.0], [hx, hy, (seat_top + hz) / 2.0])
        back_t = min(0.12, 0.25 * hx)
        b.box([-hx + back_t, 0, (seat_top + hz) / 2.0], [back_t, hy, (hz - seat_top) / 2.0])
        arm_t = min(0.1, 0.12 * hy)
        arm_top = min(hz, seat_top + 0.2)
        for s in (-1, 1):
            b.box([0, s * (hy - arm_t), (seat_top + arm_top) / 2.0], [hx, arm_t, (arm_top - seat_top) / 2.0])
    elif kind == "shelf":
        t = min(0.02, 0.05 * min(hx, hy))
        for s in (-1, 1):
            b.box([0, s * (hy - t), 0], [hx, t, hz])                            # sides
        b.box([-hx + t, 0, 0], [t, hy, hz])                                     # back
        n = max(2, int(2 * hz / 0.35))
        for k in range(n + 1):
            z = -hz + t + k * (2 * hz - 2 * t) / n
            b.box([0, 0, z], [hx, hy, t])
    elif kind in ("cabinet", "drawers", "door"):
        t = min(0.02, 0.2 * hx)
        if kind != "door":
            b.box([-t, 0, 0], [hx - t, hy, hz])                                 # carcass
        door_half = [t, hy, hz] if kind != "drawers" else [t, hy * 0.96, hz * 0.3]
        if kind == "door":
            door_half = [hx, hy, hz]
            hinge = b.root.add_body(name="leaf", pos=[0, -hy, 0])
            b.box([0, hy, 0], door_half, body=hinge)
            j = hinge.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                                range=[-1.9, 1.9], damping=2.0)
        elif kind == "cabinet":
            hinge = b.root.add_body(name="door", pos=[hx - t, -hy, 0])
            b.box([0, hy, 0], door_half, body=hinge)
            j = hinge.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                                range=[0, 1.9], damping=1.0)
        else:
            drawer = b.root.add_body(name="drawer", pos=[hx - t, 0, hz * 0.6])
            b.box([0, 0, 0], door_half, body=drawer)
            b.box([-hx * 0.8, 0, -hz * 0.25], [hx * 0.8, hy * 0.9, 0.01], body=drawer)   # drawer floor
            j = drawer.add_joint(name="slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                                 range=[0, 1.6 * hx], damping=5.0)
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    elif kind == "lamp":
        r = 0.45 * min(hx, hy)
        b.cylinder([0, 0, -hz + 0.015], r * 0.7, 0.015)
        b.cylinder([0, 0, 0], 0.012, hz * 0.8)
        b.cylinder([0, 0, hz - 0.3 * hz], r, 0.3 * hz)
    elif kind == "ball":
        b.sphere([0, 0, 0], float(min(hx, hy, hz)))
    elif kind == "cylinder":
        b.cylinder([0, 0, 0], float(min(hx, hy)), hz)
    else:
        b.box([0, 0, 0], [hx, hy, hz])
    return b.finish()


def mesh_spec(obj, path, dynamic=False):
    """A mesh scaled (non-uniformly) to fill the object's box."""
    spec = mujoco.MjSpec()
    probe = mujoco.MjSpec()
    probe.add_mesh(name="m", file=os.path.abspath(path))
    probe.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m")
    pm = probe.compile()
    v = pm.mesh_vert[pm.mesh_vertadr[0]: pm.mesh_vertadr[0] + pm.mesh_vertnum[0]]
    ext = np.maximum(v.max(0) - v.min(0), 1e-6)  # compiled mesh is centred at its centre of mass
    scale = np.asarray(obj.size, float) / ext
    spec.add_mesh(name="m", file=os.path.abspath(path), scale=list(scale))
    root = spec.worldbody.add_body(name="root")
    if dynamic and obj.movable:
        root.add_freejoint(name="free")
    color = [c / 255.0 for c in PALETTE.get(obj.proxy_class, PALETTE["object"])] + [1.0]
    root.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m", rgba=color,
                  mass=max(obj.mass, 1e-3), friction=[obj.friction, 0.005, 0.0001])
    return spec


# --- library entry points ------------------------------------------------------------------------


def assign_rough_assets(scene, kinds=None):
    """Give every recognised object a rough model (keeps assets already set)."""
    for o in scene.things():
        if o.asset:
            continue
        kind = ROUGH_KIND.get(normalize_category(o.category))
        if kind and (kinds is None or kind in kinds):
            o.asset = f"rough:{kind}"
    return scene


def build_asset_specs(scene, dynamic=False):
    """{object id: AssetSpec} for every object with an asset."""
    out = {}
    for o in scene.objects:
        if not o.asset:
            continue
        kind, _, arg = o.asset.partition(":")
        if kind == "menagerie":
            spec, qpos, ctrl = robot_spec(arg)
            out[o.id] = AssetSpec(spec, offset=[0.0, 0.0, -o.size[2] / 2.0], qpos=qpos, ctrl=ctrl,
                                  kind="robot")
        elif kind == "rough":
            out[o.id] = AssetSpec(rough_spec(o, arg, dynamic))
        elif kind == "mesh":
            out[o.id] = AssetSpec(mesh_spec(o, arg, dynamic), kind="mesh")
        else:
            raise ValueError(f"unknown asset {o.asset!r} on {o.id}")
    return out
