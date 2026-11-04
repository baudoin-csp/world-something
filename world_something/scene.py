"""The scene program's state: every object as a gravity-aligned box with physics, plus the cameras.

This is the "proxy world" of the plan. Phases 3-6 all read and write this one structure:
  * Phase 3 builds it from Phase 1-2 outputs (this file) and refines it (agent.py)
  * Phase 4 adds library assets to it (assets.py)
  * Phase 5 compiles it to MuJoCo and simulates it (codeworld.py)
  * Phase 6 renders it into flat-colour proxy frames (mjscene.py, proxy_video.py)

World frame: metres, z up, floor at z = 0 (Phase 1's frame). Cameras: OpenCV, camera-to-world.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

# Flat proxy colours, one per structural class (Mundo's proxy palette).
PALETTE = {
    "object": (238, 238, 238),
    "new_object": (243, 148, 66),
    "floor": (112, 112, 112),
    "wall": (169, 182, 198),
    "ceiling": (187, 162, 220),
    "window": (86, 200, 216),
    "door": (173, 119, 86),
    "unknown": (48, 52, 59),
}
STRUCTURE = ("floor", "wall", "ceiling")

# Typical masses (kg) by category; anything else gets density x volume.
TYPICAL_MASS = {
    "chair": 6.0, "office chair": 12.0, "desk chair": 10.0, "stool": 4.0, "armchair": 25.0,
    "sofa": 50.0, "couch": 50.0, "table": 25.0, "coffee table": 15.0, "side table": 8.0,
    "desk": 30.0, "bed": 60.0, "cabinet": 40.0, "nightstand": 15.0, "bedside table": 15.0,
    "dresser": 45.0, "wardrobe": 70.0, "shelf": 30.0, "bookshelf": 40.0, "lamp": 3.0,
    "floor lamp": 5.0, "trash bin": 2.0, "trash can": 2.0, "box": 2.0, "bag": 3.0,
    "backpack": 2.0, "suitcase": 5.0, "pillow": 0.5, "cushion": 0.5, "blanket": 1.5,
    "monitor": 4.2, "laptop": 2.0, "television": 12.0, "tv": 12.0, "keyboard": 0.8,
    "mouse": 0.1, "plant": 4.0, "potted plant": 4.0, "bottle": 0.5, "cup": 0.3, "mug": 0.3,
    "bowl": 0.4, "plate": 0.4, "book": 0.6, "apple": 0.2, "ball": 0.45, "kettle": 1.0,
    "electric kettle": 1.0, "microwave": 15.0, "refrigerator": 70.0, "toilet": 30.0,
    "sink": 20.0, "bathtub": 50.0, "basket": 1.0, "clock": 0.5, "vase": 1.0,
    "picture frame": 1.5, "picture": 1.5, "mirror": 8.0, "curtain": 3.0, "radiator": 20.0,
}
SOFT = {"pillow", "cushion", "blanket", "bag", "backpack", "curtain", "sofa", "couch", "bed"}
BOUNCY = {"ball"}
FIXED = {"bed", "wardrobe", "refrigerator", "toilet", "sink", "bathtub", "radiator", "curtain",
         "mirror", "picture", "picture frame", "television", "tv", "window", "door", "shelf",
         "bookshelf", "cabinet", "counter", "kitchen counter", "closet", "rug", "carpet", "poster",
         "door frame", "electrical outlet", "light switch", "plug", "motion sensor", "makeup vanity"}


def normalize_category(label):
    return " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())


def proxy_class_for(category):
    c = normalize_category(category)
    if c in STRUCTURE:
        return c
    if "door" in c or c == "doorway":
        return "door"
    if "window" in c:
        return "window"
    return "object"


def yaw_quat(yaw):
    """MuJoCo quaternion (w, x, y, z) for a rotation about +z."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Obj:
    """One gravity-aligned box: an object, a piece of architecture, or a library asset."""
    id: str
    label: str
    category: str
    proxy_class: str = "object"
    center: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # metres
    size: list = field(default_factory=lambda: [0.1, 0.1, 0.1])    # full extents in the box frame
    yaw: float = 0.0                                                 # radians about +z
    movable: bool = False
    mass: float = 1.0
    friction: float = 0.6
    restitution: float = 0.3
    asset: str = None          # library asset id (Phase 4), e.g. "menagerie:franka_emika_panda"
    source: str = "boxer"      # boxer | agent | architecture | asset | user
    score: float = 1.0
    notes: str = ""

    @property
    def half(self):
        return np.asarray(self.size, float) / 2.0

    @property
    def bottom(self):
        return float(self.center[2] - self.size[2] / 2.0)

    @property
    def top(self):
        return float(self.center[2] + self.size[2] / 2.0)

    def corners(self):
        """The 8 corners in world coordinates (bottom 4, then top 4)."""
        hx, hy, hz = self.half
        local = np.array([[sx * hx, sy * hy, sz * hz] for sz in (-1, 1)
                          for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        return local @ yaw_matrix(self.yaw).T + np.asarray(self.center, float)

    def contains(self, points, margin=0.0):
        local = (np.asarray(points, float) - np.asarray(self.center, float)) @ yaw_matrix(self.yaw)
        return np.all(np.abs(local) <= self.half + margin, axis=1)


@dataclass
class Camera:
    """A context view: one of the input photos with its Phase 1 camera."""
    name: str
    image: str            # path to the photo
    width: int
    height: int
    K: list               # 3x3 intrinsics at (width, height)
    c2w: list             # 4x4 camera-to-world, OpenCV axes


@dataclass
class Scene:
    objects: list = field(default_factory=list)
    cameras: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # --- access -----------------------------------------------------------------------------
    def get(self, obj_id):
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(f"no object {obj_id!r}; have {[o.id for o in self.objects]}")

    def __contains__(self, obj_id):
        return any(o.id == obj_id for o in self.objects)

    def new_id(self, base):
        base = normalize_category(base).replace(" ", "_") or "object"
        if base not in self:
            return base
        i = 2
        while f"{base}_{i}" in self:
            i += 1
        return f"{base}_{i}"

    def add(self, obj):
        if obj.id in self:
            obj.id = self.new_id(obj.id)
        self.objects.append(obj)
        return obj

    def remove(self, obj_id):
        self.objects = [o for o in self.objects if o.id != obj_id]

    def things(self):
        """Objects that are not architecture."""
        return [o for o in self.objects if o.proxy_class not in STRUCTURE]

    # --- I/O ---------------------------------------------------------------------------------
    def to_dict(self):
        return {"objects": [asdict(o) for o in self.objects],
                "cameras": [asdict(c) for c in self.cameras], "meta": self.meta}

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def from_dict(cls, d):
        return cls(objects=[Obj(**o) for o in d.get("objects", [])],
                   cameras=[Camera(**c) for c in d.get("cameras", [])], meta=d.get("meta", {}))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


# --- physics defaults -------------------------------------------------------------------------


def default_physics(category, size):
    """Mass (kg), friction, restitution and movability from the category and box size."""
    c = normalize_category(category)
    volume = float(np.prod(np.maximum(size, 1e-3)))
    mass = TYPICAL_MASS.get(c, float(np.clip(150.0 * volume, 0.1, 150.0)))
    restitution = 0.7 if c in BOUNCY else (0.1 if c in SOFT else 0.3)
    friction = 0.8 if c in SOFT else 0.6
    movable = c not in FIXED and max(size) < 1.2 and mass < 40.0
    return mass, friction, restitution, movable


def make_object(scene, label, center, size, yaw=0.0, category=None, source="boxer", score=1.0,
                proxy_class=None, **physics):
    category = normalize_category(category or label)
    mass, friction, restitution, movable = default_physics(category, size)
    obj = Obj(id=scene.new_id(category), label=str(label), category=category,
              proxy_class=proxy_class or proxy_class_for(category),
              center=[float(x) for x in center], size=[float(max(x, 0.01)) for x in size],
              yaw=float(yaw), movable=movable, mass=mass, friction=friction,
              restitution=restitution, source=source, score=float(score))
    for k, v in physics.items():
        setattr(obj, k, v)
    return scene.add(obj)


# --- Phase 1-2 inputs -------------------------------------------------------------------------


def load_phase1(phase1_dir):
    """Cameras (photo resolution) and world points from Phase 1's cameras.json + geometry.npz."""
    with open(os.path.join(phase1_dir, "cameras.json")) as f:
        cams = json.load(f)
    cameras = [Camera(name=fr["name"], image=fr["file"], width=int(fr["width"]),
                      height=int(fr["height"]), K=fr["K"], c2w=fr["c2w"]) for fr in cams["frames"]]
    geo = np.load(os.path.join(phase1_dir, "geometry.npz"))
    depth, K, c2w = geo["depth"], geo["K"], geo["c2w"]
    pts = []
    rng = np.random.default_rng(0)
    for i in range(len(depth)):
        v, u = np.nonzero(depth[i] > 0)
        if len(u) > 60000:
            keep = rng.choice(len(u), 60000, replace=False)
            u, v = u[keep], v[keep]
        rays = np.linalg.inv(K[i]) @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)
        pts.append((c2w[i, :3, :3] @ (rays * depth[i, v, u]) + c2w[i, :3, 3:4]).T)
    points = np.concatenate(pts) if pts else np.zeros((0, 3))
    return cameras, points, cams


def load_boxes(phase2_dir):
    path = os.path.join(phase2_dir, "boxes.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# --- architecture -----------------------------------------------------------------------------


def _room_yaw(xy):
    """Yaw of the tightest rectangle around the points (rooms are mostly rectangular)."""
    best = (np.inf, 0.0)
    for deg in np.arange(0.0, 90.0, 1.0):
        r = yaw_matrix(-math.radians(deg))[:2, :2]
        q = xy @ r.T
        lo, hi = np.percentile(q, [2, 98], axis=0)
        area = float(np.prod(hi - lo))
        if area < best[0]:
            best = (area, math.radians(deg))
    return best[1]


def estimate_architecture(points, wall_thickness=0.1, support_dist=0.15, min_support=0.01):
    """Floor, walls and ceiling as thin boxes, from the Phase 1 point cloud.

    The room is approximated by the tightest rectangle around the observed points. A wall is
    added only where enough points lie on it, so unobserved sides stay open ("unknown").
    """
    if len(points) < 100:
        return [], {"note": "too few points for architecture"}
    z = points[:, 2]
    top = float(np.percentile(z, 99.5))
    ceiling_band = np.abs(z - top) < 0.05
    ceiling_seen = ceiling_band.mean() > min_support and top > 2.0
    height = top if ceiling_seen else max(top, 2.5)
    walls_pts = points[(z > 0.15) & (z < height - 0.15)]
    xy = (walls_pts if len(walls_pts) > 100 else points)[:, :2]
    yaw = _room_yaw(xy)
    R2 = yaw_matrix(-yaw)[:2, :2]
    q = xy @ R2.T
    lo, hi = np.percentile(q, [0.5, 99.5], axis=0)
    center_local = (lo + hi) / 2.0
    size_xy = hi - lo
    center_xy = yaw_matrix(yaw)[:2, :2] @ center_local
    parts = [dict(id="floor", label="floor", category="floor", proxy_class="floor",
                  center=[center_xy[0], center_xy[1], -0.025],
                  size=[size_xy[0], size_xy[1], 0.05], yaw=yaw)]
    if ceiling_seen:
        parts.append(dict(id="ceiling", label="ceiling", category="ceiling", proxy_class="ceiling",
                          center=[center_xy[0], center_xy[1], height + 0.025],
                          size=[size_xy[0], size_xy[1], 0.05], yaw=yaw))
    sides = []
    for axis in (0, 1):
        for sign, bound in ((-1, lo[axis]), (1, hi[axis])):
            on_wall = np.abs(q[:, axis] - bound) < support_dist
            frac = float(on_wall.mean())
            sides.append((axis, sign, frac))
            if frac < min_support:
                continue
            c = center_local.copy()
            c[axis] = bound + sign * wall_thickness / 2.0
            length = size_xy[1 - axis] + 2 * wall_thickness
            sz = [wall_thickness, length] if axis == 0 else [length, wall_thickness]
            cw = yaw_matrix(yaw)[:2, :2] @ c
            parts.append(dict(id=f"wall_{len(parts)}", label="wall", category="wall",
                              proxy_class="wall", center=[cw[0], cw[1], height / 2.0],
                              size=[sz[0], sz[1], height], yaw=yaw))
    info = {"room_yaw_deg": math.degrees(yaw), "room_size_m": size_xy.tolist(),
            "ceiling_height_m": height, "ceiling_seen": bool(ceiling_seen),
            "wall_support": [{"axis": a, "side": s, "fraction": f} for a, s, f in sides]}
    return parts, info


# --- support snapping -------------------------------------------------------------------------


def _footprints_overlap(a, b):
    ca, cb = np.asarray(a.center[:2]), np.asarray(b.center[:2])
    ra, rb = np.linalg.norm(a.size[:2]) / 2.0, np.linalg.norm(b.size[:2]) / 2.0
    return np.linalg.norm(ca - cb) < 0.7 * (ra + rb)


def snap_to_supports(scene, tol=0.06):
    """Put objects that hover just above the floor or another object's top onto it.

    Reconstructed boxes often float by a few centimetres, which a simulator turns into drops.
    """
    things = sorted(scene.things(), key=lambda o: o.bottom)
    for o in things:
        if o.proxy_class in ("window", "door"):
            continue
        support = 0.0
        for other in things:
            if other is o or other.top > o.bottom + tol or other.top < o.bottom - tol:
                continue
            if _footprints_overlap(o, other):
                support = max(support, other.top)
        if abs(o.bottom - support) <= tol:
            o.center[2] = support + o.size[2] / 2.0
        elif o.movable:
            # Nothing under it: wall-mounted or hanging. Keep it where it is.
            o.movable = False
            o.notes = (o.notes + " attached (no support below)").strip()


def initial_proxy(phase1_dir, phase2_dir, min_score=0.0, min_size=0.03):
    """Deterministic first proxy world: Boxer's boxes plus floor, walls and ceiling."""
    cameras, points, cams = load_phase1(phase1_dir)
    scene = Scene(cameras=cameras, meta={
        "phase1": os.path.abspath(phase1_dir), "phase2": os.path.abspath(phase2_dir),
        "world": cams.get("world", {}), "gravity": cams.get("gravity", {}),
    })
    parts, info = estimate_architecture(points)
    for p in parts:
        scene.add(Obj(source="architecture", movable=False, mass=0.0, **p))
    scene.meta["architecture"] = info
    for b in load_boxes(phase2_dir):
        if b.get("score", 1.0) < min_score or max(b["size"]) < min_size:
            continue
        make_object(scene, b["label"], b["center"], b["size"], b.get("yaw", 0.0),
                    source="boxer", score=b.get("score", 1.0))
    snap_to_supports(scene)
    return scene, points
