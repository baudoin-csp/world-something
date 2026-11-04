"""Phase 5: the code world. The proxy world as a MuJoCo simulation, driven by event programs.

An event program is a list of timed events, as JSON (written by hand or by the Claude director in
director.py) or as calls on World from Python. MuJoCo steps at 960 Hz, the rate Mundo uses, and the
state is recorded at 24 fps (MiniMax-H3's output frame rate) for Phase 6's proxy video.

Events (times in seconds from the start of the program):
  spawn        add a new object (box / sphere / cylinder), released at `at` with a velocity
  push         a force on an object's centre for a while
  throw        set an object's linear (and angular) velocity
  drop         lift an object by `height` and let it fall
  place        teleport an object to a pose
  open         drive an articulated asset's hinge or slide (cabinet door, drawer, door)
  robot_pose   move a robot's actuators (by actuator or joint name) to targets
  robot_reach  move a robot's end effector to a point (damped least-squares IK)
  robot_drive  drive a mobile robot: wheels for stretch3, a carried base for g1
Before the program, `settle` lets the scene come to rest; objects that move are frozen in place
(they were unsupported or interpenetrating) and reported.
"""

import copy
import json
import math

import mujoco
import numpy as np

from world_something.assets import build_asset_specs
from world_something.mjscene import CompiledScene
from world_something.scene import make_object, normalize_category

FPS = 24
TIMESTEP = 1.0 / 960.0
END_EFFECTOR = {"panda": ("body", "hand"), "google_robot": ("site", "gripper"),
                "g1": ("body", "right_wrist_yaw_link"), "stretch3": ("body", "link_gripper_finger_left")}
_HINGE_SLIDE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
EVENT_TYPES = ("spawn", "push", "throw", "drop", "place", "open", "robot_pose", "robot_reach",
               "robot_drive")


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("direction must be non-zero")
    return v / n


def _yaw_qpos_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class EventError(ValueError):
    pass


class World:
    def __init__(self, scene, timestep=TIMESTEP, fps=FPS):
        self.scene = scene
        self.timestep, self.fps = timestep, fps
        self.substeps = max(1, int(round(1.0 / (fps * timestep))))
        self.cs = None
        self.controllers = []   # (t0, t1, fn(world, t)) before every physics step in [t0, t1)
        self.instants = []      # (t, fn(world)) once, at the first step with time >= t
        self.spawns = []        # (obj id, release time, position, yaw, velocity)
        self.frames, self.times = [], []
        self.log = []
        self.settle_report = None

    # --- building ---------------------------------------------------------------------------
    def spawn(self, label, shape="box", size=(0.2, 0.2, 0.2), position=(0.0, 0.0, 1.0), yaw_deg=0.0,
              velocity=(0.0, 0.0, 0.0), at=0.0, mass=None, restitution=None, friction=None):
        if self.cs is not None:
            raise EventError("spawn events must come before the world is built")
        kind = {"box": None, "sphere": "ball", "ball": "ball", "cylinder": "cylinder"}.get(shape, "invalid")
        if kind == "invalid":
            raise EventError(f"spawn shape must be box, sphere or cylinder, not {shape!r}")
        size = [float(s) for s in size]
        if shape in ("sphere", "ball"):
            size = [max(size)] * 3
        o = make_object(self.scene, label, position, size, math.radians(yaw_deg), source="spawn",
                        proxy_class="new_object")
        o.movable = True
        o.asset = f"rough:{kind}" if kind else None
        o.notes = f"spawned at t={at}s"
        for k, v in (("mass", mass), ("restitution", restitution), ("friction", friction)):
            if v is not None:
                setattr(o, k, float(v))
        self.spawns.append((o.id, float(at), np.asarray(position, float), math.radians(yaw_deg),
                            np.asarray(velocity, float)))
        return o.id

    def build(self):
        specs = build_asset_specs(self.scene, dynamic=True)
        self.cs = CompiledScene(self.scene, dynamic=True, timestep=self.timestep, asset_specs=specs)
        self.m, self.d = self.cs.model, self.cs.data
        self._index()
        for k, (oid, at, pos, yaw, vel) in enumerate(self.spawns):  # wait off-stage until released
            self._set_pose(oid, [1000.0 + 3 * k, 1000.0, 1.0], yaw)
        mujoco.mj_forward(self.m, self.d)
        return self

    def _index(self):
        m = self.m
        self.root, self.free = {}, {}
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        for o in self.scene.objects:
            for b in range(1, m.nbody):
                if names[b] == f"body:{o.id}" or (names[b].startswith(f"{o.id}/") and m.body_parentid[b] == 0):
                    self.root[o.id] = b
                    break
            b = self.root.get(o.id)
            if b is not None and m.body_jntnum[b] > 0:
                j = m.body_jntadr[b]
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    self.free[o.id] = (int(m.jnt_qposadr[j]), int(m.body_dofadr[b]))
        self.robots = {o.id: o.asset.split(":", 1)[1] for o in self.scene.objects
                       if o.asset and o.asset.startswith("menagerie:")}

    # --- lookups ----------------------------------------------------------------------------
    def _obj(self, oid):
        if oid not in self.scene:
            raise EventError(f"no object {oid!r}")
        return self.scene.get(oid)

    def _need_free(self, oid):
        self._obj(oid)
        if oid not in self.free:
            raise EventError(f"{oid} is fixed in place (movable=false or mounted); it cannot move")
        return self.free[oid]

    def _robot(self, rid):
        if rid not in self.robots:
            raise EventError(f"{rid!r} is not a robot; robots: {sorted(self.robots)}")
        return self.robots[rid]

    def _robot_actuators(self, rid):
        m, out = self.m, {}
        for a in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            jname = ""
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]) or ""
            for full in (name, jname):
                if full.startswith(f"{rid}/"):
                    out[full[len(rid) + 1:]] = a
        return out

    def _set_pose(self, oid, pos, yaw):
        qa, va = self.free[oid]
        self.d.qpos[qa:qa + 3] = pos
        self.d.qpos[qa + 3:qa + 7] = _yaw_qpos_quat(yaw)
        self.d.qvel[va:va + 6] = 0.0

    def pose(self, oid):
        """World position and yaw (deg) of an object's root."""
        b = self.root.get(oid)
        if b is None:
            o = self._obj(oid)
            return np.asarray(o.center, float), math.degrees(o.yaw)
        R = self.d.xmat[b].reshape(3, 3)
        return self.d.xpos[b].copy(), math.degrees(math.atan2(R[1, 0], R[0, 0]))

    # --- events -----------------------------------------------------------------------------
    def push(self, obj, direction, force=50.0, at=0.0, duration=0.3):
        self._need_free(obj)
        b, F = self.root[obj], _unit(direction) * float(force)

        def ctl(w, t):
            w.d.xfrc_applied[b, :3] += F
        self.controllers.append((float(at), float(at) + float(duration), ctl))

    def throw(self, obj, velocity, spin=(0.0, 0.0, 0.0), at=0.0):
        qa, va = self._need_free(obj)

        def fn(w):
            w.d.qvel[va:va + 3] = velocity
            w.d.qvel[va + 3:va + 6] = spin
        self.instants.append((float(at), fn))

    def drop(self, obj, height=0.5, at=0.0):
        qa, va = self._need_free(obj)

        def fn(w):
            w.d.qpos[qa + 2] += float(height)
            w.d.qvel[va:va + 6] = 0.0
        self.instants.append((float(at), fn))

    def place(self, obj, position, yaw_deg=None, at=0.0):
        self._need_free(obj)

        def fn(w):
            yaw = math.radians(yaw_deg) if yaw_deg is not None else math.radians(w.pose(obj)[1])
            w._set_pose(obj, position, yaw)
        self.instants.append((float(at), fn))

    def open(self, obj, amount=1.0, at=0.0, duration=1.0):
        self._obj(obj)
        m = self.m
        joints = [j for j in range(m.njnt)
                  if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(f"{obj}/")
                  and int(m.jnt_type[j]) in _HINGE_SLIDE]
        if not joints or obj in self.robots:
            raise EventError(f"{obj} has no door or drawer (give it a rough:cabinet, rough:drawers or "
                             "rough:door asset in Phase 4)")
        state = {}

        def ctl(w, t):
            for j in joints:
                qa, da = m.jnt_qposadr[j], m.jnt_dofadr[j]
                lo, hi = m.jnt_range[j]
                if j not in state:
                    state[j] = float(w.d.qpos[qa])
                target = state[j] + smoothstep((t - at) / duration) * (lo + amount * (hi - lo) - state[j])
                inertia = 1.0 / max(float(m.dof_invweight0[da]), 1e-6)  # joint inertia at qpos0
                wn = 2 * math.pi * 2.0  # 2 Hz, critically damped
                w.d.qfrc_applied[da] += inertia * (wn ** 2 * (target - w.d.qpos[qa]) - 2 * wn * w.d.qvel[da])
        self.controllers.append((float(at), math.inf, ctl))

    def robot_pose(self, robot, targets, at=0.0, duration=1.0):
        self._robot(robot)
        acts = self._robot_actuators(robot)
        bad = [k for k in targets if k not in acts]
        if bad:
            raise EventError(f"{robot} has no actuator/joint {bad}; available: {sorted(acts)}")
        goal = {acts[k]: float(np.clip(v, *self.m.actuator_ctrlrange[acts[k]])) for k, v in targets.items()}
        state = {}

        def ctl(w, t):
            if not state:
                state.update({a: float(w.d.ctrl[a]) for a in goal})
            s = smoothstep((t - at) / max(duration, 1e-6))
            for a, g in goal.items():
                w.d.ctrl[a] = state[a] + s * (g - state[a])
        self.controllers.append((float(at), float(at) + float(duration) + self.timestep, ctl))

    def robot_reach(self, robot, target, at=0.0, duration=2.0, iters=200):
        kind = self._robot(robot)
        etype, ename = END_EFFECTOR[kind]
        m = self.m
        objtype = mujoco.mjtObj.mjOBJ_SITE if etype == "site" else mujoco.mjtObj.mjOBJ_BODY
        eid = mujoco.mj_name2id(m, objtype, f"{robot}/{ename}")
        acts = self._robot_actuators(robot)
        joint_acts = [(a, m.actuator_trnid[a, 0]) for a in set(acts.values())
                      if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                      and int(m.jnt_type[m.actuator_trnid[a, 0]]) in _HINGE_SLIDE
                      and m.actuator_biasprm[a, 1] < 0]  # position servos only
        if kind == "g1":
            joint_acts = [(a, j) for a, j in joint_acts
                          if "right_" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                          and ("shoulder" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                               or "elbow" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                               or "wrist" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))]
        target = np.asarray(target, float)
        goal, report = {}, {}

        def solve(w):
            d = mujoco.MjData(m)
            d.qpos[:], d.qvel[:] = w.d.qpos, 0.0
            dofs = [m.jnt_dofadr[j] for _, j in joint_acts]
            for _ in range(iters):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                pos = d.site_xpos[eid] if etype == "site" else d.xpos[eid]
                err = target - pos
                if np.linalg.norm(err) < 0.005:
                    break
                jacp = np.zeros((3, m.nv))
                if etype == "site":
                    mujoco.mj_jacSite(m, d, jacp, None, eid)
                else:
                    mujoco.mj_jacBody(m, d, jacp, None, eid)
                J = jacp[:, dofs]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
                for (a, j), step in zip(joint_acts, np.clip(dq, -0.2, 0.2)):
                    qa = m.jnt_qposadr[j]
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf)
                    d.qpos[qa] = np.clip(d.qpos[qa] + step, max(lo, m.actuator_ctrlrange[a, 0]),
                                         min(hi, m.actuator_ctrlrange[a, 1]))
            mujoco.mj_kinematics(m, d)
            pos = d.site_xpos[eid] if etype == "site" else d.xpos[eid]
            report.update(reach_error_m=round(float(np.linalg.norm(target - pos)), 3))
            for a, j in joint_acts:
                goal[a] = float(d.qpos[m.jnt_qposadr[j]])

        state = {}

        def ctl(w, t):
            if not goal:
                solve(w)
                state.update({a: float(w.d.ctrl[a]) for a in goal})
                self.log.append({"t": round(t, 3), "event": "robot_reach", "robot": robot, **report})
            s = smoothstep((t - at) / max(duration, 1e-6))
            for a, g in goal.items():
                w.d.ctrl[a] = state[a] + s * (g - state[a])
        self.controllers.append((float(at), float(at) + float(duration) + self.timestep, ctl))

    def robot_drive(self, robot, distance=0.0, turn_deg=0.0, at=0.0, duration=2.0):
        """Move a free-base robot along its heading (and turn) on a smooth path.

        The base follows the path kinematically, with the matching velocity so contacts still push
        what it drives into. Stretch's wheels spin at the matching rate; G1 is carried (it has no
        walking policy here) while its joints hold their pose.
        """
        kind = self._robot(robot)
        m = self.m
        if robot not in self.free:
            raise EventError(f"{robot} ({kind}) has a fixed base and cannot drive")
        qa, va = self.free[robot]
        wheels = []
        if kind == "stretch3":
            acts = self._robot_actuators(robot)
            for name in ("left_wheel_vel", "right_wheel_vel"):
                a = acts[name]
                j = m.actuator_trnid[a, 0]
                b = m.jnt_bodyid[j]
                radius = [float(m.geom_size[g, 0]) for g in range(m.ngeom) if m.geom_bodyid[g] == b
                          and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)]
                wheels.append((a, j, b, radius[0] if radius else 0.05))
        state = {}
        T = max(float(duration), 1e-3)

        def carry(w, t):
            if not state:
                state["p0"] = w.d.qpos[qa:qa + 3].copy()
                q = w.d.qpos[qa + 3:qa + 7]
                state["yaw0"] = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            x = min(max((t - at) / T, 0.0), 1.0)
            s, ds = smoothstep(x), 6 * x * (1 - x) / T          # path parameter and its rate
            yaw = state["yaw0"] + s * math.radians(turn_deg)
            # integrate heading along the path: straight segment rotated by the mean heading so far
            heading = state["yaw0"] + 0.5 * s * math.radians(turn_deg)
            fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
            w.d.qpos[qa:qa + 2] = state["p0"][:2] + s * distance * fwd[:2]
            w.d.qpos[qa + 3:qa + 7] = _yaw_qpos_quat(yaw)
            w.d.qvel[va:va + 3] = ds * distance * fwd
            w.d.qvel[va + 3:va + 6] = [0.0, 0.0, ds * math.radians(turn_deg)]
            for a, j, b, r in wheels:  # cosmetic: wheels turn with the motion
                Rb = w.d.xmat[self.root[robot]].reshape(3, 3)
                axis = Rb.T @ (w.d.xmat[b].reshape(3, 3) @ m.jnt_axis[j])
                side = Rb.T @ (w.d.xpos[b] - w.d.xpos[self.root[robot]])
                v = ds * distance + ds * math.radians(turn_deg) * side[1]
                w.d.ctrl[a] = float(np.clip((1.0 if axis[1] > 0 else -1.0) * v / r, *m.actuator_ctrlrange[a]))

        def stop(w):
            for a, *_ in wheels:
                w.d.ctrl[a] = 0.0
        self.controllers.append((float(at), float(at) + T, carry))
        self.instants.append((float(at) + T, stop))

    # --- running ------------------------------------------------------------------------------
    def _step(self):
        d = self.d
        t = d.time
        d.xfrc_applied[:] = 0.0
        d.qfrc_applied[:] = 0.0
        for k, (oid, at, pos, yaw, vel) in enumerate(self.spawns):
            if oid in self._released:
                continue
            if t >= at - 1e-9:
                self._set_pose(oid, pos, yaw)
                qa, va = self.free[oid]
                d.qvel[va:va + 3] = vel
                self._released.add(oid)
            else:
                self._set_pose(oid, [1000.0 + 3 * k, 1000.0, 1.0], yaw)
        due = [e for e in self.instants if e[0] <= t + 1e-9]
        for e in due:
            e[1](self)
            self.instants.remove(e)
        for t0, t1, fn in self.controllers:
            if t0 <= t < t1:
                fn(self, t)
        mujoco.mj_step(self.m, d)

    def settle(self, seconds=1.0, tol=0.05, max_rebuilds=2):
        """Let the scene come to rest. Objects that move more than tol are frozen and reported."""
        if self.cs is None:
            self.build()
        frozen = []
        for _ in range(max_rebuilds + 1):
            start = {oid: self.pose(oid)[0] for oid in self.free if oid not in [s[0] for s in self.spawns]
                     and oid not in self.robots}
            self._released = set(s[0] for s in self.spawns)  # keep spawned objects parked
            saved = (self.instants, self.controllers)
            self.instants, self.controllers = [], []
            for oid, at, pos, yaw, vel in self.spawns:
                self._set_pose(oid, [1000.0 + 3 * [s[0] for s in self.spawns].index(oid), 1000.0, 1.0], yaw)
            for _ in range(int(seconds / self.timestep)):
                self.d.xfrc_applied[:] = 0.0
                mujoco.mj_step(self.m, self.d)
            self.instants, self.controllers = saved
            moved = {oid: float(np.linalg.norm(self.pose(oid)[0] - p)) for oid, p in start.items()}
            bad = [oid for oid, dist in moved.items() if dist > tol]
            if not bad:
                break
            for oid in bad:
                self.scene.get(oid).movable = False
                self.scene.get(oid).notes = (self.scene.get(oid).notes + " frozen by settle").strip()
            frozen += bad
            self.cs.close()
            self.build()
        for oid in self.free:  # write the rest poses back to the scene program
            if oid in self.robots or oid in [s[0] for s in self.spawns]:
                continue
            p, yaw = self.pose(oid)
            o = self.scene.get(oid)
            o.center, o.yaw = [float(x) for x in p], math.radians(yaw)
        self.d.time = 0.0
        self.d.qvel[:] = 0.0
        for k, (oid, at, pos, yaw, vel) in enumerate(self.spawns):
            self._set_pose(oid, [1000.0 + 3 * k, 1000.0, 1.0], yaw)
        mujoco.mj_forward(self.m, self.d)
        self._released = set()
        self.settle_report = {"seconds": seconds, "tolerance_m": tol, "frozen": frozen}
        return self.settle_report

    def run(self, duration):
        if self.cs is None:
            self.build()
        if not hasattr(self, "_released"):
            self._released = set()
        n = int(round(duration * self.fps))
        self._record()
        for _ in range(n):
            for _ in range(self.substeps):
                self._step()
            self._record()
        return self.trajectory()

    def _record(self):
        self.times.append(float(self.d.time))
        self.frames.append(self.d.qpos.copy())

    def trajectory(self):
        """Recorded qpos per frame plus each object's root pose per frame."""
        poses = {}
        m, d = self.m, mujoco.MjData(self.m)
        for oid, b in self.root.items():
            poses[oid] = np.zeros((len(self.frames), 7))
        for f, q in enumerate(self.frames):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            for oid, b in self.root.items():
                poses[oid][f, :3] = d.xpos[b]
                poses[oid][f, 3:] = d.xquat[b]
        return {"times": np.array(self.times), "qpos": np.array(self.frames), "poses": poses,
                "fps": self.fps, "timestep": self.timestep}

    # --- event programs -------------------------------------------------------------------------
    def load_program(self, program, settle_s=1.0):
        """Schedule a JSON event program: {"duration": s, "events": [{"type": ..., ...}, ...]}.

        Spawns are added first, then the world is built and settled (settling can freeze objects
        and rebuild the model), then the other events are scheduled. Returns the duration.
        """
        events = [dict(e) for e in program.get("events", [])]
        for i, e in enumerate(events):
            if e.get("type") not in EVENT_TYPES:
                raise EventError(f"event {i}: unknown type {e.get('type')!r}; use one of {EVENT_TYPES}")
        if self.cs is not None and any(e["type"] == "spawn" for e in events):
            raise EventError("spawn events must be loaded before the world is built")
        for i, e in enumerate(events):
            if e["type"] == "spawn":
                args = {k: v for k, v in e.items() if k not in ("type", "note")}
                try:
                    self.spawn(**args)
                except TypeError as err:
                    raise EventError(f"event {i} (spawn): {err}")
                self.log.append({"scheduled": "spawn", **args})
        if self.cs is None:
            self.build()
        if settle_s and self.settle_report is None:
            self.settle(settle_s)
        for i, e in enumerate(events):
            if e["type"] == "spawn":
                continue
            args = {("obj" if k == "object" else k): v for k, v in e.items() if k not in ("type", "note")}
            try:
                getattr(self, e["type"])(**args)
            except TypeError as err:
                raise EventError(f"event {i} ({e['type']}): {err}")
            self.log.append({"scheduled": e["type"], **args})
        return float(program.get("duration", 5.0))


def demo_program(scene):
    """A room + physics program built from whatever the scene contains."""
    things = {normalize_category(o.category): o for o in scene.things()}
    floor = scene.get("floor") if "floor" in scene else None
    cx, cy = (floor.center[:2] if floor else [0.0, 0.0])
    events = []
    bed = next((o for o in scene.things() if normalize_category(o.category) == "bed"), None)
    drop_at = [bed.center[0], bed.center[1], bed.top + 0.8] if bed else [cx, cy, 1.4]
    events.append({"type": "spawn", "label": "ball", "shape": "sphere", "size": [0.22, 0.22, 0.22],
                   "position": drop_at, "velocity": [0.6, -0.4, 0.0], "at": 0.3, "restitution": 0.7})
    pushable = next((o for o in scene.things() if o.movable and normalize_category(o.category) in
                     ("stool", "chair", "office chair", "box", "bean bag chair", "trash can")), None)
    if pushable:
        d = np.array([cx, cy]) - np.array(pushable.center[:2])
        d = d / (np.linalg.norm(d) or 1.0)
        events.append({"type": "push", "object": pushable.id, "direction": [float(d[0]), float(d[1]), 0.0],
                       "force": 9.0 * max(pushable.mass, 1.0), "at": 1.5, "duration": 0.5})
    door = next((o for o in scene.objects if o.asset in ("rough:cabinet", "rough:drawers", "rough:door")), None)
    if door:
        events.append({"type": "open", "object": door.id, "amount": 0.8, "at": 2.0, "duration": 1.5})
    for o in scene.objects:
        if o.asset == "menagerie:stretch3":
            events.append({"type": "robot_drive", "robot": o.id, "distance": 0.8, "at": 0.5, "duration": 3.0})
            events.append({"type": "robot_pose", "robot": o.id, "targets": {"lift": 0.9, "arm": 0.3},
                           "at": 1.0, "duration": 2.5})
        elif o.asset == "menagerie:panda":
            fwd = np.array([math.cos(o.yaw), math.sin(o.yaw), 0.0])
            target = np.array(o.center) + 0.45 * fwd
            target[2] = o.bottom + 0.35
            events.append({"type": "robot_reach", "robot": o.id, "target": target.round(3).tolist(),
                           "at": 0.5, "duration": 2.5})
    said = []
    for e in events:
        if e["type"] == "spawn":
            said.append(f"a ball drops in{' onto the bed' if bed else ''} and bounces across the floor")
        elif e["type"] == "push":
            said.append(f"the {scene.get(e['object']).label} is pushed across the floor")
        elif e["type"] == "open":
            said.append(f"the {scene.get(e['object']).label} opens")
        elif e["type"] == "robot_drive":
            said.append("a Stretch mobile robot drives forward, raising its lift and extending its arm")
        elif e["type"] == "robot_reach":
            said.append("a Franka Panda arm reaches forward")
    text = "; ".join(said)
    return {"duration": 6.0, "events": events, "description": (text[0].upper() + text[1:] + ".") if text else ""}
