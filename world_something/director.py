"""Phase 5 director: Claude Opus 5.5 writes the code world's event program from a text prompt.

Analysis by synthesis again: the director writes a program, the program is simulated, keyframes
of the result are rendered from a photo camera, and the director sees them and revises the program
(or says it is satisfied). Programs come back as JSON through structured outputs, so they always
parse; World.load_program validates them against the scene.
Requests use server-side refusal fallbacks (fallbacks="default").
"""

import base64
import io
import json
import math

import numpy as np
from PIL import Image

from world_something.agent import FALLBACK_BETA, MODEL

_VEC = {"type": "array", "items": {"type": "number"}}
EVENT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["type", "at"],
    "properties": {
        "type": {"type": "string", "enum": ["spawn", "push", "throw", "drop", "place", "open",
                                            "robot_pose", "robot_reach", "robot_drive"]},
        "at": {"type": "number"}, "duration": {"type": "number"},
        "object": {"type": "string"}, "robot": {"type": "string"}, "label": {"type": "string"},
        "shape": {"type": "string", "enum": ["box", "sphere", "cylinder"]},
        "size": _VEC, "position": _VEC, "velocity": _VEC, "spin": _VEC, "direction": _VEC,
        "target": _VEC, "force": {"type": "number"}, "height": {"type": "number"},
        "amount": {"type": "number"}, "distance": {"type": "number"},
        "turn_deg": {"type": "number"}, "yaw_deg": {"type": "number"}, "mass": {"type": "number"},
        "restitution": {"type": "number"},
        "targets": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "value"],
            "properties": {"name": {"type": "string"}, "value": {"type": "number"}}}},
        "note": {"type": "string"},
    },
}
PROGRAM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["description", "duration", "events", "satisfied"],
    "properties": {"description": {"type": "string"}, "duration": {"type": "number"},
                   "events": {"type": "array", "items": EVENT_SCHEMA},
                   "satisfied": {"type": "boolean"}},
}

SYSTEM = """You direct a physics simulation of a real room. The room was reconstructed from photos as
boxes (and some articulated models and robots); MuJoCo simulates it. You write the event program
that makes the requested action happen, as JSON matching the schema.

World: metres, z up, floor at z = 0; yaw is about +z. Time: seconds from the start; keep the
program under the duration you set (normally 4-10 s; video is 24 fps).
Events (fields besides type/at):
- spawn: label, shape (box|sphere|cylinder), size [sx,sy,sz] (sphere: diameter), position, optional
  velocity, mass, restitution. New objects are drawn in orange. Released at `at`.
- push: object, direction [x,y,z], force (N), duration. Sliding needs more than friction
  (~0.6 x mass x 9.81 N); much more tips things over.
- throw: object, velocity [x,y,z] m/s, optional spin. drop: object, height. place: object, position, yaw_deg.
- open: object (an articulated cabinet/drawers/door), amount 0-1, duration.
- robot_pose: robot, targets [{name, value}] by actuator or joint name, duration.
- robot_reach: robot, target [x,y,z] for its end effector, duration.
- robot_drive: robot (free base only), distance (m, along its heading), turn_deg, duration.
Only objects with movable=true can move. Leave room between events so their effects are visible.
Set "satisfied" to false when you write or change a program; when shown the result, set it to true
only if the simulation shows what was asked (then return the same program)."""


def scene_brief(scene, world=None):
    rows = []
    for o in scene.objects:
        r = {"id": o.id, "label": o.label, "class": o.proxy_class,
             "center": [round(c, 2) for c in o.center], "size": [round(s, 2) for s in o.size],
             "yaw_deg": round(math.degrees(o.yaw)), "top": round(o.top, 2), "movable": o.movable,
             "mass": round(o.mass, 1)}
        if o.asset:
            r["asset"] = o.asset
        rows.append(r)
    robots = {}
    if world is not None:
        import mujoco

        m = world.m
        for rid, kind in world.robots.items():
            acts = {}
            for a in sorted(set(world._robot_actuators(rid).values())):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                if not name and m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
                acts[name.split("/", 1)[1]] = [round(float(x), 2) for x in m.actuator_ctrlrange[a]]
            robots[rid] = {"model": kind, "free_base": rid in world.free, "actuators_ctrlrange": acts}
    return {"objects": rows, "robots": robots}


def to_program(data):
    """Structured-output JSON -> World program (targets lists become dicts)."""
    prog = {"description": data.get("description", ""), "duration": float(data.get("duration", 6.0)),
            "events": []}
    for e in data.get("events", []):
        e = {k: v for k, v in e.items() if v is not None}
        if "targets" in e:
            e["targets"] = {t["name"]: t["value"] for t in e["targets"]}
        prog["events"].append(e)
    return prog


def _image_block(img):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, format="PNG", optimize=True)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


class Director:
    def __init__(self, effort="high", client=None, max_tokens=32000):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.effort, self.max_tokens = effort, max_tokens
        self.messages, self.log = [], []

    def _ask(self, content):
        self.messages.append({"role": "user", "content": content})
        with self.client.beta.messages.stream(
                model=MODEL, max_tokens=self.max_tokens, system=SYSTEM, messages=self.messages,
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": PROGRAM_SCHEMA}},
                cache_control={"type": "ephemeral"}, betas=[FALLBACK_BETA], fallbacks="default") as s:
            msg = s.get_final_message()
        self.messages.append({"role": "assistant", "content": msg.content})
        if msg.stop_reason == "refusal":
            raise RuntimeError("the director declined this prompt")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        data = json.loads(text)
        self.log.append({"stop_reason": msg.stop_reason, "response": data,
                         "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}})
        return data

    def first(self, prompt, brief):
        return self._ask([{"type": "text", "text": f"Scene:\n{json.dumps(brief)}\n\nRequested action: {prompt}"}])

    def fix(self, error):
        return self._ask([{"type": "text", "text": f"That program failed validation: {error}. Return a corrected program."}])

    def review(self, keyframes, sim_log):
        content = [{"type": "text", "text": "Here is the simulated result: keyframes from one camera "
                                            "(proxy colours: grey floor, blue-grey walls, white "
                                            "objects, orange new objects and robots), then the log."}]
        for caption, img in keyframes:
            content += [{"type": "text", "text": caption}, _image_block(img)]
        content.append({"type": "text", "text": f"Log: {json.dumps(sim_log)[:6000]}\nDoes it show the "
                                                "requested action? If yes, return the same program with "
                                                "satisfied=true; if not, a revised program."})
        return self._ask(content)
