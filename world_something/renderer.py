"""Phase 7 backends: turn a proxy chunk (proxy video + cameras + context photos) into video.

  h3      Mundo's renderer: MiniMax-H3 conditioned on the proxy video, the context photos and the
          cameras (h3.py), with the LoRA and camera adapter from Phase 8. Runs on GPUs with the
          released weights, or with tiny random weights (--tiny) to test the path on a CPU.
  sglang  zero-shot MiniMax-H3 Ref2VA through an SGLang server (`sglang serve --model-variant
          ref2va`): the proxy video is the source video to re-render, the context photos are
          reference images, and chunks after the first start from the previous chunk's frame.
  ibr     a CPU stand-in with no diffusion model: every proxy pixel is textured from the context
          photos by reprojection (with occlusion tests); moving objects carry their texture along
          with their rigid motion. Useful for checking geometry, timing and stitching end to end.

Each backend returns the chunk's frames as uint8 (T, H, W, 3) at the proxy video's resolution.
"""

import base64
import json
import math
import os
import time

import numpy as np
from PIL import Image


def load_video(path):
    import imageio.v2 as imageio

    with imageio.get_reader(path) as r:
        return np.stack([f for f in r])


def resize_frames(frames, width, height):
    return np.stack([np.asarray(Image.fromarray(f).resize((width, height), Image.BICUBIC)) for f in frames])


def load_chunk(chunk_dir):
    with open(os.path.join(chunk_dir, "cameras.json")) as f:
        cams = json.load(f)
    with open(os.path.join(chunk_dir, "context.json")) as f:
        context = json.load(f)
    return {"dir": chunk_dir, "proxy": load_video(os.path.join(chunk_dir, "proxy.mp4")), "cameras": cams,
            "context": context}


# --- prompt for MiniMax-H3 --------------------------------------------------------------------------


def h3_prompt(description, context, has_prefix, scene_summary=""):
    """A Ref2VA prompt in H3's structured format (skills/h3-prompt-writing/references/ref-en.txt)."""
    n = len(context)
    pics = ", ".join(f"<Picture {i + 1}>" for i in range(n))
    lines = ["subject_definitions:",
             f"<Subject 1> is the room shown in {pics}: its furniture, materials, colours and lighting.",
             "<Video 1> is the source video: a flat-shaded proxy render of the same room in which every "
             "surface has a solid label colour (grey floor, blue-grey walls, white furniture, orange new "
             "objects and robots). It fixes the camera motion, the layout and every object's motion."]
    if has_prefix:
        lines.append(f"<Picture {n + 1}> is the first frame of [Shot 1].")
    task = "[video editing + reference generation" + (" + keyframe completion]" if has_prefix else "]")
    lines += ["", "summary:",
              f"{task} The target video is an edited version of <Video 1>: the same shot, camera path and "
              "motion, re-rendered photorealistically as <Subject 1>. Solid label colours are replaced by the "
              "real materials of the room; orange objects become real objects of their kind.",
              "", "retention_analysis:",
              "<Subject 1> (appears in [Shot 1]): fully_preserved - the room's appearance from the pictures.",
              "<Video 1> (camera movement, layout and motion): fully_preserved - geometry and timing are kept; "
              "only the appearance changes."]
    if has_prefix:
        lines.append(f"<Picture {n + 1}> ([Shot 1] first frame): fully_preserved - the video continues from it.")
    lines += ["", "detailed_description:",
              "The target video is in realistic photographic style. [Shot 1] A single continuous shot that "
              f"follows the camera motion of <Video 1> through <Subject 1>. {description}".strip(),
              scene_summary, "", "overall_soundscape:", "Quiet indoor room tone with the soft physical sounds "
              "of the objects moving.", "", "non_diegetic_music:", "None."]
    return "\n".join(l for l in lines if l is not None)


# --- sglang backend ----------------------------------------------------------------------------------


class SGLangRef2VA:
    def __init__(self, url, short_edge=768, seed=0, poll_s=10.0, timeout_s=3600.0, dry_run=False):
        self.url, self.short_edge, self.seed = url.rstrip("/"), short_edge, seed
        self.poll_s, self.timeout_s, self.dry_run = poll_s, timeout_s, dry_run

    def request(self, chunk, description, prefix_frame_path=None):
        ctx = chunk["context"][: 8 if prefix_frame_path else 9]  # Ref2VA: <= 9 images
        conditions = [{"type": "video", "uri": "file://" + os.path.join(chunk["dir"], "proxy.mp4"), "role": "reference"}]
        conditions += [{"type": "image", "uri": "file://" + c["file"], "role": "reference"} for c in ctx]
        if prefix_frame_path:
            conditions.append({"type": "image", "uri": "file://" + os.path.abspath(prefix_frame_path), "role": "reference"})
        n_frames = len(chunk["proxy"])
        h, w = chunk["proxy"].shape[1:3]
        return {"task": "ref2va", "prompt": h3_prompt(description, ctx, bool(prefix_frame_path)),
                "conditions": conditions,
                "target": {"short_edge": self.short_edge, "aspect_ratio": _aspect(w, h),
                           "duration_seconds": max(5, math.ceil(n_frames / 24.0))},
                "seed": self.seed}

    def render(self, chunk, description, prefix_frame_path=None, out_dir=None):
        import requests

        body = self.request(chunk, description, prefix_frame_path)
        if out_dir:
            with open(os.path.join(out_dir, "request.json"), "w") as f:
                json.dump(body, f, indent=1)
        if self.dry_run:
            return None
        r = requests.post(f"{self.url}/v1/videos", json=body, timeout=120)
        r.raise_for_status()
        vid = r.json()["id"]
        t0 = time.time()
        while True:
            s = requests.get(f"{self.url}/v1/videos/{vid}", timeout=60).json()
            if s.get("status") in ("completed", "succeeded"):
                break
            if s.get("status") in ("failed", "error", "cancelled"):
                raise RuntimeError(f"H3 request {vid} failed: {s}")
            if time.time() - t0 > self.timeout_s:
                raise TimeoutError(f"H3 request {vid} still {s.get('status')} after {self.timeout_s} s")
            time.sleep(self.poll_s)
        path = os.path.join(out_dir or ".", "h3_raw.mp4")
        with open(path, "wb") as f:
            f.write(requests.get(f"{self.url}/v1/videos/{vid}/content", timeout=600).content)
        frames = load_video(path)
        h, w = chunk["proxy"].shape[1:3]
        frames = _match_length(frames, len(chunk["proxy"]))
        return resize_frames(frames, w, h)


def _aspect(w, h):
    for name, r in (("16:9", 16 / 9), ("4:3", 4 / 3), ("1:1", 1.0), ("3:4", 3 / 4), ("9:16", 9 / 16)):
        if abs(w / h - r) < 0.03:
            return name
    return "auto"


def _match_length(frames, n):
    if len(frames) >= n:
        return frames[:n]
    return np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])


# --- h3 backend (Mundo conditioning) -----------------------------------------------------------------


class MundoH3:
    """Proxy-conditioned MiniMax-H3 (h3.py). `components` needs .transformer and .vae; with
    `tiny=True` they are random-weight stand-ins and the text is hashed (no Qwen3-VL)."""

    def __init__(self, model="MiniMaxAI/MiniMax-H3", lora=None, adapter=None, tiny=False, device=None,
                 steps=20, seed=0, context_short_edge=480, dtype="bfloat16"):
        import torch

        from world_something import h3

        self.h3, self.tiny, self.steps, self.seed = h3, tiny, steps, seed
        self.context_short_edge = context_short_edge if not tiny else 64
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if tiny:
            self.vae, self.transformer = h3.tiny_components()
            self.components = None
        else:
            from diffusers import ModularPipeline

            pipe = ModularPipeline.from_pretrained(model, workflow="ref2va")
            pipe.load_components(torch_dtype=getattr(torch, dtype))
            pipe.to(self.device)
            self.components, self.vae, self.transformer = pipe, pipe.vae, pipe.transformer
        adapter_module = h3.attach_adapter(self.transformer)
        if adapter:
            adapter_module.load_state_dict(torch.load(adapter, map_location="cpu"))
        if lora:
            self.transformer.load_lora_adapter(lora, adapter_name="mundo")
        self.transformer.to(self.device).eval()
        self.vae.to(self.device)

    def _text(self, prompt, context_images):
        if self.tiny:
            return self.h3.tiny_text(prompt, len(context_images))
        return self.h3.encode_presentation(self.components, prompt, context_images)

    def condition(self, chunk, description, prefix_frames=None, size=None):
        """Latents, text and cameras of one chunk (size: (w, h) canvas, default = proxy video's)."""
        h3 = self.h3
        proxy = chunk["proxy"]
        H0, W0 = proxy.shape[1:3]
        W, Hh = size or (W0 // 32 * 32, H0 // 32 * 32)
        proxy = resize_frames(proxy, W, Hh)
        cams = chunk["cameras"]
        K = np.array(cams["K"], float)
        K[0] *= W / cams["width"]
        K[1] *= Hh / cams["height"]
        c2ws = [np.array(f["c2w"]) for f in cams["frames"]]
        imgs, ctx_lat, ctx_rays = [], [], []
        for c in chunk["context"][:9]:
            img = Image.open(c["file"]).convert("RGB")
            cw, ch = h3.context_size(img.width, img.height, self.context_short_edge)
            arr = np.asarray(img.resize((cw, ch), Image.BICUBIC))
            imgs.append(arr)
            ctx_lat.append(h3.vae_encode(self.vae, arr[None]))
            Kc = np.array(c["K"], float)
            Kc[0] *= cw / c["width"]
            Kc[1] *= ch / c["height"]
            ctx_rays.append(h3.plucker_rows(Kc, np.array(c["c2w"]), cw, ch))
        text, tags = self._text(h3_prompt(description, chunk["context"], prefix_frames is not None), imgs)
        proxy_lat = h3.vae_encode(self.vae, proxy)
        T = proxy_lat.shape[2]
        spans = h3.latent_frame_spans(T)
        rays = h3.plucker_rows(K, c2ws, W, Hh, latent_spans=spans)
        prefix = None
        if prefix_frames is not None:
            prefix = h3.vae_encode(self.vae, resize_frames(prefix_frames, W, Hh))
        import torch

        plucker = torch.cat(ctx_rays + [rays, rays])  # context rows, proxy rows, target rows
        cond = h3.Condition(text=text, text_tags=tags, context=[c.to(self.device) for c in ctx_lat],
                            proxy=proxy_lat.to(self.device), plucker=plucker,
                            prefix=None if prefix is None else prefix.to(self.device))
        return cond, tuple(proxy_lat.shape[1:]), (W, Hh)

    def render(self, chunk, description, prefix_frames=None, out_dir=None):
        H0, W0 = chunk["proxy"].shape[1:3]
        size = None
        if self.tiny:  # keep the CPU test small: about 1/5 of the canvas
            size = (max(32, W0 // 5 // 32 * 32), max(32, H0 // 5 // 32 * 32))
        cond, shape, (W, Hh) = self.condition(chunk, description, prefix_frames, size=size)
        lat = self.h3.sample(self.transformer, cond, shape, steps=self.steps, seed=self.seed, device=self.device)
        frames = self.h3.vae_decode(self.vae, lat)
        return resize_frames(_match_length(frames, len(chunk["proxy"])), W0, H0)


# --- ibr stand-in ------------------------------------------------------------------------------------


class ReprojectionStandIn:
    """Texture the proxy from the context photos (no learning). Needs the Phase 5 simulation."""

    def __init__(self, sim_dir, photo_side=480, max_photos=4):
        from world_something.codeworld import World
        from world_something.scene import Scene

        self.scene = Scene.load(os.path.join(sim_dir, "scene_sim.json"))
        tz = np.load(os.path.join(sim_dir, "trajectory.npz"))
        self.qpos = tz["qpos"]
        self.world = World(self.scene).build()
        self.photo_side, self.max_photos = photo_side, max_photos
        self._photo_cache = {}

    def _photo(self, c):
        """Photo pixels, K, c2w (world) and its proxy depth at the initial state."""
        import mujoco

        from world_something.cameras import scale_K

        key = c["photo"]
        if key not in self._photo_cache:
            cam = next(x for x in self.scene.cameras if x.name == key)
            s = self.photo_side / max(cam.width, cam.height)
            w, h = int(cam.width * s), int(cam.height * s)
            K = scale_K(cam.K, w / cam.width, h / cam.height)
            img = np.asarray(Image.open(cam.image).convert("RGB").resize((w, h), Image.BILINEAR)).astype(np.float32)
            d = self.world.d
            d.qpos[:] = self.qpos[0]
            mujoco.mj_kinematics(self.world.m, d)
            r = self.world.cs.render(K, cam.c2w, w, h, depth=True)
            self._photo_cache[key] = (img, K, np.array(cam.c2w), r["depth"], r["obj"])
        return self._photo_cache[key]

    def render(self, chunk, description=None, prefix_frames=None, out_dir=None):
        import mujoco
        from scipy.spatial.transform import Rotation

        from world_something.cameras import project
        from world_something.scene import PALETTE

        cams = chunk["cameras"]
        K = np.array(cams["K"], float)
        W, H = cams["width"], cams["height"]
        m, d = self.world.m, self.world.d
        roots = {oid: b for oid, b in self.world.root.items()}
        idx_of = {o.id: i for i, o in enumerate(self.scene.objects)}
        photos = [self._photo(c) for c in chunk["context"][: self.max_photos]]
        # object root poses at the initial state (what the photos show)
        d.qpos[:] = self.qpos[0]
        mujoco.mj_kinematics(m, d)
        pose0 = {idx_of[o]: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for o, b in roots.items()}
        new_obj = {i for i, o in enumerate(self.scene.objects) if o.proxy_class == "new_object" or o.asset and o.asset.startswith("menagerie:")}
        frames = []
        Kinv = np.linalg.inv(K)
        vv, uu = np.mgrid[0:H, 0:W]
        rays = (Kinv @ np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5, np.ones(H * W)])).T
        for f in cams["frames"]:
            q = self.qpos[min(f["frame"], len(self.qpos) - 1)]
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            c2w = np.array(f["c2w_world"])
            r = self.world.cs.render(K, c2w, W, H, depth=True)
            depth, obj = r["depth"].ravel(), r["obj"].ravel()
            X = (rays * depth[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
            X0 = X.copy()
            for k, (p0, R0) in pose0.items():  # move points on moved objects back to the initial pose
                sel = obj == k
                if not sel.any():
                    continue
                b = roots[self.scene.objects[k].id]
                pf, Rf = d.xpos[b], d.xmat[b].reshape(3, 3)
                X0[sel] = (X[sel] - pf) @ Rf @ R0.T + p0
            color = np.zeros((H * W, 3), np.float32)
            best = np.full(H * W, -1.0, np.float32)
            valid = depth > 0
            for img, Kp, c2wp, dp, op in photos:
                uv, z = project(Kp, c2wp, X0)
                h, w = dp.shape
                ok = valid & (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
                ui, vi = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
                vis = np.abs(dp[vi, ui] - z[ok]) < np.maximum(0.04 * z[ok], 0.04)
                ids = np.nonzero(ok)[0][vis]
                # prefer photos that look from a similar direction and are close
                view_c = c2w[:3, 3]
                dirs_now = X[ids] - view_c
                dirs_photo = X0[ids] - c2wp[:3, 3]
                cosang = np.sum(dirs_now * dirs_photo, 1) / (np.linalg.norm(dirs_now, axis=1) * np.linalg.norm(dirs_photo, axis=1) + 1e-9)
                score = cosang / (1.0 + z[ok][vis])
                better = score > best[ids]
                tgt = ids[better]
                u, v = uv[tgt, 0], uv[tgt, 1]
                u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
                a, bb = (u - u0)[:, None], (v - v0)[:, None]
                c = (img[v0, u0] * (1 - a) * (1 - bb) + img[v0, u0 + 1] * a * (1 - bb)
                     + img[v0 + 1, u0] * (1 - a) * bb + img[v0 + 1, u0 + 1] * a * bb)
                color[tgt] = c
                best[tgt] = score[better]
            out = color.reshape(H, W, 3)
            filled = (best > -1).reshape(H, W)
            # objects not in the photos (robots, spawned things): shaded proxy colour
            palette = np.array([PALETTE.get(o.proxy_class, PALETTE["object"]) for o in self.scene.objects] + [PALETTE["unknown"]], np.float32)
            shade = 0.55 + 0.45 * np.clip(1.0 - depth.reshape(H, W) / 8.0, 0, 1)[..., None]
            flat = palette[obj.reshape(H, W)] * shade
            newmask = np.isin(obj.reshape(H, W), list(new_obj))
            out[newmask] = flat[newmask]
            hole = ~filled & ~newmask
            if hole.any():
                import cv2

                small = hole & (cv2.blur(hole.astype(np.float32), (9, 9)) < 0.9)
                out = cv2.inpaint(out.clip(0, 255).astype(np.uint8), small.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
                big = hole & ~small
                out[big] = 0.5 * out[big] + 0.5 * flat[big]
            frames.append(out.clip(0, 255).astype(np.uint8))
        return np.stack(frames)


def stitch(chunks_frames, prefix):
    """Concatenate chunk videos, dropping each later chunk's prefix (it repeats the previous one)."""
    out = [chunks_frames[0]]
    for f in chunks_frames[1:]:
        out.append(f[prefix:])
    return np.concatenate(out)
