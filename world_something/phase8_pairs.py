"""Phase 8a: training pairs (real video <-> proxy video) for the renderer's fine-tune.

  python -m world_something.phase8_pairs --capture data/ca1m-val-42898570 --out outputs/pairs/42898570 \\
      [--src-fps 10] [--windows 8] [--size 640]

A CA-1M capture has real frames with ground-truth poses, laser depth and 3D boxes. From it:
  1. a proxy world of the capture (ground-truth boxes, floor/walls/ceiling from the laser depth),
     built exactly like Phase 3's initial proxy, from a pool of photos spread over the capture
  2. windows of 124 output frames (24 fps; the capture's frames are repeated to reach 24 fps)
  3. per window: the real frames, the proxy rendered along the same cameras, up to 9 context photos
     from the pool that lie outside the window (the renderer must fetch appearance from other views),
     cameras relative to the first context photo, and a caption from the visible objects
Each pair folder: real.mp4, proxy.mp4, cameras.json, context.json, context/, caption.txt.
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


def build_pool(capture, out_dir, pool_size):
    """Oracle Phase 1/2 outputs for `pool_size` frames spread over the capture."""
    from world_something import ca1m

    ca1m.select(capture, out_dir, pool_size)
    p1, p2 = ca1m.oracle(capture, os.path.join(out_dir, "frames.json"), out_dir)
    return p1, p2


def windows(n_src, src_fps, out_frames=124, fps=24, count=8):
    """[(src index per output frame)] for up to `count` evenly spaced windows."""
    span = int(np.ceil(out_frames * src_fps / fps))
    if n_src < span:
        return []
    starts = np.unique(np.linspace(0, n_src - span, count).round().astype(int))
    return [[s + min(int(i * src_fps / fps), span - 1) for i in range(out_frames)] for s in starts]


def window_agreement(data_dir, tags, src, proxy_depths, w, h, every=12):
    """Fraction of laser-depth pixels the proxy matches within max(10 %, 5 cm), over the window."""
    from world_something import ca1m

    agree, total = 0, 0
    for k in range(0, len(src), every):
        g = ca1m.load_gt_frame(data_dir, tags[src[k]])
        d = g["depth"]
        pd = np.asarray(Image.fromarray(proxy_depths[k]).resize((d.shape[1], d.shape[0]), Image.NEAREST))
        valid = (d > 0) & (d < 10)
        ok = valid & (pd > 0) & (np.abs(pd - d) <= np.maximum(0.1 * d, 0.05))
        agree += int(ok.sum())
        total += int(valid.sum())
    return agree / max(total, 1)


def caption(scene, visible_ids, motion):
    labels = [scene.objects[k].label for k in visible_ids if scene.objects[k].proxy_class == "object"]
    seen = list(dict.fromkeys(labels))[:8]
    room = "bedroom" if any("bed" in l or "night stand" in l for l in seen) else "room"
    return (f"A handheld camera {motion} through a real {room}"
            + (f" with {', '.join(seen[:-1])} and {seen[-1]}" if len(seen) > 1 else (f" with a {seen[0]}" if seen else ""))
            + ". Natural indoor lighting, realistic materials, no people.")


def motion_words(c2ws):
    p = np.array([c[:3, 3] for c in c2ws])
    fwd = np.array([c[:3, 2] for c in c2ws])
    dist = float(np.linalg.norm(p[-1] - p[0]))
    turn = float(np.degrees(np.arccos(np.clip(np.dot(fwd[0], fwd[-1]), -1, 1))))
    if dist < 0.15 and turn < 10:
        return "holds almost still"
    if turn > 2 * np.degrees(dist):
        return "pans slowly"
    return "moves slowly"


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 8a: real/proxy training pairs from a CA-1M capture.")
    p.add_argument("--capture", required=True, help="CA-1M capture folder (python -m world_something.ca1m fetch)")
    p.add_argument("--out", required=True)
    p.add_argument("--src-fps", type=float, default=10.0, help="frame rate of the capture's frames")
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--pool", type=int, default=24, help="photos the proxy world and context come from")
    p.add_argument("--size", type=int, default=640, help="long side of the videos")
    p.add_argument("--scene", help="use this proxy world (e.g. Phase 3 with --agent run on <out>/pool) "
                                   "instead of the ground-truth boxes; same world frame")
    p.add_argument("--min-agreement", type=float, default=0.5,
                   help="skip windows whose proxy depth agrees with the laser depth on less than this fraction")
    args = p.parse_args(argv)

    from world_something import ca1m
    from world_something.cameras import align_to_first, as4, scale_K
    from world_something.codeworld import World
    from world_something.proxy_video import render_frames, select_context, visible_points, write_video
    from world_something.scene import initial_proxy

    os.makedirs(args.out, exist_ok=True)
    pool_dir = os.path.join(args.out, "pool")
    p1, p2 = build_pool(args.capture, pool_dir, args.pool)
    from world_something.scene import Scene

    scene = Scene.load(args.scene) if args.scene else initial_proxy(p1, p2)[0]
    world = World(scene).build()  # nothing moves: no settling needed
    with open(os.path.join(pool_dir, "frames.json")) as f:
        pool = json.load(f)
    data_dir = pool["data_dir"]
    tags = ca1m.frame_tags(data_dir)
    shift = np.eye(4)
    with open(os.path.join(p1, "cameras.json")) as f:
        shift[2, 3] = -json.load(f)["gravity"]["floor_offset_m"]
    pool_index = {t: tags.index(t) for t in pool["tags"]}

    made = 0
    for wi, src in enumerate(windows(len(tags), args.src_fps, count=args.windows)):
        frames_rgb, c2ws, K = [], [], None
        for i in src:
            g = ca1m.load_gt_frame(data_dir, tags[i])
            img = Image.open(os.path.join(data_dir, f"{tags[i]}.wide", "image.png")).convert("RGB")
            s = args.size / max(img.size)
            w, h = int(round(img.width * s / 32)) * 32, int(round(img.height * s / 32)) * 32
            frames_rgb.append(np.asarray(img.resize((w, h), Image.BICUBIC)))
            c2ws.append(shift @ g["c2w"])
            K = scale_K(g["K"], w / img.width, h / img.height)
        h, w = frames_rgb[0].shape[:2]
        qpos = [world.d.qpos.copy()] * len(src)
        rend = render_frames(world, qpos, K, c2ws, w, h, depth=True)
        agreement = window_agreement(data_dir, tags, src, rend["depth"], w, h)
        if agreement < args.min_agreement:
            print(f"  window {wi}: skipped, proxy depth agrees with the laser on {agreement:.0%}")
            continue
        # context: pool photos outside this window, chosen by coverage of what the window sees
        lo, hi = min(src), max(src)
        outside = [k for k, cam in enumerate(scene.cameras) if not lo - 5 <= pool_index[cam.name] <= hi + 5]
        if not outside:
            continue
        pts = visible_points(rend["depth"][::6], rend["obj"][::6], K, c2ws[::6])
        sub = type(scene)(objects=scene.objects, cameras=[scene.cameras[k] for k in outside], meta=scene.meta)
        ctx = [outside[k] for k in select_context(world, sub, pts)]
        pdir = os.path.join(args.out, f"pair_{wi:03d}")
        os.makedirs(os.path.join(pdir, "context"), exist_ok=True)
        write_video(os.path.join(pdir, "real.mp4"), frames_rgb, 24)
        write_video(os.path.join(pdir, "proxy.mp4"), rend["proxy"], 24)
        first = as4(scene.cameras[ctx[0]].c2w)
        rel = align_to_first(c2ws, first)
        with open(os.path.join(pdir, "cameras.json"), "w") as f:
            json.dump({"width": w, "height": h, "K": np.asarray(K).tolist(),
                       "reference": {"photo": scene.cameras[ctx[0]].name, "c2w_world": first.tolist()},
                       "frames": [{"frame": i, "src": tags[s_], "c2w_world": as4(c).tolist(), "c2w": r.tolist()}
                                  for i, (s_, c, r) in enumerate(zip(src, c2ws, rel))]}, f)
        context = []
        for j, k in enumerate(ctx):
            cam = scene.cameras[k]
            dst = os.path.join(pdir, "context", f"{j:02d}_{cam.name}.png")
            shutil.copy(cam.image, dst)
            context.append({"photo": cam.name, "file": os.path.abspath(dst), "width": cam.width,
                            "height": cam.height, "K": cam.K, "c2w": align_to_first([cam.c2w], first)[0].tolist()})
        with open(os.path.join(pdir, "context.json"), "w") as f:
            json.dump(context, f, indent=1)
        ids = np.unique(np.concatenate([o[o >= 0].ravel() for o in rend["obj"][::12]]))
        with open(os.path.join(pdir, "caption.txt"), "w") as f:
            f.write(caption(scene, ids.tolist(), motion_words(c2ws)))
        with open(os.path.join(pdir, "quality.json"), "w") as f:
            json.dump({"depth_agreement": agreement, "source_frames": [tags[lo], tags[hi]]}, f)
        made += 1
        print(f"  pair {wi}: source frames {tags[lo]}..{tags[hi]}, {len(ctx)} context photos, "
              f"depth agreement {agreement:.0%}")
    print(f"Phase 8a: {made} pairs -> {args.out}")


if __name__ == "__main__":
    main()
