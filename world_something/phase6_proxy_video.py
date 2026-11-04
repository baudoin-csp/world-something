"""Phase 6: proxy video. Render the simulation along a camera path in flat proxy colours, cut it
into renderer chunks and pick each chunk's context photos.

  python -m world_something.phase6_proxy_video --sim outputs/room/phase5 --out outputs/room/phase6 \\
      [--path auto | static:I | dolly:I:D | photos:I,J,K | orbit[:R[:H]] | file:cams.json]

Outputs in --out:
  proxy.mp4                 the whole proxy video (24 fps)
  manifest.json             frame rate, resolution, chunk ranges, palette, the program's description
  chunk_XX/proxy.mp4        124 frames; chunk k > 0 starts with the last 34 frames of chunk k-1
  chunk_XX/cameras.json     per frame: K and camera-to-world, both world and relative to the
                            chunk's first context photo (Mundo's alignment)
  chunk_XX/context.json     the chunk's context photos (<= 9), most informative first, with K and
                            pose relative to the first one
  chunk_XX/ids.npz          (--save-ids) object index per pixel and frame, for training masks
"""

import argparse
import json
import os
import shutil

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 6: proxy video and renderer chunks.")
    p.add_argument("--sim", required=True, help="Phase 5 output folder")
    p.add_argument("--out", required=True)
    p.add_argument("--path", default="auto", help="camera path (see module docstring)")
    p.add_argument("--size", type=int, default=640, help="long side of the video in pixels")
    p.add_argument("--chunk", type=int, default=124)
    p.add_argument("--prefix", type=int, default=34)
    p.add_argument("--max-context", type=int, default=9)
    p.add_argument("--save-ids", action="store_true")
    args = p.parse_args(argv)

    from world_something.cameras import align_to_first, as4
    from world_something.codeworld import World
    from world_something.proxy_video import (camera_path, chunk_ranges, render_frames, select_context,
                                             visible_points, write_video)
    from world_something.scene import PALETTE, Scene

    scene = Scene.load(os.path.join(args.sim, "scene_sim.json"))
    tz = np.load(os.path.join(args.sim, "trajectory.npz"))
    traj = {"times": tz["times"], "qpos": tz["qpos"], "fps": int(tz["fps"]),
            "poses": {str(k): tz["poses"][:, i] for i, k in enumerate(tz["pose_ids"])}}
    world = World(scene).build()
    if world.m.nq != traj["qpos"].shape[1]:
        raise SystemExit(f"model has nq={world.m.nq} but the trajectory has {traj['qpos'].shape[1]}: "
                         "scene_sim.json and trajectory.npz come from different runs")
    with open(os.path.join(args.sim, "program.json")) as f:
        program = json.load(f)

    ranges = chunk_ranges(len(traj["qpos"]), args.chunk, args.prefix)
    n_total = ranges[-1][1]
    qpos = np.concatenate([traj["qpos"], np.repeat(traj["qpos"][-1:], n_total - len(traj["qpos"]), 0)])
    K, c2ws, w, h = camera_path(args.path, scene, traj, n_total, long_side=args.size)
    print(f"Phase 6: rendering {n_total} frames at {w}x{h} ({len(traj['qpos'])} simulated, "
          f"{len(ranges)} chunk(s) of {args.chunk})")
    frames = render_frames(world, qpos, K, c2ws, w, h, depth=True)
    os.makedirs(args.out, exist_ok=True)
    write_video(os.path.join(args.out, "proxy.mp4"), frames["proxy"][:len(traj["qpos"])], traj["fps"])

    manifest = {"sim": os.path.abspath(args.sim), "fps": traj["fps"], "width": w, "height": h,
                "chunk_frames": args.chunk,
                "prefix_frames": args.prefix, "simulated_frames": len(traj["qpos"]),
                "camera_path": args.path, "description": program.get("description", ""),
                "palette": {k: "#%02x%02x%02x" % v for k, v in PALETTE.items()}, "chunks": []}
    for k, (s, e) in enumerate(ranges):
        cdir = os.path.join(args.out, f"chunk_{k:02d}")
        os.makedirs(os.path.join(cdir, "context"), exist_ok=True)
        pts = visible_points(frames["depth"][s:e:4], frames["obj"][s:e:4], K, c2ws[s:e:4])
        ctx = select_context(world, scene, pts, max_views=args.max_context)
        first = as4(scene.cameras[ctx[0]].c2w)
        rel = align_to_first(c2ws[s:e], first)
        write_video(os.path.join(cdir, "proxy.mp4"), frames["proxy"][s:e], traj["fps"])
        with open(os.path.join(cdir, "cameras.json"), "w") as f:
            json.dump({"width": w, "height": h, "K": np.asarray(K).tolist(),
                       "reference": {"photo": scene.cameras[ctx[0]].name, "c2w_world": first.tolist()},
                       "frames": [{"frame": s + i, "c2w_world": as4(c).tolist(), "c2w": r.tolist()}
                                  for i, (c, r) in enumerate(zip(c2ws[s:e], rel))]}, f)
        context = []
        for j, ci in enumerate(ctx):
            cam = scene.cameras[ci]
            ext = os.path.splitext(cam.image)[1]
            dst = os.path.join(cdir, "context", f"{j:02d}_{cam.name}{ext}")
            shutil.copy(cam.image, dst)
            context.append({"photo": cam.name, "file": os.path.abspath(dst), "width": cam.width,
                            "height": cam.height, "K": cam.K,
                            "c2w": align_to_first([cam.c2w], first)[0].tolist()})
        with open(os.path.join(cdir, "context.json"), "w") as f:
            json.dump(context, f, indent=1)
        if args.save_ids:
            np.savez_compressed(os.path.join(cdir, "ids.npz"), ids=np.stack(frames["obj"][s:e]),
                                objects=np.array([o.id for o in scene.objects]))
        manifest["chunks"].append({"index": k, "frames": [s, e], "dir": os.path.abspath(cdir),
                                   "context": [c["photo"] for c in context],
                                   "padded_frames": max(0, e - len(traj["qpos"]))})
        print(f"  chunk {k}: frames {s}-{e - 1}, context photos {[c['photo'] for c in context]}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Phase 6: wrote {os.path.join(args.out, 'proxy.mp4')} and {len(ranges)} chunk(s)")


if __name__ == "__main__":
    main()
