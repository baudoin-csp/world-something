"""Phase 7: render the proxy chunks into video.

  python -m world_something.phase7_render --phase6 outputs/room/phase6 --out outputs/room/phase7 \\
      --backend h3 [--lora phase8/lora --adapter phase8/adapter.pt] [--tiny]
  python -m world_something.phase7_render --phase6 ... --out ... --backend sglang --server http://localhost:30011
  python -m world_something.phase7_render --phase6 ... --out ... --backend ibr

Chunks are rendered in order. Each chunk after the first is conditioned on the frames the
previous chunk produced for the overlap (h3: its 34-frame prefix latents; sglang: the frame where
the chunk starts, as a first-frame picture), and the overlap is dropped when stitching.

Outputs: render.mp4 (whole video), chunk_XX/render.mp4, side_by_side.mp4 (proxy | render),
render.json (backend, timings).
"""

import argparse
import json
import os
import time

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 7: render proxy chunks with MiniMax-H3 (or a stand-in).")
    p.add_argument("--phase6", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=["h3", "sglang", "ibr"], default="h3")
    p.add_argument("--model", default="MiniMaxAI/MiniMax-H3", help="h3: diffusers model id or path")
    p.add_argument("--lora", help="h3: LoRA folder from Phase 8")
    p.add_argument("--adapter", help="h3: camera/role adapter weights from Phase 8")
    p.add_argument("--tiny", action="store_true", help="h3: random tiny weights (tests the path on a CPU)")
    p.add_argument("--steps", type=int, default=20, help="h3: Euler steps")
    p.add_argument("--server", default="http://localhost:30011", help="sglang: server URL")
    p.add_argument("--dry-run", action="store_true", help="sglang: only write the requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--description", help="what happens (default: the Phase 5 program's description)")
    args = p.parse_args(argv)

    from world_something.proxy_video import write_video
    from world_something.renderer import (MundoH3, ReprojectionStandIn, SGLangRef2VA, load_chunk, stitch)
    from PIL import Image

    with open(os.path.join(args.phase6, "manifest.json")) as f:
        manifest = json.load(f)
    description = args.description or manifest.get("description", "")
    prefix = manifest["prefix_frames"]
    if args.backend == "h3":
        backend = MundoH3(model=args.model, lora=args.lora, adapter=args.adapter, tiny=args.tiny,
                          steps=args.steps, seed=args.seed)
    elif args.backend == "sglang":
        backend = SGLangRef2VA(args.server, seed=args.seed, dry_run=args.dry_run)
    else:
        backend = ReprojectionStandIn(manifest["sim"])
    os.makedirs(args.out, exist_ok=True)
    rendered, timings = [], []
    previous = None
    for c in manifest["chunks"]:
        chunk = load_chunk(c["dir"])
        cdir = os.path.join(args.out, f"chunk_{c['index']:02d}")
        os.makedirs(cdir, exist_ok=True)
        t0 = time.time()
        if args.backend == "h3":
            frames = backend.render(chunk, description, None if previous is None else previous[-prefix:], cdir)
        elif args.backend == "sglang":
            first = None
            if previous is not None:
                first = os.path.join(cdir, "first_frame.png")
                Image.fromarray(previous[-prefix]).save(first)
            frames = backend.render(chunk, description, first, cdir)
        else:
            frames = backend.render(chunk, description)
        timings.append(round(time.time() - t0, 1))
        if frames is None:  # dry run
            print(f"  chunk {c['index']}: request written to {cdir}/request.json")
            continue
        write_video(os.path.join(cdir, "render.mp4"), frames, manifest["fps"])
        rendered.append(frames)
        previous = frames
        print(f"  chunk {c['index']}: {len(frames)} frames in {timings[-1]} s")
    if not rendered:
        return
    video = stitch(rendered, prefix)
    write_video(os.path.join(args.out, "render.mp4"), video, manifest["fps"])
    from world_something.renderer import load_video

    proxy = load_video(os.path.join(args.phase6, "proxy.mp4"))
    n = min(len(proxy), len(video))
    write_video(os.path.join(args.out, "side_by_side.mp4"),
                [np.concatenate([proxy[i], video[i]], axis=1) for i in range(n)], manifest["fps"])
    with open(os.path.join(args.out, "render.json"), "w") as f:
        json.dump({"backend": args.backend, "tiny": args.tiny, "lora": args.lora, "chunks": len(rendered),
                   "frames": len(video), "seconds_per_chunk": timings, "description": description}, f, indent=1)
    print(f"Phase 7: {len(video)} frames -> {os.path.join(args.out, 'render.mp4')} (+ side_by_side.mp4)")


if __name__ == "__main__":
    main()
