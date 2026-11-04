"""Phase 3: the proxy world, a scene program of gravity-aligned boxes that matches the photos.

  1. initial proxy (deterministic): Phase 2's boxes + floor/walls/ceiling estimated from Phase 1's
     points, objects snapped onto their supports, default physics per category
  2. optional agent refinement (--agent): Claude Opus 5.5 reviewer/editor rounds (agent.py)

Outputs in --out:
  scene.json            the proxy world (objects, cameras, meta) read by Phases 4-6
  initial_scene.json    the deterministic proxy before the agent
  renders/              per view: proxy colours, overlay on the photo, depth check
  metrics.json          depth agreement with Phase 1 per view, before and after the agent
  agent_log/            the agent's tool calls, results and token usage (with --agent)
"""

import argparse
import json
import os

import numpy as np


def save_renders(ws, out_dir, tag=""):
    from world_something.mjscene import overlay, save_png

    rdir = os.path.join(out_dir, "renders")
    for i, v in enumerate(ws.views):
        r = ws.compiled().render(v.K, v.c2w, v.width, v.height, depth=False)
        save_png(os.path.join(rdir, f"{i:02d}{tag}_proxy.png"), r["proxy"])
        save_png(os.path.join(rdir, f"{i:02d}{tag}_overlay.png"),
                 overlay(ws.photos[i], r, ws.scene, v.K, v.c2w))
        _, masks = ws.depth_check(i)
        save_png(os.path.join(rdir, f"{i:02d}{tag}_depthcheck.png"), ws.error_map(i, masks))


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 3: proxy world from Phase 1-2 outputs.")
    p.add_argument("--phase1", required=True)
    p.add_argument("--phase2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-score", type=float, default=0.0, help="drop Phase 2 boxes below this score")
    p.add_argument("--agent", action="store_true", help="refine with the Claude reviewer/editor loop "
                                                       "(needs ANTHROPIC_API_KEY)")
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--max-edits", type=int, default=60)
    p.add_argument("--time-limit", type=float, default=3600, help="seconds for the agent")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--image-size", type=int, default=768, help="long side of images shown to the agent")
    args = p.parse_args(argv)

    from world_something.agent import Workspace, run
    from world_something.scene import initial_proxy

    os.makedirs(args.out, exist_ok=True)
    scene, points = initial_proxy(args.phase1, args.phase2, min_score=args.min_score)
    scene.save(os.path.join(args.out, "initial_scene.json"))
    ws = Workspace(scene, points, args.phase1, max_side=args.image_size)
    score0, stats0 = ws.score()
    print(f"Phase 3: initial proxy with {len(scene.objects)} boxes, depth agreement {score0:.3f}")
    metrics = {"initial": {"depth_agreement": score0, "views": stats0, "objects": len(scene.objects)}}

    if args.agent:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--agent needs ANTHROPIC_API_KEY in the environment")
        save_renders(ws, args.out, tag="_initial")
        scene, history = run(scene, points, args.phase1, args.out, max_rounds=args.max_rounds,
                             effort=args.effort, max_edits=args.max_edits,
                             time_limit_s=args.time_limit, max_side=args.image_size)
        ws = Workspace(scene, points, args.phase1, max_side=args.image_size)
        score1, stats1 = ws.score()
        metrics["final"] = {"depth_agreement": score1, "views": stats1, "objects": len(scene.objects)}
        metrics["agent"] = history
        print(f"Phase 3: after the agent {len(scene.objects)} boxes, depth agreement {score1:.3f}")

    scene.meta["phase3"] = {"agent": bool(args.agent)}
    scene.save(os.path.join(args.out, "scene.json"))
    save_renders(ws, args.out)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    print(f"Phase 3: wrote {os.path.join(args.out, 'scene.json')} and renders/")


if __name__ == "__main__":
    main()
