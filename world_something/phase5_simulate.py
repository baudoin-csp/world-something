"""Phase 5: simulate interactions in the proxy world (MuJoCo, 960 Hz, recorded at 24 fps).

  python -m world_something.phase5_simulate --scene outputs/room/phase4/scene.json --out outputs/room/phase5 --demo
  python -m world_something.phase5_simulate --scene ... --out ... --program my_events.json
  ANTHROPIC_API_KEY=... python -m world_something.phase5_simulate --scene ... --out ... \\
      --prompt "the robot drives to the stool and a ball bounces off the bed" [--review 2]

The program is a JSON list of timed events (see codeworld.py); --demo builds one from the scene;
--prompt has the Claude director write it (and --review N lets it watch keyframes of the result and
revise it up to N times).

Outputs in --out:
  program.json       the event program that ran
  scene_sim.json     the scene as simulated (spawned objects, settled poses, frozen objects);
                     Phase 6 rebuilds the same MuJoCo model from it
  trajectory.npz     qpos per frame (24 fps), times, and every object's root pose per frame
  sim.json           settle report, event log, timings
  keyframes/         proxy renders of a few frames from the photo camera that sees the action best
"""

import argparse
import json
import os
import time

import numpy as np


def keyframes(world, traj, scene, n=6, long_side=512):
    from world_something.proxy_video import best_photo_camera, moved_objects, photo_view, render_frames

    moved = moved_objects(traj)
    cam = scene.cameras[best_photo_camera(scene, traj, moved or [o for o in traj["poses"]][:1])]
    K, c2w, w, h = photo_view(cam, long_side)
    idx = np.unique(np.linspace(0, len(traj["qpos"]) - 1, n).round().astype(int))
    frames = render_frames(world, traj["qpos"][idx], K, c2w, w, h)["proxy"]
    return [(f"t = {traj['times'][i]:.2f} s (camera of photo {cam.name})", f) for i, f in zip(idx, frames)], moved


def simulate(scene, program, settle_s=1.0, fill_chunks=True):
    from world_something.codeworld import World
    from world_something.proxy_video import chunk_frames

    world = World(scene)
    t0 = time.time()
    duration = world.load_program(program, settle_s=settle_s)
    if fill_chunks:  # run on to the end of the last renderer chunk instead of freezing the end
        duration = (chunk_frames(int(round(duration * world.fps)) + 1) - 1) / world.fps
    t1 = time.time()
    traj = world.run(duration)
    timing = {"build_and_settle_s": round(t1 - t0, 2), "simulate_s": round(time.time() - t1, 2),
              "simulated_s": duration, "physics_hz": round(1.0 / world.timestep), "fps": world.fps}
    return world, traj, timing


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 5: code world simulation.")
    p.add_argument("--scene", required=True, help="Phase 4 (or Phase 3) scene.json")
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--program", help="event program JSON")
    g.add_argument("--demo", action="store_true", help="room + physics demo built from the scene")
    g.add_argument("--prompt", help="text for the Claude director (needs ANTHROPIC_API_KEY)")
    p.add_argument("--review", type=int, default=1, help="director revisions after watching the result")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--settle", type=float, default=1.0, help="seconds to let the scene rest first")
    p.add_argument("--exact-duration", action="store_true",
                   help="stop at the program's duration instead of the end of the last 124-frame chunk")
    args = p.parse_args(argv)

    from world_something.codeworld import EventError, demo_program
    from world_something.mjscene import save_png
    from world_something.scene import Scene

    os.makedirs(args.out, exist_ok=True)
    base = Scene.load(args.scene)
    director, history = None, []
    if args.program:
        with open(args.program) as f:
            program = json.load(f)
    elif args.demo:
        program = demo_program(base)
    else:
        from world_something.codeworld import World
        from world_something.director import Director, scene_brief, to_program

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--prompt needs ANTHROPIC_API_KEY in the environment")
        probe = World(Scene.from_dict(base.to_dict())).build()
        director = Director(effort=args.effort)
        program = to_program(director.first(args.prompt, scene_brief(base, probe)))
        probe.cs.close()

    for attempt in range(args.review + 3):
        scene = Scene.from_dict(base.to_dict())
        try:
            world, traj, timing = simulate(scene, program, args.settle, not args.exact_duration)
        except EventError as e:
            if director is None:
                raise SystemExit(f"invalid program: {e}")
            print(f"Phase 5: program rejected ({e}); asking the director to fix it")
            from world_something.director import to_program
            program = to_program(director.fix(str(e)))
            continue
        frames, moved = keyframes(world, traj, scene)
        history.append({"program": program, "moved": moved})
        if director is None or len([h for h in history]) > args.review:
            break
        from world_something.director import to_program
        answer = director.review(frames, {"moved_objects": moved, "events": world.log[:40],
                                          "settle": world.settle_report})
        if answer.get("satisfied"):
            print("Phase 5: the director is satisfied with the result")
            break
        program = to_program(answer)
        print("Phase 5: the director revised the program")

    with open(os.path.join(args.out, "program.json"), "w") as f:
        json.dump(program, f, indent=1)
    scene.meta["phase5"] = {"program": os.path.abspath(os.path.join(args.out, "program.json"))}
    scene.save(os.path.join(args.out, "scene_sim.json"))
    np.savez_compressed(os.path.join(args.out, "trajectory.npz"), times=traj["times"], qpos=traj["qpos"],
                        fps=traj["fps"], timestep=traj["timestep"],
                        pose_ids=np.array(list(traj["poses"])), poses=np.stack(list(traj["poses"].values()), 1))
    with open(os.path.join(args.out, "sim.json"), "w") as f:
        json.dump({"settle": world.settle_report, "timing": timing, "moved_objects": moved,
                   "event_log": world.log, "director": director.log if director else None},
                  f, indent=1, default=str)
    for k, (caption, img) in enumerate(frames):
        save_png(os.path.join(args.out, "keyframes", f"{k:02d}.png"), img)
    print(f"Phase 5: {timing['simulated_s']} s simulated in {timing['simulate_s']} s "
          f"({len(traj['qpos'])} frames at {traj['fps']} fps); moved: {', '.join(moved) or 'nothing'}")
    if world.settle_report["frozen"]:
        print(f"Phase 5: frozen while settling (unsupported or overlapping): {world.settle_report['frozen']}")


if __name__ == "__main__":
    main()
