"""Phase 4: replace proxy boxes with assets and add robots.

  python -m world_something.phase4_assets --scene outputs/room/phase3/scene.json --out outputs/room/phase4 \\
      --rough --robot stretch3@1.2,0.4,90 --robot panda@on=makeup_vanity,180

  --rough               rough procedural models for every recognised category (assets.ROUGH_KIND)
  --robot NAME@X,Y[,YAW] or NAME@on=OBJECT_ID[,YAW] or NAME@auto
                        a MuJoCo Menagerie robot: panda, google_robot, g1, stretch3; `auto` puts
                        mobile robots on the freest floor spot and arms on the largest table
  --mesh ID=PATH        use a mesh (.obj/.stl) for object ID, e.g. a TRELLIS output
  --trellis-crops       write a crop of every object from the photo that shows it best, for an
                        image-to-3D model (TRELLIS) on a GPU machine: see `trellis` below

TRELLIS (GPU, separate environment with microsoft/TRELLIS installed):
  python -m world_something.phase4_assets trellis --crops outputs/room/phase4/crops --out outputs/room/phase4/meshes
then pass the meshes back with --mesh ID=outputs/room/phase4/meshes/ID.obj.

Outputs: scene.json (Phase 3's scene with `asset` fields and robots) and previews/.
"""

import argparse
import json
import os
import sys

import numpy as np


def parse_robot(arg, scene):
    from world_something.assets import ROBOTS, place_robot

    from world_something.assets import free_floor_spot, surface_for_robot

    name, _, where = arg.partition("@")
    if name not in ROBOTS:
        raise SystemExit(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}")
    if where in ("", "auto"):  # mobile robots on free floor, arms on the best table-height surface
        if not ROBOTS[name]["free"]:
            found = surface_for_robot(scene)
            if found:
                return place_robot(scene, name, scene.get(found[0]).center[:2], found[1], on=found[0])
        x, y, yaw = free_floor_spot(scene, ROBOTS[name]["footprint"])
        return place_robot(scene, name, [x, y], yaw)
    parts = [p for p in where.split(",") if p]
    if parts and parts[0].startswith("on="):
        on = parts[0][3:]
        yaw = float(parts[1]) if len(parts) > 1 else 0.0
        support = scene.get(on)
        return place_robot(scene, name, support.center[:2], yaw, on=on)
    if len(parts) < 2:
        raise SystemExit(f"--robot {arg}: use NAME@X,Y[,YAW] or NAME@on=OBJECT_ID[,YAW]")
    return place_robot(scene, name, [float(parts[0]), float(parts[1])],
                       float(parts[2]) if len(parts) > 2 else 0.0)


def best_crops(scene, out_dir, margin=0.15, min_pixels=900):
    """For each object, the photo where most of it is visible, cropped around it."""
    from PIL import Image, ImageOps

    from world_something.mjscene import CompiledScene

    os.makedirs(out_dir, exist_ok=True)
    cs = CompiledScene(scene)
    best = {}
    for cam in scene.cameras:
        s = 640.0 / max(cam.width, cam.height)
        w, h = int(cam.width * s), int(cam.height * s)
        K = np.array(cam.K, float)
        K[:2] *= s
        obj = cs.render(K, cam.c2w, w, h, depth=False)["obj"]
        counts = np.bincount(obj[obj >= 0].ravel(), minlength=len(scene.objects))
        for k, n in enumerate(counts):
            if n >= min_pixels and n > best.get(k, (0,))[0]:
                v, u = np.nonzero(obj == k)
                best[k] = (n, cam, [u.min() / s, v.min() / s, (u.max() + 1) / s, (v.max() + 1) / s])
    cs.close()
    written = {}
    for k, (n, cam, (u0, v0, u1, v1)) in best.items():
        o = scene.objects[k]
        if o.proxy_class in ("floor", "wall", "ceiling") or o.asset:
            continue
        mu, mv = margin * (u1 - u0), margin * (v1 - v0)
        img = ImageOps.exif_transpose(Image.open(cam.image)).convert("RGB")
        box = (max(0, int(u0 - mu)), max(0, int(v0 - mv)), min(img.width, int(u1 + mu)), min(img.height, int(v1 + mv)))
        path = os.path.join(out_dir, f"{o.id}.png")
        img.crop(box).save(path)
        written[o.id] = {"crop": path, "photo": cam.image, "box": box, "visible_pixels": int(n)}
    with open(os.path.join(out_dir, "crops.json"), "w") as f:
        json.dump(written, f, indent=1)
    return written


def run_trellis(crops_dir, out_dir, model="microsoft/TRELLIS-image-large", seed=1):
    """Image-to-3D for every crop with TRELLIS (needs a CUDA GPU and TRELLIS installed)."""
    from PIL import Image
    import trimesh
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    os.makedirs(out_dir, exist_ok=True)
    pipe = TrellisImageTo3DPipeline.from_pretrained(model)
    pipe.cuda()
    with open(os.path.join(crops_dir, "crops.json")) as f:
        crops = json.load(f)
    for oid, info in crops.items():
        out = pipe.run(Image.open(info["crop"]), seed=seed)
        glb = postprocessing_utils.to_glb(out["gaussian"][0], out["mesh"][0], simplify=0.95, texture_size=1024)
        glb_path = os.path.join(out_dir, f"{oid}.glb")
        glb.export(glb_path)
        mesh = trimesh.load(glb_path, force="mesh")
        # TRELLIS outputs are y-up; the scene is z-up
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        mesh.export(os.path.join(out_dir, f"{oid}.obj"))
        print(f"  {oid}: {glb_path}")


def save_previews(scene, out_dir, dynamic=False, n=4):
    """Proxy colours | shaded MuJoCo render, from n of the photo cameras (always valid viewpoints)."""
    from world_something.assets import build_asset_specs
    from world_something.cameras import scale_K
    from world_something.mjscene import CompiledScene, save_png, side_by_side

    cs = CompiledScene(scene, dynamic=dynamic, asset_specs=build_asset_specs(scene, dynamic))
    picks = sorted(set(np.linspace(0, len(scene.cameras) - 1, min(n, len(scene.cameras))).round().astype(int)))
    for i in picks:
        cam = scene.cameras[i]
        s = 480.0 / max(cam.width, cam.height)
        w, h = int(cam.width * s), int(cam.height * s)
        out = cs.render(scale_K(cam.K, w / cam.width, h / cam.height), cam.c2w, w, h, depth=False, rgb=True)
        save_png(os.path.join(out_dir, "previews", f"view_{i:02d}.png"), side_by_side(out["proxy"], out["rgb"]))
    cs.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "trellis":
        p = argparse.ArgumentParser(description="TRELLIS image-to-3D for Phase 4 crops (GPU).")
        p.add_argument("--crops", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="microsoft/TRELLIS-image-large")
        a = p.parse_args(argv[1:])
        run_trellis(a.crops, a.out, a.model)
        return
    p = argparse.ArgumentParser(description="Phase 4: assets and robots for the proxy world.")
    p.add_argument("--scene", required=True, help="Phase 3 scene.json")
    p.add_argument("--out", required=True)
    p.add_argument("--rough", action="store_true", help="rough procedural models for known categories")
    p.add_argument("--robot", action="append", default=[], help="NAME@X,Y[,YAW] or NAME@on=OBJECT_ID[,YAW]")
    p.add_argument("--mesh", action="append", default=[], help="ID=PATH (.obj/.stl)")
    p.add_argument("--trellis-crops", action="store_true")
    p.add_argument("--no-previews", action="store_true")
    args = p.parse_args(argv)

    from world_something.assets import assign_rough_assets
    from world_something.scene import Scene

    scene = Scene.load(args.scene)
    os.makedirs(args.out, exist_ok=True)
    if args.trellis_crops:
        crops = best_crops(scene, os.path.join(args.out, "crops"))
        print(f"Phase 4: {len(crops)} crops for TRELLIS -> {os.path.join(args.out, 'crops')}")
    for m in args.mesh:
        oid, _, path = m.partition("=")
        scene.get(oid).asset = f"mesh:{os.path.abspath(path)}"
    if args.rough:
        assign_rough_assets(scene)
    for r in args.robot:
        o = parse_robot(r, scene)
        print(f"Phase 4: robot {o.id} at {np.round(o.center, 2).tolist()}")
    scene.meta["phase4"] = {"rough": args.rough, "robots": args.robot, "meshes": args.mesh}
    scene.save(os.path.join(args.out, "scene.json"))
    n = sum(1 for o in scene.objects if o.asset)
    print(f"Phase 4: {n} objects with assets -> {os.path.join(args.out, 'scene.json')}")
    if not args.no_previews:
        save_previews(scene, args.out)


if __name__ == "__main__":
    main()
