"""Phase 1: cameras and a metric, gravity-aligned point cloud from 1-N photos of a room.

Reused tools:
  * Depth Anything 3, nested model (DA3NESTED-GIANT-LARGE-1.1): intrinsics, camera poses and
    depth for every photo in one forward pass, already in metres.
  * Open3D RANSAC plane fitting: the floor plane gives the gravity axis and z = 0.

World frame written by this script: metres, z up, floor at z = 0 (like Mundo's scene programs).
Cameras are OpenCV (x right, y down, z forward); poses are camera-to-world.

Outputs in --out:
  cameras.json       per photo: K and camera-to-world at the photo's own resolution
  points.ply         fused, coloured point cloud
  geometry.npz       DA3 depth / confidence / intrinsics at processing resolution, aligned poses
  consistency.json   multi-view depth consistency ("reprojections line up")
  boxer_input/       the same data in the CA-1M layout that Boxer reads unchanged (Phase 2)
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps

from world_something import DA3_MODEL

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# Boxer's CA-1M runner is selected by an input name starting with "ca1m".
BOXER_SEQ = "ca1m-phase1"


def list_images(folder):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
    return [os.path.join(folder, f) for f in names]


def load_rgb(path):
    """RGB uint8, with the phone's EXIF rotation applied."""
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def as_4x4(m):
    m = np.asarray(m, dtype=np.float64)
    if m.shape[-2:] == (4, 4):
        return m.copy()
    out = np.tile(np.eye(4), m.shape[:-2] + (1, 1))
    out[..., :3, :4] = m
    return out


def run_da3(images, model_id=DA3_MODEL, process_res=504, use_ray_pose=True, device="cuda"):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id).to(device)
    return model.inference(images, process_res=process_res, use_ray_pose=use_ray_pose)


# --- Gravity ---------------------------------------------------------------------------


def camera_up_hint(c2w):
    """Average 'up' of the photos: people hold phones upright and OpenCV's y axis points down."""
    up = -c2w[:, :3, 1].mean(axis=0)
    return up / np.linalg.norm(up)


def find_floor(points, up_hint, max_tilt_deg=30.0, dist=0.02, max_planes=8, min_frac=0.02):
    """Lowest large plane whose normal is within max_tilt_deg of up_hint (Open3D RANSAC).

    Returns {"normal", "height", "inlier_fraction"} with the normal oriented along up_hint and
    height = n . x for points x on the plane, or None if no such plane exists.
    """
    import open3d as o3d

    o3d.utility.random.seed(0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    total, best = len(points), None
    for _ in range(max_planes):
        if len(pcd.points) < max(100, min_frac * total):
            break
        plane, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=2000)
        n, d = np.asarray(plane[:3]), plane[3]
        n, d = n / np.linalg.norm(n), d / np.linalg.norm(n)
        if n @ up_hint < 0:
            n, d = -n, -d
        tilt = np.degrees(np.arccos(np.clip(n @ up_hint, -1.0, 1.0)))
        frac = len(inliers) / total
        if frac >= min_frac and tilt <= max_tilt_deg and (best is None or -d < best["height"]):
            best = {"normal": n, "height": float(-d), "inlier_fraction": frac}
        pcd = pcd.select_by_index(inliers, invert=True)
    return best


def gravity_transform(points, c2w, voxel=0.02, **floor_kwargs):
    """Rigid transform taking DA3's world to z-up with the floor at z = 0."""
    import open3d as o3d
    from scipy.spatial.transform import Rotation

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    sparse = np.asarray(pcd.voxel_down_sample(voxel).points)
    up_hint = camera_up_hint(c2w)
    floor = find_floor(sparse, up_hint, **floor_kwargs)
    if floor is None:  # no floor in view: trust the upright photos, put z = 0 at the lowest points
        up, height = up_hint, float(np.percentile(sparse @ up_hint, 1.0))
    else:
        up, height = floor["normal"], floor["height"]
    T = np.eye(4)
    T[:3, :3] = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0].as_matrix()
    T[2, 3] = -height
    info = {
        "floor_found": floor is not None,
        "floor_inlier_fraction": None if floor is None else floor["inlier_fraction"],
        "up_in_da3_frame": up.tolist(),
        "camera_up_hint_angle_deg": float(np.degrees(np.arccos(np.clip(up @ up_hint, -1, 1)))),
        "T_world_from_da3": T.tolist(),
    }
    return T, info


# --- Checks ------------------------------------------------------------------------------


def depth_consistency(depth, K, c2w, valid, rel_tol=0.05, max_points=20000, min_overlap=0.1):
    """Project each view's points into every other view and compare with that view's depth.

    A pair counts if at least min_overlap of the points land inside the other image. Occlusions
    lower the inlier ratio a little even for perfect geometry.
    """
    rng = np.random.default_rng(0)
    n, h, w = depth.shape
    w2c = np.linalg.inv(c2w)
    pairs = []
    for i in range(n):
        v, u = np.nonzero(valid[i])
        if len(u) > max_points:
            keep = rng.choice(len(u), max_points, replace=False)
            u, v = u[keep], v[keep]
        rays = np.linalg.inv(K[i]) @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)
        x_w = c2w[i, :3, :3] @ (rays * depth[i, v, u]) + c2w[i, :3, 3:4]
        for j in range(n):
            if i == j:
                continue
            x_j = w2c[j, :3, :3] @ x_w + w2c[j, :3, 3:4]
            z = x_j[2]
            uv = (K[j] @ x_j)[:2] / np.where(z > 1e-6, z, 1.0)
            uj, vj = np.round(uv).astype(int)
            inside = (z > 1e-6) & (uj >= 0) & (uj < w) & (vj >= 0) & (vj < h)
            if len(u) == 0 or inside.mean() < min_overlap:
                continue
            uj, vj, z = uj[inside], vj[inside], z[inside]
            ok = valid[j, vj, uj]
            if not ok.any():
                continue
            rel = np.abs(z[ok] - depth[j, vj[ok], uj[ok]]) / depth[j, vj[ok], uj[ok]]
            pairs.append({"i": i, "j": j, "overlap": float(inside.mean()),
                          "inlier_ratio": float((rel < rel_tol).mean()),
                          "median_rel_error": float(np.median(rel))})
    ratios = [p["inlier_ratio"] for p in pairs]
    return {"rel_tol": rel_tol, "num_pairs": len(pairs),
            "median_inlier_ratio": float(np.median(ratios)) if ratios else None, "pairs": pairs}


# --- Main processing ---------------------------------------------------------------------


def process_prediction(pred, images, conf_percentile=40.0, max_depth=20.0):
    """Turn a DA3 prediction into gravity-aligned cameras, points and per-view data.

    `images` are the original photos (uint8 HxWx3). Kept separate from DA3 so tests can feed a
    synthetic prediction.
    """
    # Private helper of DA3's GLB exporter: unproject depth to world points with colours.
    from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors

    depth = np.asarray(pred.depth, dtype=np.float32)
    n, h, w = depth.shape
    K = np.asarray(pred.intrinsics, dtype=np.float64)
    c2w_da3 = np.linalg.inv(as_4x4(pred.extrinsics))
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    if pred.conf is not None:
        conf = np.asarray(pred.conf)
        valid &= conf >= np.percentile(conf[valid], conf_percentile)
    depth_valid = np.where(valid, depth, 0.0).astype(np.float32)

    pts, cols = _depths_to_world_points_with_colors(
        depth_valid, K, np.linalg.inv(c2w_da3), np.asarray(pred.processed_images), None, 0.0
    )
    T, grav = gravity_transform(pts, c2w_da3)
    c2w = T @ c2w_da3
    pts = pts @ T[:3, :3].T + T[:3, 3]

    # DA3's default resize is a pure per-axis scaling, so intrinsics map back to the original
    # photos exactly, unless photos of different sizes were centre-cropped to a common size.
    same_size = len({im.shape[:2] for im in images}) == 1
    if same_size:
        H0, W0 = images[0].shape[:2]
        K_full = np.diag([W0 / w, H0 / h, 1.0]) @ K
        out_images = list(images)
    else:
        K_full, out_images = K, list(np.asarray(pred.processed_images))

    scale = getattr(pred, "scale_factor", None)
    return {
        "depth": depth_valid, "valid": valid, "K": K, "c2w": c2w,
        "K_full": K_full, "images": out_images, "points": pts, "colors": cols,
        "gravity": grav,
        "is_metric": bool(getattr(pred, "is_metric", False)),
        "metric_scale_factor": None if scale is None else float(scale),
    }


def write_boxer_input(root, geo, names):
    """Write the CA-1M layout read by Boxer's CALoader: image.png, K.json, RT.json, depth.png."""
    os.makedirs(os.path.join(root, "world.gt"), exist_ok=True)
    with open(os.path.join(root, "world.gt", "instances.json"), "w") as f:
        json.dump([], f)  # no ground truth: Phase 2 predicts the boxes
    tags = {}
    for i, name in enumerate(names):
        tag = f"{(i + 1) * 10**9:013d}"  # integer "timestamps" 1 s apart, sortable as strings
        tags[tag] = name
        wide, gt = os.path.join(root, tag + ".wide"), os.path.join(root, tag + ".gt")
        os.makedirs(os.path.join(wide, "image"), exist_ok=True)
        os.makedirs(gt, exist_ok=True)
        Image.fromarray(geo["images"][i]).save(os.path.join(wide, "image.png"))
        mm = np.clip(np.round(geo["depth"][i] * 1000.0), 0, 65535).astype(np.uint16)
        Image.fromarray(mm).save(os.path.join(wide, "depth.png"))
        for path, obj in [(os.path.join(wide, "image", "K.json"), geo["K_full"][i].tolist()),
                          (os.path.join(gt, "RT.json"), geo["c2w"][i].tolist()),
                          (os.path.join(wide, "instances.json"), [])]:
            with open(path, "w") as f:
                json.dump(obj, f)
    with open(os.path.join(root, "frames.json"), "w") as f:
        json.dump(tags, f, indent=1)


def save_outputs(out, geo, names, files, model_id):
    import open3d as o3d

    os.makedirs(out, exist_ok=True)
    frames = []
    for i, name in enumerate(names):
        hh, ww = geo["images"][i].shape[:2]
        frames.append({"name": name, "file": os.path.abspath(files[i]), "width": ww, "height": hh,
                       "K": geo["K_full"][i].tolist(), "c2w": geo["c2w"][i].tolist()})
    cams = {
        "world": {"units": "m", "up": [0, 0, 1], "floor_z": 0.0,
                  "camera_convention": "OpenCV, camera-to-world"},
        "model": model_id, "is_metric": geo["is_metric"],
        "metric_scale_factor": geo["metric_scale_factor"],
        "gravity": geo["gravity"], "frames": frames,
    }
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(cams, f, indent=1)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(geo["points"].astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(geo["colors"].astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(os.path.join(out, "points.ply"), pcd.voxel_down_sample(0.01))

    np.savez_compressed(os.path.join(out, "geometry.npz"), names=np.array(names),
                        depth=geo["depth"], K=geo["K"], c2w=geo["c2w"], K_full=geo["K_full"])
    with open(os.path.join(out, "consistency.json"), "w") as f:
        json.dump(depth_consistency(geo["depth"], geo["K"], geo["c2w"], geo["valid"]), f, indent=1)
    write_boxer_input(os.path.join(out, "boxer_input", BOXER_SEQ), geo, names)


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 1: cameras + metric, gravity-aligned points.")
    p.add_argument("--images", required=True, help="folder of photos (jpg/png)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DA3_MODEL, help="DA3 Hugging Face id or local folder")
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--conf-percentile", type=float, default=40.0,
                   help="drop this percentage of least confident pixels")
    p.add_argument("--no-ray-pose", action="store_true", help="use DA3's faster camera head")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    files = list_images(args.images)
    if not files:
        raise SystemExit(f"no images in {args.images}")
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    images = [load_rgb(f) for f in files]
    print(f"Phase 1: {len(images)} photos -> {args.model}")
    pred = run_da3(images, args.model, args.process_res, not args.no_ray_pose, args.device)
    geo = process_prediction(pred, images, args.conf_percentile)
    save_outputs(args.out, geo, names, files, args.model)

    g = geo["gravity"]
    with open(os.path.join(args.out, "consistency.json")) as f:
        cons = json.load(f)
    print(f"  metric output: {geo['is_metric']} (scale factor {geo['metric_scale_factor']})")
    print(f"  floor found: {g['floor_found']} (inliers {g['floor_inlier_fraction']}), "
          f"tilt vs upright-camera hint {g['camera_up_hint_angle_deg']:.1f} deg")
    print(f"  multi-view depth consistency (<{cons['rel_tol']:.0%}): "
          f"median inlier ratio {cons['median_inlier_ratio']} over {cons['num_pairs']} pairs")
    print(f"  wrote {args.out}/cameras.json, points.ply, boxer_input/{BOXER_SEQ}")


if __name__ == "__main__":
    main()
