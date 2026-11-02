"""Check Phases 1-2 against CA-1M ground truth, following the plan's "done when" criteria.

Phase 1 - "reprojections line up; a known object measures right"
  * camera poses vs GT after Sim(3) alignment (evo's Umeyama, as in DA3's benchmark):
    ATE and rotation error; the Sim(3) scale is 1.0 when Phase 1 is truly metric
  * depth vs the laser-scanner depth, without any rescaling (AbsRel, delta < 1.25)
  * gravity: angle between predicted and GT "down" seen from each camera (no alignment needed)
  * floor: camera height above the floor, predicted vs GT
  * multi-view depth consistency computed by Phase 1
Phase 2 - "the main objects of the room are found"
  * Phase 2 boxes moved into the GT world with the rigid (unscaled) alignment, so scale errors
    count; 3D IoU with Boxer's iou_mc9 and Hungarian matching against GT boxes visible in the
    photos; recall of main objects (longest side >= 0.5 m) and size error of matched objects
"""

import argparse
import json
import os

import numpy as np

from world_something import add_boxer_to_path
from world_something.ca1m import find_data_dir, load_gt_frame
from world_something.phase1_geometry import BOXER_SEQ

TARGETS = {  # metric: (comparison, value)
    "sim3_scale_error": ("<=", 0.10),
    "depth_absrel": ("<=", 0.15),
    "gravity_error_deg": ("<=", 3.0),
    "camera_height_error_m": ("<=", 0.10),
    "consistency_inlier_ratio": (">=", 0.70),
    "main_object_recall": (">=", 0.70),
    "matched_size_error": ("<=", 0.20),
}
NOT_OBJECTS = {"floor", "wall", "ceiling", "door_frame", "shelter"}


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def align_poses(c2w_gt, c2w_pred, with_scale):
    """Umeyama alignment of camera centres (evo). Returns r, t, s with gt ~ s * r @ pred + t."""
    if len(c2w_gt) < 3:  # too few cameras for Umeyama: align on the first camera
        T = c2w_gt[0] @ np.linalg.inv(c2w_pred[0])
        return T[:3, :3], T[:3, 3], 1.0
    from evo.core.trajectory import PosePath3D

    est = PosePath3D(poses_se3=[p.copy() for p in c2w_pred])
    r, t, s = est.align(PosePath3D(poses_se3=[p.copy() for p in c2w_gt]), correct_scale=with_scale)
    return np.asarray(r), np.asarray(t), float(s)


def evaluate_cameras(gt, K_pred, c2w_pred):
    c2w_gt = np.stack([g["c2w"] for g in gt])
    r, t, s = align_poses(c2w_gt, c2w_pred, with_scale=True)
    pos = (s * (r @ c2w_pred[:, :3, 3].T)).T + t
    ate = float(np.sqrt(np.mean(np.sum((pos - c2w_gt[:, :3, 3]) ** 2, axis=1))))
    rot_err = [_rot_angle_deg(Rg.T @ r @ Rp) for Rg, Rp in zip(c2w_gt[:, :3, :3], c2w_pred[:, :3, :3])]
    down = np.array([0.0, 0.0, -1.0])  # both worlds are z-up
    grav = [_angle_deg(Rp.T @ down, Rg.T @ down) for Rg, Rp in zip(c2w_gt[:, :3, :3], c2w_pred[:, :3, :3])]
    f_pred = (K_pred[:, 0, 0] + K_pred[:, 1, 1]) / 2
    f_gt = np.array([(g["K"][0, 0] + g["K"][1, 1]) / 2 for g in gt])
    rot_med = float(np.median(rot_err))
    if len(gt) < 3:  # alignment is trivial: scale and pose errors are not measured
        s = ate = rot_med = None
    return {
        "num_frames": len(gt),
        "sim3_scale": s, "sim3_scale_error": None if s is None else abs(s - 1.0),
        "ate_sim3_m": ate,
        "rotation_error_deg_median": rot_med,
        "gravity_error_deg": float(np.median(grav)), "gravity_error_deg_max": float(np.max(grav)),
        "focal_error_median": float(np.median(np.abs(f_pred - f_gt) / f_gt)),
    }


def gt_floor_z(gt, max_points=20000):
    """Floor height in the GT world: a low percentile of the laser points seen by the photos."""
    rng = np.random.default_rng(0)
    zs = []
    for g in gt:
        v, u = np.nonzero(g["depth"] > 0)
        if len(u) > max_points:
            keep = rng.choice(len(u), max_points, replace=False)
            u, v = u[keep], v[keep]
        rays = np.linalg.inv(g["K_depth"]) @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)
        zs.append((g["c2w"][:3, :3] @ (rays * g["depth"][v, u]) + g["c2w"][:3, 3:4])[2])
    return float(np.percentile(np.concatenate(zs), 1.0))


def evaluate_depth_and_floor(gt, depth_pred, c2w_pred):
    import cv2

    err, ratio, d1 = [], [], []
    for g, dp in zip(gt, depth_pred):
        dg = g["depth"]
        dp = cv2.resize(dp, (dg.shape[1], dg.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (dg > 0) & (dp > 0)
        err.append(np.abs(dp[m] - dg[m]) / dg[m])
        ratio.append(dg[m] / dp[m])
        d1.append(np.maximum(dp[m] / dg[m], dg[m] / dp[m]) < 1.25)
    floor = gt_floor_z(gt)
    h_gt = np.array([g["c2w"][2, 3] for g in gt]) - floor
    h_pred = c2w_pred[:, 2, 3]  # Phase 1 puts the floor at z = 0
    return {
        "depth_absrel": float(np.mean(np.concatenate(err))),
        "depth_delta1": float(np.mean(np.concatenate(d1))),
        "depth_median_gt_over_pred": float(np.median(np.concatenate(ratio))),
        "camera_height_error_m": float(np.mean(np.abs(h_pred - h_gt))),
        "camera_height_gt_m_mean": float(h_gt.mean()), "camera_height_pred_m_mean": float(h_pred.mean()),
    }


def _vertical_and_footprint(R, size):
    """Box extent along world z and its two horizontal sides (sorted), whatever the box axes."""
    up = np.abs(R[2, :])
    horiz = np.argsort(up)[:2]
    return float(np.sum(up * size)), np.sort(size[horiz])


def evaluate_boxes(data_dir, gt, fused_csv, r, t, iou_thresh=0.25, main_size=0.5):
    add_boxer_to_path()
    import torch
    from scipy.optimize import linear_sum_assignment

    from loaders.ca_loader import _load_world_obbs
    from utils.file_io import read_obb_csv
    from utils.tw.obb import iou_mc9
    from utils.tw.pose import PoseTW

    gt_all, id2inst, _ = _load_world_obbs(data_dir)
    seen = sorted({id2inst[i] for g in gt for i in g["visible_ids"] if i in id2inst})
    gt_obbs = gt_all[torch.tensor(seen, dtype=torch.long)]
    labels = gt_obbs.text_string()
    sizes = gt_obbs.bb3_diagonal.numpy()
    keep = [i for i, (lab, sz) in enumerate(zip(labels, sizes))
            if lab.lower() not in NOT_OBJECTS and sz.max() <= 3.0]  # Boxer's own GT filter
    gt_obbs = gt_obbs[torch.tensor(keep, dtype=torch.long)]
    labels, sizes = [labels[i] for i in keep], sizes[keep]
    is_main = sizes.max(axis=1) >= main_size

    pred = read_obb_csv(fused_csv) if os.path.exists(fused_csv) else {}
    pred = next(iter(pred.values())) if pred else None
    n_pred = 0 if pred is None else len(pred)
    iou = np.zeros((len(keep), n_pred))
    if n_pred and len(keep):
        T = PoseTW.from_Rt(torch.tensor(r, dtype=torch.float32).expand(n_pred, 3, 3),
                           torch.tensor(t, dtype=torch.float32).expand(n_pred, 3))
        pred = pred.transform(T)
        iou = iou_mc9(gt_obbs.float(), pred.float(), samp_per_dim=32, all_pairs=True).numpy()
    rows, cols = linear_sum_assignment(-iou) if iou.size else ([], [])
    match = {int(i): int(j) for i, j in zip(rows, cols) if iou[i, j] >= iou_thresh}

    pred_labels = pred.text_string() if n_pred else []
    table, size_err = [], []
    for i, lab in enumerate(labels):
        row = {"gt": lab, "size": [round(float(x), 2) for x in sizes[i]], "main": bool(is_main[i])}
        if i in match:
            j = match[i]
            Rg = gt_obbs.T_world_object.R[i].numpy()
            Rp = pred.T_world_object.R[j].numpy()
            hg, fg = _vertical_and_footprint(Rg, sizes[i])
            hp, fp = _vertical_and_footprint(Rp, pred.bb3_diagonal[j].numpy())
            e = float(np.mean(np.abs(np.r_[fp, hp] - np.r_[fg, hg]) / np.r_[fg, hg]))
            size_err.append(e)
            row.update(pred=pred_labels[j], iou=round(float(iou[i, j]), 3), size_error=round(e, 3))
        table.append(row)
    n_main = int(is_main.sum())
    found_main = sum(1 for i in match if is_main[i])
    return {
        "iou_threshold": iou_thresh, "main_min_size_m": main_size,
        "gt_objects_visible": len(labels), "gt_main_objects": n_main, "pred_boxes": n_pred,
        "main_object_recall": found_main / n_main if n_main else None,
        "all_object_recall": len(match) / len(labels) if labels else None,
        "precision": len(match) / n_pred if n_pred else None,
        "matched_size_error": float(np.median(size_err)) if size_err else None,
        "objects": table,
    }


def verdicts(metrics):
    out = {}
    for key, (op, target) in TARGETS.items():
        value = metrics.get(key)
        ok = None if value is None else (value <= target if op == "<=" else value >= target)
        out[key] = {"value": value, "target": f"{op} {target}", "pass": ok}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate Phases 1-2 on a CA-1M capture.")
    p.add_argument("--capture", required=True, help="CA-1M capture folder (fetched by ca1m.py)")
    p.add_argument("--phase1", required=True)
    p.add_argument("--phase2", required=True)
    p.add_argument("--out", help="evaluation.json path (default: <phase2>/../evaluation.json)")
    args = p.parse_args(argv)

    data_dir = find_data_dir(args.capture)
    with open(os.path.join(args.phase1, "cameras.json")) as f:
        frames = json.load(f)["frames"]
    tags = [fr["name"] for fr in frames]  # selected frames are named by their CA-1M timestamp
    K_pred = np.array([fr["K"] for fr in frames])
    c2w_pred = np.array([fr["c2w"] for fr in frames])
    depth_pred = np.load(os.path.join(args.phase1, "geometry.npz"))["depth"]
    gt = [load_gt_frame(data_dir, tag) for tag in tags]

    metrics = {"phase1": evaluate_cameras(gt, K_pred, c2w_pred)}
    metrics["phase1"].update(evaluate_depth_and_floor(gt, depth_pred, c2w_pred))
    with open(os.path.join(args.phase1, "consistency.json")) as f:
        metrics["phase1"]["consistency_inlier_ratio"] = json.load(f)["median_inlier_ratio"]

    r, t, _ = align_poses(np.stack([g["c2w"] for g in gt]), c2w_pred, with_scale=False)
    fused = os.path.join(args.phase2, BOXER_SEQ, "boxer_3dbbs_fused.csv")
    metrics["phase2"] = evaluate_boxes(data_dir, gt, fused, r, t)
    metrics["checks"] = verdicts({**metrics["phase1"], **metrics["phase2"]})

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.phase2)), "evaluation.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=1)

    print("\nObjects (GT visible in the photos -> matched Phase 2 box):")
    for o in metrics["phase2"]["objects"]:
        tag = "main" if o["main"] else "    "
        hit = f"{o['pred']:<22} IoU {o['iou']:.2f}  size err {o['size_error']:.0%}" if "pred" in o else "-"
        print(f"  {tag} {o['gt']:<22} {str(o['size']):<20} {hit}")
    print("\nChecks:")
    for key, v in metrics["checks"].items():
        val = "n/a" if v["value"] is None else f"{v['value']:.3f}"
        status = {True: "PASS", False: "FAIL", None: "n/a"}[v["pass"]]
        print(f"  {status:<4}  {key:<26} {val:>8}  (target {v['target']})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
