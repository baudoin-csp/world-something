"""Phase 1 post-processing, the Phase 1 -> Boxer hand-off and the evaluation, on a synthetic room.

The networks (DA3, OWLv2, BoxerNet) are replaced by exact synthetic outputs, so these tests check
everything this repo adds around them. They need the third-party code (./setup.sh) but no weights.
"""

import json
import os

import numpy as np
import pytest

from tests import synthetic
from world_something import BOXER_DIR
from world_something import phase1_geometry as p1

needs_boxer = pytest.mark.skipif(not os.path.isdir(BOXER_DIR), reason="run ./setup.sh first")


@pytest.fixture(scope="module")
def phase1(tmp_path_factory):
    pred, photos, c2w_gt = synthetic.fake_da3_prediction()
    geo = p1.process_prediction(pred, photos)
    out = tmp_path_factory.mktemp("phase1")
    tags = [str((i + 1) * 10**9) for i in range(len(photos))]
    files = [str(out / f"{t}.png") for t in tags]
    p1.save_outputs(str(out), geo, tags, files, "synthetic")
    return geo, c2w_gt, out, tags


def test_gravity_and_floor(phase1):
    geo, c2w_gt, _, _ = phase1
    assert geo["gravity"]["floor_found"]
    down = np.array([0.0, 0.0, -1.0])
    for pred, gt in zip(geo["c2w"], c2w_gt):
        # gravity seen from each camera matches, and the floor is at z = 0
        cos = (pred[:3, :3].T @ down) @ (gt[:3, :3].T @ down)
        assert np.degrees(np.arccos(min(1.0, cos))) < 0.5
        assert abs(pred[2, 3] - gt[2, 3]) < 0.02
    # relative poses are preserved (rigid transform only)
    rel_pred = np.linalg.inv(geo["c2w"][0]) @ geo["c2w"][3]
    rel_gt = np.linalg.inv(c2w_gt[0]) @ c2w_gt[3]
    assert np.allclose(rel_pred, rel_gt, atol=1e-6)


def test_intrinsics_mapped_to_full_resolution(phase1):
    geo = phase1[0]
    assert np.allclose(geo["K_full"][0], synthetic.intrinsics(synthetic.FULL_W, synthetic.FULL_H))
    assert geo["images"][0].shape == (synthetic.FULL_H, synthetic.FULL_W, 3)


def test_outputs_and_consistency(phase1):
    _, _, out, tags = phase1
    cams = json.load(open(out / "cameras.json"))
    assert [f["name"] for f in cams["frames"]] == tags
    cons = json.load(open(out / "consistency.json"))
    assert cons["num_pairs"] > 0 and cons["median_inlier_ratio"] > 0.95
    seq = out / "boxer_input" / p1.BOXER_SEQ
    assert len(list(seq.glob("*.wide"))) == len(tags)
    assert (out / "points.ply").stat().st_size > 0


@needs_boxer
def test_boxer_reads_phase1_output(phase1, monkeypatch):
    """The hand-off: Boxer's own CA-1M loader reads Phase 1's folder and sees our cameras/depth."""
    geo, _, out, _ = phase1
    from world_something import add_boxer_to_path, phase2_boxes

    add_boxer_to_path()
    import utils.demo_utils as demo_utils

    monkeypatch.setattr(demo_utils, "SAMPLE_DATA_PATH", str(out / "boxer_input"))
    phase2_boxes._allow_missing_ground_truth()
    from loaders.ca_loader import CALoader

    loader = CALoader(p1.BOXER_SEQ, start_frame=0, max_frames=99)
    assert len(loader) == len(geo["c2w"])
    datum = next(loader)
    T = datum["T_world_rig0"].matrix3x4.numpy()
    assert np.allclose(T, geo["c2w"][0][:3], atol=1e-4)
    fx, fy, cx, cy = datum["cam0"].params[:4].tolist()
    assert np.allclose([fx, fy, cx, cy], geo["K_full"][0][[0, 1, 0, 1], [0, 1, 2, 2]], atol=1e-3)
    pts = datum["sdp_w"].numpy()
    pts = pts[np.isfinite(pts).all(1)]
    # Boxer's depth points lie inside the room, with the floor at z = 0
    assert len(pts) > 1000 and pts[:, 2].min() > -0.03 and pts[:, 2].max() < synthetic.ROOM_MAX[2] + 0.03


@needs_boxer
def test_evaluation_end_to_end(phase1, tmp_path):
    """Perfect Phase 1 + Phase 2 outputs must pass every check against the synthetic GT."""
    import torch
    from world_something import add_boxer_to_path, evaluate, phase2_boxes

    geo, c2w_gt, out, tags = phase1
    capture = synthetic.write_fake_ca1m(str(tmp_path / "capture" / "123"), tags)

    # Phase 2 output: the GT table and chair, expressed in Phase 1's world (cup and floor missed).
    add_boxer_to_path()
    from utils.file_io import ObbCsvWriter2
    from utils.tw.obb import ObbTW
    from utils.tw.pose import PoseTW
    from utils.tw.tensor_utils import pad_string, string2tensor

    T_p1_gt = geo["c2w"][0] @ np.linalg.inv(c2w_gt[0])
    boxes = synthetic.GT_BOXES[:2]
    T = np.stack([T_p1_gt @ np.block([[synthetic.yaw_matrix(y), np.c_[p]], [0, 0, 0, 1]])
                  for _, p, _, y in boxes])
    size = torch.tensor([s for _, _, s, _ in boxes], dtype=torch.float32)
    obbs = ObbTW.from_lmc(
        bb3_object=torch.stack([-size[:, 0], size[:, 0], -size[:, 1], size[:, 1],
                                -size[:, 2], size[:, 2]], 1) / 2,
        T_world_object=PoseTW.from_Rt(torch.tensor(T[:, :3, :3]).float(), torch.tensor(T[:, :3, 3]).float()),
        sem_id=torch.arange(2).reshape(-1, 1), inst_id=torch.arange(2).reshape(-1, 1),
        prob=torch.full((2, 1), 0.9),
        text=torch.stack([string2tensor(pad_string(c, max_len=128)) for c, *_ in boxes]),
    )
    fused = tmp_path / "phase2" / p1.BOXER_SEQ / "boxer_3dbbs_fused.csv"
    fused.parent.mkdir(parents=True)
    writer = ObbCsvWriter2(str(fused))
    writer.write(obbs, timestamps_ns=0)
    writer.close()

    as_json = phase2_boxes.boxes_to_json(str(fused))
    assert [b["label"] for b in as_json] == ["table", "chair"]
    assert np.allclose(as_json[0]["size"], [1.2, 0.8, 0.75], atol=1e-3)

    evaluate.main(["--capture", str(tmp_path / "capture"), "--phase1", str(out),
                   "--phase2", str(tmp_path / "phase2")])
    res = json.load(open(tmp_path / "evaluation.json"))
    p1m, p2m = res["phase1"], res["phase2"]
    assert abs(p1m["sim3_scale"] - 1) < 1e-3 and p1m["ate_sim3_m"] < 1e-3
    assert p1m["gravity_error_deg"] < 0.5 and p1m["camera_height_error_m"] < 0.02
    assert p1m["depth_absrel"] < 0.01 and p1m["focal_error_median"] < 1e-6
    assert p2m["gt_objects_visible"] == 3 and p2m["gt_main_objects"] == 2  # floor excluded, cup small
    assert p2m["main_object_recall"] == 1.0 and p2m["matched_size_error"] < 0.01
    matched = {o["gt"]: o for o in p2m["objects"] if "pred" in o}
    assert set(matched) == {"table", "chair"} and min(o["iou"] for o in matched.values()) > 0.9
    assert all(v["pass"] for v in res["checks"].values())
