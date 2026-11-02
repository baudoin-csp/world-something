# world-something

Phases 1 and 2 of the Mundo rebuild plan: from a few photos of a room to
**(1)** cameras and a metric, gravity-aligned point cloud, and **(2)** labelled, gravity-aligned
7-DoF 3D boxes of the objects in it.

Almost everything is reused. This repo is the glue: about 430 lines of Python for the two
phases, and 340 more to download the example and check the results.

| Phase | Reused (pinned in `setup.sh`) | Added here |
|---|---|---|
| 1 Geometry | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) nested model (`DA3NESTED-GIANT-LARGE-1.1`): intrinsics, poses and depth, already in metres. Open3D RANSAC plane fit. | Floor plane → z axis up, floor at z = 0. Writes cameras, point cloud and Boxer's input. |
| 2 Boxes | [Boxer](https://github.com/facebookresearch/boxer), run unchanged: OWLv2 open-vocabulary 2D detector → BoxerNet 2D-to-3D lifting → multi-view fusion | Points Boxer at Phase 1's folder; exports `boxes.json` |
| Checks | evo's Umeyama alignment (as in DA3's benchmark), Boxer's 3D IoU | `evaluate.py`: compares both phases with CA-1M ground truth |

The plan left two questions open: metric scale and the gravity axis. Here, DA3's nested model
provides the metric scale (it scales its multi-view geometry with a metric depth model), and the
floor plane provides gravity. If no floor is visible, the code falls back to assuming the photos
were taken upright.

## Example 1: CA-1M `val-42898570`

This is an iPad capture from Apple's [CA-1M](https://github.com/apple/ml-cubifyanything) dataset
and the scene Boxer uses for its own demo. It comes with ground truth for every frame: camera
pose, laser-scanner depth, and gravity-aligned 3D boxes for every object. The pipeline's input
is just 12 photos (frames picked evenly from the video). The ground truth is only read by
`evaluate.py`.

## Setup (once, on a machine with internet)

```bash
conda create -n world python=3.11 -y && conda activate world   # any Python 3.10-3.12 env
./setup.sh
```

`setup.sh` does four things:

- clones DA3 and Boxer at pinned commits into `third_party/`
- installs the packages
- downloads the weights: DA3 into the Hugging Face cache, Boxer/DINOv3/OWLv2 into `third_party/boxer/ckpts/`
- downloads the CA-1M capture into `data/`

The CA-1M download asks you to accept the dataset licence; `ACCEPT_CA1M_LICENSE=1 ./setup.sh`
accepts it non-interactively. Open3D needs the system's EGL/GL libraries; on a minimal
Ubuntu image, run `apt install libegl1 libgl1` first. If compute nodes have no internet, point `HF_HOME` at a shared
folder for both setup and runs. To install the packages yourself, use `SKIP_PIP=1` (see
`requirements.txt`).

## Run (one GPU)

```bash
./run_example.sh                  # 12 photos; NUM_FRAMES=6 ./run_example.sh for fewer
```

It runs `select` (pick the frames) → `phase1_geometry` → `phase2_boxes` → `evaluate`. The
results go to `outputs/ca1m-val-42898570-12f/`. Your own photos go through the same two
commands:

```bash
python -m world_something.phase1_geometry --images path/to/photos --out outputs/room/phase1
python -m world_something.phase2_boxes --phase1 outputs/room/phase1 --out outputs/room/phase2
```

## Outputs

World frame: metres, z up, floor at z = 0. Cameras use the OpenCV convention and their poses are
camera-to-world.

- `phase1/cameras.json`: per photo, its intrinsics K and camera-to-world pose, at the photo's resolution
- `phase1/points.ply`: fused coloured point cloud (1 cm voxels)
- `phase1/consistency.json`: multi-view depth consistency (each view's points projected into the others)
- `phase1/boxer_input/ca1m-phase1/`: the same data in the CA-1M layout, which Boxer reads as-is
- `phase2/boxes.json`: fused boxes, each with `label`, `score`, `center`, `size` (x, y, z; z is height) and `yaw`
- `phase2/ca1m-phase1/`: Boxer's own outputs: per-photo and fused CSVs, and images with the boxes drawn on each photo
- `evaluation.json`: every metric below, plus a per-object table

## Checks (the plan's "done when")

| Plan criterion | Metric | Target |
|---|---|---|
| Reprojections line up | median multi-view depth-consistency inlier ratio (5 % tolerance) | ≥ 0.70 |
| A known object measures right | Sim(3) scale vs GT poses (1 = exact metres) | within 10 % |
| | depth vs laser depth, no rescaling (AbsRel) | ≤ 0.15 |
| | median size error of matched objects (footprint and height) | ≤ 20 % |
| Gravity-aligned, floor at 0 | angle between predicted and GT "down", seen from each camera | ≤ 3° |
| | camera height above the floor vs GT | ≤ 10 cm |
| Main objects are found | recall of GT objects with longest side ≥ 0.5 m, at 3D IoU ≥ 0.25 | ≥ 70 % |

Phase 2 boxes are moved into the GT world by a rigid alignment with no scale correction, so any
scale error in Phase 1 counts against Phase 2. The targets are a first guess; change them in
`TARGETS` in `world_something/evaluate.py`.

## Tests

```bash
python -m pytest tests
```

The tests use a synthetic room with exact cameras, depth and boxes in place of the networks. They
check the gravity alignment, the intrinsics mapping, Boxer's loader reading Phase 1's output, and
the evaluation. They need `third_party/` from `setup.sh` but no GPU or weights.

## Licences

All three are non-commercial:

- DA3 nested weights: CC BY-NC 4.0
- Boxer code and weights: CC BY-NC 4.0
- CA-1M data: CC BY-NC-ND 4.0
