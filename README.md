# world-something

The Mundo rebuild plan: from a few photos of a room to a simulated, interactive world and a
rendered video of it.

| Phase | What it produces | Status |
|---|---|---|
| 1 Geometry | cameras and a metric, gravity-aligned point cloud | done |
| 2 Boxes | labelled, gravity-aligned 7-DoF 3D boxes | done |
| 3 Proxy world | a scene program of boxes + physics that matches the photos, refined by a Claude agent | done |
| 4 Assets | robots (MuJoCo Menagerie) and rough articulated models in place of boxes | done |
| 5 Code world | MuJoCo simulation of scripted interactions at 960 Hz | done |
| 6 Proxy video | flat-colour proxy frames along a camera path, chunked for the renderer | done |
| 7 Renderer | MiniMax-H3 Ref2VA conditioned on proxy video + context photos | in progress |
| 8 Training | proxy/real training pairs and the LoRA fine-tune | to do |

Almost everything is reused; this repo is the glue. Phases 1-2 below come first, then the
later phases.

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

## Phase 3: proxy world

```bash
python -m world_something.phase3_proxy --phase1 outputs/room/phase1 --phase2 outputs/room/phase2 \
    --out outputs/room/phase3                       # deterministic proxy only
ANTHROPIC_API_KEY=... python -m world_something.phase3_proxy ... --agent   # + agent refinement
```

The proxy world (`phase3/scene.json`) is the scene program every later phase reads: each object
is a gravity-aligned box (`center`, `size`, `yaw`) with a palette class and physics (`movable`,
`mass`, `friction`, `restitution`), plus the cameras of the photos.

- **Initial proxy** (no network calls): Phase 2's boxes, a floor, walls and a ceiling estimated
  from Phase 1's points (a wall only where points support it), objects snapped onto whatever
  is under them, physics defaults per category (unsupported objects are marked as mounted).
- **Agent** (`--agent`, Claude Opus 5.5): analysis by synthesis, as in Mundo. Each round a
  read-only *reviewer* looks at every photo with the proxy overlaid and a depth-check image
  (red = real surface missing from the proxy, blue = box where the photo sees through), then
  accepts or lists findings. An *editor* fixes them with scene tools. The VLM never types
  coordinates for a fit. It marks rectangles on photos, and `fit.py` places the box by least
  squares; with three or more marks, one photo is held out and has to agree. The loop stops when the reviewer accepts,
  depth agreement stops improving (and is reverted if edits break it), or the round/edit/time
  budget runs out. Requests use server-side refusal fallbacks (`fallbacks="default"`).
- **Rendering**: `mjscene.py` compiles the scene to MuJoCo and renders object ids, flat proxy
  colours and depth from any OpenCV camera. The same code renders Phase 6's proxy videos.

Outputs: `scene.json`, `initial_scene.json`, `renders/` (proxy, overlay and depth check per
photo), `metrics.json` (depth agreement with Phase 1 per photo, before and after the agent) and
`agent_log/` (every tool call and result, token usage).

**Without a GPU**: `python -m world_something.ca1m oracle --frames <out>/frames.json --out <out>`
writes CA-1M's ground truth (poses, laser depth, boxes) in Phase 1/2's output format, so Phases
3-6 run on any machine. On `val-42898570` the ground truth has no bed. That makes it a useful
agent test: the depth check shows the twin beds in red.

Proxy palette: object `#eeeeee`, new object `#f39442`, floor `#707070`, wall `#a9b6c6`,
ceiling `#bba2dc`, window `#56c8d8`, door `#ad7756`, unknown `#30343b`.

Rendering headless needs `MUJOCO_GL=egl` (GPU nodes) or `MUJOCO_GL=osmesa` (CPU, `apt install
libosmesa6`).

## Phase 4: assets and robots

```bash
python -m world_something.phase4_assets --scene outputs/room/phase3/scene.json --out outputs/room/phase4 \
    --rough --robot stretch3@1.2,0.4,90 --robot panda@on=makeup_vanity,180
```

An object's `asset` field picks its model. Objects without one stay boxes.

| Asset | What it is |
|---|---|
| `menagerie:<robot>` | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) robot, starting from its keyframe: `panda` (Franka arm, fixed base, put it on a table with `on=`), `google_robot`, `g1` (Unitree humanoid), `stretch3` (Hello Robot mobile manipulator). Galaxea R1 Pro, which Mundo also shows, is not in Menagerie. |
| `rough:<kind>` | a procedural model fitted into the box, facing +x: `table`, `chair`, `stool`, `bed`, `sofa`, `shelf`, `cabinet` (hinged door), `drawers` (sliding drawer), `door` (hinged), `lamp`, `ball`, `cylinder`. `--rough` assigns one to every recognised category. |
| `mesh:<path>` | an `.obj`/`.stl` scaled into the box, e.g. from TRELLIS. MuJoCo collides with its convex hull. |

Robots are proxy class *new object* (orange): they are not in the photos. Every geom of an
asset belongs to its object, so the proxy renderer still draws flat palette colours.

TRELLIS image-to-3D (optional, GPU): `--trellis-crops` saves a crop of every object from the
photo that shows most of it. Then, in an environment with
[TRELLIS](https://github.com/microsoft/TRELLIS) installed, run
`python -m world_something.phase4_assets trellis --crops <out>/crops --out <out>/meshes` and
pass the results back with `--mesh ID=<out>/meshes/ID.obj`.

Outputs: `scene.json` (Phase 3's scene with `asset` fields and robots) and `previews/`
(proxy colours next to a shaded MuJoCo render, from four of the photo cameras).

## Phase 5: code world

```bash
python -m world_something.phase5_simulate --scene outputs/room/phase4/scene.json --out outputs/room/phase5 --demo
python -m world_something.phase5_simulate --scene ... --out ... --program events.json
ANTHROPIC_API_KEY=... python -m world_something.phase5_simulate --scene ... --out ... \
    --prompt "the robot drives past the stool and a ball bounces across the room" --review 2
```

The scene is compiled to MuJoCo, stepped at 960 Hz and recorded at 24 fps, H3's output rate.
What happens is an *event program*: a JSON list of timed events (`codeworld.py`).

| Event | Fields |
|---|---|
| `spawn` | `label`, `shape` (box/sphere/cylinder), `size`, `position`, `velocity`, `mass`, `restitution`: a new (orange) object released at `at` |
| `push` | `object`, `direction`, `force` (N), `duration` |
| `throw` / `drop` / `place` | `object` + `velocity` / `height` / `position`, `yaw_deg` |
| `open` | `object` (rough cabinet, drawers or door), `amount` 0-1, `duration` |
| `robot_pose` | `robot`, `targets` {actuator or joint name: value}, `duration` |
| `robot_reach` | `robot`, `target` [x, y, z] for the end effector (damped least-squares IK) |
| `robot_drive` | `robot` (free base: `stretch3`, `g1`), `distance`, `turn_deg`, `duration` |

Every event also has `at` (seconds). Example:
`{"duration": 6, "events": [{"type": "push", "object": "stool", "direction": [-1, 0, 0], "force": 40, "at": 1.0, "duration": 0.5}]}`.

- **Settling**: the scene first rests for 1 s. Objects that move are frozen in place and
  reported (`sim.json`); they were unsupported or overlapping another box. Rest poses are written
  back to `scene_sim.json`.
- **Director** (`--prompt`, Claude Opus 5.5): writes the program as structured JSON from the
  scene's objects and robots. With `--review N` it watches proxy keyframes of the simulated
  result and revises the program up to N times. Invalid programs are sent back with the error.
- Mobile robots follow their path kinematically, with the matching velocity so contacts still
  push things. Stretch's wheel servos are too weak in this model to drive straight, and G1 has
  no walking policy here: it is carried while its joints hold their pose. Arms (Panda, Google
  Robot, G1, Stretch's lift/arm) are driven by their own position servos.

Outputs: `program.json`, `scene_sim.json` (spawned objects and settled poses, so Phase 6 rebuilds
the same model), `trajectory.npz` (qpos per frame and every object's pose), `sim.json` (settle
report, event log, timings), `keyframes/`. On CA-1M with two robots, 6 s simulate in about 2 s on
one CPU core.

## Phase 6: proxy video

```bash
python -m world_something.phase6_proxy_video --sim outputs/room/phase5 --out outputs/room/phase6 \
    [--path auto | static:I | dolly:I:D | photos:I,J,K | orbit:R:H | file:cams.json] [--save-ids]
```

The simulation is rendered along a camera path in the proxy palette, 24 fps, 640 px on the long
side. Then it is cut the way the renderer consumes it, following Mundo:

- **Chunks** of 124 frames (17·7 + 5, i.e. 37 latent frames for H3's 4× temporal VAE). Each
  chunk after the first starts with the previous chunk's last 34 frames. Phase 5 runs the
  simulation on to the end of the last chunk, so nothing has to be padded.
- **Context photos**: up to 9 per chunk (Ref2VA's image limit). Photos are chosen greedily by how
  much of the chunk's visible proxy surface they show, with occlusion. The first one is the
  chunk's reference view.
- **Cameras** are stored in the world frame and relative to the reference view
  (`T_ref⁻¹ · T`), Mundo's alignment for the camera encoder.

`--path auto` orbits 50° around the moving objects, at a radius and start angle that keep the
camera inside the room and clear of every box. If no such orbit exists, it falls back to the photo
camera that sees the action best.

Outputs: `proxy.mp4`, `manifest.json`, and per chunk `chunk_XX/{proxy.mp4, cameras.json,
context.json, context/}`; with `--save-ids`, also `ids.npz` (object index per pixel, for training masks).

## Tests

```bash
python -m pytest tests
```

The tests use a synthetic room with exact cameras, depth and boxes in place of the networks. They
check the gravity alignment, the intrinsics mapping, Boxer's loader reading Phase 1's output, and
the evaluation. They need `third_party/` from `setup.sh` but no GPU or weights.

## Licences

Phases 1-2 inputs are non-commercial:

- DA3 nested weights: CC BY-NC 4.0
- Boxer code and weights: CC BY-NC 4.0
- CA-1M data: CC BY-NC-ND 4.0

Later phases: MuJoCo and MuJoCo Menagerie's robot models are Apache-2.0 (a few models carry their
own licence; see each folder's LICENSE). TRELLIS is MIT. The agent calls the Claude API under your
Anthropic account's terms.
