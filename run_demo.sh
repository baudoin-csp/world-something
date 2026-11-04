#!/usr/bin/env bash
# The room + physics demo, end to end, on CA-1M val-42898570:
#   photos -> Phases 1-2 (DA3 + Boxer on a GPU; ORACLE=1 uses CA-1M's ground truth instead, no GPU)
#   -> 3 proxy world -> 4 assets + robots -> 5 physics -> 6 proxy video -> 7 rendered video
#
#   ORACLE=1 RENDER=ibr ./run_demo.sh          # runs on a laptop CPU (MUJOCO_GL=osmesa if no GPU)
#   RENDER=h3 LORA=outputs/phase8/lora ADAPTER=outputs/phase8/adapter.pt ./run_demo.sh
#   RENDER=sglang SERVER=http://gpu-node:30011 ./run_demo.sh
#   AGENT=1 ...        refine the proxy world with the Claude agent (needs ANTHROPIC_API_KEY)
#   PROMPT="..." ...   let the Claude director write the physics program instead of the demo one
#   ROBOTS="--robot stretch3@auto --robot panda@auto"   (default) robots added in Phase 4
set -euo pipefail
cd "$(dirname "$0")"

NUM_FRAMES=${NUM_FRAMES:-12}
CAPTURE=data/ca1m-val-42898570
OUT=${OUT:-outputs/demo-ca1m-${NUM_FRAMES}f}
RENDER=${RENDER:-ibr}
ROBOTS=${ROBOTS:---robot stretch3@auto --robot panda@auto}
export MUJOCO_GL=${MUJOCO_GL:-egl}

[ -d "$CAPTURE" ] || python -m world_something.ca1m fetch
python -m world_something.ca1m select --capture "$CAPTURE" --out "$OUT" --num-frames "$NUM_FRAMES"
if [ "${ORACLE:-0}" = 1 ]; then
  python -m world_something.ca1m oracle --capture "$CAPTURE" --frames "$OUT/frames.json" --out "$OUT"
else
  python -m world_something.phase1_geometry --images "$OUT/images" --out "$OUT/phase1"
  python -m world_something.phase2_boxes --phase1 "$OUT/phase1" --out "$OUT/phase2"
fi

AGENT_ARGS=()
[ "${AGENT:-0}" = 1 ] && AGENT_ARGS=(--agent)
python -m world_something.phase3_proxy --phase1 "$OUT/phase1" --phase2 "$OUT/phase2" --out "$OUT/phase3" "${AGENT_ARGS[@]}"
# shellcheck disable=SC2086
python -m world_something.phase4_assets --scene "$OUT/phase3/scene.json" --out "$OUT/phase4" --rough $ROBOTS
if [ -n "${PROMPT:-}" ]; then
  python -m world_something.phase5_simulate --scene "$OUT/phase4/scene.json" --out "$OUT/phase5" --prompt "$PROMPT"
else
  python -m world_something.phase5_simulate --scene "$OUT/phase4/scene.json" --out "$OUT/phase5" --demo
fi
python -m world_something.phase6_proxy_video --sim "$OUT/phase5" --out "$OUT/phase6"

RENDER_ARGS=(--backend "$RENDER")
[ -n "${LORA:-}" ] && RENDER_ARGS+=(--lora "$LORA")
[ -n "${ADAPTER:-}" ] && RENDER_ARGS+=(--adapter "$ADAPTER")
[ -n "${SERVER:-}" ] && RENDER_ARGS+=(--server "$SERVER")
python -m world_something.phase7_render --phase6 "$OUT/phase6" --out "$OUT/phase7" "${RENDER_ARGS[@]}"
echo "Demo done: $OUT/phase6/proxy.mp4 (proxy) and $OUT/phase7/render.mp4, side_by_side.mp4"
