#!/usr/bin/env bash
# Example 1 end to end: CA-1M val-42898570 -> photos -> Phase 1 -> Phase 2 -> checks against GT.
#   NUM_FRAMES=12 ./run_example.sh      (Mundo takes 1-12 photos; any number works)
set -euo pipefail
cd "$(dirname "$0")"

NUM_FRAMES=${NUM_FRAMES:-12}
CAPTURE=data/ca1m-val-42898570
OUT=outputs/ca1m-val-42898570-${NUM_FRAMES}f

python -m world_something.ca1m fetch    # no-op once downloaded
python -m world_something.ca1m select --capture "$CAPTURE" --out "$OUT/input" --num-frames "$NUM_FRAMES"
python -m world_something.phase1_geometry --images "$OUT/input/images" --out "$OUT/phase1"
python -m world_something.phase2_boxes --phase1 "$OUT/phase1" --out "$OUT/phase2"
python -m world_something.evaluate --capture "$CAPTURE" --phase1 "$OUT/phase1" --phase2 "$OUT/phase2"
