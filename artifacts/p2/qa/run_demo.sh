#!/usr/bin/env bash
# Reproduce the QA v2 acceptance demo from the repo root.
#   artifacts/p2/qa/run_demo.sh
# Bad overlay (painted over the face) -> verify2 FAIL -> repair.py -> re-encode
# -> verify2 PASS. Writes everything under artifacts/p2/qa/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
PY=".venv/bin/python"
QA="artifacts/p2/qa"

echo "== 1. base clip + demo masks + bad IR"
$PY "$QA/scripts/build_demo.py" prep

echo "== 2. paint the deliberately bad overlay (box over the face)"
$PY "$QA/scripts/build_demo.py" paint \
  --ir "$QA/motion_ir_bad.json" --out "$QA/final_bad.mp4" \
  --overlay-out "$QA/overlay_bad.mp4" --ink-out "$QA/ink_bad.mp4"

echo "== 3. verify2 on the bad render (expected FAIL: face/person overlap)"
if $PY -m semotion.verify2 \
    --ir "$QA/motion_ir_bad.json" --source "$QA/base_clip.mp4" \
    --final "$QA/final_bad.mp4" --overlay "$QA/overlay_bad.mp4" \
    --ink-only "$QA/ink_bad.mp4" --perception "$QA/perception_demo.json" \
    --out "$QA/qa_bad/qa_metrics.json"; then
  echo "ERROR: bad demo unexpectedly PASSed" >&2
  exit 1
else
  echo "expected FAIL recorded"
fi

echo "== 4. deterministic repair (relocate to a zero-person-overlap zone)"
$PY -m semotion.repair \
  --ir "$QA/motion_ir_bad.json" --metrics "$QA/qa_bad/qa_metrics.json" \
  --masks "$QA/perception_demo.json" --out-dir "$QA/repair"

echo "== 5. re-paint from the repaired IR and encode twice (determinism)"
$PY "$QA/scripts/build_demo.py" paint \
  --ir "$QA/repair/motion_ir.repaired.json" --out "$QA/final_fixed.mp4" \
  --overlay-out "$QA/overlay_fixed.mp4" --ink-out "$QA/ink_fixed.mp4" --repeats 2

echo "== 6. verify2 on the repaired render (expected PASS)"
$PY -m semotion.verify2 \
  --ir "$QA/repair/motion_ir.repaired.json" --source "$QA/base_clip.mp4" \
  --final "$QA/final_fixed.mp4" --overlay "$QA/overlay_fixed.mp4" \
  --ink-only "$QA/ink_fixed.mp4" --perception "$QA/perception_demo.json" \
  --second-render "$QA/final_fixed_run2.mp4" \
  --out "$QA/qa_fixed/qa_metrics.json"

echo "== 7. idempotence: second repair on the repaired IR + stale metrics = no-op"
$PY -m semotion.repair \
  --ir "$QA/repair/motion_ir.repaired.json" --metrics "$QA/qa_bad/qa_metrics.json" \
  --masks "$QA/perception_demo.json" --out-dir "$QA/repair/idempotence" \
  --out-ir "$QA/repair/idempotence/motion_ir.repaired.json"

echo "done: $QA/qa_bad/qa_metrics.json (FAIL), $QA/qa_fixed/qa_metrics.json (PASS)"
