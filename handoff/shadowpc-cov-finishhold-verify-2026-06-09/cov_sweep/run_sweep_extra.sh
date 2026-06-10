#!/usr/bin/env bash
set -u
PY=.venv/Scripts/python.exe
BUN=handoff/perception-char-2026-06-08/pg
OUT=handoff/shadowpc-cov-finishhold-verify-2026-06-08/cov_sweep
W=models/gate_yolo11s_curriculum_v2.pt
M=handoff/shadowpc-firstcontact-2026-06-02/track_map.json
for K in 1.5 2.5 3.0; do
  LOG="$OUT/sweep_k${K}_stdout.txt"
  : > "$LOG"
  for G in 0 1 2 3 4 5; do
    echo "######## gate $G  K=$K ########" | tee -a "$LOG"
    "$PY" scripts/characterize_perception.py --bundle "$BUN/course_g$G" \
      --weights "$W" --map "$M" --cov-inflation "$K" \
      --json "$OUT/char_g${G}_k${K}.json" 2>&1 | tee -a "$LOG" | grep -A3 "GATE TRADE-OFF"
  done
done
echo "EXTRA SWEEP DONE"
