#!/bin/bash
# inc8 detstab live monitor: queue state + TB flightcheck (stochastic) + snapshot count per job.
# Usage: bash monitor.sh <tag1> <tag2> ...   (tag == the RUNTAG, e.g. ds_gp1p0_h06)
echo "=== QUEUE ==="
squeue -u fl3689 -o '%.12i %.26j %.8T %.7M %R'
for t in "$@"; do
  echo "=== $t ==="
  RD=/scratch/network/fl3689/diffaero/outputs/train/inc8_detstab_seed0_$t
  D=$(ls -dt $RD/*__0 2>/dev/null | head -1)
  if [ -n "$D" ]; then
    python /scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py "$D" --stride-updates 250 2>/dev/null | tail -10
  else
    echo "  (no decorated run dir yet)"
  fi
  echo "  snapshots: $(ls -d $RD/snapshots/upd* 2>/dev/null | wc -l)  latest: $(ls -d $RD/snapshots/upd* 2>/dev/null | tail -1)"
done
echo "=== $(date) ==="
