#!/usr/bin/env bash
# Supervisor for the VQ2 overnight grind: launch vq2_pose_train.py and AUTO-RESUME from
# weights/last.pt on any in-session crash, until the trainer writes the VQ2_DONE sentinel.
#
# A whole-machine ShadowPC power-off kills THIS loop too -- the agent re-launches it on wake;
# it auto-detects weights/last.pt and resumes, so re-launch is always safe and idempotent.
#
# Usage:  vq2_supervise.sh <run_dir> <logfile> -- <vq2_pose_train.py args...>
#   run_dir = <project>/<name>  (where weights/last.pt + VQ2_DONE live)
set -u
RUN_DIR="$1"; LOG="$2"; shift 2
[ "${1:-}" = "--" ] && shift
PY="C:/Users/Shadow/vq2yolo-venv/Scripts/python.exe"
cd /c/Users/Shadow/Peregrine || exit 9
mkdir -p "$(dirname "$LOG")"
for i in $(seq 1 80); do
  if [ -f "$RUN_DIR/VQ2_DONE" ]; then echo "SUPERVISOR_DONE (already) before launch #$i"; exit 0; fi
  if [ -f "$RUN_DIR/weights/last.pt" ]; then RES="--resume"; else RES=""; fi
  echo "=== SUPERVISOR launch #$i $(date '+%F %T') resume=${RES:-fresh} ===" >> "$LOG"
  "$PY" cluster/vq2_pose_train.py "$@" $RES >> "$LOG" 2>&1
  RC=$?
  echo "=== SUPERVISOR trainer exited rc=$RC $(date '+%F %T') ===" >> "$LOG"
  if [ -f "$RUN_DIR/VQ2_DONE" ]; then echo "SUPERVISOR_DONE rc=$RC"; exit 0; fi
  echo "=== SUPERVISOR will resume in 5s (relaunch $((i+1))) ===" >> "$LOG"
  sleep 5
done
echo "SUPERVISOR_GIVEUP after 80 relaunches" >> "$LOG"
exit 1
