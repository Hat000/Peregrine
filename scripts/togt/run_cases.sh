#!/usr/bin/env bash
# Run all TOGT bound cases: C++ TOGT plan (init+refine phases) then the CasADi
# multiple-shooting time-optimal refine, per case dir produced by gen_cases.py.
#
# Usage (inside WSL/Linux):
#   scripts/togt/run_cases.sh <cases_dir> [case ...]
# Env overrides:
#   TOGT_ROOT  TOGT-Planner clone with build/planners      (default ~/peregrine_togt/TOGT-Planner)
#   VENVPY     python with casadi                          (default ~/peregrine_togt/venv/bin/python)
#   REFINE_DIR scripts/togt/refine in this repo            (default: derived from this script)
#   DT         multiple-shooting node spacing              (default 0.02)
#   JOBS       concurrent refines                          (default 4)
set -uo pipefail

CASES_DIR=$(readlink -f "${1:?usage: run_cases.sh <cases_dir> [case ...]}")
shift || true
TOGT_ROOT=${TOGT_ROOT:-$HOME/peregrine_togt/TOGT-Planner}
VENVPY=${VENVPY:-$HOME/peregrine_togt/venv/bin/python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REFINE_DIR=${REFINE_DIR:-$SCRIPT_DIR/refine}
DT=${DT:-0.02}
JOBS=${JOBS:-4}
PIECES=1   # saveSegments granularity: planTOGT uses 0 corridor midpoints -> 1 = gate junctions

if [ $# -gt 0 ]; then CASES="$*"; else CASES=$(ls "$CASES_DIR"); fi

# The TOGT phase runs as a gtest case inside the tests binary; the standalone `planners`
# CLI segfaults before main() in this environment (see scripts/togt/test_peregrine.cpp).
# RaceParams also requires ABSOLUTE paths (relative ones fail to load -> garbage params).
#
# The init L-BFGS is brittle off the nominal parameter point, so each case walks a retry
# ladder: speedGuess {1,3,5,8,2}, then the same with dynamicConstCheck=false. The C++
# phase is only the warm start + crossing points -- the constraint GUARANTEE comes from
# the IPOPT refine, which enforces bounds exactly at its nodes -- so relaxing the init
# check is sound. The values that actually ran stay in the case dir (init yaml mutated
# in place) for provenance.
echo "== TOGT phase (tests --gtest_filter=PeregrineTOGT.plan) =="
togt_attempt() {
  TOGT_CONFIG_DIR="$1" TOGT_QUAD=peregrine TOGT_TRACK="$1/track.yaml" \
  TOGT_TRAJ="$1/togt_traj.csv" TOGT_WPT="$1/togt_wpt.yaml" TOGT_PIECES=$PIECES \
      "$TOGT_ROOT/build/tests/tests" --gtest_filter=PeregrineTOGT.plan \
      > "$1/planner_stdout.txt" 2>&1
}
for c in $CASES; do
  d=$CASES_DIR/$c
  [ -f "$d/track.yaml" ] || continue
  rc=1
  for dcc in true false; do
    sed -i "s/dynamicConstCheck:  .*/dynamicConstCheck:  $dcc/" "$d/init/peregrine_planning.yaml"
    for sg in 1.0 3.0 5.0 8.0 2.0; do
      sed -i "s/speedGuess:         .*/speedGuess:         $sg/" "$d/init/peregrine_planning.yaml"
      togt_attempt "$d" && { rc=0; break; }
    done
    [ $rc -eq 0 ] && break
  done
  dur=$(grep -m1 "Duration:" "$d/planner_stdout.txt" | awk '{print $2}')
  sg=$(grep -m1 "speedGuess" "$d/init/peregrine_planning.yaml" | awk '{print $2}')
  dcc=$(grep -m1 "dynamicConstCheck" "$d/init/peregrine_planning.yaml" | awk '{print $2}')
  echo "  $c: rc=$rc togt_duration=${dur:-?}s (speedGuess=$sg dynCC=$dcc)"
done

echo "== refine phase (multiple shooting, dt=$DT, jobs=$JOBS) =="
refine_one() {
  d=$1
  tol=$("$VENVPY" -c "import json;print(json.load(open('$d/meta.json'))['refine_tol_m'])")
  (cd "$REFINE_DIR" && "$VENVPY" refine_peregrine.py \
      --quad-yaml "$d/peregrine_quad.yaml" --traj-csv "$d/togt_traj.csv" \
      --wpt-yaml "$d/togt_wpt.yaml" --out-csv "$d/refined_traj.csv" \
      --tol "$tol" --tol-term 0.05 --dt "$DT" \
      --summary-json "$d/refine_summary.json") > "$d/refine_stdout.txt" 2>&1
  echo "  $(basename "$d"): $(grep -o 'REFINE_SUMMARY.*' "$d/refine_stdout.txt" | head -1)"
}
export -f refine_one 2>/dev/null || true
export VENVPY REFINE_DIR DT

pids=()
n=0
for c in $CASES; do
  d=$CASES_DIR/$c
  [ -f "$d/togt_traj.csv" ] || { echo "  $c: SKIP (no togt_traj)"; continue; }
  refine_one "$d" &
  pids+=($!)
  n=$((n+1))
  if [ $((n % JOBS)) -eq 0 ]; then wait; fi
done
wait
echo "== done =="
