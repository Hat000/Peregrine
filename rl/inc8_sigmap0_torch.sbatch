#!/bin/bash
#SBATCH --job-name=inc8_sigmap0_torch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/network/fl3689/inc8_sigmap0_torch.out
#
# inc8 sigma_p0 -- MEASURED IN THE TORCH TRAINING ENV (P2 INC8-RL, 2026-06-18). The first FLYING inc8
# policy ("rc1", job 3276449) flies the full 2-axis camera-pointing primitive IN TRAINING but dies on
# the numpy tool's emulated obs (#37 obs-fidelity gap). So measure the faithful 2-axis sigma_p0 where
# rc1 PROVABLY flies: the torch env (rl/inc8_sigmap0_torch_eval.py). This runs, per seed:
#   (1) yaw-only  -- INSTRUMENT CROSS-CHECK: must reproduce the numpy tool's yaw-only sigma_p0_lat
#       (laptop-reproduced 2026-06-18 on rc1 seed0, start=trainreset: 0.1768 m, reach 133/200). If the
#       torch instrument agrees here, its 2-axis number is trustworthy.
#   (2) 2-axis (auto) -- THE DELIVERABLE: full look-at (g_yaw=-3, g_pitch=3), GT-anchored sigma_p0_lat
#       vs the GO rule (sigma_p0_lat <= 0.08 AND lat_p99 <= 0.24).
# SANITY (deliverable 1) is folded into (2): a reach_rate>0 + success_rate>0 on --lookat auto under
# DETERMINISTIC inference IS the "rc1 flies 2-axis deterministically" precheck. If reach==0 on auto,
# that is the escape-hatch finding (training success was a stochastic-action artifact) -> REPORT it.
#
# WHY A CLUSTER JOB: the torch env subclasses diffaero + uses our torch plant; it needs the diffaero
# clone + CUDA-class torch. It CANNOT run on the dev laptop (diffaero absent, torch CPU-only, confirmed
# 2026-06-18). The evaluator was BUILT + statically validated on the laptop; this is the RUN step.
#
# ADROIT AUP: compute on SLURM only; compute nodes have no internet; output -> /scratch; run checkquota.
#
# 🚩 PRE-SYNC (the Adroit peregrine_repo is a FILE COPY, not a git clone -- no `git pull` there). BEFORE
# submitting, SYNC the new/edited files to the LOGIN node:
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch_eval.py   (the evaluator -- NEW)
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch.sbatch    (this script -- NEW)
# (the diffaero conda env + peregrine_racing_inc8.py / inc8_reward.py / inc8_estimator_emul.py are
# already present from the rc1 training sync.)
set -x
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

EVAL=/scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch_eval.py
SEEDS=${SEEDS:-0 1 2}
CKPT_SUB=${CKPT_SUB:-periodic}     # checkpoint subdir saved by the trainer lifeline (periodic|best|emergency)
NENVS=${NENVS:-256}
HORIZONS=${HORIZONS:-2.5}          # eval runs horizons*max_time/dt steps, accepting EVERY gate-4 crossing
STANDFRAC=${STANDFRAC:-1.0}        # 1.0 = every reset targets gate-0 (full course == numpy trainreset regime)
EVALSEED=${EVALSEED:-0}            # torch seed for spawn + estimator-DR reproducibility
EXTRA=${EXTRA:-}                   # submit-time passthrough (e.g. --plant lapse, --dynamics-dr)

# numpy-tool yaw-only baseline (laptop-reproduced 2026-06-18, rc1 seed0, start=trainreset) -- the
# cross-check target the torch yaw-only sigma_p0_lat must land near.
NUMPY_YAW_LAT=0.1768

echo "=================================================================================="
echo "inc8 sigma_p0 TORCH-ENV eval  seeds='${SEEDS}'  ckpt_sub=${CKPT_SUB}  n_envs=${NENVS}"
echo "  numpy-tool yaw-only baseline (cross-check target): sigma_p0_lat=${NUMPY_YAW_LAT}"
echo "=================================================================================="

for SEED in ${SEEDS}; do
  RUNNAME=inc8_recenter_seed${SEED}_rc1
  RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/${RUNNAME}
  # the trainer may write a runname-decorated subdir; resolve it (the recenter sbatch does the same).
  RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* 2>/dev/null | head -1)
  RUN=${RUNDECOR:-$RUNDIR}
  CKPT=${RUN}/${CKPT_SUB}
  echo ""
  echo "===== seed ${SEED}: ckpt=${CKPT} ====="
  if [ ! -f "${CKPT}/actor.pth" ]; then
    echo "🚩 seed ${SEED}: ${CKPT}/actor.pth NOT FOUND -- skipping (check CKPT_SUB / run dir)."
    continue
  fi

  echo "----- (1) YAW-ONLY instrument cross-check (target sigma_p0_lat ~= ${NUMPY_YAW_LAT}) -----"
  python ${EVAL} --ckpt ${CKPT} --lookat yaw --n-envs ${NENVS} --horizons ${HORIZONS} \
    --standing-frac ${STANDFRAC} --seed ${EVALSEED} ${EXTRA} \
    2>&1 | tee /scratch/network/fl3689/inc8_sigmap0_torch_seed${SEED}_yaw.txt

  echo "----- (2) 2-AXIS (auto) -- THE DELIVERABLE (sanity reach>0 folded in) -----"
  python ${EVAL} --ckpt ${CKPT} --lookat auto --n-envs ${NENVS} --horizons ${HORIZONS} \
    --standing-frac ${STANDFRAC} --seed ${EVALSEED} ${EXTRA} \
    2>&1 | tee /scratch/network/fl3689/inc8_sigmap0_torch_seed${SEED}_auto.txt
done

echo ""
echo "=================================================================================="
echo "SUMMARY (grep the machine-readable lines; compare yaw sigma_p0_lat to ${NUMPY_YAW_LAT}):"
echo "=================================================================================="
grep -h "SIGMAP0_TORCH_SUMMARY" /scratch/network/fl3689/inc8_sigmap0_torch_seed*_*.txt 2>/dev/null

# Instrument-agreement check on the yaw-only runs (|torch - numpy| < 0.04 => AGREE).
awk -v base=${NUMPY_YAW_LAT} '
  /SIGMAP0_TORCH_SUMMARY/ && /lookat=yaw/ {
    for (i=1;i<=NF;i++) if ($i ~ /^sigma_p0_lat=/) { split($i,a,"="); d=a[2]-base; if(d<0)d=-d;
      printf "  YAW CROSS-CHECK: torch sigma_p0_lat=%.4f vs numpy %.4f  delta=%.4f  %s\n",
        a[2], base, d, (d<0.04 ? "AGREE (instrument trustworthy)" : "DISAGREE (investigate before trusting 2-axis)") }
  }' /scratch/network/fl3689/inc8_sigmap0_torch_seed*_yaw.txt 2>/dev/null

echo "INC8_SIGMAP0_TORCH_DONE"

# =====================================================================================================
# LAUNCH (from the Adroit LOGIN node, after the PRE-SYNC above):
#   sbatch /scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch.sbatch
#   # single seed / different checkpoint subdir / include plant DR:
#   sbatch --export=ALL,SEEDS=0,CKPT_SUB=best,EXTRA="--dynamics-dr" \
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch.sbatch
# READ: the yaw cross-check FIRST (does the torch instrument reproduce ~0.18? if not, the 2-axis number
# is suspect -- report the discrepancy). Then the 2-axis sigma_p0_lat vs the 0.08 GO rule.
# =====================================================================================================
