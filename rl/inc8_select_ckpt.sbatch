#!/bin/bash
#SBATCH --job-name=inc8_select_ckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/fl3689/inc8_select_ckpt.out
#
# DETERMINISTIC CHECKPOINT SELECTION (inc8-deterministic-retrain, 2026-06-19). Pick the snapshot whose
# DETERMINISTIC mean (test=True) flies + centres BEST, instead of the runner's STOCHASTIC high-water
# "best" (which did NOT transfer to the mean -- the rc1 failure). Pure orchestration around the VALIDATED
# sigma_p0 instrument rl/inc8_sigmap0_torch_eval.py (seed0 yaw cross-check 0.198 vs numpy 0.177 AGREE,
# job 3278151) -- no new diffaero-importing code.
#
# STAGE 1 (CHEAP, all snapshots): deterministic reach_rate at n_envs=256, horizons=1.0 -> rank. reach is
#   the GATE (the un-measurable problem == reach 0). STAGE 2 (FULL, top-K by reach): the validated
#   n_envs=256, horizons=2.5 sigma_p0 + the yaw cross-check -> the GO/NO-GO number. Pick min sigma_p0_lat
#   among reach>=THRESH snapshots with n>=30.
#
# ADROIT AUP: SLURM only; output -> /scratch; accurate --mem; 1 GPU; checkquota. Pre-sync the eval +
# this file to /scratch/network/fl3689/peregrine_repo/rl/ before submit.
#
# SUBMIT: sbatch --export=ALL,RUNDIR=<run>,GPITCH=<trained g_pitch>,RUNTAG=<tag> rl/inc8_select_ckpt.sbatch
#   knobs: RUNDIR (required), GPITCH (MUST match the trained gain: 1.0/1.5/...; default 3.0=rc1),
#          LOOKAT (auto|yaw, default auto), TOPK (default 3), REACH_THRESH (default 0.30).
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
exec > /scratch/network/fl3689/inc8_select_ckpt${RUN_SUFFIX}.out 2>&1
set -x
hostname
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

RUNDIR=${RUNDIR:?set RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/<run>}
LOOKAT=${LOOKAT:-auto}
GPITCH=${GPITCH:-3.0}          # 🚩 MUST match the policy's TRAINED look-at pitch gain (eval default is
                              # 3.0 = rc1's gain; a detstab policy trained at g_pitch=1.0/1.5 MUST be
                              # evaluated at its OWN gain or it crashes like rc1-at-3 -> false NO-GO).
TOPK=${TOPK:-3}
REACH_THRESH=${REACH_THRESH:-0.30}
EVAL=/scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch_eval.py
RANK=/scratch/network/fl3689/inc8_select_ckpt${RUN_SUFFIX}_rank.txt
: > ${RANK}

# Candidate checkpoint dirs: all dense snapshots + the runner's best/periodic for reference.
CKPTS=$(ls -d ${RUNDIR}/snapshots/upd* 2>/dev/null; ls -d ${RUNDIR}/best ${RUNDIR}/periodic ${RUNDIR}/periodic_prev 2>/dev/null)
if [ -z "${CKPTS}" ]; then echo "NO checkpoints under ${RUNDIR} (snapshots/ best/ periodic/)"; exit 1; fi

echo "=== STAGE 1: cheap DETERMINISTIC reach over $(echo "${CKPTS}" | wc -l) checkpoints (lookat=${LOOKAT}) ==="
for CK in ${CKPTS}; do
  [ -f "${CK}/actor.pth" ] || continue
  LINE=$(python ${EVAL} --ckpt "${CK}" --lookat ${LOOKAT} --g-pitch ${GPITCH} --n-envs 256 --horizons 1.0 2>/dev/null \
          | grep SIGMAP0_TORCH_SUMMARY | tail -1)
  RR=$(echo "${LINE}"  | grep -o 'reach_rate=[0-9.]*'   | cut -d= -f2)
  SL=$(echo "${LINE}"  | grep -o 'sigma_p0_lat=[0-9.nan]*' | cut -d= -f2)
  echo "SELECT ${RR:-0.000} ${SL:-nan} ${CK}" | tee -a ${RANK}
done

echo "=== STAGE 1 RANK (by deterministic reach_rate, desc) ==="
sort -k2 -gr ${RANK} | tee ${RANK}.sorted

# Top-K candidates with reach >= threshold, for the full sigma_p0.
TOP=$(awk -v t=${REACH_THRESH} '$2+0 >= t {print $4}' ${RANK}.sorted | head -${TOPK})
if [ -z "${TOP}" ]; then
  echo "🚩 NO snapshot reaches gate-4 deterministically at reach>=${REACH_THRESH} (lookat=${LOOKAT})."
  echo "   => this config did NOT produce a deterministic-flyable mean. Report the bracket; try a lower"
  echo "   g_pitch / gentler cap, or escalate (look-at magnitude vs stable mean)."
  exit 0
fi

echo "=== STAGE 2: FULL validated sigma_p0 on top-${TOPK} reach candidates (n_envs=256 horizons=2.5) ==="
for CK in ${TOP}; do
  echo "--- FULL sigma_p0: ${CK} (lookat=${LOOKAT} g_pitch=${GPITCH}, the 2-axis deliverable) ---"
  python ${EVAL} --ckpt "${CK}" --lookat ${LOOKAT} --g-pitch ${GPITCH} --n-envs 256 --horizons 2.5 | grep -E "SIGMAP0_TORCH_SUMMARY|GO RULE|sigma_p0_lat ="
  echo "--- yaw cross-check (instrument validity): ${CK} ---"
  python ${EVAL} --ckpt "${CK}" --lookat yaw  --n-envs 256 --horizons 2.5 | grep SIGMAP0_TORCH_SUMMARY
done
echo "INC8_SELECT_DONE -- pick min sigma_p0_lat among the STAGE-2 lines with reach n>=30 (verdict in each)."
