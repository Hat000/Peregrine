#!/bin/bash
#SBATCH --job-name=inc8_sigmap0_gpsweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/network/fl3689/inc8_sigmap0_gpsweep.out
#
# DIAGNOSTIC (sigmap0-adroit 2026-06-19): the canonical run (job 3278151) showed rc1's 2-axis look-at
# (g_yaw=-3, g_pitch=3) CRASHES under DETERMINISTIC inference (reach=0, ~1 s episodes) while the
# yaw-only mode flies (seed0 reach 53%, sigma_p0_lat 0.198 == numpy 0.177 cross-check AGREE). The
# pointing primitive WORKS (band_el 66->20 deg, fix_rate 0->0.25) but g_pitch=3 destabilizes flight.
# This sweep finds the g_pitch STABILITY BOUNDARY on the one flying seed (seed0): the largest elevation
# gain that still FLIES (reach>0), and whether any flyable g_pitch gives a sigma_p0_lat below the
# yaw-only 0.198 (ideally toward the 0.08 GO target). g_yaw held at -3 throughout (isolate elevation).
# Measurement-only; same SLURM/scratch/GPU shape as the audited sbatch (AUP-clean).
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
CKPT=/scratch/network/fl3689/diffaero/outputs/train/inc8_recenter_seed0_rc1/best   # the one flying seed

run() {   # $1=label  $2..=eval args
  local label=$1; shift
  echo ""; echo "===== seed0 ${label} ====="
  python ${EVAL} --ckpt ${CKPT} --n-envs 256 --horizons 2.5 --standing-frac 1.0 --seed 0 "$@" \
    2>&1 | tee /scratch/network/fl3689/inc8_sigmap0_gpsweep_seed0_${label}.txt
}

run off      --lookat off                       # racing-line baseline: does seed0 fly with NO pointing?
run gp0p5    --lookat auto --g-yaw -3 --g-pitch 0.5
run gp1p0    --lookat auto --g-yaw -3 --g-pitch 1.0
run gp1p5    --lookat auto --g-yaw -3 --g-pitch 1.5
run gp2p0    --lookat auto --g-yaw -3 --g-pitch 2.0
run gp2p5    --lookat auto --g-yaw -3 --g-pitch 2.5

echo ""
echo "================================ GP-SWEEP SUMMARY ================================"
grep -h SIGMAP0_TORCH_SUMMARY /scratch/network/fl3689/inc8_sigmap0_gpsweep_seed0_*.txt 2>/dev/null
echo "INC8_GPSWEEP_DONE"
