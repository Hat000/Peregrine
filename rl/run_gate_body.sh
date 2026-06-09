#!/bin/bash
# Peregrine T4 gate body (env setup + run). Shared by the CPU pre-check (login, niced) and the
# GPU sbatch. Mirrors diffaero_smoke.sbatch's symlink + PYTHONPATH trick, plus the peregrine repo.
set -x
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero || exit 2
python /scratch/network/fl3689/peregrine_repo/rl/check_diffaero_gate.py
echo "GATE_BODY_DONE rc=$?"
