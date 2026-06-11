#!/bin/bash
# Parity gate runner (login-node CPU, niced -- trivial 16-env single-step algebraic check).
set -e
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/peregrine_repo/rl
echo "=== diffaero_dynamics md5 (should be cdf79613267644f94d3e89f5277bcf5f) ==="
md5sum diffaero_dynamics.py
echo "=== parity gate ==="
nice -n 19 python check_diffaero_gate.py
echo "PARITY_DONE rc=$?"
