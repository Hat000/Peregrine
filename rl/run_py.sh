#!/bin/bash
# Generic Peregrine-on-diffaero python runner: env setup (modules/conda/symlink/PYTHONPATH) + python "$@".
# Used for the login-node precheck (niced) and any one-off diffaero python invocation.
set -x
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero || exit 2
python "$@"
echo "RUN_PY_DONE rc=$?"
