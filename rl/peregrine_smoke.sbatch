#!/bin/bash
#SBATCH --job-name=peregrine_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/network/fl3689/peregrine_smoke.out
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
echo "=== SMOKE: racing PPO with PEREGRINE plant injected, n_envs=2048, capped 240s ==="
timeout 240 python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train.py \
  env=racing dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 \
  algo=ppo n_envs=2048 headless=True device=0 2>&1 | tail -80
echo "SMOKE_PYTHON_RC=${PIPESTATUS[0]} (124 == hit the 240s cap, expected)"
echo "SMOKE_DONE"
