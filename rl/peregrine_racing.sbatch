#!/bin/bash
#SBATCH --job-name=peregrine_racing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/network/fl3689/peregrine_racing.out
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
echo "=== STAGE1 inc1: PeregrineRacing (OUR 6-gate descending course) + peregrine_plant + DR, racing PPO n_envs=2048 ==="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_racing.py \
  env=racing env.name=peregrine_racing \
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 +dynamics.dr=true \
  algo=ppo n_envs=2048 n_updates=5000 save_freq=200 log_freq=10 \
  headless=True device=0 seed=0 runname=peregrine_stage1_inc1
echo "TRAIN_RC=$?"
echo "RACING_DONE"
echo "=== recent run dirs (checkpoint lives in <run>/checkpoints) ==="
ls -dt /scratch/network/fl3689/diffaero/outputs/train/*/* 2>/dev/null | head -3
