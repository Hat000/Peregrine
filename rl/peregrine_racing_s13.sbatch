#!/bin/bash
#SBATCH --job-name=peregrine_racing_s13
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/network/fl3689/peregrine_racing_s13.out
# S1.3 retrain -- fixes the four transfer blockers measured in the S1.2 live validation
# (ShadowPC 2026-06-10, handoff/shadowpc-s12-rl-live-2026-06-10/):
#   1. dynamics.controller.max_normed_thrust=3.765  -- live collective ceiling (1.0/0.2656);
#      Stage-1 trained to 5.0 and used it -> gate-4 wall even in-twin with the live clip.
#   2. +env.standing_start_frac=0.3                 -- real race spawn (23.3 m, rest, tilted pad,
#      virtual-flip frame) so the policy needs no CTBR bridge.
#   3. env.reward_weights.quadrotor.attitude=2.0 + jerk=0.3 -- tilt/aggression regularization:
#      Stage-1 rolls through 104-126 deg before EVERY gate; live that enters the sim's
#      (untwinned) +-180deg-tilt yaw-spin regime + rate-transient overshoot -> tumble.
#      VALUES ARE A STARTING POINT -- sweep if success_rate collapses.
#   4. LATENCY + TRANSIENT DR -- REQUIRES diffaero_dynamics.py edits FIRST (see handoff doc):
#      transport-delay ring buffer (rl_plant supports it; torch mirror omits it) with per-env
#      delay in {0,1,2} steps, and an asymmetric rate_gain band (-10%..+30%, live overshoot
#      measured 9.7 vs 7.85 rad/s realized on saturated steps).
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
echo "=== STAGE1.3: live-feasible action bounds + standing start + tilt regularization ==="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_racing.py \
  env=racing env.name=peregrine_racing +env.standing_start_frac=0.3 \
  env.reward_weights.quadrotor.attitude=2.0 env.reward_weights.quadrotor.jerk=0.3 \
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 +dynamics.dr=true \
  dynamics.controller.max_normed_thrust=3.765 \
  algo=ppo n_envs=2048 n_updates=5000 save_freq=200 log_freq=10 \
  headless=True device=0 seed=0 runname=peregrine_stage1_3
echo "TRAIN_RC=$?"
echo "RACING_S13_DONE"
ls -dt /scratch/network/fl3689/diffaero/outputs/train/*/* 2>/dev/null | head -3
