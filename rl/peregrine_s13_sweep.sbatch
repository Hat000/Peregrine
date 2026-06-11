#!/bin/bash
#SBATCH --job-name=s13sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/network/fl3689/s13_sweep_%j.out
# S1.3 tilt-regularization sweep: train (torch backend => DR + item-4 active) from the STANDING
# START with the LIVE thrust ceiling, then OFFLINE-eval the checkpoint (success + peak roll).
# Parameterized via --export: TAG, ATT (attitude weight), JERK (body-rate weight), NUPD, NENVS.
# All four S1.3 changes are ON: (1) max_normed_thrust=3.765, (2) standing_start_frac=0.3,
# (3) attitude/jerk regularization (swept), (4) DR (asym rate-gain band + latency) via torch backend.
set -x
: "${TAG:=att2}"; : "${ATT:=2.0}"; : "${JERK:=0.3}"; : "${NUPD:=1500}"; : "${NENVS:=2048}"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
RUNDIR=/scratch/network/fl3689/s13_runs/${TAG}
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
cd /scratch/network/fl3689/diffaero
echo "===== S1.3 SWEEP TAG=$TAG ATT=$ATT JERK=$JERK NUPD=$NUPD NENVS=$NENVS ====="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_racing.py \
  env=racing env.name=peregrine_racing +env.standing_start_frac=0.3 \
  env.reward_weights.quadrotor.attitude=${ATT} env.reward_weights.quadrotor.jerk=${JERK} \
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 +dynamics.dr=true \
  dynamics.controller.max_normed_thrust=3.765 \
  algo=ppo n_envs=${NENVS} n_updates=${NUPD} save_freq=200 log_freq=10 \
  headless=True device=0 seed=0 runname=s13_${TAG} \
  hydra.run.dir=${RUNDIR}
echo "TRAIN_RC=$?"
CKPT=$(find "$RUNDIR" -name actor.pth -path '*checkpoints*' | head -1)
if [ -z "$CKPT" ]; then CKPT=$(find "$RUNDIR" -name actor.pth | head -1); fi
echo "CKPT_DIR=$(dirname "$CKPT")"
echo "===== OFFLINE STANDING-START EVAL (clean twin, dr=off) ====="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
  --ckpt "$(dirname "$CKPT")" --n-envs 256 --max-time 40 --horizons 2.5
echo "SWEEP_DONE TAG=$TAG"
