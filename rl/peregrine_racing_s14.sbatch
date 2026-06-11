#!/bin/bash
#SBATCH --job-name=peregrine_racing_s14
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/network/fl3689/s14_run_%j.out
# S1.4 retrain -- env-coherence redesign + procedural track randomization + map-ON plant.
# (Session S15, 2026-06-11. See rl/peregrine_racing.py module docstring for the full design.)
#   * torch backend (registered by the launcher) => per-env DR ACTIVE: super-rate map ALWAYS ON
#     under DR (s~U[0.25,0.35], tau~U[0.015,0.030], alpha_max r/p~U[200,320] yaw x80/260,
#     hover +-5%, drag +-30%, latency {0,1,2} steps). G0 fixed at nominal. g=9.80665.
#   * dynamics.controller.max_normed_thrust=3.765 -- live collective ceiling.
#   * +env.course_mode=random -- procedural 6-gate courses per env (VQ1 HELD OUT for eval).
#   * +env.standing_start_frac=0.3 -- real-race standing starts (rest, tilted pad, tail-first).
#   * reward/termination = the S1.4 coherent set (defaults in peregrine_racing.RewardWeights;
#     override any with +env.rw_<name>=...).
#   * GuardedPPO (NaN-grad skip) + periodic/emergency checkpoints (launcher lifelines): the S1.3
#     5000-update run NaN'd at ~2243 and wrote nothing.
# Parameterize: sbatch --export=ALL,SEED=1,NUPD=6000,TAG=s14_seed1 rl/peregrine_racing_s14.sbatch
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

SEED=${SEED:-0}
NUPD=${NUPD:-6000}
NENVS=${NENVS:-2048}
TAG=${TAG:-s14_seed${SEED}}
RUNDIR=/scratch/network/fl3689/s14_runs/${TAG}
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"

echo "=== STAGE1.4: coherent env + random courses + map-ON plant (seed=$SEED nupd=$NUPD) ==="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_racing.py \
  env=racing env.name=peregrine_racing \
  +env.course_mode=random +env.standing_start_frac=0.3 \
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 +dynamics.dr=true \
  dynamics.controller.max_normed_thrust=3.765 \
  algo=ppo n_envs=${NENVS} n_updates=${NUPD} save_freq=200 log_freq=10 \
  headless=True device=0 seed=${SEED} runname=peregrine_stage1_4_${TAG} \
  hydra.run.dir=${RUNDIR}
echo "TRAIN_RC=$?"

# newest actor.pth by mtime (emergency/periodic beat a stale dir; checkpoints/ is written last
# on a clean run so it wins naturally)
CKPT=$(find "$RUNDIR" -name actor.pth -printf '%T@ %p\n' | sort -n | awk 'END{print $2}')
echo "CKPT=$CKPT"
if [ -n "$CKPT" ]; then
  md5sum "$CKPT"
  # belt-and-braces sidecar (the launcher also writes one at save time)
  [ -f "$(dirname "$CKPT")/actor.json" ] || \
    echo '{"act_max_thrust": 3.765, "act_max_rate": 3.14}' > "$(dirname "$CKPT")/actor.json"
  cat "$(dirname "$CKPT")/actor.json"
  echo "=== EVAL: held-out VQ1 (map-ON, the acceptance gate) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course vq1 --plant map --n-envs 256 --max-time 40 --horizons 2.5
  echo "=== EVAL: random courses (map-ON, the generalization number) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course random --plant map --n-envs 256 --max-time 40 --horizons 2.5
fi
echo "S14_DONE TAG=$TAG"
ls -dt "$RUNDIR"/* 2>/dev/null | head -5
