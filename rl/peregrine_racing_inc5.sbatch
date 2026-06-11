#!/bin/bash
#SBATCH --job-name=peregrine_racing_inc5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/network/fl3689/inc5_run_%j.out
# S1.5 (inc5) retrain -- the inc4 recipe with the MEASURED AERO ON (session INC5, 2026-06-11).
# Delta vs rl/peregrine_racing_s14.sbatch (inc4 round 3 = stage1_inc4):
#   * +dynamics.dr_aero=true -- quad body-frame drag + convex collective knot table, S16 DR
#     bands (c2 per-slot relative U[0.040,0.065]/0.052; K-table affine k~U[0.9,1.1] with hover
#     pinned h~U[0.98,1.02]; residual d1~U[0,0.08] replacing the legacy +-30% drag jitter).
#     The real sim brakes 2.2x harder at 9 m/s and has ~8 g full-stick -- inc4 learned both wrong.
#   * defaults = the S15 round-3 VALIDITY config (the stage1_inc4 winner), not the round-1 sweep.
#   * chained evals run --plant aero (the fully measured plant; eval scripts default map otherwise).
# Everything else (env, courses, PPO, lifelines) identical to s14.
# Parameterize: sbatch --export=ALL,SEED=1,TAG=inc5_s1 rl/peregrine_racing_inc5.sbatch
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
# inc4 winner (S15 round 3 "validity") weights -- the proven recipe, now the DEFAULTS
RW_TILT=${RW_TILT:-16.0}
RW_COLL=${RW_COLL:-75.0}
RW_MISS=${RW_MISS:-40.0}
RW_OOB=${RW_OOB:-75.0}
RW_FTIME=${RW_FTIME:-0.25}
RW_DACT=${RW_DACT:-1.0}
TAG=${TAG:-inc5_s${SEED}}
RUNDIR=/scratch/network/fl3689/inc5_runs/${TAG}
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"

echo "=== STAGE1.5: inc4 validity config + MEASURED AERO DR (seed=$SEED nupd=$NUPD tilt=$RW_TILT coll=$RW_COLL miss=$RW_MISS oob=$RW_OOB ftime=$RW_FTIME dact=$RW_DACT) ==="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_racing.py \
  env=racing env.name=peregrine_racing \
  +env.course_mode=random +env.standing_start_frac=0.3 +env.rw_tilt=${RW_TILT} \
  +env.rw_collision=${RW_COLL} +env.rw_miss=${RW_MISS} +env.rw_oob=${RW_OOB} \
  +env.rw_finish_time=${RW_FTIME} +env.rw_dact=${RW_DACT} \
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 \
  +dynamics.dr=true +dynamics.dr_aero=true \
  dynamics.controller.max_normed_thrust=3.765 \
  algo=ppo n_envs=${NENVS} n_updates=${NUPD} save_freq=200 log_freq=10 \
  headless=True device=0 seed=${SEED} runname=peregrine_stage1_5_${TAG} \
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
  echo "=== EVAL: held-out VQ1 (AERO-ON, the acceptance gate) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course vq1 --plant aero --n-envs 256 --max-time 40 --horizons 2.5
  echo "=== EVAL: random courses (AERO-ON, the generalization number) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course random --plant aero --n-envs 256 --max-time 40 --horizons 2.5
fi
echo "INC5_DONE TAG=$TAG"
ls -dt "$RUNDIR"/* 2>/dev/null | head -5
