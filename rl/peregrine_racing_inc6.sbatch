#!/bin/bash
#SBATCH --job-name=peregrine_racing_inc6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/network/fl3689/inc6_run_%j.out
# S17 (inc6) retrain -- the inc5 recipe with the MEASURED MOTOR MIXER ON (session S17, 2026-06-11).
# Delta vs rl/peregrine_racing_inc5.sbatch (stage1_inc5 winner):
#   * +dynamics.dr_mixer=true -- per-motor clip of collective +- rate differentials (live-deploy
#     diag 2026-06-11: thr 0 + yaw rail -> ~hover PARASITIC lift; collective ~1 -> rate-authority
#     sag). S17 DR bands (idle [0.04,0.08], kappa_err [0.060,0.085], kappa_hold [0.030,0.060],
#     zeta_yaw [0.20,0.55]). The inc5 bang-bang/dither style lived exactly in the two corners
#     this models -- 0/20 live despite 10/10 in the (mixer-blind) twin.
#   * transport-delay DR CENTERED on the measured live latency (2 ticks / 67 ms, diag Section 4):
#     +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3 -> {1,2,3} (was {0,1,2}).
#   * ACTION-RATE regularization sweep: RW_DACT (the existing R5 ||delta action||^2 weight,
#     inc5 trained at 1.0 and still railed) -- swept like S15 swept rw_tilt. Style gate:
#     ACTRATE_SUMMARY thr_p95/yaw_p95 <= 0.5 span/tick, yaw_flip <= 5%.
#   * chained evals run --plant mixer (the fully measured S17 plant).
# Everything else (env, courses, PPO, lifelines) identical to inc5.
# Parameterize: sbatch --export=ALL,SEED=1,RW_DACT=4.0,TAG=inc6_d4_s1 rl/peregrine_racing_inc6.sbatch
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
# inc5 winner (t96_s1) weights -- the proven aero-ON recipe, now the DEFAULTS
RW_TILT=${RW_TILT:-96.0}
RW_COLL=${RW_COLL:-75.0}
RW_MISS=${RW_MISS:-40.0}
RW_OOB=${RW_OOB:-75.0}
RW_FTIME=${RW_FTIME:-0.25}
RW_DACT=${RW_DACT:-4.0}
TAG=${TAG:-inc6_s${SEED}}
RUNDIR=/scratch/network/fl3689/inc6_runs/${TAG}
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"

echo "=== STAGE1.6: inc5 recipe + MIXER DR + latency {1,2,3} (seed=$SEED nupd=$NUPD tilt=$RW_TILT coll=$RW_COLL miss=$RW_MISS oob=$RW_OOB ftime=$RW_FTIME dact=$RW_DACT) ==="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_racing.py \
  env=racing env.name=peregrine_racing \
  +env.course_mode=random +env.standing_start_frac=0.3 +env.rw_tilt=${RW_TILT} \
  +env.rw_collision=${RW_COLL} +env.rw_miss=${RW_MISS} +env.rw_oob=${RW_OOB} \
  +env.rw_finish_time=${RW_FTIME} +env.rw_dact=${RW_DACT} \
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 \
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true \
  +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3 \
  dynamics.controller.max_normed_thrust=3.765 \
  algo=ppo n_envs=${NENVS} n_updates=${NUPD} save_freq=200 log_freq=10 \
  headless=True device=0 seed=${SEED} runname=peregrine_stage1_6_${TAG} \
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
  echo "=== EVAL: held-out VQ1 (MIXER-ON, the acceptance gate) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course vq1 --plant mixer --n-envs 256 --max-time 40 --horizons 2.5
  echo "=== EVAL: random courses (MIXER-ON, the generalization number) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course random --plant mixer --n-envs 256 --max-time 40 --horizons 2.5
fi
echo "INC6_DONE TAG=$TAG"
ls -dt "$RUNDIR"/* 2>/dev/null | head -5
