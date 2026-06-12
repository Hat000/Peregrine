#!/bin/bash
#SBATCH --job-name=peregrine_racing_inc7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/network/fl3689/inc7_run_%j.out
# ============================================================================================
# inc7 -- the inc6 c16 recipe under the TRAINING DOCTRINE (2026-06-12,
# handoff/laptop-training-doctrine-2026-06-12/WRITEUP.md + docs/training_doctrine.md).
#
# 🚩 DO NOT SUBMIT until ALL THREE launch gates pass:
#   GATE 1: env features below EXIST and are unit-tested (LAPTOP-INC7-ENV session):
#           +env.body_radius_lo/hi  -- contact-true aperture: pass Linf <= 0.75 - r_body,
#                                      collision band (0.75 - r_body, 1.36 + r_body],
#                                      r_body ~ U[lo, hi] per env at reset (doctrine Sec 2)
#           +env.frame_depth_m      -- volumetric frame: collision band active over
#                                      gate-frame |x| <= depth (segment-vs-slab test)
#           +dynamics.dr_force_bias -- per-env random world bias ||b|| <= 3 m/s^2 in a
#                                      per-env random speed x tilt bin (doctrine Sec 3)
#   GATE 2: V100 config-matrix parity gate (standing flag since S18; now incl. dr_force_bias
#           config) -- run FIRST at Adroit contact.
#   GATE 3: full test suite green (598+ incl. new geometry tests + frame conventions).
#
# Delta vs rl/peregrine_racing_inc6.sbatch (everything else byte-identical):
#   * REWARD UNCHANGED -- c16 winner weights verbatim (RW_CORNER=16, RW_DACT=0). Doctrine Sec 1:
#     no aggression/proximity/style terms; margin comes from geometry, not reward.
#   * CONTACT-TRUE GEOMETRY (doctrine Sec 2): body_radius U[0.28,0.38] + frame_depth 0.30.
#     Evidence: live standing gate-3 crashes at Linf 0.37-0.49 -- positions the 0.75 point-mass
#     model scores as PASSES; strikes occur up to 0.5 m before the plane on slope-0.45 descents.
#   * STRUCTURED FORCE-BIAS DR (doctrine Sec 3): the regime-binned error class global scales
#     cannot represent (the climb-bin residual that displaced the standing approach).
#   * Plant: S17 mixer unchanged. If the S20 climb-bin refit lands before launch, the refit
#     constants ride along transparently (table values, not flags) -- fold in, NOT a blocker.
#   * Selection: >= 3 seeds (0,1,2), gen AVERAGED across seeds + doctrine Sec 5 gauntlet
#     (offset tails + corridor vs contact-true aperture; robustness probes) before any live.
#
# Predicted (on record): standing clears gate 3 with >= 0.3 m corridor margin; crossing tails
# <= 0.25 m; posture unchanged (~55 deg tilt); lap cost vs inc6 <= 0.3 s; bridge parity.
#
# Parameterize: sbatch --export=ALL,SEED=0,TAG=inc7_s0 rl/peregrine_racing_inc7.sbatch
# ============================================================================================
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
# inc6 winner (c16_s0) weights verbatim -- REWARD UNCHANGED (doctrine Sec 1)
RW_TILT=${RW_TILT:-96.0}
RW_COLL=${RW_COLL:-75.0}
RW_MISS=${RW_MISS:-40.0}
RW_OOB=${RW_OOB:-75.0}
RW_FTIME=${RW_FTIME:-0.25}
RW_DACT=${RW_DACT:-0.0}
RW_CORNER=${RW_CORNER:-16.0}
# doctrine Sec 2/3 (gate 1 features)
BODY_R_LO=${BODY_R_LO:-0.28}
BODY_R_HI=${BODY_R_HI:-0.38}
FRAME_DEPTH=${FRAME_DEPTH:-0.30}
TAG=${TAG:-inc7_s${SEED}}
RUNDIR=/scratch/network/fl3689/inc7_runs/${TAG}
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"

echo "=== STAGE1.7: c16 + contact-true geometry + dr_force_bias (seed=$SEED nupd=$NUPD corner=$RW_CORNER body_r=[$BODY_R_LO,$BODY_R_HI] depth=$FRAME_DEPTH) ==="
python /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_racing.py \
  env=racing env.name=peregrine_racing \
  +env.course_mode=random +env.standing_start_frac=0.3 +env.rw_tilt=${RW_TILT} \
  +env.rw_collision=${RW_COLL} +env.rw_miss=${RW_MISS} +env.rw_oob=${RW_OOB} \
  +env.rw_finish_time=${RW_FTIME} +env.rw_dact=${RW_DACT} +env.rw_corner=${RW_CORNER} \
  +env.body_radius_lo=${BODY_R_LO} +env.body_radius_hi=${BODY_R_HI} \
  +env.frame_depth_m=${FRAME_DEPTH} \
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 \
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true \
  +dynamics.dr_force_bias=true \
  +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3 \
  dynamics.controller.max_normed_thrust=3.765 \
  algo=ppo n_envs=${NENVS} n_updates=${NUPD} save_freq=200 log_freq=10 \
  headless=True device=0 seed=${SEED} runname=peregrine_stage1_7_${TAG} \
  hydra.run.dir=${RUNDIR}
echo "TRAIN_RC=$?"

CKPT=$(find "$RUNDIR" -name actor.pth -printf '%T@ %p\n' | sort -n | awk 'END{print $2}')
echo "CKPT=$CKPT"
if [ -n "$CKPT" ]; then
  md5sum "$CKPT"
  [ -f "$(dirname "$CKPT")/actor.json" ] || \
    echo '{"act_max_thrust": 3.765, "act_max_rate": 3.14}' > "$(dirname "$CKPT")/actor.json"
  cat "$(dirname "$CKPT")/actor.json"
  echo "=== EVAL: held-out VQ1 (MIXER-ON; doctrine gauntlet item 2) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course vq1 --plant mixer --n-envs 256 --max-time 40 --horizons 2.5
  echo "=== EVAL: random courses (gen; AVERAGE over >=3 seeds before selection) ==="
  python /scratch/network/fl3689/peregrine_repo/rl/peregrine_eval.py \
    --ckpt "$(dirname "$CKPT")" --course random --plant mixer --n-envs 256 --max-time 40 --horizons 2.5
fi
echo "INC7_DONE TAG=$TAG"
ls -dt "$RUNDIR"/* 2>/dev/null | head -5
