#!/bin/bash
#SBATCH --job-name=v20_VvsC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
exec > /scratch/network/fl3689/v20_VvsC_${SLURM_JOB_ID}.out 2>&1
set -x
hostname
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

# V vs C (2026-07-27): did the VERTICAL COURSE STRUCTURE cause v2.0's +20%, or something else?
#
# THIS IS A TRUE SINGLE-VARIABLE TEST. Verified by diffing the two runs' recorded
# .hydra/overrides.yaml: V and C differ by EXACTLY THREE LINES, all course geometry --
#     ++env.course_drop_lo=-10.0
#     ++env.course_drop_hi=10.0
#     ++env.course_gates_ceiling=16.0
# (plus runname). Same warm start, same 18k updates, same 25 Hz vision cadence, same
# everything else. C ran on V100 and V on MIG -- hardware differs, config does not.
#
# NOTE this is a CLEANER contrast than the earlier vert-vs-flat head-to-head, where the two
# ENVS differed in cadence AS WELL as course and so could not attribute a cross-env delta.
# Here the POLICIES differ by the course knobs alone, and each is scored in BOTH envs.
#
# METHOD (unchanged, and the reason the v2.0 result was trustworthy): the eval env is
# REPLAYED FROM A REAL RUN'S RECORDED OVERRIDES, never hand-transcribed.
#   ENV=vert <- v20V_s0's overrides (climbing course + 25 Hz)  == DEPLOY-FAITHFUL
#   ENV=flat <- v19W_s0's overrides (flat course + 30 Hz)      == the OLD world
# Every policy sees the SAME env within a block, so the only thing varying is the actor.
#
# --time is 02:00:00 deliberately: it is the ONLY start-time knob we control, and 8 rollouts
# of n_updates=1 need nowhere near the 20 h wall. Short wall == backfill eligible.

TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_ego.py
OUTROOT=/scratch/network/fl3689/diffaero/outputs/train

REF_VERT=${OUTROOT}/ego_dual_gate_fullstack_floor_pef16_seed0_v20V_s0_mig
REF_FLAT=${OUTROOT}/ego_dual_gate_fullstack_floor_pef16_seed0_v19W_s0_mig

P_v20Vs0=${OUTROOT}/ego_dual_gate_fullstack_floor_pef16_seed0_v20V_s0_mig/checkpoints
P_v20Vs1=${OUTROOT}/ego_dual_gate_fullstack_floor_pef16_seed1_v20V_s1_mig/checkpoints
P_v20Cs0=${OUTROOT}/ego_dual_gate_fullstack_floor_pef16_seed0_v20C_s0_v100/checkpoints
P_v19Ws0=${OUTROOT}/ego_dual_gate_fullstack_floor_pef16_seed0_v19W_s0_mig/checkpoints

ovr_of () {   # $1 = reference run dir -> its recorded overrides as one command line
  sed -e 's/^- //' -e '/^hydra\.run\.dir/d' -e '/^runname=/d' -e '/^seed=/d' \
      "$1/.hydra/overrides.yaml" | tr '\n' ' '
}

for ENVMODE in vert flat; do
  case ${ENVMODE} in
    vert) REF=${REF_VERT} ;;
    flat) REF=${REF_FLAT} ;;
  esac
  if [ ! -f "${REF}/.hydra/overrides.yaml" ]; then
    echo "VC_FATAL[${ENVMODE}]: no overrides.yaml under ${REF}"; continue
  fi
  OV=$(ovr_of "${REF}")
  echo "=== ENV=${ENVMODE} built from ${REF} ==="

  for NAME in v20Vs0 v20Cs0 v20Vs1 v19Ws0; do
    case ${NAME} in
      v20Vs0) CK=${P_v20Vs0} ;;
      v20Cs0) CK=${P_v20Cs0} ;;
      v20Vs1) CK=${P_v20Vs1} ;;
      v19Ws0) CK=${P_v19Ws0} ;;
    esac
    if [ ! -f "${CK}/actor.pth" ] || [ ! -f "${CK}/critic.pth" ]; then
      echo "VC_SKIP[${ENVMODE}/${NAME}]: ${CK} lacks actor.pth+critic.pth"; continue
    fi
    TAG=vc_${ENVMODE}_${NAME}
    python ${TRAIN} ${OV} \
      hydra.run.dir=/scratch/network/fl3689/diffaero/outputs/rollout/${TAG} \
      n_envs=2048 n_updates=1 save_freq=100000 log_freq=10 seed=0 runname=${TAG} \
      ++rollout_only=true ++rollout_ckpt=${CK} ++rollout_tag=${TAG}
    echo "VC_RC[${ENVMODE}/${NAME}]=$?"
  done
done

echo "=== V-vs-C SUMMARY (grep-ready) ==="
grep -hE "ROLL_EVAL|YAW_EVAL|PITCH_EVAL|VC_" /scratch/network/fl3689/v20_VvsC_${SLURM_JOB_ID}.out || true
