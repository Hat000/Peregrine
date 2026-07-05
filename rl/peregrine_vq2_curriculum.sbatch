#!/bin/bash
#SBATCH --job-name=peregrine_vq2_curric
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# VQ2 RL controller CURRICULUM LADDER (spec T3.2) -- the slow-is-smooth staged train, warm-started
# stage-to-stage. Derived from peregrine_inc8_recenter.sbatch. This is the GPU-spend-gate rung 2
# (PORTFOLIO) once the smoke (peregrine_vq2_smoke.sbatch) is GREEN.
#
# STAGES (each continues from the prior stage's best checkpoint via +init_from -- the Xing state-teacher
# ladder, cheap): hover -> single_gate -> blackout_pass -> multi_gate. Each stage AUTO-gates the next on
# a FLIGHTCHECK (success_rate + n_passed_gates off 0); a stalled stage STOPS the ladder + reports.
#
# 🚩 SAME PRECONDITIONS as the smoke: (1) algo=appo IS wired on this branch (0392b64 applied:
# GuardedAPPO / AGENT_ALIAS["appo"]; critic reads get_state()=37-dim, independent of the obs-20/23
# arm) -- pass --export=ALL,ALGO=appo for the privileged critic; (2) the Adroit bridge + PRE-SYNC of
# the edited rl/ files. See peregrine_vq2_smoke.sbatch header for the file list + AUP.
#
# RUNTAG / REWARD (a|b|c) / A_BODY / ALGO / SEEDS passthrough via --export (see the smoke header).
set -x
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
GPUUTIL=/scratch/network/fl3689/peregrine_vq2_curric${RUN_SUFFIX}_gpu_util.log
exec > /scratch/network/fl3689/peregrine_vq2_curric${RUN_SUFFIX}.out 2>&1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

NENVS=${NENVS:-2048}
NUPD=${NUPD:-2000}        # per-stage budget (the early stages converge faster; the lap stage is the long one)
REWARD=${REWARD:-a}
A_BODY=${A_BODY:-0}
ALGO=${ALGO:-ppo}
SEED=${SEED:-0}
ENT_WEIGHT=${ENT_WEIGHT:-0.01}  # PPO/APPO entropy bonus coeff (diffaero default). 2026-07-05: the
                                # curr_a_s0b pilot's multi_gate stage drove the tanh-squashed log_std
                                # (network/agents.py LOG_STD_MIN=-5) into saturation by ~80% through its
                                # 2000-update budget -- entropy (and success_rate/n_passed_gates) then
                                # plateaued flat for the rest of the run. Override to slow the collapse.
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py
CURR=/scratch/network/fl3689/peregrine_repo/rl/vq2_curriculum.py

BASE=(
  env=racing env.name=peregrine_racing_inc8 +env.inc8=true +env.r5_arm=A
  +env.standing_start_frac=0.3
  +env.spin_rate_abort=10.0 +env.spin_time_abort=3.0
  +env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30
  +env.rw_perc_r_lo=12 +env.rw_perc_r_hi=28 +env.rw_perc_w_lo=1.5 +env.rw_perc_w_hi=1.5 +env.rw_perc_w0=0.05
  +env.lookat_r_lo=8.0 +env.lookat_r_hi=30.0
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true
  +dynamics.dr_force_bias=true +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3
  dynamics.controller.max_normed_thrust=3.765
  algo=${ALGO} algo.entropy_weight=${ENT_WEIGHT} headless=True device=0
  algo.critic_grad_norm=1.0
)
# 2026-07-05 audit-A1/A3 boundary protections (see rl/inc8_warmstart.py + rl/peregrine_train_inc8.py):
#   * critic_grad_norm=1.0 (BASE above): the unbounded critic + fresh Adam diverged at the multi_gate
#     boundary (value_loss 617 -> 269,156) and its garbage advantages destroyed the transferred policy.
#   * warmstart_reset_logstd: zero the RAW logstd after every warm-start load -- transfer the means,
#     never the exploitation schedule (3 easy stages ground std 0.223 -> 0.050 before the hard stage).
#   * critic_warmup_updates=100: actor gradients zeroed for the first 100 updates of EVERY stage so the
#     critic adapts to the new return scale before the policy moves.
BOUNDARY_OV=( +warmstart_reset_logstd=true +warmstart_reset_logstd_std=0.18 +critic_warmup_updates=100 )
case "${REWARD}" in
  a) REWARD_OV=( ) ;;
  b) REWARD_OV=( +env.rw_centering=1.0 ) ;;
  c) REWARD_OV=( +env.rw_centering=0.0 ) ;;
  *) echo "unknown REWARD=${REWARD} (a|b|c)"; exit 2 ;;
esac
if [ "${A_BODY}" = "1" ]; then ABODY_OV=( +env.obs_a_body=true ); else ABODY_OV=( ); fi

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# 2026-07-05 ladder rebuild (audit A0-A4): hover DROPPED (redundant with single_gate once the logstd
# reset exists); dual_gate INSERTED (first handoff/turn + obs[13:17] goes live BEFORE the full course).
STAGES="single_gate blackout_pass dual_gate multi_gate"
PREV_CKPT=""
for STAGE in ${STAGES}; do
  STAGE_OV=( $(python ${CURR} ${STAGE}) )
  RUNNAME=vq2_${STAGE}_seed${SEED}${RUN_SUFFIX}
  RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/${RUNNAME}
  INIT_OV=( )
  if [ -n "${PREV_CKPT}" ]; then INIT_OV=( +init_from=${PREV_CKPT} ); fi
  echo "=== VQ2 CURRICULUM stage=${STAGE} seed=${SEED}: ${NUPD} updates, init_from=${PREV_CKPT:-FRESH} ==="
  echo "    overrides: ${STAGE_OV[@]}"
  python ${TRAIN} "${BASE[@]}" "${STAGE_OV[@]}" "${REWARD_OV[@]}" "${ABODY_OV[@]}" "${INIT_OV[@]}" \
    "${BOUNDARY_OV[@]}" \
    hydra.run.dir=${RUNDIR} \
    n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
    seed=${SEED} runname=${RUNNAME} ${EXTRA:-}
  STAGE_RC=$?
  echo "STAGE_${STAGE}_RC=${STAGE_RC}"

  RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* 2>/dev/null | head -1)
  RUN=${RUNDECOR:-$RUNDIR}
  TRACE_LOG=/scratch/network/fl3689/peregrine_vq2_curric${RUN_SUFFIX}_${STAGE}_trace.txt
  python ${TBTRACE} "$RUN" --stride-updates 100 | tee ${TRACE_LOG}
  PREV_CKPT="${RUNDIR}/checkpoints"          # checkpoints save as a SIBLING of the hydra-decorated
                                              # ${RUN} dir, not inside it -- ${RUN}/checkpoints never
                                              # exists (2026-07-04: crashed single_gate/blackout_pass/
                                              # multi_gate on FileNotFoundError, hover unaffected since
                                              # stage 1 never reads PREV_CKPT).

  # FLIGHTCHECK: stop the ladder if this stage did not fly (single-gate stages pass gate-0; the lap
  # stage needs success). success_rate<0.005 AND n_passed_gates<0.1 => stalled. A nonzero STAGE_RC
  # (crash) also stops the ladder -- a crashed stage has no valid checkpoint for the next stage to
  # warm-start from, so continuing would just cascade the same failure into every remaining stage.
  VERDICT=$(awk '/FLIGHTCHECK/ {
      for (i=1;i<=NF;i++) {
        if ($i ~ /^success_rate_max=/)   { split($i,a,"="); sr=a[2]+0 }
        if ($i ~ /^n_passed_gates_max=/) { split($i,b,"="); npg=b[2]+0 }
      }
      if (sr < 0.005 && npg < 0.1) print "FAILED"; else print "FLEW";
    }' ${TRACE_LOG})
  echo "FLIGHTCHECK_VERDICT[${STAGE}]=${VERDICT:-UNKNOWN}"
  if [ "${STAGE_RC}" != "0" ] || [ "${VERDICT}" != "FLEW" ]; then
    echo "🚩 CURRICULUM STALLED at stage=${STAGE} (RC=${STAGE_RC}, verdict=${VERDICT:-UNKNOWN}) -- not advancing."
    echo "   REPORT the trace; consider RW_TC sweep / a longer stage budget / the DAgger bootstrap (spec T3.4)."
    break
  fi
done

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}
echo "VQ2_CURRICULUM_DONE"

# =====================================================================================================
# LAUNCH (Adroit login, after PRE-SYNC + serve daemon up; smoke GREEN first):
#   one seed, arm A, obs-20, symmetric, full slow ladder:
#     sbatch --export=ALL,RUNTAG=curr_a_s0 .../rl/peregrine_vq2_curriculum.sbatch
#   the PORTFOLIO = the recenter 3-seed shape x reward arms (launch as SEPARATE jobs so one seed's
#   crash can't kill the others):
#     for s in 0 1 2; do for r in a b; do
#       sbatch --export=ALL,SEED=$s,REWARD=$r,RUNTAG=curr_${r}_s$s .../peregrine_vq2_curriculum.sbatch
#     done; done
#   the a_body 23-dim A/B arm alongside: add A_BODY=1 (a distinct sidecar obs_dim=23).
#   RUNGS (speed ladder on the WINNING arm ONLY) are STAGED + GATED on T1 above-hover thrust/drag
#   validation (the command-sweep capture) -- do NOT open the >30 m spacing / high forward-speed rungs
#   until that lands. See spec T1.4 / T3.5.
# =====================================================================================================
