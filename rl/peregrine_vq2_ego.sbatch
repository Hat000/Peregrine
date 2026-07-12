#!/bin/bash
#SBATCH --job-name=peregrine_vq2_ego
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#
# VQ2 EGOCENTRIC (inc9) CURRICULUM LADDER (DESIGN.md docs/vq2-egocentric-gen/DESIGN.md §E) -- the
# slow-is-smooth staged train on the egocentric position-free env (peregrine_racing_ego), warm-started
# stage-to-stage. Derived from rl/peregrine_vq2_curriculum.sbatch but for the NEW ego generation:
#   * env=racing env.name=peregrine_racing_ego +env.ego=true (21-dim position-free obs, 16-dim
#     privileged critic, refined-B reward). NOT the inc8 gate-frame env.
#   * algo=appo EVERYWHERE (the privileged asymmetric critic). algo=ppo builds a SYMMETRIC obs-only
#     critic (get_state never consumed) = the inc8 seed-collapse root cause; the launcher's
#     post-agent-build assert (env.assert_appo_critic) FIRES on ppo/symmetric so a mis-set run raises
#     AssertionError BEFORE any compute is burned.
#   * gamma=0.9975 EVERYWHERE (LOAD-BEARING: keeps the crash penalty dominant over the banked progress
#     at the LATE gates / deployment scale -- gamma=0.99 thins there; see
#     tests/test_ego_reward.py::test_terminal_dominance_rollout_gamma_9975_holds_at_scale).
#   * standing_start_frac=1.0 (trivial dash removed) + blackout_pass DELETED (blackout is emergent from
#     the keypoint-visibility model) -- both live in rl/vq2_ego_curriculum.py, rendered per stage.
#
# STAGES (warm-started stage->stage via +init_from): single_gate -> handoff_drill -> dual_gate_full ->
# multi_gate. Per-stage update budgets: easy ~1500, hard ~5500 (DESIGN.md §E). Each stage AUTO-gates
# the next on a FLIGHTCHECK (success_rate + n_passed_gates off 0); a stalled stage STOPS the ladder +
# reports (do NOT burn compute on a diverging run -- the inc8 mechanism that cleanly halted curr_b at
# single_gate). single_gate MUST recover ~0.8 before advancing (DESIGN.md §6.7; validate-before-_COMMON).
#
# 🚩 ADROIT AUP (reference-adroit-princeton): SLURM ONLY (never compute on the login node); compute
# nodes have NO internet (PRE-STAGE all deps -- the edited rl/*.py + the diffaero clone -- before
# launch); accurate --mem; output -> /scratch; zero-GPU-util killed at 2 h; checkquota routinely.
#
# 🚩 PRE-SYNC (compute nodes have no internet): the NEW/edited ego files must be on /scratch before
# launch: rl/peregrine_train_ego.py, rl/peregrine_racing_ego.py, rl/ego_estimator.py,
# rl/gate_visibility.py, rl/ego_reward.py, rl/vq2_ego_curriculum.py (+ the base/inc8 deps the ego env +
# launcher import: rl/peregrine_racing.py, rl/peregrine_course.py, rl/diffaero_dynamics.py,
# rl/inc8_estimator_emul.py [quat_xyzw_to_matrix_torch], rl/inc8_warmstart.py [maybe_warmstart +
# reset_actor_logstd -- the warm-start LIFELINE the launcher wires; the launcher has a self-contained
# reset_actor_logstd fallback if a stale/older inc8_warmstart.py is synced]). A stale Adroit copy of any
# of these silently trains the wrong code.
set -x
RUNTAG=${RUNTAG:-ego_a}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
GPUUTIL=/scratch/network/fl3689/peregrine_vq2_ego${RUN_SUFFIX}_gpu_util.log
exec > /scratch/network/fl3689/peregrine_vq2_ego${RUN_SUFFIX}.out 2>&1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

NENVS=${NENVS:-2048}          # 2048 default; multi_gate bumps to MULTI_NENVS (4096) below
MULTI_NENVS=${MULTI_NENVS:-4096}
SEED=${SEED:-0}
# per-stage update budgets (DESIGN.md §E): easy ~1500, hard ~5500. Override via the *_UPD env vars.
UPD_single_gate=${UPD_single_gate:-1500}
UPD_handoff_drill=${UPD_handoff_drill:-1500}
UPD_dual_gate_full=${UPD_dual_gate_full:-5500}
UPD_multi_gate=${UPD_multi_gate:-5500}
# PERCEPTION-HONESTY chain defaults (2026-07-10; without these the eval NUPD lookup renders an empty
# n_updates= when the *_percept stages are run without explicit --export UPD_ values).
UPD_dual_gate_boot_floor_percept=${UPD_dual_gate_boot_floor_percept:-4000}
UPD_dual_gate_fullstack_floor_percept=${UPD_dual_gate_fullstack_floor_percept:-12000}
# ESTIMATOR-FAITHFUL (_pef) chain defaults (2026-07-11; same shape as the _percept chain)
UPD_dual_gate_boot_floor_pef=${UPD_dual_gate_boot_floor_pef:-4000}
UPD_dual_gate_fullstack_floor_pef=${UPD_dual_gate_fullstack_floor_pef:-12000}
# NODITHER fine-tune default (2026-07-12): a SHORT warm-started (from vpeffs0) dither-removal fine-tune.
UPD_dual_gate_fullstack_floor_pef_nodither=${UPD_dual_gate_fullstack_floor_pef_nodither:-3000}
ENT_WEIGHT=${ENT_WEIGHT:-0.01}   # APPO entropy bonus coeff (diffaero default)

TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_ego.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py
CURR=/scratch/network/fl3689/peregrine_repo/rl/vq2_ego_curriculum.py

# BASE recipe: the ego env + our plant + DR. algo=appo (privileged critic) is FIXED -- the launcher's
# assert refuses ppo. gamma is set PER STAGE by the curriculum _raw block (0.9975), NOT here.
BASE=(
  env=racing env.name=peregrine_racing_ego
  # (removed 2026-07-10) +env.spin_rate_abort=10.0 +env.spin_time_abort=3.0 were INC8-ONLY keys,
  # UNREAD by peregrine_racing_ego (consumed only by PeregrineRacingInc8) -- dead + misleading on
  # every ego run (and 10 rad/s would never have caught the 9.3 rad/s vn16 corkscrew anyway). The
  # ego spin gate is the +env.ego_spin_* family, set PER *_percept STAGE by the curriculum, never
  # here in BASE (B2b discipline). max_w_xy/max_w_z do not exist anywhere in this repo either --
  # the only real rate limits are the +/-3.14 action bounds (realized rate exceeds the command via
  # the plant super-rate map).
  +env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true
  +dynamics.dr_force_bias=true +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3
  dynamics.controller.max_normed_thrust=3.765
  algo=appo algo.entropy_weight=${ENT_WEIGHT} headless=True device=0
  algo.critic_grad_norm=1.0
  export.jit=false export.onnx=false
  +algo.noise_anneal=true +algo.noise_std_hold=0.30 +algo.noise_std_floor=0.30
  +algo.noise_hold_frac=1.0 +algo.noise_entropy_floor=0.0
)
# ^ NOISE CEILING (RC2 fix, 2026-07-07): a FLAT held std ceiling at 0.30 for the WHOLE run
# (std_hold==std_floor, hold_frac=1.0 -> no decay). Without it the constant entropy bonus drives
# actor_logstd to the LOG_STD_MAX bang-bang ceiling (std=exp(2)=7.39) -> undiscoverable rollouts (the
# ego single_gate collapse). Wired into peregrine_train_ego.py (mirror of the inc8 lifeline). The B2b
# lesson: the HELD ceiling is the flight win; decay-to-zero hurts easy stages -> no decay here.
# Paired with the emulated-camera nose-first virtual flip (RC1, default-ON in peregrine_racing_ego.py).
# ^ export.jit/onnx OFF: diffaero's ONNX/JIT exporter (utils/exporter.py) does NOT support the CTBR
# body-rate action_frame ("Unknown action frame: body" in post_process) -> close()'s post-train export
# raises, which would make the stage exit NONZERO and STOP the ladder even though training + the
# checkpoint (agent.save runs BEFORE export) fully succeeded. We deploy from the .pth checkpoint (never
# the ONNX/JIT artifact), so disabling both is loss-free. Fixes the 2026-07-07 ego smoke stage-crash.
# Warm-start boundary protections (mirrors the inc8 ladder): reset the exploration logstd after every
# warm-start load (transfer the means, never the ground-down exploitation schedule) + a critic warmup
# so the privileged critic adapts to the new return scale before the policy moves.
# +eval_det_steps=1200 (up from the peregrine_train_ego.py default of 300): the post-train faithful
# deterministic box-exit (_run_det_eval) was censoring episodes >10s at the 300-step default -- 1200
# gives full-lap completions headroom to actually terminate before DET_EVAL[...] aggregates.
BOUNDARY_OV=( +warmstart_reset_logstd=true +warmstart_reset_logstd_std=0.18 +critic_warmup_updates=100
              +eval_det_steps=1200 )

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# STAGES is overridable (default = the full ladder). Pass --export=ALL,STAGES=single_gate to run ONE
# stage in ISOLATION (the single_gate DIAGNOSTIC: validate a reward change to ~0.8 BEFORE the ladder can
# prematurely advance -- the flightcheck below only catches a TOTAL stall, DESIGN.md §6.7).
STAGES="${STAGES:-single_gate handoff_drill dual_gate_full multi_gate}"

# ---- OPTIONAL FAST PRECHECK (PRECHECK=1): construct + 3 updates at 512 envs on the FIRST stage, to
# catch a wiring / NaN / tensor-shape bug BEFORE the real budget is spent. Guards the 2026-07-07 reward-
# redesign env code specifically (the below-floor->contact reclassify, the area-distance GT foreshorten/
# distance computation, and the 3 new compute_ego_reward kwargs). A crash here exits nonzero immediately.
if [ "${PRECHECK:-0}" = "1" ]; then
  PC_STAGE=$(echo ${STAGES} | awk '{print $1}')
  PC_OV=( $(python ${CURR} ${PC_STAGE}) )
  echo "=== EGO PRECHECK: stage=${PC_STAGE}, 512 envs x 3 updates (wiring/NaN/dim) ==="
  python ${TRAIN} "${BASE[@]}" "${PC_OV[@]}" "${BOUNDARY_OV[@]}" \
    n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=${SEED} \
    runname=ego_precheck${RUN_SUFFIX} ${EXTRA:-}
  PRECHECK_RC=$?
  echo "EGO_PRECHECK_RC=${PRECHECK_RC}"
  if [ "${PRECHECK_RC}" != "0" ]; then
    echo "🚩 EGO PRECHECK FAILED (RC=${PRECHECK_RC}) -- aborting before the real budget. Fix the wiring."
    kill ${SMI_PID} 2>/dev/null
    exit ${PRECHECK_RC}
  fi
fi

PREV_CKPT=""
for STAGE in ${STAGES}; do
  STAGE_OV=( $(python ${CURR} ${STAGE}) )
  # per-stage update budget + n_envs (multi_gate gets the bigger env batch for throughput).
  eval "NUPD=\${UPD_${STAGE}}"
  if [ "${STAGE}" = "multi_gate" ]; then STAGE_NENVS=${MULTI_NENVS}; else STAGE_NENVS=${NENVS}; fi
  RUNNAME=ego_${STAGE}_seed${SEED}${RUN_SUFFIX}
  RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/${RUNNAME}
  INIT_OV=( )
  if [ -n "${PREV_CKPT}" ]; then INIT_OV=( +init_from=${PREV_CKPT} ); fi
  echo "=== VQ2 EGO CURRICULUM stage=${STAGE} seed=${SEED}: ${NUPD} updates, n_envs=${STAGE_NENVS}, init_from=${PREV_CKPT:-FRESH} ==="
  echo "    overrides: ${STAGE_OV[@]}"
  python ${TRAIN} "${BASE[@]}" "${STAGE_OV[@]}" "${INIT_OV[@]}" \
    "${BOUNDARY_OV[@]}" \
    hydra.run.dir=${RUNDIR} \
    n_envs=${STAGE_NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
    seed=${SEED} runname=${RUNNAME} ${EXTRA:-}
  STAGE_RC=$?
  echo "STAGE_${STAGE}_RC=${STAGE_RC}"

  RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* 2>/dev/null | head -1)
  RUN=${RUNDECOR:-$RUNDIR}
  TRACE_LOG=/scratch/network/fl3689/peregrine_vq2_ego${RUN_SUFFIX}_${STAGE}_trace.txt
  python ${TBTRACE} "$RUN" --stride-updates 100 | tee ${TRACE_LOG}
  PREV_CKPT="${RUNDIR}/checkpoints"          # the runner's end-of-run save (actor.pth + critic.pth +
                                              # actor.json -- exactly what maybe_warmstart needs) lands at
                                              # ${RUNDIR}/checkpoints, a SIBLING of the hydra-decorated
                                              # ${RUN} dir, NOT inside it -- ${RUN}/checkpoints never
                                              # exists (the proven vq2_curriculum.sbatch 2026-07-04 fix).
                                              # The NEXT stage warm-starts here via +init_from (wired LIVE
                                              # by peregrine_train_ego.py's maybe_warmstart lifeline).

  # FLIGHTCHECK early-abort: stop the ladder if this stage did not fly. success_rate<0.005 AND
  # n_passed_gates<0.1 => stalled. A nonzero STAGE_RC (crash) also stops the ladder -- a crashed stage
  # has no valid checkpoint for the next stage to warm-start from, so continuing just cascades the
  # failure. (single_gate MUST reach ~0.8 to be a healthy promote -- watch the trace; the automated
  # gate only catches a TOTAL stall, DESIGN.md §6.7.)
  VERDICT=$(awk '/FLIGHTCHECK/ {
      for (i=1;i<=NF;i++) {
        if ($i ~ /^success_rate_max=/)   { split($i,a,"="); sr=a[2]+0 }
        if ($i ~ /^n_passed_gates_max=/) { split($i,b,"="); npg=b[2]+0 }
      }
      if (sr < 0.005 && npg < 0.1) print "FAILED"; else print "FLEW";
    }' ${TRACE_LOG})
  echo "FLIGHTCHECK_VERDICT[${STAGE}]=${VERDICT:-UNKNOWN}"
  if [ "${STAGE_RC}" != "0" ] || [ "${VERDICT}" != "FLEW" ]; then
    echo "🚩 EGO CURRICULUM STALLED at stage=${STAGE} (RC=${STAGE_RC}, verdict=${VERDICT:-UNKNOWN}) -- not advancing."
    echo "   REPORT the trace; single_gate must recover ~0.8 before promoting (validate-before-_COMMON)."
    break
  fi
done

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}
echo "VQ2_EGO_CURRICULUM_DONE"

# =====================================================================================================
# LAUNCH (Adroit login, after PRE-SYNC of the ego files + smoke GREEN):
#   one seed, full ego ladder:
#     sbatch --export=ALL,SEED=0,RUNTAG=ego_a .../rl/peregrine_vq2_ego.sbatch
#   the >=5-seed portfolio (DESIGN.md §10; launch as SEPARATE jobs so one seed's crash can't kill the
#   others):
#     for s in 0 1 2 3 4; do
#       sbatch --export=ALL,SEED=$s,RUNTAG=ego_s$s .../peregrine_vq2_ego.sbatch
#     done
# =====================================================================================================
