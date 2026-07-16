#!/bin/bash
# =====================================================================================================
# v1 RETRAIN = huntfix (vpef8nc + anti-dither) + THE TWO DEPLOY-FAITHFULNESS FIXES from the plant+vision
# sysID (2026-07-16). Successor to launch_huntfix.sh. GOAL: close the sim-to-sim deploy gap that made
# vpef8nc oscillate + hit the g2->g3 wall, by making the TRAINING sim match the real VQ2:
#
#   FIX 1 (PLANT)  ++dynamics.faithful_rate=true   -- DiffAero ran a FLAT cmd->rate gain (~2.5x const);
#     real VQ2 is EXPANSIVE (gain rises 2.36->2.89x with |cmd|). The policy trained on the flat plant
#     OVER-ROTATES most at the aggressive 50-61deg g2->g3 banks = the wall + yaw overshoot. faithful_rate
#     installs the sysID-fit small-signal rate_gain=[2.359,2.363,2.163] + per-axis super_rate_s=
#     [0.296,0.284,0.316] + alpha_max=[260,260,80] (torch-parity bit-exact; 17 tests; rate_sign UNCHANGED).
#   FIX 2 (VISION) ++env.ego_vision_cadence=true   -- training fed a FRESH gate fix EVERY visible tick;
#     deploy gets valid detections only ~7-15 Hz (30 Hz frames x detector-miss) + goes dark <4.7 m. The
#     cadence gate throttles the fresh-fix mask to a 30 Hz frame clock x per-frame detect draw (p=0.35 ->
#     ~10.5 Hz valid), REUSING the estimator's existing ego-propagation + confidence-decay for the gaps
#     (NOT a new output-delay layer). Policy learns to COMMIT + coast the blind final approach instead of
#     yaw-hunting a stale/absent gate. 13 tests; sub-knobs frame_hz/detect_p/seed; default OFF byte-id.
#
# WHAT ELSE CARRIES OVER FROM huntfix (unchanged): warm from vpeffs0 (ffs0, the smooth 2-gate champion) on
# ITS OWN stage `dual_gate_fullstack_floor_pef`, course_n_gates=8, NO att-cap / NO velocity cap / NO roll
# fence (Fengyou: fence ONLY the axis the course doesn't use -- roll/speed/vel-cap = NO), and aw1's
# annealed yaw-dither discipline calibrated to the 0.7 clamp (rw_yaw_dither=0.125; flip-tax invariant
# 0.125*(2*0.7)^2 = 0.245).
#
# DESIGN-REVIEW FOLD-INS (vs huntfix):
#   * UPD 12000 -> 18000 (vpef8nc TB: n_passed/success/collision/exit_frame ALL still improving at 12k =
#     under-trained). Per-stage via UPD_<stage>; set here to 18000.
#   * yaw_dither_start 0.0 -> 1.0 = DITHER-FROM-BIRTH. huntfix ramped it in (aw1's late-anneal, for a
#     warm base that still had the hunt). We warm from the ALREADY-smooth ffs0, so hold the smoothness
#     from step 0 -- the 8-gate fine-tune can't reintroduce the rail-flip if the tax is live from birth.
#   * ONE arm -> _recover: the recovery/damping ROLL reward (rw_roll_recover, form A, bearing-weighted,
#     annealed) -- the only directive-compliant tool for ffs0's REAL ceiling = SPEED ACCUMULATION / over-
#     bank at gate 4-5 (anti-dither taxes yaw JITTER, it does NOT touch the roll-runaway). Pure penalty,
#     un-farmable, w(theta)=exp(-(theta/30deg)^2) -> FREE to bank into a genuine turn (gate off-bearing),
#     bites ONLY when banking while the gate is dead-ahead (= the over-bank-when-you-should-be-level death).
#   * SMOKE GATE (--dependency=afterok): TWO-PHASE. Phase 1 submits ONE UPD=100 smoke; you ADJUDICATE its
#     completed log (the absent-hook != inert footgun -- read a DONE log, never a live one); Phase 2
#     releases the real arms afterok the smoke jobid. Each arm ALSO runs its own PRECHECK (3 updates)
#     before its real budget -> arm-specific wiring (recovery / dr_latency) is self-guarded.
#   * HARD JOINT yaw-gate ckpt-selection = a POST-HOC DEPLOY-PICK policy (the in-training promotion is
#     n_passed-only; there is no joint-metric flag). eval_yaw_log=true emits YAW_EVAL signflips_per_s on
#     final + promoted-best; the deploy pick REJECTS any ckpt > ~4 flips/s REGARDLESS of n_passed, then
#     takes highest DET n_passed among survivors. See the closing SELECT block + POST-HOC YAW SWEEP note.
#
# ARMS (each a SEPARATE SLURM job so one seed's crash can't kill the others):
#   A  (primary), seeds 0,1 : ffs0-warm + 8-gate + faithful_rate + vision_cadence + yaw-dither-from-birth.
#   R  (recover), seed 0    : arm A + rw_roll_recover (annealed, form A, theta0=30deg). The accumulation
#                             ceiling probe. TAG=v1R_s0.
#   C  (latency), seed 0    : arm A + ++dynamics.dr_latency_max_steps=5 (extend COMMAND-latency DR to the
#                             deploy ~100 ms / 5-step tail; a DIFFERENT channel from vision cadence). C is
#                             cheap hardening; drop via ARMS="A R" if the A100 pool is tight.
#
# ---------------------------------------------------------------------------------------------------
# PHASE 1 (smoke):  bash launch_v1.sh
#     submits ONE smoke (v1smoke_s0, UPD=100, arm-A wiring = the 2 shared enables + yaw-dither). Prints
#     its jobid + the ADJUDICATION CHECKLIST + the exact Phase-2 command.
# ADJUDICATE (when the smoke COMPLETES -- a DONE log, not live):
#     1. faithful_rate BOUND : grep 'faithful_rate' <rundir>/.hydra/config.yaml  -> `faithful_rate: true`
#        under dynamics: (a force-added ++ key resolves INTO the tree; absent => it silently no-op'd).
#     2. vision_cadence BOUND: grep 'ego_vision_cadence' <rundir>/.hydra/config.yaml -> `true` under env:.
#     3. yaw_dither LIVE     : scratchpad/tb_parse.py on the event file -> env_loss/yaw_dither_pen present
#        and NONZERO (the watchdog; from birth since start=1.0).
#     4. PRECHECK clean      : EGO_PRECHECK_RC=0 in the .out, no NaN.
# PHASE 2 (arms):  MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v1.sh
#     submits A(0,1) + R(0) + C(0), each --dependency=afterok:<smoke_jobid>. If the smoke already finished
#     OK they release immediately; if still running they hold. ESCAPE HATCH: SMOKE_JID=none (or
#     SKIP_SMOKE=1) submits the arms with NO dependency.
# =====================================================================================================
set -euo pipefail
STAGE=dual_gate_fullstack_floor_pef
MODE=${MODE:-smoke}                    # smoke (phase 1, default) | arms (phase 2)
UPD=${UPD:-18000}                      # design-review bump 12k->18k (vpef8nc under-trained at 12k)
SAVEFREQ=${SAVEFREQ:-500}
NGATES=${NGATES:-8}

# --- anti-dither (aw1 discipline; MATCHED to the 0.7 clamp -> target 0.125; DITHER-FROM-BIRTH) ---
YAWDITHER=${YAWDITHER:-0.125}          # matched anti-dither target for clamp 0.7 (flip-tax 0.245)
YDSTART=${YDSTART:-1.0}                # 1.0 = ON-FROM-BIRTH (warm from already-smooth ffs0); huntfix used 0.0
YDHOLDFRAC=${YDHOLDFRAC:-0.3}          # end-hold at full for the last 30% (inert when start=1.0; kept for parity)

# --- vision cadence sub-knobs (SSOT-correct defaults; net valid fix ~10.5 Hz in the 7-15 band) ---
VCFRAMEHZ=${VCFRAMEHZ:-30.0}           # recorded stream rate (the 424-flight number)
VCDETECTP=${VCDETECTP:-0.35}           # per-frame detector-success prob -> 30*0.35 ~ 10.5 Hz valid fixes

# --- arm R recovery/damping roll reward (form A, dense bearing-weighted; annealed like vpefcap) ---
RECOVER=${RECOVER:-0.5}                # rw_roll_recover weight for arm R (>0 = ON; annealed from 0)
XLEVEL=${XLEVEL:-0}                    # rw_cross_level (form B, sparse per-crossing); 0 = OFF (form A only)

# --- arm C control-latency DR tail (deploy ~100 ms / 5-step); min stays 1 (BASE default) ---
DRLATMAX=${DRLATMAX:-5}

# --- which arms + seeds ---
ARMS=${ARMS:-"A R C"}
SEEDS_A=${SEEDS_A:-"0 1"}
SEEDS_R=${SEEDS_R:-"0"}
SEEDS_C=${SEEDS_C:-"0"}
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_vpeffs0/checkpoints}
SMOKE_JID=${SMOKE_JID:-}               # phase-2: the phase-1 smoke jobid to afterok-gate on
SKIP_SMOKE=${SKIP_SMOKE:-0}            # phase-2 escape hatch: 1 => no dependency
SBATCH_FILE="$(cd "$(dirname "$0")" && pwd)/peregrine_vq2_ego.sbatch"

# ---- vpeffs0 base recipe (its proven EXTRA: yaw clamp 0.7, ticks_hi=1, blur off, spin-abort anneal,
#      logstd reset) -- VERBATIM from launch_huntfix.sh's BASE_VPEFFS0 ----
BASE_VPEFFS0="++env.ego_blur_gate=false ++env.ego_yaw_cmd_clamp_rad_s=0.7 ++env.spin_abort_anneal=true \
++env.spin_abort_scale_start=2.6 ++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 \
++warmstart_reset_logstd=true ++warmstart_reset_logstd_std=0.18"

# ---- vpef8nc 8-gate/reward/infra block (= huntfix): 8 gates, NO velocity cap, cross_zero fix, noise
#      anneal, critic clip, n_passed ckpt-select, eval_yaw_log for the YAW_EVAL selection metric ----
VPEF8NC="++env.course_n_gates=${NGATES} ++env.rw_v_cap=0 \
++env.rw_cross_zero_m=0.75 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_det_steps=600 ++eval_yaw_log=true \
++save_freq=${SAVEFREQ}"

# ---- anti-dither (aw1 keys; the _pef stage renders NONE of these, so all four are new ++env.* adds) ----
ANTIDITHER="++env.rw_yaw_dither=${YAWDITHER} ++env.yaw_dither_anneal=true \
++env.yaw_dither_start=${YDSTART} ++env.yaw_dither_hold_frac=${YDHOLDFRAC}"

# ---- THE TWO NEW DEPLOY-FAITHFULNESS ENABLES (plant + vision sysID, 2026-07-16). faithful_rate is on
#      the dynamics node; vision_cadence + its sub-knobs are on the env node. The cadence seed is offset
#      per training seed (below) so each seed sees a DIFFERENT (reproducible) detector-miss realization. ----
FAITHFUL_SHARED="++dynamics.faithful_rate=true ++env.ego_vision_cadence=true \
++env.ego_vision_frame_hz=${VCFRAMEHZ} ++env.ego_vision_detect_p=${VCDETECTP}"

export STAGES="${STAGE}"
export PRECHECK=1
export "UPD_${STAGE}=${UPD}"

submit () {   # $1=RUNTAG  $2=SEED  $3=EXTRA  $4=n_updates-override(optional, smoke)  $5=dep(optional)
  local TAG="$1" SEED="$2" EX="$3" NUPD="${4:-}" DEP="${5:-}"
  export SEED
  export RUNTAG="${TAG}"
  export EXTRA="${EX}"
  if [ -n "${NUPD}" ]; then export "UPD_${STAGE}=${NUPD}"; else export "UPD_${STAGE}=${UPD}"; fi
  echo "=== submit ${TAG} SEED=${SEED} STAGE=${STAGE} UPD=${NUPD:-${UPD}} ngates=${NGATES}${DEP:+ dep=afterok:${DEP}} ==="
  if [ -n "${DEP}" ]; then
    sbatch --dependency="afterok:${DEP}" --gres=gpu:nvidia_a100:1 --export=ALL "${SBATCH_FILE}"
  else
    sbatch --gres=gpu:nvidia_a100:1 --export=ALL "${SBATCH_FILE}"
  fi
}

# per-seed EXTRA builder (offsets the vision-cadence RNG seed by the training seed)
extra_for () {   # $1=SEED  $2=ARMEXTRA
  local SEED="$1" ARMEXTRA="$2"
  echo "${BASE_VPEFFS0} ${VPEF8NC} ${ANTIDITHER} ${FAITHFUL_SHARED} \
++env.ego_vision_cadence_seed=$((20260715 + SEED)) ${ARMEXTRA} +init_from=${WARM}"
}

RECOVER_EXTRA="++env.rw_roll_recover=${RECOVER} ++env.rw_roll_recover_theta0_rad=0.5235988 \
++env.rw_cross_level=${XLEVEL} ++env.recovery_anneal=true ++env.recovery_start=0.0 ++env.recovery_hold_frac=0.3"
CLAT_EXTRA="++dynamics.dr_latency_max_steps=${DRLATMAX}"

if [ "${MODE}" = "smoke" ]; then
  # -------- PHASE 1: ONE smoke, arm-A wiring, UPD=100, seed 0 --------
  submit "v1smoke_s0" 0 "$(extra_for 0 "")" 100
  cat <<'EOF'

=== v1 SMOKE submitted (UPD=100, arm-A wiring: faithful_rate + vision_cadence + yaw-dither-from-birth).
    ADJUDICATE the COMPLETED smoke (a DONE log, never a live one -- the absent-hook != inert footgun):
      RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_v1smoke_s0
      1. grep -n 'faithful_rate'      $RUNDIR/.hydra/config.yaml   # expect `faithful_rate: true` (dynamics:)
      2. grep -n 'ego_vision_cadence' $RUNDIR/.hydra/config.yaml   # expect `ego_vision_cadence: true` (env:)
      3. python scratchpad/tb_parse.py <event-file>  # env_loss/yaw_dither_pen PRESENT + NONZERO
      4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v1smoke_s0.out  # == 0, no NaN
    THEN release the real arms (afterok the smoke):
      MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v1.sh
    ESCAPE HATCH (no dependency): MODE=arms SMOKE_JID=none bash launch_v1.sh
EOF
  exit 0
fi

# -------- PHASE 2: the real arms, afterok the smoke --------
DEP=""
if [ "${SKIP_SMOKE}" != "1" ] && [ "${SMOKE_JID}" != "none" ] && [ -n "${SMOKE_JID}" ]; then
  DEP="${SMOKE_JID}"
elif [ "${SKIP_SMOKE}" = "1" ] || [ "${SMOKE_JID}" = "none" ]; then
  echo "!! MODE=arms with NO smoke dependency (escape hatch) -- each arm's own PRECHECK is the only wiring guard."
else
  echo "!! MODE=arms but SMOKE_JID unset. Pass SMOKE_JID=<jobid> (afterok gate) or SMOKE_JID=none (escape). Aborting." >&2
  exit 2
fi

for ARM in ${ARMS}; do
  case "${ARM}" in
    A) for s in ${SEEDS_A}; do submit "v1A_s${s}" "${s}" "$(extra_for "${s}" "")"              "" "${DEP}"; done ;;
    R) for s in ${SEEDS_R}; do submit "v1R_s${s}" "${s}" "$(extra_for "${s}" "${RECOVER_EXTRA}")" "" "${DEP}"; done ;;
    C) for s in ${SEEDS_C}; do submit "v1C_s${s}" "${s}" "$(extra_for "${s}" "${CLAT_EXTRA}")"    "" "${DEP}"; done ;;
    *) echo "!! unknown ARM '${ARM}' (expected A / R / C) -- skipping" ;;
  esac
done

cat <<'EOF'

=== v1 arms submitted [A R C]. DEPLOY-PICK = the HARD JOINT yaw-gate selection (post-hoc):
    STEP 1 (yaw sweep): for EACH seed, run the ++rollout_only harness over the run's periodic + best_npg/
      snapshots to get YAW_EVAL signflips_per_s per ckpt (final + promoted-best already print it via
      eval_yaw_log=true). STEP 2 (HARD gate): DISCARD every ckpt with signflips_per_s > ~4.0 REGARDLESS
      of n_passed_gates (a high-flip ckpt is a yaw-hunter -- never fly it). STEP 3 (rank survivors): among
      the survivors, pick HIGHEST DET n_passed_gates; tiebreak LOWEST signflips + lowest roll_swing.
      NEVER select on value. GOAL: n_passed >= vpef8nc's DET 5.33 with signflips driven toward ~0 AND the
      g2->g3 wall gone (the faithful-rate payoff). Arm R additionally reports whether the recovery term
      lifted late-gate n_passed (the accumulation ceiling) without costing yaw smoothness.
    MONITOR: scratchpad/tb_parse.py on each run's event file (TB, NOT stdout). Watch env_loss/
      yaw_dither_pen converge DOWN + n_passed_gates climb; env_loss/roll_recover_pen (arm R) should
      shrink as it learns to level when lined up.
EOF
