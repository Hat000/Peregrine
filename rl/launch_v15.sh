#!/bin/bash
# =====================================================================================================
# v1.5 = v1 (ffs0-warm + faithful_rate + vision_cadence + yaw-dither-from-birth + roll-recover) PLUS the
# SPEED-DISCIPLINE + YAW-QUIETNESS STRUCTURAL FIX (2026-07-18). Successor to launch_v1.sh. The v1 arms
# left two failure classes UNPRICED (REWARD-LEDGER forensic): (1) VELOCITY RUNAWAY -- speed is unpriced
# and discount-favored (+12-27 reward/s-saved vs a finite crash terminal), so no policy ever brakes; and
# (2) YAW OSCILLATION -- a ~6 Hz limit cycle from a bearing-nulling controller under a sign-flip-only
# penalty ~500x too small. v1.5 makes both STRUCTURAL, all knobs config-gated + default-OFF (a config
# without the new keys trains BYTE-IDENTICAL to v1):
#
#   A OVERSPEED EPISODE-ABORT  ++env.overspeed_abort_mps=12.0  -- fatal OOB-CLASS termination when GT ||v||
#     exceeds 12 m/s (GT legal in a termination; obs untouched). Caps the runaway variable (top speed) BY
#     CONSTRUCTION -- the no-spin-abort precedent -- forcing braking into the curriculum. Pays terminal_oob,
#     KEEPS banked gate progress (death pricing UNCHANGED). Cannot fire at spawn (ego spawns at rest).
#   B PROGRESS-CREDIT SATURATION ++env.rw_progress_vcap_mps=7.5  -- cap the POSITIVE per-step progress
#     credit at rw_progress*7.5*dt so the marginal credit for going faster than ~7.5 m/s along-track is
#     ZERO (a saturation, NOT a penalty -- slow flight loses nothing, backward progress priced as-is).
#     Removes the myopic speed-magnitude gradient with zero stall risk (pairs with A for the discount tail).
#   C YAW AMPLITUDE / DUTY  ++env.rw_yaw_duty (Q 0.15 / W 0.4) + ++env.yaw_duty_free_band=0.25  -- price the
#     SUSTAINED post-clamp yaw amplitude beyond a FREE band. The champion ffs0 flies quiet at cmd |yaw|
#     absmean 0.232, INSIDE the 0.25 band -> pays EXACTLY 0 (the yaw->gate-in-view->altitude coupling is
#     preserved). Only excess amplitude is priced.
#   D YAW JERK  ++env.rw_yaw_jerk=0.05  -- L1 price on |yaw_cmd_t - yaw_cmd_{t-1}| (post-clamp): the 6 Hz
#     bang-bang. A steady turn (delta~0) pays 0; a single re-point costs one step. Scale-linear in the flip
#     amplitude (distinct from the SQUARED yaw_dither, which under-prices the moderate-amplitude flip-train).
#   E EVAL OBSERVABILITY (append-only): YAW_EVAL gains satur_duty= (fraction of eval steps at |cmd_yaw| >
#     0.95*clamp); DET_EVAL gains max_speed= (max GT episode-peak speed). New reward terms log to TB
#     (env_loss/yaw_duty_pen, /yaw_jerk_pen, /prog_sat_forfeit, /overspeed_abort_rate).
#
# WHAT CARRIES OVER FROM v1 (VERBATIM): warm from vpeffs0 (ffs0) on stage `dual_gate_fullstack_floor_pef`,
# course_n_gates=8, clamp 0.7, ticks_hi=1, blur off, spin-abort anneal, logstd reset, faithful_rate,
# ego_vision_cadence (30 Hz x p=0.35 seed 20260715+SEED), noise anneal, critic_grad_clip=10,
# ckpt_select_metric=n_passed_gates, eval_det_steps=600, eval_yaw_log=true, UPD 18000, save_freq 500. algo
# (appo) + gamma (0.9975) are set by the stage/sbatch and are UNTOUCHED here.
#
# CHANGES vs v1:
#   * yaw_dither RAISED 0.125 -> 0.4 (Q) / 0.6 (W): the v1 flip-tax was too weak; v1.5 combines a stronger
#     jerk-square (dither) with the NEW amplitude (duty) + L1-jerk terms.
#   * roll_recover=0.5 is now in the CORE of EVERY arm (v1 had it only on arm R) -- the accumulation ceiling
#     lever, on-from-birth (warm from the already-smooth ffs0; dither-from-birth philosophy), default
#     theta0=30deg. (v1 arm R annealed it in; v1.5 holds discipline from step 0 -- see NOTES.)
#
# ARMS (each a SEPARATE SLURM job so one seed's crash can't kill the others):
#   Q  (primary), seeds 0,1 : CORE (A+B+C@0.15+D + yaw_dither 0.4 + roll_recover 0.5). TAG=v15Q_s{0,1}.
#   W  (strong-quiet), seed 0 : CORE but rw_yaw_duty=0.4 + rw_yaw_dither=0.6 (heavier yaw pricing).
#                               TAG=v15W_s0.
#
# ---------------------------------------------------------------------------------------------------
# PHASE 1 (smoke):  bash launch_v15.sh
#     submits ONE smoke (v15smoke_s0, UPD=100, CORE=Q wiring). Prints its jobid + the ADJUDICATION
#     CHECKLIST + the exact Phase-2 command.
# ADJUDICATE (when the smoke COMPLETES -- a DONE log, not live; the absent-hook != inert footgun L16):
#     RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_v15smoke_s0
#     1. grep -n 'overspeed_abort_mps'  $RUNDIR/.hydra/config.yaml   # expect 12.0 (env:)
#     2. grep -n 'rw_progress_vcap_mps' $RUNDIR/.hydra/config.yaml   # expect 7.5  (env:)
#     3. grep -n 'rw_yaw_duty'          $RUNDIR/.hydra/config.yaml   # expect 0.15 (env:); yaw_duty_free_band 0.25
#     4. grep -n 'rw_yaw_jerk'          $RUNDIR/.hydra/config.yaml   # expect 0.05 (env:)
#     5. faithful_rate + ego_vision_cadence == true (the v1 fixes still bound).
#     6. python scratchpad/tb_parse.py <event-file>  # env_loss/yaw_duty_pen, /yaw_jerk_pen, /prog_sat_forfeit,
#        /overspeed_abort_rate ALL PRESENT (the wiring watchdog).
#     7. EGO_PRECHECK_RC=0 in the .out, no NaN.
#     8. FREE-BAND CALIBRATION PROOF (frozen-ffs0 YAW_EVAL -- run rollout_only on the WARM ckpt, or read the
#        smoke's first eval): signflips_per_s ~2.45 AND satur_duty ~0 AND env_loss/yaw_duty_pen ~0 -- i.e.
#        the champion's quiet 0.232-absmean yaw is INSIDE the 0.25 free band and pays ZERO duty. If duty_pen
#        is materially > 0 on frozen ffs0, the band is mis-calibrated -- STOP and re-check before the arms.
# PHASE 2 (arms):  MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v15.sh
#     submits Q(0,1) + W(0), each --dependency=afterok:<smoke_jobid>. ESCAPE HATCH: SMOKE_JID=none (or
#     SKIP_SMOKE=1) submits the arms with NO dependency.
# CLUSTER MECHANICS (overridable): GRES (default gpu:nvidia_a100:1 -- generic so MIG/V100 work),
#     SBATCH_EXCLUDE (default adroit-h11g3), SBATCH_TIMELIMIT (optional --time injection; else the sbatch
#     20h default). SBATCH_TIMELIMIT is the clean job-chain walltime lever (QOS auto-maps otherwise).
# =====================================================================================================
set -euo pipefail
STAGE=dual_gate_fullstack_floor_pef
MODE=${MODE:-smoke}                    # smoke (phase 1, default) | arms (phase 2)
UPD=${UPD:-18000}                      # v1 budget (unchanged)
SAVEFREQ=${SAVEFREQ:-500}
NGATES=${NGATES:-8}

# --- anti-dither (v1.5 RAISED from v1's 0.125; DITHER-FROM-BIRTH). Q/CORE = 0.4; W = 0.6 (below). ---
YAWDITHER=${YAWDITHER:-0.4}            # CORE/Q yaw_dither (v1 was 0.125)
YDSTART=${YDSTART:-1.0}                # 1.0 = ON-FROM-BIRTH (warm from already-smooth ffs0)
YDHOLDFRAC=${YDHOLDFRAC:-0.3}          # end-hold at full (inert when start=1.0; kept for parity)

# --- v1.5 YAW AMPLITUDE/DUTY (C) + free band. Q/CORE = 0.15; W = 0.4 (below). ---
YAWDUTY=${YAWDUTY:-0.15}               # CORE/Q rw_yaw_duty
FREEBAND=${FREEBAND:-0.25}             # yaw_duty_free_band; champion 0.232-absmean is INSIDE -> pays 0

# --- v1.5 shared anti-runaway (A,B) + yaw jerk (D) + roll-recover (now CORE, all arms) ---
OVERSPEED=${OVERSPEED:-12.0}           # overspeed_abort_mps (A); fatal OOB-class above this GT speed
PROGVCAP=${PROGVCAP:-7.5}              # rw_progress_vcap_mps (B); saturate positive progress above this m/s
YAWJERK=${YAWJERK:-0.05}               # rw_yaw_jerk (D); L1 |delta yaw_cmd| price
RECOVER=${RECOVER:-0.5}               # rw_roll_recover (CORE; on-from-birth, default theta0=30deg)

# --- vision cadence sub-knobs (v1 SSOT-correct defaults; net valid fix ~10.5 Hz) ---
VCFRAMEHZ=${VCFRAMEHZ:-30.0}
VCDETECTP=${VCDETECTP:-0.35}

# --- which arms + seeds ---
ARMS=${ARMS:-"Q W"}
SEEDS_Q=${SEEDS_Q:-"0 1"}
SEEDS_W=${SEEDS_W:-"0"}
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_vpeffs0/checkpoints}
SMOKE_JID=${SMOKE_JID:-}               # phase-2: the phase-1 smoke jobid to afterok-gate on
SKIP_SMOKE=${SKIP_SMOKE:-0}            # phase-2 escape hatch: 1 => no dependency

# --- cluster mechanics (generic gres header; SBATCH_EXCLUDE default; optional SBATCH_TIMELIMIT injection) ---
GRES=${GRES:-gpu:nvidia_a100:1}        # generic -> override for MIG/V100 (fail-loud OK)
EXCLUDE=${SBATCH_EXCLUDE:-adroit-h11g3}
TIMEOPT=""
if [ -n "${SBATCH_TIMELIMIT:-}" ]; then TIMEOPT="--time=${SBATCH_TIMELIMIT}"; fi
SBATCH_FILE="${SBATCH_FILE:-$(cd "$(dirname "$0")" && pwd)/peregrine_vq2_ego.sbatch}"

# ---- vpeffs0 base recipe (VERBATIM from v1) ----
BASE_VPEFFS0="++env.ego_blur_gate=false ++env.ego_yaw_cmd_clamp_rad_s=0.7 ++env.spin_abort_anneal=true \
++env.spin_abort_scale_start=2.6 ++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 \
++warmstart_reset_logstd=true ++warmstart_reset_logstd_std=0.18"

# ---- vpef8nc 8-gate/reward/infra block (VERBATIM from v1) ----
VPEF8NC="++env.course_n_gates=${NGATES} ++env.rw_v_cap=0 \
++env.rw_cross_zero_m=0.75 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_det_steps=600 ++eval_yaw_log=true \
++save_freq=${SAVEFREQ}"

# ---- THE TWO v1 DEPLOY-FAITHFULNESS ENABLES (VERBATIM) ----
FAITHFUL_SHARED="++dynamics.faithful_rate=true ++env.ego_vision_cadence=true \
++env.ego_vision_frame_hz=${VCFRAMEHZ} ++env.ego_vision_detect_p=${VCDETECTP}"

# ---- THE v1.5 SHARED SPEED-DISCIPLINE + YAW-QUIETNESS block (A,B,D + roll-recover CORE). rw_yaw_duty +
#      rw_yaw_dither are per-arm (built in extra_for). recovery term uses the default theta0 (30deg); it is
#      ON-FROM-BIRTH (no recovery_anneal) -- the dither-from-birth philosophy for the smooth ffs0 warm. ----
V15_SHARED="++env.overspeed_abort_mps=${OVERSPEED} ++env.rw_progress_vcap_mps=${PROGVCAP} \
++env.rw_yaw_jerk=${YAWJERK} ++env.yaw_duty_free_band=${FREEBAND} ++env.rw_roll_recover=${RECOVER}"

export STAGES="${STAGE}"
export PRECHECK=1
export "UPD_${STAGE}=${UPD}"

submit () {   # $1=RUNTAG  $2=SEED  $3=EXTRA  $4=n_updates-override(optional, smoke)  $5=dep(optional)
  local TAG="$1" SEED="$2" EX="$3" NUPD="${4:-}" DEP="${5:-}"
  export SEED
  export RUNTAG="${TAG}"
  export EXTRA="${EX}"
  if [ -n "${NUPD}" ]; then export "UPD_${STAGE}=${NUPD}"; else export "UPD_${STAGE}=${UPD}"; fi
  echo "=== submit ${TAG} SEED=${SEED} STAGE=${STAGE} UPD=${NUPD:-${UPD}} ngates=${NGATES} gres=${GRES}${DEP:+ dep=afterok:${DEP}} ==="
  if [ -n "${DEP}" ]; then
    sbatch --dependency="afterok:${DEP}" --gres="${GRES}" --exclude="${EXCLUDE}" ${TIMEOPT} --export=ALL "${SBATCH_FILE}"
  else
    sbatch --gres="${GRES}" --exclude="${EXCLUDE}" ${TIMEOPT} --export=ALL "${SBATCH_FILE}"
  fi
}

# per-arm EXTRA builder: yaw_dither + yaw_duty are per-arm (Q vs W); everything else shared. Offsets the
# vision-cadence RNG seed by the training seed (each seed sees a DIFFERENT reproducible detector-miss draw).
extra_for () {   # $1=SEED  $2=yaw_dither  $3=yaw_duty
  local SEED="$1" YD="$2" YDU="$3"
  local ANTIDITHER="++env.rw_yaw_dither=${YD} ++env.yaw_dither_anneal=true \
++env.yaw_dither_start=${YDSTART} ++env.yaw_dither_hold_frac=${YDHOLDFRAC}"
  echo "${BASE_VPEFFS0} ${VPEF8NC} ${ANTIDITHER} ${FAITHFUL_SHARED} ${V15_SHARED} \
++env.rw_yaw_duty=${YDU} ++env.ego_vision_cadence_seed=$((20260715 + SEED)) +init_from=${WARM}"
}

if [ "${MODE}" = "smoke" ]; then
  # -------- PHASE 1: ONE smoke, CORE=Q wiring, UPD=100, seed 0 --------
  submit "v15smoke_s0" 0 "$(extra_for 0 "${YAWDITHER}" "${YAWDUTY}")" 100
  cat <<'EOF'

=== v1.5 SMOKE submitted (UPD=100, CORE=Q wiring: A overspeed + B prog-saturate + C duty@0.15 + D jerk +
    yaw_dither 0.4 + roll_recover 0.5, on the v1 faithful_rate+vision_cadence base).
    ADJUDICATE the COMPLETED smoke (a DONE log, never a live one -- the absent-hook != inert footgun L16):
      RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_v15smoke_s0
      1. grep -nE 'overspeed_abort_mps|rw_progress_vcap_mps|rw_yaw_duty|yaw_duty_free_band|rw_yaw_jerk|rw_roll_recover' \
           $RUNDIR/.hydra/config.yaml   # expect 12.0 / 7.5 / 0.15 / 0.25 / 0.05 / 0.5 (all under env:)
      2. grep -nE 'faithful_rate|ego_vision_cadence' $RUNDIR/.hydra/config.yaml   # both true (v1 fixes hold)
      3. python scratchpad/tb_parse.py <event-file>   # env_loss/yaw_duty_pen, /yaw_jerk_pen,
           # /prog_sat_forfeit, /overspeed_abort_rate ALL PRESENT (wiring watchdog)
      4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v15smoke_s0.out  # == 0, no NaN
      5. FREE-BAND CALIBRATION PROOF -- frozen-ffs0 YAW_EVAL (rollout_only on the WARM ckpt, or the smoke's
         first eval): signflips_per_s ~2.45 AND satur_duty ~0 AND env_loss/yaw_duty_pen ~0. duty_pen
         materially > 0 on frozen ffs0 == the 0.25 band is mis-set -> STOP.
    THEN release the real arms (afterok the smoke):
      MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v15.sh
    ESCAPE HATCH (no dependency): MODE=arms SMOKE_JID=none bash launch_v15.sh
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
    Q) for s in ${SEEDS_Q}; do submit "v15Q_s${s}" "${s}" "$(extra_for "${s}" "${YAWDITHER}" "${YAWDUTY}")" "" "${DEP}"; done ;;
    W) for s in ${SEEDS_W}; do submit "v15W_s${s}" "${s}" "$(extra_for "${s}" "0.6" "0.4")"                 "" "${DEP}"; done ;;
    *) echo "!! unknown ARM '${ARM}' (expected Q / W) -- skipping" ;;
  esac
done

cat <<'EOF'

=== v1.5 arms submitted [Q W]. DEPLOY-PICK = the HARD JOINT yaw-gate selection (post-hoc), now JOINTLY
    gated on SPEED DISCIPLINE too:
    STEP 1 (yaw+speed sweep): for EACH seed, run ++rollout_only over the run's periodic + best_npg snapshots
      -> YAW_EVAL (signflips_per_s + satur_duty) + DET_EVAL (n_passed_gates + max_speed) per ckpt.
    STEP 2 (HARD gates): DISCARD any ckpt with signflips_per_s > ~4.0 OR satur_duty > ~0.1 (a yaw-hunter /
      rail-rider -- never fly it), REGARDLESS of n_passed. Confirm max_speed sits BELOW the 12 m/s abort with
      margin (a run pinned at the abort is riding the edge -- prefer a lower-max_speed survivor).
    STEP 3 (rank survivors): HIGHEST DET n_passed_gates; tiebreak LOWEST signflips + satur_duty + roll_swing.
      NEVER select on value. GOAL: match vpeffs0's threading on the FULL 8-gate course with signflips ~0,
      satur_duty ~0, NO velocity runaway (max_speed well under 12), and zero yaw oscillation.
    MONITOR: scratchpad/tb_parse.py on each run's event file (TB, NOT stdout). Watch env_loss/yaw_duty_pen +
      /yaw_jerk_pen + /prog_sat_forfeit converge DOWN, /overspeed_abort_rate -> ~0 (policy learns to brake),
      n_passed_gates climb, roll_recover_pen shrink.
EOF
