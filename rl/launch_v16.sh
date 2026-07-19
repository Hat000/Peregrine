#!/bin/bash
# =====================================================================================================
# v1.6 = v1.5 (overspeed-abort + progress-saturate + yaw duty/band/jerk + dither + roll-recover anneal,
# ALL wire-proven) PLUS four config-gated v1.6 mechanisms + a perception SPLIT, warm-started from the
# WIRE-PROVEN v1.5 W-arm temperament (the quiet fallback that CLOSED yaw+runaway on the wire 2026-07-19).
# The v1.5 flight (20 sessions) left FOUR classes to close: (1) a post-pass detect BLACKOUT 0.7-5.8 s on
# descend legs, (2) a suspected CAMERA MOUNT parity gap, (3) honest gate ACQUISITION, (4) the Qs1 near-gate
# PITCH vibration. v1.6 makes each STRUCTURAL, all knobs config-gated + default-OFF (a config without the
# new keys trains BYTE-IDENTICAL to v1.5):
#
#   1 CAMERA MOUNT (ABSOLUTE) ++env.ego_cam_mount_pitch_deg=20.0 -- re-aims the emulated camera's NET
#     optical-axis elevation. DISCOVERED CODE REALITY: R_camera_from_body ALREADY bakes the wire's +20 deg
#     mount (probe-verified; tests/test_gate_visibility.py pins the resulting up/down asymmetry), so this
#     20.0 is a ZERO-DELTA re-statement == byte-identical, NOT a second +20. The knob is the experimental
#     lever if the true mount later proves != 20 deg. "Mount parity" was ALREADY satisfied -- see the report.
#   2 RANGE DETECTION ++env.ego_vision_detect_mode=range -- replaces the flat detector-success p with the
#     miner-calibrated p(r) (14,938 wire frames): near-gate blackout logistic -> mid plateau -> far
#     attenuation. Reproduces the descend-leg blackout by CONSTRUCTION. Cadence + FOV mask unchanged.
#     Sub-knobs r0=2.2 k=0.4 plateau=0.99 far_start=26 far_p=0.79 (defaults; overridable).
#   3 APERTURE-MARGIN ++env.pass_margin_anneal=true -- Fengyou's SHRINK-THE-GATE constraint: a geometric
#     pass whose WEIGHTED miss m=sqrt((2*|lat|)^2+|vert|^2) exceeds the annealed margin is reclassified a
#     MISS (existing rw_terminal_miss; NO new penalty). margin anneals from the full aperture (loose, no-op)
#     DOWN to 1.0 m over the front, END-HOLD. End envelope ~ lat<=0.5 / vert<=1.0 m (miner: lateral risk 2x).
#   4 PITCH QUIETNESS ++env.rw_pitch_jerk=0.03 (shared) + ++env.rw_pitch_duty (Q 0.1 / W 0.2) + the WIDE
#     ++env.pitch_duty_free_band=0.6 -- the yaw duty/jerk mirror on the applied pitch command, against the
#     Qs1 pitch limit cycle. The band is WIDE on purpose: pitch is the PRIMARY axis, only chatter above it
#     pays (a tight band reproduces the pitch-limit-1 crawl).
#   + PERCEPTION SPLIT (stage dual_gate_fullstack_floor_pef16): rw_perception 0.02 -> 0.014 +
#     rw_perception_next 0.004 (sum 0.018 < rw_time 0.02 -> farm-neutral) -- a NEXT-gate acquisition cue.
#   E EVAL OBSERVABILITY (append-only): a PITCH_EVAL[...] line (signflips_per_s + cmd_absmean + satur_duty)
#     parallel to YAW_EVAL; new reward terms log to TB (env_loss/pitch_duty_pen, /pitch_jerk_pen).
#
# WHAT CARRIES OVER FROM v1.5 (VERBATIM): the ENTIRE V15_SHARED block (overspeed_abort 12.0,
# rw_progress_vcap_mps 7.5 -- explicitly NOT 5.0, yaw_jerk, yaw_duty_free_band, roll_recover + recovery
# anneal), the BASE_VPEFFS0 recipe (blur off, yaw clamp 0.7, spin-abort anneal, ticks_hi=1, logstd reset),
# VPEF8NC (8-gate, noise anneal, critic_grad_clip=10, ckpt_select_metric=n_passed_gates, eval_det_steps=600,
# eval_yaw_log=true, UPD 18000, save_freq 500), faithful_rate + ego_vision_cadence, per-arm yaw_dither/duty.
# algo (appo) + gamma (0.9975) are set by the stage/sbatch and are UNTOUCHED here. WARM = the v1.5 W arm.
#
# ARMS (each a SEPARATE SLURM job so one seed's crash can't kill the others):
#   Q  (primary), seeds 0,1 : CORE (v15 Q yaw duty 0.15 / dither 0.4 + roll_recover) + v16 (mount + range +
#                             margin + pitch_jerk 0.03 + pitch_duty 0.1). TAG=v16Q_s{0,1}.
#   W  (strong-quiet), seed 0 : CORE but rw_yaw_duty=0.4 + rw_yaw_dither=0.6 + rw_pitch_duty=0.2 (heavier
#                             yaw + pitch pricing -- the wire-proven quiet temperament). TAG=v16W_s0.
#
# ---------------------------------------------------------------------------------------------------
# PHASE 1 (smoke):  bash launch_v16.sh
#     submits ONE smoke (v16smoke_s0, UPD=100, CORE=Q wiring). Prints its jobid + the ADJUDICATION
#     CHECKLIST + the exact Phase-2 command.
# ADJUDICATE (when the smoke COMPLETES -- a DONE log, not live; the absent-hook != inert footgun L16):
#     RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v16smoke_s0
#     1. grep -nE 'ego_cam_mount_pitch_deg|ego_vision_detect_mode|pass_margin_anneal|pass_margin_final_m|rw_pitch_jerk|rw_pitch_duty|pitch_duty_free_band' \
#          $RUNDIR/.hydra/config.yaml   # 20.0 / range / true / 1.0 / 0.03 / 0.1 / 0.6 (all under env:)
#     2. grep -nE 'overspeed_abort_mps|rw_progress_vcap_mps|rw_yaw_duty|rw_yaw_jerk|rw_perception_next' \
#          $RUNDIR/.hydra/config.yaml   # 12.0 / 7.5 / 0.15 / 0.05 / 0.004 (v1.5 carryover + split hold)
#     3. grep -nE 'faithful_rate|ego_vision_cadence' $RUNDIR/.hydra/config.yaml   # both true
#     4. python scratchpad/tb_parse.py <event-file>   # env_loss/pitch_duty_pen, /pitch_jerk_pen,
#        # /yaw_duty_pen, /prog_sat_forfeit, /overspeed_abort_rate ALL PRESENT (wiring watchdog)
#     5. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v16smoke_s0.out  # == 0, no NaN
#     6. grep -E 'pass-margin-anneal|PITCH_EVAL' the .out: pass-margin-anneal ON (SHRINK start->1.0) fired;
#        a PITCH_EVAL[...] line printed alongside YAW_EVAL (parallel observability wired).
# PHASE 2 (arms):  MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v16.sh
#     submits Q(0,1) + W(0), each --dependency=afterok:<smoke_jobid>. ESCAPE HATCH: SMOKE_JID=none (or
#     SKIP_SMOKE=1) submits the arms with NO dependency.
# CLUSTER MECHANICS (overridable): GRES (default gpu:nvidia_a100:1), SBATCH_EXCLUDE (default adroit-h11g3),
#     SBATCH_TIMELIMIT (optional --time injection; else the sbatch 20h default).
# =====================================================================================================
set -euo pipefail
STAGE=dual_gate_fullstack_floor_pef16
MODE=${MODE:-smoke}                    # smoke (phase 1, default) | arms (phase 2)
UPD=${UPD:-18000}                      # v1.5 budget (unchanged)
SAVEFREQ=${SAVEFREQ:-500}
NGATES=${NGATES:-8}

# --- anti-dither (v1.5 values; DITHER-FROM-BIRTH). Q/CORE = 0.4; W = 0.6 (below). ---
YAWDITHER=${YAWDITHER:-0.4}            # CORE/Q yaw_dither
YDSTART=${YDSTART:-1.0}                # 1.0 = ON-FROM-BIRTH (warm from the already-smooth v1.5 W)
YDHOLDFRAC=${YDHOLDFRAC:-0.3}          # end-hold at full (inert when start=1.0; kept for parity)

# --- v1.5 YAW AMPLITUDE/DUTY + free band. Q/CORE = 0.15; W = 0.4 (below). ---
YAWDUTY=${YAWDUTY:-0.15}               # CORE/Q rw_yaw_duty
FREEBAND=${FREEBAND:-0.25}             # yaw_duty_free_band; champion 0.232-absmean is INSIDE -> pays 0

# --- v1.6 PITCH DUTY (per-arm below) + free band + jerk. Q pitch_duty = 0.1; W = 0.2. ---
PITCHDUTY=${PITCHDUTY:-0.1}            # CORE/Q rw_pitch_duty
PITCHBAND=${PITCHBAND:-0.6}            # pitch_duty_free_band -- WIDE (pitch is the primary axis)
PITCHJERK=${PITCHJERK:-0.03}           # rw_pitch_jerk (shared)

# --- v1.5 shared anti-runaway (A,B) + yaw jerk (D) + roll-recover (CORE, all arms) ---
OVERSPEED=${OVERSPEED:-12.0}           # overspeed_abort_mps (A); fatal OOB-class above this GT speed
PROGVCAP=${PROGVCAP:-7.5}              # rw_progress_vcap_mps (B); NOT 5.0 -- saturate positive progress above this m/s
YAWJERK=${YAWJERK:-0.05}               # rw_yaw_jerk (D); L1 |delta yaw_cmd| price
RECOVER=${RECOVER:-0.5}                # rw_roll_recover (CORE; on-from-birth, default theta0=30deg)

# --- v1.6 aperture-margin knobs (SHRINK-THE-GATE constraint) ---
PMFINAL=${PMFINAL:-1.0}                # pass_margin_final_m (end weighted-miss envelope)
PMLATW=${PMLATW:-2.0}                  # pass_margin_lat_weight (lateral risk ~2x vertical)
PMHOLD=${PMHOLD:-0.25}                 # pass_margin_hold_frac (END-HOLD tight for the last 25%)

# --- v1.6 range-detection knobs (miner-calibrated; defaults shown for greppability) ---
DETR0=${DETR0:-2.2}; DETK=${DETK:-0.4}; DETPLAT=${DETPLAT:-0.99}
DETFAR=${DETFAR:-26.0}; DETFARP=${DETFARP:-0.79}

# --- vision cadence sub-knobs (v1 SSOT-correct defaults; net valid fix ~10.5 Hz) ---
VCFRAMEHZ=${VCFRAMEHZ:-30.0}
VCDETECTP=${VCDETECTP:-0.35}

# --- which arms + seeds ---
ARMS=${ARMS:-"Q W"}
SEEDS_Q=${SEEDS_Q:-"0 1"}
SEEDS_W=${SEEDS_W:-"0"}
# WARM = the WIRE-PROVEN v1.5 W arm's periodic_prev (actor+critic; the quiet temperament that closed
# yaw+runaway on the wire). NOT vpeffs0 -- v1.6 grows the v1.5 W fallback forward.
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_v15W_s0/periodic_prev}
SMOKE_JID=${SMOKE_JID:-}               # phase-2: the phase-1 smoke jobid to afterok-gate on
SKIP_SMOKE=${SKIP_SMOKE:-0}            # phase-2 escape hatch: 1 => no dependency

# --- cluster mechanics (generic gres header; SBATCH_EXCLUDE default; optional SBATCH_TIMELIMIT injection) ---
GRES=${GRES:-gpu:nvidia_a100:1}        # generic -> override for MIG/V100 (fail-loud OK)
EXCLUDE=${SBATCH_EXCLUDE:-adroit-h11g3}
TIMEOPT=""
if [ -n "${SBATCH_TIMELIMIT:-}" ]; then TIMEOPT="--time=${SBATCH_TIMELIMIT}"; fi
SBATCH_FILE="${SBATCH_FILE:-$(cd "$(dirname "$0")" && pwd)/peregrine_vq2_ego.sbatch}"

# ---- vpeffs0 base recipe (VERBATIM from v1.5) ----
BASE_VPEFFS0="++env.ego_blur_gate=false ++env.ego_yaw_cmd_clamp_rad_s=0.7 ++env.spin_abort_anneal=true \
++env.spin_abort_scale_start=2.6 ++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 \
++warmstart_reset_logstd=true ++warmstart_reset_logstd_std=0.18"

# ---- vpef8nc 8-gate/reward/infra block (VERBATIM from v1.5) ----
VPEF8NC="++env.course_n_gates=${NGATES} ++env.rw_v_cap=0 \
++env.rw_cross_zero_m=0.75 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_det_steps=600 ++eval_yaw_log=true \
++save_freq=${SAVEFREQ}"

# ---- THE TWO v1 DEPLOY-FAITHFULNESS ENABLES (VERBATIM) ----
FAITHFUL_SHARED="++dynamics.faithful_rate=true ++env.ego_vision_cadence=true \
++env.ego_vision_frame_hz=${VCFRAMEHZ} ++env.ego_vision_detect_p=${VCDETECTP}"

# ---- THE v1.5 SHARED SPEED-DISCIPLINE + YAW-QUIETNESS block (VERBATIM: A,B,D + roll-recover CORE +
#      the v1-arm-R VALIDATED recovery anneal). rw_yaw_duty + rw_yaw_dither are per-arm (extra_for). ----
V15_SHARED="++env.overspeed_abort_mps=${OVERSPEED} ++env.rw_progress_vcap_mps=${PROGVCAP} \
++env.rw_yaw_jerk=${YAWJERK} ++env.yaw_duty_free_band=${FREEBAND} ++env.rw_roll_recover=${RECOVER} \
++env.recovery_anneal=true ++env.recovery_start=0.0 ++env.recovery_hold_frac=0.3"

# ---- THE v1.6 SHARED MECHANISM block (mount override + range detection + aperture-margin anneal + pitch
#      jerk). rw_pitch_duty is per-arm (extra_for). ego_cam_mount_pitch_deg=20.0 is a ZERO-DELTA
#      re-statement of the already-baked +20 deg mount (byte-identical; see the header + the report). ----
V16_SHARED="++env.ego_cam_mount_pitch_deg=20.0 ++env.ego_vision_detect_mode=range \
++env.ego_det_r0=${DETR0} ++env.ego_det_k=${DETK} ++env.ego_det_plateau=${DETPLAT} \
++env.ego_det_far_start_m=${DETFAR} ++env.ego_det_far_p=${DETFARP} \
++env.pass_margin_anneal=true ++env.pass_margin_final_m=${PMFINAL} \
++env.pass_margin_lat_weight=${PMLATW} ++env.pass_margin_hold_frac=${PMHOLD} \
++env.rw_pitch_jerk=${PITCHJERK} ++env.pitch_duty_free_band=${PITCHBAND}"

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

# per-arm EXTRA builder: yaw_dither + yaw_duty + pitch_duty are per-arm (Q vs W); everything else shared.
# Offsets the vision-cadence RNG seed by the training seed (each seed sees a DIFFERENT reproducible
# detector-miss draw).
extra_for () {   # $1=SEED  $2=yaw_dither  $3=yaw_duty  $4=pitch_duty
  local SEED="$1" YD="$2" YDU="$3" PDU="$4"
  local ANTIDITHER="++env.rw_yaw_dither=${YD} ++env.yaw_dither_anneal=true \
++env.yaw_dither_start=${YDSTART} ++env.yaw_dither_hold_frac=${YDHOLDFRAC}"
  echo "${BASE_VPEFFS0} ${VPEF8NC} ${ANTIDITHER} ${FAITHFUL_SHARED} ${V15_SHARED} ${V16_SHARED} \
++env.rw_yaw_duty=${YDU} ++env.rw_pitch_duty=${PDU} ++env.ego_vision_cadence_seed=$((20260715 + SEED)) \
+init_from=${WARM}"
}

if [ "${MODE}" = "smoke" ]; then
  # -------- PHASE 1: ONE smoke, CORE=Q wiring, UPD=100, seed 0 --------
  submit "v16smoke_s0" 0 "$(extra_for 0 "${YAWDITHER}" "${YAWDUTY}" "${PITCHDUTY}")" 100
  cat <<'EOF'

=== v1.6 SMOKE submitted (UPD=100, CORE=Q wiring: v1.5 CORE + v1.6 mount@20.0 + range-detection +
    aperture-margin anneal + pitch jerk 0.03 + pitch_duty 0.1, warm from the v1.5 W arm periodic_prev).
    ADJUDICATE the COMPLETED smoke (a DONE log, never a live one -- the absent-hook != inert footgun L16):
      RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v16smoke_s0
      1. grep -nE 'ego_cam_mount_pitch_deg|ego_vision_detect_mode|pass_margin_anneal|pass_margin_final_m|rw_pitch_jerk|rw_pitch_duty|pitch_duty_free_band' \
           $RUNDIR/.hydra/config.yaml   # expect 20.0 / range / true / 1.0 / 0.03 / 0.1 / 0.6 (all under env:)
      2. grep -nE 'overspeed_abort_mps|rw_progress_vcap_mps|rw_yaw_duty|rw_yaw_jerk|rw_perception_next|faithful_rate|ego_vision_cadence' \
           $RUNDIR/.hydra/config.yaml   # 12.0 / 7.5 / 0.15 / 0.05 / 0.004 / true / true (v1.5 carryover + split)
      3. python scratchpad/tb_parse.py <event-file>   # env_loss/pitch_duty_pen, /pitch_jerk_pen,
           # /yaw_duty_pen, /prog_sat_forfeit, /overspeed_abort_rate ALL PRESENT (wiring watchdog)
      4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v16smoke_s0.out  # == 0, no NaN
      5. grep -E 'pass-margin-anneal|PITCH_EVAL' the .out: '[pass-margin-anneal] ON: ... SHRINK <start> -> 1.000'
         fired (NOT a silent skip), AND a PITCH_EVAL[...] line printed alongside YAW_EVAL.
    THEN release the real arms (afterok the smoke):
      MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v16.sh
    ESCAPE HATCH (no dependency): MODE=arms SMOKE_JID=none bash launch_v16.sh
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
    Q) for s in ${SEEDS_Q}; do submit "v16Q_s${s}" "${s}" "$(extra_for "${s}" "${YAWDITHER}" "${YAWDUTY}" "${PITCHDUTY}")" "" "${DEP}"; done ;;
    W) for s in ${SEEDS_W}; do submit "v16W_s${s}" "${s}" "$(extra_for "${s}" "0.6" "0.4" "0.2")"                        "" "${DEP}"; done ;;
    *) echo "!! unknown ARM '${ARM}' (expected Q / W) -- skipping" ;;
  esac
done

cat <<'EOF'

=== v1.6 arms submitted [Q W]. DEPLOY-PICK = the HARD JOINT yaw+pitch+speed selection (post-hoc):
    STEP 1 (sweep): for EACH seed, run ++rollout_only over the run's periodic + best_npg snapshots
      -> YAW_EVAL (signflips_per_s + satur_duty) + PITCH_EVAL (signflips_per_s + cmd_absmean + satur_duty)
      + DET_EVAL (n_passed_gates + max_speed) per ckpt.
    STEP 2 (HARD gates): DISCARD any ckpt with YAW signflips_per_s > ~4.0 OR satur_duty > ~0.1 OR
      cmd_absmean > ~0.3 (a yaw-hunter / rail-rider) REGARDLESS of n_passed; ALSO discard PITCH satur_duty
      high / PITCH signflips_per_s elevated (the Qs1 pitch limit cycle -- the v1.6 target). Confirm
      max_speed sits BELOW the 12 m/s abort with margin.
    STEP 3 (rank survivors): HIGHEST DET n_passed_gates; tiebreak LOWEST yaw+pitch signflips + satur + roll
      swing. NEVER select on value. GOAL: match the v1.5 W threading on the FULL 8-gate course with
      signflips ~0 (yaw AND pitch), no runaway, and the descend-leg blackout SURVIVED (range detection +
      aperture margin held).
    MONITOR: scratchpad/tb_parse.py on each event file (TB, NOT stdout). Watch env_loss/pitch_duty_pen +
      /pitch_jerk_pen + /yaw_duty_pen + /prog_sat_forfeit converge DOWN, /overspeed_abort_rate -> ~0,
      n_passed_gates climb. The [pass-margin-anneal] log line's margin_m SHRINKS start -> 1.000.
EOF
