#!/bin/bash
# =====================================================================================================
# v1.9 = M3 ONLY. It keeps the ONE v1.7-era mechanism that the 3-seed v1.8 census earned, and strips the
# other three, warm-started from the v1.8 DEPLOY LEAD (v18Q_s1). Net vs v1.8: M2 OFF (blind_abort_s
# 1.2 -> 0.0) and M4 REVERTED (pass_margin_final_m 0.75 -> 1.0). M1 stays OFF (0.0). M3 unchanged.
#
# WHY EACH VERDICT (all measured 2026-07-21..22; SSOT memory/deploy-emission-2026-07-22.md):
#   M1 HANDOFF SPAWN -- OFF, and it STAYS off. Premise falsified over 161 wire flights (0/161 hand over
#     LEVEL; gate median +2.42 deg off the optical axis, dead-centre; M1's own spawn box contains 0/161).
#     CONVICTED on release pitch: 425-flight replay through the real deploy loader (scratchpad/
#     replay_sweep_v18.py) -- v1.7 dove HARDER than its v1.6 parent on ~423/425 flights; v1.8 (M1 off)
#     restored the parent level (mean -0.801 vs parent -0.805). EXONERATED on miss_rate (turning it off
#     never recovered the rate -- a SEPARATE fork). Do NOT re-enable by widening the dz band: still trains
#     a level handoff, which is 0/161.
#   M2 BLIND-ABORT -- OFF for v1.9. It is INERT in the CURRENT sim: blind_abort_rate == 0.0000 on all three
#     v1.8 arms, because THIS env never goes blind (per-frame Bernoulli detection, no occlusion/burst model)
#     while the wire is 23.7% blind. A fatal terminal that never fires cannot teach; carrying it only
#     obscures the M3-alone read. NOTE: M2 is not WRONG, it is UNTESTABLE here -- it becomes live the moment
#     the next-gen env models real blind windows (the pilot's coast ask, 2026-07-22). Revisit it THERE, not
#     as a bolt-on to an env that cannot exercise it.
#   M3 CENTERING-MULTIPLIER ON PROGRESS -- THE KEEPER, unchanged. Best frame_factor (v18Qs1 0.841, climbing)
#     and best center_penalty (0.163 vs 0.173 parent) of the census. It multiplies the positive per-step
#     progress credit by a framing factor f in [floor 0.5, 1.0], so a blind/off-axis leg banks HALF the
#     progress a framed leg banks -- paid in the currency the policy optimises, farm-proof (multiplies
#     progress: hover earns f*0). The block below is verbatim from v1.8.
#   M4 APERTURE 0.75 -- REVERTED to 1.0. It bought NOTHING measurable: pass_offset was FLAT at 0.174-0.179
#     across all 5 v1.7/v1.8 runs and BOTH aperture settings -- real training pressure, zero centering gain.
#     Its apparent "miss_rate regression" is mostly DEFINITIONAL (miss_rate is scored against the ANNEALED
#     margin, peregrine_racing_ego.py:1922, so a tighter aperture reclassifies geometric passes as misses)
#     with a real ~8.6% tail the policy simply never responded to. Removing it drops that cost while M3
#     keeps the centering. This is the pilot's "sharper flights" ask ANSWERED THE OTHER WAY: M3's coupling
#     centers better than M4's constraint did, without the miss tax.
#
# WARM = the v1.8 Q_s1 arm (v18Q_s1), our current DEPLOY LEAD -- the actor with the release dive already
# fixed (M1 off) + M3's centering baked in. v1.9 therefore builds FORWARD from the best flying policy and
# only removes two dead knobs, rather than re-deriving from v1.6. (For a CLEAN M3-vs-M3+M2+M4 adjudication
# instead of a forward lead, set WARM to the v16Q_s1 checkpoints v1.8 used -- see the WARM= line.) The dir
# carries actor.pth + critic.pth, the same shape v1.8 warm-started from.
#
# EVERYTHING ELSE IS v1.8 VERBATIM (which was v1.6 verbatim + M1-off): the v1.5 speed-discipline block,
# the v1.6 mount/range-detection/pitch-quietness block, BASE_VPEFFS0, VPEF8NC, faithful_rate +
# ego_vision_cadence, and the per-arm yaw/pitch temperaments. Every v1.7-era knob is config-gated and the
# OFF value is byte-identical v1.6, so this launcher = v1.6 + M3.
#
# FULL PER-MECHANISM DERIVATION (the falsification numbers, the M3 ledger math, the M4 Rayleigh cost fit)
# lives in the v1.8 launcher header (rl/launch_v18.sh) and the SSOT memory/deploy-emission-2026-07-22.md.
# It is NOT repeated here so this header cannot drift out of sync with it. The one-paragraph verdict per
# mechanism is at the top of this file; below is only what v1.9 SHIPS.
#
# WHAT CARRIES OVER VERBATIM: the ENTIRE V15_SHARED block (overspeed_abort 12.0, rw_progress_vcap_mps 7.5,
# yaw_jerk, yaw_duty_free_band, roll_recover + recovery anneal) and the ENTIRE V16_SHARED block (mount
# 20.0, range detection r0/k/plateau/far, aperture-margin anneal, pitch jerk + free band) with ONLY
# pass_margin_final_m moved 1.0 -> 0.75 (M4). Also verbatim: BASE_VPEFFS0 (blur OFF, yaw clamp 0.7,
# spin-abort anneal, ticks_hi=1, logstd reset), VPEF8NC (8 gates, noise anneal, critic_grad_clip=10,
# ckpt_select_metric=n_passed_gates, eval_det_steps=600, eval_yaw_log=true, UPD 18000, save_freq 500),
# faithful_rate + ego_vision_cadence, and the per-arm yaw_dither/duty/pitch_duty temperaments.
# algo=appo is FIXED IN THE SBATCH (peregrine_vq2_ego.sbatch: "algo=appo ... is FIXED") -- the launcher
# must NOT re-set it; gamma (0.9975) likewise comes from the stage.
#
# WARM = the v1.6 Q_s1 arm's end-of-run checkpoints. DELIBERATE, and NOT the census winner: Q_s0 edged
# Q_s1 in census (5.69 vs 5.65 gates) but Q_s1 is the arm that actually SET THE 7-GATE DEPLOY RECORD.
# Census rank and wire rank have diverged before in this lineage; when they disagree, the WIRE wins.
#
# ARMS (each a SEPARATE SLURM job so one seed's crash cannot kill the others). The v1.8 mechanisms are
# SHARED across all arms -- the arms vary only the yaw/pitch damping TEMPERAMENT, so the census reads the
# mechanisms against one axis instead of a fan:
#   Q  (the v1.6 lead temperament), seeds 0,1 : yaw_dither 0.4 / yaw_duty 0.15 / pitch_duty 0.1. TAG=v19Q_s{0,1}.
#   W  (heavier-damped),           seed 0    : yaw_dither 0.6 / yaw_duty 0.4  / pitch_duty 0.2. TAG=v19W_s0.
#
# ---------------------------------------------------------------------------------------------------
# PHASE 1 (smoke):  bash launch_v19.sh
#     submits ONE smoke (v19smoke_s0, UPD=100, CORE=Q wiring). Prints its jobid + the ADJUDICATION
#     CHECKLIST + the exact Phase-2 command.
# ADJUDICATE (when the smoke COMPLETES -- a DONE log, never a live one; the absent-hook != inert footgun L16):
#     RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v19smoke_s0
#     1. grep -nE 'handoff_spawn_frac|blind_abort_s|blind_abort_range_m|rw_progress_frame_mult|progress_frame_floor|progress_frame_scale_rad|pass_margin_final_m' \
#          $RUNDIR/.hydra/config.yaml   # 0.0 (M1 OFF) / 0.0 (M2 OFF -- v1.9) / 15.0 / 1.0 / 0.5 / 0.5 / 1.0 (M4 REVERTED -- v1.9) (all under env:)
#     2. grep -nE 'rw_perception|rw_perception_next|overspeed_abort_mps|rw_progress_vcap_mps|ego_vision_detect_mode|ego_cam_mount_pitch_deg' \
#          $RUNDIR/.hydra/config.yaml   # 0.016 / 0.004 / 12.0 / 7.5 / range / 20.0 (v15+v16 carryover; M3's additive raise stays)
#     3. python scratchpad/tb_scalars.py <event-file>   # THE WIRING WATCHDOG. M3 is the ONLY live mechanism now:
#          env_loss/frame_factor                                  (M3 armed; < 1.0 == it is BITING) -- THE key to watch
#          plus the v15/v16 keys: env_loss/pitch_duty_pen, /pitch_jerk_pen, /yaw_duty_pen,
#          /prog_sat_forfeit, /overspeed_abort_rate
#          env_loss/blind_abort_rate  -- EXPECT ABSENT or PINNED 0.0 (M2 is OFF in v1.9). Present-and-nonzero
#            means blind_abort_s did NOT take the 0.0 -> re-check step 1.
#     4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v19smoke_s0.out  # == 0, no NaN
#     5. grep 'pass-margin-anneal' the .out: with PMFINAL=1.0 the end value equals the v1.6 baseline, so the
#        anneal is a STRICT NO-OP (start == end == 1.677 -> 1.0 is the v1.6 shape). A '[pass-margin-anneal]
#        ... SHRINK ... -> 0.750' line here means M4 did NOT revert -> re-check PMFINAL.
#     6. SANITY on the one live rate in TB:
#          env_loss/frame_factor      -- EXPECT BELOW 1.0 early (blind/off-axis legs discounted) and CLIMBING
#            toward 1.0 as the policy learns to keep the gate framed. Pinned at 1.0 == M3 inert (wiring bug).
#            Warm-started from v18Q_s1 it should START already-high (that policy converged frame_factor 0.841)
#            and hold/climb -- a DROP at update 0 would mean the warm actor is not being loaded.
# PHASE 2 (arms):  MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v19.sh
#     submits Q(0,1) + W(0), each --dependency=afterok:<smoke_jobid>. ESCAPE HATCH: SMOKE_JID=none (or
#     SKIP_SMOKE=1) submits the arms with NO dependency.
# CLUSTER MECHANICS (overridable): GRES (default gpu:nvidia_a100:1), SBATCH_EXCLUDE (default adroit-h11g3),
#     SBATCH_TIMELIMIT (optional --time injection; else the sbatch 20h default).
# =====================================================================================================
set -euo pipefail
STAGE=dual_gate_fullstack_floor_pef16  # REUSED from v1.6 -- every v1.8 knob is a +env. key passed here
MODE=${MODE:-smoke}                    # smoke (phase 1, default) | arms (phase 2)
# ALTERNATE-HARDWARE TAGGING (2026-07-21). Appended to every RUNTAG; EMPTY default == unchanged.
# adroit's gpu partition is 3 nodes and only h11g1 carries full A100s (4 of them), so the default
# GRES=gpu:nvidia_a100:1 queues ~13 h behind 55 pending jobs. h11g2 has 8x 3g.20gb MIG slices and
# h11g3 has 4x tesla_v100 -- both far less contended. Set TAG_SUFFIX when submitting to those so the
# run dirs do NOT collide with the A100 arms of the same name:
#   MODE=arms SMOKE_JID=none GRES=gpu:3g.20gb:1   TAG_SUFFIX=_mig bash launch_v19.sh
#   MODE=arms SMOKE_JID=none GRES=gpu:tesla_v100:1 TAG_SUFFIX=_v100 SBATCH_EXCLUDE= bash launch_v19.sh
# NOTE: --exclude=adroit-h11g3 (the default) is a NO-OP for A100 requests -- h11g3 is V100-only and
# never matched gpu:nvidia_a100:1 anyway. It DOES bite on tesla_v100, hence the empty SBATCH_EXCLUDE.
TAG_SUFFIX=${TAG_SUFFIX:-}
UPD=${UPD:-18000}                      # v1.5/v1.6 budget (unchanged)
SAVEFREQ=${SAVEFREQ:-500}
NGATES=${NGATES:-8}

# --- anti-dither (v1.5 values; DITHER-FROM-BIRTH). Q/CORE = 0.4; W = 0.6 (below). ---
YAWDITHER=${YAWDITHER:-0.4}            # CORE/Q yaw_dither
YDSTART=${YDSTART:-1.0}                # 1.0 = ON-FROM-BIRTH (warm from an already-smooth ancestor)
YDHOLDFRAC=${YDHOLDFRAC:-0.3}          # end-hold at full (inert when start=1.0; kept for parity)

# --- v1.5 YAW AMPLITUDE/DUTY + free band. Q/CORE = 0.15; W = 0.4 (below). ---
YAWDUTY=${YAWDUTY:-0.15}               # CORE/Q rw_yaw_duty
FREEBAND=${FREEBAND:-0.25}             # yaw_duty_free_band; the champion 0.232 absmean is INSIDE -> pays 0

# --- v1.6 PITCH DUTY (per-arm below) + free band + jerk. Q pitch_duty = 0.1; W = 0.2. ---
PITCHDUTY=${PITCHDUTY:-0.1}            # CORE/Q rw_pitch_duty
PITCHBAND=${PITCHBAND:-0.6}            # pitch_duty_free_band -- WIDE (pitch is the primary axis)
PITCHJERK=${PITCHJERK:-0.03}           # rw_pitch_jerk (shared)

# --- v1.5 shared anti-runaway (A,B) + yaw jerk (D) + roll-recover (CORE, all arms) ---
OVERSPEED=${OVERSPEED:-12.0}           # overspeed_abort_mps (A); fatal OOB-class above this GT speed
PROGVCAP=${PROGVCAP:-7.5}              # rw_progress_vcap_mps (B); NOT 5.0
YAWJERK=${YAWJERK:-0.05}               # rw_yaw_jerk (D); L1 |delta yaw_cmd| price
RECOVER=${RECOVER:-0.5}                # rw_roll_recover (CORE; on-from-birth, default theta0=30deg)

# --- v1.6 aperture-margin knobs. PMFINAL is THE M4 REVERT: 0.75 (v1.8) -> 1.0. It bought nothing (pass_offset
#     FLAT 0.174-0.179 across all 5 v1.7/v1.8 runs and both apertures) while taxing miss_rate; M3's coupling
#     centers better without it. 1.0 == the v1.6 baseline, so the anneal is a strict no-op at this value. ---
PMFINAL=${PMFINAL:-1.0}                # pass_margin_final_m -- M4 REVERTED to the v1.6 end value (was 0.75 in v1.8)
PMLATW=${PMLATW:-2.0}                  # pass_margin_lat_weight (lateral risk ~2x vertical)
PMHOLD=${PMHOLD:-0.25}                 # pass_margin_hold_frac (END-HOLD tight for the last 25%)

# --- v1.6 range-detection knobs (miner-calibrated; defaults shown for greppability) ---
DETR0=${DETR0:-2.2}; DETK=${DETK:-0.4}; DETPLAT=${DETPLAT:-0.99}
DETFAR=${DETFAR:-26.0}; DETFARP=${DETFARP:-0.79}

# --- vision cadence sub-knobs (v1 SSOT-correct defaults; net valid fix ~10.5 Hz) ---
VCFRAMEHZ=${VCFRAMEHZ:-30.0}
VCDETECTP=${VCDETECTP:-0.35}

# --- M1: HANDOFF SPAWN REALISM -- OFF, and it STAYS off (premise falsified; convicted on release pitch). ---
HANDOFFFRAC=${HANDOFFFRAC:-0.0}       # handoff_spawn_frac; 0.0 == OFF == byte-identical v1.6
HORANGELO=${HORANGELO:-9.5}            # handoff_range_lo_m  \_ the deploy first-lock band 10.3-11.2 m,
HORANGEHI=${HORANGEHI:-11.5}           # handoff_range_hi_m  /  widened (sample a band, not a point)
HODZLO=${HODZLO:--1.5}                 # handoff_gate_dz_lo_m  \_ gate centre MINUS drone altitude
HODZHI=${HODZHI:-1.5}                  # handoff_gate_dz_hi_m  /  +-1.5 m -> elevation ~+-9 deg
HOAGL=${HOAGL:-0.5}                    # handoff_min_agl_m; floor clearance above the pad (see the header)
HOYAWJIT=${HOYAWJIT:-0.25}             # handoff_yaw_jitter_rad (== the stage's course_spawn_yaw_jitter)

# --- M2: BLIND-FLIGHT ABORT -- OFF in v1.9 (was 1.2 in v1.8). INERT in this env (blind_abort_rate 0.0000
#     on all 3 v1.8 arms: sim never goes blind). Becomes live only in the next-gen coast env; revisit there.
#     Range/grace kept at their v1.8 values but DEAD while BLINDS=0.0 (blind_abort_s 0.0 == OFF). ---
BLINDS=${BLINDS:-0.0}                   # blind_abort_s; 0.0 == OFF == byte-identical v1.6 (v1.9 turns M2 off)
BLINDRANGE=${BLINDRANGE:-15.0}         # blind_abort_range_m (inert while BLINDS=0.0)
BLINDGRACE=${BLINDGRACE:-0.0}          # blind_abort_grace_s (inert while BLINDS=0.0)

# --- M3: PROGRESS FRAMING MULTIPLIER + the (guard-capped) additive perception raise -- THE KEEPER, unchanged. ---
FRAMEMULT=${FRAMEMULT:-1.0}            # rw_progress_frame_mult; 0.0 == OFF; 1.0 == the full [floor,1] swing
FRAMEFLOOR=${FRAMEFLOOR:-0.5}          # progress_frame_floor -- NON-ZERO is the anti-freeze guarantee
FRAMESCALE=${FRAMESCALE:-0.5}          # progress_frame_scale_rad (~ the 29.35 deg vertical half-FOV)
FRAMEEXP=${FRAMEEXP:-2.0}              # progress_frame_exponent (gaussian; gradient across the WHOLE frame)
PERC=${PERC:-0.016}                    # rw_perception -- 0.014 -> 0.016, THE MAX the farm guard allows
PERCNEXT=${PERCNEXT:-0.004}            # rw_perception_next; PERC+PERCNEXT == rw_time 0.02 (the ceiling)

# --- which arms + seeds ---
ARMS=${ARMS:-"Q W"}
SEEDS_Q=${SEEDS_Q:-"0 1"}
SEEDS_W=${SEEDS_W:-"0"}
# WARM = the v1.8 DEPLOY LEAD (v18Q_s1) -- the actor with the release dive already fixed (M1 off) + M3's
# centering baked in (converged frame_factor 0.841). v1.9 builds FORWARD from the best flying policy and only
# removes two dead knobs (M2, M4). The dir carries actor.pth + critic.pth, the same warm shape v1.8 consumed.
# The _mig suffix is the hardware tag of the v1.8 run that produced the lead (MIG slice), NOT a v1.9 knob.
# CLEAN-ADJUDICATION ALTERNATIVE: to score M3-alone vs M3+M2+M4 from a common ancestor instead of building a
# forward lead, override WARM to the v1.6 base v1.8 used:
#   WARM=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed1_v16Q_s1/checkpoints
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed1_v18Q_s1_mig/checkpoints}
SMOKE_JID=${SMOKE_JID:-}               # phase-2: the phase-1 smoke jobid to afterok-gate on
SKIP_SMOKE=${SKIP_SMOKE:-0}            # phase-2 escape hatch: 1 => no dependency

# --- cluster mechanics (generic gres header; SBATCH_EXCLUDE default; optional SBATCH_TIMELIMIT injection) ---
GRES=${GRES:-gpu:nvidia_a100:1}        # generic -> override for MIG/V100 (fail-loud OK)
# SBATCH_EXCLUDE="" (explicitly empty) OMITS --exclude entirely -- REQUIRED when requesting
# tesla_v100, since the default excludes h11g3 which is the ONLY node carrying V100s.
EXCLUDE=${SBATCH_EXCLUDE-adroit-h11g3}
EXCLOPT=""
if [ -n "${EXCLUDE}" ]; then EXCLOPT="--exclude=${EXCLUDE}"; fi
TIMEOPT=""
if [ -n "${SBATCH_TIMELIMIT:-}" ]; then TIMEOPT="--time=${SBATCH_TIMELIMIT}"; fi
SBATCH_FILE="${SBATCH_FILE:-$(cd "$(dirname "$0")" && pwd)/peregrine_vq2_ego.sbatch}"

# ---- vpeffs0 base recipe (VERBATIM from v1.5/v1.6) ----
BASE_VPEFFS0="++env.ego_blur_gate=false ++env.ego_yaw_cmd_clamp_rad_s=0.7 ++env.spin_abort_anneal=true \
++env.spin_abort_scale_start=2.6 ++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 \
++warmstart_reset_logstd=true ++warmstart_reset_logstd_std=0.18"

# ---- vpef8nc 8-gate/reward/infra block (VERBATIM from v1.5/v1.6) ----
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

# ---- THE v1.6 SHARED MECHANISM block (VERBATIM except pass_margin_final_m, which M4 moves 1.0 -> 0.75).
#      ego_cam_mount_pitch_deg=20.0 remains the ZERO-DELTA re-statement of the already-baked mount. ----
V16_SHARED="++env.ego_cam_mount_pitch_deg=20.0 ++env.ego_vision_detect_mode=range \
++env.ego_det_r0=${DETR0} ++env.ego_det_k=${DETK} ++env.ego_det_plateau=${DETPLAT} \
++env.ego_det_far_start_m=${DETFAR} ++env.ego_det_far_p=${DETFARP} \
++env.pass_margin_anneal=true ++env.pass_margin_final_m=${PMFINAL} \
++env.pass_margin_lat_weight=${PMLATW} ++env.pass_margin_hold_frac=${PMHOLD} \
++env.rw_pitch_jerk=${PITCHJERK} ++env.pitch_duty_free_band=${PITCHBAND}"

# ---- THE v1.7-era SHARED MECHANISM block. For v1.9 only M3 (framing multiplier + guard-capped perception
#      raise) is LIVE; M1 (handoff_spawn_frac) and M2 (blind_abort_s) keys are still PASSED but at their OFF
#      value 0.0, and M4 rides inside V16_SHARED as PMFINAL=1.0 (reverted). All-OFF reproduces v1.6; with only
#      M3 non-zero this block == v1.6 + M3. ----
V17_SHARED="++env.handoff_spawn_frac=${HANDOFFFRAC} ++env.handoff_range_lo_m=${HORANGELO} \
++env.handoff_range_hi_m=${HORANGEHI} ++env.handoff_gate_dz_lo_m=${HODZLO} \
++env.handoff_gate_dz_hi_m=${HODZHI} ++env.handoff_min_agl_m=${HOAGL} \
++env.handoff_yaw_jitter_rad=${HOYAWJIT} \
++env.blind_abort_s=${BLINDS} ++env.blind_abort_range_m=${BLINDRANGE} \
++env.blind_abort_grace_s=${BLINDGRACE} \
++env.rw_progress_frame_mult=${FRAMEMULT} ++env.progress_frame_floor=${FRAMEFLOOR} \
++env.progress_frame_scale_rad=${FRAMESCALE} ++env.progress_frame_exponent=${FRAMEEXP} \
++env.rw_perception=${PERC} ++env.rw_perception_next=${PERCNEXT}"

export STAGES="${STAGE}"
export PRECHECK=1
export "UPD_${STAGE}=${UPD}"

submit () {   # $1=RUNTAG  $2=SEED  $3=EXTRA  $4=n_updates-override(optional, smoke)  $5=dep(optional)
  local TAG="$1${TAG_SUFFIX:-}" SEED="$2" EX="$3" NUPD="${4:-}" DEP="${5:-}"
  export SEED
  export RUNTAG="${TAG}"
  export EXTRA="${EX}"
  if [ -n "${NUPD}" ]; then export "UPD_${STAGE}=${NUPD}"; else export "UPD_${STAGE}=${UPD}"; fi
  echo "=== submit ${TAG} SEED=${SEED} STAGE=${STAGE} UPD=${NUPD:-${UPD}} ngates=${NGATES} gres=${GRES}${DEP:+ dep=afterok:${DEP}} ==="
  if [ -n "${DEP}" ]; then
    sbatch --dependency="afterok:${DEP}" --gres="${GRES}" ${EXCLOPT} ${TIMEOPT} --export=ALL "${SBATCH_FILE}"
  else
    sbatch --gres="${GRES}" ${EXCLOPT} ${TIMEOPT} --export=ALL "${SBATCH_FILE}"
  fi
}

# per-arm EXTRA builder: yaw_dither + yaw_duty + pitch_duty are per-arm (Q vs W); the v1.8 mechanisms are
# SHARED. Offsets the vision-cadence RNG seed by the training seed (each seed sees a DIFFERENT reproducible
# detector-miss draw).
extra_for () {   # $1=SEED  $2=yaw_dither  $3=yaw_duty  $4=pitch_duty
  local SEED="$1" YD="$2" YDU="$3" PDU="$4"
  local ANTIDITHER="++env.rw_yaw_dither=${YD} ++env.yaw_dither_anneal=true \
++env.yaw_dither_start=${YDSTART} ++env.yaw_dither_hold_frac=${YDHOLDFRAC}"
  echo "${BASE_VPEFFS0} ${VPEF8NC} ${ANTIDITHER} ${FAITHFUL_SHARED} ${V15_SHARED} ${V16_SHARED} \
${V17_SHARED} ++env.rw_yaw_duty=${YDU} ++env.rw_pitch_duty=${PDU} \
++env.ego_vision_cadence_seed=$((20260722 + SEED)) +init_from=${WARM}"
}

if [ "${MODE}" = "smoke" ]; then
  # -------- PHASE 1: ONE smoke, CORE=Q wiring, UPD=100, seed 0 --------
  submit "v19smoke_s0" 0 "$(extra_for 0 "${YAWDITHER}" "${YAWDUTY}" "${PITCHDUTY}")" 100
  cat <<'EOF'

=== v1.9 SMOKE submitted (UPD=100, CORE=Q wiring: v1.5 + v1.6 CORE + M1 OFF (0.0) + M2 OFF (0.0) +
    M3 framing multiplier 1.0/floor 0.5 + M4 REVERTED (aperture 1.0), warm from the v1.8 DEPLOY LEAD
    v18Q_s1). ADJUDICATE the COMPLETED smoke (a DONE log, never a live one -- the absent-hook != inert
    footgun L16):
      RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v19smoke_s0
      1. grep -nE 'handoff_spawn_frac|blind_abort_s|rw_progress_frame_mult|progress_frame_floor|progress_frame_scale_rad|pass_margin_final_m' \
           $RUNDIR/.hydra/config.yaml   # expect 0.0 (M1 OFF) / 0.0 (M2 OFF) / 1.0 / 0.5 / 0.5 / 1.0 (M4 REVERTED) (under env:)
      2. grep -nE 'rw_perception|rw_perception_next|overspeed_abort_mps|rw_progress_vcap_mps|ego_vision_detect_mode|ego_cam_mount_pitch_deg' \
           $RUNDIR/.hydra/config.yaml   # 0.016 / 0.004 / 12.0 / 7.5 / range / 20.0 (v15+v16 carryover; M3's raise stays)
      3. python scratchpad/tb_scalars.py <event-file>   # WIRING WATCHDOG -- M3 is the ONLY live mechanism:
           env_loss/frame_factor   (the key -- must be PRESENT and < 1.0)
           env_loss/pitch_duty_pen  /pitch_jerk_pen  /yaw_duty_pen  /prog_sat_forfeit  /overspeed_abort_rate
           env_loss/blind_abort_rate -- EXPECT ABSENT / 0.0 (M2 off). Present-and-nonzero => BLINDS did not take 0.0.
      4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v19smoke_s0.out  # == 0, no NaN
      5. grep 'pass-margin-anneal' the .out: with PMFINAL=1.0 the end == the v1.6 baseline, so NO 'SHRINK -> 0.750'
         line should appear. If one does, M4 did NOT revert -- re-check PMFINAL.
      6. THE ONE LIVE RATE:
           env_loss/frame_factor -- warm-started from v18Q_s1 (converged 0.841) it should START high and
             HOLD/CLIMB toward 1.0. Pinned at 1.0 => M3 inert (wiring bug). A DROP at update 0 => the warm
             actor is not being loaded (check +init_from resolved to the v18Q_s1 dir).
    THEN release the real arms (afterok the smoke):
      MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v19.sh
    ESCAPE HATCH (no dependency): MODE=arms SMOKE_JID=none bash launch_v19.sh
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
    Q) for s in ${SEEDS_Q}; do submit "v19Q_s${s}" "${s}" "$(extra_for "${s}" "${YAWDITHER}" "${YAWDUTY}" "${PITCHDUTY}")" "" "${DEP}"; done ;;
    W) for s in ${SEEDS_W}; do submit "v19W_s${s}" "${s}" "$(extra_for "${s}" "0.6" "0.4" "0.2")"                        "" "${DEP}"; done ;;
    *) echo "!! unknown ARM '${ARM}' (expected Q / W) -- skipping" ;;
  esac
done

cat <<'EOF'

=== v1.9 arms submitted [Q W]. ADJUDICATION PROTOCOL (post-hoc; TAILS, NOT MEDIANS -- one strong event
    kills a flight with a perfect median, so every column below is a WORST-EVENT column):
    STEP 1 (sweep): for EACH seed, run ++rollout_only over the run's periodic + best_npg snapshots ->
      YAW_EVAL (signflips_per_s + cmd_absmean + satur_duty) + PITCH_EVAL (same three) + DET_EVAL
      (n_passed_gates + max_speed) per ckpt.
    STEP 2 (HARD gates -- DISCARD regardless of n_passed):
      * YAW signflips_per_s > ~4.0 OR satur_duty > ~0.1 OR cmd_absmean > ~0.3 (a yaw-hunter / rail-rider;
        the cmd_absmean gate is the one the v1 pick-flights proved transfers 1:1 to the wire).
      * PITCH satur_duty high OR signflips_per_s elevated (the Qs1 limit cycle).
      * max_speed must sit BELOW the 12 m/s abort WITH MARGIN.
    STEP 3 (v1.9 = M3-only, so there is exactly ONE mechanism read plus TWO must-not-regress checks):
      * M3 (the read): frame_factor must HOLD/CLIMB toward 1.0 (warm start already at 0.841). Flat-low or
        DROPPING == the policy is buying speed with framing and eating the discount -> the coupling is
        failing; report it, do not tune it away.
      * miss_rate MUST IMPROVE vs v1.8 (M4's aperture tax is gone). If it does NOT fall, M4 was not the
        cost -- re-open the miss_rate question rather than shipping.
      * The RELEASE DIVE must stay fixed: replay the arms through scratchpad/replay_sweep_v18.py and confirm
        the worst-nose-down tail is at/under the parent (v18Qs1 -0.801). M1 is off in both v1.8 and v1.9, so
        this should hold -- but VERIFY, since the warm start and reward change could perturb it.
      * blind_abort_rate / exit_blind are EXPECTED ABSENT (M2 off). If present-nonzero, BLINDS did not take 0.0.
    STEP 4 (rank survivors): HIGHEST DET n_passed_gates; tiebreak LOWEST yaw+pitch signflips + satur +
      peak roll. NEVER select on value. Multi-seed or it does not count (~2.8x seed variance).
    MONITOR: scratchpad/tb_scalars.py on each event file (TB, NOT stdout).
EOF
