#!/bin/bash
# =====================================================================================================
# v1.8 = v1.7 WITH M1 DISABLED, and NOTHING ELSE CHANGED. One knob: ++env.handoff_spawn_frac 0.33 -> 0.0.
# M1's premise was falsified against 161 wire flights on 2026-07-21 and the built v1.7 actors were measured
# to pitch 62% HARDER at the assist release than their own v1.6 parent (full derivation in the M1 block
# below). M2 + M3 + M4 are UNTOUCHED -- their premises were never tested by that measurement, and this
# generation exists to score them WITHOUT M1 contaminating the arm. Warm start is the SAME DEPLOY-PROVEN
# v1.6 Q_s1 checkpoint v1.7 used, so v1.8-vs-v1.7 is a clean single-mechanism read.
#
# v1.6 carryover (camera mount + range detection + aperture margin + pitch quietness, ALL wire-flown). Every new knob is config-gated and DEFAULT-OFF: a config
# without the v1.8 keys trains BYTE-IDENTICAL to v1.6 (pinned by tests/test_v17_mechanisms.py's
# equivalence + RNG-neutrality tests). NO new curriculum stage -- the stage stays
# dual_gate_fullstack_floor_pef16 and every v1.8 knob rides in on EXTRA (hydra ++ beats the stage dict,
# the same way v1.6's ++env.ego_yaw_cmd_clamp_rad_s=0.7 overrides the stage's 0.35).
#
# WHERE v1.6 LANDED (the wire evidence these four mechanisms are answering):
#   yaw + runaway CLOSED on the wire (yaw absmean 0.10-0.22, flips 0.3-2.7, speed self-held under 12).
#   Census v16Q_s0 5.69 / Q_s1 5.65 gates. DEPLOY RECORD 7 gates (v16Qs1), typical 3-5, best-config mean
#   3.3. Four residual modes, in priority order -- v1.8 attacks 1-3 BY CONSTRUCTION, 4 is the pilot's ask:
#     1 GATE-0 START DIVE (the pitch-cap dependency). Flown pitch-free the policy commands pitch_cmd
#       ~ -0.99 from tick 0 -> -53 deg -> the gate leaves the frame TOP -> lost, often re-locking the
#       WRONG gate. Pitch-free 0.75-0.89 gates vs 1.75-2.0 fenced.
#     2 BLIND-COAST YAW PANIC (the tail that kills flights). v16 raw yaw desire median 0.08 but p99 2.66 /
#       max 2.86 (v15 p99 0.57), and the spikes fire DURING detection blackouts (12 consecutive blind
#       ticks -> raw yaw pegged -2.7). The deploy clamp hides it, so it is a TRAINING defect: the policy
#       has learned no disciplined blind-flight behaviour.
#     3 LATE-LATERAL / TERMINAL CENTERING. Deaths cluster as frame clips: survived crossings radial 0.28 m
#       median / 0.57 p90 vs clip deaths 0.47 / 1.42; lateral risk ~2x vertical. The 7-gate record died
#       1.10 m short of gate 8 with a 0.6 m LATERAL offset + a -3.6 rad/s roll tail spike.
#     4 PILOT'S ASK (Fengyou): "tighten our aperture once again to get sharper flights."
#
# THE FOUR MECHANISMS
#   M1 HANDOFF SPAWN REALISM -- *** DISABLED IN v1.8: ++env.handoff_spawn_frac=0.0. ITS PREMISE WAS
#     FALSIFIED 2026-07-21 AGAINST THE WIRE LOGS. *** The v1.7 header asserted:
#         spawn attitude              camera optical axis
#         VQ1 tilted pad (-17.8 deg)        +2.0 deg
#         WIRE handoff (LEVEL)             +19.5 deg   <-- ASSUMED, NEVER MEASURED
#     and concluded the gate sits ~20 deg BELOW the axis at the wire handoff. It does not. Measured over
#     161 logged flights at the FIRST TICK AFTER ASSIST RELEASE (scratchpad/handoff_elevation.py):
#         obs[4] handoff attitude      p5 -0.5011  median -0.3178  p95 -0.3075 rad
#         gate elevation off the AXIS  p5  +0.59   median  +2.42   p95  +7.90  deg
#       * 0/161 hand over LEVEL; 63% sit at EXACTLY the pad tilt -0.3107. The wire hands over ON THE PAD.
#       * 161/161 have the gate inside the inner HALF of the frame; 0/161 have it >=15 deg below.
#       * The gate is DEAD-CENTRE at handoff on every flight -- and +2.42 deg measured is precisely the
#         "+2.0 deg tilted pad" row above, i.e. the row describing TRAINING. Training ALREADY matched the
#         wire; M1 INTRODUCED the mismatch it was built to remove.
#       * M1's own spawn box (gate +-1.5 m at 9.5-11.5 m) contains 0/161 flights: all 161 sit ABOVE it,
#         gate median +4.15 m up. It trained a confident prior for a geometry that never occurs.
#     COST, replayed through the real deploy loader over 388 flights (scratchpad/replay_sweep.py; harness
#     validated to 1.65e-04 against the flown actor_mean): v1.7 pitches HARDER at the release on 385-386
#     of 388 -- worst-command median -0.856 (v16Qs1 parent) -> -1.386 v17Qs0 (+62%), -1.262 Qs1, -1.258
#     Ws0. NOTE THE TRAP: over the first second v1.7's MEAN pitch is GENTLER (-0.254 vs -0.374); only the
#     TAIL exposes it. The sim census also preferred v1.7 (5.65 vs 5.46 g). Mean-and-census said ship.
#     SIGN ANCHOR (not convention): negative == nose-DOWN, anchored on the deploy fence semantics
#     (obs[4] <= -clamp_rad is a pitched-DOWN fence) and the pilot's own observation of a start dive.
#     DO NOT re-enable M1 by raising the dz band to the measured +3..+5.5 m: that still trains a LEVEL
#     handoff, which is 0/161. The base standing-start spawn already reproduces the wire's handoff pose.
#     The gate-0 dive is therefore STILL UNEXPLAINED -- attitude is NOT the OOD term. Fengyou's standing
#     hypothesis is the live one: "the speed is slow at the start, so pitching hard doesn't hit the speed
#     limit", i.e. nothing PRICES a hard dive from rest. M2 is the mechanism aimed at that, not M1.
#   M2 BLIND-ABORT BY CONSTRUCTION ++env.blind_abort_s=1.2 (attacks 1 AND 2; THE PITCH-CAP REPLACEMENT) --
#     a FATAL terminal in the overspeed/spin-abort family, NOT reward shaping (v1.6 armed rw_pitch_duty AND
#     rw_pitch_jerk and the dive survived both). If the ACTIVE gate was ACQUIRED at least once and then
#     goes CONTINUOUSLY out of frame for > 1.2 s while the drone is still CLOSING on it inside
#     ++env.blind_abort_range_m=15, the episode terminates OOB-CLASS (pays terminal_oob, KEEPS banked gate
#     progress -- death pricing UNCHANGED, exactly like the v1.5 overspeed abort). Blindness is the
#     GEOMETRIC in-frame mask, never the stochastic detector cadence: we price WHERE THE CAMERA POINTS,
#     which the policy controls.
#     SCOPING (the whole design; TESTED both ways): ``acquired`` is CLEARED on every target ADVANCE, so
#     immediately after a pass the new gate is un-acquired and the measured 0.7-5.8 s post-pass
#     acquisition gap on descend legs banks NO clock and CANNOT abort. The grace is EVENT-DRIVEN and EXACT
#     rather than a tuned timeout. ++env.blind_abort_grace_s (default 0.0 == off) is a belt-and-braces
#     second time grace if the smoke shows spurious fires.
#   M3 CENTERING-MULTIPLIER ON PROGRESS ++env.rw_progress_frame_mult=1.0 (attacks 2 AND 3) -- the COUPLING
#     change. The ledger read: rw_perception is an ADDITIVE ~0.014/step == ~4% of progress income at speed,
#     so it WHISPERS. v1.8 instead MULTIPLIES the positive per-step progress credit by a framing factor
#     f in [++env.progress_frame_floor=0.5, 1.0]: 1.0 with the gate dead-centre, decaying to the floor
#     off-axis or out of frame (angular scale ++env.progress_frame_scale_rad=0.5 ~ the 29.35 deg vertical
#     half-FOV; gaussian exponent 2.0). A leg flown blind banks HALF the progress a framed leg banks -- paid
#     in the currency the policy actually optimises. FARM-PROOF BY CONSTRUCTION (it multiplies progress:
#     hover-and-stare earns f*0 == 0). The NON-ZERO floor is LOAD-BEARING: at floor 0 a blinded policy
#     earns nothing for moving and its best move is to STOP and search -- the freeze seen at deploy pitch
#     clamp 1. It reuses the EXACT cos(optical axis, gate centre) rw_perception already reads (no second
#     angular model) and deliberately breaks the progress potential's pure telescoping -- framed paths are
#     worth more -- which is safe because f <= 1, so the term can only ever SHRINK income (no pump).
#     + the ADDITIVE raise, CAPPED BY A CODE GUARD -- READ THIS BEFORE RETUNING: the spec asked for
#     rw_perception 0.014 -> ~0.03-0.05, but ego_reward.EgoRewardWeights.__post_init__ asserts
#     rw_perception + rw_perception_next <= rw_time, and rw_time is 0.02 (the dataclass default; the
#     curriculum never overrides it). 0.03-0.05 would RAISE at construction. The maximum legal raise is
#     therefore PERC=0.016 with PERCNEXT=0.004 (sum 0.020 == the ceiling), which is what this launcher
#     ships. Going higher REQUIRES raising rw_time in lockstep -- a 2.5x increase in per-step time pressure,
#     i.e. MORE speed urgency, which works against modes 2 and 3. Deliberately NOT done. M3's multiplier is
#     the real lever here; the additive term was never it.
#   M4 APERTURE TIGHTENING ++env.pass_margin_final_m=0.75 (the pilot's ask) -- NO new mechanism, just the
#     v1.6 curriculum's end value 1.0 -> 0.75, same END-HOLD shape. Anneal start = gate_half_opening_m *
#     sqrt(lat_w^2+1) = 0.75*sqrt(5) = 1.6771 m, still far above the end (the train-loop start<=end guard
#     passes) and a STRICT no-op at update 0. It is a PURE LATERAL tightening: |lat| <= 0.375 m (half the
#     0.75 m aperture; v1.6 allowed 0.500) while the vertical bound 0.750 still coincides with the aperture
#     itself and stays INACTIVE -- the right axis, given lateral risk ~2x vertical and a 7-gate record that
#     died on a 0.6 m LATERAL offset. OFFLINE COST CHECK: fitting the reported crossing distribution
#     (radial 0.28 median / 0.57 p90 -> isotropic Rayleigh sigma 0.238-0.266 m; the two quantiles AGREE, so
#     the fit holds) the downgrade rate moves ~4-7% -> ~14-19% of geometric passes. A ~3x tightening, NOT a
#     collapse, and annealed in over the front 75%.
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
#   Q  (the v1.6 lead temperament), seeds 0,1 : yaw_dither 0.4 / yaw_duty 0.15 / pitch_duty 0.1. TAG=v18Q_s{0,1}.
#   W  (heavier-damped),           seed 0    : yaw_dither 0.6 / yaw_duty 0.4  / pitch_duty 0.2. TAG=v18W_s0.
#
# ---------------------------------------------------------------------------------------------------
# PHASE 1 (smoke):  bash launch_v18.sh
#     submits ONE smoke (v18smoke_s0, UPD=100, CORE=Q wiring). Prints its jobid + the ADJUDICATION
#     CHECKLIST + the exact Phase-2 command.
# ADJUDICATE (when the smoke COMPLETES -- a DONE log, never a live one; the absent-hook != inert footgun L16):
#     RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v18smoke_s0
#     1. grep -nE 'handoff_spawn_frac|handoff_range_lo_m|handoff_range_hi_m|blind_abort_s|blind_abort_range_m|rw_progress_frame_mult|progress_frame_floor|pass_margin_final_m' \
#          $RUNDIR/.hydra/config.yaml   # 0.0 (M1 OFF -- THE v1.8 CHANGE) / 9.5 / 11.5 / 1.2 / 15.0 / 1.0 / 0.5 / 0.75 (all under env:)
#     2. grep -nE 'rw_perception|rw_perception_next|overspeed_abort_mps|rw_progress_vcap_mps|ego_vision_detect_mode|ego_cam_mount_pitch_deg' \
#          $RUNDIR/.hydra/config.yaml   # 0.016 / 0.004 / 12.0 / 7.5 / range / 20.0 (v15+v16 carryover)
#     3. python scratchpad/tb_parse.py <event-file>   # THE WIRING WATCHDOG -- these keys MUST be present:
#          env_loss/blind_abort_rate, env_loss/blind_clock_mean  (M2 armed)
#          env_loss/frame_factor                                  (M3 armed; < 1.0 == it is BITING)
#          metrics/exit_blind                                     (M2 exit class)
#          plus the v15/v16 keys: env_loss/pitch_duty_pen, /pitch_jerk_pen, /yaw_duty_pen,
#          /prog_sat_forfeit, /overspeed_abort_rate
#     4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v18smoke_s0.out  # == 0, no NaN
#     5. grep 'pass-margin-anneal' the .out: '[pass-margin-anneal] ON: ... SHRINK 1.677 -> 0.750' fired
#        (NOT a silent skip, and the END value is 0.750 -- the M4 change actually landed).
#     6. SANITY on the two NEW rates in TB (these are the ones that can go wrong):
#          env_loss/blind_abort_rate  -- EXPECT small but NON-ZERO early (the gate is teaching). A rate
#            pinned at ~0 from update 0 means M2 is inert (check the config); a rate that stays HIGH and
#            flat means the scoping is firing on legitimate gaps -> raise ++env.blind_abort_grace_s.
#          env_loss/frame_factor      -- EXPECT BELOW 1.0 early (blind/off-axis legs being discounted) and
#            CLIMBING toward 1.0 as the policy learns to keep the gate framed. Pinned at 1.0 == inert.
# PHASE 2 (arms):  MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v18.sh
#     submits Q(0,1) + W(0), each --dependency=afterok:<smoke_jobid>. ESCAPE HATCH: SMOKE_JID=none (or
#     SKIP_SMOKE=1) submits the arms with NO dependency.
# CLUSTER MECHANICS (overridable): GRES (default gpu:nvidia_a100:1), SBATCH_EXCLUDE (default adroit-h11g3),
#     SBATCH_TIMELIMIT (optional --time injection; else the sbatch 20h default).
# =====================================================================================================
set -euo pipefail
STAGE=dual_gate_fullstack_floor_pef16  # REUSED from v1.6 -- every v1.8 knob is a +env. key passed here
MODE=${MODE:-smoke}                    # smoke (phase 1, default) | arms (phase 2)
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

# --- v1.6 aperture-margin knobs. PMFINAL is THE M4 CHANGE: 1.0 -> 0.75 (Fengyou's sharper-flights ask). ---
PMFINAL=${PMFINAL:-0.75}               # pass_margin_final_m -- v1.6 shipped 1.0; lateral envelope 0.5 -> 0.375
PMLATW=${PMLATW:-2.0}                  # pass_margin_lat_weight (lateral risk ~2x vertical)
PMHOLD=${PMHOLD:-0.25}                 # pass_margin_hold_frac (END-HOLD tight for the last 25%)

# --- v1.6 range-detection knobs (miner-calibrated; defaults shown for greppability) ---
DETR0=${DETR0:-2.2}; DETK=${DETK:-0.4}; DETPLAT=${DETPLAT:-0.99}
DETFAR=${DETFAR:-26.0}; DETFARP=${DETFARP:-0.79}

# --- vision cadence sub-knobs (v1 SSOT-correct defaults; net valid fix ~10.5 Hz) ---
VCFRAMEHZ=${VCFRAMEHZ:-30.0}
VCDETECTP=${VCDETECTP:-0.35}

# --- v1.8 M1: HANDOFF SPAWN REALISM (the wire's from-rest acquisition geometry) ---
HANDOFFFRAC=${HANDOFFFRAC:-0.0}       # handoff_spawn_frac; 0.0 == OFF == byte-identical v1.6
HORANGELO=${HORANGELO:-9.5}            # handoff_range_lo_m  \_ the deploy first-lock band 10.3-11.2 m,
HORANGEHI=${HORANGEHI:-11.5}           # handoff_range_hi_m  /  widened (sample a band, not a point)
HODZLO=${HODZLO:--1.5}                 # handoff_gate_dz_lo_m  \_ gate centre MINUS drone altitude
HODZHI=${HODZHI:-1.5}                  # handoff_gate_dz_hi_m  /  +-1.5 m -> elevation ~+-9 deg
HOAGL=${HOAGL:-0.5}                    # handoff_min_agl_m; floor clearance above the pad (see the header)
HOYAWJIT=${HOYAWJIT:-0.25}             # handoff_yaw_jitter_rad (== the stage's course_spawn_yaw_jitter)

# --- v1.8 M2: BLIND-FLIGHT ABORT (the pitch-cap replacement) ---
BLINDS=${BLINDS:-1.2}                  # blind_abort_s; 0.0 == OFF == byte-identical v1.6
BLINDRANGE=${BLINDRANGE:-15.0}         # blind_abort_range_m
BLINDGRACE=${BLINDGRACE:-0.0}          # blind_abort_grace_s; 0 == the acquired-reset alone (see header)

# --- v1.8 M3: PROGRESS FRAMING MULTIPLIER + the (guard-capped) additive perception raise ---
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
# WARM = the v1.6 Q_s1 arm's end-of-run checkpoints -- the DEPLOY-PROVEN lead (7-gate record), NOT the
# census winner Q_s0 (5.69 vs 5.65). When census rank and wire rank disagree, the WIRE wins.
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed1_v16Q_s1/checkpoints}
SMOKE_JID=${SMOKE_JID:-}               # phase-2: the phase-1 smoke jobid to afterok-gate on
SKIP_SMOKE=${SKIP_SMOKE:-0}            # phase-2 escape hatch: 1 => no dependency

# --- cluster mechanics (generic gres header; SBATCH_EXCLUDE default; optional SBATCH_TIMELIMIT injection) ---
GRES=${GRES:-gpu:nvidia_a100:1}        # generic -> override for MIG/V100 (fail-loud OK)
EXCLUDE=${SBATCH_EXCLUDE:-adroit-h11g3}
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

# ---- THE v1.8 SHARED MECHANISM block (M1 handoff spawn + M2 blind abort + M3 framing multiplier and the
#      guard-capped perception raise; M4 rides inside V16_SHARED as PMFINAL). ALL default-OFF in code --
#      dropping this one string reproduces v1.6 byte-for-byte. ----
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

# per-arm EXTRA builder: yaw_dither + yaw_duty + pitch_duty are per-arm (Q vs W); the v1.8 mechanisms are
# SHARED. Offsets the vision-cadence RNG seed by the training seed (each seed sees a DIFFERENT reproducible
# detector-miss draw).
extra_for () {   # $1=SEED  $2=yaw_dither  $3=yaw_duty  $4=pitch_duty
  local SEED="$1" YD="$2" YDU="$3" PDU="$4"
  local ANTIDITHER="++env.rw_yaw_dither=${YD} ++env.yaw_dither_anneal=true \
++env.yaw_dither_start=${YDSTART} ++env.yaw_dither_hold_frac=${YDHOLDFRAC}"
  echo "${BASE_VPEFFS0} ${VPEF8NC} ${ANTIDITHER} ${FAITHFUL_SHARED} ${V15_SHARED} ${V16_SHARED} \
${V17_SHARED} ++env.rw_yaw_duty=${YDU} ++env.rw_pitch_duty=${PDU} \
++env.ego_vision_cadence_seed=$((20260720 + SEED)) +init_from=${WARM}"
}

if [ "${MODE}" = "smoke" ]; then
  # -------- PHASE 1: ONE smoke, CORE=Q wiring, UPD=100, seed 0 --------
  submit "v18smoke_s0" 0 "$(extra_for 0 "${YAWDITHER}" "${YAWDUTY}" "${PITCHDUTY}")" 100
  cat <<'EOF'

=== v1.8 SMOKE submitted (UPD=100, CORE=Q wiring: v1.5 + v1.6 CORE + M1 handoff spawn OFF (0.0) + M2 blind
    abort 1.2 s + M3 framing multiplier 1.0/floor 0.5 + M4 aperture 0.75, warm from the DEPLOY-PROVEN
    v1.6 Q_s1 checkpoints). ADJUDICATE the COMPLETED smoke (a DONE log, never a live one -- the
    absent-hook != inert footgun L16):
      RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v18smoke_s0
      1. grep -nE 'handoff_spawn_frac|handoff_range_lo_m|handoff_range_hi_m|blind_abort_s|blind_abort_range_m|rw_progress_frame_mult|progress_frame_floor|pass_margin_final_m' \
           $RUNDIR/.hydra/config.yaml   # expect 0.0 (M1 OFF -- THE v1.8 CHANGE) / 9.5 / 11.5 / 1.2 / 15.0 / 1.0 / 0.5 / 0.75 (under env:)
      2. grep -nE 'rw_perception|rw_perception_next|overspeed_abort_mps|rw_progress_vcap_mps|ego_vision_detect_mode|ego_cam_mount_pitch_deg' \
           $RUNDIR/.hydra/config.yaml   # 0.016 / 0.004 / 12.0 / 7.5 / range / 20.0 (v15+v16 carryover)
      3. python scratchpad/tb_parse.py <event-file>   # WIRING WATCHDOG -- ALL must be PRESENT:
           env_loss/blind_abort_rate  env_loss/blind_clock_mean  env_loss/frame_factor
           metrics/exit_blind
           env_loss/pitch_duty_pen  /pitch_jerk_pen  /yaw_duty_pen  /prog_sat_forfeit  /overspeed_abort_rate
      4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v18smoke_s0.out  # == 0, no NaN
      5. grep 'pass-margin-anneal' the .out: '[pass-margin-anneal] ON: ... SHRINK 1.677 -> 0.750' fired
         (NOT a silent skip; the END value must read 0.750 -- that IS the M4 change).
      6. THE TWO NEW RATES (the ones that can go wrong):
           env_loss/blind_abort_rate -- small but NON-ZERO early. Pinned ~0 from update 0 => M2 inert
             (re-check the config). HIGH and FLAT => the scoping is firing on legitimate gaps
             => raise ++env.blind_abort_grace_s (0.5-1.0) and re-smoke.
           env_loss/frame_factor -- BELOW 1.0 early, CLIMBING toward 1.0. Pinned at 1.0 => M3 inert.
    THEN release the real arms (afterok the smoke):
      MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v18.sh
    ESCAPE HATCH (no dependency): MODE=arms SMOKE_JID=none bash launch_v18.sh
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
    Q) for s in ${SEEDS_Q}; do submit "v18Q_s${s}" "${s}" "$(extra_for "${s}" "${YAWDITHER}" "${YAWDUTY}" "${PITCHDUTY}")" "" "${DEP}"; done ;;
    W) for s in ${SEEDS_W}; do submit "v18W_s${s}" "${s}" "$(extra_for "${s}" "0.6" "0.4" "0.2")"                        "" "${DEP}"; done ;;
    *) echo "!! unknown ARM '${ARM}' (expected Q / W) -- skipping" ;;
  esac
done

cat <<'EOF'

=== v1.8 arms submitted [Q W]. ADJUDICATION PROTOCOL (post-hoc; TAILS, NOT MEDIANS -- one strong event
    kills a flight with a perfect median, so every column below is a WORST-EVENT column):
    STEP 1 (sweep): for EACH seed, run ++rollout_only over the run's periodic + best_npg snapshots ->
      YAW_EVAL (signflips_per_s + cmd_absmean + satur_duty) + PITCH_EVAL (same three) + DET_EVAL
      (n_passed_gates + max_speed) per ckpt.
    STEP 2 (HARD gates -- DISCARD regardless of n_passed):
      * YAW signflips_per_s > ~4.0 OR satur_duty > ~0.1 OR cmd_absmean > ~0.3 (a yaw-hunter / rail-rider;
        the cmd_absmean gate is the one the v1 pick-flights proved transfers 1:1 to the wire).
      * PITCH satur_duty high OR signflips_per_s elevated (the Qs1 limit cycle).
      * max_speed must sit BELOW the 12 m/s abort WITH MARGIN.
    STEP 3 (the v1.8-specific reads -- these decide whether the MECHANISMS worked, not just whether the
      policy is quiet):
      * M1/M2: exit_blind + blind_abort_rate must FALL toward ~0 by convergence. A converged policy that
        still aborts blind is one that never learned to hold the frame -- do NOT ship it, whatever its
        gate count. Cross-check exit_frame did not simply absorb the deaths.
      * M3: frame_factor must CLIMB toward 1.0. Flat-low == the policy is buying speed with framing and
        eating the discount; that is the coupling failing and it should be reported, not tuned away.
      * M4: pass_offset_m / cross_offset_m p90 must TIGHTEN vs the v1.6 census (the whole point of 0.75).
        Watch miss_rate: a modest rise is EXPECTED (~4-7% -> ~14-19% of geometric passes get downgraded);
        a collapse means the anneal outran the policy -> lengthen PMHOLD or back PMFINAL toward 0.85.
    STEP 4 (rank survivors): HIGHEST DET n_passed_gates; tiebreak LOWEST yaw+pitch signflips + satur +
      peak roll. NEVER select on value. Multi-seed or it does not count (~2.8x seed variance).
    MONITOR: scratchpad/tb_parse.py on each event file (TB, NOT stdout).
EOF
