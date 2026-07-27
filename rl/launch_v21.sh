#!/usr/bin/env bash
# =====================================================================================================
# v2.1 ARM 1 -- ++env.ego_obs_coast=true.  DERIVED FROM launch_v20.sh, ONE TOKEN CHANGED.
#
# WHY. On the wire the gate leaves the camera frame at ~1.8 m (VFOV is only 58.7 deg with the optical
# axis pitched +20 deg up; 0 of 899 confirmed passes ever got a fix inside 1.19 m). TRAINING models
# that blackout faithfully -- gate_visibility.gate_detectable, 8 keypoints, >=4 in frame, same
# intrinsics -- and its horizon lands within centimetres of the wire's. So the RANGE is not the
# mismatch. What happens ONCE BLIND is:
#     TRAINING masks obs[11:16] to ZEROS      (100% of ticks below 1.0 m)
#     THE WIRE feeds a FILLED, COASTED lever  (94% of ticks at the last tick before a gate advance)
# The policy has therefore never been trained on the input state in which every gate outcome is
# decided -- and the asymmetry runs OPPOSITE to the obvious guess: deploy hands it MORE information
# there, not less. ++env.ego_obs_coast=true drops the `keep &= det` mask (peregrine_racing_ego.py:521)
# so the coasted lever survives the blackout in training too.
#
# ARMS.  K = coast ON (5 seeds; ~1/3 of seeds collapse in this regime, so 5 launched -> ~3.3 usable).
#        B = the same recipe with the token OFF (1 seed) -- a same-batch control, so the comparison
#            is not against a run from another week on other hardware.
# Everything else -- warm start, stage, 18k updates, vertical block, 25 Hz vision, yaw/roll temperament
# -- is VERBATIM from v2.0.
#
# PRECHECKS, in order, and each one is free:
#   t+60 s   grep ego_obs_coast $RUNDIR/.hydra/config.yaml      -> must be true. Absent => UNARMED,
#            kill immediately; every later number would be a control run wearing the arm's name.
#   t+4 min  metrics/cross_offset_m at update ~100 must NOT be bit-identical to arm B. Identical =>
#            the token reached nothing => kill.
#   t+30 min env_loss/banked_prog_mean >= 50% of B's trace at the same update AND metrics/exit_oob
#            <= 0.10. The run starts from a competent warm policy; if dropping the zero-mask has
#            destroyed its terminal behaviour it shows here. Kill and reallocate rather than pay the
#            remaining ~3.8 h x 5 seeds.
#   any seed with NaN or flat losses/value_loss at update 100: kill THAT seed, relaunch it, and do
#            not count it toward the 5.
#
# ACCEPT/REJECT, FIXED IN ADVANCE (do not move this afterwards):
#   ACCEPT iff median n_passed_gates over >=3 non-collapsed K seeds >= B + 0.20 gates, evaluated in
#   the SAME env for both. A tie is a REJECT: the arm costs a retrain of the deploy lead, so it must
#   pay for itself. Record the tail (p10/max), not only the median.
#
# ADROIT AUP -- a violation SUSPENDS THE ACCOUNT: SLURM only, never compute on a login node; compute
# nodes have NO internet so everything must be pre-staged; output to /scratch; --mem must be accurate;
# a job at zero GPU utilisation is killed at 2 h. Measured 18k-update wall: A100 ~4:18, MIG ~6:00,
# V100 ~7:02 -- so --time=6h is honest for A100/MIG and leaves the 20 h default unused.
#
# USAGE (phase 1 then phase 2, exactly as v2.0):
#   MODE=smoke bash launch_v21.sh
#   MODE=arms SMOKE_JID=<jobid> bash launch_v21.sh
# =====================================================================================================
# =====================================================================================================
# v2.0 = v1.9 (SHIPPED) + THE VERTICAL-STRUCTURE FIX + 25 Hz VISION PARITY.
# It carries the v1.9 W-arm roll damping VERBATIM (roll_jerk 0.08 / roll_duty 0.2 / free band 0.8 -- the
# arm that WON the v1.9 census and became the deploy lead), warm-starts from that arm's checkpoints, and
# adds exactly TWO things: (A) courses with genuine VERTICAL structure, (B) the vision frame clock moved
# 30 -> 25 Hz to match the measured wire. Nothing else changes.
#
# WHY (A) -- THE MEASURED DEFECT (51 deploy flights): the policy flies ~0.67 m BELOW gate centre and
# cannot close a sustained climb; the course's two consecutive "UP" gates pass at 47% / 22% vs 74-83%
# for level/down gates. ROOT CAUSE, measured offline over 30000 sampled courses (the report tool is
# `python rl/peregrine_racing_ego.py <stage> [k=v ...]`):
#     obs[10] (the coarse VERTICAL sector, {-1,0,+1}) is fed +1 on 37.6% of DEPLOY ticks
#     but the TRAINING sampler realises +1 on   1.6%  of gates.
#     64.0% of sampled gates sit PINNED on the 0.5 m floor and 45.9% of whole courses are
#     ENTIRELY FLAT (every gate's vert bucket == 0). The policy has never been trained to climb.
# !! THE CAUSE IS *NOT* course_gates_above_spawn (checked, and it is the obvious wrong suspect): turning
# the floor OFF leaves the +1 fraction at 1.60% EXACTLY -- the floor clamp only ever SHRINKS a descent
# (it converts -1 legs into 0 legs: -1 goes 59.5% -> 7.2%), it can never create a climb. The real cause
# is the sampler's DESCENT-BIASED default band drop_m=(-3, +12) (peregrine_course.py): dz = -drop lands
# in [-12, +3], so the climb half is capped at 3 m while legs are 10-20 m long and the 0.20 rad sector
# deadband needs dz > tan(0.20)*L ~ 2.0-4.1 m. Climbs steep enough to LABEL as +1 are nearly impossible
# to draw. (The HORIZ axis is healthy at 35/29/35 -- the degeneracy is vertical-only.)
#
# THE FIX IS DISTRIBUTIONAL, NOT A REWARD: widen the vertical band symmetrically (the EXISTING, already-
# wired course_drop_lo/hi keys) and bound the resulting walk with a NEW CEILING knob:
#     ++env.course_drop_lo=-10.0 ++env.course_drop_hi=10.0 ++env.course_gates_ceiling=16.0
# !! THE CEILING IS THE NEW CODE (peregrine_course.gates_ceiling_m + the course_gates_ceiling bridge in
# peregrine_racing_ego.resolve_course_overrides; default None == OFF == byte-identical). It is the exact
# symmetric counterpart of the existing gates_above_spawn_m floor, enforced in the SAME sequential walk
# clamp. WITHOUT it a symmetric drop band is a reflected random walk that drifts unboundedly upward
# (measured max gate z 48 m over 8 gates, 4.7% of gates above 20 m) -- physically absurd for a warehouse
# and it starves the floor-discipline lesson. WITH it the walk OSCILLATES inside a warehouse-scale band.
#
# MEASURED RESULT of the chosen setting (n=30000 courses x 8 gates = 240k gate samples, seed 20260725):
#     DEFAULT (v1.9)                    +1 =  1.57%   -1 =  7.27%   all-flat courses 45.9%   max z 17.5 m
#     drop(-10,10) + ceiling 16   ->    +1 = 30.13%   -1 = 20.50%   all-flat courses  2.2%   max z 16.0 m
# 30.1% lands mid-band against the 37.6% deploy tick-rate, with a healthy DOWN bucket kept (20.5%) so
# this trains vertical CONTROL, not a climb bias. Floor-pinning falls 64% -> 22%; roof-pinning is 7.6%.
# VERIFY IT OFFLINE BEFORE LAUNCH (takes ~5 s on a laptop, no cluster):
#     python rl/peregrine_racing_ego.py dual_gate_fullstack_floor_pef16 course_n_gates=8 \
#            course_drop_lo=-10.0 course_drop_hi=10.0 course_gates_ceiling=16.0
#
# !! STOP: DO NOT "FIX" THIS BY MOVING THE SECTOR DEADBAND (ego_coarse_vert_thresh_rad, 0.20 rad). It would
# raise the +1 count for free and it is the WRONG lever: the DEPLOY side reads a HAND-AUTHORED coarse-map
# JSON written to the 0.20 rad {-1,0,+1} convention, so a training-only deadband change desyncs the LABEL
# SEMANTICS from the wire. Move the GEOMETRY, never the label.
#
# WHY (B) -- SENSOR-RATE PARITY: the measured wire vision rate is ~25 Hz; training runs the frame clock
# at 30. !! AND THE 30 Hz SETTING IS A NEAR-NO-OP: the CONTROL tick is env.dt=0.0333 s == 30.03 Hz, so a
# 30 Hz frame period (33.33 ms) ~= dt and the clock fires on 99.90% of ticks -- today the ONLY thing
# thinning the fresh-fix stream is the detector-success draw. 25 Hz is EXACTLY representable (30.03/25 =
# 6/5 -> a period-6 ".FFFF." pattern = 5 frames per 6 ticks = 25.00 Hz effective) and it makes the frame
# clock BITE for the first time. NET EFFECT: fresh-fix rate 30*0.35 = 10.50 Hz -> 25*0.35 = 8.75 Hz.
# A non-frame tick DROPS the fix (no hold, no interpolation): the estimator ego-propagates its held fix
# and decays confidence -- the existing multi-rate path. This is a HARDER, more faithful regime; expect
# a small early cost in centring, and watch target_detectable_duty.
# IMU CADENCE IS UNCHANGED AND DELIBERATELY SO: ++dynamics.n_substeps=5 generates plant/IMU truth at
# ~150 Hz (~ the measured 143.3 Hz wire IMU) but the estimator consumes the LAST substep sample ONCE per
# 30 Hz control tick -- because that is what the deploy navigator does ("Estimation advances only on a
# NEW IMU sample"; single-latest-rate propagation reproduces the recorded wire leveler to median
# 0.0003 deg, rl/tools/imu_foundation.py). n_substeps also changes plant trajectories, so it is NOT a
# free knob. Left alone.
#
# EVERYTHING ELSE IS v1.9 VERBATIM: the v1.5 speed-discipline block (overspeed 12.0, vcap 7.5, yaw jerk,
# yaw duty free band, roll_recover + recovery anneal), the v1.6 block (mount 20.0, range detection,
# aperture-margin anneal at PMFINAL=1.0, pitch jerk + free band), the v1.9 roll rate-damping, M1/M2 OFF,
# M3 ON (framing multiplier 1.0 / floor 0.5 + the guard-capped perception raise 0.016/0.004), BASE_VPEFFS0,
# VPEF8NC, and faithful_rate + ego_vision_cadence. algo=appo is FIXED IN THE SBATCH -- the launcher must
# NOT re-set it; gamma (0.9975) likewise comes from the stage.
#
# WARM = the v1.9 W arm (v19W_s0_mig) -- the arm that WON the v1.9 census (roll signflips 2.6 -> 2.26,
# n_passed 5.72 -> 6.03, release-dive replay improved) and SHIPPED as the deploy lead
# ego-ckpts-v19-2026-07-24. Same shape v1.9 warm-started from (actor.pth + critic.pth). The _mig suffix
# is the hardware tag of the run that produced the lead, NOT a v2.0 knob.
#
# ARMS (each a SEPARATE SLURM job so one seed's crash cannot kill the others). ALL arms share the v1.9 W
# roll damping, the v1.9 W yaw/pitch temperament and the 25 Hz vision, so the census reads the VERTICAL
# change against ONE axis:
#   V (the vertical fix), seeds 0,1 : drop(-10,10) + ceiling 16 -> +1 ~ 30%. TAG=v20V_s{0,1}.
#   C (the control),      seed 0    : vertical knobs OFF (legacy descent-biased band, no ceiling) ->
#                                     +1 ~ 1.6%, IDENTICAL in every other respect. TAG=v20C_s0.
# C-vs-V isolates the VERTICAL change CLEANLY -- the course knobs are the only tokens that differ, and
# both arms share the warm start, the seed, and the cadence. C-vs-the-v19W_s0 parent prices the 25 Hz
# VISION change, but read it as a TREND not a controlled A/B: C is v19W CONTINUED for another 18000
# updates at 25 Hz (different cadence RNG seed too), not a re-run of it. Two reads, three jobs -- and V
# carries the seed replicate because a single seed does not count (~2.8x seed variance).
#
# !! WHAT THE VERTICAL CHANGE DOES *NOT* TOUCH (verified bit-identical over 4000 sampled courses at a
# fixed RNG seed): the SPAWN POSE, spawn_yaw, and the GATE-0 position -- x, y AND z. course_spawn_below_g0
# (0.5-6.0 m) is untouched, the drop band is drawn with the SAME shape so the RNG stream position is
# preserved, and neither the 0.5 m floor nor the 16 m ceiling can bind on gate 0. This matters: the v1.8/
# v1.9 gate-1 competence is LAUNCH-TRAINED (the v16 gate-1 wall was launch-OOD, 0/8 vs 11/12), so the
# launch geometry those checkpoints learned is preserved BY CONSTRUCTION. Only gates 1+ move.
# !! gpu-medium QOS runs 2 CONCURRENT GPUs; queuing more never helps. Submit all three, they will drain.
#
# ---------------------------------------------------------------------------------------------------
# PHASE 1 (smoke):  bash launch_v20.sh
#     submits ONE smoke (v20smoke_s0, UPD=100, CORE=V wiring). Prints its jobid + the ADJUDICATION
#     CHECKLIST + the exact Phase-2 command.
# ADJUDICATE (when the smoke COMPLETES -- a DONE log, never a live one; the absent-hook != inert footgun L16):
#     RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v20smoke_s0
#     1. grep -nE 'course_drop_lo|course_drop_hi|course_gates_ceiling|course_gates_above_spawn|ego_vision_frame_hz' \
#          $RUNDIR/.hydra/config.yaml   # -10.0 / 10.0 / 16.0 / 0.5 / 25.0  (all under env:)
#     2. grep -nE 'rw_roll_jerk|rw_roll_duty|roll_duty_free_band|rw_progress_frame_mult|pass_margin_final_m|handoff_spawn_frac|blind_abort_s' \
#          $RUNDIR/.hydra/config.yaml   # 0.08 / 0.2 / 0.8 (v1.9 W) / 1.0 / 1.0 / 0.0 / 0.0
#     3. python scratchpad/tb_scalars.py <event-file>   # THE WIRING WATCHDOG. The NEW keys are the headline:
#          env_loss/course_vert_up_frac   -- EXPECT ~0.30 (the sampler's realised +1 over ALL live gates).
#            PINNED ~0.016 == the vertical knobs did NOT take -> re-check step 1. THIS IS THE ONE TO READ.
#          env_loss/vert_sector_up_duty   -- the TICK-WEIGHTED twin: fraction of envs whose CURRENT TARGET
#            vert bucket == +1 this step. This is literally the obs[10] channel the actor sees, so it is
#            the key directly comparable to the 37.6% deploy number. Expect ~0.25-0.40.
#          env_loss/course_vert_down_frac / vert_sector_down_duty -- expect ~0.20 (down structure KEPT).
#          env_loss/target_detectable_duty -- the 25 Hz READ. Expect a modest drop vs v1.9 (fresh-fix
#            10.5 -> 8.75 Hz). A COLLAPSE means the cadence is starving the estimator -> report, do not tune.
#          env_loss/roll_jerk_pen, /roll_duty_pen  -- NON-ZERO (the v1.9 fix still armed).
#          env_loss/frame_factor  -- M3 armed; warm from v19W_s0 it should START high and hold/climb.
#          env_loss/blind_abort_rate -- EXPECT ABSENT or PINNED 0.0 (M2 is OFF).
#     4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v20smoke_s0.out  # == 0, no NaN
#     5. grep exit_ceiling / oob_rate in the trace: the course now reaches 16 m, so the OOB box top rises
#        with it (the box is the course bbox + 12 m). A JUMP in exit_ceiling would mean the policy is
#        climbing out rather than through -- report it, that is the main destabilisation risk here.
# PHASE 2 (arms):  MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v20.sh
#     submits V(0,1) + C(0), each --dependency=afterok:<smoke_jobid>. ESCAPE HATCH: SMOKE_JID=none (or
#     SKIP_SMOKE=1) submits the arms with NO dependency.
# CLUSTER MECHANICS (overridable): GRES (default gpu:nvidia_a100:1), SBATCH_EXCLUDE (default adroit-h11g3),
#     SBATCH_TIMELIMIT (optional --time injection; else the sbatch 20h default).
# =====================================================================================================
set -euo pipefail
STAGE=dual_gate_fullstack_floor_pef16  # REUSED from v1.6 -- every v2.0 knob is a +env. key passed here
MODE=${MODE:-smoke}                    # smoke (phase 1, default) | arms (phase 2)
# ALTERNATE-HARDWARE TAGGING (2026-07-21). Appended to every RUNTAG; EMPTY default == unchanged.
# adroit's gpu partition is 3 nodes and only h11g1 carries full A100s (4 of them), so the default
# GRES=gpu:nvidia_a100:1 queues ~13 h behind pending jobs. h11g2 has 8x 3g.20gb MIG slices and
# h11g3 has 4x tesla_v100 -- both far less contended. Set TAG_SUFFIX when submitting to those so the
# run dirs do NOT collide with the A100 arms of the same name:
#   MODE=arms SMOKE_JID=none GRES=gpu:3g.20gb:1   TAG_SUFFIX=_mig bash launch_v20.sh
#   MODE=arms SMOKE_JID=none GRES=gpu:tesla_v100:1 TAG_SUFFIX=_v100 SBATCH_EXCLUDE= bash launch_v20.sh
# NOTE: --exclude=adroit-h11g3 (the default) is a NO-OP for A100 requests -- h11g3 is V100-only and
# never matched gpu:nvidia_a100:1 anyway. It DOES bite on tesla_v100, hence the empty SBATCH_EXCLUDE.
TAG_SUFFIX=${TAG_SUFFIX:-}
UPD=${UPD:-18000}                      # v1.5/v1.6/v1.9 budget (unchanged)
SAVEFREQ=${SAVEFREQ:-500}
NGATES=${NGATES:-8}

# --- v2.0 VERTICAL STRUCTURE (THE fix). VERTDROP{LO,HI} widen the sampler's per-segment vertical band
#     symmetrically (dz = -drop, so a SYMMETRIC drop band == a symmetric climb/descent band); VERTCEIL is
#     the NEW ceiling that bounds the resulting walk. Measured realised +1: 1.57% -> 30.13% (n=240k gate
#     samples). Set VERTON=0 to render the arm with the vertical knobs OMITTED ENTIRELY (the C arm) --
#     that is byte-identical v1.9 course sampling, NOT "the knobs at a neutral value". ---
VERTON=${VERTON:-1}                    # 1 = pass the vertical knobs; 0 = omit them (legacy sampler)
VERTDROPLO=${VERTDROPLO:--10.0}        # course_drop_lo  -> drop_m lo  (dz hi = +10 m climb)
VERTDROPHI=${VERTDROPHI:-10.0}         # course_drop_hi  -> drop_m hi  (dz lo = -10 m descent)
VERTCEIL=${VERTCEIL:-16.0}             # course_gates_ceiling -> gates_ceiling_m (NEW; pad-relative roof)

# --- anti-dither. ALL v2.0 arms run the v1.9 W temperament (the census winner / deploy lead). ---
YAWDITHER=${YAWDITHER:-0.6}            # W yaw_dither (v1.9 W value)
YDSTART=${YDSTART:-1.0}                # 1.0 = ON-FROM-BIRTH (warm from an already-smooth ancestor)
YDHOLDFRAC=${YDHOLDFRAC:-0.3}          # end-hold at full (inert when start=1.0; kept for parity)

# --- v1.5 YAW AMPLITUDE/DUTY + free band (W values). ---
YAWDUTY=${YAWDUTY:-0.4}                # W rw_yaw_duty
FREEBAND=${FREEBAND:-0.25}             # yaw_duty_free_band

# --- v1.6 PITCH DUTY + free band + jerk (W values). ---
PITCHDUTY=${PITCHDUTY:-0.2}            # W rw_pitch_duty
PITCHBAND=${PITCHBAND:-0.6}            # pitch_duty_free_band -- WIDE (pitch is the primary axis)
PITCHJERK=${PITCHJERK:-0.03}           # rw_pitch_jerk (shared)

# --- v1.9 ROLL RATE-DAMPING, THE SHIPPED W VALUES (the close-in roll limit-cycle fix that won the v1.9
#     census: roll signflips 2.6 -> 2.26, n_passed 5.72 -> 6.03). JERK is the PRIMARY lever; DUTY is
#     secondary with a WIDE band. NO roll ANGLE fence (backfires on the course's 60 deg turns). ---
ROLLJERK=${ROLLJERK:-0.08}             # rw_roll_jerk (v1.9 W; PRIMARY)
ROLLDUTY=${ROLLDUTY:-0.2}              # rw_roll_duty (v1.9 W; secondary)
ROLLBAND=${ROLLBAND:-0.8}              # roll_duty_free_band -- WIDE; leaves the turn-in roll free

# --- v1.5 shared anti-runaway (A,B) + yaw jerk (D) + roll-recover (CORE, all arms) ---
OVERSPEED=${OVERSPEED:-12.0}           # overspeed_abort_mps (A); fatal OOB-class above this GT speed
PROGVCAP=${PROGVCAP:-7.5}              # rw_progress_vcap_mps (B); NOT 5.0
YAWJERK=${YAWJERK:-0.05}               # rw_yaw_jerk (D); L1 |delta yaw_cmd| price
RECOVER=${RECOVER:-0.5}                # rw_roll_recover (CORE; on-from-birth, default theta0=30deg)

# --- v1.6 aperture-margin knobs. PMFINAL=1.0 == the v1.6 baseline (M4 stays reverted; the anneal is a
#     strict no-op at this value). ---
PMFINAL=${PMFINAL:-1.0}                # pass_margin_final_m
PMLATW=${PMLATW:-2.0}                  # pass_margin_lat_weight (lateral risk ~2x vertical)
PMHOLD=${PMHOLD:-0.25}                 # pass_margin_hold_frac (END-HOLD tight for the last 25%)

# --- v1.6 range-detection knobs (miner-calibrated; defaults shown for greppability) ---
DETR0=${DETR0:-2.2}; DETK=${DETK:-0.4}; DETPLAT=${DETPLAT:-0.99}
DETFAR=${DETFAR:-26.0}; DETFARP=${DETFARP:-0.79}

# --- vision cadence sub-knobs. !! VCFRAMEHZ 30.0 -> 25.0 IS THE v2.0 (B) CHANGE: the measured wire
#     vision rate. At env.dt=0.0333 (30.03 Hz control) the old 30.0 fired on 99.90% of ticks (a near
#     no-op); 25.0 is exact (6/5 -> 5 frames per 6 ticks) and drops the net fresh-fix rate 10.50 ->
#     8.75 Hz. Set VCFRAMEHZ=30.0 to reproduce the v1.9 cadence exactly. ---
VCFRAMEHZ=${VCFRAMEHZ:-25.0}
VCDETECTP=${VCDETECTP:-0.35}

# --- M1: HANDOFF SPAWN REALISM -- OFF, and it STAYS off (premise falsified; convicted on release pitch). ---
HANDOFFFRAC=${HANDOFFFRAC:-0.0}       # handoff_spawn_frac; 0.0 == OFF == byte-identical v1.6
HORANGELO=${HORANGELO:-9.5}            # handoff_range_lo_m  \_ the deploy first-lock band 10.3-11.2 m,
HORANGEHI=${HORANGEHI:-11.5}           # handoff_range_hi_m  /  widened (sample a band, not a point)
HODZLO=${HODZLO:--1.5}                 # handoff_gate_dz_lo_m  \_ gate centre MINUS drone altitude
HODZHI=${HODZHI:-1.5}                  # handoff_gate_dz_hi_m  /  +-1.5 m -> elevation ~+-9 deg
HOAGL=${HOAGL:-0.5}                    # handoff_min_agl_m; floor clearance above the pad
HOYAWJIT=${HOYAWJIT:-0.25}             # handoff_yaw_jitter_rad (== the stage's course_spawn_yaw_jitter)

# --- M2: BLIND-FLIGHT ABORT -- OFF (INERT in this env: blind_abort_rate 0.0000 on every v1.8 arm). ---
BLINDS=${BLINDS:-0.0}                  # blind_abort_s; 0.0 == OFF == byte-identical v1.6
BLINDRANGE=${BLINDRANGE:-15.0}         # blind_abort_range_m (inert while BLINDS=0.0)
BLINDGRACE=${BLINDGRACE:-0.0}          # blind_abort_grace_s (inert while BLINDS=0.0)

# --- M3: PROGRESS FRAMING MULTIPLIER + the (guard-capped) additive perception raise -- THE KEEPER. ---
FRAMEMULT=${FRAMEMULT:-1.0}            # rw_progress_frame_mult; 0.0 == OFF; 1.0 == the full [floor,1] swing
FRAMEFLOOR=${FRAMEFLOOR:-0.5}          # progress_frame_floor -- NON-ZERO is the anti-freeze guarantee
FRAMESCALE=${FRAMESCALE:-0.5}          # progress_frame_scale_rad (~ the 29.35 deg vertical half-FOV)
FRAMEEXP=${FRAMEEXP:-2.0}              # progress_frame_exponent (gaussian; gradient across the WHOLE frame)
PERC=${PERC:-0.016}                    # rw_perception -- THE MAX the farm guard allows
PERCNEXT=${PERCNEXT:-0.004}            # rw_perception_next; PERC+PERCNEXT == rw_time 0.02 (the ceiling)

# --- which arms + seeds ---
ARMS=${ARMS:-"K B"}
SEEDS_K=${SEEDS_K:-"0 1 2 3 4"}
SEEDS_B=${SEEDS_B:-"0"}
# WARM = the v1.9 W arm (v19W_s0), the census winner and the SHIPPED deploy lead
# (ego-ckpts-v19-2026-07-24): roll signflips 2.6 -> 2.26, n_passed 5.72 -> 6.03, release-dive replay
# improved on 450/451 flights. v2.0 builds FORWARD from the best flying policy. The dir carries
# actor.pth + critic.pth, the same warm shape v1.9 consumed. The _mig suffix is the hardware tag of the
# run that produced the lead (MIG slice), NOT a v2.0 knob.
# CLEAN-ADJUDICATION ALTERNATIVE: to score the vertical change from the v1.8 ancestor instead of the
# v1.9 lead, override WARM to the dir v1.9 warm-started from:
#   WARM=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed1_v18Q_s1_mig/checkpoints
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v19W_s0_mig/checkpoints}
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

# ---- vpeffs0 base recipe (VERBATIM from v1.5/v1.6/v1.9) ----
BASE_VPEFFS0="++env.ego_blur_gate=false ++env.ego_yaw_cmd_clamp_rad_s=0.7 ++env.spin_abort_anneal=true \
++env.spin_abort_scale_start=2.6 ++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 \
++warmstart_reset_logstd=true ++warmstart_reset_logstd_std=0.18"

# ---- vpef8nc 8-gate/reward/infra block (VERBATIM from v1.5/v1.6/v1.9) ----
VPEF8NC="++env.course_n_gates=${NGATES} ++env.rw_v_cap=0 \
++env.rw_cross_zero_m=0.75 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_det_steps=600 ++eval_yaw_log=true \
++save_freq=${SAVEFREQ}"

# ---- THE TWO v1 DEPLOY-FAITHFULNESS ENABLES (VERBATIM wiring; VCFRAMEHZ is the v2.0 (B) change) ----
FAITHFUL_SHARED="++dynamics.faithful_rate=true ++env.ego_vision_cadence=true \
++env.ego_vision_frame_hz=${VCFRAMEHZ} ++env.ego_vision_detect_p=${VCDETECTP}"

# ---- THE v1.5 SHARED SPEED-DISCIPLINE + YAW-QUIETNESS block (VERBATIM: A,B,D + roll-recover CORE +
#      the v1-arm-R VALIDATED recovery anneal). rw_yaw_duty + rw_yaw_dither are per-arm (extra_for). ----
V15_SHARED="++env.overspeed_abort_mps=${OVERSPEED} ++env.rw_progress_vcap_mps=${PROGVCAP} \
++env.rw_yaw_jerk=${YAWJERK} ++env.yaw_duty_free_band=${FREEBAND} ++env.rw_roll_recover=${RECOVER} \
++env.recovery_anneal=true ++env.recovery_start=0.0 ++env.recovery_hold_frac=0.3"

# ---- THE v1.6 SHARED MECHANISM block (VERBATIM; ego_cam_mount_pitch_deg=20.0 remains the ZERO-DELTA
#      re-statement of the already-baked mount). ----
V16_SHARED="++env.ego_cam_mount_pitch_deg=20.0 ++env.ego_vision_detect_mode=range \
++env.ego_det_r0=${DETR0} ++env.ego_det_k=${DETK} ++env.ego_det_plateau=${DETPLAT} \
++env.ego_det_far_start_m=${DETFAR} ++env.ego_det_far_p=${DETFARP} \
++env.pass_margin_anneal=true ++env.pass_margin_final_m=${PMFINAL} \
++env.pass_margin_lat_weight=${PMLATW} ++env.pass_margin_hold_frac=${PMHOLD} \
++env.rw_pitch_jerk=${PITCHJERK} ++env.pitch_duty_free_band=${PITCHBAND}"

# ---- THE v1.7-era SHARED MECHANISM block. Only M3 is LIVE; M1 (handoff_spawn_frac) and M2
#      (blind_abort_s) keys are still PASSED but at their OFF value 0.0, and M4 rides inside V16_SHARED
#      as PMFINAL=1.0 (reverted). Identical to v1.9. ----
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

# VERTICAL-STRUCTURE override block. vert_on=1 emits the three course keys; vert_on=0 emits NOTHING, so
# the sampler falls back to its own defaults == BYTE-IDENTICAL v1.9 course sampling (the C arm). Do NOT
# "turn it off" by passing neutral values -- there is no neutral value for a band, only absence.
vert_extra () {   # $1=vert_on
  if [ "${1}" = "1" ]; then
    echo "++env.course_drop_lo=${VERTDROPLO} ++env.course_drop_hi=${VERTDROPHI} \
++env.course_gates_ceiling=${VERTCEIL}"
  fi
}

# per-arm EXTRA builder: the ONLY per-arm axis in v2.0 is the VERTICAL block (all arms share the v1.9 W
# yaw/pitch/roll temperament + the 25 Hz vision). Offsets the vision-cadence RNG seed by the training
# seed (each seed sees a DIFFERENT reproducible detector-miss draw).
extra_for () {   # $1=SEED  $2=vert_on  ($3=coast_on -- THE ONLY v2.1 axis)
  local SEED="$1" VON="${2:-1}" CON="${3:-1}"
  # v2.1 ARM 1: the ONLY difference between arm K and arm B. Appended LAST in EXTRA so it wins.
  local COAST_TOKEN=""
  if [ "${CON}" = "1" ]; then COAST_TOKEN="++env.ego_obs_coast=true"; fi
  local ANTIDITHER="++env.rw_yaw_dither=${YAWDITHER} ++env.yaw_dither_anneal=true \
++env.yaw_dither_start=${YDSTART} ++env.yaw_dither_hold_frac=${YDHOLDFRAC}"
  echo "${BASE_VPEFFS0} ${VPEF8NC} ${ANTIDITHER} ${FAITHFUL_SHARED} ${V15_SHARED} ${V16_SHARED} \
${V17_SHARED} $(vert_extra "${VON}") ++env.rw_yaw_duty=${YAWDUTY} ++env.rw_pitch_duty=${PITCHDUTY} \
++env.rw_roll_jerk=${ROLLJERK} ++env.rw_roll_duty=${ROLLDUTY} ++env.roll_duty_free_band=${ROLLBAND} \
++env.ego_vision_cadence_seed=$((20260725 + SEED)) +init_from=${WARM} ${COAST_TOKEN}"
}

if [ "${MODE}" = "smoke" ]; then
  # -------- PHASE 1: ONE smoke, CORE=V wiring (vertical ON), UPD=100, seed 0 --------
  submit "v21smoke_s0" 0 "$(extra_for 0 "${VERTON}" 1)" 100
  cat <<'EOF'

=== v2.1 ARM-1 SMOKE submitted (UPD=100, CORE=V wiring: v1.9 W roll damping + VERTICAL STRUCTURE
    (drop -10/+10, ceiling 16) + 25 Hz vision, warm from the v1.9 deploy lead v19W_s0).
    ADJUDICATE the COMPLETED smoke (a DONE log, never a live one -- the absent-hook != inert footgun L16):
      RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed0_v20smoke_s0
      1. grep -nE 'course_drop_lo|course_drop_hi|course_gates_ceiling|course_gates_above_spawn|ego_vision_frame_hz' \
           $RUNDIR/.hydra/config.yaml   # expect -10.0 / 10.0 / 16.0 / 0.5 / 25.0 (under env:)
      2. grep -nE 'rw_roll_jerk|rw_roll_duty|roll_duty_free_band|rw_progress_frame_mult|pass_margin_final_m' \
           $RUNDIR/.hydra/config.yaml   # 0.08 / 0.2 / 0.8 / 1.0 / 1.0 (v1.9 W carried forward)
      3. python scratchpad/tb_scalars.py <event-file>   # WIRING WATCHDOG -- the NEW keys are the headline:
           env_loss/course_vert_up_frac    -- EXPECT ~0.30. PINNED ~0.016 == the vertical knobs did NOT
             take (re-check step 1). THIS IS THE ONE TO READ.
           env_loss/vert_sector_up_duty    -- the TICK-WEIGHTED twin (== the obs[10] channel the actor
             sees); the key directly comparable to the 37.6% deploy number. Expect ~0.25-0.40.
           env_loss/course_vert_down_frac  -- ~0.20 (down structure KEPT, this is not a climb bias)
           env_loss/target_detectable_duty -- the 25 Hz read; a modest drop vs v1.9 is EXPECTED
             (fresh-fix 10.5 -> 8.75 Hz). A COLLAPSE == the cadence is starving the estimator.
           env_loss/roll_jerk_pen /roll_duty_pen  -- NON-ZERO (the v1.9 fix still armed)
           env_loss/frame_factor  -- M3; warm from v19W_s0 (converged high) it should START high.
           env_loss/blind_abort_rate -- EXPECT ABSENT / 0.0 (M2 off).
      4. grep EGO_PRECHECK_RC /scratch/network/fl3689/peregrine_vq2_ego_v20smoke_s0.out  # == 0, no NaN
      5. exit_ceiling / oob_rate in the trace: the course now reaches 16 m and the OOB box top rises with
         it. A JUMP in exit_ceiling == the policy is climbing OUT rather than through. That is the main
         destabilisation risk of this change -- report it, do not tune it away.
    VERIFY THE SAMPLER OFFLINE FIRST (5 s, laptop, no cluster):
      python rl/peregrine_racing_ego.py dual_gate_fullstack_floor_pef16 course_n_gates=8 \
             course_drop_lo=-10.0 course_drop_hi=10.0 course_gates_ceiling=16.0
    THEN release the real arms (afterok the smoke):
      MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v20.sh
    ESCAPE HATCH (no dependency): MODE=arms SMOKE_JID=none bash launch_v20.sh
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
    K) for s in ${SEEDS_K}; do submit "v21K_s${s}" "${s}" "$(extra_for "${s}" 1 1)" "" "${DEP}"; done ;;
    B) for s in ${SEEDS_B}; do submit "v21B_s${s}" "${s}" "$(extra_for "${s}" 1 0)" "" "${DEP}"; done ;;
    *) echo "!! unknown ARM '${ARM}' (expected K / B) -- skipping" ;;
  esac
done

cat <<'EOF'

=== v2.1 ARM 1 submitted [K=coast B=control]. ADJUDICATION PROTOCOL (post-hoc; TAILS, NOT MEDIANS -- one strong event
    kills a flight with a perfect median, so every column below is a WORST-EVENT column):
    STEP 0 (wiring, BEFORE anything else): course_vert_up_frac ~0.30 on the V arms and ~0.016 on C. If
      they MATCH, the arms are not isolating anything -- stop and fix the wiring.
    STEP 1 (sweep): for EACH seed, run ++rollout_only over the run's periodic + best_npg snapshots ->
      YAW_EVAL (signflips_per_s + cmd_absmean + satur_duty) + PITCH_EVAL + ROLL_EVAL (same three) +
      DET_EVAL (n_passed_gates + max_speed) per ckpt.
    STEP 2 (HARD gates -- DISCARD regardless of n_passed):
      * YAW signflips_per_s > ~4.0 OR satur_duty > ~0.1 OR cmd_absmean > ~0.3 (the cmd_absmean gate is
        the one the v1 pick-flights proved transfers 1:1 to the wire).
      * PITCH / ROLL satur_duty high OR signflips_per_s elevated vs the v19W_s0 parent. The v1.9 roll fix
        MUST NOT regress -- a taller course means more pitch authority spent, watch for roll bleed.
      * max_speed must sit BELOW the 12 m/s abort WITH MARGIN.
    STEP 3 (the v2.0 reads -- ONE mechanism, TWO must-not-regress checks):
      * VERTICAL (the read): V vs C on n_passed_gates, and specifically on the UP legs. The whole point
        is that V handles a +1 sector leg that C cannot. If V is NOT better on gates while
        course_vert_up_frac confirms the distribution moved, the vertical hypothesis is WRONG -- report
        that, it is a real finding, do not tune the band until it flatters.
      * exit_ceiling / oob_rate MUST NOT rise materially on V. A taller course with an auto-scaling OOB
        box invites climb-outs; that would be the change destabilising training.
      * target_detectable_duty: both arms carry 25 Hz, so compare BOTH against the v19W_s0 parent to
        price the cadence change separately from the vertical one.
    STEP 4 (rank survivors): HIGHEST DET n_passed_gates; tiebreak LOWEST yaw+pitch+roll signflips +
      satur. NEVER select on value. Multi-seed or it does not count (~2.8x seed variance).
    MONITOR: scratchpad/tb_scalars.py on each event file (TB, NOT stdout).
EOF
