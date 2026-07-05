# VQ2 climb-approach diagnosis — run 20260705_211007_loopchoke_confirm_f1

**Status: COMPLETE (2026-07-05). All 5 questions answered; verdicts + ranked fixes below.**

Flight: turn package 1c5453b (E1 pass_refine_prefer_far, R1 max_body_rate_rps 6.0, S1 forward_accel 0.25, E2 pass_arm_range 5.5, E3 turn_point_tol 0.10). Loop 23.4Hz, async-detect 22.2fps, obs age mean 84ms. `--ignore-collisions`. fly_rl late-joined ~23s of race clock (drone idle first ~23s — excluded).

Result: STALLED, gate_index=0 (meta), collisions=200. 661 nav ticks. Operator: turn (cmds 48-76) FIXED; NEW failure = climb-approach yaw oscillation (cmds ~93-160) + forward stall + final hard-left loses gate off RIGHT edge.

## Console facts (data/runs/loopchoke_confirm_console.log)
- LATE-JOIN GO at to_go=-22.74s, sim_t=82.384s (race clock). Drone armed then.
- Console `gi` (LIVE seeker active_gate_index) went 0 -> 1 at **t=86.54s** ("gate 0 PASSED -> targeting 1"). This is operator's "gate 1".
- Console dead-reckoned pos z climbs +6.1 (86.5s) -> +48.7 (104.98s) then falls to +46.7 (109s) — large climb then plateau/chaos.
- Endgame t=108.02s rate=[+1.50,+1.50,+0.14]; t=109.05 rate=[+1.50,+1.50,+0.90] — roll+pitch rail-slam to +1.5. sim_time then resets to 0.92s (race ended) -> STALL.
- video-thread: frames=1127 max_gap=181ms, gaps>200ms=0 — NO severe frame drops this flight.
- seeker-diag: cmds=540 pursuit=437 none=103 (valid_empty=101) bridged=100. So ~19% of ticks had NO valid pose (bridged/held).

## cmd->tick mapping (ESTABLISHED)
- sim_time RESETS at nav idx 629 (109.16s -> 0.03s): the RACE portion is nav idx 0..628
  (race clock 82.42 -> 109.16s = 26.74s of flight). The final 32 ticks are post-race chaos/disarm.
- Operator's "628 distinct commands over 661 nav ticks" == the 629 race ticks. So operator
  cmd# maps ~1:1 onto nav idx BUT with a running offset (render counter runs a few behind
  tick_index because loop 23.4Hz > video 22.2fps). PHYSICAL anchoring (turn signature, yaw
  extrema) is more reliable than arithmetic. Established anchors:
  - The hard TURN (yawrate_cmd 0.9 + roll ~1.0, gate 0 pass at rng 2.9->reacquire gate1 at 17m)
    runs **nav idx 67..88** (race t 3.3-4.2s). Operator "cmd 48 start / 57 bank / 76 leveled" maps here.
  - CLIMB-APPROACH oscillation = **nav idx ~90..205** (race t 4.3-9.6s). Operator cmds 93-160.
  - So operator cmd# ~= nav_idx - ~15 to -18 (operator counter lags). Treat operator labels as
    QUALITATIVE regime markers, not exact indices.

## Q1 YAW OSCILLATION — VERDICT: az-driven ~2s limit cycle, NOT a yaw-rate-loop lag cycle, NOT R1
Config in effect (deploy_profile.vq2_case_c, src/racer/deploy_profile.py:132-691):
  pursuit_yaw_slew_rps=0.9, visual_yaw_rate_cap_rps=0.9, fwd_scale_pow=4.0, forward_accel_mps2=0.25,
  fwd_point_gate_az_rad=0.25 / full 0.05, image_kaz_mps2_per_rad=12.0, image_lat_cap_mps2=4.0,
  use_soft_bearing_weight=True, yaw_slew_taper lo=3.0/hi=10.0/floor=0.35, max_body_rate_rps=6.0,
  pass_arm_range_m=5.5, pass_refine_prefer_far=True, pass_turn_point_tol_rad=0.10.

Evidence (climb-approach window nav idx 85..210, scratchpad/phase.py + yawdes.py):
- **The oscillator is `az_err` itself** (the servo's actual control error), NOT railing yaw rate.
  az swings: idx96-108 az~-0.10 (gate left) -> idx121-141 az~+0.13 (gate right) -> idx151-174
  az~-0.20..-0.47 (gate left, growing). Full period ~2.0s (idx96->146->~195), matching operator "1-2s".
- yaw itself: -0.83 (idx96) -> -0.52 (idx119) -> -1.01 (idx146) -> -0.69 (idx165) -> -0.27 (idx184)
  -> -0.02 (idx204). Amplitude ~0.5 rad, period ~2s. This is the physical yaw the operator saw.
- **yawrate_cmd tracks az at lag 0, corr 0.87** -> the yaw-rate loop is a clean proportional
  follower of az; the limit cycle is in the az/heading GEOMETRY, not in the rate loop timing.
  So hypothesis (a) "vision-lag limit cycle in the yaw-rate loop" is NOT the mechanism.
- **Hypothesis (c) R1 freed yaw = FALSE.** |yawrate_cmd|>=0.85 only 1% of ticks; max|yawrt|=0.90
  == the visual_yaw_rate_cap_rps=0.9 per-axis cap, which is SEPARATE from and far below the R1
  max_body_rate_rps=6.0 norm ceiling. Yaw rate is NOT railing and R1 did not touch it.
- **Hypothesis (b) bearing_w flapping: MINOR contributor, not primary.** bearing_w mean 0.87 in
  window, <0.7 only 10% of ticks, dips to 0.16 at idx170-171 (detection degrade). It modulates
  the az term amplitude but the oscillation persists across high-brw ticks (idx133 brw0.97, idx146
  brw0.98). It sharpens the terminal collapse (idx170 brw0.16) but does not cause the limit cycle.
- **LOG ANOMALY (important): logged `yaw_des_rad` is INCONSISTENT with logged `az` + `yaw`.**
  az = wrap(psi_world - yaw_now) per src (gate_seeker.py:2216), but wrap(yaw_des_rad - yaw_rad)
  ~= +1.5..+1.97 rad while logged az ~= 0. yaw_des_rad is stashed from atan2(los) using
  CAPTURE-TIME attitude (gate_seeker.py:2424-2427) whereas az uses the SAME psi_world but the
  discrepancy shows yaw_des_rad as-logged is NOT directly az's psi_world reference. So: trust `az`
  as the true control error; treat `yaw_des_rad` as instrumentation in a different (capture-time)
  frame, NOT as the live setpoint. (yaw_des_rad and yaw_rad even ANTI-correlate, r=-0.41 -> do
  NOT read yaw_des_rad as "where the servo is pointing".)
- **Root mechanism (hypothesis, strong):** climb-regime GEOMETRY. The gate sits LOW in a
  camera pitched +~20deg up during the climb; az from a low-in-frame, near-edge detection is
  noisier and the capture-time attitude rotation (_rpy_at) must de-rotate a large, fast-changing
  pitch/roll. Each fresh pose nudges psi_world, yaw slews toward it at 0.9 rad/s, overshoots
  (0.9 rad/s slew * ~84ms obs age + tick lag = ~0.08 rad overshoot per swing), az flips sign,
  and the cycle repeats. The forward push is choked (fwd_scale, below) so the drone does not
  close range fast enough to escape the regime -> the limit cycle sustains for ~5s until the
  detection degrades (brw->0.16, range bounces 5.4->8.1m at idx170+) and the gate exits frame.

## Q4 gate_index LOGGING — VERDICT: prompt premise is WRONG; per-tick field tracks the wire correctly
- The **per-tick `gate_index` in nav_estimate.jsonl DID advance**: 97 ticks at 0, then flips to 1
  at **nav idx 65, sim_t=85.632s**, 564 ticks at 1. It did NOT "stay 0 all flight".
- Source: rl/fly_rl.py:1640-1647 reads `gi = int(rs["active_gate_index"])` from RACE_STATUS
  (the authoritative wire) EACH tick, sets loop-local `gate_index = max(gi,0)`, and passes that
  same value into `_nav_estimate_record(..., gate_index, ...)` (fly_rl.py:1684-1685). The record
  writes `rec["gate_index"] = gate_index` (fly_rl.py:1173). No lag: field == wire, per tick.
- The **console "gate 0 PASSED -> targeting 1" printed at t=86.54s** while the field flipped at
  85.63s. That ~0.9s gap is PURELY the ~1 Hz status-line THROTTLE (fly_rl.py:1644 prints inside
  the tick that first sees gi>gate_index, then the next 1 Hz status line at 86.54s shows gi=1).
  The PASSED print itself fires on the flip tick; the human-visible status line is throttled.
- The ONE field that reads 0 is **meta.json / perf_summary `gate_index` (the SUMMARY field)** and
  meta.race_status (`active_gate_index:0, started:false`). That snapshot is taken at flight END
  (fly_rl.py:2552 `race_status=client.race_status`) AFTER the sim RESET the race (sim_time went
  109s->0.03s at nav idx 629; console shows the reset). The sim wiped RACE_STATUS back to
  active_gate_index=0/started=false on race-end, so the terminal snapshot reads 0 even though the
  race genuinely reached target gate 1 mid-flight. => cosmetic reporting artifact of end-of-race
  reset, NOT a live logging lag. The gate-0 pass was WIRE-CONFIRMED (console gi=1 comes straight
  from rs.get('active_gate_index')).
- ACTION (optional): meta/summary `gate_index` should latch the MAX active_gate_index seen during
  the flight, not the terminal (post-reset) value, so a real gate-0 pass isn't reported as gates=0.

## Q1 (continued) — OSCILLATION ROOT CAUSE: perceived-bearing swing amplified by roll feedback
Reconstructed the TRUE seen-gate world bearing psi_world = wrap(yaw + az) (scratchpad/psiworld.py,
coupling.py). Findings:
- **psi_world (the SEEN gate bearing) itself oscillates**, std 11.8 deg around its trend -- MORE than
  the drone's own yaw oscillation (8.1 deg). The physical gate is fixed, so a swinging PERCEIVED
  bearing at ~fixed position = the detection/PnP bearing is wobbling, not just yaw overshoot.
- **Feedback signature**: psi_osc (bearing oscillation) is IN PHASE with roll_cmd (corr 0.76, lag 0)
  and LAGS the drone's yaw by ~10 ticks (corr 0.63). i.e. the drone's own roll/yaw MOVES the
  perceived gate bearing, which drives more roll/yaw. The A30 capture-time attitude de-rotation
  (_rpy_at, gate_seeker.py:2170-2190) is NOT fully cancelling ego-motion for a gate that sits LOW
  in the frame during the climb (roll rotates a low/off-axis gate horizontally in image space).
- **DIVERGENT limit cycle**: |az| grows 0.060 -> 0.090 -> 0.256 rad across the approach. Amplitude
  amplifies because the forward stall (below) removes the range-closing that would break the regime.
- **YOLO on the actual climb frames (end-anchored map, scratchpad/frames_check2.py) CONFIRMS the
  geometry**: the gate sits in the BOTTOM quarter of the image the whole approach (cy% = 75,82,92,
  78,79,81) and its horizontal center SWEEPS left->right: cx% = 61->39->30->40->60->**90** (idx204,
  hard on the RIGHT edge -- exactly the operator's "gate left the FOV off the right edge"). Frames
  also carry 3-6 detections each (1 real gate box up to 141x115 px + several 11-40px spurious/decoy
  boxes) -> nearest-bearing selection can flicker between the true gate and clutter, injecting jitter.
- **Terminal loss is NOT a runaway / wrong-sign** (hypothesis e = FALSE): at idx195-215 az is
  persistently NEGATIVE (~-0.18, gate reads LEFT of nose) with bearing_w HIGH (~0.99, gate still
  well-detected) and range stuck ~4.9 m. The final yaw-LEFT is a CORRECT commanded response to a
  gate that genuinely reads left; the drone over-yaws chasing it without translating, and the gate
  slides off the right edge as the sweep continues. A commanded response to a bad perception, not a
  sign inversion or estimator runaway.

## Q2 FORWARD STALL — VERDICT: CONSEQUENCE of the az oscillation (coupled), not independent
- a_fwd = forward_accel_mps2(0.25) * ramps * fwd_scale; fwd_scale = max(cos az,0)^4 * g_point, where
  g_point (fwd_point_gate) smoothsteps 1->0 over |az| in [0.05, 0.25] rad -> forward FULLY CUT beyond
  az=0.25 rad (14 deg). PLUS use_lateral_first_budget=True with total_accel_cap=4.0 and image_lat_cap
  =4.0: when az is large a_lat rails to 4.0 and fwd_budget = sqrt(16 - a_lat^2) -> 0. THREE forward
  killers all keyed to |az|.
- **corr(|az|, fwd_scale) = -0.92**. When az is small (early/mid, |az|~0.11) fwd_scale~0.70 and the
  drone CLOSES: range 17->11->6.4 m. When the oscillation blows |az| up to 0.265 (late), fwd_scale
  crashes to 0.10 and forward is cut -> range stalls at ~4.9 m (never reaches pass_arm_range 5.5...
  it actually DID dip below 5.5 briefly but the oscillation kept az off-axis so the pass never
  committed cleanly). commanded a_fwd fell from 0.25 max to ~0.03 m/s^2. The stall is the pointing
  gate doing its job -- but the az oscillation keeps az off-axis, so forward never recovers. Coupled
  failure: az oscillates -> forward cut -> range doesn't close -> stay in the noisy low-frame climb
  regime -> az keeps oscillating. NOT an independent forward-channel bug.

## Q3 VERTICAL — VERDICT: vertical channel worked; climb commanded correctly; no bad coupling into az
- offset_z_world starts -1.9 to -2.9 m (NEGATIVE = gate ABOVE drone, matches operator "gate is above
  the drone, needs to climb"). vz_t commands -1.0 (max climb, +down convention) sustained while the
  offset is large, then correctly tapers to 0 as the drone climbs onto the gate height by ~idx142
  (offset_z_world ~0). z_off_est tracks -3.4 -> 0 cleanly. thrust rides 0.24-0.33 around hover 0.2656.
  This is a CORRECT vertical closure -- the vertical estimator (A21/A25) did its job.
- Climb-attitude coupling into the horizontal servo: pitch_cmd stays small (+0.03..+0.17) and az uses
  the capture-time attitude de-rotation, so climb PITCH does not directly corrupt az. The coupling
  that DOES matter is ROLL (banking) into a low-in-frame gate's apparent azimuth (Q1), not the climb
  per se. Vertical is exonerated; the failure is the horizontal az/roll feedback loop.

## Supporting facts for fixes
- roll_cmd max 0.66 << pursuit_roll_rate_cap 1.5 -> roll is NOT saturating; the feedback is a
  proportional-gain loop, not a rail artifact. So lowering image_kaz / adding az smoothing directly
  attacks the loop gain without fighting a cap.
- pose_age_s mean 0.160 s (seeker's fresh-pose age; higher than perf's 84ms obs-age because it
  includes worker latency + tick spacing). At pursuit_yaw_slew_rps=0.9, 0.16s = ~0.14 rad of yaw
  travel between the perceived bearing and acting on it -> real phase lag feeding overshoot.
- theta_g_deg PnP tilt std grows 1.8 -> 3.6 deg into the terminal approach (gate low/near-edge =
  worse pose). track_range fresh-update 78% (22% ZOH holds).
- range floored ~4.8 m in the climb-approach (dipped <5.5 on 56 ticks) but the pass never committed:
  pass_turn_point_tol_rad=0.10 requires being "pointed" within 0.10 rad, and az kept swinging past
  that -> arm-but-never-commit. The oscillation directly blocks the pass.

## Q5 FIX PROPOSALS — ranked by confidence; flag-level; DO NOT revert E1/R1/S1 (turn works)
Root cause to attack: a DIVERGENT perceived-bearing/roll feedback limit cycle in the low-frame climb
regime, with the forward pointing-gate + lateral-first budget starving forward so the drone can't
close range to escape. Two independent levers: (A) damp the az/roll loop gain + smooth the perceived
bearing; (B) keep SOME forward alive so range closes and breaks the regime.

--- TIER 1 (highest confidence, smallest blast radius) ---

FIX 1 [confidence HIGH] -- Add a forward-drive FLOOR under the pointing gate so range keeps closing.
  Knob: introduce a small floor on fwd_scale (e.g. clamp fwd_scale to >= 0.20 instead of ->0), OR
  raise fwd_point_gate_az_rad 0.25 -> 0.40 so forward isn't fully cut until |az|>23 deg. Predicted
  effect: even while az oscillates ~0.15-0.25 rad, a_fwd stays ~0.05 m/s^2 -> range keeps closing;
  closing range shrinks the LOS lever so a fixed lateral image error maps to smaller bearing motion,
  damping the loop AND getting the drone to pass_arm_range for a commit. Risk: LOW -- a small forward
  floor while mis-pointed reintroduces a little of the "wide arc past the gate" the pointing gate was
  added to kill, but 0.20 floor at 0.25 m/s forward_accel is only ~0.05 m/s^2, far below the A36 arc.
  Preferred first lever because it targets the ESCAPE (Q2) with one number and does not touch yaw/turn.

FIX 2 [confidence HIGH] -- Lower the image-servo lateral gain image_kaz_mps2_per_rad 12.0 -> 6-8.
  The psi_osc<->roll lag-0 corr 0.76 shows roll directly feeds the perceived-bearing swing; kaz=12
  was raised (profile note c1) for the 8-17 deg GRAZE band earlier in the flight, but in the climb
  regime it over-rolls into a low-frame gate. Predicted effect: halves the roll response to az,
  breaking the amplification -> bounded (non-divergent) az. Risk: MED -- kaz also does legitimate
  cross-track centering; too low and the earlier graze-band centering (which the turn relies on)
  weakens. Mitigate by pairing with FIX 1 (range closes -> less centering needed). Test kaz=8 first.

FIX 3 [confidence MED] -- TIGHTEN the existing lateral-demand slew. CORRECTION: image_lat_slew_mps3
  is already ON at 9.0 (deploy_profile.py:379, turn-package c2), NOT off. At kaz=12 a 0.5 rad/s az
  ramp demands ~6 m/s^3, so the 9.0 slew barely bites -> it stops one-frame decoy SNAP hops but not
  the ~2s swing. Lower it to ~4-5 m/s^3 to also damp the sustained swing. Predicted effect: slower
  roll build-up = less ego-motion fed back into the perceived bearing. Risk: MED -- too tight adds
  phase lag to legitimate centering; couples with kaz (FIX 2) -- tune ONE at a time (prefer FIX 2
  first since it attacks the same loop gain more directly).

--- TIER 2 (bigger levers / more coupled) ---

FIX 4 [confidence MED] -- Reduce the decoy clutter feeding the bearing. Frames carry 3-6 detections
  (1 real + several 11-40px spurious). Tighten the detector min-box / confidence, or make the
  nearest-bearing selection HYSTERETIC (prefer last-frame's gate within a gate-size gate) so a
  one-frame decoy can't grab the bearing. Predicted effect: removes the jitter source at the root.
  Risk: MED -- selection changes are the historically bug-prone path (A13/A31); needs a unit test.
  Best done as a follow-up after Tier-1 confirms the loop is the mechanism.

FIX 5 [confidence MED] -- Vertical-approach yaw-slew taper: the yaw_slew_taper (lo=3 hi=10 floor
  0.35) already reduces slew at close range, but the divergence is at 5-10 m. Consider extending the
  taper hi to ~14 m or raising the floor cut so the yaw setpoint moves more slowly through the
  low-frame climb band, reducing overshoot. Predicted effect: gentler yaw = less ego-motion into the
  perceived bearing. Risk: MED -- slower yaw could let a real bearing error persist; interacts with
  the turn's need for fast yaw (but the turn is at pass range, a different band).

--- NOT recommended ---
- Do NOT revert E1/R1/S1: the turn (idx67-88) is eyes-confirmed clean and R1's 6.0 norm ceiling is
  not implicated (yaw rate rails at the separate 0.9 visual cap, 1% of ticks).
- Do NOT chase yaw_des_rad sign: it's a capture-frame instrumentation artifact (Q1 log anomaly),
  not the live setpoint; az is the true, correctly-signed error.

RECOMMENDED FIRST FLIGHT: FIX 1 + FIX 2 together (forward floor 0.20 + image_kaz 8). FIX 1 lets range
close to break the regime; FIX 2 keeps the loop from re-diverging. Both are single-number flag edits
in deploy_profile.vq2_case_c, zero turn impact. Add FIX 3 (lat slew) if the swing persists.

## Estimator caveats honored
- position_ned is dead-reckoned fiction (console pos z=+48m is not real altitude); used only for
  qualitative climb confirmation, cross-checked against operator eyes + offset_z_world.
- Operator eyes are ground truth for physical direction; telemetry (az, yaw) agrees with the operator
  timeline (yaw-left/right/left, gate low-right, final loss off the RIGHT edge -- YOLO cx% hit 90%).
- Where data refines the operator: the "yaw oscillation" the operator saw is really the drone
  faithfully CHASING an oscillating PERCEIVED bearing (not a yaw-loop instability); and the terminal
  hard-left is a correct response to a persistent left-reading gate, not a runaway.

