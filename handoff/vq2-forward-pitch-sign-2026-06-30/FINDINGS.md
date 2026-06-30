# VQ2 forward-pitch-sign disambiguation — RESULT: **BRANCH A** (forward chain correct) — 2026-06-30

**Question (from the A10 blocker):** the gyro fix is validated and the estimate is trustworthy, yet when
the seeker commands FORWARD the drone pitches NOSE-UP and drifts BACKWARD, losing the gate. Offline the
controller is PROVABLY correct (forward accel -> negative / nose-down pitch cmd). So is the live inversion
(1) a gyro/attitude PITCH polarity error, or (2) a command-axis `body_rate_sign[1]` error? This probe
isolates the forward feedforward and picks exactly one.

## Verdict: **A — the forward-feedforward chain is CORRECT; it is NOT a sign.**
Pilot (Fengyou), clean-start run: *"the drone flies forward ... tilting forward [nose-down] ... it never left
the ground, it was scraping against the ground flying forward ... we run into the bottom of the first gate."*
Forward translation + nose-down (forward) tilt, matching the estimator and the command.

**Confound noted (does not change the verdict):** the drone never lifted off — it scraped forward along the
floor (see the secondary finding for WHY). So this is not a clean free-flight test. But the DIRECTIONAL
result is robust to ground contact: a forward command produced a forward TILT and forward MOTION. An inverted
pitch sign (either candidate) would have tilted the nose UP / pushed it BACKWARD (the A10 signature). It did
the opposite. Ground drag cannot invert tilt direction. A clean airborne re-confirmation is available (add a
climb phase before the inject) if belt-and-suspenders is wanted, but the sign question is answered.

| candidate | ruled out by |
|---|---|
| **B-gyro** (attitude/gyro pitch inverted) | Physical motion = FORWARD and the estimator reads nose-DOWN — they AGREE. If the attitude were inverted the estimator would read nose-down while the drone flew backward/up. It flew forward. ✗ |
| **B-command** (`body_rate_sign[1]` inverted) | The controller's FIRST response to the forward demand was `cmd_pitch = -0.77` (nose-DOWN), and the drone pitched down + flew forward. Command polarity tracks the estimator correctly. ✗ |
| **A** (chain correct; A10 = regime/acquisition) | **CONFIRMED.** Bypassing the seeker's regime logic and feeding the controller a pure forward `accel_ned` flies the drone forward, nose-down, exactly as designed. |

## How the probe drove it (full deployed stack, NO vision in the loop)
`forward_pitch_probe.py`: real `make_seeker_controller` signs (`body_rate_sign=[1,1,-1]`, `odo_att_sign=[-1,1,1]`,
`odo_rate_sign=[-1,-1,1]`), `cmd_rate_scale=0.4`, `gyro_sign=(-1,-1,-1)` — all pulled from `vq2_case_c`.
Warmup = the seeker's own settle/anchor HOLD (frame=None) to a stable attitude; then ~3 s of a pure
forward demand `accel_ned = 8 * [cos yaw, sin yaw, 0]` fed straight to `seeker.controller`. (A first run at
3 m/s² NULLED — the ~17deg forward-lean target coincided with the resting tilt; 8 m/s^2 -> ~39deg target
clears it.) Clean-start run, VQ2 confirmed (red-glow 3.3k->120k->230k px = lit warehouse, not VQ1 wireframe).

### Clean-start inject trace (run `20260630_090831_vq2_fwd_pitch_clean`)
```
 t(s) phase   cmd_pitch  est_pitch   note
 0.2  warmup   +0.000     -17.3      resting tilt, held rock-solid (zero rate)
 2.9  warmup   +0.000     -16.8
 3.0  inject   -0.767     -22.2      forward demand -> FIRST cmd is NOSE-DOWN (correct)
 3.4  inject   -0.040     -38.4      nose driving down toward the -39deg forward-lean target
 3.6  inject   -0.005     -39.1      reached target, cmd ~0
 3.9  inject   +1.350     -54.8      overshoot past target -> cmd flips NOSE-UP to correct it
 4.1  inject   +1.600     -62.2      peak overshoot
 5.0  inject   +0.440     -45.8      regulating back up toward -39
 5.9  inject   +0.116     -41.0      settled near target; closed loop, consistent
```
A textbook closed loop **in the estimator's frame**, and the pilot confirms the estimator's frame == reality
(forward + nose-down). The summary line "cmd_pitch mean +0.39 NOSE-UP" is an artifact of averaging the
overshoot-correction phase; the *initiating* command (t=3.0, −0.77) is the diagnostic one.

## What this means for the A10 blocker (commander's next target)
The controller, the attitude estimate, the gyro sign, and the command-axis sign are **all correct**. The
A10 nose-up-and-retreat is therefore **regime / acquisition**, upstream in `gate_seeker.command_visual`:
1. **Egress emits a raw nose-UP rate.** A10 logged the egress commanding `+1.50` pitch (nose-up). That is
   the *opposite* of the forward feedforward this probe validated — the egress is a different code path that
   does not go through the `accel_ned` forward-tilt geometry. Trace why egress commands +1.50 nose-up
   (should be ~level or a gentle nose-down creep).
2. **Pursuit never engages.** The forward feedforward (`_feedforward_command` -> `Setpoint.accel_ned`) is
   correct (proven here), but in A10 `cmd_rate ~= 0` for ~95% of ticks — the seeker sat in egress/hold and
   never transitioned into pursuit. Find the egress->pursuit gate (and re-check the gate-bearing `los` sign
   after the full gyro negation, which also flipped yaw — flagged in the gyro-probe handoff).
**Do NOT flip `gyro_sign` or `body_rate_sign[1]`.** Both are correct.

## Secondary finding: the warmup hold NEVER LIFTS OFF (probe-method limitation, real cause)
Pilot: *"it never left the ground, it was scraping against the ground flying forward."* The drone never got
airborne — it dragged along the floor and hit the BOTTOM of gate 1. Cause: the probe's warmup uses the
seeker's **settle/anchor HOLD**, which holds a bounded hover thrust (~0.37 observed) and has **no egress
thrust floor**. Liftoff in normal seeker flight depends on the **egress** regime's thrust floor + elevation
guard (the A7/A8 spawn-collapse fix) — the hold regime alone does not clear the start gate / leave the floor.
So this is a limitation of how the probe reached its "hover," not a new alt-hold bug. It does, however,
reinforce point 1 below: the egress regime is load-bearing (for liftoff) AND is the regime that emits the
bogus +1.50 nose-up rate. For a clean airborne re-test, add a brief CLIMB phase (thrust > hover for ~1 s)
before the inject so the pitch response is measured in free flight, not against the ground.

## Artifacts
- `forward_pitch_probe.py` — the probe (full deployed stack, pure-forward inject, onboard-video recording).
- `probe_log.json` — per-tick cmd_pitch (wire) / raw_gyro_y / gyro_y_corrected / est_roll/pitch/yaw / accel_demand.
- Recording (gitignored, on ShadowPC): `data/runs/20260630_090831_vq2_fwd_pitch_clean/` incl. `onboard.mp4`
  (2x slow-mo, gate overlay). Render: `python scripts/render_vision_video.py <run_dir> <out.mp4> --slowmo 2`.
- Flight stack untouched; handoff dir only.
