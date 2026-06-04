# CTBR control + inner-rate system-ID (live, ShadowPC 2026-06-03/04)

The flyable control stack for VQ1: how the sim's inner loop actually behaves, the plant-matched
decoupled CTBR controller built on top, and the HONEST gate-0 status. Supersedes the
"Fly on CTBR" bullet in [[reference-sim-interface]]; raw run data in `data/runs/*_gate0_*`,
`*_rate*`, `*_hsweep*`. Branch `red-team-tier-a`, **258 tests green**, `.venv` Python 3.13.

## The sim is ACRO-only — easy-mode is CLOSED (exhaustively tested)
- Sim boots **ANGLE**, but **ANY client traffic flips it to ACRO** — a heartbeat OR a timesync is
  enough. The GUI mode is **not** reflected in `HEARTBEAT.base_mode` (always MANUAL|CUSTOM,
  custom_mode=0), so you can't read the mode off the wire — only the teammate's GUI shows it.
- In every mode the **velocity/position auto-thrust is broken**: a pos/vel setpoint LEVELS the
  attitude (ANGLE's stabilizer works) but **climbs away** (thrust ignored → 57 m runaway).
  Conclusion: **WE must own thrust.** Probes: `scripts/mode_watch.py`, `mode_escape.py`,
  `angle_test.py`, `angle_vel_test.py`, `angle_att_test.py`. Do NOT re-litigate easy-mode.
- Therefore: **CTBR only** — `SET_ATTITUDE_TARGET` with the body-rate ignore mask, collective
  thrust explicit. Countdown is live: **controlling before GO = DQ**. Reset = teammate home→Race.

## Inner-rate system-ID (the CTBR plant) — MEASURED, do not rediscover
- **`body_rate_sign = [-1, 1, -1]`** — the COMMAND→true-angle sign. **Roll and yaw are inverted;
  pitch is NOT.** (Disproved the teammate's [-1,-1,-1]: +pitch cmd → nose-up, and forward flight
  needs negative pitch-rate, matching the reference's `PITCH_RATE=-0.3`.)
- **`odo_rate_sign = [+1, -1, +1]`** — sign of the **ODOMETRY `angular_rate`** vs the true attitude
  derivative. **Only pitch's rate is inverted** (same quirk as the ATTITUDE Euler pitch). Multiply
  the measured rate by this BEFORE the `-kd` damping term, else **pitch damping becomes
  ANTI-damping → tumble** (this was the "hsweep2 runaway to 77° in 0.4 s" bug).
- **Inner-loop STEADY rate gain ≈ 2.7×** (commanded rate → realized rate, signed):
  roll **-2.73**, pitch **+2.68**, yaw **-2.38**. Undo it with **`ff_gain ≈ 2.6`** (a divisor on
  the attitude→rate law) → unity-gain attitude loop. (This is STEADY gain, not the transient.)
- **Hover thrust ≈ 0.26** (CTBR normalized 0..1).
- **`body_rate_from_quats(q_prev, q_cur, dt)`** (`frames.py`) = `(R_prev⁻¹·R_cur).as_rotvec()/dt`
  — a sign-correct finite-difference body rate from consecutive ODOMETRY quaternions that
  **sidesteps the inverted ODOMETRY rate** entirely. Used by the sysid probe's trusted-rate channel.
- Tooling: `scripts/rate_sysid.py` (open-loop ± body-rate doublets / hover thrust sweep, respects
  countdown, hard-abort + force-disarm), `scripts/analyze_sysid.py` (fits gain via
  `sysid.fit_rate_gain`), `scripts/analyze_run.py` (offline pos+attitude timeline from a tlog).

## The decoupled CTBR controller (`controller.py`, `decoupled=True`)
Plant-matched cascade, fully unit-tested:
- **VERTICAL** — altitude-hold → collective: `thrust = hover + kp_alt·(z−z_t) + kd_alt·(vz−vz_t)`,
  clamped `[alt_thrust_lo, alt_thrust_hi]`. WE own thrust (sim auto-thrust climbs away). NED z+=down.
- **`tilt_comp`** (2026-06-04) — divide that thrust by `cos(roll)·cos(pitch)` (= R[2,2], the
  world-vertical fraction of body thrust), floored at 0.5 (60°). Without it, leaning forward to fly
  silently sags the vertical thrust; the soft alt-PD over-corrects and the drone **balloons UP into
  the gate** (measured 2 m climb in gate0_dec4). With it, z held **rock-flat at 0.9 m** through a
  6 m/s pass (gate0_tc1). No-op at level, so it doesn't perturb hover.
- **HORIZONTAL** — `max_speed` **velocity-targeting**: `des_vel = clip(kp_pos·err, max_speed);
  a_h = kd_vel·(des_vel − v_h)`. Caps SPEED instead of clamping the position error, so the position
  pull can't dominate and overshoot cruise (legacy PD flew 5–6 m/s open-loop). `None` ⇒ legacy PD.
  Vertical is zeroed here (alt-hold owns collective); a_h → tilt DIRECTION via `_accel_to_attitude`.
- **ATTITUDE** — `omega = (kp_att·rotvec(R_curᵀ·R_des) − kd_att·(rate·odo_rate_sign)) / ff_gain`,
  `_clip_norm` to `max_body_rate_rps`, then `· body_rate_sign`.
- **`f_up<0` dive fix** — `_accel_to_attitude` cuts throttle (`f_mag=0`) when the demanded accel
  dives faster than g, instead of the old skyward push.

Canonical fly command (the gate-0 approach config): `scripts/fly_vq1.py --mode body_rate
--decoupled --tilt-comp --hover-thrust 0.26 --ff-gain 2.6 --odo-rate-sign 1,-1,1 --kp-att 1.0
--kd-att 1.0 --max-body-rate 2.0 --max-speed <v> --kp-pos 2.0 --kd-vel 2.5 --max-accel 4.0
--rate-sign -1,1,-1 --max-gates 1 --takeoff-alt 0.0 --geofence-m 30 --max-climb-m 6`. Live vision
is unavailable → fly the GIVEN pos/vel (KF predict in-loop, vision seam intact).

## Gate-0 status — HONEST (NOT yet threaded)
- **Stable, collision-free controlled flight to the gate plane is achieved.** gate0_tc1 flew the
  ~23 m to gate 0 with **zero collisions, peak body rate 0.45 rad/s**, altitude held flat — a big
  step up from the early tumbles/balloons. tilt-comp + velocity-targeting both proven in the air.
- **But it MISSES the opening.** It crosses the gate plane at **(-23.3, +0.9, +0.9)** vs gate
  centre **(-23.30, -0.40, -0.03)** → **+1.3 m lateral (+y) and +0.9 m low**. The ~1.5 m inner
  opening is ±0.75 m, so that's a clean miss (visually confirmed by the teammate).
- **⚠️ The gate location is NOT obfuscated — VERIFIED.** Live `TRACK_INFO` recovered from the
  gate0_tc1 tlog == the saved map (`track_map_20260602_114630.json`) == **(-23.298, -0.400,
  -0.032)**, two captures days apart agreeing to the mm. The course is deterministic and honest.
- **The earlier "gate_index=1 pass" was a FALSE POSITIVE** of the detector: `gate_pass_radius_m`
  proximity sphere was **1.8 m > the 0.75 m half-opening**, so a 1.58 m-from-centre miss scored a
  pass. **FIXED**: `MissionConfig.gate_pass_radius_m` and `fly_vq1 --gate-radius` defaults dropped
  to **0.75**; the honest fast-pass test is path (2), plane-crossing INSIDE the inner square.
  Always trust path (2), not the proximity backup.

## Next levers (to actually thread gate 0, then the course)
1. **z-droop +0.9 m** (steady-state sink): raise `kp_alt` 0.02→~0.06 or nudge `hover_thrust`
   0.26→0.27. tilt-comp already kills the dynamic balloon; this is just the static offset (no
   integrator yet — an alt integral term would zero it cleanly).
2. **+y drift +1.3 m** is **SPEED-PROPORTIONAL** (grows as the drone accelerates; ~0 at low speed).
   Most likely a soft yaw-hold (effective yaw-P ≈ kp_att/ff_gain ≈ 0.38) letting yaw drift ~5°
   under a forward-flight torque, or a lateral body-coupling. Levers, in order: (a) slow the
   approach to ~2.5 m/s (halves the drift, buys correction time), (b) stiffen lateral `kp_pos` /
   add a dedicated `kp_yaw`, (c) a dedicated lateral-coupling probe if (a)/(b) don't close it.
3. Then drop `--max-gates`, raise speed, fly the full 6-gate VQ1.
