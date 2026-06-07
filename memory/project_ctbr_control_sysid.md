# CTBR control + inner-rate system-ID (live, ShadowPC 2026-06-03/04)

> **[2026-06-05 UPDATE — read first]** Two things resolved after this file was written:
> **(1)** The velocity-setpoint "easy-mode" was definitively RE-TESTED with a recording (not a GUI
> glance) → **World A confirmed**: velocity-in-ANGLE @ 250 Hz drives a `vz=0` setpoint to collective
> MAX and climbs away, and a corrective `vz=+2` is ignored (motors pinned); reproduced ×3,
> GUI-confirmed ANGLE. So CTBR is the right path — but now for a *recorded* reason. Full report:
> `handoff/shadowpc-velocity-fork-2026-06-04/REPORT.md` (also: the velocity controller only ENGAGES at
> ~250 Hz; a sparse 25 Hz leaves the drone on the pre-control pin).
> **(2) 🚩 ODOMETRY twist (`vx/vy/vz`) is `MAV_FRAME_BODY_NED` (8) but was stored as `velocity_ned`=world** — a
> world/body mix (sign-flipped vx/vy at the −X/−180° course heading) that fed the KF + controller
> damping + planner. **FIXED in `c3b5a8e`** (rotate body→world via `frames.world_vec_from_body_quat`,
> +regression test). **CONFIRMED root cause of §5's "KF velocity lags 4×"** (re-validated 2026-06-05
> offline replay: the mixed measurement read 0.21× truth = the "−0.22 vs −0.92"; the fixed KF tracks
> truth to 0.05 m/s; rotation validated to ±12° roll/±20° pitch). The "control on raw given velocity"
> workaround read the SAME corrupted field — a phantom fix. §1 roll + §7 balloon are SEPARATE, still real. The "altitude balloon"
> (open blocker below) is the SAME broken vertical auto-thrust that ignores velocity setpoints.
>
> **[2026-06-07 UPDATE — supersedes the §7 "altitude balloon" blocker below]** The balloon/limit-cycle
> is OURS (Task 3: no sim auto-thrust in CTBR), and the live VERIFY rung-1 cycle is now ROOT-CAUSED +
> FIXED. The live alt loop damped `kd_alt·(vz)` on the **KF (Navigator) vz, which LAGS** (~0.8 m/s in a
> sustained descent) → a relay limit cycle at a static hover (true vz ±0.5, ~6 Hz). The course is a
> **26 m DESCENT** so the alt loop NEEDS kd_alt damping to track the drop, and — PROVEN by a joint
> sweep — with the lagged KF vz **no (kp_alt,kd_alt) both threads the descent AND holds hover** (they
> conflict through kd_alt). **FIX: damp on RAW given vz** (`fly_vq1` default now; `--alt-kf-vz` = A/B
> back to KF). Re-tuned `FAITHFUL_TUNED_GAINS` **kp_alt 4.0→3.0, kd_alt 2.0→1.75** then threads
> (0.44–0.59) AND holds. `hover_thrust 0.2656` VALIDATED by the two-sided climb+sink vprobe (hover
> 0.265–0.267, vertical drag ~0; the rung-1 0.32 mean thrust was a relay clip-duty artifact). Re-derive
> with `scripts/fit_vertical.py` + `scripts/twin_hover.py`. Detail: `handoff/shadowpc-verify-2026-06-06/`.

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
- **⚠️ REVISED for ROLL — these open-loop signs were measured THROUGH the inverted ODOMETRY-quat
  roll, so they're self-consistent open-loop but give POSITIVE-feedback closed-loop lateral
  control.** The flight-correct ROLL set (Gate-0 saga §1 below) is **`odo_att_sign=[-1,1,1]`,
  `odo_rate_sign=[-1,-1,1]`, `body_rate_sign(rate-sign)=[1,1,-1]`**. Pitch/yaw unchanged.
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

Latest fly command (gate-0, all fixes below): `scripts/fly_vq1.py --mode body_rate --decoupled
--tilt-comp --force-saved-map --yaw-mode course --gate-corner-to-center --hover-thrust 0.27
--ff-gain 2.6 --odo-att-sign=-1,1,1 --odo-rate-sign=-1,-1,1 --rate-sign=1,1,-1 --kp-alt 0.08
--kd-alt 0.10 --alt-thrust-lo 0.20 --alt-thrust-hi 0.42 --alt-offset 0.2 --kp-att 1.6 --kd-att 1.1
--max-body-rate 2.0 --max-speed 1.5 --kp-pos 1.0 --kd-vel 2.5 --max-accel 2.5 --rate 100
--max-gates 1 --takeoff-alt 1.4 --lookahead 2.0 --gate-radius 0.75 --geofence-m 30 --max-climb-m 6`.
(Note `=` syntax for negative-leading args or argparse errors.) Vision off → GIVEN pos/vel.

## Gate-0 saga (2026-06-04) — bugs FOUND & FIXED, one OPEN. Lateral SOLVED, altitude NOT.
A long teammate-in-the-loop debug (the teammate watching the GUI is ground truth — telemetry
"passes" fooled us twice). Each fix is committed + tested (265 green). In order of discovery:

1. **🚩 ODOMETRY-quaternion ROLL is INVERTED** (the big one). The lateral loop was POSITIVE
   feedback: drifting +y, the decoded roll grew +, but at yaw≈-175° a +y accel physically needs a
   NEGATIVE roll → every "correction" reinforced the drift (runaway to +16 m). The rate sysid had
   measured roll THROUGH the inverted quaternion, so the true sim gain is **+2.73** (not -2.73). Fix
   = flip ALL THREE roll conventions together: **`odo_att_sign=[-1,1,1]`** (new attitude-feedback
   sign param; R_cur uses the true roll so the loop settles), **`odo_rate_sign` roll → -1**, and
   **`body_rate_sign` (rate-sign) roll → +1**. Pitch/yaw already worked, untouched. A single-step
   check (kp only) confirms +y drift → corrective -y accel. This turned the runaway into a stable
   (converging) loop. [Likely the SAME family as the pitch/rate inversions; suspect roll on the
   sim quaternion broadly.]

2. **Gate map position = the gate's BOTTOM-CENTRE**, not a corner (teammate: "gate is directly
   ahead, no lateral move needed, just up"). So the corner→centre offset is **VERTICAL ONLY**: lift
   half the gate height (col2 of the true quaternion) UP, keep the mapped y. gate0
   (-23.30,-0.40,-0.03) → centre **(-23.30,-0.40,-1.39)**. (An earlier +width offset was WRONG and
   flew it into the side panels.) `gates_from_track_records(corner_to_center=True)` /
   `fly_vq1 --gate-corner-to-center`.

3. **Gate FRAME from the TRUE quaternion, not the course segment.** All 6 gates face **-X**
   (quaternion col1; col0=+width=+y, col2=+height=+z-down — identical quat on every gate). The
   segment-derived frame faced along the curving COURSE PATH; its tilted "gate axis", extended back
   to the start, sat ~1.7 m to the side, so the cross-track controller DETOURED sideways then
   oscillated. Quaternion normal → straight -X axis → gate dead ahead.

4. **LIVE TRACK_INFO map is UNRELIABLE** — the chunked reassembly intermittently corrupts (parsed
   gates km away; another time placed one near the start → a bogus takeoff "pass"). The course is
   DETERMINISTIC, so trust the **saved** map: accept live only if every gate is within 3 m of saved,
   else fall back; `--force-saved-map` skips live entirely. (Saved gate0 = -23.298,-0.400,-0.032.)

5. **KF velocity LAGS the truth ~4×** (during a lateral move the drone truly did -0.92 m/s, KF
   reported -0.22) → cross-track damping 4× too weak → growing lateral oscillation. **[2026-06-05:
   this is very likely NOT lag but the ODOMETRY body-frame velocity bug — `velocity_ned` interleaved a
   sign-flipped body-frame velocity into the KF measurement at the −180° heading, so contradictory
   measurements partially cancelled to a small magnitude (−0.22 vs −0.92). FIXED `c3b5a8e`; re-run this
   measurement against the fix.]** **Control on the
   raw GIVEN horizontal velocity** (pristine), not the KF estimate. This KILLED the lateral
   oscillation (y stayed ~0). fly_vq1 does this per-axis by default (`--use-kf-state` to opt out).

6. **Detector false-pass:** plane-crossing only checked "past the plane + inside the inner square",
   so a drone on the gate AXIS but far away (on the start line) false-passed when the velocity-based
   through-sign flipped at takeoff. **Fix:** also require being within `gate_pass_depth_m` (2 m) of
   the gate plane. (Plus the earlier `gate_pass_radius_m` 1.8→0.75 proximity fix.)

7. **⛔ OPEN BLOCKER — altitude BALLOON / sim auto-thrust.** During the RUN the drone climbs away
   (z 0 → -8 m, accelerating) and aborts. Root mechanism: when our collective dips toward the
   thrust floor, the **sim's auto-thrust takes over and climbs** (the same "pos/vel auto-thrust
   climbs away" quirk) → a runaway: low thrust → auto-climb → vz grows → vz-damping pins thrust at
   the floor → climb. Runs that held altitude (e.g. gate0_roll1) kept thrust right at hover (~0.265,
   range 0.22–0.27). Using the KF (gently-lagged) vz for the VERTICAL channel helped (held to
   z≈-1.3 at x≈-6) but still ballooned later. **NOT solved.** This is the one thing blocking a
   gate-0 thread — lateral + center + frame + map + false-pass are all fixed.

## Next levers for the altitude balloon (resume here)
- Keep collective ABOVE the auto-thrust trigger AT ALL TIMES (it lives ~0.22–0.25). Raise
  `alt_thrust_lo` to ≳hover-0.02 (~0.25) so thrust can never dip into the trigger band; verify the
  drone can still descend enough to hold.
- Find the exact trigger: a thrust sweep on a RUNNING race (hold level, step collective down) to map
  where the sim flips to auto-thrust. Then floor just above it.
- Gentler takeoff so no upward momentum to fight (it built ~-1.2 m/s climb at thrust 0.52): lower
  `alt_thrust_hi`/`kp_alt`, or skip the separate takeoff (takeoff-alt 0, let the carrot z lift it
  during forward flight).
- An alt INTEGRATOR (not yet present) to hold hover precisely without the vz-damping cratering thrust.
- Then: drop `--max-gates`, raise speed, fly the full 6-gate VQ1. (Gate centres for the course are
  now correct via corner_to_center: gate1 (-46.9,-2.5,3.71), gate2 (-74.6,1.2,12.31), … all face -X.)
