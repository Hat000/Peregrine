# Start-transient fix: takeoff→RUN launch ramp (sim build 1.0.3364 recovery)

**ShadowPC, 2026-06-09.** Recovers the banked VQ1 6/6 that build **1.0.3364** broke. The plant is
PROVEN identical (prior session) — this is purely the **closed-loop start transient (issue ①)**,
hardened the *right* way (a robust ramp, not a re-roll of the tick-timing dice). Offline work +
unit tests are COMPLETE and green; **live re-fly on 1.0.3364 is the remaining step** (needs a fresh
race GO from the GUI).

## 1. Root cause — the offending command (DIAGNOSE)

The takeoff→RUN handoff is a **one-tick attitude STEP**. In `Mission.step` the moment altitude is
reached, state flips TAKEOFF→RUN and falls through **in the same tick** (`mission.py:99-113`):

- TAKEOFF setpoint: `position = directly above start`, **no velocity**, `yaw = nav.yaw` → the
  decoupled controller demands ~0 horizontal accel → **level hover**.
- RUN setpoint (planner): `position = carrot ~25 m ahead`, `velocity = cruise·los`, `yaw = course ≈ π`.

In `Controller._decoupled_body_rate` (`controller.py:281-298`), with `max_speed` set and `sp.yaw`
given, the along-track desired speed is `clip(kp_pos·(err·ad), ±max_speed)`; the 25 m carrot
saturates it to `max_speed`, so **a_h jumps 0 → kd_vel·max_speed (≈12 m/s², capped by max_accel)** in
ONE tick. `_accel_to_attitude` turns that into a desired attitude tilted ~45–50° (clamped) toward the
gate. The attitude-error term `omega = kp_att·rotvec(R_curᵀR_des)/ff_gain` therefore **steps from ~0
to a large value**, and the underdamped rate loop (τ≈0.019 s, rate_gain≈2.5) overshoots it. Live on
1.0.3364 the realized **roll** rate hit **9.19 rad/s @ t+1.22 s** (clamp 8) → tumble; at 50 Hz the
under-actuated start sinks under gate 1.

**Why ROLL specifically** (memory's prime suspect, now CONFIRMED offline): the forward-lean demand is
in **world −X**, but the drone spawns at **yaw≈−180° not perfectly aligned** with the course axis. A
world −X accel projects partly onto **body roll** when the body is yawed off −X, and the yaw error
itself couples into the rotvec roll command. See the yaw-coupling sweep below: realized roll grows
monotonically with the spawn-yaw offset.

## 2. The fix — a time-based launch ramp (FIX, offline-twin-first)

A linear **0 → 1 authority ramp over `launch_ramp_s` (default 0.6 s)** applied at RUN entry, **keyed
off the sim CLOCK** (not tick count), scaling the controller's commanded horizontal accel (hence the
desired TILT):

- `Setpoint.launch_ramp` (new field; `None` = full authority, fully backward-compatible) —
  `contracts.py`.
- `Controller._decoupled_body_rate` multiplies `a_h *= clip(launch_ramp, 0, 1)` right before
  `_accel_to_attitude` — `controller.py`. At ramp=0, `a_h=0` → **level** desired attitude = the
  current post-takeoff attitude → **~zero rate command**, regardless of tick phase. Vertical
  (alt-hold) authority is **untouched**, so the climb is never starved.
- `Mission` anchors `_run_start_ns` on the first RUN tick and ramps `clip((sim_time −
  run_start)/launch_ramp_s, 0, 1)` — `mission.py`. Sim-clock keying is what makes it **rate- and
  tick-phase invariant** (same wall-clock fraction applied no matter when/how often we tick).
- `fly_vq1.py`: `--launch-ramp-s` (default = MissionConfig 0.6); recorded in run meta.

(a) NEVER saturates the clamp, (b) always keeps fwd+vert authority (vertical untouched; forward
ramps in over 0.6 s), (c) phase-invariant by construction (sim-clock keyed).

## 3. Offline twin evidence (faithful plant + 3 decoupled clocks)

New harness `scripts/twin_launch_phase_sweep.py` co-simulates what `twin_fly_course` cannot: **physics
(500 Hz), pose stream (85 Hz, +20 ms), control (N Hz, +20 ms, phase offset)** — the ~40 ms loop delay
and the >pose-rate control re-issuing on stale state. Metric = peak transient over the first 3 s of
RUN. (Saved: `sweep_full.txt`.)

**Yaw-coupling (100 Hz) — the roll-spike mechanism + the kill:**

| spawn yaw | RAMP OFF realized roll | RAMP ON realized roll |
|----------:|-----------------------:|----------------------:|
|   −30°    | **2.32**               | 0.38 |
|   −20°    | 1.67                   | 0.27 |
|   −10°    | 1.02                   | 0.23 |
|    0°     | 0.36                   | 0.23 |
|   +20°    | 0.95                   | 0.23 |
|   +30°    | 1.61                   | 0.26 |

Ramp OFF: roll grows with the launch yaw offset (the live mechanism). Ramp ON: **flat ~0.23–0.38
rad/s, geometry-independent.**

**Phase × rate (50/100/200/250 Hz × phase 0/0.25/0.5/0.75):**
- RAMP OFF: peak |ω| = **3.14 rad/s** at every cell (the clean twin step — the twin under-models live
  latency ~25%, so it shows the mechanism but not the full live blow-up to 9.19/clamp).
- RAMP ON (0.6 s): peak |ω| = **0.79–0.85 rad/s** at every cell — clamp margin 2.5× → **~9.4×**, and
  **invariant to both rate and phase.**

**6/6 not regressed:** full-course threads 6/6 with the ramp on; worst in-plane miss is
neutral-to-better vs ramp off (e.g. 100 Hz/phase 0: 0.50 m on vs 0.58 m off). (The absolute misses
here run higher than the no-delay tune because this harness deliberately stacks the full 40 ms delay +
85 Hz pose staleness; treat it as a regression check, not an absolute predictor — g1 cross-track is
the known-marginal VQ2 target, present with the ramp OFF too.)

## 4. Control rate (Task 3)

`--faithful` now **defaults to 100 Hz** (the validated rate the banked 6/6 ran); bare default stays
50. ≥200 Hz tested in the sweep (start transient is flat there too). Rate alone does NOT fix the
transient (ramp off is 3.14 at every rate) — **the ramp is primary; higher rate is the robustness
aid**, exactly as expected.

## 5. Tests

+6 unit tests (controller ramp scaling / monotonic / clamp; mission ramp-vs-sim-time / rate
invariance / disable). **Full suite 376 green** (was 370).

## 6. LIVE VERIFY on 1.0.3364 — 🏁 6/6 RECOVERED, DETERMINISTIC (Task 4/6)

Flew `fly_vq1.py --faithful --force-saved-map --gate-corner-to-center --finish-hold-s 1.0
--sim-build 1.0.3364` (⇒ 100 Hz + 0.6 s ramp + finish-hold) **4 consecutive times** on build
1.0.3364. **ALL FOUR = CLEAN 6/6**, self-certified by `scripts/race_outcome.py` (every gate
PASS-CLEAN, finish_hold captured the terminal RACE_STATUS each time):

| run | recording | sim-recognized time | outcome |
|----:|-----------|--------------------:|---------|
| #1 | `20260610_031258_vq1_ramp`  | 35.13 s | CLEAN 6/6 |
| #2 | `20260610_031410_vq1_ramp2` | 35.46 s | CLEAN 6/6 |
| #3 | `20260610_031509_vq1_ramp3` | 34.26 s | CLEAN 6/6 |
| #4 | `20260610_031604_vq1_ramp4` | 35.38 s | CLEAN 6/6 |

The takeoff→RUN handoff was smooth every run (z dipped to ~−3 to −4 m then settled to the gate
line ~−1.3 m; **no tumble, no collision, zero aborts**). Times are tight (34.3–35.5 s), matching the
banked ~35.3 s. The spawn heading came up as both −π (#1–3) and +π (#4) — same heading, opposite
wrap — and both launched clean, exactly the phase-robustness the ramp guarantees. **The start is no
longer a dice-roll: build 1.0.3364 deterministically clears it.** Rate finding stands: 100 Hz (the
pinned `--faithful` default) + the ramp = repeatable 6/6; rate alone never was the fix.

Fallback (Task 5) NOT needed — the ramp at 100 Hz recovered the banked VQ1.
