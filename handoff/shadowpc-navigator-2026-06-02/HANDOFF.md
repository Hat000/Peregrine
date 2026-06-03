# Handoff — ShadowPC NAVIGATOR session (2026-06-02)

Written by the ShadowPC Claude session that built `racer/navigator.py`, wired it into
`Mission.run`, and prepared the first live VQ1 fly. Separate record so the laptop session owns
`memory/` consolidation — fold what you want. Sibling note: `handoff/shadowpc-firstcontact-2026-06-02/`.

## TL;DR
Built the navigator (SENSE -> ESTIMATE) + a live-fly script that wires it into `Mission.run`,
recording both streams. **229 tests green** (216 + 13 new). Committed + pushed on
`red-team-tier-a` (`58d3bc7` navigator+tests, `0bf1936` fly_vq1). The fly script's plumbing is
validated against the live (home-page) sim. **The actuated flight is still PENDING** — it needs
the sim driven home->race (GUI, human) — and **live vision is BLOCKED** (weights + ultralytics
missing; see below), so the first flight is GIVEN-STATE, which is the prescribed safe order anyway.

## What was built
**`src/racer/navigator.py`** — `Navigator` (per-tick brain) + `NavigatorConfig` + map loader.
- **Seam decision:** the navigator is the FRONT of the chain and returns a fused `NavState`;
  `Mission.step` keeps THINK->ACT (planner -> controller -> `ControlCommand`). The task's
  "produces a Setpoint ... planner -> controller" is realized as `NavState -> Mission.step ->
  Setpoint`, which preserves the existing, tested `Mission.run(navigator, transport)` seam and all
  its tests (navigator() returns NavState; the mission completes the chain). Wiring =
  `Mission.run(lambda: nav.update(client.state, latest_frame()), client)`.
- **Walking-skeleton:** given pos/vel are fed to the KF as TIGHT measurements (not bypassed), so
  VQ1 IS the VQ2 stack under-tuned — flip `use_given_position` off and the same filter runs
  vision-only. IMU predict every new sample; estimation advances on the sim clock (a faster
  control loop gets the cached state, so no covariance collapse).
- **Vision in-loop, never a crutch:** detector -> `estimate_gate_pose` (PnP, inner 1.5 m) -> map
  data-association (nearest predicted gate-centre, using the known pose) -> `localization` ->
  `KF.update_position`, BEHIND a Mahalanobis innovation gate (`nu^T S^-1 nu` vs the IMU-propagated
  prior; chi-square 16.27 @ 3 DOF). With given position anchoring, a garbage/wrong-gate fix is
  rejected and can't yank the estimate; with given position off (VQ2) the same gate protects the
  vision-only estimate. This is the deferred `project-estimator-robustness` innovation gate, sited
  where the design said it belongs (the navigator loop).
- **Detector is INJECTED** (`.detect(frame)->[GateObservation]`): the live loop passes `None` to
  fly the known map on the given state alone (safe order); tests pass a fake detector + the
  synthetic projector to exercise vision->KF with no model.
- **Map loader** (`load_track_map` / `gates_from_track_records`): TRACK_INFO / `track_map.json`
  records -> ordered `list[Gate]`, `inner_size_m=1.5` (PnP), NOT the 2.72 m outer. Through-direction
  (gate +Z) is derived from COURSE GEOMETRY (unit vector to the next gate), NOT the sim quaternion:
  the geometric direction always points down-course, so the planner can never place its carrot on
  the approach side and U-turn back through a gate. **TODO: cross-check the sim gate quaternion vs
  this geometry once vision is live** (the quats are all `(0.707,0,0,0.707)` ~ +90° yaw; convention
  unverified — geometry is the safe source for now).

**`scripts/fly_vq1.py`** — the live closed loop + recorder.
- `--dry-run` runs the FULL perception/estimation/planning loop against the live stream and prints
  what it WOULD command, but NEVER arms and NEVER sends. Run this first.
- Connects at the home page (`wait_heartbeat=False`) so the one-shot TRACK_INFO map at level-load
  is captured live; falls back to `--map` (saved deterministic course).
- Waits for an active race (started + position live) before any actuation. Bounded setpoints
  (`--cruise`, default 2.5 m/s), force-disarm on every exit/abort, and stop conditions:
  hard-collision (abort), estimator-divergence (abort), sim-stall (race ended/paused), time cap.
- `--mode {position,velocity,attitude,body_rate}`; attitude takes a calibrated `--hover-thrust`.

13 new tests (`tests/test_navigator.py`): map load (6 gates, inner 1.5, down-course normals);
given-state tracking; vision-only convergence to truth via the projector (catches frame/sign
errors end-to-end); innovation gate rejects an inconsistent fix; detector once-per-frame_id;
Navigator -> Mission.run drives a 3-gate course to FINISHED.

## BLOCKERS for live vision (ACTION NEEDED before --vision works)
1. **`models/gate_yolo11s_curriculum_v2.pt` is NOT on this VM** (no `models/` dir at all; weights
   are gitignored — move via the adroit-connector, run its CLI via PowerShell not Git Bash).
2. **`ultralytics` is NOT installed** in `.venv` (`pip install -e ".[detector]"`).
=> Until both land, fly GIVEN-STATE (`detector=None`, the default). Vision->KF is fully tested
offline and flips on with `--vision` once weights+ultralytics are present.

## Live bring-up runbook (needs the GUI driven home -> waiting -> Race!)
The sim is up but at the HOME PAGE (physics PAUSED: `sim_t=0`, no pos/vel, no map). Order:
0. Offline: `python -m pytest -q` -> 229 green.
1. **DRY-RUN** (no actuation): `python scripts/fly_vq1.py --dry-run --label dry`
   You drive home->Race while it waits. Confirms: live loop runs, map captured (live or saved),
   NavState tracks the given position, clean teardown. Read the printed pos vs the sim.
2. **CONTROL MODE** (first-contact TODO #4 — settle position/velocity easy-mode IN ANGLE MODE):
   `python scripts/control_mode_probe.py --modes attitude,position,velocity --hold-s 2`
   (attitude FIRST so the sim enters ANGLE, THEN test position/velocity). Note which RESPONDED.
3. (only if flying attitude) `python scripts/innerloop_step.py` -> feed the fitted `hover_thrust`.
4. **FLY** (slow, given-state): `python scripts/fly_vq1.py --mode <responding> --cruise 2.5 --label vq1_run1`
   Decision tree: position/velocity responded -> `--mode position` (lean on the stabilizer, safest).
   Else -> `--mode attitude --hover-thrust <fit>`. body_rate/CTBR: the controller's rate law is
   UNBUILT + un-system-ID'd -> not the first-flight choice (next lever).
5. Recording lands in `data/runs/<stamp>_vq1_run1` (gitignored). Report gate_index, collisions,
   final_state, and the race active_gate/finished.

## Notes / decisions
- Given position is pristine, so VQ1 navigation is essentially solved: we fly the KNOWN map even
  with no gate in view, so the no-gate case never stalls and a SEARCH state is not needed for VQ1
  (it remains a VQ2 / unknown-map concern). Vision is a monitored consistency check here.
- CTBR is the durable directive (position-in-ACRO ran away), but the controller's BODY_RATE law is
  unbuilt and un-tuned; the first VALID flight should use the safest interface the probe confirms
  (position/velocity easy-mode if honoured, else the geometric attitude law with a calibrated
  hover_thrust). Implementing + system-ID'ing CTBR is the speed lever after a banked VQ1.

## LIVE FLY SESSION (2026-06-02/03) — control findings (the meat; hard-won)
Flew ~10 live runs. Navigation + lifecycle are SOLVED; control is partway. Every finding below
is from the real sim, cross-checked against the shipped `PyAIPilotExample` (the wire authority).

**1. Position/velocity setpoints RUN AWAY — the sim is ACRO-only (Betaflight-style).**
`control_mode_probe` (attitude,velocity,position): a velocity cmd of −0.5 m/s -> +44.5 m climb; a
position cmd z=−1 -> +44.9 m. NOT a tracked response — a runaway (the probe's "RESPONDED" verdict is
a false positive of its crude climb>0.2 m threshold). The USER confirmed via the sim UI: the drone
**never leaves ACRO**, and resets don't change it. So position/velocity "easy-mode" is DEAD. The
reference client agrees: its only working control paths are CTBR (`SET_ATTITUDE_TARGET` with
`ATTITUDE_IGNORE` = body-rate) and velocity (`SET_POSITION_TARGET`); there is **no attitude-quat /
ANGLE path**. => the VQ1 control path is **CTBR (body-rate + collective thrust)**, as the durable
directive always said.

**2. The race COUNTDOWN — settled (was the early-start DQ).** Clicking Race sets `started=True` AND
`race_start_boot_time_ms = sim_boot_time_ms + ~2.8 s` (a FUTURE time = the GO). The countdown is
`race_start_boot − sim_boot` ticking to 0. **Controlling before GO = DQ.** `fly_vq1._wait_for_race`
now waits until `sim_boot ≥ race_start_boot + start_margin`, only accepts a FRESH countdown (GO within
the last ~2 s, not a stale race whose GO was minutes ago), and REFUSES to fly unless the drone is at
the origin (the two traps that bit early runs: a stale race + a corrupted 164 m / −8273 m start from
the earlier pos/vel runaways which the sim does NOT auto-reset). Tool: `scripts/race_observe.py`
(read-only) captured the mechanics. Reset protocol: home page clears the race + resets the drone to
(0,0,0) at the resting pose (pitch −17.8°, yaw −180°, facing down-course); navigating in re-broadcasts
the map; the drone HOLDS at the origin through the countdown AND after GO (no fall) so there's time to
take over. The drone is auto-armed at race start.

**3. 🚩 BODY-RATE SIGN: the sim INVERTS roll + yaw rate commands (pitch is correct).** THE
breakthrough. Found by OFFLINE command-vs-response replay (`scripts/analyze_run.py` extracts the
attitude/position trajectory from a run's tlog; then re-run the planner+controller on the recorded
states and compare commanded vs actual ODOMETRY rates): at the divergence onset my commanded roll/yaw
were small + POSITIVE but the drone's actual rates were NEGATIVE — a +roll/+yaw command rotates the
drone the OTHER way, so a pure-P loop has POSITIVE feedback and spirals/tumbles inverted (roll→±180°).
Pitch leveled fine because its sign is right. Fix: `Controller.body_rate_sign` (roll,pitch,yaw) maps
the trusted-FRD geometric command to the sim's actuation convention; **measured = [−1, 1, −1]**
(fly_vq1 default). Cousin of the known ATTITUDE pitch-sign inversion. After this, roll+yaw are stable
and the drone flies the RIGHT direction (one run reached x=−19 of the gate-0 x=−23 before other axes
diverged).

**4. RATE LOOP is VERY underdamped — the REMAINING blocker.** The drone's measured body rates
overshoot my commanded (clamped) rates by ~2.7× (commanded ≤1.0–2.0 rad/s, ODOMETRY shows 4–6 rad/s),
and `innerloop_step`'s attitude step showed ~90% overshoot. So a pure-P attitude→rate law oscillates:
with the sign fixed, pitch oscillates ±25°/±4 rad/s (and that oscillation PUMPS the drone upward — the
climb); give altitude/lateral more authority (`--max-accel`) and the lateral (roll→y) axis re-excites
and diverges. It's whack-a-mole because the inner rate loop is uncharacterized. Added `kd_att` rate
damping (`omega −= kd_att·measured_rate`) — helps but heavy damping then prevents the drone pitching
to fly forward. **This needs a clean rate-loop sysid, not more blind gain-tuning.**

**5. hover_thrust is MIS-CALIBRATED.** `innerloop_step` measured 0.489 — but in ACRO the level
attitude command it sent was IGNORED, so the sweep happened at the −17.8° resting tilt. The true LEVEL
hover is lower (~0.46), and even there the drone climbs, so it's still uncertain. Needs a clean level
hover measurement (command zero body-rate at a held level attitude + sweep thrust).

### THE unlock (do this next, before more flying)
A clean **body-rate STEP sysid** on a running race (post-GO): command a fixed small rate on ONE axis
(e.g. +0.3 rad/s roll, then pitch, then yaw) for ~1 s with brief level-holds between, and measure the
STEADY actual rate (ODOMETRY) + the response shape. That gives (a) the per-axis sign (confirm
[−1,1,−1]), (b) the rate SCALING (is the ~2.7× a steady gain or a transient overshoot?), and (c) the
damping. Then EITHER feedforward the scaling (so my commanded rate matches the sim's actual) OR design
a properly damped attitude controller — and the existing CTBR + sign fix should fly. Pair with a level
hover-thrust sweep. The `analyze_run.py` replay makes all of this offline-checkable from one recording.

### Recordings (data/runs/, gitignored) — the progression
`*_ctbr_gate0` (tumble, pre-sign-fix) · `*_ctbr_signfix` (roll/yaw FIXED, pitch oscillates + climbs) ·
`*_ctbr_damped` / `*_ctbr_alt` (altitude-authority experiments) · `innerloop_20260602_153048.json`
(hover/attitude sysid). These ARE the rate-loop + perception data — feed the rate sysid + the RL twin.

## Next levers (in order)
1. **Body-rate STEP sysid** (above) -> rate scaling/sign/damping -> a matched CTBR controller -> a
   stable hover -> gate 0 -> full VQ1. (THE current blocker; everything else is ready behind it.)
2. Clean LEVEL hover-thrust sweep (retire the tilted innerloop_step value).
3. Land weights + `ultralytics` (weights staged at `models/`; torch NOT installed) -> `--vision` ->
   confirm vision->KF live (watch `vfix`/`vrej`).
4. Cross-check the sim gate quaternion against the geometry-derived through-direction.

## Code state (all committed + pushed, branch red-team-tier-a, 236 tests green)
navigator + tests (58d3bc7) · fly_vq1 (0bf1936) · handoff (16f6594) · controller bounds
(0a17c28) · CTBR (8b46b67) · countdown + race_observe (263a309) · CTBR rate damping + analyze_run
(2fdfe96) · body_rate_sign (c73dd0a). New scripts: `fly_vq1.py`, `race_observe.py`, `analyze_run.py`.
Controller knobs for THIS sim: `body_rate_sign=[-1,1,-1]`, `kp_att`/`kd_att`/`max_body_rate_rps`,
`thrust_slope_mps2`, `max_accel_mps2`, `max_pos_error_m`, `hover_thrust`.
