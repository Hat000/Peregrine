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

## Next levers (in order)
1. Land weights + `ultralytics` -> `--vision` -> confirm vision->KF live (watch `vfix`/`vrej`).
2. `innerloop_step` -> calibrate `hover_thrust` -> attitude mode (then CTBR).
3. Build the BODY_RATE (CTBR) controller law for VQ2 speed.
4. Cross-check the sim gate quaternion against the geometry-derived through-direction.
