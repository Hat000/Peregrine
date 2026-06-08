# UNDERSTANDING — velocity-setpoint fork (ShadowPC, 2026-06-04)

My model BEFORE touching the sim. Written so the teammate can catch a misunderstanding at the
Phase-0 checkpoint, and so I can compare results against pre-registered predictions afterward
(no reshaping predictions to fit results). Authored by a fresh session deliberately re-opening a
conclusion the memory marks "CLOSED" — under instruction, and staying neutral about the outcome.

## What Peregrine is / where gate 0 stands / the open blocker
Peregrine is our entry in the Anduril AI Grand Prix autonomous drone-racing competition; this work
is the VQ1 control floor — getting the drone to actually fly the deterministic course in the
official sim (`FlightSim.exe`, co-located here on ShadowPC). First contact succeeded: position +
velocity are GIVEN as pristine ground truth, the gate map is captured, and a hand-built decoupled
CTBR (body-rate + collective-thrust) controller exists. On gate 0 the lateral/centering/frame/map/
false-pass bugs are all fixed and lateral is SOLVED, but ONE blocker remains under the CTBR
approach: an **altitude balloon** — during a run the drone climbs away (z 0 → -8 m) because when our
collective dips, the sim's auto-thrust is believed to take over and climb. The CTBR stack exists
specifically because the team concluded "we must own thrust" — which is exactly the conclusion this
experiment re-tests.

## The fork (in my own words) — what this one measurement decides
The binary question: **in ANGLE mode, does the sim's velocity-setpoint controller hold altitude and
track a commanded NED velocity?** Three worlds:

- **World B — easy-mode is REAL.** The sim, left in ANGLE (no client heartbeat/timesync), runs a
  proper velocity controller: a `vz=0` setpoint HOLDS altitude, and `vx/vy` setpoints translate the
  drone with z staying flat. If so, VQ1 flies on plain `SET_POSITION_TARGET` velocity setpoints and
  most of the CTBR cascade (alt-hold, tilt-comp, rate sysid, sign-hunting, the balloon fight) is
  **unnecessary**. This is what the official reference client implies (it ships a pure velocity
  setpoint as its simple path).
- **World A — easy-mode is DEAD (CTBR justified).** Velocity setpoints do not hold altitude in any
  mode: a `vz=0` HOVER still climbs away, and explicit `vz` commands don't change it (thrust is
  effectively ignored / auto-thrust runs away). Then the hand-built CTBR cascade is the right and
  necessary path, exactly as the memory says.
- **Partial.** HOVER does NOT hold (some residual climb/bias at `vz=0`), BUT explicit `vz` commands
  DO move the drone the right way (DESCEND descends, CLIMB climbs). That points to a small, nameable
  fix (e.g. a constant vz bias, a hover-point offset, or that we must always command a small vz)
  rather than abandoning velocity control.

Why this experiment is decisive: it is the FIRST clean ANGLE-mode velocity recording with validated
instruments. Every prior verdict was taken either in ACRO (heartbeats on) or by GUI-glance without a
recording. The ladder (HOVER / DESCEND / CLIMB / FORWARD) isolates the two independent questions —
"does z hold at vz=0?" and "does vz respond, both signs?" — which is exactly what separates A vs B vs
Partial.

## The contradiction I am resolving (file + line citations)

**Claim X — "easy-mode is CLOSED / DEAD" (the conclusion being re-tested):**
- `memory/project_ctbr_control_sysid.md:8-16` — "The sim is ACRO-only — easy-mode is CLOSED
  (exhaustively tested) ... In every mode the velocity/position auto-thrust is broken: a pos/vel
  setpoint LEVELS the attitude ... but climbs away (thrust ignored → 57 m runaway). Conclusion: WE
  must own thrust ... Do NOT re-litigate easy-mode."
- `handoff/shadowpc-navigator-2026-06-02/HANDOFF.md:97-105` — "Position/velocity setpoints RUN AWAY
  — the sim is ACRO-only ... a velocity cmd of −0.5 m/s -> +44.5 m climb ... The USER confirmed via
  the sim UI: the drone never leaves ACRO ... So position/velocity 'easy-mode' is DEAD."

**Claim Y — "velocity in ANGLE was never actually tested; re-test it" (the open thread):**
- `memory/reference_sim_interface.md:52` — "Live in ACRO: velocity→no climb; position→VIOLENT 57 m
  runaway ... ⇒ fly on CTBR; RE-TEST position/velocity easy-mode in ANGLE mode (send an attitude-quat
  first to enter angle)."
- `handoff/shadowpc-firstcontact-2026-06-02/HANDOFF.md:24` and `:68` — "velocity->no climb ...
  RE-TEST position/velocity easy-mode in ANGLE mode" + TODO #4 "Re-test position/velocity easy-mode
  in ANGLE mode."
- `scripts/angle_vel_test.py:1-12` (docstring, the LATER 2026-06-03 discovery) — "no client sending
  anything -> sim stays in ANGLE ...; a HEARTBEAT or TIMESYNC keepalive -> flips to ACRO; a
  position/velocity SETPOINT does NOT flip the mode (stays ANGLE) ... the reference client only ever
  sends VELOCITY ... If the sim's ANGLE velocity controller tracks, easy-mode is real."
- Reference client `...\PyAIPilotExample\controller.py:85-121` — `update_position_flight_control`
  ships a pure velocity setpoint (`vx=2, vy=0, vz=0`, no thrust, no position, no yaw) as the simple
  control path (commented-out alternative to the actuator default).

**Why X is suspect (the mechanism, verified in-repo):** the "DEAD" verdict came from
`scripts/control_mode_probe.py`, which connects with `client.connect(...)` at its default
`wait_heartbeat=True` and never sets `send_heartbeats=False` (`control_mode_probe.py:98`), so
`MavlinkClient.send_heartbeats` stays `True` (`mavlink_client.py:147`) and it streams a 2 Hz GCS
heartbeat the whole time. Per the LATER discovery, a client heartbeat flips the sim ANGLE→ACRO
(`mavlink_client.py:189-195` pump() docstring; `angle_vel_test.py:4-6`). So that "velocity runs away"
was measured **in ACRO** — the probe's attitude-quat-first trick (intended to enter ANGLE) was
defeated by its own heartbeat. The probe's `responded = climb > 0.2 m` test is also the very "false
positive" the navigator handoff disowns (`HANDOFF.md:99-100`). Net: the ANGLE-mode velocity
controller has **never** been cleanly recorded. Claim X generalized an ACRO result to "every mode."

I am NOT assuming Y is right. World A is a fully live possibility — the sim's velocity controller may
genuinely fail to hold altitude even in ANGLE. The recording decides.

## Pre-registered predictions (fill BEFORE running; compare AFTER)

Conventions: NED, **+z is DOWN** (so climbing = z decreases = vz negative; descending = vz positive).
Each phase ~3 s @ 25 Hz. yaw0 ≈ −180° (facing down-course), so forward toward gate 0 ≈ vx≈−1.0.
"z drift" = z(end) − z(start) over the phase; "holds" ≈ |z drift| < ~0.3 m.

| Phase | Command v (NED) | World B predicts | World A predicts | Partial predicts |
|-------|-----------------|------------------|------------------|------------------|
| HOVER | (0, 0, 0) | achieved v ≈ (0,0,0) ±0.1; z HOLDS (|Δz|<0.3); rpy near resting; motors modulate to hold | NOT held: vz drifts negative (climbs); z runs away upward (Δz strongly negative, accelerating → trips alt abort); achieved v ≠ cmd | vz ≠ 0 (residual climb, e.g. vz≈−0.2…−0.4 roughly constant); z drifts up but slower/bounded vs World A |
| DESCEND | (0, 0, +0.5) | vz ≈ +0.5 (±0.15); z increases ~+1.5 m over 3 s; horizontal ≈0; correct sign | vz does NOT reach +0.5; climbs anyway (vz negative); wrong sign / no vz response | vz RESPONDS toward +0.5 (drone descends), maybe with a bias offset |
| CLIMB | (0, 0, −0.5) | vz ≈ −0.5 (±0.15); z decreases ~−1.5 m over 3 s; authority confirmed both directions | climbs but uncontrolled (not a steady −0.5; runaway), ~indistinguishable from HOVER | vz RESPONDS toward −0.5 (climbs); pairs with DESCEND to prove both-direction authority |
| FORWARD | (≈−1.0, 0, 0) toward gate 0 | horizontal speed ≈1.0 toward gate (±0.2); vz≈0, z FLAT (|Δz|<0.3); small forward tilt; tracks | z NOT flat (climbs); horizontal response uncertain (ACRO "velocity→no-climb" vs "runaway" priors disagree) | horizontal tracks ≈1.0; z drifts (not flat) — same vertical defect as HOVER |

Yaw A vs B (same ladder twice): **A** = `cmd.yaw=None` → type_mask 3527, byte-identical to the
reference `VELOCITY_POSITION_MASK`. **B** = `cmd.yaw=yaw0` → type_mask 2503 (yaw bit cleared, yaw
used). Prediction: in World B both hold and track; if **B diverges (climb/yaw wander) while A holds**,
our yaw setpoint in the velocity message was the bug (the reference never sends yaw). In World A both
run away regardless of yaw.

**Decision mapping (from the matrix):** World B ⇔ HOVER holds z AND FORWARD translates with z flat.
World A ⇔ HOVER climbs regardless of vz sign AND DESCEND/CLIMB don't respond. Partial ⇔ HOVER climbs
but DESCEND/CLIMB DO respond (then name the fix).

## Phase-1 instrument validation I will require BEFORE interpreting any phase
1. **Null test** — passive, drone holding at origin: all three velocity channels read ≈0 (|v|<0.05).
   Nonzero on a stationary drone = biased pipeline → STOP and fix, do not interpret the ladder.
2. **Triangulation** — three INDEPENDENT velocity derivations every tick: `LOCAL_POSITION_NED.v`,
   `ODOMETRY.v`, and Δposition/Δt. They must agree (~0.1 m/s) during known motion. NOTE: I must tap
   the RAW messages — `mavlink_client._handle` writes BOTH LOCAL_POSITION_NED and ODOMETRY into the
   single `state.velocity_ned`/`state.position_ned` fields (`mavlink_client.py:240-245, 255-268`), so
   reading `state.velocity_ned` gives an interleaved 75/97 Hz mix, not one clean channel. Divergence
   = I don't understand the telemetry → HALT, report, do not interpret.
3. **Command echo** — log the actual SET_POSITION_TARGET fields + type_mask transmitted; confirm
   mask=3527 (yaw-ignore, == reference) / 2503 (yaw-hold), vz as intended, position/accel ignored.
4. **GUI corroboration** — the teammate's GUI is the ONLY mode ground-truth (mode is NOT on the wire;
   `HEARTBEAT.base_mode` is always MANUAL|CUSTOM). Each phase, the teammate's gross call (climbed /
   held / moved) and the reported GUI mode (want ANGLE) must match the telemetry story. Mismatch →
   GUI is truth, telemetry is SUSPECT, and the mismatch is the headline finding.

## Assumptions
- The teammate will drive the sim Home→Race fresh per run and report (a) the GUI mode each phase and
  (b) gross behavior; I supply the numbers. This is a teammate-in-the-loop measurement.
- With `send_heartbeats=False` AND `send_timesync=False`, the sim STAYS in ANGLE and KEEPS streaming
  telemetry passively (first-contact + angle_vel_test say telemetry streams for free).
- The race auto-arms; I will NOT arm. I will refuse to start unless the drone is at origin
  (|pos|<1 m), armed, and the sim clock is advancing (race actually running).
- Given position/velocity are pristine ground truth (first-contact R1), so they are a trustworthy
  reference for the null/triangulation checks.

## Things in the prompt I could NOT fully verify against the repo
- "Reached by GUI-glance, never closed with a recording" — I can corroborate the MECHANISM
  (control_mode_probe ran with heartbeats = ACRO) and I found no banked recording/handoff reporting
  an ANGLE-mode `angle_vel_test.py` result, but I cannot prove a recording was never taken. Stated as
  my read, not certainty.
- Test count drift in memory ("258" vs "265") and the exact sim build number — I'll read the build
  from the live session / ask the teammate rather than trust the memo.
- **⚠️ A live contradiction about TIMESYNC to flag:** `mavlink_client.py:148-152` says send
  `send_timesync=True` (with heartbeats off) to KEEP ANGLE while still receiving telemetry — but
  `angle_vel_test.py:5-6` says TIMESYNC *also* flips to ACRO. The task sides with both-OFF. I will go
  both-OFF (heartbeats=False, timesync=False) and rely on passive telemetry + GUI confirmation; if
  telemetry stops or the GUI shows ACRO anyway, THAT is a finding to report, not something to paper
  over.
- **Branch/push nuance:** I'm on worktree branch `claude/nervous-payne-a33d52` (≡ `origin/main`,
  cbf2e6e, which already merged red-team-tier-a), while `red-team-tier-a` is checked out in the
  sibling worktree `C:/Users/Shadow/Peregrine`. The task says push to `origin/red-team-tier-a`; I'll
  settle the exact mechanics at Phase 3 and confirm with the teammate before pushing.

## Open questions (to resolve live)
- Does passive (no-keepalive) telemetry keep flowing for the whole multi-minute ladder?
- Does the sim actually stay ANGLE across the run, or does sending a velocity setpoint (or anything
  else) flip it? (GUI is truth.)
- If World B: do yaw-A and yaw-B differ — i.e. is the yaw field in the velocity message a disturbance?
- Reproducibility: does the HOVER crux repeat across ≥2 independent races, or is it run-dependent?

## What would make this run a SUCCESS vs FAILURE
SUCCESS includes "instruments disagree / GUI contradicts telemetry → INCONCLUSIVE" reported honestly.
FAILURE = a confident verdict from one unvalidated channel, or interpreting the ladder before Phase 1
passes, or quietly fixing the test until a clean-looking result appears. I will label any
single-channel conclusion UNCORROBORATED.
