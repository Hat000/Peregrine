# First-contact runbook (sim drop 2026-05-30)

The ordered sweep to run the moment the official sim is up. Each step says what it
measures, the pass/fail read, and which open risk (R1–R11) / red-team unknown it settles.
Goal: come out of the first session with R1/R2 resolved, the control plant system-ID'd, and
the two deferred-rebuild decisions (ESKF, delayed-vision) settled by **data**, not guesswork.

## Ground rules
- **Co-locate the Python client ON the sim box** (localhost MAVLink/UDP). A remote client
  blows the <50 ms latency budget. (project-hardware-constraint.)
- **Start the simulator FIRST, then the script.** The video receiver self-heals on a late /
  gappy stream; MAVLink falls back to video-only on timeout.
- **Record real runs.** The course is deterministic ⇒ one good recording replays into many
  offline iterations (system-ID, mapping, racing line, detector auto-labels).
- **One probe at a time.** Each binds the MAVLink endpoint + video port 5600, so they can't
  run concurrently. For a capture that runs *alongside* a probe, put a `mavlink-router` /
  mirror in front; otherwise do a dedicated `record_session` pass.
- Default endpoint `udp:127.0.0.1:14550`, video `udp:5600`. Override with `--endpoint` /
  `--video-port` once the real ports are known.

## Sequence

### 0. Offline sanity (no sim needed) — `projection_check.py`
Confirms `frames.py` conventions + VFoV≈58.72° (spec's 90° is the HFoV) + the 20°-tilt sign.
Already passes synthetically; after the first recording, overlay the computed horizon on a
real frame to validate against the sim:
```
python scripts/projection_check.py
python scripts/projection_check.py --from-recording data/runs/<stamp>_<label>
```
PASS = all checks green + the drawn horizon sits where it should. Resolves the VFoV/frames trap.

### 1. Lifecycle + telemetry audit (SAFE: arms→disarms, no setpoints) — `session_lifecycle.py`
```
python scripts/session_lifecycle.py            # add --force if arming is refused
```
Keep the vehicle **still on the ground** (the attitude-bias read needs it quasi-static).
Reads:
- **Backend** (PX4 / ArduPilot / Betaflight-ish) from HEARTBEAT.
- **R1** — is `position_ned` / `velocity_ned` / `mag` / `baro` in telemetry?
- **Arming** — does it ack? normal vs `--force`? latency?
- **Clock** — is `sim_time_ns` (HIGHRES_IMU) monotonic? is TIMESYNC present?
- **Attitude bias** (NEW, red-team CRIT-2) — given attitude vs gravity-implied tilt at rest.
- STATUSTEXT narration (often reveals the whole lifecycle: "armed", "race started", "gate N").

### 2. Clock + video health (SAFE: read-only) — `clock_probe.py`
```
python scripts/clock_probe.py --seconds 10     # with the video streaming
```
Reads:
- **Video↔IMU clock alignment** (red-team CRIT-1/3): offset (imu − frame) `sim_time`,
  mean + spread. ALIGNED (small/stable) ⇒ shared clock, offset = video pipeline latency ⇒ the
  delayed-vision ring buffer is well-posed. SUSPECT (large/jittery) ⇒ independent epochs ⇒
  reconcile via TIMESYNC before fusing vision at speed.
- **Video health** (MTU / packet-loss): datagram size vs the ~1500 B MTU, chunks/frame,
  frames lost to missing chunks.

### 3. Control interface (ACTUATES — small, bounded; force-disarms) — `control_mode_probe.py`
Needs vertical clearance. Start safe:
```
python scripts/control_mode_probe.py --no-actuate        # arm + telemetry only
python scripts/control_mode_probe.py                     # then the real per-mode test
```
Resolves **R2 + position "easy mode"**: streams tiny setpoints per mode (velocity / position /
attitude / body_rate) and reports which interface actually moves the drone. velocity/position
RESPONDED ⇒ VQ1 can lean on the sim's stabilizer. Nothing responds ⇒ the sim likely needs a
mode switch (OFFBOARD/GUIDED) or a pre-arm setpoint stream — note it for a follow-up probe.

### 4. Plant system-ID (ACTUATES — climbs/descends; force-disarms) — `innerloop_step.py`
Needs more vertical clearance. Start with `--no-actuate`, then:
```
python scripts/innerloop_step.py
```
Fits the numbers the controller waits on (**R2 keystone**): `hover_thrust`, thrust→accel
slope, TWR, and the attitude-loop delay/τ/overshoot. Writes JSON to `data/runs/`. Feed
`hover_thrust` into `Controller(...)` (retires the 0.5 placeholder) and cap the planner's
accel / attitude-rate demand by the measured envelope.

### 5. Record a real traversal — `record_session.py`
```
python scripts/record_session.py --label firstpass        # Ctrl-C to stop
```
Captures both streams + a built-in `msg_audit` (MAVLink type histogram + R1 hedge fields).
This recording feeds offline mapping / system-ID / racing-line / detector auto-labels.

## Decision table (observation → action)

| Observation | Action |
|---|---|
| `position_ned` present in telemetry | Localization collapses — but still run vision→KF in-loop (walking-skeleton); use the given position as a ground-truth cross-check only. |
| `velocity_ned` present | Feed `kf.update_velocity`; else derive from IMU + vision finite-diff. |
| **Attitude bias > ~0.5° persistent at rest** | **Deferred rebuild #1 TRIGGER MET** → enable the ESKF bias-state. ≤0.5° ⇒ the linear KF stands. |
| **clock_probe = SUSPECT** (clocks not aligned) | Implement TIMESYNC reconciliation before the delayed-vision ring buffer; until then don't fuse vision at high speed. ALIGNED ⇒ **deferred rebuild #2** is safe to build when VQ2 speed needs it. |
| Max datagram > MTU and loss % climbs under load | Expect frame loss at speed; the KF coast + 3-corner fallback must cover it. Consider requesting smaller chunks; note for the detector (motion-blur + dropped frames). |
| velocity/position mode RESPONDED | VQ1 leans on the stabilizer (`ControlMode.POSITION/VELOCITY`). None ⇒ go attitude→CTBR. |
| TIMESYNC present | Use it as the cross-stream clock bridge. |
| Arming refused | Retry `--force`; read the COMMAND_ACK result + STATUSTEXT for why. |

## Red-team deferred-rebuild triggers (settle these here)
- **ESKF bias-state** ← step 1 attitude-bias read. Build iff bias is observable.
- **Delayed-vision ring buffer** ← step 2 clock alignment (prereq) + VQ2 speed. Build when both hold.
- **Hard coast + innovation gating** ← belongs in the navigator loop + mapper (not a probe).

See [[project-red-team-pass-2]] for the full triage; the Tier-A fixes are on branch `red-team-tier-a`.
