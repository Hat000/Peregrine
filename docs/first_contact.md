# First-contact runbook (sim drop 2026-05-30)

The ordered sweep to run the moment the official sim is up. Each step says what it
measures, the pass/fail read, and which open risk (R1–R11) / red-team unknown it settles.
Goal: come out of the first session with R1/R2 resolved, the control plant system-ID'd, and
the two deferred-rebuild decisions (ESKF, delayed-vision) settled by **data**, not guesswork.

## Sim interface — CONFIRMED from the shipped PyAIPilotExample (2026-06-01)
The sim zip (`AI-GP Simulator v1.0.3364.zip`) ships `FlightSim.exe` + a reference client
`PyAIPilotExample` (extracted on the dev laptop at `C:\Users\Fengy\Downloads\AIGP_sim\`). Our
client was cross-checked against it + VADR-TS-002 and aligned (on `red-team-tier-a`). Wire facts:
- MAVLink `udpin:127.0.0.1:14550` (client binds); video JPEG-UDP `:5600`, header `<IHHIIQ`
  (byte-for-byte match). Everything NED.
- The sim sends MORE than spec §4.3 lists: also `LOCAL_POSITION_NED`, `ODOMETRY`
  (pose + vel + reset_counter), `ENCAPSULATED_DATA` (RACE_STATUS = active_gate_index + race
  timing; TRACK_INFO = full gate map id/NED-pose/dims, chunked via DATA_TRANSMISSION_HANDSHAKE),
  `COLLISION` (1001 gate / 1002 env + impulse), `ACTUATOR_OUTPUT_STATUS`. Control: SET_POSITION_
  TARGET (pos/vel), SET_ATTITUDE_TARGET (attitude / CTBR), SET_ACTUATOR_CONTROL_TARGET (direct
  motor — the sample's default), arm, and `MAV_CMD 31000` = sim reset. **Trust the sample for
  wire reality; the spec message table is thinner/older.**

MUST-VERIFY live (the cross-check surfaced these — settle them in the first session):
1. Is `LOCAL_POSITION_NED` / `ODOMETRY` actually POPULATED (real ground-truth) or zero/absent?
   STRATEGY-DEFINING — if real, localization collapses (but keep vision→KF in-loop per the
   walking-skeleton directive); spec §3.3 + README say "vision-only / no absolute position".
2. `TRACK_INFO` width/height = INNER (1.5 m) or OUTER (2.7 m)? PnP needs the inner square.
   Also: is the provided map full or "rough" (FAQ), and does it arrive automatically?
3. Per-type RATES — now auto-reported by `session_lifecycle` / `record_session`. Expect camera
   30 Hz, heartbeat ≥2 Hz, physics 120 Hz; ATTITUDE/IMU/ODOMETRY rates are unspecified. And the
   COMMAND-RATE truth: spec §4.4 says <100 Hz but the sample streams setpoints at 250 Hz.
4. `SET_ACTUATOR_CONTROL_TARGET` scaling: normalized [-1,1] vs RPM vs throttle.
5. Does arming need a mode switch (OFFBOARD/GUIDED) or a pre-arm setpoint stream?
   (`session_lifecycle` + `control_mode_probe` reveal it.)

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

---

# INFORMATION TO EXTRACT FROM THE SIM — master checklist (there are a lot)

The single source of "what we still don't know and must measure." Most rows fall out of steps 1–5 +
the recording above; the rest need a follow-up probe or the organizers. Tonight's additions: the
**perception noise/latency/dropout model** (the RL speed ceiling) and the **per-corner confidence
distribution** (sets the weighted-PnP adapter-relax cutoff). Tick each off the first session(s).

### A. Connection & protocol
- [ ] Exact MAVLink endpoint(s) + video port (then override `--endpoint` / `--video-port`).
- [ ] Backend / autopilot (PX4 / ArduPilot / Betaflight-ish) + custom_mode semantics (HEARTBEAT).
- [ ] **MAVLink vs Betaflight/RC-UDP authority** — spec says MAVLink is authoritative; confirm the elodin/RC path isn't the real control channel.
- [ ] MAVLink 1 vs 2, dialect, any custom messages; full message-type histogram + per-type rate (`msg_audit`).

### B. Telemetry contents — R1 (the big hedge; per field: present? rate? units? frame? quality?)
- [ ] `position_ned` (LOCAL_POSITION_NED / GLOBAL) — **if present, localization collapses** (still run vision→KF in-loop per the walking-skeleton directive; use given pos as a cross-check only).
- [ ] `velocity_ned` — feed `kf.update_velocity`, else derive (IMU + vision finite-diff).
- [ ] Attitude (quat/euler) — rate; TRUE vs noisy/biased (drives the ESKF-bias decision).
- [ ] Angular rate (body); accel/specific-force (IMU predict); `mag_body` (free yaw?); `baro` (free z?).
- [ ] GPS / global fix? RC channels? battery? Anything else "for free."
- [ ] Attitude BIAS at rest: given attitude vs gravity-implied tilt (>~0.5° persistent ⇒ ESKF trigger).

### C. Clocks & timing
- [ ] sim_time epochs: HIGHRES_IMU.time_usec monotonic? ATTITUDE.time_boot_ms a *different* epoch? (clock-isolation assumption.)
- [ ] **Video↔IMU clock alignment** (offset mean + spread) — gates the delayed-vision ring buffer.
- [ ] TIMESYNC present? (cross-stream clock bridge.)
- [ ] End-to-end LATENCIES: command→observed-response, frame-capture→arrival, telemetry age (the <50 ms budget).
- [ ] Achievable loop rate: telemetry rate + setpoint accept rate.

### D. Video stream
- [ ] Format (JPEG/MJPEG?), resolution (confirm 640×360), color order, bit depth, frame rate (steady/variable).
- [ ] **FOV**: confirm HFoV 90° / **VFoV≈58.7°** (spec mislabels VFoV) + the 20° up-tilt sign — `projection_check --from-recording` on a real frame.
- [ ] MTU / datagram size / packet-loss / chunks-per-frame (frame loss at speed?).
- [ ] Camera intrinsics K — does the sim match our assumed K? (known-gate PnP residual.)
- [ ] Appearance: VQ1 clean/desaturated/aids-on **vs** VQ2 photoreal/cluttered/aids-off — characterize BOTH (DR validation).
- [ ] Motion blur / exposure / rolling-shutter at speed (detector robustness).

### E. Control interface — R2
- [ ] Which setpoint modes ACTUATE: POSITION / VELOCITY / ATTITUDE / BODY_RATE.
- [ ] Arming + mode handshake: OFFBOARD/GUIDED needed? pre-arm setpoint stream? COMMAND_ACK results + STATUSTEXT.
- [ ] type_mask honored? partial setpoints (vel-only, pos+yaw) accepted? required stream RATE (dropout → failsafe?).
- [ ] **Position "easy mode"** — does position control track well enough for a VQ1 floor?
- [ ] Thrust scaling: normalized [0,1] → what (hover_thrust calibrates). Yaw: absolute vs rate.

### F. Plant / dynamics — system-ID (control floor + the RL twin's dynamics spec; CAPTURE EVERYTHING)
- [ ] hover_thrust (retire the 0.5 placeholder); thrust→vertical-accel slope; TWR.
- [ ] Attitude loop: delay, time-constant τ, overshoot, settling. Body-rate tracking + max rate (if BODY_RATE works).
- [ ] Max accel / max tilt / max speed envelope (cap the planner). Drag / damping (coast-down).
- [ ] Actuator/motor latency + first-order lag → the RL randomization spec (motor τ±30%, thrust±20%, latency).

### G. Perception fidelity — the RL SPEED CEILING + weighted-PnP tuning (NEW tonight)
- [ ] **Detector per-corner CONFIDENCE distribution on REAL frames** → sets the weighted-PnP adapter-relax `kpt_conf_thresh` cutoff (the deferred decision).
- [ ] Detector accuracy on real gates: corner error / detection rate / the tail (re-run `eval_detector`/`diagnose_tail` logic on real recordings + auto-labels).
- [ ] PnP residual / pose noise on real gates (vs our ~2 px synthetic median); vision LATENCY (frame→pose) + frame DROPOUT at speed.
- [ ] **The perception NOISE/LATENCY/DROPOUT model to inject into the RL policy's observations** — else a state→action policy trained on perfect sim state CRASHES on real YOLO→PnP→KF. Measure from first-contact recordings.
- [ ] Sim-to-real transfer of the synthetic-chaos detector: does it hold on VQ1 clean? on VQ2 photoreal? (don't fine-tune on clean VQ1 — would collapse DR before the ranked round.)

### H. Course / gates — R4
- [ ] HOW are gate positions/order communicated? a MAVLink message? a file? on-screen only? "rough" per the FAQ?
- [ ] Gate count / layout / lap structure / start-finish. Gate dims — confirm inner 1.5 m, outer, depth 260 mm.
- [ ] Obstacles: present? mapped or vision-only? Track FIXED/shared across attempts (⇒ map-once-then-race)?

### I. Mission / rules mechanics
- [ ] How a run STARTS (arm? a "race start" signal / STATUSTEXT?) and ENDS. Reset behavior between attempts (sim_time reset → KF dt-clamp already guards).
- [ ] Validity: how is a gate-pass detected/scored (the 1.0 m sphere? plane-crossing?). Timing start/stop + penalties.
- [ ] **Attempts** — unlimited? best valid time counts? **Submission process + VQ1 deadline.** Is offline between-runs compute allowed (our determinism strategy — confirm not a cheat)?

### J. Compute / deploy
- [ ] Confirm onboard ~100 TOPS edge budget (FAQ) — sizes the detector + the RL policy. Where does the policy run (onboard vs client)? inference budget.

## Questions for the organizers (not sim-discoverable) — info@theaigrandprix.com
- [ ] Attempts model + best-time scoring + **VQ1 deadline** + submission mechanics.
- [ ] Gate position/order data format (R4) — exactly how is it given?
- [ ] MAVLink vs Betaflight/RC-UDP — which is authoritative for control?
- [ ] Is offline between-runs compute permitted? VQ2 timing/photoreal details; Sept physical-round hardware.

## First contact day-of order (TL;DR)
1. Stand up the sim on the **Windows VM** (finally working) + co-locate the client (localhost).
2. `projection_check` offline → `session_lifecycle` (R1 + clock + attitude bias) → `clock_probe` (alignment + video health) → `control_mode_probe` (R2) → `innerloop_step` (plant) → `record_session` (a clean traversal).
3. Bank the recording (deterministic course ⇒ replays into mapping / system-ID / racing-line / detector auto-labels + the **perception-noise model** + the **confidence distribution**).
4. Settle the two deferred rebuilds (ESKF, delayed-vision) by DATA via the decision table. Then wire `racer/navigator.py` and bank a VQ1 completion before its deadline.
