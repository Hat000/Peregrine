# VQ2 control handshake — WORKING recipe (2026-06-29)

**Sim:** Anduril AI-GP build **1.0.3379**, ShadowPC. Event **"AI-GP Virtual Qualifier R2 — TRAINING"**
( = **VQ2**; R1 = VQ1). Track "Now You See Me, Now You Don't".
**Status:** ✅ SOLVED and verified **on VQ2**. Stable, commanded CTBR flight achieved. This was the
blocker stopping us from closing the loop.

> ⚠️ **Read this if you saw the first push:** the initial version of this doc was written against
> evidence collected on the **R1 / VQ1** event by mistake (both VQ1 and VQ2 use the same track name
> "Now You See Me…", so the HUD looked right). VQ1 vs VQ2 is the **menu event**: from the main menu,
> the events are `R1` (VQ1), `R2 — SUBMISSION`, `R2 — TRAINING` (VQ2). **Press DOWN to reach
> R2 — TRAINING** (down clamps on the last item) before RACE. VQ1 renders as bare wireframe; VQ2 is
> a fully-lit indoor warehouse. Everything below is now re-verified on VQ2; VQ1 results are kept as a
> cross-check (the recipe is identical on both).

---

## TL;DR — the one-line recipe

> **ARM (`MAV_CMD_COMPONENT_ARM_DISARM`=400, p1=1), then stream `SET_ATTITUDE_TARGET` in
> BODY-RATE mode (type_mask=`0b10000000` = ignore attitude quat) at ~100 Hz with body
> roll/pitch/yaw rates (rad/s, FRD) + normalized collective thrust [0,1]. The sim is in ACRO by
> default and stays there.**

This is exactly what `ControlMode.BODY_RATE` already emits in
[`src/racer/mavlink_client.py`](../../src/racer/mavlink_client.py) and what our policy outputs —
the existing uplink transfers **directly**, no new message plumbing. `rl/fly_rl.py` is already on
this path.

**Confidence: HIGH.** On VQ2: ARM ACCEPTED; CTBR zero-rate → all four motors balance at the
commanded collective and gyro nulls (stable); a commanded body-rate step → a perfectly symmetric
motor differential and a **steady, bounded** rotation rate (not divergent); 0 collisions in clean
windows.

---

## The recipe in detail

### Mode
- Flight mode is **ACRO** (body-rate) — shown on the in-sim HUD top-right, on VQ2.
- ACRO is the **default** and needs no mode-change command. HEARTBEAT reports `autopilot=0`,
  `type=2` (QUADROTOR), `custom_mode=0`. Either keepalive regime works for arming + telemetry;
  the GCS heartbeat (`send_heartbeats=True`) is the simpler one and is what was used here.
  (An old VQ1-era note claimed "GCS heartbeat forces ACRO / timesync-only keeps ANGLE" — not
  relevant here: ACRO is the floor and there's no reachable stabilized ANGLE mode via keepalive.)

### Arm
```python
client.arm()                       # MAV_CMD_COMPONENT_ARM_DISARM (400) p1=1, no force needed
client.wait_armed(True)            # HEARTBEAT armed flag -> True
# COMMAND_ACK: {command:400, result:0 (MAV_RESULT_ACCEPTED)}
```

### Control message — `SET_ATTITUDE_TARGET` (BODY_RATE)
| field | value | meaning |
|---|---|---|
| `type_mask` | `0b10000000` (`_ATT_MASK_BODY_RATE`) | ignore attitude quaternion → use body rates + thrust |
| `q` | `[1,0,0,0]` | ignored by the mask |
| `body_roll_rate`  | rad/s, **FRD** | + = roll right |
| `body_pitch_rate` | rad/s, **FRD** | + = pitch (forward motion observed) |
| `body_yaw_rate`   | rad/s, **FRD** | yaw rate |
| `thrust` | normalized **[0,1]** | collective |

Stream rate: tested at **100 Hz**; deploy runs 30 Hz (training dt) and also works. Send via the
canonical path: `ControlCommand(mode=ControlMode.BODY_RATE, body_rate=…, thrust=…)` →
`MavlinkClient.send_command`.

### Rate gain caveat (calibration, not blocker)
On VQ2 a commanded **+0.4 rad/s** pitch produced a steady **~1.0 rad/s** measured gyro — a fixed
**~2.5×** scale (VQ1 cross-check: +0.6 → 1.56, ~2.6×). Proportional, bounded, recoverable.
For precise rate tracking calibrate this gain (or rely on the closed-loop policy). Consistent with
the known sim mixer coupling law.

---

## VQ2 evidence (objective; telemetry has NO pose on the wire, so judged from HIGHRES_IMU accel,
## ACTUATOR_OUTPUT_STATUS motors, COLLISION events, and the in-sim HUD)

Probe: [`ctbr_probe.py`](ctbr_probe.py). Raw timeseries in [`logs/`](logs/). All runs `live:True`
(a guard flags the frozen-episode replay state described below).

### 1. CTBR near-hover — clean & balanced (`logs/vq2_acro_t027_hover.json`)
thrust 0.27, body rates 0:
- first ~2.2 s: motors all **0.27** (spread 0.0), gyro **0.0**, accel steady ~10.1 → balanced,
  controlled, **0 collisions**.
- only later does it drift into the start-gate ring and log gate (id 1001) collisions — an
  **environmental** artifact of spawning *inside* the start gate, not a control failure.

### 2. CTBR maneuver — clean, recoverable pitch (`logs/vq2_pitchstep.json` + `logs/vq2_hud_pitchstep.png`)
thrust 0.35, hover then **+0.4 rad/s pitch** step, **0 collisions**, `live:True`:
- hover phase: motors all **0.350** (spread 0.0), gyro **0.0** → balanced.
- step onset: motors split into a **perfectly symmetric differential** `[0.366, 0.366, 0.334, 0.334]`
  — constant collective, pure pitch torque.
- gyro rises to a **steady ~1.0 rad/s and holds** (bounded, not divergent) → predictable,
  recoverable rate authority.
- ODOMETRY `angular_rate_body` stays 0.0 (blocked in VQ2); **raw HIGHRES_IMU `gyro_body` responds**
  — use `DroneState.gyro_body` for rate feedback in VQ2, not `angular_rate_body`.

### 3. Wire is real-time DURING active flight — not lagging (`logs/vq2_flight_health.json`)
Measured by [`telem_health.py`](telem_health.py) and re-confirmed inline by `ctbr_probe.py`
(`wire_health`) *during* a live climbing + pitch-step VQ2 flight (display/GPU lag the operator
sees is decoupled from this):
- **real-time factor = 1.00** (sim clock tracks wall-clock; idle 1.000, under 100 Hz control 1.001,
  during active flight 1.001) → physics is **not** running slow.
- full message rates: HIGHRES_IMU ~117–119 Hz, ACTUATOR_OUTPUT_STATUS ~95–96 Hz, RACE_STATUS 4 Hz
  (RACE_STATUS present ⇒ genuinely active race, not a frozen replay).
- IMU inter-arrival: mean 8.3 ms, p99 ~10 ms, **max ~10–11 ms** under load/flight → negligible
  jitter, no stalls; 0 BAD_DATA; our 100 Hz command stream delivered at exactly 100.0 Hz.
- ⇒ lag is NOT a cause of the crashes; those are environmental (start-gate spawn / ceiling).

### 4. Lift confirmed
With thrust ≥0.4 the drone climbs (accel net > g); at 0.6 the motors balance at 0.60 and it climbs
straight up until it hits the warehouse ceiling/next gate (~1 s) and crashes. The climb itself is
clean (balanced motors, gyro ≈0) — the crash is geometry, not control.

---

## What failed / VQ2-specific gotchas
- **Velocity setpoints (`SET_POSITION_TARGET_LOCAL_NED`) in ACRO** → tumble + COLLISIONs (recon,
  load day). ACRO has no velocity/position controller. Use CTBR.
- **Attitude-quat (angle) `SET_ATTITUDE_TARGET`** → not stabilized in ACRO (quat ignored); behaves
  as collective-only. Only body rates + thrust are honored.
- **VQ2 is a confined indoor course and the drone spawns *inside* the start gate.** Low thrust sags
  into the gate; high thrust climbs into the ceiling. Expect gate (id 1001) contacts during any
  stationary hover — they are environmental, not a control problem.
- **A crash ENDS the VQ2 episode and returns to the main menu**, after which the MAVLink wire keeps
  **replaying the last frozen state** (accel byte-constant, motors frozen). `ctbr_probe.py` has a
  `live` guard so this dead state is never mistaken for a clean hover.
- **`MAV_CMD_SIM_RESET` (31000) does NOT recover a crashed VQ2 episode** (it works on VQ1). After a
  VQ2 crash you must re-navigate the menu (or relaunch). Returning from a flight's pause menu can
  also leave the event list in a state where `enter` won't re-open the confirm screen — a clean
  relaunch is the reliable reset.

## Reproduce (VQ2)
```bash
# launch + navigate to VQ2:
"C:/Users/Shadow/Downloads/AI-GP Simulator v1.0.3379/AIGP_3379/FlightSim.exe" &   # then sleep 8
SF="python C:/Users/Shadow/Peregrine/scripts/sim_focus.py"
$SF keys enter; sleep 4; $SF keys enter; sleep 6      # any-button -> cached login -> ACTIVE EVENTS
$SF keys down down down; sleep 1                       # clamp on R2 - TRAINING (= VQ2)
$SF keys enter; sleep 5; $SF keys enter; sleep 7       # confirm "PILOT READY" -> RACE -> flight
# then, immediately (before it drifts into the gate):
cd handoff/vq2-control-handshake-2026-06-29
PY="C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe"
$PY ctbr_probe.py --regime acro --thrust 0.35 --settle 0.2 --hold 0.6 \
   --step pitch --step-rate 0.4 --step-s 0.7 --out logs/vq2_pitchstep.json
```

## Handoff to the deploy loop
`rl/fly_rl.py` already sends `ControlMode.BODY_RATE`. Two VQ2 deltas:
1. **Rate feedback must come from `gyro_body` (raw HIGHRES_IMU), not `angular_rate_body`** (ODOMETRY
   rate is blocked/zero in VQ2).
2. **Body-rate command → realized-rate gain is ~2.5×** — calibrate the controller's rate scale (or
   lean on the closed-loop policy) for precise tracking.

VQ1 cross-check artifacts (identical recipe, different render/environment) are kept under
`logs/vq1_*` for comparison.
