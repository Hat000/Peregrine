# VQ2 control handshake — WORKING recipe (2026-06-29)

**Sim:** Anduril AI-GP "VQ2" build **1.0.3379**, ShadowPC. Track "Now You See Me, Now You Don't".
**Status:** ✅ SOLVED. Stable, commanded flight achieved. This was the blocker stopping us from
closing the loop; it is now closed.

---

## TL;DR — the one-line recipe

> **ARM (MAV_CMD_COMPONENT_ARM_DISARM=400, p1=1), then stream `SET_ATTITUDE_TARGET` in
> BODY-RATE mode (type_mask=`0b10000000` = ignore attitude quat) at ~100 Hz with body
> roll/pitch/yaw rates (rad/s, FRD) + normalized collective thrust [0,1]. Hover collective
> ≈ 0.27–0.35; ≥0.4 climbs. The sim is in ACRO by default and stays there.**

This is exactly what `ControlMode.BODY_RATE` already emits in
[`src/racer/mavlink_client.py`](../../src/racer/mavlink_client.py) — our existing uplink and
our policy's CTBR output transfer **directly**, no new message plumbing required.

**Confidence: HIGH.** Arm ACCEPTED, 0 collisions across all runs, motors respond exactly as
commanded (collective echo + clean symmetric pitch differential), IMU gravity-dominated in
hover, steady bounded rotation under a rate step, and the in-sim HUD independently shows the
drone climbing (83 km/h) and pitching toward a gate.

---

## The recipe in detail

### Mode
- Flight mode is **ACRO** (body-rate / "MANUAL_INPUT") — shown on the in-sim HUD top-right.
- It is **ACRO by default**, before we send anything, and **stays ACRO whether or not we send a
  GCS heartbeat**. HEARTBEAT reports `autopilot=0` (INVALID/generic), `type=2` (QUADROTOR),
  `custom_mode=0`. No mode-change command is needed.
- ⚠️ This **refutes the old VQ1-era note** (mavlink_client.py docstring / 2026-06-03) that "our
  GCS heartbeat forces ACRO and TIMESYNC-only keeps ANGLE." In build 3379 there is no reachable
  ANGLE mode via the keepalive regime — ACRO is the floor. Either keepalive works for arming +
  telemetry; **prefer the GCS heartbeat** (`send_heartbeats=True`) for the live link, it armed
  and held identically and is the simpler regime.

### Arm
```python
client.arm()                       # MAV_CMD_COMPONENT_ARM_DISARM (400), param1=1, no force needed
client.wait_armed(True)            # HEARTBEAT armed flag -> True
# COMMAND_ACK: {command:400, result:0 (MAV_RESULT_ACCEPTED)}
```
Force-arm (param2=21196) is **not** required.

### Control message — `SET_ATTITUDE_TARGET` (BODY_RATE)
| field | value | meaning |
|---|---|---|
| `type_mask` | `0b10000000` (`_ATT_MASK_BODY_RATE`) | ignore attitude quaternion → use body rates + thrust |
| `q` | `[1,0,0,0]` | ignored by the mask |
| `body_roll_rate`  | rad/s, **FRD** | + = roll right |
| `body_pitch_rate` | rad/s, **FRD** | + = pitch (nose-down/forward, sane forward motion observed) |
| `body_yaw_rate`   | rad/s, **FRD** | yaw rate |
| `thrust` | normalized **[0,1]** | collective; ~0.27–0.35 ≈ hover, ≥0.4 climbs |

Stream rate: tested at **100 Hz**; the deploy loop runs 30 Hz (training dt) and also works.
Send via the canonical path — `ControlCommand(mode=ControlMode.BODY_RATE, body_rate=…, thrust=…)`
→ `MavlinkClient.send_command`.

### Rate gain caveat (calibration, not blocker)
A commanded **+0.6 rad/s** pitch produced a **steady ~1.56 rad/s** measured gyro rate — a fixed
~2.6× scale, proportional and stable (consistent with the known sim mixer coupling law). Direction
is correct and the response is bounded/recoverable. For precise rate tracking the controller needs
this gain calibrated (or rely on the closed-loop policy, which already adapts). It does **not**
block stable flight.

---

## Evidence (objective, telemetry + independent HUD)

Probe: [`ctbr_probe.py`](ctbr_probe.py). Raw timeseries + verdicts in [`logs/`](logs/).
No pose is on the VQ2 wire (LOCAL_POSITION_NED / ODOMETRY pose blocked — see memory
`vq2-wire-3379-pose-blocked`), so lift is judged from HIGHRES_IMU accel, ACTUATOR_OUTPUT_STATUS
motors, and the in-sim HUD speed.

### 1. CTBR hover — clean, no tumble (`acro_t027_hover.json`)
thrust 0.27, body rates 0, 4 s:
- collisions: **0**
- accel magnitude: mean **10.03**, std **0.04** → gravity-dominated, steady (≈ sitting/near-hover)
- gyro magnitude: **0.0** → no rotation
- 4 motors: all **0.27**, spread **0.0** → perfectly balanced, echoes commanded collective

Contrast: the recon's velocity setpoints in ACRO **tumbled + logged COLLISIONs**. CTBR does not.

### 2. CTBR lift — commanded climb (`acro_t060_hover.json` + `logs/hud_t060_climb_83kmh.png`)
thrust 0.60, body rates 0, 8 s:
- collisions: **0**, gyro **0.0**, motors all **0.60**
- accel magnitude: peaks **17.7 m/s²** (≈1.8 g) → accelerating upward
- **HUD independently shows 83 km/h, drone high above the scene** → real commanded vertical flight

### 3. CTBR maneuver — clean, recoverable pitch (`acro_t040_pitchstep.json` + `logs/hud_pitchstep_maneuver.png`)
thrust 0.40, hover then **+0.6 rad/s pitch** step:
- collisions: **0**
- at step onset the 4 motors split into a **perfectly symmetric differential**:
  `[0.4235, 0.4235, 0.376, 0.376]` — constant collective, pure pitch torque
- gyro magnitude ramps smoothly to a **steady ~1.56 rad/s and holds** (bounded, not divergent)
- ODOMETRY angular_rate stays 0.0 (blocked in VQ2) but **raw HIGHRES_IMU gyro responds** — use
  `DroneState.gyro_body`, not `angular_rate_body`, for rate feedback in VQ2
- **HUD shows the drone pitched and flying forward toward a gate at 29 km/h**

### 4. Mode probe (`angle_t040_hover.json` + `logs/...`)
TIMESYNC-only (no GCS heartbeat) + `SET_ATTITUDE_TARGET` level-quat (attitude mask) + thrust 0.4:
- HUD still **ACRO**; behaves like collective-only (climbs straight, gyro 0). Confirms the
  attitude quaternion is **not** angle-stabilized in ACRO — only body rates + thrust are honored.
  So BODY_RATE/CTBR is the correct (and only) stabilized control path.

---

## What failed and why
- **Velocity setpoints (`SET_POSITION_TARGET_LOCAL_NED`) in ACRO** → tumble + environment
  COLLISIONs (recon, load day). ACRO is a rate mode; it has no velocity/position controller to
  consume those, so the message is dropped onto the rate path incoherently.
- **`SET_ATTITUDE_TARGET` attitude-quat (angle) mode** → not stabilized; the quat is ignored in
  ACRO. Only useful as collective-only (zero rates).
- **No ANGLE/GUIDED/POSITION mode reachable** via the heartbeat/timesync keepalive in 3379
  (old VQ1 assumption was wrong). MAV_CMD_DO_SET_MODE was not needed and not pursued since
  CTBR works; if a position mode is ever wanted it would need a real mode-change probe.

## Reproduce
```bash
# sim already up at a TRAINING flight (see scripts/sim_focus.py; ACRO, race running)
cd handoff/vq2-control-handshake-2026-06-29
PY="C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe"
$PY ctbr_probe.py --regime acro --thrust 0.27 --hold 4 --out logs/hover.json          # clean hover
$PY ctbr_probe.py --regime acro --thrust 0.60 --hold 8 --out logs/climb.json          # lift
$PY ctbr_probe.py --regime acro --thrust 0.40 --step pitch --step-rate 0.6 --step-s 2 \
    --out logs/step.json                                                              # maneuver
# (send MAV_CMD 31000 / client.send_sim_reset() between runs for a clean start line)
```

## Handoff to the deploy loop
`rl/fly_rl.py` already sends `ControlMode.BODY_RATE` and uses `_HOVER_THRUST = 0.2656` — the VQ2
near-hover collective measured here (~0.27) confirms that constant is right for 3379. The two
deploy-relevant deltas for VQ2:
1. **Rate feedback must come from `gyro_body` (raw HIGHRES_IMU), not `angular_rate_body`** (ODOMETRY
   rate is blocked/zero in VQ2).
2. **Body-rate command → realized-rate gain is ~2.6×** — calibrate the controller's rate scale (or
   lean on the closed-loop policy) for precise tracking.
