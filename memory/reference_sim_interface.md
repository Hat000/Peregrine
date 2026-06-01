---
name: reference-sim-interface
description: "The AI-GP sim's ACTUAL MAVLink+video wire interface, confirmed from the shipped PyAIPilotExample reference client (vs the thinner VADR-TS-002 spec), plus the first-contact must-verify list."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 28d436be-ab10-41ec-bbe5-c7f3dd38cc9e
---

The official sim shipped 2026-06-01 as `AI-GP Simulator v1.0.3364.zip` = an inner `AIGP_3364.zip`
(`FlightSim.exe`, run + log in) + **`PyAIPilotExample`** (the official reference client) +
README. Extracted on the dev laptop at `C:\Users\Fengy\Downloads\AIGP_sim\`. The README's system
reqs: Win10/11, 8 GB RAM, GTX 970 min (RTX 3070 tested), 12 GB disk, **GPU-bound**. The example
client is authoritative for the WIRE; the spec message table (§4.3) is thinner/older — trust the
sample. Our `mavlink_client.py` + `jpeg_receiver.py` were cross-checked against both and aligned
(branch `red-team-tier-a`, commit 5501d43; `main` caught up at cfe7d82).

## Connection
- MAVLink: `udpin:127.0.0.1:14550` (CLIENT binds; sim connects to it). pymavlink `udp:` == `udpin:` (verified) so our default is fine. `wait_heartbeat()` then `target_system`.
- Video: JPEG-UDP, bind `0.0.0.0:5600`, header `<IHHIIQ` (24 B: frame_id u32, chunk_id u16, total_chunks u16, jpeg_size u32, payload_size u32, sim_time_ns u64) — **byte-for-byte == our `jpeg_receiver.HEADER_FMT`.** Everything is NED.

## Inbound telemetry the sim actually sends (sample handles all; we now do too)
HEARTBEAT, TIMESYNC, ATTITUDE (rpy + rates), HIGHRES_IMU (acc/gyro; mag/baro fields present but the sample ignores them), **LOCAL_POSITION_NED** (x,y,z + vx,vy,vz), **ODOMETRY** (full pose quat + vel + angular rate + `reset_counter`), **ENCAPSULATED_DATA**, **DATA_TRANSMISSION_HANDSHAKE**, **COLLISION**, **ACTUATOR_OUTPUT_STATUS**.

The sim REPURPOSES `ENCAPSULATED_DATA` (discriminator = `data[0]`):
- **RACE_STATUS** (type 1), `struct "<BQqqIq"` = data_type, sim_boot_ms, race_start_boot_ms(q; <0=not started), race_finish_ns(q; <0=ongoing), **active_gate_index**(I), last_gate_race_time(q).
- **TRACK_INFO** (type 2): announced by `DATA_TRANSMISSION_HANDSHAKE` (width=transfer_id, packets=chunk count), chunks = `"<BH"` (type, transfer_id) + slice, concatenated by `seqnr`. Full payload = `"<H"` num_gates then per gate `"<Hfffffffff"` (38 B) = gate_id, pos NED x/y/z, **quat w/x/y/z**, width, height. = the full ordered GATE MAP.
- **COLLISION**: `id` (1001=gate, 1002=environment), `threat_level` (1–2), `horizontal_minimum_delta` (impulse magnitude kg·m/s, misnamed).

## Control the sim accepts (Client→Sim)
- `SET_POSITION_TARGET_LOCAL_NED` (position OR velocity via type_mask).
- `SET_ATTITUDE_TARGET`: attitude-quat+thrust (mask 0b00000111) OR body-rate+thrust / CTBR (mask 0b10000000 = ATTITUDE_IGNORE).
- `SET_ACTUATOR_CONTROL_TARGET`: direct 8-ch motor/actuator — **the sample's DEFAULT path** (sends ~zero placeholders). Scaling ([-1,1] vs RPM vs throttle) UNKNOWN.
- Arm: COMMAND_LONG / MAV_CMD_COMPONENT_ARM_DISARM (p1=1). **Reset: COMMAND_LONG MAV_CMD 31000** (`send_sim_reset()`) — clean attempt loop. Sample sends NO client heartbeat (we send 2 Hz — harmless).

## Spec ↔ sample discrepancies (don't re-litigate)
- Spec §4.3 lists only 6 messages; the build sends ~11 + custom. Sample wins.
- Spec §4.4 command rate **<100 Hz** but the sample loops setpoints at **250 Hz** — confirm the real accepted rate. (Our probes stream ≤20 Hz, safe.)
- Spec §3.8 "VFoV=90°" is the HFoV; true VFoV≈58.7° (frames.py handles it).
- §3.3/README say no absolute/global position — but LOCAL_POSITION_NED+ODOMETRY (local NED from arm point) ARE sent. Reconciles (local ≠ global).

## MUST-VERIFY live at first contact (the cross-check surfaced these)
1. **Is LOCAL_POSITION_NED/ODOMETRY actually POPULATED (real ground-truth) or zero?** STRATEGY-DEFINING — if real, localization collapses (but keep vision→KF in-loop, walking-skeleton). Likely on in VQ1/dev, possibly off in VQ2.
2. **TRACK_INFO width/height = INNER (1.5 m) or OUTER (2.7 m)?** PnP needs inner. Is the map full or "rough" (FAQ)? Does it arrive automatically?
3. **Per-type RATES** — now auto-reported by `session_lifecycle`/`record_session` (`MessageRateTracker`). Expect camera 30 Hz, heartbeat ≥2 Hz, physics 120 Hz; ATTITUDE/IMU/ODOMETRY unspecified. + the 250-vs-<100 Hz command-rate truth.
4. **SET_ACTUATOR_CONTROL_TARGET scaling.**
5. **Arming handshake** — mode switch (OFFBOARD/GUIDED) or pre-arm setpoint stream needed?

Runbook with the ordered sweep: `docs/first_contact.md`. Related: [[project-master-plan]], [[reference-competition-materials]], [[project-red-team-pass-2]] (deferred ESKF/delayed-vision triggers).
