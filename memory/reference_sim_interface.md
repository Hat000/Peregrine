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
(branch `red-team-tier-a`, commit 5501d43). **FIRST CONTACT SUCCEEDED 2026-06-02 on ShadowPC (co-located w/ FlightSim.exe) — the must-verify list is RESOLVED below; the two branches were unified afterward.**

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
- Spec §3.8 "VFoV=90°" is the HFoV; true VFoV≈58.7° (frames.py handles it). Full §3.8 camera spec (reviewed 2026-06-14): camera tilt = **20° UP from body +X (MAV_FRAME_BODY_NED), SPEC-EXACT, NO tolerance, mount FIXED**; camera origin = body origin (zero translational offset, no lever arm); pinhole, no distortion; 640×360; [cx,cy]=[320,180]; [fx,fy]=[320,320] (square pixels); vision stream 30 Hz JPEG UDP:5600; HIGHRES_IMU + ATTITUDE + TIMESYNC; body↔IMU = identity; physics 120 Hz, command <100 Hz. 🚩 **VFoV=90° is MISLABELED — it is HFoV. True VFoV≈58.7° (fy=320, H=360). ALWAYS use fy=320 for vertical; never literal 90° VFoV. Audit render-prediction AND PnP for fy=320 consistency.**
- §3.3/README say no absolute/global position — but LOCAL_POSITION_NED+ODOMETRY (local NED from arm point) ARE sent. Reconciles (local ≠ global).

## FIRST CONTACT 2026-06-02 — SUCCEEDED (ShadowPC). Source: `handoff/shadowpc-firstcontact-2026-06-02/HANDOFF.md`
The 5 must-verify items, RESOLVED LIVE:
1. **Position+velocity = GIVEN, pristine ground-truth** (LOCAL_POSITION_NED 97 Hz + ODOMETRY 75 Hz; drone parks at origin, v=0; mag present). Baro FIELD present but VALUE=NaN (unusable) → take z from given position. Localization collapses for VQ1 — but KEEP vision→KF in-loop (walking-skeleton). VQ2 may turn position OFF → then z also vision-only.
2. **Gate map: width=height=2.72 m = the OUTER square; inner opening ~1.5 m is what PnP uses — do NOT feed 2.72 to inner-corner PnP.** Full ordered map, 6 gates, deterministic course (capture once, reuse). Sample: g0~(-23.3,-0.4,0), g1(-46.9,-2.5,+5.1), g2(-74.6,+1.2,+13.7)… (descending/curving; NED +z=down). Saved to `handoff/.../track_map.json`. **Trigger (R4):** broadcast ONCE at race/level LOAD to clients ALREADY connected (single ~230 B chunk, packets=1, re-sent ~4×; HANDSHAKE then ENCAPSULATED_DATA data[0]==2). PROCEDURE: connect at the HOME PAGE (`wait_heartbeat=False`), THEN navigate home→waiting→Race. `scripts/capture_track_map.py` banks it.
3. **Rates:** ATTITUDE/IMU 120, LOCAL_POS/ACTUATOR 97, ODOMETRY 75, HEARTBEAT 10, RACE_STATUS 4 Hz. **Video: true ~28.6 fps but the sim RE-SENDS each frame ~14× (~395/s) → DEDUP by frame_id (DONE, `jpeg_receiver`, commit 04a4bc8).** Command rate: 250 Hz is the sample's USED value, real max still unprobed.
4. **SET_ACTUATOR_CONTROL_TARGET scaling still UNPROBED** (8 ch = [FL,FR,BL,BR,0,0,0,0]).
5. **Arming = plain `MAV_CMD_COMPONENT_ARM_DISARM` p1=1** — ACCEPTED in 83 ms, NO force, NO OFFBOARD/GUIDED switch, NO pre-arm setpoint stream. TIMESYNC is client-initiated (`timesync_send` ts1=0 @10 Hz; sim DOES respond).

### New first-contact findings (capture — hard to re-derive)
- **🚩 ATTITUDE.pitch is SIGN-INVERTED; ATTITUDE Euler roll is ALSO inverted — use ODOMETRY quat.** ODOMETRY-quat = accel-gravity = FPV view all agree nose-DOWN; ATTITUDE Euler disagrees. **Orientation driven SOLELY from the ODOMETRY quaternion** (commit 2b81fc1). The 35.6° "attitude bias" = THIS, **NOT an ESKF-bias trigger.** The ODOMETRY quat is the TRUE attitude AS-IS — no roll inversion on the quat itself (confirmed 2026-06-12 tilted-phase quat-FD; the "roll-inversion artifact" belongs only to the ATTITUDE Euler message, not the quat). See [[project-ctbr-control-sysid]] 2026-06-12 update for full three-layer sign convention.
- **Control mode FOLLOWS the setpoint type:** body-rate→ACRO, attitude-quat→ANGLE. Live in ACRO: velocity→no climb; **position→VIOLENT 57 m runaway for a 1 m cmd** (acro has no position loop → misinterpreted, NOT a position controller); attitude + body_rate respond. ⇒ **fly on CTBR**; RE-TEST position/velocity easy-mode in ANGLE mode (send an attitude-quat first to enter angle).
- **Clocks INDEPENDENT:** video `sim_time_ns` = server UNIX-epoch ns; IMU sim_time = sim-boot-relative. Cross-stream align on `recv_monotonic` (delayed-vision = SUSPECT branch; fine for VQ1 localhost).
- **Sim PAUSES physics off-race** (home page / after the ~8-min cap): MAVLink keeps streaming but `sim_time` FREEZES → live control + system-ID need an ACTIVE running race.
- **Course (VQ1 FPV):** desaturated/high-contrast; red square gates receding (active brightest); blue racing-line aids ON (OFF in VQ2); gray block-wall corridor = obstacles. Real frames at `data/runs/20260602_000538_firstcontact_race1` (944 MB / 1797 unique frames — on ShadowPC, NOT pushed/gitignored).
- **R8 sim-reset (MAV_CMD 31000) still MURKY** — did NOT visibly reset race state (started stayed True), did NOT re-broadcast the map.

### Probes added (committed, `scripts/`): `capture_track_map.py` (connect-at-home → saves map JSON; RUN to bank), `timesync_probe.py`, `gatemap_probe.py`, `control_mode_probe.py` (patched: climb from position.z since baro NaN).
### Apply-before-vision/control TODOs: ✅ jpeg dedup by frame_id (DONE, 04a4bc8) · re-test pos/vel in ANGLE mode · probe real max setpoint rate · SET_ACTUATOR_CONTROL_TARGET scaling · `innerloop_step` system-ID (needs a running race).

Runbook: `docs/first_contact.md`. Related: [[project-master-plan]], [[reference-competition-materials]], [[project-detector-training-pipeline]], [[project-red-team-pass-2]] (deferred ESKF/delayed-vision triggers).

## 🚩 WIRE-SPEC DISCREPANCY — OUR SIM vs OFFICIAL SPEC (2026-06-14; COWORK-1)
Our ShadowPC sim (substrate for ALL our frame-convention work) streams **LOCAL_POSITION_NED (97 Hz) + ODOMETRY (75 Hz)** — position IS on the wire. The OFFICIAL public spec VADR-TS-002 §4.3 lists ONLY HEARTBEAT/ATTITUDE/HIGHRES_IMU/TIMESYNC (§4.5 adds "linear velocities"; §3.3 "absolute global position NOT exposed") → **the official eval does NOT stream position.** Likely our sim = the Elodin practice rig or an early/permissive build.
**IMPLICATIONS:** (1) the DEPLOYED stack MUST self-localize (case-C gate-relative — built); any case-A/given-pose path BREAKS on the official eval. (2) all LPN/ODOMETRY-dependent tooling (GT velocity from LOCAL_POSITION_NED, the R_y(π) conventions, L3 recordings, frame_residual_report) is OFFLINE-CALIBRATION-ONLY, tied to OUR sim, may NOT transfer. (3) IF the official sim DOES stream "linear velocities," the cold-velocity-prior problem EASES (velocity given, only position estimated) — UNVERIFIED.
**ACTION:** the moment the official VQ1 sim drops, RE-VERIFY the entire wire (messages? velocity? position? ODOMETRY? conventions? rates? arming handshake?) BEFORE trusting any LPN/ODOMETRY tooling on it. Scheduled sim-release monitor authorized (Cowork).
- 🚩 **NOT "waiting for a sim" — we HAVE a VQ1 sim + inc7 passed 5/5 ON IT (LOCAL, not officially graded) (Fengyou corrected framing, 2026-06-15).** The open question is PROVENANCE, not availability: **is our sim the official GRADED build, or the early Elodin/permissive practice rig?** FAQ "interface CONSISTENT between both rounds" makes position **ALL-OR-NOTHING across VQ1/VQ2** (RULES OUT "VQ1 has position, VQ2 strips it"): EITHER our sim is representative → position streamed BOTH rounds → inc7-given-position viable (self-loc = insurance/speed) — OR our sim is permissive → graded eval streams NO position EITHER round → inc8 self-loc MANDATORY. **ONE test settles both rounds. DEFINITIVE TEST (doable NOW, not "when it drops"): (a) check our sim's provenance/version vs the official downloadable package; (b) is the official package even RELEASED yet (FAQ "download once released" = future tense → maybe not).** Build inc8 self-loc REGARDLESS (robust superset). Our VQ1 pass = LOCAL sim; official graded submission is where the wire question gets its BINDING answer.
