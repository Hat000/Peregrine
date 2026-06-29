# VQ2 load-day recon — 2026-06-29

**Sim:** AI-GP `FlightSim.exe` → `DCGame-Win64-Shipping`, **build 1.0.3379** (HUD-confirmed).
**Track (HUD banner):** *"Now You See Me, Now You Don't"* — a dark warehouse/hangar with
numbered stations 01–20 and glowing-red gates (gate seen labeled **"R1-G7"**).
**Connection:** MAVLink `udpin:127.0.0.1:14550`, video JPEG-UDP `:5600`, 24-byte header `<IHHIIQ`
(unchanged from VQ1). Endpoints verified live.
**Captured by:** standalone `recon_capture.py` (this dir) — the spec's `scripts/vq2_loadday/`
kit **does not exist** in any of 558 commits or on disk, so I built a minimal pymavlink+JPEG
listener per the fallback instruction. It reuses the repo's `racer.vision.jpeg_receiver` and the
wire layout documented in `src/racer/mavlink_client.py`.

---

## 1. THE WIRE — present/absent per mode (PRIORITY 1)

Three passes, all binding 14550+5600 in one process:
- **training** — 50 s, passive (TIMESYNC@10Hz keepalive, no GCS heartbeat). R2-TRAINING event.
- **training_active** — 35 s, GCS-HEARTBEAT keepalive + ARM + forward-velocity setpoints (confound
  control + motion). R2-TRAINING event.
- **submission** — 40 s, passive. **R2-SUBMISSION** event (= the scored/competition wire).

| Message | training | training_active | submission (comp) | rate | notes |
|---|---|---|---|---|---|
| HEARTBEAT | ✅ | ✅ | ✅ | ~10 Hz | autopilot=0 (GENERIC), type=2 (QUADROTOR), base_mode=193 (ARMED+MANUAL+CUSTOM), status=ACTIVE |
| HIGHRES_IMU | ✅ | ✅ | ✅ | **~117 Hz** | accel + gyro only — see §1a |
| TIMESYNC | ✅* | ❌ | ✅* | ~10 Hz | *RESPONSE-ONLY: present only when WE send timesync; sim does **not** stream it unsolicited |
| ENCAPSULATED_DATA | ✅ | ✅ | ✅ | 4 Hz | carries **RACE_STATUS only** (data_type=1): active_gate_index + race timing. **No** TRACK_INFO/gate-map |
| ACTUATOR_OUTPUT_STATUS | ✅ | ✅ | ✅ | ~95 Hz | 4 motor outputs (idle 0.05) |
| COMMAND_ACK | — | ✅ | — | event | ARM (cmd 400) → result 0 (ACCEPTED). Command path works |
| COLLISION | — | ✅ | — | event | fired on contact: id=1002 (environment), threat_level=2 |
| **ATTITUDE** | ❌ | ❌ | ❌ | — | **BLOCKED** |
| **LOCAL_POSITION_NED** | ❌ | ❌ | ❌ | — | **BLOCKED** |
| **ODOMETRY** | ❌ | ❌ | ❌ | — | **BLOCKED** |
| GLOBAL_POSITION_INT | ❌ | ❌ | ❌ | — | absent |
| GATE_INFO / TRACK_INFO | ❌ | ❌ | ❌ | — | **no gate map on the wire** |
| DATA_TRANSMISSION_HANDSHAKE | ❌ | ❌ | ❌ | — | absent (no chunked TRACK_INFO transfer) |
| STATUSTEXT | ❌ | ❌ | ❌ | — | none observed |

### VERDICT: spec VADR-TS-003 §9.3 **CONFIRMED — and stronger than stated.**
- Position/attitude/odometry are **BLOCKED in BOTH training and competition.** The spec's hedge
  "TRAINING mode may expose more" is **REFUTED** — the two wires are byte-identical in message set.
- This is a **change from VQ1** (build 3364), where the sim emitted LOCAL_POSITION_NED + ODOMETRY
  (full pose) + a chunked TRACK_INFO gate map. VQ2 (3379) strips **all** absolute pose and the gate
  map. Our self-localization design assumption holds: **vision + IMU only, no free pose, no map.**
- **Confound ruled out:** absence persists with a GCS heartbeat, while armed, and while commanding
  motion (`training_active`) — it is not gated on client behaviour or armed/active state.

### 1a. HIGHRES_IMU contents (the key sub-field question)
`fields_updated` bitmask = **63 (0b111111)** in all modes → only bits 0–5 set.
- **accel xacc/yacc/zacc — PRESENT** (at rest reads ~(-3.0, 0, -9.34) m/s²; gravity-dominated).
- **gyro xgyro/ygyro/zgyro — PRESENT** (0.0 at rest, field is carried; bit set).
- **mag xmag/ymag/zmag — ABSENT (NaN)**, mag bit unset. ⚠️ **No magnetometer.**
- **baro abs_pressure/pressure_alt/temperature — ABSENT (NaN).** ⚠️ **No barometer.**

➡️ The "HIGHRES_IMU = accel/gyro/mag" assumption is **partly wrong**: accel+gyro yes, **mag NO**.
No free yaw reference (mag) and no free altitude reference (baro). Yaw/heading and z must come from
vision. Flag this to whoever owns the attitude/heading filter.

---

## 2. APPEARANCE & GLOW BLOOM (PRIORITY 1/2)

Real 640×360 camera frames (see `frames/curated/`, contact sheet `frames/curated_contact_sheet.png`).
Video stream healthy: ~25–28 fps decoded, max datagram 1400 B (< MTU, no IP-frag), ≤35 chunks/frame,
sim re-sends each frame ~14× (the known dedup'd "395 fps" illusion).

**Look:** very **low-light / high-contrast**, exactly as warned. Scene mean gray ≈ **36 / 255**
(dark); most of the frame is near-black structure with bright point sources (ceiling light grids),
**glowing red gates**, **blue floor lane-lines** converging to the next gate, and green/red pole
markers. Per-channel means are near-neutral overall (R≈G≈B≈36) — the red is **localized to the gate
glow**, not a global tint. ~2.4 % of pixels are "red-glow", ~1.4 % saturated-red, on a typical
near-gate frame.

**Bloom assessment (the range-bias concern):** measured on a clean near gate
(`01_gate_deadahead_near_static_R1G7.png`), horizontal scanline across the top bar:
- Saturated-red **core** (R≥250) = a sharply bounded ~76 px band.
- Edge falloff core→background: **~3 px on the crisp (inner/right) edge, ~12 px on the soft (outer)
  edge** — i.e. an asymmetric bloom halo of single-to-low-double-digit pixels.
- Plus a **pervasive low-amplitude red wash** across the upper scene (ceiling/ambient + multiple
  distant gates glowing), so a *low* red threshold over-segments badly.

➡️ **Implication for corner-span range estimation:** the bloom is real but **the saturated core edge
is reasonably crisp (~3 px)**. If you key the gate corners off the **saturated core (R≥250)** the
apparent square is only mildly inflated; if you key off a low red/brightness threshold you'll pick up
the ~12 px outer halo + ambient wash and **over-estimate gate size → bias range NEAR**. Recommend
PnP/corner extraction threshold on the saturated core, and validate corner repeatability at range on
frames 03/04 (far gates). At extreme near range the bar saturates and blooms strongly (frames 06/09)
— expect corner blow-out when very close.

Curated frames cover: dead-ahead near (01, 02), far/small + warehouse overview with multiple gates
(03, 04), gate at angle / aisle (05, 06), between-gates / no gate (07, 08), extreme-near red bloom
(09), and motion blur (10, 11, 12). `00_bloom_gatecrop_4x.png` is a 4× nearest-neighbour crop of a
gate for visual bloom inspection.

---

## 3. TRACK CHECK (PRIORITY 3)

**Cannot log gate positions or drone path** — position/odometry and the gate map (TRACK_INFO) are
**all blocked in training too** (see §1). The only positional signal on the wire is
RACE_STATUS.active_gate_index (an index, not coordinates). So track identity is **visual-only**:
- Track is named **"Now You See Me, Now You Don't"** (HUD banner) — a themed dark warehouse with
  numbered stations 01–20, glowing-red gates (one labeled "R1-G7"), blue lane-lines, parked drones.
- This does **not** match a known VQ1 layout in any way we can confirm numerically (no map to diff).
  Treat as a **new/unknown track**; the gate map must be built from vision, not received.

---

## 4. CONTROL / lifecycle notes (incidental)
- ARM via MAV_CMD_COMPONENT_ARM_DISARM (400) → **ACCEPTED** (result 0).
- Forward velocity setpoints (`SET_POSITION_TARGET_LOCAL_NED`, vel-only mask) while in the sim's
  default **ACRO** mode did **not** produce clean forward flight — the drone tumbled and logged
  environment COLLISIONs. Consistent with prior notes that velocity/position modes don't actuate in
  ACRO; reaching ANGLE/position likely needs the no-heartbeat (TIMESYNC-only) regime + a mode/handshake.
  **R2 not fully settled here — flagged, not in scope for recon.**

## 5. Reproduce
```
# sim up to flight (FlightSim.exe → Enter, Enter, Down×4, Enter, Enter)
python handoff/vq2-recon-2026-06-29/recon_capture.py --mode training   --seconds 50 --save-every 50 --outdir handoff/vq2-recon-2026-06-29
python handoff/vq2-recon-2026-06-29/recon_capture.py --mode training_active --seconds 35 --keepalive heartbeat --control fwd --outdir ...
python handoff/vq2-recon-2026-06-29/recon_capture.py --mode submission --seconds 40 --outdir ...   # R2-SUBMISSION event
```
Probe logs: `logs/wire_inventory_*.json` (full message-type histogram + first-instance field dumps +
IMU sub-fields) and `logs/frame_stats_*.json` (per-frame brightness/red/bloom stats for every frame).

## Headlines for the commander
1. **Pose is gone, in BOTH modes.** No ATTITUDE/LOCAL_POSITION_NED/ODOMETRY/GLOBAL_POSITION; no gate
   map (TRACK_INFO). Spec §9.3 confirmed and training is *not* more generous. Big change vs VQ1.
2. **IMU = accel + gyro ONLY. No mag, no baro (both NaN).** Lose the free yaw + free altitude we may
   have assumed. Vision must carry heading and z.
3. Wire = HEARTBEAT(10) + HIGHRES_IMU(117) + 30 Hz JPEG cam + RACE_STATUS(4, active-gate-index) +
   ACTUATOR_OUTPUT_STATUS(95); TIMESYNC is response-only.
4. Glow bloom is real but the **saturated core edge is ~3 px crisp** — threshold corners on the
   saturated red core, not a low red threshold (which catches a ~12 px halo + ambient wash → near-bias).
5. New track "Now You See Me, Now You Don't"; gate map must be built from vision (none on wire).
