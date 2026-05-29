---
name: reference-competition-materials
description: Source documents for the AI Grand Prix competition — technical spec PDF and the official updates page.
metadata: 
  node_type: memory
  type: reference
  originSessionId: 381822b6-b523-4990-a2cf-3739f8593be8
---

**Technical specification (authoritative for sim interface):**
- Path: `C:\Users\Fengy\Downloads\Projects\Anduril\260508_Technical_Spec_0002.pdf`
- Document ID: VADR-TS-002, Issue 00.02, 2026-05-08
- 11 pages, covers: simulation environment, drone/gate dimensions, coordinate frames, MAVLink protocol, vision stream packet format, contestant runtime, qualification phase rules
- Revision note: issue 00.02 added camera details (incl. the 20° upward tilt). Watch for further revisions.

**Competition updates page:**
- URL: https://www.theaigrandprix.com/previousupdates/
- Source for stage dates, eligibility, rules, prizes, registration flow

**Official rules page:** https://theaigrandprix.com/official-rules  (FAQ at theaigrandprix.com/#faq)

**Grounded rules facts (verified 2026-05-28, do not re-assume):**
- Gates must be passed in CORRECT ORDER; runs must pass gates "to count." Missed/out-of-order ⇒ invalid run. VQ2 ranks on fastest VALID time.
- FAQ: "The drone will not know the track... Gate position details may be provided only at a rough level; flight-path optimization is the team's responsibility." (So coarse gate data MAY be provided — fidelity unknown.)
- VQ1: desaturated, gates highlighted, "visual guidance aids may be active." VQ2: aids OFF, "real 3D-scanned environment" (photoreal).
- Software: external libraries/compilers/acceleration layers permitted; gen-AI coding tools allowed. FLOSS allowed BUT must be disclosed in writing to organizers + must not violate third-party license terms. ⇒ YOLO (AGPL) is usable; obligation is disclosure + AGPL compliance (trivial for internal/non-distributed use).
- IP: team retains ownership during + after; competition gets limited run/judge license ending by 2026 finals.

**Genuinely UNKNOWN (not published — must verify at first sim contact, do NOT assume):**
- How gate order is communicated; fidelity of any provided gate positions.
- What guidance aids actually do (next-gate? path? just highlighting?).
- Missed-gate mechanic (hard-invalid vs penalty time).
- Whether velocity is in the telemetry stream (spec mentions it, no carrying message listed) — biggest swing on VQ2 difficulty.
- Quality of the sim's inner-loop controller (decides position/velocity setpoints vs CTBR control).

**Useful external references for the work:**
- MAVLink 2 message definitions: https://mavlink.io/en/guide/mavlink_2.html
- MAVLink C library: https://github.com/mavlink/c_library_v2
- pymavlink / MAVSDK-Python for the actual client

**Velocity-telemetry check (2026-05-29, re-verified):** grepped the spec — "velocit" appears ONCE (§4.5 prose "linear velocities"). The §4.3 message table (authoritative) carries NO linear velocity: ATTITUDE = orientation + angular rates; HIGHRES_IMU = linear ACCELERATION + gyro + mag + baro; + HEARTBEAT, TIMESYNC. ⇒ velocity is NOT confirmed-available; design estimator to derive it (IMU integ + vision finite-diff + drag), confirm via msg_audit at first contact. (A first-contact `msg_audit` script that logs every received MAVLink msg ID/rate is the definitive resolver.)

**Web-search facts (2026-05-29; live FAQ/official-rules/previousupdates pages returned HTTP 504 — could not re-verify them directly, used cached + search):**
- Schedule: virtual quals **May–July 2026**, Round 2 cutoff ~end July; in-person **physical qualifier Sept 2026 (Southern California)**; **finals Nov 2026 (Ohio)**. $500K; Anduril + DCL + Neros + JobsOhio.
- Sensor suite (physical drone module): single **FPV camera (~12MP wide-angle) + IMU (gyro/accel), NO LiDAR**. (NB: the SIM camera is only **640×360** per spec — do not conflate; monocular + IMU, no depth, reinforces vision-primary.)
- **Onboard edge compute ~100 TOPS (FAQ, 2026-05-29):** the drone's onboard AI accelerator is Jetson-Orin-class (~100 TOPS INT8 NN inference) — generous, and our current stack uses a few % of it. Design the autonomy to USE it (heavier learned perception + a learned policy + learned depth), within the latency budget. NB TOPS = NN-inference capacity; classical KF/PnP/QP consume ~0. Most directly governs the physical qualifier/deployment envelope; VERIFY whether the virtual-qual sim enforces a compute/latency cap. Strategic implications: [[project-master-plan]] "Edge-compute budget" layer.
- **Telemetry-protocol wrinkle (third-party claim, Elodin harness post — verify):** a "practice rig" reportedly uses **Betaflight UDP packets, not MAVLink**, so tooling built there may need a shim to talk to the real qualifier sim. Our spec (VADR-TS-002) is explicit MAVLink and is authoritative for the real sim. (Aside: Betaflight-SITL UDP state packets *do* include velocity — another reason to settle R1 on the REAL sim.)
- Possible external resource: Elodin AI Grand Prix race-sim harness (elodin.systems) — license/trust TBD before any reuse.

**Research pass (2026-05-28) — new corroborated facts:**
- **Live official site is now `dcl-project.com`** (theaigrandprix.com redirects/flaky). Official rules: https://www.dcl-project.com/official-rules/ (covers eligibility/structure/legal; NOT operational scoring details). Org contact: info@theaigrandprix.com. Founded by Anduril + DCL + Neros + JobsOhio. Teams up to 8, no fee.
- **Attempts/scoring model (corroborated, VERIFY w/ organizers):** time-trial on ONE shared course; "teams can attempt runs any time within the qualification window; once all gates are passed, the run time is locked" ⇒ effectively **unlimited attempts, best valid time counts.** Track is FIXED/shared, same across your attempts (you just don't get a precise map up front). ⇒ map-once-then-iterate is legal/optimal; free to optimize on the real sim. *(Sources were intermittently 504; weight as likely, not certain.)*
- **FAQ re-confirmed:** "the drone will not know the track. Teams must detect gates and navigate using onboard sensing (primarily vision). Gate position details may be provided only at a rough level."
- **Elodin practice harness — github.com/elodin-sys/ai-grand-prix, APACHE-2.0 (reusable w/ attribution).** Real Betaflight SITL + an ~80-line UDP bridge, 1 kHz lockstep, DETERMINISTIC, GPU-rendered camera. Runs macOS/Ubuntu/**WSL** (⇒ laptop + Adroit). Camera matches official spec EXACTLY (640×360, fx=fy=320, cx=320, cy=180, +20° tilt, 30 Hz). SensorUpdate: t, tick, world pose+vel (Elodin's convenience — official sim does NOT give pose), gyro, accel, baro, mag, optional frame_rgba, `*_fresh` flags; "no GPS/depth/RPM, matching the official sim's telemetry contract." Output = RCCommand PWM 1000–2000 (throttle, roll, pitch [<1500=fwd], yaw, arm AUX1) → Betaflight RC/mixing/PID. Built in Rust+Bevy. **Differences from official:** Betaflight-UDP not MAVLink (need shim); ENU not NED; simpler aero (single drag coeff, no battery sag). USE as dev/surrogate/estimator-validation rig NOW; it is "community fan-art," NOT the scoring sim. **Source-read findings (2026-05-28, from cloning the repo): API = `autopilot(SensorUpdate)->RCCommand` Python callback @1kHz (NOT MAVLink — our `racer/elodin_adapter.py` adapts it). SensorUpdate = t, tick, world_pos[qx,qy,qz,qw,x,y,z] (ENU, scalar-last, GROUND TRUTH — official sim won't give pose), world_vel, gyro/accel (FLU body), baro (=ALTITUDE not pressure), mag (body dir of ENU-North), frame_rgba(640×360×4), last_gate_passed/next_gate_index + `*_fresh` flags. RCCommand = throttle/roll/pitch/yaw/arm/aux PWM 1000–2000 (AETR, arm≥1700, ANGLE mode default). Frames: world ENU / body FLU. ⚠️ `sim/camera.py` sets camera `far=0.65 m` → hides 10 m gates; patch before vision dev. No Windows wheel (Linux/WSL/macOS only). Elodin independently confirms VFoV≈58.72° (prose 90°=HFoV).**
- **Python 3.14 has NO CUDA PyTorch wheels yet (CPU-only)** — pytorch/pytorch#169929. Run the ML/autonomy stack on **Python 3.12** (CUDA-supported) on Azure VM + Adroit; spec's "3.14.2" is comms-only & "other environments allowed."
- **Control mental model (from Elodin + spec §5.3 "Pilot Commands → Stabilized Controller"):** we are the FPV-pilot brain over a Betaflight-like stabilizer. SET_ATTITUDE_TARGET = attitude-mode (angle) or body-rate-mode (acro/CTBR) + collective thrust. SET_POSITION_TARGET = possible "easy mode" if the stabilizer has a position loop on internal ground truth — TEST first contact.

Related: [[project-ai-grand-prix]], [[project-master-plan]]
