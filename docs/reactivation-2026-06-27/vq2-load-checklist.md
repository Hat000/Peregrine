# VQ2 Load-Day Checklist — inspect the moment the competition sim drops

*Readiness artifact. Every check run TWICE where it matters (Training-mode vs Competitive-mode connection) — the DELTA is the finding. Pre-stage the §3 inspector scripts DURING the build phase so load-day is pure inspection. Full design in session transcript.*

**Framing:** Competitive wire blocks ATTITUDE + LOCAL_POSITION_NED + ODOMETRY + GATE_INFO; only HIGHRES_IMU + 30 Hz JPEG + TIMESYNC + HEARTBEAT survive. Our `MavlinkClient` already parses every stream + presence flags, so blocked-vs-present is just reading which fields are non-None.

## Ordered checks (each: WHAT / WHY-fork-gated / HOW)
- **C0 Connectivity + mode handshake** — can we connect, get HEARTBEAT, distinguish Training vs Competitive? `MavlinkClient.connect` + `firstcontact.backend_summary` + STATUSTEXT. Gates everything.
- **C1 Blocked-telemetry (Competitive)** — are ATTITUDE/LPN/ODOMETRY/GATE_INFO actually absent? `wire_inventory.py` (MessageRateTracker + presence flags). → if ODOMETRY/LPN SURVIVE: stack collapses to case-A given-pose, AHRS fork dies, inc7/inc8 deploy direct. If GATE_INFO survives: recon-map free.
- **C5 Training-mode GT exposure** — does Training leak ODOMETRY/LPN/GATE_INFO? `wire_inventory.py --diff TRAINING COMPETITIVE`. → gates privileged distillation (Xing teacher→DAgger) AND offline boresight-calib AND VGGT gate-labeling. No GT → synthesize in our own twin (sim-to-sim gap).
- **C3 Track-fixed-WITHIN-a-load** — same gate geometry across laps of one load? `track_stability_probe.py` (capture_track_map in Training / detector+PnP in Competitive, diff). → **THE recon-map moat gate.** Per-lap shift → moat dead → reactive gate-relative + GRU only.
- **C2 Magnetometer fidelity** — usable mag in HIGHRES_IMU as a yaw anchor? `mag_probe.py` (correlate mag-heading vs Training ODOMETRY yaw). → AHRS yaw-anchor fork. No mag → vision-only heading → gate-in-view + GRU memory critical.
- **C4 Per-load randomization** — track same or randomized across loads? `track_stability_probe.py` across ≥3 loads. → generalist (layout-DR) vs specialized policy. Randomized + C3-fixed = the exact moat regime.
- **C6 Photorealism / appearance gap** — VQ2 renders vs our detector's training distribution? `detector_smoke.py` (detection rate + reproj error vs VQ1 baseline). → P5 photoreal detector + GS/NeRF hardening, or ship clean-ensemble as-is.
- **C7 Occlusion-gap frequency** — how often is the next gate NOT in FoV; longest blind stretch? `gap_audit.py` (consecutive-no-detection run-length). → Cioffi LIO gap-filler + GRU actor + terminal-coast, or lean estimator.
- **C8 IMU rate/noise/saturation** — actual Hz, noise floor, clipping at high-g? `imu_profile.py`. → learned vs classical AHRS; IMU-sat model-substitution; Modified-Polar/high-g EKF.
- **C9 Boresight/static-attitude bias** — does ε_vert≈0.215 m reproduce as a fixed offset? `firstcontact.sample_attitude_bias` + `frame_residual_report` (also re-validates R_y(π) on VQ2 ODOMETRY). → the gate-4 calibration-vs-SITT pre-check (Spike A on real VQ2).
- **C10 TIMESYNC latency + gate-ordering** — vision-to-IMU clock offset; does `active_gate_index` survive in Competitive? `clock_offset.py` + `wire_inventory.py`. → OOSM RewindKF horizon; free gate-ordering or visual-only.

## Priority (block-the-most-forks first)
**Tier A:** C0 → C1 → C5 → C3 → C2. **Tier B:** C4, C6, C7. **Tier C (tune chosen forks):** C8, C9, C10. Couplings: C5 before C3/C9 (they want GT); no-mag (C2) makes C7 safety-critical; C3∧C4 define the moat regime.

## Pre-staged tooling (write during build)
New single-command inspectors wrapping existing primitives: `wire_inventory.py` (master probe — C0/C1/C5/C10 in 30 s, `--diff` mode), `mag_probe.py` (C2), `track_stability_probe.py` (C3/C4), `detector_smoke.py` (C6), `gap_audit.py` (C7), `imu_profile.py` (C8), `clock_offset.py` (C10). Reuse as-is: `diagnose_session.py` + `frame_residual_report.py` (C9 frame-convention canary). Plus a mode-navigation note + record_session checklist.

## 🚩 OPEN LEGAL QUESTION (surfaced by the agent — gates an entire fork)
**Is a Training-derived artifact (pre-baked map, Training-calibrated boresight) LEGAL to ship under "code-audit on submit"?** If Competitive scoring forbids Training-derived constants, the offline-calibration fork is useless for the competitive run *regardless of what C5 finds*. RESOLVE before investing in offline calib/mapping. (Fengyou: inspect on VQ2 load / spec re-read — do NOT email organizers.)

---

## Post-review refinements (Gemini adversarial review, 2026-06-27)
- 🟢 **LEGALITY CLARIFIED (resolves the open gate above — GOOD NEWS read correctly):** Gemini's verdict = pre-shipping a TRACK MAP (environment state) = circumvention → DQ; but **CALIBRATION constants (boresight ε_vert = hardware intrinsic/extrinsic profile) = LEGAL.** ⇒ our **recon-map moat is FINE as designed** because it builds the map ONLINE during the recon lap at competition time (drone perceiving its environment live, like any SLAM) — NOT a pre-shipped map. Only the "pre-bake the training-track map and ship it in weights" shortcut dies (we weren't doing it). Offline boresight calib SURVIVES. **Still confirm exact wording on VQ2 load, but two forks stay open, one unused shortcut closes.**
- **C3/C4/C6 — AUDIT SEPARATELY (Gemini):** game engines often keep static geometry meshes but randomize texture seeds / lighting / wind per load. "Track fixed within a load" is NOT one binary — split into **geometry-stable vs appearance-stable vs ordering-stable**. A geometry-stable but appearance-unstable load (different shadows/lighting) breaks a relocalization pipeline that keys on appearance, even though the moat's geometry survives.
- 🚩 **CHEAPEST MOAT FALSIFIER — promote to the FIRST load-day action: THE 2-LOAD HASH (Gemini):** boot the sim twice; extract raw sim-GT XYZ of Gate-1 + capture one RGB(D) frame from the exact same spawn pose; if the gate coord shifts >5 cm OR the buffer hashes differ → the frozen-map strategy is falsified instantly, before writing a line of VGGT/DPVO code. Run this at C3/C4 — it's cheaper than `track_stability_probe.py` for the binary "is anything stable at all" question.
