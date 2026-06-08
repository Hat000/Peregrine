# ShadowPC LIVE VERIFY (roadmap D) — handoff to the offline Commander. 2026-06-06

Live verify of `FAITHFUL_TUNED_GAINS` on the real sim. **Bottom line: the faithful config TRANSFERS where
it counts (signs/attitude/lateral, we-own-thrust), and the ONE issue is the altitude loop — a `kp_alt=4.0`
relay limit cycle — exactly the bounded rung-1 STOP we designed for. Fix it OFFLINE on the twin, then come
back for the gate.** No gate flown yet (correctly gated behind a clean altitude hold).

## Read order
1. `UNDERSTANDING.md` — the config flown, the pre-registered twin prediction, the ladder, live-vs-twin gaps.
2. `RUNG1_HOVER.md` — rung 1 (hover-hold / balloon), reproduced ×2, GUI-corroborated. OBSERVATION + INTERPRETATION.
3. `VPROBE.md` — the open-loop vertical probe (sink side captured; observation only, fit deferred).

## What was verified LIVE (the wins — high confidence, reproduced ×2 + GUI)
- **No balloon / no sag / no runaway.** Altitude is *held* (net drift ≤0.003 m/s). §7 balloon does not occur.
- **We own thrust in CTBR** — motors (ACTUATOR_OUTPUT_STATUS) track our commanded collective to ±0.002,
  never pinned. Task-3's verdict confirmed live again.
- **The faithful sign set transfers** — `body_rate_sign=[1,1,-1]` / `odo_att_sign=[-1,1,1]` /
  `odo_rate_sign=[-1,-1,1]`: the drone leveled the −18° resting tilt to 0°, no roll/yaw, ≤0.17 m lateral drift.

## The one issue (the offline job)
**The alt loop limit-cycles:** thrust bang-bangs both clips (0.05↔0.40, ~67% at the ceiling), TRUE vz
oscillates **±0.5 m/s at ~6 Hz**, altitude holds ~1.39 m (~0.1 m below the 1.5 m target). Cause = `kp_alt=4.0`
far too stiff for the live plant + the loop's true latency → a relay limit cycle. The 0.40 cap *reduced* the
amplitude; it is not the cause. The ideal twin can't reproduce this (it holds tautologically: its hover ≡ the
controller's, clean vz, no latency).

## Offline plan (the next session)
1. **Fit the real vertical plant** from `vprobe_sink_extract.json` + `vprobe_climb_extract.json` (BOTH sides):
   hover thrust (vz=0 crossing) + the two-sided thrust→accel slope (drag-corrected d(vz)/dt vs collective).
   **Hover is bracketed in (0.26, 0.28) ≈ 0.27** — 0.26 sinks, 0.28 climbs — i.e. ~the twin's 0.2656 (the
   rung-1 0.32 mean was the clip-duty-cycle artifact, confirmed). Open-loop climbs/sinks are SMOOTH (GUI:
   "very constant, barely any lurch") → the plant is fine; the limit cycle is purely the closed alt loop.
2. **Fit `cmd_latency_s`** from the **~0.16–0.20 s (~6 Hz)** limit-cycle period in `rung1_hover_run1/run2_extract.json`
   (full-rate vz + thrust series — resolves the 6 Hz cycle).
3. **Add both to the twin's vertical channel** (real hover/slope + a control/measurement latency) so it
   *reproduces* the limit cycle, then **re-tune `kp_alt`/`kd_alt` (much lower `kp_alt`)** until it holds cleanly
   (vz≈0). Consider whether the KF vz feedback (vs raw) helps or hurts.
4. Hand a new alt-loop config back to ShadowPC → re-fly rung 1 (must hold, vz≈0) → then rung 2 (gate 0,
   twin predicts closest 0.13 m) → rung 3 (course; twin 6/6, worst 0.61 m @ g1).

## Artifacts (this dir)
- `UNDERSTANDING.md`, `RUNG1_HOVER.md`, `VPROBE.md`
- `rung1_hover_run1_extract.json`, `rung1_hover_run2_extract.json` — hover-hold, FULL-rate (commanded
  thrust/body-rate + KF vz, TRUE LOCAL_POSITION_NED vz, ACTUATOR motors) + summary. **For the cmd_latency fit.**
- `vprobe_sink_extract.json` + `vprobe_climb_extract.json` — open-loop probe, both sides (collective/world-vz/motor
  per-phase series). **For the hover (≈0.27, bracketed 0.26–0.28) + two-sided thrust-slope fit.**

## Live wiring added this session (`red-team-tier-a`)
- `scripts/fly_vq1.py`: `--faithful` (single-source-of-truth controller/planner via `make_controller` +
  `FAITHFUL_TUNED_PLANNER`), `--hover-hold` (rung-1 fixed-target), `--max-tilt-deg` (tilt auto-abort),
  `--print-config`; cmd-log now logs roll/pitch; meta.json records the full `controller_config` + bounds.
- `scripts/extract_run.py`: the compact cmd+telemetry extractor (auto-detects fly_vq1 vs rate_sysid schema).
- 285 tests green; no `src/` behavior changes (the faithful gains/signs are unchanged — only wired into the live runner).

**Reproduce policy honored:** rung-1 limit cycle reproduced ×2 (GUI-corroborated). Bounds enforced every run
(force-disarm on exit; geofence/climb/tilt/collision/time aborts all fired correctly when hit). `memory/` not edited.
