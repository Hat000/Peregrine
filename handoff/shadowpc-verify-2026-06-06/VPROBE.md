# Run B — open-loop vertical probe (hover + thrust-slope at the operating point). LIVE, ShadowPC 2026-06-06

**Purpose:** the offline alt-loop re-tune needs the REAL hover thrust + thrust→vertical-accel slope near
where we fly (the rung-1 bang-bang mean 0.32 is a clip-duty-cycle artifact, NOT hover). Instrument =
`scripts/rate_sysid.py --mode hover` (Task-3): a damped level-hold that steps the collective open-loop
while the ACTUATOR_OUTPUT_STATUS motor witness records what the rotors actually do.

**OBSERVATION ONLY below — the hover/slope FIT is the offline Commander's job (deferred per protocol).**

## What was flown (3 runs; the SINK + CLIMB sweeps are the data — both sides now)
0. `20260606_214138_vprobe` — aborted in `init_level` (1.5 s): the level-hold holds ATTITUDE level but has
   **no horizontal position hold**, so at the −18° resting tilt the drone dives forward ~g·tan18° ≈ 3 m/s²
   and **drifted 4 m → geofence abort** (rung-1 4 m geofence wrong for this instrument; Task 3 used 18 m).
   No vertical data. Reran with `--max-offset-m 18 --kp-hold 2.0`.

**SINK sweep — `20260606_214504_vprobe` (`vprobe_sink_extract.json`).** Init 0.27, steps 0.24→0.32 @ 2 s:

| phase | collective | motor mean | pitch | world vz (NED +=down): start → end |
|---|---|---|---|---|
| init_level | 0.270 | 0.246 | −8° (leveling) | 0.00 → −0.23 (slight climb) |
| thr_0.24 | 0.240 | 0.241 | −1° (level) | −0.23 → **+2.57** (sinks hard) |
| thr_0.26 | 0.260 | 0.259 | 0° (level) | +2.58 → **+2.85** (still sinking) |

→ **aborted on the −6 m bound by SINKING** before reaching 0.28+. **GUI:** "it sank."

**CLIMB sweep — `20260607_030629_vprobe_climb` (`vprobe_climb_extract.json`).** Init 0.27, steps 0.28→0.32 @ 2 s:

| phase | collective | motor mean | pitch | world vz (NED +=down): start → end |
|---|---|---|---|---|
| init_level | 0.270 | 0.245 | −8° (leveling) | 0.00 → −0.22 (slight climb) |
| thr_0.28 | 0.280 | 0.280 | −1° (level) | −0.23 → **−1.47** (climbs) |
| thr_0.30 | 0.300 | 0.299 | 0° (level) | −1.48 → **−3.51** (climbs faster) |

→ **aborted on the +6 m bound by CLIMBING** before reaching 0.32. **GUI:** "very smooth takeoff and climb,
very constant, barely any lurch" — i.e. the OPEN-LOOP plant climbs *smoothly* at fixed collective; the
limit cycle is purely the closed alt loop (the kp_alt relay), NOT the plant.

## What this gives (for the offline fit) — BOTH sides, hover bracketed
- **Hover is bracketed in (0.26, 0.28):** 0.26 sinks, 0.28 climbs (both flown LEVEL, so thrust ≈ vertical) →
  **hover ≈ 0.27**. Notably this is ~the twin's 0.2656 and Task-3's 0.26–0.27 — confirming the rung-1 mean
  thrust 0.32 was a clip-duty-cycle artifact (NOT hover), exactly as flagged. Init 0.27 also sat ~neutral.
- **Two-sided thrust→accel slope** at level: 0.24 (strong sink), 0.26 (mild sink), 0.28 (mild climb),
  0.30 (stronger climb) — drag-corrected d(vz)/dt vs collective gives the slope at the operating point,
  firming the 2-point thrust soft-spot the memo flagged.
- **Motor witness** tracks the commanded collective to ±0.002 on every level → we own thrust, again.
- Each open-loop level displaces several metres in 2 s, so the ±6 m budget caps each sweep at ~2 clean
  levels (same reason Task-3 sweeps aborted partway) — hence two runs (sink + climb) to cover the range.

**Data:** `vprobe_sink_extract.json` + `vprobe_climb_extract.json` (per-phase collective / world-vz / motor
series + LOCAL_POSITION_NED true vz). The Commander fits hover (vz=0 crossing ≈ 0.27) + the two-sided slope.
