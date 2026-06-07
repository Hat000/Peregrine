# Run B — open-loop vertical probe (hover + thrust-slope at the operating point). LIVE, ShadowPC 2026-06-06

**Purpose:** the offline alt-loop re-tune needs the REAL hover thrust + thrust→vertical-accel slope near
where we fly (the rung-1 bang-bang mean 0.32 is a clip-duty-cycle artifact, NOT hover). Instrument =
`scripts/rate_sysid.py --mode hover` (Task-3): a damped level-hold that steps the collective open-loop
while the ACTUATOR_OUTPUT_STATUS motor witness records what the rotors actually do.

**OBSERVATION ONLY below — the hover/slope FIT is the offline Commander's job (deferred per protocol).**

## What was flown (2 attempts; the 2nd is the data)
1. `20260606_214138_vprobe` — aborted in `init_level` (1.5 s): the level-hold holds ATTITUDE level but has
   **no horizontal position hold**, so at the −18° resting tilt the drone dives forward ~g·tan18° ≈ 3 m/s²
   and **drifted 4 m → geofence abort** (my rung-1 4 m geofence was wrong for this instrument; Task 3 used
   18 m). No vertical data. Reran with `--max-offset-m 18 --kp-hold 2.0`.
2. **`20260606_214504_vprobe` — the data** (`vprobe_sink_extract.json`). Schedule init 0.27 + steps
   0.24→0.32 @ 2 s, bounds 6 m alt / 18 m geofence / 60° tilt. Leveled near the pad, then:

| phase | collective | motor mean | pitch | world vz (NED +=down): start → end |
|---|---|---|---|---|
| init_level | 0.270 | 0.246 | −8° (leveling) | 0.00 → −0.23 (slight climb) |
| thr_0.24 | 0.240 | 0.241 | −1° (level) | −0.23 → **+2.57** (sinks hard) |
| thr_0.26 | 0.260 | 0.259 | 0° (level) | +2.58 → **+2.85** (still sinking, ~flat) |

Then **aborted on the +6 m bound by SINKING** (z reached 6 m below the start) before reaching the climb
levels 0.28/0.30/0.32. **GUI (teammate):** "it sank."

## What this gives / does NOT give (for the offline fit)
- **Captured: the sink side** (0.24 and 0.26, both flown LEVEL so thrust ≈ vertical) + the motor witness
  (motors track the commanded collective to ±0.002 → we own thrust, again). At 0.26 the drone is still
  descending → **hover is above 0.26** (already refutes the 0.26 hover; consistent with the rung-1 hint that
  real hover sits above the twin's 0.2656).
- **NOT captured: the climb side** (0.28/0.30/0.32) — the ascending sweep used the whole 6 m sink budget on
  the two sub-hover levels first (the start is NOT ground-constrained — it sinks freely below the pad, like
  Task-3's descents). The ±6 m budget can't fit all five open-loop levels from one start (each displaces
  several metres in 2 s). A complementary **descending** run (0.32→0.28) would capture the climb side and
  bracket hover in [0.26, 0.30] — deferred (teammate said push now; the Commander can request it next
  session if the sink side + the climb data already in the rung-1/hover recordings isn't enough).

**Data:** `vprobe_sink_extract.json` (per-phase collective / world-vz / motor series + the LOCAL_POSITION_NED
true vz). The Commander fits hover (vz=0 crossing) + slope (d(vz)/dt vs collective, drag-corrected) from this.
