# VQ2 self-localized SLOW-lap — ATTEMPT 4 (pursuit gate-tracking + slew-ramp + freeze-hold) — 2026-06-29

**Branch flown:** `claude/loving-galileo-92f020` @ `d6502c9`
("fix(vq2): pursuit gate-tracking + slew-ramp + freeze-hold").
**Sim:** AI-GP 1.0.3379, **R2 - TRAINING = VQ2** — lit warehouse verified by screenshot before every
flight (`a4_verify_vq2_lit_warehouse.png`). Menu navigated blind by keyboard; screenshot used ONLY
for the VQ1/VQ2 check.
**Controller:** map-free visual-servo gate-seeker, `vq2_case_c`, `red_glow`, settle 0.75 s, N=3.

## TL;DR scoreboard
| Question | Result |
|---|---|
| Released (left launch-hold)? | **YES at BOTH 2.0 and 3.0** — the freeze-hold fixed the 2.0 release (A3 only released at 3.0). Streak `d=1→2→3 → anchored` during settle, both runs. |
| Stable pursuit (no roll-over, one locked gate)? | **YES** — the temporal track holds ONE gate: `trk_range` smooth **9.3→9.7→9.3→8.4 m** (A3's 10↔30 m flap is GONE), and roll stays small (cmd_roll ≤0.10, the roll cap holds). A3 roll-over is fixed. |
| Approached the gate? | **YES (partially)** — range closes ~1.3–1.8 m (9.3→7.5 @2.0, 9.0→7.7 @3.0). First approach progress in the program. |
| Gates reached | **0 / 6** |
| Contact? | **YES** — env (id 1002): 44 @2.0, 31 @3.0. |
| Failure mode | **Pursuit PITCH saturation + start-gate contact.** With roll/yaw now tamed, the pursuit's forward-velocity demand drives `cmd_pitch` to the ±4 rad/s limit (pitch is the one pursuit axis still uncapped); the drone — still inside the start gate at spawn — pitches forward into the start-gate structure. |

Runs: `run1_speed2` (2.0), `run2_speed3` (3.0), both instrumented. Per-tick `seeker_diag.csv`;
detector + realized-IMU in `*/frames/analysis.json`.

---

## What the A3→A4 fixes achieved (all three confirmed working)
1. **Freeze-hold → release at 2.0.** Both runs anchor during the settle window; 2.0 now releases
   (A3 stalled at 2/3 because the camera drifted off-gate — fixed).
2. **Temporal gate-track → stable lock.** `seeker_diag.csv` `trk_range_m` is smooth and monotonically
   closing (9.3→9.7→9.3→8.4 @2.0; 9.0→9.7→9.2→8.45 @3.0). The A3 10↔30 m range-flap is gone.
3. **Roll cap + slew-ramp → no roll-over.** cmd_roll stays ≤0.10 rps through pursuit (A3 saturated at
   +3.4 and rolled to +112°). Realized roll ≈0° the whole approach.

The drone now does what no prior attempt did: **release → lock one gate → begin a stable, roll-free
approach** (range closing). Detector sees the gate throughout (25–26/41 frames).

## Why it still stops — pursuit PITCH saturates (the next axis)
`run1_speed2/seeker_diag.csv` (2.0):
```
t=53.89 settle  anchored=1 d=3 trkR=9.35  estP=-0.31  cmd=[+0.00,+0.00,-0.00]   <- released
t=54.36 pursuit d=4 trkR=9.68 detR=10.0   estP=-0.31  cmd=[+0.02,-0.69,-0.05]   <- pursuit begins
t=54.84 pursuit d=5 trkR=9.27 detR= 8.9   estP=-0.31  cmd=[+0.04,-2.39,-0.09]   <- pitch ramping
t=55.33 pursuit d=6 trkR=8.38 detR= 7.5   estP=+0.02  cmd=[+0.10,-4.00,-0.04]   <- PITCH SATURATED
t=55.78 reacq   d=0 trkR=8.38             estP=+1.17  cmd=[ 0, 0, 0]            <- gate lost, tumbling
```
3.0 is identical: `cmd_pitch` −1.83 → −3.74 → **−3.999** over ~1 s, then `reacq`/crash.

- **Roll & yaw are capped in pursuit; pitch is NOT.** The A3 fix added `pursuit_roll_rate_cap_rps`
  (1.5) and `pursuit_yaw_slew_rps` (1.0) — the A3 failure axes — but there is no pitch-rate cap, so
  the forward-velocity lean drives pitch straight to the ±4 rad/s controller limit.
- **Velocity error never closes (case-C, map-free).** Velocity is observable only through position
  fixes, which never fire map-free (`tsv=inf` throughout), so the dead-reckoned velocity doesn't
  track the cruise demand → the controller keeps increasing pitch → saturation.
- **Spawn-inside-the-gate makes the first forward motion a contact.** The tracked gate is ~9 m ahead,
  but the drone is physically inside the START gate; as it pitches forward it hits the start-gate
  structure. Realized IMU: a gentle real pitch builds during the stable window (gyro_y 0.7→1.8 rps,
  accel-tilt +1°→+6°), then a **44/43 rps spike at impact** (collision spin, not commanded).

So the frontier moved from A3's *roll-over* to A4's *pitch saturation + immediate start-gate contact
on first forward motion*.

## Recommended next-layer fixes (design — not done here)
1. **Cap the pursuit pitch rate** (symmetric to the roll cap), and/or ramp the forward-velocity demand
   over the first ~0.5–1 s of pursuit so the lean builds gently — pitch is the only uncapped axis now.
2. **Clear the start gate before pursuing the downrange gate.** The drone spawns inside gate 0; a brief
   "depart the spawn gate" phase (small, capped forward creep aligned with the gate normal, or a short
   straight push before full pursuit) avoids driving the lean into the surrounding start-gate frame.
3. **Bound the forward demand without velocity feedback.** Since case-C velocity can't be corrected
   map-free, cap the commanded tilt directly (max lean) rather than letting a velocity-error term wind
   up to saturation.

## Artifacts
- `run1_speed2/`, `run2_speed3/`: `fly_run*.log`, `seeker_diag.csv` (per-tick regime/anchor/track
  range+bearing/est-attitude/cmd-rates), `analysis.log`, `frames/analysis.json` (detector + realized
  IMU), sample frames.
- `analyze_run.py` — offline analyzer. `diag/seeker_diag_instrumentation.patch` — the (reverted,
  uncommitted) fly-loop diagnostic that produced the CSVs.
- `a4_verify_vq2_lit_warehouse.png` — VQ2 verification.
- Recordings: `data/runs/20260629_193758_*` (2.0), `data/runs/20260629_193951_*` (3.0).
