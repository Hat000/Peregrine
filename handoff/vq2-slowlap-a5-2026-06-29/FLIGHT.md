# VQ2 self-localized SLOW-lap — ATTEMPT 5 (bounded feedforward forward-tilt + spawn-gate egress) — 2026-06-29

**Branch flown:** `claude/loving-galileo-92f020` @ `767f667`
("fix(vq2): bounded feedforward forward-tilt + spawn-gate egress").
**Sim:** AI-GP 1.0.3379, **R2 - TRAINING = VQ2** — lit warehouse verified by screenshot before every
flight. Menu navigated blind by keyboard; screenshot used ONLY for the VQ1/VQ2 check.
**Controller:** map-free visual-servo gate-seeker, `vq2_case_c`, `red_glow`, settle 0.75 s, N=3,
egress 0.8 s, pursuit pitch-rate cap 1.5 rps.

## TL;DR scoreboard
| Question | Result |
|---|---|
| Egress cleared gate 0? | **Partially / NO clean clearance** — the egress phase runs (bounded creep), but the drone still contacts structure at close range; at 2.0 it doesn't even release in time and drifts into gate 0. |
| Pitch bounded (no nose-dive)? | **YES — confirmed in realized IMU.** `cmd_pitch` never exceeds the +1.50 cap (A4 wound to −3.999); realized pitch rate holds a steady ~1.5 rps through the whole approach. The A4 nose-dive is FIXED. |
| Gates reached | **0 / 6 controlled** — but `active_gate_index` ADVANCED to **1** in 2 of 4 runs (a program first), via drift/approach through the spawn-gate plane, not a clean flythrough. |
| Contact? | **YES** — env (1002) and, in 2 runs, gate (1001). |
| Failure mode | **Close-range STRUCTURE CONTACT during an otherwise clean, bounded approach** (3.0); **far-gate lock → drift into gate 0** (2.0). The estimator inversion seen in the CSV is *post-impact*, not causal. |

4 runs: `run1_speed2`, `run2_speed2` (2.0); `run3_speed3`, `run4_speed3` (3.0). Per-tick
`seeker_diag.csv`, onboard `frames/`, `vision_*.mp4`, fly logs, `frames/analysis.json` (detector +
realized IMU) in each.

---

## What the A4→A5 fixes achieved (confirmed in REALIZED telemetry)
- **Bounded forward tilt — no nose-dive.** Both 3.0 runs: realized pitch rate is a steady **~1.5 rps**
  (the cap) through the approach, drone level (accel-tilt ≤+10°), |a|≈10.5 — a controlled, bounded
  forward flight. A4's `cmd_pitch=−3.999` saturation is gone (the feedforward tilt has no velocity
  windup; pitch is now capped symmetrically).
- **Closest approach in the program.** `run4_speed3`: detected gate range closes **9.8 → 2.78 m**
  (A4 stalled ~8 m). `run3_speed3`: 10.5 → 8.7 m. The drone flies a real, stable, roll-free approach.
- **`active_gate_index` advanced to 1** (runs 1 & 4) — the first gi advance ever (every prior attempt
  was stuck at 0). See the honesty note below on what this does/doesn't mean.

## Why it still stops — two regimes

### 3.0 m/s (near-gate lock): clean bounded approach → STRUCTURE CONTACT at close range
`run4_speed3/seeker_diag.csv` + realized IMU (`frames/analysis.json`):
```
settle/anchor: trkR≈9.8m centered (az≈0); d=1→2→3 -> released
egress  t≈60.8 cmd_pitch=+1.50  (bounded creep)
pursuit t≈61.8 trkR=4.9m detR=2.78m cmd_pitch=+1.50  est tracks   <- CLOSEST APPROACH
"gate 0 PASSED -> targeting 1"
REALIZED IMU: pitch rate steady ~1.5 rps + level until t≈3.7s, then |a| 10->28 m/s²,
              IMPACT t≈4.1s (|a|=43.8) [run3: |a|=772 @t=2.95s], THEN attitude inverts.
```
The realized attitude is **bounded and controlled right up to impact** — the estimator inversion
(est_roll→±180°) in the CSV happens *after* the |a| spike, i.e. it is the post-contact tumble, not
the cause. The drone spawns INSIDE gate 0 and the tracked gate is the next one ~9 m downrange; as it
creeps/pursues forward it contacts structure (the gate-0 frame it is egressing, or the gate it is
approaching, or the floor). `trk_el` drifts NEGATIVE (−0.02→−0.06): the opening sits increasingly
*below* the altitude-held flight path, so the drone is not vertically aligned with the opening and
clips the frame rather than threading it.

### 2.0 m/s (far-gate lock): can't release → drift into gate 0
Both 2.0 runs lock a DISTANT, off-axis gate: `trk_range≈37–60 m`, `trk_az≈+0.33 rad (~19°)` — not
the near start gate. Far-gate detections are intermittent, so the N=3 consecutive-detection release
is hard to reach: `run2` never releases (anchor-hold the whole time); `run1` holds ~12 s before
releasing. During the hold the FROZEN spawn attitude (~−18° pitch) + hover thrust slowly translates
the drone (realized peak gyro ~0.01 rps — it barely rotates, it DRIFTS), into gate 0 — `run2` ends in
a **gate contact (id 1001)** with only 2 collisions. So at 2.0 the bottleneck is upstream:
**target selection picks a far off-axis gate over the near one**, starving the release.

## Honesty note on "gates reached = 1"
`RACE_STATUS.active_gate_index` advanced 0→1 in runs 1 and 4. The drone spawns inside gate 0, so this
reflects the spawn-gate plane being crossed (by drift in run1, by the forward approach+contact in
run4), NOT a clean, controlled flythrough of a downrange gate. Counting it as a real "gate passed"
would overstate the result — but it IS the first time the wire's gate index moved, which is real
forward progress.

## Recommended next-layer fixes (design — not done here)
1. **Vertical alignment to the opening.** Stop holding a fixed altitude during pursuit; track the gate
   `trk_el` toward zero (descend/climb so the opening is centered) so the drone threads the hole
   instead of clipping the frame. This is the proximate cause of the 3.0 close-range contact.
2. **Robust near-gate selection (2.0).** Bias first-acquisition toward the NEAREST plausible gate (the
   one the drone must fly first), not the most-centered — the far 37–60 m off-axis lock starves the
   release and lets the drone drift into gate 0. Consider relaxing N (3→2) for honest near detections.
3. **Stronger spawn-gate egress.** The 0.8 s creep isn't clearing gate 0 before contact; lengthen it
   / make it distance-based (clear a set distance along the start-gate normal) before pursuit, and
   bound thrust so the frozen-tilt hold doesn't translate into gate 0 while waiting to release.

## Artifacts (per run: run1_speed2, run2_speed2, run3_speed3, run4_speed3)
- `seeker_diag.csv` — per-tick regime / anchored / consec_det / det+trk range / trk_az,el / est-RPY /
  thr / cmd roll,pitch,yaw / tsv.
- `frames/` — onboard camera frames; `frames/analysis.json` — detector + first-6 s realized IMU.
- `vision_*.mp4` — camera feed with red_glow gate overlay + PnP range.
- `fly_run*.log` — full fly-loop console.
- `analyze_run.py`, `make_vision_video.py`, `diag/seeker_diag_instrumentation.patch` (reverted).
- Recordings: `data/runs/20260629_200914_*`, `_201133_*`, `_201327_*`, `_201525_*`.
