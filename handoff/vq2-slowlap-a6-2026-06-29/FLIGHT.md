# VQ2 self-localized SLOW-lap — ATTEMPT 6 (dead-reckon through the pass + next-gate handoff) — 2026-06-29

**Branch flown:** `claude/loving-galileo-92f020` @ `3187bfb`. **Sim:** AI-GP 1.0.3379, **R2 - TRAINING =
VQ2** — lit warehouse verified by screenshot before every flight. Menu navigated blind by keyboard;
screenshot only for the VQ1/VQ2 check. **Controller:** map-free visual-servo gate-seeker, `vq2_case_c`,
`red_glow`. 4 instrumented runs: run1/run4 @2.0, run2/run3 @3.0.

## TL;DR scoreboard
| Question | Result |
|---|---|
| Gate 1 passed? | **PARTIAL** — run2 (3.0) reached the pass; the new `passing` regime fired and `RACE_STATUS.active_gate_index` advanced to **1**. The other 3 lost the gate at 8–9 m (or far-gate lock) before the pass. Not a *clean* controlled pass (free-fall — below). |
| Post-pass pitch-up gone? | **YES — fixed.** Realized pitch stayed bounded through the approach (≤~+8°, gyro_y capped ~1.5 rps); the A5 nose-dive into the backside is gone. The dead-reckon prevented the pitch-up. |
| Acquired gate 2? | **NO** — the pass run free-fell/contacted at the gate; never reached the `acquire` regime. |
| Gates reached | **0 controlled** (gi advanced to 1 in run2). |
| Contact? | **YES** — env (1002) only, 2–8 contacts; no gate contacts (1001). |
| Failure mode | **Close-range THRUST COLLAPSE → free-fall → AHRS attitude inversion** (navigator/altitude side, as STEP 3 anticipated). |

## ⚠ Critical context: the control loop ran at ~2.3–2.6 Hz this whole flight
The per-tick CSV now logs `nav_ms` and `tick_dt_ms`: **`nav.update` averaged 360–422 ms**, so the
control loop ran at **~2.3–2.6 Hz, not 30 Hz**. At 3.0 m/s that is ~1.3 m of travel between commands —
**only ~1 command spans the entire pass**, so the pass is executed essentially open-loop and cannot be
cleanly evaluated. This is the navigator bottleneck profiled in
`handoff/vq2-video-timing-2026-06-29/ROOT_CAUSE.md` (`manhattan_lines.fit_vanishing_point`, a
2000-iter np.cross RANSAC; validated 21× speedup → ~16 ms/tick). It also causes the ~55% camera-frame
drop visible in the `.mp4`s (headers show `dropped:N`). **Recommendation: land that perf fix before
further pass tuning** — at ~2 Hz the seeker logic can't be fairly judged, and it likely contributes to
the close-range loss itself.

## What the A6 fix achieved
The pass dead-reckon **works mechanically** and **kills the pitch-up**. run2 (`seeker_diag.csv`):
```
settle/anchor trkR≈10.4  ->  egress (pitch capped 1.5)  ->  pursuit trkR=4.97 detR=3.06
"gate 0 PASSED -> targeting 1"   regime -> passing,  gi=1     <- pass machine fired
```
Realized IMU (run2, `frames/analysis.json`): level, gyro_y bounded ~1.5 rps, |a|≈10.5 through the
approach — a controlled glide, NOT the A5 nose-dive (peak gyro only 5 rps).

## Why it still stops — close-range thrust collapse → free-fall → estimate inversion
At the close-range pursuit tick, in BOTH pass-reaching runs the **commanded thrust collapses to 0.05**
(the controller's `alt_thrust_lo` floor) while `est_pitch` simultaneously diverges:
```
run2: pursuit detR=3.06  thr=0.0500  est_pitch=-1.06     run3: pursuit detR=7.46 thr=0.0500 est_pitch=-1.11
      passing            thr=0.52     est_roll=-3.13(-179deg)
```
Realized IMU at the pass (run2): **|a| drops to ~1–5 m/s² (near free-fall)** and only THEN does the
attitude flip (accel-tilt roll → ±180°). So the chain is:
1. controlled bounded approach (pitch-up fixed) ✓
2. at close range the **altitude controller cuts thrust to the 0.05 floor** (the gate's `trk_el` drifts
   negative — opening sits below the altitude-held path — and/or the z estimate corrupts) →
3. the drone **free-falls** (|a|≈1 m/s²) →
4. the accel-based AHRS has no gravity vector in free-fall → **attitude estimate inverts** (roll/yaw →
   ±180°) → contact. run1/run3 then hit hard (peak gyro 45–81 rps); run2's pass was a soft free-fall
   (peak 5 rps).

This is the navigator/altitude-control "next layer" STEP 3 predicted — **not a seeker pitch-up**.

## Other observations
- **Far-gate lock recurs** (run4 @2.0): locked a 40 m, ~19°-off-axis gate (trk_az≈0.33) despite
  `max_acquire_range_m=22` — the range gate didn't reject it; it then drifted and softly contacted
  (peak gyro 1.7 rps). Worth checking the first-acquisition range/centering gate.
- No gate contacts (1001) in any run — all env (1002).

## Recommended next steps
1. **Land the navigator perf fix first** (ROOT_CAUSE.md) so control runs ~30 Hz; re-evaluate the pass
   at real rate. Most of the close-range loss may be the ~2 Hz sparsity.
2. **Fix the close-range thrust collapse:** don't let the altitude controller cut thrust to the floor
   when the gate `trk_el` goes negative at short range / when the z estimate is degenerate — hold hover
   thrust through the dead-reckon pass (the dead-reckon should own thrust too, not just heading/pitch).
3. **AHRS free-fall guard:** freeze the attitude estimate (don't accel-level) when |a| << g, so a brief
   low-thrust window can't invert the estimate.
4. Tighten first-acquisition to reject the far off-axis gate.

## Artifacts (per run: run1_speed2, run2_speed3, run3_speed3, run4_speed2)
- `seeker_diag.csv` — per-tick regime (incl. `passing`/`acquire`), `pass_armed`/`passing`, trk range+az/el,
  est-RPY, thr, cmd rates, **`tick_dt_ms` + `nav_ms`** (loop-rate evidence).
- `frames/` onboard frames + `frames/analysis.json` (detector + realized IMU). `vision_*.mp4` faithful
  render (gaps shown as freezes; header shows dropped count). `fly_run*.log`.
- `diag/seeker_diag_instrumentation.patch` — the (reverted) buffered diagnostic; flight stack untouched.
- Recordings: `data/runs/20260629_223454_*`, `_223631_*`, `_223806_*`, `_223943_*`.
