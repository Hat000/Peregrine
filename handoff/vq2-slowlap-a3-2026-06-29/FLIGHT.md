# VQ2 self-localized SLOW-lap — ATTEMPT 3 (cold-start choreography) — 2026-06-29

**Branch flown:** `claude/loving-galileo-92f020` @ `b08e844`
("fix(vq2): cold-start choreography — own-detection anchor release + gravity-aligned AHRS").
**Sim:** AI-GP 1.0.3379, **R2 - TRAINING = VQ2** — lit warehouse verified by screenshot before
every flight (`a3_verify_vq2_lit_warehouse.png`). Menu navigated blind by keyboard; screenshot used
ONLY for the VQ1/VQ2 check.
**Controller:** map-free visual-servo gate-seeker, `vq2_case_c`, `red_glow`, settle 0.75 s,
anchor-release N=3.

## TL;DR scoreboard
| Question | Result |
|---|---|
| Launch stable (no pitch tumble through settle)? | **YES** — settle holds level + still for ~0.75 s (realized gyro≈0, accel level 1g). The A2 violent tumble is GONE: realized peak gyro **0.66 rps** pre-release (A2 was 9.7), contacts **5–10** at 2.0 (A2 was 64–71). |
| Hold released (own-detection anchor)? | **YES at 3.0 m/s** (3 consecutive detections → anchored → pursuit). **NO at 2.0** (streak broke at 2/3). Release works but is **fragile**. |
| Reached pursuit? | **YES** (3.0 run) — *new milestone*; A2 never left the hold. |
| Gates reached | **0 / 6** |
| Contact? | **YES** — env (id 1002): 5–10 at 2.0 (gentle drift), 61 at 3.0 (post-release roll-over). |
| Failure mode | **Two layers** (below): (2.0) launch-hold slowly pitches and loses the gate one detection short of release; (3.0) release succeeds but **pursuit locks onto inconsistent detections** → saturated roll → roll-over. |

Runs: `run1_speed2` (2.0, clean), `run2_speed2_diag` (2.0, instrumented CSV+tlog),
`run3_speed3_diag` (3.0, instrumented). Per-tick seeker state in `*/seeker_diag.csv`; realized
IMU + detector replay in `*/frames/analysis.json`.

---

## What the fix achieved (vs A2)
- **Settle is genuinely stable.** tlog (both runs): `t=0–0.7 s` gyro≈0, accel-tilt level, |a|=9.8.
  The cold AHRS no longer diverges into a saturated pitch during the hold.
- **Rates are bounded.** Pre-release realized gyro peaks ~0.6 rps (the clamp), not A2's 9.7 rps.
- **The own-detection anchor release FIRES** (3.0): `seeker_diag.csv` shows `consec_det` 1→2→3 then
  `anchored=1`, and the regime flips `settle/anchor → pursuit` with real steering rates commanded.
  This is the first time any attempt has left the launch-hold on the live map-free wire.

## Why it still stops — two layers

### Layer 1 (2.0 m/s): the launch-hold slowly pitches; release misses by one detection
`run2_speed2_diag/seeker_diag.csv`:
```
t=55.57 settle  dets=0  estP=+0.00  thr=0.266   <- level, still (settle OK)
t=56.65 anchor  dets=1  estP=-0.62  thr=0.325   <- detection streak building (range 10.2 m)
t=57.16 anchor  dets=2  estP=-0.92  thr=0.372   <- 2/3 ... range closing to 8.6 m
t=57.68 anchor  dets=0  estP=-1.23               <- GATE LOST -> streak resets, never reaches 3
t=58.16 anchor  dets=0  estP=-1.49 estR=+1.23    <- estimate now tumbling; gate gone (frames darken)
```
The hold commands a *persistent clamped* +0.6 rps pitch (target≠estimate mismatch on the cold AHRS).
Realized gyro confirms a steady −0.6 rps pitch for ~2 s — not a violent tumble, but enough that the
camera tilts off the gate, the detector goes dark (replay: meanBGR 27→14→9→6, detections vanish by
frame ~23), and the **3-consecutive-detection release never completes** (peaks at 2/3). The drone
then drifts into the start-gate structure (5 gentle contacts).

### Layer 2 (3.0 m/s): release succeeds, then pursuit locks onto inconsistent gates → roll-over
`run3_speed3_diag/seeker_diag.csv`:
```
t=58.62 settle  anchored=1 dets=3 range=10.2   <- RELEASED (3 consecutive)
t=59.14 pursuit dets=4 range=25.9  cmd=[+2.19,-0.47,-1.50]   <- pursuit steering begins
t=59.63 pursuit dets=5 range= 9.6  cmd=[+0.30,+2.51,-0.01]
t=60.14 pursuit dets=6 range=29.7  cmd=[+3.43,-1.84,-0.93]   estR=+1.50 (+86°)  <- rolling over
t=60.65 reacq   dets=0             estR=+1.96 (+112°)         <- gate lost, inverted
```
The detected `range` jumps **10 → 26 → 9.6 → 30 m** tick-to-tick — the map-free servo is locking onto
**different red gates / unstable PnP depth** among the several gates visible down the lit course, so
the steering bearing swings and the **roll command saturates (+3.43 rps)**. Realized gyro (tlog)
confirms roll −2.27 rps then pitch −2.6 rps after release, ending in a tumble (peak 49/61 rps at
impact, t≈2.9 s, 61 contacts). So the *new* blocker is **pursuit-phase target stability**, not launch.

## Recommended next-layer fixes (design — not done here)
1. **Stabilize the pursued target.** The servo must lock the SAME gate frame-to-frame: gate it on
   range continuity / a small temporal track, ignore detections whose range jumps >X m, and prefer
   the centered (smallest-bearing) gate, not just the closest. The 10↔30 m flapping is the proximate
   cause of the roll-over.
2. **Slew-limit the pursuit yaw/roll harder right after release.** Ramp pursuit authority over the
   first ~0.5 s post-release (a "pursuit launch ramp"), so a noisy first bearing can't saturate roll.
3. **Cure the residual hold pitch (helps 2.0 release).** The persistent clamped +0.6 pitch means the
   controller's attitude target ≠ the gravity-aligned estimate at spawn — reconcile the target to the
   measured spawn tilt, or hold on accel-levelled attitude during the anchor so the camera doesn't
   drift off before 3 detections land. Consider N=2 to make release easier given honest detections.

## Artifacts
- `run1_speed2/` `run2_speed2_diag/` `run3_speed3_diag/`: `fly_run*.log`, `seeker_diag.csv`
  (per-tick regime/anchor/est-attitude), `analysis.log`, `frames/analysis.json` (detector + realized
  IMU), sample frames.
- `analyze_run.py` — offline analyzer. `diag/seeker_diag_instrumentation.patch` — the (uncommitted)
  fly-loop diagnostic that produced the CSVs (reverted from the tree; saved for reproducibility).
- `a3_verify_vq2_lit_warehouse.png` — VQ2 verification.
- Recordings: `data/runs/20260629_190121_*` (2.0 clean), `…_190412_*` (2.0 diag), `…_190732_*` (3.0 diag).
