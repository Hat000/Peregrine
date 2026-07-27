---
name: failure-census-2026-07-27
description: "SSOT for the N=572 death census, the gyro-de-rotated VG instrument, the two classification artifacts it overturned, the 10-vs-41 resolution, and the fix-targeting exposure table."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T02:32:48.281Z
---

# Failure census + the VG instrument — 2026-07-27

Deliverable committed at **`ed6de9f8`**: `docs/failure-profile-2026-07-27/PROFILE.md` (+ `census_tables.txt`,
`handlabels.json`) and `scripts/failure_profile/*.py`. Companion to [[velocity-channel-2026-07-27]].

## 🚩 THE INSTRUMENT — gyro-de-rotated gate lever (`scripts/failure_profile/vg.py`)

A gate is a **fixed world object**, so `ṙ = −ω×r − v_body`. De-rotating the logged `rel_flu` by a
gyro-propagated attitude gives `−R_wb·r = p_drone − p_gate` — **the drone's TRUE trajectory relative to
the gate, from vision bearings and gyro alone, never touching `obs[0:3]`.**

Range enters only the transverse channel as a scale, so the error channels split cleanly:
**transverse rms 0.07 m vs radial rms 0.43 m** (corpus median ratio 4.6×). D1 measured the same split
independently as LOS-perp 0.036 m vs LOS-parallel 0.089 m median / 0.360 p90 over 18 700 fix triples.
⇒ **Any velocity or geometry estimate off the gate must use the PERPENDICULAR/TRANSVERSE subspace only.**

**Calibration ground truth = 1151 CONFIRMED GATE PASSES** (a pass has a true miss < 0.75 m by
construction). Thresholds: vertical 0.75 m → 6.3% FP · lateral 1.5 m → 2.9% FP · **lateral 1.0 m →
14.8% FP (that band is refused a mode name)** · blackout fix-gap ≥ 0.60 s / blind ≥ 0.70 → 0% FP on 291
healthy passes. Hand-label agreement **19/24 strict, 21/24** counting two aim-probe cases. An earlier
classifier scored 54%; the fix was deleting two false-precision calls.

## 🛑🛑 TWO BANKED ROOT CAUSES OVERTURNED — SAME CLASS OF ERROR

Both were **body-frame gate geometry read without compensating for the drone's own attitude/rotation.**

1. **"obs lateral-velocity carries no information"** → ROTATION CONTAMINATION. Full write-up in
   [[velocity-channel-2026-07-27]].
2. **"vertical undershoot dominates"** → PITCH COUPLING. Read from raw `rel_flu[2]` at the flown
   **+23.8° median pitch**, **66.2% of CONFIRMED PASSES classify as a vertical strike**
   (corr with ρ·sin(pitch) = **−0.861**). Gravity-levelled it drops to **6.0%**, and undershoot (9) vs
   overshoot (16) is roughly **symmetric — there is no low bias**.
   🟢 **v2.0's premise SURVIVES**: it was measured from the coarse sector `obs[9:11]`, which *is*
   gravity-levelled, not from `rel_flu[2]`. Only the expected *payoff* shrinks (vertical family is
   4.4–7.2% of deaths, not the dominant mode).

## 🛑 10-vs-41 RESOLVED — BOTH OLD INSTRUMENTS RETIRED

Same 87 v19 fatal approaches, same detector: **vision-estimate series fires 17 · KF track 11 · the
drone's ACTUAL motion 9.**

* **Vision over-counts**: the estimate's radial noise (0.43 m p50 / 1.49 m p90) is the same size as the
  detector's own thresholds (`lmin` 0.45, `reopen` 0.30). Injecting each flight's own measured noise
  into its own true trajectory manufactures the signature on **36% of draws** for A-only flights vs a
  6% baseline. Doubly exposed: `lmin` is a noisy minimum (biased down) while `reopen` is end-minus-min
  (inflated by the same noise).
* **The KF is no corrective**: its displacement drifts a median **3.90 m over a 10.3 m approach (38% of
  path length)** and it reports **4.8× more lateral excursion than really happened**. It fires less only
  because that same drift inflates `lmin` past the "was-centred" gate ⇒ **its apparent agreement with
  truth at ~10 was a coincidence of two errors.**

## 🚩 TERMINAL BLINDNESS BOUNDS EVERY AXIS CLAIM

Median at-gate log ends **1.43 m / 0.222 s before the gate plane**; 90% end > 0.5 m short;
**69% of at-gate deaths (193/278) have NO determinable kill axis**, and 109 of those read *inside* the
opening at the last observable moment. Independently corroborated: the recorder agent measured a median
**0.223 s** time-to-contact remaining over 533 CRASH sessions. → the post-impact recorder `c0d3f6ef`
exists exactly to close this, and `video_index.jsonl` already runs 70–130 ms past `ego_obs.jsonl`.

## CENSUS — N = 572 (583 CRASH minus 11 aim-probe flights)

| mode | N | % | v15 | v16 | v18 | v19 | v20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **launch/release window** | **115** | **20.1** | 18 | 49 | 19 | 21 | 3 |
| vertical undershoot | 9 | 1.6 | 3 | 2 | 3 | 1 | 0 |
| vertical overshoot | 16 | 2.8 | 1 | 8 | 3 | 2 | 0 |
| lateral diverge (conservative) | 44 | 7.7 | 11 | 14 | 2 | 9 | 1 |
| corner (both axes) | 16 | 2.8 | 1 | 11 | 1 | 3 | 0 |
| **at-gate, AXIS UNDETERMINED** | **193** | **33.7** | 41 | 76 | 16 | 36 | 1 |
| gate-to-gate blackout | 64 | 11.2 | 15 | 36 | 2 | 4 | 0 |
| obstacle leg (gate-5) | 22 | 3.8 | 10 | 8 | 1 | 3 | 0 |
| other | 93 | 16.3 | 20 | 30 | 6 | 22 | 1 |
| **total** | **572** | | 120 | 234 | 53 | 101 | 6 |

Brackets: **lateral 44 … 127 (7.7–22.2%)** · **vertical 25 … 41 (4.4–7.2%)**.
🟢 Only clean lineage trend: **blackout collapses v15/v16 14.4% → v18/v19 3.9% (Fisher p = 3.5e-04)** —
the vision-emission fix, visible in the death census. Launch share ROSE (v15 15% → v18 36%) and v18/v19
die at *earlier* gates. v20 is 6 flights — reported, not concludable.

## ▶️ FIX TARGETING — exposure CEILINGS, overlapping, not recovery promises

| fix | modes | exposure |
|---|---|---|
| **release-dive retrain (v2.1)** | launch window | **20.1%** — highest confidence, axis-independent |
| **close-in coast + bbox-range preference** | at-gate undetermined + blackout | **≤ 33.7% + 11.2%** — the only fix that attacks the blindness itself |
| **truthful lateral velocity / `ego_faithful`** | lateral + corner + part of undetermined | **7.7 … 22.2%** — worst tail (p50 1.84 m, p90 2.88 m) |
| gate-seam (`pass_drop` teleport) | obstacle leg + gate-4 cluster | 3.8 … 13.9% — 20% of deaths within 1.0 s of an advance |
| vertical-structure training (v2.0) | vertical family | 4.4 … 7.2% — smaller than expected (see the artifact above) |
| loop-rate (`vq2_ego_lean`) | margin on all at-gate | 39% of deaths flew < 30 Hz |

🚩 **v2.0's release-dive regression attacks the single largest determinable death mode in the WRONG
direction** ⇒ v2.1 is the top training priority.

## 🚩 PROBABLE SECOND INVISIBLE OBSTACLE — needs ~6 pilot flights

**30 "other" deaths cluster at 14.1 m from gate 4, IQR 1.9 m** — *tighter* than the KNOWN gate-5
obstacle's 2.7 m — with statistically indistinguishable dynamics (calm, level, vision healthy). Both
legs are ~20.5 m and both clusters sit ~6.4 m past the previous gate.
**Test: `aim_off` on the gate-4 leg, ~6 flights — exactly the probe Fengyou already ran on gate 5.**

`aim_off` census: 15 sessions / 360 ticks — 12 on the gate-5 leg (to [0,10] m), 3 on gate 1.
**Exclude those ticks always; exclude the whole session when the aim leg is the fatal leg (11 sessions).**

## Residual instrument weaknesses (stated, not hidden)

* `y_h` is a **ballistic miss, not an impact point** — a crabbing drone legitimately disagrees with
  body-frame lateral (worst case: −1.5 m body lateral vs 0.30 m ballistic miss).
* Lateral and vertical are **not separable** for the 16 corner cases + the 67-flight 1.0–1.5 m band.
  Resolver: the recorder, plus a ground-truth odometry stream on ~10 flights.
* The at-gate / mid-leg boundary at 3.5 m accounts for 2 of the 5 hand-label disagreements.
