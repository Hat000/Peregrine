# c5 — the cone-relaxation COUPLING: speed-vs-validity Pareto at gate-4

Agent **c5** (estimator/perception, opus-4.8). BRANCH: the inc8 speed lever (style cone
60deg -> 70-80deg) RAISES the gate-4 approach speed; quantify how that couples to the estimator
margin/sigma ratio and find the **MAX VALID gate-4 approach speed given the estimator**.

Offline sim only. Composes the REAL `racer.state_estimator.LinearKF` + the REAL `RewindKF`
(kf_rewind_buffer.py) + the REAL `racer.localization.gate_pose_to_world_position` fix-cov, and
RE-USES the speed-degradation models the parent agents already built/measured: a3 motion-blur ->
noise-multiplier g(v) + cadence; a4 latency v*L staleness; a1 closed-loop MC harness; b2
range-collapsing per-track bias. Residuals RESAMPLED from the MEASURED `off_ned` pool
(perception-char-2026-06-08), range-band conditioned.

Files: `c5_speed_coupling.py` (sweep), `c5_speed_coupling_results.json`. Seeded numpy RNG
(20260613) per-seed/per-cell, 400 seeds/cell; every number below reproduces on re-run. The v=37,
short-shutter, edge-L cell reproduces a1/b1 EXACTLY (filtered in-plane sigma 0.360 m; deployable
0.408 m; CPU-p90 naive in-plane 0.87 m vs a1's 0.83 m) — the harness is the a1 harness, generalised
to a speed sweep, so the anchor is validated.

---

## HEADLINE — the answer to "the number that gates the inc8 speed ladder"

**There are TWO answers, and the split is the whole point of this branch:**

1. **On the ABSOLUTE case-C pose path (current architecture): the MAX VALID gate-4 approach speed
   is ZERO useful headroom — the estimator is already INVALID at the slowest speed swept (8 m/s).**
   The deployable in-plane error is **FLOOR-DOMINATED and nearly SPEED-FLAT**: ~0.34 m at 8 m/s
   rising to ~0.41 m at 55 m/s (short shutter), i.e. **~2.5x over the 0.155 m contact margin and
   ~8x over the 0.05 m bar at EVERY speed**. The estimator does NOT gate the ladder via a speed
   cliff; it gates it via a **speed-independent floor that disqualifies the lowest rung**. So
   **no inc8 cone rung — 60deg, 70deg, 80deg, unconstrained — is estimator-valid on the absolute path.**
   Raising the cone changes the gap-to-bar by single-digit-mm; the architecture is the wall, not the speed.

2. **On the GATE-RELATIVE path (the FACTS-mandated fix; per-track bias removed, per-fix sigma driven
   to corner-reprojection accuracy): the estimator imposes a REAL, finite speed ceiling, and it is
   the number to gate the ladder with.** Once the floor+bias problem is solved, the speed-coupled
   VARIANCE alone gives:

   | gate-relative per-fix sigma | MAX-VALID speed vs **0.05 m bar** | MAX-VALID speed vs **0.155 m margin** |
   |---|---|---|
   | 0.20 m | 12 m/s (already over at 12) | **17 m/s** |
   | 0.10 m | 12 m/s (already over at 12) | **>55 m/s** (held to top of sweep) |
   | **0.05 m** | **26 m/s** | **>55 m/s** |
   | **0.03 m** | **50 m/s** | **>55 m/s** |

   **READ THIS AS THE GATE:** to fly the inc8 ladder at ~37 m/s and stay inside the **0.155 m
   contact margin**, the gate-relative estimator must reach **per-fix in-plane sigma <= ~0.10 m**.
   To meet the stricter **0.05 m 1-sigma bar** at 37 m/s it must reach **per-fix ~0.03 m**; at
   per-fix 0.05 m the 0.05 m bar holds only to **~26 m/s**. Every cone-relaxation rung that pushes
   the speed past these points must be re-verified against the achieved gate-relative per-fix sigma
   AT that speed.

---

## How the four coupling terms evolve with speed (the requested decomposition)

Sweep v in {8,12,16,20,25,30,37,41,45,50,55} m/s. (Numbers: short-shutter nominal path = edge
L=15.9 ms + RewindKF + globally-debiased fixes, accept 0.47.)

### (a) Gate-4 contact margin is FIXED 0.155 m, but TIME-TO-REACT shrinks ~1/v
The margin does not move (it is geometry). What shrinks is the planner's reaction budget — the
time between the last usable fix (geometric FoV-exit at r~1.33 m) and the plane crossing:
**166 ms @ 8 m/s -> 36 ms @ 37 m/s -> 24 ms @ 55 m/s.** This is a PLANNER/CONTROL term, not an
estimator-error term, so it does not enter the in-plane miss directly; I report it as the diagnostic
that the faster you go, the less the controller can do with a late correction. (At 55 m/s, 24 ms is
~2.4 command ticks at 100 Hz — thin.)

### (b) Fix CADENCE collapses as fps/v -> fewer effective fixes -> weaker variance averaging
30 Hz (true 28.6 fps) fixed, so fixes/metre = fps/v: **3.58 /m @ 8 m/s -> 0.77 /m @ 37 m/s ->
0.52 /m @ 55 m/s.** Effective accepted fixes over the gate-4 accept window drop **~40 -> ~8 ->
~5.6**. BUT variance averaging is only ~1/sqrt(N), so N: 40->5.6 only loosens the filtered sigma
by ~sqrt(40/5.6) ~ 2.7x IN PRINCIPLE — and in the REAL KF (velocity unobservable in case C, so
only the last ~2-4 fixes carry weight) the measured filtered in-plane sigma moves only
**0.275 -> 0.360 -> 0.366 m** across 8->37->55 m/s. **Cadence is the dominant SPEED term but it is
second-order**: 30->55 m/s adds only ~13 mm of filtered variance (short shutter).

### (c) WORLD-FIX SIGMA from motion blur (a3 model) grows with v — EXPOSURE-GATED
Short global shutter (<=0.5 ms): blur multiplier g stays ~1.00 at ALL swept speeds — blur is a
non-issue. Long exposure (8 ms): g(v) at mid-window grows **1.02 @ 8 m/s -> 1.35 @ 37 m/s -> 1.68
@ 55 m/s**, lifting the deployable in-plane from ~0.42 to ~0.59 m across the sweep. The
**blur-attributable variance growth from 30->55 m/s is ~46 mm (long) vs ~0 (short)** — so the
exposure/shutter spec is the pivot on whether speed inflates the per-fix noise materially. This is
the a3 escape-hatch E1/E2/E3: it cannot be pinned offline; a ShadowPC at-speed recording collapses it.

### (d) LATENCY STALENESS v*L grows ~v — but it is ALONG-TRACK (rewind corrects it)
v*L on the -N leg: **edge p90 0.59 m @ 37 -> 0.80 m @ 50 m/s; CPU p90 4.63 m @ 37 -> 6.26 m @ 50.**
This is ~98% ALONG-TRACK (a4) — it corrupts WHEN the plane is crossed, not the in-plane miss. The
in-plane LEAK (v*L*sin(theta), well-tracked 0.5deg heading) grows to only **5-7 mm at edge** and
**~55 mm at CPU p90, v=50** — sub-bar. RewindKF removes the first-order leak; the naive arm at CPU
latency blows up (in-plane RMS 0.87 @ 37 m/s, **1.10 @ 50 m/s**, vs rewind 0.38). **Conclusion:
latency does NOT set the speed ceiling on the in-plane miss at edge HW; on CPU HW the CPU path is
independently unacceptable on the along-track ground. RewindKF is mandatory on CPU, ~free on edge —
and its cost is speed-INVARIANT (horizon 0.5 s >> 125 ms at all speeds; mean dropped fixes 0.00).**

---

## The variance-vs-bias split at speed (obeying FACTS)

- **VARIANCE** (filtered, debiased scatter, the rewind arm): grows slowly with speed via cadence (b)
  and — only at long exposure — blur (c). 0.28 -> 0.37 -> 0.37 m across 8->37->55 (short). This is
  the term the gate-relative fix can crush (it converts to close-range corner reprojection); it is
  what the floor-removed sweep isolates.
- **BIAS** (per-track in-plane registration residual, b2 range-collapsing, evaluated at the
  last-usable-fix band): **0.191 m, SPEED-INVARIANT** (it is a geometry/gate-size systematic, not a
  motion effect — a3 + b2 both confirm). It survives global de-bias and does NOT average out within
  one gate-4 approach. On the absolute path this 0.191 m alone is **3.8x the 0.05 m bar and 1.23x
  the 0.155 m margin** — disqualifying before any variance is added, at any speed.
- **Total deployable in-plane = sqrt(variance^2 + bias^2)** — the GO/NO-GO number. Floor-dominated,
  speed-flat, ~0.34-0.59 m. NEES of the deployed arm 4.1-7.1 (vs chi2(3)=3): marginally-to-clearly
  overconfident, consistent with b1's finding that the filter P under-reports the un-modeled per-gate
  bias. The KF is honestly telling us it is ~2.5-8x over — not a paper number.

---

## What this means for the inc8 cone ladder (the tie-back)

1. **The estimator does NOT permit raising the cone on the absolute-pose path at ANY speed** — it
   is invalid at 8 m/s, so the speed ladder is moot until the architecture changes. The inc8 speed
   gain (cone 60->80deg, ~1.5 s/lap, the binding kinematic lever) is **estimator-blocked on case C
   absolute** regardless of how the cone is relaxed.
2. **The gate-relative observation is not optional — it is the gate-keeper of the entire speed
   ladder.** It is the ONLY path with a finite max-valid speed. The speed ladder must be co-designed
   with the achieved gate-relative per-fix accuracy:
   - per-fix **<= 0.10 m -> 0.155 m margin held past 55 m/s** (full inc8 ladder estimator-valid on
     the margin criterion).
   - per-fix **<= 0.03 m -> 0.05 m bar held to ~50 m/s** (full ladder valid on the stricter bar).
   - per-fix **0.05 m -> 0.05 m bar holds only to ~26 m/s** (cone relaxation past ~26 m/s would
     bust the bar — re-verify at each rung).
3. **EVERY cone-relaxation rung must be re-verified against the estimator at THAT speed**, because
   (i) cadence keeps loosening the variance, (ii) long-exposure blur inflates per-fix sigma with v,
   and (iii) the reaction-time budget shrinks ~1/v. The verification quantity is the **achieved
   gate-relative per-fix in-plane sigma at the rung's speed**, compared to the per-fix thresholds in
   the floor-removed table above. This is a per-rung GATE, not a one-time check.
4. **Exposure/shutter is the one unknown that moves the ceiling**: at short shutter, speed barely
   touches the variance; at 8 ms exposure it adds ~46 mm of variance growth 30->55 m/s and would pull
   the per-fix-0.05 bar-ceiling in from 26 m/s. Resolve via a ShadowPC at-speed recording (a3 E1-E5).

---

## Assumptions — MEASURED vs MODELED vs ASSUMED (flagged)

- **Trajectory (ASSUMED):** const-v along g3->g4 (24.4 m, ~level), speed swept 8-55 m/s. a1 showed
  the accel profile is immaterial (<0.02 m). Drag-hold pitch grows with v (atan(0.21*v/g)): 9.6deg
  @8 -> 38.4deg @37 -> 49.6deg @55 — feeds the predict-step attitude-skew Q. Not binding (process
  growth ~mm).
- **Cadence + blur (MODELED, from a3):** g(v,range) via the a3 bearing-sweep + body-rate chain;
  body rate modeled as 50 deg/s * (v/37) (a3 plausible band 50-200, set by the un-trained inc8
  policy — UN-MEASURED). first-accept range pulled in where sigma_px>2.5 px. Short/long exposure
  brackets the a3 pivot.
- **Per-track bias (MEASURED, from b2):** 0.191 m at the <=9 m last-usable band, range-collapsing,
  speed-INVARIANT. I deliberately use the b2-CORRECTED near-range value (0.191 m), NOT a2's 0.517 m
  window-mean (b2 WEAKENED that) — this is the charitable, defensible bias floor.
- **Latency (a4 / latency_results.json):** edge 5.8/15.9 ms, CPU 125 ms; in-plane leak via
  v*L*sin(0.5deg) well-tracked. RewindKF horizon 0.5 s.
- **Fix-cov (REAL model):** `gate_pose_to_world_position` with the PnP block inflated by g(v),
  divided by sqrt(PNP_FIX_COV_INFLATION) so g=1 is BIT-CONSISTENT with a1's covariance=None baseline.
  Floors (0.40/0.282 m) and the 1.4deg lever do NOT scale with blur (a3 / range_anisotropic_R).
- **CASE CONDITIONALITY:** entire analysis is the case-C worst case. If VQ2 streams
  LOCAL_POSITION_NED/ODOMETRY (case A/B) pose is pristine and this whole coupling is MOOT. Verdict
  CONDITIONAL on organizer Q1.

### Honesty caveats — what is NOT measured
- The per-fix residual pool is **~5.35 m/s data (range <= 23.3 m)** — NO measured per-fix data at
  37+ m/s. The blur multipliers are MODELED extrapolation 4-7x beyond data. So the short-shutter
  curve is a **best-case lower bound**; the long-exposure curve is the honest worst-case band. A
  single ShadowPC at-speed gate-4 recording (a3 E1-E5) collapses the whole exposure uncertainty.
- The gate-relative per-fix accuracies (0.03-0.20 m) are CANDIDATES, not measured — the actual
  achieved gate-relative sigma is the next thing to measure (a probe a different agent should build).
  This branch tells you WHICH accuracy buys WHICH speed; it does not assert gate-relative reaches it.

---

## MEMORY-DELTA:
- c5 SPEED-COUPLING DONE: the inc8 cone/speed lever couples to the estimator via (a) reaction time
  ~1/v (166->24 ms over 8->55 m/s), (b) cadence fps/v (~40->~5.6 fixes), (c) blur g(v) (short ~1.0,
  long 1.0->1.68), (d) v*L staleness (along-track, rewind-corrected). Reproduces a1/b1/a4 anchors exactly.
- **ABSOLUTE case-C is FLOOR-DOMINATED + SPEED-FLAT:** deployable in-plane ~0.34 m @8 m/s ->
  ~0.41 m @55 m/s (short), already 2.5x over 0.155 m margin / 8x over 0.05 m bar at the LOWEST speed.
  **MAX-VALID absolute speed = NO useful headroom — the estimator gates the ladder by a speed-
  INDEPENDENT floor, not a speed cliff. No inc8 cone rung is estimator-valid on the absolute path.**
- **GATE-RELATIVE (bias=0, per-fix sigma overridden) gives the REAL ceiling:** per-fix <=0.10 m holds
  the 0.155 m MARGIN past 55 m/s; per-fix 0.05 m holds the 0.05 m BAR only to ~26 m/s; per-fix 0.03 m
  holds the BAR to ~50 m/s. **GATE: to fly ~37 m/s inside the contact margin, gate-relative per-fix
  sigma must reach <=~0.10 m; for the 0.05 m bar, <=~0.03 m.** Every cone rung re-verified vs achieved
  gate-relative per-fix sigma at THAT speed.
- Latency on the in-plane miss is NOT the ceiling at edge (leak 5-7 mm); CPU path independently dead
  (naive in-plane 0.87->1.10 m at 37->50 m/s; rewind holds 0.38). RewindKF cost speed-INVARIANT.
- Exposure is the pivot: long 8 ms adds ~46 mm variance growth 30->55 m/s; short shutter adds ~0.
  Resolve via ShadowPC at-speed gate-4 recording. CONDITIONAL on organizer Q1 (VQ2 streams pose?).
