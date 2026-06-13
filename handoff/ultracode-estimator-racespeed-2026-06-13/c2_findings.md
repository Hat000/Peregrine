# c2 — Does MORE FIXES (higher cadence / fusing N over the approach) close the gate-4 gap?

Agent **c2** (estimator/perception, opus-4.8). Offline sim only. BRANCH: multi-fix fusion / higher
detector cadence. Composes the REAL `racer.state_estimator.LinearKF` + the REAL `RewindKF`
(`kf_rewind_buffer.py`) + the REAL `racer.localization.gate_pose_to_world_position` fix-cov model.
Builds directly on a1's physics (same gate-4 geometry, drag-hold attitude, 90 Hz IMU, MEASURED
`off_ned` residual pools, range-band conditioned) but the **sweep axis is DETECTOR CADENCE
(15→240 Hz)** instead of latency. Latency held at edge p50 (6 ms) to isolate the cadence effect.

Files: `c2_cadence.py` (the sweep, 28 cells × 600 seeds), `c2_cadence_results.json`;
`c2_budget.py` (analytic ceiling + compute feasibility), `c2_budget_results.json`.
Reproducible: numpy RNG seeded per-cell/per-seed (base 20260613); re-running reproduces every number.

---

## HEADLINE

**NO. More fixes does not close the gap — on EITHER term.**

- **VARIANCE (debiased arm):** raising cadence crushes the filtered in-plane 1-σ as expected, but
  **with diminishing returns and a hard floor above the bar.** At the realistic 30 Hz the debiased
  in-plane 1-σ is **0.34 m**; pushing to a physically-impossible **240 Hz** (112.8 Hz effective at
  47% accept, ~72 landed fixes) only reaches **0.157 m**; at idealized 240 Hz / 100% accept
  (~154 fixes) it reaches **0.119 m**. The cadence→∞ extrapolated in-plane variance floor is
  **0.106 m (≈ 0.075 m/axis) — still 2× over the 0.05 m bar.** Cadence **cannot** reach 0.05 m.
- **BIAS (raw, real arm):** cadence does **NOTHING**. The raw (measured-bias-included) in-plane RMS
  floors at **~0.30 m at every cadence** (15 Hz 0.53 m → 240 Hz 0.30 m, asymptoting to the
  ~0.29 m bias magnitude) and the filter NEES **explodes from 6 → 62** as cadence rises — the KF
  tracks the constant bias as truth and becomes catastrophically overconfident. This is the textbook
  result the prompt names: **averaging N zero-mean fixes does nothing to a consistent bias.**
- **Compute:** >30 Hz is feasible **only on edge HW** (edge p90 throughput ~**61 Hz** cap); the CPU
  detector is capped at **~8 Hz** and cannot even sustain 30 Hz. But this is moot — **cadence is not
  the binding constraint.**

**Verdict: per A1/A2 the binding term is BIAS (+ a variance floor above the bar). Cadence does NOT
help the validity question.** The gate-4 0.05 m in-plane 1-σ bar is unreachable by more fixes in the
absolute case-C path. This confirms a1/a2/b1: the fix is a **gate-relative observation**, not a faster
detector.

---

## 1. The cadence sweep (the core measurement) — rewind arm, edge L=6 ms, 600 seeds/cell

In-plane = sqrt(E²+D²) at the gate-4 plane crossing. **`std`** = pure scatter (the VARIANCE term;
= the debiased in-plane error). **`rms`** = total incl. bias (= the raw, deployable error).

### Realistic acceptance (47% of the 30 Hz-equivalent stream)
| cadence | eff Hz | landed fixes | per-fix σ (E) | DEBIASED in-plane σ | RAW in-plane RMS | RAW in-plane bias | RAW NEES |
|---|---|---|---|---|---|---|---|
| 15 Hz  | 7.0   | 4.3  | 0.61 | **0.448** | 0.530 | 0.291 | 6.0 |
| **30 Hz** (default) | 14.1 | 9.1 | 0.60 | **0.341** | **0.453** | 0.305 | 7.7 |
| 60 Hz  | 28.2  | 18.3 | 0.60 | **0.261** | 0.412 | 0.322 | 13.7 |
| 90 Hz  | 42.3  | 27.2 | 0.60 | **0.227** | 0.386 | 0.310 | 12.7 |
| 120 Hz | 56.4  | 36.2 | 0.60 | **0.209** | 0.367 | 0.306 | 21.2 |
| 180 Hz | 84.6  | 54.9 | 0.60 | **0.180** | 0.347 | 0.294 | 23.6 |
| 240 Hz | 112.8 | 72.1 | 0.60 | **0.157** | 0.336 | 0.289 | 35.5 |

### Idealized acceptance (100% — best-case variance crush)
| cadence | eff Hz | landed fixes | DEBIASED in-plane σ | RAW in-plane RMS | RAW NEES |
|---|---|---|---|---|---|
| 30 Hz  | 30.0  | 19.0  | **0.250** | 0.405 | 10.8 |
| 60 Hz  | 60.0  | 39.0  | **0.200** | 0.356 | 20.2 |
| 120 Hz | 120.0 | 77.0  | **0.150** | 0.323 | 34.7 |
| 240 Hz | 240.0 | 154.0 | **0.119** | 0.298 | 62.5 |

**Reads:** (a) the debiased σ falls monotonically with cadence but **never reaches 0.05 m** — even
240 Hz/100% (154 fixes, physically impossible at 37 m/s) is **2.4× over**; (b) the raw RMS is
**bias-pinned at ~0.30 m at every cadence** and improving cadence barely moves it (0.45→0.30 over a
16× rate increase) because it is the speed-independent measurement bias, not noise; (c) **NEES grows
without bound in the raw arm** (10.8→62.5) — the literal signature of a filter chasing a bias with a
shrinking covariance. The debiased-arm NEES stays bounded (4.4→11) because there's no unmodeled bias.

---

## 2. Effective independent fixes & why averaging saturates (the crux)

**How many effective independent fixes land?** From the debiased arm, `N_eff = (σ_perfix/σ_filtered)²`:

| cadence | landed fixes (N_real) | N_eff (KF realized) | ideal σ = σ_perfix/√N_real | KF / ideal ratio |
|---|---|---|---|---|
| 30 Hz / 47% | 8.9  | **6.2** | 0.20 | 1.20 |
| 30 Hz / 100% | 19.0 | **11.2** | 0.14 | 1.30 |
| 90 Hz / 100% | 58.0 | **22.8** | 0.079 | 1.60 |
| 240 Hz / 100% | 154.0 | **51.9** | 0.048 | **1.72** |

Two decisive facts:

1. **At the realistic 30 Hz/47% the KF averages only ~6 effective independent fixes** (matching a3's
   independent ~7–8 estimate, and a1's "~2–4" lower count which used the steeper degraded stream).
   That crushes a ~0.60 m per-fix σ by 1/√6 ≈ 0.41× → ~0.34 m in-plane. To reach 0.05 m/axis from
   0.60 m by **ideal** 1/√N you need **N=144 independent fixes** — already 16–20× more than land.

2. **The KF does WORSE than ideal 1/√N, and the gap WIDENS with cadence (ratio 1.2 → 1.72).** N_eff
   grows only as **N_real^0.72** (sublinear), not linearly. Cause: **in case C velocity is
   unobservable** (vision is position-only), so between fixes position information decays through
   process noise; closely-spaced fixes are increasingly correlated/redundant, not independent.
   Densifying the stream buys ever-less per added fix. Inverting the realized curve: reaching
   0.05 m/axis would need **~662 REAL landed fixes** in the 0.66 s window — vs the ~144 the ideal
   model predicts, and vs the ~7–8 that physically land at 37 m/s.

**Cadence-saturation fit** (debiased 100% arm, `in-plane_var = a + b/eff_rate`): the rate-term `b`
vanishes as cadence→∞ but the **constant variance floor `a` = 0.0113 m² → σ = 0.106 m in-plane
(~0.075 m/axis)** remains. This is the process-noise-between-fixes floor that NO fix rate beats.
**It does not clear 0.05 m.** (The empirical 240 Hz cell at 0.119 m sits just above this asymptote —
the conclusion is robust whether you read the realized cell or the extrapolated floor.)

---

## 3. The bias is cadence-invariant (the validity answer)

Per A1/A2 the binding term at gate-4 is BIAS. The raw arm makes this unmissable:

- Raw in-plane **bias** at the crossing = **0.29–0.32 m at EVERY cadence** (15→240 Hz). It is the
  measured per-fix bias (D-axis −0.346 m dominant, A3-confirmed speed-independent) which the KF
  tracks as real drift. `E[x̂ − p_true] → b` for any Kalman gain and any fix rate — **more fixes only
  make the filter MORE confident in the biased estimate** (NEES 6 → 62), never more accurate.
- Global de-bias (the debiased arm) removes that ~0.30 m and is mandatory — but A2/b1 showed a
  **per-track/per-gate residual (~0.45 m in-plane raw, b2: range-collapsing to ~0.11–0.34 m near
  transit) survives global de-bias**, and that residual is ALSO cadence-invariant (a constant within
  one track). So even the deployable debiased path carries an un-averageable bias floor on top of the
  0.106 m variance floor.

**If the binding term is BIAS, cadence does not touch the validity question — full stop.** This is
the plain statement the prompt asked for.

---

## 4. Feasibility — is >30 Hz even possible on the eval HW?

Serial single-thread budget = detector forward + PnP→KF chain (0.77 ms, MEASURED) + RewindKF replay
(~0.45 ms/fix, from the buffer docstring: horizon 0.5 s × 90 Hz IMU × ~10 µs/predict). The PnP chain
and rewind are **negligible next to the detector** at any HW.

| HW | detector p90 | frame p90 (det+PnP+rewind) | max throughput p90 | ≥60 Hz? | ≥120 Hz? |
|---|---|---|---|---|---|
| **~100 TOPS edge** (est 5–15 ms) | 15.0 ms | 16.35 ms | **~61 Hz** | YES | NO |
| **laptop CPU** (UB, 112–124 ms) | 124.3 ms | 125.7 ms | **~8 Hz** | NO | NO |

- **>30 Hz is feasible ONLY on edge HW** (≈61 Hz cap at p90; ~161 Hz at p50). The CPU detector is the
  hard wall — capped at ~8 Hz, it **cannot even sustain the inherited 30 Hz**, let alone exceed it.
- **But compute is not the constraint that matters.** Even if you could run the impossible 240 Hz, the
  debiased floor (0.106–0.12 m) and the raw bias floor (~0.30 m) both exceed the bar. And the *usable*
  fix rate at 37 m/s is gated by **exposure/blur + 4-corner visibility geometry** (a3: ~7–8 accepted
  fixes, 1.29 m spacing, gate exits FoV at r≈1.3 m), not by detector Hz. Higher cadence past ~30 Hz
  adds redundant, correlated, increasingly-blurred frames — diminishing returns into a floor.

---

## 5. Adversarial refutation — the strongest case FOR cadence, and why it fails

The strongest pro-cadence argument: *"the debiased σ is still falling at 240 Hz (0.119 m) — extrapolate
far enough and it crosses 0.05 m."* This fails three ways:
1. **The fit says it does not** — the variance floor `a` extrapolates to 0.106 m as cadence→∞; the
   falling term is the 1/rate part, and it asymptotes ABOVE the bar (process noise between fixes).
2. **Even if it did, the cadence is physically unreachable** — 0.05 m needs ~662 landed fixes → ~1000–
   2100 Hz detector (fix spacing 1.7–3.7 cm at 37 m/s). a3 measures ~7–8 real fixes; that is an >88×
   gap, and 4-corner visibility geometry caps the window regardless of detector Hz.
3. **It only ever attacks VARIANCE.** The deployable (raw / honest-debiased) error is bias-bound at
   ~0.30 m / ~0.45 m and is **cadence-invariant** — so the validity-relevant number never moves. A
   second pro-cadence line ("denser fixes make velocity observable, breaking the corr penalty") also
   fails: position-only vision gives no velocity information at any rate; N_eff stays sublinear (^0.72).

The refutation does not survive. Cadence is a variance lever only, it saturates above the bar, the
required rate is impossible, and the binding term (bias) is untouched.

---

## 6. Assumptions — MEASURED vs ASSUMED (flagged)

- **Per-fix σ ~0.50–0.60 m (MEASURED-via-real-model):** every cell assigns each fix the REAL
  `gate_pose_to_world_position` covariance (0.40 m floor ⊕ 1.4° lever ⊕ 0.30 m PnP). Per-fix σ_E
  reproduces at ~0.60 m near gate (sanity-checked stable across cadence — cadence does not change
  per-fix accuracy, only count). **The 0.40 m floor is BIAS-ABSORPTION — it cannot be lowered without
  re-opening the chi² leak it was added to fix (a1 §1), so it is itself a cadence-immune floor.**
- **Residual pool (MEASURED):** `off_ned` resampled from the 109 good fixes, range-band conditioned —
  identical to a1. RAW arm keeps the measured bias; DEBIASED subtracts the per-band mean (global
  de-bias). The per-GATE residual that survives global de-bias (a2/b1, ~0.45 m / range-collapsing
  b2 0.11–0.34 m) is NOT injected here — so the debiased σ in this sim is the **optimistic**
  variance-only floor; the deployable number is worse (b1's ~0.55 m).
- **Cadence ≥60 Hz is COUNTERFACTUAL** — the eval stream is 30 Hz and at 37 m/s only ~7–8 fixes land.
  The 60–240 Hz cells exist to prove the asymptote, not to propose a config. They are physically
  unreachable per §4 and a3.
- **Speed-invariance of per-fix error is OPTIMISTIC** — the pool is ~5.35 m/s data; 37 m/s blur
  (a3 long-exposure) would WIDEN per-fix σ, raising every number. So all c2 figures are best-case
  lower bounds, same caveat as a1/a3/b1.
- **CASE CONDITIONALITY:** this is the case-C (vision-only pose) worst case. If VQ2 streams
  LOCAL_POSITION_NED/ODOMETRY (case A/B) pose is pristine and the entire question is moot. Conditional
  on organizer Q①.

---

## 7. Recommendation (for the commander)

1. **Do NOT pursue higher detector cadence as a gate-4 validity fix.** It is a variance-only lever
   that saturates at ~0.11 m in-plane (>2× the bar), is bias-blind, and the rate required to even
   approach the bar (~1–2 kHz) is physically impossible at 37 m/s. >30 Hz also requires edge HW the
   CPU path can't provide.
2. **The 30 Hz inherited default is NOT the bottleneck** — confirms the memory note. Raising it to
   ~60 Hz on edge HW buys the debiased σ from 0.34→0.26 m (still NO-GO) at the cost of more
   blur-degraded, correlated frames. Not worth it for validity; possibly marginally useful for the
   along-track/phase term (a4), which is a separate question.
3. **Gate-RELATIVE observation remains the only candidate fix** (a1/a2/b1 unanimous): close the loop
   on the SEEN gate-4 corners. It converts the absolute-NED variance to close-range corner
   reprojection (mm-pixel-tight) AND sidesteps the per-gate bias — neither of which cadence touches.
4. **If anyone proposes "just run the detector faster," the answer is in §2–§3:** N_eff grows only as
   N_real^0.72 (vel unobservable), so 16× more fixes buys ~7× more effective fixes, lands you at
   0.12–0.16 m, and does nothing to the ~0.30 m bias. More fixes ≠ closer to valid.

---

## MEMORY-DELTA:
- c2 cadence/multi-fix-fusion branch DONE: **more fixes does NOT close the gate-4 gap.** Swept
  detector cadence 15→240 Hz on the REAL KF+RewindKF+fix-cov stack (28 cells, 600 seeds, edge L).
- DEBIASED (variance) arm: 30 Hz→0.34 m in-plane σ; 240 Hz→0.157 m (47%) / 0.119 m (100%, 154 fixes);
  **cadence→∞ variance floor = 0.106 m (~0.075 m/axis) — still 2× over the 0.05 m bar.** Process noise
  between fixes (vel UNOBSERVABLE in case C) is the floor; N_eff grows only ~N_real^0.72 (sublinear,
  correlated fixes). At realistic 30 Hz/47% the KF averages only ~6 effective independent fixes.
- RAW (real, biased) arm: in-plane RMS **bias-pinned at ~0.30 m at EVERY cadence**; NEES EXPLODES
  6→62 as cadence rises (filter tracks the constant bias as truth, gets overconfident). **Cadence does
  nothing to BIAS** — and per A1/A2 BIAS is the binding term, so cadence does NOT touch validity.
- To reach 0.05 m/axis: ~144 ideal independent fixes / ~662 real KF fixes → ~1000–2100 Hz detector
  (1.7–3.7 cm fix spacing) — impossible (a3: only ~7–8 fixes land at 37 m/s, >88× gap).
- FEASIBILITY: >30 Hz possible ONLY on edge HW (p90 ~61 Hz cap; PnP chain 0.77ms + rewind 0.45ms
  negligible vs detector). CPU detector capped ~8 Hz — can't even hold 30 Hz. But cadence is NOT the
  binding constraint. Fix = GATE-RELATIVE observation, not a faster detector. CONDITIONAL on case C / Q①.
