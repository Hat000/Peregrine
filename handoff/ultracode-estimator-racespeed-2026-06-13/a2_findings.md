# a2 — Gate-4 un-filterable in-plane BIAS floor (estimator-racespeed)

**Role:** the BIAS decomposition. The KF crushes zero-mean noise by averaging fixes;
it does **not** remove a consistent bias. Quantify the un-filterable in-plane bias
floor at gate-4 — the term no amount of filtering removes.

**Bottom line (reproduced, seed 20260613):**

| quantity | value |
|---|---|
| Gate-4 in-plane bias floor, **RAW** (no de-bias) | **≈ 0.52–0.60 m** |
| Gate-4 in-plane bias floor, **after GLOBAL de-bias** (VISION-CAL) | **≈ 0.44–0.52 m** |
| 95% CI on de-biased in-plane bias (raw-fix path) | **[0.30, 0.60] m** |
| Clears the **0.05 m** bar after de-bias? | **NO** — off by **~9–10×** |

The global de-bias barely helps at gate-4: it removes only ~0.08 m of the ~0.52 m
in-plane bias, because **gate-4's lateral (E) error is genuinely per-gate, not global.**
Absolute world-fixing cannot reach the 0.05 m in-plane bar at gate-4 by any amount of
filtering or de-biasing. This is a hard wall on the **absolute** case-C path; the fix is
a **gate-relative observation** (close on the seen gate corners), which sidesteps the
per-gate registration bias entirely.

---

## 1. The three-way decomposition (per NED axis)

Each per-fix world-fix error `off_ned = p_fix − p_true` is split into:

1. **GLOBAL bias** — constant across **all** gates → removable by VISION-CAL global de-bias.
2. **PER-GATE residual bias** — constant within a track for a given gate → does **not**
   average out within the gate-4 approach; identical every lap of a fixed track.
3. **zero-mean per-fix NOISE** — crushed by the KF.

Recovered via the **real** `vision_cal.resurvey()` (shipped robust median/MAD averager
`racer.gate_mapper.estimate_map_pose_aided`, innovation-gated maha ≤ 16.27, 4-corner
only — the KF-accepted set). Offsets are `est_centre − map = −median(off_ned)|gate`.

### Per-gate registration offset (NED, m) — `est − map`, with estimation 1σ
| gate | N | E | D | σ_N | σ_E | σ_D |
|---|---|---|---|---|---|---|
| g0 | +0.279 | +0.037 | +0.299 | 0.209 | 0.071 | 0.037 |
| g1 | +0.236 | −0.218 | +0.231 | 0.099 | 0.043 | 0.018 |
| g2 | +0.478 | +0.276 | +0.324 | 0.171 | 0.041 | 0.033 |
| g3 | +0.758 | −0.555 | +0.346 | 0.288 | 0.055 | 0.039 |
| **g4** | **+0.292** | **+0.457** | **+0.387** | 0.084 | 0.063 | 0.038 |
| g5 | +0.302 | −0.321 | +0.270 | 0.104 | 0.034 | 0.026 |

### Per-axis decomposition (NED, m)
- **GLOBAL bias** (mean across 6 gates, gate sense): `[+0.391, −0.054, +0.309]`
  → in fix sense `[−0.391, +0.054, −0.309]`, which matches the independently-measured
  `MEASURED_FIX_BIAS_NED = [−0.42, +0.06, −0.28]` (sign chain confirmed). This is the
  part VISION-CAL can remove with a single constant offset.
- **PER-GATE residual** (offset − global, gate sense): the un-removable part. Across-gate
  residual σ = **`[0.198, 0.381, 0.056]`** (N, E, D). The **E axis dominates** the
  per-gate scatter (±0.38 m), N is mid (±0.20 m), **D is small (±0.056 m)**.
- **NOISE** (per-fix std, FACTS GOOD cut, n=109): `[0.82, 0.58, 0.44]` m — averaged away
  by the KF, not part of the bias floor.

**Reproduced FACTS exactly:** per-fix bias `[−0.285, +0.064, −0.346]`, noise std
`[0.82, 0.58, 0.44]`, n=109 (course_60s, |off|<3 m, associated).

---

## 2. Projection onto the gate-4 frame (the binding numbers)

Gate-4 geometry (FACTS): motion ~along −N, gate normal ~−N ⇒ **in-plane = E (lateral) +
D (vertical); along-track = N.** Binding in-plane miss = `sqrt(E² + D²)`.

Two **independent** estimates, both reported (they agree on magnitude and verdict):

### Path A — re-survey (robust median, gate sense)
- Raw offset (E, D) = (**+0.457, +0.387**) → **in-plane = 0.599 m**
- After global de-bias (E, D) = (**+0.511, +0.078**) → **in-plane = 0.517 m**
- Estimation 1σ on the residual (E, D) ≈ (0.067, 0.041) m — tight; the bias is real, not sampling noise.

### Path B — raw KF-accepted gate-4 fix stream (mean, fix sense; n=26, range 8.3–27.3 m)
- Raw fix bias (E, D) = (**−0.380, −0.359**) → **in-plane = 0.523 m**, 95% CI [0.38, 0.67]
- After global de-bias (E, D) = (**−0.440, −0.079**) → **in-plane = 0.447 m**, 95% CI [0.30, 0.60]
- Per-axis de-biased CIs: **E = −0.44 [−0.58, −0.29]**, **D = −0.08 [−0.15, −0.01]**,
  N(along-track) = +0.15 [−0.06, +0.34].

**Sign reconciliation (not a contradiction):** Path A reports the *gate-centre* offset
(`est − map = −off`); Path B reports the *fix* bias (`+off`). Hence E shows as **+0.51 (A)**
vs **−0.44 (B)** — same physical lateral bias, opposite convention, same ~0.45–0.51 m
magnitude. The raw E `off_ned` values are 24/27 negative (median −0.484), so the lateral
error is a consistent, structured offset, not scatter.

### Range sensitivity (raw-fix path, the actual transit window)
| true range window | n | in-plane bias (E,D) m | |in-plane| |
|---|---|---|---|
| ≤ 8 m | (few) | — | — |
| ≤ 12 m (close transit) | 12 | (−0.249, −0.227) | **0.337** |
| ≤ 16 m | 17 | (−0.344, −0.286) | **0.447** |
| ≤ 27 m (full accepted) | 26 | (−0.380, −0.359) | **0.523** |

Even at the **closest** transit window (≤12 m), the in-plane bias is **0.34 m** — still
**~7× the 0.05 m bar.** Robust to which global de-bias is applied (`MEASURED_FIX_BIAS`
→ 0.447; re-survey global → 0.437).

### Headline floor
- **RAW in-plane bias floor at gate-4: ≈ 0.52–0.60 m** (point estimates of the two paths;
  I report the conservative 0.599 m).
- **After global de-bias: ≈ 0.44–0.52 m** (I report the conservative 0.517 m).
- **Does NOT clear 0.05 m** — even the lower 95% CI bound (0.30 m) is 6× the bar.

The along-track N residual (+0.15 m) is separate: it shifts *when* the plane is crossed,
not the in-plane miss, and is what latency staleness `v·L` adds to.

---

## 3. Why the KF cannot remove the per-track residual (quantitative)

The KF position update is a precision-weighted average of fixes around the predicted state.
For a fix stream `z_k = p_true + b + n_k` with **constant** bias `b` and zero-mean noise
`n_k` (σ_meas):

- The **noise** term `n_k` is averaged down. After fusing `m` fixes the variance of the
  state estimate shrinks ~`σ_meas² / (m·η)` (η<1 from process-noise growth between 30 Hz
  fixes). With σ_meas,E ≈ 0.38 m and ~10–20 accepted fixes on the g3→g4 straight, the
  **variance** contribution to E drops to a few cm — the noise is *not* the problem.
- The **bias** term `b` is the **expected value of every innovation**. The KF has no
  reference that says "this offset is wrong" — a constant `b` is indistinguishable from
  the drone genuinely being at `p_true + b`. The estimator converges to `p_true + b`, i.e.
  it **tracks the bias as truth.** Formally: `E[x̂ − p_true] → b` as `m → ∞`, for any
  Kalman gain. No amount of averaging, no R/Q tuning, removes it — that is the definition
  of an estimator bias from a biased measurement.
- The `FIX_COV_FLOOR_STD = 0.40 m` floor is **bias-absorption**, by design (per `localization.py`):
  it inflates R so the KF does **not** chase the per-gate systematics tightly. It caps the
  damage but does not remove `b`; it cannot come below the per-gate bias spread (~0.38 m E)
  without the filter over-trusting biased fixes.

So the gate-4 in-plane bias floor is a **measurement-bias → estimator-bias** transfer: the
filter's *variance* can reach the 0.05 m bar, but its *bias* is pinned at ~0.45–0.52 m in-plane.
**Total in-plane error = bias ⊕ filtered-noise ≈ 0.45–0.52 m ⊕ ~0.04 m ≈ 0.45–0.52 m.**
The bias term alone blows the 0.155 m gate-4 contact-true margin by 3×.

---

## 4. Determinism / second-calibration-lap angle

The bias is **per-track constant**: the course/seed is fixed within an attempt, so the
gate-4 registration offset `b_g4` is **identical every lap** of that track. This is exactly
what a calibration lap exploits.

**Can a 2nd calibration lap remove it? — Partially, with two hard caveats.**

1. **What a cal lap CAN remove:** if the cal lap has a *better-than-vision* truth reference
   for the drone pose (cases A/B stream `LOCAL_POSITION_NED`/`ODOMETRY`, or a slow accurate
   lap), then `vision_cal.resurvey()` measures `b_g4` directly and you subtract a **per-gate**
   offset (not just the global one). The estimation 1σ on `b_g4` is tight: (σ_E, σ_D) ≈
   (0.063, 0.038) m. So a **per-gate** de-bias drives the residual down to roughly the
   estimation uncertainty: **sqrt(0.063² + 0.038²) ≈ 0.074 m** — still **above 0.05 m**, but
   close, and a longer/slower cal lap with more inliers tightens it further (σ ∝ 1/√n).
   *This is the single most important corrective lever the determinism-per-track property buys.*

2. **What a cal lap CANNOT remove — the circularity caveat:** the re-survey here is
   reconstructed from the **same vision chain** whose systematics we are trying to measure
   (in VQ1 it worked because the GIVEN pose was ground truth). In a pure **case-C** race with
   no privileged pose, a vision-only cal lap measures `b_g4` *through the same biased PnP/
   extrinsic chain*, so any **range-correlated or geometry-correlated** systematic (depth
   bias, extrinsic mis-cal, bearing slope) is **NOT** observable and **survives the cal lap**.
   The lateral discrepancy between gates (g3 E −0.56 vs g4 E +0.46) is the fingerprint of
   exactly such a chain-level systematic, not pure independent registration noise. So the
   *floor on what a vision-only cal lap leaves* is the chain-correlated part — which the
   `lateral_range_decomposition` in `vision_cal.py` is built to separate (constant map-reg vs
   range-slope bearing bias) and should be run per-gate before trusting a cal-lap de-bias.

3. **Speed coupling:** the cal-lap re-survey here is from fixes at **5.35 m/s median**, max
   range 23–27 m. At race speed (~37 m/s post-gate-3) the last-fix-to-transit distance and
   motion blur shift the operative range band and the bias itself — the per-gate offset is
   *track*-constant but not *speed*-constant, so the cal lap must be flown at (or corrected to)
   the **race-speed range band** to be valid. The ≤12 m vs ≤27 m table above shows the bias
   moves 0.34→0.52 m across range bands.

**Verdict on cal lap:** a per-gate de-bias from a privileged-pose (case A/B) cal lap can pull
the gate-4 in-plane residual from ~0.52 m down to ~0.07 m — **a 7× improvement, landing just
above 0.05 m.** A vision-only (true case-C) cal lap removes the per-gate *constant* but leaves
the chain-correlated systematic and cannot be trusted to reach the bar. **Either way, the
absolute path is marginal at best; the robust fix is gate-relative observation** (close the
loop on the *seen* gate-4 corners for centering), which makes the per-gate map-registration
bias irrelevant to the in-plane miss entirely (FACTS §point 2, the highest-leverage fix).

---

## 5. Files
- `a2_bias.py` — the decomposition (run with `PYTHONPATH="src;handoff/ultracode-vision-case-c-2026-06-13"`).
- `a2_bias_results.json` — machine-readable results (regenerated on every run, deterministic seed 20260613).

## 6. Caveats / honesty
- Both paths use the VQ1-regime characterization where GIVEN pose = ground truth, so the
  *absolute* bias numbers are trustworthy; the *case-C cal-lap removability* in §4.2 is the
  one place the chain-circularity bites and I have flagged it explicitly rather than asserting
  the 0.07 m residual is achievable in pure case C.
- Gate-4 has n=26–27 accepted fixes; the bootstrap CI ([0.30, 0.60] m) already reflects this
  small sample. The verdict (does not clear 0.05 m) holds across the entire CI by 6×.
- D-axis per-gate residual is genuinely small (~0.06–0.08 m); the in-plane floor is
  **lateral-(E)-dominated**. If a future extrinsic/bearing re-cal kills the E systematic, the
  D residual alone (~0.08 m) would still miss 0.05 m but only by 1.6×.
