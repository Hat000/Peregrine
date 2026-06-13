# c3 — Detector / PnP accuracy levers at the gate-4 approach range

Agent c3 (estimator/perception, opus-4.8). OFFLINE analysis only. Builds on the SHIPPED
fix-cov channels (`range_anisotropic_R.py`, coeffs `range_R_coeffs.json`) and the MEASURED
per-fix pool (`perception-char-2026-06-08/characterize_course_60s.json`, n=109 good fixes).
Files: `c3_pnp_accuracy.py`, `c3_pnp_accuracy_results.json`. Seed 20260613, fully reproducible.

**BRANCH question:** can sub-pixel corner refinement, a better detector, 2-corner-fallback
handling, or gate-size calibration reduce sigma_px → sigma_depth (r²) / sigma_lateral (r) at
the gate-4 approach, and does any of it touch the BINDING term (the gate-4 in-plane miss)?

---

## HEADLINE

**The detector/PnP accuracy branch is a DEAD END for the binding gate-4 in-plane (E/D) error.**
The in-plane world-fix uncertainty at the gate-4 window is **NOT pixel-noise-limited** — it is
limited by the **1.4° attitude lever + the 0.40 m cov floor**, neither of which a detector
improvement can touch. Quantitatively: **a *perfect* detector (sigma_px → 0) reduces the
in-plane (tangential) world-fix 1-sigma by 0.000 (= 0.0–0.7% of the variance) at every gate-4
range 5–20 m.** The one place this branch DOES bite a bias — gate-size calibration — removes a
range-proportional **depth** bias that at gate-4 lands almost entirely on the **N (along-track)**
axis (measured corr(off_N, range_err) = **0.994**), i.e. it improves *when* you cross the plane,
NOT the in-plane miss.

**Verdict: FEASIBLE-WITH-CONDITIONS but LOW-LEVERAGE.** The only worthwhile item here is the
**gate-size recalibration** (true inner size ≈ **1.535 m**, not the assumed 1.5 m), which is a
cheap one-line bias fix — but it is **along-track only** at gate-4 and does not move the binding
in-plane term. Sub-pixel refinement and "a better detector" are **wasted effort** for this risk.

This confirms and sharpens a1/a2: the binding in-plane term is **VARIANCE** (the lever+floor
tangential wall, a1's 0.239 m) plus a per-gate **lateral BIAS** (a2's 0.47 m E) — and *neither is
attackable by tightening the detector*. The estimator risk does not yield to detector accuracy.

---

## ADVERSARIAL FIRST (the prompt's own challenge): is the detector already near the pixel floor?

**YES — emphatically. The detector is already WELL below the pixel-noise floor that would matter.**

| measurement | value | reading |
|---|---|---|
| measured reproj_px **median** | **0.503 px** | already 3× tighter than the assumed nominal 1.5 px |
| measured reproj_px mean / p90 | 0.99 / 1.70 px | mean inflated by a thin tail; the bulk is ~0.5 px |
| corr(reproj_px, within-gate world-fix noise) | **−0.145** | **~zero** — world-fix scatter is NOT driven by corner residual |
| within-gate noise: reproj<0.5 px bin | 1.16 m norm-std | the *tightest-reproj* fixes have the *largest* world-fix noise |
| within-gate noise: reproj 0.5–1.5 px bin | 0.72 m | — |

The corner localizer is already sub-pixel; the world-fix noise is **decorrelated** from it. So
**refinement (driving reproj 0.5 → 0.1 px) cannot reduce the world-fix error** — the residual
lives elsewhere (attitude/extrinsic lever + per-gate registration), not in corner pixels. The
adversarial hypothesis is not just plausible, it is the *measured truth*. This is the load-bearing
result of the branch: **the cheapest detector lever (refinement) buys essentially nothing.**

---

## Q1 — sigma_px → sigma_depth (r²) / sigma_lateral (r): the actual transfer, quantified

Using the shipped anisotropic R-channels with the pixel-Fisher coefficients re-derived at each
sigma_px (`c2 = 2σpx/(f·s·√4)`, `a1 = σpx/(f·√4)`, f=320, s=1.5):

### In-plane (TANGENTIAL = E lateral + D vertical at gate-4) — the BINDING axis
| range | std_lat_pixel @1.5px | attitude-lever std | **total in-plane std** | pixel share of variance |
|---|---|---|---|---|
| 5 m  | 0.012 | 0.122 | **0.307** | 0.15 % |
| 8 m  | 0.019 | 0.196 | **0.343** | 0.30 % |
| 12 m | 0.028 | 0.293 | **0.408** | 0.48 % |
| 16 m | 0.037 | 0.391 | **0.483** | 0.60 % |
| 20 m | 0.047 | 0.489 | **0.566** | 0.69 % |

Pixel noise is **0.15–0.7 % of the in-plane variance**. Driving sigma_px from 1.5 → 0.5 → 0.1 → 0
changes the total in-plane std in the **3rd–4th decimal** (e.g. r=12 m: 0.4076 → 0.4068 → 0.4066
→ 0.4066 m). **The in-plane axis is set by the lever (σ_θ·r) ⊕ the tangential floor (0.282 m),
and a perfect detector leaves it unchanged.** The `Q_floor_dominance` table makes this explicit:
`pixel_reducible_fraction = 0.000` at every range.

**Independent confirmation from the MEASURED data:** the within-gate lateral (E) noise grows as
**0.0254·r** — matching the attitude-lever slope σ_θ = 0.0244 (1.4°), and ~3× steeper than what
the *measured* 0.5 px reproj would produce (a1@0.5px = 0.0087·r). The slope is a lever signature,
not a pixel signature.

### Depth (RADIAL = along-LOS ≈ along-track N at gate-4) — NOT the in-plane binding axis
The r² depth pixel term *does* respond to sigma_px (it scales linearly): at 1.5 px the depth std
is 0.45 m @ 12 m, 1.25 m @ 20 m; at 0.5 px it drops to 0.15 / 0.42 m. **But two reasons make this
irrelevant to the binding miss:** (1) the 0.40 m radial **floor** dominates the pixel term below
~11 m range (crossover at r where c2·r² = 0.40 ⇒ **r = 11.3 m**), so in the close transit window
the depth channel is floor-limited regardless of pixels; (2) the depth axis is **along-track (N)**
at gate-4 — it perturbs the crossing *timing*, not the in-plane miss (FACTS §gate-4 geometry).

---

## Q2 — which lever attacks BIAS vs VARIANCE (the crux split)

| lever | attacks | mechanism | gate-4 effect | verdict |
|---|---|---|---|---|
| **gate-size calibration** | **BIAS (depth)** | corrects assumed inner size 1.5 → measured ~1.535 m; removes the range-proportional depth bias | **ALONG-TRACK (N) only** — corr(off_N, range_err)=0.994 | cheap, do it, but **off the binding axis** |
| sub-pixel refinement | VARIANCE (pixel) | tightens corner residual | **0.000 m** (detector already sub-pixel; decorrelated) | **wasted** |
| "better detector" | VARIANCE (pixel) | lower per-corner σpx | ≤ 0.7 % of in-plane variance | **wasted for in-plane** |
| 2-corner / P3P-fallback handling | VARIANCE (×9 infl) + a small BIAS | avoids the ×9 P3P cov inflation + P3P cheirality ambiguity on clipped gates | only 4/109 fixes are 3-corner; rare on the LEVEL g3→g4 approach where gate-4 stays in frame longer | **low frequency → low leverage here** |

### The gate-size BIAS finding (the one real bias lever in this branch)
The measured depth bias is **range-proportional**: `range_err = −0.0151·r + 0.0018` m — a clean
fractional (size-mismatch) signature with ~zero constant offset. Median fractional depth bias =
**−2.28 %**, implying the **true inner square is ≈ 1.535 m, not 1.500 m** (the detector/PnP under-
sizes the gate by 2.3 %, so it places it 2.3 % too close → depth under-estimate). Calibrating
`GATE_INNER_SIZE_M` to the measured value **removes this depth bias to ~0 residual** (cal residual
const = 0.000 m). **This is a genuine per-fix BIAS removal — exactly the "gate-size-model mismatch
is a depth bias" hypothesis in the prompt, confirmed.** BUT at gate-4 it is along-track:

- gate-4 measured off_NED = **[−0.411, −0.363, −0.302]** m; depth bias = **−0.355 m**, and
  corr(off_N, range_err) = **0.994** ⇒ the whole depth bias sits on **N (along-track)**.
- gate-4 **in-plane** bias (E, D) = **(−0.363, −0.302)**, norm **0.472 m** ← gate-size cal does
  NOT touch this. This is the per-gate **lateral registration / bearing** systematic that a2
  isolated (per-gate E residual σ = 0.38 m, the dominant un-filterable term).

So: **gate-size cal removes ~0.18 m of ALONG-TRACK bias at a 12 m last fix (and the staleness it
compounds), improving crossing-timing and the N error a4/latency cares about — but it leaves the
0.47 m in-plane bias and the 0.31–0.57 m in-plane variance wall completely intact.**

---

## Why this branch cannot move the binding term (the physics)

The gate-4 in-plane (E, D) error decomposes as:
1. **VARIANCE** = attitude-lever (σ_θ·r, 1.4°) ⊕ tangential floor (0.282 m) ⊕ pixel-lateral
   (a1·r, *negligible*). The lever and floor are **independent of the detector** — they come from
   IMU/extrinsic attitude uncertainty and the bias-absorption floor. → **detector-immune.**
2. **BIAS** = per-gate lateral registration + bearing-slope systematic (a2's 0.47 m E). This is a
   **map/extrinsic** systematic, not a corner-localization error. Better corners reproject the
   *same wrong* world ray. → **detector-immune** (refinement re-fits the same biased extrinsic).

The detector controls only the pixel term, which is 0.15–0.7 % of the in-plane variance and 0 %
of the in-plane bias. **There is no detector/PnP accuracy lever that moves the binding gate-4
in-plane miss.** The levers that *do* move it are (per a1/a2): (a) **gate-RELATIVE observation**
(close on the seen gate-4 corners — sidesteps both the per-gate bias and the absolute lever), and
(b) **shrinking σ_θ** (a tighter attitude/extrinsic estimate — that is the lever channel, an IMU/
calibration problem, NOT a detector problem). Those are out of this branch's scope but are where
the leverage actually is.

---

## What IS worth doing from this branch (the conditional FEASIBLE items)

1. **Recalibrate `GATE_INNER_SIZE_M` 1.5 → ~1.535 m** (or equivalently apply a +2.3 % depth
   correction). One-line change, removes a real per-fix depth BIAS, helps along-track / crossing
   timing and reduces the N staleness latency compounds. **Effort: trivial. Leverage: small,
   off-axis.** CAVEAT: the 1.535 m is fit from the VQ1-regime char where GIVEN pose was truth; in
   pure case-C the size could be re-measured but only through the same chain — fit it from a
   privileged-pose (case A/B) lap if available, else treat 1.535 m as the best prior.
2. **Keep the existing weighted-PnP / Tukey refinement** — it is already delivering ~0.5 px and is
   why pixel noise is a non-issue. **Do not invest further** in sub-pixel accuracy.
3. **2-corner / clipped-gate handling**: low frequency on the gate-4 approach (gate stays in frame
   on the level g3→g4 straight; only 4/109 fixes were 3-corner). Keep the ×9 P3P inflation as a
   safety valve; not worth special-casing for the gate-4 risk specifically.

---

## Assumptions — MEASURED vs ASSUMED (flagged)

- **MEASURED:** reproj_px, n_corners, off_ned, range_err, true_range from the 109 good fixes
  (course_60s, |off|<3 m, associated). Gate-size fractional bias, the reproj–noise decorrelation,
  the lever-slope match, and the gate-4 axis projection are all *direct measurements*, not models.
- **SHIPPED model (used as-is):** c2/a1 pixel-Fisher coefficients, σ_θ = 1.4°, 0.40 m floor,
  0.282 m tangential floor (`range_R_coeffs.json`), GATE_INNER_SIZE_M = 1.5, f = 320, N_eff = 4.
- **ASSUMED:** gate-4 LOS ≈ −N (FACTS geometry) ⇒ depth = along-track, lateral/vertical = in-plane.
  The transit last-fix range band (5–20 m sweep) is representative, not a single measured value.
- **HONESTY CAVEAT (same as a1/a2):** the per-fix pool is **~5.35 m/s, range ≤ 23.3 m** data —
  there is **NO 37 m/s data**. At race speed motion blur could WIDEN corner residual and re-make
  pixel noise relevant. So the "detector is already at the floor" conclusion is a **VQ1-regime
  measurement**; if 37 m/s blur pushes reproj from 0.5 px toward several px, the depth (r²) channel
  beyond 11 m would re-inflate and sub-pixel/anti-blur (shorter exposure, deblur) could then matter
  for the depth/along-track axis — but **still not for the in-plane axis**, which stays lever+floor
  limited. SHADOWPC-VISION-CAL must measure reproj_px at race speed before fully closing this.
- **CASE CONDITIONALITY:** entire analysis is the case-C worst case. If VQ2 streams pose (case A/B),
  the risk is moot. Conditional on organizer Q①.

---

## MEMORY-DELTA:
- c3 detector/PnP-accuracy branch DONE: **DEAD END for the binding gate-4 in-plane miss.** A
  *perfect* detector (σpx→0) reduces in-plane world-fix std by **0.000 m** (pixel = 0.15–0.7 % of
  in-plane variance) at every range 5–20 m. In-plane is **lever(1.4°)+floor(0.40/0.282 m) limited**,
  detector-immune.
- ADVERSARIAL CONFIRMED: detector already **sub-pixel** (reproj median **0.50 px**, not assumed
  1.5) and within-gate world-fix noise is **decorrelated** from reproj (corr **−0.145**) ⇒
  **sub-pixel refinement / "better detector" buys ~nothing**. Measured in-plane lateral noise slope
  0.0254·r matches the attitude-lever σ_θ, not pixels.
- ONE real BIAS lever: **gate-size calibration** — depth bias is range-proportional (−2.28 %),
  implied **true inner size ≈ 1.535 m** (assumed 1.5). Removing it kills the depth bias to ~0, BUT
  at gate-4 the depth bias is **ALONG-TRACK (N)** (corr off_N vs range_err = **0.994**) — improves
  crossing timing / N staleness, **does NOT touch the 0.47 m in-plane (E) bias**.
- Leverage is NOT in the detector. It is in (a) gate-RELATIVE observation and (b) shrinking σ_θ
  (attitude/extrinsic), both out of this branch. 2-corner handling low-frequency on gate-4 approach.
- CAVEAT: pool is 5 m/s / ≤23 m; 37 m/s blur could re-inflate the DEPTH (along-track) channel and
  make anti-blur relevant there — never the in-plane axis. Measure reproj_px at race speed (SHADOWPC).
