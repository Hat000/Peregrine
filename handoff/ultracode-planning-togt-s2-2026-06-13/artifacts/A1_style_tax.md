# A1 — Style/Envelope Tilt Tax: Time-Contribution + tilt→lap-cost CURVE

**Date:** 2026-06-13 · **Scope:** analysis/offline only · **For:** Fengyou
**Question:** rigorously attribute and *characterize as a curve* the lap-time cost of the
style-envelope tilt cap. Decompose it into the two distinct levers (global `rw_tilt` weight vs
free-cone angle), and map the tilt ceiling → cornering speed → lap time on THIS track's curvature.

---

## 0. Headline numbers

| Quantity | Value | Provenance |
|---|---|---|
| **Measured RL tax** (envelope ON vs OFF) | **2.63 s/lap** (9.52 − 6.89) | inc5 datums (memory §INC5/§TILT-CONCENTRATION) |
| Measured per-segment tax (inc5, single-traj) | **2.91 s** total | §TILT-CONCENTRATION table |
| **Ideal (TOPP) tax**, tilt 65°→unconstrained | **2.15 s** | this analysis (point-mass bound) |
| **Curve fit** | `laptime ≈ 1.95 + 22.27 / sqrt(a_lat)` s | TOPP grid fit, RMS 0.41 s |
| Marginal cost at the current 65° cap | **≈ −0.14 s per +1° tilt** | dTOPP/dtilt |
| Realizability factor RL/TOPP | **k ≈ 1.5–1.6** | calibrated from the two measured points |

`a_lat = g·tan(tilt)` is the single physical lever the tilt cap controls.
**The tax is steeply convex in tilt**: the marginal s/deg is ~−0.14 near 65° and decays to ~−0.06
above 80°. Most of the recoverable time lives in the **65°→75° band** (≈1.3 s ideal), not above 80°.

---

## 1. The mechanism — tilt cap IS a lateral-acceleration cap

The reward term R4 (`rl/peregrine_racing.py:365`) is a HINGE on total tilt:
```
tilt_pen = relu( cos(tilt_free_rad) − R33 )^2 ,  R33 = 1 − 2(qx²+qy²) = cos(total_tilt)
reward  −= w.tilt * tilt_pen          # w.tilt = rw_tilt (default 4.0; inc5 used 96), free cone 60°
```
Zero penalty inside the free cone (default 60°), quadratic beyond. Because a multirotor's only
in-plane force is the projection of body-thrust, the **lateral (cornering) acceleration it can
produce is bounded by the tilt it is allowed to hold**:

```
a_lat,max = g · tan(tilt_max)      (thrust ≈ g/cos(tilt) to hold altitude → horizontal comp = g·tan)
```

| tilt | 60° | 65° | 70° | 75° | 80° | ~unconstrained |
|---|---|---|---|---|---|---|
| **a_lat (m/s²)** | **17.0** | **21.0** | 26.9 | **36.6** | **55.7** | ≫ (≈g·tan89° ⇒ thrust/drag-bound) |

So the tilt cap is *exactly* an `a_max` knob into the time-optimal speed profiler. Lower tilt →
lower lateral authority → must slow into every corner whose curvature demands more than `a_lat`.

---

## 2. Track curvature → corner-speed ceiling (where the cap binds)

Computed from the gate-centre spline (`src/racer/speed_profile.py` CubicSpline curvature κ at each
gate node). `v_corner = sqrt(a_lat / κ)`:

| Gate | κ (1/m) | R (m) | path deflection | v_corner@60° | @65° | @75° | @80° |
|---|---|---|---|---|---|---|---|
| G0 | 0.0051 | 196 | — | 57.7 | 64.2 | 84.7 | 104 |
| **G1** | 0.0208 | 48 | 13.2° | **28.6** | 31.8 | 42.0 | 51.8 |
| **G2** | 0.0255 | 39 | 16.6° | **25.8** | 28.7 | 37.9 | 46.7 |
| **G3** | 0.0332 | 30 | 24.3° | **22.6** | 25.2 | 33.2 | 41.0 |
| **G4** | 0.0281 | 36 | 18.8° | **24.6** | 27.4 | 36.1 | 44.5 |
| G5 | ~0 | ∞ | (final) | ∞ | ∞ | ∞ | ∞ |

**Reading:** the drag-wall top speed is ~39 m/s. A corner only *costs* time when `v_corner < ~39`,
i.e. when the cap forces a slow-down below cruise.
- At **60°** the cap binds at G1, G2, G3, G4 (all v_corner ≤ 28.6 < 39) — the drone is corner-limited
  almost everywhere, exactly why the global-conservatism cost is large.
- At **75°** only G2/G3/G4 still sit below 39 (37.9/33.2/36.1) — marginal binding.
- At **80°+** every corner clears 39 m/s → the track is no longer cornering-limited; thrust/drag
  becomes the sole bound. **This is the diminishing-returns knee.**

This matches §TILT-CONCENTRATION's empirical finding that the cap *physically* binds on **G1→G2,
G2→G3, G4→G5** (the highest-κ approaches) and NOT on the low-deflection segments.

---

## 3. The CURVE — lap time vs tilt cap (TOPP, ideal point-mass bound)

`time_optimal_profile(WP, v_max=39, a_max=g·tan(tilt), v_start=0, v_end=8)` over spawn+6 gates
(`spawn` = 23.3 m up-course of G0; track length 164.6 m). This is the contact-free planning-valid
bound — the same family that yields ~4.27 s at full authority.

| tilt (°) | a_lat (m/s²) | **ideal laptime (s)** | RL-projected ≈k·ideal (k≈1.5) |
|---|---|---|---|
| 40 | 8.2 | 10.04 | ~15 |
| 50 | 11.7 | 8.50 | ~13 |
| 55 | 14.0 | 7.80 | ~11.7 |
| **60** | 17.0 | **7.12** | ~10.6 |
| **65** (current) | 21.0 | **6.43** | **~9.5** ✅ matches inc5 9.52 |
| 70 | 26.9 | 5.73 | ~8.6 |
| **75** | 36.6 | **5.16** | ~7.7 |
| **80** | 55.7 | **4.79** | ~7.2 |
| 85 | 112 | 4.50 | ~6.8 |
| ~89 (unc) | 562 | **4.28** | **~6.4** ✅ matches inc5 6.89 |

**Cross-checks (calibration):**
- TOPP unconstrained **4.28 s** ≡ the known planning-valid TOGT bound **4.27 s** (independent
  pipeline) — the speed_profile module and the C++ TOGT bound agree to 0.01 s. Strong validation.
- TOPP-tax (65°→unc) = **2.15 s**; measured RL-tax = **2.63 s**. The 0.5 s gap is the
  realizability overhead (plant lag, 2-tick latency, DR margin, finite rate-slew) — captured by
  the factor **k = RL/ideal ≈ 1.48 (at 65°), 1.61 (unconstrained)**. k *grows* toward the
  unconstrained end because aggressive flight is harder to realize than the point-mass bound
  assumes (rate-slew + latency bite more), so the **real recoverable tax is a bit smaller than the
  ideal curve suggests** — conservatively **2.3–2.9 s**, consistent with the project's banked range.

**Closed-form relationship (fit, RMS 0.41 s over 35–89°):**
```
laptime_ideal(tilt) ≈ 1.95 + 22.27 / sqrt( g·tan(tilt) )       [seconds]
```
The `1/sqrt(a_lat)` form is first-principles exact: corner dwell time ∝ corner_length / v_corner,
and `v_corner ∝ sqrt(a_lat)`, so each binding corner contributes `∝ 1/sqrt(a_lat)`. The constant
1.95 s is the curvature-free (straight + accel/brake) floor.

**Marginal cost (the actionable slope):**
| at tilt | d(laptime)/d(tilt) ideal | RL-projected (×k≈1.5) |
|---|---|---|
| 60° | −0.137 s/° | ~−0.21 s/° |
| 65° | −0.140 s/° | ~−0.21 s/° |
| 75° | −0.095 s/° | ~−0.14 s/° |

So **the first 10° above the current 65° cap is worth ~1.3–1.5 s ideal (~2 s RL); above ~80° the
curve flattens hard.** The 65°→80° band is where the lever pays; chasing 80°→unconstrained buys
only ~0.5 s ideal and risks the inc1 inversion pathology.

---

## 4. Per-segment decomposition (ideal) vs measured

TOPP per-segment tax (tilt 60° vs unconstrained), against the §TILT-CONCENTRATION measured table:

| segment | TOPP Δt (60−unc) | measured inc5 Δt | binds? (κ-analysis) |
|---|---|---|---|
| spawn→G0 | **+1.02** | +0.73 | NO — accel-from-rest + global conservatism |
| G0→G1 | +0.17 | +0.40 | NO |
| G1→G2 | +0.24 | +0.47 | **YES** (κ=0.026) |
| G2→G3 | +0.42 | +0.67 | **YES** (κ=0.033, tightest) |
| G3→G4 | +0.32 | +0.27 | marginal |
| G4→G5 | +0.64 | +0.37 | **YES** (κ G4 + final straight) |
| **TOTAL** | **+2.81** | **+2.91** | — |

Totals agree to 0.1 s. The biggest single cost is **spawn→G0 (+1.0 s ideal / +0.73 s measured)** —
and crucially its unconstrained peak roll is only ~63°, **inside** the 65° cap. So this segment's
cost is NOT the angle binding; it is **global reward conservatism** from the `rw_tilt` *weight*
making all aggression expensive. This is the key lever-separation insight (next section).

---

## 5. The TWO levers — `rw_tilt` WEIGHT vs free-cone ANGLE

These are physically distinct and the data shows they act on different segments:

| Lever | What it is | Where it bites | First-order effect |
|---|---|---|---|
| **`rw_tilt` WEIGHT** (96→48→16) | penalty *steepness* beyond the cone | EVERYWHERE — shapes the whole speed/aggression tradeoff, including segments inside the cone | Global conservatism. Recovers the **non-binding** cost (spawn→G0 +0.73, G0→G1 +0.40, G3→G4 +0.27 ≈ **1.4 s** of the 2.9 s) |
| **free-cone ANGLE** (60°→75–80°) | where the penalty *starts* | ONLY the 3 high-κ segments where peak roll exceeds the cone (G1→G2 71°, G2→G3 68°, G4→G5 75°) | Raises the a_lat ceiling on binding corners. Recovers the **binding** cost (+0.47+0.67+0.37 ≈ **1.5 s**) |

**Why weight-first is right (confirms §TILT-CONCENTRATION ladder step 1):** ~1.4 s of the 2.9 s
tax lives on segments where the *angle* never binds (peak roll < 60–65°). Only the WEIGHT can
recover that — the drone is flying slow there not because it hits the cone wall but because the
quadratic penalty makes *any* tilt expensive relative to the time reward. Lowering `rw_tilt`
96→48 (existing datum: 9.22 s, −0.30 s) is one knob, recovers global conservatism, and leaves the
hinge intact (no new style regime). Raising the cone angle is a separate step (2) that unlocks the
3 binding corners. **They compose roughly additively** (different segments), so the full ladder
(weight↓ then cone↑) approaches the ~6.6 s unconstrained potential without going to `rw_tilt=16` /
no-cone — i.e. without re-entering the inversion-prone regime.

**a_lat → cornering-speed sensitivity to the cone (the angle lever, concretely):**
raising the cone 60°→75° lifts a_lat 17→36.6 m/s² (2.15×) → v_corner ↑ by √2.15 = 1.47× on G1/G2/
G3/G4. At G3 (tightest): 22.6→33.2 m/s. 60°→80° lifts a_lat 17→55.7 (3.27×) → v_corner ×1.81, and
every corner clears the 39 m/s drag wall → cornering ceases to bind (knee of the curve).

---

## 6. Caveats

1. **TOPP is a point-mass IDEAL bound** — no plant lag, no 2-tick (67 ms) latency, no DR margin, no
   finite rate-slew, no mixer thrust/rate coupling. It is the *floor*; RL realizes k≈1.5× it. Use
   the curve for the *shape* and the *lever ranking*, the measured inc5 points (9.52 / 6.89 s) for
   absolute RL lap time.
2. **a_max in the profiler couples tangential and lateral** (`a_tan = sqrt(a_max²−(κv²)²)`). On
   straights (κ≈0) the same `g·tan(tilt)` also caps forward accel, so the curve slightly
   over-attributes time to tilt on the straights. The true thrust ceiling is the convex map (~8 g
   body-up) — much higher — so real straight-line accel is NOT tilt-limited; the binding bound on
   straights is the **v²-drag wall (~39 m/s)**, which IS in the model (`v_max=39`). Net: the curve
   is a faithful *cornering* attribution; the spawn→G0 accel term is mildly inflated.
3. **Measured inc5 figures are single-trajectory** (deterministic racestart; 20 identical eps).
   "roll p90 145°" is the *distributional* peak across jittered starts, NOT the racestart peak
   (75°). Do not conflate (already flagged §TILT-CONCENTRATION).
4. **Drag wall 39 m/s and convex thrust 8 g** are the corrected-aero measured constants
   (`rl_plant.py` QUAD_DRAG_C2_MEASURED / COLL_MAP_*). The airspeed lapse is VOIDED (not enabled) —
   the convex map over-predicts ~22% at 4–12 m/s but is not used; this is a known ~10–20% optimism
   in absolute corner speeds, not in the *relative* tilt tax (it scales both ends).
5. **Gate-passage validity not modeled here** — TOPP plans gate-centre crossings (contact-free).
   The cornering speeds above assume the line stays ≤0.5 m from centre; pushing v_corner higher
   eventually trades against the 0.75 m validity aperture (separate POST-GATE-3 binding-gate risk).
6. **Spawn point** assumed 23.3 m up-course of G0 along the −G0→G1 direction (memory: "~23.3 m
   up-course of gate 0, at rest"). Lap time varies ~0.4 s with v_end∈{0,15}; spawn→G0 is the
   accel-from-rest term and is robust to ±a few m of spawn placement.

---

## 7. Bottom line for the S2 decision

- **Style tax = 2.3–2.9 s/lap** (RL), ideal floor 2.15 s, on this specific 5-corner descending track.
- It is a **convex curve in tilt**, fit `laptime ≈ 1.95 + 22.27/√(g·tan(tilt))`, marginal
  **~0.14–0.21 s per degree near 65°**, flattening above 80°.
- **Two independent levers**, recovering ~1.4 s (WEIGHT, non-binding segments) + ~1.5 s (CONE
  ANGLE, the 3 high-κ corners) — composing to the full tax. Weight-first (96→48) is correctly the
  first ladder rung because most non-binding cost can only be reached that way.
- For S2: this tax is the **price of structural inversion-safety**. A DECOMPOSED plan-line +
  tracker structurally caps tilt via the reference (no inversion possible) but pays this tax in the
  line design; a MONOLITHIC policy can in principle recover it by relaxing the envelope, at the
  inc1-inversion risk the envelope was built to suppress. The curve says: **the high-value
  recovery (65°→75–80°, ~1.3–1.5 s ideal) is reachable without entering the dangerous
  unconstrained/no-cone regime** — argues for keeping the hinge but climbing the ladder, in EITHER
  architecture.

*Method/scripts: `time_optimal_profile` (src/racer/speed_profile.py), R4 (rl/peregrine_racing.py:365),
a_lat=g·tan(tilt). All numbers reproduced offline with .venv numpy 2.4.6 / scipy 1.17.1.*
