# Phase C — Adversarial Verification, lens = PLANT HONESTY + DETERMINISM

**Date:** 2026-06-13  **Reviewer seat:** Phase-C adversary (plant_honesty lens)
**Fengyou** — this is the honest adjudication of the two prototypes' achievable-time claims.
Verdict in one line: **the core correction is UPHELD; the two headline lap-time numbers are
PARTIALLY UPHELD as point-mass PLANNING BOUNDS but must not be quoted as flight times, and one
cross-check anchor (the 4.714 s corrected-TOGT) is contact-INVALID at any body radius.**

---

## 0. What I verified (method)

- Re-ran both prototypes with the repo venv (`PYTHONPATH=src .venv/Scripts/python.exe`); both run
  clean and reproduce their headline numbers exactly (C1 5.348 s @60°, 4.574 s @90°; C2 4.996 s
  geometry-fixed re-time).
- Independently re-derived every load-bearing physical constant from `src/racer/rl_plant.py`
  (the canonical measured plant), not from the prototype outputs.
- Independently re-loaded the shipped TOGT CSVs + `analysis.json` to check the cross-check anchors.
- Grepped both scripts for silent use of the falsified linear plant or the VOIDED airspeed lapse.

---

## 1. Plant-honesty of the constants — UPHELD

Independently recomputed from `rl_plant.py`:

| Quantity | Prototype claim | My independent value | OK? |
|---|---|---|---|
| Full-stick body-up accel `COLL_MAP_ACCEL_MEASURED[-1]` | 78.283 m/s² (~8 g) | 78.2828 m/s² = 7.983 g | ✅ |
| Hover knot recovers g | ~9.58 (−2.3%) | 9.5804, −2.3% | ✅ |
| Pooled quad drag `QUAD_DRAG_C2_POOLED` | 0.052 /m | 0.052 | ✅ |
| Drag @9 m/s vs old linear 0.21·v | ~2.2× | 4.212 vs 1.890 = **2.23×** | ✅ |
| Drag crossover (corrected==old) | ~4 m/s | 0.21/0.052 = **4.04 m/s** | ✅ |
| v² drag wall (pooled, g-reserve) | ~38–39 m/s | **38.65 m/s** (pure-fwd 38.80) | ✅ |
| `a_lat = g·tan(tilt)`: 60/65/75/80° | 17/21/37/55.7 | 16.99/21.03/36.60/55.62 | ✅ |
| Thrust headroom vs old 3.765 g | ~8.5× | 78.28 / 9.2305 = **8.48×** | ✅ |

**No silent use of falsified constants.** Grep confirms:
- C1 twin probe sets `linear_drag=0.0`, `quad_drag_c2=QUAD_DRAG_C2_MEASURED`, `coll_map_*=…MEASURED`.
- The VOIDED airspeed lapse (`LAPSE_*`) is **not imported or referenced** in either script. Correct.
- The only appearance of the falsified linear constants (3.765 g, 0.21 /s) is in **C2's explicit
  old-vs-corrected comparison** (`retime_old`, `TOGT_TW`, `TOGT_LINEAR_DRAG`) — i.e. as the baseline
  being refuted, never as the active corrected budget. Honest.

**Determinism / offline reproducibility — UPHELD.** Both are pure numpy/scipy (`CubicSpline`,
finite-difference TOPP), no RNG, no live sim, no network. Re-running is bit-stable. Legal under the
"offline line-iteration across attempts is LEGAL" doctrine.

---

## 2. The v² drag wall — UPHELD (and prototypes are if anything CONSERVATIVE)

The central physical correction is the v² drag wall capping the doubled thrust. Verified:
- Pooled wall ≈ **38.6–39.2 m/s**; matches C1 v_max 39.17 and the C++ corrected-TOGT max_speed
  **39.263 m/s** (independently re-read from `expl_corrected_aero/refined_traj.csv`) to <0.3%.
- The old falsified line plans **55.2 m/s** (`bound_nominal/refined`, re-read independently); at that
  speed corrected quad drag = 158.4 m/s² (16.2 g) vs old linear 0.21·55.2 = 11.6 m/s² → **13.7×**
  under-modeling. C2's "13.7×" is exact. The 55 m/s segments are flatly drag-infeasible.

**Directional nuance (in the prototypes' favor):** the dominant forward axis is nose-first (+x),
whose measured c2 = **0.042 < pooled 0.052**. So the pooled drag wall (38.6 m/s) is *conservative*
for forward flight — the true nose-first wall is ~43 m/s. Both prototypes therefore **under-claim**
top straight speed, not over-claim it. The climb axis (−z) is 0.076 (drags harder) but the course
DESCENDS, so the relevant vertical axis is +z (0.0539 ≈ pooled). No optimism injected by the drag
model. The C2 LENS-3 full-g altitude reserve on a descending course is likewise *conservative*
(gravity helps the descent). Net: pooled isotropic drag is a slightly-pessimistic, safe choice.

---

## 3. The two headline lap times — PARTIALLY UPHELD (planning bounds, not flight times)

Both numbers are SOUND **point-mass planning bounds on the corrected plant**, but neither is a
tracked / contact-valid flight time, and C1's own outputs expose two optimism sources:

### C1 — 4.574 s (90° unconstrained) / 5.348 s (60° style-respecting)
- **Centre-line geometry is contact-free by construction** (per-gate miss = 0.0 at r=0.38; max
  spline-to-gate deviation 0.0125 m). The *geometric* contact-free claim is genuine.
- **BUT the speed plan is only thrust-feasible for tilt caps ≥75°.** At 60° the integrator demands
  up to **111.6 m/s² = 11.4 g** specific force (2.2% of body samples over the 78.3 ceiling); at 65°,
  101.4 m/s² (1.4% over). So **5.348 s (60°) and 5.076 s (65°) are MILD LOWER BOUNDS** — true
  achievable is ~+0.1–0.2 s slower at those caps (corner-entry braking exceeds the thrust ball).
  C1 flags this as a limitation; I confirm the magnitude and direction.
- 75/80/90° are clean (max req ≤77.8 ≤ 78.3 ceiling). 4.574 s (90°) is a valid open-loop-feasible
  point-mass bound, **mildly optimistic** because (a) attitude is instantaneous (TOGT multiple-
  shooting is ~0.14 s slower from rate transients) and (b) the convex map is a ~0-airspeed fit that
  over-predicts thrust ~22% at 4–12 m/s (lapse correctly NOT used, so the ceiling is the 0-airspeed
  value). Honest corrected figure: **~4.6–4.7 s unconstrained, ~5.4–5.6 s at a true 60° cap.**

### C2 — 4.996 s (geometry-fixed corrected TOPP)
- Re-times the EXISTING `bound_nominal` geometry under the corrected envelope. 4.996 s vs IPOPT's
  geometry-reoptimized 4.714 s: the 0.28 s gap is exactly the reopt headroom a fixed-line TOPP
  cannot capture. Internally consistent. The corrected-no-drag 3.651 s and old-linear 5.465 s
  bracket it sensibly (drag wall costs 1.34 s; thrust headroom buys 0.47 s on fixed geometry).
- C2's reported `lap_time_s: 4.996` and `contact_free: false` are honest: those `per_gate_miss_m`
  (1.046…0.807) are the **shipped margin-0 `bound_nominal` "bound" line**, which exceeds 0.75 m by
  construction. C2 correctly states the margin-bearing cases (ref_margin 0.611, bound_free 0.355)
  are the valid lines. No misrepresentation.

---

## 4. ⚠️ The cross-check anchor (4.714 s corrected-TOGT) is CONTACT-INVALID — REFUTE as a *valid* number

This is the one place a reader could be misled. Both prototypes anchor on the C++
`expl_corrected_aero` refined lap = **4.714 s** as "the corrected-aero ceiling." I re-read its
`analysis.json` crossings:

```
per-gate miss (m): [0.749, 0.766, 0.713, 0.471, 0.701, 0.476]   max = 0.766 > 0.75
meta: gate_shape='ball', gate_margin_m=0.0, thrust_to_weight=8.0, quad_drag=0.052
```

- At **r=0 (point mass)** gate-2 already misses 0.766 > 0.75 → **INVALID even as a point mass.**
- At the **inc7-trained r=0.38**, the contact-valid band is <0.37 m → **ALL SIX gates violated.**

So **4.714 s is a margin-0, contact-tolerant ball-gate PLANNING bound, NOT a rules-valid lap.**
Under the binding doctrine (gate contact = INVALID run), it cannot be quoted as an achievable valid
time. It is fine as a *ceiling cross-check between two planning bounds* — which is how C1 uses it —
but the writeups should never let "4.71 s corrected ceiling" be read as a flyable valid lap. The
T/W=8 flat (78.45) vs convex full-stick (78.28) differ only 0.22%, so the anchor's thrust budget is
sound; the problem is purely its sub-validity gate clearance.

---

## 5. Honest corrected figures (the numbers to actually carry)

| Regime | Honest corrected-aero figure | Status |
|---|---|---|
| Falsified-linear bound (4.27 s) / shipped line (4.55 s) | **REFUTED** — phantom speed from 13.7× under-modeled drag | both prototypes agree |
| Unconstrained (≥80°) corrected **planning** bound | **~4.6–4.7 s** (C1 4.574 + ~0.1 s rate/airspeed optimism; C++ 4.714) | planning bound, contact-INVALID anchor |
| 60° style-respecting corrected **planning** bound | **~5.4–5.6 s** (C1 5.348 is a mild lower bound; +0.1–0.2 s for the 2.2% thrust leak) | planning bound, centre-line |
| Geometry-fixed corrected re-time of existing line | **~5.0 s** (C2 4.996); IPOPT reopt 4.714 | planning bound |
| **Fastest TRACKED, contact-VALID, corrected FLIGHT time** | **NONE DEMONSTRATED** | gap to inc7 9.76 s twin is real & structural |

**The structural-gap thesis survives.** Even taking the most pessimistic honest reading (~5.4 s
style-respecting planning bound, true valid flight slower still), the corrected prize is ~1.8–2×
faster than inc7's 9.76 s twin median. The gap is real; it is just smaller and slower than the
falsified 4.27 s suggested. A capable tracker (learn-the-line RL / MPCC) is REQUIRED to realize any
of these bounds — C1's naive feedforward+PD twin probe diverges 37.9 m, which is itself the
"a good tracker is required" signal, not a usable closed-loop time.

---

## 6. Determinism caveats (none fatal)

- Both prototypes use the **pooled** isotropic drag in the speed budget while the measured table is
  per-axis/per-sign (0.042–0.076). The spread is ±25% of pooled and, as shown in §2, biases the
  forward wall *conservatively*. A per-axis budget would shift the wall up slightly on straights —
  second-order, and in the safe direction.
- C1's 60/65° laps are reproducible but are lower bounds (thrust leak). The 75/80/90° laps are clean.
- C2 LENS-3 treats accel direction as instantaneous and reserves a full g for altitude (conservative
  on a descending course). Use C2's ratios/deltas as the signal, not the absolute 4.996 s to 3 d.p.

---

## 7. Bottom line for the S2 decision

- **Do NOT decompose against the shipped 4.55 s reference line** — it is drag-infeasible in its
  46–55 m/s segments. Any DECOMPOSED target must be a line REBUILT on the corrected plant first.
- The corrected ceiling (~4.6–5.6 s class depending on tilt envelope, all PLANNING bounds) is robust
  and well-cross-validated, but no contact-valid tracked time exists yet on either prototype.
- Reinforces the prior reviewer read: **retrain monolithic on the corrected plant first; decompose
  only as a fallback** — and rebuild the offline line on corrected aero before treating it as a
  decomposition target.
