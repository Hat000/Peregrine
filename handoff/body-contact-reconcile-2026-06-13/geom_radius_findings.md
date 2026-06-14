# Tilt-projected geometry radius — FINDINGS (2026-06-13)

Fengyou — this is the tilt-projected-geometry leg of the body-radius re-derivation.
Script: `geom_radius.py`; result: `geom_radius_result.json`. Flags: **[M]** measured/confirmed
input, **[X]** derived here, **[A]** assumed.

## Question
Quantify the LEGITIMATE part of the 0.38 m contact radius: how big is the 280×280×160 mm
chassis silhouette when tilted to the gate-4 approach posture (pitch −38°, roll/crab 55°)
and projected onto the gate plane (⊥ the g3→g4 approach), accounting for the 0.30 m frame tube?

## Result (all [X], from confirmed [M] inputs)
| quantity | value | note |
|---|---|---|
| flat half-side (L-inf, level) | **0.140 m** | strict lower bound |
| flat half-diagonal (Euclid, level) | **0.198 m** | strict lower bound |
| **gate-4 tilted L-inf, gate-aligned** | **0.213 m** | ← **effective radius (matches code's L-inf metric)** |
| gate-4 tilted Euclid disk radius | 0.214 m | rotation-invariant disk |
| gate-4 tilted L-inf, spin-min square | 0.159 m | basis-free min (NOT the code's basis) |
| pitch-only (−38°) L-inf | 0.149 m | diagnostic |
| roll-only (+55°) L-inf | 0.161 m | diagnostic |

Cross-checked by an independent full 360° yaw brute force: max in-plane Euclid = 0.2135 m,
max L-inf (fixed gate basis) = 0.2127 m, 3D corner norm = 0.2135 m. The in-plane Euclid
EQUALS the 3D corner norm → at this posture the worst corner lies essentially in the gate
plane (its along-normal component ≈ 0), so the silhouette captures the corner's full extent.
This is the true geometric maximum; nothing is hidden by the in-plane basis choice. Triangulates
with the provenance leg's independent 0.213 m. **[X]**

## Frame depth (0.30 m)
Adds **nothing** beyond the single-plane silhouette. A convex body at CONSTANT attitude
translating along the gate normal sweeps a **prism**; its cross-section is the same silhouette
in every plane ⊥ the normal, so "clear the full tube" = clear the one silhouette. (Constant
attitude through the crossing is **[A]** — at gate-4 the drag-hold posture is held, first-order
valid; an attitude change mid-tube could only grow it, but the gate-4 contacts that drove the
0.38 fit are steep *descents*, not in-tube tumbles.) **[X]/[A]**

## Props
Spec documents **NO** prop/rotor/motor-span extent anywhere (provenance.md §3) — chassis-only
is the primary result. The script supports a labeled prop what-if (`PROP_TIP_FULL_SPAN_M`); left
`None`. The codebase's own "rotor halo ~0.3 m" is an internal empirical inference, not spec
geometry. **[A]**

## Comparison to 0.38 and bottom line
- Tilt inflates the flat **0.198 m** half-diagonal to only **0.213 m** — **+0.015 m**. The tilt
  argument is REAL but SMALL.
- Effective rigid-body radius **0.213 m = 0.56 × 0.38**. The 0.38 budget is **0.167 m above**
  anything rigid-body geometry can justify at this posture.
- **0.38 is NOT a geometric body size.** It is an empirically-fit steep-crash halo (live crashes
  L-inf 0.37–0.49, corner-pass contact 0.60). The unmodeled ~0.17 m is rotor-wash / prop-disk /
  blade-strike envelope the spec does not document. Distrust 0.38 as *geometry*; do NOT treat
  0.213 as the *true* contact radius either — the steep-descent contacts are real and
  posture-matched to gate-4. **[X]**

## Margin implication (margin = (0.75 − r) − 0.215 at the radius-invariant gate-4 simstart linf)
geom 0.213 → **+0.322 m**; 0.30 → +0.235; 0.33 → +0.205; 0.38 → +0.155. Positive at every r;
the "does NOT close offline" headline is load-bearing **only** on choosing the steep-crash-fit
0.38 and is about the p90 estimator-error tail, not the nominal pass.
