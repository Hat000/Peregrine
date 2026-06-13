# Phase C — Adversarial Verification (lens = CONTACT-FREE + VALIDITY)

**For Fengyou.** Adjudication of the two TOGT/S2 prototypes (C1 envelope-aware corrected-aero
TOPP; C2 corrected-aero re-evaluation of the shipped TOGT lines) under the contact-validity
lens: are the per-gate misses really `< 0.75 − r` at r=0.38? Does the smooth line pass `≤0.5 m`
from EVERY centre incl. the binding gate-4? Does any quoted time assume corner-cutting that
violates the contact rule?

Everything below was re-run / re-derived from the prototype scripts and the shipped CSVs with
`PYTHONPATH=src .venv/Scripts/python.exe`. Constants read live from `src/racer/rl_plant.py`.

---

## VERDICT: PARTIALLY UPHELD

The **corrected-aero ceiling (~4.57–4.71 s unconstrained)** and the **style-respecting
~5.35 s @ 60°** are sound *planning bounds*, contact-free **on the centre line by construction**,
and cross-validated against an independent C++ TOGT result. The central correction (the falsified
linear-plant 4.27 s bound / 4.55 s shipped line are drag-infeasible; true ceiling ~4.6–4.7 s) is
**CORRECT and well-supported**.

Three claims need qualification or refutation on the contact-validity lens:

1. **C1's 60° (5.348 s) and 65° (5.076 s) are OPTIMISTIC lower bounds, not clean feasible bounds**
   — they are open-loop INFEASIBLE (demand thrust above the 78.3 m/s² ceiling on 2.2% / 1.4% of
   the lap). True achievable at those caps is ~+0.1–0.2 s slower. C1 discloses this in limitations;
   I am promoting it because Fengyou's headline interest is the *style-respecting* number.
2. **C2's parenthetical that `ref_margin` (0.7 m margin, 4.311 s) is a "contact-valid line" is
   REFUTED at the trained body radius r=0.38** (and at r=0.33, and even r=0.28). Its refined L-inf
   misses are 0.41–0.48 m; the r=0.38 band is 0.37 m → margins are NEGATIVE (gate-3 −0.111 m,
   gate-4 −0.110 m). Contact-INVALID. The only shipped TOGT line that is contact-valid at r=0.38 is
   `bound_free` (4.432 s), and it clears gate-4 by only **+0.046 m**.
3. **Neither prototype's quoted bound is a "corner-cutting" cheat** — C1 is a literal centre-line
   plan (L-inf ≤ 0.0008 m from every centre); C2's `bound_nominal` Euclidean miss ~1.05 m is NOT
   corner-cutting *outside* the gate but the opposite — a knife-edge **margin-0** line riding the
   *inside corner of the validity box* (per-axis L-inf ≈ 0.75 m). C2 correctly flags it
   `contact_free=false`.

---

## 1. Both prototypes reproduce bit-for-bit

| prototype | reported | re-run | match |
|---|---|---|---|
| C1 lap @ 60/65/75/80/90° | 5.348 / 5.076 / 4.686 / 4.574 / 4.574 s | identical | ✓ |
| C1 per-gate miss (all caps) | [0,0,0,0,0,0] | identical | ✓ |
| C1 contact-free r∈{0.38,0.33,0.28} | True | True | ✓ |
| C1 line length / max curvature | 164.65 m / 0.0334 (R 29.97 m) | identical | ✓ |
| C2 corrected TOPP re-time | 4.996 s | 4.9956 s | ✓ |
| C2 IPOPT geometry-reopt (expl) | 4.714 s | 4.7139 s | ✓ |
| C2 shipped bound_nominal refined | 4.134 s, v_max 55.2 | identical | ✓ |

---

## 2. C1 centre line is GENUINELY contact-free — no corner-cutting (UPHELD)

Rebuilt the C1 natural-cubic spline through `[START, G0..G5]` (chord-length param, 20 000 samples)
and measured closest approach to each centre in BOTH metrics (Euclidean 3-D and the gate-frame
L-inf = max(|ΔY|,|ΔZ|) that the live validity rule actually uses, per `rl/offline_rollout.py:160`,
`rl/contact_true_eval.py`):

```
G0: eucl3D 0.0037  L-inf(Y,Z) 0.0004   margin @ r=0.38 = +0.3696
G1: eucl3D 0.0028  L-inf       0.0008   margin           = +0.3692
G2: eucl3D 0.0005  L-inf       0.0002   margin           = +0.3698
G3: eucl3D 0.0024  L-inf       0.0003   margin           = +0.3697
G4: eucl3D 0.0013  L-inf       0.0001   margin           = +0.3699   (binding gate)
G5: eucl3D 0.0000  L-inf       0.0000   margin           = +0.3700
```

- C1's per-gate miss `0.0` is honest under BOTH Euclidean and L-inf. (C1 scores Euclidean perp
  distance; the live rule is L-inf — but on a centre line both are ≈0, so the metric choice does
  not change C1's contact-free verdict.)
- The C1 bound assumes **zero** corner-cutting: it is a literal centre-line. The full +0.37 m
  contact-true margin at r=0.38 is available to a tracker at every gate, *including the binding
  gate-4*. ⇒ **claim "smooth line passes ≤0.5 m (indeed ≤0.001 m) from every centre incl. gate-4"
  = UPHELD.**

**Caveat (load-bearing):** this is a PLANNING bound on the bare centre line — NO tracker-error
offset, NO through-gate normal-entry shaping. A real tracked policy consumes margin via tracking
error; the inc7 *flown* policy's gate-4 margin is **0.155 m** (simstart, r=0.38;
`memory/project_rl_increment_history.md:635`), not 0.37 m. The 0.37 m is the planning headroom, not
a flight margin.

---

## 3. C1 feasibility caveat — 60°/65° are infeasible lower bounds (PARTIAL refutation)

Open-loop thrust feasibility per tilt cap (required specific force vs the 78.283 m/s² full-stick
ceiling), excluding the first 1 m standing ramp:

```
tilt | lap_s | frac_over_ceiling | max_req_a_up | p99_req_a_up
 60  | 5.348 |   2.17 %          |  111.6       |  89.2     <- INFEASIBLE (demands >78.3)
 65  | 5.076 |   1.43 %          |  101.4       |  81.5     <- INFEASIBLE
 75  | 4.686 |   0.00 %          |   77.8       |  66.4     <- clean
 80  | 4.574 |   0.00 %          |   70.5       |  59.7     <- clean
 90  | 4.574 |   0.00 %          |   70.5       |  59.7     <- clean
```

The over-budget samples are all corner-ENTRY braking on the curved middle (s≈68–136 m) where the
plan decelerates from a 32–38 m/s straight into a ~22.6 m/s style-capped corner faster than the
thrust ball allows. ⇒ the 60° / 65° times are **mild lower bounds**; honest achievable is
~+0.1–0.2 s slower (so style-respecting ≈ **5.4–5.5 s**, not 5.35 s). The 75/80/90° bounds are
clean and feasible. This does NOT touch contact-validity (the geometry is unchanged centre line)
but it does mean the *time* at the style-respecting cap is slightly optimistic.

---

## 4. C2 corrected-aero re-time + cross-check (UPHELD)

C2's central correction is sound and independently cross-validated:

- C1 90° unconstrained TOPP **4.574 s, v_max 39.2 m/s** ≈ independent C++ TOGT corrected-aero
  refined **4.714 s, v_max 39.26 m/s** → Δ = 0.140 s (3.0 %), v_max within 0.06 m/s. C1 is 0.14 s
  OPTIMISTIC (point-mass, instant attitude) — disclosed.
- C2 corrected-aero TOPP on the fixed bound_nominal geometry = **4.996 s**, IPOPT geometry-reopt =
  **4.714 s**; the 0.29 s gap is geometry-reoptimization headroom. Consistent.
- The shipped reference line (`bound_nominal` refined) plans **v_max 55.2 m/s** with 84% collective
  saturation; corrected quad drag walls top speed at ~37–39 m/s ⇒ the shipped 4.13 s/4.55 s line is
  **drag-INFEASIBLE in its high-speed segments**. ⇒ a DECOMPOSED tracker cannot follow it at the
  planned speed; the offline line must be REBUILT on corrected aero. **UPHELD** — and it is the
  strongest single result in either prototype.

---

## 5. THE CONTACT-VALIDITY ADJUDICATION of the shipped TOGT lines (the core of this lens)

C2 reports `per_gate_miss_m = [1.046, 1.053, 1.036, 1.06, 1.052, 0.807]` and
`contact_free=false`. **These are the `bound_nominal` (margin-0) EUCLIDEAN misses** — and they are
the WRONG line to call a "bound" without the L-inf decomposition. Re-derived from each case's
`analysis.json` (refined crossings), Euclidean vs the live L-inf metric, and scored contact-true:

| case (refined) | lap_s | gate plan-margin | L-inf per gate (G0..G5) | contact-true @ r=0.38 (band 0.37) |
|---|---|---|---|---|
| `bound_nominal` | 4.134 | 0.0 m | 0.753/0.752/0.754/**0.760**/0.754/0.682 | **INVALID** — rides/exceeds the 0.75 m edge; gate-3 plane L-inf 0.760 > 0.75 even at r=0 |
| `ref_margin`    | 4.311 | 0.7 m | 0.450/0.409/0.467/**0.481**/0.480/0.367 | **INVALID** — gate-3 margin −0.111, gate-4 −0.110 (also invalid at r=0.33 and r=0.28) |
| `bound_free`    | 4.432 | 0.6 m free-cone | 0.248/0.312/0.318/**0.322**/0.324/0.310 | **VALID** — tightest = **gate-4 +0.046 m**; gate-3 +0.052 |

Key findings:

1. **The `bound_nominal` Euclidean miss ~1.05 m is NOT a corner-cut "outside the gate."** Its per-
   axis components (miss_h ≈ ±0.72–0.75, miss_v ≈ ±0.74–0.75; `analysis.json`) put it on the
   INSIDE CORNER of the validity box — L-inf ≈ 0.75 m. It is a margin-0 *planning* bound that rides
   the validity edge; the refined line even pushes L-inf to 0.760 m (slightly OUTSIDE) on gate-4.
   ⇒ `bound_nominal` and its 4.13 s are **contact-INVALID at any body radius**. C2's
   `contact_free=false` is the correct call; the prototype is honest here.

2. **C2's limitation note that "ref_margin (0.611 m) ... ARE the contact-valid lines" is REFUTED at
   the trained radius.** `ref_margin` refined L-inf 0.41–0.48 m vs the r=0.38 band 0.37 m →
   margins are negative at gate-3 (−0.111) and gate-4 (−0.110); INVALID even at r=0.33 (band 0.42)
   and r=0.28 (band 0.47, gate-3 −0.011). Only the `togt_init` (un-refined, 5.49 s) version of
   ref_margin is valid — and that is the slow initial guess, not a competitive time.

3. **The only contact-valid shipped TOGT line at r=0.38 is `bound_free` (4.432 s)**, and it clears
   the BINDING gate-4 by only **+0.046 m** of L-inf margin — razor-thin and entirely consumed by
   any tracking error. The binding gate is **gate-4** in this independent analysis too (G-index 4),
   corroborating the simstart inc7 finding (gate-4 binding, +0.155 m flown) and the prompt.

**Net:** the genuinely contact-VALID corrected-aero TOGT time on the trained r=0.38 geometry is
**~4.43 s (bound_free)**, NOT 4.13 s (margin-0, invalid) and NOT 4.27 s (falsified-linear-plant,
invalid). The C1 centre-line planning bound (4.57 s @ 90°, 5.35→~5.4–5.5 s @ 60°) is contact-free
with full +0.37 m planning margin but is a point-mass bound, ~0.14 s optimistic and tracker-blind.

---

## 6. Honest corrected numbers (contact-validity lens)

| quantity | prototype claim | honest contact-valid figure |
|---|---|---|
| corrected-aero UNCONSTRAINED ceiling | 4.574 s (C1) / 4.714 s (TOGT) | **~4.57–4.71 s** (centre-line, contact-free; +0.37 m margin). UPHELD. |
| style-respecting (60°) bound | 5.348 s (C1) | **~5.4–5.5 s** (5.348 s is an open-loop-infeasible lower bound, +0.1–0.2 s) |
| corrected-aero CONTACT-VALID @ r=0.38 (margined TOGT line) | implied "ref_margin/bound_free are valid" (C2) | **~4.43 s** (`bound_free` only; ref_margin is INVALID at r=0.38). Gate-4 margin +0.046 m. |
| shipped 4.55 s / 4.27 s linear bound | falsified | drag-INFEASIBLE (plans 55 m/s, wall at ~37–39 m/s). REFUTED — correctly. |
| binding gate | gate-4 | gate-4 confirmed independently (bound_free +0.046 m planning; inc7 flown +0.155 m). |

**Bottom line for S2:** the corrected prize is real (~4.4–4.7 s contact-valid vs inc7's ~9.76 s
twin / ~11.45 s deployment) — a STRUCTURAL ~2× gap, not a residual. BUT every sub-4.5 s figure
floating around (4.13, 4.27, 4.55) is either drag-infeasible or contact-invalid at r=0.38. The
honest contact-valid corrected-aero target to plan a DECOMPOSED line against is ~4.4 s, and it
binds at gate-4 with near-zero margin — so any decomposed line MUST be rebuilt on corrected aero
WITH a gate-4 margin guard, and the tracker must hold tracking error well under the ~0.05 m of
spare L-inf there. This reinforces "retrain monolithic on corrected plant first; decompose as
fallback."

---

## 7. Limitations of THIS verification

- I re-ran the two prototypes and re-derived from the shipped `analysis.json` / CSVs; I did NOT
  re-run the C++ TOGT pipeline (out of scope; WSL).
- C1's twin probe diverges (37.9 m) — correctly NOT used as a time; open-loop feasibility is the
  realizability signal and it is sound for ≥75°.
- The contact-true scoring here uses the SHIPPED crossing misses (planning lines), not a flown
  policy. The inc7 flown margins (gate-4 +0.155 m) come from `memory/project_rl_increment_history.md`
  Phase-0(b) and are consistent.
- Gate-4 absolute margin is PROVISIONAL pending SHADOWPC-VISION-CAL (track_map registration);
  the binding-gate RANKING is robust, the absolute +0.046/+0.155 m figures share registration
  uncertainty.
- Pooled isotropic drag c2=0.052 used in both prototypes; per-axis climb drag (0.076 on −z) brakes
  harder on the 26 m descent — a second-order shift in the drag wall, does not move the verdict.
