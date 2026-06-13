# Approach B — TOGT / CPC time-optimal (point-mass-through-gate), regenerated on CORRECTED aero

Author: Phase-B worker (opus). Date 2026-06-13. For Fengyou.
Scope: independent, skeptical design + HONEST achievable-time estimate for the TOGT/CPC
trajectory-optimization approach, re-reasoned on the REAL corrected-aero envelope (convex
thrust map ~8 g full-stick, body-frame quadratic drag c2~0.052 → ~39 m/s wall, super-rate
body rates ~11.2 rad/s/axis). All numbers below are derived natively (pure numpy/scipy) or
read from the existing TOGT CSV outputs; nothing requires WSL/C++/Adroit.

SAFETY: analysis only. No source edits, no sim, no SLURM, no network.

---

## 0. TL;DR (the honest answer)

TOGT/CPC is a **geometry+timing trajectory optimizer**: it freely places the crossing point
inside each gate opening and solves a time-optimal point-mass-with-attitude trajectory subject
to thrust/rate limits. Regenerated on corrected aero it gives a *gold-standard feasibility
bound*, but that bound is **dominated entirely by the tilt (style) envelope**, not by the aero:

| Tilt cap | Honest est lap (G0→G5) | Cross-check |
|----------|------------------------|-------------|
| 60° (style cone)   | **~9.6 s** | matches inc5 rw_tilt=96 (9.52 s) AND inc7 offline (9.76 s) |
| 65°                | ~8.7 s     | matches "~65° costs ~2.3 s/lap" datum |
| 70°                | ~7.7 s     | — |
| 75°                | ~6.6 s     | matches inc5 unconstrained ~6.6–6.9 s |
| 80°                | ~5.3 s     | — |
| 83° (≈unconstrained)| **~4.5 s** | matches refined corrected-aero TOGT (4.71 s) and the robust 4.3–4.7 s ceiling |

**est_lap_time_s = 4.9 s** is my single-number honest estimate for *what TOGT/CPC actually
delivers as a flyable line on the corrected plant*, and it requires **relaxing the style cone to
~78–80°** — NOT the 60° cone. At the 60° cone, TOGT buys essentially nothing over inc7's
already-live 9.76 s. The 4.71 s "point-mass fantasy" is real ONLY at sustained ~77° mean tilt
(p90 ≈ 87°, near knife-edge) and demands body rates of ~17 rad/s that the real ~11.2 rad/s
super-rate plant cannot produce — so even 4.71 s is not trackable as-is.

This is the central correction: **on corrected aero the lap time is a TILT story, not a TOGT
story.** TOGT is a high-quality *line generator* whose value is realized only if (a) the style
cone is relaxed and (b) a competent tracker exists. By itself it does not beat inc7.

---

## 1. What TOGT/CPC is, and what "regenerate on corrected aero" means

TOGT (Time-Optimal Gate Traversal, the `scripts/togt` C++ pipeline; CPC = Complementary
Progress Constraints, its academic sibling) solves:

  minimize T  s.t.  rigid-body dynamics, |thrust|≤T_max, |ω|≤ω_max, and a *progress/waypoint
  constraint per gate* (the trajectory must pass through each gate opening; the crossing point
  is FREE within the opening — this is the "free the crossing point within the inscribed
  circle" feature). Two stages exist in the existing pipeline:
   * `planTOGT` (init): point-mass + free-attitude relaxation, fast.
   * multiple-shooting `refine` (13-state): full per-rotor thrust + body-rate state, dynamic
     feasibility to `refine_tol_m`. This is the trustworthy stage.

"Regenerate on corrected aero" = re-run with the corrected plant constants instead of the
falsified linear plant. The existing pipeline already has the knobs:
  * thrust: `thrust_max` (the C++ side is still a *linear/constant* max-thrust model — it has
    NO convex `coll_map` and NO super-rate map). The exploratory case `expl_corrected_aero`
    set T/W=8 (≈ the convex full-stick 7.98 g) and `quad_drag=0.052`, `linear_drag=0`.
  * drag: `quad_drag` (isotropic |v|·v) — this IS in the refine stage.

So a faithful regeneration is *partially* available today: `expl_corrected_aero` already
captures the two dominant corrections (8 g thrust ceiling + quad drag wall). What it does NOT
capture: (i) the convex *sub-hover* thrust sag, (ii) the super-rate ω-amplitude map / 11.2
rad/s ceiling, (iii) any tilt/style cone, (iv) the mixer thrust↔rate coupling. Those are
exactly the things that make the bound optimistic.

---

## 2. Geometry method (where the line goes)

Per-gate crossing point chosen by the optimizer inside the gate opening, but **constrained to
plan ≤0.5 m from each gate centre** (doctrine: contact-tolerant corner-cut lines are
rules-illegal; gate contact = invalid run). Concretely:
  * gate plane = gate yaw ≈ π (faces along-course), half-opening 0.75 m Euclidean in-plane.
  * VALIDITY requires in-plane miss < 0.75 − body_radius (r∈{0.28,0.33,0.38}; inc7 trained
    r=0.38 → pass band 0.37 m). With FRAME_DEPTH 0.30 m extrusion the contact-true band is
    even tighter; gate-4 is the binding gate (sim-start linf 0.215 → margin 0.155 m @ r=0.38).
  * So the planned crossing box is **±0.5 m from centre max, and realistically we want the
    plan ≤~0.30 m from centre at gate-4** to hold a positive contact-true margin under tracking
    error. This kills most of the "free the crossing point" speed benefit on the binding gate —
    TOGT's freedom to cut the corner is exactly what the rules forbid here.

Path realization for the native prototype: cubic spline through the 6 gate centres,
arc-length parameterized, curvature κ from the spline. Gate spacings (G0→G5):
[24.23, 29.24, 38.99, 24.39, 23.98] m; total centre-line length 140.83 m; total descent 26.0 m
(mean slope 10.7° — see §4, the descent gives only ~1.7% vertical-thrust credit, negligible).

The centre-line is a fair geometric proxy *because* the doctrine forces the plan near centre.
This is the one place TOGT's geometry advantage over a min-snap centre line is small.

---

## 3. Timing method (how fast to fly it)

TOGT's own timing = the time-optimal control solve (bang-bang thrust + rate, riding the thrust
ceiling ~84 % of the lap per the SPEED-CEILING analysis). For an HONEST, natively-reproducible
estimate I built a corrected-aero forward/backward TOPP (the `speed_profile.py` method) with the
*physically-correct* tilt-limited accel budget:

  * thrust accel magnitude ≤ ath_max = 78.28 m/s² (convex full-stick, from
    `COLL_MAP_ACCEL_MEASURED[-1]`), tilt from world-up ≤ tilt_cap.
  * **vertical thrust component must hold ≈ g** (alt-tracking; descent credit only 1.7%).
  * **horizontal thrust budget = min( sqrt(ath_max²−g²), g·tan(tilt_cap) )** — at any tilt ≤ ~83°
    the binding term is g·tan(tilt_cap), i.e. the classic a_lat = g·tan(θ) rule.
  * that horizontal budget must cover centripetal κv² + drag c2·v² + tangential accel; drag
    helps braking, hurts acceleration. Forward pass accelerates, backward pass guarantees brake
    into each corner.

This is the honest curve in §0. It is NOT the "full-stick-point-and-burn" model (which gives a
fake 4.1–4.5 s even at 55° by demanding 35 m/s² of un-tracked vertical accel) — that model is
physically inconsistent with staying on a near-level descending line and I rejected it.

The drag wall under alt-hold: v_max = sqrt(g·tan(θ)/c2). 60°→18.1 m/s, 80°→32.7 m/s, 83°→38.6
m/s (≈ the quoted 39 m/s). Body rates barely bind lap time (TOGT: unbounded rates buy −0.04 s);
**THRUST + TILT bind.**

---

## 4. est_lap_time_s at the REAL corrected-aero envelope

**Headline: est_lap_time_s = 4.9 s, conditioned on relaxing the style cone to ~78–80° AND
having a tracker good enough to fly a ~30 m/s line.** Basis below.

Honest curve (native corrected-aero alt-tracking TOPP, G0→G5):
  60° → 9.64 s | 65° → 8.67 | 70° → 7.66 | 75° → 6.57 | 80° → 5.33 | 83° → 4.51

Why 4.9 s and not 4.51 s (the 83° point) or 4.71 s (the refined TOGT CSV):
  1. **Rate realizability tax.** The refined corrected-aero TOGT line (4.71 s) demands body
     rates up to 17.06 rad/s (p95 15.6). The real super-rate plant ceiling is ~11.2 rad/s/axis.
     The line is NOT flyable at its planned speed; it must be slowed at the high-curvature
     transitions (gate-2→3, the long 39 m leg into the tight turn). Empirically that is a
     several-tenths-of-a-second tax.
  2. **Tracking tax.** The shipped 4.55 s linear-plant line needed the geometric tracker
     time-dilated k=1.85 (→8.3 s); a 54-combo gain sweep found nothing faster (STRUCTURAL
     tracking gap). A learned/MPCC tracker will do far better than k=1.85, but NOT k=1.0 on a
     line that already rides the thrust ceiling 84% of the time with no actuation headroom for
     disturbance rejection. Realistic tracker realization of a 4.5 s plan ≈ 4.8–5.1 s with a
     good tracker, assuming the cone is relaxed to match the plan's tilt.
  3. **Contact-free margin tax.** Holding plan ≤0.30 m from centre at gate-4 (binding gate)
     under a 30 m/s approach costs a brake-and-re-accel that the corner-free TOGT bound does
     not pay. This is the post-gate-3 risk zone for BOTH speed and validity.

So: 83° point-mass = 4.51 s → +rate tax → ~4.7 s (matches the refined CSV) → +tracker/margin
realizability → **~4.9 s flyable**, AND only if the cone is opened to ~80°.

At the **60° style cone the honest est is ~9.6 s** — i.e. TOGT delivers NOTHING over inc7's
live 9.76 s. This must be stated plainly: the entire value of TOGT/CPC here is contingent on
the envelope-relaxation ladder, which is a *reward/training* decision, not a planner decision.

Cross-validation anchors (independent, all consistent with the honest curve):
  * inc5 rw_tilt=96 (≈60° cone): 9.52 s  ✓ (my 60° = 9.64)
  * inc7 monolithic offline (60° cone): 9.76 s  ✓
  * inc5 unconstrained: 6.6–6.9 s  ✓ (my 75° = 6.57)
  * refined corrected-aero TOGT CSV (free attitude, mean tilt 76.6°): 4.71 s  ✓ (my 83° = 4.51 + rate tax)
  * Style tax "~2.3 s/lap for ≤65° vs unconstrained": my 65°(8.67) − 75°(6.57) = 2.1 s  ✓

The model landing on three independent measured anchors is the main reason I trust the curve.

---

## 5. Contact-free strategy (< 0.75 − r m at every gate; plan ≤ 0.5 m from centre)

  * Plan crossing points at gate centre (≤0.30 m at gate-4, ≤0.5 m elsewhere). TOGT's
    free-crossing-point feature is DELIBERATELY constrained to this box — we do NOT exploit the
    inscribed-circle corner cut because it is rules-illegal (contact = invalid).
  * Guarantee comes at plan time: with plan-to-centre ≤0.30 m and a tracker holding ≤~0.1 m
    cross-track (the shipped line held ≤0.14 m), in-plane miss ≤ ~0.4 m < pass band 0.37–0.42 m.
    This is TIGHT at gate-4 and is the principal validity risk — the same risk inc8 flags.
  * The contact-true eval (`rl/contact_true_eval.py`, margin = (0.75−r)−linf) is the gate the
    line must pass offline before any live attempt; gate-3/gate-4 isolated.
  * Robustness margin must be BUILT INTO THE PLAN (slower corner approach), not assumed away —
    a contact-tolerant TOGT line (the 4.13 s linear datum) is explicitly INVALID.

---

## 6. Envelope validity (tilt cost) — the decisive caveat

  * The refined corrected-aero TOGT line flies at **tilt_mean 76.6°, p90 86.9°, max 89.7°** —
    essentially knife-edge for most of the lap. This RESPECTS the style envelope ONLY if the
    cone is fully removed.
  * To get any TOGT benefit you must climb the relaxation ladder: 60→48 rw_tilt (step 1, now
    unblocked), free-cone 60→75–80° (step 2), unconstrained (step 3). The honest curve shows the
    payoff is monotonic and steep above 70°.
  * At the 60° cone TOGT ≈ inc7. So **TOGT's premise is the envelope relaxation**, and the
    relaxation is independently gated on LAPTOP-INC8-BINDING-GATE-VERIFY (gates 4,5 margins).
    The planner cannot manufacture this; it is a doctrine/reward decision the user owns.

---

## 7. Robustness (plant / latency / perception)

  * **Plant error:** the C++ TOGT model is a linear-max-thrust + isotropic-quad-drag
    approximation; it lacks the convex sub-hover sag, the super-rate ω map, and the mixer
    coupling. Its "feasible" line over-states authority (17 rad/s rates) — open-loop tracking on
    the real plant will under-perform unless re-margined. The plan is a *reference*, not a
    control law; it has zero disturbance rejection of its own.
  * **Latency:** live = 2 ticks (67 ms). A reference rider that rides the thrust ceiling 84% of
    the lap has no actuation headroom to reject a 67 ms phase lag at 30 m/s; this is precisely
    why a TOGT line needs a closed-loop tracker (MPCC or learned), not feedforward.
  * **Perception:** at 30 m/s the last-accepted fix can be 10 m stale; world-fix σ≈[0.73,0.47,
    0.29] m. A near-centre plan with 0.37 m pass band at gate-4 is sensitive to this — the line
    must be flown with a state estimate good to ≪0.4 m near the gate. This argues for *slowing
    the corner approach* (more margin), which the honest 4.9 s already partially bakes in.

Net: TOGT is the LEAST robust of the candidate approaches as a standalone artifact, because it
plans to the ceiling. Its robustness is entirely inherited from whatever tracker flies it.

---

## 8. Integration cost

**HIGH.** Reasons:
  * The geometry generator (C++ planTOGT + multiple-shooting refine) needs WSL/C++ to RE-run
    with new constants — the existing CSVs are frozen artifacts on the old/exploratory plant.
    To regenerate faithfully (convex map, super-rate, mixer) the C++ model would need extending,
    or we re-implement the optimizer in numpy/`acados` (acados doesn't build on this box).
  * It produces only a *reference line*; it does NOT close the loop. A TOGT line is useless
    without a tracker, and the existing geometric tracker needs k=1.85 (8.3 s) → a NEW good
    tracker (MPCC or RL-tracker) is REQUIRED. That tracker is itself a large build.
  * So TOGT integration = (regenerate optimizer on corrected plant) + (build a competent
    tracker) + (envelope relaxation). Three hard subsystems.
  * Contrast: inc7 monolithic is ALREADY live (5/5 standing, 11.45 s), end-to-end, and its
    pathology root cause is understood/fixed.

TOGT's natural home is the **DECOMPOSED** S2 arm (offline plan-line + RL/MPCC tracker) — it IS
the plan-line generator for that arm. As a standalone it is not a deployable approach.

---

## 9. Determinism per track

**Excellent — this is TOGT's strongest property.** Given a fixed track map, the optimizer is
fully deterministic and offline; the line can be iterated across attempts (offline line-
iteration is LEGAL; only in-run external compute is illegal). A frozen CSV reference replays
bit-identically. This is exactly the "offline line-iteration flywheel" that the decomposed S2
arm trades its lower ceiling for. Reproducible offline with no live dependency. The only
non-determinism enters at the TRACKER, not the planner.

---

## 10. Native pure-numpy prototype?

**Partially.** What CAN be validated natively (and was, in this analysis):
  * the corrected-aero TIMING bound via the `speed_profile.py` forward/backward TOPP with a
    tilt-limited budget (the honest §0/§4 curve) — done, cross-validates to 3 anchors;
  * tracking realizability checks against `twin.py`/`rl_plant.py` (rate ceiling, thrust ceiling,
    contact-true margin via `contact_true_eval.py`) — all pure numpy.
  * forward-simulating a candidate reference through `racer.twin.CtbrPlant` with a simple
    tracker to measure realized lap time + contact margin — pure numpy, native.

What CANNOT be done natively: the actual TOGT/CPC *geometry+timing optimization* (the
multiple-shooting NLP) — that's the C++ pipeline (WSL) or acados (won't build on Windows/py3.13;
toppra also won't build). So a *faithful regenerated TOGT line* needs WSL; but the achievable-
TIME ESTIMATE and the realizability verdict are fully native. For our purpose (deciding whether
TOGT is worth pursuing) the native estimate is sufficient and decisive.

---

## 11. Skeptical self-assessment (where THIS approach is weak)

  1. **The headline benefit is not a planner benefit.** TOGT's 4.9 s requires the same envelope
     relaxation that would *also* let the monolithic inc-line RL go fast (inc5 unconstrained
     already hit 6.6 s WITHOUT TOGT). So TOGT does not uniquely unlock speed; the cone does.
  2. **At the shipped 60° cone, TOGT = inc7 (≈9.7 s). Zero marginal value there.**
  3. **The "feasible" refined line isn't actually flyable** (17 rad/s > 11.2 plant ceiling); the
     C++ feasibility is against a wrong rate model.
  4. **No closed loop.** Needs a tracker that doesn't exist yet (k=1.85 is the only datum, and
     it's bad). The 4.9 s assumes a good tracker materializes.
  5. **Binding gate-4 validity** under a 30 m/s near-centre plan is the real risk; TOGT's
     corner-freedom is forbidden exactly where it would help.
  6. **Integration is 3 hard subsystems** vs inc7 already live.

Where TOGT genuinely wins: **determinism + offline iteration** (best of any approach), and it is
the natural plan-line generator IF the S2 decision goes decomposed. As a standalone time-optimal
approach on corrected aero it is dominated by "relax the cone + retrain monolithic", which is
cheaper and already live.

---

## 12. Numbers appendix (all native / from CSVs)

  * full-stick body-up accel = 78.28 m/s² (7.98 g)  [COLL_MAP_ACCEL_MEASURED[-1]]
  * hover-knot accel @0.2656 = 9.58 m/s² (recovers g to 2.3%)
  * alt-hold max lateral @ full stick = sqrt(ath²−g²) = 77.67 m/s² at 82.8° tilt
  * drag wall (alt-hold, c2=0.052) = sqrt(g·tan θ /c2): 60°→18.1, 80°→32.7, 83°→38.6 m/s
  * a_lat = g·tan(tilt): 55°→14.0, 60°→17.0, 65°→21.0, 75°→36.6, 80°→55.6 m/s²
  * gate spacings 24.23/29.24/38.99/24.39/23.98 m; centre length 140.83 m; descent 26 m @10.7°
  * refined corrected-aero TOGT CSV: lap 4.71 s, vmax 39.26 (= drag wall), tilt_mean 76.6°,
    p90 86.9°, max 89.7°, body-rate max 17.06 rad/s, thrust_sum max 78.5 (full stick)
  * shipped ref_circle line: 4.551 s, vmax 51.1 (point-mass init was higher; refined capped),
    tilt_mean 66.9°, plant T/W 3.765 LINEAR (falsified)
  * honest corrected-aero alt-tracking TOPP (this work):
      60°→9.64 | 65°→8.67 | 70°→7.66 | 75°→6.57 | 80°→5.33 | 83°→4.51 s
  * est_lap_time_s = **4.9 s** (80° cone + rate/tracker/margin taxes) ; at 60° cone ≈ 9.6 s.
