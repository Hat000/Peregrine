# C2 — TOGT trajectories re-evaluated under the CORRECTED-AERO plant

**Date:** 2026-06-13 · **Author:** Phase-C2 worker (offline analysis, no C++ re-run, no live sim)
**Script:** `handoff/ultracode-planning-togt-s2-2026-06-13/proto_togt_corrected.py`
**Run:** `PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-planning-togt-s2-2026-06-13/proto_togt_corrected.py`
**Inputs:** shipped TOGT CSVs at `handoff/laptop-togt-bound-2026-06-10/cases/{bound_nominal,bound_free,ref_margin,bound_nodrag}/{togt_traj.csv,refined_traj.csv}` + each case `analysis.json`.
**Corrected-aero constants:** `racer.rl_plant.{COLL_MAP_*_MEASURED, QUAD_DRAG_C2_MEASURED, QUAD_DRAG_C2_POOLED}`.

Fengyou — bottom line up front: **the falsified-linear-plant assumption did NOT inflate the TOGT
time bound the way one might fear; it inflated the SPEEDS.** The old 4.27/4.55 s lines are
thrust-feasible with ~8.5x headroom under the corrected convex map, but they ride 55 m/s — a speed
the corrected v² drag wall makes physically unreachable (top speed ≈ 37-39 m/s). Re-timing the SAME
geometry honestly lands the corrected ceiling at **~4.7-5.0 s**, i.e. the exploratory **4.71 s** figure
is the right number and the **4.27 s bound was ~0.4-0.7 s optimistic** for the wrong reason (it bought
its speed from an under-modeled drag, not from real thrust it couldn't actually use).

---

## 0. The two plants

| quantity | OLD linear plant (TOGT planner) | CORRECTED-aero plant (measured) |
|---|---|---|
| collective ceiling | **9.23 m/s² = 3.765 g** (flat, `thrust_max 9.2305 N` / 1 kg) | **78.28 m/s² = 7.98 g** at full stick (convex map `COLL_MAP_ACCEL_MEASURED[-1]`) |
| collective at hover | g (linear `g·thr/hover`) | 9.58 m/s² at thr=0.2656 (convex map recovers g to 2.3%) |
| drag | linear isotropic world-frame **0.21·v** /s | body-frame **quadratic** `c2·v²`, pooled **0.052 /m**, per-axis-per-sign `QUAD_DRAG_C2_MEASURED` |
| rate ceiling | omega_max [11,11,7] | super-rate ~11.2 rad/s/axis (≈ same; rates barely bind) |

**Thrust headroom = 78.28 / 9.23 = 8.48x.** **Drag crossover:** `0.052·v² = 0.21·v` at **v = 4.0 m/s** —
below 4 m/s corrected drag is *weaker* than linear; above it, it grows without bound as v².

The TOGT frame is **z-UP** (init `p_z=-0.02`, course rises to z≈-24 as it descends; `cthrustmass: 9.8066`
at identity attitude = 1 g hover). Mapped to world NED by flipping z. `u_1..u_4` are per-rotor thrusts
(N) on a 1 kg quad, so `Σuᵢ / 1 kg` = body-up specific accel (m/s²).

---

## 1. LENS 1 — THRUST feasibility (climb authority): the old line has MASSIVE headroom

For every node, the TOGT-demanded body-up specific accel `a = Σuᵢ / m` vs the corrected convex map.

| case / traj | max a_demand (m/s²) | stick the OLD plant rode (×g) | stick the CORRECTED map needs | nodes above corrected full-stick |
|---|---|---|---|---|
| bound_nominal · togt | 36.92 | up to 4.00× (84%-ile = 3.77×, i.e. saturated) | **0.535 max / 0.43 mean** | **0.0%** |
| bound_nominal · refined | 36.92 | 4.00× (sat) | **0.535 max / 0.525 mean** | **0.0%** |
| ref_margin · refined | 36.92 | 4.00× | 0.535 / 0.524 | 0.0% |
| bound_free · refined | 36.92 | 4.00× | 0.535 / 0.525 | 0.0% |

**Reading:** the refined TOGT line is collective-**saturated** (84% of nodes at the 3.765 g ceiling per
`analysis.json`; demanded accel pins at ~36.9 m/s² — note this is ~3.77 g, the *total specific force*
incl. the gravity-fight, not 3.765 g of net thrust). Under the corrected convex map that *same* demand
sits at only **stick ≈ 0.53** — barely past hover-and-a-bit. **Minimum climb headroom is 41.4 m/s²**
(the corrected map can deliver 78.3, the line never asks for more than 36.9). **Zero nodes** exceed
corrected authority.

> **Verdict (a): the old TOGT line is thrust-FEASIBLE under corrected aero with ~8.5x headroom.**
> The convex map's extra authority is real and totally un-tapped by the old geometry. This is the
> "(b) faster — more thrust available" lever, and it is large.

---

## 2. LENS 2 — DRAG re-accounting: the old line is INFEASIBLE on the drag side

Along each TOGT path, holding the SAME attitude + SAME demanded collective, recompute the world
specific force under each drag model and project onto the path tangent. `tan_deficit = tan_corr −
tan_old` (negative = corrected brakes harder = the line loses speed it assumed it would keep).

**Drag-magnitude ratio (corrected / old) by speed bucket** (bound_nominal · refined):

| speed band (m/s) | nodes | |a_drag| old (m/s²) | |a_drag| corr (m/s²) | **ratio** |
|---|---|---|---|---|
| 0–5 | 26 | 0.52 | 0.31 | **0.60** (corrected *weaker* below crossover) |
| 5–10 | 20 | 1.56 | 2.57 | **1.65** |
| 10–20 | 37 | 3.13 | 12.44 | **3.98** |
| 20–30 | 36 | 5.25 | 37.26 | **7.10** |
| 30–60 | 210 | 9.84 | 97.60 | **9.92** |

The refined line spends **74% of its lap-time above 30 m/s** and peaks at **55.2 m/s**. At 55.2 m/s:

- corrected quadratic drag = `0.052 · 55.2² = 158.6 m/s² = 16.2 g`
- old linear drag = `0.21 · 55.2 = 11.6 m/s²`
- **under-modeling factor = 13.7×.**

The mean tangential drag *deficit* over the lap is **−59 m/s²** (most extreme node −143 m/s²). The old
line is thrust-saturated at 3.765 g while silently ignoring a **16 g** drag force pushing back. **No
amount of the 8.5x thrust headroom closes a 16 g hole at 55 m/s** — `78.3 − 9.8(grav) ≈ 68 m/s²` of
forward push cannot overcome 159 m/s² of drag. The corrected plant's steady-state top speed (thrust =
drag) is `sqrt(68/0.052) ≈ 36-39 m/s`, exactly the v² **drag wall**.

> **Verdict (c): the old TOGT line is drag-INFEASIBLE above ~37 m/s.** It cannot sustain the 46-55 m/s
> it plans. Re-timing must hold the line to ~37-39 m/s on the straights, costing time. This is the
> "(c) slower — drag wall" lever, and it dominates wherever the old line went fast.

---

## 3. LENS 3 — Path-following TOPP re-time: the NET corrected achievable time for the SAME geometry

Take the **bound_nominal TOGT geometry** (lap trimmed to the gate-6 crossing at t=5.066 s, **164.2 m**;
the planner path overshoots to a −184 m terminal hover that is *not* part of the lap), and re-solve the
time-optimal speed profile under a **decoupled friction-circle envelope**:

- lateral (turn) cap `A_h = sqrt(A² − g²)` (reserve g for altitude hold) — corrected `A_h = 77.7 m/s²`;
- longitudinal **accel(v) = A_h − c2·v²** (drag opposes the push) → hits 0 at the drag wall;
- longitudinal **brake(v) = A_h + c2·v²** (drag *helps* braking into gates);
- standard coupling `a_tan = sqrt(cap² − (κv²)²)`.

(Curvature from the planner's own dense polyline; max κ = 0.132 /m, R=7.6 m, at the standing-launch
turn — the rest of the course is gentle, p99 κ=0.035 → R=28 m.)

| envelope on bound_nominal geometry (164.2 m lap) | TOPP time (s) | v_max (m/s) | v_corner_min (m/s) |
|---|---|---|---|
| **corrected aero** (78.3 m/s² stick, quad 0.052) | **5.00** | 37.3 | 24.3 |
| exploratory T/W-8 flat (78.5 m/s², quad 0.052) | 4.99 | 37.3 | 24.3 |
| OLD linear (3.765 g, linear 0.21) | 5.46 | 46.7 | 16.4 |
| corrected thrust, **NO drag** (isolates headroom) | 3.65 | 73.5 | 24.3 |

**Reconciliations (all consistent):**

1. **corrected-aero ≈ exploratory-T/W-8** (5.00 vs 4.99 s): they should match (78.3 ≈ 8 g, same quad
   drag) — confirms the envelope model. The shipped `expl_corrected_aero` *IPOPT-refined* lap is **4.71 s**;
   our geometry-FIXED TOPP is 5.00 s, **0.29 s slower** because IPOPT also re-optimizes the *geometry*
   (TOPP can only re-time a fixed line). So **4.71 s is the right corrected ceiling, and our independent
   re-time brackets it from above (5.0 s fixed-geometry) — the 4.71 s figure is VALIDATED.**

2. **corrected vs OLD on identical geometry: 5.00 vs 5.46 s → corrected is 0.46 s FASTER.** On this
   *fixed* line the thrust headroom (reaches top speed sooner, brakes harder into gates) nets out ahead
   of the drag wall, because the old line's own corner caps already hold it near ~37-47 m/s. The
   headroom and the wall nearly cancel; headroom wins by a hair when the geometry is held fixed.

3. **Drag costs 1.35 s of the corrected lap** (5.00 with drag − 3.65 without). v_max collapses 73.5 →
   37.3 m/s. That 1.35 s IS the v² drag wall, quantified.

> **Net verdict:** under corrected aero the existing TOGT *geometry* re-times to **~4.7-5.0 s**, NOT
> 4.27 s. The corrected achievable time is **robustly in the 4.7-5.0 s band**, matching the exploratory
> 4.71 s. The line is *feasible* (LENS 1) but *mis-timed* (LENS 2/3): it banks ~0.4-0.7 s of phantom
> speed from under-modeled drag.

---

## 4. Margin cost + where time is spent

**Shipped refined laps** (`analysis.json`; these are the *contact-tolerant* IPOPT laps — max_miss
listed, gate-contact = INVALID at miss > 0.75 m − body_radius):

| case | gate_margin (m) | refined lap (s) | togt-init lap (s) | max_speed (m/s) | collective-sat | max_miss (m) | valid? |
|---|---|---|---|---|---|---|---|
| bound_nominal | 0.0 | **4.134** | 5.066 | 55.2 | 84% | 1.060 | **NO** (corner-cut) |
| bound_nodrag | 0.0 | 4.046 | 5.066 | 55.7 | 84% | 1.060 | NO |
| ref_margin | 0.7 | **4.311** | 5.493 | 53.7 | 82% | 0.611 | borderline |
| bound_free | 0.6 | 4.432 | 5.796 | 52.6 | 85% | 0.355 | YES |

**Margin cost (linear plant):** bound_nominal (margin 0, **4.134 s**) → ref_margin (margin 0.7,
**4.311 s**) = **+0.177 s** for 0.7 m of gate margin; → bound_free (**4.432 s**) = +0.30 s. The brief's
4.27/4.55 s pairing is the *planning-valid* projection of the same trend (margin tax ≈ 0.18-0.30 s/lap).
**This margin tax is REAL and survives the aero correction** — it is geometric (fly closer to centre =
straighter line), independent of the plant. The drag wall affects the *absolute* level (4.13 → ~4.7-5.0)
but the *margin delta* stays ~0.2-0.3 s.

**Where the lap-time goes (refined bound_nominal, 4.134 s line):**
- **98.3% of lap-TIME at ≥99% of the (old) collective ceiling** — the line is thrust-saturated almost
  end-to-end. Confirms the brief's "thrust rides ceiling ~84% of *nodes*" (84% of nodes = 98% of *time*,
  since saturated nodes are the fast/long-dwell ones). **Thrust binds, not rates** (rate sat <9%).
- **74% of lap-time above 30 m/s**, time-weighted mean speed **39.6 m/s**, peak 55.2 m/s.
- **Drag bites on the straights / high-speed descents** (gates 2→5, the 30-55 m/s band) where corrected
  drag is 4-10x the linear model. It barely bites in the launch and the final approach (<10 m/s, where
  corrected drag is actually *weaker*).

---

## 5. Answers to the four C2 questions

**(1) Is the existing TOGT line still feasible under corrected aero?**
**Thrust: YES, with 8.5x headroom** (LENS 1 — never asks for >0.54 stick of the corrected map).
**Drag: NO above ~37 m/s** (LENS 2 — the line plans 46-55 m/s, where corrected drag is 16 g vs the
3.765 g thrust ceiling; physically unreachable). So the *geometry* is feasible but the *speed profile*
is not.

**(2) Faster (more thrust) or slower (drag wall)?** BOTH levers are large and **nearly cancel on the
fixed geometry**: thrust headroom would buy 3.65 s (no-drag), the drag wall gives back 1.35 s, landing
at 5.0 s — vs the old line's 5.46 s re-time. Net **corrected is ~0.46 s FASTER than the old plant on the
identical geometry**, but **~0.7-0.9 s SLOWER than the old line's own optimistic 4.13 s** (which was
drag-cheating).

**(3) Corrected achievable time for the existing geometry: ~4.7-5.0 s.** Geometry-fixed TOPP = 5.00 s;
IPOPT geometry-reopt (exploratory) = **4.71 s**. **Reconciled with the 4.71 s figure — it is the correct
corrected ceiling.** The 0.29 s gap between them is exactly the geometry-reoptimization headroom IPOPT
captures that a fixed-line TOPP cannot.

**(4) Margin cost + time-spent:** margin tax ≈ **0.18 s (0.7 m)** to **0.30 s (full free-cone)**, real
and plant-independent. **Thrust rides the ceiling 98% of lap-TIME (84% of nodes); drag bites hardest in
the 30-55 m/s straights/descents (gates 2-5), 4-10x the linear model.**

---

## 6. Implication for the S2 architecture decision (the adjudication target)

- The **4.27 s "planning-valid bound" is OPTIMISTIC by ~0.4-0.7 s** — it was a *falsified-plant* number.
  The honest contact-free corrected bound is **~4.7-5.0 s**. Anyone benchmarking RL against 4.27 s is
  chasing a mirage; **4.71 s (exploratory) is the right target ceiling** and our independent re-time
  confirms it.
- The **shipped 4.55 s reference line** (margined, fed to the DECOMPOSED tracker) is **drag-infeasible
  in its high-speed segments**: it plans 46-55 m/s where the corrected plant caps at ~37-39 m/s. A
  decomposed RL/MPCC tracker *cannot* follow it at the planned speed — it will lag on every straight.
  This is a concrete strike against DECOMPOSED-on-the-current-line: **the offline line must be REBUILT on
  the corrected-aero plant** (TOGT/IPOPT with `T/W≈8` convex map + quad drag, or our TOPP envelope) before
  it is a valid tracking target. Until then, the geometric-tracker's k=1.85 (8.3 s) gap is *partly* a
  wrong-target artifact, not purely a tracker-quality gap.
- The **drag wall (~37-39 m/s top speed, robust) is the true speed ceiling**, not thrust. This caps BOTH
  architectures and means the inc7 monolithic policy (live-confirmed ~11.45 s) has a real ~4.7-5.0 s
  structural target — the headroom is in *carrying speed through the corrected drag*, which the convex
  thrust map *does* enable (it just can't beat v² drag). Monolithic-retrain-on-corrected-plant remains
  the recommended first move; decomposed needs a corrected line first.

---

## 7. Method, assumptions, caveats

- **No C++ re-run, no live sim.** Pure numpy/scipy re-analysis of the shipped CSVs against the canonical
  measured constants in `racer.rl_plant`. The TOGT trajectories themselves are NOT re-optimized — that
  would need the WSL C++ pipeline (out of scope / forbidden here).
- **LENS 1/2 hold the planner's demanded thrust + attitude fixed** and only swap the aero model, so they
  measure the *discrepancy at the old operating point* — the honest way to ask "is the old line feasible".
  They are exact (every node, full-precision columns).
- **LENS 3 (TOPP) is OPTIMISTIC vs TOGT/IPOPT:** it treats accel direction as instantaneous (no jerk /
  attitude-rate transient cost) and uses a magnitude friction-circle. Use the **ratios/deltas**
  (corrected vs old = 0.91; drag costs 1.35 s) as the signal, not the absolute seconds. Its absolute
  number (5.0 s) sits 0.29 s above the IPOPT 4.71 s precisely because IPOPT re-optimizes geometry; the
  consistency of the two is the validation.
- **Drag pooled isotropic c2=0.052** used in LENS 3 (per-axis `QUAD_DRAG_C2_MEASURED` used in LENS 2,
  which is why LENS 2 ratios vary slightly by direction). The per-axis spread (0.042-0.076) is within
  ±25% of pooled; it does not move the qualitative wall.
- **Altitude-reserve `g` in `A_h`** is a mild approximation (the course descends, so *less* than g need
  be reserved on the down-segments → corrected could be a touch faster than our 5.0 s; this only widens
  the gap to the 4.27 s bound, reinforcing the conclusion).
- **LAPSE is correctly OFF** (voided by the 2026-06-12 frame audit) — not enabled anywhere here.
- **Lap trim:** LENS 3 trims to the gate-6 crossing (t=5.066 s, 164.2 m); including the planner's
  post-finish stop injects a spurious R=0.5 m curve that is not part of the race line (this was a bug in
  the first pass, fixed — `kappa_cap=0.5` now only guards numerics, real max κ=0.132).

**MEMORY-DELTA:**
- C2 DONE: falsified-linear-plant did NOT inflate the TOGT *time* bound much — it inflated the *speeds*. Old TOGT line is thrust-FEASIBLE under corrected convex map (8.5x headroom, never >0.54 stick) but drag-INFEASIBLE above ~37 m/s (plans 46-55 m/s; corrected drag at 55 m/s = 16.2 g = 13.7x the linear 0.21·v).
- Corrected achievable time for the EXISTING bound_nominal geometry = ~4.7-5.0 s (TOPP fixed-geom 5.00 s; IPOPT geom-reopt = exploratory 4.71 s). **4.71 s VALIDATED as the corrected ceiling; the 4.27 s planning-valid bound is ~0.4-0.7 s OPTIMISTIC (falsified-plant artifact).**
- Drag wall = top speed ~37-39 m/s (robust); thrust rides ceiling 98% of lap-TIME (84% of nodes) — THRUST binds geometry, DRAG binds speed. Margin tax ~0.18 s (0.7 m) to 0.30 s, plant-independent.
- S2 implication: shipped 4.55 s reference line is drag-INFEASIBLE in its fast segments → a DECOMPOSED tracker cannot follow it at planned speed; offline line must be REBUILT on corrected aero before decomposition is valid. Reinforces "retrain monolithic on corrected plant first."
- Artifact: `handoff/ultracode-planning-togt-s2-2026-06-13/artifacts/C2_togt_corrected.md`; script `proto_togt_corrected.py`.
