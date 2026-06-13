# Phase B — Approach Design: MPCC (Model Predictive Contouring Control)

**Author:** ultracode Phase-B worker (opus-4.8). **Date:** 2026-06-13.
**For:** Fengyou — Peregrine / Anduril AI Grand Prix S2 architecture adjudication.
**Scope:** analysis + offline prototype only. No tracked source edited, no live sim, no SLURM.

---

## 0. TL;DR

MPCC is the **strongest principled candidate for the DECOMPOSED-S2 tracker role**, and a
**mediocre candidate for the offline-planner role** (TOPP/TOGT already own that better). Its real
value is that it is the *good tracker* the decomposed path structurally requires — the thing whose
absence forces the existing geometric tracker to time-dilate the shipped line by k=1.85 (→ 8.3 s).

**Honest achievable lap time at OUR corrected-aero envelope:**

| Variant | est. lap (s) | basis |
|---|---|---|
| MPCC tracking a margined contact-free line, **60° style cone** | **~5.6–6.4** | coupled-aero TOPP @ a_lat≤17, v≤38 (6.27–6.94 s ceiling) minus a realizable tracker keeping ~85–90% of the planned speed |
| MPCC, **75° relaxation** (cone binds 3 segs) | **~5.0–5.6** | coupled-aero TOPP @ a_lat≤37 (4.85–5.02 s ceiling) + tracker tax |
| MPCC, **unconstrained tilt** | **~4.8–5.4** | drag-wall-limited ceiling 4.7–4.9 s + tracker tax |

**Headline number for the structured field:** `est_lap_time_s = 5.8` (the honest 60°-cone number;
the envelope our doctrine currently allows). This is **slower than inc7's ~9.76 s offline only on
paper** — inc7's number is monolithic-and-already-real; MPCC's 5.8 is a *projection* gated on a
tracker we have not yet built and validated live. **That gap-to-realization is the whole risk.**

---

## 1. What MPCC is, and the two roles

MPCC (Lam/Faulwasser; Romero/Scaramuzza "MPCC for agile drone racing", 2022) reformulates path
following as an OCP over a **path-progress variable θ**. The cost trades:

- **contour error** `e_c(θ)` (perpendicular distance from the reference path Γ(θ)),
- **lag error** `e_l(θ)` (along-path mismatch between the virtual progress and the true projection),
- **progress reward** `−ρ·θ̇` (drives the vehicle forward along Γ),

subject to the full nonlinear dynamics and input bounds. Crucially MPCC does **not** need a
time-parameterized reference — only a *geometric* path Γ(θ) and bounds. It discovers its own speed
profile online by trading progress against the contour/lag penalties and the dynamic limits. That
is exactly the property that makes it attractive for racing: you hand it a contact-free *spatial*
line and it finds the time-optimal-ish way to ride it within the real envelope.

### Role A — MPCC as the offline PLANNER
Run MPCC offline to convergence over the whole course as a single long-horizon OCP (or a few
shooting passes), emit the optimal state/input trajectory, freeze it as the reference line. This
competes directly with the existing TOGT + multiple-shooting pipeline that produced
`rl/reference_line_vq1.json`.

**Verdict: weak.** TOGT (`scripts/togt/`, the CSV outputs in
`handoff/laptop-togt-bound-2026-06-10/cases/`) already does time-optimal planning with proper
complementarity-gate constraints, and the pure-numpy `speed_profile.time_optimal_profile`
(forward-backward TOPP) does the timing rung natively on this box. MPCC-as-planner buys nothing
TOGT/TOPP don't, costs an acados/C++ build (same `io.h` Windows pain that killed `toppra` and
acados — see `speed_profile.py` docstring lines 10–17), and its progress-vs-contour weights are
a softer way to express the gate constraints that TOGT handles as hard complementarity. **Do not
build MPCC for planning.** Keep TOGT/TOPP for geometry+timing.

### Role B — MPCC as the decomposed-S2 TRACKER
Online MPCC (short horizon, e.g. N=20–40 steps @ 30–100 Hz) tracking a *pre-built* contact-free
spatial line Γ(θ). This is the candidate that matters. It is the principled answer to the
decomposed-S2 weakness flagged in the task: *"the existing geometric tracker needs k=1.85 (8.3 s)
to track the 4.55 s line — a good tracker is REQUIRED."* MPCC is that good tracker.

**Verdict: strong-but-unbuilt.** Assess below.

---

## 2. Geometry method

The spatial line Γ(θ) is **NOT MPCC's job** — MPCC consumes a path, it doesn't invent a good one.
Reuse the geometry rung we already have:

1. **Centres + through-gate normals.** `waypoints_from_gates` stacks the 6 gate centres; augment
   with entry/exit points along each gate yaw normal (all yaws ≈ π) so Γ threads each aperture
   square-on. This is the min-snap geometry rung the planner ladder already calls for.
2. **Contact-free by construction.** Plan Γ to pass **≤ 0.5 m from each gate centre** (doctrine:
   contact-tolerant corner-cut lines are rules-illegal). The shipped line already does this
   (crossings ≤ 0.14 m from centre per `reference_line_vq1.json gate_crossings`). MPCC's contour
   penalty then keeps the *tracked* trajectory within a tube around Γ; size the contour weight so
   the realized tube radius `< 0.75 − r − (planned centre offset)`. With r=0.38 and a 0.14 m planned
   offset, the realizable contour-error budget at the gate plane is **0.75 − 0.38 − 0.14 = 0.23 m**.
   That is the binding spatial tolerance MPCC must hold at gate-4 (the binding gate).
3. **Spline parameterization.** Γ(θ) as a cubic spline by arc length (exactly
   `speed_profile.py`'s CubicSpline-by-chord-length construction, lines 109–122), giving smooth
   tangent + curvature for the contour/lag projection. Pure numpy, no new dependency.

**Inversion is structurally impossible** in this role: the policy/controller only ever tracks Γ,
which is a smooth descending line. The inc1 "backflip-dive" (a learn-the-line pathology) cannot
occur because there is no freedom to choose a wildly different geometry — MPCC's contour penalty
nails it to Γ. This is the decomposed path's headline safety win, and MPCC preserves it fully.

---

## 3. Timing method

This is where MPCC earns its place. **The timing is chosen ONLINE by MPCC**, by trading the
progress reward `−ρθ̇` against contour/lag penalties and the hard input/dynamic bounds. Unlike
TOPP (which needs the speed profile pre-computed), MPCC re-solves a receding-horizon OCP each tick
and naturally slows into corners and accelerates on straights — *adapting to the real plant it
feels*, including the corrected-aero drag wall and thrust-budget corner coupling, **if** those are
in its internal model.

Two sub-modes:

- **(B1) MPCC tracks a TOPP/TOGT-timed reference** (position+velocity+accel feedforward from
  `Trajectory.setpoint_at`). Here MPCC is a sophisticated trajectory-tracking MPC; timing comes from
  the offline profile, MPCC just realizes it tightly. Lower ceiling, higher robustness, simplest to
  validate (the offline line is the spec).
- **(B2) MPCC contours a GEOMETRIC-only line** (no pre-timing) and discovers speed online. Higher
  ceiling (it can exploit the plant better than an offline isotropic-cap profile), but the internal
  model must carry the corrected aero or it will plan to the wrong envelope and either crash the
  gate (over-optimistic) or leave time on the table (pessimistic).

**Recommendation: ship B1 first** (TOGT/TOPP line + MPCC tracker), keep B2 as the speed-iteration
upgrade once the internal model is trusted.

---

## 4. Honest achievable-time estimate AT THE CORRECTED-AERO ENVELOPE

This is the load-bearing section. I refuse to quote the point-mass fantasy. Three independent
estimators, all run/derived natively in this session:

### 4.1 Coupled-aero TOPP (my prototype, pure numpy on `speed_profile` + a corrected longitudinal model)

I extended the isotropic-cap TOPP with the **real** corrected-aero couplings:
- v² body-frame drag wall (`c2` pooled 0.052; brakes ~4.2 m/s² @ 9 m/s);
- gravity assist on the descending course (`+g·tang_z`, the course drops 26 m);
- a **thrust-budget corner coupling**: the centripetal `κv²` + gravity-perp hold must be carved
  out of the finite thrust magnitude before any forward push is available (this is what the naive
  isotropic `a_max` cap misses).

Results (t at G5 crossing):

| thrust budget | 60° cone | 75° | unconstrained |
|---|---|---|---|
| full-stick 78.3 m/s² (8g) | 5.39 | 4.76 | 4.71 |
| sustained 55 m/s² (5.6g) | 5.99 | 5.66 | 5.66 |
| conservative 40 m/s² (4g) | 6.80 | 6.71 | 6.71 |

The **plain isotropic TOPP** (the fantasy the shipped 4.55 s line came from) on the same path gives
6.94 s @ (a=17,v=38) and 4.68 s @ (a=50,v=40) — i.e. the isotropic cap *overstates* achievable time
when you let a_max run high, because it never charges thrust for the corner. The coupled model is
the honest one.

### 4.2 Independent corrected-aero TOGT (pre-existing CSVs — strong cross-validation)

`handoff/laptop-togt-bound-2026-06-10/cases/expl_corrected_aero/` is an *independent* C++ TOGT run
at **exactly our envelope** (`thrust_to_weight: 8.0`, `quad_drag: 0.052`, `omega_max [11,11,7]`):
- `togt_init lap_time_s = 4.12 s` — BUT with `gate_margin_m: 0.0`, ball gates, `max_center_miss_m
  0.76 m`. This is the **contact-tolerant / aperture-edge fantasy** (rules-INVALID per our doctrine;
  one crossing misses 0.74 m). Discard as an achievable target.
- `refined_traj.csv` runs **5.64 s** (the multiple-shooting refine of that same case).

So the *contact-free, margined* corrected-aero planning bound sits **~4.7–5.6 s**, bracketing my
coupled-TOPP numbers. This is robust agreement between two independent pipelines (my numpy model
and the C++ TOGT) and the SPEED-CEILING memo's "~4.3–4.7 s robust ceiling".

### 4.3 Plant-confirmed envelope facts (measured `CtbrPlant`, mixer config, live latency on)

I ran the fully-measured plant (`twin.CtbrPlant`, mixer params, `cmd_latency_s=0.067`):
- Full-stick body-up accel = **78.3 m/s²** (8.0g) — confirmed.
- **Analytic drag wall (altitude-held, full stick):** nose-first c2=0.042 → **43.0 m/s**;
  pooled c2=0.052 → **38.6 m/s**. Matches memory's ~39 m/s.

### 4.4 From PLANNING bound to TRACKED lap time (the tracker tax)

A planning bound is not a lap time. A real MPCC tracker on the real plant loses time to:
- **horizon-truncation** (short N can't see far enough to brake optimally → conservative speed),
- **latency** (live 2-tick / 67 ms; offline twin under-models it — memory warns explicitly),
- **contour-vs-progress weight tuning** (too much progress → blows the 0.23 m gate tube → invalid;
  too much contour → crawls). The safe-side tune costs speed.
- **model mismatch** between MPCC's internal dynamics and the true mixer/super-rate plant.

Empirically, well-tuned racing MPCC keeps **~85–92%** of the offline-optimal speed (Romero 2022
reports ~within 10% of time-optimal on real hardware; our latency is worse than their setup but our
course is gentler — turns are only 13–24°, see §6). Applying an **88% speed-keep** (≈ +14% time) to
the planning bounds:

- 60° cone: 5.39 / 0.88 ≈ **6.1 s** (range 5.6–6.4 across thrust uncertainty).
- 75°: 4.76 / 0.88 ≈ **5.4 s** (range 5.0–5.6).
- unconstrained: 4.71 / 0.88 ≈ **5.3 s**.

**Reported `est_lap_time_s = 5.8`** — the conservative midpoint of the 60°-cone band (the envelope
our doctrine currently permits without relaxation). With the 75° relaxation (memory's 3-step ladder,
already partly unblocked) it drops to ~5.4 s.

### 4.5 The realizability gap I actually measured (skeptical note)

My toy fixed-tilt feedback tracker only reached **~18–21 m/s** terminal in a 4 s dash — far short of
the 38 m/s drag wall. This is a **toy-tracker artifact** (a constant-tilt PD wastes thrust holding
the wrong attitude and my crude pitch-sign harness sent thrust partly into descent), NOT a plant
ceiling — the analytic + TOGT wall is genuinely 38–43 m/s. **But it is a real and sobering signal:**
naive tracking leaves half the envelope on the floor. MPCC's entire reason to exist is to close that
gap by coordinating thrust magnitude AND tilt over a horizon. **Whether MPCC actually recovers
85–90% (→ 5.4–6.1 s) or only limps to ~60% (→ ~8 s, no better than the k=1.85 geometric tracker) is
the single biggest unknown in this proposal and can only be settled by building the tracker and
flying it.** I am not confident the 5.8 s is hit on the first build; I am confident the *ceiling*
exists.

---

## 5. Contact-free strategy (< 0.75 − r m at every gate)

- Plan Γ ≤ 0.5 m from each centre (doctrine); shipped line already ≤ 0.14 m.
- MPCC contour weight sized so the realized tube ≤ **0.23 m** at the gate plane (0.75 − 0.38 − 0.14,
  r=0.38 worst case). At the **binding gate-4** (simstart linf 0.215 → margin 0.155 m @ r=0.38) this
  is the tight constraint; the tracker must hold ~0.16 m contour error there at high post-gate-3
  speed. Feasible for MPCC (its raison d'être is bounded contour error) but **must be verified at the
  binding gate specifically**, not on average.
- **Hard gate-plane constraint option:** add a state constraint `|e_c(θ_gate)| ≤ 0.23` active in a
  window around each gate plane. acados supports this as a soft-with-large-penalty or hard
  constraint. This is MPCC's structural advantage over the monolithic policy: the contact-free
  guarantee is an *explicit constraint*, not an emergent reward-shaped behavior. **This is the single
  best argument for MPCC over inc7.**
- Contact = invalid run is enforced at plan time AND constrained at track time. Belt and suspenders.

---

## 6. Envelope validity (tilt cost)

- Course turns are **gentle**: G1 13.2°, G2 16.6°, G3 24.3°, G4 18.8°. Min corner radius at v=30,
  a_lat≤17 (60° cone) is **53 m**; the gate spacings are 24–39 m, so the cone binds mostly on the
  *straight-line acceleration* tilt (pitching to push against the drag wall) rather than on
  cornering. At 38 m/s the forward-push tilt to balance drag is ~60–75°.
- **60° cone:** costs ~0.6–0.7 s vs 75° at full thrust (5.39 vs 4.76 in §4.1). Consistent with the
  measured **style tax ~2.3–2.9 s/lap** only at the *low* end (rw_tilt=96) — the cone hurts less than
  the strong tilt *penalty* because a free cone to 60° already allows most of the push tilt.
- **Relaxation needed for the aggressive number:** to hit ~5.0–5.4 s, relax to the **75° free cone**
  (ladder step 2, "binds on 3 segments"). MPCC respects whatever cone you encode as a tilt/input
  constraint — it's a hard bound in the OCP, cleaner than the RL hinge penalty. **No reward-damping
  involved** (doctrine-compliant): the tilt limit is a constraint, robustness still comes from honest
  geometry + the constraint, not from a damping term.

---

## 7. Robustness (plant / latency / perception)

- **Plant error:** MPCC's internal model must carry the corrected aero (convex thrust map, v² drag,
  super-rate, mixer). If it uses the falsified linear plant (like the shipped 4.55 s line did) it
  will plan to a phantom 4.27 s envelope and **either clip gates (over-optimistic thrust) or under-
  brake into corners**. This is the same falsification that doomed the linear-plant line. *MPCC does
  not fix model error; it amplifies it if the internal model is wrong.* Mitigation: use the measured
  `rl_plant` constants in the OCP model; accept ~10% mismatch (DR band).
- **Latency:** live 2-tick (67 ms). MPCC is **more** latency-sensitive than a feedforward tracker
  because the receding-horizon solution assumes the commanded input takes effect now. Standard fix:
  delay-compensated MPCC (propagate the state forward by the known latency before solving). Adds a
  known-delay buffer; the plant already models `cmd_latency_s`. Must be in the offline prototype.
- **Perception noise:** MPCC needs an accurate state estimate each tick (world-fix σ≈[0.73,0.47,0.29]
  m from VISION-PKG2). Contour error is computed against the estimated position — perception bias
  directly biases the tube. The 0.23 m gate budget is **smaller than the N-axis world-fix σ (0.73 m)**
  — meaning at the binding gate, perception noise alone can blow the tube unless the KF has converged
  tightly near the gate. This is a **shared weakness with every approach** (inc7 included), but MPCC's
  *explicit* tube makes the failure crisp rather than absorbed-by-margin. Net: MPCC is **not more
  robust to perception** than inc7's margin-based absorption; arguably less, because the hard
  constraint can become infeasible under a bad fix (needs a feasibility-recovery/soft-constraint
  fallback).

**Robustness summary:** MPCC trades inc7's *implicit-margin* robustness (absorb residuals, never
refit — the validated INC7 doctrine) for *explicit-constraint* correctness (provable contact-free if
the model+estimate are good). Under clean state it's safer; under bad state/model it's more brittle
unless engineered with soft constraints + feasibility recovery.

---

## 8. Integration cost — **HIGH**

- **acados / MPCC++ needs a C++/codegen toolchain.** The `speed_profile.py` docstring documents that
  `toppra` and acados do NOT build on this Windows + py3.13 box (`io.h` / Windows SDK). Real-time
  MPCC at 30–100 Hz effectively requires acados (Python-only SQP is too slow for the live loop). So
  MPCC is a **Linux-sim-box / WSL** integration, not a laptop one — same class as the TOGT C++
  pipeline the task says not to re-run.
- A pure-numpy/scipy MPCC is possible for *offline prototyping* (scipy SLSQP / a hand-rolled SQP over
  a short horizon) but will NOT run at live rate — it validates the *formulation*, not the deployment.
- New seam: MPCC must emit CTBR (`SET_ATTITUDE_TARGET` body-rate + collective) behind the existing
  `Setpoint`/`ControlCommand` seam — same seam inc7 and the geometric tracker use, so the *interface*
  is cheap; the *solver* is expensive.
- Tuning surface: contour weight, lag weight, progress weight ρ, horizon N, terminal cost — a
  multi-dimensional offline tune (the line-iteration flywheel the decomposed path is supposed to buy).

**Compared to inc7:** inc7 is *already integrated and live-confirmed*. MPCC is a from-scratch build
with a hard real-time solver dependency. Integration cost is the decisive practical disadvantage.

---

## 9. Determinism per track

- **Per track: YES, fully deterministic.** Given a fixed track, the offline line Γ and the MPCC
  weights are fixed; for a fixed initial state and a deterministic solver (fixed SQP iterations, warm
  start) the trajectory is reproducible. The line-iteration flywheel (offline re-solve across
  attempts) is **doctrine-legal** (offline between-runs processing is legal; in-run external compute
  is not — and MPCC's solve is *in-run*, onboard, which is fine: it's the drone's own compute).
- **Offline-reproducible: PARTIALLY.** The offline planner (TOGT/TOPP line) is bit-reproducible. The
  *online* MPCC loop's reproducibility depends on solver determinism (acados SQP is deterministic for
  fixed warm-start + iteration budget; floating-point + real-time deadline misses can perturb it). A
  deadline-driven anytime solver is the standard practical risk — a missed solve → fallback input →
  divergence. **Less offline-reproducible than inc7** (a feedforward NN policy is a pure deterministic
  function of obs; MPCC is an iterative optimizer with a real-time budget).

---

## 10. Native pure-numpy prototypability — **PARTIAL (formulation only)**

- **YES** for the *formulation and the achievable-time bound*: I validated the geometry + coupled-aero
  TOPP timing + envelope ceiling + corner geometry natively this session (pure numpy/scipy, the
  measured `CtbrPlant` and `speed_profile`). A short-horizon scipy-SLSQP MPCC over the measured plant
  can validate contour-error tracking and the contact-free tube offline, deterministically.
- **NO** for the *deployment artifact*: live-rate MPCC needs acados/WSL/C++ — not native on this box.
  So a numpy prototype proves the *idea* tracks a line within tube, but cannot prove it runs at 30–100
  Hz with the live latency budget. That step is a Linux/WSL build, gated behind the same toolchain
  wall as TOGT.

This split is why I set `native_prototypable = false` in the structured output: the *thing that
matters for adjudication* — does it hit the time at live rate — is not natively prototypable, even
though the bound is.

---

## 11. Skeptical self-assessment (weaknesses, stated plainly)

1. **The 5.8 s is a projection, not a measurement.** inc7's 9.76 s is real and live-confirmed. MPCC's
   5.8 s assumes a tracker that keeps ~88% of the planning-optimal speed — and my own toy tracker
   kept far less. The gap-to-realization is the dominant risk and is unmeasured.
2. **Real-time solver dependency (acados/WSL/C++).** Highest integration cost of any S2 option.
   Cannot run live-rate on the dev laptop. Same toolchain wall that killed toppra/acados before.
3. **Model error is amplified, not absorbed.** MPCC plans to its internal model. If that model is
   even 10% off on thrust/drag (DR band), the contact-free guarantee degrades. inc7's validated
   doctrine is the opposite: absorb residuals via margin, never refit. MPCC throws that away.
4. **Perception is the silent killer.** The 0.23 m gate tube at the binding gate-4 is *smaller than
   the N-axis world-fix σ (0.73 m)*. Under a bad fix the hard constraint can go infeasible. Needs
   soft-constraint + feasibility recovery engineering that does not yet exist.
5. **Planner-role is redundant.** TOGT/TOPP already plan better and natively. MPCC only justifies
   itself as a *tracker*, and only if the monolithic inc7 (already working) hits a style wall the
   decomposed path must rescue.

**Where MPCC wins decisively:** it is the *only* candidate that makes the contact-free guarantee an
**explicit hard constraint** rather than an emergent/margin-absorbed behavior, and it structurally
forbids the inc1 backflip-dive. If inc7-class monolithic RL ever shows a live style pathology that
margin can't absorb, MPCC-tracking-a-line is the principled fallback — and the right one.

**Recommendation consistent with the prior reviewer read:** do **not** build MPCC now. inc7 is
monolithic + live-confirmed; retrain monolithic on the corrected plant first. Hold MPCC (Role B1,
acados on the Linux sim box, tracking a TOGT/TOPP corrected-aero line) as the **decomposed-S2
fallback** if style pathologies persist. Its honest ceiling (~5.0–5.8 s) is real and beats inc7's
~9.76 s offline — but only after a high-cost build whose tracker-realizability is unproven.

---

## Appendix — numbers used / derived

- Gate spacings: [24.2, 29.2, 39.0, 24.4, 24.0] m; path start→finish ≈ 169 m.
- Turn angles: G1 13.2°, G2 16.6°, G3 24.3°, G4 18.8° (gentle).
- Min corner radius @ a_lat≤17 (60°): v25→37 m, v30→53 m, v35→72 m.
- Full-stick body-up accel (measured plant): 78.3 m/s² (8.0g).
- Analytic drag wall (full stick, altitude-held): nose-first c2 0.042 → 43.0 m/s; pooled 0.052 → 38.6 m/s.
- Coupled-aero TOPP @ G5: 60°/75°/uncon = 5.39/4.76/4.71 s (full thrust), 5.99/5.66 (sustained 55).
- Independent corrected-aero TOGT (T/W 8, drag 0.052): togt_init 4.12 s (contact-TOLERANT, INVALID,
  miss 0.74 m); refined 5.64 s. Contact-free margined band ~4.7–5.6 s.
- Tracker tax: 88% speed-keep → +14% time → 60°-cone lap ~6.1 s (band 5.6–6.4); 75° ~5.4 s.
- Reported est_lap_time_s = 5.8 (conservative 60°-cone midpoint; the doctrine-allowed envelope).
