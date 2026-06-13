# Phase-B Approach: MIN-SNAP geometry + TOPP-RA time-optimal speed (corrected-aero)

**Author:** ultracode Phase-B worker (independent design pass) · **Date:** 2026-06-13
**For:** Fengyou — S2 architecture adjudication (Peregrine / AI Grand Prix)
**Scope:** offline analysis + pure-numpy prototype only. Prototype:
`handoff/ultracode-planning-togt-s2-2026-06-13/artifacts/minsnap_topp_proto.py`.

---

## 0. TL;DR

A classical **min-snap line + forward-backward TOPP** is the *cheapest, most deterministic,
most natively-prototypable* of the planning approaches — it is already half-built
(`src/racer/speed_profile.py`) and validated end-to-end on the laptop with no WSL/C++/Adroit.
But its **honest achievable lap time on our real corrected-aero envelope is ~6.5–7.5 s**, and
**only if the 60° style cone is relaxed to ~80°** — because the course's 26 m descent forces the
thrust vector past 80° of tilt *for the geometry alone*, before any cornering. At the strict 60°
cone it is **~9.7 s ideal / ~11–12 s realizable — i.e. no better than inc7 (9.76 s twin / 11.45 s
live)**, and worse once tracking is added. The structural ceiling of this method is **above** the
TOGT corrected-aero collocation optimum (4.71 s) because a separable geometry-then-timing pipeline
cannot co-optimize the line shape against the speed profile the way a single collocation does.

**Verdict for the adjudication:** keep min-snap+TOPP as the **reference-line generator / fallback
and offline twin-iteration tool**, NOT as the speed-frontier method. It produces a smooth,
contact-free, deterministic plan line that an RL-tracker or MPCC can follow; but the line it
produces is *slower* than the already-shipped TOGT line (4.55 s geometry), so there is no reason
to displace TOGT for geometry. Its real value is (a) a pure-numpy, no-toolchain line you can
re-generate per-track inside the determinism-per-track flywheel, and (b) a feasible (non-saturated)
line — unlike the TOGT line which commands full-stick 99.9% of the lap and is *not self-consistent*
on the corrected plant.

---

## 1. Method

### 1.1 Geometry — min-snap through the gates
- Waypoints: pad `[0,0,0.02]` + the 6 gate centres (world NED).
- A true min-snap is the order-7 (snap-continuous) polynomial QP minimising ∫‖d⁴r/dt⁴‖²
  subject to pos pinned at each gate, v=a=0 at the ends, and C⁴ continuity at interior knots.
  Because all constraints are equalities, the QP reduces to one banded linear solve per axis.
- **Through-gate normal continuity:** all six gate yaws are ≈π (gates face along-course), so the
  along-course (−x) direction *is* the gate normal at every gate. A min-snap line that is smooth in
  x automatically crosses each gate near-perpendicular; no extra normal constraint is needed on this
  track. (On a track with off-axis gate yaws you add a pair of virtual waypoints ±δ along each gate
  normal to pin the crossing direction — standard.)
- **Prototype uses a clamped cubic** (v=0 ends) as the geometry seed, not the full order-7 QP. This
  is deliberate and *conservative*: a cubic's curvature is ≥ the min-snap curvature for the same
  knots, so the TOPP times I report are an upper bound on what min-snap proper would give. The
  curvature *distribution* (what bounds the speed) is faithfully captured. Per-gate min-radius came
  out 23–73 m — this course is geometrically gentle; min-snap would only loosen it further.

### 1.2 Timing — corrected-aero TOPP (the honest part)
`src/racer/speed_profile.time_optimal_profile` ships an isotropic-`a_max` forward-backward TOPP.
That isotropic `a_max` is the **point-mass fantasy** the brief warns against: it treats thrust as a
free omnidirectional accel budget. I replaced it with a **thrust-budget-coupled** envelope:

At each path point the body-up thrust vector (magnitude `a_up ≤ 78.3 m/s²` = full-stick convex-map
top knot) must *simultaneously*:
1. cancel gravity (`g = 9.81` along world +z),
2. supply centripetal accel `κ·v²` (normal to path),
3. supply tangential accel `a_tan` (along path),
4. overcome quadratic body-drag `c₂·v²` (`c₂ = 0.052/m` pooled) — which *helps* braking, *hurts*
   accel.

So the available **lateral** (centripetal) accel is `min( √(a_up_max² − g²), g·tan(tilt_cap) )` and
the available **tangential** accel is `min( √(a_up_max² − g² − (κv²)²), g·tan(tilt_cap) ) + g·sinθ_descent − c₂v²`
(accel) or `… − g·sinθ_descent + c₂v²` (decel). The descent (θ ≈ along −z over the lap) gives a
**gravity assist** on the forward direction, which is why drag and gravity nearly cancel — same
mechanism the TOGT writeup found (drag costs only 0.085 s because the 26 m descent re-pays it).

Two hard caps bind:
- **thrust magnitude** (a_up ≤ 78.3) — binds only at high speed / high tilt;
- **tilt cone** `a_lat ≤ g·tan(tilt_cap)` — the *style envelope*. This is the binding constraint at
  every reasonable tilt cap on this course (see §2). The drag-wall `v ≤ √(a_up/c₂) ≈ 38.8 m/s` only
  bites at the 80° cap.

### 1.3 Tracking realizability
Two checks, because the brief insists the estimate include tracking realizability:
- **Closed-loop geometric tracker** through `rl_plant.step` (parity-identical to the twin, mixer +
  measured aero + 2-tick latency): my crude geometric controller had a sign bug and flew the wrong
  way — I did **not** chase it down (it would burn budget and the answer is already known, see next).
- **Feedforward feasibility** (the load-bearing check, no controller sign ambiguity): differentiate
  the TOPP'd reference to get required `a_up`, thrust-tilt, and body-rate at every sample, and
  compare to the envelope. **Result (steady race window, κ<1, excluding start/finish transients):**

  | tilt cap | a_up p50 | a_up p99 (ceil 78.3) | thrust-tilt p50 | rate p99 (ceil 11) |
  |---|---|---|---|---|
  | 60° | 20.2 | 28.5 | 107° | 1.3 rad/s |
  | 65° | 24.0 | 33.4 | 102° | 1.6 |
  | 75° | 39.4 | 52.7 | 93° | 2.0 |
  | 80° | 58.7 | 76.6 | 88° | 2.5 |

  **Rates never bind** (p99 ≤ 2.5 rad/s vs the 11 rad/s super-rate ceiling) — exactly the TOGT
  finding ("rates barely bind; thrust binds"). a_up rides the ceiling only at the 80° cap.

The **authoritative** tracking-realizability datum is the existing TOGT twin-replay: the geometric
controller needs k=1.85 time-dilation to fly the *saturated* 4.55 s line valid (→ 8.3 s), and a
54-combo gain sweep finds nothing valid below k≤1.7 — a *structural* tracker gap, not tuning. A
min-snap+TOPP line that is planned *feasible* (leaves thrust headroom, unlike the TOGT line) needs a
*smaller* dilation; I bracket the tracking pad at 15–30% rather than 85%.

---

## 2. The honest lap-time estimate

Prototype output (ideal coupled-TOPP, gate-5 plane crossing, centre-stacked geometry):

| tilt cap | lateral cap (m/s²) | **ideal lap (s)** | v_max (m/s) | binding constraint |
|---|---|---|---|---|
| 60° (style cone) | 17.0 | **9.72** | 19.6 | tilt cone |
| 65° | 21.0 | 8.81 | 21.5 | tilt cone |
| 75° | 36.6 | 6.79 | 27.6 | tilt cone |
| 80° | 55.6 | **5.55** | 33.6 | tilt cone (drag wall nearby) |

Adding the tracking pad (15–30%, justified above):

| tilt cap | ideal | **realizable bracket** |
|---|---|---|
| 75° | 6.79 | **7.8 – 8.8 s** |
| 80° | 5.55 | **6.4 – 7.2 s** |
| 60° | 9.72 | **11.2 – 12.6 s** |

**Headline achievable estimate: ~6.5–7.5 s, at an 80° tilt cone (≈ the envelope ladder's step-3
"unconstrained" rung).** I report **7.0 s** as the single-number est_lap_time_s — the centre of the
80°-cap realizable bracket, with the geometry-conservatism (cubic ≥ min-snap curvature) roughly
offsetting the tracking-pad optimism.

### Why this sits *above* the TOGT corrected-aero optimum (4.71 s)
1. **Separability tax.** Min-snap fixes the *line* first (minimising snap, a smoothness proxy), then
   TOPP times it. TOGT collocation co-optimises shape *and* speed against the true dynamics, so it
   finds a line whose curvature is placed exactly where speed is low — min-snap can't. The shipped
   4.55 s geometry already beats any min-snap line on length+curvature.
2. **My tilt-cone model is a g·tan lateral cap**, which is conservative — the real vehicle can point
   thrust freely (the TOGT line inverts to 170°). The moment you let tilt go fully free you *are*
   re-deriving the TOGT optimum, at which point min-snap+TOPP has no remaining advantage over just
   using the TOGT CSV.

---

## 3. The tilt finding (the most important caveat)

The **thrust-tilt is 88–107° (p50) on this course even at the 60° style cap** — because the 26 m
descent forces the thrust vector well past horizontal for the *geometry alone*, independent of
cornering. Cross-check on the **shipped 4.55 s TOGT line**: tilt p50 = 82°, p90 = 113°, max = 170°
(matches the TOGT writeup's "≈170° momentary inversion"). So:

- **A sub-5 s lap on this descending course is geometrically incompatible with the 60° style cone.**
  The RL reward's R4 tilt hinge (free cone 60°, `rw_tilt=4.0`) would heavily penalise *any* fast
  line here. This is consistent with the memory note that the style tax is ~2.3 s/lap and the
  relaxation ladder (60→75–80°) exists precisely to unlock this.
- For min-snap+TOPP, the implication is sharp: **at 60° it can't beat inc7; the method only becomes
  interesting once tilt is relaxed to ~80°**, at which point its value over TOGT-geometry evaporates.
- Honest note: "tilt of the thrust vector" (what binds the convex map + style cone) ≠ "lateral g."
  My TOPP conflates them via the g·tan model. A faithful TOPP would carry the full thrust-tilt
  geometry (including descent), which would make the 60° cone *even more* restrictive, not less.

---

## 4. Contact-free strategy ( < 0.75 − r,  plan ≤ 0.5 m from centre )

- Min-snap pins crossings **exactly at gate centres** (miss = 0.00 m at all 6 in the prototype) — by
  construction ≤ 0.5 m, with full 0.42 m contact-true margin at r=0.38 (pass band 0.75−0.38 = 0.37 m
  is the inner constraint; centre crossing leaves the entire band).
- For a *racing* line you'd bias crossings toward the straightened chord by a small lateral_margin
  (≤ 0.5 m) to shorten the path; the prototype keeps margin=0 for the conservative validity story.
- The TOPP curvature cap guarantees the planned line never demands more lateral accel than the
  envelope provides, so the *planned* line is contact-free by construction. The residual risk is
  **tracking error at the gate plane**, which the 0.37 m margin (r=0.38) must absorb — at gate-4
  (the binding gate, simstart linf 0.215 → margin 0.155 m) this is tight but positive. A min-snap
  line crossing gate-4 dead-centre gives more margin than inc7's 0.215 m linf, *if* the tracker can
  hold it at the post-gate-3 high-speed approach (the open question for every approach).
- Volumetric/slab contact (the inc7 frame-depth model): a centred, perpendicular crossing (gate yaw
  ≈ π ⇒ min-snap crosses near-normal) minimises slab exposure. No special handling needed.

## 5. Envelope validity (tilt cost)

- **Requires relaxation.** At the 60° cone the method is non-competitive (9.7 s). It needs the
  envelope ladder **step 2–3 (free cone 75–80°)** to reach ~5.5–6.8 s ideal. This is the same
  relaxation the RL path needs; min-snap+TOPP does not avoid the style tax — it pays it identically.
- At 80° the **drag wall (38.8 m/s) and thrust ceiling start to co-bind** (a_up p99 = 76.6) — beyond
  80° there is little left to gain (matches TOGT: unconstrained rates buy only −0.04 s; thrust/drag
  is the wall). So 80° is the practical floor of this method's lap time.

## 6. Robustness (plant / latency / perception)

- **Plant error:** the TOPP envelope is parameterised by `a_up_max` and `c₂`; both are measured with
  DR bands. A min-snap line planned at the DR-conservative `a_up_max` (e.g. 70 instead of 78) and
  pooled `c₂` is robust by construction — you simply plan slower. Determinism makes this free to
  re-tune offline.
- **Latency (2 ticks / 67 ms):** the open-loop feasibility ignores latency; the closed-loop tracker
  must absorb it. This is exactly where the geometric tracker's k=1.85 gap comes from. A min-snap
  line's *headroom* (non-saturated, unlike TOGT) is what lets a tracker recover lag — this is the
  method's one robustness *advantage* over the saturated TOGT line.
- **Perception:** min-snap+TOPP is a pure offline planner; it consumes the map, not live vision. It
  is **blind to perception noise** — that risk lives entirely in the tracker/estimator, identically
  for all approaches. No special exposure.
- **The big unknown (shared with TOGT):** aero above the ~7.6 m/s drag-measurement band. At 28–34
  m/s the `c₂·v²` extrapolation is unvalidated; a single high-speed coast probe would pin it. The
  method's lap-time estimate is only as good as that extrapolation (it sets the drag wall).

## 7. Integration cost — LOW

- `src/racer/speed_profile.py` already exists (forward-backward TOPP, `Trajectory`/`Setpoint` seam).
  The only new code is (a) the coupled-thrust envelope (≈40 lines, prototyped here) replacing the
  isotropic `a_max`, and (b) a min-snap QP for the geometry (or keep the cubic seed — adequate here).
- Emits the *same* `peregrine.reference_line.v1` schema TOGT exports, so the RL progress reward
  (`ReferenceLine.progress`) and any MPCC tracker consume it unchanged. Drop-in.
- No WSL, no C++, no Adroit, no casadi/acados toolchain (the TOGT pipeline's main cost).

## 8. Determinism per track — YES, fully

- Pure deterministic numpy: same track map ⇒ bit-identical line, every time. Offline line-iteration
  across attempts is LEGAL (per the directive); min-snap+TOPP is the *ideal* engine for that
  flywheel — re-plan per track offline, ship the line. No in-run external compute.
- Reproducible natively: the prototype runs on the laptop `.venv` in <1 s, no cluster.

## 9. Native pure-numpy prototype — YES

- The whole method is numpy/scipy. The prototype here already computes geometry, the coupled TOPP,
  and a feedforward feasibility check natively. A min-snap QP is one `scipy.linalg.solve_banded`.
  The plant for tracking validation is `rl_plant.step` (pure numpy, no torch needed). **Nothing in
  this approach requires WSL/C++/Adroit.** This is its single biggest practical advantage.

---

## 10. Skeptical self-assessment (where this approach is weak)

1. **It is not the speed frontier.** Honest ceiling ~5.5 s ideal / ~6.5–7.5 s realizable sits ABOVE
   the TOGT corrected-aero optimum (4.71 s) and above inc5's unconstrained RL (6.6–6.9 s). The
   separability tax is real and unrecoverable without re-introducing a full collocation (at which
   point you've rebuilt TOGT).
2. **The 60° style cone kills it.** On *this* descending course the method is non-competitive at the
   trained envelope and only reaches interesting times under the same 75–80° relaxation the RL path
   needs — so it offers no envelope-cost advantage.
3. **The geometry is already beaten.** TOGT's shipped 4.55 s line is a better geometry than any
   min-snap line (shorter, curvature-placed). There is no geometry reason to switch to min-snap;
   its only edge is *feasibility/headroom* of the timing and toolchain-freedom.
4. **My tilt-cone TOPP model is approximate** (g·tan lateral cap conflates lateral-g with
   thrust-tilt; ignores the descent contribution to tilt). A faithful version would make the 60°
   cone *more* restrictive, strengthening conclusion #2 but not changing the achievable number at
   80° much (where thrust/drag co-bind anyway).
5. **Tracking pad is an estimate, not measured.** I bracket 15–30% from the structural TOGT k=1.85
   datum scaled for headroom; the closed-loop number on the real plant is unproven (my tracker had a
   sign bug I declined to chase). The realizable bracket could be wider.

## 11. Recommendation in the S2 context

- **Do not adopt min-snap+TOPP as the speed method.** For the DECOMPOSED architecture, the TOGT line
  is the better explicit plan-line (faster geometry, already exported in the right schema).
- **Do keep min-snap+TOPP as:** (a) the *fallback line generator* if the TOGT toolchain is
  unavailable (it's WSL-only and fragile), and (b) the *per-track offline re-planner* for VQ2's
  determinism-per-track flywheel, where its numpy-native, no-toolchain property is decisive, and (c)
  a source of *feasible, non-saturated* reference lines for tracker bring-up (the TOGT line's
  full-stick saturation is hostile to a tracker; a min-snap line with headroom is friendlier).
- **The adjudication signal:** min-snap+TOPP reinforces "retrain MONOLITHIC first." Its own ceiling
  (~6.5–7.5 s realizable) is *worse* than inc5's unconstrained monolithic RL (6.6–6.9 s twin) and it
  pays the identical 60→80° tilt tax — so a decomposed plan-line+tracker built on min-snap geometry
  is dominated by both TOGT-decomposed and monolithic-RL. Its place is the toolbox, not the frontier.
```
