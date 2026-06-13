# Phase B — HYBRID (decomposed) approach design + honest achievable-time estimate

**Approach:** explicit offline plan-line (geometry) + RL/MPCC **tracker** with **arc-length
progress reward** over `rl/reference_line.py::progress()`. The S2 *decomposed* architecture.
Inversion structurally impossible (the tracker never commands a line; it follows one). Ceiling
bought back by the offline line-iteration flywheel.

Author: independent Phase-B reviewer (Fengyou's workflow). Prototype: pure numpy/scipy, native,
no WSL/C++/Adroit. Files in this dir: `hybrid_prototype.py` (corrected-aero TOPP), `hybrid_tracker.py`
(realizability check).

---

## TL;DR (the honest number)

| Layer | Lap time (last-gate) | Status |
|---|---|---|
| Shipped reference line (falsified linear plant) | **4.551 s** | FICTION — needs 51 m/s peaks, 33 m/s mean |
| Corrected-aero **point-mass** TOPP on the same geometry (tilt 65–80°) | **4.73–4.93 s** | drag-wall-limited PLANNER FLOOR; **NOT realizable** (commands 169° tilt, 90+ rad/s axis slew) |
| **Realizable hybrid** (RL/MPCC tracker, k≈1.5–1.7 dilation) | **~7.5–8.5 s offline twin** | **the defensible estimate** |
| → fresh-sim deployment (add ~1.5 s, project rule) | **~9–10 s** | |
| Plain geometric tracker (project's own k=1.85 datum) | ~8.8–9.3 s offline | fallback if no good tracker |

**est_lap_time_s = 8.0 s (offline twin), band 7.5–8.5 s.** This is *slower* than the inc5
unconstrained monolithic twin (6.9 s) and only modestly faster than inc7 monolithic offline
(9.76 s). The hybrid does **not** reach the corrected-aero point-mass floor because that floor is
unrealizable, and the decomposition caps the ceiling below learn-the-line.

---

## 1. Geometry method

**Source geometry:** the spatial path of `rl/reference_line_vq1.json` (positions only). I *discard*
its timing — that was built on the falsified linear plant (T/W 3.765 linear, ω_max [11,11,7],
linear_drag 0.21) and demands 51.1 m/s peak / 33.3 m/s mean, both impossible on the corrected-aero
plant (drag-wall top speed ~36–39 m/s; verified below). The path itself is a smooth arc-length
cubic spline through the 6 gate centres with ≤0.14 m crossing offsets (contact-safe by
construction).

**For a *production* hybrid the geometry must be RE-PLANNED, not just re-timed.** Two reasons:
1. The shipped line's curvature near the start was shaped for a 51 m/s plant; at corrected-aero
   speeds the TOPP forces 169° commanded tilt over arc 0–36 m (the launch/first-corner zone) —
   the point-mass planner brakes/turns by *inverting* the thrust vector. That is the exact
   backflip-dive geometry the decomposition is supposed to forbid.
2. Re-planning options, in increasing fidelity:
   - **min-snap** through gate waypoints with entry/exit along each gate normal (the
     `waypoints_from_gates` + min-snap rung already scoped in `speed_profile.py`'s docstring).
     Pure-numpy, native. Gives C⁴ geometry → bounded thrust-axis slew.
   - **TOGT / multiple-shooting** (the existing `scripts/togt/` C++ pipeline) re-run with the
     CORRECTED-aero constraint set (quad-drag c2=0.052, convex thrust map, ω_max 11). **Caveat:**
     the existing `expl_corrected_aero` TOGT case (`handoff/.../cases/expl_corrected_aero`) reports
     T=5.23 s but peaks at **60 m/s** — its constraint set still does NOT encode the v² drag wall,
     so that CSV is also fiction. TOGT needs WSL and a correct drag model; do not trust the
     existing corrected CSV.

**Contact-free strategy (geometry layer):** plan crossings ≤0.13 m from centre (shipped line
already does). Contact budget at the binding gate (gate-4, r=0.38) is 0.75−0.38 = 0.37 m in-plane
L∞. With a 0.13 m plan offset the tracker has **0.24 m cross-track slack** — this is the real
constraint, and it is a *tracking-precision* problem, not a planning one (see §4). The progress
reward is taken over the *planned, contact-safe* line, so the tracker is never rewarded for
cutting a corner inside the frame.

## 2. Timing method

**Corrected-aero forward/backward TOPP** (`hybrid_prototype.py::corrected_aero_topp`), the same
forward/backward numerical-integration TOPP as `src/racer/speed_profile.py` but with the REAL
envelope:

- **Thrust authority:** full-stick body-up accel = `interp(1.0, COLL_MAP_THR/ACCEL_MEASURED)` =
  **78.3 m/s² (7.98 g)** — measured convex map, ~2.1× the linear g·thr/hover fantasy.
- **Tilt cap → horizontal accel:** at tilt θ, usable horizontal thrust `a_h = 78.3·sinθ`
  (67.8 / 70.9 / 75.6 / 77.1 m/s² at 60/65/75/80°). Lateral cornering budget = `a_h`. The lateral
  envelope is **never** binding — cornering at these gates needs <20 m/s² but a_h ≥ 68 m/s².
- **Drag-wall speed ceiling** (THE binding constraint): steady level flight at tilt θ has
  `c2·v² = a_h` ⇒ `v_top = sqrt(a_h/0.052)` = **36.1 / 36.9 / 38.1 / 38.5 m/s** at 60/65/75/80°.
  The TOPP rides this wall ~84% of the lap (v mean 34–36, max 36–38 m/s). Quad-drag (c2≈0.052,
  per-axis-per-sign QUAD_DRAG_C2_MEASURED) is the dominant physics: it both caps top speed and
  aids braking.
- **Vertical sustainability:** `a_v = 78.3·cosθ` = 39.1 / 33.1 / 20.3 / 13.6 m/s² at 60/65/75/80°.
  All ≥ g (9.8), so level flight is sustainable up to 80°; above ~83° the quad cannot hold
  altitude except while descending (the course descends 26 m, which *helps* — net favourable).

**Result (point-mass TOPP, last-gate crossing):**

| tilt cap | a_h | v_top | lap (TOPP, point-mass) |
|---|---|---|---|
| 60° | 67.8 | 36.1 | 5.046 s |
| 65° | 70.9 | 36.9 | 4.933 s |
| 75° | 75.6 | 38.1 | 4.778 s |
| 80° | 77.1 | 38.5 | 4.732 s |

The whole 60→80° sweep moves the lap by only 0.31 s, because the binding constraint is the **drag
wall** (v_top), not the lateral envelope. Past ~65° you buy almost nothing. **This matches the
prompt's exploratory 4.71 s corrected-aero figure and confirms the ROBUST ~4.3–4.7 s ceiling.**

## 3. THE REALIZABILITY TAX (why 4.7 s is fiction, and what's actually achievable)

The TOPP above is a **point-mass** model: it assumes the acceleration vector can point anywhere
instantly. The real CtbrPlant rotates its thrust vector at finite rate (inner rate loop τ=19 ms,
super-rate ceiling ~11 rad/s/axis). I measured what the corrected-aero reference *demands* of the
thrust axis (`hybrid_prototype.py`, slew analysis):

- **Commanded tilt peaks at 169–170°** (a near-inversion) over arc 0–36 m. The point-mass planner
  brakes/turns the standing-start launch by pointing thrust *backward* — physically a backflip.
- **Thrust-axis slew rate:** p95 ≈ 12.4–12.9 rad/s, **max ≈ 90–108 rad/s** — 8–10× the ~11 rad/s
  ceiling. **6.4–7.8% of samples need >11 rad/s** axis slew.
- **Implied path decel:** up to **14.3 g** along-track — exceeds the 8 g thrust authority outright.

Capping the brake budget at the real horizontal thrust authority (`hybrid_tracker`-side honest
TOPP) does NOT remove the 169° tilt / 90 rad/s slew — those originate from the launch geometry,
which was shaped for a 51 m/s plant. **Conclusion: the shipped line geometry is not cleanly
trackable at corrected-aero speeds; the hybrid MUST re-plan geometry (min-snap/TOGT) AND the
realized time carries a tracking-realizability dilation k > 1.**

**Empirical tracker test** (`hybrid_tracker.py`): a differential-flatness feedforward + position/
velocity/attitude PD on the real mixer plant **cannot** follow the corrected-aero line — gate
misses of 5–35 m, and high attitude gains go unstable (142 m divergence). My tracker is cruder
than the project's shipped geometric tracker, so this is a *lower bound* on tracker quality, not a
verdict — but it corroborates the slew analysis: the line over-drives the plant.

**Dilation estimate.** The project's own datum: the *shipped* geometric tracker needs **k=1.85**
(8.3 s) to track the 4.55 s line. The corrected-aero line is smoother (38 vs 51 m/s peaks, lower
jerk) so an RL/MPCC tracker should need *less* dilation. Defensible band:
- **RL or MPCC tracker, k≈1.5–1.7:** 4.78 × 1.5–1.7 = **7.2–8.1 s** offline twin.
- **Plain geometric tracker, k≈1.85:** 4.78 × 1.85 = **8.8 s** offline.

Taking the RL/MPCC tracker (the realistic hybrid build) and centring: **~7.5–8.5 s offline twin,
point estimate 8.0 s.** Fresh-sim deployment adds ~1.5 s (project rule, fresh-respawn post-gate-3
sensitivity) → **~9–10 s deployed.**

**Cross-checks (same plant, empirical RL):** inc5 *unconstrained monolithic* twin = 6.9 s; inc7
monolithic offline = 9.76 s / live 11.45 s. An 8.0 s hybrid twin sits between them — *plausible*
and consistent with "decomposition ceiling below learn-the-line, but the line-iteration flywheel
+ feedforward beats a from-scratch reward-shaped monolith on the start/brake zones."

## 4. Envelope validity (tilt cost)

The corrected-aero TOPP is drag-wall-limited, so it does **not need extreme tilt** for speed:
65° already gives v_top 36.9 m/s and 4.93 s; 80° only buys 0.20 s. **The hybrid can live inside or
near the 60° style cone** with a ~0.2–0.3 s/lap penalty vs 80° — far cheaper than the monolithic
2.3 s/lap tilt tax (which came from cornering, not straights). **Recommendation: plan at 65–70°
tilt cap; relaxation beyond the 60° free cone is OPTIONAL and small.** This is a genuine hybrid
advantage: the feedforward line pre-commits the tilt schedule, so the tracker need not discover
the high-tilt straight-line posture through reward.

## 5. Robustness (plant / latency / perception)

- **Plant error:** GOOD. The feedforward is computed from the *planned* line; plant mismatch is
  absorbed by the tracker's feedback, and the line is re-iterable offline against the measured
  plant. The decomposition's structural win: **inversion is impossible** (no 169° backflip — the
  tracker clamps tilt at the cap and the planned line never inverts after re-planning).
- **Latency (67 ms / 2 ticks live):** at 37.7 m/s through gate-4 this is **2.53 m ALONG-track lag**
  (benign — affects time, not validity) but cross-track error from a held 0.1 rad attitude error
  over 50 ms is only ~1 mm. The binding gate-4 cross-track budget (0.24 m, §1) is comfortable for
  a feedforward tracker *if* the line is accurate. The risk is **along-track phase**: a laggy
  tracker arrives at the gate plane late but on-line — validity-safe, time-costly (folds into k).
- **Perception noise (VQ2):** world-fix σ≈[0.73,0.47,0.29] m (N,E,D), ~47% acceptance. The hybrid
  tracks a *world-frame* line, so perception error maps directly to cross-track error. At 0.47 m
  East σ this **threatens the 0.24 m gate-4 budget** — the tracker+KF must filter to <0.2 m. This
  is the same perception bar as the monolith; not worse, but not helped by decomposition.

## 6. Integration cost — **HIGH**

- New planner stage (min-snap or corrected-TOGT) producing a contact-safe line under the REAL
  envelope. min-snap is native numpy; corrected-TOGT needs WSL + a correct drag model (the
  existing corrected CSV is fiction).
- New tracker: either an **MPCC** (acados — does NOT build on this Windows box; Linux-sim-box
  only) or an **RL-tracker** retrained with the progress reward + line-relative observations
  (a new obs layout: line-frame errors, not gate-relative). The latter is a near-from-scratch RL
  campaign — the monolithic inc-line is NOT reusable (different obs/reward).
- `progress()` exists and is cheap (O(N) at 50 Hz), so the reward primitive is free. But the
  *observation* redesign (line-relative state) and a feasible-line export pipeline are real work.
- Versus monolithic: the monolith (inc7) is **already live-confirmed**; the hybrid is a fresh
  build. Integration cost is the strongest argument against doing this now.

## 7. Determinism per-track

**Excellent and offline-reproducible.** The plan-line is computed offline per track and frozen;
the tracker is deterministic given the line and the state. Offline line-iteration across attempts
is explicitly LEGAL (determinism is per-track; in-run external compute is not). The line is a
static asset shipped with the stack — exactly the determinism contract the rules want. This is the
hybrid's **cleanest win**: per-track the behaviour is a fixed feedforward, trivially auditable.

## 8. Native-prototypable?

**Partially.** The PLANNER (corrected-aero TOPP / min-snap) and the realizability analysis are
**fully native** (this prototype: pure numpy/scipy, validated here — top speed, slew, tilt,
TOPP all reproduced). The forward-sim plant (`twin.py` / `rl_plant.py`) is native. So the
*feasibility and time-ceiling* of any candidate line can be validated offline with no WSL/Adroit.
What is NOT natively prototypable to a deployment number: (a) MPCC tracker (acados = Linux only),
(b) RL-tracker training (Adroit). So the offline prototype can VALIDATE the line and BOUND the
realizable time (via the k-dilation datum), but the final tracker-in-the-loop number needs the
sim box. The planner half is offline-native; the tracker half is not.

## 9. Skeptical self-assessment (weaknesses)

1. **The 4.7 s "ceiling" is a point-mass fiction.** My own slew analysis kills it (169° tilt,
   90 rad/s). Anyone quoting <6 s for the hybrid is quoting the unrealizable floor.
2. **The k-dilation is borrowed, not measured for *this* tracker.** k=1.5–1.7 is an
   interpolation between the project's k=1.85 geometric datum and the smoother corrected line. A
   real MPCC could beat 1.5; a mediocre RL-tracker could be worse than 1.85. The 7.5–8.5 s band
   has real uncertainty.
3. **It is probably SLOWER than just retraining the monolith.** inc5 unconstrained = 6.9 s twin
   already beats my 8.0 s hybrid estimate; inc7 (constrained) = 9.76 s. The monolith's ceiling is
   higher and it is *already live-confirmed*. The hybrid's only structural wins are
   inversion-impossibility (already solved in the monolith by the plant fix + DR) and
   determinism-audit (nice-to-have, not VQ2-ranked).
4. **Perception couples to cross-track directly** (world-frame line). The 0.24 m gate-4 budget vs
   0.47 m East σ is tight — the hybrid does not relax the perception bar.
5. **Integration cost is high and the line must be RE-PLANNED** (the shipped one is unusable as
   geometry at corrected speeds), so "reuse the 4.55 s line" is not actually available.

**Verdict:** the hybrid is the right *fallback* (prior reviewer read confirmed) — it structurally
forbids the backflip-dive and gives per-track determinism — but on the corrected-aero envelope its
honest achievable time (~8 s offline / ~9–10 s deployed) does **not** beat retraining the
monolith, which already works. Recommend it only if monolithic style pathologies *re-emerge* under
envelope relaxation; otherwise the corrected-aero TOPP here is most valuable as a **feasible
reference-line generator for a HYBRID-progress-reward MONOLITH** (the 3rd option: MPCC-cast
arc-length progress over a corrected-aero line, keeping the monolith's ceiling while importing the
hybrid's anti-inversion structure).
