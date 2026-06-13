# C1 — Envelope-Aware Corrected-Aero Time-Optimal Prototype

**Fengyou** — this is the honest achievable-lap-time-vs-tilt curve on the **REAL (corrected-aero)
plant**, not the falsified linear plant the shipped 4.55 s reference line was built on. It is a
point-mass TOPP planning bound (not a tracked time).

- Script: `handoff/ultracode-planning-togt-s2-2026-06-13/proto_envelope_topp.py`
- Machine-readable result: `handoff/ultracode-planning-togt-s2-2026-06-13/envelope_topp_result.json`
- Run: `PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-planning-togt-s2-2026-06-13/proto_envelope_topp.py`
- Pure numpy/scipy. No tracked source edited, no sim, no SLURM, no network.

---

## Headline result — lap time vs style-envelope tilt cap (CORRECTED AERO)

| tilt cap | a_lat_max = g·tan(tilt) | v_max (m/s) | **LAP (s)** | open-loop feasible? |
|---:|---:|---:|---:|:--|
| 60° | 17.0 m/s² | 38.1 | **5.348** | 97.8% (corner-entry braking 2.2% over ceiling) |
| 65° | 21.0 m/s² | 38.6 | **5.076** | 98.6% (1.4% over) |
| 75° | 36.6 m/s² | 39.2 | **4.686** | **100% feasible** (max req 77.8 ≤ 78.3) |
| 80° | 55.6 m/s² | 39.2 | **4.574** | **100% feasible** (max req 70.5) |
| 90° (unconstrained) | ∞ | 39.2 | **4.574** | **100% feasible** (max req 70.5) |

- **Style-respecting (60°) achievable bound = 5.35 s.**
- **Unconstrained (90°) bound = 4.57 s.**
- **Tilt tax (60° vs 90°) = 0.77 s** on the corrected-aero plant — much smaller than the
  ~2.3–2.9 s/lap RL tax inc5 measured. See Finding 3 for why (it is NOT a contradiction).
- 80° and 90° give the **same** lap (4.574 s): at ≥80° the lateral cap stops binding entirely —
  the v² drag wall (~39 m/s) is the dominant ceiling everywhere on this gentle course.

Per-gate in-plane miss is 0.000 m at every gate for every tilt because the planned line is the
**centre line** through the gate centres (contact-free by construction at r∈{0.28,0.33,0.38}).
That makes "contact-free" trivially true for the *plan*; the binding validity question is whether a
*tracker* can hold the centre line within the (0.75−r) band, which is the S2 tracker question
(Finding 5), not a property of the bound.

---

## Cross-check (the load-bearing validation)

The unconstrained TOPP lap **4.574 s** lands right next to the independent C++ TOGT corrected-aero
result (`handoff/laptop-togt-bound-2026-06-10/cases/expl_corrected_aero/analysis.json`):

| source | lap (s) | max speed (m/s) | note |
|---|---:|---:|---|
| TOGT corrected-aero **init** (planTOGT) | 4.123 | 60.0 | contact-tolerant (miss up to 0.78 m), T/W=8, c2=0.052 |
| TOGT corrected-aero **refined** (IPOPT) | **4.714** | **39.26** | collective saturated 91% of nodes → thrust-bound |
| **this TOPP, 90° unconstrained** | **4.574** | **39.17** | point-mass, thrust-ball budget |

- **v_max agreement is exact**: 39.17 (TOPP) vs 39.26 (TOGT refined) — the v² drag wall is
  reproduced independently. Both confirm **thrust binds, not rates** (TOGT: 91% collective
  saturation; this prototype: drag wall = thrust-vs-drag balance).
- The 0.14 s gap (4.574 vs 4.714) is expected and in the right direction: the point-mass TOPP has
  no body-rate / tilt-rate dynamics (instant attitude), so it is a hair optimistic vs the
  rate-limited multiple-shooting TOGT. **The ceiling is robust at ~4.5–4.7 s on the real plant.**

---

## Method

### 1. Geometric line
Natural cubic spline (`scipy.interpolate.CubicSpline`, chord-length parameterised) through
`[start, G0..G5]` (7 points), 5000 dense samples. Max deviation of the spline from any gate centre
= **0.0125 m** (asserted ≤ 0.5 m). Line length 164.65 m; **min radius of curvature 29.97 m**
(κ_max = 0.0334 1/m) at s≈116 m (the G3 region) — this is a **gentle** course.

### 2. Corrected-aero acceleration budget (the whole point)
Constants pulled live from `src/racer/rl_plant.py`:
- **Thrust ceiling** `A_UP_MAX = COLL_MAP_ACCEL_MEASURED[-1] = 78.283 m/s² (7.98 g)` — full-stick
  body-up specific force from the measured **convex** collective map.
- **Quad drag** `c2 = QUAD_DRAG_C2_POOLED = 0.052 1/m` — the v² drag wall.
- `g = 9.80665`.

The achievable specific-acceleration set is the **thrust ball**: `{a_thrust + grav : |a_thrust| ≤
A_UP_MAX}` in NED (a ball of radius 78.283 centred at `grav = [0,0,+g]`). Decomposed in the path
frame (tangent `t̂`, principal normal `n̂`, binormal `b̂`):

- **Cornering speed ceiling** — to hold a corner at speed v needs centripetal `a_c = κv²` on `n̂`.
  Feasible only if the **minimum** (zero-tangential) specific force fits the ball:
  `g_t² + (a_c − g_n)² + g_b² ≤ A_UP_MAX²`  ⇒  v ≤ sqrt(a_c_max_ball / κ).
- **Style envelope (lateral cap)** — `a_lat_max = g·tan(tilt_cap)`; centripetal `κv² ≤ a_lat_max`
  ⇒ v ≤ sqrt(a_lat_max/κ). This is the SAME total-tilt R4 penalises in `peregrine_racing.py`
  (`a_lat = g·tan(tilt)`). **This couples cornering speed to the tilt envelope.**
- **Top-speed (drag wall)** — straight-and-level forward thrust after gravity-cancel balances
  drag: `sqrt(A_UP_MAX² − g_perp²) + g_t = c2·v²` ⇒ v_drag ≈ 39 m/s.
- The speed ceiling is the min of those three; the **forward/backward TOPP sweep** (iterated to a
  fixed point, conservative worst-of-segment centripetal load) integrates the tangential budget
  `a_tan ≤ g_t + sqrt(A_UP_MAX² − (a_c−g_n)² − g_b²) − c2·v²` (drag aids braking, hurts accel).
  Gravity-along-tangent `g_t` is **signed** — it *helps* on the descending course (course drops
  ~26 m), which is correctly credited.

### 3. Feasibility audit
Per tilt I report the **integrator-realised** required specific force `|f| = |a_tan·t̂ + a_cen·n̂ −
grav|` against the 78.283 ceiling (excluding the first 1 m standing-start ramp, where at v≈0 the
accel direction is meaningless). At 75/80/90° the plan is **100% inside the ball**; at 60/65° a
small fraction (2.2%/1.4%) of corner-entry braking samples exceed it (see Finding 4).

### 4. Twin realizability probe (secondary, NOT the deliverable)
A naive feedforward + PD point-tracker through the fully-measured `rl_plant` (mixer + convex map +
quad drag + super-rate) **diverges** (~38 m) on the 60° plan. This is expected and informative, not
a bug: a 35+ m/s plan with 67 ms live latency and super-rate attitude lag is not trackable by a
hand-rolled P controller — exactly the "a good tracker is REQUIRED" point (the existing geometric
tracker needs k=1.85 to follow the shipped 4.55 s line). The **open-loop feasibility** block is the
realizability verdict.

---

## Findings

1. **CORRECTED-AERO CEILING IS ROBUST ~4.5–4.7 s.** Unconstrained TOPP 4.574 s ≈ TOGT corrected
   refined 4.714 s; v_max 39.2 ≈ 39.26. The doubled thrust (T/W~8) does NOT buy a sub-4 s lap — the
   v² quad-drag wall caps speed at ~39 m/s and **thrust binds** (rides the ceiling). This corrects
   the falsified-linear-plant intuition behind the 4.27 s "bound" and the 4.55 s shipped line.

2. **80°/90° are identical (4.574 s): on THIS course the tilt cap stops mattering at ≥80°.** The
   sharpest corner (R=30 m) at the drag-wall speed 39 m/s needs only a_lat = 39²·0.0334 = 50.8
   m/s² = g·tan(79°). So any cap ≥ ~79° never binds; the drag wall is the sole ceiling.

3. **The kinematic tilt tax here is only 0.77 s — NOT a contradiction of inc5's ~2.3 s.** inc5's tax
   was measured in a **slow regime** (9.52 s @ rw_tilt=96 vs 6.89 s unconstrained, i.e. ~25 m/s
   class) and is dominated by **RL reward-shaping** (the R4 hinge penalising tilt even when not
   kinematically required), not by the hard kinematic cornering cap. On a *time-optimal centre line
   at the corrected-aero ceiling*, the course is gentle enough (R_min ~30 m) that the 60° cap binds
   only on ~55% of the line and only forces ≤22.6 m/s at the single sharpest corner. **The pure
   kinematic cost of respecting 60° is small (~0.77 s); the inc5 tax is mostly shaping cost.** This
   strongly suggests envelope-ladder step 1 (rw_tilt 96→48) recovers most of the lost time **without
   needing to relax the free-cone** — the kinematic envelope barely binds.

4. **At tight caps (60/65°) corner-ENTRY braking is the binding constraint, not cornering itself.**
   The 2.2%/1.4% over-ceiling samples are all at s=68–136 m at 32–38 m/s: decelerating from a fast
   straight into a 22.6 m/s corner needs >78 m/s² braking for a few samples. So **the 60° lap 5.35 s
   is a mild lower bound** (true achievable ~+0.1–0.2 s slower to keep braking in-budget). The 75–90°
   laps are clean bounds.

5. **The bound is contact-free on the centre line, but realizing it needs a capable tracker.** The
   plan's per-gate miss is 0 by construction; the open-loop thrust demand is feasible (≥75°); but a
   naive tracker diverges. This is the **S2 architecture crux**: the achievable-time ceiling
   (~4.6–5.3 s depending on style) is real, but capturing it requires either learn-the-line RL
   (monolithic, inc7-class) or an MPCC/learned tracker on this line (decomposed). The bound itself
   does not discriminate the two; it sets the prize (~5 s class is reachable vs inc7's ~9.76 s twin
   / 11.45 s deployment — a ~2× structural gap to close).

---

## S2 read (informational; the bound feeds, not decides, the architecture)
- The corrected-aero **prize is ~4.6 s (unconstrained) / ~5.35 s (60° style)** — roughly **2×
  faster than inc7's 9.76 s twin median**. The gap is structural (tracking + speed), not residual.
- Finding 3 (kinematic tilt tax is small) **lowers the cost of the decomposed path's main worry**:
  a plan-line that respects 60–65° loses only ~0.5–0.8 s of ceiling, so a decomposed plan-line +
  RL/MPCC tracker is NOT giving up much ceiling to style — the bigger lever is tracker quality.
- Finding 1 (thrust binds, drag wall) means **neither architecture beats ~4.5 s**; the differentiator
  is which one most reliably reaches the ~5 s class while staying contact-free. Monolithic inc7 is
  live-confirmed working at 11.45 s; the decomposed line-iteration flywheel is the candidate to
  close the 11.45→~5–6 s gap. This bound makes the target concrete.

---

## Caveats / limitations
- **Point-mass planning bound**, not a tracked time. No body-rate / tilt-rate dynamics (instant
  attitude); TOGT shows that costs only ~0.04 s on rates but ~0.14 s overall vs multiple-shooting.
- **Pooled isotropic drag (c2=0.052).** The measured table is per-axis/per-sign
  ([[0.042,0.058],[0.055,0.055],[0.054,0.076]]); climb (−z, 0.076) brakes harder than the pooled
  value on the ~26 m descent — a per-axis budget would shift the drag wall slightly (second-order).
- **Convex map at ~0 airspeed.** The collective map over-predicts thrust ~22% in 4–12 m/s, but the
  **airspeed lapse is VOIDED** (FRAME-AUDIT 2026-06-12) and correctly NOT enabled here. Treat the
  78.3 m/s² ceiling as the ~0-airspeed value; at race speeds the true thrust ceiling is a touch
  lower, making the 90° bound mildly optimistic (consistent with TOGT being 0.14 s slower).
- **60/65° laps are mild lower bounds** (corner-entry braking 1.4–2.2% over the thrust ball).
- **Centre-line geometry only.** No through-gate normal entry/exit shaping (min-snap rung); a
  through-gate-aware line could trim corner curvature and shave time, OR a margined line (offset for
  tracker error) would cost time. This is the bare centre-line datum.
- Standing-start ramp excluded from the feasibility/tilt stats (v≈0 makes accel direction undefined).
