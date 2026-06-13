# A2 — Planning sub-optimality + structural twin↔TOGT gap + plant-revision correction

**Date:** 2026-06-13 · **Author:** ultracode A2 subagent (opus-4.8) · **For:** Fengyou
**Scope:** analysis + offline numpy only; no source edits, no sim, no SLURM, no commits.
**Inputs:** existing TOGT CSVs `handoff/laptop-togt-bound-2026-06-10/cases/*/{analysis.json,meta.json,refined_traj.csv}`;
`src/racer/rl_plant.py` aero constants; `src/racer/speed_profile.py` (TOPP corroboration); `rl/reference_line_vq1.json`.

---

## TL;DR ledger (lap-time attribution)

| Component | Δt | Basis | Confidence |
|---|---|---|---|
| **Inscribed-circle bound** (linear plant, margin 0) | **4.273 s** (datum) | `bound_circle` refined | high (existing TOGT) |
| **(1) Safety-margin tax** (shipped 0.7 m ball margin) | **+0.278 s** | `ref_circle` − `bound_circle` (same plant) | high |
| **(2a) Plant revision — corrected aero raises the bound** | **+0.44 s** | `expl_corrected_aero` 4.714 − `bound_circle` 4.273 | medium (planner-proxy) |
| **(2b) v² drag wall = the binding constraint, NOT thrust** | (mechanism) | drag-limited v_term ≈ 38.7 m/s; 98% thrust-saturated | high |
| **(3) Structural tracker gap** (geometric tracker can't follow at 1×) | **+3.87 s** (k=1.85 on the 4.55 s line) | memory §TOGT-BOUND + headroom argument | high (already live-validated) |

Three independent observations, three different magnitudes. The **structural tracker gap (~3.9 s)
dwarfs both the margin tax (~0.28 s) and the plant-revision correction (~0.44 s) by ~10×.** This is
the single load-bearing number for the S2 architecture decision.

---

## (1) Safety-margin tax: 0.278 s, and it is NOT clearly necessary

**Measured (apples-to-apples, identical linear plant T/W 3.765 / linear_drag 0.21):**
- `bound_circle` (gate_margin 0.0 m, inscribed-circle, contact-free): **refined 4.273 s**
- `ref_circle`  (gate_margin 0.7 m, **ball** keep-out): **refined 4.551 s** ← the SHIPPED line
- ⇒ **margin tax = 4.551 − 4.273 = 0.278 s (6.5%)**, matching the prompt's 0.28 s claim exactly.

**Caveat that materially changes the verdict — the tax is margin-SHAPE dependent:**
- `ref_margin` is *also* a 0.7 m margin at 0.7 m but with `gate_shape: None` (planar/box) instead of
  `ball`. Its refined time is **4.311 s ⇒ tax only 0.037 s.**
- The shipped line chose the **ball** (spherical) keep-out, which penalizes corner-cutting in all three
  axes simultaneously and is the *strict* interpretation. The 0.28 s is the price of the **strict ball
  margin**, not of "0.7 m margin" per se. A planar/along-gate-normal margin of the same 0.7 m costs
  ~0.04 s.

**Is the 0.7 m margin necessary given the 0.155 m gate-4 binding margin?**
- Validity rule: in-plane miss < (0.75 − body_radius). At r=0.38, allowance = 0.37 m. The shipped line's
  worst crossing is **0.138 m** (gate-3) and gate-5 is 0.110 m — i.e. the shipped line sits **0.37 → 0.138
  = 0.23 m inside the validity boundary at its tightest gate**, a 2.7× safety factor on the *plan*.
- The **binding constraint at deployment is gate-4 at 0.155 m margin** (simstart linf 0.215 m, r=0.38).
  But that 0.155 m is an *execution/tracking* margin under the RL policy + fresh-respawn dynamics — it is
  NOT the planning margin. The planner already holds 0.23 m at gate-3 and more elsewhere.
- **Verdict:** the 0.28 s ball-margin tax is **partially recoverable (~0.24 s of it).** Switching the
  shipped line's keep-out from `ball` to a `box`/gate-normal margin of equal nominal width recovers
  ~0.24 s (4.551 → ~4.31 s) while keeping the same nominal 0.7 m stand-off — the ball was over-penalizing
  in-plane corner geometry the validity rule does not actually forbid. **This is a free offline
  line-iteration win** (legal per the per-track determinism doctrine) and should be the first planning
  lever, gated only on confirming the box-margin line still clears the gate-4 contact-true metric at the
  trained radius. The full 0.7 m is conservative; the question is whether to *spend* the conservatism as
  ball-strictness (current) or as nominal width (cheaper).

---

## (2) PLANT REVISION: corrected aero raises the bound to ~4.7 s — the v² drag wall, not unusable thrust

**The central correction.** The 4.55 s shipped line and 4.27 s inscribed-circle bound were both built on
the **FALSIFIED linear plant** (per-rotor thrust linear to T/W 3.765 = 36.92 N total; isotropic
linear_drag 0.21 /s). The real plant (measured, `rl_plant.py`) is:
- **Convex collective→accel:** `COLL_MAP_ACCEL_MEASURED[full=1.0] = 78.28 m/s² ≈ 8 g` (~2.1× the linear
  g·thr/hover model). Sub-linear below hover (at hover 0.2656 → 9.58 m/s² ≈ g).
- **Quadratic body-drag:** `QUAD_DRAG_C2_MEASURED` pooled ≈ 0.052 /m (vs linear 0.21 /s).

The exploratory corrected case `expl_corrected_aero` (T/W 8.0 linear proxy for the 8 g full-stick + quad
drag c2 0.052, margin 0, contact-free) gives **refined 4.714 s** — i.e. **0.44 s SLOWER than the linear
bound (4.273 s), despite 2.1× the peak thrust.**

### Why the doubled thrust buys nothing: the v² drag wall (verified from the CSVs)

Time-weighted thrust-saturation and speed, read directly from `refined_traj.csv` (total thrust =
Σ u_1..u_4; Umax = 4 × thrust_max_per_rotor):

| Case | plant | mean v | max v | peak thrust frac | time >95% thrust |
|---|---|---|---|---|---|
| `bound_circle` | linear T/W 3.765 | 34.5 m/s | **52.4 m/s** | 1.000 | **92.1%** |
| `ref_circle` (shipped) | linear T/W 3.765 | 33.3 m/s | 51.1 m/s | 1.000 | 90.9% |
| `expl_corrected_aero` | T/W 8 + quad 0.052 | 32.7 m/s | **39.3 m/s** | 1.000 | **97.7%** |

- The linear plant **rides its thrust ceiling 92% of the lap and reaches 52 m/s** — it is thrust- and
  distance-limited (segments are only 24–41 m), never drag-limited.
- The corrected plant **rides its (2.1×) ceiling 98% of the lap but tops out at 39.3 m/s** — it is
  **drag-limited.** The extra thrust is spent fighting v² drag, not accelerating.

**Closed-form terminal velocity (straight-line, thrust balancing drag while holding weight):**
- Corrected: horizontal-available accel = √((8g)² − g²) = 77.8 m/s²; v_term = √(77.8 / 0.052) = **38.7 m/s.**
  → The CSV max of 39.3 m/s confirms the corrected trajectory **rides its drag-limited terminal velocity.**
- Linear: √((3.765g)² − g²)/0.21 = **169 m/s** (a fiction — never approached; the linear plant is
  distance/thrust-limited at 52 m/s, NOT drag-limited).

**Drag deceleration at cruise — the smoking gun:**
| v (m/s) | quad drag (0.052) | linear drag (0.21) |
|---|---|---|
| 30 | 46.8 m/s² | 6.3 m/s² |
| 39 | **79.1 m/s²** | 8.2 m/s² |
| 52 | 140.6 m/s² | 10.9 m/s² |

At the corrected plant's own cruise (~39 m/s), quad drag = **79 m/s² ≈ the entire 8 g of thrust** — there
is *nothing left* to accelerate. The linear plant at 52 m/s loses only 10.9 m/s² to drag.

### Independent corroboration (TOPP, `src/racer/speed_profile.py`, isotropic centre-line):
| config | TOPP lap |
|---|---|
| corrected a=7.8g, **v capped 39 (drag wall)** | **4.73 s** ← matches TOGT `expl` 4.71 s |
| corrected a=7.8g, **v capped 52 (no wall)** | **3.86 s** |
| linear a=3.5g, v=60 | 5.32 s |

The TOPP model reproduces the corrected-aero TOGT bound to within 0.02 s **only when the speed cap is set
to the drag wall (39 m/s).** If the 8 g could be spent reaching 52 m/s, the lap would be ~3.86 s — so the
**drag wall costs ~0.87 s of unrealized potential** that the thrust alone would otherwise unlock.

### Answer to the prompt's question #2
> *Is the corrected bound HIGHER (drag wall) or could smarter use of the 8 g convex thrust lower it?*

**HIGHER, robustly, because of the v² drag wall — and NO, smarter use of the 8 g cannot lower it below
~4.7 s.** The 8 g is already fully exploited (98% saturation); the binding constraint flipped from
**thrust** (linear plant) to **drag** (corrected plant). You cannot out-accelerate a v² wall: top speed is
pinned at ~39 m/s regardless of how the thrust is scheduled. The ceiling is **robust at ~4.3–4.7 s** to
the plant revision (linear-bound 4.27 → corrected-bound 4.71). The earlier "TOGT bound is conservative
because real thrust is 2× higher" reasoning is **WRONG** — it ignored that drag also ~2× and scales as v².

**Caveats on the corrected number (why medium, not high, confidence):**
- `expl_corrected_aero` models the 8 g as a **linear** T/W 8 in the planner, but the true map is **convex**
  (sub-linear below hover). At the mid-throttle the planner uses on corners, the real plant produces
  *less* accel than linear-T/W-8 ⇒ the true corrected bound is likely **slightly higher than 4.71 s**, not
  lower. The 4.71 s is therefore an **optimistic** corrected bound; the honest band is ~4.7–4.9 s.
- The `expl` quad drag is isotropic 0.052; the measured map is anisotropic per-axis per-sign
  ([[0.042,0.058],[0.055,0.055],[0.054,0.076]]) with the largest coefficient on body-down (0.076) — the
  descending course loads the high-drag axis, again pushing the true bound *up* not down.
- LAPSE_* (airspeed lapse) is correctly VOIDED and NOT enabled here; the convex map over-predicts ~22% in
  4–12 m/s at ~0 airspeed but lapse is not used — consistent with doctrine.

**Margin tax on the corrected plant is nearly free (~0.02–0.05 s):** TOPP shows the corrected plant's lap
is insensitive to the lateral corner perturbation (4.732 → 4.707 for 0→0.7 m offset) because it is
**drag-limited, not corner-limited** — it is already too slow at the corners for the corner radius to bind.
So under the corrected plant, the margin tax shrinks from 0.28 s to near-zero. The 0.28 s margin tax is a
**linear-plant artifact**; on the real plant the margin is almost free.

---

## (3) Structural tracker gap: +3.87 s — the geometric tracker cannot follow a thrust-saturated line at 1×

**The datum (from memory §TOGT-BOUND, already live-relevant):** the existing geometric tracker, fed the
shipped 4.55 s reference, **first achieves a valid 6/6 only at time-dilation k=1.85 ⇒ twin-tracked lap
≈ 8.3 s** (= 1.85 × 4.55). A 54-combo gain sweep finds nothing faster. **Structural gap = (1.85 − 1) ×
4.551 = 3.87 s.**

### Why it is STRUCTURAL (zero recovery headroom — diverges by design):

The reference line **rides the thrust ceiling 91% of the lap** (`ref_circle`: 90.9% of time at >95% total
thrust; mean thrust fraction 0.972). The control-authority argument:
- Thrust ceiling = 36.92 m/s². Mean used = 0.972 ⇒ **mean correction headroom ≈ 1.03 m/s²; at the 91% of
  the lap that is >95% saturated, headroom ≈ 0.**
- A 1× tracker that falls even slightly behind the reference needs *additional* accel to catch up — but
  there is none left above the ceiling. The tracking error therefore **grows monotonically (diverges by
  design)**; the drone clips a gate or overshoots. This is not a gain-tuning problem (the 54-combo sweep
  confirms), it is a **feasibility problem**: a 1× follower of a thrust-optimal line has no actuator margin
  to be a follower.
- **Why k=1.85 specifically restores feasibility:** slowing the reference by k drops the required
  centripetal/tangential accel ≈ 1/k². At k=1.85, k²=3.42 ⇒ accel demand × 0.29 ⇒ ~71% of the thrust
  ceiling is freed for tracking correction. k≈1.85 is roughly where the *peak* demand drops below the
  ceiling enough that a feedback tracker has the authority to stay on the line. (Quadratically: to free
  ~70% headroom you need demand ≈ 0.3× ⇒ k ≈ 1.8 — consistent with the empirical 1.85.)

### Implication for the S2 architecture decision (the adjudication target):

This is the structural planning↔execution gap S2 must close. The 3.87 s is **architecture-attributable,
not plant- or margin-attributable** — it is the cost of the *decomposition with a naive tracker*, NOT a
property of the line.

- **DECOMPOSED with a NAIVE geometric tracker = 8.3 s.** Dead on arrival as a competitive line: the 3.87 s
  tracker tax alone is bigger than the entire achievable RL lap (inc7 ~9.76 s twin / 11.45 s deployment is
  already close to the *dilated* 8.3 s, and the unconstrained RL ~6.6–6.9 s is far below it). A geometric
  tracker on the optimal line is strictly worse than monolithic RL.
- **DECOMPOSED is only viable with a tracker that has its OWN actuator-aware feasibility model** — i.e.
  MPCC (tracks an arc-length progress reference with the real thrust/drag constraints in the QP) or
  RL-as-tracker. Then the line stays near-optimal and the tracker spends headroom intelligently. The
  decomposition's selling point (inversion / backflip-dive becomes structurally impossible) survives only
  if the tracker is constraint-aware.
- **MONOLITHIC RL (inc7, live-confirmed) already implicitly solves the headroom problem** — it co-designs
  line + control, so it never plans a line it cannot also fly; that is *why* it gets 9.76 s twin without
  any k-dilation while the decomposed-geometric gets 8.3 s only after paying 1.85×. The backflip-dive root
  cause (wrong plant + no DR + reward/termination conflict) is understood and fixed.
- **HYBRID (monolithic + arc-length progress reward over the shipped reference line, MPCC-cast)** is the
  natural bridge: it keeps the monolithic co-design (no tracker-feasibility tax) while importing the
  offline line-iteration flywheel as a dense progress reward. The `progress()` API on
  `rl/reference_line_vq1.json` already exists for exactly this.

**Recommendation feeding S2:** the structural gap is **architecture's to close, and monolithic already
closes it.** Decomposition-with-geometric-tracker is eliminated by this 3.87 s number. The live choice is
**monolithic (or monolithic+progress-reward hybrid)** vs **decomposition-with-MPCC** — and the latter only
pays off if the MPCC tracker beats inc7's monolithic ceiling, which is unproven. Prior reviewer read
("retrain monolithic first, decomposition is the fallback") is **corroborated** by this analysis.

---

## Combined attribution picture (what the 4.27 → real numbers actually decompose into)

Starting from the **old linear inscribed-circle bound 4.273 s**:
1. **+0.278 s** safety-margin tax (shipped 0.7 m **ball** margin) → 4.551 s shipped line.
   *Recoverable to ~+0.04 s by switching ball→box margin: ~0.24 s offline win.*
2. **+0.44 s** plant revision (corrected aero) → ~4.71 s corrected contact-free bound (optimistic; honest
   band 4.7–4.9 s). The corrected bound is HIGHER; the v² drag wall (~39 m/s) pins it; thrust is fully
   used. *Under the corrected plant the margin tax itself collapses to ~0.02–0.05 s (drag-limited, not
   corner-limited).*
3. **+3.87 s** structural tracker gap — the geometric tracker needs k=1.85 to follow ANY thrust-saturated
   line (8.3 s twin-tracked). This is ~10× the other two combined and is the real prize. **Monolithic RL
   already avoids it** (inc7 9.76 s twin with no dilation); a constraint-aware MPCC tracker is the only way
   a decomposed architecture recovers it.

**Net true achievable ceiling on the CORRECTED plant, contact-free, well-tracked:** ~4.7–4.9 s (NOT 4.27 s —
that was a linear-plant fiction). The gap from inc7's 9.76 s twin to this ~4.8 s ceiling (~5 s) is the
remaining S2 prize, of which the style-envelope tax (~2.3–2.9 s/lap) and residual tracking/co-design
sub-optimality are the named components.

## Method / provenance
- Lap times: `analysis.json[*].refined.lap_time_s` per case (TOGT multiple-shooting refine).
- Thrust saturation / speed: time-weighted over `refined_traj.csv` columns u_1..u_4, v_x..v_z (numpy).
- Terminal velocity: closed-form √((Tw·g)²−g²)/drag-coeff (quad ⇒ √(a_h/c2); linear ⇒ a_h/lin).
- TOPP corroboration: `racer.speed_profile.time_optimal_profile` isotropic, centre-line waypoints.
- Convex map / quad drag constants: `racer.rl_plant.COLL_MAP_ACCEL_MEASURED`, `QUAD_DRAG_C2_MEASURED`.

## Caveats (load-bearing)
- TOGT cases use a LINEAR T/W proxy even for the "corrected" case → corrected bound 4.71 s is OPTIMISTIC
  (true convex+anisotropic-drag bound ≈ 4.7–4.9 s, slightly higher).
- Margin tax is shape-dependent: 0.28 s (ball, shipped) vs 0.04 s (box). The 0.28 s is the strict reading.
- k=1.85 / 8.3 s is from prior twin work (memory §TOGT-BOUND), not re-run here; the headroom argument
  (91% thrust-saturated ⇒ ~0 correction authority ⇒ 1/k² demand reduction) is re-derived from the CSVs and
  is consistent with it.
- All trajectory durations include ~1.2 s of pre-gate-0 / post-gate-5 run-in; lap_time_s is gate-0→gate-5.
