# A_waterfall — Canonical Gap Waterfall (Phase A-SYNTH)

**Date:** 2026-06-13 · **Author:** ultracode A-SYNTH subagent (opus-4.8) · **For:** Fengyou
**Scope:** synthesis only — reconciles A1 (style tax), A2 (planning+structural), A3 (RL gap) into ONE
canonical ledger. Analysis/offline only; no source edits, no sim, no SLURM, no commits.
**Inputs re-verified this pass:** all 11 TOGT `analysis.json` lap times re-read; `expl_corrected_aero`
vmax/thrust-saturation re-computed from `refined_traj.csv`; independent TOPP corroboration re-run via
`racer.speed_profile.time_optimal_profile`; A1/A2/A3 artifact files read in full.

---

## 0. THE CANONICAL WATERFALL (one ledger, deltas SUM)

This is the **monolithic-RL lineage** — the architecture inc7 actually IS, and the one VQ2 will ship.
Every datum is on the **warm-sim / offline-comparable** basis (fresh-sim deployment adds a flat +1.69 s
sim-init artifact, classified PHYSICS-STATE not policy — excluded from attribution, re-attached only for
deployment estimates).

| # | from → to (s) | Δt (s) | source | what closes it |
|---|---|---|---|---|
| **1** | **35.30 → 9.76** | **25.54** | A3 (35.3 dashboard VQ1; 9.76 inc7 eval table) | **already closed** — RL replaced the geometric stack (pathology removal: alt-relay limit cycle, k=1.85 global time-dilation, start-transient, descent-caution, reactive line) |
| **2** | **9.76 → 6.89** | **2.87** | A1/A3 (inc7 → inc5-R1 unconstrained datum) | **envelope relax** — rw_tilt weight ↓ (96→48) + free-cone angle ↑ (60°→75–80°); the R4 tilt-hinge style tax |
| **3** | **6.89 → 4.72** | **2.17** | A2/A3 (unconstrained RL → corrected-aero point-mass bound) | **better planner + better policy** — close the under-driven-thrust slack (RL flies ~50% of optimal speed every segment; ~1.5–1.9 s is genuine policy-vs-point-mass slack, remainder is the point-mass idealization itself) |
|  | **Σ = 30.58 = 35.30 − 4.72** | **30.58** | — | **arithmetically exact (verified)** |

**Endpoint:** corrected-aero point-mass bound **4.72 s** (honest band **4.7–4.9 s**; the 4.72 s is the
*optimistic* edge — convex sub-linear thrust + anisotropic body-down drag on a descending course push the
true bound UP, never down). The shipped 4.551 s line and 4.273 s "bound" are **FALSIFIED-linear-plant
fictions** (vmax 51–52 m/s, physically unreachable; corrected vmax is 39.3 m/s) and are NOT valid
denominators.

```
35.30 s  VQ1 geometric  ┓
                        ┃  −25.54 s  RL collapse (pathology removal)  [CLOSED]
 9.76 s  inc7 RL        ┛
 (style-ON, warm)       ┓
                        ┃  −2.87 s   envelope/style tilt-cap tax       [envelope relax]
 6.89 s  uncon. RL      ┛
 (style-OFF, inc5-R1)   ┓
                        ┃  −2.17 s   planning+policy under-driving      [planner+policy]
 4.72 s  corrected bound┛
 (point-mass, REAL plant, honest band 4.7–4.9 s)
```

---

## 1. RECONCILING THE OVERLAPS (the load-bearing part)

The three workers measured **overlapping** quantities against **different denominators**. The reconciliation
below removes every double-count so the ladder sums cleanly.

### 1a. Style tax: A1's 2.63 s vs the ledger's 2.87 s — NOT a contradiction, a basis difference
- **A1's clean measured tax = 2.63 s** = inc5 rw96 (9.52 s) − inc5-R1 unconstrained (6.89 s). Both on the
  **identical corrected-aero plant, same inc5 lineage** — the cleanest possible apples-to-apples style A/B.
- **The ledger rung 2 = 2.87 s** = inc7 (9.76 s) − inc5-R1 unconstrained (6.89 s). This crosses a lineage
  boundary: **inc7 sits +0.24 s above inc5-constrained** (9.76 vs 9.52; c16-corner-tax + contact-true geom
  lineage). So 2.87 = 2.63 (pure style) + 0.24 (lineage offset).
- **Resolution:** the canonical ledger uses **2.87 s** because the headline waterfall must pass through the
  *real, live-confirmed* inc7 9.76 s datum. The **clean style tax is 2.63 s**; the extra 0.24 s is an
  inc7-vs-inc5 lineage cost that is NOT recoverable by envelope relaxation (it is a different policy). Both
  numbers are correct for their respective denominators — do not average them. **Banked style-tax range
  2.3–2.9 s** brackets both. *No double-count: the 0.24 s lineage offset is attributed to rung 2 as
  "non-style residual carried in the style rung," explicitly flagged below.*

### 1b. A2's 3.87 s structural tracker gap is NOT a rung in this waterfall — it is an ARCHITECTURE side-bracket
This is the single most important reconciliation. **A2's headline 3.87 s does NOT appear as a delta in the
canonical ledger, and that is correct.** Here is why it would be a double-count if inserted:
- The 3.87 s = (k=1.85 − 1) × 4.551 s is the tax a **naive geometric tracker** pays to follow a
  thrust-saturated line (8.3 s twin-tracked vs the 4.55 s line). It is a property of the **DECOMPOSED-with-
  geometric-tracker** architecture.
- **The monolithic RL lineage (inc7) NEVER pays it** — it co-designs line+control, so it never plans a line
  it cannot fly. That is *why* inc7 gets 9.76 s twin with zero k-dilation while the decomposed-geometric
  gets 8.3 s only after paying 1.85×.
- The work the 3.87 s represents — closing the planning↔execution gap — is **already inside rungs 1 and 3**
  of the monolithic ledger (rung 1: RL eliminating the k=1.85 dilation pathology; rung 3: the residual
  planning/tracking slack). Adding 3.87 s as a separate rung would double-count the very dilation that rung
  1 already removed.
- **Where the 3.87 s DOES live:** it is the **S2-architecture adjudication number**, not a waterfall rung.
  It is the cost the *decomposed* path would re-incur if it used a naive tracker — i.e. the reason
  decomposition-with-geometric-tracker is dead-on-arrival (8.3 s > inc7's already-confirmed 9.76 s twin is
  WRONG-way; 8.3 s dilated-geometric is *slower* than unconstrained RL 6.9 s and only marginally faster than
  style-ON inc7). It is a **gate on the architecture choice**, recorded in §3, not a time the monolithic
  path spends.

### 1c. The 8.3 s dilated-geometric floor is a BRACKET on rung 1, not a waterfall node
A3 uses 8.3 s to *bracket* the pathology-removal claim (RL lands +1.46 s above the best the geometric
controller could ever achieve on an optimal line). It is the same 3.87 s structural number viewed from the
geometric side (8.3 = 4.55 × 1.85). It belongs in the **margin/provenance** of rung 1, NOT as a node — the
monolithic ledger goes 35.30 → 9.76 directly, and 8.3 s is the evidence that ~27 s of the collapse is
pathology removal rather than raw-speed gain.

### 1d. A2's margin tax (0.278 s) and plant-revision (+0.44 s) are INSIDE the bound, not separate rungs
- **Margin tax 0.278 s** (ball keep-out) and **plant revision +0.44 s** (linear→corrected) are decompositions
  of the *denominator* (how the 4.27 s linear-fiction relates to the 4.72 s honest bound), NOT additional
  taxes on top of inc7. They explain why the **correct endpoint is 4.72 s** (not 4.27 s). Including either as
  a waterfall rung would double-count: rung 3 already terminates at the corrected 4.72 s bound.
- **Sub-finding worth carrying:** under the corrected (drag-limited) plant the margin tax nearly vanishes
  (~0.02–0.05 s, TOPP) because the plant is too slow at corners for the corner radius to bind — the 0.278 s
  is largely a linear-plant artifact. And ~0.24 s of the linear margin tax is recoverable offline by
  switching ball→box keep-out (legal per-track line-iteration). These are **offline-line-iteration wins
  that lower the *planning* endpoint**, folded into rung 3's "better planner" intervention.

### 1e. Per-segment loss is consistent across A1 and A3 (no conflict)
A1's TOPP per-segment tilt tax (binds on G1→G2, G2→G3, G4→G5) and A3's inc7-vs-bound per-segment loss
(largest at G2→G3 +1.18 s, then start→G0 +0.95 s) describe **different rungs** (A1 = rung 2 style; A3 =
total residual) but agree structurally: the longest straight (G2→G3, 41 m) and the standing-start
(start→G0) dominate, and the loss is **broad** (RL ~50% of optimal speed in every segment) — the signature
of uniform conservatism, not one bad corner.

---

## 2. THE FULL LEDGER WITH PROVENANCE & INTERVENTIONS

| Rung | from→to | Δ (s) | source datum(s) | mechanism | intervention | confidence |
|---|---|---|---|---|---|---|
| **1. Model-based→RL collapse** | 35.30→9.76 | **25.54** | VQ1 dashboard 35.3; inc7 eval 9.76 | pathology removal (k=1.85 dilation, alt-relay limit cycle, start-transient, descent-caution, reactive non-optimal line) | **ALREADY CLOSED** (inc7 live-confirmed) | HIGH on total; MEDIUM on internal pathology split (reasoned, not A/B-measured) |
| **2. Style/envelope tilt tax** | 9.76→6.89 | **2.87** | inc7 9.76; inc5-R1 6.89 (clean tax 2.63 = 9.52−6.89, +0.24 lineage) | R4 tilt-hinge caps a_lat = g·tan(tilt); binds on 3 high-κ corners | **envelope relax** (rw_tilt 96→48 recovers ~1.4 s non-binding; cone 60°→75–80° recovers ~1.5 s on G1→G2/G2→G3/G4→G5) — better-planner-adjacent | HIGH (directly measured) |
| **3. Planning+policy slack** | 6.89→4.72 | **2.17** | inc5-R1 6.89; corrected bound 4.72 (cases/expl_corrected_aero + TOPP) | RL under-drives thrust (~50% of optimal v every segment); ~1.5–1.9 s pure policy slack, remainder point-mass idealization | **better planner + better tracker** (ride thrust ceiling 84% of lap like TOGT optimum; offline line-iteration; ball→box margin frees ~0.24 s) | MEDIUM-HIGH (bracketed: 6.89−4.72=2.17 and 7.13−4.72=2.41) |

**Endpoint band:** corrected-aero point-mass bound **4.72 s** (honest **4.7–4.9 s**). The v² drag wall
pins vmax at ~39 m/s (verified: closed-form v_term=√((8g)²−g²)/0.052=38.7 m/s ≈ CSV 39.3 m/s; at 39 m/s
quad drag = 79 m/s² ≈ the entire 8 g thrust). Doubling the thrust (T/W 3.765→8) made the bound **HIGHER**
(4.27→4.72), not lower — the "real thrust is 2× so the bound is conservative" reasoning is FALSIFIED.

---

## 3. SIDE-BRACKETS (not waterfall rungs — recorded so they are not lost or double-counted)

| Quantity | Value | Why it is NOT a rung | Where it matters |
|---|---|---|---|
| **Structural tracker gap** | **3.87 s** | tax of decomposed-geometric tracker; monolithic never pays it; the work is inside rungs 1+3 | **S2 architecture decision** — kills decomposition-with-naive-tracker |
| Best dilated-geometric floor | 8.3 s | = 4.55 × 1.85; geometric-side view of the 3.87 s | brackets rung-1's pathology-removal claim (RL +1.46 s above it) |
| RL premium over geometric floor | +1.46 s | 9.76 − 8.3; cost of style-box + general-purpose | explains rung 1 is pathology removal, not raw speed |
| Margin tax (ball keep-out) | 0.278 s | decomposes the *denominator* (linear 4.27→shipped 4.55); ~0.24 s offline-recoverable (ball→box) | folded into rung-3 "better planner" |
| Plant revision | +0.44 s | linear-fiction 4.27 → honest 4.72; explains the endpoint | sets the correct denominator (4.72, not 4.27) |
| Drag-wall unrealized potential | ~0.88 s | what thrust *would* unlock if no v² wall (TOPP cap-52 = 3.84 s) | hard physical floor; NOT recoverable |
| Fresh-sim sim-init | +1.69 s | HOME-reset sub-tick spawn-state; PHYSICS-STATE not policy | deployment estimate only (11.45 s = 9.76 + 1.69) |

---

## 4. THE SINGLE BIGGEST LEVER FOR VQ2 RANK

**VQ2 rank = fastest VALID lap.** Rung 1 (25.54 s) is already banked — inc7 is live-confirmed. The live
decision space is rungs 2 + 3, totaling **~5 s** of remaining prize. The biggest *actionable* lever is:

> **THE STYLE-ENVELOPE RELAXATION (rung 2, ~2.3–2.9 s/lap).**

Rationale for ranking it #1 over the planning/policy rung:
1. **Largest single recoverable block** (2.87 s ledger / 2.63 s clean) vs rung 3's 2.17 s, and it is the
   **only rung directly measured** end-to-end (inc5 A/B), so the time is bankable, not modeled.
2. **Lowest-risk, highest-confidence path:** A1's curve shows the high-value recovery lives in the
   **65°→75–80° band (~1.3–1.5 s ideal)**, reachable *without* entering the inc1-inversion-prone
   unconstrained/no-cone regime. The R4 hinge stays intact; only the weight and cone move.
3. **Two independent, composable sub-levers** (rw_tilt weight ↓ recovers ~1.4 s on non-binding segments;
   cone angle ↑ recovers ~1.5 s on the 3 high-κ corners) — a graded ladder, each rung verifiable offline
   before committing, gated only on the gate-4 contact-true margin (0.155 m @ r=0.38) and the POST-GATE-3
   binding-gate risk.
4. **It is already the next queued action** (MEMORY: envelope ladder step 1, rw_tilt 96→48, NOW UNBLOCKED
   post-inc7-live-confirm) — the analysis converts a queued hunch into a quantified, curve-backed plan.

**Secondary lever (rung 3, ~1.5–1.9 s of pure policy slack):** make the policy ride the thrust ceiling
(it currently under-drives to ~50% of optimal speed every segment, worst on the G2→G3 41 m straight). This
is a reward-shaping / planning-quality problem (a progress/speed reward that rewards thrust-ceiling-riding),
and is the natural pairing with the HYBRID architecture (monolithic + arc-length progress reward over the
shipped reference line, MPCC-cast). Higher ceiling than the style lever but lower confidence and higher risk
(it is exactly the regime where the inc1 backflip-dive lived).

**Architecture corollary (the S2 adjudication this feeds):** the 3.87 s structural-tracker bracket
**eliminates decomposition-with-a-naive-geometric-tracker** (8.3 s — slower than even style-ON inc7). The
live choice is **monolithic (inc7 lineage) + envelope ladder + progress reward** vs
**decomposition-with-MPCC**, and the latter only wins if an MPCC tracker beats inc7's monolithic ceiling —
unproven. Prior reviewer read ("retrain monolithic first; decomposition is the fallback") is corroborated.
**Recommendation: bank the style lever first (rung 2), in the monolithic architecture, before touching the
architecture question.**

---

## 5. OPEN UNCERTAINTIES (carry into S2)

1. **inc7 per-gate crossings are PROXY-reconstructed** (inc6 standing transitions + bimodal post-G3
   localization), NOT inc7 debug_obs (ShadowPC-only, gitignored). Two-block pre/post-G3 split is HIGH;
   intra-block per-segment distribution carries ~±0.3 s. → pull inc7 per-tick gate-transition ticks from
   ShadowPC to confirm the G2→G3 +1.18 s localization.
2. **Style-tax basis offset (2.63 vs 2.87 s):** the +0.24 s inc7-vs-inc5 lineage cost is carried inside
   rung 2 but is NOT recoverable by envelope relaxation. Re-measuring the tax on the *inc7 lineage* (inc7
   with envelope OFF) would resolve whether the clean inc7 tax is 2.63 or ~2.87 s.
3. **Corrected-aero bound is the OPTIMISTIC edge (4.72 s).** TOGT models the 8 g as linear T/W 8; the true
   map is convex sub-linear below hover and the drag is anisotropic (0.076 body-down, loaded by the
   descending course). Honest band 4.7–4.9 s; the true bound is slightly HIGHER. A convex+anisotropic TOGT
   re-run would tighten this.
4. **Internal split of the 25.54 s rung-1 collapse** (alt-relay vs k=1.85-dilation vs start-transient vs
   descent-caution) is reasoned apportionment, never A/B-measured at VQ1 level. MEDIUM confidence on the
   split; HIGH on the 25.54 s total and its attribution to pathology removal (the 8.3 s geometric floor
   brackets it).
5. **The ~0.24 s ball→box margin recovery and rung-3 planning wins must be re-verified against the gate-4
   contact-true metric at r=0.38** before shipping — pushing corner speed trades against the 0.75 m
   validity aperture (POST-GATE-3 binding-gate risk, the inc8 risk zone).

---

## 6. METHOD / PROVENANCE (this synthesis pass)
- TOGT lap times: all 11 `cases/*/analysis.json` `togt_refined.lap_time_s` re-read (bound_circle 4.273,
  ref_circle 4.551, ref_margin 4.311, expl_corrected_aero 4.714, sens_thr50 6.421, sens_thr75 4.905, …).
- vmax / thrust-saturation: re-computed from `refined_traj.csv` (v_x..v_z, u_1..u_4): corrected 39.26 m/s @
  97.7% sat; linear 52.35 m/s @ 93.1% sat.
- Independent TOPP: `racer.speed_profile.time_optimal_profile` over spawn+6 gates (164.1 m): corrected
  a=7.95g cap-39 = **4.721 s** (≡ TOGT 4.714, agree 0.007 s); cap-52 = 3.843 s (drag wall = ~0.88 s).
- Waterfall arithmetic: Σ deltas (25.54 + 2.87 + 2.17) = 30.58 = 35.30 − 4.72 (exact, verified in numpy).
- Source findings: A1_style_tax.md, A2_planning_structural.md, A3_rl_gap.md (all read in full).
