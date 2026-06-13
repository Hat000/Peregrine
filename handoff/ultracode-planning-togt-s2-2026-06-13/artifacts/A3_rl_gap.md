# A3 — Model-based → RL gap + residual RL-vs-bound attribution

**Date:** 2026-06-13 · **Agent:** A3 (Phase-A3 worker, ultracode S2-planning fan-out) · For: Fengyou
**Scope:** analysis-only; no source edits, no live sim. All numbers re-derived from repo data.
**Basis discipline:** every lap time below is on the **warm-sim / offline-comparable** basis. Fresh-sim
deployment adds a flat **+1.5–1.7 s sim-init artifact** (NOT policy) — kept out of the attribution and
re-attached only at the very end.

---

## 0. The numbers, sourced

| datum | value | source (verified) |
|---|---|---|
| VQ1 geometric valid finish | **35.3 s** | `MEMORY.md` (dashboard-confirmed) |
| inc7 RL offline median (style-constrained, warm) | **9.76 s** | `project_rl_increment_history.md` inc7 eval table |
| inc7 fresh-sim deployment lap | **11.45 s** | shadowpc-inc7-live; = 9.76 + 1.69 sim-init |
| best-geometric (optimal line, time-dilated k=1.85) | **8.3 s** | TOGT-BOUND writeup; 54-combo gain sweep finds nothing faster |
| inc5 style-constrained (rw_tilt=96) | **9.52 s** | inc5 writeup |
| inc5 Round-1 UNCONSTRAINED (envelope OFF) | **6.89 s** | inc5 writeup ("unconstrained speed datum") |
| estimated full-unconstrained potential | **~6.6 s** | tilt-concentration ladder step-3 |
| shipped reference line (falsified linear plant) | **4.551 s** | `rl/reference_line_vq1.json` lap_time_s (re-read) |
| TOGT contact-free bound (falsified linear plant) | **4.273 s** | cases/bound_circle/analysis.json |
| **corrected-aero TOGT bound (REAL plant)** | **4.714 s** | cases/expl_corrected_aero/analysis.json refined.lap_time_s |
| corrected-aero vmax (v² drag wall) | **39.3 m/s** | refined_traj.csv (re-computed) |
| thrust-50% TOGT (proves thrust binds) | 6.421 s | cases/sens_thr50 |
| thrust-75% TOGT | 4.905 s | cases/sens_thr75 |

All TOGT case lap times independently re-read from `analysis.json` files; reference-line crossings and
speeds re-computed from the JSON. Corrected-aero per-gate timing re-computed from `refined_traj.csv`.

---

## 1. THE TOP-LEVEL CHAIN

```
VQ1 geometric            35.30 s
inc7 RL offline           9.76 s   ── model-based→RL collapse = 25.54 s
   (style-constrained, warm-sim)
corrected-aero bound      4.72 s   ── residual RL→bound = 5.04 s
   (point-mass, REAL plant, physically valid optimum)
```

Fresh-sim deployment 11.45 s = 9.76 + **1.69 s sim-init artifact** (HOME-reset sub-tick spawn-state,
classified PHYSICS-STATE not policy; bimodal-char writeup). This is added back ONLY for deployment
estimates; it is excluded from all attribution because it is not a property of the policy.

---

## 2. MODEL-BASED → RL: attributing the ~25.5 s collapse

The 35.3 s VQ1 is the **old geometric controller** (CTBR + reactive line + alt balloon). The RL replaces
the whole stack. The collapse is **elimination of geometric-controller pathologies**, not "RL is a faster
flyer than the geometry allows." Proof: the geometric controller's OWN best achievable — its optimal
TOGT line, time-dilated by k=1.85 just to stay stable, with a 54-combo gain sweep finding nothing
faster — is **8.3 s**. inc7 lands at 9.76 s, i.e. *just above* that floor. So the RL did not out-fly the
geometric ceiling; it removed the ~27 s of pathology that forced 35.3 s, and did so autonomously+robustly.

**Pathology layer: 35.3 → ~8.3 s = −27.0 s (~100% of the collapse).** Sub-components (order-of-magnitude;
these were never independently A/B-measured at the VQ1 level, so the split is a reasoned apportionment, not
a measured decomposition):

| pathology subsumed by RL | mechanism | rough share |
|---|---|---|
| **k=1.85 global time-dilation** | reactive geometric tracker needs 1.85× slowdown everywhere just to stay stable on a feasible line; RL flies the line at 1× | dominant — roughly the 1.85× factor on the trackable lap (≈4.55→8.3 s is the dilation; the rest of 35.3 is below) |
| **alt-relay limit cycle + descent caution** | delay-driven thrust balloon oscillation, always-on; course descends 26 m so vertical authority is load-bearing; geometric stack throttled vertical aggression | large; the gap from 8.3 s "trackable" up to 35.3 s "shipped VQ1" is dominated by reactive-line + alt-relay + descent conservatism |
| **start transient** | rate-clamp saturation / tick-phase dice-roll at launch (later fixed at control level via launch_ramp_s=0.6) | small (seconds of settle) |
| **reactive (non-time-optimal) line geometry** | VQ1 line was reactive-planner, not TOGT-optimal; longer path + slower cornering | medium |

**Net:** ~25.5 s of the 25.54 s collapse is the RL **subsuming geometric-controller pathologies**
(time-dilation, alt-relay, start-transient, descent-caution) and replacing a hand-tuned reactive stack with
a single learned end-to-end policy. RL pays a **+1.46 s premium** over the 8.3 s dilated-geometric floor
(9.76 vs 8.3) — the cost of being style-boxed (rw_tilt 96 / 60° cone) and general-purpose rather than a
per-track hand-tuned k-sweep. That premium is bought back in the residual (style relaxation).

**Confidence:** HIGH on the top-line 25.5 s and on the *attribution to pathology-elimination* (the 8.3 s
geometric floor is a measured datum that brackets it). MEDIUM on the *internal split* among the four
pathologies — those sub-shares are reasoned apportionment, not independent VQ1-level measurements.

---

## 3. RESIDUAL: inc7 9.76 s → corrected-aero bound 4.72 s (−5.04 s)

The residual decomposes into a **style** rung (hand to A1) and a **planning+tracking+policy** rung
(hand to A2), against the **corrected-aero** point-mass bound (4.72 s — the physically valid optimum;
the shipped 4.55 s line and 4.27 s bound are on the FALSIFIED linear plant and are NOT valid optima).

```
inc7 offline                              9.76 s
  − STYLE envelope tax           −2.63 s  →  7.13 s   [A1] (measured: inc5 9.52 − 6.89)
                                                       (≈ the ~6.9 s unconstrained datum; consistent)
  − PLANNING + TRACKING + POLICY  −2.41 s  →  4.72 s  [A2]
corrected-aero point-mass bound           4.72 s
```

Cross-check on the [A2] rung: **6.89 unconstrained − 4.72 bound = 2.17 s** (using the directly-measured
unconstrained datum instead of the style-subtracted estimate). So the planning+tracking+policy residual is
**~2.2–2.4 s**, bracketed by the two independent paths.

### 3a. Style tax (−2.63 s) → A1
Measured, not modeled: inc5 rw_tilt=96 gave 9.52 s; envelope OFF (Round-1) gave 6.89 s on identical
corrected-aero plant. Tilt-concentration per-segment table attributes this 2.91 s (single-trajectory
variant) to: start→G0 +0.73 (global conservatism, cap doesn't bind), G2→G3 +0.67, G1→G2 +0.47,
G4→G5 +0.37 (these three the cap physically binds), G0→G1 +0.40, G3→G4 +0.27. a_lat = g·tan(tilt):
60°→17, 65°→21, 75°→37, 80°→55.7 m/s². Hand this to A1.

### 3b. Planning + tracking + policy-suboptimality (−2.2 to −2.4 s) → A2
This is the gap from the best **unconstrained RL** (6.6–6.9 s) to the **point-mass time-optimal**
(4.72 s). It contains (a) the structural tracking gap (the 8.3 s dilated-geometric vs 4.55 s line gap is
the geometric tracker's version of this; RL's monolithic policy closes most but not all of it), and (b)
pure RL policy-suboptimality — the policy flies at roughly **half** the optimal speed in every segment
(see §4), so even unconstrained it does not ride the thrust ceiling 84% of the lap the way the TOGT
optimum does. Hand this to A2. **What is left as PURE RL policy-suboptimality** (not planning, not style):
the unconstrained RL ~6.6 s already has the *geometry* of a near-optimal line (it's monolithic), so most of
the 6.6→4.72 = ~1.9 s is the policy not pushing thrust to the ceiling / not cornering at the optimal speed
— i.e. **~1.5–1.9 s of genuine policy-vs-point-mass-optimum slack**, with the remainder being the
point-mass idealization itself (the bound ignores attitude-slew transients the real policy must pay).

---

## 4. PER-SEGMENT: where inc7 loses the most time

inc7 standing-start per-gate crossings are reconstructed from the **inc6 standing-start proxy** (the only
logged standing-start RL per-gate timing; 30 Hz, transition ticks G0@1.97 s, G1@3.43 s, G2@4.90 s — pre-G3
is warmup-insensitive per bimodal-char) plus the bimodal-char finding that the post-G3 segment carries the
back-half time. G3 crossing reconstructed at ~7.1 s (G2→G3 = 41 m at ~18–20 m/s ≈ 2.2 s; gate-3 crossing
speed 17.4 m/s is a logged live fact). Post-G3 (gates 4,5) split by the corrected-aero proportion.

**inc7 RL (9.76 s) vs corrected-aero bound (4.72 s) — the physically valid optimum:**

| segment | bound (s) | RL (s) | Δt (s) | % of gap | bound v | RL v |
|---|---|---|---|---|---|---|
| start→G0 | 1.020 | 1.97 | +0.95 | 18.8% | 22.8 | 11.8 |
| G0→G1 | 0.638 | 1.46 | +0.82 | 16.3% | 37.6 | 16.4 |
| G1→G2 | 0.750 | 1.47 | +0.72 | 14.3% | 40.0 | 20.4 |
| **G2→G3** | 1.016 | 2.20 | **+1.18** | **23.5%** | 40.4 | 18.6 |
| G3→G4 | 0.653 | 1.36 | +0.71 | 14.0% | 36.8 | 17.7 |
| G4→G5 | 0.641 | 1.30 | +0.66 | 13.1% | 37.4 | 18.4 |
| **TOTAL** | 4.718 | 9.76 | **+5.04** | | | |

**Two-block view:** PRE-G3 (start→G3) bound 3.42 s / RL 7.10 s = **+3.68 s** ; POST-G3 (G3→G5) bound
1.29 s / RL 2.66 s = **+1.37 s**.

**Findings:**
- The loss is **broad, not localized**: RL flies at **~50%** of optimal speed in EVERY segment
  (RL ~12–20 m/s vs bound ~23–40 m/s). This is the signature of a uniformly conservative policy
  (style envelope + general-purpose margins), NOT a single bad corner.
- **Largest absolute loss is G2→G3 (+1.18 s, 23.5%)** — the longest segment (41 m), where the optimum
  reaches 40 m/s and RL only ~19 m/s. The straight where thrust should ride the ceiling longest is where
  the conservative policy leaves the most on the table.
- **Start→G0 (+0.95 s, 18.8%)** is the second-largest: standing-start acceleration deficit (RL 11.8 m/s
  avg vs 22.8 m/s optimal) — the policy ramps thrust cautiously off the line.
- Per-segment Δt is dominated by the *speed deficit*, consistent with §3's conclusion that the residual is
  uniform under-driving (thrust not at ceiling), not a geometry/line problem.

**Caveat (load-bearing):** the per-gate RL crossings are PROXY-reconstructed (inc6 standing transitions +
bimodal post-G3 localization), not inc7 debug_obs (those live only on ShadowPC, gitignored). The two-block
split (pre/post-G3) is well-grounded; the *intra-block* per-gate distribution carries ~±0.3 s uncertainty
per segment. Direct confirmation needs inc7 per-tick gate-transition ticks pulled from ShadowPC.

**Note on the reference line (4.55 s):** comparing inc7 to the *shipped reference line* (G0 1.33, G1 1.90,
G2 2.49, G3 3.37, G4 3.97, G5 4.55) inflates the gap because that line is a TOGT optimum on the **falsified
linear plant** (vmax 51 m/s — physically unreachable; corrected-aero vmax is 39 m/s). The mid-course
reference speeds (47–51 m/s at G1/G2) cannot be hit on the real plant. The **corrected-aero bound (4.72 s,
vmax 39 m/s)** is the honest denominator and is what the table above uses.

---

## 5. HANDOFFS

- **→ A1 (style):** style envelope tax = **−2.63 s** (measured, inc5 9.52−6.89). Per-segment structure in
  tilt-concentration table (binds on G1→G2, G2→G3, G4→G5; start→G0 is global-conservatism not cap-bind).
- **→ A2 (planning+tracking):** unconstrained-RL → point-mass-optimum = **−2.2 s** (6.89→4.72) of which
  ~1.5–1.9 s is policy-vs-optimum slack (under-driven thrust) and the remainder is the point-mass
  idealization (ignores attitude-slew the policy must pay). The 8.3 s dilated-geometric vs 4.55 s line gap
  is the geometric-tracker analogue of this same structural-tracking gap.

---

## 6. CAVEATS / CONFIDENCE

- Top-line collapse **25.5 s** and residual **5.04 s**: HIGH (load-bearing datums re-read from source).
- Internal split of the 27 s pathology layer (alt-relay vs dilation vs start-transient vs descent):
  MEDIUM — reasoned apportionment, never A/B-measured at VQ1 level.
- Style tax −2.63 s: HIGH (directly measured, inc5).
- Planning+tracking −2.2 s: MEDIUM-HIGH (bracketed by two independent paths: 6.89−4.72=2.17 and
  7.13−4.72=2.41).
- Per-gate RL splits: MEDIUM (proxy-reconstructed; pre/post-G3 two-block split is HIGH, intra-block ±0.3 s).
- Corrected-aero bound 4.72 s is "exploratory" (v² drag extrapolated beyond 7.6 m/s); ceiling robust
  4.3–4.7 s. Using 4.72 as the denominator is the conservative (slowest-bound) choice within that band.
- Everything is point-mass for the bound; the real policy must also pay attitude-dynamics transients the
  bound ignores, so the "pure policy-suboptimality" number is an upper bound on recoverable time.
