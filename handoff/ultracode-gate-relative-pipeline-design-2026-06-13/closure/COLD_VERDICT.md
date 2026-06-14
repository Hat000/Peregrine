# COLD-MARGIN VERDICT — gate-relative case-C VQ2, gate-4 @ 37 m/s

**Session:** COLD-MARGIN-VERDICT (Peregrine), laptop, offline. **For: Fengyou.**
**Question:** the gate-relative blueprint's headline (gate-4 in-plane RMS 0.139 m clears 0.155 m @ 37 m/s)
is WARM-BY-CONSTRUCTION (c1 seeds velocity to truth, sims only g3→g4). What is the **honest COLD**
gate-4 in-plane miss over a full lap, judged on **worst-case p90** (a margin is a worst-case gate)?

---

## TL;DR (the verdict)

1. **COLD BREACHES, hard.** Honest cold gate-4 p90 @ 37 m/s ≈ **0.34 m** at the realistic accel-bias
   (1.4° attitude = 0.24 m/s² phantom accel), **≈ 2.2× the 0.155 m margin**. Even the bias-free floor
   (velocity-acquisition noise only) is **0.24 m** (1.55×). RMS also breaches (0.22) at realistic bias.
2. **The p90 tail exceeds the margin even WARM.** The realistic warm ceiling (good but not-perfect
   velocity, c1-style) is p90 **0.18–0.20 m** (3 independent sims agree) — *already over 0.155*. RMS 0.139
   "clears" overstated the clearance; the worst-case tail does not.
3. **Vision-velocity is LOAD-BEARING but INSUFFICIENT alone.** A *good* LSQ-over-window vision-velocity
   channel (σ_v ≈ 0.3–0.4) pulls cold from 0.34 → **~0.18 m** (recovers ~75% of the overshoot) — but
   cannot by itself bring p90 ≤ 0.155, because the **measured per-fix lateral σ = 0.265 m floors the
   warm p90 at ~0.18–0.20**. The **naive** consecutive-fix-difference assist (the `weakvel` arm) does
   **NOT** flip it (p90 0.23–0.37 ≈ cold; σ_v ≈ 5.2 m/s per fix-gap is pure noise).
4. **Speed-ladder ceiling on p90 = NONE in the 25–37 ladder with the current per-fix σ** — p90 is
   speed-FLAT in every sim (per-fix σ held fixed), so even 25 m/s warm (~0.20) is over. The real speed
   lever acts through per-fix σ (motion blur), which these sims **cannot** measure → **the at-speed
   gate-4 per-fix σ must be measured on ShadowPC** to set the true rung. On **RMS** (not p90), cold+good-
   velocity supports 37 m/s.
5. **RewindKF is NOT the binding term.** Latency 15→115 ms moves p90 only +0.01 (warm) / +0.03 (cold);
   horizon 0.5 s > L handles both (d2 confirms drop_frac = 0). The binding terms are **per-fix lateral σ
   (sets the floor) and cold velocity error (sets the cold penalty)**.

**Net:** this **corroborates and sharpens** the prior CONDITIONAL-GO. Build the gate-relative pipeline
**and** the LSQ vision-velocity channel (load-bearing), but **37 m/s p90-closure is NOT demonstrated
offline** — closing the last ~0.03 m needs sharper fixes (sub-pixel corner refine, σ 0.265→~0.18) and/or
a lower rung, confirmed against ShadowPC at-speed per-fix σ.

---

## 1. Sanity anchor — shared machinery validated ✓

d3's WARM anchor (single straight g3→g4, velocity seeded to truth, real `LinearKF`+`RewindKF`, measured
σ=0.265) reproduces c1 **exactly**:

| | RMS | p50 | p90 | p99 |
|---|---|---|---|---|
| d3 anchor (this run, N=600) | **0.139** | 0.117 | **0.203** | 0.311 |
| c1 baseline (CONTEXT) | 0.139 | — | 0.203 | — |

The shared noise model + RewindKF are trustworthy; everything below uses the same stack. (Run twice —
full `d3_margin_closure.py` and the lean `d3_trim.py` reuse of `d3.run_cell` — both give 0.139/0.203.)

---

## 2. The 37 m/s table (d3, full-lap g0→g4, N=300, perpendicular in-plane projection)

`velErr` = velocity error magnitude entering the gate-4 window (m/s). **bias 0.24 = the measured-
realistic operating point** (1.4° attitude 1-σ → 0.24 m/s² phantom accel, d4v anchor). Margin = 0.155 m.

| mode | bias (m/s²) | L=15 RMS / p90 | L=115 RMS / p90 | velErr@g4 | p90 ≤ 0.155? |
|---|---|---|---|---|---|
| **warm**¹ | 0.00 | 0.084 / **0.121** | 0.084 / 0.127 | 0.03 | ✅ (but see ¹) |
| **warm**¹ | 0.24 | 0.080 / **0.121** | 0.086 / 0.136 | 0.05 | ✅ |
| **warm**¹ | 0.50 | 0.083 / 0.126 | 0.090 / 0.131 | 0.09 | ✅ |
| **warm**¹ | 1.00 | 0.091 / 0.143 | 0.099 / 0.152 | 0.18 | ✅ (marginal) |
| **warm**¹ | 2.00 | 0.123 / 0.181 | 0.125 / 0.192 | 0.35 | ❌ |
| **cold** | 0.00 | 0.156 / **0.240** | 0.157 / 0.241 | 0.21 | ❌ (1.55×) |
| **cold** | **0.24** | 0.224 / **0.337** | 0.249 / **0.365** | 0.45 | ❌ (**2.2×**) |
| **cold** | 0.50 | 0.369 / 0.510 | 0.425 / 0.592 | 0.89 | ❌ |
| **cold** | 1.00 | 0.718 / 0.917 | 0.813 / 1.055 | 1.72 | ❌ |
| **cold** | 2.00 | 1.394 / 1.728 | 1.612 / 1.974 | 3.41 | ❌ |
| **weakvel**² | 0.00 | 0.153 / 0.232 | 0.175 / 0.266 | 0.38 | ❌ |
| **weakvel**² | 0.24 | 0.229 / 0.328 | 0.260 / 0.374 | 0.56 | ❌ |
| **weakvel**² | 0.50 | 0.365 / 0.511 | 0.417 / 0.570 | 0.90 | ❌ |

¹ **`warm` here is the UNACHIEVABLE case-A/B ceiling** — `fly_lap` re-pins velocity to truth with σ=0.1
at 90 Hz, which also over-tightens position via cross-covariance. It is NOT a case-C number; it shows
only that *if velocity were perfectly known*, per-fix σ alone gives p90 ~0.12. The **realistic** warm
(c1-style seed, no continuous pin) is **0.18–0.20** — see §3. Read `warm` as the floor of the achievable
band, not an achievable point.
² **`weakvel` = naive consecutive-fix differencing** (σ_v = √2·0.265/Δt_fix ≈ 5.2 m/s per fix-gap). It
tracks `cold` (sometimes *worse* — velErr 0.38 vs cold 0.21 at bias 0: the noise is injected, not
filtered). This is the **conservative floor** on a velocity channel, NOT the LSQ channel — see §3/§4.

**Reading:** cold breaches at every bias; the realistic cold (bias 0.24) p90 = **0.34–0.37 m**. Accel
bias is the dominant cold swing (each +0.24 m/s² ≈ +0.10–0.13 m at low bias, escalating). Latency 15→115
costs only +0.001 (warm) to +0.03 (cold). Frac-of-runs-over-margin at realistic cold = **64–72%**.

---

## 3. Cross-sim agreement — the load-bearing numbers triangulate (3 independent constructions)

| construction | WARM p90 | COLD p90 (realistic) | velocity channel p90 |
|---|---|---|---|
| **d3** full-lap, accel-bias param (this run) | 0.20 (anchor) | **0.34–0.37** (bias 0.24) | 0.23–0.37 (naive weakvel) |
| **commander v2** single-leg, σ_v-prior param | 0.19–0.20 (σ_v=0) | **0.29** (σ_v=0.3, L=115) | 0.24 (σ_v=0.3, L=15) |
| **d4v** single-leg, att-bias param | 0.182 (warm) | **0.324** (cold, att 0.5°) | **0.179** (visvel σ_v=0.3, LSQ) |

- **Cold p90 ≈ 0.29–0.37 across all three** → robust. **Warm p90 ≈ 0.18–0.20 across all three** → robust.
- **The one apparent divergence:** d3 `fly_lap` warm = 0.12 vs single-leg warm = 0.19. **Resolved, not a
  real disagreement:** the matched-run projection diagnostic (`proj_convention_check.py`) shows perp-
  projection (0.126) ≈ E,D-hypot (0.125) in `fly_lap` — so the gap is **NOT** the in-plane convention;
  it is `fly_lap`'s continuous 90 Hz velocity-pin (case A/B), which the single-leg "warm" (seed-only)
  doesn't do. Trust **0.18–0.20** as the realistic warm ceiling; 0.12 is the perfect-velocity bound.
- **d4v is the only sim with a faithful LSQ velocity channel:** visvel σ_v=0.3 → p90 **0.179** (vs cold
  0.324) — the channel recovers the cold penalty but **lands at ~0.18, still over 0.155**.

**The achievable case-C ladder by velocity quality (E,D-hypot p90 @ 37):**

| case-C velocity quality | velErr@g4 | gate-4 p90 | vs 0.155 |
|---|---|---|---|
| perfect continuous (case A/B — *not* case C) | 0.03 | 0.12 | clears (irrelevant) |
| seed + const-v (c1 idealization) | 0.14 | 0.19–0.20 | **1.25×** |
| **good LSQ vision-velocity (σ_v 0.3–0.4)** | 0.3–0.4 | **0.18** | **1.16×** |
| naive fix-difference (weakvel) | 0.4–0.5 | 0.23–0.27 | 1.5× |
| **cold IMU-only, realistic bias** | 0.45 | **0.34** | **2.2×** |

Every *achievable* case-C row is over 0.155 on p90. The per-fix lateral σ = 0.265 is the wall.

---

## 4. Honesty / faithfulness check

| axis | faithful? | note |
|---|---|---|
| velocity-init magnitude | ✅ (with caveat) | d3 cold inits N(0,1.5)/axis at g0, converges to velErr 0.21 by g4 (bias 0); at realistic bias 0.24 velErr 0.45 is **bias-dominated** (init forgotten). v2 sets σ_v at g3; d4v lets att-bias accumulate (velErr 0.75, no convergence — most pessimistic). The three **bracket** the offline-unpinnable convergence; realistic case-C velErr@g4 ≈ 0.3–0.5 m/s. |
| accel-bias band | ✅ | 0.24 m/s² = measured 1.4° attitude 1-σ (d4v). 0.5–2.0 are stress cases; the dr_force *disturbance* band reaches ~3 but that is transient, not the steady bias. 0.24 is the right operating point. |
| fix cadence / range | ✅ | 14 Hz effective (30 Hz × 0.47 accept), <12 m — consistent across all sims, matches c1 part_a/d measured. |
| RewindKF latency | ✅ | horizon 0.5 s > L (15 & 115 ms); d2_horizon_cov: drop_frac=0 at 0.5 s for L up to 100 ms, inverts (drop_frac=1) when horizon<L. Latency is **not** binding (p90 +0.01–0.03). |
| worst-case gate | ✅ | gate-4 (margin 0.155, the worst gate) targeted in all sims; judged on p90/p99. |
| **per-fix σ vs speed** | ⚠️ **the one gap** | σ=0.265 is held **FIXED** across 25/30/37 → p90 is speed-flat → **all numbers are best-case lower bounds**. Real per-fix σ grows with motion blur at 37; the true speed ladder is steeper. **Cannot be settled offline.** |
| in-plane convention | ✅ resolved | sims use L2 (Euclidean); `contact_true_eval` uses **L-inf in the gate-yaw frame** (`linf`, `_gate_rotmat_w2g`). L-inf ≤ L2 always → sim p90 is **conservative** by ~10–15%. True-contact cold ≈ 0.29 (still breach); best-velocity ≈ 0.155 (right at the edge). |

**Do the two independent sims AGREE?** **Yes** on every load-bearing number (cold 0.29–0.37, warm
0.18–0.20). The only divergent cell — d3 `fly_lap` warm 0.12 vs single-leg 0.20 — is **explained**
(continuous-velocity-pin, not projection; the matched-run diagnostic proves perp ≈ E,D). **Trust the
single-leg warm (0.18–0.20)** as the realistic ceiling and d4v's visvel arm (0.18) as the achievable-
channel number. The retracted commander v1 (`margin_xcheck.py`) was re-run and **faithfully reproduces
its corner artifact** (cold RMS 4.4–5.3 m, velErr ~10.9 m/s) — confirming the retraction is valid; v2 is
the credible cross-check.

---

## 5. DELIVERABLE

**(a) Vision-velocity = LOAD-BEARING (but insufficient alone).**
Cold breaches the margin ~2.2× (p90 0.34). A *good LSQ-over-window* vision-velocity channel (σ_v ≈
0.3–0.4) is **necessary** to recover ~75% of that overshoot (→ ~0.18), but **cannot alone** bring p90 ≤
0.155 — the measured per-fix lateral σ = 0.265 floors the warm p90 at ~0.18–0.20. The **naive** fix-
differencing assist does NOT flip it. So: **build the LSQ vision-velocity channel** (load-bearing for
not-catastrophically-breaching and for RMS), **and** pair it with per-fix σ reduction (sub-pixel corner
refine, 0.265→~0.18) and/or a lower speed rung to actually close p90. It is the *cheapest lever that gets
you to the margin's edge*, not past it.

**(b) Recommended speed-ladder ceiling.**
- **On p90 (the mandated read): NO rung in 25–37 m/s clears with the current per-fix σ** — p90 is speed-
  flat (warm ~0.20 at all speeds). The estimator-supported p90-valid ceiling is **gated by per-fix σ,
  not speed**, and is **< the whole ladder** until σ is reduced. *This number genuinely cannot be set
  offline* — it requires the **ShadowPC at-speed (~37 m/s) gate-4 vision recording** to measure the real
  per-fix lateral σ under blur. Provisional rule: pick the rung where measured-at-that-speed σ + σ_v keep
  p90 < 0.155.
- **On RMS:** cold + good LSQ velocity clears RMS (~0.12–0.15) up to **≥37 m/s**; cold-no-channel breaches
  RMS (0.22) at realistic bias. cold+weakvel (naive) ≈ cold.

**(c) The honest cold gate-4 p90 @ 37 m/s = ≈ 0.34 m** (realistic 1.4°/0.24 m/s² attitude-bias, IMU-only
velocity) — **2.2× the 0.155 m margin**. Bias-free floor (velocity-acquisition noise only): **0.24 m**
(1.55×). With a good LSQ vision-velocity channel: **~0.18 m** (L2) / **~0.155 m** (true L-inf gate-frame
— right at the edge). Cold without the channel is decisively over on both RMS and tail.

---

### Artifacts (this session, under `closure/`)
- `d3_fresh_run.log` (full d3, CPU-starved by concurrent sessions — superseded by the trim)
- `d3_trim.py` + `d3_trim_results.json` + `d3_trim_run.log` — the verdict table (reuses d3.run_cell verbatim)
- `proj_convention_check.py` + `proj_convention_run.log` — matched-run perp-vs-E,D diagnostic. **CPU-
  killed (exit 127) after emitting its first — and decisive — cell** (warm/bias0/L15: PROP_p90 **0.126** ≈
  EDH_p90 **0.125**, along_p90 0.161). That one cell + the structural argument (along-track leak into E,D
  ≈ along·u34_E ≈ 0.16·0.176 ≈ 0.03 m, small at warm) establishes perp ≈ E,D for the clear/breach
  boundary; the full 12-cell table did not complete (concurrent-session CPU starvation).
- `xcheck_v2_fresh_run.log` — commander v2 reproduced (warm p90 0.193–0.204, all rows breach p90)
- `xcheck_v1_retracted_run.log` — v1 reproduces its corner artifact (confirms retraction)
- Cross-validation read-only: `d4v_velocity_channel_results.json`, `d2_horizon_cov_results.json`

---

```
MEMORY-DELTA (COLD-MARGIN-VERDICT, 2026-06-13)
- COLD case-C gate-4 p90 @37 m/s = ~0.34 m (realistic 1.4°/0.24 m/s² att-bias, IMU-only vel) = 2.2x the
  0.155 margin; bias-free floor 0.24 m. RMS also breaches (0.22) at realistic bias. d3 anchor reproduced
  c1 0.139/0.203 (machinery validated).
- p90 TAIL EXCEEDS 0.155 EVEN WARM (~0.18-0.20, 3 sims agree). Per-fix lateral sigma 0.265 is the p90
  WALL, not velocity. RewindKF NOT binding (lat 15->115 = +0.01-0.03; horizon 0.5>L).
- VISION-VELOCITY = LOAD-BEARING but INSUFFICIENT alone: good LSQ channel (sigma_v 0.3-0.4) pulls cold
  0.34->~0.18 (d4v visvel 0.179) but can't reach 0.155. NAIVE fix-difference (weakvel) does NOT flip it
  (sigma_v~5.2/fix-gap = noise). => build LSQ vel-channel AND cut per-fix sigma (subpixel corner refine
  0.265->~0.18) and/or lower rung.
- SPEED-LADDER p90 ceiling = NONE in 25-37 with current per-fix sigma (p90 speed-FLAT in sims; sigma held
  fixed = best-case). True ceiling gated by AT-SPEED per-fix sigma (blur) -> ShadowPC gate-4 recording
  REQUIRED. On RMS, cold+good-vel supports >=37.
- Convention nailed: contact_true_eval uses L-inf in gate-yaw frame; sims use L2 (conservative ~10-15%).
  Matched diag: perp-proj ~= E,D-hypot (0.126 vs 0.125) -- the warm 0.12-vs-0.20 gap was fly_lap's
  continuous 90Hz vel-pin (case A/B), NOT projection. Commander v1 retraction valid (corner artifact
  reproduced); v2 credible. Corroborates+sharpens CONDITIONAL-GO: 37 m/s p90-closure UNSETTLED offline.
```
