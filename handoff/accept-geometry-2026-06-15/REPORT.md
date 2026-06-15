# ACCEPT-GEOMETRY — does camera-pointing extend the fixable band inward?

**Pathway:** P3/P5 worker (accept-geometry). **Date:** 2026-06-15. **Model:** opus-4.8, high effort.
**Mode:** laptop, data-only re-read of already-materialised shadow rows. **NO** sim / GPU / re-fly.
**Branch:** `accept-geometry-2026-06-15`. **Data:** pooled P3 head-on gate-0 (POINTED) rows
`b1_g0_rows.json` (N=6,370) + `b2_g0_rows.json` (N=11,796) = **18,166 frames / 13,308 accepted fixes**,
extracted from `origin/claude/hardcore-lehmann-1de77c` (the rows that fed FORM_RESOLUTION). Surrogate =
`rl/fix_surrogate.py` (baked defaults). Artifacts: `accept_geometry.py`, `close_range_accuracy.py`,
`fit_and_plot.py`, `accept_geometry_curves.png`.

---

## 0. VERDICT

> **Pointing EXTENDS the fixable band inward — but only to ~12 m, NOT to the terminal 6 m.** When the
> gate is centred (head-on), the accept-rate plateau (~0.78 | in-FoV) holds from ~24 m down to **~10.5 m**,
> and *accurate* fixes (lateral MAD ≤ 0.10 m, the binding margin axis) hold down to **~12 m** — vs the
> surrogate's modelled roll-off below 16 m. **The surrogate's <16 m roll-off is largely a POINTING
> ARTIFACT** of the un-pointed inc7 fit (the crab took the gate out of frame), but a **genuine near-field
> PnP floor survives at ~12 m** (4 m lower than the surrogate's 16 m, not zero). Inside ~12 m the lateral
> fix degrades hard; inside ~10 m accept collapses (association); 6–8 m is a dead zone.

**Three deliverables (below):** (1) **YES, pointing extends inward; accurate floor = ~12 m.**
(2) **inc8 lock-window should reach ~12 m, then KF-COAST the last ~12 m — "fix-rate ≥0.50 through the
last 6 m" is geometrically UNACHIEVABLE.** (3) **YES, `accept_rlo` needs re-fit: 16.19 → ~12 m** (safe
single constant; the fuller fix is rlo≈10 + a close-range lateral-σ inflation).

---

## 1. DATA INVENTORY (what's on disk — confirmed)

Every processed frame is a row (the **denominator exists**): full rows carry the PnP chain
(`accepted`, `rel_*`, `world_fix_err_m`, `reproj_px`, `d2_rel`, `n_corners`); sparse rows
(n_det=0 / un-associated → no fix) carry only geometry. Out-of-FoV frames are present (in-FoV = 90 %).
**Range reaches 0.01 m** → the terminal band IS covered.

- **Hover sessions sit at 23.0–23.3 m** (static, el +20°); **the close band (6–16 m) comes ENTIRELY from
  the MOVING approach** (≤ 7 m/s CTBR, el ≈ −1°). All close-range analysis below is **moving-only**.
- The data is **slow (≤ 7 m/s)** — a clean isolation of pointing *geometry* from race-speed motion-blur,
  but it means the floor here is an **optimistic (lower-bound) range floor** for 30 m/s (see §5 caveat).
- `accept | offered = 100 %` at **every** range → **the χ² gate never binds** in this pointed data. The
  funnel narrows at **detection → association** (close range) and a minor `range_ok` depth-sanity dip
  (20–22 m). So "accept-rate" here = the geometric detection/association funnel, not a statistical gate.

---

## 2. POINTING EXTENDS THE ACCEPT BAND INWARD (the headline)

Accept-rate | gate-in-FoV, MOVING approach, vs the surrogate band-pass `accept_prob_in_image`:

| range (m) | N(in-fov) | **emp accept\|fov** | surrogate (rlo 16.2) | **emp / surrogate** |
|---|---|---|---|---|
| 8–10 | 200 | 0.145 | 0.001 | 229× |
| 10–12 | 207 | **0.744** | 0.005 | **160×** |
| 12–14 | 202 | **0.728** | 0.033 | **22×** |
| 14–16 | 204 | 0.779 | 0.196 | 4.0× |
| 16–18 | 202 | 0.782 | 0.583 | 1.3× |
| 18–20 | 210 | 0.710 | 0.794 | 0.9× |
| 22–24 | 899 | 0.792 | 0.836 | 0.9× |

When centred, accept-rate is a **flat ~0.78 plateau from ~24 m down to ~10.5 m**, where the surrogate
predicts a sharp roll-off to near-zero below 16 m. Empirical/surrogate is **22–160×** in the 10–14 m band.
This is the un-pointed-vs-pointed gap: the surrogate was fit on inc7 Track-3 where the **crab also pulled
the gate out of frame** at close range (the surrogate's own note: emp accept ~0.18 at 10–18 m un-pointed);
**pointing takes 10–18 m accept from ~0.18 → ~0.78 (≈4×).** ⇒ the <16 m roll-off is **mostly a pointing
artifact**, not a PnP floor — *down to ~10–12 m*.

---

## 3. BUT THE ACCURATE FIX DIES AT ~12 m (the binding qualifier)

Accept-rate persisting ≠ usable fix. Lateral (cross-track) residual is the binding margin axis
(boresight-closure: gate-4 σ_lat dominates; ceiling ≈ 0.23–0.245 m for r=0.30 @ fr=0.50). Robust scatter
(MAD) of accepted-fix lateral residual, moving:

| range (m) | N(acc) | **lat MAD** | lat median (bias) | reproj px | n_det | regime |
|---|---|---|---|---|---|---|
| 8–10 | 29 | 0.215 | **+0.55** | 1.31 | 4 | accept collapsing + badly biased |
| 10–12 | 154 | **0.31** | **+0.21** | 0.87 | 3 | accept high, **lateral DEGRADED** |
| 12–14 | 147 | 0.075 | +0.02 | 0.63 | 2 | **clean** |
| 14–18 | 317 | 0.055–0.066 | +0.06 | 0.40 | 2 | clean |
| 18–24 | 970 | 0.055–0.074 | +0.06–0.09 | 0.44 | 2–4 | clean (op band) |

- **Lateral MAD is flat (~0.05–0.075) down to 12 m, then jumps to 0.31 (10–12 m) and 0.22 (8–10 m)** with
  a large positive median **bias** (+0.21 → +0.55 m). It crosses the **0.10 budget at ~12 m** and the
  **0.245 margin ceiling at ~11 m**. So fixes at 10–12 m pass the (non-binding) χ² but are **useless for
  the margin** — accept-rate floor ≈ 10 m, **accurate-fix floor ≈ 12 m**.
- **It is a real broad widening, not flip outliers and not corner-clipping:** MAD (robust) is 0.21–0.31,
  frac|cross|>0.2 m = 0.55–0.86; `n_corners=4` at 8–12 m (clipping only onsets at 4–6 m, 65 %); the
  **vertical** axis stays tight (MAD 0.05–0.12 at all ranges) — confirming FORM_RESOLUTION §2's flip-spread
  < 0.024 m. The driver is **near-field weak-perspective PnP + multi-detection** (n_det 3–4 as the gate's
  apparent size grows → mis-association → biased-long depth `pose−true +0.8 m` @ 8–10 m + lateral bias),
  i.e. exactly the "frontal-PnP" mechanism the surrogate named — **but it onsets at ~12 m, not 16 m.**
- 6–8 m = **dead zone** (0 % accept, association collapse); 4–6 m = sporadic 3-corner catches (N=40,
  unreliable). Pointing cannot cross this floor.

---

## 4. DELIVERABLES

### (1) Does pointing extend the fixable band inward? — **YES, to ~12 m.**
- **Accept-rate** plateau extends inward from ~16 m (surrogate) to **~10.5 m** (pointed).
- **Accurate** (lateral MAD ≤ 0.10 m) extends inward from ~16 m to **~12 m** = the actionable floor.
- The surrogate's <16 m roll-off is **largely a pointing artifact** (un-pointed crab lost the gate) **+
  partially a real PnP floor relocated to ~12 m** (not 16 m, not zero). The surrogate over-set it by ~4 m.

### (2) Recommended inc8 lock-window: **reach ~12 m, then KF-COAST — NOT new fixes through 6 m.**
- The reward should drive centring to harvest accurate fixes through the **~12–18 m** band; the lock
  window's **inner edge = ~12 m**, not the terminal 6 m. The last **~12 m (≈0.4 s @ 30 m/s)** must be
  **RewindKF coast on the last good fix**, because **no accurate new fixes exist inside ~12 m** (geometric).
- **This SHARPENS the boresight-closure blocker.** Closure's Lens A: *0 of 5 closing cells survive a
  0.30 s terminal drought*; a 12 m accurate floor at 30 m/s ⇒ a **0.4 s** terminal coast ⇒ **exceeds the
  drought tolerance**. So **pointing alone — even ideal — cannot satisfy "fix-rate ≥0.50 through the last
  6 m / 0.15 s"** (0.15 s @ 30 m/s = 4.5 m range = deep in the dead zone). Terminal-lock must be reframed
  as **"last accurate fix as deep as ~12 m + low-drift RewindKF coast"**; gate-4 closure then hinges on
  **coast quality + at-speed σ**, not on getting fixes closer.

### (3) Does `accept_rlo` need re-fit for pointed geometry? — **YES.**
- **Single-constant change (recommended): `accept_rlo` 16.19 → 12.0**, keep `wlo ≈ 1.0`. This extends the
  band to the **accurate floor** (conservative-correct: it credits pointed 12–18 m fixes the current fit
  zeros out, without crediting the inaccurate 10–12 m fixes the flat σ model would mis-label as clean).
  Band-pass at rlo=12: 10 m→0.10, 12 m→0.42, 14 m→0.74, 16 m→0.81.
- **Do NOT push rlo below 12 without a σ companion.** A free rising-edge fit to the pointed accept-rate
  gives rlo≈9.7 / wlo≈0.31 — but the surrogate's σ model is **flat** (σ_lat floor 0.10, no close-range
  growth), so rlo<12 would make the emulator emit **over-confident** 10–12 m fixes (modelled σ 0.10 vs
  real MAD 0.31) → teaches the policy to over-trust point-blank fixes. The **full two-part fix** =
  `accept_rlo ≈ 10` **+** a close-range lateral-σ inflation (e.g. σ_lat: 0.10 floor → ramp to ~0.3 m below
  12 m, mirroring the MAD curve). Until that σ term exists, **rlo = 12 is the honest single constant.**

---

## 5. CAVEATS

1. **SPEED (the one real limit of this dataset).** All close-range data is ≤ 7 m/s. The ~12 m floor is
   **geometric** (gate apparent size → multi-detection / weak-perspective) so the *range* floor transfers,
   but at 30 m/s motion-blur and pose-extrapolation can only **raise** it → **12 m is an optimistic lower
   bound.** Pinning the race-speed floor needs the **30 m/s gate-4 at-speed recording** (already the
   boresight-closure binding lever #2 — this analysis does not retire it; it makes it more load-bearing).
2. **χ² non-binding here** → I report the geometric detection/association funnel; a deploy estimator with
   a tighter χ² would reject the inflated 10–12 m fixes (good — it would enforce the ~12 m accurate floor
   automatically). Worth confirming the deployed gate is tight enough to reject MAD-0.31 lateral fixes.
3. **Flip immaterial** (vertical MAD tight at all ranges) — independently confirms FORM_RESOLUTION §2; the
   close-range failure is lateral/association, not the IPPE planar flip.
4. **Minor wrinkle:** a mid-band `range_ok` dip at 20–22 m (accept|fov 0.44, range_ok 52 %) in the moving
   approach — a depth-sanity transient (pose under-reads ~2–4 m there), low-N, **not** close-range and not
   load-bearing for this verdict.

---

## MEMORY-DELTA (≤10 lines — for the overall commander to bank)

```
- ACCEPT-GEOMETRY RESOLVED (P3/P5 worker, 2026-06-15; pooled head-on POINTED gate-0, N=18,166/13,308 acc,
  MOVING ≤7 m/s): POINTING EXTENDS THE FIXABLE BAND INWARD but only to ~12 m, NOT the terminal 6 m.
  Accept|in-fov holds ~0.78 plateau down to ~10.5 m (surrogate band-pass rlo=16.2 says ~0 at 12-14 m ->
  emp/sur 22-160x) => surrogate <16 m roll-off is MOSTLY a POINTING ARTIFACT of the un-pointed inc7 fit.
- ACCURATE-fix floor = ~12 m (lateral MAD flat 0.05-0.075 to 12 m, then 0.31 @10-12 m, +0.21..+0.55 bias;
  crosses 0.245 margin ceiling at ~11 m). Real near-field PnP floor (multi-detect n_det3-4 + weak-perspective
  + biased-long depth), NOT flips (vert MAD tight, confirms FORM_RESOLUTION) and NOT clipping (nC=4 >6 m).
  6-8 m dead (assoc collapse). chi2 non-binding (accept|offer=100%; funnel narrows at detection/assoc).
- inc8 LOCK-WINDOW: reach ~12 m then KF/RewindKF COAST the last ~12 m; "fix-rate>=0.50 through last 6 m/
  0.15 s" is GEOMETRICALLY UNACHIEVABLE (0.15 s @30 m/s = 4.5 m = dead zone). SHARPENS boresight-closure:
  12 m floor @30 m/s = 0.4 s coast > the 0.30 s drought that broke all closing cells -> pointing ALONE can't
  satisfy terminal-lock; closure hinges on coast quality + at-speed sigma, not on closer fixes.
- SURROGATE accept_rlo NEEDS RE-FIT for pointed geom: 16.19 -> 12.0 (one-constant, keep wlo~1.0; conservative
  = accurate floor). Do NOT go below 12 w/o a companion close-range lateral-sigma inflation (flat sigma would
  over-credit 10-12 m); full fix = rlo~10 + sigma_lat ramp 0.10->~0.3 below 12 m. NOT applied here.
- CAVEAT: data <=7 m/s; 12 m floor is geometric (transfers) but at-speed only RAISES it -> 12 m optimistic;
  pin via the 30 m/s gate-4 at-speed recording (boresight-closure lever #2 made MORE load-bearing, not retired).
  -> [[index-vision-estimator]] [[index-rl-training]] [[index-control-sim]]
```

### NEW parked item surfaced UP (for the commander's register)
- **Deployed χ² tightness check:** in this pointed data `accept | offered = 100 %` (χ² never binds) yet
  10–12 m fixes carry lateral MAD 0.31 m. Confirm the *deployed* relinnov χ²(2,.999)=13.82 gate actually
  rejects these inflated near-field fixes — if it doesn't, the terminal coast could be seeded by a bad
  ~11 m fix. Revive-trigger: any terminal-gate-lock / near-field-fix work in inc8.
