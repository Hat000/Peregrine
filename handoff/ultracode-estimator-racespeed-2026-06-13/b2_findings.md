# b2 — ADVERSARIAL verification of A2 (bias-is-binding)

**Role:** charged to REFUTE A2's claim that the gate-4 in-plane bias floor after global
de-bias = **0.517 m** and that "the un-filterable in-plane bias is the binding term."
Recomputed INDEPENDENTLY from raw `course_bundle/frames.json` + `pg/course_g4/frames.json`
+ `characterize_g4.json` / `characterize_course_60s.json`. Seed 20260613, reproduced on re-run.

## VERDICT: **WEAKENED**

A2's **direction is right** but its **headline magnitude (0.517 m) is overstated ~1.5–2.7×**
and its **framing as an un-filterable CONSTANT is wrong** — the term is **range-collapsing**,
not constant. Two of my four attack vectors FAILED (the de-bias is not circular; it is not a
depth-scale artifact), which I report honestly. The decisive hit is on the magnitude/framing
via the range-band decomposition.

| quantity | A2 | b2 (independent) |
|---|---|---|
| in-plane bias, full window (8–27 m) | 0.523 (raw) / 0.447 (debiased) | **0.523**, CI [0.38, 0.67] — REPRODUCED |
| in-plane bias, ≤12 m | 0.337 (A2 table) | **0.337**, CI [0.27, 0.40] — REPRODUCED |
| in-plane bias, **≤9 m (last-usable band)** | *not reported* | **0.191**, CI [0.11, 0.25] |
| in-plane at nearest accepted fix (8.25 m) | — | **0.109** |
| gate-4 normal / in-plane axes | E,D / normal=N | **CONFIRMED correct** |

---

## Attack-by-attack result

### (1) Does the global de-bias GENERALIZE, or is it overfit/circular? — **A2 SURVIVES (my attack fails)**
Leave-one-gate-out (LOGO): I refit the global de-bias from the OTHER 5 gates only, then
applied it to held-out gate-4. If circular/overfit, the residual should collapse.
- full-data de-bias → g4 in-plane residual = **0.517 m**
- **LOGO de-bias (gate-4 excluded) → g4 in-plane residual = 0.620 m (LARGER, not smaller)**.

Holding gate-4 out makes the residual *grow*, so the de-bias is **not circular** and the
gate-4 E offset is genuinely not absorbed by a global constant. A2's structural claim
("per-gate E is not a global constant") is **confirmed by the hold-out, not refuted.** The
per-gate E offsets swing +0.46 (g4) to −0.55 (g3), std 0.38 m — one constant cannot fit them.
*This charge fails; reported in the interest of honesty.*

### (2) Is the gate-4 in-plane (E,D) projection correct? — **geometry CONFIRMED; window-mean critique SUCCEEDS**
Independently from the map quaternion: gate-4 col0 = **+E** (lateral), col1 = **−N** (the
flight-through normal; `normal·flight_dir(g3→g4) = 0.984`), col2 = **+D** (height/down).
Opening centre = `[-135.494, -0.800, 23.996]`. So in-plane = (E, D), along-track = N — **A2's
projection assumption is exactly right.** This attack vector fails on geometry.

**BUT** the binding moment is the **transit** (range → small), not an 8–27 m window mean.
A2's 0.45–0.52 m is the mean over a range window dominated by FAR fixes. Independently
recomputing the gate-4 transit at fid 1020 (range 0.103 m, the literal plane crossing): that
frame **locks onto gate-5, maha 127.8 ≫ 16.27 → REJECTED by the navigator chi2 gate**, so it
is not a usable transit fix. The **last usable accepted gate-4 fix is at range ≈ 8.25 m**
(closer than ~8 m the detector loses 4-corner / locks gate-5 and the fix is gated out). At
that range the in-plane error is **0.109 m**; nearest-3 mean (8.25–8.74 m) = **0.191 m**.
Data cross-validated: `‖off_ned‖ == world_fix_err_m` for 56/56 rows, and `true_range_m`
matches my GT-recomputed range-to-opening for every checked frame.

### (3) Is the residual un-averageable, or does a cal lap / weighting remove it? — **A2 WEAKENED (my attack succeeds)**
The in-plane bias is **monotone in the range cutoff** — the single most damaging finding:

| range cutoff | n | in-plane bias (m) | CI95 |
|---|---|---|---|
| ≤ 9 m | 3 | **0.191** | [0.109, 0.254] |
| ≤ 10 m | 6 | 0.265 | [0.190, 0.348] |
| ≤ 12 m | 12 | 0.337 | [0.269, 0.403] |
| ≤ 16 m | 17 | 0.447 | [0.348, 0.558] |
| ≤ 27 m (A2's window) | 26 | **0.523** | [0.383, 0.672] |

Both E and D shrink monotonically as range decreases (corr(in-plane, range) = **+0.46**;
in-plane reaches 0.69 m at 13–20 m, collapses to ~0.1 m at 8 m). A2 quoted the ≤27 m number
as the headline and stopped its own range table at ≤12 m (0.34) — it **never showed the ≤9 m
band (0.19)**, the actual last-usable-fix regime.

Consequence for filtering: this is **not** a per-fix-constant the KF must swallow whole. It is
**range-correlated**, so (a) a recency/range-weighted R that downweights far fixes — exactly
what the `range_anisotropic_R.py` r⁴ law already does — pulls the operative bias toward the
near-range value, and (b) a determinism-per-track cal lap that fits the range curve removes the
deterministic part. A2's own §4.1 conceded a per-gate cal lap reaches ~0.07 m; the range
structure makes that lever stronger than A2 credited.

### (4) Is it a depth-scale artifact the chi2 gate already removes? — **A2 SURVIVES (my attack fails)**
corr(in-plane error, |range_err|) = **−0.026 ≈ 0** — the in-plane error is **not** explained by
PnP depth-scale error, so it is not a depth artifact and the chi2/range-sanity gate does not
remove it. (The range *collapse* in attack 3 is a geometric/bearing effect, not a depth-scale
effect — corr(D, range) = −0.41 vs corr(in-plane, |range_err|) = −0.03.) The depth-decomposition
A2 leaned on is itself mildly contaminated (its n=7 GT subset includes 2 fixes with maha 17.7
and 22.2 that the live gate would reject, one a D=+1.96 m outlier), but that does not change the
no-depth-artifact conclusion. *This charge fails.*

---

## What this means for the mission
- A2's **bottom line stands in direction**: absolute case-C world-fixing does **not** reach the
  0.05 m bar at gate-4. Even the most favorable transit-band estimate (≤9 m, 0.191 m, CI
  [0.11, 0.25]) is **> the 0.05 m variance bar AND > the 0.155 m gate-4 contact-true margin**.
  Gate-relative observation remains the right architectural fix.
- A2's **headline number (0.517 m) should be corrected to a range-band-dependent 0.11–0.34 m**
  at the transit-relevant ranges (≤9–12 m). Carrying 0.52 m into the margin ledger would
  over-state the absolute-path deficit by ~2–3×.
- A2's **"un-filterable constant" framing is wrong**: the term is range-collapsing, so the
  estimator has real leverage (range-weighted R + cal-lap curve fit) before resorting to
  gate-relative. The 0.155 m margin is *plausibly* reachable at the nearest band with
  range-weighting — a live measurement worth taking, not a foregone "hard wall."

## Caveats / honesty
- The ≤9 m band is **n=3** (CI [0.11, 0.25]); the near-range collapse is real (the monotone
  trend across all bands corroborates it) but the point value is noisy. The conclusion
  "> 0.05 m and > 0.155 m" holds across the entire CI.
- All numbers are from the **5.35 m/s characterization**; at 37 m/s the last-usable-fix range
  and motion blur shift the band, and fewer fixes land in the near band — A2's §4.3 speed-coupling
  caveat is valid and un-refuted here. The transit-band bias must be re-measured at race speed
  (SHADOWPC-VISION-CAL per-gate last-fix-distance task).
- Two of four attacks (circularity, depth-artifact) FAILED — A2 is more robust than the prompt's
  framing suggested. The WEAKENED verdict rests on magnitude + the constant-vs-range-collapsing
  framing, not on a structural error.

## Files
- `b2_verify.py` — all four attacks (run `PYTHONPATH=src .venv/Scripts/python.exe b2_verify.py`).
- `b2_verify_results.json` — machine-readable, deterministic seed 20260613, reproduced on re-run.
