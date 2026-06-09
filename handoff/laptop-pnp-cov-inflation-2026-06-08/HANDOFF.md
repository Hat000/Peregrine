# Handoff — PnP world-fix covariance inflation (laptop session, 2026-06-08)

**For: a ShadowPC session** (it has the target run `data/runs/20260607_194615_course_60s`, the
per-gate bundles, the `perception-char-2026-06-08` README/JSON, and `ultralytics`+`torch`). The
laptop made the covariance code change + sized it; the authoritative before/after re-measure needs
the per-gate course bundles, which are NOT on the laptop (`data/runs` is gitignored).

## TL;DR
The `perception-char-2026-06-08` finding: the navigator's χ²₀.₉₉₉ = 16.27 innovation gate catches
~97% of catastrophic gate-fixes (leak ~1.6%) but **also drops ~20% of GOOD fixes** because the
analytic PnP world-fix covariance is ~**3.5× too tight**. Fix shipped here: a scalar inflation on the
analytic 4-corner PnP translation covariance, **`PNP_FIX_COV_INFLATION = 2.0`** (predicted good-fix
rejection 20% → ~2.6%, see sizing below). **355 tests green.** ShadowPC: re-measure on the per-gate
course bundles, confirm leak ≤~2%, pick the knee, set the constant.

## What changed (committed to the working tree, not yet git-committed)
1. **`src/racer/localization.py`**
   - New module constant `PNP_FIX_COV_INFLATION = 2.0` (variance multiplier).
   - `gate_pose_to_world_position(...)` gained `pnp_cov_inflation: float = PNP_FIX_COV_INFLATION`;
     it scales **only the analytic PnP translation block** before propagation:
     `cov = pnp_cov_inflation * (R_wc @ sigma_tt @ R_wc.T)`. The attitude lever-arm term (physically
     calibrated, range-growing) and the no-covariance fallback (0.3 m) are **untouched**.
   - `apply_gate_pose_update(...)` threads the same param (default = the constant).
   - The navigator picks this up automatically — it calls `gate_pose_to_world_position` directly
     (`navigator._process_observation`), so the live χ² gate now sees the inflated R. No navigator edit.
2. **`scripts/characterize_perception.py`**
   - `--cov-inflation FLOAT` (default = production `PNP_FIX_COV_INFLATION`; pass `1.0` for the
     pre-change baseline). Flows into `gate_pose_to_world_position`, so the per-frame `maha` reflects it.
   - New **`GATE TRADE-OFF`** report block: classifies associated fixes as GOOD (`|off|<1 m`) vs
     CATASTROPHIC (`|off|≥3 m`, matches the existing TAIL line) and prints, at χ²=16.27:
     good-fix rejection %, catastrophic leak %, bad-fix catch %. This is the turnkey before/after metric.
3. **`tests/test_localization.py`** — `test_glue_covariance_is_rotated_translation_block` now passes
   `pnp_cov_inflation=1.0` to isolate the raw rotated block; new `test_glue_inflates_analytic_pnp_covariance`
   guards the scaling (and that the fallback is independent of it). Suite: **355 passed**.

## Why inflate the PnP term ONLY (not the whole R)
The gate already MODELS the attitude lever-arm error (the `attitude_noise_std` term). Good fixes are
over-rejected because their error EXCEEDS the modeled cov — and since the attitude part is already in
there, the excess is in the **un-/under-modeled PnP pixel-noise term**: the Fisher-info cov from the
confidence-weighted refine assumes only `WEIGHTED_SIGMA_PX = 1.5 px` i.i.d. corner noise, ignoring
sub-pixel detector bias, heavy-tailed corner localisation, and the 1.5 m gate-size model mismatch.
So the optimistic term is `cov_pnp`; inflating it keeps the attitude weighting physical (matters for
VQ2 vision-only, where the KF isn't anchored to the given pos).

## Sizing (χ², dof=3) — why 2.0
Good-fix maha ≈ f·χ²(3) with f the over-tightness. 20% rejected ⇒ f = 16.27 / χ²₀.₈₀(3) = 16.27/4.642
= **3.51×**. After inflating cov by K (maha → maha/K on PnP-dominated good fixes):

| K   | eff. χ² thr | predicted GOOD-fix rejection |
|-----|-------------|------------------------------|
| 1.0 | 4.64        | 20.0% (current)              |
| 1.5 | 6.96        | 7.3%                         |
| **2.0** | **9.28** | **2.6%  ← committed default** |
| 2.5 | 11.60       | 0.9%                         |
| 3.0 | 13.92       | 0.3%                         |

2.0 clears the <5% goal with margin and relaxes the gate only √2× in distance (catastrophic leak
should stay near the current 1.6%). 2.5 is available if ShadowPC confirms leak headroom.

## Laptop sanity run (NOT the trade-off measurement — read this caveat)
Ran `characterize_perception.py` on the only bundle present on the laptop,
`handoff/shadowpc-followups-2026-06-05/task2_frames` (single off-axis gate, 1.8–23.3 m, RAW odo quat):

```
cov_inflation:   1.0      2.0      2.5
fix maha p50:    113       98       94      (responds to K -> plumbing OK)
GOOD-fix rej:    5/6      5/6      5/6      (STUCK -- see why)
CATASTROPHIC:    0/11     0/11     0/11
```

Confirms the **code path end-to-end** (the flag reaches the gate; maha moves with K). It does NOT and
CANNOT show the good-rejection improvement: task2_frames is **calibration-bias-dominated** (the
~3.5°/range yaw bias → lateral E-offset +0.70 m, clean |fix| p50 = 1.28 m, not the ~0.2–0.4 m course
noise floor). Its "good" fixes carry a *lateral* error, where the cov is dominated by the **attitude**
term (correctly NOT inflated), so PnP inflation barely moves their maha. This is the exact population
the perception-char excluded. The improvement lives in the clean head-on per-gate course fixes — hence
the re-measure must use those bundles.

## ShadowPC: the re-measure (turnkey)
1. Rebuild / locate the per-gate bundles for `data/runs/20260607_194615_course_60s` exactly as in
   `handoff/perception-char-2026-06-08/` (its README + the `scratch/` helpers). One `frames.json`
   bundle per gate.
2. For each gate bundle, sweep K and read the **GATE TRADE-OFF** block:
   ```
   for K in 1.0 1.5 2.0 2.5 3.0; do
     .venv/Scripts/python scripts/characterize_perception.py \
       --bundle <per_gate_bundle_dir> --cov-inflation $K --json char_g<ID>_k$K.json
   done
   ```
   The block prints `N`, `rejected`, `leaked` counts — **sum them across the per-gate bundles** for the
   course-level good-fix-rejection and catastrophic-leak at each K. (Or feed the `--json` dumps to the
   existing perception-char aggregator: the `maha` column now already reflects `--cov-inflation`, so its
   `maha > 16.27` test gives the gated result directly.)
3. Report **before (K=1.0) vs after (K=2.0)**: good-fix rejection (expect ~20% → <5%) and catastrophic
   leak (must stay ≤~2%; currently ~1.6%). Pick the knee; if 2.0 leaves leak headroom and rejection
   you want lower, try 2.5. Set `PNP_FIX_COV_INFLATION` in `localization.py` to the chosen value.
4. Note: `K=1.0` reproduces the pre-change baseline, so it should re-derive the original ~20% / ~1.6%
   (modulo the GATE TRADE-OFF block's thresholds — GOOD `|off|<1 m`, CATASTROPHIC `|off|≥3 m`. If the
   original perception-char used different cut points, align them, but the **before→after DELTA at a
   fixed threshold is the real metric**, and it's robust to the exact cut).

## If 2.0 doesn't cut good-rejection enough on the course bundles
Then the residual good-fix error there is attitude-lever-dominated (not PnP), so the next lever is the
attitude term, not more PnP inflation: bump `NavigatorConfig.attitude_noise_std` (1.0° → ~1.5°) — or,
bluntest, inflate the WHOLE world-fix R (PnP + attitude) so maha scales uniformly by 1/K. Prefer the
attitude knob; whole-R inflation slightly de-weights good distant fixes in the KF (a VQ2 concern).
Don't touch the detector — this is a covariance/estimator change only.

## Pointers
- Gate + cov: `src/racer/navigator.py` `_mahalanobis_position` + `vision_gate_chi2=16.27` (line 166,
  339); `src/racer/localization.py` `gate_pose_to_world_position` + `PNP_FIX_COV_INFLATION`.
- Raw analytic cov source: `src/racer/vision/gate_pose.py` `_refine_pose` (Fisher info from the
  confidence-weighted GN) → `estimate_gate_pose(compute_covariance=True)`.
