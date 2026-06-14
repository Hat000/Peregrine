# FIX-SURROGATE — analytic no-render fix model for inc8 training (2026-06-14)

**What.** An analytic surrogate for the vision→fix chain: given GROUND-TRUTH relative gate geometry
(range, bearing, in-FoV, viewing angle — all **free in the sim**) it returns `P(accept)` and a fix
covariance `σ(geometry)`, so a synthetic gate-relative position fix `(z_ned, cov_ned)` can be sampled
and fed straight to `racer.state_estimator.LinearKF` **without rendering the detector** at
thousands-of-envs training scale. This is the enabler for inc8's camera-pointing training.

**Why.** Camera pointing is the binding lever (inc7's 64° crab aims the camera off the gate → only
~33 % of frames see any gate, ~7 % yield a fix). inc8 must train the policy to keep the gate in the
camera, but the YOLO detector can't be rendered in the RL loop. The surrogate replaces the render with
a calibrated map `geometry → (accept?, fix, cov)`.

**Scope.** This is **infrastructure, not a reward.** It answers *"if the camera is pointed here, do we
get a fix and how good is it?"* — nothing about how to reward that. **No reward is designed or
implemented here** (Fengyou owns that). It does **not** import or modify any estimator/deploy file
(`state_estimator.py`, `localization.py`, `navigator.py`, `kf_rewind.py`, `fly_rl.py` are untouched).

Module: [`rl/fix_surrogate.py`](../../rl/fix_surrogate.py) · tests: [`tests/test_fix_surrogate.py`](../../tests/test_fix_surrogate.py) · 15/15 green, CPU-only, torch-free.

---

## 1. Calibration source & substrate

Calibrated to the **Track-3 vision-at-speed shadow recordings** (`handoff/simops-mastery-2026-06-13`,
6 flights `cr1b,std1..std5` / **1821 frames / 126 accepted fixes** — the navigator's real
detect→associate→PnP→localize chain run in shadow on inc7 given-pose at-speed flights).

[`build_dataset.py`](build_dataset.py) reuses the canonical bundle reader (`analyze_shadow_vision.py`)
and the in-loop projection (`predict_gates_in_camera` + `racer.frames`) — it does **not** reinvent
either. It emits per-frame GT geometry to the active gate + per-(frame,gate) candidates + labels, and
**reproduces the published funnel EXACTLY** (the calibration anchor):

| Anchor | Track-3 | dataset | ✓ |
|---|---|---|---|
| funnel | 1821→1644 det→417 assoc→127 offered→**126 accepted** | identical | ✓ |
| per-frame accept-rate | 0.069 | 0.069 | ✓ |
| any-gate-in-image coverage | 605/1821 = **0.332** | 0.332 | ✓ |
| in-image min-bearing | p50 33.5° / p90 46.1° (HFoV½ 45°) | — | — |

Heavy derived arrays are gitignored (regenerable via `build_dataset.py`); `data/summary.json` +
the fitted `models/*.json` checkpoints are tracked.

---

## 2. Sub-models (fitted, checkpointed, individually swappable)

Fitter: [`fix_surrogate_fit.py`](fix_surrogate_fit.py) (pure numpy/scipy, deterministic). Each sub-model
is a `models/*.json` checkpoint; the module bakes the same values as defaults and can reload the
checkpoints (`FixSurrogate.from_checkpoints()`) so a recalibration is a one-file swap.

### A. `accept` — P(accept | gate in image) = a range **BAND-PASS**

The measured accept-rate vs range is a sharp band-pass (rejects very-close frontal-PnP flips / partial
gates, and caps range at ~26 m), so a logistic (even quadratic) **cannot** fit it against the huge
far-range reject mass. Primary model:
`p = pmax·σ((r−rlo)/wlo)·σ((rhi−r)/whi)`, gated by `in_image` (the deterministic camera-pointing gate),
then hard-zeroed outside a generous valid-range guard `[9, 35] m`.

Fitted: **pmax 0.842, rlo 16.2, rhi 28.1, w 1.0**. Reproduction (in-image candidates):

| range band | n | emp accept | band-pass | (logit, rejected) |
|---|---|---|---|---|
| [0,10) | 158 | 0.025 | 0.000 | 0.291 |
| [10,18) | 33 | 0.182 | 0.210 | 0.257 |
| **[18,24)** | 109 | **0.835** | **0.826** | 0.245 |
| [24,32) | 29 | 0.690 | 0.718 | 0.185 |
| [32,∞) | 1067 | 0.003 | 0.000 | 0.035 |

Marginal P(accept\|in-image) 0.089 → band-pass marginal 0.085. Rank-ordering AUC (diagnostic) 0.867.
**The peak 0.835 was adversarially confirmed trustworthy** (de-biasing to one-gate-per-frame leaves it
bit-identical 0.8346; it is **not** a per-frame single-lock artifact — see §4).

### B. `sigma` — per-fix 1-σ vs range (gate-frame), MEASURED Track-3 noise

The MEASURED vision-at-speed noise (NOT the conservative 0.265 m the estimator ships). Floor = STD over
the de-tailed 10–26 m binding band; `σ(r) = max(floor, a1·r)`.

| axis (gate frame) | floor | bootstrap CI68 | a1 (m/m) | bias | **target** | ✓ |
|---|---|---|---|---|---|---|
| lateral (in-plane cross-track) | **0.104** | [0.094, 0.114] | 0.0028 | −0.034 | ~0.10 | ✓ |
| vertical (in-plane) | 0.282 | [0.260, 0.299] | 0.0 | +0.195 | ~0.24–0.41 | ✓ |
| depth (along-track) | **0.852** | [0.791, 0.900] | 0.0 | −0.334 | ~0.8 | ✓ |

Covariance shaping **mirrors the deployed gate-relative fix** (`localization.gate_relative_inplane_fix`):
a gate-plane anisotropic diagonal `diag(σ_lat², σ_vert², σ_depth²)` rotated to world NED → strictly SPD.

### C. `crab` — camera-pointing → gate-in-FoV → accept-rate (the inc8 reward lever)

Per-frame counterfactual: hold the trajectory fixed, ask *"if the policy flew at target crab C instead
of its measured crab, where would the active gate sit in frame?"* The active-gate azimuth tracks −crab;
elevation (a pitch quantity) is unchanged by yaw.

**Headline finding — ELEVATION co-binds, not just azimuth.** At the fixable (offered) range the active
gate is in the HORIZONTAL FoV after de-crabbing (`az_ok 1.00`) but in the VERTICAL FoV only **~10 %**
of the time (`|el|` p50 **44.6°** vs VFoV half 29.4° — the +20°-up-tilted camera misses the gate in the
steep transit posture). So yaw/crab alone is a *modest, elevation-limited* lever
(active-in-FoV 0.058 @ crab 66° → 0.144 @ crab 0°).

**The real prize is 2-axis pointing.** If the camera held the active gate **centred (az AND el)** through
its 18–28 m approach window, per-frame accept → the band-pass peak **0.84** vs the current in-window
**0.099** — a **×8.5 in-window fix-density gain**. That is the lever the camera-pointing reward should
target; the binding axis is **elevation (pitch / camera-tilt)**, with yaw/crab secondary.

Provenance anchors reproduced: any-gate coverage 0.332 / accept 0.069 (= inc7).

### D. `cv` — leave-one-flight-out + adversarial: calibrated, NOT overfit

| check | result |
|---|---|
| accept band-pass — held-out AUC | **0.862** (vs in-sample 0.867) |
| accept band-pass — held-out peak-window MAE | 0.059 |
| σ lateral floor — held-out STD | mean **0.098** / spread 0.018 (vs target 0.104; CI68 [0.094,0.114]) |
| σ depth floor — held-out STD | mean 0.842 / spread 0.081 (vs target 0.852) |

---

## 3. CPU consumption smoke (KF-ready)

`sample_fix(geom, rng) → (z, cov) | None` feeds `LinearKF.update_position` directly. Over **20 000**
zero-bias draws at a peak-range geometry:
* SPD + finite: 20000/20000 (every covariance strictly positive-definite, no NaN);
* **covariance self-consistency: mean Mahalanobis = 3.00 = dof**, p95 7.79 vs χ²₀.₉₅(3) = 7.81 — the
  sampled scatter exactly matches the reported covariance (the strongest KF-readiness guarantee);
* KF stays SPD/finite after applying the fix; empirical accept fraction matches `p_accept` to <0.02.

---

## 4. Adversarial verification (4 independent agents — workflow `fix-surrogate-adversarial`)

| probe | verdict | finding → action taken |
|---|---|---|
| **overfit / k-fold + LOFO** | concern | No genuine overfit (pmax 0.84±0.014 generalizes; pure-noise null control consistent with ONE shared distribution). BUT floors rest on only ~16–24 clean rows/flight → ±15–18 % sampling CI. **→ added bootstrap CI68 + per-flight n to `sigma.json`; relabeled cv.json STD vs MAD measures.** |
| **range-extrapolation** | pass | Band-pass decays cleanly (6e-6 @ 40 m); no σ unphysical; lateral a1 dormant (knee 37.8 m). **→ added the explicit valid-range guard `[9,35] m` to the module.** |
| **crab-independence** | concern | The 0.098 `P(accept\|in-image)` is a RANGE-MARGINAL lower bound; in the 18–28 m band accept-given-in-image is ~0.82 (~8×). **→ `crab_to_fix_rate` accept_rate documented as a relative slope/lower-bound; added `accept_density_in_window` (=band-pass peak 0.84) as the reward-relevant density; the ×8.5 gain already used the peak, so it stands.** |
| **competition-artifact** | pass | The 0.835 peak is a trustworthy per-gate accept (one-gate-per-frame de-bias → bit-identical). **→ corrected the mechanism wording: accept ≈ in-image→offered(association/depth/range-cap) gated, NOT lock-competition.** |

---

## 5. How inc8 calls it

```python
from rl.fix_surrogate import DEFAULT as SURR, geometry   # or FixSurrogate.from_checkpoints()

# In the training loop, per env, per active gate (all GT, free in the sim):
geom = geometry(drone_pos_ned, R_world_body, gate)        # range, azimuth, elevation, in_image, ...

# (a) shape a reward / curriculum on the camera-pointing signal (reward design is YOURS):
p = SURR.p_accept(geom)                                   # in_image-gated range band-pass
in_fov_frac, accept_rate = SURR.crab_to_fix_rate(crab_deg)   # accept_rate = RELATIVE slope (lower bound)
density = SURR.accept_density_in_window                   # 0.84 — the in-window fix-density ceiling

# (b) inject a synthetic fix into the KF without rendering:
fix = SURR.sample_fix(geom, rng)                          # Bernoulli(p_accept); None on reject
if fix is not None:
    z_ned, cov_ned = fix
    kf.update_position(z_ned, cov_ned)                    # racer.state_estimator.LinearKF / RewindKF
```

Vectorised helper `accept_prob_in_image(range_array)` is available for batched envs; geometry is cheap
to vectorise in the training tensors.

---

## 6. Caveats & limitations (carry to inc8)

1. **Small-n calibration.** Floors rest on ~16–24 clean fix rows/flight; treat them as
   lateral 0.104 ±0.015, depth 0.85 ±0.10, peak 0.83 ±0.02 (CIs in `sigma.json`), not exact points.
2. **`crab_to_fix_rate.accept_rate` is a range-MARGINAL lower bound** (≈0.006–0.014), a relative
   crab→FoV slope — NOT a fix density. The reward-relevant density is `accept_density_in_window` ≈ 0.84.
3. **Elevation co-binds.** Yaw/crab alone is a modest lever; the ×8.5 prize needs 2-axis pointing
   (elevation = pitch/camera-tilt is the binding axis at fixable range). How the policy achieves
   pointing is a control/reward question — out of scope here.
4. **Flat depth/vertical σ.** `a1=0` → these don't grow with range; harmless because accept→0 past
   ~34 m (gate not fixable there anyway), but do not query σ as a far-range depth-uncertainty model.
5. **Calibrated at inc7 crab (~64–66°).** The counterfactual crab sweep assumes the gate-vs-velocity
   residual ε is crab-invariant; a fresh at-speed recording at a different crab (L3) would tighten it.
6. **`σ` is a single swappable constant set** (like `localization.GATE_REL_INPLANE_SIGMA`): an L3
   at-speed recording recalibrates by re-running `fix_surrogate_fit.py` → new checkpoints.

---

## 7. File manifest

| path | what |
|---|---|
| `rl/fix_surrogate.py` | **the module** — `geometry`, `FixSurrogate`, `sample_fix`, `crab_to_fix_rate`, `DEFAULT` |
| `tests/test_fix_surrogate.py` | 15 tests: geometry, band-pass, σ targets, SPD, cov-consistency, KF smoke, crab, checkpoints |
| `handoff/fix-surrogate-2026-06-14/build_dataset.py` | Track-3 → GT-geometry+labels dataset (reproduces the funnel) |
| `handoff/fix-surrogate-2026-06-14/fix_surrogate_fit.py` | the 4 sub-model fits + leave-one-flight-out CV |
| `handoff/fix-surrogate-2026-06-14/models/{accept,sigma,crab,cv}.json` | fitted checkpoints (baked into the module) |
| `handoff/fix-surrogate-2026-06-14/data/summary.json` | reproduced funnel/coverage anchors |
| `handoff/fix-surrogate-2026-06-14/adversarial/VERDICTS.json` | the 4 adversarial probe verdicts (agent scratch gitignored) |
