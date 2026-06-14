# P1 VISION-ACCURACY — Dual-Form Boresight Calibration (CALIB_V2)

**Fengyou** — this is the P1-CALIB-V2 report. **VERDICT: the boresight calibration is now
FUNCTIONAL-FORM-AGNOSTIC and READY.** Both forms are built and validated against the real `src/racer`
chain behind ONE unified correction `frames.BoresightCorrection{pitch_rad, roll_rad, vert_offset_m}`
(default all-zero → **byte-identical**, proven by `np.array_equal` on both the mount and the +L lever).
Whichever form P3's gate-0/1 head-on arbiter selects, applying it is a **one-line constant**; the flight
only **selects the form + pins the value(s)**. Two estimators are folded in and validated: **#15
Bayesian-IoU pitch+roll** (angular) and **#16 level-hover regression** (metric + the angular-vs-metric
discriminator). **Escape-hatch finding (flagged UP): a SINGLE range is degenerate — P3's ≥2-range
requirement is load-bearing for the FORM, not just the magnitude.**

- Session: P1-CALIB-V2 · model opus-4.8 · effort MAX · branch `p1-calib-v2` (off `origin/p1-audit-extrinsics` a1620b3)
- Constraints honored: **no `src/` edits**; **+L kept** (localization.py:86, pinned by `test_obs_sign_faithfulness`);
  20° mount untouched (additive correction, default 0); 0.38 radius untouched; **20-dim obs contract NOT
  reopened**; yaw NOT reopened. Scratch only under `handoff/p1-vision-accuracy-2026-06-14/scratch-calib-v2/`.
- Baseline pytest from repo root: **703 passed, 35 skipped, exit 0** (`scratch-calib-v2/pytest_baseline_calibv2.txt`).
  No `src/` changed → stays green.
- **Worktree note (worktree-hygiene directive):** the harness already isolated this session in its own
  worktree, so rather than create a sibling `../Anduril-wt-calib-v2` (which would double worktrees /
  clutter sibling dirs) I created branch `p1-calib-v2` off `origin/p1-audit-extrinsics` **in-place**.
  Same functional base + inherited audit artifacts; cleaner tree. Branch pushed for the commander's merge gate.

---

## 0. What the audit settled, and what CALIB_V2 adds

The audit (a1620b3) proved ε_vert ≈ −0.215 m / 0.56° at 22 m is a **physical** sim-render-vs-mount
boresight (no code mechanism; synthetic round-trip exact to 1.5e-8 m) and built the **ANGULAR** form:
one constant `BORESIGHT_PITCH_RAD` composed into `R_camera_from_body()`, estimator
`−atan(mean_vert_residual / range)`. CALIB_V2 keeps that intact and adds the **second functional form**
the data cannot yet exclude, plus the discriminator and the stronger estimator:

| | ANGULAR (audit + roll) | METRIC (new) |
|---|---|---|
| Physical model | camera mount ORIENTATION off by (pitch, roll) | camera optical-centre TRANSLATED from body origin (FRD +Z) |
| Where applied | `R_camera_from_body()` (mount) | localization **+L lever** (both fix sites) |
| World-vert fix bias vs range | **∝ range** (back-out angle range-FLAT) | **CONSTANT** (back-out angle ∝ 1/range) |
| Constant | `pitch_rad` (+ `roll_rad`) | `vert_offset_m` |
| Estimator | #15 IoU-BO (+ audit's −atan cross-check) | #16 level-hover regression intercept |
| Motivation | the measured ε_vert vertical signature | localization.py's own note: *"the official sim says same origin; the Elodin rig offsets the camera"*; the +0.3 m vertical systematic absorbed by `FIX_COV_FLOOR_STD` |

**At 22 m the two are INDISTINGUISHABLE** (both produce −0.215 m); only **≥2 separated ranges**
discriminate them. So CALIB_V2 builds **both now**; P3's gate-0/1 arbiter (≥2 ranges) decides which
constant(s) get populated.

---

## TASK 2 — The dual-form unified patch (PROPOSED, `scratch-calib-v2/proposed_calib_dualform.patch`)

A single patch (SUPERSEDES the audit's standalone `proposed_frames_boresight.patch`) touching two files:

**(a) ANGULAR — `frames.py`.** `R_camera_from_body()` composes `pitch_rad` (about body-Y == camera-X,
the ε_vert "right" axis) **and** `roll_rad` (about body-X == camera-Z, the optical axis):
```python
R_tilted_from_body = Rotation.from_euler("Y", -(CAMERA_PITCH_RAD + BORESIGHT.pitch_rad)).as_matrix()
R_roll             = Rotation.from_euler("X", -BORESIGHT.roll_rad).as_matrix()
return _R_CAMERA_FROM_TILTED_BODY @ R_roll @ R_tilted_from_body
```
**(b) METRIC — `localization.py`.** A shared guarded helper applied at BOTH +L lever sites
(`gate_pose_to_world_position`:87, `gate_relative_inplane_fix`:163):
```python
def _apply_camera_vert_offset(position_ned, R_world_body):
    if BORESIGHT.vert_offset_m != 0.0:                      # guarded => default byte-identical
        return position_ned - R_world_body @ np.array([0.0, 0.0, BORESIGHT.vert_offset_m])
    return position_ned                                     # p_body = p_camera_centre − R_wb·t_body_cam
```
**(c) UNIFIED struct — the single source of truth** (`frames.py`):
```python
@dataclass(frozen=True)
class BoresightCorrection:
    pitch_rad: float = 0.0      # angular, body-Y == cam-X
    roll_rad: float = 0.0       # angular, body-X == cam-Z
    vert_offset_m: float = 0.0  # metric, body FRD +Z down
BORESIGHT = BoresightCorrection()      # default identity
```
The ESKF static hook and the deploy path consume this ONE struct regardless of selected form.

**Byte-identity PROVEN** (`roundtrip_dualform.py` (0)): at all-zero, `mount_corrected(0,0) ==
R_camera_from_body()` and `lever(vert_offset=0) == unpatched +L lever`, both **`np.array_equal` True**
(bit-identical, not merely allclose). Mechanism: `+0.0` is a float-exact no-op on the pitch arg, `R_roll`
is the exact identity, `M @ I == M` bit-exact, and the metric line is **skipped** at `vert_offset_m == 0`.

### Signatures (real chain, `roundtrip_dualform.py`)
```
ANGULAR (+0.56° injected):  range 8→30 m  vbias −0.078→−0.293 m   back-out angle FLAT +0.5600° (std 0.0000)
METRIC  (−0.215 m injected): range 8→30 m  vbias −0.2150 m (std 0.00000)   back-out angle +1.54°→+0.41° (∝1/r)
@22 m: angular −0.2150 vs metric −0.2150  → |diff| 2e-5 m (SAME)
ROLL (+0.8°): render@roll/decode@roll round-trips 8.9e-9 m; head-on vbias ±0.099 m at az ±20° (bearing-driven, ~0 head-on)
Each form's decode correction drives the residual to ≤1e-5 m.
```

---

## TASK 3 — Estimator #15: Bayesian-IoU pitch+roll (ANGULAR form), `iou_bo_calib.py`

MonoRace IoU-BO (arXiv 2601.15222 digest), adapted to **pitch+roll ONLY** (yaw is NOISE — NOT reopened;
§VISION-PKG2 refutation stands). For gate sightings with known drone state (the static lock test = exact)
and known map gate pose, reproject the map corners through (state estimate + candidate extrinsic) and
score **IoU(predicted_quad, detected_quad)**; maximize over (pitch, roll). Self-contained **numpy GP
(RBF) + Expected-Improvement** optimizer (no sklearn/skopt present), ~40 evals + a Nelder-Mead polish;
convex-quad IoU via `cv2.intersectConvexConvex`.

**Validation (recovers an injected extrinsic):**
```
noiseless pitch-only  : err (pitch, roll) = (+0.0000, −0.0000)°   IoU* = 1.0000
noiseless pitch+roll  : err (pitch, roll) = (+0.0000, −0.0000)°   IoU* = 1.0000
noisy (0.5 px, 8 fr)  : err (pitch, roll) = (−0.0055, −0.0080)°   IoU* = 0.9591
```
Far inside the 0.13° acceptance. **ROLL is observable only OFF-AXIS** (degenerate head-on, cf round-trip
(D) and the ESKF's "roll weakly observable head-on") → the calibration scene spans bearings. **Offline
the detector polygon is unavailable**, so we use the synthetic-projected physical-mount corners as the
"detected" quad (full IoU works offline); a **corner-reprojection-distance BO fallback** is also provided
and recovers identically (`reproj-BO` rows). IoU-BO uses full corner **shape** → stronger than the audit's
vertical-only `−atan`, which remains as the cheap pitch cross-check (round-trip (A) pins it to +0.5600°).

---

## TASK 4 — Estimator #16: level-hover regression (METRIC form + DISCRIMINATOR), `level_hover_regression.py`

From a static/LEVEL-hover recording that varies **range** to a head-on vertical reference (by altitude or
standoff), regress the gate-vertical (down +) residual against range:
```
    m_v(r) = intercept + slope · r + noise
             \______/    \______/
              METRIC       ANGULAR        intercept = camera-centre vert offset Δz (range-INVARIANT)
                                          slope = −tan(pitch_boresight) (range-PROPORTIONAL) → pitch = −atan(slope)
```
**The intercept-vs-slope split IS the functional-form discriminator** (complements P3's ε(range)).
Weighted LS (per-range m_v noise grows with the lever arm). Validated on the real chain:
```
1a NOISELESS control (proves the chain is exactly linear → 1b residual is a PnP NOISE bias, not method error):
   PURE ANGULAR : intercept −0.00000 m  pitch +0.5600°   PURE METRIC: intercept −0.21500 m  pitch −0.0000°   MIXED: −0.10000 m / +0.3000°
1b REALISTIC (0.7 px, 120 frames, WLS):
   PURE ANGULAR (eps 0.56°)   : intercept −0.0126±0.0020 m, pitch +0.5250°  → classify ANGULAR
   PURE METRIC  (voff −0.215) : intercept −0.2226±0.0020 m, pitch −0.0220°  → classify METRIC
   MIXED (0.30°, −0.10 m)     : intercept −0.1102±0.0020 m, pitch +0.2714°  → classify MIXED
```
The 1b residuals (~0.012 m intercept, ~0.035° slope at 0.7 px) are a **σ_px² PnP noise-bias floor**
(vanishes in 1a), well within acceptance. Classification uses a **physical-significance floor**
(0.05 m / 0.05°), NOT pure statistical 3σ — else the ultra-tight 120-frame noise (σ_mean ≈ 0.002 m)
flags the noise bias as "significant" and mislabels everything MIXED.

### Identifiability — the range-spread requirement (the escape-hatch question)
Two-range test at (r1, r2), per-mean noise σ_mv: `slope_sd = σ_mv·√2 / (r2−r1)`. The ε_vert angular
signal is `slope = −tan(0.56°) = −0.00977`. For 3σ angular detection, `(r2−r1) > 3√2·σ_mv / |slope|`:
```
σ_mv = 0.005 m → min Δrange  2.2 m      σ_mv = 0.020 m → min Δrange  8.7 m
σ_mv = 0.010 m → min Δrange  4.3 m      σ_mv = 0.030 m → min Δrange 13.0 m
chain-measured candidate pairs (inject 0.56°, 120-fr SEM):
   (18,22) → 14.0σ      (12,22) → 42.6σ      (10,28) → 63.1σ      (10,30) → 64.8σ
```
**Reading:** at a static lock test averaging many clean frames (σ_mv ≲ 0.005 m) even g2/g4's 4 m
separation discriminates (14σ); but to be **robust to real detector noise** (σ_mv up to 0.02–0.03 m)
recommend **Δrange ≥ ~10 m** (gate-0/1 at e.g. 12 m and ≥24 m). **A SINGLE range is ALWAYS degenerate**
(1 equation, 2 unknowns): at 22 m alone, m_v = −0.213 m is consistent with BOTH metric (intercept −0.213,
slope 0) AND angular (intercept 0, pitch +0.555°). → §SELECTOR + §ESCAPE-HATCH.

---

## TASK 5 — SELECTOR: P3 arbiter output → which constant(s) to populate

P3's gate-0/1 head-on arbiter returns m_v at ≥2 ranges (magnitude + SIGN + range-dependence). Fit the
joint model `m_v = intercept + slope·r` (the #16 regression, or simply two points), then:

| Observation across ranges | Form | Populate |
|---|---|---|
| m_v ∝ range (slope ≠ 0), intercept ≈ 0 (within floor) | **ANGULAR** | `pitch_rad = −atan(slope)` (+ `roll_rad` if a lateral lock term is also measured off-axis) |
| m_v constant (slope ≈ 0), intercept ≠ 0 | **METRIC** | `vert_offset_m = intercept` |
| both significant | **MIXED** | both: `vert_offset_m = intercept`, `pitch_rad = −atan(slope)`; the residual-minimizing split (intercept = range-invariant part, slope = range-proportional part) |

**Worked paths (chain-measured, #16):**
- **ANGULAR** — slope −0.00916 → `pitch_rad = −atan(−0.00916) = +0.00916 rad = +0.525°`; intercept
  −0.0126 m < 0.05 m floor → leave `vert_offset_m = 0`. (Audit's `−atan(m_v/r)` is the 2-point special case.)
- **METRIC** — intercept −0.2226 m → `vert_offset_m = −0.2226`; slope +0.00038 (≈ 0) → `pitch_rad = 0`.
- **MIXED** — intercept −0.110 m → `vert_offset_m = −0.110`; slope −0.00474 → `pitch_rad = +0.271°`.

Sign conventions are pinned by the harness: `pitch_rad = −atan(slope)`, `slope = −tan(eps)`,
`vert_offset_m = m_v` (down-positive residual), all re-derived from the real chain (`roundtrip_dualform.py`).
Then it is a one-line edit: `BORESIGHT = BoresightCorrection(pitch_rad=…)` / `(vert_offset_m=…)` / both.

---

## TASK 6 — ESKF COORDINATION (interface note — do NOT edit the eskf branch)

The ESKF design (`origin/p1-eskf-design`) builds a 1-DOF **pitch** bias state (δβ about camera-X) with a
scalar `static_boresight_rad` hook and a tight prior (`bias_prior_deg ≈ static residual ≈ 0.2°`); it states
the static calibration "removes the mean," δβ tracks "residual + drift." CALIB_V2's unified struct changes
how that hook is fed:

1. **The unified `frames.BORESIGHT` REPLACES the scalar `static_boresight_rad`.** The angular mean is
   ALREADY removed inside `R_camera_from_body()` (so the ESKF's `R_wc = R_world_body @ R_camera_from_body().T`
   inherits it automatically) and the metric mean inside the +L lever (so the fix `z` the ESKF ingests is
   already de-metric-biased). **The ESKF must therefore NOT re-apply the correction to `R_wc`** — doing so
   would DOUBLE-count. It should read the struct only to set its prior: **mean = 0** (already applied
   upstream), **width = the post-calibration residual** (≈0.2°). **This is a signature change on the eskf
   branch's hook (`static_boresight_rad: float` → consume `BoresightCorrection`, or pin it to 0 with a
   separate `bias_prior_deg`). FLAGGED UP to the P1 commander — I did not touch the eskf branch.**
2. **Axis consistency, no extra reconciliation.** `pitch_rad` is about body-Y ≡ camera-X — the SAME axis as
   the ESKF's δβ. Because the ESKF inherits the frames-corrected `R_wc`, the static term needs no separate
   sign mapping; the dynamic δβ composes on that same axis.
3. **If the form is METRIC, the ESKF's angular drift state STILL composes on top — NOT aliased.** The
   metric offset is removed upstream and is **range-FLAT**; whatever angular drift remains is
   **range-PROPORTIONAL**. Different range-dependence → distinguishable (the same orthogonality #16 exploits;
   consistent with the ESKF's own §3.4 bias-vs-map separability, corr 0.004). So a metric static correction +
   an angular ESKF δβ coexist cleanly.
4. **Non-obvious risk to carry:** the ESKF's δβ is purely **rotational** — it **cannot absorb a metric
   (translation) offset** (wrong range signature). If the true source is metric but the static calib is set
   to the wrong form (angular), the ESKF would systematically misfit (a range-flat residual fit by a
   range-proportional state). **→ the SELECTOR getting the FORM right is load-bearing for the ESKF too**,
   which sharpens why P3's ≥2-range discrimination matters.

---

## TASK 7 — ACCEPTANCE + regression test (PROPOSED, `scratch-calib-v2/proposed_test_calib_dualform.py`)

**Acceptance target (either form):** post-calibration mean |gate-vertical residual| **≤ 0.05 m / ≤ 0.13°**
at the gate-0/1 head-on band. Demonstrated: #15 recovers to <0.01°; #16 recovers intercept to ~0.008 m and
pitch to ~0.035°; both inside budget.

**Regression test — `19 passed / 3 skipped` standalone today** (the 3 skips are the patched-src pins that
activate when the patch lands — same discipline as the audit's 14/1). It pins, per Task 7:
- **byte-identity at all-zero** (skip-until-patch): `R_camera_from_body()` bit-identical + lever bit-identical;
- **+L preservation** (skip-until-patch): with `vert_offset_m` set, the fix shifts by **exactly**
  `−R_wb·[0,0,voff]` (the −lever direction unchanged); at zero it is bit-identical;
- **round-trip** (runs today): render@correction / decode@correction < 1 mm, for angular / roll / metric / mixed;
- **each estimator recovers an injected value** (runs today): #15 IoU-BO recovers (pitch, roll) to ≤0.05°/0.10°;
  #16 regression recovers (intercept, slope) and classifies ANGULAR / METRIC / MIXED.

---

## ESCAPE-HATCH FINDINGS (report UP)

1. **DEGENERACY → P3's ≥2-range test is load-bearing for the FORM, not just the magnitude.** A single
   range cannot separate intercept (metric) from slope (angular). With ≥2 ranges the split is identifiable;
   its quality scales with `Δrange · √n_frames / σ_per_frame`. **Recommend gate-0/1 standoffs separated by
   ≥ ~10 m** (e.g. 12 m and ≥24 m) so discrimination is robust even at pessimistic detector noise
   (σ_mv ≈ 0.02–0.03 m); g2/g4's 4 m suffices ONLY if many clean frames are averaged (σ_mv ≲ 0.005 m). This
   elevates the prior "g2/g4 pins magnitude" to "**range spread is load-bearing for the discrimination
   itself**." (If P3 can only realize a small range spread, it pins magnitude but **not** the form — then
   default to building BOTH and let the in-flight ESKF + a later wider-spread recording settle it.)
2. **cv2 `SOLVEPNP_IPPE_SQUARE` NaN edge** at a near-frontal quad with a near-uniform vertical corner shift
   (a pure metric offset or the audit's pixel-offset mechanism). Both IPPE candidates return NaN for certain
   (geometry, range) configs — an opencv numerical edge, NOT a chain bug. In flight the navigator's
   near-exact prior + IMU coast handle it; offline, a small geometry nudge / multi-frame averaging resolves
   it (the regression test retries across probe geometries). Worth noting for P3's lock test: average many
   frames and don't rely on a single dead-frontal snapshot.
3. **ESKF hook signature change** (§Task 6.1) — surfaced UP, not edited.

---

## Scratch artifacts (all under `scratch-calib-v2/`, no `src/` touched)
- `pytest_baseline_calibv2.txt` — 703 passed / 35 skipped / exit 0
- `roundtrip_dualform.py` (+ `dualform_OUT.txt`) — byte-identity, both signatures, discrimination, roll
- `iou_bo_calib.py` (+ `iou_bo_OUT.txt`) — #15 IoU-BO (GP-EI), recovers pitch+roll; reproj fallback
- `level_hover_regression.py` (+ `level_hover_OUT.txt`) — #16 regression, discriminator, identifiability, degeneracy
- `proposed_calib_dualform.patch` — the unified dual-form patch (NOT applied; supersedes the audit's frames patch)
- `proposed_test_calib_dualform.py` — the regression test (19 pass / 3 skip standalone)

---

## MEMORY-DELTA (≤10 lines — for the commander to bank; I do NOT write memory/)
- **P1 CALIB-V2 DONE (branch `p1-calib-v2` off audit a1620b3; design+scratch only, no src; baseline 703/35
  green; in-place worktree per worktree-hygiene).** Dual-form boresight READY behind ONE struct
  `frames.BoresightCorrection{pitch_rad, roll_rad, vert_offset_m}`, default all-zero → **byte-identical
  (np.array_equal proven, mount AND +L lever)**. Patch+test PROPOSED (commander is merge gate).
- **TWO FORMS, same 0.215 m @22 m signature, distinguished only by RANGE:** ANGULAR (pitch about
  body-Y≡cam-X + roll about body-X≡cam-Z, in `R_camera_from_body`) → vbias ∝ range / angle FLAT; METRIC
  (`vert_offset_m` in the +L lever, both sites, body FRD +Z) → vbias FLAT / angle ∝ 1/range. Indistinguishable
  at a single range; ≥2 separated ranges discriminate.
- **#15 IoU-BO (MonoRace, self-contained numpy GP-EI, ~40 iters + polish)** recovers pitch+roll: noiseless
  0.000°, noisy(0.5px/8fr) ≤0.008°; reproj-distance BO fallback identical. Roll observable OFF-AXIS only
  (degenerate head-on). Yaw NOT reopened.
- **#16 level-hover WLS regression** = metric estimator + discriminator: intercept=metric Δz,
  slope=−tan(pitch). Noiseless recovers EXACTLY; 0.7px/120fr classifies ANGULAR/METRIC/MIXED correctly via a
  physical-significance floor (0.05 m/0.05°); PnP noise-bias floor ~0.012 m / 0.035° (within acceptance).
- **ESCAPE-HATCH (load-bearing): a single range is DEGENERATE (intercept⊥slope unidentifiable) → P3's
  ≥2-range gate-0/1 test pins the FORM, not just the magnitude. Recommend standoff Δrange ≥ ~10 m**
  (g2/g4's 4 m OK only if σ_mv≲0.005 m via many frames). Also: cv2 IPPE_SQUARE NaN edge on near-frontal
  uniform-shift quads (opencv, not a chain bug; resolved by prior/coast/nudge).
- **SELECTOR:** angular→`pitch_rad=−atan(slope)`(+roll); metric→`vert_offset_m=intercept`;
  mixed→both. One-line constant; signs pinned by the chain.
- **ESKF COORDINATION (flagged UP, eskf branch NOT edited):** `frames.BORESIGHT` REPLACES the eskf's scalar
  `static_boresight_rad` hook — angular mean already removed in `R_camera_from_body`, metric mean in the
  lever, so the ESKF must NOT re-apply (double-count); it reads the struct only to set its prior (mean 0,
  width=residual). Metric static composes with the ESKF's angular δβ WITHOUT aliasing (range-flat vs
  range-proportional). The ESKF's rotational δβ CANNOT absorb a metric offset → SELECTOR form-choice is
  load-bearing for the ESKF too. Acceptance ≤0.05 m/0.13°; test 19 pass/3 skip. → [[index-vision-estimator]]
