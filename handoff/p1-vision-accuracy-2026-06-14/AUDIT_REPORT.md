# P1 VISION-ACCURACY — Boresight / ε_vert Code Audit

**Fengyou** — this is the P1-VISION-AUDIT report. **VERDICT: ESCAPE HATCH FIRES. There is NO
code-side mechanism that injects ε_vert. The vision chain is internally consistent (synthetic
round-trip exact to 1.5×10⁻⁸ m); ε_vert ≈ 0.215 m / 0.56° is a PHYSICAL sim-render-vs-mount
boresight. P3's static head-on/LEVEL lock test at g2/g4 is load-bearing for the magnitude+sign;
the fix is a single empirical constant `BORESIGHT_PITCH_RAD` composed into `R_camera_from_body()`.**

- Session: P1-VISION-AUDIT · model opus-4.8 · effort MAX · branch `p1-audit-extrinsics` (off `main` 8c28c84)
- Constraints honored: **no `src/` edits**, **+L kept**, 20° mount untouched, 0.38 radius untouched,
  20-dim obs contract untouched. Scratch artifacts only under `handoff/p1-vision-accuracy-2026-06-14/scratch-audit/`.
- Baseline pytest from repo root: **703 passed, 35 skipped, exit 0** (`scratch-audit/pytest_baseline.txt`).
  No `src/` changed, so it stays green.

---

## TASK 1 — Consistency grep: one camera model, one mount, no y-flip

Every projection/decode site funnels through exactly **one** intrinsics and **one** mount. There is
no second, inconsistent camera model anywhere in `src/`, `rl/`, or `scripts/`.

| Site | Intrinsics | Mount | Notes |
|---|---|---|---|
| `frames.CAMERA_INTRINSICS_K` | fx=fy=**320**, cx=**320**, cy=**180** | — | single definition |
| `frames.R_camera_from_body()` | — | `_S @ Ry(-20°)` | camera Z resolves to **+20° pitch-up** (verified) |
| `frames.vertical_fov_deg()` | **fy**=320 → 58.7° | — | uses fy, **not** a literal 90 |
| `frames.horizontal_fov_deg()` | **fx**=320 → 90° | — | spec "90° VFoV" mislabel correctly flagged in code |
| `gate_pose.project_gate_corners` | K | — | synthetic forward projector |
| `gate_pose.estimate_gate_pose` | K | — | IPPE_SQUARE / P3P decode |
| `vision/synthetic.py` labels | K (via projector) | — | normalized by `/IMAGE_HEIGHT`, **top-left origin, no y-flip** |
| `vision/detector.py` | — | — | ultralytics `kpts.xy` (abs px, top-left), **never reordered**, only subset |
| `vision/association.predict_gates_in_camera` | K | `R_camera_from_body().T` | forward model = inverse of the fix |
| `localization.gate_pose_to_world_position` | — | `R_camera_from_body().T` | the **+L** lever (line 86/87) |
| `rl/fix_surrogate.py`, `scripts/*`, `gate_mapper_synth.py` | K / inline 320/320/180 | `R_camera_from_body` | all consistent |

Findings:
- **fy=320 / cy=180 used at EVERY site.** No hardcoded `90`/`58.7`/`179.5` in any code path
  (only in a `frames.py` docstring). `gate_mapper_synth.py:173-174` inlines `320/320/180` for a
  visibility check — values match K exactly (a minor duplication nit, **not** a bias source).
- **H and V FoV are never swapped.** VFoV decodes from fy, HFoV from fx; the spec's mislabeled
  "90° VFoV" is explicitly handled.
- **No image-row / y-axis-flip convention anywhere** in the pixel path: no `flipud`, `::-1`,
  `IMAGE_HEIGHT - y`, `180 - v`. (Every "flip" hit is PnP 2-fold pose ambiguity or CTBR sign
  config — unrelated.) Labels and the live detector both use top-left-origin, y-down pixels —
  the **same** convention `project_gate_corners` emits and `estimate_gate_pose` consumes.
- **Detector-corner convention == auto-label projector convention.** Detector keypoint *i* IS
  canonical corner *i* (LL/LR/UR/UL); trained on `synthetic.py` labels written in that exact order;
  ultralytics denormalizes by the native image height (=360) = the label normalization. No constant
  detector-vs-label pixel offset is introduced **in code**.
- **cy=180 vs the true 360-row centre 179.5** is a 0.5 px choice **used identically** in projection
  and decode → internally consistent. It is a vertical bias *only if the sim renders with a different
  cy*, and even then a 0.5 px mismatch is ≤ 0.031 m at 22 m (≤14% of ε_vert; see Task 2C) — it
  cannot, alone, explain ε_vert. (A P3 item: confirm the sim's principal point.)

**Task 1 result: NO fy/cy/convention/sign inconsistency; detector convention matches the projector.**

---

## TASK 2 — Decisive synthetic round-trip (real `src/racer` chain, no sim/GPU)

Script `scratch-audit/roundtrip_boresight.py` (full log `scratch-audit/roundtrip_boresight_OUT.txt`)
drives the **real** chain: `project_gate_corners → estimate_gate_pose → gate_pose_to_world_position`
(`+L`). Parametric mount asserted equal to `frames.R_camera_from_body()` at 20°.

**(A) Internal consistency — render@20 / decode@20.** Over 60 poses (8–33 m, az ±20°, el −5..10°):

```
max |recovered drone pos − truth| = 1.48e-08 m   →  internally consistent (≪ 1 mm)
```

The K projection (fy=320, cy=180) and the +L localization lever are **exact inverses**. No
code-internal vertical bias exists at matched mount.

**(B) +δ render-only boresight, decode@20** (head-on, level, 22 m) — reproduces the field signature:

```
 δ(deg)   gate_vert(down,m)        ← field ε_vert ≈ −0.215 m
 −0.56        +0.2150
  0.00        −0.0000
 +0.56        −0.2150   ✅ EXACT match
 d(gate_vert)/dδ = −0.3840 m/deg  (= −range·π/180; geometry-exact)
```

- **Range:** at δ=+0.56°, the *angle* is flat at **0.5600°** for 8→45 m; metric vbias ∝ range
  (−0.078 m @8 m … −0.440 m @45 m). **Depth unbiased** (max |depth_err| = 0.00000 m over the sweep
  and over 290/300 mixed poses) → **depth-free**, matching the field corr(range_err, vert)≈+0.11≈0.
- **Bearing (camera-frame signature):** |gate_vert| varies 0.1899→0.2150 m across az ±28°
  (**11.7% bearing-dependent**) — a world-frame map offset would be **flat**. Over a *one-sided*
  bearing range it rises monotonically with bearing → a **positive** corr(vert, bearing),
  reconciling the field's +0.44 (the field's signed magnitude is set by its gate-azimuth + attitude
  spread; a clean level sweep is symmetric in ±az).

**(C) Alternate code-side mechanisms are observationally identical to a mount tilt** (22 m):

```
 constant +3.13 px corner offset → gate_vert −0.191 m   (cy 176.9 ↔ same)
 constant −3.13 px corner offset → gate_vert +0.188 m   (cy 183.1 ↔ same)
 half-pixel (0.5 px) cy seam     → gate_vert −0.031 m   (~14% of 0.215 → cannot alone explain ε_vert)
```

A constant vertical corner offset OR a cy shift yields the **same range-proportional vertical world
bias** (≈10–15% smaller than the equivalent mount tilt, same sign/order). **Consequence: a mount
mismatch, a sub-pixel detector bias, and a principal-point offset are observationally
indistinguishable** — so if such an offset existed in code it would produce exactly this signature,
and it MUST therefore be findable in code. Task 1 shows it is **not** there.

*(Aside: a pure vertical corner shift can tip a dead-frontal gate into the wrong IPPE 2-fold branch
(→ `None`); it resolves cleanly off-axis / with a prior — an IPPE disambiguation property, not a
chain bug. The navigator always supplies a prior at transit.)*

---

## TASK 3 — VERDICT

All three escape-hatch conditions hold:

1. **Round-trip EXACT at 0° injected boresight** — 1.48×10⁻⁸ m. ✅
2. **Grep finds no fy/cy/convention/sign inconsistency** — single K, single 20° mount, VFoV from fy
   (never swapped), no y-flip, one source of truth for both intrinsics and mount. ✅
3. **Detector-corner convention matches the auto-label projector** — same canonical order, same
   top-left y-down pixel frame, no reorder, matched normalization. ✅

> **NO code-side mechanism — ε_vert is a PHYSICAL sim-render-vs-mount boresight. P3's static
> head-on/LEVEL lock test (g2/g4) is load-bearing for the magnitude + sign; the calibration is an
> empirical constant via the `BORESIGHT_PITCH_RAD` insertion point in `frames.R_camera_from_body()`.**

ε_vert is real (field-measured) but arises **outside our code**: the actual sim/physical camera is
oriented ≈0.56° differently in pitch from the 20° model in `frames.py` (or an equivalent sub-pixel
detector/principal-point offset — Task 2C shows these are the same observable). Because our camera
model is the *only* model and is self-consistent, the bias can only be the model-vs-physical gap,
which is exactly what one empirical constant removes. **No code bug was fabricated to satisfy the brief.**

---

## TASK 4 — Boresight calibration design (delivered; magnitude is P3-supplied)

**Insertion point (one edit, one source of truth)** — `scratch-audit/proposed_frames_boresight.patch`:
add `BORESIGHT_PITCH_RAD = 0.0` (+ optional `BORESIGHT_YAW_RAD = 0.0`) beside `CAMERA_PITCH_RAD`,
and compose it INTO `R_camera_from_body()`:

```python
R_tilted_from_body = Rotation.from_euler("Y", -(CAMERA_PITCH_RAD + BORESIGHT_PITCH_RAD)).as_matrix()
R_boresight_yaw    = Rotation.from_euler("Z", -BORESIGHT_YAW_RAD).as_matrix()
return _R_CAMERA_FROM_TILTED_BODY @ R_boresight_yaw @ R_tilted_from_body
```

At default `0.0` the mount is **bit-identical** to today → the whole stack stays VQ1 byte-identical
until calibrated. Every consumer (localization fix, association forward model, `fix_surrogate`,
navigator, `characterize_perception`) calls `R_camera_from_body()`, so the single constant
propagates everywhere automatically. **No +L change, no obs-contract change, no K change, 20° mount
intact** (the boresight is an *additive correction*, not a re-pointing of the spec mount).

**Procedure (P3 supplies the measurement).** Drone **static, LEVEL** (roll=pitch=0), facing a known
gate at known range so **bearing ≈ 0** (isolates the boresight from any map/lateral term). Use **g2
and g4** head-on. Collect N frames through the live detector → `estimate_gate_pose` →
`gate_pose_to_world_position` with `BORESIGHT_PITCH_RAD = 0`. Average out pixel noise; take the mean
gate-vertical (down-positive) fix-vs-GT residual `m_v` at each gate's range.

**Estimator (verified end-to-end, `scratch-audit/calib_estimator_demo.py`):**

```python
def solve_boresight_pitch_rad(residual_m, range_m):     # down-positive residual
    return -atan(residual_m / range_m)
```

Joint over g2/g4 (mean, or LS over all frames). The audit pins the SIGN: a camera pitched UP by +ε
relative to the model gives `gate_vert = −range·tan(ε)`, so `m_v = −0.215 m @ 22 m → +0.56°`.
Demo result: injected 0.560° → **recovered 0.5524° (err −0.008°)**; post-correction residual
**≤ 0.002 m** at 18/22/**30** m (30 m not used in the fit → the single angular constant is
range-proportional and generalizes).

**Acceptance target:** post-calibration mean |gate-vertical residual| **≤ 0.05 m** (≤ ~0.13°) at the
head-on band. (Demo achieves 0.002 m.)

**Regression test** — `scratch-audit/proposed_test_boresight_calibration.py` (lands in `tests/` WITH
the patch; **14 passed / 1 skipped** standalone today, the skip being the byte-identity test that
activates once the constant exists). Pins: (1) `BORESIGHT_*_RAD == 0.0` ⇒ `R_camera_from_body()`
bit-identical to the 20°-only mount; (2) sensitivity sign `gate_vert ≈ −range·tan(ε)`;
(3) matched-mount round-trip < 1 mm; (4) estimator recovers an injected ε to ≤0.05° with
post-correction residual ≤ 0.05 m.

**Relationship to the co-equal lever.** This static constant removes the **constant** vertical
boresight (margin lever #2 / fix-ACCURACY). It composes with — does not replace — **ESKF
attitude-bias estimation**, which tracks the time-varying residual. Calibrate the constant first
(cheap, one-time); let the ESKF mop up drift. Because Task 2C shows mount-tilt / detector-sub-pixel /
cy are the same observable, this one constant corrects the **net** vertical perception bias
regardless of its physical decomposition — so calibrate at (or near) the g2/g4 transit range to keep
any second-order range mismodeling negligible if the true source is pixel-domain.

---

## Scratch artifacts (all under `scratch-audit/`, no `src/` touched)
- `pytest_baseline.txt` — 703 passed / 35 skipped / exit 0
- `roundtrip_boresight.py` + `roundtrip_boresight_OUT.txt` — Task 2 (A/B/C), full numbers
- `calib_estimator_demo.py` — Task 4 estimator, recovers injected boresight to <0.01°, residual <2 mm
- `proposed_frames_boresight.patch` — the one-edit insertion point (NOT applied)
- `proposed_test_boresight_calibration.py` — the regression test (NOT in tests/ yet; 14 pass / 1 skip)

---

## MEMORY-DELTA (≤10 lines — for the commander to bank; I do NOT write memory/)
- **P1 AUDIT VERDICT: ESCAPE HATCH — ε_vert has NO code mechanism.** Vision chain internally
  consistent: synthetic round-trip via the real `project_gate_corners→estimate_gate_pose→
  gate_pose_to_world_position(+L)` recovers truth to **1.48e-8 m** at matched mount.
- **Grep: ONE camera model (fy=320,cy=180) + ONE mount `R_camera_from_body()`@20° everywhere**
  (src/rl/scripts); VFoV from fy (never swapped); **no y-flip**; detector `kpts.xy` convention ==
  `synthetic.py` auto-label projector (top-left, no reorder). No second intrinsics/mount exists.
- **Sensitivity d(gate_vert)/dδ = −range·π/180 = −0.384 m/deg @ 22 m**; +0.56° render-vs-decode
  mismatch reproduces **−0.2150 m EXACT**, angle-flat across range (depth-free, max|depth_err|≈0),
  bearing-dependent (camera-frame, 11.7% over ±28° — world-map offset would be flat).
- **Task 2C: mount-tilt ≡ +3.13 px corner offset ≡ cy-shift** (all range-proportional vertical bias);
  cy 180-vs-179.5 half-px seam = only −0.031 m (≤14%, cannot alone explain ε_vert). ε_vert is a
  PHYSICAL sim-vs-mount boresight → **P3 static head-on/LEVEL lock test (g2/g4) is LOAD-BEARING**.
- **Calibration = ONE constant `BORESIGHT_PITCH_RAD` (default 0.0 → byte-identical) composed into
  `R_camera_from_body()`**; estimator `BORESIGHT_PITCH_RAD = −atan(mean_vert_residual / range)`
  (down-positive). Demo recovers injected 0.56°→0.552° (err 0.008°), post-residual ≤0.002 m @18/22/30 m.
- **Acceptance:** post-cal mean |vert residual| ≤ 0.05 m / ≤ ~0.13°. Regression test proposed
  (byte-identity + sign + round-trip + estimator), 14 pass/1 skip standalone. Patch + test are
  PROPOSED (not applied; commander is merge gate). +L / 20° mount / 0.38 r / 20-dim obs all intact.
- Composes with (does not replace) **ESKF attitude-bias estimation** (constant first, ESKF tracks drift).
