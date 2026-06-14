# P1-CALIB-APPLY-2 — APPLY INFRA REPORT

**Fengyou** — dual-form boresight infra is **APPLIED, byte-identical, 0 regressions, and LIVE-READY for your merge gate.**

**Session:** P1-CALIB-APPLY-2 (reports to P1 VISION-ACCURACY commander via Fengyou)
**Model:** claude-opus-4.8 · **Effort:** HIGH · **Date:** 2026-06-14
**Branch:** `calib-v2-apply2` → pushed to `origin/p1-calib-v2`
**Infra commit:** `5091c88a848a127b951adfcd67ef8b5c77abe60f`

---

## VERDICT ✅ — APPLIED. Default-all-zero, byte-identical, 0 regressions. Ready for merge.

This supersedes the prior STOP (74cc72c): the artifact was a design doc not a diff, so per your re-issue the edits were made **directly** in `src/` (no `git apply`).

## 1. Baseline (pristine src, recorded BEFORE edits) — GREEN ✅
```
703 passed, 35 skipped in 191.99s   (.venv/Scripts/python.exe, pytest 9.0.3, from repo ROOT)
```

## 2. Edits made (transcribed from CALIB_V2_DESIGN.md + the artifact, at the confirmed lines)
**`src/racer/frames.py`** (3 edits):
- `from dataclasses import dataclass` import.
- `@dataclass(frozen=True) class BoresightCorrection` with `pitch_rad=roll_rad=vert_offset_m=0.0` (all-zero default) + module-level `BORESIGHT = BoresightCorrection()`.
- `R_camera_from_body()` composes `BORESIGHT.pitch_rad` (about body-Y≡cam-X) and `BORESIGHT.roll_rad` (about body-X≡cam-Z): `_R_CAMERA_FROM_TILTED_BODY @ R_roll @ R_tilted_from_body`. At all-zero `R_roll==I` and the pitch arg `== CAMERA_PITCH_RAD` exactly.

**`src/racer/localization.py`** (helper + both lever sites + import):
- `_apply_camera_vert_offset(position_ned, R_world_body)` guarded on `vert_offset_m != 0.0` (default → line skipped → byte-identical).
- Threaded into **both** `+L` lever sites: `gate_pose_to_world_position` and `gate_relative_inplane_fix`.

## 3. ⚠️ DEVIATION-FROM-ARTIFACT (flagged for your merge-gate review) — required, low-risk, behaviorally identical
**Found a latent bug in the calib-v2 deliverable:** the artifact's `src` and its own acceptance test are inconsistent. The artifact specified `from racer.frames import BORESIGHT` (a **snapshot** binding in localization's namespace), but the test `test_metric_field_shifts_lever_and_preserves_plus_L` monkeypatches `frames.BORESIGHT` and expects the localization lever to see it. With a snapshot binding it does NOT propagate → `pos1 == pos0` → **test FAILED**. (The calib-v2 author never caught this because that session never applied the patch — the 3 pins only activate against patched src.)

**Resolution (option A, chosen):** localization reads `frames.BORESIGHT` **LIVE** — added `from racer import frames` and the helper references `frames.BORESIGHT.vert_offset_m`; the line-29 `from`-import reverts to the original (no `BORESIGHT`). Rationale:
- This realizes the design doc's own words: *"the ONE correction instance every consumer reads"* / *"single source of truth: frames.BORESIGHT."* A snapshot does not.
- It makes the two consumer paths **consistent**: `R_camera_from_body()` (frames' own global) and the localization lever now both track a single `frames.BORESIGHT`.
- **Behaviorally identical** to the artifact at the all-zero default (byte-identical — guard False) AND in production (P3 sets the value via source-edit in frames.py; localization reads it live = same value). It differs ONLY for runtime swap (which is exactly what the acceptance test exercises).
- It honors *"promote the test as-is"* — **the test is unchanged.**

The alternative (option B: keep artifact `src`, edit the test's monkeypatch target) would paper over the src bug, violate *"promote test as-is,"* and leave the mount/lever paths inconsistent. **If you prefer option B, this is a clean revert of the two localization import/helper lines + a one-line test edit — flag at the gate.**

## 4. Byte-identity verification (the load-bearing gate protecting inc7) — PASS ✅
Direct check + the 3 now-active pins in `tests/test_calib_dualform.py`:
- **MOUNT** `np.array_equal(R_camera_from_body(), 20°-only mount) == True` (bit-identical, not allclose) — `test_unified_default_zero_is_byte_identical_mount` PASS.
- **+L LEVER** at default: `_apply_camera_vert_offset` returns the input array **object unchanged** (guard False) → both fix sites byte-identical; with `vert_offset_m` set the fix shifts by exactly `−R_wb·[0,0,voff]` (+L direction preserved) — `test_metric_field_shifts_lever_and_preserves_plus_L` PASS.
- **ANGULAR compose** — `test_angular_fields_compose_into_mount` PASS.
- Defaults confirmed all-zero: `(pitch_rad, roll_rad, vert_offset_m) == (0.0, 0.0, 0.0)`.

## 5. Full suite + targeted confirms — 0 regressions ✅
```
FULL (repo ROOT):  725 passed, 35 skipped in 209.97s, exit 0   (0 failed)
```
- **Before 703 → after 725 = +22** (exactly the new test file; `tests/test_calib_dualform.py` = 22 tests). **Skipped unchanged 35 → 35.** **0 regressions** (no `failed`; the 703 baseline-passing all still pass).
- **NOTE on the 726 estimate:** actual is **725**, not 726. The file has exactly 22 tests (isolated run: "22 passed"); `703 + 22 = 725`. The task's "703 + 22 + 1 = 726" carried an off-by-one; skipped staying at 35 corroborates that no extra test un-skipped. 725/35 is the correct expected outcome.
- **+L sign preserved:** `tests/test_obs_sign_faithfulness.py` green (targeted run: 48 passed across obs-sign + frames + localization + calib_dualform).
- Untouched: 20° mount, +L sign, 0.38 radius, 20-dim obs contract. No residual/prior-width fields added. No `rl/`, sim harness, `fly_rl`/`submit_rl`, or `memory/` touched.

## 6. Real diff exported (the genuine unified diff the artifact was not)
`handoff/p1-vision-accuracy-2026-06-14/scratch-calib-v2/applied_calib_dualform.diff` (326 lines), validated: `git apply --check --reverse` succeeds → it is a real, machine-applicable diff. Committed in `5091c88`.

## 7. State
- `e` is **unknown** → defaults stay all-zero (inert). When P3's arbiter pins it, it drops in as a one-line `frames.BORESIGHT = BoresightCorrection(...)` ({form, value}); no further code change.
- `p1-calib-v2` @ `5091c88` (infra) — overall commander = sole merge gate; NOT merged to main.

---

## MEMORY-DELTA (≤10 lines)
- ✅ **P1 CALIB-V2 APPLY DONE & pushed → `p1-calib-v2` @ `5091c88`** (infra commit; report commit follows). Dual-form boresight infra LIVE, DEFAULT-ALL-ZERO byte-identical.
- **BYTE-IDENTITY PROVEN** at default-zero: MOUNT `np.array_equal` True (bit-identical 20° mount) + **+L LEVER** helper returns input unchanged (both fix sites); all 3 pins active+PASS; +L sign (`test_obs_sign_faithfulness`) green.
- **SUITE: 703 → 725 passed / 35 skipped, 0 regressions** (file = 22 tests; the prior "726" estimate was off-by-one — 725 is correct, skipped unchanged confirms it).
- **⚠️ DEVIATION-FROM-ARTIFACT (merge-gate review):** localization reads `frames.BORESIGHT` LIVE (`from racer import frames`) NOT the artifact's `from racer.frames import BORESIGHT` snapshot — required to pass the acceptance test + realize the design's "single source of truth"; byte-identical at default, identical in production, test untouched. Revertable to option-B if preferred.
- **e STILL unknown → defaults all-zero; e drops in as one-line `frames.BORESIGHT = BoresightCorrection({form,value})`.** default-zero dual-form boresight infra LIVE-READY for merge gate.
- Real machine-applicable diff: `handoff/p1-vision-accuracy-2026-06-14/scratch-calib-v2/applied_calib_dualform.diff` (validated via `git apply --check --reverse`). → [[index-vision-estimator]]
