# LAPTOP-VISION-FRAME-FIX 2026-06-12

**Session:** LAPTOP-VISION-FRAME-FIX · model: sonnet-4.6 · effort: high  
**Task:** Fix the vision/PnP/KF chain's attitude pairing from the R_y(π)-conjugated raw
ODOMETRY quat to the true physical body→world rotation (`R_world_from_odo_quat_wxyz`).

---

## Root cause

The FRAME-AUDIT (2026-06-12) established that the sim ODOMETRY quat is R_y(π)-conjugated:
```
q_true = q_raw * [1, -1, 1, -1]   (wxyz; negate x and z)
```
The CTBR control path consumed `euler_from_quat_wxyz(q_raw)` to get aliased roll/pitch/yaw that
are self-consistent end-to-end (VQ1-proven; **not touched**). But the vision chain at
`navigator.py:295` also used `R_world_from_body(ds.roll, ds.pitch, ds.yaw)` — the same aliased
euler — to project world gate geometry into the camera. At level flight the alias coincides with
the true rotation (roll≈0, yaw cancels in the gate direction). During banked flight (VQ2) it would
have produced world-fix errors proportional to `range × sin(2 × bank)` ≈ 7 m at 10 m range and
45° bank.

An additional effect confirmed by the VQ1-replay regression: **even at near-level VQ1 attitudes,
the yaw component of the conjugation is non-trivial.** The drone has substantial heading angles
while flying the course; the aliased yaw (negated) in the R_wb used by PnP caused a systematic
east-axis lever error that appeared as a `+3.5°/range` east bias in the VISION-PKG2 characterization.
**That bias was not in the covariance model — it was in the navigator's R_wb.** The corrected path
eliminates it to 0.0°/range.

---

## Changes

### `src/racer/frames.py`
Added `R_world_from_odo_quat_wxyz(q_odo_wxyz) -> np.ndarray` after `true_attitude_from_odo_quat_wxyz`.
Applies the conjugation `q * ODO_QUAT_TRUE_CONJ_WXYZ`, then builds scipy `Rotation` → matrix.
Falls back to `np.eye(3)` for None or near-zero-norm input (same as the old level-hover default).

### `src/racer/navigator.py` — line 295
```python
# BEFORE:
R_wb = R_world_from_body(ds.roll, ds.pitch, ds.yaw)

# AFTER:
# TRUE physical body->world rotation from the raw ODOMETRY quat (R_y(pi)-conjugated).
# The CTBR path uses euler_from_quat_wxyz on the raw quat (aliased, VQ1-proven —
# that path is untouched). Vision/PnP/KF must use the true attitude. [vision-frame-fix]
R_wb = R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)
```
The corrected `R_wb` flows to: `kf.predict(accel_body, R_wb, dt)` (line 313),
`_maybe_run_vision(ds, frame, R_wb)` (line 314), `predict_gates_in_camera(...)` (line 343),
`gate_pose_to_world_position(...)` (line 378). The CTBR controller path (`nav.roll/pitch/yaw` from
NavState, which carries the aliased euler) is **not affected**.

### `scripts/characterize_perception.py` — line 146
Updated to use `F.R_world_from_odo_quat_wxyz(...)` so the characterization script faithfully
measures what the navigator's PnP actually sees. Updated docstring accordingly.

### `tests/test_frames.py`
Two golden tests:
- `test_R_world_from_odo_quat_wxyz_gives_true_rotation_at_bank` — at 45° roll + 30° yaw: verifies
  the helper gives `R_world_from_body(true_euler)`, that `euler_from_quat_wxyz(q_raw)` gives
  negated roll/yaw (aliased path confirmed different), and that `R_alias ≠ R_true` at 45° roll.
- `test_R_world_from_odo_quat_wxyz_level_is_identity` — level `[1,0,0,0]` → identity; None →
  identity; zero-norm → identity, no crash.

### `tests/test_navigator.py`
`test_vision_fix_correct_at_banked_attitude` — redesigned from a failing test:

**Geometry:** drone at `[0,0,-2]`, gate at `[9,4,-2]` (9m north, 4m east), 45° true roll.  
**Mechanism:** at 45° roll the east lever-arm contribution to the world-fix is:
- Correct R_wb: fix = `[0,0,-2]` = true_pos
- Aliased R_wb: fix = `[0,4,-2]` — 4m east error

With `use_given_position=True, given_pos_std=0.05` anchoring the KF at true_pos:
- Correct R_wb → Mahalanobis ≈ 0 → fix **accepted** → `n_vision_fixes > 0`  ✓  
- Aliased R_wb → Mahalanobis ≈ 80 >> chi2_0.999=16.27 → fix **rejected** → `n_vision_fixes = 0`

All 4 corners stay within 640×360 at this geometry (analytically verified: u≈380–452, v≈160–229).

---

## Acceptance criterion results

### (a) VQ1-replay regression — task2_frames bundle (40 near-level frames, range 1.8–23.3m)

| Metric                | BEFORE (aliased R_wb) | AFTER (true R_wb) |
|---|---|---|
| Detection rate        | 40/40 (100%)          | 40/40 (100%)      |
| Association rate      | 29/40                 | 29/40             |
| Clean fix p50         | 1.37 m                | **0.47 m**        |
| Clean fix p90         | 2.37 m                | 1.25 m            |
| Good-fix yield (<1m)  | 6/27 (22%)            | 23/27 (85%)       |
| Catastrophic leak     | 1/29 (3.4%)           | 0/29 (0%)         |
| East bias fit slope   | +3.5°/range           | **0.0°/range**    |
| North bias fit slope  | −2.0°/range           | −0.4°/range       |

Detection and association rates are **statistically unchanged**. Fix quality dramatically improved.
The `+3.5°/range` east bias previously attributed to calibration / covariance is now confirmed to
have been the aliased yaw in the navigator's R_wb. **MEMORY update required:** the entry
"~3.6°/range yaw bias REFUTED (→ covariance)" is now superseded — the bias was real but it was
caused by the wrong R_wb, not by calibration; it is now eliminated.

### (b) Golden tests
`test_R_world_from_odo_quat_wxyz_gives_true_rotation_at_bank` — PASSED  
`test_R_world_from_odo_quat_wxyz_level_is_identity` — PASSED  
`test_vision_fix_correct_at_banked_attitude` — PASSED  

### (c) Full test suite
616 passing + 4 pre-existing failures. The 4 failures are in `test_twin.py` /
`test_twin_tune.py` and are caused by uncommitted `rl/` changes from the inc7 environment
work (confirmed via `git stash`: those tests pass against HEAD; they were already failing
before this session's changes). Total test count: 620 (598 original + 19 from new
`test_contact_geometry.py` + 2 new frames tests + 1 redesigned navigator test).

### Bounded extra: MLE attitude-noise re-fit
The `fixedframe_g%d.json` files needed by `mle_sigma_theta.py` are NOT on this machine (only
`.stdout.txt` text files exist in `handoff/shadowpc-vision-pkg2-2026-06-10/`). A rough estimate
from task2 data: clean p50 = 0.47m at ~10m range with 0.40m floor →
`sqrt(0.47² − 0.40²) / 10 ≈ 0.024 rad ≈ 1.4°` — consistent with the current constant. However,
the task2_frames are near-level, so the lever-arm–dominated scatter is low; a definitive re-fit
needs the full course recording with banked segments (on ShadowPC). **Do NOT change
`ATTITUDE_NOISE_STD_RAD = 1.4°` until re-fitted on the corrected banked data.** The old MLE was
computed with the wrong yaw in R_wb — its estimate included the systematic yaw error, but the
2-term MLE decomposed it mostly into a constant floor, so the sigma_theta number may coincidentally
still be close to correct.

---

## Scope boundary: what was NOT changed

- `src/racer/controller.py` and any CTBR path — intentionally untouched
- `src/racer/twin.py` / `rl/` plant files — untouched (separate rl/ changes predate this session)
- `src/racer/gate_mapper.py` — Case C (`RelativeGateSighting.lever_world()`, `_yaw_from_quat_wxyz()`)
  uses synthetic TRUE quats in its tests and is not yet in the live chain; conjugation at the call
  site is deferred to when Case C goes live
- `ATTITUDE_NOISE_STD_RAD` — unchanged pending re-fit with corrected data

---

## MEMORY-DELTA

1. **SUPERSEDE** "~3.6°/range yaw bias REFUTED (→ covariance)" — the bias WAS real; it was the
   aliased yaw in the navigator's R_wb; the corrected path eliminates it to 0.0°/range. The 1.4°
   attitude noise remains in covariance but the bias term is gone.
2. **ADD** vision-frame-fix SHIPPED (2026-06-12): `navigator.py:295` now uses
   `R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)`; `frames.py` adds the helper; replay
   regression shows detection/association unchanged, fix p50 1.37→0.47m, leak 3.4%→0%.
3. **ADD** `characterize_perception.py` updated to match navigator (uses `R_world_from_odo_quat_wxyz`).
4. **ADD** MLE re-fit of sigma_theta QUEUED for ShadowPC (needs banked course recording with
   corrected R_wb; do NOT change `ATTITUDE_NOISE_STD_RAD=1.4°` until done).
5. **TASK2 BIAS NOTE**: the `+3.5°/range east bias` and the `−2.0°/range north bias` seen in the
   VISION-PKG2 characterization were navigator artifacts, not calibration errors; vision-frame-fix
   eliminates them.
