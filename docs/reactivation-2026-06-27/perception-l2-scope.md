# Perception L2 Upgrades — Implementation Scope
*Scoping doc — READ ONLY. Generated 2026-06-28. Do NOT edit source or tests; implement from this cold.*

---

## Current data-flow map

```
Camera frame (JPEG, 30 Hz)
   │
   ▼
detector.detect(frame)
   → GateObservation (corners_px (N,2), corner_ids, corner_confidence, gate_id)
   │
   ▼ [navigator._process_observation]
predict_gates_in_camera(gates, kf.position, R_wb)
   → {gate_id: PredictedGate (R_cam_gate, t_cam_gate, center_px, corners_px)}
associate(obs, predicted)                          ← shape-normalised geometry match
   │
   ▼ gate_id
estimate_gate_pose(obs, prior=GatePose(pg.R_cam_gate, pg.t_cam_gate), compute_covariance=True)
   [src/racer/vision/gate_pose.py:303]
   │  uses SOLVEPNP_IPPE_SQUARE (4 corners) or solveP3P (3 corners)
   │  returns GatePose { R_cam_gate (3,3), t_cam_gate (3,), reproj_error_px, covariance (6,6) }
   │
   ▼ gate_pose.t_cam_gate (ONLY — R_cam_gate is NOT consumed for the position fix)
gate_pose_to_world_position(gate_pose, gate, R_wb)
   [src/racer/localization.py:90]
   lever = R_world_camera @ gate_pose.t_cam_gate   # ← ONLY t_cam_gate used here
   position_ned = gate.position_ned - lever         # = gate - L  (the "+L" sign convention)
   → (position_ned, cov)
   │
   ▼
LinearKF / RewindKF .update_position(position_ned, cov)
   │
   ▼ nav_state.position_ned
estimator_state_for_obs(ds, nav_state)             # [src/racer/estimator_obs.py:42]
build_obs(...)                                     # [rl/fly_rl.py:obs_from_zup / build_obs]
   pos_g = R_w2g @ (gate_pos - position_ned)       # ≡ R_w2g @ (+L) [pinned by test_obs_sign_faithfulness]
   → obs[0:3] = pos_g  (the "+L" slot, CRITICAL)
```

Key files:
- `src/racer/vision/gate_pose.py` — PnP / rotation / translation
- `src/racer/localization.py` — +L lever, world-fix + gate-relative fix
- `src/racer/navigator.py` — orchestration loop, prior injection, innovation gate
- `src/racer/estimator_obs.py` — estimator→policy obs seam (20-dim inc8)
- `src/racer/contracts.py` — `GateObservation`, `GatePose`, `Gate`, `NavState`
- `src/racer/frames.py` — `R_camera_from_body()`, `BORESIGHT`, `R_world_from_odo_quat_wxyz`
- `src/racer/ahrs/eqvio.py` — SE_2(3) Right-Invariant EKF (not yet wired to nav)
- `src/racer/ahrs/eqvio_landmark.py` — SOT(3) inverse-depth landmark (not wired)
- `tests/test_obs_sign_faithfulness.py` — **CRITICAL**: pins obs[0:3] = +L

---

## Upgrade A: DISCARD-PnP-ROTATION (square-flip)

### Is the ambiguity real?

**Yes — it is real and structural.**

A square planar target is symmetric under 90-degree rotation about its normal. Concretely, for a
head-on 1.5 m inner-square gate the four corners `(-h,+h), (+h,+h), (+h,-h), (-h,-h)` in gate
frame (gate_pose.py:83) map to the same four image locations under any rotation that permutes
them. This means the **gate-normal rotation** recovered by IPPE/PnP is structurally ambiguous
even when the four corners are perfectly detected. IPPE_SQUARE surfaces a well-known 2-fold
"flip" ambiguity (a tilt-toward-the-camera / tilt-away flip, documented in gate_pose.py:28–33);
the additional 90°/180°/270° symmetries mean the `R_cam_gate` rotation component about the gate
normal (+Z = downrange) is NEVER reliably recoverable from a square target.

**Confirmed by the literature:** the `candidate-stack-v2.md` cites three independent sources
(including MonoRace / Drift-Corrected-VIO + IPPE_SQUARE + confirmation in the handoff literature
synthesis) for "DISCARD rotation, keep translation". The `master-plan.md` lists it as
"DISCARD PnP rotation (square-flip ambiguity)".

### Is it load-bearing in our code?

**Critically NOT for the position fix, but YES for the IPPE disambiguation prior.**

`R_cam_gate` is consumed in exactly two downstream roles:

**Role 1 — Translation recovery (safe; rotation NOT needed).**
`gate_pose_to_world_position` (localization.py:118) uses ONLY `gate_pose.t_cam_gate`:
```python
lever = R_world_camera @ np.asarray(gate_pose.t_cam_gate, ...)   # line 119
position_ned = gate.position_ned - lever                           # line 120
```
`R_cam_gate` is never referenced here. The "+L" obs sign invariant pinned by
`test_obs_sign_faithfulness.py` depends on `t_cam_gate`, not `R_cam_gate`. **Discarding the
rotation component of GatePose does NOT touch +L.** Gate-relative in-plane fix
(`gate_relative_inplane_fix`, localization.py:160) likewise uses only `t_cam_gate`.

**Role 2 — IPPE 2-fold disambiguation prior (CRITICAL; currently uses R_cam_gate).**
`navigator._process_observation` (navigator.py:462–469) builds the PnP prior from the predicted
gate pose:
```python
pg = predicted[gate_id]   # PredictedGate.R_cam_gate = R_camera_world @ gate.R_world_gate
prior = GatePose(obs.frame_id, obs.sim_time_ns, pg.R_cam_gate, pg.t_cam_gate, 0.0, ...)
pose = estimate_gate_pose(obs, prior=prior, ...)
```
Inside `estimate_gate_pose` → `_estimate_ippe` (gate_pose.py:188–212) and `_estimate_p3p`
(gate_pose.py:215–235), the prior **rotation** is used as the IPPE tie-breaker
(`_rotation_geodesic(cands[i][0], prior.R_cam_gate)`, line 208) and as the P3P selector
(line 232). The 2-fold IPPE ambiguity is a **tilt flip**, not the square-normal rotation,
so the map-predicted attitude (R_camera_world @ gate.R_world_gate) still serves as a
good prior for THAT flip. This usage does NOT cause the "square-flip" problem — it resolves
a different, legitimate geometric ambiguity. The square-flip problem is specifically about
R_cam_gate's rotation component **about** the gate normal, which IPPE_SQUARE itself is
ambiguous about by design. The predicted prior breaks the IPPE 2-fold (camera-tilt) flip
correctly, but cannot and should not be used as authority on the gate-normal rotation.

**Role 3 — Covariance propagation (uses R, but only the [t(3), rvec(3)] layout).**
The Fisher / Monte-Carlo covariance returned in `GatePose.covariance` is a (6,6) matrix over
`[t(3), rvec(3)]` (gate_pose.py:285, 400). Only the `[:3, :3]` translation block is consumed
by `gate_pose_to_world_position` (localization.py:122). The rotation block is never used
downstream. ✓ Safe.

**Role 4 — `cam_in_gate` property (gate_pose.py:GatePose:185).**
`-R_cam_gate.T @ t_cam_gate` — not consumed anywhere in the live flight stack (no grep hits in
navigator/localization/estimator_obs). Safe to ignore for now.

**Role 5 — `_refine_pose` (gate_pose.py:238–288).**
The Gauss-Newton refinement uses R internally during iteration but the final refined R,t are only
stored in GatePose.R_cam_gate / t_cam_gate. Since only t_cam_gate propagates downstream, the
rotation refinement is computationally harmless (just extra work). However, a diverged GN step
is checked via `_refine_ok` which uses `_rotation_geodesic(R_r, R0) > 0.5` — this relies on
having a valid R0 reference from IPPE but does not use the unreliable gate-normal component.

### Minimal change to discard PnP rotation

**The upgrade is NOT to remove R_cam_gate from GatePose** — it is needed for the disambiguation
prior, for covariance estimation, and for future multi-gate fusion. The change is:

1. **Don't consume R_cam_gate for any pose-estimation purpose downstream** — which is already
   the case. The implementation is already correct in this sense; there is NO code path where the
   unreliable R_cam_gate rotation is used to influence the drone position estimate.

2. **If attitude estimation from PnP is ever attempted** (e.g., wiring PnP's `R_cam_gate` back
   into the AHRS to correct pitch/roll), this MUST NOT use the gate-normal component (the third
   column of R_cam_gate in the gate frame, which corresponds to the R_cam_gate[:, 2] column in the
   gate frame or equivalently the R components about the gate +Z axis). The valid part is the
   attitude-of-approach information: R_cam_gate encodes how the drone is tilted relative to the
   gate plane, which IS recoverable; only the in-plane rotation (spin about the normal) is
   ambiguous.

3. **For VQ2 / L2 implementation**: the "discard rotation" upgrade specifically means: when using
   PnP output for attitude correction (the ADR-VINS tight-fusion path mentioned in
   `candidate-stack-v2.md` §L3), decompose R_cam_gate into (a) the gate-plane tilt components
   (roll/pitch of the camera relative to the gate normal) and (b) the gate-normal spin. Use ONLY
   (a); derive (b) from the AHRS/EKF attitude estimate instead.

### Concrete upgrade spec (A1): Attitude observability extraction

File: `src/racer/vision/gate_pose.py` (future addition, not a change to existing code)

Add a helper function after the `estimate_gate_pose` function:

```python
def decompose_gate_tilt(R_cam_gate: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract the RELIABLE part of a square-gate PnP rotation.

    A square gate is rotationally symmetric about its +Z normal, so PnP cannot
    reliably recover the rotation ABOUT that axis (the gate-plane spin). What it
    CAN recover reliably is the tilt of the gate plane relative to the camera —
    i.e., the direction the gate normal points in the camera frame.

    Returns:
        gate_normal_cam: (3,) unit vector, the gate +Z (downrange) in the camera frame.
            This is R_cam_gate[:, 2] and IS reliable (the tilt part).
        spin_rad: float, the unreliable in-plane rotation angle about the gate normal.
            DO NOT USE for attitude estimation; exists for completeness / debugging only.

    The gate normal in the camera frame encodes the approach angle: for a head-on gate
    it is (0, 0, 1) (pointing along the optical axis). The tilt away from optical-axis
    encodes the drone's lateral/vertical offset approach angle.

    Attitude correction use (ADR-VINS path):
        gate_normal_world = R_world_camera @ gate_normal_cam
        This should align with gate.normal_ned (R_world_gate[:, 2]) under a correct
        attitude; the residual encodes attitude error PERPENDICULAR to the gate normal
        (2 DOF). The third DOF (spin about the normal) is unobservable from a square gate.
    """
    gate_normal_cam = R_cam_gate[:, 2].copy()   # gate +Z in camera frame (reliable)
    # In-plane spin: the angle of R_cam_gate's first column projected onto the gate XY plane.
    # Unreliable due to the square symmetry — for reference only.
    spin_rad = float(np.arctan2(R_cam_gate[1, 0], R_cam_gate[0, 0]))
    return gate_normal_cam, spin_rad
```

### Risk to +L obs-sign invariant

**Zero.** The "+L" obs-sign convention depends entirely on `t_cam_gate`, not `R_cam_gate`. The
test `tests/test_obs_sign_faithfulness.py` passes a `GatePose(R_cam_gate=np.eye(3), ...)` in its
noiseless construction (line 72), confirming that the sign contract does not depend on the
rotation value. Any future code that modifies how R_cam_gate is computed or used MUST not alter
t_cam_gate — that is the invariant.

### Test plan for Upgrade A

1. **Existing tests sufficient for the "discard" part**: the translation fix pipeline is already
   rotation-agnostic. Run `scripts/green_gate.py --full` to confirm no regression (expected:
   1047 pass / 44 skip, baseline on main).

2. **New test for `decompose_gate_tilt`** if the helper is added:
   - File: `tests/test_gate_pose_decompose.py`
   - Head-on gate (R_cam_gate = I): assert gate_normal_cam == (0, 0, 1), spin ~= 0.
   - Off-axis gate (known R): assert gate_normal_cam == R_cam_gate[:, 2].
   - Round-trip for the reliable tilt: gate_normal_cam must be invariant under 90-deg spin:
     construct four 90-deg rotations about Z, verify all return the same gate_normal_cam.
   - Attitude-correction residual sanity: for a given camera tilt, verify that the residual
     between gate_normal_cam (from PnP) and R_world_camera.T @ gate.normal_ned is consistent
     with the simulated attitude error.

3. **Do NOT add a test that verifies R_cam_gate is "correct"** — the whole point is it is
   ambiguous about the gate normal. The reliable quantity is gate_normal_cam (a function of R).

---

## Upgrade B: Bearing-range observation model

### What is being proposed

Rather than feeding the estimator a full 6-DOF pseudo-pose (position_ned from +L lever, with
anisotropic covariance), feed a **bearing-and-range** pair per detected corner (or per gate):

- **Bearing**: the direction of the gate corner in the camera frame (a unit 3-vector, or
  equivalently 2 spherical coordinates). This is the raw pixel observation de-projected through
  the intrinsics, NOT a PnP-derived quantity.
- **Range**: the metric depth of the gate (in metres), recoverable from the known 1.5 m gate
  size and the apparent pixel span. This is NOT the PnP solved depth; it is derived from the
  subtended-angle formula: `range = (gate_inner_size_m / 2) / tan(half_angle_rad)`.

The claim is that this measurement model is "better-conditioned at distance" because it separates
the bearing (low noise, direct from pixels) from the range (noise proportional to 1/range²) and
allows the EKF to weight them independently.

### Is this the same as what we already do?

**Partially. Key distinctions:**

The current pipeline runs `gate_pose_to_world_position` (localization.py:90) to produce a
**3-DOF world-position pseudo-measurement** from the PnP output. This collapses the bearing and
range information into a single position measurement with anisotropic covariance. The bearing
information is implicitly encoded in the position (the +L lever direction), but the EKF then
applies a 3-DOF position update that is dominated by the in-plane constraint (tight sigma 0.265 m)
and the loose along-track constraint (sigma 0.5 m + attitude lever arm).

A pure bearing-range model would instead:
1. Give the EKF a **2-DOF bearing measurement** (azimuth + elevation of the gate centre in the
   camera frame) — valid from any range, does not require knowing the range.
2. Give the EKF a **1-DOF range measurement** from the gate apparent size — range =
   `(0.75 m) / tan(apparent_half_angle)`. This IS attitude-independent and ε_vert-robust
   (the boresight vertical bias shifts the apparent gate centre but does NOT shift the subtended
   ANGLE significantly at distances > 5 m — the `candidate-stack-v2.md` §L3 "bearing-angle
   known-size range channel" observation).

### Fit to current estimator chain

**Option B1: Route through existing C2 chain (recommended for near-term)**

The bearing-range model can be approximated within the existing `LinearKF / RewindKF` 3-DOF
position update by:

1. **Range-derived depth update**: add a 1-DOF pseudo-measurement of the along-track range to
   the `localization.py` chain. The range from the gate apparent size bypasses the PnP along-
   track uncertainty.
2. **Tight in-plane constraint already approximated**: the gate-relative in-plane fix
   (`gate_relative_inplane_fix`, localization.py:160) already achieves the bearing-like
   in-plane constraint by using the tight PnP lateral sigma and loose along-track sigma.

The key **new addition** is the range channel from bearing-angle (step 1 above). This is a
1-DOF scalar update:

```
z_range = apparent_range_from_gate_span(obs.corners_px, gate.inner_size_m)
H_range = [0, 0, 0, ..., 1_along_track, ...]  # selects the along-track position from state
R_range = sigma_range(z_range)^2               # range-proportional noise
```

This replaces / supplements the `GATE_REL_ALONG_SIGMA = 0.50 m` loose prior in
`gate_relative_inplane_fix` with a **data-driven, attitude-independent range estimate** that
tightens with proximity.

Exact change to `localization.py`:
- Add `apparent_range_from_gate_span(corners_px, gate, K)` function (new, ~15 lines) that
  computes the range from the subtended angle.
- Add `gate_range_fix(range_m, gate, R_world_body, sigma_range)` function that builds a 1-DOF
  along-track measurement.
- Call from `navigator._apply_gate_relative_fix` after the existing in-plane fix, gated on its
  own 1-DOF chi2 test.

**Option B2: Route through SE_2(3) EqVIO / SOT(3) landmark (future)**

`src/racer/ahrs/eqvio.py` provides `SE23RightInvariantEKF.update_landmark(y_body, p_landmark_world)`
(eqvio.py:319) which implements the body-frame bearing measurement:
```
y = R^T (p_L - p) + noise
```
`src/racer/ahrs/eqvio_landmark.py` provides `SOT3` (inverse-depth parameterization) and
`bearing_jacobian_pose` / `bearing_jacobian_landmark` for full EqVIO co-estimation.

These are **correct, unit-tested** but NOT wired to the navigation loop. The `eqvio.py` comment
at line 57 ("next tier: gate-corner landmark bearings (SOT(3) inverse-depth)") confirms the
intended path. Wiring the full EqVIO bearing update requires:
1. Replacing `LinearKF` / `RewindKF` in `Navigator` with `SE23RightInvariantEKF`.
2. Augmenting the state with per-corner `SOT3` inverse-depth landmarks.
3. Implementing the marginalisation step (eqvio_landmark.py line 255, NEXT STEPS comment).

This is a **significant refactor** (~3–5 days) and breaks the VQ1 byte-identical fallback
(`NavigatorConfig.use_given_position=True`). It is the RIGHT long-term path but should NOT be
sequenced before the in-flight APPO critic fix and the deterministic-retrain gate are clear.

### Recommendation: sequence B1 first, B2 later

**For the next implementation sprint:**
- Implement B1 (range channel in the C2 chain) — ~1 day, no contract changes, VQ1 byte-identical.
- The EqVIO / SOT(3) path (B2) is a separate workstream that should only start after:
  (a) inc8 APPO retrain has produced a stable flyable policy, AND
  (b) the B1 range channel has been empirically validated on the sim as a useful addition.

### Overlap with EqVIO landmark work

The `eqvio.py` + `eqvio_landmark.py` modules are **research infrastructure** (correct but
NOT wired). They are the natural home for bearing-range once the full EqVIO path is pursued.
Critically, `eqvio.update_landmark` takes `y_body = R^T (p_L - p)` — a **known world landmark
position** — which in our case is one gate corner's world position (known from the track map).
The SOT(3) inverse-depth extension in `eqvio_landmark.py` is for the MAP-FREE case where the
corner world position must be estimated; if we have a track map (the recon-map path from
`candidate-stack-v2.md`), we can use `eqvio.update_landmark` directly with the known corner
world positions, bypassing the full EqVIO augmentation.

**Sequencing for the commander:**
1. B1 range channel (C2 chain) — implement now.
2. EqVIO bearing update with KNOWN map positions (eqvio.update_landmark) — after recon-map.
3. Full SOT(3) EqVIO inverse-depth (eqvio_landmark.py full wiring) — after gate-ordering solved.

### Concrete spec for Upgrade B1

**New function in `src/racer/localization.py`:**

```python
def apparent_range_from_gate_span(
    corners_px: np.ndarray,
    inner_size_m: float = GATE_INNER_SIZE_M,
    camera_matrix: np.ndarray | None = None,
) -> float | None:
    """Range (m) from the subtended angle of the gate's inner square.

    Uses the known 1.5 m inner size: range = (size/2) / tan(half_angle).
    The half-angle is the RMS corner distance from the gate centroid in image space,
    back-projected through the focal length: tan(half_angle) = apparent_radius_px / f.
    Returns None if fewer than 4 corners are supplied (3-corner P3P is not span-reliable).
    Attitude-INDEPENDENT: the subtended angle is determined by the gate width, not the
    camera tilt, so ε_vert boresight bias does not affect this measurement.
    """
    from racer.frames import CAMERA_INTRINSICS_K
    K = CAMERA_INTRINSICS_K if camera_matrix is None else camera_matrix
    pts = np.asarray(corners_px, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    centroid = pts.mean(axis=0)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    # RMS corner-centroid distance in normalised image coords (pixels / focal)
    dxn = (pts[:, 0] - centroid[0]) / fx
    dyn = (pts[:, 1] - centroid[1]) / fy
    half_angle = float(np.sqrt(np.mean(dxn**2 + dyn**2)))  # tan(half_angle) ≈ half_angle at dist
    if half_angle < 1e-6:
        return None
    return (inner_size_m / 2.0) / half_angle

def gate_range_fix(
    range_m: float,
    gate: Gate,
    R_world_body: np.ndarray,
    sigma_range: float = 0.30,
    fix_cov_floor_std: float = FIX_COV_FLOOR_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Along-track (gate-normal) range pseudo-measurement + covariance.

    Produces a 3-DOF position measurement equivalent to constraining only the along-track
    axis: z_ned = gate.position_ned - range_m * gate.normal_ned (point range_m behind gate).
    The covariance is tight along-track (sigma_range) and VERY loose in-plane (100 m), so the
    KF update is effectively a 1-DOF range correction that does NOT fight the in-plane fix.

    sigma_range grows with range: use sigma_range = max(0.05, 0.03 * range_m) in the caller
    (subtended-angle noise is proportional to pixel noise / range² → metric range noise ∝ range²
    near, but detector pixel noise is ~constant, so the actual curve is ~linear at racing ranges;
    empirically calibrate on sim data).
    """
    R_gate_to_world = np.asarray(gate.R_world_gate, dtype=np.float64)
    gate_normal_ned = R_gate_to_world[:, 2]   # gate through-direction (downrange) in world NED
    z_ned = gate.position_ned - range_m * gate_normal_ned
    # Very loose in-plane, tight along-track: effectively a 1-DOF range constraint
    var_ip = 100.0 ** 2      # 100 m in-plane variance: does not constrain lateral
    var_al = (sigma_range ** 2) + (fix_cov_floor_std ** 2)
    cov_gate = np.diag([var_ip, var_ip, var_al])
    cov_ned = R_gate_to_world @ cov_gate @ R_gate_to_world.T
    return z_ned, cov_ned
```

**Navigator wiring (`src/racer/navigator.py`):**
- Add `use_range_channel: bool = False` to `NavigatorConfig` (line ~265 neighbourhood).
- In `_apply_gate_relative_fix` (navigator.py:527), after the existing in-plane fix call,
  optionally call `apparent_range_from_gate_span` on `obs.corners_px` and apply
  `gate_range_fix` through `_apply_pos_fix` with its own 1-DOF chi2 gate (chi2(1, 0.999) = 10.83).

**GatePose contract**: no change needed. Range is derived from `obs.corners_px` (the raw
detector pixels), not from the PnP output. This means the range channel is available even in
the 3-corner P3P regime (though we should gate it to 4 corners for reliability — the RMS span
from 3 asymmetric visible corners is noisy).

**VQ1 compatibility**: `use_range_channel=False` by default. When False the navigator loop is
byte-identical to main.

### Test plan for Upgrade B1

1. **`tests/test_gate_range_fix.py`** (new):
   - Unit test `apparent_range_from_gate_span` with known corners projected at known range:
     for a head-on gate at 10 m, with perfect intrinsics, the recovered range must match to
     < 0.5% (pixel quantization noise budget).
   - Attitude-independence: rotate R_wb by ±15 deg, verify range changes < 0.1 m at 10 m
     (the ε_vert robustness claim; rotation changes the centroid pixel but not the span).
   - `gate_range_fix` covariance shape: verify that cov_ned along-track eigenvalue = sigma_range²+floor²,
     and in-plane eigenvalues >> 1000 m².
2. **`tests/test_navigator_range_channel.py`** (new):
   - Navigator with `use_range_channel=True`, synthetic gate at known range; after N fixes, KF
     along-track uncertainty must be < along-track prior uncertainty (the fix helps).
   - `use_range_channel=False` byte-identical to existing tests: wrap existing navigator tests.
3. **Regression**: `scripts/green_gate.py --full` sentinel 1091 tests must pass.

---

## VQ1-specific items to avoid

The following are VQ1-ONLY data flows that must NOT be touched by L2 upgrades:

1. **`use_given_position=True` path** (navigator.py:412): feeds `ds.position_ned` from
   `LOCAL_POSITION_NED` as a tight KF update. This is the VQ1 eval path. L2 upgrades must
   be gated behind `use_gate_relative=True` / `use_range_channel=True` (both default False)
   and must not affect the given-position path.

2. **`R_cam_gate` disambiguation prior in IPPE** (gate_pose.py:208, 232): uses the map-predicted
   gate rotation to break the IPPE 2-fold flip. This is TRACK-AGNOSTIC (works on any track via
   the prior from `predict_gates_in_camera`). The square-flip ambiguity analysis above shows
   this IS correct for the 2-fold tilt-flip, NOT a bug. Do not "fix" it by discarding the
   rotation prior.

3. **The boresight `vert_offset_m=-0.25` in `frames.BORESIGHT`** (frames.py:58): this is the
   metric boresight correction baked for the VQ1 camera rig. In VQ2 a fresh calibration is
   needed; do NOT assume this constant is correct for a different sim environment.

4. **`gate_mapper.py` / `track_map.json`**: per MEMORY.md directive these are TRAINING/DEV-only
   for VQ2. The no-map assumption means the map is either recon-built (see candidate-stack-v2.md)
   or the gate-relative fix operates without a map. No L2 upgrade should introduce a hard
   dependency on a pre-built `track_map.json` at inference time.

5. **`obs_from_zup` / `build_obs` sign convention** (rl/fly_rl.py, imported by estimator_obs.py):
   The inc7 17-dim obs `[0:3] = pos_g = R_w2g @ (+L)` is frozen by `test_obs_sign_faithfulness.py`.
   Any change to `localization.py`'s `t_cam_gate` consumption MUST preserve this sign. The
   inc8 20-dim extension adds `obs[17:20] = confidence_triple` (estimator_obs.py:129–168) which
   is ORTHOGONAL to the L2 changes.

---

## Summary table

| Item | File(s) | Lines | Risk to +L | Status |
|------|---------|-------|-----------|--------|
| Square-flip ambiguity in PnP | gate_pose.py | 83–94, 140–157 | Zero (R not used for lever) | REAL but already NOT downstream-consumed |
| R_cam_gate used for IPPE prior | gate_pose.py:208, navigator.py:468 | — | Zero (prior → R disambiguation only) | CORRECT, do not remove |
| R_cam_gate used for position fix | localization.py:119 | — | N/A | NOT used — rotation already discarded |
| decompose_gate_tilt helper (new) | gate_pose.py | after line 376 | Zero | Implement as needed for attitude fusion |
| B1 range channel | localization.py, navigator.py | after line 207, ~530 | Zero (separate scalar update) | Implement behind NavigatorConfig flag |
| B2 EqVIO bearing update (future) | ahrs/eqvio.py:319, navigator.py | — | Requires contract audit | Deferred: after recon-map + APPO stable |
| EqVIO SOT(3) (far future) | ahrs/eqvio_landmark.py | — | Breaking refactor | Deferred: after gate-ordering solved |

---

## MEMORY-DELTA (≤6 lines for the commander)

```
MEMORY-DELTA 2026-06-28 / perception-l2-scope.md:
- SQUARE-FLIP AMBIGUITY: real in PnP (structural, square symmetry → gate-normal rotation unreliable)
  but R_cam_gate is ALREADY NOT CONSUMED for the position fix in our pipeline. Discarding PnP
  rotation = zero-code-change for the position path; add decompose_gate_tilt() helper only if
  attitude-from-PnP fusion is later needed.
- BEARING-RANGE (B1): route through C2 chain, not EqVIO. Add apparent_range_from_gate_span() +
  gate_range_fix() to localization.py; gate behind NavigatorConfig.use_range_channel=False.
  Attitude-independent range channel replaces loose GATE_REL_ALONG_SIGMA=0.5 m prior.
- EQVIO (B2): eqvio.py / eqvio_landmark.py are correct + tested but UNWIRED. Sequence after
  recon-map (known corner positions → eqvio.update_landmark directly) and APPO stable.
- Spec path: docs/reactivation-2026-06-27/perception-l2-scope.md (staged, not committed).
```
