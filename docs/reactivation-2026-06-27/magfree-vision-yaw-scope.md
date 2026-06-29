# Mag-free / baro-free case-C: pinning YAW and Z from vision (VQ2 design scope)

**Author:** background recon/scoping agent (opus-4.8) · **Date:** 2026-06-29 · **Status:** SCOPE ONLY (no source/tests touched)
**Audience:** the future build agent who closes the case-C estimator under the VQ2-confirmed sensor set. Act from this doc cold.
**Sibling docs:** `case-c-integration-scope.md` (the attitude seam + build plan), `gyro-plumbing-audit.md` (gyro→DroneState), recon `handoff/vq2-recon-2026-06-29/RECON.md`.

---

## 0. The one-paragraph verdict

VQ2 (build 1.0.3379) confirms the wire carries **accel + gyro only** — `HIGHRES_IMU.fields_updated = 63` → **no magnetometer, no barometer (both NaN)**, and `ATTITUDE`/`LOCAL_POSITION_NED`/`ODOMETRY` blocked. Consequence for the attitude/position filter: **gravity (accel) observes roll/pitch; nothing inertial observes yaw or z.** A standalone ESKF/AHRS holds roll/pitch indefinitely but its **yaw is a free integrator on gyro bias (drifts unbounded)**, and double-integrated accel z drifts. **Both yaw and z MUST be pinned by vision.** The good news from the recon frames: the warehouse is a **Manhattan-world scene** — a regular rectilinear floor grid, a parallel ceiling truss, vertical pillars, and blue lane-lines converging to the next gate (frames 02/04/05/07). That structure gives an **absolute heading reference (vanishing points)** and a **ground-plane height reference (floor)** that persist even with no gate in view. **RECOMMENDATION: option (B-extended) — a light vision-yaw + vision-z correction injected into the existing ESKF + C2 navigator chain, NOT a full EqVIO-spine rewrite.** Specifically: (1) a **gate-corner-bearing yaw pseudo-measurement into the ESKF** as the primary yaw lock (relative yaw to the active gate IS observable; the square-flip is broken by the temporal/PnP-prior machinery already in `gate_pose.py`, and we use a yaw *pseudo-measurement* from the +L lever direction, not the unreliable planar-PnP rotation matrix); (2) a **vanishing-point yaw pseudo-measurement** as the absolute backstop for the no-gate-in-view stretches; (3) **z is pinned by the existing gate-relative vertical fix** (already in the C2 chain) plus an optional **floor-plane height channel** for no-gate coverage. EqVIO stays on the shelf as the documented escalation if the light path proves insufficient.

---

## 1. OBSERVABILITY — where does a reliable yaw reference come from?

### 1.0 The problem precisely

The ESKF (`ahrs/eskf.py`) error-state is `[δφ(3), δb_g(3)]`. Its only correction is the accel tilt update (`_update_accel`, eskf.py:255), whose Jacobian is `H[:, :3] = -_skew(g_hat)` (eskf.py:301). Because `g_hat ≈ [0,0,1]`, that `skew` has a **zero column on the gravity (yaw) axis** — the accel update is **structurally blind to yaw** (rank-2 in attitude). The mag update (`_update_mag`, eskf.py:325) was the yaw observer; with `mag_ned=None` it never runs (eskf.py:213 guard). So:

- **roll, pitch**: observable from gravity, bounded error. ✅
- **gyro bias (roll/pitch axes)**: observable through the accel update over time. ✅ (partial)
- **yaw**: **unobservable inertially.** Integrates `(ω_z − b̂_gz)·dt`. The yaw-axis gyro bias is **never corrected** → yaw drifts linearly at the residual bias rate. At a plausible MEMS bias of even 0.2°/s, that is ~12°/min and climbs — far past the ~1° budget the KF predict assumes (`ATTITUDE_NOISE_STD_RAD`). ❌
- **z (altitude)**: KF state, propagated by `LinearKF.predict(accel_body, R_wb, dt)`. With no baro and no given position, z is double-integrated accel → drifts quadratically. Only a vision fix with a vertical component pins it. ❌

So we need a **yaw observer** and a **z observer**, both from vision. The three candidate yaw sources, assessed:

### 1.1 Candidate (a): gate-corner PnP yaw — RESOLVING THE SQUARE-FLIP TENSION

**The prior decision** (memory: "discard-PnP-rotation … square flip") was to NOT trust the PnP **rotation matrix** `R_cam_gate` for the drone's world yaw, for two real reasons:
1. **Square-flip / 2-fold planar ambiguity**: a planar square under IPPE returns two poses (`gate_pose.py:28-34`); the wrong one is a large rotation error. Worse, a *square* has an additional 4-fold in-plane spin ambiguity if corner identity is lost.
2. **Planar-PnP rotation is noisy**: the out-of-plane tilt of a near-frontal square is weakly constrained (the classic planar-PnP shallow-angle degeneracy) → `R_cam_gate`'s tilt/yaw components are high-variance.

**This does NOT mean yaw is unobservable from the gate — it means we must extract yaw the RIGHT way.** Resolution, three points:

- **The square-flip is already handled for the quantity we use.** The whole localization chain (`gate_pose_to_world_position`, `gate_relative_inplane_fix`) consumes only `t_cam_gate` (the **translation**, the +L lever), NOT `R_cam_gate`. The 2-fold flip is disambiguated in `estimate_gate_pose` by the temporal prior + reprojection-ratio logic (`gate_pose.py:28-41`, `PRIOR_DISAMBIG_MAX_RATIO`) and the post-PnP depth-sanity + Mahalanobis + relative-innovation gates in the navigator (`navigator.py:591-672`). A flipped solve is caught and dropped. So **the lever direction `L = R_wc @ t_cam_gate` is a trustworthy, flip-protected quantity.**

- **Relative yaw to the active gate IS observable from the lever, attitude-light.** Here is the key move that sidesteps the unreliable PnP *rotation*. The drone's body-frame **bearing to the gate centre** (the direction of `t_cam_gate`, a well-conditioned PnP quantity even when the rotation is not) combined with the **known world position of the active gate** (we know which gate via `RACE_STATUS.active_gate_index`, and the gate's world position from the dev-built map / running estimate) yields a constraint on world yaw: the bearing rotated by the *current yaw estimate* must point from the drone toward the gate. Formally this is exactly the **`update_landmark` measurement already implemented in `eqvio.py:317`** (`y = R^T(p_L − p)`, a body-frame relative-position observation) — but we can inject its **yaw component** as a scalar pseudo-measurement into the ESKF without adopting the full SE_2(3) filter (see §3). Because the gate is a *distant, off-boresight* point for most of the approach, its bearing has real yaw leverage (a yaw error rotates the predicted bearing measurably). This is **relative** yaw (drone-to-gate), which is precisely what the gate-relative obs needs; the absolute warehouse-frame spin is supplied by (b).

- **Caveat — bearing yaw leverage collapses head-on.** When the gate is dead-ahead and centred, its bearing is along boresight and carries little yaw information (a yaw error slides the gate sideways only at second order). So the gate-bearing yaw lock is **strong on approach / off-axis, weak at the moment of transit.** That is acceptable: yaw drift over one inter-gate segment (≈1–2 s) at a bias-corrected residual is small, and (b) covers the gaps.

**Verdict (a): USABLE and recommended as the primary yaw lock — but as a BEARING pseudo-measurement to the known active-gate position, NOT as the PnP rotation matrix.** Cost: low — reuses `t_cam_gate` and the active-gate world position already in the loop; the ESKF needs one new scalar/2-vector update method. Robustness in low-light/bloom: inherits the detector's corner quality; the bearing (centroid direction) is far more bloom-robust than the corner *span* (RECON §2 flags span/range as the bloom-sensitive quantity, not the centroid bearing).

### 1.2 Candidate (b): vanishing points from the structured environment — ABSOLUTE heading

The recon frames are decisive (02, 04, 05, 07): the scene is a **Manhattan world**. There is a regular **floor grid** (two orthogonal horizontal vanishing directions), a **ceiling truss** of parallel beams (same two directions), vertical **pillars** (the vertical vanishing direction = gravity, already pinned by accel), and **blue lane-lines** that run straight down the racing axis and converge to a sharp vanishing point (frame 02). Crucially, this structure is **present even when no gate is in view** (frame 07).

Two orthogonal horizontal vanishing points (VPs) fix the camera's **absolute yaw in the warehouse frame** up to a 90° lattice ambiguity (which VP is "x"). The lattice ambiguity is resolved by continuity (track the VP across frames; never let it jump 90°) and/or by the lane-lines, which mark the *unique* down-course axis. The vertical VP cross-checks roll/pitch against the accel solution.

**This is the absolute, drift-free yaw reference** — the function the magnetometer would have served. It does not depend on knowing any gate or map; it depends only on the warehouse being axis-aligned (which the frames confirm). It is the backstop that bounds yaw during no-gate stretches and recovers from any accumulated gate-bearing yaw error.

**Cost: medium.** A VP estimator is new code: detect long line segments (the truss beams / floor grid / lane-lines are high-contrast against the dark scene — segmentable even at mean-gray 36), cluster by direction, intersect for VP(s), convert the dominant horizontal VP to a yaw angle, inject as a yaw pseudo-measurement into the ESKF. There is **no existing VP code in the repo** — this is the one genuinely new perception component. Mitigations that make it cheap-ish: the lane-lines alone (a single bright-blue, easily-thresholded, straight, course-aligned feature) give a usable down-course-axis yaw without a full multi-line VP solver; start there. Robustness in low-light/bloom: lines are bright-on-dark and straight → robust; the main failure is motion blur (RECON frames 10–12) smearing thin lines — gate the VP update on a blur/line-count quality metric.

**Verdict (b): the ABSOLUTE yaw anchor — RECOMMENDED as the drift bound, phased in after (a).** Start with the lane-line down-course axis (cheap), escalate to a full floor-grid/truss VP solver if needed.

### 1.3 Candidate (c): temporal corner+grid tracking in a joint EqVIO filter

`EqVIOJointEKF` (`eqvio.py:552`) already implements exactly this: track gate-corner (and, in principle, grid-feature) **bearings** over a baseline; the SE_2(3) right-invariant pose filter co-estimates **yaw + z + full pose** with the SOT(3) inverse-depth landmarks, carrying the pose↔landmark cross-covariance (`update_landmark_joint`, eqvio.py:751). Over a translating baseline, bearings to *any* tracked static feature constrain yaw and z (this is how monocular VIO works at all).

**This is the most powerful and most principled option** — a real bearings-only VIO that needs no gate map, no Manhattan assumption, just trackable features (which the textured floor/truss provides in abundance). It is also **the most expensive to wire**: it replaces the ESKF+LinearKF spine with the joint filter, needs a **feature front-end** (detect + track corners/grid points frame-to-frame — also new code, comparable to the VP front-end), landmark lifecycle management (add/drop, `add_landmark`/`drop_landmark` exist but the *tracker* feeding them does not), and a full re-validation of the obs seam against the new state representation. The metric scale comes from the gate's known size / the IMU accel — observable but with the usual VIO scale-transient.

**Verdict (c): the ESCALATION path, not the first build.** It subsumes (a) and (b) but at 3–5× the integration cost and a larger regression surface. Keep it gated and on the shelf; reach for it only if the light (a)+(b) path leaves yaw/z out of budget under aggressive motion.

### 1.4 Observability summary table

| Source | Observes | Reliability | New code | Bloom/low-light | Verdict |
|---|---|---|---|---|---|
| accel gravity (existing ESKF) | roll, pitch | high, bounded | none | n/a | keep — but yaw-blind |
| (a) gate-bearing → known active gate | **relative yaw** (drone↔gate), weak head-on | good off-axis, weak at transit | small (1 ESKF update) | bearing/centroid robust (unlike span) | **PRIMARY yaw lock** |
| (b) vanishing points / lane-lines | **absolute yaw** (warehouse frame), drift-free | high when lines visible | medium (VP/line front-end) | bright-on-dark lines robust; blur-sensitive | **ABSOLUTE backstop** |
| (c) EqVIO joint bearings | yaw + z + pose, mapless | highest, but transient scale | large (feature tracker + filter swap) | feature-dependent | **ESCALATION only** |

**YAW VERDICT: yaw IS observable.** Primary = gate-bearing pseudo-measurement to the known active-gate position (flip-safe because it uses the lever/bearing, not the PnP rotation; the existing disambiguation gates protect it). Absolute drift bound = vanishing-point / lane-line heading from the Manhattan scene. The square-flip tension is resolved by **never using `R_cam_gate` for yaw** — yaw comes from the well-conditioned bearing + the structured environment, both of which dodge the planar-PnP rotation degeneracy entirely.

---

## 2. Z (altitude) — how is z pinned?

z has three candidate observers; the first already exists.

1. **Gate-relative vertical fix (EXISTS, recommended primary).** The +L lever fix (`gate_pose_to_world_position` / `gate_relative_inplane_fix`, localization.py:91/161) produces a full 3-D world position whose **vertical (world-down) component pins z** relative to the active gate's known height. The gate-relative path already shapes covariance in the gate-plane frame; the gate-plane **in-plane Y axis is world-down-ish** (`_frame_from_through`, navigator.py:174), so the vertical centering is part of the tight in-plane fix. **z is therefore already pinned whenever a gate is in view** — no new code. The known systematic here is the **vertical boresight bias ε_vert ≈ 0.215 m** (memory footgun): it is a genuine sighting bias, owned by the boresight/ESKF pathway (`frames.BORESIGHT.vert_offset_m`, localization.py:85), not by z estimation per se. Watch it, do not re-litigate it here.
   - **Map dependency caveat:** this pins z relative to the *active gate's known world height*. We do not have a delivered gate map (TRACK_INFO blocked), but the working assumption (memory NOW block) is a dev/training-built gate map of world positions keyed by `active_gate_index`. z is pinned **relative to that map's gate heights**. If the competition track is randomized per-load, the *along-course* gate heights still come from the running map estimate; the **floor-plane channel (below) is the map-free vertical anchor.**

2. **Floor-plane height channel (NEW, recommended for no-gate coverage).** The floor grid is visible in nearly every frame (02, 04, 05, 07). Fitting the ground plane (or just using the floor-grid vanishing line / horizon) gives the **camera's metric height above the floor** directly, from the known camera tilt (20°, spec-exact) + the apparent geometry of the grid — an **absolute, map-free z reference** in the warehouse frame. This is the z-analogue of the VP yaw anchor: it bounds z drift during no-gate stretches and is independent of the gate map. Cost: medium, shares the line/grid front-end with (b). Robustness: the floor grid is lower-contrast than the lane-lines/truss in the darkest frames (07) — gate on grid-detection quality.

3. **EqVIO landmark depth (ESCALATION).** The joint filter's inverse-depth landmarks observe z through translating bearings, same as yaw. Same verdict as §1.3 — powerful, expensive, on the shelf.

**Z VERDICT:** z is pinned **primarily by the existing gate-relative vertical fix** (no new code; in budget whenever a gate is in view), backstopped by a **new floor-plane height channel** for map-free, no-gate-in-view coverage. The bearing-range channel (`apparent_range_from_gate_span`, localization.py:230) is **along-track**, not vertical — it does not pin z and must not be confused with a z source.

---

## 3. ARCHITECTURE — pick a path

### The three options, weighed

**Option A — EqVIO joint filter as the spine.** Replace ESKF+LinearKF with `EqVIOJointEKF`; gate-corner + grid-feature bearings → yaw+z+pose; ESKF demoted to roll/pitch init.
- *Pros:* maximal capability, mapless yaw/z, principled consistency (the RI-EKF NEES edge), subsumes (a)+(b)+floor.
- *Cons:* **largest integration cost** — needs a feature tracker front-end (does not exist), swaps the entire estimation spine (re-validate `make_nav_state`, the obs[0:20] seam, the `#37` fidelity pins, the C2 gates), and a big regression surface. The `use_ahrs` seam, the +L invariant, the confidence/age channels all assume the ESKF+LinearKF shapes. Disproportionate for a first pass.

**Option B — light vision-yaw/z correction into the existing ESKF + C2 chain.** Inject a yaw pseudo-measurement (gate-bearing primary + VP absolute) into the ESKF; pin z via the existing gate-relative vertical fix + a new floor-plane channel. No filter swap.
- *Pros:* **smallest delta**, composes cleanly with the just-wired `use_ahrs` seam (the ESKF is already owned and stepped by the Navigator, navigator.py:504-532), preserves +L / obs / confidence invariants byte-for-byte when gated OFF, reuses the existing fix/gate machinery. Each piece is independently gated and testable.
- *Cons:* yaw lock is weak exactly at transit (mitigated by VP backstop + short segments); needs the VP/floor front-end (new but bounded); does not get the full VIO consistency edge.

**Option C — hybrid.** B now; keep EqVIO gated and on the shelf as the documented escalation if B's yaw/z budget fails under aggressive motion.

### RECOMMENDATION: **Option C — ship B, hold EqVIO in reserve.**

Rationale: the binding constraint is **getting a self-consistent, in-budget yaw+z onto the wire with the smallest regression surface and the strongest preservation of the pinned invariants** (`+L`, obs `#37`, confidence/age, OFF==byte-identical). B does that. It slots **into the `use_ahrs` ESKF the team just wired** rather than discarding it. The square-flip tension evaporates because B never uses the PnP rotation. EqVIO is genuinely better but is a research-grade swap whose cost is not justified until B is shown insufficient — and B gives us the data to make that call. Memory's existing direction ("ESKF picked"; "vision = SUPPORT, do not lean on it") aligns with B.

How B composes with the constraints:
- **square-flip:** dodged — yaw from bearing + VP, never `R_cam_gate`.
- **low-light/bloom:** bearings (centroid) and bright lane-lines are the bloom-robust quantities; the bloom-sensitive *span* is confined to the existing along-track range channel, not the yaw/z path.
- **`use_ahrs` seam:** the yaw pseudo-measurement is a new ESKF update method called inside `_step_ahrs` (navigator.py:504); everything else in that seam (the ODOMETRY-convention re-encoding, GAP #2 routing) is unchanged.
- **+L / obs invariants:** yaw enters only through `R_wb`; as long as the AHRS still emits a true FRD→NED quat, the +L lever and obs seam are untouched (the existing OFF==byte-identical discipline from `case-c-integration-scope.md` §5 carries over verbatim).

---

## 4. CONCRETE CHANGES for the recommended path (Option B), sequenced & gated

Every step defaults **OFF / byte-identical** and is guarded by a new sub-flag, mirroring the existing `use_ahrs` / `use_gate_relative` discipline. Run `scripts/green_gate.py` (sentinel 1091) after each. Steps marked **[P]** are independent.

### Step 0 — prerequisites (already done or scoped elsewhere)
- Gyro plumbing (`gyro-plumbing-audit.md`): `DroneState.gyro_body` populated from `HIGHRES_IMU.xgyro/ygyro/zgyro`. The ESKF must consume `gyro_body`, never `angular_rate_body`. **Confirm this landed** (it is the input to everything below).
- `use_ahrs` seam wired (memory: done, cf16361 / da280e2). The ESKF is owned + stepped in `Navigator._step_ahrs` (navigator.py:504-532). **All §4 yaw updates hook in here.**
- Mag param in `ahrs_adapter` / `eskf` is now always `None` — leave it; the mag update path stays dormant (eskf.py:213 guard). Do not delete it (cheap dormant code; a future mag-bearing env could use it).

### Step 1 — [P] ESKF: make the mag-free yaw-blindness explicit + add a generic yaw pseudo-measurement
**Files:** `src/racer/ahrs/eskf.py`.
- Add a method `update_yaw(yaw_meas_world: float, yaw_noise_std: float)` (or `update_heading`): a scalar pseudo-measurement on world yaw. Reuse the existing `_apply_eskf_update` core (eskf.py:366). The measurement model is the yaw component of the attitude error: `H = [0,0,1, 0,0,0]` in the **world-yaw-about-gravity** sense — but must be expressed in the ESKF's **body-frame δφ** convention. Derive carefully: the innovation is the wrapped angle difference `wrap(yaw_meas − yaw_hat)`; the sensitivity of world-yaw to the body-frame δφ near level is `≈ δφ_z` (gravity-aligned), so `H[0,2]=1` is correct at small tilt — but at large tilt project properly (`d(yaw)/d(δφ) = e_z^world expressed in body = R_wb^T e_z_world`). Pin with a finite-difference test (the eskf bench style). This is the **single shared injection point** for both yaw sources (a) and (b).
- **Mark yaw unobservable without it:** add a docstring note / optional `attitude_uncertainty` split so the yaw-axis covariance is reported separately (it grows unbounded until a yaw update lands — useful for the confidence channel and for gating).
- **Gated:** new method is additive; no existing call path invokes it → byte-identical. Default ESKF behaviour unchanged.
- **Test:** `tests/test_eskf_yaw.py` — (i) FD-check the yaw Jacobian; (ii) on a synthetic spinning trajectory (`ahrs/traj6dof.py` + `imu_gen.py`) with a gyro-z bias, show yaw drifts without `update_yaw` and stays bounded (≤~1°) with periodic `update_yaw`; (iii) roll/pitch unaffected.

### Step 2 — [P] VP / lane-line heading front-end (absolute yaw source (b))
**Files:** new `src/racer/vision/heading_vp.py`.
- Input: a `Frame` (BGR). Output: `(yaw_world_meas, quality)` or `None`. Phase 2a (cheap): threshold the **blue lane-lines**, fit the dominant straight line, take its vanishing-point direction → the down-course axis → world yaw (the course axis defines warehouse yaw=0 by convention). Phase 2b (escalation): full multi-line detection (LSD / Hough on the bright truss+grid edges), cluster directions, intersect for the horizontal VP, resolve the 90° lattice ambiguity by temporal continuity.
- **Quality gating:** return `None` (or low quality) on motion blur (frames 10–12) / too-few lines. The navigator only injects the update when quality clears a threshold.
- **Gated:** standalone module; nothing calls it until Step 4 wires it behind a flag.
- **Test:** `tests/test_heading_vp.py` — synthetic rendered grid/lines at known yaw → recovered yaw within budget; the recon frames (`handoff/vq2-recon-2026-06-29/frames/curated/02,04,05,07`) as **smoke fixtures** (assert it returns a finite plausible heading on 02/04/05, and `None`/low-quality on the blurred 10–12). Use recon frames only as fixtures, never as a performance target.

### Step 3 — [P] Floor-plane height channel (map-free z source)
**Files:** new `src/racer/vision/floor_height.py` + a `floor_height_fix(...)` in `localization.py` mirroring `gate_range_fix` (an anisotropic world-position fix tight on the **world-down** axis, huge in-plane).
- Input: `Frame` + `R_wb` + camera tilt (20°). Output: camera height above floor → a z pseudo-fix. The fix covariance is tight in world-down, ~∞ in horizontal, so `LinearKF.update_position` acts as an effective 1-DOF z correction (exact analogue of the along-track range fix, localization.py:275). PRESERVE the +L vertical-offset convention (`_apply_camera_vert_offset`).
- **Gated:** new `NavigatorConfig.use_floor_height=False`; new localization fn is additive.
- **Test:** `tests/test_floor_height.py` — synthetic floor at known height/tilt → recovered z; OFF path byte-identical; recon frames as smoke fixtures.

### Step 4 — [D:1,2,3] Wire yaw + z corrections into the Navigator behind sub-flags
**Files:** `src/racer/navigator.py`, `NavigatorConfig`.
- New flags (all default `False`):
  - `use_gate_bearing_yaw` — source (a). In `_process_observation` (after a fix is accepted, navigator.py:619+), compute the body-frame bearing to the **known active-gate world position** and call `self._ahrs.eskf.update_yaw(...)` with the yaw implied by that bearing + the gate's world position vs the current KF position. (Reuse `t_cam_gate` direction; this is the flip-safe quantity.) Gate it on the same acceptance the fix already passed, and skip when near head-on (low yaw leverage — check the bearing's off-boresight angle).
  - `use_vp_yaw` — source (b). In `_maybe_run_vision` (per-frame), call `heading_vp` and, on sufficient quality, `self._ahrs.eskf.update_yaw(yaw_vp, σ_vp)`. This runs **even with no gate associated** (the no-gate yaw backstop).
  - `use_floor_height` — Step 3, per-frame floor z fix into the KF (innovation-gated like the other fixes).
- **Active-gate index:** thread `RACE_STATUS.active_gate_index` (ENCAPSULATED_DATA, RECON §1) into the Navigator so (a) knows which gate is the target. Confirm it reaches `DroneState`/the nav loop; if not, add it (additive field, like `gyro_body`).
- **Ordering:** yaw updates must land **before** the KF predict uses `R_wb` for the next tick (i.e. update the ESKF yaw inside/just after `_step_ahrs`), so the corrected attitude propagates position. Re-read the cached `_ahrs_odo_quat` after a yaw update (it is recomputed in `_step_ahrs`; ensure the yaw update happens first or the cache is refreshed).
- **Gated:** every flag OFF → `_step_ahrs` and the fix path are byte-identical to the current `use_ahrs` implementation.
- **Test:** extend `tests/test_casec_ahrs_loop.py` (the dual-Navigator harness from `case-c-integration-scope.md` §3): add a **yaw-drift scenario** — a trajectory with a gyro-z bias and a turning course. Truth-Nav uses true attitude; Case-C-Nav uses ESKF + the yaw/z channels. **Pass:** post-convergence yaw error p90 ≤ budget (suggest ≤~2–3° given the weak-at-transit caveat; tighten with sim data), z error within the gate-relative floor, obs[6:9] `rpy_g` tracks truth, OFF path byte-identical.

### Step 5 — [D:4] Case-C deploy profile update (extends GAP #5)
**Files:** wherever the case-C `NavigatorConfig` preset is built (`case-c-integration-scope.md` §4 step 5).
- Flip the new flags ON in the case-C profile only: `use_gate_bearing_yaw=True, use_vp_yaw=True, use_floor_height=True` (alongside the existing `use_ahrs=True, use_given_position=False, use_gate_relative=True, use_rewind_kf=True, use_range_channel=True`). Default config untouched.
- **Test:** smoke — the profile constructs and runs end-to-end on synthetic no-given-position input with a turning course; emits finite, in-bounds actions; yaw bounded.

### Step 6 — [D:5] (optional) EqVIO escalation spike, GATED, only if Step 4 yaw/z is out of budget
**Files:** a throwaway `scripts/casec_eqvio_spike.py` (NOT wired into the nav loop). Drive `EqVIOJointEKF` over the same synthetic trajectory; compare yaw/z to Option B. This is the decision gate for whether to pay for Option A later. Do not wire it into the live loop in this pass.

### Sequencing summary
Steps **1, 2, 3 are independent [P]** (one ESKF method, two vision front-ends) — three parallel agents. Then **4 → 5** is the serial spine. **6 is optional/conditional.** Net: 3 parallel up front, 2 serial, 1 optional.

---

## 5. RISKS / INVARIANTS (in addition to those in `case-c-integration-scope.md` §5)

- **Yaw Jacobian convention (highest risk).** The ESKF error is **body-frame δφ**; the yaw measurement is **world yaw about gravity**. The mapping is `H = (R_wb^T e_z_world)^T` into the δφ block, exact at all tilts; the small-angle `H[0,2]=1` is only valid near level. Get this wrong and the yaw update fights roll/pitch. **FD-pin it** (Step 1 test). The dormant `_update_mag` (eskf.py:325-362) already does a yaw-only update with `H[0,2]=1` — that is the near-level shortcut; the new method should do the proper projection and the mag method can be left as-is (dormant).
- **Angle wrapping.** `update_yaw` innovation must wrap to (−π, π]. An unwrapped 359° error injects a catastrophic correction.
- **Head-on yaw degeneracy (source a).** Gate-bearing yaw leverage → 0 as the gate centres on boresight. Gate the (a) update on off-boresight angle; rely on (b) through transit. Do NOT let a near-zero-leverage update collapse the yaw covariance (it would falsely report converged yaw).
- **VP 90° lattice ambiguity (source b).** A floor-grid VP fixes yaw only modulo 90°. Resolve by temporal continuity + the unique lane-line down-course axis. A 90° jump is a course-fatal heading flip — pin a continuity gate.
- **Bloom vs bearing vs span.** Yaw/z here use the **centroid bearing** and **bright lines** (bloom-robust); the bloom-sensitive **corner span** stays confined to the existing along-track range channel (RECON §2). Do not accidentally route yaw/z through the span.
- **Map dependency of (a) and gate-relative z.** Both reference the active gate's *known world position/height* from the dev-built map. The **VP yaw (b) and floor-height (z) channels are the map-free anchors** — they must carry the estimate when the map is absent/wrong (the randomized-competition-track risk, memory NOW block). Ensure (b)+floor can hold yaw/z *alone* (test a no-gate-map scenario).
- **OFF == byte-identical.** Every new flag defaults OFF; `_step_ahrs` and the fix path must be bit-identical to the current `use_ahrs` build when all new flags are OFF. Green-gate (1091) after each step.
- **Active-gate index provenance.** `RACE_STATUS.active_gate_index` (ENCAPSULATED_DATA, 4 Hz) is the only target signal; confirm it is parsed and threaded. (a) is wrong-gate-prone without it.
- **Master clock.** VP/floor/yaw updates ride `ds.sim_time_ns` like everything else; the per-frame VP/floor updates use the vision epoch reconciliation already in the loop (P0-b).

---

## Appendix — key file:line index (this scope)

| Thing | Location |
|---|---|
| ESKF (yaw-blind without mag) | `src/racer/ahrs/eskf.py:130` ctor, `:255` accel update (rank-2), `:325` dormant mag yaw update, `:366` generic update core |
| ESKF stepped in nav loop (where yaw update hooks in) | `src/racer/navigator.py:504` `_step_ahrs` |
| `use_ahrs` seam / config | `navigator.py:305` flag, `:418-427` construct, `:530-531` ODOMETRY re-encode |
| EqVIO joint filter (escalation) | `src/racer/ahrs/eqvio.py:552` `EqVIOJointEKF`, `:317` `update_landmark` (the bearing-yaw measurement model), `:751` `update_landmark_joint` |
| SOT(3) inverse-depth landmark | `src/racer/ahrs/eqvio_landmark.py:60` SOT3, `:130` `bearing_measurement`, `:282` anchored Jacobian |
| PnP / square-flip handling (yaw NOT taken from here) | `src/racer/vision/gate_pose.py:28-41` 2-fold + prior disambig, `:60` `PRIOR_DISAMBIG_MAX_RATIO` |
| +L lever / gate-relative z (existing) | `src/racer/localization.py:91` abs fix, `:161` gate-relative (vertical in-plane), `:230/:275` along-track range (NOT z) |
| Camera vertical-offset / boresight ε_vert | `localization.py:73-88`, `frames.BORESIGHT` |
| KF predict (z double-integration) | `src/racer/state_estimator.py:110` |
| NavState attitude/rate (GAP #2) | `state_estimator.py:215-216`, `make_nav_state:196` |
| Gate-frame axes (vertical = in-plane Y) | `navigator.py:174` `_frame_from_through` |
| AHRS test infra (yaw-drift scenarios) | `src/racer/ahrs/traj6dof.py`, `ahrs/imu_gen.py`, `ahrs/metrics.py` |
| case-C dual-Nav harness (extend for yaw) | `tests/test_casec_ahrs_loop.py` (per `case-c-integration-scope.md` §3) |
| Recon frames (VP/floor smoke fixtures) | `handoff/vq2-recon-2026-06-29/frames/curated/{02,04,05,07}.png` (structure), `{10,11,12}.png` (blur) |
| RACE_STATUS active_gate_index | ENCAPSULATED_DATA data_type=1 (RECON §1); thread into nav loop |
| green gate | `scripts/green_gate.py` (sentinel 1091) |

---

## MEMORY-DELTA

```
MAGFREE-VISION-YAW-SCOPE (2026-06-29):
- VQ2 confirmed accel+gyro ONLY (no mag, no baro). ESKF accel update is rank-2 (yaw-blind:
  H=-skew(g_hat) has a zero yaw column, eskf.py:301); yaw = free integrator on gyro-z bias =>
  drifts unbounded. z double-integrates accel => drifts. BOTH must be pinned by vision.
- YAW VERDICT: yaw IS observable. PRIMARY = gate-bearing pseudo-measurement to the KNOWN
  active-gate world position (flip-SAFE: uses the +L lever/bearing direction, NOT the PnP
  rotation R_cam_gate; existing disambig+Maha gates protect it; weak only head-on/at transit).
  ABSOLUTE backstop = vanishing-point/lane-line heading from the Manhattan warehouse (recon
  frames 02/04/05/07 = floor grid + ceiling truss + blue lane-lines => drift-free yaw, works
  with NO gate in view). Square-flip tension RESOLVED by never taking yaw from PnP rotation.
- Z VERDICT: pinned PRIMARILY by the existing gate-relative vertical fix (no new code; in-plane
  Y axis is world-down); backstopped by a NEW floor-plane height channel for map-free/no-gate
  coverage. NOT the along-track span channel (that is range, not z).
- ARCHITECTURE: recommend Option C = ship B (light yaw/z corrections into the EXISTING ESKF +
  C2 chain), hold EqVIO joint filter (eqvio.py:552, already built+tested) as the gated
  ESCALATION. B composes with the just-wired use_ahrs seam, dodges square-flip, keeps +L/obs/
  confidence invariants byte-identical OFF.
- BUILD: [P] (1) ESKF.update_yaw scalar pseudo-measurement (FD-pin the body-δφ Jacobian);
  (2) vision/heading_vp.py (lane-line first, full VP later); (3) vision/floor_height.py +
  localization.floor_height_fix; then [serial] (4) wire behind use_gate_bearing_yaw/use_vp_yaw/
  use_floor_height flags in Navigator (thread RACE_STATUS.active_gate_index); (5) case-C profile;
  (6) optional EqVIO spike. All default OFF/byte-identical; green-gate 1091 each step.
- KEY FOOTGUNS: yaw Jacobian is body-δφ vs world-yaw (FD-pin); angle-wrap the innovation;
  head-on yaw degeneracy (gate on off-boresight angle); VP 90-deg lattice ambiguity (resolve by
  continuity + lane-line); (b)+floor are the MAP-FREE anchors (randomized-track insurance).
- Staged doc: docs/reactivation-2026-06-27/magfree-vision-yaw-scope.md
```
