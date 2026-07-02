"""Navigator — the per-tick SENSE -> ESTIMATE engine that feeds :class:`Mission`.

Given the latest :class:`DroneState` (from ``mavlink_client``), the latest camera
:class:`Frame` (from ``jpeg_receiver``, or ``None`` when no new frame is ready), and the
ordered gate map, :meth:`Navigator.update` runs the existing perception + estimation chain
behind the frozen contracts and returns a fused :class:`NavState`:

    DroneState --IMU--> KF.predict
    DroneState --given pos/vel--> KF.update   (VQ1: position is GIVEN + pristine)
    Frame -> detector -> estimate_gate_pose (PnP) -> data-assoc to a map Gate ->
             localization -> KF.update_position   (behind a Mahalanobis innovation gate)
    -> make_nav_state -> NavState

``Mission.step`` consumes that NavState and owns THINK -> ACT (planner -> controller ->
ControlCommand); ``Mission.run(navigator, transport)`` is the loop. So the wiring is:
``Mission.run(lambda: nav.update(client.state, latest_frame()), client)`` — the navigator is
the front of the chain, the mission completes it. Keeping the split here (vs. folding the
state machine in) preserves the existing, tested ``Mission`` seam and all of its tests.

Design stance (first contact 2026-06-02 + the walking-skeleton directive):
- Position + velocity are GIVEN and pristine (LOCAL_POSITION_NED 97 Hz + ODOMETRY 75 Hz);
  baro is NaN so z also comes from the given position. We feed them to the KF as TIGHT
  measurements rather than bypassing it, so VQ1 *is* the VQ2 stack under-tuned: flip
  ``use_given_position`` off for VQ2 and the very same filter runs vision-only.
- Vision stays in the loop but is never a crutch: a gate fix is applied as a LOOSER
  measurement and is gated by a Mahalanobis innovation test against the IMU-propagated prior
  (the deferred ``project-estimator-robustness`` gate, which lives exactly here). With the
  given position anchoring the estimate, a rejected/garbage fix cannot corrupt it; with the
  given position off (VQ2) the same gate rejects wrong-gate "teleport" fixes.
- No-gate case: no detector, no frame, no detections, or none associate -> no vision update;
  the KF coasts on IMU + given state and ``time_since_vision_update_s`` grows. Because the map
  + position are known, the planner still flies to the next gate whether or not it is in view,
  so "no gate visible" never stalls VQ1 (it is not the headless-fly risk that an unknown-map
  lap would be).

The detector is INJECTED (anything with ``.detect(frame) -> [GateObservation]``), so:
- the live loop passes a ``GateDetector`` when weights are present, or ``None`` to fly on the
  given state alone (the safe bring-up order: known map on given state first, vision second);
- tests pass a fake detector + the synthetic projector to exercise vision -> KF with no model.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import DroneState, Frame, Gate, GateObservation, GatePose, NavState
from racer.frames import (
    ATTITUDE_NOISE_STD_RAD,
    ODO_QUAT_TRUE_CONJ_WXYZ,
    R_camera_from_body,
    R_world_from_body,
    R_world_from_odo_quat_wxyz,
    euler_from_quat_wxyz,
)
from racer.kf_rewind import RewindKF
from racer.localization import (
    FIX_COV_FLOOR_STD,
    GATE_RANGE_SIGMA_FLOOR,
    GATE_RANGE_SIGMA_REL,
    GATE_REL_ALONG_SIGMA,
    GATE_REL_INPLANE_SIGMA,
    GATE_REL_RANGE_GROWTH_A1,
    INPLANE_POS_FLOOR_STD,
    apparent_range_from_gate_span,
    gate_pose_to_world_position,
    gate_range_fix,
    gate_range_sigma,
    gate_relative_inplane_fix,
)
from racer.state_estimator import LinearKF, make_nav_state

# chi-square 99.9% quantile, 2 DOF -- the IN-PLANE relative-innovation outlier gate (BLUEPRINT §1.3).
# (The absolute 3-DOF Mahalanobis gate is 16.27; the gate-relative fix is 2-DOF in-plane -> 13.82.)
GATE_REL_CHI2_2_999 = 13.815510557964274
# chi-square 99.9% quantile, 1 DOF -- the along-track RANGE-channel innovation gate (B1, perception-l2).
# The span-derived range fix is a single along-track scalar constraint -> 1-DOF outlier test.
GATE_RANGE_CHI2_1_999 = 10.827566170662733
from racer.vision.association import (
    ASSOC_MAX_CENTER_UNITS,
    ASSOC_MAX_SIZE_RATIO,
    RANGE_ABS_TOL_M,
    RANGE_REL_TOL,
    associate,
    predict_gates_in_camera,
    range_consistent,
)
from racer.vision.gate_pose import GATE_INNER_SIZE_M, estimate_gate_pose
# Shared per-frame_id detection cache (A15/A17 double-detect fix): the navigator + the gate-seeker
# hold the SAME detector instance; routing both through detect_cached computes detect() ONCE per
# frame_id. Pure-python (no ultralytics) so the module-level import is safe on every path.
from racer.vision.detector import detect_cached
# Map-free vision yaw/z anchors (magfree-vision-yaw-scope.md). CONSUMED behind use_vp_yaw / use_floor_height;
# imported here (cheap, pure-numpy module-level) so the gated code path is a straight call. The OFF path never
# invokes them, so they cannot perturb the byte-identical default.
from racer.vision.floor_height import estimate_floor_height
from racer.vision.heading_vp import estimate_heading
# A21 vertical-channel estimator (vertical-estimator scope, 2026-07-02). CONSUMED behind
# use_vertical_estimator; imported here (cheap, pure-numpy module-level) so the gated code path is a
# straight call. The OFF path never constructs one, so it cannot perturb the byte-identical default.
from racer.vertical_estimator import VerticalEstimator, a_up_from_specific_force

_WORLD_DOWN = np.array([0.0, 0.0, 1.0])   # NED down


def _wrap_pi(a: float) -> float:
    """Wrap an angle into (-pi, pi] — used by the mag-free yaw-correction branch disambiguation."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


# Deliberately-huge in-plane 1-sigma for the floor-height z fix -> the 3-DOF KF update is effectively
# 1-DOF (world-down z) only. Mirrors localization.GATE_RANGE_INPLANE_STD's intent for the range channel.
_FLOOR_INPLANE_STD = 1000.0   # m


# ---------------------------------------------------------------------------
# Map loading: TRACK_INFO / track_map.json records -> ordered list[Gate]
# ---------------------------------------------------------------------------
def gates_from_track_records(
    records: list[dict], inner_size_m: float = GATE_INNER_SIZE_M, corner_to_center: bool = False
) -> list[Gate]:
    """Convert TRACK_INFO / track_map.json gate records into ordered :class:`Gate` objects.

    Each record carries ``position_ned`` and (outer) ``width_m`` / ``height_m`` = 2.72 m; we
    set ``inner_size_m`` (~1.5 m) for PnP, NOT the outer square (first-contact lock-in).

    The gate's through-direction (``R_world_gate``'s +Z, used by the planner's carrot + the
    mission's plane-crossing advance) is derived from the COURSE GEOMETRY — the unit vector to
    the next gate (the last gate reuses the previous segment). This is chosen over the
    sim-provided quaternion deliberately: the geometric direction is self-evidently correct
    (it always points down-course, so the planner can never place its carrot on the approach
    side and U-turn back through a gate), whereas the sim gate-quaternion -> our gate-frame
    convention is unverified and a sign error there is dangerous. The gate plane axes are then
    X = horizontal-perpendicular (right), Y = completes the frame (down-ish) — matching the
    Gate convention (X=right, Y=down, Z=downrange) for the roughly-upright gates here.

    [first contact: cross-check the sim quaternion against this once vision is live; see the
    handoff note. For VQ1 the geometry is the safe, sufficient source.]
    """
    n = len(records)
    positions = [np.asarray(r["position_ned"], dtype=np.float64) for r in records]
    gates: list[Gate] = []
    for i, r in enumerate(records):
        if n >= 2:
            j = i + 1 if i + 1 < n else i  # last gate reuses the previous segment direction
            seg = positions[j] - positions[i - 1 if j == i else i]
        else:
            seg = np.array([-1.0, 0.0, 0.0])  # lone gate: default to the -X course heading
        R = _frame_from_through(seg)
        pos = positions[i]
        if corner_to_center:
            # Use the gate's TRUE orientation quaternion (verified 2026-06-04) for the NORMAL and
            # the centre. The segment-derived frame faces along the COURSE PATH (gate-to-gate), which
            # is tilted; the real gates all face -X. With the tilted frame the cross-track "gate
            # axis" line, extended back to the start, sits ~1.7 m off to the side, so the controller
            # detours sideways to reach it then oscillates (measured gate0_front2). The quaternion
            # gives a straight -X axis -> the gate sits directly ahead, no sideways detour.
            #   quat convention: col0 = +width, col1 = normal(-X), col2 = +height(down).
            # The map position is the gate's BOTTOM-CENTRE (centred in width, base in height), so the
            # only correction is VERTICAL: lift half the height (no lateral shift). gate0 z -0.03 ->
            # -1.39.
            # The IN-PLANE axes come from the down-course geometric convention
            # (_frame_from_through: X=image-right, Y=image-down for the approaching drone), NOT the
            # raw quaternion columns: the quat axes are authored for the OPPOSITE facing, so using
            # them left right/down BOTH sign-flipped -- a 180-deg in-plane offset against the
            # detector's corner-identity convention. Measured on the course bundles (vision-pkg2
            # 2026-06-10): solved-vs-predicted gate rotation 174 deg p50 -> 27 deg with the flip
            # removed. With the raw columns the PnP disambiguation prior was ANTI-aligned, so the
            # IPPE frontal tie-break / P3P branch pick was effectively random and the 3-corner
            # association compared a detection against the diagonally-opposite predicted corners.
            # through_dir (col Z) and mission._passed (|rel @ col|) are invariant to the flip.
            h = float(r.get("height_m") or 2.72)
            q = r.get("orientation_ned_wxyz")
            if q is not None:
                Rq = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                nrm = Rq[:, 1]                          # gate normal (through-direction)
                if nrm @ seg < 0.0:                     # orient it down-course (exit side)
                    nrm = -nrm
                col2 = Rq[:, 2] if Rq[:, 2][2] >= 0.0 else -Rq[:, 2]  # height axis, pointing down
                R = _frame_from_through(nrm)            # in-plane axes: approach-view convention
                pos = pos - 0.5 * h * col2             # lift to the opening centre (no lateral shift)
            else:                                       # fallback: lift straight up (gates ~upright)
                pos = pos - np.array([0.0, 0.0, 0.5 * h])
        gates.append(
            Gate(
                gate_id=int(r["gate_id"]),
                position_ned=pos,
                R_world_gate=R,
                inner_size_m=float(inner_size_m),
            )
        )
    return gates


def _frame_from_through(through: np.ndarray) -> np.ndarray:
    """Orthonormal gate frame (X=right, Y=down, Z=downrange) from a through-direction vector."""
    z = through / (np.linalg.norm(through) + 1e-12)
    x = np.cross(_WORLD_DOWN, z)             # horizontal, perpendicular to the through-axis
    nx = float(np.linalg.norm(x))
    if nx < 1e-6:                            # through-axis is vertical: pick an arbitrary plane
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / nx
    y = np.cross(z, x)                       # completes a right-handed frame (~down)
    return np.column_stack([x, y, z])


def load_track_map(path: str | Path, inner_size_m: float = GATE_INNER_SIZE_M,
                   corner_to_center: bool = False) -> list[Gate]:
    """Load the deterministic course map JSON (``capture_track_map.py`` output) into Gates."""
    data = json.loads(Path(path).read_text())
    return gates_from_track_records(data["gates"], inner_size_m=inner_size_m,
                                    corner_to_center=corner_to_center)


# ---------------------------------------------------------------------------
# Navigator
# ---------------------------------------------------------------------------
@dataclass
class NavigatorConfig:
    # Given-state fusion (VQ1: pristine ground-truth pos/vel). Tight = trusted.
    use_given_position: bool = True
    use_given_velocity: bool = True
    given_pos_std: float = 0.05            # m, 1-sigma on the given LOCAL_POSITION_NED / ODOMETRY pos
    given_vel_std: float = 0.10            # m/s

    # Vision -> KF (kept in-loop; never the crutch). Off when no detector is supplied.
    use_vision: bool = True
    # Ignore PnP fixes beyond this range. 40 -> 32 [vision-pkg2 2026-06-10]: the long-range
    # depth-noise tail (sigma ~3% of range, occasional -12% events) produced the residual
    # catastrophic leak (a 4.9 m fix error at 38 m passed every gate); the course bundles show
    # ZERO good sub-metre fixes beyond 30 m, and no point on the course is further than ~30 m
    # from the next gate, so the cap costs nothing and removes the worst leak at the source.
    vision_max_range_m: float = 32.0
    # Robust association (racer.vision.association): a detection must agree with a map
    # gate's PREDICTED shape -- apparent-size ratio hard-gated, centre offset normalised by
    # the predicted size. Replaces the naive fixed-150px nearest-centre gate that caused
    # the measured 46% wrong-gate/junk fix tail on the collinear course. [2026-06-09]
    assoc_max_size_ratio: float = ASSOC_MAX_SIZE_RATIO
    assoc_max_center_units: float = ASSOC_MAX_CENTER_UNITS
    # Post-PnP depth sanity vs the predicted range to the associated gate (known 1.5 m gate
    # size makes PnP depth metric): reject the fix when they disagree beyond a relative
    # tolerance with an absolute floor (the floor keeps a ~1 m VQ2 prior error harmless).
    fix_range_rel_tol: float = RANGE_REL_TOL
    fix_range_abs_tol_m: float = RANGE_ABS_TOL_M
    # Mahalanobis innovation gate (3-DOF position). 16.27 = chi-square 99.9% quantile: reject
    # only egregious disagreement with the IMU-propagated prior (wrong-gate / garbage PnP), so a
    # healthy fix is never dropped. The right form of "gate the fix on agreement-with-prediction"
    # (nu^T S^-1 nu, breathing with S), per project-estimator-robustness.
    vision_gate_chi2: float = 16.27
    # Fix-covariance model constants, MEASURED on the canonical course recording [vision-pkg2
    # 2026-06-10]: the given-attitude/chain 1-sigma for the lever-arm term (was a 1.0-deg guess)
    # and the isotropic floor covering the range-independent systematics (map-centre vertical,
    # per-gate lateral, close-range depth). One source of truth each; see frames/localization.
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD
    fix_cov_floor_std: float = FIX_COV_FLOOR_STD

    # P0-b TIMESYNC + predict-forward. ``frame.sim_time_ns`` (camera/server epoch) and
    # ``DroneState.sim_time_ns`` (HIGHRES_IMU master epoch) are DISTINCT, unreconciled clocks
    # (contracts.py clock note). ``time_since_vision_update_s`` must be measured on ONE clock —
    # the IMU master clock — or it mixes epochs and reads garbage live. Two ways to put a fix's
    # timestamp on the IMU clock:
    #   reconcile_vision_clock=True : learn ``delta_epoch = frame.sim − imu.sim`` ONCE (paired via
    #     recv_monotonic_ns), then stamp a fix at its CAPTURE time ``obs.sim − delta_epoch``.
    #   reconcile_vision_clock=False: PREDICT-FORWARD — stamp every fix at a constant calibrated
    #     age ``now − vision_latency_const_s`` on the IMU clock; needs NO capture stamp and works
    #     BEFORE a live TIMESYNC trace exists (the path the BLUEPRINT §1.5 ships first).
    # delta_epoch is 0 on same-clock data (synthetic/VQ1 tests) so back-compat is exact. The full
    # capture-time OOSM RewindKF that consumes the precise stamp is a LATER step (deferred, §1.5).
    reconcile_vision_clock: bool = True
    vision_latency_const_s: float = 0.0    # predict-forward constant fix age (s); calibrated L3/L4

    # --- C2 estimator chain (case-C VQ2 gate-relative pipeline, BLUEPRINT §1.2-1.6) ---
    # OFF by default -> the VQ1 / case-A path is byte-identical (bare LinearKF, in-place fixes). Flip
    # both ON for the case-C gate-relative pipeline.
    # RewindKF OOSM wrap (§1.4/§1.5): vision fixes are applied at their CAPTURE time and the buffered
    # IMU re-propagated, so a fix corrects where the drone WAS, not where it is. ``horizon_s`` MUST be
    # strictly > the max fix age L (horizon<=L drops 100% of fixes -> divergence); 0.5 s covers the
    # CPU-class L~125 ms. Asserted at init against ``vision_latency_const_s`` (the known predict-forward L).
    use_rewind_kf: bool = False
    rewind_horizon_s: float = 0.5
    # Gate-relative in-plane +L fix (§1.2/§1.3): a SECOND, in-plane-only correction layered on top of the
    # absolute fix (which is KEPT for planning / along-track / g4->g5 handoff). It pins the terminal
    # in-plane centering miss to the SEEN opening with the tight PnP lateral sigma (NO bias floor), so the
    # per-track map bias cancels in the policy obs. Gated by its OWN relative-innovation outlier test
    # (depth-flips that reproj + the absolute Maha gate pass).
    use_gate_relative: bool = False
    gate_rel_inplane_sigma: float = GATE_REL_INPLANE_SIGMA    # single swappable calibration constant
    gate_rel_range_growth_a1: float = GATE_REL_RANGE_GROWTH_A1
    gate_rel_along_sigma: float = GATE_REL_ALONG_SIGMA
    gate_rel_chi2: float = GATE_REL_CHI2_2_999               # chi2(2, 0.999) in-plane gate
    # In-plane STATE-covariance floor on the KF [parked #74, coast-drift 2026-06-15]. DISTINCT from the
    # MEASUREMENT floor (fix_cov_floor_std): a dense gate-relative fix stream drives P -> R/N -> 0 so the
    # Kalman gain -> 0 and the filter rides the drifting IMU ("centering-blind") exactly when a
    # well-pointed inc8 policy makes fixes densest. The state floor keeps P (and thus the gain) responsive.
    # ACTIVE only in the case-C estimator chain (use_rewind_kf OR use_gate_relative); the VQ1 / case-A
    # path keeps floor=0 -> bare-filter byte-identical. Strictly-more-honest -- it only RAISES an
    # over-converged in-plane covariance toward the true systematic floor sigma_b (never lowers P).
    use_inplane_pos_floor: bool = True
    inplane_pos_floor_std: float = INPLANE_POS_FLOOR_STD     # m, sigma_b systematic centering floor (= sigma_ref)

    # B1 bearing-range channel (perception-l2-scope.md 2026-06-28). An ATTITUDE-INDEPENDENT along-track
    # range fix from the gate's apparent pixel span (known 1.5 m square), layered AFTER the in-plane fix
    # with its own 1-DOF chi2 gate. It replaces the loose GATE_REL_ALONG_SIGMA=0.5 m prior on the
    # along-track axis with a data-driven, proximity-tightening range that is robust to the ε_vert
    # boresight bias (the span sets the subtended ANGLE, not the centroid). OFF by default -> the whole
    # navigator loop (including the C2 gate-relative path) is byte-identical to main; no new RNG, no obs
    # change. Requires use_gate_relative=True to participate (the range fix augments the in-plane fix).
    use_range_channel: bool = False
    gate_range_sigma_rel: float = GATE_RANGE_SIGMA_REL       # per-metre along-track range 1-sigma growth
    gate_range_sigma_floor: float = GATE_RANGE_SIGMA_FLOOR   # m, near-field range 1-sigma floor
    gate_range_chi2: float = GATE_RANGE_CHI2_1_999           # chi2(1, 0.999) along-track range gate

    # --- L1 AHRS attitude source (case-C VQ2: SELF-ESTIMATE attitude from raw HIGHRES_IMU) ---
    # VQ2 §9.3 BLOCKS ATTITUDE / ODOMETRY, so the deployed stack has NO given attitude. When ON, the
    # Navigator owns an ESKF-based AHRS (racer.ahrs.ahrs_adapter.AHRSAttitudeSource), steps it each IMU
    # tick from (accel_body, gyro_body, dt), and sources the body->world rotation R_wb (KF predict + PnP
    # lever, GAP #1) AND the NavState attitude/rate (obs[6:12] feeder, GAP #2) from the AHRS instead of
    # the ODOMETRY quat. OFF by default -> byte-identical to the VQ1 / case-A ODOMETRY-attitude path
    # (no AHRS constructed, no extra RNG, no array drawn). The AHRS emits the TRUE FRD->NED attitude
    # DIRECTLY: the Navigator re-encodes it into the ODOMETRY-wire convention (involutory R_y(pi) quat
    # conjugation + all-axis rate negation) before handing it to make_nav_state / the obs seam, so EVERY
    # downstream consumer (build_obs's _ODO_QUAT_TRUE_CONJ/_ODO_RATE_SIGN, the controller's odo_*_sign,
    # euler_from_quat_wxyz) sees the convention it already expects -- NO double-conjugation, NO
    # recalibration. Requires ds.gyro_body to be populated (raw HIGHRES_IMU gyro).
    use_ahrs: bool = False
    ahrs_gyro_noise_std: float = 0.01      # rad/s, ESKF gyro white-noise 1-sigma (bench-validated default)
    ahrs_accel_gate_alpha: float = 10.0    # ESKF high-g accel-gating sharpness (bench-validated default)
    ahrs_accel_motion_reject: bool = False # ESKF accel rejection under SUSTAINED linear accel (the A8 fix:
                                           # |a|~=g but direction tilted -> magnitude gate misses it). OFF =
                                           # byte-identical; vq2_case_c turns it ON. → eskf.use_accel_motion_reject

    # --- Mag-free vision YAW + Z corrections into the ESKF/KF (magfree-vision-yaw-scope.md, Option B) ---
    # VQ2 has NO magnetometer + NO barometer, so the ESKF accel update is yaw-blind (eskf.py:301) and
    # double-integrated accel z drifts: BOTH yaw and z MUST be pinned by vision. These three flags wire
    # the just-built MAP-FREE vision anchors (vision/heading_vp.py vanishing-point heading,
    # vision/floor_height.py floor-grid height) + a gate-bearing yaw lock into the existing ESKF + C2
    # chain (NOT a full EqVIO swap). ALL default OFF -> the navigator (KF state x/P + obs) is byte-identical
    # to the use_ahrs build; each requires use_ahrs=True to do anything (they correct the ESKF, which only
    # exists under use_ahrs). No new RNG on any path.
    #
    # (b) ABSOLUTE yaw backstop: vanishing-point / lane-line heading from the Manhattan warehouse. Runs
    # PER FRAME even with NO gate associated (the no-gate-in-view yaw anchor). The estimator reports an
    # ABSOLUTE warehouse yaw mod 90 deg (the lattice ambiguity); the navigator disambiguates the 4 branches
    # to the gyro-propagated yaw estimate (nearest branch) before injecting -> never a 90-deg flip.
    use_vp_yaw: bool = False
    vp_yaw_min_quality: float = 0.30      # gate the VP heading update on HeadingEstimate.quality
    vp_yaw_noise_std: float = float(np.deg2rad(5.0))   # 1-sigma (rad) of the VP yaw pseudo-measurement
    vp_yaw_branch_max_rad: float = float(np.deg2rad(35.0))  # reject if the nearest branch is > this from
                                                            # the current yaw estimate (ambiguous -> skip)
    vp_yaw_ransac_iters: int = 256   # RANSAC iters for the VP fit. 256 is the 30Hz-budget point: on the
                                     # real VQ2 recon frames the heading is within 0.081deg of the iters=2000
                                     # fit (most frames exactly 0.000) while estimate_heading drops ~3.5x
                                     # (230-325ms -> 61-91ms). 2000 was arbitrary overkill (a 2-pt minimal
                                     # sample is statistically saturated by ~256 even at low inlier ratio).
                                     # This is THE 2.3Hz->30Hz fix: at 2000 the loop choked and slammed the
                                     # floor -> free-fall -> AHRS inversion -> backflip.
    # (a) PRIMARY yaw lock: gate-bearing yaw to the KNOWN active-gate world position. Flip-SAFE -- it uses
    # the well-conditioned +L lever/bearing DIRECTION (R_w2b @ (gate - p_KF)), NOT the noisy planar-PnP
    # rotation R_cam_gate. Yaw leverage collapses head-on (the gate centres on boresight), so it is gated
    # on the off-boresight bearing angle (skip when too near head-on) and weighted by it.
    use_gate_bearing_yaw: bool = False
    gate_bearing_yaw_noise_std: float = float(np.deg2rad(4.0))   # base 1-sigma (rad) off-axis
    gate_bearing_min_offaxis_rad: float = float(np.deg2rad(8.0))  # skip when the gate bearing is within
                                                                  # this of boresight (no yaw leverage)
    # Z: map-free floor-plane height channel (vision/floor_height.py) -> a z (world-down) KF correction,
    # layered ON TOP of the existing gate-relative vertical fix. Gated HARD on quality + std_m (the floor
    # is ill-conditioned near the horizon / nose-up). Tight in z, ~infinite in-plane -> effectively 1-DOF.
    use_floor_height: bool = False
    floor_height_min_quality: float = 0.30     # gate the floor-height z update on FloorHeightEstimate.quality
    floor_height_max_std_m: float = 0.50       # reject when the floor-height std_m exceeds this (near-horizon)
    floor_height_extra_std_m: float = 0.20     # extra z 1-sigma added in quadrature (model/mount systematics)
    floor_grid_cell_m: float = 2.0             # known warehouse floor-grid cell size (metric anchor; calibrate)
    floor_camera_height_ref_m: float = 0.0     # world-down z of the FLOOR plane (NED). camera z = floor_z -
                                               # height_above_floor (camera sits ABOVE the floor => smaller z).

    # --- A17 per-frame CV-backstop DECIMATION (frame-starvation fix, 2026-07-01) ---
    # The two heaviest per-frame CV backstops run every processed vision tick: vp_yaw (~67 ms, the VP
    # RANSAC + Manhattan line extraction in estimate_heading) and floor_height (~37 ms). Under the live
    # ShadowPC load (sim + fly_rl co-located) that per-frame cost choked the loop to ~10 Hz and starved
    # the video receiver. These knobs run each backstop only every N-th PROCESSED vision tick; on the
    # SKIPPED ticks the backstop is not called AT ALL (no estimate_heading / estimate_floor_height), and
    # the ESKF gyro-propagates yaw + predicts z between corrections. DEFAULT = 1 = every tick = BYTE-
    # IDENTICAL to the pre-decimation path (the modulo is always 0 at N=1). vq2_case_c sets vp_yaw=5
    # (yaw drifts slowly + the gate-bearing-yaw lock pins yaw per accepted detection + the gyro
    # integrates between) and floor_height=3 (the ONLY dedicated z pin -> keep it tighter). N<=0 is
    # treated as 1 (defensive). The update ORDER is unchanged (vp_yaw -> floor_height -> detect).
    vp_yaw_decimate: int = 1
    floor_height_decimate: int = 1

    # --- A24 vertical-velocity washout (supersedes the A21 floor-pin KF, 2026-07-02) ---
    # Own a 1-D vertical-velocity washout (racer.vertical_estimator) beside the 6-state KF: IMU
    # a_up integrated per tick with an exponential leak (bounded by construction -- no floor-pin
    # correction; floor-height is a FALSE PREMISE as a vertical FIX on this wire, see the module
    # docstring). WHY a second surface: the 6-state KF's z is dead-reckoning between sparse vision
    # fixes and STEP-TELEPORTS when one lands (run 20260702_040036: -1.435 m in one tick), so the
    # controller's vertical damper reads teleports instead of the real climb. The dedicated channel
    # exports a SMOOTH, structurally-bounded vz on NavState.vert_vz_est for the ff-owns-vertical
    # alt-hold to damp on (vert_z_est stays permanently NaN -- no absolute-altitude state). OFF by
    # default -> no estimator constructed, NavState fields stay NaN, byte-identical. vq2_case_c
    # opts in via DeployProfile.vertical_estimator (the fly_rl construction seam).
    use_vertical_estimator: bool = False


@dataclass
class _VisionDiag:
    """Last-tick vision diagnostics (off the NavState contract; for logging / the handoff)."""

    n_detections: int = 0
    n_associated: int = 0
    n_applied: int = 0
    n_rejected_gate: int = 0
    n_rejected_range: int = 0       # post-PnP depth-sanity rejections (range_consistent)
    n_rel_applied: int = 0          # gate-relative in-plane fixes applied (C2)
    n_rel_rejected: int = 0         # gate-relative fixes rejected by the relative-innovation gate (C2)
    n_range_applied: int = 0        # B1 along-track range-channel fixes applied
    n_range_rejected: int = 0       # B1 range fixes rejected by the 1-DOF range-innovation gate
    last_gate_id: int | None = None
    last_range_m: float = float("nan")
    last_reproj_px: float = float("nan")
    last_mahalanobis: float = float("nan")
    last_d2_rel: float = float("nan")   # last gate-relative in-plane innovation statistic (C2)
    last_range_span_m: float = float("nan")   # last span-derived (B1) range to the gate
    last_d2_range: float = float("nan")       # last along-track range-channel innovation statistic (B1)
    n_vp_yaw_applied: int = 0                  # vanishing-point yaw pseudo-measurements applied (mag-free)
    n_vp_yaw_rejected: int = 0                 # VP yaw skipped (low quality / ambiguous branch / no VP)
    n_floor_z_applied: int = 0                 # floor-height z corrections applied (mag-free, map-free)
    n_floor_z_rejected: int = 0                # floor-height skipped (low quality / large std / no floor)
    n_gate_bearing_yaw_applied: int = 0        # gate-bearing yaw locks applied (primary, flip-safe)
    n_gate_bearing_yaw_rejected: int = 0       # gate-bearing yaw skipped (head-on / no active gate)
    last_vp_yaw_rad: float = float("nan")      # last VP absolute heading injected (post branch-disambig)
    last_floor_height_m: float = float("nan")  # last floor-derived camera height above the floor
    last_gate_bearing_yaw_rad: float = float("nan")  # last gate-bearing-implied world yaw injected


@dataclass
class Navigator:
    """Per-tick SENSE -> ESTIMATE. Build with the ordered gate map + an optional detector.

    ``update(drone_state, frame)`` returns the current :class:`NavState`. Estimation advances
    once per new IMU sample (``DroneState.sim_time_ns``); a faster control loop that calls in
    between simply gets the cached state re-packaged. Vision runs once per new ``frame_id``.
    """

    gates: list[Gate]
    detector: object | None = None                 # .detect(Frame) -> [GateObservation]; None => no vision
    config: NavigatorConfig = field(default_factory=NavigatorConfig)
    kf: LinearKF | None = None

    initialized: bool = field(default=False, repr=False)
    n_vision_fixes: int = field(default=0, repr=False)
    n_vision_rejected: int = field(default=0, repr=False)
    # Cumulative mag-free vision yaw/z corrections applied over the run (the per-tick counts live on
    # vision_diag, which resets each frame). Diagnostics only; do not affect the estimate.
    n_vp_yaw_total: int = field(default=0, repr=False)
    n_floor_z_total: int = field(default=0, repr=False)
    n_gate_bearing_yaw_total: int = field(default=0, repr=False)
    vision_diag: _VisionDiag = field(default_factory=_VisionDiag, repr=False)
    # -- per-STEP vision timing (logging only, no behaviour change) -- pins WHICH _maybe_run_vision
    # sub-step (detect / vp_yaw [VP+Manhattan] / floor_height / pnp) is choking the loop. Mirrors
    # gate_seeker.diag_counts: an instance dict accumulated in memory, no per-tick I/O; fly_rl prints
    # a one-line [vision-timing] summary at loop exit. time.perf_counter() overhead is ~tens of ns,
    # negligible next to the ms-scale steps being measured. Wrapped defensively (see _timed_step) so
    # a timing bug can never raise into / alter the flight loop.
    vision_step_ms: dict = field(default_factory=lambda: {
        "detect": {"count": 0, "total_ms": 0.0, "max_ms": 0.0},
        "vp_yaw": {"count": 0, "total_ms": 0.0, "max_ms": 0.0},
        "floor_height": {"count": 0, "total_ms": 0.0, "max_ms": 0.0},
        "pnp": {"count": 0, "total_ms": 0.0, "max_ms": 0.0},
    }, repr=False)
    # Breakdown (step -> ms) of the single worst tick seen so far, keyed by the SUM across that
    # tick's steps -- lets [vision-timing] show which step(s) dominated the worst-work tick.
    _vision_worst_tick_ms: dict = field(default_factory=dict, repr=False)
    _vision_worst_tick_total_ms: float = field(default=0.0, repr=False)
    # Per-tick step ms accumulated by _timed_step, flushed into the worst-tick breakdown (and reset)
    # at the top of each _maybe_run_vision call.
    _vision_tick_ms: dict = field(default_factory=dict, repr=False)

    _last_sim_time_ns: int = field(default=0, repr=False)
    _reset_counter: int = field(default=0, repr=False)
    _last_frame_id: int | None = field(default=None, repr=False)
    # A17 CV-backstop decimation counter: increments once per PROCESSED vision tick (new frame_id).
    # vp_yaw runs when (_vision_tick_count % vp_yaw_decimate)==0; floor_height likewise. At decimate=1
    # (the default) the modulo is always 0 -> every tick -> byte-identical to pre-decimation behaviour.
    _vision_tick_count: int = field(default=0, repr=False)
    _last_vision_sim_time_ns: int | None = field(default=None, repr=False)
    # P0-b: learned camera/server -> IMU epoch offset (frame.sim - imu.sim, recv-paired). None
    # until the first processed frame; re-learned on reset(). 0 on same-clock data.
    _delta_epoch_ns: int | None = field(default=None, repr=False)
    # C2: True when self.kf is a RewindKF (case-C OOSM path); set at _initialize from the config.
    _rewind: bool = field(default=False, repr=False)
    # C2: R_world_gate (3,3) of the gate the last ACCEPTED gate-relative fix landed on -- the frame the
    # NavState confidence export projects P into (§1.6). None until the first gate-relative fix.
    _last_fix_gate_R: np.ndarray | None = field(default=None, repr=False)
    _gates_by_id: dict[int, Gate] = field(default_factory=dict, repr=False)
    # L1 AHRS (case-C use_ahrs): owned attitude source, constructed lazily at _initialize ONLY when
    # config.use_ahrs. None on the OFF path (the AHRS module is not even imported then). Holds the
    # last AHRS-derived ODOMETRY-convention quat/rate so make_nav_state + the obs seam share one
    # estimate (computed once per IMU tick in update()).
    _ahrs: object | None = field(default=None, repr=False)
    _ahrs_odo_quat: np.ndarray | None = field(default=None, repr=False)   # TRUE attitude re-encoded
    _ahrs_odo_rate: np.ndarray | None = field(default=None, repr=False)   # to the ODOMETRY-wire convention
    # Mag-free gate-bearing yaw lock: the latest RACE_STATUS active gate index (from ds.active_gate_index),
    # refreshed each update(). None until a RACE_STATUS arrives -> the gate-bearing yaw lock no-ops.
    _active_gate_index: int | None = field(default=None, repr=False)
    # A21 vertical-channel estimator (use_vertical_estimator): constructed + seeded at _initialize
    # ONLY when the flag is ON; None on the OFF path (never stepped, never exported -> byte-identical).
    _vert_est: VerticalEstimator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._gates_by_id = {g.gate_id: g for g in self.gates}

    # -- lifecycle ----------------------------------------------------------
    def _initialize(self, ds: DroneState) -> None:
        """Seed the KF from the first usable state.

        P0-a (case-C foundation): the position/velocity SEED is gated on the given-state
        CONFIG flags, NOT on the mere presence of ``position_ned`` / ``velocity_ned`` on the
        wire — exactly mirroring the per-tick guards (navigator.py:315/320). In TRUE case C
        (``use_given_position`` False) the seed is the ORIGIN at ``pos_std=5.0`` (P[0,0]=25)
        EVEN when LOCAL_POSITION_NED is broadcasting, so a "case C" run is genuinely vision-only
        and not secretly anchored to the ground-truth pose. Without this gate every case-C test
        is secretly case A (a hidden GT seed) — this was THE leak that invalidated case-C
        validation. [P0-a, BLUEPRINT §1.5]
        """
        use_pos = self.config.use_given_position and ds.position_ned is not None
        use_vel = self.config.use_given_velocity and ds.velocity_ned is not None
        pos = np.asarray(ds.position_ned, dtype=np.float64) if use_pos else np.zeros(3)
        vel = np.asarray(ds.velocity_ned, dtype=np.float64) if use_vel else None
        pos_std = self.config.given_pos_std if use_pos else 5.0
        # In-plane STATE-cov floor (parked #74): ACTIVE only in the case-C estimator chain (rewind/gate-
        # relative). VQ1 / case-A (both OFF) keep floor_std=0.0 -> LinearKF default -> bare-filter
        # byte-identical (the C2 gated-off invariant). See NavigatorConfig.use_inplane_pos_floor.
        floor_std = (
            self.config.inplane_pos_floor_std
            if self.config.use_inplane_pos_floor
            and (self.config.use_rewind_kf or self.config.use_gate_relative)
            else 0.0
        )
        kf = LinearKF.initialize(pos, vel, pos_std=pos_std, vel_std=1.0,
                                 inplane_pos_floor_std=floor_std)
        # C2 (case C): wrap in the RewindKF OOSM buffer so vision fixes apply at capture time. The
        # horizon MUST be strictly > the predict-forward latency L (else 100% of fixes drop -> diverge);
        # assert it loudly at init. The VQ1 / case-A path keeps the bare LinearKF (byte-identical).
        self._rewind = bool(self.config.use_rewind_kf)
        if self._rewind:
            rk = RewindKF(kf=kf, horizon_s=self.config.rewind_horizon_s)
            rk.assert_horizon_gt(self.config.vision_latency_const_s, where="Navigator._initialize")
            self.kf = rk
        else:
            self.kf = kf
        # L1 AHRS (case-C use_ahrs, GAP #1/#3/#6): construct + seed the attitude source alongside the
        # KF. Level-seed from the first accel sample (roll/pitch from gravity; yaw at the identity datum
        # -- gate-relative obs partly absorb a constant yaw offset, scope §2 GAP #6). OFF path never
        # imports the AHRS module. Re-seeded on a sim epoch reset (reset() drops it; this re-builds).
        if self.config.use_ahrs:
            from racer.ahrs.ahrs_adapter import AHRSAttitudeSource
            from racer.ahrs.eskf import ESKFAHRS
            self._ahrs = AHRSAttitudeSource(
                eskf=ESKFAHRS(gyro_noise_std=self.config.ahrs_gyro_noise_std,
                              accel_gate_alpha=self.config.ahrs_accel_gate_alpha,
                              use_accel_motion_reject=self.config.ahrs_accel_motion_reject)
            )
            self._ahrs.seed(AHRSAttitudeSource.level_seed_from_accel(ds.accel_body))
            self._ahrs_odo_quat = None
            self._ahrs_odo_rate = None
        # A24 vertical-velocity washout: seed at rest (vz=0, b_hat=0) and arm the pre-arm bias-
        # capture window (grounded -- true vz is 0). No z state (the washout carries no absolute
        # altitude). OFF path: stays None (byte-identical).
        if self.config.use_vertical_estimator:
            self._vert_est = VerticalEstimator()
            self._vert_est.seed()
        self._last_sim_time_ns = int(ds.sim_time_ns)
        self._reset_counter = int(ds.reset_counter)
        self.initialized = True

    def reset(self) -> None:
        """Drop the estimate (e.g. on a sim epoch restart); the next update re-seeds it."""
        self.kf = None
        self.initialized = False
        self._last_frame_id = None
        self._vision_tick_count = 0        # A17: restart the CV-backstop decimation phase on a sim reset
        self._last_vision_sim_time_ns = None
        self._delta_epoch_ns = None        # P0-b: re-learn the epoch offset after a sim restart
        self._last_fix_gate_R = None       # C2: drop the confidence-export gate frame on restart
        self._ahrs = None                  # L1: re-seed the AHRS on the next _initialize (sim restart)
        self._ahrs_odo_quat = None
        self._ahrs_odo_rate = None
        self._vert_est = None              # A24: re-seed the vertical channel (re-arms bias capture)

    # -- per-tick -----------------------------------------------------------
    def update(self, ds: DroneState, frame: Frame | None = None) -> NavState:
        """Advance the estimate with one telemetry snapshot (+ optional camera frame)."""
        # Refresh the active-gate target (RACE_STATUS) for the mag-free gate-bearing yaw lock. Pure
        # bookkeeping: read-only on the OFF path (the lock no-ops), so the default path is unaffected.
        self._active_gate_index = ds.active_gate_index
        if not self.initialized:
            self._initialize(ds)
            return self._nav_state(ds)

        # A sim epoch restart (ODOMETRY.reset_counter ticks) is a discontinuity, not motion:
        # re-seed rather than integrate across it.
        if int(ds.reset_counter) != self._reset_counter:
            self.reset()
            self._initialize(ds)
            return self._nav_state(ds)

        assert self.kf is not None

        # Estimation advances only on a NEW IMU sample (sim_time_ns is the master clock). When the
        # control loop ticks faster than the IMU, dt<=0 and we just re-package the current state
        # (no predict, no re-applying a stale measurement -> no covariance collapse).
        dt = (int(ds.sim_time_ns) - self._last_sim_time_ns) / 1e9

        # TRUE physical body->world rotation. Default (use_ahrs OFF): from the raw ODOMETRY quat
        # (R_y(pi)-conjugated). The CTBR path uses euler_from_quat_wxyz on the raw quat (aliased,
        # VQ1-proven -- that path is untouched). Vision/PnP/KF must use the true attitude.
        # [vision-frame-fix]
        # case-C (use_ahrs ON, GAP #1): R_wb comes from the AHRS (ODOMETRY blocked in VQ2). Step
        # the AHRS on a fresh IMU sample (dt>0) from raw HIGHRES_IMU (gyro_body / accel_body); on a
        # between-IMU control tick (dt<=0) reuse the current estimate (no step -- mirrors the KF).
        if self.config.use_ahrs:
            R_wb = self._step_ahrs(ds, dt)
        else:
            R_wb = R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)

        if dt <= 0:
            self._maybe_run_vision(ds, frame, R_wb)   # a fresh frame can still land between IMU ticks
            return self._nav_state(ds)
        self._last_sim_time_ns = int(ds.sim_time_ns)

        # 1) IMU predict (gravity-corrected specific force, given attitude). 2) vision fix
        # (looser, innovation-gated, against the propagated prior). 3) given pos/vel (tight,
        # authoritative for VQ1). Vision before given so the innovation gate compares vision to
        # the IMU prior (a meaningful disagreement check), while given still anchors the estimate.
        if self._rewind:
            self.kf.predict(ds.accel_body, R_wb, dt, int(ds.sim_time_ns))   # RewindKF: IMU-clock stamp
        else:
            self.kf.predict(ds.accel_body, R_wb, dt)
        # A24 vertical-velocity washout: integrate the VERIFIED a_up decode (specific force + the
        # TRUE R_wb) at IMU rate, before vision (mirrors the KF predict-then-correct order).
        if self._vert_est is not None:
            self._vert_est.predict(a_up_from_specific_force(ds.accel_body, R_wb), dt)
        self._maybe_run_vision(ds, frame, R_wb)
        if self.config.use_given_position and ds.position_ned is not None:
            self.kf.update_position(
                np.asarray(ds.position_ned, dtype=np.float64),
                (self.config.given_pos_std**2) * np.eye(3),
            )
        if self.config.use_given_velocity and ds.velocity_ned is not None:
            self.kf.update_velocity(
                np.asarray(ds.velocity_ned, dtype=np.float64),
                (self.config.given_vel_std**2) * np.eye(3),
            )
        return self._nav_state(ds)

    # -- L1 AHRS (case-C self-estimated attitude) ---------------------------
    def _step_ahrs(self, ds: DroneState, dt: float) -> np.ndarray:
        """Advance the AHRS one IMU tick (when dt>0) and return the TRUE FRD->NED rotation R_wb.

        Sources attitude from raw HIGHRES_IMU (``ds.gyro_body`` / ``ds.accel_body``) -- NOT the
        ODOMETRY-derived ``angular_rate_body`` (wrong sign + provenance; blocked in VQ2). Snapshots
        the TRUE attitude (quat) + TRUE FRD body rate, RE-ENCODES them into the ODOMETRY-wire
        convention, and caches them for ``_nav_state`` (GAP #2):
          * quat  : ``q_odo = q_true * ODO_QUAT_TRUE_CONJ_WXYZ`` (involutory R_y(pi) conjugation).
                    Feeding this as ``orientation_ned_wxyz`` to build_obs (which RE-applies the same
                    conjugation) recovers the TRUE attitude -> obs[6:9] correct, NO double-conjugation.
                    euler_from_quat_wxyz(q_odo) is the aliased roll/pitch/yaw the controller expects.
          * rate  : ``-(gyro_body - gyro_bias)`` = ``-true_FRD`` = the ODOMETRY-sign convention
                    (Fix option A). build_obs RE-applies ``_ODO_RATE_SIGN=[-1,-1,-1]`` -> TRUE FRD ->
                    obs[9:12] correct; the controller's ``odo_rate_sign`` likewise stays valid.
        ``dt<=0`` (between-IMU control tick): no step, reuse the cached estimate.
        """
        assert self._ahrs is not None
        gyro = ds.gyro_body
        if gyro is None:
            # No raw gyro yet (pre-first HIGHRES_IMU): hold the seed attitude, zero rate.
            gyro = np.zeros(3)
        self._ahrs.ingest(ds.accel_body, gyro, float(dt), mag_body=ds.mag_body)
        q_true = np.asarray(self._ahrs.q_wxyz, dtype=np.float64)
        # TRUE FRD body rate the AHRS just integrated (bias-corrected). On a dt<=0 tick body_rate
        # holds the last dt>0 value, so the cached ODOMETRY-convention rate stays consistent.
        rate_true_frd = np.asarray(self._ahrs.body_rate, dtype=np.float64)
        self._ahrs_odo_quat = q_true * ODO_QUAT_TRUE_CONJ_WXYZ
        self._ahrs_odo_rate = -rate_true_frd
        return self._ahrs.R_wb

    def _refresh_ahrs_attitude_cache(self) -> np.ndarray:
        """Recompute the cached ODOMETRY-convention attitude quat + the TRUE R_wb from the ESKF's
        CURRENT nominal quaternion, AFTER an in-tick vision yaw correction (``ESKFAHRS.update_yaw``).

        A ``use_vp_yaw`` / ``use_gate_bearing_yaw`` update mutates the ESKF attitude IN PLACE; the
        ``_ahrs_odo_quat`` cache (read by ``_nav_state`` -> obs[6:9]) and the local ``R_wb`` (used by
        the rest of the vision loop + the next predict) were snapshotted in ``_step_ahrs`` BEFORE the
        correction, so they must be refreshed or the corrected yaw never reaches the obs/KF this tick.
        The body RATE is unchanged by a yaw update (the bias correction is sub-tick and folded next
        step) so ``_ahrs_odo_rate`` is left as-is. Returns the refreshed TRUE FRD->NED R_wb."""
        assert self._ahrs is not None
        q_true = np.asarray(self._ahrs.q_wxyz, dtype=np.float64)
        self._ahrs_odo_quat = q_true * ODO_QUAT_TRUE_CONJ_WXYZ
        return self._ahrs.R_wb

    # -- mag-free vision yaw / z corrections (case-C; magfree-vision-yaw-scope §4) -------------
    def _current_true_rpy(self) -> tuple[float, float, float]:
        """TRUE (roll, pitch, yaw) of the CURRENT ESKF/AHRS attitude estimate, radians.

        The roll/pitch feed the vision modules (gravity-known tilt, needed to back-project a VP /
        floor pixel) and the yaw is the gyro-propagated current estimate the VP branch-disambiguation
        snaps to. Read straight off the AHRS' TRUE attitude (NOT the ODOMETRY-convention cache)."""
        assert self._ahrs is not None
        return euler_from_quat_wxyz(np.asarray(self._ahrs.q_wxyz, dtype=np.float64))

    def _time_step(self, name: str, t0: float) -> None:
        """Record one _maybe_run_vision sub-step's elapsed ms into ``vision_step_ms[name]`` + the
        current tick's running total (flushed into the worst-tick breakdown by ``_maybe_run_vision``).

        Logging only -- never touches the estimate/command. Defensively wrapped: a KeyError/whatever
        here must never propagate into the flight loop (better to silently drop a timing sample than
        crash a flight over instrumentation)."""
        try:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            bucket = self.vision_step_ms[name]
            bucket["count"] += 1
            bucket["total_ms"] += dt_ms
            if dt_ms > bucket["max_ms"]:
                bucket["max_ms"] = dt_ms
            self._vision_tick_ms[name] = self._vision_tick_ms.get(name, 0.0) + dt_ms
        except Exception:
            pass

    def _apply_vp_yaw(self, frame: Frame, R_wb: np.ndarray) -> np.ndarray:
        """(b) ABSOLUTE yaw backstop: vanishing-point heading -> ESKF yaw pseudo-measurement.

        Map-free + gate-free: runs every frame on the Manhattan structure. The estimator returns an
        absolute warehouse yaw MOD 90 deg (the lattice ambiguity) plus the four 90-deg branch headings;
        we DISAMBIGUATE by snapping to the branch NEAREST the gyro-propagated current yaw estimate
        (never a silent 90-deg flip), reject if even the nearest branch is implausibly far (ambiguous),
        else inject it via ``ESKFAHRS.update_yaw`` and REFRESH the attitude cache + R_wb so the
        correction reaches this tick's obs + downstream vision. Returns the (possibly refreshed) R_wb.
        Gated OFF / no use_ahrs -> immediate return of the input R_wb (byte-identical)."""
        if not (self.config.use_vp_yaw and self._ahrs is not None):
            return R_wb
        if frame is None or frame.image_bgr is None:
            return R_wb
        roll, pitch, yaw_hat = self._current_true_rpy()
        # "vp_yaw" times estimate_heading END TO END, which internally runs BOTH the VP RANSAC and
        # the Manhattan line extraction (racer.vision.manhattan_lines) -- there is no separate call
        # site for Manhattan in this module, so this one bucket covers both (see grep in the header).
        _t0 = time.perf_counter()
        est = estimate_heading(frame.image_bgr, roll, pitch, ransac_iters=self.config.vp_yaw_ransac_iters)
        self._time_step("vp_yaw", _t0)
        if est is None or est.quality < self.config.vp_yaw_min_quality:
            self.vision_diag.n_vp_yaw_rejected += 1
            return R_wb
        # Disambiguate the mod-90 lattice: pick the branch nearest the current (gyro) yaw estimate.
        branches = np.asarray(est.branch_headings_rad, dtype=np.float64)
        diffs = np.array([_wrap_pi(b - yaw_hat) for b in branches])
        k = int(np.argmin(np.abs(diffs)))
        if abs(diffs[k]) > self.config.vp_yaw_branch_max_rad:
            self.vision_diag.n_vp_yaw_rejected += 1
            return R_wb
        yaw_meas = float(branches[k])
        self._ahrs.eskf.update_yaw(yaw_meas, self.config.vp_yaw_noise_std)
        self.vision_diag.n_vp_yaw_applied += 1
        self.n_vp_yaw_total += 1
        self.vision_diag.last_vp_yaw_rad = yaw_meas
        return self._refresh_ahrs_attitude_cache()

    def _apply_floor_height(self, frame: Frame, R_wb: np.ndarray, t_fix_ns: int) -> None:
        """Z: map-free floor-grid camera height -> a world-down (z) KF correction.

        Backstop for the gate-relative vertical fix during no-gate / wrong-map stretches. The floor
        channel is ill-conditioned near the horizon, so it is gated HARD on quality + std_m. The
        recovered height is the camera's metres above the floor; world-down z = floor_z - height (the
        camera sits ABOVE the floor, so a larger height => a smaller/negative NED z). It is applied as a
        3-DOF world-position fix that is TIGHT in z (world-down) and ~infinite in-plane (the in-plane
        innovation is ~0 by construction since z's in-plane components are the current KF position), so
        it acts as an effective 1-DOF z correction that does not fight the in-plane gate-relative fix.
        Routed through ``_apply_pos_fix`` so it composes with BOTH the bare KF and the RewindKF (no
        ``update_position_z``); applied at the current time -> in-place for the OOSM path. Gated OFF /
        no use_ahrs -> no-op (byte-identical)."""
        if not (self.config.use_floor_height and self._ahrs is not None):
            return
        if frame is None or frame.image_bgr is None:
            return
        roll, pitch, _ = self._current_true_rpy()
        _t0 = time.perf_counter()
        est = estimate_floor_height(frame.image_bgr, roll, pitch,
                                    grid_cell_m=self.config.floor_grid_cell_m)
        self._time_step("floor_height", _t0)
        if (est is None or est.quality < self.config.floor_height_min_quality
                or est.std_m > self.config.floor_height_max_std_m):
            self.vision_diag.n_floor_z_rejected += 1
            return
        z_world = float(self.config.floor_camera_height_ref_m) - float(est.height_m)
        var_z = float(est.std_m) ** 2 + float(self.config.floor_height_extra_std_m) ** 2
        p = self.kf.position
        z_ned = np.array([p[0], p[1], z_world])                  # in-plane == current => 1-DOF in z
        cov = np.diag([_FLOOR_INPLANE_STD ** 2, _FLOOR_INPLANE_STD ** 2, var_z])
        self._apply_pos_fix(z_ned, cov, t_fix_ns)
        # A24: the dedicated vertical-velocity channel no longer consumes floor pins (the A21 KF's
        # update_z died with the washout rewrite -- floor-height was a false premise for a vertical
        # FIX on this wire; see racer.vertical_estimator). This 6-state-KF z correction is a
        # SEPARATE, independently-gated feature (``use_floor_height``) and is unaffected.
        self.vision_diag.n_floor_z_applied += 1
        self.n_floor_z_total += 1
        self.vision_diag.last_floor_height_m = float(est.height_m)

    def _apply_gate_bearing_yaw(self, pose: GatePose, gate: Gate) -> None:
        """(a) PRIMARY yaw lock: gate-bearing yaw to the KNOWN active-gate world position.

        FLIP-SAFE + NON-CIRCULAR. The world yaw is recovered by comparing TWO bearings to the active
        gate that DO NOT both depend on the current yaw estimate:
          * the WORLD bearing from the KNOWN gate position vs the KF position:
                bearing_world = atan2(dE, dN)   of (gate - p_KF)         # yaw-INDEPENDENT
          * the CAMERA-frame gate direction from the OBSERVATION (``pose.t_cam_gate``, the well-
            conditioned +L lever direction -- NOT the noisy planar-PnP rotation R_cam_gate), rotated to
            the body frame and de-tilted with the gravity-known roll/pitch into a YAW-ONLY world frame:
                d_level = Ry(pitch) Rx(roll) @ R_camera_from_body()^T @ (t_cam_gate / |t_cam_gate|)
                az_obs  = atan2(d_level_y, d_level_x)                    # observed gate azimuth, yaw-FREE
          * the drone world yaw is then ``bearing_world - az_obs`` (the rotation that maps the observed
            gate azimuth onto the true world bearing). This is INDEPENDENT of the ESKF yaw estimate, so
            it genuinely CORRECTS drift (deriving it from ``R_wb`` would be circular -- it would always
            return the current estimate and never correct anything).
        Yaw leverage collapses head-on (``az_obs -> 0``): gate on the off-boresight azimuth, and weight
        the datum sigma by ``1/|sin(az_obs)|`` so a near-head-on sighting barely tightens yaw. Only fires
        for the ACTIVE gate (RACE_STATUS.active_gate_index). Gated OFF / no active gate / no use_ahrs ->
        no-op."""
        if not (self.config.use_gate_bearing_yaw and self._ahrs is not None):
            return
        active = self._active_gate()
        if active is None or active.gate_id != gate.gate_id:
            return
        # OBSERVED gate azimuth in a yaw-only world frame (from the camera-frame lever direction).
        t = np.asarray(pose.t_cam_gate, dtype=np.float64)
        nt = float(np.linalg.norm(t))
        if nt < 1e-6:
            self.vision_diag.n_gate_bearing_yaw_rejected += 1
            return
        roll, pitch, _ = self._current_true_rpy()
        d_body = R_camera_from_body().T @ (t / nt)
        d_level = Rotation.from_euler("YX", [float(pitch), float(roll)]).as_matrix() @ d_body
        az_obs = float(np.arctan2(d_level[1], d_level[0]))      # observed azimuth (yaw-free)
        if abs(az_obs) < self.config.gate_bearing_min_offaxis_rad:
            self.vision_diag.n_gate_bearing_yaw_rejected += 1
            return
        # WORLD bearing to the KNOWN gate from the KF position (yaw-independent).
        delta = np.asarray(gate.position_ned, dtype=np.float64) - self.kf.position
        if float(np.hypot(delta[0], delta[1])) < 1e-3:
            self.vision_diag.n_gate_bearing_yaw_rejected += 1
            return
        bearing_world = float(np.arctan2(delta[1], delta[0]))
        yaw_meas = _wrap_pi(bearing_world - az_obs)
        # Weight: leverage ~ |sin(az_obs)|; inflate sigma as it collapses toward head-on.
        lev = max(abs(np.sin(az_obs)), 1e-3)
        sigma = float(self.config.gate_bearing_yaw_noise_std) / lev
        self._ahrs.eskf.update_yaw(yaw_meas, sigma)
        self.vision_diag.n_gate_bearing_yaw_applied += 1
        self.n_gate_bearing_yaw_total += 1
        self.vision_diag.last_gate_bearing_yaw_rad = yaw_meas
        # Refresh the cache so the corrected yaw reaches this tick's obs.
        self._refresh_ahrs_attitude_cache()

    def _active_gate(self) -> Gate | None:
        """The Gate the RACE_STATUS active_gate_index points at (ordered-list index), or None.

        ``active_gate_index`` is threaded onto ``DroneState`` (contracts.py). It indexes the ORDERED
        gate list; out-of-range / unset -> None (the gate-bearing yaw lock then no-ops)."""
        idx = self._active_gate_index
        if idx is None or idx < 0 or idx >= len(self.gates):
            return None
        return self.gates[idx]

    # -- vision -------------------------------------------------------------
    def _maybe_run_vision(self, ds: DroneState, frame: Frame | None, R_wb: np.ndarray) -> None:
        """Detect -> PnP -> associate -> innovation-gated KF position update, once per frame_id."""
        self.vision_diag = _VisionDiag()
        if (
            not self.config.use_vision
            or self.detector is None
            or frame is None
            or frame.image_bgr is None
            or frame.frame_id == self._last_frame_id
        ):
            return
        self._last_frame_id = frame.frame_id
        # Per-STEP timing (instrumentation only): reset this tick's step-ms accumulator, then flush
        # it into the worst-tick breakdown on every exit path via a try/finally so partial-frame
        # returns (no detections, etc.) still get credited and the worst tick stays accurate.
        self._vision_tick_ms = {}
        try:
            self._maybe_run_vision_timed(ds, frame, R_wb)
        finally:
            self._flush_vision_tick_timing()

    def _flush_vision_tick_timing(self) -> None:
        """Roll ``_vision_tick_ms`` (this tick's per-step ms) into the worst-tick breakdown if this
        tick's summed step time is the largest seen so far. Logging only; never raises."""
        try:
            tick_total = sum(self._vision_tick_ms.values())
            if tick_total > self._vision_worst_tick_total_ms:
                self._vision_worst_tick_total_ms = tick_total
                self._vision_worst_tick_ms = dict(self._vision_tick_ms)
        except Exception:
            pass

    def _maybe_run_vision_timed(self, ds: DroneState, frame: Frame, R_wb: np.ndarray) -> None:
        """Body of ``_maybe_run_vision`` after the dedup/gating checks -- split out so the per-tick
        timing flush in the caller's ``finally`` covers every exit path uniformly."""
        # P0-b: learn the camera/server -> IMU epoch offset ONCE, from this paired (frame, ds).
        # At the frame's capture instant the IMU clock reads ds.sim + (frame.recv - ds.recv)
        # (assuming 1:1 realtime), so delta_epoch = frame.sim - ds.sim - (frame.recv - ds.recv).
        # Same-clock data (recv=0, frame.sim==ds.sim) -> 0 -> back-compat is exact.
        if self.config.reconcile_vision_clock and self._delta_epoch_ns is None:
            self._delta_epoch_ns = (
                int(frame.sim_time_ns) - int(ds.sim_time_ns)
                - (int(frame.recv_monotonic_ns) - int(ds.recv_monotonic_ns))
            )
        # A17 decimation: count THIS processed vision tick, then run each CV backstop only every N-th
        # tick (N=vp_yaw_decimate / floor_height_decimate; 1 => every tick => byte-identical). On a
        # SKIPPED tick the backstop is not invoked at all (no estimate_heading / estimate_floor_height)
        # -- the ESKF gyro-propagates yaw + predicts z between. The update ORDER is unchanged.
        self._vision_tick_count += 1
        # Mag-free vision attitude/z anchors (magfree-vision-yaw-scope §4). These run PER FRAME,
        # independent of gate detection/association (the no-gate-in-view backstops): the VP heading
        # corrects the ESKF yaw (refreshing R_wb for the rest of this tick), and the floor-grid height
        # corrects the KF z. Both are OFF by default + require use_ahrs (they correct the ESKF/KF the
        # AHRS path owns); the OFF path skips them entirely -> byte-identical.
        if self._vision_tick_count % max(1, int(self.config.vp_yaw_decimate)) == 0:
            R_wb = self._apply_vp_yaw(frame, R_wb)
        if self._vision_tick_count % max(1, int(self.config.floor_height_decimate)) == 0:
            self._apply_floor_height(frame, R_wb, int(ds.sim_time_ns))

        # A15/A17 double-detect fix: route through the shared per-frame_id cache so the gate-seeker's
        # subsequent detect on the SAME frame reuses this result (detect() runs ONCE per frame).
        _t0 = time.perf_counter()
        observations = detect_cached(self.detector, frame)
        self._time_step("detect", _t0)
        self.vision_diag.n_detections = len(observations)
        if not observations:
            return

        drone_pos = self.kf.position
        predicted = predict_gates_in_camera(self.gates, drone_pos, R_wb)
        for obs in observations:
            self._process_observation(obs, predicted, drone_pos, R_wb, ds)

    def _process_observation(self, obs: GateObservation, predicted: dict, drone_pos, R_wb,
                             ds: DroneState) -> None:
        gate_id = self._associate(obs, predicted)
        if gate_id is None:
            return
        self.vision_diag.n_associated += 1
        gate = self._gates_by_id[gate_id]
        # The PnP prior (IPPE 2-fold / P3P disambiguation) is the FRESH map+attitude+KF
        # prediction, re-derived every frame -- motion-consistent by construction (the KF
        # propagates between fixes). The previous pose ESTIMATE was deliberately dropped as
        # a prior: one accepted flip made it sticky (each flipped pose endorsed the next).
        pg = predicted[gate_id]
        prior = GatePose(obs.frame_id, obs.sim_time_ns, pg.R_cam_gate, pg.t_cam_gate, 0.0,
                         gate_id=gate_id)
        _t0 = time.perf_counter()
        pose = estimate_gate_pose(obs, prior=prior, compute_covariance=True)
        self._time_step("pnp", _t0)
        if pose is None:
            return
        self.vision_diag.last_gate_id = gate_id
        self.vision_diag.last_range_m = pose.range_m
        self.vision_diag.last_reproj_px = pose.reproj_error_px
        if pose.range_m > self.config.vision_max_range_m:
            return
        # Known-gate-size depth sanity: the solved PnP depth must agree with the predicted
        # range to the associated gate, else the solver locked onto the wrong-scale
        # structure / a degenerate flip -- drop the fix at the source (don't lean on chi2).
        if not range_consistent(pose.range_m, pg.range_m,
                                self.config.fix_range_rel_tol, self.config.fix_range_abs_tol_m):
            self.n_vision_rejected += 1
            self.vision_diag.n_rejected_range += 1
            return

        position_ned, cov = gate_pose_to_world_position(
            pose, gate, R_wb, attitude_noise_std=self.config.attitude_noise_std,
            fix_cov_floor_std=self.config.fix_cov_floor_std
        )
        if pose.n_corners < 4:
            from racer.localization import P3P_FIX_COV_INFLATION

            cov = cov * P3P_FIX_COV_INFLATION

        # Mahalanobis innovation gate against the propagated prior: reject a fix that disagrees
        # with the prediction far beyond its own + the state covariance (wrong-gate / garbage).
        d2 = self._mahalanobis_position(position_ned, cov)
        self.vision_diag.last_mahalanobis = d2
        if d2 > self.config.vision_gate_chi2:
            self.n_vision_rejected += 1
            self.vision_diag.n_rejected_gate += 1
            return
        # The vision CAPTURE time on the IMU master clock (P0-b): for the RewindKF this is the OOSM
        # rewind target (apply the fix where the drone WAS); for the bare KF it is just the tsv stamp.
        t_fix_ns = self._vision_fix_time_imu_ns(ds, obs)
        # ABSOLUTE world fix (KEPT -- owns planning / along-track / g4->g5 handoff). Capture-time OOSM
        # when wrapped (degenerates to in-place at t_fix>=now); in-place for the bare KF (unchanged).
        self._apply_pos_fix(position_ned, cov, t_fix_ns)
        self.n_vision_fixes += 1
        self.vision_diag.n_applied += 1
        # P0-b: stamp the fix on the IMU master clock (NOT the raw camera/server epoch obs.sim).
        self._last_vision_sim_time_ns = t_fix_ns
        # (a) PRIMARY mag-free yaw lock: gate-bearing yaw to the KNOWN active-gate world position, on an
        # ACCEPTED fix (the same acceptance the absolute fix passed). Flip-safe (bearing, not PnP rotation);
        # skipped near head-on. OFF by default + requires use_ahrs -> byte-identical when gated off.
        if self.config.use_gate_bearing_yaw:
            self._apply_gate_bearing_yaw(pose, gate)
        # C2 gate-relative in-plane +L AUGMENT (BLUEPRINT §1.2/§1.3): a SECOND in-plane-only correction
        # applied AFTER the absolute fix, gated on its OWN relative-innovation test (the in-plane
        # backstop for depth-flips the absolute Maha + reproj gates pass).
        if self.config.use_gate_relative:
            self._apply_gate_relative_fix(pose, gate, R_wb, t_fix_ns)
        # B1 along-track RANGE channel (perception-l2): an attitude-independent range fix from the gate's
        # apparent SPAN, layered after the in-plane fix with its own 1-DOF gate. OFF by default -> skipped
        # entirely (byte-identical). Augments the gate-relative path, so it only runs when that is on too.
        if self.config.use_range_channel and self.config.use_gate_relative:
            self._apply_range_fix(obs, gate, R_wb, t_fix_ns)

    def _apply_pos_fix(self, z: np.ndarray, cov: np.ndarray, t_fix_ns: int) -> None:
        """Apply a world-position fix to the KF -- capture-time OOSM (RewindKF) or in-place (bare KF)."""
        if self._rewind:
            self.kf.update_position_at(int(t_fix_ns), z, cov)
        else:
            self.kf.update_position(z, cov)

    def _apply_gate_relative_fix(self, pose: GatePose, gate: Gate, R_wb: np.ndarray,
                                 t_fix_ns: int) -> None:
        """Gate-relative in-plane +L fix + the REQUIRED relative-innovation outlier gate (C2 §1.2/§1.3).

        The pseudo-fix world position ``z_rel == gate.position_ned - L`` equals the absolute fix; the WIN
        is the anisotropic cov (tight in-plane PnP lateral, NO bias floor; loose along-track). Reproj +
        the 3-DOF absolute Maha gate let depth-flips through, so this 2-DOF IN-PLANE innovation gate is
        the backstop: ``d2_rel = nu_ip^T S_ip^-1 nu_ip`` against the (post-absolute-fix) prior; accept
        iff ``<= chi2(2, 0.999)``. A rejected relative fix leaves the absolute estimate intact."""
        assert self.kf is not None
        z_rel, cov_rel = gate_relative_inplane_fix(
            pose, gate, R_wb,
            inplane_sigma=self.config.gate_rel_inplane_sigma,
            range_growth_a1=self.config.gate_rel_range_growth_a1,
            along_sigma=self.config.gate_rel_along_sigma,
            attitude_noise_std=self.config.attitude_noise_std,
            fix_cov_floor_std=self.config.fix_cov_floor_std,
        )
        # in-plane basis (world NED): gate-plane axes = R_world_gate columns 0 (right) and 1 (down).
        B = np.asarray(gate.R_world_gate, dtype=np.float64)[:, :2].T          # (2,3)
        nu_ip = B @ (z_rel - self.kf.x[:3])
        S_ip = B @ (self.kf.P[:3, :3] + cov_rel) @ B.T
        try:
            d2_rel = float(nu_ip @ np.linalg.solve(S_ip, nu_ip))
        except np.linalg.LinAlgError:
            d2_rel = 0.0   # singular in-plane S -> don't reject on a numerical artefact
        self.vision_diag.last_d2_rel = d2_rel
        if d2_rel > self.config.gate_rel_chi2:
            self.n_vision_rejected += 1
            self.vision_diag.n_rel_rejected += 1
            return
        self._apply_pos_fix(z_rel, cov_rel, t_fix_ns)
        self.vision_diag.n_rel_applied += 1
        self._last_fix_gate_R = np.asarray(gate.R_world_gate, dtype=np.float64).copy()

    def _apply_range_fix(self, obs: GateObservation, gate: Gate, R_wb: np.ndarray,
                         t_fix_ns: int) -> None:
        """B1 along-track RANGE pseudo-fix from the gate's apparent SPAN + a 1-DOF innovation gate.

        The span-derived range (``apparent_range_from_gate_span``) is ATTITUDE-INDEPENDENT and robust to
        the ε_vert boresight bias, so it gives a data-driven along-track constraint that replaces the
        loose GATE_REL_ALONG_SIGMA=0.5 m prior. The fix covariance is tight along-track / very loose
        in-plane, so the 3-DOF KF update is effectively a 1-DOF range correction that does NOT fight the
        in-plane gate-relative fix. Gated on the 1-DOF along-track innovation (chi2(1, 0.999)=10.83); a
        4-corner span is required (``apparent_range_from_gate_span`` returns None otherwise -> skip)."""
        assert self.kf is not None
        range_m = apparent_range_from_gate_span(obs.corners_px, gate.inner_size_m)
        self.vision_diag.last_range_span_m = float("nan") if range_m is None else float(range_m)
        if range_m is None:
            return
        sigma_range = gate_range_sigma(
            range_m, self.config.gate_range_sigma_rel, self.config.gate_range_sigma_floor)
        z_range, cov_range = gate_range_fix(
            range_m, gate, R_wb,
            sigma_range=sigma_range,
            fix_cov_floor_std=self.config.fix_cov_floor_std,
        )
        # along-track basis (world NED): gate-normal axis = R_world_gate column 2 (through-direction).
        n = np.asarray(gate.R_world_gate, dtype=np.float64)[:, 2]            # (3,)
        nu_al = float(n @ (z_range - self.kf.x[:3]))                          # scalar along-track innov
        s_al = float(n @ (self.kf.P[:3, :3] + cov_range) @ n)
        d2_range = (nu_al * nu_al / s_al) if s_al > 0.0 else 0.0
        self.vision_diag.last_d2_range = d2_range
        if d2_range > self.config.gate_range_chi2:
            self.n_vision_rejected += 1
            self.vision_diag.n_range_rejected += 1
            return
        self._apply_pos_fix(z_range, cov_range, t_fix_ns)
        self.vision_diag.n_range_applied += 1

    def _associate(self, obs: GateObservation, predicted: dict) -> int | None:
        """Match a detection to the map gate whose predicted SHAPE agrees best (or None)."""
        return associate(obs, predicted,
                         self.config.assoc_max_size_ratio, self.config.assoc_max_center_units)

    def _mahalanobis_position(self, z: np.ndarray, R: np.ndarray) -> float:
        """nu^T S^-1 nu for a position fix vs. the current KF state (H observes position)."""
        assert self.kf is not None
        nu = np.asarray(z, dtype=np.float64) - self.kf.x[:3]
        S = self.kf.P[:3, :3] + np.asarray(R, dtype=np.float64)
        try:
            return float(nu @ np.linalg.solve(S, nu))
        except np.linalg.LinAlgError:
            return 0.0   # singular S -> don't reject on a numerical artefact

    def _vision_fix_time_imu_ns(self, ds: DroneState, obs: GateObservation) -> int:
        """Effective time of a just-applied vision fix, on the IMU MASTER clock (P0-b).

        ``_nav_state`` measures ``time_since_vision_update_s`` as ``ds.sim_time_ns`` minus this,
        so this MUST be on the IMU epoch — not the raw camera/server epoch ``obs.sim_time_ns``.
          - Capture-time path (``reconcile_vision_clock`` + a learned ``delta_epoch``): the fix's
            true capture instant, ``obs.sim - delta_epoch``. delta_epoch is 0 on same-clock data,
            so this reduces to ``obs.sim`` and back-compat is exact.
          - Predict-forward fallback (no reconciliation / delta_epoch unknown): a constant
            calibrated age, ``now - vision_latency_const_s`` — needs NO usable capture stamp, so
            the chain works before a live TIMESYNC trace exists (BLUEPRINT §1.5, shipped first).
        """
        if self.config.reconcile_vision_clock and self._delta_epoch_ns is not None:
            return int(obs.sim_time_ns) - self._delta_epoch_ns
        return int(ds.sim_time_ns) - int(round(self.config.vision_latency_const_s * 1e9))

    # -- output -------------------------------------------------------------
    def _nav_state(self, ds: DroneState) -> NavState:
        if self.kf is None:                       # not yet initialized (no usable state seen)
            return make_nav_state(
                LinearKF.initialize(np.zeros(3), pos_std=5.0), ds, float("inf")
            )
        if self._last_vision_sim_time_ns is None:
            tsv = float("inf")
        else:
            tsv = max(0.0, (int(ds.sim_time_ns) - self._last_vision_sim_time_ns) / 1e9)
        # P0-c velocity: case-C velocity is NOT a separate vision-velocity surface (the d4v
        # vision-velocity channel was REFUTED, BLUEPRINT §0.4). Vision is position-only; velocity
        # is observable ONLY through position-fix differencing inside the KF (the pos/vel coupling
        # in predict's F). make_nav_state exports it as the frozen interface: NavState.velocity_ned
        # = kf.velocity (the obs vel_g = R_w2g @ vel consumes it) and pos_vel_covariance = the full
        # 6x6 KF P (its [3:6,3:6] block is the velocity covariance). No new estimator surface here.
        inplane_sig, along_sig = self._gate_frame_pos_sigma()
        # case-C (use_ahrs, GAP #2): source NavState attitude/rate from the AHRS instead of the
        # ODOMETRY-derived ds.roll/pitch/yaw + ds.angular_rate_body (both blocked in VQ2). The cached
        # values are ALREADY in the ODOMETRY-wire convention (_step_ahrs re-encodes the TRUE attitude),
        # so the controller (aliased euler) and the obs seam (build_obs re-conjugates) stay correct
        # with NO double-conjugation. OFF path passes None -> make_nav_state uses ds.* (byte-identical).
        att_override = None
        rate_override = None
        if self.config.use_ahrs and self._ahrs_odo_quat is not None:
            att_override = euler_from_quat_wxyz(self._ahrs_odo_quat)
            rate_override = self._ahrs_odo_rate
        # A24 vertical-velocity washout: export the smooth, bounded vz when the estimator is live
        # (vert_z stays permanently NaN -- no absolute-altitude state, see VerticalEstimator.z);
        # None otherwise -> the NavState fields stay NaN (make_nav_state's absent-marker, byte-identical).
        # A25: export the gate-relative z_off the SAME way (mirrored gating) -- None until a gate
        # has ever been latched (VerticalEstimator.z_off is NaN then, same absent-marker contract).
        vert_z = vert_vz = z_off = None
        if self._vert_est is not None and self._vert_est.seeded:
            vert_z, vert_vz = self._vert_est.z, self._vert_est.vz
            z_off = self._vert_est.z_off
        return make_nav_state(self.kf, ds, tsv, nav_inplane_sigma=inplane_sig,
                              nav_along_sigma=along_sig,
                              attitude_rpy_override=att_override,
                              angular_rate_override=rate_override,
                              vert_z_est=vert_z, vert_vz_est=vert_vz, z_off_est=z_off)

    def obs_drone_state(self, ds: DroneState) -> DroneState:
        """The DroneState the case-C OBS seam should consume (use_ahrs attitude routing, GAP #2).

        ``build_obs`` (via ``estimator_obs/estimator_obs20(ds, nav_state, ...)``) reads the ATTITUDE
        and BODY-RATE from the ``DroneState`` -- ``orientation_ned_wxyz`` (then RE-applies
        ``_ODO_QUAT_TRUE_CONJ``) and ``angular_rate_body`` (then RE-applies ``_ODO_RATE_SIGN``) -- NOT
        from the NavState. In VQ2 those wire fields are blocked (None / zero), so the deploy obs loop
        must hand build_obs a DroneState carrying the AHRS attitude.

        When ``use_ahrs`` is ON and an AHRS estimate exists, this returns ``ds`` with
        ``orientation_ned_wxyz`` / ``angular_rate_body`` replaced by the AHRS values RE-ENCODED into
        the ODOMETRY-wire convention (``_step_ahrs``): build_obs's re-conjugation then recovers the
        TRUE attitude/rate -> obs[6:12] correct, NO double-conjugation. When OFF (or pre-first-tick),
        returns ``ds`` UNCHANGED -> the VQ1 / case-A obs path is byte-identical.

        This is a thin, opt-in accessor: the deploy loop (the NEXT serial step) calls
        ``estimator_obs20(nav.obs_drone_state(ds), nav_state, ...)``; the OFF path / existing callers
        that pass the raw ``ds`` are unaffected.
        """
        if not (self.config.use_ahrs and self._ahrs_odo_quat is not None):
            return ds
        import dataclasses
        return dataclasses.replace(
            ds,
            orientation_ned_wxyz=np.asarray(self._ahrs_odo_quat, dtype=np.float64).copy(),
            angular_rate_body=np.asarray(self._ahrs_odo_rate, dtype=np.float64).copy(),
        )

    def _gate_frame_pos_sigma(self) -> tuple[float, float]:
        """Calibrated gate-frame position 1-sigma for the future confidence channel (C2 §1.6).

        Project the KF position covariance ``P[:3,:3]`` into the LAST-fix gate plane and return
        ``(inplane_sigma, along_sigma)`` where ``inplane = sqrt(P_g[ip0,ip0]+P_g[ip1,ip1])`` (the §1.6
        ``sigma_inplane_hat`` -- the combined in-plane 1-sigma) and ``along = sqrt(P_g[along,along])``.
        Gate-plane axes are ``R_world_gate`` columns 0,1 (in-plane) / 2 (along-track/normal). Returns
        ``(inf, inf)`` until a gate-relative fix has anchored a gate frame. BUILT for inc8; UNCONSUMED by
        the inc7 17-dim obs."""
        if self.kf is None or self._last_fix_gate_R is None:
            return float("inf"), float("inf")
        R_g2w = self._last_fix_gate_R
        P_gate = R_g2w.T @ self.kf.P[:3, :3] @ R_g2w               # NED cov -> gate frame
        inplane = float(np.sqrt(max(P_gate[0, 0] + P_gate[1, 1], 0.0)))
        along = float(np.sqrt(max(P_gate[2, 2], 0.0)))
        return inplane, along
