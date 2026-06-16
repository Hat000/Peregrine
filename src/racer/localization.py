"""Localization glue: a camera-relative gate PnP + the gate's known world pose ->
a drone world-position measurement for the Kalman filter.

The camera shares the body origin (spec 3.8) and is tilted per ``racer.frames``. With
the drone attitude TRUSTED (from telemetry) we know camera->world; the gate's world
position is known from the map; the PnP gives the gate origin in the camera frame.
Therefore:

    R_world_camera = R_world_body @ R_camera_from_body().T
    p_drone_world  = gate.position_ned - R_world_camera @ gatepose.t_cam_gate

The drone POSITION depends only on the PnP translation (and the trusted attitude), not
on the gate's estimated rotation. The fix covariance carries three measured terms: the
rotated PnP translation block (inflated, ``PNP_FIX_COV_INFLATION``), the attitude
lever-arm term (``frames.ATTITUDE_NOISE_STD_RAD``), and the constant-systematics floor
(``FIX_COV_FLOOR_STD``):

    Cov(p_drone) = K * R_wc @ Cov(t_cam_gate) @ R_wc.T
                   + sigma_theta^2 (|L|^2 I - L L^T) + sigma_floor^2 I

A camera lever-arm offset (the official sim says same origin; the Elodin rig offsets the
camera) and multi-gate PnP against all visible corners remain future refinements.
"""
from __future__ import annotations

import numpy as np

from racer import frames
from racer.contracts import Gate, GatePose
from racer.frames import ATTITUDE_NOISE_STD_RAD, R_camera_from_body
from racer.state_estimator import LinearKF


# Analytic-PnP covariance inflation [perception-char 2026-06-08]. The 4-corner world-fix covariance
# is propagated from the confidence-weighted-refine Fisher information, which models ONLY the per-
# corner pixel noise (``WEIGHTED_SIGMA_PX``); it understates the real fix error (sub-pixel detector
# bias, heavy-tailed corner localisation, the 1.5 m gate-size model mismatch). Measured consequence:
# the navigator's chi2_0.999 = 16.27 innovation gate dropped ~20% of GOOD fixes because their analytic
# cov was ~3.5x too tight (good-fix maha ~ f*chi2(3), with f ~ 3.5). This scalar inflates the analytic
# PnP translation cov so the gate keeps catching wrong-gate / depth-flip fixes (maha in the hundreds-
# to-thousands) while passing healthy ones. Conservative default; sweep + confirm on per-gate course
# bundles against the <=~2% catastrophic-leak ceiling. Does NOT touch the attitude lever-arm term
# (physically calibrated) nor the no-covariance fallback. See handoff/perception-char-2026-06-08.
PNP_FIX_COV_INFLATION = 2.0   # variance multiplier on the analytic 4-corner PnP world-fix covariance

# Isotropic 1-sigma FLOOR on the world-fix covariance [vision-pkg2 2026-06-10]. The measured fix
# error carries range-INDEPENDENT systematics no scaled term can cover: a ~+0.3 m-high constant
# vertical offset (map opening-centre vs true / camera-height), per-gate lateral offsets
# (+-0.1..0.2 m), and a close-range depth bias (~+0.5 m under 8 m). At short range the attitude
# lever-arm term vanishes (|L| -> 0) and the analytic PnP cov is mm-tight, so those constants
# made the chi2_0.999 gate reject 35% of GOOD gate-0 close fixes (maha in the hundreds). Joint
# MLE with the attitude term on the canonical course recording: floor 0.407 m (shipped 0.40),
# attitude 1.37 deg -- good-fix over-rejection 13%->0% (<1 m) / 17%->1-2% (<3 m) at unchanged
# catastrophic leak. See handoff/shadowpc-vision-pkg2-2026-06-10.
FIX_COV_FLOOR_STD = 0.40      # m, added in quadrature to every vision world-fix covariance

# In-plane (lateral/centering) STATE-covariance floor for the case-C KF [parked #74, coast-drift
# 2026-06-15]. NOT a MEASUREMENT floor (that is FIX_COV_FLOOR_STD above, added to R): this floors the KF's
# own POSITION covariance P so a dense gate-relative fix stream cannot drive it -> R/N -> 0 -- which would
# send the Kalman gain -> 0 and make the filter stop trusting fixes / ride the drifting IMU
# ("centering-blind") exactly when a well-pointed inc8 policy makes fixes densest. A floored R does NOT
# fix this (repeated floored-R updates still drive P -> R/N -> 0); only a STATE floor keeps the gain
# alive. Set to the same sigma_b "irreducible systematic centering floor" the obs confidence-reference
# uses (sigma_ref = 0.05 m, rl.estimator_emul.SIGMA_REF_M) and that the coast-drift sig_p0 (+) sig_b
# budget assumes: the KF can never honestly claim better than ~0.05 m gate-relative centering because the
# systematic bias sigma_b survives the boresight bake. Applied in-plane (horizontal) ONLY -- the vertical
# (world-down) bias is owned by the boresight/ESKF pathway. Consumed via LinearKF.inplane_pos_floor_std,
# activated only in the case-C path (NavigatorConfig.use_inplane_pos_floor + use_rewind_kf/gate_relative).
INPLANE_POS_FLOOR_STD = 0.05  # m, KF in-plane position-variance STATE floor (case-C; the sigma_b systematic)


def _apply_camera_vert_offset(position_ned: np.ndarray, R_world_body: np.ndarray) -> np.ndarray:
    """METRIC boresight: the +L lever recovers the camera OPTICAL CENTRE world position; if the camera
    centre sits at body-frame offset [0,0,vert_offset_m] (FRD +Z down) from the body origin (spec 3.8
    says 0; the render rig / CoM reference may differ), the body/drone position is
        p_body = p_camera_centre - R_world_body @ [0, 0, vert_offset_m].
    Guarded on != 0.0 so the DEFAULT path is byte-identical (line skipped). Applied identically at BOTH
    +L lever sites (gate_pose_to_world_position, gate_relative_inplane_fix). Does NOT touch +L.

    Reads ``frames.BORESIGHT`` LIVE (module attr, NOT a ``from frames import BORESIGHT`` snapshot) so it
    is genuinely "the ONE correction instance every consumer reads" (frames.py): a P3 source-edit or a
    runtime swap of ``frames.BORESIGHT`` is seen here, mirroring R_camera_from_body() reading frames'
    own global. At the all-zero default this is byte-identical either way (the guard is False)."""
    if frames.BORESIGHT.vert_offset_m != 0.0:
        return position_ned - np.asarray(R_world_body, dtype=np.float64) @ np.array(
            [0.0, 0.0, frames.BORESIGHT.vert_offset_m])
    return position_ned


def gate_pose_to_world_position(
    gate_pose: GatePose,
    gate: Gate,
    R_world_body: np.ndarray,
    default_position_std: float = 0.3,
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD,
    pnp_cov_inflation: float = PNP_FIX_COV_INFLATION,
    fix_cov_floor_std: float = FIX_COV_FLOOR_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Drone world position (NED) + 3x3 covariance from one gate sighting.

    ``R_world_body`` is the trusted attitude (e.g. ``frames.R_world_from_body(roll,
    pitch, yaw)``). If ``gate_pose.covariance`` is None, falls back to an isotropic
    ``default_position_std`` for the PnP term. When present, the analytic PnP covariance is scaled by
    ``pnp_cov_inflation`` (it is optimistic -- see ``PNP_FIX_COV_INFLATION``) before propagation.

    The fix is ``p = gate_pos - R_world_camera @ t_cam_gate``, so its error has THREE measured
    components [vision-pkg2 2026-06-10]:
    the PnP translation (``gate_pose.covariance``, pixel noise); attitude error rotating the
    lever arm ``L = R_world_camera @ t_cam_gate`` -- dominant at range (a 1.4-deg error at 20 m
    is ~0.5 m, far above the PnP cm-noise), added as the induced covariance of a small random
    rotation of L, ``sigma_theta^2 (|L|^2 I - L L^T)`` (PSD; the same skew-projection the predict
    step uses), keyed off ``attitude_noise_std`` (measured chain value, see
    ``frames.ATTITUDE_NOISE_STD_RAD``); and the range-independent constant systematics
    (map-centre/per-gate/close-range-depth offsets), dominant at SHORT range where the other two
    terms vanish, covered by the isotropic ``fix_cov_floor_std`` floor (``FIX_COV_FLOOR_STD``).
    [red-team 2026-05-30; measured + floor added vision-pkg2 2026-06-10]
    """
    R_world_camera = np.asarray(R_world_body, dtype=np.float64) @ R_camera_from_body().T
    lever = R_world_camera @ np.asarray(gate_pose.t_cam_gate, dtype=np.float64)  # gate rel. drone, world NED
    position_ned = _apply_camera_vert_offset(gate.position_ned - lever, R_world_body)
    if gate_pose.covariance is not None:
        sigma_tt = np.asarray(gate_pose.covariance, dtype=np.float64)[:3, :3]
        # Inflate the (optimistic) analytic PnP translation cov before propagating -- see
        # ``PNP_FIX_COV_INFLATION``. Only the analytic branch; the fallback below is already loose.
        cov = pnp_cov_inflation * (R_world_camera @ sigma_tt @ R_world_camera.T)
    else:
        cov = (default_position_std**2) * np.eye(3)
    if attitude_noise_std > 0.0:
        cov = cov + attitude_noise_std**2 * (float(lever @ lever) * np.eye(3) - np.outer(lever, lever))
    if fix_cov_floor_std > 0.0:
        cov = cov + (fix_cov_floor_std**2) * np.eye(3)
    return position_ned, cov


# Gate-relative in-plane per-fix lateral 1-sigma [C2-ESTIMATOR-CHAIN, MEASURED c1_gate_relative
# 2026-06-13]. The accepted (maha<=16.27, 4-corner) gate-4 PnP lateral noise, per gate-plane axis, at
# the transit band. This is the per-fix in-plane noise the gate-relative observation delivers -- it does
# NOT carry the per-track map/registration bias (the obs is referenced to the SEEN corners, not the
# map), so the 0.40 m FIX_COV_FLOOR_STD bias-absorption floor is REMOVED in-plane. SINGLE swappable
# constant by design: Track-3 shadow-mode (vision-vs-truth at speed) may recalibrate it; everything keys
# off this one number. [BLUEPRINT §1.2 / §1.5; note: per-fix sigma at 37 m/s is a best-case LOWER bound]
GATE_REL_INPLANE_SIGMA = 0.265   # m/axis, in-plane (gate-plane) gate-relative PnP lateral 1-sigma

# Range growth of the lateral sigma [range_anisotropic_R.py lateral law, a1 = sigma_px/(f*sqrt(N))].
# The in-plane sigma is ``max(GATE_REL_INPLANE_SIGMA, a1*r)``: the c1-MEASURED 0.265 is the in-band
# floor (accepted gate-4 fixes, ~9-10 m, which is exactly a1*~10.2 m -> the law and the measurement
# AGREE there, so 0.265 is NOT double-counted), and the ``a1*r`` law takes over only at LONGER range
# (r > 0.265/a1 ~= 10.2 m). The gate-relative fix is consumed inside ~12 m of gate-4 so it is essentially
# flat 0.265 in-band (matches the c1 validated 0.139 RMS arm). CAVEAT (carry): a1=0.026 EXTRAPOLATES
# BADLY past ~24 m (the depth axis extrapolates faithfully, the lateral does not) -- flag for >24 m.
GATE_REL_RANGE_GROWTH_A1 = 0.026   # per-metre lateral sigma growth law (takes over beyond ~10 m)

# Along-track (gate-normal) base 1-sigma for the gate-relative fix. LOOSE by design: the absolute fix +
# IMU own depth; the relative term must NEVER fight the absolute term on the along-track axis. Set ~the
# absolute radial sigma; the helper adds the attitude lever-arm term + the FIX_COV_FLOOR_STD on top
# (where the range/depth bias still lives). [BLUEPRINT §1.5]
GATE_REL_ALONG_SIGMA = 0.50        # m, along-track base 1-sigma (loose)


def gate_relative_inplane_fix(
    gate_pose: GatePose,
    gate: Gate,
    R_world_body: np.ndarray,
    inplane_sigma: float = GATE_REL_INPLANE_SIGMA,
    range_growth_a1: float = GATE_REL_RANGE_GROWTH_A1,
    along_sigma: float = GATE_REL_ALONG_SIGMA,
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD,
    fix_cov_floor_std: float = FIX_COV_FLOOR_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Gate-RELATIVE in-plane position pseudo-fix + anisotropic covariance (BLUEPRINT §1.2/§1.3/§1.5).

    THE core case-C fix. The PnP lever to the SEEN opening is ``L = R_world_camera @ t_cam_gate`` (the
    +L lever, world NED -- exactly as ``gate_pose_to_world_position`` builds it). The pseudo-fix world
    position ``z = gate.position_ned - L`` is the SAME measurement the absolute fix produces; the
    gate-relative WIN is NOT a different ``z`` but (1) the COVARIANCE shaping below and (2) sourcing the
    policy obs from the resulting tightly-pinned in-plane estimate, where the per-track map bias ``db``
    cancels EXACTLY: ``pos_g = R_w2g @ (gate_map - p_KF)`` with ``p_KF`` pulled to ``gate_map - L`` by
    this tight in-plane fix == ``R_w2g @ L`` (the +L direct lever), and ``gate_map``'s ``db`` is common
    to both terms (re-run: rel E_bias -0.000 vs abs/submap +0.176/+0.174, c1_gate_relative).

    Covariance is shaped in the GATE-PLANE frame (``gate.R_world_gate`` columns: X=right, Y=down are
    in-plane; Z=through is along-track) then rotated to world NED:
      - in-plane (X,Y): tight PnP lateral ``sigma = max(inplane_sigma, a1*range)`` (flat 0.265 in the
        gate-4 window, the a1*r law taking over only beyond ~10 m), NO 0.40 m floor (that floor is
        bias-absorption; the relative obs has no bias to absorb -- adding it would stop the rel arm
        clearing the margin).
      - along-track (Z): LOOSE -- ``along_sigma`` base + the attitude lever-arm term ``(sigma_theta*|L|)``
        + the ``fix_cov_floor_std`` floor (range/depth bias lives here); the absolute fix + IMU own it.

    Returns ``(z_ned, cov_ned)`` ready for ``LinearKF.update_position`` / ``RewindKF.update_position_at``
    (a 3-DOF update whose anisotropic cov makes it effectively in-plane-only; no new KF method needed).
    The caller gates it on the relative-innovation test (navigator) before applying. ``inplane_sigma``
    is the single swappable calibration constant (Track-3 may recalibrate)."""
    R_world_camera = np.asarray(R_world_body, dtype=np.float64) @ R_camera_from_body().T
    lever = R_world_camera @ np.asarray(gate_pose.t_cam_gate, dtype=np.float64)   # +L, gate rel. drone
    z_ned = _apply_camera_vert_offset(gate.position_ned - lever, R_world_body)
    r = float(np.linalg.norm(lever))
    sig_ip = max(inplane_sigma, range_growth_a1 * r)   # flat 0.265 in-band; a1*r law beyond ~10 m
    var_along = along_sigma ** 2
    if attitude_noise_std > 0.0:
        var_along += (attitude_noise_std * r) ** 2
    if fix_cov_floor_std > 0.0:
        var_along += fix_cov_floor_std ** 2
    R_gate_to_world = np.asarray(gate.R_world_gate, dtype=np.float64)            # gate-frame -> world NED
    cov_gate = np.diag([sig_ip ** 2, sig_ip ** 2, var_along])                   # (X_ip, Y_ip, Z_along)
    cov_ned = R_gate_to_world @ cov_gate @ R_gate_to_world.T
    return z_ned, cov_ned


# Gate-transit coast policy [red-team 2026-05-30, Tier B]. A 3-corner P3P fix (a gate
# clipping out of frame at transit; GatePose.n_corners < 4) is geometrically weaker and can't
# self-disambiguate, so we DON'T let it yank the estimate: inflate its measurement covariance
# so the KF leans on the IMU prediction (a "soft coast") instead of snapping to a maybe-wrong
# pose. The HARD coast (drop vision entirely within X m of a gate) and an innovation /
# Mahalanobis gate that rejects wrong-gate "teleport" fixes (master-plan NEG-3; needs the
# mapper + live range-to-gate) live in the navigator loop and are deferred to that wiring.
P3P_FIX_COV_INFLATION = 9.0   # variance multiplier (=3x std) for a 3-corner fix; tune at first contact


def apply_gate_pose_update(
    kf: LinearKF,
    gate_pose: GatePose,
    gate: Gate,
    R_world_body: np.ndarray,
    default_position_std: float = 0.3,
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD,
    pnp_cov_inflation: float = PNP_FIX_COV_INFLATION,
    fix_cov_floor_std: float = FIX_COV_FLOOR_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a gate sighting to a world-position fix and apply it to the KF.

    ``attitude_noise_std`` adds the lever-arm attitude-uncertainty term to the fix covariance
    (see ``gate_pose_to_world_position``) so distant fixes are trusted appropriately less, and
    ``fix_cov_floor_std`` the constant-systematics floor so close fixes are not over-trusted.
    ``pnp_cov_inflation`` scales the (optimistic) analytic 4-corner PnP cov so the innovation gate
    keeps a healthy fix instead of over-rejecting it. A weak 3-corner (P3P) fix has its covariance
    inflated by ``P3P_FIX_COV_INFLATION`` on top, so it nudges rather than snaps the estimate at gate
    transit (the soft gate-transit coast).
    """
    position_ned, cov = gate_pose_to_world_position(
        gate_pose, gate, R_world_body, default_position_std, attitude_noise_std,
        pnp_cov_inflation, fix_cov_floor_std
    )
    if gate_pose.n_corners < 4:
        cov = cov * P3P_FIX_COV_INFLATION
    kf.update_position(position_ned, cov)
    return position_ned, cov
