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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import DroneState, Frame, Gate, GateObservation, GatePose, NavState
from racer.frames import ATTITUDE_NOISE_STD_RAD, R_world_from_body, R_world_from_odo_quat_wxyz
from racer.localization import FIX_COV_FLOOR_STD, gate_pose_to_world_position
from racer.state_estimator import LinearKF, make_nav_state
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

_WORLD_DOWN = np.array([0.0, 0.0, 1.0])   # NED down


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


@dataclass
class _VisionDiag:
    """Last-tick vision diagnostics (off the NavState contract; for logging / the handoff)."""

    n_detections: int = 0
    n_associated: int = 0
    n_applied: int = 0
    n_rejected_gate: int = 0
    n_rejected_range: int = 0       # post-PnP depth-sanity rejections (range_consistent)
    last_gate_id: int | None = None
    last_range_m: float = float("nan")
    last_reproj_px: float = float("nan")
    last_mahalanobis: float = float("nan")


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
    vision_diag: _VisionDiag = field(default_factory=_VisionDiag, repr=False)

    _last_sim_time_ns: int = field(default=0, repr=False)
    _reset_counter: int = field(default=0, repr=False)
    _last_frame_id: int | None = field(default=None, repr=False)
    _last_vision_sim_time_ns: int | None = field(default=None, repr=False)
    _gates_by_id: dict[int, Gate] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._gates_by_id = {g.gate_id: g for g in self.gates}

    # -- lifecycle ----------------------------------------------------------
    def _initialize(self, ds: DroneState) -> None:
        """Seed the KF from the first usable state (given position if present, else origin)."""
        pos = (
            np.asarray(ds.position_ned, dtype=np.float64)
            if ds.position_ned is not None
            else np.zeros(3)
        )
        vel = (
            np.asarray(ds.velocity_ned, dtype=np.float64)
            if ds.velocity_ned is not None
            else None
        )
        pos_std = self.config.given_pos_std if ds.position_ned is not None else 5.0
        self.kf = LinearKF.initialize(pos, vel, pos_std=pos_std, vel_std=1.0)
        self._last_sim_time_ns = int(ds.sim_time_ns)
        self._reset_counter = int(ds.reset_counter)
        self.initialized = True

    def reset(self) -> None:
        """Drop the estimate (e.g. on a sim epoch restart); the next update re-seeds it."""
        self.kf = None
        self.initialized = False
        self._last_frame_id = None
        self._last_vision_sim_time_ns = None

    # -- per-tick -----------------------------------------------------------
    def update(self, ds: DroneState, frame: Frame | None = None) -> NavState:
        """Advance the estimate with one telemetry snapshot (+ optional camera frame)."""
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
        # TRUE physical body->world rotation from the raw ODOMETRY quat (R_y(pi)-conjugated).
        # The CTBR path uses euler_from_quat_wxyz on the raw quat (aliased, VQ1-proven —
        # that path is untouched). Vision/PnP/KF must use the true attitude. [vision-frame-fix]
        R_wb = R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)

        # Estimation advances only on a NEW IMU sample (sim_time_ns is the master clock). When the
        # control loop ticks faster than the IMU, dt<=0 and we just re-package the current state
        # (no predict, no re-applying a stale measurement -> no covariance collapse).
        dt = (int(ds.sim_time_ns) - self._last_sim_time_ns) / 1e9
        if dt <= 0:
            self._maybe_run_vision(ds, frame, R_wb)   # a fresh frame can still land between IMU ticks
            return self._nav_state(ds)
        self._last_sim_time_ns = int(ds.sim_time_ns)

        # 1) IMU predict (gravity-corrected specific force, given attitude). 2) vision fix
        # (looser, innovation-gated, against the propagated prior). 3) given pos/vel (tight,
        # authoritative for VQ1). Vision before given so the innovation gate compares vision to
        # the IMU prior (a meaningful disagreement check), while given still anchors the estimate.
        self.kf.predict(ds.accel_body, R_wb, dt)
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
        observations = self.detector.detect(frame)
        self.vision_diag.n_detections = len(observations)
        if not observations:
            return

        drone_pos = self.kf.position
        predicted = predict_gates_in_camera(self.gates, drone_pos, R_wb)
        for obs in observations:
            self._process_observation(obs, predicted, drone_pos, R_wb)

    def _process_observation(self, obs: GateObservation, predicted: dict, drone_pos, R_wb) -> None:
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
        pose = estimate_gate_pose(obs, prior=prior, compute_covariance=True)
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
        self.kf.update_position(position_ned, cov)
        self.n_vision_fixes += 1
        self.vision_diag.n_applied += 1
        self._last_vision_sim_time_ns = int(obs.sim_time_ns)

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
        return make_nav_state(self.kf, ds, tsv)
