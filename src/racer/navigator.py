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
from racer.kf_rewind import RewindKF
from racer.localization import (
    FIX_COV_FLOOR_STD,
    GATE_REL_ALONG_SIGMA,
    GATE_REL_INPLANE_SIGMA,
    GATE_REL_RANGE_GROWTH_A1,
    INPLANE_POS_FLOOR_STD,
    gate_pose_to_world_position,
    gate_relative_inplane_fix,
)
from racer.state_estimator import LinearKF, make_nav_state

# chi-square 99.9% quantile, 2 DOF -- the IN-PLANE relative-innovation outlier gate (BLUEPRINT §1.3).
# (The absolute 3-DOF Mahalanobis gate is 16.27; the gate-relative fix is 2-DOF in-plane -> 13.82.)
GATE_REL_CHI2_2_999 = 13.815510557964274
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
    last_gate_id: int | None = None
    last_range_m: float = float("nan")
    last_reproj_px: float = float("nan")
    last_mahalanobis: float = float("nan")
    last_d2_rel: float = float("nan")   # last gate-relative in-plane innovation statistic (C2)


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
    # P0-b: learned camera/server -> IMU epoch offset (frame.sim - imu.sim, recv-paired). None
    # until the first processed frame; re-learned on reset(). 0 on same-clock data.
    _delta_epoch_ns: int | None = field(default=None, repr=False)
    # C2: True when self.kf is a RewindKF (case-C OOSM path); set at _initialize from the config.
    _rewind: bool = field(default=False, repr=False)
    # C2: R_world_gate (3,3) of the gate the last ACCEPTED gate-relative fix landed on -- the frame the
    # NavState confidence export projects P into (§1.6). None until the first gate-relative fix.
    _last_fix_gate_R: np.ndarray | None = field(default=None, repr=False)
    _gates_by_id: dict[int, Gate] = field(default_factory=dict, repr=False)

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
        self._last_sim_time_ns = int(ds.sim_time_ns)
        self._reset_counter = int(ds.reset_counter)
        self.initialized = True

    def reset(self) -> None:
        """Drop the estimate (e.g. on a sim epoch restart); the next update re-seeds it."""
        self.kf = None
        self.initialized = False
        self._last_frame_id = None
        self._last_vision_sim_time_ns = None
        self._delta_epoch_ns = None        # P0-b: re-learn the epoch offset after a sim restart
        self._last_fix_gate_R = None       # C2: drop the confidence-export gate frame on restart

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
        if self._rewind:
            self.kf.predict(ds.accel_body, R_wb, dt, int(ds.sim_time_ns))   # RewindKF: IMU-clock stamp
        else:
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
        # P0-b: learn the camera/server -> IMU epoch offset ONCE, from this paired (frame, ds).
        # At the frame's capture instant the IMU clock reads ds.sim + (frame.recv - ds.recv)
        # (assuming 1:1 realtime), so delta_epoch = frame.sim - ds.sim - (frame.recv - ds.recv).
        # Same-clock data (recv=0, frame.sim==ds.sim) -> 0 -> back-compat is exact.
        if self.config.reconcile_vision_clock and self._delta_epoch_ns is None:
            self._delta_epoch_ns = (
                int(frame.sim_time_ns) - int(ds.sim_time_ns)
                - (int(frame.recv_monotonic_ns) - int(ds.recv_monotonic_ns))
            )
        observations = self.detector.detect(frame)
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
        # C2 gate-relative in-plane +L AUGMENT (BLUEPRINT §1.2/§1.3): a SECOND in-plane-only correction
        # applied AFTER the absolute fix, gated on its OWN relative-innovation test (the in-plane
        # backstop for depth-flips the absolute Maha + reproj gates pass).
        if self.config.use_gate_relative:
            self._apply_gate_relative_fix(pose, gate, R_wb, t_fix_ns)

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
        return make_nav_state(self.kf, ds, tsv, nav_inplane_sigma=inplane_sig,
                              nav_along_sigma=along_sig)

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
