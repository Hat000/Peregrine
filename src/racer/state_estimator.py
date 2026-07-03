"""Linear position/velocity Kalman filter (the estimation slice).

The sim HANDS us attitude (ATTITUDE message), so we do NOT estimate orientation:
the state is just position + velocity in world NED, which makes the whole filter
LINEAR (a plain KF, not an ESKF). The full error-state KF is the documented
fallback if attitude/IMU turn out to be biased.

    state x = [px, py, pz, vx, vy, vz]   (world NED, metres / m/s)

Predict: integrate the IMU. The accelerometer reports SPECIFIC FORCE in the body
frame (FRD), i.e. at rest it reads -g on the down axis (our DroneState/Elodin-
adapter convention). Kinematic world acceleration is therefore

    a_world = R_world_body @ accel_body + g_world,   g_world = [0, 0, +9.80665]  (NED, down +)

so gravity cancels at rest (the #1 sign trap — covered by a test). Constant-accel
discretisation is exact.

Update: generic linear ``update(z, H, R)`` plus conveniences:
- ``update_position`` — vision fix (drone world position from gate PnP + map; the
  PnP->world conversion lives upstream, in the mapper/localisation glue).
- ``update_position_z`` — baro altitude (NED z = -altitude_above_ref).
- ``update_velocity`` — only if a telemetry message provides velocity.

Magnetometer/yaw are NOT used here: yaw comes from the given attitude. Mag is
reserved for the ESKF fallback / an attitude cross-check.

Deferred rebuilds (noted, NOT built) [red-team 2026-05-30]:
1. ESKF bias-state to ESTIMATE attitude bias, not merely inflate Q for it (see
   ``attitude_noise_std``) -- build only if first-contact data shows the sim's given
   attitude is biased under high-G; covariance inflation masks drift, it does not remove a
   systematic bias.
2. Delayed-vision ring buffer: rewind to a fix's capture sim-time and re-propagate the
   buffered IMU (out-of-sequence handling). ``update_position`` currently applies the fix to
   the CURRENT state -- fine at VQ1 speed (~cm), matters at VQ2 speed (~0.6-1 m at 20 m/s).
   Prereq: the video capture clock must be reconciled with HIGHRES_IMU (TIMESYNC).
Both are upgrades behind this filter's interface, not rewrites of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from racer.contracts import DroneState, NavState
from racer.frames import ATTITUDE_NOISE_STD_RAD

_I3 = np.eye(3)
_Z3 = np.zeros((3, 3))
GRAVITY_NED = np.array([0.0, 0.0, 9.80665])


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix: ``_skew(v) @ w == np.cross(v, w)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

_H_POS = np.hstack([_I3, _Z3])            # observes position
_H_VEL = np.hstack([_Z3, _I3])            # observes velocity
_H_Z = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # observes pz (baro)


@dataclass
class LinearKF:
    """6-state [pos, vel] world-NED Kalman filter driven by IMU + position fixes."""

    x: np.ndarray                      # (6,)
    P: np.ndarray                      # (6,6)
    accel_noise_std: float = 0.3       # m/s^2, accel process/measurement noise (tune to the IMU)
    # rad, 1-sigma error in the GIVEN attitude [review 3A]. Measured through the vision chain
    # (frames.ATTITUDE_NOISE_STD_RAD, vision-pkg2 2026-06-10) -- an upper bound on the pure
    # given-attitude error, so the Q inflation it drives here is honest-to-conservative.
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD
    gravity_ned: np.ndarray = field(default_factory=lambda: GRAVITY_NED.copy())
    process_floor: float = 1e-6        # tiny diagonal to keep Q full-rank
    max_dt_s: float = 0.2              # reject implausibly large predict steps (sim reset/stutter) [red-team]
    # In-plane (horizontal N-E) POSITION-variance STATE floor [parked #74, coast-drift 2026-06-15]. A
    # dense gate-relative fix stream drives the position covariance P -> R/N -> ~0 (independent updates
    # accumulate), so the Kalman gain for new fixes -> 0 and the filter STOPS trusting fixes and rides the
    # drifting IMU ("centering-blind") exactly when a well-pointed inc8 policy makes fixes densest. This
    # floors the SMALLEST eigenvalue of the horizontal position block P[:2,:2] to
    # ``inplane_pos_floor_std**2`` AFTER each update, so the filter can never claim better than
    # ~sigma_floor gate-relative centering accuracy (it physically can't -- the irreducible systematic
    # bias sigma_b survives the boresight bake) and the gain stays responsive. DISTINCT from the
    # MEASUREMENT-cov floor (localization.FIX_COV_FLOOR_STD): repeated floored-R updates still drive
    # P -> R/N -> 0; only a STATE floor keeps the gain alive. 0.0 = DISABLED -> byte-identical to the bare
    # filter (the VQ1 / case-A path); the case-C estimator chain sets it ~0.05 m (= sigma_ref,
    # localization.INPLANE_POS_FLOOR_STD). Floors the in-plane (lateral/centering) horizontal eigenvalue
    # ONLY: the loose along-track horizontal eigenvalue stays >> floor (untouched), and the vertical
    # (world-down) + velocity blocks are left as-is (the boresight/ESKF pathway owns the vertical bias).
    inplane_pos_floor_std: float = 0.0

    @classmethod
    def initialize(
        cls,
        position_ned: np.ndarray,
        velocity_ned: np.ndarray | None = None,
        pos_std: float = 1.0,
        vel_std: float = 1.0,
        **kw,
    ) -> "LinearKF":
        x = np.zeros(6)
        x[:3] = np.asarray(position_ned, dtype=np.float64)
        if velocity_ned is not None:
            x[3:] = np.asarray(velocity_ned, dtype=np.float64)
        P = np.diag([pos_std**2] * 3 + [vel_std**2] * 3).astype(np.float64)
        return cls(x=x, P=P, **kw)

    # -- prediction ---------------------------------------------------------
    def predict(self, accel_body: np.ndarray, R_world_body: np.ndarray, dt: float) -> None:
        """Propagate by dt using body specific force + the given body->world rotation.

        ``dt`` must come from consecutive HIGHRES_IMU ``time_usec`` stamps (the master sim
        clock), NOT the wall-clock loop period, or the integration scales wrong. A
        non-positive or implausibly large ``dt`` (a sim reset, a clock stutter, or the first
        sample) is dropped rather than integrated across the discontinuity -- the navigator
        owns detecting a reset and re-initialising the filter. [red-team 2026-05-30]
        """
        if dt <= 0 or dt > self.max_dt_s:
            return
        specific_force_world = R_world_body @ np.asarray(accel_body, dtype=np.float64)
        a_world = specific_force_world + self.gravity_ned
        F = np.block([[_I3, dt * _I3], [_Z3, _I3]])
        B = np.vstack([0.5 * dt * dt * _I3, dt * _I3])   # (6,3)
        self.x = F @ self.x + B @ a_world
        # Acceleration noise = isotropic IMU noise + attitude-error projected through the
        # rotated specific force. A small attitude error rotates the ~9.8 m/s^2 specific-
        # force vector, injecting ~(|s| * sigma_theta) of phantom horizontal acceleration
        # that would otherwise integrate into unmodelled drift on a blind coast. Covariance
        # sigma_theta^2 * [s]x [s]x^T is PSD; at hover s~[0,0,-g] it adds g^2 to the
        # horizontal axes and 0 to vertical (gravity tilts sideways, not up/down). [review 3A]
        S = _skew(specific_force_world)
        accel_cov = self.accel_noise_std**2 * _I3 + self.attitude_noise_std**2 * (S @ S.T)
        Q = B @ accel_cov @ B.T + self.process_floor * np.eye(6)
        self.P = F @ self.P @ F.T + Q

    # -- correction ---------------------------------------------------------
    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        y = z - H @ self.x
        PHt = self.P @ H.T
        S = H @ PHt + R
        # Solve S K^T = (P H^T)^T rather than forming inv(S): identical Kalman gain, but more
        # numerically stable + faster (no explicit inverse). The Joseph-form update below keeps
        # P symmetric positive-definite. [red-team 2026-05-30]
        K = np.linalg.solve(S, PHt.T).T
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        # Joseph form: stays symmetric + positive-definite under finite precision.
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._apply_inplane_pos_floor()

    def _apply_inplane_pos_floor(self) -> None:
        """Floor the smallest eigenvalue of the horizontal (N-E) position covariance block to
        ``inplane_pos_floor_std**2`` -- the in-plane STATE floor that keeps the Kalman gain responsive
        under a dense fix stream (see the field doc; parked #74).

        Eigenvalue (not per-axis diagonal) floor: the binding gate-LATERAL/centering direction is a
        horizontal axis at the gate's yaw, so only the SMALLEST horizontal eigenvalue (whatever direction
        it points) is the over-converged one -- this raises exactly it and leaves the larger horizontal
        eigenvalue (the loose along-track direction, var >> floor) untouched. PSD + symmetry are preserved:
        raising a principal sub-block's eigenvalues adds a PSD increment ``[[dA,0],[0,0]] (dA >= 0)`` to P.
        NO-OP -- P bit-unchanged -- when disabled OR when the block already clears the floor, so a
        non-over-converging filter (VQ1, sparse fixes, the G3 in-band ~0.13 m sigma) is byte-identical."""
        floor_var = self.inplane_pos_floor_std ** 2
        if floor_var <= 0.0:
            return
        block = 0.5 * (self.P[:2, :2] + self.P[:2, :2].T)   # defensive symmetrize (Joseph keeps it so)
        w, V = np.linalg.eigh(block)
        if w[0] >= floor_var:                                # smallest eigenvalue already above the floor
            return
        w = np.maximum(w, floor_var)
        self.P[:2, :2] = (V * w) @ V.T

    def update_position(self, position_ned: np.ndarray, covariance: np.ndarray) -> None:
        self.update(np.asarray(position_ned, dtype=np.float64), _H_POS, covariance)

    def update_position_z(self, pz: float, variance: float) -> None:
        self.update(np.array([pz]), _H_Z, np.array([[variance]]))

    def update_velocity(self, velocity_ned: np.ndarray, covariance: np.ndarray) -> None:
        self.update(np.asarray(velocity_ned, dtype=np.float64), _H_VEL, covariance)

    # -- accessors ----------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()


def make_nav_state(
    kf: LinearKF,
    drone_state: DroneState,
    time_since_vision_update_s: float,
    nav_inplane_sigma: float = float("inf"),
    nav_along_sigma: float = float("inf"),
    *,
    attitude_rpy_override: tuple[float, float, float] | None = None,
    angular_rate_override: np.ndarray | None = None,
    vert_z_est: float | None = None,
    vert_vz_est: float | None = None,
    z_off_est: float | None = None,
    contact_frozen: bool | None = None,
) -> NavState:
    """Assemble a NavState from the KF (position/velocity) + given attitude/rates.

    ``nav_inplane_sigma`` / ``nav_along_sigma`` are the calibrated gate-frame position 1-sigma (C2
    confidence-channel feeder, BLUEPRINT §1.6); ``inf`` when no gate-relative fix has anchored a gate
    frame yet. ``kf`` is a ``LinearKF`` or its ``RewindKF`` proxy (``.position`` / ``.velocity`` / ``.P``
    forward to the wrapped filter).

    ``attitude_rpy_override`` / ``angular_rate_override`` (case-C ``use_ahrs``, GAP #2): when supplied,
    the NavState attitude/rate are sourced from the AHRS rather than ``drone_state.roll/pitch/yaw`` /
    ``drone_state.angular_rate_body`` (the ODOMETRY-derived fields blocked in VQ2). BOTH default ``None``
    -> the existing wire-attitude path, BYTE-IDENTICAL (no new array drawn). The caller (Navigator
    under ``use_ahrs``) supplies them already in the SAME convention the existing consumers expect
    (the controller's aliased ``roll/pitch/yaw`` + ODOMETRY-sign ``angular_rate_body``), so no
    downstream recalibration is needed.

    ``vert_z_est`` / ``vert_vz_est`` (A21 vertical-channel estimator, ``racer.vertical_estimator``):
    the smooth (altitude, vz) export the ff-owns-vertical alt-hold damps on. BOTH default ``None``
    -> the NavState fields stay NaN (the frozen absent-marker), BYTE-IDENTICAL for every existing
    caller; the Navigator supplies them only under ``use_vertical_estimator``.

    ``z_off_est`` (A25 gate-relative vertical offset, ``VerticalEstimator.z_off``): default ``None``
    -> ``NavState.z_off_est`` stays NaN, BYTE-IDENTICAL for every existing caller; the Navigator
    supplies it only under the same ``use_vertical_estimator`` gate.

    ``contact_frozen`` (A26 contact-gate export, ``VerticalEstimator.contact_frozen()``): default
    ``None`` -> ``NavState.contact_frozen`` stays ``None`` (absent-marker), BYTE-IDENTICAL for every
    existing caller; the Navigator supplies it only under the same ``use_vertical_estimator`` gate."""
    if attitude_rpy_override is None:
        roll, pitch, yaw = drone_state.roll, drone_state.pitch, drone_state.yaw
    else:
        roll, pitch, yaw = (float(attitude_rpy_override[0]), float(attitude_rpy_override[1]),
                            float(attitude_rpy_override[2]))
    if angular_rate_override is None:
        rate = np.asarray(drone_state.angular_rate_body, dtype=np.float64).copy()
    else:
        rate = np.asarray(angular_rate_override, dtype=np.float64).copy()
    return NavState(
        sim_time_ns=drone_state.sim_time_ns,
        position_ned=kf.position,
        velocity_ned=kf.velocity,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        angular_rate_body=rate,
        pos_vel_covariance=kf.P.copy(),
        time_since_vision_update_s=time_since_vision_update_s,
        nav_inplane_sigma=float(nav_inplane_sigma),
        nav_along_sigma=float(nav_along_sigma),
        vert_z_est=float("nan") if vert_z_est is None else float(vert_z_est),
        vert_vz_est=float("nan") if vert_vz_est is None else float(vert_vz_est),
        z_off_est=float("nan") if z_off_est is None else float(z_off_est),
        contact_frozen=None if contact_frozen is None else bool(contact_frozen),
    )
