"""AHRS adapter shim: wraps ESKFAHRS into the navigator's attitude-source contract.

VQ2 self-localizing scope (case-C). The scored VQ2 wire BLOCKS the ATTITUDE /
ODOMETRY MAVLink messages (VADR-TS-003 §9.3), so the deployed stack must SELF-ESTIMATE
attitude from raw HIGHRES_IMU. Today the navigator reads attitude off the ODOMETRY
quaternion at exactly two consumption sites; this adapter is the drop-in replacement
SOURCE for both (it does NOT wire itself in -- that is a later serial step behind a
``use_ahrs`` flag; see docs/reactivation-2026-06-27/case-c-integration-scope.md §4 step 3-4).

What it replaces
----------------
GAP #1  navigator.py:416
    ``R_wb = R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)``
    -> consumed by ``LinearKF.predict(accel_body, R_wb, dt)`` (state_estimator.py:110)
       and the PnP lever ``R_wc = R_wb @ R_camera_from_body().T`` (localization).
    REPLACE WITH: ``AHRSAttitudeSource.R_wb`` (this adapter), a true FRD->NED rotation.

GAP #2  state_estimator.py:213-216 (``make_nav_state``)
    ``NavState.roll/pitch/yaw = drone_state.roll/pitch/yaw``
    ``NavState.angular_rate_body = drone_state.angular_rate_body``
    -> feed obs[6:9] ``rpy_g`` and obs[9:12] ``w_flu`` via fly_rl.build_obs.
    REPLACE WITH: ``AHRSAttitudeSource.euler_rpy`` and ``.body_rate`` (this adapter).

THE CONVENTION CONTRACT (read before wiring -- the +L / frame footguns live here)
---------------------------------------------------------------------------------
1. Quaternion / rotation: ``q_wxyz`` is (w,x,y,z) scalar-FIRST, body(FRD)->world(NED).
   ``R_wb`` is the corresponding 3x3 body->world matrix. This is the SAME convention as
   ``frames.R_world_from_odo_quat_wxyz`` and ``frames.euler_from_quat_wxyz`` -- i.e. the
   TRUE physical FRD->NED attitude.

2. *** THE ODOMETRY-CONJUGATION HAZARD (the single biggest wiring footgun) ***
   The ODOMETRY quat on the wire is R_y(pi)-conjugated -- which is why navigator.py:416
   calls ``R_world_from_odo_quat_wxyz`` (applies ``_ODO_QUAT_TRUE_CONJ``) and build_obs
   re-applies ``_ODO_QUAT_TRUE_CONJ`` / ``_ODO_RATE_SIGN`` to the raw wire fields. The
   ESKF estimates the TRUE FRD->NED attitude DIRECTLY -- there is NO telemetry-frame
   conjugation to undo. Therefore:
     * For GAP #1: ``self.R_wb`` is already TRUE -- it is a drop-in for the OUTPUT of
       ``R_world_from_odo_quat_wxyz(...)``, NOT for its raw quat input. Do not pass it
       back through that helper.
     * For GAP #2: the wiring step must route this adapter's TRUE quat/euler/rate into
       the obs WITHOUT re-applying ``_ODO_QUAT_TRUE_CONJ`` / ``_ODO_RATE_SIGN``. The
       current ``estimator_state_for_obs`` keeps the RAW ODOMETRY quat precisely so
       build_obs can conjugate it; an AHRS source must bypass that double-conjugation
       (e.g. supply euler/rates as overrides to make_nav_state, the §4 step-4 plan).

3. Euler: aerospace 3-2-1 (roll, pitch, yaw) in radians, decoded by the SAME function the
   navigator's downstream uses -- ``frames.euler_from_quat_wxyz`` (scipy ZYX). Byte-faithful
   to what ``make_nav_state`` stores in ``NavState.roll/pitch/yaw`` today.

4. *** BODY-RATE DEFINITION (footgun: world->body, NOT raw integrated gyro residual) ***
   The body rate is the WORLD-frame angular velocity expressed in BODY coordinates --
   i.e. omega such that ``R_dot = R_wb @ skew(omega)`` <=> ``omega = R_i2b @ w`` with
   ``R_i2b = R_wb.T`` (world/inertial -> body). For a strapdown gyro this is EXACTLY the
   instantaneous body-frame gyro reading, BIAS-CORRECTED: ``omega = gyro - gyro_bias``.
   That is what the ESKF integrates in ``_predict`` (``omega_corrected = gyro - b_g``) and
   what propagates the quaternion (``q <- q * exp(omega*dt/2)``). The adapter returns this
   bias-corrected body rate, NOT the raw ``gyro`` (which carries the estimated bias) and
   NOT a finite-difference of the quaternion. Frame is FRD body, matching the TRUE
   ``angular_rate_body`` convention obs[9:12] expects (the obs builder then flips FRD->FLU).

Master-clock discipline
------------------------
``ingest(...)`` takes an explicit ``dt`` in seconds; the caller MUST derive it from
consecutive ``ds.sim_time_ns`` deltas (the TIMESYNC-reconciled master sim clock), the SAME
``dt`` used by ``LinearKF.predict`` -- never wall-clock. ``dt <= 0`` is a no-op (re-packages
the current estimate), mirroring the navigator's ``dt<=0`` IMU-tick guard (navigator.py:422).

Pure numpy / scipy, torch-free, additive. Does NOT alter ESKFAHRS public behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from racer.ahrs.eskf import ESKFAHRS, _quat_to_R_wxyz
from racer.frames import euler_from_quat_wxyz


@dataclass
class AHRSAttitudeSource:
    """Drop-in attitude source wrapping an ``ESKFAHRS`` for the case-C nav loop.

    Construct once (own it in the Navigator), seed it (``seed`` or first ``ingest``),
    step it on every fresh IMU sample, and read ``R_wb`` / ``q_wxyz`` / ``euler_rpy`` /
    ``body_rate`` to feed the KF predict (GAP #1) and NavState/obs (GAP #2). Re-seed on a
    sim epoch reset via ``reset()`` (mirrors Navigator.reset / _initialize).

    Parameters
    ----------
    eskf : an ``ESKFAHRS`` instance. If ``None``, a default-tuned one is constructed
           (gyro_noise_std=0.01, accel_gate_alpha=10.0 -- the bench-validated defaults).
           Pass a pre-configured filter (e.g. with ``mag_ned`` set) for mag-aided yaw.

    Notes
    -----
    The adapter holds NO state of its own beyond a seeded flag and the last bias-corrected
    body rate; the attitude/bias/covariance live in the wrapped ESKF. This keeps the
    contract a thin, auditable seam (one filter, one convention) rather than a re-implementation.
    """

    eskf: ESKFAHRS = field(default_factory=lambda: ESKFAHRS(gyro_noise_std=0.01,
                                                            accel_gate_alpha=10.0))
    _seeded: bool = field(default=False, init=False)
    _last_body_rate: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)

    # -- lifecycle --------------------------------------------------------------

    @property
    def seeded(self) -> bool:
        """True once ``seed`` (or the first ``ingest``) has set the initial attitude."""
        return self._seeded

    def seed(self, q_init_wxyz: Optional[np.ndarray] = None) -> None:
        """Seed the initial attitude (w,x,y,z body->world). ``None`` = identity (level hover).

        Use this on a clean init / sim-epoch reset. A good seed for cold-start is the
        accel-gravity levelling attitude (roll/pitch from the first stationary accel);
        identity is the safe default and the ESKF's accel update pulls tilt in within the
        alignment window. Yaw is unobservable from accel alone -- seed it from mag or motion
        if available, else accept the identity-yaw datum (the obs are gate-RELATIVE, so a
        constant yaw offset is partly absorbed -- but do NOT rely on that; see scope §2 GAP #6).
        """
        self.eskf.reset(q_init_wxyz)
        self._last_body_rate = np.zeros(3)
        self._seeded = True

    def reset(self, q_init_wxyz: Optional[np.ndarray] = None) -> None:
        """Alias for ``seed`` -- re-seed cleanly (matches the ESKF/Navigator reset vocabulary)."""
        self.seed(q_init_wxyz)

    @staticmethod
    def level_seed_from_accel(accel_body: np.ndarray) -> np.ndarray:
        """Roll/pitch-levelled seed quaternion (w,x,y,z) from one stationary accel sample.

        Yaw is left at 0 (unobservable from gravity). At rest the body specific force is
        ``sf_body = R_wb^T @ (-g_ned) = [0,0,-g]`` level, so the gravity direction in body is
        ``g_body = R_wb^T @ g_ned = -sf_body`` (``[0,0,+g]`` level). Uses the closed-form
        aerospace levelling that matches ``frames.euler_from_quat_wxyz``'s ZYX convention EXACTLY
        (so a seed -> decode round-trips):
            roll  = atan2(g_body_y, g_body_z)
            pitch = atan2(-g_body_x, hypot(g_body_y, g_body_z))
        Returns identity for a degenerate (near-zero magnitude) sample. Convenience for a
        cold-start seed -- the caller decides whether the drone is stationary enough to trust it.
        """
        a = np.asarray(accel_body, dtype=np.float64)
        n = float(np.linalg.norm(a))
        if n < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])
        g_body = -a / n   # gravity direction in body: [0,0,+1] when level
        roll = float(np.arctan2(g_body[1], g_body[2]))
        pitch = float(np.arctan2(-g_body[0], np.hypot(g_body[1], g_body[2])))
        x, y, z, w = Rotation.from_euler("ZYX", [0.0, pitch, roll]).as_quat()
        return np.array([w, x, y, z])

    # -- per-tick ---------------------------------------------------------------

    def ingest(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: float,
        mag_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process one HIGHRES_IMU sample; return the updated attitude quaternion (w,x,y,z).

        Parameters
        ----------
        accel_body : (3,) body-frame specific force (m/s^2, FRD). Rest reading = [0,0,-g].
        gyro_body : (3,) RAW body-frame angular rate (rad/s, FRD) -- the HIGHRES_IMU gyro,
                    NOT the ODOMETRY-derived ``angular_rate_body`` (blocked in VQ2).
        dt : timestep (seconds), from consecutive ``ds.sim_time_ns`` deltas (master clock).
             ``dt <= 0`` is a no-op (re-packages the current estimate).
        mag_body : (3,) optional body-frame magnetometer (only used if the wrapped ESKF was
                   built with ``mag_ned``; otherwise ignored).

        Side effect: auto-seeds at identity on the very first call if not seeded yet.
        """
        if not self._seeded:
            self.seed(None)
        gyro = np.asarray(gyro_body, dtype=np.float64)
        if dt > 0:
            # The bias-corrected body rate the ESKF actually integrates this step
            # (omega = R_i2b @ w = gyro - bias). Snapshot BEFORE the step so it is the
            # rate that propagated THIS attitude (the bias estimate updates during step()).
            self._last_body_rate = gyro - self.eskf.gyro_bias
        self.eskf.step(gyro, np.asarray(accel_body, dtype=np.float64), float(dt),
                       mag_body=mag_body)
        return self.eskf.q_wxyz

    # -- attitude outputs (the navigator's expected contract) -------------------

    @property
    def q_wxyz(self) -> np.ndarray:
        """Current TRUE attitude as (w,x,y,z) body(FRD)->world(NED) quaternion."""
        return self.eskf.q_wxyz

    @property
    def R_wb(self) -> np.ndarray:
        """Current TRUE body(FRD)->world(NED) rotation matrix.

        Drop-in for the OUTPUT of ``frames.R_world_from_odo_quat_wxyz(...)`` (navigator.py:416).
        Already conjugation-free -- do NOT pass through R_world_from_odo_quat_wxyz again.
        """
        return _quat_to_R_wxyz(self.eskf.q_wxyz)

    @property
    def euler_rpy(self) -> Tuple[float, float, float]:
        """Aerospace 3-2-1 (roll, pitch, yaw) radians -- via ``frames.euler_from_quat_wxyz``.

        Byte-faithful to what ``make_nav_state`` stores in NavState.roll/pitch/yaw (GAP #2).
        """
        return euler_from_quat_wxyz(self.eskf.q_wxyz)

    @property
    def body_rate(self) -> np.ndarray:
        """Bias-corrected TRUE FRD body angular rate (rad/s): omega = R_i2b @ w = gyro - bias.

        Drop-in for the TRUE ``angular_rate_body`` that feeds obs[9:12] ``w_flu`` (GAP #2).
        This is the world-frame angular velocity expressed in body coordinates -- the rate
        that propagates the quaternion -- NOT the raw gyro and NOT a quaternion difference.
        Returns the rate from the last ``dt>0`` ingest (zeros before the first such step).
        """
        return self._last_body_rate.copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        """Current ESKF gyro-bias estimate (rad/s, body FRD)."""
        return self.eskf.gyro_bias

    @property
    def attitude_uncertainty_rad(self) -> float:
        """1-sigma attitude uncertainty (rad) -- gate the alignment transient on this.

        During the cold-start alignment window this is large; the wiring step should
        hold/inflate the KF predict (scope §2 GAP #6) until it falls below a budget.
        """
        return self.eskf.attitude_uncertainty_rad
