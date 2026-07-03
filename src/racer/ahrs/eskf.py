"""Error-State Kalman Filter AHRS.

Estimates attitude (body->world quaternion) from gyroscope and accelerometer,
following the documented ESKF intent in state_estimator.py's module docstring.

State (error-state formulation)
--------------------------------
Nominal state  : quaternion q (w,x,y,z, body->world)
Error state    : delta_x = [delta_phi (3), delta_b_g (3)]   (6-dim)
  delta_phi   : small-angle rotation error in the BODY frame
  delta_b_g   : gyroscope bias error in the body frame

Prediction
----------
Propagate the nominal quaternion by the gyro measurement (gyro - bias_estimate):
  q_new = q * exp((omega_meas - b_g) * dt / 2)
Propagate the error-state covariance through the linearised dynamics:
  P_new = F @ P @ F^T + Q

Accelerometer update (with high-g gating)
-----------------------------------------
When |a_meas| is close to g (within threshold), the accel points mostly at -g in
body frame and carries tilt information. We form the predicted gravity direction in
the body frame from the nominal quaternion and use it as the measurement model.

Free-fall / high-|a| magnitude guard (the catastrophic-failure SAFETY NET)
--------------------------------------------------------------------------
The accelerometer equals (specific force = reaction to) gravity ONLY when |a| ~ g.
In FREE-FALL the proper acceleration collapses (|a| -> ~0), so the accel reads near
ZERO and its NORMALISED direction ``a/|a|`` points at whatever tiny residual / noise
remains -- it is NO LONGER the gravity direction. Feeding that bogus direction into
the tilt update can INVERT the attitude estimate (the VQ2 A6 free-fall tumble: a ~2 Hz
thrust slam cut commanded thrust to the alt floor -> brief near-free-fall -> the AHRS
lost its gravity reference -> est roll/yaw flipped to +-180 deg). The chi2 innovation
gate is NOT a reliable catch here (in free-fall the prior covariance is small and a
near-zero measurement can still test "consistent" enough to leak through, and a slewing
attitude makes the innovation ambiguous). The principled, unconditional catch is a
MAGNITUDE band: the accel update is SKIPPED entirely (gyro-propagate only, no bogus
gravity correction) whenever |a| falls outside ``[g*(1-tol_lo), g*(1+tol_hi)]``.
Free-fall (|a|~1, ~90% below g) is WAY outside any sane band, so even a loose band
catches it deterministically. When |a| ~ g (the overwhelming majority of ticks) the
band is a NO-OP and the update is byte-identical to before. This is COMPLEMENTARY to
(and EARLIER than) the chi2 gate, not a replacement.

CRITICAL: During high-g maneuvers, |a_meas| >> g, so the accel is dominated by
kinematic acceleration and DOES NOT point at -g. Using it naively would corrupt the
attitude estimate. The gating mechanism DOWNWEIGHTS the accel update:

  gate_weight = exp(-alpha * ((|a| - g) / g)^2)

This weight multiplies the measurement noise covariance (R_accel / gate_weight),
effectively ignoring the accel when the drone is pulling g's.

CRITICAL (empirical, this bench): magnitude gating ALONE is insufficient. A
random-direction acceleration whose MAGNITUDE happens to sit near g (e.g. ~43% of
the HIGH_G_RANDOM samples) sails through the magnitude gate while its DIRECTION is
wildly wrong, poisoning the tilt update (ESKF p90 blew up to ~67 deg). The fix is a
second, direction-aware gate: a Mahalanobis / chi-square consistency test on the
innovation itself (the same family as the relinnov chi2 gate in the C2 estimator
chain). The accel update is APPLIED ONLY IF
    innovation^T S^-1 innovation <= accel_chi2_thresh   (chi2(3, .95) = 7.815)
With this innovation gate the ESKF beats both classical filters across every high-g
scenario; without it, magnitude gating alone loses to a gyro-trusting Madgwick.
This two-gate design is what distinguishes the ESKF from Madgwick/Mahony, which lack
any principled rejection of acceleration disturbances.

Magnetometer update (optional)
-------------------------------
If mag_ned is provided (NED reference field) and mag_body is given to step(),
a yaw-only update is applied. VQ2 mag usability is unknown; the update is
gated off by default (mag_ned=None).

Frame convention
----------------
Quaternion: (w,x,y,z) scalar-FIRST, body(FRD)->world(NED). Matches frames.py.
Gravity NED: [0, 0, +9.80665] m/s^2.
Specific force at rest in body FRD: [0, 0, -g] (IMU reports gravity up).

Extension stubs
---------------
- RIANN/GRU learned gyro-denoising: replace the raw gyro with a denoised version
  before the propagation step. Interface: denoiser(gyro_history) -> omega_clean.
- Invariant EKF / EqVIO (van Goor ANU thesis): replace the linearised F/H with
  the geometrically-consistent Lie-group Jacobians on SO(3). The prediction/update
  structure here is compatible; swap _compute_F() and _accel_H() to switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


GRAVITY = 9.80665
G_NED = np.array([0.0, 0.0, GRAVITY])   # NED: down = +Z


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 with (w,x,y,z) layout."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 3x3 rotation matrix (body->world)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-z*w),    2*(x*z+y*w)],
        [2*(x*y+z*w),    1-2*(x*x+z*z),  2*(y*z-x*w)],
        [2*(x*z-y*w),    2*(y*z+x*w),    1-2*(x*x+y*y)],
    ])


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix: _skew(v) @ w == cross(v, w)."""
    x, y, z = v
    return np.array([[0., -z,  y],
                     [z,  0., -x],
                     [-y,  x,  0.]])


def _omega_exp_wxyz(omega: np.ndarray, dt: float) -> np.ndarray:
    """Closed-form quaternion exponential: exp(omega * dt / 2) = delta_q."""
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / np.linalg.norm(omega)
    half = angle / 2.0
    return np.array([np.cos(half),
                     axis[0]*np.sin(half),
                     axis[1]*np.sin(half),
                     axis[2]*np.sin(half)])


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


def _wrap_angle(a: float) -> float:
    """Wrap an angle into (-pi, pi]. Used by the yaw pseudo-measurement innovation so a near-2*pi
    raw difference (e.g. measured +179 deg vs estimated -179 deg) is the small +2 deg correction it
    physically is, NOT a catastrophic ~358 deg yank."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _yaw_jacobian_dphi(R_wb: np.ndarray) -> np.ndarray:
    """EXACT sensitivity of the aerospace ZYX world yaw to the ESKF body-frame error delta_phi.

    With ``psi = atan2(R_wb[1,0], R_wb[0,0])`` and the right/body perturbation ``dR = R_wb @ skew(e_i)``:
        d(psi)/d(delta_phi_i) = (R00 * dR[1,0] - R10 * dR[0,0]) / (R00^2 + R10^2).
    Returns the (3,) row vector (zeros at the gimbal-lock degeneracy ``R00^2+R10^2 -> 0``, |pitch|->90deg).
    Shared by ``ESKFAHRS.update_yaw`` (the measurement Jacobian) and ``yaw_uncertainty_rad`` so they
    agree exactly. FD-pinned in tests/test_vision_yaw_wiring.py."""
    R_wb = np.asarray(R_wb, dtype=np.float64)
    c = R_wb[0, 0] ** 2 + R_wb[1, 0] ** 2
    h = np.zeros(3)
    if c > 1e-12:
        for i in range(3):
            dR = R_wb @ _skew(np.eye(3)[i])
            h[i] = (R_wb[0, 0] * dR[1, 0] - R_wb[1, 0] * dR[0, 0]) / c
    return h


@dataclass
class ESKFAHRS:
    """Error-State Kalman Filter attitude estimator.

    Parameters
    ----------
    gyro_noise_std : rad/s, 1-sigma white noise on each gyro axis.
    gyro_bias_std : rad/s/sqrt(s), random-walk on gyro bias (rate noise).
    accel_noise_std : m/s^2, 1-sigma white noise on each accel axis.
    accel_gate_alpha : high-g gating sharpness. Larger = sharper rejection.
                       alpha=0 disables gating (naive filter). alpha~10 is a
                       good starting point for 4-5 g maneuvers.
    mag_ned : (3,) NED reference magnetic field vector. None = no mag update.
    mag_noise_std : magnetometer noise (normalised field units, 1-sigma).
    """
    gyro_noise_std: float = 0.01        # rad/s
    gyro_bias_std: float = 1e-4         # rad/s/sqrt(s)
    accel_noise_std: float = 0.3        # m/s^2
    accel_gate_alpha: float = 10.0      # high-g sharpness (0 = disabled)
    accel_chi2_thresh: float = 7.815    # innovation-gate threshold; chi2(3, .95)=7.815
                                        # (0 or negative = innovation gate disabled)
    # Free-fall / high-|a| magnitude guard (the catastrophic-failure safety net). The accel
    # is only the gravity reference when |a| ~ g; skip the accel update entirely (gyro-propagate
    # only) when |a| is outside [g*(1-tol_lo), g*(1+tol_hi)]. ASYMMETRIC by design:
    #   * tol_lo (low side) catches FREE-FALL (|a|->~0). The default 0.75 means "reject below
    #     0.25 g (|a| < 2.45 m/s^2)" -- free-fall |a|~1 is caught, while the synthetic bench's
    #     adversarial HIGH_G_RANDOM tail (min |a|~3.24 m/s^2, ~0.33 g) is left to the chi2 gate,
    #     preserving the ESKF<->IEKF cross-validation. A deployment can tighten this toward ~0.25.
    #   * tol_hi (high side) is a coarse high-g backstop; the smooth accel_gate_alpha weight +
    #     chi2 gate already handle sustained high-g, so the default 9.0 (reject above 10 g) is a
    #     loose last-resort net that never trips in the bench (max |a|~5.5 g).
    # tol_lo<0 or tol_hi<0 disables that side. A no-op band (very loose) leaves behaviour identical.
    accel_freefall_tol_lo: float = 0.75   # reject when |a| < g*(1-0.75) = 0.25 g  (free-fall)
    accel_freefall_tol_hi: float = 9.0    # reject when |a| > g*(1+9.0)  = 10 g    (extreme high-g)
    # ----------------------------------------------------------------------------------------------
    # Acceleration-aware accel rejection (the VQ2 A8 climb-and-retreat fix, 2026-06-30).
    # ----------------------------------------------------------------------------------------------
    # The MIDDLE regime the magnitude band CANNOT see: under SUSTAINED LINEAR ACCELERATION the
    # specific-force vector keeps |a| ~ g (so it sails through _accel_magnitude_in_band) while its
    # DIRECTION tilts away from -g. The accel then levels the estimate to a TILTED reference -> false
    # pitch -> the controller "corrects" the wrong way -> more accel -> more tilt -> a divergent loop
    # (A8: accel-apparent pitch drifted -23.6 deg over 4 s while the gyro integrated only -0.8 deg).
    #
    # The discriminating signal is the LINEAR-ACCELERATION RESIDUAL relative to a GYRO-ANCHORED
    # reference attitude (NOT the live estimate):
    #     a_lin_world = R_ref @ f_body + g_ned        (== true world kinematic accel, 0 at equilibrium)
    # |a_lin| ~ 0 when hovering/coasting (the accel IS the gravity reference -> trust it); |a_lin| grows
    # during powered tilted flight (the accel is contaminated -> distrust it). This is the Mahony/
    # complementary-filter principle (the accel may only correct SLOW gyro drift): when the accel-implied
    # gravity direction moves but the GYRO says the body did not rotate, the accel is lying.
    #
    # WHY A GYRO-ANCHORED REFERENCE (the subtle, load-bearing part). If a_lin were computed against the
    # LIVE estimate R_hat, the A8 divergence is INVISIBLE: the accel slowly drags R_hat to follow the
    # tilting specific force, so R_hat @ f_body + g_ned stays ~0 even as the estimate walks 23 deg off
    # true. The fix MUST anchor to something the accel cannot drag -- the GYRO. ``_R_ref`` is propagated
    # by the bias-corrected gyro ONLY (never accel-corrected). When the gyro is flat (the A8 segment),
    # R_ref holds level and a_lin reveals the FULL contamination (0 -> ~4 m/s^2); for a GENUINE rotation
    # R_ref turns with the gyro and a_lin stays ~0 (the accel honestly tracks -> not rejected). R_ref is
    # RE-ANCHORED to the live estimate every step the gate is inert (|a_lin| below accel_motion_anchor_thr),
    # so it tracks the trusted estimate at equilibrium and only "remembers the gyro truth" across a suspect
    # powered window -- bounding the gyro-bias drift it would otherwise accumulate.
    #
    # We DOWN-WEIGHT (not hard-cliff) the accel update by inflating R_accel proportional to |a_lin|^2:
    #     R_accel <- R_accel * (1 + (|a_lin| / accel_motion_scale)^2)
    # Graceful so it composes with the smooth gate weight + the chi2 gate + the free-fall band. At
    # equilibrium (|a_lin| ~ 0) the factor is 1.0 -> BYTE-IDENTICAL to today. The gate is OFF by default
    # (use_accel_motion_reject=False) so existing behaviour + the ESKF<->IEKF cross-validation are
    # untouched until a deployment opts in. FD/divergence pinned in tests/test_accel_motion_reject.py.
    use_accel_motion_reject: bool = False
    accel_motion_scale: float = 0.1       # m/s^2; |a_lin| at which R_accel DOUBLES. Tight by design: a
                                          # pure graceful (∝|a_lin|^2) inflation settles to an EQUILIBRIUM
                                          # tilt (accel pull vs gyro hold), so the knee must be tight enough
                                          # that sustained contamination is decisively distrusted (a_lin~0.3
                                          # -> 10x, the A8 ~2-4 m/s^2 -> 400-1600x = effectively a skip).
                                          # Crucially this is INERT where it must be: true equilibrium and
                                          # GENUINE rotation both give |a_lin| ~ 0 (the accel honestly tracks
                                          # the gyro reference) -> factor ~ 1, so it never starves the bias
                                          # correction in normal slow flight; it only bites when the body is
                                          # truly LINEARLY accelerating (when the accel is genuinely lying).
    accel_motion_anchor_thr: float = 0.3  # m/s^2; re-anchor the gyro reference to the live estimate when
                                          # |a_lin| is below this (the trusted/equilibrium regime). Above it
                                          # the reference free-runs on the gyro to expose sustained accel.
                                          # 0.3 (~0.03 g) sits above the static accel-noise floor (~0.09
                                          # m/s^2 at accel_noise_std=0.05) but well below the A8 sustained
                                          # contamination, so normal slow flight keeps re-anchoring cleanly.
    accel_motion_max_inflate: float = 1e4 # ceiling on the inflation factor (numerical guard; a huge
                                          # |a_lin| effectively skips the update without a hard branch).
    # ----------------------------------------------------------------------------------------------
    # A32 "ALWAYS FIND DOWN" robust accel trust (2026-07-03) -- spec:
    # handoff/vq2_a32_robust_estimation_spec_2026-07-03.md §3.1. Replay of run 20260703_172104
    # proved the A8 motion-reject inflation above is a SELF-LOCKING DISTRUST LOOP: the residual
    # ``a_lin = R_ref@f + g`` is ATTITUDE-REFERENCED, so once the attitude is wrong by theta the
    # gravity no longer cancels (|a_lin| ~ 2g*sin(theta/2) even for a perfectly clean, resting
    # accelerometer), the inflation rails at 1e4 (median 4,950x in the inverted tail, 671x even in
    # NORMAL flight at scale=0.1), the Kalman gain on the gravity pull goes to ~0, and theta never
    # shrinks -- the filter sat at ~150 deg roll while |a| = 9.81 exactly implied 180 deg, forever.
    # The fix is the Mahony-2008 / PX4 / ArduPilot pattern grafted into the ESKF: every rejection
    # mechanism BOUNDED and SOFT, the accel's pull toward gravity floored whenever |a| ~ g (the
    # structural recovery guarantee), the gyro-anchored reference TIME-LIMITED, and a persistence-
    # triggered attitude-reset watchdog. ALL of it behind this one flag (default False =
    # byte-identical: none of the fields below are even read on the OFF path). vq2_case_c opts in
    # via NavigatorConfig.ahrs_accel_trust_v2.
    use_accel_trust_v2: bool = False
    # v2 motion-inflation bounds (replace the three A8 values above when the flag is ON):
    accel_trust_v2_motion_scale: float = 1.0   # m/s^2 knee where R DOUBLES. Real maneuvers (~2-3
                                               # m/s^2) now inflate 5-10x, not 400-1600x -- the
                                               # gravity pull (and gyro-bias observability) stays
                                               # ALIVE through the whole flight (kills F2).
    accel_trust_v2_max_inflate: float = 25.0   # NEVER more than 25x distrust from this mechanism:
                                               # with gate=1 the correction can be slowed, never
                                               # severed (kills F1's lockout arm).
    accel_trust_v2_anchor_thr: float = 0.75    # m/s^2; re-anchor R_ref in realistic quasi-steady
                                               # flight (0.3 required near-perfect equilibrium --
                                               # in flight essentially never -> R_ref free-ran on
                                               # the gyro indefinitely, positive feedback).
    accel_ref_max_freerun_s: float = 1.0       # R_ref free-run TIME BOUND: a reference not
                                               # re-anchored within 1 s is itself gyro-drifted and
                                               # no longer evidence -> force re-anchor to the live
                                               # estimate ("no permanent grudge").
    # (3) THE RECOVERY GUARANTEE -- the single load-bearing line of A32: whenever |a|/g is inside
    # this band (the gravity direction is meaningful well inside the free-fall skip band), the
    # TOTAL effective down-weighting of the accel tilt update ((1/gate) * motion_inflate * huber)
    # is capped. At the cap, R_eff = (accel_noise_std/g)^2 * 100 ~= 9.4e-2 -> per-tick gain
    # K ~= P/(P+R) ~= 0.096 even at the 1e-2 P seed -> error time-constant ~0.6 s at 18 Hz:
    # "down" recovers within ~1 s of ANY spike no matter what the trust heuristics believe, while
    # 100x still keeps honest high-g contamination to <1% influence per tick.
    accel_trust_v2_deweight_cap: float = 100.0
    accel_trust_v2_g_band_lo: float = 0.5      # |a|/g band where the cap (and the watchdog) engage
    accel_trust_v2_g_band_hi: float = 1.5
    # (4) GRAVITY-RECOVERY WATCHDOG (PX4/ArduPilot attitude-reset pattern; belt-and-suspenders over
    # the cap): theta_g = angle(measured specific-force dir, predicted specific-force dir) tracked
    # each in-band tick. A genuine maneuver cannot hold a STEADY >25 deg gravity residual with
    # |a| ~ g for 0.5 s -- that is a wrong attitude, so confess ignorance (P bump, NOT a hard set;
    # the very next ordinary accel updates then close the error in 2-3 ticks at K~0.84) and
    # re-anchor the motion-reject reference to the accel-implied level (current yaw kept -- yaw is
    # unobservable from gravity and the vision yaw path is untouched).
    accel_wd_theta_deg: float = 25.0           # sustained disagreement threshold
    accel_wd_hold_s: float = 0.5               # persistence required before firing
    accel_wd_p_bump_deg: float = 20.0          # attitude-covariance confession (1-sigma, per axis)
    accel_wd_cooldown_s: float = 1.0           # min time between fires
    mag_ned: Optional[np.ndarray] = None
    mag_noise_std: float = 0.1          # normalised

    # Internal state (post-init)
    _q: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    _b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _P: np.ndarray = field(default_factory=lambda: np.eye(6) * 1e-2)
    # Gyro-anchored reference rotation (body->world) for the acceleration-aware reject. Propagated by
    # the bias-corrected gyro only; re-anchored to the estimate at equilibrium. Identity until used.
    _R_ref: np.ndarray = field(default_factory=lambda: np.eye(3))
    # -- A32 v2 internal state (only ever advanced when use_accel_trust_v2; inert otherwise) --
    _ref_freerun_s: float = field(default=0.0, repr=False)     # time since R_ref last re-anchored
    _elapsed_s: float = field(default=0.0, repr=False)         # internal elapsed clock (sum of dt)
    _last_dt: float = field(default=0.0, repr=False)           # dt of the current step (watchdog)
    _wd_theta_s: float = field(default=0.0, repr=False)        # consecutive in-band theta_g>thr time
    _wd_cooldown_until_s: float = field(default=-1.0, repr=False)
    # -- A32 instrumentation (read by the nav-estimate logger; NEVER fed back into the filter).
    # Written only on the v2 path -> null/NaN off-path, zero numeric impact either way. --
    last_theta_g_deg: float = field(default=float("nan"), repr=False)
    last_deweight_total: float = field(default=float("nan"), repr=False)
    watchdog_fires: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._q = np.array([1., 0., 0., 0.], dtype=np.float64)
        self._b_g = np.zeros(3, dtype=np.float64)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)
        self._R_ref = _quat_to_R_wxyz(self._q)
        if self.mag_ned is not None:
            self.mag_ned = np.asarray(self.mag_ned, dtype=np.float64)

    # -- Public interface -------------------------------------------------------

    @property
    def q_wxyz(self) -> np.ndarray:
        """Current attitude estimate as (w,x,y,z) quaternion."""
        return self._q.copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        """Current gyro bias estimate (rad/s, body FRD)."""
        return self._b_g.copy()

    @property
    def attitude_uncertainty_rad(self) -> float:
        """1-sigma attitude uncertainty: sqrt(trace(P[:3,:3]) / 3) in radians."""
        return float(np.sqrt(np.trace(self._P[:3, :3]) / 3.0))

    @property
    def yaw_uncertainty_rad(self) -> float:
        """1-sigma uncertainty of the WORLD-yaw (about-gravity) attitude error, in radians.

        VQ2 has no magnetometer, so the accel update is yaw-blind (``H = -skew(g_hat)`` has a
        zero column on the gravity/yaw axis, eskf.py:301) and the yaw-axis attitude error grows
        UNBOUNDED on gyro-z bias until a vision yaw pseudo-measurement (``update_yaw``) lands.
        This reports that yaw-axis 1-sigma SEPARATELY from the roll/pitch (``attitude_uncertainty_rad``
        averages all three and hides the runaway): project the body-frame attitude covariance
        ``P[:3,:3]`` through the SAME exact yaw Jacobian ``update_yaw`` uses, giving
        ``sqrt(h P_phi h^T)``. Useful for gating / for a confidence channel that must know when yaw is
        unobserved."""
        h = _yaw_jacobian_dphi(_quat_to_R_wxyz(self._q))   # (3,) world-yaw sensitivity to body delta_phi
        return float(np.sqrt(max(h @ self._P[:3, :3] @ h, 0.0)))

    def reset(self, q_init: Optional[np.ndarray] = None) -> None:
        """Reset to initial state. q_init: (w,x,y,z) or None for identity."""
        self._q = _normalize_quat(
            np.asarray(q_init, dtype=np.float64) if q_init is not None
            else np.array([1., 0., 0., 0.])
        )
        self._b_g = np.zeros(3)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)
        self._R_ref = _quat_to_R_wxyz(self._q)
        # A32 v2 state resets with the filter (inert unless use_accel_trust_v2).
        self._ref_freerun_s = 0.0
        self._elapsed_s = 0.0
        self._last_dt = 0.0
        self._wd_theta_s = 0.0
        self._wd_cooldown_until_s = -1.0
        self.last_theta_g_deg = float("nan")
        self.last_deweight_total = float("nan")
        self.watchdog_fires = 0

    def step(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float,
        mag_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process one IMU sample, return updated attitude quaternion (w,x,y,z).

        Parameters
        ----------
        gyro : (3,) body-frame angular rate (rad/s, FRD).
        accel : (3,) body-frame specific force (m/s^2, FRD). Rest = [0,0,-g].
        dt : timestep (seconds).
        mag_body : (3,) body-frame magnetometer (optional). Normalised before use.
        """
        if dt <= 0:
            return self._q.copy()

        self._predict(np.asarray(gyro, dtype=np.float64), dt)
        self._update_accel(np.asarray(accel, dtype=np.float64))

        if mag_body is not None and self.mag_ned is not None:
            self._update_mag(np.asarray(mag_body, dtype=np.float64))

        return self._q.copy()

    # -- Prediction ------------------------------------------------------------

    def _predict(self, gyro: np.ndarray, dt: float) -> None:
        """Propagate nominal quaternion and error-state covariance."""
        omega_corrected = gyro - self._b_g

        # Propagate nominal quaternion
        dq = _omega_exp_wxyz(omega_corrected, dt)
        self._q = _normalize_quat(_quat_multiply_wxyz(self._q, dq))

        # Propagate the GYRO-ANCHORED reference by the SAME bias-corrected rotation increment (gyro
        # ONLY -- never accel-corrected). Cheap, and only consulted when use_accel_motion_reject is on.
        # It is re-anchored to the live estimate at equilibrium inside _accel_motion_inflation.
        if self.use_accel_motion_reject:
            self._R_ref = self._R_ref @ _quat_to_R_wxyz(dq)
        # A32 v2 clocks: the R_ref free-run bound + the watchdog persistence/cooldown timers all
        # need real elapsed time. Advanced ONLY under the flag (OFF path byte-identical).
        if self.use_accel_trust_v2:
            self._ref_freerun_s += dt
            self._elapsed_s += dt
            self._last_dt = dt

        # Linearised error-state transition
        #   delta_phi_{k+1} = (I - [omega_corr] * dt) * delta_phi - dt * delta_b_g
        #   delta_b_g_{k+1} = delta_b_g   (random walk only)
        F = np.eye(6)
        F[:3, :3] = np.eye(3) - _skew(omega_corrected) * dt
        F[:3, 3:] = -np.eye(3) * dt
        # F[3:, 3:] = I (bias walk through identity)

        # Process noise Q
        sigma_phi = self.gyro_noise_std * np.sqrt(dt)
        sigma_bg = self.gyro_bias_std * np.sqrt(dt)
        Q = np.diag([sigma_phi**2]*3 + [sigma_bg**2]*3)

        self._P = F @ self._P @ F.T + Q

    # -- Accelerometer update --------------------------------------------------

    def _accel_gate_weight(self, accel_mag: float) -> float:
        """Weight in [0, 1]: 1 at |a|=g, decays as |a| deviates (high-g gating).

        When accel_gate_alpha = 0, returns 1.0 always (no gating = naive filter).
        """
        if self.accel_gate_alpha <= 0.0:
            return 1.0
        deviation = (accel_mag - GRAVITY) / GRAVITY
        return float(np.exp(-self.accel_gate_alpha * deviation**2))

    def _accel_magnitude_in_band(self, accel_mag: float) -> bool:
        """Free-fall / high-|a| magnitude guard: True iff |a| ~ g (accel IS the gravity reference).

        The accelerometer only equals the gravity reaction when ``|a| ~ g``. In FREE-FALL
        (|a| -> ~0) the normalised accel direction is meaningless and feeding it to the tilt
        update can INVERT the attitude estimate; in extreme high-g (|a| >> g) the accel is
        kinematic, not gravity. Either way the accel update must be SKIPPED. Returns True only
        when ``g*(1-tol_lo) <= |a| <= g*(1+tol_hi)`` (the regime where the accel carries tilt).
        A negative tolerance disables that side. When |a| ~ g this is True (the band is a no-op
        and the accel update proceeds byte-identically to before).
        """
        if self.accel_freefall_tol_lo >= 0.0:
            lo = GRAVITY * (1.0 - self.accel_freefall_tol_lo)
            if accel_mag < lo:
                return False
        if self.accel_freefall_tol_hi >= 0.0:
            hi = GRAVITY * (1.0 + self.accel_freefall_tol_hi)
            if accel_mag > hi:
                return False
        return True

    def _accel_motion_inflation(self, accel: np.ndarray) -> float:
        """Covariance-inflation factor (>=1) for the acceleration-aware accel rejection.

        The linear-acceleration residual w.r.t. the GYRO-ANCHORED reference (NOT the live estimate):
            a_lin_world = R_ref @ f_body + g_ned     (== true world kinematic accel, ~0 at equilibrium)
        At hover/coast |a_lin| ~ 0 (the accel IS gravity -> trust it, factor 1.0); under sustained
        powered tilted flight with the gyro flat (the A8 signature) the accel direction tilts while
        R_ref holds, so |a_lin| grows (the accel is contaminated by LINEAR accel, not rotation ->
        distrust it). Returns ``1 + (|a_lin| / accel_motion_scale)^2`` clamped to accel_motion_max_inflate.
        Returns 1.0 (no-op, byte-identical) when the gate is disabled.

        R_ref must anchor to the GYRO, not the estimate: the accel slowly drags R_hat to follow the
        tilting specific force, hiding the contamination if measured against R_hat (the A8 divergence is
        then invisible). When |a_lin| is small (the trusted regime) R_ref is RE-ANCHORED to the live
        estimate so it does not accumulate gyro-bias drift; it only free-runs on the gyro across a
        suspect powered window. g_ned = [0,0,+g] adds the gravity reaction back."""
        if not self.use_accel_motion_reject:
            return 1.0
        # A32 v2 (spec §3.1(1)): TIME-LIMIT the gyro-anchored free-run. A reference that has not
        # re-anchored within accel_ref_max_freerun_s is itself gyro-drifted and no longer evidence
        # of accel contamination -- force re-anchor to the live estimate BEFORE computing the
        # residual, so a wrong reference can never hold a grudge past the bound.
        if self.use_accel_trust_v2 and self._ref_freerun_s > self.accel_ref_max_freerun_s:
            self._R_ref = _quat_to_R_wxyz(self._q)
            self._ref_freerun_s = 0.0
        a_lin = self._R_ref @ accel + G_NED          # kinematic accel implied by the GYRO reference
        a_lin_mag = float(np.linalg.norm(a_lin))
        # Re-anchor the reference to the trusted estimate while at/near equilibrium (keeps the gyro
        # reference from drifting on bias when there is nothing to reject). v2 widens the anchor
        # threshold (0.3 -> 0.75 m/s^2: realistic quasi-steady flight re-anchors; 0.3 required
        # near-perfect equilibrium and in flight essentially never fired -> indefinite free-run).
        anchor_thr = (self.accel_trust_v2_anchor_thr if self.use_accel_trust_v2
                      else self.accel_motion_anchor_thr)
        if a_lin_mag < anchor_thr:
            self._R_ref = _quat_to_R_wxyz(self._q)
            if self.use_accel_trust_v2:
                self._ref_freerun_s = 0.0
        if self.use_accel_trust_v2:
            # BOUNDED inflation (spec §3.1(1)): knee 1.0 m/s^2 (real maneuvers inflate 5-10x, not
            # 400-1600x) and a 25x hard cap -- the gravity pull can be slowed, never severed.
            scale = max(self.accel_trust_v2_motion_scale, 1e-9)
            factor = 1.0 + (a_lin_mag / scale) ** 2
            return float(min(factor, self.accel_trust_v2_max_inflate))
        scale = max(self.accel_motion_scale, 1e-9)
        factor = 1.0 + (a_lin_mag / scale) ** 2
        return float(min(factor, self.accel_motion_max_inflate))

    def _update_accel(self, accel: np.ndarray) -> None:
        """Tilt update using accelerometer; gated by high-g weighting.

        Sign convention (critical):
          IMU specific force at rest (body FRD):  sf_body = [0, 0, -g]  (pointing UP = reaction)
          Predicted specific force from attitude:  h_body = R_wb^T @ (-G_NED) = [0, 0, -g] at rest
          so h_body = -R_wb^T @ G_NED = -(gravity in body frame).
          The measurement model is:  normalised(accel) = h_body / g = [0, 0, -1] at rest.
          Jacobian: d(h_body)/d(delta_phi) = d(-R^T G_NED)/d(delta_phi) = -skew(g_body)
          where g_body = R_wb^T @ G_NED (gravity direction in body = [0,0,+g] at rest).
        """
        accel_mag = float(np.linalg.norm(accel))
        if accel_mag < 1e-6:
            return

        # Free-fall / high-|a| magnitude guard (catastrophic-failure safety net): if |a| is
        # outside the gravity-magnitude band, the accel is NOT the gravity reference, so SKIP
        # the accel correction entirely (we already gyro-propagated in _predict). This is an
        # earlier, unconditional reject than the chi2 gate, specifically for the free-fall case
        # that can invert the attitude estimate. When |a| ~ g this is a no-op (returns in-band).
        if not self._accel_magnitude_in_band(accel_mag):
            return

        gate = self._accel_gate_weight(accel_mag)

        # Skip update if gate weight is negligible (high-g): avoids numerical noise
        if gate < 1e-4:
            return

        # Gravity DIRECTION in body frame: g_hat = R^T @ G_NED / g = [0,0,+1] at rest.
        # We work entirely in NORMALISED (unit-vector) measurement space so the
        # innovation and the Jacobian share the same units -- mixing a unit-vector
        # innovation with a g-scaled (~9.8x) Jacobian miscalibrates the Kalman gain
        # and was the root cause of the accel update *corrupting* attitude.
        R_wb = _quat_to_R_wxyz(self._q)
        g_body = R_wb.T @ G_NED                      # [0, 0, +g] at rest
        g_hat = g_body / max(np.linalg.norm(g_body), 1e-9)   # [0, 0, +1] at rest

        # Predicted specific-force direction: h_hat = -g_hat = [0,0,-1] at rest.
        h_hat = -g_hat

        # Normalised accel measurement: a_hat = [0,0,-1] at rest (matches h_hat).
        a_hat = accel / accel_mag

        # Innovation: z - h(x)
        innovation = a_hat - h_hat

        # Jacobian H (3 x 6): d(h_hat)/d(delta_phi). With the right-multiplicative
        # body-frame error convention used at injection (q <- q * exp(delta_phi)),
        # h_hat = -R^T G_NED/g and a finite-difference check (see scratch) gives
        #   d(h_hat)/d(delta_phi) = -skew(g_hat).
        # NOTE the normalisation by g: g_hat is the UNIT gravity direction, matching
        # the unit innovation above.
        H = np.zeros((3, 6))
        H[:, :3] = -_skew(g_hat)

        # Effective measurement noise: R_meas / gate (larger noise when gate small = high-g),
        # then FURTHER inflated by the acceleration-aware motion factor (A8 sustained-accel fix):
        # |a_lin| ~ 0 -> factor 1.0 (byte-identical); powered tilted flight -> factor >> 1 (the accel
        # leveling is starved before it can drag the attitude off true). Applied BEFORE the chi2 gate
        # so the gate's S reflects the down-weighted trust too.
        sigma_a = self.accel_noise_std / GRAVITY  # normalised units

        if self.use_accel_trust_v2:
            # -------------------- A32 v2: bounded-trust + always-find-down --------------------
            # Everything composes into ONE total deweight factor on R (the pattern to copy, one
            # line: R_eff = R_base / w(nu), w continuous, w > 0 everywhere; hard consequences
            # only on PERSISTENCE -- the watchdog).
            deweight = self._accel_motion_inflation(accel) / gate   # (1/gate) * motion_inflate
            # (2) chi2 hard gate -> Huber-soft (Karlgaard & Schaub one-pass IRLS): a measurement
            # at the consistency boundary passes untouched, a 10x outlier is ~10x-damped, NOTHING
            # is discarded, and readmission is instant when consistency returns (kills F4's
            # minutes-long lockout: P grows at only gyro_noise_std^2 = 1e-4 rad^2/s).
            if self.accel_chi2_thresh > 0.0:
                S = H @ self._P @ H.T + (sigma_a**2 * deweight) * np.eye(3)
                try:
                    md = float(innovation @ np.linalg.solve(S, innovation))
                except np.linalg.LinAlgError:
                    return
                if md > self.accel_chi2_thresh:
                    deweight *= md / self.accel_chi2_thresh
            # (3) THE RECOVERY GUARANTEE: whenever |a| ~ g the accel IS an unbiased absolute
            # reference for "down" -- cap the TOTAL deweight so the gravity pull is never severed
            # (structural ~0.6 s recovery time-constant; the load-bearing line of A32).
            in_band = (self.accel_trust_v2_g_band_lo * GRAVITY <= accel_mag
                       <= self.accel_trust_v2_g_band_hi * GRAVITY)
            if in_band:
                deweight = min(deweight, self.accel_trust_v2_deweight_cap)
            self.last_deweight_total = float(deweight)   # instrumentation (nav_estimate logger)
            # (4) gravity-recovery watchdog: theta_g = angle(measured specific-force dir,
            # predicted specific-force dir). Persistent gross disagreement while in-band =>
            # attitude reset (P bump + R_ref re-anchor). Runs BEFORE the update so the bumped P
            # makes THIS in-band accel sample immediately authoritative (K ~ 0.84).
            self._accel_trust_v2_watchdog(a_hat, h_hat, accel_mag, in_band)
            R_meas = (sigma_a**2) * deweight * np.eye(3)
            self._apply_eskf_update(innovation, H, R_meas)
            return

        R_meas = (sigma_a**2 / gate) * self._accel_motion_inflation(accel) * np.eye(3)

        # Direction-aware innovation gate (Mahalanobis / chi-square consistency test).
        # Magnitude gating cannot reject a near-g-magnitude disturbance whose DIRECTION
        # is wrong; this gate can. Skip the update when the innovation is inconsistent
        # with its predicted covariance S.
        if self.accel_chi2_thresh > 0.0:
            S = H @ self._P @ H.T + R_meas
            try:
                md = float(innovation @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:
                return
            if md > self.accel_chi2_thresh:
                return  # reject: accel inconsistent with attitude prior (high-g disturbance)

        # Kalman update on error state
        self._apply_eskf_update(innovation, H, R_meas)

    # -- A32 gravity-recovery watchdog (v2 only) --------------------------------

    def _accel_trust_v2_watchdog(self, a_hat: np.ndarray, h_hat: np.ndarray,
                                 accel_mag: float, in_band: bool) -> None:
        """PX4/ArduPilot-style attitude reset on PERSISTENT gross gravity disagreement (spec
        §3.1(4)). ``theta_g = angle(a_hat, h_hat)`` is the angle between the MEASURED specific-force
        direction and the attitude-PREDICTED one (== angle(a_hat, -g_hat_pred)); with |a| ~ g a
        genuine maneuver cannot hold a steady >``accel_wd_theta_deg`` residual for
        ``accel_wd_hold_s`` -- that is a wrong attitude estimate. FIRE = confess attitude ignorance
        (``P[:3,:3] += (accel_wd_p_bump_deg 1-sigma)^2`` -- a covariance bump, NOT a hard state set;
        the next ordinary accel updates close the error in 2-3 ticks) + re-anchor the motion-reject
        reference ``R_ref`` to the accel-implied LEVEL at the CURRENT estimated yaw (yaw untouched:
        unobservable from gravity; the vision yaw path is unchanged), with a
        ``accel_wd_cooldown_s`` refractory. The persistence clock accumulates only on IN-BAND ticks
        (|a| ~ g is precisely when the accel is an honest witness) and resets the moment an in-band
        tick agrees; out-of-band ticks neither accumulate nor reset (the witness is absent, the
        case stays open). Instrumentation: ``last_theta_g_deg`` / ``watchdog_fires``."""
        theta_g = float(np.degrees(np.arccos(np.clip(float(a_hat @ h_hat), -1.0, 1.0))))
        self.last_theta_g_deg = theta_g
        if not in_band:
            return
        if theta_g > self.accel_wd_theta_deg:
            self._wd_theta_s += self._last_dt
        else:
            self._wd_theta_s = 0.0
            return
        if (self._wd_theta_s < self.accel_wd_hold_s
                or self._elapsed_s < self._wd_cooldown_until_s):
            return
        # FIRE: covariance confession on the attitude block ...
        bump = float(np.radians(self.accel_wd_p_bump_deg)) ** 2
        self._P[:3, :3] = self._P[:3, :3] + np.eye(3) * bump
        # ... and re-anchor the gyro reference to the accel-implied level (roll/pitch from the
        # measured gravity direction, yaw from the CURRENT estimate -- gravity says nothing about
        # yaw and the a_lin residual is yaw-insensitive w.r.t. gravity cancellation anyway).
        g_body = -a_hat                                   # gravity direction in body (unit)
        roll_acc = float(np.arctan2(g_body[1], g_body[2]))
        pitch_acc = float(np.arctan2(-g_body[0], float(np.hypot(g_body[1], g_body[2]))))
        yaw_est = float(Rotation.from_matrix(_quat_to_R_wxyz(self._q)).as_euler("ZYX")[0])
        self._R_ref = Rotation.from_euler("ZYX", [yaw_est, pitch_acc, roll_acc]).as_matrix()
        self._ref_freerun_s = 0.0
        self._wd_theta_s = 0.0
        self._wd_cooldown_until_s = self._elapsed_s + self.accel_wd_cooldown_s
        self.watchdog_fires += 1

    # -- Magnetometer update (optional) ----------------------------------------

    def _update_mag(self, mag_body: np.ndarray) -> None:
        """Yaw update from body-frame magnetometer and known NED reference field.

        Only the yaw component is corrected (the mag is not informative about tilt
        unless the full 3D field is used, and we trust the accel for tilt).
        """
        mag_mag = float(np.linalg.norm(mag_body))
        if mag_mag < 1e-6 or np.linalg.norm(self.mag_ned) < 1e-6:
            return

        mag_body_n = mag_body / mag_mag
        mag_ned_n = self.mag_ned / np.linalg.norm(self.mag_ned)

        R_wb = _quat_to_R_wxyz(self._q)
        mag_ned_pred = R_wb @ mag_body_n    # predicted NED direction from body meas

        # Project onto horizontal plane for yaw-only sensitivity
        # yaw error = cross(mag_ned_pred_horiz, mag_ned_ref_horiz) . [0,0,1]
        h_pred = np.array([mag_ned_pred[0], mag_ned_pred[1], 0.0])
        h_ref = np.array([mag_ned_n[0], mag_ned_n[1], 0.0])
        hn_p = np.linalg.norm(h_pred)
        hn_r = np.linalg.norm(h_ref)
        if hn_p < 1e-6 or hn_r < 1e-6:
            return

        h_pred /= hn_p
        h_ref /= hn_r

        # Scalar yaw innovation: z = cross(h_pred, h_ref)[2]
        innovation = np.array([np.cross(h_pred, h_ref)[2]])

        # H: 1x6; only yaw (Z-axis) component of delta_phi contributes
        # d(yaw_meas)/d(delta_phi) ~ [0, 0, 1] (approximately, near correct attitude)
        H = np.zeros((1, 6))
        H[0, 2] = 1.0

        R_meas = np.array([[self.mag_noise_std**2]])
        self._apply_eskf_update(innovation, H, R_meas)

    # -- Vision yaw pseudo-measurement (mag-free VQ2 yaw lock) ------------------

    def update_yaw(self, yaw_meas_world: float, yaw_noise_std: float) -> None:
        """Scalar pseudo-measurement on the WORLD yaw (about gravity) — the mag-free yaw lock.

        VQ2 carries no magnetometer, so the accel tilt update is rank-2 (yaw-blind) and yaw is a
        free integrator on gyro-z bias. This injects an EXTERNAL absolute/relative world-yaw datum
        (from vision: a vanishing-point heading or a gate-bearing yaw) as the missing yaw observer,
        bounding the otherwise-unbounded yaw drift. It is the single shared injection point for ALL
        vision yaw sources; the CALLER disambiguates a mod-90 vanishing-point branch to the current
        estimate BEFORE calling (this method just consumes an unwrapped absolute yaw datum).

        Measurement model. The measured quantity is the aerospace 3-2-1 world yaw
        ``psi = atan2(R_wb[1,0], R_wb[0,0])`` (``euler_from_quat(q)[2]``). Its sensitivity to the ESKF's
        BODY-frame error ``delta_phi`` (injection convention ``q <- q * exp(delta_phi)``, a RIGHT/body
        perturbation ``R' = R_wb @ exp([delta_phi]x)``) is the EXACT atan2 derivative — NOT the naive
        ``(R_wb^T e_z_world)`` shortcut, which is only correct at zero tilt and FIGHTS roll/pitch
        otherwise. With ``dR = R_wb @ [delta_phi]x``:
            d(psi)/d(delta_phi_i) = (R00 * dR[1,0] - R10 * dR[0,0]) / (R00^2 + R10^2)
        Near level ``R_wb -> I`` this reduces to the ``H[0,2]=1`` shortcut the dormant ``_update_mag``
        uses; the full form here is exact at all tilts. **FD-pinned** in tests/test_vision_yaw_wiring.py.

        Innovation is the WRAPPED angle difference (an unwrapped ~2*pi error injects a catastrophic
        correction). ``yaw_noise_std`` is the 1-sigma (rad) of the vision yaw datum (large => weak,
        e.g. a near-head-on gate-bearing yaw with little leverage); a non-positive std is a no-op.

        Gated/additive: nothing in the default ESKF/AHRS path calls this, so it changes NOTHING unless
        a consumer (the case-C Navigator behind ``use_vp_yaw`` / ``use_gate_bearing_yaw``) invokes it.
        """
        if not np.isfinite(yaw_meas_world) or yaw_noise_std <= 0.0:
            return
        R_wb = _quat_to_R_wxyz(self._q)
        yaw_hat = float(Rotation.from_matrix(R_wb).as_euler("ZYX")[0])
        # Wrapped scalar innovation (-pi, pi].
        innovation = np.array([_wrap_angle(float(yaw_meas_world) - yaw_hat)])
        # H (1 x 6): EXACT atan2-yaw sensitivity to the body-frame delta_phi; zero on the bias block.
        H = np.zeros((1, 6))
        H[0, :3] = _yaw_jacobian_dphi(R_wb)
        if not np.any(H[0, :3]):           # gimbal-locked (|pitch|->90 deg): yaw undefined, skip
            return
        R_meas = np.array([[float(yaw_noise_std) ** 2]])
        self._apply_eskf_update(innovation, H, R_meas)

    # -- Core ESKF update ------------------------------------------------------

    def _apply_eskf_update(
        self,
        innovation: np.ndarray,
        H: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """Apply a generic ESKF correction, then reset error state into nominal."""
        PHt = self._P @ H.T
        S = H @ PHt + R_meas
        try:
            K = np.linalg.solve(S, PHt.T).T   # same solve trick as LinearKF
        except np.linalg.LinAlgError:
            return

        delta_x = K @ innovation               # (6,)
        delta_phi = delta_x[:3]
        delta_b_g = delta_x[3:]

        # Inject attitude correction into nominal quaternion
        # delta_phi is a small rotation vector in the body frame
        angle = np.linalg.norm(delta_phi)
        if angle > 1e-12:
            dq = _omega_exp_wxyz(delta_phi, 1.0)   # exp(delta_phi / 2) at unit dt
        else:
            dq = np.array([1., 0., 0., 0.])
        self._q = _normalize_quat(_quat_multiply_wxyz(self._q, dq))

        # Inject bias correction
        self._b_g = self._b_g + delta_b_g

        # Joseph-form covariance update (symmetric + PD)
        I_KH = np.eye(6) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_meas @ K.T
