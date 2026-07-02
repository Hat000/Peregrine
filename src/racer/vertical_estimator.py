"""VerticalEstimator — the 1-D vertical-velocity WASHOUT filter for the state-denied VQ2 wire
(the A24 washout fix, replacing the A21 floor-pin 3-state KF, 2026-07-02).

THE BUG THIS KILLS (run 20260702_040036, quantified via scripts/analyze_flight.py): the wire is
POSITION/VELOCITY DENIED, so the 6-state KF's z is dead-reckoning between the sparse (~a few Hz,
latent) vision ``floor_height`` pins — and each accepted pin is a TIGHT z fix, so ``est_z``
TELEPORTS (−1.435 m in ONE tick at t=1569.41) while the true motion is smooth. The controller's
ff-owns-vertical damper reads a low-passed finite difference of that z: the teleport becomes a
~−4.4 m/s vz spike -> a thrust slam -> and between pins the damper is BLIND to the real climb
(integrated a_up shows +4-5 m/s upward over 2.7 s that est_z never tracked) -> the drone climbs
over gate 0 into the ceiling. Smooth true motion vs a step-function estimate = the bug.

THE A21 FIX (superseded) tried a tiny 3-state [z, vz, bias] KF, correcting the slow drift with the
absolute floor_height pins. THIS WAS A FALSE PREMISE (2026-07-02 spec review): the warehouse has no
floor grid, so the "correction" was never a real external fix — a KF with a bias state and NO
measurements is just a washout in a Kalman costume, and the bias is fundamentally unobservable on
this wire. The vertical channel is gate-RELATIVE; absolute height is neither obtainable nor needed.

THE A24 FIX is a first-order washout (leaky integrator), NOT a Kalman filter. It integrates the
IMU-derived kinematic upward acceleration ``a_up`` (fast ~140 Hz, smooth, but drifting) with an
exponential leak (time constant ``washout_tau_s``) so the state is STRUCTURALLY bounded: it cannot
diverge regardless of the input, because a sustained bias ``b`` only pulls it to a constant offset
``tau*b``, never a ramp. A short pre-arm window captures the accelerometer/attitude-projection bias
once (grounded, at rest) so residual pitch-projection error does not masquerade as vertical motion.

    vz <- exp(-dt/tau)*vz + (a_dn_clamped - b_hat)*dt      # a_dn = -a_up (NED down-positive)

Deliberately NOT fused with any vision measurement in v1 (gate-offset-rate fusion is a documented,
default-OFF v2 hook — see the spec, not implemented here): the only candidate measurement (floor
pins) is gone, and the gate-relative offset-rate the outer loop already tracks would add only noisy,
low-frequency information at v1's confidence level.

ACCEL DECODE (VERIFIED empirically 2026-07-01, docs/accel_investigation.md + analyze_flight.py):
HIGHRES_IMU accel is SPECIFIC FORCE in body FRD, gravity-INCLUDED (on-pad |f| = 9.810 exactly),
so the kinematic upward acceleration is  a_up = -( (R_body->ned @ f)_z + g ).

Consumed by the Navigator behind ``NavigatorConfig.use_vertical_estimator`` (default OFF =
byte-identical); exported on ``NavState.vert_vz_est`` (``vert_z_est`` is permanently NaN — no
absolute-altitude state remains in this filter); the controller's ff-owns-vertical alt-hold damps
on the exported vz behind ``Controller.use_vertical_estimator``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_G = 9.80665


def a_up_from_specific_force(f_body: np.ndarray, R_wb: np.ndarray) -> float:
    """Kinematic UPWARD acceleration (m/s^2) from a specific-force sample + the body->NED rotation.

    ``a_ned = R_wb @ f + g_ned`` with ``g_ned = (0, 0, +G)`` (NED z is DOWN), so ``a_up = -a_ned_z``.
    The matrix form of the VERIFIED analyze_flight.py decode — the Navigator has the TRUE R_wb
    (AHRS) in hand, so no euler round-trip. At rest (attitude correct) this reads ~0.
    """
    a_ned_z = float(np.asarray(R_wb, dtype=np.float64)[2] @ np.asarray(f_body, dtype=np.float64)) + _G
    return -a_ned_z


@dataclass
class VerticalEstimator:
    """First-order washout (leaky integrator) vertical-velocity channel: ``(vz, b_hat)`` state.

    Call ``seed()`` once (on the pad, at rest), then ``predict(a_up, dt)`` per IMU tick. Read
    ``vz`` (NED down +, clipped to +/-3 m/s). Un-seeded, every call no-ops and ``vz`` is NaN — the
    consumer's absent-marker. There is no altitude state: ``z`` is permanently NaN (see the ``z``
    property) — this filter deliberately carries NO absolute-height estimate.
    """

    # Washout time constant (s): vz <- exp(-dt/tau)*vz + (a_dn-b_hat)*dt. Bounds the state by
    # tau*sup|a_dn-b_hat| regardless of input -- the structural cannot-diverge guarantee. Band
    # 1.5-3.0s; 2.0 keeps the outer gate-align loop well damped (zeta~=0.62 at Kd=0.25) while a
    # sustained bias residual (tau*b) stays small.
    washout_tau_s: float = 2.0
    # Reject implausibly large predict steps (sim reset / stutter) -- mirrors LinearKF.max_dt_s.
    max_dt_s: float = 0.2
    # A_UP INPUT OUTLIER CLAMP (COMMANDER ADDITION, 2026-07-02): clamp a_dn to +/-this (m/s^2)
    # BEFORE the washout update. A real slow-lap drone cannot sustain >3g vertical; this rejects
    # contact-impact spikes AND sensor glitches at the SOURCE, not just at the output clip.
    # Empirically material on the contaminated replay: full-flight rail 81%->49%, in-band 19%->52%.
    a_up_clamp_mps2: float = 30.0
    # Export clip (m/s), applied on read: the ABSOLUTE, unconditional cannot-diverge guarantee
    # regardless of what garbage the input stream carries.
    export_clip_mps: float = 3.0
    # Pre-arm bias capture window (s): the first this-many seconds after seed (grounded -- true
    # vz=0), a running mean of a_dn (post input-clamp) is captured into b_hat, then frozen.
    bias_capture_s: float = 1.0
    # Captured bias clip (m/s^2): the frozen b_hat is clamped to +/- this after capture -- a sane
    # ceiling on a real attitude-projection residual (the AHRS is cold at arm, but not THAT cold).
    bias_clip_mps2: float = 0.5

    _vz: float = field(default=float("nan"), repr=False)
    _b_hat: float = field(default=float("nan"), repr=False)
    _seeded: bool = field(default=False, repr=False)
    # Pre-arm bias-capture running state: elapsed time since seed + the running mean accumulator.
    _bias_capture_done: bool = field(default=False, repr=False)
    _bias_capture_elapsed_s: float = field(default=0.0, repr=False)
    _bias_capture_sum: float = field(default=0.0, repr=False)
    _bias_capture_n: int = field(default=0, repr=False)

    # -- lifecycle ------------------------------------------------------------
    def seed(self) -> None:
        """(Re)initialise at rest: vz=0, b_hat=0, and re-arm the pre-arm bias-capture window."""
        self._vz = 0.0
        self._b_hat = 0.0
        self._seeded = True
        self._bias_capture_done = False
        self._bias_capture_elapsed_s = 0.0
        self._bias_capture_sum = 0.0
        self._bias_capture_n = 0

    @property
    def seeded(self) -> bool:
        return self._seeded

    @property
    def z(self) -> float:
        """Permanently NaN -- this filter carries NO absolute-altitude state (A24 washout rip-out
        of the floor-pin z-correction path; the vertical channel is gate-relative only)."""
        return float("nan")

    @property
    def vz(self) -> float:
        """Estimated NED world-down velocity zdot (m/s), clipped to +/-``export_clip_mps``; NaN
        until seeded. The structural cannot-diverge guarantee: absolute, regardless of input."""
        if not self._seeded:
            return float("nan")
        return float(np.clip(self._vz, -self.export_clip_mps, self.export_clip_mps))

    # -- per-IMU-tick propagation ----------------------------------------------
    def predict(self, a_up: float, dt: float) -> None:
        """Integrate one accel sample with an exponential leak: ``a_dn = -a_up`` (a_up UP-positive,
        NED down-positive), clamped to +/-``a_up_clamp_mps2`` BEFORE the update (rejects contact-
        impact spikes / sensor glitches at the source). During the pre-arm bias-capture window
        (the first ``bias_capture_s`` seconds after seed), accumulate a running mean of the
        clamped a_dn into ``b_hat`` instead of advancing vz (grounded -- true vz is 0); once the
        window closes, b_hat freezes (clipped to +/-``bias_clip_mps2``) and the washout recurrence
        runs. ``dt <= 0`` (a between-IMU control tick) and ``dt > max_dt_s`` (sim reset / stutter --
        integrating a garbage step corrupts vz worse than skipping one) are no-ops, mirroring
        LinearKF."""
        if not self._seeded or not (0.0 < dt <= self.max_dt_s):
            return
        a_dn = float(np.clip(-float(a_up), -self.a_up_clamp_mps2, self.a_up_clamp_mps2))

        if not self._bias_capture_done:
            self._bias_capture_sum += a_dn
            self._bias_capture_n += 1
            self._bias_capture_elapsed_s += dt
            if self._bias_capture_elapsed_s >= self.bias_capture_s:
                mean_a_dn = self._bias_capture_sum / max(self._bias_capture_n, 1)
                self._b_hat = float(np.clip(mean_a_dn, -self.bias_clip_mps2, self.bias_clip_mps2))
                self._bias_capture_done = True
            return   # grounded during capture: vz stays 0, no washout advance yet

        alpha = float(np.exp(-dt / self.washout_tau_s))
        self._vz = alpha * self._vz + (a_dn - self._b_hat) * dt
