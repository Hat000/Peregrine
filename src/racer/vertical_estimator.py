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

THE A25 FIX (this file, 2026-07-02, extends A24 -- see
handoff/vq2_gate_relative_altitude_spec_2026-07-02.md): adds a GATE-RELATIVE vertical-offset state
``z_off`` (``z-hat_off`` in the spec prose) alongside the washout ``vz``. The washout alone has no
POSITION reference -- it can null a velocity but cannot seek a gate-relative height, and the
diagnosis (run 20260702_203428) showed the drone climbing monotonically into gate 0's ceiling
because nothing ever told the loop "you are above the gate, come down". ``z_off`` is latched from
the seeker's fresh pose (``offset_z_world``, world-NED, DOWN-positive: positive = gate opening is
BELOW the drone), propagated at IMU rate by the SAME washout ``vz`` used above (dead-reckoning the
gate-relative height between poses), latency-compensated for the pose's capture-to-use age,
clamped to +/-3 m, and FROZEN (along with ``vz``) during a contact event (a specific-force spike --
see ``contact_frozen``) so ceiling-scrape impacts never masquerade as fictional descent. The
controller's alt-hold then adds a proportional term ``-kp_gate*z_off`` (NEGATIVE: above-gate,
z_off>0, must REDUCE thrust to sink) -- see ``racer.controller._ff_owns_vertical_thrust``. Default
OFF (``Controller.kp_gate=0.0``, ``z_off`` never latched until a pose arrives) -- VQ1/case-A
byte-identical.
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

    # --- A25 gate-relative vertical offset (z_off) ---------------------------------------------
    # Clamp on the ẑ_off state (m): the gate-relative vertical offset is physically bounded on a
    # slow lap; this is the structural cannot-run-away guard mirroring ``export_clip_mps`` for vz.
    z_off_clip_m: float = 3.0
    # CONTACT-GATE (A25 §4, LOAD-BEARING): a |specific-force| spike above this (m/s^2, ~2g of
    # kinematic vertical accel a slow-lap drone cannot sustain) marks a contact tick (impact /
    # ceiling scrape). On a contact tick -- and for ``contact_hold_s`` after the LAST spike (a
    # refractory hold; real contact events are multi-tick) -- BOTH the vz washout update AND the
    # z_off propagate are SKIPPED (held), not merely clamped: a clamped-but-still-huge spike would
    # still inject a large Δvz. This is what stops corrupted post-impact accel from being read as
    # a fictional descent (the diagnosis's 8.4 s ceiling-lock).
    a_contact_mps2: float = 20.0
    contact_hold_s: float = 0.25

    _vz: float = field(default=float("nan"), repr=False)
    _b_hat: float = field(default=float("nan"), repr=False)
    _seeded: bool = field(default=False, repr=False)
    # Pre-arm bias-capture running state: elapsed time since seed + the running mean accumulator.
    _bias_capture_done: bool = field(default=False, repr=False)
    _bias_capture_elapsed_s: float = field(default=0.0, repr=False)
    _bias_capture_sum: float = field(default=0.0, repr=False)
    _bias_capture_n: int = field(default=0, repr=False)
    # -- A25 z_off state --
    _z_off: float = field(default=float("nan"), repr=False)
    _z_off_seen: bool = field(default=False, repr=False)
    # -- A25 contact-gate state: sim-time-like elapsed clock (seconds), monotonic via predict's dt --
    _contact_elapsed_s: float = field(default=0.0, repr=False)
    _contact_until_s: float = field(default=-1.0, repr=False)

    # -- lifecycle ------------------------------------------------------------
    def seed(self) -> None:
        """(Re)initialise at rest: vz=0, b_hat=0, and re-arm the pre-arm bias-capture window.
        Also resets the A25 z_off state (un-seen, NaN) and the contact-gate clock -- a re-seed
        (sim epoch restart) must forget any latched gate-relative offset from the prior epoch."""
        self._vz = 0.0
        self._b_hat = 0.0
        self._seeded = True
        self._bias_capture_done = False
        self._bias_capture_elapsed_s = 0.0
        self._bias_capture_sum = 0.0
        self._bias_capture_n = 0
        self._z_off = float("nan")
        self._z_off_seen = False
        self._contact_elapsed_s = 0.0
        self._contact_until_s = -1.0

    @property
    def seeded(self) -> bool:
        return self._seeded

    @property
    def z(self) -> float:
        """Permanently NaN -- this filter carries NO absolute-altitude state (A24 washout rip-out
        of the floor-pin z-correction path; the vertical channel is gate-relative only)."""
        return float("nan")

    @property
    def z_off(self) -> float:
        """Gate-relative vertical offset ẑ_off (m, NED down-positive: drone→gate; positive = drone
        ABOVE the gate). NaN until the first :meth:`latch_offset` call (no gate ever seen) -- the
        consumer's absent-marker, mirrored exactly on ``NavState.z_off_est`` (A25)."""
        if not self._z_off_seen:
            return float("nan")
        return float(np.clip(self._z_off, -self.z_off_clip_m, self.z_off_clip_m))

    def contact_frozen(self) -> bool:
        """True while a contact event (or its refractory hold) is active -- the vz washout update
        AND the z_off propagate must both be SKIPPED this tick (A25 §4)."""
        return self._contact_elapsed_s < self._contact_until_s

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
        LinearKF.

        A25: AFTER the vz washout update (same tick, same dt), propagate the gate-relative offset
        ``z_off`` by the just-updated vz: ``z_off <- z_off - vz*dt`` (vz is NED down-positive; a
        real CLIMB reads vz<0 in that convention, so ``-vz*dt`` is positive on a climb -- the gate
        gets further below as the drone rises, growing z_off MORE positive; see the spec §1.3.1 and
        the sign-pinning test ``test_propagate_sign_climb_grows_z_off_positive``). Also runs the
        CONTACT-GATE (§4): a |specific-force| spike (``|a_dn_raw - b_hat| > a_contact_mps2``) marks
        this tick (and the ``contact_hold_s`` refractory window after it) as CONTACT-FROZEN -- on
        such a tick BOTH the vz washout update and the z_off propagate are SKIPPED entirely (held),
        never merely clamped, so corrupted post-impact accel can never read as fictional descent."""
        if not self._seeded or not (0.0 < dt <= self.max_dt_s):
            return
        self._contact_elapsed_s += dt
        a_dn_raw = -float(a_up)
        a_dn = float(np.clip(a_dn_raw, -self.a_up_clamp_mps2, self.a_up_clamp_mps2))

        if not self._bias_capture_done:
            self._bias_capture_sum += a_dn
            self._bias_capture_n += 1
            self._bias_capture_elapsed_s += dt
            if self._bias_capture_elapsed_s >= self.bias_capture_s:
                mean_a_dn = self._bias_capture_sum / max(self._bias_capture_n, 1)
                self._b_hat = float(np.clip(mean_a_dn, -self.bias_clip_mps2, self.bias_clip_mps2))
                self._bias_capture_done = True
            return   # grounded during capture: vz stays 0, no washout advance yet, no contact gate

        # CONTACT-GATE (A25 §4): detect on the RAW (unclamped) specific-force-derived accel vs the
        # frozen bias -- a clamped 30 m/s^2 spike still corrupts the integrator, so the detector
        # must see the true magnitude before the a_up_clamp_mps2 clip is applied.
        if abs(a_dn_raw - self._b_hat) > self.a_contact_mps2:
            self._contact_until_s = self._contact_elapsed_s + self.contact_hold_s
        if self.contact_frozen():
            return   # HOLD: skip both the vz washout update and the z_off propagate this tick

        alpha = float(np.exp(-dt / self.washout_tau_s))
        self._vz = alpha * self._vz + (a_dn - self._b_hat) * dt

        # A25 z_off propagate: dead-reckon the gate-relative height by the just-updated vz, using
        # the SAME (post-update) vz and the SAME dt, so z_off and vz stay in lock-step. Only once a
        # gate has ever been latched (else z_off has no reference to propagate).
        if self._z_off_seen:
            self._z_off = float(np.clip(self._z_off - self._vz * dt,
                                        -self.z_off_clip_m, self.z_off_clip_m))

    # -- A25 gate-relative latch -----------------------------------------------
    def latch_offset(self, offset_z_world: float, obs_age_s: float) -> None:
        """Latch a FRESH gate-relative vertical-offset measurement (called by the seeker ONLY on a
        newly-detected pose, never a re-used cached one -- see the spec §1.4).

        Latency-compensates for the pose's capture-to-use age: over ``obs_age_s`` the drone moved
        vertically by ``-vz*obs_age_s`` (up-positive metres, since vz is down-positive and a climb
        reads vz<0), so the gate is now that much further below (or above) than the CAPTURED
        ``offset_z_world`` says:

            z_off_latched = offset_z_world_captured - vz * obs_age_s

        (spec §2.3, sign-pinned by ``test_latency_comp_climb_makes_latched_offset_more_positive``).
        Uses the CURRENT washout ``vz`` as a first-order proxy for the (unobserved) velocity during
        the blind interval -- acceptable over the sub-1s obs ages on this wire; no full OOSM buffer.
        No-op while un-seeded (mirrors ``predict``)."""
        if not self._seeded:
            return
        vz_for_latency = self._vz if np.isfinite(self._vz) else 0.0
        self._z_off = float(np.clip(
            float(offset_z_world) - vz_for_latency * float(obs_age_s),
            -self.z_off_clip_m, self.z_off_clip_m))
        self._z_off_seen = True
