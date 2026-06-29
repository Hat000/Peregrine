"""Gate-seeker — a TRANSPARENT slow pursuit guidance law for the VQ2 self-localizing lap.

Per the "slow is smooth" curriculum (2026-06-29): the FIRST closed-loop VQ2 target is a
slow, zero-contact, fully self-localized lap — NOT a fast time and NOT the RL racing
policy. The RL policy is a black box trained off-distribution (it never saw a self-
localized estimate); to ISOLATE the self-localization we want a controller whose every
command we can explain. This module is that controller: a point-and-go pursuit law that
flies the drone slowly through the ACTIVE gate's centre.

Contract
--------
``GateSeeker.command(nav, gate, active_gate_index, *, is_final_gate=False, dt=None)
   -> ControlCommand`` (BODY_RATE / CTBR — body rates FRD + normalized collective [0,1]).

Inputs:
  * ``nav`` : :class:`NavState` — the SELF pose (position + velocity in world NED, attitude
    Euler, body rates) from the case-C Navigator. We consume only ``position_ned`` /
    ``velocity_ned`` / attitude / rates — the same fields the proven decoupled CTBR controller
    reads. We do NOT read ``time_since_vision_update_s`` here (the caller owns the coast policy).
  * ``gate`` : :class:`Gate` — the ACTIVE target gate's world pose (centre + frame). This is
    the gate the navigator localizes against; on the live wire it is selected by
    ``RACE_STATUS.active_gate_index`` (``DroneState.active_gate_index``).
  * ``active_gate_index`` : int — the current target index (drives the advance bookkeeping).

Output: a CTBR :class:`ControlCommand` ready for ``MavlinkClient.send_command`` (which applies
the deploy ``cmd_rate_scale``).

Guidance law (transparent, bounded)
-----------------------------------
1. **Steer the velocity at the gate CENTRE.** Build a carrot a short distance BEYOND the gate
   along its through-direction (so the line-of-sight passes through the opening, not the front
   bar), and ask for a desired velocity of magnitude ``cruise_speed`` pointing at that carrot.
   The decoupled CTBR controller turns a (capped) velocity error into a tilt and the matching
   collective; the cross-track component is corrected at full gain (centering) while the
   along-track speed is capped — exactly the proven ReactivePlanner split.
2. **Keep the gate in the +20deg camera view + hold altitude.** Yaw faces the gate so the
   forward-and-up-tilted camera sees the opening (the estimator needs the gate in frame to fix
   yaw/z). Altitude is held by the controller's velocity-damped alt-hold around hover thrust
   (z from the estimator) — we never command a vertical velocity, so blur-free slow flight
   stays level.
3. **Slow + bounded.** ``cruise_speed`` is small (default 3 m/s); the controller caps the
   commanded body rate and tilt. A launch ramp eases the takeoff transient from the start-gate
   spawn so the attitude target never STEPS to a cruise lean (the build-1.0.3364 tumble guard).
4. **Advance.** The caller advances ``active_gate_index`` from ``RACE_STATUS`` (the authoritative
   gate-ordering signal). This module additionally exposes :meth:`should_advance` as a
   RANGE-based backstop: advance when the drone has passed the gate plane (along-track sign flips
   to the exit side) within a capture radius, OR — for the FINAL gate, which has no "next" gate to
   re-aim at — when the drone is within ``final_blowout_m`` of the gate centre, dead-reckoning
   straight through (the recon "final blow-out" past the last station).

This is an ALTERNATIVE control source to the RL policy: same uplink (BODY_RATE), different brain.
It reuses the flight-proven decoupled :class:`Controller` geometry + sim-sign compensation, so no
new CTBR math is introduced — only the slow, gate-pointing SETPOINT on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from racer.contracts import ControlCommand, ControlMode, Gate, NavState, Setpoint
from racer.controller import Controller


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Unit vector; return ``fallback`` (or a zero vector) when ``v`` is ~0."""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3) if fallback is None else np.asarray(fallback, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


# Flight-proven decoupled-CTBR sim-sign compensation (MEASURED on ShadowPC; see
# scripts/twin_fly_course._FAITHFUL_SIGNS). The ODOMETRY roll-quat + roll/pitch-rate reporting
# inversions and the roll/yaw COMMAND inversion the live sim needs. ff_gain is 1.0 here (NOT 2.5):
# the VQ2 deploy applies the ~2.5x rate compensation at the UPLINK (cmd_rate_scale=0.4), so undoing
# it again in the controller would double-compensate. Apply it EXACTLY once, at the wire.
SEEKER_SIGNS = dict(
    body_rate_sign=np.array([1.0, 1.0, -1.0]),     # roll/yaw command inversion (flight-correct set)
    odo_att_sign=np.array([-1.0, 1.0, 1.0]),       # ODOMETRY-quat roll inverted
    odo_rate_sign=np.array([-1.0, -1.0, 1.0]),     # ODOMETRY rate: roll + pitch inverted
)


def make_seeker_controller(**overrides) -> Controller:
    """Build the decoupled CTBR controller the gate-seeker drives — the flight-proven gain set
    (twin_fly_course._CANONICAL_GAINS) + the live sim-sign compensation, with ``ff_gain=1.0``
    (the uplink cmd_rate_scale owns the 2.5x compensation). ``overrides`` set Controller fields."""
    params = dict(
        mode=ControlMode.BODY_RATE, decoupled=True,
        hover_thrust=0.2656,                       # fitted plant hover (plant-matched)
        kp_pos=1.2, kd_vel=3.0, max_speed=3.0, max_accel_mps2=8.0,
        kp_att=10.0, kd_att=0.30, max_body_rate_rps=4.0, ff_gain=1.0,
        kp_alt=2.0, kd_alt=3.0, alt_thrust_lo=0.05, alt_thrust_hi=0.6, tilt_comp=True,
    )
    params.update(SEEKER_SIGNS)
    params.update(overrides)
    return Controller(**params)


@dataclass
class GateSeekerConfig:
    """Tunables for the slow gate-seeker guidance law (all conservative defaults)."""

    cruise_speed: float = 3.0          # m/s, capped forward speed toward the gate (SLOW first)
    lookahead_m: float = 2.0           # carrot distance BEYOND the gate centre along its normal
    capture_radius_m: float = 0.6      # along-track capture: count the gate passed within this of its plane
    final_blowout_m: float = 4.0       # final gate: dead-reckon straight through within this range
    launch_ramp_s: float = 0.6         # ease the takeoff->cruise tilt over this long (anti-tumble)
    yaw_mode: str = "carrot"           # "carrot" faces the line-of-sight; "course" faces the gate normal


@dataclass
class GateSeeker:
    """Slow pursuit guidance -> CTBR command. Stateless guidance + a small advance bookkeeper.

    Holds a :class:`Controller` (the decoupled CTBR inner loop) and a launch clock so the takeoff
    transient is ramped. ``plan(nav, gate, ...)`` returns the SETPOINT (for inspection / tests);
    ``command(nav, gate, ...)`` returns the CTBR :class:`ControlCommand`.
    """

    config: GateSeekerConfig = field(default_factory=GateSeekerConfig)
    controller: Controller = field(default_factory=make_seeker_controller)
    _t0_sim_ns: int | None = field(default=None, repr=False)   # first-command sim time (launch clock)
    _last_index: int | None = field(default=None, repr=False)  # last seen active_gate_index

    # -- guidance: NavState + active gate -> Setpoint -----------------------
    def plan(self, nav: NavState, gate: Gate, *, is_final_gate: bool = False) -> Setpoint:
        """Slow pursuit setpoint: a carrot just beyond the gate centre, a capped desired velocity
        pointing at it, a yaw that keeps the gate in the +20deg camera view, and a launch ramp.

        The gate normal sign is arbitrary in the map; orient it to point the way we are TRAVELLING
        (downrange, away from us) so the carrot lands on the EXIT side and the drone is pulled
        THROUGH the opening rather than to a point in front of the front bar. Disambiguate by the
        velocity direction when moving fast enough to trust it, else by the position-to-gate vector
        (takeoff / hover) — exactly the ReactivePlanner discipline (a momentary speed dip must not
        flip the carrot 180deg and U-turn)."""
        if self._t0_sim_ns is None:           # arm the launch clock on the first plan/command
            self._t0_sim_ns = int(nav.sim_time_ns)
        position = np.asarray(nav.position_ned, dtype=np.float64)
        velocity = np.asarray(nav.velocity_ned, dtype=np.float64)
        gate_pos = np.asarray(gate.position_ned, dtype=np.float64)
        to_gate = gate_pos - position

        heading_ref = velocity if float(np.linalg.norm(velocity)) > 0.5 else to_gate
        travel = np.asarray(gate.normal_ned, dtype=np.float64)
        if float(travel @ heading_ref) < 0.0:
            travel = -travel
        travel = _unit(travel, fallback=_unit(to_gate))

        carrot = gate_pos + self.config.lookahead_m * travel
        los_dir = _unit(carrot - position, fallback=travel)
        heading = travel if self.config.yaw_mode == "course" else los_dir
        yaw = float(np.arctan2(heading[1], heading[0]))

        ramp = self._launch_ramp(int(nav.sim_time_ns))
        return Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            position_ned=carrot,
            velocity_ned=self.config.cruise_speed * los_dir,
            yaw=yaw,
            launch_ramp=ramp,
        )

    # -- guidance -> CTBR ControlCommand ------------------------------------
    def command(self, nav: NavState, gate: Gate, active_gate_index: int, *,
                is_final_gate: bool = False) -> ControlCommand:
        """Full guidance: plan the slow pursuit setpoint, then run the decoupled CTBR controller.

        Returns a BODY_RATE :class:`ControlCommand` (body rates FRD + normalized collective). Tracks
        ``active_gate_index`` so :meth:`advanced_since_last` can report a wire-driven advance."""
        if self._t0_sim_ns is None:
            self._t0_sim_ns = int(nav.sim_time_ns)
        self._last_index = int(active_gate_index)
        sp = self.plan(nav, gate, is_final_gate=is_final_gate)
        return self.controller.command(nav, sp)

    # -- advance logic ------------------------------------------------------
    def should_advance(self, nav: NavState, gate: Gate, *, is_final_gate: bool = False) -> bool:
        """RANGE/geometry backstop for advancing to the next gate (the wire's
        ``active_gate_index`` increment is authoritative; this is the fallback when it lags).

        Non-final gate: advance once the drone has crossed the gate PLANE onto the exit side
        (the along-track signed distance flips sign) within ``capture_radius_m`` of the centre.
        Final gate: no next gate to re-aim at, so dead-reckon — advance (treat as cleared) once
        within ``final_blowout_m`` of the centre, letting the drone fly straight through."""
        position = np.asarray(nav.position_ned, dtype=np.float64)
        gate_pos = np.asarray(gate.position_ned, dtype=np.float64)
        velocity = np.asarray(nav.velocity_ned, dtype=np.float64)
        to_gate = gate_pos - position
        rng = float(np.linalg.norm(to_gate))

        if is_final_gate:
            return rng <= self.config.final_blowout_m

        # orient the normal downrange (same disambiguation as plan), then the signed along-track
        # distance of the drone behind the gate plane is (gate - pos) . travel ; <= 0 means we have
        # crossed to the exit side.
        heading_ref = velocity if float(np.linalg.norm(velocity)) > 0.5 else to_gate
        travel = np.asarray(gate.normal_ned, dtype=np.float64)
        if float(travel @ heading_ref) < 0.0:
            travel = -travel
        travel = _unit(travel, fallback=_unit(to_gate))
        along = float(to_gate @ travel)         # > 0 : gate ahead ; <= 0 : crossed the plane
        # lateral miss at the crossing (reject a far side-pass that never threaded the opening)
        lateral = float(np.linalg.norm(to_gate - along * travel))
        return along <= 0.0 and lateral <= max(self.config.capture_radius_m, 0.75)

    def advanced_since_last(self, active_gate_index: int) -> bool:
        """True iff ``active_gate_index`` is greater than the index of the previous command (the
        wire-driven advance). ``False`` before the first command (no baseline)."""
        return self._last_index is not None and int(active_gate_index) > self._last_index

    def reset(self) -> None:
        """Drop the launch clock + advance baseline (e.g. on a sim epoch restart)."""
        self._t0_sim_ns = None
        self._last_index = None

    # -- internals ----------------------------------------------------------
    def _launch_ramp(self, sim_time_ns: int) -> float | None:
        """Launch ramp [0,1] over ``launch_ramp_s`` from the first command, on the sim master clock.
        ``None`` once the ramp completes (full authority) or when disabled (ramp_s <= 0)."""
        if self.config.launch_ramp_s <= 0.0 or self._t0_sim_ns is None:
            return None
        elapsed_s = (int(sim_time_ns) - self._t0_sim_ns) / 1e9
        if elapsed_s >= self.config.launch_ramp_s:
            return None
        return float(np.clip(elapsed_s / self.config.launch_ramp_s, 0.0, 1.0))
