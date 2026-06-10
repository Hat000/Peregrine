"""Mission lifecycle — the loop owner. Sequences arm/takeoff -> fly the gates -> finish and,
each control tick, turns a fused :class:`NavState` into a :class:`ControlCommand` by driving
the planner + controller.

``Mission.step(nav)`` is the testable core: a small state machine (IDLE -> TAKEOFF -> RUN ->
FINISHED, plus ABORT) that, given where we are, decides what to command and when to advance to
the next gate. ``Mission.run(navigator, transport)`` is the thin driver: each tick it asks the
caller's ``navigator`` for the current NavState (the caller wires pump -> detect -> PnP ->
localize -> KF behind it) and ships ``step``'s command through ``transport`` (the mavlink
client). Keeping the I/O behind those two seams lets the whole lifecycle be unit-tested with
fakes, and lets the not-yet-built detector / mapper slot into the navigator without touching
this file.

Floor scope (the walking skeleton): reactive EXPLORE guidance through a provided ordered gate
list, proximity-based gate advance, position-hold on finish. Upgrades that slot in later: a
RACE state tracking a min-snap line; gate-plane-crossing advance for high speed; arm/disarm +
land handshake (a first-contact / session-lifecycle concern).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum

import numpy as np

from racer.contracts import ControlCommand, Gate, NavState, Setpoint
from racer.controller import Controller
from racer.planner import ReactivePlanner


class MissionState(IntEnum):
    IDLE = 0       # pre-start: hold, await start()
    TAKEOFF = 1    # climb to the hover altitude over the start point
    RUN = 2        # fly the ordered gates reactively
    FINISHED = 3   # course complete: hold
    ABORT = 4      # safe hold (failsafe)


@dataclass
class MissionConfig:
    takeoff_altitude_m: float = 1.5   # hover target; NED z = -altitude
    takeoff_tol_m: float = 0.3        # within this of target altitude -> start the run
    # Proximity pass radius. MUST be <= the inner half-opening (~0.75 m) or it FALSE-POSITIVES a
    # miss: a drone that flies 1.3 m wide of the gate is still within a 1.8 m sphere and scores a
    # bogus "pass" (measured 2026-06-04 -- a visually-confirmed miss read as gate_index=1). The
    # honest fast-pass test is path (2), plane-crossing INSIDE the inner square; this sphere is
    # only the slow-centred backup, so keep it tight.
    gate_pass_radius_m: float = 0.75
    # Plane-crossing pass also requires the drone be within this distance of the gate plane along the
    # through-axis (not just past it + inside the square) -- else a drone on the gate's AXIS but far
    # away false-passes when the through-sign flips. Generous enough for a fast fly-through sampled
    # one frame past the plane.
    gate_pass_depth_m: float = 2.0
    # Takeoff->RUN launch ramp (s): over this long after RUN begins, scale the controller's
    # horizontal-accel command (hence the desired tilt) linearly 0 -> 1 so the attitude target
    # grows smoothly instead of STEPPING to the cruise lean -- a step saturates the body-rate clamp
    # and tumbles (sim build 1.0.3364, the deterministic start-transient). The ramp is keyed off the
    # sim CLOCK, so it is invariant to control rate and to the tick/pose-stream phase that turned the
    # old transient into a dice-roll. 0 disables (legacy step). Tuned offline on the faithful twin
    # across a control-rate x tick-phase sweep (scripts/twin_launch_phase_sweep.py).
    launch_ramp_s: float = 0.6


@dataclass
class Mission:
    """Mission state machine + per-tick guidance. Inject the planner, controller, and the
    ordered gate map (from the mapper/orderer; a provided list for the floor)."""

    gates: list[Gate]
    planner: ReactivePlanner
    controller: Controller
    config: MissionConfig = field(default_factory=MissionConfig)
    state: MissionState = MissionState.IDLE
    gate_index: int = 0
    _takeoff_origin: np.ndarray | None = field(default=None, repr=False)
    _run_start_ns: int | None = field(default=None, repr=False)   # sim-time the RUN began (launch ramp)

    # -- lifecycle transitions ----------------------------------------------
    def start(self) -> None:
        """Begin the mission (IDLE -> TAKEOFF). No-op if already running."""
        if self.state == MissionState.IDLE:
            self.gate_index = 0
            self._takeoff_origin = None
            self._run_start_ns = None
            self.state = MissionState.TAKEOFF

    def abort(self) -> None:
        """Failsafe: stop flying the course and hold (e.g. lost vision for too long)."""
        self.state = MissionState.ABORT

    @property
    def target_gate(self) -> Gate | None:
        if self.state == MissionState.RUN and self.gate_index < len(self.gates):
            return self.gates[self.gate_index]
        return None

    # -- per-tick control ---------------------------------------------------
    def step(self, nav: NavState) -> ControlCommand:
        """One control iteration: advance the state machine and return the command to send."""
        if self.state == MissionState.TAKEOFF:
            if self._takeoff_origin is None:
                self._takeoff_origin = np.asarray(nav.position_ned, dtype=np.float64).copy()
            # Climb to takeoff_altitude_m ABOVE the start point, not to an absolute NED z.
            # The estimator's z origin is only zero at the pad if the baro is referenced at
            # arming; if it carries an absolute (MSL) bias, an absolute target would launch
            # the drone to the wrong height (into the ceiling or the floor). Relative-to-start
            # is robust either way and mirrors how the start xy is already captured. [red-team]
            target = self._takeoff_origin + np.array([0.0, 0.0, -self.config.takeoff_altitude_m])
            if abs(float(nav.position_ned[2]) - target[2]) <= self.config.takeoff_tol_m:
                self.state = MissionState.RUN          # reached altitude -> fall through to RUN this tick
            else:
                return self.controller.command(
                    nav, Setpoint(sim_time_ns=nav.sim_time_ns, position_ned=target, yaw=nav.yaw)
                )

        if self.state == MissionState.RUN:
            if self._run_start_ns is None:                      # first RUN tick: anchor the launch ramp
                self._run_start_ns = int(nav.sim_time_ns)
            while self.gate_index < len(self.gates) and self._passed(nav, self.gates[self.gate_index]):
                self.gate_index += 1
            if self.gate_index >= len(self.gates):
                self.state = MissionState.FINISHED
                return self._hold(nav)
            setpoint = self.planner.plan(nav, self.gates[self.gate_index])
            ramp = self._launch_ramp(nav)
            if ramp < 1.0:                                      # ramp the launch lean (build-1.0.3364 fix)
                setpoint = replace(setpoint, launch_ramp=ramp)
            return self.controller.command(nav, setpoint)

        # IDLE / FINISHED / ABORT: hold position.
        return self._hold(nav)

    def run(self, navigator, transport, max_steps: int | None = None, should_stop=None) -> MissionState:
        """Drive the mission to completion. ``navigator()`` returns the current NavState (caller
        wires perception+estimation behind it); ``transport.send_command(cmd)`` ships each
        command. Returns the terminal state. ``max_steps`` / ``should_stop`` bound the loop."""
        self.start()
        steps = 0
        while self.state not in (MissionState.FINISHED, MissionState.ABORT):
            cmd = self.step(navigator())
            transport.send_command(cmd)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            if should_stop is not None and should_stop():
                break
        return self.state

    # -- helpers ------------------------------------------------------------
    def _launch_ramp(self, nav: NavState) -> float:
        """Linear 0 -> 1 authority scale over ``launch_ramp_s`` after the RUN began, keyed off the
        sim CLOCK (``sim_time_ns``). Keying off sim-time -- not tick count -- is what makes the ramp
        invariant to the control rate AND to the tick/pose-stream phase: the same wall-clock fraction
        of the ramp is applied no matter when or how often we tick. Returns 1.0 (full authority) once
        the window has elapsed, when the ramp is disabled, or before RUN is anchored."""
        if self.config.launch_ramp_s <= 0.0 or self._run_start_ns is None:
            return 1.0
        elapsed_s = (int(nav.sim_time_ns) - self._run_start_ns) / 1e9
        return float(np.clip(elapsed_s / self.config.launch_ramp_s, 0.0, 1.0))

    def _passed(self, nav: NavState, gate: Gate) -> bool:
        rel = np.asarray(nav.position_ned, dtype=np.float64) - np.asarray(gate.position_ned, dtype=np.float64)
        # (1) Proximity: sign-agnostic, catches slow / centred passes.
        if float(np.linalg.norm(rel)) <= self.config.gate_pass_radius_m:
            return True
        # (2) Plane crossing within the opening: a fast fly-through translates >0.5 m/frame, so
        # it can pass cleanly through the 1.5 m opening yet never sample inside the 1.0 m sphere
        # -- gate_index would never advance and the planner would U-turn to re-enter. Advance
        # once the drone is on/past the gate's exit plane AND inside the inner square. Orient the
        # through-axis downrange by the drone's motion (not its position-relative-to-gate, which
        # inverts the instant it crosses the plane); fall back to approach geometry at rest.
        R = np.asarray(gate.R_world_gate, dtype=np.float64)
        through = R[:, 2]
        vel = np.asarray(nav.velocity_ned, dtype=np.float64)
        ref = vel if float(np.linalg.norm(vel)) > 1e-3 else -rel
        if through @ ref < 0.0:
            through = -through
        depth = float(rel @ through)       # signed distance along the through-axis (exit side +)
        # Must have JUST crossed the plane: on/just past it (0 <= depth) AND still near it
        # (< depth_tol). The near-bound is essential -- without it a drone sitting on the gate AXIS
        # but tens of metres away (e.g. on the start line, before the gate) false-passes the instant
        # the velocity-based through-sign flips at takeoff (measured gate0_given2 at x=-2). A real
        # fly-through is sampled within depth_tol of the plane at any sane rate.
        if not (0.0 <= depth <= self.config.gate_pass_depth_m):
            return False
        half = gate.inner_size_m / 2.0
        return abs(float(rel @ R[:, 0])) <= half and abs(float(rel @ R[:, 1])) <= half

    def _hold(self, nav: NavState) -> ControlCommand:
        return self.controller.command(
            nav,
            Setpoint(
                sim_time_ns=nav.sim_time_ns,
                position_ned=np.asarray(nav.position_ned, dtype=np.float64).copy(),
                yaw=nav.yaw,
            ),
        )
