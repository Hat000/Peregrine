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

from dataclasses import dataclass, field
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
    gate_pass_radius_m: float = 1.0   # within this of a gate centre -> count it passed, advance


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
    _takeoff_xy: np.ndarray | None = field(default=None, repr=False)

    # -- lifecycle transitions ----------------------------------------------
    def start(self) -> None:
        """Begin the mission (IDLE -> TAKEOFF). No-op if already running."""
        if self.state == MissionState.IDLE:
            self.gate_index = 0
            self._takeoff_xy = None
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
            if self._takeoff_xy is None:
                self._takeoff_xy = np.asarray(nav.position_ned, dtype=np.float64)[:2].copy()
            target = np.array([self._takeoff_xy[0], self._takeoff_xy[1], -self.config.takeoff_altitude_m])
            if abs(float(nav.position_ned[2]) - target[2]) <= self.config.takeoff_tol_m:
                self.state = MissionState.RUN          # reached altitude -> fall through to RUN this tick
            else:
                return self.controller.command(
                    nav, Setpoint(sim_time_ns=nav.sim_time_ns, position_ned=target, yaw=nav.yaw)
                )

        if self.state == MissionState.RUN:
            while self.gate_index < len(self.gates) and self._passed(nav, self.gates[self.gate_index]):
                self.gate_index += 1
            if self.gate_index >= len(self.gates):
                self.state = MissionState.FINISHED
                return self._hold(nav)
            setpoint = self.planner.plan(nav, self.gates[self.gate_index])
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
    def _passed(self, nav: NavState, gate: Gate) -> bool:
        # Floor: proximity to the gate centre. (Refinement for high speed: detect crossing the
        # gate plane on the exit side within the opening, so a fast fly-through can't skip it.)
        d = np.asarray(nav.position_ned, dtype=np.float64) - np.asarray(gate.position_ned, dtype=np.float64)
        return float(np.linalg.norm(d)) <= self.config.gate_pass_radius_m

    def _hold(self, nav: NavState) -> ControlCommand:
        return self.controller.command(
            nav,
            Setpoint(
                sim_time_ns=nav.sim_time_ns,
                position_ned=np.asarray(nav.position_ned, dtype=np.float64).copy(),
                yaw=nav.yaw,
            ),
        )
