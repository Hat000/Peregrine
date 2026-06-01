"""Planner — the THINK link. Turns the current :class:`NavState` + the target :class:`Gate`
into a :class:`Setpoint` for the controller.

:class:`ReactivePlanner` is the EXPLORE-mode floor: pure-pursuit a "carrot" placed just
beyond the gate along its through-direction, so the drone flies THROUGH the gate centre
(rather than braking to a stop on it) while yawing to face the way it is going. It targets
ONE gate at a time — choosing *which* gate is next is the orderer / mission-loop's job
(fly the gate ahead, then re-target the next; never re-target a passed gate).

This is the lowest-risk THINK floor (a slow VALID finish beats a fast invalid one). The
RACE-mode upgrade — a min-snap line through the ordered gate map, then MPCC / an RL policy —
slots into the very same Setpoint seam without touching the controller or mission loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racer.contracts import Gate, NavState, Setpoint


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 1e-9:
        return v / n
    return np.array([1.0, 0.0, 0.0]) if fallback is None else fallback


@dataclass
class ReactivePlanner:
    """Reactive line-of-sight guidance toward a single target gate."""

    cruise_speed: float = 4.0     # m/s along the line of sight (conservative; raise for RACE)
    lookahead_m: float = 2.0      # carrot distance beyond the gate along the through-direction

    def plan(self, nav: NavState, gate: Gate) -> Setpoint:
        position = np.asarray(nav.position_ned, dtype=np.float64)
        gate_pos = np.asarray(gate.position_ned, dtype=np.float64)
        to_gate = gate_pos - position

        # The map's gate normal sign is arbitrary; orient it to point the way we're travelling
        # ("through the gate, away from us") so the carrot lands on the exit side. Disambiguate
        # by the drone's VELOCITY when it is moving: position-relative-to-gate (to_gate) inverts
        # the instant the drone crosses the gate plane, so a momentary overshoot before the
        # mission advances the gate would flip the carrot to the approach side and trigger a
        # high-speed U-turn back through the gate. Velocity keeps pointing downrange through the
        # pass; fall back to to_gate only at near-zero speed (takeoff / hover). [red-team 2026-05-30]
        velocity = np.asarray(nav.velocity_ned, dtype=np.float64)
        heading_ref = velocity if float(np.linalg.norm(velocity)) > 1e-3 else to_gate
        travel = np.asarray(gate.normal_ned, dtype=np.float64)
        if travel @ heading_ref < 0.0:
            travel = -travel
        travel = _unit(travel, fallback=_unit(to_gate))

        carrot = gate_pos + self.lookahead_m * travel       # a point just beyond the gate centre
        los_dir = _unit(carrot - position, fallback=travel)
        yaw = float(np.arctan2(los_dir[1], los_dir[0]))      # NED heading (north->east) toward the carrot
        return Setpoint(
            sim_time_ns=nav.sim_time_ns,
            position_ned=carrot,
            velocity_ned=self.cruise_speed * los_dir,
            yaw=yaw,
        )
