import numpy as np
import pytest

from racer.contracts import Gate, NavState
from racer.planner import ReactivePlanner


def _gate(position, normal, gate_id=0) -> Gate:
    """A Gate whose through-direction (R_world_gate[:,2]) is `normal`."""
    n = _u(np.asarray(normal, float))
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _u(np.cross(a, n))
    y = np.cross(n, x)
    R = np.column_stack([x, y, n])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float), R_world_gate=R)


def _u(v):
    return v / np.linalg.norm(v)


def _nav(position, sim_time_ns=0):
    return NavState(sim_time_ns=sim_time_ns, position_ned=np.asarray(position, float))


def test_on_axis_approach_aims_through_gate():
    gate = _gate([10.0, 0.0, -3.0], normal=[1.0, 0.0, 0.0])     # gate 10 m north, facing north
    p = ReactivePlanner(cruise_speed=5.0, lookahead_m=2.0)
    sp = p.plan(_nav([0.0, 0.0, -3.0], sim_time_ns=99), gate)
    np.testing.assert_allclose(sp.position_ned, [12.0, 0.0, -3.0])  # carrot 2 m beyond the gate
    np.testing.assert_allclose(_u(sp.velocity_ned), [1.0, 0.0, 0.0], atol=1e-9)
    assert np.linalg.norm(sp.velocity_ned) == pytest.approx(5.0)    # at cruise speed
    assert sp.yaw == pytest.approx(0.0)                             # facing north
    assert sp.sim_time_ns == 99


@pytest.mark.parametrize("map_normal", [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
def test_normal_sign_is_corrected_to_travel_direction(map_normal):
    # Whatever sign the map stored, the carrot must land on the far (exit) side of the gate.
    gate = _gate([10.0, 0.0, 0.0], normal=map_normal)
    sp = ReactivePlanner(lookahead_m=2.0).plan(_nav([0.0, 0.0, 0.0]), gate)
    np.testing.assert_allclose(sp.position_ned, [12.0, 0.0, 0.0])   # always beyond, never behind


def test_off_axis_pursuit_points_at_carrot():
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    sp = ReactivePlanner(cruise_speed=3.0, lookahead_m=2.0).plan(_nav([0.0, 5.0, 0.0]), gate)
    # Carrot is at [12,0,0]; from [0,5,0] the line of sight points +north, -east.
    los = np.array([12.0, -5.0, 0.0])
    np.testing.assert_allclose(_u(sp.velocity_ned), _u(los), atol=1e-9)
    assert np.linalg.norm(sp.velocity_ned) == pytest.approx(3.0)
    assert sp.yaw == pytest.approx(np.arctan2(-5.0, 12.0))


def test_yaw_faces_a_gate_to_the_east():
    gate = _gate([0.0, 8.0, 0.0], normal=[0.0, 1.0, 0.0])          # gate due east, facing east
    sp = ReactivePlanner().plan(_nav([0.0, 0.0, 0.0]), gate)
    assert sp.yaw == pytest.approx(np.pi / 2)                       # heading east


def test_overshoot_with_forward_velocity_does_not_uturn():
    # [red-team 2026-05-30] Just past the gate plane but still moving forward through it: the
    # carrot must stay on the exit side, not flip to the approach side and U-turn back through
    # the gate. The velocity (not position-relative-to-gate) disambiguates the through-axis.
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    nav = NavState(sim_time_ns=0, position_ned=np.array([10.5, 0.0, 0.0]),  # 0.5 m past the plane
                   velocity_ned=np.array([5.0, 0.0, 0.0]))                   # still heading north
    sp = ReactivePlanner(lookahead_m=2.0).plan(nav, gate)
    np.testing.assert_allclose(sp.position_ned, [12.0, 0.0, 0.0])  # carrot still beyond the gate
    assert sp.velocity_ned[0] > 0.0                                # keeps going forward (no U-turn)


def test_velocity_is_a_feedforward_with_position_carrot():
    # The planner provides BOTH a position carrot and a velocity feedforward, so the floor
    # works whether the controller is in POSITION mode or runs the attitude PD law.
    gate = _gate([6.0, 0.0, -2.0], normal=[1.0, 0.0, 0.0])
    sp = ReactivePlanner().plan(_nav([0.0, 0.0, -2.0]), gate)
    assert sp.position_ned is not None and sp.velocity_ned is not None
