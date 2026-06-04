import numpy as np

from racer.contracts import ControlMode, Gate, NavState
from racer.controller import Controller
from racer.mission import Mission, MissionConfig, MissionState
from racer.planner import ReactivePlanner


def _gate(position, normal=(1.0, 0.0, 0.0), gate_id=0) -> Gate:
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, n)
    x = x / np.linalg.norm(x)
    R = np.column_stack([x, np.cross(n, x), n])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float), R_world_gate=R)


def _nav(position, sim_time_ns=0):
    return NavState(sim_time_ns=sim_time_ns, position_ned=np.asarray(position, float))


def _mission(gates):
    return Mission(gates=gates, planner=ReactivePlanner(), controller=Controller(mode=ControlMode.POSITION),
                   config=MissionConfig(takeoff_altitude_m=1.5, takeoff_tol_m=0.3, gate_pass_radius_m=1.0))


def test_starts_idle_and_holds_before_start():
    m = _mission([_gate([5.0, 0.0, -1.5])])
    assert m.state is MissionState.IDLE
    cmd = m.step(_nav([2.0, 3.0, -1.5]))
    assert m.state is MissionState.IDLE                       # still idle until start()
    np.testing.assert_allclose(cmd.position_ned, [2.0, 3.0, -1.5])  # holds where it is


def test_takeoff_climbs_then_switches_to_run():
    m = _mission([_gate([5.0, 0.0, -1.5])])
    m.start()
    assert m.state is MissionState.TAKEOFF
    cmd = m.step(_nav([0.0, 0.0, 0.0]))                       # on the ground at the start point
    assert m.state is MissionState.TAKEOFF
    np.testing.assert_allclose(cmd.position_ned, [0.0, 0.0, -1.5])  # climb 1.5 m above the start xy
    # Reaching altitude flips to RUN and immediately produces guidance toward the first gate.
    cmd = m.step(_nav([0.0, 0.0, -1.5]))
    assert m.state is MissionState.RUN
    assert cmd.position_ned[0] > 0.0                          # heading toward the gate (north)


def test_takeoff_target_is_relative_to_start_altitude():
    # [red-team 2026-05-30] If the estimator's z carries an absolute (e.g. MSL) bias so the pad
    # is not at z=0, an absolute takeoff target would launch to the wrong height. The target
    # must be takeoff_altitude_m ABOVE wherever the drone actually starts.
    m = _mission([_gate([5.0, 0.0, -11.5])])
    m.start()
    cmd = m.step(_nav([2.0, 3.0, -10.0]))                     # pad sits at z=-10 (uncalibrated baro)
    assert m.state is MissionState.TAKEOFF
    np.testing.assert_allclose(cmd.position_ned, [2.0, 3.0, -11.5])  # 1.5 m above start, not absolute -1.5


def test_gates_advance_on_proximity_then_finish():
    m = _mission([_gate([5.0, 0.0, -1.5], gate_id=0), _gate([10.0, 0.0, -1.5], gate_id=1)])
    m.start()
    m.step(_nav([0.0, 0.0, 0.0]))                             # on the ground -> capture takeoff origin
    m.step(_nav([0.0, 0.0, -1.5]))                            # reached altitude -> RUN, target gate 0
    assert m.state is MissionState.RUN and m.gate_index == 0
    m.step(_nav([5.0, 0.0, -1.5]))                            # at gate 0 -> advance to gate 1
    assert m.gate_index == 1 and m.state is MissionState.RUN
    m.step(_nav([10.0, 0.0, -1.5]))                           # at gate 1 -> finished
    assert m.state is MissionState.FINISHED


def test_fast_flythrough_advances_via_plane_crossing():
    # [red-team 2026-05-30] At speed the drone can pass cleanly through the 1.5 m opening yet
    # never sample inside the 1.0 m proximity sphere. A sample just past the gate plane, inside
    # the opening, must still advance -- otherwise gate_index sticks and the planner U-turns.
    m = _mission([_gate([5.0, 0.0, -1.5], normal=(1.0, 0.0, 0.0), gate_id=0),
                  _gate([10.0, 0.0, -1.5], gate_id=1)])
    m.start()
    m.step(_nav([0.0, 0.0, 0.0]))                            # on the ground -> capture takeoff origin
    m.step(_nav([0.0, 0.0, -1.5]))                           # reached altitude -> RUN, target gate 0
    assert m.gate_index == 0
    nav = NavState(sim_time_ns=1, position_ned=np.array([6.2, 0.5, -1.5]),
                   velocity_ned=np.array([6.0, 0.0, 0.0]))    # 1.2 m past plane, 0.5 m off-centre
    assert float(np.linalg.norm(nav.position_ned - np.array([5.0, 0.0, -1.5]))) > 1.0  # missed the sphere
    m.step(nav)
    assert m.gate_index == 1                                  # advanced via the plane crossing


def test_no_false_pass_on_axis_but_far_from_gate():
    # Regression (gate0_given2): a drone sitting on the gate's AXIS but far away (here ~20 m before
    # it, on the start line) must NOT pass. With a near-vertical/ambiguous velocity the through-sign
    # flips, so the old code saw "past the plane + inside the square" and false-advanced during
    # takeoff. The depth bound rejects it (20 m >> gate_pass_depth_m).
    m = _mission([_gate([5.0, 0.0, -1.5], normal=(1.0, 0.0, 0.0), gate_id=0)])
    m.start()
    m.step(_nav([0.0, 0.0, 0.0]))
    m.step(_nav([0.0, 0.0, -1.4]))
    nav = NavState(sim_time_ns=1, position_ned=np.array([-15.0, 0.0, -1.5]),  # 20 m before, on-axis
                   velocity_ned=np.array([-0.1, 0.0, -1.0]))                  # ambiguous -> flips through
    m.step(nav)
    assert m.gate_index == 0                                  # far from the gate -> no advance


def test_no_advance_when_past_plane_but_outside_opening():
    # Past the gate plane but 2 m off to the side: flew AROUND the gate, not through it. Must
    # NOT count as passed (a false advance would skip a gate and invalidate the run).
    m = _mission([_gate([5.0, 0.0, -1.5], normal=(1.0, 0.0, 0.0), gate_id=0)])
    m.start()
    m.step(_nav([0.0, 0.0, 0.0]))                             # on the ground -> capture takeoff origin
    m.step(_nav([0.0, 0.0, -1.5]))                            # reached altitude -> RUN
    nav = NavState(sim_time_ns=1, position_ned=np.array([6.0, 2.0, -1.5]),
                   velocity_ned=np.array([6.0, 0.0, 0.0]))
    m.step(nav)
    assert m.gate_index == 0                                  # outside the opening -> no advance


def test_finished_holds_position():
    m = _mission([])                                          # no gates -> finishes as soon as it runs
    m.start()
    m.step(_nav([0.0, 0.0, 0.0]))                             # on the ground -> capture takeoff origin
    m.step(_nav([0.0, 0.0, -1.5]))                            # TAKEOFF -> RUN -> FINISHED
    assert m.state is MissionState.FINISHED
    cmd = m.step(_nav([1.0, 2.0, -1.5]))
    np.testing.assert_allclose(cmd.position_ned, [1.0, 2.0, -1.5])  # holds


def test_abort_holds_and_sticks():
    m = _mission([_gate([5.0, 0.0, -1.5])])
    m.start()
    m.abort()
    assert m.state is MissionState.ABORT
    cmd = m.step(_nav([3.0, 0.0, -1.5]))
    assert m.state is MissionState.ABORT
    np.testing.assert_allclose(cmd.position_ned, [3.0, 0.0, -1.5])


def test_closed_loop_point_mass_flies_the_course():
    # The walking skeleton actually walks: a first-order point-mass that chases each commanded
    # position should be guided through every gate to FINISHED -- planner -> controller -> plant
    # -> proximity advance, closed.
    gates = [_gate([5.0, 0.0, -1.5], gate_id=0),
             _gate([10.0, 0.0, -1.5], gate_id=1),
             _gate([15.0, 0.0, -1.5], gate_id=2)]
    m = _mission(gates)
    m.start()
    pos = np.array([0.0, 0.0, 0.0])          # on the ground at the start point
    dt, max_speed = 0.05, 4.0
    for _ in range(4000):
        cmd = m.step(_nav(pos.copy()))
        target = cmd.position_ned if cmd.position_ned is not None else pos
        to = np.asarray(target, float) - pos
        d = float(np.linalg.norm(to))
        if d > 1e-9:
            pos = pos + (to / d) * min(d, max_speed * dt)   # move toward the command, speed-capped
        if m.state is MissionState.FINISHED:
            break
    assert m.state is MissionState.FINISHED
    assert m.gate_index == 3
    np.testing.assert_allclose(pos, [15.0, 0.0, -1.5], atol=1.0)   # ended at the last gate


def test_run_drives_to_finished_end_to_end():
    gates = [_gate([5.0, 0.0, -1.5], gate_id=0), _gate([10.0, 0.0, -1.5], gate_id=1)]
    m = _mission(gates)

    # A scripted "navigator": pretend the drone follows commands -- climb, then reach each gate.
    script = [
        _nav([0.0, 0.0, 0.0]),     # on the ground -> capture takeoff origin
        _nav([0.0, 0.0, -1.5]),    # reached altitude -> RUN
        _nav([5.0, 0.0, -1.5]),    # at gate 0
        _nav([10.0, 0.0, -1.5]),   # at gate 1 -> finish
    ]
    feed = iter(script)
    last = {"nav": script[-1]}

    def navigator():
        try:
            last["nav"] = next(feed)
        except StopIteration:
            pass
        return last["nav"]

    class _Transport:
        def __init__(self):
            self.commands = []

        def send_command(self, cmd):
            self.commands.append(cmd)

    transport = _Transport()
    final = m.run(navigator, transport, max_steps=50)
    assert final is MissionState.FINISHED
    assert m.gate_index == 2
    assert len(transport.commands) >= 4                       # one command per tick, all valid
    assert all(c.mode is ControlMode.POSITION for c in transport.commands)
