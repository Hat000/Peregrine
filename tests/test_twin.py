"""Physics tests for the offline CTBR plant twin (``racer.twin``).

Canonical conventions: world NED (Z down), body FRD, thrust along body -Z (up). These pin the
sign relationships the controller relies on (roll -> +Y, nose-down -> +X, thrust=hover -> hover).
"""
import numpy as np

from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant, CtbrPlantConfig


def _cmd(body_rate, thrust):
    return ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.asarray(body_rate, float), thrust=thrust)


def _run(plant, body_rate, thrust, seconds, dt=0.005):
    for _ in range(int(round(seconds / dt))):
        plant.step(_cmd(body_rate, thrust), dt)
    return plant.state()


def test_hover_holds_position():
    p = CtbrPlant()                                   # default hover_thrust=0.26
    s = _run(p, [0, 0, 0], 0.26, seconds=2.0)
    np.testing.assert_allclose(s.position_ned, [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(s.velocity_ned, [0, 0, 0], atol=1e-6)
    # at hover the accelerometer reads -g on body z (thrust opposing gravity)
    np.testing.assert_allclose(s.accel_body, [0, 0, -9.80665], atol=1e-6)


def test_thrust_above_hover_climbs():
    s = _run(CtbrPlant(), [0, 0, 0], 0.40, seconds=1.0)
    assert s.velocity_ned[2] < -1.0          # NED z+ = down, so climbing = vz negative
    assert s.position_ned[2] < 0.0           # gained altitude (z decreased)


def test_zero_thrust_freefalls_at_g():
    s = _run(CtbrPlant(), [0, 0, 0], 0.0, seconds=0.5, dt=0.001)
    np.testing.assert_allclose(s.velocity_ned[2], 9.80665 * 0.5, rtol=2e-3)  # vz = g*t downward
    np.testing.assert_allclose(s.velocity_ned[:2], [0, 0], atol=1e-9)


def test_positive_roll_rate_accelerates_right():
    # +roll rate -> right-wing-down -> thrust tilts +Y (right). Small rate / short time so the
    # roll stays small and the sign is unambiguous.
    s = _run(CtbrPlant(), [0.5, 0, 0], 0.26, seconds=0.3)
    assert s.roll > 0.0
    assert s.velocity_ned[1] > 0.05          # +Y (east/right)
    assert abs(s.velocity_ned[0]) < 1e-3     # no forward/back coupling


def test_nose_down_pitch_accelerates_forward():
    # negative pitch rate -> nose down -> thrust tilts +X (forward).
    s = _run(CtbrPlant(), [0, -0.5, 0], 0.26, seconds=0.3)
    assert s.pitch < 0.0
    assert s.velocity_ned[0] > 0.05          # +X (north/forward)
    assert abs(s.velocity_ned[1]) < 1e-3


def test_inner_rate_loop_first_order_lag():
    # realized rate approaches rate_gain*command with the configured time constant.
    p = CtbrPlant(CtbrPlantConfig(rate_tau_s=0.05))
    # after 4 tau the first-order response is 1-e^-4 = 98.2% of the unity target
    p_state = _run(p, [1.0, 0, 0], 0.26, seconds=0.20, dt=0.001)
    assert 0.95 < p.omega[0] < 1.0
    # a faithful gain scales the steady rate
    p2 = CtbrPlant(CtbrPlantConfig(rate_tau_s=0.05, rate_gain=np.array([2.6, 2.6, 2.6])))
    _run(p2, [1.0, 0, 0], 0.26, seconds=0.5, dt=0.001)
    np.testing.assert_allclose(p2.omega[0], 2.6, rtol=2e-2)


def test_rate_sign_inversion_models_the_sim_quirk():
    # the sim inverts the roll command sign; a faithful twin applies rate_sign so a +roll command
    # yields a NEGATIVE realized roll rate (which the controller's body_rate_sign then compensates).
    p = CtbrPlant(CtbrPlantConfig(rate_sign=np.array([-1.0, 1.0, -1.0])))
    _run(p, [1.0, 0, 0], 0.26, seconds=0.3, dt=0.001)
    assert p.omega[0] < 0.0


def test_state_is_a_well_formed_dronestate():
    s = CtbrPlant(position_ned=[1, 2, -3], velocity_ned=[0.1, 0, 0]).state()
    assert s.position_ned.shape == (3,) and s.velocity_ned.shape == (3,)
    assert s.orientation_ned_wxyz.shape == (4,)
    assert s.armed is True
    np.testing.assert_allclose(s.orientation_ned_wxyz, [1, 0, 0, 0], atol=1e-9)


def test_odometry_frame_bug_causes_lateral_oscillation_offline():
    # The whole point of the twin: reproduce the gate-0 lateral oscillation deterministically and
    # show the frame fix (c3b5a8e) cures it, running the REAL decoupled controller. Imports the demo
    # scenario so this assertion tracks scripts/twin_oscillation.py.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from twin_oscillation import run_scenario

    clean = run_scenario("clean")     # the FIXED client: true world velocity feedback
    flip = run_scenario("flip")       # the bug, worst case: body-frame velocity (vy sign-flipped)
    mix = run_scenario("mix")         # the bug as it really was: world/body interleave
    assert not clean["diverged"] and clean["settle_s"] < 3.0          # clean converges
    assert flip["diverged"]                                          # full flip -> anti-damping
    assert mix["settle_s"] > 2.0 * clean["settle_s"]                 # interleave -> slow oscillation
