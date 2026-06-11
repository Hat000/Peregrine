"""Super-rate static gain map + slew limit -- the OBJECTIVE ANCHOR for the inner-loop integration.

The characterize-sweep (handoff/shadowpc-characterize-sweep-2026-06-10/WRITEUP.md Section 1) proved
the sim's inner loop is a STATIC amplitude-dependent gain map ``g(|c|) = G0/(1 - s*min(|c|,pi)/pi)``
(s ~= 0.30, G0 = the shipped flat gains) in front of the existing first-order lag, plus a per-axis
slew limit (~260 rad/s^2 roll/pitch, ~80 yaw). These tests pin:

  1. the MEASURED sustained-gain table (the n=11 sweep data, NOT the model fitted to itself):
     |cmd| 0.30 -> g 2.505, 1.00 -> 2.712, 2.00 -> 3.043, 3.14 -> 3.50 (roll/pitch) within ~3%
     -- the one-parameter form's documented fit quality ("fits within ~3%, slightly flat at the
     low end"); a regression here means the implemented map is NOT the measured curve;
  2. slew clamping (exact per-step increment cap);
  3. legacy equivalence: params OFF (None) -> bit-identical to the pre-map update, and s=0 with no
     slew -> bit-identical to the flat-gain plant;
  4. twin == rl_plant under the map (spot check; the full battery lives in test_rl_plant_parity).

Yaw caveat (documented, deliberately UNTESTED against the level table): yaw's measured gain is
maneuver-dependent -- level-attitude yaw plateaus ~2.35-2.7 (a ~7.4 rad/s cap the plant does NOT
model) while tumbling yaw tracks the same super-rate form as roll (~3.1 at full stick). Racing yaw
commands are small; the cap gets its own measurement pass if it ever matters.
"""
from __future__ import annotations

import numpy as np

from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant, CtbrPlantConfig
from racer.twin_fit import faithful_config
from racer import rl_plant as rp

# Measured sustained gains (writeup Section 1 table; roll == pitch to 3 digits). 3% is the
# documented fit quality of the one-parameter form (worst point: pitch @0.3, +2.9% -- the "slightly
# flat at the low end").
MEASURED_TABLE = {0.30: 2.505, 1.00: 2.712, 2.00: 3.043, 3.14: 3.50}
REL_TOL = 0.03

_DT = 0.005
_HOV = 0.2656


def _cmd(body_rate, thrust=_HOV):
    return ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.asarray(body_rate, float),
                          thrust=float(thrust))


def _sustained_gain_twin(cfg, axis: int, mag: float, seconds: float = 1.2) -> float:
    """Realized steady rate / commanded rate after a sustained single-axis command."""
    plant = CtbrPlant(cfg)
    br = np.zeros(3)
    br[axis] = mag
    for _ in range(int(round(seconds / _DT))):
        plant.step(_cmd(br), _DT)
    return float(plant.omega[axis] / mag)


def _sustained_gain_rl(params, axis: int, mag: float, seconds: float = 1.2) -> float:
    st = rp.PlantState.hover(params=params)
    a = np.zeros(4)
    a[axis] = mag
    a[3] = params.hover_thrust
    for _ in range(int(round(seconds / _DT))):
        st = rp.step(st, a, _DT, params)
    return float(st.omega[axis] / mag)


def _map_params() -> rp.PlantParams:
    return rp.PlantParams(super_rate_s=rp.SUPER_RATE_S_MEASURED,
                          alpha_max_rps2=rp.ALPHA_MAX_RPS2_MEASURED.copy())


# --------------------------------------------------------------------- 1. the measured gain table
def test_twin_reproduces_measured_gain_table_roll_pitch():
    cfg = faithful_config(super_rate=True)
    for axis in (0, 1):                                   # roll, pitch (rate_sign +1 on both)
        for mag, g_meas in MEASURED_TABLE.items():
            g = _sustained_gain_twin(cfg, axis, mag)
            assert abs(g - g_meas) / g_meas < REL_TOL, (
                f"axis {axis} |cmd|={mag}: model g={g:.4f} vs measured {g_meas} "
                f"({(g - g_meas) / g_meas * 100:+.2f}% > {REL_TOL * 100:.0f}%)")


def test_rl_plant_reproduces_measured_gain_table_roll_pitch():
    params = _map_params()
    for axis in (0, 1):
        for mag, g_meas in MEASURED_TABLE.items():
            g = _sustained_gain_rl(params, axis, mag)
            assert abs(g - g_meas) / g_meas < REL_TOL


def test_yaw_follows_the_same_form():
    # Yaw uses the same map (s_yaw ~= s_roll) on its own G0=2.231. Checked against the FORM (and the
    # mid-amplitude LEVEL measurements 1.00->2.405, 2.00->2.705, which the form matches within ~3%);
    # NOT against the full-stick level plateau (~7.4 rad/s cap) -- the known unmodeled caveat.
    cfg = faithful_config(super_rate=True)
    for mag, g_meas in ((1.00, 2.405), (2.00, 2.705)):
        g = abs(_sustained_gain_twin(cfg, 2, mag))        # rate_sign yaw = -1 -> |.|
        g_form = 2.231 / (1.0 - 0.30 * mag / np.pi)
        assert abs(g - g_form) / g_form < 1e-3            # implements the form exactly
        assert abs(g - g_meas) / g_meas < REL_TOL         # and the form matches the level data here


def test_map_input_clamps_at_pi():
    # |cmd| beyond pi must not grow the gain further: min(|c|, pi) in the map input.
    cfg = faithful_config(super_rate=True)
    g_pi = _sustained_gain_twin(cfg, 0, np.pi)
    g_over = _sustained_gain_twin(cfg, 0, 5.0)
    assert abs(g_pi - 2.501 / 0.70) / (2.501 / 0.70) < 1e-3
    assert g_over < g_pi + 1e-9                           # same gain, just a bigger command


def test_max_omega_clamp_does_not_bite_the_map_ceiling():
    # The map's DC ceiling g(pi)*pi ~= 11.2 rad/s/axis (~18.5 worst-case 3-axis norm) must clear
    # max_omega_rps, or the exact flat-gain ceiling artifact returns. Guard the default.
    cfg = faithful_config(super_rate=True)
    assert cfg.max_omega_rps >= 11.5
    p = CtbrPlant(cfg)
    for _ in range(400):
        p.step(_cmd([np.pi, np.pi, np.pi]), _DT)
    assert float(np.linalg.norm(p.omega)) < cfg.max_omega_rps - 1e-6   # clamp never engaged


# --------------------------------------------------------------------------- 2. slew clamping
def test_slew_clamp_exact_per_step_increment():
    # From rest, a full-stick roll step: unclamped first increment would be
    # alpha*(g(pi)*pi) ~= 0.41*11.2 ~= 4.6 rad/s in 10 ms (460 rad/s^2) -> clamped to exactly
    # alpha_max*dt. The whole rise must respect the per-step cap; the steady state must not.
    dt = 0.01
    cfg = faithful_config(super_rate=True)
    p = CtbrPlant(cfg)
    lim = 260.0 * dt
    prev = 0.0
    increments = []
    for _ in range(60):
        p.step(_cmd([np.pi, 0, 0]), dt)
        increments.append(float(p.omega[0]) - prev)
        prev = float(p.omega[0])
    assert abs(increments[0] - lim) < 1e-12               # first step exactly at the cap
    assert abs(increments[1] - lim) < 1e-12               # still saturated
    assert max(increments) <= lim + 1e-12                 # never exceeds the cap
    assert prev > 11.0                                    # and still reaches the super-rate DC


def test_slew_clamp_is_per_axis():
    # Yaw's cap (80) is independent of roll/pitch's (260).
    dt = 0.01
    p = CtbrPlant(faithful_config(super_rate=True))
    p.step(_cmd([np.pi, 0, np.pi]), dt)
    assert abs(float(p.omega[0]) - 260.0 * dt) < 1e-12
    assert abs(float(p.omega[2]) + 80.0 * dt) < 1e-12     # yaw rate_sign -1 -> clamped negative


def test_rl_plant_slew_matches_twin_bitwise():
    dt = 0.01
    cfg = faithful_config(super_rate=True)
    params = _map_params()
    p = CtbrPlant(cfg)
    st = rp.PlantState.hover(params=params)
    rng = np.random.default_rng(7)
    for _ in range(200):
        a = np.concatenate([rng.uniform(-8, 8, 3), [rng.uniform(0.1, 0.6)]])
        p.step(_cmd(a[:3], a[3]), dt)
        st = rp.step(st, a, dt, params)
        np.testing.assert_array_equal(st.omega, p.omega)  # bit-identical


# --------------------------------------------------------------------------- 3. legacy equivalence
def test_defaults_are_off():
    assert CtbrPlantConfig().super_rate_s is None
    assert CtbrPlantConfig().alpha_max_rps2 is None
    assert rp.PlantParams().super_rate_s is None
    assert rp.PlantParams().alpha_max_rps2 is None
    assert faithful_config().super_rate_s is None         # default faithful stays the legacy plant


def test_off_is_bit_identical_to_legacy_update():
    # With both params None the rate update must produce EXACTLY the legacy floats:
    #   omega' = clip_norm(omega + alpha*(rate_gain*rate_sign*cmd - omega), max_omega)
    dt = 0.01
    cfg = faithful_config()
    p = CtbrPlant(cfg)
    alpha = 1.0 - np.exp(-dt / max(cfg.rate_tau_s, 1e-9))
    rng = np.random.default_rng(3)
    omega = p.omega.copy()
    for _ in range(300):
        a = rng.uniform(-8, 8, 3)
        target = np.asarray(cfg.rate_gain) * np.asarray(cfg.rate_sign) * a
        omega = omega + alpha * (target - omega)          # legacy reference (norm clamp not hit:
        p.step(_cmd(a), dt)                               # |target| <= 2.504*8*sqrt(3) < 25 settles)
        n = float(np.linalg.norm(omega))
        if n > cfg.max_omega_rps:
            omega = omega * (cfg.max_omega_rps / n)
        np.testing.assert_array_equal(p.omega, omega)


def test_s_zero_no_slew_equals_flat_gain():
    # s=0 turns the map into the identity (g = G0/(1-0)); with no slew the trajectory must be
    # bit-identical to the flat-gain plant.
    flat = CtbrPlant(faithful_config())
    szero = faithful_config()
    szero.super_rate_s = 0.0
    mapped = CtbrPlant(szero)
    rng = np.random.default_rng(11)
    for _ in range(200):
        a = rng.uniform(-8, 8, 3)
        th = rng.uniform(0.1, 0.6)
        flat.step(_cmd(a, th), 0.01)
        mapped.step(_cmd(a, th), 0.01)
        np.testing.assert_array_equal(mapped.omega, flat.omega)
        np.testing.assert_array_equal(mapped.pos, flat.pos)


# --------------------------------------------------------------------------- 4. batch semantics
def test_rl_plant_map_batched_matches_loop():
    params = _map_params()
    rng = np.random.default_rng(5)
    B = 16
    pos = rng.uniform(-2, 2, (B, 3)); vel = rng.uniform(-2, 2, (B, 3))
    q = rp.quat_normalize(rng.normal(0, 1, (B, 4)))
    omega = rng.uniform(-1, 1, (B, 3)); thrust = rng.uniform(0.2, 0.5, B)
    batch = rp.PlantState(pos=pos, vel=vel, quat=q, omega=omega, thrust=thrust)
    acts = np.column_stack([rng.uniform(-8, 8, (B, 3)), rng.uniform(0.1, 0.6, B)])
    out = rp.step(batch, acts, 0.02, params)
    for i in range(B):
        si = rp.PlantState(pos=pos[i], vel=vel[i], quat=q[i], omega=omega[i],
                           thrust=np.asarray(thrust[i]))
        oi = rp.step(si, acts[i], 0.02, params)
        np.testing.assert_allclose(out.omega[i], oi.omega, atol=1e-12)
        np.testing.assert_allclose(out.pos[i], oi.pos, atol=1e-12)


def test_map_with_transport_delay_uses_the_delayed_command():
    # The map reads the APPLIED (post-delay) command, not the issued one: with k=2 steps of delay
    # and a big step issued at t=0 into a hover-seeded buffer, omega stays at rest for k steps.
    params = rp.PlantParams(super_rate_s=0.30, alpha_max_rps2=np.array([260.0, 260.0, 80.0]),
                            transport_delay_steps=2)
    st = rp.PlantState.hover(params=params)               # buffer seeded with hover (zero rates)
    a = np.array([np.pi, 0.0, 0.0, params.hover_thrust])
    st = rp.step(st, a, 0.01, params)
    assert float(np.abs(st.omega).max()) == 0.0           # hover cmd still applying
    st = rp.step(st, a, 0.01, params)
    assert float(np.abs(st.omega).max()) == 0.0
    st = rp.step(st, a, 0.01, params)
    assert abs(float(st.omega[0]) - 2.6) < 1e-12          # the step arrives, slew-clamped
