"""Anchor tests for the S17 MOTOR-MIXER coupling model (live-deploy diag 2026-06-11,
``handoff/shadowpc-live-deploy-diag-2026-06-11/WRITEUP.md`` Sections 2 + 8; constants fit in
``handoff/laptop-s17-mixer-inc6-2026-06-11/fit_mixer.py``).

The model: per-motor commands = collective +- rate-loop differential demands, clipped to
[idle, 1]; the clipped motor MEAN re-enters the thrust map (parasitic lift at the bottom rail,
thrust sag at the top), the clipped DIFFERENTIAL scales the slew limit by Q = r/r_fit.

Anchors reproduced here (the measured rows the model was fit to + the rails that broke the
inc4/inc5 live transfers):
  * thr 0 + yaw 3.14 (transient)  -> motors ~[idle, 0.73, 0.73, idle] (measured [.08,.73,.73,.08])
  * thr 0 + yaw 3.14 (sustained)  -> mean lift ~hover (measured a_up 9.36 m/s^2 at commanded ZERO)
  * thr 0 + rates 0               -> free fall (collective honored; K(idle) = 0)
  * thr 0 + roll+pitch 1.5        -> max motor ~0.64 (measured 0.64)
  * hover + yaw 3.14              -> mean ~preserved (measured 0.287; the zeta_yaw anchor)
  * collective 1.0 + rate demand  -> zero up-headroom: one-sided differential (authority
                                     degraded), motor mean DROOPS (thrust sag; c100 mean 0.874)
  * benign envelope (no clipping) -> bit-identical to the mixer-OFF plant (mean preserved,
                                     slew never binding)
  * mixer OFF                     -> exact legacy step (pinned vs git-HEAD behavior via the
                                     test_measured_aero inline-legacy test; here: None == before)
"""
from __future__ import annotations

import numpy as np
import pytest

from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant, _mixer_r_fit
from racer.twin_fit import faithful_config
from racer import rl_plant as rp

_DT = 1.0 / 30.0          # the policy/probe control dt


def _mixer_params(**kw) -> rp.PlantParams:
    """The fully measured plant (map + aero + mixer), rl_plant form."""
    base = dict(
        super_rate_s=rp.SUPER_RATE_S_MEASURED,
        alpha_max_rps2=rp.ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0,
        quad_drag_c2=rp.QUAD_DRAG_C2_MEASURED.copy(),
        coll_map_thr=rp.COLL_MAP_THR_MEASURED.copy(),
        coll_map_accel=rp.COLL_MAP_ACCEL_MEASURED.copy(),
        mixer_idle=rp.MIXER_IDLE_MEASURED,
        mixer_kappa_err=rp.MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=rp.MIXER_KAPPA_HOLD_MEASURED,
        mixer_zeta_yaw=rp.MIXER_ZETA_YAW_MEASURED,
    )
    base.update(kw)
    return rp.PlantParams(**base)


def _motors(params: rp.PlantParams, c, target, omega):
    """The mixer motor set / mean / per-axis realized differential for one condition --
    replicates the step()'s internal arithmetic (kept in sync by the step-level anchors below)."""
    e = np.asarray(target, float) - np.asarray(omega, float)
    d = params.mixer_kappa_err * e + params.mixer_kappa_hold * np.asarray(omega, float)
    eta = params.mixer_zeta_yaw / (params.mixer_zeta_yaw + max(float(c), 0.0))
    d = np.concatenate([d[0:2], d[2:3] * eta])
    u = np.array([
        np.clip(c + (d[0] + d[1] + d[2]), params.mixer_idle, 1.0),
        np.clip(c + (-d[0] + d[1] - d[2]), params.mixer_idle, 1.0),
        np.clip(c + (d[0] - d[1] - d[2]), params.mixer_idle, 1.0),
        np.clip(c + (-d[0] - d[1] + d[2]), params.mixer_idle, 1.0),
    ])
    s = np.array([[1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float)
    delta = (s @ u) / 4.0
    return u, float(u.mean()), delta, d


def _target(params: rp.PlantParams, cmd: np.ndarray) -> np.ndarray:
    gain = params.rate_gain / (1.0 - params.super_rate_s
                               * np.minimum(np.abs(cmd), np.pi) / np.pi)
    return gain * params.rate_sign * cmd


def _K(c: float) -> float:
    return float(np.interp(c, rp.COLL_MAP_THR_MEASURED, rp.COLL_MAP_ACCEL_MEASURED))


# --------------------------------------------------------------------------- measured anchors
def test_yaw_rail_transient_motor_pattern():
    """thr 0 + yaw 3.14, omega=0 (the dither regime the policy lives in): the high pair sits at
    kappa_err * target ~= 0.73 -- the measured [0.08, 0.73, 0.73, 0.08] pattern."""
    p = _mixer_params()
    tgt = _target(p, np.array([0.0, 0.0, 3.14]))
    u, mean, delta, d = _motors(p, 0.0, tgt, np.zeros(3))
    hi = np.sort(u)[2:]
    lo = np.sort(u)[:2]
    assert hi == pytest.approx([0.7304, 0.7304], abs=0.02)       # measured 0.73
    assert lo == pytest.approx([p.mixer_idle] * 2, abs=1e-12)    # measured 0.08 (idle + riding)
    # the parasitic mean: ~0.39 -> ~2 g through the knot table -- the live "2 g transient"
    assert 0.36 < mean < 0.42
    assert _K(mean) == pytest.approx(20.5, abs=2.5)              # ~2.1 g uncommanded lift
    # realized yaw differential is one-sided-limited: |delta_y| < |d_y| (authority throttled)
    assert abs(delta[2]) < abs(d[2])


def test_yaw_rail_settled_lift_anchor():
    """thr 0 + yaw 3.14, omega settled at target: mean ~(0.46 + idle)/2 = 0.255 -> K ~= 8.8 m/s^2
    (measured mean 0.258, a_up 9.36 phase-avg incl. the higher transient)."""
    p = _mixer_params()
    tgt = _target(p, np.array([0.0, 0.0, 3.14]))
    u, mean, _, _ = _motors(p, 0.0, tgt, tgt.copy())             # e = 0, hold term only
    assert mean == pytest.approx(0.255, abs=0.01)
    assert _K(mean) == pytest.approx(9.36, abs=1.5)              # measured 9.36 (HOVER-class lift)


def test_free_fall_at_all_zero():
    """thr 0 + rates 0 from rest: c_eff = idle, K(idle) = 0 -> exact free fall (measured a_up
    0.64 ~ 0; the diag's 'collective 0 honored' row). Step-level through the twin."""
    cfg = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    plant = CtbrPlant(cfg)
    plant.step(ControlCommand(mode=ControlMode.BODY_RATE,
                              body_rate=np.zeros(3), thrust=0.0), _DT)
    # first step from rest: no drag (v=0), a = +g exactly (NED +Z down)
    assert plant.vel[2] == pytest.approx(cfg.g * _DT, abs=1e-12)
    assert plant.vel[0] == 0.0 and plant.vel[1] == 0.0


def test_rp_rail_max_motor_anchor():
    """thr 0 + roll+pitch 1.5 transient: the +/+ motor = kappa_err*(t_r + t_p) ~= 0.64 (measured
    0.64 -- the second independent kappa_err fit)."""
    p = _mixer_params()
    tgt = _target(p, np.array([1.5, 1.5, 0.0]))
    u, mean, _, _ = _motors(p, 0.0, tgt, np.zeros(3))
    assert u.max() == pytest.approx(0.64, abs=0.01)
    # parasitic mean exists (the tumbling z00_rp15 row measured 0.243 phase-avg)
    assert mean > 0.15


def test_hover_yaw_rail_mean_preservation():
    """hover collective + yaw 3.14 settled: mean ~0.287 (measured 0.287 -- the zeta_yaw anchor;
    'mean preserved when headroom exists')."""
    p = _mixer_params()
    tgt = _target(p, np.array([0.0, 0.0, 3.14]))
    u, mean, _, _ = _motors(p, 0.2656, tgt, tgt.copy())
    assert mean == pytest.approx(0.287, abs=0.012)


def test_c100_top_rail_mean_droop():
    """collective 1.0 + a modest differential demand (the c100 re-level row): high side clips at
    1.0 -> mean droops to ~0.875 (measured 0.874), realized differential halves."""
    p = _mixer_params()
    # choose a roll error that demands d ~= 0.25 (the diag's re-level magnitude)
    e_roll = 0.25 / p.mixer_kappa_err
    u, mean, delta, d = _motors(p, 1.0, np.array([e_roll, 0.0, 0.0]), np.zeros(3))
    assert mean == pytest.approx(0.875, abs=0.005)               # measured 0.874
    assert u.max() == 1.0                                        # top-clipped pair
    assert delta[0] / d[0] == pytest.approx(0.5, abs=0.01)       # one-sided: half authority


def test_top_rail_authority_collapse_step_level():
    """collective 1.0 + full-stick roll: the mixer-ON plant turns strictly slower than mixer-OFF
    (zero up-headroom -> one-sided differential -> Q < 1 throttles the slew), and the thrust
    SAGS (c_eff < 1 -> a_up well below the commanded K(1.0) = 78.3)."""
    cfg_on = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    cfg_off = faithful_config(super_rate=True, measured_aero=True)
    cmd = ControlCommand(mode=ControlMode.BODY_RATE,
                         body_rate=np.array([3.14, 0.0, 0.0]), thrust=1.0)
    on, off = CtbrPlant(cfg_on), CtbrPlant(cfg_off)
    on.step(cmd, _DT)
    off.step(cmd, _DT)
    # the RISE is torque-limited: Q = r/r_fit ~= 0.86 at the top rail (one-sided differential)
    # vs Q = 1 in the OFF plant -- both slew-clamped, ON strictly slower
    assert abs(on.omega[0]) < abs(off.omega[0]) * 0.90
    for _ in range(5):                                           # ~0.2 s of a "thrust pulse"
        on.step(cmd, _DT)
        off.step(cmd, _DT)
    # thrust sag: the sustained hold differential (kappa_hold * omega ~ 0.52) keeps the top
    # pair clipped -> c_eff ~ 0.74 -> a_up ~ 53 vs the commanded K(1.0) = 78.3
    assert on.vel[2] > off.vel[2] + 0.05                         # NED: larger +z vel = less lift


def test_bottom_rail_parasitic_climb_step_level():
    """thr 0 + max yaw dither (the inc5 live failure signature): the mixer-ON plant LIFTS
    (~hover-class specific force at commanded zero) while the mixer-OFF plant free-falls --
    the twin blind spot that produced 'climbing +3.2 m/s at collective 0.000' live."""
    cfg_on = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    cfg_off = faithful_config(super_rate=True, measured_aero=True)
    on, off = CtbrPlant(cfg_on), CtbrPlant(cfg_off)
    yaw = 3.14
    for _ in range(15):                                          # 0.5 s of per-tick sign dither
        cmd = ControlCommand(mode=ControlMode.BODY_RATE,
                             body_rate=np.array([0.0, 0.0, yaw]), thrust=0.0)
        on.step(cmd, _DT)
        off.step(cmd, _DT)
        yaw = -yaw
    # OFF: free fall, ~+4.9 m/s sink. ON: strongly supported (the dither keeps |e| maxed).
    assert off.vel[2] > 4.0
    assert on.vel[2] < 1.0                                       # parasitic lift cancels most of g
    assert on.vel[2] < off.vel[2] - 3.5


def test_benign_envelope_matches_mixer_off():
    """Unclipped flight (hover-band collective, small rates): u never clips, so the rate path is
    BIT-identical to mixer-OFF (the raised slew limit never binds) and the translation path
    agrees to machine epsilon (mean(clip(c +- d)) reconstructs c to ~1 ULP -- four float adds --
    so a_up wiggles in the last bit only). The mixer cannot perturb the validated benign
    envelope beyond float rounding."""
    cfg_on = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    cfg_off = faithful_config(super_rate=True, measured_aero=True)
    on, off = CtbrPlant(cfg_on), CtbrPlant(cfg_off)
    rng = np.random.default_rng(7)
    for _ in range(200):
        cmd = ControlCommand(mode=ControlMode.BODY_RATE,
                             body_rate=rng.normal(0.0, 0.05, 3),
                             thrust=0.2656 + float(rng.normal(0.0, 0.02)))
        on.step(cmd, _DT)
        off.step(cmd, _DT)
    assert np.allclose(on.pos, off.pos, atol=1e-9)
    assert np.allclose(on.vel, off.vel, atol=1e-9)
    assert np.allclose(on.q, off.q, atol=1e-12)
    assert np.allclose(on.omega, off.omega, atol=1e-12)


# --------------------------------------------------------------------------- model invariants
def test_r_fit_values():
    """The authority normalisation at the S14 slew-fit condition (hover, single-axis pi):
    roll/pitch ~0.580, yaw ~0.763 (fit_mixer.py). alpha_max was measured WITH the mixer
    throttling there, so Q = r/r_fit keeps the fit point exact."""
    p = _mixer_params()
    rf = p._mixer_r_fit
    assert rf[0] == pytest.approx(0.5797, abs=0.002)
    assert rf[1] == pytest.approx(0.5790, abs=0.002)
    assert rf[2] == pytest.approx(0.7627, abs=0.002)
    # twin's duplicate produces the identical floats (parity-pinned arithmetic)
    cfg = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    rf_twin = _mixer_r_fit(cfg.rate_gain, cfg.rate_sign, cfg.super_rate_s, cfg.hover_thrust,
                           cfg.mixer_idle, cfg.mixer_kappa_err, cfg.mixer_zeta_yaw)
    assert np.array_equal(rf, rf_twin)


def test_slew_fit_point_preserved():
    """At the S14 fit condition (hover collective, single-axis pi step from rest) the mixer-ON
    rise is slew-limited at ~alpha_max -- the measured 260/80 rad/s^2 stay exact (Q == 1 at the
    fit point by construction)."""
    cfg_on = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    cfg_off = faithful_config(super_rate=True, measured_aero=True)
    for axis, amax in ((0, 260.0), (2, 80.0)):
        cmd_v = np.zeros(3)
        cmd_v[axis] = np.pi
        on, off = CtbrPlant(cfg_on), CtbrPlant(cfg_off)
        on._thrust = off._thrust = 0.2656
        cmd = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=cmd_v, thrust=0.2656)
        on.step(cmd, _DT)
        off.step(cmd, _DT)
        # first-step increment equals the OFF plant's slew-clamped increment (both = amax*dt)
        assert on.omega[axis] == pytest.approx(off.omega[axis], rel=1e-9)
        assert abs(on.omega[axis]) == pytest.approx(amax * _DT, rel=1e-6)


def test_mixer_params_validation():
    with pytest.raises(ValueError, match="set together"):
        rp.PlantParams(mixer_idle=0.05, alpha_max_rps2=np.array([260.0, 260.0, 80.0]))
    with pytest.raises(ValueError, match="alpha_max"):
        rp.PlantParams(mixer_idle=0.05, mixer_kappa_err=0.073, mixer_kappa_hold=0.046,
                       mixer_zeta_yaw=0.34)
    with pytest.raises(ValueError, match="super_rate"):
        faithful_config(mixer=True)


def test_mixer_off_params_unchanged():
    """All four mixer fields None -> PlantParams behaves exactly as before (no derived field,
    no validation side effects); the OFF step path is pinned bit-identical by
    test_measured_aero's inline-legacy test."""
    p = rp.PlantParams()
    assert p.mixer_idle is None and p._mixer_r_fit is None
    st = rp.PlantState.hover()
    a = rp.hover_action(p)
    out = rp.step(st, a, _DT, p)
    assert np.allclose(out.pos, [0.0, 0.0, 0.0], atol=1e-12)     # hover holds


def test_batch_matches_loop_mixer():
    """The batched mixer step == per-element loop (the DiffAero vectorisation property),
    exercised across both rails."""
    p = _mixer_params()
    rng = np.random.default_rng(11)
    B = 12
    pos = rng.uniform(-2, 2, (B, 3)); vel = rng.uniform(-4, 4, (B, 3))
    q = rp.quat_normalize(rng.normal(0, 1, (B, 4)))
    omega = rng.uniform(-6, 6, (B, 3))
    thrust = np.concatenate([np.zeros(4), np.ones(4), rng.uniform(0.1, 0.9, B - 8)])
    acts = np.column_stack([rng.uniform(-3.2, 3.2, (B, 3)),
                            np.concatenate([np.zeros(4), np.ones(4),
                                            rng.uniform(0.0, 1.0, B - 8)])])
    batch = rp.PlantState(pos=pos, vel=vel, quat=q, omega=omega, thrust=thrust)
    out = rp.step(batch, acts, _DT, p)
    for i in range(B):
        si = rp.PlantState(pos=pos[i], vel=vel[i], quat=q[i], omega=omega[i],
                           thrust=np.asarray(thrust[i]))
        oi = rp.step(si, acts[i], _DT, p)
        assert np.allclose(out.pos[i], oi.pos, atol=1e-12)
        assert np.allclose(out.vel[i], oi.vel, atol=1e-12)
        assert np.allclose(out.omega[i], oi.omega, atol=1e-12)
        assert np.allclose(out.thrust[i], oi.thrust, atol=1e-12)
