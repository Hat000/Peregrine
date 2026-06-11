"""Measured-aero integration (twin-falsify campaign 2026-06-10/11) -- the OBJECTIVE ANCHORS.

The campaign (``handoff/shadowpc-twin-falsify-2026-06-10/WRITEUP.md``; 40+ recordings, predictions
committed pre-flight) FALSIFIED the twin's aero: drag is body-frame QUADRATIC and direction-
dependent (not linear 0.2111/s world-isotropic) and the collective->accel map is CONVEX (not
linear g*thr/hover; full stick 78.3 m/s^2 ~= 2.1x linear). These tests pin the integration of that
model into twin.py / rl_plant.py:

  1. drag anchors -- per-direction one-step decel == c2 * v^2 (the Section 2 per-family coast
     fits: 0.042 nose-first / 0.058 tail-first / 0.055 lateral; 0.0756 climb / 0.0539 descend),
     the Section 1 headline (at 9 m/s: measured ~4.2 m/s^2 vs the legacy twin's 1.9), and
     body-frame-ness (a yawed / inverted attitude picks the airflow-correct coefficient);
  2. the coast-replay CHECK as a closed-form surrogate -- the campaign's recordings are
     ShadowPC-local (data/runs, gitignored; NOT on this machine), so the WRITEUP's replay numbers
     (speed RMS legacy 0.81 -> candidate 0.24-0.29 m/s over 9 coast runs) cannot be reproduced
     here; instead the integrated plant is verified against the exact quad-coast solution
     v(t) = v0 / (1 + c2*v0*t) those fits imply, plus the falsification's SHAPE claims (legacy
     over-brakes below the ~4.06 m/s crossover, under-brakes 2.2x at 9 m/s);
  3. collective anchors -- the knot table through the plant (incl. 78.2828 m/s^2 full stick),
     the Section 3 measured/linear ratio column (0.38 @0.15 ... 2.12 @1.0), sub-linear below
     hover / convex above, end-knot clamping;
  4. defaults OFF + the aero-OFF step bit-identical to the legacy update (inline reference);
  5. config validation, batch semantics, twin == rl_plant spot parity (the full battery lives in
     ``test_rl_plant_parity.py``; the torch mirror's authoritative check is the config-matrix
     gate ``rl/check_diffaero_gate.py`` via ``rl/local_gate_harness.py`` / Adroit).

The torch-side helpers (``_t_interp1d`` bitwise vs the numpy reference; the dr_aero per-env
sampling bands + hover pinning) are tested at the end, skipped when torch is absent.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant, CtbrPlantConfig, _quad_c2_table, _wxyz_from_euler
from racer.twin_fit import faithful_config
from racer import rl_plant as rp

_G = 9.80665
_HOV = 0.2656
_DT = 1e-3            # one-step probes: semi-implicit Euler makes (v0 - v1)/dt == accel(v0) exactly

# WRITEUP Section 2/3 nominals (what the canonical constants must carry)
_C2_NOSE, _C2_TAIL, _C2_LAT = 0.042, 0.058, 0.055
_C2_DN, _C2_UP = 0.0539309301924107, 0.0756168595765378
_C2_POOLED = 0.052
# WRITEUP Section 3 ratio column: K(thr) / (g*thr/hover), rounded to 2 decimals there
_RATIO_TABLE = {0.15: 0.38, 0.20: 0.64, 0.2656: 0.98, 0.32: 1.15,
                0.40: 1.47, 0.55: 1.91, 0.80: 1.98, 1.00: 2.12}


def _cmd(body_rate, thrust):
    return ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.asarray(body_rate, float),
                          thrust=float(thrust))


def _aero_cfg(**overrides):
    cfg = faithful_config(super_rate=True, measured_aero=True)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _one_step_accel(cfg, vel, q_wxyz=None, thrust=_HOV):
    """World accel over one tiny step from rest-rate: exact accel(state0) by semi-implicit Euler."""
    p = CtbrPlant(cfg, velocity_ned=np.asarray(vel, float), q_wxyz=q_wxyz)
    p.step(_cmd(np.zeros(3), thrust), _DT)
    return (p.vel - np.asarray(vel, float)) / _DT


def _trim_thrust(cfg) -> float:
    """The collective where the knot map crosses g (the measured trim, ~0.2687 -- the hover knot
    reads 9.58, 2.3% under g, so trim sits slightly above the 0.2656 knot)."""
    thr = np.asarray(cfg.coll_map_thr, float)
    acc = np.asarray(cfg.coll_map_accel, float)
    i = int(np.searchsorted(acc, _G) - 1)             # acc strictly increasing on the fit region
    t = thr[i] + (thr[i + 1] - thr[i]) * (_G - acc[i]) / (acc[i + 1] - acc[i])
    assert abs(float(np.interp(t, thr, acc)) - _G) < 1e-9
    return float(t)


# ------------------------------------------------------------------ defaults / config wiring
def test_defaults_are_off_everywhere():
    for cfg in (CtbrPlantConfig(), faithful_config(), faithful_config(super_rate=True)):
        assert cfg.quad_drag_c2 is None
        assert cfg.coll_map_thr is None
        assert cfg.coll_map_accel is None
    p = rp.PlantParams()
    assert p.quad_drag_c2 is None and p.coll_map_thr is None and p.coll_map_accel is None
    assert p.linear_drag == 0.2111                       # legacy drag untouched


def test_faithful_config_measured_aero_wiring():
    cfg = faithful_config(measured_aero=True)
    assert cfg.linear_drag == 0.0                        # quad drag REPLACES linear (d1=0 pure-quad)
    np.testing.assert_array_equal(cfg.quad_drag_c2, rp.QUAD_DRAG_C2_MEASURED)
    np.testing.assert_array_equal(cfg.coll_map_thr, rp.COLL_MAP_THR_MEASURED)
    np.testing.assert_array_equal(cfg.coll_map_accel, rp.COLL_MAP_ACCEL_MEASURED)
    # rate loop untouched by the aero flag (the campaign's survivals stay exactly as-is)
    base = faithful_config()
    assert cfg.hover_thrust == base.hover_thrust and cfg.rate_tau_s == base.rate_tau_s
    np.testing.assert_array_equal(cfg.rate_gain, base.rate_gain)
    np.testing.assert_array_equal(cfg.rate_sign, base.rate_sign)
    # the canonical constants carry the WRITEUP nominals
    np.testing.assert_array_equal(rp.QUAD_DRAG_C2_MEASURED,
                                  [[_C2_NOSE, _C2_TAIL], [_C2_LAT, _C2_LAT], [_C2_DN, _C2_UP]])
    # bottom two knots: Section 7 extrapolation FLOORED at 0 (free fall at idle, Section 8's c000)
    assert rp.COLL_MAP_ACCEL_MEASURED[0] == 0.0 and rp.COLL_MAP_ACCEL_MEASURED[1] == 0.0
    assert np.all(np.diff(rp.COLL_MAP_THR_MEASURED) > 0)
    # fit-grade region strictly increasing (0.15 .. 1.0)
    assert np.all(np.diff(rp.COLL_MAP_ACCEL_MEASURED[2:]) > 0)


def test_aero_off_step_is_bit_identical_to_legacy():
    # Full independent reimplementation of the legacy step (rate lag -> attitude -> linear
    # translation); the default-config plant must reproduce it EXACTLY, float for float.
    cfg = faithful_config(super_rate=True)               # aero OFF; map ON (the pre-S16 best)
    dt = 0.01
    p = CtbrPlant(cfg, velocity_ned=np.array([3.0, -2.0, 1.0]))
    q = p.q.copy()
    omega = np.zeros(3)
    vel = p.vel.copy()
    pos = p.pos.copy()
    rng = np.random.default_rng(17)
    alpha = 1.0 - np.exp(-dt / max(cfg.rate_tau_s, 1e-9))
    for _ in range(300):
        a = rng.uniform(-8, 8, 3)
        th = float(rng.uniform(0.0, 1.0))
        p.step(_cmd(a, th), dt)
        gain = np.asarray(cfg.rate_gain) / (1.0 - np.asarray(cfg.super_rate_s)
                                            * np.minimum(np.abs(a), np.pi) / np.pi)
        domega = alpha * (gain * np.asarray(cfg.rate_sign) * a - omega)
        domega = np.clip(domega, -np.asarray(cfg.alpha_max_rps2) * dt,
                         np.asarray(cfg.alpha_max_rps2) * dt)
        omega = omega + domega                           # norm clamp not reached (|target| < 12)
        R_cur = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        R_new = R_cur * Rotation.from_rotvec(omega * dt)
        x, y, z, w = R_new.as_quat()
        q = np.array([w, x, y, z])
        a_up = cfg.g * (th / cfg.hover_thrust)
        f = R_new.as_matrix() @ np.array([0.0, 0.0, -a_up]) - cfg.linear_drag * vel
        vel = vel + (f + np.array([0.0, 0.0, cfg.g])) * dt
        pos = pos + vel * dt
        np.testing.assert_array_equal(p.omega, omega)
        np.testing.assert_array_equal(p.q, q)
        np.testing.assert_array_equal(p.vel, vel)
        np.testing.assert_array_equal(p.pos, pos)


# ------------------------------------------------------------------ 1. drag anchors (Section 2)
def test_drag_decel_per_direction_anchors():
    cfg = _aero_cfg()
    for v, expect in [([9.0, 0, 0], _C2_NOSE * 81.0),     # nose-first
                      ([-9.0, 0, 0], _C2_TAIL * 81.0),    # tail-first
                      ([0, 9.0, 0], _C2_LAT * 81.0),      # lateral +y
                      ([0, -9.0, 0], _C2_LAT * 81.0)]:    # lateral -y (symmetric within 2%)
        acc = _one_step_accel(cfg, v)
        along = -np.dot(acc, np.asarray(v) / 9.0)         # decel along track
        assert abs(along - expect) < 1e-9, (v, along, expect)
    # speed scaling is quadratic: decel(6)/decel(3) == 4 exactly
    d6 = -_one_step_accel(cfg, [6.0, 0, 0])[0]
    d3 = -_one_step_accel(cfg, [3.0, 0, 0])[0]
    assert abs(d6 / d3 - 4.0) < 1e-9


def test_headline_braking_at_9ms():
    # Section 1: "at 9 m/s real drag is ~4.2 m/s^2 vs the twin's 1.9 -- 2.2x wrong".
    iso = _aero_cfg(quad_drag_c2=_C2_POOLED)              # pooled isotropic candidate
    d_iso = -_one_step_accel(iso, [9.0, 0, 0])[0]
    assert round(d_iso, 1) == 4.2
    legacy = faithful_config()
    d_leg = -_one_step_accel(legacy, [9.0, 0, 0], thrust=legacy.hover_thrust)[0]
    assert round(d_leg, 1) == 1.9
    assert d_iso / d_leg > 2.0                            # the 2.2x under-braking falsifier


def test_drag_is_body_frame():
    cfg = _aero_cfg()
    # yawed 90 deg: world +x motion is BODY-LATERAL -> picks 0.055, not the nose 0.042
    q_yaw = _wxyz_from_euler(0.0, 0.0, np.pi / 2)
    d = -_one_step_accel(cfg, [9.0, 0, 0], q_wxyz=q_yaw)[0]
    assert abs(d - _C2_LAT * 81.0) < 1e-9
    # and the world-frame decel direction is still along -x (drag opposes velocity)
    acc = _one_step_accel(cfg, [9.0, 0, 0], q_wxyz=q_yaw)
    assert abs(acc[1]) < 1e-9 and acc[0] < 0


def test_vertical_drag_split_and_inversion():
    cfg = _aero_cfg()
    # thrust = 0 -> K(0) = 0 (floored bottom knot: free fall at idle), so a_z = g + drag only
    up = _one_step_accel(cfg, [0, 0, -5.0], thrust=0.0)   # climbing at 5 m/s
    assert abs(up[2] - (_G + _C2_UP * 25.0)) < 1e-9       # climb opposed by the 0.0756 coefficient
    dn = _one_step_accel(cfg, [0, 0, 5.0], thrust=0.0)    # descending at 5 m/s
    assert abs(dn[2] - (_G - _C2_DN * 25.0)) < 1e-9       # descent opposed by 0.0539
    # INVERTED (roll pi) while climbing: airflow hits the inverted belly == upright-descend
    # geometry -> the body-frame split must pick c_dn, not c_up
    q_inv = _wxyz_from_euler(np.pi, 0.0, 0.0)
    up_inv = _one_step_accel(cfg, [0, 0, -5.0], q_wxyz=q_inv, thrust=0.0)
    assert abs(up_inv[2] - (_G + _C2_DN * 25.0)) < 1e-9


# ------------------------------------------------------- 2. coast replay surrogate (Section 5)
def test_coast_replay_surrogate_closed_form():
    """The WRITEUP's coast replays (speed RMS legacy 0.810 -> candidate 0.240-0.289 m/s) ran
    against ShadowPC-local recordings that are NOT on this machine; this is the surrogate: the
    integrated plant must track the exact quad-coast solution v(t) = v0/(1 + c2*v0*t) that the
    Section 2 fits imply, per direction family, over a 6 s coast at the live 100 Hz step."""
    cfg = _aero_cfg()
    thr = _trim_thrust(cfg)                               # a_up == g -> level, pure-drag coast
    dt, T = 0.01, 6.0
    for dirvec, c2, v0 in [(np.array([1.0, 0, 0]), _C2_NOSE, 7.5),
                           (np.array([-1.0, 0, 0]), _C2_TAIL, 7.5),
                           (np.array([0, 1.0, 0]), _C2_LAT, 6.5)]:
        p = CtbrPlant(cfg, velocity_ned=dirvec * v0)
        errs = []
        for k in range(int(T / dt)):
            p.step(_cmd(np.zeros(3), thr), dt)
            v_model = float(np.hypot(p.vel[0], p.vel[1]))
            v_exact = v0 / (1.0 + c2 * v0 * (k + 1) * dt)
            errs.append(v_model - v_exact)
        errs = np.asarray(errs)
        rms = float(np.sqrt(np.mean(errs**2)))
        assert rms < 0.02, (dirvec, rms)                  # integration error only (~5e-3 observed)
        assert abs(errs[-1]) < 0.02
        assert abs(float(p.vel[2])) < 1e-6                # stays level: trim cancels gravity


def test_legacy_overbrakes_low_underbrakes_high():
    """The falsification's SHAPE: legacy linear 0.2111/s vs measured quad. Crossover at
    d1/c2 = 0.2111/0.052 ~= 4.06 m/s; legacy ~2x OVER-braking at 2 m/s, ~2.2x UNDER at 9."""
    iso = _aero_cfg(quad_drag_c2=_C2_POOLED)
    legacy = faithful_config()

    def decel(cfg, v, thrust):
        return -_one_step_accel(cfg, [v, 0, 0], thrust=thrust)[0]

    thr = _trim_thrust(iso)
    assert decel(legacy, 9.0, _HOV) < 0.5 * decel(iso, 9.0, thr)      # under-brakes at speed
    assert decel(legacy, 2.0, _HOV) > 1.9 * decel(iso, 2.0, thr)      # over-brakes when slow
    v_x = 0.2111 / _C2_POOLED
    assert abs(decel(legacy, v_x, _HOV) - decel(iso, v_x, thr)) < 0.02 * decel(iso, v_x, thr)


# ------------------------------------------------------------- 3. collective anchors (Section 3)
def test_collective_knot_table_through_the_plant():
    cfg = _aero_cfg()
    for thr, k_expect in zip(rp.COLL_MAP_THR_MEASURED, rp.COLL_MAP_ACCEL_MEASURED):
        acc = _one_step_accel(cfg, [0, 0, 0], thrust=float(thr))
        k_meas = _G - acc[2]                              # level: a_z = g - K(thr)
        assert abs(k_meas - k_expect) < 1e-9, (thr, k_meas, k_expect)


def test_collective_ratio_column_matches_writeup():
    # the Section 3 measured/linear ratio column, to the 2 decimals the WRITEUP quotes
    for thr, ratio in _RATIO_TABLE.items():
        k = float(np.interp(thr, rp.COLL_MAP_THR_MEASURED, rp.COLL_MAP_ACCEL_MEASURED))
        lin = _G * thr / _HOV
        assert abs(k / lin - ratio) < 0.005, (thr, k / lin, ratio)
    # headline: full stick 78.3 m/s^2, 2.12x the linear model's 36.9
    k_full = float(np.interp(1.0, rp.COLL_MAP_THR_MEASURED, rp.COLL_MAP_ACCEL_MEASURED))
    assert round(k_full, 1) == 78.3
    assert round(_G * 1.0 / _HOV, 1) == 36.9


def test_collective_convex_above_sublinear_below():
    thr = rp.COLL_MAP_THR_MEASURED
    k = rp.COLL_MAP_ACCEL_MEASURED
    lin = _G * thr / _HOV
    below = thr <= 0.20
    above = thr >= 0.32
    assert np.all(k[below] < lin[below] + 1e-12)          # sub-linear below hover
    assert np.all(k[above] > lin[above])                  # convex (above-linear) above hover
    # interp between knots is linear: midpoint of [0.2656, 0.32]
    mid = float(np.interp((0.2656 + 0.32) / 2, thr, k))
    assert abs(mid - (9.5804078429104393 + 13.576760297067407) / 2) < 1e-12


def test_collective_end_clamps():
    cfg = _aero_cfg()
    # beyond the last knot: the [0,1] stick saturates -> hold the full-stick value
    acc = _one_step_accel(cfg, [0, 0, 0], thrust=1.2)
    assert abs((_G - acc[2]) - 78.282838504684648) < 1e-9
    # below the first knot (negative command never happens live; clamps to the floored 0)
    acc = _one_step_accel(cfg, [0, 0, 0], thrust=-0.1)
    assert abs((_G - acc[2]) - 0.0) < 1e-9


# ------------------------------------------------------------------ 4/5. plumbing + parity
def test_config_validation():
    with pytest.raises(ValueError):
        rp.PlantParams(coll_map_thr=np.array([0.0, 1.0]))               # lone knot array
    with pytest.raises(ValueError):
        rp.PlantParams(coll_map_thr=np.array([0.0, 0.5, 0.5]),          # non-increasing
                       coll_map_accel=np.array([0.0, 5.0, 9.0]))
    with pytest.raises(ValueError):
        rp.PlantParams(quad_drag_c2=np.ones((2, 2)))                    # bad shape
    with pytest.raises(ValueError):
        _quad_c2_table(np.ones(4))
    p = CtbrPlant(_aero_cfg(coll_map_accel=None))                       # lone twin knot array
    with pytest.raises(ValueError):
        p.step(_cmd(np.zeros(3), _HOV), 0.01)
    # conveniences normalise: scalar and (3,) both become (3, 2)
    assert rp.PlantParams(quad_drag_c2=0.052).quad_drag_c2.shape == (3, 2)
    np.testing.assert_array_equal(rp.PlantParams(quad_drag_c2=[0.04, 0.05, 0.06]).quad_drag_c2,
                                  [[0.04, 0.04], [0.05, 0.05], [0.06, 0.06]])


def test_interp1d_matches_np_interp_bitwise():
    rng = np.random.default_rng(23)
    xp = rp.COLL_MAP_THR_MEASURED
    fp = rp.COLL_MAP_ACCEL_MEASURED
    x = np.concatenate([rng.uniform(-0.5, 1.8, 500), xp.copy(), [-1.0, 0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(rp._interp1d(x, xp, fp), np.interp(x, xp, fp))


def test_rl_plant_aero_batched_matches_loop():
    params = rp.PlantParams(linear_drag=0.0,
                            quad_drag_c2=rp.QUAD_DRAG_C2_MEASURED.copy(),
                            coll_map_thr=rp.COLL_MAP_THR_MEASURED.copy(),
                            coll_map_accel=rp.COLL_MAP_ACCEL_MEASURED.copy())
    rng = np.random.default_rng(5)
    B = 16
    pos = rng.uniform(-2, 2, (B, 3)); vel = rng.uniform(-6, 6, (B, 3))
    q = rp.quat_normalize(rng.normal(0, 1, (B, 4)))
    omega = rng.uniform(-1, 1, (B, 3)); thrust = rng.uniform(0.0, 1.0, B)
    batch = rp.PlantState(pos=pos, vel=vel, quat=q, omega=omega, thrust=thrust)
    acts = np.column_stack([rng.uniform(-8, 8, (B, 3)), rng.uniform(0.0, 1.2, B)])
    out = rp.step(batch, acts, 0.02, params)
    for i in range(B):
        si = rp.PlantState(pos=pos[i], vel=vel[i], quat=q[i], omega=omega[i],
                           thrust=np.asarray(thrust[i]))
        oi = rp.step(si, acts[i], 0.02, params)
        np.testing.assert_allclose(out.vel[i], oi.vel, atol=1e-12)
        np.testing.assert_allclose(out.pos[i], oi.pos, atol=1e-12)


def test_twin_rl_plant_aero_spot_parity():
    cfg = _aero_cfg()
    params = rp.PlantParams(super_rate_s=0.30, alpha_max_rps2=np.array([260.0, 260.0, 80.0]),
                            linear_drag=0.0,
                            quad_drag_c2=rp.QUAD_DRAG_C2_MEASURED.copy(),
                            coll_map_thr=rp.COLL_MAP_THR_MEASURED.copy(),
                            coll_map_accel=rp.COLL_MAP_ACCEL_MEASURED.copy())
    p = CtbrPlant(cfg, velocity_ned=np.array([3.0, -2.0, 1.0]))
    st = rp.PlantState(pos=np.zeros(3), vel=np.array([3.0, -2.0, 1.0]),
                       quat=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.zeros(3),
                       thrust=np.asarray(_HOV))
    rng = np.random.default_rng(2)
    for _ in range(300):
        a = np.concatenate([rng.uniform(-6, 6, 3), [rng.uniform(0.0, 1.2)]])
        p.step(_cmd(a[:3], a[3]), 0.01)
        st = rp.step(st, a, 0.01, params)
        np.testing.assert_array_equal(st.omega, p.omega)             # bit-identical rate path
    assert float(np.max(np.abs(st.vel - p.vel))) < 1e-9
    assert float(np.max(np.abs(st.pos - p.pos))) < 1e-9


# ------------------------------------------------------------------ torch mirrors (skippable)
# NB: import guarded per-test (not module-level importorskip -- that would skip the numpy anchors
# above on a torch-less machine). The authoritative mirror check stays the config-matrix gate.
try:
    import torch
except ImportError:                                       # pragma: no cover - torch in .venv here
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch absent: gate harness covers the mirror")


def _import_adapter():
    import sys
    import types
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))
    # stub diffaero's BaseDynamics (value-faithful: real grad_decay only scales gradients)
    if "diffaero.dynamics.base_dynamics" not in sys.modules:
        base_mod = types.ModuleType("diffaero.dynamics.base_dynamics")

        class BaseDynamics:
            def __init__(self, cfg, device):
                self.n_agents = int(getattr(cfg, "n_agents", 1))
                self.n_envs = int(getattr(cfg, "n_envs", 1))
                self.dt = float(cfg.dt)
                self.alpha = float(getattr(cfg, "alpha", 1.0))
                self.device = device

            def grad_decay(self, x):
                return x

            def detach(self):
                self._state = self._state.detach()

        base_mod.BaseDynamics = BaseDynamics
        pkg = types.ModuleType("diffaero")
        dyn_pkg = types.ModuleType("diffaero.dynamics")
        pkg.dynamics = dyn_pkg
        dyn_pkg.base_dynamics = base_mod
        sys.modules.setdefault("diffaero", pkg)
        sys.modules.setdefault("diffaero.dynamics", dyn_pkg)
        sys.modules["diffaero.dynamics.base_dynamics"] = base_mod
    import importlib
    return importlib.import_module("diffaero_dynamics")


@needs_torch
def test_torch_interp1d_bitwise_vs_numpy():
    dd = _import_adapter()
    rng = np.random.default_rng(31)
    xp = rp.COLL_MAP_THR_MEASURED
    fp = rp.COLL_MAP_ACCEL_MEASURED
    x = np.concatenate([rng.uniform(-0.5, 1.8, 500), xp.copy(), [0.0, 1.0]])
    ref = rp._interp1d(x, xp, fp)
    out = dd._t_interp1d(torch.tensor(x), torch.tensor(xp), torch.tensor(fp)).numpy()
    np.testing.assert_array_equal(out, ref)               # float64, bit-for-bit
    # batched per-env fp: each row must equal the shared-row reference
    fp2 = np.stack([fp, fp * 1.1, fp * 0.9])
    x2 = np.array([0.3, 0.3, 0.3])
    out2 = dd._t_interp1d(torch.tensor(x2), torch.tensor(xp), torch.tensor(fp2)).numpy()
    for i in range(3):
        np.testing.assert_array_equal(out2[i], rp._interp1d(x2[i:i + 1], xp, fp2[i])[0])


def _make_dyn(dd, n_envs=512, dr=False, dr_aero=False, params=None, seed=0):
    from types import SimpleNamespace
    torch.manual_seed(seed)
    cfg = SimpleNamespace(n_envs=n_envs, n_agents=1, dt=0.02, alpha=1.0, g=9.80665,
                          n_substeps=1, controller=None, dr=dr, dr_aero=dr_aero,
                          dr_latency_max_steps=0)
    return dd.PeregrinePlantDynamics(cfg, torch.device("cpu"), backend="torch",
                                     params=params or rp.PlantParams())


@needs_torch
def test_dr_aero_bands_and_hover_pin():
    dd = _import_adapter()
    n = 512
    dyn = _make_dyn(dd, n_envs=n, dr=True, dr_aero=True)
    dyn.reset_idx(torch.arange(n))
    # c2: each (3,2) slot within its own nominal x [0.040, 0.065]/0.052 (the Section 7 band
    # re-expressed relative to the pooled nominal -- S14 yaw-precedent), and actually spread
    c2 = dyn._dr_c2.numpy()
    lo = rp.QUAD_DRAG_C2_MEASURED * (0.040 / 0.052) * (1 - 1e-6)
    hi = rp.QUAD_DRAG_C2_MEASURED * (0.065 / 0.052) * (1 + 1e-6)
    assert np.all(c2 >= lo) and np.all(c2 <= hi)
    assert np.std(c2[:, 0, 0]) > 1e-4
    # collective table: hover point pinned to +-2%, full-stick scaled ~+-10% around the nominal
    kv = dyn._dr_coll_kvals.numpy()
    k_hov = 9.5804078429104393
    hov_col = kv[:, 4]                                    # the 0.2656 knot
    assert np.all(hov_col >= 0.98 * k_hov - 1e-5) and np.all(hov_col <= 1.02 * k_hov + 1e-5)
    full = kv[:, -1]
    full_lo = 0.98 * k_hov + 0.90 * (78.282838504684648 - k_hov)
    full_hi = 1.02 * k_hov + 1.10 * (78.282838504684648 - k_hov)
    assert np.all(full >= full_lo - 1e-4) and np.all(full <= full_hi + 1e-4)
    assert np.std(full) > 0.5
    # d1 residual in [0, 0.08]; hover collective conversion PINNED at the nominal
    d1 = dyn._dr_drag.numpy()
    assert np.all(d1 >= 0.0) and np.all(d1 <= 0.08) and np.std(d1) > 1e-3
    np.testing.assert_array_equal(dyn._dr_hover.numpy(),
                                  np.full(n, np.float32(0.2656)))
    # legacy DR (no dr_aero): fractional drag/hover bands exactly as before
    dyn2 = _make_dyn(dd, n_envs=n, dr=True, dr_aero=False, seed=1)
    dyn2.reset_idx(torch.arange(n))
    drag2 = dyn2._dr_drag.numpy()
    hov2 = dyn2._dr_hover.numpy()
    assert np.all(drag2 >= 0.2111 * 0.7 - 1e-6) and np.all(drag2 <= 0.2111 * 1.3 + 1e-6)
    assert np.all(hov2 >= 0.2656 * 0.95 - 1e-6) and np.all(hov2 <= 0.2656 * 1.05 + 1e-6)
    assert np.std(drag2) > 1e-3 and np.std(hov2) > 1e-4


@needs_torch
def test_dr_aero_step_uses_the_measured_tables():
    # full-stick punch-out: the aero-DR plant must produce ~2x the legacy upward accel
    dd = _import_adapter()
    n = 64
    dyn_aero = _make_dyn(dd, n_envs=n, dr=True, dr_aero=True)
    dyn_aero.reset_idx(torch.arange(n))
    dyn_leg = _make_dyn(dd, n_envs=n)                     # DR off, legacy params
    U = torch.zeros(n, 4)
    U[:, 0] = 5.0                                         # normed thrust 5 -> collective 1.328
    dyn_aero.step(U)
    dyn_leg.step(U)
    vz_aero = dyn_aero._state[:, 9].numpy()               # DiffAero frame: z-up world vz
    vz_leg = dyn_leg._state[:, 9].numpy()
    assert np.all(vz_aero > 1.3 * vz_leg)                 # ~(78-g) vs (49-g) net climb accel
    assert np.all(vz_leg > 0)
