"""FAITHFUL PLANT (plant-sysID gate, 2026-07-16): the cfg-gated expansive rate loop + measured aero.

The DiffAero training plant's inner rate loop defaults to a FLAT gain (``super_rate_s=None``), but
the measured VQ2 hardware is amplitude-progressive (EXPANSIVE): cmd->achieved body-rate gain rises
~2.5x->3.1x roll/pitch and ~2.23x->2.89x yaw as |command| sweeps 0.1->0.8 of full stick (ShadowPC
COMPLETE 3-axis ampsweep, refit 2026-07-16 -- the round-1 roll/pitch base was ~6-7% low from a single
doublet). ``++dynamics.faithful_rate=true`` installs the joint fit -- the recalibrated SMALL-SIGNAL
``rate_gain`` [2.359, 2.363, 2.163] + PER-AXIS ``super_rate_s`` [0.296, 0.284, 0.316] + the measured
slew clamp -- so the trained plant matches the hardware and the policy stops over-rotating at
aggressive banks in deploy. Its default is the legacy aero (linear thrust + world linear drag);
``++dynamics.faithful_aero=true`` swaps in the measured convex collective->accel map + body-frame
quadratic drag (linear_drag=0), and ``++dynamics.faithful_plant=true`` enables BOTH. The airspeed
lapse is a SEPARATE ``++dynamics.faithful_lapse=true`` sub-toggle, default OFF even under aero.

These tests pin:
  1. the new small-signal constants + closed-form reproduction of the VQ2 curve (no torch);
  2. the faithful plant's SETTLED cmd->achieved gain reproduces the measured 3-axis ampsweep within
     ~2% (RMSE ~0.04; the flat 0.1-0.2 toe the single-pole model can't match), with NO roll/pitch
     overshoot -- and that flat-MID default gain + s is only ~4% over the real curve, NOT the >10%
     footgun the earlier (too-low 2.74 reference) test wrongly claimed (no torch);
  3. cfg wiring: default-OFF leaves the params EXACTLY legacy (byte-identical); ``faithful_rate=true``
     installs the faithful rate params + torch caches; ``faithful_aero=true``/``faithful_plant=true``
     install the aero params; explicit scalar overrides win (needs torch);
  4. torch-backend parity: the torch mirror applies the super-rate map AND the measured aero
     (coll-map + quad-drag) IDENTICALLY to the numpy plant -- the production path, where a
     flat/legacy torch vs expansive/faithful numpy would silently mis-train (needs torch).

The adapter is driven off-Adroit (no diffaero pkg) via a stub-base MRO insert that provides the
BaseDynamics attributes ``PeregrinePlantDynamics.__init__`` reads, running the REAL __init__ +
backends (not a re-implementation). torch is present in the repo .venv; the torch tests skip cleanly
if it is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "rl"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from racer import rl_plant as rp  # noqa: E402

try:
    import torch
    import diffaero_dynamics as DA
    _HAVE_TORCH = True
except Exception:                                       # pragma: no cover
    torch = None
    DA = None
    _HAVE_TORCH = False

# Measured VQ2 hardware curves (ShadowPC COMPLETE 3-axis ampsweep, refit 2026-07-16): settled
# cmd->achieved rate gain vs |a| (fraction of full stick). Roll/pitch are the ampsweep curve
# (roll@0.6=2.92, pitch@0.6=2.90 -- NOT the old single-doublet 2.74/2.71). Yaw unchanged from round 1.
VQ2_ROLL = {0.1: 2.47, 0.2: 2.46, 0.4: 2.65, 0.6: 2.92, 0.8: 3.07}
VQ2_PITCH = {0.1: 2.47, 0.2: 2.46, 0.4: 2.64, 0.6: 2.90, 0.8: 3.04}
VQ2_YAW = {0.1: 2.23, 0.2: 2.31, 0.4: 2.48, 0.6: 2.67, 0.8: 2.89}
VQ2_ROLL_06 = VQ2_ROLL[0.6]                            # 2.92
VQ2_PITCH_06 = VQ2_PITCH[0.6]                          # 2.90
REL_TOL = 0.02                                          # yaw reproduces to ~0.2%
# roll/pitch: the single-pole static-gain model can't match the flat 0.1-0.2 toe exactly, so the
# per-point band is a touch looser and an RMSE bound guards the overall fit (see the reproduction test)
REL_TOL_RP = 0.025
RMSE_TOL_RP = 0.05
_PI = np.pi
_DT = 0.005


def _faithful_params() -> rp.PlantParams:
    """The plant the ``faithful_rate=true`` flag installs (rate loop only; aero left legacy)."""
    return rp.PlantParams(
        rate_gain=rp.RATE_GAIN_SMALLSIGNAL_MEASURED.copy(),
        super_rate_s=rp.SUPER_RATE_S_FAITHFUL,
        alpha_max_rps2=rp.ALPHA_MAX_RPS2_MEASURED.copy(),
    )


def _sustained_gain(params, axis, mag, seconds=1.2):
    """Realized steady body rate / commanded rate after a sustained single-axis command."""
    st = rp.PlantState.hover(params=params)
    a = np.zeros(4)
    a[axis] = mag
    a[3] = params.hover_thrust
    for _ in range(int(round(seconds / _DT))):
        st = rp.step(st, a, _DT, params)
    return abs(float(st.omega[axis] / mag))


# ================================================================= 1. constants + closed form
def test_smallsignal_constants_values():
    np.testing.assert_allclose(rp.RATE_GAIN_SMALLSIGNAL_MEASURED, [2.359, 2.363, 2.163])
    # super_rate_s is now PER-AXIS (roll, pitch, yaw)
    np.testing.assert_allclose(np.asarray(rp.SUPER_RATE_S_FAITHFUL), [0.296, 0.284, 0.316])
    # closed form g0/(1 - s*min(|a|,1)) reproduces the measured yaw table (per-axis s; yaw = axis 2)
    g0y, s = rp.RATE_GAIN_SMALLSIGNAL_MEASURED[2], float(np.asarray(rp.SUPER_RATE_S_FAITHFUL)[2])
    for a, g_meas in VQ2_YAW.items():
        g = g0y / (1.0 - s * min(a, 1.0))
        assert abs(g - g_meas) / g_meas < REL_TOL, f"yaw |a|={a}: {g:.4f} vs {g_meas}"


# ================================================================= 2. VQ2 static-gain reproduction
def test_faithful_plant_reproduces_vq2_yaw_ampsweep():
    p = _faithful_params()
    for a, g_meas in VQ2_YAW.items():
        g = _sustained_gain(p, 2, a * _PI)             # yaw command a*pi rad/s
        assert abs(g - g_meas) / g_meas < REL_TOL, (
            f"yaw |a|={a}: settled gain {g:.4f} vs VQ2 {g_meas} ({(g-g_meas)/g_meas*100:+.2f}%)")


def test_faithful_plant_reproduces_vq2_rollpitch_ampsweep_no_overshoot():
    """The faithful plant's settled gain reproduces the FULL roll/pitch ampsweep within ~2.5% per
    point and RMSE ~0.04 (the single-pole static-gain model can't match the flat 0.1-0.2 toe exactly),
    and never materially OVER-rotates -- the small-signal fit sits ON the curve, not above it."""
    p = _faithful_params()
    for axis, curve, name in ((0, VQ2_ROLL, "roll"), (1, VQ2_PITCH, "pitch")):
        res = []
        for a, g_meas in curve.items():
            g = _sustained_gain(p, axis, a * _PI)
            res.append(g - g_meas)
            assert abs(g - g_meas) / g_meas < REL_TOL_RP, (
                f"{name} |a|={a}: settled {g:.4f} vs VQ2 {g_meas} ({(g-g_meas)/g_meas*100:+.2f}%)")
            assert g <= g_meas * (1 + REL_TOL_RP), f"{name} |a|={a} overshoots: {g:.4f} > {g_meas}"
        rmse = float(np.sqrt(np.mean(np.square(res))))
        assert rmse < RMSE_TOL_RP, f"{name} ampsweep RMSE {rmse:.4f} >= {RMSE_TOL_RP}"


def test_flat_mid_default_gain_is_only_mildly_over_not_a_footgun():
    """Corrects a STALE premise. An earlier test claimed pairing the DEFAULT rate_gain
    [2.501, 2.504, 2.231] (a flat-MID fit) with the super-rate map over-rotates roll@0.6 by >10% --
    but that compared against a too-low 2.74 reference. The real ampsweep roll@0.6 is 2.92, so
    flat-mid + s lands only ~4% over (~3.04); there is NO >10% footgun to assert. We still prefer the
    small-signal gains because they sit ON the curve (asserted below), but the false claim is removed."""
    bad = rp.PlantParams(super_rate_s=rp.SUPER_RATE_S_FAITHFUL,
                         alpha_max_rps2=rp.ALPHA_MAX_RPS2_MEASURED.copy())   # default rate_gain
    gr = _sustained_gain(bad, 0, 0.6 * _PI)
    assert 1.0 < gr / VQ2_ROLL_06 < 1.08, f"flat-mid roll@0.6 {gr:.4f} vs {VQ2_ROLL_06} (expected ~+4%)"
    # the small-signal fit is strictly closer to the measured curve (on-curve, not above it)
    good = _sustained_gain(_faithful_params(), 0, 0.6 * _PI)
    assert abs(good - VQ2_ROLL_06) < abs(gr - VQ2_ROLL_06)


# ================================================================= torch-backed adapter harness
if _HAVE_TORCH:
    _FLIP = np.array([1.0, -1.0, -1.0])

    class _StubBase:
        """Minimal BaseDynamics stand-in: sets what PeregrinePlantDynamics.__init__ reads."""
        def __init__(self, cfg, device):
            self.n_envs = int(cfg.n_envs); self.n_agents = int(getattr(cfg, "n_agents", 1))
            self.dt = float(cfg.dt); self.alpha = float(getattr(cfg, "alpha", 1.0))
            self._G = float(getattr(cfg, "g", 9.80665))
            self._G_vec = torch.tensor([0.0, 0.0, -self._G])
        def grad_decay(self, x): return x
        def detach(self):
            if getattr(self, "_state", None) is not None: self._state = self._state.detach()

    class _Adapter(DA.PeregrinePlantDynamics, _StubBase):   # MRO: _Adapter -> Peregrine -> _StubBase
        def seed_ned(self, pos, vel, quat_wxyz, omega, thrust):
            pd, vd, qd, wd = DA._diffaero_from_ned_np(pos, vel, quat_wxyz, omega)
            self._state = torch.tensor(np.concatenate([pd, qd, vd, wd], axis=-1), dtype=self._state.dtype)
            self._thrust = torch.tensor(np.asarray(thrust, float), dtype=self._thrust.dtype)
            self._acc = torch.zeros_like(self._acc); self._plant_act_buf = None
        def read_ned(self):
            st = self._state.detach().cpu().numpy()
            return DA._ned_from_diffaero_np(st[..., 0:3], st[..., 3:7], st[..., 7:10], st[..., 10:13])

    def _mk(n_envs, dt, backend, **cfg_over):
        cfg = SimpleNamespace(n_envs=n_envs, n_agents=1, dt=dt, alpha=1.0, g=9.80665,
                              n_substeps=1, controller=None, dr=False, dr_latency_max_steps=0,
                              capture_specific_force=False, **cfg_over)
        a = _Adapter(cfg, torch.device("cpu"), backend=backend, params=rp.PlantParams())
        d = torch.float64
        a._state = a._state.to(d); a._acc = a._acc.to(d); a._thrust = a._thrust.to(d)
        a._rate_gain = torch.tensor(a.params.rate_gain, dtype=d)
        a._rate_sign = torch.tensor(a.params.rate_sign, dtype=d)
        a._BODY_UP = torch.tensor([0.0, 0.0, -1.0], dtype=d)
        a._g_vec_ned = torch.tensor([0.0, 0.0, a.params.g], dtype=d)
        a._super_s = (None if a.params.super_rate_s is None else
                      torch.tensor(np.broadcast_to(a.params.super_rate_s, (3,)).copy(), dtype=d))
        a._alpha_max = (None if a.params.alpha_max_rps2 is None else
                        torch.tensor(np.broadcast_to(a.params.alpha_max_rps2, (3,)).copy(), dtype=d))
        # rebuild the measured-aero torch caches at float64 too (mirrors __init__) so the aero
        # torch<->numpy parity check is bit-exact rather than tripping on the float32 default caches
        a._quad_c2 = (None if a.params.quad_drag_c2 is None else
                      torch.tensor(a.params.quad_drag_c2, dtype=d))
        a._coll_knots = (None if a.params.coll_map_thr is None else
                         torch.tensor(a.params.coll_map_thr, dtype=d))
        a._coll_kvals = (None if a.params.coll_map_accel is None else
                         torch.tensor(a.params.coll_map_accel, dtype=d))
        a._lapse_knots = (None if a.params.lapse_speed is None else
                          torch.tensor(a.params.lapse_speed, dtype=d))
        a._lapse_vals = (None if a.params.lapse_factor is None else
                         torch.tensor(a.params.lapse_factor, dtype=d))
        return a

    def _step(a, U):
        t = torch.tensor(U, dtype=a._state.dtype)
        (a._step_torch if a.backend == "torch" else a._step_numpy)(t)

    def _battery(seed=0, n=64):
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        pos = rng.uniform(-30, 30, (n, 3)); vel = rng.uniform(-20, 20, (n, 3))
        omega = rng.uniform(-10, 10, (n, 3)); thr = rng.uniform(0.1, 0.9, n)
        Us = [np.column_stack([rng.uniform(0, 3, n), rng.uniform(-6, 6, (n, 3))]) for _ in range(6)]
        return pos, vel, q, omega, thr, Us

    def _run(backend, **cfg_over):
        pos, vel, q, omega, thr, Us = _battery()
        a = _mk(pos.shape[0], 0.02, backend, **cfg_over)
        a.seed_ned(pos, vel, q, omega, thr)
        for U in Us:
            _step(a, U)
        p, v, qq, w = a.read_ned()
        out = np.concatenate([p, qq, v, w], axis=-1)
        for i in range(out.shape[0]):                  # q/-q sign alignment for comparison
            if out[i, 3] < 0: out[i, 3:7] *= -1
        return out


pytestmark_torch = pytest.mark.skipif(not _HAVE_TORCH, reason="torch/diffaero adapter unavailable")


# ================================================================= 3. cfg wiring
@pytestmark_torch
def test_default_off_params_are_byte_identical_legacy():
    a = _mk(4, 1 / 30, "torch")                        # no faithful_rate key
    assert a.params.super_rate_s is None
    assert a.params.alpha_max_rps2 is None
    np.testing.assert_array_equal(a.params.rate_gain, [2.501, 2.504, 2.231])


@pytestmark_torch
def test_default_off_backends_bit_identical_to_legacy_plant():
    """OFF: torch backend == numpy backend (== rl_plant default = the legacy flat plant) bit-for-bit
    over an extreme battery -- absence of the flag changes NOTHING numerically."""
    out_t = _run("torch")
    out_n = _run("rl_plant_numpy")
    assert float(np.max(np.abs(out_t - out_n))) < 1e-12


@pytestmark_torch
def test_faithful_rate_installs_faithful_params_and_caches():
    a = _mk(4, 1 / 30, "torch", faithful_rate=True)
    np.testing.assert_array_equal(a.params.rate_gain, [2.359, 2.363, 2.163])
    np.testing.assert_allclose(np.asarray(a.params.super_rate_s), [0.296, 0.284, 0.316])
    np.testing.assert_array_equal(a.params.alpha_max_rps2, [260.0, 260.0, 80.0])
    # torch caches reflect the faithful params (production path reads these); super_rate_s per-axis
    np.testing.assert_allclose(a._super_s.numpy(), [0.296, 0.284, 0.316])
    np.testing.assert_allclose(a._alpha_max.numpy(), [260.0, 260.0, 80.0])
    np.testing.assert_allclose(a._rate_gain.numpy(), [2.359, 2.363, 2.163])
    # faithful_rate touches ONLY the rate loop -- aero stays legacy
    assert a.params.coll_map_thr is None and a.params.quad_drag_c2 is None
    assert a.params.linear_drag == rp.PlantParams().linear_drag


@pytestmark_torch
def test_scalar_overrides_win():
    a = _mk(4, 1 / 30, "torch", super_rate_s=0.2, rate_gain=[2.3, 2.3, 2.3], alpha_max_rps2=250.0)
    np.testing.assert_array_equal(np.asarray(a.params.super_rate_s), 0.2)
    np.testing.assert_array_equal(a.params.rate_gain, [2.3, 2.3, 2.3])
    np.testing.assert_array_equal(a.params.alpha_max_rps2, [250.0, 250.0, 250.0])   # scalar broadcast
    # an override applied ON TOP of faithful_rate replaces just that field
    b = _mk(4, 1 / 30, "torch", faithful_rate=True, super_rate_s=0.28)
    np.testing.assert_array_equal(np.asarray(b.params.super_rate_s), 0.28)
    np.testing.assert_array_equal(b.params.rate_gain, [2.359, 2.363, 2.163])         # preset kept


# ================================================================= 4. torch<->numpy parity (super-rate)
@pytestmark_torch
def test_faithful_rate_torch_matches_numpy_settled_gain():
    """The torch mirror applies the super-rate map IDENTICALLY to the numpy plant across a command
    amplitude sweep spanning the map-input clamp (|a|>1) -- the CRITICAL production-path check."""
    for amp in (0.1, 0.4, 0.8, 1.2):
        for axis in (0, 1, 2):
            gains = {}
            for backend in ("rl_plant_numpy", "torch"):
                a = _mk(1, 0.02, backend, faithful_rate=True)
                a.seed_ned(np.zeros((1, 3)), np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]),
                           np.zeros((1, 3)), np.full(1, a.params.hover_thrust))
                U = np.zeros((1, 4)); U[0, 0] = 1.0; U[0, 1 + axis] = amp * _PI
                for _ in range(240):
                    _step(a, U)
                _, _, _, w = a.read_ned()
                gains[backend] = abs(float(w[0, axis]) / float(U[0, 1 + axis] * _FLIP[axis]))
            assert abs(gains["torch"] - gains["rl_plant_numpy"]) < 1e-12


@pytestmark_torch
def test_faithful_rate_torch_matches_numpy_full_state_extreme():
    """Full-state torch-vs-numpy parity under the faithful super-rate map over an extreme battery
    (inverted attitudes, |v|~35 m/s, |cmd|>pi) -- bit-exact at float64."""
    out_t = _run("torch", faithful_rate=True)
    out_n = _run("rl_plant_numpy", faithful_rate=True)
    assert float(np.max(np.abs(out_t - out_n))) < 1e-12


# ================================================================= 5. faithful AERO (static thrust + drag)
@pytestmark_torch
def test_faithful_aero_installs_measured_aero_and_leaves_rate_legacy():
    """``faithful_aero=true`` installs the convex collective map + quadratic drag + linear_drag=0 on
    the params AND the torch caches, and does NOT touch the rate loop or enable the airspeed lapse."""
    a = _mk(4, 1 / 30, "torch", faithful_aero=True)
    np.testing.assert_array_equal(a.params.coll_map_thr, rp.COLL_MAP_THR_MEASURED)
    np.testing.assert_array_equal(a.params.coll_map_accel, rp.COLL_MAP_ACCEL_MEASURED)
    np.testing.assert_array_equal(a.params.quad_drag_c2, rp.QUAD_DRAG_C2_MEASURED)     # (3, 2)
    assert a.params.linear_drag == 0.0
    # lapse EXCLUDED even under faithful_aero (own sub-toggle, default OFF)
    assert a.params.lapse_speed is None and a.params.lapse_factor is None
    # rate loop untouched -> legacy flat gain
    assert a.params.super_rate_s is None
    np.testing.assert_array_equal(a.params.rate_gain, [2.501, 2.504, 2.231])
    # torch caches reflect the faithful aero (production path reads these)
    np.testing.assert_allclose(a._coll_knots.numpy(), rp.COLL_MAP_THR_MEASURED)
    np.testing.assert_allclose(a._coll_kvals.numpy(), rp.COLL_MAP_ACCEL_MEASURED)
    np.testing.assert_allclose(a._quad_c2.numpy(), rp.QUAD_DRAG_C2_MEASURED)


@pytestmark_torch
def test_faithful_plant_installs_both_rate_and_aero():
    """``faithful_plant=true`` is the convenience that enables BOTH faithful_rate and faithful_aero
    (still no lapse)."""
    a = _mk(4, 1 / 30, "torch", faithful_plant=True)
    np.testing.assert_array_equal(a.params.rate_gain, [2.359, 2.363, 2.163])
    np.testing.assert_allclose(np.asarray(a.params.super_rate_s), [0.296, 0.284, 0.316])
    np.testing.assert_array_equal(a.params.alpha_max_rps2, [260.0, 260.0, 80.0])
    np.testing.assert_array_equal(a.params.coll_map_thr, rp.COLL_MAP_THR_MEASURED)
    np.testing.assert_array_equal(a.params.quad_drag_c2, rp.QUAD_DRAG_C2_MEASURED)
    assert a.params.linear_drag == 0.0
    assert a.params.lapse_speed is None                                    # lapse still OFF


@pytestmark_torch
def test_faithful_aero_default_off_leaves_legacy_aero():
    """No aero flag -> the params keep the legacy linear thrust map + world linear drag (untouched)."""
    a = _mk(4, 1 / 30, "torch")
    assert a.params.coll_map_thr is None and a.params.coll_map_accel is None
    assert a.params.quad_drag_c2 is None
    assert a.params.linear_drag == rp.PlantParams().linear_drag             # 0.2111 legacy
    assert a._coll_knots is None and a._quad_c2 is None


@pytestmark_torch
def test_faithful_aero_torch_matches_numpy_full_state_extreme():
    """Full-state torch-vs-numpy parity with the measured aero (convex coll-map + sign-split quad
    drag) forced ON over the extreme battery -- the production torch path must mirror numpy bit-exact
    (a numpy-only aero would silently NOT train)."""
    out_t = _run("torch", faithful_aero=True)
    out_n = _run("rl_plant_numpy", faithful_aero=True)
    assert float(np.max(np.abs(out_t - out_n))) < 1e-12


@pytestmark_torch
def test_faithful_plant_torch_matches_numpy_full_state_extreme():
    """Full-state parity with BOTH the faithful rate loop AND the measured aero on together."""
    out_t = _run("torch", faithful_plant=True)
    out_n = _run("rl_plant_numpy", faithful_plant=True)
    assert float(np.max(np.abs(out_t - out_n))) < 1e-12


# ================================================================= 6. faithful LAPSE sub-toggle (default OFF)
@pytestmark_torch
def test_faithful_lapse_requires_the_collective_map():
    """The airspeed lapse multiplies the collective map, so ``faithful_lapse=true`` without an
    installed map (no faithful_aero / no passed coll_map params) is a config error."""
    with pytest.raises(ValueError, match="faithful_lapse"):
        _mk(4, 1 / 30, "torch", faithful_lapse=True)


@pytestmark_torch
def test_faithful_lapse_is_off_under_aero_and_installs_only_when_set():
    """faithful_aero alone leaves the lapse OFF (unvalidated -- under-models the fresh climb inflow);
    the explicit ``faithful_lapse=true`` sub-toggle installs the measured curve on top of the map."""
    a = _mk(4, 1 / 30, "torch", faithful_aero=True)
    assert a.params.lapse_speed is None and a.params.lapse_factor is None
    b = _mk(4, 1 / 30, "torch", faithful_aero=True, faithful_lapse=True)
    np.testing.assert_array_equal(b.params.lapse_speed, rp.LAPSE_SPEED_MEASURED)
    np.testing.assert_array_equal(b.params.lapse_factor, rp.LAPSE_FACTOR_MEASURED)
    # and the torch lapse mirror is bit-exact if it IS turned on
    out_t = _run("torch", faithful_aero=True, faithful_lapse=True)
    out_n = _run("rl_plant_numpy", faithful_aero=True, faithful_lapse=True)
    assert float(np.max(np.abs(out_t - out_n))) < 1e-12
