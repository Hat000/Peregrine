"""Laptop unit tests for the inc8 STABILITY KNOBS (branch p2-inc8-stability-knobs-2026-06-16):

  knob #1  band_el metric (OBSERVABILITY ONLY)  -- the elevation analog of band_az; the vertical-residual
           diagnostic + the S2 g_pitch empirical sign-check. Tested two ways: (a) the masked-mean formula
           mirrors the env's inline diagnostic and DECOMPOSES orthogonally to band_az (azimuth vs
           elevation), and (b) the env source actually wires + logs it (so it cannot change training).
  knob #3  critic-width A/B (rl/inc8_critic_width.py) -- the OFF path is a true no-op (byte-identical) and
           the hidden-dim spec normalizes correctly. The real wider-critic CONSTRUCTION needs diffaero and
           is verified on the Adroit precheck (laptop has no diffaero), not here.

knob #2 (look-at gain-warmup) is tested in tests/test_inc8_lookat.py per the worker brief.

Run from repo ROOT:  .venv\\Scripts\\python.exe -m pytest tests/test_inc8_stability_knobs.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import inc8_critic_width as CW                                          # noqa: E402
import inc8_tb_trace as TB                                             # noqa: E402

DT = torch.float64
ENV_SRC = (ROOT / "rl" / "peregrine_racing_inc8.py").read_text()


# ============================================ knob #1: band_el metric (observability only) ===========
def _band_az_el(t_cam, r_lo, r_hi):
    """Replicate the env's inline band_az / band_el masked-means (peregrine_racing_inc8.step diagnostics):
    mean |atan2(X,Z)| and mean |atan2(Y,Z)| (deg) over the in-front range-gated look-band. range == |t_cam|
    since t_cam is the gate centre in the camera frame."""
    rng = torch.linalg.norm(t_cam, dim=-1)
    deg = 180.0 / torch.pi
    az = torch.atan2(t_cam[..., 0], t_cam[..., 2].clamp(min=1e-6)) * deg
    el = torch.atan2(t_cam[..., 1], t_cam[..., 2].clamp(min=1e-6)) * deg
    band = ((rng >= r_lo) & (rng <= r_hi) & (t_cam[..., 2] > 0)).to(t_cam.dtype)
    n = band.sum().clamp(min=1)
    return (az.abs() * band).sum() / n, (el.abs() * band).sum() / n


def test_band_el_matches_atan2_formula():
    """band_el == mean |atan2(Y, Z)| in degrees for a known vector (gate 1.0 m up, 0.5 m right, 10 m fwd)."""
    t = torch.tensor([[0.5, 1.0, 10.0]], dtype=DT)
    _az, el = _band_az_el(t, 0.0, 1e9)
    import math
    assert el.item() == pytest.approx(math.degrees(math.atan2(1.0, 10.0)), abs=1e-9)


def test_band_az_el_are_orthogonal_axes():
    """A PURELY-horizontal gate offset shows up in band_az only; a PURELY-vertical offset in band_el only.
    This is the whole point of adding band_el: it isolates the VERTICAL residual (the sigma_vert axis)."""
    horiz = torch.tensor([[4.0, 0.0, 12.0]], dtype=DT)           # right, level
    vert = torch.tensor([[0.0, 4.0, 12.0]], dtype=DT)            # down, centred laterally
    az_h, el_h = _band_az_el(horiz, 0.0, 1e9)
    az_v, el_v = _band_az_el(vert, 0.0, 1e9)
    assert az_h > 5.0 and el_h == pytest.approx(0.0, abs=1e-9)   # horizontal -> azimuth only
    assert el_v > 5.0 and az_v == pytest.approx(0.0, abs=1e-9)   # vertical   -> elevation only


def test_band_el_respects_range_band_and_in_front():
    """Out-of-band (too near/far) and behind-camera vectors contribute 0 to the masked mean (== band_az)."""
    t = torch.tensor([[0.0, 2.0, 5.0],      # in front, range 5 -> OUT of [8,30] band
                      [0.0, 2.0, 20.0],     # in front, range ~20 -> IN band
                      [0.0, 2.0, -20.0]],   # BEHIND the camera -> excluded
                     dtype=DT)
    _az, el = _band_az_el(t, 8.0, 30.0)
    import math
    only_in_band = math.degrees(math.atan2(2.0, 20.0))          # just the middle row
    assert el.item() == pytest.approx(only_in_band, abs=1e-6)


def test_band_el_empty_band_is_zero_not_nan():
    """No vector in the band -> 0/clamp(0,min=1) == 0 (never NaN), matching the env's sync-free guard."""
    t = torch.tensor([[0.0, 2.0, 100.0]], dtype=DT)             # range 100 -> outside [8,30]
    _az, el = _band_az_el(t, 8.0, 30.0)
    assert torch.isfinite(el) and el.item() == 0.0


def test_env_wires_and_logs_band_el():
    """The env must actually compute band_el, STACK it in metric_vec, and LOG it under inc8_band_el_abs_deg
    -- otherwise the metric is dead. (Source pin, like test_inc8_off_identity; the env needs diffaero.)"""
    assert "band_el_abs = (" in ENV_SRC                          # computed
    assert "band_el_abs.to(mdt)" in ENV_SRC                      # stacked into metric_vec
    assert "band_el_v" in ENV_SRC                                # unpacked from the single .tolist()
    assert '"inc8_band_el_abs_deg": band_el_v' in ENV_SRC        # logged into loss_components
    # and it must use atan2(Y, Z) (Y = t_cam[...,1]), the elevation axis -- not a copy of the azimuth line
    assert "torch.atan2(_tcam[..., 1], _tcam[..., 2].clamp(min=1e-6))" in ENV_SRC


def test_band_el_is_observability_only_same_band_as_az():
    """band_el reuses the SAME _look_band as band_az (no separate gating) -- it cannot perturb training."""
    # exactly one _look_band is defined and both az and el divide by its sum: pin that they share it.
    assert ENV_SRC.count("_look_band = (") == 1
    assert "(_el_deg.abs() * _look_band.to(mdt)).sum() / _look_band.sum().clamp(min=1)" in ENV_SRC


def test_tb_trace_has_band_el_column():
    """inc8_tb_trace.py reader exposes the band_el_abs_deg column and resolves the env's tag."""
    labels = [lbl for lbl, _ in TB.COLUMNS]
    assert "band_el_abs_deg" in labels
    cands = dict(TB.COLUMNS)["band_el_abs_deg"]
    assert "inc8_band_el_abs_deg" in cands
    # resolves against a realistic tag namespace (env_loss/<suffix>)
    resolved = TB._resolve_tag(["env_loss/inc8_band_az_abs_deg", "env_loss/inc8_band_el_abs_deg"], cands)
    assert resolved == "env_loss/inc8_band_el_abs_deg"


# ============================================ knob #3: critic-width A/B (off-path + parsing) ==========
def test_normalize_hidden_dims_unset_is_none():
    assert CW.normalize_hidden_dims(None) is None               # the OFF sentinel


def test_normalize_hidden_dims_int_and_list():
    assert CW.normalize_hidden_dims(512) == [512]               # bare int -> singleton
    assert CW.normalize_hidden_dims([512, 256]) == [512, 256]
    assert CW.normalize_hidden_dims((256, 128)) == [256, 128]
    assert CW.normalize_hidden_dims(["512", "256"]) == [512, 256]   # stringified (Hydra) -> ints


def test_normalize_hidden_dims_rejects_degenerate():
    for bad in ([], [0], [-4], [128, 0]):
        with pytest.raises(ValueError):
            CW.normalize_hidden_dims(bad)


def test_maybe_widen_critic_off_is_noop():
    """Unset critic_hidden_dim -> returns False and touches NOTHING (the byte-identical guarantee). Runs
    on the laptop because the off-path returns before importing torch/omegaconf/diffaero."""
    agent, optim = object(), object()
    ppo = SimpleNamespace(agent=agent, optim=optim)
    cfg = SimpleNamespace(lr=0.0026, eps=1e-8)                  # no critic_hidden_dim attribute at all
    assert CW.maybe_widen_critic(ppo, cfg) is False
    assert ppo.agent is agent and ppo.optim is optim           # identity unchanged -> no rebuild

    cfg2 = SimpleNamespace(critic_hidden_dim=None)              # explicit None == unset
    assert CW.maybe_widen_critic(ppo, cfg2) is False
    assert ppo.agent is agent and ppo.optim is optim
