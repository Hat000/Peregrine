"""Laptop unit tests for rl/inc8_noise_anneal.py -- the deterministic-stability noise-anneal lever.

Pure-python schedule math + the real actor_logstd clamp (CPU torch). No diffaero needed. Pins the
squash round-trip to diffaero's constants and verifies the OFF path is a true no-op.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl"))

import inc8_noise_anneal as na


# ---- squash round-trip (MUST match diffaero network/agents.py StochasticActor.forward) ----

def test_squash_constants_match_diffaero():
    assert na.LOG_STD_MIN == -5.0 and na.LOG_STD_MAX == 2.0


def test_init_param_zero_maps_to_diffaero_init_std():
    # actor_logstd inits to zeros; diffaero squash gives std = exp(-1.5) ~= 0.2231
    assert na.param_to_std(0.0) == pytest.approx(math.exp(-1.5), rel=1e-6)


@pytest.mark.parametrize("std", [0.03, 0.1, 0.2231, 0.4, 0.6, 1.0, 3.0])
def test_std_param_roundtrip(std):
    assert na.param_to_std(na.std_to_param(std)) == pytest.approx(std, rel=1e-5)


def test_std_to_param_saturates_above_ceiling():
    # exp(2)=7.389 is the diffaero ceiling; a larger target saturates (no NaN from atanh(>=1)).
    p = na.std_to_param(100.0)
    assert math.isfinite(p)
    assert na.param_to_std(p) <= math.exp(na.LOG_STD_MAX) + 1e-6


# ---- schedule shape ----

def _sched(**kw):
    base = dict(hold_frac=0.5, std_hold=0.6, std_floor=0.03,
                entropy_hold=0.01, entropy_floor=0.0, n_updates=4000)
    base.update(kw)
    return base


def test_hold_region_is_flat_at_hold_values():
    s = _sched()
    for i in (0, 100, 1999, 2000):  # i <= hold_frac*N -> progress 0
        v = na.schedule_values(i, s["n_updates"], s)
        assert v["progress"] == 0.0
        assert v["std_ceil"] == pytest.approx(0.6)
        assert v["entropy_weight"] == pytest.approx(0.01)


def test_end_reaches_floor():
    s = _sched()
    v = na.schedule_values(4000, s["n_updates"], s)
    assert v["progress"] == pytest.approx(1.0)
    assert v["std_ceil"] == pytest.approx(0.03)
    assert v["entropy_weight"] == pytest.approx(0.0)


def test_anneal_is_monotone_decreasing():
    s = _sched()
    prev_std, prev_ent = 1e9, 1e9
    for i in range(2000, 4001, 100):
        v = na.schedule_values(i, s["n_updates"], s)
        assert v["std_ceil"] <= prev_std + 1e-12
        assert v["entropy_weight"] <= prev_ent + 1e-12
        prev_std, prev_ent = v["std_ceil"], v["entropy_weight"]


def test_midpoint_is_geometric_mean_of_std():
    # at progress 0.5 the geometric decay gives sqrt(std_hold*std_floor)
    s = _sched()
    mid = 2000 + (4000 - 2000) * 0.5  # = 3000
    v = na.schedule_values(int(mid), s["n_updates"], s)
    assert v["std_ceil"] == pytest.approx(math.sqrt(0.6 * 0.03), rel=1e-6)


def test_progress_clamps_past_end():
    s = _sched()
    v = na.schedule_values(99999, s["n_updates"], s)
    assert v["progress"] == 1.0 and v["std_ceil"] == pytest.approx(0.03)


# ---- resolve (OFF == None, ON == dict) ----

class _Algo:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Cfg:
    def __init__(self, algo=None, n_updates=4000):
        self.algo = algo
        self.n_updates = n_updates


def test_resolve_off_when_no_algo():
    assert na.resolve_noise_anneal(_Cfg(algo=None)) is None


def test_resolve_off_when_flag_absent_or_false():
    assert na.resolve_noise_anneal(_Cfg(algo=_Algo(entropy_weight=0.01))) is None
    assert na.resolve_noise_anneal(_Cfg(algo=_Algo(noise_anneal=False, entropy_weight=0.01))) is None


def test_resolve_on_uses_defaults_and_live_entropy():
    s = na.resolve_noise_anneal(_Cfg(algo=_Algo(noise_anneal=True, entropy_weight=0.02)))
    assert s is not None
    assert s["hold_frac"] == na.DEFAULT_HOLD_FRAC
    assert s["std_hold"] == na.DEFAULT_STD_HOLD
    assert s["std_floor"] == na.DEFAULT_STD_FLOOR
    assert s["entropy_hold"] == 0.02         # defaults to live cfg.algo.entropy_weight
    assert s["entropy_floor"] == na.DEFAULT_ENTROPY_FLOOR
    assert s["n_updates"] == 4000


def test_resolve_on_respects_overrides():
    s = na.resolve_noise_anneal(_Cfg(algo=_Algo(
        noise_anneal=True, entropy_weight=0.01,
        noise_hold_frac=0.6, noise_std_hold=1.0, noise_std_floor=0.02,
        noise_entropy_hold=0.005, noise_entropy_floor=0.001)))
    assert (s["hold_frac"], s["std_hold"], s["std_floor"]) == (0.6, 1.0, 0.02)
    assert (s["entropy_hold"], s["entropy_floor"]) == (0.005, 0.001)


# ---- apply (real clamp on a torch Parameter; reproduces the rc1-saturation fix) ----

class _FakeActor:
    pass


class _FakeInner:
    pass


def _fake_agent(logstd_vals):
    import torch
    a = type("A", (), {})()
    a.entropy_weight = 0.01
    a.agent = _FakeInner()
    a.agent.actor = _FakeActor()
    a.agent.actor.actor_logstd = torch.nn.Parameter(torch.tensor([logstd_vals], dtype=torch.float32))
    return a


def test_apply_clamps_rc1_saturated_channels():
    # rc1 seed0 raw params: dims 0,3 saturated (9.0, 4.3), dims 1,2 healthy (-0.09, -0.07).
    a = _fake_agent([9.0, -0.09, -0.07, 4.3])
    s = _sched()
    v = na.apply_noise_schedule(a, 0, s)               # hold -> ceiling std 0.6
    p_ceil = na.std_to_param(0.6)
    ls = a.agent.actor.actor_logstd.detach().flatten().tolist()
    assert ls[0] == pytest.approx(p_ceil, rel=1e-5)    # 9.0 clamped down
    assert ls[3] == pytest.approx(p_ceil, rel=1e-5)    # 4.3 clamped down
    assert ls[1] == pytest.approx(-0.09, rel=1e-5)     # healthy: untouched (below ceiling)
    assert ls[2] == pytest.approx(-0.07, rel=1e-5)
    assert a.entropy_weight == pytest.approx(0.01)     # hold entropy
    # the two saturated channels now sample at <= std 0.6 instead of exp(2)=7.39
    assert na.param_to_std(ls[0]) == pytest.approx(0.6, rel=1e-5)


def test_apply_at_end_drives_all_to_floor_and_zero_entropy():
    a = _fake_agent([9.0, -0.09, -0.07, 4.3])
    s = _sched()
    na.apply_noise_schedule(a, 4000, s)                # end -> ceiling std 0.03
    ls = a.agent.actor.actor_logstd.detach().flatten().tolist()
    for p in ls:
        assert na.param_to_std(p) <= 0.03 + 1e-6       # every channel near-deterministic
    assert a.entropy_weight == pytest.approx(0.0)


def test_apply_clamp_only_lowers_never_raises():
    # a policy already tighter than the ceiling must NOT be widened by the clamp.
    a = _fake_agent([-1.0, -1.0, -1.0, -1.0])          # std well below 0.6
    before = a.agent.actor.actor_logstd.detach().flatten().tolist()
    na.apply_noise_schedule(a, 0, _sched())            # ceiling std 0.6
    after = a.agent.actor.actor_logstd.detach().flatten().tolist()
    assert after == pytest.approx(before)              # unchanged


# ---- B2 M3 (2026-07-06): the curriculum schedule targets ----

def test_b2_m3_schedule_targets():
    """B2 M3 pin (2026-07-06): the curriculum _ANNEAL overrides (std_hold 0.35, std_floor 0.10) with
    module-default hold_frac 0.5 must give: ceiling 0.35 through the FULL front half (>= the 0.18
    boundary logstd reset of rl/inc8_warmstart.py, so the clamp is a no-op at critic-warmup unfreeze),
    ceiling >= 0.15 until ~84% of the stage, and ceiling 0.10 + entropy 0.0 at stage end."""
    s = dict(hold_frac=0.5, std_hold=0.35, std_floor=0.10,
             entropy_hold=0.01, entropy_floor=0.0, n_updates=2000)
    assert na.schedule_values(0, 2000, s)["std_ceil"] == pytest.approx(0.35)
    assert na.schedule_values(100, 2000, s)["std_ceil"] >= 0.18   # boundary reset under ceiling at unfreeze
    assert na.schedule_values(1000, 2000, s)["std_ceil"] == pytest.approx(0.35)  # flat through front half
    assert na.schedule_values(1676, 2000, s)["std_ceil"] >= 0.15  # >=0.15 well past the front half
    v = na.schedule_values(2000, 2000, s)
    assert v["std_ceil"] == pytest.approx(0.10)
    assert v["entropy_weight"] == pytest.approx(0.0)
