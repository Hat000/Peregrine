"""Tests for the audit-C5 GRADED ANTI-CLIP terminal + the once-per-gate parabola LATCH (rl/ego_reward.py).

Audit finding (handoff/audit-ego-inc9-2026-07-09/verdicts.md C5): with rw_parabola_crossing ON the
frame-clip/miss terminal penalties are DROPPED (only floor+oob keep theirs), so at the de-facto fixed
cross_zero_m=4.0 a sub-aperture FRAME STRIKE -- a competition DQ -- earns ~+19 of the +20 peak, nearly
indistinguishable from a clean thread; the perfect-estimator run (vglpns0) kept a 35% collision rate with
NO reward pressure against it. ``rw_clip_terminal`` (attr clip_terminal_w) ADDS a flat -W on the frame
strike the parabola dropped, read fresh each step so the train-loop 0->W anneal needs no plumbing.

Separately the wiring audit (read_env-reward-wiring.md RED FLAGS) documents a parabola RE-PAYMENT FARM:
``crossed``==fwd_t is not idempotent per gate, so miss_terminates=false + parabola lets an oscillating
drone re-cross the SAME target plane and farm the parabola every step. ``rw_parabola_latch``
(attr parabola_latch_once) gates a once-per-gate latch that pays each gate at most once per episode.

Coverage (prompt task 3):
  (a) GOLDEN byte-identical-when-off   both fields at their OFF default -> reward/r_prog/components
                                       identical to a cfg that never mentions them, over a seeded batch.
  (b) CLIP event pays -W exactly once  on a frame strike (and NOT on a floor dive / wide miss / clean cross).
  (c) LATCH prevents double payment    a re-cross of the same gate pays 0 while the latch is set.
  (d) from_cfg parsing                 rw_clip_terminal -> clip_terminal_w, rw_parabola_latch -> parabola_latch_once.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_reward_clip_terminal.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import ego_reward as R                                                    # noqa: E402

DT = torch.float64


def _t(x):
    return torch.tensor(x, dtype=DT)


def _seeded_step_kwargs(n, g):
    """A seeded synthetic step for a batch of ``n`` envs -- every mandatory compute_ego_reward tensor plus
    the parabola inputs (cross_offset/crossed/floor_contact) and a random mix of contact/miss/cross events so
    the parabola-on terminal branch is exercised. Deterministic via the passed torch.Generator ``g``."""
    coll = torch.rand(n, generator=g, dtype=DT) < 0.30
    floor = (torch.rand(n, generator=g, dtype=DT) < 0.50) & coll             # some collisions are floor dives
    miss = (torch.rand(n, generator=g, dtype=DT) < 0.30) & ~coll
    crossed = torch.rand(n, generator=g, dtype=DT) < 0.60
    return dict(
        s_curr=torch.rand(n, generator=g, dtype=DT),
        s_prev=torch.rand(n, generator=g, dtype=DT),
        gate_passed=(torch.rand(n, generator=g, dtype=DT) < 0.2) & ~coll & ~miss,
        pass_linf=torch.rand(n, generator=g, dtype=DT),
        w_g_half=0.75,
        gate_collision=coll,
        gate_miss=miss,
        oob=torch.zeros(n, dtype=torch.bool),
        banked_progress_return=torch.rand(n, generator=g, dtype=DT) * 10.0,
        newly_finished=torch.zeros(n, dtype=torch.bool),
        time_left_s=torch.rand(n, generator=g, dtype=DT) * 3.0,
        tilt_cos_r33=torch.rand(n, generator=g, dtype=DT) * 0.1 + 0.9,
        omega=torch.randn(n, 3, generator=g, dtype=DT) * 0.1,
        action_norm=torch.rand(n, 4, generator=g, dtype=DT),
        last_action_norm=torch.rand(n, 4, generator=g, dtype=DT),
        vel_world=torch.randn(n, 3, generator=g, dtype=DT),
        curr_center=torch.randn(n, 3, generator=g, dtype=DT),
        next_center=torch.randn(n, 3, generator=g, dtype=DT),
        dt=1.0 / 30.0,
        cross_offset=torch.rand(n, generator=g, dtype=DT) * 2.0,
        crossed=crossed,
        floor_contact=floor.to(DT),
    )


# ================================================================================================
# (a) GOLDEN: both new fields at their OFF default -> byte-identical to a cfg that never mentions them.
# ================================================================================================
def test_off_is_byte_identical_over_seeded_trajectory():
    """A cfg mentioning NEITHER new key vs a cfg setting them both to their OFF values (rw_clip_terminal=0,
    rw_parabola_latch=False) must produce IDENTICAL reward, r_prog and every component over a seeded multi-
    step batch -- the parabola is ON so the new code paths are live, and OFF they are literal no-ops."""
    # base cfg: parabola ON (so the audit code paths are reachable), new keys ABSENT.
    base = dict(rw_parabola_crossing=True, rw_cross_center=20.0, rw_cross_zero_m=4.0, rw_cross_neg_cap=100.0)
    w_absent = R.EgoRewardWeights.from_cfg(SimpleNamespace(**base))
    w_off = R.EgoRewardWeights.from_cfg(SimpleNamespace(**base, rw_clip_terminal=0.0, rw_parabola_latch=False))
    # from_cfg default handling: the two weight objects are field-for-field equal.
    assert w_absent == w_off
    assert w_absent.clip_terminal_w == 0.0 and w_absent.parabola_latch_once is False

    g = torch.Generator().manual_seed(20260709)
    for _ in range(6):                                       # a short seeded "trajectory" batch
        kw = _seeded_step_kwargs(8, g)
        r_absent, c_absent, rp_absent = R.compute_ego_reward(w_absent, **kw)
        r_off, c_off, rp_off = R.compute_ego_reward(w_off, **kw)
        assert torch.equal(r_absent, r_off)
        assert torch.equal(rp_absent, rp_off)
        assert c_absent == c_off                             # identical component dicts


# ================================================================================================
# (b) CLIP event pays -W EXACTLY ONCE on a frame strike (not floor / not miss / not clean cross).
# ================================================================================================
def _clip_scenario_kwargs():
    """4 envs, parabola ON: env0 FRAME STRIKE (collision, not floor), env1 FLOOR DIVE (collision+floor),
    env2 WIDE MISS, env3 CLEAN cross. Only env0 is the frame strike the parabola dropped."""
    n = 4
    coll = _t([1.0, 1.0, 0.0, 0.0]).bool()                  # env0 frame, env1 floor
    floor = _t([0.0, 1.0, 0.0, 0.0])                        # env1 is the floor dive
    miss = _t([0.0, 0.0, 1.0, 0.0]).bool()                  # env2 wide miss
    crossed = _t([1.0, 0.0, 1.0, 1.0]).bool()              # frame/miss/clean cross the plane; floor does not
    return n, dict(
        s_curr=torch.zeros(n, dtype=DT), s_prev=torch.zeros(n, dtype=DT),
        gate_passed=_t([0.0, 0.0, 0.0, 1.0]).bool(), pass_linf=_t([0.5, 0.0, 1.2, 0.1]), w_g_half=0.75,
        gate_collision=coll, gate_miss=miss, oob=torch.zeros(n, dtype=torch.bool),
        banked_progress_return=_t([10.0, 10.0, 10.0, 10.0]),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=_t([2.0, 2.0, 2.0, 2.0]),
        tilt_cos_r33=torch.full((n,), 0.999, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT), curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1.0 / 30.0,
        cross_offset=_t([0.5, 0.0, 1.2, 0.1]), crossed=crossed, floor_contact=floor,
    )


def test_clip_terminal_pays_minus_w_once_on_frame_strike():
    """clip_terminal_w>0 subtracts EXACTLY W from the frame-strike env's reward and NOTHING from the floor
    dive / wide miss / clean cross -> reward_on - reward_off == [-W, 0, 0, 0]. Isolates the added penalty
    (all other terms are identical between the two configs) and proves it fires once, only on the frame
    strike the parabola dropped."""
    n, kw = _clip_scenario_kwargs()
    common = dict(parabola_crossing=True, cross_center=20.0, cross_zero_m=4.0, cross_neg_cap=100.0)
    W = 5.0
    w_off = R.EgoRewardWeights(clip_terminal_w=0.0, **common)
    w_on = R.EgoRewardWeights(clip_terminal_w=W, **common)
    r_off, c_off, _ = R.compute_ego_reward(w_off, **kw)
    r_on, c_on, _ = R.compute_ego_reward(w_on, **kw)
    diff = (r_on - r_off)
    assert torch.allclose(diff, _t([-W, 0.0, 0.0, 0.0]), atol=1e-9), diff
    # the extra penalty lands in terminal_pen (a terminal event); mean over 4 envs == W/4.
    assert (c_on["terminal_pen"] - c_off["terminal_pen"]) == pytest.approx(W / n, abs=1e-9)
    # OFF (W=0): the frame strike still pays only the parabola -> its terminal_pen contribution is 0 (the
    # dropped-terminal behaviour the audit flagged), so the ONLY difference on env0 is the added -W.
    term_off = R.terminal_penalty(kw["floor_contact"].bool(), torch.zeros(n, dtype=torch.bool), kw["oob"],
                                  kw["banked_progress_return"], w_off, forfeit_mask=kw["floor_contact"])
    assert term_off[0].item() == 0.0                        # env0 frame strike pays NO terminal when off


def test_clip_terminal_off_by_default_and_only_under_parabola():
    """clip_terminal_w defaults to 0 (byte-identical), and even when >0 it does nothing when the parabola is
    OFF (the legacy terminal already fully penalises the frame clip in that regime)."""
    n, kw = _clip_scenario_kwargs()
    # parabola OFF: clip_terminal_w must have NO effect (legacy terminal path owns the collision).
    w0 = R.EgoRewardWeights(parabola_crossing=False, clip_terminal_w=0.0)
    wW = R.EgoRewardWeights(parabola_crossing=False, clip_terminal_w=5.0)
    r0, _, _ = R.compute_ego_reward(w0, **kw)
    rW, _, _ = R.compute_ego_reward(wW, **kw)
    assert torch.equal(r0, rW)                              # no parabola -> clip_terminal_w inert
    assert R.EgoRewardWeights().clip_terminal_w == 0.0      # default OFF


def test_clip_terminal_read_fresh_supports_anneal():
    """clip_terminal_w is a plain mutable python float read FRESH each step: mutating it on the live weights
    object (the _egorw pattern the train-loop anneal uses) changes the next step's frame-strike penalty with
    no re-construction -- the contract the 0->W anneal relies on."""
    n, kw = _clip_scenario_kwargs()
    w = R.EgoRewardWeights(parabola_crossing=True, cross_center=20.0, cross_zero_m=4.0, clip_terminal_w=0.0)
    assert isinstance(w.clip_terminal_w, float)
    r_a, _, _ = R.compute_ego_reward(w, **kw)
    w.clip_terminal_w = 8.0                                 # anneal step (in-place attr mutation)
    r_b, _, _ = R.compute_ego_reward(w, **kw)
    assert (r_a[0].item() - r_b[0].item()) == pytest.approx(8.0, abs=1e-9)   # frame strike now pays -8 more


# ================================================================================================
# (c) LATCH prevents double payment (once-per-gate) when on; without it the re-cross FARMS the parabola.
# ================================================================================================
def test_parabola_latch_suppresses_recross_payment():
    """With latch_once + a per-env paid_latch, the FIRST crossing of a gate pays the parabola and marks the
    latch; a re-cross of the SAME gate pays 0. A caller CLEAR of the latch entry (the target-advance / episode-
    reset contract) re-enables payment. Contrast: latch off -> the re-cross farms the parabola again."""
    paid = torch.zeros(2, dtype=torch.bool)
    e = _t([0.0, 0.0])                                       # dead-centre -> +center each real crossing
    # step 1: env0 crosses (pays), env1 does not.
    r1 = R.crossing_parabola_reward(e, _t([1.0, 0.0]).bool(), 20.0, 0.75, 100.0,
                                    latch_once=True, paid_latch=paid)
    assert r1[0].item() == pytest.approx(20.0) and r1[1].item() == 0.0
    assert paid.tolist() == [True, False]                   # env0 latched in place
    # step 2: BOTH cross -> env0 is already latched (pays 0), env1 crosses for the first time (pays).
    r2 = R.crossing_parabola_reward(e, _t([1.0, 1.0]).bool(), 20.0, 0.75, 100.0,
                                    latch_once=True, paid_latch=paid)
    assert r2[0].item() == 0.0 and r2[1].item() == pytest.approx(20.0)
    assert paid.tolist() == [True, True]
    # caller CLEARS env0's latch (== target advanced / episode reset) -> a later crossing pays again.
    paid[0] = False
    r3 = R.crossing_parabola_reward(e, _t([1.0, 0.0]).bool(), 20.0, 0.75, 100.0,
                                    latch_once=True, paid_latch=paid)
    assert r3[0].item() == pytest.approx(20.0)


def test_parabola_farm_without_latch_default():
    """DEFAULT (latch_once=False / no buffer) is the byte-identical legacy behaviour: a re-cross of the same
    gate pays the parabola EVERY time (the documented re-payment farm the latch defends against)."""
    e = _t([0.0])
    crossed = _t([1.0]).bool()
    r_a = R.crossing_parabola_reward(e, crossed, 20.0, 0.75, 100.0)          # legacy signature (no latch args)
    r_b = R.crossing_parabola_reward(e, crossed, 20.0, 0.75, 100.0)          # re-cross pays AGAIN
    assert r_a.item() == pytest.approx(20.0) and r_b.item() == pytest.approx(20.0)
    # latch_once=True but no buffer supplied -> still legacy (no state to latch on) -> pays again.
    r_c = R.crossing_parabola_reward(e, crossed, 20.0, 0.75, 100.0, latch_once=True, paid_latch=None)
    assert r_c.item() == pytest.approx(20.0)


def test_parabola_latch_wired_through_compute_ego_reward():
    """The latch is reachable from compute_ego_reward: parabola_latch_once + a persisted parabola_paid buffer
    suppress the second same-gate crossing's parabola across two steps."""
    n = 1
    paid = torch.zeros(n, dtype=torch.bool)
    w = R.EgoRewardWeights(parabola_crossing=True, cross_center=20.0, cross_zero_m=0.75,
                           parabola_latch_once=True)
    kw = dict(
        s_curr=torch.zeros(n, dtype=DT), s_prev=torch.zeros(n, dtype=DT),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=_t([0.0]), w_g_half=0.75,
        gate_collision=torch.zeros(n, dtype=torch.bool), gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool), banked_progress_return=torch.zeros(n, dtype=DT),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=_t([2.0]),
        tilt_cos_r33=_t([0.999]), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT), curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1.0 / 30.0,
        cross_offset=_t([0.0]), crossed=_t([1.0]).bool(), floor_contact=_t([0.0]),
    )
    _, c1, _ = R.compute_ego_reward(w, parabola_paid=paid, **kw)
    _, c2, _ = R.compute_ego_reward(w, parabola_paid=paid, **kw)             # same gate re-cross
    assert c1["cross_parabola_reward"] == pytest.approx(20.0)
    assert c2["cross_parabola_reward"] == pytest.approx(0.0)                 # latched -> suppressed
    assert bool(paid[0]) is True


# ================================================================================================
# (d) from_cfg parsing: the short cfg keys map to the exact contract attribute names.
# ================================================================================================
def test_from_cfg_parses_new_fields():
    # absent -> defaults (byte-identical).
    w0 = R.EgoRewardWeights.from_cfg(SimpleNamespace())
    assert w0.clip_terminal_w == 0.0
    assert w0.parabola_latch_once is False
    # the SHORT keys (rw_clip_terminal / rw_parabola_latch) map to the exact contract attrs.
    w1 = R.EgoRewardWeights.from_cfg(SimpleNamespace(rw_clip_terminal=7.5, rw_parabola_latch=True))
    assert w1.clip_terminal_w == pytest.approx(7.5)
    assert isinstance(w1.clip_terminal_w, float)            # plain python float (anneal-mutable)
    assert w1.parabola_latch_once is True
    # types coerced (float / bool) as the rest of from_cfg does.
    w2 = R.EgoRewardWeights.from_cfg(SimpleNamespace(rw_clip_terminal=3, rw_parabola_latch=0))
    assert w2.clip_terminal_w == pytest.approx(3.0) and isinstance(w2.clip_terminal_w, float)
    assert w2.parabola_latch_once is False
