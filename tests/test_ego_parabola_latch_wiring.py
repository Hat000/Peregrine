"""Tests for the ENV-SIDE wiring of the once-per-gate parabola latch buffer (``parabola_paid``) in
rl/peregrine_racing_ego.py.

Contract (rl/ego_reward.py crossing_parabola_reward docstring + tests/test_ego_reward_clip_terminal.py):
the reward fn marks the caller-owned per-env bool buffer IN PLACE on a forward crossing (only when
``parabola_latch_once``); the CALLER (the env) must clear it (1) on a target ADVANCE (new gate -> new
payment window) and (2) on EVERY episode reset path INCLUDING truncation/timeout -- a stale latch would
suppress the next episode's first crossing payment.

The PeregrineRacingEgo class itself CANNOT be constructed without DiffAero (the base class import falls
back to ``object``; ``super().__init__(cfg, device)`` raises) -- so, per the integration brief, coverage
is at the REWARD-PATH level:
  (w) WIRING-PRESENT: inspect.getsource guards that the env code allocates the buffer, passes it into
      compute_ego_reward, clears on advance in step(), and clears in reset_idx (the single choke point
      both terminated AND truncated envs funnel through).
  (i) FARM DEFENSE: with the latch on, a re-cross of the SAME gate (no advance -- the
      miss_terminates=false oscillation) pays 0 after the first payment.
  (ii) ADVANCE-CLEAR: the env's post-reward ``paid[advance]=False`` re-opens the window so the NEXT
      gate's crossing pays (simulated exactly in env step order: reward THEN clear).
  (iii) RESET-CLEAR incl. TRUNCATION: a truncation-path reset clear lets the next episode's first
      crossing pay; the REGRESSION contrast (no clear) shows the stale latch suppressing it.
  (iv) BYTE-IDENTITY: latch flag OFF (default) + the buffer PASSED (as the env now always does) ->
      rewards identical to not passing the buffer, and the buffer is never mutated.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_ego_parabola_latch_wiring.py -q
"""
import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import ego_reward as R                                                     # noqa: E402
import peregrine_racing_ego as C                                          # noqa: E402

DT = torch.float64


def _t(x):
    return torch.tensor(x, dtype=DT)


def _step_kwargs(n, crossed, cross_offset=None, gate_passed=None):
    """Minimal quiescent compute_ego_reward kwargs: no contact/miss/oob/finish, zero motion -- so the
    parabola is the ONLY event-driven term that changes between steps."""
    z = torch.zeros(n, dtype=DT)
    zb = torch.zeros(n, dtype=torch.bool)
    return dict(
        s_curr=z.clone(), s_prev=z.clone(),
        gate_passed=(gate_passed if gate_passed is not None else zb.clone()),
        pass_linf=(cross_offset if cross_offset is not None else z.clone()), w_g_half=0.75,
        gate_collision=zb.clone(), gate_miss=zb.clone(), oob=zb.clone(),
        banked_progress_return=z.clone(),
        newly_finished=zb.clone(), time_left_s=torch.full((n,), 2.0, dtype=DT),
        tilt_cos_r33=torch.full((n,), 0.999, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT), curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1.0 / 30.0,
        cross_offset=(cross_offset if cross_offset is not None else z.clone()),
        crossed=crossed, floor_contact=z.clone(),
    )


_W_LATCH = R.EgoRewardWeights(parabola_crossing=True, cross_center=20.0, cross_zero_m=0.75,
                              cross_neg_cap=100.0, parabola_latch_once=True)


# ================================================================================================
# (w) WIRING-PRESENT: the cluster-only env code carries the buffer + both clears + the pass-through.
# ================================================================================================
def test_w_env_source_wires_buffer_pass_and_both_clears():
    """PeregrineRacingEgo needs diffaero to CONSTRUCT, but its methods are defined locally -- assert the
    source of __init__/step/reset_idx contains the allocation, the compute_ego_reward pass-through, the
    post-reward advance-clear, and the reset_idx clear (the terminated+truncated choke point)."""
    src_init = inspect.getsource(C.PeregrineRacingEgo.__init__)
    src_step = inspect.getsource(C.PeregrineRacingEgo.step)
    src_reset = inspect.getsource(C.PeregrineRacingEgo.reset_idx)
    assert "self._parabola_paid = torch.zeros(self.n_envs, dtype=torch.bool" in src_init, \
        "__init__ must allocate the per-env bool latch buffer"
    assert "parabola_paid=self._parabola_paid" in src_step, \
        "step() must pass the env-owned buffer into compute_ego_reward"
    assert "self._parabola_paid[advance] = False" in src_step, \
        "step() must clear the latch on a target ADVANCE"
    # the advance-clear must come AFTER the compute_ego_reward call (the crossing that advanced the
    # target is latched inside the reward; clearing first would let the SAME crossing re-pay next step).
    assert src_step.index("parabola_paid=self._parabola_paid") \
        < src_step.index("self._parabola_paid[advance] = False"), \
        "advance-clear must follow the reward computation"
    assert "self._parabola_paid[env_idx] = False" in src_reset, \
        "reset_idx must clear the latch (covers terminated AND truncated -- both funnel through it)"
    print("\n[w] env source: buffer allocated, passed into compute_ego_reward, advance-clear after "
          "reward, reset_idx clear present")


# ================================================================================================
# (i) FARM DEFENSE: re-cross of the same gate (no advance) pays 0 after the first payment.
# ================================================================================================
def test_i_same_gate_recross_pays_zero_with_latch():
    n = 2
    paid = torch.zeros(n, dtype=torch.bool)                               # env buffer (allocated at init)
    crossed = _t([1.0, 0.0]).bool()
    _, c1, _ = R.compute_ego_reward(_W_LATCH, parabola_paid=paid, **_step_kwargs(n, crossed))
    assert c1["cross_parabola_reward"] == pytest.approx(20.0 / n)         # env0 paid (mean over n)
    assert paid.tolist() == [True, False]
    # miss_terminates=false oscillation: env0 re-crosses the SAME target plane -> pays 0 (latched).
    _, c2, _ = R.compute_ego_reward(_W_LATCH, parabola_paid=paid, **_step_kwargs(n, crossed))
    assert c2["cross_parabola_reward"] == pytest.approx(0.0)
    print(f"\n[i] first cross paid {c1['cross_parabola_reward']:.2f} (mean), re-cross paid "
          f"{c2['cross_parabola_reward']:.2f} -- farm defended")


# ================================================================================================
# (ii) ADVANCE-CLEAR (env step order: reward THEN clear) -> the NEXT gate's crossing pays.
# ================================================================================================
def test_ii_advance_clear_reopens_payment_for_next_gate():
    """Simulate the env step loop exactly: crossing pays + latches inside compute_ego_reward; the env then
    clears paid[advance] (a valid pass advanced the target); the NEW gate's crossing later pays again."""
    n = 1
    paid = torch.zeros(n, dtype=torch.bool)
    crossed = _t([1.0]).bool()
    passed = _t([1.0]).bool()
    # step k: a valid centred pass -> parabola pays, latch set inside the reward call.
    _, c1, _ = R.compute_ego_reward(_W_LATCH, parabola_paid=paid,
                                    **_step_kwargs(n, crossed, gate_passed=passed))
    assert c1["cross_parabola_reward"] == pytest.approx(20.0)
    assert bool(paid[0]) is True
    # env (post-reward): advance = gate_passed & ~is_last -> clear the latch (the new gate's window).
    advance = passed.clone()
    paid[advance] = False                                                  # the env's exact clear
    assert bool(paid[0]) is False
    # step k+m: the drone crosses the NEW target gate's plane -> pays again (fresh window).
    _, c2, _ = R.compute_ego_reward(_W_LATCH, parabola_paid=paid, **_step_kwargs(n, crossed))
    assert c2["cross_parabola_reward"] == pytest.approx(20.0)
    print(f"\n[ii] pass paid {c1['cross_parabola_reward']:.2f} -> advance-clear -> next gate paid "
          f"{c2['cross_parabola_reward']:.2f}")


# ================================================================================================
# (iii) RESET-CLEAR incl. TRUNCATION + the stale-latch REGRESSION contrast.
# ================================================================================================
def test_iii_truncation_reset_clear_vs_stale_latch_regression():
    """Latch set mid-episode; the episode TRUNCATES (timeout -- no terminal event fires). The env funnels
    truncated envs through reset_idx which clears the latch -> the NEXT episode's first crossing pays.
    CONTRAST: withOUT the reset clear (the stale-latch bug), the first crossing is silently suppressed."""
    n = 1
    crossed = _t([1.0]).bool()

    def episode_then_truncate(clear_on_reset: bool):
        paid = torch.zeros(n, dtype=torch.bool)
        # episode A: a wide crossing latches (miss_terminates=false; no advance, no termination) ...
        R.compute_ego_reward(_W_LATCH, parabola_paid=paid,
                             **_step_kwargs(n, crossed, cross_offset=_t([2.0])))
        assert bool(paid[0]) is True
        # ... then the episode times out: truncated -> step() calls reset_idx(reset_indices).
        if clear_on_reset:
            env_idx = torch.tensor([0])
            paid[env_idx] = False                                          # reset_idx's exact clear
        # episode B, first crossing (dead centre):
        _, c, _ = R.compute_ego_reward(_W_LATCH, parabola_paid=paid, **_step_kwargs(n, crossed))
        return c["cross_parabola_reward"]

    paid_next = episode_then_truncate(clear_on_reset=True)
    stale_next = episode_then_truncate(clear_on_reset=False)
    assert paid_next == pytest.approx(20.0), "post-truncation reset clear must re-enable payment"
    assert stale_next == pytest.approx(0.0), "REGRESSION: a stale latch suppresses the first crossing"
    print(f"\n[iii] truncation w/ reset-clear: next-episode first cross pays {paid_next:.2f}; "
          f"stale latch (no clear) pays {stale_next:.2f} -- the bug the reset_idx clear prevents")


# ================================================================================================
# (iv) BYTE-IDENTITY: latch flag OFF + buffer passed (the env default) == no buffer; buffer unmutated.
# ================================================================================================
def test_iv_flag_off_buffer_passed_is_byte_identical_and_unmutated():
    """The env now ALWAYS passes parabola_paid; with parabola_latch_once=False (default) the reward fn
    must neither read nor mutate it -> identical rewards to omitting the buffer, over a seeded batch with
    crossings/passes/collisions mixed in, and the buffer stays all-False."""
    n = 16
    g = torch.Generator().manual_seed(20260709)
    w_off = R.EgoRewardWeights(parabola_crossing=True, cross_center=20.0, cross_zero_m=0.75,
                               cross_neg_cap=100.0, parabola_latch_once=False)   # flag OFF (default)
    paid = torch.zeros(n, dtype=torch.bool)
    for step_i in range(8):
        crossed = torch.rand(n, generator=g, dtype=DT) < 0.5
        off = torch.rand(n, generator=g, dtype=DT) * 2.0
        kw = _step_kwargs(n, crossed, cross_offset=off)
        kw["gate_collision"] = torch.rand(n, generator=g, dtype=DT) < 0.2
        r_with, c_with, rp_with = R.compute_ego_reward(w_off, parabola_paid=paid, **kw)
        r_none, c_none, rp_none = R.compute_ego_reward(w_off, parabola_paid=None, **kw)
        assert torch.equal(r_with, r_none), step_i
        assert torch.equal(rp_with, rp_none), step_i
        assert c_with == c_none, step_i
        assert not paid.any(), "flag OFF must never mutate the buffer"
    # and the env-side clears are no-ops on an all-False buffer (trivially byte-identical).
    paid[torch.zeros(n, dtype=torch.bool)] = False
    assert not paid.any()
    print(f"\n[iv] flag OFF: buffer passed == buffer omitted (8 seeded steps, rewards + components "
          f"bit-identical); buffer never mutated")
