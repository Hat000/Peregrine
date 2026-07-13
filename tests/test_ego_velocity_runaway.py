"""Tests for the ANTI-VELOCITY-RUNAWAY package (2026-07-13): the velocity-cap soft-hinge reward term
(rl/ego_reward.py) + spawn-velocity randomization (rl/peregrine_racing.py).

The champion (_pef / vpeffs0) has a velocity RUNAWAY: optimal speed ~8-9 m/s but it accelerates to
~12 m/s where control fails. This package adds (1) a one-sided soft-hinge penalty on TOTAL SPEED ||v||
that is EXACTLY ZERO at/below a soft cap and ramps up prohibitively toward a hard cap, and (2) a
spawn-velocity knob that exposes the runaway regime by starting a fraction of episodes moving fast at
the first gate. Both are DEFAULT-OFF / byte-identical to the proven _pef path.

Both pieces are PURE torch functions (no diffaero), so the whole package unit-tests on the laptop.

Coverage:
  VELOCITY-CAP  (a) exactly 0 at/below the soft cap (byte-identical -- "no speed term, don't lose
                    rewards elsewhere");
                (b) MONOTONE increasing in magnitude above soft;
                (c) ~PROHIBITIVE near the hard cap (== rw_v_cap at hard, > beyond);
                (d) CONTINUITY (C^1) at the soft knee (value AND slope 0 -> quadratic, not a cliff);
                (e) OFF when the weight is 0; default weight 0 / defaults 9 & 12;
                (f) wired into compute_ego_reward: PARITY below soft (reward unchanged vs OFF) + it
                    LOWERS reward above soft; component 'vcap_pen' exposed;
                (g) __post_init__ guard fires when v_cap_hard <= v_cap_soft with v_cap>0.
  SPAWN-VEL     (h) OFF (frac<=0) -> ZERO velocity, byte-identical, for arbitrary draws/directions;
                (i) FRACTION respected (controlled selection draw);
                (j) magnitude BOUNDED by spawn_vel_max;
                (k) DIRECTION toward the target gate (== gate 0 for a standing-start layout);
                (l) degenerate zero-length to_gate -> zero velocity.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_velocity_runaway.py -q
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import ego_reward as R                                                    # noqa: E402
from peregrine_racing import spawn_velocity_toward_gate                   # noqa: E402

DT = torch.float64


def _t(x):
    return torch.tensor(x, dtype=DT)


# ================================================================================================
# VELOCITY-CAP soft-hinge -- pure velocity_cap_penalty.
# ================================================================================================
def test_velocity_cap_exactly_zero_at_and_below_soft():
    """EXACTLY ZERO at/below the soft cap (the relu knee) -> the reward is UNCHANGED across the whole
    useful speed band (owner Fengyou: "no speed term, don't lose rewards elsewhere"). Byte-identical to
    an all-zeros tensor (not merely approx 0)."""
    speed = _t([0.0, 3.0, 8.0, 8.999, 9.0])              # all at/below soft=9
    r = R.velocity_cap_penalty(speed, rw_v_cap=30.0, v_cap_soft=9.0, v_cap_hard=12.0)
    assert torch.equal(r, torch.zeros_like(r)), r        # EXACTLY zero (relu -> 0 below the knee)


def test_velocity_cap_monotone_increasing_above_soft():
    """MONOTONE increasing in magnitude above the soft cap (the quadratic hinge grows with excess)."""
    speed = _t([9.0, 10.0, 11.0, 12.0, 13.0, 15.0])
    r = R.velocity_cap_penalty(speed, rw_v_cap=30.0, v_cap_soft=9.0, v_cap_hard=12.0)
    mag = -r                                             # penalty magnitude (>=0)
    assert (mag[1:] > mag[:-1]).all(), mag               # strictly increasing above soft
    assert mag[0].item() == pytest.approx(0.0)           # 0 exactly at the knee


def test_velocity_cap_prohibitive_near_hard():
    """At the HARD cap the normalised excess is 1 so the penalty magnitude == rw_v_cap, and it grows
    ABOVE rw_v_cap super-linearly beyond it -> prohibitive vs an O(1-2)/step progress reward."""
    rw = 30.0
    r_hard = R.velocity_cap_penalty(_t([12.0]), rw, 9.0, 12.0)
    assert r_hard.item() == pytest.approx(-rw)           # penalty == -rw_v_cap AT the hard cap
    r_beyond = R.velocity_cap_penalty(_t([15.0]), rw, 9.0, 12.0)   # excess = 2 -> 4*rw
    assert (-r_beyond).item() == pytest.approx(4.0 * rw) # 30*(6/3)^2 = 120 >> rw (runaway tail)
    assert (-r_beyond).item() > rw


def test_velocity_cap_c1_continuity_at_soft_knee():
    """C^1 at the soft knee: value AND slope are 0 there (a QUADRATIC hinge, not a linear cliff). Just
    above the knee the penalty is O(delta^2) -- vanishingly small vs the linear hinge it replaces, so the
    policy meets a smooth ramp, not a discontinuous wall ("reacts better to smooth things")."""
    rw, soft, hard = 30.0, 9.0, 12.0
    span = hard - soft
    delta = 1e-4
    at_knee = R.velocity_cap_penalty(_t([soft]), rw, soft, hard)
    just_above = R.velocity_cap_penalty(_t([soft + delta]), rw, soft, hard)
    assert at_knee.item() == 0.0                                        # value 0 at the knee
    quad = rw * (delta / span) ** 2                                     # expected quadratic value
    assert (-just_above).item() == pytest.approx(quad, rel=1e-6)
    # the quadratic value is ~span/delta * ... orders below the LINEAR hinge (rw*delta/span) -> smooth knee
    linear = rw * (delta / span)
    assert (-just_above).item() < 1e-3 * linear                        # slope -> 0 (quadratic, not linear)


def test_velocity_cap_off_when_weight_zero():
    """rw_v_cap==0 -> zeros byte-identical (default OFF on every stage). Defaults: weight 0, caps 9 & 12."""
    speed = _t([5.0, 11.0, 20.0])                        # incl. speeds ABOVE both caps
    r = R.velocity_cap_penalty(speed, rw_v_cap=0.0, v_cap_soft=9.0, v_cap_hard=12.0)
    assert torch.equal(r, torch.zeros_like(r))
    w = R.EgoRewardWeights()
    assert w.v_cap == 0.0                                # dataclass default OFF
    assert w.v_cap_soft == 9.0 and w.v_cap_hard == 12.0


def test_velocity_cap_post_init_guard():
    """The __post_init__ guard fires ONLY when armed (v_cap>0) with hard <= soft; default (v_cap=0) is a
    no-op (byte-identical) even with a degenerate hard<=soft, and a well-ordered armed config is accepted."""
    R.EgoRewardWeights()                                              # default -> no raise
    R.EgoRewardWeights(v_cap=0.0, v_cap_soft=12.0, v_cap_hard=9.0)    # OFF -> guard skipped, no raise
    R.EgoRewardWeights(v_cap=30.0, v_cap_soft=9.0, v_cap_hard=12.0)   # armed + well-ordered -> ok
    with pytest.raises(AssertionError):
        R.EgoRewardWeights(v_cap=30.0, v_cap_soft=12.0, v_cap_hard=12.0)   # hard == soft
    with pytest.raises(AssertionError):
        R.EgoRewardWeights(v_cap=30.0, v_cap_soft=12.0, v_cap_hard=9.0)    # hard < soft


# ------------------------------------------------------------------------------------------------
# VELOCITY-CAP wired into compute_ego_reward: PARITY below soft + LOWERS reward above soft.
# ------------------------------------------------------------------------------------------------
def _reward_kw(n, vel_world):
    """A minimal compute_ego_reward kwargs dict with a controllable vel_world (all other terms inert)."""
    return dict(
        s_curr=torch.zeros(n, dtype=DT), s_prev=torch.zeros(n, dtype=DT),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=torch.zeros(n, dtype=DT),
        w_g_half=0.375,
        gate_collision=torch.zeros(n, dtype=torch.bool), gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool), banked_progress_return=torch.zeros(n, dtype=DT),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=torch.zeros(n, dtype=DT),
        tilt_cos_r33=torch.ones(n, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=vel_world, curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1 / 30,
    )


def test_velocity_cap_parity_below_soft_and_penalises_above():
    """OBSERVABILITY + PARITY: ||v|| is priced from vel_world, and ||v_body|| == ||v_world|| (the obs
    carries body velocity at obs[0:3], so the priced quantity is exactly what the policy sees).
      * BELOW soft: the armed reward is BYTE-IDENTICAL to the OFF reward (the proven _pef reward is
        unchanged in the useful speed band).
      * ABOVE soft: the armed reward is STRICTLY LOWER (the cap bites), by exactly the hinge penalty."""
    n = 1
    # speed 6 m/s (below soft=9) -- along an arbitrary axis; the term prices only the NORM
    v_below = _t([[3.6, 4.8, 0.0]])                     # ||v|| = 6.0
    kw = _reward_kw(n, v_below)
    w_off = R.EgoRewardWeights(v_cap=0.0)
    w_on = R.EgoRewardWeights(v_cap=30.0, v_cap_soft=9.0, v_cap_hard=12.0)
    r_off, c_off, _ = R.compute_ego_reward(w_off, **kw)
    r_on, c_on, _ = R.compute_ego_reward(w_on, **kw)
    assert torch.equal(r_on, r_off)                     # BYTE-IDENTICAL below the soft cap
    assert c_on["vcap_pen"] == pytest.approx(0.0)       # component present + zero below soft

    # speed 11 m/s (above soft, below hard): excess = (11-9)/3 = 2/3 -> penalty 30*(2/3)^2 = 13.333
    v_above = _t([[11.0, 0.0, 0.0]])
    kw2 = _reward_kw(n, v_above)
    r_off2, _, _ = R.compute_ego_reward(w_off, **kw2)
    r_on2, c_on2, _ = R.compute_ego_reward(w_on, **kw2)
    expected_pen = 30.0 * (2.0 / 3.0) ** 2
    assert c_on2["vcap_pen"] == pytest.approx(expected_pen)
    assert (r_off2 - r_on2).item() == pytest.approx(expected_pen)     # reward lowered by exactly the hinge
    assert r_on2.item() < r_off2.item()


def test_velocity_cap_component_default_off_reward_is_pef_identical():
    """The _pef reward (default EgoRewardWeights, v_cap=0) is UNCHANGED by the new term at ANY speed --
    including a speed WELL above the caps -- because the term is a no-op when the weight is 0."""
    n = 2
    v = _t([[20.0, 0.0, 0.0], [0.0, 15.0, 0.0]])        # 20 and 15 m/s (both far above the caps)
    kw = _reward_kw(n, v)
    r_default, comps, _ = R.compute_ego_reward(R.EgoRewardWeights(), **kw)
    r_explicit_off, _, _ = R.compute_ego_reward(R.EgoRewardWeights(v_cap=0.0), **kw)
    assert torch.equal(r_default, r_explicit_off)
    assert comps["vcap_pen"] == pytest.approx(0.0)      # OFF -> no penalty even at 20 m/s


# ================================================================================================
# SPAWN-VELOCITY randomization -- pure spawn_velocity_toward_gate.
# ================================================================================================
def test_spawn_vel_off_is_zero_byte_identical():
    """spawn_vel_frac<=0 -> the WHOLE velocity tensor is zeros (byte-identical at-rest spawn) for ARBITRARY
    selection/magnitude draws and directions -> the caller leaves state[:, 7:10] at 0. This is the
    default-off parity: the proven _pef path spawns at rest."""
    n = 5
    to_gate = torch.randn(n, 3, dtype=DT)
    sel = torch.rand(n, dtype=DT)
    mag = torch.rand(n, dtype=DT)
    for frac in (0.0, -0.5):
        v0 = spawn_velocity_toward_gate(to_gate, sel, mag, spawn_vel_frac=frac, spawn_vel_max=15.0)
        assert torch.equal(v0, torch.zeros_like(v0)), (frac, v0)


def test_spawn_vel_fraction_respected():
    """With a CONTROLLED selection draw, EXACTLY the envs whose sel_draw < spawn_vel_frac move; the rest
    stay at rest (zero velocity)."""
    # sel = [0.1, 0.4, 0.6, 0.9]; frac 0.5 -> envs 0,1 move, envs 2,3 rest
    to_gate = _t([[1.0, 0.0, 0.0]] * 4)
    sel = _t([0.1, 0.4, 0.6, 0.9])
    mag = _t([1.0, 1.0, 1.0, 1.0])                      # full magnitude for the movers
    v0 = spawn_velocity_toward_gate(to_gate, sel, mag, spawn_vel_frac=0.5, spawn_vel_max=15.0)
    moving = torch.linalg.norm(v0, dim=-1) > 0
    assert moving.tolist() == [True, True, False, False]
    assert torch.equal(v0[2], torch.zeros(3, dtype=DT))
    assert torch.equal(v0[3], torch.zeros(3, dtype=DT))


def test_spawn_vel_magnitude_bounded_by_max():
    """The magnitude of every moving env's velocity is bounded by spawn_vel_max (mag_draw U[0,1) ->
    speed in [0, spawn_vel_max)); non-moving envs are exactly 0."""
    n = 256
    torch.manual_seed(0)
    to_gate = torch.randn(n, 3, dtype=DT) * 5.0          # varied ranges/directions
    sel = torch.zeros(n, dtype=DT)                       # all selected (sel=0 < frac)
    mag = torch.rand(n, dtype=DT)                        # U[0,1)
    vmax = 15.0
    v0 = spawn_velocity_toward_gate(to_gate, sel, mag, spawn_vel_frac=1.0, spawn_vel_max=vmax)
    speed = torch.linalg.norm(v0, dim=-1)
    assert (speed <= vmax + 1e-9).all(), speed.max()
    # an out-of-range mag draw is still clamped so the bound holds defensively
    v_clamp = spawn_velocity_toward_gate(to_gate, sel, torch.full((n,), 5.0, dtype=DT),
                                         spawn_vel_frac=1.0, spawn_vel_max=vmax)
    assert (torch.linalg.norm(v_clamp, dim=-1) <= vmax + 1e-9).all()


def test_spawn_vel_direction_toward_gate():
    """DIRECTION: a moving env's velocity is PARALLEL to to_gate (cos == 1). For a standing-start layout
    (to_gate = gate0_centre - spawn), this is 'toward the FIRST gate' -- the _pef deploy case."""
    spawn = _t([[0.0, 0.0, 1.0], [2.0, -3.0, 0.5]])
    gate0 = _t([[15.0, 0.0, 2.0], [2.0, 12.0, 4.0]])    # two standing-start layouts
    to_gate = gate0 - spawn
    sel = torch.zeros(2, dtype=DT)                       # both selected
    mag = _t([0.5, 1.0])
    v0 = spawn_velocity_toward_gate(to_gate, sel, mag, spawn_vel_frac=1.0, spawn_vel_max=15.0)
    for i in range(2):
        cos = torch.dot(v0[i], to_gate[i]) / (v0[i].norm() * to_gate[i].norm())
        assert cos.item() == pytest.approx(1.0)          # velocity points AT the (first) gate
    # magnitude == mag_draw * vmax along that direction
    assert v0[0].norm().item() == pytest.approx(0.5 * 15.0)
    assert v0[1].norm().item() == pytest.approx(1.0 * 15.0)


def test_spawn_vel_degenerate_zero_length_to_gate():
    """A degenerate zero-length to_gate (spawn coincident with the gate) yields ZERO velocity for that env
    (safe normalize) -- no NaN, no blow-up."""
    to_gate = _t([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sel = torch.zeros(2, dtype=DT)
    mag = _t([1.0, 1.0])
    v0 = spawn_velocity_toward_gate(to_gate, sel, mag, spawn_vel_frac=1.0, spawn_vel_max=15.0)
    assert torch.equal(v0[0], torch.zeros(3, dtype=DT))  # degenerate -> 0 (no NaN)
    assert torch.isfinite(v0).all()
    assert v0[1].norm().item() == pytest.approx(15.0)    # the well-posed env still moves
