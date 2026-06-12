"""Tests for the S1.4 env-coherence core (rl/peregrine_racing.py pure helpers).

The PeregrineRacing class itself needs diffaero (cluster-only; covered by
rl/peregrine_racing_precheck.py on Adroit). Everything the class delegates its decisions to --
crossing classification, reward composition, spawn quaternions, lookahead tables -- is pure torch
and tested here, including the coherence properties the redesign claims (see the module docstring
of rl/peregrine_racing.py: audit items C1..C6).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy.spatial.transform import Rotation  # noqa: E402

_RL = Path(__file__).resolve().parents[1] / "rl"
sys.path.insert(0, str(_RL))

from peregrine_racing import (RewardWeights, compute_reward_terms, crossing_events,  # noqa: E402
                              quat_xyzw_from_axis_angle, quat_xyzw_from_yaw_pitch,
                              quat_xyzw_mul, rel_tables, roll_from_quat_xyzw,
                              tilt_cos_from_quat_xyzw, world_to_gateframe)

HALF_IN, HALF_OUT = 0.75, 1.36


def _rel(prev, curr):
    return torch.tensor([prev], dtype=torch.float64), torch.tensor([curr], dtype=torch.float64)


# ---------------------------------------------------------------- crossing classification (C2)
def test_center_pass():
    ev = crossing_events(*_rel([-0.2, 0.0, 0.0], [0.3, 0.0, 0.0]), HALF_IN, HALF_OUT)
    assert ev["fwd"].item() and ev["pass_ok"].item() and not ev["in_frame"].item()


def test_interpolation_rescues_fast_diagonal_pass():
    # endpoint is OUTSIDE the aperture (|y|=1.0) but the plane-crossing point is at y=0.25:
    # the old post-crossing-position test called this a collision; the fix classifies the
    # crossing point. prev x=-0.25, curr x=0.75 -> f=0.25; y: 0.0 -> 1.0 interp 0.25.
    ev = crossing_events(*_rel([-0.25, 0.0, 0.0], [0.75, 1.0, 0.0]), HALF_IN, HALF_OUT)
    assert ev["pass_ok"].item() and not ev["in_frame"].item()
    assert abs(ev["linf"].item() - 0.25) < 1e-9


def test_interpolation_catches_fast_diagonal_frame_hit():
    # symmetric trap: endpoint back near the axis but the crossing point in the frame band
    ev = crossing_events(*_rel([-0.25, 1.6, 0.0], [0.75, 0.4, 0.0]), HALF_IN, HALF_OUT)
    # f = 0.25; y at crossing = 1.6 + 0.25*(-1.2) = 1.3 -> frame band (0.75, 1.36]
    assert ev["in_frame"].item() and not ev["pass_ok"].item()
    assert abs(ev["linf"].item() - 1.3) < 1e-9


def test_wide_miss_is_neither_pass_nor_frame():
    ev = crossing_events(*_rel([-0.5, 2.0, 0.0], [0.5, 2.0, 0.0]), HALF_IN, HALF_OUT)
    assert ev["fwd"].item() and not ev["pass_ok"].item() and not ev["in_frame"].item()
    assert ev["linf"].item() > HALF_OUT


def test_backward_crossing_detected_for_frame_but_not_pass():
    ev = crossing_events(*_rel([0.3, 1.0, 0.0], [-0.3, 1.0, 0.0]), HALF_IN, HALF_OUT)
    assert ev["bwd"].item() and not ev["fwd"].item() and ev["in_frame"].item()


def test_no_crossing_no_event():
    ev = crossing_events(*_rel([0.5, 0.0, 0.0], [1.5, 0.0, 0.0]), HALF_IN, HALF_OUT)
    assert not (ev["fwd"].item() or ev["bwd"].item() or ev["pass_ok"].item()
                or ev["in_frame"].item())


def test_linf_uses_both_axes():
    ev = crossing_events(*_rel([-0.5, 0.1, 1.0], [0.5, 0.1, 1.2]), HALF_IN, HALF_OUT)
    assert ev["in_frame"].item()           # z = 1.1 in the band even though y is tiny


# ---------------------------------------------------------------- frames / quaternions
def test_world_to_gateframe_matches_rotmat():
    rng = np.random.default_rng(0)
    yaw = torch.tensor(rng.uniform(-np.pi, np.pi, 32))
    d = torch.tensor(rng.normal(size=(32, 3)))
    got = world_to_gateframe(d, yaw)
    for i in range(32):
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        np.testing.assert_allclose(got[i].numpy(), R @ d[i].numpy(), atol=1e-12)


def test_quat_from_yaw_pitch_matches_scipy():
    rng = np.random.default_rng(1)
    yaw = rng.uniform(-np.pi, np.pi, 16)
    pitch = rng.uniform(-1.2, 1.2, 16)
    q = quat_xyzw_from_yaw_pitch(torch.tensor(yaw), torch.tensor(pitch)).numpy()
    for i in range(16):
        R_expect = Rotation.from_euler("ZYX", [yaw[i], pitch[i], 0.0]).as_matrix()
        R_got = Rotation.from_quat(q[i]).as_matrix()        # scipy is xyzw
        np.testing.assert_allclose(R_got, R_expect, atol=1e-12)


def test_vq1_spawn_quat_reproduced():
    # the measured S1.2 spawn quaternion (xyzw), previously hardcoded in peregrine_racing.py
    stored = np.array([-0.000135, -0.15471, -0.000862, 0.987959])
    q = quat_xyzw_from_yaw_pitch(torch.tensor([0.0]), torch.tensor([-2 * math.asin(0.15471)]))
    assert np.abs(q[0].numpy() - stored).max() < 2e-3


def test_quat_mul_and_axis_angle_match_scipy():
    rng = np.random.default_rng(2)
    a = Rotation.random(8, random_state=3)
    rv = rng.normal(scale=0.3, size=(8, 3))
    b = Rotation.from_rotvec(rv)
    got = quat_xyzw_mul(torch.tensor(a.as_quat()),
                        quat_xyzw_from_axis_angle(torch.tensor(rv))).numpy()
    expect = (a * b).as_quat()
    for i in range(8):                                       # quats are sign-ambiguous
        err = min(np.abs(got[i] - expect[i]).max(), np.abs(got[i] + expect[i]).max())
        assert err < 1e-12


def test_roll_from_quat_matches_scipy_zyx():
    """The env's peak-roll tracker must agree with the ZYX Euler roll the old eval computed
    via pytorch3d (matrix_to_euler_angles 'ZYX' last angle) -- scipy as the arbiter."""
    rots = Rotation.random(64, random_state=11)
    q = torch.tensor(rots.as_quat())                  # xyzw
    got = roll_from_quat_xyzw(q).numpy()
    expect = rots.as_euler("ZYX")[:, 2]               # (yaw, pitch, roll) -> roll
    np.testing.assert_allclose(got, expect, atol=1e-10)


def test_tilt_cos():
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert abs(tilt_cos_from_quat_xyzw(ident).item() - 1.0) < 1e-12
    roll90 = torch.tensor(Rotation.from_euler("X", 90, degrees=True).as_quat())
    assert abs(tilt_cos_from_quat_xyzw(roll90).item()) < 1e-7
    flip = torch.tensor(Rotation.from_euler("X", 180, degrees=True).as_quat())
    assert abs(tilt_cos_from_quat_xyzw(flip).item() + 1.0) < 1e-12
    yaw_only = torch.tensor(Rotation.from_euler("Z", 123, degrees=True).as_quat())
    assert abs(tilt_cos_from_quat_xyzw(yaw_only).item() - 1.0) < 1e-12


# ---------------------------------------------------------------- lookahead tables
def test_rel_tables_wrap_convention():
    rng = np.random.default_rng(4)
    gp = torch.tensor(rng.normal(scale=30, size=(3, 6, 3)))
    gy = torch.tensor(rng.uniform(-np.pi, np.pi, (3, 6)))
    rel, yrel = rel_tables(gp, gy)
    # the loop reference (the parent's [i-1] Python convention)
    for n in range(3):
        for i in range(6):
            c, s = math.cos(gy[n, i - 1]), math.sin(gy[n, i - 1])
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            np.testing.assert_allclose(rel[n, i].numpy(),
                                       R @ (gp[n, i] - gp[n, i - 1]).numpy(), atol=1e-9)
            dy = float(gy[n, i] - gy[n, i - 1])
            assert abs(float(yrel[n, i]) - math.atan2(math.sin(dy), math.cos(dy))) < 1e-9


def test_rel_tables_vq1_all_pi_yaws_give_zero_relyaw():
    import json
    data = json.loads((_RL / "peregrine_course_diffaero.json").read_text())
    gp = torch.tensor([g["pos_zup"] for g in data["gates"]]).unsqueeze(0)
    gy = torch.tensor([g["yaw"] for g in data["gates"]]).unsqueeze(0)
    _, yrel = rel_tables(gp, gy)
    assert yrel.abs().max() < 1e-6        # the fly_rl.py next_relyaw == 0 invariant


# ---------------------------------------------------------------- reward composition + coherence
def _zero_events(n=4):
    f = torch.zeros(n, dtype=torch.bool)
    return dict(gate_passed=f.clone(), gate_collision=f.clone(), gate_miss=f.clone(),
                oob=f.clone(), newly_finished=f.clone())


def _base_kwargs(n=4):
    return dict(
        prev_d2g=torch.full((n,), 10.0), curr_d2g=torch.full((n,), 10.0),
        time_left_s=torch.full((n,), 20.0),
        quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n),
        omega=torch.zeros(n, 3),
        action_norm=torch.full((n, 4), 0.5), last_action_norm=torch.full((n, 4), 0.5),
        **_zero_events(n))


def test_progress_signs():
    w = RewardWeights()
    kw = _base_kwargs()
    kw["curr_d2g"] = torch.full((4,), 9.5)            # approached 0.5 m
    r, comp = compute_reward_terms(w, **kw)
    assert torch.allclose(r, torch.full((4,), w.progress * 0.5 - w.time))
    kw["curr_d2g"] = torch.full((4,), 10.5)           # receded
    r, _ = compute_reward_terms(w, **kw)
    assert (r < 0).all()


def test_collision_cannot_be_bribed_by_progress_plus_passage():
    """Coherence T1: the largest plausible single-step gain (passage + 0.7 m progress at 30 Hz)
    must stay strictly below the frame-strike penalty."""
    w = RewardWeights()
    assert w.collision > w.passage + w.progress * 0.7
    # and a literal crash step nets negative even while approaching the gate fast
    kw = _base_kwargs()
    kw["curr_d2g"] = torch.full((4,), 9.3)
    kw["gate_collision"] = torch.ones(4, dtype=torch.bool)
    r, _ = compute_reward_terms(w, **kw)
    assert (r < 0).all()


def test_miss_ordering():
    """Coherence T2: skipping a gate (miss) must cost more than passing earns, but less than a
    frame strike."""
    w = RewardWeights()
    assert w.passage < w.miss < w.collision


def test_oob_equals_collision():
    w = RewardWeights()
    assert w.oob == w.collision


def test_time_penalty_cannot_motivate_suicide():
    """Coherence R3: the worst-case remaining time cost (full 40 s episode at 30 Hz) must not
    dwarf the terminal penalties -- ending early must never beat flying."""
    w = RewardWeights()
    assert w.time * 1201 < 1.5 * w.collision


def test_tilt_hinge_free_cone():
    w = RewardWeights()
    kw = _base_kwargs()
    # 45 deg tilt: inside the 60 deg free cone -> NO tilt cost
    kw["quat_xyzw"] = torch.tensor(
        Rotation.from_euler("X", 45, degrees=True).as_quat(), dtype=torch.float32
    ).expand(4, 4).clone()
    r45, comp45 = compute_reward_terms(w, **kw)
    assert comp45["tilt_pen"] == 0.0
    # 104 deg (the S1.2 backflip entry): costs > 2/step
    kw["quat_xyzw"] = torch.tensor(
        Rotation.from_euler("X", 104, degrees=True).as_quat(), dtype=torch.float32
    ).expand(4, 4).clone()
    r104, comp104 = compute_reward_terms(w, **kw)
    assert w.tilt * comp104["tilt_pen"] > 2.0
    assert (r104 < r45).all()


def test_finish_includes_time_bonus():
    w = RewardWeights()
    kw = _base_kwargs()
    kw["newly_finished"] = torch.ones(4, dtype=torch.bool)
    kw["gate_passed"] = torch.ones(4, dtype=torch.bool)
    kw["time_left_s"] = torch.tensor([0.0, 5.0, 10.0, 20.0])
    r, _ = compute_reward_terms(w, **kw)
    assert torch.allclose(r[1] - r[0], torch.tensor(w.finish_time * 5.0))
    assert torch.allclose(r[3] - r[2], torch.tensor(w.finish_time * 10.0))


def test_action_rate_penalty():
    w = RewardWeights()
    kw = _base_kwargs()
    r_still, _ = compute_reward_terms(w, **kw)
    kw["action_norm"] = torch.full((4, 4), 1.0)       # full-scale flip on all axes
    kw["last_action_norm"] = torch.zeros(4, 4)
    r_flip, comp = compute_reward_terms(w, **kw)
    assert torch.allclose(r_still - r_flip, torch.full((4,), w.dact * 4.0))


def test_rate_penalty():
    w = RewardWeights()
    kw = _base_kwargs()
    kw["omega"] = torch.tensor([[3.0, 0.0, 4.0]] * 4)   # |w| = 5
    r, _ = compute_reward_terms(w, **kw)
    assert torch.allclose(r, torch.full((4,), -w.time - w.rate * 5.0))


def test_corner_penalty_default_off_costs_nothing():
    """R7 (S17) defaults to 0: the term is logged but contributes nothing -- the reward of every
    pre-S17 config (incl. the in-flight inc6 dact arms) is unchanged."""
    w = RewardWeights()
    assert w.corner == 0.0
    kw = _base_kwargs()
    kw["action_norm"] = torch.tensor([[0.0, 0.5, 0.5, 1.0]] * 4)   # thr rail x yaw rail
    kw["last_action_norm"] = kw["action_norm"].clone()             # isolate from R5
    r, comp = compute_reward_terms(w, **kw)
    assert comp["corner_pen"] > 0.0                                # computed (logged)...
    assert torch.allclose(r, torch.full((4,), -w.time))            # ...but costs nothing


def test_corner_penalty_targets_the_mixer_rails():
    """R7 taxes exactly the two mixer corners (thr~rail x rate demand) and NOT mid-range
    thrust corrections or rate-free rail thrust."""
    w = RewardWeights(corner=4.0)
    kw = _base_kwargs()

    def pen(a):
        kw["action_norm"] = torch.tensor([a] * 4)
        kw["last_action_norm"] = kw["action_norm"].clone()
        r, comp = compute_reward_terms(w, **kw)
        return r, comp["corner_pen"]

    r_rail, p_rail = pen([0.0, 0.5, 0.5, 1.0])     # bottom rail x yaw rail (the inc5 killer)
    assert abs(p_rail - 0.5) < 1e-6                # |0-0.5| * ||[0,0,2*0.5]|| = 0.5
    _, p_top = pen([1.0, 1.0, 0.5, 0.5])           # top rail x roll rail (the gate-2 killer)
    assert abs(p_top - 0.5) < 1e-6
    r_mid, p_mid = pen([0.5, 0.5, 0.5, 1.0])       # SAME rate demand at mid thrust: free
    assert p_mid == 0.0
    assert (r_mid > r_rail).all()
    _, p_zr = pen([1.0, 0.5, 0.5, 0.5])            # rail thrust, zero rate: free
    assert p_zr == 0.0
    _, p_hov = pen([0.2656, 0.5, 0.5, 1.0])        # hover-collective turn: mild (~0.23)
    assert 0.2 < p_hov < 0.27


def test_weights_from_cfg_overrides():
    class Cfg:                                       # getattr-style cfg stub
        rw_collision = 40.0
        passage_bonus = 12.0                          # legacy S1.3 name still honored
        rw_corner = 8.0                               # R7 (S17) flows through the fields loop
    w = RewardWeights.from_cfg(Cfg())
    assert w.collision == 40.0 and w.passage == 12.0
    assert w.corner == 8.0
    assert w.progress == 10.0                         # untouched default
