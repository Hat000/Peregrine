"""Tests for the S1.4 procedural course sampler (rl/peregrine_course.py Part 2).

Also re-derives the VQ1 geometry stats from peregrine_course_diffaero.json and pins that the VQ1
course is INTERIOR to every sampling range -- the held-out-eval premise. If the json or the ranges
drift apart, these tests break instead of the training.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_RL = Path(__file__).resolve().parents[1] / "rl"
sys.path.insert(0, str(_RL))

from peregrine_course import (DEFAULT_COURSE_RANGES, VQ1_SPAWN_POS_ZUP,  # noqa: E402
                              VQ1_SPAWN_PITCH_RAD, load_course_zup, sample_courses)

_JSON = _RL / "peregrine_course_diffaero.json"


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------- VQ1 is interior to the ranges
def test_vq1_geometry_interior_to_sampler_ranges():
    gp, gy = load_course_zup(_JSON)
    R = DEFAULT_COURSE_RANGES
    spawn = np.array(VQ1_SPAWN_POS_ZUP)

    seg = np.diff(np.vstack([spawn, gp]), axis=0)                  # 6 segments incl. pad->g0
    horiz = np.linalg.norm(seg[:, :2], axis=1)
    drop = -seg[:, 2]                                              # +down
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    turns = _wrap(np.diff(headings))

    # pad->gate0 within the spawn ranges
    assert R["spawn_dist_m"][0] < horiz[0] < R["spawn_dist_m"][1]
    assert R["spawn_below_g0_m"][0] < -drop[0] < R["spawn_below_g0_m"][1]
    # gate->gate segments within seg_len ranges
    assert (horiz[1:] > R["seg_len_m"][0]).all() and (horiz[1:] < R["seg_len_m"][1]).all()
    # descent + grade within ranges
    assert (drop[1:] > R["drop_m"][0]).all() and (drop[1:] < R["drop_m"][1]).all()
    assert (np.abs(seg[1:, 2]) / horiz[1:] < R["max_grade"]).all()
    # heading changes well inside the turn budget
    assert (np.abs(turns) < R["turn_rad"]).all()
    # gate yaws within jitter of the path bisector convention
    bis = np.empty(len(gp))
    bis[:-1] = np.arctan2(np.sin(headings[:-1]) + np.sin(headings[1:]),
                          np.cos(headings[:-1]) + np.cos(headings[1:]))
    bis[-1] = headings[-1]
    # VQ1 map yaws are all pi; allow the documented ~12 deg deviation
    assert (np.abs(_wrap(gy - bis)) < 0.22).all()
    # spawn pitch constant matches the measured spawn quaternion (2*asin(0.15471))
    assert abs(VQ1_SPAWN_PITCH_RAD - (-2 * math.asin(0.15471))) < 2e-3


# ---------------------------------------------------------------- sampler output contract
@pytest.fixture(scope="module")
def courses():
    g = torch.Generator().manual_seed(7)
    return sample_courses(512, device="cpu", generator=g)


def test_shapes_and_finiteness(courses):
    n, G = 512, DEFAULT_COURSE_RANGES["n_gates"]
    assert courses["gate_pos"].shape == (n, G, 3)
    assert courses["gate_yaw"].shape == (n, G)
    assert courses["spawn_pos"].shape == (n, 3)
    assert courses["spawn_yaw"].shape == (n,)
    for v in courses.values():
        assert torch.isfinite(v).all()


def test_segment_lengths_in_range(courses):
    R = DEFAULT_COURSE_RANGES
    pts = torch.cat([courses["spawn_pos"].unsqueeze(1), courses["gate_pos"]], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    horiz = torch.linalg.norm(seg[..., :2], dim=-1)
    assert (horiz[:, 0] >= R["spawn_dist_m"][0] - 1e-4).all()
    assert (horiz[:, 0] <= R["spawn_dist_m"][1] + 1e-4).all()
    assert (horiz[:, 1:] >= R["seg_len_m"][0] - 1e-4).all()
    assert (horiz[:, 1:] <= R["seg_len_m"][1] + 1e-4).all()


def test_descent_and_grade(courses):
    R = DEFAULT_COURSE_RANGES
    pts = torch.cat([courses["spawn_pos"].unsqueeze(1), courses["gate_pos"]], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    horiz = torch.linalg.norm(seg[..., :2], dim=-1)
    # gate 0 sits above the pad
    assert (seg[:, 0, 2] >= R["spawn_below_g0_m"][0] - 1e-4).all()
    assert (seg[:, 0, 2] <= R["spawn_below_g0_m"][1] + 1e-4).all()
    # per-segment descent bounded by both the drop range and the grade clamp
    drop = -seg[:, 1:, 2]
    assert (drop <= R["drop_m"][1] + 1e-4).all()
    assert (drop >= R["drop_m"][0] - 1e-4).all()
    assert (seg[:, 1:, 2].abs() / horiz[:, 1:] <= R["max_grade"] + 1e-4).all()


def test_turn_budget(courses):
    R = DEFAULT_COURSE_RANGES
    pts = torch.cat([courses["spawn_pos"].unsqueeze(1), courses["gate_pos"]], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    headings = torch.atan2(seg[..., 1], seg[..., 0])
    d = headings[:, 1:] - headings[:, :-1]
    turns = torch.atan2(torch.sin(d), torch.cos(d))
    assert (turns.abs() <= R["turn_rad"] + 1e-4).all()


def test_gate_yaw_near_bisector_and_exit_sanity(courses):
    R = DEFAULT_COURSE_RANGES
    gp, gy = courses["gate_pos"], courses["gate_yaw"]
    pts = torch.cat([courses["spawn_pos"].unsqueeze(1), gp], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    headings = torch.atan2(seg[..., 1], seg[..., 0])
    bis = torch.empty_like(gy)
    bis[:, :-1] = torch.atan2(torch.sin(headings[:, :-1]) + torch.sin(headings[:, 1:]),
                              torch.cos(headings[:, :-1]) + torch.cos(headings[:, 1:]))
    bis[:, -1] = headings[:, -1]
    dev = torch.atan2(torch.sin(gy - bis), torch.cos(gy - bis))
    assert (dev.abs() <= R["yaw_jitter_rad"] + 1e-4).all()
    # every next gate lies in FRONT of the current gate's exit plane (and the pad lies behind
    # gate 0's plane) -- guarantees the spawn corridor and progress geometry make sense
    exit_dir = torch.stack([torch.cos(gy), torch.sin(gy)], dim=-1)
    nxt = gp[:, 1:, :2] - gp[:, :-1, :2]
    assert ((nxt * exit_dir[:, :-1]).sum(-1) > 0).all()
    behind = ((courses["spawn_pos"][:, :2] - gp[:, 0, :2]) * exit_dir[:, 0]).sum(-1)
    assert (behind < 0).all()


def test_min_pair_separation(courses):
    R = DEFAULT_COURSE_RANGES
    pts = torch.cat([courses["spawn_pos"].unsqueeze(1), courses["gate_pos"]], dim=1)
    d = torch.linalg.norm(pts[:, :, None, :2] - pts[:, None, :, :2], dim=-1)
    G1 = pts.shape[1]
    d = d + torch.eye(G1) * 1e9
    assert (d.amin(dim=(1, 2)) >= R["min_pair_dist_m"] - 1e-4).all()


def test_spawn_yaw_tail_first(courses):
    gy0 = courses["gate_yaw"][:, 0]
    expect = torch.atan2(torch.sin(gy0 + math.pi), torch.cos(gy0 + math.pi))
    d = courses["spawn_yaw"] - expect
    assert torch.atan2(torch.sin(d), torch.cos(d)).abs().max() < 1e-5


def test_determinism_and_variability():
    a = sample_courses(16, generator=torch.Generator().manual_seed(3))
    b = sample_courses(16, generator=torch.Generator().manual_seed(3))
    c = sample_courses(16, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a["gate_pos"], b["gate_pos"])
    assert not torch.equal(a["gate_pos"], c["gate_pos"])
    # courses genuinely differ from one another within a draw
    assert (a["gate_pos"][0] - a["gate_pos"][1]).abs().max() > 1.0


# ---------------------------------------------------------------- gate-1 out-of-fov forcing (pefcap)
def test_g1_out_of_fov_default_off_is_byte_identical():
    """The new g1_out_of_fov_bearing_rad range key defaults None -> OFF. A draw WITHOUT it must be
    byte-identical to the same-seed draw that passes it as None (the OFF branch adds no RNG)."""
    assert DEFAULT_COURSE_RANGES["g1_out_of_fov_bearing_rad"] is None
    a = sample_courses(32, generator=torch.Generator().manual_seed(11), n_gates=2)
    b = sample_courses(32, generator=torch.Generator().manual_seed(11), n_gates=2,
                       g1_out_of_fov_bearing_rad=None)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_g1_out_of_fov_on_forces_a_turn_from_sampled_distances():
    """With the knob ON the segment-0->1 turn is REPLACED by the geometry-adaptive turn computed from
    the sampled spawn distance + spacing: gate_pos changes, and the realized |turn[0]| lands in a sane
    band (target bearing + a small parallax term), never a hairpin."""
    on = sample_courses(2000, generator=torch.Generator().manual_seed(0), n_gates=2,
                        seg_len_m=(10.0, 20.0), spawn_dist_m=(8.0, 15.0),
                        spawn_heading=0.0, g1_out_of_fov_bearing_rad=(0.82, 0.95))
    off = sample_courses(2000, generator=torch.Generator().manual_seed(0), n_gates=2,
                         seg_len_m=(10.0, 20.0), spawn_dist_m=(8.0, 15.0), spawn_heading=0.0)
    assert not torch.equal(on["gate_pos"], off["gate_pos"])
    # realized turn at gate 0 = heading of (g1-g0) minus heading of (g0-spawn); |turn| past the FOV edge.
    g0, g1, sp = on["gate_pos"][:, 0], on["gate_pos"][:, 1], on["spawn_pos"]
    h_in = torch.atan2((g0 - sp)[:, 1], (g0 - sp)[:, 0])
    h_out = torch.atan2((g1 - g0)[:, 1], (g1 - g0)[:, 0])
    turn = torch.atan2(torch.sin(h_out - h_in), torch.cos(h_out - h_in)).abs()
    assert float(turn.min()) > 0.785, float(turn.min())            # every turn past the ~45deg half-HFOV edge
    assert float(turn.max()) < 1.6, float(turn.max())              # never a hairpin (well under 90deg+parallax)
    # BOTH signs present (left/right), not a one-sided bias.
    signed = torch.atan2(torch.sin(h_out - h_in), torch.cos(h_out - h_in))
    assert float((signed > 0).float().mean()) > 0.3 and float((signed < 0).float().mean()) > 0.3


def test_g1_out_of_fov_inert_on_single_gate():
    """Needs a segment 1 (n_gates>=2). On a 1-gate course the knob is inert (no turn[0] to force) and
    the draw is byte-identical to the OFF draw."""
    a = sample_courses(16, generator=torch.Generator().manual_seed(2), n_gates=1)
    b = sample_courses(16, generator=torch.Generator().manual_seed(2), n_gates=1,
                       g1_out_of_fov_bearing_rad=(0.82, 0.95))
    assert torch.equal(a["gate_pos"], b["gate_pos"])
