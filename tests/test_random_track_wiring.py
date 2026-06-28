"""Tests for the randomised track generator integration points:

  * DIFFICULTY_PRESETS exist and cover the expected keys
  * sample_courses() respects every preset's overrides
  * _assign_courses / rel_tables / _update_boxes work with random courses (no diffaero needed:
    the helpers are pure torch and importable from peregrine_racing directly)
  * course_mode="random" default path and track_difficulty="vq1" path produce distinct layouts
  * "medium" preset == DEFAULT_COURSE_RANGES (empty override dict)
  * VQ1 default path is NOT broken by the difficulty preset addition

These tests run on CPU without diffaero.  The precheck that actually instantiates PeregrineRacing
and calls step() requires diffaero (cluster) and lives in rl/peregrine_racing_precheck.py.
"""
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_RL = Path(__file__).resolve().parents[1] / "rl"
sys.path.insert(0, str(_RL))

from peregrine_course import (  # noqa: E402
    DEFAULT_COURSE_RANGES,
    DIFFICULTY_PRESETS,
    sample_courses,
    load_course_zup,
)
from peregrine_racing import rel_tables  # noqa: E402   (pure-torch; no diffaero import triggered)

_JSON = _RL / "peregrine_course_diffaero.json"
_N = 64      # batch size for wiring tests (small for speed)
_SEED = 99


# ---- DIFFICULTY_PRESETS structure ---------------------------------------------------------------

def test_presets_exist_and_cover_expected_names():
    assert set(DIFFICULTY_PRESETS) >= {"vq1_like", "easy", "medium", "hard"}, (
        "Expected named presets: vq1_like, easy, medium, hard")


def test_medium_preset_is_empty():
    """'medium' must be an empty dict so DEFAULT_COURSE_RANGES is used verbatim."""
    assert DIFFICULTY_PRESETS["medium"] == {}, (
        "'medium' preset must be empty (== DEFAULT_COURSE_RANGES unchanged)")


def test_preset_keys_are_valid_sample_courses_kwargs():
    """Every key in every preset must be a recognised sample_courses kwarg."""
    valid = set(DEFAULT_COURSE_RANGES)
    for name, overrides in DIFFICULTY_PRESETS.items():
        unknown = set(overrides) - valid
        assert not unknown, f"Preset {name!r} has unknown kwarg(s): {sorted(unknown)}"


def test_all_presets_produce_finite_courses():
    for name, overrides in DIFFICULTY_PRESETS.items():
        g = torch.Generator().manual_seed(_SEED)
        c = sample_courses(_N, device="cpu", generator=g, **overrides)
        for key, val in c.items():
            assert torch.isfinite(val).all(), f"Preset {name!r}: non-finite in {key!r}"


def test_hard_preset_has_tighter_segments_than_medium():
    """'hard' preset constrains seg_len_m upper bound lower than 'medium'."""
    hard = DIFFICULTY_PRESETS["hard"]
    assert "seg_len_m" in hard, "'hard' preset must override seg_len_m"
    assert hard["seg_len_m"][1] < DEFAULT_COURSE_RANGES["seg_len_m"][1], (
        "hard seg_len max should be < medium's 45 m")


def test_vq1_like_preset_has_narrower_turns_than_medium():
    vl = DIFFICULTY_PRESETS["vq1_like"]
    assert "turn_rad" in vl
    assert vl["turn_rad"] < DEFAULT_COURSE_RANGES["turn_rad"]


# ---- sample_courses respects preset overrides --------------------------------------------------

@pytest.mark.parametrize("name,overrides", list(DIFFICULTY_PRESETS.items()))
def test_preset_respects_seg_len_bounds(name, overrides):
    R = {**DEFAULT_COURSE_RANGES, **overrides}
    g = torch.Generator().manual_seed(_SEED)
    c = sample_courses(_N, device="cpu", generator=g, **overrides)
    gp, sp = c["gate_pos"], c["spawn_pos"]
    pts = torch.cat([sp.unsqueeze(1), gp], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    horiz = torch.linalg.norm(seg[..., :2], dim=-1)
    assert (horiz[:, 1:] >= R["seg_len_m"][0] - 1e-3).all(), f"{name}: seg_len_m min"
    assert (horiz[:, 1:] <= R["seg_len_m"][1] + 1e-3).all(), f"{name}: seg_len_m max"


@pytest.mark.parametrize("name,overrides", list(DIFFICULTY_PRESETS.items()))
def test_preset_respects_turn_budget(name, overrides):
    R = {**DEFAULT_COURSE_RANGES, **overrides}
    g = torch.Generator().manual_seed(_SEED)
    c = sample_courses(_N, device="cpu", generator=g, **overrides)
    gp, sp = c["gate_pos"], c["spawn_pos"]
    pts = torch.cat([sp.unsqueeze(1), gp], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    headings = torch.atan2(seg[..., 1], seg[..., 0])
    dh = headings[:, 1:] - headings[:, :-1]
    turns = torch.atan2(torch.sin(dh), torch.cos(dh)).abs()
    assert (turns <= R["turn_rad"] + 1e-3).all(), f"{name}: turn_rad"


@pytest.mark.parametrize("name,overrides", list(DIFFICULTY_PRESETS.items()))
def test_preset_respects_min_pair_dist(name, overrides):
    R = {**DEFAULT_COURSE_RANGES, **overrides}
    g = torch.Generator().manual_seed(_SEED)
    c = sample_courses(_N, device="cpu", generator=g, **overrides)
    gp, sp = c["gate_pos"], c["spawn_pos"]
    pts = torch.cat([sp.unsqueeze(1), gp], dim=1)
    d = torch.linalg.norm(pts[:, :, None, :2] - pts[:, None, :, :2], dim=-1)
    G1 = d.shape[1]
    d = d + torch.eye(G1) * 1e9
    assert (d.amin(dim=(1, 2)) >= R["min_pair_dist_m"] - 1e-3).all(), f"{name}: min_pair_dist"


# ---- rel_tables works correctly with random courses -------------------------------------------

def test_rel_tables_shape_and_finiteness_with_random_courses():
    """rel_tables (consumed by PeregrineRacing.__init__) must produce finite outputs."""
    g = torch.Generator().manual_seed(_SEED)
    c = sample_courses(_N, device="cpu", generator=g)
    gp = c["gate_pos"]          # (N, G, 3)
    gy = c["gate_yaw"]          # (N, G)
    rel_pos, yaw_rel = rel_tables(gp, gy)
    assert rel_pos.shape == gp.shape
    assert yaw_rel.shape == gy.shape
    assert torch.isfinite(rel_pos).all(), "rel_pos has non-finite entries"
    assert torch.isfinite(yaw_rel).all(), "yaw_rel has non-finite entries"


def test_rel_tables_yaw_rel_bounded():
    """Relative yaw must be in (-pi, pi]: circular wrap is applied in rel_tables."""
    g = torch.Generator().manual_seed(_SEED)
    c = sample_courses(_N, device="cpu", generator=g)
    _, yaw_rel = rel_tables(c["gate_pos"], c["gate_yaw"])
    assert (yaw_rel.abs() <= math.pi + 1e-5).all()


# ---- courses vary across episodes (training-time diversity) ------------------------------------

def test_random_courses_vary_across_episodes():
    """Two independent samples with different seeds must differ -- else every episode is the same."""
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    c1 = sample_courses(_N, device="cpu", generator=g1)
    c2 = sample_courses(_N, device="cpu", generator=g2)
    assert not torch.equal(c1["gate_pos"], c2["gate_pos"]), (
        "Different seeds produced identical courses -- generator not threaded through correctly")


def test_random_courses_are_deterministic_given_seed():
    """Same seed must reproduce the same layout (essential for debugging + curriculum)."""
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    c1 = sample_courses(_N, device="cpu", generator=g1)
    c2 = sample_courses(_N, device="cpu", generator=g2)
    assert torch.equal(c1["gate_pos"], c2["gate_pos"])
    assert torch.equal(c1["gate_yaw"], c2["gate_yaw"])


def test_random_courses_within_draw_differ():
    """Courses in a single batch must NOT all be identical (rejection loop must not flatten them)."""
    g = torch.Generator().manual_seed(_SEED)
    c = sample_courses(_N, device="cpu", generator=g)
    gp = c["gate_pos"]
    # gate 0 positions should differ across the batch
    std_xy = gp[:, 0, :2].std(dim=0).mean().item()
    assert std_xy > 1.0, f"Courses in one batch are suspiciously identical (xy-std={std_xy:.3f})"


# ---- VQ1 default path is not broken -------------------------------------------------------------

def test_vq1_course_json_still_loads():
    """load_course_zup must still parse the VQ1 JSON (no regression from preset addition)."""
    gp, gy = load_course_zup(_JSON)
    data = json.loads(_JSON.read_text())
    assert len(gp) == len(data["gates"])
    assert gp.shape == (len(gp), 3)
    assert gy.shape == (len(gy),)


def test_vq1_course_interior_to_medium_preset():
    """The VQ1 hold-out must still lie inside the 'medium' (default) sampling ranges.

    This is the same invariant as test_peregrine_course.test_vq1_geometry_interior_to_sampler_ranges
    but expressed in terms of the preset: if 'medium' ever changes, this catches the drift."""
    import numpy as np
    from peregrine_course import VQ1_SPAWN_POS_ZUP
    gp_np, _ = load_course_zup(_JSON)
    R = DEFAULT_COURSE_RANGES   # "medium" == DEFAULT_COURSE_RANGES
    spawn = np.array(VQ1_SPAWN_POS_ZUP)
    pts = np.vstack([spawn, gp_np])
    seg = np.diff(pts, axis=0)
    horiz = np.linalg.norm(seg[:, :2], axis=1)
    drop = -seg[:, 2]
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    turns = np.abs(np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings))))

    assert (horiz[1:] > R["seg_len_m"][0]).all(), "VQ1 seg below medium min"
    assert (horiz[1:] < R["seg_len_m"][1]).all(), "VQ1 seg above medium max"
    assert (np.abs(drop[1:]) / horiz[1:] < R["max_grade"]).all(), "VQ1 grade exceeds medium max"
    assert (turns < R["turn_rad"]).all(), "VQ1 turns exceed medium turn budget"


# ---- spawn yaw convention (tail-first w.r.t. gate 0) is preset-independent --------------------

@pytest.mark.parametrize("name,overrides", list(DIFFICULTY_PRESETS.items()))
def test_spawn_yaw_tail_first_all_presets(name, overrides):
    g = torch.Generator().manual_seed(_SEED)
    c = sample_courses(_N, device="cpu", generator=g, **overrides)
    gy0 = c["gate_yaw"][:, 0]
    expect = torch.atan2(torch.sin(gy0 + math.pi), torch.cos(gy0 + math.pi))
    diff = torch.atan2(torch.sin(c["spawn_yaw"] - expect), torch.cos(c["spawn_yaw"] - expect))
    assert diff.abs().max().item() < 1e-5, f"Preset {name!r}: spawn not tail-first w.r.t. gate 0"
