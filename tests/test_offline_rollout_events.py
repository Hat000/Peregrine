"""Regression tests for the S15 review findings on the laptop deployment/eval path:
offline_rollout's S1.4 gate-event classification (must mirror peregrine_racing.crossing_events)
and fly_rl's checkpoint sidecar (the action-bounds-follow-the-checkpoint mechanism)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pymavlink")

_RL = Path(__file__).resolve().parents[1] / "rl"
sys.path.insert(0, str(_RL))

import fly_rl                                       # noqa: E402
import offline_rollout as orr                       # noqa: E402


def _ned(rel_zup, gate=0):
    """Build an NED position whose Z-up gate-frame coords (gate yaw=pi) equal rel_zup."""
    rel = np.asarray(rel_zup, dtype=np.float64)
    zup = np.linalg.inv(orr._R_W2G) @ rel + orr._GATE_POS_ZUP[gate]
    return zup * orr._FLIP


# --------------------------------------------------------------- gate_event vs the env semantics
def test_forward_center_pass():
    assert orr.gate_event(_ned([-0.3, 0, 0]), _ned([0.3, 0, 0]), 0) == "pass"


def test_forward_frame_band_collision():
    assert orr.gate_event(_ned([-0.3, 1.0, 0]), _ned([0.3, 1.0, 0]), 0) == "collision"


def test_forward_wide_miss():
    assert orr.gate_event(_ned([-0.3, 2.0, 0]), _ned([0.3, 2.0, 0]), 0) == "miss"


def test_backward_through_open_aperture_is_no_event():
    """Review finding (major): bwd through the OPEN aperture must be a non-event, exactly like
    peregrine_racing.crossing_events / the live sim -- NOT a collision."""
    assert orr.gate_event(_ned([0.3, 0.1, 0]), _ned([-0.3, 0.1, 0]), 0) is None


def test_backward_through_frame_band_collides():
    assert orr.gate_event(_ned([0.3, 1.0, 0]), _ned([-0.3, 1.0, 0]), 0) == "collision"


def test_backward_wide_is_no_event():
    assert orr.gate_event(_ned([0.3, 5.0, 0]), _ned([-0.3, 5.0, 0]), 0) is None


def test_interpolation_rescues_fast_diagonal():
    # endpoint off-aperture but the crossing point dead-centre (the C2 fix, numpy mirror)
    assert orr.gate_event(_ned([-0.25, 0.0, 0]), _ned([0.75, 1.0, 0]), 0) == "pass"


def test_frame_strike_other_gates_skips_target():
    prev, cur = _ned([-0.3, 1.0, 0], gate=2), _ned([0.3, 1.0, 0], gate=2)
    assert orr.frame_strike_other_gates(prev, cur, target=2) is None
    assert orr.frame_strike_other_gates(prev, cur, target=0) == 2


def test_half_outer_matches_env():
    assert orr._HALF_OUTER == pytest.approx(2.72 / 2.0)
    assert orr._HALF_OPEN == pytest.approx(1.5 / 2.0)


# --------------------------------------------------------------- the checkpoint sidecar
def test_sidecar_applies_and_absence_keeps_legacy(tmp_path, capsys):
    saved = fly_rl._ACT_MAX.copy()
    try:
        # absence: bounds untouched, loud warning printed
        fly_rl._apply_checkpoint_sidecar(str(tmp_path / "nosuch_actor.pth"))
        assert np.array_equal(fly_rl._ACT_MAX, saved)
        assert "WARNING" in capsys.readouterr().out
        # presence: thrust bound follows the checkpoint
        (tmp_path / "a_actor.json").write_text(json.dumps({"act_max_thrust": 3.765}))
        fly_rl._apply_checkpoint_sidecar(str(tmp_path / "a_actor.pth"))
        assert fly_rl._ACT_MAX[0] == pytest.approx(3.765)
        assert fly_rl._ACT_MAX[1] == pytest.approx(saved[1])      # rates untouched
    finally:
        fly_rl._ACT_MAX[:] = saved


def test_inc3_sidecar_committed():
    """stage1_inc3 was TRAINED at max_normed_thrust=3.765 (peregrine_racing_s13.sbatch) -- its
    sidecar must exist next to the checkpoint so any deploy of it rescales correctly."""
    sc = _RL / "checkpoints" / "stage1_inc3_actor.json"
    assert sc.exists()
    meta = json.loads(sc.read_text())
    assert meta["act_max_thrust"] == pytest.approx(3.765)
