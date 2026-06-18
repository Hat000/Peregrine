"""Tests for the inc8 GT-anchored sigma_p0 instrument + the 20-dim actor loader fix.

These pin (a) load_actor now infers the obs width from the checkpoint (17 inc7 / 20 inc8) so an
inc8 actor loads, with the inc7 path byte-unchanged, and (b) gate4_true_crossing_yz extracts a
ground-truth lateral/vertical crossing offset (NOT estim_err) from a course-completing rollout.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_INC7 = _ROOT / "rl" / "checkpoints" / "stage1_inc7_actor.pth"
_INC8 = _ROOT / "rl" / "checkpoints" / "stage1_inc8_actor.pth"


def test_load_actor_inc7_is_17dim():
    from fly_rl import load_actor
    a = load_actor(str(_INC7))
    assert a.head[0].linear.in_features == 17


@pytest.mark.skipif(not _INC8.exists(), reason="inc8 actor not present")
def test_load_actor_infers_20dim_inc8():
    """The loader fix: a 20-dim inc8 actor loads (previously raised a size-mismatch)."""
    from fly_rl import load_actor
    a = load_actor(str(_INC8))
    assert a.head[0].linear.in_features == 20


def test_gate4_crossing_is_gt_anchored_lateral():
    """gate4_true_crossing_yz returns the TRUE-trajectory gate-4 in-plane offset (y lateral, z
    vertical), NOT a KF/estimator error. inc7 on truth obs flies the course and crosses gate-4."""
    from contact_true_eval import (_build_plant_params, _build_start, run_episode,
                                   BODY_RADIUS_NOM, FRAME_DEPTH_NOM)
    from fly_rl import load_actor, _TRAIN_DT
    from inc8_sigmap0_eval import gate4_true_crossing_yz, GATE4_IDX

    actor = load_actor(str(_INC7))
    params = _build_plant_params("mixer")
    st, tgt, vflip = _build_start("simstart", 0)
    result, pos_traj = run_episode(actor, st, tgt, vflip, params,
                                   body_radius=BODY_RADIUS_NOM, frame_depth=FRAME_DEPTH_NOM,
                                   max_time=40.0, start_label="simstart", record_pos=True)
    # inc7 completes the course on truth obs -> a gate-4 pass crossing exists.
    assert any(c.gate == GATE4_IDX and c.verdict == "pass" for c in result.crossings)
    yz = gate4_true_crossing_yz(result, pos_traj, _TRAIN_DT)
    assert yz is not None
    y, z = yz
    # Bounded, finite, sub-metre in-plane offset (a real centering offset, not a blow-up).
    assert np.isfinite(y) and np.isfinite(z)
    assert abs(y) < 0.42 and abs(z) < 0.42   # inside the contact-true pass band


def test_gate4_crossing_none_when_not_reached():
    """No gate-4 pass -> None (the reach-rate denominator stays honest)."""
    from inc8_sigmap0_eval import gate4_true_crossing_yz
    from contact_true_eval import EpisodeResult
    r = EpisodeResult(outcome="MISS", crossings=[])
    assert gate4_true_crossing_yz(r, np.zeros((5, 3)), 0.01) is None
