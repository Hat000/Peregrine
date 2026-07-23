"""M+1 centre+size emit: rel_pos = range x bearing, no PnP translation.

The numbers asserted here are the ones that justify the path existing at all -- measured on 300 real
failure-mined frames: 100% of observations emit (was 92%), and inside the 30 m valid cap the depth
agrees with PnP to 0.425 m median / 2.4% relative.
"""
import numpy as np
import pytest

from racer.contracts import Frame, GateObservation
from racer.frames import CAMERA_INTRINSICS_K as K
from racer.vision.centre_emit import (
    GATE_INNER_SIZE_M, bearing_from_px, depth_from_corner_pairs, emit_from_observation,
)

F = 320.0


def _square_at(depth_m, cx=320.0, cy=180.0):
    """The 4 inner corners of a square-on gate at ``depth_m``, canonical LL,LR,UR,UL."""
    half_px = F * (GATE_INNER_SIZE_M / 2.0) / depth_m
    return np.array([[cx - half_px, cy + half_px],    # 0 LL
                     [cx + half_px, cy + half_px],    # 1 LR
                     [cx + half_px, cy - half_px],    # 2 UR
                     [cx - half_px, cy - half_px]],   # 3 UL
                    dtype=np.float64)


def _obs(corners, ids=None, centre=(320.0, 180.0), bbox=None, conf=None):
    return GateObservation(
        frame_id=0, sim_time_ns=0,
        corners_px=np.asarray(corners, dtype=np.float64),
        corner_ids=None if ids is None else np.asarray(ids, dtype=int),
        corner_confidence=conf,
        bbox_xywh=None if bbox is None else np.asarray(bbox, dtype=np.float64),
        centre_px=None if centre is None else np.asarray(centre, dtype=np.float64),
        centre_confidence=0.9 if centre is not None else None,
    )


@pytest.mark.parametrize("depth", [3.0, 8.0, 15.0, 27.0])
def test_depth_recovers_a_square_on_gate(depth):
    z, n = depth_from_corner_pairs(_square_at(depth))
    assert n == 6                                   # 4 sides + 2 diagonals all vote
    assert z == pytest.approx(depth, rel=1e-6)


def test_min_over_pairs_beats_foreshortening():
    """Foreshortening only SHRINKS an apparent length, which only INFLATES range -- so the minimum
    over pairs is the least-foreshortened estimate. Squash one axis (a tilt) and the un-squashed
    dimension must still carry the true range."""
    q = _square_at(10.0)
    c = q.mean(axis=0)
    q_tilt = c + (q - c) * np.array([1.0, 0.45])     # foreshorten vertically ~ a 63 deg tilt
    z, _ = depth_from_corner_pairs(q_tilt)
    assert z == pytest.approx(10.0, rel=0.02)        # horizontal pair is untouched -> min is right
    # the naive average would read FAR because the squashed pairs inflate range
    assert z < 10.0 * 1.05


def test_two_corners_are_enough_when_identified():
    """The whole reason the rule is min-over-PAIRS: it degrades to any 2 identified corners, which is
    the cropped-gate regime M+1 exists for."""
    q = _square_at(12.0)
    z, n = depth_from_corner_pairs(q[[0, 1]], corner_ids=[0, 1])      # one known side
    assert n == 1 and z == pytest.approx(12.0, rel=1e-6)
    z, n = depth_from_corner_pairs(q[[0, 2]], corner_ids=[0, 2])      # one known diagonal
    assert n == 1 and z == pytest.approx(12.0, rel=1e-6)


def test_unidentified_corner_pair_carries_no_scale():
    """Order alone is ambiguous: without corner_ids a 2-row array has no known metric separation."""
    q = _square_at(12.0)
    assert depth_from_corner_pairs(q[[0, 2]], corner_ids=None)[1] <= 1


def test_bearing_is_the_ambiguity_free_half():
    ray, (az, el) = bearing_from_px((K[0, 2], K[1, 2]))
    assert ray == pytest.approx([0.0, 0.0, 1.0])      # centre pixel -> straight down the axis
    assert (az, el) == pytest.approx((0.0, 0.0))
    _, (az_r, _) = bearing_from_px((K[0, 2] + F, K[1, 2]))
    assert az_r == pytest.approx(np.pi / 4)           # one focal length right = 45 deg


def test_emit_is_range_times_bearing():
    z = 9.0
    e = emit_from_observation(_obs(_square_at(z)))
    assert e.range_src == "corners" and e.n_pairs == 6
    assert e.depth_m == pytest.approx(z, rel=1e-6)
    assert e.p_cam == pytest.approx([0.0, 0.0, z], abs=1e-6)
    assert e.range_m == pytest.approx(z, rel=1e-6)


def test_off_axis_centre_moves_rel_pos_not_depth():
    """Depth comes from SIZE, bearing from the CENTRE pixel -- they are independent by construction."""
    z = 9.0
    e = emit_from_observation(_obs(_square_at(z), centre=(K[0, 2] + F, K[1, 2])))
    assert e.depth_m == pytest.approx(z, rel=1e-6)    # unchanged by the bearing
    assert e.p_cam[0] == pytest.approx(z, rel=1e-6)   # 45 deg right => x == z
    assert e.range_m > e.depth_m                      # ranges along a slanted ray


def test_corner_free_observation_still_emits_via_bbox():
    """'As long as the centre is emitted, we take it' -- a 0-corner detection must still produce a
    3-D emit, flagged as the coarse source so a consumer can prefer a coasted track range."""
    e = emit_from_observation(_obs(np.zeros((0, 2)), ids=None, bbox=(320.0, 180.0, 100.0, 100.0)))
    assert e is not None and e.range_src == "bbox" and e.n_pairs == 0
    assert e.depth_m > 0.0


def test_no_centre_means_no_emit():
    assert emit_from_observation(_obs(_square_at(9.0), centre=None)) is None


def test_contract_admits_corner_free_only_with_a_centre():
    """The lifted floor is CONDITIONAL: a centre buys the exemption, its absence does not."""
    GateObservation(frame_id=0, sim_time_ns=0, corners_px=np.zeros((0, 2)),
                    corner_ids=np.zeros((0,), dtype=int), centre_px=np.array([320.0, 180.0]))
    with pytest.raises(AssertionError):
        GateObservation(frame_id=0, sim_time_ns=0, corners_px=np.zeros((2, 2)),
                        corner_ids=np.array([0, 1]))          # 2 corners, no centre -> still invalid


def test_pnp_declines_a_corner_free_observation():
    """estimate_gate_pose must not be handed a degenerate set; the emit path owns these."""
    from racer.vision.gate_pose import estimate_gate_pose
    obs = _obs(np.zeros((0, 2)), ids=np.zeros((0,), dtype=int))
    assert estimate_gate_pose(obs) is None
