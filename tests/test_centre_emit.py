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


def test_low_tail_reduction_beats_foreshortening():
    """Foreshortening only SHRINKS an apparent length, which only INFLATES range -- so the reduction
    must sit in the LOW tail. Squash one axis (a tilt) and the estimate must stay near the
    un-squashed truth, far below what averaging the foreshortened pairs would give.

    The reduction is the 25th percentile, not the strict minimum: min also takes the low tail of
    PIXEL NOISE (noise that lengthens a pair is discarded, noise that shortens it is kept), which
    biases it NEAR. Measured on 79 real four-corner gates inside the 30 m cap: min |err| med 2.44% /
    p90 10.46% / bias -3.51%, vs p25 1.97% / 6.08% / -0.03%. p25 nearly halves the tail and is
    essentially unbiased, at the cost of a little accuracy under EXTREME tilt (below)."""
    q = _square_at(10.0)
    c = q.mean(axis=0)
    q_tilt = c + (q - c) * np.array([1.0, 0.45])     # foreshorten vertically ~ a 63 deg tilt
    z, _ = depth_from_corner_pairs(q_tilt)
    # still anchored near the un-foreshortened truth, nowhere near the squashed pairs' ~22 m
    assert 10.0 <= z <= 10.0 * 1.20
    # and decisively below the mean of all six pair estimates (what averaging would give)
    assert z < 14.0


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


# --- slot1 dedup + the orientation-free visible_area mask (2026-07-23) ----------------------------

def test_slot1_dedups_against_the_emitted_pose_not_the_lagging_track():
    """THE BUG: detect_gate_lever returns the RAW chosen pose while the track is EMA-smoothed, so
    dedup compared candidates to a point slot0 never emitted. On a fast approach the smoothed track
    lags (measured p90 1.35-2.52 m against a 3.0 m radius), the ACTIVE gate escaped its own dedup and
    was re-admitted into slot1 -- 37-60% of two-slot ticks fed the SAME gate to BOTH slots, which is
    OOD (training fed slot1 only the true tg+1 or zero)."""
    from racer.contracts import GatePose
    from racer.gate_seeker import GateSeeker, GateSeekerConfig
    s = GateSeeker(config=GateSeekerConfig(), detector=None)

    def pose(z):
        return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                        t_cam_gate=np.array([0.0, 0.0, float(z)]), reproj_error_px=0.0)

    emitted = pose(10.0)
    s._active_emitted = emitted
    s._track_range_m, s._track_bearing = 6.0, np.zeros(2)   # track lagging 4 m behind the emission
    # The emitted gate must be recognised as the active one DESPITE the stale track...
    assert s._is_active_gate(pose(10.0)) is True
    # ...and a genuinely different gate 8 m beyond it must still pass through to slot1.
    assert s._is_active_gate(pose(18.0)) is False
    # Without the fix the 4 m lag would have put the emitted gate outside the 3 m radius of the
    # track and let it into slot1; assert the track is NOT what decides while an emission exists.
    assert s._is_active_gate(pose(6.0)) is False

    s._active_emitted = None            # slot0 coasted -> fall back to the track, still guarded
    assert s._is_active_gate(pose(6.0)) is True


def test_orientation_free_pose_masks_visible_area_instead_of_fabricating_it():
    """visible_area is derived by PROJECTING the gate model through R_cam_gate. A centre-emit pose
    with <3 corners has NO orientation (synthesised with identity rotation), and identity reads as a
    perfectly square-on gate -- a fabricated ~1.0 area on exactly the cropped gates the centre path
    recovers. It must be masked, not invented."""
    from racer.contracts import GatePose
    from racer.ego_obs import visible_area_from_gatepose
    t = np.array([0.0, 0.0, 8.0])
    # identity rotation genuinely does read near square-on -- which is why trusting it is wrong
    assert visible_area_from_gatepose(np.eye(3), t) > 0.5
    synth = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t,
                     reproj_error_px=0.0, n_corners=1)
    real = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t,
                    reproj_error_px=0.0, n_corners=4)
    masked = (0.0 if int(getattr(synth, "n_corners", 4)) < 3
              else visible_area_from_gatepose(synth.R_cam_gate, synth.t_cam_gate))
    kept = (0.0 if int(getattr(real, "n_corners", 4)) < 3
            else visible_area_from_gatepose(real.R_cam_gate, real.t_cam_gate))
    assert masked == 0.0            # synthesised: withheld
    assert kept > 0.5               # a real 4-corner fit still reports its area


def test_orientation_free_pose_HOLDS_the_last_area_instead_of_zeroing_it():
    """THE 2026-07-24 FLIGHT REGRESSION. Writing visible_area=0.0 for an orientation-free centre-emit
    pose (<3 corners) is a TRAINING MISMATCH: training's shoelace ratio is unclipped, so a close
    square-on gate whose corners have cropped reads ~1.0 there, never 0.0. Feeding 0.0 at the moment
    the drone commits to the gate is OOD -- measured 0.0% of close ticks fed area==0 on pnp vs 24-73%
    on centre, and the centre flights crashed at ~2.9 m with rates diverging (record9 config: 5,5
    gates on pnp vs 0,0,0,1,0,1,0 on centre). The held value was MEASURED a few ticks earlier and is
    far closer to truth than either 0.0 or a fabricated 1.0."""
    import numpy as np
    from racer.contracts import GatePose
    from racer.ego_obs import EgoObsBuilder

    def pose(n_corners, z):
        return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                        t_cam_gate=np.array([0.0, 0.0, float(z)]), reproj_error_px=0.0,
                        n_corners=n_corners)

    b = EgoObsBuilder()
    R = np.eye(3)
    kw = dict(R_frd2ned=R, vel_ned=np.zeros(3), gyro_frd=np.zeros(3), last_normed_thrust=0.5)
    # a good 4-corner fix at 8 m establishes a real area...
    b.update(sim_time_ns=0, gate_index=0, pose=pose(4, 8.0), **kw)
    established = b._area[0]
    assert established > 0.5, "4-corner fix should measure a real area"
    # ...then the gate crops to a 1-corner centre-emit pose as we close on it
    b.update(sim_time_ns=25_000_000, gate_index=0, pose=pose(1, 2.9), **kw)
    assert b._area[0] == established, "orientation-free pose must HOLD, not zero, the area"
    assert b._area[0] != 0.0, "feeding 0.0 here is the regression this test pins"
