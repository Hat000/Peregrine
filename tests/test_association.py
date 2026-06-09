"""Robust association + post-PnP depth sanity (racer.vision.association) -- synthetic geometry.

These encode the two measured catastrophic-tail mechanisms from the course characterization
(handoff/perception-char-2026-06-08, 46% raw tail): a near detection matching a far collinear
background gate (87/312), and a solved depth wildly off the predicted range to the associated
gate (56/312). All scenes are built with the synthetic projector -- no detector model needed.
"""
import numpy as np

from racer.contracts import DroneState, Frame, Gate, GateObservation
from racer.frames import R_camera_from_body
from racer.navigator import Navigator, NavigatorConfig
from racer.vision.association import (
    apparent_size_px,
    associate,
    associate_scored,
    predict_gates_in_camera,
    range_consistent,
)
from racer.vision.gate_pose import estimate_gate_pose, project_gate_corners

_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _gate_facing_north(position, gate_id=0, inner=1.5) -> Gate:
    """Upright gate whose through-direction is +X (north): X=East, Y=Down, Z=North."""
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float), R_world_gate=R,
                inner_size_m=inner)


def _project_gate(gate: Gate, drone_pos, R_wb=None) -> np.ndarray:
    """The 4 inner corners the camera at ``drone_pos`` would see for this gate (ground truth)."""
    R_wb = np.eye(3) if R_wb is None else R_wb
    R_camera_world = (R_wb @ R_camera_from_body().T).T
    t = R_camera_world @ (gate.position_ned - np.asarray(drone_pos, float))
    R = R_camera_world @ gate.R_world_gate
    return project_gate_corners(R, t, inner_size_m=gate.inner_size_m)


def _obs(corners, ids=None, conf=True):
    n = corners.shape[0]
    return GateObservation(
        frame_id=0, sim_time_ns=0, corners_px=corners,
        corner_ids=None if ids is None else np.asarray(ids, int),
        corner_confidence=np.ones(n) if conf else None, score=0.9,
    )


# ---------------------------------------------------------------------------
# predict_gates_in_camera
# ---------------------------------------------------------------------------
def test_predicted_gates_carry_corners_matching_ground_truth():
    g = _gate_facing_north([9.0, 0.0, -2.5], gate_id=3)
    pred = predict_gates_in_camera([g], np.zeros(3), np.eye(3))
    assert set(pred) == {3}
    np.testing.assert_allclose(pred[3].corners_px, _project_gate(g, [0, 0, 0]), atol=1e-9)
    np.testing.assert_allclose(pred[3].center_px, pred[3].corners_px.mean(axis=0), atol=2.0)
    assert abs(pred[3].range_m - np.linalg.norm(g.position_ned)) < 1e-9


def test_gate_behind_camera_is_excluded():
    behind = _gate_facing_north([-9.0, 0.0, -2.5], gate_id=0)
    ahead = _gate_facing_north([9.0, 0.0, -2.5], gate_id=1)
    pred = predict_gates_in_camera([behind, ahead], np.zeros(3), np.eye(3))
    assert set(pred) == {1}


# ---------------------------------------------------------------------------
# association: the measured wrong-gate mechanism
# ---------------------------------------------------------------------------
def test_collinear_gates_associate_by_apparent_size():
    # The course failure: gates recede in a line, so every predicted CENTRE clusters near the
    # image centre and the retired nearest-centre rule matched a near detection to a far gate.
    # Shape-consistency must resolve both directions on a 10 m / 33 m collinear pair.
    g0 = _gate_facing_north([10.0, 0.0, -2.5], gate_id=0)
    g1 = _gate_facing_north([33.0, 0.0, -2.5], gate_id=1)
    pred = predict_gates_in_camera([g0, g1], np.zeros(3), np.eye(3))
    assert associate(_obs(_project_gate(g0, [0, 0, 0])), pred) == 0
    assert associate(_obs(_project_gate(g1, [0, 0, 0])), pred) == 1


def test_detection_matching_no_gate_scale_is_refused():
    # Only mapped gate is 30 m out; the detection is of a gate-sized object at 10 m (3x the
    # predicted apparent size) -> no association, no fix (the old rule matched it happily).
    far = _gate_facing_north([30.0, 0.0, -2.5], gate_id=0)
    seen = _gate_facing_north([10.0, 0.0, -2.5], gate_id=0)
    pred = predict_gates_in_camera([far], np.zeros(3), np.eye(3))
    assert associate(_obs(_project_gate(seen, [0, 0, 0])), pred) is None


def test_center_gate_is_scale_normalised():
    # The same metric offset (a ~2 m lateral prior error) is a huge pixel offset on a near
    # gate and a small one on a far gate; normalising by apparent size keeps the near gate
    # associable where a fixed pixel gate would refuse it...
    near = _gate_facing_north([4.0, 0.0, -1.0], gate_id=0)
    pred = predict_gates_in_camera([near], np.zeros(3), np.eye(3))
    obs_corners = _project_gate(near, [0.0, 2.2, 0.0])    # true drone 2.2 m east of the prior
    px_off = float(np.linalg.norm(obs_corners.mean(0) - pred[0].corners_px.mean(0)))
    assert px_off > 150.0                                  # the retired fixed gate would refuse
    assert associate(_obs(obs_corners), pred) == 0
    # ...while a far gate whose centre merely drifts near the detection stays refused: same
    # detected corners, but the only candidate sits at 30 m (wrong scale AND many units off).
    far_only = predict_gates_in_camera([_gate_facing_north([30.0, 0.0, -1.0], gate_id=7)],
                                       np.zeros(3), np.eye(3))
    assert associate(_obs(obs_corners), far_only) is None


def test_best_of_multiple_compatible_gates_wins():
    # Two same-range gates side by side: the detection must go to the one it overlaps.
    left = _gate_facing_north([12.0, -3.0, -2.0], gate_id=0)
    right = _gate_facing_north([12.0, 3.0, -2.0], gate_id=1)
    pred = predict_gates_in_camera([left, right], np.zeros(3), np.eye(3))
    gid, score = associate_scored(_obs(_project_gate(right, [0, 0, 0])), pred)
    assert gid == 1
    assert score < 0.1                                     # essentially exact agreement


def test_three_corner_observation_compares_like_for_like():
    g = _gate_facing_north([8.0, 0.0, -2.0], gate_id=0)
    pred = predict_gates_in_camera([g], np.zeros(3), np.eye(3))
    corners4 = _project_gate(g, [0, 0, 0])
    ids = [0, 1, 2]
    assert associate(_obs(corners4[ids], ids=ids), pred) == 0


def test_mid_transit_gate_without_projectable_corners_is_skipped():
    # Drone essentially in the gate plane: corners are at/behind the camera -> the gate is
    # not shape-checkable and must not be associated to (the P3P transit coast owns this).
    g = _gate_facing_north([0.05, 0.0, 0.0], gate_id=0)
    pred = predict_gates_in_camera([g], np.zeros(3), np.eye(3))
    if 0 in pred:                                          # origin may still be marginally in front
        assert pred[0].corners_px is None
        some_obs = _obs(np.array([[100.0, 100.0], [200.0, 100.0],
                                  [200.0, 200.0], [100.0, 200.0]]))
        assert associate(some_obs, pred) is None


def test_apparent_size_is_rms_corner_radius():
    sq = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    assert abs(apparent_size_px(10.0 * sq) - 10.0 * np.sqrt(2.0)) < 1e-9


# ---------------------------------------------------------------------------
# post-PnP depth sanity: the measured depth-flip/junk mechanism
# ---------------------------------------------------------------------------
def test_range_consistency_bounds():
    assert range_consistent(10.5, 10.0)                    # good-fix depth noise passes
    assert range_consistent(2.4, 1.6)                      # close range: absolute 1 m floor
    assert not range_consistent(20.0, 10.0)                # the 2x depth-junk signature
    assert not range_consistent(5.0, 10.0)                 # and the under-range direction
    assert not range_consistent(16.0, 138.0)               # wrong-gate residue (huge mismatch)


def test_predicted_prior_breaks_frontal_ippe_ambiguity():
    # A near-frontal square is the genuinely ambiguous IPPE case (ratio ~1): with the fresh
    # map+attitude predicted pose as prior, the solver must come back near the TRUE rotation,
    # not the flipped branch -- across a band of small noise seeds.
    g = _gate_facing_north([12.0, 0.3, -2.6], gate_id=0)   # slightly off-axis -> a real 2-fold pair
    R_camera_world = R_camera_from_body()
    t_true = R_camera_world @ g.position_ned
    R_true = R_camera_world @ g.R_world_gate
    corners = project_gate_corners(R_true, t_true, 1.5)
    pred = predict_gates_in_camera([g], np.zeros(3), np.eye(3))
    pg = pred[0]
    from racer.contracts import GatePose
    prior = GatePose(0, 0, pg.R_cam_gate, pg.t_cam_gate, 0.0, gate_id=0)
    rng = np.random.default_rng(7)
    for _ in range(10):
        noisy = corners + rng.normal(0.0, 0.3, size=corners.shape)
        pose = estimate_gate_pose(_obs(noisy), prior=prior)
        assert pose is not None
        cos = (np.trace(pose.R_cam_gate.T @ R_true) - 1.0) / 2.0
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 10.0


# ---------------------------------------------------------------------------
# navigator integration: a wrong-depth solve never reaches the KF
# ---------------------------------------------------------------------------
def _ds(sim_time_ns, position):
    return DroneState(sim_time_ns=int(sim_time_ns), orientation_ned_wxyz=_LEVEL_Q.copy(),
                      accel_body=_HOVER_ACCEL.copy(),
                      position_ned=np.asarray(position, float))


def _frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8))


def test_navigator_depth_sanity_rejects_wrong_depth_fix():
    # Map gate at 10 m; the detection is of a same-size gate at 6.5 m: apparent size ratio
    # ~1.54 slips the association gate, but the solved depth disagrees with the predicted
    # range by 3.5 m (> max(1, 0.25*10)) -> the depth sanity drops the fix at the source.
    map_gate = _gate_facing_north([10.0, 0.0, -2.5], gate_id=0)
    seen_gate = _gate_facing_north([6.5, 0.0, -2.5], gate_id=0)

    class _Det:
        def detect(self, frame):
            return [GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                    corners_px=_project_gate(seen_gate, [0, 0, 0]),
                                    corner_confidence=np.ones(4), score=0.9)]

    nav = Navigator(gates=[map_gate], detector=_Det(),
                    config=NavigatorConfig(use_given_position=True, given_pos_std=0.05))
    nav.update(_ds(0, [0, 0, 0]), _frame(0, 0))
    for k in range(1, 6):
        nav.update(_ds(k * 10_000_000, [0, 0, 0]), _frame(k, k * 10_000_000))
    assert nav.vision_diag.n_associated > 0                # it DID associate...
    assert nav.vision_diag.n_rejected_range > 0            # ...and the depth sanity dropped it
    assert nav.n_vision_fixes == 0
