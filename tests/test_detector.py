import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import Frame, GatePose
from racer.vision.detector import (
    GateDetector,
    observations_from_keypoints,
    observations_from_results,
)
from racer.vision.gate_pose import _rotation_geodesic, estimate_gate_pose, project_gate_corners

_R = Rotation.from_euler("y", 0.3).as_matrix()
_T = np.array([0.3, -0.2, 5.0])


def _frame(frame_id=5, sim_time_ns=123):
    return Frame(frame_id=frame_id, sim_time_ns=sim_time_ns, image_bgr=np.zeros((360, 640, 3), np.uint8))


def _corners():
    return project_gate_corners(_R, _T)  # (4,2), canonical order


# -- pure core -------------------------------------------------------------
def test_four_corner_detection_round_trips_through_pnp():
    frame = _frame()
    kxy = _corners()[None]                       # (1,4,2)
    obs = observations_from_keypoints(frame, kxy, np.ones((1, 4)), np.array([0.9]))
    assert len(obs) == 1
    o = obs[0]
    assert o.frame_id == 5 and o.sim_time_ns == 123 and o.gate_id is None
    assert o.score == pytest.approx(0.9) and o.corner_ids is None and o.corners_px.shape == (4, 2)
    # End-to-end: detector output -> GateObservation -> PnP recovers the true pose.
    gp = estimate_gate_pose(o)
    np.testing.assert_allclose(gp.t_cam_gate, _T, atol=1e-3)
    assert _rotation_geodesic(gp.R_cam_gate, _R) < 2e-3 and gp.n_corners == 4


def test_low_confidence_keypoint_becomes_three_corner_observation():
    frame = _frame()
    corners = _corners()
    conf = np.array([[0.9, 0.9, 0.1, 0.9]])      # corner 2 below threshold
    obs = observations_from_keypoints(frame, corners[None], conf, np.array([0.8]), kpt_conf_thresh=0.5)
    assert len(obs) == 1
    o = obs[0]
    assert o.corners_px.shape == (3, 2)
    np.testing.assert_array_equal(o.corner_ids, [0, 1, 3])
    np.testing.assert_allclose(o.corners_px, corners[[0, 1, 3]])
    # Recovers via the P3P fallback when given a temporal prior.
    prior = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=_R, t_cam_gate=_T, reproj_error_px=0.0)
    gp = estimate_gate_pose(o, prior=prior)
    assert gp.n_corners == 3
    np.testing.assert_allclose(gp.t_cam_gate, _T, atol=1e-3)


def test_low_score_detection_is_dropped():
    frame = _frame()
    obs = observations_from_keypoints(frame, _corners()[None], np.ones((1, 4)), np.array([0.1]),
                                      score_thresh=0.25)
    assert obs == []


def test_fewer_than_three_visible_corners_dropped():
    frame = _frame()
    conf = np.array([[0.9, 0.9, 0.1, 0.1]])      # only 2 corners clear the threshold
    obs = observations_from_keypoints(frame, _corners()[None], conf, np.array([0.9]))
    assert obs == []


def test_multiple_detections_and_passthrough():
    frame = _frame()
    kxy = np.stack([_corners(), _corners() + 3.0])   # (2,4,2)
    scores = np.array([0.9, 0.4])
    bboxes = np.array([[100.0, 90.0, 50.0, 40.0], [200.0, 90.0, 50.0, 40.0]])
    obs = observations_from_keypoints(frame, kxy, np.ones((2, 4)), scores, bboxes_xywh=bboxes)
    assert len(obs) == 2
    assert obs[0].score == pytest.approx(0.9) and obs[1].score == pytest.approx(0.4)
    np.testing.assert_array_equal(obs[0].bbox_xywh, [100.0, 90.0, 50.0, 40.0])
    assert obs[0].corner_confidence.shape == (4,)


def test_bad_keypoint_shape_raises():
    with pytest.raises(ValueError):
        observations_from_keypoints(_frame(), np.zeros((1, 3, 2)), np.ones((1, 3)), np.array([0.9]))


# -- ultralytics Results extraction (duck-typed fake; no ultralytics needed) -----
class _Attr:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_results(kxy, kconf, scores, bboxes=None):
    return _Attr(
        boxes=_Attr(conf=np.asarray(scores), xywh=None if bboxes is None else np.asarray(bboxes)),
        keypoints=_Attr(xy=np.asarray(kxy), conf=np.asarray(kconf)),
    )


def test_observations_from_results_extracts_and_adapts():
    frame = _frame()
    res = _fake_results(_corners()[None], np.ones((1, 4)), [0.9])
    obs = observations_from_results(frame, res)
    assert len(obs) == 1 and obs[0].corners_px.shape == (4, 2)


def test_observations_from_results_handles_no_detections():
    frame = _frame()
    assert observations_from_results(frame, _Attr(boxes=None, keypoints=None)) == []
    empty = _Attr(boxes=_Attr(conf=np.zeros(0), xywh=None), keypoints=_Attr(xy=np.zeros((0, 4, 2)), conf=np.zeros((0, 4))))
    assert observations_from_results(frame, empty) == []


# -- GateDetector wrapper with an injected fake model ----------------------
class _FakeModel:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def predict(self, img, **kwargs):
        self.calls += 1
        return self._results


def test_gate_detector_delegates_to_model():
    frame = _frame()
    model = _FakeModel([_fake_results(_corners()[None], np.ones((1, 4)), [0.9])])
    det = GateDetector(model)
    obs = det.detect(frame)
    assert model.calls == 1 and len(obs) == 1
    assert det.detect(_frame()) is not None


def test_gate_detector_empty_results():
    det = GateDetector(_FakeModel([]))
    assert det.detect(_frame()) == []
