import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import GateObservation
from racer.vision.gate_pose import _rotation_geodesic, estimate_gate_pose
from racer.vision.synthetic import (
    SyntheticSample,
    render_gate_sample,
    to_yolo_pose_label,
    write_dataset,
)


def test_render_shape_and_dtype():
    s = render_gate_sample(np.random.default_rng(0), pose=(np.eye(3), np.array([0.0, 0.0, 6.0])))
    assert s.image_bgr.shape == (360, 640, 3)
    assert s.image_bgr.dtype == np.uint8
    assert s.visible and s.keypoints_px.shape == (4, 2)


def test_label_format_and_normalization():
    s = SyntheticSample(
        image_bgr=np.zeros((360, 640, 3), np.uint8),
        keypoints_px=np.array([[100.0, 200.0], [300.0, 200.0], [300.0, 50.0], [100.0, 50.0]]),
        bbox_xywh=np.array([80.0, 40.0, 240.0, 180.0]),
        visible=True, R_cam_gate=np.eye(3), t_cam_gate=np.zeros(3),
    )
    parts = to_yolo_pose_label(s).split()
    assert len(parts) == 17                      # class + 4 bbox + 4*(x,y,v)
    assert parts[0] == "0"
    vals = list(map(float, parts[1:]))
    assert vals[0] == pytest.approx(200 / 640)   # cx
    assert vals[1] == pytest.approx(130 / 360)   # cy
    assert vals[2] == pytest.approx(240 / 640)   # w
    assert vals[3] == pytest.approx(180 / 360)   # h
    assert vals[4] == pytest.approx(100 / 640)   # kp0 x
    assert vals[5] == pytest.approx(200 / 360)   # kp0 y
    assert [parts[i] for i in (7, 10, 13, 16)] == ["2", "2", "2", "2"]   # visibility flags
    assert all(0.0 <= v <= 1.0 for v in vals if v != 2.0)


def test_label_none_when_not_visible():
    s = render_gate_sample(np.random.default_rng(0), pose=(np.eye(3), np.array([0.0, 0.0, -1.0])))
    assert s.visible is False
    assert to_yolo_pose_label(s) is None


def test_canonical_corner_order_frontal():
    s = render_gate_sample(np.random.default_rng(0), pose=(np.eye(3), np.array([0.0, 0.0, 6.0])))
    ll, lr, ur, ul = s.keypoints_px
    assert ll[0] < lr[0] and ul[0] < ur[0]       # left corners left of right corners
    assert ll[1] > ul[1] and lr[1] > ur[1]       # lower corners below upper (image y grows down)


def test_keypoints_are_pnp_consistent():
    # The generated keypoints must round-trip through gate_pose in the same convention.
    R = Rotation.from_euler("y", 0.4).as_matrix()
    t = np.array([0.3, 0.0, 6.0])
    s = render_gate_sample(np.random.default_rng(0), pose=(R, t))
    obs = GateObservation(frame_id=0, sim_time_ns=0, corners_px=s.keypoints_px)
    gp = estimate_gate_pose(obs)
    assert gp is not None
    np.testing.assert_allclose(gp.t_cam_gate, t, atol=1e-3)
    assert _rotation_geodesic(gp.R_cam_gate, R) < 2e-3


def test_label_string_is_pnp_consistent():
    # [YOLO constraint 1] Parse the actual written YOLO-pose row back into pixel keypoints
    # and round-trip through gate_pose. This proves the on-disk label ORDER matches the PnP
    # convention -- a wrong keypoint order would train the detector to emit corners that make
    # the PnP silently produce garbage poses.
    from racer.frames import IMAGE_HEIGHT, IMAGE_WIDTH

    R = Rotation.from_euler("y", 0.35).as_matrix()
    t = np.array([0.2, -0.1, 6.0])
    s = render_gate_sample(np.random.default_rng(3), pose=(R, t))
    row = to_yolo_pose_label(s).split()           # class cx cy w h (x y v)*4
    kp = [[float(row[5 + 3 * i]) * IMAGE_WIDTH, float(row[6 + 3 * i]) * IMAGE_HEIGHT]
          for i in range(4)]
    obs = GateObservation(frame_id=0, sim_time_ns=0, corners_px=np.array(kp))
    gp = estimate_gate_pose(obs)
    assert gp is not None
    np.testing.assert_allclose(gp.t_cam_gate, t, atol=1e-2)
    assert _rotation_geodesic(gp.R_cam_gate, R) < 5e-3


def test_bbox_encloses_keypoints():
    s = render_gate_sample(np.random.default_rng(1), pose=(np.eye(3), np.array([0.0, 0.0, 7.0])))
    x, y, w, h = s.bbox_xywh
    assert (s.keypoints_px[:, 0] >= x - 1e-6).all() and (s.keypoints_px[:, 0] <= x + w + 1e-6).all()
    assert (s.keypoints_px[:, 1] >= y - 1e-6).all() and (s.keypoints_px[:, 1] <= y + h + 1e-6).all()


def test_determinism():
    a = render_gate_sample(np.random.default_rng(7))
    b = render_gate_sample(np.random.default_rng(7))
    assert np.array_equal(a.image_bgr, b.image_bgr)
    np.testing.assert_array_equal(a.keypoints_px, b.keypoints_px)


def test_write_dataset(tmp_path):
    yaml_path = write_dataset(tmp_path, n_train=4, n_val=2, seed=0)
    assert yaml_path.exists()
    imgs = list((tmp_path / "images" / "train").glob("*.png"))
    lbls = list((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(imgs) == 4 and len(lbls) == 4
    assert len(list((tmp_path / "images" / "val").glob("*.png"))) == 2
    # Every label is a well-formed 17-field YOLO-pose row.
    for lbl in lbls:
        assert len(lbl.read_text().split()) == 17
    assert "kpt_shape: [4, 3]" in yaml_path.read_text()
    assert "flip_idx: [1, 0, 3, 2]" in yaml_path.read_text()
