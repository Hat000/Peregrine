import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import GateObservation
from racer.vision.gate_pose import _rotation_geodesic, estimate_gate_pose
from racer.vision.synthetic import (
    SyntheticSample,
    _in_frame,
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


# -- curriculum levels + transit / clipping / occlusion edge cases [2026-05-29] -----------
def test_clipped_corner_label_flags_offframe():
    # A 3-corner sample (one corner off the left edge): clamped into [0,1] with v=0; the
    # other three v=2 -- exactly what trains YOLO to emit the weak corner the P3P path drops.
    kp = np.array([[100.0, 300.0], [300.0, 300.0], [300.0, 50.0], [-40.0, 50.0]])  # corner 3 off-frame
    s = SyntheticSample(
        image_bgr=np.zeros((360, 640, 3), np.uint8), keypoints_px=kp,
        bbox_xywh=np.array([0.0, 50.0, 300.0, 250.0]), visible=True,
        R_cam_gate=np.eye(3), t_cam_gate=np.zeros(3), visibility=np.array([2, 2, 2, 0]),
    )
    parts = to_yolo_pose_label(s).split()
    assert len(parts) == 17
    assert [parts[i] for i in (7, 10, 13, 16)] == ["2", "2", "2", "0"]   # per-corner visibility
    assert float(parts[14]) == pytest.approx(0.0)                        # off-frame x clamped to 0
    assert all(0.0 <= float(parts[5 + 3 * i]) <= 1.0 for i in range(4))  # all coords valid [0,1]


def test_generator_produces_three_corner_clips():
    # Aggressive close poses push a corner off-frame -> usable 3-corner samples, from L1 up.
    rng = np.random.default_rng(0)
    clips = sum(
        1 for _ in range(400)
        if (s := render_gate_sample(rng, level=1)).visible and (s.visibility == 0).any()
    )
    assert clips > 0


def test_occlusion_marks_a_corner_occluded():
    # L2 occluders cover a corner (in-frame but hidden) -> v=1 (occluded) 3-corner samples.
    rng = np.random.default_rng(1)
    occ = sum(
        1 for _ in range(400)
        if (s := render_gate_sample(rng, level=2)).visible and (s.visibility == 1).any()
    )
    assert occ > 0


def test_close_range_transit_outer_offscreen_inner_visible():
    # The drone mid-transit: outer boundary leaves the frame, all 4 inner corners stay in.
    s = render_gate_sample(np.random.default_rng(0), level=1, pose=(np.eye(3), np.array([0.0, 0.0, 1.8])))
    assert s.visible and (s.visibility == 2).all()      # a 4-corner transit sample
    assert _in_frame(s.keypoints_px)
    assert s.bbox_xywh[3] > 0.8 * 360                    # outer clipped -> bbox spans near full height


def test_level3_chaos_preserves_shape_and_pnp():
    # L3 chaos is photometric: image is heavily edited, keypoints stay the geometric projection.
    R = Rotation.from_euler("y", 0.3).as_matrix()
    t = np.array([0.2, 0.0, 6.0])
    s = render_gate_sample(np.random.default_rng(2), level=3, pose=(R, t))
    assert s.image_bgr.shape == (360, 640, 3) and s.image_bgr.dtype == np.uint8
    gp = estimate_gate_pose(GateObservation(frame_id=0, sim_time_ns=0, corners_px=s.keypoints_px))
    assert gp is not None
    np.testing.assert_allclose(gp.t_cam_gate, t, atol=1e-3)


def test_level2_adds_directional_shading():
    # L1 gate ring is uniform; L2 has a directional luminance gradient (the sim-lighting fix).
    # Average over seeds since the gradient DIRECTION is random (its magnitude across the ring
    # varies seed to seed); the point is L2 has clear spatial variance where L1 has ~none.
    import racer.vision.synthetic as syn
    R, t = np.eye(3), np.array([0.0, 0.0, 5.0])
    inner = syn.project_gate_corners(R, t, syn.GATE_INNER_SIZE_M)
    outer = syn.project_gate_corners(R, t, syn.GATE_OUTER_SIZE_M)
    ring = np.zeros((360, 640), np.uint8)
    cv2.fillPoly(ring, [outer.round().astype(np.int32)], 1)
    cv2.fillPoly(ring, [inner.round().astype(np.int32)], 0)
    mask = ring == 1

    def lum_std(level, seed):
        img = np.zeros((360, 640, 3), np.uint8)
        syn._draw_gate(img, inner, outer, np.random.default_rng(seed), level)
        return float(img.mean(axis=2)[mask].std())

    l1 = float(np.mean([lum_std(1, sd) for sd in range(6)]))
    l2 = float(np.mean([lum_std(2, sd) for sd in range(6)]))
    assert l1 < 2.0       # uniform across the ring
    assert l2 > 3.0       # directional gradient adds clear spatial variance


def test_write_dataset_respects_level(tmp_path):
    for lvl in (1, 3):
        d = tmp_path / f"l{lvl}"
        write_dataset(d, n_train=3, n_val=1, level=lvl, seed=0)
        lbls = list((d / "labels" / "train").glob("*.txt"))
        assert len(lbls) == 3
        for lbl in lbls:
            assert len(lbl.read_text().split()) == 17


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
