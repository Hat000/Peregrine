"""Laptop tests for the VQ2 photoreal dataset generator (NO Blender required).

Covers the load-bearing correctness: intrinsics reproduce K, the corner projection matches the
deployed estimator's frames.py chain, labels round-trip through PnP, the albumentations
post-pipeline is keypoint-aware, presets validate, and the full procedural-backend pipeline
writes a well-formed YOLO-pose dataset whose on-disk labels recover the gate pose.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import GateObservation
from racer.frames import (
    R_world_from_body,
    project_camera_point,
    world_point_in_camera,
)
from racer.vision.gate_pose import _rotation_geodesic, estimate_gate_pose
from racer.vision.synthetic import SyntheticSample, to_yolo_pose_label

from racer.vision.blender_gen import contract
from racer.vision.blender_gen.augment import AugmentConfig, augment_frame
from racer.vision.blender_gen.config import (
    AppearanceConfig,
    ScenarioPreset,
    available_presets,
    load_preset,
    preset_from_dict,
)
from racer.vision.blender_gen.contract import (
    CAMERA_INTRINSICS_K,
    GATE_INNER_SIZE_M,
    R_OPTICAL_TO_BLENDER_WORLD,
    gate_object_points,
    project_gate_corners,
)
from racer.vision.blender_gen.dataset import ProceduralBackend, generate_dataset, sample_gate_color_bgr
from racer.vision.blender_gen.geometry import (
    GateRender,
    ViewpointConfig,
    load_course_gates,
    optical_pose,
    sample_frame,
    sample_frames,
)
from racer.vision.blender_gen.intrinsics import (
    K_from_blender_params,
    blender_camera_params,
    horizontal_fov_rad,
    reprojection_max_error_px,
)
from racer.vision.blender_gen.labels import frame_label_rows, gate_render_to_row, to_yolo_pose_row


# ------------------------------------------------------------------ intrinsics
def test_intrinsics_reproduce_K_to_floating_point():
    assert reprojection_max_error_px() < 1e-9


def test_intrinsics_lens_and_fov():
    p = blender_camera_params()
    assert p.lens_mm == pytest.approx(18.0)               # 320 * 36 / 640
    assert p.sensor_fit == "HORIZONTAL"
    assert p.shift_x == 0.0 and p.shift_y == 0.0
    assert np.rad2deg(horizontal_fov_rad(p)) == pytest.approx(90.0, abs=1e-6)


def test_intrinsics_roundtrip_matrix():
    np.testing.assert_allclose(K_from_blender_params(blender_camera_params()), CAMERA_INTRINSICS_K, atol=1e-9)


def test_intrinsics_offcentre_principal_point_rejected():
    K = CAMERA_INTRINSICS_K.copy()
    K[0, 2] += 5.0
    with pytest.raises(ValueError, match="principal point"):
        blender_camera_params(K)


# ------------------------------------------------------------- frame convention
def test_optical_to_blender_world_is_proper_involution():
    M = R_OPTICAL_TO_BLENDER_WORLD
    np.testing.assert_allclose(M @ M, np.eye(3), atol=1e-12)              # involutory (own inverse)
    assert np.linalg.det(M) == pytest.approx(1.0)                         # proper rotation, no reflection
    np.testing.assert_allclose(M, Rotation.from_euler("x", np.pi).as_matrix(), atol=1e-12)


def test_corner_projection_matches_frames_chain():
    # The generator's labels MUST equal the deployed estimator's projection of the same gate.
    gates = load_course_gates()
    gate = gates[2]
    body = np.array([-60.0, 1.0, 8.0])
    R_wb = R_world_from_body(0.1, -0.05, np.deg2rad(178.0))
    R_cg, t_cg = optical_pose(gate, body, R_wb)
    corners_gen = project_gate_corners(R_cg, t_cg, GATE_INNER_SIZE_M)
    corners_ref = []
    for p_g in gate_object_points(GATE_INNER_SIZE_M):
        p_world = gate.R_world_gate @ p_g + gate.position_ned
        uv = project_camera_point(world_point_in_camera(p_world, R_wb, body))
        assert uv is not None
        corners_ref.append(uv)
    np.testing.assert_allclose(corners_gen, np.array(corners_ref), atol=1e-9)


# --------------------------------------------------------------------- labels
def test_label_row_matches_synthetic_writer():
    # Identical formatting/contract to the procedural generator the detector also trains on.
    kp = np.array([[100.0, 200.0], [300.0, 200.0], [300.0, 50.0], [100.0, 50.0]])
    vis = np.array([2, 2, 1, 0])
    bbox = np.array([80.0, 40.0, 240.0, 180.0])
    mine = to_yolo_pose_row(kp, vis, bbox)
    s = SyntheticSample(image_bgr=np.zeros((360, 640, 3), np.uint8), keypoints_px=kp,
                        bbox_xywh=bbox, visible=True, R_cam_gate=np.eye(3), t_cam_gate=np.zeros(3),
                        visibility=vis)
    assert mine == to_yolo_pose_label(s)
    assert len(mine.split()) == 17


def test_label_roundtrips_through_pnp():
    # An on-disk label parsed back into pixels must recover the gate pose (order == PnP order).
    R = Rotation.from_euler("y", 0.3).as_matrix()
    t = np.array([0.4, -0.2, 9.0])
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M)
    outer = project_gate_corners(R, t, contract.GATE_OUTER_SIZE_M)
    gr = GateRender(gate_id=0, R_cam_gate=R, t_cam_gate=t, keypoints_px=inner, outer_px=outer,
                    bbox_xywh=np.array([0.0, 0.0, 640.0, 360.0]),
                    visibility=np.array([2, 2, 2, 2]), visible=True)
    row = gate_render_to_row(gr).split()
    kp = np.array([[float(row[5 + 3 * i]) * contract.IMAGE_WIDTH,
                    float(row[6 + 3 * i]) * contract.IMAGE_HEIGHT] for i in range(4)])
    gp = estimate_gate_pose(GateObservation(frame_id=0, sim_time_ns=0, corners_px=kp))
    assert gp is not None
    np.testing.assert_allclose(gp.t_cam_gate, t, atol=1e-2)
    assert _rotation_geodesic(gp.R_cam_gate, R) < 5e-3


def test_label_emits_8_keypoints_inner_then_outer():
    # The VQ2 row carries 8 keypoints: inner 0..3 (== PnP corners) then outer 4..7 (gate frame).
    R = Rotation.from_euler("y", 0.2).as_matrix()
    t = np.array([0.3, -0.1, 8.0])
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M)
    outer = project_gate_corners(R, t, contract.GATE_OUTER_SIZE_M)
    gr = GateRender(gate_id=0, R_cam_gate=R, t_cam_gate=t, keypoints_px=inner, outer_px=outer,
                    bbox_xywh=np.array([0.0, 0.0, 640.0, 360.0]),
                    visibility=np.array([2, 2, 2, 2]), visible=True)
    f = gate_render_to_row(gr).split()
    assert len(f) == 29                                                  # 5 + 8*(x y v)
    kp = np.array([[float(f[5 + 3 * i]) * contract.IMAGE_WIDTH,
                    float(f[6 + 3 * i]) * contract.IMAGE_HEIGHT] for i in range(8)])
    np.testing.assert_allclose(kp[0:4], inner, atol=0.2)                 # inner keypoints
    np.testing.assert_allclose(kp[4:8], outer, atol=0.2)                 # outer keypoints
    assert outer[:, 0].max() - outer[:, 0].min() > inner[:, 0].max() - inner[:, 0].min()


# --------------------------------------------------------- albumentations aug
def test_albumentations_keypoint_alignment():
    # Apply a KNOWN geometric translate through the same KeypointParams config and assert the
    # corners move with the image (keypoint-aware). Proves geometric augs transform the labels.
    A = pytest.importorskip("albumentations")
    W = contract.IMAGE_WIDTH
    comp = A.Compose(
        [A.Affine(translate_percent={"x": 0.1, "y": 0.0}, scale=1.0, rotate=0.0, shear=0.0, p=1.0)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    img = np.zeros((contract.IMAGE_HEIGHT, W, 3), np.uint8)
    out = comp(image=img, keypoints=[(200.0, 180.0)])
    x, y = out["keypoints"][0][:2]
    assert abs(abs(x - 200.0) - 0.1 * W) < 2.0                            # moved ~0.1*W in x
    assert abs(y - 180.0) < 2.0                                           # y unchanged


def test_augment_photometric_preserves_keypoints():
    # Photometric-only aug must NOT move the corners (labels unchanged).
    gates = load_course_gates()
    fs = _frame_with_label(gates)
    g = fs.labeled_gates[0]
    cfg = AugmentConfig(enable=True, photometric_p=1.0, lighting_p=0.0,
                        brightness_p=0.0, geometric_p=0.0)
    img = np.full((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3), 120, np.uint8)
    _, out_gates = augment_frame(img, fs.gates, cfg, np.random.default_rng(0))
    g2 = [x for x in out_gates if x.gate_id == g.gate_id][0]
    np.testing.assert_allclose(g2.keypoints_px, g.keypoints_px, atol=1e-9)


def test_augment_geometric_keeps_bbox_enclosing_corners():
    gates = load_course_gates()
    fs = _frame_with_label(gates)
    cfg = AugmentConfig(enable=True, photometric_p=0.0, lighting_p=0.0,
                        brightness_p=0.0, geometric_p=1.0)
    img = np.full((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3), 120, np.uint8)
    _, out_gates = augment_frame(img, fs.gates, cfg, np.random.default_rng(1))
    for g in out_gates:
        if not g.visible:
            continue
        x, y, w, h = g.bbox_xywh
        inb = (g.visibility != contract.V_OFF)
        kp = g.keypoints_px[inb]
        assert (kp[:, 0] >= x - 1.0).all() and (kp[:, 0] <= x + w + 1.0).all()
        assert (kp[:, 1] >= y - 1.0).all() and (kp[:, 1] <= y + h + 1.0).all()


# ------------------------------------------------------------------- presets
def test_all_presets_load_and_validate():
    names = available_presets()
    assert set(names) >= {"vq1_faithful", "appearance_broad", "hard_visual",
                          "long_range", "terminal_approach"}
    for n in names:
        p = load_preset(n)
        assert p.name == n
        assert p.viewpoint.range_min_m < p.viewpoint.range_max_m


def test_preset_roundtrip():
    p = load_preset("appearance_broad")
    assert preset_from_dict(p.to_dict()).to_dict() == p.to_dict()


def test_preset_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown"):
        preset_from_dict({"name": "x", "viewpoint": {"bogus_key": 1}})


def test_preset_bad_range_rejected():
    with pytest.raises(ValueError, match="range"):
        preset_from_dict({"name": "x", "viewpoint": {"range_min_m": 30.0, "range_max_m": 2.0}})


def test_preset_bad_engine_rejected():
    with pytest.raises(ValueError, match="engine"):
        preset_from_dict({"name": "x", "render": {"engine": "RAYTRACE9000"}})


def test_preset_bad_lighting_mode_rejected():
    with pytest.raises(ValueError, match="lighting_mode"):
        preset_from_dict({"name": "x", "appearance": {"lighting_mode": "disco"}})


def test_preset_bad_negative_fraction_rejected():
    with pytest.raises(ValueError, match="negative_fraction"):
        preset_from_dict({"name": "x", "negative_fraction": 1.5})


def test_negatives_preset_is_all_negative():
    p = load_preset("negatives")
    assert p.negative_fraction == 1.0
    assert p.appearance.lighting_mode == "dark_arena"


# ----------------------------------------------------------------- viewpoints
def _frame_with_label(gates):
    rng = np.random.default_rng(3)
    for _ in range(200):
        fs = sample_frame(rng, gates, ViewpointConfig())
        if fs.labeled_gates:
            return fs
    raise AssertionError("no labelled frame sampled")


def test_viewpoint_envelope_and_visibility():
    cfg = ViewpointConfig()
    count = 0
    multi = 0
    for fs in sample_frames(120, cfg, seed=5):
        labs = fs.labeled_gates
        assert labs
        count += 1
        multi += (len(labs) > 1)
        for g in labs:
            assert 0.6 * cfg.range_min_m <= g.range_m <= 1.4 * cfg.range_max_m
            assert set(int(v) for v in g.visibility) <= {0, 1, 2}
            assert int((g.visibility == contract.V_VIS).sum()) >= 3
    assert count == 120
    assert multi > 0                                                     # co-visibility appears


def test_sampled_gate_label_roundtrips_through_pnp():
    gates = load_course_gates()
    fs = _frame_with_label(gates)
    g = fs.labeled_gates[0]
    # a fully-visible gate's keypoints must recover its sampled optical pose
    if int((g.visibility == contract.V_VIS).sum()) == 4:
        obs = GateObservation(frame_id=0, sim_time_ns=0, corners_px=g.keypoints_px)
        gp = estimate_gate_pose(obs)
        assert gp is not None
        np.testing.assert_allclose(gp.t_cam_gate, g.t_cam_gate, atol=1e-2)


def test_terminal_preset_produces_three_corner_clips():
    # Close range + 20 deg uptilt should push a corner out of frame -> 3-corner samples.
    p = load_preset("terminal_approach")
    clips = 0
    for fs in sample_frames(150, p.viewpoint, seed=7):
        for g in fs.labeled_gates:
            if (g.visibility == contract.V_OFF).any():
                clips += 1
    assert clips > 0


# --------------------------------------------------------- procedural backend
def test_gate_color_vq1_faithful_is_extracted_red():
    ap = AppearanceConfig(use_vq1_red=True, gate_hue_jitter=0.0,
                          gate_sat_range=(1.0, 1.0), gate_val_range=(1.0, 1.0))
    b, g, r = sample_gate_color_bgr(np.random.default_rng(0), ap)
    assert r > 200 and g < 110 and b < 60                                # vivid orange-red


def test_generate_dataset_procedural_end_to_end(tmp_path):
    preset = load_preset("vq1_faithful")
    yaml_path = generate_dataset(tmp_path, preset, ProceduralBackend(),
                                 n_train=8, n_val=2, seed=0)
    assert yaml_path.exists()
    txt = yaml_path.read_text()
    assert "kpt_shape: [8, 3]" in txt and "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]" in txt
    imgs = sorted((tmp_path / "images" / "train").glob("*.png"))
    lbls = sorted((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(imgs) == 8 and len(lbls) == 8
    assert len(list((tmp_path / "images" / "val").glob("*.png"))) == 2
    for lbl in lbls:
        for line in lbl.read_text().splitlines():
            assert len(line.split()) == 29          # class + cx cy w h + 8*(x y v)
            assert line.split()[0] == "0"


def test_generated_label_recovers_pose(tmp_path):
    # End-to-end: a written label (no geometric aug) parsed back recovers a gate pose -> the
    # rendered pixels, the on-disk label and PnP all agree.
    preset = ScenarioPreset(name="t", augment=AugmentConfig(enable=False))
    generate_dataset(tmp_path, preset, ProceduralBackend(), n_train=12, n_val=1, seed=1)
    recovered = 0
    for lbl in (tmp_path / "labels" / "train").glob("*.txt"):
        for line in lbl.read_text().splitlines():
            r = line.split()
            vis = [int(r[7 + 3 * i]) for i in range(4)]
            if sum(v == 2 for v in vis) < 4:
                continue
            kp = np.array([[float(r[5 + 3 * i]) * contract.IMAGE_WIDTH,
                            float(r[6 + 3 * i]) * contract.IMAGE_HEIGHT] for i in range(4)])
            gp = estimate_gate_pose(GateObservation(frame_id=0, sim_time_ns=0, corners_px=kp))
            if gp is not None and 1.0 < gp.range_m < 40.0:
                recovered += 1
    assert recovered > 0


def test_generate_dataset_with_negatives(tmp_path):
    # negative_fraction => some images carry an EMPTY label file (ultralytics background) and the
    # positives still round-trip. 50% of 12 = 6 empty + 6 labelled.
    preset = ScenarioPreset(name="halfneg", negative_fraction=0.5,
                            augment=AugmentConfig(enable=False))
    generate_dataset(tmp_path, preset, ProceduralBackend(), n_train=12, n_val=2, seed=3)
    lbls = sorted((tmp_path / "labels" / "train").glob("*.txt"))
    imgs = sorted((tmp_path / "images" / "train").glob("*.png"))
    assert len(imgs) == 12 and len(lbls) == 12
    empties = [p for p in lbls if p.read_text().strip() == ""]
    positives = [p for p in lbls if p.read_text().strip() != ""]
    assert len(empties) == 6 and len(positives) == 6
    for p in positives:
        for line in p.read_text().splitlines():
            assert len(line.split()) == 29


def test_all_presets_including_negatives_validate():
    assert "negatives" in available_presets()
    for n in available_presets():
        load_preset(n)                                          # raises on any schema violation


def test_gate_ring_mask_is_annulus_under_keypoints():
    # The mask carves the inner opening: keypoint-centre pixel is background, a point on the ring
    # (between inner and outer edge) is gate. Built directly from a known GateRender.
    from racer.vision.blender_gen.masks import gate_ring_mask
    R = np.eye(3)
    t = np.array([0.0, 0.0, 6.0])
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M)
    outer = project_gate_corners(R, t, contract.GATE_OUTER_SIZE_M)
    gr = GateRender(gate_id=0, R_cam_gate=R, t_cam_gate=t, keypoints_px=inner, outer_px=outer,
                    bbox_xywh=np.array([0.0, 0.0, 640.0, 360.0]),
                    visibility=np.array([2, 2, 2, 2]), visible=True)
    m = gate_ring_mask([gr])
    assert m.shape == (contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH) and m.dtype == np.uint8
    cen = inner.mean(axis=0).round().astype(int)
    assert m[cen[1], cen[0]] == 0                                         # opening is background
    # midpoint between an inner and the matching outer corner lies on the ring
    mid = ((inner[0] + outer[0]) / 2).round().astype(int)
    assert m[mid[1], mid[0]] > 0                                          # ring is gate
    assert set(np.unique(m).tolist()) <= {0, 1}                           # one gate -> {0,1}


def test_generate_dataset_emits_masks(tmp_path):
    preset = load_preset("vq1_faithful")
    generate_dataset(tmp_path, preset, ProceduralBackend(), n_train=6, n_val=2, seed=0,
                     emit_masks=True)
    import cv2
    masks = sorted((tmp_path / "masks" / "train").glob("*.png"))
    imgs = sorted((tmp_path / "images" / "train").glob("*.png"))
    assert len(masks) == len(imgs) == 6                                   # one mask per image
    nonzero = 0
    for mp in masks:
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        assert m.shape == (contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH)
        nonzero += int((m > 0).any())
    assert nonzero == 6                                                   # every positive frame has gate pixels


def test_generate_dataset_determinism(tmp_path):
    preset = load_preset("vq1_faithful")
    generate_dataset(tmp_path / "a", preset, ProceduralBackend(), n_train=4, n_val=1, seed=42)
    generate_dataset(tmp_path / "b", preset, ProceduralBackend(), n_train=4, n_val=1, seed=42)
    for i in range(4):
        a = (tmp_path / "a" / "labels" / "train" / f"{i:06d}.txt").read_text()
        b = (tmp_path / "b" / "labels" / "train" / f"{i:06d}.txt").read_text()
        assert a == b
