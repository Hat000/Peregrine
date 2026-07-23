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

from racer.vision.blender_gen import contract, geometry
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
            # Far cap only. The old NEAR bound (0.6 * range_min) silently dropped any gate closer
            # than that WHILE THE BACKEND STILL RENDERED IT, so the most visually dominant gate in
            # the frame became unlabelled background. See _has_labellable_area.
            assert g.range_m <= 1.4 * cfg.range_max_m
            assert set(int(v) for v in g.visibility) <= {0, 1, 2}
            # A label now requires VISIBLE AREA, not >=3 in-frame corners. The corner rule was a
            # 4-keypoint-PnP legacy: the 8-keypoint rescue needs >=4 usable of 8 and segmentation
            # needs only pixels, so a cropped close gate showing one corner is still labellable.
            assert geometry._clipped_area_px(np.asarray(g.outer_px, float)) >= geometry.MIN_LABEL_AREA_PX
    assert count == 120
    assert multi > 0                                                     # co-visibility appears


def test_close_cropped_gates_are_labelled_not_left_as_background():
    """REGRESSION (2026-07-22). The generator rendered every gate in the spec but labels.py skipped
    any with visible=False, and visible required >=3 in-frame inner corners AND the centre in
    frame. A close cropped gate was therefore drawn into the image and presented to training as
    BACKGROUND -- teaching the detector to suppress exactly what it must fire on. Signature of the
    damage: render arms averaged ~1.0 labelled gates/frame vs 2.62 for hand-labelled real frames."""
    cfg = ViewpointConfig()
    few_corner_labelled = no_centre_labelled = 0
    for fs in sample_frames(200, cfg, seed=11):
        for g in fs.labeled_gates:
            if int((g.visibility == contract.V_VIS).sum()) < 3:
                few_corner_labelled += 1
            c = np.asarray(g.keypoints_px, float).mean(axis=0)
            if not (0 <= c[0] <= contract.IMAGE_WIDTH and 0 <= c[1] <= contract.IMAGE_HEIGHT):
                no_centre_labelled += 1
    assert few_corner_labelled > 0, "gates with <3 in-frame corners must now be labelled"
    assert no_centre_labelled > 0, "gates whose CENTRE is off-frame must now be labelled"


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


# ==========================================================================================
# TRUE-SILHOUETTE seg target: backend routing + the augmentation mask warp
# ==========================================================================================
def test_augment_carries_the_mask_through_the_geometric_warp():
    """THE trap this feature can fall into. The silhouettes are measured on the CLEAN render but the
    AUGMENTED image is what gets written, and four presets enable geometric warps. If the mask does
    not ride the same transform, every label in those runs is plausibly-shaped and systematically
    offset -- the kind of poisoning no census can see."""
    from racer.vision.blender_gen.augment import augment_frame_with_masks

    # A head-on gate at 8 m: small and well inside the frame, so nothing is clipped either before or
    # after the warp and the comparison below is exact rather than confounded by border clipping.
    import cv2
    R, t = np.eye(3), np.array([0.0, 0.0, 8.0])
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M)
    outer = project_gate_corners(R, t, contract.GATE_OUTER_SIZE_M)
    g = GateRender(gate_id=0, R_cam_gate=R, t_cam_gate=t, keypoints_px=inner, outer_px=outer,
                   bbox_xywh=np.array([0.0, 0.0, 640.0, 360.0]),
                   visibility=np.array([2, 2, 2, 2]), visible=True,
                   outer_visibility=np.array([2, 2, 2, 2]))
    cfg = AugmentConfig(enable=True, photometric_p=0.0, lighting_p=0.0,
                        brightness_p=0.0, geometric_p=1.0)
    img = np.full((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3), 120, np.uint8)

    ring = np.zeros((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH), np.uint8)
    hole = np.zeros_like(ring)
    cv2.fillPoly(ring, [g.outer_px.round().astype(np.int32)], 1)           # "ring" stand-in
    cv2.fillPoly(hole, [g.keypoints_px.round().astype(np.int32)], 1)
    mask = np.ascontiguousarray(np.dstack([ring, hole]))
    before = np.argwhere(mask[..., 1] > 0).mean(axis=0)[::-1]              # (x, y) centroid

    _, out_gates, out_mask = augment_frame_with_masks(
        img, [g], cfg, np.random.default_rng(7), mask=mask)
    g2 = [x for x in out_gates if x.gate_id == g.gate_id][0]
    assert out_mask is not None and out_mask.shape == mask.shape
    assert out_mask[..., 1].any(), "the warp emptied the mask"
    after = np.argwhere(out_mask[..., 1] > 0).mean(axis=0)[::-1]
    assert float(np.linalg.norm(after - before)) > 1.0, \
        "geometric_p=1.0 did not actually warp anything -- the test would be vacuous"

    # The warped mask must agree with the WARPED keypoints. Compare against a fresh rasterisation
    # of them rather than centroids: a warp routinely pushes corners off-frame, and a clipped
    # polygon's centroid is not its quad's centroid.
    expect = np.zeros_like(hole)
    cv2.fillPoly(expect, [g2.keypoints_px.round().astype(np.int32)], 1)
    got = out_mask[..., 1] > 0
    iou = float((got & (expect > 0)).sum()) / float(max((got | (expect > 0)).sum(), 1))
    assert iou > 0.9, f"mask and keypoints disagree after the warp (IoU {iou:.3f})"


def test_augment_frame_still_returns_two_values():
    """Back-compat: the mask arm is additive; the frozen 2-tuple caller must be untouched."""
    gates = load_course_gates()
    fs = _frame_with_label(gates)
    cfg = AugmentConfig(enable=True, photometric_p=1.0, lighting_p=0.0,
                        brightness_p=0.0, geometric_p=0.0)
    img = np.full((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3), 120, np.uint8)
    out = augment_frame(img, fs.gates, cfg, np.random.default_rng(0))
    assert isinstance(out, tuple) and len(out) == 2


def test_procedural_backend_falls_back_to_flat_quads_LOUDLY(tmp_path, capsys):
    """The procedural backend has no 3D scene, so it cannot measure a silhouette. It must say so and
    stamp the target on disk -- a run that quietly emits a DIFFERENT target than the Blender run
    would contaminate any merged dataset with no trace."""
    from racer.vision.blender_gen.dataset import SEG_SOURCE_FLAT_QUAD

    preset = load_preset("vq1_faithful")
    generate_dataset(tmp_path, preset, ProceduralBackend(), n_train=3, n_val=1, seed=0,
                     emit_seg=True)
    out = capsys.readouterr().out
    assert "WARNING" in out and "FLAT" in out
    assert (tmp_path / "seg" / "SOURCE.txt").read_text().startswith(SEG_SOURCE_FLAT_QUAD)
    assert list((tmp_path / "seg" / "train").glob("*.txt")), "fallback must still emit labels"


def test_silhouette_backend_target_comes_from_the_masks_not_the_quads(tmp_path, capsys):
    """A backend that CAN measure silhouettes must have them used verbatim. The fake below returns a
    mask deliberately unlike the projected quads (a plain rectangle), so if the writer quietly fell
    back to seg_rows_from_corners the polygon would not match."""
    import cv2
    from racer.vision.blender_gen.dataset import SEG_SOURCE_SILHOUETTE

    RING = (100, 40, 300, 240)      # x0,y0,x1,y1 -- nothing like a projected gate
    HOLE = (150, 90, 250, 190)

    class FakeSilhouetteBackend(ProceduralBackend):
        emit_silhouettes = False

        def __init__(self):
            self._sil = {}

        def render(self, frame, preset, rng):
            img = ProceduralBackend.render(self, frame, preset, rng)
            self._sil = {}
            for gr in frame.gates:
                if not gr.visible:
                    continue
                ring = np.zeros((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH), bool)
                ring[RING[1]:RING[3], RING[0]:RING[2]] = True
                ring[HOLE[1]:HOLE[3], HOLE[0]:HOLE[2]] = False
                opening = np.zeros_like(ring)
                opening[HOLE[1]:HOLE[3], HOLE[0]:HOLE[2]] = True
                self._sil[int(gr.gate_id)] = (ring, opening)
                break                                    # one gate is enough for the identity check
            return img

        def gate_silhouettes(self):
            return self._sil

    preset = load_preset("vq1_faithful")
    generate_dataset(tmp_path, preset, FakeSilhouetteBackend(), n_train=4, n_val=1, seed=0,
                     emit_seg=True, emit_masks=True)
    assert (tmp_path / "seg" / "SOURCE.txt").read_text().startswith(SEG_SOURCE_SILHOUETTE)
    assert "WARNING" not in capsys.readouterr().out

    texts = [p.read_text() for p in sorted((tmp_path / "seg" / "train").glob("*.txt"))]
    hits = 0
    for t in texts:
        for line in t.splitlines():
            f = line.split()
            if int(f[0]) != 0:
                continue
            poly = np.array([float(v) for v in f[1:]]).reshape(-1, 2) * [contract.IMAGE_WIDTH,
                                                                        contract.IMAGE_HEIGHT]
            if abs(poly[:, 0].min() - RING[0]) < 2 and abs(poly[:, 1].max() - (RING[3] - 1)) < 2:
                hits += 1
    assert hits >= 1, f"no class-0 polygon matched the injected silhouette:\n{texts}"

    # and the instance raster is the silhouette too, not the flat annulus
    m = cv2.imread(str(tmp_path / "masks" / "train" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert m[(RING[1] + HOLE[1]) // 2, (RING[0] + RING[2]) // 2] > 0     # on the injected ring
    assert m[(HOLE[1] + HOLE[3]) // 2, (HOLE[0] + HOLE[2]) // 2] == 0    # inside the injected hole


# ==========================================================================================
# OCCLUSION: which gates keep a label once something stands in FRONT of them
# ==========================================================================================
def _gate_at(z_m: float, gate_id: int = 0) -> GateRender:
    """A head-on GateRender at ``z_m`` metres, accepted by the on-screen-extent rule."""
    R, t = np.eye(3), np.array([0.0, 0.0, float(z_m)])
    inner = project_gate_corners(R, t, GATE_INNER_SIZE_M)
    outer = project_gate_corners(R, t, contract.GATE_OUTER_SIZE_M)
    return GateRender(gate_id=gate_id, R_cam_gate=R, t_cam_gate=t, keypoints_px=inner,
                      outer_px=outer, bbox_xywh=geometry._bbox(outer),
                      visibility=np.array([2, 2, 2, 2]), visible=True)


def _mask_of_area(n_px: int) -> np.ndarray:
    """A boolean silhouette holding exactly ``n_px`` set pixels."""
    m = np.zeros(contract.IMAGE_HEIGHT * contract.IMAGE_WIDTH, bool)
    m[:n_px] = True
    return m.reshape(contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH)


def test_occlusion_floor_is_the_seg_sliver_floor_by_import():
    """The pose-label occlusion floor and the seg polygon sliver floor are ONE number. If they ever
    drift, a gate can keep a pose row while the seg path refuses to draw it (or the reverse) -- two
    answers to the same question, which is precisely how this rule ended up derived in four places."""
    from racer.vision import seg_labels

    assert geometry.MIN_UNOCCLUDED_AREA_PX == seg_labels.MIN_VISIBLE_AREA_PX
    # ...and it is NOT the on-screen-extent floor: those measure different things (silhouette
    # pixels vs bounding-box area) and forcing them equal would drop small distant gates.
    assert geometry.MIN_UNOCCLUDED_AREA_PX < geometry.MIN_LABEL_AREA_PX


def test_occlusion_drop_is_measured_area_not_a_corner_count():
    """REGRESSION (2026-07-22, the FOURTH site). bpy_photoreal.occlude_blocked_keypoints ended with
    ``if (visibility == V_VIS).sum() < 3: gr.visible = False`` -- the retired ">=3 in-frame corners"
    rule wearing an occlusion costume. It counted OFF-FRAME corners too, so it re-killed the very
    crops the area rule exists to rescue, and it ran before augment split labelled from unlabelled,
    so the gate could never come back: rendered into the image, handed to training as background.

    A gate with ZERO corners still V_VIS but a large visible silhouette must keep its label."""
    gr = _gate_at(4.0)
    gr.visibility = np.array([contract.V_OFF, contract.V_OFF, contract.V_OCC, contract.V_OCC])
    census = geometry.apply_measured_occlusion([gr], [_mask_of_area(5000)])
    assert gr.visible, "no corner is V_VIS but 5000 px of gate are on screen -- that is a label"
    assert census == {"labelled": 1, "hidden": 0}


def test_fully_hidden_gate_loses_its_label():
    """The OPPOSITE poisoning, and the reason the drop was not simply deleted. A gate completely
    behind a pillar, a wall or a nearer gate's frame is not in the image at all; labelling it would
    teach the detector to hallucinate a gate through solid objects."""
    hidden, half, sliver = _gate_at(30.0, 0), _gate_at(6.0, 1), _gate_at(25.0, 2)
    census = geometry.apply_measured_occlusion(
        [hidden, half, sliver],
        [_mask_of_area(0),            # entirely behind something
         _mask_of_area(900),          # a pillar takes most of it, plenty still visible
         _mask_of_area(20)],          # a few pixels peeking past an occluder
    )
    assert not hidden.visible, "a gate with no visible pixels must not be labelled"
    assert half.visible, "a half-hidden gate is still a gate"
    assert not sliver.visible, "a 20 px sliver is below one YOLO cell of evidence"
    assert census == {"labelled": 1, "hidden": 2}


def test_the_threshold_sits_in_the_measured_empty_band():
    """The threshold was chosen inside a measured GAP, not tuned on a percentile: over 80 rendered
    frames the occluded gates landed at 0-45 px and the smallest legitimately visible gate at 100 px.
    Anything in 46..95 gives the same verdict, so pin that the boundary behaves as advertised."""
    for area, expect in ((45, False), (63, False), (64, True), (96, True), (100, True)):
        g = _gate_at(20.0)
        geometry.apply_measured_occlusion([g], [_mask_of_area(area)])
        assert g.visible is expect, f"{area} px silhouette -> visible={g.visible}, wanted {expect}"


def test_missing_measurement_keeps_the_label_and_is_not_treated_as_occlusion():
    """No measurement is not evidence of occlusion. If the id pass fails, the extent rule has still
    established the gate is on screen; silently dropping it would re-create the background poisoning
    on every frame of a run whose id pass broke. (The backend shouts about it separately.)"""
    gr = _gate_at(8.0)
    census = geometry.apply_measured_occlusion([gr], [None])
    assert gr.visible and census == {"labelled": 1, "hidden": 0}


def test_occlude_blocked_keypoints_never_decides_whole_gate_visibility():
    """bpy_photoreal imports bpy, so no laptop test can call it -- and that is exactly where the
    fourth copy of this rule hid for months. Parse the source instead: the raycast may write
    per-corner ``visibility`` (real information the pose loss uses) but must never assign
    ``gr.visible``, which is the whole-gate label decision and belongs with the measurement."""
    import ast
    from pathlib import Path

    src = (Path(geometry.__file__).parent / "bpy_photoreal.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "occlude_blocked_keypoints")
    # Every attribute written by an assignment target, including through a subscript
    # (``gr.visibility[c] = V_OCC`` is a Subscript wrapping the Attribute).
    assigned = set()
    for n in ast.walk(fn):
        targets = (n.targets if isinstance(n, ast.Assign)
                   else [n.target] if isinstance(n, ast.AugAssign) else [])
        for t in targets:
            assigned |= {sub.attr for sub in ast.walk(t) if isinstance(sub, ast.Attribute)}
    assert "visible" not in assigned, (
        "occlude_blocked_keypoints is dropping whole gates again -- that decision needs the measured "
        "silhouette (backends/blender.py), not four rays through four corners")
    assert "visibility" in assigned, "the per-corner V_OCC marking is real information; keep it"


def test_disjoint_batches_separates_overlapping_openings():
    """The opening PROXY is solid, so two proxies in one id render occlude each other. The classic
    down-course racing shot -- a far gate framed inside a near gate's opening -- is exactly that
    case, and batching them together would return the far opening EMPTY. Pure-python, so it is
    testable without Blender even though it lives in a bpy leaf."""
    from racer.vision.blender_gen.bpy_idmask import disjoint_batches

    near = np.array([[100.0, 100.0], [400.0, 100.0], [400.0, 300.0], [100.0, 300.0]])
    far_inside = np.array([[230.0, 180.0], [270.0, 180.0], [270.0, 210.0], [230.0, 210.0]])
    elsewhere = np.array([[500.0, 20.0], [600.0, 20.0], [600.0, 80.0], [500.0, 80.0]])

    batches = disjoint_batches([near, far_inside, elsewhere])
    where = {i: b for b, ids in enumerate(batches) for i in ids}
    assert where[0] != where[1], "a gate seen through another gate's opening must render alone"
    assert where[0] == where[2], "non-overlapping openings should still share one render"
    assert sorted(i for b in batches for i in b) == [0, 1, 2], "every opening must be rendered once"

    assert disjoint_batches([near]) == [[0]]
    assert disjoint_batches([]) == []


# ------------------------------------------------------------------------------------------------
# vq2_dark_red: the appearance-matched preset + the printed-signage decal (bpy_signage)
#
# The domain gap these cover was MEASURED, not assumed: an M+1 model trained on the old synthetic
# data and validated on real hand-labelled frames peaked at epoch 10 (pose mAP50 0.503) and decayed
# to 0.282 by epoch 100, and background median gray was 104 synthetic against 37 real.
# ------------------------------------------------------------------------------------------------
def test_dark_red_presets_are_centred_on_the_real_venue():
    """The two new presets must actually encode the measured real-VQ2 look, not merely load.

    Each assertion below is a number someone could quietly regress while 'tidying' a preset, and
    every one of them was calibrated against the real-frame statistics.
    """
    from racer.vision.blender_gen.config import load_preset

    for name in ("vq2_dark_red", "vq2_dark_red_partial"):
        ap = load_preset(name).appearance
        rc = load_preset(name).render
        # DARK. The old presets left this at the (0.2, 2.0) default and rendered a lit room.
        assert ap.ambient_strength_range[1] <= 0.3, f"{name}: world too bright to be a dark hangar"
        # SELF-LIT. Real gates glow hard enough to light the floor; 0.25-0.5 does not read at all
        # against a dark world.
        assert ap.gate_emission_range[0] >= 0.5, f"{name}: gate is not emissive enough to glow"
        # RED STAYS RED. vq1_partial's 0.1 (+/-18 deg) plus bright lighting washed it to salmon.
        assert ap.gate_hue_jitter <= 0.04, f"{name}: hue jitter too wide, red will drift"
        # PRINTED. A bare gate is the texture the detector overfits to.
        assert ap.gate_signage_prob >= 0.5, f"{name}: gates would render mostly bare"
        # CLIPPING, not film. AgX cannot output saturation 205 AND value 206 at once; real does.
        assert rc.view_transform == "Standard", f"{name}: film tonemap desaturates the glowing gate"
        # CENTRED BUT NOT COLLAPSED -- every randomization axis must still have width.
        assert ap.ambient_strength_range[0] < ap.ambient_strength_range[1]
        assert ap.gate_emission_range[0] < ap.gate_emission_range[1]
        assert ap.exposure_range[0] < ap.exposure_range[1]
        assert len(ap.hdri_include) >= 3, f"{name}: too few environments left to vary over"


def test_legacy_presets_keep_the_old_appearance_defaults():
    """The new appearance keys must be strictly OPT-IN: any preset that does not name them has to
    keep byte-identical behaviour, or this change silently re-renders every existing arm."""
    from racer.vision.blender_gen.bpy_signage import wants_signage_material
    from racer.vision.blender_gen.config import load_preset

    for name in ("vq1_faithful", "vq1_partial", "appearance_broad", "terminal_approach",
                 "long_range", "hard_visual", "negatives"):
        p = load_preset(name)
        assert p.appearance.gate_signage_prob == 0.0, f"{name} unexpectedly opted into signage"
        assert p.appearance.gate_hue_offset == 0.0, f"{name} unexpectedly opted into a hue shift"
        assert p.appearance.hdri_include == (), f"{name} unexpectedly filtered its env maps"
        assert p.render.view_transform == "AgX", f"{name} unexpectedly changed tonemap"
        assert not wants_signage_material(p.appearance), f"{name} must keep the plain material path"

    # The photoreal path used to HARDCODE the HDRI strength to rng.uniform(0.7, 1.3) and ignore
    # ambient_strength_range entirely. Making the key live is the fix, but the three legacy
    # photoreal arms must keep the appearance distribution their on-disk datasets were rendered
    # with -- so the ex-hardcoded constant MOVED into those presets rather than being replaced by
    # the (0.2, 2.0) default. Drop these pins and those arms silently re-render nearly twice as
    # bright at the top of the range, with no diff anywhere to say why the pixels moved.
    for name in ("vq1_faithful", "vq1_partial", "appearance_broad"):
        ap = load_preset(name).appearance
        assert ap.photoreal is True, f"{name} is expected to be a photoreal preset"
        assert ap.ambient_strength_range == (0.7, 1.3), (
            f"{name} must pin the ex-hardcoded HDRI strength to stay reproducible")


def test_signage_new_keys_are_validated():
    from racer.vision.blender_gen.config import preset_from_dict

    base = {"name": "t", "appearance": {}}
    for bad in ({"gate_signage_prob": 1.5}, {"gate_hue_offset": 0.9}, {"hdri_include": [""]}):
        with pytest.raises(ValueError):
            preset_from_dict({**base, "appearance": bad})
    # and the good values round-trip
    p = preset_from_dict({**base, "appearance": {"gate_signage_prob": 0.9, "gate_hue_offset": 0.011,
                                                 "hdri_include": ["hangar"]}})
    assert p.appearance.hdri_include == ("hangar",)


def test_signage_decal_lands_only_on_the_gate_ring():
    """Ink must never sit where the gate has a HOLE.

    The decal is mapped through OBJECT coordinates over the gate's outer square, so the opening
    occupies the central (1 - INNER/OUTER)/2 .. 1 - that band. Ink inside it is invisible at best,
    and a sign that the layout constants drifted away from the frozen gate geometry at worst.
    """
    from racer.vision.blender_gen.bpy_signage import BAND, build_decal_alpha

    assert BAND == pytest.approx(
        (1.0 - contract.GATE_INNER_SIZE_M / contract.GATE_OUTER_SIZE_M) / 2.0)
    n = 512
    lo, hi = int(BAND * n) + 3, int((1.0 - BAND) * n) - 3     # 3 px in from the opening edge
    for seed in range(6):
        a = build_decal_alpha(seed, n)
        assert a.shape == (n, n)
        assert float(a[lo:hi, lo:hi].max()) == 0.0, f"seed {seed}: ink inside the gate OPENING"
        assert 0.01 < float(a.mean()) < 0.35, f"seed {seed}: coverage {a.mean():.3f} implausible"


def test_signage_decals_vary_between_gates():
    """Eight identical atlases would be a single memorisable texture -- the failure this replaces."""
    from racer.vision.blender_gen.bpy_signage import build_decal_alpha

    sigs = {build_decal_alpha(s, 256).tobytes() for s in range(8)}
    assert len(sigs) == 8, "signage atlases are not varying with the seed"


def test_signage_marquee_sits_on_the_top_bar():
    """The AI-GP wordmark is the most recognisable marking on the real gate; it belongs on the TOP
    bar. This also pins the image-space orientation -- if the layout ever flips, the glyphs render
    upside down in Blender and nothing else in the suite would catch it."""
    from racer.vision.blender_gen.bpy_signage import BAND, build_decal_alpha

    n = 512
    top = int(BAND * n)
    ink_top = sum(float(build_decal_alpha(s, n)[:top, :].sum()) for s in range(6))
    ink_bottom = sum(float(build_decal_alpha(s, n)[n - top:, :].sum()) for s in range(6))
    assert ink_top > ink_bottom, "top bar should carry the heaviest marking (the marquee)"


def test_gate_hue_offset_shifts_the_anchor_and_defaults_to_a_no_op():
    """gate_hue_offset exists because the contract-FROZEN anchor (255, 50, 0) sits at hue 11.8 deg
    while gate pixels over 400 real frames average 17.8 deg. Zero must reproduce the anchor exactly
    so no existing preset moves."""
    import colorsys

    from racer.vision.blender_gen.bpy_signage import gate_base_rgb_linear

    class _Ap:
        gate_hue_jitter = 0.0
        gate_hue_offset = 0.0
        gate_sat_range = (1.0, 1.0)
        gate_val_range = (1.0, 1.0)

    rng = np.random.default_rng(0)
    anchor_h = colorsys.rgb_to_hsv(*[c / 255.0 for c in contract.VQ1_GATE_RED_RGB])[0]

    def hue_of(ap):
        lin = gate_base_rgb_linear(rng, ap)
        srgb = [1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308 else c * 12.92 for c in lin]
        return colorsys.rgb_to_hsv(*srgb)[0]

    assert hue_of(_Ap()) == pytest.approx(anchor_h, abs=1e-6)

    shifted = _Ap()
    shifted.gate_hue_offset = 0.011
    assert hue_of(shifted) == pytest.approx(anchor_h + 0.011, abs=1e-6)
