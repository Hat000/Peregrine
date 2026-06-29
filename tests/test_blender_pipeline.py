"""Tests for the VQ2 Blender data-generation harness (tools/blender_pipeline).

Blender-FREE: runs on the laptop venv with only racer.* + numpy + OpenCV. Covers:
  * projection correctness (known pose -> known pixels on a synthetic head-on case),
  * the ROUND-TRIP (sample -> project labels -> mock-render -> red_glow_detector -> corners
    match the labels within tolerance) -- the core correctness proof,
  * the YOLO-pose dataset-format writer (rows + data.yaml + layout),
  * sampler bounds (ranges, in-frame, far/oblique coverage),
  * the appearance loader.

The harness lives under tools/ (not importable as racer.*), so we add it to sys.path here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from blender_pipeline.appearance import AppearanceParams, load_appearance  # noqa: E402
from blender_pipeline.camera_sampler import (  # noqa: E402
    CameraPose,
    SamplerConfig,
    corners_in_frame,
    sample_pose,
    sample_poses,
)
from blender_pipeline.dataset import DATA_YAML, write_dataset, yolo_pose_row  # noqa: E402
from blender_pipeline.projector import project_gate  # noqa: E402
from blender_pipeline.renderer import MockRenderer  # noqa: E402

from racer.frames import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from racer.vision.gate_pose import (  # noqa: E402
    GATE_INNER_SIZE_M,
    estimate_gate_pose,
    project_gate_corners,
)
from racer.vision.red_glow_detector import detect_red_glow_candidates  # noqa: E402
from racer.vision.synthetic import V_OFF, V_VIS  # noqa: E402


# --------------------------------------------------------------------------- projection
def test_projection_headon_known_pixels():
    """A head-on gate at range R: corners are symmetric about the principal point, square,
    with a pixel half-width = fx * (size/2) / R. Validates we reuse the canonical projector
    with the right convention (no fork)."""
    R = 10.0
    Rt = np.eye(3)
    t = np.array([0.0, 0.0, R])
    proj = project_gate(Rt, t)
    assert proj is not None and proj.fully_visible
    c = proj.corners_px
    half = 320.0 * (GATE_INNER_SIZE_M / 2.0) / R  # fx * (s/2)/R = 320*0.75/10 = 24 px
    cx, cy = IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0
    # IPPE_SQUARE order 0=LL,1=LR,2=UR,3=UL with gate Y DOWN -> image:
    expect = np.array([
        [cx - half, cy + half],  # 0 lower-left
        [cx + half, cy + half],  # 1 lower-right
        [cx + half, cy - half],  # 2 upper-right
        [cx - half, cy - half],  # 3 upper-left
    ])
    assert np.allclose(c, expect, atol=1e-6), f"{c}\n vs \n{expect}"
    # matches the canonical projector exactly (defence: no convention drift)
    assert np.allclose(c, project_gate_corners(Rt, t), atol=1e-9)


def test_projection_behind_camera_returns_none():
    proj = project_gate(np.eye(3), np.array([0.0, 0.0, -5.0]))
    assert proj is None


def test_projection_visibility_flags_offframe():
    """A gate centred but huge/near so corners fall off-frame -> those corners flagged V_OFF."""
    proj = project_gate(np.eye(3), np.array([0.0, 0.0, 1.2]))  # very near -> corners blow past edges
    assert proj is not None and proj.in_front
    assert np.any(proj.visibility == V_OFF)


# --------------------------------------------------------------------------- sampler
def test_sampler_pose_is_valid_and_in_range():
    cfg = SamplerConfig()
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = sample_pose(rng, cfg)
        assert cfg.range_min_m <= p.range_m <= cfg.range_max_m + 1e-6
        # R is a proper rotation
        assert np.allclose(p.R_cam_gate @ p.R_cam_gate.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(p.R_cam_gate), 1.0, atol=1e-9)
        assert abs(p.gate_roll_deg) <= cfg.roll_max_deg + 1e-9


def test_sampler_visible_only_all_in_frame():
    poses = sample_poses(120, seed=3, visible_only=True)
    assert len(poses) >= 100  # most draws succeed
    for p in poses:
        assert corners_in_frame(p, margin_px=1.0)
        proj = project_gate(p.R_cam_gate, p.t_cam_gate)
        assert proj is not None and proj.fully_visible


def test_sampler_covers_far_and_oblique():
    """The sampler must DELIBERATELY reach the detector's weak spots (far + oblique)."""
    poses = sample_poses(400, seed=7, visible_only=True)
    ranges = np.array([p.range_m for p in poses])
    obliq = np.array([max(abs(p.gate_yaw_deg), abs(p.gate_pitch_deg)) for p in poses])
    assert (ranges > 15.0).mean() > 0.05, "should include far (>15 m) gates"
    assert (obliq > 35.0).mean() > 0.05, "should include strongly oblique gates"
    assert ranges.min() < 6.0, "should include near gates (the detection-critical band)"


def test_sampler_deterministic():
    a = sample_poses(20, seed=42)
    b = sample_poses(20, seed=42)
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert np.allclose(pa.t_cam_gate, pb.t_cam_gate)
        assert np.allclose(pa.R_cam_gate, pb.R_cam_gate)


# --------------------------------------------------------------------------- mock renderer
def test_mock_render_shape_and_darkness():
    params = load_appearance()
    r = MockRenderer(seed=0)
    pose = CameraPose(np.eye(3), np.array([0.0, 0.0, 8.0]), 8.0, 0, 0, 0, 0, 0)
    img = r.render(pose, params)
    assert img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3) and img.dtype == np.uint8
    # low-light: most of the frame dark; a saturated red core present
    gray = img.mean(axis=2)
    assert (gray < 32).mean() > 0.5, "scene should be predominantly dark (VQ2 low-light look)"
    b, g, rr = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    core = (rr >= 250) & ((rr - g) >= 80) & ((rr - b) >= 80)
    assert core.sum() > 100, "a saturated red gate core should be rendered"


# --------------------------------------------------------------------------- ROUND-TRIP (the proof)
def _roundtrip_corner_error(pose: CameraPose, params: AppearanceParams) -> float | None:
    """Project labels -> mock-render -> red_glow_detect -> match detected inner corners to the
    projected labels (Hungarian-free: nearest by canonical order after both are canonically
    ordered). Returns the max per-corner pixel error, or None if the detector found nothing."""
    proj = project_gate(pose.R_cam_gate, pose.t_cam_gate)
    if proj is None or not proj.fully_visible:
        return None
    img = MockRenderer(seed=0, add_noise=False).render(pose, params)
    cands = detect_red_glow_candidates(img)
    if not cands:
        return None
    # pick the candidate whose centroid is nearest the label centroid (the right gate)
    label_ctr = proj.corners_px.mean(axis=0)
    best = min(cands, key=lambda c: np.linalg.norm(c.centroid - label_ctr))
    # both are in canonical [LL,LR,UR,UL] order already -> compare row-wise
    err = np.linalg.norm(best.corners_px - proj.corners_px, axis=1)
    return float(err.max())


def test_roundtrip_headon():
    """The flagship correctness check: a clean head-on gate must round-trip to within a few px.
    Proves the labeling geometry, the mock renderer, and the detector share ONE corner convention
    (a wrong corner order would blow this up by tens of px / a 90-deg rotation)."""
    params = load_appearance()
    pose = CameraPose(np.eye(3), np.array([0.0, 0.0, 8.0]), 8.0, 0, 0, 0, 0, 0)
    err = _roundtrip_corner_error(pose, params)
    assert err is not None, "detector should find the head-on gate"
    assert err < 6.0, f"head-on corner round-trip error {err:.2f}px too large"


def test_roundtrip_population():
    """Round-trip over a population of near/mid sampled poses: the detector should recover the
    labeled corners on a solid majority, and when it does the corners must agree tightly."""
    params = load_appearance()
    # restrict to the near/mid band where the classical detector resolves a clean hole
    cfg = SamplerConfig(range_min_m=4.0, range_max_m=12.0, far_frac=0.0,
                        oblique_frac=0.0, oblique_max_deg=25.0, roll_max_deg=15.0)
    poses = sample_poses(60, seed=11, cfg=cfg, visible_only=True)
    errs = [e for p in poses if (e := _roundtrip_corner_error(p, params)) is not None]
    assert len(errs) >= int(0.6 * len(poses)), (
        f"detector recovered only {len(errs)}/{len(poses)} sampled gates")
    errs = np.array(errs)
    # the corner CONVENTION must be exact: a mis-ordered corner would show as a large-tail error.
    assert np.median(errs) < 8.0, f"median round-trip error {np.median(errs):.2f}px"
    assert (errs < 12.0).mean() > 0.75, f"too many large-error round-trips: {errs}"


def test_roundtrip_pnp_recovers_range():
    """End-to-end: labels -> render -> detect -> estimate_gate_pose recovers the TRUE range.
    Ties the harness to the real PnP the deployed stack runs (not just pixel agreement)."""
    params = load_appearance()
    pose = CameraPose(np.eye(3), np.array([0.0, 0.0, 8.0]), 8.0, 0, 0, 0, 0, 0)
    img = MockRenderer(seed=0, add_noise=False).render(pose, params)
    cands = detect_red_glow_candidates(img)
    assert cands
    from racer.contracts import Frame
    from racer.vision.red_glow_detector import RedGlowGateDetector
    frame = Frame(frame_id=0, sim_time_ns=0, image_bgr=img)
    obs = RedGlowGateDetector().detect(frame)
    assert obs, "red-glow detector should emit an observation"
    gp = estimate_gate_pose(obs[0])
    assert gp is not None
    est_range = float(np.linalg.norm(gp.t_cam_gate))
    assert abs(est_range - 8.0) < 1.5, f"PnP range {est_range:.2f} vs true 8.0"


# --------------------------------------------------------------------------- dataset writer
def test_yolo_pose_row_format():
    proj = project_gate(np.eye(3), np.array([0.0, 0.0, 10.0]))
    assert proj is not None
    row = yolo_pose_row(proj)
    assert row is not None
    parts = row.split()
    # class + 4 bbox + 4*(x,y,v) = 1 + 4 + 12 = 17 fields
    assert len(parts) == 17, f"expected 17 fields, got {len(parts)}: {row}"
    assert parts[0] == "0"  # class id
    # all corner x/y normalized into [0,1]; v in {0,1,2}
    vals = [float(x) for x in parts[1:]]
    for i in range(5, 17, 3):
        nx, ny, v = vals[i - 1], vals[i], vals[i + 1]
        assert 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        assert int(v) in (V_OFF, 1, V_VIS)


def test_write_dataset(tmp_path):
    out = str(tmp_path / "ds")
    r = MockRenderer(seed=0)
    cfg = SamplerConfig(range_min_m=4.0, range_max_m=12.0, far_frac=0.0)
    summary = write_dataset(out, r, n_train=5, n_val=3, seed=1, cfg=cfg)
    assert summary["n_train"] == 5 and summary["n_val"] == 3
    # files exist + layout correct
    assert os.path.exists(os.path.join(out, "data.yaml"))
    for split, n in (("train", 5), ("val", 3)):
        imgs = os.listdir(os.path.join(out, "images", split))
        lbls = os.listdir(os.path.join(out, "labels", split))
        assert len(imgs) == n and len(lbls) == n
    # data.yaml carries the canonical kpt_shape + flip_idx
    yaml = open(os.path.join(out, "data.yaml")).read()
    assert "kpt_shape: [4, 3]" in yaml and "flip_idx: [1, 0, 3, 2]" in yaml
    # a non-empty label round-trips to a valid 17-field row
    any_lbl = open(os.path.join(out, "labels", "train",
                                sorted(os.listdir(os.path.join(out, "labels", "train")))[0])).read()
    if any_lbl.strip():
        assert len(any_lbl.split()) == 17


def test_data_yaml_matches_canonical_keypoint_contract():
    """The harness label contract must match the detector's 4-inner-corner expectation."""
    assert "kpt_shape: [4, 3]" in DATA_YAML
    assert "flip_idx: [1, 0, 3, 2]" in DATA_YAML


# --------------------------------------------------------------------------- appearance
def test_appearance_loads_and_has_recon_values():
    p = load_appearance()
    assert isinstance(p, AppearanceParams)
    # gate emissive core = vivid orange-red near (255, 56, 16)
    assert p.gate_emission_rgb[0] >= 250
    assert p.gate_emission_rgb[1] < 120 and p.gate_emission_rgb[2] < 80
    # scene is low-light (~36/255)
    assert 25.0 <= p.scene_mean_gray <= 50.0
    assert p.core_threshold == 250  # the binding 'use the saturated core' threshold


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_render_deterministic_given_seed(seed):
    params = load_appearance()
    pose = CameraPose(np.eye(3), np.array([0.5, -0.3, 9.0]), 9.0, 0, 0, 5, -5, 3)
    a = MockRenderer(seed=seed).render(pose, params)
    b = MockRenderer(seed=seed).render(pose, params)
    assert np.array_equal(a, b)
