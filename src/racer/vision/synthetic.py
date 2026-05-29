"""Procedural synthetic gate dataset for pre-training a YOLO-pose corner detector.

Renders domain-randomized gate images with PIXEL-PERFECT labels (the 4 inner-square
corners, in the locked canonical order) by reusing ``gate_pose.project_gate_corners``.
The point of this data is to teach appearance-invariant *geometry* (find a square
gate's 4 corners); photoreal realism + course-specific accuracy come later from
fine-tuning on real sim frames (the auto-label flywheel). A Blender renderer can be
swapped in as a higher-fidelity image source behind the same label pipeline.

Labels are written in ultralytics YOLO-pose format. Keypoint order matches
``gate_pose`` exactly: 0=lower-left, 1=lower-right, 2=upper-right, 3=upper-left
(gate frame X-right, Y-down). ``flip_idx=[1,0,3,2]`` keeps horizontal-flip
augmentation correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from racer.frames import CAMERA_INTRINSICS_K, IMAGE_HEIGHT, IMAGE_WIDTH
from racer.vision.gate_pose import GATE_INNER_SIZE_M, project_gate_corners

GATE_OUTER_SIZE_M = 2.7  # spec 3.7 outer boundary
FLIP_IDX = [1, 0, 3, 2]  # horizontal flip swaps left<->right corners


@dataclass
class SyntheticSample:
    image_bgr: np.ndarray      # (H, W, 3) uint8
    keypoints_px: np.ndarray   # (4, 2) inner corners, canonical order
    bbox_xywh: np.ndarray      # (4,) pixel bbox [x, y, w, h] (top-left + size, from outer square)
    visible: bool              # all 4 inner corners in front of and within the image
    R_cam_gate: np.ndarray     # (3, 3) ground-truth pose used to render
    t_cam_gate: np.ndarray     # (3,)


def sample_gate_pose(rng: np.random.Generator, min_dist: float = 2.5, max_dist: float = 14.0):
    """A random gate pose in the camera frame whose centre projects near the image centre."""
    u = rng.uniform(0.3 * IMAGE_WIDTH, 0.7 * IMAGE_WIDTH)
    v = rng.uniform(0.3 * IMAGE_HEIGHT, 0.7 * IMAGE_HEIGHT)
    d = rng.uniform(min_dist, max_dist)
    ray = np.linalg.inv(CAMERA_INTRINSICS_K) @ np.array([u, v, 1.0])
    ray /= ray[2]
    t = d * ray                                   # gate centre at depth ~d through pixel (u, v)
    tilt = Rotation.from_euler(
        "xyz",
        [rng.uniform(-0.35, 0.35), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)],
    )
    return tilt.as_matrix(), t                    # base orientation I = gate facing camera (IPPE native)


def _random_background(rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(0, 130, size=3).astype(np.int16)
    img = np.tile(base, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    grad = np.linspace(rng.uniform(-50, 50), rng.uniform(-50, 50), IMAGE_WIDTH)
    img = np.ascontiguousarray(np.clip(img + grad[None, :, None], 0, 255).astype(np.uint8))
    for _ in range(int(rng.integers(0, 6))):      # clutter / distractors
        p1 = (int(rng.integers(0, IMAGE_WIDTH)), int(rng.integers(0, IMAGE_HEIGHT)))
        p2 = (int(rng.integers(0, IMAGE_WIDTH)), int(rng.integers(0, IMAGE_HEIGHT)))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        cv2.rectangle(img, p1, p2, color, -1)
    return img


def _draw_gate(img: np.ndarray, inner_px: np.ndarray, outer_px: np.ndarray,
               rng: np.random.Generator) -> None:
    """Paint the gate ring (between outer and inner squares); the opening shows background."""
    ring = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(ring, [outer_px.round().astype(np.int32)], 1)
    cv2.fillPoly(ring, [inner_px.round().astype(np.int32)], 0)
    bright = int(rng.integers(0, 3))   # one vivid channel -> a distinct gate colour
    color = np.array(
        [int(rng.integers(120, 256)) if i == bright else int(rng.integers(0, 160)) for i in range(3)],
        dtype=np.int16,
    )
    shade = rng.uniform(0.7, 1.0)
    img[ring == 1] = np.clip(color * shade, 0, 255).astype(np.uint8)


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)     # motion/defocus
    gain = rng.uniform(0.7, 1.3)
    bias = rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        noise = rng.normal(0, rng.uniform(2, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _in_frame(pts: np.ndarray) -> bool:
    return bool(
        (pts[:, 0] >= 0).all() and (pts[:, 0] <= IMAGE_WIDTH).all()
        and (pts[:, 1] >= 0).all() and (pts[:, 1] <= IMAGE_HEIGHT).all()
    )


def render_gate_sample(rng: np.random.Generator, pose=None) -> SyntheticSample:
    R, t = sample_gate_pose(rng) if pose is None else pose
    blank = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    try:
        inner_px = project_gate_corners(R, t, GATE_INNER_SIZE_M)
        outer_px = project_gate_corners(R, t, GATE_OUTER_SIZE_M)
    except ValueError:
        return SyntheticSample(blank, np.zeros((4, 2)), np.zeros(4), False, R, t)

    visible = _in_frame(inner_px)
    img = _random_background(rng)
    _draw_gate(img, inner_px, outer_px, rng)
    img = _augment(img, rng)

    x0, y0 = outer_px.min(axis=0)
    x1, y1 = outer_px.max(axis=0)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(IMAGE_WIDTH), x1), min(float(IMAGE_HEIGHT), y1)
    bbox = np.array([x0, y0, x1 - x0, y1 - y0])
    return SyntheticSample(img, inner_px, bbox, visible, R, t)


def to_yolo_pose_label(sample: SyntheticSample, class_id: int = 0) -> str | None:
    """One ultralytics YOLO-pose row: ``class cx cy w h (x y v)*4`` (normalized). None if not visible."""
    if not sample.visible:
        return None
    x, y, w, h = sample.bbox_xywh
    fields = [class_id, (x + w / 2) / IMAGE_WIDTH, (y + h / 2) / IMAGE_HEIGHT,
              w / IMAGE_WIDTH, h / IMAGE_HEIGHT]
    for px, py in sample.keypoints_px:
        fields += [px / IMAGE_WIDTH, py / IMAGE_HEIGHT, 2]   # v=2: labeled + visible
    return " ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in fields)


DATA_YAML = """\
# Synthetic gate dataset for YOLO-pose (ultralytics).
path: {path}
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [4, 3]   # 4 corners, (x, y, visibility)
flip_idx: [1, 0, 3, 2]   # horizontal flip swaps left<->right corners
"""


def write_dataset(out_dir, n_train: int, n_val: int, seed: int = 0) -> Path:
    """Generate a YOLO-pose dataset (images + labels + data.yaml). Returns the data.yaml path."""
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        made = 0
        while made < n:
            sample = render_gate_sample(rng)
            label = to_yolo_pose_label(sample)
            if label is None:
                continue
            cv2.imwrite(str(img_dir / f"{made:06d}.png"), sample.image_bgr)
            (lbl_dir / f"{made:06d}.txt").write_text(label + "\n")
            made += 1
    yaml_path = out / "data.yaml"
    yaml_path.write_text(DATA_YAML.format(path=str(out.resolve())))
    return yaml_path
