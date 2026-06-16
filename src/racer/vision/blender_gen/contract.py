"""FROZEN single-source-of-truth for the VQ2 photoreal dataset.

Every module (pure-Python core AND the bpy render backend) reads its conventions from
HERE, so the on-disk labels, the deployed PnP estimator, and the Blender camera can never
silently disagree. Most values are re-exported from the canonical stack
(``racer.frames`` / ``racer.vision.gate_pose`` / ``racer.vision.synthetic``) rather than
re-typed, so a change there propagates instead of forking.

DATASET CONTRACT (also in handoff/.../DATASET_CONTRACT.md):

  * Class:        0 = "gate" (single class, ``nc=1``).
  * Keypoints:    4 inner-square corners, ``kpt_shape=[4, 3]`` (x, y, visibility).
  * Keypoint ORDER (canonical, == gate_pose IPPE_SQUARE object-point order):
        0 = lower-left, 1 = lower-right, 2 = upper-right, 3 = upper-left
    in the gate frame (X-right, Y-DOWN, Z-through/downrange). MUST match the detector.
  * flip_idx:     [1, 0, 3, 2] (horizontal flip swaps left<->right corners).
  * Visibility:   2 = visible, 1 = occluded-in-frame, 0 = off-frame (clamped into [0,1]).
  * Label row:    ``class cx cy w h (x y v)*4`` normalized to [0,1] (ultralytics YOLO-pose).
  * Image:        640x360 BGR (OpenCV order on disk via cv2.imwrite -> PNG/JPG).
"""
from __future__ import annotations

import numpy as np

# --- camera intrinsics + image size (canonical; do NOT redefine) ----------------------
from racer.frames import (
    CAMERA_INTRINSICS_K,
    CAMERA_PITCH_RAD,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

# --- gate geometry + keypoint convention (canonical) ----------------------------------
from racer.vision.gate_pose import (  # noqa: F401  (re-exported for backends)
    GATE_INNER_SIZE_M,          # 1.5 m inner opening (what PnP uses)
    gate_object_points,         # (4,3) inner-square corners, IPPE_SQUARE order
    project_gate_corners,       # the canonical 3D->2D projector (labels reuse THIS)
)
from racer.vision.synthetic import (  # noqa: F401
    FLIP_IDX,                   # [1, 0, 3, 2]
    V_OCC,
    V_OFF,
    V_VIS,
)

# Outer boundary: the REAL gate is 2.72 m (track_map width_m/height_m); synthetic.py uses a
# rounded 2.7. We use the real value here -- it only affects the bounding box, never the
# keypoints/PnP, so the 0.02 m difference is immaterial. The opaque ring is OUTER..INNER.
GATE_OUTER_SIZE_M = 2.72
# Gate frame depth (spec VADR-TS-002 sec 3.7: 260 mm). The keypoints stay on the gate-frame
# Z=0 plane (planar PnP model); the 3D ring mesh is built symmetric about Z=0 so the rendered
# inner corners land on the projected keypoints. Front/back parallax is <=GATE_DEPTH/2 / range
# (<0.4 deg at >=2 m) -- negligible vs the trained corner sigma; flagged for terminal (<2 m).
GATE_DEPTH_M = 0.26

N_CORNERS = 4

# --- VQ1 gate colour (EXTRACTED from real sim frames) ---------------------------------
# Median of the saturated warm ring across handoff/shadowpc-followups-2026-06-05/task2_frames
# (close 1.8 m, mid 5 m, 10 m): RGB ~ (255, 50, 0), OpenCV-HSV ~ (6, 255, 255). A vivid
# orange-red, NOT pure red. This is the VQ1-faithful BASE; appearance_broad randomizes hue/sat
# widely AROUND it to kill the "gambling-red" overfit. Stored RGB (0..255); convert per engine.
VQ1_GATE_RED_RGB = (255, 50, 0)
VQ1_GATE_RED_RGB_LINEAR = tuple(  # sRGB->linear for Blender Principled BSDF base_color
    (c / 255.0) ** 2.4 if (c / 255.0) > 0.04045 else (c / 255.0) / 12.92
    for c in VQ1_GATE_RED_RGB
)

# --- Blender camera intrinsics recipe (see intrinsics.py for the math + the test) ------
# sensor_fit HORIZONTAL + centered principal point => lens = fx * sensor_width / W reproduces
# K exactly; sensor_width is a free gauge (lens scales with it). 36 mm full-frame -> 18 mm.
BLENDER_SENSOR_WIDTH_MM = 36.0
BLENDER_SENSOR_FIT = "HORIZONTAL"

# --- optical-frame rendering: Blender-world == OpenCV optical frame --------------------
# A Blender camera at identity looks down its local -Z with +Y up. Rolling it 180 deg about
# its X axis (matrix_world = Rx(pi)) makes: image-right = world +X, image-DOWN = world +Y,
# forward = world +Z -- i.e. Blender-world coordinates ARE OpenCV optical coordinates. So we
# place the camera with this matrix at the origin and put every gate at its optical pose
# (R_cam_gate, t_cam_gate); the render then matches project_gate_corners(R_cam_gate,
# t_cam_gate) to sub-pixel. Rx(pi) = diag(1, -1, -1) (a proper rotation, its own inverse).
BLENDER_CAM_ROTATION_X_RAD = float(np.pi)
R_OPTICAL_TO_BLENDER_WORLD = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# --- operational envelope (label-bearing range; matches the deployed co-visibility) ----
RANGE_MIN_M = 2.0
RANGE_MAX_M = 30.0

# --- 8-keypoint scheme: 4 INNER corners (0..3) then 4 OUTER corners (4..7), each LL,LR,UR,UL ----
# Inner corners (0..3) are the PnP keypoints (the gate OPENING); outer corners (4..7) are the gate
# FRAME's outer square. flip_idx swaps left<->right WITHIN each group (inner [1,0,3,2] + outer
# [5,4,7,6] shifted by 4).
N_KEYPOINTS = 8
KEYPOINT_FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6]

# --- YOLO data.yaml (ultralytics pose) -------------------------------------------------
DATA_YAML = """\
# VQ2 photoreal gate dataset for YOLO-pose (ultralytics). Generated by racer.vision.blender_gen.
path: {path}
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [8, 3]                 # 4 inner-square corners then 4 outer-square corners, (x, y, visibility)
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]   # horizontal flip swaps left<->right within inner + outer groups
"""

__all__ = [
    "CAMERA_INTRINSICS_K", "CAMERA_PITCH_RAD", "IMAGE_WIDTH", "IMAGE_HEIGHT",
    "GATE_INNER_SIZE_M", "GATE_OUTER_SIZE_M", "GATE_DEPTH_M", "N_CORNERS",
    "gate_object_points", "project_gate_corners",
    "FLIP_IDX", "V_VIS", "V_OCC", "V_OFF", "N_KEYPOINTS", "KEYPOINT_FLIP_IDX",
    "VQ1_GATE_RED_RGB", "VQ1_GATE_RED_RGB_LINEAR",
    "BLENDER_SENSOR_WIDTH_MM", "BLENDER_SENSOR_FIT",
    "BLENDER_CAM_ROTATION_X_RAD", "R_OPTICAL_TO_BLENDER_WORLD",
    "RANGE_MIN_M", "RANGE_MAX_M", "DATA_YAML",
]
