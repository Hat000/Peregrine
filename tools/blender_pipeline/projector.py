"""Project a gate's 4 inner corners to pixel labels, with per-corner visibility/clip flags.

Thin layer over the CANONICAL projector ``racer.vision.gate_pose.project_gate_corners`` (which
itself uses ``racer.frames.CAMERA_INTRINSICS_K``): we do NOT fork the projection math. Given a
:class:`~camera_sampler.CameraPose` (gate-in-camera optical pose), we emit:

  * ``corners_px`` -- (4,2) inner-square corners in IPPE_SQUARE order (0=LL,1=LR,2=UR,3=UL),
                      the exact order ``gate_pose`` / the detector / ``synthetic.py`` use.
  * ``visibility`` -- (4,) per-corner flag {2 visible, 0 off-frame} (matching synthetic V_VIS/V_OFF;
                      we don't model an occluder here so V_OCC=1 is not produced by the projector).
  * ``bbox_xywh``  -- pixel bbox spanning the 4 corners (clamped to the image), for the YOLO row.

This is the ground-truth label source. The mock renderer draws to MATCH these corners; the
round-trip test re-detects them and checks agreement -- proving the labeling geometry, the mock
renderer, and the detector all share ONE corner convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racer.frames import IMAGE_HEIGHT, IMAGE_WIDTH
from racer.vision.gate_pose import GATE_INNER_SIZE_M, project_gate_corners
from racer.vision.synthetic import V_OFF, V_VIS

N_CORNERS = 4


@dataclass(frozen=True)
class ProjectedGate:
    """Projected gate label for one sample."""

    corners_px: np.ndarray            # (4,2) IPPE_SQUARE order
    visibility: np.ndarray            # (4,) V_VIS / V_OFF
    bbox_xywh: np.ndarray             # (4,) x,y,w,h pixels (clamped to image)
    in_front: bool                    # all 4 corners had optical Z > 0
    n_visible: int                    # corners with V_VIS (in-frame)

    @property
    def fully_visible(self) -> bool:
        return self.in_front and self.n_visible == N_CORNERS


def _visibility_flags(corners_px: np.ndarray) -> np.ndarray:
    """Per-corner V_VIS if inside the image rectangle, else V_OFF."""
    x, y = corners_px[:, 0], corners_px[:, 1]
    inside = (x >= 0) & (x <= IMAGE_WIDTH - 1) & (y >= 0) & (y <= IMAGE_HEIGHT - 1)
    return np.where(inside, V_VIS, V_OFF).astype(int)


def _bbox_xywh(corners_px: np.ndarray) -> np.ndarray:
    x0 = float(np.clip(corners_px[:, 0].min(), 0, IMAGE_WIDTH - 1))
    y0 = float(np.clip(corners_px[:, 1].min(), 0, IMAGE_HEIGHT - 1))
    x1 = float(np.clip(corners_px[:, 0].max(), 0, IMAGE_WIDTH - 1))
    y1 = float(np.clip(corners_px[:, 1].max(), 0, IMAGE_HEIGHT - 1))
    return np.array([x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)], dtype=np.float64)


def project_gate(R_cam_gate: np.ndarray, t_cam_gate: np.ndarray,
                 inner_size_m: float = GATE_INNER_SIZE_M) -> ProjectedGate | None:
    """Project the gate to pixel corners + flags. Returns None if any corner is at/behind the
    camera (optical Z <= 0) -- an unusable pose (no label). Reuses the canonical projector."""
    try:
        corners = project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m)
    except ValueError:
        return None  # corner(s) behind the camera
    corners = np.asarray(corners, dtype=np.float64)
    if not np.isfinite(corners).all():
        return None
    vis = _visibility_flags(corners)
    return ProjectedGate(
        corners_px=corners,
        visibility=vis,
        bbox_xywh=_bbox_xywh(corners),
        in_front=True,
        n_visible=int((vis == V_VIS).sum()),
    )
