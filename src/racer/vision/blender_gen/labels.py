"""YOLO-pose label writing for the photoreal dataset (8 keypoints).

One ultralytics pose row ``class cx cy w h (x y v)*8`` normalized to [0,1], where the 8 keypoints
are the 4 INNER-square corners (0..3, the PnP gate-opening keypoints) followed by the 4 OUTER-square
corners (4..7, the gate frame). Off-frame corners are clamped into [0,1] with v=0; occluded corners
carry v=1; visible corners v=2. NOTE: this 8-keypoint scheme is a VQ2 superset of the 4-keypoint
``racer.vision.synthetic.to_yolo_pose_label`` -- the first 4 keypoints (+visibility) are byte-identical
to that 4-corner writer, so 4-corner PnP still parses the inner block directly.
"""
from __future__ import annotations

import numpy as np

from .contract import IMAGE_HEIGHT, IMAGE_WIDTH, V_OFF, V_VIS
from .geometry import GateRender


def to_yolo_pose_row(
    keypoints_px: np.ndarray, visibility: np.ndarray, bbox_xywh: np.ndarray, class_id: int = 0
) -> str:
    """One normalized YOLO-pose row for ``len(keypoints_px)`` keypoints (8 for VQ2: inner then outer)."""
    x, y, w, h = (float(v) for v in bbox_xywh)
    fields: list = [
        class_id,
        (x + w / 2) / IMAGE_WIDTH, (y + h / 2) / IMAGE_HEIGHT,
        w / IMAGE_WIDTH, h / IMAGE_HEIGHT,
    ]
    for (px, py), v in zip(np.asarray(keypoints_px, dtype=float), np.asarray(visibility)):
        nx = min(max(float(px) / IMAGE_WIDTH, 0.0), 1.0)
        ny = min(max(float(py) / IMAGE_HEIGHT, 0.0), 1.0)
        fields += [nx, ny, int(v)]
    return " ".join(f"{vv:.6g}" if isinstance(vv, float) else str(vv) for vv in fields)


def _outer_visibility(gr: GateRender) -> np.ndarray:
    """Per-corner visibility for the OUTER keypoints; in-frame V_VIS/V_OFF if not precomputed."""
    if gr.outer_visibility is not None:
        return np.asarray(gr.outer_visibility)
    vis = np.full(4, V_VIS, dtype=int)
    for c in range(4):
        x, y = float(gr.outer_px[c, 0]), float(gr.outer_px[c, 1])
        vis[c] = V_VIS if (0.0 <= x <= IMAGE_WIDTH - 1 and 0.0 <= y <= IMAGE_HEIGHT - 1) else V_OFF
    return vis


def gate_render_to_row(gr: GateRender, class_id: int = 0) -> str | None:
    """A label row for one GateRender (8 keypoints: inner 0..3 then outer 4..7), or None if the gate
    is not a usable (labelled) positive."""
    if not gr.visible:
        return None
    kpts = np.concatenate([np.asarray(gr.keypoints_px, dtype=float),
                           np.asarray(gr.outer_px, dtype=float)], axis=0)          # (8,2)
    vis = np.concatenate([np.asarray(gr.visibility), _outer_visibility(gr)], axis=0)  # (8,)
    return to_yolo_pose_row(kpts, vis, gr.bbox_xywh, class_id)


def frame_label_rows(labeled_gates, class_id: int = 0) -> list[str]:
    """All label rows for one frame's labelled gates (one row per gate)."""
    rows = [gate_render_to_row(gr, class_id) for gr in labeled_gates]
    return [r for r in rows if r is not None]
