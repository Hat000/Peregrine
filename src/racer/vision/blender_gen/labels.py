"""YOLO-pose label writing for the photoreal dataset.

Byte-for-byte the same row format as ``racer.vision.synthetic.to_yolo_pose_label`` (the
detector is trained across both generators, so the contract MUST be identical): one
ultralytics pose row ``class cx cy w h (x y v)*4`` normalized to [0,1]. Off-frame corners are
clamped into [0,1] with v=0; occluded corners carry v=1; visible corners v=2. A round-trip
test asserts this matches the synthetic writer on shared inputs.
"""
from __future__ import annotations

import numpy as np

from .contract import IMAGE_HEIGHT, IMAGE_WIDTH
from .geometry import GateRender


def to_yolo_pose_row(
    keypoints_px: np.ndarray, visibility: np.ndarray, bbox_xywh: np.ndarray, class_id: int = 0
) -> str:
    """One normalized YOLO-pose row. Identical formatting to synthetic.to_yolo_pose_label."""
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


def gate_render_to_row(gr: GateRender, class_id: int = 0) -> str | None:
    """A label row for one GateRender, or None if it is not a usable (labelled) gate."""
    if not gr.visible:
        return None
    return to_yolo_pose_row(gr.keypoints_px, gr.visibility, gr.bbox_xywh, class_id)


def frame_label_rows(labeled_gates, class_id: int = 0) -> list[str]:
    """All label rows for one frame's labelled gates (one row per gate)."""
    rows = [gate_render_to_row(gr, class_id) for gr in labeled_gates]
    return [r for r in rows if r is not None]
