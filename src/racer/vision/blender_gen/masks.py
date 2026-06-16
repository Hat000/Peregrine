"""Gate-ring segmentation masks for the VQ2 dataset (pure-Python, no Blender).

A banked hedge against the keypoint-vs-segmentation question: alongside the YOLO-pose keypoint
labels we can cheaply emit a per-frame instance mask, because the gate-ring polygon is already
KNOWN -- it is the square annulus between the projected OUTER corners (``GateRender.outer_px``)
and the INNER keypoints (``GateRender.keypoints_px``). No extra render pass, no Blender: we just
rasterize the polygons that the labels are already built from, so a mask and the pose label of
the same gate describe the same pixels.

Conventions:
  * One single-channel (H, W) uint8 mask per frame. 0 = background; gate instance ``i`` paints its
    ring pixels with value ``i + 1`` (capped at 255). Threshold ``> 0`` gives a binary gate mask;
    distinct nonzero values give instances.
  * Painted FAR -> NEAR (same z-order as the render), and each gate carves ONLY its own inner hole
    (== ProceduralBackend._draw_ring) -- so a nearer gate's ring overwrites a farther one where the
    structural frames overlap, while a farther gate seen THROUGH a nearer gate's opening is kept.
  * Built from the SAME (labelled) gates that get a pose row, so masks and keypoints stay consistent;
    pass the post-augment gates so a geometric warp is already baked into the corners.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import IMAGE_HEIGHT, IMAGE_WIDTH
from .geometry import GateRender


def gate_ring_mask(
    labeled_gates: list[GateRender], h: int = IMAGE_HEIGHT, w: int = IMAGE_WIDTH
) -> np.ndarray:
    """Instance mask (H, W) uint8 for one frame's labelled gates (0 = background, i+1 per gate).

    Paints far -> near so the z-order matches the rendered pixels; each gate's annulus is
    ``fillPoly(outer) AND NOT fillPoly(inner)``. Empty (all-zero) for a negative frame.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    order = sorted(range(len(labeled_gates)), key=lambda k: -labeled_gates[k].range_m)  # far first
    for paint_value, idx in enumerate(order, start=1):
        gr = labeled_gates[idx]
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(ring, [gr.outer_px.round().astype(np.int32)], 1)
        cv2.fillPoly(ring, [gr.keypoints_px.round().astype(np.int32)], 0)   # carve own opening
        mask[ring == 1] = min(paint_value, 255)
    return mask


def gate_ring_polygons_norm(
    labeled_gates: list[GateRender], h: int = IMAGE_HEIGHT, w: int = IMAGE_WIDTH
) -> list[str]:
    """Optional YOLO-seg rows (one per gate): ``class x1 y1 ... x4 y4`` of the OUTER square,
    normalized to [0,1]. The opening (hole) is NOT representable in single-polygon YOLO-seg, so
    this is the filled-outer approximation; the raster :func:`gate_ring_mask` is the faithful
    annulus. Provided for callers that want a polygon arm; clamped into frame."""
    rows = []
    for gr in labeled_gates:
        pts = gr.outer_px.copy().astype(float)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, w) / w
        pts[:, 1] = np.clip(pts[:, 1], 0.0, h) / h
        coords = " ".join(f"{v:.6g}" for v in pts.reshape(-1))
        rows.append(f"0 {coords}")
    return rows
