"""Per-frame gate instance masks for the VQ2 dataset.

TWO SOURCES, and they are not equivalent -- read this before using either.

:func:`instance_mask_from_silhouettes` is the REAL one. It packs the per-gate silhouettes the
Blender backend MEASURED with an object-id render pass (:mod:`racer.vision.blender_gen.bpy_idmask`)
into one instance raster. Those pixels are the gate as the camera actually sees it: front face,
back face, and whatever inner side walls the 0.26 m depth exposes at close range and off-axis.

:func:`gate_ring_mask` is the FLAT-QUAD FALLBACK, kept only because the procedural (no-Blender)
backend genuinely draws flat quads -- its ``_draw_ring`` rasterises the identical annulus, so for
that backend the mask and the pixels still agree exactly. For the Blender backend it is now WRONG:
it rasterises the annulus between two projected squares, which is the shape a paper-thin gate would
have. Head-on the error is small; at 2 m and off-axis it omits the side walls entirely and
over-states the opening, i.e. it is worst in precisely the regime the segmentation path exists to
fix. It is NOT retired outright because deleting it would leave the procedural backend silently
emitting nothing, which is a worse failure than emitting a documented approximation -- the dataset
writer prints a loud warning and stamps ``seg/SOURCE.txt`` whenever this path is taken.

Conventions (unchanged, both sources):
  * One single-channel (H, W) uint8 mask per frame. 0 = background; gate instance ``i`` paints its
    pixels with value ``i + 1`` (capped at 255). Threshold ``> 0`` gives a binary gate mask;
    distinct nonzero values give instances.
  * Painted FAR -> NEAR (same z-order as the render), so where two gates overlap the NEARER one
    wins -- matching what the camera sees. That makes the raster lossy for overlapping instances
    (a far gate seen through a near gate's opening loses those pixels to the near gate); the YOLO
    seg polygons are the lossless target and the raster is a debugging/aux artefact.
  * Built from the SAME (post-augment) gates that get a pose row, so masks and keypoints agree.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import IMAGE_HEIGHT, IMAGE_WIDTH
from .geometry import GateRender


def instance_mask_from_silhouettes(
    labeled_gates: list[GateRender], ring_masks: list, h: int = IMAGE_HEIGHT, w: int = IMAGE_WIDTH
) -> np.ndarray:
    """Instance raster (H, W) uint8 from RENDERED per-gate silhouettes.

    ``ring_masks[i]`` is the boolean silhouette of ``labeled_gates[i]`` (None = no mask measured for
    that gate, e.g. the id pass failed; it is skipped rather than back-filled with a guess).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    order = sorted(range(len(labeled_gates)), key=lambda k: -labeled_gates[k].range_m)  # far first
    for paint_value, idx in enumerate(order, start=1):
        m = ring_masks[idx] if idx < len(ring_masks) else None
        if m is None:
            continue
        mask[np.asarray(m).astype(bool)] = min(paint_value, 255)
    return mask


def gate_ring_mask(
    labeled_gates: list[GateRender], h: int = IMAGE_HEIGHT, w: int = IMAGE_WIDTH
) -> np.ndarray:
    """FLAT-QUAD FALLBACK instance mask (see the module docstring -- prefer the rendered silhouette).

    Paints far -> near so the z-order matches the rendered pixels; each gate's annulus is
    ``fillPoly(outer) AND NOT fillPoly(inner)`` -- byte-identical to ``ProceduralBackend._draw_ring``,
    which is why it remains exactly right for that backend and only that backend. Empty (all-zero)
    for a negative frame.
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
