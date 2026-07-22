"""Additive gate-CENTRE keypoint label -- the 5th keypoint for the "M+1" deploy detector.

M+1 adds ONE directly-regressed keypoint to the four inner corners: the gate-OPENING CENTRE. The
centre is the whole reason the model exists. Measured on real hand labels, 55.7% of gate views have
the centre IN frame while one or more inner corners are cropped OFF-frame, and a further ~11% look
straight THROUGH the gate (centre itself off-frame). That cropped-but-centred regime is exactly
where the corner-only model M is blind, because ultralytics stores an off-frame keypoint clamped to
the image border with v=0 -- so anything that reconstructs the centre from the *stored* corners is
wrong precisely where it matters.

HOW THE SYNTHETIC CENTRE IS EXACT EVEN OFF-FRAME. For a render the gate pose is known, so the true
projected centre is the DIAGONAL INTERSECTION of the projected inner corners: a homography maps
lines to lines and preserves incidence, so the image of the square's centre is where the images of
its two diagonals cross. This is a projective invariant (verified elsewhere to 2.3e-13 px), so it is
exact regardless of where the corners land. The load-bearing fact is that ``GateRender.keypoints_px``
holds the true projected corner pixels BEFORE the [0,1] clamp that ``labels.to_yolo_pose_row``
applies -- so intersecting them recovers the centre exactly where the pose row has already destroyed
the corner coordinates. We must therefore compute the centre from ``keypoints_px`` in the SAME
post-augment GateRender the pose row is written from (see dataset._write_split); a centre computed
from any earlier/clamped copy would be silently misaligned after the albumentations warp.

WHY A SIDECAR, NOT A 9TH KEYPOINT. Emitting the centre as ``center/<split>/<frame>.txt`` (one line
per labelled gate, mirroring seg/ and geom/) leaves the FROZEN 8-keypoint pose contract and the
hand-labeler's byte format untouched. A 9-keypoint superset row would force coordinated edits to
contract.py AND tools/gate_labeler/labelio.py and risk the deploy PnP parse -- the sidecar avoids all
of it and is strictly additive. The M+1 dataset builder then assembles the final 5-keypoint pose
rows (inner 4 + centre) from the pose label + this sidecar.

SIDECAR ROW: ``cx_norm cy_norm v`` -- cx/W, cy/H, %.6g, UNCLAMPED (may lie outside [0,1]); v is 2
when the centre pixel is inside [0,W-1]x[0,H-1], else 0. Unclamped ON PURPOSE: it is lossless (the
project's other geometry sidecars are lossless for the same reason), the overlay tool can draw the
true centre, and the builder re-applies the [0,1] pose-row clamp when it writes the final keypoint --
so nothing downstream regresses while the off-frame ground truth is preserved for evaluation.

NOTE the centre carries only v=2 / v=0 (in-frame / off-frame), never v=1: unlike a corner, a centre
is not a distinguished physical point that another gate can occlude in a way we label -- the frozen
rule for it is purely the in-frame test, per the M+1 brief.
"""
from __future__ import annotations

import numpy as np

# REUSE the canonical diagonal-intersection primitive rather than re-deriving it. gate_lines.
# centre_from_quad IS this geometry (diagonals of the LL,LR,UR,UL quad: 0-2 and 1-3), verified to
# 2.3e-13 px, and this codebase's whole bug history is one geometry rule copied until the copies
# disagreed -- so import it, do not restate it.
from racer.vision.gate_lines import centre_from_quad

from .contract import IMAGE_HEIGHT, IMAGE_WIDTH, V_OFF, V_VIS


def centre_visibility(cx: float, cy: float) -> int:
    """V_VIS if the centre pixel is inside the image, else V_OFF (the "through the gate" case).

    Inclusive 0..W-1 / 0..H-1 bounds -- the SAME in-frame test labels.py and labelio.py use for
    every other keypoint, so the centre's flag can never disagree with a corner's on the same pixel.
    """
    return V_VIS if (0.0 <= cx <= IMAGE_WIDTH - 1 and 0.0 <= cy <= IMAGE_HEIGHT - 1) else V_OFF


def centre_from_inner_px(inner_px) -> tuple[float, float, int] | None:
    """(cx, cy, v) for the diagonal-intersection centre of an inner-corner quad, or None.

    ``inner_px`` is (4,2) in canonical LL,LR,UR,UL order (may be off-frame -- that is the point).
    None only when the diagonals are parallel, i.e. the square is seen exactly edge-on and its
    centre is genuinely at infinity; callers that need a guaranteed row handle that case explicitly
    (see :func:`frame_centre_rows`)."""
    c = centre_from_quad(np.asarray(inner_px, dtype=float))
    if c is None:
        return None
    cx, cy = float(c[0]), float(c[1])
    return cx, cy, centre_visibility(cx, cy)


def encode_centre_row(cx: float, cy: float, v: int) -> str:
    """One sidecar line ``cx_norm cy_norm v``. Coordinates are UNCLAMPED (see the module note)."""
    return f"{cx / IMAGE_WIDTH:.6g} {cy / IMAGE_HEIGHT:.6g} {int(v)}"


def decode_centre_row(line: str) -> tuple[float, float, int]:
    """Parse a sidecar line back to PIXELS + visibility: (cx_px, cy_px, v)."""
    p = line.split()
    if len(p) != 3:
        raise ValueError(f"bad centre row: expected 3 fields, got {len(p)}: {line!r}")
    return float(p[0]) * IMAGE_WIDTH, float(p[1]) * IMAGE_HEIGHT, int(float(p[2]))


def gate_centre_row(gr) -> str:
    """One centre sidecar row for one post-augment :class:`GateRender`.

    Reads ``gr.keypoints_px`` -- the UNCLAMPED post-augment inner corners -- so the centre rides
    the same albumentations warp as the pose row. Degenerate (edge-on) quads fall back to the
    corner centroid with v=0 so this ALWAYS returns exactly one row: the centre sidecar must stay
    1:1 and in order with the pose rows, or the builder would pair gate A's corners with gate B's
    centre. The centroid of an edge-on square is off-frame in practice and v=0 masks it anyway."""
    inner = np.asarray(gr.keypoints_px, dtype=float)
    got = centre_from_inner_px(inner)
    if got is None:
        cx, cy = float(inner[:, 0].mean()), float(inner[:, 1].mean())
        return encode_centre_row(cx, cy, V_OFF)
    return encode_centre_row(*got)


def frame_centre_rows(labeled_gates) -> list[str]:
    """One centre row per labelled gate, IN THE SAME ORDER as ``labels.frame_label_rows``.

    Exactly ``len(labeled_gates)`` rows (never fewer) so the sidecar zips 1:1 with the pose label.
    """
    return [gate_centre_row(g) for g in labeled_gates]
