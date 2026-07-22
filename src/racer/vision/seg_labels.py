"""8-keypoint POSE labels -> YOLO SEGMENTATION labels, for the line-based gate solver.

WHY SEGMENTATION AND NOT KEYPOINTS. A keypoint that leaves the frame is unlearnable in stock
ultralytics (augmentation zeros + clamps it, the pose loss masks v=0, and inference clips the
prediction back onto the border), so a cropped gate returns corners pinned to the image edge --
measured 12.3 px error against homography truth vs 0.73 px in-frame. A MASK has no such failure
mode: the target is simply the visible pixels, and off-frame geometry is CLIPPED, which is the
correct label rather than a corrupted one. The gate solver then fits LINES to the mask boundary,
and a line is determined by the in-frame pixels it passes through -- so it survives cropping. This
is the whole architectural argument for the second path.

TWO CLASSES, BOTH CONVEX QUADS -- ``gate_frame`` (the 2.72 m outer square) and ``gate_opening``
(the 1.5 m inner square). Deliberately NOT one annulus polygon: a ring needs a
polygon-with-a-bridge encoding that rasterises with a seam, and it forces the solver to recover the
opening via contour-hierarchy hole-finding (RETR_CCOMP), which is exactly the fragile step. Two
nested convex instances give the solver both squares directly -- 8 edge lines, no hole logic -- and
each target is an easy convex blob. ``gate_opening`` is genuinely visible (you can see through it);
``gate_frame`` is AMODAL over its own opening, which segmentation models handle well and which
makes its outer boundary -- the thing we actually fit -- unambiguous.

GEOMETRY IS RECONSTRUCTED, NOT READ. Stored labels clamp off-frame corners to the border with v=0,
so their coordinates are meaningless. We never use them: we fit the gate-plane homography to the
USABLE keypoints only (v>0) and re-project BOTH canonical squares through it, recovering the true
corner positions including the off-frame ones. Measured over 2826 real rows, 59.1% carry all 4
inner corners and the rest are mostly 3-inner-plus-outers -- both are >=4 usable points, which is
all the homography needs (subject to the diagonal rule in ``gate_plane_homography``).

Run the self-check:  python -m racer.vision.seg_labels
"""
from __future__ import annotations

import numpy as np

from racer.vision.gate_pose import gate_plane_homography, project_gate_squares

CLASS_FRAME = 0        # the 2.72 m outer square
CLASS_OPENING = 1      # the 1.5 m inner square
CLASS_NAMES = {CLASS_FRAME: "gate_frame", CLASS_OPENING: "gate_opening"}

N_KEYPOINTS = 8
_POSE_ROW_FIELDS = 5 + 3 * N_KEYPOINTS      # class cx cy w h (x y v)*8

# A polygon must keep enough area in-frame to be a meaningful target; below this it is a sliver at
# the edge that teaches the model nothing and inflates the instance count.
MIN_VISIBLE_AREA_PX = 64.0


def parse_pose_row(row: str, img_w: int, img_h: int):
    """One pose label row -> (keypoints_px (8,2), visibility (8,)). None if the row is not a pose row."""
    p = row.split()
    if len(p) != _POSE_ROW_FIELDS:
        return None
    kp = np.empty((N_KEYPOINTS, 2), float)
    vis = np.empty(N_KEYPOINTS, int)
    for i in range(N_KEYPOINTS):
        x, y, v = p[5 + 3 * i:8 + 3 * i]
        kp[i] = (float(x) * img_w, float(y) * img_h)
        vis[i] = int(float(v))
    return kp, vis


def clip_polygon(poly: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon to the image rectangle.

    Off-frame corners are the NORMAL case here, so this is load-bearing rather than defensive: the
    clipped outline is the true visible silhouette, which is exactly what a mask label must contain.
    Returns an (n,2) array, possibly empty."""
    def _clip(pts, inside, intersect):
        out = []
        for i in range(len(pts)):
            cur, prv = pts[i], pts[i - 1]
            c_in, p_in = inside(cur), inside(prv)
            if c_in:
                if not p_in:
                    out.append(intersect(prv, cur))
                out.append(cur)
            elif p_in:
                out.append(intersect(prv, cur))
        return out

    def _lerp(a, b, t):
        return a + (b - a) * t

    pts = [np.asarray(p, float) for p in poly]
    edges = (
        (lambda p: p[0] >= 0.0,        lambda a, b: _lerp(a, b, (0.0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= img_w - 1,  lambda a, b: _lerp(a, b, (img_w - 1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= 0.0,        lambda a, b: _lerp(a, b, (0.0 - a[1]) / (b[1] - a[1]))),
        (lambda p: p[1] <= img_h - 1,  lambda a, b: _lerp(a, b, (img_h - 1 - a[1]) / (b[1] - a[1]))),
    )
    for inside, intersect in edges:
        if not pts:
            return np.empty((0, 2))
        pts = _clip(pts, inside, intersect)
    return np.asarray(pts, float).reshape(-1, 2)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def to_yolo_seg_row(class_id: int, poly: np.ndarray, img_w: int, img_h: int) -> str:
    """``class x1 y1 x2 y2 ...`` with coordinates normalised to [0,1] (the ultralytics seg format)."""
    xy = np.asarray(poly, float).copy()
    xy[:, 0] = np.clip(xy[:, 0] / img_w, 0.0, 1.0)
    xy[:, 1] = np.clip(xy[:, 1] / img_h, 0.0, 1.0)
    return f"{int(class_id)} " + " ".join(f"{v:.6g}" for v in xy.reshape(-1))


def seg_rows_from_pose_row(row: str, img_w: int = 640, img_h: int = 360):
    """The seg rows for ONE gate. Returns (rows, reason): rows is [] when the gate is unusable and
    ``reason`` names why, so a converter can report a census instead of silently dropping data."""
    parsed = parse_pose_row(row, img_w, img_h)
    if parsed is None:
        return [], "not-a-pose-row"
    kp, vis = parsed
    usable = vis > 0                      # v=0 means off-frame AND clamped -> coordinates are junk
    if int(usable.sum()) < 4:
        return [], "fewer-than-4-usable"
    H = gate_plane_homography(kp, usable)
    if H is None:
        return [], "degenerate-homography"
    squares = project_gate_squares(H)
    if squares is None:
        return [], "vanishing-line"
    inner, outer = squares

    rows = []
    for cid, quad in ((CLASS_FRAME, outer), (CLASS_OPENING, inner)):
        vis_poly = clip_polygon(quad, img_w, img_h)
        if polygon_area(vis_poly) < MIN_VISIBLE_AREA_PX:
            continue                      # entirely (or all but) off-frame -> nothing to segment
        rows.append(to_yolo_seg_row(cid, vis_poly, img_w, img_h))
    if not rows:
        return [], "no-visible-area"
    return rows, "ok"


def seg_label_from_pose_label(text: str, img_w: int = 640, img_h: int = 360):
    """Whole-file conversion. Returns (seg_text, census) where census counts per-gate outcomes.
    A negative frame (empty pose label) converts to an empty seg label -- WRITE that empty file,
    ultralytics treats a missing label as an unlabelled image rather than a true negative."""
    rows, census = [], {}
    for line in text.splitlines():
        if not line.strip():
            continue
        r, why = seg_rows_from_pose_row(line, img_w, img_h)
        census[why] = census.get(why, 0) + 1
        rows.extend(r)
    return ("\n".join(rows) + "\n") if rows else "", census


if __name__ == "__main__":
    W, H = 640, 360

    def _pose_row(inner, outer, vis):
        f = ["0", "0.5", "0.5", "0.5", "0.5"]
        for (x, y), v in zip(list(inner) + list(outer), vis):
            f += [f"{x / W:.6g}", f"{y / H:.6g}", str(v)]
        return " ".join(f)

    # head-on centred gate: inner half 60 px -> outer half 60 * 1.36/0.75 = 108.8 px
    cx, cy, s = 320.0, 180.0, 60.0
    so = s * 1.36 / 0.75
    inner = [(cx - s, cy + s), (cx + s, cy + s), (cx + s, cy - s), (cx - s, cy - s)]
    outer = [(cx - so, cy + so), (cx + so, cy + so), (cx + so, cy - so), (cx - so, cy - so)]
    rows, why = seg_rows_from_pose_row(_pose_row(inner, outer, [2] * 8), W, H)
    assert why == "ok" and len(rows) == 2, (why, rows)
    assert rows[0].startswith("0 ") and rows[1].startswith("1 ")

    # the load-bearing case: outer corners OFF-FRAME and stored clamped at v=0. Their coordinates
    # are junk, so the fit must ignore them and RECONSTRUCT the outer square from the inner 4.
    junk = [(0.0, 0.0)] * 4
    rows2, why2 = seg_rows_from_pose_row(_pose_row(inner, junk, [2, 2, 2, 2, 0, 0, 0, 0]), W, H)
    assert why2 == "ok" and len(rows2) == 2, (why2, rows2)
    got = np.array([float(v) for v in rows2[0].split()[1:]]).reshape(-1, 2) * [W, H]
    assert abs(polygon_area(got) - (2 * so) ** 2) < 1.0, polygon_area(got)   # full outer square

    # a gate cropped by the frame edge: the polygon must be CLIPPED, not dropped
    off = [(-200.0, 200.0), (100.0, 200.0), (100.0, -50.0), (-200.0, -50.0)]
    rows3, why3 = seg_rows_from_pose_row(_pose_row(off, junk, [2, 2, 2, 2, 0, 0, 0, 0]), W, H)
    assert why3 == "ok", why3
    p3 = np.array([float(v) for v in rows3[-1].split()[1:]]).reshape(-1, 2)
    assert (p3 >= 0.0).all() and (p3 <= 1.0).all(), p3

    # 3 inner + 1 outer = 4 usable, straddling BOTH diagonals ({0,2} and {1,5}) -> solvable.
    in4 = [inner[0], inner[1], inner[2], (0.0, 0.0)]
    out4 = [(0.0, 0.0), outer[1], (0.0, 0.0), (0.0, 0.0)]
    rows4, why4 = seg_rows_from_pose_row(_pose_row(in4, out4, [2, 2, 2, 0, 0, 2, 0, 0]), W, H)
    assert why4 == "ok", why4

    # ...but 4 usable points that pile onto ONE diagonal ({0,2,4} vs {1}) are collinear-degenerate.
    # The homography would "solve" and hand back a plausible-looking wrong plane, so it must be
    # REJECTED here rather than quietly poisoning the training set.
    in5 = [inner[0], (0.0, 0.0), inner[2], (0.0, 0.0)]
    out5 = [outer[0], (0.0, 0.0), outer[2], (0.0, 0.0)]
    _, why_deg = seg_rows_from_pose_row(_pose_row(in5, out5, [2, 0, 2, 0, 2, 0, 2, 0]), W, H)
    assert why_deg == "degenerate-homography", why_deg

    # too few usable points must be REPORTED, not silently emitted
    _, why5 = seg_rows_from_pose_row(_pose_row(inner, junk, [2, 2, 0, 0, 0, 0, 0, 0]), W, H)
    assert why5 == "fewer-than-4-usable", why5

    text, census = seg_label_from_pose_label(_pose_row(inner, outer, [2] * 8) + "\n", W, H)
    assert census == {"ok": 1} and len(text.splitlines()) == 2
    assert seg_label_from_pose_label("", W, H) == ("", {})
    print("seg_labels self-check OK")
