"""Pose -> segmentation label conversion, and the dataset builder's identity guarantees.

The conversion's whole value is that it recovers geometry the POSE label cannot express: a corner
that left the frame is stored clamped to the border with v=0, so its coordinates are junk. These
tests pin that the junk is ignored and the true square is RECONSTRUCTED through the gate-plane
homography, and that the builder never pairs an image with another frame's label.

Run: python -m pytest tests/test_seg_labels.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from racer.vision.seg_labels import (  # noqa: E402
    CLASS_FRAME,
    CLASS_OPENING,
    MIN_VISIBLE_AREA_PX,
    clip_polygon,
    polygon_area,
    polygons_from_mask,
    seg_label_from_pose_label,
    seg_rows_from_pose_row,
    seg_rows_from_silhouette,
)

W, H = 640, 360
_S = 1.36 / 0.75            # outer/inner half-size ratio


def pose_row(inner, outer, vis):
    f = ["0", "0.5", "0.5", "0.5", "0.5"]
    for (x, y), v in zip(list(inner) + list(outer), vis):
        f += [f"{x / W:.8g}", f"{y / H:.8g}", str(int(v))]
    return " ".join(f)


def square(cx, cy, half):
    """LL,LR,UR,UL in image pixels (gate frame is Y-DOWN, so LL is the +y corner)."""
    return [(cx - half, cy + half), (cx + half, cy + half),
            (cx + half, cy - half), (cx - half, cy - half)]


def poly_of(row):
    return np.array([float(v) for v in row.split()[1:]]).reshape(-1, 2) * [W, H]


def test_full_gate_gives_both_classes_in_order():
    rows, why = seg_rows_from_pose_row(
        pose_row(square(320, 180, 60), square(320, 180, 60 * _S), [2] * 8), W, H)
    assert why == "ok" and len(rows) == 2
    assert int(rows[0].split()[0]) == CLASS_FRAME
    assert int(rows[1].split()[0]) == CLASS_OPENING
    assert polygon_area(poly_of(rows[1])) < polygon_area(poly_of(rows[0]))


def test_offframe_outer_is_reconstructed_not_read():
    """THE point of the module: outer corners stored clamped at (0,0)/v=0 must be ignored, and the
    true outer square recovered from the inner 4. Reading them would give a degenerate polygon."""
    inner = square(320, 180, 60)
    junk = [(0.0, 0.0)] * 4
    rows, why = seg_rows_from_pose_row(pose_row(inner, junk, [2, 2, 2, 2, 0, 0, 0, 0]), W, H)
    assert why == "ok"
    got = polygon_area(poly_of(rows[0]))
    assert got == pytest.approx((2 * 60 * _S) ** 2, rel=1e-3)


def test_partial_inner_still_solves_when_diagonals_are_covered():
    """3 inner + 1 outer, straddling both diagonals -> 4 points in general position -> solvable."""
    inner, outer = square(320, 180, 60), square(320, 180, 60 * _S)
    kp_in = [inner[0], inner[1], inner[2], (0.0, 0.0)]
    kp_out = [(0.0, 0.0), outer[1], (0.0, 0.0), (0.0, 0.0)]
    rows, why = seg_rows_from_pose_row(pose_row(kp_in, kp_out, [2, 2, 2, 0, 0, 2, 0, 0]), W, H)
    assert why == "ok"
    assert polygon_area(poly_of(rows[1])) == pytest.approx((2 * 60) ** 2, rel=1e-3)


def test_four_points_on_one_diagonal_are_rejected():
    """All 8 keypoints lie on two diagonals, so 4 points sharing one are collinear. The homography
    would still 'solve' and return a confident wrong plane -- it must be refused, not emitted."""
    inner, outer = square(320, 180, 60), square(320, 180, 60 * _S)
    kp_in = [inner[0], (0.0, 0.0), inner[2], (0.0, 0.0)]
    kp_out = [outer[0], (0.0, 0.0), outer[2], (0.0, 0.0)]
    _, why = seg_rows_from_pose_row(pose_row(kp_in, kp_out, [2, 0, 2, 0, 2, 0, 2, 0]), W, H)
    assert why == "degenerate-homography"


def test_fewer_than_four_usable_is_reported_not_silently_dropped():
    _, why = seg_rows_from_pose_row(
        pose_row(square(320, 180, 60), [(0.0, 0.0)] * 4, [2, 2, 0, 0, 0, 0, 0, 0]), W, H)
    assert why == "fewer-than-4-usable"


def test_cropped_gate_is_clipped_into_frame():
    inner = [(-200.0, 260.0), (150.0, 260.0), (150.0, -40.0), (-200.0, -40.0)]
    rows, why = seg_rows_from_pose_row(
        pose_row(inner, [(0.0, 0.0)] * 4, [2, 2, 2, 2, 0, 0, 0, 0]), W, H)
    assert why == "ok"
    for r in rows:
        p = np.array([float(v) for v in r.split()[1:]]).reshape(-1, 2)
        assert (p >= 0.0).all() and (p <= 1.0).all()


def test_clip_polygon_keeps_area_and_drops_fully_outside():
    inside = np.array(square(320, 180, 50), float)
    assert polygon_area(clip_polygon(inside, W, H)) == pytest.approx(100.0 ** 2)
    outside = np.array(square(-500, -500, 20), float)
    assert len(clip_polygon(outside, W, H)) == 0


def test_negative_frame_converts_to_empty_label():
    assert seg_label_from_pose_label("", W, H) == ("", {})


def test_builder_rekeys_colliding_stems(tmp_path):
    """The pose file lists mix datasets that each number frames from 000000. Keying outputs by stem
    silently overwrote 437 of 1572 labels, pairing images with another dataset's geometry."""
    import build_seg_dataset as B

    a = tmp_path / "ds_a" / "images" / "train"
    b = tmp_path / "ds_b" / "images" / "train"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "000001.png").write_bytes(b"x")
        lab = Path(str(d).replace("images", "labels"))
        lab.mkdir(parents=True, exist_ok=True)
    row = pose_row(square(320, 180, 60), square(320, 180, 60 * _S), [2] * 8)
    (tmp_path / "ds_a" / "labels" / "train" / "000001.txt").write_text(row + "\n")
    (tmp_path / "ds_b" / "labels" / "train" / "000001.txt").write_text(row + "\n")

    from collections import Counter
    out = tmp_path / "seg"
    kept = B.convert_split([a / "000001.png", b / "000001.png"], out, "train", Counter())
    assert len(kept) == 2, "both frames must survive"
    assert len({p.name for p in kept}) == 2, f"names still collide: {[p.name for p in kept]}"
    assert len(list((out / "labels" / "train").glob("*.txt"))) == 2


def test_builder_label_lives_beside_its_own_image(tmp_path):
    """Ultralytics finds a label by substituting /images/ -> /labels/ on the IMAGE path. If the new
    lists pointed at the ORIGINAL images, it would read their POSE labels and never see the seg
    ones -- training on the wrong target, with no error anywhere."""
    import build_seg_dataset as B
    from collections import Counter

    src = tmp_path / "ds" / "images" / "train"
    src.mkdir(parents=True)
    (src / "000001.png").write_bytes(b"x")
    lab = tmp_path / "ds" / "labels" / "train"
    lab.mkdir(parents=True)
    (lab / "000001.txt").write_text(
        pose_row(square(320, 180, 60), square(320, 180, 60 * _S), [2] * 8) + "\n")

    out = tmp_path / "seg"
    kept = B.convert_split([src / "000001.png"], out, "train", Counter())
    img = kept[0]
    assert img.is_relative_to(out), "listed image must live in the OUT tree, not the source tree"
    derived = Path(str(img).replace("images", "labels")).with_suffix(".txt")
    assert derived.exists(), f"ultralytics would look for {derived}"
    assert derived.read_text().strip(), "and it must not be empty"


# ==========================================================================================
# TRUE-SILHOUETTE path: rendered masks -> polygons
# ==========================================================================================
def _ring_mask(h=H, w=W, outer=(200, 60, 440, 300), inner=(260, 110, 380, 250)):
    """A head-on gate: filled outer rect with the inner rect carved out. (x0,y0,x1,y1)."""
    m = np.zeros((h, w), bool)
    m[outer[1]:outer[3], outer[0]:outer[2]] = True
    m[inner[1]:inner[3], inner[0]:inner[2]] = False
    return m


def _rect_mask(rect, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[rect[1]:rect[3], rect[0]:rect[2]] = True
    return m


def test_polygons_from_mask_traces_the_outer_boundary_and_fills_holes():
    """A YOLO-seg row is ONE closed polygon and cannot carry a hole, so the annulus must come back
    as its filled outer boundary -- otherwise the rasterised target would have a bridge seam."""
    polys = polygons_from_mask(_ring_mask())
    assert len(polys) == 1
    assert abs(polygon_area(polys[0]) - 240 * 240) < 0.02 * 240 * 240   # the FILLED outer square
    assert len(polys[0]) == 4                                          # a real square stays a square


def test_polygons_from_mask_drops_slivers_and_orders_by_area():
    big, small = _rect_mask((100, 100, 200, 200)), _rect_mask((10, 10, 15, 15))   # 25 px < 64 floor
    polys = polygons_from_mask(big | small)
    assert len(polys) == 1, "a sub-MIN_VISIBLE_AREA_PX blob must not become an instance"
    assert polygon_area(polys[0]) > MIN_VISIBLE_AREA_PX
    two = polygons_from_mask(big | _rect_mask((10, 10, 40, 40)))
    assert len(two) == 2 and polygon_area(two[0]) > polygon_area(two[1]), "largest first"


def test_polygons_from_mask_does_not_straighten_a_non_quad():
    """The whole point of the rendered silhouette is that a close gate is NOT a quad -- the depth
    of the frame adds extra silhouette edges. Simplification must keep them."""
    oct_mask = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = 320.0, 180.0
    oct_mask[(np.abs(xx - cx) < 120) & (np.abs(yy - cy) < 120)
             & (np.abs(xx - cx) + np.abs(yy - cy) < 190)] = True        # a regular-ish octagon
    poly = polygons_from_mask(oct_mask)[0]
    assert len(poly) >= 8, f"octagon collapsed to {len(poly)} vertices -- it was straightened"


def test_silhouette_rows_are_frame_plus_opening():
    ring = _ring_mask()
    opening = _rect_mask((260, 110, 380, 250))
    rows, why = seg_rows_from_silhouette(ring, opening, W, H)
    assert why == "ok" and len(rows) == 2
    assert rows[0].startswith(f"{CLASS_FRAME} ") and rows[1].startswith(f"{CLASS_OPENING} ")
    frame_poly = np.array([float(v) for v in rows[0].split()[1:]]).reshape(-1, 2) * [W, H]
    open_poly = np.array([float(v) for v in rows[1].split()[1:]]).reshape(-1, 2) * [W, H]
    assert abs(polygon_area(frame_poly) - 240 * 240) < 0.02 * 240 * 240
    assert abs(polygon_area(open_poly) - 120 * 140) < 0.02 * 120 * 140


def test_opening_matches_contour_hierarchy_when_the_gate_is_fully_in_frame():
    """The opening is derived from a rendered PROXY PLANE rather than RETR_CCOMP hole-finding,
    because the hole stops being a topological hole the moment the gate is cropped. This pins that
    the proxy answers the SAME question when contour hierarchy can also answer it."""
    import cv2

    ring = _ring_mask()
    cnts, hier = cv2.findContours(ring.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = [c for c, h in zip(cnts, hier[0]) if h[3] != -1]
    assert len(holes) == 1, "sanity: an uncropped ring really does have one contour-hierarchy hole"
    hole_mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(hole_mask, holes, -1, 1, -1)
    proxy = _rect_mask((260, 110, 380, 250))
    iou = (hole_mask.astype(bool) & proxy).sum() / (hole_mask.astype(bool) | proxy).sum()
    # Not 1.0 by construction: a hole contour is traced through the pixel CENTRES of the ring's
    # inner boundary, so filling it insets the region by ~1 px on each side (0.97 for a 120x140
    # hole). Anything below ~0.96 would mean a real disagreement, not the tracing convention.
    assert iou > 0.96, f"proxy opening disagrees with the contour hole (IoU {iou:.3f})"


def test_cropped_gate_split_into_two_bars_still_yields_ONE_frame_instance():
    """Inside ~2 m the gate is wider than the image, so the visible ring can be two disconnected
    vertical bars. Unioning the opening back in reconnects them -- otherwise one gate would emit two
    class-0 instances and the association downstream would see a gate that is not there."""
    ring = _rect_mask((0, 0, 40, H)) | _rect_mask((600, 0, W, H))
    assert len(polygons_from_mask(ring)) == 2, "sanity: the ring alone really is two components"
    opening = _rect_mask((40, 0, 600, H))
    rows, why = seg_rows_from_silhouette(ring, opening, W, H)
    assert sum(1 for r in rows if r.startswith(f"{CLASS_FRAME} ")) == 1, rows
    assert why == "ok"


def test_closed_hole_emits_no_opening_rather_than_inventing_one():
    """A gate oblique enough that its own side walls close the hole has NO see-through region. The
    honest label is 'no opening', never a quad projected on faith."""
    rows, why = seg_rows_from_silhouette(_rect_mask((200, 60, 440, 300)), None, W, H)
    assert why == "ok-no-opening"
    assert len(rows) == 1 and rows[0].startswith(f"{CLASS_FRAME} ")
    # ... and the same when the hole survives but is a sub-floor sliver (a distant gate)
    _, why2 = seg_rows_from_silhouette(_rect_mask((200, 60, 440, 300)),
                                       _rect_mask((300, 150, 306, 156)), W, H)
    assert why2 == "ok-no-opening"


def test_empty_silhouette_is_reported_not_emitted():
    rows, why = seg_rows_from_silhouette(np.zeros((H, W), bool), np.zeros((H, W), bool), W, H)
    assert rows == [] and why == "no-visible-area"


def test_silhouette_rows_are_normalised_into_frame():
    ring = _ring_mask(outer=(0, 0, 300, 300), inner=(50, 50, 250, 250))    # touching the border
    rows, _ = seg_rows_from_silhouette(ring, _rect_mask((50, 50, 250, 250)), W, H)
    for r in rows:
        v = np.array([float(x) for x in r.split()[1:]])
        assert (v >= 0.0).all() and (v <= 1.0).all(), r
