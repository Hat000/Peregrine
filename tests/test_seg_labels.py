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
    clip_polygon,
    polygon_area,
    seg_label_from_pose_label,
    seg_rows_from_pose_row,
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
