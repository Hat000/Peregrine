"""The hand-labeler's AREA target: the label is the quad as drawn, clipped -- not a refit.

The pose label cannot express a close gate. Off-frame keypoints are stored CLAMPED onto the border
with v=0, so any consumer has to refit the gate-plane homography to whatever survived, which needs
>= 4 in-frame keypoints with >= 2 per diagonal -- blocked on 52.9% of real flight ticks inside 2 m.
These tests pin that the area path has no such limit, and that the two errors it CAN still make
(a past-horizon outer corner, a bow-tie) are refused rather than written as wrong pixels.

Run: python -m pytest tests/test_gate_labeler_area.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "gate_labeler"))
sys.path.insert(0, str(ROOT / "scripts"))

import labelio  # noqa: E402
from racer.vision.seg_labels import (  # noqa: E402
    MIN_VISIBLE_AREA_PX, clip_polygon, polygon_area, seg_label_from_pose_label,
)

W, H = labelio.IMAGE_WIDTH, labelio.IMAGE_HEIGHT
CX, CY = W / 2.0, H / 2.0


def square(half, cx=CX, cy=CY):
    """Inner quad in the labeler's LL,LR,UR,UL order (gate frame Y-DOWN)."""
    return [(cx - half, cy + half), (cx + half, cy + half),
            (cx + half, cy - half), (cx - half, cy - half)]


def gate(half, **kw):
    inner = square(half)
    return dict(inner=inner, outer=labelio.derive_outer_from_inner(inner), **kw)


def rows_of(text, cls):
    return [r for r in text.splitlines() if r.split()[0] == str(cls)]


def row_area(text, cls):
    """px^2 of the polygon actually written for ``cls``."""
    v = [float(t) for t in rows_of(text, cls)[0].split()[1:]]
    return polygon_area(np.array([(v[i] * W, v[i + 1] * H) for i in range(0, len(v), 2)]))


# A rotated quad that ENGULFS the frame, corners far outside on all four sides -- i.e. a gate you
# are about to fly through. Axis-aligned squares are a degenerate test case here: clamping a corner
# onto the border and CLIPPING the edge give the same rectangle, which hides the loss entirely.
ENGULFING_DIAMOND = [(-360.0, 180.0), (320.0, 760.0), (1000.0, 180.0), (320.0, -400.0)]


# ---------------------------------------------------------------------------------------
# THE HEADLINE: zero corners in frame still labels
# ---------------------------------------------------------------------------------------
def test_gate_with_no_corner_in_frame_still_produces_both_masks():
    g = gate(400.0)                       # inner square engulfs the 640x360 frame
    kpts = list(g["inner"]) + list(g["outer"])
    assert all(labelio.keypoint_visibility(x, y) == labelio.V_OFF for x, y in kpts)

    text, reports = labelio.encode_seg_label([g])
    assert len(text.splitlines()) == 2, text
    assert reports[0]["gate_frame"]["status"] == "ok"
    assert reports[0]["gate_opening"]["status"] == "ok"


def test_pose_path_cannot_express_what_the_area_path_can():
    """The regression this whole change exists to prevent: the SAME gate through the pose row."""
    g = gate(400.0)
    pose_text = labelio.encode_label([g])
    derived, census = seg_label_from_pose_label(pose_text, W, H)
    assert derived == "" and census == {"fewer-than-4-usable": 1}, census

    direct, _ = labelio.encode_seg_label([g])
    assert len(direct.splitlines()) == 2


@pytest.mark.parametrize("half,cx,cy", [
    (120.0, CX, CY),            # fully inside
    (120.0, 0.0, CY),           # cropped left
    (120.0, CX, 0.0),           # cropped top
    (260.0, 40.0, 300.0),       # two edges cropped, one corner in frame
    (400.0, CX, CY),            # nothing in frame
])
def test_written_area_equals_the_clipped_quad(half, cx, cy):
    """No reconstruction anywhere: the polygon on disk IS the drawn quad clipped."""
    inner = square(half, cx, cy)
    outer = labelio.derive_outer_from_inner(inner)
    text, _ = labelio.encode_seg_label([{"inner": inner, "outer": outer}])
    assert rows_of(text, labelio.SEG_OPENING_CLASS), text
    expected = polygon_area(clip_polygon(inner, W, H))
    assert row_area(text, labelio.SEG_OPENING_CLASS) == pytest.approx(expected, rel=1e-3)


def test_rows_are_normalised_inside_the_unit_square():
    text, _ = labelio.encode_seg_label([gate(400.0), gate(90.0)])
    for row in text.splitlines():
        vals = [float(v) for v in row.split()[1:]]
        assert vals and all(0.0 <= v <= 1.0 for v in vals), row


def test_gate_entirely_outside_the_frame_writes_nothing():
    inner = square(60.0, cx=-500.0, cy=-500.0)
    text, reports = labelio.encode_seg_label(
        [{"inner": inner, "outer": labelio.derive_outer_from_inner(inner)}])
    assert text == ""
    assert reports[0]["gate_opening"]["status"] == "no-visible-area"


def test_sliver_below_the_area_floor_is_dropped():
    inner = square(120.0, cx=-120.0 + 0.2, cy=CY)     # a ~0.4 px wide strip
    text, reports = labelio.encode_seg_label(
        [{"inner": inner, "outer": labelio.derive_outer_from_inner(inner)}])
    assert reports[0]["gate_opening"]["area"] < MIN_VISIBLE_AREA_PX
    assert not rows_of(text, labelio.SEG_OPENING_CLASS)


# ---------------------------------------------------------------------------------------
# The two ways an area label can still be WRONG, both refused
# ---------------------------------------------------------------------------------------
def test_bowtie_is_refused_not_shoelaced():
    inner = square(120.0)
    bow = [inner[0], inner[1], inner[3], inner[2]]           # corners clicked out of order
    assert not labelio.quad_is_simple(bow)
    _, rep = labelio.seg_rows_for_gate(bow, labelio.derive_outer_from_inner(inner))
    assert rep["gate_opening"]["status"] == "self-intersecting"


def test_past_horizon_outer_is_suppressed_but_recoverable_by_hand():
    """derive_outer_from_inner substitutes a PLACEHOLDER past the vanishing line. Harmless as a
    v=0 keypoint; as a polygon vertex it would be invented pixels."""
    edge_on = [(247.5, 275.7), (297.8, 547.3), (523.1, 24.1), (317.9, 24.1)]
    outer = labelio.derive_outer_from_inner(edge_on)
    assert not all(labelio.outer_validity(edge_on))

    _, rep = labelio.seg_rows_for_gate(edge_on, outer)
    assert rep["gate_frame"]["status"] == "past-horizon"
    assert rep["gate_opening"]["status"] == "ok"             # the clicked square is still real

    # dragging the outer corners marks them detached = MEASURED, and the class comes back
    _, rep2 = labelio.seg_rows_for_gate(edge_on, outer, outer_detached=[True] * 4)
    assert rep2["gate_frame"]["status"] == "ok"


def test_partially_detached_outer_still_suppressed():
    """One hand-placed corner does not vouch for the other three."""
    edge_on = [(247.5, 275.7), (297.8, 547.3), (523.1, 24.1), (317.9, 24.1)]
    outer = labelio.derive_outer_from_inner(edge_on)
    bad = [i for i, ok in enumerate(labelio.outer_validity(edge_on)) if not ok]
    detached = [False] * 4
    detached[bad[0]] = True                                   # fix only the first broken one
    _, rep = labelio.seg_rows_for_gate(edge_on, outer, outer_detached=detached)
    expected_ok = len(bad) == 1
    assert (rep["gate_frame"]["status"] == "ok") is expected_ok, rep


# ---------------------------------------------------------------------------------------
# Round-trip: the sidecar keeps what the pose row destroys
# ---------------------------------------------------------------------------------------
def test_geometry_sidecar_round_trips_off_frame_handles_exactly():
    g = gate(400.0, occluded=[False] * 8, outer_detached=[False] * 4)
    back = labelio.decode_geometry(json.loads(json.dumps(labelio.encode_geometry([g]))))
    assert back[0]["inner"] == [tuple(p) for p in g["inner"]]
    assert back[0]["outer"] == [tuple(p) for p in g["outer"]]
    assert back[0]["outer_detached"] == [False] * 4


def test_reloading_through_the_pose_row_corrupts_the_area():
    """Why the sidecar exists: edit-save-edit-save through the lossy path is data destruction."""
    quad = ENGULFING_DIAMOND
    assert labelio.quad_is_simple(quad)
    g = {"inner": quad, "outer": labelio.derive_outer_from_inner(quad)}
    before, _ = labelio.encode_seg_label([g])

    clamped = labelio.decode_label(labelio.encode_label([g]))[0]
    after, _ = labelio.encode_seg_label([{"inner": clamped["inner"], "outer": clamped["outer"]}])
    a0 = row_area(before, labelio.SEG_OPENING_CLASS)
    a1 = row_area(after, labelio.SEG_OPENING_CLASS)
    # the drawn opening covers the whole frame; the clamped one collapses to a diamond inside it
    assert a0 == pytest.approx((W - 1) * (H - 1), rel=1e-3)
    assert a1 < 0.6 * a0, (a0, a1)

    # ...and via the sidecar it survives byte-for-byte
    kept = labelio.decode_geometry(labelio.encode_geometry([g]))[0]
    resaved, _ = labelio.encode_seg_label([{"inner": kept["inner"], "outer": kept["outer"]}])
    assert resaved == before


def test_occlusion_and_detachment_survive_the_sidecar():
    g = gate(120.0, occluded=[False, True, False, False, False, False, True, False],
             outer_detached=[True, False, True, False])
    back = labelio.decode_geometry(labelio.encode_geometry([g]))[0]
    assert back["vis"][1] == labelio.V_OCC
    assert back["outer_detached"] == [True, False, True, False]


def test_negative_frame_writes_an_empty_area_label():
    text, reports = labelio.encode_seg_label([])
    assert text == "" and reports == []


# ---------------------------------------------------------------------------------------
# The pose target must be untouched by all of this
# ---------------------------------------------------------------------------------------
def test_pose_encoding_is_unchanged():
    inner = square(100.0)
    row = labelio.encode_gate_row(inner, labelio.derive_outer_from_inner(inner))
    assert row.split()[0] == "0" and len(row.split()) == 29
    assert row.split()[1] == "0.5"
    assert [row.split()[5 + 3 * i + 2] for i in range(4)] == ["2"] * 4


# ---------------------------------------------------------------------------------------
# The dataset builder must take authored area labels VERBATIM
# ---------------------------------------------------------------------------------------
def test_builder_prefers_the_authored_area_label(tmp_path):
    from collections import Counter

    import build_seg_dataset as bsd

    labels = tmp_path / "labels"
    (labels / "seg").mkdir(parents=True)
    pose = labels / "f0.txt"
    g = gate(400.0)
    pose.write_text(labelio.encode_label([g]))
    authored, _ = labelio.encode_seg_label([g])
    (labels / "seg" / "f0.txt").write_text(authored)

    census = Counter()
    text, direct = bsd.seg_text_for(pose, census)
    assert direct and text == authored
    assert census["direct-area-frames"] == 1 and census["direct-area-rows"] == 2

    # ...and without the sidecar the same frame converts to NOTHING, which is the point
    (labels / "seg" / "f0.txt").unlink()
    text2, direct2 = bsd.seg_text_for(pose, Counter())
    assert not direct2 and text2 == ""


def test_builder_scan_skips_the_seg_subdirectory(tmp_path):
    import build_seg_dataset as bsd

    root = tmp_path / "ds"
    (root / "labels" / "seg").mkdir(parents=True)
    (root / "images").mkdir()
    (root / "labels" / "f0.txt").write_text(labelio.encode_label([gate(100.0)]))
    (root / "labels" / "seg" / "f0.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n")
    (root / "images" / "f0.png").write_bytes(b"")

    pairs = bsd.scan_pairs(root)
    assert [lp.name for _, lp in pairs] == ["f0.txt"]
    assert all("seg" not in lp.parts for _, lp in pairs)


def test_builder_finds_the_render_layout_seg_labels(tmp_path):
    """Two on-disk layouts produce area labels; missing the render one would have silently
    re-derived 3300 frames from their clamped pose rows."""
    from collections import Counter

    import build_seg_dataset as bsd

    root = tmp_path / "render"
    (root / "labels" / "train").mkdir(parents=True)
    (root / "seg" / "train").mkdir(parents=True)
    pose = root / "labels" / "train" / "000000.txt"
    g = gate(400.0)
    pose.write_text(labelio.encode_label([g]))
    authored, _ = labelio.encode_seg_label([g])
    (root / "seg" / "train" / "000000.txt").write_text(authored)

    assert bsd.direct_seg_path(pose) == root / "seg" / "train" / "000000.txt"
    census = Counter()
    text, direct = bsd.seg_text_for(pose, census)
    assert direct and text == authored and census["direct-area-rows"] == 2


def test_direct_seg_path_is_none_when_absent(tmp_path):
    import build_seg_dataset as bsd

    (tmp_path / "labels").mkdir()
    p = tmp_path / "labels" / "x.txt"
    p.write_text("")
    assert bsd.direct_seg_path(p) is None
