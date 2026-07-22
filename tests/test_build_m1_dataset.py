"""Tests for the M+1 (5-keypoint) dataset builder.

Covers the assembly correctness and the collision-safety this builder inherits from
build_seg_dataset: the synthetic 8-kpt pose row + centre sidecar splice into a well-formed 5-kpt
row (inner block byte-identical, centre appended and re-clamped), a centre/pose count mismatch is
DROPPED rather than silently mis-paired, negatives survive as empty files, same-stem frames from
different source datasets are re-keyed apart, and a hand gate assembles a 5-kpt row with a centre.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tools" / "gate_labeler"))

import labelio  # noqa: E402
import build_m1_dataset as m1  # noqa: E402
from racer.vision.blender_gen.center_label import centre_from_inner_px, encode_centre_row  # noqa: E402


def _pose8_row(inner, outer):
    """An 8-keypoint render pose row (inner then outer), byte-shaped like labels.py writes."""
    return labelio.encode_gate_row(inner, outer)


def _centre_row(inner):
    return encode_centre_row(*centre_from_inner_px(np.asarray(inner, float)))


# a plain in-frame gate (all corners on screen)
_INNER = [(220.0, 240.0), (420.0, 240.0), (420.0, 90.0), (220.0, 90.0)]   # LL,LR,UR,UL
_OUTER = [(160.0, 300.0), (480.0, 300.0), (480.0, 40.0), (160.0, 40.0)]


# ----------------------------------------------------------------------------------------------
# render splice
# ----------------------------------------------------------------------------------------------
def test_render_splice_makes_5kpt_row_inner_verbatim():
    pose = _pose8_row(_INNER, _OUTER)
    crow = _centre_row(_INNER)
    row = m1.m1_row_from_render(pose, crow)
    f = row.split()
    assert len(f) == 5 + 3 * 5, f                       # class cx cy w h (x y v)*5
    # class + bbox + inner block are byte-identical to the source pose row's first 17 fields
    assert f[:5 + 3 * 4] == pose.split()[:5 + 3 * 4]
    # the appended centre matches the image centre of the (in-frame) gate, v=2
    cx = float(f[-3]) * m1.IMG_W
    cy = float(f[-2]) * m1.IMG_H
    assert abs(cx - 320.0) < 1.0 and abs(cy - 165.0) < 1.0 and f[-1] == "2"


def test_render_offframe_centre_is_reclamped_v0():
    # a centre sidecar row sitting OFF-frame (x>W): the pose row must clamp it to [0,1] with v=0.
    pose = _pose8_row(_INNER, _OUTER)
    row = m1.m1_row_from_render(pose, "1.28 0.49 0")     # cx_norm=1.28 -> off-frame right
    f = row.split()
    assert f[-1] == "0" and 0.0 <= float(f[-3]) <= 1.0   # v=0, coordinate re-clamped into [0,1]


def test_render_frame_count_mismatch_is_dropped():
    from collections import Counter
    c = Counter()
    two_pose = _pose8_row(_INNER, _OUTER) + "\n" + _pose8_row(_INNER, _OUTER)
    text, ctext = m1.convert_render_frame(two_pose, _centre_row(_INNER) + "\n", c)  # 2 poses, 1 centre
    assert text is None and c["render-center-count-mismatch"] == 1


def test_render_negative_stays_empty():
    from collections import Counter
    c = Counter()
    text, ctext = m1.convert_render_frame("", "", c)
    assert text == "" and ctext == "" and c["render-negative"] == 1


# ----------------------------------------------------------------------------------------------
# hand assembly
# ----------------------------------------------------------------------------------------------
def test_hand_gate_all_visible_uses_diagonal_centre():
    from collections import Counter
    census, how = Counter(), Counter()
    gate = {"inner": _INNER, "outer": _OUTER, "occluded": [False] * 8, "outer_detached": [False] * 4}
    geom = labelio.decode_geometry(labelio.encode_geometry([gate]))
    text, ctext = m1.hand_gate_rows(geom, census, how)
    f = text.split()
    assert len(f) == 5 + 3 * 5
    assert how["diag-handles"] == 1                      # all-visible -> exact handle diagonal
    cx = float(f[-3]) * m1.IMG_W
    assert abs(cx - 320.0) < 1.0 and f[-1] == "2"


# ----------------------------------------------------------------------------------------------
# end-to-end: collision-safety + layout + data.yaml
# ----------------------------------------------------------------------------------------------
def _make_render(root: Path, stem: str = "000000"):
    import cv2
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "center" / "train").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / "train" / f"{stem}.png"), np.zeros((360, 640, 3), np.uint8))
    (root / "labels" / "train" / f"{stem}.txt").write_text(_pose8_row(_INNER, _OUTER) + "\n")
    (root / "center" / "train" / f"{stem}.txt").write_text(_centre_row(_INNER) + "\n")


def _make_hand(batch: Path, stem: str = "handframe"):
    import cv2
    (batch / "frames").mkdir(parents=True, exist_ok=True)
    (batch / "labels" / "geom").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(batch / "frames" / f"{stem}.png"), np.zeros((360, 640, 3), np.uint8))
    gate = {"inner": _INNER, "outer": _OUTER, "occluded": [False] * 8, "outer_detached": [False] * 4}
    (batch / "labels" / "geom" / f"{stem}.json").write_text(
        json.dumps(labelio.encode_geometry([gate])))
    (batch / "labels" / f"{stem}.txt").write_text(labelio.encode_label([gate]))


def test_end_to_end_build_collision_safe(tmp_path):
    # two render datasets that BOTH number their frame 000000 -- the stem-collision trap.
    dsa, dsb = tmp_path / "dsA", tmp_path / "dsB"
    _make_render(dsa); _make_render(dsb)
    batch = tmp_path / "hand"
    _make_hand(batch)
    out = tmp_path / "m1"

    class Args:
        render = [str(dsa), str(dsb)]
        hand = [str(batch)]
    Args.out = str(out)
    assert m1.build(Args) == 0

    # data.yaml carries the 5-kpt contract
    yaml = (out / "data.yaml").read_text()
    assert "kpt_shape: [5, 3]" in yaml
    assert "flip_idx: [1, 0, 3, 2, 4]" in yaml

    # BOTH same-stem render frames survived, re-keyed apart (not one overwriting the other)
    train_labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert len(train_labels) == 2, [p.name for p in train_labels]
    assert {p.stem for p in train_labels} == {"dsA__000000", "dsB__000000"}
    # every train label is a well-formed 5-kpt row, and its image is mirrored beside it
    for lp in train_labels:
        assert len(lp.read_text().split()) == 5 + 3 * 5
        assert (out / "images" / "train" / (lp.stem + ".png")).exists()
        assert (out / "center" / "train" / (lp.stem + ".txt")).exists()

    # the hand frame is the VAL oracle
    val_labels = list((out / "labels" / "val").glob("*.txt"))
    assert len(val_labels) == 1 and len(val_labels[0].read_text().split()) == 5 + 3 * 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
