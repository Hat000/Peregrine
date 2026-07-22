"""Pure, importable label ENCODING for the VQ2 gate hand-labeler (NO server / browser deps).

Mirrors src/racer/vision/blender_gen/labels.py:to_yolo_pose_row EXACTLY (field order,
normalization divisors, [0,1] keypoint clamp, ``%.6g`` float formatting) so hand labels are
byte-compatible with the synthetic dataset writer. Source of truth:

  * contract.py  -- IMAGE 640x360, GATE_INNER_SIZE_M=1.5, GATE_OUTER_SIZE_M=2.72,
                    V_VIS/V_OCC/V_OFF = 2/1/0, keypoint order inner LL,LR,UR,UL then
                    outer LL,LR,UR,UL, KEYPOINT_FLIP_IDX=[1,0,3,2,5,4,7,6]
  * labels.py    -- row = ``class cx cy w h (x y v)*8``; x normalized by WIDTH, y by HEIGHT;
                    keypoints clamped into [0,1] AFTER normalization; in-frame test uses
                    inclusive bounds 0..W-1 / 0..H-1
  * gate_pose.py -- gate frame is X-right, Y-DOWN; LL,LR,UR,UL is the IPPE_SQUARE order

Unit-testable without the browser:  python labelio.py  runs a self-check.
"""
from __future__ import annotations

# --- frozen constants (contract.py) ----------------------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
GATE_INNER_SIZE_M = 1.5
GATE_OUTER_SIZE_M = 2.72
N_KEYPOINTS = 8
KEYPOINT_FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6]
V_VIS, V_OCC, V_OFF = 2, 1, 0

# Canonical gate-plane squares (metres, gate frame X-right / Y-DOWN), LL,LR,UR,UL --
# identical to gate_pose.gate_object_points ordering. Used by derive_outer_from_inner;
# the JS in index.html mirrors these EXACTLY.
_HI = GATE_INNER_SIZE_M / 2.0   # 0.75
_HO = GATE_OUTER_SIZE_M / 2.0   # 1.36
INNER_PLANE = [(-_HI, +_HI), (+_HI, +_HI), (+_HI, -_HI), (-_HI, -_HI)]  # LL,LR,UR,UL (Y-down)
OUTER_PLANE = [(-_HO, +_HO), (+_HO, +_HO), (+_HO, -_HO), (-_HO, -_HO)]


# --- row encoding (byte-identical to labels.py:to_yolo_pose_row) -----------------------

def to_yolo_pose_row(keypoints_px, visibility, bbox_xywh, class_id=0,
                     img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT) -> str:
    """One normalized YOLO-pose row: ``class cx cy w h (x y v)*N``.

    keypoints_px: [(px, py)] pixel coords (may be off-frame; clamped to [0,1] after norm).
    visibility:   [int] one of V_VIS/V_OCC/V_OFF per keypoint.
    bbox_xywh:    (x, y, w, h) pixel top-left + size, ALREADY clamped to the frame.
    """
    x, y, w, h = (float(v) for v in bbox_xywh)
    fields = [
        int(class_id),
        (x + w / 2) / img_w, (y + h / 2) / img_h,
        w / img_w, h / img_h,
    ]
    for (px, py), v in zip(keypoints_px, visibility):
        nx = min(max(float(px) / img_w, 0.0), 1.0)
        ny = min(max(float(py) / img_h, 0.0), 1.0)
        fields += [nx, ny, int(v)]
    return " ".join(f"{vv:.6g}" if isinstance(vv, float) else str(vv) for vv in fields)


def keypoint_visibility(px, py, occluded=False, img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT) -> int:
    """2 = visible in-frame, 1 = occluded-but-in-frame, 0 = off-frame.

    In-frame test matches labels.py:_outer_visibility: inclusive 0..W-1 / 0..H-1 pixel index.
    Off-frame trumps occluded.
    """
    in_frame = (0.0 <= float(px) <= img_w - 1) and (0.0 <= float(py) <= img_h - 1)
    if not in_frame:
        return V_OFF
    return V_OCC if occluded else V_VIS


def bbox_from_outer(outer_px, img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT):
    """bbox (x, y, w, h) = the OUTER-square pixel extent clamped to the frame.

    The outer square is the gate's visual extent (opaque ring = OUTER..INNER); clamping
    keeps the normalized bbox inside [0,1] even when corners hang off-frame.
    """
    xs = [float(p[0]) for p in outer_px]
    ys = [float(p[1]) for p in outer_px]
    x_min = min(max(min(xs), 0.0), img_w - 1.0)
    x_max = min(max(max(xs), 0.0), img_w - 1.0)
    y_min = min(max(min(ys), 0.0), img_h - 1.0)
    y_max = min(max(max(ys), 0.0), img_h - 1.0)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def encode_gate_row(inner_px, outer_px, occluded=None, class_id=0,
                    img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT) -> str:
    """One label row for one gate. inner_px/outer_px: 4 (x,y) pixel pairs each, LL,LR,UR,UL.
    occluded: optional 8 bools (inner 0-3 then outer 4-7); default all False."""
    if len(inner_px) != 4 or len(outer_px) != 4:
        raise ValueError("need exactly 4 inner + 4 outer corners (LL,LR,UR,UL)")
    kpts = [(float(p[0]), float(p[1])) for p in list(inner_px) + list(outer_px)]  # inner||outer
    occ = list(occluded) if occluded is not None else [False] * N_KEYPOINTS
    if len(occ) != N_KEYPOINTS:
        raise ValueError("occluded must have 8 entries (inner 0-3 then outer 4-7)")
    vis = [keypoint_visibility(px, py, o, img_w, img_h) for (px, py), o in zip(kpts, occ)]
    bbox = bbox_from_outer(outer_px, img_w, img_h)
    return to_yolo_pose_row(kpts, vis, bbox, class_id, img_w, img_h)


def encode_label(gates, img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT) -> str:
    """Full label-file content for one frame.

    gates: list of dicts, each {"inner": [(x,y)]*4, "outer": [(x,y)]*4,
                                "occluded": [bool]*8 (optional)}  -- pixel coords, LL,LR,UR,UL.
    Returns "" for a negative frame (write the EMPTY file -- do not omit it); otherwise one
    row per gate, newline-terminated.
    """
    rows = [encode_gate_row(g["inner"], g["outer"], g.get("occluded"),
                            g.get("class_id", 0), img_w, img_h) for g in gates]
    return ("\n".join(rows) + "\n") if rows else ""


def decode_label(text, img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT):
    """Parse a label file back into pixel-space gates (for resume-editing).

    Returns [{"inner": [(x,y)]*4, "outer": [(x,y)]*4, "vis": [int]*8, "class_id": int}].
    NOTE: off-frame corners were stored CLAMPED, so decoded positions sit on the frame edge.
    """
    gates = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 + 3 * N_KEYPOINTS:
            raise ValueError(f"bad row: expected {5 + 3 * N_KEYPOINTS} fields, got {len(parts)}")
        kpts, vis = [], []
        for i in range(N_KEYPOINTS):
            nx, ny, v = parts[5 + 3 * i:8 + 3 * i]
            kpts.append((float(nx) * img_w, float(ny) * img_h))
            vis.append(int(float(v)))
        gates.append({"class_id": int(parts[0]),
                      "inner": kpts[:4], "outer": kpts[4:], "vis": vis})
    return gates


# --- 4-point homography (inner clicks -> outer corners); JS mirror lives in index.html --

def _gauss_solve(a, b):
    """Solve A x = b (n x n) by Gaussian elimination with partial pivoting. Pure Python."""
    n = len(b)
    m = [list(row) + [bv] for row, bv in zip(a, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("degenerate corner configuration (singular homography system)")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def solve_homography(src_pts, dst_pts):
    """3x3 H with h8=1 mapping 4 src (X,Y) -> 4 dst (u,v):  [u,v,1]^T ~ H [X,Y,1]^T (DLT)."""
    a, b = [], []
    for (X, Y), (u, v) in zip(src_pts, dst_pts):
        a.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y]); b.append(u)
        a.append([0, 0, 0, X, Y, 1, -v * X, -v * Y]); b.append(v)
    h = _gauss_solve(a, b)
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def apply_homography(h, X, Y):
    w = h[2][0] * X + h[2][1] * Y + h[2][2]
    return ((h[0][0] * X + h[0][1] * Y + h[0][2]) / w,
            (h[1][0] * X + h[1][1] * Y + h[1][2]) / w)


# --- horizon guard ---------------------------------------------------------------------
# The outer square is 1.813x the inner one, so deriving it EXTRAPOLATES away from the clicked
# quad. Under extreme perspective that extrapolation can reach the gate plane's VANISHING LINE:
# the homogeneous w -> 0 (point at infinity) and then goes NEGATIVE (the point wraps to the
# opposite side of the image). Such a corner has NO valid pixel -- the real gate frame there is
# behind the camera plane. The homography is still exact; the RESULT is simply undefined.
#
# Unguarded this is a silent label-poisoner: a wrapped corner can land back INSIDE the frame and
# get stored v=2 (visible) at a meaningless position. The only correct label for it is
# v=0 (off-frame, masked by the pose loss), so we place it well outside the border along the
# direction of its own inner corner -- finite, renderable, and v=0 automatically.
_W_EPS = 0.08          # |w| below this = at/over the horizon (inner w's are O(0.5..1.5))
_SANE_SPAN = 4.0       # a corner further than this many image-widths out is "at infinity" too


def outer_validity(inner_px, img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT):
    """Per-outer-corner geometric validity (True = a real pixel exists for it)."""
    h = solve_homography(INNER_PLANE, inner_px)
    out = []
    for (X, Y) in OUTER_PLANE:
        w = h[2][0] * X + h[2][1] * Y + h[2][2]
        if w <= _W_EPS:
            out.append(False); continue
        x = (h[0][0] * X + h[0][1] * Y + h[0][2]) / w
        y = (h[1][0] * X + h[1][1] * Y + h[1][2]) / w
        out.append(abs(x) <= _SANE_SPAN * img_w and abs(y) <= _SANE_SPAN * img_h)
    return out


def derive_outer_from_inner(inner_px, img_w=IMAGE_WIDTH, img_h=IMAGE_HEIGHT):
    """4 OUTER corner pixels from the 4 clicked INNER pixels (LL,LR,UR,UL), via the plane
    homography canonical-inner-square -> image. Exact projective geometry: inner and outer
    squares are concentric + coplanar, so the same H projects both.

    HORIZON-GUARDED: a corner whose w is at/past the vanishing line (or that lands absurdly far
    out) has no valid pixel, so it is replaced by a finite point pushed outside the frame along
    its inner corner's direction from the quad centre. It then encodes as v=0 (off-frame) instead
    of masquerading as a visible keypoint. See the note above."""
    h = solve_homography(INNER_PLANE, inner_px)
    valid = outer_validity(inner_px, img_w, img_h)
    cx = sum(p[0] for p in inner_px) / 4.0
    cy = sum(p[1] for p in inner_px) / 4.0
    span = 1.5 * max(img_w, img_h)
    out = []
    for i, (X, Y) in enumerate(OUTER_PLANE):
        if valid[i]:
            out.append(apply_homography(h, X, Y))
            continue
        ix, iy = inner_px[i]                       # push out past this corner, away from centre
        dx, dy = ix - cx, iy - cy
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        out.append((cx + dx / n * span, cy + dy / n * span))
    return out


# --- self-check (run: python labelio.py) ------------------------------------------------
if __name__ == "__main__":
    # Head-on gate centered in frame, 100 px inner half-size -> outer half = 100*1.813...
    cx, cy, s = 320.0, 180.0, 100.0
    inner = [(cx - s, cy + s), (cx + s, cy + s), (cx + s, cy - s), (cx - s, cy - s)]
    outer = derive_outer_from_inner(inner)
    so = s * (_HO / _HI)  # 181.33...
    exp = [(cx - so, cy + so), (cx + so, cy + so), (cx + so, cy - so), (cx - so, cy - so)]
    assert all(abs(a - b) < 1e-6 for p, q in zip(outer, exp) for a, b in zip(p, q)), outer
    row = encode_gate_row(inner, outer)
    parts = row.split()
    assert len(parts) == 29 and parts[0] == "0"
    assert parts[1] == "0.5"                                 # centered horizontally
    # vertical: outer extent clamps to [0, 359] -> cy = (0 + 359/2)/360, NOT exactly 0.5
    assert abs(float(parts[2]) - (359.0 / 2) / 360.0) < 1e-6
    assert parts[7] == "2"                                   # inner LL visible
    # outer corners are off-frame vertically (cy±181 outside 0..359) -> v=0, coords clamped
    assert parts[5 + 3 * 4 + 2] == "0", row   # outer LL (kpt 4) visibility
    txt = encode_label([{"inner": inner, "outer": outer}])
    back = decode_label(txt)
    assert len(back) == 1 and back[0]["vis"][:4] == [2, 2, 2, 2]
    assert encode_label([]) == ""                            # negative frame = empty file

    # --- horizon guard (regression: the near-edge-on gate that produced a 12000 px outer) ------
    # Extreme perspective: the outer square reaches the gate plane's vanishing line, so at least
    # one outer corner has NO valid pixel. It must be pushed off-frame and encode v=0 -- never
    # wrap back inside the image and get stored as visible.
    edge_on = [(247.5, 275.7), (297.8, 547.3), (523.1, 24.1), (317.9, 24.1)]  # LL,LR,UR,UL
    val = outer_validity(edge_on)
    assert not all(val), f"expected an invalid outer corner, got {val}"
    o2 = derive_outer_from_inner(edge_on)
    for i, ok in enumerate(val):
        if not ok:
            x, y = o2[i]
            assert abs(x) < 1e4 and abs(y) < 1e4, f"guarded corner still absurd: {o2[i]}"
            assert keypoint_visibility(x, y) == V_OFF, f"invalid outer {i} must encode v=0"
    row2 = encode_gate_row(edge_on, o2).split()
    for i, ok in enumerate(val):                 # visibility field of outer keypoint i (kpt 4+i)
        if not ok:
            assert row2[5 + 3 * (4 + i) + 2] == "0", row2
    print("labelio self-check OK")
    print(row)
