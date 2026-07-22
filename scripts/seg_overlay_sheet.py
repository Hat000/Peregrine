"""Render a contact sheet of the WRITTEN seg polygons on the WRITTEN images -- the label arbiter.

Every label bug in this project was found by looking at an overlay, never by a census: the Blender
generator's unlabelled-near-gate bug survived a sampler that reported 2.58 gates/frame while disk
said 1.00, and two silent dataset bugs (stem collisions, images/->labels/ path resolution) were only
caught by rendering. So this reads BACK OFF DISK -- image file and seg file, paired by stem, nothing
in memory -- and draws exactly what a trainer would consume.

  class 0 gate_frame   -> cyan outline + translucent fill, vertices dotted
  class 1 gate_opening -> yellow outline + translucent fill

Also prints the gates/frame signature computed off disk (rows per file), because a collapse toward
1.0 is the fingerprint of gates being dropped again.

Run:
  python scripts/seg_overlay_sheet.py --data <dataset_dir> [--split train] [--limit 24]
                                      [--cols 4] [--out sheet.png] [--sort-by-gates]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_CLASS_BGR = {0: (255, 255, 0), 1: (0, 220, 255)}      # 0 frame = cyan, 1 opening = yellow
_CLASS_NAME = {0: "gate_frame", 1: "gate_opening"}


def parse_seg(text: str, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    """``class x1 y1 x2 y2 ...`` (normalised) -> [(class_id, (n,2) pixel polygon)]."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 7 or (len(f) - 1) % 2:
            continue
        pts = np.array([float(v) for v in f[1:]], dtype=float).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        out.append((int(float(f[0])), pts))
    return out


def draw(img: np.ndarray, polys: list[tuple[int, np.ndarray]]) -> np.ndarray:
    vis = img.copy()
    fill = img.copy()
    for cid, pts in polys:
        col = _CLASS_BGR.get(cid, (200, 200, 200))
        ip = pts.round().astype(np.int32)
        cv2.fillPoly(fill, [ip], col)
        cv2.polylines(vis, [ip], True, col, 1, cv2.LINE_AA)
    vis = cv2.addWeighted(fill, 0.28, vis, 0.72, 0.0)
    for cid, pts in polys:                              # vertices ON TOP of the blend, so a
        col = _CLASS_BGR.get(cid, (200, 200, 200))      # straightened-into-a-quad polygon is
        for x, y in pts:                                # obvious at a glance
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, col, -1, cv2.LINE_AA)
    return vis


def contact_sheet(tiles: list[np.ndarray], cols: int, pad: int = 4) -> np.ndarray:
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (th + pad) + pad, cols * (tw + pad) + pad, 3), 24, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        sheet[y:y + th, x:x + tw] = t
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description="seg-label overlay contact sheet (reads off disk)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sort-by-gates", action="store_true",
                    help="show the busiest (most instances) frames first, not the first N")
    args = ap.parse_args()

    root = Path(args.data)
    img_dir, seg_dir = root / "images" / args.split, root / "seg" / args.split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if not imgs:
        print(f"no images under {img_dir}")
        return 2

    # ---- gates/frame signature, straight off disk -------------------------------------------
    per_frame, n_frame_rows, n_open_rows, missing = [], 0, 0, 0
    for p in imgs:
        s = seg_dir / f"{p.stem}.txt"
        if not s.exists():
            missing += 1
            continue
        cls = [int(float(l.split()[0])) for l in s.read_text().splitlines() if l.strip()]
        per_frame.append(cls.count(0))
        n_frame_rows += cls.count(0)
        n_open_rows += cls.count(1)
    arr = np.array(per_frame, dtype=float) if per_frame else np.zeros(1)
    print(f"[seg] {len(imgs)} images, {len(per_frame)} seg files ({missing} missing)")
    print(f"[seg] gates/frame (class-0 rows): mean={arr.mean():.3f} median={np.median(arr):.1f} "
          f"min={arr.min():.0f} max={arr.max():.0f}  zero-gate frames={int((arr == 0).sum())}")
    print(f"[seg] rows: gate_frame={n_frame_rows} gate_opening={n_open_rows} "
          f"(openings per frame-instance = {n_open_rows / max(n_frame_rows, 1):.3f})")
    src = root / "seg" / "SOURCE.txt"
    if src.exists():
        print(f"[seg] SOURCE.txt: {src.read_text().strip().splitlines()}")

    # ---- the sheet ---------------------------------------------------------------------------
    order = list(range(len(imgs)))
    if args.sort_by_gates and per_frame:
        order.sort(key=lambda i: -(per_frame[i] if i < len(per_frame) else 0))
    tiles = []
    for i in order[:args.limit]:
        p = imgs[i]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = seg_dir / f"{p.stem}.txt"
        polys = parse_seg(s.read_text(), w, h) if s.exists() else []
        vis = draw(img, polys)
        n0 = sum(1 for c, _ in polys if c == 0)
        n1 = sum(1 for c, _ in polys if c == 1)
        npts = sum(len(q) for _, q in polys)
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
        cv2.putText(vis, f"{p.stem}  f{n0}/o{n1}  {npts}pt", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"{p.stem}  f{n0}/o{n1}  {npts}pt", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        if args.scale != 1.0:
            vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        tiles.append(vis)

    sheet = contact_sheet(tiles, max(1, args.cols))
    out = Path(args.out) if args.out else root / f"seg_overlay_{args.split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"[seg] contact sheet ({len(tiles)} tiles) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
