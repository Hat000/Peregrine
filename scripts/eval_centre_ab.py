"""A/B the two mask -> gate-centre solvers on HAND-LABELLED REAL FRAMES.

The question: given the same predicted seg masks, does fitting a QUAD to the whole mask boundary
and intersecting its diagonals beat fitting LINES to contour segments and solving a homography?

GROUND TRUTH is the diagonal intersection of the hand-labelled INNER quad, read from the labeler's
exact unclamped geometry sidecar (labels/geom/*.json), not from the pose row -- the pose row clamps
off-frame corners onto the border, which would destroy exactly the cropped gates this is about.
Using the diagonal formula for the truth as well is a DEFINITION, not circularity: the GT quad is
four human-placed handles, the prediction is a machine-fitted quad from a mask, and the formula in
between is exact projective geometry that adds no free parameters to either side.

⚠ THE MATCHER IS THE EASIEST THING TO GET WRONG HERE, and getting it wrong invents error out of
nothing. An earlier harness matched every GT centre to its NEAREST prediction: on a frame with 3
labelled gates and 1 prediction, all three matched that one prediction and two of them scored as
huge misses that no solver could have avoided. Matching here is GREEDY ONE-TO-ONE -- sort every
(gt, pred) pair by distance and consume both sides -- and COVERAGE (how many labelled gates got a
prediction at all) is reported SEPARATELY from the error over matched pairs, because they are
different failure modes and averaging them together hides both.

Both arms are fed the IDENTICAL masks: inference runs once per frame per model and both solvers
consume the same ``pair_gate_instances`` output. Anything that differs is the solver.

Usage:
  python scripts/eval_centre_ab.py                       # both models, table + gallery
  python scripts/eval_centre_ab.py --no-gallery --device cpu
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "gate_labeler"))

from racer.vision.gate_lines import (  # noqa: E402
    SegGateLineDetector,
    _on_border,            # private on purpose, but the gallery must colour clipped fits with the
                           # SAME rule the solver used -- a second copy would drift
    centre_from_masks_quad,
    centre_from_seg_masks,
    pair_gate_instances,
)

DEFAULT_BATCH = "C:/Users/Shadow/vq2_label_batch_2026-07-22"
DEFAULT_VAL = ("C:/Users/Shadow/AppData/Local/Temp/claude/C--Users-Shadow-Peregrine/"
               "1b6563db-3b60-4944-905f-3646a289f780/scratchpad/real_val.txt")
DEFAULT_OUT = "C:/Users/Shadow/centre_ab_2026-07-22"
WEIGHTS = {
    "v2": "C:/Users/Shadow/Peregrine/runs/segment/gate_seg_v2/weights/best.pt",
    "s1": "C:/Users/Shadow/Peregrine/runs/segment/gate_seg_s1/weights/best.pt",
}
PAD_X, PAD_Y = 160, 90          # gallery canvas margin, so OFF-SCREEN centres are still drawable


def quad_centre(q):
    """Diagonal intersection of a 4-point quad -- the definition of a projected square's centre."""
    q = np.asarray(q, float)
    d1 = np.cross([q[0][0], q[0][1], 1.0], [q[2][0], q[2][1], 1.0])
    d2 = np.cross([q[1][0], q[1][1], 1.0], [q[3][0], q[3][1], 1.0])
    p = np.cross(d1, d2)
    return None if abs(p[2]) < 1e-12 else p[:2] / p[2]


def load_truth(batch: Path, stems):
    """[(stem, image_path, [centre_px], [inner_quad], [above_area_floor])] per val stem with geometry.

    Read through the labeler's own ``decode_geometry`` so the truth is the EXACT unclamped handles.
    The pose .txt rows would clamp off-frame corners onto the border, which is precisely the
    information the cropped-gate case turns on."""
    from labelio import decode_geometry            # the labeler's own decoder, not a re-derivation
    from racer.vision.seg_labels import clip_polygon, polygon_area

    out, missing = [], []
    for s in stems:
        gp = batch / "labels" / "geom" / f"{s}.json"
        img = next((p for p in (batch / "frames" / f"{s}{e}" for e in (".png", ".jpg", ".jpeg"))
                    if p.exists()), None)
        if not gp.exists() or img is None:
            missing.append(s)
            continue
        gates = decode_geometry(json.loads(gp.read_text()))
        cs, qs, big = [], [], []
        for g in gates:
            c = quad_centre(g["inner"])
            if c is not None and np.isfinite(c).all():
                cs.append(c)
                qs.append(np.asarray(g["inner"], float))
                big.append(polygon_area(clip_polygon(g["inner"], 640, 360)) >= 200.0)
        if cs:
            out.append((s, img, cs, qs, big))
        else:
            missing.append(s)                      # a negative frame: labelled, but no gate in it
    return out, missing


def greedy_match(gt_pts, pred_pts):
    """[(gt_i, pred_j, dist)] -- ONE-TO-ONE, cheapest pair first. See the module note: a nearest-
    neighbour matcher lets one prediction absorb several GT gates and manufactures huge errors."""
    pairs = sorted((float(np.linalg.norm(np.asarray(g) - np.asarray(p))), i, j)
                   for i, g in enumerate(gt_pts) for j, p in enumerate(pred_pts))
    used_g, used_p, out = set(), set(), []
    for d, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, d))
    return out


class Arm:
    """One solver's tally. Errors and coverage are kept apart on purpose.

    Coverage is also tracked over the SOLVABLE subset -- labelled gates whose visible inner square
    clears the 200 px^2 area floor both solvers apply to a mask. 10 of the 54 labelled gates are
    below it (distant gates a few pixels across), so a headline coverage number alone reads as a
    solver failure when it is really a floor that neither arm was ever going to clear."""

    def __init__(self, name):
        self.name = name
        self.n_gt = self.n_pred = self.n_offscreen = self.n_big = self.n_big_hit = 0
        self.err = []                    # matched pairs only
        self.err_gt_offscreen = []       # subset where the LABELLED centre is outside the image

    def add_frame(self, gt_pts, preds, wh, big):
        self.n_gt += len(gt_pts)
        self.n_pred += len(preds)
        self.n_big += sum(big)
        w, h = wh
        for c in preds:
            if not (0 <= c[0] < w and 0 <= c[1] < h):
                self.n_offscreen += 1
        per_gt = {}
        for i, j, d in greedy_match(gt_pts, preds):
            self.err.append(d)
            per_gt[i] = d
            self.n_big_hit += bool(big[i])
            g = gt_pts[i]
            if not (0 <= g[0] < w and 0 <= g[1] < h):
                self.err_gt_offscreen.append(d)
        return per_gt

    def row(self):
        e = np.array(self.err) if self.err else np.array([np.nan])
        cov = 100.0 * len(self.err) / max(self.n_gt, 1)
        big = 100.0 * self.n_big_hit / max(self.n_big, 1)
        return (self.name, f"{len(self.err)}/{self.n_gt}", f"{cov:5.1f}%", f"{big:5.1f}%",
                f"{np.median(e):7.1f}", f"{np.percentile(e, 90):7.1f}",
                f"{self.n_pred:5d}", f"{self.n_offscreen:4d}")


def run_model(tag, weights, truth, device, strict_quad=False):
    """Score every arm for one seg model. Inference runs ONCE per frame; the arms share its masks."""
    det = SegGateLineDetector.load(weights, device=device)
    arms = {
        f"{tag} line-solver (baseline)": Arm(f"{tag} line-solver (baseline)"),
        f"{tag} quad+diagonal (new)": Arm(f"{tag} quad+diagonal (new)"),
    }
    if strict_quad:
        n = f"{tag} quad, no clipped fits"
        arms[n] = Arm(n)
    per_frame = []
    for stem, img_path, gt_pts, gt_quads, big in truth:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        pairs = pair_gate_instances(*det.masks_for(im))
        base, new, strict = [], [], []
        for f, o in pairs:
            got = centre_from_seg_masks(f, o, image_wh=(w, h))
            if got is not None:
                base.append(got[0])
            got = centre_from_masks_quad(f, o, image_wh=(w, h))
            if got is not None:
                new.append(got)
            if strict_quad:
                got = centre_from_masks_quad(f, o, image_wh=(w, h), max_border_edges=0)
                if got is not None:
                    strict.append(got[0])
        d_base = arms[f"{tag} line-solver (baseline)"].add_frame(gt_pts, base, (w, h), big)
        d_new = arms[f"{tag} quad+diagonal (new)"].add_frame(
            gt_pts, [c for c, _ in new], (w, h), big)
        if strict_quad:
            arms[f"{tag} quad, no clipped fits"].add_frame(gt_pts, strict, (w, h), big)
        per_frame.append({"stem": stem, "img": img_path, "im": im, "pairs": pairs,
                          "gt": gt_pts, "gt_quads": gt_quads,
                          "base": base, "new": new, "d_base": d_base, "d_new": d_new})
    return arms, per_frame


# --- gallery ----------------------------------------------------------------------------------
# Drawn on a PADDED canvas: a centre that legitimately sits outside the image is the whole point of
# this path, and a gallery that clips it to the border shows the one thing that is not happening.

def _canvas(im):
    h, w = im.shape[:2]
    c = np.full((h + 2 * PAD_Y, w + 2 * PAD_X, 3), 28, np.uint8)
    c[PAD_Y:PAD_Y + h, PAD_X:PAD_X + w] = im
    cv2.rectangle(c, (PAD_X - 1, PAD_Y - 1), (PAD_X + w, PAD_Y + h), (70, 70, 70), 1)
    return c


def _pt(p):
    return (int(round(p[0])) + PAD_X, int(round(p[1])) + PAD_Y)


def _cross(canvas, p, colour, r=9, t=2):
    x, y = _pt(p)
    hh, ww = canvas.shape[:2]
    if not (-4 * r <= x < ww + 4 * r and -4 * r <= y < hh + 4 * r):
        return False
    cv2.line(canvas, (x - r, y), (x + r, y), colour, t, cv2.LINE_AA)
    cv2.line(canvas, (x, y - r), (x, y + r), colour, t, cv2.LINE_AA)
    return True


def _poly(canvas, q, colour, t=1):
    p = [_pt(v) for v in q]
    for i in range(len(p)):
        cv2.line(canvas, p[i], p[(i + 1) % len(p)], colour, t, cv2.LINE_AA)


def _mask_outline(canvas, mask, colour):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(canvas, [c + np.array([[PAD_X, PAD_Y]])], -1, colour, 1, cv2.LINE_AA)


def _panel(rec, which):
    c = _canvas(rec["im"])
    for f, o in rec["pairs"]:
        _mask_outline(c, f, (90, 90, 60))
        if o is not None:
            _mask_outline(c, o, (120, 110, 40))
    for q in rec["gt_quads"]:
        _poly(c, q, (170, 170, 170))
    for g in rec["gt"]:
        _cross(c, g, (255, 255, 255), r=11, t=2)
    if which == "new":
        for cen, quad in rec["new"]:
            # ORANGE = the quad still has an image-border edge, i.e. only 3 gate edges were visible
            # and this is a fit to the crop, biased inward. Magenta = all four edges are gate edges.
            clipped = any(_on_border(quad[i], quad[(i + 1) % 4], *rec["im"].shape[1::-1])
                          for i in range(4))
            col = (0, 165, 255) if clipped else (255, 0, 255)
            _poly(c, quad, col, 2)
            _cross(c, cen, col)
    else:
        for cen in rec["base"]:
            _cross(c, cen, (255, 0, 255))
    label = "existing: lines -> homography" if which == "base" else "new: hull quad -> diagonals"
    cv2.putText(c, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return c


def _score(rec):
    """Worst-first key for the NEW arm: unmatched labelled gates rank above any finite error."""
    n_unmatched = len(rec["gt"]) - len(rec["d_new"])
    worst = max(rec["d_new"].values()) if rec["d_new"] else 0.0
    return (-n_unmatched, -worst)


def write_gallery(out_dir: Path, per_frame, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "img").mkdir(exist_ok=True)
    recs = sorted(per_frame, key=_score)
    cards = []
    for k, rec in enumerate(recs):
        side = np.concatenate([_panel(rec, "base"),
                               np.full((_panel(rec, "base").shape[0], 4, 3), 60, np.uint8),
                               _panel(rec, "new")], axis=1)
        name = f"{k:03d}_{rec['stem'][:70]}.jpg"
        cv2.imwrite(str(out_dir / "img" / name), side, [cv2.IMWRITE_JPEG_QUALITY, 88])
        nb = len(rec["gt"]) - len(rec["d_new"])
        nbb = len(rec["gt"]) - len(rec["d_base"])
        cards.append(
            f'<div class=c><img src="img/{html.escape(name)}" loading=lazy>'
            f'<div class=m><b>{html.escape(rec["stem"])}</b><br>'
            f'{len(rec["gt"])} labelled gate(s) &middot; '
            f'<span class=n>new</span> err '
            f'{", ".join(f"{v:.0f}" for v in rec["d_new"].values()) or "-"} px'
            f'{f" &middot; {nb} unmatched" if nb else ""}<br>'
            f'<span class=b>existing</span> err '
            f'{", ".join(f"{v:.0f}" for v in rec["d_base"].values()) or "-"} px'
            f'{f" &middot; {nbb} unmatched" if nbb else ""}</div></div>')
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>centre A/B " + tag + "</title>"
        "<style>body{background:#141414;color:#ddd;font:14px/1.45 system-ui,sans-serif;margin:24px}"
        "h1{font-size:18px;font-weight:600}p{color:#999;max-width:70em}"
        ".c{margin:0 0 26px}img{width:100%;max-width:1600px;display:block;border:1px solid #333}"
        ".m{color:#aaa;padding:6px 2px}.n{color:#f4f}.b{color:#8cf}"
        "code{color:#ccc}</style>"
        f"<h1>gate-centre A/B &mdash; {html.escape(tag)}, worst-first for the NEW arm</h1>"
        "<p>Left: the existing line/homography solver. Right: the new convex-hull quad, whose "
        "outline is drawn too. <b>White cross</b> = hand-labelled ground truth (diagonal "
        "intersection of the labelled inner quad), <b>white outline</b> = that labelled quad, "
        "<b>magenta</b> = the solver's centre. Olive outlines are the predicted masks BOTH arms "
        "were fed. The frame is inset in a padded canvas so a centre that legitimately falls "
        "outside the image is still drawn where it actually is.</p>"
        "<p><b style='color:#fa5'>Orange</b> on the right marks a quad that still has an image "
        "border edge: only three gate edges were visible, so nothing could be extended and the fit "
        "is the visible trapezoid, biased toward the middle of the frame. That is the new method's "
        "one structural weakness &mdash; look at these first.</p>"
        + "".join(cards), encoding="utf-8")
    return out_dir / "index.html"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=DEFAULT_BATCH)
    ap.add_argument("--val", default=DEFAULT_VAL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default=None, help="'cpu' to stay off a busy GPU")
    ap.add_argument("--models", default="v2,s1")
    ap.add_argument("--gallery-model", default="v2")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--strict-quad", action="store_true",
                    help="also score a variant that refuses border-clipped quads (diagnostic)")
    args = ap.parse_args()

    batch = Path(args.batch)
    stems = [s.strip() for s in Path(args.val).read_text().splitlines() if s.strip()]
    truth, missing = load_truth(batch, stems)
    n_gates = sum(len(c) for _, _, c, _, _ in truth)
    n_off = sum(1 for _, _, cs, _, _ in truth for c in cs
                if not (0 <= c[0] < 640 and 0 <= c[1] < 360))
    n_small = sum(1 for _, _, _, _, b in truth for v in b if not v)
    print(f"{len(truth)}/{len(stems)} val frames carry at least one labelled gate "
          f"({len(missing)} skipped: no geometry sidecar, or a negative frame), "
          f"{n_gates} labelled gates.\n"
          f"  {n_off} have an OFF-SCREEN centre; {n_small} are below the 200 px^2 mask-area floor "
          f"both solvers apply (a few pixels across).\n")

    rows, gallery = [], None
    for tag in [t.strip() for t in args.models.split(",") if t.strip()]:
        arms, per_frame = run_model(tag, WEIGHTS[tag], truth, args.device, args.strict_quad)
        rows += [a.row() for a in arms.values()]
        offs = arms[f"{tag} quad+diagonal (new)"].err_gt_offscreen
        base_offs = arms[f"{tag} line-solver (baseline)"].err_gt_offscreen
        print(f"[{tag}] on the {len(offs)}/{len(base_offs)} matched gates whose LABELLED centre is "
              f"off-screen: new med {np.median(offs) if offs else float('nan'):.1f} px, "
              f"baseline med {np.median(base_offs) if base_offs else float('nan'):.1f} px")
        if not args.no_gallery and tag == args.gallery_model:
            gallery = write_gallery(Path(args.out), per_frame, tag)

    hdr = ("arm", "matched", "cov", "cov>=floor", "med px", "p90 px", "preds", "off-sc")
    wid = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    line = "  ".join("-" * w for w in wid)
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(hdr, wid)))
    print(line)
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, wid)))
    print(line)
    print("cov = labelled gates that got a ONE-TO-ONE matched prediction; cov>=floor is the same "
          "over gates\nabove the 200 px^2 area floor; med/p90 are over MATCHED pairs only.\n"
          "preds = centres emitted over all frames; off-sc = those that land outside the image.")
    if gallery:
        print(f"\ngallery: {gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
