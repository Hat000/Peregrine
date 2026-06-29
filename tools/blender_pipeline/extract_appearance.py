"""Re-extract the VQ2 appearance numbers from the recon frames (provenance for APPEARANCE_SPEC).

Run against the recon handoff dir to regenerate the quantitative appearance characterization
(gate emissive core RGB, asymmetric bloom falloff, scene brightness, lane-line / grid /
structure colors). Pure analysis -- reads the curated frames + frame_stats_*.json, prints a
report and (optionally) writes the machine-readable numbers. No Blender, no model.

    python tools/blender_pipeline/extract_appearance.py \
        --recon handoff/vq2-recon-2026-06-29

This is the script cited in APPEARANCE_SPEC.md §7 so the spec numbers stay reproducible.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc


def _median_rgb(img_bgr: np.ndarray, mask: np.ndarray) -> list[float] | None:
    if mask.sum() == 0:
        return None
    return [float(v) for v in np.median(img_bgr[mask], axis=0)[::-1]]  # BGR -> RGB


def extract_gate_core(img_bgr: np.ndarray) -> dict:
    """Saturated emissive core color + bloom asymmetry from one near-gate frame."""
    b, g, r = (img_bgr[:, :, i].astype(int) for i in range(3))
    core = (r >= 250) & ((r - g) >= 80) & ((r - b) >= 80)
    plateau = (r >= 252) & ((r - g) >= 120) & ((r - b) >= 120)
    out = {
        "core_px": int(core.sum()),
        "core_median_rgb": _median_rgb(img_bgr, core),
        "plateau_median_rgb": _median_rgb(img_bgr, plateau),
    }
    # asymmetric bloom: red profile along the centroid row, both directions from the core band
    if core.sum() > 0:
        ys, xs = np.where(core)
        row = int(round(ys.mean()))
        redrow = r[row]
        above = np.where(redrow >= 250)[0]
        if len(above):
            L, R = int(above.min()), int(above.max())

            def halfmax(start, step):
                x = start
                while 0 <= x < len(redrow) and redrow[x] >= 128:
                    x += step
                return abs(x - start)

            out["bloom_left_halfmax_px"] = halfmax(L - 1, -1)
            out["bloom_right_halfmax_px"] = halfmax(R + 1, 1)
            out["core_span_px"] = R - L
    return out


def extract_lane_lines(img_bgr: np.ndarray) -> dict:
    b, g, r = (img_bgr[:, :, i].astype(int) for i in range(3))
    lane = (b >= 110) & ((b - r) >= 40) & (b >= g)
    core = (b >= 200) & ((b - r) >= 60) & (b >= g - 10)
    return {"lane_median_rgb": _median_rgb(img_bgr, lane),
            "lane_core_median_rgb": _median_rgb(img_bgr, core)}


def extract_structure(img_bgr: np.ndarray) -> dict:
    b, g, r = (img_bgr[:, :, i].astype(int) for i in range(3))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    green = (g >= 120) & ((g - r) >= 50) & ((g - b) >= 30)
    ceil = (gray > 180)
    dark = (gray > 5) & (gray < 40)
    return {"green_marker_median_rgb": _median_rgb(img_bgr, green),
            "ceiling_light_median_rgb": _median_rgb(img_bgr, ceil),
            "dark_structure_median_rgb": _median_rgb(img_bgr, dark)}


def aggregate_frame_stats(recon_dir: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(recon_dir, "logs", "frame_stats_*.json"))):
        d = json.load(open(f))
        keys = ["mean_gray", "median_gray", "p95_gray", "frac_dark_lt32",
                "frac_bright_gt224", "mean_r", "mean_g", "mean_b", "red_glow_frac"]
        agg = {}
        for k in keys:
            v = np.array([x[k] for x in d], float)
            agg[k] = {"mean": float(v.mean()), "median": float(np.median(v))}
        out[os.path.basename(f)] = {"n": len(d), **agg}
    return out


def main() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"extract_appearance requires OpenCV: {_CV2_ERR}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True, help="path to handoff/vq2-recon-2026-06-29")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    fc = os.path.join(args.recon, "frames", "curated")
    report = {"frame_stats": aggregate_frame_stats(args.recon), "frames": {}}
    near = os.path.join(fc, "01_gate_deadahead_near_static_R1G7.png")
    start = os.path.join(fc, "02_gate_deadahead_from_startpad.png")
    if os.path.exists(near):
        img = cv2.imread(near)
        report["frames"]["gate_core"] = extract_gate_core(img)
        report["frames"]["structure"] = extract_structure(img)
    if os.path.exists(start):
        report["frames"]["lane_lines"] = extract_lane_lines(cv2.imread(start))

    print(json.dumps(report, indent=2))
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
