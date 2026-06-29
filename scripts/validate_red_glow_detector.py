"""Validate the VQ2 red-glow gate detector on a directory of recon frames.

Runs :func:`racer.vision.red_glow_detector.detect_red_glow_candidates` over every image in a
frames dir, runs PnP on the primary candidate, and prints a per-frame report: detected?
corner span (px), the SATURATED-CORE vs NAIVE-low-threshold span (to quantify the bloom-driven
apparent-size inflation that biases range NEAR), centroid, PnP range + reprojection error.

The recon frames are NOT committed to the repo (a small 4-frame subset lives under
``tests/fixtures/vq2_recon/`` for the pinned pytest). Point this at the full curated set:

    python scripts/validate_red_glow_detector.py --frames <path>/frames/curated

Default ``--frames`` is the committed fixtures dir, so it runs out-of-the-box (smaller sample).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import cv2

from racer.contracts import GateObservation
from racer.vision.gate_pose import estimate_gate_pose
from racer.vision.red_glow_detector import RedGlowParams, detect_red_glow_candidates

_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "vq2_recon")


def _naive_low_threshold_span(img: np.ndarray) -> float | None:
    """Apparent gate span from a LOW red threshold (catches bloom halo + ambient wash)."""
    b, g, r = (img[:, :, i].astype(int) for i in range(3))
    m = ((r > 120) & (r - g > 40) & (r - b > 40)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return float(max(w, h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=_DEFAULT, help="dir of recon frames (.png/.jpg)")
    args = ap.parse_args()

    params = RedGlowParams()
    files = sorted(
        f for f in glob.glob(os.path.join(args.frames, "*"))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and not os.path.basename(f).startswith("00_")
        and "contact" not in os.path.basename(f)
        and "analysis" not in os.path.basename(f)
    )
    # de-dup .png/.jpg of the same stem (prefer .png)
    seen: dict[str, str] = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem not in seen or f.lower().endswith(".png"):
            seen[stem] = f
    files = [seen[s] for s in sorted(seen)]

    hdr = f"{'frame':44s} {'ndet':>4} {'score':>5} {'span':>5} {'naive':>5} {'infl%':>5} {'PnP_m':>6} {'reproj':>6}"
    print(hdr)
    print("-" * len(hdr))
    n_detect = 0
    for fn in files:
        name = os.path.basename(fn)
        img = cv2.imread(fn)
        if img is None:
            continue
        cands = detect_red_glow_candidates(img, params)
        if not cands:
            print(f"{name:44s} {0:>4}")
            continue
        n_detect += 1
        c0 = cands[0]
        span = float(np.linalg.norm(c0.corners_px.max(0) - c0.corners_px.min(0)))
        ns = _naive_low_threshold_span(img)
        infl = (ns / span - 1.0) * 100.0 if ns else 0.0
        try:
            obs = GateObservation(
                frame_id=0, sim_time_ns=0, corners_px=c0.corners_px,
                corner_confidence=np.full(4, c0.score),
            )
            gp = estimate_gate_pose(obs)
            pnp, rep = f"{gp.range_m:6.1f}", f"{gp.reproj_error_px:6.1f}"
        except Exception as exc:  # noqa: BLE001
            pnp, rep = f"{'ERR':>6}", f"{type(exc).__name__[:6]:>6}"
        print(f"{name:44s} {len(cands):>4} {c0.score:5.2f} {span:5.0f} "
              f"{(ns or 0):5.0f} {infl:5.0f} {pnp} {rep}")
    print(f"\n{n_detect}/{len(files)} frames produced a gate candidate.")


if __name__ == "__main__":
    main()
