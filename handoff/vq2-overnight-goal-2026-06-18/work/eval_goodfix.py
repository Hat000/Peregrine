"""Accuracy-INCLUSIVE eval: the metric that actually matters for PnP.

We don't care about detection per se -- we care about getting an ACCURATE pose. So a gate-present frame
that yields no accurate fix (missed detection OR bad PnP) scores 0, in the same denominator. Reports the
"good-fix rate" = fraction of gate-present frames whose PnP range is within a tolerance of the telemetry
ground-truth range, MISS = FAIL. (Conditioning accuracy on detection -- as range-median-over-hits does --
flatters every model and hides the misses; this fixes that.)

  * task2_frames (40, single gate-0, clean approach): good-fix @ {0.5, 1, 2} m -- the PRECISION metric.
  * course_bundle (240, multi-gate): valid-fix (detect + PnP <3 m of a real gate) -- the loose "usable" bar.

Usage:  python eval_goodfix.py <weights.pt> [more_weights ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import task2_gate_pnp as T  # noqa: E402
from racer.contracts import Frame  # noqa: E402
from racer.navigator import load_track_map  # noqa: E402
from racer.vision.detector import GateDetector  # noqa: E402

# course_bundle images are gitignored -> they live in the main checkout, not the worktree.
COURSE = Path("C:/Users/Shadow/Peregrine/handoff/perception-char-2026-06-08/course_bundle")
MAP = _ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"


def _frame(fr, bundle):
    img = cv2.imread(str(bundle / fr["png"]))
    return Frame(frame_id=fr["frame_id"], sim_time_ns=fr["sim_time_ns"], image_bgr=img,
                 recv_monotonic_ns=0, jpeg_bytes=None)


def task2_goodfix(weights):
    d = json.loads((T.BUNDLE / "frames.json").read_text())
    gmap = np.asarray(d["gate0_map_ned"], float)
    det = GateDetector.load(weights, score_thresh=0.25, kpt_conf_thresh=0.5)
    errs = []
    for fr in d["frames"]:
        obs = [o for o in det.detect(_frame(fr, T.BUNDLE)) if o.corners_px.shape[0] == 4]
        gt = float(fr["range_m"]); e = np.inf
        if obs:
            R_wc = T.R_world_camera(fr["odo_q_wxyz"]); dr = np.asarray(fr["drone_position_ned"], float); best = 9e9
            for o in obs:
                t, _ = T.solve_t_known_R(o.corners_px, R_wc.T @ T._R_WORLD_GATE)
                di = float(np.linalg.norm(dr + R_wc @ t - gmap))
                if di < best:
                    best = di; e = abs(np.linalg.norm(t) - gt) if di < 3.0 else np.inf
        errs.append(e)
    return np.array(errs)


def course_validfix(weights):
    gates = {g.gate_id: g for g in load_track_map(MAP, corner_to_center=False)}
    gp = {gid: np.asarray(g.position_ned, float) for gid, g in gates.items()}
    d = json.loads((COURSE / "frames.json").read_text())
    det = GateDetector.load(weights, score_thresh=0.25, kpt_conf_thresh=0.5)
    band = [fr for fr in d["frames"] if 2 <= fr["range_m"] <= 30]; valid = 0
    for fr in band:
        g = gates[int(fr["nearest_gate_id"])]
        obs = [o for o in det.detect(_frame(fr, COURSE)) if o.corners_px.shape[0] == 4]
        if not obs:
            continue
        R_wc = T.R_world_camera(fr["odo_q_wxyz"]); dr = np.asarray(fr["drone_position_ned"], float)
        Rg = np.asarray(g.R_world_gate, float)
        if any(min(np.linalg.norm(dr + R_wc @ T.solve_t_known_R(o.corners_px, R_wc.T @ Rg)[0] - p)
                   for p in gp.values()) < 3.0 for o in obs):
            valid += 1
    return valid, len(band)


def main() -> int:
    weights = sys.argv[1:] or [str(T.WEIGHTS)]
    print("task2_frames (40 gate-present): GOOD-FIX rate, MISS = FAIL")
    print(f"  {'weights':32}{'fix<0.5m':>12}{'fix<1m':>10}{'fix<2m':>10}")
    for w in weights:
        e = task2_goodfix(w); N = len(e)
        g = lambda t: f"{int((e < t).sum())}/{N} ({100*(e < t).mean():.0f}%)"
        print(f"  {Path(w).name:32}{g(0.5):>12}{g(1.0):>10}{g(2.0):>10}")
    print("\ncourse_bundle (204 band): VALID-FIX rate (detect + PnP <3 m of a real gate), MISS = FAIL")
    for w in weights:
        v, N = course_validfix(w)
        print(f"  {Path(w).name:32}{v}/{N} ({100*v/N:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
