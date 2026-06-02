"""Quantitative detector eval: corner-localisation error vs geometry, by curriculum level.

Replaces eyeballing a handful of demo frames with a distribution over many held-out
samples, split by whether the gate is WELL-FRAMED (all 4 inner corners in frame) or
OVERFLOW/CLIPPED (>=1 corner off-frame -> the close-range transit case). Corner error is
measured only over corners that are genuinely in-frame in the ground truth, so a clamped
off-frame label can't pollute the number.

Run:  .venv\\Scripts\\python.exe scripts\\eval_detector.py [weights] [n_per_level]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.contracts import Frame                                   # noqa: E402
from racer.frames import IMAGE_HEIGHT, IMAGE_WIDTH                  # noqa: E402
from racer.vision.detector import GateDetector                     # noqa: E402
from racer.vision.gate_pose import estimate_gate_pose              # noqa: E402
from racer.vision.synthetic import V_OFF, render_gate_sample       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# v2 (multi-gate, 6k) is the chosen deployment detector: best corner + pose error on this
# fixed-set eval (2026-06-02). v3's --hard tail-oversampling regressed the bulk without moving
# the high-roll corner-swap (~0.7%, geometric aliasing -> belongs in the solver/prior). [Peregrine]
WEIGHTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "gate_yolo11s_curriculum_v2.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> int:
    print(f"image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}   weights: {WEIGHTS.name}   n/level: {N}")
    det = GateDetector.load(WEIGHTS)
    rng = np.random.default_rng(0)

    print(f"\n{'lvl':>3} {'detect':>7} {'bucket':>10} {'n':>4} "
          f"{'med_px':>7} {'p90_px':>7} {'max_px':>7} {'med_pose':>9}")
    for level in (1, 2, 3):
        errs, n_off, dists, pose_errs, dets = [], [], [], [], 0
        made = 0
        while made < N:
            s = render_gate_sample(rng, level=level)
            if not s.visible:
                continue
            made += 1
            obs = det.detect(Frame(frame_id=made, sim_time_ns=0, image_bgr=s.image_bgr.copy()))
            if not obs:
                errs.append(np.nan); n_off.append(int((s.visibility == V_OFF).sum()))
                dists.append(float(np.linalg.norm(s.t_cam_gate))); pose_errs.append(np.nan)
                continue
            dets += 1
            o = max(obs, key=lambda z: z.score)
            ids = o.corner_ids if o.corner_ids is not None else np.arange(4)
            ids = np.asarray(ids).astype(int)
            # error only over corners that are truly in-frame in the GT
            keep = [j for j, cid in enumerate(ids) if s.visibility[cid] != V_OFF]
            if keep:
                gt = s.keypoints_px[ids[keep]]
                errs.append(float(np.mean(np.linalg.norm(o.corners_px[keep] - gt, axis=1))))
            else:
                errs.append(np.nan)
            n_off.append(int((s.visibility == V_OFF).sum()))
            dists.append(float(np.linalg.norm(s.t_cam_gate)))
            gp = estimate_gate_pose(o)
            pose_errs.append(float(np.linalg.norm(gp.t_cam_gate - s.t_cam_gate)) if gp else np.nan)

        errs = np.array(errs); n_off = np.array(n_off)
        dists = np.array(dists); pose_errs = np.array(pose_errs)
        ok = ~np.isnan(errs)
        well = ok & (n_off == 0)
        over = ok & (n_off > 0)
        for name, m in (("ALL", ok), ("well-framed", well), ("overflow", over)):
            e = errs[m]; p = pose_errs[m]; p = p[~np.isnan(p)]
            print(f"{level:>3} {dets / N * 100:>6.0f}% {name:>10} {int(m.sum()):>4} "
                  f"{pct(e,50):>7.1f} {pct(e,90):>7.1f} {pct(e,100):>7.1f} {pct(p,50)*100:>7.0f}cm")
        print(f"    dist range {dists.min():.1f}-{dists.max():.1f} m  "
              f"(overflow share {100*over.sum()/max(1,ok.sum()):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
