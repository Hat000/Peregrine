"""Local demo of the Adroit smoke-test detector.

Reproduces, on this laptop, what the smoke run on Adroit produced: it generates synthetic
gates (curriculum L1/L2/L3), runs the trained YOLO-pose weights through our REAL SENSE seam
(GateDetector -> GateObservation -> estimate_gate_pose), and visualises predicted vs ground-
truth corners + the recovered camera-relative pose.

The smoke model was YOLO11n trained on L2 only for 5 epochs, so expect it to nail L1/L2 and
visibly struggle on L3 "chaos" -- which is exactly why the real run uses the full curriculum.

Run:  .venv\\Scripts\\python.exe scripts\\demo_smoke.py
Needs the [detector] extra (ultralytics). Writes annotated PNGs to demo/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.contracts import Frame                                  # noqa: E402
from racer.vision.detector import GateDetector                     # noqa: E402
from racer.vision.gate_pose import _rotation_geodesic, estimate_gate_pose  # noqa: E402
from racer.vision.synthetic import render_gate_sample              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "gate_yolo11n_smoke.pt"
OUT = ROOT / "demo"
GREEN, RED, CYAN, WHITE = (0, 200, 0), (0, 0, 230), (255, 255, 0), (255, 255, 255)


def draw_quad(img, pts, color, label=False):
    pi = [tuple(np.round(p).astype(int)) for p in np.asarray(pts)]
    for a, b in zip(pi, pi[1:] + pi[:1]):
        cv2.line(img, a, b, color, 2)
    for j, p in enumerate(pi):
        cv2.circle(img, p, 5, color, -1)
        if label:
            cv2.putText(img, str(j), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main() -> int:
    if not WEIGHTS.exists():
        sys.exit(f"weights not found: {WEIGHTS}")
    print(f"loading trained weights: {WEIGHTS.name}")
    det = GateDetector.load(WEIGHTS)
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(123)

    summary = {}
    for level in (1, 2, 3):
        rows = []
        for i in range(4):
            s = render_gate_sample(rng, level=level)
            if not s.visible:
                continue
            frame = Frame(frame_id=i, sim_time_ns=0, image_bgr=s.image_bgr.copy())
            obs_list = det.detect(frame)                          # the real adapter
            img = s.image_bgr.copy()
            draw_quad(img, s.keypoints_px, GREEN, label=True)     # ground truth

            row = {"detected": bool(obs_list)}
            if obs_list:
                obs = max(obs_list, key=lambda o: o.score)
                ids = obs.corner_ids if obs.corner_ids is not None else np.arange(4)
                draw_quad(img, obs.corners_px, RED)               # prediction
                gt = s.keypoints_px[np.asarray(ids).astype(int)]
                row["corner_err_px"] = float(np.mean(np.linalg.norm(obs.corners_px - gt, axis=1)))
                row["n_corners"] = int(obs.corners_px.shape[0])
                gp = estimate_gate_pose(obs)                      # PnP on the PREDICTED corners
                txt = f"L{level}  conf={obs.score:.2f}  corner_err={row['corner_err_px']:.1f}px  ({row['n_corners']} corners)"
                if gp is not None:
                    row["pos_err_m"] = float(np.linalg.norm(gp.t_cam_gate - s.t_cam_gate))
                    row["rot_err_deg"] = float(np.degrees(_rotation_geodesic(gp.R_cam_gate, s.R_cam_gate)))
                    txt += f"  ->  pose dt={row['pos_err_m'] * 100:.0f}cm dr={row['rot_err_deg']:.1f}deg @ {np.linalg.norm(s.t_cam_gate):.1f}m"
            else:
                txt = f"L{level}  MISSED (no detection)"
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv2.putText(img, "green = ground truth    red = detector prediction", (8, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
            cv2.imwrite(str(OUT / f"L{level}_{i}.png"), img)
            rows.append(row)
        summary[level] = rows

    print(f"\n=== DETECTOR DEMO  ({WEIGHTS.name}) ===")
    for level in (1, 2, 3):
        rows = summary[level]
        n = len(rows)
        ndet = sum(r["detected"] for r in rows)
        ce = [r["corner_err_px"] for r in rows if "corner_err_px" in r]
        pe = [r["pos_err_m"] for r in rows if "pos_err_m" in r]
        line = f"L{level}: detected {ndet}/{n}"
        if ce:
            line += f" | mean corner err {np.mean(ce):4.1f}px"
        if pe:
            line += f" | mean pose err {np.mean(pe) * 100:4.0f}cm"
        print(line)
    print(f"\nannotated images -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
