"""End-to-end demo: ``GateDetector.load(<spec>)`` + ``detect()`` on recorded flight frames.

Proves a weights spec (.pt, TensorRT .engine, or .onnx) runs through the EXACT flight-stack
detector path (same class, same thresholds) and yields gate observations on real recorded
frames — the smoke test for an exported engine before any bench or flight.

Run (ShadowPC — ALWAYS the .venv python):
  .venv/Scripts/python.exe scripts/demo_seeker_detect.py \
      --weights models/gate_clean_ens_course_L110_fp16_384x640.engine \
      --run-dir data/runs/20260701_224640_vq2_slow_seeker_a19c_f1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.recording import RecordingReader                        # noqa: E402
from racer.vision.detector import GateDetector                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="detector spec: .pt / .engine / .onnx")
    ap.add_argument("--run-dir", required=True, help="recorded session dir (video.bin + video_index.jsonl)")
    ap.add_argument("--n", type=int, default=5, help="print the first N frames WITH detections")
    args = ap.parse_args()

    det = GateDetector.load(args.weights)
    hits = scanned = 0
    for frame in RecordingReader(args.run_dir).frames():
        scanned += 1
        obs = det.detect(frame)
        if not obs:
            continue
        hits += 1
        print(f"frame {frame.frame_id} (t={frame.sim_time_ns / 1e9:.2f}s): {len(obs)} gate obs")
        for o in obs:
            corners = ", ".join(f"({x:.1f},{y:.1f})" for x, y in o.corners_px)
            print(f"    score={float(o.score):.3f} corners_px=[{corners}]")
        if hits >= args.n:
            break
    print(f"\n{hits} frames with detections (scanned {scanned}) -- weights={args.weights}")
    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
