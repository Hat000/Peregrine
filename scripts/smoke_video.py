"""Receive video frames from the sim and write the first N to disk.

Usage:  python scripts\\smoke_video.py [n_frames]
        n_frames defaults to 30
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from racer.vision.jpeg_receiver import JpegUdpReceiver


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out_dir = Path("data/runs/smoke_video")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"listening on UDP 5600 for {n} frames. writing to {out_dir}/")
    saved = 0
    with JpegUdpReceiver() as rx:
        for frame in rx.frames():
            path = out_dir / f"frame_{frame.frame_id:06d}.jpg"
            cv2.imwrite(str(path), frame.image_bgr)
            saved += 1
            print(f"  frame {frame.frame_id}  shape={frame.image_bgr.shape}  sim_t={frame.sim_time_ns}")
            if saved >= n:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
