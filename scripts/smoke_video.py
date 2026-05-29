"""Receive video frames from the sim and write the first N to disk.

Usage:  python scripts\\smoke_video.py [n_frames] [idle_timeout_s]
        n_frames defaults to 30, idle_timeout_s to 10.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    out_dir = Path("data/runs/smoke_video")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"listening on UDP {VIDEO_PORT} for {n} frames (idle timeout {timeout:g}s). writing to {out_dir}/")
    saved = 0
    # Bounded wait: fail fast if the sim is down or the port is blocked instead of hanging.
    with JpegUdpReceiver() as rx:
        for frame in rx.frames(max_wait_s=timeout):
            path = out_dir / f"frame_{frame.frame_id:06d}.jpg"
            cv2.imwrite(str(path), frame.image_bgr)
            saved += 1
            print(f"  frame {frame.frame_id}  shape={frame.image_bgr.shape}  sim_t={frame.sim_time_ns}")
            if saved >= n:
                break
    if saved == 0:
        print(
            f"ERROR: no frames within {timeout:g}s. Is the sim running and streaming "
            f"JPEG/UDP on port {VIDEO_PORT}?",
            file=sys.stderr,
        )
        return 1
    if saved < n:
        print(f"WARNING: stream went idle after {saved}/{n} frames.", file=sys.stderr)
    print(f"saved {saved} frames to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
