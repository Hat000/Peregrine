"""Offline latency + dedup-collapse measurement for the opt-in ensemble (objective E).

NOT a flight / not wiring the live stack -- it only times ``detect()`` on 40 recorded frames
through the WORKTREE's detector with the real weights, to report the deploy latency number and
confirm the per-gate dedup collapse. Run on the GPU venv:

  PYTHONPATH unused -- the script forces the worktree src onto sys.path[0] itself.
  C:/Users/Shadow/vq2yolo-venv/Scripts/python.exe handoff/vq2-ensemble-support-2026-06-19/measure_latency.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]          # worktree root
ROOT = Path("C:/Users/Shadow/Peregrine")          # main checkout: weights + recorded frames
sys.path.insert(0, str(WT / "src"))               # force the worktree's racer.* first

import cv2  # noqa: E402
from racer.contracts import Frame  # noqa: E402
import racer.vision.detector as _d  # noqa: E402
from racer.vision.detector import EnsembleGateDetector, GateDetector  # noqa: E402

assert "worktrees" in _d.__file__, f"imported the wrong detector: {_d.__file__}"

CHAMP = str(ROOT / "runs/pose/runs/vq2_pose_8kp/weights/best.pt")
M = ROOT / "models"
ENS = f"{M}/gate_clean_ens_precision_L107.pt++{M}/gate_clean_ens_course_L110.pt"
BUNDLE = ROOT / "handoff/shadowpc-followups-2026-06-05/task2_frames"

d = json.loads((BUNDLE / "frames.json").read_text())
frames = [Frame(frame_id=fr["frame_id"], sim_time_ns=fr["sim_time_ns"],
                image_bgr=cv2.imread(str(BUNDLE / fr["png"]))) for fr in d["frames"]]
print(f"detector module: {_d.__file__}")
print(f"frames: {len(frames)} @ {frames[0].image_bgr.shape}\n")


def bench(det, label, dedup=None):
    if dedup is not None:
        det.dedup_px = dedup
    for _ in range(3):
        det.detect(frames[0])                      # warmup (CUDA graph / autotune)
    times, nobs = [], []
    for f in frames:
        t0 = time.perf_counter()
        obs = det.detect(f)
        times.append((time.perf_counter() - t0) * 1e3)
        nobs.append(len(obs))
    t = np.array(times)
    print(f"{label:36} mean {t.mean():5.1f} ms  p95 {np.percentile(t, 95):5.1f} ms  "
          f"obs/frame {np.mean(nobs):.2f}")
    return t.mean(), float(np.percentile(t, 95)), float(np.mean(nobs))


champ = GateDetector.load(CHAMP, score_thresh=0.25, kpt_conf_thresh=0.5)
assert isinstance(champ, GateDetector) and not isinstance(champ, EnsembleGateDetector)
bench(champ, "champion (single model)")

ens = GateDetector.load(ENS, score_thresh=0.25, kpt_conf_thresh=0.5)
assert isinstance(ens, EnsembleGateDetector) and len(ens.models) == 2
_, _, u = bench(ens, "ensemble pure-union (dedup_px=0)", dedup=0.0)
mean12, p95_12, dd = bench(ens, "ensemble fuse-dedup (dedup_px=12)", dedup=12.0)
print(f"\ndedup: union {u:.2f} -> {dd:.2f} obs/frame  ({100 * (1 - dd / u):.0f}% fewer KF updates/frame)")
print(f"deploy config latency: mean {mean12:.1f} ms / p95 {p95_12:.1f} ms  "
      f"(budget 50 ms -> {'PASS' if p95_12 <= 50 else 'OVER'})")
