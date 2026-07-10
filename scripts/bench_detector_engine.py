"""Offline parity + latency bench: exported gate-detector engine vs its source .pt.

Runs the SAME recorded in-race frames through the flight-stack ``GateDetector`` (same
thresholds the flight uses) once per weights spec — separate sequential passes so the two
models never contend — then reports:

  PARITY : per-frame detection-count agreement, greedy centroid-matched observation pairs,
           mean/median/p95 corner-keypoint px error over matched pairs, unmatched counts.
  LATENCY: per-frame ``detect()`` wall ms (mean/median/p95/max) per spec + speedup.

PASS gate (printed with the verdict): mean matched-corner error < 2.0 px, identical-count
frames >= 95%, unmatched observations <= 3% per side. fp16 quantization moves keypoints a
sub-pixel amount in the normal case; anything beyond these bounds means the engine is NOT a
drop-in for the .pt and must not fly.

NOTE: these are UNCONTENDED numbers (no sim rendering on the GPU). The decisive
contended-in-flight number is measured live via fly_rl's [vision-timing] log.

Run (ShadowPC — ALWAYS the .venv python):
  .venv/Scripts/python.exe scripts/bench_detector_engine.py \
      --pt models/gate_clean_ens_course_L110.pt \
      --engine models/gate_clean_ens_course_L110_fp16_384x640.engine \
      --run-dir data/runs/20260701_224640_vq2_slow_seeker_a19c_f1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.recording import RecordingReader                        # noqa: E402
from racer.vision.detector import GateDetector                     # noqa: E402


def log_versions() -> None:
    import torch
    import ultralytics

    print(f"python     : {sys.executable}")
    print(f"torch      : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"ultralytics: {ultralytics.__version__}")
    try:
        import tensorrt

        print(f"tensorrt   : {tensorrt.__version__}")
    except ImportError:
        print("tensorrt   : (not installed)")
    if torch.cuda.is_available():
        print(f"gpu        : {torch.cuda.get_device_name(0)}")


def load_frames(run_dirs: list[str], max_frames: int) -> list:
    frames = []
    for rd in run_dirs:
        for f in RecordingReader(rd).frames():
            frames.append(f)
            if len(frames) >= max_frames:
                return frames
    return frames


def run_pass(spec: str, frames: list, warmup: int) -> tuple[list, np.ndarray]:
    """One detector over all frames: returns (per-frame obs lists, per-frame wall ms)."""
    det = GateDetector.load(spec)
    for _ in range(warmup):  # cuDNN autotune / TRT ctx / predictor setup off the timed path
        det.detect(frames[0])
    all_obs, times_ms = [], []
    for f in frames:
        t0 = time.perf_counter()
        obs = det.detect(f)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        all_obs.append(obs)
    del det
    try:
        import torch

        torch.cuda.empty_cache()  # the two passes must not fight over VRAM
    except Exception:
        pass
    return all_obs, np.asarray(times_ms)


def centroid(o) -> np.ndarray:
    return np.asarray(o.corners_px, dtype=np.float64).mean(axis=0)


def match_frame(obs_a: list, obs_b: list, match_px: float):
    """Greedy nearest-centroid matching of one frame's observations. Returns
    (pairs, unmatched_a, unmatched_b); pairs are (obs_a_i, obs_b_j) with centroid dist < match_px."""
    pairs, used_b = [], set()
    cands = sorted(
        ((float(np.linalg.norm(centroid(a) - centroid(b))), i, j)
         for i, a in enumerate(obs_a) for j, b in enumerate(obs_b)),
        key=lambda t: t[0])
    used_a = set()
    for d, i, j in cands:
        if d >= match_px or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((obs_a[i], obs_b[j]))
    return pairs, len(obs_a) - len(used_a), len(obs_b) - len(used_b)


def stats(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def fmt(s: dict) -> str:
    return f"mean {s['mean']:7.2f}  median {s['median']:7.2f}  p95 {s['p95']:7.2f}  max {s['max']:7.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True, help="reference .pt weights")
    ap.add_argument("--engine", required=True, help="exported .engine (or .onnx) under test")
    ap.add_argument("--run-dir", action="append", required=True,
                    help="recorded session dir(s); repeat the flag for more frames")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--match-px", type=float, default=30.0,
                    help="max centroid distance for pairing the same gate across detectors")
    ap.add_argument("--json", default=None, help="write the full metrics dict to this path")
    args = ap.parse_args()

    log_versions()
    frames = load_frames(args.run_dir, args.max_frames)
    print(f"\nframes     : {len(frames)} from {len(args.run_dir)} run(s)")
    if len(frames) < 200:
        print("WARNING: <200 frames — parity gate calls for >=200 in-race frames")

    print(f"\n=== pass 1/2: {args.pt}")
    obs_pt, t_pt = run_pass(args.pt, frames, args.warmup)
    print(f"=== pass 2/2: {args.engine}")
    obs_en, t_en = run_pass(args.engine, frames, args.warmup)

    # ---- parity ----
    corner_errs, score_dev = [], []
    n_pairs = un_pt = un_en = 0
    count_equal = 0
    for oa, ob in zip(obs_pt, obs_en):
        if len(oa) == len(ob):
            count_equal += 1
        pairs, ua, ub = match_frame(oa, ob, args.match_px)
        n_pairs += len(pairs)
        un_pt += ua
        un_en += ub
        for a, b in pairs:
            ca = np.asarray(a.corners_px, dtype=np.float64)
            cb = np.asarray(b.corners_px, dtype=np.float64)
            corner_errs.append(float(np.linalg.norm(ca - cb, axis=1).mean()))
            score_dev.append(abs(float(a.score) - float(b.score)))
    corner_errs = np.asarray(corner_errs)
    total_pt = sum(len(o) for o in obs_pt)
    total_en = sum(len(o) for o in obs_en)
    count_agree = count_equal / len(frames) if frames else float("nan")
    un_pt_frac = un_pt / total_pt if total_pt else 0.0
    un_en_frac = un_en / total_en if total_en else 0.0
    err = stats(corner_errs)

    print("\n---- PARITY (engine vs .pt, flight thresholds) ----")
    print(f"observations       : pt={total_pt}  engine={total_en}  matched pairs={n_pairs}")
    print(f"count agreement    : {count_agree * 100:.1f}% of frames identical obs count")
    print(f"unmatched          : pt-only {un_pt} ({un_pt_frac * 100:.1f}%)  "
          f"engine-only {un_en} ({un_en_frac * 100:.1f}%)")
    print(f"corner err px      : {fmt(err)}")
    if score_dev:
        print(f"score |delta|      : mean {np.mean(score_dev):.4f}  max {np.max(score_dev):.4f}")

    checks = {
        "mean corner err < 2.0 px": bool(err["mean"] < 2.0),
        "count agreement >= 95%": bool(count_agree >= 0.95),
        "unmatched pt-only <= 3%": bool(un_pt_frac <= 0.03),
        "unmatched engine-only <= 3%": bool(un_en_frac <= 0.03),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    parity_pass = all(checks.values())

    # ---- latency ----
    s_pt, s_en = stats(t_pt), stats(t_en)
    print("\n---- LATENCY (uncontended detect() wall ms) ----")
    print(f".pt    : {fmt(s_pt)}")
    print(f"engine : {fmt(s_en)}")
    print(f"speedup: mean {s_pt['mean'] / s_en['mean']:.2f}x  median {s_pt['median'] / s_en['median']:.2f}x")

    print(f"\nVERDICT: {'PARITY PASS' if parity_pass else 'PARITY FAIL'}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "pt": args.pt, "engine": args.engine, "n_frames": len(frames),
            "total_obs_pt": total_pt, "total_obs_engine": total_en, "matched_pairs": n_pairs,
            "count_agreement": count_agree, "unmatched_pt": un_pt, "unmatched_engine": un_en,
            "corner_err_px": err, "latency_pt_ms": s_pt, "latency_engine_ms": s_en,
            "checks": checks, "parity_pass": parity_pass,
        }, indent=2), encoding="utf-8")
        print(f"json -> {args.json}")
    return 0 if parity_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
