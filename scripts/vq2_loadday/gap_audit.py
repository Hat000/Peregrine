"""scripts/vq2_loadday/gap_audit.py — VQ2 load-day occlusion-gap audit (C7).

Measures the distribution of gate-detection gaps (consecutive frames with zero detections)
from a recorded detection stream or a live session.  The longest gap determines the
required GRU memory depth for blind-coast navigation, and whether a LIO gap-filler is
necessary.

Inputs:
  (a) A JSONL detection log (one JSON object per frame: ``frame_id``, ``sim_time_ns``,
      ``n_detections`` — see ``--det-log``).  Written by Mission.run or the VQ1 pipeline.
  (b) A video session directory (``--session data/runs/<stamp>_<label>/``) — reads
      ``video_index.jsonl`` to derive frame arrival times and combines with a detection
      log in the same directory.
  (c) A synthetic benchmark with user-specified gap parameters (``--synthetic``) for
      offline unit-testability.

Computed metrics:
  - total_frames / detected_frames / gap_frames
  - detection_rate: detected_frames / total_frames
  - gap lengths (frames): max, mean, p90, p99, histogram
  - gap durations (ms): converted via per-frame sim-time deltas
  - c7_gru_depth_frames: recommended GRU memory depth = max gap + 2 (conservative cover)
  - c7_gap_filler_needed: True if max_gap_ms > 200 ms (the LIO / gap-filler design trigger)

Usage (detection log JSONL):
  python scripts/vq2_loadday/gap_audit.py --det-log data/det_log.jsonl

Usage (session directory):
  python scripts/vq2_loadday/gap_audit.py --session data/runs/<stamp>/

Usage (synthetic — offline test with no real data):
  python scripts/vq2_loadday/gap_audit.py --synthetic 500 0.7

Usage (save JSON):
  python scripts/vq2_loadday/gap_audit.py --det-log data/det_log.jsonl --out data/gap_audit.json

Degrades gracefully: if no detection log is found, reports "stream absent" and exits 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Gap-filler trigger: if the longest blind-coast gap exceeds this many ms, the GRU / LIO
# gap-filler is load-bearing.
_GAP_FILLER_THRESHOLD_MS = 200.0

# Default frame rate (30 Hz spec) — used only when sim_time_ns deltas are unavailable.
_DEFAULT_FPS = 30.0


def _detect_log_from_session(session_dir: Path) -> Path | None:
    """Look for a detection JSONL log in the session directory."""
    candidates = [
        session_dir / "det_log.jsonl",
        session_dir / "detections.jsonl",
        session_dir / "detection_log.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_det_log(det_log_path: Path) -> list[dict]:
    """Load a per-frame detection JSONL.  Fields used: frame_id, sim_time_ns, n_detections."""
    records: list[dict] = []
    with open(det_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append({
                    "frame_id": int(obj.get("frame_id", len(records))),
                    "sim_time_ns": int(obj.get("sim_time_ns", 0)),
                    "n_detections": int(obj.get("n_detections", 0)),
                })
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    return records


def _load_video_index(session_dir: Path) -> list[dict]:
    """Load video_index.jsonl (frame arrival metadata, no detection info)."""
    idx_path = session_dir / "video_index.jsonl"
    if not idx_path.exists():
        return []
    records: list[dict] = []
    with open(idx_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append({
                    "frame_id": int(obj.get("frame_id", len(records))),
                    "sim_time_ns": int(obj.get("sim_time_ns", 0)),
                    "n_detections": 0,  # no detection info in the index itself
                })
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    return records


def _make_synthetic(n_frames: int, detection_rate: float, rng_seed: int = 42) -> list[dict]:
    """Generate a synthetic frame sequence for offline testing."""
    rng = np.random.default_rng(rng_seed)
    detected = rng.random(n_frames) < detection_rate
    dt_ns = int(1e9 / _DEFAULT_FPS)
    records: list[dict] = []
    for i in range(n_frames):
        records.append({
            "frame_id": i,
            "sim_time_ns": i * dt_ns,
            "n_detections": 1 if detected[i] else 0,
        })
    return records


def analyze(records: list[dict]) -> dict:
    """Compute gap-audit metrics from a list of per-frame detection records.

    This is the pure analysis function — no I/O, fully unit-testable.
    Each record must have: ``frame_id`` (int), ``sim_time_ns`` (int), ``n_detections`` (int).
    """
    if not records:
        return {"error": "no frame records provided"}

    n = len(records)
    # Sort by frame_id to ensure sequence order
    records_sorted = sorted(records, key=lambda r: r["frame_id"])

    detected = np.array([r["n_detections"] > 0 for r in records_sorted], dtype=bool)
    times_ns = np.array([r["sim_time_ns"] for r in records_sorted], dtype=np.int64)

    n_detected = int(detected.sum())
    n_gap = n - n_detected
    detection_rate = n_detected / n if n > 0 else 0.0

    # Compute per-frame dt_ms for gap-duration conversion
    # Use adjacent sim_time_ns deltas; fall back to 1/_DEFAULT_FPS if times are unavailable
    dt_ns_arr = np.diff(times_ns)
    if len(dt_ns_arr) > 0 and times_ns.max() > times_ns.min():
        # Use median inter-frame time (robust to resets / clock jumps)
        median_dt_ns = float(np.median(dt_ns_arr[dt_ns_arr > 0])) if np.any(dt_ns_arr > 0) else 0.0
    else:
        median_dt_ns = 1e9 / _DEFAULT_FPS
    frame_dt_ms = median_dt_ns / 1e6 if median_dt_ns > 0 else (1000.0 / _DEFAULT_FPS)

    # Find runs of consecutive gap frames
    gap_lengths: list[int] = []
    i = 0
    while i < n:
        if not detected[i]:
            run_len = 0
            while i < n and not detected[i]:
                run_len += 1
                i += 1
            gap_lengths.append(run_len)
        else:
            i += 1

    if not gap_lengths:
        max_gap_frames = 0
        max_gap_ms = 0.0
        p90_gap_frames = 0.0
        p99_gap_frames = 0.0
        mean_gap_frames = 0.0
    else:
        gl = np.array(gap_lengths, dtype=np.float64)
        max_gap_frames = int(gl.max())
        max_gap_ms = max_gap_frames * frame_dt_ms
        p90_gap_frames = float(np.percentile(gl, 90))
        p99_gap_frames = float(np.percentile(gl, 99))
        mean_gap_frames = float(gl.mean())

    # Histogram of gap lengths (bins: 1, 2-5, 6-15, 16-30, 31-100, >100)
    bins = [1, 2, 6, 16, 31, 101, max_gap_frames + 2]
    hist = {}
    if gap_lengths:
        gl_int = np.array(gap_lengths, dtype=int)
        labels = ["=1", "2-5", "6-15", "16-30", "31-100", ">100"]
        boundaries = [(1, 2), (2, 6), (6, 16), (16, 31), (31, 101), (101, 10**9)]
        for label, (lo, hi) in zip(labels, boundaries):
            hist[label] = int(((gl_int >= lo) & (gl_int < hi)).sum())

    # GRU depth recommendation: cover the max gap with 2-frame margin
    gru_depth_frames = max_gap_frames + 2

    # Gap-filler trigger
    gap_filler_needed = max_gap_ms > _GAP_FILLER_THRESHOLD_MS

    return {
        "n_frames": n,
        "n_detected_frames": n_detected,
        "n_gap_frames": n_gap,
        "detection_rate": round(detection_rate, 4),
        "n_gap_runs": len(gap_lengths),
        "max_gap_frames": max_gap_frames,
        "max_gap_ms": round(max_gap_ms, 2),
        "p90_gap_frames": round(p90_gap_frames, 1),
        "p99_gap_frames": round(p99_gap_frames, 1),
        "mean_gap_frames": round(mean_gap_frames, 2),
        "frame_dt_ms": round(frame_dt_ms, 3),
        "gap_histogram": hist,
        "c7_gru_depth_frames": gru_depth_frames,
        "c7_gap_filler_needed": gap_filler_needed,
        "c7_gap_filler_threshold_ms": _GAP_FILLER_THRESHOLD_MS,
    }


def _print_result(res: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  GAP AUDIT (C7) — gate detection occlusion gaps")
    print(f"{'='*65}")
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return

    print(f"\n  Frames total     : {res['n_frames']}")
    print(f"  Detected frames  : {res['n_detected_frames']}  "
          f"({res['detection_rate']*100:.1f}%)")
    print(f"  Gap frames       : {res['n_gap_frames']}")
    print(f"  Gap runs         : {res['n_gap_runs']}")
    print(f"  Frame dt         : {res['frame_dt_ms']:.1f} ms")
    print(f"\n  Gap lengths (frames):")
    print(f"    max   : {res['max_gap_frames']} frames = {res['max_gap_ms']:.0f} ms")
    print(f"    p99   : {res['p99_gap_frames']:.0f} frames")
    print(f"    p90   : {res['p90_gap_frames']:.0f} frames")
    print(f"    mean  : {res['mean_gap_frames']:.1f} frames")
    print(f"\n  Gap length histogram (# of runs):")
    for k, v in res.get("gap_histogram", {}).items():
        bar = "#" * min(v, 40)
        print(f"    {k:>6} frames: {v:5d}  {bar}")
    print(f"\n--- C7 VERDICT:")
    print(f"    max_gap_ms      = {res['max_gap_ms']:.0f} ms  "
          f"(threshold {res['c7_gap_filler_threshold_ms']:.0f} ms)")
    print(f"    gru_depth_frames= {res['c7_gru_depth_frames']}  "
          f"(= max_gap + 2, conservative)")
    print(f"    gap_filler_needed: {res['c7_gap_filler_needed']}")
    if res["c7_gap_filler_needed"]:
        print(f"    => MAX GAP EXCEEDS {res['c7_gap_filler_threshold_ms']:.0f} ms. "
              f"LIO / GRU gap-filler is LOAD-BEARING.")
    else:
        print(f"    => Max gap within tolerance. GRU memory at {res['c7_gru_depth_frames']} "
              f"frames may suffice.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--det-log", metavar="JSONL_PATH",
        help="Per-frame detection JSONL (fields: frame_id, sim_time_ns, n_detections)"
    )
    src.add_argument(
        "--session", metavar="SESSION_DIR",
        help="Session directory (data/runs/<stamp>/); looks for det_log.jsonl inside"
    )
    src.add_argument(
        "--synthetic", nargs=2, metavar=("N_FRAMES", "DETECT_RATE"),
        help="Synthetic benchmark: N_FRAMES frames at DETECT_RATE (e.g. 500 0.7)"
    )
    ap.add_argument("--out", default=None, help="Write audit JSON here")
    args = ap.parse_args()

    if args.synthetic:
        n_frames = int(args.synthetic[0])
        rate = float(args.synthetic[1])
        print(f"  Running synthetic benchmark: {n_frames} frames at {rate*100:.0f}% detect rate")
        records = _make_synthetic(n_frames, rate)
    elif args.det_log:
        det_log_path = Path(args.det_log)
        if not det_log_path.exists():
            print(f"\n  ERROR: stream absent / not connected: det-log not found at {det_log_path}",
                  file=sys.stderr)
            print(f"  -> Run a flight session first to produce a detection log.", file=sys.stderr)
            return 2
        print(f"  Loading detection log: {det_log_path}")
        records = _load_det_log(det_log_path)
    else:  # --session
        session_dir = Path(args.session)
        if not session_dir.exists():
            print(f"\n  ERROR: stream absent / not connected: session dir not found at {session_dir}",
                  file=sys.stderr)
            return 2
        det_log_path = _detect_log_from_session(session_dir)
        if det_log_path is None:
            print(f"\n  ERROR: stream absent / not connected: no detection log found in {session_dir}",
                  file=sys.stderr)
            print(f"  -> Expected one of: det_log.jsonl, detections.jsonl, detection_log.jsonl",
                  file=sys.stderr)
            return 2
        print(f"  Loading detection log: {det_log_path}")
        records = _load_det_log(det_log_path)

    if not records:
        print("  ERROR: no frame records loaded. Detection log empty?", file=sys.stderr)
        return 2

    res = analyze(records)
    _print_result(res)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print(f"\n  Audit written to {out_path}")

    return 0 if not res.get("c7_gap_filler_needed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
