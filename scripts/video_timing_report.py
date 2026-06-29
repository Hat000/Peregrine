"""video_timing_report.py — frame-timing / drop monitor for a recorded sim session.

WHY: the live VQ2 camera is a chunked-JPEG UDP stream (sender ~28.6 fps, each frame re-sent
~14x). When the receiver thread is briefly starved the OS socket buffer overflows and WHOLE
frames are lost — invisible to meta.json's "dropped" (that only counts recorder-queue overflow)
and even to the receiver's partials_evicted (a frame with zero chunks received never becomes a
"partial"). The only complete evidence is the gap in the per-frame ``frame_id`` sequence logged
in ``video_index.jsonl``. This tool reads those timestamps and reports drop %, effective fps,
inter-frame jitter, and the "surge" points (long real-time holes) that make playback lurch.

USAGE
  python scripts/video_timing_report.py <session_dir> [--csv] [--ledger runs_video_timing.csv]
  python scripts/video_timing_report.py --latest            # newest data/runs/* session
  python scripts/video_timing_report.py data/runs/*_f1      # globs ok (shell-expanded)

A run is HEALTHY when drop% is low and the effective fps is stable across runs. Run this after
every flight (or point --ledger at one CSV to watch consistency over time).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

# Thresholds (tune as the capture improves).
DROP_WARN_PCT = 5.0
DROP_FAIL_PCT = 20.0
SURGE_DT_MS = 60.0           # a real-time inter-frame hole bigger than this == visible surge
NOMINAL_FPS = 28.6           # sender's true frame rate (jpeg_receiver docstring)


def _load_index(session: Path) -> list[dict]:
    idxp = session / "video_index.jsonl"
    if not idxp.exists():
        raise FileNotFoundError(f"no video_index.jsonl in {session}")
    return [json.loads(l) for l in idxp.read_text().splitlines() if l.strip()]


def analyze(session: Path) -> dict:
    idx = _load_index(session)
    if len(idx) < 2:
        return {"session": session.name, "kept": len(idx), "error": "too few frames"}
    fids = [int(r["frame_id"]) for r in idx]
    recv = [int(r["recv_monotonic_ns"]) / 1e6 for r in idx]   # ms
    sim = [int(r["sim_time_ns"]) / 1e6 for r in idx]          # ms

    span = fids[-1] - fids[0] + 1                  # frames the sender numbered across the window
    kept = len(fids)
    dropped = span - kept
    drop_pct = 100.0 * dropped / span if span else 0.0

    # frame_id gaps (consecutive missing) + the worst one
    gaps = [(fids[i - 1], fids[i] - fids[i - 1] - 1)
            for i in range(1, len(fids)) if fids[i] - fids[i - 1] > 1]
    longest_gap = max((g[1] for g in gaps), default=0)

    d_recv = [recv[i] - recv[i - 1] for i in range(1, len(recv))]
    d_sim = [sim[i] - sim[i - 1] for i in range(1, len(sim))]
    dur_s = (recv[-1] - recv[0]) / 1000.0
    eff_fps = (kept - 1) / dur_s if dur_s > 0 else 0.0     # frames actually captured / real time
    burst_fps = 1000.0 / st.median(d_recv) if d_recv else 0.0   # instantaneous (median gap)
    surges = [(i, round(d_recv[i - 1], 1)) for i in range(1, len(recv)) if d_recv[i - 1] > SURGE_DT_MS]

    if drop_pct >= DROP_FAIL_PCT:
        verdict = "FAIL"
    elif drop_pct >= DROP_WARN_PCT:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "session": session.name, "kept": kept, "expected_span": span, "dropped": dropped,
        "drop_pct": round(drop_pct, 1), "longest_gap": longest_gap, "n_gaps": len(gaps),
        "duration_s": round(dur_s, 2), "eff_fps": round(eff_fps, 1), "burst_fps": round(burst_fps, 1),
        "recv_dt_ms_p50": round(st.median(d_recv), 1),
        "recv_dt_ms_p95": round(sorted(d_recv)[int(0.95 * (len(d_recv) - 1))], 1),
        "recv_dt_ms_max": round(max(d_recv), 1),
        "recv_dt_ms_std": round(st.pstdev(d_recv), 1),
        "sim_dt_ms_p50": round(st.median(d_sim), 1),
        "n_surges": len(surges), "surges": surges[:12], "gaps": gaps[:12], "verdict": verdict,
        "_per_frame": list(zip(fids, sim, recv, [0] + d_recv)),
    }


def print_report(a: dict) -> None:
    if a.get("error"):
        print(f"  {a['session']}: {a['error']}")
        return
    bar = a["verdict"]
    print(f"\n=== {a['session']}  [{bar}] ===")
    print(f"  frames kept   : {a['kept']} of {a['expected_span']} sent  "
          f"-> DROPPED {a['dropped']} ({a['drop_pct']}%)   longest hole {a['longest_gap']} frames")
    print(f"  effective fps : {a['eff_fps']}  (sender nominal {NOMINAL_FPS})   "
          f"burst fps {a['burst_fps']}   over {a['duration_s']}s")
    print(f"  recv dt (ms)  : p50={a['recv_dt_ms_p50']} p95={a['recv_dt_ms_p95']} "
          f"max={a['recv_dt_ms_max']} std={a['recv_dt_ms_std']}   (sim dt p50={a['sim_dt_ms_p50']})")
    print(f"  surges (>{SURGE_DT_MS:.0f}ms holes): {a['n_surges']}   gaps in frame_id: {a['n_gaps']}")
    if a["surges"]:
        print(f"    surge @frame-idx(dt ms): {a['surges']}")


def write_csv(session: Path, a: dict) -> Path:
    out = session / "frame_timestamps.csv"
    lines = ["frame_id,sim_ms,recv_ms,dt_recv_ms"]
    for fid, sim, recv, dt in a["_per_frame"]:
        lines.append(f"{fid},{sim:.3f},{recv:.3f},{dt:.3f}")
    out.write_text("\n".join(lines) + "\n")
    return out


def append_ledger(ledger: Path, a: dict) -> None:
    new = not ledger.exists()
    with open(ledger, "a", encoding="utf-8") as f:
        if new:
            f.write("session,verdict,kept,expected,dropped,drop_pct,eff_fps,burst_fps,"
                    "recv_dt_p95_ms,recv_dt_max_ms,longest_gap,n_surges\n")
        f.write(f"{a['session']},{a['verdict']},{a['kept']},{a['expected_span']},{a['dropped']},"
                f"{a['drop_pct']},{a['eff_fps']},{a['burst_fps']},{a['recv_dt_ms_p95']},"
                f"{a['recv_dt_ms_max']},{a['longest_gap']},{a['n_surges']}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session dir(s) containing video_index.jsonl")
    ap.add_argument("--latest", action="store_true", help="use the newest data/runs/* session")
    ap.add_argument("--csv", action="store_true", help="write frame_timestamps.csv into each session")
    ap.add_argument("--ledger", default=None, help="append a one-line summary to this CSV (cross-run consistency)")
    args = ap.parse_args()

    sessions = [Path(s) for s in args.sessions]
    if args.latest:
        runs = sorted(Path("data/runs").glob("*"), key=lambda p: p.stat().st_mtime)
        runs = [r for r in runs if (r / "video_index.jsonl").exists()]
        if not runs:
            print("no sessions with video_index.jsonl under data/runs", file=sys.stderr)
            return 2
        sessions = [runs[-1]]
    if not sessions:
        ap.error("give a session dir, a glob, or --latest")

    worst = "PASS"
    rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
    for s in sessions:
        if not (s / "video_index.jsonl").exists():
            print(f"  skip {s} (no video_index.jsonl)", file=sys.stderr)
            continue
        a = analyze(s)
        print_report(a)
        if args.csv and not a.get("error"):
            print(f"  wrote {write_csv(s, a)}")
        if args.ledger and not a.get("error"):
            append_ledger(Path(args.ledger), a)
        if not a.get("error") and rank[a["verdict"]] > rank[worst]:
            worst = a["verdict"]
    if args.ledger:
        print(f"\n  ledger -> {args.ledger}")
    # non-zero exit on FAIL so it can gate a post-flight check / CI
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
