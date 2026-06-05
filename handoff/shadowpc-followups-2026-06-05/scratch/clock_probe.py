"""Throwaway: verify the clock epochs before building the sysid extract join.

Question: do telemetry `time_usec` (HIGHRES_IMU / ODOMETRY / ACTUATOR_OUTPUT_STATUS) share ONE
sim epoch, and does commands.jsonl's `sim_time_ns` (= state.sim_time_ns) live on it? If yes, join
commands<->telemetry on the sim clock (clean, no recv jitter). Also report recv-clock (tlog prefix
= msg._timestamp) for reference. Read-only; prints a table.

Usage: python clock_probe.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("C:/Users/Shadow/Peregrine/.claude/worktrees/blissful-sutherland-c8ebd9/src")
sys.path.insert(0, str(SRC))

from racer.recording import RecordingReader  # noqa: E402


def probe(run: Path) -> None:
    reader = RecordingReader(run)
    meta = reader.meta
    # per-type: count, time_usec min/max (if present), recv min/max
    types: dict[str, dict] = {}
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        d = types.setdefault(t, {"n": 0, "tu_min": None, "tu_max": None,
                                 "rv_min": None, "rv_max": None})
        d["n"] += 1
        tu = getattr(msg, "time_usec", None)
        if tu is not None:
            tu = int(tu)
            d["tu_min"] = tu if d["tu_min"] is None else min(d["tu_min"], tu)
            d["tu_max"] = tu if d["tu_max"] is None else max(d["tu_max"], tu)
        rv = getattr(msg, "_timestamp", None)  # tlog prefix: recv unix seconds (float)
        if rv is not None:
            rv = float(rv)
            d["rv_min"] = rv if d["rv_min"] is None else min(d["rv_min"], rv)
            d["rv_max"] = rv if d["rv_max"] is None else max(d["rv_max"], rv)

    print(f"\n===== {run.name} =====  schema={meta.get('schema')} mode={meta.get('mode')}")
    print(f"{'type':>24} {'n':>6} {'time_usec min':>18} {'time_usec max':>18} "
          f"{'recv_unix min':>16} {'recv_unix max':>16}")
    for t in sorted(types):
        d = types[t]
        tumin = "" if d["tu_min"] is None else f"{d['tu_min']}"
        tumax = "" if d["tu_max"] is None else f"{d['tu_max']}"
        rvmin = "" if d["rv_min"] is None else f"{d['rv_min']:.3f}"
        rvmax = "" if d["rv_max"] is None else f"{d['rv_max']:.3f}"
        print(f"{t:>24} {d['n']:>6} {tumin:>18} {tumax:>18} {rvmin:>16} {rvmax:>16}")

    # commands.jsonl sim clock
    cj = run / "commands.jsonl"
    if cj.exists():
        smin = smax = None
        n = 0
        for line in cj.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            s = r.get("sim_time_ns", r.get("sim_t"))
            if s is None:
                continue
            s = int(s)
            n += 1
            smin = s if smin is None else min(smin, s)
            smax = s if smax is None else max(smax, s)
        if smin is not None:
            print(f"  commands.jsonl: n={n}  sim_clock(ns) [{smin} .. {smax}]  "
                  f"= usec [{smin // 1000} .. {smax // 1000}]")
            # compare to ODOMETRY/HIGHRES time_usec window
            for ref in ("HIGHRES_IMU", "ODOMETRY", "ACTUATOR_OUTPUT_STATUS"):
                d = types.get(ref)
                if d and d["tu_min"] is not None:
                    overlap = not (smax // 1000 < d["tu_min"] or smin // 1000 > d["tu_max"])
                    print(f"     vs {ref:>22} time_usec [{d['tu_min']} .. {d['tu_max']}]  "
                          f"OVERLAP={overlap}")
    else:
        print("  (no commands.jsonl)")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        probe(Path(a))
