"""SHADOWPC-INC6-DIAG H3: timing forensics from debug_obs.jsonl.

Per run: wall loop period (t_mono diffs), sim-time advance per tick, sim-time/wall
ratio (sim speed), odometry staleness, and the live-vs-twin trajectory divergence
over the first 60 ticks (same-action comparison comes later if needed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = sorted((ROOT / "data" / "runs").glob("20260612_*_inc6_*_f*"))


def load(run):
    rows = []
    p = run / "debug_obs.jsonl"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    print(f"{'run':<38} {'n':>5} {'wall_dt med/p95/max ms':>24} {'simdt med/p95 ms':>18} "
          f"{'sim/wall':>8} {'odo_age med/p95 ms':>20} {'age>67ms %':>10}")
    for run in RUNS:
        rows = load(run)
        if not rows or len(rows) < 10:
            print(f"{run.name:<38} {'(no debug_obs / too short)':>30}")
            continue
        t = np.array([r["t_mono"] for r in rows])
        s = np.array([r["sim_time_ns"] for r in rows], dtype=np.float64) / 1e6  # ms
        age = np.array([r["odo_age_ms"] for r in rows])
        dt_w = np.diff(t) * 1e3
        dt_s = np.diff(s)
        # sim/wall over the whole run (robust to per-tick quantization)
        ratio = (s[-1] - s[0]) / ((t[-1] - t[0]) * 1e3)
        frozen = float(np.mean(dt_s <= 0.0)) * 100
        print(f"{run.name:<38} {len(rows):>5} "
              f"{np.median(dt_w):>8.1f}/{pct(dt_w,95):>6.1f}/{dt_w.max():>7.1f} "
              f"{np.median(dt_s):>9.1f}/{pct(dt_s,95):>7.1f} "
              f"{ratio:>8.3f} "
              f"{np.median(age):>10.1f}/{pct(age,95):>8.1f} "
              f"{float(np.mean(age>67))*100:>9.1f}%"
              + (f"  [simdt<=0: {frozen:.0f}%]" if frozen > 1 else ""))

    # detail: first 40 ticks of standing f1 — track when live pos diverges from plan
    f1 = load(ROOT / "data" / "runs" / "20260612_034852_inc6_standing_f1")
    print("\nstanding_f1 first 40 ticks:")
    print(f"{'k':>3} {'dt_w ms':>8} {'dt_s ms':>8} {'age':>6} {'pos_ned':>30} {'thr':>6} {'rate_frd':>26} {'o12':>6}")
    prev_t, prev_s = None, None
    for r in f1[:40]:
        dt_w = (r["t_mono"] - prev_t) * 1e3 if prev_t else 0.0
        dt_s = (r["sim_time_ns"] - prev_s) / 1e6 if prev_s else 0.0
        prev_t, prev_s = r["t_mono"], r["sim_time_ns"]
        p = r["pos_ned"]
        print(f"{r['k']:>3} {dt_w:>8.1f} {dt_s:>8.1f} {r['odo_age_ms']:>6.1f} "
              f"[{p[0]:+7.2f},{p[1]:+7.2f},{p[2]:+7.2f}] {r['collective']:>6.3f} "
              f"[{r['rate_frd'][0]:+5.2f},{r['rate_frd'][1]:+5.2f},{r['rate_frd'][2]:+5.2f}] "
              f"{r['obs'][12]:>6.2f}")


if __name__ == "__main__":
    main()
