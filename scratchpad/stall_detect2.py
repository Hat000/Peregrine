"""Second pass: "hovers and waits" may mean NOT CLOSING, not NOT MOVING.

The first pass looked for near-zero SPEED and found it on 1/370 flights, so the literal
reading is wrong.  But a drone that keeps flying while the range to its locked gate stops
shrinking looks exactly like hesitation from the outside, and it is the more diagnostic
quantity anyway: it is the failure the pilot actually cares about (no progress), and it
survives the case where the drone is circling or sliding laterally at speed.

Measured per flight, only while a gate is actually locked (pose_seen):
  * CLOSURE  = -d(dist)/dt on the emitted slot0 gate. Positive = closing.
  * DRIFT    = a run of >= DRIFT_TICKS consecutive ticks with closure < CLOSE_MPS,
               after SETTLE, i.e. sustained failure to make ground on the locked gate.
  * SWITCH   = |d dist| > SWITCH_M between consecutive fixes (the seeker changed target).
Then: do drifts follow switches more often than chance?
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs")

SETTLE = 30
CLOSE_MPS = 0.5           # closing slower than this = not making ground
DRIFT_TICKS = 15          # ~0.5 s
SWITCH_M = 5.0
LOOKBACK = 40


def main() -> int:
    n_flights = 0
    n_with_drift = 0
    drift_lens: list[int] = []
    drift_after_switch = 0
    drift_speeds: list[float] = []
    drift_ranges: list[float] = []
    post_switch_drift = 0
    n_switch_total = 0
    switch_windows_examined = 0

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        if len(rows) < SETTLE + DRIFT_TICKS + 5:
            continue
        n_flights += 1

        t = [r["sim_time_ns"] / 1e9 for r in rows]
        seen = [bool(r.get("pose_seen")) and r.get("dist") is not None for r in rows]
        dist = [float(r["dist"]) if seen[i] else None for i, r in enumerate(rows)]
        spd = [(r["obs"][0] ** 2 + r["obs"][1] ** 2 + r["obs"][2] ** 2) ** 0.5 for r in rows]

        # closure per tick (only where two consecutive fixes exist and no target switch)
        closure: list[float | None] = [None] * len(rows)
        switches = []
        for i in range(1, len(rows)):
            if not (seen[i] and seen[i - 1]):
                continue
            dt = t[i] - t[i - 1]
            if dt <= 0:
                continue
            dd = dist[i] - dist[i - 1]
            if abs(dd) > SWITCH_M:
                switches.append(i)
                continue                       # a switch is not a closure measurement
            closure[i] = -dd / dt
        n_switch_total += len(switches)

        # sustained drift runs
        runs_found = []
        i = SETTLE
        while i < len(rows):
            if closure[i] is not None and closure[i] < CLOSE_MPS:
                j = i
                while j < len(rows) and closure[j] is not None and closure[j] < CLOSE_MPS:
                    j += 1
                if j - i >= DRIFT_TICKS:
                    runs_found.append((i, j))
                i = max(j, i + 1)
            else:
                i += 1

        if runs_found:
            n_with_drift += 1
            for a, b in runs_found:
                drift_lens.append(b - a)
                drift_speeds.append(sum(spd[a:b]) / (b - a))
                if dist[a] is not None:
                    drift_ranges.append(dist[a])
                if any(a - LOOKBACK <= s <= a for s in switches):
                    drift_after_switch += 1

        # base rate: how often does a switch lead to a drift within LOOKBACK?
        for s in switches:
            switch_windows_examined += 1
            if any(s <= a <= s + LOOKBACK for a, _b in runs_found):
                post_switch_drift += 1

    print(f"{n_flights} flights scored")
    print()
    print(f"NOT-CLOSING drift (closure < {CLOSE_MPS} m/s for >= {DRIFT_TICKS} ticks, after tick {SETTLE}):")
    print(f"  flights with >=1 drift : {n_with_drift} / {n_flights} "
          f"= {100.0*n_with_drift/max(1,n_flights):.1f}%")
    print(f"  total drift episodes   : {len(drift_lens)}")
    if drift_lens:
        s = sorted(drift_lens)
        print(f"  drift length (ticks)   : median {s[len(s)//2]}  p95 {s[min(len(s)-1,int(.95*len(s)))]}"
              f"  max {s[-1]} (~{s[-1]/29.0:.1f} s)")
        sp = sorted(drift_speeds)
        print(f"  SPEED during drift     : median {sp[len(sp)//2]:.2f} m/s "
              f"(so it is {'MOVING, not hovering' if sp[len(sp)//2] > 2.0 else 'genuinely slow'})")
    if drift_ranges:
        d = sorted(drift_ranges)
        print(f"  gate range at drift    : p5 {d[max(0,int(.05*len(d))-1)]:.1f}  "
              f"median {d[len(d)//2]:.1f}  p95 {d[min(len(d)-1,int(.95*len(d)))]:.1f} m")
    print()
    print(f"ASSOCIATION WITH TARGET SWITCHES ({n_switch_total} switches total):")
    print(f"  drifts preceded by a switch within {LOOKBACK} ticks: "
          f"{drift_after_switch} / {len(drift_lens)}"
          + (f" = {100.0*drift_after_switch/len(drift_lens):.1f}%" if drift_lens else ""))
    print(f"  switches followed by a drift within {LOOKBACK} ticks: "
          f"{post_switch_drift} / {switch_windows_examined}"
          + (f" = {100.0*post_switch_drift/max(1,switch_windows_examined):.1f}%"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
