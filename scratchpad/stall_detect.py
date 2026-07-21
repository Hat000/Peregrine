"""Is the pilot's "it skips a gate, re-locks, then hovers and waits" visible in the logs?

Fengyou's hypothesis: training spawns are ALWAYS from rest at the pad
(standing_start_frac=1.0), so "at rest" is perfectly correlated with "start of episode,
gate ~10 m dead ahead".  A mid-course stall is then off-distribution and the policy has
no learned behaviour for "restart from rest with the gate somewhere odd".

Measured here, per flight:
  * STALL   = speed < STALL_MPS for >= STALL_TICKS consecutive ticks, starting after
              SETTLE ticks (so the launch itself is never counted).
  * SWITCH  = the emitted slot0 target jumps by > SWITCH_M between consecutive fixes,
              i.e. the seeker changed which gate it is flying at.
  * The association: does a stall tend to FOLLOW a switch?

obs[0:3] is body-FLU velocity, so speed = |obs[0:3]|.  'dist' is the emitted slot0 range.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs")

STALL_MPS = 1.5
STALL_TICKS = 15          # ~0.5 s at ~29 Hz
SETTLE = 30               # ignore the launch transient
SWITCH_M = 5.0
LOOKBACK = 40             # ticks before a stall in which a switch counts as "preceding"


def speed(o: list[float]) -> float:
    return (o[0] ** 2 + o[1] ** 2 + o[2] ** 2) ** 0.5


def main() -> int:
    n_flights = n_with_stall = n_stall_after_switch = 0
    stall_lens: list[int] = []
    stall_dists: list[float] = []
    n_switches = 0
    all_speed: list[float] = []

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        if len(rows) < SETTLE + STALL_TICKS + 5:
            continue
        n_flights += 1

        spd = [speed(r["obs"]) for r in rows]
        all_speed.extend(spd[SETTLE:])

        # target switches
        switches = []
        prev = None
        for i, r in enumerate(rows):
            if not r.get("pose_seen"):
                continue
            d = r.get("dist")
            if d is None:
                continue
            if prev is not None and abs(d - prev[1]) > SWITCH_M:
                switches.append(i)
            prev = (i, d)
        n_switches += len(switches)

        # stalls
        flight_stalls = []
        i = SETTLE
        while i < len(rows):
            if spd[i] < STALL_MPS:
                j = i
                while j < len(rows) and spd[j] < STALL_MPS:
                    j += 1
                if j - i >= STALL_TICKS:
                    flight_stalls.append((i, j))
                i = j
            else:
                i += 1

        if flight_stalls:
            n_with_stall += 1
            for a, b in flight_stalls:
                stall_lens.append(b - a)
                d = rows[a].get("dist")
                if d is not None:
                    stall_dists.append(float(d))
                if any(a - LOOKBACK <= s <= a for s in switches):
                    n_stall_after_switch += 1

    print(f"{n_flights} flights scored (speed = |obs[0:3]|, body FLU)")
    print()
    print(f"  sustained stalls (< {STALL_MPS} m/s for >= {STALL_TICKS} ticks, after tick {SETTLE}):")
    print(f"    flights with >=1 stall : {n_with_stall} / {n_flights} "
          f"= {100.0*n_with_stall/max(1,n_flights):.1f}%")
    print(f"    total stalls           : {len(stall_lens)}")
    if stall_lens:
        s = sorted(stall_lens)
        print(f"    stall length (ticks)   : median {s[len(s)//2]}  max {s[-1]} "
              f"(~{s[-1]/29.0:.1f} s)")
    if stall_dists:
        d = sorted(stall_dists)
        print(f"    gate range AT stall (m): p5 {d[max(0,int(.05*len(d))-1)]:.1f}  "
              f"median {d[len(d)//2]:.1f}  p95 {d[min(len(d)-1,int(.95*len(d)))]:.1f}")
    print()
    print(f"  target switches (|d dist| > {SWITCH_M} m): {n_switches} total")
    print(f"    stalls preceded by a switch within {LOOKBACK} ticks: "
          f"{n_stall_after_switch} / {len(stall_lens)}"
          + (f" = {100.0*n_stall_after_switch/len(stall_lens):.1f}%" if stall_lens else ""))
    if all_speed:
        a = sorted(all_speed)
        print()
        print(f"  speed after tick {SETTLE}: p5 {a[int(.05*len(a))]:.2f}  "
              f"median {a[len(a)//2]:.2f}  p95 {a[int(.95*len(a))]:.2f} m/s")
        print(f"  fraction of all post-launch ticks below {STALL_MPS} m/s: "
              f"{100.0*sum(1 for v in all_speed if v < STALL_MPS)/len(all_speed):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
