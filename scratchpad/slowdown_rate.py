"""How often does the "hover and wait" actually happen, and is it blindness-driven?

Corrected specification. The first two passes missed it:
  * threshold 1.5 m/s was too strict -- the behaviour sits at 1.3-1.8 m/s, and cruise is
    ~7 m/s, so a 75% slowdown reads as hovering to a human but is not < 1.5.
  * the closure test required pose_seen on CONSECUTIVE ticks, which excluded precisely the
    blind episodes where this happens.

SLOWDOWN = >= SLOW_TICKS consecutive ticks under SLOW_MPS, after SETTLE. For each, report
how much of it was BLIND (no gate fix), which is the candidate driver.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs")
SETTLE, SLOW_MPS, SLOW_TICKS = 30, 2.5, 15


def main() -> int:
    n_flights = n_hit = 0
    eps: list[tuple[int, float, float, int]] = []   # len, mean speed, blind frac, gate_index
    cruise: list[float] = []

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        if len(rows) < SETTLE + SLOW_TICKS + 5:
            continue
        n_flights += 1
        spd = [(r["obs"][0] ** 2 + r["obs"][1] ** 2 + r["obs"][2] ** 2) ** 0.5 for r in rows]
        seen = [bool(r.get("pose_seen")) for r in rows]
        cruise.extend(spd[SETTLE:])

        found = False
        i = SETTLE
        while i < len(rows):
            if spd[i] < SLOW_MPS:
                j = i
                while j < len(rows) and spd[j] < SLOW_MPS:
                    j += 1
                if j - i >= SLOW_TICKS:
                    found = True
                    blind = 1.0 - sum(seen[i:j]) / (j - i)
                    eps.append((j - i, sum(spd[i:j]) / (j - i), blind,
                                rows[i].get("gate_index", -1)))
                i = max(j, i + 1)
            else:
                i += 1
        if found:
            n_hit += 1

    c = sorted(cruise)
    print(f"{n_flights} flights.  cruise speed after tick {SETTLE}: median {c[len(c)//2]:.2f} m/s")
    print()
    print(f"SLOWDOWN = < {SLOW_MPS} m/s for >= {SLOW_TICKS} ticks (~0.5 s), after tick {SETTLE}")
    print(f"  flights affected : {n_hit} / {n_flights} = {100.0*n_hit/max(1,n_flights):.1f}%")
    print(f"  episodes         : {len(eps)}")
    if not eps:
        return 0
    lens = sorted(e[0] for e in eps)
    spds = sorted(e[1] for e in eps)
    blinds = sorted(e[2] for e in eps)
    print(f"  duration (ticks) : median {lens[len(lens)//2]}  p95 "
          f"{lens[min(len(lens)-1,int(.95*len(lens)))]}  max {lens[-1]} "
          f"(~{lens[-1]/29.0:.1f} s)")
    print(f"  speed during     : median {spds[len(spds)//2]:.2f} m/s "
          f"({100*(1-spds[len(spds)//2]/c[len(c)//2]):.0f}% below cruise)")
    print(f"  BLIND fraction   : median {blinds[len(blinds)//2]*100:.0f}%  "
          f"p25 {blinds[max(0,int(.25*len(blinds))-1)]*100:.0f}%  "
          f"p75 {blinds[min(len(blinds)-1,int(.75*len(blinds)))]*100:.0f}%")
    fully_blind = sum(1 for e in eps if e[2] >= 0.9)
    print(f"  episodes >=90% BLIND: {fully_blind} / {len(eps)} "
          f"= {100.0*fully_blind/len(eps):.0f}%")
    from collections import Counter
    print(f"  gate_index at onset : {dict(Counter(e[3] for e in eps).most_common(6))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
