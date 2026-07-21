"""What is the ACTUAL miss geometry at gate crossings on the wire?

M4 tightens the training aperture from 1.0 -> 0.75 m weighted miss, which is a purely
LATERAL squeeze: |lat| <= 0.375 m at pass_margin_lat_weight=2.0. Before crediting that
with fixing side slams, check the obvious M1-style question: does 0.75 m even describe
where the misses ARE?

At closest approach the gate is abeam, so rel_flu[0] ~ 0 and the remaining components ARE
the miss vector: rel_flu[1] = lateral (+left), rel_flu[2] = vertical (+up). For each gate
advance we scan back for the tick of minimum range and read the decomposition there.

Reported separately for gates that were PASSED (an advance followed) -- the misses we
survived -- so the lateral spread can be compared against the 0.375 m target directly.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs")
LOOKBACK = 25          # ticks before the advance to search for closest approach
WEIGHT_LAT = 2.0       # pass_margin_lat_weight


def q(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(p * len(s))))]


def main() -> int:
    lat, vert, weighted, radial = [], [], [], []
    n_adv = 0

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        if len(rows) < 10:
            continue
        for i in range(1, len(rows)):
            gi, gp = rows[i].get("gate_index"), rows[i - 1].get("gate_index")
            if gi is None or gp is None or gi <= gp:
                continue
            n_adv += 1
            # closest approach in the window before the advance
            best = None
            for j in range(max(0, i - LOOKBACK), i):
                r = rows[j]
                if not r.get("pose_seen") or not r.get("rel_flu"):
                    continue
                d = r.get("dist")
                if d is None:
                    continue
                if best is None or float(d) < best[0]:
                    best = (float(d), r["rel_flu"])
            if best is None:
                continue
            _d, rel = best
            la, ve = abs(float(rel[1])), abs(float(rel[2]))
            lat.append(la)
            vert.append(ve)
            radial.append((la ** 2 + ve ** 2) ** 0.5)
            weighted.append(((WEIGHT_LAT * la) ** 2 + ve ** 2) ** 0.5)

    n = len(lat)
    print(f"{n_adv} gate advances across the log set; {n} with a usable closest-approach fix")
    print()
    print(f"  {'quantity':<34}{'median':>9}{'p90':>9}{'p95':>9}{'max':>9}")
    print("  " + "-" * 70)
    for name, xs in (("|lateral| (m)", lat), ("|vertical| (m)", vert),
                     ("radial (m)", radial), ("weighted miss (lat x2) (m)", weighted)):
        print(f"  {name:<34}{q(xs,.5):>9.2f}{q(xs,.90):>9.2f}{q(xs,.95):>9.2f}{max(xs):>9.2f}")
    print()
    over_lat = sum(1 for v in lat if v > 0.375)
    over_w = sum(1 for v in weighted if v > 0.75)
    print(f"  passes with |lateral| > 0.375 m (M4's implied lateral bound): "
          f"{over_lat} / {n} = {100.0*over_lat/max(1,n):.0f}%")
    print(f"  passes with weighted miss > 0.75 m (M4's target):             "
          f"{over_w} / {n} = {100.0*over_w/max(1,n):.0f}%")
    print()
    lat_dom = sum(1 for a, b in zip(lat, vert) if a > b)
    print(f"  passes where LATERAL exceeds VERTICAL error: {lat_dom} / {n} "
          f"= {100.0*lat_dom/max(1,n):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
