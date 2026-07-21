"""Does M1's handoff-spawn training box actually cover the wire's handoff?

M1 (v1.7) spawns the drone mid-air at the moment of handoff and trains from there:
    handoff_range_lo_m = 9.5    handoff_range_hi_m  = 11.5
    handoff_gate_dz_lo_m = -1.5 handoff_gate_dz_hi_m = +1.5
    handoff_min_agl_m = 0.5     handoff_yaw_jitter_rad = 0.25

If the wire hands over OUTSIDE that box, M1 trained a confident prior for a condition
that never occurs, and applies it anyway -- which would explain a HARDER start dive
rather than a softer one.  Measured here from every flight's first post-release tick,
using rel_flu (TRUE unflipped body FLU: +x forward, +y left, +z up).
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs")

BOX_RANGE = (9.5, 11.5)
BOX_DZ = (-1.5, 1.5)


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]


def main() -> int:
    rng: list[float] = []
    dz: list[float] = []
    dy: list[float] = []
    in_box = 0

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        rel = next((r for r in rows
                    if not r.get("assist", False) and r.get("pose_seen") and r.get("rel_flu")),
                   None)
        if rel is None:
            continue
        x, y, z = rel["rel_flu"]
        r = (x * x + y * y + z * z) ** 0.5
        rng.append(r)
        dz.append(z)
        dy.append(y)
        if BOX_RANGE[0] <= r <= BOX_RANGE[1] and BOX_DZ[0] <= z <= BOX_DZ[1]:
            in_box += 1

    n = len(rng)
    print(f"{n} flights with a gate fix at the first post-release tick")
    print()
    print(f"  {'quantity':<22}{'p5':>9}{'median':>9}{'p95':>9}   M1 training box")
    print("  " + "-" * 68)
    print(f"  {'gate range (m)':<22}{pct(rng,.05):>9.2f}{pct(rng,.5):>9.2f}{pct(rng,.95):>9.2f}"
          f"   [{BOX_RANGE[0]}, {BOX_RANGE[1]}]")
    print(f"  {'gate dz, +up (m)':<22}{pct(dz,.05):>9.2f}{pct(dz,.5):>9.2f}{pct(dz,.95):>9.2f}"
          f"   [{BOX_DZ[0]}, {BOX_DZ[1]}]")
    print(f"  {'gate dy, +left (m)':<22}{pct(dy,.05):>9.2f}{pct(dy,.5):>9.2f}{pct(dy,.95):>9.2f}"
          f"   (not constrained)")
    print()
    print(f"  flights INSIDE the M1 box (range AND dz): {in_box} / {n} "
          f"= {100.0 * in_box / max(1, n):.1f}%")
    above = sum(1 for z in dz if z > BOX_DZ[1])
    print(f"  flights with the gate ABOVE the box top (+{BOX_DZ[1]} m): {above} / {n} "
          f"= {100.0 * above / max(1, n):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
