"""Find the actual slow flights and print their context, instead of an aggregate null.

An aggregate "0.3% of flights" does not refute a pilot who watched one drone sit there --
a rare event is still an event, and the one instance is more informative than the rate.
This ranks every flight by its SLOWEST sustained window and dumps the context around it:
what the seeker was emitting, whether the gate index had just advanced, and whether the
drone was blind.

Integrity check first: obs[0:3] is the ESTIMATOR's body-FLU velocity, which is what the
policy acts on but not necessarily what the drone did. kf_pos_ned gives an independent
position-derived speed. If the two disagree badly, the estimator is the story.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs")
SETTLE = 30
WIN = 15


def main() -> int:
    scored = []
    dis_obs, dis_kf = [], []

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        if len(rows) < SETTLE + WIN + 5:
            continue
        t = [r["sim_time_ns"] / 1e9 for r in rows]
        spd = [(r["obs"][0] ** 2 + r["obs"][1] ** 2 + r["obs"][2] ** 2) ** 0.5 for r in rows]

        # independent position-derived speed
        kf = [r.get("kf_pos_ned") for r in rows]
        kspd = [None] * len(rows)
        for i in range(1, len(rows)):
            if not (kf[i] and kf[i - 1]):
                continue
            dt = t[i] - t[i - 1]
            if dt <= 0:
                continue
            d = sum((kf[i][a] - kf[i - 1][a]) ** 2 for a in range(3)) ** 0.5
            kspd[i] = d / dt
        for a, b in zip(spd, kspd):
            if b is not None:
                dis_obs.append(a)
                dis_kf.append(b)

        # slowest sustained window after settle
        best = None
        for i in range(SETTLE, len(rows) - WIN):
            m = sum(spd[i:i + WIN]) / WIN
            if best is None or m < best[0]:
                best = (m, i)
        if best:
            scored.append((best[0], best[1], run.name, rows))

    scored.sort()
    n = len(dis_obs)
    mo = sorted(dis_obs)[n // 2]
    mk = sorted(dis_kf)[n // 2]
    print(f"INTEGRITY: median obs-speed {mo:.2f} m/s vs median kf-derived {mk:.2f} m/s "
          f"({'consistent' if abs(mo - mk) < 1.5 else 'DISAGREE -- estimator suspect'})")
    print()
    print(f"{len(scored)} flights. Five SLOWEST sustained {WIN}-tick windows after tick {SETTLE}:")
    print()
    for mean_spd, i, name, rows in scored[:5]:
        r = rows[i]
        gi = r.get("gate_index")
        gi_end = rows[min(len(rows) - 1, i + WIN)].get("gate_index")
        seen = sum(1 for q in rows[i:i + WIN] if q.get("pose_seen"))
        d0 = r.get("dist")
        print(f"  {name}")
        print(f"    slowest window: mean {mean_spd:.2f} m/s at tick {i} "
              f"(t={rows[i]['sim_time_ns']/1e9 - rows[0]['sim_time_ns']/1e9:.1f} s), "
              f"flight len {len(rows)}")
        print(f"    gate_index {gi} -> {gi_end} | gate locked on {seen}/{WIN} ticks "
              f"| emitted range {d0 if d0 is None else round(float(d0),1)} m")
        # what did it do right after?
        j = min(len(rows) - 1, i + WIN)
        later = rows[j]
        print(f"    after the window: speed "
              f"{(later['obs'][0]**2+later['obs'][1]**2+later['obs'][2]**2)**0.5:.2f} m/s, "
              f"gate_index {later.get('gate_index')}, pose_seen {later.get('pose_seen')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
