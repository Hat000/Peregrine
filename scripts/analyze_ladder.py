"""Offline analyzer for a velocity_ladder ladder.json: prints a downsampled per-phase trace
(vz/motors/z shape for settle-vs-runaway) and a per-axis v_lpn-vs-v_odo divergence breakdown
(to tell a real telemetry split from a body-vs-world FRAME difference). Read-only; no sim.

Usage: python scripts/analyze_ladder.py data/runs/<session>            (or .../ladder.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = Path(sys.argv[1])
    if p.is_dir():
        p = p / "ladder.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    m, R = d["meta"], d["records"]
    print(f"== {m.get('label')} yaw={m.get('yaw_mode')} rate-target n_rec={len(R)} aborted={m.get('aborted')}")
    nt = m.get("null_test")            # velocity_ladder schema: {axis: {max: ..}}
    if nt:
        nl = f"{nt['v_lpn']['max']}/{nt['v_odo']['max']}/{nt['v_fd']['max']}"
    else:                              # descend_arrest schema: null_max = {axis: float}
        nm = m.get("null_max", {})
        nl = f"{nm.get('v_lpn')}/{nm.get('v_odo')}/{nm.get('v_fd')}"
    print(f"   null v_lpn/odo/fd max: {nl}  telemetry_live={m.get('telemetry_live')}  "
          f"echo_mask={m.get('command_echo',{}).get('mask')}  verdict={m.get('suggested_verdict')}")
    if not R:
        return 0
    # group by phase
    phases: dict[str, list] = {}
    for r in R:
        phases.setdefault(r["phase"], []).append(r)
    for name, recs in phases.items():
        t0 = recs[0]["sim_time_ns"]
        dur = (recs[-1]["sim_time_ns"] - t0) / 1e9
        n = max(1, len(recs) // 30)
        has_w = recs[0].get("v_odo_world") is not None
        print(f"\n-- PHASE {name}: cmd_v={recs[0]['cmd_v']} ticks={len(recs)} dur={dur:.2f}s "
              f"(odo_world={'yes' if has_w else 'NO -> raw body shown'})")
        print("   t_ms   relz   vzL  vzOw  vzF |  vxL  vxOw | pitch  mtr_mean  |dvx_w| |dvz_w|")
        for i, r in enumerate(recs):
            if i % n and i != len(recs) - 1:
                continue
            ms = (r["sim_time_ns"] - t0) / 1e6
            vL, vF = r["v_lpn"], r["v_fd"]
            vO = r.get("v_odo_world") or r["v_odo"]   # prefer world-rotated; fall back to raw body
            relz = r["rel_pos"][2] if r["rel_pos"] else float("nan")
            pit = r["rpy_deg"][1]
            mt = r["actuators"]
            mtr = (sum(mt) / len(mt)) if mt else float("nan")
            dvx = abs(vL[0] - vO[0]) if (vL and vO) else float("nan")
            dvz = abs(vL[2] - vO[2]) if (vL and vO) else float("nan")
            print(f"  {ms:5.0f} {relz:+.3f} {vL[2]:+.2f} {vO[2]:+.2f} {vF[2]:+.2f} | "
                  f"{vL[0]:+.2f} {vO[0]:+.2f} | {pit:+5.0f}  {mtr:.3f}   {dvx:.2f}    {dvz:.2f}")
        # axis-resolved divergence summary, in the COMMON world frame (v_lpn vs v_odo_world)
        def _wodo(r):
            return r.get("v_odo_world") or r["v_odo"]
        rr = [r for r in recs if r["v_lpn"] and _wodo(r)]
        dvx = np.nanmax([abs(r["v_lpn"][0] - _wodo(r)[0]) for r in rr] or [np.nan])
        dvy = np.nanmax([abs(r["v_lpn"][1] - _wodo(r)[1]) for r in rr] or [np.nan])
        dvz = np.nanmax([abs(r["v_lpn"][2] - _wodo(r)[2]) for r in rr] or [np.nan])
        print(f"   max|v_lpn - v_odo_world| per axis:  vx={dvx:.2f}  vy={dvy:.2f}  vz={dvz:.2f}   "
              f"(small now = the two sim sources AGREE once in a common frame -> telemetry trustworthy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
