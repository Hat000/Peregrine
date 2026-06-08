"""Gate-miss + initial-transient analyzer for the 2026-06-07 moving runs (rung 2/3).

Loads a fly_vq1 commands.jsonl trajectory (full per-tick pos/vel/att/thrust/body_rate) + the saved
gate map (corner_to_center) and reports:
  * per-gate plane-crossing IN-PLANE miss + 3D closest approach (vs the ~0.75 m inner half-opening),
  * the hover->forward INITIAL TRANSIENT (peak lateral excursion / bank / climb / commanded rate)
    -- the run-to-run variability the GUI flagged,
  * the DESCENT max sink rate + the alt-loop THROTTLE cycling during cruise (audible the whole race).

Usage: .venv\\Scripts\\python handoff/shadowpc-reverify-2026-06-07/scratch/gate_analyze.py <run_dir> [map.json]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from racer.navigator import load_track_map

DEF_MAP = "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"


def load_traj(run: Path):
    rows = [json.loads(l) for l in open(run / "commands.jsonl") if l.strip()]
    t = np.array([r["sim_t"] for r in rows], dtype=np.float64) * 1e-9
    return dict(t=t - t[0],
                pos=np.array([r["pos"] for r in rows], dtype=np.float64),
                vel=np.array([r["vel"] for r in rows], dtype=np.float64),
                roll=np.degrees([r["roll"] for r in rows]),
                thr=np.array([r["thrust"] for r in rows], dtype=np.float64),
                br=np.array([r["body_rate"] for r in rows], dtype=np.float64),
                state=[r["state"] for r in rows])


def transient(tr, window=6.0):
    run_i = next((i for i, s in enumerate(tr["state"]) if s == "RUN"), 0)
    t0 = tr["t"][run_i]
    m = (tr["t"] >= t0) & (tr["t"] <= t0 + window)
    pos, roll, br, tt = tr["pos"][m], tr["roll"][m], tr["br"][m], tr["t"][m]
    y, alt = pos[:, 1], -pos[:, 2]
    ky, kalt, kr = int(np.argmax(np.abs(y))), int(np.argmax(alt)), int(np.argmax(np.abs(roll)))
    brn = np.linalg.norm(br, axis=1); kb = int(np.argmax(brn))
    print(f"  INITIAL TRANSIENT (first {window:.0f}s of RUN, from t={t0:.1f}s):")
    print(f"    peak |lateral y|     = {abs(y[ky]):.2f} m (y={y[ky]:+.2f}) @t={tt[ky]:.1f}s")
    print(f"    peak altitude        = {alt[kalt]:.2f} m @t={tt[kalt]:.1f}s")
    print(f"    peak |measured bank| = {abs(roll[kr]):.1f} deg @t={tt[kr]:.1f}s")
    print(f"    peak |cmd body-rate| = {brn[kb]:.2f} rad/s @t={tt[kb]:.1f}s")


def gate_misses(tr, gates):
    pos = tr["pos"]
    print("  PER-GATE (in-plane = offset in the opening plane; PASS if < inner half-opening):")
    for i, g in enumerate(gates):
        c = np.asarray(g.position_ned, dtype=np.float64)
        R = np.asarray(g.R_world_gate, dtype=np.float64)
        right, down, nrm = R[:, 0], R[:, 1], R[:, 2]
        d = (pos - c) @ nrm
        dist3 = np.linalg.norm(pos - c, axis=1)
        cr = np.where(np.sign(d[1:]) * np.sign(d[:-1]) < 0)[0]
        best = None
        for k in cr:
            frac = d[k] / (d[k] - d[k + 1]) if d[k] != d[k + 1] else 0.0
            cp = pos[k] + frac * (pos[k + 1] - pos[k])
            rel = cp - c
            tot = float(np.linalg.norm(cp - c))
            if best is None or tot < best[0]:
                best = (tot, float(np.hypot(rel @ right, rel @ down)), float(rel @ right), float(rel @ down), float(tr["t"][k]))
        half = g.inner_size_m / 2.0
        if best is not None:
            tot, ip, mr, md, tc = best
            flag = "PASS" if ip < half else f"MISS(> {half:.2f})"
            print(f"  g{i} ({c[0]:7.1f},{c[1]:5.1f},{c[2]:6.1f}): in-plane {ip:.2f} m "
                  f"(right {mr:+.2f}, down {md:+.2f}) @t={tc:.1f}s | closest3D {tot:.2f} m  -> {flag}")
        else:
            print(f"  g{i} ({c[0]:7.1f},{c[1]:5.1f},{c[2]:6.1f}): no crossing; closest3D {dist3.min():.2f} m")


def descent_throttle(tr):
    vz = tr["vel"][:, 2]
    print(f"  DESCENT: max sink {vz.max():+.2f} m/s | max climb {vz.min():+.2f} m/s (NED +=down)")
    t, thr = tr["t"], tr["thr"]
    m = (t >= 12.0) & (t <= t[-1] - 3.0)
    if m.sum() > 20:
        th = thr[m]
        print(f"  THROTTLE (cruise 12s..end-3s): mean {th.mean():.3f} std {th.std():.3f} "
              f"hi(>=0.6) {np.mean(th >= 0.6 - 1e-3) * 100:.0f}% lo(<=0.05) {np.mean(th <= 0.05 + 1e-3) * 100:.0f}%")


def main():
    run = Path(sys.argv[1]); mp = sys.argv[2] if len(sys.argv) > 2 else DEF_MAP
    tr = load_traj(run); gates = load_track_map(mp, corner_to_center=True)
    print(f"=== {run.name}  ({len(tr['t'])} ticks, {tr['t'][-1]:.1f}s, final {tr['state'][-1]}) ===")
    transient(tr)
    gate_misses(tr, gates)
    descent_throttle(tr)
    print()


if __name__ == "__main__":
    raise SystemExit(main())
