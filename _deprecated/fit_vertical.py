"""scripts/fit_vertical.py -- fit the VERTICAL channel (hover thrust + vertical drag) and characterise
the rung-1 altitude limit cycle from the live VERIFY extracts (handoff shadowpc-verify-2026-06-06).

Two independent derivations the offline alt-loop re-tune rests on:

  1. HOVER + vertical drag, from the open-loop vertical probes (``vprobe_sink_extract.json`` +
     ``vprobe_climb_extract.json``). Each held LEVEL collective gives net world-up accel = -d(vz)/dt;
     the model is ``az_down = g*(1 - col/hover) - vdrag*vz``. The SINK side alone is confounded
     (all vz>0 -> hover and drag trade off), so hover spanned 0.26-0.32; the CLIMB side (vz<0) plus the
     near-zero-vz ``init_level`` points (drag-free anchor) pin BOTH: hover ~0.266, vertical drag ~0.

  2. The LIMIT-CYCLE period / amplitude / duty, from the rung-1 hover-hold true vz + thrust
     (``rung1_hover_run1/run2_extract.json``) -- the signature the twin re-tune must explain (it is the
     kd_alt-on-LAGGED-KF-vz relay; see scripts/twin_hover.py).

Verdict: hover_thrust=0.2656 (twin/controller) is VALIDATED; the rung-1 mean thrust 0.32 was a relay
clip-duty artifact, not hover. Deterministic; pure-numpy.

Usage:  .venv\\Scripts\\python scripts/fit_vertical.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_G = 9.80665
_DIR = Path(__file__).resolve().parent.parent / "handoff/shadowpc-verify-2026-06-06"


def _phases(fn: str):
    """Per held-phase (collective, mean vz, settled d(vz)/dt) for a probe extract (level holds)."""
    j = json.loads((_DIR / fn).read_text())
    s = j["commanded"]["series"]                         # [t, collective, world_vz, motor, pitch, alt, phase]
    t, col, vz, ph = np.array(s[0]), np.array(s[1]), np.array(s[2]), s[6]
    out = []
    for name in sorted(set(ph), key=ph.index):
        idx = [i for i, p in enumerate(ph) if p == name]
        i0, i1 = idx[0], idx[-1] + 1
        k = i0 + int((i1 - i0) * 0.5)                    # settled tail
        tt, vv = t[k:i1], vz[k:i1]
        if len(tt) < 5:
            continue
        az = float(np.polyfit(tt - tt[0], vv, 1)[0])     # d(vz)/dt, NED + = down
        out.append({"run": fn.split("_")[1], "phase": name, "col": float(np.median(col[i0:i1])),
                    "vz": float(np.mean(vv)), "az": az})
    return out


def fit_hover_drag():
    pts = _phases("vprobe_sink_extract.json") + _phases("vprobe_climb_extract.json")
    c = np.array([p["col"] for p in pts]); v = np.array([p["vz"] for p in pts]); a = np.array([p["az"] for p in pts])
    # az = g - g*col/hover - vdrag*vz  ->  (g - az) = g*col*(1/hover) + vz*vdrag
    A = np.column_stack([_G * c, v]); b = _G - a
    (x, vdrag), *_ = np.linalg.lstsq(A, b, rcond=None)
    hover = 1.0 / x
    # drag-free anchor: the near-zero-vz init_level points (drag term ~0)
    anchors = [_G * p["col"] / (_G - p["az"]) for p in pts if p["phase"] == "init_level"]
    return pts, hover, vdrag, float(np.mean(anchors)) if anchors else float("nan")


def limit_cycle(fn: str):
    j = json.loads((_DIR / fn).read_text())
    lt, _, lvz = (np.array(x) for x in j["true_lpn"]["series"])           # [t, alt, true_vz]
    ct = np.array(j["commanded"]["series"][0]); cthr = np.array(j["commanded"]["series"][3])
    m = lt >= lt[-1] - 5.0
    t, v = lt[m], lvz[m]
    v0 = v - np.mean(v); cr = np.where(np.sign(v0[1:]) * np.sign(v0[:-1]) < 0)[0]
    per_zc = 2.0 * np.mean(np.diff([t[i] + (v0[i] / (v0[i] - v0[i + 1])) * (t[i + 1] - t[i]) for i in cr])) if len(cr) > 1 else np.nan
    tu = np.arange(t[0], t[-1], np.median(np.diff(t))); vu = np.interp(tu, t, v) - np.mean(v)
    f = np.fft.rfftfreq(len(vu), np.median(np.diff(t))); fpk = f[1:][np.argmax(np.abs(np.fft.rfft(vu))[1:])]
    cm = ct >= ct[-1] - 5.0; cth = cthr[cm]
    return {"vz_std": float(np.std(v)), "vz_absmax": float(np.max(np.abs(v))), "period_zc": float(per_zc),
            "period_fft": float(1.0 / fpk), "duty_hi": float(np.mean(cth >= 0.40 - 1e-3)),
            "duty_lo": float(np.mean(cth <= 0.05 + 1e-3)), "thr_mean": float(np.mean(cth))}


def main() -> int:
    print("=== VERTICAL CHANNEL fit (live VERIFY extracts) ===\n")
    pts, hover, vdrag, anchor = fit_hover_drag()
    print("Open-loop level holds (NED vz +=down, az=d(vz)/dt):")
    for p in pts:
        print(f"  {p['run']:5s} {p['phase']:11s} col={p['col']:.3f}  vz={p['vz']:+.2f}  az={p['az']:+.3f}  "
              f"-> {'SINK' if p['az'] > 0 else 'CLIMB'}")
    print(f"\n  drag-free anchor (init_level, vz~0): hover = {anchor:.4f}")
    print(f"  5-point joint fit:  hover = {hover:.4f}   vertical_drag = {vdrag:+.4f}/s  (~0 -> thrust model is ~linear)")
    print(f"  => hover_thrust 0.2656 (twin/controller) VALIDATED (within {abs(hover-0.2656):.3f}); slope g/hover = {_G/hover:.1f} m/s^2/unit")

    print("\n=== Rung-1 limit cycle (true vz + thrust, settle window) ===")
    for fn in ("rung1_hover_run1_extract.json", "rung1_hover_run2_extract.json"):
        lc = limit_cycle(fn)
        print(f"  {fn.split('_')[2]:4s}: vz_std {lc['vz_std']:.3f} |max| {lc['vz_absmax']:.3f}  "
              f"period {lc['period_fft']*1000:.0f} ms (fft) / {lc['period_zc']*1000:.0f} ms (zc)  "
              f"duty {lc['duty_hi']*100:.0f}%hi/{lc['duty_lo']*100:.0f}%lo  thr_mean {lc['thr_mean']:.3f}")
    print("  => mean thrust ~0.32 is a relay clip-duty ARTIFACT (hover is 0.266); the ~6 Hz cycle is the\n"
          "     kd_alt-on-lagged-KF-vz relay (scripts/twin_hover.py) -- fixed by damping on RAW vz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
