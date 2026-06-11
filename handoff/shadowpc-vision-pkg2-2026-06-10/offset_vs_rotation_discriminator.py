"""Discriminate CONSTANT world offsets (map / PnP-bias) from RANGE-PROPORTIONAL rotation
(true chain-rotation calibration) per error channel, per gate.

For each good fix, express off in the LEVER frame: u_along = L_hat (depth), u_horiz =
normalize(L_hat x down) (horizontal cross-track, 'yaw channel'), u_vert = L_hat x u_horiz
('pitch channel'). For each gate and channel, fit off_channel = a + b*range:
  * a dominant, b~0   -> constant offset (map error / PnP bias)  -> NOT a rotation cal
  * b dominant, a~0   -> rotation (angle = atan(b))              -> calibratable IF same across gates
Also pool across gates for the global split, and report close vs far depth bias.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                      # noqa: E402
from racer.navigator import load_track_map         # noqa: E402

CHAR_DIR = ROOT / "handoff/shadowpc-assoc-flipfix-2026-06-09"
PG_DIR = ROOT / "handoff/perception-char-2026-06-08/pg"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
DOWN = np.array([0.0, 0.0, 1.0])


def load_good():
    gates = {g.gate_id: g for g in load_track_map(MAP, corner_to_center=True)}
    out = []
    for gi in range(6):
        char = json.loads((CHAR_DIR / f"char_g{gi}_robust_k1.0.json").read_text())
        meta = json.loads((PG_DIR / f"course_g{gi}/frames.json").read_text())
        by_fid = {fr["frame_id"]: fr for fr in meta["frames"]}
        for r in char["rows"]:
            if not (r.get("associated") and "world_fix_err_m" in r):
                continue
            if r["world_fix_err_m"] >= 1.5 or r["n_corners"] != 4:
                continue
            fr = by_fid[r["frame_id"]]
            p = np.asarray(fr["drone_position_ned"], float)
            off = np.asarray(r["off_ned"], float)
            L = gates[r["gate_id"]].position_ned - (p + off)
            nL = float(np.linalg.norm(L)); Lh = L / nL
            uh = np.cross(Lh, DOWN); uh /= np.linalg.norm(uh)
            uv = np.cross(Lh, uh)                     # completes the frame (vertical-ish)
            out.append(dict(gate_id=r["gate_id"], range_m=nL,
                            along=float(off @ Lh), horiz=float(off @ uh), vert=float(off @ uv)))
    return out


def fit_line(x, y):
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * np.asarray(x)
    se = np.std(np.asarray(y) - yhat)
    n = len(x)
    sb = se / (np.std(x) * np.sqrt(n)) if np.std(x) > 0 else float("inf")
    return a, b, sb, se


def main():
    rows = load_good()
    print(f"good fixes: {len(rows)}\n")
    print("per-gate channel fits: off = a + b*range   [a in m; b in m/m -> deg; +-1sig on b]")
    print(f"{'gate':>4} {'N':>3} | {'along: a':>9} {'b(=scale%)':>10} | "
          f"{'horiz: a':>9} {'b -> deg':>14} | {'vert: a':>9} {'b -> deg':>14}")
    for g in sorted({r["gate_id"] for r in rows}):
        sub = [r for r in rows if r["gate_id"] == g]
        x = [r["range_m"] for r in sub]
        if len(sub) < 6 or np.std(x) < 1.0:
            print(f"{g:>4} {len(sub):>3}   (insufficient range spread)")
            continue
        line = f"{g:>4} {len(sub):>3} |"
        for ch in ("along", "horiz", "vert"):
            y = [r[ch] for r in sub]
            a, b, sb, _ = fit_line(x, y)
            if ch == "along":
                line += f" {a:+8.2f}m {100*b:+6.2f}+-{100*sb:4.2f}% |"
            else:
                line += f" {a:+8.2f}m {np.degrees(np.arctan(b)):+6.2f}+-{np.degrees(sb):4.2f}deg |"
        print(line)

    print("\npooled (all gates):")
    x = [r["range_m"] for r in rows]
    for ch in ("along", "horiz", "vert"):
        y = [r[ch] for r in rows]
        a, b, sb, se = fit_line(x, y)
        unit = f"{100*b:+6.2f}+-{100*sb:4.2f}%" if ch == "along" else \
               f"{np.degrees(np.arctan(b)):+6.2f}+-{np.degrees(sb):4.2f}deg"
        print(f"  {ch:>5}: a {a:+6.2f} m   b {unit}   resid-std {se:.2f} m")

    print("\npooled channel stats by range band (mean +- std, m):")
    print(f"{'band':>8} {'N':>4} {'along':>14} {'horiz':>14} {'vert':>14}")
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 30)):
        sub = [r for r in rows if lo <= r["range_m"] < hi]
        if not sub:
            continue
        line = f"{f'{lo}-{hi}m':>8} {len(sub):>4}"
        for ch in ("along", "horiz", "vert"):
            v = np.array([r[ch] for r in sub])
            line += f"  {v.mean():+5.2f}+-{v.std():4.2f}"
        print(line)

    # angular view per band: channel/range (what sigma_theta must cover at that range)
    print("\nANGULAR view by range band (channel error / range, deg):")
    print(f"{'band':>8} {'N':>4} {'horiz(yaw)':>16} {'vert(pitch)':>16}")
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 30)):
        sub = [r for r in rows if lo <= r["range_m"] < hi]
        if not sub:
            continue
        h = np.degrees([r["horiz"] / r["range_m"] for r in sub])
        v = np.degrees([r["vert"] / r["range_m"] for r in sub])
        print(f"{f'{lo}-{hi}m':>8} {len(sub):>4}  {h.mean():+5.2f}+-{h.std():4.2f}     "
              f"{v.mean():+5.2f}+-{v.std():4.2f}")


if __name__ == "__main__":
    main()
