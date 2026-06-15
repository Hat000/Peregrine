"""analyze_atspeed.py — sigma-vs-speed + terminal-window fix density from shadow_gate4 rows.

Consumes one or more shadow_gate4 *_rows.json files (each a {"rows":[...]} with per-frame
speed_mps, true_range_m, g4_bearing_deg, associated/offered/accepted, rel_cross/rel_vert/rel_along).

Outputs, restricted to a MATCHED geometry window (head-on bearing band + range band so the speed
comparison is not confounded by viewing geometry):
  (A) sigma_lat / sigma_vert / sigma_depth per SPEED bin (on accepted fixes), + a weighted linear
      fit sigma(speed) and a projection to 30 m/s vs the 0.245 m lateral ceiling.
  (B) terminal-window (last TERM_M metres) accepted-fix COUNT + per-frame fix-rate vs speed, and a
      kinematic projection of how many accepted fixes land in the last 6 m at 30 m/s @30 Hz.
  (C) effective in-plane bias (mean rel_cross/rel_vert) + one-signed depth (rel_along) bias vs speed.

Usage:
  .venv/Scripts/python.exe handoff/at-speed-sigma-2026-06-15/analysis/analyze_atspeed.py \
      --rows 'handoff/.../analysis/*_rows.json' --bearing-max 20 --range 12 28 --term-m 6
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

import numpy as np


def load_rows(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for p in sorted(globmod.glob(pat)):
            d = json.loads(Path(p).read_text())
            for r in d.get("rows", []):
                r["_src"] = Path(p).name
                rows.append(r)
    return rows


def _f(r, k):
    v = r.get(k)
    return float(v) if v is not None else float("nan")


def std_or_nan(a):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    return (float(a.std()), int(a.size)) if a.size else (float("nan"), 0)


def mean_or_nan(a):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    return (float(a.mean()), int(a.size)) if a.size else (float("nan"), 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="glob(s) for shadow_gate4 *_rows.json")
    ap.add_argument("--bearing-max", type=float, default=20.0, help="keep frames with g4_bearing<=this (head-on matched geometry)")
    ap.add_argument("--range", type=float, nargs=2, default=[12.0, 28.0], help="keep true_range in [lo,hi] m")
    ap.add_argument("--term-m", type=float, default=6.0, help="terminal-window length (m of range)")
    ap.add_argument("--speed-bins", default="0,6,10,14,18,22,26,30,40")
    ap.add_argument("--ceiling", type=float, default=0.245, help="sigma_lat ceiling for closure")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load_rows(args.rows)
    if not rows:
        print("no rows matched"); return 2
    bins = [float(x) for x in args.speed_bins.split(",")]
    rlo, rhi = args.range

    # matched-geometry accepted fixes
    acc = [r for r in rows if r.get("accepted")
           and np.isfinite(_f(r, "g4_bearing_deg")) and _f(r, "g4_bearing_deg") <= args.bearing_max
           and rlo <= _f(r, "true_range_m") <= rhi]
    print(f"loaded {len(rows)} frames from {len(set(r['_src'] for r in rows))} rows-files")
    print(f"matched-geometry ACCEPTED fixes (bearing<={args.bearing_max}, range[{rlo},{rhi}]): {len(acc)}\n")

    print("=== (A) sigma vs SPEED (accepted, matched geometry) ===")
    print(f"{'speed bin':>14} {'N':>5} {'sig_lat':>8} {'sig_vert':>9} {'sig_depth':>10} {'med_rng':>8} {'med_spd':>8}")
    fit_x, fit_yl, fit_yv, fit_w = [], [], [], []
    out_bins = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [r for r in acc if lo <= _f(r, "speed_mps") < hi]
        if len(b) < 4:
            continue
        sl, nl = std_or_nan([_f(r, "rel_cross") for r in b])
        sv, _ = std_or_nan([_f(r, "rel_vert") for r in b])
        sd, _ = std_or_nan([_f(r, "rel_along") for r in b])
        medr = float(np.median([_f(r, "true_range_m") for r in b]))
        meds = float(np.median([_f(r, "speed_mps") for r in b]))
        print(f"  [{lo:>4g},{hi:>4g})  {nl:>5} {sl:>8.3f} {sv:>9.3f} {sd:>10.3f} {medr:>8.1f} {meds:>8.2f}")
        fit_x.append(meds); fit_yl.append(sl); fit_yv.append(sv); fit_w.append(nl)
        out_bins.append(dict(lo=lo, hi=hi, n=nl, sig_lat=sl, sig_vert=sv, sig_depth=sd,
                             med_range=medr, med_speed=meds))

    proj = {}
    if len(fit_x) >= 2:
        x = np.array(fit_x); w = np.array(fit_w, float)
        for nm, y in [("lat", np.array(fit_yl)), ("vert", np.array(fit_yv))]:
            m, c = np.polyfit(x, y, 1, w=np.sqrt(w))
            p30 = m * 30 + c
            # flat-model reference = weighted mean (null hypothesis: speed-independent)
            flat = float(np.average(y, weights=w))
            print(f"  fit sigma_{nm}(v) = {m:+.5f}*v {c:+.4f}  -> @30m/s = {p30:.3f} m   "
                  f"(flat/weighted-mean = {flat:.3f})")
            proj[nm] = dict(slope=float(m), intercept=float(c), at30=float(p30), flat=flat)
        if "lat" in proj:
            v = "CLEARS" if proj["lat"]["at30"] <= args.ceiling else "EXCEEDS"
            print(f"  >>> sigma_lat @30 m/s (fit) = {proj['lat']['at30']:.3f} m vs ceiling {args.ceiling} -> {v}")
            v2 = "CLEARS" if proj["lat"]["flat"] <= args.ceiling else "EXCEEDS"
            print(f"  >>> sigma_lat flat-model    = {proj['lat']['flat']:.3f} m vs ceiling {args.ceiling} -> {v2}")

    print("\n=== (C) effective in-plane BIAS + signed depth vs speed (accepted, matched) ===")
    print(f"{'speed bin':>14} {'N':>5} {'bias_lat':>9} {'bias_vert':>10} {'bias_depth':>11}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [r for r in acc if lo <= _f(r, "speed_mps") < hi]
        if len(b) < 4:
            continue
        bl, n = mean_or_nan([_f(r, "rel_cross") for r in b])
        bv, _ = mean_or_nan([_f(r, "rel_vert") for r in b])
        bd, _ = mean_or_nan([_f(r, "rel_along") for r in b])
        print(f"  [{lo:>4g},{hi:>4g})  {n:>5} {bl:>+9.3f} {bv:>+10.3f} {bd:>+11.3f}")

    # (B) terminal-window analysis: per recording-source the last term_m of range
    print(f"\n=== (B) TERMINAL-WINDOW (last {args.term_m} m of range) accepted-fix density vs speed ===")
    print("  per source-lap: terminal speed, frames-in-window, accepted-in-window, fix-rate")
    print(f"{'source':>32} {'term_spd':>9} {'fr_in':>6} {'acc_in':>7} {'rate':>6}")
    term_pts = []
    for src in sorted(set(r["_src"] for r in rows)):
        sr = [r for r in rows if r["_src"] == src]
        win = [r for r in sr if _f(r, "true_range_m") <= args.term_m and _f(r, "true_range_m") > 0]
        if not win:
            continue
        # terminal speed = median speed of frames in the last window that are moving
        spd = float(np.median([_f(r, "speed_mps") for r in win]))
        fr_in = len(win)
        acc_in = sum(1 for r in win if r.get("accepted"))
        rate = acc_in / fr_in if fr_in else float("nan")
        print(f"{src[:32]:>32} {spd:>9.2f} {fr_in:>6} {acc_in:>7} {rate:>6.2f}")
        term_pts.append(dict(src=src, term_speed=spd, frames=fr_in, accepted=acc_in, rate=rate))

    # kinematic projection to 30 m/s: per-range accept probability (matched geometry) * frames in last 6 m
    print(f"\n  --- kinematic projection to 30 m/s in the last {args.term_m} m ---")
    # accept probability per metre of range from ALL matched (bearing) frames in the window range
    near = [r for r in rows if np.isfinite(_f(r, "g4_bearing_deg")) and _f(r, "g4_bearing_deg") <= args.bearing_max
            and 0 < _f(r, "true_range_m") <= args.term_m]
    if near:
        p_acc = sum(1 for r in near if r.get("accepted")) / len(near)
        # frames available in last term_m at 30 m/s @ fps: time = term_m/30, frames = time*fps
        t_win = args.term_m / 30.0
        frames_avail = t_win * args.fps
        exp_acc = p_acc * frames_avail
        print(f"  per-frame accept-prob in last {args.term_m} m (head-on, all speeds pooled): {p_acc:.3f} (N={len(near)})")
        print(f"  at 30 m/s the last {args.term_m} m lasts {t_win*1000:.0f} ms = {frames_avail:.1f} frames @ {args.fps:g} Hz")
        print(f"  => expected accepted fixes in terminal window @30 m/s ~= {exp_acc:.2f}")
        print(f"  => terminal effective fix-RATE @30 m/s ~= {p_acc:.2f}  (requirement: >= 0.50)")
        proj["terminal"] = dict(p_accept=p_acc, n=len(near), frames_avail_30=frames_avail,
                                expected_accepted_30=exp_acc)

    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(n_rows=len(rows), n_accepted_matched=len(acc), bins=out_bins, projection=proj,
                 terminal_per_lap=term_pts, params=vars(args)), indent=2, default=float))
        print(f"\nwrote -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
