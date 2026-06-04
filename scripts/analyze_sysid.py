"""Offline: read a rate_sysid run's commands.jsonl and report the inner-loop plant.

Self-contained (no sim): segments the log by phase, then per probed axis fits the STEADY rate
gain (measured ODOMETRY rate vs the RAW commanded rate -> signed gain: <0 confirms the sim's
roll/yaw inversion, |gain| is the scaling to feed-forward) and the step OVERSHOOT/tau (so we
know whether an apparent 2.7x is a steady gain or a transient overshoot). For --mode hover it
fits the LEVEL hover thrust from the vertical-velocity slope at each thrust level.

Writes ``sysid_result.json`` into the session dir (the controller build consumes it).

Usage:  python scripts/analyze_sysid.py data/runs/<stamp>_<label> [--settle-frac 0.4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.sysid import fit_hover_thrust, fit_rate_gain, step_response_metrics

_AXIS_NAME = {0: "roll", 1: "pitch", 2: "yaw"}
_EXPECTED_SIGN = {0: -1, 1: +1, 2: -1}   # measured: sim inverts roll + yaw


def _load_rows(session: Path) -> list[dict]:
    rows = []
    with open(session / "commands.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _segments(rows: list[dict]) -> list[dict]:
    """Consecutive runs of identical phase name -> segment dicts with index/time bounds."""
    segs: list[dict] = []
    for i, r in enumerate(rows):
        if not segs or r["phase"] != segs[-1]["phase"]:
            segs.append({"phase": r["phase"], "kind": r["kind"], "axis": r["axis"],
                         "i0": i, "i1": i, "t0": r["t"], "t1": r["t"]})
        else:
            segs[-1]["i1"] = i
            segs[-1]["t1"] = r["t"]
    return segs


def _arr(rows, lo, hi, key, idx=None):
    out = []
    for r in rows[lo:hi + 1]:
        v = r[key]
        out.append(v[idx] if idx is not None else v)
    return np.asarray(out, dtype=np.float64)


def analyze_rate(rows, segs, settle_frac: float) -> dict:
    result = {"axes": {}}
    # gain is COMMAND (raw wire) -> TRUE angle rate (finite-diff quaternion). The ODOMETRY rate is
    # logged too and its sign vs the true rate is reported (it is inverted on >=1 axis).
    print(f"\n{'axis':>6} {'cmd':>7} {'true_steady':>12} {'gain':>7} {'odo/true':>9} {'overshoot':>10} {'tau_s':>7}")
    for axis in (0, 1, 2):
        steps = [s for s in segs if s["kind"] == "step" and s["axis"] == axis]
        if not steps:
            continue
        cmd_pts, true_pts = [], []
        odo_ratios = []
        per_dir = []
        for s in steps:
            t = _arr(rows, s["i0"], s["i1"], "t")
            cmd = _arr(rows, s["i0"], s["i1"], "cmd", axis)
            true = _arr(rows, s["i0"], s["i1"], "true_rate", axis)
            odo = _arr(rows, s["i0"], s["i1"], "meas_rate", axis)
            cmd_const = float(np.median(cmd))
            t_cut = s["t1"] - settle_frac * (s["t1"] - s["t0"])     # steady = last settle_frac
            sel = t >= t_cut
            true_steady = float(np.mean(true[sel])) if sel.any() else float("nan")
            odo_steady = float(np.mean(odo[sel])) if sel.any() else float("nan")
            cmd_pts.extend(cmd[sel].tolist())
            true_pts.extend(true[sel].tolist())
            if abs(true_steady) > 0.2:
                odo_ratios.append(odo_steady / true_steady)
            t0a = s["t0"]                                            # transient window from baseline
            base_lo = next((j for j in range(s["i0"], -1, -1) if rows[j]["t"] < t0a - 0.3), s["i0"])
            tw = _arr(rows, base_lo, s["i1"], "t")
            yw = _arr(rows, base_lo, s["i1"], "true_rate", axis)
            m = step_response_metrics(tw, yw, t_step=t0a, settle_frac=settle_frac)
            gain_dir = (true_steady / cmd_const) if abs(cmd_const) > 1e-6 else float("nan")
            odo_ratio = (odo_steady / true_steady) if abs(true_steady) > 0.2 else float("nan")
            per_dir.append({"cmd": cmd_const, "true_steady": true_steady, "gain": gain_dir,
                            "odo_over_true": odo_ratio,
                            "overshoot": m["overshoot"], "tau_s": m["tau_s"], "delay_s": m["delay_s"]})
            print(f"{_AXIS_NAME[axis]:>6} {cmd_const:+7.2f} {true_steady:+12.3f} {gain_dir:+7.2f} "
                  f"{odo_ratio:+9.2f} {m['overshoot']:10.0%} {m['tau_s']:7.3f}")
        fit = fit_rate_gain(cmd_pts, true_pts)
        sign = int(np.sign(fit["gain"])) if fit["gain"] == fit["gain"] else 0
        ok = "OK" if sign == _EXPECTED_SIGN[axis] else "!! UNEXPECTED"
        overs = [d["overshoot"] for d in per_dir if d["overshoot"] == d["overshoot"]]
        odo_sign = int(np.sign(np.mean(odo_ratios))) if odo_ratios else 0
        result["axes"][_AXIS_NAME[axis]] = {
            "gain": fit["gain"], "offset": fit["offset"], "r2": fit["r2"],
            "sign": sign, "expected_sign": _EXPECTED_SIGN[axis],
            "abs_gain": abs(fit["gain"]) if fit["gain"] == fit["gain"] else float("nan"),
            "odo_rate_sign": odo_sign,        # ODOMETRY rate sign vs the true angle rate
            "mean_overshoot": float(np.mean(overs)) if overs else float("nan"),
            "directions": per_dir,
        }
        print(f"  -> {_AXIS_NAME[axis]:>5}: gain={fit['gain']:+.2f} (|gain|={abs(fit['gain']):.2f}, "
              f"sign={sign:+d} expect {_EXPECTED_SIGN[axis]:+d} {ok})  r2={fit['r2']:.3f}  "
              f"overshoot~{np.mean(overs) if overs else float('nan'):.0%}  odo_rate_sign={odo_sign:+d}")
    # the headline numbers the controller needs
    g = {a: result["axes"][a]["abs_gain"] for a in result["axes"]}
    s = {a: result["axes"][a]["sign"] for a in result["axes"]}
    od = {a: result["axes"][a]["odo_rate_sign"] for a in result["axes"]}
    result["summary"] = {
        "body_rate_sign": [s.get("roll", -1), s.get("pitch", 1), s.get("yaw", -1)],
        "odo_rate_sign": [od.get("roll", 1), od.get("pitch", -1), od.get("yaw", 1)],
        "abs_gain": g,
        "verdict": ("STEADY GAIN" if any(v > 1.4 for v in g.values() if v == v) and
                    all((result["axes"][a]["mean_overshoot"] < 0.6)
                        for a in result["axes"] if result["axes"][a]["mean_overshoot"] == result["axes"][a]["mean_overshoot"])
                    else "see per-axis overshoot"),
    }
    return result


def analyze_hover(rows, segs, settle_frac: float) -> dict:
    thr_segs = [s for s in segs if s["phase"].startswith("thr_")]
    thrusts, up_accels = [], []
    print(f"\n{'thrust':>7} {'vz0':>7} {'vz1':>7} {'dvz/dt(NED)':>12} {'up_accel':>9}")
    for s in thr_segs:
        thr = float(rows[s["i0"]]["thrust"])
        t = _arr(rows, s["i0"], s["i1"], "t")
        vz = _arr(rows, s["i0"], s["i1"], "vel", 2)        # NED down+; climb -> vz<0
        t_cut = s["t1"] - max(settle_frac, 0.5) * (s["t1"] - s["t0"])
        sel = t >= t_cut
        if sel.sum() < 3:
            continue
        slope = float(np.polyfit(t[sel], vz[sel], 1)[0])    # d vz/dt (NED)
        up_accel = -slope                                   # world-up acceleration
        thrusts.append(thr)
        up_accels.append(up_accel)
        print(f"{thr:7.3f} {vz[0]:+7.2f} {vz[-1]:+7.2f} {slope:+12.3f} {up_accel:+9.3f}")
    hover, slope = fit_hover_thrust(thrusts, up_accels)
    print(f"\n  -> level hover_thrust = {hover:.3f}   slope = {slope:.1f} (m/s^2)/thrust")
    return {"thrusts": thrusts, "up_accels": up_accels,
            "hover_thrust": hover, "slope_mps2_per_thrust": slope}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--settle-frac", type=float, default=0.4, help="fraction of each phase tail = steady")
    args = ap.parse_args()
    session = Path(args.session)
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8")) if (session / "meta.json").exists() else {}
    rows = _load_rows(session)
    if not rows:
        print("no rows in commands.jsonl", file=sys.stderr)
        return 1
    segs = _segments(rows)
    mode = meta.get("mode", "rate")
    print(f"== analyze_sysid {session.name} ==  mode={mode}  rows={len(rows)}  phases={len(segs)}  "
          f"aborted={meta.get('aborted')}")

    if mode == "hover":
        result = {"mode": "hover", **analyze_hover(rows, segs, args.settle_frac)}
    else:
        result = {"mode": "rate", **analyze_rate(rows, segs, args.settle_frac)}
        s = result["summary"]
        print(f"\n  body_rate_sign (cmd->true) = {s['body_rate_sign']}")
        print(f"  odo_rate_sign (odo vs true)= {s['odo_rate_sign']}")
        print(f"  |gain| per axis            = { {k: round(v,2) for k,v in s['abs_gain'].items()} }")
        print(f"  verdict                    = {s['verdict']}")

    (session / "sysid_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {session / 'sysid_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
