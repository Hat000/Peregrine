"""Windup-curve fit from the 2026-06-10 controlled magnitude sweep (payload 1 of the
SHADOWPC-CHARACTERIZE-SWEEP session).

Extends handoff/shadowpc-2ndorder-resysid-2026-06-10/fit_2nd_order.py (imports its
models/fitters; same conventions: wire cmd = FRD body-rate setpoint, output = ODOMETRY
raw rate * [-1,-1,1], target = G_shipped * rate_sign * cmd with G FIXED at the validated
steady gains). New here:

  * sweep loader: rate_sysid runs probed ONE axis per run at ONE magnitude; fit windows
    are cut around the step phases ([t0-0.15 s, step_end+0.8 s]) so the saturated
    transient is not diluted by minutes of hold chatter.
  * per-(axis, magnitude) 2nd-order fits -> wn(|target|), zeta(|target|) = the windup curve.
  * raw (model-free) per-step metrics: peak |rate| / |G*cmd| and, for the long yaw steps,
    the PLATEAU gain = the first direct measurement of the saturated DC gain.

Usage (repo root):
  .venv\\Scripts\\python.exe handoff\\shadowpc-characterize-sweep-2026-06-10\\fit_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "shadowpc-2ndorder-resysid-2026-06-10"))

import fit_2nd_order as f2

_AXES = f2._AXES
_NAME_AXIS = {"roll": 0, "pitch": 1, "yaw": 2}

# run-dir name -> (magnitude, [probed axes]); all under data/runs/.
RUNS = {
    "20260610_223428_sweep_r03":    (0.30, ["pitch", "roll"]),   # aborted in last re-level
    "20260610_225207_sweep_r03b":   (0.30, ["pitch", "roll", "yaw"]),
    "20260610_225305_sweep_r10":    (1.00, ["pitch", "roll", "yaw"]),
    "20260610_225755_sweep_r10y":   (1.00, ["yaw"]),             # 1.2 s long steps
    "20260610_225400_sweep_r20":    (2.00, ["pitch", "roll"]),   # aborted before yaw
    "20260610_225513_sweep_r20p":   (2.00, ["pitch"]),
    "20260610_225543_sweep_r20r":   (2.00, ["roll"]),            # aborted after +step
    "20260610_225702_sweep_r20r2":  (2.00, ["roll"]),
    "20260610_225735_sweep_r20y":   (2.00, ["yaw"]),             # 1.0 s long steps
    "20260610_225830_sweep_r314p":  (3.14, ["pitch"]),           # aborted 84 deg (full rise)
    "20260610_225846_sweep_r314r":  (3.14, ["roll"]),            # aborted 83 deg (full rise)
    "20260610_230008_sweep_r314p2": (3.14, ["pitch"]),
    "20260610_230027_sweep_r314r2": (3.14, ["roll"]),
    "20260610_225902_sweep_r314y":  (3.14, ["yaw"]),             # 1.2 s long steps
}

MAGS = [0.30, 1.00, 2.00, 3.14]


def load_run(run: Path):
    """-> (t_w, w_true (N,3), cmd rows). Telemetry from the tlog (raw 75 Hz ODOMETRY,
    NOT the EMA-smoothed true_rate column); commands from commands.jsonl."""
    t_w, w = f2.load_rates(run)
    rows = [json.loads(l) for l in (run / "commands.jsonl").read_text().splitlines()]
    return t_w, w, rows


def step_spans(rows, axis):
    """Contiguous step-phase spans on `axis` -> list of (t_start, t_end, cmd_value),
    in sim time (s)."""
    spans = []
    for r in rows:
        if r["kind"] != "step" or r["axis"] != axis:
            continue
        t = r["sim_time_ns"] * 1e-9
        v = r["cmd"][axis]
        if spans and abs(t - spans[-1][1]) < 0.1 and spans[-1][2] == v:
            spans[-1][1] = t
        else:
            spans.append([t, t, v])
    return [(s, e, v) for s, e, v in spans if e - s > 0.04]


def step_windows(t_w, w, rows, axis, pre_s=0.15, post_s=0.8):
    """Fit windows cut around each step on `axis`: [t0-pre, t_end+post]. The command
    series passed for ZOH is the FULL run (the post-step re-level is known input)."""
    t_u = np.array([r["sim_time_ns"] * 1e-9 for r in rows])
    u = np.array([r["cmd"] for r in rows])
    wins = []
    for t0, t1, _v in step_spans(rows, axis):
        m = (t_w >= t0 - pre_s) & (t_w <= t1 + post_s)
        if m.sum() >= 8:
            wins.append((t_w[m], w[m], (t_u, u)))
    return wins


def raw_step_metrics(t_w, w, rows, axis, mag):
    """Model-free per-step numbers, IN-STEP only (the post-step window is the re-level =
    a different known input; including it contaminated the first cut of this table):
    signed peak toward the target in [t0, t_end+0.05], time to cross 0.8*|target|,
    value at step end, plateau mean over the last 40% (long steps only)."""
    out = []
    for t0, t1, v in step_spans(rows, axis):
        sgn = np.sign(f2._G_SHIPPED[axis] * f2._RATE_SIGN[axis] * v)
        ws = sgn * w[:, axis]                      # + = toward the target
        tgt = abs(f2._G_SHIPPED[axis] * v)
        m_pk = (t_w >= t0) & (t_w <= t1 + 0.05)
        m_in = (t_w >= t0) & (t_w <= t1)
        m_pl = (t_w >= t1 - 0.4 * (t1 - t0)) & (t_w <= t1)
        if m_pk.sum() < 3 or m_in.sum() < 2:
            continue
        peak = float(np.max(ws[m_pk]))
        cross = np.nonzero(ws[m_in] >= 0.8 * tgt)[0]
        t80 = float(t_w[m_in][cross[0]] - t0) if len(cross) else np.nan
        end = float(ws[m_in][-1])
        plateau = float(np.mean(ws[m_pl])) if (m_pl.sum() >= 3 and t1 - t0 > 0.5) else np.nan
        out.append({"t0": t0, "dur": t1 - t0, "cmd": v, "tgt": tgt, "peak": peak,
                    "t80": t80, "end": end, "plateau": plateau})
    return out


def main() -> int:
    runs_dir = _REPO / "data" / "runs"
    loaded = {name: load_run(runs_dir / name) for name in RUNS}

    # ---------------- raw metrics table ----------------
    print("=== raw per-step metrics (model-free, IN-STEP; target = G_shipped*|cmd|) ===")
    print(f"{'mag':>5} {'axis':<6} {'n':>2} {'dur(s)':>7} {'tgt':>6} {'peak (each step)':>22} "
          f"{'peak/tgt':>9} {'t80(ms)':>8} {'end/tgt':>8} {'plateau/cmd':>12}")
    plateau_gains = {}
    for mag in MAGS:
        for axis in range(3):
            peaks, plats, durs, t80s, ends = [], [], [], [], []
            tgt = f2._G_SHIPPED[axis] * mag
            for name, (m, axes) in RUNS.items():
                if m != mag or _AXES[axis] not in axes:
                    continue
                t_w, w, rows = loaded[name]
                for s in raw_step_metrics(t_w, w, rows, axis, mag):
                    peaks.append(s["peak"])
                    durs.append(s["dur"])
                    ends.append(s["end"] / s["tgt"])
                    if np.isfinite(s["t80"]):
                        t80s.append(s["t80"])
                    if np.isfinite(s["plateau"]):
                        plats.append(abs(s["plateau"] / s["cmd"]))
            if not peaks:
                continue
            pk_str = ",".join(f"{p:.2f}" for p in sorted(peaks, reverse=True)[:4])
            pl = np.mean(plats) if plats else np.nan
            if plats:
                plateau_gains[(axis, mag)] = pl
            print(f"{mag:>5.2f} {_AXES[axis]:<6} {len(peaks):>2} {np.mean(durs):>7.2f} {tgt:>6.2f} "
                  f"{pk_str:>22} {max(peaks)/tgt:>9.2f} "
                  f"{np.mean(t80s)*1e3 if t80s else float('nan'):>8.0f} "
                  f"{np.mean(ends):>8.2f} {pl if plats else float('nan'):>12.3f}")

    # ---------------- per-(axis, mag) 2nd-order fits (G fixed) ----------------
    print("\n=== windup curve: per-(axis, mag) 2nd-order fit, G FIXED at shipped ===")
    print(f"{'mag':>5} {'axis':<6} {'tgt':>6} {'wn(rad/s)':>10} {'zeta':>7} {'d(ms)':>6} "
          f"{'overshoot':>9} {'RMSE':>7} {'n_win':>6}")
    x0s = [(30.0, 0.4), (25.0, 0.7), (50.0, 1.0), (15.0, 0.3), (70.0, 0.8)]
    curve = {}
    for mag in MAGS:
        for axis in range(3):
            wins = []
            for name, (m, axes) in RUNS.items():
                if m != mag or _AXES[axis] not in axes:
                    continue
                t_w, w, rows = loaded[name]
                post = 0.8 if mag >= 1.0 else 1.2
                wins += step_windows(t_w, w, rows, axis, post_s=post)
            if not wins:
                continue
            best = None
            for delay in (0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040):
                p, fv = f2.fit_axis_fixedG([wins], axis, delay, x0s)
                if best is None or fv < best[1]:
                    best = (p, fv, delay)
            (wn, z), err, d = best
            ovs = 100 * np.exp(-np.pi * z / np.sqrt(1 - z * z)) if z < 1 else 0.0
            curve[(axis, mag)] = (wn, z, d, ovs, err)
            print(f"{mag:>5.2f} {_AXES[axis]:<6} {f2._G_SHIPPED[axis]*mag:>6.2f} {wn:>10.1f} "
                  f"{z:>7.3f} {d*1e3:>6.0f} {ovs:>8.0f}% {err:>7.3f} {len(wins):>6}")

    # ---------------- comparison vs the cc6921d n=1 anchors ----------------
    print("\n=== cc6921d (n=1 tumble) anchors for reference ===")
    print("SATURATED  roll wn=21.0 z=0.393 (26%) | pitch wn=28.3 z=0.467 (19%) | yaw wn=11.3 z=0.273 (41%, caveated)")
    print("small/mid  roll wn=67.6 z=0.824 ( 1%) | pitch wn=66.9 z=0.699 ( 5%) | yaw wn=109 z=5.3 (overdamped)")

    print("\n=== saturated DC gain (yaw long-step plateaus; shipped G_yaw=2.231) ===")
    for (axis, mag), g in sorted(plateau_gains.items()):
        print(f"  {_AXES[axis]} mag {mag:.2f}: plateau gain {g:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
