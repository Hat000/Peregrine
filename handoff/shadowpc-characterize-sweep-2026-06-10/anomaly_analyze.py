"""Anomaly-boundary analysis (payload 2, SHADOWPC-CHARACTERIZE-SWEEP 2026-06-10).

Input: rate_sysid --mode anomaly runs (one sustained single-axis step, zero command on
the other axes, tilt abort off). For each run:

  * telemetry from the tlog: ODOMETRY quat + raw body rates * [-1,-1,1] (true FRD),
  * the step span + wire command from commands.jsonl,
  * tilt(t) = arccos(R22) from the quat (rotation of body z off world z; immune to the
    roll-reporting sign artifact),
  * EXPECTED probed-axis rate from the PI-windup fit (fit_windup.py, sweep-only params),
  * divergence triggers: (a) off-axis |rate| > OFF_THR rad/s while its command is zero
    (the yaw-spin signature: a rate appearing on an axis nobody commanded), and
    (b) probed-axis |measured - windup-model| > ON_THR rad/s, 3 consecutive samples.

Prints a per-run trace (75 Hz, the step window) + the first-trigger state per run, then
a boundary summary table.

Usage (repo root):
  .venv\\Scripts\\python.exe handoff\\shadowpc-characterize-sweep-2026-06-10\\anomaly_analyze.py [run ...]
  (no args: all data/runs/*_anom_* runs)
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
from fit_windup import sim_windup
from racer.frames import euler_from_quat_wxyz
from racer.recording import RecordingReader

_AXES = f2._AXES
OFF_THR = 1.5      # rad/s on an uncommanded axis
ON_THR = 2.0       # rad/s deviation from the windup model on the probed axis
# PI-windup params (fit_windup.py sweep-only fit): Kp, Ki, I_max, alpha_max, delay
WINDUP = {0: (124.7, 2526.0, 0.1063, 259.0, 0.010),
          1: (96.0, 2196.0, 0.1044, 284.0, 0.010),
          2: (14.2, 60673.0, 0.0086, 78.0, 0.020)}


def load_telemetry(run: Path):
    """-> t (N,), w_true (N,3), q (N,4 wxyz), pos (N,3)."""
    ts, ws, qs, ps = [], [], [], []
    for m in RecordingReader(run).iter_mavlink():
        if m.get_type() == "ODOMETRY":
            ts.append(m.time_usec * 1e-6)
            ws.append([m.rollspeed, m.pitchspeed, m.yawspeed])
            qs.append([m.q[0], m.q[1], m.q[2], m.q[3]])
            ps.append([m.x, m.y, m.z])
    return (np.asarray(ts), np.asarray(ws) * f2._ODO_RAW_TO_TRUE,
            np.asarray(qs), np.asarray(ps))


def analyze(run: Path, verbose=True):
    rows = [json.loads(l) for l in (run / "commands.jsonl").read_text().splitlines()]
    step = [r for r in rows if r["kind"] == "astep"]
    if not step:
        print(f"{run.name}: no astep rows")
        return None
    cmd_vec = np.asarray(step[0]["cmd"], dtype=float)
    probed = [a for a in range(3) if abs(cmd_vec[a]) > 1e-6]
    axis = probed[int(np.argmax([abs(cmd_vec[a]) for a in probed]))]   # primary (for the trace)
    mag = cmd_vec[axis]
    t0 = step[0]["sim_time_ns"] * 1e-9
    t1 = step[-1]["sim_time_ns"] * 1e-9
    t_u = np.array([r["sim_time_ns"] * 1e-9 for r in rows])
    u = np.array([r["cmd"] for r in rows])

    t, w, q, pos = load_telemetry(run)
    m = (t >= t0 - 0.10) & (t <= t1 + 0.05)
    t, w, q, pos = t[m], w[m], q[m], pos[m]
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    rpy = np.degrees(np.array([euler_from_quat_wxyz(qq) for qq in q]))

    preds = {}
    for a in probed:
        Kp, Ki, Imax, amax, d = WINDUP[a]
        u_t = f2._zoh(t, t_u, u[:, a], d)
        preds[a] = sim_windup(t, u_t, a, Kp, Ki, Imax, amax, w[0, a])
    pred = preds[axis]

    off_axes = [a for a in range(3) if a not in probed]                # uncommanded only
    dev_on = np.max(np.stack([np.abs(w[:, a] - preds[a]) for a in probed]), axis=0)
    trig = None
    for k in range(len(t)):
        if t[k] < t0:
            continue
        off_hit = [a for a in off_axes if abs(w[k, a]) > OFF_THR]
        on_hit = (k >= 2 and np.all(dev_on[max(0, k - 2):k + 1] > ON_THR))
        if off_hit or on_hit:
            why = ("offaxis:" + "+".join(_AXES[a] for a in off_hit)) if off_hit else "onaxis-dev"
            trig = {"t": t[k] - t0, "tilt": tilt[k], "why": why,
                    "rates": w[k].round(2).tolist(), "rpy": rpy[k].round(0).tolist(),
                    "alt": -pos[k, 2]}
            break

    if verbose:
        print(f"\n--- {run.name}: {_AXES[axis]} {mag:+.2f} sustained "
              f"(target {f2._G_SHIPPED[axis]*f2._RATE_SIGN[axis]*mag:+.2f} rad/s), "
              f"step {t1-t0:.2f}s ---")
        print(f"{'t-t0':>7} {'tilt':>6} {'roll_w':>7} {'pitch_w':>8} {'yaw_w':>7} "
              f"{'pred':>7} {'dev':>6}  rpy")
        for k in range(0, len(t), 2):
            mark = " <-- trigger" if (trig and abs(t[k] - t0 - trig["t"]) < 0.014) else ""
            print(f"{t[k]-t0:>7.3f} {tilt[k]:>6.1f} {w[k,0]:>7.2f} {w[k,1]:>8.2f} "
                  f"{w[k,2]:>7.2f} {pred[k]:>7.2f} {dev_on[k]:>6.2f}  "
                  f"({rpy[k,0]:+4.0f},{rpy[k,1]:+4.0f},{rpy[k,2]:+5.0f}){mark}")
    return {"run": run.name, "axis": _AXES[axis], "mag": mag, "trig": trig,
            "max_tilt": float(np.max(tilt)),
            "peak_rates": np.max(np.abs(w), axis=0).round(2).tolist()}


def main() -> int:
    if len(sys.argv) > 1:
        runs = [Path(a) for a in sys.argv[1:]]
    else:
        runs = sorted((_REPO / "data" / "runs").glob("*_anom_*"))
    results = [analyze(r) for r in runs]
    print("\n=== anomaly boundary summary ===")
    print(f"{'run':<32} {'axis':<6} {'mag':>6} {'max_tilt':>9} {'trigger':>12} "
          f"{'t(s)':>6} {'tilt@trig':>10} {'rates@trig':>20}")
    for r in results:
        if r is None:
            continue
        tg = r["trig"]
        if tg:
            print(f"{r['run']:<32} {r['axis']:<6} {r['mag']:>6.2f} {r['max_tilt']:>9.0f} "
                  f"{tg['why']:>12} {tg['t']:>6.2f} {tg['tilt']:>10.0f} {str(tg['rates']):>20}")
        else:
            print(f"{r['run']:<32} {r['axis']:<6} {r['mag']:>6.2f} {r['max_tilt']:>9.0f} "
                  f"{'NONE':>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
