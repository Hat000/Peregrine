"""Static amplitude-gain curve from the sustained-rotation (anomaly-mode) runs +
S1.2 tumble post-90deg re-examination (was the excluded 'anomaly regime' actually
anomalous, or just super-rate gain + collision contact?).

Part 1: per run, sustained gain = mean(probed rate)/cmd over the settled window.
Part 2: simulate the tumble 10.77..11.40 s under the STATIC-GAIN model
        (target = g(|cmd|)*sign*cmd, 1st-order tau + slew limit) and print the trace
        alongside measured + COLLISION timestamps from the tlog.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "rl"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "handoff" / "shadowpc-2ndorder-resysid-2026-06-10"))

import fit_2nd_order as f2
from anomaly_analyze import load_telemetry
from racer.recording import RecordingReader
import json

runs_dir = _REPO / "data" / "runs"

print("=== sustained gains (settled window of each sustained rotation) ===")
print(f"{'run':<34} {'axis':<6} {'cmd':>6} {'win(s)':>11} {'mean rate':>10} {'gain':>6}")
SUST = [
    ("20260611_021217_anom_r05", 0), ("20260611_020417_anom_r10", 0),
    ("20260611_021156_anom_r20", 0), ("20260611_020731_anom_r314", 0),
    ("20260611_021237_anom_p10", 1), ("20260611_020751_anom_p314", 1),
    ("20260611_021010_anom_rpy314", 0), ("20260611_021010_anom_rpy314", 1),
    ("20260611_021010_anom_rpy314", 2), ("20260611_021029_anom_ry314", 0),
    ("20260611_021029_anom_ry314", 2),
]
gains = {}
for name, axis in SUST:
    run = runs_dir / name
    rows = [json.loads(l) for l in (run / "commands.jsonl").read_text().splitlines()]
    step = [r for r in rows if r["kind"] == "astep"]
    cmd = step[0]["cmd"][axis]
    if abs(cmd) < 1e-6:
        continue
    t0, t1 = step[0]["sim_time_ns"] * 1e-9, step[-1]["sim_time_ns"] * 1e-9
    t, w, q, pos = load_telemetry(run)
    lo = t0 + max(0.5, 0.4 * (t1 - t0))
    m = (t >= lo) & (t <= t1)
    if m.sum() < 5:
        continue
    mean_rate = float(np.mean(w[m, axis]))
    g = mean_rate / cmd * f2._RATE_SIGN[axis]
    gains.setdefault((axis, abs(cmd)), []).append(g)
    print(f"{name:<34} {f2._AXES[axis]:<6} {cmd:>+6.2f} {lo-t0:>5.2f}..{t1-t0:<4.2f} "
          f"{mean_rate:>10.2f} {g:>6.3f}")

print("\n=== static gain curve g(|cmd|) (mean per axis x |cmd|; shipped small-signal in parens) ===")
for (axis, c), gs in sorted(gains.items()):
    print(f"  {f2._AXES[axis]:<6} |cmd|={c:.2f}: g={np.mean(gs):.3f}   (shipped {f2._G_SHIPPED[axis]:.3f})")
sr = 2.5 / (1 - 0.30 * np.array([0.5, 1.0, 2.0, 3.14]) / 3.14)
print(f"  super-rate g=2.5/(1-0.30*|c|/3.14) predicts: "
      f"{', '.join(f'{c}:{v:.2f}' for c, v in zip([0.5,1.0,2.0,3.14], sr))}")

# ---------------- part 2: tumble post-90deg ----------------
print("\n=== S1.2 tumble: COLLISION messages in the tlog (t = last ODOMETRY sim time) ===")
f1 = runs_dir / "20260610_205414_rl_s12_f1"
last_odo = float("nan")
for m in RecordingReader(f1).iter_mavlink():
    if m.get_type() == "ODOMETRY":
        last_odo = m.time_usec * 1e-6
    elif m.get_type() == "COLLISION":
        print(f"  sim_t~{last_odo:.3f}s id={m.id} threat={m.threat_level}")

print("\n=== tumble 10.77..11.40 s: measured vs STATIC-GAIN model (tau=20ms, slew 260) ===")
ckpt = r"C:\Users\Shadow\Downloads\stage1_inc1_actor.pth"
t_u, u = f2.reconstruct_cmds_flight1(f1, ckpt, t0=10.76, t1=11.40)
t_w, w = f2.load_rates(f1)
m = (t_w >= 10.77) & (t_w <= 11.40)
t, wme = t_w[m], w[m]
_, _, q, pos = load_telemetry(f1)
tq, _, q_all, _ = load_telemetry(f1)
qm = q_all[(tq >= 10.77) & (tq <= 11.40)]
tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qm[:, 1] ** 2 + qm[:, 2] ** 2), -1, 1)))

G_SAT = 3.56   # measured sustained saturated gain (roll/pitch); yaw measured 3.16 in the 3-axis tumble probe
TAU, SLEW, DELAY = 0.020, 260.0, 0.015


def sim_static(axis, g_sat):
    u_t = f2._zoh(t, t_u, u[:, axis], DELAY)
    wv = wme[0, axis]
    out = [wv]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        tgt = g_sat * f2._RATE_SIGN[axis] * u_t[k - 1]
        dw = (tgt - wv) * (1 - np.exp(-dt / TAU))
        dw = np.clip(dw, -SLEW * dt, SLEW * dt)
        wv += dw
        out.append(wv)
    return np.asarray(out)


preds = {0: sim_static(0, G_SAT), 1: sim_static(1, G_SAT), 2: sim_static(2, 3.16)}
print(f"{'t':>7} {'tilt':>6} {'roll m/p':>15} {'pitch m/p':>15} {'yaw m/p':>15}")
for k in range(0, len(t), 2):
    print(f"{t[k]:>7.3f} {tilt[k]:>6.0f} "
          f"{wme[k,0]:>7.2f}/{preds[0][k]:>6.2f} "
          f"{wme[k,1]:>7.2f}/{preds[1][k]:>6.2f} "
          f"{wme[k,2]:>7.2f}/{preds[2][k]:>6.2f}")
for axis in range(3):
    e = float(np.sqrt(np.mean((preds[axis] - wme[:, axis]) ** 2)))
    print(f"  {f2._AXES[axis]} static-gain model RMSE over 10.77-11.40: {e:.2f} rad/s")
