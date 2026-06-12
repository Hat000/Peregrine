"""Fit the live rate-loop transient from the rollfix flights.

Model (rl_plant form): omega' lags toward target = gain(super_rate) * S_live * wire
with first-order tau and transport delay d ticks. Scan (d, tau) per axis against the
recorded wire -> realized TRUE rates (w_raw*[1,-1,1]) over the first 150 ticks of
both rollfix flights; report RMSE grid + best.

Then: closed-loop FIXED counterfactual against the emulated live plant with the
best-fit (d, tau) -- does inc6 still finish from simstart?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import numpy as np

DT = 1.0 / 30.0
S_REP = np.array([1.0, -1.0, 1.0])
S_LIVE = np.array([-1.0, 1.0, -1.0])
RATE_GAIN = np.array([2.501, 2.504, 2.231])
SR = 0.30  # SUPER_RATE_S_MEASURED scalar


def load_run(name):
    rows = []
    with open(ROOT / "data" / "runs" / name / "debug_obs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def sim_rates(wire, d, tau, n):
    om = np.zeros(3)
    out = []
    for k in range(n):
        cmd = wire[max(k - d, 0)] if k >= d else np.zeros(3)
        gain = RATE_GAIN / (1.0 - SR * np.minimum(np.abs(cmd), np.pi) / np.pi)
        target = gain * S_LIVE * cmd
        a = 1.0 - np.exp(-DT / tau)
        om = om + a * (target - om)
        out.append(om.copy())
    return np.array(out)


runs = ["20260612_044219_inc6_rollfix_f1", "20260612_044438_inc6_rollfix_f2"]
N = 150
data = []
for nm in runs:
    rows = load_run(nm)[:N]
    wire = np.array([r["rate_frd"] for r in rows])
    w_true = np.array([np.array(r["w_raw"]) * S_REP for r in rows])
    data.append((wire, w_true))

print("RMSE grid (mean over both flights, all axes):")
print(f"{'d\\tau':>6} " + " ".join(f"{t:>7.3f}" for t in [0.019, 0.03, 0.05, 0.07, 0.10, 0.15]))
best = None
for d in range(0, 5):
    row = []
    for tau in [0.019, 0.03, 0.05, 0.07, 0.10, 0.15]:
        errs = []
        for wire, w_true in data:
            pred = sim_rates(wire, d, tau, len(wire))
            errs.append(np.sqrt(np.mean((pred - w_true) ** 2)))
        e = float(np.mean(errs))
        row.append(e)
        if best is None or e < best[0]:
            best = (e, d, tau)
    print(f"{d:>6} " + " ".join(f"{e:>7.3f}" for e in row))
print(f"best: rmse={best[0]:.3f} at d={best[1]} tau={best[2]}")

# per-axis at best
d, tau = best[1], best[2]
for ax, nm in enumerate(["roll", "pitch", "yaw"]):
    errs, gains = [], []
    for wire, w_true in data:
        pred = sim_rates(wire, d, tau, len(wire))
        errs.append(np.sqrt(np.mean((pred[:, ax] - w_true[:, ax]) ** 2)))
        den = float(np.dot(pred[:, ax], pred[:, ax]))
        gains.append(float(np.dot(w_true[:, ax], pred[:, ax]) / den) if den > 1e-9 else np.nan)
    print(f"  {nm:5s}: rmse={np.mean(errs):.3f}  live~{np.mean(gains):+.3f} x model")
