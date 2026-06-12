"""Translational audit on the rollfix flights: measured world accel (vel_ned FD)
vs model prediction  a = K(c_applied) * b3(live attitude) - g*z - drag(v).

Per-tick residuals split along the thrust axis vs perpendicular; plus a scale fit
live_K ~ alpha * model_K over the flight (thrust-axis component).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.rl_plant import (COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED,
                            QUAD_DRAG_C2_MEASURED)

G = 9.80665
D = 2  # applied-command transport delay (ticks), best-fit


def load_run(name):
    rows = []
    with open(ROOT / "data" / "runs" / name / "debug_obs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def K_of(c):
    return np.interp(c, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)


def analyze(name, k0=2, k1=240):
    rows = load_run(name)
    t = np.array([r["t_mono"] for r in rows])
    v = np.array([r["vel_ned"] for r in rows])
    n = min(k1, len(rows) - 1)
    meas_T, pred_T, perp_err = [], [], []
    print(f"\n==== {name} ====")
    print(f"{'k':>3} {'coll':>5} {'|v|':>5} {'a_meas.b3':>9} {'a_pred.b3':>9} {'perp_err':>8}")
    for k in range(max(k0, 1), n):
        dt = t[k + 1] - t[k - 1]
        if dt <= 0:
            continue
        a_meas = (v[k + 1] - v[k - 1]) / dt              # world NED accel
        q = np.array(rows[k]["q_raw_wxyz"], dtype=np.float64)
        Rm = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        b3_up = -Rm[:, 2]                                 # thrust direction (NED, -body z)
        c = rows[max(k - D, 0)]["collective"]
        vv = v[k]
        speed = float(np.linalg.norm(vv))
        # drag: quad body drag c2 per body axis: a_drag_body = -c2 * |v_b| * v_b
        v_body = Rm.T @ vv
        c2 = np.asarray(QUAD_DRAG_C2_MEASURED, dtype=np.float64)
        if c2.ndim == 2:                     # per-axis [pos, neg] asymmetric coefficients
            c2 = np.where(v_body >= 0, c2[:, 0], c2[:, 1])
        a_drag_body = -c2 * np.abs(v_body) * v_body
        a_drag = Rm @ a_drag_body
        a_pred = K_of(c) * b3_up + np.array([0.0, 0.0, G]) + a_drag
        am_T = float(a_meas @ b3_up)
        ap_T = float(a_pred @ b3_up)
        meas_T.append(am_T - float((np.array([0.0, 0.0, G]) + a_drag) @ b3_up))
        pred_T.append(K_of(c))
        perp = (a_meas - a_pred) - ((a_meas - a_pred) @ b3_up) * b3_up
        perp_err.append(float(np.linalg.norm(perp)))
        if k % 12 == 0:
            print(f"{k:>3} {c:>5.2f} {speed:>5.1f} {am_T:>9.2f} {ap_T:>9.2f} "
                  f"{perp_err[-1]:>8.2f}")
    meas_T = np.array(meas_T); pred_T = np.array(pred_T)
    alpha = float(np.dot(meas_T, pred_T) / np.dot(pred_T, pred_T))
    rms = float(np.sqrt(np.mean((meas_T - pred_T) ** 2)))
    print(f"thrust-axis: measured_K ~ {alpha:+.3f} x model_K   rms={rms:.2f} m/s^2  "
          f"(n={len(meas_T)})")
    print(f"perp residual: med={np.median(perp_err):.2f} p90={np.percentile(perp_err,90):.2f} m/s^2")


analyze("20260612_044219_inc6_rollfix_f1")
analyze("20260612_044438_inc6_rollfix_f2")
