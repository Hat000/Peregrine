"""Independent forward-feasibility check of a trajopt solution.

Re-integrates the optimal control sequence (a_c, omega) through a fresh numpy RK4 of the
SAME plant dynamics, starting from rest, and reports the position drift vs the optimizer's
state trajectory + the gate misses of the re-integrated path. A small drift confirms the
multiple-shooting solution is dynamically consistent (no collocation slack hiding an
infeasible jump), i.e. the lap time is achievable by an open-loop command sequence.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/planning/validate_forward.py \
        --csv scripts/planning/out_drag/trajectory.csv --drag
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plant_params as PP  # noqa: E402


def quat_mult(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def rot(q, v):
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    r = quat_mult(quat_mult(q, np.array([0, *v])), qc)
    return r[1:4]


def rot_inv(q, v):
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    r = quat_mult(quat_mult(qc, np.array([0, *v])), q)
    return r[1:4]


def deriv(x, u, use_drag):
    p, v, q = x[0:3], x[3:6], x[6:10]
    a_c, w = u[0], u[1:4]
    vdot = rot(q, np.array([0, 0, -a_c])) + np.array([0, 0, PP.G])
    if use_drag:
        vb = rot_inv(q, v)
        sb = np.sqrt(vb @ vb + 1e-9)
        vdot = vdot + rot(q, -PP.QUAD_DRAG_C2 * sb * vb)
    qdot = 0.5 * quat_mult(q, np.array([0, *w]))
    return np.concatenate([v, vdot, qdot])


def rk4(x, u, dt, use_drag):
    k1 = deriv(x, u, use_drag)
    k2 = deriv(x + 0.5 * dt * k1, u, use_drag)
    k3 = deriv(x + 0.5 * dt * k2, u, use_drag)
    k4 = deriv(x + dt * k3, u, use_drag)
    x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    x[6:10] /= np.linalg.norm(x[6:10])
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--course", default="rl/peregrine_course_diffaero.json")
    ap.add_argument("--drag", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        rdr = csv.reader(f)
        next(rdr)
        for r in rdr:
            rows.append([float(x) for x in r])
    arr = np.array(rows)
    t, P, V, Q, U = arr[:, 0], arr[:, 1:4], arr[:, 4:7], arr[:, 7:11], arr[:, 11:15]
    N = len(t)

    # re-integrate from rest using the optimizer's controls + its per-node dt (diff of t).
    x = np.concatenate([P[0] - V[0] * 0, [0, 0, 0], [1, 0, 0, 0]])
    # start state: rest at the pad (NED z down small); reproduce the optimizer's x_init.
    x = np.array([0.0, 0.0, 0.02, 0, 0, 0, 1, 0, 0, 0])
    traj = np.zeros((N, 10))
    t_prev = 0.0
    for k in range(N):
        dt = t[k] - t_prev
        x = rk4(x, U[k], dt, args.drag)
        traj[k] = x
        t_prev = t[k]

    pos_drift = np.linalg.norm(traj[:, 0:3] - P, axis=1)
    vel_drift = np.linalg.norm(traj[:, 3:6] - V, axis=1)

    # gate misses of the RE-INTEGRATED path
    c = json.loads(Path(args.course).read_text())
    gates_ned = np.array([[g["pos_zup"][0], -g["pos_zup"][1], -g["pos_zup"][2]]
                          for g in c["gates"]])
    misses = []
    for g in gates_ned:
        d = np.linalg.norm(traj[:, 0:3] - g, axis=1)
        misses.append(float(d.min()))

    out = {
        "n_nodes": N,
        "pos_drift_max_m": float(pos_drift.max()),
        "pos_drift_mean_m": float(pos_drift.mean()),
        "vel_drift_max_mps": float(vel_drift.max()),
        "reintegrated_gate_misses_m": [round(m, 4) for m in misses],
        "reintegrated_max_gate_miss_m": float(np.max(misses)),
        "all_gates_passed_forward": bool(np.max(misses) <= PP.GATE_BALL_RADIUS + 0.05),
        "use_drag": args.drag,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
