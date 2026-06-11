"""Peregrine driver for the TOGT multiple-shooting time-optimal refinement.

Adapted from TOGT-Planner/scripts/togt_refine/togt_refine.py (MIT, (c) 2024 FSC Lab):
same flow -- load the TOGT trajectory (csv) + its gate-junction waypoints (wpt yaml from
``saveSegments(wpt, piecesPerSegment)``), warm-start a multiple-shooting NLP (IPOPT) whose
only objective is total time, with the quadrotor 13-state model from ``quadrotor.py``
(per-rotor thrust + body-rate state bounds + optional PEREGRINE linear drag) -- but takes
all paths/parameters as CLI args and writes a machine-readable summary line.

The refined trajectory is THE BOUND for our plant class: collective+rate-constrained
point-mass-with-attitude, no inner-loop lag/slew (checked downstream by the twin replay,
scripts/twin_track_reference.py).

Usage (from this directory, casadi venv):
  python refine_peregrine.py --quad-yaml CASE/peregrine_quad.yaml \
      --traj-csv CASE/togt_traj.csv --wpt-yaml CASE/togt_wpt.yaml \
      --out-csv CASE/refined_traj.csv [--tol 0.01] [--tol-term 0.05] [--dt 0.02]
"""
import argparse
import csv
import json
import sys

import numpy as np

from optimization import Optimization
from quadrotor import Quadrotor
from trajectory import Trajectory


def save_traj(res, opt: Optimization, csv_f: str) -> float:
    """Write the refined trajectory CSV (upstream format: t, p, v, q, w, u columns;
    the a_lin/a_rot/jerk/snap columns are zero placeholders exactly as upstream).
    Returns the lap time (sum of optimized node durations)."""
    labels = ['t',
              "p_x", "p_y", "p_z",
              "v_x", "v_y", "v_z",
              "q_w", "q_x", "q_y", "q_z",
              "w_x", "w_y", "w_z",
              "a_lin_x", "a_lin_y", "a_lin_z",
              "a_rot_x", "a_rot_y", "a_rot_z",
              "u_1", "u_2", "u_3", "u_4",
              "jerk_x", "jerk_y", "jerk_z",
              "snap_x", "snap_y", "snap_z"]
    zeros12 = [0.0] * 12
    x = res['x'].full().flatten()
    with open(csv_f, 'w', newline='') as f:
        w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(labels)
        t = 0.0
        s = opt._xinit
        u = x[opt._Horizon * opt._X_dim: opt._Horizon * opt._X_dim + opt._U_dim]
        u_last = u
        w.writerow([t, *s[0:10], *s[10:13], *zeros12[:6], *u, *zeros12[6:]])
        for i in range(opt._seg_num):
            dt = x[-(opt._seg_num) + i]
            for j in range(opt._Ns[i]):
                idx = opt._N_wp_base[i] + j
                t += dt
                s = x[idx * opt._X_dim: (idx + 1) * opt._X_dim]
                if idx != opt._Horizon - 1:
                    u = x[opt._Horizon * opt._X_dim + (idx + 1) * opt._U_dim:
                          opt._Horizon * opt._X_dim + (idx + 2) * opt._U_dim]
                    u_last = u
                else:
                    u = u_last
                w.writerow([t, *s[0:13], *zeros12[:6], *u, *zeros12[6:]])
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quad-yaml", required=True)
    ap.add_argument("--traj-csv", required=True, help="TOGT trajectory csv (warm start)")
    ap.add_argument("--wpt-yaml", required=True, help="gate waypoints+durations yaml (saveSegments)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--tol", type=float, default=0.01, help="gate-waypoint ball radius (m)")
    ap.add_argument("--tol-term", type=float, default=0.05, help="terminal full-state ball radius")
    ap.add_argument("--dt", type=float, default=0.02, help="multiple-shooting node spacing (s)")
    ap.add_argument("--summary-json", default=None, help="optional summary output path")
    args = ap.parse_args()

    quad = Quadrotor(args.quad_yaml)
    togt = Trajectory(args.dt, args.traj_csv, args.wpt_yaml)
    togt.print()

    opt = Optimization(quad, togt._wpt_num, togt._Ns, args.tol, args.tol_term)
    opt.set_initial_guess(togt._xut0)
    opt._xinit = togt._xinit
    opt._xend = togt._xend

    print("\nTime optimization start ......\n")
    opt.define_opt_t()
    res = opt.solve_opt_t(togt._xinit, togt._xend, np.array(togt._waypoints).flatten())
    stats = opt._opt_t_solver.stats()
    lap = save_traj(res, opt, args.out_csv)

    summary = {
        "lap_time_total_s": lap,
        "ipopt_status": stats.get("return_status", "?"),
        "ipopt_success": bool(stats.get("success", False)),
        "iter_count": int(stats.get("iter_count", -1)),
        "tol": args.tol, "tol_term": args.tol_term, "dt": args.dt,
        "quad_yaml": args.quad_yaml,
    }
    print("REFINE_SUMMARY: " + json.dumps(summary))
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=1)
    # Non-converged solves still write a trajectory; the caller decides via the summary.
    return 0 if summary["ipopt_success"] else 2


if __name__ == "__main__":
    sys.exit(main())
