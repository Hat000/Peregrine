"""Sanity checks for fit_sweep.py results.

1. Numeric trace dump: measured vs modeled rate through a +step window (roll 0.3 and
   roll 3.14), under (a) the per-mag sweep fit, (b) the cc6921d saturated params,
   (c) the cc6921d small/mid params, (d) the shipped 1st-order.
2. Cross-eval: RMSE on the S1.2 TUMBLE windows under the sweep mag-3.14 params vs the
   cc6921d saturated params (does the sweep fit also explain the n=1 maneuver?).
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
import fit_sweep as fs

runs_dir = _REPO / "data" / "runs"

# fitted sweep params (from fit_sweep.py output, this session)
SWEEP = {  # (axis, mag) -> (wn, zeta, delay)
    (0, 0.30): (75.2, 0.526, 0.000), (0, 3.14): (66.2, 0.485, 0.015),
    (1, 3.14): (61.8, 0.493, 0.010),
}
CC_SAT = {0: (21.0, 0.393, 0.015), 1: (28.3, 0.467, 0.030)}
CC_SMALL = {0: (67.6, 0.824, 0.000), 1: (66.9, 0.699, 0.000)}


def trace(run_name, axis, mag, n_step=0):
    t_w, w, rows = fs.load_run(runs_dir / run_name)
    spans = fs.step_spans(rows, axis)
    t0, t1, v = spans[n_step]
    t_u = np.array([r["sim_time_ns"] * 1e-9 for r in rows])
    u = np.array([r["cmd"] for r in rows])
    m = (t_w >= t0 - 0.10) & (t_w <= t1 + 0.45)
    tt, ww = t_w[m], w[m]
    G = f2._G_SHIPPED[axis]

    def model2(wn, z, d):
        u_t = f2._zoh(tt, t_u, u[:, axis], d)
        w0 = ww[0, axis]
        wd0 = (ww[1, axis] - ww[0, axis]) / max(tt[1] - tt[0], 1e-3)
        return f2.sim_second_order(tt, u_t, axis, G, wn, z, w0, wd0)

    u_t0 = f2._zoh(tt, t_u, u[:, axis], 0.0)
    first = f2.sim_first_order(tt, u_t0, axis, G, f2._TAU_SHIPPED, ww[0, axis])
    msw = model2(*SWEEP[(axis, mag)])
    msat = model2(*CC_SAT[axis])
    msml = model2(*CC_SMALL[axis])
    print(f"\n--- {run_name} {f2._AXES[axis]} step {n_step}: cmd {v:+.2f}, target {G*v:+.2f}, "
          f"dur {t1-t0:.2f}s ---")
    print(f"{'t-t0':>7} {'cmd':>6} {'meas':>7} {'sweep':>7} {'ccSAT':>7} {'ccSML':>7} {'1st':>7}")
    for k in range(len(tt)):
        print(f"{tt[k]-t0:>7.3f} {u_t0[k]:>6.2f} {ww[k, axis]:>7.2f} {msw[k]:>7.2f} "
              f"{msat[k]:>7.2f} {msml[k]:>7.2f} {first[k]:>7.2f}")


trace("20260610_225207_sweep_r03b", 0, 0.30)          # roll +0.3
trace("20260610_225846_sweep_r314r", 0, 3.14)          # roll +3.14 (0.14 s, aborted run)

# ---- cross-eval on the S1.2 tumble ----
print("\n=== cross-eval on the S1.2 tumble windows (RMSE rad/s) ===")
ckpt = r"C:\Users\Shadow\Downloads\stage1_inc1_actor.pth"
f1 = runs_dir / "20260610_205414_rl_s12_f1"
t_w, w = f2.load_rates(f1)
t_u, u = f2.reconstruct_cmds_flight1(f1, ckpt, t1=10.99)
tumble = f2.build_windows(t_w, w, t_u, u, max_len_s=2.0)
for axis in (0, 1):
    wn_s, z_s, d_s = SWEEP[(axis, 3.14)]
    wn_c, z_c, d_c = CC_SAT[axis]
    e_sweep = f2.axis_rmse(tumble, axis, "second", (f2._G_SHIPPED[axis], wn_s, z_s), d_s)
    e_cc = f2.axis_rmse(tumble, axis, "second", (f2._G_SHIPPED[axis], wn_c, z_c), d_c)
    e_1st = f2.axis_rmse(tumble, axis, "first", (f2._G_SHIPPED[axis], f2._TAU_SHIPPED), 0.0)
    print(f"{f2._AXES[axis]:<6} sweep-3.14 {e_sweep:7.3f} | cc6921d-sat {e_cc:7.3f} | shipped-1st {e_1st:7.3f}")
