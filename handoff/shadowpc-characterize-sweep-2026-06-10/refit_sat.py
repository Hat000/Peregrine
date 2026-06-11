"""Re-fit the saturated (3.14) LTI 2nd-order with the MEASURED static gain (3.50/3.53)
instead of the shipped 2.5 -- what transient (wn, zeta) remains once the DC is right.
Result: roll wn=50.3 zeta=0.84 (1% overshoot), pitch wn=82.7 zeta=1.04 (0%) -- i.e. the
"saturated windup overshoot" was the static gain all along."""
import sys
from pathlib import Path

import numpy as np

_R = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_R / "src"), str(_R / "handoff" / "shadowpc-2ndorder-resysid-2026-06-10"),
                str(_R / "handoff" / "shadowpc-characterize-sweep-2026-06-10")]
import fit_2nd_order as f2
import fit_sweep as fs

f2._G_SHIPPED = np.array([3.50, 3.53, 2.231])   # measured sustained saturated gains
runs_dir = _R / "data" / "runs"
x0s = [(70.0, 0.6), (50.0, 0.8), (30.0, 0.5), (90.0, 1.0)]
names = {0: ["20260610_225846_sweep_r314r", "20260610_230027_sweep_r314r2"],
         1: ["20260610_225830_sweep_r314p", "20260610_230008_sweep_r314p2"]}
for axis in (0, 1):
    wins = []
    for name in names[axis]:
        t_w, w, rows = fs.load_run(runs_dir / name)
        wins += fs.step_windows(t_w, w, rows, axis, post_s=0.8)
    best = None
    for d in (0.0, 0.005, 0.010, 0.015, 0.020):
        p, fv = f2.fit_axis_fixedG([wins], axis, d, x0s)
        if best is None or fv < best[1]:
            best = (p, fv, d)
    (wn, z), err, d = best
    ovs = 100 * np.exp(-np.pi * z / np.sqrt(1 - z * z)) if z < 1 else 0.0
    print(f"{f2._AXES[axis]} @3.14 with G={f2._G_SHIPPED[axis]}: wn={wn:.1f} zeta={z:.3f} "
          f"d={d*1e3:.0f}ms overshoot={ovs:.0f}% rmse={err:.3f}")
