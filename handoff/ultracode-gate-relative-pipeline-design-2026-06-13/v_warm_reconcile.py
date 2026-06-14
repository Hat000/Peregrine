"""v_warm_reconcile -- why does d3 'warm' p90=0.124 clear, but a perfect-velocity STRAIGHT leg
p90=0.222 fail? Both lat_sigma=0.265. Reconcile the discrepancy -- it decides whether the
'velocity channel closes the margin' (claim 5) is real or an artifact of d3's warm construction.

Hypotheses:
  H1  d3 warm's continuous update_velocity(truth) couples vel->pos via cross-covariance, tightening
      the POSITION estimate beyond what fixes alone give. Test: perfect-seed + continuous vel-update
      on the SAME straight leg. If p90 drops to ~0.12, the velocity CHANNEL (not just prior) is the
      lever and claim 5 stands. If it stays ~0.22, d3 warm clears for a DIFFERENT reason (fix count).
  H2  d3's multi-gate run lands MANY more usable fixes near g4 (it accepts fixes vs every upcoming
      gate, and the lap pre-tightens covariance). Count d3-window fixes vs straight-leg fixes.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/v_warm_reconcile.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))
from racer.frames import R_world_from_body  # noqa: E402
from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 5511
MARGIN = 0.155
LAT = 0.265
RADIAL = 0.50
ACCEL_NOISE = 0.3
IMU_DT = 1.0 / 90.0
FIX_DT = 1.0 / 14.0
DRAG = 0.21
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])


def leg(v_race, vel_update, fix_window_m=12.0, n_mc=800):
    """Perfect velocity seed. vel_update in {None,'continuous'}. Returns p90 + fix count."""
    seg = G4 - G3; L = float(np.linalg.norm(seg)); uhat = seg / L
    T = L / v_race; n_steps = int(T / IMU_DT)
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(DRAG * v_race / GRAVITY_NED[2]))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    ab = R_wb.T @ (-GRAVITY_NED)
    e1 = np.cross(uhat, np.array([0.0, 0.0, 1.0])); e1 /= np.linalg.norm(e1)
    e2 = np.cross(uhat, e1); e2 /= np.linalg.norm(e2)
    linf = []; nfix = []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 13 * s + int(v_race) * 7 + (1 if vel_update else 0) * 911)
        p0 = G3.copy(); v_init = uhat * v_race
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v_init, pos_std=0.5, vel_std=0.05)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t = 0.0; t_ns = 0; next_fix = 0.0; nf = 0
        for k in range(n_steps):
            t += IMU_DT; t_ns += int(IMU_DT * 1e9)
            rk.predict(ab + r.normal(0, ACCEL_NOISE, 3), R_wb, IMU_DT, t_ns)
            if vel_update == "continuous":
                rk.update_velocity(uhat * v_race, (0.1**2) * np.eye(3), sim_time_ns=t_ns)
            p_true = p0 + uhat * v_race * t
            if t >= next_fix and float(np.linalg.norm(G4 - p_true)) < fix_window_m:
                next_fix = t + FIX_DT; nf += 1
                nl = r.normal(0, LAT) * e1 + r.normal(0, LAT) * e2 + r.normal(0, RADIAL) * uhat
                z = p_true + nl
                cov = LAT**2 * (np.outer(e1, e1) + np.outer(e2, e2)) + RADIAL**2 * np.outer(uhat, uhat)
                rk.update_position(z, cov, sim_time_ns=t_ns)
        err = rk.position - (p0 + uhat * v_race * (n_steps * IMU_DT))
        eE = float(err @ e1); eD = float(err @ e2)
        linf.append(max(abs(eE), abs(eD))); nfix.append(nf)
    return float(np.percentile(linf, 90)), float(np.mean(nfix))


def main():
    print("RECONCILE d3-warm-clears (p90 0.124) vs perfect-vel-straight-leg-fails (p90 0.222)\n")
    p90, nf = leg(37.0, None)
    print(f"  perfect vel, NO vel-update     : Linf_p90={p90:.3f}  ({nf:.1f} fixes)")
    p90, nf = leg(37.0, "continuous")
    print(f"  perfect vel, CONTINUOUS vel-upd: Linf_p90={p90:.3f}  ({nf:.1f} fixes)"
          f"   <-- d3 'warm' mechanism on a straight leg")
    print()
    print("  => If continuous-vel-update on a STRAIGHT leg still fails p90, then d3's warm clears")
    print("     ONLY via the multi-gate lap pre-tightening (more fixes / smaller entering-P), and")
    print("     the 'velocity channel' lever (claim 5) does NOT itself close the g3->g4 floor.")
    print()
    # widen the window to emulate the multi-gate pre-convergence (more fixes feeding g4):
    print("  window sweep (perfect vel, no vel-update) -- emulates extra fixes from a longer approach:")
    for w in [12.0, 18.0, 24.0]:
        p90, nf = leg(37.0, None, fix_window_m=w)
        print(f"    window={w:>4.0f} m -> Linf_p90={p90:.3f}  ({nf:.1f} fixes)")


if __name__ == "__main__":
    main()
