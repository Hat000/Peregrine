"""G3 ripple check for the in-plane STATE-cov floor (parked #74). Re-runs the VALIDATED g3 'rel' arm
(racer.kf_rewind.RewindKF + production gate-relative cov) with the floor OFF vs ON, at (a) the production
14 Hz fix rate and (b) a DENSE fix rate (a well-pointed inc8 stream), to confirm:
  - the floor does NOT regress the C2 G3 case-C margin (rel RMS/p90/p99/E_bias/D_bias ~unchanged);
  - at 14 Hz the in-band in-plane sigma stays >> 0.05 m so the floor is a NO-OP (bit-identical) -> the
    over-convergence the coast-drift iid MC implied does NOT reproduce with the realistic ~4-5-fix
    correlated stream;
  - even at a DENSE fix rate the floor only pins the covariance (honest), it does not move the centering
    error distribution adversely.

Mirrors handoff/c2-estimator-chain-2026-06-13/g3_margin_sim.py exactly, but parameterizes the KF floor +
the fix rate. Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/kf-pfloor-2026-06-15/g3_pfloor_ripple.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from racer.frames import ATTITUDE_NOISE_STD_RAD, R_world_from_body  # noqa: E402
from racer.kf_rewind import RewindKF  # noqa: E402
from racer.localization import (  # noqa: E402
    FIX_COV_FLOOR_STD,
    GATE_REL_INPLANE_SIGMA,
    INPLANE_POS_FLOOR_STD,
)
from racer.state_estimator import LinearKF  # noqa: E402

SEED = 20260613
V_RACE = 37.0
G3_BOTTOM = np.array([-111.5, -5.1, 24.57])
G4_BOTTOM = np.array([-135.5, -0.8, 25.36])
DT_IMU = 1.0 / 90.0
ABS_AXIS_SIGMA = 0.50
PER_AXIS_SIG = GATE_REL_INPLANE_SIGMA


def _inplane_min_sigma(P):
    w = np.linalg.eigvalsh(0.5 * (P[:2, :2] + P[:2, :2].T))
    return float(np.sqrt(max(w[0], 0.0)))


def _run_rel(floor_std: float, fix_hz: float, n_mc: int = 600) -> dict:
    seg = G4_BOTTOM - G3_BOTTOM
    Lseg = float(np.linalg.norm(seg))
    uhat = seg / Lseg
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    g = np.array([0.0, 0.0, 9.80665])
    accel_body = R_wb.T @ (-g)
    T = Lseg / V_RACE
    n_steps = int(T / DT_IMU)
    fix_dt = 1.0 / fix_hz
    E_errs, D_errs, ip_errs, min_sigs, n_fixes = [], [], [], [], []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 7000 * 3 + s)
        p0 = G3_BOTTOM.copy()
        v0 = uhat * V_RACE
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5,
                                 inplane_pos_floor_std=floor_std)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t, t_ns, next_fix, nf = 0.0, 0, 0.0, 0
        for _ in range(n_steps):
            t += DT_IMU
            t_ns += int(DT_IMU * 1e9)
            rk.predict(accel_body, R_wb, DT_IMU, t_ns)
            p_true = p0 + uhat * V_RACE * t
            rng_g4 = float(np.linalg.norm(G4_BOTTOM - p_true))
            if t >= next_fix and rng_g4 < 12.0:
                next_fix += fix_dt
                nf += 1
                nE, nD = r.normal(0, PER_AXIS_SIG), r.normal(0, PER_AXIS_SIG)
                var_along = ABS_AXIS_SIGMA ** 2 + (ATTITUDE_NOISE_STD_RAD * rng_g4) ** 2 + FIX_COV_FLOOR_STD ** 2
                z = p_true.copy()
                z[1] += nE
                z[2] += nD
                z[0] += r.normal(0, ABS_AXIS_SIGMA)
                cov = np.diag([var_along, PER_AXIS_SIG ** 2, PER_AXIS_SIG ** 2])
                rk.update_position(z, cov, sim_time_ns=t_ns)
        p_true_final = p0 + uhat * V_RACE * (n_steps * DT_IMU)
        err = rk.position - p_true_final
        E_errs.append(float(err[1]))
        D_errs.append(float(err[2]))
        ip_errs.append(float(np.hypot(err[1], err[2])))
        min_sigs.append(_inplane_min_sigma(rk.P))
        n_fixes.append(nf)
    E, D, ip = np.array(E_errs), np.array(D_errs), np.array(ip_errs)
    return dict(
        floor_std=floor_std, fix_hz=fix_hz, n_mc=n_mc, n_fixes_mean=float(np.mean(n_fixes)),
        inplane_rms=float(np.sqrt(np.mean(ip ** 2))),
        inplane_p50=float(np.percentile(ip, 50)),
        inplane_p90=float(np.percentile(ip, 90)),
        inplane_p99=float(np.percentile(ip, 99)),
        E_bias=float(E.mean()), D_bias=float(D.mean()),
        min_inplane_sigma_p50=float(np.percentile(min_sigs, 50)),
        min_inplane_sigma_p99=float(np.percentile(min_sigs, 99)),
    )


def main():
    out = {"seed": SEED, "v_race": V_RACE, "per_axis_inplane_sigma": PER_AXIS_SIG,
           "floor_std": INPLANE_POS_FLOOR_STD, "cases": []}
    print("=" * 104)
    print("G3 'rel' arm -- in-plane STATE-cov floor ripple (floor OFF vs ON), production 14 Hz + a dense stream")
    print("=" * 104)
    hdr = (f"{'fix_hz':>7} {'floor':>6} {'n_fix':>6} {'RMS':>7} {'p50':>7} {'p90':>7} {'p99':>7} "
           f"{'E_bias':>8} {'D_bias':>8} {'sig_p50':>8} {'sig_p99':>8}")
    print(hdr)
    for fix_hz in (14.0, 60.0, 120.0):
        for floor in (0.0, INPLANE_POS_FLOOR_STD):
            x = _run_rel(floor, fix_hz)
            out["cases"].append(x)
            print(f"{x['fix_hz']:7.0f} {x['floor_std']:6.2f} {x['n_fixes_mean']:6.1f} "
                  f"{x['inplane_rms']:7.3f} {x['inplane_p50']:7.3f} {x['inplane_p90']:7.3f} "
                  f"{x['inplane_p99']:7.3f} {x['E_bias']:+8.3f} {x['D_bias']:+8.3f} "
                  f"{x['min_inplane_sigma_p50']:8.4f} {x['min_inplane_sigma_p99']:8.4f}")
    p = Path(__file__).resolve().parent / "g3_pfloor_ripple_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
