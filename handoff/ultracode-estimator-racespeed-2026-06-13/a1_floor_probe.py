"""a1_floor_probe.py -- the decisive "can the floor come down" experiment.

Holds the best cell (const-v, edge L=6ms, 47% acceptance ~14Hz, DEBIASED so it is the pure
VARIANCE question) and sweeps the per-fix measurement 1-sigma down by overriding the fix-cov
floor + PnP default. Measures the filtered in-plane scatter at the gate-4 crossing as a
function of per-fix sigma. Answers: what per-fix accuracy would the KF need for the filtered
in-plane 1-sigma to reach <0.05 m at 37 m/s?  Reuses a1_sim machinery but with an overridable
fix covariance.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/a1_floor_probe.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from racer.frames import R_world_from_body  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402
import a1_sim as A  # the sim module (reuse geometry + residual pools)  # noqa: E402

HERE = Path(__file__).resolve().parent


def run_floor_cell(per_fix_sigma_m, accept, eff_label, n_seeds=400, base_seed=A.BASE_SEED):
    """Const-v, edge L=6ms, DEBIASED, ISOTROPIC per-fix cov = per_fix_sigma^2 I (override).
    Pure variance question: how low must per-fix sigma be for filtered in-plane sigma<0.05?"""
    pools, means, _ = A.load_residual_pools()
    L_s = 0.006
    t_cross = A.SEG_LEN / A.SPEED
    R_wb = R_world_from_body(0.0, A.DRAG_HOLD_PITCH_RAD, A.YAW_RAD)
    g = GRAVITY_NED
    det_dt = 1.0 / A.DETECTOR_HZ
    cov = (per_fix_sigma_m ** 2) * np.eye(3)
    n_imu = int(np.ceil(t_cross / A.IMU_DT)) + 2

    errs = []
    Pds = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + 7919 * s + int(per_fix_sigma_m * 1000))
        p0 = A.truth_pos(0.0, 0.0); v0 = A.truth_vel(0.0, 0.0)
        p_init = p0 + rng.normal(0, 0.6, 3); v_init = v0 + rng.normal(0, 0.5, 3)
        rkf = RewindKF(kf=LinearKF.initialize(p_init, v_init, pos_std=0.6, vel_std=0.5),
                       horizon_s=A.HORIZON_S)
        det_times = np.arange(0.0, t_cross + 1e-9, det_dt)
        accepted = rng.random(det_times.shape[0]) < accept
        events = []
        for tc, ok in zip(det_times, accepted):
            if not ok:
                continue
            p_true_cap = A.truth_pos(tc, 0.0)
            rng_to_g4 = A.SEG_LEN - float(np.dot(p_true_cap - A.G3, A.SEG_HAT))
            band = A.band_for_range(max(0.0, rng_to_g4))
            pool = pools[band]
            # DEBIASED zero-mean noise, but SCALED so its std matches per_fix_sigma (pure
            # variance probe): take the de-biased residual and rescale to isotropic per_fix_sigma
            resid = pool[rng.integers(0, pool.shape[0])] - means[band]
            # rescale per-axis to the target sigma (keeps shape, sets magnitude)
            band_std = pool.std(axis=0, ddof=1)
            resid = resid / band_std * per_fix_sigma_m
            z = p_true_cap + resid
            events.append([tc + L_s, tc, z])
        events.sort(key=lambda e: e[0])
        ev_i = 0
        t = 0.0
        for k in range(1, n_imu + 1):
            t_prev = t; t = k * A.IMU_DT
            if t > t_cross:
                dt = t_cross - t_prev; t = t_cross
            else:
                dt = A.IMU_DT
            if dt > 0:
                accel_body = R_wb.T @ (np.zeros(3) - g)
                accel_body = accel_body + rng.normal(0, 0.3, 3)
                rkf.predict(accel_body, R_wb, dt, sim_time_ns=int(round(t * 1e9)))
            while ev_i < len(events) and events[ev_i][0] <= t + 1e-12:
                _, cap_t, z = events[ev_i]
                rkf.update_position_at(int(round(cap_t * 1e9)), z, cov)
                ev_i += 1
            if t >= t_cross:
                break
        errs.append(rkf.position - A.truth_pos(t_cross, 0.0))
        Pds.append(np.diag(rkf.P)[:3].copy())
    errs = np.asarray(errs); Pds = np.asarray(Pds)
    std = errs.std(axis=0, ddof=1)
    claimed = np.sqrt(Pds.mean(axis=0))
    inplane_std = float(np.sqrt(std[1] ** 2 + std[2] ** 2))
    inplane_claimed = float(np.sqrt(claimed[1] ** 2 + claimed[2] ** 2))
    return dict(per_fix_sigma=per_fix_sigma_m, eff=eff_label, accept=accept,
                std_NED=std.tolist(), claimed_NED=claimed.tolist(),
                inplane_std=inplane_std, inplane_claimed=inplane_claimed,
                E_std=float(std[1]), D_std=float(std[2]))


def main():
    rows = []
    print("Floor-down probe (const-v, L=6ms, ~14Hz/47% accept, pure-variance DEBIASED):")
    print(f"{'per-fix sig':>11} | {'E std':>6} {'D std':>6} | {'inplane std':>11} {'claimed':>8} | clears0.05?")
    for sig in [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03]:
        r = run_floor_cell(sig, 0.47, "14Hz_47pct")
        clears = r["inplane_std"] < 0.05
        rows.append(r)
        print(f"{sig:>11.3f} | {r['E_std']:>6.3f} {r['D_std']:>6.3f} | "
              f"{r['inplane_std']:>11.3f} {r['inplane_claimed']:>8.3f} | {clears}")
    # also at higher fix rates to see if rate can substitute for accuracy
    print("\nSame, at 30Hz/100% accept (idealised: every detector frame accepted):")
    for sig in [0.50, 0.30, 0.20, 0.15, 0.10]:
        r = run_floor_cell(sig, 1.0, "30Hz_100pct")
        clears = r["inplane_std"] < 0.05
        rows.append(r)
        print(f"{sig:>11.3f} | {r['E_std']:>6.3f} {r['D_std']:>6.3f} | "
              f"{r['inplane_std']:>11.3f} {r['inplane_claimed']:>8.3f} | {clears}")
    (HERE / "a1_floor_probe.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {HERE / 'a1_floor_probe.json'}")


if __name__ == "__main__":
    main()
