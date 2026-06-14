"""TRUSTWORTHY accel-bias observability on a REALISTIC continuous trajectory with CONSTANT-RATE fixes.

The generator-based study (accel_bias_observability.py) is faithful for the boresight (a per-fix GEOMETRY
property, gap-insensitive) but NOT for accel-bias, whose observability is acutely GAP-sensitive (per-gap
bias drift ~ 0.5*b_a*tau^2). The geometric generator's artificial inter-gate gaps (capped 2 s -> huge -2R
coupling) make accel-bias look too easy. This module instead flies a CONTINUOUS race trajectory, samples
the IMU at 90 Hz, and applies position fixes at a CONSTANT realised rate (fr*30 Hz), so the inter-fix
dynamics are realistic. Predict is sub-stepped at IMU rate (production Q); the accel-bias coupling
-0.5 dt^2 R_wb / -dt R_wb is the EXACT constant-accel discretisation.

Reports sigma(b_a) vs (fix-rate, laps) and the laps-to-budget (|b_a_horiz| < g*sin(0.6deg)=0.103 m/s^2),
the GYRO-DEAD confirmation, and a clean NEES MC.  NO src/ edits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch-eskf"))
import eskf_geometry as G   # noqa: E402  (R_world_from_body, load_course, skew)

ACCEL_NOISE_STD = 0.3
ATT_NOISE_STD = np.deg2rad(1.4)
GRAVITY = 9.80665
IMU_DT = 1.0 / 90.0
DET_HZ = 30.0
FIX_FLOOR = 0.40
A1 = 0.026
PNP_BASE = 0.05
G_SIN_BUDGET = GRAVITY * np.sin(np.deg2rad(0.6))   # 0.103 m/s^2


def Q_pv_step():
    s = np.array([0.0, 0.0, -GRAVITY])
    S = G.skew(s)
    accel_cov = ACCEL_NOISE_STD ** 2 * np.eye(3) + ATT_NOISE_STD ** 2 * (S @ S.T)
    B = np.vstack([0.5 * IMU_DT * IMU_DT * np.eye(3), IMU_DT * np.eye(3)])
    return B @ accel_cov @ B.T


_QPV = Q_pv_step()


def fly_trajectory(gates, *, speed=30.0, n_laps=1):
    """Continuous const-speed flight through the gates (looped). Returns per-IMU-step arrays:
    times, p (world), v (world), R_wb (body->world). Attitude: yaw=heading, pitch from climb slope,
    roll~0. A real lap returns gate5->gate0 (a big turn) -- the maneuver that excites the bias."""
    waypts = [gates[i] for _ in range(n_laps) for i in range(len(gates))]
    waypts = [gates[0]] + waypts + [gates[0]]            # start and end at gate 0
    segs = []
    for a, b in zip(waypts[:-1], waypts[1:]):
        d = np.linalg.norm(b - a)
        if d < 1e-6:
            continue
        segs.append((a, b, d))
    times, ps, vs, Rs = [], [], [], []
    t = 0.0
    for a, b, d in segs:
        dirw = (b - a) / d
        n_steps = max(1, int(round((d / speed) / IMU_DT)))
        v = dirw * speed
        yaw = np.arctan2(dirw[1], dirw[0])
        pitch = np.arctan2(-dirw[2], np.hypot(dirw[0], dirw[1]))   # climb slope -> body pitch
        R_wb = G.R_world_from_body(0.0, pitch, yaw)
        for i in range(n_steps):
            p = a + dirw * (speed * i * IMU_DT)
            times.append(t); ps.append(p); vs.append(v); Rs.append(R_wb)
            t += IMU_DT
    return np.array(times), np.array(ps), np.array(vs), Rs


def run_filter_cov(times, ps, vs, Rs, *, fix_interval_s, sigma_z_floor=FIX_FLOOR, q_ba=0.0,
                   prior_ba=0.5, dead_gyro=False, true_ba=None, seed=None, bias_axes=(0, 1, 2)):
    """Augmented [p,v,b_a(+gyro)] filter over the trajectory. ``bias_axes`` selects which BODY accel-bias
    axes are estimated (0=along-track/X, 1=lateral/Y, 2=vertical/Z). The recommended design estimates the
    MARGIN-relevant in-plane subspace (1,2) and DROPS the unobservable along-track axis (0). Covariance-
    only unless true_ba+seed given (then also runs the mean for an MC error/NEES). Returns (P, sigma trace,
    optional (err,nees)); err/sigma are over the SELECTED axes."""
    ax = list(bias_axes)
    na = len(ax)
    nb = na + (3 if dead_gyro else 0)
    dim = 6 + nb
    bi = slice(6, 6 + na)                                    # bias block
    P = np.zeros((dim, dim))
    P[:3, :3] = 5.0 ** 2 * np.eye(3); P[3:6, 3:6] = 1.0 ** 2 * np.eye(3)
    P[bi, bi] = prior_ba ** 2 * np.eye(na)
    if dead_gyro:
        P[6 + na:6 + na + 3, 6 + na:6 + na + 3] = np.deg2rad(0.5) ** 2 * np.eye(3)
    rng = np.random.default_rng(seed) if seed is not None else None
    x = np.zeros(dim)
    if rng is not None:
        x[:3] = ps[0] + rng.normal(0, 5.0, 3); x[3:6] = vs[0]
    tb_full = np.zeros(3) if true_ba is None else np.asarray(true_ba, float)
    tb = tb_full[ax]                                         # truth on the estimated axes
    next_fix = times[0] + fix_interval_s
    sig_tr = []
    for k in range(1, len(times)):
        R_wb = Rs[k]
        Rsel = R_wb[:, ax]                                   # (3,na) coupling on the selected body axes
        F = np.eye(dim)
        F[:3, 3:6] = IMU_DT * np.eye(3)
        F[:3, bi] = -0.5 * IMU_DT * IMU_DT * Rsel
        F[3:6, bi] = -IMU_DT * Rsel
        Q = np.zeros((dim, dim)); Q[:6, :6] = _QPV; Q[bi, bi] = (q_ba ** 2) * IMU_DT * np.eye(na)
        P = F @ P @ F.T + Q
        if rng is not None:
            a_true_world = (vs[k] - vs[k - 1]) / IMU_DT          # ~0 within a segment, spike at turns
            a_body_meas = R_wb.T @ (a_true_world - np.array([0, 0, GRAVITY])) + tb_full \
                + rng.normal(0, ACCEL_NOISE_STD, 3)             # carries the FULL true body bias
            corr = np.zeros(3); corr[ax] = x[bi]                 # subtract only the ESTIMATED-axis bias
            a_world_est = R_wb @ (a_body_meas - corr) + np.array([0, 0, GRAVITY])
            x[:3] = x[:3] + IMU_DT * x[3:6] + 0.5 * IMU_DT * IMU_DT * a_world_est
            x[3:6] = x[3:6] + IMU_DT * a_world_est
        if times[k] >= next_fix:
            next_fix += fix_interval_s
            r = 12.0
            sz2 = PNP_BASE ** 2 + (A1 * r) ** 2 + sigma_z_floor ** 2
            R = sz2 * np.eye(3)
            H = np.zeros((3, dim)); H[:, :3] = np.eye(3)
            S = H @ P @ H.T + R
            Kk = np.linalg.solve(S, (P @ H.T).T).T
            if rng is not None:
                z = ps[k] + rng.multivariate_normal(np.zeros(3), R)
                x = x + Kk @ (z - H @ x)
            P = (np.eye(dim) - Kk @ H) @ P; P = 0.5 * (P + P.T)
            sd = np.sqrt(np.maximum(np.diag(P[bi, bi]), 0.0))
            sig_tr.append(float(np.linalg.norm(sd[[a in (1, 2) for a in ax]] if na > 1 else sd)))
    out = (P, np.array(sig_tr))
    if rng is not None:
        be = x[bi] - tb
        nees = float(be @ np.linalg.solve(P[bi, bi], be))
        out = out + ((be, nees, np.sqrt(np.diag(P[bi, bi]))),)
    return out


def _horiz(P, bi0=6, na=2):
    """norm of the in-plane (lateral+vertical) bias sigma. For the (1,2) subspace that is both axes."""
    return float(np.linalg.norm(np.sqrt(np.maximum(np.diag(P[bi0:bi0 + na, bi0:bi0 + na]), 0.0))))


def main():
    gates, _ = G.load_course()
    print(f"# realistic continuous trajectory; budget |b_a_inplane| < {G_SIN_BUDGET:.3f} m/s^2")
    print("# RECOMMENDED design = 2-axis accel-bias on BODY lateral(Y)+vertical(Z) (the gate-plane axes);")
    print("# along-track(X) is DROPPED (unobservable, aliased w/ velocity, maps to gate-normal).\n")
    out = {"budget": G_SIN_BUDGET, "axes": "body Y,Z (lateral,vertical)", "by_rate": {}}
    YZ = (1, 2)

    print("=== (1) 2-axis (Y,Z) sigma(b_a_inplane) vs laps, CONSTANT-rate fixes, q_ba=0 ===")
    print(f"{'fix-rate':<10}{'tau_s':>7}  " + "  ".join(f"{n}lap" for n in (1, 2, 3, 6, 12)))
    for fr, tag in ((0.07, "fr0.07"), (0.15, "fr0.15"), (0.25, "fr0.25")):
        tau = 1.0 / (fr * DET_HZ); row = []
        for nl in (1, 2, 3, 6, 12):
            times, ps, vs, Rs = fly_trajectory(gates, speed=30.0, n_laps=nl)
            P, _ = run_filter_cov(times, ps, vs, Rs, fix_interval_s=tau, q_ba=0.0, bias_axes=YZ)
            row.append(_horiz(P))
        flags = ["<" if h < G_SIN_BUDGET else " " for h in row]
        print(f"{tag:<10}{tau:>7.3f}  " + "  ".join(f"{h:.3f}{f}" for h, f in zip(row, flags)))
        out["by_rate"][tag] = {"tau_s": tau, "sigma_inplane_by_lap": dict(zip((1, 2, 3, 6, 12), row))}

    print("\n=== (2) OBSERVABILITY-SUBSPACE: 3-axis (X,Y,Z) MC reveals along-track X is UNOBSERVABLE ===")
    tau = 1.0 / (0.07 * DET_HZ)
    for axes, lbl in (((0, 1, 2), "3-axis X,Y,Z"), (YZ, "2-axis Y,Z (recommended)")):
        E = []; N = []
        for s in range(80):
            times, ps, vs, Rs = fly_trajectory(gates, speed=30.0, n_laps=6)
            _, _, (be, ne, sd) = run_filter_cov(times, ps, vs, Rs, fix_interval_s=tau, q_ba=0.0,
                                                true_ba=[0.10, -0.08, 0.05], seed=s, bias_axes=axes)
            E.append(be); N.append(ne)
        E = np.array(E)
        print(f"  {lbl:<26}: err_mean {np.array2string(E.mean(0),precision=3)}  "
              f"err_rms {np.array2string(np.sqrt((E**2).mean(0)),precision=3)}  "
              f"NEES {np.mean(N):.2f} (target ~{len(axes)}.0)")
        out[f"mc_{len(axes)}axis"] = dict(err_mean=E.mean(0).tolist(),
                                          err_rms=np.sqrt((E ** 2).mean(0)).tolist(), nees=float(np.mean(N)))

    print("\n=== (3) laps-to-budget (2-axis Y,Z), q_ba=0 ===")
    for fr, tag in ((0.07, "fr0.07"), (0.15, "fr0.15"), (0.25, "fr0.25")):
        tau = 1.0 / (fr * DET_HZ); lap_hit = None
        for nl in range(1, 31):
            times, ps, vs, Rs = fly_trajectory(gates, speed=30.0, n_laps=nl)
            P, _ = run_filter_cov(times, ps, vs, Rs, fix_interval_s=tau, q_ba=0.0, bias_axes=YZ)
            if _horiz(P) < G_SIN_BUDGET:
                lap_hit = nl; break
        print(f"  {tag}: {'lap '+str(lap_hit) if lap_hit else '>30 laps'}")
        out["by_rate"][tag]["laps_to_budget"] = lap_hit

    print("\n=== (4) drift floor: steady-state sigma_inplane vs random walk q_ba (2-axis, fr0.07, 12 laps) ===")
    times, ps, vs, Rs = fly_trajectory(gates, speed=30.0, n_laps=12)
    tau = 1.0 / (0.07 * DET_HZ)
    out["drift"] = []
    for q in (0.0, 0.005, 0.01, 0.02, 0.05):
        P, _ = run_filter_cov(times, ps, vs, Rs, fix_interval_s=tau, q_ba=q, bias_axes=YZ)
        h = _horiz(P)
        print(f"  q_ba={q:.3f} -> ss sigma_inplane {h:.3f} m/s^2  {'<budget' if h<G_SIN_BUDGET else 'OVER'}")
        out["drift"].append(dict(q_ba=q, ss_sigma=h))

    print("\n=== (5) GYRO-DEAD confirmation (2-axis accel + dead gyro, fr0.07, 6 laps) ===")
    times, ps, vs, Rs = fly_trajectory(gates, speed=30.0, n_laps=6)
    P, _ = run_filter_cov(times, ps, vs, Rs, fix_interval_s=1.0 / (0.07 * DET_HZ), q_ba=0.0,
                          dead_gyro=True, bias_axes=YZ)
    sba = _horiz(P); sbg = np.rad2deg(np.sqrt(np.diag(P[8:11, 8:11])))
    print(f"  accel-bias inplane {sba:.3f} m/s^2 (LIVE); gyro-bias {np.array2string(sbg,precision=3)} deg/s "
          f"(prior 0.500 -> UNCHANGED = DEAD; zero coupling, no measurement)")
    out["gyro_dead"] = dict(sba_inplane=sba, sbg_degps=sbg.tolist())

    with open("accelbias_realistic_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n# wrote accelbias_realistic_results.json")


if __name__ == "__main__":
    main()
