"""ACCEL-BIAS ESKF observability + the GYRO-DEAD-STATE demonstration (P1-ESKF-RESCOPE, 2026-06-14).

Re-scope of the dropped online boresight corrector. The boresight is now an OFFLINE EXTRINSIC BAKE
(frames.BoresightCorrection, calib-v2) -- NOT a filter state. This study asks the re-scoped question:
which TIME-VARYING IMU-bias states can the [p,v] LinearKF actually estimate, and do they buy margin?

ACCEL-BIAS (3-axis, body FRD). It enters the production KF PREDICT directly:
    a_world = R_world_body @ (accel_body - b_a) + g           (b_a = body accelerometer bias)
A constant body-frame b_a injects a world acceleration error R_wb @ b_a that ROTATES with attitude and
integrates into velocity/position drift between fixes; the position fixes then observe it. This is the
classic aided-INS accel-bias problem. CRUCIALLY in our GIVEN-attitude architecture b_a is also the sink
for a given-attitude TILT bias's gravity leakage: a small tilt error delta_theta leaves an uncompensated
g*sin(theta) horizontal accel == skew(specific_force_world)@delta_theta -- the SAME term the predict
already INFLATES Q for via attitude_noise_std^2 (S S^T) (state_estimator.py:117-118). The ESKF ESTIMATES
the MEAN of that term instead of merely inflating -- the "estimate-not-inflate" principle, RE-TARGETED
from the (now offline-baked) boresight fix-lever term skew(L) to the predict specific-force term skew(s).

The margin (margin-closure-envelope) binds on "effective attitude/accel bias <= ~0.6 deg" with the
explicit model accel = g*sin(theta): 0.6 deg <-> g*sin(0.6deg) = 0.103 m/s^2. So removing b_a directly
shrinks the binding margin term. Headline metric: does sigma(b_a_horizontal) drop below 0.103 m/s^2 at the
realized fix stream, and at what fix count?

GYRO-BIAS: attitude is GIVEN (ODOMETRY quat via frames.R_world_from_odo_quat_wxyz), NOT gyro-integrated in
the LinearKF (state = [p,v] only). A gyro-bias state has ZERO coupling to [p,v] and NO measurement -> a
DEAD state. Demonstrated below (its posterior == its prior, forever).

NO src/ edits. Pure numpy. Imports the realized-distribution geometry from ../scratch-eskf/eskf_geometry.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch-eskf"))
import eskf_geometry as G   # noqa: E402  (skew, R_world_from_body, make_fix_stream, load_course)

# production constants (mirror src/racer/state_estimator.py + localization.py)
ACCEL_NOISE_STD = 0.3
ATT_NOISE_STD = np.deg2rad(1.4)
GRAVITY = 9.80665
IMU_DT = 1.0 / 90.0
A1_RANGE_GROWTH = 0.026
PNP_BASE_STD = 0.05
FIX_FLOOR = 0.40
G_SIN_BUDGET = GRAVITY * np.sin(np.deg2rad(0.6))    # 0.103 m/s^2 == the 0.6 deg effective-bias budget


def Q_pv(tau):
    s = np.array([0.0, 0.0, -GRAVITY])
    S = G.skew(s)
    accel_cov = ACCEL_NOISE_STD ** 2 * np.eye(3) + ATT_NOISE_STD ** 2 * (S @ S.T)
    B = np.vstack([0.5 * tau * tau * np.eye(3), tau * np.eye(3)])
    return B @ accel_cov @ B.T


def fix_cov_world(rec):
    """Conservative absolute-fix cov (localization.gate_pose_to_world_position): PnP(range-grown) +
    full 1.4deg attitude lever-arm + 0.40 m floor. The SAFE bound on observability."""
    r = rec["range_m"]
    L = rec["L"]
    pnp = (PNP_BASE_STD ** 2 + (A1_RANGE_GROWTH * r) ** 2) * np.eye(3)
    lever = ATT_NOISE_STD ** 2 * (float(L @ L) * np.eye(3) - np.outer(L, L))
    return pnp + lever + FIX_FLOOR ** 2 * np.eye(3)


def F_accelbias(R_wb, dt, dim):
    """Augmented [p,v,b_a] transition. b_a enters via a_world = R_wb @ (accel - b_a) + g, so the
    state-error coupling is dp += -0.5 dt^2 R_wb db_a ; dv += -dt R_wb db_a ; db_a unchanged."""
    F = np.eye(dim)
    F[:3, 3:6] = dt * np.eye(3)
    F[:3, 6:9] = -0.5 * dt * dt * R_wb
    F[3:6, 6:9] = -dt * R_wb
    return F


def cov_recursion_accelbias(recs, *, prior_ba=0.5, q_ba=0.02, imu_dt=IMU_DT, extra_dead_gyro=False):
    """Augmented-KF covariance propagation for [p(3), v(3), b_a(3)] (+ optional DEAD gyro b_g(3)).
    prior_ba = cold accel-bias 1-sigma (m/s^2); q_ba = bias random-walk (m/s^2/sqrt-s).
    Returns final P and the b_a sigma trace. The dead-gyro block (if added) has ZERO coupling/measure."""
    nb = 3 + (3 if extra_dead_gyro else 0)
    dim = 6 + nb
    P = np.zeros((dim, dim))
    P[:3, :3] = 5.0 ** 2 * np.eye(3)
    P[3:6, 3:6] = 1.0 ** 2 * np.eye(3)
    P[6:9, 6:9] = prior_ba ** 2 * np.eye(3)
    if extra_dead_gyro:
        P[9:12, 9:12] = np.deg2rad(0.5) ** 2 * np.eye(3)     # gyro-bias prior 0.5 deg/s
    trace = []
    t_prev = None
    for k, rec in enumerate(recs):
        t_now = rec.get("t", k * imu_dt)
        tau = imu_dt if t_prev is None else min(max(t_now - t_prev, imu_dt), 2.0)
        n_sub = max(1, int(round(tau / imu_dt)))
        for _ in range(n_sub):
            F = np.eye(dim)
            F[:6, :6] = np.eye(6)
            F[:3, 3:6] = imu_dt * np.eye(3)
            F[:3, 6:9] = -0.5 * imu_dt * imu_dt * rec["R_wb"]
            F[3:6, 6:9] = -imu_dt * rec["R_wb"]
            # dead gyro block: F == I (no coupling), pure (optional) random walk
            Q = np.zeros((dim, dim))
            Q[:6, :6] = Q_pv(imu_dt)
            Q[6:9, 6:9] = (q_ba ** 2) * imu_dt * np.eye(3)
            if extra_dead_gyro:
                Q[9:12, 9:12] = (np.deg2rad(0.01) ** 2) * imu_dt * np.eye(3)   # tiny gyro RW
            P = F @ P @ F.T + Q
        # position fix: H = [I,0,0,(0)]
        H = np.zeros((3, dim))
        H[:, :3] = np.eye(3)
        R = fix_cov_world(rec)
        S = H @ P @ H.T + R
        Kk = np.linalg.solve(S, (P @ H.T).T).T
        P = (np.eye(dim) - Kk @ H) @ P
        P = 0.5 * (P + P.T)
        trace.append(np.sqrt(np.maximum(np.diag(P[6:9, 6:9]), 0.0)))
        t_prev = t_now          # FIX: advance the clock so the next gap is real (was missing -> n_sub
        #                         always 1 -> gaps ignored -> bias coupling suppressed -> false "unobservable")
    return P, np.array(trace)


def mc_accelbias(recs, *, true_ba, prior_ba=0.5, q_ba=0.02, n_mc=200, seed=0):
    """FAITHFUL [p,v,b_a] KF MC: predict is SUB-STEPPED at IMU rate (production 90 Hz Q, white accel noise
    per imu step) -- NOT one-big-step-per-gap (which 72x-inflates Q_vv and artificially aids the bias).
    Truth integrates the SAME per-step constant accel; the fix lands at the gap end. Confirms unbiased
    estimate + honest NEES (~3 for the 3-axis bias) under the CORRECT noise model = matches cov_recursion."""
    rng = np.random.default_rng(seed)
    dim = 9
    ts = np.array([r["t"] for r in recs])
    ps = np.array([r["p_drone"] for r in recs])
    err = np.zeros((n_mc, 3))
    nees = np.zeros(n_mc)
    for m in range(n_mc):
        x = np.zeros(dim)
        x[:3] = ps[0] + rng.normal(0, 5.0, 3)
        x[3:6] = (ps[1] - ps[0]) / max(ts[1] - ts[0], IMU_DT)
        P = np.zeros((dim, dim))
        P[:3, :3] = 5.0 ** 2 * np.eye(3)
        P[3:6, 3:6] = 1.0 ** 2 * np.eye(3)
        P[6:9, 6:9] = prior_ba ** 2 * np.eye(3)
        p_true = ps[0].copy()
        v_true = (ps[1] - ps[0]) / max(ts[1] - ts[0], IMU_DT)
        for k, rec in enumerate(recs):
            if k > 0:
                tau = max(ts[k] - ts[k - 1], IMU_DT)
                n_sub = max(1, int(round(min(tau, 2.0) / IMU_DT)))
                dt = min(tau, 2.0) / n_sub
                a_true = 2.0 * (ps[k] - p_true - v_true * tau) / tau ** 2          # const-accel to hit p_k
                R_wb = rec["R_wb"]
                for _ in range(n_sub):                                            # SUB-STEP at IMU rate
                    a_body_meas = R_wb.T @ (a_true - np.array([0, 0, GRAVITY])) + true_ba \
                        + rng.normal(0, ACCEL_NOISE_STD, 3)                        # white accel noise / step
                    a_world_est = R_wb @ (a_body_meas - x[6:9]) + np.array([0, 0, GRAVITY])
                    F = F_accelbias(R_wb, dt, dim)
                    x[:3] = x[:3] + dt * x[3:6] + 0.5 * dt * dt * a_world_est
                    x[3:6] = x[3:6] + dt * a_world_est
                    Q = np.zeros((dim, dim))
                    Q[:6, :6] = Q_pv(dt)
                    Q[6:9, 6:9] = (q_ba ** 2) * dt * np.eye(3)
                    P = F @ P @ F.T + Q
                    p_true = p_true + dt * v_true + 0.5 * dt * dt * a_true         # truth integrates same accel
                    v_true = v_true + dt * a_true
            H = np.zeros((3, dim))
            H[:, :3] = np.eye(3)
            R = fix_cov_world(rec)
            z = p_true + rng.multivariate_normal(np.zeros(3), R)
            y = z - H @ x
            S = H @ P @ H.T + R
            Kk = np.linalg.solve(S, (P @ H.T).T).T
            x = x + Kk @ y
            P = (np.eye(dim) - Kk @ H) @ P
            P = 0.5 * (P + P.T)
        be = x[6:9] - true_ba
        err[m] = be
        nees[m] = float(be @ np.linalg.solve(P[6:9, 6:9], be))
    return err, nees


def main():
    gates, src = G.load_course()
    print(f"# course: {Path(src).name}; {len(gates)} gates; budget |b_a_horiz| <= {G_SIN_BUDGET:.3f} m/s^2 "
          f"(= g*sin(0.6deg))")
    out = {"course_src": src, "g_sin_budget_ms2": G_SIN_BUDGET, "scenarios": {}}

    SCN = {
        "g4_terminal_inband":   ([4], 5.0, 12.0, 2, 1, "terminal in-band only, sparse"),
        "pre_terminal_g0to3":   ([0, 1, 2, 3], 5.0, 30.0, 7, 1, "lap-1 gates 0-3: b_a carried INTO first g4"),
        "g4_full_approach":     ([4], 5.0, 30.0, 7, 1, "single gate, full approach"),
        "pooled_lap":           (list(range(6)), 5.0, 30.0, 7, 1, "all 6 gates, 1 lap"),
        "pooled_3laps":         (list(range(6)), 5.0, 30.0, 7, 3, "all 6 gates, 3 laps"),
        "pooled_3laps_fr0.25":  (list(range(6)), 5.0, 30.0, 25, 3, "all gates, 3 laps, fr 0.25"),
    }
    print("\n=== (A) ACCEL-BIAS posterior 1-sigma (m/s^2) per body axis; horiz = hypot(x,y) vs budget ===")
    print(f"{'scenario':<24}{'Nfix':>5}{'sd_bx':>8}{'sd_by':>8}{'sd_bz':>8}{'sd_horiz':>10}{'<budget?':>9}")
    for name, (gids, rmn, rmx, npg, nl, note) in SCN.items():
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
        recs = G.make_fix_stream(rng, gates_ned=gates, gate_ids=gids, r_min=rmn, r_max=rmx,
                                 n_fix_per_gate=npg, crab_mean_deg=37.5, crab_spread_deg=8.0, n_laps=nl)
        P, tr = cov_recursion_accelbias(recs)
        sd = np.sqrt(np.diag(P[6:9, 6:9]))
        horiz = float(np.hypot(sd[0], sd[1]))
        ok = "YES" if horiz < G_SIN_BUDGET else "no"
        print(f"{name:<24}{len(recs):>5}{sd[0]:>8.3f}{sd[1]:>8.3f}{sd[2]:>8.3f}{horiz:>10.3f}{ok:>9}")
        out["scenarios"][name] = dict(n_fix=len(recs), sd_bx=float(sd[0]), sd_by=float(sd[1]),
                                      sd_bz=float(sd[2]), sd_horiz=horiz, under_budget=bool(horiz < G_SIN_BUDGET),
                                      note=note)

    # (B) MANEUVERING is the observability handle (parallel to the boresight diversity finding)
    print("\n=== (B) MANEUVERING EXCITATION is the handle: straight-constant-attitude vs maneuvering ===")
    rng = np.random.default_rng(7)
    # straight: a SINGLE gate approach (near-constant attitude) -- b_a confounded with velocity
    recs_straight = G.make_fix_stream(rng, gates_ned=gates, gate_ids=[4], r_min=5.0, r_max=30.0,
                                      n_fix_per_gate=42, crab_mean_deg=37.5, crab_spread_deg=1.0, n_laps=1)
    # maneuvering: all 6 gates (heading/pitch vary a lot) at the same total fix count
    recs_man = G.make_fix_stream(rng, gates_ned=gates, gate_ids=list(range(6)), r_min=5.0, r_max=30.0,
                                 n_fix_per_gate=7, crab_mean_deg=37.5, crab_spread_deg=8.0, n_laps=1)
    for label, recs in (("straight 1-gate", recs_straight), ("maneuvering 6-gate", recs_man)):
        P, _ = cov_recursion_accelbias(recs)
        sd = np.sqrt(np.diag(P[6:9, 6:9]))
        print(f"  {label:<20} N={len(recs):>3} -> sd_horiz={np.hypot(sd[0],sd[1]):.3f} m/s^2 "
              f"(sd_z={sd[2]:.3f})")
    out["maneuvering"] = dict(straight_n=len(recs_straight), maneuver_n=len(recs_man))

    # (C) GYRO is a DEAD state: zero coupling -> posterior == prior (no information ever)
    print("\n=== (C) GYRO-BIAS is a DEAD state (attitude GIVEN, not gyro-integrated) ===")
    rng = np.random.default_rng(3)
    recs = G.make_fix_stream(rng, gates_ned=gates, gate_ids=list(range(6)), r_min=5.0, r_max=30.0,
                             n_fix_per_gate=25, crab_mean_deg=37.5, crab_spread_deg=8.0, n_laps=3)
    P, _ = cov_recursion_accelbias(recs, extra_dead_gyro=True)
    sd_ba = np.sqrt(np.diag(P[6:9, 6:9]))
    sd_bg = np.rad2deg(np.sqrt(np.diag(P[9:12, 9:12])))
    print(f"  after N={len(recs)} fixes: accel-bias sd horiz={np.hypot(sd_ba[0],sd_ba[1]):.3f} m/s^2 "
          f"(LIVE, shrank from prior 0.5)")
    print(f"  gyro-bias sd = {np.array2string(sd_bg, precision=3)} deg/s "
          f"(prior was 0.500 deg/s -> UNCHANGED = DEAD; zero coupling to [p,v], no measurement)")
    out["gyro_dead"] = dict(sd_ba_horiz=float(np.hypot(sd_ba[0], sd_ba[1])),
                            sd_bg_degps=sd_bg.tolist(), gyro_prior_degps=0.5)

    # (D) MC: unbiased + honest NEES (~3 for 3-axis)
    print("\n=== (D) MONTE-CARLO: convergence to injected b_a + NEES (honest cov?) ===")
    rng = np.random.default_rng(11)
    recs_mc = G.make_fix_stream(rng, gates_ned=gates, gate_ids=list(range(6)), r_min=5.0, r_max=30.0,
                                n_fix_per_gate=12, crab_mean_deg=37.5, crab_spread_deg=8.0, n_laps=2)
    true_ba = np.array([0.10, -0.08, 0.05])
    err, nees = mc_accelbias(recs_mc, true_ba=true_ba, n_mc=200, seed=5)
    print(f"  N={len(recs_mc)} fixes, true b_a={true_ba}: est-error mean={np.array2string(err.mean(0), precision=3)} "
          f"m/s^2, rms_horiz={np.sqrt((err[:,:2]**2).sum(1).mean()):.3f}; NEES mean={nees.mean():.2f} (target ~3.0)")
    out["mc"] = dict(n_fix=len(recs_mc), err_mean=err.mean(0).tolist(),
                     rms_horiz=float(np.sqrt((err[:, :2] ** 2).sum(1).mean())), nees_mean=float(nees.mean()))

    with open("accel_bias_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n# wrote accel_bias_results.json")


if __name__ == "__main__":
    main()
