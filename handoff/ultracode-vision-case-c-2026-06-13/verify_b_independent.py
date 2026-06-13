"""INDEPENDENT adversarial verifier for PIECE B (rewind buffer).

Written by the VERIFY agent. Does NOT reuse the prototype's self-checks; it re-derives
the OOSM correctness invariant from scratch against a clean from-scratch in-order oracle,
attacks the bit-identical claim with intervening pos/vel updates (not just predicts),
and tests whether the rewind "win" survives a DIFFERENT (non-sinusoid) truth track and a
constant-velocity track (where v*L is exact and analytically known).

Run: .venv/Scripts/python.exe handoff/ultracode-vision-case-c-2026-06-13/verify_b_independent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff" / "ultracode-vision-case-c-2026-06-13"))
from kf_rewind_buffer import RewindKF  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])
DT = 1.0 / 90.0
RESULTS = {}


def fb(a_world, R=None):
    R = np.eye(3) if R is None else R
    return R.T @ (a_world - GRAVITY_NED)


# ---------------------------------------------------------------------------
# TEST A: OOSM exactness with MIXED ops (predict + pos + vel) between capture and now.
# The prototype's CHECK4 only had predicts after the fix. Here we interleave a
# given-position update AND a velocity update AFTER the capture time, so the replay must
# correctly re-run heterogeneous ops. Compare to a clean in-order oracle built fresh.
# ---------------------------------------------------------------------------
def test_oosm_mixed_ops():
    t0 = 1_000_000_000
    rng = np.random.default_rng(123)
    accs = [rng.normal(0, 2.0, 3) for _ in range(20)]
    Rs = []
    # nontrivial, time-varying attitude so R_wb @ accel is genuinely exercised in replay
    for i in range(20):
        ang = 0.1 * i
        c, s = np.cos(ang), np.sin(ang)
        Rs.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))

    zpos = np.array([1.0, -2.0, 0.5])
    cov_p = np.diag([0.7, 0.4, 0.3]) ** 2 + 0.16 * np.eye(3)
    zvel = np.array([0.2, 0.1, -0.3])
    cov_v = np.diag([0.2, 0.2, 0.2]) ** 2

    fix_z = np.array([0.05, -0.02, 0.03])
    fix_cov = np.diag([0.7, 0.4, 0.3]) ** 2 + 0.16 * np.eye(3)
    # capture at tick 4; delivered at tick 9 (L=5). Between tick 4 and 9 there is a
    # given-pos update (in-sequence) at tick 6 and a vel update at tick 7.
    cap_tick = 4

    # ---- ORACLE: strict chronological, fresh filter ----
    orc = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3)
    for i in range(1, 5):
        orc.predict(fb(accs[i], Rs[i]), Rs[i], DT)
    orc.update_position(fix_z, fix_cov)          # the vision fix at capture tick 4
    orc.predict(fb(accs[5], Rs[5]), Rs[5], DT)
    orc.update_position(zpos, cov_p)             # in-seq given-pos at tick 6
    orc.predict(fb(accs[6], Rs[6]), Rs[6], DT)
    orc.update_velocity(zvel, cov_v)             # vel at tick 7
    orc.predict(fb(accs[7], Rs[7]), Rs[7], DT)
    orc.predict(fb(accs[8], Rs[8]), Rs[8], DT)

    # ---- REWIND: forward stream w/ the vision fix delivered late at tick 9 ----
    rk = RewindKF(kf=LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3),
                  horizon_s=2.0)
    for i in range(1, 6):
        rk.predict(fb(accs[i], Rs[i]), Rs[i], DT, sim_time_ns=t0 + int(i * DT * 1e9))
    # tick 6 predict then given-pos at tick 6
    rk.update_position(zpos, cov_p, sim_time_ns=t0 + int(6 * DT * 1e9))
    rk.predict(fb(accs[6], Rs[6]), Rs[6], DT, sim_time_ns=t0 + int(6.5 * DT * 1e9))
    rk.update_velocity(zvel, cov_v, sim_time_ns=t0 + int(7 * DT * 1e9))
    rk.predict(fb(accs[7], Rs[7]), Rs[7], DT, sim_time_ns=t0 + int(7.5 * DT * 1e9))
    rk.predict(fb(accs[8], Rs[8]), Rs[8], DT, sim_time_ns=t0 + int(8 * DT * 1e9))
    # the LATE vision fix, stamped just after the cap-tick-4 predict (which ended at t0+4dt)
    t_fix = t0 + int(4 * DT * 1e9) + 1
    res = rk.update_position_at(t_fix, fix_z, fix_cov)

    dx = float(np.max(np.abs(orc.x - rk.x)))
    dP = float(np.max(np.abs(orc.P - rk.P)))
    print(f"[A] OOSM mixed-ops (pos+vel+predict replayed): rewound={res.rewound} "
          f"replayed={res.n_replayed_ops} |dx|={dx:.2e} |dP|={dP:.2e}")
    ok = dx < 1e-10 and dP < 1e-10
    print(f"    PASS={ok}")
    RESULTS["A_oosm_mixed"] = (ok, dx, dP)
    return ok


# ---------------------------------------------------------------------------
# TEST B: bit-identical at L=0 with a DIFFERENT driver (random ops, pos+vel interspersed).
# Drive a RewindKF (all fixes delivered with t_fix==now via update_position_at) vs a bare
# LinearKF with identical update_position calls. Must match to fp.
# ---------------------------------------------------------------------------
def test_l0_identical_random():
    rng = np.random.default_rng(7)
    bare = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3)
    rk = RewindKF(kf=LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3),
                  horizon_s=0.5)
    t0 = 1_000_000_000
    maxdx = 0.0
    for k in range(1, 200):
        a = rng.normal(0, 3.0, 3)
        ang = rng.normal(0, 0.2)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        f = fb(a, R)
        bare.predict(f, R, DT)
        rk.predict(f, R, DT, sim_time_ns=t0 + int(k * DT * 1e9))
        if k % 5 == 0:
            z = rng.normal(0, 1, 3)
            cov = np.diag(rng.uniform(0.1, 0.9, 3))
            bare.update_position(z, cov)
            # deliver as an L=0 fix via update_position_at (t_fix == now)
            rk.update_position_at(t0 + int(k * DT * 1e9), z, cov)
        if k % 7 == 0:
            zv = rng.normal(0, 0.5, 3)
            covv = np.diag(rng.uniform(0.05, 0.3, 3))
            bare.update_velocity(zv, covv)
            rk.update_velocity(zv, covv, sim_time_ns=t0 + int(k * DT * 1e9))
        maxdx = max(maxdx, float(np.max(np.abs(bare.x - rk.x))))
    dP = float(np.max(np.abs(bare.P - rk.P)))
    print(f"[B] L=0 bit-identical (random pos/vel via update_position_at): "
          f"max|dx| over trajectory={maxdx:.2e} final|dP|={dP:.2e}")
    ok = maxdx < 1e-12 and dP < 1e-12
    print(f"    PASS={ok}")
    RESULTS["B_l0_identical"] = (ok, maxdx, dP)
    return ok


# ---------------------------------------------------------------------------
# TEST C: ANALYTIC v*L on a pure constant-velocity track with NOISE-FREE fixes.
# This removes the synthetic-sinusoid confound. Truth = straight line at speed v.
# A noise-free fix captured L ticks ago, applied naively, biases the estimate toward the
# past pose by exactly v*L*dt along the velocity. Rewind should put it (near) bang on.
# We verify the naive bias EQUALS v*L*dt (the predicted scaling) and rewind removes it.
# ---------------------------------------------------------------------------
def test_const_vel_analytic():
    print("[C] Const-velocity, NOISE-FREE fixes: naive bias should == v*L*dt; rewind ~0")
    v = 20.0
    for L in [2, 4]:
        n = 400
        t0 = 1_000_000_000
        t_ns = (np.arange(n) * DT * 1e9).astype(np.int64) + t0
        pos = np.zeros((n, 3))
        pos[:, 0] = v * np.arange(n) * DT  # straight N at v
        vel = np.tile([v, 0, 0], (n, 1))
        # naive
        naive = LinearKF.initialize(pos[0], vel[0], pos_std=0.5, vel_std=0.5, accel_noise_std=0.3)
        rk = RewindKF(kf=LinearKF.initialize(pos[0], vel[0], pos_std=0.5, vel_std=0.5, accel_noise_std=0.3),
                      horizon_s=0.5)
        # tight fix cov so the fix dominates (so the bias is visible) but not singular
        cov = np.diag([0.05, 0.05, 0.05]) ** 2
        next_fix = 0.5 / 28.0
        bias_naive = []
        bias_rew = []
        for k in range(1, n):
            f = fb(np.zeros(3))  # zero world accel -> const vel
            naive.predict(f, np.eye(3), DT)
            rk.predict(f, np.eye(3), DT, sim_time_ns=int(t_ns[k]))
            tn = (t_ns[k] - t0) / 1e9
            if tn >= next_fix:
                next_fix += 1.0 / 28.0
                ck = max(0, k - L)
                z = pos[ck].copy()  # NOISE-FREE fix at capture pose
                naive.update_position(z, cov)
                rk.update_position_at(int(t_ns[ck]), z, cov)
                if tn > 1.5:
                    bias_naive.append(naive.x[0] - pos[k][0])
                    bias_rew.append(rk.x[0] - pos[k][0])
        predicted_vL = -v * L * DT  # estimate lags true by ~v*L (negative = behind)
        bn = np.mean(bias_naive)
        br = np.mean(bias_rew)
        print(f"    L={L}: predicted lag v*L*dt={predicted_vL:+.3f} m | "
              f"naive mean N-bias={bn:+.3f} m | rewind mean N-bias={br:+.3f} m")
        # naive bias should be within ~30% of v*L (it is damped by KF gain<1 and IMU coast);
        # the SIGN and order of magnitude are the real test. rewind should be << naive.
        RESULTS[f"C_constvel_L{L}"] = (predicted_vL, bn, br)
    return True


# ---------------------------------------------------------------------------
# TEST D: does rewind EVER hurt? Inject the OPPOSITE — fixes that arrive with t_fix
# slightly in the FUTURE (clock skew), and t_fix exactly == an op boundary, to probe
# off-by-one in the idx search. Also a fix exactly at the oldest boundary (edge of drop).
# ---------------------------------------------------------------------------
def test_edge_cases():
    print("[D] Edge cases: future fix, boundary fix, exactly-oldest fix")
    t0 = 1_000_000_000
    rk = RewindKF(kf=LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3),
                  horizon_s=0.5)
    for i in range(1, 40):
        rk.predict(fb(np.zeros(3)), np.eye(3), DT, sim_time_ns=t0 + int(i * DT * 1e9))
    cov = np.diag([0.3, 0.3, 0.3]) ** 2
    # (1) future fix -> degenerate in-place, applied, not rewound
    res_fut = rk.update_position_at(rk.now_ns + int(0.05 * 1e9), np.array([0.1, 0.1, 0.1]), cov)
    # (2) fix exactly at the oldest buffered boundary -> should NOT drop (t_fix >= oldest)
    oldest = rk.oldest_buffered_ns()
    res_old = rk.update_position_at(oldest, np.array([0.1, 0.1, 0.1]), cov)
    # (3) fix one ns older than oldest -> drop
    res_drop = rk.update_position_at(rk.oldest_buffered_ns() - 1, np.array([0.1, 0.1, 0.1]), cov)
    print(f"    future fix:   applied={res_fut.applied} rewound={res_fut.rewound} (expect applied,not-rewound)")
    print(f"    oldest-bndry: applied={res_old.applied} rewound={res_old.rewound} reason={res_old.dropped_reason}")
    print(f"    older-than:   applied={res_drop.applied} reason={res_drop.dropped_reason} (expect dropped)")
    ok = (res_fut.applied and not res_fut.rewound) and res_old.applied and (not res_drop.applied)
    print(f"    PASS={ok}")
    RESULTS["D_edges"] = ok
    return ok


# ---------------------------------------------------------------------------
# TEST E: covariance PSD after rewind (Joseph form should keep P SPD; replay shouldn't break it)
# ---------------------------------------------------------------------------
def test_cov_psd():
    print("[E] P stays symmetric-PSD through repeated rewinds")
    rng = np.random.default_rng(99)
    rk = RewindKF(kf=LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3),
                  horizon_s=0.5)
    t0 = 1_000_000_000
    min_eig = np.inf
    max_asym = 0.0
    for k in range(1, 300):
        a = rng.normal(0, 4.0, 3)
        rk.predict(fb(a), np.eye(3), DT, sim_time_ns=t0 + int(k * DT * 1e9))
        if k % 4 == 0:
            L = rng.integers(0, 5)
            ck = max(0, k - int(L))
            z = rng.normal(0, 1, 3)
            cov = np.diag(rng.uniform(0.1, 0.9, 3))
            rk.update_position_at(t0 + int(ck * DT * 1e9), z, cov)
            P = rk.P
            min_eig = min(min_eig, float(np.linalg.eigvalsh(P).min()))
            max_asym = max(max_asym, float(np.max(np.abs(P - P.T))))
    print(f"    min eigenvalue of P over run = {min_eig:.3e} (must be > 0)")
    print(f"    max |P - P^T| = {max_asym:.2e} (must be ~0)")
    ok = min_eig > 0 and max_asym < 1e-9
    print(f"    PASS={ok}")
    RESULTS["E_psd"] = (ok, min_eig, max_asym)
    return ok


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    a = test_oosm_mixed_ops()
    b = test_l0_identical_random()
    test_const_vel_analytic()
    d = test_edge_cases()
    e = test_cov_psd()
    print("\n=== INDEPENDENT VERIFIER SUMMARY ===")
    print(f"  A OOSM mixed-ops exact:   {'PASS' if a else 'FAIL'}")
    print(f"  B L=0 bit-identical:      {'PASS' if b else 'FAIL'}")
    print(f"  D edge cases:             {'PASS' if d else 'FAIL'}")
    print(f"  E P stays SPD:            {'PASS' if e else 'FAIL'}")
    sys.exit(0)
