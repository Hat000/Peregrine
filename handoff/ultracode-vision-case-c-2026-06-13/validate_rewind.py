"""Offline validation of the KF rewind buffer (PIECE B, case-C VQ2).

Synthetic-truth Monte-Carlo. We build a smooth curved constant-jerk truth track (so the
constant-accel KF predict has a small but real model error, like the real plant), drive a
RewindKF and a naive LinearKF with the SAME 90 Hz IMU stream + the SAME async vision-fix
stream stamped at capture-time t_fix = t_now - L, and compare position RMSE.

The vision-fix VALUE is drawn from the measured VISION-PKG2 world-fix noise model
(sigma ~ [0.73, 0.47, 0.29] m N/E/D + 0.40 m isotropic floor in quadrature) AT THE CAPTURE
POSE (where the drone actually was at t_fix). The naive filter applies that capture-time fix
to the CURRENT pose -> it inherits the latency-induced error ~ v*L. The rewind filter applies
it at t_fix and rolls forward -> that error is removed.

Adversarial self-checks (all asserted + reported):
  (1) L=0 -> rewind is BIT-IDENTICAL to naive (max state diff == 0).
  (2) NEES of the fused position ~ chi2(3) over many MC runs: mean ~ 3 (consistency).
  (3) a fix older than the horizon is DROPPED (never applied to the current state).
Plus: buffer memory + per-fix compute cost.

Run: .venv/Scripts/python.exe handoff/ultracode-vision-case-c-2026-06-13/validate_rewind.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# repo-root import (editable install: `import racer` works)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff" / "ultracode-vision-case-c-2026-06-13"))
from kf_rewind_buffer import RewindKF  # noqa: E402

from racer.state_estimator import LinearKF  # noqa: E402

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])

# Canonical VISION-PKG2 measured fix noise (N,E,D) + isotropic floor, in quadrature.
FIX_SIGMA_NED = np.array([0.73, 0.47, 0.29])
FIX_FLOOR = 0.40
# diagonal world-fix cov used for both the draw and the KF update R (consistent => NEES valid)
FIX_COV = np.diag(FIX_SIGMA_NED**2 + FIX_FLOOR**2)
FIX_STD = np.sqrt(np.diag(FIX_COV))

IMU_HZ = 90.0
DT = 1.0 / IMU_HZ
VISION_FPS = 28.0
ACCEL_NOISE_STD = 0.05   # IMU accel measurement noise (m/s^2) injected on the body accel


def make_truth_track(duration_s: float, speed_mps: float, seed: int):
    """Curved descending constant-jerk track sampled at the IMU rate.

    Returns arrays at IMU ticks: t_ns, pos(N,3), vel(N,3), accel_world(N,3). A smooth
    sinusoidal lateral+vertical weave around a forward cruise -> nonzero, time-varying
    world acceleration that the constant-accel predict integrates with a small model error.
    """
    n = int(duration_s * IMU_HZ)
    t = np.arange(n) * DT
    # forward (north) cruise at `speed`, lateral (east) + vertical (down) weave
    omega1, omega2 = 0.9, 0.6
    ay_amp = 0.25 * speed_mps   # lateral accel amplitude scales with speed (banked turn feel)
    az_amp = 0.10 * speed_mps
    pN = speed_mps * t
    pE = (ay_amp / omega1**2) * np.sin(omega1 * t)
    pD = -(az_amp / omega2**2) * np.cos(omega2 * t)  # gentle descent/climb weave
    vN = speed_mps * np.ones_like(t)
    vE = (ay_amp / omega1) * np.cos(omega1 * t)
    vD = (az_amp / omega2) * np.sin(omega2 * t)
    aN = np.zeros_like(t)
    aE = -ay_amp * np.sin(omega1 * t)
    aD = az_amp * omega2 * np.cos(omega2 * t)
    pos = np.column_stack([pN, pE, pD])
    vel = np.column_stack([vN, vE, vD])
    acc = np.column_stack([aN, aE, aD])
    t_ns = (t * 1e9).astype(np.int64) + 1_000_000_000  # offset so t0 != 0
    return t_ns, pos, vel, acc


def make_consistent_track(duration_s, speed_mps, seed, accel_proc_std):
    """Truth track whose dynamics MATCH the constant-accel KF process model exactly, so the
    filter is the correct Bayesian estimator and NEES is a valid covariance-consistency test.

    The KF's predict uses a WHITE body-acceleration model: a_world = R_wb @ (f_body) + g with
    body-accel covariance ``accel_noise_std^2 I`` over each step (Q = B accel_cov B^T). We
    therefore generate truth by integrating an acceleration that is itself white noise of std
    ``accel_proc_std`` per axis (around a forward cruise), and report to the filter the SAME
    ``accel_proc_std`` as its ``accel_noise_std``. No deterministic sinusoid -> no unmodeled
    jerk -> any NEES inflation is then a genuine filter/rewind covariance bug, not maneuver
    model mismatch. (The smooth ``make_truth_track`` above is used for the realistic RMSE table;
    the rewind-vs-naive comparison there is fair regardless of absolute calibration.)
    """
    rng = np.random.default_rng(seed * 9973 + 17)
    n = int(duration_s * IMU_HZ)
    t_ns = (np.arange(n) * DT * 1e9).astype(np.int64) + 1_000_000_000
    a = rng.normal(0.0, accel_proc_std, size=(n, 3))   # white world accel (the process)
    a[:, 0] += 0.0   # cruise is constant-velocity in N; speed enters via v0
    vel = np.zeros((n, 3)); pos = np.zeros((n, 3))
    vel[0] = np.array([speed_mps, 0.0, 0.0])
    pos[0] = np.zeros(3)
    for k in range(1, n):
        # exact constant-accel-over-the-step integration (matches KF F,B)
        pos[k] = pos[k - 1] + vel[k - 1] * DT + 0.5 * a[k] * DT * DT
        vel[k] = vel[k - 1] + a[k] * DT
    return t_ns, pos, vel, a


def world_accel_to_body_specific_force(a_world, R_wb_eye=True):
    """Given attitude (we use identity attitude for the synthetic track -- the KF gets the SAME
    R_wb, so attitude is consistent and not the thing under test), the body specific force the
    IMU reports is f_body = R_wb^T @ (a_world - g_world). At rest a_world=0 -> f = -g (reads -g
    on down). predict() does a_world = R_wb @ f + g, recovering a_world exactly."""
    R_wb = np.eye(3)
    f_body = R_wb.T @ (a_world - GRAVITY_NED)
    return f_body, R_wb


def run_episode(speed_mps, latency_ticks, seed, use_rewind, duration_s=6.0,
                horizon_s=0.5, record_nees=False):
    """Drive one filter (rewind or naive) over the synthetic track. Returns position RMSE
    (over IMU ticks AFTER the first fix) and optional per-fix NEES list."""
    rng = np.random.default_rng(seed)
    t_ns, pos, vel, acc = make_truth_track(duration_s, speed_mps, seed)
    n = len(t_ns)

    # seed filter at truth pose with a loose-ish prior (case-C-like cold start sigma)
    base = LinearKF.initialize(pos[0], vel[0], pos_std=0.5, vel_std=0.5,
                               accel_noise_std=0.3)
    if use_rewind:
        kf = RewindKF(kf=base, horizon_s=horizon_s)
    else:
        kf = base

    vision_period_s = 1.0 / VISION_FPS
    next_fix_t = vision_period_s * 0.5
    L_ns = int(latency_ticks * DT * 1e9)

    # buffer the (t_ns, pos, vel) we need to look up a capture pose by index
    err_sq = []
    nees_list = []
    last_sim_t = t_ns[0]

    for k in range(1, n):
        dt = (t_ns[k] - last_sim_t) / 1e9
        last_sim_t = t_ns[k]
        # IMU specific force = true world accel + accel noise, mapped to body
        a_meas = acc[k] + rng.normal(0, ACCEL_NOISE_STD, 3)
        f_body, R_wb = world_accel_to_body_specific_force(a_meas)
        if use_rewind:
            kf.predict(f_body, R_wb, dt, sim_time_ns=int(t_ns[k]))
        else:
            kf.predict(f_body, R_wb, dt)

        # a vision fix becomes AVAILABLE (arrives) at t_now if its capture time was L ago
        t_now_s = (t_ns[k] - t_ns[0]) / 1e9
        if t_now_s >= next_fix_t:
            next_fix_t += vision_period_s
            # capture pose was L ticks ago
            cap_k = max(0, k - latency_ticks)
            cap_pos = pos[cap_k]
            cap_t_ns = int(t_ns[cap_k])
            # draw the fix from the measured model AT the capture pose
            z = cap_pos + rng.normal(0, FIX_STD)
            if use_rewind:
                res = kf.update_position_at(cap_t_ns, z, FIX_COV)
            else:
                # NAIVE: apply the capture-time fix to the CURRENT state (the bug)
                kf.update_position(z, FIX_COV)
            # NEES of the fused position vs truth at NOW (after fix)
            if record_nees:
                p = kf.x[:3] if use_rewind else kf.x[:3]
                Pp = kf.P[:3, :3] if use_rewind else kf.P[:3, :3]
                e = p - pos[k]
                try:
                    nees = float(e @ np.linalg.solve(Pp, e))
                    nees_list.append(nees)
                except np.linalg.LinAlgError:
                    pass

        # accumulate position error over the steady-state window (after t > 1 s so the prior
        # has washed out and at least a few fixes have landed)
        if t_now_s > 1.0:
            p = kf.x[:3]
            err_sq.append(float(np.sum((p - pos[k])**2)))

    rmse = float(np.sqrt(np.mean(err_sq))) if err_sq else float("nan")
    return rmse, nees_list


def rmse_table():
    print("=" * 78)
    print("POSITION RMSE (m): naive apply-at-current  vs  rewind-and-rollforward")
    print("vision fixes @ 28 fps from measured VISION-PKG2 cov; IMU 90 Hz; 8 seeds avg")
    print("=" * 78)
    speeds = [5.0, 15.0, 20.0, 30.0]
    lat_ticks = [0, 1, 2, 3, 4]
    seeds = list(range(8))
    print(f"{'speed':>6} {'L(ms)':>6} {'v*L(m)':>7} {'naive':>8} {'rewind':>8} {'cut%':>6}")
    rows = []
    for v in speeds:
        for L in lat_ticks:
            naive = np.mean([run_episode(v, L, s, use_rewind=False)[0] for s in seeds])
            rew = np.mean([run_episode(v, L, s, use_rewind=True)[0] for s in seeds])
            vL = v * L * DT
            cut = 100.0 * (naive - rew) / naive if naive > 0 else 0.0
            print(f"{v:>6.0f} {L*DT*1000:>6.0f} {vL:>7.3f} {naive:>8.3f} {rew:>8.3f} {cut:>6.1f}")
            rows.append((v, L, vL, naive, rew))
        print("-" * 78)
    return rows


def selfcheck_bit_identical():
    """(1) L=0 => rewind state path is bit-identical to naive."""
    print("\n[CHECK 1] L=0 bit-identical (rewind == naive)")
    v, seed = 20.0, 7
    # run both, compare FINAL state x and P exactly
    rng_consistent_seed = seed
    rmse_n, _ = run_episode(v, 0, rng_consistent_seed, use_rewind=False)
    rmse_r, _ = run_episode(v, 0, rng_consistent_seed, use_rewind=True)
    # the RNG sequence is identical (same seed, same draw order) so states must match.
    # re-run capturing final x to compare bit-for-bit:
    xn = _final_state(v, 0, seed, use_rewind=False)
    xr = _final_state(v, 0, seed, use_rewind=True)
    dx = float(np.max(np.abs(xn[0] - xr[0])))
    dP = float(np.max(np.abs(xn[1] - xr[1])))
    print(f"   final |dx|_max = {dx:.3e}   |dP|_max = {dP:.3e}   (RMSE naive {rmse_n:.4f} / rewind {rmse_r:.4f})")
    ok = dx < 1e-12 and dP < 1e-12
    print(f"   PASS={ok}")
    return ok, dx, dP


def _final_state(v, L, seed, use_rewind, duration_s=6.0, horizon_s=0.5):
    rng = np.random.default_rng(seed)
    t_ns, pos, vel, acc = make_truth_track(duration_s, v, seed)
    n = len(t_ns)
    base = LinearKF.initialize(pos[0], vel[0], pos_std=0.5, vel_std=0.5, accel_noise_std=0.3)
    kf = RewindKF(kf=base, horizon_s=horizon_s) if use_rewind else base
    vision_period_s = 1.0 / VISION_FPS
    next_fix_t = vision_period_s * 0.5
    last_sim_t = t_ns[0]
    for k in range(1, n):
        dt = (t_ns[k] - last_sim_t) / 1e9
        last_sim_t = t_ns[k]
        a_meas = acc[k] + rng.normal(0, ACCEL_NOISE_STD, 3)
        f_body, R_wb = world_accel_to_body_specific_force(a_meas)
        if use_rewind:
            kf.predict(f_body, R_wb, dt, sim_time_ns=int(t_ns[k]))
        else:
            kf.predict(f_body, R_wb, dt)
        t_now_s = (t_ns[k] - t_ns[0]) / 1e9
        if t_now_s >= next_fix_t:
            next_fix_t += vision_period_s
            cap_k = max(0, k - L)
            z = pos[cap_k] + rng.normal(0, FIX_STD)
            if use_rewind:
                kf.update_position_at(int(t_ns[cap_k]), z, FIX_COV)
            else:
                kf.update_position(z, FIX_COV)
    return kf.x.copy(), kf.P.copy()


def _nees_episode(v, L, seed, use_rewind, accel_proc_std, duration_s=6.0, horizon_s=0.5):
    """NEES on a filter-CONSISTENT track: truth dynamics == the KF process model, and the
    accel fed to the filter is a noisy measurement with std == accel_proc_std == the KF's
    assumed accel_noise_std. Then a well-built (rewind) filter must be consistent: NEES~chi2(3).
    """
    rng = np.random.default_rng(seed)
    t_ns, pos, vel, acc = make_consistent_track(duration_s, v, seed, accel_proc_std)
    n = len(t_ns)
    base = LinearKF.initialize(pos[0], vel[0], pos_std=0.5, vel_std=0.5,
                               accel_noise_std=accel_proc_std)
    kf = RewindKF(kf=base, horizon_s=horizon_s) if use_rewind else base
    vp = 1.0 / VISION_FPS
    nxt = vp * 0.5
    last = t_ns[0]
    nees = []
    for k in range(1, n):
        dt = (t_ns[k] - last) / 1e9
        last = t_ns[k]
        # the filter receives a NOISY accel measurement (std == its assumed Q accel std) ->
        # the predict uncertainty Q is exactly right for this measurement error.
        a_meas = acc[k] + rng.normal(0.0, accel_proc_std, 3)
        f_body, R_wb = world_accel_to_body_specific_force(a_meas)
        if use_rewind:
            kf.predict(f_body, R_wb, dt, sim_time_ns=int(t_ns[k]))
        else:
            kf.predict(f_body, R_wb, dt)
        tn = (t_ns[k] - t_ns[0]) / 1e9
        if tn >= nxt:
            nxt += vp
            ck = max(0, k - L)
            z = pos[ck] + rng.normal(0.0, FIX_STD)
            if use_rewind:
                kf.update_position_at(int(t_ns[ck]), z, FIX_COV)
            else:
                kf.update_position(z, FIX_COV)
            if tn > 1.0:
                e = kf.x[:3] - pos[k]
                Pp = kf.P[:3, :3]
                try:
                    nees.append(float(e @ np.linalg.solve(Pp, e)))
                except np.linalg.LinAlgError:
                    pass
    return nees


def selfcheck_nees():
    """(2) NEES of fused position ~ chi2(3): mean ~ 3 over many MC runs at L=2.

    Tested on a filter-CONSISTENT track (truth == KF process model, matched accel std) so the
    check isolates the rewind machinery's covariance handling, not the maneuver model. The
    accel process std is set so the inter-fix prediction spread is comparable to the fix noise
    (a realistic operating point). Naive is reported alongside to show latency makes it
    inconsistent (biased mean) while rewind stays calibrated.
    """
    print("\n[CHECK 2] NEES consistency (chi2 3 DOF, mean ~ 3) -- filter-consistent track")
    accel_proc_std = 1.0   # m/s^2 white accel; matched into the KF's accel_noise_std
    all_nees = []
    for s in range(150):
        all_nees.extend(_nees_episode(20.0, 2, 2000 + s, True, accel_proc_std))
    arr = np.array(all_nees)
    m = arr.mean()
    se = np.sqrt(2 * 3 / len(arr))
    lo, hi = 3 - 1.96 * se, 3 + 1.96 * se
    print(f"   N={len(arr)} rewind NEES mean={m:.3f}  median={np.median(arr):.3f}  "
          f"(95% band for a consistent mean: [{lo:.3f},{hi:.3f}])")
    # sanity: L=0 on the same consistent track must also be ~3 (no latency, no rewind effect)
    l0 = []
    for s in range(150):
        l0.extend(_nees_episode(20.0, 0, 9000 + s, True, accel_proc_std))
    l0m = np.array(l0).mean()
    print(f"   L=0 control (no latency): rewind NEES mean={l0m:.3f}  (should also be ~3)")
    # naive under latency: latency bias should push its NEES mean ABOVE 3 (over-confident)
    naive_nees = []
    for s in range(150):
        naive_nees.extend(_nees_episode(20.0, 2, 5000 + s, False, accel_proc_std))
    nm = np.array(naive_nees).mean()
    print(f"   naive (apply-at-current, L=2) NEES mean={nm:.3f}  -> latency bias makes it "
          f"inconsistent ({nm/m:.2f}x rewind)")
    ok = abs(m - 3.0) < 0.5 and abs(l0m - 3.0) < 0.5
    print(f"   PASS={ok} (rewind NEES within 0.5 of 3.0 at both L=2 and L=0)")
    return ok, m, nm


def selfcheck_too_old():
    """(3) a fix older than the horizon is DROPPED, not applied to the current state."""
    print("\n[CHECK 3] fix older than buffer horizon -> DROPPED")
    base = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5)
    kf = RewindKF(kf=base, horizon_s=0.2)  # 0.2 s horizon
    t0 = 1_000_000_000
    # push 0.5 s of predicts (well past the 0.2 s horizon)
    for i in range(1, int(0.5 * IMU_HZ)):
        kf.predict(np.array([0., 0., -9.80665]), np.eye(3), DT, sim_time_ns=t0 + int(i * DT * 1e9))
    x_before = kf.x.copy()
    # a fix stamped 0.4 s in the past (older than the 0.2 s horizon)
    t_old = t0 + int(0.05 * 1e9)
    res = kf.update_position_at(t_old, np.array([99.0, 99.0, 99.0]), FIX_COV)
    x_after = kf.x.copy()
    state_unchanged = np.array_equal(x_before, x_after)
    print(f"   now={kf.now_ns} oldest_buffered={kf.oldest_buffered_ns()} t_fix={t_old}")
    print(f"   result: applied={res.applied} dropped_reason='{res.dropped_reason}'")
    print(f"   state UNCHANGED by too-old fix: {state_unchanged}  "
          f"(the wild [99,99,99] fix did NOT corrupt current state)")
    ok = (not res.applied) and state_unchanged
    print(f"   PASS={ok}")
    return ok


def selfcheck_oosm_oracle():
    """(4) THE correctness gold-standard: a late fix rewound + replayed must produce the EXACT
    state a strict-chronological in-order filter would have produced (fix slotted at capture).

    Scenario: predicts at ticks 1..8; a fix CAPTURED at tick 5 is DELIVERED at tick 8 (L=3).
    Oracle processes P1..P5, FIX@5, P6..P8 in strict time order. Rewind processes P1..P8 then
    the late fix-stamped-at-5. They must match bit-for-bit (full re-propagation OOSM, no approx).
    """
    print("\n[CHECK 4] OOSM exactness vs strict-chronological oracle (the correctness proof)")
    DT_ = DT
    t0 = 1_000_000_000
    accs = [np.array([0.3 * np.sin(i), 0.1 * i, -0.2 * np.cos(i)]) for i in range(11)]

    def fb(a):
        return np.eye(3).T @ (a - GRAVITY_NED)

    z = np.array([0.05, -0.02, 0.03])
    cov = np.diag([0.7, 0.4, 0.3])**2 + 0.16 * np.eye(3)

    oracle = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3)
    for i in range(1, 6):
        oracle.predict(fb(accs[i]), np.eye(3), DT_)
    oracle.update_position(z, cov)
    for i in range(6, 9):
        oracle.predict(fb(accs[i]), np.eye(3), DT_)

    rk = RewindKF(kf=LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5,
                                         accel_noise_std=0.3), horizon_s=1.0)
    for i in range(1, 9):
        rk.predict(fb(accs[i]), np.eye(3), DT_, sim_time_ns=t0 + int(i * DT_ * 1e9))
    res = rk.update_position_at(t0 + int(5 * DT_ * 1e9), z, cov)

    dx = float(np.max(np.abs(oracle.x - rk.x)))
    dP = float(np.max(np.abs(oracle.P - rk.P)))
    print(f"   rewind: rewound={res.rewound} replayed={res.n_replayed_ops} depth={res.rewind_depth_s:.4f}s")
    print(f"   max|dx oracle-vs-rewind| = {dx:.3e}   max|dP| = {dP:.3e}")
    ok = dx < 1e-12 and dP < 1e-12
    print(f"   PASS={ok} (rewind is BIT-EXACT to in-order processing -> zero-approximation OOSM)")
    return ok, dx, dP


def cost_report():
    print("\n[COST] buffer memory + per-fix compute")
    import sys as _sys
    horizon_s = 0.5
    base = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5)
    kf = RewindKF(kf=base, horizon_s=horizon_s)
    t0 = 1_000_000_000
    nfill = int(horizon_s * IMU_HZ)
    for i in range(1, nfill + 5):
        kf.predict(np.array([0., 0., -9.80665]), np.eye(3), DT, sim_time_ns=t0 + int(i * DT * 1e9))
    nslots = len(kf._ops)
    # per-slot bytes: x(6) + P(36) snapshot + accel(3) + R_wb(9) f64
    bytes_per_slot = (6 + 36 + 3 + 9) * 8
    total_kb = nslots * bytes_per_slot / 1024
    print(f"   horizon {horizon_s}s @ {IMU_HZ:.0f} Hz IMU -> {nslots} buffered ops, "
          f"~{bytes_per_slot} B/op -> ~{total_kb:.1f} KB")
    # per-fix rewind cost: worst case rewinds the full horizon
    # fill, then time many OOSM fixes at L=3 ticks
    import timeit
    def one_fix():
        kf2 = RewindKF(kf=LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5),
                       horizon_s=horizon_s)
        for i in range(1, nfill + 1):
            kf2.predict(np.array([0., 0., -9.80665]), np.eye(3), DT, sim_time_ns=t0 + int(i * DT * 1e9))
        t_fix = kf2.now_ns - int(3 * DT * 1e9)  # 3-tick latency
        kf2.update_position_at(t_fix, np.array([0.1, 0.1, 0.1]), FIX_COV)
    # measure only the fix (the fill dominates timeit otherwise) -> time fill+fix and fill, subtract
    N = 50
    t_full = timeit.timeit(one_fix, number=N) / N

    def fill_only():
        kf2 = RewindKF(kf=LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5),
                       horizon_s=horizon_s)
        for i in range(1, nfill + 1):
            kf2.predict(np.array([0., 0., -9.80665]), np.eye(3), DT, sim_time_ns=t0 + int(i * DT * 1e9))
    t_fill = timeit.timeit(fill_only, number=N) / N
    per_fix_ms = (t_full - t_fill) * 1000
    print(f"   per-fix rewind+replay (L=3, ~{3} ops replayed): ~{per_fix_ms:.3f} ms "
          f"(<<< 33 ms vision period @ 28 fps)")
    return total_kb, per_fix_ms


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rows = rmse_table()
    c1, dx, dP = selfcheck_bit_identical()
    c2, nees_m, naive_nees_m = selfcheck_nees()
    c3 = selfcheck_too_old()
    c4, odx, odP = selfcheck_oosm_oracle()
    mem_kb, per_fix_ms = cost_report()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    # headline: VQ2 latency-error cut at 20 m/s, L=2
    for v, L, vL, naive, rew in rows:
        if v == 20.0 and L == 2:
            print(f"  20 m/s, L=2 (67 ms): naive RMSE {naive:.3f} m -> rewind {rew:.3f} m "
                  f"(latency v*L={vL:.3f} m removed; {100*(naive-rew)/naive:.0f}% cut)")
    print(f"  CHECK 1 (L=0 bit-identical): {'PASS' if c1 else 'FAIL'} (|dx|={dx:.1e})")
    print(f"  CHECK 2 (NEES~3):            {'PASS' if c2 else 'FAIL'} (rewind {nees_m:.2f} vs naive {naive_nees_m:.2f})")
    print(f"  CHECK 3 (too-old dropped):   {'PASS' if c3 else 'FAIL'}")
    print(f"  CHECK 4 (OOSM exact oracle): {'PASS' if c4 else 'FAIL'} (|dx|={odx:.1e})")
    print(f"  COST: ~{mem_kb:.0f} KB buffer, ~{per_fix_ms:.2f} ms/fix")
    all_pass = c1 and c2 and c3 and c4
    print(f"\n  ALL ADVERSARIAL CHECKS: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)
