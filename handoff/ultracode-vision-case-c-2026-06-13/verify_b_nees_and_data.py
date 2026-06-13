"""Adversarial scrutiny of the NEES-consistency claim and the empirical fix-noise model.

1. NEES: what is chi2(3) mean/median? Is rewind 2.89/2.23 actually consistent or
   slightly UNDER-confident? Run a larger, independent NEES MC and bootstrap the CI.
2. Empirical fix error: load the perception-char gate-correct rows, compute accepted-fix
   off_ned sigma (N,E,D) and compare to the canonical [0.73,0.47,0.29] + 0.40 floor the
   harness draws from. Is the harness fix-noise model representative of REAL data, so the
   RMSE *floor* (the residual after latency removal) is believable?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff" / "ultracode-vision-case-c-2026-06-13"))
from kf_rewind_buffer import RewindKF  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])
DT = 1.0 / 90.0
FIX_SIGMA_NED = np.array([0.73, 0.47, 0.29])
FIX_FLOOR = 0.40
FIX_COV = np.diag(FIX_SIGMA_NED**2 + FIX_FLOOR**2)
FIX_STD = np.sqrt(np.diag(FIX_COV))


def fb(a, R=None):
    R = np.eye(3) if R is None else R
    return R.T @ (a - GRAVITY_NED)


# ---- chi2(3) reference ----
def chi2_reference():
    d = stats.chi2(df=3)
    print("[chi2(3) reference] mean=3.000 median=%.3f  P(X<2.89)=%.3f  P(X<2.23)=%.3f"
          % (d.median(), d.cdf(2.89), d.cdf(2.23)))
    # An UNbiased estimator's NEES mean is exactly 3 and median is 2.366. A mean of 2.89 and
    # median 2.23 would indicate MILD under-confidence (conservative cov), not over-confidence.
    return d.median()


# ---- independent NEES MC (consistent track), larger N, bootstrap CI on the MEAN ----
def make_consistent_track(n, v, seed, acc_std):
    rng = np.random.default_rng(seed * 7919 + 3)
    t_ns = (np.arange(n) * DT * 1e9).astype(np.int64) + 1_000_000_000
    a = rng.normal(0, acc_std, (n, 3))
    vel = np.zeros((n, 3)); pos = np.zeros((n, 3)); vel[0] = [v, 0, 0]
    for k in range(1, n):
        pos[k] = pos[k-1] + vel[k-1]*DT + 0.5*a[k]*DT*DT
        vel[k] = vel[k-1] + a[k]*DT
    return t_ns, pos, vel, a


def nees_mc(L, n_eps=300, acc_std=1.0):
    nees = []
    for s in range(n_eps):
        rng = np.random.default_rng(40000 + s)
        n = int(6.0 * 90)
        t_ns, pos, vel, a = make_consistent_track(n, 20.0, 40000 + s, acc_std)
        rk = RewindKF(kf=LinearKF.initialize(pos[0], vel[0], pos_std=0.5, vel_std=0.5, accel_noise_std=acc_std),
                      horizon_s=0.5)
        nxt = 0.5/28.0; last = t_ns[0]
        for k in range(1, n):
            dt = (t_ns[k]-last)/1e9; last = t_ns[k]
            am = a[k] + rng.normal(0, acc_std, 3)
            rk.predict(fb(am), np.eye(3), dt, sim_time_ns=int(t_ns[k]))
            tn = (t_ns[k]-t_ns[0])/1e9
            if tn >= nxt:
                nxt += 1.0/28.0
                ck = max(0, k-L)
                z = pos[ck] + rng.normal(0, FIX_STD)
                rk.update_position_at(int(t_ns[ck]), z, FIX_COV)
                if tn > 1.0:
                    e = rk.x[:3]-pos[k]
                    try:
                        nees.append(float(e @ np.linalg.solve(rk.P[:3,:3], e)))
                    except np.linalg.LinAlgError:
                        pass
    arr = np.array(nees)
    # bootstrap CI on the mean
    rng = np.random.default_rng(0)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(400)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"[NEES indep, L={L}] N={len(arr)} mean={arr.mean():.3f} median={np.median(arr):.3f} "
          f"bootstrap95%CI=[{lo:.3f},{hi:.3f}]")
    return arr.mean(), np.median(arr)


# ---- empirical fix-error from real perception-char data ----
def empirical_fix_error():
    base = Path(__file__).resolve().parents[2] / "handoff" / "perception-char-2026-06-08"
    all_rows = []
    for g in range(6):
        with open(base / f"characterize_g{g}.json") as f:
            d = json.load(f)
        all_rows.extend(d["rows"])
    rows = [r for r in all_rows if r.get("associated") and r.get("n_corners", 0) >= 1]
    # "gate-correct" / accepted: pass the chi2_0.999=16.27 maha gate (what the live filter accepts)
    accepted = [r for r in rows if r.get("maha", 1e9) <= 16.27]
    print(f"[empirical] total associated rows={len(rows)} ; accepted (maha<=16.27)={len(accepted)}")
    off = np.array([r["off_ned"] for r in accepted])  # N,E,D world-fix error
    sigma = off.std(axis=0)
    mean = off.mean(axis=0)
    print(f"[empirical] accepted off_ned MEAN (bias) N/E/D = {mean.round(3)}")
    print(f"[empirical] accepted off_ned STD   (sigma) N/E/D = {sigma.round(3)}")
    print(f"[canonical model] sigma N/E/D = {FIX_SIGMA_NED.round(3)} ; "
          f"per-axis incl floor = {FIX_STD.round(3)}")
    # robust spread (MAD-based, less tail-sensitive) to see if N-sigma is tail-driven
    med = np.median(off, axis=0)
    mad = np.median(np.abs(off - med), axis=0) * 1.4826
    print(f"[empirical] accepted off_ned MEDIAN N/E/D = {med.round(3)} ; robust sigma (MAD) = {mad.round(3)}")
    # also restrict to race-window range (the report's '132 gate-correct rows, true_range 0.71-23.3 m')
    tr_all = np.array([r["true_range_m"] for r in accepted])
    win = (tr_all >= 0.5) & (tr_all <= 24.0)
    offw = off[win]
    print(f"[empirical] within true_range[0.5,24] m: n={win.sum()} "
          f"sigma N/E/D={offw.std(axis=0).round(3)} MAD={ (np.median(np.abs(offw-np.median(offw,axis=0)),axis=0)*1.4826).round(3) }")
    # range-dependence of DEPTH error (range_err_m = pose_range - true_range along ray)
    tr = np.array([r["true_range_m"] for r in accepted])
    re = np.array([r["range_err_m"] for r in accepted])
    # bin by range, report std of range_err per bin to see if depth noise grows w/ range
    print("[empirical] depth-error (range_err_m) vs true_range bins:")
    for lo, hi in [(0, 5), (5, 10), (10, 16), (16, 25)]:
        m = (tr >= lo) & (tr < hi)
        if m.sum() >= 3:
            print(f"    range [{lo:2d},{hi:2d}) m: n={m.sum():3d} "
                  f"depth_err mean={re[m].mean():+.3f} std={re[m].std():.3f}")
    return sigma, mean


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    chi2_reference()
    nees_mc(2)
    nees_mc(0)
    print()
    empirical_fix_error()
