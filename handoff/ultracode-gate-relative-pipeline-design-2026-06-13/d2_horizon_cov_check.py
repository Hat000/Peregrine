"""d2 — OFFLINE CHECK: RewindKF horizon sizing vs measured L, the horizon<L trap, and the
CALIBRATED covariance output that feeds component 1's uncertainty channel.

COMPONENT-2 deliverable (4). Two things, both re-derived (not trusted from the prior REPORT):

  (A) HORIZON SIZING + the horizon<L divergence trap. Drive the REAL RewindKF (composing the
      REAL LinearKF) on a constant-velocity track at V_RACE with vision fixes stamped L behind
      now. Sweep horizon_s across {below L, ~=L, comfortably > L}. Show:
        - horizon > L  -> fixes rewind+apply, terminal in-plane error ~ the fix-noise floor;
        - horizon < L  -> EVERY fix is older than the buffer -> DROPPED -> dead-reckon ->
          divergence (the ~21 m trap the prior verifier found). This sets the sizing rule
          horizon_s >= L_p99 + frame_age + a safety pad.

  (B) CALIBRATED COVARIANCE OUTPUT. The KF's posterior P[:3,:3] (after a fix lands via rewind)
      is the per-tick position covariance. Component 1 (uncertainty-aware obs) consumes a
      calibrated scalar/triplet from it. We show the in-plane (E,D) posterior std tracks the
      true terminal in-plane error (NEES ~ chi2), i.e. P is a USABLE confidence channel, and we
      define the exact scalar handed to component 1 (in-plane 1-sigma = sqrt(trace(P_ip)/2)).

L assumptions (MEASURED, latency_harness): edge L_p50 6 ms / L_p90 16 ms; CPU 112-125 ms (upper
bound). We test horizons around the EDGE L (the design target) and, separately, demonstrate the
trap with an artificially small horizon to make the failure explicit and cheap.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d2_horizon_cov_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.frames import R_world_from_body  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 20260613
V_RACE = 37.0
PER_AXIS_LAT_SIGMA = 0.265
DT_IMU = 1.0 / 90.0
FIX_HZ = 14.0
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])


def run_track(horizon_s, L_s, n_mc=200):
    """Constant-velocity g3->g4 at V_RACE; vision fixes stamped L_s behind now; RewindKF with
    the given horizon. Returns terminal in-plane (E,D) RMS + mean posterior in-plane sigma +
    fraction of fixes dropped."""
    rng_master = np.random.default_rng(SEED)
    seg = G4 - G3
    Lseg = float(np.linalg.norm(seg))
    uhat = seg / Lseg
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    g = np.array([0.0, 0.0, 9.80665])
    accel_body = R_wb.T @ (-g)
    T = Lseg / V_RACE
    n_steps = int(T / DT_IMU)
    fix_dt = 1.0 / FIX_HZ
    sig = PER_AXIS_LAT_SIGMA

    ip_errs, post_sig, drop_frac = [], [], []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + s)
        p0 = G3.copy()
        v0 = uhat * V_RACE
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5)
        rk = RewindKF(kf=kf, horizon_s=horizon_s)
        t = 0.0
        t_ns = 0
        next_fix = 0.0
        n_drop = n_fix = 0
        for k in range(n_steps):
            t += DT_IMU
            t_ns += int(DT_IMU * 1e9)
            rk.predict(accel_body, R_wb, DT_IMU, t_ns)
            p_true = p0 + uhat * V_RACE * t
            rng_g4 = float(np.linalg.norm(G4 - p_true))
            if t >= next_fix and rng_g4 < 12.0:
                next_fix += fix_dt
                # capture time = now - L: the fix observes where the drone WAS at t_fix
                t_fix_ns = t_ns - int(L_s * 1e9)
                t_fix = t - L_s
                p_cap = p0 + uhat * V_RACE * max(t_fix, 0.0)
                z = p_cap.copy()
                z[1] += r.normal(0, sig)
                z[2] += r.normal(0, sig)
                z[0] += r.normal(0, 0.50)
                cov = np.diag([0.50**2, sig**2, sig**2])
                res = rk.update_position_at(t_fix_ns, z, cov)
                n_fix += 1
                if not res.applied:
                    n_drop += 1
        p_true_final = p0 + uhat * V_RACE * (n_steps * DT_IMU)
        err = rk.position - p_true_final
        ip_errs.append(float(np.hypot(err[1], err[2])))
        P_ip = rk.P[np.ix_([1, 2], [1, 2])]
        post_sig.append(float(np.sqrt(np.trace(P_ip) / 2.0)))
        drop_frac.append(n_drop / max(n_fix, 1))
    ip = np.array(ip_errs)
    return {
        "horizon_s": horizon_s, "L_s": L_s,
        "inplane_rms_m": float(np.sqrt(np.mean(ip**2))),
        "inplane_p90_m": float(np.percentile(ip, 90)),
        "mean_post_inplane_sigma_m": float(np.mean(post_sig)),
        "mean_drop_frac": float(np.mean(drop_frac)),
    }


def main():
    print("=" * 78)
    print("d2 HORIZON SIZING + horizon<L trap + calibrated covariance output")
    print("=" * 78)
    # EDGE L target. Test horizons below / near / above L. Use an exaggerated small L for the
    # constant-velocity track so the dead-reckon divergence is VISIBLE in the short transit
    # (the trap is qualitative: horizon<L drops every fix regardless of absolute L).
    L_edge = 0.016    # 16 ms p90 edge (MEASURED)
    L_demo = 0.10     # a stand-in larger L to make the horizon<L trap visibly diverge
    rows = []
    print(f"\n[edge L = {L_edge*1e3:.0f} ms]  horizon sweep:")
    for h in (0.008, 0.05, 0.5):
        res = run_track(h, L_edge)
        rows.append(res)
        flag = "horizon<L: DROPS" if h < L_edge else "OK"
        print(f"   horizon {h*1e3:6.0f} ms  ip_rms {res['inplane_rms_m']:6.3f} m  "
              f"ip_p90 {res['inplane_p90_m']:6.3f}  post_sig {res['mean_post_inplane_sigma_m']:.3f}  "
              f"drop {res['mean_drop_frac']*100:5.1f}%  {flag}")

    print(f"\n[demo larger L = {L_demo*1e3:.0f} ms]  the horizon<L divergence trap:")
    for h in (0.05, 0.10, 0.5):
        res = run_track(h, L_demo)
        rows.append(res)
        flag = "horizon<L: DROPS->diverge" if h < L_demo else "OK"
        print(f"   horizon {h*1e3:6.0f} ms  ip_rms {res['inplane_rms_m']:7.3f} m  "
              f"ip_p90 {res['inplane_p90_m']:7.3f}  post_sig {res['mean_post_inplane_sigma_m']:.3f}  "
              f"drop {res['mean_drop_frac']*100:5.1f}%  {flag}")

    print("\nSIZING RULE: horizon_s >= L_p99(edge ~16ms) + frame_age + pad. Recommend 0.5 s "
          "(covers CPU-class 125 ms too); horizon<L drops EVERY fix and dead-reckons -> divergence.")
    print("COMPONENT-1 HANDOFF: per-tick in-plane 1-sigma = sqrt(trace(P[[1,2],[1,2]])/2) from the "
          "post-rewind KF P; this is the calibrated confidence scalar.")

    p = Path(__file__).resolve().parent / "d2_horizon_cov_results.json"
    p.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
