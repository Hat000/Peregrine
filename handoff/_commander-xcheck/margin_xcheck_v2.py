"""COMMANDER cross-check v2 — clean cold-velocity-PRIOR sweep on the straight g3->g4 binding leg.

v1 (margin_xcheck.py) had a CORNER ARTIFACT: a piecewise-linear multi-leg path has instantaneous
velocity-direction changes at each gate, and a constant gravity-comp accel_body across legs means
the KF never turns its velocity vector -> the cold arm lags catastrophically (4-5 m, vel_err ~11 m/s).
That is a sim artifact, NOT case-C physics (the real policy flies smooth turns the IMU measures).

v2 mirrors c1_gate_relative.py's faithful single-leg geometry (straight g3->g4, perfectly centered,
true in-plane offset = 0 at g4) but replaces c1's WARM assumption (v0 seeded to truth) with a COLD
VELOCITY PRIOR: v0 = uhat*V + N(0, sigma_v_prior), and NO velocity update during the window (case C
is position-only). This isolates the load-bearing swing variable: how the velocity prior entering
gate-4 sets the last-fix->crossing terminal extrapolation, hence the in-plane miss.

sigma_v_prior=0, att=0 should recover c1's ~0.139 m warm baseline. The case-C realistic prior is
bounded separately (attitude-bias phantom accel a_ph = g*sin(theta) ~ 0.24 m/s^2 over the coast).

Run: from repo root, .venv\\Scripts\\python.exe handoff/_commander-xcheck/margin_xcheck_v2.py
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
from scipy.spatial.transform import Rotation  # noqa: E402

SEED = 20260613
MARGIN_G4 = 0.155
BAR = 0.05
G = np.array([0.0, 0.0, 9.80665])
LAT_SIGMA = 0.265      # measured gate-relative per-fix lateral per-axis sigma (E,D), near band
RADIAL_SIGMA = 0.50    # along-LOS (N) PnP noise
FIX_HZ = 14.0
FIX_RANGE_M = 12.0
IMU_HZ = 90.0

# g3 / g4 opening centres in NED (fly_rl._GATE_POS_ZUP*[1,-1,-1]); matches c1.
G3_NED = np.array([-111.49374389648438,  5.099989891052246,  -23.208040833473206]) * np.array([1, -1, -1])
G4_NED = np.array([-135.49374389648438,  0.7999902367591858, -23.995653748512268]) * np.array([1, -1, -1])


def run(v_race, sigma_v_prior, att_deg, latency_ms, n_mc, seed0):
    dt = 1.0 / IMU_HZ
    fix_dt = 1.0 / FIX_HZ
    L_ns = int(latency_ms * 1e6)
    att = np.radians(att_deg)
    seg = G4_NED - G3_NED
    L_seg = float(np.linalg.norm(seg))
    uhat = seg / L_seg
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * v_race / 9.80665))
    R_wb_true = R_world_from_body(0.0, pitch, yaw)
    accel_body = R_wb_true.T @ (-G)            # gravity-comp, const-vel leg
    n_steps = int((L_seg / v_race) / dt)

    ip, ve = [], []
    for s in range(n_mc):
        r = np.random.default_rng(seed0 + 41 * s)
        theta = r.normal(0, att, 3) if att > 0 else np.zeros(3)
        R_wb_est = Rotation.from_rotvec(theta).as_matrix() @ R_wb_true
        p0 = G3_NED.copy()
        v0 = uhat * v_race + r.normal(0, sigma_v_prior, 3)   # COLD velocity prior
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5,
                                 vel_std=max(sigma_v_prior, 0.3))
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t = 0.0
        t_ns = 0
        next_fix = 0.0
        for k in range(n_steps):
            t += dt
            t_ns += int(dt * 1e9)
            rk.predict(accel_body, R_wb_est, dt, t_ns)             # NO velocity update (case C)
            p_true = p0 + uhat * v_race * t
            rng = float(np.linalg.norm(G4_NED - p_true))
            if t >= next_fix and rng < FIX_RANGE_M:
                next_fix += fix_dt
                # measurement reflects the CAPTURE-time truth (t - L), applied now via OOSM rewind.
                # (Generating z from the CURRENT truth but stamping it at t-L was a v2a bug: it fed a
                # too-fresh fix to a too-old timestamp -> a v*L along-track error that the 14deg g3->g4
                # heading projected onto E, falsely blowing up the L=115ms arm. RewindKF was correct.)
                t_cap = t - latency_ms / 1000.0
                p_cap = p0 + uhat * v_race * max(t_cap, 0.0)
                z = p_cap.copy()
                z[0] += r.normal(0, RADIAL_SIGMA)
                z[1] += r.normal(0, LAT_SIGMA)
                z[2] += r.normal(0, LAT_SIGMA)
                cov = np.diag([RADIAL_SIGMA**2, LAT_SIGMA**2, LAT_SIGMA**2])
                rk.update_position_at(t_ns - L_ns, z, cov)
        p_true_final = p0 + uhat * v_race * (n_steps * dt)
        err = rk.position - p_true_final
        ip.append(float(np.hypot(err[1], err[2])))
        ve.append(float(np.linalg.norm(rk.velocity - uhat * v_race)))
    ip = np.array(ip)
    return dict(v=v_race, sigma_v=sigma_v_prior, att=att_deg, L=latency_ms, n=len(ip),
                ip_rms=float(np.sqrt(np.mean(ip**2))), ip_p50=float(np.percentile(ip, 50)),
                ip_p90=float(np.percentile(ip, 90)), ip_p95=float(np.percentile(ip, 95)),
                vel_err=float(np.mean(ve)),
                rms_clears=bool(np.sqrt(np.mean(ip**2)) < MARGIN_G4),
                p90_clears=bool(np.percentile(ip, 90) < MARGIN_G4))


def main():
    N = 600
    print("=" * 104)
    print("COMMANDER cross-check v2: gate-4 in-plane error, straight g3->g4, COLD velocity-prior sweep")
    print(f"  margin={MARGIN_G4} m  bar={BAR} m  lat_sigma={LAT_SIGMA}  fix {FIX_HZ}Hz<{FIX_RANGE_M}m  N={N}")
    print("=" * 104)
    print(f"{'v':>4} {'sig_v':>5} {'att':>4} {'L_ms':>5} {'ip_rms':>7} {'ip_p50':>7} {'ip_p90':>7} {'ip_p95':>7} {'vel_e':>6} {'rms<155?':>8} {'p90<155?':>8}")
    rows = []
    for v in (37.0, 30.0, 25.0):
        for sig in (0.0, 0.3, 0.6, 1.0):
            for att in (0.0, 1.4):
                for L in (15.0, 115.0):
                    if att == 0.0 and L == 15.0:
                        pass
                    # prune: full grid only at 37; lighter at 30/25
                    if v != 37.0 and (att == 1.4 or L == 15.0):
                        continue
                    res = run(v, sig, att, L, N, SEED + int(v) + int(sig * 10) + int(att * 10) + int(L))
                    rows.append(res)
                    print(f"{v:>4.0f} {sig:>5.1f} {att:>4.1f} {L:>5.0f} "
                          f"{res['ip_rms']:>7.3f} {res['ip_p50']:>7.3f} {res['ip_p90']:>7.3f} {res['ip_p95']:>7.3f} "
                          f"{res['vel_err']:>6.3f} {str(res['rms_clears']):>8} {str(res['p90_clears']):>8}")
    out = Path(__file__).resolve().parent / "margin_xcheck_v2_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
