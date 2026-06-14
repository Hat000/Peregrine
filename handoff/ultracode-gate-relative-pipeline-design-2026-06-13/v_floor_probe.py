"""v_floor_probe -- WHAT is the binding term? velocity prior, or the per-fix lateral NOISE FLOOR?

d3 attributes the cold failure to the velocity prior (claim 3). My straight-leg check shows p90
barely moves with velErr (0->0.21 m/s: 0.250->0.258 Linf). That points at the per-fix lateral
noise (0.265 m/axis), averaged over only ~N fixes in the g3->g4 window, as the real binding term.

Decisive probes (all straight g3->g4, perfectly centred truth, real LinearKF+RewindKF):
  A. PERFECT velocity (seed truth, vel_std tiny) + NO velocity updates, bias=0. If p90 is STILL
     >> 0.155, the floor is the noise, not the velocity prior.
  B. Vary the per-fix lateral sigma (0.10 .. 0.30) at perfect velocity. Find the sigma that
     CLEARS p90<0.155 -> the c1 claim was "per-fix lateral <= ~0.30 clears the margin (edge)".
  C. d3-style continuous warm (update_velocity(truth, 0.1^2) every tick): how much does that
     PHYSICALLY-UNAVAILABLE continuous truth injection buy vs a one-time perfect seed? This tests
     whether d3's "warm clears" is reachable even in principle.
  D. n-fixes leverage: number of accepted fixes in the window is the variance reducer. Count them.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/v_floor_probe.py
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

SEED = 99127
MARGIN = 0.155
RADIAL_SIGMA = 0.50
ACCEL_NOISE_STD = 0.3
IMU_DT = 1.0 / 90.0
FIX_DT = 1.0 / 14.0
LINEAR_DRAG = 0.21
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])


def leg(v_race, lat_sigma, vel_mode, accel_bias_mag=0.0, fix_window_m=12.0, n_mc=800):
    """vel_mode: 'perfect' (seed truth, tiny vel_std, no vel update),
                 'warm_cont' (d3-style: update_velocity(truth,0.1^2) EVERY tick),
                 'cold' (seed +-1.5, vel_std 1.5, no vel update)."""
    seg = G4 - G3; L = float(np.linalg.norm(seg)); uhat = seg / L
    T = L / v_race; n_steps = int(T / IMU_DT)
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(LINEAR_DRAG * v_race / GRAVITY_NED[2]))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    accel_body_nom = R_wb.T @ (-GRAVITY_NED)
    e1 = np.cross(uhat, np.array([0.0, 0.0, 1.0])); e1 /= np.linalg.norm(e1)
    e2 = np.cross(uhat, e1); e2 /= np.linalg.norm(e2)
    l2, linf, nfix = [], [], []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 17 * s + int(lat_sigma * 1000) * 7
                                  + int(accel_bias_mag * 1000) * 101
                                  + {"perfect": 1, "warm_cont": 2, "cold": 3}[vel_mode] * 5003)
        d = r.normal(0, 1, 3); d /= (np.linalg.norm(d) + 1e-12)
        abias = accel_bias_mag * d
        p0 = G3.copy()
        if vel_mode == "cold":
            dv = r.normal(0, 1, 2); dv /= (np.linalg.norm(dv) + 1e-12)
            v_init = uhat * v_race + 1.5 * (dv[0] * e1 + dv[1] * e2); vstd = 1.5
        else:
            v_init = uhat * v_race; vstd = 0.05
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v_init, pos_std=0.5, vel_std=vstd)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t = 0.0; t_ns = 0; next_fix = 0.0; nf = 0
        for k in range(n_steps):
            t += IMU_DT; t_ns += int(IMU_DT * 1e9)
            rk.predict(accel_body_nom + abias + r.normal(0, ACCEL_NOISE_STD, 3), R_wb, IMU_DT, t_ns)
            if vel_mode == "warm_cont":
                v_true = uhat * v_race
                rk.update_velocity(v_true, (0.1**2) * np.eye(3), sim_time_ns=t_ns)
            p_true = p0 + uhat * v_race * t
            if t >= next_fix and float(np.linalg.norm(G4 - p_true)) < fix_window_m:
                next_fix = t + FIX_DT; nf += 1
                nlat = (r.normal(0, lat_sigma) * e1 + r.normal(0, lat_sigma) * e2
                        + r.normal(0, RADIAL_SIGMA) * uhat)
                z = p_true + nlat
                cov = (lat_sigma**2) * (np.outer(e1, e1) + np.outer(e2, e2)) \
                    + (RADIAL_SIGMA**2) * np.outer(uhat, uhat)
                rk.update_position(z, cov, sim_time_ns=t_ns)
        err = rk.position - (p0 + uhat * v_race * (n_steps * IMU_DT))
        eE = float(err @ e1); eD = float(err @ e2)
        l2.append(float(np.hypot(eE, eD))); linf.append(float(max(abs(eE), abs(eD)))); nfix.append(nf)
    l2 = np.array(l2); linf = np.array(linf)
    return dict(l2_rms=float(np.sqrt(np.mean(l2**2))), l2_p90=float(np.percentile(l2, 90)),
                linf_p90=float(np.percentile(linf, 90)), linf_p99=float(np.percentile(linf, 99)),
                linf_over=float(np.mean(linf >= MARGIN)), nfix=float(np.mean(nfix)))


def main():
    print("=" * 100)
    print("FLOOR PROBE -- is the binding term the velocity prior or the per-fix lateral NOISE FLOOR?")
    print("=" * 100)

    print("\n[A] PERFECT velocity (truth-seeded, no vel update), bias=0, lat_sigma=0.265 -- "
          "if p90 still >>0.155, the FLOOR is the noise, not velocity")
    r = leg(37.0, 0.265, "perfect")
    print(f"   perfect-vel  Linf_p90={r['linf_p90']:.3f}  Linf_p99={r['linf_p99']:.3f}  "
          f"over={r['linf_over']:.3f}  (~{r['nfix']:.1f} fixes in window)")
    r = leg(37.0, 0.265, "cold")
    print(f"   cold (+-1.5) Linf_p90={r['linf_p90']:.3f}  Linf_p99={r['linf_p99']:.3f}  over={r['linf_over']:.3f}")
    r = leg(37.0, 0.265, "warm_cont")
    print(f"   warm_cont    Linf_p90={r['linf_p90']:.3f}  Linf_p99={r['linf_p99']:.3f}  over={r['linf_over']:.3f}"
          f"   <-- d3 'warm' = continuous truth-velocity injection EVERY tick (UNAVAILABLE in case C)")

    print("\n[B] per-fix lateral sigma sweep at PERFECT velocity (find sigma clearing p90<0.155):")
    print(f"   {'lat_sig':>7} | {'Linf_p90':>8} {'Linf_p99':>8} {'over':>6}")
    for ls in [0.10, 0.15, 0.20, 0.25, 0.265, 0.30]:
        r = leg(37.0, ls, "perfect")
        flag = "  <- clears p90" if r['linf_p90'] < MARGIN else ""
        print(f"   {ls:>7.3f} | {r['linf_p90']:>8.3f} {r['linf_p99']:>8.3f} {r['linf_over']:>6.3f}{flag}")

    print("\n[C] speed ladder at PERFECT velocity, lat_sigma=0.265 (does the FLOOR move with speed?):")
    for v in [15.0, 25.0, 37.0, 50.0]:
        r = leg(v, 0.265, "perfect")
        print(f"   v={v:>5.0f}  Linf_p90={r['linf_p90']:.3f}  (~{r['nfix']:.1f} fixes) "
              f"-- fewer fixes at higher speed => higher floor")


if __name__ == "__main__":
    main()
