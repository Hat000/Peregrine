"""v_margin_independent_check -- ADVERSARIAL re-derivation of the d3 margin-closure verdict.

This does NOT import d3. It composes the SAME real stack (racer.state_estimator.LinearKF +
kf_rewind_buffer.RewindKF) but with an INDEPENDENT trajectory construction and INDEPENDENT
probes, to test whether d3's headline claims survive:

  CLAIM 2: cold case-C fails the 0.155 m margin at p90 even at ZERO bias (p90~0.235, ~37% over).
  CLAIM 3: velocity prior (via accel bias) is the swing, not speed / not latency.
  CLAIM 5: margin closes only with sigma_v<=~0.3 AND eff accel bias<=~0.1.

Adversarial probes (things d3 could have gotten wrong that would FLIP the verdict):
  P1  L2 vs L-inf geometry: the REAL contact margin is L-inf (max(|E|,|D|)); d3 scores L2
      (hypot(E,D)). L-inf <= L2, so d3's "fail" is CONSERVATIVE. Re-score the SAME runs both ways.
  P2  cold v_init prior magnitude: d3 seeds cold velocity with +-1.5 m/s (vel_std 1.5). Is the
      "37% over" an artifact of an over-pessimistic cold seed, or does it persist when the lap
      has CONVERGED the velocity (the realistic entering-g4 prior, ~0.14-0.21 m/s per d3 itself)?
      Probe: seed cold with the d3-MEASURED entering-g4 velocity error (0.14 m/s), short straight.
  P3  along-track contamination: d3 projects out the along-track (u34) component, but uses the
      g3->g4 axis as the in-plane basis at a point that is the SPLINE ENDPOINT (curving). Re-derive
      on a pure straight g3->g4 leg where in-plane is unambiguous.
  P4  fix-window sensitivity: does the verdict depend on FIX_WINDOW_M=12? sweep 8/12/16.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/v_margin_independent_check.py
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

SEED = 770413
MARGIN = 0.155
PER_AXIS_LAT_SIGMA = 0.265
RADIAL_SIGMA = 0.50
ACCEL_NOISE_STD = 0.3
IMU_DT = 1.0 / 90.0
FIX_DT = 1.0 / 14.0  # 30Hz * 0.47 accept
LINEAR_DRAG = 0.21

G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])


def straight_leg_margin(v_race, accel_bias_mag, vel_init_err_mps, latency_ms,
                        fix_window_m=12.0, n_mc=600, horizon_s=0.5):
    """Pure straight g3->g4 const-velocity leg. The drone flies EXACTLY centred (truth in-plane
    offset = 0), so the in-plane miss is purely the estimator error. Velocity prior degraded by
    vel_init_err_mps (a controlled lateral seed error) + an integrated constant accel bias.
    Returns L2 and L-inf percentile stats so we can compare geometry conventions on the SAME runs.
    """
    seg = G4 - G3
    L = float(np.linalg.norm(seg))
    uhat = seg / L
    T = L / v_race
    n_steps = int(T / IMU_DT)
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(LINEAR_DRAG * v_race / GRAVITY_NED[2]))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    accel_body_nom = R_wb.T @ (-GRAVITY_NED)  # const-v specific force
    # in-plane basis perp to uhat
    e1 = np.cross(uhat, np.array([0.0, 0.0, 1.0])); e1 /= np.linalg.norm(e1)
    e2 = np.cross(uhat, e1); e2 /= np.linalg.norm(e2)
    L_s = latency_ms / 1e3

    l2, linf = [], []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 17 * s + int(v_race) * 31
                                  + int(accel_bias_mag * 1000) * 7
                                  + int(vel_init_err_mps * 1000) * 101 + int(latency_ms) * 53)
        # constant random-direction body accel bias of fixed magnitude
        d = r.normal(0, 1, 3); d /= (np.linalg.norm(d) + 1e-12)
        accel_bias_body = accel_bias_mag * d
        p0 = G3.copy()
        # velocity prior: truth + a lateral error of magnitude vel_init_err_mps (random dir in-plane)
        dvdir = r.normal(0, 1, 2); dvdir /= (np.linalg.norm(dvdir) + 1e-12)
        v_init = uhat * v_race + vel_init_err_mps * (dvdir[0] * e1 + dvdir[1] * e2)
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v_init, pos_std=0.5,
                                 vel_std=max(0.3, vel_init_err_mps))
        rk = RewindKF(kf=kf, horizon_s=horizon_s)
        t = 0.0; t_ns = 0; next_fix = 0.0
        queue = []
        for k in range(n_steps):
            t += IMU_DT
            t_ns += int(IMU_DT * 1e9)
            accel_meas = accel_body_nom + accel_bias_body + r.normal(0, ACCEL_NOISE_STD, 3)
            rk.predict(accel_meas, R_wb, IMU_DT, t_ns)
            p_true = p0 + uhat * v_race * t
            rng_to_g4 = float(np.linalg.norm(G4 - p_true))
            if t >= next_fix and rng_to_g4 < fix_window_m:
                next_fix = t + FIX_DT
                nlat = (r.normal(0, PER_AXIS_LAT_SIGMA) * e1
                        + r.normal(0, PER_AXIS_LAT_SIGMA) * e2
                        + r.normal(0, RADIAL_SIGMA) * uhat)
                z = p_true + nlat
                cov = (PER_AXIS_LAT_SIGMA**2) * (np.outer(e1, e1) + np.outer(e2, e2)) \
                    + (RADIAL_SIGMA**2) * np.outer(uhat, uhat)
                queue.append((t + L_s, t_ns, z.copy(), cov.copy()))
            queue.sort(key=lambda e: e[0])
            while queue and queue[0][0] <= t + 1e-12:
                _, cap_ns, z, cov = queue.pop(0)
                rk.update_position_at(cap_ns, z, cov)
        p_true_final = p0 + uhat * v_race * (n_steps * IMU_DT)
        err = rk.position - p_true_final
        eE = float(err @ e1); eD = float(err @ e2)
        l2.append(float(np.hypot(eE, eD)))
        linf.append(float(max(abs(eE), abs(eD))))
    l2 = np.array(l2); linf = np.array(linf)
    return dict(
        l2_rms=float(np.sqrt(np.mean(l2**2))), l2_p90=float(np.percentile(l2, 90)),
        l2_p99=float(np.percentile(l2, 99)), l2_frac_over=float(np.mean(l2 >= MARGIN)),
        linf_rms=float(np.sqrt(np.mean(linf**2))), linf_p90=float(np.percentile(linf, 90)),
        linf_p99=float(np.percentile(linf, 99)), linf_frac_over=float(np.mean(linf >= MARGIN)),
    )


def main():
    print("=" * 110)
    print("INDEPENDENT MARGIN CHECK -- straight g3->g4 leg, real LinearKF+RewindKF, L2 vs L-inf")
    print(f"  margin={MARGIN} m  lat_sigma={PER_AXIS_LAT_SIGMA}  radial_sigma={RADIAL_SIGMA}")
    print("=" * 110)

    print("\n[P1+P2] velocity-prior error sweep @ 37 m/s, bias=0, GPU 15ms  "
          "(d3 cold entering-g4 velErr ~0.14-0.21 m/s; d3 cold SEED is +-1.5 m/s)")
    print(f"{'velErr m/s':>10} | {'L2_p90':>7} {'L2_p99':>7} {'L2_over':>7} | "
          f"{'Linf_p90':>8} {'Linf_p99':>8} {'Linf_over':>9}")
    for ve in [0.0, 0.05, 0.10, 0.15, 0.21, 0.5, 1.0, 1.5]:
        r = straight_leg_margin(37.0, 0.0, ve, 15.0)
        print(f"{ve:>10.2f} | {r['l2_p90']:>7.3f} {r['l2_p99']:>7.3f} {r['l2_frac_over']:>7.3f} | "
              f"{r['linf_p90']:>8.3f} {r['linf_p99']:>8.3f} {r['linf_frac_over']:>9.3f}")

    print("\n[P3] accel-bias sweep @ 37 m/s, velErr=0.15 (realistic cold entering-g4), GPU 15ms")
    print(f"{'bias m/s2':>9} | {'L2_p90':>7} {'L2_p99':>7} | {'Linf_p90':>8} {'Linf_p99':>8}")
    for b in [0.0, 0.05, 0.1, 0.24, 0.3, 0.5]:
        r = straight_leg_margin(37.0, b, 0.15, 15.0)
        print(f"{b:>9.2f} | {r['l2_p90']:>7.3f} {r['l2_p99']:>7.3f} | "
              f"{r['linf_p90']:>8.3f} {r['linf_p99']:>8.3f}")

    print("\n[P4] fix-window sweep @ 37 m/s, velErr=0.15, bias=0")
    print(f"{'window m':>9} | {'L2_p90':>7} {'Linf_p90':>8}")
    for w in [8.0, 12.0, 16.0]:
        r = straight_leg_margin(37.0, 0.0, 0.15, 15.0, fix_window_m=w)
        print(f"{w:>9.1f} | {r['l2_p90']:>7.3f} {r['linf_p90']:>8.3f}")

    print("\n[latency] velErr=0.15, bias=0.1, 15 vs 115 ms")
    for lat in [15.0, 115.0]:
        r = straight_leg_margin(37.0, 0.1, 0.15, lat, n_mc=400)
        print(f"  lat={lat:>5.0f}ms  L2_p90={r['l2_p90']:.3f}  Linf_p90={r['linf_p90']:.3f}")


if __name__ == "__main__":
    main()
