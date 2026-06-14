"""COMMANDER independent cross-check of the gate-4 margin under COLD case-C velocity.

Independent of the d3 workflow agent (written before reading its sim) to test whether
my own honest cold-velocity full-path sim corroborates or changes the headline margin number.

The load-bearing gap in c1_gate_relative.py: it seeds velocity to truth (v0 = uhat*V) AND
sims only the g3->g4 window -> a WARM velocity prior by construction. The case-C reality:
vision is position-only (NO given-velocity update); velocity = IMU integration only, whose
dominant error is the constant attitude-bias phantom accel a_ph = skew(theta)@g (gravity-comp
makes it independent of leg attitude, constant per run). Position fixes pin position and only
WEAKLY correct velocity through the KF cross-covariance. So velocity drifts over the lap.

This sim flies the real multi-leg path g0->g4 at const speed, drives the REAL LinearKF via the
REAL RewindKF (OOSM at capture time), applies ONLY gate-relative position fixes (zero-mean
lateral noise, the measured 0.265 m/axis), and measures the in-plane (E,D) error of the KF
estimate vs truth at the gate-4 plane crossing. WARM control arm adds update_velocity to pin
velocity (reproduces c1). COLD arm is the case-C reality.

Run: from repo root, .venv\\Scripts\\python.exe handoff/_commander-xcheck/margin_xcheck.py
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
from racer.localization import FIX_COV_FLOOR_STD  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SEED = 20260613
MARGIN_G4 = 0.155
G = np.array([0.0, 0.0, 9.80665])

# Gate opening centres, Z-up (fly_rl._GATE_POS_ZUP) -> NED = zup*[1,-1,-1].
_GATE_POS_ZUP = np.array([
    [-23.2979679107666,    0.39990234375,       1.3919580206274986],
    [-46.89374923706055,   2.499990224838257,  -3.708041787147522],
    [-74.59375,           -1.2000097036361694, -12.308041214942932],
    [-111.49374389648438,  5.099989891052246,  -23.208040833473206],
    [-135.49374389648438,  0.7999902367591858, -23.995653748512268],
    [-159.19374084472656,  4.399990081787109,  -24.60804045200348],
])
_FLIP = np.array([1.0, -1.0, -1.0])
GATES_NED = _GATE_POS_ZUP * _FLIP            # opening centres in NED

# measured gate-relative per-fix lateral per-axis sigma (E,D), near band (c1 / perception-char)
LAT_SIGMA = 0.265
RADIAL_SIGMA = 0.50                          # along-LOS (N near g4) PnP noise
FIX_HZ = 14.0                                # 30 Hz * ~47% acceptance
FIX_RANGE_M = 12.0                           # gate-relative fixes only inside this range
IMU_HZ = 90.0


def fly_path(v_race, att_sigma_deg, accel_bias, arm, n_mc, latency_ms, seed0):
    """Return dict of gate-4 in-plane (E,D) error stats + velocity-error-entering-g4 stats.

    arm: 'cold' (case-C: no given-velocity) or 'warm' (pins velocity each tick, ~VQ1/c1).
    """
    dt = 1.0 / IMU_HZ
    fix_dt = 1.0 / FIX_HZ
    L_ns = int(latency_ms * 1e6)
    att_sigma = np.radians(att_sigma_deg)
    g4 = GATES_NED[4]

    # cruise attitude for the g3->g4 leg (nose-down drag-hold; only sets accel_body gravity-comp)
    seg34 = GATES_NED[4] - GATES_NED[3]
    yaw = float(np.arctan2(seg34[1], seg34[0]))
    pitch = float(-np.arctan(0.21 * v_race / 9.80665))
    R_wb_true = R_world_from_body(0.0, pitch, yaw)
    accel_body = R_wb_true.T @ (-G) + accel_bias   # gravity-comp + optional accel bias (body)

    ip_errs, vel_errs = [], []
    for s in range(n_mc):
        r = np.random.default_rng(seed0 + 31 * s)
        # constant per-run attitude error theta -> phantom accel skew(theta)@g (const)
        theta = r.normal(0.0, att_sigma, 3) if att_sigma > 0 else np.zeros(3)
        R_err = Rotation.from_rotvec(theta).as_matrix()
        R_wb_est = R_err @ R_wb_true

        # truth: piecewise-linear const-speed path through g0..g4; start AT g0.
        legs = [GATES_NED[i + 1] - GATES_NED[i] for i in range(4)]
        leg_len = [float(np.linalg.norm(d)) for d in legs]
        uhat = [d / l for d, l in zip(legs, leg_len)]

        # KF init at g0; COLD velocity = noisy/zero prior, WARM = truth-ish.
        p0 = GATES_NED[0].copy()
        v_dir0 = uhat[0]
        if arm == "warm":
            v_init = v_dir0 * v_race + r.normal(0, 0.3, 3)
            vstd = 0.5
        else:  # cold: realistic case-C launch velocity uncertainty
            v_init = v_dir0 * v_race + r.normal(0, 1.0, 3)
            vstd = 2.0
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v_init, pos_std=0.5, vel_std=vstd)
        rk = RewindKF(kf=kf, horizon_s=0.5)

        t_ns = 0
        next_fix_t = 0.0
        t = 0.0
        seg_i = 0
        seg_s = 0.0                          # distance into current leg
        vel_err_at_g4 = None
        crossed_g4 = False
        ip_at_g4 = None

        # integrate until g4 crossing or path end
        max_t = (sum(leg_len) / v_race) * 1.2
        n_steps = int(max_t / dt)
        for k in range(n_steps):
            t += dt
            t_ns += int(dt * 1e9)
            seg_s += v_race * dt
            while seg_i < 3 and seg_s > leg_len[seg_i]:
                seg_s -= leg_len[seg_i]
                seg_i += 1
            # truth position + velocity
            p_true = GATES_NED[seg_i] + uhat[seg_i] * seg_s
            v_true = uhat[seg_i] * v_race

            rk.predict(accel_body, R_wb_est, dt, t_ns)
            if arm == "warm":
                # pin velocity (VQ1-like / c1 warm): tight given-velocity update
                rk.update_velocity(v_true, (0.10**2) * np.eye(3), sim_time_ns=t_ns)

            # gate-relative position fix for the upcoming target gate within FIX_RANGE
            tgt = min(seg_i + 1, 5) if seg_s > 1.0 else seg_i + 1
            tgt = min(tgt, 5)
            # use the nearest forward gate within range
            for gcand in range(seg_i, 6):
                rng = float(np.linalg.norm(GATES_NED[gcand] - p_true))
                if rng < FIX_RANGE_M:
                    tgt = gcand
                    break
            else:
                tgt = None
            if tgt is not None and t >= next_fix_t:
                next_fix_t += fix_dt
                # gate-relative position fix: truth + zero-mean lateral (E,D) + along-track radial.
                # NO map bias (gate-relative drops it); NO 0.40 floor (no bias to absorb).
                z = p_true.copy()
                z[0] += r.normal(0, RADIAL_SIGMA)
                z[1] += r.normal(0, LAT_SIGMA)
                z[2] += r.normal(0, LAT_SIGMA)
                cov = np.diag([RADIAL_SIGMA**2, LAT_SIGMA**2, LAT_SIGMA**2])
                t_cap = t_ns - L_ns
                rk.update_position_at(t_cap, z, cov)

            # detect g4 plane crossing (N along-track): truth seg_i==3 and seg_s reaches leg_len[3]
            if seg_i == 3 and seg_s >= leg_len[3] - v_race * dt and not crossed_g4:
                crossed_g4 = True
                err = rk.position - g4
                ip_at_g4 = float(np.hypot(err[1], err[2]))
                vel_err_at_g4 = float(np.linalg.norm(rk.velocity - v_true))
                break

        if ip_at_g4 is not None:
            ip_errs.append(ip_at_g4)
            vel_errs.append(vel_err_at_g4)

    ip = np.array(ip_errs)
    ve = np.array(vel_errs)
    return dict(
        arm=arm, v_race=v_race, att_sigma_deg=att_sigma_deg, accel_bias=float(np.linalg.norm(accel_bias)),
        latency_ms=latency_ms, n=len(ip),
        ip_rms=float(np.sqrt(np.mean(ip**2))) if len(ip) else float("nan"),
        ip_p50=float(np.percentile(ip, 50)) if len(ip) else float("nan"),
        ip_p90=float(np.percentile(ip, 90)) if len(ip) else float("nan"),
        ip_p95=float(np.percentile(ip, 95)) if len(ip) else float("nan"),
        vel_err_mean=float(np.mean(ve)) if len(ve) else float("nan"),
        vel_err_p90=float(np.percentile(ve, 90)) if len(ve) else float("nan"),
        clears_margin_rms=bool(np.sqrt(np.mean(ip**2)) < MARGIN_G4) if len(ip) else False,
        clears_margin_p90=bool(np.percentile(ip, 90) < MARGIN_G4) if len(ip) else False,
    )


def main():
    N_MC = 400
    rows = []
    print("=" * 110)
    print("COMMANDER cross-check: gate-4 in-plane error, FULL g0->g4 path, gate-relative fixes only")
    print(f"  margin = {MARGIN_G4} m | lat_sigma={LAT_SIGMA} | fix {FIX_HZ}Hz<{FIX_RANGE_M}m | N_MC={N_MC}")
    print("=" * 110)
    hdr = f"{'arm':>5} {'v':>4} {'att°':>5} {'L_ms':>5} {'n':>4} {'ip_rms':>7} {'ip_p50':>7} {'ip_p90':>7} {'ip_p95':>7} {'vel_err':>8} {'rms<m?':>6} {'p90<m?':>6}"
    print(hdr)
    # warm baseline (should recover c1 ~0.11-0.14)
    configs = []
    for arm in ("warm", "cold"):
        for v in (37.0, 30.0, 25.0):
            for att in (0.0, 0.7, 1.4):
                for L in (15.0, 115.0):
                    configs.append((arm, v, att, L))
    # prune: warm only needs att=1.4 / L=115 sanity; full sweep on cold
    for (arm, v, att, L) in configs:
        if arm == "warm" and not (att in (0.0, 1.4) and L == 115.0):
            continue
        res = fly_path(v, att, np.zeros(3), arm, N_MC, L, SEED + int(v) + int(att * 10) + int(L))
        rows.append(res)
        print(f"{arm:>5} {v:>4.0f} {att:>5.1f} {L:>5.0f} {res['n']:>4} "
              f"{res['ip_rms']:>7.3f} {res['ip_p50']:>7.3f} {res['ip_p90']:>7.3f} {res['ip_p95']:>7.3f} "
              f"{res['vel_err_mean']:>8.3f} {str(res['clears_margin_rms']):>6} {str(res['clears_margin_p90']):>6}")

    out = Path(__file__).resolve().parent / "margin_xcheck_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
