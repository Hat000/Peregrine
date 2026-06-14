"""v3b -- REFUTE the CONDITIONAL closure (claim 5): does a velocity channel sigma_v<=0.3 +
low accel bias actually clear the margin, AND does it survive the REALISTIC attitude bias?

claim 5 says: (i) sigma_v<=~0.3 m/s velocity channel (d4v gives p90 0.179) AND (ii) effective
accel bias<=~0.1 m/s2 (att<=0.6deg) -> p90-clear (not p99). We test the closure with an IDEALISED
velocity pseudo-update at fixed sigma_v (the BEST a velocity channel could do), composed with the
REAL cold IMU drift, at biases spanning the OPTIMISTIC (<=0.1) and the REALISTIC-MEASURED (0.24).

The adversarial point: claim 5's p90-clear assumes BOTH a good velocity channel AND att<=0.6deg.
But the MEASURED 1-sigma attitude error is 1.4deg (bias 0.24). If the velocity channel CANNOT be
paired with att<=0.6deg, the closure collapses. We quantify the velocity channel's bias-robustness.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-gate-relative-pipeline-design-2026-06-13"))
import d3_margin_closure as d3
from racer.state_estimator import LinearKF, GRAVITY_NED
from racer.frames import R_world_from_body
from kf_rewind_buffer import RewindKF

MARGIN = d3.MARGIN_G4


def stats(m):
    m = np.array(m)
    return dict(rms=float(np.sqrt(np.mean(m**2))), p90=float(np.percentile(m, 90)),
                p99=float(np.percentile(m, 99)), frac_over=float(np.mean(m >= MARGIN)),
                p90_clear=bool(np.percentile(m, 90) < MARGIN))


def fly_visvel(rng, v_race, bias, sigma_v, lat_ms=15.0, horizon_s=0.5, bias_mode="horizontal"):
    """cold IMU drift + an IDEALISED velocity pseudo-update at fixed sigma_v (the velocity channel
    at its design point), every fix. bias injected as attitude-tilt horizontal phantom (faithful)."""
    L_s = lat_ms / 1e3
    truth = d3.get_truth(v_race); T = truth["T"]; g4 = d3.GATES[4]
    _, _, u34 = d3.seg_axes(d3.GATES[3], d3.GATES[4])
    p0t, v0t, _ = d3.truth_at(truth, 0.0)
    kf = LinearKF.initialize(p0t + rng.normal(0, 0.5, 3), v0t + rng.normal(0, 1.5, 3),
                             pos_std=1.0, vel_std=1.5)
    rk = RewindKF(kf=kf, horizon_s=horizon_s)
    if bias_mode == "horizontal":
        d = np.array([rng.normal(), rng.normal(), 0.0])
    else:
        d = rng.normal(0, 1, 3)
    d /= (np.linalg.norm(d) + 1e-12)
    bias_b = bias * d
    t = 0.0; t_ns = 0; nextf = 0.0; q = []
    fix_dt = 1.0 / (d3.DETECTOR_HZ * d3.ACCEPT)
    n_steps = int(np.ceil(T / d3.IMU_DT))
    for _ in range(n_steps):
        dt = min(d3.IMU_DT, T - t)
        if dt <= 1e-9:
            break
        t += dt; t_ns += int(round(dt * 1e9))
        pt, vt, at = d3.truth_at(truth, t)
        spd = float(np.linalg.norm(vt)); yaw = float(np.arctan2(vt[1], vt[0]))
        pitch = d3.drag_hold_pitch(max(spd, 1.0)); R_wb = R_world_from_body(0.0, pitch, yaw)
        ab_true = R_wb.T @ (at - GRAVITY_NED)
        rk.predict(ab_true + bias_b + rng.normal(0, d3.ACCEL_NOISE_STD, 3), R_wb, dt, t_ns)
        tg = None; tu = None
        for k in range(1, 5):
            gk = d3.GATES[k]
            if pt[0] > gk[0] - 1.0 and float(np.linalg.norm(gk - pt)) < d3.FIX_WINDOW_M:
                tg = gk; _, _, tu = d3.seg_axes(d3.GATES[k - 1], d3.GATES[k]); break
        if tg is not None and t >= nextf:
            nextf = t + fix_dt
            e1 = np.cross(tu, np.array([0., 0., 1.]))
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(tu, np.array([0., 1., 0.]))
            e1 /= np.linalg.norm(e1); e2 = np.cross(tu, e1); e2 /= np.linalg.norm(e2)
            n = rng.normal(0, d3.PER_AXIS_LAT_SIGMA) * e1 + rng.normal(0, d3.PER_AXIS_LAT_SIGMA) * e2 \
                + rng.normal(0, d3.RADIAL_SIGMA) * tu
            z = pt + n
            cov = (d3.PER_AXIS_LAT_SIGMA**2) * (np.outer(e1, e1) + np.outer(e2, e2)) \
                + (d3.RADIAL_SIGMA**2) * np.outer(tu, tu)
            q.append((t + L_s, t_ns, z.copy(), cov.copy(), vt.copy()))
        q.sort(key=lambda e: e[0])
        while q and q[0][0] <= t + 1e-12:
            _, cap, z, cov, vt_at = q.pop(0)
            rk.update_position_at(cap, z, cov)
            # idealised velocity channel: a velocity measurement at the design sigma_v (best case)
            v_meas = vt_at + rng.normal(0, sigma_v, 3)
            rk.update_velocity(v_meas, (sigma_v**2) * np.eye(3), sim_time_ns=t_ns)
    ptf, _, _ = d3.truth_at(truth, T)
    err = rk.position - ptf
    return float(np.linalg.norm(err - np.dot(err, u34) * u34))


def run(bias, sigma_v, n_mc=500, bias_mode="horizontal"):
    return stats([fly_visvel(np.random.default_rng(4242 + 101 * s + int(bias * 100) * 7
                                                    + int(sigma_v * 100) * 13),
                             37.0, bias, sigma_v, bias_mode=bias_mode) for s in range(n_mc)])


def main():
    out = {}
    print("v3b -- CONDITIONAL CLOSURE refute: velocity channel x accel bias @37 m/s, n_mc=500")
    print("%-8s %-8s | %7s %7s %7s %8s %6s" % ("sigma_v", "bias", "rms", "p90", "p99", "fracOver", "p90OK"))
    for sv in [0.3, 0.4, 0.5]:
        for b in [0.0, 0.0856, 0.1, 0.171, 0.240]:
            r = run(b, sv)
            out[f"sv{sv}_b{b}"] = r
            print("%-8.2f %-8.3f | %7.3f %7.3f %7.3f %8.3f %6s" % (
                sv, b, r["rms"], r["p90"], r["p99"], r["frac_over"], "Y" if r["p90_clear"] else "N"))
    Path(__file__).resolve().with_name("v3b_closure_refute_results.json").write_text(json.dumps(out, indent=2))
    print("wrote v3b_closure_refute_results.json")


if __name__ == "__main__":
    main()
