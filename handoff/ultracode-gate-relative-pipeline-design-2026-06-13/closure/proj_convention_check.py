"""COLD-MARGIN closure -- in-plane PROJECTION-CONVENTION diagnostic.

The verdict pivots on ONE ambiguity. Two faithful sims disagree on the WARM p90:
  - d3 multi-gate (fly_lap) reports the PROPER perpendicular projection  err - (err.u34) u34
    -> warm p90 ~0.125 (CLEARS 0.155).
  - the single-leg sims (c1 anchor, commander v2, d4v) report the E,D-HYPOT  hypot(err_E, err_D)
    -> warm p90 ~0.18-0.20 (BREACHES 0.155).
The 0.155 m margin is stated (CONTEXT.md) as "E lateral + D vertical in-plane; N along-track" -- i.e.
the E,D-hypot convention -- but the gate actually faces along u34 (g3->g4 has ~10 deg yaw,
u34_E ~ 0.176), so the along-track radial error (sigma 0.50) LEAKS into hypot(E,D). The proper
perpendicular projection removes that leak. This script quantifies the gap on MATCHED runs by
re-using d3's EXACT machinery (imported, not re-implemented) and reporting BOTH projections plus the
along-track error, so the verdict can be read on whichever convention the margin truly uses.

Read-only on the real filter; writes only JSON. Run from repo root:
  PYTHONPATH=src .venv/Scripts/python.exe \
    handoff/ultracode-gate-relative-pipeline-design-2026-06-13/closure/proj_convention_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_D3_DIR = _HERE.parents[1]
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-estimator-racespeed-2026-06-13"))
sys.path.insert(0, str(_D3_DIR))

import d3_margin_closure as d3  # noqa: E402  (imports the REAL LinearKF/RewindKF + truth machinery)

SEED = d3.SEED
IMU_DT = d3.IMU_DT
DETECTOR_HZ = d3.DETECTOR_HZ
ACCEPT = d3.ACCEPT
PER_AXIS_LAT_SIGMA = d3.PER_AXIS_LAT_SIGMA
RADIAL_SIGMA = d3.RADIAL_SIGMA
ACCEL_NOISE_STD = d3.ACCEL_NOISE_STD
FIX_WINDOW_M = d3.FIX_WINDOW_M
GATES = d3.GATES
MARGIN = d3.MARGIN_G4


def fly_lap_dual(rng, v_race, accel_bias_body, vel_mode, latency_ms, horizon_s=0.5):
    """EXACT copy of d3.fly_lap's dynamics, but returns BOTH in-plane conventions + along-track.

    Any divergence from d3.fly_lap would invalidate the comparison, so the loop below is line-for-
    line d3.fly_lap; only the final return is extended (proper projection, E,D-hypot, along-track)."""
    L_s = latency_ms / 1e3
    truth = d3.get_truth(v_race)
    T = truth["T"]
    g4 = GATES[4]
    _, _, u34 = d3.seg_axes(GATES[3], GATES[4])

    p0_true, v0_true, _ = d3.truth_at(truth, 0.0)
    p_init = p0_true + rng.normal(0, 0.5, 3)
    if vel_mode == "warm":
        v_init = v0_true.copy()
    else:
        v_init = v0_true + rng.normal(0, 1.5, 3)
    kf = d3.LinearKF.initialize(p_init, v_init, pos_std=1.0, vel_std=1.5)
    rk = d3.RewindKF(kf=kf, horizon_s=horizon_s)

    t = 0.0
    t_ns = 0
    last_fix_z = None
    last_fix_t = None
    fix_apply_queue = []
    fix_dt = 1.0 / (DETECTOR_HZ * ACCEPT)
    next_fix_t = 0.0
    n_steps = int(np.ceil(T / IMU_DT))

    for _ in range(n_steps):
        dt = min(IMU_DT, T - t)
        if dt <= 1e-9:
            break
        t += dt
        t_ns += int(round(dt * 1e9))
        p_true, v_true, a_true = d3.truth_at(truth, t)

        spd = float(np.linalg.norm(v_true))
        yaw = float(np.arctan2(v_true[1], v_true[0]))
        pitch = d3.drag_hold_pitch(max(spd, 1.0))
        R_wb = d3.R_world_from_body(0.0, pitch, yaw)

        accel_body_true = R_wb.T @ (a_true - d3.GRAVITY_NED)
        accel_meas = accel_body_true + accel_bias_body + rng.normal(0, ACCEL_NOISE_STD, 3)
        rk.predict(accel_meas, R_wb, dt, t_ns)
        if vel_mode == "warm":
            rk.update_velocity(v_true, (0.1**2) * np.eye(3), sim_time_ns=t_ns)

        target_gate = None
        target_u = None
        for k in range(1, 5):
            gk = GATES[k]
            if p_true[0] > gk[0] - 1.0 and float(np.linalg.norm(gk - p_true)) < FIX_WINDOW_M:
                target_gate = gk
                _, _, target_u = d3.seg_axes(GATES[k - 1], GATES[k])
                break
        if target_gate is not None and t >= next_fix_t:
            next_fix_t = t + fix_dt
            uhat = target_u
            e1 = np.cross(uhat, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(uhat, np.array([0.0, 1.0, 0.0]))
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(uhat, e1)
            e2 /= np.linalg.norm(e2)
            n_lat = rng.normal(0, PER_AXIS_LAT_SIGMA) * e1 \
                + rng.normal(0, PER_AXIS_LAT_SIGMA) * e2 \
                + rng.normal(0, RADIAL_SIGMA) * uhat
            z = p_true + n_lat
            cov = (PER_AXIS_LAT_SIGMA**2) * (np.outer(e1, e1) + np.outer(e2, e2)) \
                + (RADIAL_SIGMA**2) * np.outer(uhat, uhat)
            fix_apply_queue.append((t + L_s, t_ns, z.copy(), cov.copy()))

        fix_apply_queue.sort(key=lambda e: e[0])
        while fix_apply_queue and fix_apply_queue[0][0] <= t + 1e-12:
            _, cap_ns, z, cov = fix_apply_queue.pop(0)
            rk.update_position_at(cap_ns, z, cov)
            if vel_mode == "weakvel":
                if last_fix_z is not None and last_fix_t is not None:
                    dt_fix = (cap_ns - last_fix_t) / 1e9
                    if dt_fix > 1e-3:
                        v_meas = (z - last_fix_z) / dt_fix
                        sig_v = np.sqrt(2.0) * PER_AXIS_LAT_SIGMA / dt_fix
                        rk.update_velocity(v_meas, (sig_v**2) * np.eye(3), sim_time_ns=t_ns)
                last_fix_z = z.copy()
                last_fix_t = cap_ns

    p_true_final, _, _ = d3.truth_at(truth, T)
    err = rk.position - p_true_final
    along = float(np.dot(err, u34))
    inplane_vec = err - along * u34
    inplane_proper = float(np.linalg.norm(inplane_vec))     # d3's perpendicular projection
    inplane_edhypot = float(np.hypot(err[1], err[2]))       # c1/v2/d4v E,D-hypot convention
    return inplane_proper, inplane_edhypot, abs(along)


def run_cell(v_race, bias_mag, vel_mode, latency_ms, n_mc=800):
    proper, edhyp, alongs = [], [], []
    for s in range(n_mc):
        rng = np.random.default_rng(
            SEED + 101 * s + int(v_race) * 13 + int(bias_mag * 100) * 7
            + {"cold": 1, "warm": 2, "weakvel": 3}[vel_mode] * 1009 + int(latency_ms) * 53)
        d = rng.normal(0, 1, 3)
        d /= (np.linalg.norm(d) + 1e-12)
        out = fly_lap_dual(rng, v_race, bias_mag * d, vel_mode, latency_ms)
        proper.append(out[0]); edhyp.append(out[1]); alongs.append(out[2])
    pr = np.array(proper); ed = np.array(edhyp); al = np.array(alongs)

    def stat(a):
        return dict(rms=float(np.sqrt(np.mean(a**2))), p50=float(np.percentile(a, 50)),
                    p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                    p90_clears=bool(np.percentile(a, 90) < MARGIN),
                    rms_clears=bool(np.sqrt(np.mean(a**2)) < MARGIN))
    return dict(v=v_race, bias=bias_mag, vel_mode=vel_mode, latency_ms=latency_ms, n_mc=n_mc,
                proper=stat(pr), edhypot=stat(ed), along_track_rms=float(np.sqrt(np.mean(al**2))),
                along_track_p90=float(np.percentile(al, 90)))


def main():
    np.random.seed(SEED)
    rows = []
    print("=" * 108)
    print("PROJECTION-CONVENTION diagnostic -- d3 machinery, BOTH in-plane conventions, MATCHED runs")
    print(f"  margin={MARGIN} m | per-axis lat sigma={PER_AXIS_LAT_SIGMA} | radial sigma={RADIAL_SIGMA} | N_MC=800")
    print("=" * 108)
    hdr = ("%-8s %5s %5s %6s | %8s %8s %7s | %8s %8s %7s | %8s" %
           ("velmode", "v", "bias", "lat", "PROP_rms", "PROP_p90", "clr?",
            "EDH_rms", "EDH_p90", "clr?", "along_p90"))
    print(hdr)
    # realistic bias band: 0 (floor) and 0.24 (= meas 1.4deg att-bias phantom accel, d4v anchor)
    for vm in ("warm", "cold", "weakvel"):
        for bias in (0.0, 0.24):
            for lat in (15.0, 115.0):
                r = run_cell(37.0, bias, vm, lat, n_mc=800)
                rows.append(r)
                print("%-8s %5.0f %5.2f %6.0f | %8.3f %8.3f %7s | %8.3f %8.3f %7s | %8.3f" % (
                    vm, 37, bias, lat,
                    r["proper"]["rms"], r["proper"]["p90"], "Y" if r["proper"]["p90_clears"] else "N",
                    r["edhypot"]["rms"], r["edhypot"]["p90"], "Y" if r["edhypot"]["p90_clears"] else "N",
                    r["along_track_p90"]))
    out = _HERE.parent / "proj_convention_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
