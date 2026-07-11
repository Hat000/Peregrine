"""rl/tools/leveler_bench.py -- ACCEPTANCE / VALIDATION harness for the estimator-faithful package
(rl/ego_ins_emul.py). Three gates, kept strictly separate:

  MODE A -- WIRE-REPLAY PARITY (algorithm correctness, NOT the faithfulness band): drive the
    translated BatchedESKFLeveler (NED/FRD instantiation, float64) through the a5/a7 recorded
    HIGHRES_IMU stream at the RECORDED ego_obs tick schedule (latest sample at each tick, dt = the
    tick gap -- the verified deploy rate contract) and compare per tick against the recorded wire
    obs[3:5] via the yaw-invariant tilt angle. PASS: median <~0.1-0.5 deg (the residual is which
    sample the wire actually held). This pins that the translation IS the deploy algorithm.
    ALSO Mode-A velocity (graft: parity's Mode C): replay the BatchedNavKF (predict-only -- the a5
    flight was map-free, gates=[] -> zero fixes; corroborated below by the kf_pos parity) through
    the SAME emulated attitude and compare v_obs = diag(-1,1,-1) @ (R^T v_ned) vs recorded obs[0:3]
    and kf_pos vs recorded kf_pos_ned.

  MODE B -- MATCHED-DT PASS BAND (the faithfulness benchmark, derived from REAL data, not vibes):
    rerun the leveler over the same recorded 143 Hz IMU stream on (i) the recorded tick grid
    (~57 ms median -- directly comparable to the measured wire benchmark ~5-7 deg median / ~20-27
    p90 / ~35 max per tick) and (ii) a SYNTHETIC 33.3 ms grid (the training tick), scoring per-tick
    divergence of the tick-stepped leveler vs the DENSE trapezoid gyro integral (the truth proxy).
    Band (ii) is the in-env T3 pass band the PRECHECK rollout is scored against.

  MODE C -- DENSE-STEPPING CONTROL (the aliasing attribution): step the leveler at EVERY IMU
    sample and score the same per-tick divergence. Result (a5): ~0.36 deg median -- the residual
    ZOH-vs-trapezoid discretization floor -- vs 1.5 deg (33 ms grid) / 5.2 deg (57 ms grid) when
    tick-stepped: the divergence is SUBSAMPLING ALIASING, not filter error. The EXACT ZOH negative
    control (truth itself piecewise-constant at tick rate -> leveler tracks truth to float
    precision, the n_substeps=1 in-env analog) is pinned synthetically in
    tests/test_estimator_faithful.py. In-env consequence: ~0 deg T3 divergence at n_substeps=1 is
    EXPECTED; ~0 at n_substeps=5 under a matched-|f| trajectory is a STOP-SHIP signal (escalate K
    once, then intra-tick rate-loop sysid realism -- never an invented noise constant).

Usage:
  python leveler_bench.py <handoff_dir>            # full tlog-driven run (laptop)
  python leveler_bench.py --fixture <fixture.json> # committed-fixture run (CI/Adroit, Mode A only)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parent))

from ego_ins_emul import (BatchedESKFLeveler, BatchedNavKF, GRAVITY,                 # noqa: E402
                          quat_from_roll_pitch_yaw_zyx, quat_to_R_wxyz)
from imu_foundation import propagate_dir, g_hat_from_roll_pitch                     # noqa: E402

G_NED = [0.0, 0.0, +GRAVITY]
# obs-frame (virtual-flipped FLU) from true-FRD vector map: RZ_PI @ FLIP = diag(-1, 1, -1)
OBS_FROM_FRD = np.array([-1.0, 1.0, -1.0])


# ------------------------------------------------------------------ data loading
def load_from_handoff(d: Path):
    from imu_foundation import load_highres_imu
    t_us, acc, gyr = load_highres_imu(str(d / "mavlink.tlog"))
    ticks = []
    with open(d / "ego_obs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ticks.append({"t_us": r["sim_time_ns"] / 1000.0, "obs": r["obs"][:8],
                          "kf_pos_ned": r.get("kf_pos_ned")})
    return t_us.astype(np.float64), acc, gyr, ticks


def load_from_fixture(p: Path):
    d = json.loads(p.read_text())
    imu = d["imu"]
    return (np.asarray(imu["t_us"], dtype=np.float64), np.asarray(imu["accel_frd"]),
            np.asarray(imu["gyro_true_frd"]), d["ticks"])


def latest_at(t_us, arr, t):
    """The newest sample at/before time t (the drained-receiver / navigator contract)."""
    i = int(np.searchsorted(t_us, t, side="right")) - 1
    return arr[max(i, 0)]


# ------------------------------------------------------------------ frame helpers
def tilt_obs_from_R_ned(R):
    """Wire obs tilt vector from a NED body->world R: the obs extraction is roll_pitch_zup on
    R_obs = (FLIP R FLIP) @ RZ_PI, whose tilt vector is [R20, -R21, R22] of the NED R."""
    u = np.array([R[2, 0], -R[2, 1], R[2, 2]])
    return u / np.linalg.norm(u)


def seed_from_obs(roll_obs, pitch_obs):
    """Invert the obs mapping for the NED-core seed: roll_ned = -roll_obs, pitch_ned = pitch_obs,
    yaw datum 0 (both derived from the FLIP/RZ_PI chain; pinned by round-trip in the tests)."""
    return quat_from_roll_pitch_yaw_zyx(
        torch.tensor([-roll_obs], dtype=torch.float64),
        torch.tensor([pitch_obs], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64))


def _stats(a):
    a = np.asarray(a)
    return {"n": int(a.size), "median_deg": float(np.median(a)),
            "p90_deg": float(np.percentile(a, 90)), "max_deg": float(a.max())}


# ------------------------------------------------------------------ MODE A: wire replay
def mode_a_wire_replay(t_us, acc, gyr, ticks):
    """Two attitude parity reads:
      * RE-ANCHORED (the translation pin): per tick, seed a fresh leveler at the WIRE's recorded
        k-1 roll/pitch, one step, compare at k -- isolates the single-step algorithm from open-loop
        accumulation. The wire's own tickrate_step_check baseline is median ~0.0003 deg.
      * FREE-RUNNING (the end-to-end read): seed once at tick 0 and never re-anchor. The tail here
        is DOMINATED by which-sample ambiguity (a mid-tick DroneState refresh / the 1-2 recorded
        low-|a| accel-update outlier ticks) compounding open-loop -- expected, reported for honesty.
    Velocity/pos: free-running BatchedNavKF (predict-only, map-free) through the free-running
    emulated attitude vs recorded obs[0:3] / kf_pos_ned."""
    idx0 = torch.tensor([0])
    # --- re-anchored single-step parity ---
    reanch = []
    for k in range(1, len(ticks)):
        tp, tk = ticks[k - 1]["t_us"], ticks[k]["t_us"]
        lev1 = BatchedESKFLeveler(1, g_world=G_NED, dtype=torch.float64)
        lev1.reset_idx(idx0, seed_from_obs(ticks[k - 1]["obs"][3], ticks[k - 1]["obs"][4]))
        lev1.step(torch.tensor(latest_at(t_us, gyr, tk)).unsqueeze(0).double(),
                  torch.tensor(latest_at(t_us, acc, tk)).unsqueeze(0).double(), (tk - tp) * 1e-6)
        u_emul = tilt_obs_from_R_ned(quat_to_R_wxyz(lev1.q)[0].numpy())
        u_wire = g_hat_from_roll_pitch(ticks[k]["obs"][3], ticks[k]["obs"][4])
        reanch.append(math.degrees(math.acos(float(np.clip(np.dot(u_emul, u_wire), -1, 1)))))

    # --- free-running replay (attitude + KF velocity/pos) ---
    lev = BatchedESKFLeveler(1, g_world=G_NED, dtype=torch.float64)
    kf = BatchedNavKF(1, g_world=G_NED, dtype=torch.float64)
    lev.reset_idx(idx0, seed_from_obs(ticks[0]["obs"][3], ticks[0]["obs"][4]))
    kf.reset_idx(idx0, torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64))
    att_err, vel_err, pos_err = [], [], []
    for k in range(1, len(ticks)):
        tp, tk = ticks[k - 1]["t_us"], ticks[k]["t_us"]
        dt = (tk - tp) * 1e-6
        gy = torch.tensor(latest_at(t_us, gyr, tk), dtype=torch.float64).unsqueeze(0)
        ac = torch.tensor(latest_at(t_us, acc, tk), dtype=torch.float64).unsqueeze(0)
        lev.step(gy, ac, dt)
        R = quat_to_R_wxyz(lev.q)[0].numpy()
        kf.predict(ac, torch.tensor(R, dtype=torch.float64).unsqueeze(0), dt)
        u_emul = tilt_obs_from_R_ned(R)
        u_wire = g_hat_from_roll_pitch(ticks[k]["obs"][3], ticks[k]["obs"][4])
        att_err.append(math.degrees(math.acos(float(np.clip(np.dot(u_emul, u_wire), -1, 1)))))
        v_obs_emul = OBS_FROM_FRD * (R.T @ kf.velocity[0].numpy())
        vel_err.append(float(np.linalg.norm(v_obs_emul - np.asarray(ticks[k]["obs"][0:3]))))
        if ticks[k].get("kf_pos_ned") is not None:
            pos_err.append(float(np.linalg.norm(kf.position[0].numpy()
                                                - np.asarray(ticks[k]["kf_pos_ned"]))))
    out = {"attitude_tilt_vs_wire_REANCHORED": _stats(reanch),
           "attitude_tilt_vs_wire_free_running": _stats(att_err)}
    ve = np.asarray(vel_err)
    out["velocity_obs_vs_wire_m_s_free_running"] = {
        "median": float(np.median(ve)), "p90": float(np.percentile(ve, 90)), "max": float(ve.max())}
    if pos_err:
        pe = np.asarray(pos_err)
        out["kf_pos_vs_wire_m_free_running"] = {
            "median": float(np.median(pe)), "p90": float(np.percentile(pe, 90)), "max": float(pe.max())}
    return out


# ------------------------------------------------------------------ MODE B/C: divergence bands
def divergence_band(t_us, acc, gyr, tick_times_us, seed_roll_obs, seed_pitch_obs, dense=False):
    """Per-tick divergence of the (tick- or dense-)stepped leveler vs the dense trapezoid gyro
    integral -- the acceptance benchmark method, run on the TRANSLATED leveler. Comparison space:
    NED/FRD tilt vector u = R^T e_z (row 2 of R); the dense integral propagates u with the raw
    true-FRD gyro (frame-consistent, yaw-invariant)."""
    lev = BatchedESKFLeveler(1, g_world=G_NED, dtype=torch.float64)
    lev.reset_idx(torch.tensor([0]), seed_from_obs(seed_roll_obs, seed_pitch_obs))
    angles, amags = [], []
    for k in range(1, len(tick_times_us)):
        tp, tk = tick_times_us[k - 1], tick_times_us[k]
        R_prev = quat_to_R_wxyz(lev.q)[0].numpy()
        u_prev = R_prev[2, :] / np.linalg.norm(R_prev[2, :])
        if dense:
            # step at EVERY IMU sample inside (tp, tk] -- the ZOH / n_substeps=1 analog
            i0 = int(np.searchsorted(t_us, tp, side="right"))
            i1 = int(np.searchsorted(t_us, tk, side="right"))
            t_cursor = tp
            for j in range(i0, i1):
                lev.step(torch.tensor(gyr[j]).unsqueeze(0).double(),
                         torch.tensor(acc[j]).unsqueeze(0).double(), (t_us[j] - t_cursor) * 1e-6)
                t_cursor = t_us[j]
            if tk > t_cursor:
                lev.step(torch.tensor(latest_at(t_us, gyr, tk)).unsqueeze(0).double(),
                         torch.tensor(latest_at(t_us, acc, tk)).unsqueeze(0).double(),
                         (tk - t_cursor) * 1e-6)
        else:
            # ONE step per tick on the LATEST sample -- the deploy rate contract
            lev.step(torch.tensor(latest_at(t_us, gyr, tk)).unsqueeze(0).double(),
                     torch.tensor(latest_at(t_us, acc, tk)).unsqueeze(0).double(),
                     (tk - tp) * 1e-6)
        R_now = quat_to_R_wxyz(lev.q)[0].numpy()
        u_emul = R_now[2, :] / np.linalg.norm(R_now[2, :])
        u_dense = propagate_dir(u_prev, t_us, gyr, tp, tk)
        angles.append(math.degrees(math.acos(float(np.clip(np.dot(u_emul, u_dense), -1, 1)))))
        amags.append(np.linalg.norm(latest_at(t_us, acc, tk)) / GRAVITY)
    s = _stats(angles)
    a = np.asarray(amags)
    s["sf_mag_g"] = {"median": float(np.median(a)), "p90": float(np.percentile(a, 90))}
    return s


def main():
    args = sys.argv[1:]
    if args and args[0] == "--fixture":
        t_us, acc, gyr, ticks = load_from_fixture(Path(args[1]))
        full = False
    else:
        t_us, acc, gyr, ticks = load_from_handoff(Path(args[0]))
        full = True

    print("=== MODE A: wire-replay parity (translated leveler + KF vs recorded obs) ===")
    print(json.dumps(mode_a_wire_replay(t_us, acc, gyr, ticks), indent=2))

    tick_t = np.array([tk["t_us"] for tk in ticks])
    r0, p0 = ticks[0]["obs"][3], ticks[0]["obs"][4]
    print("\n=== MODE B(i): divergence band at the RECORDED tick grid (vs the wire benchmark) ===")
    print(json.dumps(divergence_band(t_us, acc, gyr, tick_t, r0, p0), indent=2))
    grid33 = np.arange(tick_t[0], tick_t[-1], 1e6 / 30.0)
    print("\n=== MODE B(ii): divergence band at the 33.3 ms TRAINING tick grid (the T3 pass band) ===")
    print(json.dumps(divergence_band(t_us, acc, gyr, grid33, r0, p0), indent=2))
    print("\n=== MODE C: ZOH negative control (dense stepping ~= n_substeps=1) -- must be ~0 ===")
    print(json.dumps(divergence_band(t_us, acc, gyr, grid33, r0, p0, dense=True), indent=2))
    if not full:
        print("(fixture mode: same data, shorter IMU margin)")


if __name__ == "__main__":
    main()
