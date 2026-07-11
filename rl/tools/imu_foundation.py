"""IMU measured-data foundation for the ESTIMATOR-FAITHFUL ACTOR OBS package.

Extracts, from a VQ2 flight handoff dir (mavlink.tlog + ego_obs.jsonl):
  (1) PAD-IDLE per-axis gyro/accel bias + noise (THE training sensor constants),
  (2) HIGHRES_IMU rate/jitter characterization,
  (3) the ACCEPTANCE BENCHMARK: per-tick tilt divergence of the wire leveler
      (ego_obs obs[3:5]) vs gyro-only tilt propagation over the same tick.

PROVENANCE / FOOTGUNS (pinned empirically, 2026-07-11):
  * The raw tlog HIGHRES_IMU gyro is ALL-AXES NEGATED vs true FRD (the nav negates
    it before use; corr -1.0 raw / +1.0 negated vs commanded rates). We negate here.
    The accel is NOT negated (pad-idle z ~ -9.8 = correct FRD specific force).
  * HIGHRES_IMU.time_usec and ego_obs sim_time_ns are the SAME sim-boot epoch
    (t_us == sim_time_ns/1000); verified by window overlap in this script.
  * Tilt propagation is YAW-INVARIANT: g_body = R^T e_z_world does not depend on
    yaw, and body-side propagation d/dt g_body = -omega x g_body preserves that,
    so gyro-drifting yaw never enters the comparison.

MEASURED RESULTS (a5 pad-idle 2026-07-10 dataset, this script, run 2026-07-11) --
the constants rl/ego_ins_emul.py bakes with provenance comments:
  * the sim IMU is NOISELESS: 11,856 samples over 101.16 s at fs=143.33 Hz contain
    EXACTLY 1 unique gyro row and 1 unique accel row; gyro bias = [0,0,0] rad/s
    EXACT, gyro/accel noise density = 0.
  * accel magnitude offset +0.003351 m/s^2 (direction-inseparable from the modeled
    tilted pad, roll ~-0.01 / pitch ~-17.80 deg matching obs[4]=-0.31071 rad).
  * IMU rate median 143.3 Hz, p99 dt 14.0 ms (bimodal: skipped-sample tail).
  * a5 flight window |a|: median 2.98 g / p90 4.03 g / max 5.21 g.
  * acceptance benchmark (47 flight ticks, median tick dt ~57 ms): per-tick
    leveler-vs-dense-gyro divergence median ~5.3 / p90 ~26.7 / max ~34.5 deg
    (a7: 2.9/10.5/15.6) -- the ~7/20 deg wire benchmark's method-variant envelope.
  * tickrate_step_check median ~0.0003 deg: the deployed AHRS steps ONCE PER
    CONTROL TICK with the LATEST gyro sample -- the error is SAMPLING ALIASING.

Usage:
  python imu_foundation.py <handoff_dir>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

G = 9.80665


# ---------------------------------------------------------------------------
# (0) tlog parsing
# ---------------------------------------------------------------------------

def load_highres_imu(tlog_path: str):
    """Return (t_us[N], accel[N,3] FRD m/s^2, gyro[N,3] TRUE-FRD rad/s).

    FOOTGUN APPLIED: raw tlog gyro is all-axes negated vs true FRD -> negate.
    """
    from pymavlink import mavutil     # lazy: fixture-mode consumers (pytest) don't need pymavlink
    m = mavutil.mavlink_connection(tlog_path, dialect="common", robust_parsing=True)
    t, acc, gyr = [], [], []
    while True:
        msg = m.recv_match(type="HIGHRES_IMU", blocking=False)
        if msg is None:
            break
        t.append(msg.time_usec)
        acc.append((msg.xacc, msg.yacc, msg.zacc))
        gyr.append((msg.xgyro, msg.ygyro, msg.zgyro))
    t = np.asarray(t, dtype=np.int64)
    acc = np.asarray(acc, dtype=np.float64)
    gyr = -np.asarray(gyr, dtype=np.float64)   # <-- all-axes negation footgun
    order = np.argsort(t, kind="stable")
    return t[order], acc[order], gyr[order]


# ---------------------------------------------------------------------------
# (1) pad-idle window + sensor constants
# ---------------------------------------------------------------------------

def pad_idle_stats(t_us, acc, gyr, t_flight_start_us, g_tol=0.01):
    """Longest contiguous pre-flight run with | |a| - g | < g_tol * g.

    Duplicate-timestamp samples are dropped first (keep the LAST per timestamp):
    a paused sim spews the same frozen sample under one time_usec (a7: 36,137
    copies of a single sample at t=4.49 s), which would poison the statistics."""
    _, keep = np.unique(t_us[::-1], return_index=True)      # last occurrence wins
    keep = len(t_us) - 1 - keep
    t_us, acc, gyr = t_us[np.sort(keep)], acc[np.sort(keep)], gyr[np.sort(keep)]
    amag = np.linalg.norm(acc, axis=1)
    ok = (np.abs(amag - G) < g_tol * G) & (t_us < t_flight_start_us)
    # longest contiguous True run
    best_s = best_e = -1
    s = None
    for i, v in enumerate(ok):
        if v and s is None:
            s = i
        if (not v or i == len(ok) - 1) and s is not None:
            e = i if v else i - 1
            if best_s < 0 or e - s > best_e - best_s:
                best_s, best_e = s, e
            s = None
    sl = slice(best_s, best_e + 1)
    tw = t_us[sl]
    dt = np.diff(tw) * 1e-6
    fs = 1.0 / np.median(dt)
    out = {
        "n_samples": int(best_e - best_s + 1),
        # bit-identical-sample evidence: 1 unique row over the whole window == the sim
        # models NO sensor noise (a5: 11,856 samples, 1 unique gyro row, 1 unique accel row)
        "n_unique_gyro_rows": int(len(np.unique(gyr[sl], axis=0))),
        "n_unique_accel_rows": int(len(np.unique(acc[sl], axis=0))),
        "window_s": float((tw[-1] - tw[0]) * 1e-6),
        "fs_hz": float(fs),
        # gyro: pad truth is omega=0 -> mean = bias, std = white noise at fs
        "gyro_bias_rad_s": gyr[sl].mean(axis=0).tolist(),
        "gyro_std_rad_s": gyr[sl].std(axis=0, ddof=1).tolist(),
        "gyro_noise_density_rad_s_sqrtHz": (gyr[sl].std(axis=0, ddof=1) / math.sqrt(fs)).tolist(),
        # accel: pad tilt unknown (modeled tilted pad) -> per-axis mean is
        # tilt+bias mixed; only |mean|-g and the per-axis std are clean.
        "accel_mean_m_s2": acc[sl].mean(axis=0).tolist(),
        "accel_mean_mag_minus_g_m_s2": float(np.linalg.norm(acc[sl].mean(axis=0)) - G),
        "accel_std_m_s2": acc[sl].std(axis=0, ddof=1).tolist(),
        "accel_noise_density_m_s2_sqrtHz": (acc[sl].std(axis=0, ddof=1) / math.sqrt(fs)).tolist(),
        "accel_tilt_from_mean_deg": {
            "roll": float(math.degrees(math.atan2(acc[sl].mean(axis=0)[1], -acc[sl].mean(axis=0)[2]))),
            "pitch": float(math.degrees(math.asin(np.clip(acc[sl].mean(axis=0)[0] / np.linalg.norm(acc[sl].mean(axis=0)), -1, 1)))),
        },
    }
    return out, sl


# ---------------------------------------------------------------------------
# (2) rate / jitter
# ---------------------------------------------------------------------------

def rate_stats(t_us, label):
    dt = np.diff(t_us) * 1e-6
    dt = dt[dt > 0]
    return {
        "label": label,
        "n": int(len(dt) + 1),
        "rate_hz_median": float(1.0 / np.median(dt)),
        "rate_hz_mean": float(1.0 / dt.mean()),
        "dt_ms": {
            "median": float(np.median(dt) * 1e3),
            "p1": float(np.percentile(dt, 1) * 1e3),
            "p99": float(np.percentile(dt, 99) * 1e3),
            "max": float(dt.max() * 1e3),
            "std": float(dt.std(ddof=1) * 1e3),
        },
    }


# ---------------------------------------------------------------------------
# (3) acceptance benchmark: leveler vs gyro-only tilt propagation
# ---------------------------------------------------------------------------

def g_hat_from_roll_pitch(roll, pitch):
    """Unit 'world-vertical in body' vector from leveled roll/pitch (yaw-free).
    Same formula for FRD/NED (gravity dir) and FLU/Z-up (up dir): R^T e_z with
    R = Rz(yaw) Ry(pitch) Rx(roll); yaw drops out."""
    return np.array([
        -math.sin(pitch),
        math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
    ])


def propagate_dir(v, t_us, gyr, t0_us, t1_us):
    """Propagate body-frame fixed-world-direction v across [t0,t1] using gyro only:
    v' = exp(-[omega]x dt) v, trapezoid omega between consecutive IMU samples,
    with partial edge sub-intervals held at the boundary sample's omega."""
    i0 = np.searchsorted(t_us, t0_us, side="left")
    i1 = np.searchsorted(t_us, t1_us, side="right") - 1
    if i1 < i0:  # no sample inside: single hold using nearest sample
        j = min(max(i0 - 1, 0), len(t_us) - 1)
        return _rot(v, gyr[j], (t1_us - t0_us) * 1e-6)
    # leading edge: t0 -> first inside sample, hold omega at that sample
    v = _rot(v, gyr[i0], (t_us[i0] - t0_us) * 1e-6)
    for j in range(i0, i1):
        w = 0.5 * (gyr[j] + gyr[j + 1])          # trapezoid
        v = _rot(v, w, (t_us[j + 1] - t_us[j]) * 1e-6)
    # trailing edge: last inside sample -> t1
    v = _rot(v, gyr[i1], (t1_us - t_us[i1]) * 1e-6)
    return v


def _rot(v, omega, dt):
    """Rodrigues: rotate v by exp(-[omega]x * dt) (fixed world dir seen from a
    body rotating at omega)."""
    if dt <= 0:
        return v
    th = np.linalg.norm(omega) * dt
    if th < 1e-12:
        return v
    k = omega / np.linalg.norm(omega)
    # v' = exp(-[omega]x dt) v = cos(th) v - sin(th) (k x v) + (1-cos th)(k.v) k
    return math.cos(th) * v - math.sin(th) * np.cross(k, v) + (1 - math.cos(th)) * np.dot(k, v) * k


# The obs body frame is the VIRTUAL-FLIPPED FLU frame (ego_obs.py: R_obs = R_zup @ RZ_PI,
# w_obs = RZ_PI @ FLIP @ w_frd_true). Rates transform as vectors under the constant body
# re-labelings, so the true-FRD gyro expressed in the obs frame is
#   w_obs = diag(-1,-1,1) @ diag(1,-1,-1) @ w_frd_true = diag(-1,1,-1) @ w_frd_true
# (== diag(1,-1,1) @ RAW tlog gyro, since true FRD = -raw). Empirically adjudicated: this
# multiplier is the unique minimum over all 8 sign combos (a5: median 5.3 deg vs >=17.7 for
# every other combo), and matches the VQ2_GYRO_SIGN=(-1,-1,-1) wire correction chain exactly.
OBS_FRAME_FROM_TRUE_FRD = np.array([-1.0, 1.0, -1.0])


def acceptance_benchmark(obs_path, t_us, gyr):
    """ACCEPTANCE BENCHMARK: per-tick tilt divergence of the wire leveler vs gyro-only
    tilt propagation.

    Method (yaw-invariant TILT comparison):
      1. ticks: (t_k = sim_time_ns/1000 us, roll_k = obs[3], pitch_k = obs[4]).
      2. tilt vector u(roll, pitch) = R^T e_z = [-sin p, cos p sin r, cos p cos r]
         (yaw-free by construction; obs[3:5] are the virtual-flipped-FLU leveled angles).
      3. propagate u_{k-1} across (t_{k-1}, t_k] using ONLY the IMU gyro expressed in the
         obs body frame (OBS_FRAME_FROM_TRUE_FRD * true-FRD gyro): u' = exp(-[w]x dt) u,
         trapezoid w between consecutive IMU samples, boundary sub-intervals held at the
         edge sample's w (propagate_dir above).
      4. divergence angle_k = arccos(u_pred . u(roll_k, pitch_k)), stats over the flight.

    ALSO returns the tick-rate-stepping diagnostic: the same comparison but propagating with
    the single obs[5:8] body-rate AT tick k over the whole tick gap. On a5/a7 this is ~0,
    proving the deployed AHRS stepped ONCE PER CONTROL TICK with the latest gyro sample
    (dt = full tick gap) -- the divergence vs the dense-IMU integral is ALIASING error.
    """
    ticks = []
    with open(obs_path) as f:
        for line in f:
            r = json.loads(line)
            ticks.append((r["sim_time_ns"] / 1000.0, r["obs"][3], r["obs"][4],
                          np.array(r["obs"][5:8], dtype=np.float64)))
    g = gyr * OBS_FRAME_FROM_TRUE_FRD
    t0 = ticks[0][0]
    angles, alias0 = [], []
    for (tp, rp, pp, _), (tk, rk, pk, wk) in zip(ticks[:-1], ticks[1:]):
        u_prev = g_hat_from_roll_pitch(rp, pp)
        u_obs = g_hat_from_roll_pitch(rk, pk)
        v_pred = propagate_dir(u_prev, t_us, g, tp, tk)
        angles.append(((tk - t0) * 1e-6,
                       math.degrees(math.acos(np.clip(np.dot(v_pred, u_obs), -1.0, 1.0)))))
        # tick-rate-stepping proof: single step with the CURRENT tick's obs rates
        v_1 = _rot(u_prev, wk, (tk - tp) * 1e-6)
        alias0.append(math.degrees(math.acos(np.clip(np.dot(v_1, u_obs), -1.0, 1.0))))
    a = np.array([x[1] for x in angles])
    z = np.array(alias0)
    summary = {
        "n_ticks": len(a),
        "median_deg": float(np.median(a)),
        "p90_deg": float(np.percentile(a, 90)),
        "max_deg": float(a.max()),
        "tickrate_step_check_median_deg": float(np.median(z)),
        "tickrate_step_check_p90_deg": float(np.percentile(z, 90)),
        "tickrate_step_check_max_deg": float(z.max()),
    }
    return angles, summary


def main():
    d = Path(sys.argv[1])
    print(f"=== {d.name} ===")
    t_us, acc, gyr = load_highres_imu(str(d / "mavlink.tlog"))
    print(f"HIGHRES_IMU samples: {len(t_us)}  span {(t_us[-1]-t_us[0])/1e6:.1f} s")

    obs_path = d / "ego_obs.jsonl"
    with open(obs_path) as f:
        rows = [json.loads(line) for line in f]
    t_flight0 = rows[0]["sim_time_ns"] / 1000.0
    t_flight1 = rows[-1]["sim_time_ns"] / 1000.0
    print(f"ego_obs ticks: {len(rows)}  flight window {t_flight0/1e6:.3f} -> {t_flight1/1e6:.3f} s "
          f"({(t_flight1-t_flight0)/1e6:.2f} s)")
    print(f"IMU epoch check: imu t in [{t_us[0]/1e6:.3f}, {t_us[-1]/1e6:.3f}] s "
          f"(must bracket the flight window)")

    # (1) pad-idle
    pad, sl = pad_idle_stats(t_us, acc, gyr, t_flight0)
    print("\n--- PAD-IDLE SENSOR CONSTANTS ---")
    print(json.dumps(pad, indent=2))

    # (2) rate/jitter
    print("\n--- IMU RATE/JITTER ---")
    print(json.dumps(rate_stats(t_us, "full log"), indent=2))
    fl = (t_us >= t_flight0) & (t_us <= t_flight1)
    if fl.sum() > 2:
        print(json.dumps(rate_stats(t_us[fl], "flight window"), indent=2))

    # flight-window specific force (the high-g regime evidence)
    if fl.sum() > 2:
        am = np.linalg.norm(acc[fl], axis=1) / G
        print(f"\nflight |specific force| (g): median {np.median(am):.2f} "
              f"p90 {np.percentile(am, 90):.2f} max {am.max():.2f}")

    # (3) acceptance benchmark
    print("\n--- ACCEPTANCE BENCHMARK (per-tick leveler-vs-gyro tilt divergence) ---")
    angles, s = acceptance_benchmark(obs_path, t_us, gyr)
    print(json.dumps(s, indent=2))
    for tr, ang in angles:
        print(f"  t={tr:6.3f}s  {ang:7.2f} deg")


if __name__ == "__main__":
    main()
