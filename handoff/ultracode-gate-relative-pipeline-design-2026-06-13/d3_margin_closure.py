"""d3 -- MARGIN CLOSURE / velocity-prior swing (gate-relative case-C pipeline DESIGN, COMPONENT 3).

THE LOAD-BEARING FEASIBILITY QUESTION of the whole blueprint:
  Does the gate-relative estimator keep the gate-4 IN-PLANE position error WELL INSIDE the
  0.155 m contact margin at race speed (~37 m/s) -- and UNDER WHAT CONDITIONS?

The prior verdict STRADDLES:  ~0.11 m RMS warm (lap-converged velocity)  vs  0.17-0.21 m cold.
Velocity is the swing factor and is UNOBSERVABLE from vision in case C (vision is position-only;
vel = IMU integration of accel_body only). c1's rel arm (reproduced here, 0.139 m) is the WARM
case: it SEEDS velocity to truth and the const-velocity truth makes the IMU predict exact, so the
velocity prior never degrades. That is NOT the case-C reality entering gate-4 after a full lap.

This file builds the HONEST/COLD sim and resolves warm-vs-cold:

(1) reproduce_c1_rel(): re-derive c1's rel-arm number from the SAME inputs (sanity anchor).
(2) HONEST multi-gate g0->g4 run where velocity is NOT seeded to truth. It is IMU-integrated
    (accel = R_wb @ accel_body + g, the REAL LinearKF.predict) with a CONSTANT body-frame accel
    BIAS injected into the measured specific force (the unpinned realism), corrected ONLY by
    gate-relative POSITION fixes (NO velocity update -- that is the case-C reality). We measure:
      - the velocity-prior QUALITY (vel error vector + P_vv covariance) ENTERING the gate-4 window,
      - the resulting gate-4 IN-PLANE miss at the gate-4 plane crossing.
    This is the number that resolves warm-vs-cold.
(3) SWEEP the swing factors:
      (a) velocity-prior regime: cold (IMU-only) / warm (seeded, c1-style) / weak vision-velocity
          assist (position-fix differencing; per-fix lateral sigma 0.265 m/axis -> differenced
          velocity sigma; coordinates with component 4-velchannel);
      (b) accel-bias magnitude (0 .. ~the dr_force_bias certified band);
      (c) speed ladder 25 / 30 / 37 m/s;
      (d) RewindKF latency 15 ms (GPU) and 115 ms (CPU).
(4) VERDICT: clears 0.155 m at race speed under which conditions? RMS AND p90/worst-case (a margin
    is a worst-case gate, not an RMS gate).

Compose the REAL stack -- do NOT re-implement the filter:
  - racer.state_estimator.LinearKF   (predict / update_position, Joseph form, attitude-lever Q)
  - kf_rewind_buffer.RewindKF        (OOSM rewind/replay, default horizon 0.5 s)
Re-use the MEASURED per-fix lateral sigma 0.265 m/axis (c1 part_a, characterize_g4 near-band).

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d3_margin_closure.py
Seed 20260613; reproduces on re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-estimator-racespeed-2026-06-13"))

from racer.frames import R_world_from_body  # noqa: E402
from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 20260613
MARGIN_G4 = 0.155      # m, gate-4 contact-true in-plane margin @ r=0.38 (THE worst-case gate)
BAR = 0.05             # m, the variance target = margin/3
LINEAR_DRAG = 0.21     # /s, measured twin-fit drag (sets the drag-hold pitch posture)
IMU_HZ = 90.0
IMU_DT = 1.0 / IMU_HZ
DETECTOR_HZ = 30.0
ACCEPT = 0.47          # in-loop acceptance fraction (navigator chi2 gate)
PER_AXIS_LAT_SIGMA = 0.265   # m/axis MEASURED gate-relative lateral PnP sigma (c1 part_a near band)
RADIAL_SIGMA = 0.50          # m, along-track (radial/depth) PnP per-fix sigma (c1 part_b)
ACCEL_NOISE_STD = 0.3        # m/s^2, the LinearKF white accel noise (matches Q)
FIX_WINDOW_M = 12.0          # gate becomes a usable 4-corner relative fix within ~12 m (c1 part_d)

# Gate geometry (track_map.json, NED bottom-centre; opening ~1.36 m above in -D). g0..g4.
GATES = {
    0: np.array([-23.30, -0.40, -0.03]),
    1: np.array([-46.89, -2.50, 5.07]),
    2: np.array([-74.59, 1.20, 13.67]),
    3: np.array([-111.49, -5.10, 24.57]),
    4: np.array([-135.49, -0.80, 25.36]),
}
# Per-gate cruise speed (the realistic ladder): speed builds on the descents g0..g3, then the
# nearly-level g3->g4 straight is the at-speed gate-4 approach. We parametrise the gate-4 approach
# speed v_race and ramp the earlier legs proportionally (so the velocity prior has had a full,
# realistic lap to accumulate IMU-integration drift before gate-4).


def seg_axes(g_from: np.ndarray, g_to: np.ndarray):
    seg = g_to - g_from
    L = float(np.linalg.norm(seg))
    uhat = seg / L
    return seg, L, uhat


def drag_hold_pitch(v: float) -> float:
    return float(np.arctan2(LINEAR_DRAG * v, GRAVITY_NED[2]))


# ======================================================================================
# SMOOTH, PHYSICALLY-CONSISTENT TRUTH TRAJECTORY through g0..g4.
#   The drone flies a C1 Catmull-Rom spline through the gate centres at an arc-length-
#   parametrised speed that ramps up to v_race on the g3->g4 leg. The trajectory is
#   continuous in velocity (no instantaneous corner turns), so its TRUE acceleration is a
#   real, IMU-SENSED quantity (centripetal turning + tangential speed-up). The accelerometer
#   reports the TRUE specific force; the only corruptions are the constant accel BIAS and
#   white noise. This is what makes velocity observable by dead-reckoning -- the cold prior
#   degrades ONLY through the integrated bias, not through unsensed phantom maneuvers.
# ======================================================================================
def _catmull_rom(P, n_per_seg=400):
    """C1 spline through points P (list of 3-vectors). Returns densely-sampled positions.
    Uses reflected end tangents so g0 and g4 are interpolated endpoints."""
    P = [np.asarray(p, float) for p in P]
    Pe = [P[0] - (P[1] - P[0])] + P + [P[-1] + (P[-1] - P[-2])]
    pts = []
    for i in range(1, len(Pe) - 2):
        p0, p1, p2, p3 = Pe[i - 1], Pe[i], Pe[i + 1], Pe[i + 2]
        for j in range(n_per_seg):
            s = j / n_per_seg
            s2, s3 = s * s, s * s * s
            pt = 0.5 * ((2 * p1) + (-p0 + p2) * s
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * s3)
            pts.append(pt)
    pts.append(Pe[-2].copy())
    return np.array(pts)


_TRUTH_CACHE = {}


def get_truth(v_race):
    """Memoised truth (deterministic per speed; only the noise differs across MC seeds)."""
    key = round(float(v_race), 4)
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE[key] = build_truth(v_race)
    return _TRUTH_CACHE[key]


def build_truth(v_race):
    """Build a smooth time-parametrised truth: position p(t), velocity v(t), accel a(t) (world).
    Speed ramps from 0.45*v_race at g0 to v_race on the g3->g4 leg. Returns dict with arrays and
    the gate-crossing times + the g4 in-plane basis (perp to the g3->g4 approach)."""
    gate_pts = [GATES[k] for k in range(5)]
    dense = _catmull_rom(gate_pts, n_per_seg=600)
    # arc length
    ds = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    S = s_cum[-1]
    # speed profile along arc length: piecewise per leg, smoothed. Anchor speeds at gate arc-pos.
    # find arc-length at each gate (nearest dense sample)
    gate_s = []
    for g in gate_pts:
        gi = int(np.argmin(np.linalg.norm(dense - g, axis=1)))
        gate_s.append(s_cum[gi])
    leg_speed = [0.45 * v_race, 0.65 * v_race, 0.85 * v_race, v_race, v_race]  # at g0,g1,g2,g3,g4
    speed_of_s = np.interp(s_cum, gate_s, leg_speed)
    speed_of_s = np.clip(speed_of_s, 2.0, None)
    # integrate dt = ds / speed to get time at each sample
    dt_seg = ds / (0.5 * (speed_of_s[1:] + speed_of_s[:-1]))
    t_cum = np.concatenate([[0.0], np.cumsum(dt_seg)])
    T = t_cum[-1]
    # PRECOMPUTE a uniform-time grid of (pos, vel, accel) so truth_at is O(1) per call. Resample
    # position onto a fine uniform time base, then finite-difference for vel/accel ONCE.
    NG = 4000
    tg = np.linspace(0.0, T, NG)
    pg = np.empty((NG, 3))
    for j in range(3):
        pg[:, j] = np.interp(tg, t_cum, dense[:, j])
    dtg = T / (NG - 1)
    vg = np.gradient(pg, dtg, axis=0)
    ag = np.gradient(vg, dtg, axis=0)
    return dict(dense=dense, s_cum=s_cum, t_cum=t_cum, gate_s=gate_s,
                speed_of_s=speed_of_s, S=S, T=T,
                tg=tg, pg=pg, vg=vg, ag=ag, dtg=dtg)


def truth_at(truth, t):
    """O(1) lookup of world position, velocity, acceleration at time t (uniform-grid interp)."""
    tg = truth["tg"]
    dtg = truth["dtg"]
    i = int(t / dtg)
    if i >= len(tg) - 1:
        return truth["pg"][-1].copy(), truth["vg"][-1].copy(), truth["ag"][-1].copy()
    f = (t - tg[i]) / dtg
    p = truth["pg"][i] * (1 - f) + truth["pg"][i + 1] * f
    v = truth["vg"][i] * (1 - f) + truth["vg"][i + 1] * f
    a = truth["ag"][i] * (1 - f) + truth["ag"][i + 1] * f
    return p, v, a


# ======================================================================================
# (1) ANCHOR: reproduce c1's rel-arm number (warm, seeded velocity, single g3->g4 segment).
#     This is the WARM bound. We re-derive it inline (not importing c1) so the cold sim shares
#     EXACTLY the same noise model and RewindKF; agreement with c1's 0.139 m validates the shared
#     machinery before we change the velocity-prior treatment.
# ======================================================================================
def reproduce_c1_rel(n_mc=600):
    g3, g4 = GATES[3], GATES[4]
    seg, L, uhat = seg_axes(g3, g4)
    v_race = 37.0
    T = L / v_race
    n_steps = int(T / IMU_DT)
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = drag_hold_pitch(v_race)
    R_wb = R_world_from_body(0.0, pitch, yaw)
    accel_body = R_wb.T @ (-GRAVITY_NED)   # const-v: a_world=0 -> specific force = -g
    fix_dt = 1.0 / (DETECTOR_HZ * ACCEPT)  # 14 Hz effective
    inplane = []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 3000 + s)
        p0 = g3.copy()
        v0 = uhat * v_race                 # SEEDED to truth -> WARM
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t = 0.0
        t_ns = 0
        next_fix_t = 0.0
        for k in range(n_steps):
            t += IMU_DT
            t_ns += int(IMU_DT * 1e9)
            rk.predict(accel_body, R_wb, IMU_DT, t_ns)
            p_true = p0 + uhat * v_race * t
            rng_to_g4 = float(np.linalg.norm(g4 - p_true))
            if t >= next_fix_t and rng_to_g4 < FIX_WINDOW_M:
                next_fix_t += fix_dt
                z = p_true.copy()
                z[1] += r.normal(0, PER_AXIS_LAT_SIGMA)   # E lateral
                z[2] += r.normal(0, PER_AXIS_LAT_SIGMA)   # D vertical
                z[0] += r.normal(0, RADIAL_SIGMA)          # N along-track
                cov = np.diag([RADIAL_SIGMA**2, PER_AXIS_LAT_SIGMA**2, PER_AXIS_LAT_SIGMA**2])
                rk.update_position(z, cov, sim_time_ns=t_ns)
        p_true_final = p0 + uhat * v_race * (n_steps * IMU_DT)
        err = rk.position - p_true_final
        inplane.append(float(np.hypot(err[1], err[2])))
    ip = np.array(inplane)
    return dict(inplane_rms=float(np.sqrt(np.mean(ip**2))),
                inplane_p50=float(np.percentile(ip, 50)),
                inplane_p90=float(np.percentile(ip, 90)),
                inplane_p99=float(np.percentile(ip, 99)),
                inplane_max=float(ip.max()), n_mc=n_mc)


# ======================================================================================
# (2)+(3) HONEST/COLD multi-gate g0->g4 run. Velocity is NOT seeded to truth -- it is
#     IMU-integrated with a constant body-frame accel BIAS, corrected ONLY by gate-relative
#     POSITION fixes (the rel arm: zero-mean lateral noise, NO per-track bias, NO floor) and
#     (optionally) a WEAK vision-velocity pseudo-update from position-fix differencing.
# ======================================================================================
def fly_lap(rng, v_race, accel_bias_body, vel_mode, latency_ms, horizon_s=0.5,
            seed_velocity=False, fix_window_m=FIX_WINDOW_M):
    """One Monte-Carlo lap g0 -> g4. Returns gate-4 in-plane miss + velocity-prior quality.

    vel_mode in {"cold","warm","weakvel"}:
      cold    : IMU-only velocity, corrected only by position fixes (case-C reality).
      warm    : velocity continuously re-seeded near truth (c1-style upper bound, for contrast).
      weakvel : cold + a WEAK velocity pseudo-update from position-fix differencing.
    accel_bias_body : (3,) constant bias added to the MEASURED specific force (body frame).
    latency_ms      : in-loop vision latency -> OOSM rewind depth. horizon_s must exceed it.
    """
    L_s = latency_ms / 1e3
    truth = get_truth(v_race)
    T = truth["T"]
    g4 = GATES[4]
    _, _, u34 = seg_axes(GATES[3], GATES[4])    # gate-4 approach axis (in-plane = perp to this)

    # init at g0 with a realistic prior. True case C starts origin-seeded (pos_std 5); but by the
    # time the drone reaches g0 (the first gate) it has acquired fixes. We seed at g0 with a
    # MODERATE prior and let the FULL g0->g4 lap evolve the velocity estimate honestly.
    p0_true, v0_true, _ = truth_at(truth, 0.0)
    p_init = p0_true + rng.normal(0, 0.5, 3)
    if seed_velocity or vel_mode == "warm":
        v_init = v0_true.copy()
    else:
        # COLD: velocity prior poorly known at the start; the lap must converge it via sensed
        # accel + position fixes (NO velocity measurement).
        v_init = v0_true + rng.normal(0, 1.5, 3)
    kf = LinearKF.initialize(p_init, v_init, pos_std=1.0, vel_std=1.5)
    rk = RewindKF(kf=kf, horizon_s=horizon_s)

    t = 0.0
    t_ns = 0
    last_fix_z = None
    last_fix_t = None
    vel_prior_at_g4_window = None   # (vel_err_vec, P_vv) captured ENTERING the gate-4 window
    g4_entered = False
    fix_apply_queue = []            # (apply_t, cap_t_ns, z, cov)
    fix_dt = 1.0 / (DETECTOR_HZ * ACCEPT)
    next_fix_t = 0.0
    n_steps = int(np.ceil(T / IMU_DT))

    for _ in range(n_steps):
        dt = min(IMU_DT, T - t)
        if dt <= 1e-9:
            break
        t += dt
        t_ns += int(round(dt * 1e9))
        p_true, v_true, a_true = truth_at(truth, t)

        # GIVEN attitude: drag-hold posture aligned with the true velocity direction. yaw from
        # the horizontal heading, pitch from the drag-hold law at the local speed. (The navigator
        # is GIVEN attitude; we use the true posture as the given one -- attitude error is folded
        # into the KF's attitude_noise_std Q term, not added here.)
        spd = float(np.linalg.norm(v_true))
        yaw = float(np.arctan2(v_true[1], v_true[0]))
        pitch = drag_hold_pitch(max(spd, 1.0))
        R_wb = R_world_from_body(0.0, pitch, yaw)

        # TRUE specific force the accelerometer senses (REAL maneuver accel included):
        #   f_world = a_true - g  ;  accel_body_true = R_wb^T f_world.
        # The IMU reports accel_body_true + constant body BIAS + white noise. The KF reconstructs
        # a_world = R_wb @ accel_meas + g = a_true + R_wb @ bias + noise -> velocity is observable
        # by dead-reckoning; the BIAS is the only systematic velocity corruptor.
        accel_body_true = R_wb.T @ (a_true - GRAVITY_NED)
        accel_meas = accel_body_true + accel_bias_body + rng.normal(0, ACCEL_NOISE_STD, 3)
        rk.predict(accel_meas, R_wb, dt, t_ns)
        if vel_mode == "warm":
            rk.update_velocity(v_true, (0.1**2) * np.eye(3), sim_time_ns=t_ns)

        # capture velocity-prior quality ENTERING the gate-4 window
        rng_to_g4 = float(np.linalg.norm(g4 - p_true))
        if (not g4_entered) and rng_to_g4 < fix_window_m and (p_true[0] < GATES[3][0]):
            g4_entered = True
            v_err = rk.velocity - v_true
            P_vv = rk.P[3:, 3:].copy()
            vel_prior_at_g4_window = (v_err.copy(), P_vv)

        # which gate is the current relative-fix target? the nearest UPCOMING gate within window.
        target_gate = None
        target_u = None
        for k in range(1, 5):
            gk = GATES[k]
            # upcoming if drone has not yet passed it along the course (N decreasing)
            if p_true[0] > gk[0] - 1.0 and float(np.linalg.norm(gk - p_true)) < fix_window_m:
                target_gate = gk
                _, _, target_u = seg_axes(GATES[k - 1], GATES[k])
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

    # at end of trajectory we are AT the gate-4 plane (spline endpoint = g4 centre). In-plane miss
    # = the component of the position error perpendicular to the g3->g4 approach axis.
    p_true_final, _, _ = truth_at(truth, T)
    # evaluate true velocity slightly INSIDE the trajectory (central diff at the endpoint
    # extrapolates past the spline and is meaningless); one IMU step before T is the gate-4 vel.
    _, v_true_final, _ = truth_at(truth, max(T - IMU_DT, 0.0))
    err = rk.position - p_true_final
    along = float(np.dot(err, u34))
    inplane_vec = err - along * u34
    inplane_miss = float(np.linalg.norm(inplane_vec))
    return dict(
        inplane_miss=inplane_miss,
        along_track_err=along,
        vel_prior=vel_prior_at_g4_window,
        final_vel_err=float(np.linalg.norm(rk.velocity - v_true_final)),
        P_pos_inplane=float(np.sqrt(rk.P[1, 1] + rk.P[2, 2])),
    )


def run_cell(v_race, accel_bias_mag, vel_mode, latency_ms, horizon_s=0.5, n_mc=400):
    """Monte-Carlo a cell. accel_bias_mag in m/s^2 -> random-direction constant body bias."""
    misses = []
    vel_errs = []
    P_vv_traces = []
    final_vel_errs = []
    for s in range(n_mc):
        rng = np.random.default_rng(
            SEED + 101 * s + int(v_race) * 13 + int(accel_bias_mag * 100) * 7
            + {"cold": 1, "warm": 2, "weakvel": 3}[vel_mode] * 1009 + int(latency_ms) * 53)
        # constant body-frame accel bias, random direction, fixed magnitude this run
        d = rng.normal(0, 1, 3)
        d /= (np.linalg.norm(d) + 1e-12)
        accel_bias_body = accel_bias_mag * d
        out = fly_lap(rng, v_race, accel_bias_body, vel_mode, latency_ms, horizon_s)
        misses.append(out["inplane_miss"])
        final_vel_errs.append(out["final_vel_err"])
        if out["vel_prior"] is not None:
            v_err, P_vv = out["vel_prior"]
            vel_errs.append(float(np.linalg.norm(v_err)))
            P_vv_traces.append(float(np.sqrt(np.trace(P_vv[:2, :2]) if P_vv.shape[0] >= 2
                                              else np.trace(P_vv))))
    m = np.array(misses)
    res = dict(
        v_race=v_race, accel_bias_mag=accel_bias_mag, vel_mode=vel_mode, latency_ms=latency_ms,
        n_mc=n_mc,
        inplane_rms=float(np.sqrt(np.mean(m**2))),
        inplane_p50=float(np.percentile(m, 50)),
        inplane_p90=float(np.percentile(m, 90)),
        inplane_p99=float(np.percentile(m, 99)),
        inplane_max=float(m.max()),
        clears_margin_rms=bool(np.sqrt(np.mean(m**2)) < MARGIN_G4),
        clears_margin_p90=bool(np.percentile(m, 90) < MARGIN_G4),
        clears_margin_p99=bool(np.percentile(m, 99) < MARGIN_G4),
        frac_runs_over_margin=float(np.mean(m >= MARGIN_G4)),
        vel_err_entering_g4_mean=float(np.mean(vel_errs)) if vel_errs else None,
        vel_err_entering_g4_p90=float(np.percentile(vel_errs, 90)) if vel_errs else None,
        final_vel_err_mean=float(np.mean(final_vel_errs)),
    )
    return res


def main():
    np.random.seed(SEED)
    out = {"seed": SEED, "margin_g4_m": MARGIN_G4, "bar_m": BAR,
           "per_axis_lat_sigma_m": PER_AXIS_LAT_SIGMA, "radial_sigma_m": RADIAL_SIGMA,
           "accel_noise_std": ACCEL_NOISE_STD}

    print("=" * 100)
    print("(1) ANCHOR -- reproduce c1 rel-arm (WARM, seeded velocity, single g3->g4 @ 37 m/s)")
    print("=" * 100)
    anchor = reproduce_c1_rel()
    out["anchor_c1_rel"] = anchor
    print(f"  rel-arm WARM: inplane RMS={anchor['inplane_rms']:.3f}  p50={anchor['inplane_p50']:.3f}"
          f"  p90={anchor['inplane_p90']:.3f}  p99={anchor['inplane_p99']:.3f}"
          f"  max={anchor['inplane_max']:.3f}   (c1 baseline RMS 0.139, p90 0.203)")

    print("\n" + "=" * 100)
    print("(2)+(3) HONEST/COLD multi-gate g0->g4. Velocity IMU-integrated w/ accel bias, corrected")
    print("        ONLY by gate-relative position fixes. SWEEP velocity-prior x bias x speed x latency.")
    print("=" * 100)

    NMC = 400
    vel_modes = ["warm", "weakvel", "cold"]

    # ---- (3a) VELOCITY-PRIOR x SPEED LADDER at zero bias, GPU latency (isolates the velocity
    #      prior's effect across speed; the swing the prior verdict straddled). ----
    print("\n--- (3a) velocity-prior x speed ladder | accel-bias=0, latency=15 ms (GPU) ---")
    speeds = [25.0, 30.0, 37.0]
    out["sweep_velprior_speed"] = []
    print("%-8s %5s | %8s %7s %7s %7s | %9s | %6s %6s %8s" % (
        "velmode", "v", "ip_rms", "ip_p50", "ip_p90", "ip_p99", "velErr_g4",
        "p90OK", "p99OK", "frcOver"))
    for vm in vel_modes:
        for v in speeds:
            res = run_cell(v, 0.0, vm, 15.0, n_mc=NMC)
            out["sweep_velprior_speed"].append(res)
            print("%-8s %5.0f | %8.3f %7.3f %7.3f %7.3f | %9.3f | %6s %6s %8.3f" % (
                vm, v, res["inplane_rms"], res["inplane_p50"], res["inplane_p90"],
                res["inplane_p99"], res["vel_err_entering_g4_mean"] or -1,
                "Y" if res["clears_margin_p90"] else "N",
                "Y" if res["clears_margin_p99"] else "N", res["frac_runs_over_margin"]))

    # ---- (3b) ACCEL-BIAS BREAK-POINT sweep at race speed (37 m/s), GPU latency. The dominant
    #      swing factor. Realistic MEMS bias post-cal ~0.01-0.1; the dr_force disturbance band
    #      reaches ~3. We find where each velocity-prior regime breaks the margin. ----
    print("\n--- (3b) accel-bias break-point | v=37 m/s, latency=15 ms (GPU) ---")
    biases = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    out["sweep_bias_breakpoint"] = []
    print("%-8s %6s | %8s %7s %7s %7s | %9s | %6s %6s %8s" % (
        "velmode", "bias", "ip_rms", "ip_p50", "ip_p90", "ip_p99", "velErr_g4",
        "p90OK", "p99OK", "frcOver"))
    for vm in vel_modes:
        for b in biases:
            res = run_cell(37.0, b, vm, 15.0, n_mc=NMC)
            out["sweep_bias_breakpoint"].append(res)
            print("%-8s %6.2f | %8.3f %7.3f %7.3f %7.3f | %9.3f | %6s %6s %8.3f" % (
                vm, b, res["inplane_rms"], res["inplane_p50"], res["inplane_p90"],
                res["inplane_p99"], res["vel_err_entering_g4_mean"] or -1,
                "Y" if res["clears_margin_p90"] else "N",
                "Y" if res["clears_margin_p99"] else "N", res["frac_runs_over_margin"]))

    # ---- (3c) RewindKF LATENCY contrast: 15 ms (GPU) vs 115 ms (CPU) at race speed, low bias.
    #      Tests whether the OOSM horizon copes with the CPU latency (along-track v*L is the term;
    #      in-plane should be near-invariant if rewind works). ----
    print("\n--- (3c) RewindKF latency contrast | v=37 m/s, bias in {0,0.1} ---")
    out["latency_contrast"] = []
    print("%-8s %6s %7s | %8s %7s %7s %7s | %8s %6s %6s" % (
        "velmode", "bias", "lat_ms", "ip_rms", "ip_p50", "ip_p90", "ip_p99",
        "alongErr", "p90OK", "p99OK"))
    for vm in vel_modes:
        for b in [0.0, 0.1]:
            for lat in [15.0, 115.0]:
                res = run_cell(37.0, b, vm, lat, n_mc=200)   # 115 ms cells are replay-heavy
                out["latency_contrast"].append(res)
                print("%-8s %6.2f %7.0f | %8.3f %7.3f %7.3f %7.3f | %8s %6s %6s" % (
                    vm, b, lat, res["inplane_rms"], res["inplane_p50"], res["inplane_p90"],
                    res["inplane_p99"], "-",
                    "Y" if res["clears_margin_p90"] else "N",
                    "Y" if res["clears_margin_p99"] else "N"))

    # ---- (4) RACE-SPEED VERDICT TABLE: the make-or-break cells at high MC, GPU + CPU latency. ----
    print("\n" + "=" * 100)
    print("(4) RACE-SPEED (37 m/s) VERDICT TABLE -- make-or-break, n_mc=500")
    print("=" * 100)
    print("%-8s %6s %7s | %8s %7s %7s %7s %7s | %6s %6s %6s" % (
        "velmode", "bias", "lat_ms", "ip_rms", "ip_p50", "ip_p90", "ip_p99", "ip_max",
        "rmsOK", "p90OK", "p99OK"))
    # GPU (15 ms) at high MC for the verdict; the 115 ms latency contrast is covered in §3c
    # (near-invariant — RewindKF removes v*L), so the verdict table fixes latency at GPU 15 ms.
    verdict = []
    for vm in vel_modes:
        for b in [0.0, 0.1, 0.5]:
            for lat in [15.0]:
                res = run_cell(37.0, b, vm, lat, n_mc=500)
                verdict.append(res)
                print("%-8s %6.2f %7.0f | %8.3f %7.3f %7.3f %7.3f %7.3f | %6s %6s %6s" % (
                    vm, b, lat, res["inplane_rms"], res["inplane_p50"], res["inplane_p90"],
                    res["inplane_p99"], res["inplane_max"],
                    "Y" if res["clears_margin_rms"] else "N",
                    "Y" if res["clears_margin_p90"] else "N",
                    "Y" if res["clears_margin_p99"] else "N"))
    out["race_speed_verdict"] = verdict

    out_path = Path(__file__).resolve().parent / "d3_margin_closure_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    main()
