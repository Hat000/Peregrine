# ============================================================================
# 🚩 SUPERSEDED 2026-06-19 (Fengyou caught a real error) -- DO NOT REUSE margin_at()/W_EFF.
# W_EFF = 0.75 - 0.215 - r DOUBLE-COUNTS THE DRONE: 0.215 (chassis half-diagonal)
# AND r (the contact-halo crash radius) are BOTH the drone, subtracted twice
# (MARGIN = 0.235 @ r=0.30 -> the false 0.08 bar = 0.235/3). The SIM is correct:
# pass = offset < 0.75 - r -> real gate clearance ~0.37-0.47 m; the real centering
# bar is sigma_p0_lat ~0.15 m (p99 ~0.45). Measured 0.15-0.20 = MARGINAL-PASSING,
# NOT NO-GO. Kept as the worked example / active engine (cited BY NAME in the
# correction): the 4-D sweep machinery is reusable, but the W_EFF/MARGIN value and
# ANY 0.08-derived verdict are WRONG. The "near-field-estimator pivot" this fed is
# RETRACTED -- RL stays the tool. See MEMORY.md:8 + project_rl_increment_history.md
# §inc8-2026-06-19.
# ============================================================================
"""MARGIN-CLOSURE-ENVELOPE -- gate-4 contact margin re-derived at the MEASURED vision quality.

The headline CANNOT-SETTLE-OFFLINE was computed at the PESSIMISTIC modeled lateral sigma=0.265 m AND
at the worst-case contact radius r=0.38. Track-3 shadow recording MEASURED lateral sigma~0.10 m with
~zero bias at the gate-4 band, and the central reporting radius is r=0.30. This module re-derives the
gate-4 in-plane contact margin at sigma=0.10 and sweeps the 4-D operating space to produce the ENVELOPE
that decides race-the-cap vs raise-it.

ENGINE = the d3/g3 COLD case-C machinery, re-composed on the PRODUCTION stack (do NOT re-implement the
filter):
  - racer.state_estimator.LinearKF  (predict/update_position, Joseph form, attitude-lever Q)
  - racer.kf_rewind.RewindKF        (PRODUCTIONIZED OOSM rewind/replay; the g3 engine's filter)
  - smooth Catmull-Rom g0->g4 truth (ported verbatim from d3_margin_closure: C1 velocity, real
    IMU-sensed maneuver accel, so cold velocity degrades ONLY through the integrated accel bias).

Case-C reality = COLD: velocity is UNOBSERVABLE from position-only vision; it is dead-reckoned from the
accelerometer and corrected ONLY by gate-relative POSITION fixes (no velocity update). The only
systematic velocity corruptor is a constant effective attitude/accel BIAS (the unpinned realism).

SWEPT KNOBS (the 4-D operating space):
  - inplane_sigma : per-fix in-plane lateral 1-sigma. DEFAULT 0.10 (measured); 0.265 = modeled baseline.
  - att_bias_deg  : effective SYSTEMATIC attitude error in [0, 1.4] deg -> accel bias = g*sin(theta).
                    (The KF's INTERNAL attitude_noise_std stays at the production 1.4 deg Q-inflation;
                     this swept term is the SYSTEMATIC part on top -- the v3-vindicated non-double-count.)
  - bias_mode     : "random3d" (d3 default; ~1/3 energy wasted along-track) or
                    "inplane" (v3 conservative: phantom forced into the gate-4 in-plane).
  - fix_rate      : fraction of 30 Hz detector frames that yield an ACCEPTED fix. DEFAULT 0.07 measured;
                    swept to 0.50. fix_dt = 1/(DETECTOR_HZ*fix_rate). The NEW binding lever (camera
                    pointing): 0.07 -> 2.1 Hz fixes (sparse, cold velocity coasts long); 0.47 -> 14 Hz.
  - v_race        : gate-4 approach speed (speed ladder).
  - radius r      : contact radius band {0.21,0.26,0.30,0.33,0.38}, central 0.30. NOT a Monte-Carlo
                    axis -- frac-over at every r is read off the SAME in-plane miss distribution via
                    MARGIN(r) = W_EFF - r, W_EFF = 0.535 m (= 0.75 gate-clear-halfwidth - 0.215 chassis).

VERDICT per cell = p90(miss) < MARGIN(r) AND p99(miss) < MARGIN(r)  (a margin is a worst-case gate).

Run (full sweep, ~overnight):
  PYTHONPATH=src ../../../.venv/Scripts/python.exe handoff/margin-closure-envelope-2026-06-14/margin_envelope.py
Robust: every cell is try/except-guarded (errors skipped + logged, never abort); partial results are
checkpointed to results/ after every cell. Seed 20260614; reproduces on re-run.
[MARGIN-CLOSURE-ENVELOPE 2026-06-14]
"""
from __future__ import annotations

import os

# Pin BLAS to a single thread PER PROCESS (set before numpy import). The KF matmuls are 6x6 -- BLAS
# threading gives no speedup and, under the process Pool, 11 procs x multi-threaded numpy on 14 cores
# oversubscribes catastrophically. One thread/proc => clean ~11x scaling. [MARGIN-CLOSURE-ENVELOPE]
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from racer.frames import R_world_from_body  # noqa: E402
from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from racer.kf_rewind import RewindKF  # noqa: E402  (PRODUCTIONIZED OOSM -- the g3 engine's filter)

# -------------------------------------------------------------------------------------------------
# Constants (anchored to d3_margin_closure + g3_margin_sim + the memory footguns).
# -------------------------------------------------------------------------------------------------
SEED = 20260614
G = 9.80665
MARGIN_G4_AT_R038 = 0.155       # m, contact-true in-plane margin @ r=0.38 (the anchor)
W_EFF = MARGIN_G4_AT_R038 + 0.38  # 0.535 m = 0.75 gate-clear-halfwidth - 0.215 chassis half-diag
RADIUS_BAND = [0.21, 0.26, 0.30, 0.33, 0.38]
CENTRAL_R = 0.30
LINEAR_DRAG = 0.21              # /s, measured twin-fit drag (sets the drag-hold pitch posture)
IMU_HZ = 90.0
IMU_DT = 1.0 / IMU_HZ
DETECTOR_HZ = 30.0
RADIAL_SIGMA = 0.50            # m, along-track (depth) PnP per-fix sigma (loose by design)
ACCEL_NOISE_STD = 0.3         # m/s^2, the LinearKF white accel noise (matches Q)
FIX_WINDOW_M = 12.0           # gate becomes a usable 4-corner relative fix within ~12 m
SIGMA_MEASURED = 0.10         # MEASURED gate-4-band lateral sigma (simops shadow, 20-26 m band)
SIGMA_MODELED = 0.265         # the pessimistic modeled constant (production GATE_REL_INPLANE_SIGMA)
FIXRATE_MEASURED = 0.07       # MEASURED accept fraction (simops: 7% of frames yield accepted fix)

GATES = {
    0: np.array([-23.30, -0.40, -0.03]),
    1: np.array([-46.89, -2.50, 5.07]),
    2: np.array([-74.59, 1.20, 13.67]),
    3: np.array([-111.49, -5.10, 24.57]),
    4: np.array([-135.49, -0.80, 25.36]),
}


def margin_at(r: float) -> float:
    return W_EFF - r


def att_deg_to_accel_bias(theta_deg: float) -> float:
    """Effective systematic attitude error (deg) -> phantom horizontal accel magnitude (m/s^2).
    g*sin(theta): a fixed attitude misalignment rotates gravity into a constant specific-force bias.
    (g*sin(1.4 deg)=0.2396, g*sin(0.6 deg)=0.1027 -- matches d4v's exact mapping.)"""
    return float(G * np.sin(np.deg2rad(theta_deg)))


# =================================================================================================
# Smooth, physically-consistent truth trajectory g0..g4  (ported verbatim from d3_margin_closure).
# =================================================================================================
def seg_axes(g_from, g_to):
    seg = g_to - g_from
    L = float(np.linalg.norm(seg))
    return seg, L, seg / L


def drag_hold_pitch(v: float) -> float:
    return float(np.arctan2(LINEAR_DRAG * v, GRAVITY_NED[2]))


def _catmull_rom(P, n_per_seg=600):
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
    key = round(float(v_race), 4)
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE[key] = build_truth(v_race)
    return _TRUTH_CACHE[key]


def build_truth(v_race):
    gate_pts = [GATES[k] for k in range(5)]
    dense = _catmull_rom(gate_pts, n_per_seg=600)
    ds = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    gate_s = []
    for g in gate_pts:
        gi = int(np.argmin(np.linalg.norm(dense - g, axis=1)))
        gate_s.append(s_cum[gi])
    leg_speed = [0.45 * v_race, 0.65 * v_race, 0.85 * v_race, v_race, v_race]
    speed_of_s = np.clip(np.interp(s_cum, gate_s, leg_speed), 2.0, None)
    dt_seg = ds / (0.5 * (speed_of_s[1:] + speed_of_s[:-1]))
    t_cum = np.concatenate([[0.0], np.cumsum(dt_seg)])
    T = t_cum[-1]
    NG = 4000
    tg = np.linspace(0.0, T, NG)
    pg = np.empty((NG, 3))
    for j in range(3):
        pg[:, j] = np.interp(tg, t_cum, dense[:, j])
    dtg = T / (NG - 1)
    vg = np.gradient(pg, dtg, axis=0)
    ag = np.gradient(vg, dtg, axis=0)
    base = dict(T=T, tg=tg, pg=pg, vg=vg, ag=ag, dtg=dtg)

    # ---- PRECOMPUTE the deterministic per-IMU-step driver sequence (IDENTICAL across MC seeds;
    #      only the noise + bias differ per lap). This lifts every scipy Rotation / trig / target-gate
    #      computation out of the hot Monte-Carlo loop. The values are bit-for-bit what the original
    #      per-step code produced -- only WHEN they are computed changes. ----
    _, _, u34 = seg_axes(GATES[3], GATES[4])
    yaw34 = float(np.arctan2(u34[1], u34[0]))
    R_wb_g4 = R_world_from_body(0.0, drag_hold_pitch(v_race), yaw34)
    gate_basis = {}
    for gk_i in range(1, 5):
        _, _, uk = seg_axes(GATES[gk_i - 1], GATES[gk_i])
        e1k, e2k = _inplane_basis(uk)
        gate_basis[gk_i] = (uk, e1k, e2k)

    n_alloc = int(np.ceil(T / IMU_DT)) + 1
    S_p = np.empty((n_alloc, 3)); S_v = np.empty((n_alloc, 3)); S_R = np.empty((n_alloc, 3, 3))
    S_ab = np.empty((n_alloc, 3)); S_dt = np.empty(n_alloc); S_ts = np.empty(n_alloc)
    S_tns = np.empty(n_alloc, dtype=np.int64); S_tgt = np.full(n_alloc, -1, dtype=np.int64)
    S_u = np.zeros((n_alloc, 3)); S_e1 = np.zeros((n_alloc, 3)); S_e2 = np.zeros((n_alloc, 3))
    tt = 0.0; tns = 0; ns = 0
    for k in range(n_alloc):
        dt = min(IMU_DT, T - tt)
        if dt <= 1e-9:
            break
        tt += dt; tns += int(round(dt * 1e9))
        p_t, v_t, a_t = truth_at(base, tt)
        spd = float(np.linalg.norm(v_t)); yaw = float(np.arctan2(v_t[1], v_t[0]))
        R_wb = R_world_from_body(0.0, drag_hold_pitch(max(spd, 1.0)), yaw)
        S_p[k] = p_t; S_v[k] = v_t; S_R[k] = R_wb; S_ab[k] = R_wb.T @ (a_t - GRAVITY_NED)
        S_dt[k] = dt; S_ts[k] = tt; S_tns[k] = tns
        for gk_i in range(1, 5):
            gk = GATES[gk_i]
            if p_t[0] > gk[0] - 1.0 and float(np.linalg.norm(gk - p_t)) < FIX_WINDOW_M:
                uk, e1k, e2k = gate_basis[gk_i]
                S_tgt[k] = gk_i; S_u[k] = uk; S_e1[k] = e1k; S_e2[k] = e2k
                break
        ns = k + 1

    base.update(dict(
        n_steps=ns, u34=u34, R_wb_g4=R_wb_g4,
        p_init_true=pg[0].copy(), v_init_true=vg[0].copy(), p_final_true=truth_at(base, T)[0],
        s_p=S_p[:ns], s_v=S_v[:ns], s_R=S_R[:ns], s_ab=S_ab[:ns], s_dt=S_dt[:ns],
        s_ts=S_ts[:ns], s_tns=S_tns[:ns], s_tgt=S_tgt[:ns], s_u=S_u[:ns], s_e1=S_e1[:ns],
        s_e2=S_e2[:ns]))
    return base


def truth_at(truth, t):
    tg, dtg = truth["tg"], truth["dtg"]
    i = int(t / dtg)
    if i >= len(tg) - 1:
        return truth["pg"][-1].copy(), truth["vg"][-1].copy(), truth["ag"][-1].copy()
    f = (t - tg[i]) / dtg
    p = truth["pg"][i] * (1 - f) + truth["pg"][i + 1] * f
    v = truth["vg"][i] * (1 - f) + truth["vg"][i + 1] * f
    a = truth["ag"][i] * (1 - f) + truth["ag"][i + 1] * f
    return p, v, a


def _inplane_basis(uhat):
    """Two orthonormal world directions spanning the plane perpendicular to approach axis uhat."""
    e1 = np.cross(uhat, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(uhat, np.array([0.0, 1.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(uhat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


# =================================================================================================
# One Monte-Carlo cold lap g0->g4. Velocity IMU-integrated w/ systematic accel bias, corrected ONLY
# by gate-relative POSITION fixes (case-C reality). Returns the gate-4 in-plane miss.
# =================================================================================================
def fly_lap(rng, v_race, inplane_sigma, accel_bias_mag, bias_mode, fix_rate, vel_mode,
            latency_ms, horizon_s=0.5, fix_window_m=FIX_WINDOW_M):
    """One Monte-Carlo cold lap g0->g4 on the PRECOMPUTED deterministic driver sequence. The per-step
    attitude/specific-force/target-geometry come from get_truth(v_race) (computed once, cached); only
    the accel noise + the systematic bias + the per-fix vision noise are drawn here per lap."""
    L_s = latency_ms / 1e3
    tr = get_truth(v_race)
    u34 = tr["u34"]

    # constant body-frame accel bias for this run
    if accel_bias_mag <= 0.0:
        accel_bias_body = np.zeros(3)
    elif bias_mode == "inplane":
        e1, e2 = _inplane_basis(u34)
        ang = rng.uniform(0, 2 * np.pi)
        d_world = np.cos(ang) * e1 + np.sin(ang) * e2     # in the gate-4 in-plane (world)
        accel_bias_body = tr["R_wb_g4"].T @ (accel_bias_mag * d_world)
    else:  # random3d (d3 default)
        d = rng.normal(0, 1, 3)
        d /= (np.linalg.norm(d) + 1e-12)
        accel_bias_body = accel_bias_mag * d

    # init at g0 with a realistic prior; COLD = velocity prior poorly known (lap must converge it).
    p_init = tr["p_init_true"] + rng.normal(0, 0.5, 3)
    if vel_mode == "warm":
        v_init = tr["v_init_true"].copy()
    else:
        v_init = tr["v_init_true"] + rng.normal(0, 1.5, 3)
    kf = LinearKF.initialize(p_init, v_init, pos_std=1.0, vel_std=1.5)
    rk = RewindKF(kf=kf, horizon_s=horizon_s)

    s_p, s_v, s_R, s_ab = tr["s_p"], tr["s_v"], tr["s_R"], tr["s_ab"]
    s_dt, s_ts, s_tns = tr["s_dt"], tr["s_ts"], tr["s_tns"]
    s_tgt, s_e1, s_e2, s_u = tr["s_tgt"], tr["s_e1"], tr["s_e2"], tr["s_u"]
    n_steps = tr["n_steps"]

    fix_apply_queue = []
    fix_dt = 1.0 / (DETECTOR_HZ * fix_rate)
    next_fix_t = 0.0
    last_fix_z = None
    last_fix_t = None
    warm = (vel_mode == "warm")
    weakvel = (vel_mode == "weakvel")
    sig2_ip = inplane_sigma ** 2
    sig2_rad = RADIAL_SIGMA ** 2

    for k in range(n_steps):
        t = s_ts[k]
        t_ns = int(s_tns[k])
        # TRUE specific force + constant body BIAS + white noise -> velocity observable by
        # dead-reckoning, BIAS the only systematic corruptor.
        accel_meas = s_ab[k] + accel_bias_body + rng.normal(0, ACCEL_NOISE_STD, 3)
        rk.predict(accel_meas, s_R[k], s_dt[k], t_ns)
        if warm:
            rk.update_velocity(s_v[k], (0.1 ** 2) * np.eye(3), sim_time_ns=t_ns)

        if s_tgt[k] >= 0 and t >= next_fix_t:
            next_fix_t = t + fix_dt
            e1, e2, target_u = s_e1[k], s_e2[k], s_u[k]
            n_lat = (rng.normal(0, inplane_sigma) * e1
                     + rng.normal(0, inplane_sigma) * e2
                     + rng.normal(0, RADIAL_SIGMA) * target_u)
            z = s_p[k] + n_lat
            cov = sig2_ip * (np.outer(e1, e1) + np.outer(e2, e2)) + sig2_rad * np.outer(target_u, target_u)
            fix_apply_queue.append((t + L_s, t_ns, z.copy(), cov.copy()))

        if fix_apply_queue:
            fix_apply_queue.sort(key=lambda e: e[0])
            while fix_apply_queue and fix_apply_queue[0][0] <= t + 1e-12:
                _, cap_ns, z, cov = fix_apply_queue.pop(0)
                rk.update_position_at(cap_ns, z, cov)
                if weakvel:
                    if last_fix_z is not None and last_fix_t is not None:
                        dt_fix = (cap_ns - last_fix_t) / 1e9
                        if dt_fix > 1e-3:
                            v_meas = (z - last_fix_z) / dt_fix
                            sig_v = np.sqrt(2.0) * inplane_sigma / dt_fix
                            rk.update_velocity(v_meas, (sig_v ** 2) * np.eye(3), sim_time_ns=t_ns)
                    last_fix_z = z.copy()
                    last_fix_t = cap_ns

    err = rk.position - tr["p_final_true"]
    along = float(np.dot(err, u34))
    inplane_vec = err - along * u34
    return float(np.linalg.norm(inplane_vec))


def run_cell(v_race, inplane_sigma, att_bias_deg, bias_mode, fix_rate, vel_mode,
             latency_ms=15.0, n_mc=400):
    """Monte-Carlo one operating-space cell -> in-plane miss distribution + frac-over at all radii."""
    accel_bias_mag = att_deg_to_accel_bias(att_bias_deg)
    misses = []
    for s in range(n_mc):
        rng = np.random.default_rng(
            SEED + 101 * s + int(round(v_race)) * 13 + int(round(att_bias_deg * 100)) * 7
            + int(round(inplane_sigma * 1000)) * 17 + int(round(fix_rate * 1000)) * 23
            + {"cold": 1, "warm": 2, "weakvel": 3}[vel_mode] * 1009
            + {"random3d": 0, "inplane": 1}[bias_mode] * 3001 + int(round(latency_ms)) * 53)
        misses.append(fly_lap(rng, v_race, inplane_sigma, accel_bias_mag, bias_mode, fix_rate,
                              vel_mode, latency_ms))
    m = np.array(misses)
    p90 = float(np.percentile(m, 90))
    p99 = float(np.percentile(m, 99))
    frac_over = {f"{r:.2f}": float(np.mean(m >= margin_at(r))) for r in RADIUS_BAND}
    clears = {f"{r:.2f}": bool(p90 < margin_at(r) and p99 < margin_at(r)) for r in RADIUS_BAND}
    return dict(
        v_race=v_race, inplane_sigma=inplane_sigma, att_bias_deg=att_bias_deg,
        accel_bias_mag=accel_bias_mag, bias_mode=bias_mode, fix_rate=fix_rate, vel_mode=vel_mode,
        latency_ms=latency_ms, n_mc=n_mc,
        inplane_rms=float(np.sqrt(np.mean(m ** 2))),
        inplane_p50=float(np.percentile(m, 50)), inplane_p90=p90, inplane_p99=p99,
        inplane_max=float(m.max()),
        frac_over=frac_over, clears_p90_and_p99=clears,
    )


# =================================================================================================
# Sweep driver -- robust (per-cell try/except + checkpoint) + multiprocessing across cells.
# =================================================================================================
def _checkpoint(out, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)


def _cell_worker(spec):
    """Top-level (picklable) worker: run one cell, returning (label, result-or-None, error-or-None).
    Each cell's seed is deterministic & order-independent, so multiprocessing is reproducible."""
    sg, v, b, fr, vm, bm, nmc = spec
    label = f"sig={sg} v={v} bias={b}deg fr={fr} {vm}/{bm}"
    try:
        t0 = time.time()
        res = run_cell(v, sg, b, bm, fr, vm, n_mc=nmc)
        res["wall_s"] = round(time.time() - t0, 2)
        return (label, res, None)
    except Exception as e:  # noqa: BLE001 -- overnight robustness: skip + log, never abort
        return (label, None, {"cell": label, "error": repr(e), "trace": traceback.format_exc()})


def build_grid(quick: bool, nmc: int):
    """Return the list of cell specs (sg, v, b, fr, vm, bm, nmc)."""
    if quick:
        primary = [(0.10, 37.0, b, fr, "cold", bm)
                   for b in [0.0, 1.4] for fr in [0.07, 0.50] for bm in ["random3d"]]
        baseline, ref = [], []
        nmc = min(nmc, 60)
    else:
        # PRIMARY: sigma=0.10 full factorial -- speed x bias x fixrate x bias-mode.
        primary = [(0.10, v, b, fr, "cold", bm)
                   for v in [25.0, 37.0, 45.0, 55.0]
                   for b in [0.0, 0.3, 0.6, 0.9, 1.4]
                   for fr in [0.07, 0.15, 0.25, 0.35, 0.50]
                   for bm in ["random3d", "inplane"]]
        # BASELINE: sigma=0.265 at RACE SPEED (37) only -- the "old envelope" comparison slice
        # (d3 already mapped 0.265 across speeds; here we only need the side-by-side at 37 m/s).
        baseline = [(0.265, 37.0, b, fr, "cold", bm)
                    for b in [0.0, 0.3, 0.6, 0.9, 1.4]
                    for fr in [0.07, 0.15, 0.25, 0.35, 0.50]
                    for bm in ["random3d", "inplane"]]
        # REF: warm ceiling + weakvel floor at the measured sigma (context only).
        ref = [(0.10, 37.0, b, fr, vm, "random3d")
               for b in [0.0, 0.6, 1.4] for fr in [0.07, 0.25, 0.50] for vm in ["warm", "weakvel"]]
    return [(sg, v, b, fr, vm, bm, nmc) for (sg, v, b, fr, vm, bm) in (primary + baseline + ref)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmc", type=int, default=600, help="Monte-Carlo laps per cell")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    ap.add_argument("--procs", type=int, default=11, help="worker processes")
    ap.add_argument("--out", type=str, default="margin_envelope_results.json")
    args = ap.parse_args()

    t_start = time.time()
    out = {
        "meta": {
            "what": "gate-4 contact-margin envelope at MEASURED vision quality (cold case-C)",
            "seed": SEED, "w_eff": W_EFF, "central_r": CENTRAL_R, "radius_band": RADIUS_BAND,
            "margin_at_r": {f"{r:.2f}": margin_at(r) for r in RADIUS_BAND},
            "sigma_measured": SIGMA_MEASURED, "sigma_modeled": SIGMA_MODELED,
            "fixrate_measured": FIXRATE_MEASURED, "radial_sigma": RADIAL_SIGMA,
            "accel_noise_std": ACCEL_NOISE_STD, "fix_window_m": FIX_WINDOW_M,
            "att_bias_to_accel": "accel = g*sin(att_bias_deg)",
            "n_mc": args.nmc, "regime": "COLD (case-C; velocity IMU-only, position fixes only)",
        },
        "cells": [], "errors": [],
    }
    out_path = _HERE / "results" / args.out

    specs = build_grid(args.quick, args.nmc)
    total = len(specs)
    cr = f"{CENTRAL_R:.2f}"
    print(f"[margin_envelope] {total} cells x nmc={args.nmc}  procs={args.procs} (quick={args.quick})",
          flush=True)

    done = 0
    with mp.Pool(processes=args.procs) as pool:
        for label, res, err in pool.imap_unordered(_cell_worker, specs, chunksize=1):
            done += 1
            if res is not None:
                out["cells"].append(res)
                print(f"[{done}/{total}] {label:52s} -> p90 {res['inplane_p90']:.3f} "
                      f"p99 {res['inplane_p99']:.3f} | r0.30 clear={res['clears_p90_and_p99'][cr]} "
                      f"({res['wall_s']}s)", flush=True)
            else:
                out["errors"].append(err)
                print(f"[{done}/{total}] {label:52s} -> ERROR {err['error']} (skipped)", flush=True)
            _checkpoint(out, out_path)  # checkpoint after EVERY cell

    out["meta"]["wall_total_s"] = round(time.time() - t_start, 1)
    out["meta"]["n_cells_ok"] = len(out["cells"])
    out["meta"]["n_cells_err"] = len(out["errors"])
    _checkpoint(out, out_path)
    print(f"\n[margin_envelope] done: {len(out['cells'])} ok / {len(out['errors'])} err "
          f"in {out['meta']['wall_total_s']}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
