"""COAST-DRIFT — does the informed terminal RewindKF coast close gate-4 r=0.30 @30 m/s?

The terminal-lock reframe (accept-geometry-2026-06-15 + boresight-closure-2026-06-14):
  ACCURATE vision fixes die at a near-field PnP/association floor ~12 m. So gate-4 terminal-lock =
  deepest accurate fix at ~r_floor (~12 m) + RewindKF COAST (predict-only, NO fixes) the final ~r_floor
  metres to the gate plane (~0.4 s @30 m/s). The boresight-closure adversarial Lens A broke ALL closing
  cells at a 0.30 s terminal *drought*; a 12 m floor @30 m/s is a 0.4 s coast -> EXCEEDS that. BUT a
  drought = zero info; an INFORMED coast dead-reckons with BOUNDED drift seeded by the velocity estimate
  from the dense accurate-band fix stream + low IMU/accel bias. THIS settles it.

This file runs three CROSS-VALIDATING analyses on the PRODUCTION KF stack (do NOT re-implement the
filter): racer.state_estimator.LinearKF + racer.kf_rewind.RewindKF, via the margin-closure engine
(margin_envelope as ME) for the physically-consistent gate-4 truth/posture/in-plane-basis.

  A1  TRANSPARENT COAST PROPAGATION (the boundary deliverable).  Predict-only propagation of the
      PRODUCTION covariance over the terminal coast, from a 12 m state with a SPECIFIED (sig_p, sig_v),
      using the REAL per-step gate-4 posture/specific-force/dt sequence. sigma_v_lat is the swept
      independent variable (the key uncertain input). Reads terminal in-plane lateral sigma + the
      accel-bias mean offset. -> the σ_v / accel-bias CLOSURE BOUNDARY.

  A2  FIX-STREAM CONDITIONING (pins the input).  Runs the production KF (RewindKF, case-C, position-only)
      over the cold lap with POINTED fixes in the ACCURATE band [r_floor, r_acc_max] (coast inside
      r_floor). The KF covariance is DETERMINISTIC given the predict/update schedule (P does not depend on
      measurement VALUES), so one pass yields the achieved sigma_v_lat / sigma_p_lat at the gate-4 floor.

  A3  FULL-LAP MC CLOSURE (honest p90/p99 + bootstrap CI).  Same accurate-band-fix + terminal-coast lap,
      Monte-Carlo (fix noise + accel noise + systematic bias), gate-4 in-plane MISS distribution, bootstrap
      CIs, vs MARGIN(r). The honest closure verdict with the full non-linear bias integration over the lap.

Engine fidelity: margin_envelope.fly_lap fixes THROUGHOUT the last FIX_WINDOW_M=12 m (the OLD model: gate
"becomes usable within 12 m"). The accept-geometry finding INVERTS that: accurate fixes EXIST in the
[~12, ~24] m band and DIE inside ~12 m. So fly_lap_coast fixes only in [r_floor, r_acc_max] and COASTS the
inner r_floor. Everything else (LinearKF init, RewindKF horizon, accel-bias model, accel white noise,
latency queue, truth driver, in-plane reduction onto u34) is ME.fly_lap / margin_driver_v2.fly_lap_v2
verbatim.

[COAST-DRIFT 2026-06-15]  PYTHONPATH not needed: ME inserts <repo>/src.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ME_DIR = _HERE.parents[0] / "margin-closure-envelope-2026-06-14"
_BC_DIR = _HERE.parents[0] / "boresight-closure-2026-06-14"
for d in (_ME_DIR, _BC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import margin_envelope as ME            # noqa: E402  (production engine; inserts <repo>/src for racer.*)
import margin_driver_v2 as MD           # noqa: E402  (anisotropic-sigma + labelled in-plane basis)
from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from racer.kf_rewind import RewindKF     # noqa: E402

SEED = 20260615
G = ME.G
ACCEL_NOISE_STD = ME.ACCEL_NOISE_STD     # 0.3 m/s^2 LinearKF white accel noise
RADIAL_SIGMA = ME.RADIAL_SIGMA           # 0.50 m along-track per-fix sigma (loose by design)
DETECTOR_HZ = ME.DETECTOR_HZ             # 30 Hz frames
IMU_DT = ME.IMU_DT                       # 1/90 s

# ---- terminal-lock geometry -------------------------------------------------------------------
R_FLOOR_DEFAULT = 12.0       # m, ACCURATE-fix floor (accept-geometry): coast inside this
R_ACC_MAX = 24.0             # m, outer edge of the accurate band (accept-geometry 18-24 m clean; the
                             #     gate-3->gate-4 leg is 24 m so the band opens right at gate-3)
# POINTED per-fix in-plane sigma in the accurate band. accept-geometry lateral MAD 0.05-0.075 m
# (-> sigma ~0.075-0.11 via MAD/0.6745); at-speed-sigma accepted-fix sigma_lat 0.13-0.17 m (moving CTBR,
# range 10-22 m). Central pointed sigma_lat ~0.10; swept 0.06..0.17. sigma_vert ~0.10.
SIGMA_LAT_POINTED = 0.10
SIGMA_VERT_POINTED = 0.10
# POINTED accept fraction in the accurate band: ~0.78 | in-FoV (accept-geometry), ~75-97% in-FoV
# (at-speed head-on). effective fix-rate of 30 Hz frames ~0.60 (0.78 x ~0.78 in-FoV). Swept.
FIXRATE_POINTED = 0.60

# margin (engine-canonical W_EFF=0.535). prompt cites in-plane budget 0.245 @ r=0.30 / 0.155 @ r=0.38;
# the engine gives M(0.30)=0.235, M(0.38)=0.155. We report against BOTH 0.235 and 0.245 for r=0.30.
def M(r):
    return ME.margin_at(r)
BUDGET_PROMPT = {0.30: 0.245, 0.38: 0.155}


def gate4_terminal_steps(v_race, r_floor, r_acc_max):
    """Indices of the per-step truth driver in the gate-4 approach, split into the accurate band
    [r_floor, r_acc_max] (fixable) and the terminal coast [0, r_floor]. Returns (tr, idx_coast,
    idx_band, k_floor) where k_floor is the first coast step."""
    tr = ME.get_truth(v_race)
    g4 = ME.GATES[4]
    s_p = tr["s_p"]
    n = tr["n_steps"]
    rng_to_g4 = np.linalg.norm(s_p[:n] - g4, axis=1)
    # only the FINAL approach (monotonically closing, x not yet past the gate)
    approaching = s_p[:n, 0] > g4[0] - 0.5
    idx_band = np.where(approaching & (rng_to_g4 <= r_acc_max) & (rng_to_g4 > r_floor))[0]
    idx_coast = np.where(approaching & (rng_to_g4 <= r_floor))[0]
    k_floor = int(idx_coast[0]) if len(idx_coast) else n
    return tr, idx_coast, idx_band, k_floor, rng_to_g4


# =================================================================================================
# A1 — TRANSPARENT COAST COVARIANCE PROPAGATION  (the boundary deliverable)
# =================================================================================================
def a1_coast_propagate(v_race, sig_v_lat, sig_p_lat=0.10, sig_v_vert=None, sig_p_vert=0.10,
                       cov_pv_lat=0.0, accel_bias_deg=0.0, bias_mode="inplane",
                       r_floor=R_FLOOR_DEFAULT):
    """Predict-only propagation of the PRODUCTION covariance over the gate-4 terminal coast.

    Build a 6-state KF at the r_floor crossing with P diagonal in the (e1=lat, e2=vert, u=along) basis
    (optionally with a pos-vel cross-covariance cov_pv_lat on the lateral axis), then run the PRODUCTION
    LinearKF.predict over EXACTLY the real per-step gate-4 coast steps (real R_wb / specific-force / dt),
    NO fix updates. Read terminal in-plane lateral 1-sigma (sqrt of e1-projected position variance) and
    the accel-bias mean offset (difference of the biased vs unbiased predicted position, projected on e1).

    sig_v_lat is the SWEPT key input. Returns dict with terminal sigmas + bias offset + the 2D in-plane
    miss quantiles (MC-sampled from the terminal Gaussian + bias)."""
    if sig_v_vert is None:
        sig_v_vert = sig_v_lat
    tr, idx_coast, idx_band, k_floor, rng_to_g4 = gate4_terminal_steps(v_race, r_floor, R_ACC_MAX)
    e1, e2 = MD._label_inplane_axes(tr["u34"])       # e1 lateral (binding), e2 vertical
    u = tr["u34"]
    s_R, s_ab, s_dt = tr["s_R"], tr["s_ab"], tr["s_dt"]

    # ---- build P0 in world NED from per-axis (pos,vel) sigmas in the (e1,e2,u) frame ----
    # position block (3x3) and velocity block (3x3) and pos-vel cross (3x3), each rotated to world.
    def axis_blocks(s_p_lat, s_p_vert, s_p_u, s_v_lat, s_v_vert, s_v_u, c_pv_lat):
        Pp = (s_p_lat**2 * np.outer(e1, e1) + s_p_vert**2 * np.outer(e2, e2)
              + s_p_u**2 * np.outer(u, u))
        Pv = (s_v_lat**2 * np.outer(e1, e1) + s_v_vert**2 * np.outer(e2, e2)
              + s_v_u**2 * np.outer(u, u))
        Ppv = c_pv_lat * np.outer(e1, e1)            # pos-vel cross only on lateral (others ~0)
        return Pp, Pv, Ppv
    # along-track pos/vel sigmas: loose (IMU+absolute own depth). Use along sigma ~0.5 m / 0.5 m/s.
    Pp, Pv, Ppv = axis_blocks(sig_p_lat, sig_p_vert, 0.5, sig_v_lat, sig_v_vert, 0.5, cov_pv_lat)
    P0 = np.block([[Pp, Ppv], [Ppv.T, Pv]])
    P0 = 0.5 * (P0 + P0.T) + 1e-9 * np.eye(6)

    # accel bias (in-plane, worst-case along the binding lateral axis e1)
    bmag = ME.att_deg_to_accel_bias(accel_bias_deg)
    if bmag <= 0:
        bias_world = np.zeros(3)
    elif bias_mode == "inplane":
        bias_world = bmag * e1                        # all phantom accel on the binding axis (conservative)
    else:
        bias_world = bmag * (e1 + e2) / np.sqrt(2)

    # ---- run TWO predict chains over the coast steps: covariance (P) + biased/unbiased mean ----
    kf = LinearKF(x=np.zeros(6), P=P0.copy())
    x_unb = np.zeros(6); x_bia = np.zeros(6)
    # seed both means at the same (arbitrary) start; only the DIFFERENCE matters for the bias offset.
    for k in idx_coast:
        R = s_R[k]; ab = s_ab[k]; dt = s_dt[k]
        # covariance: production predict (Q from true specific force). The mean it carries is irrelevant.
        kf.predict(ab, R, dt)
        # explicit mean integration (same F,B) for unbiased vs +bias (bias added in WORLD frame -> body)
        sf_w = R @ ab
        a_w = sf_w + GRAVITY_NED
        F = np.block([[np.eye(3), dt*np.eye(3)], [np.zeros((3,3)), np.eye(3)]])
        B = np.vstack([0.5*dt*dt*np.eye(3), dt*np.eye(3)])
        x_unb = F @ x_unb + B @ a_w
        x_bia = F @ x_bia + B @ (a_w + bias_world)
    Pf = kf.P
    # terminal in-plane lateral / vertical position sigma (project pos block onto e1 / e2)
    Ppos = Pf[:3, :3]
    sig_lat_f = float(np.sqrt(max(e1 @ Ppos @ e1, 0.0)))
    sig_vert_f = float(np.sqrt(max(e2 @ Ppos @ e2, 0.0)))
    sig_v_lat_f = float(np.sqrt(max(e1 @ Pf[3:, 3:] @ e1, 0.0)))
    # bias mean offset on the binding lateral axis
    dpos = (x_bia - x_unb)[:3]
    bias_off_lat = float(e1 @ dpos)
    bias_off_vert = float(e2 @ dpos)

    # 2D in-plane miss magnitude p90/p99: sample terminal Gaussian (lat,vert) + add bias mean offset
    rng = np.random.default_rng(SEED + int(round(sig_v_lat*1e3)) + int(round(accel_bias_deg*1e3)))
    n_s = 200_000
    lat = rng.normal(bias_off_lat, sig_lat_f, n_s)
    vert = rng.normal(bias_off_vert, sig_vert_f, n_s)
    mag = np.sqrt(lat**2 + vert**2)
    t_coast = float(np.sum(s_dt[idx_coast]))
    return dict(
        v_race=v_race, r_floor=r_floor, t_coast=t_coast, n_coast_steps=int(len(idx_coast)),
        sig_v_lat_in=sig_v_lat, sig_p_lat_in=sig_p_lat,
        sig_lat_terminal=sig_lat_f, sig_vert_terminal=sig_vert_f, sig_v_lat_terminal=sig_v_lat_f,
        accel_bias_deg=accel_bias_deg, bias_off_lat=bias_off_lat, bias_off_vert=bias_off_vert,
        miss_p50=float(np.percentile(mag, 50)), miss_p90=float(np.percentile(mag, 90)),
        miss_p99=float(np.percentile(mag, 99)),
    )


# =================================================================================================
# fly_lap_coast — ME.fly_lap_v2 with the accurate-band fix window + terminal coast. Returns the gate-4
# in-plane miss AND (deterministic-P pass) the KF covariance at the gate-4 r_floor crossing.
# =================================================================================================
def fly_lap_coast(rng, v_race, sigma_lat, sigma_vert, accel_bias_mag, bias_mode, fix_rate,
                  r_floor, r_acc_max=R_ACC_MAX, latency_ms=15.0, horizon_s=0.5,
                  capture_floor_cov=False, noise_off=False, lat_fix_bias_m=0.0, diag=False,
                  fix_corr_tau=0.0):
    """One cold lap g0->g4, fixes ONLY in each gate's accurate band [r_floor, r_acc_max], COAST inside
    r_floor. Mirrors margin_driver_v2.fly_lap_v2 except the fix-availability test. If capture_floor_cov,
    also returns the KF P (and the e1/e2 lateral/vert sigmas) at the FIRST gate-4 coast step. If
    noise_off, zero the per-fix + accel white noise (covariance-only deterministic pass). lat_fix_bias_m
    adds a constant per-fix lateral bias along e1 (residual chain bias; ~0 measured, swept for honesty).
    If diag, also return the ACTUAL lateral/vert estimate error at the r_floor coast-start (the truth the
    dense fix stream really delivers) so the covariance self-assessment can be checked."""
    tr = ME.get_truth(v_race)
    u34 = tr["u34"]
    g4 = ME.GATES[4]
    e1g4, e2g4 = MD._label_inplane_axes(u34)

    if accel_bias_mag <= 0.0:
        accel_bias_body = np.zeros(3)
    elif bias_mode == "inplane":
        e1, e2 = ME._inplane_basis(u34)
        ang = rng.uniform(0, 2*np.pi)
        d_world = np.cos(ang)*e1 + np.sin(ang)*e2
        accel_bias_body = tr["R_wb_g4"].T @ (accel_bias_mag * d_world)
    else:
        d = rng.normal(0, 1, 3); d /= (np.linalg.norm(d)+1e-12)
        accel_bias_body = accel_bias_mag * d

    p_init = tr["p_init_true"] + (0 if noise_off else rng.normal(0, 0.5, 3))
    v_init = tr["v_init_true"] + (0 if noise_off else rng.normal(0, 1.5, 3))
    kf = LinearKF.initialize(p_init, v_init, pos_std=1.0, vel_std=1.5)
    rk = RewindKF(kf=kf, horizon_s=horizon_s)

    s_p, s_v, s_R, s_ab = tr["s_p"], tr["s_v"], tr["s_R"], tr["s_ab"]
    s_dt, s_ts, s_tns = tr["s_dt"], tr["s_ts"], tr["s_tns"]
    s_e1, s_e2, s_u = tr["s_e1"], tr["s_e2"], tr["s_u"]
    n_steps = tr["n_steps"]
    # per-step approached-gate + range (replaces s_tgt's <12 m window with the accurate band)
    GP = {k: ME.GATES[k] for k in range(1, 5)}

    fix_queue = []
    fix_dt = 1.0 / (DETECTOR_HZ * fix_rate)
    next_fix_t = 0.0
    L_s = latency_ms / 1e3
    sig2_lat = sigma_lat**2; sig2_vert = sigma_vert**2; sig2_rad = RADIAL_SIGMA**2
    floor_cov = None
    captured = False
    diag_out = None
    # AR(1) temporally-correlated lateral/vert fix error (real PnP frame-to-frame correlation): the
    # REALISM lever. tau=0 -> iid (per-fix independent). tau>0 -> a correlated component that does NOT
    # average out, inflating the achieved sigma_v. We split the per-fix in-plane noise into a correlated
    # AR(1) state carried across fixes (advanced once per accepted fix). KF cov still uses the full sigma.
    e_corr_lat = 0.0; e_corr_vert = 0.0

    for k in range(n_steps):
        t = s_ts[k]; t_ns = int(s_tns[k])
        accel_meas = s_ab[k] + accel_bias_body + (0 if noise_off else rng.normal(0, ACCEL_NOISE_STD, 3))
        rk.predict(accel_meas, s_R[k], s_dt[k], t_ns)

        # capture the gate-4 covariance at the first coast step (range just inside r_floor)
        if (capture_floor_cov or diag) and not captured:
            rg4 = float(np.linalg.norm(s_p[k] - g4))
            if s_p[k, 0] > g4[0] - 0.5 and rg4 <= r_floor:
                P = rk.P
                floor_cov = dict(
                    range_m=rg4, t=t,
                    sig_p_lat=float(np.sqrt(max(e1g4 @ P[:3,:3] @ e1g4, 0.0))),
                    sig_p_vert=float(np.sqrt(max(e2g4 @ P[:3,:3] @ e2g4, 0.0))),
                    sig_v_lat=float(np.sqrt(max(e1g4 @ P[3:,3:] @ e1g4, 0.0))),
                    sig_v_vert=float(np.sqrt(max(e2g4 @ P[3:,3:] @ e2g4, 0.0))),
                    cov_pv_lat=float(e1g4 @ P[:3,3:] @ e1g4),
                    P=P.copy(),
                )
                captured = True
                if diag:
                    perr = rk.position - s_p[k]
                    verr = rk.velocity - s_v[k]
                    diag_out = dict(
                        range_m=rg4,
                        perr_lat=float(e1g4 @ perr), perr_vert=float(e2g4 @ perr),
                        verr_lat=float(e1g4 @ verr), verr_vert=float(e2g4 @ verr),
                    )

        # which gate is being approached, and is it in the ACCURATE band?
        tgt = -1; e1k = e2k = uk = None
        for gk_i in (1, 2, 3, 4):
            gk = GP[gk_i]
            rg = float(np.linalg.norm(gk - s_p[k]))
            if s_p[k, 0] > gk[0] - 1.0 and r_floor < rg <= r_acc_max:
                tgt = gk_i
                e1k_raw, e2k_raw, uk = s_e1[k], s_e2[k], s_u[k]
                if abs(e1k_raw[2]) > abs(e2k_raw[2]):
                    e1k, e2k = e2k_raw, e1k_raw
                else:
                    e1k, e2k = e1k_raw, e2k_raw
                break

        if tgt >= 0 and t >= next_fix_t:
            next_fix_t = t + fix_dt
            if noise_off:
                n_lat = np.zeros(3)
            elif fix_corr_tau > 0.0:
                # AR(1): rho from the actual fix interval. correlated part carried in e_corr_*.
                rho = float(np.exp(-fix_dt / fix_corr_tau))
                e_corr_lat = rho*e_corr_lat + np.sqrt(1-rho*rho)*rng.normal(0, sigma_lat)
                e_corr_vert = rho*e_corr_vert + np.sqrt(1-rho*rho)*rng.normal(0, sigma_vert)
                n_lat = (e_corr_lat*e1k + e_corr_vert*e2k + rng.normal(0, RADIAL_SIGMA)*uk)
            else:
                n_lat = (rng.normal(0, sigma_lat)*e1k + rng.normal(0, sigma_vert)*e2k
                         + rng.normal(0, RADIAL_SIGMA)*uk)
            z = s_p[k] + n_lat + lat_fix_bias_m * e1k
            cov = (sig2_lat*np.outer(e1k, e1k) + sig2_vert*np.outer(e2k, e2k)
                   + sig2_rad*np.outer(uk, uk))
            fix_queue.append((t + L_s, t_ns, z.copy(), cov.copy()))

        if fix_queue:
            fix_queue.sort(key=lambda e: e[0])
            while fix_queue and fix_queue[0][0] <= t + 1e-12:
                _, cap_ns, z, cov = fix_queue.pop(0)
                rk.update_position_at(cap_ns, z, cov)

    err = rk.position - tr["p_final_true"]
    along = float(np.dot(err, u34))
    inplane_vec = err - along * u34
    miss = float(np.linalg.norm(inplane_vec))
    if diag:
        d = dict(miss=miss, miss_lat=float(e1g4 @ inplane_vec), miss_vert=float(e2g4 @ inplane_vec))
        if diag_out:
            d.update(diag_out)
        return miss, d
    if capture_floor_cov:
        return miss, floor_cov
    return miss


# =================================================================================================
# A2 — FIX-STREAM CONDITIONING (deterministic covariance pass).  P is independent of measurement VALUES,
# so a single noise-off pass yields the achieved sigma_v_lat / sigma_p_lat at the gate-4 floor.
# =================================================================================================
def a2_achieved_sigma_v(v_race, sigma_lat, sigma_vert, fix_rate, r_floor, r_acc_max=R_ACC_MAX):
    rng = np.random.default_rng(SEED)
    _, fc = fly_lap_coast(rng, v_race, sigma_lat, sigma_vert, 0.0, "inplane", fix_rate,
                          r_floor, r_acc_max, capture_floor_cov=True, noise_off=True)
    return fc


# =================================================================================================
# A3 — FULL-LAP MC CLOSURE with bootstrap CI.
# =================================================================================================
def _bootstrap_ci(samples, q, n_boot=2000, alpha=0.10, seed=SEED):
    rng = np.random.default_rng(seed)
    m = np.asarray(samples)
    n = len(m)
    bq = np.empty(n_boot)
    for b in range(n_boot):
        bq[b] = np.percentile(m[rng.integers(0, n, n)], q)
    lo = float(np.percentile(bq, 100*alpha/2)); hi = float(np.percentile(bq, 100*(1-alpha/2)))
    return lo, hi


def a3_mc_cell(v_race, sigma_lat, sigma_vert, accel_bias_deg, bias_mode, fix_rate, r_floor,
               n_mc=600, n_boot=2000):
    accel_bias_mag = ME.att_deg_to_accel_bias(accel_bias_deg)
    misses = np.empty(n_mc)
    for s in range(n_mc):
        seed = (SEED + 101*s + int(round(v_race))*13 + int(round(accel_bias_deg*100))*7
                + int(round(sigma_lat*1000))*17 + int(round(fix_rate*1000))*23
                + int(round(r_floor*100))*131 + {"random3d": 0, "inplane": 1}[bias_mode]*3001)
        rng = np.random.default_rng(seed)
        misses[s] = fly_lap_coast(rng, v_race, sigma_lat, sigma_vert, accel_bias_mag, bias_mode,
                                  fix_rate, r_floor)
    p90 = float(np.percentile(misses, 90)); p99 = float(np.percentile(misses, 99))
    p90_lo, p90_hi = _bootstrap_ci(misses, 90, n_boot)
    p99_lo, p99_hi = _bootstrap_ci(misses, 99, n_boot)
    clears = {f"{r:.2f}": bool(p90 < M(r) and p99 < M(r)) for r in ME.RADIUS_BAND}
    clears_cih = {f"{r:.2f}": bool(p90_hi < M(r) and p99_hi < M(r)) for r in ME.RADIUS_BAND}
    return dict(
        v_race=v_race, sigma_lat=sigma_lat, sigma_vert=sigma_vert, accel_bias_deg=accel_bias_deg,
        bias_mode=bias_mode, fix_rate=fix_rate, r_floor=r_floor, n_mc=n_mc,
        p50=float(np.percentile(misses, 50)), p90=p90, p99=p99,
        p90_ci=[p90_lo, p90_hi], p99_ci=[p99_lo, p99_hi],
        rms=float(np.sqrt(np.mean(misses**2))), max=float(misses.max()),
        clears=clears, clears_ci_honest=clears_cih,
    )


if __name__ == "__main__":
    t0 = time.time()
    print("[coast-drift] in-plane basis (g3->g4):")
    _tr = ME.get_truth(30.0)
    _e1, _e2 = MD._label_inplane_axes(_tr["u34"])
    print(f"  u34={_tr['u34']}  e1(lat)={_e1} |z|={abs(_e1[2]):.3f}  e2(vert)={_e2} |z|={abs(_e2[2]):.3f}")
    # quick smoke of the three analyses
    a1 = a1_coast_propagate(30.0, sig_v_lat=0.30, accel_bias_deg=0.0)
    print(f"[A1 smoke] v30 sigv0.30 t_coast={a1['t_coast']:.3f}s steps={a1['n_coast_steps']} "
          f"-> sig_lat_term={a1['sig_lat_terminal']:.3f} miss_p90={a1['miss_p90']:.3f} p99={a1['miss_p99']:.3f}")
    fc = a2_achieved_sigma_v(30.0, SIGMA_LAT_POINTED, SIGMA_VERT_POINTED, FIXRATE_POINTED, R_FLOOR_DEFAULT)
    print(f"[A2 smoke] achieved at r_floor={fc['range_m']:.1f}m: sig_v_lat={fc['sig_v_lat']:.3f} "
          f"sig_p_lat={fc['sig_p_lat']:.3f} cov_pv_lat={fc['cov_pv_lat']:+.4f}")
    c = a3_mc_cell(30.0, SIGMA_LAT_POINTED, SIGMA_VERT_POINTED, 0.0, "inplane", FIXRATE_POINTED,
                   R_FLOOR_DEFAULT, n_mc=300, n_boot=500)
    print(f"[A3 smoke] v30 fr{FIXRATE_POINTED} b0 -> p90={c['p90']:.3f} p99={c['p99']:.3f} "
          f"clears r0.30={c['clears']['0.30']} CIhonest={c['clears_ci_honest']['0.30']}")
    print(f"[coast-drift] smoke done in {time.time()-t0:.1f}s")
