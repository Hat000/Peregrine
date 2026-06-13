"""c2_cadence.py -- does MORE FIXES (higher cadence / fusing N fixes) close the gate-4 gap?

BRANCH (c2): multi-fix fusion / higher detector cadence.
QUESTION: averaging N independent zero-mean fixes reduces VARIANCE ~1/sqrt(N) but does NOTHING
to a consistent BIAS. Using the A1/A2 numbers and the REAL stack:
  (1) how many EFFECTIVE independent fixes land in the gate-4 approach,
  (2) what variance reduction is achievable by raising cadence above 30 Hz,
  (3) is it enough to bring the DE-BIASED in-plane sigma under 0.05 m,
  (4) and if the binding term is BIAS (A1/A2), does cadence touch the validity question at all.

Composes (does NOT re-implement):
  - racer.state_estimator.LinearKF                  (predict / update_position, Joseph form)
  - handoff/.../kf_rewind_buffer.RewindKF           (OOSM rewind/replay)
  - racer.localization.gate_pose_to_world_position  (REAL fix-cov: 0.40 floor + 1.4deg lever + PnP)

Builds on a1_sim's physics (same geometry, attitude, IMU model, residual pools) but the
SWEEP AXIS is DETECTOR CADENCE (15..240 Hz) instead of latency, and it adds the analytic
1/sqrt(N) reference + an effective-independent-fix count derived from the KF's own averaging.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c2_cadence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from racer.localization import gate_pose_to_world_position  # noqa: E402
from racer.contracts import Gate, GatePose  # noqa: E402
from racer.frames import (  # noqa: E402
    R_world_from_body,
    R_camera_from_body,
    ATTITUDE_NOISE_STD_RAD,
)
from kf_rewind_buffer import RewindKF  # noqa: E402

HERE = Path(__file__).resolve().parent
PERCEPTION = ROOT / "handoff" / "perception-char-2026-06-08" / "characterize_course_60s.json"

# ----------------------------------------------------------------------------- #
# Geometry (track_map, FACTS) -- gate-4 window  (identical to a1_sim)
# ----------------------------------------------------------------------------- #
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])
SEG = G4 - G3
SEG_LEN = float(np.linalg.norm(SEG))       # ~24.4 m
SEG_HAT = SEG / SEG_LEN
SPEED = 37.0                                # m/s post-gate-3 cruise

DRAG_ACCEL = 0.21 * SPEED
DRAG_HOLD_PITCH_RAD = float(np.arctan2(DRAG_ACCEL, GRAVITY_NED[2]))   # ~38.4 deg
YAW_RAD = float(np.arctan2(SEG_HAT[1], SEG_HAT[0]))

IMU_HZ = 90.0
IMU_DT = 1.0 / IMU_HZ

# ----------------------------------------------------------------------------- #
# THE SWEEP: detector cadence (Hz). 30 Hz = inherited DiffAero default; everything
# above is the "higher cadence" branch question. We also report effective fix rate
# = cadence * acceptance. Acceptance held at the canonical 47% AND an idealized 100%.
# ----------------------------------------------------------------------------- #
CADENCE_HZ = [15.0, 30.0, 60.0, 90.0, 120.0, 180.0, 240.0]
ACCEPT_CELLS = {"47pct": 0.47, "100pct": 1.0}
ERROR_MODELS = ["raw", "debiased"]
# Latency held at edge p50 (6 ms) -- a4/a1 show latency is ~all along-track at gate-4 and
# barely touches the in-plane variance at edge L; we want the cadence effect isolated.
L_MS = 6.0
HORIZON_S = 0.5
N_SEEDS = 600
BASE_SEED = 20260613


def load_residual_pools():
    d = json.loads(PERCEPTION.read_text())
    rows = d["rows"]
    bands = {"[0,8)": [], "[8,16)": [], "[16,24)": []}
    for r in rows:
        o = r.get("off_ned")
        rg = r.get("range_m")
        if o is None or rg is None:
            continue
        o = np.asarray(o, dtype=np.float64)
        if np.linalg.norm(o) >= 3.0:
            continue
        if 0.0 <= rg < 8.0:
            bands["[0,8)"].append(o)
        elif 8.0 <= rg < 16.0:
            bands["[8,16)"].append(o)
        elif 16.0 <= rg < 24.0:
            bands["[16,24)"].append(o)
    pools = {k: np.asarray(v) for k, v in bands.items() if len(v) > 0}
    means = {k: v.mean(axis=0) for k, v in pools.items()}
    stds = {k: v.std(axis=0, ddof=1) for k, v in pools.items()}
    return pools, means, stds


def band_for_range(rng_to_g4: float) -> str:
    if rng_to_g4 < 8.0:
        return "[0,8)"
    return "[8,16)"


def fix_cov_real(drone_pos: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """REAL fix covariance via racer.localization.gate_pose_to_world_position (covariance=None
    -> 0.3 m PnP default + 1.4deg attitude lever + 0.40 m floor). Identical to a1_sim."""
    R_wc = R_wb @ R_camera_from_body().T
    lever_world = G4 - drone_pos
    t_cam_gate = R_wc.T @ lever_world
    gate = Gate(gate_id=4, position_ned=G4.copy(), R_world_gate=np.eye(3))
    gp = GatePose(
        frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
        t_cam_gate=t_cam_gate, reproj_error_px=0.3, gate_id=4,
        covariance=None, n_corners=4,
    )
    _pos, cov = gate_pose_to_world_position(gp, gate, R_wb)
    return cov


def truth_pos(t: float) -> np.ndarray:
    return G3 + SEG_HAT * (SPEED * t)


def truth_vel(t: float) -> np.ndarray:
    return SEG_HAT * SPEED


def run_cell(cadence_hz: float, accept: float, error_model: str, pools, means,
             n_seeds: int, base_seed: int):
    """Monte-Carlo one (cadence, acceptance, error-model) cell at edge latency.
    Returns per-axis stats at the gate-4 plane crossing (rewind arm = deployed path)
    plus the count of fixes that actually landed before the crossing."""
    L_s = L_MS / 1e3
    t_cross = SEG_LEN / SPEED
    R_wb = R_world_from_body(0.0, DRAG_HOLD_PITCH_RAD, YAW_RAD)
    g = GRAVITY_NED
    det_dt = 1.0 / cadence_hz
    debias = (error_model == "debiased")

    err_rw = []
    P_rw_diag = []
    nees_rw = []
    n_fix_applied = []     # fixes actually fused before the crossing (this seed)
    per_fix_sig_E = []     # per-fix 1-sigma E assigned (sanity that cadence doesn't change it)
    per_fix_sig_D = []

    n_imu = int(np.ceil(t_cross / IMU_DT)) + 2

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + 1009 * s + int(cadence_hz) * 17
                                    + int(accept * 100) * 7 + (1 if debias else 0))
        p0 = truth_pos(0.0)
        v0 = truth_vel(0.0)
        pos_std0, vel_std0 = 0.6, 0.5
        p_init = p0 + rng.normal(0, pos_std0, 3)
        v_init = v0 + rng.normal(0, vel_std0, 3)

        kf_rw_inner = LinearKF.initialize(p_init, v_init, pos_std=pos_std0, vel_std=vel_std0)
        rkf = RewindKF(kf=kf_rw_inner, horizon_s=HORIZON_S)

        det_times = np.arange(0.0, t_cross + 1e-9, det_dt)
        accepted = rng.random(det_times.shape[0]) < accept
        fix_apply_events = []
        applied_sig = []
        for tc, ok in zip(det_times, accepted):
            if not ok:
                continue
            p_true_cap = truth_pos(tc)
            rng_to_g4 = SEG_LEN - float(np.dot(p_true_cap - G3, SEG_HAT))
            band = band_for_range(max(0.0, rng_to_g4))
            pool = pools[band]
            resid = pool[rng.integers(0, pool.shape[0])].copy()
            if debias:
                resid = resid - means[band]
            z = p_true_cap + resid
            cov = fix_cov_real(p_true_cap, R_wb)
            apply_t = tc + L_s
            fix_apply_events.append([apply_t, tc, z, cov])
            applied_sig.append((float(np.sqrt(cov[1, 1])), float(np.sqrt(cov[2, 2]))))
        fix_apply_events.sort(key=lambda e: e[0])
        ev_i = 0
        n_applied = 0

        t = 0.0
        for k in range(1, n_imu + 1):
            t_prev = t
            t = k * IMU_DT
            if t > t_cross:
                dt = t_cross - t_prev
                t = t_cross
            else:
                dt = IMU_DT
            if dt > 0:
                a_world = np.zeros(3)              # const-v: zero world accel
                sf_world = a_world - g
                accel_body_true = R_wb.T @ sf_world
                accel_body_meas = accel_body_true + rng.normal(0, kf_rw_inner.accel_noise_std, 3)
                t_ns = int(round(t * 1e9))
                rkf.predict(accel_body_meas, R_wb, dt, sim_time_ns=t_ns)
            while ev_i < len(fix_apply_events) and fix_apply_events[ev_i][0] <= t + 1e-12:
                apply_t, cap_t, z, cov = fix_apply_events[ev_i]
                t_fix_ns = int(round(cap_t * 1e9))
                res = rkf.update_position_at(t_fix_ns, z, cov)
                if res.applied:
                    n_applied += 1
                ev_i += 1
            if t >= t_cross:
                break

        p_true_cross = truth_pos(t_cross)
        e_rw = rkf.position - p_true_cross
        err_rw.append(e_rw)
        P_rw_diag.append(np.diag(rkf.P)[:3].copy())
        Ppos_rw = rkf.P[:3, :3]
        nees_rw.append(float(e_rw @ np.linalg.solve(Ppos_rw, e_rw)))
        n_fix_applied.append(n_applied)
        if applied_sig:
            arr = np.asarray(applied_sig)
            per_fix_sig_E.append(float(arr[:, 0].mean()))
            per_fix_sig_D.append(float(arr[:, 1].mean()))

    err_rw = np.asarray(err_rw)
    P_rw_diag = np.asarray(P_rw_diag)

    bias = err_rw.mean(axis=0)
    rms = np.sqrt((err_rw ** 2).mean(axis=0))
    std = err_rw.std(axis=0, ddof=1)
    claimed_sigma = np.sqrt(P_rw_diag.mean(axis=0))
    inplane_miss = np.sqrt(err_rw[:, 1] ** 2 + err_rw[:, 2] ** 2)
    inplane_rms = float(np.sqrt((inplane_miss ** 2).mean()))
    inplane_std = float(np.sqrt(std[1] ** 2 + std[2] ** 2))
    inplane_claimed = float(np.sqrt(claimed_sigma[1] ** 2 + claimed_sigma[2] ** 2))

    out = dict(
        cadence_hz=cadence_hz, accept=accept, error_model=error_model,
        eff_fix_hz=float(cadence_hz * accept),
        t_cross_s=float(t_cross), n_seeds=n_seeds,
        mean_fixes_applied=float(np.mean(n_fix_applied)),
        bias_NED=bias.tolist(),
        rms_NED=rms.tolist(),
        std_NED=std.tolist(),                       # pure scatter (the VARIANCE term)
        claimed_sigma_NED=claimed_sigma.tolist(),
        per_fix_sigma_E=float(np.mean(per_fix_sig_E)) if per_fix_sig_E else None,
        per_fix_sigma_D=float(np.mean(per_fix_sig_D)) if per_fix_sig_D else None,
        inplane_rms=inplane_rms,
        inplane_std=inplane_std,                    # VARIANCE-only in-plane (debiased = this)
        inplane_claimed_sigma=inplane_claimed,
        inplane_bias=float(np.sqrt(bias[1] ** 2 + bias[2] ** 2)),
        nees_mean=float(np.mean(nees_rw)),
        clears_005_inplane_std=bool(inplane_std < 0.05),
        clears_005_inplane_rms=bool(inplane_rms < 0.05),
    )
    return out


def effective_independent_fixes(rows_debiased):
    """From the DEBIASED arm (pure noise), back out the EFFECTIVE independent-fix count N_eff
    that the KF actually achieves, two ways:
      (a) from variance reduction:  N_eff = (per_fix_sigma / filtered_sigma)^2  per axis.
      (b) compare to the analytic ideal 1/sqrt(N) over the count of fixes that landed.
    This is the crux number: how much averaging the cadence buys vs the ideal."""
    out = []
    for r in rows_debiased:
        sigE_fix = r["per_fix_sigma_E"]
        sigD_fix = r["per_fix_sigma_D"]
        sigE_filt = r["std_NED"][1]
        sigD_filt = r["std_NED"][2]
        neff_E = (sigE_fix / sigE_filt) ** 2 if sigE_filt > 0 else float("nan")
        neff_D = (sigD_fix / sigD_filt) ** 2 if sigD_filt > 0 else float("nan")
        out.append(dict(
            cadence_hz=r["cadence_hz"], accept=r["accept"],
            eff_fix_hz=r["eff_fix_hz"],
            mean_fixes_applied=r["mean_fixes_applied"],
            per_fix_sigma_E=sigE_fix, per_fix_sigma_D=sigD_fix,
            filtered_sigma_E=sigE_filt, filtered_sigma_D=sigD_filt,
            N_eff_from_variance_E=neff_E, N_eff_from_variance_D=neff_D,
            N_eff_mean=float(np.nanmean([neff_E, neff_D])),
            # ideal 1/sqrt(N) over the landed fix count, for the SAME per-fix sigma:
            ideal_filtered_sigma_E=sigE_fix / np.sqrt(max(r["mean_fixes_applied"], 1.0)),
            kf_vs_ideal_ratio_E=(sigE_filt / (sigE_fix / np.sqrt(max(r["mean_fixes_applied"], 1.0))))
            if sigE_fix else float("nan"),
        ))
    return out


def main():
    np.seterr(all="raise")
    pools, means, stds = load_residual_pools()

    cells = []
    for accept_name, accept in ACCEPT_CELLS.items():
        for cad in CADENCE_HZ:
            for em in ERROR_MODELS:
                res = run_cell(cad, accept, em, pools, means, N_SEEDS, BASE_SEED)
                res["cell_id"] = f"{cad:.0f}Hz|{accept_name}|{em}"
                cells.append(res)
                print(f"{res['cell_id']:>22} | eff {res['eff_fix_hz']:6.1f}Hz | "
                      f"fixes {res['mean_fixes_applied']:5.1f} | "
                      f"perfix sigE {str(round(res['per_fix_sigma_E'],3)) if res['per_fix_sigma_E'] else '--':>5} | "
                      f"filt std E/D {res['std_NED'][1]:.3f}/{res['std_NED'][2]:.3f} | "
                      f"inplane std {res['inplane_std']:.3f} | rms {res['inplane_rms']:.3f} | "
                      f"bias {res['inplane_bias']:.3f} | NEES {res['nees_mean']:.2f} | "
                      f"clrStd {int(res['clears_005_inplane_std'])}")

    # effective-independent-fix analysis from the DEBIASED, 100%-accept arm (cleanest)
    deb_100 = [c for c in cells if c["error_model"] == "debiased" and c["accept"] == 1.0]
    deb_47 = [c for c in cells if c["error_model"] == "debiased" and c["accept"] == 0.47]
    neff_100 = effective_independent_fixes(deb_100)
    neff_47 = effective_independent_fixes(deb_47)

    # The headline asymptote question: extrapolate the variance floor as cadence -> infinity.
    # Fit filtered_inplane_std vs 1/sqrt(eff_fix_hz) for the debiased 100% arm to find the
    # cadence-saturation asymptote (process-noise growth between fixes caps the averaging).
    x = np.array([c["eff_fix_hz"] for c in deb_100])
    y = np.array([c["inplane_std"] for c in deb_100])
    # model: y^2 = a + b / x  (variance floor a as rate->inf, plus 1/rate averaging term b/x)
    A = np.vstack([np.ones_like(x), 1.0 / x]).T
    coef, *_ = np.linalg.lstsq(A, y ** 2, rcond=None)
    var_floor_inf = float(coef[0])              # variance as cadence->inf
    sigma_floor_inf = float(np.sqrt(max(var_floor_inf, 0.0)))

    out = dict(
        meta=dict(
            branch="c2 multi-fix fusion / higher cadence",
            geometry=dict(g3=G3.tolist(), g4=G4.tolist(), seg_len_m=SEG_LEN,
                          speed_mps=SPEED, t_cross_s=SEG_LEN / SPEED,
                          drag_hold_pitch_deg=float(np.rad2deg(DRAG_HOLD_PITCH_RAD))),
            imu_hz=IMU_HZ, latency_ms=L_MS, horizon_s=HORIZON_S,
            cadence_hz=CADENCE_HZ, accept_cells=ACCEPT_CELLS,
            attitude_noise_std_rad=float(ATTITUDE_NOISE_STD_RAD),
            n_seeds=N_SEEDS, base_seed=BASE_SEED,
            residual_bands={k: dict(n=int(v.shape[0]), bias=means[k].tolist(),
                                    std=stds[k].tolist()) for k, v in pools.items()},
            note=("Sweep axis = DETECTOR CADENCE at fixed edge L=6 ms. DEBIASED arm = pure "
                  "VARIANCE (zero-mean); RAW arm carries the measured per-fix bias. "
                  "N_eff = (per_fix_sigma/filtered_sigma)^2. Cadence->inf asymptote from "
                  "y^2 = a + b/eff_rate fit (a = process-noise-limited variance floor)."),
        ),
        cells=cells,
        effective_fix_analysis_100pct=neff_100,
        effective_fix_analysis_47pct=neff_47,
        cadence_saturation=dict(
            fit_model="inplane_var = a + b/eff_fix_hz (debiased 100% arm)",
            a_var_floor=var_floor_inf, b_rate_term=float(coef[1]),
            sigma_floor_as_cadence_inf_m=sigma_floor_inf,
            clears_005_even_at_infinite_cadence=bool(sigma_floor_inf < 0.05),
        ),
    )
    (HERE / "c2_cadence_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'c2_cadence_results.json'}  ({len(cells)} cells)")
    print(f"\nCADENCE->INF in-plane VARIANCE floor (debiased 100%): "
          f"sigma = {sigma_floor_inf:.3f} m  (clears 0.05 m: "
          f"{sigma_floor_inf < 0.05})")
    return out


if __name__ == "__main__":
    main()
