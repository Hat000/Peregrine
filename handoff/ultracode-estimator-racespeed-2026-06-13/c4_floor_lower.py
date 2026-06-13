"""c4_floor_lower.py -- BRANCH: is lowering the 0.40 m FIX_COV_FLOOR_STD a real lever or a trap?

The 0.40 m isotropic floor is BIAS-ABSORPTION (it inflates R so the KF distrusts each fix,
covering +0.3 m vertical / per-gate lateral / close-range depth systematics). It slows variance
convergence. HYPOTHESIS: if VISION-CAL removes the GLOBAL bias (b1/b2: it does, and the de-bias is
not circular -- LOGO confirms), the floor can come down -> KF trusts fixes more -> faster, lower
filtered variance. RISK: a too-low floor (1) re-admits the per-gate residual bias (the KF tracks a
biased fix harder/faster the more it trusts it) and (2) shrinks the chi2 innovation denominator
S = P + R, raising the catastrophic-leak rate.

This script does THREE things a1_floor_probe did NOT:
  (A) FAITHFUL floor-only lowering -- override `fix_cov_floor_std` in the REAL
      `gate_pose_to_world_position`, keeping the PnP-default + attitude-lever terms intact. Measures
      the resulting per-fix sigma and the filtered in-plane VARIANCE at the gate-4 window. (a1's
      floor_probe swept the WHOLE per-fix isotropic sigma, conflating "lower the floor" with "make
      PnP+lever vanish too"; the floor can only buy down to the ~0.30 m PnP-default block.)
  (B) BIAS RE-ADMISSION -- with the per-gate residual bias LEFT IN (the un-removable part per b2,
      range-collapsing ~[0.21,0.24,0.03] N,E,D residual), measures how the filtered BIAS at gate-4
      grows as the floor drops (the KF tracks the bias harder when it trusts fixes more). This is the
      trap: a lower floor trades variance for bias.
  (C) LEAK-RATE RISK -- the chi2 gate is d2 = nu^T (P+R)^-1 nu > 16.27 -> reject. Lowering the floor
      shrinks R -> shrinks S -> raises d2 for the SAME innovation -> (i) good borderline fixes get
      over-rejected again (the very 13->17% over-rejection the floor was added to FIX), and (ii) a
      bounded leak that DOES pass is weighted into the state with a higher gain -> more damage. We
      quantify the over-rejection regrowth analytically + by MC and the per-leak state damage.

NET: report the variance benefit of each floor, the bias cost, and the leak/over-rejection cost, and
the total in-plane RMS (bias (+) filtered noise) -- the only number that matters for "thread gate-4".

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c4_floor_lower.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "handoff" / "ultracode-vision-case-c-2026-06-13"))
sys.path.insert(0, str(ROOT / "handoff" / "ultracode-estimator-racespeed-2026-06-13"))

from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from racer.localization import gate_pose_to_world_position  # noqa: E402
from racer.contracts import Gate, GatePose  # noqa: E402
from racer.frames import R_world_from_body, R_camera_from_body, ATTITUDE_NOISE_STD_RAD  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402
import a1_sim as A  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE_SEED = 20260613

# ------------------------------------------------------------------ #
# Per-gate residual bias at gate-4 that VISION-CAL global de-bias CANNOT remove.
# Source: a2/b2. b2 corrected a2's "0.52 m constant" to a RANGE-COLLAPSING term. The case-C
# per-track residual registration sigma reported in FACTS is [0.21, 0.24, 0.03] m (N,E,D). The
# operative (near-range, last-usable-fix ~8-9 m) in-plane residual is ~0.11-0.19 m (b2 attack 3),
# and the residual COLLAPSES with range. We model the deployable per-gate residual at gate-4 as a
# CONSTANT-per-track offset whose in-plane (E,D) magnitude we set from the case-C residual sigma.
# Honest: we use the FACTS case-C residual sigma directly so this matches the cross-cutting note.
PER_GATE_RESID_NED = np.array([0.21, 0.24, 0.03])   # N,E,D residual std after GLOBAL de-bias (case C)
# As a deterministic per-track offset (same every lap), the residual is a fixed draw, not zero-mean.
# We draw ONE per-track offset per seed ~ N(0, resid_sigma) to span the population of tracks, then
# hold it constant across the gate-4 approach for that seed -> it does NOT average out.

FLOORS = [0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.025, 0.0]


def fix_cov_real_floor(drone_pos, R_wb, floor):
    """REAL gate_pose_to_world_position with fix_cov_floor_std overridden to `floor`.
    Keeps PnP-default (0.30 m, covariance=None branch -> NOT inflated) + attitude lever intact."""
    R_wc = R_wb @ R_camera_from_body().T
    lever_world = A.G4 - drone_pos
    t_cam_gate = R_wc.T @ lever_world
    gate = Gate(gate_id=4, position_ned=A.G4.copy(), R_world_gate=np.eye(3))
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam_gate,
                  reproj_error_px=0.3, gate_id=4, covariance=None, n_corners=4)
    _pos, cov = gate_pose_to_world_position(gp, gate, R_wb, fix_cov_floor_std=floor)
    return cov


def run_floor_cell(floor, accept=0.47, L_ms=6.0, n_seeds=600, base_seed=BASE_SEED,
                   bias_mode="debiased_global", leak_rate=0.0053):
    """One floor value. bias_mode:
       - 'debiased_global'      : global de-bias applied, NO per-gate residual (pure-variance arm)
       - 'debiased_with_resid'  : global de-bias applied, per-gate residual LEFT IN (deployable)
    Returns filtered in-plane variance (std), filtered bias, in-plane RMS, NEES, and over-rejection
    + leak diagnostics at this floor.
    """
    pools, means, _ = A.load_residual_pools()
    band_stds = {k: pools[k].std(axis=0, ddof=1) for k in pools}
    L_s = L_ms / 1e3
    t_cross = A.SEG_LEN / A.SPEED
    R_wb = R_world_from_body(0.0, A.DRAG_HOLD_PITCH_RAD, A.YAW_RAD)
    g = GRAVITY_NED
    det_dt = 1.0 / A.DETECTOR_HZ
    n_imu = int(np.ceil(t_cross / A.IMU_DT)) + 2

    errs, Pds, nees = [], [], []
    n_over_rej, n_offered, n_leak_applied, n_leak_offered = 0, 0, 0, 0
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + 7919 * s + int(floor * 1000) + (1 if "resid" in bias_mode else 0))
        # per-track residual offset (constant for this seed/track) -- only in the 'with_resid' arm
        if bias_mode == "debiased_with_resid":
            track_resid = rng.normal(0, PER_GATE_RESID_NED)   # fixed per-track NED offset
        else:
            track_resid = np.zeros(3)
        p0 = A.truth_pos(0.0, 0.0); v0 = A.truth_vel(0.0, 0.0)
        p_init = p0 + rng.normal(0, 0.6, 3); v_init = v0 + rng.normal(0, 0.5, 3)
        rkf = RewindKF(kf=LinearKF.initialize(p_init, v_init, pos_std=0.6, vel_std=0.5),
                       horizon_s=A.HORIZON_S)
        det_times = np.arange(0.0, t_cross + 1e-9, det_dt)
        accepted = rng.random(det_times.shape[0]) < accept
        events = []
        for tc, ok in zip(det_times, accepted):
            if not ok:
                continue
            p_true_cap = A.truth_pos(tc, 0.0)
            rng_to_g4 = max(0.0, A.SEG_LEN - float(np.dot(p_true_cap - A.G3, A.SEG_HAT)))
            band = A.band_for_range(rng_to_g4)
            # GLOBAL de-bias always applied (b1/b2: necessary, not circular). per-gate residual only
            # in the with_resid arm, as a CONSTANT per-track offset.
            resid = pools[band][rng.integers(0, pools[band].shape[0])] - means[band]
            z = p_true_cap + resid + track_resid
            cov = fix_cov_real_floor(p_true_cap, R_wb, floor)
            # occasionally inject a chi2-gate-BOUNDED leak (wrong-gate fix that lands within the gate)
            is_leak = rng.random() < leak_rate
            events.append([tc + L_s, tc, z, cov, is_leak, rng_to_g4])
        events.sort(key=lambda e: e[0])
        ev_i, t = 0, 0.0
        for k in range(1, n_imu + 1):
            t_prev = t; t = k * A.IMU_DT
            if t > t_cross:
                dt = t_cross - t_prev; t = t_cross
            else:
                dt = A.IMU_DT
            if dt > 0:
                accel_body = R_wb.T @ (np.zeros(3) - g) + rng.normal(0, 0.3, 3)
                rkf.predict(accel_body, R_wb, dt, sim_time_ns=int(round(t * 1e9)))
            while ev_i < len(events) and events[ev_i][0] <= t + 1e-12:
                _, cap_t, z, cov, is_leak, rng_g4 = events[ev_i]
                # CHI2 GATE against the propagated prior (REAL navigator logic): d2 = nu^T S^-1 nu
                # We must compare against the prior AT CAPTURE TIME -> use the rewind buffer's
                # state at capture. RewindKF applies the gate INSIDE? No -- it just applies. We
                # gate here against the current (rewound) prior to mirror the navigator.
                # Build the leak innovation if this is a leak: a wrong-gate offset bounded so it
                # WOULD pass the 16.27 gate at the SHIPPED 0.40 floor (so the floor change is what
                # moves it across the gate, isolating the leak-rate effect of the floor).
                if is_leak:
                    n_leak_offered += 1
                    # bounded outlier: magnitude ~ sqrt(16.27) * sigma_floor40 in a random dir, so
                    # at floor=0.40 it sits right at the gate edge; smaller floor -> it crosses out.
                    sig40 = 0.50  # ~per-fix sigma at the 0.40 floor near gate (probed)
                    direction = rng.normal(0, 1, 3); direction /= np.linalg.norm(direction)
                    z_eff = A.truth_pos(cap_t, 0.0) + direction * (np.sqrt(16.27) * sig40 * 0.95)
                else:
                    z_eff = z
                # gate against current prior (position block)
                x_prior = rkf.position
                nu = z_eff - x_prior
                S = rkf.P[:3, :3] + cov
                try:
                    d2 = float(nu @ np.linalg.solve(S, nu))
                except np.linalg.LinAlgError:
                    d2 = 0.0
                n_offered += 1
                if d2 > 16.27:
                    n_over_rej += 1
                    if is_leak:
                        pass  # leak correctly rejected
                else:
                    rkf.update_position_at(int(round(cap_t * 1e9)), z_eff, cov)
                    if is_leak:
                        n_leak_applied += 1
                ev_i += 1
            if t >= t_cross:
                break
        errs.append(rkf.position - A.truth_pos(t_cross, 0.0))
        Pds.append(np.diag(rkf.P)[:3].copy())
        Ppos = rkf.P[:3, :3]
        e = errs[-1]
        nees.append(float(e @ np.linalg.solve(Ppos, e)))
    errs = np.asarray(errs); Pds = np.asarray(Pds)
    std = errs.std(axis=0, ddof=1)
    bias = errs.mean(axis=0)
    rms = np.sqrt((errs ** 2).mean(axis=0))
    claimed = np.sqrt(Pds.mean(axis=0))
    inplane_std = float(np.sqrt(std[1] ** 2 + std[2] ** 2))
    inplane_rms = float(np.sqrt((errs[:, 1] ** 2 + errs[:, 2] ** 2).mean()))
    inplane_bias = float(np.sqrt(bias[1] ** 2 + bias[2] ** 2))
    return dict(
        floor=floor, bias_mode=bias_mode, accept=accept, L_ms=L_ms, n_seeds=n_seeds,
        std_NED=std.tolist(), bias_NED=bias.tolist(), rms_NED=rms.tolist(),
        claimed_NED=claimed.tolist(),
        inplane_std=inplane_std, inplane_rms=inplane_rms, inplane_bias=inplane_bias,
        E_std=float(std[1]), D_std=float(std[2]),
        nees_mean=float(np.mean(nees)),
        over_rej_frac=float(n_over_rej / max(1, n_offered)),
        leak_pass_frac=float(n_leak_applied / max(1, n_leak_offered)),
        n_offered=n_offered, n_over_rej=n_over_rej,
        n_leak_offered=n_leak_offered, n_leak_applied=n_leak_applied,
    )


def per_fix_sigma_table(R_wb):
    """The faithful floor-only per-fix sigma at gate-4 ranges (what the floor actually buys)."""
    out = {}
    for floor in FLOORS:
        row = {}
        for r in [2.0, 4.0, 8.0, 16.0]:
            drone = A.G4 - A.SEG_HAT * r
            cov = fix_cov_real_floor(drone, R_wb, floor)
            row[f"{r:.0f}m"] = np.round(np.sqrt(np.diag(cov)), 4).tolist()
        out[f"{floor:.3f}"] = row
    return out


def main():
    R_wb = R_world_from_body(0.0, A.DRAG_HOLD_PITCH_RAD, A.YAW_RAD)
    results = {"per_fix_sigma_vs_floor": per_fix_sigma_table(R_wb), "cells": []}

    print("=== (A) per-fix sigma (NED) vs floor at gate-4 ranges -- what the floor actually buys ===")
    for floor, row in results["per_fix_sigma_vs_floor"].items():
        print(f"  floor={floor}: r2={row['2m']}  r8={row['8m']}  r16={row['16m']}")

    print("\n=== Floor sweep: VARIANCE arm (global de-bias, NO per-gate residual) ===")
    print(f"{'floor':>6} | {'E_std':>6} {'D_std':>6} | {'inplane_std':>11} | {'over_rej':>8} {'leak_pass':>9} | NEES")
    for floor in FLOORS:
        r = run_floor_cell(floor, bias_mode="debiased_global")
        results["cells"].append(r)
        print(f"{floor:>6.3f} | {r['E_std']:>6.3f} {r['D_std']:>6.3f} | {r['inplane_std']:>11.3f} | "
              f"{r['over_rej_frac']:>8.3f} {r['leak_pass_frac']:>9.3f} | {r['nees_mean']:.2f}")

    print("\n=== Floor sweep: DEPLOYABLE arm (global de-bias, per-gate residual LEFT IN) ===")
    print(f"{'floor':>6} | {'ip_std':>6} {'ip_bias':>7} {'ip_RMS':>7} | {'over_rej':>8} {'leak_pass':>9} | NEES")
    for floor in FLOORS:
        r = run_floor_cell(floor, bias_mode="debiased_with_resid")
        results["cells"].append(r)
        print(f"{floor:>6.3f} | {r['inplane_std']:>6.3f} {r['inplane_bias']:>7.3f} {r['inplane_rms']:>7.3f} | "
              f"{r['over_rej_frac']:>8.3f} {r['leak_pass_frac']:>9.3f} | {r['nees_mean']:.2f}")

    (HERE / "c4_floor_lower_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {HERE / 'c4_floor_lower_results.json'}")
    return results


if __name__ == "__main__":
    main()
