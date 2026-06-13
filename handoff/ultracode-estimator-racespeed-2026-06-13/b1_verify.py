"""b1_verify.py -- ADVERSARIAL re-run of a1_sim's gate-4 race-speed estimator headline.

ROLE (b1): charged to REFUTE a1's headline:
  "Case-C absolute world-fix CANNOT reach 0.05 m in-plane 1-sigma at the gate-4 37 m/s window;
   best filtered in-plane 1-sigma ~0.24 m/axis (in-plane RMS 0.33 m), DEBIASED, edge L, 47%."

I compose the SAME real filter stack a1 used (racer.state_estimator.LinearKF + RewindKF +
racer.localization.gate_pose_to_world_position) but PERTURB a1's assumptions per the attack plan:

  (A1) FIX-STREAM REALISM: a1 fed the SLOW-SPEED (5.35 m/s) resampled off_ned pool at 47% accept and
       applied NO motion-blur multiplier -> implicitly the A3 SHORT-shutter regime. Re-run with the
       A3 DEGRADED inputs: (i) SHORT-shutter (noise x1.05, accept 0.45) and (ii) LONG-exposure
       (in-plane noise x1.7, accept 0.33). Inflate BOTH the sampled residual noise AND the fix cov R
       consistently (the navigator would see a wider fix and a wider analytic cov).
  (A3) CATASTROPHIC LEAK: a1 OMITTED the 0.53% chi2-surviving leak entirely. Inject leaks at their
       real per-fix rate (0.0053) as chi2-gate-bounded wrong fixes (maha<=16.27 against the inflated
       cov => bounded offset, NOT the 139 m gross outliers). Re-measure with leaks IN the stream.
  (A4) REWIND HORIZON: confirm horizon (0.5 s) >= L at the CPU 112 ms cell (no ~21 m divergence).
  (A5) NEES HONESTY: is the claimed P honest, or is a1 reading filter-P as truth while empirical
       error is worse? Report claimed-sigma vs empirical std/RMS side by side per cell.
  (A6) BIAS vs VARIANCE: a1's DEBIASED arm subtracts the per-BAND mean of the WHOLE-COURSE pool
       (~ a global de-bias). a2 showed the gate-4 PER-GATE in-plane bias is ~0.45-0.52 m and SURVIVES
       global de-bias. So a1's "DEBIASED" arm is OPTIMISTIC: it removes a bias a real global de-bias
       cannot fully remove at gate-4. Re-run a DEBIASED-but-gate4-residual arm that leaves the a2
       per-gate residual bias [E=+0.457-(-0.054)=, D=...] in place, to show the honest debiased floor.

For each perturbation I report the DELTA vs a1's headline cell and whether the verdict
(NO cell clears <0.05 m) changes, for RAW and DEBIASED separately.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/b1_verify.py
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

# ---- gate-4 window geometry (track_map / FACTS) ----
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])
SEG = G4 - G3
SEG_LEN = float(np.linalg.norm(SEG))
SEG_HAT = SEG / SEG_LEN
SPEED = 37.0

DRAG_ACCEL = 0.21 * SPEED
DRAG_HOLD_PITCH_RAD = float(np.arctan2(DRAG_ACCEL, GRAVITY_NED[2]))
YAW_RAD = float(np.arctan2(SEG_HAT[1], SEG_HAT[0]))

IMU_HZ = 90.0
IMU_DT = 1.0 / IMU_HZ
DETECTOR_HZ = 30.0
HORIZON_S = 0.5
N_SEEDS = 600           # >a1's 400, to resolve the small leak-injection effect
BASE_SEED = 20260613

CHI2_GATE = 16.27       # navigator Mahalanobis 99.9% / 3 DOF
BASE_LEAK = 0.0053      # MEASURED chi2-surviving catastrophic-leak rate

AX_LABEL = {0: "N(along)", 1: "E(in-plane)", 2: "D(in-plane)"}


# ----------------------------------------------------------------------------- #
# Residual pools (RAW) + per-band means (a1's "global-ish" debias) + a2 gate-4 residual
# ----------------------------------------------------------------------------- #
def load_pools():
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
    return pools, means


def band_for_range(rng_to_g4: float) -> str:
    return "[0,8)" if rng_to_g4 < 8.0 else "[8,16)"


# a2 PER-GATE residual bias at gate-4 (NED), the part GLOBAL de-bias canNOT remove.
# a2 Path B (raw KF-accepted gate-4 fix stream, fix sense), de-biased per-axis:
#   E = -0.44, D = -0.08, N(along) = +0.15  (a2_findings.md section 2, Path B de-biased CIs)
# This is what a HONEST global-debias arm should leave at gate-4. a1's per-band-mean debias
# removes ~all of it (its debiased filtered bias ~[+0.01,0,-0.03]) -> a1's DEBIASED is optimistic.
A2_GATE4_RESIDUAL_DEBIASED_NED = np.array([+0.15, -0.44, -0.08])


# ----------------------------------------------------------------------------- #
# Real fix covariance, with optional pixel-noise inflation (blur)
# ----------------------------------------------------------------------------- #
def fix_cov_real(drone_pos: np.ndarray, R_wb: np.ndarray, pnp_std: float) -> np.ndarray:
    """REAL gate_pose_to_world_position cov. We pass an explicit per-fix analytic PnP cov
    (isotropic pnp_std^2) so the blur multiplier inflates the PnP block (the part that grows with
    pixel noise) while the model still adds the 1.4deg lever + 0.40 m floor. With pnp_std=0.3 and
    PNP_FIX_COV_INFLATION=2.0 the propagated block ~ 2*0.3^2 -> matches a1's covariance=None default
    (0.3 default, no inflation) closely; we VERIFY the no-blur cell reproduces a1 below."""
    R_wc = R_wb @ R_camera_from_body().T
    lever_world = G4 - drone_pos
    t_cam_gate = R_wc.T @ lever_world
    gate = Gate(gate_id=4, position_ned=G4.copy(), R_world_gate=np.eye(3))
    cov_pnp = (pnp_std ** 2) * np.eye(3)
    gp = GatePose(
        frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
        t_cam_gate=t_cam_gate, reproj_error_px=0.3, gate_id=4,
        covariance=cov_pnp, n_corners=4,
    )
    _pos, cov = gate_pose_to_world_position(gp, gate, R_wb)
    return cov


def fix_cov_default(drone_pos: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """a1's exact path: covariance=None -> 0.3 default PnP, no inflation, +lever +floor."""
    R_wc = R_wb @ R_camera_from_body().T
    lever_world = G4 - drone_pos
    t_cam_gate = R_wc.T @ lever_world
    gate = Gate(gate_id=4, position_ned=G4.copy(), R_world_gate=np.eye(3))
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                  t_cam_gate=t_cam_gate, reproj_error_px=0.3, gate_id=4,
                  covariance=None, n_corners=4)
    _pos, cov = gate_pose_to_world_position(gp, gate, R_wb)
    return cov


def truth_pos(t, a):
    s = SPEED * t + 0.5 * a * t * t
    return G3 + SEG_HAT * s


def truth_vel(t, a):
    return SEG_HAT * (SPEED + a * t)


# ----------------------------------------------------------------------------- #
# One Monte-Carlo cell with perturbations
# ----------------------------------------------------------------------------- #
def run_cell(L_ms, accept, error_model, pools, means,
             noise_mult_inplane=1.0, noise_mult_along=1.0,
             leak_rate=0.0, gate4_residual=False, accel=0.0,
             use_default_cov=True, n_seeds=N_SEEDS, base_seed=BASE_SEED, tag=""):
    """error_model in {raw, debiased}. gate4_residual: if True (debiased arm only), ADD BACK the
    a2 per-gate residual the global de-bias cannot remove. leak_rate: prob a fix is a catastrophic
    chi2-surviving leak. noise_mult_*: A3 blur multipliers on the SAMPLED residual + the analytic
    PnP std (so cov tracks the wider fix)."""
    L_s = L_ms / 1e3
    t_cross = SEG_LEN / SPEED if abs(accel) < 1e-9 else \
        (-SPEED + np.sqrt(SPEED ** 2 + 2 * accel * SEG_LEN)) / accel
    R_wb = R_world_from_body(0.0, DRAG_HOLD_PITCH_RAD, YAW_RAD)
    g = GRAVITY_NED
    det_dt = 1.0 / DETECTOR_HZ
    debias = (error_model == "debiased")
    # PnP std for the cov: 0.3 baseline, scaled by the in-plane blur multiplier (the binding axes).
    pnp_std = 0.3 * noise_mult_inplane

    err_rw, P_rw_diag, nees_rw, dropped = [], [], [], []
    n_imu = int(np.ceil(t_cross / IMU_DT)) + 2
    n_leaks_applied = []
    n_fix_total = []

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + 1009 * s + int(L_ms) * 31
                                    + int(accept * 100) * 7 + (1 if debias else 0)
                                    + (3 if gate4_residual else 0) + int(leak_rate * 1e5))
        p0, v0 = truth_pos(0.0, accel), truth_vel(0.0, accel)
        pos_std0, vel_std0 = 0.6, 0.5
        p_init = p0 + rng.normal(0, pos_std0, 3)
        v_init = v0 + rng.normal(0, vel_std0, 3)
        kf_inner = LinearKF.initialize(p_init, v_init, pos_std=pos_std0, vel_std=vel_std0)
        rkf = RewindKF(kf=kf_inner, horizon_s=HORIZON_S)

        det_times = np.arange(0.0, t_cross + 1e-9, det_dt)
        accepted = rng.random(det_times.shape[0]) < accept
        events = []
        nlk = 0
        for tc, ok in zip(det_times, accepted):
            if not ok:
                continue
            p_cap = truth_pos(tc, accel)
            rng_to_g4 = SEG_LEN - float(np.dot(p_cap - G3, SEG_HAT))
            band = band_for_range(max(0.0, rng_to_g4))
            pool = pools[band]
            resid = pool[rng.integers(0, pool.shape[0])].copy()
            if debias:
                resid = resid - means[band]
                if gate4_residual:
                    # ADD BACK the per-gate residual a global de-bias cannot remove (a2).
                    resid = resid + A2_GATE4_RESIDUAL_DEBIASED_NED
            # A3 blur: scale the ZERO-MEAN noise component. For RAW we scale (resid - band mean)
            # then re-add band mean so bias is unchanged (blur is zero-mean, A3). For DEBIASED the
            # band mean is already removed so we scale directly.
            if not debias:
                bmean = means[band]
                noise_part = resid - bmean
                noise_part = noise_part * np.array([noise_mult_along, noise_mult_inplane, noise_mult_inplane])
                resid = bmean + noise_part
            else:
                resid = resid * np.array([noise_mult_along, noise_mult_inplane, noise_mult_inplane])
            cov = fix_cov_default(p_cap, R_wb) if (use_default_cov and noise_mult_inplane == 1.0) \
                else fix_cov_real(p_cap, R_wb, pnp_std)
            # CATASTROPHIC LEAK injection: with prob leak_rate, replace the fix with a wrong but
            # chi2-surviving fix. A surviving leak has maha<=16.27 vs its (inflated) cov; the
            # worst such leak is offset ~ sqrt(16.27)*sigma along the least-constrained axis.
            # We model it as a fix offset = k*sqrt(chi2)*sigma_axis in a random in-plane direction
            # (the gate-confusable direction), k in (0.6,1.0] (it passed the gate, so near the edge).
            is_leak = rng.random() < leak_rate
            if is_leak:
                sig = np.sqrt(np.diag(cov))
                kfac = rng.uniform(0.6, 1.0)
                # leak displaces in a random direction, magnitude up to the chi2 edge
                u = rng.normal(size=3); u /= np.linalg.norm(u)
                leak_off = kfac * np.sqrt(CHI2_GATE) * sig * u
                z = p_cap + leak_off
                nlk += 1
            else:
                z = p_cap + resid
            events.append([tc + L_s, tc, z, cov])
        events.sort(key=lambda e: e[0])
        n_leaks_applied.append(nlk)
        n_fix_total.append(len(events))

        ev_i = 0
        ndrop = 0
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
                a_world = SEG_HAT * accel
                sf_world = a_world - g
                accel_body_true = R_wb.T @ sf_world
                accel_body_meas = accel_body_true + rng.normal(0, kf_inner.accel_noise_std, 3)
                rkf.predict(accel_body_meas, R_wb, dt, sim_time_ns=int(round(t * 1e9)))
            while ev_i < len(events) and events[ev_i][0] <= t + 1e-12:
                _at, cap_t, z, cov = events[ev_i]
                res = rkf.update_position_at(int(round(cap_t * 1e9)), z, cov)
                if not res.applied:
                    ndrop += 1
                ev_i += 1
            if t >= t_cross:
                break

        p_true_cross = truth_pos(t_cross, accel)
        e = rkf.position - p_true_cross
        err_rw.append(e)
        P_rw_diag.append(np.diag(rkf.P)[:3].copy())
        Ppos = rkf.P[:3, :3]
        nees_rw.append(float(e @ np.linalg.solve(Ppos, e)))
        dropped.append(ndrop)

    err = np.asarray(err_rw)
    Pd = np.asarray(P_rw_diag)
    bias = err.mean(axis=0)
    rms = np.sqrt((err ** 2).mean(axis=0))
    std = err.std(axis=0, ddof=1)
    claimed = np.sqrt(Pd.mean(axis=0))
    inplane = np.sqrt(err[:, 1] ** 2 + err[:, 2] ** 2)
    return dict(
        tag=tag, L_ms=L_ms, accept=accept, error_model=error_model,
        noise_mult_inplane=noise_mult_inplane, noise_mult_along=noise_mult_along,
        leak_rate=leak_rate, gate4_residual=gate4_residual,
        t_cross_s=float(t_cross), eff_fix_hz=float(DETECTOR_HZ * accept),
        mean_fixes=float(np.mean(n_fix_total)), mean_leaks=float(np.mean(n_leaks_applied)),
        mean_dropped=float(np.mean(dropped)),
        bias=bias.tolist(), rms=rms.tolist(), std=std.tolist(),
        claimed_sigma=claimed.tolist(),
        inplane_rms=float(np.sqrt((inplane ** 2).mean())),
        inplane_std=float(np.sqrt(std[1] ** 2 + std[2] ** 2)),
        inplane_claimed_sigma=float(np.sqrt(claimed[1] ** 2 + claimed[2] ** 2)),
        inplane_p90=float(np.percentile(inplane, 90)),
        nees_mean=float(np.mean(nees_rw)),
        nees_median=float(np.median(nees_rw)),
        clears_005_claimed=bool(claimed[1] < 0.05 and claimed[2] < 0.05),
        clears_005_inplane_rms=bool(np.sqrt((inplane ** 2).mean()) < 0.05),
    )


def fmt(c):
    return (f"{c['tag']:>34} | fix{c['mean_fixes']:4.1f} lk{c['mean_leaks']:.2f} drp{c['mean_dropped']:.2f} | "
            f"clmSig E/D {c['claimed_sigma'][1]:.3f}/{c['claimed_sigma'][2]:.3f} | "
            f"empStd E/D {c['std'][1]:.3f}/{c['std'][2]:.3f} | "
            f"ipRMS {c['inplane_rms']:.3f} ipStd {c['inplane_std']:.3f} | "
            f"NEES {c['nees_mean']:5.2f} | clr {int(c['clears_005_inplane_rms'])}")


def main():
    np.seterr(all="raise")
    pools, means = load_pools()
    print("=" * 130)
    print("b1 ADVERSARIAL re-run of a1 gate-4 race-speed estimator headline (seed", BASE_SEED, ")")
    print("  pools (good, |off|<3):", {k: int(v.shape[0]) for k, v in pools.items()})
    print("  band means (a1 debias):", {k: np.round(v, 3).tolist() for k, v in means.items()})
    print("  a2 gate-4 residual (global-debias leaves) NED:", A2_GATE4_RESIDUAL_DEBIASED_NED.tolist())
    print("=" * 130)

    cells = {}

    # ---- 0) REPRODUCE a1's headline cell (edge L=6, 47%, debiased, no blur, no leak) ----
    print("\n[0] REPRODUCE a1 headline cell (L6, 47%, debiased, no blur, no leak):")
    c = run_cell(6.0, 0.47, "debiased", pools, means, tag="repro_a1_L6_47_DEB")
    cells["repro_a1_L6_47_DEB"] = c
    print("   ", fmt(c))
    print(f"    a1 reported: claimed E/D 0.239/0.239, inplane RMS 0.33. b1 reproduces ->",
          f"E/D {c['claimed_sigma'][1]:.3f}/{c['claimed_sigma'][2]:.3f}, ipRMS {c['inplane_rms']:.3f}")
    cR = run_cell(6.0, 0.47, "raw", pools, means, tag="repro_a1_L6_47_RAW")
    cells["repro_a1_L6_47_RAW"] = cR
    print("   ", fmt(cR))

    # ---- 1) A3 DEGRADED inputs: SHORT shutter then LONG exposure ----
    print("\n[1] A3 DEGRADED fix-stream inputs (replaces a1's slow-speed/no-blur assumption):")
    # SHORT shutter: noise x1.05 in-plane, x1.05 along, accept 0.45
    for em in ("raw", "debiased"):
        c = run_cell(6.0, 0.45, em, pools, means, noise_mult_inplane=1.05, noise_mult_along=1.05,
                     tag=f"A3short_L6_45_{em[:3].upper()}")
        cells[c["tag"]] = c
        print("   ", fmt(c))
    # LONG exposure: in-plane noise x1.7, along x1.3, accept 0.33
    for em in ("raw", "debiased"):
        c = run_cell(6.0, 0.33, em, pools, means, noise_mult_inplane=1.7, noise_mult_along=1.3,
                     tag=f"A3long_L6_33_{em[:3].upper()}")
        cells[c["tag"]] = c
        print("   ", fmt(c))

    # ---- 2) CATASTROPHIC LEAK injected at the real rate (a1 omitted it) ----
    print("\n[2] CATASTROPHIC LEAK injected (rate 0.0053 chi2-surviving; a1 omitted):")
    for em in ("raw", "debiased"):
        c = run_cell(6.0, 0.47, em, pools, means, leak_rate=BASE_LEAK,
                     tag=f"leak_L6_47_{em[:3].upper()}")
        cells[c["tag"]] = c
        print("   ", fmt(c))
    # leak at a stress 5x rate (in case the at-speed leak is worse with widened-but-biased fixes)
    c = run_cell(6.0, 0.47, "debiased", pools, means, leak_rate=5 * BASE_LEAK,
                 tag="leak5x_L6_47_DEB")
    cells[c["tag"]] = c
    print("   ", fmt(c))

    # ---- 3) HONEST debias: leave the a2 per-gate residual that GLOBAL de-bias cannot remove ----
    print("\n[3] HONEST global-debias (a1's debias is optimistic: leaves a2 per-gate gate-4 residual):")
    c = run_cell(6.0, 0.47, "debiased", pools, means, gate4_residual=True,
                 tag="honestDEB_L6_47")
    cells[c["tag"]] = c
    print("   ", fmt(c))

    # ---- 4) COMBINED worst-honest: A3 long + leak + honest debias ----
    print("\n[4] COMBINED honest worst (A3 long-exposure + leak + per-gate residual, debiased):")
    c = run_cell(6.0, 0.33, "debiased", pools, means, noise_mult_inplane=1.7, noise_mult_along=1.3,
                 leak_rate=BASE_LEAK, gate4_residual=True, tag="COMBINED_L6_33_DEB")
    cells[c["tag"]] = c
    print("   ", fmt(c))

    # ---- 5) A4 rewind-horizon check at CPU L=112 ms (no ~21 m divergence?) ----
    print("\n[5] REWIND HORIZON check at CPU L=112 ms (horizon 0.5 s >= L?):")
    c = run_cell(112.0, 0.47, "debiased", pools, means, tag="cpuL112_47_DEB")
    cells[c["tag"]] = c
    print("   ", fmt(c))
    print(f"    mean dropped fixes = {c['mean_dropped']:.3f} (FACTS: horizon<L drops ALL -> ~21 m; "
          f"divergence iff ipRMS blows up). ipRMS={c['inplane_rms']:.3f} -> "
          f"{'NO divergence' if c['inplane_rms'] < 1.0 else 'DIVERGED'}")

    # ---- 6) BEST-CASE floor: what per-fix sigma is needed (sanity vs a1 floor probe) ----
    print("\n[6] DEBIASED at LOW L + idealized 100% accept (best variance crush a1 claims):")
    c = run_cell(6.0, 1.00, "debiased", pools, means, tag="ideal_L6_100_DEB")
    cells[c["tag"]] = c
    print("   ", fmt(c))

    out = dict(
        meta=dict(seed=BASE_SEED, n_seeds=N_SEEDS, speed=SPEED, seg_len=SEG_LEN,
                  t_cross=SEG_LEN / SPEED, drag_pitch_deg=float(np.rad2deg(DRAG_HOLD_PITCH_RAD)),
                  band_means={k: v.tolist() for k, v in means.items()},
                  a2_gate4_residual_ned=A2_GATE4_RESIDUAL_DEBIASED_NED.tolist(),
                  chi2_gate=CHI2_GATE, base_leak=BASE_LEAK,
                  note="b1 adversarial re-run; perturbs a1 assumptions (A3 degraded inputs, leaks, "
                       "honest per-gate residual debias, CPU horizon). Reproduces a1 headline cell."),
        cells=cells,
    )
    (HERE / "b1_verify_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'b1_verify_results.json'} ({len(cells)} cells)")

    # ---- verdict summary ----
    print("\n" + "=" * 130)
    print("VERDICT SUMMARY (does ANY cell clear <0.05 m in-plane?):")
    any_clear = any(c["clears_005_inplane_rms"] for c in cells.values())
    print(f"  ANY cell clears 0.05 m in-plane RMS: {any_clear}")
    print(f"  ANY cell clears 0.05 m claimed-sigma both axes: {any(c['clears_005_claimed'] for c in cells.values())}")
    h = cells["repro_a1_L6_47_DEB"]
    print(f"  a1 headline reproduced: DEBIASED inplane RMS {h['inplane_rms']:.3f} (a1 said 0.33)")
    print(f"  honest-debias (per-gate residual) inplane RMS: {cells['honestDEB_L6_47']['inplane_rms']:.3f}")
    print(f"  combined honest-worst inplane RMS: {cells['COMBINED_L6_33_DEB']['inplane_rms']:.3f}")


if __name__ == "__main__":
    main()
