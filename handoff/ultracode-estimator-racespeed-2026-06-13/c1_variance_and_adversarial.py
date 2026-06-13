"""c1 (part 2) — the VARIANCE question for gate-relative + the ADVERSARIAL refutations.

Part (a)/(b) established: a TRUE gate-relative observation removes the per-track MAP bias (rel arm
E_bias/D_bias ~ 0.00 m vs abs +0.18/+0.06), and the achievable in-plane miss is set by the PnP
LATERAL noise (NOT the map/registration bias, NOT the absolute world-fix variance floor). This file:

  1. NEAR-BAND variance: evaluate the gate-relative in-plane miss at the BINDING transit band
     (the last-usable accepted fix, range ~8 m, where b2 showed lateral collapses to ~0.10 m), not
     averaged over 0-12 m. Sweep per-fix lateral sigma -> filtered in-plane 1-sigma, find where the
     0.05 m bar and the 0.155 m margin are met.
  2. The split: is the gate-relative path VARIANCE-limited (good news: averageable) and does it have
     ANY residual bias (the PnP chain-correlated lateral systematic that a relative obs does NOT
     remove -- the honest floor)?
  3. ADVERSARIAL: (i) loses the absolute anchor for planning? (ii) PnP depth-flip at close range?
     (iii) 3-corner P3P clipping -> only 2-DOF? Quantify each against the data.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c1_variance_and_adversarial.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.frames import R_world_from_body  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 20260613
CHI2_GATE = 16.27
BAR = 0.05
MARGIN_G4 = 0.155
V_RACE = 37.0
G4_BOTTOM_NED = np.array([-135.5, -0.8, 25.36])
G3_BOTTOM_NED = np.array([-111.5, -5.1, 24.57])


def load_rows(path):
    d = json.load(open(path))
    return [r for r in d["rows"] if r.get("associated") and "world_fix_err_m" in r]


def lateral_pnp_err(r):
    tce = float(r["t_cam_err_m"]); re = float(r["range_err_m"])
    return float(np.sqrt(max(tce * tce - re * re, 0.0)))


def near_band_lateral():
    """The BINDING transit-band lateral PnP noise: accepted gate-4 fixes, range bands near transit."""
    g4 = load_rows(_REPO / "handoff/perception-char-2026-06-08/characterize_g4.json")
    course = load_rows(_REPO / "handoff/perception-char-2026-06-08/characterize_course_60s.json")
    rows = [r for r in (g4 + course) if r.get("gate_id") == 4 and np.isfinite(r["maha"])
            and r["maha"] <= CHI2_GATE and r["n_corners"] == 4]
    out = {}
    for lo, hi in [(0, 6), (4, 9), (6, 9), (8, 10), (8, 12), (0, 9)]:
        band = [r for r in rows if lo <= r["true_range_m"] < hi]
        if not band:
            continue
        lat = np.array([lateral_pnp_err(r) for r in band])
        # signed in (E,D): we only have magnitude; report magnitude stats + a mean(=possible residual
        # systematic if the lateral error is consistently one-signed -- but t_cam_err is a magnitude,
        # so we CANNOT see sign here. The relative-obs residual bias is bounded by the per-fix-mean
        # lateral magnitude only if it is one-signed; the WORLD-fix per-gate E offset (a2 +0.46/b2 near
        # 0.19) is the upper bound on any chain-correlated lateral systematic that survives.)
        out[f"[{lo},{hi})"] = dict(n=len(band), lat_rms=float(np.sqrt(np.mean(lat**2))),
                                   lat_mean=float(np.mean(lat)), lat_p50=float(np.percentile(lat, 50)),
                                   lat_min=float(lat.min()))
    return out, rows


def filtered_variance_sweep(per_axis_sigmas):
    """Run the REAL KF on the straight g3->g4 approach with a TRUE gate-relative obs (zero bias,
    per-axis lateral sigma), measure filtered in-plane 1-sigma at the gate-4 plane. Sweep sigma."""
    dt_imu = 1.0 / 90.0
    seg = G4_BOTTOM_NED - G3_BOTTOM_NED
    L_seg = float(np.linalg.norm(seg)); uhat = seg / L_seg
    T = L_seg / V_RACE; n_steps = int(T / dt_imu)
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    g = np.array([0.0, 0.0, 9.80665]); accel_body = R_wb.T @ (-g)
    fix_hz = 14.0; fix_dt = 1.0 / fix_hz
    N_MC = 800
    results = {}
    for sig in per_axis_sigmas:
        E_e, D_e = [], []
        for s in range(N_MC):
            r = np.random.default_rng(SEED + 13 * s + int(sig * 1000))
            p0 = G3_BOTTOM_NED.copy(); v0 = uhat * V_RACE
            kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5)
            rk = RewindKF(kf=kf, horizon_s=0.5)
            t_ns = 0; next_fix = 0.0; t = 0.0
            for k in range(n_steps):
                t += dt_imu; t_ns += int(dt_imu * 1e9)
                rk.predict(accel_body, R_wb, dt_imu, t_ns)
                p_true = p0 + uhat * V_RACE * t
                rng_g4 = float(np.linalg.norm(G4_BOTTOM_NED - p_true))
                if t >= next_fix and rng_g4 < 12.0:
                    next_fix += fix_dt
                    z = p_true.copy()
                    z[1] += r.normal(0, sig); z[2] += r.normal(0, sig)
                    z[0] += r.normal(0, 0.50)  # along-track radial (doesn't enter in-plane)
                    cov = np.diag([0.50**2, sig**2, sig**2])  # TIGHT: no bias-absorption floor
                    rk.update_position(z, cov, sim_time_ns=t_ns)
            p_final = p0 + uhat * V_RACE * (n_steps * dt_imu)
            err = rk.position - p_final
            E_e.append(err[1]); D_e.append(err[2])
        E = np.array(E_e); D = np.array(D_e)
        ip = np.sqrt(E**2 + D**2)
        results[f"{sig:.3f}"] = dict(
            per_axis_in=sig, filt_E_sigma=float(E.std()), filt_D_sigma=float(D.std()),
            filt_inplane_1sigma=float(np.sqrt(E.std()**2 + D.std()**2) / np.sqrt(2)),
            inplane_rms=float(np.sqrt(np.mean(ip**2))),
            clears_bar=bool(max(E.std(), D.std()) < BAR),
            clears_margin=bool(float(np.sqrt(np.mean(ip**2))) < MARGIN_G4))
    return results


def adversarial(rows):
    """Quantify the three refutations from the data."""
    out = {}
    # (iii) 3-corner / P3P clipping: at what range does the gate clip to <4 corners, and how many
    # accepted near fixes are 3-corner?
    g4all = load_rows(_REPO / "handoff/perception-char-2026-06-08/characterize_g4.json") + \
        load_rows(_REPO / "handoff/perception-char-2026-06-08/characterize_course_60s.json")
    g4all = [r for r in g4all if r.get("gate_id") == 4]
    p3p = [r for r in g4all if r["n_corners"] == 3]
    p3p_acc = [r for r in p3p if np.isfinite(r["maha"]) and r["maha"] <= CHI2_GATE]
    out["p3p"] = dict(n_3corner=len(p3p), n_3corner_accepted=len(p3p_acc),
                      n_4corner=len([r for r in g4all if r["n_corners"] == 4]),
                      ranges_3corner=[round(r["true_range_m"], 2) for r in p3p])
    # (ii) depth-flip: rows with huge t_cam_err but small reproj (the frontal-PnP ambiguity). Count
    # accepted (maha<=gate) ones at near range -- if the chi2 gate catches them, the relative obs is safe.
    flips = [r for r in g4all if r["t_cam_err_m"] > 3.0]  # gross PnP-relative error
    flips_accepted = [r for r in flips if np.isfinite(r["maha"]) and r["maha"] <= CHI2_GATE]
    out["depth_flip"] = dict(
        n_gross_pnp_err=len(flips),
        n_gross_accepted_after_chi2=len(flips_accepted),
        accepted_flip_ranges=[round(r["true_range_m"], 2) for r in flips_accepted],
        note="gross PnP-relative errors (>3 m) are depth-flips/wrong-gate; count surviving chi2 gate")
    # The maha gate operates on the WORLD fix; for the relative obs we'd gate on reproj + a relative
    # innovation. Report reproj separation between clean and flip fixes.
    clean = [r for r in g4all if r["t_cam_err_m"] <= 1.0 and r["n_corners"] == 4]
    out["reproj_separation"] = dict(
        clean_reproj_p50=float(np.percentile([r["reproj_px"] for r in clean], 50)) if clean else None,
        clean_reproj_p90=float(np.percentile([r["reproj_px"] for r in clean], 90)) if clean else None,
        flip_reproj_p50=float(np.percentile([r["reproj_px"] for r in flips], 50)) if flips else None,
        note="if flips have distinguishable reproj, a relative-innovation gate can reject them")
    return out


def main():
    np.random.seed(SEED)
    res = {"seed": SEED}
    print("=" * 84)
    print("1. NEAR-BAND lateral PnP noise (the binding transit band -- gate-relative achievable miss)")
    print("=" * 84)
    nb, rows = near_band_lateral()
    res["near_band_lateral"] = nb
    print(f" {'band':>10} {'n':>3} {'lat_rms':>8} {'lat_mean':>9} {'lat_p50':>8} {'lat_min':>8}")
    for band, b in nb.items():
        print(f" {band:>10} {b['n']:3d} {b['lat_rms']:8.3f} {b['lat_mean']:9.3f} {b['lat_p50']:8.3f} "
              f"{b['lat_min']:8.3f}")

    print("\n" + "=" * 84)
    print("2. FILTERED in-plane 1-sigma vs per-fix lateral sigma (TRUE gate-relative, tight R, no floor)")
    print("=" * 84)
    sweep = filtered_variance_sweep([0.40, 0.30, 0.265, 0.20, 0.15, 0.105, 0.08, 0.05])
    res["variance_sweep"] = sweep
    print(f" {'per_fix_lat':>11} {'filt_E':>7} {'filt_D':>7} {'filt_inplane_1sig':>18} {'ip_rms':>7} "
          f"{'<0.05?':>7} {'<margin?':>9}")
    for k, v in sweep.items():
        print(f" {v['per_axis_in']:11.3f} {v['filt_E_sigma']:7.3f} {v['filt_D_sigma']:7.3f} "
              f"{v['filt_inplane_1sigma']:18.3f} {v['inplane_rms']:7.3f} "
              f"{('YES' if v['clears_bar'] else 'NO'):>7} {('YES' if v['clears_margin'] else 'NO'):>9}")

    print("\n" + "=" * 84)
    print("3. ADVERSARIAL refutations (from data)")
    print("=" * 84)
    adv = adversarial(rows)
    res["adversarial"] = adv
    print(" (iii) P3P/3-corner clipping at gate-4:")
    print(f"   3-corner fixes: {adv['p3p']['n_3corner']} (accepted after chi2: "
          f"{adv['p3p']['n_3corner_accepted']}); 4-corner: {adv['p3p']['n_4corner']}; "
          f"3-corner ranges: {adv['p3p']['ranges_3corner']}")
    print(" (ii) Depth-flip / gross PnP error:")
    print(f"   gross PnP-rel err(>3 m): {adv['depth_flip']['n_gross_pnp_err']}; "
          f"surviving chi2 gate: {adv['depth_flip']['n_gross_accepted_after_chi2']} "
          f"(ranges {adv['depth_flip']['accepted_flip_ranges']})")
    print(f"   reproj: clean p50={adv['reproj_separation']['clean_reproj_p50']}, "
          f"clean p90={adv['reproj_separation']['clean_reproj_p90']}, "
          f"flip p50={adv['reproj_separation']['flip_reproj_p50']}")

    outpath = Path(__file__).resolve().parent / "c1_variance_and_adversarial_results.json"
    outpath.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
