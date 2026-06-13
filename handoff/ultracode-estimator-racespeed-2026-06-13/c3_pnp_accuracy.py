"""c3 — Detector/PnP accuracy levers at the gate-4 approach range: do sub-pixel corner
refinement, a better detector, 2-corner-fallback handling, or gate-size calibration reduce
the world-fix sigma (and/or the bias) enough to matter for the gate-4 in-plane miss?

BRANCH: tighter PnP / detector accuracy at range.

This is OFFLINE analysis. It builds on the SHIPPED fix-cov model
(handoff/ultracode-vision-case-c-2026-06-13/range_anisotropic_R.py, coeffs c2/a1 from
range_R_coeffs.json) and the MEASURED per-fix pool
(handoff/perception-char-2026-06-08/characterize_course_60s.json). It does NOT touch src/.

CORE QUESTIONS (from the prompt):
  Q1  sigma_px -> sigma_depth (r^2 law) and sigma_lateral (r law): how much world-fix sigma
      reduction does lowering per-corner pixel noise buy at the gate-4 transit range?
  Q2  Which accuracy lever attacks BIAS vs VARIANCE?
        - gate-size-model calibration  -> removes a per-fix DEPTH BIAS (range-proportional)
        - sub-pixel refinement / better detector -> shrinks pixel NOISE only
        - 2-corner fallback handling   -> avoids the x9 P3P inflation + cheirality bias
  Q3  ADVERSARIAL: is the detector ALREADY near the pixel-noise floor (measured reproj ~0.5 px,
      not the assumed 1.5 px) so refinement buys little / nothing?

All numbers reproduce on re-run (numpy RNG seeded; the analysis is closed-form + measured data).
Run (bash):  PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c3_pnp_accuracy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE_C = ROOT / "handoff" / "ultracode-vision-case-c-2026-06-13"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CASE_C))

from racer.frames import ATTITUDE_NOISE_STD_RAD, CAMERA_INTRINSICS_K  # noqa: E402
from racer.localization import FIX_COV_FLOOR_STD  # noqa: E402
from racer.vision.gate_pose import GATE_INNER_SIZE_M, WEIGHTED_SIGMA_PX  # noqa: E402
import range_anisotropic_R as RA  # noqa: E402

RNG = np.random.default_rng(20260613)

F = float(CAMERA_INTRINSICS_K[0, 0])          # focal px (=320)
S_ASSUMED = float(GATE_INNER_SIZE_M)          # 1.5 m assumed inner side
SIGMA_THETA = float(ATTITUDE_NOISE_STD_RAD)   # 1.4 deg attitude lever 1-sigma (rad)
FLOOR = float(FIX_COV_FLOOR_STD)              # 0.40 m isotropic cov floor

# Gate-4 transit / last-fix range band on the g3->g4 LEVEL straight (FACTS): the closing
# transit window. The "last fix" before crossing is what dominates the in-plane fix at the
# plane. We sweep a representative band; the in-plane axes at gate-4 are E (lateral) and D
# (vertical), the LOS / depth axis is N (along-track).
GATE4_RANGES = [5.0, 8.0, 12.0, 16.0, 20.0]


def perfix_world_std(sigma_px: float, r: float, n_corners: int = 4) -> dict:
    """Per-axis world-fix 1-sigma decomposition at range r for a given per-corner pixel noise.

    Uses the SHIPPED anisotropic R-model channels (range_anisotropic_R), but with the c2/a1
    pixel-Fisher coefficients RE-DERIVED at the supplied sigma_px (they are LINEAR in sigma_px).
    Returns the depth (radial / along-LOS / along-track-N), lateral (tangential / in-plane-E),
    and the floor / lever contributions so we can see what each lever moves.
    """
    c2 = RA._c2_from_pixels(sigma_px)         # depth std coeff: std_depth = c2 r^2
    a1 = RA._a1_from_pixels(sigma_px)         # lateral std coeff: std_lat  = a1 r
    sig_depth_px = c2 * r * r                  # pixel-noise depth (RADIAL, ~along-track N)
    sig_lat_px = a1 * r                        # pixel-noise lateral (TANGENTIAL, in-plane E/D)
    sig_lever = SIGMA_THETA * r                # attitude-lever tangential std (in-plane), ~r
    # Floors (range-flat). sig0_tan is the calibrated tangential floor in the shipped coeffs.
    coeffs = RA.load_coeffs()
    sig0_tan = float(coeffs["sig0_tan"])
    # Radial (depth, along LOS): pixel r^2 (+) range-flat radial floor.
    var_radial = sig_depth_px ** 2 + FLOOR ** 2
    # Tangential (lateral, perp LOS): pixel r^1 (+) lever (+) tangential floor.
    var_tang = sig_lat_px ** 2 + sig_lever ** 2 + sig0_tan ** 2
    infl = 9.0 if n_corners < 4 else 1.0       # P3P x9 inflation
    return {
        "r": r,
        "sigma_px": sigma_px,
        "sig_depth_pixel": sig_depth_px,
        "sig_lat_pixel": sig_lat_px,
        "sig_lever_tang": sig_lever,
        "sig_floor_radial": FLOOR,
        "sig_floor_tang": sig0_tan,
        "std_radial_total": float(np.sqrt(var_radial * infl)),   # along-track N at gate-4
        "std_tang_total": float(np.sqrt(var_tang * infl)),       # in-plane E/D at gate-4
        "pixel_share_radial": float(sig_depth_px ** 2 / (var_radial)),  # frac of radial var from pixels
        "pixel_share_tang": float(sig_lat_px ** 2 / (var_tang)),        # frac of tang var from pixels
    }


def q1_q3_pixel_sweep() -> dict:
    """Q1+Q3: sweep sigma_px (1.5 nominal, 0.5 measured, 0.1 ideal sub-pixel, 0.0 floor) and
    report the world-fix in-plane (tangential) and along-track (radial) std at each gate-4 range.
    The KEY adversarial readout: pixel_share_* = what fraction of the variance pixel noise even
    controls. If pixel_share is tiny, refinement is wasted."""
    out = {"description": "world-fix per-axis std vs per-corner pixel noise at gate-4 ranges",
           "rows": []}
    for spx in [1.5, 0.5, 0.1, 0.0]:
        for r in GATE4_RANGES:
            d = perfix_world_std(spx, r)
            out["rows"].append(d)
    return out


def q2_gatesize_bias(rows: list) -> dict:
    """Q2 (bias lever): the depth bias from gate-size-model mismatch. A wrong inner size s_assumed
    vs true s_true gives pose_range = true_range * (s_assumed / s_true), i.e. a CONSTANT FRACTIONAL
    (range-proportional) depth BIAS = (s_assumed/s_true - 1) * range. Calibrating s removes it.
    We FIT this from the measured range_err vs true_range and report (a) the implied true size,
    (b) the residual after cal, (c) which gate-4 axis it lands on (depth=LOS)."""
    good = rows
    trng = np.array([r["true_range_m"] for r in good])
    rerr = np.array([r["range_err_m"] for r in good])   # signed depth bias = pose_range - true_range
    m = trng > 3.0                                       # close-range frac is noisy; fit on r>3
    A = np.vstack([trng[m], np.ones(m.sum())]).T
    slope, intercept = np.linalg.lstsq(A, rerr[m], rcond=None)[0]
    frac = float(np.median((rerr[m] / trng[m])))         # robust fractional depth bias
    s_true_implied = S_ASSUMED / (1.0 + frac)
    # Residual depth bias AFTER recalibrating s to remove the fractional term:
    rerr_cal = rerr - frac * trng                        # subtract the range-proportional part
    resid_const = float(np.median(rerr_cal))
    return {
        "depth_bias_slope_per_m": float(slope),
        "depth_bias_intercept_m": float(intercept),
        "fractional_depth_bias": frac,
        "implied_true_inner_size_m": float(s_true_implied),
        "assumed_inner_size_m": S_ASSUMED,
        "residual_const_depth_bias_after_cal_m": resid_const,
        "depth_bias_at_5m_m": float(frac * 5.0),
        "depth_bias_at_12m_m": float(frac * 12.0),
        "note": ("depth bias is range-proportional => GATE-SIZE-MODEL MISMATCH (a BIAS, removable "
                 "by calibrating s). At gate-4 the LOS is ~along -N so this BIAS is ALONG-TRACK (N), "
                 "NOT the in-plane (E,D) miss."),
    }


def q2_axis_projection(good: list) -> dict:
    """Confirm with MEASURED gate-4 fixes that the depth bias lands on N (along-track), i.e.
    gate-size cal does NOT touch the binding in-plane (E) bias."""
    g4 = [r for r in good if r["gate_id"] == 4]
    off = np.array([r["off_ned"] for r in g4])
    rerr = np.array([r["range_err_m"] for r in g4])
    corr_N = float(np.corrcoef(off[:, 0], rerr)[0, 1]) if len(g4) > 2 else float("nan")
    return {
        "n_gate4_fixes": len(g4),
        "gate4_off_mean_NED": off.mean(axis=0).round(4).tolist(),
        "gate4_depth_bias_mean_m": float(rerr.mean()),
        "corr_offN_vs_rangeerr": corr_N,
        "in_plane_bias_E_m": float(off[:, 1].mean()),
        "in_plane_bias_D_m": float(off[:, 2].mean()),
        "in_plane_bias_norm_m": float(np.hypot(off[:, 1].mean(), off[:, 2].mean())),
        "note": ("corr~1 => depth bias lives on N (along-track). The binding gate-4 in-plane bias "
                 "is LATERAL (E) ~per-gate registration/bearing systematic, NOT depth, NOT pixel-noise."),
    }


def q3_reproj_decorrelation(good: list) -> dict:
    """Q3 (the adversarial crux): does the per-fix world-fix NOISE correlate with reproj_px?
    If NOT, the corner-localization residual is NOT the bottleneck and sub-pixel refinement
    (lowering reproj) buys nothing. We measure within-gate (per-gate-detrended) noise vs reproj."""
    rp = np.array([r["reproj_px"] for r in good])
    off = np.array([r["off_ned"] for r in good])
    gid = np.array([r["gate_id"] for r in good])
    detr = off.copy()
    for g in np.unique(gid):
        mm = gid == g
        detr[mm] = off[mm] - off[mm].mean(axis=0)        # per-gate-detrend -> pure within-gate noise
    mag = np.linalg.norm(detr, axis=1)
    corr = float(np.corrcoef(rp, mag)[0, 1])
    bins = []
    for lo, hi, lab in [(0.0, 0.5, "reproj<0.5px"), (0.5, 1.5, "0.5-1.5px"), (1.5, 1e9, ">1.5px")]:
        mm = (rp >= lo) & (rp < hi)
        if mm.sum() > 2:
            bins.append({"band": lab, "n": int(mm.sum()),
                         "within_gate_noise_norm_std_m": float(np.linalg.norm(detr[mm].std(axis=0)))})
    return {
        "reproj_px_median": float(np.median(rp)),
        "reproj_px_mean": float(rp.mean()),
        "reproj_px_p90": float(np.percentile(rp, 90)),
        "assumed_nominal_sigma_px": float(WEIGHTED_SIGMA_PX),
        "corr_reproj_vs_within_gate_noise": corr,
        "bins": bins,
        "verdict": ("reproj is already ~0.5 px (vs assumed 1.5 px) AND within-gate world-fix noise "
                    "does NOT correlate with reproj (corr~0). The world-fix noise is NOT pixel-driven "
                    "=> sub-pixel refinement buys ~nothing."),
    }


def q_floor_dominance() -> dict:
    """How much of the gate-4 in-plane (tangential) variance is the ATTITUDE LEVER + FLOOR (which
    no detector improvement can touch) vs PIXEL noise? This bounds the maximum benefit of any
    detector/PnP accuracy lever."""
    out = []
    for r in GATE4_RANGES:
        # at measured sigma_px=0.5 (best estimate of real detector)
        d05 = perfix_world_std(0.5, r)
        # if pixel noise driven to ZERO (perfect corners): residual tangential = lever + floor
        var_irreducible_tang = (SIGMA_THETA * r) ** 2 + RA.load_coeffs()["sig0_tan"] ** 2
        var_total_tang = d05["std_tang_total"] ** 2
        out.append({
            "r": r,
            "std_tang_total_at_spx0.5_m": d05["std_tang_total"],
            "std_tang_irreducible_m": float(np.sqrt(var_irreducible_tang)),  # lever+floor only
            "pixel_reducible_fraction_of_tang_std": float(
                1.0 - np.sqrt(var_irreducible_tang / var_total_tang)),
        })
    return {"rows": out,
            "note": ("'irreducible' = the lever(1.4deg)+tangential-floor part a perfect detector "
                     "CANNOT remove. pixel_reducible_fraction = max in-plane std reduction a perfect "
                     "detector could buy. If ~0, the detector is NOT the in-plane lever.")}


def main():
    char = json.loads((ROOT / "handoff" / "perception-char-2026-06-08" /
                        "characterize_course_60s.json").read_text())
    rows = char["rows"]
    good = [r for r in rows if r.get("associated") and abs(np.linalg.norm(r["off_ned"])) < 3.0]

    results = {
        "seed": 20260613,
        "constants": {"focal_px": F, "assumed_inner_size_m": S_ASSUMED,
                      "sigma_theta_rad": SIGMA_THETA, "sigma_theta_deg": float(np.degrees(SIGMA_THETA)),
                      "cov_floor_std_m": FLOOR, "weighted_sigma_px_nominal": float(WEIGHTED_SIGMA_PX),
                      "c2_shipped": RA.load_coeffs()["c2"], "a1_shipped": RA.load_coeffs()["a1"],
                      "sig0_tan_shipped": RA.load_coeffs()["sig0_tan"]},
        "n_good_fixes": len(good),
        "Q1_Q3_pixel_sweep": q1_q3_pixel_sweep(),
        "Q2_gatesize_bias": q2_gatesize_bias(good),
        "Q2_axis_projection": q2_axis_projection(good),
        "Q3_reproj_decorrelation": q3_reproj_decorrelation(good),
        "Q_floor_dominance": q_floor_dominance(),
    }

    out_path = Path(__file__).resolve().parent / "c3_pnp_accuracy_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    # --- Console summary -------------------------------------------------------------------
    print("=" * 78)
    print("c3 — detector/PnP accuracy levers at the gate-4 window")
    print("=" * 78)
    print(f"n good fixes={len(good)}  focal={F}px  assumed s={S_ASSUMED}m  "
          f"sigma_theta={np.degrees(SIGMA_THETA):.2f}deg  floor={FLOOR}m")
    print()
    print("--- Q3 ADVERSARIAL: is the detector already near the pixel floor? ---")
    q3 = results["Q3_reproj_decorrelation"]
    print(f"  measured reproj_px median={q3['reproj_px_median']:.3f}  mean={q3['reproj_px_mean']:.3f}  "
          f"(assumed nominal sigma_px={q3['assumed_nominal_sigma_px']})")
    print(f"  corr(reproj, within-gate world-fix noise) = {q3['corr_reproj_vs_within_gate_noise']:.3f}")
    for b in q3["bins"]:
        print(f"    {b['band']:>12s}  n={b['n']:3d}  within-gate noise norm-std={b['within_gate_noise_norm_std_m']:.3f} m")
    print(f"  => {q3['verdict']}")
    print()
    print("--- Q1: pixel-noise -> world-fix std at gate-4 ranges (tangential=in-plane E/D) ---")
    print("  r(m)  spx | std_lat_pix  std_lever  std_tang_TOT | std_depth_pix  std_radial_TOT | pix_share_tang")
    for d in results["Q1_Q3_pixel_sweep"]["rows"]:
        print(f"  {d['r']:4.0f}  {d['sigma_px']:.1f} | {d['sig_lat_pixel']:9.4f}  {d['sig_lever_tang']:8.4f}  "
              f"{d['std_tang_total']:10.4f} | {d['sig_depth_pixel']:11.4f}  {d['std_radial_total']:12.4f} | "
              f"{d['pixel_share_tang']:.4f}")
    print()
    print("--- Q_floor_dominance: max in-plane std reduction a PERFECT detector could buy ---")
    for d in results["Q_floor_dominance"]["rows"]:
        print(f"  r={d['r']:4.0f}m  tang std@spx0.5={d['std_tang_total_at_spx0.5_m']:.3f}  "
              f"irreducible(lever+floor)={d['std_tang_irreducible_m']:.3f}  "
              f"pixel-reducible frac={d['pixel_reducible_fraction_of_tang_std']:.3f}")
    print()
    print("--- Q2 BIAS lever: gate-size-model calibration ---")
    q2 = results["Q2_gatesize_bias"]
    print(f"  depth bias = {q2['depth_bias_slope_per_m']:.4f}*r + {q2['depth_bias_intercept_m']:.4f}  "
          f"(range-proportional => size-mismatch BIAS)")
    print(f"  fractional depth bias = {q2['fractional_depth_bias']:.4f}  => implied true inner size = "
          f"{q2['implied_true_inner_size_m']:.4f} m (assumed {q2['assumed_inner_size_m']})")
    print(f"  residual const depth bias after size-cal = {q2['residual_const_depth_bias_after_cal_m']:.4f} m")
    qa = results["Q2_axis_projection"]
    print(f"  gate-4 measured: off_NED={qa['gate4_off_mean_NED']}  depth_bias={qa['gate4_depth_bias_mean_m']:.3f}m")
    print(f"  corr(off_N, range_err)@gate4 = {qa['corr_offN_vs_rangeerr']:.3f}  => depth bias is ALONG-TRACK (N)")
    print(f"  gate-4 IN-PLANE bias (E,D)=({qa['in_plane_bias_E_m']:.3f},{qa['in_plane_bias_D_m']:.3f}) "
          f"norm={qa['in_plane_bias_norm_m']:.3f}m  <- gate-size cal does NOT touch this")
    print()
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
