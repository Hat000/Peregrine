"""Diagnostic: resolve the A2(covariance) vs A3(actual MC miss) gap. Measure the ACTUAL lateral/vert
estimate error at the r_floor coast-start across MC laps (the truth the dense pointed fix stream really
delivers), compare to the KF covariance self-assessment, and decompose the terminal miss (lateral vs
vertical). [COAST-DRIFT 2026-06-15]"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coast_drift as CD  # noqa: E402
import margin_envelope as ME  # noqa: E402


def run(v=30.0, sigma_lat=0.10, sigma_vert=0.10, fix_rate=0.60, r_floor=12.0, bias_deg=0.0,
        bias_mode="inplane", lat_fix_bias_m=0.0, n_mc=800):
    bmag = ME.att_deg_to_accel_bias(bias_deg)
    perr_lat = []; perr_vert = []; verr_lat = []; verr_vert = []
    miss_lat = []; miss_vert = []; miss = []
    rfound = []
    for s in range(n_mc):
        seed = (CD.SEED + 101*s + int(round(v))*13 + int(round(bias_deg*100))*7
                + int(round(sigma_lat*1000))*17 + int(round(fix_rate*1000))*23
                + int(round(r_floor*100))*131)
        rng = np.random.default_rng(seed)
        m, d = CD.fly_lap_coast(rng, v, sigma_lat, sigma_vert, bmag, bias_mode, fix_rate,
                                r_floor, lat_fix_bias_m=lat_fix_bias_m, diag=True)
        miss.append(m); miss_lat.append(d["miss_lat"]); miss_vert.append(d["miss_vert"])
        if "perr_lat" in d:
            perr_lat.append(d["perr_lat"]); perr_vert.append(d["perr_vert"])
            verr_lat.append(d["verr_lat"]); verr_vert.append(d["verr_vert"])
            rfound.append(d["range_m"])
    A = lambda x: np.asarray(x)
    # covariance self-assessment (deterministic pass)
    fc = CD.a2_achieved_sigma_v(v, sigma_lat, sigma_vert, fix_rate, r_floor)
    print(f"\n=== v={v} sig_lat={sigma_lat} fr={fix_rate} r_floor={r_floor} bias={bias_deg}deg "
          f"latfixbias={lat_fix_bias_m} n_mc={n_mc} ===")
    print(f"  coast-start range captured: {np.mean(rfound):.2f} m (n={len(rfound)})")
    print(f"  COAST-START actual error (lateral binding axis):")
    print(f"    pos: mean={np.mean(perr_lat):+.4f}  std(sig_p_lat_actual)={np.std(perr_lat):.4f} m"
          f"   [KF cov sig_p_lat={fc['sig_p_lat']:.4f}]")
    print(f"    vel: mean={np.mean(verr_lat):+.4f}  std(sig_v_lat_actual)={np.std(verr_lat):.4f} m/s"
          f" [KF cov sig_v_lat={fc['sig_v_lat']:.4f}]")
    print(f"  COAST-START actual error (vertical axis):")
    print(f"    pos std={np.std(perr_vert):.4f} m  vel std={np.std(verr_vert):.4f} m/s")
    print(f"  TERMINAL miss decomposition (gate plane):")
    print(f"    lateral: mean={np.mean(miss_lat):+.4f} std={np.std(miss_lat):.4f}")
    print(f"    vertical:mean={np.mean(miss_vert):+.4f} std={np.std(miss_vert):.4f}")
    print(f"    |miss| 2D: p50={np.percentile(miss,50):.4f} p90={np.percentile(miss,90):.4f} "
          f"p99={np.percentile(miss,99):.4f}")
    # kinematic check: terminal lateral std vs sqrt(sig_p0^2 + (sig_v0*t)^2) using ACTUAL coast-start sigmas
    t_coast = r_floor / v
    pred = np.sqrt(np.std(perr_lat)**2 + (np.std(verr_lat)*t_coast)**2)
    print(f"  KINEMATIC CHECK: sqrt(sig_p0^2+(sig_v0*t_coast)^2) [actual sigmas, t={t_coast:.3f}]"
          f" = {pred:.4f}  vs terminal lateral std {np.std(miss_lat):.4f}")
    return dict(sig_v_lat_actual=float(np.std(verr_lat)), sig_p_lat_actual=float(np.std(perr_lat)),
                miss_lat_std=float(np.std(miss_lat)), miss_p90=float(np.percentile(miss, 90)),
                miss_p99=float(np.percentile(miss, 99)), cov_sig_v=fc['sig_v_lat'])


if __name__ == "__main__":
    # baseline pointed
    run(v=30.0, sigma_lat=0.10, fix_rate=0.60, r_floor=12.0, bias_deg=0.0)
    # degrade fix quality
    run(v=30.0, sigma_lat=0.15, fix_rate=0.60, r_floor=12.0, bias_deg=0.0)
    # sparse fixes (closer to inc7-ish)
    run(v=30.0, sigma_lat=0.10, fix_rate=0.20, r_floor=12.0, bias_deg=0.0)
    # accel bias 0.6 deg
    run(v=30.0, sigma_lat=0.10, fix_rate=0.60, r_floor=12.0, bias_deg=0.6, bias_mode="inplane")
    # deeper floor / shallower floor
    run(v=30.0, sigma_lat=0.10, fix_rate=0.60, r_floor=14.0, bias_deg=0.0)
    run(v=30.0, sigma_lat=0.10, fix_rate=0.60, r_floor=10.0, bias_deg=0.0)
