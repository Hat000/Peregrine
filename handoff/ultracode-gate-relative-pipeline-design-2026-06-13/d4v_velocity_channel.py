"""d4v -- VELOCITY CHANNEL for case-C gate-relative pipeline (component 4).

Velocity is obs[3:6] (vel_g = R_w2g @ vel) the inc8 policy already consumes, and it is the
SWING VARIABLE for the gate-4 0.155 m in-plane margin (estimator-racespeed REPORT, commander
refinement: per-fix/2.8 warm vs per-fix/1.6 cold -> in-plane 0.11 m vs 0.21 m, straddling the margin).

In TRUE case C vision is POSITION-ONLY. Velocity is observable only through:
  (1) POSITION-FIX DIFFERENCING -- (p_fix[k] - p_fix[k-1]) / dt  (implicit in the KF already)
  (2) a weak DIRECT vision-velocity measurement (inter-frame PnP translation delta) -- DEFERRED in
      vision-case-c P3-1, REOPENED in estimator-racespeed as a possible margin lever.
Otherwise vel = IMU integration of accel_body only, which DRIFTS with accel bias.

This file quantifies, all re-derived (no trusted prose):
  (A) ACCEL-BIAS DRIFT BOUND over a lap: v-error = b_a*T (linear), p-error = 0.5*b_a*T^2 (quadratic).
      Couples into BOTH position (the gate-4 in-plane miss) AND vel_g (the policy obs).
  (B) POSITION-FIX DIFFERENCING noise propagation: differencing two per-axis 0.265 m fixes over dt
      amplifies noise by sqrt(2)/dt. Quantify vs IMU-only velocity drift. Which wins entering gate-4?
  (C) The KF as the optimal differencer: run the REAL LinearKF and read out the velocity 1-sigma at
      the gate-4 crossing for three velocity regimes: COLD IMU-only, position-fix-difference (the KF
      default), and a candidate VISION-VELOCITY assist. Map each to the gate-4 in-plane miss.
  (D) DECISION: cold IMU-only vs position-fix-difference vs vision-velocity, with the margin
      consequence of each, and the criterion for whether the vision-velocity channel is worth building.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d4v_velocity_channel.py
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

from racer.frames import R_world_from_body  # noqa: E402
from racer.localization import FIX_COV_FLOOR_STD  # noqa: E402
from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 20260613
MARGIN_G4 = 0.155      # m, gate-4 contact-true in-plane margin @ r=0.38
BAR = 0.05             # m, the margin/3 stretch bar
V_RACE = 37.0          # m/s post-gate-3 cruise (ASSUMED, memory/plan)
G = 9.80665

# Gate-4 geometry (track_map / FACTS). Motion ~ -N; gate normal ~ -N => in-plane = (E,D); along = N.
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])
SEG = G4 - G3
SEG_LEN = float(np.linalg.norm(SEG))   # ~24.4 m
SEG_HAT = SEG / SEG_LEN
T_TRANSIT = SEG_LEN / V_RACE           # ~0.66 s g3->g4 transit

# MEASURED gate-relative per-fix LATERAL per-axis sigma (c1, reproduced bit-exact): lat_rms 0.375 ->
# per-axis = 0.375/sqrt(2) = 0.265 m. This is the gate-relative in-plane per-axis noise (map bias removed).
PER_AXIS_LAT_SIGMA = 0.265
# absolute (radial / along-track) per-axis world-fix sigma near the gate (a1/b1).
ABS_AXIS_SIGMA = 0.50

# IMU / KF noise model (state_estimator.py defaults, FACTS): white accel 0.3 m/s^2, attitude 1.4deg.
ACCEL_NOISE_STD = 0.3
IMU_HZ = 90.0
IMU_DT = 1.0 / IMU_HZ
FIX_HZ = 14.0          # 30 Hz detector x 47% acceptance ~ 14 Hz landed fixes
FIX_DT = 1.0 / FIX_HZ


# =====================================================================================
# (A) ACCEL-BIAS DRIFT BOUND -- analytic, the deterministic term the KF Q does NOT model
# =====================================================================================
def part_a_accel_bias_drift():
    """Constant accel bias b_a (or attitude bias b_theta -> phantom accel g*b_theta) drifts:
       velocity error  dv(T) = b_a * T          (LINEAR  -> couples into vel_g obs[3:6])
       position error  dp(T) = 0.5 * b_a * T^2  (QUADRATIC -> couples into the gate-4 in-plane miss)
    These are UN-modeled by Q (Q is zero-mean white) -> the KF is OVERCONFIDENT about them between
    fixes. A vision POSITION fix resets dp but NOT dv (case C: position-only) -> dv keeps integrating
    until the position-fix-difference slowly observes it.

    Bias source magnitudes (case C, the realistic bracket):
      - residual accel bias after the s18 thrust-lapse refit: small but nonzero; bound it at a few
        cm/s^2 to ~0.1 m/s^2 (the s18 work targets the b3/along-thrust axis; lateral/vertical less
        characterized). We bracket 0.03 / 0.10 m/s^2.
      - GIVEN-ATTITUDE bias: 1.4deg is the MEASURED attitude-noise 1-sigma (frames.ATTITUDE_NOISE_STD).
        A *systematic* component b_theta tilts the 9.8 m/s^2 specific-force vector -> phantom horizontal
        accel a = g*sin(b_theta) ~ g*b_theta. At 0.5deg -> 0.086 m/s^2; 1.0deg -> 0.171 m/s^2. This is
        the DOMINANT bias source (lever arm is g, not the IMU). It lands on the HORIZONTAL (in-plane E +
        along-track N) axes (gravity tilts sideways), benign on D.
    """
    out = {"note": "dv=b_a*T (linear, -> vel_g); dp=0.5*b_a*T^2 (quadratic, -> gate-4 in-plane miss)"}
    rows = []
    # bias sources -> effective horizontal accel bias
    sources = {
        "accel_bias_0.03": 0.03,
        "accel_bias_0.10": 0.10,
        "att_bias_0.5deg": G * np.sin(np.deg2rad(0.5)),
        "att_bias_1.0deg": G * np.sin(np.deg2rad(1.0)),
        "att_bias_1.4deg(meas_1sig)": G * np.sin(np.deg2rad(1.4)),
    }
    # coast durations of interest:
    #   T_transit ~0.66 s (g3->g4 leg)
    #   T_coast: worst case = time since the LAST accepted position fix. With 14 Hz fixes the typical
    #            gap is ~0.07 s, but the gate-4 terminal coast (after the gate exits FoV at r<~1.3 m,
    #            or during an acceptance dropout) can reach a few tenths of a second. We bracket
    #            0.1 / 0.3 / 0.66 / 1.0 s. A FULL LAP (~11.45 s baseline) is the unbounded-coast bound
    #            (only matters if vision drops out entirely).
    Ts = [0.1, 0.3, 0.66, 1.0, 11.45]
    for name, ba in sources.items():
        r = {"source": name, "eff_accel_bias_mps2": round(ba, 4)}
        for T in Ts:
            r[f"dv@{T}s"] = round(ba * T, 4)              # velocity error (m/s) -> vel_g obs
            r[f"dp@{T}s"] = round(0.5 * ba * T * T, 4)    # position error (m) -> in-plane miss
        rows.append(r)
    out["rows"] = rows
    # The KEY coupling numbers entering gate-4: between fixes (T~0.07-0.3 s) vs a terminal coast.
    out["key"] = {
        "between_fix_gap_s": round(FIX_DT, 4),
        "dv_per_fix_gap_att1deg": round(sources["att_bias_1.0deg"] * FIX_DT, 4),
        "dp_per_fix_gap_att1deg": round(0.5 * sources["att_bias_1.0deg"] * FIX_DT**2, 5),
        "dv_terminal_coast_0.3s_att1deg": round(sources["att_bias_1.0deg"] * 0.3, 4),
        "dp_terminal_coast_0.3s_att1deg": round(0.5 * sources["att_bias_1.0deg"] * 0.3**2, 4),
    }
    return out


# =====================================================================================
# (B) POSITION-FIX DIFFERENCING noise propagation -- the OPEN-LOOP differencer
# =====================================================================================
def part_b_fix_differencing():
    """Naive velocity from two consecutive gate-relative position fixes:
         v_hat = (p_fix[k] - p_fix[k-1]) / dt
       Each fix has per-axis noise sigma_p (lateral = 0.265 m gate-relative; radial/along ~0.50 m).
       The DIFFERENCE of two independent noisy fixes has noise sqrt(2)*sigma_p; dividing by dt:
         sigma_v(naive) = sqrt(2) * sigma_p / dt
       This is HUGE at the fix cadence: at dt=1/14 s, lateral sigma_v = sqrt(2)*0.265*14 = 5.2 m/s.
       That is why you NEVER difference raw fixes -- you let the KF do it (optimal smoothing over
       many fixes + IMU prior). Quantify the naive number to show why, and the variance-reduction the
       KF buys by averaging N fixes."""
    out = {}
    for axis, sig in [("lateral_gate_rel", PER_AXIS_LAT_SIGMA), ("radial_abs", ABS_AXIS_SIGMA)]:
        rows = []
        for dt in [FIX_DT, 0.1, 0.2, 0.5, T_TRANSIT]:
            sv = np.sqrt(2.0) * sig / dt
            rows.append({"dt_s": round(dt, 4), "sigma_v_naive_mps": round(sv, 3)})
        out[axis] = {"per_fix_sigma_m": sig, "naive_difference": rows}
    # The KF is the optimal differencer: with N fixes over the transit and an IMU velocity prior, the
    # velocity error is bounded NOT by the single-pair difference but by the regression slope variance.
    # For a least-squares slope over N fixes uniformly spaced across a window W:
    #   sigma_v(LSQ) = sigma_p * sqrt(12 / (N*(N^2-1))) / dt_fix  ~ sigma_p * sqrt(12/N) / W  for large N
    # quantify over the gate-4 window.
    W = T_TRANSIT
    N_fix = max(2, int(FIX_HZ * W))   # ~9 fixes over the 0.66 s transit
    lsq_rows = []
    for sig, label in [(PER_AXIS_LAT_SIGMA, "lateral"), (ABS_AXIS_SIGMA, "radial")]:
        # exact LSQ slope variance for N points spaced dt over window W=(N-1)*dt:
        dt = W / (N_fix - 1)
        # var(slope) = sigma^2 / sum((t_i - tbar)^2); for uniform spacing sum = dt^2 * N*(N^2-1)/12
        denom = dt**2 * N_fix * (N_fix**2 - 1) / 12.0
        sigma_v_lsq = sig / np.sqrt(denom)
        lsq_rows.append({"axis": label, "N_fix": N_fix, "window_s": round(W, 3),
                         "sigma_v_lsq_mps": round(float(sigma_v_lsq), 3)})
    out["lsq_over_window"] = {
        "note": "KF/LSQ velocity over the whole g3->g4 window (N fixes) -- the right way to difference",
        "rows": lsq_rows,
    }
    return out


# =====================================================================================
# (C) THE REAL KF: velocity 1-sigma at gate-4 crossing, three velocity regimes
# =====================================================================================
def _attitude():
    yaw = float(np.arctan2(SEG_HAT[1], SEG_HAT[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / G))   # drag-hold nose-down (sign per c1)
    R_wb = R_world_from_body(0.0, pitch, yaw)
    accel_body = R_wb.T @ (-GRAVITY_NED)           # const-v cruise specific force
    return R_wb, accel_body


def part_c_kf_velocity(n_mc=600):
    """Run the REAL LinearKF over the g3->g4 transit, perfectly-centered truth (in-plane offset 0),
    under three velocity regimes and ALSO an accel-bias perturbation. Read out:
      - velocity 1-sigma (per-axis + in-plane) at the gate-4 crossing  -> feeds vel_g obs noise
      - in-plane POSITION miss at the crossing                          -> the margin number
    Regimes:
      COLD     : KF seeded with a WRONG velocity prior (vel_std large, +1.5 m/s offset), no vision-vel
                 -> velocity observable ONLY through position-fix differencing during the transit.
      WARM     : KF seeded with the TRUE velocity (lap-converged), no vision-vel (the c1 assumption).
      VISVEL   : COLD seed + a weak vision-velocity update each fix (sigma_v = candidate). Tests whether
                 a direct velocity measurement recovers the warm-prior margin from a cold start.
    A constant horizontal accel bias (from the given-attitude systematic) is injected in all arms to
    expose the un-modeled drift the KF Q does not cover.
    """
    R_wb, accel_body_nom = _attitude()
    g = GRAVITY_NED

    # injected systematic: 0.5deg given-attitude bias -> horizontal phantom accel, on E (in-plane).
    att_bias_rad = np.deg2rad(0.5)
    a_phantom = G * np.sin(att_bias_rad)            # ~0.086 m/s^2
    # phantom accel acts in the world horizontal; put it on E (a lateral tilt). The TRUE drone does
    # NOT accelerate (we keep truth const-v); the bias is an ERROR in accel_body the KF integrates.
    accel_bias_world = np.array([0.0, a_phantom, 0.0])
    # accel_body the KF actually reads = true specific force + bias expressed in body frame
    accel_body_biased = accel_body_nom + R_wb.T @ accel_bias_world

    p0 = G3.copy()
    v0 = SEG_HAT * V_RACE
    n_steps = int(T_TRANSIT / IMU_DT)

    _REGIME_SEED = {"cold": 1, "warm": 2, "visvel": 3}   # deterministic (no PYTHONHASHSEED dep)
    def run(regime, vis_vel_sigma=None, n=n_mc):
        v_err_E, v_err_D, v_err_N = [], [], []
        ip_pos = []
        rseed = _REGIME_SEED[regime] * 100003 + int((vis_vel_sigma or 0) * 1000)
        for s in range(n):
            r = np.random.default_rng(SEED + 13 * s + rseed)
            if regime == "warm":
                v_init = v0.copy()
                vel_std0 = 0.15
            else:  # cold / visvel
                v_init = v0 + np.array([0.0, 1.5, 0.0])   # +1.5 m/s wrong lateral vel prior
                vel_std0 = 1.5
            kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v_init,
                                     pos_std=0.5, vel_std=vel_std0,
                                     accel_noise_std=ACCEL_NOISE_STD)
            rk = RewindKF(kf=kf, horizon_s=0.5)
            t = 0.0
            t_ns = 0
            next_fix = 0.0
            for k in range(n_steps):
                t += IMU_DT
                t_ns += int(IMU_DT * 1e9)
                # KF reads the BIASED accel (+ white IMU noise); truth stays const-v.
                a_meas = accel_body_biased + r.normal(0, ACCEL_NOISE_STD, 3)
                rk.predict(a_meas, R_wb, IMU_DT, t_ns)
                p_true = p0 + SEG_HAT * V_RACE * t
                rng_to_g4 = float(np.linalg.norm(G4 - p_true))
                if t >= next_fix and rng_to_g4 < 12.0:
                    next_fix += FIX_DT
                    # gate-relative position fix: zero-mean lateral noise (map bias removed), radial
                    # along-track noise. NO 0.40 floor (no bias to absorb) -- the c1 'rel' arm model.
                    nE = r.normal(0, PER_AXIS_LAT_SIGMA)
                    nD = r.normal(0, PER_AXIS_LAT_SIGMA)
                    nN = r.normal(0, ABS_AXIS_SIGMA)
                    z = p_true + np.array([nN, nE, nD])
                    cov = np.diag([ABS_AXIS_SIGMA**2, PER_AXIS_LAT_SIGMA**2, PER_AXIS_LAT_SIGMA**2])
                    rk.update_position(z, cov, sim_time_ns=t_ns)
                    if regime == "visvel" and vis_vel_sigma is not None:
                        # weak DIRECT vision-velocity update (inter-frame PnP translation delta).
                        # measurement = true velocity + noise; observes vel directly (H_vel).
                        nv = r.normal(0, vis_vel_sigma, 3)
                        zv = v0 + nv
                        Rv = (vis_vel_sigma**2) * np.eye(3)
                        rk.kf.update_velocity(zv, Rv)
            # at gate-4 plane: velocity + in-plane position error
            v_est = rk.velocity
            v_err = v_est - v0
            v_err_E.append(v_err[1]); v_err_D.append(v_err[2]); v_err_N.append(v_err[0])
            p_true_final = p0 + SEG_HAT * V_RACE * (n_steps * IMU_DT)
            perr = rk.position - p_true_final
            ip_pos.append(float(np.hypot(perr[1], perr[2])))
        vE = np.array(v_err_E); vD = np.array(v_err_D); vN = np.array(v_err_N)
        ip = np.array(ip_pos)
        return dict(
            regime=regime,
            vis_vel_sigma=vis_vel_sigma,
            vel_sigma_E=float(vE.std()), vel_bias_E=float(vE.mean()),
            vel_sigma_D=float(vD.std()),
            vel_sigma_N=float(vN.std()), vel_bias_N=float(vN.mean()),
            vel_inplane_sigma=float(np.hypot(vE.std(), vD.std())),
            vel_inplane_rms=float(np.sqrt(np.mean(vE**2 + vD**2))),
            inplane_pos_rms=float(np.sqrt(np.mean(ip**2))),
            inplane_pos_p90=float(np.percentile(ip, 90)),
        )

    res = {"injected_att_bias_deg": 0.5, "a_phantom_mps2": round(float(a_phantom), 4)}
    res["cold"] = run("cold")
    res["warm"] = run("warm")
    # candidate vision-velocity accuracies (inter-frame PnP translation delta over 1/30 s):
    # a PnP translation delta has ~per-fix position noise / inter-frame dt -> noisy. A plausible
    # accuracy after smoothing a few frames is ~0.3-1.0 m/s. Sweep.
    for vvs in [1.0, 0.5, 0.3]:
        res[f"visvel_{vvs}"] = run("visvel", vis_vel_sigma=vvs)
    return res


def main():
    np.random.seed(SEED)
    results = {"seed": SEED, "margin_g4_m": MARGIN_G4, "bar_m": BAR, "v_race": V_RACE,
               "t_transit_s": round(T_TRANSIT, 4)}

    print("=" * 88)
    print("(A) ACCEL-BIAS DRIFT BOUND  -- dv=b_a*T (vel_g),  dp=0.5*b_a*T^2 (gate-4 in-plane)")
    print("=" * 88)
    a = part_a_accel_bias_drift()
    results["part_a"] = a
    print(f"  {'source':>28} {'a_bias':>8} {'dv@.3s':>7} {'dp@.3s':>7} {'dv@.66s':>8} "
          f"{'dp@.66s':>8} {'dv@11.45':>9} {'dp@11.45':>9}")
    for r in a["rows"]:
        print(f"  {r['source']:>28} {r['eff_accel_bias_mps2']:8.4f} {r['dv@0.3s']:7.3f} "
              f"{r['dp@0.3s']:7.3f} {r['dv@0.66s']:8.3f} {r['dp@0.66s']:8.3f} "
              f"{r['dv@11.45s']:9.3f} {r['dp@11.45s']:9.3f}")
    print(f"\n  KEY: between-fix gap {a['key']['between_fix_gap_s']}s @1deg-att: "
          f"dv={a['key']['dv_per_fix_gap_att1deg']} m/s, dp={a['key']['dp_per_fix_gap_att1deg']} m")
    print(f"       terminal coast 0.3s @1deg-att: dv={a['key']['dv_terminal_coast_0.3s_att1deg']} m/s, "
          f"dp={a['key']['dp_terminal_coast_0.3s_att1deg']} m")

    print("\n" + "=" * 88)
    print("(B) POSITION-FIX DIFFERENCING noise:  sigma_v(naive) = sqrt(2)*sigma_p/dt")
    print("=" * 88)
    b = part_b_fix_differencing()
    results["part_b"] = b
    for axis in ("lateral_gate_rel", "radial_abs"):
        print(f"\n  {axis} (per-fix sigma_p = {b[axis]['per_fix_sigma_m']} m):")
        for row in b[axis]["naive_difference"]:
            print(f"    dt={row['dt_s']:.4f}s -> sigma_v_naive = {row['sigma_v_naive_mps']:8.3f} m/s")
    print(f"\n  KF/LSQ over the g3->g4 window (the RIGHT way):")
    for row in b["lsq_over_window"]["rows"]:
        print(f"    {row['axis']:>8}: N={row['N_fix']} fixes over {row['window_s']}s "
              f"-> sigma_v_lsq = {row['sigma_v_lsq_mps']:.3f} m/s")

    print("\n" + "=" * 88)
    print("(C) REAL LinearKF velocity 1-sigma + in-plane POSITION miss at gate-4 (with 0.5deg att-bias)")
    print("=" * 88)
    c = part_c_kf_velocity()
    results["part_c"] = c
    print(f"  injected att-bias 0.5deg -> phantom horiz accel {c['a_phantom_mps2']} m/s^2 (on E)")
    print(f"\n  {'regime':>12} {'visvel_sig':>10} {'velE_sig':>8} {'velE_bias':>9} {'vel_ip_sig':>10} "
          f"{'pos_ip_rms':>10} {'pos_ip_p90':>10} {'<margin?':>9}")
    for key in ("cold", "warm", "visvel_1.0", "visvel_0.5", "visvel_0.3"):
        x = c[key]
        vv = f"{x['vis_vel_sigma']}" if x['vis_vel_sigma'] else "-"
        marg = "YES" if x["inplane_pos_rms"] < MARGIN_G4 else "NO"
        print(f"  {key:>12} {vv:>10} {x['vel_sigma_E']:8.3f} {x['vel_bias_E']:+9.3f} "
              f"{x['vel_inplane_sigma']:10.3f} {x['inplane_pos_rms']:10.3f} "
              f"{x['inplane_pos_p90']:10.3f} {marg:>9}")

    outpath = Path(__file__).resolve().parent / "d4v_velocity_channel_results.json"
    outpath.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {outpath}")
    return results


if __name__ == "__main__":
    main()
