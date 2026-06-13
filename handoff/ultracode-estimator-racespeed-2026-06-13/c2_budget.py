"""c2_budget.py -- analytic cross-checks for the cadence branch:
  (1) how many EFFECTIVE independent fixes to bring per-fix 0.50-0.60 m down to 0.05 m by pure
      1/sqrt(N) averaging (the ideal ceiling -- KF does WORSE);
  (2) the cadence physically required at 37 m/s to land that many fixes in the gate-4 window;
  (3) the COMPUTE budget: is >30 Hz even feasible on the eval HW? (edge vs CPU detector +
      PnP chain + the RewindKF replay cost that GROWS with cadence).

Pure arithmetic on the MEASURED latency_results.json + a3 cadence numbers + c2 sim outputs.
Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c2_budget.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAT = ROOT / "handoff" / "ultracode-vision-case-c-2026-06-13" / "latency_results.json"

SEED = 20260613
np.random.seed(SEED)

# ---- per-fix in-plane 1-sigma at gate-4 approach (A1/A3 measured) ----
PER_FIX_SIGMA = {"near_<=8m": 0.60, "mid_8-16m": 0.50}   # in-plane per-axis, from a1 cov model
SPEED = 37.0
SEG_LEN = 24.39
T_WINDOW = SEG_LEN / SPEED                                # ~0.659 s

# gate-4 in-plane axes are E,D. The bar is per-axis 1-sigma < 0.05 m (variance/3 of margin),
# i.e. in-plane combined sqrt(E^2+D^2) target ~ 0.05*sqrt(2) ~ 0.0707 m if both axes hit 0.05.
PER_AXIS_BAR = 0.05


def n_ideal_to_reach(per_fix_sigma, target_sigma):
    """Ideal independent-fix count to crush per_fix_sigma -> target by 1/sqrt(N)."""
    return (per_fix_sigma / target_sigma) ** 2


def main():
    lat = json.loads(LAT.read_text())
    cad = json.loads((HERE / "c2_cadence_results.json").read_text())

    out = {"seed": SEED}

    # (1) ideal 1/sqrt(N) ceiling -- the BEST cadence could do if every extra fix were independent
    ideal = {}
    for band, s in PER_FIX_SIGMA.items():
        n_for_bar = n_ideal_to_reach(s, PER_AXIS_BAR)
        ideal[band] = dict(
            per_fix_sigma=s,
            N_ideal_for_005=float(n_for_bar),
            note=f"IDEAL independent fixes to reach 0.05 m/axis from {s} m; KF does WORSE (corr).",
        )
    out["ideal_1oversqrtN_ceiling"] = ideal

    # The KF's REALIZED efficiency vs ideal (from c2 sim, debiased 100% arm): how many real
    # landed fixes it takes the KF to reach a given filtered sigma, vs ideal.
    deb100 = cad["effective_fix_analysis_100pct"]
    # the KF needs N_real such that filtered ~ 0.05; extrapolate from the realized curve.
    # filtered_sigma ~ per_fix / sqrt(N_eff); N_eff grows sublinearly in N_real (corr penalty).
    # fit N_eff = alpha * N_real^beta on the realized data, then invert for filtered=0.05.
    Nreal = np.array([r["mean_fixes_applied"] for r in deb100])
    Neff = np.array([r["N_eff_from_variance_E"] for r in deb100])
    # log-log fit N_eff = alpha N_real^beta
    b, loga = np.polyfit(np.log(Nreal), np.log(Neff), 1)
    alpha = float(np.exp(loga)); beta = float(b)
    # filtered_E = per_fix / sqrt(N_eff); per_fix ~ 0.60 near gate. need filtered=0.05 ->
    # N_eff_needed = (0.60/0.05)^2 = 144. invert N_eff=alpha N_real^beta -> N_real.
    per_fix_near = 0.60
    Neff_needed = (per_fix_near / PER_AXIS_BAR) ** 2
    Nreal_needed = float((Neff_needed / alpha) ** (1.0 / beta))
    out["kf_realized_efficiency"] = dict(
        Neff_vs_Nreal_fit=f"N_eff = {alpha:.3f} * N_real^{beta:.3f}",
        Neff_needed_for_005=float(Neff_needed),
        Nreal_fixes_needed_for_005=Nreal_needed,
        note=("KF N_eff grows ~N_real^%.2f (sublinear: vel unobservable in case C so packed "
              "fixes are correlated). Need ~%.0f REAL landed fixes for 0.05 m/axis -- vs ~%.0f "
              "ideal independent." % (beta, Nreal_needed, Neff_needed)),
    )

    # (2) cadence physically required to land Nreal_needed fixes in the gate-4 window
    # at 47% acceptance and at 100% acceptance, over T_WINDOW.
    for acc_name, acc in [("47pct", 0.47), ("100pct", 1.0)]:
        cad_needed = Nreal_needed / (acc * T_WINDOW)
        out.setdefault("cadence_required_for_005", {})[acc_name] = dict(
            acceptance=acc, t_window_s=T_WINDOW,
            detector_hz_required=float(cad_needed),
            fix_spacing_m=float(SPEED / cad_needed),
            note=("detector Hz to land ~%.0f fixes @ %s acceptance over the %.2f s g3->g4 "
                  "window at 37 m/s" % (Nreal_needed, acc_name, T_WINDOW)),
        )

    # a3 REALITY: at 37 m/s the detector lands only ~7-8 accepted fixes over the whole approach
    # (30 Hz * 47% * window, with the [3,24] m accept range). The required cadence is ORDERS above.
    out["a3_reality_check"] = dict(
        a3_effective_accepted_fixes="~7-8 over the whole g3->g4 approach (30Hz, 37 m/s)",
        a3_fix_spacing_m_at_30Hz=1.29,
        verdict=("a3 measures ~7-8 real fixes; reaching 0.05 m would need ~%.0f -- a >%.0fx "
                 "increase in landed fixes, impossible at the realistic accept window."
                 % (Nreal_needed, Nreal_needed / 7.5)),
    )

    # (3) COMPUTE budget -- is >30 Hz feasible on the eval HW?
    edge = lat["stage_ii_detector_edge_estimate"]
    cpu = lat["stage_ii_detector_cpu_upper_bound"]
    pnp_chain_p50 = lat["budget"]["budgets"][0]["pnp_chain_p50_ms"]   # 0.767 ms (same both HW)
    pnp_chain_p90 = lat["budget"]["budgets"][0]["pnp_chain_p90_ms"]

    # RewindKF replay cost GROWS with cadence: each fix replays up to horizon_s*imu_hz predicts;
    # but also more fixes/sec. Per the kf_rewind_buffer docstring: ~10 us/predict, ~45 predicts
    # for 0.5 s @ 90 Hz -> <0.5 ms/fix. We model rewind cost/sec = fixes/sec * replay_predicts*10us.
    HORIZON_S, IMU_HZ, US_PER_PREDICT = 0.5, 90.0, 10e-3   # 10 us = 0.01 ms
    replay_predicts = HORIZON_S * IMU_HZ                    # ~45 (worst-case full-horizon rewind)
    rewind_ms_per_fix = replay_predicts * US_PER_PREDICT    # ~0.45 ms

    budget = {}
    for hw, det_p50, det_p90, label in [
        ("edge", edge["edge_fwd_p50_ms"], edge["edge_fwd_p90_ms"], "~100 TOPS edge"),
        ("cpu_upper_bound", cpu["p50_ms"], cpu["p90_ms"], "laptop CPU (UPPER BOUND)"),
    ]:
        # per-frame serial pipeline cost (detector + PnP chain). KF/rewind cost is per-fix.
        frame_ms_p50 = det_p50 + pnp_chain_p50 + rewind_ms_per_fix
        frame_ms_p90 = det_p90 + pnp_chain_p90 + rewind_ms_per_fix
        max_hz_p50 = 1000.0 / frame_ms_p50
        max_hz_p90 = 1000.0 / frame_ms_p90
        budget[hw] = dict(
            label=label,
            detector_p50_ms=det_p50, detector_p90_ms=det_p90,
            pnp_chain_p50_ms=pnp_chain_p50, pnp_chain_p90_ms=pnp_chain_p90,
            rewind_ms_per_fix=rewind_ms_per_fix,
            frame_ms_p50=frame_ms_p50, frame_ms_p90=frame_ms_p90,
            max_throughput_hz_p50=max_hz_p50, max_throughput_hz_p90=max_hz_p90,
            supports_60Hz=bool(max_hz_p90 >= 60),
            supports_120Hz=bool(max_hz_p90 >= 120),
            note=("Serial single-thread bound. Detector is the gate; PnP chain 0.77ms + rewind "
                  "%.2fms are negligible next to the detector forward pass." % rewind_ms_per_fix),
        )
    out["compute_budget"] = budget

    # The decisive feasibility sentence: even if the eval HW could run 60-100 Hz (edge can),
    # the EXPOSURE/blur and 4-corner-visibility geometry, NOT compute, cap the USABLE fix rate
    # at 37 m/s -- and even the impossible 240 Hz cell does not clear the bar (c2 sim).
    out["feasibility_verdict"] = dict(
        edge_max_hz_p90=float(budget["edge"]["max_throughput_hz_p90"]),
        cpu_max_hz_p90=float(budget["cpu_upper_bound"]["max_throughput_hz_p90"]),
        compute_allows_gt_30Hz_on_edge=bool(budget["edge"]["max_throughput_hz_p90"] > 30),
        compute_allows_gt_30Hz_on_cpu=bool(budget["cpu_upper_bound"]["max_throughput_hz_p90"] > 30),
        c2_sim_240Hz_debiased_inplane_std_m=0.119,   # from c2_cadence_results (100% accept)
        c2_sim_cadence_inf_floor_m=cad["cadence_saturation"]["sigma_floor_as_cadence_inf_m"],
        bottom_line=("Compute permits >30 Hz ONLY on edge HW (edge p90 ~%.0f Hz cap); CPU is "
                     "capped at ~%.1f Hz and cannot even sustain 30 Hz. BUT cadence is NOT the "
                     "binding constraint: even infinite cadence floors the DEBIASED in-plane "
                     "1-sigma at ~%.3f m (>0.05 m), and the RAW (real, biased) error never drops "
                     "below ~0.30 m at any cadence. More fixes do NOT close the gap."
                     % (budget["edge"]["max_throughput_hz_p90"],
                        budget["cpu_upper_bound"]["max_throughput_hz_p90"],
                        cad["cadence_saturation"]["sigma_floor_as_cadence_inf_m"])),
    )

    (HERE / "c2_budget_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'c2_budget_results.json'}")


if __name__ == "__main__":
    main()
