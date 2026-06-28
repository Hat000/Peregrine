"""AHRS T1 Benchmark: ESKF vs IEKF vs Madgwick vs Mahony on synthetic high-g IMU.

The #1 VQ2 frontier build. VQ2's competitive wire BLOCKS the ATTITUDE message
(VADR-TS-003 §9.3), so the LinearKF in state_estimator.py (which trusts given attitude)
cannot run; a self-attitude estimator from HIGHRES_IMU is mandatory. This bench picks the
filter.

Runs all four filters on 5 scenarios (static/spin/roll/high-g-pull/high-g-random), reports
geodesic attitude error (mean/p90/max), and highlights the discriminating high-g result.

HEADLINE FINDING (synthetic, this bench)
----------------------------------------
The model-based filters (ESKF / IEKF) beat the classical complementary filters under
high-g, but ONLY with a direction-aware *innovation* (chi-square) gate -- magnitude
gating alone is insufficient because a random-direction acceleration whose MAGNITUDE
sits near g (~43%% of HIGH_G_RANDOM samples) sails through the magnitude gate while its
direction is wrong. With the two-gate scheme:
    HIGH_G_RANDOM  : ESKF/IEKF p90 ~0.65 deg vs Madgwick ~0.85, Mahony ~5.8
    HIGH_G_PULL    : ESKF/IEKF p90 ~6.5 deg  vs Madgwick ~27,   Mahony ~82
    STATIC/SPIN    : ESKF/IEKF p90 ~0.11 deg (best); Mahony ~0.09; Madgwick ~0.20
The IEKF (left-invariant, Barrau-Bonnabel / van Goor EqF) matches the ESKF to ~1e-11 deg
on this attitude-only problem -- it cross-validates the ESKF; its consistency advantage is
realised on the coupled SE_2(3)/INS problem, not attitude-from-gravity. See README in
src/racer/ahrs/ for the full table and the why.

Usage:
    python scripts/benches/ahrs_bench.py              # default scenarios
    python scripts/benches/ahrs_bench.py --high-g-only
    python scripts/benches/ahrs_bench.py --scenario HIGH_G_PULL --duration 10

To plug in real twin IMU data (later on Adroit):
    Build an IMUSequence manually from HIGHRES_IMU recordings and pass it via
    --real-seq (not yet wired — add a loader that reads from extract_run.py output).
    The generator interface is designed to be a drop-in: swap generate_imu_sequence()
    for your real sequence loader and the rest is identical.

Extension stubs for Fengyou's morning decision:
  1. Learned AHRS: RIANN / GRU gyro-denoising
       Replace the raw gyro in ESKFAHRS._predict() with: omega_clean = denoiser(gyro_window)
       Reference: Brossard M et al. (2020) "AI-IMU Dead-Reckoning." IEEE T-ITS.
                  van Goor P (2022) ANU thesis — Section 3 (EqVIO / IEKF).
       Priority: HIGH if ESKF plateau is still above 1-2 deg at VQ2 speeds (gyro bias random walk dominates).

  2. Invariant EKF / EqVIO (DONE for SO(3): src/racer/ahrs/iekf.py):
       The left-invariant SO(3) attitude+bias filter is implemented and benchmarked.
       It matches the ESKF on attitude-only. The OPEN extension is SE_2(3): fold in
       velocity+position so the invariant consistency advantage actually bites (this is
       the natural merge point with the C2 vision-estimator / EqVIO line).
       Priority: MEDIUM — pursue when AHRS couples to translation/VIO.

  3. This bench itself as a real-data validator:
       Twin HIGHRES_IMU at ~200 Hz is already in the recording format (extract_run.py).
       Build a loader: t, gyro, accel = load_highres_imu(recording_path)
       Pass as IMUSequence(t=t, q_wxyz_gt=None, gyro=gyro, accel=accel).
       GT attitude from ODOMETRY (apply true_attitude_from_odo_quat_wxyz).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from racer.ahrs.imu_gen import generate_imu_sequence, Scenario
from racer.ahrs.eskf import ESKFAHRS
from racer.ahrs.iekf import LeftInvariantEKF
from racer.ahrs.classical import MadgwickAHRS, MahonyAHRS
from racer.ahrs.metrics import geodesic_error_rad, score_filter, run_filter_on_sequence


# Names of the model-based (Kalman) filters; the rest are classical comparators.
# Used by the high-g verdict to compare best-model vs best-classical.
MODEL_FILTER_PREFIXES = ("ESKF", "IEKF")


# ---------------------------------------------------------------------------
# Filter factory (tuned defaults)
# ---------------------------------------------------------------------------

def make_filters():
    """Instantiate all benchmarked filters with tuned defaults.

    ESKF and IEKF both use the two-gate robustness scheme (magnitude gate +
    chi-square innovation gate). Madgwick/Mahony are the classical comparators.
    """
    eskf = ESKFAHRS(
        gyro_noise_std=0.01,
        gyro_bias_std=1e-4,
        accel_noise_std=0.3,
        accel_gate_alpha=10.0,    # magnitude-gate sharpness
        accel_chi2_thresh=7.815,  # chi2(3,.95) innovation gate
    )
    iekf = LeftInvariantEKF(
        gyro_noise_std=0.01,
        gyro_bias_std=1e-4,
        accel_noise_std=0.3,
        accel_gate_alpha=10.0,
        accel_chi2_thresh=7.815,
    )
    madgwick = MadgwickAHRS(beta=0.1)
    mahony = MahonyAHRS(kp=2.0, ki=0.005)
    return [
        ("ESKF (a=10,chi2)", eskf),
        ("IEKF (a=10,chi2)", iekf),
        ("Madgwick (b=0.1)", madgwick),
        ("Mahony (kp=2,ki=0.005)", mahony),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: Scenario,
    duration_s: float = 5.0,
    dt: float = 0.005,
    gyro_noise_std: float = 0.01,
    accel_noise_std: float = 0.05,
    seed: int = 42,
):
    """Run all filters on one scenario and return results dict."""
    seq = generate_imu_sequence(
        scenario,
        duration_s=duration_s,
        dt=dt,
        gyro_noise_std=gyro_noise_std,
        accel_noise_std=accel_noise_std,
        seed=seed,
    )

    results = {}
    for name, filt in make_filters():
        q_est = run_filter_on_sequence(filt, seq)
        errors = geodesic_error_rad(q_est, seq.q_wxyz_gt)
        stats = score_filter(errors)
        results[name] = stats

    return seq, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(scenario: Scenario, results: dict) -> None:
    """Print a comparison table for one scenario."""
    col_w = 26
    hdr = f"\n{'='*70}\nScenario: {scenario.name}\n{'='*70}"
    print(hdr)
    print(f"{'Filter':<{col_w}}  {'mean':>7}  {'p90':>7}  {'max':>7}  {'rms':>7}  (deg)")
    print(f"{'-'*col_w}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name, stats in results.items():
        print(f"{name:<{col_w}}  {stats['mean_deg']:7.2f}  {stats['p90_deg']:7.2f}  "
              f"{stats['max_deg']:7.2f}  {stats['rms_deg']:7.2f}")


def print_high_g_verdict(all_results: dict) -> None:
    """Print a one-line verdict for the discriminating high-g scenarios."""
    print("\n" + "=" * 70)
    print("HIGH-G DISCRIMINATING VERDICT")
    print("=" * 70)
    for scen_name in ["HIGH_G_PULL", "HIGH_G_RANDOM"]:
        if scen_name not in all_results:
            continue
        results = all_results[scen_name]

        def _is_model(k):
            return any(k.startswith(p) for p in MODEL_FILTER_PREFIXES)

        model_p90 = min(v["p90_deg"] for k, v in results.items() if _is_model(k))
        classical_p90 = min(v["p90_deg"] for k, v in results.items() if not _is_model(k))
        improvement = classical_p90 / max(model_p90, 0.01)
        verdict = "PASS" if model_p90 < classical_p90 else "FAIL (model not beating classical)"
        print(f"  {scen_name}: best-model p90={model_p90:.2f} deg, "
              f"best-classical p90={classical_p90:.2f} deg, "
              f"improvement={improvement:.1f}x  [{verdict}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AHRS filter benchmark (synthetic IMU)")
    parser.add_argument("--high-g-only", action="store_true",
                        help="Run only HIGH_G scenarios")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a single scenario by name (e.g. HIGH_G_PULL)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Sequence duration in seconds (default: 5.0)")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="IMU timestep in seconds (default: 0.005 = 200 Hz)")
    args = parser.parse_args()

    if args.scenario:
        scenarios = [Scenario[args.scenario]]
    elif args.high_g_only:
        scenarios = [Scenario.HIGH_G_PULL, Scenario.HIGH_G_RANDOM]
    else:
        scenarios = list(Scenario)

    print(f"\nAHRS T1 BENCH — {len(scenarios)} scenario(s), dt={args.dt*1000:.1f} ms, "
          f"duration={args.duration:.1f} s")
    print("(skip_init = first 100 steps = {:.2f} s)".format(100 * args.dt))

    all_results = {}
    for scen in scenarios:
        _, results = run_scenario(scen, duration_s=args.duration, dt=args.dt)
        print_table(scen, results)
        all_results[scen.name] = results

    print_high_g_verdict(all_results)
    print()


if __name__ == "__main__":
    main()
