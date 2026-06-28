"""AHRS T1 Benchmark: compare ESKF vs Madgwick vs Mahony on synthetic high-g IMU.

Runs all three filters on 5 scenarios (static/spin/roll/high-g-pull/high-g-random),
reports geodesic attitude error (mean/p90/max), and highlights the discriminating
high-g result where ESKF accel-gating should outperform classical filters.

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

  2. Invariant EKF / EqVIO (SO(3) right-invariant):
       Swap _compute_F and _accel_H in eskf.py for Lie-group Jacobians.
       Advantage: consistency maintained under large attitudes (no linearisation error).
       Priority: MEDIUM — try if ESKF shows bias on long roll/yaw arcs.

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
from racer.ahrs.classical import MadgwickAHRS, MahonyAHRS
from racer.ahrs.metrics import geodesic_error_rad, score_filter, run_filter_on_sequence


# ---------------------------------------------------------------------------
# Filter factory (tuned defaults)
# ---------------------------------------------------------------------------

def make_filters():
    """Instantiate all three filters with tuned defaults."""
    eskf = ESKFAHRS(
        gyro_noise_std=0.01,
        gyro_bias_std=1e-4,
        accel_noise_std=0.3,
        accel_gate_alpha=10.0,  # key parameter: gating sharpness
    )
    madgwick = MadgwickAHRS(beta=0.1)
    mahony = MahonyAHRS(kp=2.0, ki=0.005)
    return [
        ("ESKF (alpha=10)", eskf),
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
        eskf_p90 = min(v["p90_deg"] for k, v in results.items() if "ESKF" in k)
        classical_p90 = max(v["p90_deg"] for k, v in results.items() if "ESKF" not in k)
        improvement = classical_p90 / max(eskf_p90, 0.01)
        verdict = "PASS" if eskf_p90 < classical_p90 else "FAIL (ESKF not beating classical)"
        print(f"  {scen_name}: ESKF p90={eskf_p90:.2f} deg, "
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
