"""scripts/vq2_loadday/imu_profile.py — VQ2 load-day IMU characterisation (C2/C8).

Probes HIGHRES_IMU: actual message rate, per-axis noise floor (static std at rest),
saturation/clipping at high-g (are we hitting the accelerometer ceiling during manoeuvres?),
and magnetometer presence + usability (as a yaw anchor).

These feed two forks:
  C2  Magnetometer fidelity: usable mag -> AHRS yaw-anchor viable.
      No mag (or too noisy) -> vision-only heading -> gate-in-view + GRU memory critical.
  C8  IMU rate/noise/saturation: actual Hz + noise floor inform the AHRS design choice
      (classical EKF vs learned), and the IMU-sat model-substitution (Modified-Polar at
      high-g -> swap simulated IMU if clipping distorts the EKF update).

Usage (live sim, pump 5 s pre-arm for noise floor):
  python scripts/vq2_loadday/imu_profile.py --seconds 5

Usage (replay a tlog):
  python scripts/vq2_loadday/imu_profile.py --tlog data/runs/<session>/mavlink.tlog

Usage (save JSON for later analysis):
  python scripts/vq2_loadday/imu_profile.py --seconds 5 --out data/imu_profile.json

Degrades gracefully: "stream absent / not connected" when the sim is unreachable.
No crash when mag field is zero or absent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer.firstcontact import MessageRateTracker, pump_for
from racer.mavlink_client import MavlinkClient

# g constant — used to check if the accelerometer is near saturation
_G = 9.80665

# Magnetometer noise floor threshold: below this std the sensor is probably not present
# or is all-zeros (VQ2 may not simulate a magnetometer).
_MAG_NOISE_THRESHOLD = 1e-4  # uT equivalent

# IMU saturation check: if any axis max |accel| exceeds this fraction of the typical
# max-g spec (16g for common MEMS parts), flag it as potentially clipping.
# For the sim this is a proxy — we can't read the actual clip register.
_ACCEL_CLIP_THRESHOLD_MS2 = 120.0  # ~12g; common MEMS floor


def _collect_imu_samples(
    client: MavlinkClient, seconds: float
) -> list[dict]:
    """Pump client for `seconds` and return list of {time_usec, acc, mag, baro}."""
    samples = []

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        client.pump()
        s = client.state
        if s.accel_body is not None and s.sim_time_ns > 0:
            samples.append({
                "time_usec": s.sim_time_ns // 1000,
                "recv_ns": s.recv_monotonic_ns,
                "acc": s.accel_body.copy().tolist(),
                "mag": s.mag_body.copy().tolist() if s.mag_body is not None else None,
                "baro_hpa": s.baro_pressure_hpa,
            })
        time.sleep(0.005)
    return samples


def _drain_tlog_samples(tlog_path: Path, client: MavlinkClient) -> list[dict]:
    """Replay a tlog and collect the same per-IMU-tick samples."""
    import os
    os.environ.setdefault("MAVLINK20", "1")
    from pymavlink import mavutil

    conn = mavutil.mavlink_connection(str(tlog_path), input=True)
    samples = []
    while True:
        msg = conn.recv_match(blocking=False)
        if msg is None:
            break
        if msg.get_type() == "BAD_DATA":
            continue
        client._handle(msg)
        s = client.state
        if msg.get_type() == "HIGHRES_IMU" and s.accel_body is not None:
            samples.append({
                "time_usec": int(msg.time_usec),
                "recv_ns": time.monotonic_ns(),
                "acc": s.accel_body.copy().tolist(),
                "mag": s.mag_body.copy().tolist() if s.mag_body is not None else None,
                "baro_hpa": s.baro_pressure_hpa,
            })
    return samples


def _analyze(samples: list[dict], tracker_hz: float | None = None) -> dict:
    """Compute the IMU profile metrics from a list of tick dicts."""
    if not samples:
        return {"error": "no IMU samples collected"}

    n = len(samples)
    acc = np.array([s["acc"] for s in samples], dtype=np.float64)  # (N, 3)

    # Rate: use recv timestamps when we have them; else sample count / first-last span
    recv_ns = [s["recv_ns"] for s in samples if s["recv_ns"] > 0]
    if len(recv_ns) > 1:
        span_s = (recv_ns[-1] - recv_ns[0]) / 1e9
        rate_hz = (len(recv_ns) - 1) / span_s if span_s > 0 else 0.0
    elif tracker_hz is not None:
        rate_hz = tracker_hz
    else:
        times = [s["time_usec"] for s in samples]
        span_us = times[-1] - times[0]
        rate_hz = (n - 1) / (span_us / 1e6) if span_us > 0 else 0.0

    # Noise floor: std per axis (best measured at rest / pre-arm; in flight it's dominated
    # by vibration, but still shows if sensors are reading anything).
    acc_mean = acc.mean(axis=0)
    acc_std = acc.std(axis=0)

    # Saturation check: did any sample hit the clipping ceiling?
    acc_max = np.abs(acc).max(axis=0)
    any_clipped = bool(np.any(acc_max > _ACCEL_CLIP_THRESHOLD_MS2))

    # Net specific-force magnitude — should be ~g at rest; deviation = vibration / accel
    acc_norm = np.linalg.norm(acc, axis=1)
    acc_norm_mean = float(acc_norm.mean())
    acc_norm_std = float(acc_norm.std())

    # Magnetometer
    mag_samples = [s["mag"] for s in samples if s["mag"] is not None]
    mag_result: dict
    if not mag_samples:
        mag_result = {"present": False, "note": "mag_body always None (absent or not emitted)"}
    else:
        mag = np.array(mag_samples, dtype=np.float64)
        mag_mean = mag.mean(axis=0)
        mag_std = mag.std(axis=0)
        mag_norm = np.linalg.norm(mag, axis=1)
        # Is it all-zero (VQ2 may not simulate a real magnetometer)?
        all_zero = bool(np.allclose(mag, 0.0, atol=1e-9))
        # Is it constant (no useful variation for heading)?
        low_variance = bool(np.all(mag_std < _MAG_NOISE_THRESHOLD))
        mag_result = {
            "present": True,
            "all_zero": all_zero,
            "low_variance": low_variance,
            "mean_nT": mag_mean.tolist(),
            "std_nT": mag_std.tolist(),
            "norm_mean": float(mag_norm.mean()),
            "norm_std": float(mag_norm.std()),
            "usable_as_yaw_anchor": (not all_zero and not low_variance),
            "note": (
                "all-zero — magnetometer not simulated" if all_zero
                else "low-variance — may not be usable as yaw anchor" if low_variance
                else "appears to have variation; usability needs correlation vs ODO yaw"
            ),
        }

    # Baro presence
    baro_vals = [s["baro_hpa"] for s in samples if s["baro_hpa"] is not None]
    baro_result = {
        "present": len(baro_vals) > 0,
        "mean_hpa": float(np.mean(baro_vals)) if baro_vals else None,
        "std_hpa": float(np.std(baro_vals)) if baro_vals else None,
    }

    return {
        "n_samples": n,
        "rate_hz": round(rate_hz, 2),
        "accel_mean_frd_ms2": acc_mean.tolist(),
        "accel_std_frd_ms2": acc_std.tolist(),
        "accel_max_abs_frd_ms2": acc_max.tolist(),
        "accel_norm_mean_ms2": round(acc_norm_mean, 4),
        "accel_norm_std_ms2": round(acc_norm_std, 4),
        "accel_norm_vs_g_pct": round(100.0 * (acc_norm_mean / _G - 1.0), 3),
        "accel_clipping_suspected": any_clipped,
        "accel_clip_threshold_ms2": _ACCEL_CLIP_THRESHOLD_MS2,
        "mag": mag_result,
        "baro": baro_result,
        # C2 / C8 verdict fields
        "c8_rate_ok": rate_hz >= 90.0,    # spec floor: 90 Hz usable for EKF
        "c2_mag_usable": mag_result.get("usable_as_yaw_anchor", False),
    }


def _print_profile(prof: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  IMU PROFILE (C2/C8)")
    print(f"{'='*60}")
    if "error" in prof:
        print(f"  ERROR: {prof['error']}")
        return

    print(f"\n  Rate           : {prof['rate_hz']:.1f} Hz  "
          f"({'OK (>=90)' if prof['c8_rate_ok'] else 'LOW (<90 Hz)'})")
    print(f"  n_samples      : {prof['n_samples']}")
    print(f"\n  Accelerometer (body FRD, m/s^2):")
    mn = prof["accel_mean_frd_ms2"]
    sd = prof["accel_std_frd_ms2"]
    mx = prof["accel_max_abs_frd_ms2"]
    for i, ax in enumerate(["x(fwd)", "y(right)", "z(down)"]):
        print(f"    {ax:<10}  mean={mn[i]:>+8.4f}  std={sd[i]:>8.4f}  max|.|={mx[i]:>8.3f}")
    print(f"  |acc| mean     : {prof['accel_norm_mean_ms2']:.4f} m/s^2  "
          f"(vs 1g={_G:.4f}; bias={prof['accel_norm_vs_g_pct']:+.2f}%)")
    print(f"  |acc| std      : {prof['accel_norm_std_ms2']:.4f} m/s^2")
    clip = prof["accel_clipping_suspected"]
    print(f"  Clipping (>={prof['accel_clip_threshold_ms2']} m/s^2): "
          f"{'SUSPECTED' if clip else 'none detected'}")

    mag = prof["mag"]
    print(f"\n  Magnetometer:")
    if not mag.get("present"):
        print(f"    ABSENT — vision-only heading; gate-in-view + GRU memory critical (C2)")
    else:
        print(f"    all_zero        : {mag.get('all_zero')}")
        print(f"    low_variance    : {mag.get('low_variance')}")
        norm = mag.get("norm_mean", 0)
        print(f"    |mag| mean      : {norm:.2f}")
        print(f"    usable as yaw   : {'YES' if mag.get('usable_as_yaw_anchor') else 'NO'}")
        print(f"    note            : {mag.get('note')}")

    baro = prof["baro"]
    print(f"\n  Barometer       : {'present' if baro['present'] else 'ABSENT'}")
    if baro["present"]:
        print(f"    mean={baro['mean_hpa']:.2f} hPa  std={baro['std_hpa']:.4f} hPa")

    print(f"\n--- C8 verdict: rate_ok={prof['c8_rate_ok']}  clip_suspected={clip}")
    print(f"--- C2 verdict: mag_usable_as_yaw_anchor={prof['c2_mag_usable']}")
    if not prof["c2_mag_usable"]:
        print(f"    => No reliable mag yaw anchor.  "
              f"Vision heading + GRU memory become load-bearing on loss of gate FoV.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="How long to pump the live sim (s); longer = better noise estimate")
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--tlog", default=None, help="Replay a recorded tlog instead of live sim")
    ap.add_argument("--out", default=None, help="Write profile JSON here")
    args = ap.parse_args()

    import os
    os.environ.setdefault("MAVLINK20", "1")

    client = MavlinkClient(args.endpoint)
    client.send_heartbeats = False
    client.send_timesync = True

    tracker = MessageRateTracker()

    if args.tlog:
        tlog_path = Path(args.tlog)
        print(f"  Replaying tlog: {tlog_path}", flush=True)
        samples = _drain_tlog_samples(tlog_path, client)
        tracker_hz = None
    else:
        def _tap(msg):
            t = msg.get_type()
            if t != "BAD_DATA":
                tracker.record(t, time.monotonic_ns())

        client.on_message = _tap
        try:
            print(f"  Connecting to {args.endpoint} (timeout {args.connect_timeout:.0f}s)...", flush=True)
            client.connect(wait_heartbeat=True, timeout_s=args.connect_timeout)
            print(f"  Connected.  Pumping {args.seconds:.0f}s (pre-arm noise floor best)...", flush=True)
        except (TimeoutError, OSError, Exception) as exc:
            print(f"\n  ERROR: stream absent / not connected: {exc}", file=sys.stderr)
            print(f"  -> Start the sim first, or use --tlog to replay a recording.", file=sys.stderr)
            return 2
        samples = _collect_imu_samples(client, args.seconds)
        tracker_hz = tracker.rate_hz("HIGHRES_IMU")

    if not samples:
        print("  ERROR: no HIGHRES_IMU samples received.  Stream absent?", file=sys.stderr)
        return 2

    prof = _analyze(samples, tracker_hz=tracker_hz)
    _print_profile(prof)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(prof, indent=2, default=str), encoding="utf-8")
        print(f"\n  Profile written to {out_path}")

    return 0 if prof.get("c8_rate_ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
