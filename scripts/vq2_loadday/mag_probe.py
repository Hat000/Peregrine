"""scripts/vq2_loadday/mag_probe.py — VQ2 load-day magnetometer usability probe (C2).

Probes whether the magnetometer inside HIGHRES_IMU is usable as a yaw anchor by
correlating mag-derived heading with a reference yaw (Training ODOMETRY quaternion
when present).  If the mag is absent, all-zero, or low-variance the probe flags it and
upgrades the "gate-in-view + GRU memory" path to load-bearing.

C2 gate for the AHRS design:
  mag usable  → AHRS can fuse mag for yaw → relaxed gate-in-view requirement
  mag absent  → vision-only heading → gate-in-view + GRU become critical

Key outputs (all in the JSON / printed summary):
  - mag_present / all_zero / low_variance
  - mag_heading_std_deg: std of the raw mag heading (deg) — useful signal?
  - corr_vs_odo_r: Pearson r between mag heading and ODOMETRY yaw (NaN if no ODOMETRY)
  - c2_usable: bool verdict (present + enough variance + |r| > 0.7 if reference available)

Usage (live sim, 10 s pre-arm for best noise read):
  python scripts/vq2_loadday/mag_probe.py --seconds 10

Usage (tlog replay):
  python scripts/vq2_loadday/mag_probe.py --tlog data/runs/<session>/mavlink.tlog

Usage (save JSON):
  python scripts/vq2_loadday/mag_probe.py --seconds 10 --out data/mag_probe.json

Degrades gracefully: "stream absent / not connected" when the sim is unreachable.
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

# Magnetometer variance threshold: below this the sensor is probably not present or
# is all-zeros (VQ2 may not simulate a real magnetometer).
_MAG_STD_THRESHOLD = 1e-3   # nT / arbitrary units; very tight — catches completely constant fields

# Correlation threshold for declaring the mag "correlated with ODOMETRY yaw".
# We're comparing raw mag heading vs true yaw — a well-calibrated mag should give |r| > 0.7
# for any significant yaw excursion (pre-arm static = no excursion → correlation is not
# meaningful for small angular range, so we skip the correlation gate if range < 20 deg).
_CORR_THRESH = 0.70
_MIN_YAW_RANGE_DEG = 20.0   # below this yaw range the correlation test is inconclusive

# Minimum magnetic-field norm (nT) to distinguish real sensor from all-zero stub.
_MAG_NORM_FLOOR = 1.0   # nT; an all-zero field reads 0.000


def _mag_heading_deg(mx: float, my: float) -> float:
    """Magnetic heading from FRD x/y components.  NED convention: atan2(east, north).
    In the FRD body frame: x=fwd (north proxy), y=right (east proxy) at level flight."""
    return float(np.degrees(np.arctan2(my, mx)) % 360.0)


def _collect_samples(client: MavlinkClient, seconds: float) -> list[dict]:
    """Pump and collect per-IMU-tick samples with mag + ODOMETRY yaw."""
    samples: list[dict] = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        client.pump()
        s = client.state
        if s.sim_time_ns > 0:
            mag = s.mag_body
            yaw_rad = s.yaw  # from ODOMETRY quaternion; 0.0 if not yet received
            odo_valid = s.orientation_ned_wxyz is not None
            samples.append({
                "sim_time_ns": s.sim_time_ns,
                "recv_ns": s.recv_monotonic_ns,
                "mag": mag.tolist() if mag is not None else None,
                "yaw_rad": float(yaw_rad),
                "odo_valid": odo_valid,
            })
        time.sleep(0.005)
    return samples


def _drain_tlog_samples(tlog_path: Path, client: MavlinkClient) -> list[dict]:
    """Replay a tlog and collect the same per-tick samples."""
    import os
    os.environ.setdefault("MAVLINK20", "1")
    from pymavlink import mavutil

    conn = mavutil.mavlink_connection(str(tlog_path), input=True)
    samples: list[dict] = []
    while True:
        msg = conn.recv_match(blocking=False)
        if msg is None:
            break
        if msg.get_type() == "BAD_DATA":
            continue
        client._handle(msg)
        s = client.state
        if msg.get_type() == "HIGHRES_IMU" and s.sim_time_ns > 0:
            mag = s.mag_body
            samples.append({
                "sim_time_ns": s.sim_time_ns,
                "recv_ns": s.recv_monotonic_ns,
                "mag": mag.tolist() if mag is not None else None,
                "yaw_rad": float(s.yaw),
                "odo_valid": s.orientation_ned_wxyz is not None,
            })
    return samples


def analyze(samples: list[dict]) -> dict:
    """Compute the mag-probe metrics from a list of per-tick dicts.

    This is the pure analysis function — no I/O, unit-testable.
    """
    if not samples:
        return {"error": "no samples collected"}

    n = len(samples)

    # ------------------------------------------------------------------ mag arrays
    mag_raw = [s["mag"] for s in samples if s["mag"] is not None]
    if not mag_raw:
        return {
            "n_samples": n,
            "mag_present": False,
            "all_zero": False,
            "low_variance": False,
            "mag_heading_std_deg": None,
            "mag_heading_range_deg": None,
            "corr_vs_odo_r": None,
            "corr_valid": False,
            "c2_usable": False,
            "note": "mag_body always None (absent or not emitted by HIGHRES_IMU)",
        }

    mag = np.array(mag_raw, dtype=np.float64)   # (M, 3)
    mag_mean = mag.mean(axis=0)
    mag_std = mag.std(axis=0)
    mag_norm = np.linalg.norm(mag, axis=1)

    all_zero = bool(np.allclose(mag, 0.0, atol=1e-9))
    low_variance = bool(np.all(mag_std < _MAG_STD_THRESHOLD))
    norm_ok = bool(mag_norm.mean() > _MAG_NORM_FLOOR)

    # Raw mag heading (FRD x/y → heading deg 0-360)
    headings_deg = np.array([_mag_heading_deg(float(m[0]), float(m[1])) for m in mag_raw])
    # Unwrap to remove 0/360 discontinuities before computing std/range
    headings_unwrap = np.unwrap(np.radians(headings_deg))
    heading_std_deg = float(np.degrees(np.std(headings_unwrap)))
    heading_range_deg = float(np.degrees(np.ptp(headings_unwrap)))

    # ------------------------------------------------------------------ correlation vs ODOMETRY yaw
    # Only use samples where ODOMETRY is valid AND we have a mag reading
    paired_idx = [
        i for i, s in enumerate(samples)
        if s["mag"] is not None and s["odo_valid"]
    ]
    corr_r: float | None = None
    corr_valid = False
    if len(paired_idx) >= 10:
        yaws_rad = np.array([samples[i]["yaw_rad"] for i in paired_idx])
        mag_paired = np.array([samples[i]["mag"] for i in paired_idx], dtype=np.float64)
        heads_p = np.array([_mag_heading_deg(float(m[0]), float(m[1])) for m in mag_paired])
        # Unwrap both for correlation
        heads_p_uw = np.degrees(np.unwrap(np.radians(heads_p)))
        yaws_deg = np.degrees(np.unwrap(yaws_rad))
        yaw_range = float(np.ptp(yaws_deg))
        if yaw_range >= _MIN_YAW_RANGE_DEG and not all_zero and not low_variance:
            r_mat = np.corrcoef(heads_p_uw, yaws_deg)
            corr_r = float(r_mat[0, 1])
            corr_valid = True

    # ------------------------------------------------------------------ C2 verdict
    # Usable = present + not all-zero + enough variance + norm OK
    # If we have ODOMETRY correlation, require |r| >= threshold too.
    usable = (
        bool(mag_raw)
        and not all_zero
        and not low_variance
        and norm_ok
    )
    if usable and corr_valid and corr_r is not None:
        usable = bool(abs(corr_r) >= _CORR_THRESH)

    note: str
    if all_zero:
        note = "all-zero — magnetometer not simulated (stub field)"
    elif low_variance:
        note = "low-variance — field not varying; can't distinguish headings"
    elif not norm_ok:
        note = f"field norm too low ({mag_norm.mean():.3f}) — may be a stub"
    elif corr_valid and corr_r is not None and abs(corr_r) < _CORR_THRESH:
        note = (
            f"mag heading poorly correlated with ODOMETRY yaw "
            f"(|r|={abs(corr_r):.2f} < {_CORR_THRESH:.2f}) — unreliable as yaw anchor"
        )
    elif corr_valid:
        note = (
            f"mag heading correlated with ODOMETRY yaw "
            f"(|r|={abs(corr_r):.2f}); looks usable"
        )
    else:
        note = (
            "mag has variance but ODOMETRY absent or yaw range too small for correlation test; "
            "looks present but usability unconfirmed"
        )

    return {
        "n_samples": n,
        "n_mag_samples": len(mag_raw),
        "mag_present": True,
        "all_zero": all_zero,
        "low_variance": low_variance,
        "norm_ok": norm_ok,
        "mag_mean_frd": mag_mean.tolist(),
        "mag_std_frd": mag_std.tolist(),
        "mag_norm_mean": float(mag_norm.mean()),
        "mag_norm_std": float(mag_norm.std()),
        "mag_heading_std_deg": round(heading_std_deg, 3),
        "mag_heading_range_deg": round(heading_range_deg, 3),
        "n_paired_with_odo": len(paired_idx),
        "corr_vs_odo_r": round(corr_r, 4) if corr_r is not None else None,
        "corr_valid": corr_valid,
        "corr_yaw_range_deg_required": _MIN_YAW_RANGE_DEG,
        "c2_usable": usable,
        "note": note,
    }


def _print_result(res: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  MAG PROBE (C2) — magnetometer yaw-anchor usability")
    print(f"{'='*60}")
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return

    print(f"\n  n_samples        : {res['n_samples']}  (mag: {res['n_mag_samples']})")
    print(f"  mag_present      : {res['mag_present']}")
    print(f"  all_zero         : {res.get('all_zero')}")
    print(f"  low_variance     : {res.get('low_variance')}")
    print(f"  norm_ok          : {res.get('norm_ok')}  "
          f"(mean |B|={res.get('mag_norm_mean', 0):.3f})")
    print(f"  heading_std_deg  : {res.get('mag_heading_std_deg')}")
    print(f"  heading_range_deg: {res.get('mag_heading_range_deg')}")
    print(f"  n_paired_odo     : {res.get('n_paired_with_odo')}  "
          f"corr_valid={res.get('corr_valid')}")
    if res.get("corr_vs_odo_r") is not None:
        print(f"  corr_vs_odo_r    : {res['corr_vs_odo_r']:.4f}  "
              f"(threshold {_CORR_THRESH:.2f})")
    print(f"\n--- C2 VERDICT: c2_usable={res['c2_usable']}")
    print(f"    note: {res['note']}")
    if not res["c2_usable"]:
        print(f"\n  => No reliable mag yaw anchor.")
        print(f"     Vision heading + GRU memory become load-bearing on loss of gate FoV.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="How long to pump the live sim (s); longer = better correlation estimate")
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--tlog", default=None, help="Replay a recorded tlog instead of live sim")
    ap.add_argument("--out", default=None, help="Write result JSON here")
    args = ap.parse_args()

    import os
    os.environ.setdefault("MAVLINK20", "1")

    client = MavlinkClient(args.endpoint)
    client.send_heartbeats = False
    client.send_timesync = True

    if args.tlog:
        tlog_path = Path(args.tlog)
        print(f"  Replaying tlog: {tlog_path}", flush=True)
        samples = _drain_tlog_samples(tlog_path, client)
        print(f"  Done. {len(samples)} IMU ticks replayed.", flush=True)
    else:
        tracker = MessageRateTracker()

        def _tap(msg):
            t = msg.get_type()
            if t != "BAD_DATA":
                tracker.record(t, time.monotonic_ns())

        client.on_message = _tap
        try:
            print(f"  Connecting to {args.endpoint} (timeout {args.connect_timeout:.0f}s)...", flush=True)
            client.connect(wait_heartbeat=True, timeout_s=args.connect_timeout)
            print(f"  Connected. Pumping {args.seconds:.0f}s...", flush=True)
        except (TimeoutError, OSError, Exception) as exc:
            print(f"\n  ERROR: stream absent / not connected: {exc}", file=sys.stderr)
            print(f"  -> Start the sim first, or use --tlog to replay a recording.", file=sys.stderr)
            return 2
        samples = _collect_samples(client, args.seconds)

    if not samples:
        print("  ERROR: no IMU samples received. Stream absent?", file=sys.stderr)
        return 2

    res = analyze(samples)
    _print_result(res)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print(f"\n  Result written to {out_path}")

    return 0 if res.get("c2_usable", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
