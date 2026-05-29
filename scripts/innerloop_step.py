"""First-contact probe: system-ID the inner-loop stabilizer + airframe (risk R2 keystone).

Targets the SPEC MAVLink interface (SET_ATTITUDE_TARGET); it does NOT touch Elodin. Arms,
then runs small bounded STEP experiments and fits the numbers the controller needs:

  thrust sweep  -> hover_thrust + d(accel)/d(thrust). Commands LEVEL attitude at several
                   thrust levels and measures the steady net-UP acceleration (from HIGHRES_IMU
                   rotated by the GIVEN attitude), then linear-fits the thrust at zero net
                   accel. This calibrates Controller.hover_thrust + the placeholder throttle
                   scale (the keystone the attitude path waits on).
  attitude step -> delay / rise / tau / overshoot. Steps pitch level->tilt and fits the
                   measured-pitch response, bounding how fast the planner may demand attitude.

SAFETY: this ACTUATES with bounded thrust/attitude steps and force-disarms on exit. It WILL
climb/descend during the thrust sweep -- run it with vertical clearance. Magnitudes are
conservative; tune via flags. Start with --no-actuate (arm + plan only). Best run alongside
record_session.py to also capture the raw stream.

Usage:
  python scripts/innerloop_step.py [--endpoint udp:127.0.0.1:14550]
        [--thrust-levels 0.35,0.45,0.55,0.65] [--dwell-s 1.5]
        [--tilt-deg 10] [--baseline-s 1.0] [--step-hold-s 2.0]
        [--force] [--no-actuate] [--out PATH] [--connect-timeout 15]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer import frames
from racer.contracts import ControlCommand, ControlMode
from racer.firstcontact import backend_summary, stream_setpoint, telemetry_summary
from racer.mavlink_client import MavlinkClient
from racer.recording import session_stamp
from racer.sysid import fit_hover_thrust, step_response_metrics

_G = 9.80665


def _euler_to_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body attitude quaternion (w,x,y,z), intrinsic 3-2-1 (frames/contracts convention)."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _up_accel(client: MavlinkClient) -> float:
    """Net UPWARD kinematic acceleration (m/s^2) from the IMU specific force + given attitude.
    0 at hover; >0 climbing. a_world = R_wb @ f_body + g; up = -(a_world_down)."""
    s = client.state
    R = frames.R_world_from_body(s.roll, s.pitch, s.yaw)
    return float(-((R @ s.accel_body)[2] + _G))


def _record_segments(client: MavlinkClient, segments, sample, rate_hz: float = 30.0):
    """Send a sequence of (command, seconds) segments, recording (t, sample()) throughout.
    Returns (t, y, segment_start_times) with t relative to the first sample."""
    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    ts: list[float] = []
    ys: list[float] = []
    starts: list[float] = []
    for cmd, secs in segments:
        starts.append(time.monotonic() - t0)
        end = time.monotonic() + secs
        while time.monotonic() < end:
            client.send_command(cmd)
            client.pump()
            ts.append(time.monotonic() - t0)
            ys.append(float(sample()))
            time.sleep(dt)
    return np.array(ts), np.array(ys), starts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--thrust-levels", default="0.35,0.45,0.55,0.65")
    ap.add_argument("--dwell-s", type=float, default=1.5, help="per thrust level")
    ap.add_argument("--tilt-deg", type=float, default=10.0, help="attitude step size")
    ap.add_argument("--baseline-s", type=float, default=1.0)
    ap.add_argument("--step-hold-s", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-actuate", action="store_true", help="arm + print the plan only")
    ap.add_argument("--out", default=None, help="JSON output path (default data/runs/innerloop_<stamp>.json)")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    levels = [float(x) for x in args.thrust_levels.split(",") if x.strip()]

    client = MavlinkClient(args.endpoint)
    print(f"connecting MAVLink {args.endpoint} (waiting for heartbeat) ...")
    try:
        client.connect(timeout_s=args.connect_timeout)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("heartbeat OK.")
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        client.pump()
        time.sleep(0.02)
    print(f"  backend:   {backend_summary(client)}")
    print(f"  telemetry: {telemetry_summary(client)}")
    if not np.any(client.state.accel_body):
        print("  WARNING: accel_body is all-zero -> no HIGHRES_IMU? thrust-sweep sysid needs it.")

    result: dict = {"endpoint": args.endpoint, "thrust_levels": levels}
    try:
        if args.no_actuate:
            print("\n--no-actuate: arming to confirm, then disarming (no steps).")
            client.arm(force=args.force)
            print(f"  armed: {client.wait_armed(True, timeout_s=5.0)}")
            return 0

        print(f"\n[arm] {'force ' if args.force else ''}arming ...")
        client.arm(force=args.force)
        if not client.wait_armed(True, timeout_s=5.0):
            print("  arming refused -> cannot system-ID. Try --force / run session_lifecycle.py.", file=sys.stderr)
            return 1
        yaw = client.state.yaw
        level_q = _euler_to_wxyz(0.0, 0.0, yaw)

        # -- experiment 1: thrust sweep -> hover_thrust + slope ----------------
        print("\n[thrust sweep] measuring net-up accel at each thrust level ...")
        sweep = []
        for thr in levels:
            cmd = ControlCommand(mode=ControlMode.ATTITUDE, attitude_quat_wxyz=level_q, thrust=thr)
            samples = stream_setpoint(client, cmd, args.dwell_s, metric=lambda: _up_accel(client))
            steady = float(np.mean(samples[len(samples) // 2:])) if samples else float("nan")
            sweep.append({"thrust": thr, "up_accel": steady})
            print(f"  thrust={thr:.2f} -> steady up-accel = {steady:+.2f} m/s^2")
        hover, slope = fit_hover_thrust([p["thrust"] for p in sweep], [p["up_accel"] for p in sweep])
        max_up = slope * (1.0 - hover) if slope == slope and hover == hover else float("nan")
        twr = (max_up + _G) / _G if max_up == max_up else float("nan")
        result["thrust_sweep"] = {"samples": sweep, "hover_thrust": hover, "slope_mps2_per_thrust": slope,
                                  "max_up_accel": max_up, "twr_estimate": twr}
        print(f"  => hover_thrust={hover:.3f}  slope={slope:.1f} (m/s^2)/thrust  "
              f"max_up_accel~{max_up:.1f}  TWR~{twr:.2f}")

        # -- experiment 2: attitude step -> lag --------------------------------
        thr_h = float(np.clip(hover if hover == hover else 0.5, 0.1, 0.9))
        tilt = np.deg2rad(args.tilt_deg)
        print(f"\n[attitude step] pitch 0 -> {args.tilt_deg:g} deg at thrust {thr_h:.2f} ...")
        segments = [
            (ControlCommand(mode=ControlMode.ATTITUDE, attitude_quat_wxyz=level_q, thrust=thr_h), args.baseline_s),
            (ControlCommand(mode=ControlMode.ATTITUDE, attitude_quat_wxyz=_euler_to_wxyz(0.0, tilt, yaw),
                            thrust=thr_h), args.step_hold_s),
        ]
        t, y, starts = _record_segments(client, segments, lambda: client.state.pitch)
        metrics = step_response_metrics(t, y, t_step=starts[1])
        result["attitude_step"] = {"tilt_deg": args.tilt_deg, "metrics": metrics,
                                   "t": t.tolist(), "pitch": y.tolist()}
        print(f"  delay={metrics['delay_s']:.3f}s  rise(10-90%)={metrics['rise_time_s']:.3f}s  "
              f"tau~{metrics['tau_s']:.3f}s  overshoot={metrics['overshoot']:.0%}  "
              f"steady={np.rad2deg(metrics['steady_state']):+.1f} deg")
    finally:
        print("\n[safety] returning to level + disarming ...")
        client.disarm(force=True)
        client.wait_armed(False, timeout_s=3.0)

    # -- persist + recommend --------------------------------------------------
    out = Path(args.out) if args.out else Path("data/runs") / f"innerloop_{session_stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")

    sw = result.get("thrust_sweep", {})
    at = result.get("attitude_step", {}).get("metrics", {})
    print("\n==== innerloop_step summary (feeds Controller) ====")
    if sw:
        print(f"  Controller(hover_thrust={sw['hover_thrust']:.3f})   # was the 0.5 placeholder")
        print(f"  throttle map: up_accel ~ {sw['slope_mps2_per_thrust']:.1f}*(thrust - hover) "
              f"-> thrust = hover + a_des_up/{sw['slope_mps2_per_thrust']:.1f} (more exact than |f|/g)")
        print(f"  TWR ~ {sw['twr_estimate']:.2f}; plan the line within this accel envelope.")
    if at:
        print(f"  attitude loop: delay~{at['delay_s']:.3f}s, tau~{at['tau_s']:.3f}s -> "
              "cap the planner's attitude-rate demand accordingly (latency comp).")
    print("  NOTE: Elodin is NOT the source of truth; this system-ID'd the spec MAVLink sim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
