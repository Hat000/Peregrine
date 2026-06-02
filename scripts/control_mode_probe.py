"""First-contact probe: which control interface does the sim actually honour?

Targets the SPEC MAVLink interface (VADR-TS-002 sec 4): SET_POSITION_TARGET_LOCAL_NED
(velocity / position) and SET_ATTITUDE_TARGET (attitude / body-rate). It does NOT touch
Elodin (an RC-callback surrogate, NOT the scoring sim). Resolves R2 + the position
"easy-mode" question: arms, then streams small bounded setpoints in each mode and watches
the observable response -- baro-derived altitude (climb), measured attitude, gyro, and
pos/vel if present -- to see which interface moves the drone in the commanded direction.

  velocity   : command +climb_rate up; expect altitude to rise (exercises the
               SET_POSITION_TARGET 'easy-mode' message path)
  position   : command an absolute NED setpoint target_alt above the arming origin;
               expect a climb that then holds (true position 'easy mode')
  attitude   : command a small pitch; expect MEASURED pitch to track it (thrust uncalibrated
               -> altitude may drift; that's what innerloop_step is for)
  body_rate  : command a small pitch RATE; expect the gyro to track it (CTBR path)

SAFETY: this ACTUATES the drone with small, short setpoints and force-disarms on exit. Start
with --no-actuate (arm + telemetry only). Magnitudes/durations are deliberately tiny. If
NOTHING responds, the sim may need a mode switch (OFFBOARD/GUIDED) or a pre-arm setpoint
stream -- a follow-up probe (note it and tell me).

Usage:
  python scripts/control_mode_probe.py [--endpoint udp:127.0.0.1:14550] [--no-actuate]
        [--modes velocity,position,attitude,body_rate] [--hold-s 2] [--settle-s 1]
        [--climb-rate 0.5] [--target-alt 1.0] [--tilt-deg 5] [--rate-dps 15] [--thrust 0.5]
        [--force] [--connect-timeout 15]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, ControlMode
from racer.firstcontact import (
    backend_summary,
    baro_altitude_m,
    drain_statustexts,
    stream_setpoint as _stream,
    telemetry_summary,
)
from racer.mavlink_client import MavlinkClient

_ALL_MODES = ["velocity", "position", "attitude", "body_rate"]


def _given_alt_m(client: MavlinkClient) -> float | None:
    """Altitude (m, up positive) from the GIVEN LOCAL_POSITION_NED z. Baro reads NaN in this sim,
    so the climb-based verdicts use the provided position instead (R1: position is given)."""
    p = client.state.position_ned
    return None if p is None else -float(p[2])


def _euler_to_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body attitude quaternion (w,x,y,z), intrinsic 3-2-1 yaw-pitch-roll (contracts convention)."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _settle(client: MavlinkClient, seconds: float) -> None:
    """Command zero NED velocity (a hover/hold request) while pumping, to damp between tests."""
    _stream(client, ControlCommand(mode=ControlMode.VELOCITY, velocity_ned=np.zeros(3)), seconds, lambda: None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--modes", default=",".join(_ALL_MODES), help="comma list from " + ",".join(_ALL_MODES))
    ap.add_argument("--no-actuate", action="store_true", help="arm + telemetry only; send no setpoints")
    ap.add_argument("--hold-s", type=float, default=2.0, help="per-mode actuation duration")
    ap.add_argument("--settle-s", type=float, default=1.0, help="hover/hold time between modes")
    ap.add_argument("--climb-rate", type=float, default=0.5, help="velocity test: up m/s")
    ap.add_argument("--target-alt", type=float, default=1.0, help="position test: m above origin")
    ap.add_argument("--tilt-deg", type=float, default=5.0, help="attitude test: pitch command")
    ap.add_argument("--rate-dps", type=float, default=15.0, help="body_rate test: pitch rate deg/s")
    ap.add_argument("--thrust", type=float, default=0.5, help="thrust for attitude/body_rate [0,1]")
    ap.add_argument("--force", action="store_true", help="force-arm (bypass prearm checks)")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in _ALL_MODES]
    if bad:
        print(f"unknown modes: {bad}; valid: {_ALL_MODES}", file=sys.stderr)
        return 2

    client = MavlinkClient(args.endpoint)
    print(f"connecting MAVLink {args.endpoint} (waiting for heartbeat) ...")
    try:
        client.connect(timeout_s=args.connect_timeout)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("heartbeat OK.")

    # brief audit
    seen = len(client.statustexts)
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        client.pump()
        seen = drain_statustexts(client, seen)
        time.sleep(0.02)
    print(f"  backend:   {backend_summary(client)}")
    print(f"  telemetry: {telemetry_summary(client)}")
    if _given_alt_m(client) is None:
        print("  WARNING: no given position -> the climb-based velocity/position verdicts will be blind.")

    results: list[dict] = []
    try:
        print(f"\n[arm] sending {'FORCE ' if args.force else ''}arm ...")
        client.last_command_ack = None
        client.arm(force=args.force)
        ack = client.wait_command_ack(_arm_cmd(), timeout_s=3.0)
        if ack is not None:
            print(f"  COMMAND_ACK: {ack['result_name']}")
        armed = client.wait_armed(True, timeout_s=5.0)
        print(f"  armed: {armed}")

        if args.no_actuate:
            print("\n--no-actuate: skipping setpoint tests.")
        elif not armed:
            print("\narming refused -> cannot test control modes (setpoints need an armed vehicle).")
            print("  try --force, or run session_lifecycle.py to dig into the arm refusal.")
        else:
            for mode in modes:
                results.append(_run_mode(client, mode, args))
                _settle(client, args.settle_s)
    finally:
        print("\n[safety] disarming ...")
        client.disarm(force=True)
        client.wait_armed(False, timeout_s=3.0)

    # -- report ---------------------------------------------------------------
    print("\n==== control_mode_probe summary ====")
    print(f"  backend: {backend_summary(client)}")
    if results:
        for r in results:
            print(f"  {r['mode']:10s} cmd={r['commanded']:24s} obs={r['observed']:28s} -> {r['verdict']}")
        live = [r["mode"] for r in results if r["responded"]]
        print(f"\n  LIVE interfaces: {live or 'NONE responded'}")
        if not live:
            print("  Nothing responded -> the sim likely needs a mode switch (OFFBOARD/GUIDED) or a")
            print("  pre-arm setpoint stream before it accepts commands. Flag for a follow-up probe.")
        elif "position" in live or "velocity" in live:
            print("  Position/velocity 'easy mode' appears HONOURED -> VQ1 can lean on the stabilizer.")
    if client.track_gates:
        print(f"  R4: gate map ARRIVED while active-piloting -> {len(client.track_gates)} gates (TRACK_INFO).")
    else:
        print("  R4: still NO gate map (arm + control stream alone did not trigger TRACK_INFO this run).")
    print("  NOTE: Elodin is NOT the source of truth; this characterised the spec MAVLink sim.")
    return 0


def _arm_cmd() -> int:
    from pymavlink import mavutil

    return mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM


def _run_mode(client: MavlinkClient, mode: str, args) -> dict:
    print(f"\n[{mode}] actuating {args.hold_s:g}s ...")
    base_alt = _given_alt_m(client)
    base_pitch = client.state.pitch

    if mode == "velocity":
        cmd = ControlCommand(mode=ControlMode.VELOCITY, velocity_ned=np.array([0.0, 0.0, -args.climb_rate]))
        alts = _stream(client, cmd, args.hold_s, lambda: _given_alt_m(client))
        d = (max(alts) - base_alt) if (alts and base_alt is not None) else float("nan")
        responded = bool(d == d and d > 0.2)  # d==d filters NaN
        return _result(mode, f"vz={-args.climb_rate:+.2f} m/s", f"climb {d:+.2f} m", responded)

    if mode == "position":
        cmd = ControlCommand(mode=ControlMode.POSITION, position_ned=np.array([0.0, 0.0, -args.target_alt]))
        alts = _stream(client, cmd, args.hold_s, lambda: _given_alt_m(client))
        d = (max(alts) - base_alt) if (alts and base_alt is not None) else float("nan")
        responded = bool(d == d and d > 0.2)
        return _result(mode, f"z={-args.target_alt:+.2f} m (NED)", f"climb {d:+.2f} m", responded)

    if mode == "attitude":
        tilt = math.radians(args.tilt_deg)
        q = _euler_to_wxyz(0.0, tilt, client.state.yaw)
        cmd = ControlCommand(mode=ControlMode.ATTITUDE, attitude_quat_wxyz=q, thrust=args.thrust)
        pitches = _stream(client, cmd, args.hold_s, lambda: client.state.pitch)
        peak = max((abs(p) for p in pitches), default=float("nan"))
        responded = bool(peak == peak and peak > 0.5 * abs(tilt))
        return _result(mode, f"pitch={args.tilt_deg:+.1f} deg", f"peak |pitch|={math.degrees(peak):+.1f} deg", responded)

    if mode == "body_rate":
        rate = math.radians(args.rate_dps)
        cmd = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.array([0.0, rate, 0.0]), thrust=args.thrust)
        gyros = _stream(client, cmd, args.hold_s, lambda: client.state.angular_rate_body[1])
        peak = max((abs(g) for g in gyros), default=float("nan"))
        responded = bool(peak == peak and peak > 0.5 * abs(rate))
        return _result(mode, f"q={args.rate_dps:+.1f} deg/s", f"peak |gyro_y|={math.degrees(peak):+.1f} deg/s", responded)

    return _result(mode, "?", "?", False)


def _result(mode: str, commanded: str, observed: str, responded: bool) -> dict:
    verdict = "RESPONDED" if responded else "no clear response"
    print(f"  cmd {commanded} -> obs {observed}: {verdict}")
    return {"mode": mode, "commanded": commanded, "observed": observed, "responded": responded, "verdict": verdict}


if __name__ == "__main__":
    raise SystemExit(main())
