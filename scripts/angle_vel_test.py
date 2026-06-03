"""Easy-mode via VELOCITY in ANGLE -- the reference client's actual control path.

Discovery chain (2026-06-03, with the teammate watching the GUI):
  * no client sending anything -> sim stays in ANGLE, drone holds at origin;
  * a HEARTBEAT or TIMESYNC keepalive -> flips to ACRO;
  * a position/velocity SETPOINT does NOT flip the mode (stays ANGLE) -- but a POSITION target
    runs away (this sim's SET_POSITION_TARGET *position* interface is broken/unsupported);
  * the reference client only ever sends VELOCITY (SET_POSITION_TARGET velocity mask).
So: tap in passively (telemetry streams for free, no keepalive needed -> stays ANGLE) and send
ONLY velocity setpoints. If the sim's ANGLE velocity controller tracks, easy-mode is real and we
fly the course on velocity setpoints (no CTBR needed).

Sends NO heartbeat / NO timesync (those force ACRO); receives telemetry passively. The race must
already be RUNNING (teammate started it); this taps the live stream -- it does NOT wait for a GO.

Phases (teammate: watch the GUI mode stays ANGLE; I measure achieved vs commanded velocity):
  HOVER  velocity=(0,0,0)         -> holds in place, no climb?
  FWD    velocity= v toward gate0 -> moves at v, no climb?
  UP     velocity=(0,0,-0.5)      -> climbs at 0.5 m/s, controlled?

Bounded: abort + force-disarm on offset/altitude/tilt/collision.

Usage:  python scripts/angle_vel_test.py --vel 1.0 --label avel1   # race must be live first
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.contracts import ControlCommand, ControlMode
from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--vel", type=float, default=1.0, help="forward speed toward gate 0 (m/s)")
    ap.add_argument("--climb", type=float, default=0.5, help="UP phase climb rate (m/s)")
    ap.add_argument("--phase-s", type=float, default=3.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--settle-s", type=float, default=1.0, help="passive observe before commanding")
    ap.add_argument("--wait-live-s", type=float, default=40.0, help="wait this long for live+armed telemetry")
    ap.add_argument("--max-offset-m", type=float, default=15.0)
    ap.add_argument("--max-alt-m", type=float, default=10.0)
    ap.add_argument("--max-tilt-deg", type=float, default=80.0)
    ap.add_argument("--label", default="avel")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    args = ap.parse_args()

    c = MavlinkClient(args.endpoint)
    c.send_heartbeats = False     # NO keepalive -> stay in ANGLE
    c.send_timesync = False
    c.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)
    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(probe="angle_vel_test", label=args.label, mode="velocity_in_angle")
    c.on_message = lambda m: (m.get_type() != "BAD_DATA" and m.get_msgbuf()
                              and recorder.record_mavlink(bytes(m.get_msgbuf())))
    print(f"recording -> {session}")
    print(">>> TAP-IN (no heartbeat/timesync). Race must be LIVE. TEAMMATE: confirm GUI = ANGLE.\n")

    # wait for a LIVE + ARMED drone (the teammate restarts the race; telemetry streams passively)
    deadline = time.monotonic() + args.wait_live_s
    last_w = 0.0
    while time.monotonic() < deadline:
        c.pump()
        s = c.state
        if s.position_ned is not None and s.sim_time_ns > 0 and s.armed:
            break
        now = time.monotonic()
        if now - last_w >= 1.0:
            print(f"  waiting for live+armed race: pos={'y' if s.position_ned is not None else 'n'} "
                  f"armed={s.armed} sim_t={s.sim_time_ns/1e9:.1f}   ", end="\r", flush=True)
            last_w = now
        time.sleep(0.01)
    s = c.state
    if s.position_ned is None or s.sim_time_ns == 0 or not s.armed:
        print("\nno live+armed telemetry -> restart the race (Home->Race). aborting.", file=sys.stderr)
        recorder.close()
        return 1
    t0 = time.monotonic()                        # brief passive settle
    while time.monotonic() - t0 < args.settle_s:
        c.pump()
        time.sleep(0.01)
    origin = np.asarray(s.position_ned, dtype=np.float64).copy()
    yaw0 = float(s.yaw)
    fwd = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])     # body-forward in world (toward gate 0)
    print(f"  live: armed={s.armed} pos={np.round(origin,2)} rpy_deg="
          f"({np.degrees(s.roll):+.0f},{np.degrees(s.pitch):+.0f},{np.degrees(s.yaw):+.0f}) forward={np.round(fwd,2)}")

    phases = [
        ("HOVER", np.zeros(3)),
        ("FWD", args.vel * fwd),
        ("UP", np.array([0.0, 0.0, -args.climb])),
    ]
    aborted = None
    try:
        for name, vel_cmd in phases:
            print(f"\n>>> PHASE {name} vel_cmd={np.round(vel_cmd,2)} ({args.phase_s:g}s) -- GUI mode? (want ANGLE)")
            end = time.monotonic() + args.phase_s
            last = 0.0
            while time.monotonic() < end:
                c.pump()                       # recv only (no keepalive)
                c.send_command(ControlCommand(mode=ControlMode.VELOCITY,
                                              velocity_ned=vel_cmd, yaw=yaw0))
                s = c.state
                pos = np.asarray(s.position_ned, dtype=np.float64)
                rel = pos - origin
                if any(cc["threat_level"] >= 2 for cc in c.collisions):
                    aborted = "collision"
                elif float(np.hypot(rel[0], rel[1])) > args.max_offset_m:
                    aborted = f"offset {np.hypot(rel[0],rel[1]):.0f}m"
                elif abs(float(rel[2])) > args.max_alt_m:
                    aborted = f"alt {rel[2]:+.0f}m"
                elif max(abs(s.roll), abs(s.pitch)) > np.radians(args.max_tilt_deg):
                    aborted = f"tilt {np.degrees(max(abs(s.roll),abs(s.pitch))):.0f}deg"
                if aborted:
                    break
                now = time.monotonic()
                if now - last >= 0.3:
                    v = np.asarray(s.velocity_ned, dtype=np.float64)
                    print(f"   {name:5s} cmd={np.round(vel_cmd,2)} ACHIEVED_vel={np.round(v,2)} "
                          f"rel_pos={np.round(rel,2)} rpy=({np.degrees(s.roll):+.0f},"
                          f"{np.degrees(s.pitch):+.0f},{np.degrees(s.yaw):+.0f})   ", end="\r", flush=True)
                    last = now
                time.sleep(1.0 / args.rate)
            s = c.state
            v = np.asarray(s.velocity_ned, dtype=np.float64)
            rel = np.asarray(s.position_ned, dtype=np.float64) - origin
            print(f"\n   end {name}: cmd={np.round(vel_cmd,2)} achieved_vel={np.round(v,2)} rel_pos={np.round(rel,2)}")
            if aborted:
                print(f"   *** ABORT during {name}: {aborted}")
                break
    except KeyboardInterrupt:
        aborted = "ctrl-c"
    finally:
        if aborted:
            print("\n[safety] force-disarming ...")
            try:
                c.send_heartbeats = True
                c.disarm(force=True)
                c.wait_armed(False, timeout_s=3.0)
            except Exception as exc:
                print(f"  disarm error: {exc}", file=sys.stderr)
        else:
            print("\n[done] no runaway -> leaving armed (sent only velocity setpoints).")
        recorder.add_meta(aborted=aborted)
        recorder.close()
    print(f"\n==== angle_vel_test: aborted={aborted}  recording={session} ====")
    print("  KEY: did HOVER hold (no climb) and FWD move at the commanded speed, all in ANGLE? -> easy-mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
