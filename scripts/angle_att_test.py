"""Easy-mode via ATTITUDE + explicit thrust in ANGLE -- the last untested variant.

Findings so far (teammate watching GUI): tapping in with NO keepalive keeps the sim in ANGLE; a
velocity/position setpoint stays ANGLE but CLIMBS away -- because the sim's velocity/position
AUTO-THRUST is broken. BUT the velocity cmd also LEVELLED the attitude (pitch -18->0), so ANGLE's
attitude stabilizer works. SET_ATTITUDE_TARGET gives the THRUST explicitly (0..1) while ANGLE holds
the commanded attitude -- so WE own thrust (the broken bit) and the sim owns attitude (the hard
bit). If our thrust actually controls altitude, this is easy-mode AND simpler than CTBR.

Test 1 (thrust sweep at LEVEL attitude): does altitude rate track our thrust? If vz goes
down/flat/up as thrust rises through hover, thrust is honored -> controllable. If it climbs hard
regardless, attitude mode also ignores our thrust (auto-thrust) -> dead, go CTBR.
Test 2 (only if 1 works): a small forward tilt -> does it fly forward, ANGLE-stabilized?

Sends NO heartbeat/timesync (stay ANGLE); race must be LIVE + armed. Bounded abort + force-disarm.

Usage:  python scripts/angle_att_test.py --thrusts 0.18,0.25,0.32 --label aatt1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, ControlMode
from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp


def _quat(roll, pitch, yaw):
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--thrusts", default="0.18,0.25,0.32", help="level-attitude thrust sweep")
    ap.add_argument("--dwell-s", type=float, default=1.5)
    ap.add_argument("--fwd-pitch-deg", type=float, default=8.0, help="test-2 forward tilt (nose-down)")
    ap.add_argument("--fwd-thrust", type=float, default=0.25)
    ap.add_argument("--fwd-s", type=float, default=3.0)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--max-offset-m", type=float, default=12.0)
    ap.add_argument("--max-alt-m", type=float, default=6.0)
    ap.add_argument("--max-tilt-deg", type=float, default=80.0)
    ap.add_argument("--wait-live-s", type=float, default=90.0)
    ap.add_argument("--fwd", action="store_true", help="also run test-2 (forward tilt) if you ask")
    ap.add_argument("--label", default="aatt")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    args = ap.parse_args()

    c = MavlinkClient(args.endpoint)
    c.send_heartbeats = False
    c.send_timesync = False
    c.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)
    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(probe="angle_att_test", label=args.label, mode="attitude_in_angle")
    c.on_message = lambda m: (m.get_type() != "BAD_DATA" and m.get_msgbuf()
                              and recorder.record_mavlink(bytes(m.get_msgbuf())))
    print(f"recording -> {session}")
    print(">>> TAP-IN (no keepalive -> ANGLE). Race must be LIVE+armed. TEAMMATE: confirm GUI=ANGLE.\n")

    deadline = time.monotonic() + args.wait_live_s
    lw = 0.0
    while time.monotonic() < deadline:
        c.pump()
        s = c.state
        if s.position_ned is not None and s.sim_time_ns > 0 and s.armed:
            break
        now = time.monotonic()
        if now - lw >= 1.0:
            print(f"  waiting for live+armed race: pos={'y' if s.position_ned is not None else 'n'} "
                  f"armed={s.armed}   ", end="\r", flush=True)
            lw = now
        time.sleep(0.01)
    s = c.state
    if s.position_ned is None or not s.armed:
        print("\nno live+armed race -> restart (Home->Race). aborting.", file=sys.stderr)
        recorder.close()
        return 1
    origin = np.asarray(s.position_ned, dtype=np.float64).copy()
    yaw0 = float(s.yaw)
    print(f"  live: pos={np.round(origin,2)} rpy_deg=({np.degrees(s.roll):+.0f},{np.degrees(s.pitch):+.0f},"
          f"{np.degrees(s.yaw):+.0f})")

    phases = [(f"LVL@{t:.2f}", _quat(0.0, 0.0, yaw0), float(t), args.dwell_s)
              for t in (float(x) for x in args.thrusts.split(","))]
    if args.fwd:
        phases.append((f"FWD@{args.fwd_thrust:.2f}",
                       _quat(0.0, -np.radians(args.fwd_pitch_deg), yaw0), args.fwd_thrust, args.fwd_s))

    aborted = None
    try:
        for name, q, thr, dwell in phases:
            print(f"\n>>> PHASE {name} (attitude+thrust={thr:.2f}, {dwell:g}s) -- GUI mode? (want ANGLE)")
            vz0 = float(np.asarray(c.state.velocity_ned, dtype=np.float64)[2])
            end = time.monotonic() + dwell
            last = 0.0
            while time.monotonic() < end:
                c.pump()
                c.send_command(ControlCommand(mode=ControlMode.ATTITUDE, attitude_quat_wxyz=q, thrust=thr))
                s = c.state
                rel = np.asarray(s.position_ned, dtype=np.float64) - origin
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
                    print(f"   {name:9s} thr={thr:.2f} vz={v[2]:+.2f} (up={-v[2]:+.2f}) rel={np.round(rel,2)} "
                          f"rpy=({np.degrees(s.roll):+.0f},{np.degrees(s.pitch):+.0f},{np.degrees(s.yaw):+.0f})   ",
                          end="\r", flush=True)
                    last = now
                time.sleep(1.0 / args.rate)
            s = c.state
            v = np.asarray(s.velocity_ned, dtype=np.float64)
            rel = np.asarray(s.position_ned, dtype=np.float64) - origin
            print(f"\n   end {name}: thrust={thr:.2f} -> vz={v[2]:+.2f} m/s (up={-v[2]:+.2f}) rel_pos={np.round(rel,2)}")
            if aborted:
                print(f"   *** ABORT during {name}: {aborted}")
                break
    except KeyboardInterrupt:
        aborted = "ctrl-c"
    finally:
        print("\n[safety] force-disarming ...")
        try:
            c.send_heartbeats = True
            c.disarm(force=True)
            c.wait_armed(False, timeout_s=3.0)
        except Exception as exc:
            print(f"  disarm error: {exc}", file=sys.stderr)
        recorder.add_meta(aborted=aborted)
        recorder.close()
    print(f"\n==== angle_att_test: aborted={aborted}  recording={session} ====")
    print("  KEY: did vz (climb rate) DECREASE as thrust dropped (thrust honored -> controllable),")
    print("       or climb hard regardless (auto-thrust -> dead, go CTBR)? Watch the per-thrust vz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
