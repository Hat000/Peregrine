"""Test the teammate's discovery: no client = ANGLE; our HEARTBEAT seems to force ACRO.

If the sim defaults to ANGLE (a real stabilized mode) and only flips to ACRO because of our 2 Hz
GCS heartbeat, then a SILENT client (no heartbeat) should keep the drone in ANGLE -- where a
position/velocity setpoint may actually TRACK (the easy-mode we thought was dead was only ever
tested while we were forcing ACRO). This probe sends NO heartbeat and walks three phases, with the
teammate watching the GUI mode each phase:

  OBSERVE   send NOTHING for a few s. Expect: GUI = ANGLE, drone hovers stably on its own.
  POS_HOLD  send POSITION = current position. Expect (if ANGLE has a position loop): holds, stays
            ANGLE. If it runs away or flips to ACRO, position is still dead.
  POS_UP    send POSITION = start + 2 m UP. Expect: climbs ~2 m and holds (a tracked response, not
            the 45 m ACRO runaway).

NO heartbeat, NO arming (the race auto-arms), bounded aborts + (best-effort) disarm. Reuses
fly_vq1._wait_for_race (which now pumps WITHOUT a heartbeat because send_heartbeats=False).

Usage:  python scripts/angle_test.py --label angle1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from fly_vq1 import _wait_for_race

from racer.contracts import ControlCommand, ControlMode
from racer.firstcontact import backend_summary, telemetry_summary
from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp


class _Frames:
    def get(self):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--phase-s", type=float, default=4.0)
    ap.add_argument("--climb-m", type=float, default=2.0, help="POS_UP target height above start (m)")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--max-offset-m", type=float, default=15.0)
    ap.add_argument("--max-alt-m", type=float, default=12.0)
    ap.add_argument("--max-tilt-deg", type=float, default=85.0)
    ap.add_argument("--wait-seconds", type=float, default=600.0)
    ap.add_argument("--start-margin-s", type=float, default=0.3)
    ap.add_argument("--max-start-offset-m", type=float, default=5.0)
    ap.add_argument("--no-wait-start", action="store_true")
    ap.add_argument("--arm", action="store_true", help="explicitly arm (default: rely on race auto-arm)")
    ap.add_argument("--label", default="angle")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    client = MavlinkClient(args.endpoint)
    client.send_heartbeats = False          # NO heartbeat (the suspected ACRO trigger)
    client.send_timesync = True             # but keep telemetry alive via TIMESYNC (reference client style)
    print(f"connecting {args.endpoint} (wait_heartbeat=False, heartbeat=OFF, timesync=ON) ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)
    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(probe="angle_test", label=args.label, send_heartbeats=False)
    client.on_message = lambda m: (m.get_type() != "BAD_DATA" and m.get_msgbuf()
                                   and recorder.record_mavlink(bytes(m.get_msgbuf())))
    print(f"recording -> {session}")
    print(">>> NO heartbeat is being sent. TEAMMATE: watch the GUI flight-mode indicator each phase.")

    aborted = None
    try:
        if not _wait_for_race(client, _Frames(), args):
            print("no fresh race GO -> aborting.", file=sys.stderr)
            return 1
        print(f"  backend: {backend_summary(client)}")
        print(f"  {telemetry_summary(client)}  armed={client.state.armed}")
        if args.arm and not client.state.armed:
            client.arm()
            client.wait_armed(True, timeout_s=4.0)
            print(f"  (explicit arm) armed={client.state.armed}")

        origin = np.asarray(client.state.position_ned, dtype=np.float64).copy()
        yaw0 = float(client.state.yaw)
        print(f"  origin={np.round(origin,2)}  yaw={np.degrees(yaw0):+.1f}  armed={client.state.armed}")

        def cmd_for(name, hold):
            if name == "OBSERVE":
                return None
            if name == "POS_HOLD":
                return ControlCommand(mode=ControlMode.POSITION, position_ned=hold, yaw=yaw0)
            if name == "POS_UP":
                return ControlCommand(mode=ControlMode.POSITION,
                                      position_ned=origin + np.array([0.0, 0.0, -args.climb_m]), yaw=yaw0)
            return None

        dt = 1.0 / args.rate
        for name in ("OBSERVE", "POS_HOLD", "POS_UP"):
            hold = np.asarray(client.state.position_ned, dtype=np.float64).copy()
            print(f"\n>>> PHASE {name}  ({args.phase_s:g}s) -- TEAMMATE: GUI mode now? (ANGLE/ACRO)")
            end = time.monotonic() + args.phase_s
            last = 0.0
            while time.monotonic() < end:
                client.pump()                    # send_heartbeats=False -> pure recv (+ setpoints below)
                s = client.state
                cmd = cmd_for(name, hold)
                if cmd is not None:
                    client.send_command(cmd)
                pos = np.asarray(s.position_ned, dtype=np.float64)
                rel = pos - origin
                if any(c["threat_level"] >= 2 for c in client.collisions):
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
                if now - last >= 0.4:
                    v = np.asarray(s.velocity_ned, dtype=np.float64)
                    print(f"   {name:8s} rpy=({np.degrees(s.roll):+4.0f},{np.degrees(s.pitch):+4.0f},"
                          f"{np.degrees(s.yaw):+4.0f}) vel=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) "
                          f"rel=({rel[0]:+.1f},{rel[1]:+.1f},{rel[2]:+.1f})   ", end="\r", flush=True)
                    last = now
                time.sleep(dt)
            s = client.state
            rel = np.asarray(s.position_ned, dtype=np.float64) - origin
            v = np.asarray(s.velocity_ned, dtype=np.float64)
            print(f"\n   end {name}: rel_pos=({rel[0]:+.2f},{rel[1]:+.2f},{rel[2]:+.2f}) "
                  f"vel=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) tilt={np.degrees(max(abs(s.roll),abs(s.pitch))):.0f}deg")
            if aborted:
                print(f"   *** ABORT during {name}: {aborted}")
                break
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C) ...")
        aborted = "ctrl-c"
    finally:
        print("\n[safety] force-disarming ...")
        try:
            client.send_heartbeats = True       # ok to talk now; ensure disarm lands
            client.disarm(force=True)
            client.wait_armed(False, timeout_s=3.0)
        except Exception as exc:
            print(f"  disarm error: {exc}", file=sys.stderr)
        recorder.add_meta(aborted=aborted)
        recorder.close()
    print(f"\n==== angle_test: aborted={aborted}  recording={session} ====")
    print("  KEY: did POS_HOLD/POS_UP TRACK (held / climbed ~target) and stay ANGLE? -> easy-mode lives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
