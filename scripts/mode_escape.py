"""Definitive test: can we LEAVE ACRO and use attitude / velocity / position 'easy mode'?

The records contradict themselves: first contact saw the UI switch to ANGLE on an attitude-quat
setpoint, but later body-rate runs reported "never leaves ACRO" and position/velocity runaways.
First-contact TODO #4 (re-test pos/vel in ANGLE mode, entered via an attitude setpoint) was never
actually done. If ANGLE + position/velocity works, the whole CTBR effort is unnecessary -- so this
is worth a clean, bounded check. The teammate WATCHES the sim's flight-mode indicator and reports
it per phase.

Three phases, each a few seconds, measuring ACHIEVED vs COMMANDED so the verdict is unambiguous
(not the old climb>0.2 m false positive):
  A attitude_level  send a LEVEL attitude-quat + hover-ish thrust -> does the UI show ANGLE? does
                    the drone hold ~level and roughly hover (vs tumble/run away)?
  B velocity_fwd    send a small velocity setpoint toward gate 0 -> does the achieved velocity
                    TRACK the command (~1 m/s) or run away?
  C position_hold   send the current position as a setpoint -> does it HOLD, or drift/run away?

Bounded: hard-abort + force-disarm on offset/altitude/tilt/collision/time. Respects the countdown
(reuses fly_vq1._wait_for_race). Records the contact.

Usage:  python scripts/mode_escape.py --label escape1
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from fly_vq1 import _arm_cmd, _wait_for_race

from racer.contracts import ControlCommand, ControlMode
from racer.firstcontact import backend_summary, telemetry_summary
from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp


def _level_quat(yaw: float) -> np.ndarray:
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class _Frames:    # _wait_for_race expects a .get()
    def get(self):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--thrust", type=float, default=0.25, help="attitude-phase collective (near hover)")
    ap.add_argument("--vel", type=float, default=1.0, help="velocity-phase speed toward gate 0 (m/s)")
    ap.add_argument("--phase-s", type=float, default=3.0)
    ap.add_argument("--phases", default="A,B,C", help="which phases to run (A=attitude,B=velocity,C=position)")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--max-offset-m", type=float, default=12.0)
    ap.add_argument("--max-alt-m", type=float, default=8.0)
    ap.add_argument("--max-tilt-deg", type=float, default=80.0)
    ap.add_argument("--wait-seconds", type=float, default=240.0)
    ap.add_argument("--start-margin-s", type=float, default=0.3)
    ap.add_argument("--max-start-offset-m", type=float, default=5.0)
    ap.add_argument("--no-wait-start", action="store_true")
    ap.add_argument("--label", default="escape")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    client = MavlinkClient(args.endpoint)
    print(f"connecting {args.endpoint} (wait_heartbeat=False) ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)
    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(probe="mode_escape", label=args.label)
    print(f"recording -> {session}")
    client.on_message = lambda m: (m.get_type() != "BAD_DATA" and m.get_msgbuf()
                                   and recorder.record_mavlink(bytes(m.get_msgbuf())))

    aborted = None
    armed = False
    try:
        if not _wait_for_race(client, _Frames(), args):
            print("no fresh race GO -> aborting.", file=sys.stderr)
            return 1
        print(f"  backend: {backend_summary(client)}")
        print(f"  {telemetry_summary(client)}")
        client.last_command_ack = None
        client.arm()
        client.wait_command_ack(_arm_cmd(), timeout_s=3.0)
        armed = client.wait_armed(True, timeout_s=5.0)
        print(f"  armed={armed}")
        if not armed:
            return 1

        yaw_hold = float(client.state.yaw)
        origin = np.asarray(client.state.position_ned, dtype=np.float64).copy()
        level_q = _level_quat(yaw_hold)
        fwd = np.array([np.cos(yaw_hold), np.sin(yaw_hold), 0.0])    # body-forward in world (toward gate 0)
        print(f"  yaw_hold={np.degrees(yaw_hold):+.1f} deg  origin={np.round(origin,2)}  forward={np.round(fwd,2)}")

        phases = [
            ("A attitude_level", lambda pos: ControlCommand(mode=ControlMode.ATTITUDE,
                                                            attitude_quat_wxyz=level_q, thrust=args.thrust)),
            ("B velocity_fwd", lambda pos: ControlCommand(mode=ControlMode.VELOCITY,
                                                          velocity_ned=args.vel * fwd, yaw=yaw_hold)),
            ("C position_hold", None),    # position captured at phase start
        ]
        want = {p.strip().upper() for p in args.phases.split(",")}
        phases = [p for p in phases if p[0][0] in want]
        dt = 1.0 / args.rate
        for name, mk in phases:
            print(f"\n>>> PHASE {name}  ({args.phase_s:g}s) -- TEAMMATE: what does the sim mode say (ACRO/ANGLE)?")
            hold_pos = np.asarray(client.state.position_ned, dtype=np.float64).copy()
            end = time.monotonic() + args.phase_s
            last = 0.0
            while time.monotonic() < end:
                client.pump()
                s = client.state
                if mk is None:
                    cmd = ControlCommand(mode=ControlMode.POSITION, position_ned=hold_pos, yaw=yaw_hold)
                else:
                    cmd = mk(hold_pos)
                client.send_command(cmd)
                pos = np.asarray(s.position_ned, dtype=np.float64)
                vel = np.asarray(s.velocity_ned, dtype=np.float64)
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
                    print(f"   {name[:16]:16s} rpy=({np.degrees(s.roll):+4.0f},{np.degrees(s.pitch):+4.0f},"
                          f"{np.degrees(s.yaw):+4.0f}) vel=({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f}) "
                          f"pos=({pos[0]:+.1f},{pos[1]:+.1f},{pos[2]:+.1f})   ", end="\r", flush=True)
                    last = now
                time.sleep(dt)
            # phase verdict
            s = client.state
            vel = np.asarray(s.velocity_ned, dtype=np.float64)
            rel = np.asarray(s.position_ned, dtype=np.float64) - origin
            print(f"\n   end {name}: vel=({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f}) "
                  f"rel_pos=({rel[0]:+.1f},{rel[1]:+.1f},{rel[2]:+.1f}) tilt={np.degrees(max(abs(s.roll),abs(s.pitch))):.0f}deg")
            if aborted:
                print(f"   *** ABORT during {name}: {aborted}")
                break
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C) ...")
        aborted = "ctrl-c"
    finally:
        print("\n[safety] force-disarming ...")
        try:
            client.disarm(force=True)
            client.wait_armed(False, timeout_s=3.0)
        except Exception as exc:
            print(f"  disarm error: {exc}", file=sys.stderr)
        recorder.add_meta(aborted=aborted)
        recorder.close()
    print(f"\n==== mode_escape: aborted={aborted}  recording={session} ====")
    print("  Verdict: did ANY phase TRACK (held level / matched velocity / held position)?  -> see above + UI mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
