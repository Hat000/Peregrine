"""Find the sim's flight-mode switch. Read-only HEARTBEAT mode decoder (+ optional client probes).

The reference sample never changes modes (it arms + flies ACRO actuator/body-rate), and our
client-sent attitude/velocity setpoints did NOT pull the sim out of ACRO (teammate confirmed via
the UI). So the ACRO/ANGLE mode is almost certainly sim/GUI-side. This tool decodes the live
HEARTBEAT ``base_mode`` flags + ``custom_mode`` and prints on every CHANGE, so:

  * the teammate can toggle the GUI mode (the "upper-right" indicator) while we WATCH what
    base_mode/custom_mode value each mode maps to -- then we can replay it from the client; and
  * ``--try-modes`` sweeps client SET_MODE / MAV_CMD_DO_SET_MODE attempts and reports whether ANY
    of them moves the heartbeat (does the sim accept a client mode change at all?).

NO arming, NO setpoints, NO actuation. Safe to run anytime, even at the home page.

Usage:
  python scripts/mode_watch.py --seconds 120                 # watch; toggle the GUI mode, I'll log it
  python scripts/mode_watch.py --try-modes --seconds 60      # also try client mode commands
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pymavlink import mavutil

from racer.mavlink_client import MavlinkClient

_FLAGS = [
    ("SAFETY_ARMED", mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
    ("MANUAL_INPUT", mavutil.mavlink.MAV_MODE_FLAG_MANUAL_INPUT_ENABLED),
    ("HIL", mavutil.mavlink.MAV_MODE_FLAG_HIL_ENABLED),
    ("STABILIZE", mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED),
    ("GUIDED", mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED),
    ("AUTO", mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED),
    ("TEST", mavutil.mavlink.MAV_MODE_FLAG_TEST_ENABLED),
    ("CUSTOM_MODE", mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
]


def _decode(base_mode: int | None) -> str:
    if base_mode is None:
        return "?"
    on = [name for name, bit in _FLAGS if base_mode & bit]
    return "|".join(on) if on else "(none)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--silent", action="store_true",
                    help="send NOTHING (no heartbeat): pure passive recv. Tests if OUR heartbeat forces ACRO.")
    ap.add_argument("--try-modes", action="store_true", help="also send client SET_MODE/DO_SET_MODE attempts")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    c = MavlinkClient(args.endpoint)
    print(f"connecting {args.endpoint} (wait_heartbeat=False) ... READ-ONLY (no arm, no setpoints)")
    c.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)
    print(">>> Toggle the sim's flight mode in the GUI (the upper-right indicator) while I watch.")
    print("    I'll log every base_mode / custom_mode change so we learn the ANGLE/position value.\n")

    if args.silent:
        print("    [--silent] sending NOTHING (no heartbeat). If the mode stays ANGLE now but goes")
        print("    ACRO when a heartbeat is sent, OUR heartbeat is the trigger.\n")

    def _recv_only():
        # drain RX WITHOUT sending a heartbeat (pump() would send one)
        while True:
            msg = c.conn.recv_match(blocking=False)
            if msg is None:
                break
            c._handle(msg)

    prev = None
    last = 0.0
    tried = False
    hb_announced = False
    start = time.monotonic()
    end = start + args.seconds
    hb_at = start + args.seconds / 2.0   # in --silent: stay passive, then start heartbeating here
    try_at = start + 4.0                 # let a baseline establish first
    while time.monotonic() < end:
        now = time.monotonic()
        if args.silent and now < hb_at:
            _recv_only()                 # phase 1: send NOTHING
        else:
            if args.silent and not hb_announced:
                print(f"\n\n>>> t+{now-start:.0f}s: NOW SENDING our 2 Hz heartbeat. Watch for a flip to ACRO ...\n")
                hb_announced = True
            c.pump()                     # phase 2 (or non-silent): sends the 2 Hz heartbeat
        base = c.state.status_flags
        key = (base, c.custom_mode, c.autopilot, c.vehicle_type)
        if key != prev:
            print(f"\n*** MODE CHANGE  base_mode={base} [{_decode(base)}]  custom_mode={c.custom_mode}  "
                  f"autopilot={c.autopilot} type={c.vehicle_type}  armed={c.state.armed}")
            prev = key
        if now - last >= 2.0:
            print(f"  base_mode={base} [{_decode(base)}] custom_mode={c.custom_mode} armed={c.state.armed}   ",
                  end="\r", flush=True)
            last = now

        if args.try_modes and not tried and now >= try_at:
            tried = True
            print("\n\n--try-modes: sweeping client mode commands; watch for a MODE CHANGE above ...")
            tgt_sys = (c.conn.target_system or 1) if c.conn else 1
            # (a) SET_MODE with each base-mode flag (the classic switch).
            for name, bit in _FLAGS:
                if name in ("SAFETY_ARMED", "HIL", "TEST"):
                    continue
                print(f"  SET_MODE base_mode={name}")
                c.conn.mav.set_mode_send(tgt_sys, bit, 0)
                _drain(c, 0.8)
            # (b) MAV_CMD_DO_SET_MODE with CUSTOM_MODE_ENABLED + custom_mode 0..8 (PX4-style).
            cmf = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            for cm in range(0, 9):
                print(f"  DO_SET_MODE custom_mode={cm}")
                c.send_command_long(mavutil.mavlink.MAV_CMD_DO_SET_MODE, float(cmf), float(cm), 0.0)
                _drain(c, 0.8)
            print("--try-modes done. If nothing changed above, the client cannot set the mode.\n")
        time.sleep(0.004)
    print("\ndone.")
    return 0


def _drain(c: MavlinkClient, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        c.pump()
        ack = c.last_command_ack
        if ack and ack.get("command") == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            print(f"      -> COMMAND_ACK DO_SET_MODE: {ack['result_name']}")
            c.last_command_ack = None
        time.sleep(0.004)


if __name__ == "__main__":
    raise SystemExit(main())
