"""Read-only observer: watch RACE_STATUS + drone state across a fresh race start.

Connects at the HOME PAGE (wait_heartbeat=False) and logs, as you navigate home -> waiting
room -> Race, exactly how the race start works -- so we can pin the COUNTDOWN mechanics
(does ``started`` flip at level-load or at GO? what are sim_boot vs race_start_boot?) and
confirm the drone actually resets to the origin. NO arming, NO setpoints, NO actuation.

Prints a line whenever started/finished/active_gate/reset_counter changes, plus a ~2 Hz
heartbeat with sim_t, position, attitude, and the RACE_STATUS timing fields.

Usage:  python scripts/race_observe.py [--seconds 120]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.mavlink_client import MavlinkClient


def _fmt_rs(rs: dict | None) -> str:
    if rs is None:
        return "RACE_STATUS=none"
    cd = (rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]) / 1000.0
    return (f"started={rs['started']} finished={rs['finished']} gate={rs['active_gate_index']} "
            f"sim_boot={rs['sim_boot_time_ms']}ms race_start={rs['race_start_boot_time_ms']}ms "
            f"(start-now={cd:+.2f}s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args()

    c = MavlinkClient(args.endpoint)
    print(f"connecting {args.endpoint} (wait_heartbeat=False) ... observe ONLY, no actuation")
    c.connect(wait_heartbeat=False)
    print(">>> Do a FULL reset now: home page -> waiting room -> Race.  I'll log the transition.")

    end = time.monotonic() + args.seconds
    last = 0.0
    prev_key = None
    while time.monotonic() < end:
        c.pump()
        s = c.state
        rs = c.race_status
        key = None if rs is None else (rs["started"], rs["finished"], rs["active_gate_index"], s.reset_counter)
        now = time.monotonic()
        if key != prev_key and rs is not None:
            pos = "none" if s.position_ned is None else f"({s.position_ned[0]:+.1f},{s.position_ned[1]:+.1f},{s.position_ned[2]:+.1f})"
            print(f"\n*** CHANGE  sim_t={s.sim_time_ns/1e9:8.3f}s reset_ctr={s.reset_counter} pos={pos} "
                  f"map={len(c.track_gates or [])}  {_fmt_rs(rs)}")
            prev_key = key
        if now - last >= 0.5:
            pos = "none" if s.position_ned is None else f"({s.position_ned[0]:+6.1f},{s.position_ned[1]:+6.1f},{s.position_ned[2]:+6.1f})"
            print(f"  sim_t={s.sim_time_ns/1e9:8.3f}s pos={pos} rpy=({s.roll:+.2f},{s.pitch:+.2f},{s.yaw:+.2f}) "
                  f"map={len(c.track_gates or [])}  {_fmt_rs(rs)}   ", end="\r", flush=True)
            last = now
        time.sleep(0.004)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
