"""Connect to the sim's MAVLink endpoint and print telemetry for 10 seconds.

Usage:  python scripts\\smoke_mavlink.py [endpoint]
        endpoint defaults to udp:127.0.0.1:14550
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.mavlink_client import MavlinkClient


def main() -> int:
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "udp:127.0.0.1:14550"
    print(f"connecting to {endpoint}...")
    client = MavlinkClient(endpoint)
    client.connect(timeout_s=15.0)
    print("heartbeat received. printing telemetry for 10s.")
    t_end = time.monotonic() + 10.0
    last_log = 0.0
    while time.monotonic() < t_end:
        client.pump()
        now = time.monotonic()
        if now - last_log >= 0.5:
            s = client.state
            print(
                f"t={s.timestamp_s:7.2f}  "
                f"rpy=({s.roll:+.2f},{s.pitch:+.2f},{s.yaw:+.2f})  "
                f"acc=({s.xacc:+.2f},{s.yacc:+.2f},{s.zacc:+.2f})  "
                f"v=({s.vx:+.2f},{s.vy:+.2f},{s.vz:+.2f})"
            )
            last_log = now
        time.sleep(0.005)
    if client.unknown_msg_types:
        print(f"observed unknown message types: {sorted(client.unknown_msg_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
