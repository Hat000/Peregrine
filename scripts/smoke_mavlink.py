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
            t_s = s.sim_time_ns / 1e9
            ax, ay, az = s.accel_body
            v = s.velocity_ned
            vstr = f"({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})" if v is not None else "n/a"
            pos = s.position_ned
            pstr = f"({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})" if pos is not None else "n/a"
            baro = f"{s.baro_pressure_hpa:.1f}" if s.baro_pressure_hpa is not None else "n/a"
            mag = "yes" if s.mag_body is not None else "no"
            print(
                f"t={t_s:8.2f}  armed={s.armed}  "
                f"rpy=({s.roll:+.2f},{s.pitch:+.2f},{s.yaw:+.2f})  "
                f"acc=({ax:+.2f},{ay:+.2f},{az:+.2f})  "
                f"baro={baro}  mag={mag}  pos={pstr}  v={vstr}"
            )
            last_log = now
        time.sleep(0.005)
    if client.unknown_msg_types:
        print(f"observed unknown message types: {sorted(client.unknown_msg_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
