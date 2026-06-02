"""Definitive R4 test: full PyAIPilotExample startup replication to trigger the TRACK_INFO map.

The official sample is the only client known to receive the gate map. It does THREE things we had
only ever tested separately: client TIMESYNC@10Hz (a background thread) + a real disarmed->armed
transition while already connected + a continuous control stream. This does all three together.
Control = ZERO motors, so the drone does NOT fly (it just needs to BE an active, armed pilot at the
moment of the arm transition). Watches for DATA_TRANSMISSION_HANDSHAKE + the reassembled gate map.

Run in a RUNNING race. Force-disarms on exit. If the map arrives it prints gate dims (settles the
inner-1.5 vs outer-2.7 question). If not, the push is likely one-time at race-start before we
connect -> connect at the staging room through Race!, or fall back to vision gate-discovery.

Usage:  python scripts/gatemap_probe.py [--endpoint udp:127.0.0.1:14550] [--seconds 15]
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pymavlink import mavutil

from racer.firstcontact import backend_summary, mission_summary, telemetry_summary
from racer.mavlink_client import MavlinkClient

_ARM = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--seconds", type=float, default=15.0, help="armed+control watch window")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    client = MavlinkClient(args.endpoint)
    counts: Counter = Counter()
    client.on_message = lambda m: counts.update([m.get_type()])

    print(f"connecting {args.endpoint} (waiting for heartbeat) ...")
    client.connect(timeout_s=args.connect_timeout)
    t = time.monotonic()
    while time.monotonic() - t < 1.0:  # pump briefly so the state/summary are populated
        client.pump()
        time.sleep(0.005)
    print("heartbeat OK.")
    print(f"  backend:   {backend_summary(client)}")
    print(f"  telemetry: {telemetry_summary(client)}")
    print(f"  mission:   {mission_summary(client)}")

    # TIMESYNC@10Hz in the background, exactly like the sample's timesync thread.
    stop = threading.Event()

    def ts_loop() -> None:
        while not stop.is_set():
            try:
                client.conn.mav.timesync_send(int(time.time_ns()), 0)  # tc1=client, ts1=0 -> request
            except Exception:
                pass
            time.sleep(0.1)

    threading.Thread(target=ts_loop, daemon=True).start()

    try:
        print("\n[reset] force-disarming for a clean baseline ...")
        client.disarm(force=True)
        client.wait_armed(False, timeout_s=3.0)

        print("[arm] arming -> a clean disarmed->armed transition while timesyncing ...")
        client.last_command_ack = None
        client.arm()
        ack = client.wait_command_ack(_ARM, timeout_s=3.0)
        armed = client.wait_armed(True, timeout_s=5.0)
        print(f"  arm ack={ack['result_name'] if ack else 'none'}  armed={armed}")

        print(f"[control] streaming ZERO motors @~100Hz for {args.seconds:g}s, watching for the map ...")
        end = time.monotonic() + args.seconds
        announced = False
        while time.monotonic() < end:
            client.send_actuator_control([0.0] * 8)  # zero -> no thrust, drone sits; still an active stream
            client.pump()
            if not announced and client.track_gates:
                announced = True
                print(f"  >>> GATE MAP ARRIVED: {len(client.track_gates)} gates")
            time.sleep(0.01)
    finally:
        stop.set()
        print("[safety] disarming ...")
        client.disarm(force=True)
        client.wait_armed(False, timeout_s=3.0)

    print("\n==== gatemap_probe summary ====")
    print(f"  TIMESYNC responses: {counts.get('TIMESYNC', 0)}   "
          f"DATA_TRANSMISSION_HANDSHAKE: {counts.get('DATA_TRANSMISSION_HANDSHAKE', 0)}")
    if client.track_gates:
        print(f"  R4 RESOLVED: {len(client.track_gates)} gates received.")
        for g in client.track_gates[:6]:
            print(f"    gate {g['gate_id']}: pos_ned={g['position_ned'].round(2)} "
                  f"w={g['width_m']:.3f} h={g['height_m']:.3f}")
    else:
        print("  R4 STILL no map (full sample replication failed) -> likely a ONE-TIME push at")
        print("     race-start/level-load BEFORE we connect. Next: connect at the STAGING room and stay")
        print("     through Race!, or PARK R4 and use vision gate-discovery (gates are clearly visible).")
    print(f"  msg types: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
