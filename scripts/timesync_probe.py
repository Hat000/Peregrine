"""First-contact probe: does client-initiated TIMESYNC@10Hz unlock the TRACK_INFO gate map?

READ-ONLY (no arm, no setpoints). The official PyAIPilotExample is the only client known to
receive the gate map, and TIMESYNC@10Hz (client-initiated) is the one sample behaviour we had
NOT replicated when the map failed to arrive. This isolates it: connect, request TIMESYNC at
10 Hz, and watch for (a) TIMESYNC responses and (b) the gate map (DATA_TRANSMISSION_HANDSHAKE +
the reassembled TRACK_INFO). Run it in a RUNNING race (the sim freezes physics on the home page).

If the map arrives, it also prints the first gates' width/height -> resolves the inner(1.5)-vs-
outer(2.7) dims question. If it does NOT, the trigger is something else (arm + 250 Hz control, or
a request message) and we escalate.

Usage:
  python scripts/timesync_probe.py [--endpoint udp:127.0.0.1:14550] [--seconds 20] [--ts-hz 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.firstcontact import backend_summary, mission_summary, telemetry_summary
from racer.mavlink_client import MavlinkClient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--ts-hz", type=float, default=10.0, help="TIMESYNC request rate (sample uses 10)")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    client = MavlinkClient(args.endpoint)
    counts: Counter = Counter()
    client.on_message = lambda m: counts.update([m.get_type()])

    print(f"connecting {args.endpoint} (waiting for heartbeat) ...")
    client.connect(timeout_s=args.connect_timeout)
    print("heartbeat OK.")
    print(f"  backend:   {backend_summary(client)}")
    print(f"  telemetry: {telemetry_summary(client)}")
    print(f"  mission:   {mission_summary(client)}")

    print(f"\nrequesting TIMESYNC @ {args.ts_hz:g} Hz for {args.seconds:g}s, watching for the gate map ...")
    ts_dt = 1.0 / args.ts_hz
    last_ts = 0.0
    sim_t0 = client.state.sim_time_ns
    t_start = time.monotonic()
    end = t_start + args.seconds
    announced = False
    while time.monotonic() < end:
        now = time.monotonic()
        if now - last_ts >= ts_dt:
            client.conn.mav.timesync_send(int(time.time_ns()), 0)  # tc1=client time, ts1=0 -> request
            last_ts = now
        client.pump()
        if not announced and client.track_gates:
            announced = True
            print(f"  >>> GATE MAP ARRIVED after ~{now - t_start:.1f}s: {len(client.track_gates)} gates")
        time.sleep(0.002)

    sim_t1 = client.state.sim_time_ns
    print("\n==== timesync_probe summary ====")
    advanced = (sim_t1 - sim_t0) / 1e9
    print(f"  sim_time advanced {advanced:.3f}s ({'running' if advanced > 0 else 'FROZEN -> not in a live race'})")
    print(f"  TIMESYNC responses received: {counts.get('TIMESYNC', 0)}")
    print(f"  handshakes seen: {counts.get('DATA_TRANSMISSION_HANDSHAKE', 0)}")
    if client.track_gates:
        print(f"  R4 RESOLVED: GATE MAP RECEIVED -> {len(client.track_gates)} gates (TIMESYNC is part of the trigger).")
        for g in client.track_gates[:4]:
            print(f"     gate {g['gate_id']}: pos_ned={g['position_ned'].round(2)} "
                  f"w={g['width_m']:.3f} h={g['height_m']:.3f}  (inner~1.5 / outer~2.7?)")
    else:
        print("  R4: still NO gate map even with TIMESYNC -> trigger needs more (arm + 250Hz control, or a request).")
    print(f"  msg types seen: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
