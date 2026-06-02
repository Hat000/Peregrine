"""Capture the TRACK_INFO gate map and save it to JSON (the deterministic course map).

KEY TIMING (learned at first contact): the map is broadcast ONCE, at race/level LOAD, to whoever
is ALREADY connected. Connecting after the level has loaded misses the one-shot. So:
  1. start this AT THE HOME PAGE (before any race/level is loaded),
  2. then navigate the sim: home -> waiting room -> Race! while it runs.
We connect with wait_heartbeat=False so not a single datagram is consumed before our handler runs.

The course is deterministic, so capture once and reuse the saved map for every future run on this
track. Read-only: no arm, no setpoints. Saves data/runs/track_map_<stamp>.json.

Usage:  python scripts/capture_track_map.py [--seconds 120] [--out data/runs]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.mavlink_client import MavlinkClient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--seconds", type=float, default=120.0, help="how long to wait for the map")
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    client = MavlinkClient(args.endpoint)
    print(f"connecting {args.endpoint} (wait_heartbeat=False so nothing is consumed) ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)
    print(">>> NOW navigate the sim: home -> waiting room -> Race!  (the map loads WITH the level).")

    end = time.monotonic() + args.seconds
    last = 0.0
    while time.monotonic() < end and not client.track_gates:
        client.pump()
        now = time.monotonic()
        if now - last >= 3.0:
            rs = client.race_status
            print(f"  .. waiting for the map  (race started={rs['started'] if rs else '?'})")
            last = now
        time.sleep(0.003)

    if not client.track_gates:
        print("\nNO gate map captured. Re-run and be sure to START THIS AT THE HOME PAGE, then enter the race.",
              file=sys.stderr)
        return 1

    gates = client.track_gates
    out = []
    print(f"\n*** captured {len(gates)} gates ***")
    for g in gates:
        rec = {
            "gate_id": int(g["gate_id"]),
            "position_ned": [float(x) for x in g["position_ned"]],
            "orientation_ned_wxyz": [float(x) for x in g["orientation_ned_wxyz"]],
            "width_m": float(g["width_m"]),
            "height_m": float(g["height_m"]),
        }
        out.append(rec)
        print(f"  gate {rec['gate_id']}: ned={[round(x, 2) for x in rec['position_ned']]} "
              f"quat_wxyz={[round(x, 3) for x in rec['orientation_ned_wxyz']]} "
              f"w={rec['width_m']:.3f} h={rec['height_m']:.3f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"track_map_{stamp}.json"
    path.write_text(json.dumps(
        {"captured_utc": stamp, "endpoint": args.endpoint, "num_gates": len(out),
         "dims_note": "width/height are the OUTER gate square (~2.72 m); inner opening ~1.5 m is what PnP uses",
         "gates": out}, indent=2))
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
