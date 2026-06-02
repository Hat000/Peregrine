"""Offline: extract the attitude + position trajectory from a recorded run's MAVLink tlog.

Replays ``data/runs/<stamp>/mavlink.tlog`` and prints a downsampled timeline of sim-time,
position (NED), and attitude (roll/pitch/yaw from the ODOMETRY quaternion -- the trusted
source) + body rates, and flags when the attitude diverges (|roll| or |pitch| > 60 deg) so a
flip/runaway is obvious. No sim needed.

Usage:  python scripts/analyze_run.py data/runs/<stamp>_<label> [--every 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.frames import euler_from_quat_wxyz
from racer.recording import RecordingReader


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--every", type=int, default=5, help="print every Nth ODOMETRY sample")
    ap.add_argument("--flip-deg", type=float, default=60.0)
    args = ap.parse_args()

    reader = RecordingReader(args.session)
    rows = []
    for msg in reader.iter_mavlink():
        if msg.get_type() != "ODOMETRY":
            continue
        q = np.array([float(v) for v in msg.q], dtype=np.float64)   # w,x,y,z
        roll, pitch, yaw = euler_from_quat_wxyz(q)
        rows.append((float(msg.time_usec) / 1e6, msg.x, msg.y, msg.z,
                     np.degrees(roll), np.degrees(pitch), np.degrees(yaw),
                     msg.rollspeed, msg.pitchspeed, msg.yawspeed))

    if not rows:
        print("no ODOMETRY in this recording", file=sys.stderr)
        return 1
    t0 = rows[0][0]
    print(f"{len(rows)} ODOMETRY samples over {rows[-1][0] - t0:.1f}s")
    print(f"{'t':>7} {'x':>8} {'y':>7} {'z':>8} | {'roll':>7} {'pitch':>7} {'yaw':>7} | "
          f"{'p':>6} {'q':>6} {'r':>6}  (deg, rad/s)")
    flip_t = None
    for i, r in enumerate(rows):
        t, x, y, z, ro, pi, ya, p, q_, rr = r
        if flip_t is None and (abs(ro) > args.flip_deg or abs(pi) > args.flip_deg):
            flip_t = t - t0
        if i % args.every == 0 or i == len(rows) - 1:
            print(f"{t - t0:7.2f} {x:8.1f} {y:7.1f} {z:8.1f} | {ro:7.1f} {pi:7.1f} {ya:7.1f} | "
                  f"{p:6.2f} {q_:6.2f} {rr:6.2f}")
    if flip_t is not None:
        print(f"\n*** attitude exceeded {args.flip_deg:g} deg at t+{flip_t:.2f}s (flip/runaway onset)")
    else:
        print("\nattitude stayed within bounds (no flip detected)")
    # peak rates
    arr = np.array([(r[7], r[8], r[9]) for r in rows])
    print(f"peak |body rates| (rad/s): roll={np.abs(arr[:,0]).max():.2f} "
          f"pitch={np.abs(arr[:,1]).max():.2f} yaw={np.abs(arr[:,2]).max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
