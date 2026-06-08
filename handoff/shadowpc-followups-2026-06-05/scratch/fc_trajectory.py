"""Throwaway: does firstcontact_race1 actually approach gate 0? Map its given trajectory +
attitude, and check video<->telemetry clock alignment (recv).

Clocks (memory): video sim_time = server UNIX ns; align via recv-monotonic. The tlog prefix gives
msg._timestamp (recv UNIX sec, from the monotonic->unix bridge in meta). Video_index gives
recv_monotonic_ns -> convert to recv UNIX via the same meta bridge. Then nearest-recv pairs them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path("C:/Users/Shadow/Peregrine/.claude/worktrees/blissful-sutherland-c8ebd9/src")
sys.path.insert(0, str(SRC))
from racer.frames import euler_from_quat_wxyz  # noqa: E402
from racer.recording import RecordingReader  # noqa: E402

RUN = Path("C:/Users/Shadow/Peregrine/data/runs/20260602_000538_firstcontact_race1")
GATE0 = np.array([-23.30, -0.40, -0.03])


def main() -> None:
    reader = RecordingReader(RUN)
    meta = reader.meta
    t0u = int(meta["t0_unix_ns"])
    t0m = int(meta["t0_monotonic_ns"])

    lpn, odo = [], []  # (recv_unix_s, ...)
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            lpn.append((float(msg._timestamp), float(msg.x), float(msg.y), float(msg.z)))
        elif t == "ODOMETRY":
            q = [float(v) for v in msg.q]
            r, p, y = euler_from_quat_wxyz(q)
            odo.append((float(msg._timestamp), float(msg.x), float(msg.y), float(msg.z),
                        np.degrees(r), np.degrees(p), np.degrees(y)))
    lpn.sort(); odo.sort()
    print(f"LPN n={len(lpn)} ODO n={len(odo)}")
    if not lpn:
        return
    r0 = lpn[0][0]
    lpn_a = np.array([(t - r0, x, y, z) for (t, x, y, z) in lpn])

    # video frames: dedup by frame_id, convert recv_monotonic -> recv_unix via meta bridge
    seen = set()
    frames = []
    for e in reader.iter_video_index():
        fid = e["frame_id"]
        if fid in seen:
            continue
        seen.add(fid)
        recv_unix = (t0u + (e["recv_monotonic_ns"] - t0m)) / 1e9
        frames.append((recv_unix, fid))
    frames.sort()
    print(f"video frames: {len(list(seen))} unique (index entries differ; ~14x resend)")
    if frames:
        print(f"video recv span: {frames[0][0]-r0:.2f}..{frames[-1][0]-r0:.2f}s (rel LPN start)  "
              f"n_unique={len(frames)}")

    # trajectory: dist to gate0 + yaw, downsampled
    print(f"\n{'t_rel':>7} {'x':>8} {'y':>7} {'z':>7} {'dist_g0':>8} {'yaw_deg':>8}")
    # nearest odo yaw by recv
    odo_t = np.array([o[0] for o in odo])
    for i in range(0, len(lpn_a), max(1, len(lpn_a) // 40)):
        tr, x, y, z = lpn_a[i]
        d = float(np.linalg.norm(np.array([x, y, z]) - GATE0))
        j = int(np.argmin(np.abs(odo_t - (r0 + tr))))
        yaw = odo[j][6]
        print(f"{tr:7.2f} {x:8.2f} {y:7.2f} {z:7.2f} {d:8.2f} {yaw:8.1f}")

    # where is the drone closest to gate 0, and facing -X (yaw ~ +/-180)?
    dists = np.linalg.norm(lpn_a[:, 1:4] - GATE0, axis=1)
    imin = int(np.argmin(dists))
    print(f"\nclosest approach to gate0: dist={dists[imin]:.2f}m at t_rel={lpn_a[imin,0]:.2f}s "
          f"pos=({lpn_a[imin,1]:.2f},{lpn_a[imin,2]:.2f},{lpn_a[imin,3]:.2f})")
    # x-range of the flight
    print(f"x range: [{lpn_a[:,1].min():.2f} .. {lpn_a[:,1].max():.2f}]  "
          f"y range: [{lpn_a[:,2].min():.2f} .. {lpn_a[:,2].max():.2f}]")


if __name__ == "__main__":
    main()
