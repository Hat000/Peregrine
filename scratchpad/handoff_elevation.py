"""Where does the gate ACTUALLY sit in the camera at the wire handoff?

v1.7's M1 was designed from this premise (peregrine_racing_ego.py, M1 header):

    spawn attitude              optical-axis elevation
    VQ1 tilted pad (-17.8 deg)        +2.0 deg
    WIRE handoff (LEVEL)             +19.5 deg
  "at the wire handoff -- level, ~4 m up, gate 10-11 m out at roughly its own
   height -- it sits ~20 deg BELOW [the axis]"

Two claims are load-bearing there: (a) the wire hands over LEVEL, and (b) the gate
is at roughly the drone's own height.  Both are checkable from the logs.

rel_flu is the TRUE unflipped body FLU vector to the gate (+x fwd, +y left, +z up),
so it ALREADY contains the attitude -- the gate's elevation off the body forward axis
is atan2(z, x), and the camera sits +20 deg (ego_cam_mount_pitch_deg) above that axis.
Elevation off the OPTICAL AXIS is therefore atan2(z,x) - 20 deg.  Zero = dead centre.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs")
MOUNT_DEG = 20.0
VFOV_HALF_DEG = 29.35        # 58.7 deg vertical FOV


def q(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(p * len(s))))]


def main() -> int:
    obs4, elev_body, elev_axis = [], [], []

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        r = next((r for r in rows
                  if not r.get("assist", False) and r.get("pose_seen") and r.get("rel_flu")),
                 None)
        if r is None:
            continue
        x, _y, z = r["rel_flu"]
        obs4.append(r["obs"][4])
        e = math.degrees(math.atan2(z, x))
        elev_body.append(e)
        elev_axis.append(e - MOUNT_DEG)

    n = len(obs4)
    print(f"{n} flights, first tick after assist release")
    print()
    print(f"  {'quantity':<34}{'p5':>9}{'median':>9}{'p95':>9}")
    print("  " + "-" * 61)
    print(f"  {'obs[4] pitch attitude (rad)':<34}{q(obs4,.05):>9.4f}{q(obs4,.5):>9.4f}"
          f"{q(obs4,.95):>9.4f}")
    print(f"  {'gate elev off BODY fwd (deg)':<34}{q(elev_body,.05):>9.2f}"
          f"{q(elev_body,.5):>9.2f}{q(elev_body,.95):>9.2f}")
    print(f"  {'gate elev off OPTICAL AXIS (deg)':<34}{q(elev_axis,.05):>9.2f}"
          f"{q(elev_axis,.5):>9.2f}{q(elev_axis,.95):>9.2f}")
    print()

    level = sum(1 for v in obs4 if abs(v) < 0.05)
    padded = sum(1 for v in obs4 if abs(v + 0.3107) < 0.05)
    print(f"  handed over LEVEL   (|obs[4]| < 0.05 rad): {level:>4} / {n} "
          f"= {100.0*level/n:.1f}%")
    print(f"  handed over AT PAD TILT (obs[4] ~ -0.3107): {padded:>4} / {n} "
          f"= {100.0*padded/n:.1f}%")
    print()
    centred = sum(1 for e in elev_axis if abs(e) <= VFOV_HALF_DEG / 2)
    below20 = sum(1 for e in elev_axis if e <= -15.0)
    print(f"  gate within the inner half of the frame (|elev| <= {VFOV_HALF_DEG/2:.1f} deg): "
          f"{centred} / {n} = {100.0*centred/n:.1f}%")
    print(f"  gate >=15 deg BELOW the axis (M1's premise): {below20} / {n} "
          f"= {100.0*below20/n:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
