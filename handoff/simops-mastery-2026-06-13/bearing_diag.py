"""bearing_diag.py — why does inc7-at-speed yield so few accepted vision fixes?

For each frame in a per-gate bundle, compute (independent of the detector) the TRUE geometry from
the given pose + ODOMETRY attitude (the same R_wb the navigator's PnP uses):
  - range to the bundle's primary gate
  - off-axis BEARING of that gate from the camera optical axis (camera = body + R_camera_from_body,
    +20deg pitch up, optical Z-forward). HFoV 90deg => +-45deg horizontal; VFoV ~58.7 => +-29deg.
  - whether the gate centre projects INSIDE the 640x360 image at all
Cross with the detector/association outcome from the characterize JSON (detected / associated).

If bearings are systematically > ~45deg (gate outside HFoV) on the frames that fail to associate,
the low fix yield is the inc7 CRAB geometry (camera pointed off-gate), not a chain bug.

Usage: .venv\\Scripts\\python handoff/simops-mastery-2026-06-13/bearing_diag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from racer import frames as F
from racer.navigator import load_track_map
from racer.vision.association import predict_gates_in_camera

BUNDLES = ROOT / "handoff/simops-mastery-2026-06-13/bundles"
LOGS = ROOT / "handoff/simops-mastery-2026-06-13/logs"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
W, H = 640, 360


def main() -> int:
    gates = load_track_map(str(MAP), corner_to_center=True)

    print(f"{'gate':>4} {'N':>4} {'det':>4} {'assoc':>5} "
          f"{'bearing(deg): p50':>17} {'p90':>6} {'max':>6}  {'in-FoV%':>7}  {'range p50':>9}")
    pooled = []
    for g in range(6):
        bdir = BUNDLES / f"cr1b_g{g}"
        fj = bdir / "frames.json"
        if not fj.exists():
            continue
        d = json.loads(fj.read_text())
        # detector/assoc outcome per frame_id from the characterize JSON
        outcome = {}
        cj = LOGS / f"char_cr1b_g{g}.json"
        if cj.exists():
            for r in json.loads(cj.read_text())["rows"]:
                outcome[r["frame_id"]] = (r.get("detected", False), r.get("associated", False))
        bearings, in_fov, ranges, det_n, assoc_n = [], 0, [], 0, 0
        for fr in d["frames"]:
            drone = np.asarray(fr["drone_position_ned"], float)
            R_wb = F.R_world_from_odo_quat_wxyz(np.asarray(fr["odo_q_wxyz"], float))
            # faithful navigator projection of all gates into the camera; read the target gate g
            predicted = predict_gates_in_camera(gates, drone, R_wb)
            pg = predicted.get(g)
            if pg is None:                  # gate behind camera / not projected
                bearing, inside, rng = 180.0, False, float("nan")
            else:
                t = pg.t_cam_gate
                bearing = float(np.degrees(np.arctan2(np.hypot(t[0], t[1]), t[2])))
                px, py = pg.center_px
                inside = (0 <= px < W) and (0 <= py < H)
                rng = pg.range_m
            bearings.append(bearing)
            ranges.append(rng)
            in_fov += int(inside)
            det, assoc = outcome.get(fr["frame_id"], (False, False))
            det_n += int(det)
            assoc_n += int(assoc)
            pooled.append({"gate": g, "range": rng, "bearing": bearing, "inside": inside,
                           "det": det, "assoc": assoc, "speed": fr["speed_mps"]})
        b = np.array(bearings)
        print(f"{g:>4} {len(b):>4} {det_n:>4} {assoc_n:>5} "
              f"{np.percentile(b,50):>17.1f} {np.percentile(b,90):>6.1f} {b.max():>6.1f}  "
              f"{100*in_fov/len(b):>6.0f}%  {np.nanpercentile(ranges,50):>9.1f}")

    # association vs bearing: do fixes only happen when the gate is within FoV?
    pa = [p for p in pooled if p["assoc"]]
    pin = [p for p in pooled if p["inside"]]
    pin_det = [p for p in pin if p["det"]]
    print(f"\nPOOLED (N={len(pooled)}): detector fired {sum(p['det'] for p in pooled)}, "
          f"associated {len(pa)}")
    print(f"gate centre projects INSIDE image in {len(pin)}/{len(pooled)} frames "
          f"({100*len(pin)/len(pooled):.0f}%)")
    if pin_det:
        print(f"  of in-FoV frames: detector fired {sum(p['det'] for p in pin)}/{len(pin)}, "
              f"associated {sum(p['assoc'] for p in pin)}/{len(pin)}")
    if pa:
        ba = np.array([p["bearing"] for p in pa])
        print(f"associated-frame bearings: p50 {np.percentile(ba,50):.1f}  "
              f"max {ba.max():.1f} deg  (HFoV edge = 45 deg)")
    # how many frames have the gate within the nominal +-45 deg HFoV half-angle?
    within45 = [p for p in pooled if p["bearing"] <= 45]
    print(f"frames with target-gate bearing <= 45 deg (within HFoV half-angle): "
          f"{len(within45)}/{len(pooled)} ({100*len(within45)/len(pooled):.0f}%); "
          f"of those, associated {sum(p['assoc'] for p in within45)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
