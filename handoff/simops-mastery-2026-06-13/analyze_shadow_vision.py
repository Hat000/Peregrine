"""analyze_shadow_vision.py — PART B aggregator (SIMOPS-MASTERY-SHADOWVISION).

Pool the per-gate `characterize_perception.py` JSON dumps (the navigator's REAL detect->associate->
PnP->localize chain, run in SHADOW on an inc7 given-pose flight) into the vision-vs-absolute report
the mission asks for.

Sign convention: characterize records off_ned = pos_fix - drone_truth. With corner_to_center=True a
PERFECT fix returns the given drone position, so off_ned IS the chain's world-fix error. The
vision-derived gate position = drone_truth + R_world_cam @ t_cam_gate, and the absolute gate is fixed,
so (vision_gate - absolute_gate) = -(pos_fix - drone_truth) = -off_ned. We therefore report
VISION-ABSOLUTE offset = -off_ned (per the mission's wording); |offset| is identical either way.

Reports, per gate and pooled:
  * detection -> association -> offered(depth+cap) -> chi2-accepted FUNNEL
  * VISION-ABSOLUTE offset bias+sigma per axis (N/E/D), |offset|, reproj, n_corners, range, speed
  * LATERAL (cross-track, horizontal-perpendicular-to-range) sigma vs the modeled ~0.265 m
  * the camera-coverage geometry: per frame, the min-bearing in-view gate (why fixes are scarce)

Usage: .venv\\Scripts\\python handoff/simops-mastery-2026-06-13/analyze_shadow_vision.py [--tag cr1b]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from racer import frames as F
from racer.navigator import load_track_map
from racer.vision.association import predict_gates_in_camera

HERE = ROOT / "handoff/simops-mastery-2026-06-13"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
CHI2 = 16.27
W, Himg = 640, 360


def _stats(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
                p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="cr1b,std1,std2,std3,std4,std5",
                    help="comma list of flight tags to pool (bundles/<tag>_g<g>, logs/char_<tag>_g<g>.json)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    gates = load_track_map(str(MAP), corner_to_center=True)
    gates_by_id = {g.gate_id: g for g in gates}

    # ---- load all accepted/associated rows + per-frame drone pose from bundles ----
    rows = []                # associated+solved rows, enriched
    frame_pose = {}          # (tag, frame_id) -> (drone_pos, R_wb, speed)
    funnel = {"frames": 0, "detected": 0, "associated": 0, "offered": 0, "chi2_accepted": 0}
    for tag in tags:
        for g in range(6):
            cj = HERE / f"logs/char_{tag}_g{g}.json"
            bj = HERE / f"bundles/{tag}_g{g}/frames.json"
            if not cj.exists() or not bj.exists():
                continue
            bundle = {fr["frame_id"]: fr for fr in json.loads(bj.read_text())["frames"]}
            for fr in bundle.values():
                R_wb = F.R_world_from_odo_quat_wxyz(np.asarray(fr["odo_q_wxyz"], float))
                frame_pose[(tag, fr["frame_id"])] = (np.asarray(fr["drone_position_ned"], float),
                                                     R_wb, fr["speed_mps"])
            for r in json.loads(cj.read_text())["rows"]:
                funnel["frames"] += 1
                funnel["detected"] += int(r.get("detected", False))
                if not r.get("associated") or "off_ned" not in r:
                    continue
                funnel["associated"] += 1
                offered = r.get("range_ok", True) and r.get("range_cap_ok", True)
                funnel["offered"] += int(offered)
                accepted = offered and np.isfinite(r.get("maha", np.inf)) and r["maha"] <= CHI2
                funnel["chi2_accepted"] += int(accepted)
                # vision-absolute offset = -off_ned
                off = -np.asarray(r["off_ned"], float)
                gid = r["gate_id"]
                drone = frame_pose.get((tag, r["frame_id"]), (None,))[0]
                # cross-track (lateral) = horizontal offset component perpendicular to the
                # horizontal drone->gate direction
                lateral = along = float("nan")
                if drone is not None and gid in gates_by_id:
                    d_h = (gates_by_id[gid].position_ned - drone)[:2]
                    n = np.linalg.norm(d_h)
                    if n > 1e-6:
                        u = d_h / n                       # along-track unit (horizontal)
                        perp = np.array([-u[1], u[0]])    # cross-track unit (horizontal)
                        along = float(off[:2] @ u)
                        lateral = float(off[:2] @ perp)
                rows.append(dict(tag=tag, gate=gid, range=r["true_range_m"], speed=r["speed_mps"],
                                 score=r["score"], n_corners=r["n_corners"], reproj=r["reproj_px"],
                                 offN=off[0], offE=off[1], offD=off[2],
                                 absfix=r["world_fix_err_m"], lateral=lateral, along=along,
                                 vert=off[2], offered=offered, accepted=accepted,
                                 range_ok=r.get("range_ok", True), maha=r.get("maha", float("nan"))))

    # ---- camera-coverage geometry: per frame, the min-bearing gate that is IN the image ----
    cov = {"frames": 0, "any_gate_in_image": 0, "min_bearing_in_image": []}
    for fid, (drone, R_wb, speed) in frame_pose.items():
        cov["frames"] += 1
        predicted = predict_gates_in_camera(gates, drone, R_wb)
        best = None
        for gid, pg in predicted.items():
            px, py = pg.center_px
            if 0 <= px < W and 0 <= py < Himg:
                t = pg.t_cam_gate
                bearing = float(np.degrees(np.arctan2(np.hypot(t[0], t[1]), t[2])))
                if best is None or bearing < best:
                    best = bearing
        if best is not None:
            cov["any_gate_in_image"] += 1
            cov["min_bearing_in_image"].append(best)

    # ---- report ----
    print(f"\n========= PART B: SHADOW VISION vs ABSOLUTE (pooled: {','.join(tags)}) =========")
    print("FUNNEL (navigator chain on inc7 given-pose at-speed flight):")
    f = funnel
    print(f"  frames {f['frames']}  -> detected {f['detected']} ({100*f['detected']/f['frames']:.0f}%)"
          f"  -> associated {f['associated']} ({100*f['associated']/f['frames']:.0f}%)"
          f"  -> offered(depth+cap) {f['offered']} ({100*f['offered']/f['frames']:.0f}%)"
          f"  -> chi2-accepted {f['chi2_accepted']} ({100*f['chi2_accepted']/f['frames']:.0f}%)")

    print("\nCAMERA COVERAGE (independent of detector): per frame, is ANY gate centre in the image?")
    c = cov
    mb = np.array(c["min_bearing_in_image"]) if c["min_bearing_in_image"] else np.array([])
    print(f"  {c['any_gate_in_image']}/{c['frames']} frames ({100*c['any_gate_in_image']/c['frames']:.0f}%) "
          f"have >=1 gate centre projected inside the 640x360 image")
    if mb.size:
        print(f"  in-image min-bearing of the nearest in-view gate: "
              f"p50 {np.percentile(mb,50):.1f} deg  p90 {np.percentile(mb,90):.1f} deg "
              f"(HFoV half-angle = 45 deg)")

    # per-gate associated table
    print(f"\nVISION-ABSOLUTE OFFSET per associated sighting (sign = vision - absolute):")
    print(f"  {'gate':>4} {'Nfix':>4} {'rng p50':>7} {'spd p50':>7} {'biasN':>6} {'biasE':>6} "
          f"{'biasD':>6} {'sigN':>5} {'sigE':>5} {'sigD':>5} {'|off|p50':>8} {'reproj p50':>10}")
    for g in range(6):
        gr = [r for r in rows if r["gate"] == g]
        if not gr:
            continue
        N = np.array([r["offN"] for r in gr]); E = np.array([r["offE"] for r in gr])
        D = np.array([r["offD"] for r in gr]); af = np.array([r["absfix"] for r in gr])
        rp = np.array([r["reproj"] for r in gr]); rng = np.array([r["range"] for r in gr])
        spd = np.array([r["speed"] for r in gr])
        print(f"  {g:>4} {len(gr):>4} {np.percentile(rng,50):>7.1f} {np.percentile(spd,50):>7.1f} "
              f"{N.mean():>6.2f} {E.mean():>6.2f} {D.mean():>6.2f} "
              f"{N.std():>5.2f} {E.std():>5.2f} {D.std():>5.2f} "
              f"{np.percentile(af,50):>8.2f} {np.percentile(rp,50):>10.2f}")

    # pooled "clean" subset: offered fixes with |off|<3 m (drop frontal flips / wrong-gate)
    clean = [r for r in rows if r["offered"] and r["absfix"] < 3.0]
    print(f"\nCLEAN OFFERED SUBSET (offered & |off|<3 m): N={len(clean)} of {len(rows)} associated")
    if clean:
        for nm, key in [("offset N", "offN"), ("offset E", "offE"), ("offset D (vert)", "offD"),
                        ("LATERAL (cross-track)", "lateral"), ("along-track", "along"),
                        ("|offset|", "absfix"), ("reproj px", "reproj")]:
            s = _stats([r[key] for r in clean if np.isfinite(r[key])])
            if s["n"]:
                print(f"  {nm:>22}: mean {s['mean']:+.3f}  sigma {s['std']:.3f}  "
                      f"p50 {s['p50']:.3f}  p90 {s['p90']:.3f}  (m)")
        lat = _stats([abs(r["lateral"]) for r in clean if np.isfinite(r["lateral"])])
        latsig = _stats([r["lateral"] for r in clean if np.isfinite(r["lateral"])])["std"]
        print(f"\n  >>> MEASURED lateral (cross-track) sigma = {latsig:.3f} m  "
              f"vs MODELED ~0.265 m  ({'consistent' if latsig <= 0.45 else 'WIDER than model'})")
        print(f"  >>> |lateral| p50 {lat['p50']:.3f}  p90 {lat['p90']:.3f} m")

        # scatter vs range (clean subset): lateral / vertical / along-track sigma by range band
        print(f"\n  SCATTER vs RANGE (clean subset, sigma in m):")
        print(f"  {'band':>10} {'N':>4} {'lat_bias':>8} {'lat_sig':>7} {'vert_sig':>8} "
              f"{'along_sig':>9} {'|off|p50':>8}")
        for lab, lo, hi in [("<10m", 0, 10), ("10-20m", 10, 20), ("20-26m", 20, 26),
                            (">26m", 26, 999)]:
            cb = [r for r in clean if lo <= r["range"] < hi]
            if not cb:
                continue
            la = np.array([r["lateral"] for r in cb if np.isfinite(r["lateral"])])
            ve = np.array([r["offD"] for r in cb])
            al = np.array([r["along"] for r in cb if np.isfinite(r["along"])])
            af = np.array([r["absfix"] for r in cb])
            print(f"  {lab:>10} {len(cb):>4} {la.mean():>8.3f} {la.std():>7.3f} {ve.std():>8.3f} "
                  f"{al.std():>9.3f} {np.percentile(af,50):>8.3f}")

    if args.json:
        out = {"tags": tags, "funnel": funnel, "coverage": {k: v for k, v in cov.items()
               if k != "min_bearing_in_image"}, "n_associated": len(rows), "n_clean": len(clean),
               "rows": rows}
        Path(args.json).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
