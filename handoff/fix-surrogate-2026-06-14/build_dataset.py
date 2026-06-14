"""build_dataset.py -- shared substrate for the inc8 FIX-SURROGATE calibration.

Reuses the CANONICAL Track-3 bundle reader (handoff/simops-mastery-2026-06-13/
analyze_shadow_vision.py) and the in-loop projection (racer.vision.association.
predict_gates_in_camera + racer.frames) -- it does NOT reinvent either.

For every recorded frame it emits the GROUND-TRUTH relative geometry to the bundle's
*active* gate g (range, azimuth, elevation, total bearing, in-FoV, viewing angle) --
all FREE in the sim -- joined to the navigator-chain LABELS from the per-gate
characterize_perception dumps (detected / associated / offered / chi2-accepted / maha).
This is exactly the (geometry -> P(accept), sigma) map the analytic fix surrogate fits.

It also re-derives the canonical any-gate camera coverage so the dataset is self-checking
against the published funnel (1821 frames -> 126 accepted; 605/1821 in-image).

Outputs (handoff/fix-surrogate-2026-06-14/data/):
  * geom_frames.json   -- per-frame GT geometry-to-active-gate + labels (accept model + crab map)
  * geom_frames.npz    -- the same as flat numpy arrays (fast load for the sub-model fits)
  * fix_errors.json    -- the pooled associated fix-error rows (lateral/along/vert/range) for sigma(geom)
  * summary.json       -- the reproduced funnel/coverage numbers (calibration anchors)

Usage: <venv>/python handoff/fix-surrogate-2026-06-14/build_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                                          # noqa: E402
from racer.navigator import load_track_map                            # noqa: E402
from racer.vision.association import predict_gates_in_camera          # noqa: E402

SIMOPS = ROOT / "handoff/simops-mastery-2026-06-13"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
POOLED = SIMOPS / "logs/shadow_vision_pooled.json"
OUT = Path(__file__).resolve().parent / "data"
TAGS = ["cr1b", "std1", "std2", "std3", "std4", "std5"]
CHI2 = 16.27                       # navigator vision_gate_chi2 (chi2_0.999, 3 DOF) -- matches analyze_shadow_vision
W, H = F.IMAGE_WIDTH, F.IMAGE_HEIGHT


def _bearing_az_el(t_cam: np.ndarray) -> tuple[float, float, float]:
    """Off-boresight geometry of a camera-frame point (X-right, Y-down, Z-fwd), degrees.

    azimuth = horizontal angle (+ = gate to the right of boresight; the yaw-crab axis),
    elevation = vertical angle (+ = gate below boresight; pitch axis),
    bearing = total off-axis angle (matches characterize_perception's bearing_deg)."""
    tx, ty, tz = float(t_cam[0]), float(t_cam[1]), float(t_cam[2])
    az = float(np.degrees(np.arctan2(tx, tz)))
    el = float(np.degrees(np.arctan2(ty, tz)))
    bearing = float(np.degrees(np.arctan2(np.hypot(tx, ty), tz)))
    return bearing, az, el


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gates = load_track_map(str(MAP), corner_to_center=True)
    gates_by_id = {g.gate_id: g for g in gates}
    R_cb = F.R_camera_from_body()                                     # camera <- body (fixed mount)

    records: list[dict] = []
    cands: list[dict] = []          # per-(frame, gate) candidate rows (accept model + crab map)
    funnel = {"frames": 0, "detected": 0, "associated": 0, "offered": 0, "accepted": 0}
    cov = {"frames": 0, "any_gate_in_image": 0}
    missing_pose = 0
    frame_counter = 0

    for tag in TAGS:
        for g in range(6):
            cj = SIMOPS / f"logs/char_{tag}_g{g}.json"
            bj = SIMOPS / f"bundles/{tag}_g{g}/frames.json"
            if not cj.exists() or not bj.exists():
                continue
            bundle = {fr["frame_id"]: fr for fr in json.loads(bj.read_text())["frames"]}
            char_rows = {r["frame_id"]: r for r in json.loads(cj.read_text())["rows"]}
            gate_g = gates_by_id[g]
            normal_g = np.asarray(gate_g.R_world_gate, float)[:, 2]    # gate "through" axis, world NED
            for fid, r in char_rows.items():
                funnel["frames"] += 1
                detected = bool(r.get("detected", False))
                associated = bool(r.get("associated", False)) and "off_ned" in r
                funnel["detected"] += int(detected)
                offered = accepted = False
                if associated:
                    funnel["associated"] += 1
                    offered = bool(r.get("range_ok", True)) and bool(r.get("range_cap_ok", True))
                    funnel["offered"] += int(offered)
                    accepted = offered and np.isfinite(r.get("maha", np.inf)) and r["maha"] <= CHI2
                    funnel["accepted"] += int(accepted)
                fr = bundle.get(fid)
                if fr is None:
                    missing_pose += 1
                    continue
                drone = np.asarray(fr["drone_position_ned"], float)
                R_wb = F.R_world_from_odo_quat_wxyz(np.asarray(fr["odo_q_wxyz"], float))
                R_wc = R_wb @ R_cb.T                                   # world <- camera
                # --- true crab: horizontal angle between velocity and body-forward (camera azimuth) ---
                vel = np.asarray(fr.get("lpn_vel", [float("nan")] * 3), float)
                vh = float(np.hypot(vel[0], vel[1]))
                fwd = R_wb @ np.array([1.0, 0.0, 0.0])                 # body-forward in world NED
                head_body = float(np.arctan2(fwd[1], fwd[0]))
                head_vel = float(np.arctan2(vel[1], vel[0])) if vh > 1e-6 else head_body
                crab = float(np.degrees((head_body - head_vel + np.pi) % (2 * np.pi) - np.pi))
                assoc_gate_id = int(r["gate_id"]) if associated else -1
                # --- GT geometry to the ACTIVE gate g (all free in sim) ---
                L = gate_g.position_ned - drone                       # +L lever, world NED
                rng = float(np.linalg.norm(L))
                t_cam = R_wc.T @ L                                     # gate origin in camera frame
                bearing, az, el = _bearing_az_el(t_cam)
                proj = F.project_camera_point(t_cam)
                in_fov = bool(proj is not None and 0.0 <= proj[0] < W and 0.0 <= proj[1] < H)
                px, py = (proj if proj is not None else (float("nan"), float("nan")))
                if rng > 1e-6:
                    cosv = abs(float((L / rng) @ normal_g))
                    view_ang = float(np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0))))
                else:
                    view_ang = float("nan")
                # --- canonical any-gate camera coverage (matches analyze_shadow_vision) ---
                cov["frames"] += 1
                predicted = predict_gates_in_camera(gates, drone, R_wb)
                best_b = None
                for gid, pg in predicted.items():
                    qx, qy = pg.center_px
                    if 0 <= qx < W and 0 <= qy < H:
                        tb, _, _ = _bearing_az_el(pg.t_cam_gate)
                        best_b = tb if best_b is None else min(best_b, tb)
                any_in_image = best_b is not None
                cov["any_gate_in_image"] += int(any_in_image)
                records.append(dict(
                    tag=tag, active_gate=g, frame_id=int(fid), frame_idx=frame_counter,
                    speed=float(fr.get("speed_mps", r.get("speed_mps", float("nan")))), crab_deg=crab,
                    range_g=rng, azimuth_g=az, elevation_g=el, bearing_g=bearing,
                    in_fov_g=in_fov, px_g=float(px), py_g=float(py), view_angle_g=view_ang,
                    tz_g=float(t_cam[2]),
                    detected=detected, associated=associated, assoc_gate=assoc_gate_id,
                    offered=offered, accepted=bool(accepted),
                    accepted_to_g=bool(accepted and assoc_gate_id == g),
                    maha=float(r.get("maha", float("nan"))),
                    any_gate_in_image=any_in_image,
                    min_bearing_in_image=(float(best_b) if best_b is not None else float("nan")),
                ))
                # --- per-(frame, gate) CANDIDATE rows: geometry to EVERY gate + accept-to-that-gate.
                # Accepts attach to whichever gate the detector locked, not the active one, so the
                # accept model + crab sweep are fit over candidates, not just the active gate. We store
                # t_cam so the crab map can re-yaw the camera (rotate about cam-Y) and re-project.
                for k, gk in gates_by_id.items():
                    Lk = gk.position_ned - drone
                    rk = float(np.linalg.norm(Lk))
                    tck = R_wc.T @ Lk
                    bk, ak, ek = _bearing_az_el(tck)
                    pk = F.project_camera_point(tck)
                    ink = bool(pk is not None and 0.0 <= pk[0] < W and 0.0 <= pk[1] < H)
                    nrmk = np.asarray(gk.R_world_gate, float)[:, 2]
                    vk = (float(np.degrees(np.arccos(np.clip(abs(float((Lk / rk) @ nrmk)), 0.0, 1.0))))
                          if rk > 1e-6 else float("nan"))
                    cands.append(dict(
                        frame_idx=frame_counter, tag=tag, active_gate=g, gate=k,
                        range=rk, azimuth=ak, elevation=ek, bearing=bk, view_angle=vk,
                        tcx=float(tck[0]), tcy=float(tck[1]), tcz=float(tck[2]),
                        in_image=ink, crab_deg=crab,
                        accepted_to_k=bool(accepted and assoc_gate_id == k),
                        offered_to_k=bool(offered and assoc_gate_id == k),
                    ))
                frame_counter += 1

    # ---- pooled fix-error rows for sigma(geometry) (canonical decomposition, reused verbatim) ----
    pooled = json.loads(POOLED.read_text())
    fix_rows = [dict(tag=r["tag"], gate=r["gate"], range=r["range"], speed=r["speed"],
                     lateral=r.get("lateral"), along=r.get("along"), vert=r.get("offD"),
                     offN=r["offN"], offE=r["offE"], offD=r["offD"], absfix=r["absfix"],
                     reproj=r["reproj"], n_corners=r["n_corners"],
                     offered=r["offered"], accepted=r["accepted"])
                for r in pooled["rows"]]

    # ---- persist ----
    (OUT / "geom_frames.json").write_text(json.dumps(records, indent=1, default=float))
    (OUT / "fix_errors.json").write_text(json.dumps(fix_rows, indent=1, default=float))
    (OUT / "cand_frames.json").write_text(json.dumps(cands, indent=0, default=float))
    # flat npz (per-frame, active-gate)
    keys_f = ["range_g", "azimuth_g", "elevation_g", "bearing_g", "view_angle_g", "tz_g",
              "speed", "crab_deg", "maha", "px_g", "py_g", "min_bearing_in_image"]
    keys_b = ["in_fov_g", "detected", "associated", "offered", "accepted", "accepted_to_g",
              "any_gate_in_image"]
    arrs = {k: np.array([rec[k] for rec in records], float) for k in keys_f}
    arrs.update({k: np.array([rec[k] for rec in records], bool) for k in keys_b})
    arrs["active_gate"] = np.array([rec["active_gate"] for rec in records], int)
    arrs["assoc_gate"] = np.array([rec["assoc_gate"] for rec in records], int)
    arrs["frame_idx"] = np.array([rec["frame_idx"] for rec in records], int)
    arrs["tag"] = np.array([rec["tag"] for rec in records])
    np.savez(OUT / "geom_frames.npz", **arrs)
    # flat npz (per-(frame,gate) candidates)
    ck_f = ["range", "azimuth", "elevation", "bearing", "view_angle", "tcx", "tcy", "tcz", "crab_deg"]
    ck_b = ["in_image", "accepted_to_k", "offered_to_k"]
    ck_i = ["frame_idx", "active_gate", "gate"]
    carrs = {k: np.array([c[k] for c in cands], float) for k in ck_f}
    carrs.update({k: np.array([c[k] for c in cands], bool) for k in ck_b})
    carrs.update({k: np.array([c[k] for c in cands], int) for k in ck_i})
    carrs["tag"] = np.array([c["tag"] for c in cands])
    np.savez(OUT / "cand_frames.npz", **carrs)

    crab_arr = arrs["crab_deg"][np.isfinite(arrs["crab_deg"])]
    cand_in = carrs["in_image"]
    cand_acc = carrs["accepted_to_k"]
    summary = dict(
        n_records=len(records), n_candidates=len(cands), missing_pose=missing_pose,
        funnel=funnel, coverage=cov,
        accept_rate=funnel["accepted"] / funnel["frames"] if funnel["frames"] else float("nan"),
        in_fov_any_frac=cov["any_gate_in_image"] / cov["frames"] if cov["frames"] else float("nan"),
        n_fix_rows=len(fix_rows),
        crab_deg_p50=float(np.percentile(np.abs(crab_arr), 50)) if crab_arr.size else float("nan"),
        crab_deg_p90=float(np.percentile(np.abs(crab_arr), 90)) if crab_arr.size else float("nan"),
        crab_deg_mean_signed=float(np.mean(crab_arr)) if crab_arr.size else float("nan"),
        p_accept_given_in_image=float(cand_acc[cand_in].mean()) if cand_in.any() else float("nan"),
        n_in_image_cand=int(cand_in.sum()),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- self-check print ----
    print("=== build_dataset summary ===")
    print(f"records={len(records)}  missing_pose={missing_pose}")
    print(f"FUNNEL  frames {funnel['frames']} -> detected {funnel['detected']} "
          f"-> associated {funnel['associated']} -> offered {funnel['offered']} "
          f"-> accepted {funnel['accepted']}  (accept-rate {summary['accept_rate']:.3f})")
    print(f"COVERAGE any-gate-in-image {cov['any_gate_in_image']}/{cov['frames']} "
          f"= {summary['in_fov_any_frac']:.3f}")
    # active-gate in-FoV fraction (the camera-pointing lever)
    inf = arrs["in_fov_g"]
    acc = arrs["accepted"]
    accg = arrs["accepted_to_g"]
    print(f"ACTIVE-GATE in_fov_g frac = {inf.mean():.3f}  ({int(inf.sum())}/{len(inf)})")
    print(f"P(accepted | in_fov_g)      = {acc[inf].mean():.3f}  ({int(acc[inf].sum())}/{int(inf.sum())})")
    print(f"P(accepted | ~in_fov_g)     = {acc[~inf].mean():.3f}  ({int(acc[~inf].sum())}/{int((~inf).sum())})")
    print(f"P(accepted_to_g | in_fov_g) = {accg[inf].mean():.3f}  ({int(accg[inf].sum())}/{int(inf.sum())})")
    print(f"P(accepted | any_in_image)  = {acc[arrs['any_gate_in_image']].mean():.3f}")
    print(f"CANDIDATES n={len(cands)}  in_image={int(cand_in.sum())}  "
          f"P(accept_to_k | in_image_k) = {summary['p_accept_given_in_image']:.3f}  "
          f"({int(cand_acc[cand_in].sum())}/{int(cand_in.sum())})")
    print(f"CRAB (deg, signed): mean {summary['crab_deg_mean_signed']:+.1f}  "
          f"|crab| p50 {summary['crab_deg_p50']:.1f}  p90 {summary['crab_deg_p90']:.1f}")
    print(f"active-gate bearing_g (deg): p50 {np.percentile(arrs['bearing_g'],50):.1f} "
          f"p90 {np.percentile(arrs['bearing_g'],90):.1f}")
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
