"""shadow_allgates.py — multi-gate shadow replay in ONE detection pass per frame.

Same production C2 chain as shadow_gate4.py (GateDetector -> associate_scored -> estimate_gate_pose ->
gate_pose_to_world_position + gate_relative_inplane_fix -> 2-DOF relinnov gate, GT-anchored prior),
but runs the detector ONCE per frame and scores EVERY requested gate from that single detect() so a
speed-ramp lap (each gate crossed at a different speed) yields per-(speed,range,gate) fix rows without
re-detecting per gate. Emits the same per-frame/per-gate rows shadow_gate4 emits (speed_mps,
true_range_m, g4_bearing_deg, associated/offered/accepted, rel_cross/rel_vert/rel_along, abs_*,
d2_rel, nav_*_sigma) -> feed analyze_atspeed.py.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe handoff/at-speed-sigma-2026-06-15/analysis/shadow_allgates.py \
      --glob 'data/runs/*_atspd_s2_*' --gates 1,2,3,4,5 \
      --json handoff/at-speed-sigma-2026-06-15/analysis/s2_allgates_rows.json
"""
from __future__ import annotations

import argparse
import bisect
import glob as globmod
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from racer import frames as F
from racer.contracts import Frame, GatePose
from racer.localization import (P3P_FIX_COV_INFLATION, gate_pose_to_world_position,
                                gate_relative_inplane_fix)
from racer.navigator import NavigatorConfig, load_track_map
from racer.recording import RecordingReader
from racer.vision.association import associate_scored, predict_gates_in_camera, range_consistent
from racer.vision.detector import GateDetector
from racer.vision.gate_pose import estimate_gate_pose

WEIGHTS = ROOT / "models/gate_yolo11s_curriculum_v2.pt"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
W_IMG, H_IMG = 640, 360


def _nearest(keys, items, k):
    if not keys:
        return None
    i = bisect.bisect_left(keys, k)
    cands = [j for j in (i, i - 1) if 0 <= j < len(keys)]
    b = min(cands, key=lambda j: abs(keys[j] - k))
    return items[b]


def aligned_frames(reader):
    meta = reader.meta
    t0u, t0m = int(meta["t0_unix_ns"]), int(meta["t0_monotonic_ns"])
    lpn, odo = [], []
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            lpn.append((float(msg._timestamp), [float(msg.x), float(msg.y), float(msg.z)],
                        [float(msg.vx), float(msg.vy), float(msg.vz)]))
        elif t == "ODOMETRY":
            odo.append((float(msg._timestamp), [float(v) for v in msg.q]))
    lpn.sort(key=lambda r: r[0]); odo.sort(key=lambda r: r[0])
    lt = [r[0] for r in lpn]; ot = [r[0] for r in odo]
    out, seen = [], set()
    for e in reader.iter_video_index():
        fid = e["frame_id"]
        if fid in seen:
            continue
        seen.add(fid)
        recv = (t0u + (e["recv_monotonic_ns"] - t0m)) / 1e9
        lp = _nearest(lt, lpn, recv); od = _nearest(ot, odo, recv)
        if lp is None or od is None:
            continue
        out.append({"frame_id": fid, "sim_time_ns": int(e["sim_time_ns"]),
                    "offset": e["offset"], "length": e["length"],
                    "pos": np.array(lp[1]), "vel": np.array(lp[2]), "q": np.array(od[1])})
    return out


def crab_deg(vel, R_wb):
    vh = vel[:2]
    if np.linalg.norm(vh) < 0.5:
        return float("nan")
    fwd = (R_wb @ np.array([1.0, 0.0, 0.0]))[:2]
    if np.linalg.norm(fwd) < 1e-6:
        return float("nan")
    vh = vh / np.linalg.norm(vh); fwd = fwd / np.linalg.norm(fwd)
    return float(np.degrees(np.arccos(np.clip(vh @ fwd, -1.0, 1.0))))


def in_image(c, t):
    return bool(t[2] > 0 and 0 <= c[0] < W_IMG and 0 <= c[1] < H_IMG)


def process_session(sess, det, gates, gate_ids, cfg):
    reader = RecordingReader(sess)
    meta = reader.meta
    gbi = {g.gate_id: g for g in gates}
    geom = {}
    for gid in gate_ids:
        g = gbi[gid]
        Rg2w = np.asarray(g.R_world_gate, dtype=np.float64)
        geom[gid] = (g, Rg2w, Rg2w[:, :2].T)
    P_prior = (cfg.given_pos_std ** 2) * np.eye(3)
    chi2 = cfg.gate_rel_chi2
    fr_list = aligned_frames(reader)
    vidf = open(sess / "video.bin", "rb")
    rows = []
    for fr in fr_list:
        drone, vel, q = fr["pos"], fr["vel"], fr["q"]
        R_wb = F.R_world_from_odo_quat_wxyz(q)
        speed = float(np.linalg.norm(vel))
        cr = crab_deg(vel, R_wb)
        predicted = predict_gates_in_camera(gates, drone, R_wb)
        vidf.seek(fr["offset"]); jpeg = vidf.read(fr["length"])
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        frame = Frame(frame_id=fr["frame_id"], sim_time_ns=fr["sim_time_ns"], image_bgr=img,
                      recv_monotonic_ns=0, jpeg_bytes=None)
        obs_list = det.detect(frame)
        for gid in gate_ids:
            g, Rg2w, B = geom[gid]
            pg = predicted.get(gid)
            in_fov = bool(pg is not None and in_image(pg.center_px, pg.t_cam_gate))
            bearing = (float(np.degrees(np.arctan2(np.hypot(pg.t_cam_gate[0], pg.t_cam_gate[1]),
                                                   pg.t_cam_gate[2]))) if pg is not None else float("nan"))
            true_rng = float(np.linalg.norm(g.position_ned - drone))
            best = None
            for o in obs_list:
                sc = associate_scored(o, predicted)
                if sc is None or sc[0] != gid:
                    continue
                if best is None or sc[1] < best[0]:
                    best = (sc[1], o)
            row = dict(session=sess.name, frame_id=fr["frame_id"], gate=gid, speed_mps=round(speed, 3),
                       true_range_m=round(true_rng, 3), g4_bearing_deg=round(bearing, 2),
                       g4_in_fov=in_fov, crab_deg=(None if np.isnan(cr) else round(cr, 2)),
                       n_det=len(obs_list), associated=False)
            if best is not None and pg is not None:
                _, o = best
                prior = GatePose(o.frame_id, o.sim_time_ns, pg.R_cam_gate, pg.t_cam_gate, 0.0, gate_id=gid)
                pose = estimate_gate_pose(o, prior=prior, compute_covariance=True)
                if pose is not None:
                    row["associated"] = True
                    range_cap_ok = bool(pose.range_m <= cfg.vision_max_range_m)
                    range_ok = bool(range_consistent(pose.range_m, pg.range_m, cfg.fix_range_rel_tol,
                                                     cfg.fix_range_abs_tol_m))
                    offered = range_cap_ok and range_ok
                    pos_fix, cov_abs = gate_pose_to_world_position(
                        pose, g, R_wb, attitude_noise_std=cfg.attitude_noise_std,
                        fix_cov_floor_std=cfg.fix_cov_floor_std)
                    if pose.n_corners < 4:
                        cov_abs = cov_abs * P3P_FIX_COV_INFLATION
                    z_rel, cov_rel = gate_relative_inplane_fix(
                        pose, g, R_wb, inplane_sigma=cfg.gate_rel_inplane_sigma,
                        range_growth_a1=cfg.gate_rel_range_growth_a1, along_sigma=cfg.gate_rel_along_sigma,
                        attitude_noise_std=cfg.attitude_noise_std, fix_cov_floor_std=cfg.fix_cov_floor_std)
                    err_abs = pos_fix - drone
                    err_rel = z_rel - drone
                    nu_ip = B @ err_rel
                    S_ip = B @ (P_prior + cov_rel) @ B.T
                    try:
                        d2_rel = float(nu_ip @ np.linalg.solve(S_ip, nu_ip))
                    except np.linalg.LinAlgError:
                        d2_rel = 0.0
                    accepted = bool(offered and d2_rel <= chi2)
                    S = P_prior + cov_rel
                    K = P_prior @ np.linalg.solve(S, np.eye(3))
                    P_post = (np.eye(3) - K) @ P_prior
                    Pg = Rg2w.T @ P_post @ Rg2w
                    row.update(
                        n_corners=int(pose.n_corners), reproj_px=float(pose.reproj_error_px),
                        pose_range_m=float(pose.range_m), score=float(o.score),
                        offered=offered, accepted=accepted, d2_rel=d2_rel,
                        abs_cross=float(Rg2w[:, 0] @ err_abs), abs_vert=float(Rg2w[:, 1] @ err_abs),
                        abs_along=float(Rg2w[:, 2] @ err_abs),
                        rel_cross=float(Rg2w[:, 0] @ err_rel), rel_vert=float(Rg2w[:, 1] @ err_rel),
                        rel_along=float(Rg2w[:, 2] @ err_rel),
                        rel_inplane=float(np.hypot(Rg2w[:, 0] @ err_rel, Rg2w[:, 1] @ err_rel)),
                        nav_inplane_sigma=float(np.sqrt(max(Pg[0, 0] + Pg[1, 1], 0.0))),
                        nav_along_sigma=float(np.sqrt(max(Pg[2, 2], 0.0))),
                        world_fix_err_m=float(np.linalg.norm(err_abs)))
            rows.append(row)
    vidf.close()
    return {"session": sess.name, "final_state": meta.get("final_state"),
            "collisions": meta.get("collisions"), "contact": bool((meta.get("collisions") or 0) > 0),
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--gates", default="1,2,3,4,5")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--map", default=str(MAP))
    ap.add_argument("--json", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    gate_ids = [int(x) for x in args.gates.split(",")]
    sessions = sorted({Path(p) for pat in args.glob for p in globmod.glob(pat)})
    if args.limit:
        sessions = sessions[:args.limit]
    if not sessions:
        print("no sessions"); return 2
    cfg = NavigatorConfig()
    gates = load_track_map(args.map, corner_to_center=True)
    det = GateDetector.load(Path(args.weights), score_thresh=0.25, kpt_conf_thresh=0.5)
    print(f"allgates shadow · {len(sessions)} laps · gates {gate_ids}")
    all_rows, per = [], []
    for s in sessions:
        t0 = time.time()
        r = process_session(s, det, gates, gate_ids, cfg)
        per.append({k: v for k, v in r.items() if k != "rows"})
        all_rows += r["rows"]
        acc = sum(1 for x in r["rows"] if x.get("accepted"))
        print(f"  {s.name}: {r['final_state']} coll={r['collisions']} rows={len(r['rows'])} "
              f"accepted={acc} ({time.time()-t0:.0f}s)")
    Path(args.json).write_text(json.dumps({"sessions": per, "rows": all_rows}, indent=2, default=float))
    print(f"wrote {len(all_rows)} rows -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
