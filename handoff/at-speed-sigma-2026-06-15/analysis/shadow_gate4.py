"""shadow_gate4.py — L3 AT-SPEED SHADOW analysis (offline, GT-anchored).

Replays the PRODUCTION C2 vision/estimator chain in SHADOW (logged, NOT driving control) over the
recorded inc7 at-speed laps, focused on GATE-4 (the binding band), and pins against ground truth:

  (i)  the EFFECTIVE attitude/accel bias at gate-4 = (vision fix − GT) projected into the gate-4 plane
       (the residual in-plane bias the gate-relative +L fix CANNOT cancel — the margin-binding factor);
  (ii) the gate-relative +L fix ACCEPT-RATE + in-plane sigma at the binding band (the relinnov gate);
  (iii) the crab ↔ gate-in-FoV ↔ fix-rate correlation.

WHY GT-anchored: in case-A the navigator holds the given (pristine) position tight (given_pos_std), so
the KF prior at vision time IS ≈ GT. We therefore anchor the relinnov prior at x_KF=GT, P=given_pos_std²·I
— which makes the in-plane innovation nu_ip = B@(z_rel−GT) the fix's TRUE in-plane error vs truth (exactly
the quantity the margin needs), and the accept decision is dominated by cov_rel (≥0.265² in-plane ≫ 0.05²),
so the prior approximation is immaterial to accept. The bias measurement (i) is KF-free (raw fix vs GT).

Chain reused verbatim from production (no re-implementation): GateDetector → associate_scored →
estimate_gate_pose (IPPE/P3P + refine) → gate_pose_to_world_position (absolute fix, KEPT) +
gate_relative_inplane_fix (+L in-plane pseudo-fix) → the navigator's 2-DOF in-plane relinnov gate
(chi2(2,0.999)=13.82). Attitude R_wb = R_world_from_odo_quat_wxyz (R_y(pi)-conjugated, the in-loop frame).
Gates from the surveyed map with corner_to_center=True (perfect fix returns GT → residual IS the error).

Per-frame pose alignment: nearest LOCAL_POSITION_NED (GT pos+vel) + ODOMETRY (attitude quat) by RECV clock
(video recv bridged unix via meta t0; tlog stamps are unix seconds) — identical to export_frame_bundle.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe handoff/l3-atspeed-recording-2026-06-14/analysis/shadow_gate4.py \
      --glob 'data/runs/*_l3atspeed_*_f1' --also 'data/runs/*_l3smoke_*_f1' \
      --json handoff/l3-atspeed-recording-2026-06-14/analysis/shadow_gate4_rows.json
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
from racer.localization import (
    P3P_FIX_COV_INFLATION,
    gate_pose_to_world_position,
    gate_relative_inplane_fix,
)
from racer.navigator import GATE_REL_CHI2_2_999, NavigatorConfig, load_track_map
from racer.recording import RecordingReader
from racer.vision.association import associate_scored, predict_gates_in_camera, range_consistent
from racer.vision.detector import GateDetector
from racer.vision.gate_pose import estimate_gate_pose

WEIGHTS = ROOT / "models/gate_yolo11s_curriculum_v2.pt"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
W_IMG, H_IMG = 640, 360


# --------------------------------------------------------------------------- alignment
def _nearest(keys: list[float], items: list, k: float):
    if not keys:
        return None, float("inf")
    i = bisect.bisect_left(keys, k)
    cands = [j for j in (i, i - 1) if 0 <= j < len(keys)]
    b = min(cands, key=lambda j: abs(keys[j] - k))
    return items[b], abs(keys[b] - k)


def aligned_frames(reader: RecordingReader) -> list[dict]:
    """Per video frame (dedup by frame_id): nearest LPN (GT pos+vel) + ODO (quat) by recv clock."""
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
    lpn_t = [r[0] for r in lpn]; odo_t = [r[0] for r in odo]
    out, seen = [], set()
    for e in reader.iter_video_index():
        fid = e["frame_id"]
        if fid in seen:
            continue
        seen.add(fid)
        recv = (t0u + (e["recv_monotonic_ns"] - t0m)) / 1e9
        lp, lg = _nearest(lpn_t, lpn, recv)
        od, og = _nearest(odo_t, odo, recv)
        if lp is None or od is None:
            continue
        out.append({"frame_id": fid, "sim_time_ns": int(e["sim_time_ns"]),
                    "offset": e["offset"], "length": e["length"],
                    "pos": np.array(lp[1]), "vel": np.array(lp[2]), "q": np.array(od[1]),
                    "lpn_gap_ms": round(lg * 1000, 2), "odo_gap_ms": round(og * 1000, 2)})
    return out


# --------------------------------------------------------------------------- geometry
def crab_deg(vel: np.ndarray, R_wb: np.ndarray) -> float:
    """Horizontal sideslip: angle between GT velocity heading and the body-forward (camera-bore) axis."""
    vh = vel[:2]
    if np.linalg.norm(vh) < 0.5:
        return float("nan")
    fwd = (R_wb @ np.array([1.0, 0.0, 0.0]))[:2]
    if np.linalg.norm(fwd) < 1e-6:
        return float("nan")
    vh = vh / np.linalg.norm(vh); fwd = fwd / np.linalg.norm(fwd)
    return float(np.degrees(np.arccos(np.clip(vh @ fwd, -1.0, 1.0))))


def in_image(center_px, t_cam) -> bool:
    return bool(t_cam[2] > 0 and 0 <= center_px[0] < W_IMG and 0 <= center_px[1] < H_IMG)


# --------------------------------------------------------------------------- per-frame shadow chain
def process_session(sess: Path, det: GateDetector, gates, gate_id: int, cfg: NavigatorConfig) -> dict:
    reader = RecordingReader(sess)
    meta = reader.meta
    gates_by_id = {g.gate_id: g for g in gates}
    g4 = gates_by_id[gate_id]
    R_g2w = np.asarray(g4.R_world_gate, dtype=np.float64)     # cols: 0=cross(right) 1=down 2=along(normal)
    B = R_g2w[:, :2].T                                        # (2,3) in-plane projector
    P_prior = (cfg.given_pos_std ** 2) * np.eye(3)            # case-A anchored prior cov
    chi2 = cfg.gate_rel_chi2

    fr_list = aligned_frames(reader)
    vidf = open(sess / "video.bin", "rb")
    rows = []
    funnel = dict(frames=0, g4_in_fov=0, any_in_fov=0, detected=0, g4_assoc=0, g4_offered=0,
                  g4_rel_accepted=0)
    for fr in fr_list:
        funnel["frames"] += 1
        drone, vel, q = fr["pos"], fr["vel"], fr["q"]
        R_wb = F.R_world_from_odo_quat_wxyz(q)
        speed = float(np.linalg.norm(vel))
        predicted = predict_gates_in_camera(gates, drone, R_wb)
        # --- coverage geometry (detector-independent) ---
        pg4 = predicted.get(gate_id)
        g4_in_fov = bool(pg4 is not None and in_image(pg4.center_px, pg4.t_cam_gate))
        g4_bearing = (float(np.degrees(np.arctan2(np.hypot(pg4.t_cam_gate[0], pg4.t_cam_gate[1]),
                                                  pg4.t_cam_gate[2]))) if pg4 is not None else float("nan"))
        true_rng4 = float(np.linalg.norm(g4.position_ned - drone))
        any_fov = any(in_image(pg.center_px, pg.t_cam_gate) for pg in predicted.values())
        funnel["g4_in_fov"] += int(g4_in_fov); funnel["any_in_fov"] += int(any_fov)
        cr = crab_deg(vel, R_wb)

        vidf.seek(fr["offset"]); jpeg = vidf.read(fr["length"])
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        frame = Frame(frame_id=fr["frame_id"], sim_time_ns=fr["sim_time_ns"], image_bgr=img,
                      recv_monotonic_ns=0, jpeg_bytes=None)
        obs_list = det.detect(frame)
        funnel["detected"] += int(len(obs_list) > 0)

        # keep the best-agreeing detection that associates to gate_id (navigator processes each obs;
        # multiple obs->same gate in one frame is rare, so one gate-4 sighting per frame).
        best = None
        for o in obs_list:
            scored = associate_scored(o, predicted)
            if scored is None or scored[0] != gate_id:
                continue
            if best is None or scored[1] < best[0]:
                best = (scored[1], o)
        row = dict(session=sess.name, frame_id=fr["frame_id"], speed_mps=round(speed, 3),
                   true_range_m=round(true_rng4, 3), g4_bearing_deg=round(g4_bearing, 2),
                   g4_in_fov=g4_in_fov, any_in_fov=any_fov, crab_deg=(None if np.isnan(cr) else round(cr, 2)),
                   n_det=len(obs_list), associated=False)
        if best is not None:
            funnel["g4_assoc"] += 1
            row["associated"] = True
            _, o = best
            prior = GatePose(o.frame_id, o.sim_time_ns, pg4.R_cam_gate, pg4.t_cam_gate, 0.0, gate_id=gate_id)
            pose = estimate_gate_pose(o, prior=prior, compute_covariance=True)
            if pose is not None:
                range_cap_ok = bool(pose.range_m <= cfg.vision_max_range_m)
                range_ok = bool(range_consistent(pose.range_m, pg4.range_m, cfg.fix_range_rel_tol,
                                                 cfg.fix_range_abs_tol_m))
                offered = range_cap_ok and range_ok
                # absolute fix (KEPT) + gate-relative +L in-plane fix (the C2 augment)
                pos_fix, cov_abs = gate_pose_to_world_position(
                    pose, g4, R_wb, attitude_noise_std=cfg.attitude_noise_std,
                    fix_cov_floor_std=cfg.fix_cov_floor_std)
                if pose.n_corners < 4:
                    cov_abs = cov_abs * P3P_FIX_COV_INFLATION
                z_rel, cov_rel = gate_relative_inplane_fix(
                    pose, g4, R_wb, inplane_sigma=cfg.gate_rel_inplane_sigma,
                    range_growth_a1=cfg.gate_rel_range_growth_a1, along_sigma=cfg.gate_rel_along_sigma,
                    attitude_noise_std=cfg.attitude_noise_std, fix_cov_floor_std=cfg.fix_cov_floor_std)
                # project errors into the gate-4 plane: cross(right), vert(down), along(normal)
                err_abs = pos_fix - drone
                err_rel = z_rel - drone
                abs_cross, abs_vert, abs_along = (float(R_g2w[:, 0] @ err_abs),
                                                  float(R_g2w[:, 1] @ err_abs), float(R_g2w[:, 2] @ err_abs))
                rel_cross, rel_vert, rel_along = (float(R_g2w[:, 0] @ err_rel),
                                                  float(R_g2w[:, 1] @ err_rel), float(R_g2w[:, 2] @ err_rel))
                # relinnov gate (anchored prior x_KF=GT): nu_ip = B(z_rel-GT); S_ip = B(P+cov_rel)B^T
                nu_ip = B @ err_rel
                S_ip = B @ (P_prior + cov_rel) @ B.T
                try:
                    d2_rel = float(nu_ip @ np.linalg.solve(S_ip, nu_ip))
                except np.linalg.LinAlgError:
                    d2_rel = 0.0
                accepted = bool(offered and d2_rel <= chi2)
                # posterior gate-frame cov export (mirror navigator._gate_frame_pos_sigma): apply the
                # in-plane fix to the anchored prior, project P_post into the gate-4 plane.
                S = P_prior + cov_rel
                K = P_prior @ np.linalg.solve(S, np.eye(3))
                P_post = (np.eye(3) - K) @ P_prior
                Pg = R_g2w.T @ P_post @ R_g2w
                nav_inplane_sigma = float(np.sqrt(max(Pg[0, 0] + Pg[1, 1], 0.0)))
                nav_along_sigma = float(np.sqrt(max(Pg[2, 2], 0.0)))
                funnel["g4_offered"] += int(offered)
                funnel["g4_rel_accepted"] += int(accepted)
                row.update(
                    n_corners=int(pose.n_corners), reproj_px=float(pose.reproj_error_px),
                    pose_range_m=float(pose.range_m), score=float(o.score),
                    range_ok=range_ok, range_cap_ok=range_cap_ok, offered=offered,
                    abs_cross=abs_cross, abs_vert=abs_vert, abs_along=abs_along,
                    abs_inplane=float(np.hypot(abs_cross, abs_vert)),
                    rel_cross=rel_cross, rel_vert=rel_vert, rel_along=rel_along,
                    rel_inplane=float(np.hypot(rel_cross, rel_vert)),
                    d2_rel=d2_rel, accepted=accepted,
                    nav_inplane_sigma=nav_inplane_sigma, nav_along_sigma=nav_along_sigma,
                    world_fix_err_m=float(np.linalg.norm(err_abs)))
        rows.append(row)
    vidf.close()
    return {"session": sess.name, "final_state": meta.get("final_state"),
            "collisions": meta.get("collisions"), "gate_index": meta.get("gate_index"),
            "contact": bool((meta.get("collisions") or 0) > 0), "funnel": funnel, "rows": rows}


# --------------------------------------------------------------------------- aggregate / report
def _st(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"), p50=float("nan"), p90=float("nan"))
    return dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
                p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="data/runs/*_l3atspeed_*_f1")
    ap.add_argument("--also", default=None, help="extra glob (e.g. the smoke lap) pooled in")
    ap.add_argument("--gate", type=int, default=4)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--map", default=str(MAP))
    ap.add_argument("--json", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap sessions (debug)")
    args = ap.parse_args()

    sessions = sorted({Path(p) for g in [args.glob, args.also] if g for p in globmod.glob(g)})
    if args.limit:
        sessions = sessions[:args.limit]
    if not sessions:
        print("no sessions matched", file=sys.stderr); return 2
    cfg = NavigatorConfig()
    gates = load_track_map(args.map, corner_to_center=True)
    det = GateDetector.load(Path(args.weights), score_thresh=0.25, kpt_conf_thresh=0.5)
    print(f"gate-{args.gate} shadow analysis · {len(sessions)} sessions · weights={Path(args.weights).name}")
    print(f"constants: given_pos_std={cfg.given_pos_std} inplane_sigma={cfg.gate_rel_inplane_sigma} "
          f"along_sigma={cfg.gate_rel_along_sigma} chi2(2,.999)={cfg.gate_rel_chi2:.3f} "
          f"max_range={cfg.vision_max_range_m} attitude_noise={np.degrees(cfg.attitude_noise_std):.2f}deg")

    per = []
    for s in sessions:
        t0 = time.time()
        r = process_session(s, det, gates, args.gate, cfg)
        per.append(r)
        f = r["funnel"]
        print(f"  {s.name}: {r['final_state']} coll={r['collisions']} "
              f"{'CONTACT' if r['contact'] else 'clean'} | frames {f['frames']} g4FoV {f['g4_in_fov']} "
              f"det {f['detected']} g4assoc {f['g4_assoc']} offered {f['g4_offered']} "
              f"relAcc {f['g4_rel_accepted']} ({time.time()-t0:.0f}s)")

    all_rows = [row for r in per for row in r["rows"]]
    clean_sessions = [r for r in per if not r["contact"]]
    report(per, all_rows, args.gate, cfg)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"gate": args.gate, "sessions": [{k: v for k, v in r.items() if k != "rows"} for r in per],
             "constants": dict(given_pos_std=cfg.given_pos_std, inplane_sigma=cfg.gate_rel_inplane_sigma,
                               along_sigma=cfg.gate_rel_along_sigma, chi2=cfg.gate_rel_chi2,
                               vision_max_range_m=cfg.vision_max_range_m,
                               attitude_noise_deg=float(np.degrees(cfg.attitude_noise_std))),
             "rows": all_rows}, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote per-frame rows -> {args.json}")
    return 0


def report(per, all_rows, gate_id, cfg) -> None:
    n_sess = len(per); n_contact = sum(r["contact"] for r in per)
    F_ = {k: sum(r["funnel"][k] for r in per) for r in per[:1] for k in per[0]["funnel"]}
    print(f"\n========== GATE-{gate_id} SHADOW @ AT-SPEED (pooled {n_sess} laps, {n_contact} contact) ==========")
    print("FUNNEL (geometry → detector → associate → offered → relinnov-accepted), pooled:")
    fr = F_["frames"]
    print(f"  frames {fr}  g4-in-FoV {F_['g4_in_fov']} ({100*F_['g4_in_fov']/fr:.0f}%)  "
          f"any-gate-FoV {F_['any_in_fov']} ({100*F_['any_in_fov']/fr:.0f}%)  detected {F_['detected']} "
          f"({100*F_['detected']/fr:.0f}%)")
    print(f"  g4-associated {F_['g4_assoc']} ({100*F_['g4_assoc']/fr:.1f}%)  "
          f"g4-offered {F_['g4_offered']} ({100*F_['g4_offered']/fr:.1f}%)  "
          f"g4-relinnov-accepted {F_['g4_rel_accepted']} ({100*F_['g4_rel_accepted']/fr:.1f}%)")

    offered = [r for r in all_rows if r.get("offered")]
    accepted = [r for r in all_rows if r.get("accepted")]
    print(f"\n--- (i) EFFECTIVE attitude/accel BIAS at gate-{gate_id} = (fix − GT) in the gate plane ---")
    print(f"    OFFERED N={len(offered)}  ACCEPTED N={len(accepted)}  (sign: +cross=gate-right, "
          f"+vert=gate-down, +along=beyond gate)")
    for label, subset in [("offered", offered), ("relinnov-accepted", accepted)]:
        if not subset:
            continue
        med_rng = float(np.median([r["true_range_m"] for r in subset]))
        print(f"  [{label}] median range {med_rng:.1f} m")
        for nm, key in [("ABS  cross", "abs_cross"), ("ABS  vert ", "abs_vert"), ("ABS  along", "abs_along"),
                        ("ABS  |inplane|", "abs_inplane"),
                        ("REL  cross", "rel_cross"), ("REL  vert ", "rel_vert"), ("REL  along", "rel_along"),
                        ("REL  |inplane|", "rel_inplane")]:
            s = _st([r.get(key) for r in subset])
            ang = np.degrees(np.arctan2(abs(s["mean"]), med_rng)) if np.isfinite(s["mean"]) else float("nan")
            print(f"    {nm:>16}: bias(mean) {s['mean']:+.3f}  sigma {s['std']:.3f}  "
                  f"p50 {s['p50']:+.3f}  p90 {s['p90']:+.3f} m   (≈{ang:.2f}° @ {med_rng:.0f}m)")

    print(f"\n--- (ii) FIX ACCEPT-RATE + sigma at the binding band (relinnov gate, chi2={cfg.gate_rel_chi2:.2f}) ---")
    if offered:
        acc = 100.0 * len(accepted) / len(offered)
        d2 = _st([r["d2_rel"] for r in offered])
        print(f"    relinnov accept-rate (of OFFERED): {len(accepted)}/{len(offered)} = {acc:.0f}%   "
              f"d2_rel p50 {d2['p50']:.2f} p90 {d2['p90']:.2f}")
        si = _st([r["rel_cross"] for r in accepted]); sv = _st([r["rel_vert"] for r in accepted])
        sip = _st([r["rel_inplane"] for r in accepted])
        print(f"    accepted REL in-plane: cross sigma {si['std']:.3f}  vert sigma {sv['std']:.3f}  "
              f"|inplane| p50 {sip['p50']:.3f} p90 {sip['p90']:.3f} m")
        nis = _st([r["nav_inplane_sigma"] for r in accepted]); nas = _st([r["nav_along_sigma"] for r in accepted])
        print(f"    exported gate-frame cov: nav_inplane_sigma p50 {nis['p50']:.3f}  "
              f"nav_along_sigma p50 {nas['p50']:.3f} m")

    print(f"\n--- (iii) CRAB ↔ gate-in-FoV ↔ fix-rate ---")
    has_crab = [r for r in all_rows if r.get("crab_deg") is not None]
    print(f"    crab (sideslip) over all moving frames: {_st([r['crab_deg'] for r in has_crab])}")
    print(f"    {'crab band':>12} {'frames':>7} {'g4FoV%':>7} {'g4assoc%':>9} {'offered':>8} {'accepted':>9}")
    for lab, lo, hi in [("0-15°", 0, 15), ("15-30°", 15, 30), ("30-45°", 30, 45),
                        ("45-60°", 45, 60), (">60°", 60, 999)]:
        band = [r for r in has_crab if lo <= r["crab_deg"] < hi]
        if not band:
            continue
        nf = len(band); fov = sum(r["g4_in_fov"] for r in band); asc = sum(r["associated"] for r in band)
        ofd = sum(bool(r.get("offered")) for r in band); ac = sum(bool(r.get("accepted")) for r in band)
        print(f"    {lab:>12} {nf:>7} {100*fov/nf:>6.0f}% {100*asc/nf:>8.1f}% {ofd:>8} {ac:>9}")
    print(f"    {'g4 bearing':>12} {'frames':>7} {'g4FoV%':>7} {'g4assoc%':>9} {'offered':>8} {'accepted':>9}")
    valid_b = [r for r in all_rows if np.isfinite(r.get("g4_bearing_deg", float('nan')))]
    for lab, lo, hi in [("0-15°", 0, 15), ("15-30°", 15, 30), ("30-45°", 30, 45),
                        ("45-60°", 45, 60), (">60°", 60, 999)]:
        band = [r for r in valid_b if lo <= r["g4_bearing_deg"] < hi]
        if not band:
            continue
        nf = len(band); fov = sum(r["g4_in_fov"] for r in band); asc = sum(r["associated"] for r in band)
        ofd = sum(bool(r.get("offered")) for r in band); ac = sum(bool(r.get("accepted")) for r in band)
        print(f"    {lab:>12} {nf:>7} {100*fov/nf:>6.0f}% {100*asc/nf:>8.1f}% {ofd:>8} {ac:>9}")

    # --- close / no-close cross-reference vs the C2 G3 thresholds ---
    print(f"\n--- CLOSE / NO-CLOSE cross-reference (C2 G3: rel σ_inplane model 0.265; "
          f"MARGIN(r)=0.535−r; central r=0.30 → 0.235 m, worst r=0.38 → 0.155 m; G3 was bias-FREE) ---")
    subset = accepted if accepted else offered
    if subset:
        bias_ip = float(np.hypot(np.nanmean([r["rel_cross"] for r in subset]),
                                 np.nanmean([r["rel_vert"] for r in subset])))
        sig_cross = float(np.nanstd([r["rel_cross"] for r in subset]))
        p90_ip = float(np.nanpercentile([r["rel_inplane"] for r in subset], 90))
        print(f"    MEASURED gate-{gate_id} (on {'accepted' if accepted else 'offered'} N={len(subset)}): "
              f"effective in-plane bias |b| = {bias_ip:.3f} m, cross-track σ = {sig_cross:.3f} m, "
              f"|in-plane| p90 = {p90_ip:.3f} m")
        for r_arm, marg in [(0.30, 0.235), (0.38, 0.155)]:
            verdict = "CLOSES" if p90_ip <= marg else "does NOT close"
            print(f"      vs MARGIN(r={r_arm})={marg:.3f} m : measured p90 {p90_ip:.3f} → {verdict}")
        print(f"    NOTE: no margin-closure-envelope handoff present — raw numbers above are for "
              f"morning synthesis. The bias |b| is the residual the +L fix does NOT cancel (the binding factor).")


if __name__ == "__main__":
    raise SystemExit(main())
