"""render_vision_video.py — FAITHFUL onboard-vision video from a recorded session.

Unlike a naive "write every captured frame at constant fps" renderer (which compresses the stream's
dropped-frame holes into single steps and makes motion SURGE), this one plays back on the real
``recv_monotonic_ns`` clock: each captured frame is held for its true duration, so a gap where frames
were dropped shows as an honest brief FREEZE with a "STREAM GAP" banner -- never a surge. Overlays the
red_glow detector (gate quad + corners + PnP range) and a live drop counter.

USAGE
  python scripts/render_vision_video.py <session_dir> <out_mp4> [--slowmo 4] [--out-fps 30]

--slowmo N  : play N times slower than real time (default 4, so a 430ms hole is a ~1.7s freeze).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import cv2
import imageio.v2 as imageio          # H.264 (libx264) writer -- browser-playable, unlike cv2's mp4v
from racer.contracts import Frame
from racer.vision.red_glow_detector import RedGlowGateDetector
from racer.vision.detector import GateDetector
from racer.vision.gate_pose import estimate_gate_pose
from racer.frames import CAMERA_INTRINSICS_K as _K, R_camera_from_body

GAP_MS = 60.0   # a recv hole bigger than this is flagged as a stream gap / dropped frames
_R_CB = R_camera_from_body()   # body(FRD) -> camera(optical), +20deg mount baked in


def _project_flu(flu):
    """Project a body-FLU point [fwd,left,up] (metres) to 640x360 image (u,v,depth_m), or None if
    behind/at the camera. FLU->FRD = [fwd,-left,-up]; then R_camera_from_body + pinhole K."""
    p_cam = _R_CB @ np.array([flu[0], -flu[1], -flu[2]], dtype=np.float64)
    z = p_cam[2]
    if z <= 0.05:
        return None
    return float(_K[0, 0] * p_cam[0] / z + _K[0, 2]), float(_K[1, 1] * p_cam[1] / z + _K[1, 2]), float(z)


def _load_fed_points(session):
    """Map video frame_id -> the target the policy was FED on that frame (slot0 active gate):
    {frame_id: (rel_flu [fwd,left,up] with z-bias baked in, dist_m, gate_index)}. Keyed by frame_id --
    the ONLY exact cross-stream join, because the video and ego logs stamp sim_time_ns from DIFFERENT
    epochs. Later ticks on the same frame overwrite -> the most-settled fed point for that frame. {}
    if the session predates rel_flu/frame_id logging."""
    p = Path(session) / "ego_obs.jsonl"
    if not p.exists():
        return {}
    fed = {}
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        fid, flu = r.get("frame_id"), r.get("rel_flu")
        if fid is None or flu is None:
            continue
        fed[int(fid)] = (flu, r.get("dist"), r.get("gate_index"))
    return fed


# Fallback ONLY for pre-2026-07-19 sessions whose meta.json predates the seeker fields. This is
# the current champion M-engine (what recent v1/v15/v16 flights flew); a render off it prints a
# loud warning because it is a GUESS, not the recorded flight detector.
_FALLBACK_WEIGHTS = "C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06_fp16_384x640.engine"


def _load_flight_detector(session, weights_override):
    """Load the SAME detector the flight actually used, from <session>/meta.json (seeker_detector +
    seeker_weights, written by fly_rl on the ego path). --weights forces an override. Old sessions
    with no recorded detector fall back to the current champion M-engine with a loud warning."""
    if weights_override:
        print(f"[render] detector OVERRIDE (--weights, may NOT match the flight): {weights_override}")
        return GateDetector.load(weights_override)
    meta = {}
    mp = Path(session) / "meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception as e:
            print(f"[render] WARNING: could not read {mp}: {e}")
    detname, weights = meta.get("seeker_detector"), meta.get("seeker_weights")
    if detname == "red_glow":
        print("[render] flight detector (from meta): red_glow classical -> overlaying red_glow")
        return RedGlowGateDetector()
    if weights:
        print(f"[render] flight detector (from meta.json, matches the flight): {weights}")
        return GateDetector.load(weights)
    print(f"[render] WARNING: {mp} has no seeker_detector/seeker_weights (pre-2026-07-19 flight). "
          f"Falling back to the current champion M-engine, which may NOT match this flight:\n"
          f"         {_FALLBACK_WEIGHTS}\n"
          f"         pass --weights <engine> if this flight used a different detector.")
    return GateDetector.load(_FALLBACK_WEIGHTS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session"); ap.add_argument("out_mp4")
    ap.add_argument("--slowmo", type=float, default=4.0)
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--weights", default=None,
                    help="Override the detector engine to overlay. DEFAULT (unset): use the SAME "
                         "detector the flight flew, read from <session>/meta.json (seeker_detector "
                         "+ seeker_weights), so the overlay matches the pilot's view. Only set this "
                         "to force a different engine than the flight used.")
    ap.add_argument("--last-seconds", type=float, default=0.0,
                    help="render only the last N seconds (by recv_monotonic_ns); 0 = whole recording. "
                         "The flight is at the END of the recording (the long pre-GO wait precedes it).")
    args = ap.parse_args()

    session = Path(args.session)
    idx = [json.loads(l) for l in (session / "video_index.jsonl").read_text().splitlines() if l.strip()]
    if args.last_seconds > 0 and idx:
        t_end_ns = max(r.get("recv_monotonic_ns", 0) for r in idx)
        cutoff = t_end_ns - args.last_seconds * 1e9
        n_before = len(idx)
        idx = [r for r in idx if r.get("recv_monotonic_ns", 0) >= cutoff]
        print(f"[render] trimmed to last {args.last_seconds:g}s: {len(idx)}/{n_before} frames")
    blob = (session / "video.bin").read_bytes()
    det = _load_flight_detector(session, args.weights)
    # the POINT the policy was fed on each frame (slot0 active gate, z-bias offset baked in) + distance
    fed = _load_fed_points(session)
    z_bias = 0.0
    try:
        z_bias = float((json.loads((session / "meta.json").read_text()).get("ego_gate_z_bias") or 0.0))
    except Exception:
        pass
    if fed:
        print(f"[render] fed-point overlay: {len(fed)} frames with a policy target; z-bias offset {z_bias:+.2f} m")
    else:
        print("[render] no rel_flu/frame_id in ego_obs.jsonl (pre-2026-07-19 flight) -- fed-point overlay OFF")
    S = args.scale
    W, H = 640 * S, 360 * S
    # H.264 (yuv420p + faststart) via imageio-ffmpeg's bundled ffmpeg. cv2's mp4v is NOT playable
    # in a browser <video> tag (MPEG-4 Part 2); libx264/yuv420p is, and faststart streams instantly.
    vw = imageio.get_writer(args.out_mp4, fps=args.out_fps, codec="libx264",
                            macro_block_size=1, pixelformat="yuv420p",
                            ffmpeg_params=["-movflags", "+faststart", "-preset", "veryfast"])

    # decode + detect every captured frame once; remember recv time + frame_id
    frames = []
    for rec in idx:
        jpg = blob[rec["offset"]:rec["offset"] + rec["length"]]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fr = Frame(frame_id=rec["frame_id"], sim_time_ns=rec["sim_time_ns"], image_bgr=img,
                   recv_monotonic_ns=rec.get("recv_monotonic_ns", 0))
        obs = det.detect(fr)
        big = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        best = None
        for o in obs:
            pts = (np.asarray(o.corners_px, dtype=np.float32) * S).astype(np.int32)
            cv2.polylines(big, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
            for (x, y) in pts:
                cv2.circle(big, (int(x), int(y)), 4, (0, 200, 255), -1)
            try:
                pose = estimate_gate_pose(o, compute_covariance=False)
                if pose is not None and np.isfinite(pose.range_m):
                    r = float(pose.range_m); best = r if best is None else min(best, r)
                    c = pts.mean(axis=0).astype(int)
                    cv2.putText(big, f"{r:.1f}m", (c[0] - 20, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass
        # --- the POINT fed to the policy on this frame (slot0 active gate, z-bias offset baked in) ---
        fp = fed.get(rec["frame_id"])
        fed_dist = None
        if fp is not None:
            flu, dist, gi = fp
            fed_dist = dist
            pr = _project_flu(flu)
            if pr is not None:
                u, v, _z = pr
                cu, cv = int(round(u * S)), int(round(v * S))
                # MAGENTA cross-diamond = where we're actually aiming the drone (fed to the policy).
                cv2.drawMarker(big, (cu, cv), (255, 0, 255), cv2.MARKER_DIAMOND, 20 + 6 * S, 2, cv2.LINE_AA)
                cv2.drawMarker(big, (cu, cv), (255, 0, 255), cv2.MARKER_CROSS, 12 + 4 * S, 1, cv2.LINE_AA)
                label = f"FED g{gi}" + (f"  {dist:.1f} m" if dist is not None else "")
                cv2.putText(big, label, (cu + 12, cv + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 255), 2, cv2.LINE_AA)
                if abs(z_bias) > 1e-6:
                    cv2.putText(big, f"z-off {z_bias:+.2f}m", (cu + 12, cv + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 255), 1, cv2.LINE_AA)
        frames.append({"recv_ms": rec["recv_monotonic_ns"] / 1e6, "fid": rec["frame_id"],
                       "img": big, "dets": len(obs), "best": best, "fed_dist": fed_dist})

    if len(frames) < 2:
        raise SystemExit("too few frames")

    t0 = frames[0]["recv_ms"]
    t_end = frames[-1]["recv_ms"]
    out_dt_ms = 1000.0 / args.out_fps
    # one tick of real time per output frame is out_dt/slowmo
    real_step = out_dt_ms / args.slowmo
    dropped_cum = 0
    j = 0
    t = t0
    while t <= t_end + 1e-6:
        # advance to the most recent captured frame whose recv_ms <= t
        while j + 1 < len(frames) and frames[j + 1]["recv_ms"] <= t:
            gap_frames = frames[j + 1]["fid"] - frames[j]["fid"] - 1
            if gap_frames > 0:
                dropped_cum += gap_frames
            j += 1
        cur = frames[j]
        canvas = cur["img"].copy()
        # header
        cv2.rectangle(canvas, (0, 0), (W, 26), (0, 0, 0), -1)
        hdr = (f"VQ2 cam  t+{(cur['recv_ms']-t0)/1000:5.2f}s  fid={cur['fid']}  "
               f"dets={cur['dets']}" + (f"  R~{cur['best']:.1f}m" if cur['best'] else "")
               + f"   dropped so far: {dropped_cum}")
        cv2.putText(canvas, hdr, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # gap banner: are we sitting in a hole (next captured frame is far ahead in real time)?
        nxt = frames[j + 1] if j + 1 < len(frames) else None
        if nxt is not None:
            hole = nxt["recv_ms"] - cur["recv_ms"]
            missing = nxt["fid"] - cur["fid"] - 1
            if hole > GAP_MS and (t - cur["recv_ms"]) > GAP_MS:
                cv2.rectangle(canvas, (0, H - 30), (W, H), (0, 0, 90), -1)
                cv2.putText(canvas, f"STREAM GAP  held {t-cur['recv_ms']:.0f}ms  (~{missing} frames dropped)",
                            (8, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
        vw.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))   # imageio wants RGB
        t += real_step

    vw.close()
    total_span = frames[-1]["fid"] - frames[0]["fid"] + 1
    print(f"wrote {args.out_mp4}  ({len(frames)} captured / {total_span} sent, "
          f"{dropped_cum} dropped) slowmo={args.slowmo}x out_fps={args.out_fps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
