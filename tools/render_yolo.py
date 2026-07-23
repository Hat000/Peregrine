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
from racer.vision.centre_emit import emit_from_observation
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
        # rel_flu1 = the SLOT1 (next-gate) lever fed alongside it. Carried so the overlay can show
        # what slot1 was actually filled with -- the slot that fed the SAME gate as slot0 on 37-60%
        # of two-slot ticks before b518441d, and whose tg+1 identity is still a heuristic.
        fed[int(fid)] = (flu, r.get("dist"), r.get("gate_index"), r.get("rel_flu1"))
    return fed


# Fallback ONLY for pre-2026-07-19 sessions whose meta.json predates the seeker fields. This is
# the current champion M-engine (what recent v1/v15/v16 flights flew); a render off it prints a
# loud warning because it is a GUESS, not the recorded flight detector.
_FALLBACK_WEIGHTS = "C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06_fp16_384x640.engine"


_PILOTS_JSON = Path(__file__).resolve().parent / "pilot_panel_logs" / "pilots.json"


def _detector_from_launch_log(session_name):
    """(detector, weights) recovered from the panel's stored LAUNCH RECORD, or (None, None).

    fly_rl only began writing the flown detector into meta.json on 2026-07-19, so 908 of 1229
    sessions have no detector in their meta and would otherwise render off a GUESS. The panel has
    kept the full launch argv for every flight it started since long before that -- which recovers
    the true engine for 786 of those 908. This is the same record that holds the only surviving copy
    of a flight's assist/arrestor knobs, so it is the authority whenever meta.json is silent."""
    try:
        recs = json.loads(_PILOTS_JSON.read_text())
    except Exception:
        return None, None
    for p in recs if isinstance(recs, list) else []:
        if p.get("session") != session_name:
            continue
        cfg = p.get("config") or {}
        det, weights = cfg.get("seeker_detector"), cfg.get("seeker_weights")
        if not (det or weights):        # pre-``config`` records keep only the argv
            cmd = p.get("cmd") or []
            if "--seeker-detector" in cmd:
                det = cmd[cmd.index("--seeker-detector") + 1]
            if "--seeker-weights" in cmd:
                weights = cmd[cmd.index("--seeker-weights") + 1]
        return det, weights
    return None, None


def _load_flight_detector(session, weights_override):
    """Load the SAME detector the flight actually used. Source order: --weights override, then
    <session>/meta.json (seeker_detector + seeker_weights, written by fly_rl on the ego path since
    2026-07-19), then the panel's launch record for that session, and only then a warned fallback."""
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
    # meta.json is silent (pre-2026-07-19 flight) -- recover the flown detector from the panel's
    # launch record before resorting to a guess.
    detname, weights = _detector_from_launch_log(Path(session).name)
    if detname == "red_glow":
        print("[render] flight detector (from the panel launch record): red_glow classical")
        return RedGlowGateDetector()
    if weights:
        print(f"[render] flight detector (from the panel LAUNCH RECORD, matches the flight; "
              f"meta.json predates the seeker fields): {weights}")
        return GateDetector.load(weights)
    print(f"[render] WARNING: {mp} has no seeker_detector/seeker_weights and this session is not in "
          f"the panel launch record, so the flown detector is UNKNOWN. Falling back to the current "
          f"champion M-engine, which may NOT match this flight:\n"
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
    ap.add_argument("--emit", choices=["auto", "pnp", "centre"], default="auto",
                    help="Gate-emit mode for the overlay. 'auto' (default) reads emit_mode from "
                         "meta.json so the render matches the flight. Force one to A/B the SAME "
                         "recording both ways -- e.g. re-render a PnP flight with --weights <M+1 "
                         "engine> --emit centre to see what the centre path would have seen. "
                         "'centre' needs a 5-keypoint model or every gate goes unlabelled.")
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
    emit_mode = "pnp"
    valid_cap = 0.0
    try:
        _m = json.loads((session / "meta.json").read_text())
        z_bias = float(_m.get("ego_gate_z_bias") or 0.0)
        # Render the SAME emit the flight flew. A centre-mode flight whose overlay showed PnP ranges
        # would be reporting numbers the policy never saw -- and would show NOTHING at all on exactly
        # the cropped gates that mode exists to recover (estimate_gate_pose declines <3 corners).
        # KEY MISMATCH (fixed 2026-07-23): fly_rl writes emit_mode inside _meta_seeker_constants,
        # i.e. NESTED under "seeker_constants" -- reading it at the top level resolved to None and
        # silently fell back to "pnp". Every centre-mode flight was therefore rendered through the
        # PnP path, labelling each gate with a range the policy never saw (one measured case: 20.2 m
        # on screen vs 15.7 m actually fed, a 29% gap). That is precisely what da2e1b66 set out to
        # prevent, so the guard has to survive the key moving: check nested FIRST (where it is
        # today), then top level (where it is also written now, and where a human looks).
        emit_mode = str((_m.get("seeker_constants") or {}).get("emit_mode")
                        or _m.get("emit_mode") or "pnp")
        # the flight's own valid-range cap, so the overlay can mark what the seeker dropped
        valid_cap = float((_m.get("seeker_constants") or {}).get("max_valid_range_m") or 0.0)
    except Exception:
        pass
    if args.emit != "auto" and args.emit != emit_mode:
        print(f"[render] emit OVERRIDE: {emit_mode} (flown) -> {args.emit} (rendered). Ranges below "
              f"are NOT what this flight fed the policy.")
        emit_mode = args.emit
    else:
        print(f"[render] gate emit mode (from meta.json, matches the flight): {emit_mode}"
              + ("  -- ranges below are centre+size, as flown" if emit_mode == "centre" else ""))
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
    last_fp = None          # last fed point, held across video frames that carry no control tick
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
        best_trusted = False       # is `best` corner-sourced (a measurement) or a bbox guess?
        n_drawn = 0                # detections that SURVIVED the range cap -- what the header reports
        for o in obs:
            # RANGE-CAP FILTER, applied BEFORE anything is drawn. This overlay re-runs the detector,
            # so without it the frame shows candidates the seeker had already dropped -- an 83.1 m
            # label on screen reads as "the cap is not working" when in fact the cap removed it from
            # the pool before selection (verified on 20260723_221720: 0 of 397 candidates above 30 m
            # ever reached selection). Render what the FLIGHT considered, nothing more.
            _e = emit_from_observation(o) if getattr(o, "centre_px", None) is not None else None
            _r = None
            if emit_mode == "centre" and _e is not None:
                _r = float(_e.depth_m)
            else:
                try:
                    _p = estimate_gate_pose(o, compute_covariance=False)
                    _r = None if _p is None or not np.isfinite(_p.range_m) else float(_p.range_m)
                except Exception:
                    _r = None
            if valid_cap and _r is not None and _r > valid_cap:
                continue                     # dropped by the flight's own cap -> not drawn at all
            n_drawn += 1
            pts = (np.asarray(o.corners_px, dtype=np.float32) * S).astype(np.int32)
            # M+1 carries 0-4 inner corners (the centre survives cropping, the corners do not), so the
            # quad is only closed when there IS one. Fewer than 3 => just mark the corners that made it.
            if pts.shape[0] >= 3:
                cv2.polylines(big, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
            elif pts.shape[0] == 2:
                cv2.line(big, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2, cv2.LINE_AA)
            for (x, y) in pts:
                cv2.circle(big, (int(x), int(y)), 4, (0, 200, 255), -1)

            # --- the M+1 REGRESSED CENTRE: the measurement the flight actually steered on ----------
            emit = emit_from_observation(o) if getattr(o, "centre_px", None) is not None else None
            anchor = None
            if emit is not None:
                ecu = int(round(float(o.centre_px[0]) * S)); ecv = int(round(float(o.centre_px[1]) * S))
                # CYAN circle+dot = the regressed centre keypoint (distinct from the magenta FED point,
                # which is what the POLICY got after the z-bias offset and any track smoothing).
                cv2.circle(big, (ecu, ecv), 7 + 2 * S, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(big, (ecu, ecv), 2, (255, 255, 0), -1, cv2.LINE_AA)
                anchor = (ecu, ecv)

            r = None
            if emit_mode == "centre" and emit is not None:
                r = float(emit.depth_m)          # as flown: min-over-pairs (or bbox) + centre bearing
            else:
                try:
                    pose = estimate_gate_pose(o, compute_covariance=False)
                    if pose is not None and np.isfinite(pose.range_m):
                        r = float(pose.range_m)
                except Exception:
                    r = None
            if r is not None:
                # The header's R~ must not be driven by a bbox GUESS. A corner-sourced range is a
                # measurement; a bbox one conflates range with tilt and fires on false positives
                # (a wall panel with no corners still has a box). Prefer corners; fall back to bbox
                # only when nothing this frame had corners at all.
                trust = emit is None or emit.range_src == "corners"
                if trust:
                    best = r if (best is None or not best_trusted) else min(best, r)
                    best_trusted = True
                elif not best_trusted:
                    best = r if best is None else min(best, r)
                c = anchor if anchor is not None else (pts.mean(axis=0).astype(int)
                                                       if pts.shape[0] else None)
                if c is not None:
                    # Tag the SOURCE when it is the coarse one: a bbox-derived range conflates range
                    # with tilt and reads FAR on a cropped gate, so it must never look like a measurement.
                    src = "" if emit is None or emit.range_src == "corners" else " bbox?"
                    col = (0, 255, 255) if not src else (0, 165, 255)
                    # BEYOND THE FLIGHT'S RANGE CAP: this overlay re-runs the detector, so it shows
                    # candidates the seeker DROPPED before selection. Left unmarked, an 83 m label
                    # reads as "the cap is not working" when the cap in fact removed it from the pool
                    # (verified on 20260723_221720: 0 of 397 candidates above 30 m). Mark them CUT.
                    cv2.putText(big, f"{r:.1f}m{src}", (int(c[0]) - 20, int(c[1]) - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        # --- the POINT fed to the policy on this frame (slot0 active gate, z-bias offset baked in) ---
        # HOLD the last fed point across frames that carry no control tick. There are MORE video
        # frames than control ticks (measured 243 vs 184 on 20260723_221720), so 28.4% of frames had
        # no tick to paint and the magenta marker simply vanished -- which reads as the policy losing
        # the gate. It was not: rel_flu was present on 184/184 ticks, i.e. 100%. The blink was purely
        # an artifact of this overlay. Held frames draw HOLLOW so the distinction stays honest --
        # a held marker is the last known aim point, not a fresh one.
        fp = fed.get(rec["frame_id"])
        fresh_fed = fp is not None
        if fp is None:
            fp = last_fp                      # carry the previous tick's aim point
        else:
            last_fp = fp
        fed_dist = None
        if fp is not None:
            flu, dist, gi, flu1 = fp
            fed_dist = dist
            pr = _project_flu(flu)
            if pr is not None:
                u, v, _z = pr
                cu, cv = int(round(u * S)), int(round(v * S))
                # MAGENTA cross-diamond = where we're actually aiming the drone (fed to the policy).
                # FRESH = solid diamond + cross. HELD (no control tick on this video frame) = the
                # diamond alone, thinner: same aim point, but not a new measurement.
                cv2.drawMarker(big, (cu, cv), (255, 0, 255), cv2.MARKER_DIAMOND, 20 + 6 * S,
                               2 if fresh_fed else 1, cv2.LINE_AA)
                if fresh_fed:
                    cv2.drawMarker(big, (cu, cv), (255, 0, 255), cv2.MARKER_CROSS, 12 + 4 * S, 1, cv2.LINE_AA)
                label = (f"FED g{gi}" if fresh_fed else f"FED g{gi} (held)") + \
                        (f"  {dist:.1f} m" if dist is not None else "")
                cv2.putText(big, label, (cu + 12, cv + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 255), 2, cv2.LINE_AA)
                if abs(z_bias) > 1e-6:
                    cv2.putText(big, f"z-off {z_bias:+.2f}m", (cu + 12, cv + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 255), 1, cv2.LINE_AA)
            # --- SLOT1: the NEXT-gate lever fed alongside slot0 (--ego-slot1) --------------------
            # Drawn separately and in ORANGE because slot0 and slot1 pointing at the SAME gate is a
            # real failure mode (37-60% of two-slot ticks before b518441d), and the tg+1 identity is
            # still only "nearest non-active" -- no map prior, no notion of which gate is truly next.
            # Seeing both markers at once is the only way to catch that by eye.
            if flu1 is not None:
                pr1 = _project_flu(flu1)
                if pr1 is not None:
                    u1, v1, _z1 = pr1
                    c1u, c1v = int(round(u1 * S)), int(round(v1 * S))
                    d1 = float(np.linalg.norm(np.asarray(flu1, dtype=np.float64)))
                    cv2.drawMarker(big, (c1u, c1v), (0, 165, 255), cv2.MARKER_SQUARE,
                                   16 + 5 * S, 2 if fresh_fed else 1, cv2.LINE_AA)
                    cv2.putText(big, f"SLOT1  {d1:.1f} m", (c1u + 12, c1v + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
        frames.append({"recv_ms": rec["recv_monotonic_ns"] / 1e6, "fid": rec["frame_id"],
                       "img": big, "dets": n_drawn, "best": best, "fed_dist": fed_dist})

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
               + f"  emit={emit_mode}"
               + (f"  fed {cur['fed_dist']:.1f}m" if cur.get("fed_dist") is not None else "")
               + f"   dropped so far: {dropped_cum}")
        cv2.putText(canvas, hdr, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # LEGEND: the two markers mean different things and the difference is the whole diagnostic --
        # cyan is what the DETECTOR saw this frame, magenta is what the POLICY was fed (after the
        # z-bias offset, the det-hold and any track smoothing). They separating is the signal.
        cv2.rectangle(canvas, (0, 26), (W, 46), (0, 0, 0), -1)
        cv2.circle(canvas, (14, 36), 5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "regressed centre (detector)", (26, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (250, 36), (255, 0, 255), cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
        cv2.putText(canvas, "FED to policy", (262, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
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
