"""render_cmd_indicator.py — onboard video with a PER-FRAME COMMAND indicator + frame-lag overlay.

Standing flight-test render tool (pilot request, VQ2 A16/A17). Two things the stock
scripts/render_vision_video.py does NOT show, both needed to judge the frame-drop fix:

  1. FRAME LAG / DROPS — plays on the real recv_monotonic clock so a dropped-frame hole shows as an
     honest FREEZE + "STREAM GAP" banner + a running dropped counter (same faithful playback as the
     stock tool).
  2. COMMAND CADENCE — overlays, per frame, the control command IN EFFECT at that frame (matched to
     the nearest nav_estimate tick by sim_time_ns): the command #, its body-rate + thrust, and a
     PULSE marker that flashes each time a NEW command tick lands. Lets you SEE how fast commands
     update relative to how fast video frames arrive (the whole point: are we command-starved, frame-
     starved, or both?). A cumulative "cmds seen" count is shown too.

Deliberately NO detector overlay: the stock tool draws the classical red_glow detector, which is NOT
the YOLO the flight actually uses and misled the read (A16). Pass --yolo <weights> to overlay the
REAL YOLO detections instead (slow — runs detect per frame at render time).

USAGE
  python handoff/tools/render_cmd_indicator.py <session_dir> <out_mp4> [--slowmo 2] [--out-fps 30]
         [--yolo models/gate_clean_ens_course_L110.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import cv2
from racer.contracts import Frame

GAP_MS = 60.0   # a recv hole bigger than this is flagged as a stream gap / dropped frames


def _load_ticks(session: Path):
    """nav_estimate.jsonl -> list of (sim_time_ns, tick_index, body_rate, thrust), sorted by time."""
    p = session / "nav_estimate.jsonl"
    if not p.exists():
        return []
    ticks = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ticks.append((int(r["sim_time_ns"]), int(r.get("tick_index", -1)),
                      r.get("body_rate", [0.0, 0.0, 0.0]), float(r.get("thrust", 0.0))))
    ticks.sort(key=lambda t: t[0])
    return ticks


def _cmd_for_frame(sim_ns: int, ticks):
    """The command IN EFFECT at a frame = the latest tick with sim_time_ns <= the frame's. Returns
    (tick_index, body_rate, thrust, ordinal) where ordinal is that tick's position in the list."""
    lo, hi, best = 0, len(ticks) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if ticks[mid][0] <= sim_ns:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    if best < 0:
        return None
    _, tidx, br, thr = ticks[best]
    return tidx, br, thr, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session"); ap.add_argument("out_mp4")
    ap.add_argument("--slowmo", type=float, default=2.0)
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--yolo", default=None, help="optional YOLO weights to overlay the REAL detections")
    ap.add_argument("--from-go", dest="from_go", action="store_true", default=True,
                    help="(default) trim the video to the ARMED FLIGHT window (from GO to crash), "
                         "dropping the pre-GO waiting room so the video matches the live flight.")
    ap.add_argument("--full", dest="from_go", action="store_false",
                    help="render the WHOLE recording incl. the pre-GO waiting room (no trim).")
    ap.add_argument("--pre-go-s", type=float, default=0.5,
                    help="seconds of lead-in before GO to keep (default 0.5) so the launch is visible.")
    args = ap.parse_args()

    session = Path(args.session)
    idx = [json.loads(l) for l in (session / "video_index.jsonl").read_text().splitlines() if l.strip()]
    blob = (session / "video.bin").read_bytes()
    ticks = _load_ticks(session)
    S = args.scale
    W, H = 640 * S, 360 * S

    det = None
    if args.yolo:
        from racer.vision.gate_detector import GateDetector  # lazy: only when requested
        det = GateDetector.load(args.yolo)

    vw = cv2.VideoWriter(args.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.out_fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed (mp4v)")

    frames = []
    for rec in idx:
        jpg = blob[rec["offset"]:rec["offset"] + rec["length"]]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        big = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        dets = 0
        if det is not None:
            fr = Frame(frame_id=rec["frame_id"], sim_time_ns=rec["sim_time_ns"], image_bgr=img,
                       recv_monotonic_ns=rec.get("recv_monotonic_ns", 0))
            try:
                obs = det.detect(fr); dets = len(obs)
                for o in obs:
                    pts = (np.asarray(o.corners_px, dtype=np.float32) * S).astype(np.int32)
                    cv2.polylines(big, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                pass
        frames.append({"recv_ms": rec["recv_monotonic_ns"] / 1e6, "fid": rec["frame_id"],
                       "sim_ns": int(rec["sim_time_ns"]), "img": big, "dets": dets})

    if len(frames) < 1:
        raise SystemExit("no decodable frames in video.bin -- recorder captured nothing (severe drops)")

    # --- FROM-GO trim (pilot request): the recorder attaches during the pre-GO passive wait, so the
    # recording spans (long waiting room) + (armed flight); the flight is the TAIL, ending at the crash
    # (both video & nav_estimate end at disarm). nav_estimate = armed flight ONLY, so its duration is
    # the flight length. Keep only the last (nav_duration + pre_go) seconds of recv-time so the video
    # STARTS at GO and lines up with what the pilot saw live. (Clocks differ - epoch vs sim-boot - so we
    # anchor by DURATION-from-the-end, not absolute time.) ---
    if args.from_go and ticks and len(frames) >= 2:
        nav_dur_s = (ticks[-1][0] - ticks[0][0]) / 1e9
        recv_end = frames[-1]["recv_ms"]
        flight_start = recv_end - (nav_dur_s + args.pre_go_s) * 1000.0
        kept = [f for f in frames if f["recv_ms"] >= flight_start]
        if len(kept) >= 2:
            print(f"[from-go] trimmed to the flight window: {len(kept)}/{len(frames)} frames "
                  f"(last {nav_dur_s + args.pre_go_s:.1f}s, GO->crash; use --full for the whole recording)")
            frames = kept
        else:
            print(f"[from-go] flight window too small ({len(kept)} frames) -- keeping full recording")

    if len(frames) < 2:
        # single frame: still emit a short clip so the path is valid
        canvas = frames[0]["img"].copy()
        cv2.putText(canvas, "ONLY 1 FRAME CAPTURED (severe drops)", (8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
        for _ in range(args.out_fps):
            vw.write(canvas)
        vw.release()
        print(f"wrote {args.out_mp4}  (1 frame only -- severe drops)")
        return 0

    t0 = frames[0]["recv_ms"]; t_end = frames[-1]["recv_ms"]
    # END-ANCHOR the command track: the recorder's frame timestamps (epoch-ns) and nav_estimate
    # (sim-boot-ns) use DIFFERENT clocks and can't be matched directly, but BOTH streams end at the
    # crash/disarm and the armed flight is the last chunk of the recording. So map playback time t
    # (real recv-time) to nav-time by seconds-from-the-end. Driving off t (not the held frame) means
    # during a STREAM-GAP freeze the command counter KEEPS ADVANCING -> shows commands still flowing
    # while the video is frozen (the whole point). Approximate (±recorder tail); labelled on-screen.
    nav_start_ns = ticks[0][0] if ticks else 0
    nav_end_ns = ticks[-1][0] if ticks else 0
    out_dt_ms = 1000.0 / args.out_fps
    real_step = out_dt_ms / args.slowmo
    dropped_cum = 0
    cmds_seen = set()
    j = 0; t = t0; last_ordinal = -1
    while t <= t_end + 1e-6:
        while j + 1 < len(frames) and frames[j + 1]["recv_ms"] <= t:
            gap_frames = frames[j + 1]["fid"] - frames[j]["fid"] - 1
            if gap_frames > 0:
                dropped_cum += gap_frames
            j += 1
        cur = frames[j]
        canvas = cur["img"].copy()

        # ---- command overlay: end-anchored to the crash, driven by playback time t (see note above) ----
        cmd = None
        if ticks:
            sec_from_end = (t_end - t) / 1000.0
            target_ns = nav_end_ns - int(sec_from_end * 1e9)
            if target_ns >= nav_start_ns - 200_000_000:   # 0.2s margin before the first armed tick
                cmd = _cmd_for_frame(max(target_ns, nav_start_ns), ticks)
        pulse = False
        if cmd is not None:
            tidx, br, thr, ordinal = cmd
            cmds_seen.add(tidx)
            pulse = ordinal != last_ordinal   # a NEW command tick landed on this frame
            last_ordinal = ordinal

        # header (frame-lag info)
        cv2.rectangle(canvas, (0, 0), (W, 26), (0, 0, 0), -1)
        hdr = f"VQ2 telem  t+{(cur['recv_ms']-t0)/1000:5.2f}s  fid={cur['fid']}  dropped: {dropped_cum}"
        if det is not None:
            hdr += f"  dets={cur['dets']}"
        cv2.putText(canvas, hdr, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # command line + pulse marker
        cv2.rectangle(canvas, (0, 26), (W, 52), (0, 0, 0), -1)
        if cmd is not None:
            cline = (f"cmd#{tidx}  rate[r,p,y]=[{br[0]:+.2f},{br[1]:+.2f},{br[2]:+.2f}]  "
                     f"thr={thr:.2f}   cmds seen: {len(cmds_seen)}")
            cv2.putText(canvas, cline, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
            # pulse dot: bright green flash on a new command tick, dim otherwise
            cv2.circle(canvas, (W - 22, 39), 10, (0, 255, 0) if pulse else (0, 90, 0), -1)
        else:
            cv2.putText(canvas, "no command (pre-arm / waiting room)", (8, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        # stream-gap banner (dropped-frame hole)
        nxt = frames[j + 1] if j + 1 < len(frames) else None
        if nxt is not None:
            hole = nxt["recv_ms"] - cur["recv_ms"]
            missing = nxt["fid"] - cur["fid"] - 1
            if hole > GAP_MS and (t - cur["recv_ms"]) > GAP_MS:
                cv2.rectangle(canvas, (0, H - 30), (W, H), (0, 0, 90), -1)
                cv2.putText(canvas, f"STREAM GAP  held {t-cur['recv_ms']:.0f}ms  (~{missing} frames dropped)",
                            (8, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
        vw.write(canvas)
        t += real_step

    vw.release()
    span = frames[-1]["fid"] - frames[0]["fid"] + 1
    print(f"wrote {args.out_mp4}  ({len(frames)} captured / {span} sent, {dropped_cum} dropped; "
          f"{len(cmds_seen)} distinct commands over {len(ticks)} nav ticks) slowmo={args.slowmo}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
