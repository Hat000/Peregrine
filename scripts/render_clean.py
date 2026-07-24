"""Clean onboard-cockpit video from a recorded session -- NO overlay, for showcase use.

Same faithful recv_monotonic playback as tools/render_yolo.py (holds each frame its true duration,
no motion surge), but draws nothing on top: just the raw onboard camera the drone flew. The sim's own
cyan corridor HUD is part of video.bin (the deploy domain), so the beauty of the shot is intact.

USAGE  python scripts/render_clean.py <session_dir> <out_mp4> [--last-seconds N] [--slowmo 1.5] [--scale 2]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, cv2
import imageio.v2 as imageio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("out_mp4")
    ap.add_argument("--slowmo", type=float, default=1.5)
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--last-seconds", type=float, default=0.0)
    ap.add_argument("--quality", type=int, default=7)   # imageio libx264 quality (lower = smaller file)
    a = ap.parse_args()

    session = Path(a.session)
    idx = [json.loads(l) for l in (session / "video_index.jsonl").read_text().splitlines() if l.strip()]
    if a.last_seconds > 0 and idx:
        t_end = max(r.get("recv_monotonic_ns", 0) for r in idx)
        idx = [r for r in idx if r.get("recv_monotonic_ns", 0) >= t_end - a.last_seconds * 1e9]
    blob = (session / "video.bin").read_bytes()
    S = a.scale; W, H = 640 * S, 360 * S

    frames = []
    for rec in idx:
        jpg = blob[rec["offset"]:rec["offset"] + rec["length"]]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        big = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        frames.append({"recv_ms": rec["recv_monotonic_ns"] / 1e6, "img": big})
    if len(frames) < 2:
        raise SystemExit("too few frames")

    t0, t_end = frames[0]["recv_ms"], frames[-1]["recv_ms"]
    out_dt = 1000.0 / a.out_fps
    step = out_dt / a.slowmo
    wr = imageio.get_writer(a.out_mp4, fps=a.out_fps, codec="libx264", quality=a.quality,
                            macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
    j = 0; t = t0
    while t <= t_end + 1e-6:
        while j + 1 < len(frames) and frames[j + 1]["recv_ms"] <= t:
            j += 1
        wr.append_data(cv2.cvtColor(frames[j]["img"], cv2.COLOR_BGR2RGB))
        t += step
    wr.close()
    print(f"wrote {a.out_mp4}  ({len(frames)} frames, slowmo {a.slowmo}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
