"""Render 'what the telemetry vision saw' for an attempt-4 run: the live VQ2 camera feed with the
red_glow detector's gate overlay (corners + quad) and the PnP range it would track. Slowed ~3.75x
(8 fps from a ~30 fps stream) and upscaled 2x for visibility.

Usage: python make_vision_video.py <session_dir> <out_mp4>
"""
import sys
from pathlib import Path
sys.path.insert(0, "src")
import json
import numpy as np
import cv2
from racer.contracts import Frame
from racer.vision.red_glow_detector import RedGlowGateDetector
from racer.vision.gate_pose import estimate_gate_pose

session = Path(sys.argv[1]); out = Path(sys.argv[2])
idx = [json.loads(l) for l in (session/"video_index.jsonl").read_text().splitlines() if l.strip()]
blob = (session/"video.bin").read_bytes()
det = RedGlowGateDetector()

SCALE = 2
W, H = 640*SCALE, 360*SCALE
FPS = 8
vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
if not vw.isOpened():
    raise SystemExit("VideoWriter failed to open (codec mp4v)")

t0 = idx[0]["sim_time_ns"]
for i, rec in enumerate(idx):
    jpg = blob[rec["offset"]:rec["offset"]+rec["length"]]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    fr = Frame(frame_id=rec["frame_id"], sim_time_ns=rec["sim_time_ns"], image_bgr=img,
               recv_monotonic_ns=rec.get("recv_monotonic_ns", 0))
    obs = det.detect(fr)
    big = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    best_rng = None
    for o in obs:
        pts = (np.asarray(o.corners_px, dtype=np.float32) * SCALE).astype(np.int32)
        cv2.polylines(big, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        for (x, y) in pts:
            cv2.circle(big, (int(x), int(y)), 4, (0, 200, 255), -1)
        try:
            pose = estimate_gate_pose(o, compute_covariance=False)
            if pose is not None and np.isfinite(pose.range_m):
                r = float(pose.range_m)
                if best_rng is None or r < best_rng:
                    best_rng = r
                c = pts.mean(axis=0).astype(int)
                cv2.putText(big, f"{r:.1f}m", (c[0]-20, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass
    t_rel = (rec["sim_time_ns"] - t0) / 1e9
    hdr = f"VQ2 cam  frame {i:02d}/{len(idx)-1}  t+{t_rel:4.2f}s  dets={len(obs)}"
    if best_rng is not None:
        hdr += f"  trkR~{best_rng:.1f}m"
    cv2.rectangle(big, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(big, hdr, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    vw.write(big)

vw.release()
print(f"wrote {out}  ({len(idx)} frames @ {FPS} fps, {W}x{H})")
