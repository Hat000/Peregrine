"""Offline replay: decode recorded VQ2 launch frames and run the red_glow detector on each,
so we can see WHETHER perception fires while the drone is inside the start gate (tsv=inf cause).

Usage: python replay_detector.py <run_session_dir> <out_dir>
"""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import cv2
from racer.contracts import Frame
from racer.vision.red_glow_detector import RedGlowGateDetector

session = Path(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

idx = [json.loads(l) for l in (session / "video_index.jsonl").read_text().splitlines() if l.strip()]
blob = (session / "video.bin").read_bytes()
det = RedGlowGateDetector()

print(f"frames in recording: {len(idx)}")
results = []
for i, rec in enumerate(idx):
    jpg = blob[rec["offset"]: rec["offset"] + rec["length"]]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  frame {i}: DECODE FAIL")
        continue
    fr = Frame(frame_id=rec["frame_id"], sim_time_ns=rec["sim_time_ns"],
               image_bgr=img, recv_monotonic_ns=rec.get("recv_monotonic_ns", 0))
    obs = det.detect(fr)
    # brightness/red stats to characterize the view
    b, g, r = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    redmask = (img[..., 2].astype(int) > 120) & (img[..., 2].astype(int) - img[..., 1] > 40) & \
              (img[..., 2].astype(int) - img[..., 0] > 40)
    red_frac = float(redmask.mean())
    n = len(obs)
    rng = [f"{o.frame_id}" for o in obs]
    print(f"  frame {i:2d} id={rec['frame_id']} dets={n} "
          f"meanBGR=({b:.0f},{g:.0f},{r:.0f}) red_frac={red_frac:.3f}")
    results.append({"i": i, "frame_id": rec["frame_id"], "n_detections": n,
                    "mean_bgr": [round(b, 1), round(g, 1), round(r, 1)],
                    "red_frac": round(red_frac, 4)})
    # save first, middle, last + any frame with detections
    if i in (0, len(idx)//2, len(idx)-1) or n > 0:
        cv2.imwrite(str(outdir / f"frame_{i:02d}_id{rec['frame_id']}_dets{n}.png"), img)

ndet = sum(1 for r in results if r["n_detections"] > 0)
print(f"\nframes with >=1 detection: {ndet}/{len(results)}")
(outdir / "detector_replay.json").write_text(json.dumps(results, indent=2))
print(f"wrote {outdir/'detector_replay.json'} + sample frames")
