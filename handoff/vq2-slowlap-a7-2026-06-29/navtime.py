"""Steady-state navigator cost on the A7 recorded frames, independent of the 15-tick crash.
Times estimate_heading (the A6 VP-RANSAC bottleneck) at the new flight iters=256 vs the old 2000,
plus the red_glow detector + pose, skipping cold-start warmup frames. Answers: did the loop work
per frame actually drop below the 33ms/30Hz budget at steady state?
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, cv2
from racer.contracts import Frame
from racer.vision.heading_vp import estimate_heading
from racer.vision.red_glow_detector import RedGlowGateDetector
from racer.vision.gate_pose import estimate_gate_pose

sess = Path(sys.argv[1])
idx = [json.loads(l) for l in (sess/"video_index.jsonl").read_text().splitlines() if l.strip()]
blob = (sess/"video.bin").read_bytes()
det = RedGlowGateDetector()

imgs = []
for rec in idx:
    img = cv2.imdecode(np.frombuffer(blob[rec["offset"]:rec["offset"]+rec["length"]], np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        imgs.append(img)
print(f"decoded {len(imgs)} frames")

def time_fn(fn, frames, warmup=5):
    ts = []
    for k, im in enumerate(frames):
        t0 = time.perf_counter()
        fn(im)
        dt = (time.perf_counter()-t0)*1e3
        if k >= warmup:
            ts.append(dt)
    a = np.array(ts)
    return a

# sample a manageable subset spread across the flight (every Nth) to keep it quick
sub = imgs[::max(1, len(imgs)//120)]
print(f"timing on {len(sub)} sampled frames (skip 5 warmup)\n")

for iters in (2000, 256):
    a = time_fn(lambda im: estimate_heading(im, 0.0, 0.0, ransac_iters=iters), sub)
    print(f"estimate_heading iters={iters:4d}: mean={a.mean():6.1f}ms  p50={np.median(a):6.1f}  p95={np.percentile(a,95):6.1f}  max={a.max():6.1f}")

ad = time_fn(lambda im: det.detect(Frame(0,0,im,0)), sub)
print(f"red_glow detect          : mean={ad.mean():6.1f}ms  p50={np.median(ad):6.1f}  p95={np.percentile(ad,95):6.1f}")

# combined per-frame proxy (detect + heading@256) — the dominant vision work in nav.update
comb = time_fn(lambda im: (det.detect(Frame(0,0,im,0)), estimate_heading(im,0,0,ransac_iters=256)), sub)
print(f"detect + heading@256     : mean={comb.mean():6.1f}ms  p50={np.median(comb):6.1f}  p95={np.percentile(comb,95):6.1f}")
print(f"\n33.3ms budget (30Hz): detect+heading@256 p95 {'<=' if np.percentile(comb,95)<=33.3 else '>'} budget")
