"""Attempt-2 launch analysis: (a) replay the red_glow detector on the recorded frames to confirm
the start gate was VISIBLE at launch, and (b) decode the tlog HIGHRES_IMU for the first ~2.5 s to
show the realized body-rate / acceleration tumble. Writes JSON + saves a few frames.

Usage: python analyze_run.py <session_dir> <out_dir>
"""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import cv2
from pymavlink import mavutil
from racer.contracts import Frame
from racer.vision.red_glow_detector import RedGlowGateDetector

session = Path(sys.argv[1]); outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)

# ---------- (a) detector replay ----------
idx = [json.loads(l) for l in (session/"video_index.jsonl").read_text().splitlines() if l.strip()]
blob = (session/"video.bin").read_bytes()
det = RedGlowGateDetector()
det_rows = []
print(f"[detector] {len(idx)} frames")
for i, rec in enumerate(idx):
    jpg = blob[rec["offset"]: rec["offset"]+rec["length"]]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    fr = Frame(frame_id=rec["frame_id"], sim_time_ns=rec["sim_time_ns"], image_bgr=img,
               recv_monotonic_ns=rec.get("recv_monotonic_ns", 0))
    obs = det.detect(fr)
    b, g, r = img[...,0].mean(), img[...,1].mean(), img[...,2].mean()
    det_rows.append({"i": i, "frame_id": rec["frame_id"], "n_detections": len(obs),
                     "mean_bgr": [round(b,1),round(g,1),round(r,1)]})
    if i < 12 or len(obs) > 0:
        print(f"  f{i:2d} id={rec['frame_id']} dets={len(obs)} meanBGR=({b:.0f},{g:.0f},{r:.0f})")
    if i in (0,1,2,len(idx)//2,len(idx)-1) or len(obs) > 0:
        cv2.imwrite(str(outdir/f"frame_{i:02d}_id{rec['frame_id']}_dets{len(obs)}.png"), img)
ndet = sum(1 for r in det_rows if r["n_detections"]>0)
print(f"[detector] frames with >=1 detection: {ndet}/{len(det_rows)}")

# ---------- (b) tlog HIGHRES_IMU first ~2.5 s ----------
tlog = mavutil.mavlink_connection(str(session/"mavlink.tlog"))
imu = []
t0 = None
while True:
    m = tlog.recv_match(type=["HIGHRES_IMU"], blocking=False)
    if m is None:
        break
    t_us = int(m.time_usec)
    if t0 is None:
        t0 = t_us
    dt = (t_us - t0)/1e6
    if dt > 2.5:
        continue
    imu.append({"t_s": round(dt,3),
                "gyro_rps": [round(m.xgyro,3), round(m.ygyro,3), round(m.zgyro,3)],
                "acc_mps2": [round(m.xacc,2), round(m.yacc,2), round(m.zacc,2)]})
print(f"[tlog] HIGHRES_IMU samples in first 2.5s: {len(imu)}")
# print a downsampled view (every ~10th) + the peak gyro
for s in imu[::max(1,len(imu)//25)]:
    print(f"  t={s['t_s']:.3f}s gyro={s['gyro_rps']} acc={s['acc_mps2']}")
if imu:
    gmax = max(imu, key=lambda s: max(abs(v) for v in s["gyro_rps"]))
    amax = max(imu, key=lambda s: abs(s["acc_mps2"][2]))
    print(f"[tlog] peak |gyro| @ t={gmax['t_s']}s -> {gmax['gyro_rps']} rps")
    print(f"[tlog] peak |zacc| @ t={amax['t_s']}s -> {amax['acc_mps2']} m/s^2")

(outdir/"analysis.json").write_text(json.dumps(
    {"detector": det_rows, "imu_first_2_5s": imu}, indent=2))
print(f"wrote {outdir/'analysis.json'}")
