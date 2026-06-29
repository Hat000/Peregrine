"""Attempt-5 analysis: detector replay (gate visibility/range) + tlog HIGHRES_IMU realized attitude
(accel tilt + gyro) over the first ~5 s, to tell whether the close-range failure is a REAL tumble or
an estimator-only divergence. Usage: python analyze_run.py <session_dir> <out_dir>
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import cv2
from pymavlink import mavutil
from racer.contracts import Frame
from racer.vision.red_glow_detector import RedGlowGateDetector

session = Path(sys.argv[1]); outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)

idx = [json.loads(l) for l in (session/"video_index.jsonl").read_text().splitlines() if l.strip()]
blob = (session/"video.bin").read_bytes()
det = RedGlowGateDetector()
rows = []
for i, rec in enumerate(idx):
    jpg = blob[rec["offset"]:rec["offset"]+rec["length"]]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    fr = Frame(frame_id=rec["frame_id"], sim_time_ns=rec["sim_time_ns"], image_bgr=img,
               recv_monotonic_ns=rec.get("recv_monotonic_ns", 0))
    obs = det.detect(fr)
    rows.append({"i": i, "frame_id": rec["frame_id"], "n_detections": len(obs)})
    if i < 24 or len(obs) > 0:
        cv2.imwrite(str(outdir/f"frame_{i:02d}_id{rec['frame_id']}_dets{len(obs)}.png"), img)
print(f"[detector] {len(rows)} frames; >=1 detection: {sum(1 for r in rows if r['n_detections']>0)}")

tlog = mavutil.mavlink_connection(str(session/"mavlink.tlog"))
imu = []; t0 = None
while True:
    m = tlog.recv_match(type=["HIGHRES_IMU"], blocking=False)
    if m is None:
        break
    t_us = int(m.time_usec)
    if t0 is None: t0 = t_us
    dt = (t_us - t0)/1e6
    if dt > 6.0: continue
    ax, ay, az = m.xacc, m.yacc, m.zacc
    pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
    roll = math.degrees(math.atan2(ay, -az))
    imu.append({"t_s": round(dt,3), "acc_pitch_deg": round(pitch,1), "acc_roll_deg": round(roll,1),
                "gyro_rps": [round(m.xgyro,3), round(m.ygyro,3), round(m.zgyro,3)],
                "amag": round(float(np.linalg.norm([ax,ay,az])),1)})
print(f"[tlog] {len(imu)} HIGHRES_IMU samples first 6s")
for s in imu[::max(1,len(imu)//32)]:
    print(f"  t={s['t_s']:.3f} acc_tilt(r,p)=({s['acc_roll_deg']:+.0f},{s['acc_pitch_deg']:+.0f})deg "
          f"gyro={s['gyro_rps']} |a|={s['amag']}")
if imu:
    gmax = max(imu, key=lambda s: max(abs(v) for v in s["gyro_rps"]))
    print(f"[tlog] peak |gyro| @t={gmax['t_s']}s -> {gmax['gyro_rps']} rps")
(outdir/"analysis.json").write_text(json.dumps({"detector": rows, "imu_first_6s": imu}, indent=2))
print(f"wrote {outdir/'analysis.json'}")
