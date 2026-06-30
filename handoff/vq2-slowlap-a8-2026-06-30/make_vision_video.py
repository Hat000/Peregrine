"""FAITHFUL onboard-vision video: plays back on the real recv clock so dropped-frame holes show as
honest freezes + a STREAM GAP banner (never a surge). Overlays the red_glow detector (gate quad +
corners + PnP range). Slowed for viewing. Usage: python make_vision_video.py <session> <out.mp4> [--slowmo 4]
A8: 0% frame drops (perf fix) -> smooth, no STREAM GAP banners. run1 shows the egress + forward dive down
the lit VQ2 warehouse toward the red gates; the crash (~110m) is at the end.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, cv2
from racer.contracts import Frame
from racer.vision.red_glow_detector import RedGlowGateDetector
from racer.vision.gate_pose import estimate_gate_pose

ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("out")
ap.add_argument("--slowmo", type=float, default=4.0); ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()
sess = Path(a.session)
idx = [json.loads(l) for l in (sess/"video_index.jsonl").read_text().splitlines() if l.strip()]
blob = (sess/"video.bin").read_bytes()
det = RedGlowGateDetector()
S = 2; W, H = 640*S, 360*S
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
frames = []
for rec in idx:
    img = cv2.imdecode(np.frombuffer(blob[rec["offset"]:rec["offset"]+rec["length"]], np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    obs = det.detect(Frame(frame_id=rec["frame_id"], sim_time_ns=rec["sim_time_ns"], image_bgr=img,
                           recv_monotonic_ns=rec.get("recv_monotonic_ns", 0)))
    big = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    best = None
    for o in obs:
        pts = (np.asarray(o.corners_px, np.float32)*S).astype(np.int32)
        cv2.polylines(big, [pts.reshape(-1,1,2)], True, (0,255,0), 2, cv2.LINE_AA)
        for x,y in pts: cv2.circle(big,(int(x),int(y)),4,(0,200,255),-1)
        try:
            pose = estimate_gate_pose(o, compute_covariance=False)
            if pose is not None and np.isfinite(pose.range_m):
                r=float(pose.range_m); best=r if best is None else min(best,r)
                c=pts.mean(0).astype(int); cv2.putText(big,f"{r:.1f}m",(c[0]-20,c[1]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
        except Exception: pass
    frames.append({"recv": rec["recv_monotonic_ns"]/1e6, "fid": rec["frame_id"], "img": big, "dets": len(obs), "best": best})

t0=frames[0]["recv"]; tend=frames[-1]["recv"]; step=(1000.0/a.fps)/a.slowmo
j=0; dropped=0; t=t0
while t<=tend+1e-6:
    while j+1<len(frames) and frames[j+1]["recv"]<=t:
        dropped += frames[j+1]["fid"]-frames[j]["fid"]-1; j+=1
    cur=frames[j]; canvas=cur["img"].copy()
    cv2.rectangle(canvas,(0,0),(W,26),(0,0,0),-1)
    hdr=f"VQ2 cam t+{(cur['recv']-t0)/1000:5.2f}s fid={cur['fid']} dets={cur['dets']}"+(f" R~{cur['best']:.1f}m" if cur['best'] else "")+f"  dropped:{dropped}"
    cv2.putText(canvas,hdr,(8,19),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    nxt=frames[j+1] if j+1<len(frames) else None
    if nxt and (nxt["recv"]-cur["recv"])>60 and (t-cur["recv"])>60:
        cv2.rectangle(canvas,(0,H-30),(W,H),(0,0,90),-1)
        cv2.putText(canvas,f"STREAM GAP held {t-cur['recv']:.0f}ms (~{nxt['fid']-cur['fid']-1} frames dropped)",(8,H-9),cv2.FONT_HERSHEY_SIMPLEX,0.6,(80,80,255),2,cv2.LINE_AA)
    vw.write(canvas); t+=step
vw.release()
print(f"wrote {a.out} ({len(frames)} captured / {frames[-1]['fid']-frames[0]['fid']+1} sent, {dropped} dropped)")
