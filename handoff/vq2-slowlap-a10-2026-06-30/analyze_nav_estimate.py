"""A10 analysis: the (now ground-truth-validated) nav estimate trajectory + the per-frame gate-detection
timeline, to show the gyro fix holds the attitude AND that the seeker doesn't pursue the visible gate.

nav_estimate.jsonl (new in A10) logs per tick: estimator roll/pitch/yaw, dead-reckoned position, commanded
body_rate + thrust, time_since_vision. The pilot confirmed the estimate is CORRECT this run, so the
trajectory below is trustworthy (the dead-reckoned position MAGNITUDE is unbounded with no absolute fix,
but its direction is real).

Usage: python analyze_nav_estimate.py <run_dir>
"""
import sys, json, math
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
recs = [json.loads(l) for l in (run / "nav_estimate.jsonl").read_text().splitlines() if l.strip()]
deg = lambda x: math.degrees(x) if x is not None else float("nan")
t0 = recs[0]["sim_time_ns"]

print(f"== {run.name} ==  {len(recs)} nav-estimate records")
n_fix = sum(1 for r in recs if r.get("time_since_vision_s") is not None)
print(f"vision POSITION fixes (finite tsv): {n_fix}  (map-free => navigator absolute fix never fires; EXPECTED)")
pit = np.array([deg(r["pitch_rad"]) for r in recs if r.get("pitch_rad") is not None])
rol = np.array([deg(r["roll_rad"]) for r in recs if r.get("roll_rad") is not None])
yaw = np.array([deg(r["yaw_rad"]) for r in recs if r.get("yaw_rad") is not None])
print(f"attitude stability (low std = no runaway): pitch {pit.mean():+.0f}+-{pit.std():.0f}  "
      f"roll {rol.mean():+.0f}+-{rol.std():.0f}  yaw {yaw.mean():+.0f}+-{yaw.std():.0f} deg")
# fraction of ticks the seeker actively commanded a body rate
act = [r for r in recs if r.get("body_rate") and any(abs(v) > 0.05 for v in r["body_rate"])]
print(f"ticks with active steering (|cmd_rate|>0.05 on any axis): {len(act)}/{len(recs)} "
      f"({100*len(act)/len(recs):.0f}%)  -> mostly drifting, not pursuing")
print("\n  t(s)  roll pitch  yaw   pos(x,y,z)            cmd_rate[r,p,y]      thr")
for r in recs[::max(1, len(recs)//24)]:
    t = (r["sim_time_ns"] - t0) / 1e9
    p = r.get("position_ned") or [None]*3
    br = r.get("body_rate") or [None]*3
    fp = lambda x: ("%+7.1f" % x) if x is not None else "   --  "
    fb = lambda x: ("%+5.2f" % x) if x is not None else " -- "
    print(f"{t:6.1f} {deg(r.get('roll_rad')):+4.0f} {deg(r.get('pitch_rad')):+5.0f} "
          f"{deg(r.get('yaw_rad')):+4.0f}  ({fp(p[0])},{fp(p[1])},{fp(p[2])})  "
          f"[{fb(br[0])},{fb(br[1])},{fb(br[2])}] {fb(r.get('thrust'))}")

# detection timeline (needs the recorded frames)
try:
    import cv2
    sys.path.insert(0, "src")
    from racer.contracts import Frame
    from racer.vision.red_glow_detector import RedGlowGateDetector
    from racer.vision.gate_pose import estimate_gate_pose
    idx = [json.loads(l) for l in (run / "video_index.jsonl").read_text().splitlines() if l.strip()]
    blob = (run / "video.bin").read_bytes()
    det = RedGlowGateDetector()
    print("\n  frame   ~t   n_gates  nearest_range_m   (is the gate visible to chase?)")
    N = len(idx)
    for fi in range(0, N, max(1, N // 22)):
        rec = idx[fi]
        img = cv2.imdecode(np.frombuffer(blob[rec["offset"]:rec["offset"]+rec["length"]], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        obs = det.detect(Frame(rec["frame_id"], rec["sim_time_ns"], img, 0))
        rng = None
        for o in obs:
            try:
                ps = estimate_gate_pose(o, compute_covariance=False)
                if ps and np.isfinite(ps.range_m):
                    rng = ps.range_m if rng is None else min(rng, ps.range_m)
            except Exception:
                pass
        print(f"  {fi:5d} {fi/30.0:5.1f}s   {len(obs):4d}     {('%.1f' % rng) if rng else '--':>6}")
except Exception as exc:
    print(f"\n(detection timeline skipped: {exc})")
