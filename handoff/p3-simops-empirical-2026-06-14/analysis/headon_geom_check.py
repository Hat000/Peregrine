"""headon_geom_check.py — DETECTOR-FREE recording-geometry precondition check (P3, 2026-06-14).

Confirms the boresight recordings actually delivered HEAD-ON gate sightings (azimuth ~ 0) IN-FRAME across a
useful range span — the precondition for the boresight ε analysis — using ONLY GT pose (LPN) + attitude (ODO)
+ the surveyed map + the camera projection. NO detector, NO PnP, NO ε computed (that waits for batch 2).

Per video frame: nearest LPN (GT pos) + ODO (quat) by recv clock (== shadow_gate4.aligned_frames), then
predict_gates_in_camera -> for the target gate report azimuth=atan2(tx,tz), elevation=atan2(ty,tz),
range=|t_cam|, center_px, in_image. Head-on ⇒ |azimuth| small AND center_px_x ~ cx=320.
"""
from __future__ import annotations

import argparse, bisect, glob as gm, sys
from pathlib import Path
import numpy as np

def _setup(srcdir): sys.path.insert(0, str(srcdir))

def _nearest(keys, items, k):
    if not keys: return None
    i = bisect.bisect_left(keys, k)
    cands = [j for j in (i, i-1) if 0 <= j < len(keys)]
    return items[min(cands, key=lambda j: abs(keys[j]-k))]

def run(sess, gates, gate_id, F, predict, RecordingReader):
    reader = RecordingReader(sess); meta = reader.meta
    t0u, t0m = int(meta["t0_unix_ns"]), int(meta["t0_monotonic_ns"])
    lpn, odo = [], []
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED": lpn.append((float(msg._timestamp), [float(msg.x),float(msg.y),float(msg.z)]))
        elif t == "ODOMETRY": odo.append((float(msg._timestamp), [float(v) for v in msg.q]))
    lpn.sort(key=lambda r:r[0]); odo.sort(key=lambda r:r[0])
    lt=[r[0] for r in lpn]; ot=[r[0] for r in odo]
    rows=[]; seen=set()
    for e in reader.iter_video_index():
        fid=e["frame_id"]
        if fid in seen: continue
        seen.add(fid)
        recv=(t0u+(e["recv_monotonic_ns"]-t0m))/1e9
        lp=_nearest(lt,lpn,recv); od=_nearest(ot,odo,recv)
        if lp is None or od is None: continue
        drone=np.array(lp[1]); q=np.array(od[1]); R_wb=F.R_world_from_odo_quat_wxyz(q)
        pred=predict(gates, drone, R_wb); pg=pred.get(gate_id)
        if pg is None: continue
        t=np.asarray(pg.t_cam_gate, float)
        if t[2] <= 0: continue
        az=np.degrees(np.arctan2(t[0], t[2])); el=np.degrees(np.arctan2(t[1], t[2]))
        rng=float(np.linalg.norm(t)); cx,cy=pg.center_px
        inimg=bool(0<=cx<640 and 0<=cy<360)
        rows.append(dict(az=az, el=el, rng=rng, cx=float(cx), cy=float(cy), inimg=inimg))
    return rows

def stat(a):
    a=np.asarray(a,float)
    return (len(a), float(np.mean(a)), float(np.percentile(a,50)),
            float(np.percentile(np.abs(a),90)), float(a.min()), float(a.max())) if a.size else (0,)*6

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to a worktree src/ with racer")
    ap.add_argument("--map", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--gate", type=int, default=0)
    a=ap.parse_args()
    _setup(a.src)
    from racer import frames as F
    from racer.navigator import load_track_map
    from racer.vision.association import predict_gates_in_camera as predict
    from racer.recording import RecordingReader
    gates=load_track_map(a.map, corner_to_center=True)
    sessions=sorted(Path(p) for p in gm.glob(a.glob))
    print(f"head-on geom check · gate-{a.gate} · {len(sessions)} sessions (corner_to_center=True)")
    allrows=[]
    for s in sessions:
        rows=run(s, gates, a.gate, F, predict, RecordingReader)
        infov=[r for r in rows if r["inimg"]]
        allrows+=rows
        if not rows: print(f"  {s.name:42s} N=0"); continue
        az=[r['az'] for r in infov] or [r['az'] for r in rows]
        rg=[r['rng'] for r in infov] or [r['rng'] for r in rows]
        cxs=[r['cx'] for r in infov] or [0]
        print(f"  {s.name:42s} N={len(rows):4d} inFoV={len(infov):4d}({100*len(infov)/len(rows):3.0f}%) "
              f"|az|p50={np.percentile(np.abs(az),50):5.1f}deg az.p50={np.percentile(az,50):+5.1f} "
              f"cx.med={np.median(cxs):5.0f} rng[{min(rg):4.1f},{np.percentile(rg,50):4.1f},{max(rg):4.1f}]m")
    infov=[r for r in allrows if r["inimg"]]
    if infov:
        print(f"\n  POOLED in-FoV N={len(infov)}: "
              f"az mean {np.mean([r['az'] for r in infov]):+.2f} |az|p50 {np.percentile([abs(r['az']) for r in infov],50):.1f}deg "
              f"|az|p90 {np.percentile([abs(r['az']) for r in infov],90):.1f}deg ; "
              f"el p50 {np.percentile([r['el'] for r in infov],50):+.1f}deg ; "
              f"cx.med {np.median([r['cx'] for r in infov]):.0f} (cx=320 head-on) ; "
              f"range [{min(r['rng'] for r in infov):.1f}, {max(r['rng'] for r in infov):.1f}] m")
    return 0

if __name__=="__main__": raise SystemExit(main())
