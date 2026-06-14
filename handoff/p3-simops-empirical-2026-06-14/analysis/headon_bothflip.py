"""headon_bothflip.py — IPPE both-flip inspection for the boresight FORM (P3 closeout, 2026-06-14).

For gate-0 head-on sightings across range, exposes BOTH cv2.SOLVEPNP_IPPE_SQUARE solutions, computes each
flip's rel_vert vs GT (full production world-fix chain), identifies the GT-correct flip (t_cam closest to the
GT-predicted gate translation), and reports per range bin:
  - rv_loreproj : rel_vert of the LOWER-reprojection flip = what a deploy estimator WITHOUT a GT prior picks
  - rv_GTcorrect: rel_vert of the GT-correct flip = what the shadow chain (GT-prior disambiguated) used in the rows
  - flip_spread : |rv_loreproj - rv_hireproj| (how far apart the two solutions place the gate vertically)
  - ambig e2/e1 : reprojection-error ratio (≈1 = genuinely ambiguous near-frontal; large = one flip clearly better)
  - sel==GT %   : how often lowest-reproj == GT-correct (deploy-time flip reliability)
The far 23 m anchor (weak perspective) is the suspect: if ambig≈1 + flip_spread large + sel==GT low there, the
production-deploy flip is unreliable at the anchor and rv_loreproj ≠ rv_GTcorrect — the commander's exact worry.
"""
from __future__ import annotations
import argparse, bisect, glob as gm, statistics as st, sys
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--map", required=True)
    ap.add_argument("--glob", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--gate", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    a=ap.parse_args(); sys.path.insert(0, a.src)
    import cv2
    from racer import frames as F
    from racer.contracts import Frame, GatePose
    from racer.frames import CAMERA_INTRINSICS_K
    from racer.localization import gate_pose_to_world_position
    from racer.navigator import NavigatorConfig, load_track_map
    from racer.recording import RecordingReader
    from racer.vision.association import associate_scored, predict_gates_in_camera
    from racer.vision.detector import GateDetector
    from racer.vision.gate_pose import _solve, gate_object_points, _ordered_corners
    cfg=NavigatorConfig(); gates=load_track_map(a.map, corner_to_center=True)
    g={x.gate_id:x for x in gates}[a.gate]; R_g2w=np.asarray(g.R_world_gate,float)
    det=GateDetector.load(Path(a.weights), score_thresh=0.25, kpt_conf_thresh=0.5)
    objp=gate_object_points()
    sessions=sorted(Path(p) for p in gm.glob(a.glob))
    if a.limit: sessions=sessions[:a.limit]
    def near(keys,items,k):
        if not keys: return None
        i=bisect.bisect_left(keys,k); c=[j for j in (i,i-1) if 0<=j<len(keys)]
        return items[min(c,key=lambda j:abs(keys[j]-k))]
    rows=[]
    for sess in sessions:
        reader=RecordingReader(sess); meta=reader.meta
        t0u,t0m=int(meta["t0_unix_ns"]),int(meta["t0_monotonic_ns"])
        lpn,odo=[],[]
        for m in reader.iter_mavlink():
            t=m.get_type()
            if t=="LOCAL_POSITION_NED": lpn.append((float(m._timestamp),[float(m.x),float(m.y),float(m.z)]))
            elif t=="ODOMETRY": odo.append((float(m._timestamp),[float(v) for v in m.q]))
        lpn.sort(key=lambda r:r[0]); odo.sort(key=lambda r:r[0]); lt=[r[0] for r in lpn]; ot=[r[0] for r in odo]
        vid=open(sess/"video.bin","rb"); seen=set()
        for e in reader.iter_video_index():
            fid=e["frame_id"]
            if fid in seen: continue
            seen.add(fid)
            recv=(t0u+(e["recv_monotonic_ns"]-t0m))/1e9
            lp=near(lt,lpn,recv); od=near(ot,odo,recv)
            if lp is None or od is None: continue
            drone=np.array(lp[1]); q=np.array(od[1]); R_wb=F.R_world_from_odo_quat_wxyz(q)
            pred=predict_gates_in_camera(gates, drone, R_wb); pg=pred.get(a.gate)
            if pg is None or pg.t_cam_gate[2]<=0: continue
            rng=float(np.linalg.norm(g.position_ned-drone))
            if rng>26: continue
            vid.seek(e["offset"]); jpeg=vid.read(e["length"])
            img=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
            fr=Frame(frame_id=fid,sim_time_ns=int(e["sim_time_ns"]),image_bgr=img,recv_monotonic_ns=0,jpeg_bytes=None)
            best=None
            for o in det.detect(fr):
                sc=associate_scored(o,pred)
                if sc is None or sc[0]!=a.gate: continue
                if best is None or sc[1]<best[0]: best=(sc[1],o)
            if best is None: continue
            o=best[1]
            corners,ids,conf=_ordered_corners(o)
            if corners.shape[0]!=4: continue
            cands=_solve(objp, corners, CAMERA_INTRINSICS_K)
            if len(cands)<2: continue
            t_gt=np.asarray(pg.t_cam_gate,float); fl=[]
            for (R,t,reproj) in cands:
                gp=GatePose(fid,int(e["sim_time_ns"]),R,t,float(reproj),gate_id=a.gate,n_corners=4)
                pos,_=gate_pose_to_world_position(gp,g,R_wb,attitude_noise_std=cfg.attitude_noise_std,
                                                  fix_cov_floor_std=cfg.fix_cov_floor_std)
                fl.append((float(reproj), float(R_g2w[:,1]@(pos-drone)), float(np.linalg.norm(t-t_gt))))
            fl.sort(key=lambda x:x[0])  # [0]=lower reproj
            gt=min(fl,key=lambda x:x[2])  # GT-correct = t closest to GT
            rows.append(dict(rng=rng, rv_lo=fl[0][1], rv_hi=fl[1][1], rv_gt=gt[1],
                             spread=abs(fl[0][1]-fl[1][1]), ambig=fl[1][0]/(fl[0][0]+0.1),
                             sel_is_gt=abs(fl[0][1]-gt[1])<1e-9))
        vid.close()
    print(f"both-flip · gate-{a.gate} · N={len(rows)} (sessions={len(sessions)})")
    print(f"{'bin(m)':>8} {'N':>4} {'rv_loreproj':>11} {'rv_GTcorrect':>12} {'flip_spread':>11} {'ambig e2/e1':>11} {'sel==GT%':>8}")
    for lo,hi in [(10,14),(14,18),(18,22),(22,26)]:
        b=[r for r in rows if lo<=r["rng"]<hi]
        if len(b)<3: continue
        print(f"{f'{lo}-{hi}':>8} {len(b):>4} {st.mean([r['rv_lo'] for r in b]):>+11.3f} {st.mean([r['rv_gt'] for r in b]):>+12.3f} "
              f"{st.mean([r['spread'] for r in b]):>11.3f} {st.median([r['ambig'] for r in b]):>11.1f} {100*sum(r['sel_is_gt'] for r in b)/len(b):>7.0f}%")

if __name__=="__main__": raise SystemExit(main())
