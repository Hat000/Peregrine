import json, numpy as np, sys
from pathlib import Path
from scipy.spatial.transform import Rotation
ROOT=Path("C:/Users/Fengy/Downloads/Projects/Anduril"); sys.path.insert(0,str(ROOT/"src"))
from racer.rl_plant import (COLL_MAP_THR_MEASURED,COLL_MAP_ACCEL_MEASURED,QUAD_DRAG_C2_MEASURED,
                            LAPSE_SPEED_MEASURED,LAPSE_FACTOR_MEASURED)
G=9.80665
def Rm(q): q=np.asarray(q,float); return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
for name in ["20260612_044219_inc6_rollfix_f1","20260612_044438_inc6_rollfix_f2"]:
    D=Path(f"handoff/shadowpc-refit-dataset-2026-06-12/extracted/{name}")
    rows=[json.loads(l) for l in open(D/"debug_obs.jsonl",encoding="utf-8") if json.loads(l).get("type")!="header"]
    t=np.array([r["t_mono"] for r in rows]); v=np.array([r["vel_ned"] for r in rows])
    print(f"\n=== {name.split('_',2)[2]}: live measured accel_E  vs  OUR-MODEL accel_E (from live attitude) ===")
    print(f"{'k':>3} {'sp':>5} {'roll':>5} {'meas_aE':>8} {'modl_aE':>8} {'thr_aE':>7} {'drag_aE':>7}")
    for k in range(39,63,3):
        dt=t[k+1]-t[k-1]
        if not (0.05<dt<0.09): continue
        meas_a=(v[k+1]-v[k-1])/dt
        R=Rm(rows[k]["q_raw_wxyz"]); c=rows[max(k-2,0)]["collective"]
        K=np.interp(c,COLL_MAP_THR_MEASURED,COLL_MAP_ACCEL_MEASURED)
        L=np.interp(np.linalg.norm(v[k]),LAPSE_SPEED_MEASURED,LAPSE_FACTOR_MEASURED)
        a_up=K*L
        thr=a_up*(R@np.array([0,0,-1.0]))
        vb=R.T@v[k]; c2=QUAD_DRAG_C2_MEASURED; cc=np.where(vb>=0,c2[:,0],c2[:,1])
        drag=R@(-cc*np.abs(vb)*vb)
        modl_a=thr+drag+np.array([0,0,G])
        roll=np.degrees(Rotation.from_matrix(R).as_euler("ZYX")[2])
        print(f"{k:>3} {np.linalg.norm(v[k]):>5.1f} {roll:>5.0f} {meas_a[1]:>8.2f} {modl_a[1]:>8.2f} {thr[1]:>7.2f} {drag[1]:>7.2f}")
