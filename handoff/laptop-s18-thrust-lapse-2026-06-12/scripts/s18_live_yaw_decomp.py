import json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
def load(name):
    D=Path(f"handoff/shadowpc-refit-dataset-2026-06-12/extracted/{name}")
    return [json.loads(l) for l in open(D/"debug_obs.jsonl",encoding="utf-8") if json.loads(l).get("type")!="header"]
def Rm(q): q=np.asarray(q,float); return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
for name in ["20260612_044219_inc6_rollfix_f1","20260612_044438_inc6_rollfix_f2"]:
    rows=load(name); vel=np.array([r["vel_ned"] for r in rows])
    print(f"\n=== {name.split('_',2)[2]} ===")
    print(f"{'k':>3} {'sp':>5} {'coll':>5} {'cmdR':>6} {'cmdP':>6} {'cmdY':>6} {'wR':>6} {'wP':>6} {'wY':>6} {'sslip':>6}")
    for k in range(39,72,3):
        if k>=len(rows): break
        r=rows[k]; R=Rm(r["q_raw_wxyz"]); vb=R.T@vel[k]
        cr=r["rate_frd"]; w=r["w_raw"]
        sslip=np.degrees(np.arctan2(vb[1],abs(vb[0])))  # sideslip angle
        print(f"{k:>3} {np.linalg.norm(vel[k]):>5.1f} {r['collective']:>5.2f} {cr[0]:>6.2f} {cr[1]:>6.2f} {cr[2]:>6.2f} {w[0]:>6.2f} {w[1]:>6.2f} {w[2]:>6.2f} {sslip:>6.0f}")
