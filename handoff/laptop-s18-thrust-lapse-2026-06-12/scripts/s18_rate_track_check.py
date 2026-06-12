import json, numpy as np
from pathlib import Path
G0=np.array([2.501,2.504,2.231]); S=0.30; SIGN=np.array([-1.,1.,-1.])  # command->rate (live)
def g(c): 
    c=np.abs(c); return G0/(1-S*np.minimum(c,np.pi)/np.pi)
D=Path("handoff/shadowpc-refit-dataset-2026-06-12/extracted/20260612_044219_inc6_rollfix_f1")
rows=[json.loads(l) for l in open(D/"debug_obs.jsonl",encoding="utf-8") if json.loads(l).get("type")!="header"]
print("Predicted steady realized rate = g(|cmd|)*sign*cmd  vs  actual w_raw (true rate)")
print(f"{'k':>3} {'sp':>5} {'--- PITCH ---':>16} {'--- YAW ---':>14}")
print(f"{'':>3} {'':>5} {'cmd':>5} {'pred':>6} {'actual':>7} {'cmd':>6} {'pred':>6} {'actual':>7}")
vel=np.array([r['vel_ned'] for r in rows])
for k in range(33,69,3):
    if k>=len(rows): break
    cr=np.array(rows[max(k-2,0)]["rate_frd"])  # 2-tick transport delay
    pred=g(cr)*SIGN*cr
    w=rows[k]["w_raw"]
    sp=np.linalg.norm(vel[k])
    print(f"{k:>3} {sp:>5.1f} {cr[1]:>5.2f} {pred[1]:>6.2f} {w[1]:>7.2f}   {cr[2]:>6.2f} {pred[2]:>6.2f} {w[2]:>7.2f}")
