import json, numpy as np, sys
from pathlib import Path
from scipy.spatial.transform import Rotation
ROOT=Path("C:/Users/Fengy/Downloads/Projects/Anduril"); sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"rl"))
from racer.rl_plant import *
from racer.rl_plant import step as plant_step
import offline_rollout as OR
D=Path("handoff/shadowpc-refit-dataset-2026-06-12/extracted/20260612_044219_inc6_rollfix_f1")
rows=[json.loads(l) for l in open(D/"debug_obs.jsonl",encoding="utf-8") if json.loads(l).get("type")!="header"]
def eul(q): q=np.asarray(q,float); return np.degrees(Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler("ZYX"))[::-1] # r,p,y
def P(sign):
    return PlantParams(transport_delay_steps=0,rate_sign=np.array(sign,float),super_rate_s=SUPER_RATE_S_MEASURED,
      alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),linear_drag=0.0,quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
      coll_map_thr=COLL_MAP_THR_MEASURED.copy(),coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
      mixer_idle=MIXER_IDLE_MEASURED,mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
      mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
      lapse_speed=LAPSE_SPEED_MEASURED.copy(),lapse_factor=LAPSE_FACTOR_MEASURED.copy())
K0=36; DT=1/30
def replay(params):
    r0=rows[K0]; twin=PlantState(pos=np.array(r0["pos_ned"]),vel=np.array(r0["vel_ned"]),quat=np.array(r0["q_raw_wxyz"]),
              omega=np.array(r0["w_raw"]),thrust=np.asarray(r0["collective"])); cp=None; tr=[]
    for k in range(K0,K0+27):
        if cp is not None: twin=plant_step(twin,np.concatenate([cp,[clp]]),DT,params)
        cp=np.array(rows[k]["rate_frd"]); clp=rows[k]["collective"]; tr.append(twin)
    return tr
tr=replay(P([-1,1,-1]))
print("twin rate_sign=[-1,+1,-1] (current).  r/p/y = roll/pitch/yaw deg;  vE = East vel")
print(f"{'k':>3} | {'LIVE r':>6} {'p':>6} {'y':>6} {'vE':>6} | {'TWIN r':>6} {'p':>6} {'y':>6} {'vE':>6}")
for i,k in enumerate(range(K0,K0+27)):
    if k%3: continue
    L=eul(rows[k]["q_raw_wxyz"]); T=eul(tr[i].quat)
    lvE=rows[k]["vel_ned"][1]; tvE=tr[i].vel[1]
    print(f"{k:>3} | {L[0]:>6.0f} {L[1]:>6.0f} {L[2]:>6.0f} {lvE:>6.1f} | {T[0]:>6.0f} {T[1]:>6.0f} {T[2]:>6.0f} {tvE:>6.1f}")
