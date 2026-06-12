"""Analyze the rollfix confirmation flights: tick-aligned diff vs the twin.

Twin emulation of the live sim: plant with rate_sign [-1,+1,-1] (measured), driven
OPEN-LOOP by the recorded wire commands. Compare realized TRUE rates (w_raw*[1,-1,1]),
attitude (raw quat euler), vz, pos. Also closed-loop FIXED reference positions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import numpy as np

from racer.frames import euler_from_quat_wxyz
from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            COLL_MAP_THR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED, PlantParams,
                            QUAD_DRAG_C2_MEASURED, SUPER_RATE_S_MEASURED,
                            step as plant_step)
from offline_rollout import make_start

DT = 1.0 / 30.0
S_REP = np.array([1.0, -1.0, 1.0])


def load_run(name):
    rows = []
    with open(ROOT / "data" / "runs" / name / "debug_obs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def params_live(lat):
    return PlantParams(transport_delay_steps=lat,
                       rate_sign=np.array([-1.0, 1.0, -1.0]),
                       super_rate_s=SUPER_RATE_S_MEASURED,
                       alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                       linear_drag=0.0,
                       quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                       coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                       coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
                       mixer_idle=MIXER_IDLE_MEASURED,
                       mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                       mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
                       mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)


def analyze(name, n=90, lat=2):
    rows = load_run(name)
    st, _ = make_start("simstart", type("A", (), {
        "gate": 0, "handoff_dist": 3.0, "handoff_speed": 10.0, "thrust0": -1.0}))
    p = params_live(lat)
    twin = []
    for k in range(min(n, len(rows))):
        twin.append({"quat": st.quat.copy(), "vel": st.vel.copy(),
                     "pos": st.pos.copy(), "omega": st.omega.copy()})
        act = np.concatenate([rows[k]["rate_frd"], [rows[k]["collective"]]])
        st = plant_step(st, act, DT, p)

    print(f"\n==== {name} (live-sim twin, open-loop recorded wire, lat={lat}) ====")
    print(f"{'k':>3} {'wire r/p/y,thr':>30} {'LIVE w_true':>23} {'TWIN omega':>23} "
          f"{'LIVE rpy':>23} {'TWIN rpy':>23} {'LIVE pos y,z':>14} {'TWIN pos y,z':>14}")
    for k in range(0, min(n, len(rows)), 3):
        r = rows[k]
        c = r["rate_frd"]
        wl = np.array(r["w_raw"]) * S_REP
        wt = twin[k]["omega"]
        rl = np.degrees(euler_from_quat_wxyz(np.array(r["q_raw_wxyz"])))
        rt = np.degrees(euler_from_quat_wxyz(twin[k]["quat"]))
        lp = r["pos_ned"]
        tp = twin[k]["pos"]
        print(f"{k:>3} [{c[0]:+5.2f},{c[1]:+5.2f},{c[2]:+5.2f}],{r['collective']:.2f} "
              f"[{wl[0]:+5.1f},{wl[1]:+5.1f},{wl[2]:+5.1f}] "
              f"[{wt[0]:+5.1f},{wt[1]:+5.1f},{wt[2]:+5.1f}] "
              f"[{rl[0]:+5.0f},{rl[1]:+5.0f},{rl[2]:+6.0f}] "
              f"[{rt[0]:+5.0f},{rt[1]:+5.0f},{rt[2]:+6.0f}] "
              f"{lp[1]:+6.2f},{lp[2]:+6.2f} {tp[1]:+6.2f},{tp[2]:+6.2f}")

    w_live = np.array([np.array(r["w_raw"]) * S_REP for r in rows[:n]])
    w_twin = np.array([t["omega"] for t in twin])
    m = min(len(w_live), len(w_twin))
    for ax, nm in enumerate(["roll", "pitch", "yaw"]):
        rmse = float(np.sqrt(np.mean((w_live[:m, ax] - w_twin[:m, ax]) ** 2)))
        denom = float(np.dot(w_twin[:m, ax], w_twin[:m, ax]))
        g = float(np.dot(w_live[:m, ax], w_twin[:m, ax]) / denom) if denom > 1e-9 else float("nan")
        print(f"  {nm:5s}: rmse(live-twin)={rmse:6.3f} rad/s   live ~ {g:+.3f} x twin")


analyze("20260612_044219_inc6_rollfix_f1")
