"""Tick-aligned obs/action diff: live rollfix_f1 vs the FIXED counterfactual
(emulated live plant, lat 2). Localize the first materially diverging obs dim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import numpy as np
import torch

from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            COLL_MAP_THR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED, PlantParams,
                            QUAD_DRAG_C2_MEASURED, SUPER_RATE_S_MEASURED,
                            step as plant_step)
import fly_rl
from fly_rl import OBS_LABELS, _HOVER_THRUST, build_obs, load_actor, policy_step
from offline_rollout import make_start

CKPT = str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth")
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


def main():
    actor = load_actor(CKPT)
    live = load_run("20260612_044219_inc6_rollfix_f1")
    st, gate = make_start("simstart", type("A", (), {
        "gate": 0, "handoff_dist": 3.0, "handoff_speed": 10.0, "thrust0": -1.0}))
    p = params_live(2)
    last = 0.0
    N = 70
    print(f"{'k':>3} {'maxdiff':>8} {'dim':>14} {'live_val':>9} {'cf_val':>9} "
          f"{'live act r/p/y,thr':>28} {'cf act':>28}")
    for k in range(N):
        tel = SimpleNamespace(position_ned=st.pos.copy(), velocity_ned=st.vel.copy(),
                              orientation_ned_wxyz=st.quat.copy(),
                              angular_rate_body=st.omega * S_REP)
        obs_cf = build_obs(tel, gate, last, virtual_flip=True)
        rate_frd, coll, last = policy_step(actor, obs_cf, 0.0, virtual_flip=True)
        coll = float(np.clip(last * _HOVER_THRUST, 0.0, 1.0))
        if k < len(live):
            r = live[k]
            ol = np.array(r["obs"])
            d = np.abs(ol - np.asarray(obs_cf, dtype=np.float64))
            wd = int(np.argmax(d))
            la = r["rate_frd"] + [r["collective"]]
            ca = list(np.round(rate_frd, 2)) + [round(coll, 3)]
            if k < 12 or k % 3 == 0:
                print(f"{k:>3} {d.max():>8.3f} {OBS_LABELS[wd]:>14} {ol[wd]:>9.3f} "
                      f"{float(obs_cf[wd]):>9.3f} "
                      f"[{la[0]:+5.2f},{la[1]:+5.2f},{la[2]:+5.2f}],{la[3]:.3f} "
                      f"[{ca[0]:+5.2f},{ca[1]:+5.2f},{ca[2]:+5.2f}],{ca[3]:.3f}")
        st = plant_step(st, np.concatenate([rate_frd, [coll]]), DT, p)


main()
