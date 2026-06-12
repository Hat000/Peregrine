"""H4b: tick-aligned realized-rate + vertical-force comparison, twin(open-loop, live
actions) vs live telemetry, for standing f1/f2/f6 and bridge f3.

If realized rates match but positions diverge -> attitude/collective-map gap.
If realized rates differ per axis -> rate-loop gap (gain/sign/slew/mixer).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import numpy as np

from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            COLL_MAP_THR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED, PlantParams,
                            QUAD_DRAG_C2_MEASURED, SUPER_RATE_S_MEASURED,
                            step as plant_step)
from fly_rl import _ODO_RATE_SIGN
from offline_rollout import make_start

DT = 1.0 / 30.0


def load_run(name):
    rows = []
    with open(ROOT / "data" / "runs" / name / "debug_obs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def params_mixer(lat):
    return PlantParams(transport_delay_steps=lat,
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


class A:
    gate = 0; handoff_dist = 3.0; handoff_speed = 10.0; thrust0 = -1.0


def analyze(name, n=70, lat=2):
    rows = load_run(name)
    st, _ = make_start("simstart", A)
    p = params_mixer(lat)
    twin = []
    for k in range(min(n, len(rows))):
        act = np.concatenate([rows[k]["rate_frd"], [rows[k]["collective"]]])
        twin.append({"omega": st.omega.copy(), "pos": st.pos.copy(), "vel": st.vel.copy()})
        st = plant_step(st, act, DT, p)

    print(f"\n==== {name} (twin open-loop, lat={lat}) ====")
    print(f"{'k':>3} {'cmd r/p/y,thr':>32} {'LIVE w_frd':>24} {'TWIN omega':>24} "
          f"{'LIVE vz':>8} {'TWIN vz':>8}")
    vz_prev_live = 0.0
    for k in range(0, min(n, len(rows)), 2):
        r = rows[k]
        c = r["rate_frd"]
        w_live = np.array(r["w_raw"]) * _ODO_RATE_SIGN
        w_twin = twin[k]["omega"]
        vz_live = r["vel_ned"][2]
        vz_twin = twin[k]["vel"][2]
        print(f"{k:>3} [{c[0]:+5.2f},{c[1]:+5.2f},{c[2]:+5.2f}],{r['collective']:.3f} "
              f"[{w_live[0]:+6.2f},{w_live[1]:+6.2f},{w_live[2]:+6.2f}] "
              f"[{w_twin[0]:+6.2f},{w_twin[1]:+6.2f},{w_twin[2]:+6.2f}] "
              f"{vz_live:>8.2f} {vz_twin:>8.2f}")

    # per-axis realized-rate agreement over the window
    w_live = np.array([np.array(r["w_raw"]) * _ODO_RATE_SIGN for r in rows[:n]])
    w_twin = np.array([t["omega"] for t in twin])
    m = min(len(w_live), len(w_twin))
    for ax, nm in enumerate(["roll", "pitch", "yaw"]):
        rmse = float(np.sqrt(np.mean((w_live[:m, ax] - w_twin[:m, ax]) ** 2)))
        # best gain fit live = g * twin
        denom = float(np.dot(w_twin[:m, ax], w_twin[:m, ax]))
        g = float(np.dot(w_live[:m, ax], w_twin[:m, ax]) / denom) if denom > 1e-9 else float("nan")
        print(f"  {nm:5s}: rmse(live-twin)={rmse:6.3f} rad/s   live≈{g:+.3f}×twin")


for nm in ["20260612_034852_inc6_standing_f1", "20260612_035014_inc6_standing_f2",
           "20260612_035415_inc6_standing_f6"]:
    analyze(nm)
