"""H4c: attitude + vertical specific-force comparison, twin(open-loop, live actions)
vs live telemetry. ASCII output only (cp1252 console).

Columns per tick:
  live rpy (true, artifact-undone) vs twin rpy
  live a_up = -(dvz/dt) + ... measured net vertical accel vs twin's
  tilt angle (angle between body -z thrust axis and world up)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import numpy as np

from racer.frames import R_world_from_body, euler_from_quat_wxyz
from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            COLL_MAP_THR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED, PlantParams,
                            QUAD_DRAG_C2_MEASURED, SUPER_RATE_S_MEASURED,
                            quat_rotate, step as plant_step)
from fly_rl import _ODO_RATE_SIGN
from offline_rollout import make_start

DT = 1.0 / 30.0

print("coll_map_thr  :", np.round(COLL_MAP_THR_MEASURED, 4).tolist())
print("coll_map_accel:", np.round(COLL_MAP_ACCEL_MEASURED, 2).tolist())


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


def live_true_R(q_raw):
    roll, pitch, yaw = euler_from_quat_wxyz(np.array(q_raw))
    return R_world_from_body(-roll, pitch, yaw), (-roll, pitch, yaw)


def twin_R(quat):
    return np.stack([quat_rotate(quat, e) for e in np.eye(3)], axis=-1)


def analyze(name, n=72, lat=2):
    rows = load_run(name)
    st, _ = make_start("simstart", A)
    p = params_mixer(lat)
    twin = []
    for k in range(min(n + 1, len(rows))):
        twin.append({"quat": st.quat.copy(), "vel": st.vel.copy(), "pos": st.pos.copy()})
        act = np.concatenate([rows[k]["rate_frd"], [rows[k]["collective"]]])
        st = plant_step(st, act, DT, p)

    print(f"\n==== {name} ====")
    print(f"{'k':>3} {'thr':>5} {'LIVE rpy deg':>24} {'TWIN rpy deg':>24} "
          f"{'cosL':>6} {'cosT':>6} {'aupL':>6} {'aupT':>6}")
    for k in range(0, n - 2, 3):
        r = rows[k]
        RL, rpyL = live_true_R(r["q_raw_wxyz"])
        RT = twin_R(twin[k]["quat"])
        # body -z in world frame, z-component (NED: up-thrust component = -R[2,2])
        cosL = float(RL[2, 2])
        cosT = float(RT[2, 2])
        # measured net vertical accel (NED, m/s^2): dvz/dt; a_up = g - dvz/dt
        vz0 = rows[max(k - 1, 0)]["vel_ned"][2]
        vz1 = rows[min(k + 1, len(rows) - 1)]["vel_ned"][2]
        dt = (rows[min(k + 1, len(rows) - 1)]["t_mono"] - rows[max(k - 1, 0)]["t_mono"]) or 1e-9
        aupL = 9.81 - (vz1 - vz0) / dt
        tz0 = twin[max(k - 1, 0)]["vel"][2]
        tz1 = twin[min(k + 1, n)]["vel"][2]
        aupT = 9.81 - (tz1 - tz0) / (2 * DT)
        dL = np.degrees(rpyL)
        qT = twin[k]["quat"]
        rT, pT, yT = euler_from_quat_wxyz(qT)  # twin quat is true: no artifact
        # note: euler_from_quat assumes reported convention; for the twin's true quat
        # the same extraction gives (roll, pitch, yaw) directly
        dT = np.degrees([rT, pT, yT])
        print(f"{k:>3} {r['collective']:>5.3f} "
              f"[{dL[0]:+7.1f},{dL[1]:+7.1f},{dL[2]:+7.1f}] "
              f"[{dT[0]:+7.1f},{dT[1]:+7.1f},{dT[2]:+7.1f}] "
              f"{cosL:>6.2f} {cosT:>6.2f} {aupL:>6.1f} {aupT:>6.1f}")


analyze("20260612_034852_inc6_standing_f1")
analyze("20260612_035014_inc6_standing_f2", n=48)
