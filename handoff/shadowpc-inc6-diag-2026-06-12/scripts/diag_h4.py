"""SHADOWPC-INC6-DIAG H4: plant-gap forensics.

A) Closed-loop twin (mixer, lat2) vs live, tick-aligned pos comparison.
B) OPEN-LOOP: live-recorded action sequence driven through the twin from simstart —
   removes policy feedback; divergence onset/axis localizes the plant component.
C) Rate tracking live: commanded rate_frd vs realized FRD body rate (w_raw*sign),
   per-axis lag + gain; same statistic for the twin under its own commands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
from fly_rl import _HOVER_THRUST, _ODO_RATE_SIGN, load_actor, policy_step
from offline_rollout import make_start, obs_from_truth

CKPT = str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth")
DT = 1.0 / 30.0


def load_run(name):
    rows = []
    with open(ROOT / "data" / "runs" / name / "debug_obs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def mixer_params(lat):
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


def closed_loop(actor, lat, n):
    st, gate = make_start("simstart", A)
    params = mixer_params(lat)
    last = 0.0
    out = []
    for k in range(n):
        obs = obs_from_truth(st, gate, last, True)
        rate_frd, coll, last = policy_step(actor, obs, 0.0, virtual_flip=True)
        coll = last * _HOVER_THRUST
        out.append({"pos": st.pos.copy(), "omega": st.omega.copy(),
                    "act": np.concatenate([rate_frd, [coll]])})
        st = plant_step(st, np.concatenate([rate_frd, [coll]]), DT, params)
    return out


def open_loop(rows, lat, n):
    st, _ = make_start("simstart", A)
    params = mixer_params(lat)
    out = []
    for k in range(min(n, len(rows))):
        act = np.concatenate([rows[k]["rate_frd"], [rows[k]["collective"]]])
        out.append({"pos": st.pos.copy(), "omega": st.omega.copy()})
        st = plant_step(st, act, DT, params)
    return out


def main():
    actor = load_actor(CKPT)
    live = load_run("20260612_034852_inc6_standing_f1")
    N = 90

    cl = closed_loop(actor, 2, N)
    ol = open_loop(live, 2, N)

    print("tick-aligned positions, standing_f1 (NED):")
    print(f"{'k':>3} {'LIVE pos':>28} {'TWIN closed-loop':>28} {'TWIN open-loop(live acts)':>28} {'|live-ol|':>9}")
    for k in range(0, N, 3):
        lp = np.array(live[k]["pos_ned"])
        cp = cl[k]["pos"]
        op = ol[k]["pos"]
        d = np.linalg.norm(lp - op)
        print(f"{k:>3} [{lp[0]:+7.2f},{lp[1]:+7.2f},{lp[2]:+7.2f}] "
              f"   [{cp[0]:+7.2f},{cp[1]:+7.2f},{cp[2]:+7.2f}] "
              f"   [{op[0]:+7.2f},{op[1]:+7.2f},{op[2]:+7.2f}] {d:>9.2f}")

    # C) rate tracking: live realized vs commanded, lag scan
    cmd = np.array([r["rate_frd"] for r in live], dtype=np.float64)
    w_frd = np.array([np.array(r["w_raw"]) * _ODO_RATE_SIGN for r in live])
    n = min(len(cmd), 600)
    cmd, w_frd = cmd[:n], w_frd[:n]
    print("\nLIVE rate tracking (first %d ticks), per-axis corr & gain at lag L:" % n)
    for ax, name in enumerate(["roll", "pitch", "yaw"]):
        best = None
        for L in range(0, 8):
            c = cmd[: n - L, ax]
            w = w_frd[L:n, ax]
            if np.std(c) < 1e-6:
                continue
            r = float(np.corrcoef(c, w)[0, 1])
            g = float(np.dot(w, c) / np.dot(c, c))
            if best is None or r > best[1]:
                best = (L, r, g)
        print(f"  {name:5s}: best lag={best[0]} ticks  corr={best[1]:+.3f}  gain={best[2]:+.3f}")

    # twin closed-loop same statistic (its own commands, known-good reference)
    cmd_t = np.array([r["act"][:3] for r in cl])
    w_t = np.array([r["omega"] for r in cl])
    print("TWIN closed-loop rate tracking:")
    for ax, name in enumerate(["roll", "pitch", "yaw"]):
        best = None
        for L in range(0, 8):
            c = cmd_t[: N - L, ax]
            w = w_t[L:N, ax]
            if np.std(c) < 1e-6:
                continue
            r = float(np.corrcoef(c, w)[0, 1])
            g = float(np.dot(w, c) / np.dot(c, c))
            if best is None or r > best[1]:
                best = (L, r, g)
        print(f"  {name:5s}: best lag={best[0]} ticks  corr={best[1]:+.3f}  gain={best[2]:+.3f}")

    # where does open-loop part from live? per-axis first 1m divergence
    d = np.array([np.abs(np.array(live[k]["pos_ned"]) - ol[k]["pos"]) for k in range(N)])
    for ax, name in enumerate(["x", "y", "z"]):
        k1 = int(np.argmax(d[:, ax] > 1.0)) if (d[:, ax] > 1.0).any() else -1
        print(f"open-loop |live-twin| {name} crosses 1 m at tick {k1}")

    # realized vs commanded rates early ticks table
    print("\nlive realized FRD rates vs commanded (2-tick shifted), first 36 ticks:")
    print(f"{'k':>3} {'cmd r/p/y':>26} {'realized r/p/y (k)':>26}")
    for k in range(0, 36, 2):
        c = cmd[k]
        w = w_frd[k]
        print(f"{k:>3} [{c[0]:+5.2f},{c[1]:+5.2f},{c[2]:+5.2f}]    [{w[0]:+5.2f},{w[1]:+5.2f},{w[2]:+5.2f}]")


if __name__ == "__main__":
    main()
