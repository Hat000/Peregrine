"""Open-loop replay ACCEPTANCE test (audit fix): seed the twin at the live TRUE
state (CONJ_Y attitude, omega = -w_raw), rate_sign=[+1,+1,+1] (no command
inversion in the true frame), drive with the exact recorded wire commands, and
require it to reproduce the live attitude AND the correctly-SIGNED East velocity.
Shown side by side with the pre-audit config (RAW attitude, rate_sign=[-1,+1,-1])
which reproduces attitude/speed but mirrors East (the S18 smoking gun).
Windows: rollfix f1+f2 (gate-0 flare) and two pre-fix bridge banked segments.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from racer.rl_plant import (PlantParams, PlantState, step as plant_step,
                            SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
                            QUAD_DRAG_C2_MEASURED, COLL_MAP_THR_MEASURED,
                            COLL_MAP_ACCEL_MEASURED, LAPSE_SPEED_MEASURED,
                            LAPSE_FACTOR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED)

DATA = ROOT / "handoff/shadowpc-refit-dataset-2026-06-12/extracted"
QCONJ = np.array([1.0, -1.0, 1.0, -1.0])    # reported -> true (R_y(pi) conjugation)
DT = 1.0 / 30.0


def load(run):
    rows = [json.loads(l) for l in open(DATA / run / "debug_obs.jsonl", encoding="utf-8")]
    return [r for r in rows if r.get("type") != "header"]


def eul(q):
    q = np.asarray(q, float)
    return np.degrees(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("ZYX"))[::-1]


def params(rate_sign):
    return PlantParams(
        transport_delay_steps=0, rate_sign=np.array(rate_sign, float),
        super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
        coll_map_thr=COLL_MAP_THR_MEASURED.copy(), coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
        mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
        lapse_speed=LAPSE_SPEED_MEASURED.copy(), lapse_factor=LAPSE_FACTOR_MEASURED.copy())


def replay(rows, k0, nt, true_frame):
    r0 = rows[k0]
    if true_frame:
        q0 = np.array(r0["q_raw_wxyz"]) * QCONJ
        w0 = -np.array(r0["w_raw"])
        P = params([1, 1, 1])
    else:
        q0 = np.array(r0["q_raw_wxyz"])
        w0 = np.array(r0["w_raw"]) * np.array([1.0, -1.0, 1.0])   # bcc93f9 reading
        P = params([-1, 1, -1])
    st = PlantState(pos=np.array(r0["pos_ned"]), vel=np.array(r0["vel_ned"]),
                    quat=q0, omega=w0, thrust=np.asarray(r0["collective"]))
    cp = None; clp = 0.0; tr = []
    for k in range(k0, k0 + nt):
        if cp is not None:
            st = plant_step(st, np.concatenate([cp, [clp]]), DT, P)
        cp = np.array(rows[k]["rate_frd"]); clp = rows[k]["collective"]
        tr.append(st)
    return tr


def show(run, k0, nt):
    rows = load(run)
    nt = min(nt, len(rows) - k0)
    tr_true = replay(rows, k0, nt, True)
    tr_raw = replay(rows, k0, nt, False)
    print(f"\n=== {run.split('_', 2)[2]}  k0={k0} ({nt} ticks) ===")
    print(f"{'k':>3} | {'LIVE(true) r':>12} {'p':>5} {'y':>5} {'vE':>6} {'vN':>6} | "
          f"{'FIXED r':>8} {'p':>5} {'y':>5} {'vE':>6} {'vN':>6} | {'OLD vE':>7}")
    errs_t, errs_r = [], []
    for i, k in enumerate(range(k0, k0 + nt)):
        Lq = np.array(rows[k]["q_raw_wxyz"]) * QCONJ
        L = eul(Lq); lv = rows[k]["vel_ned"]
        T = eul(tr_true[i].quat); tv = tr_true[i].vel
        Ov = tr_raw[i].vel
        errs_t.append(tv[1] - lv[1]); errs_r.append(Ov[1] - lv[1])
        if k % 3 == 0:
            print(f"{k:>3} | {L[0]:>12.0f} {L[1]:>5.0f} {L[2]:>5.0f} {lv[1]:>6.1f} {lv[0]:>6.1f} | "
                  f"{T[0]:>8.0f} {T[1]:>5.0f} {T[2]:>5.0f} {tv[1]:>6.1f} {tv[0]:>6.1f} | {Ov[1]:>7.1f}")
    print(f"  vE err @end: FIXED {errs_t[-1]:+.2f} m/s   OLD(raw/mirror) {errs_r[-1]:+.2f} m/s")


show("20260612_044219_inc6_rollfix_f1", 36, 27)
show("20260612_044438_inc6_rollfix_f2", 36, 27)
# pre-fix bridge runs: pick a banked window (gate-1 orbit phase)
for run in ["20260612_040440_inc6_bridge_f1", "20260612_040748_inc6_bridge_f2"]:
    rows = load(run)
    R = Rotation.from_quat(np.array([r["q_raw_wxyz"] for r in rows])[:, [1, 2, 3, 0]]).as_matrix()
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    spd = np.linalg.norm(np.array([r["vel_ned"] for r in rows]), axis=1)
    ks = [k for k in range(10, len(rows) - 30) if tilt[k] > 40 and spd[k] > 8]
    if ks:
        show(run, ks[0], 27)
