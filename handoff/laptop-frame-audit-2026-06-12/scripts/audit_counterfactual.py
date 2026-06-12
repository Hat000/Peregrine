"""Closed-loop counterfactual against the FRAME-AUDIT-emulated live sim
(TRUE physics rate_sign [+1,+1,+1]; telemetry emitted in the conjugated frame):

  FIXED  : current fly_rl constants (conjugation undo, wire [+1,-1,+1])
  BCC93F9: the 2026-06-12 diag constants (quat as-is, rates [+1,-1,+1],
           wire [-1,-1,-1]) -- expected to reproduce the live rollfix failure
           (+E displacement at the gate-0 plane, then OOD wander).

Run: .venv/Scripts/python.exe handoff/laptop-frame-audit-2026-06-12/scripts/audit_counterfactual.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import fly_rl
from fly_rl import build_obs, load_actor, policy_step, N_GATES
import offline_rollout as OR
from racer.rl_plant import (PlantParams, PlantState, step as plant_step,
                            SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
                            QUAD_DRAG_C2_MEASURED, COLL_MAP_THR_MEASURED,
                            COLL_MAP_ACCEL_MEASURED, LAPSE_SPEED_MEASURED,
                            LAPSE_FACTOR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED)

DT = 1.0 / 30.0
PARAMS = PlantParams(
    transport_delay_steps=2, rate_sign=np.array([1.0, 1.0, 1.0]),
    super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
    linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
    coll_map_thr=COLL_MAP_THR_MEASURED.copy(), coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
    mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
    mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
    lapse_speed=LAPSE_SPEED_MEASURED.copy(), lapse_factor=LAPSE_FACTOR_MEASURED.copy())


def run(label, conj, rate_sign_undo, act_map):
    saved = (fly_rl._ODO_QUAT_TRUE_CONJ.copy(), fly_rl._ODO_RATE_SIGN.copy(),
             fly_rl._ACT_FLU_TO_FRD.copy())
    fly_rl._ODO_QUAT_TRUE_CONJ[:] = conj
    fly_rl._ODO_RATE_SIGN[:] = rate_sign_undo
    fly_rl._ACT_FLU_TO_FRD[:] = act_map
    try:
        actor = load_actor(str(ROOT / "rl/checkpoints/stage1_inc6_actor.pth"))
        st, gate = OR.make_start("simstart", type("A", (), {
            "gate": 0, "handoff_dist": 3.0, "handoff_speed": 10.0, "thrust0": -1.0})())
        last = 0.0
        crossed_g0_at = None
        for k in range(int(25 / DT)):
            obs = build_obs(OR.telemetry_from_truth(st), gate, last, virtual_flip=True)
            rate_frd, coll, last = policy_step(actor, obs, 0.0, virtual_flip=True)
            prev = st.pos.copy()
            st = plant_step(st, np.concatenate([rate_frd, [coll]]), DT, PARAMS)
            if crossed_g0_at is None and prev[0] > -23.3 >= st.pos[0]:
                crossed_g0_at = (k * DT, st.pos.copy())
            ev = OR.gate_event(prev, st.pos, gate)
            if ev == "pass":
                if gate == N_GATES - 1:
                    print(f"  {label}: FINISHED 6/6 t={(k+1)*DT:.2f}s")
                    return
                gate += 1
            elif ev in ("collision", "miss"):
                print(f"  {label}: {ev.upper()} at gate {gate} t={(k+1)*DT:.2f}s "
                      f"pos={np.round(st.pos,1).tolist()}")
                return
            zup = st.pos * fly_rl._FLIP
            pts = np.vstack([fly_rl._GATE_POS_ZUP, [0.0, 0.0, -0.02]])
            if np.any(zup < pts.min(0) - [15, 15, 12]) or np.any(zup > pts.max(0) + [15, 15, 12]):
                g0 = (f"gate-0 plane crossed at E={crossed_g0_at[1][1]:+.1f} m t={crossed_g0_at[0]:.2f}s"
                      if crossed_g0_at else "never crossed gate-0 plane")
                print(f"  {label}: OOB t={(k+1)*DT:.2f}s gates={gate} "
                      f"pos={np.round(st.pos,1).tolist()}  [{g0}]")
                return
        g0 = (f"gate-0 plane crossed at E={crossed_g0_at[1][1]:+.1f} m"
              if crossed_g0_at else "never crossed gate-0 plane")
        print(f"  {label}: TIMEOUT gates={gate}  [{g0}]")
    finally:
        fly_rl._ODO_QUAT_TRUE_CONJ[:] = saved[0]
        fly_rl._ODO_RATE_SIGN[:] = saved[1]
        fly_rl._ACT_FLU_TO_FRD[:] = saved[2]


print("counterfactual vs FRAME-AUDIT-emulated live sim (true physics + conjugated telemetry):")
run("FIXED  (audit)  ", [1, -1, 1, -1], [-1, -1, -1], [1, -1, 1])
run("BCC93F9 (diag)  ", [1, 1, 1, 1],   [1, -1, 1],   [-1, -1, -1])
