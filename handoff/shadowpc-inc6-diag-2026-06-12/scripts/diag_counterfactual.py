"""Closed-loop counterfactual: emulate the REAL live sim and fly it with the buggy
vs the corrected deployment mapping.

REAL live sim model (measured this session):
  - plant: rl_plant mixer config but rate_sign = [-1, +1, -1]  (roll AND yaw cmd inverted)
  - telemetry: reported quat = TRUE quat (no roll undo!); reported rates =
    true_omega * [ +1, -1, +1 ] (pitch-rate reporting inverted; involutory)

Deployment mappings:
  BUGGY (shipped fly_rl): build_obs (undoes a roll inversion that does not exist;
    rates * [-1,-1,+1]), wire = rate_flu * [1,-1,-1]
  FIXED: attitude = reported quat as-is; rates = reported * [+1,-1,+1];
    wire = rate_flu * [-1,-1,-1]  (negate roll wire too)

Success criterion: BUGGY reproduces the live signature (no gate 0, |w| blowup /
lateral sweep); FIXED finishes the course.
"""
from __future__ import annotations

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
                            quat_rotate, step as plant_step)
import fly_rl
from fly_rl import (_FLIP, _GATE_POS_ZUP, _HOVER_THRUST, _RZ_PI_BODY, N_GATES,
                    build_obs, load_actor, obs_from_zup)
from offline_rollout import gate_event, frame_strike_other_gates, make_start

CKPT = str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth")
DT = 1.0 / 30.0
S_LIVE = np.array([-1.0, 1.0, -1.0])          # measured live command->rate signs
REP_RATE_SIGN = np.array([1.0, -1.0, 1.0])    # true->reported (involutory)


def live_params(lat):
    return PlantParams(transport_delay_steps=lat,
                       rate_sign=S_LIVE.copy(),
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


def reported_telemetry(st):
    """What the REAL sim reports for true state st."""
    return SimpleNamespace(
        position_ned=st.pos.copy(),
        velocity_ned=st.vel.copy(),
        orientation_ned_wxyz=st.quat.copy(),          # reported AS-IS (measured)
        angular_rate_body=st.omega * REP_RATE_SIGN,   # pitch-rate reporting inverted
    )


def obs_fixed(tel, gate, last_normed):
    """Corrected deployment obs: quat as-is, rates reported*[1,-1,1]."""
    q = np.asarray(tel.orientation_ned_wxyz, dtype=np.float64)
    R_frd2ned = np.stack([quat_rotate(q, e) for e in np.eye(3)], axis=-1)
    R_b2w_zup = (_FLIP[:, None] * R_frd2ned) * _FLIP[None, :]
    w_frd = np.asarray(tel.angular_rate_body) * REP_RATE_SIGN
    return obs_from_zup(np.asarray(tel.position_ned) * _FLIP,
                        np.asarray(tel.velocity_ned) * _FLIP,
                        R_b2w_zup, w_frd * _FLIP, gate, last_normed,
                        virtual_flip=True)


@torch.no_grad()
def policy_u(actor, obs):
    mean = actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy().astype(np.float64)
    a = np.tanh(mean)
    act = fly_rl._ACT_MIN + (fly_rl._ACT_MAX - fly_rl._ACT_MIN) * (a + 1.0) / 2.0
    normed = float(act[0])
    rate_flu = act[1:4]
    rate_flu = _RZ_PI_BODY @ rate_flu          # virtual flip (both mappings use it)
    coll = float(np.clip(normed * _HOVER_THRUST, 0.0, 1.0))
    return rate_flu, coll, normed


def run(mapping: str, lat=2, max_s=30.0, label=""):
    actor = load_actor(CKPT)
    st, gate = make_start("simstart", type("A", (), {
        "gate": 0, "handoff_dist": 3.0, "handoff_speed": 10.0, "thrust0": -1.0}))
    p = live_params(lat)
    last = 0.0
    n = int(max_s / DT)
    outcome = "TIMEOUT"
    w_max = 0.0
    spin_t = 0.0
    for k in range(n):
        tel = reported_telemetry(st)
        if mapping == "buggy":
            obs = build_obs(tel, gate, last, virtual_flip=True)
            rate_flu, coll, last = policy_u(actor, obs)
            wire = rate_flu * np.array([1.0, -1.0, -1.0])
        else:
            obs = obs_fixed(tel, gate, last)
            rate_flu, coll, last = policy_u(actor, obs)
            wire = rate_flu * np.array([-1.0, -1.0, -1.0])
        prev = st.pos.copy()
        st = plant_step(st, np.concatenate([wire, [last * _HOVER_THRUST]]), DT, p)
        wm = float(np.linalg.norm(st.omega))
        w_max = max(w_max, wm)
        spin_t = spin_t + DT if wm > 6.0 else 0.0
        ev = gate_event(prev, st.pos, gate)
        struck = frame_strike_other_gates(prev, st.pos, gate)
        t = (k + 1) * DT
        if struck is not None:
            outcome = f"COLLISION(g{struck} frame)"
            break
        if ev == "pass":
            print(f"    t={t:6.2f}s gate {gate} PASS  pos={np.round(st.pos,1).tolist()}")
            if gate == N_GATES - 1:
                outcome = "FINISHED"
                break
            gate += 1
        elif ev in ("collision", "miss"):
            outcome = f"{ev.upper()} at gate {gate}"
            break
        if spin_t > 2.0:
            outcome = f"SPIN_ABORT (|w|>{6.0} sustained)"
            break
        zup = st.pos * _FLIP
        pts = np.vstack([_GATE_POS_ZUP, [0.0, 0.0, -0.02]])
        if np.any(zup < pts.min(0) - [15, 15, 12]) or np.any(zup > pts.max(0) + [15, 15, 12]):
            outcome = "OOB"
            break
    print(f"  [{label or mapping}] outcome={outcome}  gates={gate if outcome!='FINISHED' else 6}"
          f"  t={t:.2f}s  w_max={w_max:.1f} rad/s  final pos={np.round(st.pos,1).tolist()}")


print("REAL-SIM EMULATION (rate_sign [-1,+1,-1], true reporting artifacts), simstart:")
for lat in (0, 2):
    print(f"-- latency {lat}:")
    run("buggy", lat=lat, label=f"BUGGY  lat{lat}")
    run("fixed", lat=lat, label=f"FIXED  lat{lat}")
