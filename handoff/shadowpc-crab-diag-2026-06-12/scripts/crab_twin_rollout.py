"""crab_twin_rollout.py -- roll inc6 in the offline twin (rl_plant mixer) and log
TRUE physical attitude + course + obs per tick, with the SAME columns as
crab_live_extract.py, so twin and live are directly comparable.

Reuses offline_rollout's exact obs/action/plant machinery (the parity-tested twin).
Logs the TRUE plant attitude (euler of st.quat -- the plant's quat IS true physical
FRD->NED, no telemetry conjugation in the twin) and the policy obs.

Usage:
  python crab_twin_rollout.py --start racestart
  python crab_twin_rollout.py --start handoff --handoff-speed 5 --handoff-dist 18
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import offline_rollout as oro  # noqa: E402
from fly_rl import _HOVER_THRUST, _GATE_POS_ZUP, _FLIP  # noqa: E402
from racer.rl_plant import (PlantParams, step as plant_step,  # noqa: E402
                            ALPHA_MAX_RPS2_MEASURED, SUPER_RATE_S_MEASURED,
                            QUAD_DRAG_C2_MEASURED, COLL_MAP_THR_MEASURED,
                            COLL_MAP_ACCEL_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED)

GATE_NED = _GATE_POS_ZUP * _FLIP


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def quat_rpy_deg(q_wxyz):
    yaw, pitch, roll = Rotation.from_quat(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_euler("ZYX")
    return np.degrees([roll, pitch, yaw])


def tilt_deg(q_wxyz):
    """Angle between body-up (-z FRD = world thrust axis) and world-up. Yaw-invariant;
    this is the quantity rw_tilt penalizes. tilt = acos(R[2,2]) where R[2,2]=body-z . world-z;
    body-up is -bodyz so tilt-from-vertical = acos(-R[2,2])? We want angle of thrust axis from
    up: thrust axis = body -z; world up = -world z (NED). cos = (R@[-0,-0,-1]).( -e_z) = R[2,2]."""
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def mixer_params():
    return PlantParams(
        transport_delay_steps=0, rate_sign=np.array([1.0, 1.0, 1.0]),
        super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
        coll_map_thr=COLL_MAP_THR_MEASURED.copy(), coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
        mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)


def run(args):
    actor = oro.load_actor(args.checkpoint)
    params = mixer_params()
    st, gate = oro.make_start(args.start, args)
    dt = oro._TRAIN_DT
    last_normed = 0.0
    rows = []
    outcome = "TIMEOUT"
    n_steps = int(round(args.max_time / dt))
    for k in range(n_steps):
        obs = oro.obs_from_truth(st, gate, last_normed, args.virtual_flip)
        rate_frd, collective, last_normed = oro.policy_step(
            actor, obs, 0.0, args.virtual_flip, 0.0, yaw_scale=args.yaw_scale)
        collective = last_normed * _HOVER_THRUST
        action = np.concatenate([rate_frd, [collective]])
        prev = st.pos.copy()
        st = plant_step(st, action, dt, params)
        roll, pitch, yaw = quat_rpy_deg(st.quat)
        tilt = tilt_deg(st.quat)
        vel = st.vel
        spd_h = float(np.hypot(vel[0], vel[1]))
        course_vel = np.degrees(np.arctan2(vel[1], vel[0])) if spd_h > 0.2 else np.nan
        gvec = GATE_NED[min(gate, len(GATE_NED) - 1)] - st.pos
        gate_course = np.degrees(np.arctan2(gvec[1], gvec[0]))
        rows.append(dict(
            k=k, t=(k + 1) * dt, gi=gate, spd=float(np.linalg.norm(vel)), spd_h=spd_h,
            roll=roll, pitch=pitch, yaw=yaw, tilt=tilt, course_vel=course_vel, gate_course=gate_course,
            crab_vs_vel=wrap_deg(course_vel - yaw) if not np.isnan(course_vel) else np.nan,
            crab_vs_gate=wrap_deg(gate_course - yaw),
            obs_rpy_r=np.degrees(obs[6]), obs_rpy_p=np.degrees(obs[7]), obs_rpy_y=np.degrees(obs[8]),
            obs_posg_y=obs[1], obs_velg_y=obs[4]))
        ev = oro.gate_event(prev, st.pos, gate)
        if ev == "pass":
            if gate == 5:
                outcome = f"FINISHED@{(k+1)*dt:.2f}s"
                break
            gate += 1
        elif ev in ("collision", "miss"):
            outcome = f"{ev.upper()}@gate{gate}"
            break
        zup = st.pos * _FLIP
        pts = np.vstack([_GATE_POS_ZUP, [0.0, 0.0, -0.02]])
        if np.any(zup < pts.min(0) - [15, 15, 12]) or np.any(zup > pts.max(0) + [15, 15, 12]):
            outcome = f"OOB@gate{gate}"
            break
    return outcome, rows


def summarize(name, outcome, rows, speed_min=3.0):
    cruise = [r for r in rows if r["spd"] >= speed_min and not np.isnan(r["crab_vs_vel"])]
    print(f"\n==== TWIN {name}  outcome={outcome}  ({len(rows)} ticks, {len(cruise)} cruise) ====")
    if not cruise:
        print("  (no cruise ticks)")
        return
    for key in ["roll", "pitch", "yaw", "tilt", "course_vel", "gate_course",
                "crab_vs_vel", "crab_vs_gate", "obs_posg_y", "obs_velg_y",
                "obs_rpy_r", "obs_rpy_p", "obs_rpy_y"]:
        v = np.array([r[key] for r in cruise], float)
        v = v[~np.isnan(v)]
        print(f"  {key:14s} mean={np.mean(v):+8.2f}  med={np.median(v):+8.2f}  "
              f"[{np.min(v):+8.2f}, {np.max(v):+8.2f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth"))
    ap.add_argument("--start", default="racestart")
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--handoff-dist", type=float, default=20.0)
    ap.add_argument("--handoff-speed", type=float, default=5.0)
    ap.add_argument("--yaw-scale", type=float, default=1.0)
    ap.add_argument("--thrust0", type=float, default=-1.0)
    ap.add_argument("--virtual-flip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-time", type=float, default=40.0)
    args = ap.parse_args()
    outcome, rows = run(args)
    summarize(f"{args.start} vflip={args.virtual_flip}", outcome, rows)
