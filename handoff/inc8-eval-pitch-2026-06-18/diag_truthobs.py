"""Decisive test: apply the look-at (2-axis) but fly the policy on TRUTH obs (perfect pose, no
estimator noise). Isolates whether the 2-axis breakage is obs-fidelity (eval/train gap) or the
look-at pitch intrinsically destabilizing the flight.

Custom rollout mirroring contact_true_eval.run_episode's loop but:
 - obs = obs_from_truth (perfect pose) ALWAYS
 - look-at geom computed from TRUTH pose (fix_surrogate.geometry to the current target gate)
 - look-at injected via the FIXED convention (_ACT_FLU_TO_FRD), gated [8,30] & t_cam_z>0
 - plant = mixer with _RATE_SIGN_LIVE (the eval plant)
"""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import contact_true_eval as CTE
from contact_true_eval import (_build_start, _build_plant_params, _score_gate, _lookat_dlook_flu,
                               BODY_RADIUS_NOM, FRAME_DEPTH_NOM)
from fly_rl import (load_actor, policy_step, _GATE_POS_ZUP, _FLIP, _ACT_FLU_TO_FRD,
                    _ACT_MIN, _ACT_MAX, _HOVER_THRUST, _TRAIN_DT, N_GATES)
from offline_rollout import obs_from_truth
from racer.rl_plant import step as plant_step
import fix_surrogate as FS
from estimator_emul import make_ned_gate
from scipy.spatial.transform import Rotation as Rot

CKPT = str(_ROOT / "rl" / "checkpoints" / "inc8_rc1_seed0_actor.pth")
actor = load_actor(CKPT)
params = _build_plant_params("mixer")
GATES = [make_ned_gate(g, _GATE_POS_ZUP[g], np.pi) for g in range(N_GATES)]


def R_from_quat_wxyz(q):
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def run(g_yaw, g_pitch, max_time=40.0, verbose=False):
    st, gate, vflip = _build_start("trainreset", 0)
    dt = _TRAIN_DT
    n_steps = int(round(max_time / dt))
    last_normed = 0.0
    outcome, furthest = "TIMEOUT", -1
    for k in range(n_steps):
        obs = obs_from_truth(st, gate, last_normed, vflip)   # TRUTH obs (perfect pose)
        # pad to 20-dim with a neutral confidence triple [1,1,0] (high conf, fresh) -- the policy
        # consumes [17:20]; with truth obs there's no KF, so feed the "perfect" triple.
        obs20 = np.concatenate([obs, np.array([1.0, 1.0, 0.0], np.float32)]).astype(np.float32)
        rate_frd, _coll, last_normed = policy_step(actor, obs20, 0.0, vflip)
        # look-at from TRUTH geom to current target gate
        R_wb = R_from_quat_wxyz(st.quat)
        geom = FS.geometry(st.pos, R_wb, GATES[gate])
        if (8.0 <= geom.range_m <= 30.0) and (float(geom.t_cam[2]) > 0.0):
            dlook = _lookat_dlook_flu(geom.t_cam, g_yaw, g_pitch)
            flu = rate_frd * _ACT_FLU_TO_FRD
            flu = np.clip(flu + dlook, _ACT_MIN[1:4], _ACT_MAX[1:4])
            rate_frd = flu * _ACT_FLU_TO_FRD
        action = np.concatenate([rate_frd, [last_normed * _HOVER_THRUST]])
        prev = st.pos.copy()
        st = plant_step(st, action, dt, params)
        # collisions on non-target gates
        hit = False
        for g in range(N_GATES):
            if g == gate:
                continue
            v, _ = _score_gate(prev, st.pos, g, BODY_RADIUS_NOM, FRAME_DEPTH_NOM, _GATE_POS_ZUP)
            if v == "collision":
                outcome, hit = "COLLISION", True
                break
        if hit:
            break
        v, _ = _score_gate(prev, st.pos, gate, BODY_RADIUS_NOM, FRAME_DEPTH_NOM, _GATE_POS_ZUP)
        if v == "pass":
            furthest = gate
            if gate == N_GATES - 1:
                outcome = "FINISHED"; break
            gate += 1
        elif v == "collision":
            outcome = "COLLISION"; break
        elif v == "miss":
            outcome = "MISS"; break
    return outcome, furthest


print("TRUTH-OBS rollout (perfect pose), look-at applied via fixed convention:")
for gy, gp, name in [(-3.0, 0.0, "yaw-only"), (-3.0, 1.0, "2axis gp=1"),
                     (-3.0, 2.0, "2axis gp=2"), (-3.0, 3.0, "2axis gp=3"),
                     (0.0, 0.0, "lookat-zero")]:
    o, g = run(gy, gp)
    print(f"  {name:14s} (gy={gy} gp={gp}): outcome={o:9s} furthest_gate={g}")
