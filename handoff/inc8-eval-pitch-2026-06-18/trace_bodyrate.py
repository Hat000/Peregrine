"""Matched-state trace: does the eval's look-at injection reproduce TRAINING's realized
body rate, axis by axis? Isolates the (injection-map x plant rate_sign) difference.

Three realized-rate paths on ONE matched pre-step state + ONE policy FLU action + one dlook:
  (a) TRAINING : add dlook to the FLU action, FLU->FRD via _FLIP=[1,-1,-1] (diffaero adapter),
                 plant rate_sign=[1,1,-1] (trained-world; peregrine_racing PlantParams default).
  (b) EVAL-CUR : policy FRD = policy_flu*_ACT_FLU_TO_FRD ([1,-1,1], policy_step wire map);
                 reconstruct FLU via *_FLIP, add dlook, back via *_FLIP; plant rate_sign=[1,1,1].
  (c) EVAL-FIX : same as (b) but reconstruct/re-map via *_ACT_FLU_TO_FRD (the wire map policy_step
                 actually used), plant rate_sign=[1,1,1].

rl_plant.step is the bit-parity reference for diffaero (tests/test_rl_plant_parity), so running it
with the training rate_sign IS the training realized rate. We use the SAME mixer plant for all three
and vary ONLY rate_sign + the injection map -- isolating exactly the quantity under investigation.

Also confirms t_cam (numpy fix_surrogate.geometry vs torch batched_geometry) is identical, i.e.
the 'leading hypothesis' (t_cam Y divergence) is false.
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import torch

from racer.rl_plant import PlantParams, PlantState, step as plant_step
from racer.rl_plant import (SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
                            QUAD_DRAG_C2_MEASURED, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED)
from fly_rl import _FLIP, _ACT_FLU_TO_FRD, _HOVER_THRUST, _GATE_POS_ZUP, _TRAIN_DT
import inc8_reward as R8
import inc8_estimator_emul as IE
import fix_surrogate as FS
from estimator_emul import make_ned_gate

np.set_printoptions(precision=6, suppress=True)

# ---- baked maps ----
print("=" * 78)
print("CONSTANTS")
print("=" * 78)
print(f"  _FLIP (training FLU<->FRD adapter)   = {_FLIP}")
print(f"  _ACT_FLU_TO_FRD (policy_step wire)   = {_ACT_FLU_TO_FRD}")
RATE_SIGN_TRAIN = np.array([1.0, 1.0, -1.0])   # PlantParams default (trained-world)
RATE_SIGN_EVAL = np.array([1.0, 1.0, 1.0])     # _RATE_SIGN_LIVE
print(f"  rate_sign TRAIN (trained-world)      = {RATE_SIGN_TRAIN}")
print(f"  rate_sign EVAL  (_RATE_SIGN_LIVE)    = {RATE_SIGN_EVAL}")

# ---- matched pre-step state: an approach to gate-0, gate in front, range in band ----
# Place the drone ~18 m before gate-0 along -x_ned (the course runs toward -x), gate ahead.
gate0_zup = _GATE_POS_ZUP[0]
gate0_ned = gate0_zup * _FLIP
# drone 18 m back along x (NED), slight lateral/vertical offset so uy/ux are non-trivial
drone_ned = gate0_ned + np.array([18.0, 0.6, -0.4])
vel_ned = np.array([-12.0, 0.2, -0.1])
# attitude: roughly nose toward -x (tail-first training-native frame: identity quat in z-up == ?)
# Use a mild pitch/yaw so t_cam has both ux and uy components.
from scipy.spatial.transform import Rotation as Rot
# body->world NED: yaw ~180 (heading -x), small pitch down
R_wb = Rot.from_euler("ZYX", [np.pi, np.radians(-8.0), np.radians(3.0)]).as_matrix()
q_wxyz = Rot.from_matrix(R_wb).as_quat()[[3, 0, 1, 2]]

# ---- t_cam: numpy fix_surrogate.geometry vs torch batched_geometry ----
gate_obj = make_ned_gate(0, gate0_zup, np.pi)
geom_np = FS.geometry(drone_ned, R_wb, gate_obj)
tcam_np = geom_np.t_cam

R_cb, K, flip_t, g_t = IE._const(torch.device("cpu"), torch.float64)
tcam_torch = IE.batched_geometry(
    torch.tensor(drone_ned)[None], torch.tensor(R_wb)[None],
    torch.tensor(gate0_ned)[None], torch.tensor(gate_obj.R_world_gate)[None],
    R_cb, K)["t_cam"][0].numpy()

print("\n" + "=" * 78)
print("t_cam COMPARISON (leading hypothesis test)")
print("=" * 78)
print(f"  numpy fix_surrogate.geometry t_cam = {tcam_np}")
print(f"  torch batched_geometry      t_cam = {tcam_torch}")
print(f"  |delta| per component             = {np.abs(tcam_np - tcam_torch)}")
print(f"  range_m = {geom_np.range_m:.3f}  in_image={geom_np.in_image}")

# ---- dlook (FLU action conv) for yaw-only and 2-axis, via the pinned primitive ----
R_BC = R8.r_body_from_camera(None, torch.float64)
FLIP_RATE = torch.tensor(R8._FLIP_FRD_FLU, dtype=torch.float64)
G_YAW, G_PITCH = -3.0, 3.0


def dlook_flu(g_yaw, g_pitch):
    d = R8.lookat_correction(torch.tensor(tcam_np), float(g_yaw), float(g_pitch), R_BC, FLIP_RATE)
    return d.numpy()


dlook_yaw_only = dlook_flu(G_YAW, 0.0)
dlook_2axis = dlook_flu(G_YAW, G_PITCH)
print("\n" + "=" * 78)
print("dlook (FLU action [roll,pitch,yaw]) from the pinned lookat_correction")
print("=" * 78)
print(f"  yaw-only (g_yaw={G_YAW}, g_pitch=0)      dlook = {dlook_yaw_only}")
print(f"  2-axis   (g_yaw={G_YAW}, g_pitch={G_PITCH})    dlook = {dlook_2axis}")

# ---- one plant step, three ways; report realized omega ----
ACT_LO, ACT_HI = -3.14, 3.14


def mixer_params(rate_sign):
    return PlantParams(rate_sign=rate_sign.copy(),
                       super_rate_s=SUPER_RATE_S_MEASURED,
                       alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                       linear_drag=0.0,
                       quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                       coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                       coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
                       mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                       mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)


P_TRAIN = mixer_params(RATE_SIGN_TRAIN)
P_EVAL = mixer_params(RATE_SIGN_EVAL)


def make_state():
    return PlantState(pos=drone_ned.copy(), vel=vel_ned.copy(), quat=q_wxyz.copy(),
                      omega=np.array([0.1, -0.2, 0.3]), thrust=np.float64(_HOVER_THRUST))


def realized_training(policy_flu, dlook):
    """Add dlook in FLU, clip, FLU->FRD via _FLIP, plant rate_sign [1,1,-1]."""
    flu = np.clip(policy_flu + dlook, ACT_LO, ACT_HI)
    rate_frd = flu * _FLIP
    action = np.concatenate([rate_frd, [_HOVER_THRUST]])
    return plant_step(make_state(), action, _TRAIN_DT, P_TRAIN).omega


def realized_eval(policy_flu, dlook, recon_map):
    """policy_step FRD = policy_flu*_ACT_FLU_TO_FRD; reconstruct FLU via recon_map, add dlook, clip,
    re-map via recon_map; plant rate_sign [1,1,1]."""
    rate_frd_policy = policy_flu * _ACT_FLU_TO_FRD
    flu = rate_frd_policy * recon_map
    flu = np.clip(flu + dlook, ACT_LO, ACT_HI)
    rate_frd = flu * recon_map
    action = np.concatenate([rate_frd, [_HOVER_THRUST]])
    return plant_step(make_state(), action, _TRAIN_DT, P_EVAL).omega


# a representative policy FLU action: roll/pitch modest, yaw near the rail (the policy dithers yaw)
policy_flu = np.array([0.4, -0.5, 2.6])

for name, dlook in [("YAW-ONLY", dlook_yaw_only), ("2-AXIS", dlook_2axis)]:
    w_train = realized_training(policy_flu, dlook)
    w_cur = realized_eval(policy_flu, dlook, _FLIP)
    w_fix = realized_eval(policy_flu, dlook, _ACT_FLU_TO_FRD)
    print("\n" + "=" * 78)
    print(f"REALIZED omega (rad/s, FRD)  [{name}]  policy_flu={policy_flu}")
    print("=" * 78)
    print(f"  (a) TRAINING            omega = {w_train}")
    print(f"  (b) EVAL-CURRENT(_FLIP) omega = {w_cur}   d(b-a) = {w_cur - w_train}")
    print(f"  (c) EVAL-FIX(_ACT_FLU)  omega = {w_fix}   d(c-a) = {w_fix - w_train}")
    print(f"  >> current divergence |b-a| per axis = {np.abs(w_cur - w_train)}")
    print(f"  >> fixed   divergence |c-a| per axis = {np.abs(w_fix - w_train)}")
