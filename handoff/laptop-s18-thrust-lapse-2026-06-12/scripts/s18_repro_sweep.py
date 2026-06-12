"""S18 verdict probe -- does ANY plausible thrust-lapse curve reproduce inc6's live
standing-start failure (gate-0 plane crossed at ~+5 m East, then wander)? Sweeps lapse depth
and high-speed persistence; reports outcome + the East offset at the gate-0 plane.

The as-fit lapse recovers to 1.0 by 15 m/s, but the live East drift develops at 16-18 m/s --
so this tests the identifiability alternative (the high-speed residual is real thrust loss, not
drag recovery) and a depth sweep, to see whether the thrust channel can explain the lateral miss
at all."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "rl"))

from racer.rl_plant import (PlantParams, PlantState, step as plant_step,
                            SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
                            QUAD_DRAG_C2_MEASURED, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED)
from fly_rl import load_actor, policy_step, _HOVER_THRUST, _FLIP, _GATE_POS_ZUP, N_GATES, _R_W2G
import offline_rollout as OR
from types import SimpleNamespace

_RATE_SIGN_LIVE = OR._RATE_SIGN_LIVE
ACTOR = load_actor(str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth"))
DT = 1.0 / 30.0
_ARGS = SimpleNamespace(gate=0, thrust0=-1.0, handoff_dist=3.0, handoff_speed=5.1)
GATE0_N = float((_GATE_POS_ZUP[0] * _FLIP)[0])   # gate-0 North coordinate (NED)


def base_aero(latency=2, lapse=None):
    kw = dict(transport_delay_steps=latency, rate_sign=_RATE_SIGN_LIVE.copy(),
              super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
              linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
              coll_map_thr=COLL_MAP_THR_MEASURED.copy(), coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
              mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
              mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)
    if lapse is not None:
        kw["lapse_speed"], kw["lapse_factor"] = lapse
    return PlantParams(**kw)


def run(params, n=360):
    st, gate = OR.make_start("simstart", _ARGS)
    last_normed = 1.0
    best_dN, e_at_plane, max_e, outcome = 1e9, None, 0.0, "TIMEOUT"
    for k in range(n):
        obs = OR.obs_from_truth(st, gate, last_normed, True)
        rate_frd, collective, last_normed = policy_step(ACTOR, obs, 0.0, True, 0.0, yaw_scale=1.0)
        collective = last_normed * _HOVER_THRUST
        prev = st.pos.copy()
        st = plant_step(st, np.concatenate([rate_frd, [collective]]), DT, params)
        zup = st.pos * _FLIP
        # East (pos_y) at the tick of CLOSEST approach to the gate-0 plane (capture the lateral miss
        # whether the vehicle passes cleanly or wanders short of the plane)
        if gate == 0:
            dN = abs(st.pos[0] - GATE0_N)
            if dN < best_dN:
                best_dN, e_at_plane = dN, st.pos[1]
        max_e = max(max_e, abs(st.pos[1]))
        ev = OR.gate_event(prev, st.pos, gate)
        if OR.frame_strike_other_gates(prev, st.pos, gate) is not None:
            outcome = "COLLISION"; break
        if ev == "pass":
            if gate == N_GATES - 1:
                outcome = "FINISHED"; break
            gate += 1
        elif ev in ("collision", "miss"):
            outcome = ev.upper(); break
        pts = np.vstack([_GATE_POS_ZUP, [0.0, 0.0, -0.02]])
        lo = pts.min(0) - [15, 15, 12]; hi = pts.max(0) + [15, 15, 12]
        if np.any(zup < lo) or np.any(zup > hi):
            outcome = "OOB"; break
    return outcome, gate, e_at_plane, max_e


CURVES = {
    "as-fit (recover@15)":   ([0, 4, 8, 12, 15], [1.0, 0.78, 0.80, 0.92, 1.0]),
    "persist-0.72":          ([0, 4, 8, 30],     [1.0, 0.72, 0.72, 0.72]),
    "deep-early/recover@14":  ([0, 4, 8, 12, 14], [1.0, 0.55, 0.60, 0.85, 1.0]),
    "vdeep-early/recover@14": ([0, 4, 8, 12, 14], [1.0, 0.45, 0.50, 0.80, 1.0]),
    "deep-early/recover@18":  ([0, 4, 8, 14, 18], [1.0, 0.55, 0.58, 0.80, 1.0]),
}
print("LIVE target: gate-0 plane crossed at East ~ +5..+7 m, then wander (NOT finished)\n")
print(f"{'curve':>22} {'lat':>5} {'outcome':>9} {'gates':>6} {'E@plane':>8} {'maxE':>6}")
for lat in (0, 2, 3):
    print(f"  --- latency {lat} ---")
    # no-lapse baseline
    o, g, ep, me = run(base_aero(lat, None))
    print(f"{'NO LAPSE (mixer)':>22} {lat:>5} {o:>9} {g:>6} {str(round(ep,1) if ep else None):>8} {me:>6.1f}")
    for name, curve in CURVES.items():
        ls = (np.array(curve[0], float), np.array(curve[1], float))
        o, g, ep, me = run(base_aero(lat, ls))
        print(f"{name:>22} {lat:>5} {o:>9} {g:>6} {str(round(ep,1) if ep else None):>8} {me:>6.1f}")
