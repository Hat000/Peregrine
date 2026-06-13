"""HYBRID tracking-realizability check (Phase B).

Forward-simulate the REAL measured-aero CtbrPlant (mixer) tracking the corrected-aero re-timed
reference with a differential-flatness feedforward + position/velocity PD. Measures the actual
realized lap time and gate-plane miss distances -> the REALIZABILITY TAX of the decomposed/hybrid
architecture (does a tracker hit the planner line, or does it need k>1 dilation like the existing
8.3 s geometric tracker?).

Tracker = the standard quadrotor geometric controller cast for CTBR (body-rate + collective):
  - desired accel a_des = a_ref_ff + Kp*(p_ref - p) + Kd*(v_ref - v) + g_vec
  - thrust direction b3_des = a_des/|a_des|; collective from the measured convex thrust map inverse
  - body-rate command from attitude error (proportional on the rotation vector b3 -> b3_des)
This is exactly the tracker class the hybrid would deploy (RL-tracker would do no better than a
well-tuned flatness tracker on a feasible line; if anything RL helps at the saturated edges).
"""
from __future__ import annotations
import json
import numpy as np
from scipy.interpolate import CubicSpline

from racer.rl_plant import (PlantParams, PlantState, step as plant_step,
    quat_rotate, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
import importlib.util, sys, os
_here = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location('hp', os.path.join(_here, 'hybrid_prototype.py'))
hp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hp)

G = 9.80665
HOVER = 0.2656
AMAX_UP = hp.AMAX_UP


def collective_from_a_up(a_up):
    """Invert the measured convex thrust map: body-up specific accel -> normalized collective."""
    return float(np.interp(a_up, COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED))


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)]])


def track(reference, params, dt=1/200., kp=8.0, kd=4.0, k_att=12.0, tilt_max_deg=75.0):
    """reference: dict with t, pos(N,3 NED), vel(N,3), acc(N,3). Returns realized log."""
    tref = reference['t']; pref = reference['pos']; vref = reference['vel']; aref = reference['acc']
    T = tref[-1]
    # start at rest at the line's start, level
    st = PlantState(pos=pref[0].copy(), vel=np.zeros(3),
                    quat=np.array([1., 0., 0., 0.]), omega=np.zeros(3),
                    thrust=np.float64(HOVER))
    n = int(np.ceil(T / dt)) + 1
    g_vec = np.array([0., 0., G])  # gravity in NED (+z down); to hover we need -g on body-up
    log_p = [st.pos.copy()]; log_t = [0.0]; log_v=[st.vel.copy()]
    tilt_max = np.radians(tilt_max_deg)
    for k in range(n):
        t = k * dt
        p_r = np.array([np.interp(t, tref, pref[:, i]) for i in range(3)])
        v_r = np.array([np.interp(t, tref, vref[:, i]) for i in range(3)])
        a_r = np.array([np.interp(t, tref, aref[:, i]) for i in range(3)])
        # desired specific force (thrust accel) = a_ref - g + PD ; in NED, body-up must cancel g.
        # accel = f_thrust + g_vec(down). To realize a_r: f_thrust = a_r - g_vec.
        a_des = a_r + kp * (p_r - st.pos) + kd * (v_r - st.vel)
        f_thrust = a_des - g_vec        # required specific force (up is -z)
        # thrust magnitude (specific accel) and direction
        a_mag = np.linalg.norm(f_thrust)
        if a_mag < 1e-6:
            b3_des = np.array([0., 0., -1.])
            a_mag = G
        else:
            b3_des = f_thrust / a_mag   # this is the body -z (up) axis in world
        # tilt-cap: limit b3_des tilt from vertical-up = [0,0,-1]
        up = np.array([0., 0., -1.])
        cos_t = np.clip(np.dot(b3_des, up), -1, 1)
        tilt = np.arccos(cos_t)
        if tilt > tilt_max:
            # rotate b3_des back toward up to the cap
            axis = np.cross(up, b3_des)
            if np.linalg.norm(axis) > 1e-6:
                axis /= np.linalg.norm(axis)
                from scipy.spatial.transform import Rotation as Rot
                b3_des = Rot.from_rotvec(axis * tilt_max).apply(up)
        # collective from required up-accel (projection onto current b3, but use a_mag along b3_des)
        a_up_cmd = a_mag  # specific accel along thrust axis
        coll = np.clip(collective_from_a_up(a_up_cmd), 0.0, 1.0)
        # current body-up axis
        R = quat_to_R(st.quat)
        b3 = R @ np.array([0., 0., -1.])
        # attitude error: rotation taking b3 -> b3_des (reduced attitude, ignore yaw)
        err = np.cross(b3, b3_des)
        s = np.linalg.norm(err)
        c = np.clip(np.dot(b3, b3_des), -1, 1)
        ang = np.arctan2(s, c)
        if s > 1e-9:
            err_axis_world = err / s
        else:
            err_axis_world = np.zeros(3)
        # body-rate command in WORLD -> body
        omega_world = k_att * ang * err_axis_world
        omega_body = R.T @ omega_world
        rate = np.clip(omega_body, -11.0, 11.0)
        action = np.concatenate([rate, [coll]])
        st = plant_step(st, action, dt, params)
        log_p.append(st.pos.copy()); log_t.append(t + dt); log_v.append(st.vel.copy())
        if np.linalg.norm(st.pos) > 1e4 or not np.all(np.isfinite(st.pos)):
            break
    return np.array(log_t), np.array(log_p), np.array(log_v)


def gate_miss(traj_p, gates=hp.GATES):
    """For each gate, find nearest approach and in-plane L-inf miss (gate yaw ~pi => plane normal
    along world X). Returns list of (gate, time_idx, euclid_miss, inplane_linf)."""
    out = []
    for gi, gc in enumerate(gates):
        d = np.linalg.norm(traj_p - gc, axis=1)
        i = int(np.argmin(d))
        rel = traj_p[i] - gc
        # gate plane normal ~ world X (along-course); in-plane = Y,Z
        inplane = max(abs(rel[1]), abs(rel[2]))
        out.append((gi, i, float(d[i]), float(inplane)))
    return out


def main():
    pos, d = hp.load_geometry()
    r, tang, kappa, s = hp.resample_path(pos, n=900)
    params = hp.measured_plant(mixer=True)

    for tilt in [65, 75]:
        v, t, info = hp.corrected_aero_topp(r, tang, kappa, s, tilt)
        # build reference dict (vel along tangent, accel finite-diff)
        vel = v[:, None] * tang
        acc = np.zeros_like(vel)
        acc[1:-1] = (vel[2:] - vel[:-2]) / np.maximum((t[2:] - t[:-2])[:, None], 1e-9)
        ref = dict(t=t, pos=r, vel=vel, acc=acc)
        planner_lap = hp.gate_cross_times(r, t, s)[-1][1]
        print(f"\n===== TILT CAP {tilt} deg | planner TOPP lap = {planner_lap:.3f} s =====")
        for (kp, kd, katt) in [(10, 5, 15)]:
            lt, lp, lv = track(ref, params, dt=1/200., kp=kp, kd=kd, k_att=katt, tilt_max_deg=tilt+5)
            gm = gate_miss(lp)
            # realized lap = time at nearest approach to last gate
            lap = lt[gm[-1][1]]
            all_pass = all(m[3] < 0.75 - 0.38 for m in gm)
            print(f"  tracker kp={kp} kd={kd} katt={katt}: realized last-gate t={lap:.3f}s  "
                  f"(planner {planner_lap:.3f}s, tax {lap-planner_lap:+.3f}s)")
            for gi, i, dd, ip in gm:
                flag = 'OK' if ip < 0.75-0.38 else ('miss>0.37' if ip<0.75 else 'OUTSIDE')
                print(f"     gate{gi}: euclid={dd:.2f}m  inplane_Linf={ip:.3f}m  [{flag} vs 0.37 margin@r0.38]")
    return


if __name__ == '__main__':
    main()
