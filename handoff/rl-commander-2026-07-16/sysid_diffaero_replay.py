"""Replay ``sysid_program.csv`` through DiffAero's REAL action->plant pipeline (DR OFF, nominal
plant, 1 env, CPU) and log ground-truth state to ``sysid_diffaero_log.csv``; then print a per-axis
response SUMMARY.

PIPELINE (byte-identical to PeregrinePlantDynamics._step_numpy with DR off, minus the Z-up<->NED
frame wrapper -- we run the plant in its NATIVE NED/FRD frame and log there):
    a = [a_thrust, a_roll, a_pitch, a_yaw]        raw policy action in [-1, 1]  (CSV row)
    act = ACT_MIN + (ACT_MAX-ACT_MIN)*(a+1)/2     rescale_action ->
          normed_thrust in [0,5], rate_flu in [-3.14,3.14] rad/s (FLU body)
    rate_frd  = rate_flu * _FLIP                  _FLIP=[1,-1,-1]  (FLU->FRD)   [dda._action_...]
    collective = normed_thrust * hover_thrust     hover_thrust=0.2656 -> [0,1] wire collective
    ps = rl_step(ps, [rate_frd, collective], dt, PlantParams())   racer.rl_plant (DR-off default)
    accel_ned = (vel_new - vel_old) / dt          finite-diff (rl_plant emits no accel)

The plant integrates in WORLD NED (X north, Y east, Z DOWN) / BODY FRD (X fwd, Y right, Z down),
quat wxyz = R_world_body. The DiffAero training ENV wraps this in a Z-up world + FLU body frame via
the involutory flip [1,-1,-1] (world) / [1,-1,-1] (body): Zup = NED*[1,-1,-1]. We log NED/FRD
(closest to the VQ2/MAVLink frame) AND the FLU body rate the env actually observes.

LOG COLUMNS (sysid_diffaero_log.csv), all GROUND TRUTH:
    t
    a_thrust,a_roll,a_pitch,a_yaw     raw policy action [-1,1]                    (echoed from program)
    normed_thrust                     rescaled act[0], g-units [0,5]              (obs channel feed)
    cmd_roll_flu,cmd_pitch_flu,cmd_yaw_flu   rescaled act[1:4], FLU rad/s (policy-issued rate cmd)
    cmd_thrust                        collective = normed_thrust*hover, [0,1]     (plant thrust input)
    cmd_wx,cmd_wy,cmd_wz              FRD body-rate command fed to the plant (= cmd_flu*[1,-1,-1])
    wx,wy,wz                          achieved GT body rate, FRD (rad/s, plant native ps.omega)
    wx_flu,wy_flu,wz_flu             achieved GT body rate, FLU (= ps.omega*[1,-1,-1]; env obs frame)
    qw,qx,qy,qz                       attitude quat wxyz, R_world_body (NED)
    roll,pitch,yaw                    ZYX Euler of that quat (rad, NED FRD)
    vx,vy,vz                          GT world velocity, NED (m/s, Z DOWN)
    px,py,pz                          GT world position, NED (m, Z DOWN)
    ax,ay,az                          GT world linear accel, NED (m/s^2, finite-diff)
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = r"C:\Users\Fengy\Downloads\Projects\Anduril\.claude\worktrees\cool-heyrovsky-e08624"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "rl"))

from racer.rl_plant import PlantParams, PlantState, step as rl_step
import diffaero_dynamics as dda           # for the REAL _FLIP + action adapter

HERE = os.path.dirname(os.path.abspath(__file__))
PROG_CSV = os.path.join(HERE, "sysid_program.csv")
SEG_JSON = os.path.join(HERE, "sysid_program_segments.json")
LOG_CSV = os.path.join(HERE, "sysid_diffaero_log.csv")

# rescale_action bounds (cfg/dynamics/quad.yaml controller block; == fly_rl _ACT_MIN/_ACT_MAX)
ACT_MIN = np.array([0.0, -3.14, -3.14, -3.14])
ACT_MAX = np.array([5.0, 3.14, 3.14, 3.14])
FLIP = dda._FLIP                          # [1,-1,-1] FLU<->FRD body-axis flip (the REAL constant)


def rescale_action(a):
    """DiffAero BaseEnv.rescale_action: tanh action a in [-1,1] -> physical [normed_thrust, rate_flu]."""
    return ACT_MIN + (ACT_MAX - ACT_MIN) * (a + 1.0) / 2.0


def quat_to_euler_zyx(q):
    """wxyz R_world_body -> (roll, pitch, yaw) rad, ZYX intrinsic."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sp, -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def replay():
    prog = np.genfromtxt(PROG_CSV, delimiter=",", names=True)
    t_arr = np.atleast_1d(prog["t"])
    a = np.stack([prog["a_thrust"], prog["a_roll"], prog["a_pitch"], prog["a_yaw"]], axis=-1)
    a = np.atleast_2d(a)
    n = a.shape[0]

    params = PlantParams()                # DR OFF: legacy flat-gain nominal, no map/aero/mixer
    dt = float(t_arr[1] - t_arr[0]) if n > 1 else 0.025
    hover = params.hover_thrust

    ps = PlantState.hover(params=params)  # level identity attitude, zero vel/rate, thrust=hover
    v_prev = ps.vel.copy()

    header = ["t", "a_thrust", "a_roll", "a_pitch", "a_yaw", "normed_thrust",
              "cmd_roll_flu", "cmd_pitch_flu", "cmd_yaw_flu", "cmd_thrust",
              "cmd_wx", "cmd_wy", "cmd_wz", "wx", "wy", "wz",
              "wx_flu", "wy_flu", "wz_flu", "qw", "qx", "qy", "qz",
              "roll", "pitch", "yaw", "vx", "vy", "vz", "px", "py", "pz", "ax", "ay", "az"]
    out = []
    for k in range(n):
        act = rescale_action(a[k])                       # [normed_thrust, roll_flu, pitch_flu, yaw_flu]
        rate_frd, normed_thrust = dda._action_diffaero_to_ctbr_np(act)   # REAL adapter
        collective = normed_thrust * hover
        plant_action = np.concatenate([rate_frd, [collective]])
        ps = rl_step(ps, plant_action, dt, params)
        acc_ned = (ps.vel - v_prev) / dt
        v_prev = ps.vel.copy()

        w_frd = ps.omega
        w_flu = ps.omega * FLIP
        roll, pitch, yaw = quat_to_euler_zyx(ps.quat)
        out.append([
            t_arr[k], a[k, 0], a[k, 1], a[k, 2], a[k, 3], normed_thrust,
            act[1], act[2], act[3], collective,
            rate_frd[0], rate_frd[1], rate_frd[2],
            w_frd[0], w_frd[1], w_frd[2], w_flu[0], w_flu[1], w_flu[2],
            ps.quat[0], ps.quat[1], ps.quat[2], ps.quat[3],
            roll, pitch, yaw,
            ps.vel[0], ps.vel[1], ps.vel[2], ps.pos[0], ps.pos[1], ps.pos[2],
            acc_ned[0], acc_ned[1], acc_ned[2],
        ])

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in out:
            w.writerow(["%.6g" % v for v in r])
    print("wrote %s  (%d rows, dt=%.4f s)" % (LOG_CSV, len(out), dt))
    return header, np.array(out), dt


# --------------------------------------------------------------------------- response summary
def _col(header, data, name):
    return data[:, header.index(name)]


def _cross_time(tt, yy, thr):
    """First time |y| crosses thr, linearly interpolated between samples (sub-dt resolution)."""
    for i in range(1, len(yy)):
        if yy[i] >= thr:
            if yy[i] == yy[i - 1]:
                return tt[i]
            f = (thr - yy[i - 1]) / (yy[i] - yy[i - 1])
            return tt[i - 1] + f * (tt[i] - tt[i - 1])
    return float("nan")


def rise_time_10_90(t, y, y_ss, t0):
    """10->90% rise time of |y| toward steady y_ss from step onset t0 (interpolated crossings).
    Starts ONE sample before t0 (the pre-step baseline ~0) so both crossings are bracketed."""
    idx0 = int(np.argmax(t >= t0 - 1e-9))
    i0 = max(idx0 - 1, 0)
    tt, yy = t[i0:], np.abs(y[i0:])
    t_lo = _cross_time(tt, yy, 0.1 * abs(y_ss))
    t_hi = _cross_time(tt, yy, 0.9 * abs(y_ss))
    return t_hi - t_lo


def summary(header, data, dt):
    t = _col(header, data, "t")
    meta = json.load(open(SEG_JSON))
    segs = {s["name"]: s for s in meta["segments"]}

    print("\n" + "=" * 100)
    print("PER-AXIS RATE-STEP RESPONSE  (DiffAero nominal plant, DR OFF, flat-gain; dt=%.4f s)" % dt)
    print("signed gain = achieved_rate / commanded_rate (identical in FLU or FRD; = rate_gain*rate_sign)")
    print("=" * 100)
    print("%-8s %-6s | %9s %9s | %9s %10s | %8s | %s"
          % ("axis", "step", "cmd_flu", "ach_flu_ss", "gain", "|gain|", "rise1090", "sign"))
    axis_map = {"roll": ("cmd_roll_flu", "wx_flu"),
                "pitch": ("cmd_pitch_flu", "wy_flu"),
                "yaw": ("cmd_yaw_flu", "wz_flu")}
    gains = {}
    for axis, (cmd_name, ach_name) in axis_map.items():
        cmd = _col(header, data, cmd_name)
        ach = _col(header, data, ach_name)
        for tag in ("step_pos", "step_neg"):
            seg = segs[f"{axis}_{tag}"]
            t0, t1 = seg["t0"], seg["t1"]
            win = (t >= t0) & (t < t1)
            cmd_val = float(np.median(cmd[win]))
            # steady = mean over last 40% of the step window
            ss_win = (t >= t0 + 0.6 * (t1 - t0)) & (t < t1)
            ach_ss = float(np.mean(ach[ss_win]))
            g = ach_ss / cmd_val if cmd_val != 0 else float("nan")
            rt = rise_time_10_90(t, ach, ach_ss, t0)
            sign = "INVERTED" if g < 0 else "same"
            print("%-8s %-6s | %9.3f %9.3f | %9.3f %10.3f | %8.3f | %s"
                  % (axis, tag.split("_")[1], cmd_val, ach_ss, g, abs(g), rt, sign))
            gains.setdefault(axis, []).append(g)

    # effective first-order time constant from rise time (t_r,10-90 ~ 2.197*tau for 1st order)
    print("\nrate_tau_s (param) = %.4f s ; expected 10-90%% rise = 2.197*tau = %.1f ms"
          % (PlantParams().rate_tau_s, 2.197 * PlantParams().rate_tau_s * 1000))

    # thrust response ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("THRUST-STEP RESPONSE  (vertical accel az, NED Z-down: az<0 = upward accel)")
    print("=" * 100)
    az = _col(header, data, "az")
    nt = _col(header, data, "normed_thrust")
    pitch = _col(header, data, "pitch")
    roll = _col(header, data, "roll")
    qx = _col(header, data, "qx")
    qy = _col(header, data, "qy")
    vz = _col(header, data, "vz")
    g0 = PlantParams().g
    ld = PlantParams().linear_drag
    # az(NED) = g - a_up*R22 - linear_drag*vz.  R22 = body-up . world-up = cos(tilt).
    # Recover the thrust-map output cleanly despite residual tilt AND drag:
    #   a_up_rec = (g - linear_drag*vz - az) / R22.
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    print("%-16s | %8s %10s | %8s %7s %7s %8s | %9s %7s"
          % ("segment", "normed_T", "cmd_a_up", "az_ss", "R22", "tilt", "vz", "a_up_rec", "gain"))
    for tag in ("thrust_step_pos", "thrust_step_neg"):
        seg = segs[tag]
        t0, t1 = seg["t0"], seg["t1"]
        win = (t >= t0) & (t < t1)
        ss_win = (t >= t0 + 0.5 * (t1 - t0)) & (t < t1)
        nt_val = float(np.median(nt[win]))
        cmd_a_up = g0 * nt_val                       # legacy linear thrust map a_up = g*normed_thrust
        az_ss = float(np.mean(az[ss_win]))
        r22_ss = float(np.mean(r22[ss_win]))
        vz_ss = float(np.mean(vz[ss_win]))
        tilt_deg = float(np.degrees(np.arccos(np.clip(r22_ss, -1, 1))))
        a_up_rec = (g0 - ld * vz_ss - az_ss) / r22_ss if r22_ss != 0 else float("nan")
        gain = a_up_rec / cmd_a_up if cmd_a_up != 0 else float("nan")
        print("%-16s | %8.3f %10.3f | %8.3f %7.3f %7.1f %8.2f | %9.3f %7.3f"
              % (tag, nt_val, cmd_a_up, az_ss, r22_ss, tilt_deg, vz_ss, a_up_rec, gain))
    print("(a_up_rec = (g - linear_drag*vz - az_ss)/R22 removes residual tilt + drag contamination;"
          "\n gain = a_up_rec/cmd_a_up. cmd_a_up uses the legacy linear map a_up = g*normed_thrust ->"
          " gain~1 confirms it. neg step normed_T=0 -> cmd_a_up=0, gain undefined. vz in m/s NED, +down.)")

    # exact analytic cross-check ---------------------------------------------
    rg, rs = PlantParams().rate_gain, PlantParams().rate_sign
    print("\nANALYTIC signed FLU gain (rate_gain*rate_sign) = [%.3f, %.3f, %.3f]  (roll,pitch,yaw)"
          % (rg[0] * rs[0], rg[1] * rs[1], rg[2] * rs[2]))


def main():
    header, data, dt = replay()
    summary(header, data, dt)


if __name__ == "__main__":
    main()
