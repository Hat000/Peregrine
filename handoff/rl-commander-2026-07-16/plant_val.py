"""Per-tick IMU-match plant validation (Fengyou's criterion: same accel/gyro every tick = accurate).
Replay each VQ2 sysid_vq2_log.csv COMMAND (a_*) through the numpy rl_plant, faithful_rate ON vs OFF,
and overlay achieved gyro + BODY SPECIFIC FORCE vs the VQ2 IMU, per tick. No position/velocity compared
(pose-blind wire; accel is the ground truth). Frames: world NED (Z down, g=+9.80665 +Z), body FRD."""
import sys, os, csv, numpy as np
ROOT = r"C:\Users\Fengy\Downloads\Projects\Anduril\.claude\worktrees\cool-heyrovsky-e08624"
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "rl"))
from racer.rl_plant import (PlantParams, PlantState, step as rl_step,
                            RATE_GAIN_SMALLSIGNAL_MEASURED, SUPER_RATE_S_FAITHFUL, ALPHA_MAX_RPS2_MEASURED)
import diffaero_dynamics as dda
ACT_MIN = np.array([0.0, -3.14, -3.14, -3.14]); ACT_MAX = np.array([5.0, 3.14, 3.14, 3.14])
def rescale(a): return ACT_MIN + (ACT_MAX - ACT_MIN) * (a + 1.0) / 2.0

def quat_R(q):  # wxyz R_world_body
    w, x, y, z = q; return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def replay(cmd, params, dt):
    ps = PlantState.hover(params=params); vprev = ps.vel.copy(); G = params.g
    W, F = [], []
    for a in cmd:
        act = rescale(a); rate_frd, nt = dda._action_diffaero_to_ctbr_np(act)
        coll = nt * params.hover_thrust
        ps = rl_step(ps, np.concatenate([rate_frd, [coll]]), dt, params)
        a_world = (ps.vel - vprev) / dt; vprev = ps.vel.copy()
        f_body = quat_R(ps.quat).T @ (a_world - np.array([0.0, 0.0, G]))   # IMU specific force, body FRD
        W.append(ps.omega.copy()); F.append(f_body)
    return np.array(W), np.array(F)

def best_map(sim, meas):   # per meas-axis: pick sim-axis + sign maximizing |corr|; return mapped sim + map
    mapped = np.zeros_like(meas); info = []
    for j in range(3):
        best = (-1.0, 0, 1.0)                       # (score=|corr|, simAx, sign)
        for i in range(3):
            a, b = sim[:, i], meas[:, j]
            c = 0.0 if (a.std() < 1e-9 or b.std() < 1e-9) else float(np.corrcoef(a, b)[0, 1])
            if abs(c) > best[0]: best = (abs(c), i, 1.0 if c >= 0 else -1.0)
        sc, i, s = best; mapped[:, j] = s*sim[:, i]; info.append((j, i, s, round(sc, 2)))
    return mapped, info

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))

import glob
for path in sorted(glob.glob(os.path.join("vq2sysid", "*.csv"))):
    name = os.path.basename(path)[:-4]
    d = np.genfromtxt(path, delimiter=",", names=True)
    prog = np.isfinite(d["a_yaw"]) & (d["a_thrust"] != None)          # prog rows have a_* filled
    prog = np.isfinite(d["a_thrust"]) & np.isfinite(d["a_roll"]) & np.isfinite(d["a_pitch"]) & np.isfinite(d["a_yaw"])
    t = d["sim_time_ns"][prog].astype(float)
    cmd = np.stack([d["a_thrust"], d["a_roll"], d["a_pitch"], d["a_yaw"]], axis=-1)[prog]
    gyro = np.stack([d["gyro_x"], d["gyro_y"], d["gyro_z"]], axis=-1)[prog]
    accel = np.stack([d["accel_x"], d["accel_y"], d["accel_z"]], axis=-1)[prog]
    dt = float(np.median(np.diff(t)))/1e9
    print(f"\n================= {name} =================")
    print(f"  prog ticks={prog.sum()}  dt={dt*1000:.1f}ms ({1/dt:.0f}Hz)  "
          f"|cmd| max r/p/y={np.abs(cmd[:,1]).max():.2f}/{np.abs(cmd[:,2]).max():.2f}/{np.abs(cmd[:,3]).max():.2f} thr={cmd[:,0].min():.2f}..{cmd[:,0].max():.2f}")
    FAITH = PlantParams(rate_gain=RATE_GAIN_SMALLSIGNAL_MEASURED, super_rate_s=SUPER_RATE_S_FAITHFUL,
                        alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)
    # These are SINGLE-AXIS diagnostics: pick the active RATE axis (0=roll,1=pitch,2=yaw) or thrust-only.
    act = int(np.argmax([np.abs(cmd[:, 1]).max(), np.abs(cmd[:, 2]).max(), np.abs(cmd[:, 3]).max()]))
    axname = "roll pitch yaw".split()[act]
    cmd_rate = np.abs(rescale(cmd)[:, 1+act])            # commanded rate magnitude (rad/s), active axis
    is_rate = cmd_rate.max() > 0.3
    W = {}; F = {}
    for tag, params in (("flat", PlantParams()), ("faithful", FAITH)):
        W[tag], F[tag] = replay(cmd, params, dt)
    if is_rate:
        # sign of the VQ2<->sim map on the active axis (yaw carries the -1 legacy alias)
        sgn = 1.0 if np.corrcoef(W["flat"][:, act], gyro[:, act])[0, 1] >= 0 else -1.0
        for tag in ("flat", "faithful"):
            r = rmse(sgn*W[tag][:, act], gyro[:, act])
            print(f"  [{tag:9s}] {axname}-rate per-tick RMSE vs VQ2 gyro = {r:.3f} rad/s   (map sign {sgn:+.0f})")
        print(f"     AMP-GAIN {axname} (mean|achieved rate|/|cmd rate|, binned):")
        for lo, hi in ((0.3, 1.0), (1.0, 2.0), (2.0, 3.2)):
            m = (cmd_rate >= lo) & (cmd_rate < hi)
            if m.sum() < 3: continue
            gain = lambda Wt: np.abs(Wt[:, act])[m].mean()/cmd_rate[m].mean()
            vq = np.abs(gyro[:, act])[m].mean()/cmd_rate[m].mean()
            print(f"       |cmd|~[{lo:.1f},{hi:.1f}) n={m.sum():3d}:  VQ2={vq:.2f}   flat={gain(W['flat']):.2f}   faithful={gain(W['faithful']):.2f}")
    else:  # thrust-only capture -> accel_z channel (specific force). Caveat: DiffAero IC=hover, VQ2 has vel.
        for tag in ("flat", "faithful"):
            rz = rmse(F[tag][:, 2], accel[:, 2])
            print(f"  [{tag:9s}] accel_z per-tick RMSE vs VQ2 = {rz:.2f} m/s^2   "
                  f"(sim f_z range {F[tag][:,2].min():.1f}..{F[tag][:,2].max():.1f}  VQ2 {accel[:,2].min():.1f}..{accel[:,2].max():.1f})")
        print("     [accel caveat] DiffAero replays from hover@0vel; VQ2 has takeoff+fwd vel -> needs gyro-forced/vel-matched replay for a clean read")
