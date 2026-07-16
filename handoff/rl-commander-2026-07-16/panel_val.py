"""Racing-regime rate validation: replay the REAL deploy command (rate_frd) through the numpy plant,
flat vs faithful, and compare achieved body-rate to the REAL gyro obs[5:8] tick-by-tick, in the actual
vpef8nc oscillation (osc_*) and the deep vpeffs0 accumulation (deep_*). obs[5:8]=gyro=real sensor GT."""
import sys, os, glob, json, numpy as np
ROOT = r"C:\Users\Fengy\Downloads\Projects\Anduril\.claude\worktrees\cool-heyrovsky-e08624"
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "rl"))
from racer.rl_plant import (PlantParams, PlantState, step as rl_step,
                            RATE_GAIN_SMALLSIGNAL_MEASURED, SUPER_RATE_S_FAITHFUL, ALPHA_MAX_RPS2_MEASURED)
FAITH = dict(rate_gain=RATE_GAIN_SMALLSIGNAL_MEASURED, super_rate_s=SUPER_RATE_S_FAITHFUL,
             alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)

def replay_rates(cmd_frd, coll, dt, params):
    """Feed the logged FRD rate command + collective straight into the plant; return achieved omega (FRD)."""
    ps = PlantState.hover(params=params); W = []
    for k in range(len(cmd_frd)):
        ps = rl_step(ps, np.concatenate([cmd_frd[k], [coll[k]]]), dt, params)
        W.append(ps.omega.copy())
    return np.array(W)

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    t = np.array([r["sim_time_ns"] for r in rows], float)
    cmd = np.array([r["rate_frd"] for r in rows], float)                     # commanded body rate FRD
    ach = np.array([r["obs"][5:8] for r in rows], float)                     # achieved gyro (real GT)
    coll = np.array([r.get("collective", 0.2656) for r in rows], float)
    dt = float(np.median(np.diff(t)))/1e9
    return cmd, ach, coll, dt

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))

for group in ("osc", "deep"):
    files = sorted(glob.glob(os.path.join("panels", group+"_*.jsonl")))
    C = np.zeros((0, 3)); A = np.zeros((0, 3)); Wf = np.zeros((0, 3)); Wt = np.zeros((0, 3))
    for f in files:
        cmd, ach, coll, dt = load(f)
        Wflat = replay_rates(cmd, coll, dt, PlantParams())
        Wfaith = replay_rates(cmd, coll, dt, PlantParams(**FAITH))
        C = np.vstack([C, cmd]); A = np.vstack([A, ach]); Wf = np.vstack([Wf, Wflat]); Wt = np.vstack([Wt, Wfaith])
    # per-axis sign map plant->real (yaw carries the -1 legacy alias)
    print(f"\n===== {group.upper()}  ({len(files)} flights, {len(C)} ticks pooled) =====")
    for ax, nm in enumerate("roll pitch yaw".split()):
        m = np.abs(C[:, ax]) > 0.15                                          # ticks with a real command
        if m.sum() < 20:
            print(f"  {nm}: <20 active ticks"); continue
        sgn = 1.0 if np.corrcoef(Wf[m, ax], A[m, ax])[0, 1] >= 0 else -1.0
        rf = rmse(sgn*Wf[m, ax], A[m, ax]); rt = rmse(sgn*Wt[m, ax], A[m, ax])
        # realized gain (|achieved|/|cmd|) binned by |cmd|, VQ2 vs flat vs faithful
        gains = []
        for lo, hi in ((0.15, 0.4), (0.4, 0.7), (0.7, 3.2)):
            b = m & (np.abs(C[:, ax]) >= lo) & (np.abs(C[:, ax]) < hi)
            if b.sum() < 10: gains.append(None); continue
            vq = np.abs(A[b, ax]).mean()/np.abs(C[b, ax]).mean()
            gf = np.abs(Wf[b, ax]).mean()/np.abs(C[b, ax]).mean()
            gt = np.abs(Wt[b, ax]).mean()/np.abs(C[b, ax]).mean()
            gains.append((lo, hi, b.sum(), vq, gf, gt))
        print(f"  {nm:5s} RMSE flat={rf:.3f} faithful={rt:.3f} rad/s (sign {sgn:+.0f}, n={m.sum()})")
        for g in gains:
            if g: print(f"        |cmd|~[{g[0]:.2f},{g[1]:.2f}) n={g[2]:3d}:  VQ2={g[3]:.2f}  flat={g[4]:.2f}  faithful={g[5]:.2f}")
