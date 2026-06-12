"""rate_track.py -- realized-vs-commanded body-rate tracking fidelity, binned by
airspeed, for LIVE recordings and the TWIN. The gap-analysis core (branch 2b).

Commanded = wire rate_frd (FRD, what policy_step emitted).
Realized (live) = true FRD body rate = -w_raw   (FRAME-AUDIT conjugation).
Realized (twin) = st.omega  (plant true FRD body rate, rate_sign [1,1,1]).

For each axis we lag-align realized[k] against commanded[k-d] (d in 1..3, pick the d
maximizing |corr| pooled), then report slope (gain) via least-squares through points
with |cmd|>0.3 rad/s, split into airspeed bins. Gain<1 => the drone under-rotates vs
command (mixer authority loss); this is what would widen/slow the live trajectory.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))
AX = ["roll", "pitch", "yaw"]
BINS = [(0, 6), (6, 9), (9, 12), (12, 15), (15, 22)]


def best_lag(cmd, real, lags=(0, 1, 2, 3)):
    best, bd = -1, 1
    for d in lags:
        if d == 0:
            c, r = cmd, real
        else:
            c, r = cmd[:-d], real[d:]
        m = np.abs(c) > 0.3
        if m.sum() < 20:
            continue
        cc = np.corrcoef(c[m], r[m])[0, 1]
        if abs(cc) > best:
            best, bd = abs(cc), d
    return bd


def gains_by_speed(cmd, real, speed, label):
    """cmd,real: (N,3); speed:(N,). Print per-axis lag, then gain per speed bin."""
    print(f"\n---- {label}  (N={len(cmd)}) ----")
    for j, ax in enumerate(AX):
        d = best_lag(cmd[:, j], real[:, j])
        c = cmd[:-d, j] if d else cmd[:, j]
        r = real[d:, j] if d else real[:, j]
        sp = speed[d:] if d else speed
        row = []
        for lo, hi in BINS:
            m = (np.abs(c) > 0.3) & (sp >= lo) & (sp < hi)
            if m.sum() < 15:
                row.append("   --   ")
                continue
            g = float(np.sum(c[m] * r[m]) / np.sum(c[m] * c[m]))
            row.append(f"{g:5.2f}({m.sum():3d})")
        print(f"  {ax:5s} lag={d}  " + "  ".join(f"{lo}-{hi}:{v}" for (lo, hi), v in zip(BINS, row)))


def load_live(run_dir: Path):
    cmd, real, spd = [], [], []
    for line in (run_dir / "debug_obs.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("type") == "header" or "w_raw" not in d:
            continue
        cmd.append(d["rate_frd"])
        real.append([-v for v in d["w_raw"]])      # true FRD = -w_raw
        spd.append(float(np.linalg.norm(d["vel_ned"])))
    return np.array(cmd), np.array(real), np.array(spd)


def run_twin(start, **kw):
    import offline_rollout as oro
    from fly_rl import _HOVER_THRUST, _FLIP, _GATE_POS_ZUP
    import crab_twin_rollout as ct
    actor = oro.load_actor(str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth"))
    params = ct.mixer_params()
    from types import SimpleNamespace
    args = SimpleNamespace(start=start, gate=0, handoff_dist=kw.get("dist", 3.0),
                           handoff_speed=kw.get("speed", 10.0), thrust0=-1.0)
    st, gate = oro.make_start(start, args)
    from racer.rl_plant import step as plant_step
    dt = oro._TRAIN_DT
    last_normed, cmd, real, spd = 0.0, [], [], []
    for k in range(int(40 / dt)):
        obs = oro.obs_from_truth(st, gate, last_normed, True)
        rate_frd, collective, last_normed = oro.policy_step(actor, obs, 0.0, True, 0.0)
        collective = last_normed * _HOVER_THRUST
        prev = st.pos.copy()
        st = plant_step(st, np.concatenate([rate_frd, [collective]]), dt, params)
        cmd.append(rate_frd.copy()); real.append(st.omega.copy())
        spd.append(float(np.linalg.norm(st.vel)))
        ev = oro.gate_event(prev, st.pos, gate)
        if ev == "pass":
            if gate == 5:
                break
            gate += 1
        elif ev in ("collision", "miss"):
            break
    return np.array(cmd), np.array(real), np.array(spd)


if __name__ == "__main__":
    base = ROOT / "data" / "runs"
    # LIVE
    for name in ["20260612_183920_rl_inc6_frameaudit_std_f1",
                 "20260612_184209_rl_inc6_frameaudit_brg_f1",
                 "20260612_184246_rl_inc6_frameaudit_brg_f2"]:
        c, r, s = load_live(base / name)
        gains_by_speed(c, r, s, f"LIVE {name[-20:]}")
    # TWIN
    c, r, s = run_twin("simstart")
    gains_by_speed(c, r, s, "TWIN simstart (standing)")
    c, r, s = run_twin("handoff", dist=3.0, speed=10.0)
    gains_by_speed(c, r, s, "TWIN handoff (bridge)")
