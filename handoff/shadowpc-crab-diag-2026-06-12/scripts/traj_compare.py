"""traj_compare.py -- closed-loop twin (simstart) vs live std_f1, tick-aligned.

Both start from the IDENTICAL physical pose (simstart == the live tilted-pad spawn)
and share the obs+action code, so any divergence is the residual PLANT gap (live VQ1
sim vs rl_plant mixer twin). Tracks where the standing-start trajectory diverges and
whether it reproduces the live gate-3 clip (~1.2 m short / 0.7 m low of the twin).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))
import offline_rollout as oro
import crab_twin_rollout as ct
from fly_rl import _HOVER_THRUST, _FLIP, _GATE_POS_ZUP
from racer.rl_plant import step as plant_step
GATE_NED = _GATE_POS_ZUP * _FLIP


def twin_traj(start="simstart"):
    actor = oro.load_actor(str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth"))
    params = ct.mixer_params()
    from types import SimpleNamespace
    st, gate = oro.make_start(start, SimpleNamespace(start=start, gate=0,
                              handoff_dist=3.0, handoff_speed=10.0, thrust0=-1.0))
    dt, last_normed, traj = oro._TRAIN_DT, 0.0, []
    for k in range(int(40 / dt)):
        obs = oro.obs_from_truth(st, gate, last_normed, True)
        rate_frd, collective, last_normed = oro.policy_step(actor, obs, 0.0, True, 0.0)
        st = plant_step(st, np.concatenate([rate_frd, [last_normed * _HOVER_THRUST]]), dt, params)
        traj.append(st.pos.copy())
        ev = oro.gate_event(traj[-2] if len(traj) > 1 else st.pos, st.pos, gate)
        # recompute event with proper prev
        gate_done = False
        if k > 0:
            ev = oro.gate_event(prev, st.pos, gate)
            if ev == "pass":
                print(f"  TWIN gate {gate} @ t={(k+1)*dt:5.2f}s pos={np.round(st.pos,1).tolist()}")
                if gate == 5:
                    gate_done = True
                gate += 1
            elif ev in ("collision", "miss"):
                print(f"  TWIN {ev} @ gate {gate} t={(k+1)*dt:.2f}s pos={np.round(st.pos,1).tolist()}")
                gate_done = True
        prev = st.pos.copy()
        if gate_done:
            break
    return np.array(traj)


def live_traj(name):
    rows = [json.loads(l) for l in (ROOT / "data/runs" / name / "debug_obs.jsonl").read_text().splitlines()]
    return np.array([d["pos_ned"] for d in rows if "pos_ned" in d])


if __name__ == "__main__":
    print("TWIN simstart gate passes:")
    tw = twin_traj("simstart")
    lv = live_traj("20260612_183920_rl_inc6_frameaudit_std_f1")
    n = min(len(tw), len(lv))
    print(f"\nTick-aligned divergence (twin vs live std_f1), N={n}:")
    print("  k     t   live_pos_ned              twin_pos_ned            |dpos| dN dE dD")
    for k in range(0, n, 15):
        d = tw[k] - lv[k]
        print(f"  {k:3d} {(k+1)*0.0333:5.2f}  "
              f"[{lv[k][0]:7.1f},{lv[k][1]:6.1f},{lv[k][2]:6.1f}]  "
              f"[{tw[k][0]:7.1f},{tw[k][1]:6.1f},{tw[k][2]:6.1f}]  "
              f"{np.linalg.norm(d):4.1f}  {d[0]:+5.1f}{d[1]:+5.1f}{d[2]:+5.1f}")
    # gate-3 region: where live is near gate3 x=-111.5
    g3 = GATE_NED[3]
    print(f"\n  gate3 centre NED = {np.round(g3,1).tolist()}  (opening; clip = |y|,|z| offset)")
    for tag, tr in [("live", lv), ("twin", tw)]:
        # closest approach in x to gate3 plane
        i = int(np.argmin(np.abs(tr[:, 0] - g3[0])))
        off = tr[i] - g3
        print(f"  {tag} nearest gate3 x-plane @k={i}: pos={np.round(tr[i],2).tolist()} "
              f"lateral_off(E)={off[1]:+.2f} vert_off(D)={off[2]:+.2f}")
