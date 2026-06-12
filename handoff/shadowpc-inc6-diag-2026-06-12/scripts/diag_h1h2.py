"""SHADOWPC-INC6-DIAG H1/H2: step-0 obs diff (live vs twin simstart) + action-path check.

H1: for every standing run, take the k=0 debug_obs record, rebuild the obs from the
    recorded raw telemetry via fly_rl.build_obs, and diff vs the recorded obs AND vs
    the twin's simstart obs0. Localize which of the 17 dims diverge.
H2: feed the recorded obs through the on-disk inc6 actor; compare actor_mean / wire
    command to the recorded values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import numpy as np
import torch

import fly_rl
from fly_rl import OBS_LABELS, build_obs, load_actor, policy_step
from offline_rollout import make_start, obs_from_truth

CKPT = str(ROOT / "rl" / "checkpoints" / "stage1_inc6_actor.pth")
RUNS = sorted((ROOT / "data" / "runs").glob("20260612_*_inc6_standing_f*"))
BRIDGE = sorted((ROOT / "data" / "runs").glob("20260612_04*_inc6_bridge_f*"))


class T:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def first_steps(run: Path, n=1):
    rows = []
    with open(run / "debug_obs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "header":
                continue
            rows.append(r)
            if len(rows) >= n:
                break
    return rows


def main():
    actor = load_actor(CKPT)

    # twin simstart reference (mixer plant irrelevant for obs0)
    class A:
        gate = 0; handoff_dist = 3.0; handoff_speed = 10.0; thrust0 = -1.0
    st, gate = make_start("simstart", A)
    obs_twin = obs_from_truth(st, gate, 0.0, True).astype(np.float64)
    with torch.no_grad():
        mean_twin = actor(torch.as_tensor(obs_twin[None], dtype=torch.float32))[0].numpy()
    r0, c0, n0 = policy_step(actor, obs_twin.astype(np.float32), 0.0, virtual_flip=True)
    print("TWIN simstart obs0 :", np.round(obs_twin, 5).tolist())
    print("TWIN action        : rate_frd=%s collective=%.4f normed=%.5f"
          % (np.round(r0, 4).tolist(), c0, n0))
    print()

    hdr = f"{'run':<38} {'max|obs_live-obs_rebuilt|':>26} {'max|obs_live-obs_twin|':>24} {'worst dim':>12} {'mean_err':>10} {'act_err':>10}"
    print(hdr)
    obs0s = []
    for run in RUNS + BRIDGE:
        r = first_steps(run, 1)[0]
        obs_live = np.array(r["obs"], dtype=np.float64)
        tel = T(position_ned=np.array(r["pos_ned"]), velocity_ned=np.array(r["vel_ned"]),
                orientation_ned_wxyz=np.array(r["q_raw_wxyz"]),
                angular_rate_body=np.array(r["w_raw"]))
        obs_rb = build_obs(tel, r["gate_index"], 0.0, virtual_flip=True).astype(np.float64)
        d_rb = np.abs(obs_live - obs_rb)
        d_tw = np.abs(obs_live - obs_twin)
        wd = int(np.argmax(d_tw))
        with torch.no_grad():
            mean = actor(torch.as_tensor(obs_live[None].astype(np.float32)))[0].numpy()
        mean_err = float(np.max(np.abs(mean - np.array(r["actor_mean"]))))
        # recompute wire command from recorded obs
        rr, cc, nn = policy_step(actor, obs_live.astype(np.float32), 0.0, virtual_flip=True)
        act_err = float(np.max(np.abs(rr - np.array(r["rate_frd"])))) \
            + abs(cc - r["collective"])
        print(f"{run.name:<38} {float(d_rb.max()):>26.6f} {float(d_tw.max()):>24.5f} "
              f"{OBS_LABELS[wd]:>12} {mean_err:>10.2e} {act_err:>10.2e}")
        obs0s.append(obs_live)

    obs0s = np.array(obs0s[:10])  # standing only
    print("\nper-dim spread across 10 standing step-0 obs (max-min):")
    spread = obs0s.max(0) - obs0s.min(0)
    for i, lab in enumerate(OBS_LABELS):
        mark = " <-- varies" if spread[i] > 1e-4 else ""
        print(f"  {lab:<16} live_f1={obs0s[0][i]:+10.5f} twin={obs_twin[i]:+10.5f} "
              f"diff={obs0s[0][i]-obs_twin[i]:+10.5f} spread={spread[i]:.5f}{mark}")


if __name__ == "__main__":
    main()
