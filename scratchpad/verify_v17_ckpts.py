"""SUPERSEDED -- ITS CHANNEL LABELS ARE WRONG.  Use verify_v18_ckpts.py.

This script unpacks the action vector as (roll, pitch, yaw, thrust).  The ego action
vector is [thrust, roll, pitch, yaw] (fly_rl.py:748, ``rate_flu = act[1:4]``), so the
column printed here as "pitch" is actually ROLL and the nose-DOWN verdict is meaningless.
Kept only as the record of the error.  Its other flaw: the synthetic at-rest obs is out of
distribution -- every arm answers it with ~zero collective -- so the launch-window question
it claims to answer is only answerable over LOGGED obs (see replay_sweep_v18.py).

Pre-release gate for the v1.7 actors -- run through the REAL deploy loader.

Uses fly_rl.load_ego_actor (not a hand-rolled reconstruction), so this checks the
exact path that will run on the wire: the 21-dim obs assertion, the hardcoded ego
action bounds, and tanh-squash + rescale.  Then it exercises the two states that
matter at launch:

  * LEVEL handoff  -- obs[4] = 0, what the wire hands us at assist-release
  * PAD tilt       -- obs[4] = -0.3107 rad (-17.80 deg), the VQ1 pad at rest

A NEGATIVE pitch action at the LEVEL handoff is the start dive (a_pitch < 0 is
nose-DOWN).  Killing that is the whole point of v1.7 M1+M2, so it is reported
explicitly against the v16 parent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

DEPLOY = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix")
sys.path.insert(0, str(DEPLOY / "rl"))
sys.path.insert(0, str(DEPLOY / "src"))

CKPT_DIR = Path(r"C:\Users\Fengy\Downloads\Projects\wt-fix\data\ego-ckpts-v17-2026-07-21")
PAD_TILT_RAD = -0.3107
OBS_DIM = 21


def build_obs(pitch_rad: float, gate_range_m: float = 10.0) -> torch.Tensor:
    """Quiet at-rest observation, gate centred and valid in slot0, next gate UP."""
    obs = torch.zeros(OBS_DIM)
    obs[4] = pitch_rad
    obs[9:11] = torch.tensor([0.0, 1.0])
    obs[11:16] = torch.tensor([gate_range_m, 0.0, 0.0, 1.0, 1.0])
    return obs


def main() -> int:
    import fly_rl

    ckpts = sorted(CKPT_DIR.glob("*_actor.pth"))
    parent = Path(r"C:\Users\Fengy\Downloads\Projects\Anduril-ego-deploy\data"
                  r"\ego-ckpts-v16-2026-07-19\v16Qs1_actor.pth")
    if parent.exists():
        ckpts.append(parent)

    bad = 0
    for p in ckpts:
        tag = "v16Qs1 (PARENT)" if "v16" in p.name else p.name
        print(f"=== {tag} ===")
        try:
            actor = fly_rl.load_ego_actor(str(p))
        except Exception as e:                                  # noqa: BLE001
            print(f"  FAIL load: {type(e).__name__}: {e}")
            bad += 1
            continue
        print("  loaded OK via deploy load_ego_actor")

        for label, pitch in (("LEVEL handoff", 0.0),
                             ("PAD tilt -17.8", PAD_TILT_RAD)):
            with torch.no_grad():
                raw = actor(build_obs(pitch).unsqueeze(0)).squeeze(0)
            a = torch.tanh(raw)
            if not bool(torch.isfinite(a).all()):
                print(f"  {label:<15} NON-FINITE  {a.tolist()}")
                bad += 1
                continue
            roll, pitch_a, yaw, thr = (float(v) for v in a[:4])
            nose = "DOWN" if pitch_a < 0 else "up  "
            print(f"  {label:<15} roll={roll:+.3f} pitch={pitch_a:+.3f} ({nose}) "
                  f"yaw={yaw:+.3f} thrust={thr:+.3f}")
        print()

    print(f"{len(ckpts)} checkpoint(s) checked, {bad} problem(s).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
