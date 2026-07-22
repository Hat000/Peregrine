"""Pre-release gate for the v1.8 actors -- run through the REAL deploy command path.

Supersedes verify_v17_ckpts.py, which decoded the action vector BY HAND as
(roll, pitch, yaw, thrust).  That labelling is WRONG: the ego action vector is
[thrust, roll, pitch, yaw] (fly_rl.py:748 ``rate_flu = act[1:4]``), so that script's
"pitch" column was actually ROLL.  This one never touches the raw action -- it calls
fly_rl.policy_step and reports the FRD body-rate command that is actually streamed to
the vehicle, so a channel cannot be mislabelled.

Sign chain, re-derived from code rather than memory (fly_rl.py:748-759):
    rate_flu   = act[1:4]                       # [roll, pitch, yaw], body FLU
    virtual_flip: rate_flu = diag(-1,-1,+1) @ rate_flu
    rate_frd   = rate_flu * [1, -1, 1]          # _ACT_FLU_TO_FRD
  => with virtual_flip ON (deploy default):  rate_frd[1] = +act[2]
  => NOSE-DOWN is rate_frd[1] < 0.   (line 766 fences exactly this: max(cmd, 0.0))

Two states are exercised, both taken from the wire:
  * LEVEL handoff -- obs[4] = 0, what assist-release hands the policy
  * PAD tilt      -- obs[4] = -0.3107 rad (-17.80 deg), the VQ1 pad at rest (n=100)

The v1.8 claim under test is that killing M1 (handoff_spawn_frac 0.33 -> 0.0) removes
the release-pitch regression v1.7 introduced (v1.7 pitched 62% harder nose-down at
release on 385-386/388 replayed flights).  So the LEVEL row is printed for every arm
alongside the v16 parent, and the v18-vs-v17 delta is called out.

Usage:  py verify_v18_ckpts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

DEPLOY = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix")
sys.path.insert(0, str(DEPLOY / "rl"))
sys.path.insert(0, str(DEPLOY / "src"))

DATA = Path(r"C:\Users\Fengy\Downloads\Projects\wt-fix\data")
PAD_TILT_RAD = -0.3107          # measured resting obs[4], n=100
OBS_DIM = 21

# The arms, newest first.  (label, path)
ARMS = [
    ("v18Qs1  <- DEPLOY PICK", DATA / "ego-ckpts-v18-2026-07-22" / "v18Qs1_actor.pth"),
    ("v18Qs0", DATA / "ego-ckpts-v18-2026-07-22" / "v18Qs0_actor.pth"),
    ("v18Ws0", DATA / "ego-ckpts-v18-2026-07-22" / "v18Ws0_actor.pth"),
    ("v17Qs0  (M1 ON)", DATA / "ego-ckpts-v17-2026-07-21" / "v17Qs0_actor.pth"),
    ("v17Qs1  (M1 ON)", DATA / "ego-ckpts-v17-2026-07-21" / "v17Qs1_actor.pth"),
    ("v16Qs1  (PARENT)", DATA / "ego-ckpts-v16-2026-07-19" / "v16Qs1_final_actor.pth"),
]


def build_obs(pitch_rad: float, gate_range_m: float = 10.0) -> np.ndarray:
    """Quiet at-rest observation: gate centred and valid in slot0, next gate UP."""
    obs = np.zeros(OBS_DIM, dtype=np.float64)
    obs[4] = pitch_rad
    obs[9:11] = (0.0, 1.0)                       # coarse sector: next gate UP
    obs[11:16] = (gate_range_m, 0.0, 0.0, 1.0, 1.0)   # slot0: range, bearing, valid
    return obs


def main() -> int:
    import fly_rl

    print("rate_frd = FRD body-rate command actually streamed to the vehicle.")
    print("rate_frd[1] < 0 == NOSE-DOWN (the axis fly_rl.py:769 fences).")
    print("Fences OFF here -- we want the policy's RAW intent, not the fenced result.\n")
    print(f"{'arm':<24} {'state':<15} {'wx(roll)':>9} {'wy(pitch)':>10} "
          f"{'wz(yaw)':>8} {'coll':>6}   nose")
    print("-" * 92)

    level_pitch: dict[str, float] = {}
    bad = 0
    for label, path in ARMS:
        if not path.exists():
            print(f"{label:<24} MISSING {path}")
            bad += 1
            continue
        try:
            actor = fly_rl.load_ego_actor(str(path))
        except Exception as e:                                   # noqa: BLE001
            print(f"{label:<24} FAIL load: {type(e).__name__}: {e}")
            bad += 1
            continue

        for state, pitch in (("LEVEL handoff", 0.0), ("PAD tilt -17.8", PAD_TILT_RAD)):
            with torch.no_grad():
                rate_frd, coll, _ = fly_rl.policy_step(
                    actor, build_obs(pitch), virtual_flip=True)
            if not np.isfinite(rate_frd).all() or not np.isfinite(coll):
                print(f"{label:<24} {state:<15} NON-FINITE {rate_frd} {coll}")
                bad += 1
                continue
            nose = "DOWN" if rate_frd[1] < 0 else "up"
            print(f"{label:<24} {state:<15} {rate_frd[0]:>+9.3f} {rate_frd[1]:>+10.3f} "
                  f"{rate_frd[2]:>+8.3f} {coll:>6.3f}   {nose}")
            if state.startswith("LEVEL"):
                level_pitch[label] = float(rate_frd[1])
        print()

    # The v1.8 thesis, stated as a number.
    pick = level_pitch.get("v18Qs1  <- DEPLOY PICK")
    v17 = [v for k, v in level_pitch.items() if k.startswith("v17")]
    parent = level_pitch.get("v16Qs1  (PARENT)")
    if pick is not None and v17 and parent is not None:
        print("RELEASE-PITCH CHECK (LEVEL handoff, nose-down rate; less negative = better)")
        print(f"  v16 parent {parent:+.3f}   v17 mean {np.mean(v17):+.3f}   "
              f"v18 pick {pick:+.3f}")
        print(f"  v18 vs v17: {pick - float(np.mean(v17)):+.3f} rad/s   "
              f"v18 vs parent: {pick - parent:+.3f} rad/s")

    print(f"\n{len(ARMS)} checkpoint(s) checked, {bad} problem(s).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
