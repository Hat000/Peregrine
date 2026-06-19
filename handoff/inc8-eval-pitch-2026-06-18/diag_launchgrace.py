"""Does 2-axis fly the WITH-VELOCITY course if it survives the rest-launch? Disable the look-at for
the first N control steps (launch grace), then enable full 2-axis. If the drone then flies the course
and crosses gate-4, the breakage is specifically the rest-launch transient (camera far off-axis at
zero speed) and the with-velocity look-at is stable -> a faithful 2-axis gate-4 σ_p0 is measurable
with a launch grace. If it still dies mid-course, 2-axis is intrinsically unflyable on the eval obs."""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import contact_true_eval as CTE
from fly_rl import load_actor

CKPT = str(_ROOT / "rl" / "checkpoints" / "inc8_rc1_seed0_actor.pth")
actor = load_actor(CKPT)
params = CTE._build_plant_params("mixer")

# launch grace via monkeypatch: zero dlook for the first GRACE calls of each episode
_orig = CTE._lookat_dlook_flu
STATE = {"n": 0, "grace": 0}
def _graced(t_cam, gy, gp):
    STATE["n"] += 1
    if STATE["n"] <= STATE["grace"]:
        return np.zeros(3)
    return _orig(t_cam, gy, gp)
CTE._lookat_dlook_flu = _graced


def furthest(result):
    return max([c.gate for c in result.crossings if c.verdict == "pass"], default=-1)


def cross4(result):
    return any(c.gate == 4 and c.verdict == "pass" for c in result.crossings)


def run(grace, gp, seed):
    STATE["n"] = 0
    STATE["grace"] = grace
    st, tgt, vflip = CTE._build_start("trainreset", 0)
    result, _ = CTE.run_episode(actor, st, tgt, vflip, params,
                                max_time=40.0, start_label="trainreset", record_pos=True,
                                estim_emul=True, obs_dim=20, emul_seed=seed,
                                lookat=True, lookat_g_pitch=gp)
    return furthest(result), cross4(result)


for grace in [0, 8, 15, 25, 40]:
    rows = [run(grace, 3.0, s) for s in range(8)]
    nc4 = sum(1 for _, c in rows if c)
    print(f"  launch_grace={grace:3d} steps, 2-axis gp=3: cross_g4={nc4}/8  furthest={[g for g,_ in rows]}")
