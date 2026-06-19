"""g_pitch magnitude sweep + attitude/geometry log to pin the 2-axis breakage mechanism.
Runs single trainreset episodes (emul_seed sweep) at several g_pitch; reports outcome + furthest gate.
Also logs attitude(euler)/speed/range/t_cam at the first fired look-at steps for g_pitch=3."""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import contact_true_eval as CTE
from fly_rl import load_actor
from scipy.spatial.transform import Rotation as Rot

CKPT = str(_ROOT / "rl" / "checkpoints" / "inc8_rc1_seed0_actor.pth")
actor = load_actor(CKPT)
params = CTE._build_plant_params("mixer")


def furthest_gate(result):
    g = -1
    for c in result.crossings:
        if c.verdict == "pass":
            g = max(g, c.gate)
    return g


def run_one(g_pitch, seed):
    st, tgt, vflip = CTE._build_start("trainreset", 0)
    kw = {"lookat": True, "lookat_g_pitch": g_pitch} if g_pitch is not None else {"lookat": False}
    result, _ = CTE.run_episode(actor, st, tgt, vflip, params,
                                max_time=40.0, start_label="trainreset", record_pos=True,
                                estim_emul=True, obs_dim=20, emul_seed=seed, **kw)
    return result.outcome, furthest_gate(result)


print("g_pitch sweep (5 seeds each): outcome / furthest passed gate")
for gp in [None, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    rows = [run_one(gp, s) for s in range(5)]
    label = "lookat-OFF" if gp is None else f"g_pitch={gp}"
    reached4 = sum(1 for _, g in rows if g >= 4)
    print(f"  {label:14s}: {['g%d/%s' % (g, o[:4]) for o, g in rows]}  reach_g4={reached4}/5")
