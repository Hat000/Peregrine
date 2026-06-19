"""Can 2-axis fly the gate-4 APPROACH if we skip the gate-0 rest-launch transient? Start trainreset
at gate g (1 m behind), 2-axis look-at, check whether gate-4 is reached AND crossed (the σ_p0 needs
the gate-4 crossing). If a later start lets 2-axis cross gate-4, the faithful 2-axis σ_p0 is
measurable from there."""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import contact_true_eval as CTE
from fly_rl import load_actor, N_GATES

CKPT = str(_ROOT / "rl" / "checkpoints" / "inc8_rc1_seed0_actor.pth")
actor = load_actor(CKPT)
params = CTE._build_plant_params("mixer")


def crossed_g4(result):
    return any(c.gate == 4 and c.verdict == "pass" for c in result.crossings)


def furthest(result):
    return max([c.gate for c in result.crossings if c.verdict == "pass"], default=-1)


def run(start_gate, g_pitch, seed):
    st, tgt, vflip = CTE._build_start("trainreset", start_gate)
    kw = {"lookat": True, "lookat_g_pitch": g_pitch}
    result, _ = CTE.run_episode(actor, st, tgt, vflip, params,
                                max_time=40.0, start_label="trainreset", record_pos=True,
                                estim_emul=True, obs_dim=20, emul_seed=seed, **kw)
    return result.outcome, furthest(result), crossed_g4(result)


for sg in range(0, 5):
    for gp in [0.0, 3.0]:
        rows = [run(sg, gp, s) for s in range(8)]
        n_cross4 = sum(1 for *_, c in rows if c)
        tag = "yaw " if gp == 0.0 else "2ax "
        print(f"  start_gate={sg} {tag}(gp={gp}): cross_g4={n_cross4}/8  "
              f"furthest={[g for _, g, _ in rows]}")
