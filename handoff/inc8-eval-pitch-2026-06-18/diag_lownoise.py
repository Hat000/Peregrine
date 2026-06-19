"""Isolate obs-fidelity vs intrinsic: run 2-axis through the TRUSTED run_episode but with a
near-zero-noise EmulConfig (KF tracks truth tightly -> obs ~= truth). If 2-axis flies under
low noise but not under nominal DR, the breakage is the eval/train obs gap (#37). If it crashes
even under low noise, the look-at pitch is intrinsically destabilizing this policy in the harness."""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import contact_true_eval as CTE
from fly_rl import load_actor, N_GATES
from estimator_emul import EmulConfig

CKPT = str(_ROOT / "rl" / "checkpoints" / "inc8_rc1_seed0_actor.pth")
actor = load_actor(CKPT)
params = CTE._build_plant_params("mixer")

NOMINAL = EmulConfig()
LOWNOISE = EmulConfig(sigma_lat_lo=0.005, sigma_lat_hi=0.005, bias_mag_lo=0.0, bias_mag_hi=0.0,
                      inject_bias=False, imu_accel_noise=0.01, pos_std_init=0.05, vel_std_init=0.05)


def furthest(result):
    return max([c.gate for c in result.crossings if c.verdict == "pass"], default=-1)


def run(cfg, g_pitch, seed):
    st, tgt, vflip = CTE._build_start("trainreset", 0)
    kw = {"lookat": True, "lookat_g_pitch": g_pitch}
    result, _ = CTE.run_episode(actor, st, tgt, vflip, params,
                                max_time=40.0, start_label="trainreset", record_pos=True,
                                estim_emul=True, obs_dim=20, emul_seed=seed, emul_config=cfg, **kw)
    return result.outcome, furthest(result)


for cfgname, cfg in [("NOMINAL-DR", NOMINAL), ("LOW-NOISE", LOWNOISE)]:
    print(f"\n{cfgname}:")
    for gp in [0.0, 1.0, 3.0]:
        rows = [run(cfg, gp, s) for s in range(5)]
        reach4 = sum(1 for _, g in rows if g >= 4)
        print(f"  g_pitch={gp}: {['g%d/%s' % (g, o[:4]) for o, g in rows]}  reach_g4={reach4}/5")
