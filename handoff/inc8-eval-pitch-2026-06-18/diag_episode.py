"""Single-episode diagnostic: run yaw-only and 2-axis from trainreset (with the YAW fix in place)
and log per-step geometry/rates to see WHERE and WHY 2-axis dies."""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import contact_true_eval as CTE
from fly_rl import load_actor, _GATE_POS_ZUP, _FLIP, _TRAIN_DT
import numpy as np

CKPT = str(_ROOT / "rl" / "checkpoints" / "inc8_rc1_seed0_actor.pth")
actor = load_actor(CKPT)
params = CTE._build_plant_params("mixer")

# Monkeypatch a per-step logger into the look-at block by wrapping _lookat_dlook_flu.
LOG = []
_orig = CTE._lookat_dlook_flu
def _logged(t_cam, gy, gp):
    d = _orig(t_cam, gy, gp)
    LOG.append((np.asarray(t_cam, float).copy(), d.copy()))
    return d
CTE._lookat_dlook_flu = _logged

def run(label, lookat_kw):
    LOG.clear()
    st, tgt, vflip = CTE._build_start("trainreset", 0)
    result, pos = CTE.run_episode(actor, st, tgt, vflip, params,
                                  max_time=40.0, start_label="trainreset", record_pos=True,
                                  estim_emul=True, obs_dim=20, emul_seed=0, **lookat_kw)
    print("="*78)
    print(f"{label}: outcome={result.outcome}  n_steps_lookat_fired={len(LOG)}  traj_len={len(pos)}")
    for c in result.crossings:
        print(f"   crossing gate={c.gate} verdict={c.verdict} linf={c.linf:.3f} t={c.t:.2f}")
    # final few positions (NED) + gate0/1 positions
    print(f"   gate0_ned={_GATE_POS_ZUP[0]*_FLIP}  gate1_ned={_GATE_POS_ZUP[1]*_FLIP}")
    print(f"   start_ned={pos[0]}  end_ned={pos[-1]}  (n={len(pos)})")
    # sample dlook magnitudes
    if LOG:
        dlooks = np.array([d for _, d in LOG])
        tcams = np.array([t for t, _ in LOG])
        print(f"   dlook over fired steps: roll[{dlooks[:,0].min():+.3f},{dlooks[:,0].max():+.3f}] "
              f"pitch[{dlooks[:,1].min():+.3f},{dlooks[:,1].max():+.3f}] "
              f"yaw[{dlooks[:,2].min():+.3f},{dlooks[:,2].max():+.3f}]")
        print(f"   first 3 fired (t_cam -> dlook):")
        for i in range(min(3, len(LOG))):
            print(f"      t_cam={tcams[i]}  dlook={dlooks[i]}")
    # full trajectory altitude/pos trace every 15 steps until death
    print("   traj (every 10 steps): k  pos_ned(x,y,z)")
    for k in range(0, len(pos), 10):
        print(f"      k={k:4d} t={k*_TRAIN_DT:5.2f}  pos={np.round(pos[k],2)}")

run("YAW-ONLY", {"lookat": True, "lookat_g_pitch": 0.0})
print()
run("2-AXIS", {})   # auto -> full 2-axis
