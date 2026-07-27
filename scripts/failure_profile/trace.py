"""Print a human-readable death trace for hand-labelling."""
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vg

ROWS = {r["run"]: r for r in pickle.load(open(os.path.join(HERE, "sessions.pkl"), "rb"))}


def trace(run, ntail=34, quiet=False):
    r = ROWS[run]
    recs, seek, timing, meta = vg.load_session(r["path"])
    d = vg.build(recs, seek, meta)
    t = d["t"]
    n = len(t)
    print(f"\n{'='*118}\n{run}   ckpt={r['ckpt']}  lineage={r['lineage']}  "
          f"final={r['final_state']} coll={r['collisions']}  dur={r.get('t_end',0):.2f}s  "
          f"ticks={n}  hz={r.get('hz_med',0):.1f}  zbias={r['zbias']} "
          f"clamp={r['pitch_clamp']} coast={r['obs_coast']} sector={r['sector_mode']}")
    print(f"  gate segments: " + "  ".join(
        f"g{int(g)}:{int((d['gi']==g).sum())}t/{t[d['gi']==g][-1]-t[d['gi']==g][0]:.2f}s"
        for g in np.unique(d["gi"])))
    print(f"  aim_ticks={r.get('aim_ticks')} gates={r.get('aim_gates')} "
          f"max_lat={r.get('aim_max_lat')} max_vert={r.get('aim_max_vert')}")
    i0 = max(0, n - ntail)
    print(f"\n  {'k':>4} {'t':>6} {'gi':>2} {'rho':>6} {'fwdL':>6} {'latL':>6} {'upL':>6} "
          f"{'roll':>6} {'pitch':>6} {'thr':>5} {'rollcmd':>7} {'pitcmd':>7} {'gyroN':>5} "
          f"{'nc':>3} {'src':>7} {'reason':>17} fresh")
    for i in range(i0, n):
        cr, sr = np.cos(d["roll"][i]), np.sin(d["roll"][i])
        cp, sp = np.cos(d["pitch"][i]), np.sin(d["pitch"][i])
        Rl = (np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
              @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        rl = Rl @ d["rel"][i] if np.isfinite(d["rel"][i]).all() else np.array([np.nan] * 3)
        rl[2] += d["zbias"] * vg.ZBIAS_COS
        gn = float(np.linalg.norm(d["w_flu"][i]))
        print(f"  {d['k'][i]:4d} {t[i]:6.3f} {d['gi'][i]:2d} {d['rho'][i]:6.2f} "
              f"{rl[0]:6.2f} {rl[1]:6.2f} {rl[2]:6.2f} "
              f"{np.degrees(d['roll'][i]):6.1f} {np.degrees(d['pitch'][i]):6.1f} "
              f"{d['collective'][i]:5.2f} {d['rate_frd'][i,0]:7.2f} {d['rate_frd'][i,1]:7.2f} "
              f"{gn:5.2f} {d['ncorn'][i]:3.0f} {str(d['rsrc'][i])[:7]:>7} "
              f"{str(d['reason'][i])[:17]:>17} {'F' if d['fresh'][i] else '.'}")
    if quiet:
        return
    keys = ["d_end", "s_end", "dt_since_fresh", "blind_travel_m", "speed_end", "t_star",
            "miss_h", "miss_v", "miss_abs", "rms_radial", "rms_trans",
            "last_up_lev", "last_lat_lev", "last_rho", "blind_frac_1s", "vempty_frac_1s",
            "max_fix_gap_15s", "roll_flips_1s", "roll_max_1s", "gyro_roll_flips_1s",
            "vg_lmin", "vg_reopen", "raw_lmin", "raw_reopen", "v_outward_end", "seg_rho_min",
            "pitch_fenced_frac", "pitchcmd_min_1s", "thrust_med_1s"]
    print("  --- VG features ---")
    out = []
    for k in keys:
        v = r.get(k, None)
        out.append(f"{k}={v:.2f}" if isinstance(v, float) and np.isfinite(v) else f"{k}={v}")
    for i in range(0, len(out), 5):
        print("   " + "  ".join(out[i:i + 5]))
    pe = np.asarray(r.get("p_end", [np.nan] * 3), float)
    ve = np.asarray(r.get("v_end", [np.nan] * 3), float)
    dtf = r.get("dt_since_fresh", np.nan)
    pd_ = pe + ve * (dtf if np.isfinite(dtf) else 0.0)
    print(f"   p_end(drone-gate,leveled)=[{pe[0]:+.2f},{pe[1]:+.2f},{pe[2]:+.2f}]  "
          f"v_end=[{ve[0]:+.2f},{ve[1]:+.2f},{ve[2]:+.2f}]  "
          f"p_at_logend=[{pd_[0]:+.2f},{pd_[1]:+.2f},{pd_[2]:+.2f}] |p|={np.linalg.norm(pd_):.2f}")
    mv = np.asarray(r.get("miss_vec", [np.nan] * 3), float)
    print(f"   miss_vec=[{mv[0]:+.2f},{mv[1]:+.2f},{mv[2]:+.2f}]  "
          f"|miss_h|={r.get('miss_h', float('nan')):.2f} miss_v={r.get('miss_v', float('nan')):+.2f} "
          f"(gate half-opening 0.75 m)")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        trace(a)
