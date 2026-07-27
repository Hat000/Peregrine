import collections
import os
import pickle

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
rows = pickle.load(open(os.path.join(OUT, "sessions.pkl"), "rb"))
ok = [r for r in rows if not str(r.get("note", "")).startswith("ERR") and r.get("n_ticks", 0) >= 8]
print(f"total={len(rows)} usable={len(ok)}")


def g(r, k, d=np.nan):
    v = r.get(k, d)
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def q(name, vals, qs=(5, 25, 50, 75, 90, 99)):
    a = np.array([v for v in vals if np.isfinite(v)])
    if not a.size:
        print(f"  {name}: n=0"); return
    s = "  ".join(f"p{x}={np.percentile(a, x):+.2f}" for x in qs)
    print(f"  {name}: n={len(a)} {s} max={a.max():+.2f}")


print("\n=== final_state x lineage ===")
c = collections.Counter((r["lineage"], r["final_state"]) for r in ok)
lin = sorted({r["lineage"] for r in ok})
print("            " + "".join(f"{s:>10}" for s in ("CRASH", "SIM_RESET", "STALLED")))
for L in lin:
    print(f"  {L:10s}" + "".join(f"{c[(L, s)]:>10d}" for s in ("CRASH", "SIM_RESET", "STALLED")))

cr = [r for r in ok if r["final_state"] == "CRASH"]
print(f"\nCRASH sessions: {len(cr)}")

print("\n=== t_end (death time) ===")
q("all", [g(r, "t_end") for r in cr])
print("  t_end<=2.5s:", sum(1 for r in cr if g(r, "t_end") <= 2.5),
      " t_end<=2.5 & max_gate==0:", sum(1 for r in cr if g(r, "t_end") <= 2.5 and g(r, "max_gate") == 0))
print("  max_gate==0:", sum(1 for r in cr if g(r, "max_gate") == 0))

print("\n=== fit availability ===")
print("  fit_ok:", sum(1 for r in cr if r.get("fit_ok")), "/", len(cr))
print("  no fresh fix at all:", sum(1 for r in cr if g(r, "n_fresh") == 0))

print("\n=== d_end (|drone-gate| at last fresh fix) ===")
q("d_end", [g(r, "d_end") for r in cr])
for th in (2, 3, 3.5, 4, 5, 8):
    print(f"   d_end> {th}: {sum(1 for r in cr if g(r,'d_end')>th)}")

print("\n=== dt_since_fresh / blind_travel ===")
q("dt_since_fresh", [g(r, "dt_since_fresh") for r in cr])
q("blind_travel_m", [g(r, "blind_travel_m") for r in cr])

print("\n=== miss_h / miss_v (fitted ballistic closest approach) ===")
q("miss_h", [g(r, "miss_h") for r in cr])
q("miss_v", [g(r, "miss_v") for r in cr])
q("s_end (how far short of closest approach the log ends)", [g(r, "s_end") for r in cr])
q("t_star", [g(r, "t_star") for r in cr])

print("\n=== last_up_lev (z_bias-corrected gate-above-drone at last fix) ===")
q("last_up_lev", [g(r, "last_up_lev") for r in cr])
q("last_lat_lev", [g(r, "last_lat_lev") for r in cr])
q("last_rho", [g(r, "last_rho") for r in cr])

print("\n=== residual split (THE separator) ===")
q("rms_radial", [g(r, "rms_radial") for r in cr])
q("rms_trans", [g(r, "rms_trans") for r in cr])
rr = [g(r, "rms_radial") / g(r, "rms_trans") for r in cr
      if np.isfinite(g(r, "rms_radial")) and g(r, "rms_trans") > 1e-6]
q("radial/transverse ratio", rr)

print("\n=== aim_off ===")
naim = [r for r in ok if g(r, "aim_ticks") > 0]
print(f"  sessions with aim_off ticks: {len(naim)}")
print("  aim ticks total:", int(sum(g(r, "aim_ticks") for r in naim)))
print("  by lineage:", collections.Counter(r["lineage"] for r in naim))
print("  gates touched:", collections.Counter(tuple(r.get("aim_gates") or []) for r in naim).most_common(10))
print("  max lat/vert:", max((g(r, "aim_max_lat") for r in naim), default=0),
      max((g(r, "aim_max_vert") for r in naim), default=0))
print("  aim_key present (null or not):", sum(1 for r in ok if g(r, "aim_key_present") > 0))

print("\n=== death gate index (gi_end) x died-far ===")
cnt = collections.Counter()
for r in cr:
    far = g(r, "d_end") > 3.5
    cnt[(int(g(r, "gi_end")) if np.isfinite(g(r, "gi_end")) else -1, "far" if far else "near")] += 1
gates = sorted({k[0] for k in cnt})
print("  gate   near    far")
for gg in gates:
    print(f"  {gg:4d} {cnt[(gg,'near')]:6d} {cnt[(gg,'far')]:6d}")

print("\n=== blackout / thrash in last 1 s ===")
q("blind_frac_1s", [g(r, "blind_frac_1s") for r in cr])
q("vempty_frac_1s", [g(r, "vempty_frac_1s") for r in cr])
q("roll_flips_1s", [g(r, "roll_flips_1s") for r in cr])
q("hz_med", [g(r, "hz_med") for r in ok])

print("\n=== speed at end ===")
q("speed_end", [g(r, "speed_end") for r in cr])
q("v_outward", [g(r, "v_outward") for r in cr])
