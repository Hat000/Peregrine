"""ADJUDICATE ego_det_geometric. Three questions, in order:
 (1) DID THE MASK ACTUALLY FIRE? An arm that did not engage cannot be judged on outcome.
 (2) WHAT DID IT COST? +972 us/tick was predicted = 3.3% of tick budget; freshness is the one
     CONFIRMED lever, so measure achieved rate + fresh-fix fraction, not just gates.
 (3) OUTCOME vs matched control -- and check for a SESSION effect before trusting a cross-session read.
"""
import os as _os
# Corpus root. Defaults to the in-repo flight corpus; override for an external snapshot:
#   PEREGRINE_RUNS=/path/to/runs python3 <this script>
_RUNS = _os.environ.get("PEREGRINE_RUNS", "data/runs")

import json, glob, os
import numpy as np

def load(root):
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        mp, op = os.path.join(d, "meta.json"), os.path.join(d, "ego_obs.jsonl")
        if not (os.path.exists(mp) and os.path.exists(op)): continue
        try:
            m = json.load(open(mp, encoding="utf-8"))
            recs = [json.loads(l) for l in open(op, encoding="utf-8") if l.strip()]
        except Exception: continue
        if len(recs) < 5: continue
        yield m, recs, os.path.basename(d)

def cell(m):
    return (str(m.get("ego_ckpt","")).split("/")[-1] == "v19Ws0_actor.pth"
            and float(m.get("rate_hz",0)) == 30.0
            and m.get("ego_pitch_clamp") == 20.0 and m.get("ego_gate_z_bias") == 0.3
            and str(m.get("ego_aim_offsets","")).strip() == "4:0,3;5:0,3"
            and float(m.get("ego_aim_release",12.0) or 12.0) == 12.0
            and float(m.get("ego_aim_fade",0.0) or 0.0) == 0.0)

G = {}
for root in (_RUNS,):
    for m, recs, name in load(root):
        if not cell(m): continue
        fg = 1.0 if m.get("ego_fix_gain") is None else float(m["ego_fix_gain"])
        dg = bool(m.get("ego_det_geometric", False))
        if abs(fg-1.0) < 1e-6 and dg:      k = "B: det_geo ON  (fg 1.0)"
        elif abs(fg-1.0) < 1e-6:           k = "CONTROL: det_geo off (fg 1.0)"
        elif abs(fg-0.15) < 1e-6:          k = "A: fix_gain 0.15 (det off)"
        else: continue
        t = np.array([r["sim_time_ns"] for r in recs], float)/1e9
        ach = (len(recs)-1)/max(t[-1]-t[0], 1e-9)
        fresh = 100*np.mean([bool(r.get("pose_seen")) for r in recs])
        # did the mask fire? slot0 zeroed at close range
        near_tot = near_zero = 0
        has_key = any("det_geom" in r for r in recs[:400])
        for r in recs:
            rel = r.get("rel_flu"); o = r.get("obs")
            if not rel or not o or len(o) < 16: continue
            rng = float(np.linalg.norm(rel))
            if rng < 2.0:
                near_tot += 1
                if float(np.abs(np.asarray(o[11:14], float)).sum()) <= 1e-6: near_zero += 1
        G.setdefault(k, []).append(dict(mg=max(r.get("gate_index",-1) for r in recs),
            ach=ach, fresh=fresh, nz=(near_zero, near_tot), key=has_key, name=name))

print("(1) DID THE MASK FIRE?  slot0 zeroed at range < 2.0 m")
for k in ("B: det_geo ON  (fg 1.0)", "CONTROL: det_geo off (fg 1.0)", "A: fix_gain 0.15 (det off)"):
    v = G.get(k, [])
    if not v: continue
    z = sum(x["nz"][0] for x in v); t = sum(x["nz"][1] for x in v)
    print("   %-30s det_geom key logged: %-5s   zeroed %5d / %5d ticks = %5.1f%%"
          % (k, any(x["key"] for x in v), z, t, 100*z/max(t,1)))
print()
print("(2) WHAT DID IT COST?")
print("   %-30s %8s %8s" % ("", "ach Hz", "fresh %"))
for k, v in G.items():
    print("   %-30s %8.2f %8.1f" % (k, np.median([x["ach"] for x in v]), np.median([x["fresh"] for x in v])))
print()
print("(3) OUTCOME")
for k, v in G.items():
    a = np.array([x["mg"] for x in v], float)
    print("   %-30s n=%3d  mean %.3f  median %.1f  max %d  >=4: %3.0f%%"
          % (k, len(a), a.mean(), np.median(a), int(a.max()), 100*(a>=4).mean()))
