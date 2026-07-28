"""DID THE LATERAL TRIM MOVE THE DRONE, OR ONLY THE NUMBER?

The logged `rel_flu` is the lever the POLICY WAS FED -- offset INCLUDED (ego_obs.py). The
injection is  rel_flu += [0, -aim_lat, +aim_vert],  so the TRUE gate lever is recovered by

    rel_true = rel_logged + [0, +aim_lat, -aim_vert]

Two distinguishable outcomes, and they are the whole experiment:

  * PERCEIVED lateral (vs the offset gate, i.e. straight off the log) stays at its old value
    ~ +0.19 m  AND  TRUE lateral moves to ~ +0.19 - 0.25 = -0.06
        => the drone faithfully chased the shifted target: THE TRIM WORKS, the bias is actuatable.
  * TRUE lateral stays at ~ +0.19 and PERCEIVED goes to ~ +0.44
        => the drone ends up where it always did regardless of where we told it the gate was:
           THE OFFSET IS NOT WHAT SETS THE FINAL LATERAL POSITION.
"""
import json, glob, os, math, sys
import numpy as np
from math import comb

ZBIAS_COS = math.cos(math.radians(20.0))
RNG_MAX = 5.0


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def Rlevel(roll, pitch):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def signtest(k, n):
    return min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) * 2 / 2 ** n)


def collect(roots, deinject):
    """Returns (perceived_lat, true_lat) at the closest fresh fix before each PASS."""
    per, tru = [], []
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            recs = load(os.path.join(d, "ego_obs.jsonl"))
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            zb = float(meta.get("ego_gate_z_bias", 0.0))
            adv = {}
            for i in range(1, len(recs)):
                g0, g1 = recs[i - 1].get("gate_index"), recs[i].get("gate_index")
                if g0 is not None and g1 == g0 + 1:
                    adv[g0] = recs[i - 1]["sim_time_ns"]
            for g, ta in adv.items():
                c = [r for r in recs if r.get("gate_index") == g and r.get("pose_seen")
                     and r.get("rel_flu") and len(r.get("obs", [])) == 21
                     and r["sim_time_ns"] < ta
                     and float(np.linalg.norm(r["rel_flu"])) <= RNG_MAX]
                if not c:
                    continue
                r = c[-1]
                o = r["obs"]
                R = Rlevel(-o[3], -o[4])
                rel = np.array(r["rel_flu"], dtype=float)
                q = R @ rel
                q[2] += zb * ZBIAS_COS
                per.append(-q[1])
                a = r.get("aim_off")
                rel_t = rel if (a is None or not deinject) else rel + np.array([0.0, a[0], -a[1]])
                qt = R @ rel_t
                qt[2] += zb * ZBIAS_COS
                tru.append(-qt[1])
    return np.array(per), np.array(tru)


def show(tag, v):
    if len(v) < 3:
        print("%-38s n=%3d  (too few)" % (tag, len(v))); return
    se = v.std(ddof=1) / math.sqrt(len(v))
    L, R_ = int((v > 0).sum()), int((v < 0).sum())
    print("%-38s n=%3d  mean %+0.3f  median %+0.3f  t=%+5.2f  L/R %d/%d  sign p=%.4f"
          % (tag, len(v), v.mean(), np.median(v), v.mean() / se, L, R_, signtest(min(L, R_), L + R_)))


base_p, _ = collect(["rate30/data/runs", "set3/data/runs"], True)
trim_p, trim_t = collect(["trim/data/runs"], True)

print("BASELINE (rate-30, no lateral trim; offset was vertical-only and released at 12 m)")
show("  lateral at the pass", base_p)
print("\nTRIM COHORT (lateral +0.25 on every gate, release 0.0 => armed through the gate)")
show("  PERCEIVED (vs the OFFSET gate)", trim_p)
show("  TRUE      (offset de-injected)", trim_t)
print("\n  prediction if the trim WORKS : perceived ~ +0.19,  true ~ -0.06")
print("  prediction if it does NOT    : perceived ~ +0.44,  true ~ +0.19")
print("  observed shift in TRUE lateral vs baseline: %+0.3f m  (a working trim moves it -0.25)"
      % (trim_t.mean() - base_p.mean()))
