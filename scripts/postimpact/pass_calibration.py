"""ARE THE LATERAL STRIKES DIRECTIONAL?  All five classified lateral kills across both cohorts
put the drone LEFT of the gate.  Under a fair-coin null that is p = 2^-5 = 0.031 -- suggestive
enough to test against the one calibration surface we can actually measure: CONFIRMED PASSES.

If the whole loop carries a left bias, the passes are off-centre the same way and the strikes are
just the tail of one distribution.  If the passes are centred, the strikes are a separate mode.

PASS DETECTION: a gate_index advance i -> i+1 means gate i was passed (racer.race_outcome).
CLOSEST-APPROACH LATERAL: the levelled lateral at the last FRESH fix strictly before the advance
and inside RNG_MAX, so the advance seam can never contribute.
"""
import json, glob, os, math, sys
import numpy as np

ZBIAS_COS = math.cos(math.radians(20.0))
RNG_MAX = 5.0


def load(p):
    out = []
    if os.path.exists(p):
        for ln in open(p, "r", encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def Rlevel(roll, pitch):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


rows = []
for root in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        recs = load(os.path.join(d, "ego_obs.jsonl"))
        if not recs:
            continue
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        zb = float(meta.get("ego_gate_z_bias", 0.0))
        # advances
        adv = {}
        for i in range(1, len(recs)):
            g0, g1 = recs[i - 1].get("gate_index"), recs[i].get("gate_index")
            if g0 is not None and g1 == g0 + 1:
                adv[g0] = recs[i - 1]["sim_time_ns"]
        for g, tadv in sorted(adv.items()):
            cand = [r for r in recs
                    if r.get("gate_index") == g and r.get("pose_seen") and r.get("rel_flu")
                    and r.get("obs") and len(r["obs"]) == 21
                    and r["sim_time_ns"] < tadv
                    and float(np.linalg.norm(r["rel_flu"])) <= RNG_MAX
                    and r.get("aim_off") is None]
            if not cand:
                continue
            r = cand[-1]
            o = r["obs"]
            q = Rlevel(-o[3], -o[4]) @ np.array(r["rel_flu"], dtype=float)
            q[2] += zb * ZBIAS_COS
            p = -q
            rows.append((os.path.basename(d)[9:15], g, float(np.linalg.norm(r["rel_flu"])),
                         p[1], p[2]))

lat = np.array([r[3] for r in rows])
vert = np.array([r[4] for r in rows])
print("CONFIRMED PASSES, closest fresh fix inside %.1f m, aim-offset ticks excluded: n=%d"
      % (RNG_MAX, len(rows)))
print("  lateral (+ = drone LEFT of gate) : mean %+0.3f  median %+0.3f  sd %.3f  "
      "| left %d / right %d" % (lat.mean(), np.median(lat), lat.std(),
                                int((lat > 0).sum()), int((lat < 0).sum())))
print("  vertical(+ = drone ABOVE gate)   : mean %+0.3f  median %+0.3f  sd %.3f"
      % (vert.mean(), np.median(vert), vert.std()))
se = lat.std(ddof=1) / math.sqrt(len(lat))
print("  lateral mean / SE = %+0.2f  (|t| > 2 would be a real offset)" % (lat.mean() / se))

print("\n  by gate index:")
for g in sorted(set(r[1] for r in rows)):
    v = np.array([r[3] for r in rows if r[1] == g])
    print("    gate %d  n=%2d  lat mean %+0.3f  median %+0.3f  [min %+0.2f, max %+0.2f]"
          % (g, len(v), v.mean(), np.median(v), v.min(), v.max()))

print("\n  the five classified LATERAL KILLS for comparison: +1.07 +1.10 +1.28 +1.57 +0.92")
print("  pass |lat| p50 %.2f  p90 %.2f  max %.2f" %
      (np.percentile(abs(lat), 50), np.percentile(abs(lat), 90), abs(lat).max()))
