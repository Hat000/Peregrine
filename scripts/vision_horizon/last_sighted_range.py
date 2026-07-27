"""SUSPECT MY OWN INSTRUMENT. "0% fresh fixes inside 1 m" is now load-bearing for the whole
campaign, so name the artifact that could fabricate it and stratify by that artifact's own size.

THE ARTIFACT: I binned per-TICK by norm(rel_flu). But when pose_seen is False, rel_flu is the
PROPAGATED belief, not a measurement. A long coast marches the BELIEVED range down through the
low bins on its own, so the low-range bins are filled preferentially with exactly the not-seen
ticks. The statistic "fraction of ticks at range r that are fresh" is therefore partly circular:
range is a function of freshness. And n was only 19 ticks in the 0-1 m bin.

NON-CIRCULAR INSTRUMENT: work per APPROACH, not per tick, and use only ranges measured AT a fresh
fix (a fix is a measurement; the coast is not).
  * LAST-SIGHTED RANGE per approach = the range at the final fresh fix before the gate. It is
    read off a measurement, never off a coast, so the propagation cannot manufacture it.
  * Restrict to CONFIRMED PASSES, where the drone provably reached the gate plane, so "the log
    ended early" cannot masquerade as "vision stopped".
If vision truly dies at ~1.5 m, the last-sighted range should pile up around 1.5 m on approaches
that provably continued to 0 m. If instead it is spread wide, the per-tick curve was an artifact.
"""
import json, glob, os, math, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "CORPUS/data/runs"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def rng_of(r):
    rel = r.get("rel_flu")
    if not rel:
        return None
    a = r.get("aim_off")
    v = np.array(rel, dtype=float)
    if a is not None:
        v = v + np.array([0.0, a[0], -a[1]])
    return float(np.linalg.norm(v))


last_sighted, min_seen_any, closed_to = [], [], []
per_tick_fresh = []          # the ORIGINAL circular statistic, for side-by-side
n_runs = 0
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    p = os.path.join(d, "ego_obs.jsonl")
    if not os.path.exists(p):
        continue
    try:
        recs = load(p)
    except Exception:
        continue
    if not recs:
        continue
    n_runs += 1
    adv = {}
    for i in range(1, len(recs)):
        g0, g1 = recs[i - 1].get("gate_index"), recs[i].get("gate_index")
        if g0 is not None and g1 == g0 + 1:
            adv[g0] = recs[i - 1]["sim_time_ns"]
    for r in recs:
        rr = rng_of(r)
        if rr is not None and rr <= 12:
            per_tick_fresh.append((rr, 1.0 if r.get("pose_seen") else 0.0))
    for g, ta in adv.items():
        leg = [r for r in recs if r.get("gate_index") == g and r["sim_time_ns"] < ta]
        fresh = [rng_of(r) for r in leg if r.get("pose_seen") and rng_of(r) is not None]
        allr = [rng_of(r) for r in leg if rng_of(r) is not None]
        if not fresh or not allr:
            continue
        last_sighted.append(min(fresh))          # closest range at which a MEASUREMENT existed
        closed_to.append(min(allr))              # closest believed range reached on the leg

ls = np.array(last_sighted)
cl = np.array(closed_to)
pt = np.array(per_tick_fresh)
print("runs read: %d   CONFIRMED PASSES with >=1 fresh fix on the leg: %d" % (n_runs, len(ls)))
print()
print("A. NON-CIRCULAR -- LAST-SIGHTED RANGE per confirmed pass (range read AT a fix, never a coast)")
print("   p10 %.2f  p25 %.2f  MEDIAN %.2f  p75 %.2f  p90 %.2f  min %.2f  max %.2f m"
      % tuple(np.percentile(ls, [10, 25, 50, 75, 90, 0, 100])))
for thr in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    print("   fraction of passes whose LAST fix was closer than %.1f m: %5.1f%%"
          % (thr, 100 * np.mean(ls < thr)))
print()
print("B. how close the BELIEF closed on the same legs (this one IS coast-driven, for contrast)")
print("   p10 %.2f  MEDIAN %.2f  p90 %.2f" % tuple(np.percentile(cl, [10, 50, 90])))
print("   median BLIND DISTANCE (last-sighted minus closest-believed) = %.2f m"
      % float(np.median(ls - cl)))
print()
print("C. the ORIGINAL per-tick statistic, reproduced for side-by-side (CIRCULAR -- range depends")
print("   on freshness through the coast). If A and C disagree, C was the artifact.")
for lo, hi in [(0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 6), (6, 10)]:
    m = (pt[:, 0] >= lo) & (pt[:, 0] < hi)
    if m.sum() < 5:
        continue
    print("   %4g-%-4g m  n=%5d  fresh %5.1f%%" % (lo, hi, m.sum(), 100 * pt[m, 1].mean()))
