"""SEAM-FREE approach speed into gate 0.

The previous window straddled the RACE_STATUS advance, so on a PASS the seeker
dropped gate 0 and rel_flu jumped to the NEXT gate -- giving passes a spurious
NEGATIVE closing speed (p10 -28.8 m/s) and manufacturing separation. That is the
pass_drop teleport the profile documents, and it would have produced a fake result.

FIX: measure the traverse speed over a FIXED RANGE BAND (6.0 m -> 3.0 m) on the
approach to gate 0. Both bands lie strictly BEFORE the plane, so neither group can
touch the seam, and both groups are measured over the identical geometry.
Windows containing a lever jump > 1.5 m between consecutive ticks are rejected.
"""
import json, os, math
import numpy as np

RUNS = r"C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs"
DT = 1.0 / 30.0
R_HI, R_LO = 6.0, 3.0


def lineage(label):
    l = (label or "").lower()
    for tag in ("v20", "v19", "v18", "v16", "v15", "vtracka", "vpef", "v1"):
        if tag in l:
            return tag
    return "other"


rows = []
for sess in sorted(os.listdir(RUNS)):
    d = os.path.join(RUNS, sess)
    mp, lp = os.path.join(d, "meta.json"), os.path.join(d, "ego_obs.jsonl")
    if not (os.path.exists(mp) and os.path.exists(lp)):
        continue
    try:
        meta = json.load(open(mp))
    except Exception:
        continue
    ticks = []
    with open(lp) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ticks.append(json.loads(line))
                except Exception:
                    pass
    if len(ticks) < 8:
        continue
    maxgate = max([t.get("gate_index", 0) or 0 for t in ticks] + [meta.get("gate_index", 0) or 0])

    # index where gate 0 is decided
    idx_end, outcome = None, None
    for i in range(1, len(ticks)):
        gi, gp = ticks[i].get("gate_index", 0), ticks[i - 1].get("gate_index", 0)
        if gi is not None and gp is not None and gp == 0 and gi == 1:
            idx_end, outcome = i - 1, "pass"
            break
    if idx_end is None and meta.get("final_state") == "CRASH" and maxgate == 0:
        idx_end, outcome = len(ticks) - 1, "die"
    if outcome is None:
        continue

    # range series STRICTLY on the gate-0 approach (indices <= idx_end)
    seq = [(i, float(np.linalg.norm(t["rel_flu"])))
           for i, t in enumerate(ticks[:idx_end + 1]) if t.get("rel_flu")]
    if len(seq) < 10:
        continue
    # LAST downward crossing of R_HI, then the LAST downward crossing of R_LO after it
    i_hi = None
    for k in range(len(seq) - 1, 0, -1):
        if seq[k][1] <= R_HI < seq[k - 1][1]:
            i_hi = k
            break
    if i_hi is None:
        continue
    i_lo = None
    for k in range(i_hi, len(seq)):
        if seq[k][1] <= R_LO:
            i_lo = k
            break
    if i_lo is None or i_lo <= i_hi:
        continue
    span = seq[i_hi:i_lo + 1]
    jumps = [abs(span[j + 1][1] - span[j][1]) for j in range(len(span) - 1)]
    if jumps and max(jumps) > 1.5:
        continue                                   # seam / teleport contamination
    dt = (span[-1][0] - span[0][0]) * DT
    if dt <= 1e-3:
        continue
    spd = (span[0][1] - span[-1][1]) / dt
    rows.append(dict(sess=sess, lin=lineage(meta.get("label")), outcome=outcome, spd=spd))

print(f"flights with a clean 6->3 m traverse of the gate-0 approach: {len(rows)}")
P = np.array([r["spd"] for r in rows if r["outcome"] == "pass"])
D = np.array([r["spd"] for r in rows if r["outcome"] == "die"])


def q(a, name):
    if a.size == 0:
        return f"{name}: n=0"
    return (f"{name}: n={a.size:4d}  p10={np.percentile(a,10):5.2f} p50={np.percentile(a,50):5.2f} "
            f"p90={np.percentile(a,90):5.2f} p99={np.percentile(a,99):5.2f} max={a.max():5.2f} m/s")


print(q(P, "  PASSED gate 0 "))
print(q(D, "  DIED at gate 0"))
if P.size and D.size:
    allv = np.concatenate([P, D])
    ranks = allv.argsort().argsort().astype(float)
    auc = (ranks[:P.size].sum() - P.size * (P.size - 1) / 2) / (P.size * D.size)
    auc = max(auc, 1 - auc)
    print(f"\n  median: died {np.median(D):.2f} vs passed {np.median(P):.2f} "
          f"({np.median(D)-np.median(P):+.2f} m/s)")
    print(f"  AUC (speed separates die-from-pass) = {auc:.3f}   [0.5 = nothing; "
          f"profile's best resolved modes reach 0.72-0.73]")
    print("\n  TAIL VIEW - fraction of each group above a speed threshold:")
    for thr in (5, 6, 7, 8, 9, 10):
        print(f"    > {thr:2d} m/s :  passed {100*(P>thr).mean():5.1f}%   died {100*(D>thr).mean():5.1f}%")
    print("\n  and the converse - conditional death rate BY speed band:")
    for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 10), (10, 99)):
        p = ((P > lo) & (P <= hi)).sum()
        dd = ((D > lo) & (D <= hi)).sum()
        n = p + dd
        if n >= 8:
            print(f"    {lo:2d}-{hi:2d} m/s : n={n:4d}   died {dd:4d}  = {100*dd/n:5.1f}%")

print("\n  per-lineage:")
print(f"  {'lin':10s} {'nPass':>6s} {'nDie':>5s} {'pass p50':>9s} {'die p50':>8s} {'gap':>7s}")
for L in sorted({r["lin"] for r in rows}):
    p = np.array([r["spd"] for r in rows if r["lin"] == L and r["outcome"] == "pass"])
    dd = np.array([r["spd"] for r in rows if r["lin"] == L and r["outcome"] == "die"])
    if p.size + dd.size < 10:
        continue
    mp_ = np.median(p) if p.size else float('nan')
    md = np.median(dd) if dd.size else float('nan')
    print(f"  {L:10s} {p.size:6d} {dd.size:5d} {mp_:9.2f} {md:8.2f} {md-mp_:7.2f}")
