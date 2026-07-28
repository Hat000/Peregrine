"""ADJUDICATE fix_gain 0.15 against a MATCHED control. Single-cell: v19Ws0 / rate 30 / clamp 20 /
z_bias 0.30 / dodge 4:0,3;5:0,3 armed / fade 0 / release 12. The ONLY difference is fix_gain."""
import os as _os
# Corpus root. Defaults to the in-repo flight corpus; override for an external snapshot:
#   PEREGRINE_RUNS=/path/to/runs python3 <this script>
_RUNS = _os.environ.get("PEREGRINE_RUNS", "data/runs")

import json, glob, os
import numpy as np

def load(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        mp, op = os.path.join(d, "meta.json"), os.path.join(d, "ego_obs.jsonl")
        if not (os.path.exists(mp) and os.path.exists(op)): continue
        try:
            m = json.load(open(mp, encoding="utf-8"))
            recs = [json.loads(l) for l in open(op, encoding="utf-8") if l.strip()]
        except Exception: continue
        if len(recs) < 5: continue
        out.append((m, recs, os.path.basename(d)))
    return out

def cell(m):
    return (str(m.get("ego_ckpt","")).split("/")[-1] == "v19Ws0_actor.pth"
            and float(m.get("rate_hz",0)) == 30.0
            and m.get("ego_pitch_clamp") == 20.0
            and m.get("ego_gate_z_bias") == 0.3
            and str(m.get("ego_aim_offsets","")).strip() == "4:0,3;5:0,3"
            and float(m.get("ego_aim_release", 12.0) or 12.0) == 12.0
            and float(m.get("ego_aim_fade", 0.0) or 0.0) == 0.0)

arms = {"fix_gain 0.15 (NEW)": [], "fix_gain 1.0 / absent (CONTROL)": []}
for root in (_RUNS,):
    for m, recs, name in load(root):
        if not cell(m): continue
        fg = m.get("ego_fix_gain", None)
        fg = 1.0 if fg is None else float(fg)
        mg = max([r.get("gate_index", -1) for r in recs])
        key = "fix_gain 0.15 (NEW)" if abs(fg - 0.15) < 1e-6 else ("fix_gain 1.0 / absent (CONTROL)" if abs(fg - 1.0) < 1e-6 else None)
        if key: arms[key].append(mg)

for k, v in arms.items():
    if not v: print("%-32s n=0" % k); continue
    a = np.array(v, dtype=float)
    print("%-32s n=%3d  mean max gate %.3f  median %.1f  max %d  | dist %s"
          % (k, len(a), a.mean(), np.median(a), int(a.max()),
             {int(g): int((a == g).sum()) for g in sorted(set(a.tolist()))}))

A = np.array(arms["fix_gain 0.15 (NEW)"], dtype=float)
B = np.array(arms["fix_gain 1.0 / absent (CONTROL)"], dtype=float)
if len(A) and len(B):
    # Mann-Whitney U, normal approx with tie correction
    x = np.concatenate([A, B]); r = x.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(x); s = x[order]; rr = np.empty(len(x))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j+1] == s[i]: j += 1
        rr[order[i:j+1]] = (i + j) / 2.0 + 1
        i = j + 1
    R1 = rr[:len(A)].sum(); n1, n2 = len(A), len(B)
    U = R1 - n1*(n1+1)/2.0
    mu = n1*n2/2.0
    _, cnt = np.unique(x, return_counts=True)
    tie = (cnt**3 - cnt).sum()
    sd = np.sqrt(n1*n2/12.0 * ((n1+n2+1) - tie/((n1+n2)*(n1+n2-1))))
    z = (U - mu)/sd
    from math import erf, sqrt
    p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    print()
    print("  DELTA = %+.3f gates   (Mann-Whitney z=%.2f, p=%.4f)" % (A.mean()-B.mean(), z, p))
    print("  reached gate >=2:  0.15 -> %.0f%%   control -> %.0f%%" % (100*(A>=2).mean(), 100*(B>=2).mean()))
    print("  reached gate >=4:  0.15 -> %.0f%%   control -> %.0f%%" % (100*(A>=4).mean(), 100*(B>=4).mean()))
