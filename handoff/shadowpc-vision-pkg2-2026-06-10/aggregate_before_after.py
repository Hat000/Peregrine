"""Course-level BEFORE/AFTER acceptance table (sums over the 6 per-gate bundles).

BEFORE = before_g*.json  (pre-pkg2 production: broken in-plane map frame, attitude 1.0 deg,
                          no cov floor, no range cap mirrored)
AFTER  = after_g*.json   (map frame fixed, attitude 1.4 deg, floor 0.40 m, 32 m cap)

Each population is classified with ITS OWN chain's maha (what that chain's chi2 gate saw).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHI2 = 16.27


def load(pattern):
    per_gate, rows = {}, []
    for gi in range(6):
        d = json.loads((HERE / (pattern % gi)).read_text())
        rs = [r for r in d["rows"] if r.get("associated") and "world_fix_err_m" in r]
        per_gate[gi] = rs
        rows += rs
    return rows, per_gate


def stats(rows):
    sane = [r for r in rows if r.get("range_ok", True)]
    offered = [r for r in sane if r.get("range_cap_ok", True)]
    capped = [r for r in sane if not r.get("range_cap_ok", True)]
    fin = [r for r in offered if np.isfinite(r["maha"])]
    out = dict(solved=len(rows), sane=len(sane), capped=len(capped), offered=len(offered))
    for thr, nm in ((1.0, "g1"), (3.0, "g3")):
        good = [r for r in fin if r["world_fix_err_m"] < thr]
        rej = sum(r["maha"] > CHI2 for r in good)
        out[nm] = (rej, len(good))
    cat = [r for r in fin if r["world_fix_err_m"] >= 3.0]
    out["cat"] = (sum(r["maha"] <= CHI2 for r in cat), len(cat))
    acc = [r for r in fin if r["maha"] <= CHI2]
    accf = np.array([r["world_fix_err_m"] for r in acc])
    out["acc"] = (len(acc), float(np.percentile(accf, 50)), float(np.percentile(accf, 90)))
    return out


def show(nm, s):
    g1r, g1n = s["g1"]; g3r, g3n = s["g3"]; ckl, ckn = s["cat"]
    na, p50, p90 = s["acc"]
    print(f"{nm:7s} solved {s['solved']:3d}  offered {s['offered']:3d} (cap-dropped {s['capped']})\n"
          f"        good<1m rej {g1r:2d}/{g1n:3d} = {100*g1r/max(g1n,1):4.1f}%   "
          f"good<3m rej {g3r:2d}/{g3n:3d} = {100*g3r/max(g3n,1):4.1f}%\n"
          f"        catastrophic offered {ckn}  leaked {ckl}  -> leak {100*ckl/max(s['solved'],1):.2f}% of solved\n"
          f"        ACCEPTED fixes {na:3d}  |err| p50 {p50:.2f}  p90 {p90:.2f} m")


def main():
    before, bg = load("before_g%d.json")
    after, ag = load("after_g%d.json")
    print("== course-level acceptance table ==")
    show("BEFORE", stats(before))
    show("AFTER", stats(after))

    print("\nper-gate good<1m over-rejection (before -> after):")
    for gi in range(6):
        o = []
        for rs in (bg[gi], ag[gi]):
            fin = [r for r in rs if r.get("range_ok", True) and r.get("range_cap_ok", True)
                   and np.isfinite(r["maha"]) and r["world_fix_err_m"] < 1.0]
            rej = sum(r["maha"] > CHI2 for r in fin)
            o.append((rej, len(fin)))
        print(f"  gate {gi}:  {o[0][0]:2d}/{o[0][1]:3d}  ->  {o[1][0]:2d}/{o[1][1]:3d}")

    sb, sa = stats(before), stats(after)
    print("\n== acceptance criteria ==")
    g3b = 100 * sb["g3"][0] / max(sb["g3"][1], 1)
    g3a = 100 * sa["g3"][0] / max(sa["g3"][1], 1)
    g1b = 100 * sb["g1"][0] / max(sb["g1"][1], 1)
    g1a = 100 * sa["g1"][0] / max(sa["g1"][1], 1)
    lb = 100 * sb["cat"][0] / sb["solved"]; la = 100 * sa["cat"][0] / sa["solved"]
    print(f"  good-fix yield UP / over-rejection below 17%:  <3m {g3b:.1f}% -> {g3a:.1f}%   "
          f"<1m {g1b:.1f}% -> {g1a:.1f}%   accepted {sb['acc'][0]} -> {sa['acc'][0]}")
    print(f"  catastrophic leak <= 1.1% of solved:           {lb:.2f}% -> {la:.2f}% "
          f"({sb['cat'][0]} -> {sa['cat'][0]} fixes)")


if __name__ == "__main__":
    main()
