"""Course-level GATE TRADE-OFF aggregator for the PnP cov-inflation re-measure.

Reads the per-gate characterize_perception JSON dumps (char_g{0..5}_k{K}.json), which carry
per-frame rows with `world_fix_err_m` (|fix| world error) and `maha` (Mahalanobis vs the inflated
cov), and SUMS the GOOD-fix-rejection / CATASTROPHIC-leak counts across all 6 gate bundles -- the
course-level metric the navigator's chi2_0.999=16.27 gate would actually deliver.

Thresholds match scripts/characterize_perception.py's GATE TRADE-OFF block:
  GOOD = associated fix with |fix| < 1.0 m   (we WANT to keep these)
  CATASTROPHIC = associated fix with |fix| >= 3.0 m   (we MUST reject these)
  gate: reject iff maha > 16.27
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHI2 = 16.27
GOOD_MAX_M, CAT_MIN_M = 1.0, 3.0


def classify(rows):
    good = good_rej = cat = cat_leak = 0
    for r in rows:
        if not r.get("associated") or "world_fix_err_m" not in r:
            continue
        m = r.get("maha")
        if m is None or not math.isfinite(m):
            continue
        off = r["world_fix_err_m"]
        if off < GOOD_MAX_M:
            good += 1
            if m > CHI2:
                good_rej += 1
        elif off >= CAT_MIN_M:
            cat += 1
            if m <= CHI2:
                cat_leak += 1
    return good, good_rej, cat, cat_leak


def main():
    for K in ("1.0", "1.5", "2.0", "2.5", "3.0"):
        if not (HERE / f"char_g0_k{K}.json").exists():
            continue
        tg = tgr = tc = tcl = 0
        per_gate = []
        for G in range(6):
            f = HERE / f"char_g{G}_k{K}.json"
            if not f.exists():
                per_gate.append((G, None))
                continue
            rows = json.loads(f.read_text()).get("rows", [])
            g, gr, c, cl = classify(rows)
            per_gate.append((G, (g, gr, c, cl)))
            tg += g; tgr += gr; tc += c; tcl += cl
        print(f"\n==== K={K}  (cov_inflation={K}) ====")
        print(f"{'gate':>5} {'GOOD':>5} {'rej':>4} {'rej%':>5}   {'CAT':>4} {'leak':>4} {'leak%':>6}")
        for G, v in per_gate:
            if v is None:
                print(f"{G:>5}   (missing json)")
                continue
            g, gr, c, cl = v
            grp = 100.0 * gr / g if g else float("nan")
            clp = 100.0 * cl / c if c else float("nan")
            print(f"{G:>5} {g:>5} {gr:>4} {grp:>4.0f}%   {c:>4} {cl:>4} {clp:>5.0f}%")
        GR = 100.0 * tgr / tg if tg else float("nan")
        CL = 100.0 * tcl / tc if tc else float("nan")
        CATCH = 100.0 * (tc - tcl) / tc if tc else float("nan")
        print(f"{'ALL':>5} {tg:>5} {tgr:>4} {GR:>4.0f}%   {tc:>4} {tcl:>4} {CL:>5.0f}%   (bad-fix catch {CATCH:.0f}%)")
        print(f"  -> COURSE good-fix rejection = {tgr}/{tg} = {GR:.1f}% ;  "
              f"catastrophic leak = {tcl}/{tc} = {CL:.1f}%")


if __name__ == "__main__":
    main()
