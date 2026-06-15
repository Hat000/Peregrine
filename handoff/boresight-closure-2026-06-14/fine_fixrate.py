"""Pin the MIN fix-rate that closes r=0.30 post-bake at the measured operating bias, with a fine
fix-rate grid + bootstrap 90% CI (CI-honest: upper CI of BOTH p90 and p99 below MARGIN(0.30)=0.235).
ANISO sigma [lat 0.1914, vert 0.1008]. Both speeds, both bias modes. [BORESIGHT-CLOSURE 2026-06-14]"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[0] / "margin-closure-envelope-2026-06-14"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import margin_envelope as ME           # noqa: E402
import margin_rerun as R               # noqa: E402  (reuses cell_misses, boot_ci, sigmas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmc", type=int, default=6000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=11)
    args = ap.parse_args()
    sl, sv = R.SIG_LAT, R.SIG_VERT
    M30 = ME.margin_at(0.30)
    M38 = ME.margin_at(0.38)
    frs = [0.22, 0.26, 0.30, 0.33, 0.38, 0.42]
    out = dict(margin_r030=M30, margin_r038=M38, sigma=[sl, sv], nmc=args.nmc, nboot=args.nboot, cells=[])
    print(f"[fine] aniso[{sl},{sv}] r0.30 margin={M30:.3f}  nmc={args.nmc} boot={args.nboot}")
    t0 = time.time()
    with mp.Pool(args.procs) as pool:
        for v in (30.0, 37.0):
            for bm in ("random3d", "inplane"):
                row = []
                for fr in frs:
                    m = R.cell_misses(pool, v, sl, sv, 0.0, bm, fr, 0.0, args.nmc)
                    p90, p99 = float(np.percentile(m, 90)), float(np.percentile(m, 99))
                    p90lo, p90hi = R.boot_ci(m, 90, args.nboot)
                    p99lo, p99hi = R.boot_ci(m, 99, args.nboot)
                    c30 = "CLOSE" if (p90hi < M30 and p99hi < M30) else (
                        "KNIFE_EDGE" if (p99lo < M30 <= p99hi) else "NO_CLOSE")
                    c38 = "CLOSE" if (p90hi < M38 and p99hi < M38) else (
                        "KNIFE_EDGE" if (p99lo < M38 <= p99hi) else "NO_CLOSE")
                    rec = dict(v=v, bias_mode=bm, fix_rate=fr, p50=float(np.percentile(m, 50)),
                               p90=p90, p99=p99, p90_ci=[p90lo, p90hi], p99_ci=[p99lo, p99hi],
                               verdict_r030=c30, verdict_r038=c38)
                    out["cells"].append(rec)
                    row.append((fr, p99, c30))
                    print(f"  v{v:.0f} {bm:8s} fr{fr:.2f}: p90 {p90:.3f}[{p90lo:.3f},{p90hi:.3f}] "
                          f"p99 {p99:.3f}[{p99lo:.3f},{p99hi:.3f}] r30 {c30:10s} r38 {c38}", flush=True)
                # min fr that CLOSES r0.30 (CI-honest) for this (v,bm)
                closer = next((fr for fr, p99, c in row if c == "CLOSE"), None)
                print(f"  -> v{v:.0f} {bm:8s} MIN fr closing r0.30 = {closer}", flush=True)
    out["wall_s"] = round(time.time() - t0, 1)
    (_HERE / "fine_fixrate_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[fine] done {out['wall_s']}s -> fine_fixrate_results.json")


if __name__ == "__main__":
    main()
