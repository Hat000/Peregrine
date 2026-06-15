"""Finish the verify_clustering sweep: run the 32 bursty cells (the main sweep was stopped
under external machine contention before reaching them), MERGE into verify_clustering_results.json,
and finalize meta (n_ok). gap_s (80, load-bearing) is already complete in the JSON; gap_m is
partially covered. This appends the bursty schedule cells so the artifact carries all three lenses.
[BORESIGHT-CLOSURE ADVERSARIAL VERIFY 2026-06-14]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "margin-closure-envelope-2026-06-14"))

import verify_clustering_drone as VD  # noqa: E402

SIGMA_LAT = 0.1914
SIGMA_VERT = 0.1008
VS = [30.0, 37.0]
FRS = [0.35, 0.50]
BIASES = [0.0, 0.6]
BMODES = ["inplane", "random3d"]
BURSTS = [3, 5]
RES = _HERE / "verify_clustering_results.json"


def _worker(spec):
    v, fr, b, bm, bn = spec
    res = VD.run_cell_drought(v, SIGMA_LAT, SIGMA_VERT, b, bm, fr, "cold", latency_ms=15.0,
                              vert_fix_bias_m=0.0, schedule="bursty", burst_n=bn,
                              terminal_gap_s=0.0, terminal_gap_m=0.0, n_mc=2000)
    res["kind"] = "bursty"
    return res


def main():
    d = json.loads(RES.read_text())
    have_bursty = any(c.get("kind") == "bursty" for c in d["cells"])
    if have_bursty:
        print("bursty cells already present; nothing to do.")
        return
    specs = [(v, fr, b, bm, bn) for v in VS for fr in FRS for b in BIASES
             for bm in BMODES for bn in BURSTS]
    print(f"running {len(specs)} bursty cells (nmc=2000)...", flush=True)
    t0 = time.time()
    with mp.Pool(processes=8) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, specs, chunksize=1), 1):
            d["cells"].append(res)
            print(f"[{i}/{len(specs)}] bursty v={res['v_race']} fr={res['fix_rate']} "
                  f"bias={res['att_bias_deg']} {res['bias_mode']} bn={res['burst_n']} -> "
                  f"p99 {res['inplane_p99']:.3f} CIhi {res['p99_ci90'][1]:.3f} "
                  f"r0.30={res['clears_ci']['0.30']}", flush=True)
            RES.write_text(json.dumps(d, indent=2))  # checkpoint
    from collections import Counter
    kinds = Counter(c["kind"] for c in d["cells"])
    d["meta"]["n_ok"] = len(d["cells"])
    d["meta"]["kinds"] = dict(kinds)
    d["meta"]["note"] = ("gap_s 80/80 COMPLETE (load-bearing terminal-drought-in-seconds); "
                         "bursty 32/32 COMPLETE; gap_m partial (v=30 full, v=37 partial) -- "
                         "main sweep stopped under external machine contention. Verdict rests on "
                         "the complete gap_s set.")
    RES.write_text(json.dumps(d, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s. total cells {len(d['cells'])} {dict(kinds)}", flush=True)


if __name__ == "__main__":
    main()
