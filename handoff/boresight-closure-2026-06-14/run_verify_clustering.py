"""Driver for the terminal-drought / clustering adversarial refutation. Sweeps the cells the main
margin analysis reported as CLOSING (v in {30,37}, fr in {0.35,0.50}, bias 0.0 and 0.6 deg, aniso
sigma [lat 0.1914, vert 0.1008], COLD, latency 15 ms, bias_mode 'inplane' AND 'random3d') across:

  (A) terminal-gap in SECONDS D in {0.0, 0.15, 0.30, 0.50, 0.75}
  (B) terminal-gap in METRES  M in {2, 4, 6, 8}   (range-to-gate-4 cutoff)
  (C) bursty (clustered) schedule at the SAME mean rate (D=0), burst_n in {3, 5}

nmc=2000 (or --nmc). Each cell reports p90/p99 + bootstrap 90% CI, CI-honest closure at r=0.30 and
0.38, and how many fixes the drought dropped. Writes verify_clustering_results.json.

CLOSURE (CI-honest) = upper-CI(p90) < MARGIN(r) AND upper-CI(p99) < MARGIN(r).
MARGIN(0.30)=0.235, MARGIN(0.38)=0.155.

Run:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    <venv-python> handoff/boresight-closure-2026-06-14/run_verify_clustering.py --nmc 2000
[BORESIGHT-CLOSURE ADVERSARIAL VERIFY 2026-06-14]
"""
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
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "margin-closure-envelope-2026-06-14"))

import verify_clustering_drone as VD  # noqa: E402
import margin_envelope as ME  # noqa: E402

SIGMA_LAT = 0.1914
SIGMA_VERT = 0.1008
VS = [30.0, 37.0]
FRS = [0.35, 0.50]
BIASES = [0.0, 0.6]
BMODES = ["inplane", "random3d"]
GAPS_S = [0.0, 0.15, 0.30, 0.50, 0.75]
GAPS_M = [2.0, 4.0, 6.0, 8.0]
BURSTS = [3, 5]


def build_specs(nmc):
    specs = []
    # (A) terminal-gap in seconds (D=0 is the regular-cadence baseline)
    for v in VS:
        for fr in FRS:
            for b in BIASES:
                for bm in BMODES:
                    for D in GAPS_S:
                        specs.append(dict(kind="gap_s", v=v, fr=fr, bias=b, bm=bm,
                                          schedule="regular", terminal_gap_s=D,
                                          terminal_gap_m=0.0, burst_n=3, n_mc=nmc))
    # (B) terminal-gap in metres
    for v in VS:
        for fr in FRS:
            for b in BIASES:
                for bm in BMODES:
                    for M in GAPS_M:
                        specs.append(dict(kind="gap_m", v=v, fr=fr, bias=b, bm=bm,
                                          schedule="regular", terminal_gap_s=0.0,
                                          terminal_gap_m=M, burst_n=3, n_mc=nmc))
    # (C) bursty schedule, no terminal drought (does burstiness ALONE matter?)
    for v in VS:
        for fr in FRS:
            for b in BIASES:
                for bm in BMODES:
                    for bn in BURSTS:
                        specs.append(dict(kind="bursty", v=v, fr=fr, bias=b, bm=bm,
                                          schedule="bursty", terminal_gap_s=0.0,
                                          terminal_gap_m=0.0, burst_n=bn, n_mc=nmc))
    return specs


def _worker(spec):
    label = (f"{spec['kind']} v={spec['v']} fr={spec['fr']} bias={spec['bias']} {spec['bm']} "
             f"D={spec['terminal_gap_s']}s M={spec['terminal_gap_m']}m sched={spec['schedule']}"
             f"(bn={spec['burst_n']})")
    try:
        t0 = time.time()
        res = VD.run_cell_drought(
            spec["v"], SIGMA_LAT, SIGMA_VERT, spec["bias"], spec["bm"], spec["fr"], "cold",
            latency_ms=15.0, vert_fix_bias_m=0.0, schedule=spec["schedule"],
            burst_n=spec["burst_n"], terminal_gap_s=spec["terminal_gap_s"],
            terminal_gap_m=spec["terminal_gap_m"], n_mc=spec["n_mc"])
        res["kind"] = spec["kind"]
        res["wall_s"] = round(time.time() - t0, 2)
        return (label, res, None)
    except Exception as e:  # noqa: BLE001
        return (label, None, {"cell": label, "error": repr(e), "trace": traceback.format_exc()})


def _checkpoint(out, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmc", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", type=str, default="verify_clustering_results.json")
    args = ap.parse_args()

    specs = build_specs(args.nmc)
    out = {
        "meta": {
            "what": "ADVERSARIAL terminal-drought + bursty-clustering refutation of gate-4 r=0.30 "
                    "closure at fr>=0.35 (anisotropic sigma, post-bake bias=0)",
            "claim": "Post-bake, gate-4 r=0.30 CLOSES once mean fix-rate reaches ~0.35-0.50 "
                     "(aniso sigma [lat 0.1914, vert 0.1008], bias<=0).",
            "sigma_lat": SIGMA_LAT, "sigma_vert": SIGMA_VERT, "vel_mode": "cold",
            "latency_ms": 15.0, "vert_fix_bias_m": 0.0, "n_mc": args.nmc,
            "margin_030": ME.margin_at(0.30), "margin_038": ME.margin_at(0.38),
            "closure_rule": "CI-honest: upper-CI(p90) < MARGIN AND upper-CI(p99) < MARGIN",
            "gaps_s": GAPS_S, "gaps_m": GAPS_M, "bursts": BURSTS,
            "n_specs": len(specs), "seed_base": ME.SEED,
        },
        "cells": [], "errors": [],
    }
    out_path = _HERE / args.out
    total = len(specs)
    print(f"[verify_clustering] {total} cells x nmc={args.nmc} procs={args.procs}", flush=True)

    done = 0
    with mp.Pool(processes=args.procs) as pool:
        for label, res, err in pool.imap_unordered(_worker, specs, chunksize=1):
            done += 1
            if res is not None:
                out["cells"].append(res)
                print(f"[{done}/{total}] {label:78s} -> p90 {res['inplane_p90']:.3f} "
                      f"p99 {res['inplane_p99']:.3f} (CIhi {res['p99_ci90'][1]:.3f}) "
                      f"drop={res['mean_dropped']:.1f} r0.30={res['clears_ci']['0.30']} "
                      f"({res['wall_s']}s)", flush=True)
            else:
                out["errors"].append(err)
                print(f"[{done}/{total}] {label:78s} -> ERROR {err['error']}", flush=True)
            _checkpoint(out, out_path)

    out["meta"]["n_ok"] = len(out["cells"])
    out["meta"]["n_err"] = len(out["errors"])
    _checkpoint(out, out_path)
    print(f"\n[verify_clustering] done {len(out['cells'])} ok / {len(out['errors'])} err -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
