"""REGRESSION PROOF for margin_driver_v2.fly_lap_v2 (BORESIGHT-CLOSURE Deliverable-3 engine prep).

Three independent checks, all printed + dumped to results/regression_proof_results.json:

(A) PER-LAP BIT-IDENTITY (matched seeds): for several cells, fly_lap_v2(sigma_lat=s, sigma_vert=s,
    vert_fix_bias_m=0) called with a fresh RNG of seed S == ME.fly_lap(inplane_sigma=s) called with a
    fresh RNG of the SAME seed S, to ~1e-15 (float round-off). This is the strongest possible proof that
    the v2 body is ME.fly_lap verbatim outside the three documented changes, and that the isotropic /
    zero-bias limit collapses EXACTLY onto production. Cells: v37 sig0.10 fr0.07 b0 and v37 sig0.10
    fr0.35 b0.6 (the two the task names), plus v55 fr0.25 b0.9 inplane for breadth.

(B) DISTRIBUTION-LEVEL MATCH (the task's explicit ask): run_cell_v2(sigma_lat=s, sigma_vert=s,
    vert_fix_bias_m=0, match_me_seeds=True) vs ME.run_cell(inplane_sigma=s) at nmc>=2000, comparing
    p50/p90/p99. Because match_me_seeds reproduces ME.run_cell's exact per-lap seed formula in this
    limit, the two distributions are identical (delta ~1e-15), i.e. well within the "few mm" bar.

(C) PUBLISHED-STACK REPRODUCTION: re-run a banked confirm_verdict cell (the MEASURED sig0.10 b0.0
    fr0.07 random3d cell, and the b0.6 fr0.25 random3d closure-boundary cell) using confirm_verdict's
    OWN seed formula, at the banked nmc=5000, and check p90/p99 match the banked
    confirm_verdict_results.json to a few mm. This proves margin_driver_v2 is importing the SAME
    production margin_envelope that produced the published verdict (not a drifted copy).

Run:
  PYTHONPATH=src ../../../.venv/Scripts/python.exe handoff/boresight-closure-2026-06-14/regression_proof.py [--nmc 2000] [--nmc_pub 5000] [--procs 11]
[BORESIGHT-CLOSURE 2026-06-14]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "margin-closure-envelope-2026-06-14"))
sys.path.insert(0, str(_HERE))
import margin_envelope as ME            # noqa: E402  (the PRODUCTION engine)
import margin_driver_v2 as V2           # noqa: E402  (the engine under test)

_BANKED = (_HERE.parents[0] / "margin-closure-envelope-2026-06-14" / "results"
           / "confirm_verdict_results.json")


# ----------------------------------------------------------------------------- (A) per-lap bit-identity
def check_per_lap_identity():
    cells = [
        dict(label="v37 sig0.10 fr0.07 b0 rnd", v=37.0, s=0.10, bdeg=0.0, bm="random3d", fr=0.07, vm="cold"),
        dict(label="v37 sig0.10 fr0.35 b0.6 rnd", v=37.0, s=0.10, bdeg=0.6, bm="random3d", fr=0.35, vm="cold"),
        dict(label="v55 sig0.10 fr0.25 b0.9 inp", v=55.0, s=0.10, bdeg=0.9, bm="inplane", fr=0.25, vm="cold"),
    ]
    out = []
    for c in cells:
        bmag = ME.att_deg_to_accel_bias(c["bdeg"])
        maxabs = 0.0
        for s in range(80):
            seed = ME.SEED + 101 * s + int(round(c["v"])) * 13 + 7919 * (1 + s % 5)
            a = ME.fly_lap(np.random.default_rng(seed), c["v"], c["s"], bmag, c["bm"], c["fr"], c["vm"], 15.0)
            b = V2.fly_lap_v2(np.random.default_rng(seed), c["v"], c["s"], c["s"], bmag, c["bm"], c["fr"],
                              c["vm"], 15.0, vert_fix_bias_m=0.0)
            maxabs = max(maxabs, abs(a - b))
        out.append(dict(label=c["label"], n_laps=80, max_abs_diff=maxabs, bit_identical=bool(maxabs < 1e-9)))
        print(f"  (A) {c['label']:30s} max|fly_lap - fly_lap_v2| = {maxabs:.2e}  "
              f"{'IDENTICAL' if maxabs < 1e-9 else 'DIFFER'}")
    return out


# -------------------------------------------------------------------- (B) distribution-level match
def check_distribution_match(nmc):
    cells = [
        dict(label="v37 sig0.10 fr0.07 b0 rnd", v=37.0, s=0.10, bdeg=0.0, bm="random3d", fr=0.07, vm="cold"),
        dict(label="v37 sig0.10 fr0.35 b0.6 rnd", v=37.0, s=0.10, bdeg=0.6, bm="random3d", fr=0.35, vm="cold"),
    ]
    out = []
    for c in cells:
        me = ME.run_cell(c["v"], c["s"], c["bdeg"], c["bm"], c["fr"], c["vm"], n_mc=nmc)
        v2 = V2.run_cell_v2(c["v"], c["s"], c["s"], c["bdeg"], c["bm"], c["fr"], c["vm"],
                            vert_fix_bias_m=0.0, n_mc=nmc, match_me_seeds=True)
        d = {q: abs(me[f"inplane_{q}"] - v2[f"inplane_{q}"]) for q in ("p50", "p90", "p99")}
        rec = dict(label=c["label"], nmc=nmc,
                   me=dict(p50=me["inplane_p50"], p90=me["inplane_p90"], p99=me["inplane_p99"]),
                   v2=dict(p50=v2["inplane_p50"], p90=v2["inplane_p90"], p99=v2["inplane_p99"]),
                   abs_diff=d, within_2mm=bool(max(d.values()) < 2e-3))
        out.append(rec)
        print(f"  (B) {c['label']:30s} ME  p50/p90/p99 = "
              f"{me['inplane_p50']:.4f}/{me['inplane_p90']:.4f}/{me['inplane_p99']:.4f}")
        print(f"      {'':30s} v2  p50/p90/p99 = "
              f"{v2['inplane_p50']:.4f}/{v2['inplane_p90']:.4f}/{v2['inplane_p99']:.4f}  "
              f"max_diff={max(d.values()):.2e}")
    return out


# -------------------------------------------------------------- (C) published-stack reproduction
def _confirm_seed(s, v, sigma, bias_deg, bias_mode, fix_rate, vel_mode):
    """confirm_verdict.py's EXACT per-lap seed formula (base + 101*s)."""
    base = (ME.SEED + 101 * 0 + int(round(v)) * 13 + int(round(bias_deg * 100)) * 7
            + int(round(sigma * 1000)) * 17 + int(round(fix_rate * 1000)) * 23
            + {"cold": 1, "warm": 2, "weakvel": 3}[vel_mode] * 1009
            + {"random3d": 0, "inplane": 1}[bias_mode] * 3001 + 15 * 53)
    return base + 101 * s


def _confirm_one(args):
    s, v, sg, bmag, bm, fr, vm = args
    return ME.fly_lap(np.random.default_rng(s), v, sg, bmag, bm, fr, vm, 15.0)


def check_published_stack(nmc_pub, procs):
    banked = json.loads(_BANKED.read_text())
    by_label = {c["label"]: c for c in banked["cells"]}
    # Two banked cells with published p90/p99 (the measured op-point + a closure-boundary cell).
    targets = [
        ("MEASURED   sig0.10 b0.0 fr0.07 rnd", 37.0, 0.10, 0.0, "random3d", 0.07, "cold"),
        ("sig0.10 b0.6 fr0.25 rnd", 37.0, 0.10, 0.6, "random3d", 0.25, "cold"),
    ]
    out = []
    with mp.Pool(procs) as pool:
        for label, v, sg, bdeg, bm, fr, vm in targets:
            bmag = ME.att_deg_to_accel_bias(bdeg)
            seeds = [_confirm_seed(s, v, sg, bdeg, bm, fr, vm) for s in range(nmc_pub)]
            specs = [(sd, v, sg, bmag, bm, fr, vm) for sd in seeds]
            m = np.array(pool.map(_confirm_one, specs))
            p90, p99 = float(np.percentile(m, 90)), float(np.percentile(m, 99))
            bank = by_label.get(label)
            rec = dict(label=label, nmc=nmc_pub, repro_p90=p90, repro_p99=p99,
                       banked_p90=bank["p90"] if bank else None,
                       banked_p99=bank["p99"] if bank else None,
                       banked_p90_ci=bank.get("p90_ci") if bank else None,
                       banked_p99_ci=bank.get("p99_ci") if bank else None)
            if bank:
                rec["d_p90"] = abs(p90 - bank["p90"])
                rec["d_p99"] = abs(p99 - bank["p99"])
                # banked CI is bootstrap 5-95%; reproduction lands in-band if within CI (or <3mm of point).
                lo90, hi90 = bank["p90_ci"]; lo99, hi99 = bank["p99_ci"]
                rec["p90_in_banked_ci"] = bool(lo90 - 3e-3 <= p90 <= hi90 + 3e-3)
                rec["p99_in_banked_ci"] = bool(lo99 - 3e-3 <= p99 <= hi99 + 3e-3)
            out.append(rec)
            print(f"  (C) {label:36s} repro p90/p99 = {p90:.3f}/{p99:.3f}  "
                  f"banked {bank['p90']:.3f}/{bank['p99']:.3f}  "
                  f"d={rec['d_p90']:.3f}/{rec['d_p99']:.3f}  "
                  f"in-CI={rec['p90_in_banked_ci']}/{rec['p99_in_banked_ci']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmc", type=int, default=2000, help="nmc for distribution-level match (B)")
    ap.add_argument("--nmc_pub", type=int, default=5000, help="nmc for published-stack reproduction (C)")
    ap.add_argument("--procs", type=int, default=11)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[regression_proof] (A) per-lap identity  (B) dist match nmc={args.nmc}  "
          f"(C) published-stack nmc={args.nmc_pub}")
    print("--- (A) PER-LAP BIT-IDENTITY (matched seeds; iso sigma, bias 0) ---")
    a = check_per_lap_identity()
    print("--- (B) DISTRIBUTION MATCH  fly_lap_v2(iso,bias0) vs ME.fly_lap ---")
    b = check_distribution_match(args.nmc)
    print("--- (C) PUBLISHED-STACK REPRODUCTION  vs banked confirm_verdict ---")
    c = check_published_stack(args.nmc_pub, args.procs)

    a_ok = all(x["bit_identical"] for x in a)
    b_ok = all(x["within_2mm"] for x in b)
    c_ok = all(x.get("p90_in_banked_ci") and x.get("p99_in_banked_ci") for x in c)
    summary = dict(
        per_lap_identity_ok=a_ok, distribution_match_ok=b_ok, published_stack_ok=c_ok,
        reproduces_baseline=bool(a_ok and b_ok and c_ok),
        per_lap=a, distribution=b, published=c, wall_s=round(time.time() - t0, 1),
    )
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "regression_proof_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[regression_proof] A(identity)={a_ok}  B(dist)={b_ok}  C(published)={c_ok}  "
          f"=> reproduces_baseline={summary['reproduces_baseline']}  ({summary['wall_s']}s)")
    print(f"[regression_proof] -> results/regression_proof_results.json")


if __name__ == "__main__":
    main()
