"""ADVERSARIAL Attacks 1+2+3 on r=0.30 closure robustness to the sigma. (PARALLEL version.)

Engine = handoff/boresight-closure-2026-06-14/margin_driver_v2 (regression-proven == ME.fly_lap in the
iso/zero-bias limit; verified to machine precision in this session). COLD case-C, bias=0 (unless noted),
latency=15 ms, vert_fix_bias_m=0 (POST-bake).

ATTACK 1 -- upper-CI sigma: re-run r=0.30 at the cluster-bootstrap UPPER-edge sigma
           (aniso [0.210,0.117] CI90 / [0.214,0.120] CI95 from verify_sigma_cluster_boot; iso-equiv
           0.170/0.173) at fr 0.35 & 0.50, v 30 & 37, both bias modes. p99 bootstrap CI on the cells.
ATTACK 2 -- at-speed sigma growth: sweep sigma_lat in {0.19,0.25,0.30,0.40} (vert HELD 0.10 AND vert
           SCALED with the L3 ratio), fr=0.50, v 30 & 37, bias 0. Bisection -> sigma_lat CEILING for
           r=0.30 closure. Plus a physically-motivated blur-scaled point (linear in speed).
ATTACK 3 -- lateral dominance: honest aniso [0.19,0.10] vs naive iso-0.10 over the fix-rate ladder;
           p99 over-optimism + first-closing fr gap.

CLOSURE rule = production: p90 < MARGIN(r) AND p99 < MARGIN(r). MARGIN(0.30)=0.235, MARGIN(0.38)=0.155.
[verify_sigma adversarial, 2026-06-14]
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import margin_driver_v2 as DV  # noqa: E402
import margin_envelope as ME   # noqa: E402

MARGIN_030 = ME.margin_at(0.30)
MARGIN_038 = ME.margin_at(0.38)
NMC = 3000
NBOOT = 2000
SEED = 20260614
PROCS = 6   # share the box with the two sibling verifier sessions


def _misses(v, slat, svert, fr, bm, nmc, bias=0.0, bias_m=0.0):
    accel = ME.att_deg_to_accel_bias(bias)
    out = np.empty(nmc)
    for s in range(nmc):
        seed = DV._seed_for(s, v, bias, slat, svert, fr, "cold", bm, 15.0, bias_m, False)
        rng = np.random.default_rng(seed)
        out[s] = DV.fly_lap_v2(rng, v, slat, svert, accel, bm, fr, "cold", 15.0, vert_fix_bias_m=bias_m)
    return out


def _summ(m, with_boot=False):
    p50, p90, p99 = (float(np.percentile(m, q)) for q in (50, 90, 99))
    d = dict(p50=p50, p90=p90, p99=p99,
             close_030_point=bool(p90 < MARGIN_030 and p99 < MARGIN_030),
             close_038_point=bool(p90 < MARGIN_038 and p99 < MARGIN_038))
    if with_boot:
        rng = np.random.default_rng(SEED)
        n = len(m)
        ps = np.array([np.percentile(m[rng.integers(0, n, n)], 99) for _ in range(NBOOT)])
        lo, hi = float(np.percentile(ps, 5)), float(np.percentile(ps, 95))
        d["p99_ci90"] = [lo, hi]
        d["close_030_honest"] = bool(p90 < MARGIN_030 and hi < MARGIN_030)
    return d


# ---- one picklable task: ("a1"|"a2sweep"|"a3", params...) ----
def _task(spec):
    kind = spec[0]
    if kind == "a1":
        _, lab, sl, sv, v, fr, bm = spec
        m = _misses(v, sl, sv, fr, bm, NMC)
        r = _summ(m, with_boot=True)
        r.update(sigma_label=lab, sigma_lat=sl, sigma_vert=sv, v=v, fr=fr, bias_mode=bm)
        return ("a1", r)
    if kind == "a2sweep":
        _, vp, v, bm, sl, sv = spec
        m = _misses(v, sl, sv, 0.50, bm, NMC)
        r = _summ(m)
        r.update(vert_policy=vp, v=v, bias_mode=bm, sigma_lat=sl, sigma_vert=sv, fr=0.50)
        return ("a2sweep", r)
    if kind == "a3":
        _, v, bm, fr = spec
        mh = _misses(v, 0.1914, 0.1008, fr, bm, NMC)
        mn = _misses(v, 0.10, 0.10, fr, bm, NMC)
        h, n = _summ(mh), _summ(mn)
        return ("a3", dict(v=v, bias_mode=bm, fr=fr,
                           honest_p90=h["p90"], honest_p99=h["p99"], honest_close030=h["close_030_point"],
                           naive_iso010_p90=n["p90"], naive_iso010_p99=n["p99"], naive_close030=n["close_030_point"],
                           p99_optimism_honest_minus_naive=round(h["p99"] - n["p99"], 4),
                           p99_optimism_pct=round(100 * (h["p99"] - n["p99"]) / h["p99"], 1)))
    raise ValueError(kind)


def build_specs():
    specs = []
    sig_sets = {
        "central_aniso_0.191/0.101": (0.1914, 0.1008),
        "upperCI90_aniso_0.210/0.117": (0.2103, 0.1168),
        "upperCI95_aniso_0.214/0.120": (0.2139, 0.1201),
        "central_iso_0.153": (0.152948, 0.152948),
        "upperCI90_iso_0.170": (0.170118, 0.170118),
    }
    for lab, (sl, sv) in sig_sets.items():
        for v in (30.0, 37.0):
            for fr in (0.35, 0.50):
                for bm in ("random3d", "inplane"):
                    specs.append(("a1", lab, sl, sv, v, fr, bm))
    RATIO = 0.1008 / 0.1914
    for vp in ("vert_held_0.10", "vert_scaled_ratio"):
        for v in (30.0, 37.0):
            for bm in ("random3d", "inplane"):
                for sl in (0.19, 0.25, 0.30, 0.40):
                    sv = 0.1008 if vp == "vert_held_0.10" else sl * RATIO
                    specs.append(("a2sweep", vp, v, bm, sl, sv))
    for v in (30.0, 37.0):
        for bm in ("random3d", "inplane"):
            for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
                specs.append(("a3", v, bm, fr))
    return specs


def bisection_ceiling():
    """sigma_lat CEILING for r=0.30 closure at fr=0.50 (point p90&p99<MARGIN030). Serial (cheap)."""
    RATIO = 0.1008 / 0.1914
    res = {}

    def closes_at(v, sl, vp, bm, nmc=2500):
        sv = 0.1008 if vp == "vert_held_0.10" else sl * RATIO
        m = _misses(v, sl, sv, 0.50, bm, nmc)
        return float(np.percentile(m, 90)) < MARGIN_030 and float(np.percentile(m, 99)) < MARGIN_030

    for vp in ("vert_held_0.10", "vert_scaled_ratio"):
        for v in (30.0, 37.0):
            for bm in ("random3d", "inplane"):
                key = f"{vp}|v{int(v)}|{bm}"
                lo, hi = 0.10, 0.55
                if not closes_at(v, lo, vp, bm):
                    res[key] = {"ceiling_sigma_lat": "<0.10"}; continue
                if closes_at(v, hi, vp, bm):
                    res[key] = {"ceiling_sigma_lat": ">0.55"}; continue
                for _ in range(7):
                    mid = 0.5 * (lo + hi)
                    if closes_at(v, mid, vp, bm):
                        lo = mid
                    else:
                        hi = mid
                c = round(0.5 * (lo + hi), 4)
                res[key] = {"ceiling_sigma_lat": c, "headroom_over_L3_0.191": round(c - 0.1914, 4),
                            "ratio_ceiling_over_L3": round(c / 0.1914, 2)}
    return res


def main():
    t0 = time.time()
    specs = build_specs()
    out = {"_meta": {"engine": "margin_driver_v2 (==ME.fly_lap iso/zerobias limit, verified 1e-9)",
                     "margin_030": MARGIN_030, "margin_038": MARGIN_038, "nmc": NMC, "nboot": NBOOT,
                     "seed": SEED, "procs": PROCS,
                     "regime": "COLD case-C, bias=0, latency 15ms, POST-bake (vert_fix_bias_m=0)",
                     "blur_scaling_note": "linear pixel-blur ~ speed*exposure/range; at fixed ~22m, "
                                          "18->30 m/s = x1.67 (0.191->0.319), 18->37 = x2.06 (0.191->0.394)"},
           "attack1_upperCI_sigma": [], "attack2_atspeed_sweep": [], "attack3_lateral_dominance": {"ladder": []}}
    a3rows = []
    with mp.Pool(processes=PROCS) as pool:
        for kind, r in pool.imap_unordered(_task, specs, chunksize=1):
            if kind == "a1":
                out["attack1_upperCI_sigma"].append(r)
            elif kind == "a2sweep":
                out["attack2_atspeed_sweep"].append(r)
            else:
                a3rows.append(r)
    out["attack3_lateral_dominance"]["ladder"] = a3rows

    # first-closing fr
    def first_fr(rows, key):
        d = {}
        for v in (30.0, 37.0):
            for bm in ("random3d", "inplane"):
                frs = [x["fr"] for x in rows if x["v"] == v and x["bias_mode"] == bm and x[key]]
                d[f"v{int(v)}|{bm}"] = min(frs) if frs else None
        return d
    out["attack3_lateral_dominance"]["first_fr_close030_honest"] = first_fr(a3rows, "honest_close030")
    out["attack3_lateral_dominance"]["first_fr_close030_naive_iso010"] = first_fr(a3rows, "naive_close030")

    print("== bisection ceiling (serial) ==", flush=True)
    out["attack2_atspeed_ceiling"] = bisection_ceiling()

    out["_meta"]["wall_s"] = round(time.time() - t0, 1)
    (HERE / "verify_sigma_margin_results.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE verify_sigma_margin_results.json ({out['_meta']['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
