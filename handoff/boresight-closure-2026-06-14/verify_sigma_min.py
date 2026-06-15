"""ADVERSARIAL verify_sigma -- MINIMAL pooled (4 workers) for a CPU-saturated box (sibling verifier
owns a 10-proc pool). No per-cell bootstrap (point p90/p99 closure is sufficient for refutation; the
production re-run already supplies bootstrap CIs and they do not flip the verdict). NMC=1500.

Engine = margin_driver_v2 (==ME.fly_lap 1e-9 in iso/zerobias limit, verified this session).
COLD case-C, latency15ms, POST-bake. CLOSURE = p90<MARGIN(r) AND p99<MARGIN(r). MARGIN(0.30)=0.235.

A1 upper-CI sigma; A2 at-speed sigma_lat sweep + ceiling (computed from the sweep, no bisection);
A3 honest-vs-naive lateral dominance. All random3d (optimistic) except A2 also does inplane.
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

M030 = ME.margin_at(0.30)
M038 = ME.margin_at(0.38)
NMC = 1500
RATIO = 0.1008 / 0.1914
PROCS = 4


def _cell(spec):
    tag, v, sl, sv, fr, bm, extra = spec
    accel = ME.att_deg_to_accel_bias(0.0)
    m = np.empty(NMC)
    for s in range(NMC):
        seed = DV._seed_for(s, v, 0.0, sl, sv, fr, "cold", bm, 15.0, 0.0, False)
        m[s] = DV.fly_lap_v2(np.random.default_rng(seed), v, sl, sv, accel, bm, fr, "cold", 15.0)
    p50, p90, p99 = (float(np.percentile(m, q)) for q in (50, 90, 99))
    return dict(tag=tag, v=v, sigma_lat=sl, sigma_vert=sv, fr=fr, bias_mode=bm, extra=extra,
                p50=round(p50, 4), p90=round(p90, 4), p99=round(p99, 4),
                close030=bool(p90 < M030 and p99 < M030), close038=bool(p90 < M038 and p99 < M038))


def build():
    specs = []
    # A1: r=0.30 at central + upper-CI sigmas, fr 0.35 & 0.50, v 30 & 37, random3d
    a1sig = {"central_iso_0.153": (0.152948, 0.152948),
             "upperCI90_iso_0.170": (0.170118, 0.170118),
             "upperCI90_aniso_0.210/0.117": (0.2103, 0.1168),
             "upperCI95_aniso_0.214/0.120": (0.2139, 0.1201)}
    for lab, (sl, sv) in a1sig.items():
        for v in (30.0, 37.0):
            for fr in (0.35, 0.50):
                specs.append(("A1", v, sl, sv, fr, "random3d", lab))
    # A2: at-speed lateral sweep fr=0.50, vert held 0.10 AND scaled, v 30&37, both bias modes
    a2lat = [0.19, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
    for vp in ("held", "scaled"):
        for v in (30.0, 37.0):
            for bm in ("random3d", "inplane"):
                for sl in a2lat:
                    sv = 0.1008 if vp == "held" else round(sl * RATIO, 4)
                    specs.append(("A2", v, sl, sv, 0.50, bm, vp))
    # A3: honest [0.19,0.10] vs naive iso-0.10 ladder, v 30&37, random3d
    for v in (30.0, 37.0):
        for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
            specs.append(("A3h", v, 0.1914, 0.1008, fr, "random3d", "honest"))
            specs.append(("A3n", v, 0.10, 0.10, fr, "random3d", "naive_iso010"))
    return specs


def ceiling_from_sweep(a2rows):
    """Largest sigma_lat that still closes r=0.30 (point) per config, from the discrete sweep grid."""
    out = {}
    cfgs = {}
    for r in a2rows:
        key = f"{r['extra']}|v{int(r['v'])}|{r['bias_mode']}"
        cfgs.setdefault(key, []).append(r)
    for key, rows in cfgs.items():
        rows = sorted(rows, key=lambda x: x["sigma_lat"])
        closing = [x["sigma_lat"] for x in rows if x["close030"]]
        nonclosing = [x["sigma_lat"] for x in rows if not x["close030"]]
        if not closing:
            out[key] = {"ceiling_sigma_lat": f"<{rows[0]['sigma_lat']}"}
        elif not nonclosing:
            out[key] = {"ceiling_sigma_lat": f">{rows[-1]['sigma_lat']}"}
        else:
            last_close = max(closing)
            first_open = min(x for x in nonclosing if x > last_close) if any(x > last_close for x in nonclosing) else None
            mid = round((last_close + first_open) / 2, 3) if first_open else last_close
            out[key] = {"ceiling_sigma_lat_bracket": [last_close, first_open], "ceiling_mid": mid,
                        "ratio_over_L3": round(mid / 0.1914, 2)}
    return out


def main():
    t0 = time.time()
    specs = build()
    results = []
    with mp.Pool(PROCS) as pool:
        for i, r in enumerate(pool.imap_unordered(_cell, specs, chunksize=1), 1):
            results.append(r)
            print(f"[{i}/{len(specs)}] {r['tag']:4s} v{int(r['v'])} lat{r['sigma_lat']} vert{r['sigma_vert']} "
                  f"fr{r['fr']} {r['bias_mode'][:4]} {r['extra'][:22]:22s} p90 {r['p90']:.3f} p99 {r['p99']:.3f} "
                  f"c030={r['close030']}", flush=True)

    a1 = [r for r in results if r["tag"] == "A1"]
    a2 = [r for r in results if r["tag"] == "A2"]
    a3h = [r for r in results if r["tag"] == "A3h"]
    a3n = [r for r in results if r["tag"] == "A3n"]

    # A3 pairing + first-closing fr
    a3rows = []
    for v in (30.0, 37.0):
        for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
            h = next(x for x in a3h if x["v"] == v and x["fr"] == fr)
            n = next(x for x in a3n if x["v"] == v and x["fr"] == fr)
            a3rows.append(dict(v=v, fr=fr, honest_p90=h["p90"], honest_p99=h["p99"], honest_close030=h["close030"],
                               naive_p90=n["p90"], naive_p99=n["p99"], naive_close030=n["close030"],
                               p99_optimism=round(h["p99"] - n["p99"], 4),
                               p99_optimism_pct=round(100 * (h["p99"] - n["p99"]) / h["p99"], 1)))

    def first(key, v):
        frs = [r["fr"] for r in a3rows if r["v"] == v and r[key]]
        return min(frs) if frs else None

    out = {"_meta": {"engine": "margin_driver_v2 (==ME.fly_lap 1e-9 iso/zerobias)", "nmc": NMC,
                     "margin_030": M030, "margin_038": M038, "procs": PROCS,
                     "regime": "COLD case-C bias=0 latency15ms POST-bake", "wall_s": round(time.time() - t0, 1),
                     "note": "point-closure verdict (no per-cell bootstrap; prod re-run supplies CIs)"},
           "attack1_upperCI_sigma": sorted(a1, key=lambda x: (x["extra"], x["v"], x["fr"])),
           "attack2_atspeed_sweep": sorted(a2, key=lambda x: (x["extra"], x["v"], x["bias_mode"], x["sigma_lat"])),
           "attack2_ceiling": ceiling_from_sweep(a2),
           "attack2_blur_extrapolation": {"L3_speed": 18.0, "L3_sigma_lat": 0.1914,
                                          "v30_linear_blur": round(0.1914 * 30 / 18, 3),
                                          "v37_linear_blur": round(0.1914 * 37 / 18, 3)},
           "attack3_lateral_dominance": {"ladder": a3rows,
                                         "first_fr_close030_honest": {f"v{int(v)}": first("honest_close030", v) for v in (30.0, 37.0)},
                                         "first_fr_close030_naive_iso010": {f"v{int(v)}": first("naive_close030", v) for v in (30.0, 37.0)}}}
    (HERE / "verify_sigma_results.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE verify_sigma_results.json ({out['_meta']['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
