"""ADVERSARIAL verify_sigma -- SERIAL single-process + per-cell DISK CHECKPOINT (the harness tears down
backgrounded process groups on the wrapper's exit-127, which BrokenPipe-kills any mp.Pool; serial has no
pipe to break, and the checkpoint survives a wrapper teardown). Box is now ~free (sibling pool ended).

Engine = margin_driver_v2 (==ME.fly_lap 1e-9 in iso/zerobias limit, verified this session).
COLD case-C, latency15ms, POST-bake. CLOSURE = p90<MARGIN(r) AND p99<MARGIN(r). MARGIN(0.30)=0.235.
NMC=2000. Resumes by skipping cells already in the checkpoint.
[verify_sigma adversarial, 2026-06-14]
"""
import json
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
NMC = 2000
RATIO = 0.1008 / 0.1914
CKPT = HERE / "verify_sigma_ckpt.json"
OUT = HERE / "verify_sigma_results.json"


def misses(v, sl, sv, fr, bm, nmc=NMC):
    accel = ME.att_deg_to_accel_bias(0.0)
    o = np.empty(nmc)
    for s in range(nmc):
        seed = DV._seed_for(s, v, 0.0, sl, sv, fr, "cold", bm, 15.0, 0.0, False)
        o[s] = DV.fly_lap_v2(np.random.default_rng(seed), v, sl, sv, accel, bm, fr, "cold", 15.0)
    return o


def cell(tag, v, sl, sv, fr, bm, extra):
    m = misses(v, sl, sv, fr, bm)
    p50, p90, p99 = (float(np.percentile(m, q)) for q in (50, 90, 99))
    return dict(tag=tag, v=v, sigma_lat=sl, sigma_vert=sv, fr=fr, bias_mode=bm, extra=extra,
                p50=round(p50, 4), p90=round(p90, 4), p99=round(p99, 4),
                close030=bool(p90 < M030 and p99 < M030), close038=bool(p90 < M038 and p99 < M038))


def build():
    specs = []
    a1sig = {"central_iso_0.153": (0.152948, 0.152948),
             "upperCI90_iso_0.170": (0.170118, 0.170118),
             "upperCI90_aniso_0.210/0.117": (0.2103, 0.1168),
             "upperCI95_aniso_0.214/0.120": (0.2139, 0.1201)}
    for lab, (sl, sv) in a1sig.items():
        for v in (30.0, 37.0):
            for fr in (0.35, 0.50):
                specs.append(("A1", v, sl, sv, fr, "random3d", lab))
    # Trimmed for a contended box: random3d (optimistic/headline) only; held=binding axis. lat grid
    # focused around the ceiling. v30 already mostly in checkpoint; v37 is the binding speed.
    a2lat = [0.19, 0.22, 0.24, 0.25, 0.28, 0.30, 0.40]
    for vp in ("held", "scaled"):
        for v in (30.0, 37.0):
            for bm in ("random3d",):
                for sl in a2lat:
                    sv = 0.1008 if vp == "held" else round(sl * RATIO, 4)
                    specs.append(("A2", v, sl, sv, 0.50, bm, vp))
    for v in (30.0, 37.0):
        for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
            specs.append(("A3h", v, 0.1914, 0.1008, fr, "random3d", "honest"))
            specs.append(("A3n", v, 0.10, 0.10, fr, "random3d", "naive_iso010"))
    return specs


def skey(s):
    return f"{s[0]}|{s[1]}|{s[2]}|{s[3]}|{s[4]}|{s[5]}|{s[6]}"


def main():
    t0 = time.time()
    specs = build()
    done = {}
    if CKPT.exists():
        for r in json.loads(CKPT.read_text()).get("cells", []):
            done[f"{r['tag']}|{r['v']}|{r['sigma_lat']}|{r['sigma_vert']}|{r['fr']}|{r['bias_mode']}|{r['extra']}"] = r
    cells = list(done.values())
    for i, s in enumerate(specs, 1):
        k = skey(s)
        if k in done:
            continue
        r = cell(*s)
        cells.append(r)
        CKPT.write_text(json.dumps({"cells": cells}, indent=1))
        print(f"[{i}/{len(specs)}] {r['tag']:4s} v{int(r['v'])} lat{r['sigma_lat']} vert{r['sigma_vert']} "
              f"fr{r['fr']} {r['bias_mode'][:4]} {str(r['extra'])[:18]:18s} p90 {r['p90']:.3f} p99 {r['p99']:.3f} "
              f"c030={r['close030']}", flush=True)

    a1 = [r for r in cells if r["tag"] == "A1"]
    a2 = [r for r in cells if r["tag"] == "A2"]
    a3h = [r for r in cells if r["tag"] == "A3h"]
    a3n = [r for r in cells if r["tag"] == "A3n"]

    a3rows = []
    for v in (30.0, 37.0):
        for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
            h = next((x for x in a3h if x["v"] == v and x["fr"] == fr), None)
            n = next((x for x in a3n if x["v"] == v and x["fr"] == fr), None)
            if not h or not n:
                continue
            a3rows.append(dict(v=v, fr=fr, honest_p90=h["p90"], honest_p99=h["p99"], honest_close030=h["close030"],
                               naive_p90=n["p90"], naive_p99=n["p99"], naive_close030=n["close030"],
                               p99_optimism=round(h["p99"] - n["p99"], 4),
                               p99_optimism_pct=round(100 * (h["p99"] - n["p99"]) / h["p99"], 1)))

    def first(key, v):
        frs = [r["fr"] for r in a3rows if r["v"] == v and r[key]]
        return min(frs) if frs else None

    def ceiling(a2rows):
        out = {}
        cfgs = {}
        for r in a2rows:
            cfgs.setdefault(f"{r['extra']}|v{int(r['v'])}|{r['bias_mode']}", []).append(r)
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
                opens_above = [x for x in nonclosing if x > last_close]
                first_open = min(opens_above) if opens_above else None
                mid = round((last_close + first_open) / 2, 3) if first_open else last_close
                out[key] = {"bracket_close_open": [last_close, first_open], "ceiling_mid": mid,
                            "ratio_over_L3_0.191": round(mid / 0.1914, 2),
                            "headroom_m": round(mid - 0.1914, 3)}
        return out

    out = {"_meta": {"engine": "margin_driver_v2 (==ME.fly_lap 1e-9 iso/zerobias)", "nmc": NMC,
                     "margin_030": M030, "margin_038": M038,
                     "regime": "COLD case-C bias=0 latency15ms POST-bake random3d",
                     "wall_s": round(time.time() - t0, 1)},
           "attack1_upperCI_sigma": sorted(a1, key=lambda x: (str(x["extra"]), x["v"], x["fr"])),
           "attack2_atspeed_sweep": sorted(a2, key=lambda x: (x["extra"], x["v"], x["bias_mode"], x["sigma_lat"])),
           "attack2_ceiling": ceiling(a2),
           "attack2_blur_extrapolation": {"L3_speed_mps": 18.0, "L3_sigma_lat": 0.1914,
                                          "v30_linear_blur_pred": round(0.1914 * 30 / 18, 3),
                                          "v37_linear_blur_pred": round(0.1914 * 37 / 18, 3),
                                          "note": "linear pixel-blur ~ speed*exposure/range at fixed ~22m range"},
           "attack3_lateral_dominance": {"ladder": a3rows,
                                         "first_fr_close030_honest": {f"v{int(v)}": first("honest_close030", v) for v in (30.0, 37.0)},
                                         "first_fr_close030_naive_iso010": {f"v{int(v)}": first("naive_close030", v) for v in (30.0, 37.0)}}}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"WROTE {OUT.name} ({out['_meta']['wall_s']}s, {len(cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
