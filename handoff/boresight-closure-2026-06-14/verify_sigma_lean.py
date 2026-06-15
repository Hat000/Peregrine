"""ADVERSARIAL verify_sigma -- LEAN, FAST decision-cell-only version (serial; box is contended by a
sibling clustering verifier running a 10-proc pool). Computes ONLY the verdict-critical numbers.

Engine = margin_driver_v2 (regression-proven == ME.fly_lap to 1e-9 in iso/zerobias limit, this session).
COLD case-C, latency 15ms, POST-bake (vert_fix_bias_m=0). CLOSURE = p90<MARGIN(r) AND p99<MARGIN(r).
MARGIN(0.30)=0.235. We also report the 1500x p99 bootstrap UPPER edge (HONEST closure) on A1 cells.

A1 upper-CI sigma  -> r=0.30 at central/upperCI iso + upperCI aniso, fr 0.35 & 0.50, v 30 & 37.
A2 at-speed ceiling -> bisection sigma_lat ceiling for r=0.30 @ fr=0.50 (vert held 0.10 & vert scaled);
                       + discrete blur points (0.25/0.30/0.40) at v37 random3d.
A3 lateral dominance -> honest [0.19,0.10] vs naive iso-0.10 first-closing fr (v30/37 random3d).
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
NBOOT = 1500
RATIO = 0.1008 / 0.1914  # 0.5266


def misses(v, sl, sv, fr, bm, nmc=NMC, bias=0.0):
    accel = ME.att_deg_to_accel_bias(bias)
    o = np.empty(nmc)
    for s in range(nmc):
        seed = DV._seed_for(s, v, bias, sl, sv, fr, "cold", bm, 15.0, 0.0, False)
        o[s] = DV.fly_lap_v2(np.random.default_rng(seed), v, sl, sv, accel, bm, fr, "cold", 15.0)
    return o


def pp(m):
    return float(np.percentile(m, 50)), float(np.percentile(m, 90)), float(np.percentile(m, 99))


def boot_hi99(m, seed=20260614):
    rng = np.random.default_rng(seed)
    n = len(m)
    ps = np.array([np.percentile(m[rng.integers(0, n, n)], 99) for _ in range(NBOOT)])
    return float(np.percentile(ps, 95))


def closes030(m):
    _, p90, p99 = pp(m)
    return bool(p90 < M030 and p99 < M030)


def attack1():
    sig = {"central_iso_0.153": (0.152948, 0.152948),
           "upperCI90_iso_0.170": (0.170118, 0.170118),
           "upperCI90_aniso_0.210/0.117": (0.2103, 0.1168),
           "upperCI95_aniso_0.214/0.120": (0.2139, 0.1201)}
    rows = []
    for lab, (sl, sv) in sig.items():
        for v in (30.0, 37.0):
            for fr in (0.35, 0.50):
                m = misses(v, sl, sv, fr, "random3d")
                p50, p90, p99 = pp(m)
                hi = boot_hi99(m)
                rows.append(dict(sigma_label=lab, v=v, fr=fr, p50=round(p50, 4), p90=round(p90, 4),
                                 p99=round(p99, 4), p99_hi_ci90=round(hi, 4),
                                 close030_point=bool(p90 < M030 and p99 < M030),
                                 close030_honest=bool(p90 < M030 and hi < M030)))
                print(f"  A1 {lab:30s} v{int(v)} fr{fr}: p90 {p90:.3f} p99 {p99:.3f} "
                      f"hiCI {hi:.3f} close030={p90<M030 and p99<M030}", flush=True)
    return rows


def attack2():
    out = {"ceiling": {}, "discrete_blur_v37_random3d": [], "discrete_blur_v30_random3d": []}

    def cl(v, sl, vp, bm, nmc=2000):
        sv = 0.1008 if vp == "held" else sl * RATIO
        return closes030(misses(v, sl, sv, 0.50, bm, nmc))

    for vp in ("held", "scaled"):
        for v in (30.0, 37.0):
            for bm in ("random3d", "inplane"):
                key = f"{vp}|v{int(v)}|{bm}"
                lo, hi = 0.10, 0.55
                if not cl(v, lo, vp, bm):
                    out["ceiling"][key] = "<0.10"; print(f"  A2 ceil {key}: <0.10", flush=True); continue
                if cl(v, hi, vp, bm):
                    out["ceiling"][key] = ">0.55"; print(f"  A2 ceil {key}: >0.55", flush=True); continue
                for _ in range(7):
                    mid = 0.5 * (lo + hi)
                    if cl(v, mid, vp, bm):
                        lo = mid
                    else:
                        hi = mid
                c = round(0.5 * (lo + hi), 4)
                out["ceiling"][key] = {"ceiling_sigma_lat": c, "headroom_over_0.191": round(c - 0.1914, 4),
                                       "ratio_over_L3": round(c / 0.1914, 2)}
                print(f"  A2 ceil {key}: {c}  (ratio {c/0.1914:.2f}x L3)", flush=True)

    for v, bucket in ((37.0, "discrete_blur_v37_random3d"), (30.0, "discrete_blur_v30_random3d")):
        for sl in (0.19, 0.25, 0.30, 0.40):
            m = misses(v, sl, 0.1008, 0.50, "random3d")  # vert held
            p50, p90, p99 = pp(m)
            out[bucket].append(dict(sigma_lat=sl, sigma_vert=0.1008, p90=round(p90, 4), p99=round(p99, 4),
                                    close030=bool(p90 < M030 and p99 < M030)))
            print(f"  A2 blur v{int(v)} lat{sl}/vert0.10 fr0.50: p90 {p90:.3f} p99 {p99:.3f} "
                  f"close030={p90<M030 and p99<M030}", flush=True)
    # blur-scaling map (linear pixel blur ~ speed at fixed range): 18->30 x1.67 ->0.319 ; 18->37 x2.06 ->0.394
    out["blur_extrapolation"] = {"L3_speed_mps": 18.0, "L3_sigma_lat": 0.1914,
                                 "v30_linear": round(0.1914 * 30 / 18, 3), "v37_linear": round(0.1914 * 37 / 18, 3)}
    return out


def attack3():
    rows = []
    for v in (30.0, 37.0):
        for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
            mh = misses(v, 0.1914, 0.1008, fr, "random3d")
            mn = misses(v, 0.10, 0.10, fr, "random3d")
            _, h90, h99 = pp(mh)
            _, n90, n99 = pp(mn)
            rows.append(dict(v=v, fr=fr, honest_p90=round(h90, 4), honest_p99=round(h99, 4),
                             honest_close030=bool(h90 < M030 and h99 < M030),
                             naive_iso010_p90=round(n90, 4), naive_iso010_p99=round(n99, 4),
                             naive_close030=bool(n90 < M030 and n99 < M030),
                             p99_optimism=round(h99 - n99, 4),
                             p99_optimism_pct=round(100 * (h99 - n99) / h99, 1)))
            print(f"  A3 v{int(v)} fr{fr}: honest p99 {h99:.3f}({'C' if h90<M030 and h99<M030 else 'x'}) "
                  f"naive p99 {n99:.3f}({'C' if n90<M030 and n99<M030 else 'x'}) "
                  f"optim {100*(h99-n99)/h99:.0f}%", flush=True)

    def first(rows, key, v):
        frs = [r["fr"] for r in rows if r["v"] == v and r[key]]
        return min(frs) if frs else None
    return {"ladder": rows,
            "first_fr_close030_honest": {f"v{int(v)}": first(rows, "honest_close030", v) for v in (30.0, 37.0)},
            "first_fr_close030_naive_iso010": {f"v{int(v)}": first(rows, "naive_close030", v) for v in (30.0, 37.0)}}


def main():
    t0 = time.time()
    out = {"_meta": {"engine": "margin_driver_v2 (==ME.fly_lap 1e-9 iso/zerobias)", "margin_030": M030,
                     "margin_038": M038, "nmc": NMC, "nboot": NBOOT,
                     "regime": "COLD case-C, bias=0, latency15ms, POST-bake", "mode": "random3d (optimistic) for A1/A3; both for A2 ceiling"}}
    print("== A1 upper-CI sigma ==", flush=True)
    out["attack1_upperCI_sigma"] = attack1()
    print("== A2 at-speed ceiling ==", flush=True)
    out["attack2_atspeed"] = attack2()
    print("== A3 lateral dominance ==", flush=True)
    out["attack3_lateral_dominance"] = attack3()
    out["_meta"]["wall_s"] = round(time.time() - t0, 1)
    (HERE / "verify_sigma_results.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE verify_sigma_results.json ({out['_meta']['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
