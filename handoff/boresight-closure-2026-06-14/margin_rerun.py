"""BORESIGHT-CLOSURE Deliverable-3 -- GATE-4 COLD case-C margin RE-RUN (post-bake + real L3 sigma).

ONE question: does the gate-4 COLD contact margin CLOSE once the measured -0.25 m vertical boresight
bias (epsilon_vert) is BAKED OUT and the REAL (L3-measured) gate-4 anisotropic per-fix sigma is used?

ENGINE: handoff/boresight-closure-2026-06-14/margin_driver_v2.py (fly_lap_v2 / run_cell_v2). That file is
ME.fly_lap (production LinearKF+RewindKF cold case-C) VERBATIM outside three documented changes
(anisotropic draw, anisotropic cov, +e2 vertical fix bias). regression_proof.py PROVES it reproduces the
production baseline bit-for-bit in the isotropic/zero-bias limit AND reproduces the banked confirm_verdict
p90/p99 to 1-2 mm. So results here are the production stack, just with the real anisotropic gate-4 sigma
and the optional pre-bake bias channel.

BAKE SEMANTICS (read before running)
------------------------------------
POST-bake  == vert_fix_bias_m = 0.0   (the deployed +L fix is UNBIASED; == ME.fly_lap; the bake on main
              cancels epsilon_vert at the lever). This is the column the verdict rides on.
PRE-bake   == vert_fix_bias_m = +0.25 (inject the measured gate-DOWN -0.25 m epsilon along the world-UP
              in-plane axis e2; coefficient sign per margin_driver_v2 header: +0.25 -> -0.25 m gate-DOWN).
The PRE vs POST delta at the key cells QUANTIFIES how load-bearing the bake is.

SIGMA (from the sigma-recal worker, banked in sigma_gate4_l3.json):
  iso 0.1529  ==  sqrt((lat^2+vert^2)/2)  (the single-sigma equivalent for ME's one-sigma engine)
  aniso       ==  [lat 0.1914, vert 0.1008]  (the HONEST shape: post-bake the vertical axis is clean 0.10,
                  but the lateral axis is crab-coupled 0.19 at this high-crab gate -> miss is LATERAL-dom)
  iso 0.19    ==  conservative single-sigma (lateral floor as a both-axes upper bound)
  iso 0.10    ==  clean-vertical-only (UNDER-states the in-plane miss; kept as a reference column)
  iso 0.265   ==  the OLD pessimistic modeled constant (baseline anchor, grid C)

GRID
----
A. POST-bake sweep:  v in {30,37} x fr in {0.07,0.15,0.25,0.35,0.50} x att_bias in {0.0,0.6,1.4}
   x sigma in {iso0.10, aniso[0.19,0.10], iso0.19} x bias_mode in {random3d, inplane}.  nmc=A_NMC (>=2000).
B. PRE-bake delta: SAME engine, vert_fix_bias_m=+0.25, at fr in {0.07,0.25} x bias in {0.0,0.6}
   x v in {30,37} x aniso sigma x bias_mode in {random3d,inplane}.  nmc=A_NMC.
C. BASELINE anchor: iso 0.265, POST-bake, fr 0.07, bias 0, v37, random3d -- sanity vs banked confirm_verdict.
D. DECISION cells (HIGH nmc D_NMC>=4000 + bootstrap 90% CI on p90 AND p99, mirroring confirm_verdict.py):
   the measured op point (fr0.07,bias0) + closure-boundary fix-rates {0.15,0.25,0.35} at r=0.30 AND r=0.38,
   POST-bake, ANISO sigma, BOTH bias modes, v in {30,37}.  PLUS the matched PRE-bake measured op point so
   the bake delta is CI-honest at the decision point.  Closure is HONEST iff UPPER CI of p99 < MARGIN(r).

Run:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    ../../../.venv/Scripts/python.exe handoff/boresight-closure-2026-06-14/margin_rerun.py \
      [--a_nmc 2500] [--d_nmc 5000] [--nboot 2000] [--procs 11]
[BORESIGHT-CLOSURE 2026-06-14]
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

import numpy as np

_HERE = Path(__file__).resolve().parent
_ME_DIR = _HERE.parents[0] / "margin-closure-envelope-2026-06-14"
for _p in (str(_ME_DIR), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import margin_envelope as ME           # noqa: E402  (production engine + constants)
import margin_driver_v2 as V2          # noqa: E402  (anisotropic + bias engine, regression-proven)

# ---- the recalibrated L3 gate-4 sigma (sigma_gate4_l3.json / the prompt's input) ----
SIG_ISO = 0.1529
SIG_LAT = 0.1914
SIG_VERT = 0.1008
SIG_ISO_CONS = 0.19          # conservative single-sigma (lateral floor as both-axes bound)
SIG_ISO_CLEAN = 0.10         # clean-vertical-only reference
SIG_MODELED = 0.265          # old pessimistic baseline
PRE_BAKE_BIAS = 0.25         # +0.25 on e2 == the measured -0.25 m gate-DOWN epsilon (PRE-bake)
RADII = ME.RADIUS_BAND       # [0.21,0.26,0.30,0.33,0.38]
MARGIN = {f"{r:.2f}": ME.margin_at(r) for r in RADII}

# Named sigma configs: (label, sigma_lat, sigma_vert)
SIGMAS = {
    "iso0.10":        (SIG_ISO_CLEAN, SIG_ISO_CLEAN),
    "aniso0.19/0.10": (SIG_LAT, SIG_VERT),
    "iso0.19":        (SIG_ISO_CONS, SIG_ISO_CONS),
    "iso0.1529":      (SIG_ISO, SIG_ISO),
    "iso0.265":       (SIG_MODELED, SIG_MODELED),
}


# =================================================================================================
# Cell-level MC (returns the full miss array so D can bootstrap, and A/B/C read percentiles + frac).
# Seeds: reuse run_cell_v2's per-lap seed via fly_lap_v2 directly with a per-cell deterministic base,
# so every cell is reproducible and order-independent under multiprocessing.
# =================================================================================================
def _cell_base_seed(v, sig_lat, sig_vert, bias_deg, bias_mode, fix_rate, vert_bias):
    return (ME.SEED + int(round(v)) * 13 + int(round(bias_deg * 100)) * 7
            + int(round(sig_lat * 1000)) * 17 + int(round(fix_rate * 1000)) * 23
            + 1009  # cold
            + {"random3d": 0, "inplane": 1}[bias_mode] * 3001 + 15 * 53
            + int(round(sig_vert * 1000)) * 131 + int(round(vert_bias * 1000)) * 211)


def _lap_worker(args):
    seed, v, sig_lat, sig_vert, bmag, bias_mode, fix_rate, vert_bias = args
    rng = np.random.default_rng(seed)
    return V2.fly_lap_v2(rng, v, sig_lat, sig_vert, bmag, bias_mode, fix_rate, "cold", 15.0,
                         vert_fix_bias_m=vert_bias)


def cell_misses(pool, v, sig_lat, sig_vert, bias_deg, bias_mode, fix_rate, vert_bias, nmc):
    bmag = ME.att_deg_to_accel_bias(bias_deg)
    base = _cell_base_seed(v, sig_lat, sig_vert, bias_deg, bias_mode, fix_rate, vert_bias)
    specs = [(base + 101 * s, v, sig_lat, sig_vert, bmag, bias_mode, fix_rate, vert_bias)
             for s in range(nmc)]
    # chunksize cuts Windows spawn-pool IPC overhead: ~ nmc/(4*procs) tasks per dispatch.
    chunk = max(1, nmc // 44)
    return np.array(pool.map(_lap_worker, specs, chunksize=chunk))


def summarize(m):
    p50, p90, p99 = (float(np.percentile(m, q)) for q in (50, 90, 99))
    frac = {f"{r:.2f}": float(np.mean(m >= ME.margin_at(r))) for r in RADII}
    clears = {f"{r:.2f}": bool(p90 < ME.margin_at(r) and p99 < ME.margin_at(r)) for r in RADII}
    return dict(p50=p50, p90=p90, p99=p99, rms=float(np.sqrt(np.mean(m ** 2))), maxm=float(m.max()),
                frac_over=frac, clears_p90_and_p99=clears)


def boot_ci(m, q, nboot, seed=7):
    rng = np.random.default_rng(seed)
    n = len(m)
    vals = np.array([np.percentile(m[rng.integers(0, n, n)], q) for _ in range(nboot)])
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


# =================================================================================================
def run_grid_A(pool, nmc):
    """POST-bake operating sweep (vert_bias=0). FOCUSED coverage (per-lap cost ~75 ms => keep cell count
    bounded): the HONEST aniso sigma gets the full fix-rate x bias x bias_mode factorial at both speeds;
    the iso0.10 (clean-vertical reference) and iso0.19 (conservative) sigmas get the fix-rate sweep at
    bias 0, random3d only (the sigma-shape comparison). The verdict already rests on the nmc=5000 grid D;
    grid A is the full-radius envelope SHAPE + the fix-rate-is-binding picture."""
    cells = []
    n = 0
    total = 2 * (5 * 3 * 2 + 5 * 2 * 1)   # speeds * (aniso full + 2 ref sigmas at b0/rnd)
    for v in (30.0, 37.0):
        # honest aniso: full fr x bias x bias_mode
        sl, sv = SIGMAS["aniso0.19/0.10"]
        for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
            for b in (0.0, 0.6, 1.4):
                for bm in ("random3d", "inplane"):
                    m = cell_misses(pool, v, sl, sv, b, bm, fr, 0.0, nmc)
                    cells.append(dict(grid="A", bake="post", v=v, sigma="aniso0.19/0.10",
                                      sigma_lat=sl, sigma_vert=sv, fix_rate=fr, att_bias_deg=b,
                                      bias_mode=bm, vert_fix_bias_m=0.0, nmc=nmc, **summarize(m)))
                    n += 1
        # reference sigmas: fr sweep at bias 0, random3d (sigma-shape comparison)
        for sig_label in ("iso0.10", "iso0.19"):
            sl, sv = SIGMAS[sig_label]
            for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
                m = cell_misses(pool, v, sl, sv, 0.0, "random3d", fr, 0.0, nmc)
                cells.append(dict(grid="A", bake="post", v=v, sigma=sig_label,
                                  sigma_lat=sl, sigma_vert=sv, fix_rate=fr, att_bias_deg=0.0,
                                  bias_mode="random3d", vert_fix_bias_m=0.0, nmc=nmc, **summarize(m)))
                n += 1
        print(f"  [A] v{v:.0f} done ({n}/{total} cells)", flush=True)
    return cells


def run_grid_B(pool, nmc):
    """PRE-bake (vert_bias=+0.25) at the key cells, ANISO sigma -> bake DELTA vs A."""
    cells = []
    for v in (30.0, 37.0):
        sl, sv = SIGMAS["aniso0.19/0.10"]
        for fr in (0.07, 0.25):
            for b in (0.0, 0.6):
                for bm in ("random3d", "inplane"):
                    m = cell_misses(pool, v, sl, sv, b, bm, fr, PRE_BAKE_BIAS, nmc)
                    rec = dict(grid="B", bake="pre", v=v, sigma="aniso0.19/0.10",
                               sigma_lat=sl, sigma_vert=sv, fix_rate=fr, att_bias_deg=b,
                               bias_mode=bm, vert_fix_bias_m=PRE_BAKE_BIAS, nmc=nmc, **summarize(m))
                    cells.append(rec)
                    print(f"  [B] v{v:.0f} {bm:8s} fr{fr:.2f} b{b:.1f} pre: p90 {rec['p90']:.3f} "
                          f"p99 {rec['p99']:.3f}", flush=True)
    return cells


def run_grid_C(pool, nmc):
    """BASELINE anchor: iso 0.265 post-bake fr0.07 bias0 v37 random3d -- sanity vs banked confirm_verdict."""
    sl, sv = SIGMAS["iso0.265"]
    m = cell_misses(pool, 37.0, sl, sv, 0.0, "random3d", 0.07, 0.0, nmc)
    rec = dict(grid="C", bake="post", v=37.0, sigma="iso0.265", sigma_lat=sl, sigma_vert=sv,
               fix_rate=0.07, att_bias_deg=0.0, bias_mode="random3d", vert_fix_bias_m=0.0,
               nmc=nmc, banked_ref=dict(p90=0.210, p99=0.297, note="confirm_verdict MEASURED-equiv but iso0.265; "
                                        "banked 0.265 b0 fr0.07 was not in confirm; compare envelope"),
               **summarize(m))
    return [rec]


def run_grid_D(pool, nmc, nboot):
    """DECISION cells: high nmc + bootstrap 90% CI on p90 AND p99. POST-bake aniso, both bias modes,
    v in {30,37}, fr in {0.07,0.15,0.25,0.35} read at r=0.30 and r=0.38. PLUS the matched PRE-bake
    measured op point (CI-honest bake delta)."""
    sl, sv = SIGMAS["aniso0.19/0.10"]
    targets = []
    for v in (30.0, 37.0):
        for bm in ("random3d", "inplane"):
            for fr in (0.07, 0.15, 0.25, 0.35):
                targets.append(dict(v=v, fr=fr, bm=bm, b=0.0, vbias=0.0, tag="post"))
    # matched PRE-bake measured op point (fr0.07,b0) for the CI-honest delta, both bias modes, both speeds
    for v in (30.0, 37.0):
        for bm in ("random3d", "inplane"):
            targets.append(dict(v=v, fr=0.07, bm=bm, b=0.0, vbias=PRE_BAKE_BIAS, tag="pre"))
    # one bias-stress decision cell at the closure boundary (b0.6, fr0.25) both modes, v30/37
    for v in (30.0, 37.0):
        for bm in ("random3d", "inplane"):
            targets.append(dict(v=v, fr=0.25, bm=bm, b=0.6, vbias=0.0, tag="post"))

    cells = []
    for t in targets:
        m = cell_misses(pool, t["v"], sl, sv, t["b"], t["bm"], t["fr"], t["vbias"], nmc)
        p90, p99 = float(np.percentile(m, 90)), float(np.percentile(m, 99))
        p90lo, p90hi = boot_ci(m, 90, nboot)
        p99lo, p99hi = boot_ci(m, 99, nboot)
        verdicts = {}
        for r in (0.30, 0.38):
            mr = ME.margin_at(r)
            # HONEST closure: upper CI of BOTH p90 and p99 below margin.
            if p90hi < mr and p99hi < mr:
                v_ = "CLOSE"
            elif p99lo >= mr:
                v_ = "NO_CLOSE"
            elif p99lo < mr <= p99hi:
                v_ = "KNIFE_EDGE"   # CI straddles the margin
            else:
                v_ = "NO_CLOSE"
            verdicts[f"{r:.2f}"] = v_
        rec = dict(grid="D", bake=t["tag"], v=t["v"], sigma="aniso0.19/0.10", sigma_lat=sl,
                   sigma_vert=sv, fix_rate=t["fr"], att_bias_deg=t["b"], bias_mode=t["bm"],
                   vert_fix_bias_m=t["vbias"], nmc=nmc, nboot=nboot,
                   p50=float(np.percentile(m, 50)), p90=p90, p99=p99,
                   p90_ci=[p90lo, p90hi], p99_ci=[p99lo, p99hi],
                   frac_over={f"{r:.2f}": float(np.mean(m >= ME.margin_at(r))) for r in RADII},
                   margin={f"{r:.2f}": ME.margin_at(r) for r in (0.30, 0.38)},
                   verdict=verdicts)
        cells.append(rec)
        print(f"  [D] v{t['v']:.0f} {t['bm']:8s} fr{t['fr']:.2f} b{t['b']:.1f} {t['tag']:4s} "
              f"p90 {p90:.3f}[{p90lo:.3f},{p90hi:.3f}] p99 {p99:.3f}[{p99lo:.3f},{p99hi:.3f}] "
              f"r30 {verdicts['0.30']:10s} r38 {verdicts['0.38']}", flush=True)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_nmc", type=int, default=2500)
    ap.add_argument("--d_nmc", type=int, default=5000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--grids", type=str, default="CDBA",
                    help="which grids to run, e.g. 'CDBA' (all) or 'BA' (resume A+B, keep checkpointed C/D)")
    args = ap.parse_args()
    grids = set(args.grids.upper())

    t0 = time.time()
    out_path = _HERE / "margin_rerun_results.json"
    # RESUME: if not running C/D, load the existing checkpoint so its C/D cells survive.
    prev = {}
    if out_path.exists() and not ({"C", "D"} <= grids):
        try:
            prev = json.loads(out_path.read_text())
        except Exception:  # noqa: BLE001
            prev = {}

    out = dict(meta=dict(
        what="GATE-4 COLD case-C margin RE-RUN: post-bake (eps removed) + real L3 anisotropic sigma",
        seed=ME.SEED, w_eff=ME.W_EFF, radius_band=RADII, margin_at_r=MARGIN,
        sigma_iso=SIG_ISO, sigma_aniso=[SIG_LAT, SIG_VERT], sigma_conservative=SIG_ISO_CONS,
        sigma_modeled=SIG_MODELED, pre_bake_bias_m=PRE_BAKE_BIAS,
        bake="post == vert_fix_bias_m 0 (deployed +L fix unbiased); pre == +0.25 (measured -0.25m gate-DOWN eps)",
        a_nmc=args.a_nmc, d_nmc=args.d_nmc, nboot=args.nboot, latency_ms=15.0,
        regime="COLD case-C (velocity IMU-only, position fixes only)",
        engine="margin_driver_v2 (regression-proven == ME.fly_lap in iso/zero-bias limit)",
    ), cells_A=prev.get("cells_A", []), cells_B=prev.get("cells_B", []),
        cells_C=prev.get("cells_C", []), cells_D=prev.get("cells_D", []), errors=prev.get("errors", []))
    if prev:
        out["meta"]["resumed_from_checkpoint"] = True
        out["meta"]["grids_run_this_pass"] = sorted(grids)

    with mp.Pool(args.procs) as pool:
        try:
            if "C" in grids:
                print(f"[rerun] grid C (baseline anchor iso0.265)...", flush=True)
                out["cells_C"] = run_grid_C(pool, args.a_nmc)
                c = out["cells_C"][0]
                print(f"  [C] iso0.265 post fr0.07 b0 v37: p90 {c['p90']:.3f} p99 {c['p99']:.3f} "
                      f"(banked-ref p90 0.210 p99 0.297 @ iso0.10; 0.265 should be HIGHER)", flush=True)
                _ckpt(out, out_path)

            if "D" in grids:
                print(f"[rerun] grid D (decision cells, nmc={args.d_nmc}, bootstrap CI)...", flush=True)
                out["cells_D"] = run_grid_D(pool, args.d_nmc, args.nboot)
                _ckpt(out, out_path)

            if "B" in grids:
                print(f"[rerun] grid B (pre-bake delta, nmc={args.a_nmc})...", flush=True)
                out["cells_B"] = run_grid_B(pool, args.a_nmc)
                _ckpt(out, out_path)

            if "A" in grids:
                print(f"[rerun] grid A (post-bake sweep, nmc={args.a_nmc})...", flush=True)
                out["cells_A"] = run_grid_A(pool, args.a_nmc)
                _ckpt(out, out_path)
        except Exception as e:  # noqa: BLE001
            out["errors"].append(dict(error=repr(e), trace=traceback.format_exc()))
            _ckpt(out, out_path)
            raise

    out["meta"]["wall_total_s"] = round(time.time() - t0, 1)
    _ckpt(out, out_path)
    print(f"\n[rerun] done in {out['meta']['wall_total_s']}s -> {out_path}", flush=True)
    print(f"[rerun] A={len(out['cells_A'])} B={len(out['cells_B'])} C={len(out['cells_C'])} "
          f"D={len(out['cells_D'])} cells", flush=True)


def _ckpt(out, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)


if __name__ == "__main__":
    main()
