"""COAST-DRIFT comprehensive sweep + regression + boundary. Writes results.json + prints tables.

Sections:
  R   REGRESSION — my coast engine vs the boresight engine (MD.run_cell_v2) under boresight's own
      conditions (sparse fr=0.07, sigma=0.19, fix-to-gate): must reproduce NO-CLOSE, proving the engine
      is not accidentally optimistic.
  C   CORRELATION (the binding realism lever) — AR(1) fix-error correlation tau_c sweep: achieved
      sigma_p0 / sigma_v0 at the r_floor coast-start + terminal miss. (iid averages out; correlation
      cannot -> terminal POSITION centering is the real binding factor, coasted rigidly.)
  S   PER-FIX SIGMA — at-speed sigma_z {0.10,0.15,0.20,0.25} x correlation.
  B   A1 BOUNDARY — transparent: terminal random miss vs the GIVEN sigma_v_lat (the key uncertain input).
  M   MAIN CLOSURE SWEEP — speed {22,30,37} x r_floor {10,12,14} x accel-bias {0,0.3,0.6 deg} x
      sigma_z {0.10,0.15} x tau_c {0,0.1} with bootstrap CIs vs MARGIN(r).
[COAST-DRIFT 2026-06-15]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coast_drift as CD  # noqa: E402
import margin_envelope as ME  # noqa: E402
import margin_driver_v2 as MD  # noqa: E402

OUT = {}
T0 = time.time()


def mc(v, sg_lat, sg_vert, bias_deg, bias_mode, fr, r_floor, tau=0.0, lat_bias=0.0, n_mc=600,
       n_boot=1000):
    bmag = ME.att_deg_to_accel_bias(bias_deg)
    miss = np.empty(n_mc); ml = np.empty(n_mc); mv = np.empty(n_mc)
    pl=[]; vl=[]; pv=[]; rr=[]
    for s in range(n_mc):
        seed = (CD.SEED + 101*s + int(round(v))*13 + int(round(bias_deg*100))*7
                + int(round(sg_lat*1000))*17 + int(round(fr*1000))*23 + int(round(r_floor*100))*131
                + int(round(tau*1000))*211 + int(round(lat_bias*1000))*311
                + {"random3d":0,"inplane":1}[bias_mode]*3001)
        rng = np.random.default_rng(seed)
        m, d = CD.fly_lap_coast(rng, v, sg_lat, sg_vert, bmag, bias_mode, fr, r_floor,
                                lat_fix_bias_m=lat_bias, fix_corr_tau=tau, diag=True)
        miss[s]=m; ml[s]=d["miss_lat"]; mv[s]=d["miss_vert"]
        if "perr_lat" in d:
            pl.append(d["perr_lat"]); vl.append(d["verr_lat"]); pv.append(d["perr_vert"]); rr.append(d["range_m"])
    def boot(q):
        rng=np.random.default_rng(CD.SEED); bq=np.empty(n_boot)
        for b in range(n_boot): bq[b]=np.percentile(miss[rng.integers(0,n_mc,n_mc)],q)
        return float(np.percentile(bq,5)), float(np.percentile(bq,95))
    p90=float(np.percentile(miss,90)); p99=float(np.percentile(miss,99))
    p90c=boot(90); p99c=boot(99)
    return dict(v=v, sg_lat=sg_lat, sg_vert=sg_vert, bias_deg=bias_deg, bias_mode=bias_mode, fr=fr,
        r_floor=r_floor, tau=tau, lat_bias=lat_bias, n_mc=n_mc,
        p50=float(np.percentile(miss,50)), p90=p90, p99=p99, p90_ci=p90c, p99_ci=p99c,
        rms=float(np.sqrt(np.mean(miss**2))), maxm=float(miss.max()),
        miss_lat_mean=float(np.mean(ml)), miss_lat_std=float(np.std(ml)),
        miss_vert_mean=float(np.mean(mv)), miss_vert_std=float(np.std(mv)),
        sig_p0=float(np.std(pl)) if pl else None, sig_v0=float(np.std(vl)) if vl else None,
        verr0_mean=float(np.mean(vl)) if vl else None,
        coast_start_r=float(np.mean(rr)) if rr else None,
        clears={f"{r:.2f}":bool(p90<CD.M(r) and p99<CD.M(r)) for r in ME.RADIUS_BAND},
        clears_cihi={f"{r:.2f}":bool(p90c[1]<CD.M(r) and p99c[1]<CD.M(r)) for r in ME.RADIUS_BAND})


# ---------- R: REGRESSION vs boresight engine ----------
print("="*70); print("R. REGRESSION — engine sanity vs boresight (must NOT be optimistic)")
reg = {}
# boresight cell: fr=0.07, aniso sigma [0.191,0.101], cold, fix-to-gate (the OLD 12 m window)
bc = MD.run_cell_v2(30.0, 0.191, 0.101, 0.0, "random3d", 0.07, "cold", n_mc=600)
print(f"  boresight engine (MD) v30 fr0.07 sig[0.191,0.101] b0 fix-to-gate: "
      f"p90={bc['inplane_p90']:.3f} p99={bc['inplane_p99']:.3f} clear0.30={bc['clears_p90_and_p99']['0.30']}")
# my coast engine, SAME sparse conditions but r_floor=12 coast (the reframe): the IMPROVEMENT
mine_sparse = mc(30.0, 0.191, 0.101, 0.0, "random3d", 0.07, 12.0, n_mc=600)
print(f"  my coast  (sparse fr0.07 sig[0.191,0.101] r_floor12 coast): "
      f"p90={mine_sparse['p90']:.3f} p99={mine_sparse['p99']:.3f} clear0.30={mine_sparse['clears']['0.30']}")
# my coast engine, pointed conditions
mine_pointed = mc(30.0, 0.10, 0.10, 0.0, "inplane", 0.60, 12.0, n_mc=600)
print(f"  my coast  (POINTED fr0.60 sig[0.10,0.10] r_floor12 coast): "
      f"p90={mine_pointed['p90']:.3f} p99={mine_pointed['p99']:.3f} clear0.30={mine_pointed['clears']['0.30']}")
OUT["regression"]=dict(boresight_engine=bc, mine_sparse=mine_sparse, mine_pointed=mine_pointed)

# ---------- C: CORRELATION sweep ----------
print("="*70); print("C. CORRELATION (AR1 tau_c) — the binding realism lever (v30 sig0.15 fr0.6 r12 b0)")
print(f"  {'tau_c':>6} {'sig_p0':>7} {'sig_v0':>7} {'miss_lat_std':>12} {'p90':>6} {'p99':>6} {'clr0.30':>7} {'clr0.38':>7}")
corr=[]
for tau in (0.0, 0.05, 0.10, 0.20, 0.50, 1.0):
    r = mc(30.0, 0.15, 0.10, 0.0, "inplane", 0.60, 12.0, tau=tau, n_mc=600)
    corr.append(r)
    print(f"  {tau:6.2f} {r['sig_p0']:7.4f} {r['sig_v0']:7.4f} {r['miss_lat_std']:12.4f} "
          f"{r['p90']:6.3f} {r['p99']:6.3f} {str(r['clears']['0.30']):>7} {str(r['clears']['0.38']):>7}")
OUT["correlation"]=corr

# ---------- S: per-fix sigma x correlation ----------
print("="*70); print("S. PER-FIX SIGMA x correlation (v30 fr0.6 r12 b0)")
print(f"  {'sig_z':>6} {'tau':>5} {'sig_p0':>7} {'p90':>6} {'p99':>6} {'p99_cihi':>8} {'clr0.30':>7} {'clr0.38':>7}")
sigsweep=[]
for sg in (0.10, 0.15, 0.20, 0.25):
    for tau in (0.0, 0.10, 0.30):
        r = mc(30.0, sg, 0.10, 0.0, "inplane", 0.60, 12.0, tau=tau, n_mc=600)
        sigsweep.append(r)
        print(f"  {sg:6.2f} {tau:5.2f} {r['sig_p0']:7.4f} {r['p90']:6.3f} {r['p99']:6.3f} "
              f"{r['p99_ci'][1]:8.3f} {str(r['clears']['0.30']):>7} {str(r['clears']['0.38']):>7}")
OUT["sigma_sweep"]=sigsweep

# ---------- B: A1 transparent sigma_v boundary ----------
print("="*70); print("B. A1 BOUNDARY — terminal random miss vs GIVEN sigma_v_lat (v30 r12, sig_p0=0.10)")
print(f"  {'sig_v_in':>9} {'sig_lat_term':>12} {'miss_p90':>9} {'miss_p99':>9} {'clr0.30(0.235)':>14}")
bnd=[]
for sv in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    a = CD.a1_coast_propagate(30.0, sig_v_lat=sv, sig_p_lat=0.10, sig_v_vert=sv, accel_bias_deg=0.0, r_floor=12.0)
    bnd.append(a)
    print(f"  {sv:9.2f} {a['sig_lat_terminal']:12.4f} {a['miss_p90']:9.4f} {a['miss_p99']:9.4f} "
          f"{str(a['miss_p99']<0.235):>14}")
OUT["a1_boundary"]=bnd

# ---------- M: MAIN CLOSURE SWEEP ----------
print("="*70); print("M. MAIN CLOSURE SWEEP (bootstrap-CI-honest)")
print(f"  {'v':>4} {'r_fl':>5} {'bias':>5} {'sig':>5} {'tau':>5} {'t_coast':>7} {'p90':>6} {'p99':>6} "
      f"{'p99cihi':>8} {'clr.30':>7} {'clr.30CI':>9} {'clr.38':>7}")
main=[]
for v in (22.0, 30.0, 37.0):
    for r_floor in (10.0, 12.0, 14.0):
        for bias in (0.0, 0.3, 0.6):
            for sg in (0.10, 0.15):
                for tau in (0.0, 0.10):
                    r = mc(v, sg, 0.10, bias, "inplane", 0.60, r_floor, tau=tau, n_mc=600)
                    main.append(r)
                    tc = r_floor/v
                    print(f"  {v:4.0f} {r_floor:5.1f} {bias:5.2f} {sg:5.2f} {tau:5.2f} {tc:7.3f} "
                          f"{r['p90']:6.3f} {r['p99']:6.3f} {r['p99_ci'][1]:8.3f} "
                          f"{str(r['clears']['0.30']):>7} {str(r['clears_cihi']['0.30']):>9} "
                          f"{str(r['clears']['0.38']):>7}")
OUT["main_sweep"]=main

OUT["meta"]=dict(seed=CD.SEED, M_030=CD.M(0.30), M_038=CD.M(0.38), budget_prompt=CD.BUDGET_PROMPT,
                 sigma_pointed=CD.SIGMA_LAT_POINTED, r_acc_max=CD.R_ACC_MAX, wall_s=round(time.time()-T0,1))
(Path(__file__).resolve().parent/"results.json").write_text(json.dumps(OUT, indent=1))
print("="*70); print(f"DONE in {OUT['meta']['wall_s']}s -> results.json  M(0.30)={CD.M(0.30):.3f} M(0.38)={CD.M(0.38):.3f}")
