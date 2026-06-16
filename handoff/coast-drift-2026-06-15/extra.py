"""COAST-DRIFT additions (commander, 2026-06-15):
  P1  GATE-DIFFICULTY SWEEP — closure boundary vs the in-plane BUDGET B itself (0.10-0.30 m), spanning
      VQ1 gate-4 (loose) through tighter VQ2-like gates. The verdict is the GENERAL terminal-approach
      physics, not gate-4's specific margin.
  P2  VIO-TRIGGER BACK-SOLVE — decompose the terminal miss into (a) POSITION-CENTERING (set by the last
      gate fix; VIO-IMMUNE — odometry is gate-relative-blind) and (b) VELOCITY-DRIFT σ_v0·t_coast
      (VIO-ADDRESSABLE — VINS observes velocity directly). For each budget B back-solve the lateral
      coast-drift bound a backbone upgrade would need to hit to close — the trigger spec for parked #70,
      WITHOUT building VIO. Also flags when the binding term is centering (=> the lever is a near-field
      gate estimator / better per-fix sigma, NOT VIO).
[COAST-DRIFT 2026-06-15]"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coast_drift as CD  # noqa: E402
import margin_envelope as ME  # noqa: E402

OUT = {}; T0 = time.time()


def mc_full(v, sg_lat, sg_vert, bias_deg, bias_mode, fr, r_floor, tau=0.0, n_mc=1500):
    """MC -> full miss array + coast-start (perr0, verr0) lateral arrays for the decomposition."""
    bmag = ME.att_deg_to_accel_bias(bias_deg)
    miss = np.empty(n_mc); ml = np.empty(n_mc)
    pl=[]; vl=[]
    for s in range(n_mc):
        seed = (CD.SEED + 101*s + int(round(v))*13 + int(round(bias_deg*100))*7
                + int(round(sg_lat*1000))*17 + int(round(fr*1000))*23 + int(round(r_floor*100))*131
                + int(round(tau*1000))*211 + {"random3d":0,"inplane":1}[bias_mode]*3001)
        rng = np.random.default_rng(seed)
        m, d = CD.fly_lap_coast(rng, v, sg_lat, sg_vert, bmag, bias_mode, fr, r_floor,
                                fix_corr_tau=tau, diag=True)
        miss[s]=m; ml[s]=d["miss_lat"]
        if "perr_lat" in d: pl.append(d["perr_lat"]); vl.append(d["verr_lat"])
    return miss, np.asarray(ml), np.asarray(pl), np.asarray(vl)


# ============================================================================================
# P1 — GATE-DIFFICULTY SWEEP: closure vs budget B for representative operating points
# ============================================================================================
print("="*78); print("P1. GATE-DIFFICULTY SWEEP — closure vs in-plane budget B (v=30, r_floor=12)")
OPS = [
    ("pointed-clean",     dict(sg_lat=0.10, sg_vert=0.10, bias_deg=0.0, fr=0.60, tau=0.0)),
    ("pointed-realistic", dict(sg_lat=0.15, sg_vert=0.10, bias_deg=0.3, fr=0.60, tau=0.10)),
    ("degraded",          dict(sg_lat=0.20, sg_vert=0.10, bias_deg=0.6, fr=0.60, tau=0.30)),
]
B_GRID = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.235, 0.25, 0.30]
p1 = {}
for name, kw in OPS:
    miss, ml, pl, vl = mc_full(30.0, kw["sg_lat"], kw["sg_vert"], kw["bias_deg"], "inplane",
                               kw["fr"], 12.0, tau=kw["tau"], n_mc=1500)
    p90 = float(np.percentile(miss,90)); p99 = float(np.percentile(miss,99))
    # bootstrap p99 hi
    rng=np.random.default_rng(CD.SEED); bq=np.empty(1500)
    for b in range(1500): bq[b]=np.percentile(miss[rng.integers(0,len(miss),len(miss))],99)
    p99hi=float(np.percentile(bq,95))
    closeB = {f"{B:.3f}": bool(p99 < B) for B in B_GRID}
    closeB_ci = {f"{B:.3f}": bool(p99hi < B) for B in B_GRID}
    # smallest B that closes (p99 and p99hi)
    Bmin = next((B for B in B_GRID if p99 < B), None)
    Bmin_ci = next((B for B in B_GRID if p99hi < B), None)
    p1[name] = dict(p50=float(np.percentile(miss,50)), p90=p90, p99=p99, p99_cihi=p99hi,
                    sig_p0=float(np.std(pl)), sig_v0=float(np.std(vl)),
                    Bmin_close=Bmin, Bmin_close_cihonest=Bmin_ci, closeB=closeB, closeB_ci=closeB_ci)
    print(f"  {name:18s}: p90={p90:.3f} p99={p99:.3f}(cihi {p99hi:.3f}) sig_p0={np.std(pl):.3f} "
          f"sig_v0={np.std(vl):.3f} -> closes B>= {Bmin} (CI-honest {Bmin_ci})")
print("   closure grid (p99<B): B = " + "  ".join(f"{B:.3f}" for B in B_GRID))
for name,_ in OPS:
    print(f"   {name:18s} " + "  ".join(("  Y  " if p1[name]['closeB'][f'{B:.3f}'] else "  n  ") for B in B_GRID))
OUT["gate_difficulty"]=p1

# ============================================================================================
# P2 — VIO-TRIGGER BACK-SOLVE: decompose centering (VIO-immune) vs velocity-drift (VIO-addressable),
#       back-solve the coast-drift bound a backbone must hit to close r=0.30 at each budget B.
# ============================================================================================
print("="*78); print("P2. VIO-TRIGGER BACK-SOLVE (v=30, r_floor=12, t_coast=0.404)")
t_coast = 12.0/30.0
# Use the realistic operating point's achieved coast-start sigmas to split the terminal miss.
miss, ml, pl, vl = mc_full(30.0, 0.15, 0.10, 0.3, "inplane", 0.60, 12.0, tau=0.10, n_mc=1500)
sig_p0 = float(np.std(pl)); sig_v0 = float(np.std(vl)); off = float(np.mean(ml))
# A1-style decomposition via the production coast: centering-only (sig_v0=0) vs velocity-only (sig_p0=0)
cen = CD.a1_coast_propagate(30.0, sig_v_lat=0.0, sig_p_lat=sig_p0, sig_v_vert=0.0, sig_p_vert=sig_p0,
                            accel_bias_deg=0.0, r_floor=12.0)
vel = CD.a1_coast_propagate(30.0, sig_v_lat=sig_v0, sig_p_lat=0.0, sig_v_vert=sig_v0, sig_p_vert=0.0,
                            accel_bias_deg=0.0, r_floor=12.0)
print(f"  ACHIEVED (IMU-only, realistic): sig_p0(centering)={sig_p0:.4f}  sig_v0(velocity)={sig_v0:.4f}"
      f"  det-offset(artifact)={off:+.4f}")
print(f"  centering-only terminal lateral sigma = {cen['sig_lat_terminal']:.4f}  (VIO-IMMUNE: gate-relative)")
print(f"  velocity-only terminal lateral sigma  = {vel['sig_lat_terminal']:.4f}  = sig_v0*t_coast (VIO-ADDRESSABLE)")
# For each budget B: the max velocity-drift term D (1-sigma lateral) s.t. p99(2D miss) < B, given the
# VIO-immune centering floor. We approximate the 2D-magnitude p99 from per-axis sigma via sampling.
def miss_p99_from(sig_lat, sig_vert, mean_off=0.0):
    rng=np.random.default_rng(7)
    lat=rng.normal(mean_off, sig_lat, 200000); vert=rng.normal(0.0, sig_vert, 200000)
    return float(np.percentile(np.sqrt(lat**2+vert**2),99))
cen_floor_lat = cen['sig_lat_terminal']; cen_floor_vert = cen['sig_vert_terminal']
proc_floor = 0.0094  # process-noise lateral sigma (already inside cen via production Q, kept for note)
print(f"\n  {'B(budget)':>9} {'IMU p99':>8} {'closes?':>7} {'cen-floor p99':>13} {'VIO need D_lat(1sig)':>20} {'VINS drift/12m':>15}")
backsolve=[]
imu_p99 = float(np.percentile(miss,99))
for B in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.235]:
    closes_imu = imu_p99 < B
    cen_p99 = miss_p99_from(cen_floor_lat, cen_floor_vert, mean_off=off)
    # back-solve D_lat: find velocity-drift 1-sigma s.t. p99(sqrt((cen_lat^2+D^2)) , cen_vert) < B
    D_need = None
    for D in np.linspace(0.0, 0.6, 601):
        if miss_p99_from(np.sqrt(cen_floor_lat**2 + D**2), cen_floor_vert, mean_off=off) < B:
            D_need = float(D)
        else:
            break
    feasible = (cen_p99 < B)   # is the VIO-immune centering floor already under B?
    drift_pct = (D_need/12.0*100.0) if (D_need is not None) else None
    backsolve.append(dict(B=B, imu_p99=imu_p99, closes_imu=bool(closes_imu), cen_floor_p99=cen_p99,
                          centering_feasible=bool(feasible), D_lat_need=D_need, vins_drift_pct_12m=drift_pct))
    dstr = f"{D_need:.3f}" if D_need is not None else "  -inf(centering>B)"
    pstr = f"{drift_pct:.2f}%" if drift_pct is not None else "  n/a"
    print(f"  {B:9.3f} {imu_p99:8.3f} {str(closes_imu):>7} {cen_p99:13.3f} {dstr:>20} {pstr:>15}")
OUT["vio_backsolve"]=dict(t_coast=t_coast, sig_p0=sig_p0, sig_v0=sig_v0, det_offset=off,
                          centering_terminal_lat=cen_floor_lat, centering_terminal_vert=cen_floor_vert,
                          velocity_terminal_lat=vel['sig_lat_terminal'], imu_p99=imu_p99,
                          rows=backsolve)
OUT["meta"]=dict(seed=CD.SEED, t_coast=t_coast, wall_s=round(time.time()-T0,1))
(Path(__file__).resolve().parent/"extra_results.json").write_text(json.dumps(OUT, indent=1))
print("="*78); print(f"DONE in {OUT['meta']['wall_s']}s -> extra_results.json")
