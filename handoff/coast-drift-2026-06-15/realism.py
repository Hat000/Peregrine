"""COAST-DRIFT realism overlay — the BINDING term is per-approach gate-relative lateral centering bias.

WHY: with iid fix noise + whole-lap conditioning the production KF over-converges (sig_p ~1 mm) and rides
the IMU, so per-fix noise/AR(1)-correlation barely move the coast-start position (sweep/extra showed
sig_p0~0.001 even at tau=0.30). The PHYSICALLY-REAL terminal-centering error is the per-approach
gate-relative lateral bias that does NOT average out — a fix that places the drone systematically off in
the gate frame for THIS approach (at-speed ep_lat ~0 +-0.02 m; accept-geometry lateral median +0.02..+0.09
m). It is drawn ONCE per approach (constant across the band's fixes), so the KF converges to truth+b and
COASTS it rigidly -> terminal miss inherits b. This is VIO-IMMUNE (odometry is gate-relative-blind). This
overlay sweeps sigma_b (per-approach centering 1-sigma) — the true binding lever — vs the budget B and
folds it into the VIO trigger.  [COAST-DRIFT 2026-06-15]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coast_drift as CD  # noqa: E402
import margin_envelope as ME  # noqa: E402

OUT={}; T0=time.time()
B_GRID=[0.10,0.12,0.14,0.16,0.18,0.20,0.235,0.25,0.30]


def mc_sigb(v, sg_lat, sg_vert, bias_deg, fr, r_floor, sigma_b, tau=0.10, n_mc=1500):
    """MC with a per-lap gate-relative lateral centering bias ~ N(0, sigma_b) (constant across the
    approach's fixes -> non-averaging -> coasts rigidly)."""
    bmag = ME.att_deg_to_accel_bias(bias_deg)
    miss=np.empty(n_mc); ml=np.empty(n_mc)
    for s in range(n_mc):
        seed=(CD.SEED+101*s+int(round(v))*13+int(round(bias_deg*100))*7+int(round(sg_lat*1000))*17
              +int(round(fr*1000))*23+int(round(r_floor*100))*131+int(round(tau*1000))*211
              +int(round(sigma_b*1000))*409+5001)
        rng=np.random.default_rng(seed)
        b_approach=float(rng.normal(0.0, sigma_b)) if sigma_b>0 else 0.0
        m,d=CD.fly_lap_coast(rng, v, sg_lat, sg_vert, bmag, "inplane", fr, r_floor,
                             lat_fix_bias_m=b_approach, fix_corr_tau=tau, diag=True)
        miss[s]=m; ml[s]=d["miss_lat"]
    p90=float(np.percentile(miss,90)); p99=float(np.percentile(miss,99))
    rng=np.random.default_rng(CD.SEED); bq=np.empty(1200)
    for b in range(1200): bq[b]=np.percentile(miss[rng.integers(0,n_mc,n_mc)],99)
    p99hi=float(np.percentile(bq,95))
    Bmin=next((B for B in B_GRID if p99<B), None)
    Bmin_ci=next((B for B in B_GRID if p99hi<B), None)
    return dict(sigma_b=sigma_b, v=v, sg_lat=sg_lat, bias_deg=bias_deg, r_floor=r_floor, tau=tau,
        p50=float(np.percentile(miss,50)), p90=p90, p99=p99, p99_cihi=p99hi,
        miss_lat_mean=float(np.mean(ml)), miss_lat_std=float(np.std(ml)),
        Bmin_close=Bmin, Bmin_close_cihonest=Bmin_ci,
        closeB={f"{B:.3f}":bool(p99<B) for B in B_GRID},
        closeB_ci={f"{B:.3f}":bool(p99hi<B) for B in B_GRID})


print("="*82)
print("R1. CENTERING-BIAS x GATE-DIFFICULTY (v30 r12 sg_lat0.15 fr0.6 bias0.3 tau0.10)")
print("    sigma_b = per-approach gate-relative lateral centering 1-sigma (the VIO-immune binding term)")
print(f"  {'sig_b':>6} {'miss_std':>8} {'p90':>6} {'p99':>6} {'p99cihi':>8} {'closes B>=':>11} {'(CI-honest)':>11}")
rows=[]
for sb in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
    r=mc_sigb(30.0, 0.15, 0.10, 0.3, 0.60, 12.0, sb)
    rows.append(r)
    print(f"  {sb:6.2f} {r['miss_lat_std']:8.4f} {r['p90']:6.3f} {r['p99']:6.3f} {r['p99_cihi']:8.3f}"
          f" {str(r['Bmin_close']):>11} {str(r['Bmin_close_cihonest']):>11}")
OUT["centering_vs_difficulty"]=rows

print("\n   closure grid (p99<B):  B = " + "  ".join(f"{B:.3f}" for B in B_GRID))
for r in rows:
    print(f"   sig_b={r['sigma_b']:.2f}  " + "  ".join(("  Y  " if r['closeB'][f'{B:.3f}'] else "  n  ") for B in B_GRID))

# R2 — gate-4 (B=0.235) and a tight VQ2 gate (B=0.155=r0.38; B=0.12) vs centering bias + speed
print("="*82)
print("R2. SPEED x CENTERING at fixed gates  (r_floor=12, sg_lat0.15 fr0.6 bias0.3 tau0.10)")
print(f"  {'v':>4} {'sig_b':>6} {'t_coast':>7} {'p99':>6} {'clr B0.235':>10} {'clr B0.155':>10} {'clr B0.120':>10}")
r2=[]
for v in (22.0, 30.0, 37.0):
    for sb in (0.04, 0.08):
        r=mc_sigb(v, 0.15, 0.10, 0.3, 0.60, 12.0, sb)
        r2.append(r)
        print(f"  {v:4.0f} {sb:6.2f} {12.0/v:7.3f} {r['p99']:6.3f} "
              f"{str(r['p99']<0.235):>10} {str(r['p99']<0.155):>10} {str(r['p99']<0.120):>10}")
OUT["speed_x_centering"]=r2

# R3 — VIO trigger restated with the centering floor. VIO addresses ONLY velocity-drift; centering is
# immune. Trigger = the terminal velocity 1-sigma at which sig_v0*t_coast would itself breach the budget.
print("="*82); print("R3. VIO-TRIGGER (parked #70) — restated against the binding term")
t_coast=12.0/30.0
# velocity-drift term that would breach budget B (worst case: all on lateral, centering negligible)
print(f"  budget B   max coast velocity-drift D=sig_v0*t_coast (p99<B)   => VIO trigger sig_v0 (m/s)")
trig=[]
def p99_2d(sl, sv=0.01, off=0.056):
    rng=np.random.default_rng(7); lat=rng.normal(off,sl,200000); vert=rng.normal(0,sv,200000)
    return float(np.percentile(np.sqrt(lat**2+vert**2),99))
for B in (0.120, 0.155, 0.235):
    # solve D s.t. p99(2D, lateral sigma=D, vert small, plus the deterministic offset) < B
    D=None
    for d in np.linspace(0,0.6,601):
        if p99_2d(d) < B: D=float(d)
        else: break
    sigv_trig=(D/t_coast) if D is not None else None
    trig.append(dict(B=B, D_max=D, sigv_trigger=sigv_trig))
    print(f"   B={B:.3f}   D_max={D:.3f} m  ({'%.3f'%(D/12*100)}%/12m)   => sig_v0 >= "
          f"{('%.3f'%sigv_trig) if sigv_trig else 'n/a'} m/s triggers VIO")
print(f"   IMU-only ACHIEVED sig_v0 ~ 0.012-0.024 m/s (this analysis) -> coast velocity-drift ~0.005-0.010 m")
OUT["vio_trigger_restated"]=dict(t_coast=t_coast, imu_sig_v0_range=[0.012,0.024], rows=trig)

OUT["meta"]=dict(seed=CD.SEED, wall_s=round(time.time()-T0,1))
(Path(__file__).resolve().parent/"realism_results.json").write_text(json.dumps(OUT,indent=1))
print("="*82); print(f"DONE in {OUT['meta']['wall_s']}s -> realism_results.json")
