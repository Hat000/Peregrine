"""COAST-DRIFT final boundary — the terminal miss is CENTERING-dominated; the coast is cheap.

Why A1 (not the MC) owns the centering boundary: fed the iid synthetic fix stream, the production KF
over-converges (sig_p -> ~1 mm) and rides the IMU, nearly ignoring band fixes -> the MC cannot transmit a
realistic per-fix/centering error into the coast-start state (sweep + realism showed miss is flat in
per-fix sigma, correlation tau, AND per-approach bias sigma_b). The MC's VALID, narrow finding is that the
COAST ITSELF adds negligible drift (velocity sig_v0*t_coast + process + accel-bias all <~0.02 m, robust
across speed/floor/bias). For the binding term — the gate-relative lateral CENTERING at the last accurate
fix (~12 m), which the coast preserves RIGIDLY — A1 propagates a REALISTIC coast-start (sig_p0, sig_v0)
through the production coast Q. sig_p0 = the measured per-fix lateral sigma partially averaged over the
band + the non-averaging per-approach bias (at-speed sigma_lat 0.13-0.17, ep_lat~0+-0.02; realistic
sig_p0 ~ 0.06-0.12 m). This file sweeps (sig_p0 x sig_v0) -> terminal miss -> closure vs the budget B,
splits VIO-immune (centering) vs VIO-addressable (velocity-drift), and reports the VIO trigger.
[COAST-DRIFT 2026-06-15]"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coast_drift as CD  # noqa: E402

OUT={}; T0=time.time()
B_GRID=[0.10,0.12,0.14,0.155,0.18,0.20,0.235,0.30]
t_coast=12.0/30.0


def term(sig_p0, sig_v0, v=30.0, r_floor=12.0, bias_deg=0.0):
    """A1 production coast: rigid-carry sig_p0 + velocity drift + process + bias, -> 2D miss quantiles."""
    a=CD.a1_coast_propagate(v, sig_v_lat=sig_v0, sig_p_lat=sig_p0, sig_v_vert=sig_v0, sig_p_vert=sig_p0,
                            accel_bias_deg=bias_deg, bias_mode="inplane", r_floor=r_floor)
    return a

# F1 — CENTERING x VELOCITY -> terminal miss + the coast's marginal cost
print("="*84)
print("F1. terminal miss p99 vs (sig_p0 centering, sig_v0 velocity) [A1 production coast, v30 r12 b0]")
print("    'coast cost' = p99(sig_v0) - p99(sig_v0=0) at fixed sig_p0 (what the coast ADDS)")
print(f"  {'sig_p0':>7} | " + "  ".join(f"sv={sv:.2f}" for sv in (0.01,0.05,0.10,0.20)) + "   | coast_cost(sv.20)")
f1=[]
for sp in (0.05,0.06,0.08,0.10,0.12,0.15,0.20):
    row={}
    for sv in (0.01,0.05,0.10,0.20):
        row[sv]=term(sp,sv)["miss_p99"]
    cost=row[0.20]-row[0.01]
    f1.append(dict(sig_p0=sp, p99=row, coast_cost=cost))
    print(f"  {sp:7.2f} | " + "  ".join(f"{row[sv]:6.3f}" for sv in (0.01,0.05,0.10,0.20))
          + f"   | {cost:+.4f}")
OUT["centering_x_velocity"]=f1

# F2 — GATE-DIFFICULTY: smallest budget B that closes vs centering sig_p0 (velocity fixed at the achieved
#      sig_v0~0.02; bias 0.3 deg). The verdict is GENERAL terminal-approach physics, not gate-4.
print("="*84)
print("F2. GATE-DIFFICULTY boundary: closure (p99<B) vs centering sig_p0 (sig_v0=0.02, bias0.3, v30 r12)")
print("    B = in-plane budget; 0.235=gate-4 r0.30, 0.155=gate-4 r0.38, <=0.14 = tight VQ2-like gates")
print("   closure grid:  B = " + "  ".join(f"{B:.3f}" for B in B_GRID))
f2=[]
for sp in (0.04,0.06,0.08,0.10,0.12,0.15,0.20):
    a=term(sp,0.02,bias_deg=0.3); p99=a["miss_p99"]; p90=a["miss_p90"]
    Bmin=next((B for B in B_GRID if p99<B), None)
    f2.append(dict(sig_p0=sp,p90=p90,p99=p99,Bmin=Bmin,closeB={f"{B:.3f}":bool(p99<B) for B in B_GRID}))
    print(f"   sig_p0={sp:.2f} (p99={p99:.3f}) " + "  ".join(("  Y  " if p99<B else "  n  ") for B in B_GRID)
          + f"  -> closes B>={Bmin}")
OUT["gate_difficulty"]=f2

# F3 — VIO TRIGGER (parked #70). VIO observes velocity -> shrinks sig_v0 (already tiny); it is BLIND to
#      the gate-relative centering sig_p0 (odometry, not a gate anchor). So:
#       - centering-only p99 (sig_v0=0): the VIO-IMMUNE floor. If it already breaches B, VIO cannot close.
#       - the velocity-drift term sig_v0*t_coast that would breach B = the VIO trigger.
print("="*84)
print("F3. VIO-TRIGGER (parked #70): velocity-drift trigger vs the VIO-immune centering floor")
print(f"  IMU-only achieved sig_v0 ~ 0.012-0.024 m/s -> coast velocity-drift ~ {0.018*t_coast:.3f} m (tiny)")
print(f"  {'B':>6} {'cen-floor p99 (sig_p0=0.10)':>26} {'VIO-immune<B?':>13} {'sig_v0 trigger (m/s)':>20}")
f3=[]
for B in (0.120,0.155,0.235):
    cen=term(0.10,0.0)["miss_p99"]            # centering-only floor at a representative sig_p0=0.10
    immune_ok = cen < B
    # sig_v0 at which velocity-drift alone (sig_p0 negligible) breaches B:
    trig=None
    for sv in np.linspace(0,0.8,801):
        if term(0.02,sv)["miss_p99"] < B: trig=float(sv)
        else: break
    f3.append(dict(B=B,cen_floor_p99=cen,vio_immune_ok=bool(immune_ok),sigv_trigger=trig))
    print(f"  {B:6.3f} {cen:26.3f} {str(immune_ok):>13} {('%.3f'%trig) if trig else 'n/a':>20}")
OUT["vio_trigger"]=dict(t_coast=t_coast, imu_sig_v0=[0.012,0.024], rows=f3)

OUT["meta"]=dict(seed=CD.SEED, wall_s=round(time.time()-T0,1), note="A1 production-Q coast; centering=rigid carry")
(Path(__file__).resolve().parent/"final_results.json").write_text(json.dumps(OUT,indent=1))
print("="*84); print(f"DONE {OUT['meta']['wall_s']}s -> final_results.json  M(0.30)={CD.M(0.30):.3f} M(0.38)={CD.M(0.38):.3f}")
