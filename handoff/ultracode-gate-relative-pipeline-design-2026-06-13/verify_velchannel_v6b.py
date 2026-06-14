"""Follow-up probes after v6:
  (P1) What smoothing window would the honest PnP-delta need to reach the design's assumed
       sigma_v in [0.3,1.0] m/s? And does the smoothing TIME fit inside the 0.66 s transit?
  (P2) Sanity: re-run the design's OWN visvel path at L=0 but with the HONEST cov for the
       sigma_v it claims (i.e. keep zv=v0+noise GT-cheat but check that even granting the cheat,
       the p90 is over margin). Already have: visvel_1.0 p90=0.224 (V2). Confirm against margin.
  (P3) Re-derive the LSQ in-plane *position* consequence directly: warm vel sigma -> in-plane pos.
       Cross-check the part_c warm 0.119 m against a pure position-fix-only KF (no vel cheat).
  (P4) Does the design's d4v sim ever exercise latency? (it calls update_position in-sequence).
       Confirm the d4v headline numbers are an L=0 measurement, so the RewindKF-default mandate
       is UNTESTED by the artifact.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))
from racer.frames import R_world_from_body
from racer.state_estimator import LinearKF, GRAVITY_NED
from kf_rewind_buffer import RewindKF

V_RACE=37.0; G=9.80665
G3=np.array([-111.5,-5.1,24.57]); G4=np.array([-135.5,-0.8,25.36])
SEG=G4-G3; SEG_LEN=float(np.linalg.norm(SEG)); SEG_HAT=SEG/SEG_LEN; T_TRANSIT=SEG_LEN/V_RACE
PER_AXIS_LAT_SIGMA=0.265; ABS_AXIS_SIGMA=0.50
MARGIN=0.155

print("="*80); print("P1  honest PnP-delta: smoothing window needed for sigma_v in [0.3,1.0]")
print("="*80)
# sigma_v(delta over window W_s) = sqrt(2)*sigma_p / W_s   (two endpoint positions)
# OR, if we LSQ-fit the slope over n frames in window W_s: sigma_v = sigma_p*sqrt(12/(n(n^2-1)))/dt_f
dt_f=1/30.0
print("  two-endpoint delta:  sigma_v = sqrt(2)*sigma_p / W_window")
for W in [0.1,0.2,0.3,0.5,0.66]:
    sv=np.sqrt(2)*PER_AXIS_LAT_SIGMA/W
    print(f"    W={W:.2f}s ({W/dt_f:.0f} frames): sigma_v_lat={sv:.2f} m/s")
print("  LSQ slope over the window (better):  sigma_v = sigma_p*sqrt(12/(n(n^2-1)))/dt_frame")
for W in [0.1,0.2,0.3,0.5,0.66]:
    n=max(2,int(30.0*W)); dt=W/(n-1)
    sv=PER_AXIS_LAT_SIGMA*np.sqrt(12/(n*(n**2-1)))/dt
    print(f"    W={W:.2f}s (n={n}): sigma_v_lat={sv:.2f} m/s")
print("  NOTE: an LSQ vel slope over window W IS the position-fix-differencing the KF already does.")
print("        A SEPARATE vision-vel channel that LSQ-fits the same fixes is NOT independent info.\n")

print("="*80); print("P3  warm-vel position consequence cross-check (position-fix-only, no vel cheat)")
print("="*80)
def kf_posonly(n_mc=600, warm=True):
    yaw=float(np.arctan2(SEG_HAT[1],SEG_HAT[0])); pitch=float(-np.arctan(0.21*V_RACE/G))
    R_wb=R_world_from_body(0.0,pitch,yaw); accel_nom=R_wb.T@(-GRAVITY_NED)
    a_ph=G*np.sin(np.deg2rad(0.5)); ab=accel_nom+R_wb.T@np.array([0.0,a_ph,0.0])
    p0=G3.copy(); v0=SEG_HAT*V_RACE; n_steps=int(T_TRANSIT/(1/90.0)); ip=[]
    for s in range(n_mc):
        r=np.random.default_rng(20260613+13*s+111)
        if warm: v_init=v0.copy(); vstd=0.15
        else: v_init=v0+np.array([0.0,1.5,0.0]); vstd=1.5
        kf=LinearKF.initialize(p0+r.normal(0,0.3,3),v_init,pos_std=0.5,vel_std=vstd,accel_noise_std=0.3)
        rk=RewindKF(kf=kf,horizon_s=0.5); t=0.0;t_ns=0;next_fix=0.0
        for k in range(n_steps):
            t+=1/90.0; t_ns+=int(1/90.0*1e9); rk.predict(ab+r.normal(0,0.3,3),R_wb,1/90.0,t_ns)
            p_true=p0+SEG_HAT*V_RACE*t
            if t>=next_fix and np.linalg.norm(G4-p_true)<12.0:
                next_fix+=1/14.0
                z=p_true+np.array([r.normal(0,ABS_AXIS_SIGMA),r.normal(0,PER_AXIS_LAT_SIGMA),r.normal(0,PER_AXIS_LAT_SIGMA)])
                cov=np.diag([ABS_AXIS_SIGMA**2,PER_AXIS_LAT_SIGMA**2,PER_AXIS_LAT_SIGMA**2])
                rk.update_position(z,cov,sim_time_ns=t_ns)
        pf=p0+SEG_HAT*V_RACE*(n_steps*(1/90.0)); perr=rk.position-pf; ip.append(float(np.hypot(perr[1],perr[2])))
    ip=np.array(ip); return float(np.sqrt(np.mean(ip**2))), float(np.percentile(ip,90))
rms,p90=kf_posonly(warm=True); print(f"  WARM pos-only: RMS={rms:.3f} p90={p90:.3f}  (design said warm 0.119/0.182)")
rms,p90=kf_posonly(warm=False); print(f"  COLD pos-only: RMS={rms:.3f} p90={p90:.3f}  (design said cold 0.215/0.324)")
print()

print("="*80); print("P4  did the d4v artifact exercise latency? (read its call path)")
print("="*80)
print("  d4v part_c calls rk.update_position(z,cov,sim_time_ns=t_ns) -- t_ns IS now -> in-sequence,")
print("  zero rewind. And vis-vel via rk.kf.update_velocity (bypass). So the d4v headline numbers")
print("  are an L=0, no-rewind measurement. The RewindKF-default mandate (L~115ms) is NOT in the sim.")
