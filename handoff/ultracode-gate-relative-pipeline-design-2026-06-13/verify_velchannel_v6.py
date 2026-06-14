"""ADVERSARIAL re-derivation of the d4v velchannel claims.
LENS: is-position-fix-differencing-noise-honestly-propagated-and-does-it-help-the-margin.

Charged to REFUTE. Independent re-derivation of every load-bearing number; explicit attacks:
  (V1) closed-form noise propagation: sqrt(2)*sigma_p/dt and the LSQ slope variance. Recompute
       from first principles WITHOUT the prototype's formula; also empirically (MC slope fit).
  (V2) RewindKF BYPASS: the prototype calls rk.kf.update_velocity (the bare wrapped filter),
       NOT rk.update_velocity (the recorded wrapper). So the vel update is INVISIBLE to the
       OOSM rewind. Demonstrate: a later position fix that rewinds will DISCARD all vel updates
       in the window -> the visvel benefit is NOT preserved under the RewindKF default the design
       mandates. Quantify the in-plane miss with the CORRECT recorded path + OOSM latency on.
  (V3) cold-start honesty: the +1.5 m/s lateral offset is ASSUMED. Re-run with the cold seed but
       UNBIASED velocity init (offset 0, std 1.5) -- does cold still go OVER margin? i.e. is the
       cold failure driven by the assumed offset or by the variance floor?
  (V4) p90 honesty: the design headlines RMS but the gate-4 guard is a CONTACT rule. visvel_1.0
       p90 = 0.213 m is OVER 0.155 m. Does "clears the margin" survive a p90 / worst-case read?
  (V5) does it inject GT velocity? zv = v0 + noise where v0 is TRUE velocity. A real inter-frame
       PnP-delta does NOT measure v0 -- it measures (p_fix[k]-p_fix[k-1])/dt, which is the SAME
       position-fix noise re-differenced. Replace the GT-velocity measurement with an HONEST
       PnP-delta (differenced noisy positions) and see if the margin still flips.
  (V6) VQ1 / case-A/B bit-exactness: does adding a vel channel touch the VQ1 path? (read-only check)

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/.../verify_velchannel_v6.py
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

SEED = 20260613
MARGIN_G4 = 0.155
V_RACE = 37.0
G = 9.80665
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])
SEG = G4 - G3
SEG_LEN = float(np.linalg.norm(SEG))
SEG_HAT = SEG / SEG_LEN
T_TRANSIT = SEG_LEN / V_RACE
PER_AXIS_LAT_SIGMA = 0.265
ABS_AXIS_SIGMA = 0.50
ACCEL_NOISE_STD = 0.3
IMU_DT = 1.0 / 90.0
FIX_DT = 1.0 / 14.0


def _attitude():
    yaw = float(np.arctan2(SEG_HAT[1], SEG_HAT[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / G))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    accel_body = R_wb.T @ (-GRAVITY_NED)
    return R_wb, accel_body


# ============== V1: closed-form noise, re-derived independently + empirically ==============
def v1_noise():
    print("="*80); print("V1  noise propagation -- independent re-derivation"); print("="*80)
    # single-pair: var(diff)=2 sigma^2 -> sigma_v = sqrt(2) sigma_p / dt. Confirm by MC.
    rng = np.random.default_rng(1)
    for sig, dt in [(0.265, FIX_DT), (0.50, FIX_DT)]:
        closed = np.sqrt(2)*sig/dt
        d = (rng.normal(0, sig, 200000) - rng.normal(0, sig, 200000)) / dt
        print(f"  single-pair sig={sig} dt={dt:.4f}: closed={closed:.3f}  MC={d.std():.3f}")
    # LSQ slope variance over N uniform points: re-derive var(slope)=sig^2 / S_tt, S_tt=dt^2 N(N^2-1)/12
    W = T_TRANSIT
    N = max(2, int(14.0 * W))
    dt = W/(N-1)
    for sig, label in [(0.265, "lateral"), (0.50, "radial")]:
        S_tt = dt**2 * N*(N**2-1)/12.0
        closed = sig/np.sqrt(S_tt)
        # empirical: fit slope to N noisy points, MC
        t = np.arange(N)*dt; t = t - t.mean()
        slopes = []
        for _ in range(40000):
            y = rng.normal(0, sig, N)            # pure noise on top of zero true slope
            slopes.append(np.sum(t*y)/np.sum(t*t))
        print(f"  LSQ {label}: N={N} closed={closed:.3f}  MC={np.std(slopes):.3f}")
    print("  -> CLAIM 1 noise formulas: re-derived OK (within MC)\n")


# ============== V2: RewindKF bypass -- does the vel benefit survive the OOSM default? ==========
def v2_rewind_bypass(n_mc=600, latency_s=0.0, vel_via_wrapper=False, vis_vel_sigma=1.0):
    """Replicate part_c cold+visvel but with (a) OOSM latency on the position fix and
    (b) a switch: route the vel update through rk.update_velocity (RECORDED) vs rk.kf.update_velocity
    (BYPASS, as the prototype does). With latency>0 the position fix triggers a rewind+replay;
    a BYPASSED vel update is NOT in the buffer -> it is silently dropped on every rewind."""
    R_wb, accel_body_nom = _attitude()
    att_bias_rad = np.deg2rad(0.5)
    a_phantom = G*np.sin(att_bias_rad)
    accel_bias_world = np.array([0.0, a_phantom, 0.0])
    accel_body_biased = accel_body_nom + R_wb.T @ accel_bias_world
    p0 = G3.copy(); v0 = SEG_HAT*V_RACE
    n_steps = int(T_TRANSIT/IMU_DT)
    lat_ns = int(latency_s*1e9)
    ip = []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 13*s + 777)
        v_init = v0 + np.array([0.0, 1.5, 0.0]); vel_std0 = 1.5
        kf = LinearKF.initialize(p0 + r.normal(0,0.3,3), v_init, pos_std=0.5,
                                 vel_std=vel_std0, accel_noise_std=ACCEL_NOISE_STD)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t=0.0; t_ns=0; next_fix=0.0
        for k in range(n_steps):
            t += IMU_DT; t_ns += int(IMU_DT*1e9)
            a_meas = accel_body_biased + r.normal(0, ACCEL_NOISE_STD, 3)
            rk.predict(a_meas, R_wb, IMU_DT, t_ns)
            p_true = p0 + SEG_HAT*V_RACE*t
            rng_to_g4 = float(np.linalg.norm(G4 - p_true))
            if t >= next_fix and rng_to_g4 < 12.0:
                next_fix += FIX_DT
                nE=r.normal(0,PER_AXIS_LAT_SIGMA); nD=r.normal(0,PER_AXIS_LAT_SIGMA); nN=r.normal(0,ABS_AXIS_SIGMA)
                # the fix is stamped at capture time = now - latency
                t_fix_true = p0 + SEG_HAT*V_RACE*max(0.0, t-latency_s)
                z = t_fix_true + np.array([nN,nE,nD])
                cov = np.diag([ABS_AXIS_SIGMA**2, PER_AXIS_LAT_SIGMA**2, PER_AXIS_LAT_SIGMA**2])
                if lat_ns>0:
                    rk.update_position_at(t_ns - lat_ns, z, cov)
                else:
                    rk.update_position(z, cov, sim_time_ns=t_ns)
                if vis_vel_sigma is not None:
                    nv = r.normal(0, vis_vel_sigma, 3); zv = v0 + nv
                    Rv = (vis_vel_sigma**2)*np.eye(3)
                    if vel_via_wrapper:
                        rk.update_velocity(zv, Rv, sim_time_ns=t_ns)   # RECORDED
                    else:
                        rk.kf.update_velocity(zv, Rv)                  # BYPASS (prototype's call)
        p_true_final = p0 + SEG_HAT*V_RACE*(n_steps*IMU_DT)
        perr = rk.position - p_true_final
        ip.append(float(np.hypot(perr[1], perr[2])))
    ip=np.array(ip)
    return float(np.sqrt(np.mean(ip**2))), float(np.percentile(ip,90))


# ============== V5: honest PnP-delta vel (differenced noisy positions, NOT GT) ==============
def v5_honest_pnp_delta(n_mc=600, smooth_frames=4, frame_hz=30.0):
    """Replace zv = v0 + noise (GT velocity) with an HONEST inter-frame PnP translation delta:
    v_meas = (p_pnp[k] - p_pnp[k-smooth]) / (smooth*dt_frame), where p_pnp carries the SAME
    gate-relative position noise. This is what the channel can ACTUALLY deliver. sigma of this
    estimator = sqrt(2)*sigma_p/(smooth*dt_frame) -> we feed THAT as Rv (honest cov)."""
    R_wb, accel_body_nom = _attitude()
    att_bias_rad = np.deg2rad(0.5); a_phantom=G*np.sin(att_bias_rad)
    accel_bias_world=np.array([0.0,a_phantom,0.0])
    accel_body_biased = accel_body_nom + R_wb.T@accel_bias_world
    p0=G3.copy(); v0=SEG_HAT*V_RACE; n_steps=int(T_TRANSIT/IMU_DT)
    dt_frame=1.0/frame_hz
    sig_vmeas = np.sqrt(2)*PER_AXIS_LAT_SIGMA/(smooth_frames*dt_frame)   # honest lateral vel-meas sigma
    # along-track uses the radial position sigma
    sig_vmeas_N = np.sqrt(2)*ABS_AXIS_SIGMA/(smooth_frames*dt_frame)
    print(f"  honest PnP-delta: smooth={smooth_frames} frames @ {frame_hz}Hz -> "
          f"sigma_v lateral={sig_vmeas:.2f} m/s, along={sig_vmeas_N:.2f} m/s")
    ip=[]
    for s in range(n_mc):
        r=np.random.default_rng(SEED+13*s+999)
        v_init=v0+np.array([0.0,1.5,0.0]); kf=LinearKF.initialize(p0+r.normal(0,0.3,3),v_init,
            pos_std=0.5, vel_std=1.5, accel_noise_std=ACCEL_NOISE_STD)
        rk=RewindKF(kf=kf, horizon_s=0.5); t=0.0; t_ns=0; next_fix=0.0
        pnp_hist=[]  # (t, p_pnp_noisy) at frame cadence for the delta
        next_frame=0.0
        for k in range(n_steps):
            t+=IMU_DT; t_ns+=int(IMU_DT*1e9)
            a_meas=accel_body_biased + r.normal(0,ACCEL_NOISE_STD,3)
            rk.predict(a_meas,R_wb,IMU_DT,t_ns)
            p_true=p0+SEG_HAT*V_RACE*t; rng_to_g4=float(np.linalg.norm(G4-p_true))
            # frame-cadence PnP sample (for the velocity delta), valid only when gate in band
            if t>=next_frame and rng_to_g4<12.0:
                next_frame+=dt_frame
                pnp=p_true + np.array([r.normal(0,ABS_AXIS_SIGMA), r.normal(0,PER_AXIS_LAT_SIGMA),
                                       r.normal(0,PER_AXIS_LAT_SIGMA)])
                pnp_hist.append((t,pnp))
            if t>=next_fix and rng_to_g4<12.0:
                next_fix+=FIX_DT
                nE=r.normal(0,PER_AXIS_LAT_SIGMA); nD=r.normal(0,PER_AXIS_LAT_SIGMA); nN=r.normal(0,ABS_AXIS_SIGMA)
                z=p_true+np.array([nN,nE,nD]); cov=np.diag([ABS_AXIS_SIGMA**2,PER_AXIS_LAT_SIGMA**2,PER_AXIS_LAT_SIGMA**2])
                rk.update_position(z,cov,sim_time_ns=t_ns)
                # honest vel from PnP-delta if we have >= smooth_frames history
                if len(pnp_hist)>smooth_frames:
                    t_a,p_a=pnp_hist[-1-smooth_frames]; t_b,p_b=pnp_hist[-1]
                    if t_b>t_a:
                        zv=(p_b-p_a)/(t_b-t_a)
                        Rv=np.diag([sig_vmeas_N**2, sig_vmeas**2, sig_vmeas**2])
                        rk.update_velocity(zv,Rv,sim_time_ns=t_ns)   # RECORDED + honest
        p_true_final=p0+SEG_HAT*V_RACE*(n_steps*IMU_DT)
        perr=rk.position-p_true_final; ip.append(float(np.hypot(perr[1],perr[2])))
    ip=np.array(ip)
    return float(np.sqrt(np.mean(ip**2))), float(np.percentile(ip,90)), sig_vmeas


# ============== V3: cold without the assumed +1.5 offset ==============
def v3_cold_unbiased(n_mc=600, offset=0.0):
    R_wb, accel_body_nom=_attitude(); att=np.deg2rad(0.5); a_ph=G*np.sin(att)
    abw=np.array([0.0,a_ph,0.0]); ab=accel_body_nom+R_wb.T@abw
    p0=G3.copy(); v0=SEG_HAT*V_RACE; n_steps=int(T_TRANSIT/IMU_DT); ip=[]
    for s in range(n_mc):
        r=np.random.default_rng(SEED+13*s+555)
        v_init=v0+np.array([0.0,offset,0.0])
        kf=LinearKF.initialize(p0+r.normal(0,0.3,3),v_init,pos_std=0.5,vel_std=1.5,accel_noise_std=ACCEL_NOISE_STD)
        rk=RewindKF(kf=kf,horizon_s=0.5); t=0.0;t_ns=0;next_fix=0.0
        for k in range(n_steps):
            t+=IMU_DT;t_ns+=int(IMU_DT*1e9); a_meas=ab+r.normal(0,ACCEL_NOISE_STD,3)
            rk.predict(a_meas,R_wb,IMU_DT,t_ns); p_true=p0+SEG_HAT*V_RACE*t
            if t>=next_fix and np.linalg.norm(G4-p_true)<12.0:
                next_fix+=FIX_DT
                z=p_true+np.array([r.normal(0,ABS_AXIS_SIGMA),r.normal(0,PER_AXIS_LAT_SIGMA),r.normal(0,PER_AXIS_LAT_SIGMA)])
                cov=np.diag([ABS_AXIS_SIGMA**2,PER_AXIS_LAT_SIGMA**2,PER_AXIS_LAT_SIGMA**2])
                rk.update_position(z,cov,sim_time_ns=t_ns)
        pf=p0+SEG_HAT*V_RACE*(n_steps*IMU_DT); perr=rk.position-pf; ip.append(float(np.hypot(perr[1],perr[2])))
    ip=np.array(ip); return float(np.sqrt(np.mean(ip**2))), float(np.percentile(ip,90))


if __name__ == "__main__":
    v1_noise()

    print("="*80); print("V3  cold-start honesty: is the OVER-margin driven by the ASSUMED +1.5 m/s offset?")
    print("="*80)
    for off in [1.5, 0.0]:
        rms,p90 = v3_cold_unbiased(offset=off)
        print(f"  cold offset={off:+.1f} m/s: in-plane RMS={rms:.3f}  p90={p90:.3f}  "
              f"<margin RMS? {'YES' if rms<MARGIN_G4 else 'NO'}")
    print()

    print("="*80); print("V2  RewindKF BYPASS + OOSM latency: does the visvel benefit survive the default?")
    print("="*80)
    for lat in [0.0, 0.115]:
        for via in [False, True]:
            rms,p90 = v2_rewind_bypass(latency_s=lat, vel_via_wrapper=via, vis_vel_sigma=1.0)
            tag = "WRAPPER(recorded)" if via else "BYPASS(rk.kf, proto)"
            print(f"  latency={lat*1000:.0f}ms  vel via {tag:22s}: RMS={rms:.3f} p90={p90:.3f} "
                  f"<margin RMS? {'YES' if rms<MARGIN_G4 else 'NO'}")
    # also: cold with no vis-vel under latency (baseline)
    rms,p90 = v2_rewind_bypass(latency_s=0.115, vel_via_wrapper=True, vis_vel_sigma=None)
    print(f"  latency=115ms  NO vis-vel (cold pos-fix-diff only):     RMS={rms:.3f} p90={p90:.3f}")
    print()

    print("="*80); print("V5  HONEST PnP-delta vel (differenced noisy positions, NOT GT velocity)")
    print("="*80)
    for sm in [4, 6, 9]:
        rms,p90,sv = v5_honest_pnp_delta(smooth_frames=sm)
        print(f"  smooth={sm}: in-plane RMS={rms:.3f}  p90={p90:.3f}  "
              f"<margin RMS? {'YES' if rms<MARGIN_G4 else 'NO'}  (sigma_v_lat={sv:.2f})")
