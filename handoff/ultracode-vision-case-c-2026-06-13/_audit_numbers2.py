import json, numpy as np
base = "handoff/perception-char-2026-06-08"
rows = []
for g in range(6):
    d = json.load(open(f"{base}/characterize_g{g}.json"))
    rows += [r for r in d["rows"] if r.get("associated")]

tr = np.array([r["true_range_m"] for r in rows])
pr = np.array([r["pose_range_m"] for r in rows])
re = np.array([r["range_err_m"] for r in rows])
off = np.array([r["off_ned"] for r in rows])
maha = np.array([r.get("maha", np.nan) for r in rows])

# Replicate the LIVE acceptance: range cap 32 m (on POSE range) + depth-sanity (range_consistent:
# |pose-pred| <= rel_tol*pred + abs_tol). We don't have 'pred' here, but the depth-sanity rejects
# wrong-scale; approximate accepted set with pose_range<=32 AND chi2 maha<=16.27.
acc = (pr <= 32.0) & (maha <= 16.27)
print(f"accepted-set n={acc.sum()} / {len(rows)}  (pose_range<=32 & maha<=16.27)")
print(f"  per-axis world-fix err (N,E,D): mean={off[acc].mean(0).round(3)} std={off[acc].std(0).round(3)}")
print(f"  vs canonical sigma[0.73,0.47,0.29] bias[-0.42,+0.06,-0.28]")
# leak in accepted set
wfe = np.linalg.norm(off[acc],axis=1)
print(f"  accepted world_fix_err: med={np.median(wfe):.3f} p90={np.percentile(wfe,90):.3f} max={wfe.max():.3f}")
print(f"  accepted fixes with |err|>3m (leak): {np.mean(wfe>3)*100:.2f}%")

# ============ KF DEAD-RECKONING DRIFT (case C between vision fixes) ============
# predict() Q per step: B accel_cov B^T, accel_cov = sig_a^2 I + sig_th^2 (S S^T), S=skew(specific_force_world)
# Position-variance growth from accel white noise sig_a over coast time T (random-walk on accel):
#   pos var ~ sig_a^2 * T^3 / 3  (continuous-time double integrator with accel PSD ~ sig_a^2*dt... 
#   but here accel_noise_std is a per-sample std applied as discrete Q). Use the discrete Q sum.
sig_a = 0.3      # accel_noise_std
sig_th = np.deg2rad(1.4)
g = 9.80665
dt = 1/75.0      # IMU ~75 Hz
print("\n=== Case-C dead-reckoning POSITION drift between vision fixes (1-sigma) ===")
print(f"  accel_noise_std={sig_a} m/s^2, attitude_noise={np.rad2deg(sig_th):.1f}deg, IMU dt={dt*1000:.1f}ms")
# attitude-error injected horizontal accel at near-hover: |specific_force|~g -> phantom accel ~ g*sig_th
phantom = g*sig_th
print(f"  attitude-error phantom horizontal accel (1-sigma, hover s~g): {phantom:.3f} m/s^2  (DOMINATES the {sig_a} accel noise)")
# Simulate the actual discrete KF predict pos-variance growth over coast time, hover specific force.
def drift_sigma(T, sf_world):
    # integrate discrete Q exactly like state_estimator.predict
    I3=np.eye(3); Z3=np.zeros((3,3))
    P=np.zeros((6,6))
    S=np.array([[0,-sf_world[2],sf_world[1]],[sf_world[2],0,-sf_world[0]],[-sf_world[1],sf_world[0],0]])
    accel_cov = sig_a**2*I3 + sig_th**2*(S@S.T)
    n=int(round(T/dt))
    for _ in range(n):
        F=np.block([[I3,dt*I3],[Z3,I3]]); B=np.vstack([0.5*dt*dt*I3, dt*I3])
        Q=B@accel_cov@B.T
        P=F@P@F.T+Q
    return np.sqrt(np.diag(P)[:3])  # pos sigma per axis (N,E,D)
sf_hover = np.array([0,0,-g])   # specific force world at hover (reads -g up... here NED down +, at rest sf_world ~ [0,0,-g])
for T in [0.1,0.2,0.5,1.0,2.0]:
    s = drift_sigma(T, sf_hover)
    print(f"  coast T={T:4.1f}s: pos 1-sigma (N,E,D)=[{s[0]:.3f},{s[1]:.3f},{s[2]:.3f}] m  horiz~{np.hypot(s[0],s[1]):.3f} m")

# ============ ASYNC LATENCY: position error from applying fix at t_now vs t_capture ============
print("\n=== Async-fix position error = v * (t_now - t_capture) ===")
# in-loop latency unknown; bracket. command-side latency=67ms; vision pipeline (detect+PnP) adds more.
for v in [5.0, 15.0, 20.0, 30.0]:
    print(f"  v={v:2.0f} m/s:", "  ".join(f"dt={int(dt_ms)}ms->{v*dt_ms/1000:.2f}m" for dt_ms in [33,67,100,150]))
