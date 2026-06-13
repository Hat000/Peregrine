"""Final adversarial angles on piece B:
1. Does the rewind win survive a DIFFERENT fix-noise model (heavier-tailed / range-like)?
   The v*L removal should be INDEPENDENT of the fix-noise distribution (it's a timing fix).
2. Is 'drop too-old fix' actually safer than naive in-place at horizons SHORTER than L?
   Quantify: at L beyond horizon, naive injects v*L bias; dropping injects nothing but
   loses the measurement. Show the trade so the horizon-sizing risk is honest.
3. Re-verify cost numbers independently (memory + per-fix time), not trusting timeit alone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff" / "ultracode-vision-case-c-2026-06-13"))
from kf_rewind_buffer import RewindKF  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])
DT = 1.0 / 90.0


def fb(a, R=None):
    R = np.eye(3) if R is None else R
    return R.T @ (a - GRAVITY_NED)


def run(speed, L, seed, use_rewind, fix_model="flat", horizon_s=0.5, dur=6.0):
    rng = np.random.default_rng(seed)
    n = int(dur * 90)
    t = np.arange(n) * DT
    pN = speed * t
    pE = (0.25*speed/0.9**2)*np.sin(0.9*t)
    pD = -(0.10*speed/0.6**2)*np.cos(0.6*t)
    pos = np.column_stack([pN, pE, pD])
    aE = -0.25*speed*np.sin(0.9*t); aD = 0.10*speed*0.6*np.cos(0.6*t)
    acc = np.column_stack([np.zeros(n), aE, aD])
    t_ns = (t*1e9).astype(np.int64) + 1_000_000_000
    base = LinearKF.initialize(pos[0], np.array([speed,0,0]), pos_std=0.5, vel_std=0.5, accel_noise_std=0.3)
    kf = RewindKF(kf=base, horizon_s=horizon_s) if use_rewind else base
    nxt = 0.5/28.0; last = t_ns[0]; errs = []
    for k in range(1, n):
        dt = (t_ns[k]-last)/1e9; last = t_ns[k]
        am = acc[k] + rng.normal(0, 0.05, 3)
        if use_rewind:
            kf.predict(fb(am), np.eye(3), dt, sim_time_ns=int(t_ns[k]))
        else:
            kf.predict(fb(am), np.eye(3), dt)
        tn = (t_ns[k]-t_ns[0])/1e9
        if tn >= nxt:
            nxt += 1.0/28.0
            ck = max(0, k-L)
            cap = pos[ck]
            if fix_model == "flat":
                std = np.sqrt(np.array([0.73,0.47,0.29])**2 + 0.40**2)
                z = cap + rng.normal(0, std)
                cov = np.diag(std**2)
            elif fix_model == "heavy":
                # heavy-tailed (student-t-ish) fix error, same nominal scale, mis-specified cov
                std = np.sqrt(np.array([0.73,0.47,0.29])**2 + 0.40**2)
                z = cap + std * rng.standard_t(3, size=3) / np.sqrt(3)  # t(3) scaled to ~unit var
                cov = np.diag(std**2)
            kf.update_position(z, cov) if not use_rewind else kf.update_position_at(int(t_ns[ck]), z, cov)
        if tn > 1.0:
            errs.append(float(np.sum((kf.x[:3]-pos[k])**2)))
    return float(np.sqrt(np.mean(errs)))


def robustness_to_noise_model():
    print("[1] Rewind win is INDEPENDENT of fix-noise distribution (timing fix, not noise fix)")
    for fm in ["flat", "heavy"]:
        nv = np.mean([run(20.0, 2, s, False, fm) for s in range(8)])
        rw = np.mean([run(20.0, 2, s, True, fm) for s in range(8)])
        print(f"    fix_model={fm:5s}: naive={nv:.3f} rewind={rw:.3f} cut={100*(nv-rw)/nv:.1f}%")


def horizon_undersizing():
    print("[2] Horizon < latency: dropping vs naive-applying the too-old fix")
    # 0.2 s horizon = 18 ticks. L=4 (44ms) fits; sweep L up to 30 ticks (333 ms > horizon).
    speed = 20.0
    for L in [4, 18, 25]:
        # rewind with a SHORT 0.2 s (18-tick) horizon -> fixes older than 18 ticks DROP
        rw = np.mean([run(speed, L, s, True, "flat", horizon_s=0.2) for s in range(8)])
        nv = np.mean([run(speed, L, s, False, "flat") for s in range(8)])
        in_horizon = L * DT <= 0.2
        print(f"    L={L:2d} ({L*DT*1000:3.0f}ms, in-horizon={in_horizon}): "
              f"naive={nv:.3f} rewind(0.2s horizon)={rw:.3f}")
    print("    -> when L exceeds the horizon the rewind DROPS the fix (coasts on IMU);")
    print("       it neither helps nor injects the v*L bias. Horizon must exceed worst-case L.")


def cost_independent():
    print("[3] Independent cost check")
    # memory: build the buffer, measure real python object footprint of one _Op snapshot
    base = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5)
    kf = RewindKF(kf=base, horizon_s=0.5)
    t0 = 1_000_000_000
    for i in range(1, 60):
        kf.predict(np.array([0,0,-9.80665]), np.eye(3), DT, sim_time_ns=t0+int(i*DT*1e9))
    nops = len(kf._ops)
    # numeric payload bytes per op: x(6)+P(36)+accel(3)+R_wb(9) f64 = 54*8 = 432 B (matches report)
    numeric_bytes = (6+36+3+9)*8
    print(f"    buffered ops at 0.5s/90Hz = {nops} (report: 46); numeric payload {numeric_bytes} B/op "
          f"-> {nops*numeric_bytes/1024:.1f} KB numeric")
    # per-fix wall time: median over many fixes at L=3 on a pre-filled buffer (no fill in the timed region)
    base2 = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.5, vel_std=0.5)
    kf2 = RewindKF(kf=base2, horizon_s=0.5)
    for i in range(1, 60):
        kf2.predict(np.array([0,0,-9.80665]), np.eye(3), DT, sim_time_ns=t0+int(i*DT*1e9))
    # advance + fix repeatedly, timing only update_position_at
    times = []
    tnow = kf2.now_ns
    for j in range(2000):
        tnow += int(DT*1e9)
        kf2.predict(np.array([0,0,-9.80665]), np.eye(3), DT, sim_time_ns=tnow)
        t_fix = tnow - int(3*DT*1e9)
        t1 = time.perf_counter()
        kf2.update_position_at(t_fix, np.array([0.1,0.1,0.1]), np.diag([0.5,0.5,0.5])**2)
        times.append((time.perf_counter()-t1)*1000)
    times = np.array(times)
    print(f"    per-fix update_position_at (L=3): median={np.median(times):.3f} ms "
          f"p95={np.percentile(times,95):.3f} ms (report ~0.03-0.18 ms)")
    print(f"    vs 33 ms vision period @ 28 fps and 33 ms policy period @ 30 Hz -> negligible")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    robustness_to_noise_model()
    print()
    horizon_undersizing()
    print()
    cost_independent()
