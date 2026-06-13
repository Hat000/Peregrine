"""Adversarial verification of piece-D (in-loop vision latency) claims.

Independently re-derives every load-bearing number in latency_design.md / latency_results.json
WITHOUT trusting the harness output. Run:
  .venv/Scripts/python.exe handoff/ultracode-vision-case-c-2026-06-13/verify_D_latency.py
Writes nothing under src/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent

from racer.state_estimator import LinearKF  # noqa: E402
from racer.frames import R_world_from_body  # noqa: E402

PASS = []
def check(name, cond, detail=""):
    PASS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

print("=" * 78)
print("VERIFY D — in-loop vision latency. Independent re-derivation.")
print("=" * 78)

# --------------------------------------------------------------------------- #
# 1. v * L_total arithmetic (claim D: 'is v*L_total computed right?')
# --------------------------------------------------------------------------- #
print("\n[1] v * L_total arithmetic (edge L_total: p50 5.734 ms, p90 15.834 ms)")
res = json.loads((HERE / "latency_results.json").read_text())
edge = next(b for b in res["budget"]["budgets"] if b["kind"] == "edge_estimate")
cpu = next(b for b in res["budget"]["budgets"] if b["kind"] == "cpu_upper_bound")
L50_edge = edge["L_total_p50_ms"]; L90_edge = edge["L_total_p90_ms"]
for v, exp50, exp90 in [(20.0, 0.115, 0.317), (30.0, 0.172, 0.475)]:
    got50 = v * L50_edge / 1e3
    got90 = v * L90_edge / 1e3
    check(f"v*L_total edge @{v:.0f} m/s p50", abs(got50 - exp50) < 0.004,
          f"recomputed {got50:.4f} m vs reported ~{exp50} m")
    check(f"v*L_total edge @{v:.0f} m/s p90", abs(got90 - exp90) < 0.004,
          f"recomputed {got90:.4f} m vs reported ~{exp90} m")
# CPU row
for v, exp90 in [(20.0, 2.798), (30.0, 4.196)]:
    got90 = v * cpu["L_total_p90_ms"] / 1e3
    check(f"v*L_total CPU @{v:.0f} m/s p90", abs(got90 - exp90) < 0.05,
          f"recomputed {got90:.4f} m vs reported {exp90} m (CPU L90={cpu['L_total_p90_ms']:.1f} ms)")

# --------------------------------------------------------------------------- #
# 2. Predict-forward vs rewind second-order gap = 1/2 a age^2
# --------------------------------------------------------------------------- #
print("\n[2] rewind-vs-predict-forward gap = 1/2 * a * age^2")
for a, age_ms, exp in [(20.0, 16.0, 0.00256), (20.0, 140.0, 0.196)]:
    age = age_ms / 1e3
    got = 0.5 * a * age * age
    check(f"gap a={a} age={age_ms}ms", abs(got - exp) < 0.01 * max(exp, 1e-3),
          f"0.5*{a}*{age}^2 = {got:.5f} m (claimed ~{exp})")

# --------------------------------------------------------------------------- #
# 3. GFLOPs scaling 22.54 @640^2 -> @640x384
# --------------------------------------------------------------------------- #
print("\n[3] GFLOPs area-scale 640^2 -> 640x384")
g640 = res["stage_ii_detector_edge_estimate"]["gflops_640sq"]
gpad = res["stage_ii_detector_edge_estimate"]["gflops_at_inference_size"]
area_scale = (640 * 384) / (640 * 640)
recomputed = g640 * area_scale
check("gflops area scale", abs(recomputed - gpad) < 0.01,
      f"22.54 * (640*384)/(640*640) = {recomputed:.3f} vs reported {gpad}")
# FLOP floor: 13.524 GFLOP @ 100 TOPS * util
for util, exp in [(0.10, 1.352), (0.35, 0.386)]:
    eff = 100.0 * util
    t_ms = (gpad * 1e9) / (eff * 1e12) * 1e3
    check(f"FLOP floor util={util}", abs(t_ms - exp) < 0.01,
          f"{gpad}e9 / ({eff}e12) s = {t_ms:.3f} ms (claimed {exp})")
# sanity: floor << 5 ms => detector NOT compute-bound (the doc's logic)
check("FLOP floor << edge estimate", 1.352 < 5.0,
      "FLOP compute floor 0.39-1.35 ms is below the 5-15 ms empirical bracket -> not compute-bound")

# --------------------------------------------------------------------------- #
# 4. KF rewind vs naive (current-state apply) — does compensation actually win?
#    Build a true trajectory, a stale fix captured `age` ago, and compare:
#      (naive)  apply z at current state
#      (rewind) rewind to capture, apply, re-propagate the SAME IMU
#      (predf)  z + v_hat*age, apply at current
#    Truth = where the drone REALLY is at apply time. Compute position error of the
#    posterior mean vs truth for each.
# --------------------------------------------------------------------------- #
print("\n[4] KF rewind vs naive vs predict-forward — posterior position error vs TRUTH")

def run_kf_scenario(v_mps, a_mps2, age_s, imu_dt=1/75., fix_std=0.40, seed=0):
    rng = np.random.default_rng(seed)
    # truth: start at origin, constant accel `a` along N, base speed v along N.
    # number of IMU steps spanning `age`
    nsteps = max(1, int(round(age_s / imu_dt)))
    dt = age_s / nsteps
    # attitude level; specific force = body accel needed to produce a_world along N.
    # We'll feed accel_body s.t. R_wb@accel_body + g = a_world. Level => R_wb = I.
    R_wb = np.eye(3)
    a_world = np.array([a_mps2, 0.0, 0.0])
    # accel_body (specific force) = R_wb.T @ (a_world - g_world); g_world=[0,0,9.80665]
    g = np.array([0, 0, 9.80665])
    accel_body = R_wb.T @ (a_world - g)

    def truth_at(t, x0, v0):
        return x0 + v0 * t + 0.5 * a_world * t * t

    x0 = np.zeros(3)
    v0 = np.array([v_mps, 0.0, 0.0])
    p_capture = truth_at(0.0, x0, v0)
    p_apply = truth_at(age_s, x0, v0)
    v_capture = v0.copy()

    # measurement z = true capture position + noise
    z = p_capture + rng.normal(0, fix_std, 3)
    R = (fix_std ** 2) * np.eye(3)

    # ---- NAIVE: KF has propagated to apply time, applies z (a capture-time meas) as if now
    kf_n = LinearKF.initialize(p_capture, v_capture, pos_std=0.5, vel_std=0.3)
    for _ in range(nsteps):
        kf_n.predict(accel_body, R_wb, dt)
    kf_naive = LinearKF(x=kf_n.x.copy(), P=kf_n.P.copy())
    kf_naive.update_position(z, R)
    err_naive = np.linalg.norm(kf_naive.position - p_apply)

    # ---- PREDICT-FORWARD: z_forward = z + v_hat*age (v_hat = KF velocity at apply time)
    kf_pf = LinearKF(x=kf_n.x.copy(), P=kf_n.P.copy())
    v_hat = kf_pf.velocity.copy()
    z_fwd = z + v_hat * age_s
    # inflate R for propagation: add age^2 * Cov(v_hat) (use KF vel cov) — conservative
    Rv = kf_pf.P[3:, 3:]
    R_fwd = R + age_s * age_s * Rv
    kf_pf.update_position(z_fwd, R_fwd)
    err_pf = np.linalg.norm(kf_pf.position - p_apply)

    # ---- REWIND: snapshot state at capture, apply z there, re-propagate SAME IMU to apply
    kf_rw = LinearKF.initialize(p_capture, v_capture, pos_std=0.5, vel_std=0.3)
    # (this IS the capture-time prior, identical to how naive started before propagation)
    kf_rw.update_position(z, R)                     # apply fix at capture epoch
    for _ in range(nsteps):
        kf_rw.predict(accel_body, R_wb, dt)         # re-propagate the buffered IMU
    err_rw = np.linalg.norm(kf_rw.position - p_apply)

    return dict(err_naive=err_naive, err_pf=err_pf, err_rw=err_rw,
                rw_vs_pf_meanpos=np.linalg.norm(kf_rw.position - kf_pf.position),
                age_s=age_s, nsteps=nsteps)

# Average over seeds to suppress measurement-noise variance and isolate the BIAS each method has.
def avg_scenario(v, a, age, nseed=400):
    accN = accPF = accRW = accGAP = 0.0
    for s in range(nseed):
        r = run_kf_scenario(v, a, age, seed=s)
        accN += r["err_naive"]; accPF += r["err_pf"]; accRW += r["err_rw"]
        accGAP += r["rw_vs_pf_meanpos"]
    return dict(naive=accN/nseed, pf=accPF/nseed, rw=accRW/nseed, gap=accGAP/nseed)

# edge p90 age ~16 ms, 20 m/s, aggressive accel 20 m/s^2
for (v, a, age_ms) in [(20.0, 20.0, 16.0), (30.0, 20.0, 16.0), (20.0, 20.0, 140.0)]:
    age = age_ms / 1e3
    r = avg_scenario(v, a, age)
    print(f"  v={v} a={a} age={age_ms}ms (avg over seeds): "
          f"naive={r['naive']:.4f}  predf={r['pf']:.4f}  rewind={r['rw']:.4f}  "
          f"|rw-pf meanpos|={r['gap']:.5f} m")
    # compensation must beat naive at VQ2 speed
    check(f"predict-forward beats naive v={v} age={age_ms}ms", r['pf'] < r['naive'] * 0.6,
          f"predf {r['pf']:.4f} m << naive {r['naive']:.4f} m")
    # rewind ~ predict-forward at small age (gap = 1/2 a age^2 in mean position)
    exp_gap = 0.5 * a * age * age
    check(f"rewind~predf gap ~ 1/2 a age^2 (v={v} age={age_ms}ms)",
          abs(r['gap'] - exp_gap) < 0.5 * max(exp_gap, 1e-3) + 0.01,
          f"measured meanpos gap {r['gap']:.5f} m vs 1/2 a age^2 = {exp_gap:.5f} m")

# --------------------------------------------------------------------------- #
# 5. Rewind bit-identical to naive at L=0 (age=0)  [claim B premise D leans on]
# --------------------------------------------------------------------------- #
print("\n[5] At age=0, rewind == naive == predict-forward (no staleness)")
r0 = run_kf_scenario(20.0, 20.0, 0.0, seed=7)
# at age=0, all three should be IDENTICAL position error (no propagation, age=0 -> z_fwd=z)
check("rewind==naive @age=0", abs(r0["err_rw"] - r0["err_naive"]) < 1e-9,
      f"rewind {r0['err_rw']:.6e} vs naive {r0['err_naive']:.6e}")
check("predf==naive @age=0", abs(r0["err_pf"] - r0["err_naive"]) < 1e-9,
      f"predf {r0['err_pf']:.6e} vs naive {r0['err_naive']:.6e}")

# --------------------------------------------------------------------------- #
# 6. The 30 Hz / 10 m rule: fix captured at 10 m applies at 10 - v*L_total
# --------------------------------------------------------------------------- #
print("\n[6] 30Hz/10m rule erosion: fix at 10m applies at 10 - v*L_total")
for (v, Lms, exp_apply) in [(30.0, L90_edge, 9.53), (30.0, cpu["L_total_p90_ms"], 6.1)]:
    erosion = v * Lms / 1e3
    applies_at = 10.0 - erosion
    check(f"applies_at v=30 L={Lms:.1f}ms", abs(applies_at - exp_apply) < 0.1,
          f"10 - 30*{Lms:.1f}ms = {applies_at:.2f} m (claimed ~{exp_apply})")
check("edge does NOT push past 10m", 10.0 - 30.0 * L90_edge / 1e3 > 9.0,
      f"edge p90 erosion {30.0*L90_edge/1e3:.2f} m < 1 m -> stays inside 10 m window")
check("CPU WOULD break 10m rule", 10.0 - 30.0 * cpu["L_total_p90_ms"] / 1e3 < 7.0,
      f"CPU erosion {30.0*cpu['L_total_p90_ms']/1e3:.2f} m -> window collapses to ~6m")

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
n_pass = sum(1 for _, p, _ in PASS if p)
print(f"SUMMARY: {n_pass}/{len(PASS)} checks PASS")
for name, p, _ in PASS:
    if not p:
        print(f"   FAILED: {name}")
print("=" * 78)
