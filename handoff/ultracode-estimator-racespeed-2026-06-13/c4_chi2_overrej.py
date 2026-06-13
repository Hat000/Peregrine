"""c4_chi2_overrej.py -- ANALYTIC companion: the chi2 over-rejection / leak cost of lowering the floor.

The navigator gate is d2 = nu^T S^-1 nu > 16.27 -> reject, with S = P_prior + R(floor). The floor was
ADDED (vision-pkg2) precisely because at SHORT range the analytic PnP cov is mm-tight and the
constant systematics (the per-gate residual + close-range depth) make a GOOD fix's innovation large
relative to its (too-tight) cov -> maha in the hundreds -> 13-35% of GOOD close fixes rejected. The
0.40 m floor inflates R so a good-but-biased fix passes.

This computes, WITHOUT the KF dynamics, the over-rejection probability for a good fix that carries the
per-gate residual bias `b` (the part global de-bias leaves), as a function of the floor. A good fix's
innovation is nu = b + noise; its maha against S = P + R. As the floor drops:
  - R shrinks -> S shrinks -> maha grows -> a good biased fix is MORE likely rejected (over-rejection
    regrowth -- the exact failure the floor cures).
We report P(reject | good biased fix) vs floor, for a representative prior P and the gate-4 residual.
This is the mechanism behind the MC `over_rej_frac` and isolates it cleanly.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c4_chi2_overrej.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "handoff" / "ultracode-vision-case-c-2026-06-13"))
sys.path.insert(0, str(ROOT / "handoff" / "ultracode-estimator-racespeed-2026-06-13"))

from racer.frames import R_world_from_body, R_camera_from_body  # noqa: E402
from racer.localization import gate_pose_to_world_position  # noqa: E402
from racer.contracts import Gate, GatePose  # noqa: E402
import a1_sim as A  # noqa: E402

HERE = Path(__file__).resolve().parent
GATE = 16.27
FLOORS = [0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.025, 0.0]
# per-gate residual the GLOBAL de-bias leaves (case-C residual sigma, FACTS). Treated as a per-track
# CONSTANT bias vector. We span the population via its sigma; the mean per-track |b| in-plane ~ this.
PER_GATE_RESID_NED = np.array([0.21, 0.24, 0.03])
# per-fix zero-mean noise std near gate (the [0,8) band measured) -- the floor-INDEPENDENT measurement
# scatter that is ALSO present in the innovation. From FACTS [0,8): std [0.97, 0.55, 0.57].
NEAR_NOISE_STD = np.array([0.97, 0.55, 0.57])


def fix_cov(drone_pos, R_wb, floor):
    R_wc = R_wb @ R_camera_from_body().T
    t_cam_gate = R_wc.T @ (A.G4 - drone_pos)
    gate = Gate(gate_id=4, position_ned=A.G4.copy(), R_world_gate=np.eye(3))
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam_gate,
                  reproj_error_px=0.3, gate_id=4, covariance=None, n_corners=4)
    _p, cov = gate_pose_to_world_position(gp, gate, R_wb, fix_cov_floor_std=floor)
    return cov


def p_reject_good(floor, R_wb, prior_sigma, range_m, n_mc=200000, seed=20260613):
    """MC over-rejection probability for a GOOD fix carrying the per-gate residual bias + measurement
    noise, against the chi2 gate at this floor. The KF has converged the prior to ~prior_sigma; the
    measurement noise floor is the [0,8) band. S = diag(prior_sigma^2) + R(floor)."""
    rng = np.random.default_rng(seed + int(floor * 1000) + int(range_m))
    drone = A.G4 - A.SEG_HAT * range_m
    R = fix_cov(drone, R_wb, floor)
    P = np.diag(prior_sigma ** 2)
    S = P + R
    Sinv = np.linalg.inv(S)
    # innovation = (residual bias the de-bias leaves, a per-track constant drawn ~N(0,resid)) + noise
    # We integrate over the population of tracks (resid draw) AND the per-fix noise.
    b = rng.normal(0, PER_GATE_RESID_NED, size=(n_mc, 3))
    n = rng.normal(0, NEAR_NOISE_STD, size=(n_mc, 3))
    nu = b + n
    d2 = np.einsum("ij,jk,ik->i", nu, Sinv, nu)
    return float((d2 > GATE).mean())


def p_leak_pass(floor, R_wb, prior_sigma, range_m, leak_world_offset):
    """For a wrong-gate LEAK whose true world offset is `leak_world_offset` (m, a fixed gross error),
    does it pass the gate at this floor? maha of the leak innovation vs S. Lower floor -> smaller S
    -> higher maha -> leak MORE likely rejected (good for leaks), BUT the bounded leaks that DO pass
    are weighted with a higher Kalman gain -> more state damage (we report gain too)."""
    drone = A.G4 - A.SEG_HAT * range_m
    R = fix_cov(drone, R_wb, floor)
    P = np.diag(prior_sigma ** 2)
    S = P + R
    nu = np.asarray(leak_world_offset, float)
    d2 = float(nu @ np.linalg.solve(S, nu))
    # Kalman gain (scalar proxy): K = P (P+R)^-1 ; the fraction of the innovation pulled into the state
    K = P @ np.linalg.inv(S)
    pull = float(np.linalg.norm(K @ nu))   # how far a passing leak yanks the state (m)
    K_inplane = float(np.linalg.norm((K @ nu)[1:3]))
    return dict(d2=d2, passes=bool(d2 <= GATE), state_pull_m=pull, state_pull_inplane_m=K_inplane)


def main():
    R_wb = R_world_from_body(0.0, A.DRAG_HOLD_PITCH_RAD, A.YAW_RAD)
    # representative converged prior 1-sigma at the gate-4 approach (from a1: filtered ~0.24 m/axis
    # in-plane at the 0.40 floor; tighter as floor drops -- but the GATE uses the live P, so we report
    # over-rejection at a band of prior sigmas to bracket).
    out = {"over_rejection_good_fix": {}, "leak_behaviour": {}, "gate_chi2": GATE}

    print("=== Over-rejection P(reject | GOOD biased fix) vs floor (the failure the floor was added to cure) ===")
    print("    range=4 m, converged prior 1-sigma per axis as noted")
    for prior_s in [0.24, 0.15, 0.10]:
        prior_sigma = np.array([prior_s, prior_s, prior_s])
        row = {}
        line = f"  prior_sigma={prior_s:.2f}: "
        for floor in FLOORS:
            p = p_reject_good(floor, R_wb, prior_sigma, 4.0)
            row[f"{floor:.3f}"] = p
            line += f"f{floor:.2f}->{p*100:4.1f}% "
        out["over_rejection_good_fix"][f"prior_{prior_s:.2f}"] = row
        print(line)

    print("\n=== Leak behaviour: a bounded wrong-gate offset (in-plane 0.9 m) vs floor ===")
    print("    range=4 m, prior_sigma=0.24; reports pass? + how far a passing leak pulls the state")
    leak_off = np.array([0.0, 0.64, 0.64])   # ~0.9 m in-plane bounded leak (sub-gross, gate-edge class)
    prior_sigma = np.array([0.24, 0.24, 0.24])
    for floor in FLOORS:
        r = p_leak_pass(floor, R_wb, prior_sigma, 4.0, leak_off)
        out["leak_behaviour"][f"{floor:.3f}"] = r
        print(f"  f{floor:.3f}: d2={r['d2']:6.2f} passes={r['passes']!s:>5} "
              f"state_pull={r['state_pull_m']:.3f} m (in-plane {r['state_pull_inplane_m']:.3f} m)")

    (HERE / "c4_chi2_overrej_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'c4_chi2_overrej_results.json'}")


if __name__ == "__main__":
    main()
