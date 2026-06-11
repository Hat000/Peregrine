"""P3 joint vertical fit: collective map K(thr) + vertical drag D(vz), fit TOGETHER.

The per-segment regressions in fit_aero.py conflate K and drag (short steps, vz/thr
colinearity). Here every near-level airborne sample from the vertical-rich runs enters one
least-squares system:

    a_z - g = -K(thr)*cos(roll)cos(pitch) + D(vz)
    K(thr)  = piecewise-linear over knots at the probed stick levels
    D(vz)   = -c_dn*|vz|*vz  for vz>0 (descending)   [drag opposes velocity; NED +z down]
              -c_up*|vz|*vz  for vz<0 (climbing)      (directional quadratic; the horizontal
                                                       fit found quad, so quad here too)

Excluded: on-pad samples (z > -0.3), the first 0.15 s of each constant-thrust step (motor
spin-up transient), |tilt| > 25 deg. coll_speed (fast horizontal) is NOT fit -- it is
EVALUATED under the hover-family fit: a K shift with airspeed = translational-lift coupling.

Usage: .venv\\Scripts\\python.exe handoff\\shadowpc-twin-falsify-2026-06-10\\fit_vertical_joint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runs as R

KNOTS = np.array([0.0, 0.10, 0.15, 0.20, 0.2656, 0.32, 0.40, 0.45, 0.55, 0.60, 0.80, 1.00])
FIT_RUNS = ["vert_mid", "vert_high", "coll_hover", "det1", "det2", "det3"]
EVAL_RUNS = ["coll_speed"]


def knot_weights(thr: np.ndarray) -> np.ndarray:
    """(N, K) linear-interp weights of each sample's thr onto the knots."""
    W = np.zeros((len(thr), len(KNOTS)))
    idx = np.clip(np.searchsorted(KNOTS, thr) - 1, 0, len(KNOTS) - 2)
    lo, hi = KNOTS[idx], KNOTS[idx + 1]
    w = (thr - lo) / (hi - lo)
    W[np.arange(len(thr)), idx] = 1.0 - w
    W[np.arange(len(thr)), idx + 1] = w
    return W


def gather(labels):
    A_rows, y_rows, meta = [], [], []
    for lb in labels:
        run = R.load(lb)
        acc = R.accel(run)
        ct = np.cos(run.rpy[:, 0]) * np.cos(run.rpy[:, 1])
        # step-onset mask: drop 0.15 s after any thrust jump > 0.02
        jump = np.abs(np.diff(run.thr, prepend=run.thr[0])) > 0.02
        recent = np.zeros(len(run.t), dtype=bool)
        for i in np.where(jump)[0]:
            recent |= (run.t >= run.t[i]) & (run.t < run.t[i] + 0.15)
        m = ((run.pos[:, 2] < -0.3) & (ct > np.cos(np.radians(25.0))) & ~recent)
        m[:3] = m[-3:] = False
        vz = run.vel[m, 2]
        W = knot_weights(run.thr[m]) * (-ct[m][:, None])          # K columns
        dn = np.where(vz > 0, -np.abs(vz) * vz, 0.0)              # c_dn column
        up = np.where(vz < 0, -np.abs(vz) * vz, 0.0)              # c_up column
        A_rows.append(np.column_stack([W, dn, up]))
        y_rows.append(acc[m, 2] - R.G)
        meta.append((lb, int(m.sum())))
    return np.vstack(A_rows), np.concatenate(y_rows), meta


def main():
    A, y, meta = gather(FIT_RUNS)
    print("fit samples per run:", meta, "total", len(y))
    # ridge-stabilize knots that may be thinly covered
    lam = 1e-3
    AtA = A.T @ A + lam * np.eye(A.shape[1])
    coef = np.linalg.solve(AtA, A.T @ y)
    K = coef[:len(KNOTS)]
    c_dn, c_up = coef[len(KNOTS)], coef[len(KNOTS) + 1]
    resid = y - A @ coef
    print(f"\nfit rms = {np.std(resid):.3f} m/s^2   c_dn(descend) = {c_dn:.4f} /m   "
          f"c_up(climb) = {c_up:.4f} /m")
    print("\n  thr     K_meas   twin g*thr/0.2656   ratio")
    for k, kv in zip(KNOTS, K):
        tw = R.G * k / 0.2656
        print(f"  {k:5.3f} {kv:9.2f} {tw:12.2f}      {kv/tw if tw else float('nan'):9.2f}"
              if tw else f"  {k:5.3f} {kv:9.2f} {tw:12.2f}          --")
    # sanity: hover knot should be ~g
    i_h = int(np.argmin(np.abs(KNOTS - 0.2656)))
    print(f"\n  hover knot K({KNOTS[i_h]}) = {K[i_h]:.2f} (g = {R.G:.2f})")

    # per-run residual check
    print("\nper-run rms under the joint fit:")
    for lb in FIT_RUNS + EVAL_RUNS:
        Ae, ye, _ = gather([lb])
        r = ye - Ae @ coef
        tag = "FIT " if lb in FIT_RUNS else "EVAL"
        print(f"  [{tag}] {lb:12s} n={len(ye):5d} rms={np.std(r):.3f} bias={np.mean(r):+.3f} m/s^2")

    np.save(Path(__file__).parent / "vert_fit_coef.npy",
            {"knots": KNOTS, "K": K, "c_dn": c_dn, "c_up": c_up}, allow_pickle=True)
    print("\nsaved vert_fit_coef.npy")


if __name__ == "__main__":
    main()
