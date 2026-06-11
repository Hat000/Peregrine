"""MLE for the honest fix-covariance model + offline gate-trade-off sweep.

Model per solved 4-corner fix (world frame, zero-mean -- the live gate's assumption):

    r_i ~ N(0,  K * Sigma_pnp_i  +  sigma_theta^2 (|L_i|^2 I - L_i L_i^T)  +  sigma_0^2 I)

  K            production PNP_FIX_COV_INFLATION (=2.0), held fixed
  sigma_theta  the given-attitude / chain angular 1-sigma (the lever-arm term) -- THE deliverable
  sigma_0      an isotropic FLOOR covering the measured constant systematics (map-vertical ~0.3 m,
               close-range depth bias, per-gate lateral wander) that no angular term can cover at
               short range (lever -> 0) -- candidate second constant

Fits (sigma_theta) alone and (sigma_theta, sigma_0) jointly by grid+polish MLE on the offered
(depth-sane) non-catastrophic fixes, then replays the chi2_0.999 gate over a (sigma_theta,
sigma_0) grid to print the measured trade-off: good-fix rejection (<1 m and <3 m) vs
catastrophic leak -- the acceptance numbers, computed offline from the instrumented dumps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                      # noqa: E402
from racer.localization import PNP_FIX_COV_INFLATION, P3P_FIX_COV_INFLATION  # noqa: E402

HERE = Path(__file__).resolve().parent
CHI2 = 16.27
K = PNP_FIX_COV_INFLATION


def load(pattern: str):
    rows = []
    for gi in range(6):
        p = HERE / (pattern % gi)
        d = json.loads(p.read_text())
        assert abs(d["cov_inflation"] - K) < 1e-9, "dump made at non-production K"
        for r in d["rows"]:
            if not (r.get("associated") and "world_fix_err_m" in r):
                continue
            R_wb = F.R_world_from_body(*F.euler_from_quat_wxyz(np.asarray(r["odo_q_wxyz"], float)))
            R_wc = R_wb @ F.R_camera_from_body().T
            lever = R_wc @ np.asarray(r["t_cam_solved"], float)
            rows.append(dict(
                gate_id=r["gate_id"], err=r["world_fix_err_m"], off=np.asarray(r["off_ned"], float),
                lever=lever, range_m=r["true_range_m"], n_corners=r["n_corners"],
                cov_pnp=(np.asarray(r["cov_t_world"], float) if r.get("cov_t_world") is not None
                         else None),
                range_ok=r.get("range_ok", True), maha_dump=r["maha"]))
    return rows


def fix_cov(r, sig_th, sig_0):
    """Replicate localization.gate_pose_to_world_position + the harness P3P inflation."""
    if r["cov_pnp"] is not None:
        cov = K * r["cov_pnp"]
    else:
        cov = 0.09 * np.eye(3)                       # default_position_std = 0.3
    L = r["lever"]
    cov = cov + sig_th ** 2 * (float(L @ L) * np.eye(3) - np.outer(L, L))
    cov = cov + sig_0 ** 2 * np.eye(3)
    if r["n_corners"] < 4:
        cov = cov * P3P_FIX_COV_INFLATION
    return cov


def nll(rows, sig_th, sig_0):
    tot = 0.0
    for r in rows:
        S = fix_cov(r, sig_th, sig_0)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return np.inf
        tot += 0.5 * (logdet + float(r["off"] @ np.linalg.solve(S, r["off"])))
    return tot / len(rows)


def gate_replay(rows, sig_th, sig_0):
    offered = [r for r in rows if r["range_ok"]]
    res = {}
    for thr, nm in ((1.0, "g1"), (3.0, "g3")):
        good = [r for r in offered if r["err"] < thr]
        rej = sum(float(r["off"] @ np.linalg.solve(fix_cov(r, sig_th, sig_0), r["off"])) > CHI2
                  for r in good)
        res[nm] = (rej, len(good))
    cat = [r for r in offered if r["err"] >= 3.0]
    leak = sum(float(r["off"] @ np.linalg.solve(fix_cov(r, sig_th, sig_0), r["off"])) <= CHI2
               for r in cat)
    res["cat"] = (leak, len(cat))
    return res


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "fixedframe_g%d.json"
    rows = load(pattern)
    n4 = [r for r in rows if r["n_corners"] == 4 and r["cov_pnp"] is not None]
    print(f"{pattern}: solved {len(rows)} (4-corner w/ cov {len(n4)}), "
          f"offered {sum(r['range_ok'] for r in rows)}")

    # sanity: replicate the dump's maha at production settings (sig_th=1deg, sig_0=0)
    d1 = np.deg2rad(1.0)
    chk = [abs(float(r["off"] @ np.linalg.solve(fix_cov(r, d1, 0.0), r["off"])) - r["maha_dump"])
           for r in rows if np.isfinite(r["maha_dump"])]
    print(f"maha replication vs dump: max abs diff {max(chk):.2e}  (must be ~0)")

    # MLE on offered non-catastrophic (the population the gate must pass)
    fit_rows = [r for r in rows if r["range_ok"] and r["err"] < 3.0 and r["n_corners"] == 4
                and r["cov_pnp"] is not None]
    print(f"MLE population: {len(fit_rows)} offered non-catastrophic 4-corner fixes")

    r1 = minimize(lambda x: nll(fit_rows, abs(x[0]), 0.0), x0=[np.deg2rad(1.5)],
                  method="Nelder-Mead")
    s1 = abs(r1.x[0])
    print(f"\n1-param MLE (sigma_theta only):  sigma_theta = {np.degrees(s1):.2f} deg   "
          f"nll {r1.fun:.4f}")
    r2 = minimize(lambda x: nll(fit_rows, abs(x[0]), abs(x[1])),
                  x0=[np.deg2rad(1.0), 0.3], method="Nelder-Mead")
    s2t, s2f = abs(r2.x[0]), abs(r2.x[1])
    print(f"2-param MLE (theta + floor):     sigma_theta = {np.degrees(s2t):.2f} deg   "
          f"floor = {s2f:.3f} m   nll {r2.fun:.4f}")

    # profile: how flat is the likelihood?
    print("\nNLL profile (rows: sigma_theta deg, cols: floor m):")
    floors = [0.0, 0.15, 0.25, 0.35, 0.5]
    print("         " + "".join(f"{f:>8.2f}" for f in floors))
    for th in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        line = f"  {th:5.2f}  "
        for f0 in floors:
            line += f"{nll(fit_rows, np.deg2rad(th), f0):8.3f}"
        print(line)

    # gate replay sweep: the acceptance numbers, offline
    print("\nGATE REPLAY (chi2_0.999, production K=2):  good-rej<1m  good-rej<3m  cat-leak")
    print(f"  {'sig_th':>6} {'floor':>6} | {'<1m':>12} {'<3m':>12} {'leak':>10}")
    base = gate_replay(rows, d1, 0.0)
    print(f"  {1.0:6.2f} {0.0:6.2f} | {base['g1'][0]:3d}/{base['g1'][1]:3d} ({100*base['g1'][0]/base['g1'][1]:3.0f}%) "
          f"{base['g3'][0]:3d}/{base['g3'][1]:3d} ({100*base['g3'][0]/base['g3'][1]:3.0f}%) "
          f"{base['cat'][0]}/{base['cat'][1]}   <- production baseline")
    for th in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
        for f0 in (0.0, 0.25, 0.35, 0.5):
            g = gate_replay(rows, np.deg2rad(th), f0)
            print(f"  {th:6.2f} {f0:6.2f} | {g['g1'][0]:3d}/{g['g1'][1]:3d} ({100*g['g1'][0]/g['g1'][1]:3.0f}%) "
                  f"{g['g3'][0]:3d}/{g['g3'][1]:3d} ({100*g['g3'][0]/g['g3'][1]:3.0f}%) "
                  f"{g['cat'][0]}/{g['cat'][1]}")


if __name__ == "__main__":
    main()
