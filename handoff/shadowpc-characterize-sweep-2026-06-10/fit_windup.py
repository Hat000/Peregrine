"""Nonlinear PI-windup model fit for the sim's inner rate loop (payload-1 analysis,
SHADOWPC-CHARACTERIZE-SWEEP 2026-06-10).

WHY: the per-magnitude LTI 2nd-order fits do NOT cross-generalize (the saturated regime
is state-dependent -- check_fits.py: sweep mag-3.14 params give RMSE 6.5 on the S1.2
tumble where the tumble's own LTI fit gives 0.25, and vice versa the tumble params are
~10x off on from-rest steps). The measured signatures point at one mechanism:

  * small signal: ~1st-order tau ~19 ms with a real ~10% overshoot,
  * saturated from rest: slew-limited rise (~250 rad/s^2), peak ~1.45x target,
    then a SUSTAINED ~+2.5 rad/s elevation above the DC target (wound integrator),
  * long steps eventually settle back to the validated DC gain (yaw plateaus).

MODEL (per axis; target = G_shipped * rate_sign * cmd, G FIXED as everywhere else):

    e      = target - omega
    I     += e * dt;  I = clip(I, +-I_max)        # integrator with anti-windup clamp
    omega += dt * clip(Kp*e + Ki*I, +-alpha_max)  # torque-limited

Params (Kp, Ki, I_max, alpha_max) + shared input delay d (grid). Fit jointly over ALL
sweep magnitudes; cross-evaluated on the S1.2 tumble (never fit unless --with-tumble).

Usage (repo root):
  .venv\\Scripts\\python.exe handoff\\shadowpc-characterize-sweep-2026-06-10\\fit_windup.py
  [--with-tumble]   include the tumble windows in the fit (needs torch + checkpoint)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "rl"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "handoff" / "shadowpc-2ndorder-resysid-2026-06-10"))

import fit_2nd_order as f2
import fit_sweep as fs

_AXES = f2._AXES


def sim_windup(t, u_t, axis, Kp, Ki, Imax, amax, w0, n_sub=6):
    """Integrate the PI-windup model over telemetry timestamps with ZOH input."""
    G = f2._G_SHIPPED[axis]
    w, I = w0, 0.0
    out = np.empty(len(t))
    out[0] = w
    for k in range(1, len(t)):
        dt = (t[k] - t[k - 1]) / n_sub
        tgt = G * f2._RATE_SIGN[axis] * u_t[k - 1]
        for _ in range(n_sub):
            e = tgt - w
            I = min(max(I + e * dt, -Imax), Imax)
            w = w + dt * min(max(Kp * e + Ki * I, -amax), amax)
        out[k] = w
    return out


def windup_rmse(windows, axis, params, delay):
    Kp, Ki, Imax, amax = params
    errs, n = 0.0, 0
    for t, wmeas, (t_u, u) in windows:
        u_t = f2._zoh(t, t_u, u[:, axis], delay)
        pred = sim_windup(t, u_t, axis, Kp, Ki, Imax, amax, wmeas[0, axis])
        e = float(np.sum((pred - wmeas[:, axis]) ** 2))
        if not np.isfinite(e):
            return 1e9
        errs += e
        n += len(t)
    return np.sqrt(errs / max(n, 1))


def fit_axis_windup(window_sets, axis, delay, x0s):
    """Equal-weight sum of per-set RMSE (a set = one magnitude's windows), so the
    many small-signal windows don't drown the few saturated ones."""

    def loss(x):
        p = np.exp(x)
        return sum(windup_rmse(ws, axis, p, delay) for ws in window_sets)

    best = None
    for x0 in x0s:
        r = minimize(loss, np.log(np.asarray(x0)), method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 1200})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x), best.fun


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-tumble", action="store_true")
    ap.add_argument("--checkpoint", default=r"C:\Users\Shadow\Downloads\stage1_inc1_actor.pth")
    args = ap.parse_args()
    runs_dir = _REPO / "data" / "runs"

    # window sets: one per (mag) so each magnitude weighs equally in the loss
    loaded = {name: fs.load_run(runs_dir / name) for name in fs.RUNS}
    sets_by_axis = {0: [], 1: [], 2: []}
    for mag in fs.MAGS:
        for axis in range(3):
            wins = []
            for name, (m, axes) in fs.RUNS.items():
                if m != mag or _AXES[axis] not in axes:
                    continue
                t_w, w, rows = loaded[name]
                post = 0.8 if mag >= 1.0 else 1.2
                wins += fs.step_windows(t_w, w, rows, axis, post_s=post)
            if wins:
                sets_by_axis[axis].append((mag, wins))

    # the S1.2 tumble (cross-eval, optionally in-fit)
    tumble = None
    try:
        f1 = runs_dir / "20260610_205414_rl_s12_f1"
        t_w, w = f2.load_rates(f1)
        t_u, u = f2.reconstruct_cmds_flight1(f1, args.checkpoint, t1=10.99)
        tumble = f2.build_windows(t_w, w, t_u, u, max_len_s=2.0)
    except Exception as exc:
        print(f"(tumble unavailable: {exc})")

    x0s = [(53.0, 5000.0, 0.03, 250.0), (80.0, 3000.0, 0.05, 200.0),
           (53.0, 2000.0, 0.10, 300.0), (40.0, 8000.0, 0.02, 150.0)]
    print("=== PI-windup joint fit per axis (G fixed at shipped; all magnitudes) ===")
    print(f"{'axis':<6} {'Kp(1/s)':>8} {'Ki(1/s^2)':>10} {'I_max':>7} {'amax(r/s^2)':>12} "
          f"{'d(ms)':>6} {'loss':>7}")
    fits = {}
    for axis in range(3):
        wsets = [w for _m, w in sets_by_axis[axis]]
        if args.with_tumble and tumble:
            wsets = wsets + [tumble]
        best = None
        for delay in (0.0, 0.010, 0.015, 0.020, 0.030):
            p, fv = fit_axis_windup(wsets, axis, delay, x0s)
            if best is None or fv < best[1]:
                best = (p, fv, delay)
        (Kp, Ki, Imax, amax), fv, d = best
        fits[axis] = best
        print(f"{_AXES[axis]:<6} {Kp:>8.1f} {Ki:>10.0f} {Imax:>7.4f} {amax:>12.0f} "
              f"{d*1e3:>6.0f} {fv:>7.3f}")

    print("\n=== per-magnitude RMSE under the joint PI-windup fit (vs the per-mag LTI fits) ===")
    print(f"{'axis':<6} {'mag':>5} {'windup':>8}")
    for axis in range(3):
        (p, _fv, d) = fits[axis]
        for mag, wins in sets_by_axis[axis]:
            print(f"{_AXES[axis]:<6} {mag:>5.2f} {windup_rmse(wins, axis, p, d):>8.3f}")

    if tumble:
        print("\n=== tumble cross-eval (RMSE rad/s; cc6921d-sat LTI for reference) ===")
        cc = {0: (21.0, 0.393, 0.015), 1: (28.3, 0.467, 0.030), 2: (11.3, 0.273, 0.0)}
        for axis in range(3):
            (p, _fv, d) = fits[axis]
            e_w = windup_rmse(tumble, axis, p, d)
            wn, z, dc = cc[axis]
            e_c = f2.axis_rmse(tumble, axis, "second", (f2._G_SHIPPED[axis], wn, z), dc)
            print(f"{_AXES[axis]:<6} windup {e_w:7.3f} | cc6921d-sat LTI {e_c:7.3f}")

        # peak validation on the tumble
        print("\n=== tumble peak |rate| under the windup model ===")
        for axis in range(3):
            (p, _fv, d) = fits[axis]
            meas = pk = 0.0
            for t, wmeas, (t_u2, u2) in tumble:
                meas = max(meas, float(np.max(np.abs(wmeas[:, axis]))))
                u_t = f2._zoh(t, t_u2, u2[:, axis], d)
                pk = max(pk, float(np.max(np.abs(sim_windup(
                    t, u_t, axis, *p, wmeas[0, axis])))))
            print(f"{_AXES[axis]:<6} measured {meas:5.2f}  windup-model {pk:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
