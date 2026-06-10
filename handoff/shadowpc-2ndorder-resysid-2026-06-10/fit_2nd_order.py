"""2nd-order inner-rate-loop re-system-ID (S1.2 follow-up, ShadowPC 2026-06-10).

WHY: the live VQ1 sim's inner rate loop OVERSHOOTS. On saturated step commands
(the RL tumble, run 20260610_205414_rl_s12_f1) the realized body rate peaked at
9.77 rad/s where the shipped first-order model (rate_gain 2.501 x cmd 3.14,
tau 0.019 s) can never exceed 7.85 -- and the rate kept RISING ~35 ms after the
command flipped sign (momentum), which a first-order lag cannot do at all. This
under-modelled transient is one of the four S1.2 transfer blockers
(handoff/shadowpc-s12-rl-live-2026-06-10/).

MODEL: keep the existing DC structure target = rate_gain * rate_sign * cmd and
replace the first-order lag with a damped second-order response per axis:

    omega_ddot = wn^2 * (target - omega) - 2 * zeta * wn * omega_dot

(first-order is the zeta->inf limit with 2*zeta/wn = tau held fixed). A shared
input transport delay `d` (ZOH command shifted by d) is co-fit so the 2nd-order
poles are not forced to absorb pure latency.

DATA (all with KNOWN inputs; wire cmd = FRD body-rate setpoint):
  A. rate1 / rate2   open-loop +-0.3 rad/s doublets (commands.jsonl, 2026-06-03)
                      -> small-signal anchor (the regime the 1st-order fit came from)
  B. course_60s      the canonical 6/6 closed-loop run, 100 Hz commands.jsonl
                      -> mid-amplitude, rich excitation
  C. rl_s12_f1       the S1.2 tumble: commands RECONSTRUCTED by replaying the
                      deterministic policy (tanh(actor_mean)) over the recorded
                      telemetry -- the policy ran bang-bang (outputs pinned at
                      +-1), so the wire command is known wherever consecutive
                      replay samples agree; cross-checked against the live
                      console print at t=10.75 ([+3.14,-3.14,+3.14]).
                      -> the large-signal overshoot regime
  Measured output everywhere: ODOMETRY angular_rate * [-1,-1,1] (raw -> true FRD,
  re-verified in the S1.2 session: corr +-0.998 vs quat-derived rates).

FIT: per-axis (G, wn, zeta) + shared delay d, simulation error (RMSE of the
integrated model trace vs measured omega over each window, datasets weighted
equally). Baselines: the SHIPPED first-order params and a REFIT first-order
(G, tau) + d. Validation: peak realized rates in the tumble window.

Usage (repo root):
  .venv\\Scripts\\python.exe handoff\\shadowpc-2ndorder-resysid-2026-06-10\\fit_2nd_order.py
  [--no-flight1]   skip dataset C (no torch / checkpoint needed then)
  [--checkpoint PATH]  default C:\\Users\\Shadow\\Downloads\\stage1_inc1_actor.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "rl"))
sys.path.insert(0, str(_REPO / "scripts"))

from scipy.optimize import minimize

from racer.recording import RecordingReader

_ODO_RAW_TO_TRUE = np.array([-1.0, -1.0, 1.0])
_RATE_SIGN = np.array([1.0, 1.0, -1.0])          # PHYSICS: yaw cmd inverted (rl_plant)
_G_SHIPPED = np.array([2.501, 2.504, 2.231])
_TAU_SHIPPED = 0.0190
_AXES = ["roll", "pitch", "yaw"]


# ---------------------------------------------------------------- data loading
def load_rates(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """ODOMETRY -> (t_s, w_true (N,3)). Keeps ALL samples; caller selects ranges.
    Sim time resets between race epochs -> split later by command time overlap."""
    ts, ws = [], []
    for m in RecordingReader(run).iter_mavlink():
        if m.get_type() == "ODOMETRY":
            ts.append(m.time_usec * 1e-6)
            ws.append([m.rollspeed, m.pitchspeed, m.yawspeed])
    return np.asarray(ts), np.asarray(ws) * _ODO_RAW_TO_TRUE


def load_cmds_sysid(run: Path) -> tuple[np.ndarray, np.ndarray]:
    t, u = [], []
    for line in (run / "commands.jsonl").read_text().splitlines():
        r = json.loads(line)
        t.append(r["sim_time_ns"] * 1e-9)
        u.append(r["cmd"])
    return np.asarray(t), np.asarray(u)


def load_cmds_course(run: Path) -> tuple[np.ndarray, np.ndarray]:
    t, u = [], []
    for line in (run / "commands.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("body_rate") is None:
            continue
        t.append(r["sim_t"] * 1e-9)
        u.append(r["body_rate"])
    return np.asarray(t), np.asarray(u)


def reconstruct_cmds_flight1(run: Path, checkpoint: str,
                             t0: float = 10.76, t1: float = 11.15
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Replay the deterministic policy over recorded telemetry -> wire commands.
    Valid because the policy ran bang-bang (saturated) through the window; the
    +-13 ms transition-time ambiguity (75 Hz replay vs the 30 Hz live tick) is
    small vs the ~100 ms transient being fitted."""
    from types import SimpleNamespace

    import torch
    from racer.frames import R_world_from_body, euler_from_quat_wxyz

    import fly_rl

    actor = fly_rl.load_actor(checkpoint)
    t, u = [], []
    for m in RecordingReader(run).iter_mavlink():
        if m.get_type() != "ODOMETRY":
            continue
        tt = m.time_usec * 1e-6
        if not (t0 <= tt <= t1):
            continue
        q = np.array([m.q[0], m.q[1], m.q[2], m.q[3]])
        r, p, y = euler_from_quat_wxyz(q)
        s = SimpleNamespace(
            position_ned=np.array([m.x, m.y, m.z]),
            velocity_ned=R_world_from_body(r, p, y) @ np.array([m.vx, m.vy, m.vz]),
            orientation_ned_wxyz=q,
            angular_rate_body=np.array([m.rollspeed, m.pitchspeed, m.yawspeed]),
        )
        # feedback value barely matters under saturation; 5.0 = saturated thrust
        obs = fly_rl.build_obs(s, 0, 5.0, virtual_flip=True)
        rate_frd, _, _ = fly_rl.policy_step(actor, obs, 0.0, True)
        t.append(tt)
        u.append(rate_frd)
    return np.asarray(t), np.asarray(u)


def build_windows(t_w, w, t_u, u, max_len_s=2.0, gap_s=0.2):
    """-> list of (t (N,), w (N,3), u_zoh (N,3)): telemetry samples covered by
    commands, split at command gaps and capped in length."""
    out = []
    if len(t_u) < 2:
        return out
    # contiguous command spans
    splits = np.where(np.diff(t_u) > gap_s)[0]
    starts = np.r_[0, splits + 1]
    ends = np.r_[splits, len(t_u) - 1]
    for s0, e0 in zip(starts, ends):
        lo, hi = t_u[s0], t_u[e0]
        m = (t_w >= lo) & (t_w <= hi)
        if m.sum() < 8:
            continue
        tt, ww = t_w[m], w[m]
        n_per = max(8, int(max_len_s / max(np.median(np.diff(tt)), 1e-3)))
        for k0 in range(0, len(tt), n_per):
            sl = slice(k0, min(k0 + n_per, len(tt)))
            if sl.stop - sl.start < 8:
                continue
            out.append((tt[sl], ww[sl], (t_u[s0:e0 + 1], u[s0:e0 + 1])))
    return out


# ---------------------------------------------------------------- simulation
def _zoh(t_q, t_u, u, delay):
    idx = np.clip(np.searchsorted(t_u, t_q - delay, side="right") - 1, 0, len(t_u) - 1)
    return u[idx]


def sim_first_order(t, u_t, axis, G, tau, w0, n_sub=8):
    w = w0
    out = np.empty(len(t))
    out[0] = w
    for k in range(1, len(t)):
        dt = (t[k] - t[k - 1]) / n_sub
        tgt = G * _RATE_SIGN[axis] * u_t[k - 1]
        a = 1.0 - np.exp(-dt / max(tau, 1e-4))
        for _ in range(n_sub):
            w = w + a * (tgt - w)
        out[k] = w
    return out


def sim_second_order(t, u_t, axis, G, wn, zeta, w0, wd0, n_sub=8):
    w, wd = w0, wd0
    out = np.empty(len(t))
    out[0] = w
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, len(t)):
            dt = (t[k] - t[k - 1]) / n_sub
            tgt = G * _RATE_SIGN[axis] * u_t[k - 1]
            for _ in range(n_sub):
                wd = wd + dt * (wn * wn * (tgt - w) - 2.0 * zeta * wn * wd)
                w = w + dt * wd
            out[k] = w
    return out


def axis_rmse(windows, axis, model, params, delay):
    errs, n = 0.0, 0
    for t, wmeas, (t_u, u) in windows:
        u_t = _zoh(t, t_u, u[:, axis], delay)
        w0 = wmeas[0, axis]
        if model == "first":
            pred = sim_first_order(t, u_t, axis, params[0], params[1], w0)
        else:
            wd0 = (wmeas[1, axis] - wmeas[0, axis]) / max(t[1] - t[0], 1e-3)
            pred = sim_second_order(t, u_t, axis, params[0], params[1], params[2], w0, wd0)
        with np.errstate(over="ignore", invalid="ignore"):
            e = float(np.sum((pred - wmeas[:, axis]) ** 2))
        if not np.isfinite(e):
            return 1e9          # unstable Euler during optimizer exploration
        errs += e
        n += len(t)
    return np.sqrt(errs / max(n, 1))


def dataset_loss(datasets, axis, model, params, delay):
    """Equal-weight sum of per-dataset RMSE (keeps the small tumble set influential)."""
    return sum(axis_rmse(wins, axis, model, params, delay) for _, wins in datasets)


# ---------------------------------------------------------------- fitting
def fit_axis(datasets, axis, model, delay, x0s):
    best = None
    for x0 in x0s:
        r = minimize(lambda x: dataset_loss(datasets, axis, model, np.exp(x), delay),
                     np.log(np.asarray(x0)), method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x), best.fun


def fit_axis_fixedG(windows_list, axis, delay, x0s):
    """Fit (wn, zeta) with G FIXED at the shipped steady gain (the validated
    small-signal DC anchor; the tumble never reaches steady state at saturation,
    so letting G float lets the optimizer fake the overshoot as extra gain)."""
    G = _G_SHIPPED[axis]
    best = None
    for x0 in x0s:
        r = minimize(lambda x: sum(axis_rmse(w, axis, "second",
                                             (G, np.exp(x[0]), np.exp(x[1])), delay)
                                   for w in windows_list),
                     np.log(np.asarray(x0)), method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x), best.fun


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=r"C:\Users\Shadow\Downloads\stage1_inc1_actor.pth")
    ap.add_argument("--no-flight1", action="store_true")
    ap.add_argument("--runs-dir", default=str(_REPO / "data" / "runs"))
    ap.add_argument("--tumble-end", type=float, default=10.99,
                    help="tumble fit window end (sim s). Beyond ~10.99 the drone "
                         "is past 90 deg tilt = the sim's yaw-spin anomaly regime; "
                         "rate-loop ID there would fit the anomaly, not the loop.")
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    smallmid = []  # (name, windows) -- the regime the shipped 1st-order came from
    for name in ("20260603_160653_rate1", "20260603_160940_rate2"):
        t_w, w = load_rates(runs / name)
        t_u, u = load_cmds_sysid(runs / name)
        smallmid.append((name.split("_")[-1], build_windows(t_w, w, t_u, u)))
    t_w, w = load_rates(runs / "20260607_194615_course_60s")
    t_u, u = load_cmds_course(runs / "20260607_194615_course_60s")
    smallmid.append(("course", build_windows(t_w, w, t_u, u)))

    f1 = runs / "20260610_205414_rl_s12_f1"
    t_w, w = load_rates(f1)
    t_u, u = reconstruct_cmds_flight1(f1, args.checkpoint, t1=args.tumble_end)
    tumble = build_windows(t_w, w, t_u, u, max_len_s=2.0)
    datasets = smallmid + [("tumble", tumble)]
    for name, wins in datasets:
        n = sum(len(t) for t, _, _ in wins)
        print(f"dataset {name:<8} windows={len(wins):3d}  samples={n}")

    # ---- fit A: saturated regime (tumble only), G fixed ----
    # ---- fit B: small/mid regime (rate1+rate2+course), G fixed ----
    x0s = [(30.0, 0.4), (25.0, 0.7), (50.0, 1.0), (15.0, 0.3)]
    fitA, fitB = {}, {}
    for axis in range(3):
        bestA = bestB = None
        for delay in (0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040):
            pA, fA = fit_axis_fixedG([tumble], axis, delay, x0s)
            if bestA is None or fA < bestA[1]:
                bestA = (pA, fA, delay)
            pB, fB = fit_axis_fixedG([w for _, w in smallmid], axis, delay, x0s)
            if bestB is None or fB < bestB[1]:
                bestB = (pB, fB, delay)
        fitA[axis], fitB[axis] = bestA, bestB

    print("\n=== recovered 2nd-order params (G FIXED at shipped steady gain) ===")
    print("regime    axis    G_fix    wn(rad/s)   zeta   d(ms)   tau_eq(ms)  step-overshoot")
    for tag, fits in (("SATURATED", fitA), ("small/mid", fitB)):
        for axis in range(3):
            (wn, z), err, d = fits[axis]
            ovs = 100 * np.exp(-np.pi * z / np.sqrt(1 - z * z)) if z < 1 else 0.0
            print(f"{tag:<9} {_AXES[axis]:<6} {_G_SHIPPED[axis]:>6.3f} {wn:>10.1f} "
                  f"{z:>7.3f} {d*1e3:>6.0f} {2*z/wn*1e3:>11.1f} {ovs:>10.0f}%")

    # ---- cross-evaluation matrix ----
    print("\n=== per-dataset RMSE (rad/s): shipped-1st vs 2nd(sat-fit) vs 2nd(small-fit) ===")
    print(f"{'dataset':<9} {'axis':<6} {'shipped1st':>11} {'2nd-sat':>9} {'2nd-small':>10}")
    for name, wins in datasets:
        for axis in range(3):
            e0 = axis_rmse(wins, axis, "first", (_G_SHIPPED[axis], _TAU_SHIPPED), 0.0)
            (wnA, zA), _, dA = fitA[axis]
            eA = axis_rmse(wins, axis, "second", (_G_SHIPPED[axis], wnA, zA), dA)
            (wnB, zB), _, dB = fitB[axis]
            eB = axis_rmse(wins, axis, "second", (_G_SHIPPED[axis], wnB, zB), dB)
            print(f"{name:<9} {_AXES[axis]:<6} {e0:>11.4f} {eA:>9.4f} {eB:>10.4f}")

    # ---- validation: tumble peak |rate| (the 9.7 the 1st-order could not make) ----
    print("\n=== tumble-window peak |rate| (rad/s) ===")
    print(f"{'axis':<6} {'measured':>9} {'shipped1st':>11} {'2nd-sat':>9}")
    for axis in range(3):
        meas = pk_s = pk_A = 0.0
        for t, wmeas, (t_u2, u2) in tumble:
            meas = max(meas, float(np.max(np.abs(wmeas[:, axis]))))
            w0 = wmeas[0, axis]
            wd0 = (wmeas[1, axis] - wmeas[0, axis]) / max(t[1] - t[0], 1e-3)
            pk_s = max(pk_s, float(np.max(np.abs(sim_first_order(
                t, _zoh(t, t_u2, u2[:, axis], 0.0), axis,
                _G_SHIPPED[axis], _TAU_SHIPPED, w0)))))
            (wnA, zA), _, dA = fitA[axis]
            pk_A = max(pk_A, float(np.max(np.abs(sim_second_order(
                t, _zoh(t, t_u2, u2[:, axis], dA), axis,
                _G_SHIPPED[axis], wnA, zA, w0, wd0)))))
        print(f"{_AXES[axis]:<6} {meas:>9.2f} {pk_s:>11.2f} {pk_A:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
