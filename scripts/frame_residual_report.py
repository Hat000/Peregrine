"""frame_residual_report.py -- standing per-axis frame/force residual report for live
RL recordings (LAPTOP-FRAME-AUDIT 2026-06-12; re-run this after EVERY live session).

For each session dir containing a debug_obs.jsonl (fly_rl --debug-obs):
  1. FORCE RESIDUAL, regime-binned: measured world accel (central FD of pristine
     vel_ned -- the external invariant) minus the plant force model evaluated at the
     TRUE attitude (ODOMETRY quat conjugated by [1,-1,1,-1]), per axis, binned by
     airspeed x tilt x collective. Systematic structure here = plant gap; an EAST
     sign error = a mirror crept back in.
  2. MIRROR CANARY: the same East residual under the AS-IS quat. Healthy telemetry
     shows as-is FAR WORSE than true at bank (anti-correlated); if as-is ever beats
     true, the sim's telemetry convention changed -- stop and re-audit.
  3. RATE CANARY: quat-FD(true attitude) vs -w_raw per-axis gain (expect ~1.0).
  4. OPEN-LOOP REPLAY (optional, --replay): seed the mixer twin at tick k0 and drive
     the recorded wire commands; report per-axis velocity error after N ticks.

Usage:
  .venv\\Scripts\\python.exe scripts\\frame_residual_report.py data/runs/<session> [...]
  .venv\\Scripts\\python.exe scripts\\frame_residual_report.py --replay 36 data/runs/<session>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.frames import ODO_QUAT_TRUE_CONJ_WXYZ
from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            COLL_MAP_THR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED, PlantParams, PlantState,
                            QUAD_DRAG_C2_MEASURED, SUPER_RATE_S_MEASURED,
                            step as plant_step)

_G = 9.80665
_SPEED_BINS = [(0, 4), (4, 8), (8, 12), (12, 18), (18, 40)]
_TILT_BINS = [(0, 15), (15, 35), (35, 90)]
_COLL_BINS = [(0.0, 0.2), (0.2, 0.35), (0.35, 1.01)]


def load(session: Path) -> dict | None:
    f = session / "debug_obs.jsonl"
    if not f.exists():
        print(f"  !! no debug_obs.jsonl in {session}", file=sys.stderr)
        return None
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r.get("type") != "header"]
    if len(rows) < 10:
        return None
    return dict(
        t=np.array([r["t_mono"] for r in rows]),
        pos=np.array([r["pos_ned"] for r in rows]),
        vel=np.array([r["vel_ned"] for r in rows]),
        q=np.array([r["q_raw_wxyz"] for r in rows]),
        w=np.array([r["w_raw"] for r in rows]),
        coll=np.array([r["collective"] for r in rows]),
        wire=np.array([r["rate_frd"] for r in rows]),
    )


def model_accel(Rk, coll, v):
    K = np.interp(coll, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
    vb = Rk.T @ v
    cc = np.where(vb >= 0, QUAD_DRAG_C2_MEASURED[:, 0], QUAD_DRAG_C2_MEASURED[:, 1])
    return K * (Rk @ [0.0, 0.0, -1.0]) + Rk @ (-cc * np.abs(vb) * vb) + np.array([0, 0, _G])


def report(d: dict, name: str) -> None:
    n = len(d["t"])
    q_true = d["q"] * ODO_QUAT_TRUE_CONJ_WXYZ[None, :]
    Rt = Rotation.from_quat(q_true[:, [1, 2, 3, 0]]).as_matrix()
    Rr = Rotation.from_quat(d["q"][:, [1, 2, 3, 0]]).as_matrix()
    res_t, res_r, meas_e, mdl_te, mdl_re = [], [], [], [], []
    keys = []
    for k in range(2, n - 2):
        dt = d["t"][k + 1] - d["t"][k - 1]
        if not (0.05 < dt < 0.09):
            continue
        a = (d["vel"][k + 1] - d["vel"][k - 1]) / dt
        if not np.all(np.isfinite(a)) or np.max(np.abs(a)) > 90:
            continue
        v = d["vel"][k]
        c = d["coll"][max(k - 2, 0)]
        tilt = np.degrees(np.arccos(np.clip(Rt[k][2, 2], -1, 1)))
        mt = model_accel(Rt[k], c, v)
        mr = model_accel(Rr[k], c, v)
        res_t.append(mt - a)
        res_r.append(mr - a)
        meas_e.append(a[1]); mdl_te.append(mt[1]); mdl_re.append(mr[1])
        keys.append((np.linalg.norm(v), tilt, c))
    res_t, res_r = np.array(res_t), np.array(res_r)
    keys = np.array(keys)
    print(f"\n==== {name}  ({len(res_t)} usable ticks) ====")

    print("-- force residual (TRUE attitude), median model-meas per axis, by speed x tilt --")
    print(f"{'speed':>7} {'tilt':>7} {'n':>5}   {'N':>7} {'E':>7} {'D':>7}   (m/s^2; |.|>3 sustained = plant gap)")
    for slo, shi in _SPEED_BINS:
        for tlo, thi in _TILT_BINS:
            m = (keys[:, 0] >= slo) & (keys[:, 0] < shi) & (keys[:, 1] >= tlo) & (keys[:, 1] < thi)
            if m.sum() < 15:
                continue
            med = np.median(res_t[m], axis=0)
            print(f"{slo:>3}-{shi:<3} {tlo:>3}-{thi:<3} {m.sum():>5}   "
                  f"{med[0]:>+7.2f} {med[1]:>+7.2f} {med[2]:>+7.2f}")

    print("-- by collective (all speeds, tilt<35) --")
    for clo, chi in _COLL_BINS:
        m = (keys[:, 2] >= clo) & (keys[:, 2] < chi) & (keys[:, 1] < 35)
        if m.sum() < 15:
            continue
        med = np.median(res_t[m], axis=0)
        print(f"  coll {clo:.2f}-{chi:.2f}  n={m.sum():<5}  N{med[0]:+.2f}  E{med[1]:+.2f}  D{med[2]:+.2f}")

    bank = keys[:, 1] > 30
    if bank.sum() >= 20:
        me = np.array(meas_e)[bank]
        ct = np.corrcoef(np.array(mdl_te)[bank], me)[0, 1]
        cr = np.corrcoef(np.array(mdl_re)[bank], me)[0, 1]
        flag = "OK" if ct > max(cr + 0.2, 0.6) else "!! MIRROR CANARY TRIPPED -- RE-AUDIT"
        print(f"-- mirror canary (East corr at tilt>30): TRUE {ct:+.2f}  AS-IS {cr:+.2f}   {flag}")

    fd, wn = [], []
    for k in range(n - 1):
        dtk = d["t"][k + 1] - d["t"][k]
        if 0.02 < dtk < 0.06:
            fd.append(Rotation.from_matrix(Rt[k].T @ Rt[k + 1]).as_rotvec() / dtk)
            wn.append(-d["w"][k])
    fd, wn = np.array(fd), np.array(wn)
    gains = [np.polyfit(wn[:, ax], fd[:, ax], 1)[0] if wn[:, ax].std() > 0.1 else np.nan
             for ax in range(3)]
    print(f"-- rate canary: quat-FD(true) ~ -w_raw gain = "
          f"[{gains[0]:+.3f}, {gains[1]:+.3f}, {gains[2]:+.3f}]  (expect ~+1.0)")


def replay(d: dict, name: str, k0: int, nt: int = 27) -> None:
    P = PlantParams(
        transport_delay_steps=0, rate_sign=np.array([1.0, 1.0, 1.0]),
        super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
        coll_map_thr=COLL_MAP_THR_MEASURED.copy(), coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
        mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)
    nt = min(nt, len(d["t"]) - k0 - 1)
    q0 = d["q"][k0] * ODO_QUAT_TRUE_CONJ_WXYZ
    st = PlantState(pos=d["pos"][k0].copy(), vel=d["vel"][k0].copy(),
                    quat=q0 / np.linalg.norm(q0), omega=-d["w"][k0],
                    thrust=np.float64(d["coll"][k0]))
    cp = None; clp = 0.0
    for k in range(k0, k0 + nt):
        if cp is not None:
            st = plant_step(st, np.concatenate([cp, [clp]]), 1.0 / 30.0, P)
        cp = d["wire"][k]; clp = d["coll"][k]
    err = st.vel - d["vel"][k0 + nt - 1]
    print(f"-- open-loop replay {name} k0={k0} +{nt} ticks: vel err NED = "
          f"[{err[0]:+.2f}, {err[1]:+.2f}, {err[2]:+.2f}] m/s "
          f"(live vE={d['vel'][k0 + nt - 1][1]:+.1f}, twin vE={st.vel[1]:+.1f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session dirs containing debug_obs.jsonl")
    ap.add_argument("--replay", type=int, default=-1,
                    help="also open-loop-replay 27 ticks from this tick index")
    args = ap.parse_args()
    for s in args.sessions:
        d = load(Path(s))
        if d is None:
            continue
        report(d, Path(s).name)
        if args.replay >= 0:
            replay(d, Path(s).name, args.replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
