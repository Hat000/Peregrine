"""scripts/twin_tune.py -- coordinate-descent gain tuning for the decoupled CTBR controller on the
CANONICAL plant twin (Task A, offline). Flies the REAL Navigator/Mission/Planner/Controller through
all 6 gates (``twin_fly_course.fly``) for each candidate gain set and keeps the best.

PRE-REGISTRATION (fixed BEFORE the search -- what "good" means):
  * FINISH 6/6, no divergence, every gate plane CROSSED, on CLEAN (fixed-client c3b5a8e) feedback.
  * max in-plane opening miss across gates 0-5 < 0.40 m (target); < 0.30 m stretch.
  * do NOT regress the already-threaded gates (< 0.20 m each).
  * flight time is a SECONDARY (lexicographic) objective: never trade gate margin for speed.

METRIC -- the honest one is the IN-PLANE OPENING MISS at each gate's plane crossing
(``twin_fly_course.gate_plane_miss``), NOT distance-to-centre. Distance-to-centre inflates along the
through-axis for a fast or TRUNCATED pass: mission.run stops at FINISHED ~gate_pass_radius_m before
the LAST plane, so a dead-centre final gate reads ~the radius (0.74 m) though its in-plane miss is
~0. We therefore fly THROUGH the final gate (``flythrough_s``) and score the in-plane miss.

FINDING (2026-06-05): with the honest metric the BASELINE already threads all 6 gates (worst ~0.09 m,
g0); flying faster genuinely DEGRADES cross-track centring (max_speed 8 -> 0.18 m, 10 -> 0.39 m). So
the objective is to minimise centring, NOT to go fast: we minimise the SUM OF SQUARED plane-misses
(the worst gate dominates via its square, every other gate is protected -- a minimax-only score let
the optimiser wreck good gates to shave the worst). Time only breaks ~1 mm^2 ties.

SCORE (sortable tuple, smaller = better): (not_finished_or_uncrossed, missed_gates, ss@1e-3, time).

SPEED: search at dt=0.02 (per-gate miss is stable vs dt=0.01; ~2x faster); the winner is re-VALIDATED
at dt=0.01 (the report + test fidelity). Deterministic: the twin has no randomness, so coordinate
descent reproduces exactly.

Usage:  python scripts/twin_tune.py            # run the search, print before/after + the tuned dict
        python scripts/twin_tune.py --passes 3 # more coordinate-descent passes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # twin_fly_course (sibling script)

import numpy as np

from racer.planner import ReactivePlanner
from racer.twin_fit import faithful_config
from twin_fly_course import _FAITHFUL_SIGNS, fly, make_controller

N_GATES = 6
PASS_BAR_M = 0.40            # pre-registered target (max in-plane opening miss across all gates)
STRETCH_M = 0.30            # pre-registered stretch
SEARCH_DT = 0.02            # coarse dt for the search (ranking stable vs 0.01)
VALIDATE_DT = 0.01          # fine dt for the final report + the test
SEARCH_MAX_S = 46.0         # generous so a slower-but-valid candidate isn't time-capped out
FLYTHROUGH_S = 2.0         # fly THROUGH the last gate so its plane is crossed + measurable

# Which params go to the controller vs the planner.
CTRL_KEYS = ("kp_pos", "kd_vel", "max_speed", "kp_att", "kd_att", "kp_alt", "kd_alt")
PLAN_KEYS = ("lookahead_m", "cruise_speed")

# Baseline = the canonical _controller() / planner currently in twin_fly_course.py.
BASELINE = {
    "kp_pos": 1.2, "kd_vel": 3.0, "max_speed": 5.0, "kp_att": 10.0, "kd_att": 0.30,
    "kp_alt": 2.0, "kd_alt": 3.0, "lookahead_m": 3.0, "cruise_speed": 5.0,
}

# Per-dim candidate values (coordinate descent tries these holding the others at the current best).
GRID = {
    "kp_pos": [0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0],
    "kd_vel": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0],   # high end = lateral damping (faithful)
    "max_speed": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "kp_att": [6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
    "kd_att": [0.15, 0.30, 0.5, 0.8, 1.2],
    "kp_alt": [1.0, 2.0, 3.0, 4.0],
    "kd_alt": [2.0, 3.0, 4.0, 5.0],
    "lookahead_m": [1.5, 2.0, 3.0, 4.0, 5.0],
    "cruise_speed": [4.0, 5.0, 6.0, 8.0],
}
# High-impact dims first (the terminal cross-track of g5 is dominated by lateral authority + approach
# geometry + speed); the attitude/altitude gains already thread the climbing gates, so they go last.
ORDER = ["kp_pos", "max_speed", "kd_vel", "lookahead_m", "cruise_speed",
         "kp_att", "kd_att", "kp_alt", "kd_alt"]


def evaluate(params: dict, *, dt: float = SEARCH_DT, max_s: float = SEARCH_MAX_S, n: int = N_GATES,
             faithful: bool = False) -> dict:
    """Fly the course once with ``params`` and return the scored result. Per-gate quality is the
    in-plane opening miss; a gate whose plane was never crossed falls back to its (large) 3D
    distance-to-centre so it is penalised as the genuine miss it is.

    ``faithful=True`` (Task C): fly the SIM-FAITHFUL plant (``faithful_config``) with the restored
    live sim-sign compensation (``_FAITHFUL_SIGNS``: body_rate/odo_att/odo_rate signs + ff_gain +
    plant hover); the outer gains are tuned on top. ``False`` (Task A): the canonical twin."""
    ctrl = make_controller(signs=_FAITHFUL_SIGNS if faithful else None,
                           **{k: params[k] for k in CTRL_KEYS})
    plan = ReactivePlanner(cruise_speed=params["cruise_speed"], lookahead_m=params["lookahead_m"],
                           yaw_mode="course")
    r = fly(n, dt=dt, max_s=max_s, velocity_mode="clean", controller=ctrl, planner=plan,
            flythrough_s=FLYTHROUGH_S, plant_config=faithful_config() if faithful else None)
    pm = r["plane_miss"]
    closest = r["closest"]
    miss = [float(pm[i]) if pm[i] is not None else float(closest[i]) for i in range(n)]
    crossed = all(pm[i] is not None for i in range(n))
    finished = bool(r["final"].name == "FINISHED" and r["gate_index"] == n)
    if not np.all(np.isfinite(miss)):              # divergence / NaN -> hard fail
        miss, finished, crossed = [1e6] * n, False, False
    worst = float(np.max(miss))
    ss = float(np.sum(np.square(miss)))            # sum of squared in-plane miss (the objective)
    return {"worst": worst, "ss": ss, "finished": finished, "crossed": crossed,
            "gate_index": int(r["gate_index"]), "t_s": float(r["t_s"]), "miss": miss}


def score_key(ev: dict) -> tuple:
    """Sortable score, smaller is better: (finished+all-crossed) > fewer-missed > sum-squared-miss
    (1e-3 buckets) > time. The squared sum minimises the worst gate while protecting the rest; time
    only breaks near-ties, so gate margin is never traded for speed."""
    ok = ev["finished"] and ev["crossed"]
    missed = 0 if ev["finished"] else (N_GATES - ev["gate_index"])
    return (0 if ok else 1, missed, round(ev["ss"], 3), round(ev["t_s"], 3))


def coordinate_descent(passes: int = 2, *, verbose: bool = True, faithful: bool = False,
                       search_dt: float = SEARCH_DT) -> tuple[dict, dict, int]:
    """Greedy coordinate descent from BASELINE over GRID. Returns (best_params, best_ev, n_evals).
    ``faithful`` -> tune on the sim-faithful plant with restored live signs (Task C). ``search_dt``:
    the faithful plant's inner-loop tau (~0.019 s) needs the live 100 Hz dt=0.01 (dt=0.02 is too
    coarse and mis-ranks); the canonical twin is stable at 0.02."""
    cur = dict(BASELINE)
    cur_ev = evaluate(cur, faithful=faithful, dt=search_dt)
    best_key = score_key(cur_ev)
    n_eval = 1
    if verbose:
        print(f"  baseline: ss={cur_ev['ss']:.4f}  worst={cur_ev['worst']:.3f}  "
              f"t={cur_ev['t_s']:.1f}s  {cur_ev['gate_index']}/{N_GATES}")
    for p in range(passes):
        improved = False
        for dim in ORDER:
            best_val = cur[dim]
            for val in GRID[dim]:
                if val == cur[dim]:
                    continue
                trial = dict(cur)
                trial[dim] = val
                ev = evaluate(trial, faithful=faithful, dt=search_dt)
                n_eval += 1
                k = score_key(ev)
                if k < best_key:
                    best_key, best_val, cur_ev, improved = k, val, ev, True
            if best_val != cur[dim]:
                cur[dim] = best_val
                if verbose:
                    print(f"  pass{p} {dim:12s} -> {best_val:<5g}  ss={cur_ev['ss']:.4f}  "
                          f"worst={cur_ev['worst']:.3f}  t={cur_ev['t_s']:.1f}s  "
                          f"[{' '.join(f'{c:.2f}' for c in cur_ev['miss'])}]")
        if not improved:
            if verbose:
                print(f"  pass{p}: no improvement -> converged")
            break
    return cur, cur_ev, n_eval


def _fmt(ev: dict) -> str:
    return (f"{ev['gate_index']}/{N_GATES} {'FIN' if ev['finished'] else 'run':3s}  "
            f"ss={ev['ss']:.4f}  worst={ev['worst']:.3f}  t={ev['t_s']:.1f}s  "
            f"[{'  '.join(f'g{i} {c:.2f}' for i, c in enumerate(ev['miss']))}]")


# ---- TUNED RESULT (from `python scripts/twin_tune.py --passes 2`, 2026-06-05; deterministic) -------
# Validated at dt=0.01: 6/6 FINISHED, every plane crossed, worst in-plane miss 0.069 m (was 0.092),
# ss 0.0074 (was 0.0249), t 29.3 s (was 34.6). All 6 gates < 0.07 m -- meets target + stretch.
# Changes vs baseline: kp_pos 1.2->2.0, kd_vel 3.0->4.0, max_speed 5.0->6.0 (more lateral authority
# centres the cross-track gates AND allows a touch more speed -- not a margin-for-speed trade).
# The test (tests/test_twin_tune.py) re-validates these without re-running the search.
TUNED_GAINS = {"kp_pos": 2.0, "kd_vel": 4.0, "max_speed": 6.0, "kp_att": 10.0, "kd_att": 0.3,
               "kp_alt": 2.0, "kd_alt": 3.0}              # controller fields (make_controller(**...))
TUNED_PLANNER = {"lookahead_m": 3.0, "cruise_speed": 5.0}  # planner fields (ReactivePlanner(**..., yaw_mode="course"))

# ---- LIVE-READY RESULT (Task C: `python scripts/twin_tune.py --faithful`, 2026-06-05; deterministic)
# Outer gains re-tuned at the LIVE 100 Hz rate (dt=0.01) on the SIM-FAITHFUL plant
# (racer.twin_fit.faithful_config) with the restored live sim-sign compensation (_FAITHFUL_SIGNS:
# body_rate_sign=[1,1,-1], odo_att_sign=[-1,1,1], odo_rate_sign=[-1,-1,1], ff_gain=2.5, hover=0.2656).
# THIS IS THE CONFIG THE LIVE VERIFY FLIGHT USES (the outer gains transfer; the signs are measured).
# Result: THREADS all 6 gates (worst in-plane miss 0.61 m at g1 < 0.75 m half-opening = valid passes;
# per-gate [0.13, 0.61, 0.14, 0.26, 0.14, 0.05]), t 31.9 s. HONEST caveat: ~9x less tight than the
# canonical twin (0.61 vs 0.069) -- the faithful plant's drag + fast tau make the reactive lateral
# loop a delicate optimum (kp_pos 1.2->0.6, lookahead 3->5 to tame the cross-track oscillation; more
# lateral authority re-oscillates). g1 (first cross-track) is marginal -> a VQ2 racing-line/RL target.
# The canonical-tuned gains do NOT transfer (1/6) -- the faithful re-tune is essential.
FAITHFUL_TUNED_GAINS = {"kp_pos": 0.6, "kd_vel": 2.0, "max_speed": 6.0, "kp_att": 10.0,
                        "kd_att": 0.15, "kp_alt": 4.0, "kd_alt": 2.0}
FAITHFUL_TUNED_PLANNER = {"lookahead_m": 5.0, "cruise_speed": 8.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passes", type=int, default=2, help="coordinate-descent passes")
    ap.add_argument("--faithful", action="store_true",
                    help="Task C: tune on the sim-faithful plant with restored live signs")
    args = ap.parse_args()

    twin = "SIM-FAITHFUL plant + restored live signs" if args.faithful else "CANONICAL twin"
    print(f"CTBR gain tuning on the {twin} (clean feedback). Pre-reg: all 6 gates' in-plane "
          f"miss < {PASS_BAR_M:.2f} m, finish 6/6, every plane crossed, no divergence.\n")
    search_dt = VALIDATE_DT if args.faithful else SEARCH_DT     # faithful tau needs the live 100 Hz dt
    t0 = time.perf_counter()
    print("Searching (dt=%.3f)..." % search_dt)
    best, _, n_eval = coordinate_descent(passes=args.passes, faithful=args.faithful, search_dt=search_dt)
    dt_search = time.perf_counter() - t0

    # Re-validate baseline + winner at the fine dt (the report / test fidelity).
    base_ev = evaluate(BASELINE, dt=VALIDATE_DT, faithful=args.faithful)
    tuned_ev = evaluate(best, dt=VALIDATE_DT, faithful=args.faithful)

    print(f"\n{n_eval} evals in {dt_search:.0f}s (search dt={search_dt}); validated at dt={VALIDATE_DT}\n")
    print("  BEFORE (baseline): " + _fmt(base_ev))
    print("  AFTER  (tuned)   : " + _fmt(tuned_ev))
    print("\n  gain changes (baseline -> tuned):")
    for k in ORDER:
        if abs(best[k] - BASELINE[k]) > 1e-9:
            print(f"    {k:12s} {BASELINE[k]:<6g} -> {best[k]:<6g}")
    worst = tuned_ev["worst"]
    verdict = ("MEETS < %.2f m" % PASS_BAR_M) if worst < PASS_BAR_M else ("ABOVE %.2f m" % PASS_BAR_M)
    print(f"\n  tuned worst in-plane miss = {worst:.3f} m ({verdict}); pre-reg stretch < "
          f"{STRETCH_M:.2f} m: {'yes' if worst < STRETCH_M else 'no'}")
    prefix = "FAITHFUL_TUNED" if args.faithful else "TUNED"
    print(f"\n  # paste into {prefix}_GAINS / {prefix}_PLANNER:")
    print(f"  {prefix}_GAINS = {{" + ", ".join(f"'{k}': {best[k]:g}" for k in CTRL_KEYS) + "}")
    print(f"  {prefix}_PLANNER = {{" + ", ".join(f"'{k}': {best[k]:g}" for k in PLAN_KEYS) + "}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
