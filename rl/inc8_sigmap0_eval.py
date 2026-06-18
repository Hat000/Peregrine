"""rl/inc8_sigmap0_eval.py -- inc8 GO-gate instrument: GT-anchored terminal centering sigma_p0.

WHY THIS EXISTS (and why it is NOT contact_true_eval --estim-emul).
The inc8 GO gate is the PHYSICAL terminal-centering spread at the binding gate-4 plane:
``sigma_p0_lat <= ~0.08 m`` (miss p99 ~= 3*sigma_p0). contact_true_eval's --estim-emul path reports
``g4_inplane_p90/p99`` from ``EstimatorEmulator.gate4_inplane_error_series()``, which is
``inplane_err = |KF_pos - truth_pos|`` in the gate frame -- the ESTIMATOR error (estim_err), floored
at the ~0.115-0.13 gate-relative KF RMS REGARDLESS of policy. That is the wrong quantity for the GO
gate (MEMORY: "estim_err is floored ... NOT the gate"), and the warm-start GO criterion (warmstart
build REPORT sec.6/8) explicitly says: crown on the PHYSICAL GT-anchored sigma_p0, NOT estim_err.

WHAT sigma_p0 IS (GT-anchored, this tool).
The TRUE drone position relative to the gate-4 centre at the gate-4 plane crossing, in the gate frame:
  - lateral  y = rel[1]  (horizontal cross-track)  -> sigma_p0_LAT  (the binding GO axis)
  - vertical z = rel[2]  (altitude cross-track)     -> sigma_p0_vert (the dominated axis, reported)
``rel = _R_W2G @ (pos_ned * _FLIP - gate_pos_zup[4])`` -- the SAME Z-up obs-frame transform and the
SAME linear plane-crossing interpolation that contact_true_eval._score_gate uses, but we keep y and z
SEPARATE instead of collapsing to ``linf = max(|y|,|z|)``. ``pos_ned`` is the PLANT TRUTH trajectory
(``run_episode(..., record_pos=True)``), so the crossing offset is ground-truth -- never the KF.

sigma_p0 is a DISTRIBUTION statistic: it needs an ENSEMBLE. The spread is induced by the per-episode
estimator DR (``estimator_emul.EmulConfig``: per-fix sigma_lat ~ U[0.05,0.15], one-signed PnP bias ~
U[0,0.19], IMU/accept noise) -- so it is only meaningful with ``--estim-emul`` ON and many episodes
(each a fresh ``emul_seed``). With the emul OFF the simstart episode is deterministic (sigma_p0 == 0,
a single perfect-obs crossing) -- the tool warns and reports that single offset as a reference only.

GO RULE.  sigma_p0_lat <= 0.08 m AND |lateral miss| p99 ~= 3*sigma_p0 <= ~0.24 m, evaluated on the
episodes that REACH gate-4 (a low reach-rate is itself a NO-GO signal and is reported).

PROVISIONAL.  The obs the policy flies on here comes from the estimator EMUL (fix_surrogate -> LinearKF),
NOT a real detector->PnP->KF chain (#37). Any GO/NO-GO from this tool is PROVISIONAL until the
real-detector vertical spike confirms emul fidelity.

Usage (from repo root, .venv active):
  .venv\\Scripts\\python.exe rl/inc8_sigmap0_eval.py ^
      --ckpt outputs/train/inc8_warmstart_seed0_ws1/periodic ^
      --estim-emul --n-episodes 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_RL = Path(__file__).resolve().parent
_SRC = _RL.parent / "src"
for _p in (str(_SRC), str(_RL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fly_rl import _FLIP, _GATE_POS_ZUP, _R_W2G, _TRAIN_DT, load_actor  # noqa: E402
from estimator_emul import actor_obs_dim                               # noqa: E402
from contact_true_eval import (_build_plant_params, _build_start,       # noqa: E402
                               run_episode, BODY_RADIUS_NOM, FRAME_DEPTH_NOM)

GATE4_IDX = 4


def gate4_true_crossing_yz(result, pos_traj: np.ndarray, dt: float) -> tuple[float, float] | None:
    """Lateral (y) and vertical (z) gate-frame offset of the TRUE trajectory at the gate-4 plane
    crossing, or None if gate-4 was never passed.

    Uses the gate-4 PASS time the canonical scorer (run_episode/_score_gate) already recorded, then
    re-interpolates the SAME crossing with y and z kept separate. ``pos_traj`` is plant truth (NED)."""
    for c in result.crossings:
        if c.gate == GATE4_IDX and c.verdict == "pass":
            k = int(round(c.t / dt))
            k = min(max(k, 1), len(pos_traj) - 1)
            g = _GATE_POS_ZUP[GATE4_IDX]
            prev_rel = _R_W2G @ (pos_traj[k - 1] * _FLIP - g)
            cur_rel = _R_W2G @ (pos_traj[k] * _FLIP - g)
            denom = (cur_rel[0] - prev_rel[0]) or 1e-9
            f = -prev_rel[0] / denom
            y = prev_rel[1] + f * (cur_rel[1] - prev_rel[1])
            z = prev_rel[2] + f * (cur_rel[2] - prev_rel[2])
            return float(y), float(z)
    return None


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def evaluate(ckpt: str, n_episodes: int, estim_emul: bool, obs_dim: int,
             plant: str, base_seed: int, max_time: float) -> dict:
    actor = load_actor(ckpt)
    obs_dim = obs_dim if obs_dim > 0 else actor_obs_dim(actor)
    params = _build_plant_params(plant)

    if not estim_emul:
        n_episodes = 1  # deterministic without DR -> a single crossing; sigma is undefined

    lat: list[float] = []
    vert: list[float] = []
    reached = 0
    finished = 0
    locks: list[float] = []
    fixrates: list[float] = []

    st0, tgt0, vflip = _build_start("simstart", 0)
    dt = _TRAIN_DT
    for i in range(n_episodes):
        st, tgt = st0, tgt0
        result, pos_traj = run_episode(
            actor, st, tgt, vflip, params,
            body_radius=BODY_RADIUS_NOM, frame_depth=FRAME_DEPTH_NOM,
            max_time=max_time, start_label="simstart", record_pos=True,
            estim_emul=estim_emul, obs_dim=obs_dim, emul_seed=base_seed + i,
        )
        if result.success:
            finished += 1
        if result.emulator is not None:
            lk = result.emulator.terminal_gate_lock_frac()
            fr = result.emulator.gate4_band_fix_rate()
            if not np.isnan(lk):
                locks.append(lk)
            if not np.isnan(fr):
                fixrates.append(fr)
        yz = gate4_true_crossing_yz(result, pos_traj, dt)
        if yz is not None:
            reached += 1
            lat.append(yz[0])
            vert.append(yz[1])

    lat_a = np.asarray(lat)
    vert_a = np.asarray(vert)
    inplane = np.hypot(lat_a, vert_a) if lat_a.size else np.array([])
    return {
        "ckpt": Path(ckpt).name,
        "obs_dim": obs_dim,
        "estim_emul": estim_emul,
        "n_episodes": n_episodes,
        "n_reached_g4": reached,
        "n_finished": finished,
        "reach_rate": reached / max(n_episodes, 1),
        # GT-anchored terminal centering (the GO axis)
        "sigma_p0_lat": float(np.std(lat_a)) if lat_a.size else float("nan"),
        "bias_lat": float(np.mean(lat_a)) if lat_a.size else float("nan"),
        "lat_abs_p90": _pct(np.abs(lat_a), 90),
        "lat_abs_p99": _pct(np.abs(lat_a), 99),
        # vertical (dominated axis, reported for completeness)
        "sigma_p0_vert": float(np.std(vert_a)) if vert_a.size else float("nan"),
        "bias_vert": float(np.mean(vert_a)) if vert_a.size else float("nan"),
        "vert_abs_p90": _pct(np.abs(vert_a), 90),
        "vert_abs_p99": _pct(np.abs(vert_a), 99),
        # combined in-plane (reference)
        "inplane_p90": _pct(inplane, 90),
        "inplane_p99": _pct(inplane, 99),
        # pointing retention (the inc8 lever)
        "terminal_lock_mean": float(np.mean(locks)) if locks else float("nan"),
        "g4_fix_rate_mean": float(np.mean(fixrates)) if fixrates else float("nan"),
    }


def _go_verdict(m: dict, sigma_target: float = 0.08) -> str:
    s = m["sigma_p0_lat"]
    p99 = m["lat_abs_p99"]
    if np.isnan(s):
        return "NO-DATA (no gate-4 crossings)"
    ok_sigma = s <= sigma_target
    ok_p99 = (not np.isnan(p99)) and p99 <= 3.0 * sigma_target
    verdict = "GO" if (ok_sigma and ok_p99) else "NO-GO"
    return (f"{verdict} (sigma_p0_lat={s:.3f}{'<=' if ok_sigma else '>'}{sigma_target} ; "
            f"lat_p99={p99:.3f}{'<=' if ok_p99 else '>'}{3*sigma_target:.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="actor.pth (or dir holding actor.pth)")
    ap.add_argument("--n-episodes", type=int, default=200,
                    help="ensemble size for the sigma_p0 distribution (estim-emul ON)")
    ap.add_argument("--estim-emul", action="store_true",
                    help="fly on the emulated KF obs (REQUIRED for a meaningful sigma; OFF = "
                         "deterministic single crossing, sigma==0)")
    ap.add_argument("--obs-dim", type=int, default=0, help="17 inc7 / 20 inc8; 0 = infer")
    ap.add_argument("--plant", default="mixer", choices=["map", "flat", "aero", "mixer"])
    ap.add_argument("--base-seed", type=int, default=0, help="emul_seed = base-seed + episode index")
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--sigma-target", type=float, default=0.08)
    args = ap.parse_args()

    ckpt = args.ckpt
    if Path(ckpt).is_dir():                      # accept a checkpoint DIR (warm-start outputs)
        ckpt = str(Path(ckpt) / "actor.pth")

    if not args.estim_emul:
        print("!! --estim-emul OFF: the simstart episode is DETERMINISTIC (no per-episode DR), so "
              "sigma_p0 is undefined (single crossing). Pass --estim-emul for the GO gate.")

    m = evaluate(ckpt, args.n_episodes, args.estim_emul, args.obs_dim,
                 args.plant, args.base_seed, args.max_time)

    print(f"\n{'='*78}")
    print(f"inc8 GT-ANCHORED sigma_p0  ckpt={m['ckpt']}  obs_dim={m['obs_dim']}  "
          f"estim_emul={'ON' if m['estim_emul'] else 'OFF'}  plant={args.plant}")
    print(f"{'='*78}")
    print(f"  episodes={m['n_episodes']}  reached_g4={m['n_reached_g4']} "
          f"(reach_rate={m['reach_rate']:.2f})  finished={m['n_finished']}")
    print(f"  -- GT-anchored TERMINAL CENTERING at the gate-4 plane (TRUE trajectory) --")
    print(f"  LATERAL  sigma_p0_lat = {m['sigma_p0_lat']:.4f} m   bias = {m['bias_lat']:+.4f} m   "
          f"|miss| p90 = {m['lat_abs_p90']:.4f}  p99 = {m['lat_abs_p99']:.4f}")
    print(f"  VERTICAL sigma_p0_vert= {m['sigma_p0_vert']:.4f} m   bias = {m['bias_vert']:+.4f} m   "
          f"|miss| p90 = {m['vert_abs_p90']:.4f}  p99 = {m['vert_abs_p99']:.4f}")
    print(f"  in-plane (combined) |miss| p90 = {m['inplane_p90']:.4f}  p99 = {m['inplane_p99']:.4f}")
    print(f"  -- pointing retention (inc8 lever) --")
    print(f"  terminal_gate_lock(mean) = {m['terminal_lock_mean']:.3f}   "
          f"g4_band_fix_rate(mean) = {m['g4_fix_rate_mean']:.3f}")
    print(f"  >> GO RULE (sigma_p0_lat<={args.sigma_target}): {_go_verdict(m, args.sigma_target)}")
    print(f"  >> PROVISIONAL: emul obs, not real detector->PnP->KF (#37). Crown only after the "
          f"real-detector spike.\n")

    print(f"SIGMA_P0_SUMMARY ckpt={m['ckpt']} obs_dim={m['obs_dim']} "
          f"estim_emul={'ON' if m['estim_emul'] else 'OFF'} n_ep={m['n_episodes']} "
          f"reach={m['n_reached_g4']}/{m['n_episodes']} "
          f"sigma_p0_lat={m['sigma_p0_lat']:.4f} lat_p90={m['lat_abs_p90']:.4f} "
          f"lat_p99={m['lat_abs_p99']:.4f} sigma_p0_vert={m['sigma_p0_vert']:.4f} "
          f"term_lock={m['terminal_lock_mean']:.3f} fixrate={m['g4_fix_rate_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
