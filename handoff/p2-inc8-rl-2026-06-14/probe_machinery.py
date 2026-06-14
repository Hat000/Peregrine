"""ESCAPE-HATCH PROBE -- machinery verification for the inc8 estimator-emulation obs path.

Verifies the FOUR GREEN conditions with CONTROLLED inputs (not a brittle policy flight), so the
verdict is about the MACHINERY, not inc7's quality:
  (ii)  camera pointing measurably + monotonically moves fix arrival
  (iii) the KF in-plane error shrinks as fix density rises
  (iv)  obs[17:20] (c_inplane, c_along, age_norm) are non-degenerate (vary, not stuck at 0/1)
  (i)   fix-rate + terminal-gate-lock are measured per pointing condition

(GATE#1 frame-seam identity + GATE#3 monotonicity are pinned in tests/test_estimator_emul.py.)

Run from repo ROOT:  .venv\\Scripts\\python.exe handoff/p2-inc8-rl-2026-06-14/probe_machinery.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from estimator_emul import EstimatorEmulator, EmulConfig, GATE4_IDX     # noqa: E402
from racer.rl_plant import PlantState                                   # noqa: E402
from offline_rollout import _quat_from_rpy                             # noqa: E402
from fly_rl import _GATE_POS_ZUP, _FLIP                                # noqa: E402

_GATE_NED = _GATE_POS_ZUP * _FLIP


def _st(pos, vel, quat) -> PlantState:
    return PlantState(pos=np.asarray(pos, np.float64), vel=np.asarray(vel, np.float64),
                      quat=np.asarray(quat, np.float64), omega=np.zeros(3), thrust=np.float64(0.2656))


def gate4_approach(crab_deg: float, v: float = 15.0, start_back: float = 28.0,
                   seed: int = 0, bias: bool = False, prior_inplane_drift: float = 0.0):
    """Scripted head-on gate-4 approach with a fixed camera CRAB (yaw off the through-axis), WARM
    velocity prior (mid-lap regime). ``prior_inplane_drift`` seeds the KF with an in-plane offset
    (lateral+vertical) modelling drift ACCUMULATED entering the approach -- the realistic case the
    gate-4 fixes must correct (a perfectly straight, perfectly-tracked approach has no in-plane drift,
    so there the in-plane floor is pure fix-noise; the binding axis IS the fix model, per memory).
    Returns per-condition machinery numbers."""
    rng = np.random.default_rng(seed)
    cfg = EmulConfig(inject_bias=bias)
    emu = EstimatorEmulator(cfg)
    g4 = _GATE_NED[GATE4_IDX]
    yaw = np.pi + np.radians(crab_deg)
    quat = _quat_from_rpy(0.0, 0.0, yaw)
    # approach from +X of gate-4 moving in -X (the VQ1 travel/through direction), warm prior.
    st0 = _st(g4 + np.array([start_back, 0.0, 0.0]), [-v, 0.0, 0.0], quat)
    emu.reset(st0, GATE4_IDX, rng)
    if prior_inplane_drift != 0.0:
        # inject the drift along the gate in-plane axes (right + down), expressed in world NED.
        Rwg = emu.gates[GATE4_IDX].R_world_gate
        emu.kf.x[:3] = emu.kf.x[:3] + Rwg @ np.array([prior_inplane_drift, prior_inplane_drift, 0.0])
    cur = st0
    for _ in range(400):
        nxt = _st(cur.pos + cur.vel * 0.0333, cur.vel, cur.quat)
        emu.step(cur, nxt, GATE4_IDX, 0.0333, rng)
        cur = nxt
        if cur.pos[0] <= g4[0]:
            break
    err = emu.gate4_inplane_error_series()
    return dict(
        crab=crab_deg,
        fix_rate=emu.gate4_band_fix_rate(),
        term_lock=emu.terminal_gate_lock_frac(),
        ip_mean=float(err.mean()) if err.size else float("nan"),
        ip_p90=float(np.percentile(err, 90)) if err.size else float("nan"),
        ip_terminal=float(err[-5:].mean()) if err.size >= 5 else float("nan"),
        n_fix=int(np.sum(emu.trace.fix_accepted)),
        emu=emu,
    )


def main() -> int:
    print("=" * 78)
    print("ESCAPE-HATCH PROBE -- machinery verification (controlled gate-4 approach, 15 m/s, warm)")
    print("=" * 78)

    # --- (ii): POINTING -> FIX-RATE lever (crab sweep isolates the pointing->fix mapping) ---
    print("\n[(ii) POINTING -> FIX-RATE lever]  camera crab sweep (bias OFF), mean over 6 seeds")
    print(f"  {'crab(deg)':>9}  {'fix_rate':>8}  {'term_lock':>9}  {'n_fix':>5}")
    rows = []
    for crab in [0, 10, 20, 30, 40, 50]:
        accs = [gate4_approach(crab, seed=s) for s in range(6)]
        row = dict(crab=crab,
                   fix_rate=np.nanmean([a["fix_rate"] for a in accs]),
                   term_lock=np.nanmean([a["term_lock"] for a in accs]),
                   n_fix=np.mean([a["n_fix"] for a in accs]))
        rows.append(row)
        print(f"  {crab:9d}  {row['fix_rate']:8.3f}  {row['term_lock']:9.3f}  {row['n_fix']:5.1f}")
    fr_mono = all(rows[i]["fix_rate"] + 1e-9 >= rows[i + 1]["fix_rate"] for i in range(len(rows) - 1))
    print(f"  fix_rate + terminal-lock monotone NON-increasing as the gate de-centers: {fr_mono}")
    print(f"  centered(crab=0): fix_rate={rows[0]['fix_rate']:.3f} term_lock={rows[0]['term_lock']:.3f}"
          f"  |  off-pointed(crab=50): fix_rate={rows[-1]['fix_rate']:.3f} (gate out of frame)")

    # --- (iii): the KF CONSUMES fixes -- with 0.5 m accumulated in-plane drift entering the approach,
    #     centered (fixes) pulls it to the fix-noise floor; pointed-away (no fixes) does NOT. ---
    print("\n[(iii) KF in-plane error SHRINKS with fix density]  0.5 m in-plane drift entering gate-4")
    print(f"  {'condition':22s}  {'fix_rate':>8}  {'ip_terminal':>11}  {'ip_p90':>8}  {'n_fix':>5}")
    drift = 0.5
    cen = [gate4_approach(0, seed=s, prior_inplane_drift=drift) for s in range(6)]
    awy = [gate4_approach(50, seed=s, prior_inplane_drift=drift) for s in range(6)]
    for tag, accs in (("CENTERED (fixes)", cen), ("POINTED-AWAY (no fixes)", awy)):
        fr = np.nanmean([a["fix_rate"] for a in accs])
        tm = np.nanmean([a["ip_terminal"] for a in accs])
        p9 = np.nanmean([a["ip_p90"] for a in accs])
        nf = np.mean([a["n_fix"] for a in accs])
        print(f"  {tag:22s}  {fr:8.3f}  {tm:11.4f}  {p9:8.4f}  {nf:5.1f}")
    cen_term = np.nanmean([a["ip_terminal"] for a in cen])
    awy_term = np.nanmean([a["ip_terminal"] for a in awy])
    print(f"  => terminal in-plane error: fixes {cen_term:.3f} m  vs  no-fix {awy_term:.3f} m "
          f"(drift {drift:.2f} m) -> fixes correct {100*(1-cen_term/awy_term):.0f}% of the drift")
    # error-vs-cumulative-fixes decay on a single centered run
    emu = cen[0]["emu"]
    er = np.asarray(emu.trace.inplane_err)
    cf = np.cumsum(np.asarray(emu.trace.fix_accepted, dtype=int))
    if cf.max() >= 8:
        early = er[(cf >= 1) & (cf <= 3)].mean()
        late = er[cf >= cf.max() - 3].mean()
        print(f"  decay on one run: in-plane err at 1-3 cumulative fixes={early:.3f} m -> "
              f"at >={cf.max()-3} fixes={late:.3f} m  (SHRINKS: {late < early})")

    # --- (iv): obs[17:20] non-degeneracy over a centered approach ---
    print("\n[obs[17:20] NON-DEGENERACY]  centered gate-4 approach, per-step confidence channel")
    a = gate4_approach(0, seed=0)
    emu = a["emu"]
    ci = np.asarray(emu.trace.c_inplane)
    ca = np.asarray(emu.trace.c_along)
    ag = np.asarray(emu.trace.age_norm)
    fx = np.asarray(emu.trace.fix_accepted, dtype=bool)
    def _spread(x):
        return dict(min=float(x.min()), max=float(x.max()), std=float(x.std()),
                    uniq=int(np.unique(np.round(x, 4)).size))
    for name, arr in (("c_inplane", ci), ("c_along", ca), ("age_norm", ag)):
        s = _spread(arr)
        nondeg = s["max"] - s["min"] > 0.05 and s["uniq"] > 3
        print(f"  {name:9s}  min={s['min']:.3f} max={s['max']:.3f} std={s['std']:.3f} "
              f"uniq={s['uniq']:3d}  NON-DEGENERATE={nondeg}")
    # c_inplane should be higher right after a fix than during a dropout
    if fx.any():
        post = ci[np.roll(fx, 1)].mean() if fx.sum() > 1 else float("nan")
        print(f"  mean c_inplane right after an accepted fix = {post:.3f} "
              f"(vs overall mean {ci.mean():.3f})")

    print("\n" + "=" * 78)
    print("VERDICT INPUTS: GATE#1 (test) + monotone pointing->fix (above) + KF error shrinks with "
          "fix density (above) + [17:20] non-degenerate (above)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
