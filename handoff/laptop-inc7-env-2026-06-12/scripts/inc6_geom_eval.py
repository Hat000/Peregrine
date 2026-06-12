"""inc6_geom_eval.py -- re-run inc6 through the GEOMETRY-HONEST eval (LAPTOP-INC7-ENV,
2026-06-12). The env's own validation of the inc7 contact-true geometry: trajectories are
generated under the LEGACY contact rules (the twin has no contact model, exactly like the
training env inc6 saw), then SCORED post-hoc against the contact-true geometry
(body radius r + 0.30 m frame extrusion, rl/offline_rollout.gate_event). Expectation on
record (training-doctrine WRITEUP Part 3): the standing gate-3 funnel -- which legacy
scoring calls a comfortable pass and which crashed live 4/4 -- scores as the near-miss /
strike it really is, while the bridge (0.2 m lower on the same funnel) survives at small r.

Conditions mirror the doctrine forensics: standing (simstart) and bridge (handoff) x
{nominal, climb-bin residual at full measured magnitude, residual + live latency 2}.

Usage:  .venv\\Scripts\\python.exe handoff\\laptop-inc7-env-2026-06-12\\scripts\\inc6_geom_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))
sys.path.insert(0, str(_ROOT / "handoff" / "laptop-training-doctrine-2026-06-12" / "scripts"))

import numpy as np

from fly_rl import N_GATES, _FLIP, _GATE_POS_ZUP, _R_W2G, load_actor
from offline_rollout import gate_event
from doctrine_probes import CKPT, DT, rollout

RADII = (0.28, 0.33, 0.38)
DEPTH = 0.30
RES_FULL = np.array([+2.79, 0.0, -2.36])      # measured climb-bin residual (q2 forensics sign)


def contact_true_scan(trace, r, depth):
    """First contact-true collision along the trace: (t, gate, gate-frame x of the step
    endpoints, crossing/contact Linf proxy) or None. Scores EVERY gate every step with the
    inflated+extruded model -- exactly what the live sim's body-halo contact would do."""
    pos = trace["pos"]
    for k in range(1, len(pos)):
        for g in range(N_GATES):
            ev = gate_event(pos[k - 1], pos[k], g, body_radius=r, frame_depth=depth)
            if ev == "collision":
                rel0 = _R_W2G @ (pos[k - 1] * _FLIP - _GATE_POS_ZUP[g])
                rel1 = _R_W2G @ (pos[k] * _FLIP - _GATE_POS_ZUP[g])
                linf1 = max(abs(rel1[1]), abs(rel1[2]))
                return (trace["t"][k], g, rel0[0], rel1[0], linf1)
    return None


def corridor_linf(trace, g, x_at):
    """In-plane Linf of the approach corridor at gate-frame x = x_at (interp over the trace)."""
    rel = np.array([_R_W2G @ (p * _FLIP - _GATE_POS_ZUP[g]) for p in trace["pos"]])
    m = (rel[:, 0] >= x_at - 2.5) & (rel[:, 0] <= 0.3)
    if m.sum() < 3:
        return float("nan")
    r = rel[m][np.argsort(rel[m][:, 0])]
    return float(np.interp(x_at, r[:, 0], np.max(np.abs(r[:, 1:3]), axis=1)))


def main():
    actor = load_actor(CKPT)
    conds = [
        ("standing nominal",      dict(start_kind="simstart")),
        ("standing res_full",     dict(start_kind="simstart", residual_ned=RES_FULL,
                                       residual_bin={})),
        ("standing res_full lat2", dict(start_kind="simstart", residual_ned=RES_FULL,
                                        residual_bin={}, latency=2)),
        ("bridge nominal",        dict(start_kind="handoff", handoff_speed=10.0,
                                       handoff_dist=3.0)),
        ("bridge res_full",       dict(start_kind="handoff", handoff_speed=10.0,
                                       handoff_dist=3.0, residual_ned=RES_FULL,
                                       residual_bin={})),
    ]
    print("=" * 100)
    print("INC6 GEOMETRY-HONEST EVAL -- legacy-generated trajectories, contact-true scoring")
    print(f"(body radius r in {RADII}, frame depth {DEPTH} m; mixer plant; ckpt {Path(CKPT).name})")
    print("=" * 100)
    for tag, kw in conds:
        outcome, passes, tr = rollout(actor, **kw)
        cross = {g: max(abs(y), abs(z)) for g, y, z, _t in passes}
        cross_s = " ".join(f"g{g}:{v:.2f}" for g, v in sorted(cross.items()))
        print(f"\n[{tag}]  legacy outcome: {outcome}")
        print(f"  legacy crossing Linf: {cross_s}")
        print(f"  corridor Linf at x=-1 m (g3): {corridor_linf(tr, 3, -1.0):.2f}   "
              f"at x=-0.30 m (slab entry): {corridor_linf(tr, 3, -DEPTH):.2f}")
        for r in RADII:
            hit = contact_true_scan(tr, r, DEPTH)
            band = 0.75 - r
            if hit is None:
                margins = {g: band - v for g, v in cross.items()}
                worst = min(margins.items(), key=lambda kv: kv[1]) if margins else (-1, 0)
                print(f"  r={r:.2f}: CLEAN under contact-true geometry "
                      f"(pass band {band:.2f}; worst margin g{worst[0]} {worst[1]:+.2f} m)")
            else:
                t, g, x0, x1, linf = hit
                print(f"  r={r:.2f}: CONTACT-TRUE COLLISION g{g} at t={t:.2f}s "
                      f"(gate-frame x {x0:+.2f}->{x1:+.2f}, step-end Linf {linf:.2f}, "
                      f"pass band {band:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
