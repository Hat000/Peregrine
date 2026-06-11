"""Coast-only replay: seed each twin variant from the LIVE state at coast start and replay
just the decay -- isolates the DRAG model from accel-phase thrust-map errors (the full-trace
RMS in replay_twin.py rewards legacy's compensating errors: too-low peak + too-slow decay).

Usage: .venv\\Scripts\\python.exe handoff\\shadowpc-twin-falsify-2026-06-10\\replay_coast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import runs as R
from replay_twin import CandidateMixed, CandidatePlant, seed_q
from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant
from racer.twin_fit import faithful_config

COAST_RUNS = ["drag_back08", "drag_back17", "drag_back25", "drag_back32",
              "drag_fwd17", "drag_fwd25", "drag_lat17p", "drag_lat17n", "drag_lat25p"]


def coast_replay(label, plant_cls, cfg):
    run = R.load(label)
    m = run.seg("coast")
    ks = np.where(m)[0]
    k0, k1 = ks[0] + 5, ks[-1]            # skip the first ~50 ms (phase transition)
    plant = plant_cls(cfg, position_ned=run.pos[k0], velocity_ned=run.vel[k0],
                      q_wxyz=seed_q(run.rpy[k0]))
    errs = []
    for k in range(k0, k1):
        dt = run.t[k + 1] - run.t[k]
        plant.step(ControlCommand(mode=ControlMode.BODY_RATE, body_rate=run.cmd[k],
                                  thrust=float(run.thr[k])), dt)
        errs.append(np.hypot(plant.vel[0], plant.vel[1])
                    - np.hypot(run.vel[k + 1, 0], run.vel[k + 1, 1]))
    errs = np.asarray(errs)
    return float(np.sqrt(np.mean(errs ** 2))), float(errs[-1])


def main():
    variants = [("legacy", CtbrPlant, faithful_config(False)),
                ("cand_quad", CandidatePlant, faithful_config(True)),
                ("cand_mix", CandidateMixed, faithful_config(True))]
    print(f"{'run':14s}" + "".join(f" | {nm:>9s} rms / end-err" for nm, *_ in variants))
    tot = {nm: [] for nm, *_ in variants}
    for lb in COAST_RUNS:
        row = f"{lb:14s}"
        for nm, cls, cfg in variants:
            rms, e = coast_replay(lb, cls, cfg)
            tot[nm].append(rms)
            row += f" | {rms:9.3f} / {e:+7.3f}"
        print(row)
    print("\nmean coast speed-RMS: " +
          "  ".join(f"{nm}={np.mean(v):.3f}" for nm, v in tot.items()))


if __name__ == "__main__":
    main()
