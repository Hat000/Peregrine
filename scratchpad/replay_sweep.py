"""Sweep the launch-window counterfactual across EVERY logged flight.

One flight is one initial condition.  The question "did v1.7 kill the start dive?"
needs the distribution over initial conditions, and it needs TAILS, not means: the
start dive is a single hard nose-down command at handoff, so the statistic that
matters is the WORST command in the first few ticks, per flight.

Only the first N_TICKS after assist release are scored -- beyond that the open-loop
replay has drifted from any state the alternative policy would actually be in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

DEPLOY = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix")
sys.path.insert(0, str(DEPLOY / "rl"))
sys.path.insert(0, str(DEPLOY / "src"))

SCRATCH = Path(__file__).resolve().parent
RUNS = DEPLOY / "data" / "runs"
V17_DIR = Path(r"C:\Users\Fengy\Downloads\Projects\wt-fix\data\ego-ckpts-v17-2026-07-21")

RATE_MAX, THRUST_MAX, PITCH = 3.14, 3.765, 2
N_TICKS = 4           # ticks after release that the open-loop replay can be trusted for


def pitch_of(raw: torch.Tensor) -> torch.Tensor:
    return torch.tanh(raw[:, PITCH]) * RATE_MAX


def main() -> int:
    import fly_rl

    models = {"v16Qs1_PARENT": SCRATCH / "control" / "v16Qs1_actor.pth"}
    for p in sorted(V17_DIR.glob("*_actor.pth")):
        models[p.stem.replace("_actor", "")] = p
    actors = {n: fly_rl.load_ego_actor(str(p)) for n, p in models.items()}

    worst: dict[str, list[float]] = {n: [] for n in models}
    used = 0
    skipped = 0

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            skipped += 1
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        if len(rows) < 8:
            skipped += 1
            continue
        assist = [bool(r.get("assist", False)) for r in rows]
        release = next((i for i, a in enumerate(assist) if not a), None)
        if release is None or release + N_TICKS >= len(rows):
            skipped += 1
            continue
        obs = torch.tensor([r["obs"] for r in rows[release:release + N_TICKS]],
                           dtype=torch.float32)
        used += 1
        for n, actor in actors.items():
            with torch.no_grad():
                worst[n].append(float(pitch_of(actor(obs)).min()))

    print(f"{used} flights scored, {skipped} skipped (no obs log / no assist release)")
    print(f"Worst (most nose-DOWN) pitch-rate command in the first {N_TICKS} ticks "
          f"after assist release, rad/s")
    print()
    print(f"  {'model':<16}{'mean':>9}{'median':>9}{'p90 worst':>11}{'MAX DIVE':>11}"
          f"{'flights worse than parent':>28}")
    print("  " + "-" * 82)

    base = torch.tensor(worst["v16Qs1_PARENT"])
    for n in models:
        v = torch.tensor(worst[n])
        srt = v.sort().values                      # most negative first
        p90 = float(srt[max(0, int(0.10 * len(srt)) - 1)])
        cmp_ = "" if n == "v16Qs1_PARENT" else f"{int((v < base).sum()):>18} / {len(v)}"
        print(f"  {n:<16}{float(v.mean()):>+9.3f}{float(v.median()):>+9.3f}"
              f"{p90:>+11.3f}{float(v.min()):>+11.3f}{cmp_:>28}")

    print()
    print("  negative = nose DOWN; 'worse than parent' = dives harder on that flight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
