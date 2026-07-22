"""v1.8 pre-release gate: the launch-window counterfactual over EVERY logged flight.

Same instrument as replay_sweep.py (which produced the "v1.7 pitches 62% harder at
release on 385-386/388" finding), extended to the v1.8 arms and re-pointed at the v16
parent from the release asset.

WHY THIS AND NOT A SYNTHETIC PROBE: a hand-built at-rest obs is out of distribution --
every arm answers it with ~zero collective, so its pitch column is noise.  One flight's
logged obs at assist release is one REAL initial condition; the distribution over 400+
of them is the only honest read on "does v1.8 still dive at the handoff?".

TAILS, NOT MEANS (Fengyou, 2026-07-19): the start dive is a single hard nose-down
command, so the per-flight statistic is the WORST command in the first N_TICKS -- and
the columns that decide are p90-worst and MAX DIVE, not the mean.

Only N_TICKS after release are scored; past that the open-loop replay has drifted from
any state the alternative policy would actually occupy.

Sign: pitch_of() == rate_frd[1] under the deploy virtual_flip, so NEGATIVE = nose-DOWN
(fly_rl.py:748-759; the fence at :769 clips exactly this axis at 0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

DEPLOY = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix")
sys.path.insert(0, str(DEPLOY / "rl"))
sys.path.insert(0, str(DEPLOY / "src"))

DATA = Path(r"C:\Users\Fengy\Downloads\Projects\wt-fix\data")
RUNS = DEPLOY / "data" / "runs"

RATE_MAX, PITCH = 3.14, 2      # act[2] is pitch: [thrust, roll, pitch, yaw]
N_TICKS = 4

MODELS = [
    ("v16Qs1_PARENT", DATA / "ego-ckpts-v16-2026-07-19" / "v16Qs1_final_actor.pth"),
    ("v17Qs0", DATA / "ego-ckpts-v17-2026-07-21" / "v17Qs0_actor.pth"),
    ("v17Qs1", DATA / "ego-ckpts-v17-2026-07-21" / "v17Qs1_actor.pth"),
    ("v17Ws0", DATA / "ego-ckpts-v17-2026-07-21" / "v17Ws0_actor.pth"),
    ("v18Qs0", DATA / "ego-ckpts-v18-2026-07-22" / "v18Qs0_actor.pth"),
    ("v18Qs1", DATA / "ego-ckpts-v18-2026-07-22" / "v18Qs1_actor.pth"),
    ("v18Ws0", DATA / "ego-ckpts-v18-2026-07-22" / "v18Ws0_actor.pth"),
]


def pitch_of(raw: torch.Tensor) -> torch.Tensor:
    return torch.tanh(raw[:, PITCH]) * RATE_MAX


def main() -> int:
    import fly_rl

    missing = [n for n, p in MODELS if not p.exists()]
    if missing:
        print(f"MISSING checkpoints: {missing}")
        return 1
    actors = {n: fly_rl.load_ego_actor(str(p)) for n, p in MODELS}

    worst: dict[str, list[float]] = {n: [] for n, _ in MODELS}
    used = skipped = 0
    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            skipped += 1
            continue
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        if len(rows) < 8:
            skipped += 1
            continue
        release = next((i for i, r in enumerate(rows) if not bool(r.get("assist", False))),
                       None)
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
    print(f"Worst (most nose-DOWN) pitch-rate command in the first {N_TICKS} ticks after "
          f"assist release, rad/s.  NEGATIVE = nose DOWN.\n")
    print(f"  {'model':<16}{'mean':>9}{'median':>9}{'p90 worst':>11}{'MAX DIVE':>11}"
          f"{'dives harder than parent':>27}")
    print("  " + "-" * 83)

    base = torch.tensor(worst["v16Qs1_PARENT"])
    for n, _ in MODELS:
        v = torch.tensor(worst[n])
        srt = v.sort().values                       # most negative first
        p90 = float(srt[max(0, int(0.10 * len(srt)) - 1)])
        cmp_ = "" if n == "v16Qs1_PARENT" else f"{int((v < base).sum()):>17} / {len(v)}"
        print(f"  {n:<16}{float(v.mean()):>+9.3f}{float(v.median()):>+9.3f}"
              f"{p90:>+11.3f}{float(v.min()):>+11.3f}{cmp_:>27}")

    # The v1.8 thesis: M1 OFF should pull the release dive back toward the parent.
    print()
    for fam in ("v17", "v18"):
        vs = torch.tensor([w for n, _ in MODELS if n.startswith(fam) for w in worst[n]])
        print(f"  {fam} pooled: mean {float(vs.mean()):+.3f}  "
              f"p90-worst {float(vs.sort().values[int(0.10 * len(vs)) - 1]):+.3f}  "
              f"max dive {float(vs.min()):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
