"""Replay REAL wire observations from the 9-gate record flight through the v1.7 actors.

This is a counterfactual, not a simulation: it asks "given the exact obs the wire
handed v16Qs1 at tick k, what would v17 have commanded?".  The trajectories would
diverge the moment the commands differ, so ONLY the first few ticks after the
takeoff-assist releases are meaningful -- which is exactly where the start dive lives.

ACTION VECTOR ORDER (derived from the log, not assumed):
    actor_mean = [thrust, roll, pitch, yaw]   (raw, pre-squash)
    act_raw    = [thrust in [0, 3.765] g,  rate_flu * 3.14 rad/s  x3]
Verified on the record flight tick 0: tanh(-0.57521) rescaled to [0,3.765] = 0.90515
== act_raw[0], and tanh(-0.26687)*3.14 = -0.8186 == act_raw[2].  So the PITCH channel
is index 2.  Sign: rate_frd[1] = +a_pitch, and a_pitch < 0 is nose-DOWN.

HARNESS VALIDATION: v16Qs1 is replayed first and its output compared against the
actor_mean recorded in the log.  If that does not match to ~1e-4 the harness is wrong
and every other number here is void.
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
RUN = DEPLOY / "data" / "runs" / "20260721_040555_v16pick_Qs1_f1"
V17_DIR = Path(r"C:\Users\Fengy\Downloads\Projects\wt-fix\data\ego-ckpts-v17-2026-07-21")

RATE_MAX = 3.14
THRUST_MAX = 3.765
PITCH = 2


def load_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def squash(raw: torch.Tensor) -> tuple[float, float, float, float]:
    """Deploy rescale: thrust -> [0, THRUST_MAX], rates -> +-RATE_MAX."""
    t = torch.tanh(raw)
    thrust = THRUST_MAX * (float(t[0]) + 1.0) / 2.0
    return thrust, float(t[1]) * RATE_MAX, float(t[2]) * RATE_MAX, float(t[3]) * RATE_MAX


def main() -> int:
    import fly_rl

    rows = load_rows(RUN / "ego_obs.jsonl")
    obs = torch.tensor([r["obs"] for r in rows], dtype=torch.float32)
    logged = torch.tensor([r["actor_mean"] for r in rows], dtype=torch.float32)
    assist = [bool(r.get("assist", False)) for r in rows]

    release = next((i for i, a in enumerate(assist) if not a), None)
    print(f"flight {RUN.name}: {len(rows)} ticks, assist releases at k={release}")
    print(f"resting obs[4] = {obs[0, 4]:+.4f} rad = {float(obs[0,4])*57.2958:+.2f} deg")
    print()

    models = {"v16Qs1 (PARENT, flew this)": SCRATCH / "control" / "v16Qs1_actor.pth"}
    for p in sorted(V17_DIR.glob("*_actor.pth")):
        models[p.stem.replace("_actor", "")] = p

    # ---- harness validation on the parent -------------------------------------
    parent = fly_rl.load_ego_actor(str(models["v16Qs1 (PARENT, flew this)"]))
    with torch.no_grad():
        repro = parent(obs)
    err = float((repro - logged).abs().max())
    print(f"HARNESS CHECK  max|replayed - logged actor_mean| = {err:.2e}", end="  ")
    if err > 1e-3:
        print("MISMATCH -- harness invalid, aborting.")
        return 1
    print("OK (replay reproduces the flown policy)")
    print()

    # ---- launch window ---------------------------------------------------------
    if release is None:
        print("no assist release in this flight")
        return 1
    lo, hi = max(0, release - 2), min(len(rows), release + 12)

    print(f"PITCH RATE COMMAND across the assist release (k={release}), rad/s")
    print("negative = nose DOWN.  Open-loop counterfactual: trustworthy for a few ticks only.")
    header = "  k  assist  " + "".join(f"{n[:12]:>14}" for n in models)
    print(header)
    print("  " + "-" * (len(header) - 2))

    outs = {}
    for name, path in models.items():
        actor = fly_rl.load_ego_actor(str(path))
        with torch.no_grad():
            outs[name] = actor(obs)

    for k in range(lo, hi):
        cells = "".join(f"{squash(outs[n][k])[PITCH]:>+14.3f}" for n in models)
        print(f"  {k:<3}  {str(assist[k]):<6}{cells}")

    # ---- summary over the first second of free flight --------------------------
    win = slice(release, min(len(rows), release + 30))
    print()
    print(f"FIRST ~1 s OF FREE FLIGHT (k={win.start}..{win.stop - 1})")
    print(f"  {'model':<28}{'mean pitch':>12}{'min (worst dive)':>20}{'ticks nose-down':>18}")
    for name in models:
        p = torch.stack([torch.tensor(squash(outs[name][k])[PITCH])
                         for k in range(win.start, win.stop)])
        print(f"  {name:<28}{float(p.mean()):>+12.3f}{float(p.min()):>+20.3f}"
              f"{int((p < 0).sum()):>15} /{len(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
