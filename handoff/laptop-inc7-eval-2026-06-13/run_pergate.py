"""Run offline_rollout.py N times on the VQ1 course and aggregate per-gate L-inf stats.

Usage (from repo root, .venv active):
  python handoff/laptop-inc7-eval-2026-06-13/run_pergate.py --ckpt <actor.pth> [--n 30]
"""
import argparse
import subprocess
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "rl" / "offline_rollout.py"
PYTHON = sys.executable


def run_one(ckpt: str, body_radius: float, frame_depth: float, seed: int):
    cmd = [
        PYTHON, str(SCRIPT),
        "--checkpoint", ckpt,
        "--plant", "mixer",
        "--start", "simstart",
        "--body-radius", str(body_radius),
        "--frame-depth", str(frame_depth),
        "--skip-check",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


def parse_output(text: str):
    gate_linf = {}   # gate_id -> Linf
    outcome = "UNKNOWN"
    for line in text.splitlines():
        m = re.search(r"gate (\d) PASS.*Linf=([0-9.]+)m", line)
        if m:
            gate_linf[int(m.group(1))] = float(m.group(2))
        m = re.search(r"gate (\d) COLLISION.*off=\[([+\-0-9.]+),([+\-0-9.]+)\]", line)
        if m:
            gate_linf[int(m.group(1))] = None  # mark collision at gate
        m = re.search(r"\[outcome\] (\w+)", line)
        if m:
            outcome = m.group(1)
    return gate_linf, outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--body-radius", type=float, default=0.33, help="nominal: midpoint of [0.28,0.38]")
    ap.add_argument("--frame-depth", type=float, default=0.30)
    ap.add_argument("--n", type=int, default=30, help="number of episodes to run")
    args = ap.parse_args()

    per_gate_linf = defaultdict(list)  # gate -> list of L-inf values (only for passed gates)
    per_gate_collision = defaultdict(int)
    outcomes = []

    for i in range(args.n):
        print(f"\r  episode {i+1}/{args.n} ...", end="", flush=True)
        out = run_one(args.ckpt, args.body_radius, args.frame_depth, i)
        linf_map, outcome = parse_output(out)
        outcomes.append(outcome)
        for g, v in linf_map.items():
            if v is None:
                per_gate_collision[g] += 1
            else:
                per_gate_linf[g].append(v)

    print()
    n = args.n
    n_finish = outcomes.count("FINISHED")
    print(f"\n=== PER-GATE ANALYSIS (body_radius={args.body_radius}, frame_depth={args.frame_depth}, n={n}) ===")
    print(f"Overall: {n_finish}/{n} FINISHED  {n_finish/n*100:.1f}%")
    print(f"\n{'gate':>4}  {'n_pass':>6}  {'n_coll':>6}  {'Linf_p50':>8}  {'Linf_p90':>8}  "
          f"{'Linf_max':>8}  {'margin_p5':>9}  {'margin_min':>10}")
    half_open = 0.75
    for g in range(6):
        passes = per_gate_linf.get(g, [])
        colls = per_gate_collision.get(g, 0)
        if passes:
            arr = np.array(passes)
            margins = (half_open - args.body_radius) - arr
            print(f"{g:>4}  {len(passes):>6}  {colls:>6}  {np.median(arr):>8.3f}  "
                  f"{np.percentile(arr,90):>8.3f}  {arr.max():>8.3f}  "
                  f"{np.percentile(margins,5):>9.3f}  {margins.min():>10.3f}")
        else:
            print(f"{g:>4}  {0:>6}  {colls:>6}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  "
                  f"{'N/A':>9}  {'N/A':>10}")


if __name__ == "__main__":
    main()
