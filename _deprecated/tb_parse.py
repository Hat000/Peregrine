"""Dump the key training scalars from a run's TensorBoard event file(s) for the report.

Usage: python tb_parse.py <run_dir>   (searches recursively for *tfevents* and prints sampled series
for success_rate / n_passed_gates / l_episode / total_reward / collision / progress).
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = sys.argv[1] if len(sys.argv) > 1 else "."
event_files = glob.glob(os.path.join(root, "**", "*tfevents*"), recursive=True)
print("RUN_DIR:", root)
print("EVENT_FILES:", event_files)

KEYS = ("success", "passed", "l_episode", "reward", "survive", "collision", "progress")


def sample(ev, k=14):
    n = len(ev)
    if n == 0:
        return []
    idxs = sorted(set([round(i * (n - 1) / (k - 1)) for i in range(k)]))
    return [(ev[i].step, round(ev[i].value, 4)) for i in idxs]


seen = set()
for ef in event_files:
    d = os.path.dirname(ef)
    acc = EventAccumulator(d, size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    want = [t for t in tags if any(kk in t.lower() for kk in KEYS)]
    for t in want:
        if t in seen:
            continue
        seen.add(t)
        ev = acc.Scalars(t)
        print(f"\n[{t}] n={len(ev)}")
        for step, val in sample(ev):
            print(f"   step={step:>6} {val}")
print("\nTB_PARSE_DONE")
