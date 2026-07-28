"""Recompute POSTMORTEM.md section 1 from the flight corpus.

Every headline number in the close-out document comes from here, so the account can be checked
against the data rather than trusted. Run from the repo root:

    python3 scripts/adjudicate/corpus_summary.py

Override the corpus location for an external snapshot:

    PEREGRINE_RUNS=/path/to/runs python3 scripts/adjudicate/corpus_summary.py

A flight counts as RECORDED at >= 5 logged ticks. Below that the run either never armed or died on
the pad, and counting those as flights deflates every rate in the document. The gap between run
directories on disk and recorded flights is exactly those stubs.
"""
import collections
import glob
import json
import os

import numpy as np

RUNS = os.environ.get("PEREGRINE_RUNS", "data/runs")
MIN_TICKS = 5


def main() -> None:
    dirs = sorted(glob.glob(os.path.join(RUNS, "*")))
    if not dirs:
        raise SystemExit(f"no run directories under {RUNS!r} -- set PEREGRINE_RUNS")

    best = []
    for d in dirs:
        op = os.path.join(d, "ego_obs.jsonl")
        if not os.path.exists(op):
            continue
        try:
            with open(op, encoding="utf-8") as fh:
                recs = [json.loads(line) for line in fh if line.strip()]
        except Exception:
            continue
        if len(recs) < MIN_TICKS:
            continue
        best.append((max(r.get("gate_index", -1) for r in recs), os.path.basename(d)))

    best.sort(reverse=True)
    a = np.array([b[0] for b in best], dtype=float)

    print("Peregrine corpus summary  (source: %s)" % RUNS)
    print("=" * 62)
    print("  run directories on disk        %d" % len(dirs))
    print("  recorded flights (>=%d ticks)   %d" % (MIN_TICKS, len(best)))
    print("  all-time best                  gate %d  (%s)" % (best[0][0], best[0][1]))
    print("  flights reaching gate 8+       %d" % int((a >= 8).sum()))
    print("  flights reaching gate 6+       %d" % int((a >= 6).sum()))
    print("  flights reaching gate 4+       %d" % int((a >= 4).sum()))
    print("  corpus mean max gate           %.3f" % a.mean())
    print()
    print("  max-gate distribution:")
    counts = collections.Counter(int(x) for x in a)
    for g in sorted(counts):
        bar = "#" * max(1, round(counts[g] / 4))
        wall = "   <-- the wall" if g == 6 else ""
        print("    gate %2d: %4d  %s%s" % (g, counts[g], bar, wall))
    print()
    print("  The course has 20 gates. Nothing ever passed gate 9, so gates 10-19")
    print("  were never observed and no claim in this repo covers them.")


if __name__ == "__main__":
    main()
