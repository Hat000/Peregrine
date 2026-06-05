"""Throwaway: is ODOMETRY.time_usec monotonic vs recv? Where are the jumps?"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path("C:/Users/Shadow/Peregrine/.claude/worktrees/blissful-sutherland-c8ebd9/src")
sys.path.insert(0, str(SRC))
from racer.recording import RecordingReader  # noqa: E402


def probe(run: Path, mtype: str = "ODOMETRY") -> None:
    reader = RecordingReader(run)
    seq = []  # (recv, time_usec)
    for msg in reader.iter_mavlink():
        if msg.get_type() != mtype:
            continue
        seq.append((float(msg._timestamp), int(msg.time_usec)))
    print(f"\n===== {run.name} / {mtype} =====  n={len(seq)}")
    if not seq:
        return
    r0 = seq[0][0]
    u0 = seq[0][1]
    # monotonic check on time_usec
    drops = 0
    big_jumps = []
    prev = seq[0][1]
    for i, (r, u) in enumerate(seq):
        if u < prev:
            drops += 1
        if abs((u - prev)) > 500_000 and i > 0:  # >0.5s jump between consecutive
            big_jumps.append((i, prev, u, u - prev))
        prev = u
    print(f"  non-monotonic steps (u<prev): {drops}")
    print(f"  consecutive jumps >0.5s: {len(big_jumps)} -> {big_jumps[:8]}")
    # show first 8 and a sampling
    print("  idx  recv_rel(s)  simusec_rel(s)")
    idxs = list(range(min(8, len(seq)))) + list(range(0, len(seq), max(1, len(seq)//12)))
    for i in sorted(set(idxs)):
        r, u = seq[i]
        print(f"  {i:5d}  {r-r0:10.3f}  {(u-u0)/1e6:12.3f}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        probe(Path(a))
