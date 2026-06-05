"""Throwaway: confirm the extract JSON carries the sysid signal (don't ship blind).

For a *_rate* extract: per axis, gather samples with a steady non-trivial cmd, compute the realized
body rate by quaternion finite-difference between consecutive samples, and report realized/cmd. We
EXPECT |gain| ~2.4-2.7 with roll/yaw sign-inverted vs cmd (memo sysid), and the raw ODOMETRY rate
sign-inverted on pitch. This re-derives the plant from MY json to prove the join is coherent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("C:/Users/Shadow/Peregrine/.claude/worktrees/blissful-sutherland-c8ebd9/src")
sys.path.insert(0, str(SRC))
import numpy as np  # noqa: E402

from racer.frames import body_rate_from_quats  # noqa: E402

AX = {0: "roll", 1: "pitch", 2: "yaw"}


def check(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    S = d["samples"]
    print(f"\n=== {d['run']}  n={len(S)} ===")
    # realized rate via quaternion finite-diff (sign-correct), aligned to sample i
    rea = [np.zeros(3)]
    for i in range(1, len(S)):
        dt = (S[i]["t_ns"] - S[i - 1]["t_ns"]) / 1e9
        rea.append(body_rate_from_quats(S[i - 1]["odo_q_wxyz"], S[i]["odo_q_wxyz"], dt))
    for ax in (0, 1, 2):
        cmd = np.array([(s["cmd_body_rate"][ax] if s["cmd_body_rate"] else 0.0) for s in S])
        rr = np.array([rea[i][ax] for i in range(len(S))])
        odo = np.array([s["odo_angular_rate"][ax] for s in S])
        sel = np.abs(cmd) > 0.1                      # steady doublet held
        if sel.sum() < 5:
            print(f"  {AX[ax]:>5}: (no steady cmd>0.1)")
            continue
        # robust gain = slope through origin of realized vs cmd on held samples
        gain = float(np.sum(cmd[sel] * rr[sel]) / np.sum(cmd[sel] * cmd[sel]))
        odo_ratio = float(np.median(odo[sel] / rr[sel])) if np.all(np.abs(rr[sel]) > 1e-3) else float("nan")
        print(f"  {AX[ax]:>5}: realized/cmd gain={gain:+.2f}  odo/realized={odo_ratio:+.2f}  "
              f"n_held={int(sel.sum())}  cmd~{np.median(cmd[sel]):+.2f}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        check(Path(a))
