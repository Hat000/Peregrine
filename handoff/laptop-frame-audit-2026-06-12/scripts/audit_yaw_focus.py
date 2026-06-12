"""Yaw-axis discrimination, clean regime: rollfix f1/f2 smooth coordinated phase
(post-fix, pre-divergence) + course-rate vs candidate yaw-rate during the banked
turn. Test A pins b3 (2 of 3 DOF); this pins rotation ABOUT the thrust axis.
Also: per-axis gain of quat-FD(CONJ_Y) vs -w_raw."""
import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "handoff/shadowpc-refit-dataset-2026-06-12/extracted"

CANDS = {
    "RAW":    np.array([1.0,  1.0,  1.0,  1.0]),
    "CONJ_Y": np.array([1.0, -1.0,  1.0, -1.0]),
}


def load(run):
    rows = [json.loads(l) for l in open(DATA / run / "debug_obs.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("type") != "header"]
    return dict(t=np.array([r["t_mono"] for r in rows]),
                vel=np.array([r["vel_ned"] for r in rows]),
                q=np.array([r["q_raw_wxyz"] for r in rows]),
                w=np.array([r["w_raw"] for r in rows]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


for run in ["20260612_044219_inc6_rollfix_f1", "20260612_044438_inc6_rollfix_f2"]:
    d = load(run)
    n = len(d["t"])
    R = {c: Rotation.from_quat((d["q"] * s[None, :])[:, [1, 2, 3, 0]]).as_matrix()
         for c, s in CANDS.items()}
    print(f"\n=== {run.split('_', 2)[2]} ===")
    print(f"{'k':>3} {'spd':>5} {'tilt':>4} | {'course':>7} {'crs_rate':>8} | "
          f"{'RAW yaw':>8} {'yawFD':>6} {'sideslip':>8} | {'CONJY yaw':>9} {'yawFD':>6} {'sideslip':>8}")
    for k in range(8, min(n - 2, 70), 4):
        dt = d["t"][k + 1] - d["t"][k - 1]
        if not (0.05 < dt < 0.09):
            continue
        v = d["vel"][k]
        sp = np.linalg.norm(v[:2])
        if sp < 4.0:
            continue
        course = np.arctan2(v[1], v[0])
        v1, v0 = d["vel"][k + 1], d["vel"][k - 1]
        crate = wrap(np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])) / dt
        tilt = np.degrees(np.arccos(np.clip(R["RAW"][k][2, 2], -1, 1)))
        cells = []
        for c in CANDS:
            Rk = R[c][k]
            yaw = np.arctan2(Rk[1, 0], Rk[0, 0])
            rv = Rotation.from_matrix(R[c][k - 1].T @ R[c][k + 1]).as_rotvec() / dt
            ss = np.degrees(wrap(course - yaw))
            cells.append(f"{np.degrees(yaw):>8.0f} {rv[2]:>6.2f} {ss:>8.0f}")
        print(f"{k:>3} {sp:>5.1f} {tilt:>4.0f} | {np.degrees(course):>7.0f} "
              f"{crate:>8.2f} | {cells[0]} | {cells[1]}")

# --- gain of quat-FD(CONJ_Y) vs -w_raw, all 17 runs ---
print("\n=== gain check: quat-FD(CONJ_Y) vs -w_raw (all runs) ===")
RUNS = sorted(p.name for p in DATA.iterdir() if p.is_dir())
fd, wr = [[], [], []], [[], [], []]
for run in RUNS:
    d = load(run)
    q = d["q"] * CANDS["CONJ_Y"][None, :]
    Rm = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    for k in range(1, len(d["t"]) - 1):
        dtk = d["t"][k + 1] - d["t"][k]
        if not (0.02 < dtk < 0.06):
            continue
        rv = Rotation.from_matrix(Rm[k].T @ Rm[k + 1]).as_rotvec() / dtk
        for ax in range(3):
            fd[ax].append(rv[ax])
            wr[ax].append(-d["w"][k][ax])
for ax, nm in enumerate(["roll", "pitch", "yaw"]):
    a, b = np.array(fd[ax]), np.array(wr[ax])
    print(f"  {nm:5s}: corr={np.corrcoef(a, b)[0, 1]:+.3f}  "
          f"gain(fd~ -w_raw)={np.polyfit(b, a, 1)[0]:+.3f}  n={len(a)}")
