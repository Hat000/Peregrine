"""H4d: which attitude/rate convention is the live sim actually using?

For two candidate attitude models:
  RAW    : R = R(q_raw)              (reported quat taken at face value)
  UNDONE : R = R(-roll, pitch, yaw)  (fly_rl's roll-inversion artifact undo)

compute quat-finite-difference body rates  Omega_b = 2/dt * vec(q^-1 * q_next)
over the flight, and fit per-axis sign/gain of w_raw against each model's Omega.
Do it separately for the LEVEL phase (|pitch|<25 deg) and TILTED phase (>40 deg).
The model+sign-set that stays consistent across both phases is the real convention.

Also: verify velocity consistency per model: |R @ v_body - v_fd(pos)|.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from racer.frames import R_world_from_body, euler_from_quat_wxyz


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_from_R(R):
    # Shepperd's method
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s if s else 0, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def body_rates_fd(quats, dts):
    """Omega_body (rad/s) between consecutive wxyz quats (world-from-body)."""
    out = []
    for k in range(len(quats) - 1):
        dq = qmul(qconj(quats[k]), quats[k + 1])
        if dq[0] < 0:
            dq = -dq
        ang = 2.0 * np.arccos(np.clip(dq[0], -1, 1))
        ax = dq[1:4]
        n = np.linalg.norm(ax)
        w = (ang / max(dts[k], 1e-9)) * (ax / n) if n > 1e-12 else np.zeros(3)
        out.append(w)
    return np.array(out)


def load_run(name):
    rows = []
    with open(ROOT / "data" / "runs" / name / "debug_obs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "header":
                rows.append(r)
    return rows


def analyze(name, kmax=600):
    rows = load_run(name)[:kmax]
    t = np.array([r["t_mono"] for r in rows])
    dts = np.diff(t)
    q_raw = [np.array(r["q_raw_wxyz"], dtype=np.float64) for r in rows]
    q_raw = [q / np.linalg.norm(q) for q in q_raw]
    w_raw = np.array([r["w_raw"] for r in rows], dtype=np.float64)

    # model RAW
    om_raw = body_rates_fd(q_raw, dts)
    # model UNDONE: rebuild quats from (-roll, pitch, yaw)
    q_und = []
    for q in q_raw:
        roll, pitch, yaw = euler_from_quat_wxyz(q)
        q_und.append(q_from_R(R_world_from_body(-roll, pitch, yaw)))
    om_und = body_rates_fd(q_und, dts)

    pitch_und = np.array([euler_from_quat_wxyz(q)[1] for q in q_raw])  # pitch same both models
    lvl = np.abs(np.degrees(pitch_und[:-1])) < 25
    tlt = np.abs(np.degrees(pitch_und[:-1])) > 40

    def fit(om, mask, label):
        res = []
        for ax in range(3):
            x = w_raw[:-1][mask, ax]
            y = om[mask, ax]
            if np.std(x) < 1e-3 or mask.sum() < 20:
                res.append((float("nan"), float("nan")))
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            g = float(np.dot(y, x) / np.dot(x, x))
            res.append((c, g))
        print(f"    {label:<22} n={int(mask.sum()):>4}  "
              + "  ".join(f"{nm}: corr{c:+.2f} gain{g:+.2f}" for nm, (c, g)
                          in zip(["roll", "pitch", "yaw "], res)))

    print(f"\n==== {name} ====")
    print("  w_raw vs quat-FD body rates, per attitude model & phase:")
    print("  model RAW (reported quat as-is):")
    fit(om_raw, lvl, "level |pitch|<25deg")
    fit(om_raw, tlt, "tilted |pitch|>40deg")
    print("  model UNDONE (roll negated):")
    fit(om_und, lvl, "level |pitch|<25deg")
    fit(om_und, tlt, "tilted |pitch|>40deg")

    # velocity consistency per model: R @ v_body vs vel_ned (client value uses RAW quat)
    # pos-FD ground truth:
    pos = np.array([r["pos_ned"] for r in rows])
    v_fd = np.gradient(pos, axis=0) / np.gradient(t)[:, None]
    vN = np.array([r["vel_ned"] for r in rows])
    err_client = np.linalg.norm(vN - v_fd, axis=1)
    print(f"  vel_ned(client, raw-quat) vs pos-FD: med={np.median(err_client):.2f} "
          f"p95={np.percentile(err_client,95):.2f} m/s  (sanity)")


analyze("20260612_034852_inc6_standing_f1")
analyze("20260612_035458_inc6_standing_f7")
analyze("20260612_040440_inc6_bridge_f1")
