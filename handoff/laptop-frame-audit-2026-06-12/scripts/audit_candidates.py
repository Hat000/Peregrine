"""LAPTOP-FRAME-AUDIT: which interpretation of the ODOMETRY quat matches the
EXTERNAL invariants (FD of pristine vel_ned/pos_ned)?

Candidates (the only attitude re-interpretations that preserve hover/identity):
  RAW     q as-is                          (bcc93f9 assumption)
  CONJ_X  (w,  x, -y, -z) = R_x(pi)-conj   roll keep, pitch&yaw flip
  CONJ_Y  (w, -x,  y, -z) = R_y(pi)-conj   roll&yaw flip, pitch keep
  CONJ_Z  (w, -x, -y,  z) = R_z(pi)-conj   roll&pitch flip, yaw keep

Tests (tilted-phase only; level-flight correlation is inadmissible):
  A  force projection: measured world accel (central FD of vel_ned) vs
     model = K(coll,d=2)*L(|v|)*(R@[0,0,-1]) + quad_drag(R) + g, per axis,
     binned by tilt angle (invariant across candidates).
  B  heading vs course: candidate body-x heading vs velocity course at speed
     (only ticks where |sin(yaw)| > 0.3 discriminate).
  C  quat-FD body rate of each candidate vs the raw rate channel (per-axis
     corr sign -> pins the rate-channel mapping for the winning attitude).
  D  pos-FD vs vel_ned during bank (integrity of the translational truth).
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from racer.rl_plant import (COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            QUAD_DRAG_C2_MEASURED, LAPSE_SPEED_MEASURED,
                            LAPSE_FACTOR_MEASURED)

G = 9.80665
DATA = ROOT / "handoff/shadowpc-refit-dataset-2026-06-12/extracted"

CANDS = {
    "RAW":    np.array([1.0,  1.0,  1.0,  1.0]),
    "CONJ_X": np.array([1.0,  1.0, -1.0, -1.0]),
    "CONJ_Y": np.array([1.0, -1.0,  1.0, -1.0]),
    "CONJ_Z": np.array([1.0, -1.0, -1.0,  1.0]),
}


def load(run):
    rows = [json.loads(l) for l in open(DATA / run / "debug_obs.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("type") != "header"]
    return dict(
        t=np.array([r["t_mono"] for r in rows]),
        pos=np.array([r["pos_ned"] for r in rows]),
        vel=np.array([r["vel_ned"] for r in rows]),
        q=np.array([r["q_raw_wxyz"] for r in rows]),
        w=np.array([r["w_raw"] for r in rows]),
        coll=np.array([r["collective"] for r in rows]),
    )


def Rmats(q_wxyz):
    return Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_matrix()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


RUNS = sorted(p.name for p in DATA.iterdir() if p.is_dir())

# ---------------------------------------------------------------- gather per-tick samples
samp = {c: {"model": [], } for c in CANDS}
meas_all, tilt_all, run_id = [], [], []
hd = {c: [] for c in CANDS}          # heading-course error samples
course_all = []
fdrate = {c: [[], [], []] for c in CANDS}   # quat-FD rates per axis
wraw_axis = [[], [], []]
posfd, velmid = [], []

for ri, run in enumerate(RUNS):
    d = load(run)
    n = len(d["t"])
    if n < 8:
        continue
    R = {c: Rmats(d["q"] * s[None, :]) for c, s in CANDS.items()}
    for k in range(2, n - 2):
        dt = d["t"][k + 1] - d["t"][k - 1]
        if not (0.05 < dt < 0.09):
            continue
        a_meas = (d["vel"][k + 1] - d["vel"][k - 1]) / dt
        if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
            continue
        v = d["vel"][k]
        K = np.interp(d["coll"][k - 2], COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
        L = np.interp(np.linalg.norm(v), LAPSE_SPEED_MEASURED, LAPSE_FACTOR_MEASURED)
        tilt = np.degrees(np.arccos(np.clip(R["RAW"][k][2, 2], -1, 1)))
        meas_all.append(a_meas)
        tilt_all.append(tilt)
        run_id.append(ri)
        for c in CANDS:
            Rk = R[c][k]
            thr = K * L * (Rk @ [0.0, 0.0, -1.0])
            vb = Rk.T @ v
            cc = np.where(vb >= 0, QUAD_DRAG_C2_MEASURED[:, 0], QUAD_DRAG_C2_MEASURED[:, 1])
            drag = Rk @ (-cc * np.abs(vb) * vb)
            samp[c]["model"].append(thr + drag + np.array([0.0, 0.0, G]))
        # --- B: heading vs course ---
        sp = np.linalg.norm(v[:2])
        yaw_raw = np.arctan2(R["RAW"][k][1, 0], R["RAW"][k][0, 0])
        if sp > 6.0 and abs(np.sin(yaw_raw)) > 0.3:
            course = np.arctan2(v[1], v[0])
            course_all.append(course)
            for c in CANDS:
                bx = R[c][k][:, 0]
                hd[c].append(abs(wrap(np.arctan2(bx[1], bx[0]) - course)))
        # --- C: quat-FD rates (one-step) ---
        dtk = d["t"][k + 1] - d["t"][k]
        if 0.02 < dtk < 0.06:
            for c in CANDS:
                rv = Rotation.from_matrix(R[c][k].T @ R[c][k + 1]).as_rotvec() / dtk
                for ax in range(3):
                    fdrate[c][ax].append(rv[ax])
            for ax in range(3):
                wraw_axis[ax].append(d["w"][k][ax])
        # --- D: pos-FD vs vel (banked only) ---
        if tilt > 30:
            posfd.append((d["pos"][k + 1] - d["pos"][k - 1]) / dt)
            velmid.append(v)

meas = np.array(meas_all)
tilt = np.array(tilt_all)
print(f"runs={len(RUNS)}  samples={len(meas)}  banked(>35deg)={np.sum(tilt > 35)}")

# ---------------------------------------------------------------- A: per-axis, tilt-binned
BINS = [(0, 15), (15, 35), (35, 90)]
AX = ["N", "E", "D"]
print("\n=== TEST A: force-projection residual  (median |model-meas| m/s^2  /  corr) ===")
hdr = "cand    bin(tilt) n     " + "   ".join(f"{a}: med|r| corr " for a in AX)
print(hdr)
for c in CANDS:
    M = np.array(samp[c]["model"])
    for lo, hi in BINS:
        m = (tilt >= lo) & (tilt < hi)
        if m.sum() < 20:
            continue
        cells = []
        for ax in range(3):
            r = np.median(np.abs(M[m, ax] - meas[m, ax]))
            cor = np.corrcoef(M[m, ax], meas[m, ax])[0, 1] if M[m, ax].std() > 1e-6 else np.nan
            cells.append(f"{r:7.2f} {cor:+5.2f}")
        print(f"{c:7s} {lo:>3}-{hi:<3}  {m.sum():5d}  " + "   ".join(cells))

print("\n=== TEST B: heading vs course (median |err| deg; speed>6, |sin yaw|>0.3) ===")
for c in CANDS:
    if hd[c]:
        print(f"{c:7s}  n={len(hd[c]):5d}  med={np.degrees(np.median(hd[c])):7.1f} deg")

print("\n=== TEST C: quat-FD(candidate) vs raw rate channel (per-axis corr) ===")
for c in CANDS:
    cors = []
    for ax in range(3):
        a = np.array(fdrate[c][ax]); b = np.array(wraw_axis[ax])
        cors.append(np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 else np.nan)
    print(f"{c:7s}  corr(roll,pitch,yaw) = [{cors[0]:+.2f}, {cors[1]:+.2f}, {cors[2]:+.2f}]")

print("\n=== TEST D: pos-FD vs vel_ned during bank (tilt>30) ===")
pf, vm = np.array(posfd), np.array(velmid)
if len(pf):
    for ax in range(3):
        cor = np.corrcoef(pf[:, ax], vm[:, ax])[0, 1]
        g = np.polyfit(vm[:, ax], pf[:, ax], 1)[0]
        print(f"  axis {AX[ax]}: n={len(pf)}  corr={cor:+.3f}  gain={g:+.3f}")
