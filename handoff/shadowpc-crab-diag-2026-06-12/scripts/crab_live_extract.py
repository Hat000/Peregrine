"""crab_live_extract.py -- reconstruct TRUE physical attitude + course from live
frame-audit recordings and compare against what the policy PERCEIVED (obs).

For each tick in debug_obs.jsonl we have the raw telemetry the deploy loop read
(pos_ned, vel_ned, q_raw_wxyz, w_raw) AND the 17-dim obs the policy consumed.

We reconstruct:
  * TRUE physical attitude  = euler(conjugate(q_raw))   [FRAME-AUDIT R_y(pi)]
  * heading (true yaw, NED), roll, pitch  -- deg
  * course_vel  = atan2(vE, vN)  from PRISTINE vel_ned   (external invariant)
  * gate-course = direction toward the active gate (pristine pos -> gate)  (deg)
  * crab_vs_vel  = wrap(course_vel - heading)   (sideslip: nose vs velocity)
  * crab_vs_gate = wrap(gate_course - heading)  (heading error vs where gate is)
  * obs rpy_g (what the policy saw, incl. virtual flip), obs pos_g / vel_g lateral

The point: pos_ned / vel_ned are the TRUSTED anchor (frame audit: pos-FD==vel
through 60 deg bank). If the policy SEES the lateral drift in pos_g/vel_g but the
physical attitude crabs anyway -> not an obs-blindness story. If the policy's
perceived heading says "aligned" while the external course shows a steady crab
-> obs heading bug.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

# FRAME-AUDIT conjugation (negate x,z of wxyz quat) -> true FRD->NED attitude.
ODO_CONJ = np.array([1.0, -1.0, 1.0, -1.0])
# Gate centres (NED) from fly_rl (_GATE_POS_ZUP * [1,-1,-1]).
from fly_rl import _GATE_POS_ZUP, _FLIP  # noqa: E402
GATE_NED = _GATE_POS_ZUP * _FLIP


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def true_rpy_deg(q_raw_wxyz):
    qt = np.asarray(q_raw_wxyz, float) * ODO_CONJ
    n = qt @ qt
    if n < 1e-12:
        return np.zeros(3)
    yaw, pitch, roll = Rotation.from_quat([qt[1], qt[2], qt[3], qt[0]]).as_euler("ZYX")
    return np.degrees([roll, pitch, yaw])


def asis_rpy_deg(q_raw_wxyz):
    q = np.asarray(q_raw_wxyz, float)
    n = q @ q
    if n < 1e-12:
        return np.zeros(3)
    yaw, pitch, roll = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("ZYX")
    return np.degrees([roll, pitch, yaw])


def process(run_dir: Path, speed_min=3.0):
    rows = []
    header = None
    for line in (run_dir / "debug_obs.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("type") == "header":
            header = d
            continue
        if "q_raw_wxyz" not in d:
            continue
        rows.append(d)
    if not rows:
        return None
    out = []
    for d in rows:
        pos = np.array(d["pos_ned"], float)
        vel = np.array(d["vel_ned"], float)
        q = d["q_raw_wxyz"]
        gi = d.get("gate_index", 0)
        roll, pitch, yaw = true_rpy_deg(q)
        spd_h = float(np.hypot(vel[0], vel[1]))
        spd = float(np.linalg.norm(vel))
        course_vel = np.degrees(np.arctan2(vel[1], vel[0])) if spd_h > 0.2 else np.nan
        gvec = GATE_NED[min(gi, len(GATE_NED) - 1)] - pos
        gate_course = np.degrees(np.arctan2(gvec[1], gvec[0]))
        obs = d.get("obs", [np.nan] * 17)
        out.append(dict(
            k=d.get("k"), t=d.get("sim_time_ns", 0) / 1e9, gi=gi, spd=spd, spd_h=spd_h,
            roll=roll, pitch=pitch, yaw=yaw,
            course_vel=course_vel, gate_course=gate_course,
            crab_vs_vel=wrap_deg(course_vel - yaw) if not np.isnan(course_vel) else np.nan,
            crab_vs_gate=wrap_deg(gate_course - yaw),
            obs_posg_y=obs[1], obs_velg_y=obs[4],
            obs_rpy_r=np.degrees(obs[6]), obs_rpy_p=np.degrees(obs[7]), obs_rpy_y=np.degrees(obs[8]),
        ))
    return header, out


def summarize(name, header, rows, speed_min=3.0):
    cruise = [r for r in rows if r["spd"] >= speed_min and not np.isnan(r["crab_vs_vel"])]
    nall = len(rows)
    print(f"\n==== {name}  ({nall} ticks, {len(cruise)} cruise@>{speed_min}m/s, "
          f"vflip={header.get('virtual_flip')}) ====")
    if not cruise:
        print("  (no cruise ticks)")
        return
    def stat(key):
        v = np.array([r[key] for r in cruise], float)
        v = v[~np.isnan(v)]
        return np.mean(v), np.median(v), np.min(v), np.max(v)
    for key in ["roll", "pitch", "yaw", "course_vel", "gate_course",
                "crab_vs_vel", "crab_vs_gate", "obs_posg_y", "obs_velg_y",
                "obs_rpy_r", "obs_rpy_p", "obs_rpy_y"]:
        m, md, lo, hi = stat(key)
        print(f"  {key:14s} mean={m:+8.2f}  med={md:+8.2f}  [{lo:+8.2f}, {hi:+8.2f}]")


if __name__ == "__main__":
    runs = sys.argv[1:]
    if not runs:
        base = ROOT / "data" / "runs"
        runs = sorted(str(p) for p in base.glob("20260612_18*rl_inc6_frameaudit*"))
    for r in runs:
        rd = Path(r)
        res = process(rd)
        if res is None:
            print(f"\n==== {rd.name}: no usable ticks ====")
            continue
        header, rows = res
        summarize(rd.name, header, rows)
