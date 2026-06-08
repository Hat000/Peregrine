"""Phase-0 SELECTION probe (orientation only, NOT the Phase-1 analysis).

For each candidate run: message-type histogram + rates, yaw range, lateral (y)
excursion, peak |world vy|, and a coarse body-vs-world vx check at the yaw=-180 segment.
Goal: pick a run that (a) has BOTH LOCAL_POSITION_NED and ODOMETRY (needed for the
interleaving mix) and (b) has a clear lateral move while pointing ~-180 (course -X).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from racer.frames import euler_from_quat_wxyz, world_vec_from_body_quat  # noqa: E402
from racer.recording import RecordingReader  # noqa: E402

RUNS = REPO.parent.parent.parent / "data" / "runs"  # main worktree data/runs
# NOTE: this worktree has no data/runs; resolve to the main checkout explicitly.
MAIN_RUNS = Path("C:/Users/Shadow/Peregrine/data/runs")


def _frame_name(val: int) -> str:
    """MAV_FRAME enum int -> name (e.g. 12 -> MAV_FRAME_BODY_FRD)."""
    from pymavlink import mavutil
    try:
        return mavutil.mavlink.enums["MAV_FRAME"][int(val)].name
    except Exception:  # noqa: BLE001
        return f"?{val}"


def probe(run: str) -> None:
    rdir = MAIN_RUNS / run
    rd = RecordingReader(rdir)
    hist: Counter = Counter()
    odo = []   # (t, roll_deg, pitch_deg, yaw_deg, vx_body, vy_body, vz_body, x, y, z, q)
    lpn = []   # (t, vx, vy, vz, x, y)
    frame_ids: Counter = Counter()
    child_frame_ids: Counter = Counter()
    for m in rd.iter_mavlink():
        t = m.get_type()
        hist[t] += 1
        ts = getattr(m, "_timestamp", 0.0)
        if t == "ODOMETRY":
            q = np.array([float(v) for v in m.q], dtype=np.float64)
            roll, pitch, yaw = euler_from_quat_wxyz(q)
            # ADD #1: the sim-declared frames for pose vs twist (MAV_FRAME enums).
            frame_ids[int(getattr(m, "frame_id", -1))] += 1
            child_frame_ids[int(getattr(m, "child_frame_id", -1))] += 1
            odo.append((ts, np.degrees(roll), np.degrees(pitch), np.degrees(yaw),
                        float(m.vx), float(m.vy), float(m.vz),
                        float(m.x), float(m.y), float(m.z), q))
        elif t == "LOCAL_POSITION_NED":
            lpn.append((ts, float(m.vx), float(m.vy), float(m.vz),
                        float(m.x), float(m.y)))

    dur = rd.meta.get("duration_s", 0.0)
    print(f"\n===== {run}  (final_state={rd.meta.get('final_state')}, "
          f"gate_index={rd.meta.get('gate_index')}, dur={dur:.1f}s) =====")
    top = ", ".join(f"{k}:{v}({v/dur:.0f}Hz)" for k, v in hist.most_common(8))
    print(f"  msgs: {top}")
    print(f"  ODOMETRY={hist.get('ODOMETRY',0)}  LOCAL_POSITION_NED={hist.get('LOCAL_POSITION_NED',0)}")
    # ADD #1: definitive sim-declared frames (MAV_FRAME enum ints + names).
    fr = ", ".join(f"{k}({_frame_name(k)}):{v}" for k, v in frame_ids.most_common())
    ch = ", ".join(f"{k}({_frame_name(k)}):{v}" for k, v in child_frame_ids.most_common())
    print(f"  ODOMETRY.frame_id       = {fr}")
    print(f"  ODOMETRY.child_frame_id = {ch}")

    if odo:
        roll = np.array([r[1] for r in odo])
        pitch = np.array([r[2] for r in odo])
        yaw = np.array([r[3] for r in odo])
        x = np.array([r[7] for r in odo])
        y = np.array([r[8] for r in odo])
        print(f"  ODO yaw deg: min={yaw.min():.0f} max={yaw.max():.0f}  "
              f"roll deg:[{roll.min():.1f},{roll.max():.1f}]  pitch deg:[{pitch.min():.1f},{pitch.max():.1f}]")
        print(f"  ODO x:[{x.min():.1f},{x.max():.1f}]  y:[{y.min():.2f},{y.max():.2f}]  "
              f"y-excursion={y.max()-y.min():.2f} m")
        # ADD #2: is there a segment with REAL pitch AND roll (not just yaw)? Count samples
        # where |roll|>5 and |pitch|>5 simultaneously (attitude-varied) and report the peak.
        rr = np.abs(roll); pp = np.abs(pitch)
        varied = (rr > 5.0) & (pp > 5.0)
        print(f"  ATTITUDE-VARIED (|roll|>5 & |pitch|>5): {int(varied.sum())} samples; "
              f"peak |roll|={rr.max():.1f} peak |pitch|={pp.max():.1f}")
    if lpn:
        vy = np.array([r[2] for r in lpn])
        vx = np.array([r[1] for r in lpn])
        print(f"  LPN world vx:[{vx.min():.2f},{vx.max():.2f}]  "
              f"vy:[{vy.min():.2f},{vy.max():.2f}]  max|vy|={np.abs(vy).max():.2f} m/s")

    # coarse body-vs-world check: at samples where yaw is near +-180, is ODO body vx
    # ~ equal-and-opposite to its rotated-world vx? (just a sanity peek, not Phase 1)
    if odo:
        near180 = [r for r in odo if abs(abs(r[3]) - 180.0) < 20.0]
        if near180:
            flips = 0
            shown = 0
            for r in near180:
                vb = np.array([r[4], r[5], r[6]])
                vw = world_vec_from_body_quat(vb, r[10])
                if abs(vb[0]) > 0.3 and np.sign(vw[0]) != np.sign(vb[0]):
                    flips += 1
                if shown < 3 and abs(vb[0]) > 0.3:
                    print(f"    near-180 sample yaw={r[3]:.0f}: body vx={vb[0]:+.2f} "
                          f"-> world vx={vw[0]:+.2f}  (flip={np.sign(vw[0])!=np.sign(vb[0])})")
                    shown += 1
            print(f"  near-180 ODO samples={len(near180)}, body-vs-world vx sign-flips "
                  f"(|vx|>0.3)={flips}")


if __name__ == "__main__":
    for run in sys.argv[1:]:
        try:
            probe(run)
        except Exception as e:  # noqa: BLE001
            print(f"\n!! {run}: {type(e).__name__}: {e}")
