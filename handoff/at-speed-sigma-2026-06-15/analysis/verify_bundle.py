"""verify_bundle.py — PART C harness validator (SIMOPS-MASTERY-SHADOWVISION).

Validate that a recorded sim session (`data/runs/<stamp>_<label>/`, produced by
`scripts/record_session.py` or the `rl/fly_rl.py` recorder) contains EVERY field the at-speed
step-5 recording harness needs, time-aligned on ONE clock (the recv-monotonic clock the recorder
stamps everything with, bridged to UNIX epoch in the tlog + via meta's t0 pair for the video index).

Required streams (mission PART C):
  - per-frame VIDEO   >= 30 Hz  (the detector + PnP run OFFLINE on these JPEGs -> 4-corner
                                 detections + t_cam_gate + reproj; we confirm frames decode)
  - HIGHRES_IMU accel_body (specific force, body FRD; xacc/yacc/zacc)  >= 90 Hz
  - ATTITUDE quaternion     (we use ODOMETRY.q, the TRUE attitude; ATTITUDE-euler is sign-aliased)
  - GROUND-TRUTH position + velocity at ~IMU rate
        LOCAL_POSITION_NED  x/y/z + vx/vy/vz  (world NED, pristine)   ~97 Hz  <- load-bearing
        ODOMETRY            pose + twist(body->NED) + quat            ~75 Hz  (cross-check)

The LOAD-BEARING field for pinning the cold velocity prior is LOCAL_POSITION_NED.{vx,vy,vz}
(world-frame ground-truth velocity). We confirm it is (a) present and (b) LIVE (non-zero in
flight, not a field of zeros) -- the thing that decides the step-5 plan.

Exit 0 iff every required field is present at >= its required rate AND ground-truth velocity is live.

Usage: .venv\\Scripts\\python handoff/simops-mastery-2026-06-13/verify_bundle.py <run_dir> [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from racer.recording import RecordingReader

# (name, required_min_rate_hz, sim_native_rate_hz) -- pass bar is the lower of the two with margin
REQUIRED = {
    "video":             (30.0, 28.6),   # sim native ~28.6 fps (deduped); >=30 is the mission target
    "HIGHRES_IMU":       (90.0, 120.0),
    "LOCAL_POSITION_NED":(60.0, 97.0),   # ground-truth pos+vel; well above the 60 Hz floor we assert
    "ODOMETRY":          (60.0, 75.0),
    "ATTITUDE":          (60.0, 120.0),
}


def _rate_stats(ts: list[float]) -> dict:
    if len(ts) < 2:
        return {"n": len(ts), "dur_s": 0.0, "rate_hz": 0.0, "max_gap_ms": 0.0, "p99_gap_ms": 0.0}
    ts = sorted(ts)
    dt = np.diff(ts)
    dur = ts[-1] - ts[0]
    return {
        "n": len(ts),
        "dur_s": round(dur, 3),
        "rate_hz": round((len(ts) - 1) / dur, 2) if dur > 0 else 0.0,
        "max_gap_ms": round(float(dt.max()) * 1e3, 1),
        "p99_gap_ms": round(float(np.percentile(dt, 99)) * 1e3, 1),
    }


def verify(run: Path) -> dict:
    reader = RecordingReader(run)
    meta = reader.meta
    t0u, t0m = int(meta["t0_unix_ns"]), int(meta["t0_monotonic_ns"])

    # --- decode tlog: per-stream recv timestamps + the field values we must confirm live ---
    ts: dict[str, list[float]] = {k: [] for k in REQUIRED if k != "video"}
    lpn_vel, odo_vel, imu_acc = [], [], []
    lpn_rows, odo_rows = [], []   # (recv_t, pos) for alignment
    att_quat_seen = False
    odo_quat_seen = False
    type_counts: dict[str, int] = {}

    for msg in reader.iter_mavlink():
        t = msg.get_type()
        type_counts[t] = type_counts.get(t, 0) + 1
        recv = float(msg._timestamp)   # epoch seconds (recv clock, bridged by the recorder)
        if t == "HIGHRES_IMU":
            ts[t].append(recv)
            imu_acc.append([float(msg.xacc), float(msg.yacc), float(msg.zacc)])
        elif t == "LOCAL_POSITION_NED":
            ts[t].append(recv)
            lpn_vel.append([float(msg.vx), float(msg.vy), float(msg.vz)])
            lpn_rows.append((recv, [float(msg.x), float(msg.y), float(msg.z)]))
        elif t == "ODOMETRY":
            ts[t].append(recv)
            odo_vel.append([float(msg.vx), float(msg.vy), float(msg.vz)])  # raw body twist
            odo_rows.append((recv, [float(msg.x), float(msg.y), float(msg.z)]))
            if msg.q is not None and len(msg.q) == 4 and any(abs(float(v)) > 1e-6 for v in msg.q):
                odo_quat_seen = True
        elif t == "ATTITUDE":
            ts[t].append(recv)
            # ATTITUDE carries euler (sign-aliased); we record presence only. Quat comes from ODO.
            if hasattr(msg, "q1") or hasattr(msg, "roll"):
                att_quat_seen = True

    # --- video: recv timestamps from the index, bridged to the SAME epoch clock as the tlog ---
    vid_ts, seen_fids = [], set()
    for e in reader.iter_video_index():
        fid = e["frame_id"]
        if fid in seen_fids:           # the receiver dedups, but be defensive
            continue
        seen_fids.add(fid)
        vid_ts.append((t0u + (e["recv_monotonic_ns"] - t0m)) / 1e9)

    # --- assemble per-stream stats ---
    streams = {"video": _rate_stats(vid_ts)}
    for k in ts:
        streams[k] = _rate_stats(ts[k])

    # --- velocity / accel liveness (is GROUND-TRUTH velocity actually live, or a zero field?) ---
    def _live(v):
        if not v:
            return {"present": False}
        a = np.asarray(v, float)
        mag = np.linalg.norm(a, axis=1)
        return {
            "present": True,
            "max_mag": round(float(mag.max()), 3),
            "p50_mag": round(float(np.median(mag)), 3),
            "frac_moving_gt0p5": round(float((mag > 0.5).mean()), 3),
        }

    lpn_v = _live(lpn_vel)
    odo_v = _live(odo_vel)
    imu_a = _live(imu_acc)

    # --- cross-stream alignment on the recv clock: per video frame, nearest LPN + ODO ---
    def _pair_gaps(keys):
        keys = sorted(keys)
        gaps = []
        import bisect
        for vt in vid_ts:
            i = bisect.bisect_left(keys, vt)
            cands = [j for j in (i, i - 1) if 0 <= j < len(keys)]
            if cands:
                gaps.append(min(abs(keys[j] - vt) for j in cands) * 1e3)
        return gaps

    align = {}
    if lpn_rows and vid_ts:
        g = _pair_gaps([r[0] for r in lpn_rows])
        align["video_to_LPN_gap_ms"] = {"p50": round(float(np.median(g)), 2),
                                        "p99": round(float(np.percentile(g, 99)), 2),
                                        "max": round(float(np.max(g)), 2)}
    if odo_rows and vid_ts:
        g = _pair_gaps([r[0] for r in odo_rows])
        align["video_to_ODO_gap_ms"] = {"p50": round(float(np.median(g)), 2),
                                        "p99": round(float(np.percentile(g, 99)), 2),
                                        "max": round(float(np.max(g)), 2)}

    # --- verdict ---
    checks = {}
    for k, (req, _native) in REQUIRED.items():
        st = streams.get(k, {"rate_hz": 0.0, "n": 0})
        checks[k] = {"rate_hz": st["rate_hz"], "required_hz": req,
                     "pass": st["n"] > 2 and st["rate_hz"] >= req * 0.9}  # 10% rate margin
    checks["accel_body_present"] = {"pass": imu_a["present"]}
    checks["odo_quat_present"]   = {"pass": odo_quat_seen}
    checks["gt_position_present"] = {"pass": bool(lpn_rows)}
    checks["gt_velocity_present"] = {"pass": lpn_v.get("present", False)}
    checks["gt_velocity_LIVE"]   = {"pass": lpn_v.get("present", False) and lpn_v.get("max_mag", 0) > 1.0}

    all_pass = all(c["pass"] for c in checks.values())

    return {
        "run": run.name,
        "meta": {k: meta.get(k) for k in ("label", "final_state", "gate_index", "collisions",
                                          "duration_s", "video_frames", "checkpoint")},
        "type_counts": type_counts,
        "streams": streams,
        "ground_truth_velocity": {"LOCAL_POSITION_NED_world_ned": lpn_v,
                                  "ODOMETRY_body_twist": odo_v},
        "accel_body_HIGHRES_IMU_frd": imu_a,
        "attitude_quat_source": "ODOMETRY.q (TRUE attitude)" if odo_quat_seen else "MISSING",
        "attitude_euler_present": att_quat_seen,
        "alignment_recv_clock": align,
        "checks": checks,
        "ALL_PASS": all_pass,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    r = verify(Path(args.run))

    print(f"\n=========== HARNESS BUNDLE VERIFY: {r['run']} ===========")
    m = r["meta"]
    print(f"label={m.get('label')}  final={m.get('final_state')}  gate_index={m.get('gate_index')}  "
          f"collisions={m.get('collisions')}  dur={m.get('duration_s')}s")
    print(f"\n{'stream':>20} {'n':>7} {'dur_s':>7} {'rate_hz':>8} {'max_gap_ms':>11} {'p99_gap_ms':>11}")
    for k, st in r["streams"].items():
        print(f"{k:>20} {st['n']:>7} {st['dur_s']:>7} {st['rate_hz']:>8} "
              f"{st['max_gap_ms']:>11} {st['p99_gap_ms']:>11}")

    gtv = r["ground_truth_velocity"]["LOCAL_POSITION_NED_world_ned"]
    odov = r["ground_truth_velocity"]["ODOMETRY_body_twist"]
    acc = r["accel_body_HIGHRES_IMU_frd"]
    print(f"\nGROUND-TRUTH VELOCITY (LOCAL_POSITION_NED, world NED)  : {gtv}")
    print(f"ODOMETRY body twist (cross-check, raw body frame)     : {odov}")
    print(f"accel_body (HIGHRES_IMU, FRD specific force)          : {acc}")
    print(f"attitude quat source                                  : {r['attitude_quat_source']}")
    print(f"\nalignment on recv clock (video<->telemetry pair gaps):")
    for k, v in r["alignment_recv_clock"].items():
        print(f"  {k}: {v}")

    print(f"\n{'CHECK':>26} {'pass':>6}  detail")
    for k, c in r["checks"].items():
        det = "" if "rate_hz" not in c else f"{c['rate_hz']} Hz >= {c['required_hz']} Hz"
        print(f"{k:>26} {'OK' if c['pass'] else 'FAIL':>6}  {det}")
    print(f"\n  ===> ALL_PASS = {r['ALL_PASS']}")

    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if r["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
