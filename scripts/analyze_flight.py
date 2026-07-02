"""analyze_flight.py — the standard post-flight read: a clock-aligned vertical-channel story.

Ingests EVERYTHING a run directory records and aligns it on the sim master clock
(HIGHRES_IMU.time_usec, the same clock nav_estimate.jsonl / video_index.jsonl stamp):

  mavlink.tlog        RECEIVED telemetry: HIGHRES_IMU (accel+gyro), ACTUATOR_OUTPUT_STATUS
                      (realized motor outputs), COLLISION, RACE_STATUS.
  commands.jsonl      OUTGOING SET_ATTITUDE_TARGET mirror (commanded body rates + collective)
                      — absent on runs that predate command mirroring (falls back to the
                      derived thrust in nav_estimate.jsonl).
  nav_estimate.jsonl  the estimator's per-tick view (attitude / position / time_since_vision).
  video_index.jsonl   per-frame sim_time_ns -> vision fps over the flight window.

ACCEL DECODE (verified empirically, 2026-07-01 — docs/accel_investigation.md): HIGHRES_IMU
accel is SPECIFIC FORCE in body FRD (|f| = 9.810 exactly at rest, ~0 in ballistic arcs), so
the kinematic vertical acceleration is   a_up = -( [R_body->ned @ f]_z + g ).

Usage:
  .venv\\Scripts\\python.exe scripts\\analyze_flight.py data\\runs\\<run_dir> [--rows 40] [--full]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")

import numpy as np

G = 9.80665

# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def load_tlog(path: Path) -> dict:
    """Parse mavlink.tlog -> imu/actuator/collision/race arrays on the sim clock."""
    from pymavlink import mavutil

    conn = mavutil.mavlink_connection(str(path))
    imu_t, acc, gyr = [], [], []
    act_t, motors = [], []
    collisions = []          # (sim_time_s_of_last_imu, id, threat_level)
    race = None
    types: dict[str, int] = {}
    while True:
        msg = conn.recv_match(blocking=False)
        if msg is None:
            break
        t = msg.get_type()
        types[t] = types.get(t, 0) + 1
        if t == "HIGHRES_IMU":
            imu_t.append(msg.time_usec)
            acc.append((msg.xacc, msg.yacc, msg.zacc))
            gyr.append((getattr(msg, "xgyro", 0.0), getattr(msg, "ygyro", 0.0),
                        getattr(msg, "zgyro", 0.0)))
        elif t == "ACTUATOR_OUTPUT_STATUS":
            act_t.append(msg.time_usec)
            motors.append(tuple(float(x) for x in list(msg.actuator)[:4]))
        elif t == "COLLISION":
            collisions.append({
                "sim_time_s": (imu_t[-1] / 1e6) if imu_t else None,
                "id": int(msg.id),
                "threat_level": int(getattr(msg, "threat_level", 0)),
            })
        elif t == "ENCAPSULATED_DATA":
            try:
                from racer.mavlink_client import parse_race_status
                raw = bytes(msg.data)
                if raw and raw[0] == 1:
                    rs = parse_race_status(raw)
                    if rs is not None:
                        race = rs
            except Exception:
                pass
    conn.close()
    return {
        "types": types,
        "imu_t": np.asarray(imu_t, dtype=np.int64),
        "acc": np.asarray(acc, dtype=np.float64).reshape(-1, 3),
        "gyr": np.asarray(gyr, dtype=np.float64).reshape(-1, 3),
        "act_t": np.asarray(act_t, dtype=np.int64),
        "motors": np.asarray(motors, dtype=np.float64).reshape(-1, 4),
        "collisions": collisions,
        "race": race,
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------

def rot_b2n(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body FRD -> world NED rotation from ZYX (yaw-pitch-roll) Euler angles —
    the navigator's convention (racer.frames.euler_from_quat_wxyz inverse)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def a_up_from_specific_force(f_body: np.ndarray, roll: float, pitch: float, yaw: float) -> float:
    """Kinematic UPWARD acceleration (m/s^2) from a specific-force sample + attitude.
    a_ned = R@f + g_ned  (g_ned = (0,0,+G), NED z is DOWN)  ->  a_up = -a_ned_z."""
    a_ned_z = float(rot_b2n(roll, pitch, yaw)[2] @ f_body) + G
    return -a_ned_z


def nearest_at_or_before(times: np.ndarray, t: float) -> int:
    """Index of the last element of ``times`` <= t (clipped to 0)."""
    i = int(np.searchsorted(times, t, side="right")) - 1
    return max(i, 0)


# ---------------------------------------------------------------------------
# the story
# ---------------------------------------------------------------------------

def analyze(run_dir: Path, max_rows: int, full: bool) -> int:
    tlog_path = run_dir / "mavlink.tlog"
    if not tlog_path.exists():
        print(f"ERROR: {tlog_path} not found", file=sys.stderr)
        return 2
    meta = {}
    if (run_dir / "meta.json").exists():
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    tl = load_tlog(tlog_path)
    nav = load_jsonl(run_dir / "nav_estimate.jsonl")
    cmds = load_jsonl(run_dir / "commands.jsonl")
    vidx = load_jsonl(run_dir / "video_index.jsonl")

    imu_t, acc, gyr = tl["imu_t"], tl["acc"], tl["gyr"]
    act_t, motors = tl["act_t"], tl["motors"]
    if len(imu_t) == 0:
        print("ERROR: no HIGHRES_IMU in tlog", file=sys.stderr)
        return 2
    imu_s = imu_t / 1e6

    print(f"==== analyze_flight: {run_dir} ====")
    print(f"meta: label={meta.get('label')} final_state={meta.get('final_state')} "
          f"rate_hz={meta.get('rate_hz')} duration_s={meta.get('duration_s', 0):.1f} "
          f"cmd_rate_scale={meta.get('cmd_rate_scale')}")
    print(f"tlog msg counts: { {k: v for k, v in sorted(tl['types'].items(), key=lambda kv: -kv[1])} }")

    # --- flight window: motor spin-up -> collision / clock-freeze / end ---------------
    spin_i = np.where(motors.max(axis=1) > 1e-6)[0] if len(motors) else np.array([], int)
    t_spin = float(act_t[spin_i[0]] / 1e6) if len(spin_i) else float(imu_s[0])
    # liftoff = first accel CHANGE after spin-up (on-pad physics is static -> fields constant)
    t_lift = None
    after = np.where(imu_s > t_spin)[0]
    for j in after[1:]:
        if not np.array_equal(acc[j], acc[j - 1]):
            t_lift = float(imu_s[j])
            break
    # sim clock-freeze tail (the sim halts its physics clock at race end / crash and
    # re-sends the last state -> identical time_usec at the stream tail)
    t_end = float(imu_s[-1])
    frozen_tail = 0
    k = len(imu_t) - 1
    while k > 0 and imu_t[k] == imu_t[k - 1]:
        frozen_tail += 1
        k -= 1
    colls = tl["collisions"]
    t_coll = colls[0]["sim_time_s"] if colls else None

    print(f"\n-- flight window (sim clock) --")
    print(f"motor spin-up t={t_spin:.3f}s   liftoff(first accel change) "
          f"t={t_lift:.3f}s" if t_lift else f"motor spin-up t={t_spin:.3f}s   liftoff: not detected")
    if t_coll is not None:
        for c in colls:
            print(f"COLLISION at t={c['sim_time_s']:.3f}s  id={c['id']} "
                  f"({'environment' if c['id'] == 1002 else 'gate'}) threat={c['threat_level']}")
    else:
        print("COLLISION: none")
    if frozen_tail:
        print(f"sim clock FROZE at t={t_end:.3f}s ({frozen_tail} trailing HIGHRES_IMU re-sends "
              f"of the last state — race-end/crash halt, NOT a receive bug)")

    # --- IMU health over the LIVE window (spin -> freeze/end) --------------------------
    w = (imu_s >= t_spin) & (imu_s <= t_end)
    wi = np.where(w)[0]
    if len(wi) > 2:
        live = wi[:-frozen_tail] if frozen_tail else wi
        dts = np.diff(imu_t[live]) / 1e3  # ms
        dts = dts[dts > 0]
        rep = sum(1 for a, b in zip(live[:-1], live[1:])
                  if np.array_equal(acc[a], acc[b]) and np.array_equal(gyr[a], gyr[b]))
        print(f"\n-- HIGHRES_IMU health (live window {t_spin:.2f}..{t_end:.2f}s) --")
        print(f"n={len(live)}  median dt={np.median(dts):.1f}ms  p95={np.percentile(dts, 95):.1f}ms "
              f"max gap={dts.max():.1f}ms  identical acc+gyro repeats={rep} "
              f"({100 * rep / max(len(live) - 1, 1):.1f}%; on-pad static physics counts here)")
    # accel convention check: pre-spin |f|
    pre = imu_s < t_spin
    if pre.sum() > 3:
        m = float(np.linalg.norm(acc[pre], axis=1).mean())
        f0 = acc[pre].mean(axis=0)
        tilt = np.degrees(np.arcsin(np.clip(f0[0] / max(m, 1e-9), -1, 1)))
        print(f"accel convention: on-pad |f|={m:.3f} m/s^2 (== g -> SPECIFIC FORCE, gravity-included); "
              f"on-pad f=({f0[0]:+.2f},{f0[1]:+.2f},{f0[2]:+.2f}) -> resting-tilt pitch~{tilt:+.1f} deg")

    # --- per-tick vertical-channel story ------------------------------------------------
    # master rows = nav_estimate ticks when present (they carry attitude + estimator z);
    # otherwise synthesize rows from the IMU at ~30 Hz with zero attitude (degraded mode).
    rows = []
    if nav:
        for r in nav:
            st = r.get("sim_time_ns")
            if st is None:
                continue
            rows.append({
                "t": st / 1e9,
                "roll": r.get("roll_rad") or 0.0,
                "pitch": r.get("pitch_rad") or 0.0,
                "yaw": r.get("yaw_rad") or 0.0,
                "thrust_cmd": r.get("thrust"),
                "est_z": (r.get("position_ned") or [None, None, None])[2],
                "tsv": r.get("time_since_vision_s"),
                "tick": r.get("tick_index"),
            })
    else:
        print("\nNOTE: no nav_estimate.jsonl -> degraded story (attitude assumed level; "
              "a_up is then only valid while the drone is near-level).")
        step = max(int(round((1.0 / 30.0) / max(np.median(np.diff(imu_s[imu_s > t_spin])), 1e-3))), 1)
        for j in wi[::step]:
            rows.append({"t": float(imu_s[j]), "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                         "thrust_cmd": None, "est_z": None, "tsv": None, "tick": None})

    # attach commands.jsonl (preferred commanded-thrust source: the RAW uplink mirror)
    cmd_t = np.asarray([c["sim_time_ns"] / 1e9 for c in cmds if c.get("sim_time_ns")],
                       dtype=np.float64) if cmds else np.array([])
    cmd_thr = np.asarray([c.get("thrust") if c.get("thrust") is not None else np.nan
                          for c in cmds if c.get("sim_time_ns")], dtype=np.float64) if cmds else np.array([])

    # decode each row
    vz = 0.0
    prev_t = None
    for r in rows:
        t = r["t"]
        i = nearest_at_or_before(imu_s, t)
        f = acc[i]
        r["f_z"] = float(f[2])
        r["a_up"] = a_up_from_specific_force(f, r["roll"], r["pitch"], r["yaw"])
        r["gyro_pitch"] = float(gyr[i][1])
        if len(act_t):
            r["motors"] = float(motors[nearest_at_or_before(act_t / 1e6, t)].mean())
        else:
            r["motors"] = None
        if len(cmd_t):
            j = nearest_at_or_before(cmd_t, t)
            r["thrust_wire"] = None if np.isnan(cmd_thr[j]) else float(cmd_thr[j])
        else:
            r["thrust_wire"] = None
        # crude integrated vertical velocity (upward +) between rows — the vertical-velocity
        # estimator candidate signal; drifts without a floor_height correction, shown raw here.
        if prev_t is not None and 0 < t - prev_t < 1.0:
            vz += r["a_up"] * (t - prev_t)
        r["vz_int"] = vz
        prev_t = t

    if rows:
        t0 = rows[0]["t"]
        stride = 1 if full else max(len(rows) // max_rows, 1)
        print(f"\n-- vertical channel (aligned on the sim clock; t rel. first tick @ {t0:.3f}s) --")
        print(f"{'t(s)':>7} {'thr_cmd':>7} {'thr_wire':>8} {'motors':>6} {'f_z':>8} {'a_up':>7} "
              f"{'vz_int':>7} {'est_z':>7} {'alt':>6} {'pitch':>6} {'tsv':>5}")
        printed_coll = False
        for k, r in enumerate(rows):
            is_coll = (t_coll is not None and not printed_coll and r["t"] >= t_coll)
            if k % stride and not is_coll and k != len(rows) - 1:
                continue
            if is_coll:
                printed_coll = True
            alt = -r["est_z"] if r["est_z"] is not None else None
            print(f"{r['t'] - t0:7.2f} "
                  f"{r['thrust_cmd'] if r['thrust_cmd'] is not None else float('nan'):7.3f} "
                  f"{r['thrust_wire'] if r['thrust_wire'] is not None else float('nan'):8.3f} "
                  f"{r['motors'] if r['motors'] is not None else float('nan'):6.3f} "
                  f"{r['f_z']:8.2f} {r['a_up']:7.2f} {r['vz_int']:7.2f} "
                  f"{r['est_z'] if r['est_z'] is not None else float('nan'):7.2f} "
                  f"{alt if alt is not None else float('nan'):6.2f} "
                  f"{np.degrees(r['pitch']):6.1f} "
                  f"{r['tsv'] if r['tsv'] is not None else float('nan'):5.2f}"
                  + ("  <-- COLLISION" if is_coll else ""))

    # --- summary -----------------------------------------------------------------------
    print("\n-- summary --")
    if len(rows) > 1:
        span = rows[-1]["t"] - rows[0]["t"]
        # exclude frozen-clock duplicate ticks from the loop-rate estimate
        uniq_t = len({round(r["t"], 6) for r in rows})
        print(f"loop: {len(rows)} ticks over {span:.2f}s sim -> {uniq_t / max(span, 1e-9):5.1f} Hz "
              f"(unique-tick rate; {len(rows) - uniq_t} frozen-clock duplicate ticks)")
    if vidx:
        vt = np.asarray([v["sim_time_ns"] / 1e9 for v in vidx])
        vw = vt[(vt >= t_spin) & (vt <= t_end)]
        if len(vw) > 1:
            print(f"vision: {len(vw)} frames in window -> {(len(vw) - 1) / max(vw[-1] - vw[0], 1e-9):.1f} fps")
    tsvs = [r["tsv"] for r in rows if r.get("tsv") is not None]
    if tsvs:
        print(f"obs age (time_since_vision): mean={np.mean(tsvs):.2f}s max={np.max(tsvs):.2f}s "
              f"({len(tsvs)}/{len(rows)} ticks with a vision fix)")
    else:
        print("obs age: NO tick ever had a vision fix (time_since_vision null throughout)")
    thr = [r["thrust_cmd"] for r in rows if r.get("thrust_cmd") is not None]
    if thr:
        print(f"thrust cmd: min={min(thr):.3f} mean={np.mean(thr):.3f} max={max(thr):.3f} "
              f"(hover=0.2656)")
    if cmds:
        n_br = sum(1 for c in cmds if c.get("mode") == "BODY_RATE")
        print(f"commands.jsonl: {len(cmds)} outgoing commands ({n_br} BODY_RATE) — raw uplink logged")
    else:
        print("commands.jsonl: ABSENT (run predates outgoing-command mirroring)")
    if t_coll is not None:
        print(f"collision: YES at sim t={t_coll:.3f}s "
              f"({t_coll - (t_lift if t_lift else t_spin):+.2f}s after "
              f"{'liftoff' if t_lift else 'spin-up'})")
    else:
        print("collision: none")
    fl = (imu_s >= (t_lift if t_lift else t_spin)) & (imu_s <= (t_coll if t_coll else t_end))
    if fl.sum() > 3:
        m = np.linalg.norm(acc[fl], axis=1)
        print(f"in-flight |specific force|: p5={np.percentile(m, 5):.2f} p50={np.percentile(m, 50):.2f} "
              f"p95={np.percentile(m, 95):.2f} m/s^2 (hover==g={G:.2f}; <g == descending/ballistic)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="data/runs/<stamp>_<label> session directory")
    ap.add_argument("--rows", type=int, default=40,
                    help="target number of table rows (subsampled; default 40)")
    ap.add_argument("--full", action="store_true", help="print EVERY tick (no subsampling)")
    args = ap.parse_args()
    return analyze(args.run_dir, max_rows=args.rows, full=args.full)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    raise SystemExit(main())
