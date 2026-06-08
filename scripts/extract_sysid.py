"""Offline: extract a downsampled sysid JSON from a recorded run (tlog + commands.jsonl).

Produces the data the laptop needs to FIT the CTBR plant twin (rate_gain, rate_sign, rate_tau,
thrust hover/slope) and validate it -- without the sim. One sample per ~rate Hz, joined from the
raw MAVLink telemetry and our logged commands.

Join design (validated in handoff/shadowpc-followups-2026-06-05/scratch/clock_*.py):
  * Backbone = ODOMETRY (75 Hz; carries the orientation quaternion, body rates, body twist, world
    pos, and a sim clock ``time_usec``). Downsampled to ~``rate`` Hz on the sim clock.
  * HIGHRES_IMU / ODOMETRY / ACTUATOR_OUTPUT_STATUS share ONE sim epoch (``time_usec``), and
    commands.jsonl's ``sim_time_ns`` lives on it -> commands joined by ZERO-ORDER HOLD on the sim
    clock (the command in effect at each sample), actuators by nearest ``time_usec``.
  * LOCAL_POSITION_NED has NO ``time_usec`` (only a different-epoch ``time_boot_ms``), so it is
    paired to the backbone by nearest RECEIVE time (tlog prefix; both ~ms apart) -- no clock map.
  * The sim clock is frozen during the pre-race countdown then jumps once to the running-race clock
    (physics pauses off-race); restricting the emit window to the command span drops that region.

Velocity frame: the raw ODOMETRY twist (vx/vy/vz) is BODY FRD; the c3b5a8e fix rotates it to world
via ``frames.world_vec_from_body_quat``. We emit BOTH ``odo_vel_body`` (raw) and ``odo_vel_world``
(rotated = the FIXED world velocity), plus ``lpn_vel`` (LOCAL_POSITION_NED, independent world) as a
per-run cross-check that the fix holds (validation.rot_odo_vs_lpn_vel_rms_mps).

Usage:
  python scripts/extract_sysid.py <run_dir> -o <out.json> [--rate 50] [--commit <sha>]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.frames import world_vec_from_body_quat
from racer.recording import RecordingReader


def _round(v, n=6):
    if v is None:
        return None
    return [round(float(x), n) for x in np.asarray(v, dtype=np.float64).ravel()]


def _load_commands(run: Path) -> tuple[list[dict], str | None]:
    """Return (sorted command rows, schema name|None). Each row -> {sim_ns, body_rate, thrust, ctx}."""
    cj = run / "commands.jsonl"
    if not cj.exists():
        return [], None
    out = []
    schema = None
    for line in cj.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "cmd" in r:                       # rate_sysid schema
            schema = "rate_sysid"
            sim_ns = int(r.get("sim_time_ns", 0) or 0)
            br = r.get("cmd")
            ctx = {k: r[k] for k in ("phase", "kind", "axis") if k in r}
        elif "body_rate" in r:               # fly_vq1 --cmd-log schema
            schema = "fly_vq1"
            sim_ns = int(r.get("sim_t", 0) or 0)
            br = r.get("body_rate")
            ctx = {k: r[k] for k in ("state",) if k in r}
        else:
            continue
        if sim_ns <= 0:
            continue
        out.append({"sim_ns": sim_ns, "body_rate": br,
                    "thrust": r.get("thrust"), "ctx": ctx})
    out.sort(key=lambda d: d["sim_ns"])
    return out, schema


def _nearest(sorted_keys: list[float], items: list, key: float):
    """Nearest item to ``key`` in a list sorted by ``sorted_keys``; returns (item, gap)."""
    if not sorted_keys:
        return None, None
    i = bisect.bisect_left(sorted_keys, key)
    cands = []
    if i < len(sorted_keys):
        cands.append(i)
    if i > 0:
        cands.append(i - 1)
    best = min(cands, key=lambda j: abs(sorted_keys[j] - key))
    return items[best], abs(sorted_keys[best] - key)


def extract(run: Path, rate_hz: float, commit: str | None) -> dict:
    reader = RecordingReader(run)
    meta = reader.meta

    odo, lpn, act = [], [], []
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        recv = float(getattr(msg, "_timestamp", 0.0))     # tlog prefix: recv unix seconds
        if t == "ODOMETRY":
            odo.append({
                "sim_us": int(msg.time_usec), "recv": recv,
                "q": [float(v) for v in msg.q],            # w,x,y,z
                "rate": [float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)],
                "vb": [float(msg.vx), float(msg.vy), float(msg.vz)],   # BODY FRD twist
                "pos": [float(msg.x), float(msg.y), float(msg.z)],     # world NED
            })
        elif t == "LOCAL_POSITION_NED":
            lpn.append({
                "recv": recv,
                "pos": [float(msg.x), float(msg.y), float(msg.z)],
                "vel": [float(msg.vx), float(msg.vy), float(msg.vz)],  # world NED
            })
        elif t == "ACTUATOR_OUTPUT_STATUS":
            act.append({
                "sim_us": int(getattr(msg, "time_usec", 0)),
                "motors": [float(x) for x in list(msg.actuator)[:4]],
            })
    odo.sort(key=lambda d: d["sim_us"])
    lpn.sort(key=lambda d: d["recv"])
    act.sort(key=lambda d: d["sim_us"])

    cmds, cmd_schema = _load_commands(run)

    # Emit window = command span (active flight) intersected with telemetry; pad 0.1 s before.
    if cmds:
        t0_ns = cmds[0]["sim_ns"] - 100_000_000
        t1_ns = cmds[-1]["sim_ns"] + 20_000_000
    else:                                   # no commands (e.g. firstcontact) -> whole odo span
        t0_ns = odo[0]["sim_us"] * 1000 if odo else 0
        t1_ns = odo[-1]["sim_us"] * 1000 if odo else 0

    lpn_recv = [d["recv"] for d in lpn]
    act_us = [d["sim_us"] for d in act]
    cmd_ns = [d["sim_ns"] for d in cmds]

    step_us = 1e6 / rate_hz
    samples = []
    next_emit_us = None                     # accumulator -> true ~rate Hz from 75 Hz ODOMETRY
    rot_resid = []                          # rotated-ODO vs LPN world-velocity (fix cross-check)
    lpn_gaps_ms, act_gaps_ms = [], []
    for o in odo:
        sus = o["sim_us"]
        if sus * 1000 < t0_ns or sus * 1000 > t1_ns:
            continue
        if next_emit_us is None:
            next_emit_us = sus + step_us
        elif sus >= next_emit_us:
            while next_emit_us <= sus:       # catch up if we fell behind a tick
                next_emit_us += step_us
        else:
            continue
        q = np.array(o["q"], dtype=np.float64)
        vb = np.array(o["vb"], dtype=np.float64)
        vw = world_vec_from_body_quat(vb, q)             # the c3b5a8e fix
        # nearest LPN (by recv) + nearest ACTUATOR (by sim clock)
        lpn_i, lpn_gap = _nearest(lpn_recv, lpn, o["recv"])
        act_i, act_gap = _nearest(act_us, act, sus)
        if lpn_gap is not None:
            lpn_gaps_ms.append(lpn_gap * 1000.0)
        if act_gap is not None:
            act_gaps_ms.append(act_gap / 1000.0)
        if lpn_i is not None:
            rot_resid.append(float(np.linalg.norm(vw - np.array(lpn_i["vel"]))))
        # ZOH command: latest with sim_ns <= sample
        ci = bisect.bisect_right(cmd_ns, sus * 1000) - 1
        cmd = cmds[ci] if ci >= 0 else None
        samples.append({
            "t_ns": sus * 1000,
            "t_s": round((sus * 1000 - (odo[0]["sim_us"] * 1000)) / 1e9, 4),
            "odo_q_wxyz": _round(q),
            "odo_angular_rate": _round(o["rate"]),         # BODY FRD, RAW (sign-inverted on pitch)
            "odo_vel_body": _round(vb),                    # BODY FRD, raw twist
            "odo_vel_world": _round(vw),                   # WORLD NED -- FIXED (rotated) velocity
            "odo_pos": _round(o["pos"]),                   # WORLD NED
            "lpn_pos": None if lpn_i is None else _round(lpn_i["pos"]),
            "lpn_vel": None if lpn_i is None else _round(lpn_i["vel"]),   # WORLD NED (independent)
            "actuators": None if act_i is None else _round(act_i["motors"], 5),
            "cmd_body_rate": None if cmd is None else _round(cmd["body_rate"]),
            "cmd_thrust": None if (cmd is None or cmd["thrust"] is None) else round(float(cmd["thrust"]), 5),
            "cmd_age_ms": None if cmd is None else round((sus * 1000 - cmd["sim_ns"]) / 1e6, 2),
            "ctx": None if cmd is None else (cmd["ctx"] or None),
            "lpn_gap_ms": None if lpn_gap is None else round(lpn_gap * 1000.0, 2),
        })

    val = {
        "n_samples": len(samples),
        "window_s": round((t1_ns - t0_ns) / 1e9, 3),
        "rot_odo_vs_lpn_vel_rms_mps": round(float(np.sqrt(np.mean(np.square(rot_resid)))), 4) if rot_resid else None,
        "rot_odo_vs_lpn_vel_max_mps": round(float(np.max(rot_resid)), 4) if rot_resid else None,
        "lpn_pair_gap_ms_med": round(float(np.median(lpn_gaps_ms)), 2) if lpn_gaps_ms else None,
        "lpn_pair_gap_ms_max": round(float(np.max(lpn_gaps_ms)), 2) if lpn_gaps_ms else None,
        "act_pair_gap_ms_max": round(float(np.max(act_gaps_ms)), 2) if act_gaps_ms else None,
    }
    return {
        "schema": "racer.sysid_extract/v1",
        "run": run.name,
        "source_commit": commit,
        "rate_hz": rate_hz,
        "command_schema": cmd_schema,
        "fields_present": {
            "odo": bool(odo), "lpn": bool(lpn), "actuators": bool(act),
            "commands": bool(cmds),
        },
        "frames": {
            "odo_q_wxyz": "body(FRD)->world(NED), MAVLink scalar-first (w,x,y,z)",
            "odo_angular_rate": "BODY FRD rad/s, RAW ODOMETRY (sign-inverted on pitch per sysid; "
                                "prefer quaternion finite-difference for a sign-correct rate)",
            "odo_vel_body": "BODY FRD m/s, raw ODOMETRY twist (child_frame_id=MAV_FRAME_BODY_NED=8)",
            "odo_vel_world": "WORLD NED m/s = world_vec_from_body_quat(odo_vel_body, q) -- the "
                             "c3b5a8e FIX (rotated). Use THIS as the corrected velocity.",
            "odo_pos": "WORLD NED m (ODOMETRY pose)",
            "lpn_pos": "WORLD NED m (LOCAL_POSITION_NED, independent)",
            "lpn_vel": "WORLD NED m/s (LOCAL_POSITION_NED, independent, pristine) -- cross-check for odo_vel_world",
            "actuators": "4 motor outputs [0..1] (ACTUATOR_OUTPUT_STATUS, parser-independent witness)",
            "cmd_body_rate": "OUR commanded body rate [roll,pitch,yaw] rad/s (wire SET_ATTITUDE_TARGET), ZOH",
            "cmd_thrust": "OUR commanded collective [0..1], ZOH",
        },
        "clock": "t_ns = sim clock (time_usec*1000), shared by ODOMETRY/ACTUATOR/commands; LPN paired by recv",
        "meta": {k: meta.get(k) for k in (
            "label", "mode", "probe", "thrust", "mag", "axes", "aborted", "final_state",
            "yaw_hold_deg", "duration_s", "created_utc") if k in meta},
        "validation": val,
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--commit", default=None)
    args = ap.parse_args()
    data = extract(Path(args.run), args.rate, args.commit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    v = data["validation"]
    print(f"{data['run']:>34}  n={v['n_samples']:<4} win={v['window_s']}s  "
          f"cmd={data['command_schema']}  rot-vs-lpn RMS={v['rot_odo_vs_lpn_vel_rms_mps']} "
          f"max={v['rot_odo_vs_lpn_vel_max_mps']}  lpn_gap_med={v['lpn_pair_gap_ms_med']}ms  "
          f"-> {out}  ({out.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
