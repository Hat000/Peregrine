"""VQ2 control-handshake probe: test CTBR (collective-thrust + body-rate) hover/maneuver.

Reuses the CANONICAL uplink (`racer.mavlink_client.MavlinkClient` -> SET_ATTITUDE_TARGET) so
whatever flies here transfers directly to the deploy loop (rl/fly_rl.py). No hand-rolled
MAVLink.

Regimes (the recon flagged that a GCS heartbeat appears to force ACRO, while the reference
client's TIMESYNC-only keepalive leaves the sim in ANGLE):
  * --regime acro     : send our 2Hz GCS heartbeat (forces ACRO) + CTBR body-rate setpoints.
  * --regime angle    : TIMESYNC-only (no heartbeat) + ATTITUDE (level quat) + thrust.
  * --regime timesync_ctbr : TIMESYNC-only + CTBR (controls whether CTBR needs the heartbeat).

Tests (per regime):
  * hover  : body rates = 0 (or level quat), thrust held at --thrust for --hold s.
  * step   : after a hover settle, add a body-rate step (+pitch by default) for --step-s.

Objective verdicts (no GUI needed):
  * COLLISION events during the active window (env=1002 / gate=1001)  -> any = NOT clean.
  * ACTUATOR_OUTPUT_STATUS: 4 motor outputs -> mean + spread (balanced ~= small spread).
  * HIGHRES_IMU accel magnitude: ~9.8 and steady = gravity-dominated (not thrashing).
  * gyro magnitude: small in hover = not tumbling.
  * RACE_STATUS active_gate_index + any LOCAL_POSITION_NED / ODOMETRY z drift (lift).

Writes a JSON timeseries + a verdict summary; prints the summary.

Usage (from repo root, .venv active):
  python handoff/vq2-control-handshake-2026-06-29/ctbr_probe.py --regime acro \
      --thrust 0.27 --hold 4 --step pitch --step-rate 0.3 --step-s 0.5 --out logs/acro_t027.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")

import numpy as np

# Find repo src/ (worktree root is parents[2]); fall back to the main clone.
for _cand in (Path(__file__).resolve().parents[2] / "src", Path("C:/Users/Shadow/Peregrine/src")):
    if (_cand / "racer" / "mavlink_client.py").exists():
        sys.path.insert(0, str(_cand))
        break

from racer.contracts import ControlCommand, ControlMode  # noqa: E402
from racer.mavlink_client import MavlinkClient  # noqa: E402

_STEP_AXIS = {"roll": 0, "pitch": 1, "yaw": 2}


def _accel_mag(state) -> float:
    a = state.accel_body
    return float(np.linalg.norm(a)) if a is not None else float("nan")


def _gyro_mag(state) -> float:
    g = state.gyro_body
    return float(np.linalg.norm(g)) if g is not None else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="acro",
                    choices=["acro", "angle", "timesync_ctbr"])
    ap.add_argument("--thrust", type=float, default=0.27,
                    help="normalized collective [0,1] held during hover")
    ap.add_argument("--rate-hz", type=float, default=100.0, help="setpoint stream rate")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds before hover window")
    ap.add_argument("--hold", type=float, default=4.0, help="hover hold seconds")
    ap.add_argument("--step", default="none", choices=["none", "roll", "pitch", "yaw"])
    ap.add_argument("--step-rate", type=float, default=0.3, help="body-rate step (rad/s)")
    ap.add_argument("--step-s", type=float, default=0.5, help="step duration seconds")
    ap.add_argument("--force-arm", action="store_true", help="use the 21196 arm bypass")
    ap.add_argument("--endpoint", default="udpin:127.0.0.1:14550")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outp = Path(args.out)
    if not outp.is_absolute():
        outp = Path(__file__).resolve().parent / outp
    outp.parent.mkdir(parents=True, exist_ok=True)

    client = MavlinkClient(args.endpoint)
    # Keepalive regime ---------------------------------------------------------
    if args.regime == "acro":
        client.send_heartbeats = True       # heartbeat -> ACRO (the hypothesis)
        client.send_timesync = False
    else:                                    # angle / timesync_ctbr
        client.send_heartbeats = False      # stay quiet -> sim keeps ANGLE
        client.send_timesync = True

    print(f"[probe] connecting {args.endpoint} regime={args.regime} ...")
    client.connect(wait_heartbeat=True, timeout_s=15.0)
    # let the link + target settle, learn telemetry
    t_warm = time.monotonic()
    while time.monotonic() - t_warm < 1.5:
        client.pump()
        time.sleep(0.01)
    print(f"[probe] linked. armed={client.state.armed} base_mode set; "
          f"autopilot={client.autopilot} type={client.vehicle_type} custom_mode={client.custom_mode}")

    # ARM ----------------------------------------------------------------------
    client.last_command_ack = None
    client.arm(force=args.force_arm)
    armed = client.wait_armed(True, timeout_s=5.0)
    ack = client.wait_command_ack(400, timeout_s=2.0)
    print(f"[probe] arm -> armed={armed} ack={ack}")
    if not armed:
        print("[probe] ARM FAILED, aborting", file=sys.stderr)
        # still dump what we have
        outp.write_text(json.dumps({"regime": args.regime, "armed": False, "ack": ack}, indent=2))
        return

    # Build the setpoint --------------------------------------------------------
    use_ctbr = args.regime in ("acro", "timesync_ctbr")
    level_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity = level

    def make_cmd(rate_vec: np.ndarray) -> ControlCommand:
        if use_ctbr:
            return ControlCommand(mode=ControlMode.BODY_RATE,
                                  body_rate=rate_vec, thrust=float(args.thrust))
        return ControlCommand(mode=ControlMode.ATTITUDE,
                              attitude_quat_wxyz=level_q, thrust=float(args.thrust))

    dt = 1.0 / args.rate_hz
    n_collisions_before = len(client.collisions)
    samples: list[dict] = []
    phase_marks: dict[str, float] = {}

    total = args.settle + args.hold + (args.step_s if args.step != "none" else 0.0)
    t0 = time.monotonic()
    next_send = t0
    last_log = 0.0
    z0 = None  # first observed down-position, for lift measurement

    print(f"[probe] streaming {('CTBR' if use_ctbr else 'ATTITUDE')} thrust={args.thrust} "
          f"for {total:.1f}s (settle={args.settle} hold={args.hold} step={args.step})")
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= total:
            break
        # phase -> rate vector
        rate_vec = np.zeros(3, dtype=np.float64)
        if args.step != "none" and t >= (args.settle + args.hold):
            phase = "step"
            rate_vec[_STEP_AXIS[args.step]] = args.step_rate
        elif t >= args.settle:
            phase = "hover"
        else:
            phase = "settle"
        phase_marks.setdefault(phase, t)

        if now >= next_send:
            client.send_command(make_cmd(rate_vec))
            next_send += dt
        client.pump()

        # log at ~50 Hz
        if now - last_log >= 0.02:
            last_log = now
            s = client.state
            pz = None
            if s.position_ned is not None:
                pz = float(s.position_ned[2])
                if z0 is None:
                    z0 = pz
            act = client.actuator_outputs
            motors = None
            if act is not None:
                motors = [float(x) for x in act.get("motors", [])[:4]]
            samples.append({
                "t": round(t, 4), "phase": phase,
                "accel_mag": round(_accel_mag(s), 4),
                "gyro_mag": round(_gyro_mag(s), 5),
                "ar_body": [round(float(x), 4) for x in s.angular_rate_body],
                "pz": pz,
                "motors": motors,
                "race_gate": (client.race_status or {}).get("active_gate_index"),
                "n_coll": len(client.collisions),
            })
        time.sleep(0.001)

    # disarm to leave a clean state
    try:
        client.disarm()
    except Exception:
        pass

    new_coll = client.collisions[n_collisions_before:]

    # ---- verdict ----
    def _phase_rows(p):
        return [r for r in samples if r["phase"] == p]

    hover_rows = _phase_rows("hover")
    step_rows = _phase_rows("step")

    def _stat(rows, key, idx=None):
        vals = []
        for r in rows:
            v = r[key]
            if idx is not None and v is not None:
                v = v[idx] if len(v) > idx else None
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vals.append(v)
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "n": len(vals)}

    motor_means = None
    motor_spread = None
    if hover_rows and hover_rows[-1]["motors"] is not None:
        m = np.array([r["motors"] for r in hover_rows if r["motors"] is not None])
        if m.size:
            motor_means = [round(float(x), 4) for x in m.mean(axis=0)]
            # spread across the 4 motors, averaged over hover
            motor_spread = round(float(np.mean(m.max(axis=1) - m.min(axis=1))), 4)

    # lift: change in down-position over hover (negative = climbed)
    lift = None
    pzs = [r["pz"] for r in hover_rows if r["pz"] is not None]
    if len(pzs) >= 2:
        lift = round(pzs[0] - pzs[-1], 3)  # >0 means z decreased = climbed

    verdict = {
        "regime": args.regime,
        "thrust": args.thrust,
        "step": args.step,
        "armed": True,
        "arm_ack": ack,
        "n_samples": len(samples),
        "collisions_in_window": len(new_coll),
        "collisions": new_coll[:20],
        "hover_accel_mag": _stat(hover_rows, "accel_mag"),
        "hover_gyro_mag": _stat(hover_rows, "gyro_mag"),
        "hover_motor_means": motor_means,
        "hover_motor_spread": motor_spread,
        "hover_lift_m": lift,
        "step_gyro_mag": _stat(step_rows, "gyro_mag") if step_rows else None,
        "step_ar_axis": _stat(step_rows, "ar_body", idx=_STEP_AXIS.get(args.step, 1)) if step_rows else None,
        "race_gate_seen": sorted({r["race_gate"] for r in samples if r["race_gate"] is not None}),
        "had_position": any(r["pz"] is not None for r in samples),
        "had_motors": any(r["motors"] is not None for r in samples),
        "autopilot": client.autopilot,
        "custom_mode": client.custom_mode,
    }
    # heuristic clean-hover flag
    g = verdict["hover_accel_mag"]
    gy = verdict["hover_gyro_mag"]
    verdict["clean_hover"] = bool(
        verdict["collisions_in_window"] == 0
        and g is not None and abs(g["mean"] - 9.8) < 3.0 and g["std"] < 4.0
        and gy is not None and gy["mean"] < 1.0
    )

    out = {"verdict": verdict, "phase_marks": phase_marks, "samples": samples}
    outp.write_text(json.dumps(out, indent=2))
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        if k in ("collisions", "samples"):
            continue
        print(f"  {k:22s}: {v}")
    print(f"[probe] wrote {outp}")


if __name__ == "__main__":
    main()
