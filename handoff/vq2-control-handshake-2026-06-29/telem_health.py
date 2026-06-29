"""VQ2 telemetry-health probe: is the sim lagging on the CONTROL/wire end?

Display (GPU) lag and physics/telemetry lag can be decoupled. This measures the wire directly,
independent of what the screen looks like:

  * per-type inbound message RATE vs the recon baseline (HIGHRES_IMU ~117 Hz,
    ACTUATOR_OUTPUT_STATUS ~95 Hz, RACE_STATUS ~4 Hz, HEARTBEAT ~1 Hz).
  * REAL-TIME FACTOR: how fast the sim clock (HIGHRES_IMU.time_usec) advances vs wall-clock.
    1.0 = real time; <1.0 = the sim physics is running slow (a true lag that WOULD hurt control,
    since our setpoint stream + the policy run on wall-clock).
  * inter-arrival JITTER / STALLS: max gap between consecutive HIGHRES_IMU messages (wall-clock).
    Big gaps = the tick loop is hitching (the kind of stall that can desync a control loop).
  * our OUTGOING setpoint stream: achieved send rate vs requested (are WE keeping up?).

Optionally streams a benign CTBR setpoint (zero rates + given thrust) while measuring, to see the
wire under control load. Pure measurement otherwise.

Usage:
  python telem_health.py --seconds 10 --out logs/health_idle.json
  python telem_health.py --seconds 10 --stream --thrust 0.3 --out logs/health_streamed.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")

import numpy as np

for _cand in (Path(__file__).resolve().parents[2] / "src", Path("C:/Users/Shadow/Peregrine/src")):
    if (_cand / "racer" / "mavlink_client.py").exists():
        sys.path.insert(0, str(_cand))
        break

from racer.contracts import ControlCommand, ControlMode  # noqa: E402
from racer.mavlink_client import MavlinkClient  # noqa: E402

# recon baseline rates (Hz) for the high-rate streams
BASELINE = {"HIGHRES_IMU": 117.0, "ACTUATOR_OUTPUT_STATUS": 95.0, "RACE_STATUS": 4.0,
            "HEARTBEAT": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--stream", action="store_true", help="also send a benign CTBR stream")
    ap.add_argument("--thrust", type=float, default=0.3)
    ap.add_argument("--rate-hz", type=float, default=100.0)
    ap.add_argument("--endpoint", default="udpin:127.0.0.1:14550")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outp = Path(args.out)
    if not outp.is_absolute():
        outp = Path(__file__).resolve().parent / outp
    outp.parent.mkdir(parents=True, exist_ok=True)

    client = MavlinkClient(args.endpoint)
    client.send_heartbeats = True
    client.send_timesync = False

    # tap raw inbound messages for precise per-type arrival timing + sim clock
    counts: dict[str, int] = defaultdict(int)
    last_arr: dict[str, float] = {}
    gaps: dict[str, list] = defaultdict(list)
    imu_sim_us: list[int] = []     # HIGHRES_IMU.time_usec (sim clock)
    imu_wall: list[float] = []     # wall-clock arrival (perf_counter)

    def on_msg(msg):
        t = msg.get_type()
        if t == "BAD_DATA":
            counts["BAD_DATA"] += 1
            return
        now = time.perf_counter()
        counts[t] += 1
        if t in last_arr:
            gaps[t].append(now - last_arr[t])
        last_arr[t] = now
        if t == "HIGHRES_IMU":
            imu_sim_us.append(int(msg.time_usec))
            imu_wall.append(now)

    client.on_message = on_msg
    print(f"[health] connecting {args.endpoint} ...")
    client.connect(wait_heartbeat=True, timeout_s=15.0)

    if args.stream:
        client.last_command_ack = None
        client.arm()
        client.wait_armed(True, timeout_s=4.0)

    dt = 1.0 / args.rate_hz
    sends = 0
    t0 = time.perf_counter()
    next_send = t0
    zero = np.zeros(3)
    loop_iters = 0
    while time.perf_counter() - t0 < args.seconds:
        now = time.perf_counter()
        if args.stream and now >= next_send:
            client.send_command(ControlCommand(mode=ControlMode.BODY_RATE,
                                               body_rate=zero, thrust=float(args.thrust)))
            sends += 1
            next_send += dt
        client.pump()
        loop_iters += 1
        time.sleep(0.0005)
    wall = time.perf_counter() - t0
    if args.stream:
        try:
            client.disarm()
        except Exception:
            pass

    # --- analysis ---
    rates = {t: round(n / wall, 1) for t, n in sorted(counts.items())}
    rate_health = {}
    for t, base in BASELINE.items():
        got = rates.get(t, 0.0)
        rate_health[t] = {"got_hz": got, "baseline_hz": base,
                          "pct_of_baseline": round(100.0 * got / base, 1) if base else None}

    # real-time factor from the IMU sim clock
    rtf = None
    sim_span_s = None
    if len(imu_sim_us) > 10:
        sim_span_s = (imu_sim_us[-1] - imu_sim_us[0]) / 1e6
        wall_span = imu_wall[-1] - imu_wall[0]
        if wall_span > 0:
            rtf = round(sim_span_s / wall_span, 3)

    def gap_stats(t):
        g = gaps.get(t, [])
        if not g:
            return None
        gms = [x * 1000.0 for x in g]
        return {"n": len(gms), "mean_ms": round(statistics.mean(gms), 2),
                "p50_ms": round(statistics.median(gms), 2),
                "p99_ms": round(sorted(gms)[int(len(gms) * 0.99)], 2),
                "max_ms": round(max(gms), 2)}

    out = {
        "window_s": round(wall, 3),
        "streamed": bool(args.stream),
        "real_time_factor": rtf,
        "sim_clock_span_s": round(sim_span_s, 3) if sim_span_s is not None else None,
        "rate_health": rate_health,
        "imu_gap_ms": gap_stats("HIGHRES_IMU"),
        "actuator_gap_ms": gap_stats("ACTUATOR_OUTPUT_STATUS"),
        "all_rates_hz": rates,
        "bad_data": counts.get("BAD_DATA", 0),
        "send_requested_hz": args.rate_hz if args.stream else None,
        "send_achieved_hz": round(sends / wall, 1) if args.stream else None,
        "pump_loop_hz": round(loop_iters / wall, 1),
    }
    outp.write_text(json.dumps(out, indent=2))

    print("\n=== TELEMETRY HEALTH ===")
    print(f"  window               : {out['window_s']}s  streamed={out['streamed']}")
    print(f"  REAL-TIME FACTOR     : {rtf}   (1.0 = real time; <1 = sim physics lagging)")
    print(f"  sim-clock span       : {out['sim_clock_span_s']}s of wall {out['window_s']}s")
    for t, h in rate_health.items():
        print(f"  rate {t:22s}: {h['got_hz']:6.1f} Hz  ({h['pct_of_baseline']}% of ~{h['baseline_hz']})")
    print(f"  IMU inter-arrival    : {out['imu_gap_ms']}")
    print(f"  ACTUATOR inter-arr   : {out['actuator_gap_ms']}")
    print(f"  BAD_DATA frames      : {out['bad_data']}")
    if args.stream:
        print(f"  our send rate        : {out['send_achieved_hz']} / {out['send_requested_hz']} Hz requested")
    print(f"  pump loop rate       : {out['pump_loop_hz']} Hz")
    print(f"[health] wrote {outp}")


if __name__ == "__main__":
    main()
