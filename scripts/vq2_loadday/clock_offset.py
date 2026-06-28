"""scripts/vq2_loadday/clock_offset.py — VQ2 load-day TIMESYNC clock-offset probe (C10).

Measures the vision↔IMU clock offset and whether key ordering fields survive on the wire.
Addresses VADR-TS-003 master-plan hole #10 (``TODO(clock): reconcile epochs via TIMESYNC``
in contracts.py).

Two sub-checks:

  C10a  TIMESYNC round-trip offset: send TIMESYNC, receive the echo, compute
        (tc1_echo - ts1_echo)/2 = one-way latency (ns).  Over N rounds, report the
        median/std round-trip offset.  The vision stream's sim_time_ns is stamped
        by the sim (HIGHRES_IMU.time_usec epoch) — if this drifts from the TIMESYNC
        epoch, the IMU→vision alignment is wrong.

  C10b  Gate-ordering field survival: does active_gate_index in RACE_STATUS
        (ENCAPSULATED_DATA) arrive, and does it increment correctly (C10b)?

Usage (live sim):
  python scripts/vq2_loadday/clock_offset.py --seconds 15

Usage (tlog replay — only C10b from pre-recorded RACE_STATUS packets):
  python scripts/vq2_loadday/clock_offset.py --tlog data/runs/<session>/mavlink.tlog

Usage (save JSON):
  python scripts/vq2_loadday/clock_offset.py --seconds 15 --out data/clock_offset.json

Degrades gracefully: "stream absent / not connected" when the sim is unreachable.
TIMESYNC echo may not arrive in all sim versions — the probe reports it as "inconclusive"
rather than erroring.

Note on C10a limitations:
  The TIMESYNC sub-check measures the MAVLink round-trip time, NOT the vision UDP→MAVLink
  epoch skew directly.  True vision↔IMU alignment needs a simultaneous HIGHRES_IMU timestamp
  and a video frame sim_time_ns from the same event — that requires a live session and is
  flagged in the output as NEEDS_LIVE_SESSION.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer.firstcontact import MessageRateTracker, pump_for
from racer.mavlink_client import MavlinkClient

# How many TIMESYNC rounds to attempt (each takes ~one round-trip).
_TIMESYNC_ROUNDS = 20

# Round-trip latency cap: if round-trip > this, the measurement is likely noisy (network
# anomaly), discard for the median calculation.
_RTT_CAP_MS = 50.0

# C10b: gate-ordering verdict thresholds.
# If active_gate_index never advances during the capture window, it's inconclusive.
_MIN_GATE_ADVANCES = 1   # we expect at least this many gate increments in 15 s of flight


def _send_and_recv_timesync(
    client: MavlinkClient, n: int = _TIMESYNC_ROUNDS, inter_s: float = 0.1
) -> list[dict]:
    """Send N TIMESYNC probes and collect the echoed (tc1, ts1) pairs.

    The sim echoes TIMESYNC with tc1=our_tc1 (unchanged) and ts1=sim_now.
    Round-trip latency = recv_ns - send_ns.
    Clock offset (sim minus our clock) ≈ (ts1_echo - tc1_echo) - rtt/2.

    Returns a list of round dicts: send_ns, recv_ns, tc1_sent, ts1_echo, rtt_ns.
    """
    assert client.conn is not None
    results: list[dict] = []

    # Monkey-patch the client's _handle to capture TIMESYNC echoes
    _timesync_echoes: list[dict] = []

    original_handle = client._handle

    def _patched_handle(msg):
        if msg.get_type() == "TIMESYNC":
            # Echo: tc1 field should match what we sent
            _timesync_echoes.append({
                "recv_ns": time.monotonic_ns(),
                "tc1": int(getattr(msg, "tc1", 0)),
                "ts1": int(getattr(msg, "ts1", 0)),
            })
        original_handle(msg)

    client._handle = _patched_handle

    for _ in range(n):
        tc1 = int(time.time_ns())
        send_ns = time.monotonic_ns()
        client.conn.mav.timesync_send(tc1, 0)
        # Drain for up to 0.2 s to collect the echo
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            msg = client.conn.recv_match(blocking=False)
            if msg is not None:
                client._handle(msg)
            # Check if we got a matching echo
            for echo in list(_timesync_echoes):
                if echo["tc1"] == tc1:
                    rtt_ns = echo["recv_ns"] - send_ns
                    results.append({
                        "send_ns": send_ns,
                        "recv_ns": echo["recv_ns"],
                        "tc1_sent": tc1,
                        "ts1_echo": echo["ts1"],
                        "rtt_ns": rtt_ns,
                        # clock offset: sim time minus our time at mid-trip
                        "offset_ns": echo["ts1"] - (tc1 + rtt_ns // 2),
                    })
                    _timesync_echoes.remove(echo)
                    break
            else:
                time.sleep(0.005)
                continue
            break
        time.sleep(inter_s)

    client._handle = original_handle
    return results


def analyze_timesync(rounds: list[dict]) -> dict:
    """Compute C10a metrics from a list of round dicts.

    Pure function, no I/O.
    """
    if not rounds:
        return {
            "n_rounds": 0,
            "n_echoed": 0,
            "echo_rate": 0.0,
            "median_rtt_ms": None,
            "std_rtt_ms": None,
            "median_offset_ms": None,
            "std_offset_ms": None,
            "c10a_timesync_works": False,
            "note": "no TIMESYNC echoes received (sim may not echo TIMESYNC)",
        }

    rtts_ns = np.array([r["rtt_ns"] for r in rounds], dtype=np.float64)
    offsets_ns = np.array([r["offset_ns"] for r in rounds], dtype=np.float64)

    # Discard outliers (> _RTT_CAP_MS ms round-trip)
    cap_ns = _RTT_CAP_MS * 1e6
    good = rtts_ns < cap_ns
    n_good = int(good.sum())

    if n_good == 0:
        return {
            "n_rounds": _TIMESYNC_ROUNDS,
            "n_echoed": len(rounds),
            "echo_rate": len(rounds) / _TIMESYNC_ROUNDS,
            "median_rtt_ms": float(np.median(rtts_ns)) / 1e6,
            "std_rtt_ms": float(np.std(rtts_ns)) / 1e6,
            "median_offset_ms": None,
            "std_offset_ms": None,
            "c10a_timesync_works": False,
            "note": f"all {len(rounds)} echoes had RTT > {_RTT_CAP_MS} ms — network noise?",
        }

    median_rtt_ms = float(np.median(rtts_ns[good])) / 1e6
    std_rtt_ms = float(np.std(rtts_ns[good])) / 1e6
    median_offset_ms = float(np.median(offsets_ns[good])) / 1e6
    std_offset_ms = float(np.std(offsets_ns[good])) / 1e6

    return {
        "n_rounds": _TIMESYNC_ROUNDS,
        "n_echoed": len(rounds),
        "echo_rate": round(len(rounds) / _TIMESYNC_ROUNDS, 3),
        "n_good_rounds": n_good,
        "median_rtt_ms": round(median_rtt_ms, 3),
        "std_rtt_ms": round(std_rtt_ms, 3),
        "median_offset_ms": round(median_offset_ms, 3),
        "std_offset_ms": round(std_offset_ms, 3),
        "c10a_timesync_works": True,
        "note": (
            f"TIMESYNC echoed {len(rounds)}/{_TIMESYNC_ROUNDS} rounds; "
            f"RTT median={median_rtt_ms:.1f} ms; "
            f"offset median={median_offset_ms:.1f} ms (sim minus our clock at mid-trip)"
        ),
    }


def analyze_gate_ordering(race_status_sequence: list[dict]) -> dict:
    """C10b: check whether active_gate_index arrives and increments.

    Pure function, no I/O.
    ``race_status_sequence`` = list of race_status dicts in arrival order (each has
    ``active_gate_index``, ``started``, ``finished``).
    """
    if not race_status_sequence:
        return {
            "n_race_status_pkts": 0,
            "active_gate_index_present": False,
            "gate_indices_seen": [],
            "n_gate_advances": 0,
            "race_ever_started": False,
            "race_ever_finished": False,
            "c10b_gate_ordering_works": False,
            "note": "no RACE_STATUS (ENCAPSULATED_DATA) packets received",
        }

    n = len(race_status_sequence)
    gate_indices = [int(r.get("active_gate_index", 0)) for r in race_status_sequence]
    started_any = any(r.get("started", False) for r in race_status_sequence)
    finished_any = any(r.get("finished", False) for r in race_status_sequence)

    # Count gate advances: consecutive increases
    n_advances = 0
    for i in range(1, len(gate_indices)):
        if gate_indices[i] > gate_indices[i - 1]:
            n_advances += 1

    unique_indices = sorted(set(gate_indices))

    works = (
        len(unique_indices) > 1 or n_advances >= _MIN_GATE_ADVANCES or started_any
    )
    # If no race started and all gate indices are 0, it's inconclusive (pre-flight)
    if all(gi == 0 for gi in gate_indices) and not started_any:
        works = None  # type: ignore[assignment]  # inconclusive

    return {
        "n_race_status_pkts": n,
        "active_gate_index_present": True,
        "gate_indices_seen": unique_indices,
        "n_gate_advances": n_advances,
        "race_ever_started": started_any,
        "race_ever_finished": finished_any,
        "c10b_gate_ordering_works": bool(works) if works is not None else None,
        "note": (
            "RACE_STATUS present; "
            f"gate_indices seen: {unique_indices}; "
            f"n_gate_advances={n_advances}; "
            f"started={started_any} finished={finished_any}"
        ),
    }


def _collect_live(
    endpoint: str, connect_timeout: float, seconds: float
) -> tuple[MavlinkClient, list[dict], list[dict]]:
    """Connect, run TIMESYNC probes, and pump for ``seconds``.

    Returns (client, timesync_rounds, race_status_sequence).
    """
    client = MavlinkClient(endpoint)
    client.send_heartbeats = False
    client.send_timesync = True

    tracker = MessageRateTracker()

    def _tap(msg):
        t = msg.get_type()
        if t != "BAD_DATA":
            tracker.record(t, time.monotonic_ns())

    client.on_message = _tap

    print(f"  Connecting to {endpoint} (timeout {connect_timeout:.0f}s)...", flush=True)
    client.connect(wait_heartbeat=True, timeout_s=connect_timeout)
    print(f"  Connected.  Pumping {seconds:.0f}s...", flush=True)

    race_status_sequence: list[dict] = []

    # Patch to collect RACE_STATUS packets in order
    original_handle = client._handle

    def _race_tap(msg):
        original_handle(msg)
        if msg.get_type() == "ENCAPSULATED_DATA" and client.race_status is not None:
            # Capture the latest race_status each time ENCAPSULATED_DATA fires
            rs = dict(client.race_status)
            if not race_status_sequence or race_status_sequence[-1] != rs:
                race_status_sequence.append(rs)

    client._handle = _race_tap

    # Run TIMESYNC probes first (before pumping noisily)
    print(f"  Running {_TIMESYNC_ROUNDS} TIMESYNC probes...", flush=True)
    timesync_rounds = _send_and_recv_timesync(client, n=_TIMESYNC_ROUNDS)
    print(f"  Echoes received: {len(timesync_rounds)}/{_TIMESYNC_ROUNDS}", flush=True)

    # Pump the remaining time for RACE_STATUS + telemetry
    pump_for(client, seconds, show_statustext=True)

    client._handle = original_handle
    return client, timesync_rounds, race_status_sequence


def _drain_tlog(tlog_path: Path) -> tuple[MavlinkClient, list[dict]]:
    """Replay a tlog, collecting RACE_STATUS packets.  TIMESYNC probes not possible offline."""
    import os
    os.environ.setdefault("MAVLINK20", "1")
    from pymavlink import mavutil

    client = MavlinkClient()
    race_status_sequence: list[dict] = []

    conn = mavutil.mavlink_connection(str(tlog_path), input=True)
    while True:
        msg = conn.recv_match(blocking=False)
        if msg is None:
            break
        if msg.get_type() == "BAD_DATA":
            continue
        client._handle(msg)
        if msg.get_type() == "ENCAPSULATED_DATA" and client.race_status is not None:
            rs = dict(client.race_status)
            if not race_status_sequence or race_status_sequence[-1] != rs:
                race_status_sequence.append(rs)

    return client, race_status_sequence


def _print_result(res: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  CLOCK OFFSET PROBE (C10) — TIMESYNC + gate ordering")
    print(f"{'='*65}")
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return

    ts = res.get("timesync", {})
    print(f"\n--- C10a: TIMESYNC clock offset ---")
    print(f"  n_rounds         : {ts.get('n_rounds')}  echoed: {ts.get('n_echoed')}")
    print(f"  echo_rate        : {ts.get('echo_rate')}")
    if ts.get("median_rtt_ms") is not None:
        print(f"  RTT median       : {ts['median_rtt_ms']:.3f} ms  "
              f"std={ts['std_rtt_ms']:.3f} ms")
        print(f"  Clock offset med : {ts['median_offset_ms']:.3f} ms  "
              f"std={ts['std_offset_ms']:.3f} ms")
    print(f"  timesync_works   : {ts.get('c10a_timesync_works')}")
    print(f"  note             : {ts.get('note')}")

    go = res.get("gate_ordering", {})
    print(f"\n--- C10b: Gate ordering (RACE_STATUS) ---")
    print(f"  n_pkts           : {go.get('n_race_status_pkts')}")
    print(f"  gate_indices_seen: {go.get('gate_indices_seen')}")
    print(f"  n_gate_advances  : {go.get('n_gate_advances')}")
    print(f"  race_started     : {go.get('race_ever_started')}")
    print(f"  gate_ordering_ok : {go.get('c10b_gate_ordering_works')}")
    print(f"  note             : {go.get('note')}")

    print(f"\n--- C10 SUMMARY ---")
    if res.get("needs_live_session"):
        print(f"  NOTE: vision↔IMU epoch skew requires live session + video; "
              f"TIMESYNC round-trip is a proxy only.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="How long to pump the live sim after TIMESYNC probes (s)")
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--tlog", default=None,
                    help="Replay a tlog (C10b only — no TIMESYNC probes offline)")
    ap.add_argument("--out", default=None, help="Write result JSON here")
    args = ap.parse_args()

    import os
    os.environ.setdefault("MAVLINK20", "1")

    if args.tlog:
        tlog_path = Path(args.tlog)
        print(f"  Replaying tlog: {tlog_path}  (C10b only — TIMESYNC probes need live sim)", flush=True)
        client, race_status_seq = _drain_tlog(tlog_path)
        timesync_rounds: list[dict] = []
        print(f"  Done.  {len(race_status_seq)} RACE_STATUS transitions captured.", flush=True)
    else:
        try:
            client, timesync_rounds, race_status_seq = _collect_live(
                args.endpoint, args.connect_timeout, args.seconds
            )
        except (TimeoutError, OSError, Exception) as exc:
            print(f"\n  ERROR: stream absent / not connected: {exc}", file=sys.stderr)
            print(f"  -> Start the sim first, or use --tlog to replay a recording.", file=sys.stderr)
            return 2

    ts_result = analyze_timesync(timesync_rounds)
    go_result = analyze_gate_ordering(race_status_seq)

    result = {
        "timesync": ts_result,
        "gate_ordering": go_result,
        "needs_live_session": (
            "vision↔IMU epoch skew needs live session + video feed; "
            "TIMESYNC round-trip is a proxy for the MAVLink epoch skew only"
        ),
    }

    _print_result(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\n  Result written to {out_path}")

    # Exit 0 only if TIMESYNC echo works OR gate ordering is confirmed
    ts_ok = ts_result.get("c10a_timesync_works", False)
    go_ok = go_result.get("c10b_gate_ordering_works") is True
    return 0 if (ts_ok or go_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
