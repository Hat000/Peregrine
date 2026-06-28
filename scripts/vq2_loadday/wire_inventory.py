"""scripts/vq2_loadday/wire_inventory.py — VQ2 load-day wire probe (C0/C1/C5/C10).

Connects (or replays a recorded session), inventories which MAVLink streams are present,
their rates, and which fields are non-None in the DroneState snapshot.  The pivotal mode
is ``--diff TRAINING COMPETITIVE``: it reads TWO saved inventory JSON files (one captured
in Training mode, one in Competitive mode) and reports what Training exposes that
Competitive blocks.  That diff is check C5 (Training GT exposure) — the gate for
privileged-distillation, offline boresight calibration, and VGGT gate-labelling forks.

Usage (live sim):
  python scripts/vq2_loadday/wire_inventory.py --label training --seconds 30
  python scripts/vq2_loadday/wire_inventory.py --label competitive --seconds 30

Usage (diff two saved inventories — C5):
  python scripts/vq2_loadday/wire_inventory.py \\
      --diff data/wire_inventory_training.json data/wire_inventory_competitive.json

Usage (offline — replay a recorded tlog; no sim needed):
  python scripts/vq2_loadday/wire_inventory.py --tlog data/runs/<session>/mavlink.tlog

Degrades gracefully when no sim is reachable: "stream absent / not connected" message,
non-zero exit code, no crash.

Checks addressed:
  C0  Connectivity + mode handshake (can we get a HEARTBEAT?)
  C1  Blocked-telemetry in Competitive (ATTITUDE/LPN/ODOMETRY/GATE_INFO absent?)
  C5  Training GT exposure (--diff mode: what does Training expose that Competitive hides?)
  C10 TIMESYNC presence (clock-offset check preamble; active_gate_index in RACE_STATUS)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer.firstcontact import (
    MessageRateTracker,
    backend_summary,
    pump_for,
    telemetry_summary,
)
from racer.mavlink_client import MavlinkClient

# VQ2 §9.3 wires: present in Training, BLOCKED in Competitive scoring.
# These are the streams whose presence = GT exposure.
_TRAINING_ONLY_STREAMS = {
    "ATTITUDE", "LOCAL_POSITION_NED", "ODOMETRY", "GATE_INFO",
}

# Streams that MUST survive in Competitive (per VADR-TS-003 §9.3 / §4.6).
_COMPETITIVE_REQUIRED = {
    "HIGHRES_IMU", "HEARTBEAT", "TIMESYNC",
    # Vision arrives as UDP video (not MAVLink), so it won't appear in the rate table,
    # but ENCAPSULATED_DATA carries RACE_STATUS — still present?
    "ENCAPSULATED_DATA",
}

# These flags on DroneState are only populated by the blocked streams.
# None = blocked; non-None = leaking GT.
_GT_PRESENCE_FIELDS = {
    "position_ned":          "LOCAL_POSITION_NED / ODOMETRY",
    "velocity_ned":          "LOCAL_POSITION_NED / ODOMETRY",
    "orientation_ned_wxyz":  "ODOMETRY",
}


def _drain_tlog(tlog_path: Path, client: MavlinkClient, tracker: MessageRateTracker) -> None:
    """Replay a tlog (pymavlink format) into the client and tracker without a live sim."""
    os.environ.setdefault("MAVLINK20", "1")
    import os
    from pymavlink import mavutil

    conn = mavutil.mavlink_connection(str(tlog_path), input=True)
    while True:
        msg = conn.recv_match(blocking=False)
        if msg is None:
            break
        t = msg.get_type()
        if t == "BAD_DATA":
            continue
        recv = time.monotonic_ns()
        tracker.record(t, recv)
        client._handle(msg)


def _collect_live(
    endpoint: str,
    seconds: float,
    connect_timeout: float,
) -> tuple[MavlinkClient, MessageRateTracker]:
    """Connect to the sim and pump for `seconds`.  Raises on connection failure."""
    client = MavlinkClient(endpoint)
    # Stay silent (no GCS heartbeat) so we don't flip the sim to ACRO:
    # mirror the reference client's keepalive approach.
    client.send_heartbeats = False
    client.send_timesync = True

    tracker = MessageRateTracker()
    original_on_message = client.on_message

    def _tap(msg):
        t = msg.get_type()
        if t != "BAD_DATA":
            tracker.record(t, time.monotonic_ns())
        if original_on_message:
            original_on_message(msg)

    client.on_message = _tap

    print(f"  Connecting to {endpoint} (timeout {connect_timeout:.0f}s)...", flush=True)
    client.connect(wait_heartbeat=True, timeout_s=connect_timeout)
    print(f"  Connected.  Backend: {backend_summary(client)}", flush=True)
    print(f"  Pumping {seconds:.0f}s ...", flush=True)
    pump_for(client, seconds, show_statustext=True)
    return client, tracker


def _build_inventory(
    client: MavlinkClient, tracker: MessageRateTracker, label: str
) -> dict:
    """Assemble the inventory dict from a pumped client + tracker."""
    s = client.state

    # Per-stream rate table
    rates = tracker.rates()
    streams: dict[str, dict] = {}
    for msg_type, r in rates.items():
        streams[msg_type] = {
            "present": True,
            "count": r["count"],
            "rate_hz": round(r["hz"], 2),
        }

    # DroneState field presence (None = stream was blocked)
    field_presence: dict[str, dict] = {}
    for field, source in _GT_PRESENCE_FIELDS.items():
        val = getattr(s, field, None)
        field_presence[field] = {
            "non_none": val is not None,
            "source_stream": source,
        }

    # Presence flags for the key streams
    blocked = {t: (t not in streams) for t in _TRAINING_ONLY_STREAMS}
    required_ok = {t: (t in streams) for t in _COMPETITIVE_REQUIRED}

    # RACE_STATUS (via ENCAPSULATED_DATA) — does active_gate_index arrive?
    race_status = client.race_status
    gate_map_gates = len(client.track_gates) if client.track_gates else 0

    return {
        "label": label,
        "backend": {
            "autopilot": client.autopilot,
            "vehicle_type": client.vehicle_type,
            "custom_mode": client.custom_mode,
        },
        "streams": streams,
        "gt_field_presence": field_presence,
        "training_only_blocked": blocked,   # True = blocked (good for Competitive)
        "competitive_required_ok": required_ok,
        "race_status": race_status,
        "gate_map_gates": gate_map_gates,
        "statustexts": client.statustexts[-20:],
        "unknown_msg_types": sorted(client.unknown_msg_types),
    }


def _print_inventory(inv: dict) -> None:
    label = inv["label"]
    print(f"\n{'='*60}")
    print(f"  WIRE INVENTORY  label={label!r}")
    print(f"{'='*60}")
    print(f"\n{'stream':<30} {'present':>8} {'count':>7} {'rate_hz':>9}")
    print("-" * 57)
    for t, r in sorted(inv["streams"].items()):
        print(f"{t:<30} {'YES':>8} {r['count']:>7} {r['rate_hz']:>9.1f}")

    print(f"\n--- GT field presence (non-None = GT surviving on wire) ---")
    for field, info in inv["gt_field_presence"].items():
        status = "NON-NONE (GT PRESENT)" if info["non_none"] else "None    (blocked/absent)"
        print(f"  {field:<28} {status}  [{info['source_stream']}]")

    print(f"\n--- VQ2 §9.3 blocked-stream check (True = stream absent as expected) ---")
    for t, blocked in inv["training_only_blocked"].items():
        flag = "BLOCKED (expected for Competitive)" if blocked else "PRESENT (GT leaking!)"
        print(f"  {t:<28} {flag}")

    print(f"\n--- Competitive-required streams ---")
    for t, ok in inv["competitive_required_ok"].items():
        print(f"  {t:<28} {'OK' if ok else 'MISSING!'}")

    rs = inv.get("race_status")
    if rs:
        print(f"\n--- RACE_STATUS (via ENCAPSULATED_DATA) ---")
        print(f"  active_gate_index: {rs.get('active_gate_index')}  "
              f"started: {rs.get('started')}  finished: {rs.get('finished')}")
    else:
        print(f"\n  RACE_STATUS: absent (ENCAPSULATED_DATA not seen or not decoded)")
    print(f"\n  GATE_MAP: {inv.get('gate_map_gates', 0)} gates received via TRACK_INFO")

    if inv.get("unknown_msg_types"):
        print(f"\n  UNKNOWN message types (NEW in VQ2?): {inv['unknown_msg_types']}")


def _diff(path_a: Path, path_b: Path) -> None:
    """C5: diff two inventory files, highlighting what Training exposes that Competitive hides."""
    with open(path_a) as f:
        inv_a = json.load(f)
    with open(path_b) as f:
        inv_b = json.load(f)

    label_a = inv_a.get("label", str(path_a))
    label_b = inv_b.get("label", str(path_b))

    print(f"\n{'='*70}")
    print(f"  C5 DIFF: {label_a!r}  vs  {label_b!r}")
    print(f"  (streams present in A but absent in B = Training GT exposure)")
    print(f"{'='*70}")

    streams_a = set(inv_a.get("streams", {}).keys())
    streams_b = set(inv_b.get("streams", {}).keys())

    only_a = streams_a - streams_b
    only_b = streams_b - streams_a
    shared = streams_a & streams_b

    if only_a:
        print(f"\n  Streams ONLY in {label_a!r}  (Training-exclusive GT exposure):")
        for t in sorted(only_a):
            r = inv_a["streams"][t]
            training_only_flag = "  <-- SPEC §9.3 TRAINING-ONLY" if t in _TRAINING_ONLY_STREAMS else ""
            print(f"    {t:<30} {r['rate_hz']:>7.1f} Hz  (n={r['count']}){training_only_flag}")
    else:
        print(f"\n  No streams exclusive to {label_a!r}.")

    if only_b:
        print(f"\n  Streams ONLY in {label_b!r}  (unexpected in Competitive?):")
        for t in sorted(only_b):
            r = inv_b["streams"][t]
            print(f"    {t:<30} {r['rate_hz']:>7.1f} Hz  (n={r['count']})")

    print(f"\n  Shared streams ({len(shared)} total):")
    for t in sorted(shared):
        ra = inv_a["streams"][t]
        rb = inv_b["streams"][t]
        print(f"    {t:<30} A: {ra['rate_hz']:>6.1f} Hz  B: {rb['rate_hz']:>6.1f} Hz")

    print(f"\n--- C5 GT-field presence delta ---")
    fp_a = inv_a.get("gt_field_presence", {})
    fp_b = inv_b.get("gt_field_presence", {})
    for field in set(list(fp_a.keys()) + list(fp_b.keys())):
        a_val = fp_a.get(field, {}).get("non_none", False)
        b_val = fp_b.get(field, {}).get("non_none", False)
        src = fp_a.get(field, fp_b.get(field, {})).get("source_stream", "?")
        if a_val != b_val:
            print(f"  {field:<28} A={'YES' if a_val else 'NO':>3}  B={'YES' if b_val else 'NO':>3}  "
                  f"[{src}]  <-- DELTA")
        else:
            print(f"  {field:<28} A={'YES' if a_val else 'NO':>3}  B={'YES' if b_val else 'NO':>3}  "
                  f"[{src}]")

    # C5 verdict
    c5_gt_leaked = bool(only_a & _TRAINING_ONLY_STREAMS)
    field_leaked = any(
        fp_a.get(f, {}).get("non_none") and not fp_b.get(f, {}).get("non_none")
        for f in fp_a
    )
    print(f"\n  C5 VERDICT:")
    print(f"    Training leaks spec-blocked MAVLink streams: {'YES' if c5_gt_leaked else 'NO'}")
    print(f"    Training has non-None GT fields that Competitive lacks: {'YES' if field_leaked else 'NO'}")
    if c5_gt_leaked or field_leaked:
        print(f"    => TRAINING DOES EXPOSE GT — privileged distillation / offline calib VIABLE")
    else:
        print(f"    => No Training/Competitive delta detected — same wire both modes")
        print(f"       (Could mean: same endpoint; same mode; or VQ2 exposes GT in Competitive too)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", default="inventory", help="Tag for this capture (e.g. 'training', 'competitive')")
    ap.add_argument("--seconds", type=float, default=30.0, help="How long to pump the live sim (s)")
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550", help="MAVLink UDP endpoint")
    ap.add_argument("--connect-timeout", type=float, default=10.0, help="Heartbeat wait timeout (s)")
    ap.add_argument("--out", default=None, help="Write inventory JSON here (for --diff later)")
    ap.add_argument(
        "--tlog", default=None,
        help="Replay a recorded tlog instead of connecting to a live sim"
    )
    ap.add_argument(
        "--diff", nargs=2, metavar=("TRAINING_JSON", "COMPETITIVE_JSON"),
        help="Diff two saved inventory files (C5 mode)"
    )
    args = ap.parse_args()

    if args.diff:
        _diff(Path(args.diff[0]), Path(args.diff[1]))
        return 0

    import os
    os.environ.setdefault("MAVLINK20", "1")

    if args.tlog:
        tlog_path = Path(args.tlog)
        print(f"  Replaying tlog: {tlog_path}", flush=True)
        client = MavlinkClient()
        tracker = MessageRateTracker()
        _drain_tlog(tlog_path, client, tracker)
        print(f"  Done.  {sum(r['count'] for r in tracker.rates().values())} messages replayed.")
    else:
        try:
            client, tracker = _collect_live(
                args.endpoint, args.seconds, args.connect_timeout
            )
        except (TimeoutError, OSError, Exception) as exc:
            print(f"\n  ERROR: stream absent / not connected: {exc}", file=sys.stderr)
            print(f"  -> Start the sim first, or use --tlog to replay a recording.", file=sys.stderr)
            return 2

    inv = _build_inventory(client, tracker, args.label)
    _print_inventory(inv)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(inv, indent=2, default=str), encoding="utf-8")
        print(f"\n  Inventory written to {out_path}")

    # Exit 1 if any Competitive-required stream is missing (C0 connectivity gate)
    missing = [t for t, ok in inv["competitive_required_ok"].items() if not ok]
    if missing:
        print(f"\n  WARNING: required streams missing: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
