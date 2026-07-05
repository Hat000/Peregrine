"""Dump the raw race wire (ENCAPSULATED_DATA + COLLISION) from a recorded session to JSONL.

Evidence-package tool for the RL team: one JSONL line per relevant MAVLink message, in
recording order, with the RACE_STATUS payload decoded (data_type==1) and COLLISION events
included as their own lines. Read-only against the recording -- never modifies a session dir.

Struct reference (verbatim from src/racer/mavlink_client.py, NOT re-derived; this script
imports parse_race_status directly rather than re-implementing the decode):
  RACE_STATUS payload (ENCAPSULATED_DATA.data[0] == 1), struct "<BQqqIq":
    data_type(B), sim_boot_ms(Q), race_start_boot_ms(q), race_finish_ns(q),
    active_gate_index(I), last_gate_race_time(q).  (<0 = not started / ongoing.)

Usage:
    python handoff/tools/dump_race_wire.py <session_dir> [-o out.jsonl]

If -o is omitted, writes to stdout.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# Make "racer" importable when run from anywhere: this repo keeps it under src/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from racer.mavlink_client import parse_race_status  # noqa: E402
from racer.recording import RecordingReader  # noqa: E402

_ENCAP_RACE_STATUS = 1
_ENCAP_TRACK_INFO = 2


def _msg_timestamp(msg) -> float:
    """Best-effort recv timestamp for a pymavlink message read back from a tlog.

    RecordingReader.iter_mavlink() uses mavutil.mavlink_connection(...) on the tlog file,
    which (per pymavlink's mavutil.mavlogfile reader) stamps each parsed message with a
    ``_timestamp`` attribute: the record's 8-byte big-endian UNIX-epoch-microsecond prefix
    written by racer.recording.Recorder (see recording.py _TLOG_STAMP / _writer_loop),
    converted to float seconds since the UNIX epoch. This is a RECEIVE-time stamp (recorder
    stamps at time.monotonic_ns() receive, then converts to a synthesised UNIX epoch via the
    meta.json t0 bridge) -- not a write-time or send-time stamp. Fall back to 0.0 if absent
    (should not happen for a tlog-backed reader, but keep this tool from crashing on odd input).
    """
    ts = getattr(msg, "_timestamp", None)
    return float(ts) if ts is not None else 0.0


def dump_session(session_dir: Path):
    """Yield one JSON-serializable dict per ENCAPSULATED_DATA / COLLISION message, in order."""
    for msg in RecordingReader(session_dir).iter_mavlink():
        t = msg.get_type()
        if t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            data_type = raw[0] if raw else None
            record = {
                "recv_timestamp": _msg_timestamp(msg),
                "msg_type": "ENCAPSULATED_DATA",
                "data_type": data_type,
                "raw_hex": raw.hex(),
            }
            if data_type == _ENCAP_RACE_STATUS:
                decoded = parse_race_status(raw)
                record["decoded"] = decoded
                if decoded is None:
                    record["decode_error"] = (
                        f"payload too short for RACE_STATUS: {len(raw)} bytes, "
                        f"need >= {struct.calcsize('<BQqqIq')}"
                    )
            elif data_type == _ENCAP_TRACK_INFO:
                record["decoded"] = None
                record["note"] = "TRACK_INFO chunk (gate map); not decoded by this tool"
            yield record
        elif t == "COLLISION":
            yield {
                "recv_timestamp": _msg_timestamp(msg),
                "msg_type": "COLLISION",
                "data_type": None,
                "raw_hex": bytes(msg.get_msgbuf()).hex(),
                "decoded": {
                    "id": int(msg.id),
                    "threat_level": int(getattr(msg, "threat_level", 0)),
                    "impulse": float(getattr(msg, "horizontal_minimum_delta", 0.0)),
                },
            }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path, help="recorded session directory")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output JSONL path (default: stdout)")
    args = ap.parse_args()

    out_f = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        n = 0
        for record in dump_session(args.session_dir):
            out_f.write(json.dumps(record, default=str) + "\n")
            n += 1
        if args.out:
            print(f"wrote {n} lines to {args.out}", file=sys.stderr)
    finally:
        if args.out:
            out_f.close()


if __name__ == "__main__":
    main()
