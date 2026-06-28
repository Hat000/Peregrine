"""scripts/vq2_loadday/two_load_hash.py — cheapest moat falsifier (C3/C4).

Captures a minimal "fingerprint" from one sim load, then diffs two fingerprints across
two boots to answer the binary "is anything stable?" question:

  - Gate-1 NED coordinate (from TRACK_INFO gate map via ENCAPSULATED_DATA)
  - One RGB frame buffer hash from a fixed spawn pose

If the Gate-1 coordinate shifts >5 cm across boots OR the frame hashes differ,
per-load randomization is likely → the frozen-map strategy is falsified.

This is the CHEAPEST MOAT FALSIFIER: run it before writing any VGGT/DPVO code.

CAPTURE (run once per boot, save a fingerprint JSON):
  python scripts/vq2_loadday/two_load_hash.py capture \\
      --label load1 --out data/two_load_hash_load1.json

DIFF (compare two fingerprints):
  python scripts/vq2_loadday/two_load_hash.py diff \\
      data/two_load_hash_load1.json data/two_load_hash_load2.json

The capture mode connects to the live sim, waits for the TRACK_INFO gate map to arrive
(from ENCAPSULATED_DATA + DATA_TRANSMISSION_HANDSHAKE), grabs one video frame hash, then
writes the fingerprint.  No flight needed — the sim announces the track on connection.

Degrades gracefully: if TRACK_INFO never arrives (Competitive mode blocks GATE_INFO), it
still captures whatever it can and notes that the gate map was absent.

Note on Competitive mode: VADR-TS-003 §9.3 blocks GATE_INFO in Competitive.  The sim
delivers the track via ENCAPSULATED_DATA (our TRACK_INFO parsing code), which is a
non-standard MAVLink repurposing — it may or may not survive in Competitive.  The
presence/absence of track_gates after 30 s is itself a C1 finding (GATE_INFO status).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer.mavlink_client import MavlinkClient

# Gate-coordinate shift threshold for the "moat falsified" verdict.
_GATE_SHIFT_THRESHOLD_M = 0.05   # 5 cm

# How long to wait for the track map to arrive (TRACK_INFO can take a few seconds).
_TRACK_WAIT_S = 20.0

# How long to wait for the first video frame after connecting.
_VIDEO_WAIT_S = 10.0


def _wait_for_track(client: MavlinkClient, timeout_s: float) -> list[dict] | None:
    """Pump until track_gates is populated or timeout; return gates or None."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        client.pump()
        if client.track_gates:
            return client.track_gates
        time.sleep(0.02)
    return None


def _grab_video_frame_hash(endpoint: str = "udp:127.0.0.1:14550", timeout_s: float = 10.0) -> str | None:
    """Try to grab one JPEG frame from the video UDP stream and return its SHA-256 hex.

    The video arrives on a separate UDP port (usually 5600).  We import JpegUdpReceiver
    to grab one frame rather than re-implementing the UDP logic.  Returns None if
    the video port is unreachable or the receiver isn't available.
    """
    try:
        from racer.vision.jpeg_receiver import JpegUdpReceiver
        receiver = JpegUdpReceiver()
        receiver.start()
        frame = None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            f = receiver.latest_frame()
            if f is not None and f.jpeg_bytes is not None:
                frame = f
                break
            time.sleep(0.05)
        receiver.stop()
        if frame is None or frame.jpeg_bytes is None:
            return None
        return hashlib.sha256(frame.jpeg_bytes).hexdigest()
    except Exception as exc:
        # Not a hard error: video may not be running, or the receiver may not be importable.
        return f"error:{type(exc).__name__}:{exc}"


def _serialize_gates(gates: list[dict]) -> list[dict]:
    """Convert numpy arrays in gate dicts to plain lists for JSON serialization."""
    out = []
    for g in gates:
        row = {}
        for k, v in g.items():
            if isinstance(v, np.ndarray):
                row[k] = v.tolist()
            else:
                row[k] = v
        out.append(row)
    return out


def _capture(
    endpoint: str,
    connect_timeout: float,
    label: str,
    out_path: Path,
) -> int:
    import os
    os.environ.setdefault("MAVLINK20", "1")

    client = MavlinkClient(endpoint)
    client.send_heartbeats = False
    client.send_timesync = True

    print(f"  Connecting to {endpoint} (timeout {connect_timeout:.0f}s)...", flush=True)
    try:
        client.connect(wait_heartbeat=True, timeout_s=connect_timeout)
    except (TimeoutError, OSError, Exception) as exc:
        print(f"\n  ERROR: stream absent / not connected: {exc}", file=sys.stderr)
        print(f"  -> Start the sim first.", file=sys.stderr)
        return 2
    print(f"  Connected.  Waiting up to {_TRACK_WAIT_S:.0f}s for TRACK_INFO gate map...", flush=True)

    gates = _wait_for_track(client, _TRACK_WAIT_S)

    if gates is None:
        print(f"  WARNING: TRACK_INFO did not arrive in {_TRACK_WAIT_S:.0f}s")
        print(f"    Possible causes: Competitive mode blocks GATE_INFO; sim didn't send TRACK_INFO yet.")
        gate_map_present = False
        gate_1_ned = None
    else:
        gate_map_present = True
        # Gate-1 is the first gate (index 0 = gate_id 0 in most sim layouts; take gate_id min)
        gates_sorted = sorted(gates, key=lambda g: g["gate_id"])
        gate_1 = gates_sorted[0]
        gate_1_ned = gate_1["position_ned"].tolist() if isinstance(gate_1["position_ned"], np.ndarray) else list(gate_1["position_ned"])
        print(f"  Gate map received: {len(gates)} gates.  Gate-0 NED = {gate_1_ned}")

    # Grab one video frame hash from the fixed spawn pose.
    print(f"  Grabbing one video frame hash (timeout {_VIDEO_WAIT_S:.0f}s)...", flush=True)
    frame_hash = _grab_video_frame_hash(endpoint, _VIDEO_WAIT_S)
    if frame_hash and not frame_hash.startswith("error:"):
        print(f"  Frame hash (SHA-256): {frame_hash[:16]}...", flush=True)
    else:
        print(f"  Video frame: {frame_hash or 'none received'}", flush=True)

    # Capture sim boot time from RACE_STATUS if available
    sim_boot_ms = None
    if client.race_status:
        sim_boot_ms = client.race_status.get("sim_boot_time_ms")

    fingerprint = {
        "label": label,
        "capture_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": endpoint,
        "gate_map_present": gate_map_present,
        "gate_1_ned": gate_1_ned,
        "n_gates": len(gates) if gates else 0,
        "all_gates": _serialize_gates(gates) if gates else [],
        "frame_hash_sha256": frame_hash,
        "sim_boot_ms": sim_boot_ms,
        "race_status": client.race_status,
        "statustexts_first_10": client.statustexts[:10],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fingerprint, indent=2, default=str), encoding="utf-8")
    print(f"\n  Fingerprint written to {out_path}")
    return 0


def _diff(path_a: Path, path_b: Path) -> int:
    """Compare two fingerprints and report the moat falsification verdict."""
    with open(path_a) as f:
        fp_a = json.load(f)
    with open(path_b) as f:
        fp_b = json.load(f)

    label_a = fp_a.get("label", str(path_a))
    label_b = fp_b.get("label", str(path_b))

    print(f"\n{'='*65}")
    print(f"  TWO-LOAD HASH DIFF (C3/C4 moat falsifier)")
    print(f"  A: {label_a!r}  ({fp_a.get('capture_time_utc', '?')})")
    print(f"  B: {label_b!r}  ({fp_b.get('capture_time_utc', '?')})")
    print(f"{'='*65}")

    gate_verdict = "UNKNOWN"
    gate_shift_m = None

    ned_a = fp_a.get("gate_1_ned")
    ned_b = fp_b.get("gate_1_ned")

    if ned_a is None and ned_b is None:
        print(f"\n  Gate-1 NED: ABSENT in both (TRACK_INFO not received in either load)")
        print(f"    => Cannot assess geometry stability.  Check C1 (GATE_INFO blocked?).")
    elif ned_a is None or ned_b is None:
        print(f"\n  Gate-1 NED: present in only one load (A: {ned_a is not None}, B: {ned_b is not None})")
        gate_verdict = "UNSTABLE (map appeared in one load only)"
    else:
        shift = float(np.linalg.norm(np.array(ned_a) - np.array(ned_b)))
        gate_shift_m = shift
        stable = shift <= _GATE_SHIFT_THRESHOLD_M
        gate_verdict = "STABLE" if stable else f"SHIFTED {shift:.3f} m (>{_GATE_SHIFT_THRESHOLD_M} m threshold)"
        print(f"\n  Gate-1 NED  A: {[round(x, 4) for x in ned_a]}")
        print(f"  Gate-1 NED  B: {[round(x, 4) for x in ned_b]}")
        print(f"  Gate-1 shift  : {shift:.4f} m  => {gate_verdict}")

    hash_a = fp_a.get("frame_hash_sha256")
    hash_b = fp_b.get("frame_hash_sha256")

    print(f"\n  Frame hash A: {hash_a}")
    print(f"  Frame hash B: {hash_b}")

    # Only compare if both are real hashes (not error strings or None)
    def _is_real_hash(h):
        return isinstance(h, str) and len(h) == 64 and not h.startswith("error:")

    if _is_real_hash(hash_a) and _is_real_hash(hash_b):
        frames_match = hash_a == hash_b
        print(f"  Frames identical: {frames_match}")
        if not frames_match:
            print(f"    => Frame buffers differ (lighting/textures/random seed changed across loads?)")
            print(f"       Note: pixel difference alone may not falsify the GEOMETRY moat — only the NED shift does.")
    else:
        frames_match = None
        print(f"  Frame comparison: SKIPPED (one or both hashes are not valid SHA-256)")
        print(f"    (video may not have been running, or JpegUdpReceiver not available)")

    # n_gates
    ng_a = fp_a.get("n_gates", 0)
    ng_b = fp_b.get("n_gates", 0)
    print(f"\n  n_gates   A: {ng_a}   B: {ng_b}   {'SAME' if ng_a == ng_b else 'DIFFERENT!'}")

    # All-gates comparison (geometry)
    if fp_a.get("all_gates") and fp_b.get("all_gates") and ng_a == ng_b:
        max_shift = 0.0
        for ga, gb in zip(fp_a["all_gates"], fp_b["all_gates"]):
            s = float(np.linalg.norm(np.array(ga["position_ned"]) - np.array(gb["position_ned"])))
            max_shift = max(max_shift, s)
        print(f"  All-gates max position shift: {max_shift:.4f} m  "
              f"({'STABLE' if max_shift <= _GATE_SHIFT_THRESHOLD_M else 'SHIFTED'})")

    print(f"\n--- VERDICT ---")
    moat_falsified = (
        (gate_shift_m is not None and gate_shift_m > _GATE_SHIFT_THRESHOLD_M)
        or gate_verdict.startswith("UNSTABLE")
        or gate_verdict.startswith("SHIFTED")
    )

    if moat_falsified:
        print(f"  MOAT FALSIFIED: geometry shifts across loads.")
        print(f"  => Frozen-map / recon-map strategy dead.  Gate-relative + GRU only.")
        return 1
    elif gate_shift_m is not None and gate_shift_m <= _GATE_SHIFT_THRESHOLD_M:
        print(f"  GATE GEOMETRY STABLE across these two loads (shift {gate_shift_m:.4f} m < {_GATE_SHIFT_THRESHOLD_M} m).")
        if frames_match is False:
            print(f"  BUT: frame buffers DIFFER (appearance unstable — lighting/texture seed changes).")
            print(f"       Geometry moat survives; appearance-based relocalization may not.")
        elif frames_match:
            print(f"  AND: frame buffers IDENTICAL (appearance stable too).")
        else:
            print(f"  Frame comparison was inconclusive (no video or hash error).")
        print(f"  => Run again on ≥3 loads to confirm C4 (per-load randomization).")
        return 0
    else:
        print(f"  INCONCLUSIVE: gate map absent in one or both loads.")
        print(f"  => Confirm TRACK_INFO is being sent (check C1 GATE_INFO blocking).")
        return 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Capture a fingerprint from one sim boot")
    cap.add_argument("--label", default="load", help="Label for this capture (e.g. 'load1')")
    cap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    cap.add_argument("--connect-timeout", type=float, default=10.0)
    cap.add_argument("--out", required=True, help="Write fingerprint JSON here")

    dif = sub.add_parser("diff", help="Diff two saved fingerprints (C3/C4 moat falsifier)")
    dif.add_argument("fingerprint_a", help="First fingerprint JSON (load 1)")
    dif.add_argument("fingerprint_b", help="Second fingerprint JSON (load 2)")

    args = ap.parse_args()

    if args.cmd == "capture":
        return _capture(
            endpoint=args.endpoint,
            connect_timeout=args.connect_timeout,
            label=args.label,
            out_path=Path(args.out),
        )
    else:  # diff
        return _diff(Path(args.fingerprint_a), Path(args.fingerprint_b))


if __name__ == "__main__":
    raise SystemExit(main())
