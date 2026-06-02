"""First-contact probe: HOW and WHEN does the sim deliver the TRACK_INFO gate map?

The gate map is PUSHED, unsolicited (the official PyAIPilotExample sends no request): a
DATA_TRANSMISSION_HANDSHAKE announces a transfer (transfer_id = msg.width, chunk count =
msg.packets), then N ENCAPSULATED_DATA packets whose data[0]==2 carry the chunks (reassembled
by seqnr) into <H num_gates> + per-gate <Hfffffffff>. RACE_STATUS is the same envelope with
data[0]==1. The TRIGGER/timing is a LIVE unknown, so this probe logs the raw handshake +
encapsulated traffic and walks phases (passive -> optional arm -> optional sim reset) to show
WHERE, if anywhere, the map appears -- then prints a diagnosis of why it didn't.

Tip: if the map is a one-shot sent at connect, wait_heartbeat() may have consumed it before the
handler loop started -> use --reset to force a fresh broadcast WHILE we are listening, and/or
start the race in the sim UI (the map may be gated on race start -- watch RACE_STATUS.started).

SAFE by default (listens + >=2 Hz heartbeat only). --arm arms (force-disarms on exit);
--reset sends MAV_CMD 31000.

Usage:
  python scripts/track_probe.py [--endpoint udp:127.0.0.1:14550] [--phase-s 8] [--arm] [--reset]
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.firstcontact import backend_summary, drain_statustexts, mission_summary
from racer.mavlink_client import MavlinkClient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--phase-s", type=float, default=8.0, help="observation seconds per phase")
    ap.add_argument("--arm", action="store_true", help="also try arming (force-disarms on exit)")
    ap.add_argument("--reset", action="store_true", help="also send MAV_CMD 31000 sim reset to re-trigger")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    client = MavlinkClient(args.endpoint)
    diag = {"handshakes": 0, "encap": Counter(), "track_chunks": 0}

    def tap(msg) -> None:
        # Runs BEFORE _handle each pump, so we log the raw wire while _handle still reassembles
        # into client.track_gates / client.race_status.
        t = msg.get_type()
        if t == "DATA_TRANSMISSION_HANDSHAKE":
            diag["handshakes"] += 1
            print(f"    HANDSHAKE  transfer_id(width)={int(msg.width)}  packets={int(msg.packets)}  "
                  f"size={int(getattr(msg, 'size', 0))}")
        elif t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            dt = raw[0] if raw else -1
            diag["encap"][dt] += 1
            if dt == 2:
                diag["track_chunks"] += 1
                tid = struct.unpack_from("<BH", raw, 0)[1] if len(raw) >= 3 else -1
                if diag["track_chunks"] <= 3 or diag["track_chunks"] % 25 == 0:
                    print(f"    TRACK chunk seqnr={int(getattr(msg, 'seqnr', -1))} transfer_id={tid} "
                          f"bytes={len(raw)}")

    client.on_message = tap

    print(f"connecting MAVLink {args.endpoint} (waiting for heartbeat) ...")
    try:
        client.connect(timeout_s=args.connect_timeout)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"heartbeat OK. backend: {backend_summary(client)}\n")

    captured_phase: str | None = None

    def observe(label: str) -> None:
        nonlocal captured_phase
        print(f"[{label}] observing {args.phase_s:g}s ...")
        seen = len(client.statustexts)
        had = client.track_gates is not None
        end = time.monotonic() + args.phase_s
        while time.monotonic() < end:
            client.pump()
            seen = drain_statustexts(client, seen)
            if not had and client.track_gates is not None:
                had = True
                captured_phase = label
                gates = client.track_gates
                print(f"    *** GATE MAP CAPTURED in [{label}]: {len(gates)} gates ***")
                for g in gates[:3]:
                    p = g["position_ned"]
                    print(f"      gate {g['gate_id']}: ned=({p[0]:+.1f},{p[1]:+.1f},{p[2]:+.1f}) "
                          f"w={g['width_m']:.2f} h={g['height_m']:.2f}")
            time.sleep(0.01)
        print(f"    mission: {mission_summary(client)}")

    try:
        observe("passive (pre-arm)")
        if args.arm and client.track_gates is None:
            print("\n[arm] arming to see if the map is sent once armed ...")
            client.arm()
            client.wait_armed(True, timeout_s=5.0)
            observe("armed")
        if args.reset and client.track_gates is None:
            print("\n[reset] sending MAV_CMD 31000 sim reset (re-broadcast while we listen) ...")
            client.send_sim_reset()
            observe("after reset")
    finally:
        if client.state.armed:
            print("\n[safety] force-disarming")
            client.disarm(force=True)
            client.wait_armed(False, timeout_s=3.0)

    # -- diagnosis -----------------------------------------------------------
    print("\n==== track_probe summary ====")
    print(f"  DATA_TRANSMISSION_HANDSHAKE seen: {diag['handshakes']}")
    enc = dict(diag["encap"])
    print(f"  ENCAPSULATED_DATA by data[0]:     {enc}   (1=RACE_STATUS, 2=TRACK_INFO)")
    print(f"  TRACK_INFO chunks seen:           {diag['track_chunks']}")
    print(f"  RACE_STATUS captured:             {client.race_status is not None}"
          + (f"  (started={client.race_status['started']})" if client.race_status else ""))
    if client.track_gates is not None:
        print(f"  GATE MAP: {len(client.track_gates)} gates  (captured in phase: {captured_phase})")
    else:
        print("  GATE MAP: NOT captured. Likely cause:")
        if diag["handshakes"] == 0 and enc.get(2, 0) == 0:
            print("   * No handshake AND no track chunks in any phase -> the sim isn't broadcasting the")
            print("     map here. It's gated on a trigger we haven't hit: try --reset, START THE RACE in")
            print("     the sim UI (watch RACE_STATUS.started), or it was a one-shot at connect we missed")
            print("     (wait_heartbeat may have eaten it) -> --reset re-broadcasts while we're listening.")
        elif enc.get(2, 0) > 0 and diag["handshakes"] == 0:
            print("   * Track chunks arrived but NO handshake -> our parser drops chunks without one.")
            print("     The sim may not send DATA_TRANSMISSION_HANDSHAKE; we'd parse chunks directly off")
            print("     ENCAPSULATED_DATA(data[0]==2). Send me the chunk lines above and I'll adapt the parser.")
        else:
            print("   * Handshake + chunks seen but never completed -> UDP loss or a seqnr/transfer_id")
            print("     mismatch. Compare the HANDSHAKE transfer_id(width) with the TRACK chunk transfer_id")
            print("     above; if they differ, that's the bug. Paste the lines and I'll fix the parser.")
    print("  NOTE: the delivery MECHANISM is from the official PyAIPilotExample; this probe verifies the")
    print("  TIMING/TRIGGER live (which we never observed before).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
