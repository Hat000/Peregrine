"""First-contact probe: arm/disarm lifecycle + clock + event characterisation.

Targets the SPEC MAVLink interface (VADR-TS-002 sec 4): arms via COMMAND_LONG /
MAV_CMD_COMPONENT_ARM_DISARM and reads HEARTBEAT / COMMAND_ACK / STATUSTEXT. It does NOT
touch Elodin (a separate Apache-2.0 surrogate with an RC-callback API, NOT the scoring sim).

Run it the moment the sim is up to learn: which backend we're talking to, whether we can
arm (and how -- normal vs --force), how long arming takes, what the armed flag + any
STATUSTEXT narrate about the session, and where the master clock comes from.

SAFE: it arms then disarms; it sends NO flight setpoints (that's control_mode_probe.py).

Usage:
  python scripts/session_lifecycle.py [--endpoint udp:127.0.0.1:14550]
        [--audit-s 3] [--hold-s 3] [--force] [--no-arm] [--connect-timeout 15]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pymavlink import mavutil

from racer.firstcontact import (
    MessageRateTracker,
    backend_summary,
    drain_statustexts,
    mission_summary,
    rate_warnings,
    sample_attitude_bias,
    telemetry_summary,
)
from racer.mavlink_client import MavlinkClient

_ARM = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--audit-s", type=float, default=3.0, help="passive observation before arming")
    ap.add_argument("--hold-s", type=float, default=3.0, help="observation while armed")
    ap.add_argument("--force", action="store_true", help="use force-arm (bypass prearm checks)")
    ap.add_argument("--no-arm", action="store_true", help="audit only; do not arm")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    client = MavlinkClient(args.endpoint)
    type_counts: Counter = Counter()
    rates = MessageRateTracker()

    def _tap(msg) -> None:
        t = msg.get_type()
        type_counts.update([t])
        rates.record(t, time.monotonic_ns())

    client.on_message = _tap

    print(f"connecting MAVLink {args.endpoint} (waiting for heartbeat) ...")
    try:
        client.connect(timeout_s=args.connect_timeout)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("heartbeat OK.\n")

    def observe(seconds: float, dt: float = 0.02) -> None:
        seen = len(client.statustexts)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            client.pump()
            seen = drain_statustexts(client, seen)
            time.sleep(dt)

    # -- passive audit --------------------------------------------------------
    print(f"[audit] observing {args.audit_s:g}s before any command ...")
    observe(args.audit_s)
    print(f"  backend:   {backend_summary(client)}")
    print(f"  telemetry: {telemetry_summary(client)}")
    sim_t0 = client.state.sim_time_ns
    if client.state.position_ned is not None:
        print("  R1: position IS in telemetry (LOCAL_POSITION_NED present) -> localisation collapses.")
    else:
        print("  R1: no position telemetry -> vision is the sole position source (as expected).")
    print(f"  mission:   {mission_summary(client)}")
    if client.track_gates:
        print(f"  R4: sim PROVIDES a {len(client.track_gates)}-gate map (TRACK_INFO) -> order + "
              "positions given; refine with vision (FAQ warns VQ2 may be 'rough').")
    else:
        print("  R4: no gate map yet (TRACK_INFO not seen this window; may need the race to start).")

    # Attitude-bias check (red-team CRIT-2 / the ESKF decision): at rest the GIVEN attitude
    # should match the gravity-implied tilt; a persistent residual = a static bias the linear
    # KF can't see and the ESKF bias-state would. Keep the vehicle STILL on the ground for this.
    print("  [attitude-bias] sampling given-vs-gravity tilt (keep it still, pre-arm) ...")
    bias = sample_attitude_bias(client, seconds=2.0)
    if bias is None:
        print("    not quasi-static (|accel| far from 1 g) -> can't infer; ensure it's at rest.")
    else:
        worst = max(abs(bias["roll_bias_deg"]), abs(bias["pitch_bias_deg"]))
        print(f"    roll {bias['roll_bias_deg']:+.2f}+/-{bias['roll_bias_std_deg']:.2f} deg, "
              f"pitch {bias['pitch_bias_deg']:+.2f}+/-{bias['pitch_bias_std_deg']:.2f} deg "
              f"[n={bias['n']}/{bias['n_total']}]")
        print(f"    -> {worst:.1f} deg max bias: "
              + ("given attitude NOT clean; weigh the ESKF bias-state." if worst > 0.5
                 else "given attitude looks trustworthy at rest (linear KF OK)."))

    armed_ok: bool | None = None
    arm_latency: float | None = None
    try:
        if not args.no_arm:
            # -- arm ----------------------------------------------------------
            print(f"\n[arm] sending {'FORCE ' if args.force else ''}arm ...")
            client.last_command_ack = None
            t_send = time.monotonic()
            client.arm(force=args.force)
            ack = client.wait_command_ack(_ARM, timeout_s=3.0)
            if ack is not None:
                print(f"  COMMAND_ACK: {ack['result_name']} ({ack['result']})")
            else:
                print("  no COMMAND_ACK within 3s (sim may not ack; relying on the armed flag)")
            armed_ok = client.wait_armed(True, timeout_s=5.0)
            arm_latency = time.monotonic() - t_send
            if armed_ok:
                print(f"  ARMED confirmed via HEARTBEAT in {arm_latency * 1000:.0f} ms")
            else:
                print("  armed flag did NOT set within 5s -- arming likely refused")
                if not args.force:
                    print("  hint: retry with --force, or read the COMMAND_ACK / STATUSTEXT above")

            # -- observe while armed -----------------------------------------
            print(f"\n[hold] observing {args.hold_s:g}s while armed "
                  "(does it move/climb on its own? what events fire?) ...")
            observe(args.hold_s, dt=0.05)
            print(f"  telemetry now: {telemetry_summary(client)}")

            # -- disarm -------------------------------------------------------
            print("\n[disarm] sending disarm ...")
            client.last_command_ack = None
            client.disarm(force=args.force)
            ack = client.wait_command_ack(_ARM, timeout_s=3.0)
            if ack is not None:
                print(f"  COMMAND_ACK: {ack['result_name']}")
            print(f"  disarm confirmed: {client.wait_armed(False, timeout_s=5.0)}")
    finally:
        if client.state.armed:  # never leave it armed
            print("\n[safety] still armed -> forcing disarm")
            client.disarm(force=True)
            client.wait_armed(False, timeout_s=3.0)

    # -- summary --------------------------------------------------------------
    sim_t1 = client.state.sim_time_ns
    print("\n==== session_lifecycle summary ====")
    print(f"  backend:      {backend_summary(client)}")
    if armed_ok is None:
        arm_str = "not attempted"
    elif armed_ok:
        arm_str = f"ARMED in {arm_latency * 1000:.0f} ms" + (" (force)" if args.force else "")
    else:
        arm_str = "REFUSED (see ack/statustext; try --force)"
    print(f"  arming:       {arm_str}")
    print(f"  master clock: sim_time_ns (HIGHRES_IMU) advanced {(sim_t1 - sim_t0) / 1e9:.3f}s, "
          f"{'monotonic' if sim_t1 >= sim_t0 else 'NON-MONOTONIC!'}")
    print(f"  TIMESYNC:     {'present' if type_counts.get('TIMESYNC') else 'NOT seen'}")
    if bias is not None:
        print(f"  attitude-bias: roll {bias['roll_bias_deg']:+.2f} / pitch {bias['pitch_bias_deg']:+.2f} deg "
              "at rest (>~0.5 deg -> consider ESKF bias-state)")
    print(f"  mission:      {mission_summary(client)}")
    print("  msg rates (count, Hz):")
    print(rates.report())
    for w in rate_warnings(rates):
        print(f"    ! {w}")
    n = len(client.statustexts)
    print(f"  STATUSTEXT:   {n} message(s)" + (" -- lifecycle clues, read above" if n else ""))
    print("  NOTE: Elodin is NOT the source of truth; this characterised the spec MAVLink sim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
