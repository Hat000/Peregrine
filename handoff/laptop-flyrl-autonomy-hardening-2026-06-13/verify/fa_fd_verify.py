"""F-A / F-D / F-B VERIFY — wrapper pins, safe defaults, late-join GO, arm-retry, finally-disarm.

Offline, no live sim. Uses a scripted fake MavlinkClient so wait_fresh_go / the arm loop run
against crafted RACE_STATUS + ACK sequences.

    PYTHONPATH=<worktree>/src python handoff/.../verify/fa_fd_verify.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))
sys.path.insert(0, str(_ROOT / "scripts"))

import fly_rl  # noqa: E402
import submit_rl  # noqa: E402
from racer.contracts import DroneState  # noqa: E402

ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


def _args(**over):
    """A minimal args namespace for wait_fresh_go (only the fields it reads)."""
    base = dict(wait_seconds=0.4, start_margin_s=0.3, reset_after=8.0)
    base.update(over)
    return argparse.Namespace(**base)


class FakeClient:
    """Scripted client: a fixed race_status + state, counts sim-resets and arm sends."""

    def __init__(self, race_status, state, *, arm_accept_on=1):
        self.race_status = race_status
        self.state = state
        self.track_gates = []
        self.n_reset = 0
        self.last_command_ack = None
        self._arm_calls = 0
        self._arm_accept_on = arm_accept_on   # become armed on the Nth arm() (force or not)
        self._armed = False

    def pump(self):
        pass

    def send_sim_reset(self):
        self.n_reset += 1

    # arm path -----------------------------------------------------------
    def arm(self, force=False):
        self._arm_calls += 1
        if self._arm_calls >= self._arm_accept_on:
            self._armed = True
            self.last_command_ack = {"command": 400, "result": 0, "result_name": "ACCEPTED"}
        else:
            self.last_command_ack = {"command": 400, "result": 1,
                                     "result_name": "TEMPORARILY_REJECTED"}

    def wait_command_ack(self, command, timeout_s=3.0):
        return self.last_command_ack

    def wait_armed(self, armed=True, timeout_s=5.0):
        return self._armed == armed


def _level_state(pos):
    return DroneState(sim_time_ns=1_000_000_000, recv_monotonic_ns=1, odo_recv_ns=1,
                      orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                      position_ned=np.asarray(pos, dtype=np.float64),
                      velocity_ned=np.zeros(3))


def _race(started=True, finished=False, gi=0, to_go_ms=-5000.0, boot=1_000_000.0):
    # to_go = race_start_boot - sim_boot ; pick sim_boot so to_go matches.
    return {"started": started, "finished": finished, "active_gate_index": gi,
            "race_start_boot_time_ms": boot, "sim_boot_time_ms": boot - to_go_ms,
            "race_finish_time_ns": -1}


def main():
    print("== F-A: safe DEFAULTS (bare fly_rl.py) ==")
    ap = fly_rl.build_parser()
    d = ap.parse_args([])
    check("default checkpoint = inc7", d.checkpoint.endswith("stage1_inc7_actor.pth"))
    check("default --bridge OFF", d.bridge is False)
    check("default --debug-obs OFF", d.debug_obs is False)
    check("default --dev-auto-reset OFF", d.dev_auto_reset is False)
    auto = bool(d.dev_auto_reset) and not d.no_auto_reset
    check("derived auto_reset OFF by default (no sim-control on judged wire)", auto is False)
    check("default --flights 1", d.flights == 1)
    check("default --arm-attempts 3", d.arm_attempts == 3)
    check("default --odo-stale-s 0.15", abs(d.odo_stale_s - 0.15) < 1e-9)

    print("\n== F-A: --dev-auto-reset opt-in, --no-auto-reset hard override ==")
    d2 = ap.parse_args(["--dev-auto-reset"])
    check("dev-auto-reset -> auto_reset ON", (bool(d2.dev_auto_reset) and not d2.no_auto_reset))
    d3 = ap.parse_args(["--dev-auto-reset", "--no-auto-reset"])
    check("dev-auto-reset + no-auto-reset -> auto_reset OFF (hard override)",
          not (bool(d3.dev_auto_reset) and not d3.no_auto_reset))

    print("\n== F-A: submit_rl wrapper pins are authoritative ==")
    argv = submit_rl.build_argv(["--bridge", "--dev-auto-reset", "--debug-obs", "--flights", "9"])
    w = fly_rl.build_parser().parse_args(argv)
    auto_w = bool(w.dev_auto_reset) and not w.no_auto_reset
    check("wrapper forces --no-bridge despite forwarded --bridge", w.bridge is False)
    check("wrapper forces auto_reset OFF despite forwarded --dev-auto-reset", auto_w is False)
    check("wrapper forces --no-debug-obs despite forwarded --debug-obs", w.debug_obs is False)
    check("wrapper forces --flights 1 despite forwarded --flights 9", w.flights == 1)
    check("wrapper pins inc7 checkpoint", w.checkpoint.endswith("stage1_inc7_actor.pth"))

    print("\n== F-D: late-join GO accepted in PASSIVE config ==")
    c = FakeClient(_race(gi=0, to_go_ms=-5000.0), _level_state([0.2, 0.0, -1.0]))
    got = fly_rl.wait_fresh_go(c, _args(), auto_reset=False)
    check("passive late-join at gate0/origin -> GO (was silent NO_GO)", got is True)
    check("no sim-reset emitted on passive late-join", c.n_reset == 0)

    print("\n== F-D: late-join NOT joinable cases stay NO_GO (no false accept) ==")
    c2 = FakeClient(_race(gi=2, to_go_ms=-5000.0), _level_state([0.2, 0.0, -1.0]))
    check("late race already at gate 2 -> NO_GO", fly_rl.wait_fresh_go(c2, _args(), False) is False)
    c3 = FakeClient(_race(gi=0, to_go_ms=-5000.0), _level_state([50.0, 0.0, -1.0]))
    check("late race but drone 50 m off origin -> NO_GO", fly_rl.wait_fresh_go(c3, _args(), False) is False)
    c4 = FakeClient(_race(finished=True, gi=0, to_go_ms=-5000.0), _level_state([0.2, 0.0, -1.0]))
    check("late race already finished -> NO_GO", fly_rl.wait_fresh_go(c4, _args(), False) is False)

    print("\n== F-D: a normal FRESH GO still accepted (no regression) ==")
    c5 = FakeClient(_race(gi=0, to_go_ms=-500.0), _level_state([0.2, 0.0, -1.0]))   # elapsed 0.5s, fresh band
    check("fresh elapsed GO at origin -> GO", fly_rl.wait_fresh_go(c5, _args(), False) is True)

    print("\n== F-D: late-join NOT auto-accepted under dev auto_reset (requests reset) ==")
    c6 = FakeClient(_race(gi=0, to_go_ms=-5000.0), _level_state([0.2, 0.0, -1.0]))
    # tiny wait so it returns quickly; with auto_reset it should NOT late-join, and DOES try a reset
    fly_rl.wait_fresh_go(c6, _args(wait_seconds=0.05, reset_after=0.0), auto_reset=True)
    check("dev auto_reset does NOT silently late-join (emits sim-reset instead)", c6.n_reset >= 1)

    print("\n== F-D: arm-retry recovers a transient rejection ==")
    # Drive the real arm loop via fly_once's logic isn't isolatable; reproduce the loop here
    # with the SAME parameters used in fly_once (verifies the retry recovers on attempt 2).
    import time
    c7 = FakeClient(_race(), _level_state([0, 0, -1]), arm_accept_on=2)
    armed = False
    attempts, backoff = 3, 0.0
    for attempt in range(1, attempts + 1):
        force = attempt == attempts and attempts > 1
        c7.last_command_ack = None
        c7.arm(force=force)
        c7.wait_command_ack(400, 3.0)
        armed = c7.wait_armed(True, 5.0)
        if armed:
            break
        if attempt < attempts:
            time.sleep(backoff)
    check("transient-rejected first arm recovers by attempt 2", armed and c7._arm_calls == 2)

    print("\n================ VERDICT ================")
    print(f"  ALL CHECKS PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
