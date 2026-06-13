"""F-C VERIFY-AFTER — does the fix CATCH the failures, and stay SILENT on a healthy run?

Run AFTER the F-C edits. Three things to confirm:
  1. PARSER FIX: odo_recv_ns is set ONLY by ODOMETRY, so during a selective ODOMETRY drop it
     freezes while the shared recv_monotonic_ns keeps advancing -> the D1 staleness is now
     observable (it was not before).
  2. GATE LOGIC: telemetry_health() classifies healthy/stale/NaN/zero-norm/inf correctly and
     never feeds a bad snapshot to the wire.
  3. NO FALSE-TRIP: benign ODOMETRY jitter (a few dropped 75 Hz frames) stays "ok"; only a
     genuine multi-hundred-ms stall flips to "stale".

    PYTHONPATH=<worktree>/src python handoff/.../verify/fc_verify_after.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))
sys.path.insert(0, str(_ROOT / "scripts"))

import fly_rl  # noqa: E402
from racer.contracts import DroneState  # noqa: E402
from racer.mavlink_client import MavlinkClient  # noqa: E402


class _Msg:
    def __init__(self, mtype, **kw):
        self._t = mtype
        self.__dict__.update(kw)

    def get_type(self):
        return self._t


def _odometry(q, pos, vel, w, rc=0):
    return _Msg("ODOMETRY", q=list(q), x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                rollspeed=w[0], pitchspeed=w[1], yawspeed=w[2], reset_counter=rc)


def _lpn(pos, vel):
    return _Msg("LOCAL_POSITION_NED", x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2])


def _state(**over):
    base = dict(sim_time_ns=1_000_000_000, recv_monotonic_ns=1, odo_recv_ns=1,
                orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_rate_body=np.array([0.1, 0.0, -0.1]),
                position_ned=np.array([-5.0, 0.5, -1.0]),
                velocity_ned=np.array([6.0, 0.0, 0.0]))
    base.update(over)
    return DroneState(**base)


STALE_S = 0.15
ok = True


def check(label, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


def main():
    import time

    print("== 1. PARSER FIX: odo_recv_ns isolates ODOMETRY freshness ==")
    c = MavlinkClient("udp:127.0.0.1:14550")
    c._handle(_odometry([1, 0, 0, 0], [-5, 0, -1], [6, 0, 0], [0, 0, 0]))
    odo0, shared0 = int(c.state.odo_recv_ns), int(c.state.recv_monotonic_ns)
    q_frozen = np.asarray(c.state.orientation_ned_wxyz).copy()
    for k in range(30):
        c._handle(_lpn([-5 + 0.2 * k, 0, -1], [6, 0, 0]))   # LPN only; ODOMETRY dropped
    odo1, shared1 = int(c.state.odo_recv_ns), int(c.state.recv_monotonic_ns)
    check("odo_recv_ns FROZEN during ODOMETRY outage", odo1 == odo0)
    check("shared recv_monotonic_ns ADVANCED (LPN bumps it)", shared1 > shared0)
    check("attitude quat frozen verbatim (the D1 staleness)",
          np.allclose(np.asarray(c.state.orientation_ned_wxyz), q_frozen))
    # the gate now sees it as stale once enough wall-time passes
    now_ns = c.state.odo_recv_ns + int(0.2e9)   # 200 ms after the last ODOMETRY
    health, age = fly_rl.telemetry_health(c.state, now_ns, STALE_S)
    check(f"telemetry_health flags it 'stale' at 200 ms age (was invisible before): {health}",
          health == "stale" and age > STALE_S)

    print("\n== 2. GATE LOGIC: classify each degraded case ==")
    base_now = _state().odo_recv_ns + int(0.01e9)   # 10 ms age -> fresh
    h, _ = fly_rl.telemetry_health(_state(), base_now, STALE_S)
    check(f"healthy state -> 'ok' ({h})", h == "ok")
    h, _ = fly_rl.telemetry_health(_state(orientation_ned_wxyz=np.zeros(4)), base_now, STALE_S)
    check(f"zero-norm quat -> 'non_finite' ({h})  [R4a: build_obs no longer reached]",
          h == "non_finite")
    h, _ = fly_rl.telemetry_health(
        _state(orientation_ned_wxyz=np.array([1.0, np.nan, 0.0, 0.0])), base_now, STALE_S)
    check(f"NaN quat -> 'non_finite' ({h})  [R4b]", h == "non_finite")
    h, _ = fly_rl.telemetry_health(
        _state(position_ned=np.array([np.nan, 0.0, -1.0])), base_now, STALE_S)
    check(f"NaN position -> 'non_finite' ({h})  [R5a]", h == "non_finite")
    h, _ = fly_rl.telemetry_health(
        _state(orientation_ned_wxyz=np.array([1.0, np.inf, 0.0, 0.0])), base_now, STALE_S)
    check(f"inf quat -> 'non_finite' ({h})  [R5b]", h == "non_finite")
    h, _ = fly_rl.telemetry_health(_state(velocity_ned=np.array([np.inf, 0, 0])), base_now, STALE_S)
    check(f"inf velocity -> 'non_finite' ({h})", h == "non_finite")
    h, _ = fly_rl.telemetry_health(_state(angular_rate_body=np.array([np.nan, 0, 0])), base_now, STALE_S)
    check(f"NaN rate -> 'non_finite' ({h})", h == "non_finite")
    h, _ = fly_rl.telemetry_health(DroneState(), base_now, STALE_S)
    check(f"pre-first-fix (all None) -> 'no_fix' ({h})  [does NOT arm recovery timer]",
          h == "no_fix")

    # a confirmed-bad snapshot must never produce a finite wire command via build_obs:
    # the gate returns non_finite BEFORE build_obs is ever called -> the R4 ValueError path
    # is now unreachable. (We assert telemetry_health gates it; the loop hovers instead.)

    print("\n== 3. NO FALSE-TRIP: benign 75 Hz jitter stays 'ok' ==")
    odo_period_ms = 1000.0 / 75.0
    for dropped in range(0, 12):
        age_ms = odo_period_ms * (dropped + 1)   # 'dropped' missed frames + one control tick
        s = _state()
        now_ns = int(s.odo_recv_ns) + int(age_ms * 1e6)
        h, _ = fly_rl.telemetry_health(s, now_ns, STALE_S)
        tag = "ok" if h == "ok" else "STALE"
        flag = "(benign)" if age_ms <= 60 else ""
        print(f"    {dropped:2d} dropped frames  age={age_ms:6.1f} ms -> {tag:5s} {flag}")
    # boundary: just under vs just over 150 ms
    s = _state()
    h_lo, _ = fly_rl.telemetry_health(s, int(s.odo_recv_ns) + int(0.149e9), STALE_S)
    h_hi, _ = fly_rl.telemetry_health(s, int(s.odo_recv_ns) + int(0.151e9), STALE_S)
    check("149 ms age -> 'ok' (no false-trip just under threshold)", h_lo == "ok")
    check("151 ms age -> 'stale' (catches just over threshold)", h_hi == "stale")
    check("benign single dropped frame (26.7 ms) -> 'ok'",
          fly_rl.telemetry_health(_state(),
                                  int(_state().odo_recv_ns) + int(0.0267e9), STALE_S)[0] == "ok")

    print("\n================ VERDICT ================")
    print(f"  ALL CHECKS PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
