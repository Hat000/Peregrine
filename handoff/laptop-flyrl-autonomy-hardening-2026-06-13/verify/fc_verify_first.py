"""F-C VERIFY-FIRST — does the degraded-input failure actually PROPAGATE to the wire?

Run BEFORE any behaviour change, against the *current unedited* fly_rl.py + the real
shipped DroneState/MavlinkClient parser.  Mirrors the autonomy-audit scratch probes
(d1_odo_drop_real_parser / num2_quat_degenerate / num1_nan_propagation) but folded into
one self-contained script so the WRITEUP can cite reproducible evidence.

For each case we exercise the REAL functions and report whether the bad input reaches the
wire (build_obs raises mid-loop, OR a NaN/inf body_rate/thrust is produced).  If any case
is already sanitised, the F-C escape hatch applies and that guard can be dropped.

    PYTHONPATH=<worktree>/src python handoff/.../verify/fc_verify_first.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))
sys.path.insert(0, str(_ROOT / "scripts"))

import fly_rl  # noqa: E402
from racer.contracts import DroneState  # noqa: E402
from racer.mavlink_client import MavlinkClient  # noqa: E402


# ---------------------------------------------------------------------------
# Fake MAVLink messages — drive the REAL MavlinkClient._handle offline.
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, mtype: str, **kw):
        self._t = mtype
        self.__dict__.update(kw)

    def get_type(self) -> str:
        return self._t


def _odometry(q_wxyz, pos, vel, w, reset_counter=0) -> _Msg:
    return _Msg("ODOMETRY", q=list(q_wxyz),
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                rollspeed=w[0], pitchspeed=w[1], yawspeed=w[2],
                reset_counter=reset_counter)


def _lpn(pos, vel) -> _Msg:
    return _Msg("LOCAL_POSITION_NED",
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2])


def _nominal_state(**over) -> DroneState:
    """A healthy level-hover-ish state with everything finite + non-None."""
    base = dict(
        sim_time_ns=1_000_000_000, recv_monotonic_ns=1,
        orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body=np.array([0.1, 0.0, -0.1]),
        position_ned=np.array([-5.0, 0.5, -1.0]),
        velocity_ned=np.array([6.0, 0.0, 0.0]),
        accel_body=np.zeros(3),
    )
    base.update(over)
    return DroneState(**base)


def _try_build_and_step(actor, s, label):
    """Return (raised, rate_frd, collective, normed) for a state."""
    try:
        obs = fly_rl.build_obs(s, target_gate=0, last_normed_thrust=0.0,
                               virtual_flip=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] build_obs RAISED: {type(exc).__name__}: {exc}")
        return True, None, None, None
    rate_frd, collective, normed = fly_rl.policy_step(actor, obs, virtual_flip=True)
    finite = bool(np.all(np.isfinite(rate_frd)) and np.isfinite(collective))
    print(f"  [{label}] build_obs OK -> wire rate={np.round(rate_frd,4).tolist()} "
          f"thrust={collective:.4f}  all-finite-on-wire={finite}")
    return False, rate_frd, collective, normed


def main() -> int:
    ckpt = _ROOT / "rl" / "checkpoints" / "stage1_inc7_actor.pth"
    actor = fly_rl.load_actor(str(ckpt))
    print(f"\nloaded inc7 actor: {ckpt.name}\n")

    # ---- baseline sanity: healthy state flies clean ----------------------
    print("== BASELINE (healthy state) ==")
    _try_build_and_step(actor, _nominal_state(), "healthy")

    # ---- D1: ODOMETRY-only drop through the REAL parser ------------------
    print("\n== D1: frozen ODOMETRY while LPN keeps refreshing (REAL _handle) ==")
    c = MavlinkClient("udp:127.0.0.1:14550")
    # one ODOMETRY at a 35-deg bank, then the drone keeps rolling but ODOMETRY stops;
    # LPN keeps delivering fresh pos/vel.
    q_bank = [np.cos(np.radians(17.5)), np.sin(np.radians(17.5)), 0.0, 0.0]  # ~35 deg roll
    c._handle(_odometry(q_bank, [-5.0, 0.0, -1.0], [6.0, 0.0, 0.0], [4.0, 0.0, 0.0]))
    shared0 = c.state.recv_monotonic_ns
    q_frozen = np.asarray(c.state.orientation_ned_wxyz).copy()
    w_frozen = np.asarray(c.state.angular_rate_body).copy()
    # now 30 LPN updates arrive (true attitude would have rolled to ~310 deg meanwhile)
    for k in range(30):
        c._handle(_lpn([-5.0 + 0.2 * k, 0.0, -1.0], [6.0, 0.0, 0.0]))
    shared1 = c.state.recv_monotonic_ns
    q_after = np.asarray(c.state.orientation_ned_wxyz)
    w_after = np.asarray(c.state.angular_rate_body)
    quat_frozen = np.allclose(q_after, q_frozen) and np.allclose(w_after, w_frozen)
    stamp_moved = shared1 != shared0
    print(f"  shared recv_monotonic_ns advanced during ODOMETRY outage: {stamp_moved} "
          f"(delta={shared1 - shared0} ns)")
    print(f"  ODOMETRY quat+rate frozen verbatim across the outage: {quat_frozen}")
    print(f"  -> any age computed from the SHARED stamp reads ~0 ms while attitude is stale.")
    raised, *_ = _try_build_and_step(actor, c.state, "D1-stale-attitude")
    d1_propagates = (not raised) and quat_frozen and stamp_moved
    print(f"  D1 PROPAGATES (stale attitude -> wire, invisible to existing guards): {d1_propagates}")

    # ---- R4a: zero-norm ODOMETRY quat ------------------------------------
    print("\n== R4a: zero-norm quat -> build_obs ==")
    r4a, *_ = _try_build_and_step(actor, _nominal_state(
        orientation_ned_wxyz=np.zeros(4)), "R4a-zero-norm")

    # ---- R4b: NaN-component quat -----------------------------------------
    print("\n== R4b: NaN-component quat -> build_obs ==")
    r4b, *_ = _try_build_and_step(actor, _nominal_state(
        orientation_ned_wxyz=np.array([1.0, np.nan, 0.0, 0.0])), "R4b-nan-quat")

    # ---- R5a: NaN position (non-raising path) ----------------------------
    print("\n== R5a: NaN position -> build_obs -> policy_step -> wire ==")
    raised5a, rate5a, coll5a, _ = _try_build_and_step(actor, _nominal_state(
        position_ned=np.array([np.nan, 0.0, -1.0])), "R5a-nan-pos")
    nan_on_wire = (not raised5a) and (
        not np.all(np.isfinite(rate5a)) or not np.isfinite(coll5a))

    # ---- R5b: inf quat (REPORT: flows through without raising) ------------
    print("\n== R5b: inf-component quat -> build_obs -> wire ==")
    raised5b, rate5b, coll5b, _ = _try_build_and_step(actor, _nominal_state(
        orientation_ned_wxyz=np.array([1.0, np.inf, 0.0, 0.0])), "R5b-inf-quat")
    inf_on_wire = (not raised5b) and (
        not np.all(np.isfinite(rate5b)) or not np.isfinite(coll5b))

    # ---- demonstrate np.clip does NOT tame NaN/inf -----------------------
    print("\n== clip inertness ==")
    print(f"  np.clip(nan, -3.14, 3.14) = {np.clip(np.nan, -3.14, 3.14)}")
    print(f"  np.clip(inf, -3.14, 3.14) = {np.clip(np.inf, -3.14, 3.14)}")

    # ---- verdict ---------------------------------------------------------
    print("\n================ VERDICT ================")
    print(f"  D1 (frozen attitude reaches wire, undetected): {d1_propagates}")
    print(f"  R4a (zero-norm quat raises in build_obs):      {r4a}")
    print(f"  R4b (NaN quat raises in build_obs):            {r4b}")
    print(f"  R5a (NaN position -> NaN on wire):             {nan_on_wire}")
    print(f"  R5b (inf quat -> non-finite on wire):          {inf_on_wire}")
    any_prop = d1_propagates or r4a or r4b or nan_on_wire or inf_on_wire
    print(f"\n  ANY failure propagates on current code (F-C needed): {any_prop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
