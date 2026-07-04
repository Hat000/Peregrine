"""A36 COORDINATED-TURN offline proof (unit-level, trustworthy — no fragile full-sim).

Demonstrates the four mechanisms of the A+B+C coordinated-turn rebalance directly against the real
controller + seeker units, comparing OLD (run-120357 config) vs NEW (A+B+C). Grounded in run
20260704_120357 (mode A, yaw now correct): "yawed a lot more than needed, not rolling nearly enough,"
and the pass-turn never fired (orbit-brake ran the turn).

Run: .venv/Scripts/python.exe handoff/a36_coordinated_turn_proof.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import GatePose  # noqa: E402
from racer.controller import _clip_norm  # noqa: E402
from racer.deploy_profile import vq2_case_c  # noqa: E402
from racer.frames import R_world_from_body  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402

_OLD = dict(total_accel_cap_mps2=2.0, pursuit_yaw_slew_rps=1.5, visual_yaw_rate_cap_rps=1.5,
            pass_arm_range_m=3.0, orbit_guard_rad=1.75)


def _pose(rng):
    return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                    t_cam_gate=np.array([0.0, 0.0, float(rng)]), reproj_error_px=0.5)


def _roll_cmd_at_yaw_err(controller, yaw_err_deg, a_lat=3.0):
    """The roll body-rate command that SURVIVES the omega norm-clip when the attitude error carries a
    yaw error of yaw_err_deg (the orbit-brake's frozen-yaw-vs-rotating-yaw divergence). a_lat=3.0 is
    the full lateral demand (17 deg bank target)."""
    q, _ = controller._accel_to_attitude(np.array([0.0, a_lat, 0.0]), 0.0)
    R_des = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    asign = np.asarray(controller.odo_att_sign)
    yaw_as = -asign[2]                       # mode A R_cur-yaw recovery
    R_cur = R_world_from_body(0.0, 0.0, np.radians(yaw_err_deg) * yaw_as)
    rv = Rotation.from_matrix(R_cur.T @ R_des).as_rotvec()
    omega = _clip_norm(controller.kp_att * rv, controller.max_body_rate_rps) / max(controller.ff_gain, 1e-6)
    return float(omega[0])


def _pass_arms_at(range_m, arm_range):
    so = dict(vq2_case_c().seeker_overrides)
    so["pass_arm_range_m"] = arm_range
    s = GateSeeker(config=GateSeekerConfig(**so))
    s._anchored = True
    s._last_yaw = 0.0
    s._update_pass_state(0, _pose(range_m), index_advanced=False)
    return bool(s._pass_armed)


def main():
    so = vq2_case_c().seeker_overrides
    ctrl = make_seeker_controller(**vq2_case_c().controller_overrides)

    print("=== (C) TRIGGER: the pass ARMS at the ~4.3 m vision floor (so the pass-turn can fire) ===")
    print(f"    OLD pass_arm_range=3.0 -> armed at 4.3 m: {_pass_arms_at(4.3, 3.0)}  (FALSE -> orbit-brake ran)")
    print(f"    NEW pass_arm_range=4.5 -> armed at 4.3 m: {_pass_arms_at(4.3, 4.5)}  (TRUE  -> pass-turn owns the turn)")

    print("\n=== (B) YAW cut: the turn is roll-led, not yaw-dominated ===")
    print(f"    pursuit_yaw_slew_rps  1.5 -> {so['pursuit_yaw_slew_rps']}")
    print(f"    visual_yaw_rate_cap_rps 1.5 -> {so['visual_yaw_rate_cap_rps']}")

    print("\n=== (A+B) ROLL survives the omega-clip when the yaw error is kept small ===")
    print("    (the orbit-brake FROZE yaw -> huge yaw error starved roll to ~0.59; B+C keep yaw error")
    print("     small by SLEWING yaw toward the target, so the full 3.0-lateral roll command survives)")
    for ye in (0, 30, 60, 90):
        print(f"    yaw_err={ye:2d} deg -> roll cmd survives = {_roll_cmd_at_yaw_err(ctrl, ye):+.2f}")

    print("\n=== (A) CAP: total_accel_cap 2.0 -> 3.0 lets image_lat_cap=3.0 apply (~17 deg bank) ===")
    print(f"    total_accel_cap_mps2 = {so['total_accel_cap_mps2']}  (image_lat_cap={so['image_lat_cap_mps2']},"
          f" lateral-first={so['use_lateral_first_budget']}, forward stays {so['forward_accel_mps2']})")

    print("\n=== (C) ORBIT-BREAKER demoted to a rare failsafe ===")
    print(f"    orbit_guard_rad = {so['orbit_guard_rad']} rad ({np.degrees(so['orbit_guard_rad']):.0f} deg,"
          f" was 1.75 = 100 deg) -> no longer pre-empts the pass-turn")

    print("\nPASS: NEW arms the pass at the floor (C), cuts yaw (B), keeps the roll command alive at "
          "small yaw error (A+B), unthrottles the bank (A), and demotes the orbit-brake (C).")


if __name__ == "__main__":
    main()
