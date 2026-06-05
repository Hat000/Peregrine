"""Task A -- the gain-tuning harness (scripts/twin_tune.py) on the canonical twin.

Validates: (1) the in-plane opening-miss metric, (2) that g5's distance-to-centre is a
truncation ARTIFACT (mission.run stops at FINISHED ~gate_pass_radius_m before the last plane), and
(3) that the TUNED gains thread all 6 gates -- without re-running the (slow) search.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from racer.contracts import Gate
from racer.planner import ReactivePlanner
from twin_fly_course import fly, gate_plane_miss, make_controller
from twin_tune import PASS_BAR_M, TUNED_GAINS, TUNED_PLANNER


def test_gate_plane_miss_interpolates_the_opening_offset():
    # Gate frame = identity: width=col0 (x), height=col1 (y), through=col2 (z). A path that pierces
    # the z=0 plane at (0.3, 0.4) is sqrt(0.3^2+0.4^2)=0.5 m off-centre in the opening.
    gate = Gate(gate_id=0, position_ned=np.zeros(3), R_world_gate=np.eye(3))
    crossing = np.array([[0.3, 0.4, -1.0], [0.3, 0.4, 1.0]])
    assert gate_plane_miss(crossing, gate) == pytest.approx(0.5)
    # A path that never reaches the plane -> no crossing -> None (a genuine miss, not a 0).
    short = np.array([[0.3, 0.4, -1.0], [0.3, 0.4, -0.5]])
    assert gate_plane_miss(short, gate) is None


def test_final_gate_distance_to_centre_is_a_truncation_artifact():
    # Without a flythrough, mission.run truncates at FINISHED: the dead-centre final gate reads its
    # distance-to-centre ~= gate_pass_radius_m (0.75 m) and its plane is NEVER crossed...
    trunc = fly(6, velocity_mode="clean", max_s=46.0, dt=0.02, flythrough_s=0.0)
    assert trunc["final"].name == "FINISHED" and trunc["gate_index"] == 6
    assert trunc["closest"][5] > 0.5                       # inflated along the through-axis
    assert trunc["plane_miss"][5] is None                  # plane not crossed (truncated short)
    assert max(trunc["closest"][:5]) < 0.2                 # the fly-through gates are unaffected
    # ...but flying THROUGH it shows the drone was dead-centre all along (in-plane miss ~ 0).
    through = fly(6, velocity_mode="clean", max_s=46.0, dt=0.02, flythrough_s=2.0)
    assert through["plane_miss"][5] is not None and through["plane_miss"][5] < 0.1


def test_tuned_gains_thread_all_six_gates():
    # The committed TUNED_GAINS/TUNED_PLANNER (scripts/twin_tune.py) fly the real stack through all
    # 6 gates dead-centre on the clean canonical twin -- finish, every plane crossed, well inside the
    # opening. Validated at the fine dt (matches the committed report numbers).
    ctrl = make_controller(**TUNED_GAINS)
    plan = ReactivePlanner(yaw_mode="course", **TUNED_PLANNER)
    r = fly(6, velocity_mode="clean", max_s=46.0, dt=0.01, controller=ctrl, planner=plan,
            flythrough_s=2.0)
    assert r["final"].name == "FINISHED" and r["gate_index"] == 6
    miss = r["plane_miss"]
    assert all(m is not None for m in miss)                # every gate plane crossed
    assert max(miss) < 0.5                                 # the task's bar (all gates < 0.5 m)
    assert max(miss) < PASS_BAR_M                          # and the pre-registered target (< 0.40 m)
