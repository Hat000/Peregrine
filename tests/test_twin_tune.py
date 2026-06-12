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
from racer.twin_fit import faithful_config
from twin_fly_course import _FAITHFUL_SIGNS, fly, gate_plane_miss, make_controller
from twin_tune import (FAITHFUL_TUNED_GAINS, FAITHFUL_TUNED_PLANNER, PASS_BAR_M, TUNED_GAINS,
                       TUNED_PLANNER)


def _fly_faithful(gains, planner_params):
    """Fly the 6-gate course on the SIM-FAITHFUL plant with restored live signs at the live 100 Hz."""
    return fly(6, velocity_mode="clean", max_s=44.0, dt=0.01, flythrough_s=2.0,
               controller=make_controller(signs=_FAITHFUL_SIGNS, **gains),
               planner=ReactivePlanner(yaw_mode="course", **planner_params),
               plant_config=faithful_config())


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


def test_faithful_tuned_gains_thread_the_faithful_plant():
    # Task C: the live-ready config (restored live sim-signs + faithful-re-tuned outer gains) threads
    # all 6 gates on the SIM-FAITHFUL plant at the live 100 Hz. Worst ~0.61 m when tuned (pre
    # vision-frame-fix, aliased-R_wb estimation); ~0.20 m with the corrected true-attitude pairing
    # (2026-06-12). Keep the 0.75 m validity bar -- this pins the LIVE config's plant transfer, not
    # its centring.
    r = _fly_faithful(FAITHFUL_TUNED_GAINS, FAITHFUL_TUNED_PLANNER)
    miss = r["plane_miss"]
    assert r["final"].name == "FINISHED" and r["gate_index"] == 6
    assert all(m is not None for m in miss)                # every plane crossed
    assert max(miss) < 0.75                                # threads the inner opening (valid passes)


def test_canonical_gains_transfer_to_the_faithful_plant_with_true_attitude():
    # SUPERSEDES test_canonical_gains_do_not_transfer_to_the_faithful_plant (2026-06-12,
    # vision-frame-fix 8d7b0b3 + wire-convention twin emit): the
    # canonical gains' historical non-transfer (miss/stall on the faithful plant, the Task-C
    # rationale) was largely an ESTIMATION artifact, not plant dynamics. The old navigator built
    # R_wb from the faithful twin's report-sign euler (roll-mirror alias), mis-rotating the IMU
    # predict and corrupting the KF between given-position updates -- canonical gains were the
    # casualty. With the wire contract honored end-to-end (twin emits the raw R_y(pi)-conjugated
    # quat; navigator un-conjugates to the TRUE attitude), the canonical gains thread the faithful
    # plant dead-centre (worst in-plane miss ~0.07 m vs ~0.20 m for the faithful-tuned set).
    # This test now pins the corrected twin<->navigator convention seam ON THE FAITHFUL PLANT: a
    # reintroduced quat-convention mismatch degrades this flight well past the 0.5 m bar (it read
    # ~0.86 m on the canonical course while the twin emitted TRUE quats). NOTE: FAITHFUL_TUNED_GAINS
    # remain the LIVE config (VQ1-proven on the real sim; the twin under-models latency -- live gain
    # changes are live decisions, not twin conclusions).
    r = _fly_faithful(TUNED_GAINS, TUNED_PLANNER)
    assert r["final"].name == "FINISHED" and r["gate_index"] == 6
    assert all(m is not None for m in r["plane_miss"])     # every plane crossed
    assert max(r["plane_miss"]) < 0.5                      # dead-centre only with true attitude pairing
