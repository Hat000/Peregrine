"""SYSTEM-ID REGISTRATION: ODOMETRY R_y(pi) quaternion conjugation (#3d).

PAIR: src/racer/frames.py (ODO_QUAT_TRUE_CONJ_WXYZ + true_attitude_from_odo_quat_wxyz
+ R_world_from_odo_quat_wxyz, consumed by scripts/frame_residual_report.py and the
vision/PnP path) registered against an INDEPENDENT canonical numpy R_y(pi) matrix.

WHAT IS REGISTERED
------------------
The sim's ODOMETRY attitude quaternion is NOT the FRD->NED attitude as-is; it is the
attitude expressed in an R_y(pi)-conjugated frame pair (180 deg rotation of BOTH the
world and body axes about Y). The code realizes the un-conjugation as an elementwise
wxyz scale ODO_QUAT_TRUE_CONJ_WXYZ = [1, -1, 1, -1] (negate x and z). This test PROVES
that scale is exactly the R_y(pi) frame-pair conjugation, i.e.

    R_true = Ry @ R_raw @ Ry,   Ry = R_y(pi) = diag(-1, +1, -1).

THE FOOTGUN (why an independent reference is mandatory)
------------------------------------------------------
Internal consistency cannot catch a conjugation: a proper frame conjugation passes
every internal check (quat-FD vs the rate channel, twist round-trip, level flight).
The EXISTING test (tests/test_frames.py::test_R_world_from_odo_quat_wxyz_gives_true_
rotation_at_bank) builds q_raw = q_true * SCALE and recovers it with the SAME SCALE --
a self-consistent round-trip. That round-trip PASSES even if the scale were silently
regressed to a DIFFERENT involutory axis mirror (e.g. R_x(pi) = [1,1,-1,-1]). This
test instead compares against an R_y(pi) rotation matrix BUILT INDEPENDENTLY from
Rotation.from_rotvec([0, pi, 0]) -- so a wrong-axis mirror is caught (negative control
below blows the residual to ~1.08, vs ~1e-16 for the correct R_y(pi)).

DOES NOT constrain the CTBR control path (euler_from_quat_wxyz on the RAW quat is a
self-consistent alias, VQ1-proven, intentionally left as-is). DOES NOT touch inc7
(obs_dim 17) -- pure frame math, no obs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# mirror the sys.path.insert pattern other tests/ files use for racer.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.frames import (  # noqa: E402
    ODO_QUAT_TRUE_CONJ_WXYZ,
    R_world_from_body,
    R_world_from_odo_quat_wxyz,
    true_attitude_from_odo_quat_wxyz,
)


def _R_from_wxyz(q):
    q = np.asarray(q, float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


# Independent canonical R_y(pi): 180 deg about Y, built from a rotvec -- NOT from the
# quaternion scale under test. Equals diag(-1, +1, -1).
_RY_PI = Rotation.from_rotvec([0.0, np.pi, 0.0]).as_matrix()


def test_conj_constant_pinned():
    """The elementwise wxyz scale is exactly [1, -1, 1, -1] (negate x and z)."""
    assert np.array_equal(ODO_QUAT_TRUE_CONJ_WXYZ, [1.0, -1.0, 1.0, -1.0])
    # canonical R_y(pi) sanity (independent of the scale)
    np.testing.assert_allclose(_RY_PI, np.diag([-1.0, 1.0, -1.0]), atol=1e-12)


def test_scale_equals_ry_pi_frame_pair_conjugation():
    """[1,-1,1,-1] quat scale == R_y(pi) frame-pair conjugation, vs INDEPENDENT matrix.

    For random TRUE attitudes: the raw ODOMETRY quat (q_true * scale) decoded to a
    rotation must satisfy R_true == Ry @ R_raw @ Ry, where Ry is the independently
    constructed R_y(pi). The code helper R_world_from_odo_quat_wxyz(q_raw) must also
    return R_true. Tight tolerance (1e-9) -- this is an exact algebraic identity.
    """
    rng = np.random.default_rng(20260618)
    worst = 0.0
    for _ in range(2000):
        roll = rng.uniform(-np.pi, np.pi)
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)  # unambiguous 3-2-1
        q_xyzw = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
        q_true = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        R_true = _R_from_wxyz(q_true)

        q_raw = true_attitude_from_odo_quat_wxyz(q_true)  # == q_true * scale (involutory)
        R_raw = _R_from_wxyz(q_raw)

        # registration: independent R_y(pi) matrix conjugation reconstructs R_true
        worst = max(worst, float(np.max(np.abs(R_true - _RY_PI @ R_raw @ _RY_PI))))
        # the production helper must agree with R_true on the raw quat
        worst = max(worst, float(np.max(np.abs(R_world_from_odo_quat_wxyz(q_raw) - R_true))))
    assert worst < 1e-9, f"R_y(pi) conjugation registration broke: max err {worst:.3e}"


def test_wrong_axis_mirror_is_caught_by_independent_reference():
    """NEGATIVE CONTROL: the independent R_y(pi) reference rejects a wrong-axis mirror.

    The existing self-consistent round-trip would PASS a regression to R_x(pi) =
    [1,1,-1,-1]; the independent matrix reference must FAIL it -- proving discriminating
    power that internal consistency lacks.
    """
    wrong_scale = np.array([1.0, 1.0, -1.0, -1.0])  # R_x(pi), a different involution
    roll, pitch, yaw = 0.6, -0.3, 0.9
    q_xyzw = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    q_true = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    R_true = _R_from_wxyz(q_true)

    # self-consistent round-trip with the WRONG scale still recovers R_true (no power)
    q_raw_wrong = q_true * wrong_scale
    R_roundtrip = _R_from_wxyz(q_raw_wrong * wrong_scale)
    np.testing.assert_allclose(R_roundtrip, R_true, atol=1e-12)

    # the independent R_y(pi) reference REJECTS the wrong-axis mirror
    resid = float(np.max(np.abs(R_true - _RY_PI @ _R_from_wxyz(q_raw_wrong) @ _RY_PI)))
    assert resid > 0.5, (
        f"independent R_y(pi) reference failed to catch a wrong-axis mirror "
        f"(resid {resid:.3e}) -- the test lacks discriminating power"
    )


def test_conjugation_is_involutory():
    """Applying the scale twice returns the input (frames helper)."""
    rng = np.random.default_rng(7)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    q2 = true_attitude_from_odo_quat_wxyz(true_attitude_from_odo_quat_wxyz(q))
    np.testing.assert_allclose(q2, q, atol=1e-15)


def test_euler_effect_roll_yaw_negated_pitch_intact():
    """R_y(pi) conjugation negates ROLL and YAW, leaves PITCH intact (3-2-1 ZYX).

    Verified by comparing the raw ODOMETRY rotation to R_world_from_body(-roll, +pitch,
    -yaw) -- a separate construction from Euler, independent of the quat scale.
    """
    roll, pitch, yaw = np.deg2rad(45.0), np.deg2rad(-10.0), np.deg2rad(30.0)
    q_xyzw = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    q_true = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    q_raw = true_attitude_from_odo_quat_wxyz(q_true)
    R_raw = _R_from_wxyz(q_raw)
    R_negrollyaw = R_world_from_body(-roll, pitch, -yaw)
    np.testing.assert_allclose(R_raw, R_negrollyaw, atol=1e-12)
