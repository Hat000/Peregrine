"""Tests for the SOT(3) inverse-depth landmark parameterization (eqvio_landmark.py).

Run:
    C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe -m pytest tests/test_eqvio_landmark.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from racer.ahrs.iekf import _Exp_so3, _skew
from racer.ahrs.eqvio import _make_X, _Exp_se23
from racer.ahrs.eqvio_landmark import (
    SOT3, E_Z, project_unit, reconstruct_world_point, bearing_measurement,
    bearing_residual, d_project_d_v, bearing_jacobian_pose, bearing_jacobian_landmark,
    triangulate_inverse_depth,
)


class TestSOT3Group:

    def test_bearing_is_unit(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            s = SOT3(Q=_Exp_so3(rng.normal(0, 1, 3)), rho=abs(rng.normal(1, 0.5)) + 0.1)
            np.testing.assert_allclose(np.linalg.norm(s.bearing()), 1.0, atol=1e-12)

    def test_ray_is_inverse_depth_scaled(self):
        s = SOT3(Q=np.eye(3), rho=0.25)   # depth = 4
        np.testing.assert_allclose(s.ray(), [0, 0, 4.0], atol=1e-12)

    def test_retract_keeps_rho_positive(self):
        s = SOT3(Q=np.eye(3), rho=0.5)
        for sv in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            s2 = s.retract(np.array([0, 0, 0, sv]))
            assert s2.rho > 0.0
            np.testing.assert_allclose(s2.rho, 0.5 * np.exp(sv), atol=1e-12)

    def test_retract_local_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            base = SOT3(Q=_Exp_so3(rng.normal(0, 0.5, 3)), rho=abs(rng.normal(1, 0.3)) + 0.2)
            xi = np.concatenate([rng.normal(0, 0.2, 3), rng.normal(0, 0.3, 1)])
            other = base.retract(xi)
            xi_rec = base.local(other)
            np.testing.assert_allclose(xi_rec, xi, atol=1e-9)


class TestBearingModel:

    def test_project_unit(self):
        np.testing.assert_allclose(project_unit(np.array([3.0, 0, 0])), [1, 0, 0], atol=1e-12)

    def test_d_project_finite_difference(self):
        rng = np.random.default_rng(2)
        for _ in range(10):
            v = rng.normal(0, 2, 3)
            J = d_project_d_v(v)
            Jn = np.zeros((3, 3)); eps = 1e-6
            for k in range(3):
                e = np.zeros(3); e[k] = eps
                Jn[:, k] = (project_unit(v + e) - project_unit(v - e)) / (2 * eps)
            np.testing.assert_allclose(J, Jn, atol=1e-6)

    def test_reconstruct_then_measure_is_identity(self):
        """Reconstruct a world point from a SOT(3) bearing, then re-measure it: the bearing
        must come back unchanged (model self-consistency)."""
        rng = np.random.default_rng(3)
        R_wb = _Exp_so3(rng.normal(0, 1, 3)); p_wb = rng.normal(0, 3, 3)
        sot = SOT3(Q=_Exp_so3(rng.normal(0, 0.5, 3)), rho=0.3)
        p_L = reconstruct_world_point(sot, R_wb, p_wb)
        b = bearing_measurement(p_L, R_wb, p_wb)
        np.testing.assert_allclose(b, sot.bearing(), atol=1e-9)

    def test_bearing_residual_zero_at_truth(self):
        rng = np.random.default_rng(4)
        R_wb = _Exp_so3(rng.normal(0, 1, 3)); p_wb = rng.normal(0, 3, 3)
        p_L = p_wb + R_wb @ np.array([5.0, 1.0, -0.5])
        b = bearing_measurement(p_L, R_wb, p_wb)
        r = bearing_residual(b, p_L, R_wb, p_wb)
        np.testing.assert_allclose(r, 0.0, atol=1e-12)


class TestBearingJacobians:

    def test_pose_jacobian_right_invariant_finite_difference(self):
        """Analytic right-invariant pose Jacobian must match the finite difference of the
        bearing under the SE_2(3) right-invariant error X = Exp(xi) X_hat."""
        rng = np.random.default_rng(5)
        R = _Exp_so3(rng.normal(0, 1, 3)); v = rng.normal(0, 2, 3); p = rng.normal(0, 3, 3)
        Xh = _make_X(R, v, p)
        p_L = p + R @ np.array([6.0, 1.5, -1.0])

        def meas(xi):
            X = _Exp_se23(xi) @ Xh
            return bearing_measurement(p_L, X[0:3, 0:3], X[0:3, 4])

        Hn = np.zeros((3, 9)); eps = 1e-6
        for k in range(9):
            e = np.zeros(9); e[k] = eps
            Hn[:, k] = (meas(e) - meas(-e)) / (2 * eps)
        H = bearing_jacobian_pose(p_L, R, p, right_invariant=True)
        np.testing.assert_allclose(H, Hn, atol=1e-5)

    def test_landmark_jacobian_scale_invariance(self):
        """d bearing / d (inverse-depth scale s) == 0 (bearing is scale-invariant): a single
        bearing cannot constrain depth — it must come from parallax."""
        rng = np.random.default_rng(6)
        R_wb = _Exp_so3(rng.normal(0, 1, 3)); p_wb = rng.normal(0, 3, 3)
        sot = SOT3(Q=_Exp_so3(rng.normal(0, 0.5, 3)), rho=0.4)

        def meas(xi):
            s2 = sot.retract(xi)
            p_L = reconstruct_world_point(s2, R_wb, p_wb)
            return bearing_measurement(p_L, R_wb, p_wb)

        Hn = np.zeros((3, 4)); eps = 1e-6
        for k in range(4):
            e = np.zeros(4); e[k] = eps
            Hn[:, k] = (meas(e) - meas(-e)) / (2 * eps)
        H = bearing_jacobian_landmark(sot, R_wb, p_wb)
        # scale column (index 3) must be ~0 in BOTH analytic and numeric
        np.testing.assert_allclose(H[:, 3], 0.0, atol=1e-9)
        np.testing.assert_allclose(Hn[:, 3], 0.0, atol=1e-6)
        # direction columns must match
        np.testing.assert_allclose(H[:, 0:3], Hn[:, 0:3], atol=1e-5)


class TestTriangulation:

    def test_depth_recovered_from_parallax(self):
        """Two bearings from a baseline recover the world point and inverse depth — the
        multi-frame ingredient that makes the SOT(3) rho observable."""
        R0 = np.eye(3); p0 = np.zeros(3)
        R1 = np.eye(3); p1 = np.array([0.0, 2.0, 0.0])    # 2 m baseline
        p_L = np.array([10.0, 1.0, 0.5])
        b0 = bearing_measurement(p_L, R0, p0)
        b1 = bearing_measurement(p_L, R1, p1)
        p_est, inv_depth = triangulate_inverse_depth(
            np.array([b0, b1]), np.array([R0, R1]), np.array([p0, p1]))
        np.testing.assert_allclose(p_est, p_L, atol=1e-6)
        true_depth = np.linalg.norm(R0.T @ (p_L - p0))
        np.testing.assert_allclose(inv_depth, 1.0 / true_depth, atol=1e-6)
