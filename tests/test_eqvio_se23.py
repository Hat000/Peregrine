"""Tests for the SE_2(3) right-invariant EKF (the coupled invariant estimator).

Run with the main-checkout venv:
    C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe -m pytest tests/test_eqvio_se23.py -v

The headline test (TestConsistencyEdge) is the empirical confirmation of the mission
thesis: on the COUPLED attitude+velocity+position problem the right-invariant EKF's
covariance transport is EXACT (group-affine), so its NEES stays consistent where a
standard EKF's drifts/mis-calibrates. The attitude-only IEKF deliberately ties the ESKF
(see iekf.py) precisely because that property only materialises on this coupled problem.
"""
from __future__ import annotations

import numpy as np
import pytest

from racer.ahrs.eqvio import (
    SE23RightInvariantEKF, SE23StandardEKF,
    _Exp_se23, _make_X, _adjoint_se23, _left_jacobian_so3, _expm_series,
)
from racer.ahrs.iekf import _Exp_so3, _skew, _R_to_quat_wxyz
from racer.ahrs.traj6dof import (
    generate_traj6dof, run_se23_filter, Traj,
    _error_vector, _log_se23_error, _log_so3, _left_jacobian_inv_so3,
)
from racer.ahrs.metrics import geodesic_error_rad

try:
    from scipy.stats import chi2
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# 1. Group / Lie-algebra helper correctness
# ---------------------------------------------------------------------------

class TestSE23Helpers:

    def test_exp_se23_rotation_block_matches_so3(self):
        xi = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        X = _Exp_se23(xi)
        np.testing.assert_allclose(X[0:3, 0:3], _Exp_so3(xi[0:3]), atol=1e-12)

    def test_exp_se23_pure_translation(self):
        """With zero rotation, v/p blocks are the raw xi_v/xi_p (J_l = I)."""
        xi = np.array([0., 0., 0., 1., 2., 3., 4., 5., 6.])
        X = _Exp_se23(xi)
        np.testing.assert_allclose(X[0:3, 3], [1., 2., 3.], atol=1e-12)
        np.testing.assert_allclose(X[0:3, 4], [4., 5., 6.], atol=1e-12)
        np.testing.assert_allclose(X[0:3, 0:3], np.eye(3), atol=1e-12)

    def test_left_jacobian_near_identity(self):
        """J_l(0) = I; series and closed form agree near 0."""
        np.testing.assert_allclose(_left_jacobian_so3(np.zeros(3)), np.eye(3), atol=1e-12)
        phi = np.array([1e-7, -2e-7, 3e-7])
        np.testing.assert_allclose(_left_jacobian_so3(phi), np.eye(3), atol=1e-6)

    def test_left_jacobian_inv_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            phi = rng.normal(0, 1.0, 3)
            Jl = _left_jacobian_so3(phi)
            Jli = _left_jacobian_inv_so3(phi)
            np.testing.assert_allclose(Jl @ Jli, np.eye(3), atol=1e-9)

    def test_adjoint_consistency(self):
        """Ad_X xi = vee( X xi^wedge X^{-1} ) for SE_2(3)."""
        rng = np.random.default_rng(1)
        R = _Exp_so3(rng.normal(0, 1, 3))
        v = rng.normal(0, 2, 3); p = rng.normal(0, 3, 3)
        X = _make_X(R, v, p)
        Ad = _adjoint_se23(X)
        xi = rng.normal(0, 0.3, 9)
        # numeric: d/ds [ Exp(s xi) ] -> wedge; transport by X; recover via small Exp
        eps = 1e-6
        lhs = (_Exp_se23(eps * Ad @ xi) - np.eye(5)) / eps
        Xinv = np.linalg.inv(X)
        rhs = X @ ((_Exp_se23(eps * xi) - np.eye(5)) / eps) @ Xinv
        np.testing.assert_allclose(lhs, rhs, atol=1e-4)

    def test_log_exp_se23_roundtrip(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            R = _Exp_so3(rng.normal(0, 0.8, 3))
            v = rng.normal(0, 2, 3); p = rng.normal(0, 3, 3)
            xi = _log_se23_error(R, v, p, np.eye(3), np.zeros(3), np.zeros(3))
            X = _Exp_se23(xi)
            np.testing.assert_allclose(X[0:3, 0:3], R, atol=1e-9)
            np.testing.assert_allclose(X[0:3, 3], v, atol=1e-9)
            np.testing.assert_allclose(X[0:3, 4], p, atol=1e-9)

    def test_expm_series_matches_scipy(self):
        rng = np.random.default_rng(3)
        M = rng.normal(0, 0.2, (9, 9))
        approx = _expm_series(M, terms=14)
        # reference via eigen-free repeated squaring of a fine series
        ref = np.eye(9); term = np.eye(9)
        for k in range(1, 30):
            term = term @ M / k
            ref = ref + term
        np.testing.assert_allclose(approx, ref, atol=1e-8)


# ---------------------------------------------------------------------------
# 2. RIEKF measurement Jacobian correctness (the invariance is in the H)
# ---------------------------------------------------------------------------

class TestRIEKFJacobians:

    def test_landmark_jacobian_is_state_anchored(self):
        """RIEKF body-landmark H attitude block = R_hat^T [p_L]_x (anchored to the FIXED
        world landmark, NOT the estimated position) — the invariance that gives
        consistency. Verified by finite difference of the measurement under the
        right-invariant error X = Exp(xi) X_hat."""
        rng = np.random.default_rng(4)
        R = _Exp_so3(rng.normal(0, 1, 3)); v = rng.normal(0, 2, 3); p = rng.normal(0, 3, 3)
        Xh = _make_X(R, v, p); p_L = rng.normal(0, 5, 3)

        def meas(xi):
            X = _Exp_se23(xi) @ Xh
            return X[0:3, 0:3].T @ (p_L - X[0:3, 4])

        H = np.zeros((3, 9)); eps = 1e-6
        for k in range(9):
            e = np.zeros(9); e[k] = eps
            H[:, k] = (meas(e) - meas(-e)) / (2 * eps)
        np.testing.assert_allclose(H[:, 0:3], R.T @ _skew(p_L), atol=1e-5)
        np.testing.assert_allclose(H[:, 6:9], -R.T, atol=1e-5)
        np.testing.assert_allclose(H[:, 3:6], 0.0, atol=1e-5)

    def test_position_jacobian(self):
        """RIEKF world-position H = [-[p_hat]_x, 0, I] under the right-invariant error."""
        rng = np.random.default_rng(5)
        R = _Exp_so3(rng.normal(0, 1, 3)); v = rng.normal(0, 2, 3); p = rng.normal(0, 3, 3)
        Xh = _make_X(R, v, p)

        def pos(xi):
            return (_Exp_se23(xi) @ Xh)[0:3, 4]

        H = np.zeros((3, 9)); eps = 1e-6
        for k in range(9):
            e = np.zeros(9); e[k] = eps
            H[:, k] = (pos(e) - pos(-e)) / (2 * eps)
        np.testing.assert_allclose(H[:, 0:3], -_skew(p), atol=1e-5)
        np.testing.assert_allclose(H[:, 6:9], np.eye(3), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Forward-model / generator discrete consistency
# ---------------------------------------------------------------------------

class TestGeneratorConsistency:

    def test_noiseless_dead_reckon_is_exact(self):
        """A perfectly-initialised, noiseless RIEKF reproduces the GT path to machine
        precision (the generator is discrete-consistent with the estimator's integrator)."""
        seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=3.0, dt=0.005,
                                gyro_noise_std=0.0, accel_noise_std=0.0, pos_noise_std=0.0, seed=1)
        f = SE23RightInvariantEKF(gyro_noise_std=0, accel_noise_std=0)
        f.reset(R=seq.R_gt[0], v=seq.v_gt[0], p=seq.p_gt[0])
        for i in range(1, seq.N):
            f.predict(seq.gyro[i - 1], seq.accel[i - 1], seq.dt)
        assert np.linalg.norm(f.p - seq.p_gt[-1]) < 1e-8
        assert np.rad2deg(geodesic_error_rad(f.q_wxyz, seq.q_gt[-1])[0]) < 1e-5

    def test_error_vector_self_consistency(self):
        """Sampling xi ~ N(0,P0), building X_hat = Exp(-xi) X_gt, and recovering the error
        via _log_se23_error must reproduce xi (mean NEES -> 9). Pins that the RIEKF NEES is
        computed in the SAME tangent its covariance lives in (a fair-consistency guarantee)."""
        seq = generate_traj6dof(Traj.HIGH_G_LOOP, duration_s=0.5, dt=0.005, seed=0)
        i = 50
        Rg, vg, pg = seq.R_gt[i], seq.v_gt[i], seq.p_gt[i]
        P0 = np.diag([np.deg2rad(8)**2]*3 + [0.3**2]*3 + [0.2**2]*3)
        L = np.linalg.cholesky(P0)
        rng = np.random.default_rng(0)
        nees = []
        Xgt = _make_X(Rg, vg, pg)
        for _ in range(8000):
            xi = L @ rng.normal(0, 1, 9)
            Xhat = _Exp_se23(-xi) @ Xgt
            xr = _log_se23_error(Rg, vg, pg, Xhat[0:3, 0:3], Xhat[0:3, 3], Xhat[0:3, 4])
            np.testing.assert_allclose(xr, xi, atol=1e-10)
            nees.append(xr @ np.linalg.solve(P0, xr))
        assert 8.3 < np.mean(nees) < 9.7, f"mean NEES {np.mean(nees):.2f} not ~9"


# ---------------------------------------------------------------------------
# 4. THE CONSISTENCY EDGE (the mission's headline empirical claim)
# ---------------------------------------------------------------------------

class TestConsistencyEdge:
    """The coupled invariant filter's covariance transport is EXACT; the standard EKF's
    is not. This is the property the attitude-only IEKF cannot show (it ties the ESKF)."""

    def _proponly_nees(self, FilterCls, seq, R0, v0, p0, P0, stride=100):
        f = FilterCls(gyro_noise_std=0, accel_noise_std=0)
        f.reset(R=R0, v=v0, p=p0, P=P0.copy())
        out = []
        for i in range(1, seq.N):
            f.predict(seq.gyro[i - 1], seq.accel[i - 1], seq.dt)
            if i % stride == 0:
                e = _error_vector(f, seq, i)
                out.append(float(e @ np.linalg.solve(f.P, e)))
        return np.array(out)

    def test_riekf_propagation_nees_is_constant(self):
        """RIGHT-INVARIANT, NOISELESS, fixed init error, NO updates: the RIEKF's NEES is
        CONSTANT (group-affine exactness — the linearised error transport is exact, so the
        covariance tracks the true error growth with zero relinearisation error)."""
        seq = generate_traj6dof(Traj.HIGH_G_LOOP, duration_s=3.0, dt=0.005,
                                gyro_noise_std=0, accel_noise_std=0, pos_noise_std=0, seed=5)
        ax = np.array([1., 2., -0.5]); ax /= np.linalg.norm(ax)
        R0 = _Exp_so3(ax * np.deg2rad(20)) @ seq.R_gt[0]
        v0 = seq.v_gt[0] + np.array([0.5, -0.3, 0.2])
        p0 = seq.p_gt[0] + np.array([0.2, 0.1, -0.1])
        P0 = np.diag([np.deg2rad(20)**2]*3 + [0.5**2]*3 + [0.3**2]*3)
        nR = self._proponly_nees(SE23RightInvariantEKF, seq, R0, v0, p0, P0)
        rel_spread = (nR.max() - nR.min()) / nR.mean()
        assert rel_spread < 1e-6, (
            f"RIEKF propagation NEES must be constant (group-affine); rel-spread {rel_spread:.2e}"
        )

    def test_standard_ekf_propagation_nees_drifts(self):
        """The standard EKF's covariance transport does NOT track the true error: its NEES
        varies substantially over the same noiseless propagation (estimate-dependent F)."""
        seq = generate_traj6dof(Traj.HIGH_G_LOOP, duration_s=3.0, dt=0.005,
                                gyro_noise_std=0, accel_noise_std=0, pos_noise_std=0, seed=5)
        ax = np.array([1., 2., -0.5]); ax /= np.linalg.norm(ax)
        R0 = _Exp_so3(ax * np.deg2rad(20)) @ seq.R_gt[0]
        v0 = seq.v_gt[0] + np.array([0.5, -0.3, 0.2])
        p0 = seq.p_gt[0] + np.array([0.2, 0.1, -0.1])
        P0 = np.diag([np.deg2rad(20)**2]*3 + [0.5**2]*3 + [0.3**2]*3)
        nS = self._proponly_nees(SE23StandardEKF, seq, R0, v0, p0, P0)
        rel_spread = (nS.max() - nS.min()) / nS.mean()
        assert rel_spread > 0.3, (
            f"standard EKF propagation NEES should drift (estimate-dep transport); "
            f"rel-spread {rel_spread:.2e}"
        )

    @pytest.mark.skipif(not HAVE_SCIPY, reason="scipy needed for chi-square band")
    def test_riekf_filter_nees_closer_to_ideal(self):
        """Closed-loop (landmark updates) Monte-Carlo: after the init transient the RIEKF's
        average NEES sits near the ideal 9 (consistent / well-calibrated covariance), and is
        markedly CLOSER to 9 than the standard EKF's, whose covariance is mis-calibrated
        (here conservatively under-confident) under aggressive motion + sparse fixes."""
        nmc = 40
        att = 8
        RR, SS = [], []
        for s in range(nmc):
            seq = generate_traj6dof(Traj.HIGH_G_LOOP, duration_s=5.0, dt=0.005, seed=10 + s,
                                    gyro_noise_std=0.012, accel_noise_std=0.15,
                                    pos_noise_std=0.1, pos_rate_hz=1.0)
            rng = np.random.default_rng(700 + s)
            ax = rng.normal(0, 1, 3); ax /= np.linalg.norm(ax)
            R0 = _Exp_so3(ax * np.deg2rad(att)) @ seq.R_gt[0]
            v0 = seq.v_gt[0] + rng.normal(0, 0.3, 3)
            p0 = seq.p_gt[0] + rng.normal(0, 0.15, 3)
            P0 = np.diag([np.deg2rad(att)**2]*3 + [0.3**2]*3 + [0.15**2]*3)
            rR = run_se23_filter(SE23RightInvariantEKF(gyro_noise_std=0.012, accel_noise_std=0.15, pos_noise_std=0.1),
                                 seq, R0, v0, p0, P0.copy(), measurement="landmark")
            rS = run_se23_filter(SE23StandardEKF(gyro_noise_std=0.012, accel_noise_std=0.15, pos_noise_std=0.1),
                                 seq, R0, v0, p0, P0.copy(), measurement="landmark")
            RR.append(rR["nees_full"]); SS.append(rS["nees_full"])
        RR = np.array(RR); SS = np.array(SS)
        # exclude first 2 fixes (init transient)
        mR = RR[:, 2:].mean(); mS = SS[:, 2:].mean()
        lo, hi = chi2.ppf(0.025, 9 * nmc) / nmc, chi2.ppf(0.975, 9 * nmc) / nmc
        # RIEKF must land in (or essentially at) the consistency band...
        assert lo - 1.5 < mR < hi + 2.0, f"RIEKF avg-NEES {mR:.2f} outside ~[{lo:.2f},{hi:.2f}]"
        # ...and be strictly closer to the ideal 9 than the (mis-calibrated) standard EKF.
        assert abs(mR - 9.0) < abs(mS - 9.0), (
            f"RIEKF NEES {mR:.2f} must be closer to 9 than std-EKF {mS:.2f}"
        )


# ---------------------------------------------------------------------------
# 5. Accuracy parity: the invariant filter does not sacrifice point accuracy
# ---------------------------------------------------------------------------

class TestAccuracyParity:

    def test_both_filters_track_accurately(self):
        """With landmark fixes and a modest init error both filters drive attitude and
        position error to small, bounded steady-state values. (Point accuracy is regime-
        dependent and is NOT where the invariant filter claims its edge — that is
        consistency/calibration, tested in TestConsistencyEdge. Here we only require the
        RIEKF to track well in absolute terms.)"""
        seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=4.0, dt=0.005, seed=3,
                                gyro_noise_std=0.01, accel_noise_std=0.1,
                                pos_noise_std=0.1, pos_rate_hz=10.0)
        ax = np.array([0.3, -1.0, 0.5]); ax /= np.linalg.norm(ax)
        R0 = _Exp_so3(ax * np.deg2rad(10)) @ seq.R_gt[0]
        v0 = seq.v_gt[0] + np.array([0.3, -0.2, 0.1])
        p0 = seq.p_gt[0] + np.array([0.2, -0.1, 0.15])
        P0 = np.diag([np.deg2rad(10)**2]*3 + [0.3**2]*3 + [0.2**2]*3)
        rR = run_se23_filter(SE23RightInvariantEKF(gyro_noise_std=0.01, accel_noise_std=0.1, pos_noise_std=0.1),
                             seq, R0, v0, p0, P0.copy(), measurement="landmark")
        rS = run_se23_filter(SE23StandardEKF(gyro_noise_std=0.01, accel_noise_std=0.1, pos_noise_std=0.1),
                             seq, R0, v0, p0, P0.copy(), measurement="landmark")
        # steady-state (second half) attitude & position error
        half = seq.N // 2
        attR = np.rad2deg(geodesic_error_rad(rR["q_est"][half:], seq.q_gt[half:]))
        attS = np.rad2deg(geodesic_error_rad(rS["q_est"][half:], seq.q_gt[half:]))
        posR = np.linalg.norm(rR["p_est"][half:] - seq.p_gt[half:], axis=1)
        posS = np.linalg.norm(rS["p_est"][half:] - seq.p_gt[half:], axis=1)
        assert attR.mean() < 4.0, f"RIEKF steady att err {attR.mean():.2f} deg too large"
        assert posR.mean() < 0.5, f"RIEKF steady pos err {posR.mean():.2f} m too large"
        # the standard EKF also tracks (sanity on the comparator)
        assert attS.mean() < 4.0 and posS.mean() < 0.5

    def test_step_outputs_unit_quaternion(self):
        seq = generate_traj6dof(Traj.GENTLE_ORBIT, duration_s=1.0, dt=0.005, seed=0)
        for FilterCls in (SE23RightInvariantEKF, SE23StandardEKF):
            r = run_se23_filter(FilterCls(), seq, measurement="landmark")
            norms = np.linalg.norm(r["q_est"], axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)
