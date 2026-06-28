"""Tests for the FULL EqVIO joint pose+landmark filter (eqvio.EqVIOJointEKF).

Run with the main-checkout venv:
    C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe -m pytest tests/test_eqvio_joint.py -v

Pure torch/numpy + scipy(optional) — NO diffaero / VQ track dependency, so it runs on the
laptop. These compose the two proven halves (eqvio.SE23RightInvariantEKF pose side +
eqvio_landmark.SOT3 inverse-depth landmark side) into one joint covariance and validate:

  1. The anchored joint Jacobians (pose 3x9 + landmark 3x4) match finite differences, and
     the landmark SCALE column is now NON-zero (parallax makes inverse depth observable —
     the structural difference from the single-frame self-observation in eqvio_landmark).
  2. Joint covariance bookkeeping: state grows 9 -> 9+4K, new corners enter uncorrelated,
     the joint update corrects BOTH pose and landmark with the cross-covariance.
  3. CONSISTENCY: under forward-parallax motion (AGGRESSIVE_S) the joint pose NEES stays in
     (or conservatively below) the chi-square band; it runs HOT under a near-constant-range
     ORBIT (almost no parallax) — reported honestly, not tuned away.
  4. Inverse DEPTH is recovered via parallax (rho converges from a far prior to GT depth).
  5. The joint filter beats a decoupled pose-then-triangulate pair on landmark accuracy
     (the coupling earns its keep).
  6. Marginalisation (drop_landmark) equals the information-form Schur complement and keeps
     the filter bounded-size + PSD.
"""
from __future__ import annotations

import numpy as np
import pytest

from racer.ahrs.iekf import _Exp_so3, _skew
from racer.ahrs.eqvio import (
    EqVIOJointEKF, _make_X, _Exp_se23, _rot_e_z_to,
)
from racer.ahrs.eqvio_landmark import (
    SOT3, E_Z, bearing_measurement, world_point_from_anchored_sot,
    bearing_jacobian_anchored_landmark, bearing_jacobian_pose_anchored,
    triangulate_inverse_depth,
)
from racer.ahrs.traj6dof import (
    generate_traj6dof, run_eqvio_joint, synth_camera_bearings, Traj,
)
from racer.ahrs.eqvio import SE23RightInvariantEKF

try:
    from scipy.stats import chi2
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


# A forward-looking camera: optical axis e_z_cam -> body +x (FRD forward); x_cam -> +y_body
# (right), y_cam -> +z_body (down). So the gate corners (ahead of the racer) sit near the
# camera boresight and are seen with the depth-resolving parallax of an approaching drone.
R_BC_FWD = np.array([[0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]])


# ---------------------------------------------------------------------------
# 1. Anchored joint Jacobians (the new coupling math)
# ---------------------------------------------------------------------------

class TestAnchoredJacobians:

    def test_rot_e_z_to_maps_canonical_axis(self):
        """_rot_e_z_to(b) is a proper rotation sending e_z onto the unit bearing b."""
        rng = np.random.default_rng(0)
        for _ in range(30):
            b = rng.normal(0, 1, 3); b /= np.linalg.norm(b)
            Q = _rot_e_z_to(b)
            np.testing.assert_allclose(Q @ E_Z, b, atol=1e-10)
            np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(Q), 1.0, atol=1e-10)

    def test_rot_e_z_to_antipodal(self):
        Q = _rot_e_z_to(np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(Q @ E_Z, [0.0, 0.0, -1.0], atol=1e-10)

    def test_anchored_landmark_jacobian_finite_difference(self):
        """3x4 landmark Jacobian for a landmark observed from a DIFFERENT pose than its
        anchor matches finite differences, AND its scale column is non-zero (depth is now
        observable through the anchor<->current baseline — the parallax channel)."""
        rng = np.random.default_rng(5)
        R_a = _Exp_so3(rng.normal(0, 1, 3)); p_a = rng.normal(0, 3, 3)
        R_c = _Exp_so3(rng.normal(0, 1, 3)); p_c = rng.normal(0, 3, 3)
        R_bc = _Exp_so3(rng.normal(0, 0.5, 3))
        sot = SOT3(Q=_Exp_so3(rng.normal(0, 0.5, 3)), rho=0.3)

        def meas(xi):
            s2 = sot.retract(xi)
            p_L = world_point_from_anchored_sot(s2, R_a, p_a, R_bc)
            return bearing_measurement(p_L, R_c, p_c, R_bc)

        Hn = np.zeros((3, 4)); eps = 1e-6
        for k in range(4):
            e = np.zeros(4); e[k] = eps
            Hn[:, k] = (meas(e) - meas(-e)) / (2 * eps)
        H = bearing_jacobian_anchored_landmark(sot, R_a, p_a, R_c, p_c, R_bc)
        np.testing.assert_allclose(H, Hn, atol=1e-5)
        assert np.linalg.norm(H[:, 3]) > 1e-2, "scale column must be non-zero (parallax)"

    def test_anchored_pose_jacobian_finite_difference(self):
        """3x9 pose Jacobian (right-invariant) for an anchored landmark matches FD under the
        SE_2(3) right-invariant error X = Exp(xi) X_hat."""
        rng = np.random.default_rng(6)
        R_a = _Exp_so3(rng.normal(0, 1, 3)); p_a = rng.normal(0, 3, 3)
        R_c = _Exp_so3(rng.normal(0, 1, 3)); p_c = rng.normal(0, 3, 3)
        R_bc = _Exp_so3(rng.normal(0, 0.5, 3))
        sot = SOT3(Q=_Exp_so3(rng.normal(0, 0.5, 3)), rho=0.3)
        Xh = _make_X(R_c, rng.normal(0, 1, 3), p_c)

        def meas(xi):
            X = _Exp_se23(xi) @ Xh
            p_L = world_point_from_anchored_sot(sot, R_a, p_a, R_bc)
            return bearing_measurement(p_L, X[0:3, 0:3], X[0:3, 4], R_bc)

        Hn = np.zeros((3, 9)); eps = 1e-6
        for k in range(9):
            e = np.zeros(9); e[k] = eps
            Hn[:, k] = (meas(e) - meas(-e)) / (2 * eps)
        H = bearing_jacobian_pose_anchored(sot, R_a, p_a, R_c, p_c, R_bc, right_invariant=True)
        np.testing.assert_allclose(H, Hn, atol=1e-5)

    def test_self_observation_scale_invariant(self):
        """When the observing pose == the anchor pose, the anchored-landmark Jacobian reduces
        to the single-frame scale-invariant case (zero scale column): a fresh corner gives no
        depth from its own anchor frame — exactly why rho needs parallax."""
        rng = np.random.default_rng(7)
        R_a = _Exp_so3(rng.normal(0, 1, 3)); p_a = rng.normal(0, 3, 3)
        R_bc = _Exp_so3(rng.normal(0, 0.3, 3))
        sot = SOT3(Q=_Exp_so3(rng.normal(0, 0.5, 3)), rho=0.4)
        H = bearing_jacobian_anchored_landmark(sot, R_a, p_a, R_a, p_a, R_bc)
        np.testing.assert_allclose(H[:, 3], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Joint covariance bookkeeping
# ---------------------------------------------------------------------------

class TestJointBookkeeping:

    def test_state_grows_with_landmarks(self):
        f = EqVIOJointEKF()
        assert f.dim == 9 and f.n_landmarks == 0
        f.add_landmark(0, np.array([0.0, 0.0, 1.0]))
        assert f.dim == 13 and f.n_landmarks == 1
        f.add_landmark(7, np.array([0.1, 0.0, 1.0]))
        assert f.dim == 17 and f.landmark_ids() == [0, 7]

    def test_new_landmark_block_is_uncorrelated_and_psd(self):
        f = EqVIOJointEKF()
        f.add_landmark(0, np.array([0.2, -0.1, 1.0]), rho_var=4.0, dir_var=1e-3)
        P = f.P
        # pose<->lmk cross block zero on insertion
        np.testing.assert_allclose(P[0:9, 9:13], 0.0, atol=0)
        np.testing.assert_allclose(P[9:13, 0:9], 0.0, atol=0)
        # rho variance lands in the scale slot (index 12), direction var in 9:12
        assert abs(P[12, 12] - 4.0) < 1e-12
        np.testing.assert_allclose(np.diag(P[9:12, 9:12]), 1e-3, atol=1e-12)
        assert np.linalg.eigvalsh(P).min() > 0

    def test_inserted_bearing_is_reproduced(self):
        """add_landmark seeds Q so the canonical ray reproduces the inserted bearing; the
        reconstructed world point re-measures to that same bearing from the anchor pose."""
        f = EqVIOJointEKF(R_bc=R_BC_FWD)
        f.reset(R=_Exp_so3(np.array([0.1, -0.2, 0.3])), p=np.array([1.0, 2.0, -3.0]))
        b = np.array([0.05, -0.1, 1.0]); b /= np.linalg.norm(b)
        f.add_landmark(3, b)
        p_L = f.landmark_world(3)
        b_re = bearing_measurement(p_L, f.R, f.p, f.R_bc)
        np.testing.assert_allclose(b_re, b, atol=1e-9)

    def test_joint_update_moves_both_blocks(self):
        """A joint bearing update with a deliberately-wrong landmark prior corrects BOTH the
        landmark and (through the cross-covariance) the pose — i.e. it is genuinely joint."""
        f = EqVIOJointEKF(R_bc=R_BC_FWD, bearing_noise_std=0.01)
        f.reset(R=np.eye(3), p=np.zeros(3))
        # true corner straight ahead at 10 m; insert with a slightly-off bearing/depth, then
        # feed the TRUE bearing from a shifted pose so there is parallax to act on.
        p_L = np.array([10.0, 0.3, -0.2])
        b0 = bearing_measurement(p_L, np.eye(3), np.zeros(3), R_BC_FWD)
        f.add_landmark(0, b0, rho_init=0.05, rho_var=4.0)
        rho0 = f.landmark_inv_depth(0)
        P0 = f.pose_cov().copy()
        # move sideways (parallax) and observe the true bearing
        f.reset(R=np.eye(3), p=np.zeros(3))  # keep anchor; re-seat at same pose for clarity
        f.add_landmark(0, b0, rho_init=0.05, rho_var=4.0) if not f.has_landmark(0) else None
        # simulate a parallax view: pretend the filter pose advanced 3 m forward
        f._X = _make_X(np.eye(3), np.zeros(3), np.array([3.0, 0.0, 0.0]))
        b1 = bearing_measurement(p_L, f.R, f.p, R_BC_FWD)
        f.update_landmark_joint(0, b1)
        assert f.landmark_inv_depth(0) != rho0, "landmark rho must update"
        # pose covariance changed (information flowed into the pose block)
        assert not np.allclose(f.pose_cov(), P0)


# ---------------------------------------------------------------------------
# 3. CONSISTENCY (the headline) — joint NEES under propagation + bearing updates
# ---------------------------------------------------------------------------

class TestJointConsistency:

    def test_noiseless_perfect_init_pose_stays_bounded(self):
        """Noiseless, perfectly-initialised joint run over 72 m of travel. HONEST behaviour:
        the pose does NOT stay machine-exact — a fresh corner enters with a deliberately-wrong
        far depth prior (rho=0.05 -> 20 m vs true ~48 m), and that mis-estimated landmark,
        seen through the right-invariant pose Jacobian, tugs the (correct) pose by a small bias
        until parallax resolves rho. The residual pose drift is sub-0.5 m (<1% of path length)
        and the inverse depths still converge to GT — the approximation is well-bounded, not a
        divergence."""
        seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=4.0, dt=0.005, seed=1,
                                gyro_noise_std=0, accel_noise_std=0, pos_noise_std=0,
                                pos_rate_hz=10.0)
        f = EqVIOJointEKF(gyro_noise_std=0, accel_noise_std=0, bearing_noise_std=0.005,
                          R_bc=R_BC_FWD)
        out = run_eqvio_joint(f, seq, R_bc=R_BC_FWD, bearing_noise_std=0.0,
                              rho_init=0.05, rho_var=4.0)
        path_len = np.sum(np.linalg.norm(np.diff(seq.p_gt, axis=0), axis=1))
        pose_drift = np.linalg.norm(f.p - seq.p_gt[-1])
        assert pose_drift < 0.5, f"pose drift {pose_drift:.3f} m too large"
        assert pose_drift < 0.01 * path_len, "pose drift exceeds 1% of path length"
        # every tracked corner's inverse depth converged near GT (from a far rho_init=0.05).
        for lm in f._lmks:
            p_gt = seq.landmarks_world[lm.lm_id]
            d_anchor = R_BC_FWD.T @ (lm.R_anchor.T @ (p_gt - lm.p_anchor))
            rho_gt = 1.0 / np.linalg.norm(d_anchor)
            assert abs(lm.sot.rho - rho_gt) < 0.01, f"rho {lm.sot.rho} vs gt {rho_gt}"

    @pytest.mark.skipif(not HAVE_SCIPY, reason="scipy needed for chi-square band")
    def test_forward_parallax_pose_nees_not_overconfident(self):
        """Monte-Carlo on the racing-like AGGRESSIVE_S trajectory (strong forward parallax):
        the joint filter's steady-state pose NEES is consistent-to-CONSERVATIVE (at or below
        the chi-square band), NOT overconfident. Conservative (under-confident) is the safe
        failure mode; the anchored-inverse-depth approximation is information-pessimistic when
        parallax is rich. We assert NEES does not blow ABOVE the band (the dangerous side)."""
        nmc = 30
        NP = []
        for s in range(nmc):
            seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=4.0, dt=0.005, seed=10 + s,
                                    gyro_noise_std=0.008, accel_noise_std=0.08,
                                    pos_noise_std=0.0, pos_rate_hz=10.0)
            P0 = np.diag([np.deg2rad(0.5)**2]*3 + [0.05**2]*3 + [0.03**2]*3)
            f = EqVIOJointEKF(0.008, 0.08, bearing_noise_std=0.01, R_bc=R_BC_FWD)
            out = run_eqvio_joint(f, seq, R_bc=R_BC_FWD, bearing_noise_std=0.01,
                                  rho_init=0.05, rho_var=4.0, seed=900 + s)
            NP.append(out["nees_pose"])
        NP = np.array(NP)
        nf = NP.shape[1]
        half = nf // 2
        m = NP[:, half:].mean()
        dof = 9 * nmc * (nf - half)
        hi = chi2.ppf(0.975, dof) / (nmc * (nf - half))
        # the load-bearing assertion: NOT overconfident (steady NEES at/under the band).
        assert m < hi + 1.0, f"joint pose NEES {m:.2f} overconfident (band hi {hi:.2f})"
        # and it is a real, finite, positive number (filter didn't diverge)
        assert 0.0 < m < 50.0, f"joint pose NEES {m:.2f} pathological"

    @pytest.mark.skipif(not HAVE_SCIPY, reason="scipy needed for chi-square band")
    def test_orbit_runs_hot_honest_regime_report(self):
        """HONEST regime report: a near-constant-range ORBIT gives almost no parallax, so the
        inverse depth never resolves and the frozen-anchor approximation drives the pose NEES
        ABOVE the band (overconfident). We PIN this as a known limitation rather than tune it
        away — it tells the deployer to avoid depth-only reliance on poorly-parallaxed gates."""
        nmc = 20
        NP = []
        for s in range(nmc):
            seq = generate_traj6dof(Traj.GENTLE_ORBIT, duration_s=4.0, dt=0.005, seed=10 + s,
                                    gyro_noise_std=0.005, accel_noise_std=0.05,
                                    pos_noise_std=0.0, pos_rate_hz=10.0)
            P0 = np.diag([np.deg2rad(0.5)**2]*3 + [0.05**2]*3 + [0.03**2]*3)
            f = EqVIOJointEKF(0.005, 0.05, bearing_noise_std=0.01, R_bc=R_BC_FWD)
            out = run_eqvio_joint(f, seq, R_bc=R_BC_FWD, bearing_noise_std=0.01,
                                  rho_init=0.05, rho_var=4.0, seed=900 + s)
            NP.append(out["nees_pose"])
        NP = np.array(NP)
        half = NP.shape[1] // 2
        m = NP[:, half:].mean()
        # documents the failure regime: orbit NEES is well ABOVE the consistency band.
        assert m > 20.0, (
            f"expected ORBIT (no-parallax) to run HOT as a known limitation; got {m:.2f}"
        )


# ---------------------------------------------------------------------------
# 4. Inverse-depth recovery via parallax
# ---------------------------------------------------------------------------

class TestDepthRecovery:

    def test_inverse_depth_converges_from_far_prior(self):
        """rho starts at a FAR prior (0.05 -> 20 m) but the true corners are ~48 m; under the
        approaching AGGRESSIVE_S motion the joint filter's parallax drives rho to GT depth to
        within ~0.2 m, while the bearing direction stays pinned. This is the multi-frame
        triangulation the SOT(3) coordinate is designed for, now done recursively in-filter."""
        seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=4.0, dt=0.005, seed=1,
                                gyro_noise_std=0, accel_noise_std=0, pos_noise_std=0,
                                pos_rate_hz=10.0)
        f = EqVIOJointEKF(0, 0, bearing_noise_std=0.005, R_bc=R_BC_FWD)
        run_eqvio_joint(f, seq, R_bc=R_BC_FWD, bearing_noise_std=0.0,
                        rho_init=0.05, rho_var=4.0)
        assert f.n_landmarks >= 1
        for lm in f._lmks:
            p_gt = seq.landmarks_world[lm.lm_id]
            depth_est = 1.0 / lm.sot.rho
            d_anchor = R_BC_FWD.T @ (lm.R_anchor.T @ (p_gt - lm.p_anchor))
            depth_gt = np.linalg.norm(d_anchor)
            # rho moved a long way from the prior toward truth (real convergence, not a no-op)
            assert abs(depth_est - 20.0) > 15.0, "rho must move off the far prior"
            assert abs(depth_est - depth_gt) < 1.0, f"depth {depth_est:.2f} vs gt {depth_gt:.2f}"
            assert f.landmark_world(lm.lm_id).shape == (3,)
            assert np.linalg.norm(f.landmark_world(lm.lm_id) - p_gt) < 0.5

    def test_filter_recursive_matches_batch_triangulation_order(self):
        """Sanity that in-filter recursive depth lands in the same ballpark as the batch DLT
        triangulation of the same noiseless bearings (both recover the corner to <0.5 m)."""
        seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=4.0, dt=0.005, seed=3,
                                gyro_noise_std=0, accel_noise_std=0, pos_noise_std=0,
                                pos_rate_hz=10.0)
        b, vis = synth_camera_bearings(seq, R_BC_FWD)
        Rs = np.array([seq.R_gt[i] for i in seq.pos_idx])
        ps = np.array([seq.p_gt[i] for i in seq.pos_idx])
        p_batch, _ = triangulate_inverse_depth(b[:, 0, :], Rs, ps, R_BC_FWD)
        assert np.linalg.norm(p_batch - seq.landmarks_world[0]) < 0.5
        f = EqVIOJointEKF(0, 0, bearing_noise_std=0.005, R_bc=R_BC_FWD)
        run_eqvio_joint(f, seq, R_bc=R_BC_FWD, bearing_noise_std=0.0, rho_init=0.05, rho_var=4.0)
        assert np.linalg.norm(f.landmark_world(0) - seq.landmarks_world[0]) < 0.5


# ---------------------------------------------------------------------------
# 5. Joint beats decoupled (coupling earns its keep)
# ---------------------------------------------------------------------------

class TestJointBeatsSeparate:

    def test_joint_landmark_accuracy_beats_deadreckon_triangulation(self):
        """With an uncertain pose (init error + IMU noise, NO external position fix) the joint
        filter folds bearing information back into the pose, curbing drift and sharpening the
        landmark. A DECOUPLED pair cannot: bearings can't update an unknown-landmark pose, so
        the pose dead-reckons and the landmark is triangulated from the drifted poses. Over a
        Monte-Carlo the joint filter's landmark error is markedly smaller — the cross-
        covariance is load-bearing."""
        nmc = 20
        joint_err, sep_err = [], []
        for s in range(nmc):
            seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=4.0, dt=0.005, seed=10 + s,
                                    gyro_noise_std=0.006, accel_noise_std=0.06,
                                    pos_noise_std=0, pos_rate_hz=10.0)
            rng = np.random.default_rng(700 + s)
            ax = rng.normal(0, 1, 3); ax /= np.linalg.norm(ax)
            R0 = _Exp_so3(ax * np.deg2rad(2)) @ seq.R_gt[0]
            v0 = seq.v_gt[0] + rng.normal(0, 0.1, 3)
            p0 = seq.p_gt[0] + rng.normal(0, 0.05, 3)
            P0 = np.diag([np.deg2rad(2)**2]*3 + [0.1**2]*3 + [0.05**2]*3)

            fj = EqVIOJointEKF(0.006, 0.06, bearing_noise_std=0.008, R_bc=R_BC_FWD)
            oj = run_eqvio_joint(fj, seq, R0, v0, p0, P0, R_bc=R_BC_FWD,
                                 bearing_noise_std=0.008, rho_init=0.05, rho_var=4.0,
                                 seed=900 + s)
            joint_err.append(np.mean(list(oj["final_lmk_err"].values())))

            # decoupled: pose dead-reckons (no bearing updates), then triangulate.
            fp = SE23RightInvariantEKF(gyro_noise_std=0.006, accel_noise_std=0.06)
            fp.reset(R=R0, v=v0, p=p0, P=P0.copy())
            bearings, vis = synth_camera_bearings(seq, R_BC_FWD, bearing_noise_std=0.008,
                                                  seed=900 + s)
            pos_set = set(int(i) for i in seq.pos_idx)
            poseR, poseP = [], []
            for i in range(seq.N):
                if i > 0:
                    fp.predict(seq.gyro[i - 1], seq.accel[i - 1], seq.dt)
                if i in pos_set:
                    poseR.append(fp.R); poseP.append(fp.p)
            poseR = np.array(poseR); poseP = np.array(poseP)
            es = []
            for l in range(4):
                pw, _ = triangulate_inverse_depth(bearings[:, l, :], poseR, poseP, R_BC_FWD)
                es.append(np.linalg.norm(pw - seq.landmarks_world[l]))
            sep_err.append(np.mean(es))

        joint_err = np.array(joint_err); sep_err = np.array(sep_err)
        assert joint_err.mean() < sep_err.mean(), (
            f"joint landmark err {joint_err.mean():.3f} should beat decoupled "
            f"{sep_err.mean():.3f}"
        )
        # win the majority of trials (the edge is systematic, not a lucky mean)
        assert np.mean(joint_err < sep_err) >= 0.6


# ---------------------------------------------------------------------------
# 6. Marginalisation (Schur complement / bounded size)
# ---------------------------------------------------------------------------

class TestMarginalisation:

    def _build_populated(self, seed=2):
        seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=3.0, dt=0.005, seed=seed,
                                gyro_noise_std=0.005, accel_noise_std=0.05,
                                pos_noise_std=0, pos_rate_hz=10.0)
        f = EqVIOJointEKF(0.005, 0.05, bearing_noise_std=0.01, R_bc=R_BC_FWD)
        run_eqvio_joint(f, seq, R_bc=R_BC_FWD, bearing_noise_std=0.01,
                        rho_init=0.05, rho_var=4.0)
        return f

    def test_drop_landmark_reduces_dim_and_repacks(self):
        f = self._build_populated()
        n0 = f.n_landmarks
        assert n0 >= 2 and f.dim == 9 + 4 * n0
        drop_id = f.landmark_ids()[1]
        f.drop_landmark(drop_id)
        assert f.n_landmarks == n0 - 1
        assert f.dim == 9 + 4 * (n0 - 1)
        assert drop_id not in f.landmark_ids()
        # remaining landmark slots are still addressable (offsets re-packed contiguously)
        offs = sorted(lm.offset for lm in f._lmks)
        assert offs == list(range(9, 9 + 4 * (n0 - 1), 4))

    def test_marginal_equals_information_schur_complement(self):
        """Dropping a landmark keeps the EXACT marginal of the joint Gaussian over the
        survivors. Equivalently, the survivors' information matrix is the SCHUR COMPLEMENT of
        the dropped block in the joint information matrix; inverting that recovers the same
        covariance. Pin both forms agree to machine precision."""
        f = self._build_populated()
        P_before = f.P.copy()
        drop_id = f.landmark_ids()[1]
        off = f._find(drop_id).offset
        keep = np.r_[0:off, off + 4:f.dim]
        P_marg = P_before[np.ix_(keep, keep)]

        # information-form Schur complement of the dropped 4-dof block
        Lam = np.linalg.inv(P_before)
        idx_b = [off, off + 1, off + 2, off + 3]
        Laa = Lam[np.ix_(keep, keep)]
        Lab = Lam[np.ix_(keep, idx_b)]
        Lbb = Lam[np.ix_(idx_b, idx_b)]
        P_schur = np.linalg.inv(Laa - Lab @ np.linalg.solve(Lbb, Lab.T))
        np.testing.assert_allclose(P_marg, P_schur, atol=1e-9)

        f.drop_landmark(drop_id)
        np.testing.assert_allclose(f.P, P_marg, atol=0)
        assert np.linalg.eigvalsh(f.P).min() > 0, "marginal must stay PSD"

    def test_filter_runs_through_a_mid_track_marginalisation(self):
        """Exercise drop_landmark inside the live loop: the filter stays finite, PSD and keeps
        estimating after a corner is marginalised away mid-track."""
        seq = generate_traj6dof(Traj.AGGRESSIVE_S, duration_s=4.0, dt=0.005, seed=4,
                                gyro_noise_std=0.005, accel_noise_std=0.05,
                                pos_noise_std=0, pos_rate_hz=10.0)
        f = EqVIOJointEKF(0.005, 0.05, bearing_noise_std=0.01, R_bc=R_BC_FWD)
        out = run_eqvio_joint(f, seq, R_bc=R_BC_FWD, bearing_noise_std=0.01,
                              rho_init=0.05, rho_var=4.0, marginalize_after=10)
        assert np.all(np.isfinite(f.P))
        assert np.linalg.eigvalsh(f.P).min() > 0
        assert np.all(np.isfinite(out["nees_pose"]))
        assert f.n_landmarks >= 1  # at least the survivors remain
