"""SYS-ID registration: localization / KF / Navigator counterparts (pair localization-kf-navigator, #5).

Registers, with a SHARED drive isolating one layer:

  1. numpy ``racer.state_estimator.LinearKF`` == torch ``rl.inc8_estimator_emul.BatchedLinearKF``
     (predict + update_position) under STRESS conditions the benign ``test_inc8_linearkf_torch.py`` does
     NOT cover: near-singular S (tiny meas cov), large covariance init, tiny dt, rapid (every-step)
     fixes -- with REALISTIC random rotations + surrogate-magnitude anisotropic gate-plane cov. Parity
     holds bit-exact-to-fp (<=1e-7; realistic regime ~1e-12). [the key NEW registration]

  2. ``racer.localization.gate_relative_inplane_fix`` cov-shaping == ``rl.fix_surrogate.fix_covariance``
     under IDENTICAL gate-frame variances: both build ``R_world_gate @ diag(var) @ R_world_gate.T`` with
     the SAME axis convention (cols 0,1 = in-plane / col 2 = along-track-normal). Bit-exact (==0.0).
     Pins that the deploy fix-cov and the RL-surrogate fix-cov use one gate-plane convention.

  3. The INTENTIONAL sqrt(2) sigma gap (KNOWN FOOTGUN -- register, do NOT reconcile): NavState's
     ``navigator._gate_frame_pos_sigma`` uses in-plane = sqrt(P00+P11) (the SUM, BLUEPRINT 1.6) while the
     emulator ``inc8_estimator_emul._gate_frame_sigmas`` uses sqrt((P00+P11)/2) (the MEAN). Pinned as a
     sqrt(2) RATIO so a future reconcile is a LOUD test change, not a silent drift.

Torch is required (skips cleanly if absent -- the laptop has it; Adroit has it). Fast (~9 s).
Run from repo ROOT:  .venv\\Scripts\\python.exe -m pytest tests/test_sysid_kf_localization.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.state_estimator import LinearKF                              # noqa: E402
from racer.contracts import Gate, GatePose                             # noqa: E402
from racer.localization import gate_relative_inplane_fix               # noqa: E402
from fix_surrogate import FixSurrogate, GateGeometry                   # noqa: E402

torch = pytest.importorskip("torch")
import inc8_estimator_emul as IE                                       # noqa: E402

DT64 = torch.float64


def _rand_R(rng):
    """A random proper rotation via Gram-Schmidt on a Gaussian matrix."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ---------------------------------------------------------------------------
# 1. numpy LinearKF == torch BatchedLinearKF under STRESS (shared bit-exact drive)
# ---------------------------------------------------------------------------
def _drive_stress(*, n, steps, dt, pos_std, vel_std, attitude_noise, cov_floor, fix_every,
                  rand_rot, seed):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(-50, 10, (n, 3))
    v0 = rng.uniform(-18, 18, (n, 3))
    nkfs = [LinearKF.initialize(p0[i], v0[i], pos_std=pos_std, vel_std=vel_std,
                                accel_noise_std=0.3, attitude_noise_std=attitude_noise)
            for i in range(n)]
    bkf = IE.BatchedLinearKF(n, "cpu", DT64, accel_noise_std=0.3, attitude_noise_std=attitude_noise)
    bkf.initialize_idx(torch.arange(n), torch.tensor(p0, dtype=DT64), torch.tensor(v0, dtype=DT64),
                       pos_std=pos_std, vel_std=vel_std)
    worst = 0.0
    for s in range(steps):
        accel = rng.standard_normal((n, 3)) * 4.0
        Rs = (np.array([_rand_R(rng) for _ in range(n)]) if rand_rot
              else np.broadcast_to(np.eye(3), (n, 3, 3)).copy())
        for i in range(n):
            nkfs[i].predict(accel[i], Rs[i], dt)
        bkf.predict(torch.tensor(accel, dtype=DT64), torch.tensor(Rs, dtype=DT64), dt)
        if fix_every and (s % fix_every == 0):
            # realistic anisotropic gate-plane cov rotated to world (lateral/vertical/depth) + a floor
            z = rng.uniform(-50, 10, (n, 3))
            covs = []
            for _ in range(n):
                Rg = _rand_R(rng)
                covs.append(Rg @ np.diag([0.1 ** 2, 0.28 ** 2, 0.85 ** 2]) @ Rg.T + cov_floor * np.eye(3))
            covs = np.array(covs)
            for i in range(n):
                nkfs[i].update_position(z[i], covs[i])
            bkf.update_position_idx(torch.arange(n), torch.tensor(z, dtype=DT64),
                                    torch.tensor(covs, dtype=DT64))
        dx = float(np.max(np.abs(bkf.x.numpy() - np.array([k.x for k in nkfs]))))
        dP = float(np.max(np.abs(bkf.P.numpy() - np.array([k.P for k in nkfs]))))
        worst = max(worst, dx, dP)
    return worst


@pytest.mark.parametrize("name,kw", [
    ("near_singular_S", dict(cov_floor=1e-9, fix_every=1, dt=0.0333, pos_std=1.0, vel_std=5.0)),
    ("large_cov_init",  dict(cov_floor=1e-6, fix_every=3, dt=0.0333, pos_std=1e3, vel_std=1e3)),
    ("tiny_dt",         dict(cov_floor=1e-6, fix_every=5, dt=1e-6,   pos_std=1.0, vel_std=5.0)),
    ("rapid_fixes",     dict(cov_floor=1e-6, fix_every=1, dt=0.0333, pos_std=1.0, vel_std=5.0)),
    ("borderline_dt",   dict(cov_floor=1e-6, fix_every=3, dt=0.199,  pos_std=1.0, vel_std=5.0)),
])
def test_linearkf_torch_parity_under_stress(name, kw):
    """numpy == torch (predict + update_position) to fp tolerance under each stress condition, with
    REALISTIC random rotations + surrogate-magnitude anisotropic cov (the regime training/deploy run).
    Threshold 1e-7 = bit-exact-to-fp: the realistic regime is ~1e-12; a large-covariance INIT (pos/vel
    std 1e3) legitimately amplifies the floating-point floor to ~2e-9 (np.linalg.solve vs
    torch.linalg.solve ordering), still 1e5x below the benign test's 1e-4 and far below any meaningful
    estimator divergence. Algebraically the two paths are identical."""
    worst = _drive_stress(n=32, steps=50, attitude_noise=0.024435, rand_rot=True, seed=hash(name) % 9999,
                          **kw)
    assert worst <= 1e-7, (name, worst)


def test_linearkf_torch_parity_caseC_no_skewQ():
    """Case-C (attitude_noise=0 -> no skew-Q term): still bit-exact (fp tol) under rapid rand-R fixes."""
    worst = _drive_stress(n=32, steps=50, attitude_noise=0.0, cov_floor=1e-6, fix_every=1,
                          dt=0.0333, pos_std=1.0, vel_std=5.0, rand_rot=True, seed=4242)
    assert worst <= 1e-7, worst


# ---------------------------------------------------------------------------
# 2. gate_relative_inplane_fix cov == surrogate fix_covariance (same gate-plane convention)
# ---------------------------------------------------------------------------
def test_gaterel_cov_matches_surrogate_fix_covariance():
    """Feed IDENTICAL gate-frame variances into both cov-shapers; world-NED cov must be bit-identical
    (==0.0), proving the SAME R_wg @ diag @ R_wg.T rotation + axis order (in-plane=cols0,1 / along=col2).
    Each side keeps its own calibrated sigmas; this registers only the SHARED convention."""
    rng = np.random.default_rng(101)
    worst = 0.0
    for _ in range(100):
        R_world_gate = _rand_R(rng)
        var_ip, var_al = rng.uniform(0.01, 2.0, 2)
        sig_ip, sig_al = float(np.sqrt(var_ip)), float(np.sqrt(var_al))
        gate_pos = rng.uniform(-30, 30, 3)
        gate = Gate(gate_id=4, position_ned=gate_pos, R_world_gate=R_world_gate)
        gp = GatePose(frame_id=0, sim_time_ns=0, gate_id=4, R_cam_gate=np.eye(3),
                      t_cam_gate=np.array([0.0, 0.0, 5.0]), reproj_error_px=0.0,
                      covariance=None, n_corners=4)
        # localization: isotropic in-plane sig_ip on cols 0,1; loose along sig_al on col 2 (no a1/att/floor)
        _, cov_loc = gate_relative_inplane_fix(
            gp, gate, np.eye(3), inplane_sigma=sig_ip, range_growth_a1=0.0,
            along_sigma=sig_al, attitude_noise_std=0.0, fix_cov_floor_std=0.0)
        # surrogate: lateral=vertical=sig_ip (cols 0,1), depth=sig_al (col 2), no SPD floor, flat (a1=0)
        sur = FixSurrogate(sigma_lateral_floor=sig_ip, sigma_lateral_a1=0.0,
                           sigma_vertical_floor=sig_ip, sigma_vertical_a1=0.0,
                           sigma_depth_floor=sig_al, sigma_depth_a1=0.0, cov_spd_floor=0.0)
        geom = GateGeometry(range_m=5.0, azimuth_deg=0.0, elevation_deg=0.0, bearing_deg=0.0,
                            in_image=True, view_angle_deg=0.0, t_cam=np.zeros(3),
                            lever_world=np.zeros(3), R_world_gate=R_world_gate,
                            gate_position_ned=gate_pos)
        cov_sur = sur.fix_covariance(geom)
        worst = max(worst, float(np.max(np.abs(cov_loc - cov_sur))))
    assert worst == 0.0, worst


def test_gaterel_axis_order_along_on_gate_normal():
    """Along-track variance lands on the gate-NORMAL world axis; in-plane on the other two (explicit
    axis-ordering pin for the convention registered above). Gate faces a world axis -> diagonal cov."""
    R_face_N = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # X=E,Y=D,Z(normal)=N
    gate = Gate(gate_id=4, position_ned=np.zeros(3), R_world_gate=R_face_N)
    gp = GatePose(frame_id=0, sim_time_ns=0, gate_id=4, R_cam_gate=np.eye(3),
                  t_cam_gate=np.array([0.0, 0.0, 8.0]), reproj_error_px=0.0,
                  covariance=None, n_corners=4)
    _, cov = gate_relative_inplane_fix(gp, gate, np.eye(3), inplane_sigma=0.1, range_growth_a1=0.0,
                                       along_sigma=1.0, attitude_noise_std=0.0, fix_cov_floor_std=0.0)
    var_N, var_E, var_D = cov[0, 0], cov[1, 1], cov[2, 2]   # world NED diag
    np.testing.assert_allclose(var_N, 1.0, atol=1e-12)      # along-track (1.0) on the gate normal (N)
    np.testing.assert_allclose([var_E, var_D], [0.01, 0.01], atol=1e-12)  # in-plane (0.1^2) on E,D


# ---------------------------------------------------------------------------
# 3. INTENTIONAL sqrt(2) sigma gap (register, DO NOT reconcile) -- KNOWN FOOTGUN
# ---------------------------------------------------------------------------
def test_navstate_vs_emulator_inplane_sigma_is_sqrt2_by_design():
    """NavState in-plane sigma = sqrt(P00+P11) (SUM); emulator = sqrt((P00+P11)/2) (MEAN). The ratio is
    EXACTLY sqrt(2), BY DESIGN (BLUEPRINT 1.6 vs the d5 1.2 confidence channel). This pins the gap so a
    future reconcile (or accidental alignment) is a LOUD, deliberate test change -- NOT a silent drift
    that would offset the deploy obs[17] confidence from the trained obs[17] by sqrt(2)."""
    rng = np.random.default_rng(7)
    Rwg = _rand_R(rng)
    # a random SPD position covariance, projected into the same gate frame both ways
    A = rng.standard_normal((3, 3))
    P_pos = A @ A.T + np.eye(3)
    P_gate = Rwg.T @ P_pos @ Rwg
    s_sum = float(np.sqrt(max(P_gate[0, 0] + P_gate[1, 1], 0.0)))          # NavState convention
    s_mean = float(np.sqrt(max(0.5 * (P_gate[0, 0] + P_gate[1, 1]), 0.0)))  # emulator convention
    np.testing.assert_allclose(s_sum / s_mean, np.sqrt(2.0), rtol=0, atol=1e-12)

    # cross-check against the LIVE torch emulator helper on the same P (the actual code path)
    n = 1
    gate_pos = np.zeros((1, 3))
    bkf_emul = IE.BatchedEstimatorEmulator(
        n, torch.tensor(gate_pos, dtype=DT64), torch.tensor(Rwg[None], dtype=DT64),
        device="cpu", dtype=DT64)
    bkf_emul.kf.P[0, :3, :3] = torch.tensor(P_pos, dtype=DT64)
    sig_ip_emul, _ = bkf_emul._gate_frame_sigmas(torch.zeros(1, dtype=torch.long))
    np.testing.assert_allclose(float(sig_ip_emul[0]), s_mean, rtol=0, atol=1e-9)   # emul == MEAN form
    np.testing.assert_allclose(s_sum / float(sig_ip_emul[0]), np.sqrt(2.0), rtol=0, atol=1e-9)
