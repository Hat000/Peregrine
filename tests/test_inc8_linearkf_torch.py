"""S2 PARITY GATE -- batched torch LinearKF == numpy racer.state_estimator.LinearKF (<=1e-4).

Drives BOTH the per-env numpy filter (loop) and the batched torch filter through the SAME
(accel_body, R_wb, dt) predict sequence with the SAME interleaved (z, cov) position updates, then
asserts state x and covariance P agree to <=1e-4 at every step. Exercises both case-C
(attitude_noise=0) and the attitude-error Q term (skew). Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_inc8_linearkf_torch.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.state_estimator import LinearKF                            # noqa: E402
import inc8_estimator_emul as IE                                      # noqa: E402

DT64 = torch.float64


def _rand_R(rng):
    """A random proper rotation via Gram-Schmidt on a Gaussian matrix."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@pytest.mark.parametrize("attitude_noise", [0.0, 0.024435])   # case-C + a non-zero skew-Q case
def test_batched_kf_matches_numpy_over_sequence(attitude_noise):
    rng = np.random.default_rng(7)
    n, steps, accel_noise = 96, 60, 0.3
    dt = 0.0333
    pos_std, vel_std = 1.0, 5.0

    # init
    p0 = rng.uniform(-50, 10, (n, 3))
    v0 = rng.uniform(-18, 18, (n, 3))
    nkfs = [LinearKF.initialize(p0[i], v0[i], pos_std=pos_std, vel_std=vel_std,
                                accel_noise_std=accel_noise, attitude_noise_std=attitude_noise)
            for i in range(n)]
    bkf = IE.BatchedLinearKF(n, "cpu", DT64, accel_noise_std=accel_noise,
                             attitude_noise_std=attitude_noise)
    bkf.initialize_idx(torch.arange(n), torch.tensor(p0, dtype=DT64), torch.tensor(v0, dtype=DT64),
                       pos_std=pos_std, vel_std=vel_std)
    assert np.max(np.abs(bkf.x.numpy() - np.array([k.x for k in nkfs]))) <= 1e-12
    assert np.max(np.abs(bkf.P.numpy() - np.array([k.P for k in nkfs]))) <= 1e-12

    for s in range(steps):
        accel = rng.standard_normal((n, 3)) * 4.0
        Rs = np.array([_rand_R(rng) for _ in range(n)])
        # numpy predict
        for i in range(n):
            nkfs[i].predict(accel[i], Rs[i], dt)
        # torch predict (all envs)
        bkf.predict(torch.tensor(accel, dtype=DT64), torch.tensor(Rs, dtype=DT64), dt)

        # interleave a per-env random fix every ~3 steps
        if s % 3 == 1:
            mask = rng.random(n) < 0.6
            idx = np.nonzero(mask)[0]
            if idx.size:
                z = rng.uniform(-50, 10, (idx.size, 3))
                covs = []
                for j in range(idx.size):
                    L = rng.standard_normal((3, 3)) * 0.2
                    cov = L @ L.T + 0.05 * np.eye(3)         # SPD
                    covs.append(cov)
                covs = np.array(covs)
                for k, i in enumerate(idx):
                    nkfs[i].update_position(z[k], covs[k])
                bkf.update_position_idx(torch.tensor(idx), torch.tensor(z, dtype=DT64),
                                        torch.tensor(covs, dtype=DT64))

        dx = np.max(np.abs(bkf.x.numpy() - np.array([k.x for k in nkfs])))
        dP = np.max(np.abs(bkf.P.numpy() - np.array([k.P for k in nkfs])))
        assert dx <= 1e-4, (s, dx)
        assert dP <= 1e-4, (s, dP)


def test_predict_drops_bad_dt():
    """A non-positive / implausibly-large dt is dropped wholesale (mirrors LinearKF.predict guard)."""
    n = 4
    bkf = IE.BatchedLinearKF(n, "cpu", DT64)
    bkf.initialize_idx(torch.arange(n), torch.zeros(n, 3, dtype=DT64), torch.zeros(n, 3, dtype=DT64),
                       pos_std=1.0, vel_std=1.0)
    x0, P0 = bkf.x.clone(), bkf.P.clone()
    bkf.predict(torch.ones(n, 3, dtype=DT64), torch.eye(3, dtype=DT64).expand(n, 3, 3), 0.0)
    assert torch.equal(bkf.x, x0) and torch.equal(bkf.P, P0)
    bkf.predict(torch.ones(n, 3, dtype=DT64), torch.eye(3, dtype=DT64).expand(n, 3, 3), 0.5)
    assert torch.equal(bkf.x, x0) and torch.equal(bkf.P, P0)


def test_update_keeps_P_spd_and_symmetric():
    n = 8
    rng = np.random.default_rng(3)
    bkf = IE.BatchedLinearKF(n, "cpu", DT64)
    bkf.initialize_idx(torch.arange(n), torch.tensor(rng.uniform(-5, 5, (n, 3)), dtype=DT64),
                       torch.zeros(n, 3, dtype=DT64), pos_std=1.0, vel_std=2.0)
    z = torch.tensor(rng.uniform(-5, 5, (n, 3)), dtype=DT64)
    cov = torch.eye(3, dtype=DT64).expand(n, 3, 3) * 0.1
    bkf.update_position_idx(torch.arange(n), z, cov)
    P = bkf.P.numpy()
    assert np.max(np.abs(P - np.transpose(P, (0, 2, 1)))) < 1e-10        # symmetric
    for i in range(n):
        assert np.all(np.linalg.eigvalsh(P[i]) > 0)                     # SPD
