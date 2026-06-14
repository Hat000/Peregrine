"""RewindKF OOSM wrapper — G2 estimator units (C2, BLUEPRINT §1.4/§1.5).

Pins:
  - zero-latency / predict-forward-at-now (t_fix >= now) is BIT-IDENTICAL to bare LinearKF.update_position;
  - a capture-time OOSM fix == the in-order oracle (replay is exact, no approximation);
  - the posterior P stays symmetric positive-definite through a rewind (SPD-preserving);
  - horizon MUST be strictly > L (assert) -- horizon<=L drops 100% of fixes -> divergence;
  - a fix older than the buffer horizon is DROPPED (never applied to the current state).

Torch-free (numpy only). [C2-ESTIMATOR-CHAIN 2026-06-13]
"""
import numpy as np
import pytest

from racer.kf_rewind import RewindKF
from racer.state_estimator import LinearKF

_R = np.eye(3)
_HOVER = np.array([0.0, 0.0, -9.80665])     # specific force at rest (FRD): a_world = R@a + g = 0
_DT = 1.0 / 90.0
_DT_NS = int(_DT * 1e9)


def _fresh(pos=(0.0, 0.0, 0.0), vel=(10.0, 0.0, 0.0)):
    return LinearKF.initialize(np.asarray(pos, float), np.asarray(vel, float), pos_std=0.5, vel_std=0.5)


# ---------------------------------------------------------------------------
# (1) zero-latency degeneracy: t_fix >= now == bare update_position, BIT-IDENTICAL
# ---------------------------------------------------------------------------
def test_update_position_at_now_is_bit_identical_to_bare_update():
    rng = np.random.default_rng(7)
    bare = _fresh()
    rk = RewindKF(kf=_fresh(), horizon_s=0.5)
    t_ns = 0
    for k in range(60):
        t_ns += _DT_NS
        accel = _HOVER + rng.normal(0, 0.2, 3)        # identical inputs to both filters
        bare.predict(accel, _R, _DT)
        rk.predict(accel, _R, _DT, t_ns)
        if k % 10 == 5:
            z = bare.position + rng.normal(0, 0.3, 3)
            cov = np.diag([0.25, 0.25, 0.25])
            bare.update_position(z, cov)
            res = rk.update_position_at(t_ns, z, cov)   # t_fix == now -> in-place
            assert res.applied and not res.rewound
    # zero-latency fixes must leave the wrapped filter EXACTLY where the bare filter is.
    np.testing.assert_array_equal(rk.x, bare.x)
    np.testing.assert_array_equal(rk.P, bare.P)


def test_future_fix_degenerates_in_place():
    rk = RewindKF(kf=_fresh(), horizon_s=0.5)
    t_ns = 0
    for _ in range(10):
        t_ns += _DT_NS
        rk.predict(_HOVER, _R, _DT, t_ns)
    res = rk.update_position_at(t_ns + 5 * _DT_NS, np.array([1.0, 0.0, 0.0]), 0.04 * np.eye(3))
    assert res.applied and not res.rewound and res.rewind_depth_s == 0.0


# ---------------------------------------------------------------------------
# (2) OOSM capture-time fix == in-order oracle (exact replay)
# ---------------------------------------------------------------------------
def test_oosm_capture_time_equals_in_order_oracle():
    rng = np.random.default_rng(11)
    accels = [_HOVER + rng.normal(0, 0.2, 3) for _ in range(5)]
    z = np.array([0.12, -0.05, 0.03])
    cov = np.diag([0.09, 0.09, 0.09])

    # ORACLE: apply the fix IN ORDER, right after predict #1 (its capture step).
    oracle = _fresh()
    oracle.predict(accels[0], _R, _DT)
    oracle.update_position(z, cov)
    for a in accels[1:]:
        oracle.predict(a, _R, _DT)

    # OOSM: all predicts first, then the late fix stamped back at predict #1's time.
    rk = RewindKF(kf=_fresh(), horizon_s=0.5)
    t_ns = 0
    times = []
    for a in accels:
        t_ns += _DT_NS
        rk.predict(a, _R, _DT, t_ns)
        times.append(t_ns)
    res = rk.update_position_at(times[0], z, cov)   # capture time = predict #1's stamp
    assert res.applied and res.rewound and res.n_replayed_ops == 4
    np.testing.assert_allclose(rk.x, oracle.x, atol=1e-12, rtol=0)
    np.testing.assert_allclose(rk.P, oracle.P, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# (3) SPD-preserving through a rewind
# ---------------------------------------------------------------------------
def test_rewind_keeps_P_symmetric_positive_definite():
    rng = np.random.default_rng(3)
    rk = RewindKF(kf=_fresh(), horizon_s=0.5)
    t_ns = 0
    times = []
    for _ in range(20):
        t_ns += _DT_NS
        rk.predict(_HOVER + rng.normal(0, 0.2, 3), _R, _DT, t_ns)
        times.append(t_ns)
    res = rk.update_position_at(times[5], rng.normal(0, 0.3, 3), 0.09 * np.eye(3))
    assert res.rewound
    P = rk.P[:3, :3]
    np.testing.assert_allclose(P, P.T, atol=1e-12)               # symmetric
    assert np.all(np.linalg.eigvalsh(rk.P) > 0)                  # positive-definite (full 6x6)


# ---------------------------------------------------------------------------
# (4) horizon MUST be strictly > L  (the horizon<=L divergence trap)
# ---------------------------------------------------------------------------
def test_assert_horizon_gt_rejects_horizon_le_L():
    RewindKF(kf=_fresh(), horizon_s=0.5).assert_horizon_gt(0.125)   # 0.5 > 0.125 -> OK
    with pytest.raises(AssertionError, match="strictly"):
        RewindKF(kf=_fresh(), horizon_s=0.10).assert_horizon_gt(0.10)   # horizon == L -> raise
    with pytest.raises(AssertionError):
        RewindKF(kf=_fresh(), horizon_s=0.05).assert_horizon_gt(0.10)   # horizon < L -> raise


def test_horizon_below_L_drops_every_fix():
    # A small horizon with fixes stamped L behind now: every OOSM fix is older than the buffer ->
    # DROPPED (applied=False) -> the filter dead-reckons (the ~21 m trap, by construction).
    L_ns = int(0.10 * 1e9)
    rk = RewindKF(kf=_fresh(), horizon_s=0.05)        # horizon < L
    t_ns = 0
    n_fix = n_drop = 0
    for k in range(40):
        t_ns += _DT_NS
        rk.predict(_HOVER, _R, _DT, t_ns)
        if k > 10 and k % 3 == 0:
            res = rk.update_position_at(t_ns - L_ns, np.zeros(3), 0.09 * np.eye(3))
            n_fix += 1
            n_drop += int(not res.applied)
    assert n_fix > 0 and n_drop == n_fix          # 100% dropped


def test_fix_older_than_horizon_leaves_state_untouched():
    rk = RewindKF(kf=_fresh(), horizon_s=0.05)
    t_ns = 0
    for _ in range(20):
        t_ns += _DT_NS
        rk.predict(_HOVER, _R, _DT, t_ns)
    x0, P0 = rk.x.copy(), rk.P.copy()
    res = rk.update_position_at(t_ns - int(0.5 * 1e9), np.array([99.0, 0.0, 0.0]), 0.01 * np.eye(3))
    assert not res.applied and res.dropped_reason is not None
    np.testing.assert_array_equal(rk.x, x0)       # too-old fix NEVER touches the current state
    np.testing.assert_array_equal(rk.P, P0)
