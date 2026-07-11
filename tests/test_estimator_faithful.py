"""Tests for the ESTIMATOR-FAITHFUL ACTOR OBS package (2026-07-11; owner directive: THE ACTOR
NEVER SEES GROUND TRUTH). Layers:

  (T0) TRANSLATION PARITY: the batched-torch BatchedESKFLeveler / BatchedNavKF (float64) against
       VERBATIM-vendored numpy excerpts of the deploy code (tests/_deploy_ref_eskf.py @ ego-deploy
       e1aa4d1), on randomized sequences engineered to hit EVERY gate branch (freefall skip,
       high-band skip, gate<1e-4 skip, chi2 reject, motion-reject inflate + stateful re-anchor,
       accepted update; KF dt-drop, Joseph update, in-plane eigen floor). Agreement <= 1e-9.
  (CONV) CONVENTION PINS: hover specific force holds level in the Z-up/FLU instantiation;
       roll/pitch seed round-trips through the extraction; tilt-vector formula == R row-2.
  (ZOH) THE ALIASING PREMISE (synthetic negative control, graft #5): with truth itself
       piecewise-constant-rate at tick rate (the n_substeps=1 in-env analog) the tick-rate leveler
       tracks truth to float precision; with intra-tick rate variation (the n_substeps=5 analog)
       it diverges -- the fatal channel EXISTS only when intra-tick dynamics exist.
  (FAIL) FAILURE MODES: spin-regime stability (1000 steps, ||w|| to 6 rad/s, |a| 0-5.2 g: finite,
       unit quat, P symmetric-PSD); episode-boundary isolation (reset_idx touches ONLY idx).
  (INIT) deploy-faithful takeoff seeding: leveler q0 == truth roll/pitch exactly, P0 == the
       measured pad-converged eskf_p_launch (guards accidental randomization).
  (OFF) DEFAULT-OFF byte-identity: legacy outputs equal the legacy formulas AND the RNG stream
       state equals a twin-generator replay of the documented legacy draw sequence (draw-count
       invariance, graft #5); the faithful path draws strictly fewer (its own draw count).
  (COUP) velocity-attitude coupling: the obs velocity error appears exactly when the attitude
       lies (shared-R corruption), not when att stays 'gt'.
  (WIRE) ENV WIRING: stub-self executed _step_estimator passes the plant's captured sf to the
       estimator; get_state's critic truth source is knob-gated; source-pins (getsource).
  (ACC)  ACCEPTANCE (fixture replay, runs everywhere -- tests/data/a5_flight_ticks.json): Mode-A
       re-anchored parity vs the recorded wire + the recorded-grid divergence band == the measured
       wire benchmark + the 33 ms training band + KF dead-reckoning parity.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_estimator_faithful.py -q
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "rl" / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import _deploy_ref_eskf as REF                                            # noqa: E402
from ego_ins_emul import (BatchedESKFLeveler, BatchedNavKF, GRAVITY,      # noqa: E402
                          ESKF_P0_DIAG, eskf_p_launch,
                          quat_from_roll_pitch_yaw_zyx, quat_to_R_wxyz,
                          tilt_vector_from_roll_pitch, tilt_angle_rad)
from ego_estimator import (BatchedEgoEstimator, EgoEstimatorConfig,        # noqa: E402
                           _euler_roll_pitch_from_R)
import peregrine_racing_ego as PRE                                        # noqa: E402

DT64 = torch.float64
G_NED = [0.0, 0.0, +GRAVITY]
G_ZUP = [0.0, 0.0, -GRAVITY]
FIXTURE = ROOT / "tests" / "data" / "a5_flight_ticks.json"


# ================================================================================================
# (T0) TRANSLATION PARITY -- every gate branch, batched torch vs verbatim numpy.
# ================================================================================================
def _branch_sequence(rng: np.random.Generator, n_steps: int):
    """Per-step (gyro, accel, dt) cycling through the branch cases (NED/FRD; rest sf = [0,0,-g])."""
    seq = []
    for k in range(n_steps):
        case = k % 6
        gyro = rng.normal(0.0, 1.5, 3)
        dt = float(rng.uniform(0.01, 0.15))
        if case == 0:      # ACCEPT: near-gravity magnitude, near-true direction
            accel = np.array([0.0, 0.0, -GRAVITY]) + rng.normal(0, 0.15, 3)
        elif case == 1:    # |a| < 1e-6 skip
            accel = rng.normal(0, 1, 3) * 1e-9
        elif case == 2:    # freefall band skip (0.1 g)
            d = rng.normal(0, 1, 3); accel = 0.1 * GRAVITY * d / np.linalg.norm(d)
        elif case == 3:    # high-|a| band skip (11 g)
            d = rng.normal(0, 1, 3); accel = 11.0 * GRAVITY * d / np.linalg.norm(d)
        elif case == 4:    # smooth gate < 1e-4 skip (3 g)
            d = rng.normal(0, 1, 3); accel = 3.0 * GRAVITY * d / np.linalg.norm(d)
        else:              # chi2 REJECT candidate: |a| ~ g, direction sideways
            accel = np.array([-GRAVITY, 0.0, 0.0]) + rng.normal(0, 0.05, 3)
        seq.append((gyro, accel, dt))
    return seq


@pytest.mark.parametrize("motion_reject", [True, False])
def test_t0_eskf_translation_parity_all_branches(motion_reject):
    N, n_steps = 6, 60
    rng = np.random.default_rng(42 + int(motion_reject))
    seqs = [_branch_sequence(rng, n_steps) for _ in range(N)]
    lev = BatchedESKFLeveler(N, g_world=G_NED, dtype=DT64,
                             use_accel_motion_reject=motion_reject)
    lev.reset_idx(torch.arange(N), torch.tensor([[1.0, 0, 0, 0]] * N, dtype=DT64),
                  P0=ESKF_P0_DIAG)                                # deploy cold boot for parity
    refs = [REF.ESKFAHRS(use_accel_motion_reject=motion_reject) for _ in range(N)]
    saw_update = False
    for k in range(n_steps):
        gy = torch.tensor(np.stack([s[k][0] for s in seqs]), dtype=DT64)
        ac = torch.tensor(np.stack([s[k][1] for s in seqs]), dtype=DT64)
        dts = torch.tensor([s[k][2] for s in seqs], dtype=DT64)
        lev.step(gy, ac, dts)
        for i in range(N):
            r = refs[i]
            r._predict(seqs[i][k][0], seqs[i][k][2])
            r._update_accel(np.asarray(seqs[i][k][1]))
            assert np.allclose(lev.q[i].numpy(), r._q, atol=1e-9), (i, k)
            assert np.allclose(lev.b_g[i].numpy(), r._b_g, atol=1e-9), (i, k)
            assert np.allclose(lev.P[i].numpy(), r._P, atol=1e-9), (i, k)
            if motion_reject:
                assert np.allclose(lev.R_ref[i].numpy(), r._R_ref, atol=1e-9), (i, k)
        if bool(lev.last_update_mask.any()):
            saw_update = True
    assert saw_update                        # the accept branch actually exercised


def test_t0_eskf_motion_reject_reanchor_branch_parity():
    """The A8 STATEFUL re-anchor, exercised deterministically: perturb R_ref 0.02 rad off R(q)
    (|a_lin| ~ g*0.02 ~ 0.196 < anchor_thr 0.3) and feed a perfect gravity sample with zero gyro
    -> both sides must RE-ANCHOR R_ref to the pre-update R(q); order (inside R_meas construction,
    BEFORE the chi2 gate) is shared with the deploy reference so parity pins position too."""
    lev = BatchedESKFLeveler(1, g_world=G_NED, dtype=DT64, use_accel_motion_reject=True)
    lev.reset_idx(torch.tensor([0]), torch.tensor([[1.0, 0, 0, 0]], dtype=DT64), P0=ESKF_P0_DIAG)
    ref = REF.ESKFAHRS(use_accel_motion_reject=True)
    th = 0.02
    Rp = np.array([[math.cos(th), -math.sin(th), 0.0],
                   [math.sin(th), math.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])
    lev.R_ref = torch.tensor(Rp, dtype=DT64).unsqueeze(0)
    ref._R_ref = Rp.copy()
    q_before = lev.q[0].numpy().copy()
    accel = np.array([0.0, 0.0, -GRAVITY])
    lev.step(torch.zeros(1, 3, dtype=DT64), torch.tensor(accel, dtype=DT64).unsqueeze(0), 1 / 30.)
    ref._predict(np.zeros(3), 1 / 30.)
    ref._update_accel(accel)
    assert np.allclose(lev.R_ref[0].numpy(), ref._R_ref, atol=1e-12)
    # exercised: R_ref snapped to the pre-update R(q) (zero gyro -> q unchanged by predict)
    assert np.allclose(ref._R_ref, REF._quat_to_R_wxyz(q_before), atol=1e-12)
    assert not np.allclose(ref._R_ref, Rp, atol=1e-6)                      # it actually MOVED


def test_t0_eskf_dt_nonpositive_is_noop():
    lev = BatchedESKFLeveler(2, g_world=G_NED, dtype=DT64)
    q_before = lev.q.clone(); P_before = lev.P.clone()
    lev.step(torch.ones(2, 3, dtype=DT64), torch.tensor([[0., 0., -GRAVITY]] * 2, dtype=DT64),
             torch.tensor([0.0, -0.1], dtype=DT64))
    assert torch.equal(lev.q, q_before) and torch.equal(lev.P, P_before)


def test_t0_navkf_translation_parity_predict_update_floor_dtdrop():
    N = 4
    rng = np.random.default_rng(7)
    kf = BatchedNavKF(N, g_world=G_NED, dtype=DT64, fix_chi2_thresh=0.0)   # gate off for parity
    kf.reset_idx(torch.arange(N), torch.zeros(N, 3, dtype=DT64), torch.zeros(N, 3, dtype=DT64))
    refs = [REF.LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=5.0, vel_std=1.0,
                                    inplane_pos_floor_std=0.05) for _ in range(N)]
    # attitude per env: random rotation
    qs = torch.tensor(rng.normal(size=(N, 4)), dtype=DT64)
    qs = qs / torch.linalg.norm(qs, dim=-1, keepdim=True)
    R = quat_to_R_wxyz(qs)
    saw_drop = saw_floor = False
    for k in range(40):
        sf = torch.tensor(rng.normal(0, 8, (N, 3)), dtype=DT64)
        dts = torch.tensor(rng.uniform(0.01, 0.3, N), dtype=DT64)          # some > 0.2 -> DROPPED
        kf.predict(sf, R, dts)
        for i in range(N):
            refs[i].predict(sf[i].numpy(), R[i].numpy(), float(dts[i]))
            if float(dts[i]) > 0.2:
                saw_drop = True
        if k % 3 == 0:                                                     # position fix
            z = torch.tensor(rng.normal(0, 2, (N, 3)), dtype=DT64)
            cov = torch.eye(3, dtype=DT64).expand(N, 3, 3) * float(rng.uniform(0.1, 0.5)) ** 2
            kf.update_position(z, cov, torch.ones(N, dtype=torch.bool))
            for i in range(N):
                p2 = refs[i].P[:2, :2].copy()
                refs[i].update_position(z[i].numpy(), cov[i].numpy())
                if np.linalg.eigvalsh(0.5 * (p2 + p2.T))[0] < 0.05 ** 2:
                    saw_floor = True
        for i in range(N):
            assert np.allclose(kf.x[i].numpy(), refs[i].x, atol=1e-9), (i, k)
            assert np.allclose(kf.P[i].numpy(), refs[i].P, atol=1e-9), (i, k)
    assert saw_drop


def test_t0_navkf_chi2_gate_rejects_teleport_class_fix():
    kf = BatchedNavKF(1, g_world=G_NED, dtype=DT64)                        # deploy 16.27 gate ON
    kf.reset_idx(torch.tensor([0]), torch.zeros(1, 3, dtype=DT64), torch.zeros(1, 3, dtype=DT64))
    # converge P a bit so a wild fix is inconsistent
    R = torch.eye(3, dtype=DT64).unsqueeze(0)
    for _ in range(5):
        kf.predict(torch.tensor([[0., 0., -GRAVITY]], dtype=DT64), R, 1 / 30.)
        kf.update_position(torch.zeros(1, 3, dtype=DT64),
                           torch.eye(3, dtype=DT64).unsqueeze(0) * 0.4 ** 2,
                           torch.ones(1, dtype=torch.bool))
    x_before = kf.x.clone()
    wild = torch.tensor([[80.0, -60.0, 40.0]], dtype=DT64)                 # teleport-class outlier
    kf.update_position(wild, torch.eye(3, dtype=DT64).unsqueeze(0) * 0.4 ** 2,
                       torch.ones(1, dtype=torch.bool))
    assert torch.equal(kf.x, x_before)                                     # REJECTED, state untouched


# ================================================================================================
# (CONV) CONVENTION PINS.
# ================================================================================================
def test_conv_hover_specific_force_holds_level_zup():
    """Z-up/FLU instantiation: at level attitude the hover sample [0,0,+g] is gravity-consistent
    -- accel updates fire (gate==1) and the leveler HOLDS level over 300 ticks."""
    lev = BatchedESKFLeveler(3, g_world=G_ZUP, dtype=DT64)
    lev.reset_idx(torch.arange(3), torch.tensor([[1.0, 0, 0, 0]] * 3, dtype=DT64))
    sf = torch.tensor([[0.0, 0.0, +GRAVITY]] * 3, dtype=DT64)
    for _ in range(300):
        lev.step(torch.zeros(3, 3, dtype=DT64), sf, 1 / 30.)
    rp = _euler_roll_pitch_from_R(lev.R_wb())
    assert rp.abs().max() < 1e-9
    assert bool(lev.last_update_mask.all())                                # accept branch live at 1 g


def test_conv_seed_roundtrip_and_tilt_vector_formula():
    g = torch.Generator().manual_seed(3)
    roll = (torch.rand(32, generator=g, dtype=DT64) - 0.5) * 2.0
    pitch = (torch.rand(32, generator=g, dtype=DT64) - 0.5) * 2.4
    q = quat_from_roll_pitch_yaw_zyx(roll, pitch, torch.zeros_like(roll))
    R = quat_to_R_wxyz(q)
    rp = _euler_roll_pitch_from_R(R)                                       # THE obs-parity extraction
    assert torch.allclose(rp[:, 0], roll, atol=1e-9)
    assert torch.allclose(rp[:, 1], pitch, atol=1e-9)
    # tilt vector u(roll, pitch) == row 2 of R (R^T e_z) -- the benchmark comparison space
    u = tilt_vector_from_roll_pitch(roll, pitch)
    assert torch.allclose(u, R[:, 2, :], atol=1e-9)


# ================================================================================================
# (ZOH) THE ALIASING PREMISE -- synthetic negative control (graft #5).
# ================================================================================================
def _integrate_truth(q, w_seq_dt):
    """Right-multiply truth quat by exp(w dt / 2) per (w, dt)."""
    from ego_ins_emul import rotvec_to_quat_wxyz, quat_mul_wxyz, normalize_quat
    for w, dt in w_seq_dt:
        q = normalize_quat(quat_mul_wxyz(q, rotvec_to_quat_wxyz(w * dt)))
    return q


def test_zoh_tickrate_truth_is_tracked_exactly_and_intratick_variation_diverges():
    """n_substeps=1 analog: truth piecewise-constant-rate per tick + leveler fed that same rate ->
    tracks truth to float precision (the fatal channel CANNOT exist in-sim at n_substeps=1).
    n_substeps=5 analog: truth integrates 5 different substep rates, leveler gets only the LAST
    over the full tick -> genuine divergence. Accel fed at 3 g (gate-dead, matching the measured
    flight regime) so this isolates the gyro/aliasing channel."""
    tick = 1 / 30.
    g = torch.Generator().manual_seed(11)
    d = torch.randn(3, generator=g, dtype=DT64)
    sf3g = (3.0 * GRAVITY * d / d.norm()).unsqueeze(0)                     # gate-dead sample

    # --- ZOH (tick-rate truth): EXACT tracking ---
    lev = BatchedESKFLeveler(1, g_world=G_ZUP, dtype=DT64)
    q_true = torch.tensor([[1.0, 0, 0, 0]], dtype=DT64)
    lev.reset_idx(torch.tensor([0]), q_true.clone())
    for k in range(100):
        w = torch.randn(1, 3, generator=g, dtype=DT64) * 3.0               # up to ~3 rad/s
        q_true = _integrate_truth(q_true, [(w, tick)])
        lev.step(w, sf3g, tick)
    err = tilt_angle_rad(_euler_roll_pitch_from_R(quat_to_R_wxyz(q_true)),
                         _euler_roll_pitch_from_R(lev.R_wb()))
    assert float(err) < 1e-9                                               # EXACT (rad)

    # --- intra-tick variation (n_substeps=5 analog): the aliasing channel appears ---
    lev2 = BatchedESKFLeveler(1, g_world=G_ZUP, dtype=DT64)
    q_true2 = torch.tensor([[1.0, 0, 0, 0]], dtype=DT64)
    lev2.reset_idx(torch.tensor([0]), q_true2.clone())
    per_tick = []
    for k in range(100):
        subs = [torch.randn(1, 3, generator=g, dtype=DT64) * 3.0 for _ in range(5)]
        prev_true = q_true2.clone()
        q_true2 = _integrate_truth(q_true2, [(w, tick / 5) for w in subs])
        prev_emul_rp = _euler_roll_pitch_from_R(lev2.R_wb())
        lev2.step(subs[-1], sf3g, tick)                                    # LATEST sample only
        # per-tick divergence vs the dense truth increment applied to the emul state
        q_dense = _integrate_truth(
            quat_from_roll_pitch_yaw_zyx(prev_emul_rp[:, 0], prev_emul_rp[:, 1],
                                         torch.zeros(1, dtype=DT64)),
            [(w, tick / 5) for w in subs])
        per_tick.append(float(tilt_angle_rad(_euler_roll_pitch_from_R(lev2.R_wb()),
                                             _euler_roll_pitch_from_R(quat_to_R_wxyz(q_dense)))))
    med = float(np.median(np.rad2deg(per_tick)))
    assert med > 0.5, med                                                  # the channel EXISTS


# ================================================================================================
# (FAIL) FAILURE MODES.
# ================================================================================================
def test_fail_spin_regime_stability_1000_steps_float32():
    N = 64
    lev = BatchedESKFLeveler(N, g_world=G_ZUP, dtype=torch.float32)
    kf = BatchedNavKF(N, g_world=G_ZUP, dtype=torch.float32)
    g = torch.Generator().manual_seed(5)
    q0 = torch.randn(N, 4, generator=g)
    lev.reset_idx(torch.arange(N), q0 / q0.norm(dim=-1, keepdim=True))
    kf.reset_idx(torch.arange(N), torch.zeros(N, 3), torch.zeros(N, 3))
    for k in range(1000):
        w = torch.randn(N, 3, generator=g) * 2.0
        w = w.clamp(-6.0, 6.0)
        mag = torch.rand(N, 1, generator=g) * 5.2 * GRAVITY                # |a| in [0, 5.2 g]
        d = torch.randn(N, 3, generator=g)
        sf = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9) * mag
        lev.step(w, sf, 1 / 30.)
        kf.predict(sf, lev.R_wb(), 1 / 30.)
        if k % 50 == 0:
            kf.update_position(torch.randn(N, 3, generator=g) * 3,
                               torch.eye(3).expand(N, 3, 3) * 0.4 ** 2,
                               torch.rand(N, generator=g) < 0.5)
    assert bool(torch.isfinite(lev.q).all() and torch.isfinite(lev.P).all())
    assert bool(torch.isfinite(kf.x).all() and torch.isfinite(kf.P).all())
    assert float((torch.linalg.norm(lev.q, dim=-1) - 1).abs().max()) < 1e-3
    sym = (lev.P - lev.P.transpose(-1, -2)).abs().max()
    assert float(sym) < 5e-4                                               # Joseph keeps symmetry
    #                                                                        (float32 slack)
    ev = torch.linalg.eigvalsh(0.5 * (lev.P + lev.P.transpose(-1, -2)).double())
    assert float(ev.min()) > -1e-6                                         # PSD within float32 slack


def test_fail_episode_boundary_isolation():
    cfg = EgoEstimatorConfig(att_model="eskf", vel_model="kf", rate_model="sampled")
    gp = torch.tensor([[8.0, 0.0, 2.0], [20.0, 4.0, 2.5]])
    est = BatchedEgoEstimator(6, gp, torch.zeros(2), config=cfg)
    p = torch.randn(6, 3, generator=torch.Generator().manual_seed(1))
    v = torch.randn(6, 3, generator=torch.Generator().manual_seed(2)) * 0.3
    q = torch.randn(6, 4, generator=torch.Generator().manual_seed(3))
    q = q / q.norm(dim=-1, keepdim=True)
    w = torch.zeros(6, 3)
    est.reset_idx(torch.arange(6), p, v, q)
    sf = torch.zeros(6, 3); sf[:, 2] = GRAVITY
    for _ in range(10):
        est.step(p, v, q, w, 1 / 30., detectable=torch.ones(6, 2, dtype=torch.bool),
                 apparent_area=torch.ones(6, 2), sf_body=sf)
    keep = [0, 2, 4]
    snap = (est._eskf.q[keep].clone(), est._eskf.b_g[keep].clone(), est._eskf.P[keep].clone(),
            est._eskf.R_ref[keep].clone(), est._navkf.x[keep].clone(), est._navkf.P[keep].clone(),
            est._gate_pos_datum[keep].clone(), est._Rz_neg_yaw0[keep].clone())
    idx = torch.tensor([1, 3, 5])
    est.reset_idx(idx, p[idx], v[idx], q[idx])
    after = (est._eskf.q[keep], est._eskf.b_g[keep], est._eskf.P[keep], est._eskf.R_ref[keep],
             est._navkf.x[keep], est._navkf.P[keep], est._gate_pos_datum[keep],
             est._Rz_neg_yaw0[keep])
    for a, b in zip(snap, after):
        assert torch.equal(a, b)                                           # untouched envs byte-safe
    # and the RESET envs' filter state is fresh (no cross-episode persistence)
    rp = _euler_roll_pitch_from_R(quat_xyzw_to_matrix(q[idx]))
    rp_seed = _euler_roll_pitch_from_R(quat_to_R_wxyz(est._eskf.q[idx]))
    assert torch.allclose(rp, rp_seed, atol=1e-5)


def quat_xyzw_to_matrix(q_xyzw):
    from inc8_estimator_emul import quat_xyzw_to_matrix_torch
    return quat_xyzw_to_matrix_torch(q_xyzw)


# ================================================================================================
# (INIT) deploy-faithful takeoff seeding.
# ================================================================================================
def test_init_seed_is_truth_roll_pitch_with_pad_converged_P():
    cfg = EgoEstimatorConfig(att_model="eskf", vel_model="kf", rate_model="sampled")
    gp = torch.tensor([[8.0, 0.0, 2.0]])
    est = BatchedEgoEstimator(16, gp, torch.zeros(1), config=cfg)
    g = torch.Generator().manual_seed(9)
    q = torch.randn(16, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    p = torch.randn(16, 3, generator=g)
    v = torch.randn(16, 3, generator=g)
    est.reset_idx(torch.arange(16), p, v, q)
    R_true = quat_xyzw_to_matrix(q)
    rp_true = _euler_roll_pitch_from_R(R_true)
    rp_seed = _euler_roll_pitch_from_R(quat_to_R_wxyz(est._eskf.q))
    assert torch.allclose(rp_true, rp_seed, atol=1e-5)                     # seed == truth EXACTLY
    # P0 == the measured pad-converged launch covariance built from the seeded g_hat
    g_world = torch.tensor(G_ZUP, dtype=est._eskf.dtype or torch.float32)
    g_hat = torch.einsum("mji,j->mi", quat_to_R_wxyz(est._eskf.q), g_world) / GRAVITY
    assert torch.allclose(est._eskf.P, eskf_p_launch(g_hat), atol=1e-9)
    # KF seed: datum pos 0, datum-frame truth velocity
    assert torch.equal(est._navkf.x[:, :3], torch.zeros(16, 3))
    vel_datum = torch.einsum("mij,mj->mi", est._Rz_neg_yaw0, v)
    assert torch.allclose(est._navkf.x[:, 3:], vel_datum, atol=1e-6)


# ================================================================================================
# (OFF) DEFAULT-OFF byte-identity + RNG draw-count invariance (graft #5).
# ================================================================================================
def _twin_replay_legacy_draws(seed, N, G, n_steps, dtype, *, skip_gyro_white=False,
                              dr_accel_bias=True, bias_model="legacy"):
    """Replay the DOCUMENTED legacy draw sequence on a twin generator; returns its final state.
    reset_idx: rand(m,G) n_eff; [dr_accel_bias] rand(m,3); [legacy bias] rand(m,G)+rand(m,G).
    step: [not sampled] randn(N,3) gyro-AR; [not kf] randn(N,3) white; randn(N,G,3) fix noise;
          rand(N,G) teleport; randn(N,G,3) wild; rand(N,G) missed; randn(N,G) normal ang;
          randn(N,G,3) axis; rand(N,G) flip; randn(N,G) area."""
    tg = torch.Generator()
    tg.manual_seed(seed)
    def randn(*s): torch.randn(*s, generator=tg, dtype=dtype)
    def rand(*s): torch.rand(*s, generator=tg, dtype=dtype)
    rand(N, G)                                     # n_eff
    if dr_accel_bias:
        rand(N, 3)                                 # accel bias
    if bias_model == "legacy":
        rand(N, G); rand(N, G)                     # mag + sign
    for _ in range(n_steps):
        if not skip_gyro_white:
            randn(N, 3)                            # gyro AR(1)
            randn(N, 3)                            # accel white
        randn(N, G, 3)                             # fix noise
        rand(N, G)                                 # teleport
        randn(N, G, 3)                             # wild
        rand(N, G)                                 # missed
        randn(N, G)                                # normal angle
        randn(N, G, 3)                             # axis
        rand(N, G)                                 # flip
        randn(N, G)                                # visible area
    return tg.get_state()


def _run_estimator(cfg, seed, n_steps=15):
    gen = torch.Generator(); gen.manual_seed(seed)
    N, G = 5, 2
    gp = torch.tensor([[6.0, 0.0, 2.0], [16.0, 3.0, 2.5]])
    est = BatchedEgoEstimator(N, gp, torch.zeros(G), config=cfg, generator=gen)
    p = torch.randn(N, 3, generator=torch.Generator().manual_seed(70))
    v = torch.randn(N, 3, generator=torch.Generator().manual_seed(71)) * 0.4
    q = torch.randn(N, 4, generator=torch.Generator().manual_seed(72))
    q = q / q.norm(dim=-1, keepdim=True)
    w = torch.randn(N, 3, generator=torch.Generator().manual_seed(73))
    est.reset_idx(torch.arange(N), p, v, q)
    sf = torch.zeros(N, 3); sf[:, 2] = GRAVITY
    out = None
    for k in range(n_steps):
        out = est.step(p + 0.05 * k, v, q, w, 1 / 30.,
                       detectable=torch.ones(N, G, dtype=torch.bool),
                       apparent_area=torch.full((N, G), 0.8),
                       sf_body=sf if (est._att_emul or est._vel_emul) else None)
    return est, out, gen.get_state(), (q, w, v)


def test_off_default_outputs_are_the_legacy_formulas_and_draw_count_invariant():
    est, out, gstate, (q, w, v) = _run_estimator(EgoEstimatorConfig(), seed=1001)
    # legacy roll/pitch == the PERFECT truth extraction (the measured-fatal lie, unchanged by default)
    assert torch.equal(out.roll_pitch, _euler_roll_pitch_from_R(quat_xyzw_to_matrix(q)))
    # RNG stream state == the twin replay of the documented legacy draw sequence (draw-count pin)
    twin = _twin_replay_legacy_draws(1001, 5, 2, 15, out.rel_pos.dtype)
    assert torch.equal(gstate, twin)


def test_off_faithful_path_owns_its_own_draw_count():
    cfg = EgoEstimatorConfig(att_model="eskf", vel_model="kf", rate_model="sampled")
    est, out, gstate, (q, w, v) = _run_estimator(cfg, seed=1001)
    # the faithful path draws NOTHING new -- it executes the legacy sequence MINUS gyro+white
    twin = _twin_replay_legacy_draws(1001, 5, 2, 15, out.rel_pos.dtype, skip_gyro_white=True)
    assert torch.equal(gstate, twin)
    # rates == the raw sample verbatim (measured-zero noise), roll_pitch != truth in general
    assert torch.equal(out.body_rates, w)


# ================================================================================================
# (COUP) velocity-attitude coupling.
# ================================================================================================
def test_coup_velocity_error_appears_exactly_with_the_lying_attitude():
    """Same inputs, two configs: att='gt'+vel='kf' stays near truth (no attitude corruption to
    inherit); att='eskf'+vel='kf' under a high-g intra-tick-varying profile accumulates velocity
    error THROUGH the shared lying R -- the deploy double-injection reproduced."""
    def run(att):
        cfg = EgoEstimatorConfig(att_model=att, vel_model="kf", rate_model="sampled",
                                 noise_scale=0.0)
        gp = torch.tensor([[10.0, 0.0, 2.0]])
        est = BatchedEgoEstimator(1, gp, torch.zeros(1), config=cfg)
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        est.reset_idx(torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1, 3), q)
        g = torch.Generator().manual_seed(21)
        vel_err = 0.0
        for k in range(40):
            w = torch.randn(1, 3, generator=g) * 3.0                       # aggressive rates
            d = torch.randn(1, 3, generator=g)
            sf = d / d.norm() * 3.0 * GRAVITY                              # 3 g regime
            out = est.step(torch.zeros(1, 3), torch.zeros(1, 3), q, w, 1 / 30.,
                           detectable=torch.zeros(1, 1, dtype=torch.bool),  # no fixes: pure DR
                           apparent_area=torch.zeros(1, 1), sf_body=sf)
            vel_err = float(torch.linalg.norm(out.velocity))               # truth v == 0
        return vel_err
    err_gt = run("gt")
    err_eskf = run("eskf")
    # both drift (strapdown at 3 g with truth pinned still is unphysical -- this test only
    # compares the CHANNELS): the lying-attitude arm must differ from the truth-attitude arm,
    # proving the projection/integration actually consumes the emulated R.
    assert abs(err_eskf - err_gt) > 1.0, (err_gt, err_eskf)


# ================================================================================================
# (WIRE) ENV WIRING -- stub-self execution + source pins (test_perception_honesty conventions).
# ================================================================================================
class _RecordingEstimatorSF:
    def __init__(self):
        self.calls = []

    def step(self, p, v, q, w, dt, detectable=None, prev_quat=None, apparent_area=None,
             blur_extra_miss=None, sf_body=None, gyro_sample=None):
        self.calls.append({"sf_body": None if sf_body is None else sf_body.clone(),
                           "w": w.clone()})
        return "EST-SENTINEL"


def _mk_sf_stub(needs_sf, sf_tensor):
    if not PRE._HAVE_DIFFAERO:
        pytest.skip("peregrine_racing base / inc8_estimator_emul not importable here")
    n = 2
    stub = object.__new__(PRE.PeregrineRacingEgo)
    stub._p = torch.zeros(n, 3, dtype=DT64)
    stub._q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DT64).expand(n, 4).contiguous()
    stub._v = torch.zeros(n, 3, dtype=DT64)
    stub._w = torch.zeros(n, 3, dtype=DT64)
    stub.gate_pos = torch.tensor([[10.0, 0.0, 0.0]], dtype=DT64).expand(n, 3).reshape(n, 1, 3).contiguous()
    stub.gate_yaw = torch.zeros(n, 1, dtype=DT64)
    stub.dt = 1.0 / 30.0
    stub._cam_flip = False
    stub._ego_cfg = SimpleNamespace(far_cap_m=30.0)
    stub._blur_gate = False
    stub._est_needs_sf = needs_sf
    stub.dynamics = SimpleNamespace(_sf_body_flu=sf_tensor)
    stub._estimator = _RecordingEstimatorSF()
    return stub


def test_wire_step_estimator_passes_the_captured_sf_when_armed():
    marker = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DT64)
    stub = _mk_sf_stub(True, marker)
    est, _ = PRE.PeregrineRacingEgo._step_estimator(stub, stub._q.clone())
    assert est == "EST-SENTINEL"
    rec = stub._estimator.calls[-1]
    assert rec["sf_body"] is not None and torch.equal(rec["sf_body"], marker)


def test_wire_step_estimator_passes_none_when_off_and_raises_without_capture():
    stub = _mk_sf_stub(False, None)
    PRE.PeregrineRacingEgo._step_estimator(stub, stub._q.clone())
    assert stub._estimator.calls[-1]["sf_body"] is None                    # default-off: None
    stub2 = _mk_sf_stub(True, None)
    with pytest.raises(RuntimeError, match="capture_specific_force"):     # L16: loud, never a no-op
        PRE.PeregrineRacingEgo._step_estimator(stub2, stub2._q.clone())


def _mk_get_state_stub(att_model, rp_marker):
    if not PRE._HAVE_DIFFAERO:
        pytest.skip("peregrine_racing base / inc8_estimator_emul not importable here")
    n, G = 3, 2
    stub = object.__new__(PRE.PeregrineRacingEgo)
    stub._ego_on = True
    g = torch.Generator().manual_seed(31)
    q = torch.randn(n, 4, generator=g)
    stub._q = q / q.norm(dim=-1, keepdim=True)
    stub._p = torch.randn(n, 3, generator=g)
    stub._v = torch.randn(n, 3, generator=g)
    stub._w = torch.randn(n, 3, generator=g)
    stub.gate_pos = torch.randn(n, G, 3, generator=g) * 5
    stub.target_gates = torch.zeros(n, dtype=torch.long)
    stub.n_gates = G
    stub._est_att_model = att_model
    est = SimpleNamespace(roll_pitch=rp_marker, confidence=torch.full((n, G), 0.5))
    stub._estimator = SimpleNamespace(estimate=lambda: est)
    return stub


def test_wire_critic_truth_source_is_knob_gated():
    """get_state slot [3:5]: 'gt' -> est.roll_pitch verbatim (byte-identical legacy, incl. the
    cold-reset ordering edge); emulated -> a DIRECT truth extraction (the frozen privileged critic
    never sees the lying leveler -- the silent-critic-corruption trap disarmed)."""
    marker = torch.full((3, 2), 0.123456)
    s_gt = _mk_get_state_stub("gt", marker)
    state_gt = PRE.PeregrineRacingEgo.get_state(s_gt)
    assert torch.equal(state_gt[:, 3:5], marker)                           # legacy: est passthrough
    s_em = _mk_get_state_stub("eskf", marker)
    state_em = PRE.PeregrineRacingEgo.get_state(s_em)
    rp_true = _euler_roll_pitch_from_R(quat_xyzw_to_matrix(s_em._q))
    assert torch.allclose(state_em[:, 3:5], rp_true, atol=1e-6)            # emulated: TRUTH direct
    assert not torch.allclose(state_em[:, 3:5], marker)


def test_wire_source_pins():
    src_se = inspect.getsource(PRE.PeregrineRacingEgo._step_estimator)
    assert "self._dyn_sf_body_flu() if self._est_needs_sf else None" in src_se
    assert "sf_body=sf_body" in src_se
    src_gs = inspect.getsource(PRE.PeregrineRacingEgo.get_state)
    assert 'self._est_att_model == "gt"' in src_gs
    assert "_euler_roll_pitch_from_R(R_wb)" in src_gs
    src_step = inspect.getsource(PRE.PeregrineRacingEgo.step)
    for key in ("eskf_tilt_err_deg_mean", "eskf_tilt_err_deg_p90", "eskf_accel_update_duty",
                "kf_vel_err_mean", "sf_mag_g_mean"):
        assert key in src_step, key                                        # the L16 precheck keys


def test_wire_default_off_config_builds_no_subfilters():
    est = BatchedEgoEstimator(2, torch.tensor([[5.0, 0.0, 2.0]]), torch.zeros(1),
                              config=EgoEstimatorConfig())
    assert est._eskf is None and est._navkf is None
    assert not est._att_emul and not est._vel_emul and not est._rate_sampled


# ================================================================================================
# (ACC) ACCEPTANCE -- fixture replay (runs everywhere; the committed a5 flight ticks).
# ================================================================================================
@pytest.mark.skipif(not FIXTURE.exists(), reason="a5 fixture not present")
def test_acc_wire_replay_parity_and_divergence_bands_from_fixture():
    from leveler_bench import load_from_fixture, mode_a_wire_replay, divergence_band
    t_us, acc, gyr, ticks = load_from_fixture(FIXTURE)
    a = mode_a_wire_replay(t_us, acc, gyr, ticks)
    # translation parity: single-step re-anchored median == the wire's own tickrate check (~3e-4)
    assert a["attitude_tilt_vs_wire_REANCHORED"]["median_deg"] < 0.02
    # KF dead-reckoning parity (map-free flight: zero fixes)
    assert a["kf_pos_vs_wire_m_free_running"]["median"] < 0.10
    tick_t = np.array([tk["t_us"] for tk in ticks])
    r0, p0 = ticks[0]["obs"][3], ticks[0]["obs"][4]
    # recorded-grid band == the measured wire benchmark envelope (median 4-10 / p90 15-35 / max>25)
    b = divergence_band(t_us, acc, gyr, tick_t, r0, p0)
    assert 3.5 <= b["median_deg"] <= 10.0, b
    assert 15.0 <= b["p90_deg"] <= 35.0, b
    assert b["max_deg"] > 25.0, b
    assert b["sf_mag_g"]["median"] >= 2.5                                  # matched-|f| gate
    # 33.3 ms training-dt band (the in-env T3 pass band): nonzero but smaller than the 57 ms band
    grid33 = np.arange(tick_t[0], tick_t[-1], 1e6 / 30.0)
    b33 = divergence_band(t_us, acc, gyr, grid33, r0, p0)
    assert 0.8 <= b33["median_deg"] <= 6.0, b33
    assert b33["p90_deg"] <= 20.0, b33
    # dense-stepping control: the non-aliasing floor is several-fold smaller than the tick band
    bd = divergence_band(t_us, acc, gyr, grid33, r0, p0, dense=True)
    assert bd["median_deg"] < 0.5 * b33["median_deg"], (bd, b33)
