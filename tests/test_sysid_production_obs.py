"""SYS-ID #6 -- PRODUCTION-vs-TRAINED obs registration (laptop arm, 2026-06-18).

Registers the deploy obs builder (src/racer/estimator_obs.py) against the trained 20-dim contract
(rl/fly_rl.build_obs/obs_from_zup [0:17] + the d5 confidence triple [17:20]). Pins what was MEASURED:

  R1 (inc7 BYTE-IDENTITY): under MATCHED state (estimator pos/vel == wire pos/vel) the deploy seam
     estimator_state_for_obs(ds, nav) -> build_obs is element-wise BYTE-IDENTICAL to the trained
     build_obs over random pose/vel/attitude/gate/thrust. obs[0:17] delta == 0.0 exactly. This is the
     inc7 (obs_dim 17) contract -- it must stay byte-identical.

  R2 (the obs[17:20] gap CLOSED -- present + faithful): the production confidence-triple builder is NOW
     promoted into src/racer/estimator_obs.py (`confidence_triple` / `confidence_triple_from_sigmas` /
     `estimator_obs20`). This test asserts it is PRESENT and BYTE-FAITHFUL: driven on the same gate-frame
     covariance + fix clock as the trained encoder (EstimatorEmulator.confidence_channel) it agrees to
     <= 1e-6 (float32-exact). [GAP CLOSED 2026-06-19; the full sweep + edge cases live in
     tests/test_deploy_obs20.py -- this is the system-id registration counterpart.]

  R3 (the sigma sqrt(2) FOOTGUN -- INTENTIONAL, do NOT reconcile): the navigator's NavState export
     nav_inplane_sigma = sqrt(P_E + P_D) is sqrt(2)x the emulator/spike in-plane sigma_hat
     = sqrt((P_E + P_D)/2). A production obs[17:20] builder fed NavState.nav_inplane_sigma RAW into the
     d5 encoding clip(sigma_ref/sigma_hat) would mis-report c_inplane by up to ~0.29; dividing by sqrt(2)
     recovers the trained value to machine epsilon. Pinned here so the relationship is explicit for
     whoever promotes the builder. (The deploy spike sidesteps this by re-projecting the raw KF P, not
     consuming the NavState export.)

Drives BOTH sides with identical state, isolating the obs layer (matched-state -> no estimator
divergence; the obs builder is the only thing under test). build_obs pulls torch -> importorskip.
[SYS-ID 2026-06-18 production-vs-trained-obs]
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("torch")   # fly_rl imports torch at module top

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

from fly_rl import _GATE_POS_ZUP, _GATE_YAW_ZUP, build_obs, make_gate_map  # noqa: E402
from racer.contracts import DroneState, NavState  # noqa: E402
from racer.estimator_obs import (  # noqa: E402
    confidence_triple,
    estimator_obs,
    estimator_obs20,
    estimator_state_for_obs,
)

_SRC_RACER = Path(__file__).resolve().parents[1] / "src" / "racer"
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _ds(pos, vel, q=_LEVEL_Q):
    return DroneState(sim_time_ns=0, orientation_ned_wxyz=np.asarray(q, float),
                      angular_rate_body=np.zeros(3), accel_body=np.array([0.0, 0.0, -9.80665]),
                      position_ned=np.asarray(pos, float), velocity_ned=np.asarray(vel, float))


def _nav(pos, vel):
    return NavState(sim_time_ns=0, position_ned=np.asarray(pos, float),
                    velocity_ned=np.asarray(vel, float))


def test_r1_inc7_deploy_equals_trained_byte_identical():
    """R1: deploy seam == trained build_obs, ELEMENT-WISE BYTE-IDENTICAL under matched state.

    The deploy obs builder is a PURE pos/vel substitution into the canonical builder; with the
    estimator pos/vel set EQUAL to the wire (matched state) the obs must be byte-identical to the
    shipped 17-dim build_obs the inc7 policy trained on. Any layout/sign drift trips here."""
    rng = np.random.default_rng(7)
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    max_abs = 0.0
    for _ in range(400):
        pos = rng.normal(0, 25, 3)
        vel = rng.normal(0, 8, 3)
        q = Rotation.random(random_state=rng).as_quat()        # xyzw
        ds = _ds(pos, vel, q=[q[3], q[0], q[1], q[2]])
        tg = int(rng.integers(0, len(_GATE_POS_ZUP)))
        lnt = float(rng.uniform(0, 2))
        trained = build_obs(ds, tg, lnt, gate_map=gm)
        deploy = estimator_obs(ds, _nav(pos, vel), tg, lnt, gate_map=gm)
        assert trained.shape == (17,) and deploy.shape == (17,), "inc7 obs must be 17-dim"
        np.testing.assert_array_equal(deploy, trained)         # byte-identical (obs[0:17] delta 0.0)
        max_abs = max(max_abs, float(np.max(np.abs(
            deploy.astype(np.float64) - trained.astype(np.float64)))))
    assert max_abs == 0.0, f"deploy 17-dim diverged from trained by {max_abs:.3e} (must be 0.0)"


def _emul_triple_for(P, t_since_fix, target_gate=0):
    """The trained obs[17:20] for a controlled gate-frame KF cov + fix clock: the encoder the inc8 actor
    was actually selected on (EstimatorEmulator.confidence_channel, VQ1 all-pi gate frame)."""
    from estimator_emul import EstimatorEmulator, EmulConfig
    emu = EstimatorEmulator(EmulConfig())

    class _S:
        pos = np.zeros(3)
        vel = np.zeros(3)

    emu.reset(_S(), target_gate, np.random.default_rng(0))
    emu.kf.P[:3, :3] = np.asarray(P, dtype=np.float64)
    emu._t_since_fix = float(t_since_fix)
    return np.asarray(emu.confidence_channel(target_gate), dtype=np.float64), emu.gates[target_gate].R_world_gate


def test_r2_production_obs1720_builder_present_and_faithful():
    """R2 (GAP CLOSED): a production obs[17:20]/confidence-triple BUILDER now exists in src/racer and is
    byte-faithful to the trained encoder.

    (a) PRESENT: estimator_obs exposes the confidence-triple builders + the 20-dim assembler.
    (b) FAITHFUL: production `confidence_triple` (NavState contract) == the trained
        EstimatorEmulator.confidence_channel on the SAME gate-frame cov + fix clock, to <= 1e-6
        (float32-exact) -- with the sqrt(2) reconciliation applied in the builder (R3 stays the
        un-reconciled wire pin). Drives both on identical inputs, isolating the obs[17:20] layer."""
    import racer.estimator_obs as eo
    public = {n for n in dir(eo) if not n.startswith("_")}
    for name in ("confidence_triple", "confidence_triple_from_sigmas", "estimator_obs20", "estimator_obs_auto"):
        assert name in public, f"production obs[17:20] builder '{name}' missing from src/racer.estimator_obs"

    # the 20-dim assembler runs end-to-end and is 17-dim ++ triple
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    _ds0 = _ds([1.0, 2.0, -3.0], [0.5, 0.0, 0.0])
    _nav0 = NavState(sim_time_ns=0, position_ned=np.array([1.0, 2.0, -3.0]),
                     velocity_ned=np.array([0.5, 0.0, 0.0]),
                     nav_inplane_sigma=0.07, nav_along_sigma=0.08, time_since_vision_update_s=0.03)
    _o20 = estimator_obs20(_ds0, _nav0, 0, 0.4, gate_map=gm)
    assert _o20.shape == (20,) and _o20.dtype == np.float32
    np.testing.assert_array_equal(_o20[:17], estimator_obs(_ds0, _nav0, 0, 0.4, gate_map=gm))

    rng = np.random.default_rng(619)
    worst = 0.0
    for scale in (0.02, 0.07, 0.2):
        for _ in range(40):
            A = rng.normal(0.0, scale, (3, 3))
            P = A @ A.T + np.eye(3) * 1e-9                       # SPD world-NED position cov
            t = float(rng.uniform(0.0, 0.25))
            emul_triple, Rwg = _emul_triple_for(P, t)
            P_gate = Rwg.T @ P @ Rwg
            nav_ip = float(np.sqrt(max(P_gate[0, 0] + P_gate[1, 1], 0.0)))   # navigator export (sum-sqrt)
            nav_al = float(np.sqrt(max(P_gate[2, 2], 0.0)))
            nav = NavState(sim_time_ns=0, nav_inplane_sigma=nav_ip, nav_along_sigma=nav_al,
                           time_since_vision_update_s=t)
            prod = confidence_triple(nav).astype(np.float64)
            worst = max(worst, float(np.max(np.abs(prod - emul_triple))))
    assert worst <= 1e-6, f"production confidence_triple diverged from the trained encoder by {worst:.3e}"


def test_r3_navstate_sigma_is_sqrt2_times_emul_hat_intentional():
    """R3: PIN the INTENTIONAL sqrt(2) gap (do NOT reconcile).

    navigator._gate_frame_pos_sigma / NavState.nav_inplane_sigma = sqrt(P_E + P_D);
    emulator/spike sigma_inplane_hat                              = sqrt((P_E + P_D)/2).
    Ratio is EXACTLY sqrt(2). Consequence: feeding the NavState export RAW into the d5 encoding
    clip(sigma_ref/sigma_hat) mis-reports c_inplane (here >0.05 worst-case); dividing by sqrt(2) first
    recovers the TRAINED value to ~0. Both pinned so the promotion path is unambiguous."""
    sigma_ref = 0.05
    rng = np.random.default_rng(3)
    worst_ratio_err = 0.0
    worst_c_raw_err = 0.0
    worst_c_corr_err = 0.0
    for _ in range(500):
        a = float(rng.uniform(0.0005, 0.02))   # P_E (gate-frame in-plane axis 0 variance)
        b = float(rng.uniform(0.0005, 0.02))   # P_D (gate-frame in-plane axis 1 variance)
        nav_sigma = np.sqrt(a + b)                       # navigator NavState export (navigator.py:626)
        emul_sigma = np.sqrt(0.5 * (a + b))              # emul/spike sigma_inplane_hat
        worst_ratio_err = max(worst_ratio_err, abs(nav_sigma / emul_sigma - np.sqrt(2.0)))
        c_true = np.clip(sigma_ref / emul_sigma, 0, 1)
        c_raw = np.clip(sigma_ref / nav_sigma, 0, 1)            # naive: NavState export raw
        c_corr = np.clip(sigma_ref / (nav_sigma / np.sqrt(2.0)), 0, 1)   # corrected: /sqrt(2)
        worst_c_raw_err = max(worst_c_raw_err, abs(c_raw - c_true))
        worst_c_corr_err = max(worst_c_corr_err, abs(c_corr - c_true))
    assert worst_ratio_err < 1e-12, (
        f"NavState/emul in-plane sigma ratio drifted from sqrt(2) by {worst_ratio_err:.3e}")
    assert worst_c_raw_err > 0.05, (
        "the sqrt(2) gap should MEASURABLY perturb c_inplane if NavState export is fed raw "
        f"(worst err {worst_c_raw_err:.3e}); footgun no longer load-bearing?")
    assert worst_c_corr_err < 1e-9, (
        f"dividing NavState.nav_inplane_sigma by sqrt(2) must recover the trained c_inplane "
        f"(worst err {worst_c_corr_err:.3e})")
