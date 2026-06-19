"""Production obs[17:20] confidence channel -- the deploy-side 20-dim obs builder (deploy-obs20 2026-06-19).

Promotes ``rl/spike_vertical_slice.deploy_confidence_triple`` into ``src/racer/estimator_obs`` as the
PRODUCTION confidence-channel builder and pins it BYTE-FAITHFUL (laptop counterpart of the Adroit-deferred
train<->deploy elementwise parity test). The deploy stack now emits the FULL 20-dim obs the inc8 policy was
trained on, sourced from the NavState contract (Path A) -- the /sqrt(2) reconciliation applied in the obs
builder, not the navigator export (test_sysid_production_obs.py R3 stays the un-reconciled wire pin).

WHAT THIS PINS:
  1. CONSTANTS: estimator_obs.SIGMA_REF_M / TAU_STALE_S == the trainer's (estimator_emul) == 0.05 / 0.10.
  2. THREE-WAY BUILDER PARITY (the core fidelity claim): production ``confidence_triple`` (NavState) ==
     spike ``deploy_confidence_triple`` (raw P) == trained ``EstimatorEmulator.confidence_channel`` to
     <= 1e-6 (float32-exact 0.0) across a pose/cov/staleness/gate-frame sweep.
  3. FULL 20-dim PARITY: production ``estimator_obs20`` == the trained 20-dim obs (build_obs[0:17] ++
     emul triple) element-wise: obs[0:17] delta == 0.0 (byte-identical inc7 seam) AND obs[17:20] <= 1e-6.
  4. inc7 17-dim path BYTE-IDENTICAL (the JUDGED VQ1 path -- 20-dim is strictly opt-in).
  5. EDGE CASES: cold-start / no-fix (sigma=inf -> [0,0,1]); stale-fix age ramp; degenerate / over-
     converged cov (sigma -> 0 -> c=1, the inplane-floor-OFF #74 regime); the last-fix gate frame
     (production-from-NavState == spike-from-raw-P on a REAL Navigator run).
  6. DISPATCH: estimator_obs_auto picks 20-dim for a 20-wide actor, 17-dim otherwise.
  7. The confidence builders import + run TORCH-FREE (G0; subprocess).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_deploy_obs20.py -q
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.contracts import DroneState, NavState  # noqa: E402
from racer.estimator_obs import (  # noqa: E402
    SIGMA_REF_M,
    TAU_STALE_S,
    confidence_triple,
    confidence_triple_from_sigmas,
    estimator_obs,
    estimator_obs20,
    estimator_obs_auto,
)

torch = pytest.importorskip("torch")  # estimator_emul / fly_rl import torch at module top

from estimator_emul import (  # noqa: E402
    SIGMA_REF_M as EMUL_SIGMA_REF,
    TAU_STALE_S as EMUL_TAU_STALE,
    EmulConfig,
    EstimatorEmulator,
    ned_gate_frame,
)
from fly_rl import (  # noqa: E402
    N_GATES,
    _GATE_POS_ZUP,
    _GATE_YAW_ZUP,
    build_obs,
    make_gate_map,
)
from spike_vertical_slice import deploy_confidence_triple  # noqa: E402

_TOL = 1e-6          # float32-exact bar (the d5 obs[17:20] are float32 the actor consumed)
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- helpers
def _spd_pos_cov(rng, scale):
    """A random SPD 3x3 world-NED position covariance at a given magnitude scale (sigmas ~ scale)."""
    A = rng.normal(0.0, scale, (3, 3))
    return A @ A.T + np.eye(3) * 1e-9


def _emul_confidence(P, yaw, t, tg=0):
    """Drive the REAL trained encoder: set the emul's KF position cov + fix clock + gate-frame yaw and
    return (triple_float64, Rwg). This is the obs[17:20] the inc8 actor was selected on."""
    emu = EstimatorEmulator(EmulConfig(), gate_yaw=np.full(N_GATES, float(yaw)))

    class _S:
        pos = np.zeros(3)
        vel = np.zeros(3)

    emu.reset(_S(), tg, np.random.default_rng(0))
    emu.kf.P[:3, :3] = np.asarray(P, dtype=np.float64)
    emu._t_since_fix = float(t)
    triple = np.asarray(emu.confidence_channel(tg), dtype=np.float64)
    return triple, emu.gates[tg].R_world_gate


def _navstate_sigmas_from_P(P, Rwg):
    """The navigator's NavState export from a world-NED position cov projected into gate frame Rwg
    (navigator._gate_frame_pos_sigma: in-plane = sqrt(P_E+P_D) (sum), along = sqrt(P_along))."""
    P_gate = Rwg.T @ np.asarray(P, dtype=np.float64) @ Rwg
    nav_ip = float(np.sqrt(max(P_gate[0, 0] + P_gate[1, 1], 0.0)))
    nav_al = float(np.sqrt(max(P_gate[2, 2], 0.0)))
    return nav_ip, nav_al


def _ds(pos, vel, q=_LEVEL_Q):
    return DroneState(sim_time_ns=0, orientation_ned_wxyz=np.asarray(q, float),
                      angular_rate_body=np.zeros(3), accel_body=np.array([0.0, 0.0, -9.80665]),
                      position_ned=np.asarray(pos, float), velocity_ned=np.asarray(vel, float))


def _nav(pos, vel, nav_ip=float("inf"), nav_al=float("inf"), tsv=float("inf")):
    return NavState(sim_time_ns=0, position_ned=np.asarray(pos, float),
                    velocity_ned=np.asarray(vel, float),
                    nav_inplane_sigma=nav_ip, nav_along_sigma=nav_al,
                    time_since_vision_update_s=tsv)


# --------------------------------------------------------------------------- 1. constants
def test_constants_pinned_equal_across_modules():
    """estimator_obs's d5 constants equal the trainer's encoder constants AND the navigator state floor
    (no silent drift across the obs builder / emulator / navigator boundary)."""
    from racer.navigator import INPLANE_POS_FLOOR_STD
    assert SIGMA_REF_M == EMUL_SIGMA_REF == 0.05
    assert TAU_STALE_S == EMUL_TAU_STALE == 0.10
    assert INPLANE_POS_FLOOR_STD == SIGMA_REF_M    # the sigma_b state floor aliases the confidence sigma_ref
    cfg = EmulConfig()
    assert cfg.sigma_ref == SIGMA_REF_M and cfg.tau_stale == TAU_STALE_S


# --------------------------------------------------------------------------- 2. three-way builder parity
def test_three_way_builder_parity_sweep():
    """production confidence_triple (NavState) == spike deploy_confidence_triple (raw P) ==
    trained EstimatorEmulator.confidence_channel, to <= 1e-6, across cov scale / staleness / gate yaw.

    All three encode the SAME P / gate frame / fix clock; agreement proves the production builder is the
    byte-faithful promotion of the spike AND the trained encoder (the #37 fidelity claim, productionized)."""
    rng = np.random.default_rng(20260619)
    worst_prod_emul = worst_prod_spike = worst_spike_emul = 0.0
    n = 0
    for scale in (0.005, 0.02, 0.05, 0.1, 0.25, 0.5):       # tight (c->1) ... loose (c small)
        for _ in range(60):
            P = _spd_pos_cov(rng, scale)
            yaw = float(rng.uniform(-np.pi, np.pi))
            t = float(rng.uniform(0.0, 0.3))                # fresh (age 0) ... stale (age 1)
            emul_triple, Rwg = _emul_confidence(P, yaw, t)
            spike_triple, _ = deploy_confidence_triple(P, Rwg, t)
            nav_ip, nav_al = _navstate_sigmas_from_P(P, Rwg)
            prod_triple = confidence_triple(_nav(np.zeros(3), np.zeros(3), nav_ip, nav_al, t))

            e = emul_triple.astype(np.float64)
            s = spike_triple.astype(np.float64)
            p = prod_triple.astype(np.float64)
            worst_prod_emul = max(worst_prod_emul, float(np.max(np.abs(p - e))))
            worst_prod_spike = max(worst_prod_spike, float(np.max(np.abs(p - s))))
            worst_spike_emul = max(worst_spike_emul, float(np.max(np.abs(s - e))))
            # bounds sanity
            assert np.all(prod_triple >= 0.0) and np.all(prod_triple <= 1.0)
            n += 1
    assert n >= 300
    assert worst_prod_emul <= _TOL, f"production vs trained emul worst {worst_prod_emul:.3e}"
    assert worst_prod_spike <= _TOL, f"production vs spike worst {worst_prod_spike:.3e}"
    assert worst_spike_emul <= _TOL, f"spike vs trained emul worst {worst_spike_emul:.3e}"


def test_from_sigmas_core_matches_spike_bitexact():
    """confidence_triple_from_sigmas fed the EMUL-convention sigmas reproduces the spike's triple
    bit-exactly (same clip/guard arithmetic on the same inputs)."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for scale in (0.01, 0.05, 0.3):
        for _ in range(40):
            P = _spd_pos_cov(rng, scale)
            yaw = float(rng.uniform(-np.pi, np.pi))
            t = float(rng.uniform(0.0, 0.2))
            Rwg = ned_gate_frame(yaw)
            spike_triple, (sig_ip, sig_al) = deploy_confidence_triple(P, Rwg, t)
            core = confidence_triple_from_sigmas(sig_ip, sig_al, t)
            worst = max(worst, float(np.max(np.abs(core.astype(np.float64) - spike_triple.astype(np.float64)))))
    assert worst == 0.0, f"core vs spike must be bit-identical, got {worst:.3e}"


# --------------------------------------------------------------------------- 3. full 20-dim parity
def test_full_obs20_equals_trained_elementwise():
    """estimator_obs20 (deploy) == trained 20-dim obs element-wise: obs[0:17] delta == 0.0 (byte-identical
    inc7 seam under matched state) AND obs[17:20] <= 1e-6 -- across pose / vel / attitude / cov / staleness."""
    rng = np.random.default_rng(2026)
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    Rwg = ned_gate_frame(np.pi)                              # VQ1 course is all-pi
    worst_017 = worst_1720 = 0.0
    for _ in range(120):
        pos = rng.normal(0, 25, 3)
        vel = rng.normal(0, 8, 3)
        q = Rotation.random(random_state=rng).as_quat()     # xyzw -> wxyz
        ds = _ds(pos, vel, q=[q[3], q[0], q[1], q[2]])
        tg = int(rng.integers(0, N_GATES))
        lnt = float(rng.uniform(0, 2))
        P = _spd_pos_cov(rng, float(rng.uniform(0.01, 0.4)))
        t = float(rng.uniform(0.0, 0.25))

        emul_triple, _ = _emul_confidence(P, np.pi, t, tg)
        trained20 = np.concatenate(
            [build_obs(ds, tg, lnt, gate_map=gm).astype(np.float32),
             emul_triple.astype(np.float32)]).astype(np.float32)

        nav_ip, nav_al = _navstate_sigmas_from_P(P, Rwg)
        nav = _nav(pos, vel, nav_ip, nav_al, t)             # matched pos/vel -> obs[0:17] byte-identical
        deploy20 = estimator_obs20(ds, nav, tg, lnt, gate_map=gm)

        assert deploy20.shape == (20,) and deploy20.dtype == np.float32
        d017 = float(np.max(np.abs(deploy20[:17].astype(np.float64) - trained20[:17].astype(np.float64))))
        d1720 = float(np.max(np.abs(deploy20[17:].astype(np.float64) - trained20[17:].astype(np.float64))))
        worst_017 = max(worst_017, d017)
        worst_1720 = max(worst_1720, d1720)
    assert worst_017 == 0.0, f"obs[0:17] must be byte-identical, got {worst_017:.3e}"
    assert worst_1720 <= _TOL, f"obs[17:20] diverged by {worst_1720:.3e}"


def test_inc7_17dim_path_byte_identical():
    """The inc7 17-dim deploy seam is UNCHANGED -- estimator_obs == build_obs under matched state
    (the JUDGED VQ1 path; the 20-dim channel is strictly additive)."""
    rng = np.random.default_rng(17)
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    for _ in range(40):
        pos = rng.normal(0, 20, 3)
        vel = rng.normal(0, 6, 3)
        q = Rotation.random(random_state=rng).as_quat()
        ds = _ds(pos, vel, q=[q[3], q[0], q[1], q[2]])
        tg = int(rng.integers(0, N_GATES))
        lnt = float(rng.uniform(0, 2))
        direct = build_obs(ds, tg, lnt, gate_map=gm)
        wired = estimator_obs(ds, _nav(pos, vel), tg, lnt, gate_map=gm)
        assert wired.shape == (17,)
        np.testing.assert_array_equal(wired, direct)


# --------------------------------------------------------------------------- 4. edge cases
def test_cold_start_no_fix_yet():
    """No accepted fix yet: navigator exports inf sigmas + inf fix-clock -> triple == [0, 0, 1]
    (zero confidence, fully stale). Matches the d5 encoder fed an unbounded covariance."""
    triple = confidence_triple(_nav(np.zeros(3), np.zeros(3)))   # all defaults = inf
    assert triple.dtype == np.float32
    np.testing.assert_array_equal(triple, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_cold_start_train_deploy_divergence_documented():
    """CHARACTERIZATION of the KNOWN, BOUNDED pre-first-fix cold-start gap (review 2026-06-19).

    The deploy navigator has NO gate frame until the first accepted fix -> exports inf -> triple [0,0,1];
    the TRAINED emul's cold KF carries a FINITE P=I (pos_std=1.0, no step yet) -> triple [0.05,0.05,1].
    So obs[17:18] read 0.0 (deploy) vs ~0.05 (trained) in the pre-acquisition window ONLY. This is an
    ESTIMATOR-STATE difference (the builder faithfully encodes whatever sigma it is handed), NOT a builder
    bug -- byte-faithfulness holds AFTER the first fix. Pinned so the divergence is explicit, not silently
    untested. age_norm==1 in BOTH (the channel is flagged stale); the gap closes on the first fix."""
    emu = EstimatorEmulator(EmulConfig())

    class _S:
        pos = np.zeros(3)
        vel = np.zeros(3)

    emu.reset(_S(), 0, np.random.default_rng(0))                 # cold KF, no step -> P=I, t_since=1e3
    trained_cold = np.asarray(emu.confidence_channel(0), dtype=np.float64)
    np.testing.assert_allclose(trained_cold, [0.05, 0.05, 1.0], atol=1e-9)

    deploy_cold = confidence_triple(_nav(np.zeros(3), np.zeros(3))).astype(np.float64)   # inf -> [0,0,1]
    np.testing.assert_array_equal(deploy_cold, [0.0, 0.0, 1.0])

    # the divergence is exactly the emul's cold c (~0.05) on the two confidence axes; age_norm matches.
    np.testing.assert_allclose(np.abs(trained_cold[:2] - deploy_cold[:2]), [0.05, 0.05], atol=1e-9)
    assert trained_cold[2] == deploy_cold[2] == 1.0             # BOTH flag the channel fully stale


def test_stale_fix_age_ramp():
    """age_norm = clip(t_since_fix / TAU_STALE, 0, 1): linear in [0, tau], saturates at 1 beyond tau."""
    nav_ip, nav_al = 0.07, 0.08
    for t, expect in [(0.0, 0.0), (0.05, 0.5), (0.10, 1.0), (0.5, 1.0), (1e3, 1.0)]:
        triple = confidence_triple(_nav(np.zeros(3), np.zeros(3), nav_ip, nav_al, t))
        assert abs(float(triple[2]) - expect) <= 1e-6, (t, float(triple[2]))


def test_degenerate_overconverged_cov_reads_max_confidence():
    """sigma -> 0 (over-converged cov, e.g. the inplane-floor OFF #74 regime) -> c == 1.0 (the degenerate
    MAX-confidence guard), matching the emul/spike `if sigma > 0 else 1.0`. Also small-but-positive
    sigma below sigma_ref clips to 1.0."""
    # exact-zero sigma -> 1.0 on the guard
    triple0 = confidence_triple(_nav(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(triple0[:2], np.array([1.0, 1.0], dtype=np.float32))
    # sigma below sigma_ref (tight) -> clip to 1.0; the /sqrt(2) keeps the in-plane guard correct
    nav_ip = SIGMA_REF_M * np.sqrt(2.0) * 0.5     # -> sigma_inplane_hat = 0.5*sigma_ref < sigma_ref
    triple1 = confidence_triple(_nav(np.zeros(3), np.zeros(3), nav_ip, SIGMA_REF_M * 0.5, 0.0))
    assert float(triple1[0]) == 1.0 and float(triple1[1]) == 1.0


def test_production_from_navstate_equals_spike_on_real_navigator():
    """END-TO-END last-fix gate-frame path: drive the REAL case-C Navigator to convergence, then the
    production triple from its NavState (Path A, contract-only) == the spike triple from the raw KF P +
    last-fix gate frame (Path B). Proves Path A reads the right gate-frame-projected sigmas off NavState
    and reconciles the sqrt(2) correctly on genuine estimator state (not just synthetic covariances)."""
    from spike_vertical_slice import drive_navigator
    nav, gate, _, ns = drive_navigator()
    spike_triple, _ = deploy_confidence_triple(nav.kf.P[:3, :3], gate.R_world_gate,
                                               ns.time_since_vision_update_s)
    prod_triple = confidence_triple(ns)
    assert np.max(np.abs(prod_triple.astype(np.float64) - spike_triple.astype(np.float64))) <= _TOL
    # the NavState really did anchor a gate frame (finite sigmas) -> this exercised the live export path
    assert np.isfinite(ns.nav_inplane_sigma) and np.isfinite(ns.nav_along_sigma)


# --------------------------------------------------------------------------- 5. dispatch seam
def test_estimator_obs_auto_dispatch_by_width():
    """estimator_obs_auto returns 17-dim for an inc7-width actor and 20-dim for an inc8-width actor,
    and the 20-dim head matches the 17-dim path byte-for-byte (purely additive)."""
    rng = np.random.default_rng(3)
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    pos, vel = rng.normal(0, 10, 3), rng.normal(0, 4, 3)
    ds = _ds(pos, vel)
    nav = _nav(pos, vel, 0.07, 0.08, 0.03)
    o17 = estimator_obs_auto(ds, nav, 2, 0.4, 17, gate_map=gm)
    o20 = estimator_obs_auto(ds, nav, 2, 0.4, 20, gate_map=gm)
    assert o17.shape == (17,) and o20.shape == (20,)
    np.testing.assert_array_equal(o20[:17], o17)            # 20-dim head == the 17-dim obs
    expect_triple = confidence_triple(nav)
    np.testing.assert_array_equal(o20[17:], expect_triple)


# --------------------------------------------------------------------------- 6. torch-free (G0)
def test_confidence_builders_run_torch_free():
    """The PURE-numpy confidence builders (confidence_triple / confidence_triple_from_sigmas) import AND
    RUN without pulling torch. NOTE: estimator_obs / estimator_obs20 / estimator_obs_auto reach torch
    LAZILY via fly_rl.build_obs, so they are deliberately NOT exercised here (only the [17:20] core is)."""
    snippet = (
        "import sys; sys.path.insert(0, r'{src}');"
        "import numpy as np;"
        "from racer.estimator_obs import confidence_triple, confidence_triple_from_sigmas;"
        "from racer.contracts import NavState;"
        "t = confidence_triple(NavState(sim_time_ns=0, nav_inplane_sigma=0.0707, "
        "nav_along_sigma=0.08, time_since_vision_update_s=0.05));"
        "assert t.shape == (3,) and t.dtype == np.float32;"
        "c = confidence_triple_from_sigmas(0.05, 0.05, 0.0);"
        "assert abs(float(c[0]) - 1.0) < 1e-7;"
        "assert 'torch' not in sys.modules, 'confidence builders pulled torch';"
        "print('OBS20_TORCH_FREE_OK')"
    ).format(src=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OBS20_TORCH_FREE_OK" in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
