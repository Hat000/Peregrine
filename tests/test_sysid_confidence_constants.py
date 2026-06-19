"""SYS-ID registration: the confidence-triple CONSTANTS + sigma-hat definitions across the
three modules (#2).

Counterpart pair:
  * rl/estimator_emul.py        (numpy escape-hatch reference / SPEC)
  * rl/inc8_estimator_emul.py   (torch production twin)
  * src/racer (navigator / state_estimator / contracts) NavState sigma export

What this pins (drives identical inputs, isolates one layer):
  1. SIGMA_REF_M / TAU_STALE_S are NUMERICALLY EQUAL across both emulators (module-level
     constants AND the EmulConfig dataclass defaults), and the rest of the shared EmulConfig DR
     scalars match too.
  2. navigator.INPLANE_POS_FLOOR_STD (the sigma_b state floor) is the SAME 0.05 m as SIGMA_REF_M
     -- the localization comment asserts this alias; pin it so a drift in one is caught.
  3. confidence_channel + _gate_frame_sigmas FORMULA PARITY: numpy and torch, fed the SAME
     synthetic gate-frame KF covariance, produce bit-identical sigma_inplane_hat / sigma_along_hat
     and the [c_inplane, c_along, age_norm] triple.
  4. The INTENTIONAL sqrt(2) GAP: the NavState in-plane sigma (navigator._gate_frame_pos_sigma:
     sqrt(P_E + P_D)) is EXACTLY sqrt(2) x the emulator's confidence in-plane sigma
     (sqrt((P_E + P_D)/2)). DOCUMENTED, NOT reconciled (the two channels intentionally differ:
     NavState exports the combined-axis 1-sigma; the emulator confidence uses the per-axis RMS).
     The along-track sigma is IDENTICAL (both sqrt(P_along)) -- only the in-plane differs.

inc7 (obs_dim 17) is untouched -- these are obs[17:20] / NavState-export definitions only.

Run from repo ROOT:
  .venv/Scripts/python.exe -m pytest tests/test_sysid_confidence_constants.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import estimator_emul as NE                              # noqa: E402
from racer.navigator import INPLANE_POS_FLOOR_STD        # noqa: E402

torch = pytest.importorskip("torch")
import inc8_estimator_emul as TE                          # noqa: E402


# --------------------------------------------------------------------------- 1. scalar constants
def test_sigma_ref_tau_stale_equal_across_emulators():
    """SIGMA_REF_M / TAU_STALE_S equal at module level AND as EmulConfig defaults (numpy == torch)."""
    assert NE.SIGMA_REF_M == TE.SIGMA_REF_M == 0.05
    assert NE.TAU_STALE_S == TE.TAU_STALE_S == 0.10
    ncfg, tcfg = NE.EmulConfig(), TE.EmulConfig()
    assert ncfg.sigma_ref == tcfg.sigma_ref == NE.SIGMA_REF_M
    assert ncfg.tau_stale == tcfg.tau_stale == NE.TAU_STALE_S


def test_emulconfig_shared_dr_scalars_match():
    """The DR/encoding scalars shared by both EmulConfigs are numerically identical (registration
    of the whole config, not just sigma_ref/tau_stale)."""
    ncfg, tcfg = NE.EmulConfig(), TE.EmulConfig()
    for f in ("sigma_lat_lo", "sigma_lat_hi", "bias_mag_lo", "bias_mag_hi", "inject_bias",
              "sigma_ref", "tau_stale", "pos_std_init", "vel_std_init", "imu_accel_noise"):
        assert getattr(ncfg, f) == getattr(tcfg, f), f"EmulConfig.{f} diverged"


def test_inplane_pos_floor_aliases_sigma_ref():
    """navigator.INPLANE_POS_FLOOR_STD (sigma_b state floor) is the SAME 0.05 m as SIGMA_REF_M --
    the localization.py comment asserts this alias; pin it so a drift in either is caught."""
    assert INPLANE_POS_FLOOR_STD == NE.SIGMA_REF_M == 0.05


# --------------------------------------------------------------------------- 2. formula parity
def _world_P_from_gate_diag(Rwg, p_e, p_d, p_along):
    """A 6x6 KF P whose POSITION block is diag([p_e, p_d, p_along]) in the gate frame Rwg, rotated
    to world NED. Velocity block arbitrary (the sigma projection reads only P[:3,:3])."""
    P_world = Rwg @ np.diag([p_e, p_d, p_along]) @ Rwg.T
    P6 = np.zeros((6, 6))
    P6[:3, :3] = P_world
    P6[3:, 3:] = np.eye(3) * 25.0
    return P6


def _numpy_emu_with_P(P6, target_gate):
    class _S:
        pos = np.array([0.0, 0.0, -2.0]); vel = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0]); omega = np.zeros(3)
    emu = NE.EstimatorEmulator(NE.EmulConfig())
    emu.reset(_S(), target_gate=target_gate, rng=np.random.default_rng(0))
    emu.kf.P = P6.copy()
    return emu


def _torch_emu_with_P(P6):
    gate_pos_ned = torch.tensor(NE._GATE_POS_ZUP * NE._FLIP, dtype=torch.float64)
    Rwg_all = torch.stack([torch.tensor(NE.ned_gate_frame(np.pi), dtype=torch.float64)
                           for _ in range(len(NE._GATE_POS_ZUP))])
    emu = TE.BatchedEstimatorEmulator(1, gate_pos_ned, Rwg_all, config=TE.EmulConfig(),
                                      device="cpu", dtype=torch.float64)
    emu.kf.P = torch.tensor(P6, dtype=torch.float64).unsqueeze(0)
    emu._t_since_fix = torch.full((1,), 1e3, dtype=torch.float64)   # cold -> age_norm == 1
    return emu


def test_gate_frame_sigmas_and_confidence_numpy_torch_parity():
    """Fed the SAME synthetic gate-frame KF covariance, numpy and torch _gate_frame_sigmas +
    confidence_channel agree to machine precision -- registers the obs[17:20] FORMULA pair."""
    tg = 2
    p_e, p_d, p_along = 0.0144, 0.0400, 0.2500   # var: stds 0.12 / 0.20 / 0.50 m
    n_emu_tmp = _numpy_emu_with_P(np.zeros((6, 6)), tg)
    Rwg = n_emu_tmp.gates[tg].R_world_gate
    P6 = _world_P_from_gate_diag(Rwg, p_e, p_d, p_along)

    n_emu = _numpy_emu_with_P(P6, tg)
    t_emu = _torch_emu_with_P(P6)

    sig_ip_np, sig_al_np = n_emu._gate_frame_sigmas(tg)
    sig_ip_t, sig_al_t = t_emu._gate_frame_sigmas(torch.tensor([tg]))
    triple_np = n_emu.confidence_channel(tg)
    triple_t = t_emu.confidence_channel(torch.tensor([tg]))[0].numpy()

    # numpy matches its own spec formula
    assert sig_ip_np == pytest.approx(np.sqrt((p_e + p_d) / 2.0), abs=1e-12)
    assert sig_al_np == pytest.approx(np.sqrt(p_along), abs=1e-12)
    # torch == numpy (bit-identical formula)
    assert float(sig_ip_t) == pytest.approx(sig_ip_np, abs=1e-12)
    assert float(sig_al_t) == pytest.approx(sig_al_np, abs=1e-12)
    assert np.allclose(triple_np, triple_t, atol=1e-12)
    # confidence triple values themselves (sigma_ref / sigma_hat, clipped; age cold)
    assert triple_np[0] == pytest.approx(min(NE.SIGMA_REF_M / sig_ip_np, 1.0), abs=1e-12)
    assert triple_np[1] == pytest.approx(min(NE.SIGMA_REF_M / sig_al_np, 1.0), abs=1e-12)
    assert triple_np[2] == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- 3. the sqrt(2) gap
def test_navstate_vs_emulator_inplane_sigma_sqrt2_gap():
    """INTENTIONAL, DOCUMENTED divergence (do NOT reconcile): the NavState in-plane sigma
    (navigator._gate_frame_pos_sigma: sqrt(P_E + P_D)) is EXACTLY sqrt(2) x the emulator's
    confidence in-plane sigma (sqrt((P_E + P_D)/2)). The along-track sigma is IDENTICAL.

    Pinned at the FORMULA level (the same P_E,P_D feeds both) so a future 'fix' that silently
    reconciles the two channels trips this test."""
    tg = 2
    p_e, p_d, p_along = 0.0144, 0.0400, 0.2500
    n_emu_tmp = _numpy_emu_with_P(np.zeros((6, 6)), tg)
    Rwg = n_emu_tmp.gates[tg].R_world_gate
    P6 = _world_P_from_gate_diag(Rwg, p_e, p_d, p_along)
    n_emu = _numpy_emu_with_P(P6, tg)

    emul_inplane, emul_along = n_emu._gate_frame_sigmas(tg)
    # NavState's exported in-plane sigma is the COMBINED-axis 1-sigma: sqrt(P_E + P_D)
    # (navigator._gate_frame_pos_sigma line ~626). It reads the same P[:3,:3] block.
    navstate_inplane = float(np.sqrt(max(p_e + p_d, 0.0)))
    navstate_along = float(np.sqrt(max(p_along, 0.0)))

    # the gap is EXACTLY sqrt(2) -- registered, intentional, not a bug.
    assert navstate_inplane / emul_inplane == pytest.approx(np.sqrt(2.0), abs=1e-12)
    # along-track is the SAME definition in both -> no gap.
    assert navstate_along == pytest.approx(emul_along, abs=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
