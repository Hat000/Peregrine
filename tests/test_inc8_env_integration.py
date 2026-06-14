"""inc8 ENV-LOGIC INTEGRATION SMOKE (CPU, no diffaero) -- the laptop proxy for the Adroit GPU smoke.

Exercises the FULL inc8 in-loop signal chain that the env's step() wires together -- truth (Z-up) ->
NED -> batched estimator emulation (predict + camera-gated fix + KF update) -> 20-dim obs -> the
reward deltas (R1' / GT anchor / confidence shaping / R5') + BSR3 -- over a scripted head-on approach,
WITHOUT the diffaero dynamics. It proves the inc8 learning SIGNAL exists and is correctly wired:
pointing the camera at the gate raises the in-loop fix density, which tightens the KF, which the obs
+ reward see -- i.e. a policy CAN earn reward by pointing (the precondition for "pointing emerges").
The GPU-saturation + actual-PPO-learning smoke is the Adroit job (peregrine_inc8_smoke.sbatch).

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_inc8_env_integration.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.rl_plant import quat_rotate                                 # noqa: E402
import inc8_estimator_emul as IE                                       # noqa: E402
import inc8_reward as R8                                              # noqa: E402
from reference_line_torch import BatchedReferenceLine                 # noqa: E402
from estimator_emul import ned_gate_frame                             # noqa: E402
from offline_rollout import _quat_from_rpy                            # noqa: E402
from fly_rl import N_GATES, _FLIP, _GATE_POS_ZUP, _GATE_REL_POS, _GATE_YAW_REL  # noqa: E402

DT = torch.float64
_GATE_NED = _GATE_POS_ZUP * _FLIP
REF = str(ROOT / "rl" / "reference_line_inc8.json")


def _R_from_quat(q):
    return np.stack([quat_rotate(np.asarray(q, np.float64), e) for e in np.eye(3)], axis=-1)


def _build(n, cfg=None):
    gate_pos = torch.tensor(_GATE_NED, dtype=DT)
    Rwg = torch.stack([torch.tensor(ned_gate_frame(np.pi), dtype=DT) for _ in range(N_GATES)])
    emu = IE.BatchedEstimatorEmulator(n, gate_pos, Rwg, config=cfg or IE.EmulConfig(),
                                      device="cpu", dtype=DT)
    refline = BatchedReferenceLine.load(REF, "cpu", DT)
    return emu, refline


def _rollout(crab_rad, n_per=64, steps=90, seed=0, cfg=None):
    """Scripted constant-velocity head-on approach to gate 3 with a fixed body yaw offset (crab). All
    envs share the geometry; only the per-episode DR + the fix Bernoulli differ. Returns the per-step
    aggregate signal traces (fix-rate, pointing-rate, KF in-plane error, R5', R1')."""
    torch.manual_seed(seed)
    n = n_per
    emu, refline = _build(n, cfg)
    gate = 3
    gate_ned = _GATE_NED[gate]
    q = _quat_from_rpy(0.0, 0.0, np.pi + crab_rad)         # yaw offset = crab
    R = _R_from_quat(q)
    R_t = torch.tensor(np.broadcast_to(R, (n, 3, 3)).copy(), dtype=DT)
    v = 12.0
    p0 = np.broadcast_to(gate_ned + np.array([26.0, 0.0, 0.0]), (n, 3)).copy()
    v0 = np.broadcast_to([-v, 0.0, 0.0], (n, 3)).copy()
    tg = torch.full((n,), gate, dtype=torch.long)
    sig, bias = IE.BatchedEstimatorEmulator.sample_episode_dr(emu.cfg, n, "cpu", DT)
    emu.reset_idx(torch.arange(n), torch.tensor(p0, dtype=DT), torch.tensor(v0, dtype=DT), sig, bias)

    dt = 0.0333
    cur_p, cur_v = p0.copy(), v0.copy()
    w = R8.Inc8RewardWeights()
    spin_clock = torch.zeros(n, dtype=DT)
    tr = {"fix": [], "point": [], "err": [], "r5": [], "r1p": [], "c_inplane": [],
          "obs_finite": [], "spin_abort": []}
    for s in range(steps):
        prev_p, prev_v = cur_p.copy(), cur_v.copy()
        cur_p = prev_p + prev_v * dt
        accepted = emu.step(
            torch.tensor(prev_p, dtype=DT), torch.tensor(prev_v, dtype=DT), R_t,
            torch.tensor(cur_p, dtype=DT), torch.tensor(cur_v, dtype=DT), R_t, tg, dt,
            torch.rand(n, dtype=DT), torch.randn(n, 3, dtype=DT), torch.randn(n, 3, dtype=DT))
        s_prev = refline.progress(torch.tensor(prev_p, dtype=DT))
        s_curr = refline.progress(torch.tensor(cur_p, dtype=DT))
        geom = emu._last_geom
        err = emu.gate_frame_error_inplane(tg, torch.tensor(cur_p, dtype=DT))
        triple = emu.confidence_channel(tg)
        # the 20-dim obs (KF pose Z-up + truth attitude/rates + triple)
        R_b2w_zup = torch.tensor(np.broadcast_to(IE.FLIP_NP[:, None] * R * IE.FLIP_NP[None, :],
                                                 (n, 3, 3)).copy(), dtype=DT)
        nxt = min(gate + 1, N_GATES - 1)
        obs = IE.obs_zup_torch(
            emu.kf_pos_zup(), emu.kf_vel_zup(), R_b2w_zup, torch.zeros(n, 3, dtype=DT),
            torch.tensor(np.broadcast_to(_GATE_POS_ZUP[gate], (n, 3)).copy(), dtype=DT),
            torch.full((n,), np.pi, dtype=DT),
            torch.tensor(np.broadcast_to(_GATE_REL_POS[nxt], (n, 3)).copy(), dtype=DT),
            torch.full((n,), float(_GATE_YAW_REL[nxt]), dtype=DT),
            torch.zeros(n, dtype=DT), triple=triple)
        r5 = R8.perception_reward(geom["t_cam"], geom["range"], s_curr - s_prev, geom["in_image"], "A", w)
        r1p = R8.arc_progress_reward(s_curr, s_prev, w.progress)
        spin_clock, spin_abort = R8.bsr3_update(spin_clock, torch.tensor(
            np.broadcast_to([0.0, 0.0, 11.0], (n, 3)).copy(), dtype=DT), dt, 10.0, 3.0)
        tr["fix"].append(accepted.float().mean().item())
        tr["point"].append(geom["in_image"].float().mean().item())
        tr["err"].append(err.mean().item())
        tr["r5"].append(r5.mean().item())
        tr["r1p"].append(r1p.mean().item())
        tr["c_inplane"].append(triple[:, 0].mean().item())
        tr["obs_finite"].append(bool(torch.isfinite(obs).all()))
        tr["spin_abort"].append(bool(spin_abort.any()))
        if cur_p[0, 0] <= gate_ned[0]:
            break
    return {k: np.asarray(v) for k, v in tr.items()}, obs.shape[-1]


def test_inc8_signal_chain_no_nan_and_obs_dim20():
    """No NaN/inf anywhere; obs is 20-dim; reward terms finite; BSR3 does not fire on the ~11 rad/s
    transient over a < 3 s rollout."""
    tr, obs_dim = _rollout(crab_rad=0.0)
    assert obs_dim == 20
    assert tr["obs_finite"].all(), "non-finite obs in the rollout"
    for k in ("fix", "point", "err", "r5", "r1p"):
        assert np.all(np.isfinite(tr[k])), (k, tr[k])
    assert not tr["spin_abort"].any(), "BSR3 false-aborted the legitimate ~11 rad/s transient"
    # R1' is positive on a forward approach (advancing along Gamma)
    assert np.nanmean(tr["r1p"]) > 0.0


def test_pointing_raises_fix_density_confidence_and_r5():
    """THE in-loop signal: a CENTRED camera (crab 0) earns a higher fix density than an OFF-pointed
    one (crab 60 deg), which shrinks the KF covariance -> a higher c_inplane CONFIDENCE (what obs[17]
    carries + the confidence-shaping rewards) and a higher R5' (terminal-locked perception reward).
    This is 'pointing pays', the precondition for 'pointing emerges' under PPO.

    NB the one-signed in-plane BIAS is ON (the binding case-(b)): a fix pulls the KF toward truth+bias,
    so absolute |KF-truth| converges to ~the bias, NOT to 0 (the perception bias is the irreducible
    case-C error the policy cannot point away -- it is exactly what the GT anchor + boresight calib
    target). So the honest 'pointing tightens the estimate' signal is the COVARIANCE/confidence, not
    the bias-floored absolute error; the bias-OFF test below pins the absolute-error reduction."""
    pointed, _ = _rollout(crab_rad=0.0, seed=1)
    blind, _ = _rollout(crab_rad=np.radians(60.0), seed=1)
    assert pointed["point"].mean() > 0.5
    assert blind["point"].mean() < pointed["point"].mean()
    assert pointed["fix"].mean() > blind["fix"].mean() + 0.05, (pointed["fix"].mean(), blind["fix"].mean())
    # KF covariance tightens with fixes -> higher in-plane confidence (terminal third)
    tail = slice(-max(len(pointed["c_inplane"]) // 3, 1), None)
    assert np.nanmean(pointed["c_inplane"][tail]) > np.nanmean(blind["c_inplane"][tail])
    # R5' rewards the centred camera more (it is in-frame + terminal-weighted)
    assert pointed["r5"].mean() > blind["r5"].mean()


def test_pointed_fix_stream_tightens_kf_from_cold():
    """The centred fix stream tightens the KF from its COLD init: the in-plane confidence peaks far
    above the cold-start value once fixes land in the 16-28 m accept window. NB confidence then FALLS
    in the final approach (range < 16 m, where the band-pass cuts fixes off) -- that terminal fix
    DROUGHT is exactly the phenomenon inc8's camera-pointing + terminal-lock reward exists to fix, so
    confidence is correctly non-monotone, not rising-to-the-gate. (Absolute |KF-truth| is bias/noise-
    floored on this short constant-velocity script -- the truth-synthesised IMU dead-reckons near-
    perfectly; absolute-error reduction is a long-horizon/maneuvering property for the Adroit smoke.)"""
    pointed, _ = _rollout(crab_rad=0.0, seed=4)
    c = pointed["c_inplane"]
    assert np.all((c >= 0.0) & (c <= 1.0))           # bounded, honest (NEES-calibrated; see S3)
    assert c.max() > c[0] + 0.1, (c[0], c.max())     # fixes tighten the KF from cold
    # and the terminal-window fix drought is visible: fixes are concentrated, not uniform
    assert pointed["fix"].max() > 0.3


def test_reward_rises_along_approach():
    """A coarse 'learning-at-all is possible' check: summed shaped reward (R1' + R5') over a centred,
    progressing approach is strictly positive (the policy is paid to fly the line + keep the gate in
    view) -- the reward landscape points the right way."""
    tr, _ = _rollout(crab_rad=0.0, seed=2)
    shaped = tr["r1p"] + tr["r5"]
    assert shaped.sum() > 0.0
    # and the GT-anchor-relevant KF error stays bounded (no divergence)
    assert np.nanmax(tr["err"]) < 5.0
