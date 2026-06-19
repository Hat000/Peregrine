"""Estimator -> policy-obs wiring (C2 step 4, BLUEPRINT §1.2/§1.6 / item 4).

In case-C mode the 17-dim policy observation's gate-relative ``pos_g`` / ``vel_g`` must be sourced from
the ESTIMATOR (the +L gate-relative offset, IMU-propagated to now) instead of the given wire pose. The
estimator already folds the gate-relative in-plane +L correction into the KF (navigator's
``_apply_gate_relative_fix``), so ``NavState.position_ned`` IS that estimate -- gate-relative-corrected
when a fresh in-band fix is accepted, the absolute-KF continuation otherwise (the §1.2 fallback,
automatic). The map bias cancels in the obs: ``pos_g = R_w2g @ (gate_map - p_est)`` with ``p_est`` pulled
tight to ``gate_map - L`` == ``R_w2g @ (+L)`` (the +L lever; pinned by tests/test_obs_sign_faithfulness).

The wiring is deliberately THIN: substitute the estimator's pos/vel into the wire ``DroneState`` and let
the EXISTING, canonical obs builder run -- so the 17-dim layout/sign are bit-identical to what inc7
consumes (G1). The trusted GIVEN attitude/rates stay from the wire (the estimator does not estimate
attitude). ``estimator_obs`` (the inc7 17-dim seam) does NOT append obs[17:20].

inc8 (obs_dim 20) self-localizing deploy ALSO needs the d5 confidence channel obs[17:20]
= [c_inplane, c_along, age_norm]. ``confidence_triple`` builds it from the ``NavState`` contract
(BYTE-FAITHFUL to the trained ``rl/estimator_emul.EstimatorEmulator.confidence_channel`` and the
``rl/spike_vertical_slice.deploy_confidence_triple`` it promotes); ``estimator_obs20`` is the 20-dim
assembler ([0:17] via ``estimator_obs`` ++ [17:20] via ``confidence_triple``); ``estimator_obs_auto``
dispatches 17-vs-20 by the actor's input width. THE sqrt(2) RECONCILIATION lives HERE in the obs builder
(``NavState.nav_inplane_sigma = sqrt(P_E+P_D)`` is sqrt(2)x the d5 ``sigma_inplane_hat = sqrt((P_E+P_D)/2)``
-- the navigator export stays INTENTIONALLY un-reconciled, pinned by tests/test_sysid_production_obs.py R3;
the /sqrt(2) is applied once, at this boundary). The inc7 17-dim default is UNCHANGED (opt-in 20-dim).

``estimator_state_for_obs`` + the confidence-triple builders are pure/torch-free (the estimator chain
imports this module without pulling torch, G0; pinned by tests/test_c2_torch_free.py). ``estimator_obs`` /
``estimator_obs20`` are the one-call conveniences that lazily import the shipped ``fly_rl.build_obs``
(torch) -- use them from the deploy loop / G1 tests. [C2-ESTIMATOR-CHAIN 2026-06-13; obs20 2026-06-19]
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

from racer.contracts import DroneState, NavState


def estimator_state_for_obs(ds: DroneState, nav_state: NavState) -> DroneState:
    """Wire ``DroneState`` with position/velocity REPLACED by the estimator's (NavState).

    THE estimator->obs seam: ``build_obs(estimator_state_for_obs(ds, nav), target_gate, ...)`` sources
    ``pos_g`` / ``vel_g`` from the gate-relative +L estimate while keeping the trusted given attitude
    (``orientation_ned_wxyz``) and body rates from the wire. Pure + torch-free."""
    return dataclasses.replace(
        ds,
        position_ned=np.asarray(nav_state.position_ned, dtype=np.float64).copy(),
        velocity_ned=np.asarray(nav_state.velocity_ned, dtype=np.float64).copy(),
    )


def _build_obs():
    """Lazily import the shipped obs builder (rl/fly_rl.build_obs) -- pulls torch, so kept out of the
    module top-level to keep the estimator chain importable torch-free (G0)."""
    rl_dir = Path(__file__).resolve().parents[2] / "rl"
    if str(rl_dir) not in sys.path:
        sys.path.insert(0, str(rl_dir))
    from fly_rl import build_obs   # noqa: E402  (torch-importing; lazy by design)
    return build_obs


def estimator_obs(
    ds: DroneState,
    nav_state: NavState,
    target_gate: int,
    last_normed_thrust: float,
    *,
    gate_map=None,
    virtual_flip: bool = False,
) -> np.ndarray:
    """17-dim policy obs sourcing pos_g/vel_g from the ESTIMATOR (case-C gate-relative pipeline).

    Reuses the canonical ``fly_rl.build_obs`` on a wire-state whose pos/vel are the estimator's, so the
    obs layout/sign are bit-identical to inc7's (G1). ``gate_map`` threads the runtime per-gate yaw
    (P4-C05); ``None`` uses the hardcoded VQ1 path. Does NOT append obs[17:20] (inc8)."""
    build_obs = _build_obs()
    return build_obs(estimator_state_for_obs(ds, nav_state), target_gate, last_normed_thrust,
                     virtual_flip=virtual_flip, gate_map=gate_map)


# ---------------------------------------------------------------------------
# obs[17:20] -- the d5 confidence channel (inc8). Pure numpy / torch-free.
# ---------------------------------------------------------------------------
# FROZEN d5 contract (MEMORY OBS CONTRACT 2026-06-13). These MUST equal the trainer's encoder
# constants (rl/estimator_emul.SIGMA_REF_M / TAU_STALE_S and navigator.INPLANE_POS_FLOOR_STD);
# defined locally so src/racer stays torch-free + rl-free, and pinned equal across modules by
# tests/test_sysid_confidence_constants.py + tests/test_deploy_obs20.py.
SIGMA_REF_M = 0.05          # confidence normalizer: c == 1 <=> at-or-better-than the 1-sigma bar
TAU_STALE_S = 0.10          # staleness horizon (~3 ticks @ 30 Hz)

# NavState.nav_inplane_sigma = sqrt(P_E + P_D) (navigator._gate_frame_pos_sigma); the d5 encoder uses
# sigma_inplane_hat = sqrt((P_E + P_D)/2) = nav_inplane_sigma / sqrt(2). The export is INTENTIONALLY
# the sum-sqrt (test R3); the obs builder applies the /sqrt(2) -- the ONLY place the correction lives.
_SQRT2 = float(np.sqrt(2.0))


def confidence_triple_from_sigmas(
    sigma_inplane_hat: float,
    sigma_along_hat: float,
    t_since_fix: float,
    *,
    sigma_ref: float = SIGMA_REF_M,
    tau_stale: float = TAU_STALE_S,
) -> np.ndarray:
    """The d5 obs[17:20] encoder core: ``[c_inplane, c_along, age_norm]`` (float32).

    BYTE-faithful to ``rl/estimator_emul.EstimatorEmulator.confidence_channel`` and
    ``rl/spike_vertical_slice.deploy_confidence_triple`` (the same clip/guard arithmetic):

        c = clip(sigma_ref / sigma_hat, 0, 1)          # 1.0 when sigma_hat <= 0 (degenerate /
                                                       #   over-converged cov reads MAX confidence)
        age_norm = clip(t_since_fix / tau_stale, 0, 1)

    The ``sigma_*_hat`` MUST already be in the d5 (emul) convention -- in-plane = sqrt((P_E+P_D)/2),
    along-track = sqrt(P_along). ``confidence_triple`` applies the NavState /sqrt(2) before calling this.
    ``sigma_hat == inf`` (no fix yet) -> c == 0 (min confidence). Returns float32 (the dtype the actor
    saw: ``EstimatorEmulator.obs`` casts the triple to float32)."""
    sig_ip = float(sigma_inplane_hat)
    sig_al = float(sigma_along_hat)
    c_ip = float(np.clip(sigma_ref / sig_ip, 0.0, 1.0)) if sig_ip > 0 else 1.0
    c_al = float(np.clip(sigma_ref / sig_al, 0.0, 1.0)) if sig_al > 0 else 1.0
    age = float(np.clip(float(t_since_fix) / tau_stale, 0.0, 1.0))
    return np.array([c_ip, c_al, age], dtype=np.float32)


def confidence_triple(
    nav_state: NavState,
    *,
    sigma_ref: float = SIGMA_REF_M,
    tau_stale: float = TAU_STALE_S,
) -> np.ndarray:
    """obs[17:20] = ``[c_inplane, c_along, age_norm]`` (float32) from the ``NavState`` contract.

    THE PRODUCTION confidence-channel builder (Path A: reads only the NavState contract, decoupled from
    Navigator internals -- consistent with how the deploy seam already reads position_ned/velocity_ned).
    Inputs, all already on the NavState (no extension required):

      * ``nav_inplane_sigma`` = sqrt(P_E + P_D)  -> sigma_inplane_hat = nav_inplane_sigma / sqrt(2)
            THE sqrt(2) RECONCILIATION (test R3 / FOOTGUN): the navigator export is the COMBINED-axis
            1-sigma (sum-sqrt); the d5 encoder wants the per-axis RMS (sqrt((P_E+P_D)/2)). Feeding the
            raw export mis-reports c_inplane by up to ~0.29 -- the /sqrt(2) is applied HERE, never in the
            navigator (R3 pins the export un-reconciled).
      * ``nav_along_sigma`` = sqrt(P_along)      -> sigma_along_hat (NO sqrt(2); identical definition)
      * ``time_since_vision_update_s``           -> age_norm

    Cold-start (no accepted fix yet): the navigator exports ``inf`` for both sigmas and ``inf`` for the
    fix clock -> triple == [0, 0, 1] (zero confidence, fully stale). Over-converged cov (sigma -> 0,
    e.g. inplane-floor OFF): c -> 1 (the degenerate MAX-confidence guard). Both match the d5 encoder.
    BYTE-faithful to the trained ``EstimatorEmulator.confidence_channel`` (<= float32 epsilon) GIVEN the
    same gate-frame covariance / fix clock -- pinned by tests/test_deploy_obs20.py.

    KNOWN, BOUNDED cold-start distribution-shift (deploy-integration carry-forward): the byte-faithfulness
    holds only AFTER the first gate-relative fix anchors a gate frame. PRE-first-fix the navigator has no
    gate frame -> exports ``inf`` -> this builder emits c==0; but the TRAINER's cold KF carries a FINITE
    P=I (pos_std=1.0) and emits c~=0.05. So obs[17:18] read 0.0 (deploy) vs ~0.05 (trained) in the
    pre-acquisition window ONLY -- both flag the channel STALE (age_norm==1, far below the c==1 trustworthy
    bar) and they converge on the first accepted fix. This is an ESTIMATOR-STATE difference (the builder
    faithfully encodes whatever sigma it is handed), not a builder defect; characterized + pinned by
    tests/test_deploy_obs20.py::test_cold_start_train_deploy_divergence_documented."""
    sigma_inplane_hat = float(nav_state.nav_inplane_sigma) / _SQRT2
    sigma_along_hat = float(nav_state.nav_along_sigma)
    return confidence_triple_from_sigmas(
        sigma_inplane_hat, sigma_along_hat, float(nav_state.time_since_vision_update_s),
        sigma_ref=sigma_ref, tau_stale=tau_stale,
    )


def estimator_obs20(
    ds: DroneState,
    nav_state: NavState,
    target_gate: int,
    last_normed_thrust: float,
    *,
    gate_map=None,
    virtual_flip: bool = False,
    sigma_ref: float = SIGMA_REF_M,
    tau_stale: float = TAU_STALE_S,
) -> np.ndarray:
    """20-dim inc8 deploy obs = [0:17] estimator seam ++ [17:20] confidence triple (float32).

    [0:17] via ``estimator_obs`` (bit-identical to the inc7 contract under matched state, G1); [17:20]
    via ``confidence_triple`` (the d5 channel from the NavState contract). Mirrors how the trainer
    assembles the 20-dim obs (``EstimatorEmulator.obs(obs_dim=20)`` = obs17 ++ confidence_channel),
    so the deployed inc8 obs == the obs the inc8 actor trained on. Pulls torch lazily (via estimator_obs)."""
    obs17 = estimator_obs(ds, nav_state, target_gate, last_normed_thrust,
                          gate_map=gate_map, virtual_flip=virtual_flip)
    triple = confidence_triple(nav_state, sigma_ref=sigma_ref, tau_stale=tau_stale)
    return np.concatenate([obs17.astype(np.float32), triple]).astype(np.float32)


def estimator_obs_auto(
    ds: DroneState,
    nav_state: NavState,
    target_gate: int,
    last_normed_thrust: float,
    actor_obs_dim: int,
    *,
    gate_map=None,
    virtual_flip: bool = False,
    sigma_ref: float = SIGMA_REF_M,
    tau_stale: float = TAU_STALE_S,
) -> np.ndarray:
    """Deploy-time obs SELECTION seam: 20-dim (inc8) when ``actor_obs_dim >= 20``, else 17-dim (inc7).

    ``actor_obs_dim`` is the actor's input width (``load_actor`` infers it from the checkpoint's first
    layer / sidecar; ``estimator_emul.actor_obs_dim`` reads the same). This keeps the 17-vs-20 dispatch
    in the obs layer so the deploy loop only chooses by width -- the inc7 path stays byte-identical and
    the 20-dim path is opt-in (selected ONLY for a 20-dim inc8 checkpoint)."""
    if int(actor_obs_dim) >= 20:
        return estimator_obs20(ds, nav_state, target_gate, last_normed_thrust,
                               gate_map=gate_map, virtual_flip=virtual_flip,
                               sigma_ref=sigma_ref, tau_stale=tau_stale)
    return estimator_obs(ds, nav_state, target_gate, last_normed_thrust,
                         gate_map=gate_map, virtual_flip=virtual_flip)
