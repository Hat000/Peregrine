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
attitude). NO obs[17:20] confidence channel is appended here -- that is the later inc8 step.

``estimator_state_for_obs`` is pure/torch-free (the estimator chain imports it without pulling torch,
G0). ``estimator_obs`` is the one-call convenience that lazily imports the shipped ``fly_rl.build_obs``
(torch) -- use it from the deploy loop / G1 tests. [C2-ESTIMATOR-CHAIN 2026-06-13]
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
