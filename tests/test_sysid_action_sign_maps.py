"""SYS-ID registration: the ACTION / RATE SIGN MAPS (pair #3c).

This is a PURE-CONSTANT registration test. It pins the exact values, the involution
property, and the cross-map relationships of every sign matrix that participates in the
RL action -> wire / look-at reconstruction pipeline. It is the laptop-runnable, ALWAYS-ON
guard for the area of the recently-fixed yaw-injection bug (commit ec4cb03 / dddf5d9):
the project's other tests that touch these maps (tests/test_confirmed_cr1_01.py,
cr2_01, cr3_02, p3_c06) are SKIPPED whenever the ShadowPC live recordings are absent
(the canonical-laptop default), so the SOURCE-CONSTANT invariants they encode were
otherwise unpinned on a clean checkout. This test imports the LIVE constants (so a future
edit to fly_rl / offline_rollout / inc8_reward is caught) and has no data dependency.

DO NOT auto-change any value asserted here -- these are load-bearing SIGN/FRAME
conventions. A divergence is a commander-review item, not a worker fix.

REGISTERED FACTS (all measured 2026-06-18, see scratch/action-sign-maps/measure_output.txt):
  * fly_rl._FLIP            == [ 1, -1, -1]   (FLU<->FRD & Z-up<->NED frame flip; training adapter)
  * fly_rl._ACT_FLU_TO_FRD  == [ 1, -1,  1]   (policy_step LIVE wire map: rate_frd = rate_flu * this)
  * fly_rl._RZ_PI_BODY      == diag(-1,-1, 1) (virtual body flip, pi about body z)
  * offline_rollout._RATE_SIGN_LIVE == [1, 1, 1]  (eval plant rate sign -- NO inversion any axis)
  * inc8_reward._FLIP_FRD_FLU == [1, -1, -1]  (inc8 reward body-rate FRD<->FLU flip; == _FLIP)

  * _FLIP, _ACT_FLU_TO_FRD, _FLIP_FRD_FLU are all INVOLUTORY (v*m*m == v); _RZ_PI_BODY^2 == I.
  * _FLIP and _ACT_FLU_TO_FRD AGREE on roll/pitch (axes 0,1) and DIFFER on YAW (axis 2:
    -1 vs +1). This yaw-only difference IS the yaw-injection bug: contact_true_eval's look-at
    reconstruction (lines 365/367) must invert/re-apply _ACT_FLU_TO_FRD (the LIVE wire map),
    NOT _FLIP -- reconstructing through _FLIP sign-flips the look-at's yaw realized rate while
    leaving roll/pitch bit-identical (the exact symptom the bug exhibited; FIXED ec4cb03).
  * _FLIP == _FLIP_FRD_FLU (the inc8 reward flip is the same map as the training FLU<->FRD adapter).
  * Trained-semantics identity: _FLIP * trained_plant_rate_sign([1,1,-1]) == _ACT_FLU_TO_FRD,
    i.e. the live wire map equals the realized FRD of the TRAINING closed loop
    (adapter [1,-1,-1] composed onto the training plant [1,1,-1]) -- which is why deploying
    on the live plant rate_sign [1,1,1] preserves the trained control semantics.

FOOTGUN NOTE: the legacy CTBR _rate_sign alias and the live rate_sign [1,1,1] are
self-consistent and INTENTIONAL -- registered here, never "fixed".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_RL = Path(__file__).resolve().parents[1] / "rl"
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_RL))
sys.path.insert(0, str(_SRC))

import fly_rl  # noqa: E402
from inc8_reward import _FLIP_FRD_FLU  # noqa: E402
from offline_rollout import _RATE_SIGN_LIVE  # noqa: E402
import contact_true_eval as cte  # noqa: E402


def _involutory(m: np.ndarray) -> bool:
    """A diagonal sign vector m (applied elementwise) is involutory iff v*m*m == v."""
    v = np.array([0.31, -0.72, 0.95, -0.13, 0.51])[: len(m)]
    return bool(np.allclose(v * m * m, v))


# --------------------------------------------------------------------------- exact values
def test_flip_exact():
    assert np.array_equal(fly_rl._FLIP, np.array([1.0, -1.0, -1.0]))


def test_act_flu_to_frd_exact():
    # The LIVE wire map. yaw element is +1 (the corrected value; the OLD bcc93f9 map was -1).
    assert np.array_equal(fly_rl._ACT_FLU_TO_FRD, np.array([1.0, -1.0, 1.0]))


def test_rz_pi_body_exact():
    assert np.array_equal(fly_rl._RZ_PI_BODY, np.diag([-1.0, -1.0, 1.0]))


def test_rate_sign_live_exact():
    # The eval plant rate sign -- NO inversion on any axis (FRAME-AUDIT 2026-06-12).
    assert np.array_equal(np.asarray(_RATE_SIGN_LIVE, dtype=np.float64),
                          np.array([1.0, 1.0, 1.0]))


def test_flip_frd_flu_exact():
    assert np.array_equal(np.asarray(_FLIP_FRD_FLU, dtype=np.float64),
                          np.array([1.0, -1.0, -1.0]))


# --------------------------------------------------------------------------- involution
def test_all_sign_maps_involutory():
    assert _involutory(fly_rl._FLIP)
    assert _involutory(fly_rl._ACT_FLU_TO_FRD)
    assert _involutory(np.asarray(_FLIP_FRD_FLU, dtype=np.float64))


def test_rz_pi_body_involutory():
    assert np.allclose(fly_rl._RZ_PI_BODY @ fly_rl._RZ_PI_BODY, np.eye(3))


# ------------------------------------------------------- the yaw-injection-bug relationship
def test_flip_and_act_agree_rollpitch_differ_yaw():
    """_FLIP and _ACT_FLU_TO_FRD agree on roll/pitch (axes 0,1) and DIFFER on YAW (axis 2).

    This is THE registration of the recently-fixed yaw-injection bug: the look-at FLU action
    must be reconstructed via _ACT_FLU_TO_FRD (the live wire map), not _FLIP. Both maps are
    involutory and identical on roll/pitch, so reconstructing through _FLIP leaves roll/pitch
    bit-identical but sign-flips the yaw realized rate -- exactly the bug's signature."""
    flip = fly_rl._FLIP
    act = fly_rl._ACT_FLU_TO_FRD
    # roll/pitch agree
    assert np.array_equal(flip[:2], act[:2])
    # yaw differs, and specifically -1 (FLIP) vs +1 (ACT)
    assert flip[2] == -1.0
    assert act[2] == 1.0
    assert flip[2] != act[2]


# --------------------------------------------------------------------------- cross-map facts
def test_flip_equals_flip_frd_flu():
    """The inc8 reward's body-rate FRD<->FLU flip is the SAME map as the training adapter."""
    assert np.array_equal(fly_rl._FLIP, np.asarray(_FLIP_FRD_FLU, dtype=np.float64))


def test_trained_semantics_identity():
    """_FLIP * trained_plant_rate_sign([1,1,-1]) == _ACT_FLU_TO_FRD.

    The live wire map equals the realized FRD of the TRAINING closed loop (the [1,-1,-1]
    adapter composed onto the training plant rate_sign [1,1,-1]); deploying on the live
    plant rate_sign [1,1,1] then preserves the trained control semantics."""
    trained_plant_rate_sign = np.array([1.0, 1.0, -1.0])
    realized = fly_rl._FLIP * trained_plant_rate_sign
    assert np.array_equal(realized, fly_rl._ACT_FLU_TO_FRD)


# ------------------------------------------------- call-site wiring (contact_true_eval) ----
def test_contact_true_eval_reexports_live_constants():
    """contact_true_eval must reconstruct the look-at FRD via the LIVE wire map _ACT_FLU_TO_FRD
    (the actual fly_rl object), use _FLIP_FRD_FLU for _LOOKAT_FLIP, and run the eval plant on
    _RATE_SIGN_LIVE. Pins that the post-fix call sites read the right constants."""
    assert cte._ACT_FLU_TO_FRD is fly_rl._ACT_FLU_TO_FRD
    assert np.array_equal(cte._LOOKAT_FLIP.cpu().numpy(),
                          np.asarray(_FLIP_FRD_FLU, dtype=np.float64))
    assert np.array_equal(np.asarray(cte._RATE_SIGN_LIVE, dtype=np.float64),
                          np.array([1.0, 1.0, 1.0]))
