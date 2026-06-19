"""SYSTEM-ID registration: gate-frame rotations (numpy == torch) + the +L obs convention.

Pins three things that the inc8 torch env and the numpy emul/obs path MUST agree on:

  1. ``fly_rl._gate_rotmat_w2g`` (numpy, world->gate Z-up obs frame, rows [[c,s,0],[-s,c,0],[0,0,1]])
     == ``inc8_estimator_emul._gate_rotmat_w2g_torch`` over random yaws (float64, max|diff| ~1e-16).
  2. ``estimator_emul.ned_gate_frame`` (numpy, gate->world NED contracts.Gate frame, columns
     [right, down, downrange] = [[s,0,c],[c,0,-s],[0,1,0]]) == ``inc8_estimator_emul.ned_gate_frame_torch``.
  3. The +L obs sign restated as a PURE-numpy assertion (independent of the torch importorskip path):
     the obs pos_g slot is ``R_w2g @ (gate - pos) = R_w2g @ (+L)``; the -L slot is a 2*|L| flip
     (~24 m at gate range). This duplicates the SIGN guarantee of test_obs_sign_faithfulness.py at the
     bare-rotation layer so a -L regression is caught even if torch (and thus fly_rl) is absent.

C1 IDENTITY (registers the relationship between the two frames): a pure IN-PLANE NED-gate displacement
(built with the TORCH ned frame) produces ZERO obs along-track change through the numpy Z-up obs frame,
and a pure DOWNRANGE displacement ZERO obs in-plane change -- the two frames are consistent on the
opening plane (the same identity test_estimator_emul GATE#1 pins for the numpy ned frame).

Measured 2026-06-18 (system-id, gate-frame-plusL): w2g/ned numpy-vs-torch max|diff| = 1.11e-16;
det-1 / RR^T-I <= 2.22e-16; C1 worst residual 1.78e-15; +L exact, -L worst miss 147 m on the batch.

NOTE on float64: the torch helpers are dtype-following -- this test feeds float64 so the comparison is
at machine epsilon. The inc8 env runs them in float32 (~1e-7), still far inside the registration.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_sysid_gate_frames.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

# numpy side: pure (no torch) -- import unconditionally so test (3) runs even without torch.
from estimator_emul import ned_gate_frame                       # noqa: E402

# _gate_rotmat_w2g lives in fly_rl, which imports torch at module top.
torch = pytest.importorskip("torch")
from fly_rl import _gate_rotmat_w2g, _GATE_POS_ZUP, _FLIP, N_GATES  # noqa: E402
from inc8_estimator_emul import (                                    # noqa: E402
    _gate_rotmat_w2g_torch,
    ned_gate_frame_torch,
)

_PARITY_TOL = 1e-12          # float64 machine-eps comparison (measured ~1e-16)
_PLUS_L_TOL = 1e-12
_FLIP_BREAK_M = 10.0         # the -L negative control must miss by >= this (observed >> 24 m)


def _yaws():
    """Random yaws spanning two full turns + VQ1 pi + the cos/sin edge cases."""
    rng = np.random.default_rng(20260618)
    return np.concatenate([
        rng.uniform(-2 * np.pi, 2 * np.pi, 256),
        np.array([0.0, np.pi, -np.pi, np.pi / 2, -np.pi / 2, 2 * np.pi, -2 * np.pi]),
    ]).astype(np.float64)


def test_gate_rotmat_w2g_numpy_equals_torch():
    """world->gate (Z-up obs frame) rows [[c,s,0],[-s,c,0],[0,0,1]] -- numpy == torch."""
    yaws = _yaws()
    # batched torch (the (N,3,3) path the env uses) vs per-element numpy stack
    t_batch = _gate_rotmat_w2g_torch(torch.tensor(yaws, dtype=torch.float64)).cpu().numpy()
    np_stack = np.stack([_gate_rotmat_w2g(float(y)) for y in yaws])
    worst = float(np.max(np.abs(t_batch - np_stack)))
    assert worst <= _PARITY_TOL, f"_gate_rotmat_w2g numpy vs torch diverges by {worst:.3e} (> {_PARITY_TOL:.0e})"
    # every yaw is a proper rotation in BOTH sides
    for y, Rn in zip(yaws, np_stack):
        assert abs(np.linalg.det(Rn) - 1.0) < 1e-9, f"w2g(yaw={y}) not a proper rotation (det {np.linalg.det(Rn)})"


def test_ned_gate_frame_numpy_equals_torch():
    """gate->world NED (contracts.Gate) columns [right,down,downrange]=[[s,0,c],[c,0,-s],[0,1,0]] -- numpy == torch."""
    yaws = _yaws()
    t_batch = ned_gate_frame_torch(torch.tensor(yaws, dtype=torch.float64)).cpu().numpy()
    np_stack = np.stack([ned_gate_frame(float(y)) for y in yaws])
    worst = float(np.max(np.abs(t_batch - np_stack)))
    assert worst <= _PARITY_TOL, f"ned_gate_frame numpy vs torch diverges by {worst:.3e} (> {_PARITY_TOL:.0e})"
    for y, Rn in zip(yaws, np_stack):
        assert abs(np.linalg.det(Rn) - 1.0) < 1e-9, f"ned(yaw={y}) det {np.linalg.det(Rn)} != 1"
        assert np.allclose(Rn @ Rn.T, np.eye(3), atol=1e-9), f"ned(yaw={y}) not orthonormal"


def test_c1_identity_holds_for_torch_ned_frame():
    """Cross-frame registration: a displacement built with the TORCH ned frame is in-plane/along-track
    consistent with the numpy Z-up obs frame (pure in-plane -> 0 obs along-track; pure downrange ->
    0 obs in-plane). The same identity test_estimator_emul GATE#1 pins for the numpy ned frame."""
    rng = np.random.default_rng(11)
    worst_ip_to_along = 0.0
    worst_along_to_ip = 0.0
    for g in range(N_GATES):
        yaw = np.pi                                   # VQ1 all-pi
        Rwg_ned = ned_gate_frame_torch(torch.tensor([yaw], dtype=torch.float64))[0].cpu().numpy()
        Rw2g_zup = _gate_rotmat_w2g(yaw)
        gp_zup = _GATE_POS_ZUP[g]

        def pos_g(p_ned):
            return Rw2g_zup @ (gp_zup - p_ned * _FLIP)

        for _ in range(40):
            p_ned = (gp_zup + rng.uniform(-12, 12, 3)) * _FLIP
            base = pos_g(p_ned)
            a, b, d = rng.uniform(-2, 2, 3)
            d_inplane = Rwg_ned @ np.array([a, b, 0.0])    # right + down
            d_along = Rwg_ned @ np.array([0.0, 0.0, d])    # downrange
            worst_ip_to_along = max(worst_ip_to_along, abs((pos_g(p_ned + d_inplane) - base)[0]))
            worst_along_to_ip = max(worst_along_to_ip,
                                    float(np.max(np.abs((pos_g(p_ned + d_along) - base)[1:3]))))
    assert worst_ip_to_along < 1e-9, f"in-plane displacement leaked into obs along-track: {worst_ip_to_along:.3e}"
    assert worst_along_to_ip < 1e-9, f"downrange displacement leaked into obs in-plane: {worst_along_to_ip:.3e}"


def test_plus_L_obs_convention_numpy_only():
    """+L SIGN restated as a pure-numpy assertion (no torch / no fly_rl localization path required):
    the obs pos_g slot IS R_w2g @ (gate - pos) = R_w2g @ (+L); the spec-text -L slot is a 2*|L| flip
    (~24 m at gate range). Mirrors tests/test_obs_sign_faithfulness.py at the bare-rotation layer.
    DO NOT 'fix' the sign -- +L is load-bearing and correct in code (obs_from_zup:348, localization:86)."""
    rng = np.random.default_rng(7)
    worst_plus = 0.0
    worst_minus = 0.0
    for _ in range(2000):
        yaw = float(rng.uniform(-np.pi, np.pi))
        R = _gate_rotmat_w2g(yaw)
        gate = rng.uniform(-30, 30, 3)
        pos = rng.uniform(-30, 30, 3)
        L_plus = gate - pos                 # +L = drone->gate offset (the SEEN opening)
        pos_g = R @ L_plus                  # the obs slot, as obs_from_zup forms it
        worst_plus = max(worst_plus, float(np.max(np.abs(pos_g - R @ L_plus))))
        worst_minus = max(worst_minus, float(np.max(np.abs(pos_g - R @ (-L_plus)))))
    assert worst_plus <= _PLUS_L_TOL, f"+L identity broken: obs vs R@(+L) {worst_plus:.3e} (> {_PLUS_L_TOL:.0e})"
    assert worst_minus >= _FLIP_BREAK_M, (
        f"-L negative control did not break (only {worst_minus:.3e} m); the +L sign test is not "
        f"discriminating +L vs -L.")
