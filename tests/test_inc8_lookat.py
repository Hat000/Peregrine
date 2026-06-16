"""Unit tests for the inc8 architecture-pivot S0/S1 pieces (rl/inc8_reward.py):
  - the look-at PRIMITIVE: the baked camera<->body matrix is PINNED to canonical numpy
    frames.R_camera_from_body() (the documented sin-sign-flip trap is structurally impossible), and the
    correction ALWAYS reduces the camera-frame angular error to the gate (cross-product direction);
  - the dense terminal-sigma_p0 CENTERING reward shape;
  - OFF == the zero term (byte-identical) for both.

The FLU/FRD rate flip matches diffaero_dynamics._FLIP (pinned here when importable); the ACTION-execution
sign (does adding the FLU correction turn the drone toward the gate) is verified on the GPU by the
inc8_band_az_abs_deg diagnostic in the first few hundred steps. Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_inc8_lookat.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "src"))

import inc8_reward as RW                                              # noqa: E402
from racer import frames                                             # noqa: E402

DT = torch.float64


def _angles(t):
    """Camera frame [X right, Y down, Z fwd] -> (azimuth, elevation) deg (matches visibility_2axis)."""
    x, y, z = t[..., 0], t[..., 1], t[..., 2]
    a = torch.rad2deg(torch.atan2(x, z))
    b = torch.rad2deg(torch.atan2(y, torch.sqrt(x * x + z * z)))
    return a, b


# ===================================================== matrix parity (the sin-sign-flip guard)
def test_camera_matrix_parity_vs_canonical_frames():
    """r_body_from_camera() MUST equal frames.R_camera_from_body().T to 1e-6 -- the ONE source of truth.
    Catches any transcription/sign error in the baked constant AND any future BORESIGHT mount change."""
    baked = RW.r_body_from_camera(dtype=DT).numpy()
    canonical = frames.R_camera_from_body().T          # v_body = R @ v_cam
    assert np.allclose(baked, canonical, atol=1e-6), f"\nbaked=\n{baked}\ncanonical=\n{canonical}"


def test_flip_matches_diffaero():
    """The FRD<->FLU body-rate flip must be diag(1,-1,-1); pin to diffaero_dynamics._FLIP when importable
    (the action-rate convention source)."""
    assert RW._FLIP_FRD_FLU == (1.0, -1.0, -1.0)
    try:
        import diffaero_dynamics as DD
    except Exception:
        pytest.skip("diffaero_dynamics not importable on this host (GPU/cluster only)")
    flip = np.asarray(DD._FLIP).reshape(-1).tolist()
    assert flip == [1.0, -1.0, -1.0]


# ===================================================== look-at direction (cross-product correctness)
def _recover_omega_cam(t_cam, g_yaw, g_pitch):
    """End-to-end: lookat_correction returns the FLU body rate; recover the camera-frame omega it
    implies (R_cb @ (omega_flu * flip)) so the test exercises the REAL r_bc + flip, then check it
    reduces the gate's camera-frame angle under a small camera rotation."""
    r_bc = RW.r_body_from_camera(dtype=DT)
    flip = torch.tensor(RW._FLIP_FRD_FLU, dtype=DT)
    w_flu = RW.lookat_correction(t_cam, g_yaw, g_pitch, r_bc, flip)
    w_frd = w_flu * flip                               # involutory -> back to FRD
    w_cam = w_frd @ r_bc                               # R_cb = r_bc.T -> omega_cam = r_bc.T @ w_frd
    return w_cam


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_yaw_only_always_reduces_azimuth(seed):
    rng = np.random.default_rng(seed)
    t = torch.tensor(np.stack([rng.uniform(-8, 8, 500), rng.uniform(-6, 6, 500),
                               rng.uniform(3, 30, 500)], axis=-1), dtype=DT)
    w_cam = _recover_omega_cam(t, g_yaw=2.0, g_pitch=0.0)
    a0, _ = _angles(t)
    tp = t - 0.02 * torch.linalg.cross(w_cam, t)       # gate moves opposite the camera rotation
    a1, _ = _angles(tp)
    # yaw-only must never INCREASE |azimuth| (reduces it whenever there is any to reduce)
    assert torch.all(a1.abs() <= a0.abs() + 1e-9)
    assert torch.mean(a0.abs() - a1.abs()) > 0          # on average it strictly reduces


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_lookat_reduces_both_axes(seed):
    rng = np.random.default_rng(seed)
    t = torch.tensor(np.stack([rng.uniform(-8, 8, 500), rng.uniform(-6, 6, 500),
                               rng.uniform(3, 30, 500)], axis=-1), dtype=DT)
    w_cam = _recover_omega_cam(t, g_yaw=2.0, g_pitch=2.0)
    a0, b0 = _angles(t)
    tp = t - 0.02 * torch.linalg.cross(w_cam, t)
    a1, b1 = _angles(tp)
    assert torch.all((a1.abs() + b1.abs()) <= (a0.abs() + b0.abs()) + 1e-9)


def test_yaw_only_leaves_elevation_first_order():
    """g_pitch=0 -> the recovered omega_cam has ZERO component about camera-X (the elevation axis), so
    elevation is untouched to first order (the cheap-axis isolation S0 relies on)."""
    t = torch.tensor([[3.0, 2.0, 12.0], [-5.0, -1.0, 20.0]], dtype=DT)
    w_cam = _recover_omega_cam(t, g_yaw=2.0, g_pitch=0.0)
    assert torch.allclose(w_cam[:, 0], torch.zeros(2, dtype=DT), atol=1e-9)   # no camera-X (pitch) rot


def test_lookat_off_is_zero():
    t = torch.tensor([[3.0, 2.0, 12.0]], dtype=DT)
    r_bc = RW.r_body_from_camera(dtype=DT)
    flip = torch.tensor(RW._FLIP_FRD_FLU, dtype=DT)
    out = RW.lookat_correction(t, 0.0, 0.0, r_bc, flip)
    assert torch.count_nonzero(out) == 0


# ===================================================== dense centering reward
def test_centering_off_is_zero():
    err = torch.tensor([0.2, 0.1], dtype=DT)
    rng = torch.tensor([4.0, 20.0], dtype=DT)
    assert torch.count_nonzero(RW.centering_reward(err, rng, 0.0, 8.0, 3.0)) == 0


def test_centering_ramps_up_near_gate():
    """Same estimator error penalised MUCH harder near the crossing than far (the sigma_p0 emphasis)."""
    err = torch.tensor([0.2, 0.2, 0.2], dtype=DT)
    rng = torch.tensor([1.0, 8.0, 25.0], dtype=DT)             # near / mid / far
    r = RW.centering_reward(err, rng, rw_centering=1.0, r_near=8.0, w_near=3.0)
    assert r[0] < r[1] < r[2] <= 0                            # more negative (bigger penalty) near gate
    assert abs(r[1].item()) == pytest.approx(0.5 * 0.2, abs=1e-6)   # sigmoid(0)=0.5 at r_near
    assert r[2].abs() < 0.05 * 0.2 * 1.0 + 1e-9               # far -> negligible
