"""B2 arms (2026-07-06 diagnosis): A1 lookat_max_rate cap + A2 centering-metric rename.

A1: rl/inc8_reward.lookat_rate_clamp -- per-axis magnitude cap on the look-at injected body-rate
correction (diagnosis RC6: pitch primitive injects 1.5-2.3 rad/s = 48-73% of authority in the
descent class). Default 0.0 == unclamped == byte-identical (identity OBJECT, zero float ops).
A2: the TB scalar 'inc8_centering' logged the deliberately-OFF rw_centering term while the ACTIVE
configured term is rw_through_centering (weight 10) -- a name collision that read as a dead reward.
Renamed to 'inc8_near_centering'; 'inc8_through_centering' (the active term) already logs.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_vq2_b2_arms.py -q
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import inc8_estimator_emul as IE                                       # noqa: E402
import inc8_reward as R8                                               # noqa: E402

DT = torch.float64


def _tail_first_pose_ned(gate_dist_m: float = 25.0):
    """Canonical tail-first training pose (mirrors tests/test_vq2_audit_fixes.py)."""
    gate_pos_ned = torch.zeros(1, 3, dtype=DT)
    R_wg = IE.ned_gate_frame_torch(torch.zeros(1, dtype=DT))
    drone_pos_ned = torch.tensor([[-gate_dist_m, 0.0, 0.0]], dtype=DT)
    yaw = math.pi
    q = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], dtype=DT)
    R_zup = IE.quat_xyzw_to_matrix_torch(q)
    f = torch.tensor([1.0, -1.0, -1.0], dtype=DT)
    R_ned = R_zup * f.view(1, 3, 1) * f.view(1, 1, 3)
    return drone_pos_ned, R_ned, gate_pos_ned, R_wg


def _rz_pi():
    return torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=DT))


def _off_axis_dlook():
    """Look-at correction for a 30-deg-off / 8-m-high gate through the FIXED (tail-mount) frame
    with the curriculum analytic gains (+3/+3) -- big enough to exercise a 1.0 rad/s cap."""
    pos, R_ned, gp, Rwg = _tail_first_pose_ned()
    gp = gp.clone()
    gp[0, 1] = -25.0 * math.tan(math.radians(30.0))
    gp[0, 2] = -8.0
    R_cb, K, _, _ = IE._const("cpu", DT)
    geom = IE.batched_geometry(pos, R_ned, gp, Rwg, R_cb @ _rz_pi(), K)
    r_bc = R8.r_body_from_camera(dtype=DT, tail_mount=True)
    flip = torch.tensor(R8._FLIP_FRD_FLU, dtype=DT)
    return R8.lookat_correction(geom["t_cam"], 3.0, 3.0, r_bc, flip)


# ============================================================ A1: lookat_max_rate cap
def test_lookat_rate_clamp_default_is_identity():
    """max_rate <= 0 (the default knob value 0.0) must return the INPUT TENSOR ITSELF -- no copy,
    no float op -> byte-identical legacy path, mirroring lookat_warmup_factor's <=0 convention."""
    dlook = _off_axis_dlook()
    assert R8.lookat_rate_clamp(dlook, 0.0) is dlook
    assert R8.lookat_rate_clamp(dlook, -1.0) is dlook


def test_lookat_rate_clamp_caps_and_preserves_sign_shape_dtype():
    """Active cap: per-axis |out| <= max_rate, signs preserved (symmetric clamp), in-bound
    components bit-exact, shape/dtype unchanged; a huge cap is numerically a no-op."""
    dlook = _off_axis_dlook()
    assert dlook.abs().max().item() > 1.0, "test setup: raw correction must exceed the 1.0 cap"
    out = R8.lookat_rate_clamp(dlook, 1.0)
    assert out.shape == dlook.shape and out.dtype == dlook.dtype
    assert out.abs().max().item() <= 1.0
    assert torch.equal(torch.sign(out), torch.sign(dlook))
    inb = dlook.abs() <= 1.0
    assert torch.equal(out[inb], dlook[inb]), "in-bound components must pass through bit-exact"
    assert torch.equal(R8.lookat_rate_clamp(dlook, 1e9), dlook)


def test_env_wires_lookat_max_rate_default_off():
    """SOURCE PIN (the env class needs diffaero to construct; the wiring is pinned textually):
    the knob is read with default 0.0 and the step-site clamp goes through the pinned helper."""
    src = (ROOT / "rl" / "peregrine_racing_inc8.py").read_text(encoding="utf-8")
    assert 'getattr(cfg, "lookat_max_rate", 0.0)' in src
    assert "R8.lookat_rate_clamp(dlook, self._lookat_max_rate)" in src


# ============================================================ A2: centering metric rename
def test_env_emits_renamed_centering_tags():
    """SOURCE PIN: the colliding scalar is renamed inc8_centering -> inc8_near_centering and the
    ACTIVE rw_through_centering term keeps its own inc8_through_centering tag. The old key must be
    GONE from the env (kill the collision; old event files are handled by the tb_trace fallback)."""
    src = (ROOT / "rl" / "peregrine_racing_inc8.py").read_text(encoding="utf-8")
    assert '"inc8_near_centering"' in src
    assert '"inc8_through_centering"' in src
    assert '"inc8_centering"' not in src, "the colliding tag must no longer be emitted"


def test_tb_trace_columns_carry_rename_with_fallback():
    """inc8_tb_trace must read the NEW tag first, keep the old tags as fallback candidates
    (pre-B2 event files), and surface the ACTIVE through_centering column."""
    import inc8_tb_trace as T
    cols = dict(T.COLUMNS)
    assert "centering" not in cols, "old display label must be renamed"
    assert cols["near_centering"][0] == "inc8_near_centering"
    assert "inc8_centering" in cols["near_centering"], "old-tag fallback for pre-B2 event files"
    assert cols["through_centering"][0] == "inc8_through_centering"
