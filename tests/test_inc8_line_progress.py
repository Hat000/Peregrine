"""TRACK-AGNOSTIC optimal-line shaping mechanism -- the rw_line_progress / rw_speed_ref reward terms.

These terms add a DENSE projected-progress signal along an ARBITRARY precomputed optimal line (bridge
offline-exact -> RL): project the drone position onto the line, reward the forward arc-length delta,
and optionally a soft v(s) speed reference. The line FILE is a config input -- on a new track (VQ2) we
feed that track's optimal-line JSON; the mechanism is unchanged. This pins:

  * the mechanism is TRACK-AGNOSTIC: a synthetic straight-line fixture (NOT the VQ1 line) loads and
    projects correctly -- proving the path is decoupled from any specific track;
  * the term FIRES: forward motion along the line pays positive line_progress, and the speed term
    penalises an off-profile speed (advancing-gated);
  * OFF == byte-identical: weight 0.0 -> the EXACT zero tensor (the inc8 contract).

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_inc8_line_progress.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import inc8_reward as R8                                               # noqa: E402
from time_optimal_line import TimeOptimalLine                         # noqa: E402

DT = torch.float64


def _synthetic_line_json(tmp_path) -> str:
    """A throwaway STRAIGHT-line fixture along -X (arbitrary, NOT the VQ1 track): 0->-100 m at a
    constant 30 m/s reference. Proves the mechanism takes any optimal-line file."""
    n = 51
    xs = np.linspace(0.0, -100.0, n)
    pos = np.stack([xs, np.zeros(n), np.full(n, -2.0)], axis=1)
    payload = {
        "schema": "peregrine.reference_line.v1",
        "frame": "world NED",
        "pos_ned": pos.tolist(),
        "speed_ref_mps": np.full(n, 30.0).tolist(),
    }
    p = tmp_path / "synthetic_line.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_track_agnostic_load_and_project(tmp_path):
    """An ARBITRARY optimal-line file (synthetic straight line) loads and projects to monotone
    arc-length -- the mechanism is not tied to the VQ1 line."""
    line = TimeOptimalLine.load(_synthetic_line_json(tmp_path), "cpu", DT)
    assert abs(line.total_arc_m - 100.0) < 1e-6
    # walk along -X: arc-length increases monotonically; v_ref is the constant 30 m/s
    pts = torch.tensor([[0.0, 0, -2], [-25, 0, -2], [-50, 0, -2], [-100, 0, -2]], dtype=DT)
    s = line.progress(pts)
    assert torch.all(torch.diff(s) > 0), s
    assert float(s[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(s[-1]) == pytest.approx(100.0, abs=1e-6)
    vref = line.speed_ref(pts)
    assert torch.allclose(vref, torch.full((4,), 30.0, dtype=DT), atol=1e-6)


def test_line_progress_fires_forward_and_clamps_backward():
    """line_progress pays the FORWARD arc-length delta (rw>0) and clamps a backward step to 0."""
    s_prev = torch.tensor([0.0, 5.0, 10.0], dtype=DT)
    s_curr = torch.tensor([2.0, 3.0, 10.0], dtype=DT)        # +2 (fwd), -2 (back), 0 (still)
    lp = R8.line_progress_reward(s_curr, s_prev, 4.0)
    assert torch.allclose(lp, torch.tensor([8.0, 0.0, 0.0], dtype=DT))


def test_speed_profile_penalises_offspeed_advancing_only():
    """speed term penalises |v - v_ref| beyond the dead-band, ONLY while advancing along the line."""
    speed = torch.tensor([10.0, 40.0, 31.0, 10.0], dtype=DT)
    vref = torch.full((4,), 30.0, dtype=DT)
    ds = torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=DT)       # last one is backward -> gated off
    sp = R8.speed_profile_reward(speed, vref, ds, 1.0, 2.0)  # rw=1, tol=2
    # |10-30|-2=18 -> -18 ; |40-30|-2=8 -> -8 ; |31-30|<2 -> 0 ; backward -> 0
    assert torch.allclose(sp, torch.tensor([-18.0, -8.0, 0.0, 0.0], dtype=DT))


def test_off_is_exact_zero_byte_identical():
    """weight 0.0 -> the EXACT zero tensor (right shape/dtype) for BOTH terms (the inc8 OFF contract)."""
    s_prev = torch.linspace(0, 9, 7, dtype=DT)
    s_curr = s_prev + 1.0
    speed = torch.full((7,), 20.0, dtype=DT)
    vref = torch.full((7,), 30.0, dtype=DT)
    ds = s_curr - s_prev
    lp0 = R8.line_progress_reward(s_curr, s_prev, 0.0)
    sp0 = R8.speed_profile_reward(speed, vref, ds, 0.0, 2.0)
    for z in (lp0, sp0):
        assert z.shape == (7,) and z.dtype == DT
        assert torch.count_nonzero(z) == 0
