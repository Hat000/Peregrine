"""Tests for rl/racing_line.py -- the online batched NON-OPTIMAL racing-line generator + GVF query.

Pins the properties the reward depends on: the line hits every gate centre, crosses each gate HEAD-ON
(tangent == gate normal at the centre), has monotone arc length (a valid progress potential), and the
GVF query returns the correct nearest-point arc length / cross-track / tangent.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from rl.racing_line import build_racing_line, gate_normals_from_yaw  # noqa: E402


def _yaw_to_normal(yaw):
    return torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])


def test_line_hits_spawn_and_gate_centre():
    spawn = torch.tensor([[0.0, 0.0, 0.0]])
    gate = torch.tensor([[[10.0, 0.0, 3.0]]])          # 10 m ahead, 3 m up
    yaw = torch.tensor([[0.0]])                        # faces +x (down-course)
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=32)
    assert torch.allclose(line.samples[:, 0, :], spawn, atol=1e-5)
    assert torch.allclose(line.samples[:, -1, :], gate[:, 0, :], atol=1e-5)


def test_head_on_tangent_at_gate_equals_normal():
    # gate normal must be the LINE tangent at the crossing (head-on approach, no angle)
    spawn = torch.tensor([[0.0, 0.0, 0.0]])
    gate = torch.tensor([[[12.0, 4.0, -5.0]]])         # off-axis (lateral + below)
    yaw = torch.tensor([[0.3]])
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=64)
    n = gate_normals_from_yaw(yaw)[0, 0]               # (3,)
    tang_end = line.tangent[0, -1, :]
    # tangent at the final sample aligns with the gate normal (cos ~ 1)
    cos = torch.dot(tang_end, n) / (tang_end.norm() * n.norm())
    assert cos > 0.999, f"gate crossing not head-on: cos={cos:.4f}"


def test_arclen_monotone_and_zero_start():
    spawn = torch.zeros(4, 3)
    gate = torch.randn(4, 1, 3) * 3.0 + torch.tensor([12.0, 0.0, 0.0])
    yaw = torch.zeros(4, 1)
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=24)
    assert torch.allclose(line.arclen[:, 0], torch.zeros(4), atol=1e-6)
    assert (line.arclen[:, 1:] - line.arclen[:, :-1] >= -1e-6).all()   # non-decreasing


def test_query_on_line_zero_perp_and_correct_s():
    spawn = torch.tensor([[0.0, 0.0, 0.0]])
    gate = torch.tensor([[[10.0, 0.0, 0.0]]])          # straight line along +x
    yaw = torch.tensor([[0.0]])
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=64)
    # a point exactly on the line (x=4) -> perp ~ 0, s ~ 4
    p = torch.tensor([[4.0, 0.0, 0.0]])
    s, perp, tang, _ = line.query(p)
    assert perp.item() < 1e-3, f"on-line perp not ~0: {perp.item()}"
    assert abs(s.item() - 4.0) < 0.1, f"on-line s wrong: {s.item()}"
    assert torch.dot(tang[0], torch.tensor([1.0, 0.0, 0.0])) > 0.999


def test_query_perpendicular_offset():
    spawn = torch.tensor([[0.0, 0.0, 0.0]])
    gate = torch.tensor([[[10.0, 0.0, 0.0]]])
    yaw = torch.tensor([[0.0]])
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=64)
    # 2 m off the line laterally at x=5 -> perp ~ 2, s ~ 5
    p = torch.tensor([[5.0, 2.0, 0.0]])
    s, perp, tang, _ = line.query(p)
    assert abs(perp.item() - 2.0) < 0.05, f"perp offset wrong: {perp.item()}"
    assert abs(s.item() - 5.0) < 0.2, f"s wrong: {s.item()}"


def test_query_inward_points_to_line():
    # inward unit (for the GVF alignment reward) must point FROM the drone TO the line, and be unit length
    spawn = torch.tensor([[0.0, 0.0, 0.0]])
    gate = torch.tensor([[[10.0, 0.0, 0.0]]])          # line along +x at y=z=0
    yaw = torch.tensor([[0.0]])
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=64)
    p = torch.tensor([[5.0, 0.0, 2.0]])                # 2 m ABOVE the line
    s, perp, tang, inw = line.query(p)
    assert inw[0, 2] < -0.9, f"inward not toward the line (should be -z): {inw[0].tolist()}"
    assert abs(inw[0].norm().item() - 1.0) < 1e-4       # unit
    assert abs((inw[0] * tang[0]).sum().item()) < 1e-4  # inward is perpendicular to the tangent


def test_query_env_idx_subset():
    # all gates straight ahead along +x (yaw 0, level) -> straight lines; on-axis points are on-line
    spawn = torch.zeros(3, 3)
    gate = torch.tensor([[[10.0, 0.0, 0.0]], [[8.0, 0.0, 0.0]], [[15.0, 0.0, 0.0]]])
    yaw = torch.zeros(3, 1)
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=32)
    idx = torch.tensor([0, 2])
    p = torch.tensor([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    s, perp, tang, _ = line.query(p, env_idx=idx)
    assert s.shape == (2,) and perp.shape == (2,) and tang.shape == (2, 3)
    assert (perp < 1e-3).all()                          # on-line -> ~0 cross-track
    assert abs(s[0].item() - 3.0) < 0.1 and abs(s[1].item() - 5.0) < 0.1


def test_multi_gate_hits_all_centres_head_on():
    spawn = torch.tensor([[0.0, 0.0, 0.0]])
    gate = torch.tensor([[[10.0, 0.0, 0.0], [22.0, 5.0, 2.0]]])   # 2 gates
    yaw = torch.tensor([[0.0, 0.5]])
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=64)
    # both gate centres appear as samples (nearest sample ~ 0 distance)
    for g in range(2):
        c = gate[0, g]
        d = torch.linalg.norm(line.samples[0] - c, dim=-1).min()
        assert d < 0.05, f"gate {g} centre not on line: min dist {d:.3f}"
    # tangent head-on at the FINAL gate
    n_last = gate_normals_from_yaw(yaw)[0, 1]
    cos = torch.dot(line.tangent[0, -1], n_last) / (line.tangent[0, -1].norm() * n_last.norm())
    assert cos > 0.999
