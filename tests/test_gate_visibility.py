"""HAND-COMPUTED tests for rl/gate_visibility.py -- the keypoint-visibility + occlusion model.

The load-bearing claim of the no-map architecture: when the drone is inside the CURRENT gate's
emergent blackout (the gate has left the FOV under the 20 deg camera up-tilt), the NEXT gate
10-20 m down-course is still detectable. These tests pin that, plus the emergent-blackout range,
the max detection range, off-center dropouts, and image-space annulus occlusion -- all from truth
geometry (no noise, no estimate).

Frame convention (matches rl.peregrine_racing): inputs are DiffAero Z-up / FLU. A LEVEL drone with
identity body->world matrix faces +x (down-course); a gate at yaw=0 straight ahead is centered in
azimuth. The camera is mounted 20 deg pitch-UP, so a gate at the drone's own altitude sits in the
LOWER part of the image -- the source of the emergent blackout when it drops out the bottom at close
range, and of the up/down asymmetry (gates ABOVE stay in view further than gates BELOW).

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_gate_visibility.py -q
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import gate_visibility as GV                                             # noqa: E402

DT = torch.float64


# ---- shared helpers -----------------------------------------------------------------------------
def _level_pose():
    """Identity body->world (Z-up): level drone facing +x (down-course)."""
    return torch.eye(3, dtype=DT).unsqueeze(0)


def _n_vis_centered(range_m, z_off=0.0, y_off=0.0):
    """n_visible_corners for ONE centered gate at (range_m, y_off, z_off) Z-up, drone level at origin."""
    drone_pos = torch.zeros(1, 3, dtype=DT)
    gate_pos = torch.tensor([[[float(range_m), float(y_off), float(z_off)]]], dtype=DT)
    gate_yaw = torch.zeros(1, 1, dtype=DT)
    det, n = GV.gate_detectable(drone_pos, _level_pose(), gate_pos, gate_yaw, is_quat=False)
    return int(n[0, 0]), bool(det[0, 0])


# ================================================================================================
# (a) EMERGENT BLACKOUT: centered, level gate at the drone's altitude. Below some range the gate
#     leaves the FOV (bottom + sides) under the 20 deg up-tilt and < 4 corners remain in-frame.
# ================================================================================================
def test_emergent_blackout_range_centered_level_gate():
    # Bisect the boundary where >= 4 corners are in-frame (detectable) for a centered same-height gate.
    def has_4(range_m):
        n, _ = _n_vis_centered(range_m)
        return n >= 4

    lo, hi = 0.5, 3.0
    assert not has_4(lo) and has_4(hi)          # bracket the transition
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if has_4(mid):
            hi = mid
        else:
            lo = mid
    blackout_range = hi
    # MEASURED (not hardcoded to a guess): ~1.29 m. Pin with a tolerance around the computed value.
    assert abs(blackout_range - 1.289) < 0.02, blackout_range
    # concretely: just inside the blackout < 4 corners; just outside >= 4.
    assert _n_vis_centered(blackout_range - 0.05)[0] < 4
    assert _n_vis_centered(blackout_range + 0.05)[0] >= 4
    print(f"\n[a] emergent blackout (centered level same-height gate) = {blackout_range:.4f} m")


# ================================================================================================
# (b) MAX DETECTION RANGE: a centered gate stays detectable out to the 30 m far cap; the cap (not
#     the FOV) is what ends it -- at 30 m all 8 corners project in-frame; at 30.01 m the centre-range
#     cap trips.
# ================================================================================================
def test_max_detection_range_is_far_cap_limited():
    # Well inside the cap: all 8 corners in-frame.
    n30, d30 = _n_vis_centered(30.0)
    assert d30 and n30 == 8, (n30, d30)
    # Just past the 30 m cap: still 8 corners project in-frame, but the far cap makes it undetectable.
    n31, d31 = _n_vis_centered(30.01)
    assert n31 == 8 and not d31, (n31, d31)
    # A custom, tighter cap ends detection earlier even with 8 in-frame corners.
    drone_pos = torch.zeros(1, 3, dtype=DT)
    gate_pos = torch.tensor([[[20.0, 0.0, 0.0]]], dtype=DT)
    gate_yaw = torch.zeros(1, 1, dtype=DT)
    det, n = GV.gate_detectable(drone_pos, _level_pose(), gate_pos, gate_yaw,
                                far_cap_m=15.0, is_quat=False)
    assert int(n[0, 0]) == 8 and not bool(det[0, 0])       # in-frame but beyond the 15 m cap
    print(f"\n[b] max detection range = 30 m (far cap); FOV alone keeps 8 corners to >=30 m")


# ================================================================================================
# (c) OFF-CENTER DROPOUT: gates far off the bearing leave the frame. The 20 deg up-tilt gives a
#     characteristic asymmetry -- a gate BELOW drops out sooner than one equally far ABOVE.
# ================================================================================================
def test_offcenter_gates_drop_out():
    rng = 15.0
    assert _n_vis_centered(rng)[1]                              # centered: detectable
    # lateral: at |y|=16 m (bearing ~47 deg > HFoV/2=45 deg) the gate is out of frame.
    assert not _n_vis_centered(rng, y_off=16.0)[1]
    assert not _n_vis_centered(rng, y_off=-16.0)[1]
    assert _n_vis_centered(rng, y_off=12.0)[1]                  # still in at 12 m lateral
    # vertical: the up-tilt makes a gate BELOW (z=-10) fully invisible while an equal gate ABOVE
    # (z=+10) is still fully visible -- the signature of the 20 deg mount.
    n_low, d_low = _n_vis_centered(rng, z_off=-10.0)
    n_high, d_high = _n_vis_centered(rng, z_off=+10.0)
    assert not d_low and n_low == 0, (n_low, d_low)
    assert d_high and n_high == 8, (n_high, d_high)
    # push UP until it exits the top: z=+20 out.
    assert not _n_vis_centered(rng, z_off=+20.0)[1]
    print(f"\n[c] off-center: lateral out at |y|>=16 m; vertical asymmetry z=-10 OUT / z=+10 IN")


# ================================================================================================
# (d) OCCLUSION: a near gate whose projected frame annulus covers the far gate's corners makes the
#     far gate undetectable -- even though the far gate alone is comfortably detectable.
# ================================================================================================
def test_near_gate_occludes_far_gate():
    drone_pos = torch.zeros(1, 3, dtype=DT)
    # far gate at 20 m offset +5 m in y so it projects behind the near gate's frame ring;
    # near gate at 4 m centered (big projected annulus).
    far_xyz = [20.0, 5.0, 0.0]
    near_xyz = [4.0, 0.0, 0.0]

    # far gate ALONE: comfortably detectable (all 8 corners in-frame).
    gp_alone = torch.tensor([[far_xyz]], dtype=DT)
    gy_alone = torch.zeros(1, 1, dtype=DT)
    d_alone, n_alone = GV.gate_detectable(drone_pos, _level_pose(), gp_alone, gy_alone, is_quat=False)
    assert bool(d_alone[0, 0]) and int(n_alone[0, 0]) == 8, (n_alone, d_alone)

    # with the near gate in front: the far gate's corners fall in the near gate's projected annulus
    # (outer quad minus inner hole) -> occluded -> far gate undetectable. Near gate stays detectable.
    gp = torch.tensor([[near_xyz, far_xyz]], dtype=DT)
    gy = torch.zeros(1, 2, dtype=DT)
    det, n = GV.gate_detectable(drone_pos, _level_pose(), gp, gy, is_quat=False)
    assert bool(det[0, 0]), "near gate should still be detectable"
    assert not bool(det[0, 1]), "far gate should be occluded to undetectable"
    assert int(n[0, 1]) < int(n_alone[0, 0]), "occlusion must reduce far-gate visible corners"
    print(f"\n[d] occlusion: far gate {int(n_alone[0,0])} corners alone -> "
          f"{int(n[0,1])} behind the near gate's annulus (undetectable)")


def test_far_gate_seen_through_near_opening_is_not_occluded():
    """Sanity on the annulus rule: a far gate that projects INSIDE the near gate's opening (the hole,
    not the frame material) is seen THROUGH it and stays detectable -- occlusion is the annulus only."""
    drone_pos = torch.zeros(1, 3, dtype=DT)
    gp = torch.tensor([[[6.0, 0.0, 0.0], [18.0, 0.0, 0.0]]], dtype=DT)   # both centered, colinear
    gy = torch.zeros(1, 2, dtype=DT)
    det, n = GV.gate_detectable(drone_pos, _level_pose(), gp, gy, is_quat=False)
    assert bool(det[0, 1]) and int(n[0, 1]) == 8, (n, det)   # far gate seen through the near hole


# ================================================================================================
# (e) THE KEY ASSERTION: with two gates 10-20 m apart along the course, when the drone is inside the
#     CURRENT (near) gate's emergent blackout, the NEXT (far) gate is still detectable. This is the
#     load-bearing claim of the whole no-map architecture.
# ================================================================================================
@pytest.mark.parametrize("spacing", [10.0, 15.0, 20.0])
def test_next_gate_visible_while_current_gate_blacked_out(spacing):
    drone_pos = torch.zeros(1, 3, dtype=DT)
    # Drone 1.0 m up-course of the current gate => inside its emergent blackout (~1.29 m boundary),
    # both gates at the drone's altitude, straight ahead (the course line), yaw=0.
    near_x = 1.0
    gp = torch.tensor([[[near_x, 0.0, 0.0], [near_x + spacing, 0.0, 0.0]]], dtype=DT)
    gy = torch.zeros(1, 2, dtype=DT)
    det, n = GV.gate_detectable(drone_pos, _level_pose(), gp, gy, far_cap_m=30.0, is_quat=False)

    # current gate: blacked out (< 4 corners, not detectable)
    assert int(n[0, 0]) < 4 and not bool(det[0, 0]), (
        f"current gate should be in emergent blackout, got n={int(n[0,0])}")
    # next gate: detectable (within 30 m, >= 4 corners) -- THE load-bearing claim
    assert bool(det[0, 1]) and int(n[0, 1]) >= 4, (
        f"next gate must stay detectable through the blackout, got n={int(n[0,1])}")
    print(f"\n[e] spacing={spacing:.0f} m: current n={int(n[0,0])} (blackout) | "
          f"next n={int(n[0,1])} det={bool(det[0,1])}")


# ================================================================================================
# Batching + input-form parity (the model is used batched over ~2048 envs; quat or matrix pose).
# ================================================================================================
def test_batched_and_quat_matrix_parity():
    # batch of 3 envs x 2 gates, mixed float32 (the env's dtype).
    drone_pos = torch.zeros(3, 3, dtype=torch.float32)
    gp = torch.tensor([
        [[15.0, 0.0, 0.0], [16.0, 0.0, 0.0]],
        [[15.0, 16.0, 0.0], [16.0, 0.0, 0.0]],     # env1 gate0 laterally out
        [[15.0, 0.0, -10.0], [16.0, 0.0, 0.0]],    # env2 gate0 below -> out
    ], dtype=torch.float32)
    gy = torch.zeros(3, 2, dtype=torch.float32)
    R = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(3, 3, 3)
    det, n = GV.gate_detectable(drone_pos, R, gp, gy, is_quat=False)
    assert det.shape == (3, 2) and n.shape == (3, 2)
    assert bool(det[0, 0]) and not bool(det[1, 0]) and not bool(det[2, 0])
    assert bool(det[0, 1]) and bool(det[1, 1]) and bool(det[2, 1])

    # identity XYZW quat == identity matrix pose.
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DT)
    R1 = torch.eye(3, dtype=DT).unsqueeze(0)
    gp1 = torch.tensor([[[15.0, 3.0, 2.0]]], dtype=DT)
    gy1 = torch.zeros(1, 1, dtype=DT)
    dp = torch.zeros(1, 3, dtype=DT)
    dq, nq = GV.gate_detectable(dp, q, gp1, gy1, is_quat=True)
    dR, nR = GV.gate_detectable(dp, R1, gp1, gy1, is_quat=False)
    assert bool((nq == nR).all()) and bool((dq == dR).all())
