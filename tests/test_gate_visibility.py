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


# ================================================================================================
# (h) APPARENT PROJECTED AREA (Fengyou 2026-07-07): the vision-faithful "how square-on" cue. Normalized
#     projected inner-opening area in [0,1], 1 == square-on, range-invariant, monotone falloff with tilt.
# ================================================================================================
def _apparent_headon(range_m, tilt_deg=0.0):
    """gate_apparent_area for ONE gate straight ahead at range_m (Z-up), yaw=tilt (the gate-plane tilt
    relative to the view ray), drone level at origin."""
    drone_pos = torch.zeros(1, 3, dtype=DT)
    import math
    gate_pos = torch.tensor([[[float(range_m), 0.0, 0.0]]], dtype=DT)
    gate_yaw = torch.full((1, 1), math.radians(tilt_deg), dtype=DT)
    return GV.gate_apparent_area(drone_pos, _level_pose(), gate_pos, gate_yaw, is_quat=False)[0, 0].item()


def test_apparent_area_square_on_is_one_and_range_invariant():
    """A head-on (square-on) opening reads ~1 at EVERY range (the range normalization cancels the
    inverse-square shrink) -- exactly 'normalized to gate size so max square-on square is 1'."""
    for r in (5.0, 10.0, 15.0, 25.0):
        a = _apparent_headon(r, 0.0)
        assert a == pytest.approx(1.0, abs=1e-3), (r, a)


def test_apparent_area_monotone_falloff_with_tilt_and_in_range():
    """Tilting the gate plane off the view ray foreshortens the projected opening -> the ratio DROPS
    monotonically (0 deg -> 1, ~60 deg -> ~0.6, ~80 deg -> ~0.2), staying in [0,1]. It is FLATTER than a
    pure cosine near square-on (tolerant of small misalignments) -- the projected-area, not |cos|, model."""
    vals = [_apparent_headon(15.0, d) for d in (0, 20, 40, 60, 80)]
    for a in vals:
        assert 0.0 <= a <= 1.0, vals
    # strictly decreasing from 40 deg on (0 and 20 both saturate at ~1 -- the tolerant near-square-on band)
    assert vals[0] == pytest.approx(1.0, abs=1e-3)
    assert vals[2] > vals[3] > vals[4], vals               # 40 > 60 > 80
    assert vals[3] < 0.75 and vals[4] < 0.35, vals          # clearly foreshortened at 60/80 deg


def test_apparent_area_gate_behind_camera_is_zero():
    """A gate BEHIND the camera (no corner in front) has a meaningless projection -> area 0."""
    drone_pos = torch.zeros(1, 3, dtype=DT)
    import math
    gate_pos = torch.tensor([[[-15.0, 0.0, 0.0]]], dtype=DT)     # behind the +x-facing level drone
    gate_yaw = torch.full((1, 1), math.pi, dtype=DT)
    a = GV.gate_apparent_area(drone_pos, _level_pose(), gate_pos, gate_yaw, is_quat=False)[0, 0].item()
    assert a == 0.0, a


# ================================================================================================
# (i) MOTION-BLUR helpers (PERCEPTION-HONESTY package, 2026-07-10): gate_los_perp_rate = the
#     LOS-perpendicular angular rate driving image sweep; blur_extra_miss_prob = the soft-band ramp.
#     gate_detectable itself is UNTOUCHED by the package (pinned in tests/test_perception_honesty.py).
# ================================================================================================
def test_los_perp_rate_zero_omega_is_zero():
    p = torch.zeros(1, 3, dtype=DT)
    g = torch.tensor([[[12.0, 3.0, -2.0]]], dtype=DT)
    w = torch.zeros(1, 3, dtype=DT)
    r = GV.gate_los_perp_rate(p, _level_pose(), g, w, is_quat=False)
    assert r.shape == (1, 1)
    assert r[0, 0].item() == pytest.approx(0.0, abs=1e-12)


def test_los_perp_rate_roll_about_los_is_blur_free():
    """omega PARALLEL to the LOS (pure roll with the gate dead on the roll axis) sweeps the gate's
    line of sight by ~0 -- roll-about-LOS is first-order blur-free (documented limitation; the A2
    curve may later swap the abscissa for plain ||omega||)."""
    p = torch.zeros(1, 3, dtype=DT)
    g = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT)          # gate dead ahead on body +x
    w = torch.tensor([[5.0, 0.0, 0.0]], dtype=DT)             # pure roll about +x == the LOS
    r = GV.gate_los_perp_rate(p, _level_pose(), g, w, is_quat=False)
    assert r[0, 0].item() == pytest.approx(0.0, abs=1e-9)


def test_los_perp_rate_pure_yaw_geometry():
    """Pure yaw w_z: a gate dead ahead OR abeam (both LOS in the horizontal plane, ⟂ z) sweeps at the
    full |w_z|; a gate elevated 45 deg sweeps at |w_z|·sin(angle(omega, LOS)) = |w_z|·sin(45)."""
    p = torch.zeros(3, 3, dtype=DT)
    g = torch.tensor([[[10.0, 0.0, 0.0]],
                      [[0.0, 10.0, 0.0]],
                      [[10.0, 0.0, 10.0]]], dtype=DT)
    w = torch.tensor([[0.0, 0.0, 2.0]] * 3, dtype=DT)
    R = torch.eye(3, dtype=DT).unsqueeze(0).expand(3, 3, 3)
    r = GV.gate_los_perp_rate(p, R, g, w, is_quat=False)
    assert r[0, 0].item() == pytest.approx(2.0, abs=1e-9)              # dead ahead
    assert r[1, 0].item() == pytest.approx(2.0, abs=1e-9)              # abeam
    import math as _m
    assert r[2, 0].item() == pytest.approx(2.0 * _m.sin(_m.pi / 4), abs=1e-9)  # elevated 45 deg


def test_los_perp_rate_quat_matrix_parity_and_batched_shape():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=DT)
    R = torch.eye(3, dtype=DT).unsqueeze(0).expand(2, 3, 3)
    p = torch.zeros(2, 3, dtype=DT)
    g = torch.tensor([[[10.0, 2.0, 1.0], [15.0, -3.0, 0.5]],
                      [[8.0, 0.0, -1.0], [20.0, 5.0, 2.0]]], dtype=DT)
    w = torch.tensor([[0.5, -1.0, 2.0], [3.0, 0.0, -0.5]], dtype=DT)
    rq = GV.gate_los_perp_rate(p, q, g, w, is_quat=True)
    rR = GV.gate_los_perp_rate(p, R, g, w, is_quat=False)
    assert rq.shape == (2, 2) and rR.shape == (2, 2)
    assert torch.allclose(rq, rR, atol=1e-9)


def test_los_perp_rate_invariant_under_rigid_camera_flip():
    """The pi-about-body-z camera virtual flip is a RIGID rotation of the mount: expressing the same
    physical state in the flipped frame (R_wb @ Rz(pi), rates rotated accordingly) leaves the
    LOS-perpendicular rate magnitude IDENTICAL -- why the env can (and does) pass the UNFLIPPED
    self._q here even though detection itself uses the flipped camera."""
    import math as _m
    Rz = torch.tensor([[_m.cos(_m.pi), -_m.sin(_m.pi), 0.0],
                       [_m.sin(_m.pi), _m.cos(_m.pi), 0.0],
                       [0.0, 0.0, 1.0]], dtype=DT)
    R_wb = _level_pose()[0]                                            # (3,3)
    p = torch.zeros(1, 3, dtype=DT)
    g = torch.tensor([[[10.0, 4.0, -2.0]]], dtype=DT)
    w = torch.tensor([[0.7, -1.3, 2.1]], dtype=DT)
    r_unflipped = GV.gate_los_perp_rate(p, R_wb.unsqueeze(0), g, w, is_quat=False)
    # flipped frame: body axes rotated by Rz -> same physical rates are Rz^T @ w in the new frame.
    r_flipped = GV.gate_los_perp_rate(p, (R_wb @ Rz).unsqueeze(0), g,
                                      (Rz.T @ w[0]).unsqueeze(0), is_quat=False)
    assert torch.allclose(r_unflipped, r_flipped, atol=1e-9)


def test_blur_extra_miss_prob_ramp():
    """The PLACEHOLDER soft-band ramp: 0 below lo (blur-free band), miss_max at/above hi, monotone
    non-decreasing, clamped to [0, miss_max]. This ONE function is the A2-curve slot-in point."""
    rates = torch.tensor([[0.0, 1.99, 2.0, 3.0, 4.0, 9.0]], dtype=DT)
    m = GV.blur_extra_miss_prob(rates, 2.0, 4.0, 1.0)
    assert m[0, 0].item() == 0.0 and m[0, 1].item() == 0.0             # free below lo
    assert m[0, 2].item() == pytest.approx(0.0, abs=1e-9)              # ramp starts at lo
    assert m[0, 3].item() == pytest.approx(0.5, abs=1e-9)              # midpoint
    assert m[0, 4].item() == pytest.approx(1.0, abs=1e-9)              # miss_max at hi
    assert m[0, 5].item() == pytest.approx(1.0, abs=1e-9)              # clamped above hi
    diffs = m[0, 1:] - m[0, :-1]
    assert bool((diffs >= -1e-12).all())                               # monotone non-decreasing
    # miss_max scales the plateau.
    m2 = GV.blur_extra_miss_prob(rates, 2.0, 4.0, 0.4)
    assert m2[0, 5].item() == pytest.approx(0.4, abs=1e-9)


def test_blur_extra_miss_prob_degenerate_lo_equals_hi_is_a_step():
    """lo == hi (a possible A2 recalibration outcome): the 1e-9 denom guard turns the ramp into a
    STEP -- 0 at/below the threshold, miss_max above it. Pinned because the A2 recalibration touches
    exactly these numbers; note the env's hard cut (rate < hi is strict) already kills detectability
    AT rate == hi, so the 0-extra-miss AT the threshold is never load-bearing."""
    rates = torch.tensor([[1.0, 2.0, 2.0 + 1e-6, 5.0]], dtype=DT)
    m = GV.blur_extra_miss_prob(rates, 2.0, 2.0, 1.0)
    assert m[0, 0].item() == 0.0
    assert m[0, 1].item() == 0.0                                       # exactly AT lo==hi: still 0
    assert m[0, 2].item() == pytest.approx(1.0)                        # any epsilon above: miss_max
    assert m[0, 3].item() == pytest.approx(1.0)


def test_hard_cut_boundary_rate_exactly_hi_is_undetectable():
    """The env composes ``det & (rate < hi)`` -- STRICT '<', so rate exactly == hi is already
    blur-cut. Continuity with the soft band holds because blur_extra_miss_prob(hi) == miss_max
    (== 1.0 in the stages): the miss probability reaches 1 exactly where the hard cut takes over.
    miss_max is a stage knob -- if it is ever set < 1.0 there is a discontinuity AT hi (soft band
    tops out below certain-miss, then the hard cut kills it): pinned here so the boundary semantics
    are explicit for the A2 recalibration owner."""
    hi = 4.0
    rate = torch.tensor([[hi]], dtype=DT)
    det = torch.tensor([[True]])
    assert not bool((det & (rate < hi))[0, 0])                         # == hi -> hard-cut
    assert GV.blur_extra_miss_prob(rate, 2.0, hi, 1.0)[0, 0].item() == pytest.approx(1.0)
    # a hair below hi survives the hard cut with near-saturated soft miss (continuous handover).
    rate2 = torch.tensor([[hi - 1e-9]], dtype=DT)
    assert bool((det & (rate2 < hi))[0, 0])
