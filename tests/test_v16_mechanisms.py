"""Unit tests for the v1.6 config-gated mechanisms (all default-OFF == byte-identical to v1.5):

  1 CAMERA MOUNT re-aim (cam_mount_pitch_delta_R): the +20 deg mount is ALREADY baked in
    R_camera_from_body, so this knob is an ABSOLUTE override. Sign + visibility: at the baked/None/20.0
    setting a gate 25 deg ABOVE body-forward is detectable while 25 deg BELOW is NOT (vertical half-FOV
    ~29.35 deg, optical axis +20); 0.0 removes the mount (the below gate becomes detectable); 20.0 and
    None both return a zero delta (byte-identical skip).
  2 RANGE detection p(r) (range_detect_prob): logistic near-blackout -> plateau -> far attenuation;
    p(r0)=0.5*plateau, p(mid)~plateau, p(>far_start)~plateau*far_p; and cadence_effective_detectable_p.
  4 APERTURE-MARGIN math + PASS->MISS reclassification (aperture_margin_reclassify): the weighted miss,
    the lateral-weighting asymmetry, and the full-aperture margin_start no-op guarantee.
  5 PITCH duty/jerk (pitch_duty_penalty / pitch_jerk_penalty): free-band zero-cost below the band,
    linear to the rail above it, L1 jerk, and OFF == byte-identical zeros.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_v16_mechanisms.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import peregrine_racing_ego as EGO                                       # noqa: E402
import ego_reward as ER                                                 # noqa: E402
import gate_visibility as GV                                            # noqa: E402
from peregrine_racing import crossing_events, world_to_gateframe        # noqa: E402
from inc8_estimator_emul import _RZ_PI_BODY_NP                          # noqa: E402

DT = torch.float64


# ================================================================================================
# Mechanism 1 -- CAMERA MOUNT re-aim (absolute override of the baked +20 deg).
# ================================================================================================
def _cam_R(target_mount):
    """Replicate the env's _cam_R_wb composition at identity attitude: cam = R_wb @ Rz_cam @ delta."""
    cam = torch.tensor(np.eye(3) @ np.array(_RZ_PI_BODY_NP), dtype=DT)
    delta = EGO.cam_mount_pitch_delta_R(target_mount, device=None, dtype=DT)
    if delta is not None:
        cam = cam @ delta
    return cam.unsqueeze(0)                                             # (1,3,3)


def _detectable_at_elevation(cam_R, el_deg, rng=10.0):
    """A gate at range ``rng`` on the camera's central azimuth (-x for identity+flip) at elevation
    ``el_deg`` (Z-up), facing back toward the drone. Returns bool detectable."""
    az0 = math.pi                                                       # camera looks along world -x
    e = math.radians(el_deg)
    gx = rng * math.cos(e) * math.cos(az0)
    gy = rng * math.cos(e) * math.sin(az0)
    gz = rng * math.sin(e)
    drone = torch.zeros(1, 3, dtype=DT)
    gate_pos = torch.tensor([[[gx, gy, gz]]], dtype=DT)
    gate_yaw = torch.tensor([[az0]], dtype=DT)
    det, _ = GV.gate_detectable(drone, cam_R, gate_pos, gate_yaw, far_cap_m=30.0, is_quat=False)
    return bool(det[0, 0])


def test_mount_none_and_20_are_zero_delta_byte_identical():
    # The baked mount is +20; both unset (None) and an explicit 20.0 must produce NO delta (skip).
    assert EGO.cam_mount_pitch_delta_R(None) is None
    assert EGO.cam_mount_pitch_delta_R(20.0) is None
    assert EGO.BAKED_CAM_MOUNT_PITCH_DEG == 20.0
    # a non-baked target DOES produce a delta
    assert EGO.cam_mount_pitch_delta_R(0.0, dtype=DT) is not None
    assert EGO.cam_mount_pitch_delta_R(25.0, dtype=DT) is not None


def test_mount_20deg_up_down_asymmetry():
    # spec test: with the +20 deg mount, a gate 25 deg ABOVE body-forward is detectable, 25 deg BELOW is
    # NOT (optical axis +20, vertical half-FOV ~29.35 -> band ~[-9,+49]). Baseline (None) == 20.0 == baked.
    for target in (None, 20.0):
        cam_R = _cam_R(target)
        assert _detectable_at_elevation(cam_R, +25.0) is True, target
        assert _detectable_at_elevation(cam_R, -25.0) is False, target


def test_mount_zero_removes_tilt_below_becomes_detectable():
    # Removing the mount (target 0) re-centres the FOV on body-forward -> the 25-deg-BELOW gate that was
    # dark at +20 is now detectable (band ~[-29,+29]); a symmetric +25 stays detectable.
    cam0 = _cam_R(0.0)
    assert _detectable_at_elevation(cam0, -25.0) is True
    assert _detectable_at_elevation(cam0, +25.0) is True


def test_mount_sign_higher_target_aims_higher():
    # Monotone: a HIGHER absolute mount pitches the optical axis UP -> a high gate (+45) that is out of
    # view at 0 deg comes into view at 45 deg, while a low gate (-25) drops out.
    assert _detectable_at_elevation(_cam_R(45.0), +45.0) is True
    assert _detectable_at_elevation(_cam_R(0.0), +45.0) is False


def test_mount_half_fov_vertical_matches_2935():
    # sanity: the projection's vertical half-angle (fy=320, H=360) is ~29.35 deg (the spec value).
    from inc8_estimator_emul import CAMERA_INTRINSICS_K_NP, IMAGE_HEIGHT
    fy = CAMERA_INTRINSICS_K_NP[1, 1]
    half_v = math.degrees(math.atan2(IMAGE_HEIGHT / 2.0, fy))
    assert abs(half_v - 29.35) < 0.1


# ================================================================================================
# Mechanism 2 -- RANGE-dependent detection probability.
# ================================================================================================
def test_range_detect_prob_anchors():
    r0, k, plat, far, farp = 2.2, 0.4, 0.99, 26.0, 0.79
    rng = torch.tensor([[0.0, 2.2, 10.0, 26.0, 30.0]], dtype=DT)
    p = EGO.range_detect_prob(rng, r0, k, plat, far, farp)[0]
    assert p[1].item() == pytest.approx(0.5 * plat, abs=1e-6)           # r == r0 -> 0.5*plateau
    assert p[2].item() == pytest.approx(plat, abs=1e-3)                 # mid -> plateau
    assert p[3].item() == pytest.approx(plat, abs=1e-3)                 # r == far_start (not >) -> plateau
    assert p[4].item() == pytest.approx(plat * farp, abs=1e-3)          # r > far_start -> plateau*far_p
    assert p[0].item() < 0.02                                           # near-gate blackout ~0


def test_range_detect_prob_monotone_up_to_far_start():
    rng = torch.linspace(0.0, 26.0, 50, dtype=DT).reshape(1, -1)
    p = EGO.range_detect_prob(rng, 2.2, 0.4, 0.99, 26.0, 0.79)[0]
    assert torch.all(p[1:] - p[:-1] >= -1e-9)                           # non-decreasing on [0, far_start]


def test_cadence_effective_detectable_p_gates():
    gen = torch.Generator().manual_seed(0)
    geo = torch.ones(4, 3, dtype=torch.bool)
    # off-frame -> all False
    out = EGO.cadence_effective_detectable_p(geo, False, torch.ones_like(geo, dtype=DT), gen, dtype=DT)
    assert not out.any()
    # p==1 everywhere -> equals the geometric mask; p==0 -> all False
    out1 = EGO.cadence_effective_detectable_p(geo, True, torch.ones_like(geo, dtype=DT), gen, dtype=DT)
    assert bool(out1.all())
    out0 = EGO.cadence_effective_detectable_p(geo, True, torch.zeros_like(geo, dtype=DT), gen, dtype=DT)
    assert not out0.any()


# ================================================================================================
# Mechanism 4 -- APERTURE-MARGIN weighted miss + PASS->MISS reclassification.
# ================================================================================================
def test_crossing_events_exposes_y_z():
    # a straight forward crossing at (y,z)=(0.4,-0.3): ev must expose the signed offsets that linf builds on.
    prev = torch.tensor([[[-1.0, 0.4, -0.3]]], dtype=DT)               # x<0
    curr = torch.tensor([[[+1.0, 0.4, -0.3]]], dtype=DT)               # x>=0 -> fwd crossing at x=0
    ev = crossing_events(prev, curr, 0.75, 1.36)
    assert bool(ev["fwd"][0, 0])
    assert ev["y"][0, 0].item() == pytest.approx(0.4, abs=1e-6)
    assert ev["z"][0, 0].item() == pytest.approx(-0.3, abs=1e-6)
    assert ev["linf"][0, 0].item() == pytest.approx(0.4, abs=1e-6)     # max(|y|,|z|)


def test_aperture_margin_reclassify_weighted_and_lateral_asymmetry():
    gate_passed = torch.tensor([True, True, True, True, False])
    lat = torch.tensor([0.0, 0.5, 0.0, 0.6, 5.0], dtype=DT)
    vert = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0], dtype=DT)
    # lat_weight=2, margin=1.0: m = [0, 1.0, 1.0, 1.2, (not a pass)]
    kept, down = EGO.aperture_margin_reclassify(gate_passed, lat, vert, 2.0, 1.0)
    assert down.tolist() == [False, False, False, True, False]         # only the 1.2 (lateral) downgraded
    assert kept.tolist() == [True, True, True, False, False]
    # lateral 0.5 (m=1.0, kept) vs vertical 1.0 (m=1.0, kept): the SAME m, but a lateral 0.6 (m=1.2) is
    # downgraded while a vertical 1.0 is kept -> lateral risk is weighted 2x (the asymmetry).


def test_aperture_margin_full_aperture_is_noop():
    # margin_start = half*sqrt(lat_weight^2+1) must NOT reclassify ANY geometric pass (linf < half).
    half, lw = 0.75, 2.0
    margin_start = half * math.sqrt(lw ** 2 + 1.0)
    # corner pass just inside the aperture on BOTH axes (the worst case for the weighted miss)
    gate_passed = torch.tensor([True])
    lat = torch.tensor([half - 1e-3], dtype=DT)
    vert = torch.tensor([half - 1e-3], dtype=DT)
    _, down = EGO.aperture_margin_reclassify(gate_passed, lat, vert, lw, margin_start)
    assert not bool(down.any())


# ================================================================================================
# Mechanism 5 -- PITCH duty / jerk (mirror of the yaw terms on channel 2).
# ================================================================================================
def test_pitch_duty_free_band_zero_below_linear_above():
    # below/at the free band -> EXACTLY 0
    below = torch.tensor([0.0, 0.3, 0.6], dtype=DT)
    assert torch.allclose(ER.pitch_duty_penalty(below, 0.1, 0.6), torch.zeros(3, dtype=DT))
    # at the rail (3.0) with band 0.6 -> excess 1.0 -> -rw_pitch_duty
    at_rail = torch.tensor([3.0], dtype=DT)
    assert ER.pitch_duty_penalty(at_rail, 0.1, 0.6).item() == pytest.approx(-0.1, abs=1e-6)
    # halfway in the excess band: |pitch|=1.8 -> excess=(1.8-0.6)/(3.0-0.6)=0.5 -> -0.05
    mid = torch.tensor([1.8], dtype=DT)
    assert ER.pitch_duty_penalty(mid, 0.1, 0.6).item() == pytest.approx(-0.05, abs=1e-6)


def test_pitch_duty_off_is_byte_identical_zeros():
    pc = torch.tensor([0.0, 2.0, 3.0], dtype=DT)
    assert torch.all(ER.pitch_duty_penalty(pc, 0.0, 0.6) == 0)


def test_pitch_jerk_l1_and_off():
    d = torch.tensor([0.0, 0.5, -1.0], dtype=DT)
    assert torch.allclose(ER.pitch_jerk_penalty(d, 0.03), -0.03 * d.abs())
    assert torch.all(ER.pitch_jerk_penalty(d, 0.0) == 0)               # OFF -> byte-identical


def test_pitch_weights_resolve_from_cfg():
    # from_cfg's generic loop must pick up rw_pitch_duty / rw_pitch_jerk and pitch_duty_free_band.
    class _Cfg:
        rw_pitch_duty = 0.2
        rw_pitch_jerk = 0.03
        pitch_duty_free_band = 0.6
    w = ER.EgoRewardWeights.from_cfg(_Cfg())
    assert w.pitch_duty == 0.2 and w.pitch_jerk == 0.03 and w.pitch_duty_free_band == 0.6
    # defaults (absent keys) -> OFF / wide band
    w0 = ER.EgoRewardWeights.from_cfg(type("C", (), {})())
    assert w0.pitch_duty == 0.0 and w0.pitch_jerk == 0.0 and w0.pitch_duty_free_band == 0.6


# ================================================================================================
# Mechanism 6 -- ROLL duty / jerk (2026-07-22 close-in roll limit-cycle fix; the BYTE-FOR-BYTE mirror of
# the pitch terms on channel 1, WIDER free band 0.8 -- roll is NOT the primary axis; the jerk term is the
# primary lever, a smooth turn-in pays ~0 while the growing-amplitude limit cycle pays the most).
# ================================================================================================
def test_roll_duty_free_band_zero_below_linear_above():
    # below/at the free band (WIDE 0.8) -> EXACTLY 0 (normal turn-in roll stays untaxed)
    below = torch.tensor([0.0, 0.4, 0.8], dtype=DT)
    assert torch.allclose(ER.roll_duty_penalty(below, 0.1, 0.8), torch.zeros(3, dtype=DT))
    # at the rail (3.0) with band 0.8 -> excess 1.0 -> -rw_roll_duty
    at_rail = torch.tensor([3.0], dtype=DT)
    assert ER.roll_duty_penalty(at_rail, 0.1, 0.8).item() == pytest.approx(-0.1, abs=1e-6)
    # halfway in the excess band: |roll|=1.9 -> excess=(1.9-0.8)/(3.0-0.8)=0.5 -> -0.05
    mid = torch.tensor([1.9], dtype=DT)
    assert ER.roll_duty_penalty(mid, 0.1, 0.8).item() == pytest.approx(-0.05, abs=1e-6)
    assert ER.ROLL_CMD_RAIL == 3.0                                      # same fixed authority rail as pitch


def test_roll_duty_off_is_byte_identical_zeros():
    rc = torch.tensor([0.0, 2.0, 3.0], dtype=DT)
    assert torch.all(ER.roll_duty_penalty(rc, 0.0, 0.8) == 0)


def test_roll_jerk_l1_and_off():
    d = torch.tensor([0.0, 0.5, -1.0], dtype=DT)
    assert torch.allclose(ER.roll_jerk_penalty(d, 0.03), -0.03 * d.abs())  # L1: a STEADY roll (0) pays 0
    assert torch.all(ER.roll_jerk_penalty(d, 0.0) == 0)                    # OFF -> byte-identical


def test_roll_penalties_are_negative_and_nonfarmable():
    # both terms are <= 0 for ANY input (pure penalties, no positive-reward path)
    rc = torch.tensor([0.0, 0.5, 1.5, 3.0, -2.4], dtype=DT)
    assert torch.all(ER.roll_duty_penalty(rc, 0.2, 0.8) <= 0)
    d = torch.tensor([0.0, 0.9, -2.0, 1.1], dtype=DT)
    assert torch.all(ER.roll_jerk_penalty(d, 0.05) <= 0)


def test_roll_weights_resolve_from_cfg():
    # from_cfg's generic loop must pick up rw_roll_duty / rw_roll_jerk and roll_duty_free_band.
    class _Cfg:
        rw_roll_duty = 0.15
        rw_roll_jerk = 0.04
        roll_duty_free_band = 0.8
    w = ER.EgoRewardWeights.from_cfg(_Cfg())
    assert w.roll_duty == 0.15 and w.roll_jerk == 0.04 and w.roll_duty_free_band == 0.8
    # defaults (absent keys) -> OFF / wide band
    w0 = ER.EgoRewardWeights.from_cfg(type("C", (), {})())
    assert w0.roll_duty == 0.0 and w0.roll_jerk == 0.0 and w0.roll_duty_free_band == 0.8
