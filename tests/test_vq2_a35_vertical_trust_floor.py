"""A35 — magnitude-gated vertical trust floor (Fix 2) + settle hold-hover floor (Fix 3).

Run 20260704_024434 (post-A34, clean feed): the drone climbed ABOVE gate 1 and hit the ceiling
because as the +20 deg camera lost the now-below gate, bearing_w collapsed to 0.04 -> the A32 soft
weight crushed the honest +4.7 m z_off innovation to weight 0.02 -> z_off_est drifted to the WRONG
SIGN (-3.5 m, "climb more") for 96 ticks -> the PD held ~hover and never used V-2's open floor. Also
the settle/anchor hold band [0.90,1.12]x hover let collective sit sub-hover -> a startup dip to the
ground before egress.

Fix 2 (``use_zoff_big_trust``): a LARGE (>=zoff_big_innov_m) + sign-consistent (zoff_big_sign_consec)
z_off innovation is a REAL excursion -> floor its correction weight (>=zoff_big_innov_min_w), reseed
even from low bearing weight (the exception to A33 V-1's block), and freeze the propagate while
distrusted. NARROW: small/noisy offsets keep the A32/A28 behaviour byte-for-byte.

Fix 3 (``hold_thrust_lo_frac`` 0.90 -> 1.00 in vq2_case_c): the settle/anchor hold floors collective
at hover -> never sinks off the line.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlCommand, ControlMode  # noqa: E402
from racer.deploy_profile import vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402
from racer.vertical_estimator import VerticalEstimator  # noqa: E402


def _seeded(**overrides):
    base = dict(use_zoff_filter=True, export_clip_mps=2.5, use_soft_innov_weight=True,
                zoff_reseed_min_w=0.3)
    base.update(overrides)
    ve = VerticalEstimator(**base)
    ve.seed()
    for _ in range(60):
        ve.predict(0.0, 0.02)   # close the bias-capture window
    ve.latch_offset(0.0, 0.05, weight=1.0)   # clean initial lock at the flight path
    return ve


def _sustained_climb(ve, offset=4.7, weight=0.04, n=30, a_up=-0.5):
    """Simulate the drone climbing above the gate: n fresh latches all saying +offset at LOW bearing
    weight (gate leaving the up-tilted FOV), with a drifting IMU vz between latches."""
    for _ in range(n):
        for _ in range(3):
            ve.predict(a_up, 0.03)
        ve.latch_offset(offset, 0.05, weight=weight)


# ---------------------------------------------------------------------------
# Fix 2 — the ceiling killer
# ---------------------------------------------------------------------------
def test_a34_path_diverges_to_wrong_sign():
    """Reproduce the flight failure: without the trust floor, a sustained large +offset at low
    bearing weight drives z_off_est NEGATIVE (wrong sign, 'climb more')."""
    ve = _seeded(use_zoff_big_trust=False)
    _sustained_climb(ve)
    assert ve.z_off < 0.0, f"A34 path should diverge to wrong sign, got {ve.z_off:+.2f}"


def test_a35_tracks_the_true_offset_sign():
    """The fix: z_off_est tracks toward the true +4.7 m (correct sign, drone above -> descend)."""
    ve = _seeded(use_zoff_big_trust=True)
    _sustained_climb(ve)
    assert ve.z_off > 2.0, f"A35 must track the +4.7 excursion, got {ve.z_off:+.2f}"
    # and within a few metres of the truth (the clamp at the PD bounds authority anyway)
    assert abs(ve.z_off - 4.7) < 3.0, f"A35 z_off should be near +4.7, got {ve.z_off:+.2f}"


def test_a35_pd_gate_term_flips_to_descend():
    """With the correct sign, the PD gate-term (-kp_gate*clip(z_off,+/-3)) becomes NEGATIVE ->
    commands descent toward the V-2 floor, instead of holding/climbing."""
    kp, clip = 0.04, 3.0
    ve34 = _seeded(use_zoff_big_trust=False); _sustained_climb(ve34)
    ve35 = _seeded(use_zoff_big_trust=True);  _sustained_climb(ve35)
    tg34 = -kp * float(np.clip(ve34.z_off, -clip, clip))
    tg35 = -kp * float(np.clip(ve35.z_off, -clip, clip))
    assert tg34 >= 0.0, f"A34 term should hold/climb (>=0), got {tg34:+.4f}"
    assert tg35 < 0.0, f"A35 term should DEMAND descent (<0), got {tg35:+.4f}"


def test_a35_freeze_engages_while_distrusted_and_large():
    """The propagate-freeze (c): while a large offset is distrusted (low bearing weight), the
    predict step must NOT drift z_off on the unfed IMU vz."""
    ve = _seeded(use_zoff_big_trust=True)
    # build up the big+persistent+consistent state at low weight
    _sustained_climb(ve, n=5)
    # once engaged, a run of pure predict() (no latch) must NOT drive z_off away
    z_before = ve.z_off
    for _ in range(20):
        ve.predict(-1.0, 0.03)   # a strong wrong-signed vz that WOULD drift z_off if not frozen
    # z_off should be ~held (frozen), not driven far by the fabricated vz
    assert abs(ve.z_off - z_before) < 1.5, \
        f"freeze must hold z_off (~{z_before:+.2f}), drifted to {ve.z_off:+.2f}"


def test_a35_small_offsets_keep_a32_behaviour_byte_identical():
    """NARROW exception: small/noisy offsets never trip the big-trust path -> byte-identical to the
    pure-A34 estimator (does not undo A32's frame-starvation fix / A28 calm vertical)."""
    def run(big_trust):
        ve = _seeded(use_zoff_big_trust=big_trust)
        tr = []
        for k in range(40):
            for _ in range(3):
                ve.predict(-0.3, 0.03)
            ve.latch_offset(0.4, 0.05, weight=0.9)   # small offset, good weight -> normal path
            tr.append(round(ve.z_off, 9))
        return tr
    assert run(True) == run(False), "small-offset behaviour must be byte-identical (big-trust inert)"


def test_a35_default_off_is_byte_identical():
    """use_zoff_big_trust defaults False; a full mixed stream reproduces the pure-A34 estimator."""
    def run(with_field):
        kw = dict(use_zoff_filter=True, export_clip_mps=2.5, use_soft_innov_weight=True,
                  zoff_reseed_min_w=0.3)
        if with_field:
            kw["use_zoff_big_trust"] = False   # explicit default
        ve = VerticalEstimator(**kw); ve.seed()
        for _ in range(60):
            ve.predict(0.0, 0.02)
        ve.latch_offset(0.0, 0.05, weight=1.0)
        tr = []
        for k in range(40):
            for _ in range(3):
                ve.predict(-0.5, 0.03)
            o, w = (0.5, 0.9) if k % 3 else (4.7, 0.04)
            ve.latch_offset(o, 0.05, weight=w)
            tr.append(round(ve.z_off, 9))
        return tr
    assert run(True) == run(False)
    assert VerticalEstimator().use_zoff_big_trust is False


# ---------------------------------------------------------------------------
# Fix 3 — settle hold-hover floor
# ---------------------------------------------------------------------------
def _hold_seeker():
    ov = vq2_case_c().seeker_overrides
    return GateSeeker(config=GateSeekerConfig(**ov),
                      controller=make_seeker_controller(**(vq2_case_c().controller_overrides or {})))


def test_fix3_settle_hold_never_sub_hover():
    """With hold_thrust_lo_frac=1.0 the settle/anchor hold floors collective at hover -> no sink."""
    s = _hold_seeker()
    hover = s.controller.hover_thrust
    for raw in (0.0, 0.15, 0.20, 0.239, hover, 0.30, 0.5):
        cmd = ControlCommand(mode=ControlMode.BODY_RATE, sim_time_ns=0,
                             body_rate=np.zeros(3), thrust=raw)
        out = s._bound_hold_thrust(cmd)
        assert out.thrust >= hover - 1e-9, f"hold floored below hover: raw {raw} -> {out.thrust}"


def test_fix3_profile_wiring_and_default():
    ov = vq2_case_c().seeker_overrides
    assert ov["hold_thrust_lo_frac"] == 1.0        # Fix 3
    assert ov["hold_thrust_hi_frac"] == 1.12       # A31 anti-swell cap kept
    # field default unchanged (byte-identical off-path)
    assert GateSeekerConfig().hold_thrust_lo_frac == 0.6
    assert vq1_case_a().seeker_overrides is None


# ---------------------------------------------------------------------------
# Profile wiring (Fix 2)
# ---------------------------------------------------------------------------
def test_fix2_profile_wiring():
    veo = vq2_case_c().nav_config.vertical_estimator_overrides
    assert veo["use_zoff_big_trust"] is True
    assert veo["use_soft_innov_weight"] is True and veo["zoff_reseed_min_w"] == 0.3
    # defaults off
    e = VerticalEstimator()
    assert e.use_zoff_big_trust is False
    assert e.zoff_big_innov_m == 2.5 and e.zoff_big_innov_min_w == 0.5 and e.zoff_big_sign_consec == 3
