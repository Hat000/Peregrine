"""Unit tests for rl/inc8_reward.py -- the inc8 reward deltas + the BSR3 spin-gate (torch-pure).

Pins the shapes the design depends on: arc-progress linearity, the truth-seen GT anchor, the
annealed confidence shaping, the COWORK-3 2-axis terminal-lock perception reward (arms A/B/C, the
quartic visibility, the terminal ramp, progress-gating), and the BSR3 sustained-spin gate that does
NOT false-abort a legitimate ~11 rad/s super-rate transient. Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_inc8_reward.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import inc8_reward as RW                                              # noqa: E402

DT = torch.float64


def _t(x):
    return torch.tensor(x, dtype=DT)


# ============================================================ spine terms
def test_arc_progress_linear():
    r = RW.arc_progress_reward(_t([5.0, 3.0, 1.0]), _t([4.0, 3.0, 2.0]), 10.0)
    assert torch.allclose(r, _t([10.0, 0.0, -10.0]))    # advance / hold / backtrack


def test_gt_estimerr_anchor_is_negative_flat():
    r = RW.gt_estimerr_anchor(_t([0.0, 0.1, 0.5]), 2.0)
    assert torch.allclose(r, _t([0.0, -0.2, -1.0]))     # flat weight, no proximity schedule


def test_confidence_anneal_ramp():
    w = RW.Inc8RewardWeights(conf_anneal_warmup_steps=100.0, conf_anneal_ramp_steps=100.0)
    assert RW.confidence_anneal(50.0, w) == 0.0          # before warmup
    assert RW.confidence_anneal(150.0, w) == pytest.approx(0.5)
    assert RW.confidence_anneal(250.0, w) == 1.0         # saturated
    # default (warmup 0, ramp 1) -> ~instantly on
    assert RW.confidence_anneal(1.0, RW.Inc8RewardWeights()) == 1.0


def test_confidence_shaping():
    # triple = [c_inplane, c_along, age_norm]; reward = w*anneal*(c_inplane - age_norm)
    triple = _t([[1.0, 0.5, 0.0], [0.2, 0.2, 1.0]])
    r = RW.confidence_shaping_reward(triple, rw_conf_shape=0.1, anneal=1.0)
    assert torch.allclose(r, _t([0.1, -0.08]))
    # anneal=0 zeros it
    assert torch.allclose(RW.confidence_shaping_reward(triple, 0.1, 0.0), _t([0.0, 0.0]))


# ============================================================ R5' visibility + terminal weight
def test_visibility_peaks_centered_and_bounded():
    # centred gate (t_cam on +Z optical axis) -> v == 1; off-axis -> v < 1; all in [0,1]
    t_centered = _t([[0.0, 0.0, 20.0]])
    assert RW.visibility_2axis(t_centered, 45.0, 29.5).item() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    tc = _t(rng.uniform(-20, 20, (500, 3)))
    tc[:, 2] = tc[:, 2].abs() + 1.0      # in front of the camera
    v = RW.visibility_2axis(tc, 45.0, 29.5)
    assert torch.all(v >= 0.0) and torch.all(v <= 1.0)


def test_visibility_elevation_tighter_than_azimuth():
    """The narrower vertical FoV (sigma_b < sigma_a) penalises an elevation offset MORE than the same
    azimuth offset (the 2-axis novelty vs the 1-axis SWIFT term)."""
    Z = 20.0
    off = Z * np.tan(np.radians(25.0))   # a 25-deg offset on each axis
    v_az = RW.visibility_2axis(_t([[off, 0.0, Z]]), 45.0, 29.5).item()     # azimuth-only
    v_el = RW.visibility_2axis(_t([[0.0, off, Z]]), 45.0, 29.5).item()     # elevation-only
    assert v_el < v_az, (v_el, v_az)


def test_terminal_weight_bandpass():
    """w_term is a BAND-PASS: peak in [r_lo, r_hi], SUPPRESSED to ~w0 BOTH terminal (<r_lo) and far
    (>r_hi). The two-sided suppression -- especially terminal -- is the fix for convergence run 3273701,
    where a one-sided low-pass plateau rewarded easy <5 m pointing equally and yielded 0 fixes."""
    r_lo, r_hi, w_lo, w_hi, w0 = 12.0, 28.0, 1.5, 1.5, 0.05
    tw = lambda r: RW.terminal_weight(_t([r]), r_lo, r_hi, w_lo, w_hi, w0).item()
    assert tw(20.0) > 0.9                                 # peak: full weight in the fixable band
    assert tw(3.0) < 0.12 and tw(40.0) < 0.12            # terminal AND far strongly suppressed
    assert tw(3.0) == pytest.approx(w0, abs=0.05)        # terminal decays to the floor w0
    assert tw(3.0) * 5.0 < tw(20.0)                      # THE headline: w_term(3) << w_term(20)
    assert tw(12.0) == pytest.approx(0.525, abs=0.03)    # lower sigmoid edge ~ half-peak (+w0)
    assert tw(28.0) == pytest.approx(0.525, abs=0.03)    # upper sigmoid edge ~ half-peak (+w0)
    assert tw(8.0) < tw(12.0) < tw(16.0)                 # monotone rising through the lower edge


# ============================================================ R5' arms
def test_perception_arms():
    w = RW.Inc8RewardWeights()
    ds = _t([0.5]); in_img = torch.tensor([True])
    # IN-BAND, centred (range 20 m): arm A w_term ~ peak (~1) so A ~ B; both ~ rw_perc * v(=1) * ds
    t_band = _t([[0.0, 0.0, 20.0]]); rng_band = _t([20.0])
    rA = RW.perception_reward(t_band, rng_band, ds, in_img, "A", w).item()
    rB = RW.perception_reward(t_band, rng_band, ds, in_img, "B", w).item()
    rC = RW.perception_reward(t_band, rng_band, ds, in_img, "C", w).item()
    assert rB == pytest.approx(w.perc * 1.0 * 1.0 * 0.5)   # arm B flat: rw_perc * v(=1) * ds
    assert rA == pytest.approx(rB, rel=0.05)               # in-band, w_term ~ 1 so A ~ B
    assert rC == 0.0
    # TERMINAL (range 4 m): arm A is BAND-PASS-SUPPRESSED (w_term ~ w0) so A << B -- the conv-run fix
    t_term = _t([[0.0, 0.0, 4.0]]); rng_term = _t([4.0])
    rA_term = RW.perception_reward(t_term, rng_term, ds, in_img, "A", w).item()
    rB_term = RW.perception_reward(t_term, rng_term, ds, in_img, "B", w).item()
    assert rA_term < 0.2 * rB_term                         # terminal pointing NOT rewarded like the band
    # FAR (range 40 m): arm A also suppressed (upper band edge) so A << B; arm B stays flat
    t_far = _t([[0.0, 0.0, 40.0]]); rng_far = _t([40.0])
    rA_far = RW.perception_reward(t_far, rng_far, ds, in_img, "A", w).item()
    rB_far = RW.perception_reward(t_far, rng_far, ds, in_img, "B", w).item()
    assert rA_far < 0.2 * rB_far


def test_perception_progress_gated_and_in_image():
    w = RW.Inc8RewardWeights()
    t_cam = _t([[0.0, 0.0, 4.0]]); rng = _t([4.0])
    # backing up (ds<0) -> zero (anti-loiter)
    assert RW.perception_reward(t_cam, rng, _t([-0.3]), torch.tensor([True]), "A", w).item() == 0.0
    # out of image -> zero even if ds>0
    assert RW.perception_reward(t_cam, rng, _t([0.5]), torch.tensor([False]), "A", w).item() == 0.0


def test_fix_bonus_reward():
    """Direct per-ACCEPTED-fix bonus: progress-gated (anti-loiter), exactly zero when disabled, earnable
    ONLY by a real accepted fix WHILE advancing (delta_s > 0)."""
    acc = torch.tensor([True, True, False, True])
    ds = _t([0.5, -0.2, 0.5, 0.0])               # advancing / backing / advancing / holding
    # disabled (rw_fix_bonus=0) -> exactly the zero term (byte-identical inc8)
    assert torch.equal(RW.fix_bonus_reward(acc, ds, 0.0), torch.zeros_like(ds))
    # only env 0 pays: accepted AND advancing; backing/not-accepted/holding -> 0
    assert torch.allclose(RW.fix_bonus_reward(acc, ds, 1.5), _t([1.5, 0.0, 0.0, 0.0]))


# ============================================================ BSR3 spin-gate
def test_bsr3_disabled_is_noop():
    clock = torch.zeros(4, dtype=DT)
    omega = torch.full((4, 3), 100.0, dtype=DT)         # absurd rate
    c, abort = RW.bsr3_update(clock, omega, 0.0333, rate_abort=0.0, time_abort=3.0)
    assert torch.equal(c, clock) and not abort.any()


def test_bsr3_does_not_false_abort_super_rate_transient():
    """A legitimate ~11 rad/s super-rate transient (the measured plant amplifies the command) lasting
    < time_abort must NOT abort. Sustained > 3.0 s WOULD (a true tumble)."""
    dt, rate_abort, time_abort = 0.0333, 10.0, 3.0
    clock = torch.zeros(1, dtype=DT)
    # a ~0.5 s burst at 11 rad/s (15 ticks): over the threshold but a brief transient
    aborted = False
    for _ in range(15):
        clock, abort = RW.bsr3_update(clock, _t([[0.0, 0.0, 11.0]]), dt, rate_abort, time_abort)
        aborted |= bool(abort.any())
    assert not aborted, "BSR3 false-aborted a legitimate ~11 rad/s transient"
    # the clock then resets when the rate drops back under threshold
    clock, abort = RW.bsr3_update(clock, _t([[0.0, 0.0, 2.0]]), dt, rate_abort, time_abort)
    assert clock.item() == 0.0 and not abort.any()


def test_bsr3_aborts_sustained_spin():
    dt, rate_abort, time_abort = 0.0333, 10.0, 3.0
    clock = torch.zeros(1, dtype=DT)
    aborted = False
    steps = int(3.5 / dt)                                # 3.5 s sustained > 3.0 s
    for _ in range(steps):
        clock, abort = RW.bsr3_update(clock, _t([[0.0, 0.0, 13.0]]), dt, rate_abort, time_abort)
        aborted |= bool(abort.any())
    assert aborted, "BSR3 failed to abort a sustained > 3 s spin"
