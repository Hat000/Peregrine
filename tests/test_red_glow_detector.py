"""Validation of the VQ2 red-glow gate detector against REAL recon camera frames.

Fixtures (``tests/fixtures/vq2_recon/``) are a small handful of the 2026-06-29 VQ2 load-day
recon frames (640x360, glowing-red gates in a dark warehouse):
  - 01_gate_near, 02_gate_near : gate dead-ahead, near        -> MUST detect + PnP cleanly
  - 08_no_gate                 : between gates, facing away   -> MUST reject (no detection)
  - 09_blowout                 : extreme-near red bloom        -> graceful (no fabricated pose)

These pin the detector's behaviour to the TRUE VQ2 appearance, and demonstrate the
saturated-core-vs-naive-low-threshold span inflation that motivates the core threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from racer.contracts import Frame, GateObservation
from racer.vision.gate_pose import estimate_gate_pose
from racer.vision.red_glow_detector import (
    RedGlowGateDetector,
    RedGlowParams,
    detect_red_glow_candidates,
)

FIX = Path(__file__).parent / "fixtures" / "vq2_recon"


def _load(name: str) -> np.ndarray:
    img = cv2.imread(str(FIX / name))
    assert img is not None, f"missing fixture {name}"
    assert img.shape == (360, 640, 3), f"{name} unexpected shape {img.shape}"
    return img


def _frame(name: str) -> Frame:
    return Frame(frame_id=1, sim_time_ns=42, image_bgr=_load(name))


# --------------------------------------------------------------------------- #
# DETECTION on near dead-ahead gates (01, 02)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["01_gate_near.png", "02_gate_near.png"])
def test_detects_near_gate(name):
    cands = detect_red_glow_candidates(_load(name))
    assert cands, f"{name}: expected >=1 gate candidate"
    primary = cands[0]
    # Primary opening is reasonably square and a plausible apparent size for a near gate.
    span = float(np.linalg.norm(primary.corners_px.max(0) - primary.corners_px.min(0)))
    assert 30.0 <= span <= 200.0, f"{name}: implausible primary span {span:.0f}px"
    # The near dead-ahead gate sits near the horizontal image centre (cx=320).
    assert abs(primary.centroid[0] - 320.0) < 120.0, f"{name}: primary not centred"


@pytest.mark.parametrize("name", ["01_gate_near.png", "02_gate_near.png"])
def test_near_gate_pnp_plausible_range(name):
    """The full contract: detector corners -> GateObservation -> PnP recovers a sane range."""
    det = RedGlowGateDetector()
    obs = det.detect(_frame(name))
    assert obs, f"{name}: detector produced no observations"
    o = obs[0]
    assert o.corners_px.shape == (4, 2)
    assert o.corner_ids is None  # full 4 corners -> canonical [0,1,2,3]
    assert o.corner_confidence is not None and o.corner_confidence.shape == (4,)
    gp = estimate_gate_pose(o)
    # Near gate: a few metres to ~40 m, low reprojection error (corners are self-consistent).
    assert 2.0 <= gp.range_m <= 45.0, f"{name}: implausible PnP range {gp.range_m:.1f}m"
    assert gp.reproj_error_px < 5.0, f"{name}: high reproj error {gp.reproj_error_px:.1f}px"


# --------------------------------------------------------------------------- #
# REJECTION on no-gate frames (08) — clutter must NOT trip a detection
# --------------------------------------------------------------------------- #
def test_rejects_no_gate_frame():
    cands = detect_red_glow_candidates(_load("08_no_gate.png"))
    assert cands == [], "no-gate frame must yield zero gate candidates"


# --------------------------------------------------------------------------- #
# GRACEFUL on extreme-near blow-out (09) — never fabricate a 4-corner pose
# --------------------------------------------------------------------------- #
def test_blowout_does_not_fabricate_pose():
    """At extreme near range the bar saturates/blooms with no resolvable opening. The detector
    must not emit a bogus opening (downstream dead-reckons on IMU instead)."""
    det = RedGlowGateDetector()
    obs = det.detect(_frame("09_blowout.png"))
    # Either no detection, or any detection must still be a structurally valid observation
    # (the GateObservation __post_init__ asserts 3..4 corners); we forbid a near-full-frame blob.
    for o in obs:
        span = float(np.linalg.norm(o.corners_px.max(0) - o.corners_px.min(0)))
        assert span < 400.0, "blow-out produced an implausibly large opening"


# --------------------------------------------------------------------------- #
# Synthetic clutter rejection — color/shape gates reject non-red and streaks
# --------------------------------------------------------------------------- #
def test_rejects_blue_and_orange_clutter():
    img = np.zeros((360, 640, 3), np.uint8)
    # Blue chevron block (B high) — must be rejected by red-dominance.
    img[100:160, 100:200] = (255, 30, 20)   # BGR: strong blue
    # Orange floor beam streak (R+G high, elongated) — rejected by red-dominance + aspect.
    img[300:312, 50:600] = (0, 160, 255)    # BGR: orange, R-G small, very elongated
    cands = detect_red_glow_candidates(img)
    assert cands == [], "blue/orange clutter must not be detected as a gate"


def test_detects_synthetic_red_ring():
    """A synthetic saturated-red square ring with a dark hole -> exactly one centred candidate."""
    img = np.zeros((360, 640, 3), np.uint8)
    # outer red square 280..360 x, 140..220 y (80x80); inner dark hole 296..344 x, 156..204 y
    img[140:220, 280:360] = (40, 40, 255)   # saturated red, red-dominant
    img[156:204, 296:344] = (0, 0, 0)        # dark opening
    cands = detect_red_glow_candidates(img)
    assert len(cands) == 1
    c = cands[0]
    # opening centroid ~ ring centre (320, 180)
    assert abs(c.centroid[0] - 320) < 8 and abs(c.centroid[1] - 180) < 8


# --------------------------------------------------------------------------- #
# CORE-vs-NAIVE span inflation — the binding recon finding, quantified
# --------------------------------------------------------------------------- #
def _naive_low_threshold_span(img: np.ndarray) -> float | None:
    """Apparent gate span using a LOW red threshold (catches bloom halo + ambient wash)."""
    b, g, r = (img[:, :, i].astype(int) for i in range(3))
    m = ((r > 120) & (r - g > 40) & (r - b > 40)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return float(max(w, h))


def test_core_threshold_avoids_near_bias():
    """The saturated CORE opening span must be SMALLER than the naive low-threshold span on a
    near gate — a low threshold inflates the apparent square (halo+wash) and biases range NEAR."""
    img = _load("01_gate_near.png")
    cands = detect_red_glow_candidates(img)
    assert cands
    core_span = float(np.linalg.norm(cands[0].corners_px.max(0) - cands[0].corners_px.min(0)))
    naive_span = _naive_low_threshold_span(img)
    assert naive_span is not None
    # The naive blob over-segments the gate region (halo+wash) -> larger apparent extent.
    assert naive_span > core_span, (
        f"core span {core_span:.0f}px should be < naive span {naive_span:.0f}px"
    )
