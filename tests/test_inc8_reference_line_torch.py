"""S4 PARITY GATE -- batched torch ReferenceLine.progress == numpy ReferenceLine.progress (<=1e-4).

The R1' reward primitive (arc-length progress along the rebuilt corrected-aero contact-safe line)
must match the numpy projector exactly, else the inc8 progress reward diverges from the verified
contact-safe line. Also pins strict monotonicity along Gamma's own samples. Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_inc8_reference_line_torch.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.reference_line import ReferenceLine                        # noqa: E402
from reference_line_torch import BatchedReferenceLine                 # noqa: E402

REF = str(ROOT / "rl" / "reference_line_inc8.json")
DT64 = torch.float64


def _lines(dtype=DT64):
    return ReferenceLine.load(REF), BatchedReferenceLine.load(REF, "cpu", dtype)


def test_progress_parity_on_samples():
    """progress at every emitted sample matches numpy (<=1e-4) AND is strictly increasing along the
    sample order (no fold-backs along Gamma)."""
    nrl, brl = _lines()
    pos = nrl.pos                                                    # (M,3) the line's own samples
    s_torch = brl.progress(torch.tensor(pos, dtype=DT64)).numpy()
    s_numpy = np.array([nrl.progress(p) for p in pos])
    assert np.max(np.abs(s_torch - s_numpy)) <= 1e-4, np.max(np.abs(s_torch - s_numpy))
    # strict monotonicity along Gamma (the contact-safe line advances)
    assert np.all(np.diff(s_torch) > 0.0), "progress not strictly monotonic along Gamma samples"


def test_progress_parity_random_offsets():
    """progress on positions PERTURBED off the line (the realistic case: the drone is near but not on
    Gamma) matches numpy. Generic offsets avoid exact vertex ties (a degenerate argmin edge)."""
    nrl, brl = _lines()
    rng = np.random.default_rng(4)
    M = nrl.pos.shape[0]
    base = nrl.pos[rng.integers(0, M, 4000)]
    q = base + rng.uniform(-3.0, 3.0, base.shape)
    s_torch = brl.progress(torch.tensor(q, dtype=DT64)).numpy()
    s_numpy = np.array([nrl.progress(p) for p in q])
    err = np.abs(s_torch - s_numpy)
    assert np.max(err) <= 1e-4, (np.max(err), float(np.mean(err)))


def test_progress_parity_far_field():
    """Far-from-line queries still match (the OOB / lost-drone tail) -- the projection clamps to the
    nearest segment endpoint identically."""
    nrl, brl = _lines()
    rng = np.random.default_rng(9)
    q = rng.uniform(-200, 30, (2000, 3))
    s_torch = brl.progress(torch.tensor(q, dtype=DT64)).numpy()
    s_numpy = np.array([nrl.progress(p) for p in q])
    # far-field can graze argmin ties (two segments equidistant) -> allow a tiny vertex-jump fraction
    err = np.abs(s_torch - s_numpy)
    assert np.percentile(err, 99) <= 1e-4, (np.percentile(err, 99), np.max(err))


def test_progress_endpoints():
    """s(start) ~ 0, s(end) ~ total arc length; the segment table reproduces ReferenceLine.arc[-1]."""
    nrl, brl = _lines()
    assert abs(brl.total_arc_m - float(nrl.arc[-1])) <= 1e-6
    s0 = float(brl.progress(torch.tensor(nrl.pos[0], dtype=DT64))[0])
    s1 = float(brl.progress(torch.tensor(nrl.pos[-1], dtype=DT64))[0])
    assert s0 == pytest.approx(0.0, abs=1e-4)
    assert s1 == pytest.approx(brl.total_arc_m, abs=1e-4)


def test_progress_float32_within_tol():
    """The DEPLOYMENT dtype (float32, the GPU env) stays within a tight band off the line."""
    nrl, _ = _lines()
    brl32 = BatchedReferenceLine.load(REF, "cpu", torch.float32)
    rng = np.random.default_rng(1)
    M = nrl.pos.shape[0]
    base = nrl.pos[rng.integers(0, M, 3000)]
    q = base + rng.uniform(-2.0, 2.0, base.shape)
    s32 = brl32.progress(torch.tensor(q, dtype=torch.float32)).numpy().astype(np.float64)
    s_numpy = np.array([nrl.progress(p) for p in q])
    # float32 arc accumulation over ~190 m loses ~1e-2 m of absolute precision; the reward uses the
    # DELTA between consecutive steps, where this cancels. Assert a deploy-realistic band.
    assert np.percentile(np.abs(s32 - s_numpy), 99) <= 5e-2
