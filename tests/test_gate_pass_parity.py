"""fly_rl.gate_pass_event must be BYTE-IDENTICAL to the training gate-advance test.

Deploy-parity fix 2026-07-05: fly_rl's RL loop re-anchors gate_index at the PHYSICAL plane
crossing — the geometry the policy trained on (peregrine_racing.crossing_events) — instead of
RACE_STATUS.active_gate_index, which fires ~9 m up-course of the plane. This suite pins
fly_rl's local copy to rl/offline_rollout.gate_event (the audited numpy mirror of the torch
env classification) on its legacy point-mass path (body_radius=0, frame_depth=0):

  * randomized segment fuzz over every gate — crossings, straddles, backward, wide misses,
    near-plane degeneracies — asserting identical classification;
  * directed boundary cases: L-inf at the 0.75 / 1.36 band edges, backward-through-aperture
    (non-event), backward-through-frame (collision), the 9 m-early wire-fire point (must be
    a NON-event), hover short of the plane.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fly_rl import N_GATES, _FLIP, _GATE_POS_ZUP, _R_W2G, gate_pass_event  # noqa: E402
from offline_rollout import gate_event  # noqa: E402


def _ned(gate: int, rel) -> np.ndarray:
    """Gate-frame point -> NED (inverse of the w2g used by both implementations)."""
    rel = np.asarray(rel, dtype=np.float64)
    return (_R_W2G.T @ rel + _GATE_POS_ZUP[gate]) * _FLIP


def test_fuzz_parity_all_gates():
    rng = np.random.default_rng(0)
    n_events = 0
    for g in range(N_GATES):
        for _ in range(4000):
            spread = float(rng.choice([0.3, 0.8, 1.3, 3.0]))   # in-plane offsets
            along = float(rng.choice([0.05, 0.5, 2.0]))        # through-axis offsets
            p = _ned(g, [rng.uniform(-along, along),
                         rng.uniform(-spread, spread), rng.uniform(-spread, spread)])
            c = _ned(g, [rng.uniform(-along, along),
                         rng.uniform(-spread, spread), rng.uniform(-spread, spread)])
            a = gate_pass_event(p, c, g)
            b = gate_event(p, c, g)
            assert a == b, f"gate {g}: fly_rl={a!r} != offline_rollout={b!r} (prev={p}, cur={c})"
            n_events += a is not None
    assert n_events > 500   # the fuzz genuinely exercises events, not just None == None


def test_directed_boundary_cases():
    g = 2
    # clean centred forward pass
    assert gate_pass_event(_ned(g, [-0.3, 0.10, -0.05]), _ned(g, [0.2, 0.12, -0.02]), g) == "pass"
    # L-inf at the crossing point just inside / outside the 0.75 opening
    assert gate_pass_event(_ned(g, [-0.1, 0.749, 0.0]), _ned(g, [0.1, 0.749, 0.0]), g) == "pass"
    assert gate_pass_event(_ned(g, [-0.1, 0.751, 0.0]), _ned(g, [0.1, 0.751, 0.0]), g) == "collision"
    # beyond the 1.36 outer band: clean miss
    assert gate_pass_event(_ned(g, [-0.1, 2.0, 0.0]), _ned(g, [0.1, 2.0, 0.0]), g) == "miss"
    # backward through the OPEN aperture: non-event (matches the env)
    assert gate_pass_event(_ned(g, [0.1, 0.0, 0.0]), _ned(g, [-0.1, 0.0, 0.0]), g) is None
    # backward through the frame band: collision
    assert gate_pass_event(_ned(g, [0.1, 1.0, 0.0]), _ned(g, [-0.1, 1.0, 0.0]), g) == "collision"
    # the ~9 m-early wire-fire point: approaching, no plane crossing -> MUST be a non-event
    assert gate_pass_event(_ned(g, [-9.5, 0.0, 0.0]), _ned(g, [-8.8, 0.0, 0.0]), g) is None
    # hovering just short of the plane: non-event
    assert gate_pass_event(_ned(g, [-0.2, 0.0, 0.0]), _ned(g, [-0.05, 0.0, 0.0]), g) is None
    # vertical (z) excursion dominates L-inf at the crossing
    assert gate_pass_event(_ned(g, [-0.1, 0.0, 0.9]), _ned(g, [0.1, 0.0, 0.9]), g) == "collision"
    # parity on every directed segment too
    for p, c in [([-0.3, 0.10, -0.05], [0.2, 0.12, -0.02]), ([-0.1, 0.751, 0.0], [0.1, 0.751, 0.0]),
                 ([0.1, 1.0, 0.0], [-0.1, 1.0, 0.0]), ([-9.5, 0.0, 0.0], [-8.8, 0.0, 0.0])]:
        assert gate_pass_event(_ned(g, p), _ned(g, c), g) == gate_event(_ned(g, p), _ned(g, c), g)


def test_interpolation_is_at_the_crossing_point():
    """A fast diagonal step whose ENDPOINT is far off-axis but whose interpolated crossing
    point is inside the opening must PASS (the training geometry's whole point)."""
    g = 0
    p = _ned(g, [-0.05, 0.70, 0.0])
    c = _ned(g, [0.95, 1.90, 0.0])    # endpoint L-inf 1.9 (outside everything)
    # crossing at f = 0.05 -> y = 0.70 + 0.05*1.2 = 0.76 -> collision band, NOT a wide miss
    assert gate_pass_event(p, c, g) == "collision" == gate_event(p, c, g)
    p2 = _ned(g, [-0.05, 0.60, 0.0])
    c2 = _ned(g, [0.95, 1.80, 0.0])   # crossing y = 0.66 < 0.75 -> pass despite the endpoint
    assert gate_pass_event(p2, c2, g) == "pass" == gate_event(p2, c2, g)
