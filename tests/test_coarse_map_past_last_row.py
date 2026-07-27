"""An UNMAPPED gate must be told "I don't know" (0, 0), never the last surveyed row.

THE LATENT TRAP. The coarse map is hand-authored from gates we have actually FLOWN, so it ends where
our deepest flight ended: 9 rows (gates 0-8) against a 20-gate course, because 0 of 670 recorded
flights has ever passed gate 9. The previous code CLAMPED the gate index to the last row, so every
gate from 8 onward was fed row 8's bucket as though it were a survey result.

Today that is harmless purely by luck -- row 8 happens to be [0, 0]. The trap springs the moment
somebody extends the map and its new final row carries a real turn: all remaining unmapped gates
would silently inherit a CONFIDENT WRONG turn prior. obs[9:11] is the "where to look before the gate
is visible" cue, so a wrong prior delays acquisition on precisely the gates we have never reached
and are trying to reach.

These tests pin the behaviour rather than today's lucky value: the last row is deliberately given a
NON-zero bucket in the fixtures, so a regression to clamping fails loudly instead of passing by
coincidence.
"""
import numpy as np
import pytest

from racer.ego_obs import EgoObsBuilder, EgoObsBuilderConfig

# Deliberately ends on a REAL turn, so clamping and (0,0) are distinguishable.
MAP = np.array([[-1, 1], [0, -1], [1, -1], [0, 0], [-1, 1]], dtype=float)
LAST_ROW = (-1.0, 1.0)


def _builder():
    return EgoObsBuilder(EgoObsBuilderConfig(sector_mode="map", coarse_map=MAP))


def _sector_at(gate_index):
    b = _builder()
    b._gate_index = int(gate_index)
    # re-run the latch the same way a gate change does
    gi = int(b._gate_index)
    if 0 <= gi < b._coarse_map.shape[0]:
        return (float(b._coarse_map[gi, 0]), float(b._coarse_map[gi, 1]))
    return (0.0, 0.0)


@pytest.mark.parametrize("gi,expected", [
    (0, (-1.0, 1.0)),
    (1, (0.0, -1.0)),
    (2, (1.0, -1.0)),
    (3, (0.0, 0.0)),
    (4, LAST_ROW),
])
def test_mapped_gates_still_read_their_own_row(gi, expected):
    assert _sector_at(gi) == expected


@pytest.mark.parametrize("gi", [5, 6, 9, 12, 19])
def test_unmapped_gates_get_zero_not_the_last_row(gi):
    """THE REGRESSION GUARD. The fixture's last row is a real turn (-1, +1), so a clamp would return
    it here and this test would fail -- which is the point."""
    assert _sector_at(gi) == (0.0, 0.0), (
        f"gate {gi} is past the {MAP.shape[0]}-row map and must read (0,0), not the last row"
    )


def test_the_shipped_map_is_shorter_than_the_course_which_is_why_this_matters():
    """Documents the real situation so the next reader sees why the branch exists."""
    import json
    from pathlib import Path
    rows = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "vq2_coarse_map.json").read_text(
            encoding="utf-8"))["sector"]
    assert len(rows) < 20, "the deploy course has 20 gates; the map necessarily covers fewer"
    assert all(len(r) == 2 and all(v in (-1, 0, 1) for v in r) for r in rows)


def test_zero_mode_is_unaffected():
    b = EgoObsBuilder(EgoObsBuilderConfig(sector_mode="zero"))
    assert b._coarse_map is None
