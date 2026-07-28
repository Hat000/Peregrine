"""Tests for context-window tracking and threshold warnings."""

from nocturne.config import ContextConfig
from nocturne.context_tracker import ContextTracker


def _tracker(window=1000):
    return ContextTracker(ContextConfig(
        window_tokens=window, warn_thresholds=[0.75, 0.90],
        warn_before_compaction_at=0.92,
    ))


def test_no_warning_below_threshold():
    t = _tracker()
    assert t.update({"input_tokens": 500, "output_tokens": 100}) == []
    assert 0.55 < t.fraction < 0.65


def test_threshold_fires_once():
    t = _tracker()
    w1 = t.update({"input_tokens": 760, "output_tokens": 0})
    assert len(w1) == 1 and w1[0].kind == "threshold"
    # same level again -> no repeat
    assert t.update({"input_tokens": 760, "output_tokens": 0}) == []


def test_compaction_warning():
    t = _tracker()
    warns = t.update({"input_tokens": 950, "output_tokens": 0})
    kinds = {w.kind for w in warns}
    assert "compaction" in kinds
    assert "threshold" in kinds  # crosses 75 and 90 too


def test_cache_tokens_count_toward_context():
    t = _tracker()
    t.update({"input_tokens": 100, "cache_read_input_tokens": 700,
              "cache_creation_input_tokens": 100, "output_tokens": 0})
    assert t.current == 900
    assert t.fraction == 0.9


def test_report_and_remaining():
    t = _tracker(window=1000)
    t.update({"input_tokens": 250, "output_tokens": 0})
    assert t.remaining_tokens == 750
    assert "percent" in t.report().lower()


def test_reset_clears_fired():
    t = _tracker()
    t.update({"input_tokens": 800, "output_tokens": 0})
    t.reset()
    assert t.current == 0
    # after reset the threshold can fire again
    assert len(t.update({"input_tokens": 800, "output_tokens": 0})) >= 1
