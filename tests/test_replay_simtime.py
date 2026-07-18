"""REPLAY-RATCHET P0.1 -- sim-time-indexed tape playback (fly_rl.replay_row_for_sim_time).

Pins the PURE row selector that fixes the tape player's cumulative command lag. Background
(handoff/ratchet-p0-2026-07-17/REPORT.md): the eval sim + tape player are run-to-run
DETERMINISTIC, but the champion tape re-banks 0/5 gates because the player consumed ONE ROW PER
WALL TICK at an under-run ~25.88 ms effective tick (Windows sleep granularity) vs the tape's
25 ms grid -> ~72 ms command lag by gate 0 -> a gate-0 FRAME strike where the champion threaded
the aperture. --sysid-replay-simtime selects the row DUE at the current sim-time instead, so the
command stream tracks sim physics regardless of loop jitter.

These tests need no sim: the selector is integer-exact on (sim_time_ns, anchor_ns, tick_ns,
n_rows). Importing fly_rl pulls in torch (the deploy stack); the module skips if it is absent,
matching tests/test_tape_extract.py's golden cross-check.
"""
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_RL = _REPO / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

try:
    import fly_rl  # noqa: E402  (imports torch)
    _IMPORT_ERR = ""
except Exception as exc:                                  # pragma: no cover
    fly_rl = None
    _IMPORT_ERR = f"{type(exc).__name__}: {exc}"

pytestmark = pytest.mark.skipif(
    fly_rl is None, reason=f"fly_rl not importable here ({_IMPORT_ERR}); "
                           f"runs where the deploy stack runs")

_TICK40 = 25_000_000        # ns per row on a 40 Hz tape grid == round(1e9/40)


# --- exact grid ------------------------------------------------------------------------------

def test_exact_25ms_grid_one_row_per_tick():
    """On the exact 25 ms grid, row k owns [anchor + k*tick, anchor + (k+1)*tick): the boundary,
    the mid-window, and 1 ns before the next boundary all select k."""
    anchor = 1_000_000_000            # a non-zero anchor (first program tick is rarely sim t=0)
    n_rows = 433
    for k in range(n_rows):
        base = anchor + k * _TICK40
        assert fly_rl.replay_row_for_sim_time(base, anchor, _TICK40, n_rows) == k
        assert fly_rl.replay_row_for_sim_time(base + _TICK40 // 2, anchor, _TICK40, n_rows) == k
        assert fly_rl.replay_row_for_sim_time(base + _TICK40 - 1, anchor, _TICK40, n_rows) == k


# --- the fix: sim-time tracking vs the wall-tick drift ---------------------------------------

def test_stretched_wakes_simtime_tracks_walltick_lags():
    """The REPORT's failure mode, on a synthesized under-run schedule: wake i lands at sim-time
    anchor + i*25.88 ms (the measured 38.6 Hz effective tick) while the tape grid is 25 ms.

      * WALL-TICK mode consumes exactly one row per wake -> its implied row == i, which falls
        progressively BEHIND the tape position the sim has actually reached (~3 rows by wake 111,
        the champion's gate-0 pass region -> the frame strike).
      * SIM-TIME mode selects floor((sim_time - anchor)/25 ms) -> it stays within 1 row of the
        real-valued tape position at every wake, i.e. it does NOT drift.
    """
    stretched_ns = 25_880_000         # 25.88 ms measured effective wake interval (REPORT Step 2)
    anchor = 0
    n_rows = 433

    lag_at = {}
    for i in range(160):
        st = anchor + i * stretched_ns
        simtime_row = fly_rl.replay_row_for_sim_time(st, anchor, _TICK40, n_rows)
        if simtime_row is None:       # ran off the end of the tape's sim-time span
            break
        ideal = st / _TICK40          # real-valued tape position the sim has reached
        # SIM-TIME mode never drifts: it is floor(ideal), so within [0,1) below the ideal.
        assert 0.0 <= ideal - simtime_row < 1.0
        assert abs(simtime_row - round(ideal)) <= 1
        lag_at[i] = ideal - i         # how far the WALL-TICK row (== i) trails the sim

    # wall-tick drift grows and is ~3 rows behind by the gate-0 region (~row 111).
    assert lag_at[40] < lag_at[111]                      # the lag accumulates
    assert 2.5 <= lag_at[111] <= 4.5                     # "~3 rows behind sim by row ~111"
    # integer row actually skipped past the wall-tick index by then:
    assert (int(111 * stretched_ns / _TICK40) - 111) == 3


# --- clamp below zero ------------------------------------------------------------------------

def test_clamp_below_zero_holds_row0():
    """Pre-anchor sim-clock jitter (st < anchor) holds row 0 rather than going negative."""
    anchor = 2_000_000_000
    assert fly_rl.replay_row_for_sim_time(anchor, anchor, _TICK40, 433) == 0
    assert fly_rl.replay_row_for_sim_time(anchor - 1, anchor, _TICK40, 433) == 0
    assert fly_rl.replay_row_for_sim_time(anchor - 50 * _TICK40, anchor, _TICK40, 433) == 0


# --- end of tape: last row held for its FULL window ------------------------------------------

def test_end_of_tape_only_after_last_full_window():
    """Row n-1 owns its full tick window; None comes only once that window has elapsed."""
    n = 5
    anchor = 7_000_000_000
    last = n - 1                                          # 4
    assert fly_rl.replay_row_for_sim_time(anchor + last * _TICK40, anchor, _TICK40, n) == last
    # anywhere inside the last row's 25 ms window still returns the last row (held, not ended):
    assert fly_rl.replay_row_for_sim_time(anchor + n * _TICK40 - 1, anchor, _TICK40, n) == last
    # exactly at / past the end of that window -> program over:
    assert fly_rl.replay_row_for_sim_time(anchor + n * _TICK40, anchor, _TICK40, n) is None
    assert fly_rl.replay_row_for_sim_time(anchor + 3 * n * _TICK40, anchor, _TICK40, n) is None


# --- tick_ns rounding + degenerate guards ----------------------------------------------------

def test_tick_ns_rounding_rate40():
    """--rate 40 -> tick_ns = round(1e9/40) = 25_000_000 exactly (the map _fly_ego applies), and a
    25 ms sim advance is exactly one row on that grid. A non-dividing rate rounds to the nearest
    ns (rate 30 -> 33_333_333)."""
    assert int(round(1e9 / 40)) == 25_000_000
    assert int(round(1e9 / 30)) == 33_333_333
    tick40 = int(round(1e9 / 40))
    assert fly_rl.replay_row_for_sim_time(25_000_000, 0, tick40, 100) == 1
    assert fly_rl.replay_row_for_sim_time(25_000_000 - 1, 0, tick40, 100) == 0


def test_degenerate_inputs_return_none():
    """Empty tape / non-positive tick -> program over (never index into nothing)."""
    assert fly_rl.replay_row_for_sim_time(0, 0, _TICK40, 0) is None
    assert fly_rl.replay_row_for_sim_time(0, 0, 0, 100) is None
