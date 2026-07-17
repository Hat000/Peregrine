"""REPLAY-RATCHET tape extractor (rl/tape_extract.py) -- format + fidelity pins.

What is load-bearing here:

  * The tape must reproduce the RECORDED WIRE: tape_extract inverts fly_rl's
    sysid_wire_from_action (fly_rl.py:745-759) so that the --sysid-replay player re-emits the
    logged rate_frd/collective exactly (modulo the log's own 4/5-dp rounding).  The inversion
    is pinned two ways: (1) an algebraic round-trip through tape_extract's own forward map,
    (2) a GOLDEN cross-check against fly_rl.sysid_wire_from_action itself (skipped if torch
    is unavailable) -- if fly_rl's map ever drifts, (2) fails and the tape format is dead.

  * The player steps ONE ROW PER TICK at --rate and IGNORES the t column (fly_rl.py:2061),
    while recording loops under-run their target rate (champion 202317: 26.4 Hz effective vs
    40).  The ZOH resample is therefore the ONLY faithful primary output; its semantics
    (row i = the command ACTIVE at t_i = i/rate in the recording) are pinned on synthetic data.

  * The REAL champion log (data/runs/20260714_202317_panel_run_f1, commit 28404fa: 5 gates
    banked, 1 collision) is parsed end-to-end: schema, gate-pass indices, thrust-channel
    representability, emitted-CSV compatibility with the player's exact csv parse.
"""
import csv
import json
import math
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_RL = _REPO / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import tape_extract as tx  # noqa: E402

_CHAMPION = _REPO / "data" / "runs" / "20260714_202317_panel_run_f1"

needs_champion = pytest.mark.skipif(
    not (_CHAMPION / "ego_obs.jsonl").is_file(),
    reason="champion run 20260714_202317_panel_run_f1 not checked out (28404fa logs)")


# --- inversion / forward map ------------------------------------------------------------------

def test_roundtrip_exact_both_flips():
    cases = [
        [0.3, -0.9, 0.5, 0.0],
        [-0.6, 0.0, 0.0, 0.0],          # hover
        [1.0, 1.0, -1.0, 1.0],          # rails
        [-1.0, -1.0, 1.0, -1.0],
    ]
    for flip in (False, True):
        for a in cases:
            rf, coll, normed = tx.forward_wire(a, flip)
            a2 = tx.invert_wire(rf, normed, flip)
            assert max(abs(x - y) for x, y in zip(a, a2)) < 1e-12
            rf2, coll2, _ = tx.forward_wire(a2, flip)
            assert max(abs(x - y) for x, y in zip(rf, rf2)) < 1e-12
            assert abs(coll - coll2) < 1e-12


def test_flip_conventions_differ_only_in_roll_pitch_sign():
    a = [0.1, 0.4, -0.3, 0.2]
    rf_off, _, _ = tx.forward_wire(a, False)
    rf_on, _, _ = tx.forward_wire(a, True)
    assert rf_on[0] == -rf_off[0]
    assert rf_on[1] == -rf_off[1]
    assert rf_on[2] == rf_off[2]


def test_invert_rejects_out_of_range():
    with pytest.raises(ValueError):
        tx.invert_wire([3.5, 0.0, 0.0], 1.0, False)      # |rate| > 3.14
    with pytest.raises(ValueError):
        tx.invert_wire([0.0, 0.0, 0.0], 5.2, False)      # normed > 5 g


def test_golden_cross_check_against_fly_rl():
    """The tape format is DEFINED by fly_rl.sysid_wire_from_action; pin byte-equality."""
    try:
        import fly_rl  # noqa: F401  (imports torch)
    except Exception as exc:                              # pragma: no cover
        pytest.skip(f"fly_rl not importable here ({type(exc).__name__}); "
                    f"golden check runs where the deploy stack runs")
    import numpy as np
    rng = np.random.default_rng(7)
    for flip in (False, True):
        for _ in range(50):
            a = rng.uniform(-1.0, 1.0, size=4)
            rf_g, coll_g, normed_g = fly_rl.sysid_wire_from_action(a, flip)
            rf_m, coll_m, normed_m = tx.forward_wire(list(a), flip)
            assert np.allclose(np.asarray(rf_g, dtype=float), rf_m, atol=1e-12)
            assert abs(coll_g - coll_m) < 1e-12
            assert abs(normed_g - normed_m) < 1e-12


# --- ZOH resample + truncation (synthetic) ----------------------------------------------------

def _mk_rec(k, t_ns, gate, rate_frd, normed, assist=False):
    return {"k": k, "sim_time_ns": t_ns, "gate_index": gate, "rate_frd": list(rate_frd),
            "normed_thrust": normed,
            "collective": min(max(normed * tx._HOVER_THRUST, 0.0), 1.0),
            "assist": assist, "kf_pos_ned": [0.0, 0.0, 0.0]}


def test_resample_zoh_semantics():
    # source ticks at 0, 30, 90 ms; 40 Hz grid = 25 ms -> n_out = floor(90/25)+1 = 4 rows at
    # 0/25/50/75 ms, each holding the command ACTIVE at that recording time: at 25 ms r1
    # (sent at 30 ms) is not yet active -> r0; at 50/75 ms -> r1.  r2's hold starts at 90 ms,
    # past the last grid tick (sub-tick tail, <25 ms of plant exposure in the recording).
    recs = [_mk_rec(0, 0, 0, [0.1, 0, 0], 1.0),
            _mk_rec(1, 30_000_000, 0, [0.2, 0, 0], 1.0),
            _mk_rec(2, 90_000_000, 0, [0.3, 0, 0], 1.0)]
    rows, rate_err, coll_err = tx.build_source_rows(recs, False)
    assert rate_err < 1e-9 and coll_err < 1e-9
    tape = tx.resample_zoh(rows, 40.0)
    assert [round(r["t"], 3) for r in tape] == [0.0, 0.025, 0.05, 0.075]
    assert [r["src_k"] for r in tape] == [0, 0, 1, 1]


def test_resample_holds_across_gap():
    recs = [_mk_rec(0, 0, 0, [0.1, 0, 0], 1.0),
            _mk_rec(1, 25_000_000, 0, [0.2, 0, 0], 1.0),
            _mk_rec(2, 130_000_000, 0, [0.4, 0, 0], 1.0)]   # 105 ms stall (champion max 104)
    rows, _, _ = tx.build_source_rows(recs, False)
    tape = tx.resample_zoh(rows, 40.0)
    # 0..130 ms at 25 ms -> 6 rows; ticks at 50/75/100/125 ms all hold r1 through the stall.
    assert [r["src_k"] for r in tape] == [0, 1, 1, 1, 1, 1]
    assert round(tape[-1]["t"], 3) == 0.125


def test_truncate_at_gate():
    recs = [_mk_rec(i, i * 25_000_000, g, [0.0, 0, 0], 1.0)
            for i, g in enumerate([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])]
    analysis = tx.analyze(recs)
    kept, cut = tx.truncate_at_gate(recs, 0, 2, analysis)
    assert cut == 3                       # first record with gate_index > 0
    assert len(kept) == 6                 # rows 0..3 plus 2 margin ticks
    assert [r["gate_index"] for r in kept] == [0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        tx.truncate_at_gate(recs, 5, 2, analysis)   # gate 5 never passed


def test_analyze_flags_nan_and_nonmonotonic():
    recs = [_mk_rec(0, 0, 0, [0.1, 0, 0], 1.0),
            _mk_rec(1, 25_000_000, 0, [float("nan"), 0, 0], 1.0)]
    assert any("NaN" in e for e in tx.analyze(recs)["errors"])
    recs = [_mk_rec(0, 25_000_000, 0, [0.1, 0, 0], 1.0),
            _mk_rec(1, 0, 0, [0.1, 0, 0], 1.0),
            _mk_rec(2, 50_000_000, 0, [0.1, 0, 0], 1.0)]
    assert any("non-monotonic" in e for e in tx.analyze(recs)["errors"])


# --- the REAL champion log --------------------------------------------------------------------

@needs_champion
def test_champion_schema_and_gates():
    records, meta, _ = tx.load_run(_CHAMPION)
    assert len(records) == 287
    assert meta["ego_rate_scale"] == 1.2 and meta["virtual_flip"] is True
    a = tx.analyze(records)
    assert not a["errors"]
    ks = [(t["k"], t["gate_from"], t["gate_to"]) for t in a["gate_transitions"]]
    assert ks == [(76, 0, 1), (132, 1, 2), (159, 2, 3), (193, 3, 4), (243, 4, 5)]
    assert a["collective_consistency_err"] < 1e-5      # a_thrust carries the FULL thrust channel
    assert 25.0 < a["effective_hz"] < 28.0             # the under-run that makes ZOH mandatory


@needs_champion
def test_champion_inversion_reproduces_logged_wire():
    records, _, _ = tx.load_run(_CHAMPION)
    for flip in (False, True):
        rows, rate_err, coll_err = tx.build_source_rows(records, flip)
        assert len(rows) == 287
        # rates: the inversion re-emits the LOGGED values exactly (mapping-logic pin).
        assert rate_err < 1e-9
        # collective: bounded by the log's independent 5-dp rounding of normed_thrust and
        # collective (champion measured 5.7e-6) -- NOT a mapping error.
        assert coll_err < 2e-5
        assert all(abs(v) <= 1.0 + 1e-6 for r in rows for v in r["a"])


@needs_champion
def test_champion_end_to_end_cli(tmp_path):
    out = tmp_path / "tape_g1.csv"
    rc = tx.main([str(_CHAMPION), "-o", str(out), "--truncate-at-gate", "1",
                  "--margin-ticks", "12", "--validate"])
    assert rc == 0
    assert out.is_file() and out.with_suffix(".raw.csv").is_file()
    sidecar = json.loads(out.with_suffix(".tapemeta.json").read_text())
    assert sidecar["cut_src_index"] == 132             # gate 1 -> 2 pass
    assert sidecar["n_source_ticks"] == 132 + 1 + 12
    assert sidecar["recorded_ego_rate_scale"] == 1.2
    assert "--ego-rate-scale 1.2" in sidecar["replay_cmd"]
    assert "--no-virtual-flip" in sidecar["replay_cmd"]
    assert "--sysid-climb-s 0" in sidecar["replay_cmd"]

    # parse the tape EXACTLY like the player does (fly_rl.py:2057-2063)
    with open(out, newline="") as f:
        srows = [r for r in csv.reader(f) if r]
    prog = [[float(x) for x in r[1:5]] for r in srows[1:]]
    labels = [(r[5] if len(r) > 5 else "") for r in srows[1:]]
    assert all(len(p) == 4 for p in prog)
    assert all(abs(v) <= 1.0 + 1e-6 for p in prog for v in p)
    assert labels[0].startswith("g0") and labels[-1].startswith("g2")
    # 40 Hz grid over the truncated span: rows == floor(t_end*40)+1, and the tape is LONGER
    # in rows than its source span at 26.4 Hz effective (ZOH upsamples ~1.5x)
    records, _, _ = tx.load_run(_CHAMPION)
    t_end = (records[144]["sim_time_ns"] - records[0]["sim_time_ns"]) / 1e9
    assert len(prog) == math.floor(t_end * 40.0 + 1e-9) + 1

    # spot-check fidelity at the tape row covering source k=100: forward(tape row) must equal
    # the logged wire values of the row it holds
    with open(out.with_suffix(".raw.csv"), newline="") as f:
        rrows = [r for r in csv.reader(f) if r]
    raw = rrows[1 + 100]
    a = [float(x) for x in raw[1:5]]
    rf, coll, _ = tx.forward_wire(a, False)
    rec = records[100]
    assert max(abs(x - y) for x, y in zip(rf, rec["rate_frd"])) < 1e-6
    assert abs(coll - rec["collective"]) < 1e-5
