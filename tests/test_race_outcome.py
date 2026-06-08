"""Tests for the authoritative race-outcome oracle (RACE_STATUS + COLLISION -> per-gate verdict)."""
from __future__ import annotations

from racer.race_outcome import analyze_outcome


def _rs(t, active, started=True, finished=False, finish_ns=-1, last=-1):
    return {"t": t, "active_gate_index": active, "started": started, "finished": finished,
            "finish_ns": finish_ns, "last_gate_race_time": last}


def _clean_3gate():
    # active 0 -> 1 -> 2 -> 3 ; finishes at the last sample with a recognized time
    return [
        _rs(0.0, 0, started=False),
        _rs(1.0, 0),
        _rs(2.0, 1),                                   # gate 0 passed @2
        _rs(3.0, 2),                                   # gate 1 passed @3
        _rs(4.0, 3, finished=True, finish_ns=35_300_000_000, last=35_300_000_000),  # gate 2 passed @4 + finish
    ]


def test_clean_finish_no_collisions():
    out = analyze_outcome(_clean_3gate(), [])
    assert out["finished"] and out["clean_finish"]
    assert out["gates_passed"] == 3
    assert out["n_pass_clean"] == 3 and out["n_pass_contact"] == 0
    assert [p["gate"] for p in out["passes"]] == [0, 1, 2]
    assert out["recognized_time_ns"] == 35_300_000_000


def test_pass_with_contact_flags_that_gate():
    # a gate collision coincident with the gate-1 pass (@3.0)
    cols = [{"t": 3.05, "id": 1001, "threat_level": 2, "impulse": 1.0}]
    out = analyze_outcome(_clean_3gate(), cols)
    assert out["n_pass_contact"] == 1 and out["n_pass_clean"] == 2
    contact = [p for p in out["passes"] if p["contact"]]
    assert len(contact) == 1 and contact[0]["gate"] == 1 and contact[0]["threat_level"] == 2
    assert not out["clean_finish"]                      # contact => not clean


def test_environment_collision_breaks_clean_finish():
    cols = [{"t": 2.5, "id": 1002, "threat_level": 1, "impulse": 0.5}]
    out = analyze_outcome(_clean_3gate(), cols)
    assert out["n_env_collisions"] == 1
    assert out["n_pass_contact"] == 0                  # env hit is not a gate contact
    assert not out["clean_finish"]


def test_gate_hit_without_pass_is_unmatched():
    # a gate collision far from any pass (no advance) -> hit that did not pass
    cols = [{"t": 10.0, "id": 1001, "threat_level": 1, "impulse": 0.8}]
    out = analyze_outcome(_clean_3gate(), cols)
    assert out["n_gate_collisions"] == 1
    assert out["n_gate_collisions_no_pass"] == 1
    assert out["n_pass_contact"] == 0
    assert not out["clean_finish"]


def test_contact_window_boundary():
    # single gate passed @2.0 (no other pass nearby); collision @2.6 is 0.6 s away
    rs = [_rs(0.0, 0, started=False), _rs(1.0, 0), _rs(2.0, 1, finished=True, finish_ns=10_000_000_000)]
    cols = [{"t": 2.6, "id": 1001, "threat_level": 1, "impulse": 0.5}]
    out = analyze_outcome(rs, cols, contact_window_s=0.5)   # 0.6 > 0.5 -> outside window
    assert out["n_pass_contact"] == 0 and out["n_gate_collisions_no_pass"] == 1
    out2 = analyze_outcome(rs, cols, contact_window_s=1.0)  # 0.6 <= 1.0 -> contact on gate 0
    assert out2["n_pass_contact"] == 1 and out2["n_gate_collisions_no_pass"] == 0


def test_each_collision_matches_at_most_one_pass():
    # two passes close together, one collision -> only one pass flagged contact
    rs = [_rs(0.0, 0, started=False), _rs(1.0, 0), _rs(2.0, 1), _rs(2.1, 2)]
    cols = [{"t": 2.05, "id": 1001, "threat_level": 1, "impulse": 0.5}]
    out = analyze_outcome(rs, cols)
    assert out["n_pass_contact"] == 1


def test_no_race_status_returns_error():
    out = analyze_outcome([], [{"t": 1.0, "id": 1001, "threat_level": 1, "impulse": 0.1}])
    assert "error" in out and not out["clean_finish"]
    assert out["n_gate_collisions"] == 1


def test_multi_gate_jump_counts_all_passed():
    # active jumps 0 -> 3 in one sample (e.g. sparse RACE_STATUS) -> gates 0,1,2 all passed
    rs = [_rs(0.0, 0, started=False), _rs(1.0, 0), _rs(2.0, 3, finished=True, finish_ns=10_000_000_000)]
    out = analyze_outcome(rs, [])
    assert out["gates_passed"] == 3 and [p["gate"] for p in out["passes"]] == [0, 1, 2]
    assert out["clean_finish"]
