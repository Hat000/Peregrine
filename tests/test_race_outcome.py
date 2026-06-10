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


def test_pre_race_residue_is_discarded_and_only_final_epoch_scored():
    # Regression for data/runs/20260607_200505_vq1: the recording BEGINS with the previous
    # race's residual RACE_STATUS (already finished @35.32 s, active_gate_index already 6),
    # then the sim resets to 0 and the live run climbs 0->5. The stale baseline must NOT be
    # counted: gates_passed should be 5 (the live epoch), finish must NOT latch from the residue.
    rs = [
        _rs(0.0, 6, started=True, finished=True, finish_ns=35_316_936_492, last=35_316_936_492),
        _rs(5.4, 0, started=False),                    # sim resets for the new run
        _rs(16.5, 1, last=6_812_578_201),              # gate 0
        _rs(21.3, 2, last=11_567_998_886),             # gate 1
        _rs(27.0, 3, last=17_231_594_085),             # gate 2
        _rs(34.6, 4, last=24_798_587_799),             # gate 3
        _rs(39.3, 5, last=29_576_770_782),             # gate 4
    ]
    out = analyze_outcome(rs, [])
    assert out["had_pre_race_residue"]
    assert out["gates_passed"] == 5 and [p["gate"] for p in out["passes"]] == [0, 1, 2, 3, 4]
    assert out["max_active_gate_index"] == 5
    assert not out["finished"] and not out["clean_finish"]   # this run's finish was not recorded
    assert out["recognized_time_ns"] is None                 # 35.32 s belonged to the PRIOR race
    assert out["started"]


def test_gate_clip_without_collision_message_is_5_of_6_not_finished():
    # Regression for data/runs/20260607_200906_vq1: the drone clipped the 6th gate, so the sim
    # did NOT advance past gate 5 and never finished -- but this sim emitted NO COLLISION frame
    # for the light clip. The honest verdict is 5 passes, not finished, and (lacking collision
    # data) no contact flag; clean_finish stays False because the race did not finish.
    rs = [_rs(0.0, 0, started=False)] + [_rs(float(i), i, last=i * 5_000_000_000) for i in range(1, 6)]
    out = analyze_outcome(rs, [])                            # no COLLISION events available
    assert out["gates_passed"] == 5 and out["max_active_gate_index"] == 5
    assert not out["finished"] and not out["clean_finish"]
    assert out["n_gate_collisions"] == 0 and out["n_pass_contact"] == 0
    assert not out["had_pre_race_residue"]


def test_multi_gate_jump_counts_all_passed():
    # active jumps 0 -> 3 in one sample (e.g. sparse RACE_STATUS) -> gates 0,1,2 all passed
    rs = [_rs(0.0, 0, started=False), _rs(1.0, 0), _rs(2.0, 3, finished=True, finish_ns=10_000_000_000)]
    out = analyze_outcome(rs, [])
    assert out["gates_passed"] == 3 and [p["gate"] for p in out["passes"]] == [0, 1, 2]
    assert out["clean_finish"]
