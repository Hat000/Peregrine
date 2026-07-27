"""POST-IMPACT FLIGHT RECORDER — the aftermath capture that closes the at-gate evidence gap.

THE GAP [2026-07-27]: every stop condition in ``_fly_ego``'s flight loop is evaluated at the TOP
of the tick body, while the per-tick forensics record is appended at the BOTTOM (after the command
send). So a ``break`` — the hard-collision abort above all — discards the detecting tick whole and
never samples again: ``ego_obs.jsonl`` always ends BEFORE the strike it is supposed to explain, and
the killing axis (lateral gate strike / vertical undershoot / floor) is unrecoverable.

``fly_rl.post_impact_capture`` closes it by sampling — never commanding — for a bounded window
after the terminal tick. These tests pin the properties that make it safe to run DEFAULT-ON:

  * it NEVER sends a command (it is a recorder; the wire stays as quiet as it is today);
  * it restores ``client.collisions`` to its entry length, so ``result["collisions"]``,
    ``meta.json`` collisions, the batch summary and the next flight's ``n_coll0`` cannot move;
  * it never raises — a recorder must not cost the caller the flight log it already holds;
  * every row is flagged ``post_terminal=True`` and carries NO command channel, so aftermath can
    never be replayed as flight (``rl/tape_extract.py`` would otherwise turn a crashed drone's
    tumble into a ``--sysid-replay`` command tape);
  * the terminal-state gate is CRASH-only — an already-INVALID run, so the extra armed window
    cannot change any verdict.

Plus the wire half: the ``COLLISION`` handler now keeps the FULL MAVLink payload, so gate-vs-
environment (id 1001/1002) and any contact geometry the sim publishes survive to disk.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402

from racer.mavlink_client import MavlinkClient  # noqa: E402


# --------------------------------------------------------------------------- #
# a fake wire: exposes exactly the surface post_impact_capture is allowed to touch
# --------------------------------------------------------------------------- #
class FakeClient:
    """Minimal stand-in for MavlinkClient. Counts commands so the no-command contract is provable."""

    def __init__(self, *, gyro=(0.1, -0.2, 0.3), accel=(0.0, 0.0, -9.81),
                 quat=(1.0, 0.0, 0.0, 0.0), armed=True, race_status=None,
                 frame_id=7, collide_after=None, pump_raises=False):
        self.collisions: list[dict] = []
        self.race_status = race_status
        self.commands: list = []          # anything sent here is a CONTRACT VIOLATION
        self.n_pumps = 0
        self._collide_after = collide_after
        self._pump_raises = pump_raises
        self._t = 1_000_000_000
        self.state = SimpleNamespace(
            sim_time_ns=self._t,
            gyro_body=(None if gyro is None else np.asarray(gyro, dtype=np.float64)),
            accel_body=(None if accel is None else np.asarray(accel, dtype=np.float64)),
            orientation_ned_wxyz=(None if quat is None else np.asarray(quat, dtype=np.float64)),
            armed=armed)
        self._latest_frame = None if frame_id is None else SimpleNamespace(frame_id=frame_id)

    def pump(self):
        if self._pump_raises:
            raise RuntimeError("wire exploded")
        self.n_pumps += 1
        self._t += 1_000_000
        self.state.sim_time_ns = self._t
        if self._collide_after is not None and self.n_pumps >= self._collide_after:
            self._collide_after = None
            self.collisions.append({"id": 1002, "threat_level": 2, "impulse": 1.5,
                                    "src": 0, "action": 0,
                                    "altitude_minimum_delta": 0.0,
                                    "time_to_minimum_delta": 0.0,
                                    "recv_monotonic_ns": 1, "sim_time_ns": self._t})

    def send_command(self, *a, **kw):      # must never be reached
        self.commands.append((a, kw))


def _capture(client, **over):
    kw = dict(seconds=0.06, tick=0.01, k0=215, gate_index=4, final_state="CRASH")
    kw.update(over)
    return fly_rl.post_impact_capture(client, **kw)


# --------------------------------------------------------------------------- #
# the safety contract
# --------------------------------------------------------------------------- #
def test_capture_never_sends_a_command():
    """The whole justification for DEFAULT-ON: it records, it does not fly."""
    c = FakeClient()
    rows, _, _ = _capture(c)
    assert rows, "expected at least one aftermath row"
    assert c.commands == [], f"post-impact capture SENT {len(c.commands)} command(s) — it must not"
    assert c.n_pumps > 0, "it must still pump the link to receive telemetry"


def test_capture_restores_the_collision_ledger():
    """Aftershocks are RETURNED, not left on the client — so no existing metric moves."""
    c = FakeClient(collide_after=1)
    c.collisions.append({"id": 1001, "threat_level": 2, "impulse": 4.0})   # the terminal contact
    n_before = len(c.collisions)
    rows, contacts, _ = _capture(c)
    assert len(c.collisions) == n_before, "client.collisions must be restored to its entry length"
    assert len(contacts) == 1 and contacts[0]["id"] == 1002, "the aftershock must be returned"
    assert c.collisions[0]["id"] == 1001, "the pre-existing ledger must be untouched"
    # the per-tick counter tracks the aftershock as it lands
    assert rows[-1]["n_contacts"] == 1


def test_capture_never_raises_and_still_restores_on_failure():
    """A recorder must not cost the caller the flight log it already holds."""
    c = FakeClient(pump_raises=True)
    c.collisions.append({"id": 1001, "threat_level": 2, "impulse": 4.0})
    rows, contacts, elapsed = _capture(c)          # must NOT propagate RuntimeError
    assert rows == [] and contacts == []
    assert elapsed >= 0.0
    assert len(c.collisions) == 1, "the ledger must still be restored on the failure path"


def test_capture_window_is_bounded():
    c = FakeClient()
    _, _, elapsed = _capture(c, seconds=0.05)
    assert 0.05 <= elapsed < 1.0, f"window ran {elapsed:.3f}s — it must stay bounded"


def test_zero_seconds_captures_nothing():
    c = FakeClient()
    rows, contacts, _ = _capture(c, seconds=0.0)
    assert rows == [] and contacts == [] and c.n_pumps == 0


# --------------------------------------------------------------------------- #
# the row schema — aftermath must be unmistakable, and never replayable as flight
# --------------------------------------------------------------------------- #
def test_every_row_is_flagged_post_terminal():
    rows, _, _ = _capture(FakeClient())
    assert all(r["post_terminal"] is True for r in rows)
    assert all(r["final_state"] == "CRASH" for r in rows)


def test_rows_carry_no_command_channel():
    """tape_extract.analyze() keys off rate_frd/collective/normed_thrust. Aftermath has none of
    them, so it can never be mistaken for — or replayed as — commanded flight."""
    rows, _, _ = _capture(FakeClient())
    for r in rows:
        for forbidden in ("rate_frd", "collective", "normed_thrust", "obs",
                          "actor_mean", "act_raw"):
            assert forbidden not in r, f"aftermath row must not carry '{forbidden}'"


def test_k_continues_the_flight_sequence():
    rows, _, _ = _capture(FakeClient(), k0=215)
    ks = [r["k"] for r in rows]
    assert ks == list(range(215, 215 + len(ks))), "k must continue ego_obs's sequence for joins"
    assert all(a < b for a, b in zip([r["sim_time_ns"] for r in rows],
                                     [r["sim_time_ns"] for r in rows][1:]))
    assert all(a <= b for a, b in zip([r["t_since_terminal_s"] for r in rows],
                                      [r["t_since_terminal_s"] for r in rows][1:]))


def test_rows_carry_the_axis_discriminator_and_the_video_join():
    """gyro+accel in body FRD are what separate a lateral strike from a floor hit; frame_id is
    the join back to the recorded video the pilot's eyes adjudicate from."""
    rows, _, _ = _capture(FakeClient(gyro=(0.1, -0.2, 0.3), accel=(1.0, 2.0, -9.81), frame_id=99))
    r = rows[0]
    assert r["gyro_frd"] == pytest.approx([0.1, -0.2, 0.3])
    assert r["accel_frd"] == pytest.approx([1.0, 2.0, -9.81])
    assert r["quat_ned_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert r["frame_id"] == 99 and r["armed"] is True


def test_rows_survive_a_silent_wire():
    """VQ2 blocks ODOMETRY and the race can go quiet — every field must be None-tolerant."""
    c = FakeClient(gyro=None, accel=None, quat=None, frame_id=None, race_status=None)
    rows, _, _ = _capture(c, gate_index=4)
    r = rows[0]
    assert r["gyro_frd"] is None and r["accel_frd"] is None and r["quat_ned_wxyz"] is None
    assert r["frame_id"] is None
    assert r["gate_index"] == 4, "must fall back to the terminal-tick gate index"


def test_gate_index_prefers_live_race_status():
    c = FakeClient(race_status={"active_gate_index": 5})
    rows, _, _ = _capture(c, gate_index=4)
    assert rows[0]["gate_index"] == 5


def test_rows_are_json_serializable():
    """They are written with json.dumps at flush; a stray numpy array would lose the whole file."""
    rows, _, _ = _capture(FakeClient())
    for r in rows:
        json.loads(json.dumps(r))


# --------------------------------------------------------------------------- #
# the terminal-state gate + CLI surface
# --------------------------------------------------------------------------- #
def test_post_terminal_states_is_crash_only():
    """Widening this is a deliberate act: FINISHED/SIM_RESET must keep today's prompt teardown
    (SIM_RESET exists to CUT commands; a finished run must not fly on a latched command)."""
    assert fly_rl._POST_TERMINAL_STATES == ("CRASH",)
    for state in ("FINISHED", "SIM_RESET", "TIMEOUT", "STALLED", "SYSID_DONE"):
        assert state not in fly_rl._POST_TERMINAL_STATES


def test_cli_default_is_on_and_can_be_disabled():
    ap = fly_rl.build_parser()
    assert ap.parse_args([]).ego_post_terminal_s == pytest.approx(0.5)
    assert ap.parse_args(["--ego-post-terminal-s", "0"]).ego_post_terminal_s == 0.0
    assert ap.parse_args(["--ego-post-terminal-s", "1.5"]).ego_post_terminal_s == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# the wire half: the FULL COLLISION payload now survives
# --------------------------------------------------------------------------- #
def _collision_msg(**over):
    f = dict(id=1001, src=1, action=0, threat_level=2, time_to_minimum_delta=0.25,
             altitude_minimum_delta=-1.75, horizontal_minimum_delta=3.5)
    f.update(over)
    m = SimpleNamespace(**f)
    m.get_type = lambda: "COLLISION"
    return m


def test_collision_capture_is_additive_and_complete():
    """id 1001=gate vs 1002=environment is the strongest failure classifier the sim publishes and
    was previously persisted only as an integer COUNT. The two nominally-unused floats are kept
    because the sim already repurposes horizontal_minimum_delta — they are the only candidates
    for a contact geometry."""
    c = MavlinkClient()
    c._handle(_collision_msg())
    rec = c.collisions[0]
    # unchanged (existing readers: race_outcome, the threat_level>=2 aborts)
    assert rec["id"] == 1001 and rec["threat_level"] == 2
    assert rec["impulse"] == pytest.approx(3.5)
    assert "recv_monotonic_ns" in rec and "sim_time_ns" in rec
    # additive
    assert rec["src"] == 1 and rec["action"] == 0
    assert rec["altitude_minimum_delta"] == pytest.approx(-1.75)
    assert rec["time_to_minimum_delta"] == pytest.approx(0.25)


def test_collision_capture_tolerates_a_sparse_message():
    """Absent fields must default, never raise — the sim is free to omit any of them."""
    m = SimpleNamespace(id=1002, threat_level=1, horizontal_minimum_delta=0.2)
    m.get_type = lambda: "COLLISION"
    c = MavlinkClient()
    c._handle(m)
    rec = c.collisions[0]
    assert rec["id"] == 1002 and rec["src"] == 0 and rec["action"] == 0
    assert rec["altitude_minimum_delta"] == 0.0 and rec["time_to_minimum_delta"] == 0.0


def test_collision_records_stay_json_serializable():
    """They are embedded verbatim in ego_postimpact.jsonl's contact row."""
    c = MavlinkClient()
    c._handle(_collision_msg())
    json.loads(json.dumps(c.collisions[0]))


# --------------------------------------------------------------------------- #
# backward compatibility: ego_obs.jsonl is UNTOUCHED, and aftermath is a separate stream
# --------------------------------------------------------------------------- #
_RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"


@pytest.mark.skipif(not _RUNS.is_dir(), reason="banked session logs not present in this checkout")
def test_banked_ego_obs_logs_contain_no_post_terminal_rows():
    """The aftermath goes to ego_postimpact.jsonl, never into ego_obs.jsonl. Every consumer of
    ego_obs.jsonl (rl/tape_extract.py, scripts/replay_fix_gain.py, scripts/mine_close_range_
    failures.py, tools/render_yolo.py) therefore keeps its exact input contract: they require
    obs/rate_frd/collective/normed_thrust on EVERY row and would either KeyError or silently
    replay a crashed drone's tumble as commanded flight if aftermath were merged in."""
    sessions = sorted(p for p in _RUNS.iterdir() if (p / "ego_obs.jsonl").exists())[:40]
    if not sessions:
        pytest.skip("no banked ego sessions with ego_obs.jsonl")
    required = ("sim_time_ns", "rate_frd", "collective", "normed_thrust", "gate_index", "obs")
    for s in sessions:
        for line in (s / "ego_obs.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            assert "post_terminal" not in r, f"{s.name}: aftermath leaked into ego_obs.jsonl"
            for key in required:
                assert key in r, f"{s.name}: ego_obs row lost required key '{key}'"


@pytest.mark.skipif(not _RUNS.is_dir(), reason="banked session logs not present in this checkout")
def test_new_meta_and_log_fields_are_additive_and_absent_tolerant():
    """Pre-fix sessions have no ego_post_terminal_s and no ego_postimpact.jsonl. Analysis must
    read both with .get()/exists() defaults — pinned here so the schema stays additive."""
    sessions = sorted(p for p in _RUNS.iterdir() if (p / "meta.json").exists())[:40]
    if not sessions:
        pytest.skip("no banked sessions")
    for s in sessions:
        m = json.loads((s / "meta.json").read_text(encoding="utf-8"))
        # absent on every pre-fix session -> readers MUST default rather than index
        assert float(m.get("ego_post_terminal_s", 0.0)) >= 0.0
        assert not (s / "ego_postimpact.jsonl").exists() or \
            (s / "ego_postimpact.jsonl").stat().st_size > 0
