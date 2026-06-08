"""Authoritative per-gate PASS-vs-COLLISION verdict from a recording's SIM signals.

The sim tells us the truth directly -- we just have to consume it instead of our geometric
plane-crossing heuristic or eyeballing the GUI (both historically unreliable: the gate-0-saga
false-passes; the live pass-vs-collision misreads). Used to GRADE runs and as the RL validation
oracle. The two authoritative signals (parsed exactly as :mod:`racer.mavlink_client`):

* ``RACE_STATUS`` (ENCAPSULATED_DATA, data[0]==1): ``active_gate_index`` (the gate that is NEXT),
  ``started``/``finished``, ``race_finish_time_ns``, ``last_gate_race_time``. An
  ``active_gate_index`` increment i->i+1 means gate i was PASSED.
* ``COLLISION``: ``id`` (1001=gate, 1002=environment), ``threat_level`` (1..2), impulse.

:func:`analyze_outcome` is PURE (lists of plain dicts in -> verdict out) so it is unit-tested
without a recording; :func:`load_from_recording` does the tlog I/O. ``scripts/race_outcome.py`` is
the thin CLI.
"""
from __future__ import annotations

from pathlib import Path

_ENCAP_RACE_STATUS = 1
GATE_COLLISION_ID = 1001
ENV_COLLISION_ID = 1002


def analyze_outcome(race_samples: list[dict], collisions: list[dict],
                    contact_window_s: float = 0.5) -> dict:
    """Per-gate pass/collision verdict from RACE_STATUS samples + COLLISION events.

    ``race_samples``: dicts with ``t`` (s), ``active_gate_index``, ``started``, ``finished``,
    ``finish_ns``, ``last_gate_race_time``. ``collisions``: dicts with ``t`` (s), ``id``
    (1001 gate / 1002 env), ``threat_level``, ``impulse``. Both on the same time base (recv clock).

    A gate collision within ``contact_window_s`` of a pass marks that pass PASS+CONTACT (clipped
    but advanced); each gate collision is matched to at most one pass; leftovers are
    ``unmatched_gate_hits`` (a hit that did not pass). ``clean_finish`` requires finished with zero
    contact, zero unmatched gate hits, and zero environment collisions.
    """
    rs = sorted(race_samples, key=lambda s: s["t"])
    gate_hits = sorted([c for c in collisions if int(c.get("id", 0)) == GATE_COLLISION_ID],
                       key=lambda c: c["t"])
    env_hits = sorted([c for c in collisions if int(c.get("id", 0)) == ENV_COLLISION_ID],
                      key=lambda c: c["t"])

    if not rs:
        return {"error": "no RACE_STATUS in recording", "n_gate_collisions": len(gate_hits),
                "n_env_collisions": len(env_hits), "passes": [], "clean_finish": False}

    started_t = next((s["t"] for s in rs if s.get("started")), None)
    finished = any(s.get("finished") for s in rs)
    finish_t = next((s["t"] for s in rs if s.get("finished")), None)
    finish_sample = next((s for s in rs if s.get("finished")), None)
    recognized_time_ns = None
    if finish_sample is not None:
        fin = finish_sample.get("finish_ns", -1)
        recognized_time_ns = finish_sample.get("last_gate_race_time") if fin < 0 else fin

    # passes: each time active_gate_index climbs, gates [prev..new-1] were passed at that t.
    passes: list[dict] = []
    cur = rs[0]["active_gate_index"]
    for s in rs:
        a = s["active_gate_index"]
        if a > cur:
            for g in range(cur, a):
                passes.append({"gate": g, "t": s["t"]})
            cur = a
    max_active = cur

    used: set[int] = set()
    for p in passes:
        contact = None
        for i, h in enumerate(gate_hits):
            if i not in used and abs(h["t"] - p["t"]) <= contact_window_s:
                contact, _ = h, used.add(i)
                break
        p["contact"] = contact is not None
        p["threat_level"] = int(contact["threat_level"]) if contact else 0
        p["verdict"] = "PASS+CONTACT" if contact else "PASS-CLEAN"
    unmatched = [h for i, h in enumerate(gate_hits) if i not in used]

    n_contact = sum(1 for p in passes if p["contact"])
    return {
        "started": started_t is not None, "finished": finished,
        "started_t": started_t, "finish_t": finish_t, "recognized_time_ns": recognized_time_ns,
        "gates_passed": len(passes), "max_active_gate_index": max_active, "passes": passes,
        "n_pass_clean": len(passes) - n_contact, "n_pass_contact": n_contact,
        "n_gate_collisions": len(gate_hits), "n_gate_collisions_no_pass": len(unmatched),
        "unmatched_gate_hits": unmatched, "n_env_collisions": len(env_hits), "env_collisions": env_hits,
        "clean_finish": bool(finished and n_contact == 0 and not unmatched and not env_hits),
    }


def load_from_recording(session_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Extract RACE_STATUS samples + COLLISION events from a recording's ``mavlink.tlog``."""
    from racer.mavlink_client import parse_race_status
    from racer.recording import RecordingReader

    race_samples, collisions = [], []
    for m in RecordingReader(session_dir).iter_mavlink():
        ty = m.get_type()
        if ty == "ENCAPSULATED_DATA":
            raw = bytes(m.data)
            if raw and raw[0] == _ENCAP_RACE_STATUS:
                rs = parse_race_status(raw)
                if rs is not None:
                    race_samples.append({
                        "t": float(m._timestamp), "active_gate_index": rs["active_gate_index"],
                        "started": rs["started"], "finished": rs["finished"],
                        "finish_ns": rs["race_finish_time_ns"],
                        "last_gate_race_time": rs["last_gate_race_time"],
                    })
        elif ty == "COLLISION":
            collisions.append({
                "t": float(m._timestamp), "id": int(m.id),
                "threat_level": int(getattr(m, "threat_level", 0)),
                "impulse": float(getattr(m, "horizontal_minimum_delta", 0.0)),
            })
    return race_samples, collisions
