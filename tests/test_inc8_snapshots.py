"""Laptop unit tests for rl/inc8_snapshots.py -- dense checkpoint retention cadence + OFF no-op."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl"))

import inc8_snapshots as sn


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_resolve_off_when_unset_or_zero():
    assert sn.resolve_snapshots(_Cfg()) is None
    assert sn.resolve_snapshots(_Cfg(snapshot_every=0)) is None
    assert sn.resolve_snapshots(_Cfg(snapshot_every=-5)) is None


def test_resolve_on_reads_defaults():
    s = sn.resolve_snapshots(_Cfg(snapshot_every=100, n_updates=4000))
    assert s == {"every": 100, "from_frac": 0.0, "n_updates": 4000}


def test_resolve_on_reads_from_frac():
    s = sn.resolve_snapshots(_Cfg(snapshot_every=50, snapshot_from_frac=0.4, n_updates=4000))
    assert s["from_frac"] == 0.4


def test_should_snapshot_cadence():
    s = {"every": 100, "from_frac": 0.0, "n_updates": 4000}
    assert sn.should_snapshot(100, s)
    assert sn.should_snapshot(4000, s)
    assert not sn.should_snapshot(150, s)
    assert not sn.should_snapshot(0, s)        # update 0 never snapshots


def test_should_snapshot_from_frac_gate():
    s = {"every": 100, "from_frac": 0.4, "n_updates": 4000}   # gate at update 1600
    assert not sn.should_snapshot(1500, s)
    assert sn.should_snapshot(1600, s)
    assert sn.should_snapshot(2000, s)


def test_should_snapshot_from_frac_skipped_when_n_updates_unknown():
    s = {"every": 100, "from_frac": 0.4, "n_updates": 0}
    assert sn.should_snapshot(100, s)          # no n_updates -> from_frac gate disabled


def test_maybe_write_snapshot_saves_on_hit(tmp_path):
    saved = {}

    class _Agent:
        def save(self, path):
            saved["path"] = path
            os.makedirs(path, exist_ok=True)

    class _Logger:
        logdir = str(tmp_path)

    s = {"every": 100, "from_frac": 0.0, "n_updates": 4000}
    d = sn.maybe_write_snapshot(_Agent(), _Logger(), 200, s)
    assert d is not None and d.endswith(os.path.join("snapshots", "upd00200"))
    assert saved["path"] == d

    saved.clear()
    assert sn.maybe_write_snapshot(_Agent(), _Logger(), 250, s) is None
    assert saved == {}                         # no save on a non-cadence update
