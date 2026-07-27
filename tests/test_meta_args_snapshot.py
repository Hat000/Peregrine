"""meta.json must record the VALUE of every flown knob, not a hand-curated subset.

THE BUG THIS PINS (2026-07-27, measured on real flights). The panel computes ``recipe_drift`` -- the
list of knob keys whose flown value differs from the picked model's recipe pin -- and fly_rl records
that LIST. On sessions ``20260727_0526`` through ``20260727_0540`` it correctly reported
``ego_assist_thrust`` as drifted on 7 of 9 flights (the pilot had turned takeoff assist off because
the transition OUT of assist sinks the drone right before a gate). But ``ego_assist_thrust`` is not
one of the ~40 hand-curated meta keys, so the FLOWN VALUE was unrecoverable from the log: the cohort
knew a knob had moved and could not say to what. That silently confounds every comparison against it.

A curated list can only ever record the knobs someone remembered to add. ``_meta_args_snapshot`` is
deliberately generic -- ``vars(args)`` filtered to JSON-safe scalars -- because the failure mode is
exactly "a knob was added and the recorder was not updated". These tests pin the properties that make
it a real fix rather than one more curated key:

  * it captures a knob that NO curated key records (the actual 07-27 hole);
  * it captures knobs that do not exist yet, i.e. it is not an enumeration;
  * it is JSON-serializable, since it is written straight into meta.json;
  * it never raises, because a recorder that can throw would cost a flight;
  * it is ADDITIVE -- the curated keys are untouched, so no existing reader changes behaviour.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402


def _args(**kw):
    base = dict(ego_assist_thrust=1.3, ego_takeoff_assist=True, rate=30.0,
                ego_ckpt="ckpts/v19Ws0_actor.pth", ego_aim_offsets="4:0,3;5:0,3",
                recipe_drift="ego_aim_offsets,ego_assist_thrust")
    base.update(kw)
    return SimpleNamespace(**base)


def test_captures_the_knob_that_recipe_drift_named_but_meta_never_recorded():
    """The exact 07-27 hole: recipe_drift said ego_assist_thrust moved, nothing said to what."""
    snap = fly_rl._meta_args_snapshot(_args(ego_assist_thrust=0.0))
    assert snap["ego_assist_thrust"] == 0.0
    # and the drifted-off case is distinguishable from the recipe pin
    assert fly_rl._meta_args_snapshot(_args())["ego_assist_thrust"] == 1.3


def test_every_key_named_by_recipe_drift_is_resolvable_to_a_value():
    """The invariant that was violated: a knob may not be flagged as drifted while its value is
    unrecoverable. Whatever the panel names in --recipe-drift must be answerable from the snapshot."""
    args = _args()
    drift = fly_rl._parse_recipe_drift(args.recipe_drift)
    snap = fly_rl._meta_args_snapshot(args)
    assert drift, "fixture must actually exercise a drift list"
    missing = [k for k in drift if k not in snap]
    assert not missing, f"recipe_drift named knobs with no recorded value: {missing}"


def test_is_not_an_enumeration_so_a_future_knob_is_captured_for_free():
    """The failure mode is 'a knob was added and the recorder was not updated' -- so a knob nobody
    has written yet must still land in the snapshot."""
    snap = fly_rl._meta_args_snapshot(_args(ego_knob_that_does_not_exist_yet=0.42))
    assert snap["ego_knob_that_does_not_exist_yet"] == 0.42


def test_output_is_json_serializable_because_it_is_written_into_meta_json():
    class Weird:
        def __str__(self):
            return "weird-repr"

    snap = fly_rl._meta_args_snapshot(_args(some_path=Path("C:/x/y.pth"), odd=Weird(),
                                            seq=[1, "a", None], nested=(2, 3)))
    json.dumps(snap)                      # must not raise
    assert snap["some_path"].endswith("y.pth")
    assert snap["odd"] == "weird-repr"
    assert snap["seq"] == [1, "a", None]


def test_never_raises_because_a_throwing_recorder_would_cost_a_flight():
    class Explodes:
        def __str__(self):
            raise RuntimeError("boom")

    snap = fly_rl._meta_args_snapshot(_args(bad=Explodes(), good=7))
    assert snap["good"] == 7               # the good keys survive
    assert "bad" not in snap               # the bad one is dropped, not fatal
    assert fly_rl._meta_args_snapshot(object()) == {}   # no vars() at all -> empty, no raise


def test_private_attributes_are_skipped():
    snap = fly_rl._meta_args_snapshot(_args(_internal="hidden", visible=1))
    assert "_internal" not in snap
    assert snap["visible"] == 1


def test_is_additive_the_curated_keys_are_untouched():
    """No existing reader of meta.json may change behaviour: args_all is recorded ALONGSIDE the
    curated keys, and the curated block still writes them itself."""
    src = (Path(__file__).resolve().parents[1] / "rl" / "fly_rl.py").read_text(encoding="utf-8")
    assert '"args_all": _meta_args_snapshot(args),' in src
    for curated in ('"ego_fix_gain"', '"ego_aim_offsets"', '"seeker_constants"', '"recipe_drift"',
                    '"ego_gate_z_bias"', '"ego_post_terminal_s"'):
        assert curated in src, f"curated meta key {curated} must survive the additive change"
