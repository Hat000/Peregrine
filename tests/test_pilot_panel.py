"""Pilot-panel recipe guards (Patch-1 WP6) — the model-pick re-pin + recipe-drift computation as PURE
functions, unit-tested without launching the web server.

Motivation (2026-07-19): UI knob state RODE ACROSS model switches -- ego_yaw_clamp 0.35 and
ego_gate_z_bias 0.25 silently rode into a v16 batch and corrupted it. WP5 pins those knobs in the
recipes; WP6a resets recipe-managed knobs on a model-pick; WP6b/c surface any residual drift into
meta.json + a flight-start warning. These test the pure logic behind that.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import pilot_panel as P  # noqa: E402  (import is side-effect-free: the server starts only under __main__)


def test_recipe_v1_pins_the_two_ride_in_knobs():
    """WP5: the base recipe explicitly pins BOTH knobs that rode in -- yaw 0.7 and z-bias 0.0 -- so
    v1/v15/v16 all RESULT in those safe values on a model-pick (v15/v16 inherit the _V1 base)."""
    for recipe in (P._V1_RECIPE, P._V15_RECIPE, P._V16_RECIPE):
        assert recipe["ego_yaw_clamp"] == 0.7
        assert recipe["ego_gate_z_bias"] == 0.0


def test_repin_clears_prior_model_ride_in():
    """WP6a: picking v16 with a z-bias 0.25 + yaw 0.35 riding from a prior model forces both to the
    recipe pins, while a non-recipe field (label / endpoint) passes through untouched."""
    cur = {"ego_gate_z_bias": "0.25", "ego_yaw_clamp": "0.35", "label": "keepme", "endpoint": "x"}
    new = P.repin_recipe(cur, P._V16_RECIPE)
    assert float(new["ego_gate_z_bias"]) == 0.0
    assert float(new["ego_yaw_clamp"]) == 0.7
    assert new["label"] == "keepme" and new["endpoint"] == "x"


def test_repin_resets_managed_knob_absent_from_picked_recipe_to_default():
    """WP6a: a recipe-managed knob (some recipe pins it) that the PICKED recipe does NOT pin resets to
    its schema default -- it cannot ride at its prior-model value."""
    default_pitch = P._schema_default("ego_pitch_clamp")
    cur = {"ego_pitch_clamp": "3.0"}                      # rode from a prior model
    new = P.repin_recipe(cur, {"ego_yaw_clamp": 0.7})    # a bare recipe that does not pin pitch
    assert float(new["ego_pitch_clamp"]) == float(default_pitch)
    assert float(new["ego_yaw_clamp"]) == 0.7


def test_managed_keys_cover_the_incident_knobs():
    """The recipe-managed set (union of all recipe pins, minus label) includes the two knobs that rode
    in -- so a model-pick resets them regardless of which recipe is picked next."""
    managed = P._recipe_managed_keys()
    assert "ego_yaw_clamp" in managed and "ego_gate_z_bias" in managed
    assert "label" not in managed


def test_drift_flags_ride_in_and_is_clean_when_matched():
    """WP6b: a flown config that differs from the picked model's recipe pins reports drift; a config
    produced by repin (== the recipe) reports NONE."""
    dirty = {"ego_ckpt": "ckpts/v16Qs0_final_actor.pth",
             "ego_gate_z_bias": "0.25", "ego_yaw_clamp": "0.35"}
    drift = P.compute_recipe_drift(dirty)
    assert "ego_gate_z_bias" in drift and "ego_yaw_clamp" in drift
    clean = P.repin_recipe({"ego_ckpt": "v16Qs0_final_actor.pth"}, P._V16_RECIPE)
    clean["ego_ckpt"] = "v16Qs0_final_actor.pth"
    assert P.compute_recipe_drift(clean) == []


def test_drift_unknown_or_missing_model_is_empty():
    """A model with no recipe (or no ego_ckpt at all, e.g. a VQ1/gate-seeker flight) has no drift."""
    assert P.compute_recipe_drift({"ego_ckpt": "ckpts/mystery.pth"}) == []
    assert P.compute_recipe_drift({}) == []


def test_build_cmd_passes_recipe_drift_flag_and_warns_when_dirty():
    """WP6b/c end-to-end (server side): build_cmd appends --recipe-drift (so fly_rl records it +
    warns) and adds a UI warning when the flown config drifts from the picked model's recipe."""
    vals = {"ego_ckpt": "ckpts/v16Qs0_final_actor.pth", "ego_gate_z_bias": "0.25"}
    argv, warn = P.build_cmd(vals)
    assert "--recipe-drift" in argv
    keys = argv[argv.index("--recipe-drift") + 1]
    assert "ego_gate_z_bias" in keys
    assert any("recipe drift" in w for w in warn)


def test_build_cmd_no_drift_flag_when_clean():
    """A recipe-matched (repin) config adds NO --recipe-drift flag and NO drift warning."""
    clean = P.repin_recipe({"ego_ckpt": "v16Qs0_final_actor.pth"}, P._V16_RECIPE)
    clean["ego_ckpt"] = "ckpts/v16Qs0_final_actor.pth"
    argv, warn = P.build_cmd(clean)
    assert "--recipe-drift" not in argv
    assert not any("recipe drift" in w for w in warn)


def test_acquire_range_is_a_knob_and_defaults_to_the_record_value():
    """2026-07-21: there are TWO range caps and only one was reachable. --ego-max-valid-range (30)
    drops far poses from the candidate pool; ``max_acquire_range_m`` (22) then bounds what may be
    LOCKED -- and it was a hard-coded GateSeekerConfig default with no flag and no panel field, so a
    gate at 25 m was VALID but UNLOCKABLE and nothing the pilot could set closed that blind window.
    It is now a real knob; the default reproduces every flight up to and including the 9-gate record."""
    spec = P.BY_KEY["ego_max_acquire_range"]
    assert spec["flag"] == "--ego-max-acquire-range"
    assert spec["default"] == 22.0
    assert spec["group"] == P.BY_KEY["ego_max_valid_range"]["group"]   # sits beside the cap it hid behind

    argv, _ = P.build_cmd({"ego_ckpt": "ckpts/v16Qs1_final_actor.pth"})
    assert argv[argv.index("--ego-max-acquire-range") + 1] == "22.0"
    argv, _ = P.build_cmd({"ego_ckpt": "ckpts/v16Qs1_final_actor.pth",
                           "ego_max_acquire_range": "28"})
    assert argv[argv.index("--ego-max-acquire-range") + 1] == "28"


def test_acquire_range_is_not_recipe_managed():
    """The acquire cap is a PROBE knob, not a recipe pin: picking a model must not silently reset it
    (unlike the ride-in knobs of WP5/WP6a), so a sweep survives a checkpoint switch."""
    assert "ego_max_acquire_range" not in P._recipe_managed_keys()


# --- poll cost (2026-07-22 launch-latency bug) ---------------------------------------------
# refresh_pilots() runs every 2 s from the browser and holds _LOCK, which launch() also needs.
# It used to re-read EVERY tracked pilot's log AND json.loads every line of its ego_timing.jsonl
# (one line per control tick) on every poll. With 1061 accumulated pilots that measured 1.44 s per
# refresh against a 2 s interval, so polls queued and pressing Launch waited ~30 s behind them.
# Two guards below: results must be unchanged, and the work must not scale with history.

def _mk_pilot(tmp_path, key, started_ts, *, session="s1", gates=3):
    log = tmp_path / f"{key}.log"
    log.write_text(f"recording -> data\\runs\\{session}\nflight 1: FINISHED gates={gates}\n")
    return dict(id=key, label="t", cmd=[], cmd_str="", log=str(log), pid=1,
                started="00:00:00", started_ts=started_ts, session=None,
                state="LAUNCHING", gates=None, warnings=[], config={})


def test_parse_log_caches_on_mtime_and_reparses_when_the_log_grows(tmp_path, monkeypatch):
    import tools.pilot_panel as pp
    log = tmp_path / "a.log"
    log.write_text("Waiting PASSIVELY\n")
    pp._LOG_CACHE.clear()
    assert pp._parse_log(str(log))[1] == "WAITING"
    reads = {"n": 0}
    real = pp.Path.read_text

    def counting(self, *a, **k):
        reads["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(pp.Path, "read_text", counting)
    for _ in range(5):
        pp._parse_log(str(log))
    assert reads["n"] == 0, "an unchanged log must not be re-read"

    # a LIVE pilot keeps writing -- the cache must miss and pick the new state up
    log.write_text("recording -> data\\runs\\sess9\nflight 1: CRASH gates=7\n")
    sess, state, gates = pp._parse_log(str(log))
    assert (sess, state, gates) == ("sess9", "CRASH", 7)


def test_refresh_is_bounded_by_history_not_proportional_to_it(tmp_path, monkeypatch):
    import tools.pilot_panel as pp
    monkeypatch.setattr(pp, "PILOTS", {}, raising=False)
    monkeypatch.setattr(pp, "_procs", {}, raising=False)
    monkeypatch.setattr(pp, "_save_state", lambda: None)
    monkeypatch.setattr(pp, "_loop_hz", lambda s: None)
    pp._LOG_CACHE.clear()
    for i in range(pp._REFRESH_RECENT * 4):
        pp.PILOTS[f"p{i}"] = _mk_pilot(tmp_path, f"p{i}", float(i))

    seen = []
    real = pp._parse_log
    monkeypatch.setattr(pp, "_parse_log", lambda p: (seen.append(p), real(p))[1])
    rows = pp.refresh_pilots()

    assert len(rows) == len(pp.PILOTS), "every pilot must still be shown in the UI"
    assert len(seen) <= pp._REFRESH_RECENT, \
        f"poll examined {len(seen)} pilots; must stay bounded by _REFRESH_RECENT"
    # and it must be the NEWEST ones that got refreshed
    assert any(f"p{len(pp.PILOTS) - 1}.log" in p for p in seen)


def test_live_pilot_is_always_refreshed_even_when_old(tmp_path, monkeypatch):
    """A pilot that survived a panel restart is old by started_ts but still writing. It must not
    fall out of the refresh window, or its gate count would freeze in the UI."""
    import tools.pilot_panel as pp
    monkeypatch.setattr(pp, "PILOTS", {}, raising=False)
    monkeypatch.setattr(pp, "_procs", {}, raising=False)
    monkeypatch.setattr(pp, "_save_state", lambda: None)
    monkeypatch.setattr(pp, "_loop_hz", lambda s: None)
    pp._LOG_CACHE.clear()

    pp.PILOTS["old_live"] = _mk_pilot(tmp_path, "old_live", 0.0, session="live", gates=1)
    for i in range(pp._REFRESH_RECENT * 2):
        pp.PILOTS[f"p{i}"] = _mk_pilot(tmp_path, f"p{i}", float(i + 100))

    class _Live:
        def poll(self):
            return None

    pp._procs["old_live"] = _Live()
    assert "old_live" in pp._refresh_keys()
    pp.refresh_pilots()
    assert pp.PILOTS["old_live"]["running"] is True
    assert pp.PILOTS["old_live"]["gates"] == 1


def test_state_is_written_only_when_something_changed(tmp_path, monkeypatch):
    """pilots.json is 4 MB here; writing it every 2 s was pure disk churn next to the sim."""
    import tools.pilot_panel as pp
    monkeypatch.setattr(pp, "PILOTS", {}, raising=False)
    monkeypatch.setattr(pp, "_procs", {}, raising=False)
    monkeypatch.setattr(pp, "_loop_hz", lambda s: None)
    pp._LOG_CACHE.clear()
    pp.PILOTS["p0"] = _mk_pilot(tmp_path, "p0", 1.0)
    saves = {"n": 0}
    monkeypatch.setattr(pp, "_save_state", lambda: saves.__setitem__("n", saves["n"] + 1))

    pp.refresh_pilots()
    assert saves["n"] == 1, "first poll learns the pilot's state and must persist it"
    pp.refresh_pilots()
    pp.refresh_pilots()
    assert saves["n"] == 1, "nothing changed -> no further writes"
