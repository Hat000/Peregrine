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
