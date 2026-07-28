"""The panel's recipe pins are a TRAIN/DEPLOY CONTRACT. These tests pin the contract, not the values.

Every recipe key that mirrors a training-side constant is a place where a silent divergence costs
flights, and this project has now paid for that twice in one file:

  * ``rate`` was pinned 40.0 in _V1_RECIPE while the knob's own schema default is 30.0 and its help
    reads "training dt = 30". Because ``rate`` is inside ``_recipe_managed_keys()`` -- the set a
    model-pick RESETS -- every model pick silently re-armed it. Measured over all 670 recorded
    flights: rate 30 (n=189) reached mean max gate 2.460 against rate 40 (n=481) 1.769.
    🛑 THE OBVIOUS MECHANISM IS FALSE. The loop is COMPUTE-BOUND and never achieves its commanded
    rate: commanded 30 achieves 21.72 Hz median, commanded 40 achieves 25.34 -- so commanding 30
    lands FURTHER from the 30.03 Hz training cadence and still wins. What pays is VISION FRESHNESS
    (fresh-fix fraction 94.9% at commanded 30 vs 75.6% at 40). These tests therefore pin the VALUE
    the panel must hand the pilot, and deliberately do NOT assert the cadence-matching rationale.
  * ``ego_stale_horizon`` was pinned 0.6 from _V17 onward while training uses 0.5, so obs[14] =
    clamp(1 - age/horizon, 0, 1) reported ~20% more confidence for the same staleness.

And one hole of a different shape: v2.0 shipped with NO MODEL_DEFAULTS entry at all, so its six
flown sessions inherited whatever the panel last held -- all six at rate 40, with ego_pitch_clamp
drifting 20/20/20/0/0/30 across the cohort. A checkpoint absent from MODEL_DEFAULTS gets neither a
reset nor a pin, which is precisely the hole _recipe_managed_keys() exists to close.

The final test is the general one: it will fail for the NEXT checkpoint someone ships without a
recipe entry, which is the failure mode rather than any particular number.
"""
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import pilot_panel as P  # noqa: E402

TRAINING_DT_HZ = 30.0          # env.dt = 0.0333 s == 30.03 Hz
TRAINING_STALE_HORIZON_S = 0.5  # rl/ego_estimator.py EgoEstimatorCfg.stale_horizon_s


def _schema_default(key):
    """The panel's schema default -- what an UNPINNED knob resolves to after a model-pick reset."""
    for k in P.SCHEMA:
        if k["key"] == key:
            return k["default"]
    raise AssertionError(f"knob {key} not in SCHEMA")


def _resolved(recipe, key):
    """What the pilot actually gets: the recipe pin if there is one, else the schema default.

    A model-pick resets every key in ``_recipe_managed_keys()`` to its schema default and then
    overlays the chosen model's recipe -- so an ABSENT key is not a gap, it is the default. The
    legacy vpef*/vtrackA* lineages pin almost nothing and correctly land on 30 Hz that way; it was
    the _V1_RECIPE-derived lineages, which pinned 40.0 explicitly, that were wrong.
    """
    return recipe[key] if key in recipe else _schema_default(key)


@pytest.mark.parametrize("name,recipe", sorted(P.MODEL_DEFAULTS.items()))
def test_every_model_pick_lands_the_trained_loop_rate(name, recipe):
    """A model-pick may never hand the pilot a loop rate the policy was not trained at."""
    assert _resolved(recipe, "rate") == TRAINING_DT_HZ, (
        f"{name} resolves rate={_resolved(recipe, 'rate')}; training dt is {TRAINING_DT_HZ} Hz"
    )


@pytest.mark.parametrize("name,recipe", sorted(P.MODEL_DEFAULTS.items()))
def test_every_model_pick_lands_the_trained_stale_horizon(name, recipe):
    got = _resolved(recipe, "ego_stale_horizon")
    assert got == TRAINING_STALE_HORIZON_S, (
        f"{name} resolves ego_stale_horizon={got}; training uses {TRAINING_STALE_HORIZON_S}"
    )


def test_rate_stays_recipe_managed_so_a_model_pick_resets_it():
    """The fix is to pin it CORRECTLY, never to delete it: a knob in no recipe is never reset and
    rides across model switches at whatever the pilot last typed."""
    assert "rate" in P._recipe_managed_keys()
    assert "ego_stale_horizon" in P._recipe_managed_keys()


def test_the_rate_knob_default_and_help_agree_with_the_pin():
    """The 40.0 pin was caught because the knob's own help contradicted it. Keep them consistent so
    the next reader is not misled the same way."""
    assert _schema_default("rate") == TRAINING_DT_HZ
    helps = [k.get("help", "") for k in P.SCHEMA if k["key"] == "rate"]
    assert helps and "30" in helps[0], "the help text must keep naming the training dt"


def test_v20_has_a_recipe_entry():
    """v2.0 shipped with none; its six flights inherited the stale rate-40 pin and drifted on
    ego_pitch_clamp across the cohort."""
    assert "v20Vs0_actor.pth" in P.MODEL_DEFAULTS
    assert P.MODEL_DEFAULTS["v20Vs0_actor.pth"]["label"] == "v20pick_Vs0"


def test_every_shipped_checkpoint_in_the_panel_has_a_recipe():
    """THE GENERAL GUARD. Whatever checkpoints the panel offers, each must carry a recipe -- so this
    fails for the next model shipped without one, not merely for v20."""
    listed = {str(v) for v in (getattr(P, "CURRENT_CKPTS", None) or [])}
    missing = {c for c in listed if c.endswith(".pth") and c not in P.MODEL_DEFAULTS}
    assert not missing, f"checkpoints offered by the panel with no recipe entry: {sorted(missing)}"


def test_aim_offsets_stay_pinned_empty_at_the_base():
    """Unchanged by this commit, re-pinned here because it is the other half of the same contract:
    an aim offset must never ride across a model pick into a checkpoint it was not measured on."""
    assert P._V1_RECIPE["ego_aim_offsets"] == ""
    assert "ego_aim_offsets" in P._recipe_managed_keys()
