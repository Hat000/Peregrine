"""fly_rl wiring for the 2026-07-25 vertical-target package: --ego-fix-gain, --seeker-propagate-range,
and the meta.json provenance gap (deploy_profile was never recorded).

Both new knobs are DEFAULT-OFF -- fix_gain 1.0 is the historical snap, track_propagate_range False
leaves the tracked range on its EMA -- so the pins here are about the WIRING, not the behaviour (the
behaviour lives in test_ego_deploy_obs.py::test_fix_gain_* and
test_seeker_advance_prior_supersede.py::test_propagate_range_*):

  * argparse declares both with the OFF default, so a bare command line reproduces every flight to date;
  * the gain reaches EgoObsBuilderConfig and the flag reaches GateSeekerConfig;
  * ``seeker.propagate`` is handed the body-FRD velocity ONLY when the flag is set (so the ego loop's
    propagate call stays bearing-only otherwise);
  * meta.json records deploy_profile (BOTH paths -- 494 banked sessions cannot be split by profile
    because it was never written) plus both new knobs as FLOWN, so an A/B flight is self-describing.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402

_SRC = (_RL / "fly_rl.py").read_text(encoding="utf-8")


# -- argparse: both knobs exist and default to today's behaviour ---------------------------------
def test_argparse_defaults_reproduce_todays_behaviour():
    ns = fly_rl.build_parser().parse_args([])
    assert ns.ego_fix_gain == 1.0                # 1.0 == snap to every fix == the pre-change builder
    assert ns.seeker_propagate_range is False    # range stays frozen on its EMA


def test_argparse_accepts_the_training_gain_and_the_flag():
    ns = fly_rl.build_parser().parse_args(["--ego-fix-gain", "0.154", "--seeker-propagate-range"])
    assert ns.ego_fix_gain == 0.154              # K = 1/N_eff at the training N_eff ~ U[4,9] mean
    assert ns.seeker_propagate_range is True


# -- threading ----------------------------------------------------------------------------------
def test_fix_gain_reaches_the_obs_builder_config():
    """The gain is passed into EgoObsBuilderConfig(...) in _fly_ego, not silently dropped."""
    assert re.search(r'fix_gain=float\(getattr\(args,\s*"ego_fix_gain",\s*1\.0\)\)', _SRC)


def test_propagate_range_reaches_the_seeker_config_on_the_ego_path():
    """GateSeekerConfig gets track_propagate_range, guarded by _ego_path like its neighbours, so the
    classical (VQ1 / case-A) seeker construction is untouched."""
    i = _SRC.index('"track_propagate_range": bool(getattr(args, "seeker_propagate_range", False))')
    assert "if _ego_path else {}" in _SRC[i:i + 200]


def test_velocity_is_only_handed_to_propagate_when_the_flag_is_on():
    """seeker.propagate gets vel_frd=..., and the vector is built ONLY under the flag -- so with the
    flag off the call is the old bearing-only propagate with vel_frd=None."""
    assert "seeker.propagate(s.gyro_body, _prop_dt, vel_frd=_prop_vel_frd)" in _SRC
    i = _SRC.index("_prop_vel_frd = None")
    block = _SRC[i:_SRC.index("seeker.propagate(s.gyro_body", i)]
    assert 'getattr(args, "seeker_propagate_range", False)' in block
    assert "nav_state.velocity_ned is not None" in block
    # body FRD = R^T v_ned -- the SAME transform EgoObsBuilder.update applies before its FLU flip.
    assert "R_frd2ned.T @ np.asarray(nav_state.velocity_ned" in block


# -- meta.json provenance -------------------------------------------------------------------------
def test_meta_records_deploy_profile_on_both_paths():
    """deploy_profile sits in the COMMON add_meta block (outside the ego-only ** dict): it is not an
    ego flag, and the classical path needs it recorded too."""
    key = 'deploy_profile=str(getattr(args, "deploy_profile", ""))'
    assert key in _SRC
    ego_dict = _SRC.index('**({"ego_ckpt": str(args.ego_ckpt),')
    assert _SRC.index(key) < ego_dict, "deploy_profile must be recorded for BOTH paths"


def test_meta_records_both_new_knobs_as_flown():
    assert '"ego_fix_gain": float(getattr(args, "ego_fix_gain", 1.0))' in _SRC
    assert '"seeker_propagate_range": bool(getattr(args, "seeker_propagate_range", False))' in _SRC


def test_seeker_constants_record_the_range_propagation():
    """_meta_seeker_constants carries it too -- that dict is the canonical record of the seeker
    discipline a flight flew, and it must follow the ego-path/dataclass-default convention."""
    args_ego = SimpleNamespace(ego_ckpt="ckpts/x.pth", seeker_propagate_range=True)
    args_cls = SimpleNamespace(ego_ckpt=None)
    assert fly_rl._meta_seeker_constants(args_ego)["track_propagate_range"] is True
    assert fly_rl._meta_seeker_constants(args_cls)["track_propagate_range"] is False


def test_seeker_constants_default_off_when_the_flag_is_absent():
    """An args namespace built without the new flag (an older launcher) must not crash or flip it on."""
    args = SimpleNamespace(ego_ckpt="ckpts/x.pth")
    assert fly_rl._meta_seeker_constants(args)["track_propagate_range"] is False
