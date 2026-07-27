"""PER-GATE AIM OFFSET (racer.ego_obs.parse_aim_offsets + the EgoObsBuilder / fly_rl / panel wiring).

The repo form of the pilot's ShadowPC-only ``aim_off`` dodge for the two invisible obstacles
measured ~14.5 m short of gate 4 and gate 5.

Four layers:
  1. the grammar -- a mistyped dodge must ABORT, never silently fly as a no-op;
  2. the SIGN + FRAME contract, pinned against the deltas measured off the pilot's eight flown
     ``aim_off`` sessions (see parse_aim_offsets' docstring for the table);
  3. the range gate + the blast radius: only slot0, only the listed gate, nothing else in the obs;
  4. the wiring: DEFAULT-OFF is BYTE-IDENTICAL against the pre-change builder, the CLI flags exist
     with OFF defaults and reach the builder config + meta.json, and the panel exposes it WITHOUT
     letting it into a pinned recipe.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from racer import frames
from racer.contracts import GatePose
from racer.ego_obs import (EGO_OBS_DIM, EgoObsBuilder, EgoObsBuilderConfig,
                           parse_aim_offsets)

_ROOT = Path(__file__).resolve().parents[1]
_RL = _ROOT / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))


# =================================================================================================
# 1. grammar
# =================================================================================================
def test_parses_the_pilots_flown_spec():
    assert parse_aim_offsets("4:0,10;5:0,10") == {4: (0.0, 10.0), 5: (0.0, 10.0)}


def test_parses_the_gate1_probe_sweep_including_negatives():
    assert parse_aim_offsets("1:0,-3") == {1: (0.0, -3.0)}
    assert parse_aim_offsets("1:-3,0") == {1: (-3.0, 0.0)}


def test_blank_is_off():
    for blank in ("", "   ", None, ";", "  ;; "):
        assert parse_aim_offsets(blank) == {}


def test_whitespace_and_semicolons_both_separate():
    assert parse_aim_offsets("4:1,2 5:3,4") == parse_aim_offsets("4:1,2;5:3,4")


@pytest.mark.parametrize("bad", [
    "4",                # no ':'
    "4:10",             # one offset, not two
    "4:1,2,3",          # three offsets
    "x:1,2",            # non-numeric gate
    "4:a,2",            # non-numeric offset
    "-1:1,2",           # negative gate index
    "4:1,2;4:3,4",      # same gate twice -- which one wins is not a thing the pilot should guess
    "4:nan,2",          # non-finite
    "4:1,inf",
])
def test_malformed_specs_raise(bad):
    """A dodge that silently does nothing for a whole flight is worse than a pad abort."""
    with pytest.raises(ValueError):
        parse_aim_offsets(bad)


# =================================================================================================
# 2/3. the builder
# =================================================================================================
_CAM_PITCH_POSE_R = np.eye(3)


def _pose(r, lateral=0.0, vert=0.0, fid=1):
    rel_body = np.array([r, lateral, vert], dtype=np.float64)
    return GatePose(frame_id=fid, sim_time_ns=0, R_cam_gate=_CAM_PITCH_POSE_R,
                    t_cam_gate=frames.R_camera_from_body() @ rel_body, reproj_error_px=0.5)


def _drive(builder, n=90, gate_of=None):
    """A representative flight segment: closing on a laterally offset gate, with detection gaps
    and a gate advance at k=60 (the same driver the D1 suite uses)."""
    out = []
    r = 14.0
    for k in range(n):
        t_ns = int(k * (1e9 / 30))
        r = max(1.2, r - 0.25)
        gi = (0 if k < 60 else 1) if gate_of is None else gate_of(k)
        pose = _pose(r, lateral=0.5 - 0.01 * k, vert=0.1) if (k % 3 == 0 and k % 17 != 0) else None
        out.append(builder.update(
            sim_time_ns=t_ns, gate_index=gi, R_frd2ned=np.eye(3),
            vel_ned=np.array([7.0, 0.4 + 0.01 * k, -0.2]),
            gyro_frd=np.array([0.05, -0.02, 0.3]),
            pose=pose, last_normed_thrust=1.0))
    return out


def _pose_flu(fwd, left=0.0, up=0.0, fid=1):
    """A GatePose whose decoded slot0 lever is EXACTLY body FLU [fwd, left, up].

    Inverts both halves of ``rel_pos_body_frd_from_gatepose`` -- the mount rotation AND the metric
    boresight ``vert_offset_m`` (-0.25 m, FRD +Z down) -- so the exact-value assertions below test
    the aim offset and nothing else. (``_pose`` above skips the metric half, which is fine for the
    differential drives but puts a silent +0.25 m on ``up``.)"""
    frd = np.array([fwd, -left, -up], dtype=np.float64)
    t = frames.R_camera_from_body() @ (frd - np.array([0.0, 0.0, frames.BORESIGHT.vert_offset_m]))
    return GatePose(frame_id=fid, sim_time_ns=0, R_cam_gate=_CAM_PITCH_POSE_R,
                    t_cam_gate=t, reproj_error_px=0.5)


def _one_tick(cfg, *, rel_body, gate_index=0, thrust=1.0):
    """Feed ONE fresh fix at a known body-FLU lever and return (obs, last_diag)."""
    b = EgoObsBuilder(cfg)
    obs = b.update(sim_time_ns=0, gate_index=gate_index, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3),
                   pose=_pose_flu(*rel_body), last_normed_thrust=thrust)
    return obs, b.last_diag


def test_the_flown_sign_convention_lateral_is_right_and_vertical_is_up():
    """PINS the convention recovered from the pilot's logs: +lateral moves the perceived gate
    RIGHT (body FLU left DECREASES) and +vertical moves it UP (body FLU up INCREASES).

    Measured release deltas, aim_off -> None, body FLU [fwd, left, up]:
        [0.0, -3.0] -> D_up +2.916      [0.0, +3.0] -> D_up -3.039
        [-3.0, 0.0] -> D_left -3.026    [0.0, +10.0] -> D_up -9.68 .. -10.26 (x4)
    i.e. applying [lat, vert] gives left -= lat and up += vert."""
    rel = (20.0, 1.0, 2.0)
    off, _ = _one_tick(EgoObsBuilderConfig(), rel_body=rel)
    on, diag = _one_tick(EgoObsBuilderConfig(aim_offsets={0: (-3.0, 10.0)}, aim_release_m=0.0),
                         rel_body=rel)
    # the diagnostic lever (what ego_obs.jsonl logs, offset baked in -- the pilot's own shape)
    np.testing.assert_allclose(diag["rel_flu"], [20.0, 1.0 - (-3.0), 2.0 + 10.0], atol=1e-9)
    # and in the VIRTUAL-FLIPPED obs the policy actually reads: obs[11:14] = [-fwd, -left, +up]
    np.testing.assert_allclose(on[11] - off[11], 0.0, atol=1e-5)
    np.testing.assert_allclose(on[12] - off[12], -3.0, atol=1e-5)   # obs[12] += lateral
    np.testing.assert_allclose(on[13] - off[13], +10.0, atol=1e-5)  # obs[13] += vertical


def test_the_offset_is_body_frame_and_leaves_the_range_channel_alone():
    """The DECISIVE discriminator against the GateSeeker._valid_poses camera-frame injection.

    ``frames.R_camera_from_body()`` maps a camera +Y (down) offset b to body FLU
    (+0.342b, 0, -0.940b) -- a 10 m vertical there would ALSO move the perceived RANGE by 3.42 m.
    The pilot's flown releases put |D_fwd| <= 0.28 m (one tick of ownship motion) and the FULL
    magnitude on UP, so his lever is body-frame. This pins both halves: our forward channel does
    not move, and the camera-frame leak factor is what it was when that was measured (a mount
    change would surface here, not in a flight)."""
    leak = (frames.R_camera_from_body().T @ np.array([0.0, 1.0, 0.0]))
    leak_flu = leak * np.array([1.0, -1.0, -1.0])
    assert leak_flu[0] == pytest.approx(0.342, abs=0.005), "camera mount moved; re-derive the frame claim"
    assert leak_flu[2] == pytest.approx(-0.940, abs=0.005)

    rel = (20.0, 0.0, 0.0)
    off, _ = _one_tick(EgoObsBuilderConfig(), rel_body=rel)
    on, _ = _one_tick(EgoObsBuilderConfig(aim_offsets={0: (0.0, 10.0)}, aim_release_m=0.0),
                      rel_body=rel)
    assert abs(on[11] - off[11]) < 1e-5, "a 10 m vertical must not move the perceived RANGE"
    assert on[13] - off[13] == pytest.approx(10.0, abs=1e-5)


def test_only_the_listed_gate_index_is_offset():
    rel = (20.0, 0.0, 0.0)
    cfg = dict(aim_offsets={5: (0.0, 10.0)}, aim_release_m=0.0)
    on5, _ = _one_tick(EgoObsBuilderConfig(**cfg), rel_body=rel, gate_index=5)
    on4, _ = _one_tick(EgoObsBuilderConfig(**cfg), rel_body=rel, gate_index=4)
    base4, _ = _one_tick(EgoObsBuilderConfig(), rel_body=rel, gate_index=4)
    assert on5[13] == pytest.approx(10.0, abs=1e-5)
    assert on4.tobytes() == base4.tobytes()


def test_the_range_gate_releases_below_the_threshold():
    """Hard step by default (what flew): full offset above ``aim_release_m``, nothing at or below.
    Range is read off the HELD belief, offset EXCLUDED, so the offset cannot hold itself open."""
    cfg = EgoObsBuilderConfig(aim_offsets={0: (0.0, 10.0)}, aim_release_m=12.0)
    above, d_above = _one_tick(cfg, rel_body=(12.5, 0.0, 0.0))
    below, d_below = _one_tick(cfg, rel_body=(11.5, 0.0, 0.0))
    assert d_above["aim_off"] == [0.0, 10.0]
    assert d_below["aim_off"] is None
    base, _ = _one_tick(EgoObsBuilderConfig(), rel_body=(11.5, 0.0, 0.0))
    assert below.tobytes() == base.tobytes()
    assert above[13] == pytest.approx(10.0, abs=1e-5)
    # a +10 m offset inflates |rel| by ~4 m at 12 m: if the gate read the OFFSET lever it would
    # still be armed here. It must not be.
    assert np.linalg.norm(np.array([11.5, 0.0, 10.0])) > 12.0


def test_the_fade_is_linear_and_off_by_default():
    assert EgoObsBuilderConfig().aim_fade_m == 0.0
    cfg = EgoObsBuilderConfig(aim_offsets={0: (0.0, 10.0)}, aim_release_m=10.0, aim_fade_m=10.0)
    for rng, want in ((20.0, 10.0), (15.0, 5.0), (12.0, 2.0), (10.0, None), (25.0, 10.0)):
        _, d = _one_tick(cfg, rel_body=(rng, 0.0, 0.0))
        if want is None:
            assert d["aim_off"] is None
        else:
            assert d["aim_off"][1] == pytest.approx(want, abs=1e-6)


def test_no_belief_means_no_offset_never_a_fabricated_lever():
    """Before the first fix the slot is blank. An aim offset SHIFTS a lever; it must never
    conjure one (that would feed the policy a gate 10 m up that vision never saw)."""
    b = EgoObsBuilder(EgoObsBuilderConfig(aim_offsets={0: (0.0, 10.0)}, aim_release_m=0.0))
    obs = b.update(sim_time_ns=0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                   gyro_frd=np.zeros(3), pose=None, last_normed_thrust=0.0)
    assert b.last_diag["aim_off"] is None
    assert b.last_diag["rel_flu"] is None
    np.testing.assert_array_equal(obs[11:14], 0.0)


def test_a_masked_slot_stays_exactly_zero():
    """The offset is applied BEFORE the confidence mask, exactly as the pilot's flights show
    (his logs carry an offset rel_flu while obs[11:14] is all zeros through a blackout)."""
    cfg = EgoObsBuilderConfig(aim_offsets={0: (0.0, 10.0)}, aim_release_m=0.0)
    b = EgoObsBuilder(cfg)
    b.update(sim_time_ns=0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
             gyro_frd=np.zeros(3), pose=_pose_flu(20.0), last_normed_thrust=1.0)
    obs = b.update(sim_time_ns=int(2.0e9), gate_index=0, R_frd2ned=np.eye(3),  # 2 s -> conf 0
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=None, last_normed_thrust=1.0)
    np.testing.assert_array_equal(obs[11:14], 0.0)
    assert obs[14] == 0.0
    assert b.last_diag["rel_flu"][2] > 9.0        # the LOGGED lever still carries it (pilot parity)


def test_arming_moves_only_slot0_rel_pos():
    """Blast radius: nothing outside obs[11:14] may move -- in particular NOT slot1 (obs[16:19]),
    which the pilot's flights leave continuous across every release, and NOT the coarse sector
    (obs[9:11]), which is latched off the honest first-fix lever."""
    def gate_of(k):
        return 0 if k < 60 else 1
    off = _drive(EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True)), gate_of=gate_of)
    on = _drive(EgoObsBuilder(EgoObsBuilderConfig(
        slot1_enabled=True, aim_offsets={0: (1.5, 4.0), 1: (0.0, 4.0)}, aim_release_m=0.0)),
        gate_of=gate_of)
    moved = np.zeros(EGO_OBS_DIM, dtype=bool)
    for x, y in zip(off, on):
        moved |= (x != y)
        assert np.all(np.isfinite(y))
    assert moved[11:14].any(), "arming the offset changed nothing at all"
    assert not moved[:11].any(), f"channels before slot0 moved: {np.where(moved[:11])[0]}"
    assert not moved[14:].any(), f"channels after slot0 rel moved: {np.where(moved[14:])[0] + 14}"


def test_the_held_belief_is_not_contaminated():
    """The offset is applied at OUTPUT, per tick -- it is NOT baked into the held belief. That is
    what the pilot's logs show: dropping the flag removes the full 10 m from a COASTED lever in a
    single tick, which a belief-baked offset could not do. Consequence: propagation, the fix
    blend, the sector latch and slot0_hint_frd all keep running on the honest lever."""
    cfg = EgoObsBuilderConfig(aim_offsets={0: (0.0, 10.0)}, aim_release_m=0.0)
    b = EgoObsBuilder(cfg)
    b.update(sim_time_ns=0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
             gyro_frd=np.zeros(3), pose=_pose_flu(20.0), last_normed_thrust=1.0)
    np.testing.assert_allclose(b._rel_flu[0], [20.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(b.slot0_hint_frd(), [20.0, 0.0, 0.0], atol=1e-9)
    assert b.last_diag["rel_flu"][2] == pytest.approx(10.0, abs=1e-9)


def test_the_sector_latch_uses_the_honest_lever():
    """obs[9:11] is the FIRST-FIX elevation bucket. A +10 m aim offset must not flip it -- the
    sector is a course prior, not an aim."""
    rel = (20.0, 0.0, 0.0)
    _, d_off = _one_tick(EgoObsBuilderConfig(sector_mode="auto"), rel_body=rel)
    _, d_on = _one_tick(EgoObsBuilderConfig(sector_mode="auto", aim_offsets={0: (0.0, 10.0)},
                                            aim_release_m=0.0), rel_body=rel)
    assert d_on["sector"] == d_off["sector"]


# =================================================================================================
# 4. wiring: DEFAULT-OFF is BYTE-IDENTICAL
# =================================================================================================
def _old_builder_class(tmp_path):
    """Import the LAST PRE-AIM-OFFSET revision of ego_obs.py straight out of git.

    Walks the file's history back to the newest revision that does not yet know about
    ``aim_offsets``, so this keeps comparing against the real pre-change builder no matter how
    many commits land on top (pinning HEAD would compare the new file with itself)."""
    import importlib.util
    log = subprocess.run(["git", "log", "--format=%H", "--", "src/racer/ego_obs.py"],
                         cwd=str(_ROOT), capture_output=True, text=True)
    if log.returncode != 0:
        pytest.skip("git history for src/racer/ego_obs.py unavailable")
    text = None
    for sha in log.stdout.split():
        blob = subprocess.run(["git", "show", f"{sha}:src/racer/ego_obs.py"], cwd=str(_ROOT),
                              capture_output=True, text=True)
        if blob.returncode != 0 or "class EgoObsBuilder" not in blob.stdout:
            continue
        if "aim_offsets" not in blob.stdout:
            text = blob.stdout
            break
    if text is None:
        pytest.skip("no pre-aim-offset revision of src/racer/ego_obs.py found in history")
    p = tmp_path / "ego_obs_pre_aim.py"
    p.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ego_obs_pre_aim", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ego_obs_pre_aim"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_off_is_byte_identical_to_the_pre_change_builder(tmp_path):
    """The whole point: with --ego-aim-offsets unset, every emitted obs must be bit-for-bit what
    the previous build produced -- tick for tick over a replayed flight segment."""
    old = _old_builder_class(tmp_path)
    a = _drive(EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True)))
    b = _drive(old.EgoObsBuilder(old.EgoObsBuilderConfig(slot1_enabled=True)))
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.dtype == y.dtype == np.float32
        assert x.tobytes() == y.tobytes(), f"tick {i} diverged with the flag at its default"


def test_default_off_leaves_last_diag_untouched(tmp_path):
    """No new key when the knob is unset -- so a default flight's ego_obs.jsonl is unchanged and
    every existing analysis script keeps parsing it identically."""
    old = _old_builder_class(tmp_path)
    nb, ob = EgoObsBuilder(EgoObsBuilderConfig()), old.EgoObsBuilder(old.EgoObsBuilderConfig())
    _drive(nb, 40)
    _drive(ob, 40)
    assert set(nb.last_diag) == set(ob.last_diag)
    assert "aim_off" not in nb.last_diag


def test_configured_but_inactive_still_logs_a_null_aim_off():
    """Once configured the key is present on EVERY tick (null while not applying) -- the pilot's
    own shape, and what scripts/d1_velocity_replay.py + scripts/replay_fix_gain.py already cut on
    (both treat a missing/None/zero aim_off as clean)."""
    b = EgoObsBuilder(EgoObsBuilderConfig(aim_offsets={5: (0.0, 10.0)}))
    _drive(b, 40)                                     # gate 0/1 only -- never gate 5
    assert "aim_off" in b.last_diag and b.last_diag["aim_off"] is None


def test_cli_flags_exist_with_off_defaults():
    import fly_rl
    ap = fly_rl.build_parser()
    assert ap.parse_args([]).ego_aim_offsets == ""
    assert ap.parse_args([]).ego_aim_release == 12.0
    assert ap.parse_args([]).ego_aim_fade == 0.0
    a = ap.parse_args(["--ego-aim-offsets", "4:0,10;5:0,10", "--ego-aim-release", "14"])
    assert parse_aim_offsets(a.ego_aim_offsets) == {4: (0.0, 10.0), 5: (0.0, 10.0)}
    assert a.ego_aim_release == 14.0


def test_the_flag_reaches_the_builder_and_meta_json():
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    assert "aim_offsets=_aim_offsets," in src
    assert 'aim_release_m=float(getattr(args, "ego_aim_release", 12.0))' in src
    assert '"ego_aim_offsets": str(getattr(args, "ego_aim_offsets", "") or "")' in src
    assert '**({"aim_off": d["aim_off"]} if "aim_off" in d else {})' in src


def test_panel_exposes_it_next_to_z_bias_but_never_pins_it_in_a_recipe():
    """It sits with the other seeker/aim knobs, and it is NOT in any _V*_RECIPE: recipes are
    release contracts, and a silent ride-in already corrupted a v16 batch once."""
    sys.path.insert(0, str(_ROOT / "tools"))
    import pilot_panel as P
    spec = P.BY_KEY["ego_aim_offsets"]
    assert spec["default"] == ""                       # blank => build_cmd omits the flag entirely
    assert spec["group"] == P.BY_KEY["ego_gate_z_bias"]["group"]
    for key in ("ego_aim_offsets", "ego_aim_release", "ego_aim_fade"):
        assert key in P.BY_KEY
        for name in dir(P):
            if name.startswith("_V") and name.endswith("_RECIPE"):
                assert key not in getattr(P, name), f"{key} must not be pinned in {name}"
    argv, _ = P.build_cmd({})
    assert "--ego-aim-offsets" not in argv             # default launch is byte-identical
