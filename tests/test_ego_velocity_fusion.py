"""D1 vision-referenced LATERAL velocity (racer.ego_velocity + its EgoObsBuilder / fly_rl wiring).

Three layers:
  1. the estimator itself, on synthetic kinematics where the true DR error is known;
  2. every DEGENERATE case (no track, stale/coasted track, gate advance, blackout, oversized
     tick gap, NaN/Inf, out-of-band range/baseline) -- none may emit NaN/Inf or a wild jump;
  3. the wiring: DEFAULT-OFF is BYTE-IDENTICAL against the pre-change builder, the CLI flag
     exists with the OFF default, and it reaches both the builder config and meta.json.
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from racer import frames
from racer.contracts import GatePose
from racer.ego_obs import EGO_OBS_DIM, EgoObsBuilder, EgoObsBuilderConfig
from racer.ego_velocity import LateralVelocityFuser, VelocityFusionConfig, _rot_body

_ROOT = Path(__file__).resolve().parents[1]
_RL = _ROOT / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))


# =================================================================================================
# 1. the estimator on synthetic kinematics
# =================================================================================================
def _run(fuser, *, v_true, v_dr, w=(0.0, 0.0, 0.0), r0=(10.0, 0.0, 0.0), dt=1.0 / 30.0,
         n=120, fix_every=3, noise=0.0, seed=0):
    """Fly a straight line past a world-FIXED gate; feed the fuser the DR velocity and the
    exact lever the geometry implies. Returns the fuser."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r0, dtype=np.float64)
    v_true = np.asarray(v_true, dtype=np.float64)
    v_dr = np.asarray(v_dr, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    for k in range(n):
        dR = _rot_body(w, dt)
        r = dR @ r - v_true * dt            # exact body-frame kinematics of a fixed landmark
        fuser.propagate(w, v_dr, dt)
        if k % fix_every == 0:
            meas = r + (rng.normal(0.0, noise, 3) if noise > 0 else 0.0)
            fuser.on_fix(meas)
    return fuser


def test_recovers_a_pure_lateral_dr_error():
    """DR says 'no lateral motion' while the drone really slides LEFT at 1 m/s: the correction
    must recover ~+1 m/s on the left axis.

    The forward channel picks up a small share of it, and that is GEOMETRY, not a bug: the
    correction lives in the plane perpendicular to the line of sight, so once the drift has
    pushed the gate off boresight by an angle ``th`` a purely-left error appears as
    (-sin th cos th, cos^2 th) there. Over this run th reaches ~0.12 rad, bounding the forward
    share at ~0.12 -- asserted below so a real leak (a full-magnitude forward correction)
    still fails."""
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[6.0, 1.0, 0.0], v_dr=[6.0, 0.0, 0.0], r0=(22.0, 0.0, 0.0), n=45)
    c = f.correction()
    assert f.n_updates > 5
    assert c[1] == pytest.approx(1.0, abs=0.12)       # the lateral error is recovered
    assert abs(c[0]) < 0.2 and abs(c[0]) < 0.3 * abs(c[1])   # forward stays a geometric residue


def test_sign_is_right_for_the_opposite_drift():
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, -1.5, 0.0], v_dr=[8.0, 0.0, 0.0])
    assert f.correction()[1] == pytest.approx(-1.5, abs=0.15)


def test_vertical_error_is_recovered_too_when_the_gate_is_ahead():
    """With the gate on the forward axis, UP is also LOS-perpendicular and observable."""
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, 0.0, 0.8], v_dr=[8.0, 0.0, 0.0])
    assert f.correction()[2] == pytest.approx(0.8, abs=0.1)


def test_along_los_error_is_NOT_invented():
    """A pure CLOSING-speed error is unobservable from bearing. The estimator must not pretend
    to see it -- the emitted correction stays ~0 rather than guessing."""
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[9.0, 0.0, 0.0], v_dr=[6.0, 0.0, 0.0])
    assert np.linalg.norm(f.correction()) < 0.25      # 3 m/s of range-rate error -> ~nothing


def test_zero_error_stays_zero():
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, 1.2, -0.4], v_dr=[8.0, 1.2, -0.4])
    assert np.linalg.norm(f.correction()) < 0.05


def test_survives_body_rotation():
    """The lever is de-rotated with the gyro, so a yawing/rolling drone must not fabricate a
    lateral velocity out of its own rotation -- the failure mode that produced the original
    '53% sign disagreement' reading."""
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, 0.0, 0.0], v_dr=[8.0, 0.0, 0.0], w=[0.0, 0.0, 0.6], n=90)
    assert np.linalg.norm(f.correction()) < 0.15

    f2 = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f2, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0], w=[0.0, 0.0, 0.6], n=90)
    assert f2.correction()[1] == pytest.approx(1.0, abs=0.2)


def test_measurement_noise_is_averaged_not_tracked():
    """With a low gain the estimate must sit near the true bias, not chase per-fix noise."""
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.10))
    _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0], noise=0.05, n=400, seed=3)
    assert f.correction()[1] == pytest.approx(1.0, abs=0.25)
    assert np.linalg.norm(f.correction()) < 1.6


def test_gain_zero_is_inert():
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.0))
    _run(f, v_true=[8.0, 2.0, 0.0], v_dr=[8.0, 0.0, 0.0])
    assert np.all(f.correction() == 0.0)
    np.testing.assert_array_equal(f.correct([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])


# =================================================================================================
# 2. degenerate cases -- never NaN, never wild
# =================================================================================================
def test_no_track_ever_emits_zero():
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    for _ in range(50):
        f.propagate([0.0, 0.0, 0.1], [7.0, 0.5, 0.0], 1 / 30)
    assert np.all(f.correction() == 0.0)
    np.testing.assert_allclose(f.correct([7.0, 0.5, 0.0]), [7.0, 0.5, 0.0])


def test_stale_or_coasted_track_does_not_update():
    """A coasted seeker tick supplies NO pose, so ``on_fix`` is simply never called: the
    correction holds instead of drifting."""
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0])
    held = f.correction().copy()
    n0 = f.n_updates
    for _ in range(10):                                # ~0.33 s of coasting, inside hold_s
        f.propagate([0.0, 0.0, 0.0], [8.0, 0.0, 0.0], 1 / 30)
    assert f.n_updates == n0
    np.testing.assert_allclose(f.correction(), held, atol=1e-9)


def test_blackout_holds_then_decays_to_zero():
    cfg = VelocityFusionConfig(gain=0.5, hold_s=0.5, decay_tau_s=0.4)
    f = LateralVelocityFuser(cfg)
    _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0])
    held = np.linalg.norm(f.correction())
    assert held > 0.5
    for _ in range(int(0.45 * 30)):                    # still inside the hold
        f.propagate([0.0, 0.0, 0.0], [8.0, 0.0, 0.0], 1 / 30)
    assert np.linalg.norm(f.correction()) == pytest.approx(held, rel=1e-6)
    for _ in range(int(3.0 * 30)):                     # long blackout -> back to raw DR
        f.propagate([0.0, 0.0, 0.0], [8.0, 0.0, 0.0], 1 / 30)
    assert np.linalg.norm(f.correction()) < 0.05


def test_gate_advance_drops_the_window_but_keeps_the_bias():
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0])
    before = f.correction().copy()
    f.on_gate_change()
    np.testing.assert_allclose(f.correction(), before, atol=1e-12)
    assert f._anchor is None


def test_oversized_tick_gap_drops_the_window():
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0], n=30)
    f.propagate([0.0, 0.0, 0.0], [8.0, 0.0, 0.0], 0.9)   # > max_tick_dt_s
    assert f._anchor is None
    assert np.all(np.isfinite(f.correction()))


def test_out_of_band_range_is_rejected():
    cfg = VelocityFusionConfig(gain=0.5)
    for r in (0.4, 60.0):                                # too close (gate fills frame) / too far
        f = LateralVelocityFuser(cfg)
        _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0], r0=(r, 0.0, 0.0), n=20, dt=1 / 300)
        assert f.n_updates == 0
        assert np.all(f.correction() == 0.0)


def test_nonfinite_inputs_never_latch():
    f = LateralVelocityFuser(VelocityFusionConfig(gain=0.5))
    _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0])
    good = f.correction().copy()
    f.propagate([np.nan, 0.0, 0.0], [8.0, 0.0, 0.0], 1 / 30)
    f.propagate([0.0, 0.0, 0.0], [np.inf, 0.0, 0.0], 1 / 30)
    f.on_fix([np.nan, 1.0, 2.0])
    f.on_fix([np.inf, np.inf, np.inf])
    assert np.all(np.isfinite(f.correction()))
    np.testing.assert_allclose(f.correction(), good, atol=1e-9)
    out = f.correct([np.nan, 1.0, 2.0])
    assert out[1] == 1.0 and np.isnan(out[0])            # input returned unchanged, not garbage


def test_bias_is_hard_clamped():
    cfg = VelocityFusionConfig(gain=1.0, max_bias_mps=0.5, innov_clip_mps=99.0)
    f = LateralVelocityFuser(cfg)
    _run(f, v_true=[8.0, 6.0, 0.0], v_dr=[8.0, 0.0, 0.0])
    assert np.linalg.norm(f.correction()) <= 0.5 + 1e-9


def test_innovation_clip_bounds_a_single_outlier():
    cfg = VelocityFusionConfig(gain=1.0, innov_clip_mps=0.3)
    f = LateralVelocityFuser(cfg)
    _run(f, v_true=[8.0, 0.0, 0.0], v_dr=[8.0, 0.0, 0.0], n=60)
    f.on_fix([9.0, 40.0, 0.0])                           # a teleport-class wrong pose
    assert np.linalg.norm(f.correction()) <= 0.3 + 1e-9


def test_baseline_band_rejects_too_short_and_too_long_windows():
    cfg = VelocityFusionConfig(gain=1.0, min_baseline_s=0.35, max_baseline_s=1.0)
    f = LateralVelocityFuser(cfg)
    _run(f, v_true=[8.0, 1.0, 0.0], v_dr=[8.0, 0.0, 0.0], dt=1 / 300, fix_every=1, n=30)
    assert f.n_updates == 0                              # every window shorter than min_baseline_s


# =================================================================================================
# 3. wiring: DEFAULT-OFF is BYTE-IDENTICAL
# =================================================================================================
_CAM_PITCH_POSE_R = np.eye(3)


def _pose(r, lateral=0.0, vert=0.0, fid=1):
    rel_body = np.array([r, lateral, vert], dtype=np.float64)
    return GatePose(frame_id=fid, sim_time_ns=0, R_cam_gate=_CAM_PITCH_POSE_R,
                    t_cam_gate=frames.R_camera_from_body() @ rel_body, reproj_error_px=0.5)


def _drive(builder, n=90):
    """A representative flight segment: closing on a laterally offset gate, rolling, with
    detection gaps and a gate advance."""
    out = []
    r = 14.0
    for k in range(n):
        t_ns = int(k * (1e9 / 30))
        r = max(1.2, r - 0.25)
        gi = 0 if k < 60 else 1
        pose = _pose(r, lateral=0.5 - 0.01 * k, vert=0.1) if (k % 3 == 0 and k % 17 != 0) else None
        out.append(builder.update(
            sim_time_ns=t_ns, gate_index=gi, R_frd2ned=np.eye(3),
            vel_ned=np.array([7.0, 0.4 + 0.01 * k, -0.2]),
            gyro_frd=np.array([0.05, -0.02, 0.3]),
            pose=pose, last_normed_thrust=1.0))
    return out


_OLD_MODULE_SRC = None


def _old_builder_class(tmp_path):
    """Import the PRE-CHANGE ego_obs.py straight out of git HEAD and return its builder class."""
    import importlib.util
    src = subprocess.run(["git", "show", "HEAD:src/racer/ego_obs.py"], cwd=str(_ROOT),
                         capture_output=True, text=True)
    if src.returncode != 0 or "class EgoObsBuilder" not in src.stdout:
        pytest.skip("git HEAD copy of src/racer/ego_obs.py unavailable")
    p = tmp_path / "ego_obs_head.py"
    p.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ego_obs_head", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ego_obs_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_off_is_byte_identical_to_the_pre_change_builder(tmp_path):
    """The whole point: with --ego-vel-fuse unset, every emitted obs must be bit-for-bit what
    the previous build produced (the fuser is not even constructed)."""
    old = _old_builder_class(tmp_path)
    a = _drive(EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True)))
    b = _drive(old.EgoObsBuilder(old.EgoObsBuilderConfig(slot1_enabled=True)))
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.dtype == y.dtype == np.float32
        assert x.tobytes() == y.tobytes(), f"tick {i} diverged with the flag at its default"


def test_default_off_leaves_last_diag_untouched(tmp_path):
    old = _old_builder_class(tmp_path)
    nb, ob = EgoObsBuilder(EgoObsBuilderConfig()), old.EgoObsBuilder(old.EgoObsBuilderConfig())
    _drive(nb, 40)
    _drive(ob, 40)
    assert set(nb.last_diag) == set(ob.last_diag)
    assert "vfuse" not in nb.last_diag


def test_arming_the_gain_moves_only_obs_0_to_3():
    off = _drive(EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True)))
    on = _drive(EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True, vel_fuse_gain=0.3)))
    moved = np.zeros(EGO_OBS_DIM, dtype=bool)
    for x, y in zip(off, on):
        moved |= (x != y)
        assert np.all(np.isfinite(y))
    assert moved[0:3].any(), "arming the gain changed nothing at all"
    assert not moved[3:].any(), f"channels outside obs[0:3] moved: {np.where(moved[3:])[0] + 3}"


def test_armed_builder_exposes_diagnostics():
    b = EgoObsBuilder(EgoObsBuilderConfig(vel_fuse_gain=0.3))
    _drive(b, 60)
    d = b.last_diag.get("vfuse")
    assert d is not None
    assert set(d) >= {"bias", "corr", "baseline_s", "anchored", "age_s", "n_upd", "n_clip"}
    assert all(np.isfinite(d["corr"]))


def test_builder_reset_clears_the_fuser():
    b = EgoObsBuilder(EgoObsBuilderConfig(vel_fuse_gain=0.5))
    _drive(b, 60)
    b.reset()
    assert np.all(b._vfuse.bias == 0.0)
    assert b._vfuse.n_updates == 0


# -- fly_rl wiring --------------------------------------------------------------------------------
def test_cli_flag_defaults_off_and_parses():
    import fly_rl
    assert fly_rl.build_parser().parse_args([]).ego_vel_fuse == 0.0
    assert fly_rl.build_parser().parse_args(["--ego-vel-fuse", "0.1"]).ego_vel_fuse == 0.1


def test_flag_reaches_the_builder_config_and_meta():
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    assert 'vel_fuse_gain=float(getattr(args, "ego_vel_fuse", 0.0))' in src
    assert '"ego_vel_fuse": float(getattr(args, "ego_vel_fuse", 0.0))' in src
