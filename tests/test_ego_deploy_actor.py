"""EGO deploy actor path — hardcoded action bounds, the policy_step pipeline, checkpoint smoke.

Pins the fly_ego action contract [ego-deploy 2026-07-09]:
  * ego bounds are HARDCODED thrust [0, 3.765] / rates +-3.14 rad/s (the sbatch override
    dynamics.controller.max_normed_thrust=3.765; the ego launcher writes NO sidecar, so the
    generic loader's legacy [0,5] fallback would overdrive thrust ~33%);
  * deterministic deploy action = tanh(mu) -> rescale_action onto those bounds -> virtual
    pi-flip -> FLU->FRD -> hover-scaled collective (policy_step, reused unchanged);
  * the obs[8] feedback value is the RESCALED normed_thrust in g-units (policy_step's third
    return), NOT the [0,1] wire collective and NOT tanh(mu);
  * the vczext checkpoint strict-loads into the width-inferred _ActorMean(21) and produces
    finite, in-bounds actions on synthetic approach obs (skipped when the gitignored .pth is
    absent — pull it per the ckpt logistics note);
  * directional sanity is SOFT: the commanded FRD rates for a gate offset left vs right are
    PRINTED for human review (run with `pytest -s`), never asserted.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402
from racer.ego_obs import EGO_OBS_DIM, EgoObsBuilder, EgoObsBuilderConfig  # noqa: E402

_CKPT = _RL / "checkpoints" / "vczext_final_actor.pth"


@pytest.fixture()
def ego_bounds():
    """Apply the ego action bounds; RESTORE the module globals afterwards (they are mutated in
    place, exactly like the sidecar path — leaking them would poison other tests)."""
    amin, amax = fly_rl._ACT_MIN.copy(), fly_rl._ACT_MAX.copy()
    fly_rl._apply_ego_action_bounds()
    yield
    fly_rl._ACT_MIN[:] = amin
    fly_rl._ACT_MAX[:] = amax


class _StubActor(torch.nn.Module):
    """Returns a fixed pre-tanh mean regardless of obs (pipeline-math isolation)."""

    def __init__(self, mean4):
        super().__init__()
        self.mean4 = torch.as_tensor(mean4, dtype=torch.float32)

    def forward(self, x):
        return self.mean4.expand(x.shape[0], 4)


def test_ego_bounds_hardcoded(ego_bounds):
    np.testing.assert_allclose(fly_rl._ACT_MIN, [0.0, -3.14, -3.14, -3.14])
    np.testing.assert_allclose(fly_rl._ACT_MAX, [3.765, 3.14, 3.14, 3.14])


def test_rail_actions_map_to_trained_bounds(ego_bounds):
    """mu at the rails: tanh -> +-1 -> exactly the trained bounds after rescale (never [0,5])."""
    obs = np.zeros(EGO_OBS_DIM, dtype=np.float32)
    rate_frd, coll, normed = fly_rl.policy_step(
        _StubActor([50.0, 50.0, 50.0, 50.0]), obs, virtual_flip=False)
    assert normed == pytest.approx(3.765, abs=1e-4)             # thrust rail = 3.765 g-units
    # FLU (3.14,3.14,3.14) -> FRD via [1,-1,1]
    np.testing.assert_allclose(rate_frd, [3.14, -3.14, 3.14], atol=1e-4)
    assert coll == pytest.approx(1.0, abs=1e-4)                 # 3.765*0.2656 = 0.999984
    rate_frd, coll, normed = fly_rl.policy_step(
        _StubActor([-50.0, -50.0, -50.0, -50.0]), obs, virtual_flip=False)
    assert normed == pytest.approx(0.0, abs=1e-4)               # thrust floor = 0
    np.testing.assert_allclose(rate_frd, [-3.14, 3.14, -3.14], atol=1e-4)
    assert coll == pytest.approx(0.0, abs=1e-4)


def test_obs8_feedback_is_rescaled_normed_thrust(ego_bounds):
    """mu=0 -> tanh=0 -> act[0] = midpoint 1.8825 g-units. The feedback value (third return)
    must be exactly that — NOT the [0,1] collective (0.5) and NOT tanh(mu) (0.0)."""
    obs = np.zeros(EGO_OBS_DIM, dtype=np.float32)
    rate_frd, coll, normed = fly_rl.policy_step(_StubActor([0.0, 0.0, 0.0, 0.0]), obs,
                                                virtual_flip=False)
    mid = 0.0 + (3.765 - 0.0) * (np.tanh(0.0) + 1.0) / 2.0
    assert normed == pytest.approx(mid, abs=1e-6)               # 1.8825
    assert coll == pytest.approx(mid * fly_rl._HOVER_THRUST, abs=1e-6)
    assert normed != pytest.approx(coll)                        # the two channels must differ
    np.testing.assert_allclose(rate_frd, 0.0, atol=1e-7)


def test_virtual_flip_action_unflip(ego_bounds):
    """virtual_flip: rate_flu -> Rz(pi) diag(-1,-1,1) -> FLU->FRD [1,-1,1]; net wire map is
    (-x, +y, z) of the unflipped command. Pinned so the obs-side flip is never applied twice."""
    obs = np.zeros(EGO_OBS_DIM, dtype=np.float32)
    mu = [0.0, 0.3, -0.5, 0.7]
    r_noflip, _, _ = fly_rl.policy_step(_StubActor(mu), obs, virtual_flip=False)
    r_flip, _, _ = fly_rl.policy_step(_StubActor(mu), obs, virtual_flip=True)
    np.testing.assert_allclose(r_flip, r_noflip * np.array([-1.0, -1.0, 1.0]), atol=1e-9)


def test_ego_yaw_clamp_yaw_only(ego_bounds):
    """--ego-yaw-clamp (despin mirror of training clamp_yaw_command): clips ONLY the yaw-rate
    command, after yaw_scale, invariant under virtual_flip (yaw is body z under both the Rz(pi)
    flip and FLU->FRD); 0.0 = bit-identical off. Roll/pitch keep full +/-3.14 authority — the
    whole point vs --max-rate."""
    obs = np.zeros(EGO_OBS_DIM, dtype=np.float32)
    railed = _StubActor([0.0, 50.0, -50.0, 50.0])          # roll/pitch/yaw all at the rails
    # off (default): yaw rails at 3.14
    r_off, _, _ = fly_rl.policy_step(railed, obs, virtual_flip=False)
    np.testing.assert_allclose(r_off, [3.14, 3.14, 3.14], atol=1e-4)
    # clamp 0.7: yaw clipped, roll/pitch untouched at the rails
    r_cl, _, _ = fly_rl.policy_step(railed, obs, virtual_flip=False, yaw_clamp=0.7)
    np.testing.assert_allclose(r_cl, [3.14, 3.14, 0.7], atol=1e-4)
    # negative rail clips to -0.7 (FLU z == FRD z: no sign surprise on the wire channel)
    r_neg, _, _ = fly_rl.policy_step(_StubActor([0.0, 0.0, 0.0, -50.0]), obs,
                                     virtual_flip=False, yaw_clamp=0.7)
    assert r_neg[2] == pytest.approx(-0.7, abs=1e-4)
    # applied AFTER yaw_scale: a sub-clamp command scaled past the clamp still clips
    r_sc, _, _ = fly_rl.policy_step(_StubActor([0.0, 0.0, 0.0, 0.35]), obs,
                                    virtual_flip=False, yaw_scale=3.0, yaw_clamp=0.7)
    raw_yaw = 3.14 * np.tanh(0.35)                          # symmetric bounds: rescale == scale
    assert raw_yaw * 3.0 > 0.7                              # scale alone would exceed the clamp
    assert r_sc[2] == pytest.approx(0.7, abs=1e-4)
    # virtual_flip flips x/y only — the clamped yaw channel rides through unchanged
    r_fl, _, _ = fly_rl.policy_step(railed, obs, virtual_flip=True, yaw_clamp=0.7)
    assert r_fl[2] == pytest.approx(0.7, abs=1e-4)
    # sub-clamp commands pass through bit-identical to clamp-off
    mild = _StubActor([0.0, 0.2, -0.1, 0.05])
    r_a, _, _ = fly_rl.policy_step(mild, obs, virtual_flip=False)
    r_b, _, _ = fly_rl.policy_step(mild, obs, virtual_flip=False, yaw_clamp=0.7)
    np.testing.assert_allclose(r_b, r_a, atol=0.0)


def _approach_obs_batch() -> np.ndarray:
    """Synthetic in-distribution approach obs (virtual-flipped tail-first frame): flying at the
    gate 6-14 m out, slight offsets, fresh confident slot0, slot1 zeros."""
    rows = []
    rng = np.random.default_rng(3)
    for r in (6.0, 10.0, 14.0):
        for lat in (-1.5, 0.0, 1.5):
            rows.append(np.concatenate([
                [-2.5, 0.1 * lat, 0.0],                    # velocity (flipped FLU)
                [0.0, -0.05],                              # roll, pitch
                rng.normal(0, 0.1, 3),                     # body rates
                [1.88],                                    # last normed thrust ~ hover
                [0.0, 0.0],                                # sector
                [-r, -lat, 0.25], [1.0], [0.8],            # slot0 (flipped FLU), conf, area
                [0.0, 0.0, 0.0], [0.0], [0.0],             # slot1 pinned
            ]))
    return np.asarray(rows, dtype=np.float32)


@pytest.mark.skipif(not _CKPT.exists(), reason="vczext_final_actor.pth not pulled (gitignored "
                    "release-side artifact; see handoff/ego-deploy-2026-07-09/REPORT.md)")
def test_checkpoint_strict_load_and_smoke(ego_bounds):
    actor = fly_rl.load_ego_actor(str(_CKPT))
    obs = torch.as_tensor(_approach_obs_batch())
    with torch.no_grad():
        mu = actor(obs)
    assert mu.shape == (obs.shape[0], 4)
    assert torch.isfinite(mu).all()
    # pre-tanh mu -> in-bounds rescaled actions (tanh guarantees it; pin the numbers anyway)
    a = np.tanh(mu.numpy())
    act = fly_rl._ACT_MIN + (fly_rl._ACT_MAX - fly_rl._ACT_MIN) * (a + 1.0) / 2.0
    assert (act[:, 0] >= 0.0).all() and (act[:, 0] <= 3.765 + 1e-6).all()
    assert (np.abs(act[:, 1:]) <= 3.14 + 1e-6).all()
    # end-to-end policy_step on one row: finite command, collective in [0,1]
    rate_frd, coll, normed = fly_rl.policy_step(actor, _approach_obs_batch()[4])
    assert np.isfinite(rate_frd).all() and 0.0 <= coll <= 1.0 and 0.0 <= normed <= 3.765


@pytest.mark.skipif(not _CKPT.exists(), reason="vczext_final_actor.pth not pulled")
def test_checkpoint_rejects_wrong_width(tmp_path, ego_bounds):
    """An inc7 (17-dim) style dict must be rejected LOUD, never silently flown."""
    d = torch.load(str(_CKPT), map_location="cpu", weights_only=False)
    sd = dict(d["actor_mean"])
    sd["head.0.linear.weight"] = torch.zeros(256, 17)
    bad = tmp_path / "bad_actor.pth"
    torch.save({"actor_mean": sd, "actor_logstd": d["actor_logstd"]}, str(bad))
    with pytest.raises(SystemExit):
        fly_rl.load_ego_actor(str(bad))


@pytest.mark.skipif(not _CKPT.exists(), reason="vczext_final_actor.pth not pulled")
def test_directional_sanity_soft_print(ego_bounds):
    """SOFT (never asserts on direction): gate 10 m ahead offset 2 m LEFT vs RIGHT of the nose;
    print the commanded FRD rates for each so a human reviews the steering signs. FRD: +yaw_rate
    = nose RIGHT, +pitch_rate = nose DOWN, +roll_rate = right wing down."""
    from racer import frames
    from racer.contracts import GatePose

    actor = fly_rl.load_ego_actor(str(_CKPT))
    print("\n[directional-sanity] vczext_final — gate 10 m ahead, 2 m lateral offset:")
    for label, lat in (("LEFT ", -2.0), ("RIGHT", +2.0)):
        b = EgoObsBuilder(EgoObsBuilderConfig())
        rel_body = np.array([10.0, lat, 0.0])                     # body FRD, level drone
        t_cam = frames.R_camera_from_body() @ rel_body
        pose = GatePose(frame_id=1, sim_time_ns=0, R_cam_gate=np.eye(3),
                        t_cam_gate=t_cam, reproj_error_px=0.5)
        # deploy configuration: obs built flip-ON (builder default) + action un-flip in
        # policy_step(virtual_flip=True) — the SAME pairing _fly_ego runs.
        obs = b.update(sim_time_ns=1_000_000_000, gate_index=0, R_frd2ned=np.eye(3),
                       vel_ned=np.array([2.5, 0.0, 0.0]), gyro_frd=np.zeros(3),
                       pose=pose, last_normed_thrust=1.88)
        rate_frd, coll, _ = fly_rl.policy_step(actor, obs, virtual_flip=True)
        print(f"  gate {label}: wire FRD rates=[roll {rate_frd[0]:+.3f}, "
              f"pitch {rate_frd[1]:+.3f}, yaw {rate_frd[2]:+.3f}] rad/s  "
              f"collective={coll:.3f}  slot0(obs)={obs[11:14].round(2).tolist()}")
    print("  (human review: LEFT vs RIGHT should steer opposite ways in yaw and/or roll)")


def test_pitch_clamp_blocks_nose_down_past_cap(ego_bounds):
    """Perception fence (--ego-pitch-clamp): past the nose-down cap, block FURTHER nose-down
    (rate_frd[1] < 0, empirically the nose-down command sign) but always pass nose-up recovery;
    inactive above the cap and when off. obs[4] = leveled body pitch (nose-down negative)."""
    cap = float(np.radians(30.0))
    nd = _StubActor([0.0, 0.0, 50.0, 0.0])    # -> rate_frd[1] < 0 (nose-down) at virtual_flip=False
    nu = _StubActor([0.0, 0.0, -50.0, 0.0])   # -> rate_frd[1] > 0 (nose-up)
    past = np.zeros(EGO_OBS_DIM, np.float32); past[4] = -0.60   # -34 deg: past the 30 deg cap
    within = np.zeros(EGO_OBS_DIM, np.float32); within[4] = -0.30  # -17 deg: within cap (resting tilt)
    ps = lambda actor, obs, cap_: fly_rl.policy_step(
        actor, obs, virtual_flip=False, pitch_clamp_rad=cap_)[0][1]
    # sanity: unfenced, the stubs really are nose-down / nose-up
    assert ps(nd, past, 0.0) < 0 and ps(nu, past, 0.0) > 0
    # past cap: nose-down hard-blocked to 0, nose-up untouched
    assert ps(nd, past, cap) == pytest.approx(0.0, abs=1e-9)
    assert ps(nu, past, cap) > 0
    # within cap: fence inactive (nose-down passes)
    assert ps(nd, within, cap) < 0


def test_roll_clamp_blocks_roll_past_cap_both_sides(ego_bounds):
    """Attitude fence (--ego-roll-clamp): the SYMMETRIC mirror of the pitch fence. Past +cap block
    FURTHER positive roll (rate_frd[0] -> min(.,0)); past -cap block FURTHER negative roll
    (rate_frd[0] -> max(.,0)); roll-toward-level always passes; inactive within the band and when
    off. obs[3] = leveled body roll; rate_frd[0] = the FRD roll-rate command (act[1] via [1,-1,1])."""
    cap = float(np.radians(45.0))
    pr = _StubActor([0.0, 50.0, 0.0, 0.0])     # -> rate_frd[0] > 0 (positive roll) at virtual_flip=False
    nr = _StubActor([0.0, -50.0, 0.0, 0.0])    # -> rate_frd[0] < 0 (negative roll)
    past_pos = np.zeros(EGO_OBS_DIM, np.float32); past_pos[3] = 0.90   # +51.6 deg: past the +45 cap
    past_neg = np.zeros(EGO_OBS_DIM, np.float32); past_neg[3] = -0.90  # -51.6 deg: past the -45 cap
    within = np.zeros(EGO_OBS_DIM, np.float32); within[3] = 0.30       # +17 deg: within the band
    ps = lambda actor, obs, cap_: fly_rl.policy_step(
        actor, obs, virtual_flip=False, roll_clamp_rad=cap_)[0][0]
    # sanity: unfenced, the stubs really roll +/-
    assert ps(pr, past_pos, 0.0) > 0 and ps(nr, past_pos, 0.0) < 0
    # past +cap: positive roll hard-blocked to 0, negative (toward level) untouched
    assert ps(pr, past_pos, cap) == pytest.approx(0.0, abs=1e-9)
    assert ps(nr, past_pos, cap) < 0
    # past -cap: negative roll hard-blocked to 0, positive (toward level) untouched
    assert ps(nr, past_neg, cap) == pytest.approx(0.0, abs=1e-9)
    assert ps(pr, past_neg, cap) > 0
    # within the band: fence inactive (both directions pass)
    assert ps(pr, within, cap) > 0 and ps(nr, within, cap) < 0
    # off (cap=0): bit-identical passthrough even past the attitude limit
    r_off = fly_rl.policy_step(pr, past_pos, virtual_flip=False)[0]
    r_on0 = fly_rl.policy_step(pr, past_pos, virtual_flip=False, roll_clamp_rad=0.0)[0]
    np.testing.assert_allclose(r_on0, r_off, atol=0.0)
