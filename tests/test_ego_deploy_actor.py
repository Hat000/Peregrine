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
