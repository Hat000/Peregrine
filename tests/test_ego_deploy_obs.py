"""EGO deploy obs builder — parity against the VERBATIM training reference + frame checks.

Part 1 (the load-bearing test): ``racer.ego_obs.ego_actor_obs_np`` (the deploy assembly/masking
core) is pinned ELEMENT-EXACT against test-local reference functions lifted VERBATIM from the
training source (chaum worktree ``rl/peregrine_racing_ego.py`` — ``ego_window_indices`` lines
241-253 and the ``ego_actor_obs`` masking/concat core lines 291-333), over randomized fixtures
(G in {1,2,4}, random target gates incl. course-end clamping, random conf/det/area incl. exact
zeros, obs_coast both ways). Any silent divergence in window indexing, keep-logic, mask
multiply or concat ORDER kills the policy (proven: H6) — this is the tripwire.

Part 2: the stateful ``EgoObsBuilder`` frame contract — camera-optical -> body FRD (+20 deg
mount + the METRIC -0.25 m boresight vert bake inherited LIVE from frames.BORESIGHT) -> FLU ->
the virtual pi-about-body-z flip; the staleness/det-hold masking cliff; the static coarse
sector; the apparent-area analog.

[ego-deploy 2026-07-09]
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from racer import frames
from racer.contracts import GatePose
from racer.ego_obs import (
    EGO_OBS_DIM,
    EgoObsBuilder,
    EgoObsBuilderConfig,
    ego_actor_obs_np,
    rel_pos_body_frd_from_gatepose,
    roll_pitch_zup,
    visible_area_from_gatepose,
)

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))


# =====================================================================================
# VERBATIM TRAINING REFERENCE — lifted from the chaum worktree training source.
# Provenance: rl/peregrine_racing_ego.py @ branch claude/optimistic-chaum-6c893b,
#   WINDOW/PER_SLOT constants lines 78-81, ego_window_indices lines 241-253,
#   ego_actor_obs masking/concat core lines 291-333. Copied (not imported) so the deploy
#   worktree pins the CONTRACT, not a moving file. Do NOT "improve" this code.
# =====================================================================================
WINDOW = 2                              # [current, next]                (line 79)
PER_SLOT = 5                            # rel_pos(3) + conf(1) + area(1) (line 80)


def _ref_ego_window_indices(target_gates, n_gates):
    # --- verbatim: peregrine_racing_ego.py:241-253 ---
    ar = torch.arange(WINDOW, device=target_gates.device)
    raw = target_gates.long().unsqueeze(-1) + ar.unsqueeze(0)           # (N,WINDOW)
    valid = raw < n_gates
    gidx = raw.clamp(max=n_gates - 1)
    return gidx, valid


def _ref_ego_actor_obs(est, detectable, target_gates, last_collective, sector, n_gates,
                       obs_coast=False):
    # --- verbatim core: peregrine_racing_ego.py:291-333 (docstring dropped) ---
    N = est.rel_pos.shape[0]
    dev, dt = est.rel_pos.device, est.rel_pos.dtype
    ar = torch.arange(N, device=dev)
    tg = target_gates.long()

    if last_collective.dim() == 1:
        last_collective = last_collective.unsqueeze(-1)                 # (N,1)

    gidx, valid = _ref_ego_window_indices(tg, n_gates)                  # (N,WINDOW)

    # per-slot gather of rel_pos / confidence / visible_area, then mask
    slots = []
    for k in range(WINDOW):
        g = gidx[:, k]                                                  # (N,)
        rel = est.rel_pos[ar, g]                                        # (N,3) body-frame
        conf = est.confidence[ar, g]                                    # (N,)
        area = est.visible_area[ar, g]                                  # (N,)
        det = detectable[ar, g]                                         # (N,)
        keep = valid[:, k] & (conf > 0.0)                               # (N,) in-horizon test
        if not obs_coast:
            keep = keep & det                       # legacy hard-mask on this-step visibility
        keep_f = keep.to(dt)
        rel = rel * keep_f.unsqueeze(-1)
        conf = conf * keep_f
        area = area * keep_f
        slots.append(torch.cat([rel, conf.unsqueeze(-1), area.unsqueeze(-1)], dim=-1))  # (N,5)

    coarse_sector = sector[ar, tg].to(dt)                               # (N,2)

    obs = torch.cat([
        est.velocity,                                                   # (N,3) body velocity
        est.roll_pitch,                                                 # (N,2)
        est.body_rates,                                                 # (N,3)
        last_collective,                                                # (N,1)
        coarse_sector,                                                  # (N,2)
        *slots,                                                         # 2 x (N,5)
    ], dim=-1)
    return obs
# ===================================== end verbatim ==================================


@pytest.mark.parametrize("n_gates", [1, 2, 4])
@pytest.mark.parametrize("obs_coast", [False, True])
def test_assembly_parity_vs_training_reference(n_gates, obs_coast):
    """Deploy assembly == the verbatim training assembly, element-exact, over random fixtures."""
    rng = np.random.default_rng(20260709 + n_gates + int(obs_coast))
    N, G = 128, n_gates

    rel_pos = rng.normal(0.0, 12.0, size=(N, G, 3)).astype(np.float32)
    # confidence/area: mix of zeros (masked), tiny, mid, one (exercise the conf>0 edge)
    confidence = rng.choice(
        [0.0, 1e-6, 0.3, 0.72, 1.0], size=(N, G), p=[0.3, 0.1, 0.2, 0.2, 0.2]).astype(np.float32)
    visible_area = rng.uniform(0.0, 1.0, size=(N, G)).astype(np.float32)
    detectable = rng.uniform(size=(N, G)) < 0.6
    velocity = rng.normal(0.0, 5.0, size=(N, 3)).astype(np.float32)
    roll_pitch = rng.normal(0.0, 0.5, size=(N, 2)).astype(np.float32)
    body_rates = rng.normal(0.0, 2.0, size=(N, 3)).astype(np.float32)
    last_coll = rng.uniform(0.0, 3.765, size=(N,)).astype(np.float32)
    sector = rng.integers(-1, 2, size=(N, G, 2)).astype(np.float32)
    # random target gates incl. the LAST gate (slot1 must clamp+mask at course end)
    target = rng.integers(0, G, size=(N,))

    est = SimpleNamespace(
        rel_pos=torch.as_tensor(rel_pos), confidence=torch.as_tensor(confidence),
        visible_area=torch.as_tensor(visible_area), velocity=torch.as_tensor(velocity),
        roll_pitch=torch.as_tensor(roll_pitch), body_rates=torch.as_tensor(body_rates))
    ref = _ref_ego_actor_obs(
        est, torch.as_tensor(detectable), torch.as_tensor(target),
        torch.as_tensor(last_coll), torch.as_tensor(sector), G,
        obs_coast=obs_coast).numpy()
    assert ref.shape == (N, EGO_OBS_DIM)

    mine = np.stack([
        ego_actor_obs_np(
            velocity=velocity[i], roll_pitch=roll_pitch[i], body_rates=body_rates[i],
            last_collective=float(last_coll[i]), sector=sector[i], rel_pos=rel_pos[i],
            confidence=confidence[i], visible_area=visible_area[i], detectable=detectable[i],
            target_gate=int(target[i]), n_gates=G, obs_coast=obs_coast)
        for i in range(N)])
    np.testing.assert_allclose(mine, ref, rtol=0.0, atol=1e-6)


def test_single_gate_slot1_always_masked():
    """n_gates=1 (the champion's whole training life): slot1 is window-invalid -> zeros even
    with a confident, detectable gate parked in its clamped index."""
    obs = ego_actor_obs_np(
        velocity=np.zeros(3), roll_pitch=np.zeros(2), body_rates=np.zeros(3),
        last_collective=1.0, sector=np.zeros((1, 2)), rel_pos=np.array([[3.0, 2.0, 1.0]]),
        confidence=np.array([1.0]), visible_area=np.array([0.9]), detectable=np.array([True]),
        target_gate=0, n_gates=1)
    assert obs.shape == (EGO_OBS_DIM,)
    np.testing.assert_array_equal(obs[16:21], 0.0)          # slot1 pinned zero
    np.testing.assert_allclose(obs[11:14], [3.0, 2.0, 1.0])  # slot0 passes through


def test_nan_guard_zeroes_nonfinite():
    """The get_observations finite-guard (peregrine_racing_ego.py:759-761): non-finite -> 0."""
    obs = ego_actor_obs_np(
        velocity=np.array([np.nan, 1.0, np.inf]), roll_pitch=np.zeros(2),
        body_rates=np.zeros(3), last_collective=1.0, sector=np.zeros((1, 2)),
        rel_pos=np.array([[1.0, np.nan, 2.0]]), confidence=np.array([1.0]),
        visible_area=np.array([0.5]), detectable=np.array([True]), target_gate=0, n_gates=1)
    assert np.isfinite(obs).all()
    assert obs[0] == 0.0 and obs[2] == 0.0 and obs[12] == 0.0
    assert obs[1] == 1.0


def test_flip_constants_match_fly_rl():
    """ego_obs redefines the body flip constants (src/ must not import rl/); pin them equal to
    fly_rl's _FLIP (body part) and _RZ_PI_BODY so a drift is impossible."""
    import fly_rl
    from racer import ego_obs as eo
    np.testing.assert_array_equal(eo._FLIP_FRD_FLU, fly_rl._FLIP)
    np.testing.assert_array_equal(eo._RZ_PI_BODY, fly_rl._RZ_PI_BODY)


# =====================================================================================
# Part 2 — the stateful builder: frames, boresight bake, masking cliff, sector, area.
# =====================================================================================
_CAM_PITCH = np.deg2rad(20.0)


def _pose_dead_ahead(r: float, lateral: float = 0.0, vert_body: float = 0.0) -> GatePose:
    """A GatePose whose body-FRD gate centre is (r, lateral, vert_body) for a LEVEL drone:
    t_cam_gate = R_camera_from_body() @ rel_body (frames convention v_cam = R_cb v_body).
    R_cam_gate = identity-ish (head-on) — only used by the area channel."""
    rel_body = np.array([r, lateral, vert_body], dtype=np.float64)
    t_cam = frames.R_camera_from_body() @ rel_body
    return GatePose(frame_id=1, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam,
                    reproj_error_px=0.5)


def test_rel_pos_inherits_mount_and_metric_boresight():
    """camera-optical -> body FRD goes through frames.R_camera_from_body() AND adds the LIVE
    METRIC vert offset (BORESIGHT.vert_offset_m = -0.25, FRD +Z down) — the same dual-form bake
    the localization +L lever applies."""
    r = 12.0
    pose = _pose_dead_ahead(r)
    rel_frd = rel_pos_body_frd_from_gatepose(pose.t_cam_gate)
    voff = frames.BORESIGHT.vert_offset_m
    assert voff == -0.25   # the P3-baked calibration this adapter must inherit
    np.testing.assert_allclose(rel_frd, [r, 0.0, voff], atol=1e-9)


def test_builder_dead_ahead_frames_and_flip():
    """Level drone, gate r=10 m dead ahead, flying at it at 3 m/s: the virtual-flipped obs must
    read rel ~ (-10, 0, +0.25) FLU and velocity ~ (-3, 0, 0) (tail-first trained frame)."""
    b = EgoObsBuilder(EgoObsBuilderConfig())
    obs = b.update(sim_time_ns=1_000_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.array([3.0, 0.0, 0.0]), gyro_frd=np.zeros(3),
                   pose=_pose_dead_ahead(10.0), last_normed_thrust=1.5)
    assert obs.shape == (EGO_OBS_DIM,)
    np.testing.assert_allclose(obs[0:3], [-3.0, 0.0, 0.0], atol=1e-6)   # velocity, flipped FLU
    np.testing.assert_allclose(obs[3:5], [0.0, 0.0], atol=1e-9)         # roll, pitch (level)
    np.testing.assert_allclose(obs[5:8], 0.0, atol=1e-9)                # rates
    assert obs[8] == pytest.approx(1.5)                                 # g-units feedback
    np.testing.assert_allclose(obs[11:14], [-10.0, 0.0, 0.25], atol=1e-5)  # rel, flipped FLU
    assert obs[14] == pytest.approx(1.0)                                # fresh fix conf
    assert obs[15] == pytest.approx(1.0, abs=0.05)                      # head-on area ~ 1
    np.testing.assert_array_equal(obs[16:21], 0.0)                      # slot1 pinned


def test_builder_masking_cliff_det_hold():
    """Coast-OFF champion parity: slot0 masks to ZEROS once the fix age exceeds det_hold (0.2 s),
    with confidence = 1 - age/0.5 while held. No coasted rel_pos ever leaks past the hold."""
    b = EgoObsBuilder(EgoObsBuilderConfig())
    t0 = 1_000_000_000
    obs = b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3),
                   pose=_pose_dead_ahead(10.0), last_normed_thrust=0.0)
    assert obs[14] == pytest.approx(1.0)
    # +0.1 s, no new fix: still inside det_hold -> fed, conf = 1 - 0.1/0.5 = 0.8
    obs = b.update(sim_time_ns=t0 + 100_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=None, last_normed_thrust=0.0)
    assert obs[14] == pytest.approx(0.8, abs=1e-6)
    assert abs(obs[11]) > 5.0                                            # rel still fed
    # +0.3 s: past det_hold (0.2) -> the blackout cliff: slot0 ZEROS (conf would be 0.4)
    obs = b.update(sim_time_ns=t0 + 300_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=None, last_normed_thrust=0.0)
    np.testing.assert_array_equal(obs[11:16], 0.0)
    # re-acquisition: a fresh pose snaps the slot back
    obs = b.update(sim_time_ns=t0 + 400_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3),
                   pose=_pose_dead_ahead(9.0), last_normed_thrust=0.0)
    assert obs[14] == pytest.approx(1.0)


def test_builder_gap_propagation_translates_rel():
    """Between fixes (inside det_hold) the held rel_pos is ego-propagated: flying forward 3 m/s
    for 0.1 s shrinks the ahead-range by ~0.3 m (training estimator parity)."""
    b = EgoObsBuilder(EgoObsBuilderConfig())
    t0 = 1_000_000_000
    b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3),
             vel_ned=np.array([3.0, 0.0, 0.0]), gyro_frd=np.zeros(3),
             pose=_pose_dead_ahead(10.0), last_normed_thrust=0.0)
    obs = b.update(sim_time_ns=t0 + 100_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.array([3.0, 0.0, 0.0]), gyro_frd=np.zeros(3),
                   pose=None, last_normed_thrust=0.0)
    # flipped FLU ahead-component is -x: was -10.0, drone moved +0.3 toward it -> -9.7
    assert obs[11] == pytest.approx(-9.7, abs=1e-3)


def test_builder_gate_index_change_resets_slot():
    """RACE_STATUS advance: the new slot0 starts COLD (masked) until its first fix — the deploy
    analog of the training window promotion (no teleport, no stale carry-over)."""
    b = EgoObsBuilder(EgoObsBuilderConfig())
    t0 = 1_000_000_000
    obs = b.update(sim_time_ns=t0, gate_index=3, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                   gyro_frd=np.zeros(3), pose=_pose_dead_ahead(8.0), last_normed_thrust=0.0)
    assert obs[14] == pytest.approx(1.0)
    obs = b.update(sim_time_ns=t0 + 33_000_000, gate_index=4, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=None, last_normed_thrust=0.0)
    np.testing.assert_array_equal(obs[11:16], 0.0)


def test_builder_sector_static_elevation_bucket():
    """Sector analog: horiz = 0 always (single-gate training value); vert = +1 for a gate well
    above the leveled horizon at FIRST fix, then HELD STATIC as the geometry changes."""
    b = EgoObsBuilder(EgoObsBuilderConfig())
    t0 = 1_000_000_000
    # gate 10 m ahead, 4 m ABOVE (body FRD z = -4): elevation atan2(4,10) ~ 21.8 deg > deadband
    obs = b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                   gyro_frd=np.zeros(3), pose=_pose_dead_ahead(10.0, vert_body=-4.0),
                   last_normed_thrust=0.0)
    np.testing.assert_allclose(obs[9:11], [0.0, 1.0])
    # later fix with the gate now LEVEL: the sector must NOT move (static per gate)
    obs = b.update(sim_time_ns=t0 + 33_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=_pose_dead_ahead(5.0),
                   last_normed_thrust=0.0)
    np.testing.assert_allclose(obs[9:11], [0.0, 1.0])


def test_builder_sector_zero_mode():
    b = EgoObsBuilder(EgoObsBuilderConfig(sector_mode="zero"))
    obs = b.update(sim_time_ns=1_000_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3),
                   pose=_pose_dead_ahead(10.0, vert_body=-4.0), last_normed_thrust=0.0)
    np.testing.assert_array_equal(obs[9:11], 0.0)


def test_visible_area_head_on_vs_oblique_vs_edge_on():
    """Range-invariant squareness: head-on ~1 (any range), oblique in between and monotonically
    below head-on, edge-on -> ~0, corner-behind-camera -> exactly 0."""
    from scipy.spatial.transform import Rotation
    t = np.array([0.0, 0.0, 10.0])
    a_head = visible_area_from_gatepose(np.eye(3), t)
    assert a_head == pytest.approx(1.0, abs=1e-3)
    a_head_far = visible_area_from_gatepose(np.eye(3), np.array([0.0, 0.0, 25.0]))
    assert a_head_far == pytest.approx(a_head, abs=1e-3)              # range-invariant
    a_45 = visible_area_from_gatepose(Rotation.from_euler("y", 45, degrees=True).as_matrix(), t)
    assert 0.05 < a_45 < a_head
    a_edge = visible_area_from_gatepose(Rotation.from_euler("y", 89.5, degrees=True).as_matrix(), t)
    assert a_edge < 0.05
    behind = visible_area_from_gatepose(np.eye(3), np.array([0.0, 0.0, -5.0]))
    assert behind == 0.0                                              # gate behind the camera
    # a STEEPLY tilted close gate puts one corner behind the image plane -> 0 (all_front guard)
    straddle = visible_area_from_gatepose(
        Rotation.from_euler("y", 80, degrees=True).as_matrix(), np.array([0.0, 0.0, 0.5]))
    assert straddle == 0.0


def test_roll_pitch_training_extraction_roundtrip():
    """The Z-up extraction matches the ZYX construction it inverts (yaw never recovered)."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(7)
    for _ in range(20):
        r, p, y = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-np.pi, np.pi)
        R = Rotation.from_euler("ZYX", [y, p, r]).as_matrix()
        ro, po = roll_pitch_zup(R)
        assert ro == pytest.approx(r, abs=1e-9)
        assert po == pytest.approx(p, abs=1e-9)


def test_slot1_stub_rejected():
    with pytest.raises(NotImplementedError):
        EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
