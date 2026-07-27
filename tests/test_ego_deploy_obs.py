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


def test_slot1_enabled_constructs():
    """slot1_enabled is no longer a stub: the builder constructs and emits a 21-dim obs (slot1
    masks to zero until a next_pose is supplied)."""
    b = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    obs = b.update(sim_time_ns=1_000_000_000, gate_index=0, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3),
                   pose=_pose_dead_ahead(10.0), last_normed_thrust=0.0)
    assert obs.shape == (EGO_OBS_DIM,)
    np.testing.assert_array_equal(obs[16:21], 0.0)   # no next_pose yet -> slot1 zero


# =====================================================================================
# SLOT1 (WINDOW=2 next-gate) — the RL-owned SINK activation (2026-07-12).
# The H6 tripwire at the builder level: slot1_enabled=False OR next_pose=None must stay
# byte-identical to the single-gate deploy; a supplied next_pose fills slot1 through the
# SAME masking; it masks back to zero on staleness/None/gate-change.
# =====================================================================================
def _run_seq(builder, next_poses=None):
    """Drive a scripted sequence (fix, gap, stale-out, re-acq, gate advance) and return the
    stacked obs. next_poses: optional list of the per-tick next_pose (else all None)."""
    t0 = 1_000_000_000
    script = [
        # (dt_ns, gate_index, pose, vel_ned)
        (0,            0, _pose_dead_ahead(10.0), np.array([3.0, 0.0, 0.0])),
        (100_000_000,  0, None,                   np.array([3.0, 0.0, 0.0])),   # gap (inside det_hold)
        (300_000_000,  0, None,                   np.array([3.0, 0.0, 0.0])),   # stale past det_hold
        (400_000_000,  0, _pose_dead_ahead(8.0),  np.array([3.0, 0.0, 0.0])),   # re-acquire
        (450_000_000,  1, None,                   np.array([3.0, 0.0, 0.0])),   # gate advance -> cold
    ]
    out = []
    for i, (dt, gi, pose, vel) in enumerate(script):
        npose = None if next_poses is None else next_poses[i]
        out.append(builder.update(
            sim_time_ns=t0 + dt, gate_index=gi, R_frd2ned=np.eye(3), vel_ned=vel,
            gyro_frd=np.zeros(3), pose=pose, next_pose=npose, last_normed_thrust=0.0).copy())
    return np.stack(out)


def test_slot1_disabled_or_none_is_byte_identical():
    """H6 TRIPWIRE (builder level): slot1_enabled=False vs slot1_enabled=True fed next_pose=None
    every tick must be byte-identical across the whole scripted flight (slot1 zeros either way)."""
    off = _run_seq(EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=False)))
    on_none = _run_seq(EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True)))
    np.testing.assert_array_equal(off, on_none)
    np.testing.assert_array_equal(off[:, 16:21], 0.0)   # slot1 pinned zero throughout


def test_slot1_populated_carries_next_gate_through_same_masking():
    """A supplied next_pose fills obs[16:21] with the NEXT gate's rel_pos/conf/area through the
    SAME frame + masking as slot0, and leaves obs[0:16] byte-identical to the slot1-off build."""
    b_on = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    b_off = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=False))
    kw = dict(sim_time_ns=1_000_000_000, gate_index=0, R_frd2ned=np.eye(3),
              vel_ned=np.array([3.0, 0.0, 0.0]), gyro_frd=np.zeros(3),
              pose=_pose_dead_ahead(10.0), last_normed_thrust=1.5)
    # next gate 18 m ahead, 3 m to the RIGHT (body FRD +y), head-on quad
    next_pose = _pose_dead_ahead(18.0, lateral=3.0)
    obs_on = b_on.update(next_pose=next_pose, **kw)
    obs_off = b_off.update(**kw)
    # slot0 + all non-slot1 channels UNCHANGED by activating slot1
    np.testing.assert_array_equal(obs_on[0:16], obs_off[0:16])
    np.testing.assert_allclose(obs_on[11:14], [-10.0, 0.0, 0.25], atol=1e-5)   # slot0 (flipped FLU)
    # slot1: rel_body_frd = [18, 3, -0.25(boresight)] -> FLU [18,-3,0.25] -> virtual-flip [-18,3,0.25]
    np.testing.assert_allclose(obs_on[16:19], [-18.0, 3.0, 0.25], atol=1e-5)
    assert obs_on[19] == pytest.approx(1.0)              # fresh next-gate conf
    assert obs_on[20] == pytest.approx(1.0, abs=0.05)    # head-on area ~ 1
    # cross-check the FULL 21-dim against the VERBATIM training reference at n_gates=2, fed the
    # builder's own per-slot channels (proves the 2-slot assembly IS the training code path).
    d = b_on.last_diag
    est = SimpleNamespace(
        rel_pos=torch.as_tensor(np.stack([obs_on[11:14], obs_on[16:19]])[None]).float(),
        confidence=torch.as_tensor([[obs_on[14], obs_on[19]]]).float(),
        visible_area=torch.as_tensor([[obs_on[15], obs_on[20]]]).float(),
        velocity=torch.as_tensor(obs_on[0:3][None]).float(),
        roll_pitch=torch.as_tensor(obs_on[3:5][None]).float(),
        body_rates=torch.as_tensor(obs_on[5:8][None]).float())
    sec = np.tile(obs_on[9:11], (2, 1))[None]           # (1,G=2,2); only tg=0 row is read
    ref = _ref_ego_actor_obs(
        est, torch.as_tensor([[True, True]]), torch.as_tensor([0]),
        torch.as_tensor([obs_on[8]]).float(), torch.as_tensor(sec).float(),
        2, obs_coast=False).numpy()[0]
    np.testing.assert_allclose(obs_on, ref, rtol=0.0, atol=1e-6)
    assert d["slot1_enabled"] is True and d["pose_seen1"] is True


def test_slot1_masks_to_zero_when_next_pose_goes_stale():
    """slot1 masks INDEPENDENTLY of slot0: with slot0 fed fresh every tick, dropping next_pose
    past det_hold (0.2 s) zeros obs[16:21] while slot0 stays live."""
    b = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    t0 = 1_000_000_000
    kw0 = dict(gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3), gyro_frd=np.zeros(3),
               last_normed_thrust=0.0)
    b.update(sim_time_ns=t0, pose=_pose_dead_ahead(10.0),
             next_pose=_pose_dead_ahead(18.0, lateral=3.0), **kw0)
    # +0.1 s inside det_hold, next_pose dropped: still fed (conf decays)
    obs = b.update(sim_time_ns=t0 + 100_000_000, pose=_pose_dead_ahead(9.0), next_pose=None, **kw0)
    assert obs[19] == pytest.approx(0.8, abs=1e-6)   # slot1 conf = 1 - 0.1/0.5
    assert abs(obs[16]) > 5.0                          # slot1 rel still fed
    # +0.3 s past det_hold: slot1 blackout cliff -> zeros; slot0 (fresh) stays live
    obs = b.update(sim_time_ns=t0 + 300_000_000, pose=_pose_dead_ahead(7.0), next_pose=None, **kw0)
    np.testing.assert_array_equal(obs[16:21], 0.0)     # slot1 masked
    assert obs[14] == pytest.approx(1.0)               # slot0 fresh, unaffected


def test_slot1_reset_on_gate_change_clears_both_slots():
    """RACE_STATUS advance resets BOTH slots: with no fresh poses on the new gate, slot0 AND slot1
    start cold (masked)."""
    b = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    t0 = 1_000_000_000
    obs = b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                   gyro_frd=np.zeros(3), pose=_pose_dead_ahead(10.0),
                   next_pose=_pose_dead_ahead(18.0, lateral=3.0), last_normed_thrust=0.0)
    assert obs[14] == pytest.approx(1.0) and obs[19] == pytest.approx(1.0)   # both populated
    obs = b.update(sim_time_ns=t0 + 33_000_000, gate_index=1, R_frd2ned=np.eye(3),
                   vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=None, next_pose=None,
                   last_normed_thrust=0.0)
    np.testing.assert_array_equal(obs[11:16], 0.0)     # slot0 cold
    np.testing.assert_array_equal(obs[16:21], 0.0)     # slot1 cold


# =====================================================================================
# COARSE-MAP 'map' sector mode (2026-07-12) — the horizontal turn prior the _pef
# champions trained on but 'auto' cannot compute (no next-gate geometry on the wire).
# =====================================================================================
def _map_update(builder, gate_index, t_ns=0):
    """update() with no fix (blind approach) — isolates the STATIC sector channel."""
    return builder.update(
        sim_time_ns=t_ns, gate_index=gate_index, R_frd2ned=np.eye(3),
        vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=None, last_normed_thrust=0.0)


def test_sector_map_feeds_static_bucket_before_any_fix():
    """'map' latches coarse_map[active_gate_index] into obs[9:11] on the gate-change boundary —
    live through the blind approach (pose=None), promoting on advance, clamping past the last row."""
    cmap = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])   # g0 right-up, g1 straight, g2 left-down
    b = EgoObsBuilder(EgoObsBuilderConfig(sector_mode="map", coarse_map=cmap, virtual_flip=False))
    np.testing.assert_allclose(_map_update(b, 0, 0)[9:11], [-1.0, 1.0])      # blind, no fix yet
    np.testing.assert_allclose(_map_update(b, 1, 1_000_000)[9:11], [0.0, 0.0])   # promote
    np.testing.assert_allclose(_map_update(b, 9, 2_000_000)[9:11], [1.0, -1.0])  # clamp to last row


def test_sector_map_requires_valid_coarse_map():
    with pytest.raises(ValueError):
        EgoObsBuilder(EgoObsBuilderConfig(sector_mode="map", coarse_map=None))
    with pytest.raises(ValueError):   # buckets must be in {-1,0,1}
        EgoObsBuilder(EgoObsBuilderConfig(sector_mode="map", coarse_map=np.array([[2.0, 0.0]])))


def test_sector_auto_zero_untouched_by_map_addition():
    """Regression: adding 'map' left 'auto'/'zero' emitting (0,0) before a fix (no perturbation)."""
    for mode in ("auto", "zero"):
        b = EgoObsBuilder(EgoObsBuilderConfig(sector_mode=mode, virtual_flip=False))
        np.testing.assert_allclose(_map_update(b, 0, 0)[9:11], [0.0, 0.0])


# ---------------------------------------------------------------------------
# OBS FIX GAIN (2026-07-25): rel_new = (1-K)*propagated_held + K*fix, K forced to 1 on
# re-acquisition. Training reference: chaum rl/ego_estimator.py:722-734 -- K = 1/N_eff clamped to
# [0,1], and reacq = (t_since_fix > stale_horizon_s) => K=1.
# ---------------------------------------------------------------------------
def _held(b: EgoObsBuilder) -> np.ndarray:
    return np.asarray(b.last_diag["rel_flu"], dtype=np.float64)


def test_fix_gain_default_is_the_snap():
    """DEFAULT-OFF: fix_gain defaults to 1.0, and a fresh fix then lands the held lever exactly on
    the fix -- the pre-2026-07-25 behaviour, with no blend arithmetic in the way."""
    assert EgoObsBuilderConfig().fix_gain == 1.0
    b = EgoObsBuilder(EgoObsBuilderConfig(virtual_flip=False))
    t0 = 1_000_000_000
    pose2 = _pose_dead_ahead(8.0, vert_body=-1.0)
    b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.array([5.0, 0.0, 0.0]),
             gyro_frd=np.zeros(3), pose=_pose_dead_ahead(10.0), last_normed_thrust=0.0)
    b.update(sim_time_ns=t0 + 33_000_000, gate_index=0, R_frd2ned=np.eye(3),
             vel_ned=np.array([5.0, 0.0, 0.0]), gyro_frd=np.zeros(3), pose=pose2,
             last_normed_thrust=0.0)
    # the held lever IS the second fix (no trace of the propagated 10 m prior)
    expect = rel_pos_body_frd_from_gatepose(pose2.t_cam_gate) * np.array([1.0, -1.0, -1.0])
    np.testing.assert_allclose(_held(b), expect, atol=1e-12)


def test_fix_gain_blends_fix_into_the_propagated_belief():
    """K<1: the accepted fix moves the held lever exactly K of the way from the PROPAGATED prior
    (not from the previous raw fix) toward the new fix."""
    K, dt_ns = 0.25, 33_000_000
    cfg = dict(virtual_flip=False, stale_horizon_s=1.0, det_hold_s=1.0)
    snap = EgoObsBuilder(EgoObsBuilderConfig(**cfg))                     # K=1 reference arm
    blend = EgoObsBuilder(EgoObsBuilderConfig(fix_gain=K, **cfg))
    t0, v = 1_000_000_000, np.array([4.0, 0.0, 0.0])
    p0, p1 = _pose_dead_ahead(10.0), _pose_dead_ahead(8.0, vert_body=-1.0)
    for b in (snap, blend):
        b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=v,
                 gyro_frd=np.zeros(3), pose=p0, last_normed_thrust=0.0)
    # the prior both arms hold entering tick 2 = the first fix ego-propagated one tick forward.
    # Zero body rates => the rotation is identity and only the -v_flu*dt translation acts.
    prior = _held(snap).copy() - np.array([4.0, 0.0, 0.0]) * (dt_ns / 1e9)
    for b in (snap, blend):
        b.update(sim_time_ns=t0 + dt_ns, gate_index=0, R_frd2ned=np.eye(3), vel_ned=v,
                 gyro_frd=np.zeros(3), pose=p1, last_normed_thrust=0.0)
    fix = _held(snap)                                                   # K=1 arm == the raw fix
    np.testing.assert_allclose(_held(blend), (1.0 - K) * prior + K * fix, atol=1e-12)
    assert not np.allclose(_held(blend), fix)                           # the blend is not a snap


def test_fix_gain_snaps_on_first_acquisition():
    """A COLD slot has no belief to blend into -- the first fix snaps whatever the gain."""
    b = EgoObsBuilder(EgoObsBuilderConfig(fix_gain=0.154, virtual_flip=False))
    ref = EgoObsBuilder(EgoObsBuilderConfig(virtual_flip=False))
    for x in (b, ref):
        x.update(sim_time_ns=1_000_000_000, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                 gyro_frd=np.zeros(3), pose=_pose_dead_ahead(10.0, vert_body=-2.0),
                 last_normed_thrust=0.0)
    np.testing.assert_allclose(_held(b), _held(ref), atol=1e-12)


def test_fix_gain_snaps_on_reacquisition_past_the_stale_horizon():
    """Training's reacq branch (ego_estimator.py:730-732): a belief older than stale_horizon_s has
    drifted, so its first fresh fix uses K=1 rather than the slow blend. A fix INSIDE the horizon
    must still blend -- both halves are pinned so the boundary cannot silently move."""
    K, horizon = 0.154, 0.5
    cfg = dict(fix_gain=K, virtual_flip=False, stale_horizon_s=horizon, det_hold_s=horizon)
    t0, far, near = 1_000_000_000, _pose_dead_ahead(20.0), _pose_dead_ahead(6.0, vert_body=-3.0)

    stale = EgoObsBuilder(EgoObsBuilderConfig(**cfg))
    stale.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                 gyro_frd=np.zeros(3), pose=far, last_normed_thrust=0.0)
    stale.update(sim_time_ns=t0 + int(1.5 * horizon * 1e9), gate_index=0, R_frd2ned=np.eye(3),
                 vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=near, last_normed_thrust=0.0)
    ref = EgoObsBuilder(EgoObsBuilderConfig(virtual_flip=False))        # a pure snap
    ref.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
               gyro_frd=np.zeros(3), pose=near, last_normed_thrust=0.0)
    np.testing.assert_allclose(_held(stale), _held(ref), atol=1e-12)    # SNAPPED, prior discarded

    fresh = EgoObsBuilder(EgoObsBuilderConfig(**cfg))
    fresh.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                 gyro_frd=np.zeros(3), pose=far, last_normed_thrust=0.0)
    fresh.update(sim_time_ns=t0 + int(0.5 * horizon * 1e9), gate_index=0, R_frd2ned=np.eye(3),
                 vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=near, last_normed_thrust=0.0)
    assert abs(float(_held(fresh)[0]) - 20.0) < abs(float(_held(fresh)[0]) - 6.0)   # still near 20


def test_fix_gain_applies_to_slot1_too():
    """The gain is per-slot and identical: slot1 (next gate) blends by the same rule as slot0."""
    K = 0.25
    cfg = dict(virtual_flip=False, slot1_enabled=True, stale_horizon_s=1.0, det_hold_s=1.0)
    t0, dt_ns = 1_000_000_000, 33_000_000
    p0, p1 = _pose_dead_ahead(24.0), _pose_dead_ahead(20.0, vert_body=-2.0)
    snap = EgoObsBuilder(EgoObsBuilderConfig(**cfg))
    blend = EgoObsBuilder(EgoObsBuilderConfig(fix_gain=K, **cfg))
    for b in (snap, blend):
        b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                 gyro_frd=np.zeros(3), pose=_pose_dead_ahead(10.0), next_pose=p0,
                 last_normed_thrust=0.0)
    prior = np.asarray(snap.last_diag["rel_flu1"], dtype=np.float64)    # zero velocity => no drift
    for b in (snap, blend):
        b.update(sim_time_ns=t0 + dt_ns, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                 gyro_frd=np.zeros(3), pose=_pose_dead_ahead(9.0), next_pose=p1,
                 last_normed_thrust=0.0)
    fix1 = np.asarray(snap.last_diag["rel_flu1"], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(blend.last_diag["rel_flu1"], dtype=np.float64),
                               (1.0 - K) * prior + K * fix1, atol=1e-12)


def test_fix_gain_is_clamped_to_unit_interval():
    """Out-of-range gains clamp like training's ``(1/n_eff).clamp(0,1)``: >1 snaps, <0 holds."""
    t0, p0, p1 = 1_000_000_000, _pose_dead_ahead(10.0), _pose_dead_ahead(6.0, vert_body=-2.0)
    cfg = dict(virtual_flip=False, stale_horizon_s=1.0, det_hold_s=1.0)
    hi = EgoObsBuilder(EgoObsBuilderConfig(fix_gain=5.0, **cfg))
    lo = EgoObsBuilder(EgoObsBuilderConfig(fix_gain=-1.0, **cfg))
    for b in (hi, lo):
        b.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                 gyro_frd=np.zeros(3), pose=p0, last_normed_thrust=0.0)
    prior_lo = _held(lo).copy()
    for b in (hi, lo):
        b.update(sim_time_ns=t0 + 33_000_000, gate_index=0, R_frd2ned=np.eye(3),
                 vel_ned=np.zeros(3), gyro_frd=np.zeros(3), pose=p1, last_normed_thrust=0.0)
    ref = EgoObsBuilder(EgoObsBuilderConfig(**cfg))
    ref.update(sim_time_ns=t0, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
               gyro_frd=np.zeros(3), pose=p1, last_normed_thrust=0.0)
    np.testing.assert_allclose(_held(hi), _held(ref), atol=1e-12)       # >1 -> snap
    np.testing.assert_allclose(_held(lo), prior_lo, atol=1e-12)         # <0 -> ignore the fix


def test_fix_gain_one_is_bitwise_identical_to_the_snap_over_a_mixed_sequence():
    """The byte-identity guarantee end to end: at the default gain every emitted obs must be
    BITWISE equal to the untouched path -- across fixes, gaps past det_hold, re-acquisitions and
    gate advances. (Empirically confirmed against the pre-change module over 50 real flights by
    scripts/replay_fix_gain.py --check-identical; this is the in-tree tripwire.)"""
    rng = np.random.default_rng(7)
    a = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    b = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True, fix_gain=1.0))
    t = 1_000_000_000
    for k in range(400):
        t += 33_000_000
        gi = k // 150                                   # two gate advances over the sequence
        seen = (k % 7) < 4                              # fixes with realistic gaps
        pose = (_pose_dead_ahead(12.0 - 0.02 * (k % 150), lateral=float(rng.normal(0, 0.3)),
                                 vert_body=float(rng.normal(0, 0.3))) if seen else None)
        nxt = _pose_dead_ahead(26.0, lateral=float(rng.normal(0, 0.5))) if (k % 5) == 0 else None
        kw = dict(sim_time_ns=t, gate_index=gi, R_frd2ned=np.eye(3),
                  vel_ned=np.array([6.0, 0.2, -0.1]), gyro_frd=np.array([0.1, -0.2, 0.3]),
                  pose=pose, next_pose=nxt, last_normed_thrust=1.0)
        assert a.update(**kw).tobytes() == b.update(**kw).tobytes(), f"diverged at tick {k}"
