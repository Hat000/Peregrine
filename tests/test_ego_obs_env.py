"""Tests for the VQ2 EGOCENTRIC obs env (component C, rl/peregrine_racing_ego.py).

The PeregrineRacingEgo class itself needs diffaero (cluster-only), but every decision it delegates to
is a PURE torch function tested here (the same pattern as tests/test_peregrine_racing_core.py):
  * build_coarse_map / set-override      -- the static per-gate 3x3 sector, auto-fill + overridable
  * ego_window_indices                    -- target-relative [current,next,next-next], clamped + masked
  * ego_actor_obs                         -- the 26-dim position-free actor obs + window promotion
  * ego_critic_state                      -- the privileged GT critic layout
  * apply_contact_kill + crossing_events  -- HARD kill-on-contact (done + large negative)

Coverage (prompt task 4):
  (a) obs shape == 26, fixed across steps/gate-counts; critic GT layout as documented.
  (b) HANDOFF: target advances -> window promotes (tg+1 -> slot 0) + coarse_sector switches to
      sector[tg]; a gate's rel_pos does NOT teleport across the promotion.
  (c) COARSE MAP: sector[tg] fed matches the prebuilt index; auto-fill buckets a known geometry;
      overridable.
  (d) VISIBLE_AREA in [0,1], ~1 head-on, small at a sharp angle, masked (0) undetectable.
  (e) KILL ON CONTACT: a contact step terminates (done) with a large negative reward.
  (f) NO WORLD POSITION: translate the whole scene (drone + all gates) + global yaw -> actor obs
      UNCHANGED.
  (g) MASKING: an out-of-window / undetectable slot is zeros + confidence 0 + visible_area 0.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_ego_obs_env.py -q
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import ego_estimator as EE                                                 # noqa: E402
import peregrine_racing_ego as C                                          # noqa: E402
from peregrine_racing import crossing_events, world_to_gateframe          # noqa: E402
from inc8_estimator_emul import quat_xyzw_to_matrix_torch                 # noqa: E402

DT = torch.float64


# ---- builders -----------------------------------------------------------------------------------
def _identity_quat(n=1):
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DT)
    return q.unsqueeze(0).expand(n, 4).contiguous()


def _yaw_pitch_roll_quat(yaw=0.0, pitch=0.0, roll=0.0, n=1):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = torch.tensor([x, y, z, w], dtype=DT)
    return q.unsqueeze(0).expand(n, 4).contiguous()


def _make_est(n, gate_pos, gate_yaw, cfg=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return EE.BatchedEgoEstimator(n, gate_pos, gate_yaw, config=cfg or EE.EgoEstimatorConfig(),
                                  device=torch.device("cpu"), dtype=DT, generator=gen)


def _straight_course(n, G, spacing=12.0):
    """A straight course along +x at the given spacing, all gates facing -x (yaw=pi), spawn at origin."""
    xs = spacing * (1.0 + torch.arange(G, dtype=DT))
    gate_pos = torch.stack([xs, torch.zeros(G, dtype=DT), torch.zeros(G, dtype=DT)], dim=-1)
    gate_pos = gate_pos.unsqueeze(0).expand(n, G, 3).contiguous()
    gate_yaw = torch.zeros(n, G, dtype=DT)
    spawn = torch.zeros(n, 3, dtype=DT)
    return gate_pos, gate_yaw, spawn


# ================================================================================================
# (a) obs shape 26 fixed across steps / gate counts; critic layout.
# ================================================================================================
@pytest.mark.parametrize("G", [1, 2, 3, 5, 8])
def test_a_obs_shape_fixed_across_gate_counts(G):
    N = 16
    gate_pos, gate_yaw, spawn = _straight_course(N, G)
    est = _make_est(N, gate_pos, gate_yaw)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.randn(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    detect = torch.ones(N, G, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1 / 30, detectable=detect, prev_quat=q)
    sector = C.build_coarse_map(gate_pos, spawn)
    tg = torch.zeros(N, dtype=torch.long)
    last_coll = torch.zeros(N, dtype=DT)
    obs = C.ego_actor_obs(out, detect, tg, last_coll, sector, G)
    assert obs.shape == (N, C.EGO_OBS_DIM) == (N, 21), obs.shape
    # fixed across several steps + advancing target
    for step_i in range(5):
        out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1 / 30, detectable=detect, prev_quat=q)
        tg = torch.clamp(tg + (step_i % 2), max=G - 1)
        obs = C.ego_actor_obs(out, detect, tg, last_coll, sector, G)
        assert obs.shape == (N, 21), (G, step_i, obs.shape)


def test_a_critic_state_layout():
    """Critic GT layout (EGO_CRITIC_DIM=16): [v(3), rp(2), rates(3), then per slot rel_pos(3)+conf(1)].
    The rel_pos entries are TRUE (equal the exact GT rel_pos, not the noised actor rel_pos)."""
    N, G = 8, 4
    gate_pos, gate_yaw, spawn = _straight_course(N, G)
    q = _yaw_pitch_roll_quat(yaw=0.3, pitch=-0.1, roll=0.05, n=N)
    pos = torch.randn(N, 3, dtype=DT)
    vel = torch.randn(N, 3, dtype=DT)
    rates = torch.randn(N, 3, dtype=DT) * 0.1
    est = _make_est(N, gate_pos, gate_yaw, seed=3)
    est.reset_idx(torch.arange(N), pos, vel, q)
    detect = torch.ones(N, G, dtype=torch.bool)
    est.step(pos, vel, q, rates, dt=1 / 30, detectable=detect, prev_quat=q)
    e = est.estimate()

    R_wb = quat_xyzw_to_matrix_torch(q)
    rel_true = C.true_rel_pos_body(gate_pos, pos, R_wb)                    # (N,G,3)
    vel_body_true = torch.einsum("nji,nj->ni", R_wb, vel)
    tg = torch.full((N,), 1, dtype=torch.long)                            # target = gate 1
    state = C.ego_critic_state(rel_true, vel_body_true, e.roll_pitch, rates, e.confidence, tg, G)
    assert state.shape == (N, C.EGO_CRITIC_DIM) == (N, 16), state.shape
    # [0:3] true body velocity
    assert torch.allclose(state[:, 0:3], vel_body_true), "critic vel must be TRUE body velocity"
    # [3:5] roll_pitch ; [5:8] rates (true)
    assert torch.allclose(state[:, 5:8], rates), "critic rates must be the true rates"
    # window slot 0 rel_pos = TRUE rel_pos of gate tg=1 (NOT the noised actor value)
    slot0_relpos = state[:, 8:11]
    assert torch.allclose(slot0_relpos, rel_true[:, 1, :]), "critic slot0 must be TRUE rel_pos of tg"
    # slot 1 = gate 2 (WINDOW=2: current=gate1, next=gate2)
    assert torch.allclose(state[:, 12:15], rel_true[:, 2, :])
    print(f"\n[a] critic state layout OK, dim={state.shape[1]} (GT rel_pos + true v/rp/rates + conf)")


# ================================================================================================
# (b) HANDOFF: window promotes (tg+1 -> slot 0), sector switches, NO teleport.
# ================================================================================================
def test_b_handoff_promotes_window_no_teleport():
    N, G = 4, 5
    gate_pos, gate_yaw, spawn = _straight_course(N, G, spacing=12.0)
    est = _make_est(N, gate_pos, gate_yaw, seed=1)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    detect = torch.ones(N, G, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1 / 30, detectable=detect, prev_quat=q)
    sector = C.build_coarse_map(gate_pos, spawn)
    last_coll = torch.zeros(N, dtype=DT)

    tg0 = torch.zeros(N, dtype=torch.long)
    obs0 = C.ego_actor_obs(out, detect, tg0, last_coll, sector, G)
    tg1 = torch.ones(N, dtype=torch.long)
    obs1 = C.ego_actor_obs(out, detect, tg1, last_coll, sector, G)

    # slot layout: [9:11]=sector, then slot0 @ [11:16], slot1 @ [16:21]. rel_pos = first 3.
    def slot_relpos(obs, k):
        base = 11 + 5 * k
        return obs[:, base:base + 3]

    # NO TELEPORT: gate 1's rel_pos was slot1 at tg=0; after handoff it is slot0 at tg=1 -- SAME value.
    g1_before = slot_relpos(obs0, 1)          # gate 1 in slot 1 (next) at tg=0
    g1_after = slot_relpos(obs1, 0)           # gate 1 in slot 0 (current) at tg=1
    d = (g1_before - g1_after).abs().max().item()
    print(f"\n[b] gate-1 rel_pos slot1(tg0) vs slot0(tg1) max|delta| = {d:.2e} (no teleport)")
    assert d < 1e-12, d
    # (WINDOW=2: gate 2 is not in the window at tg=0; it enters slot1 fresh at tg=1 -- nothing to promote)

    # coarse_sector switches from sector[0] to sector[1]
    sec_before = obs0[:, 9:11]
    sec_after = obs1[:, 9:11]
    assert torch.allclose(sec_before, sector[:, 0, :].to(DT)), "tg=0 feeds sector[0]"
    assert torch.allclose(sec_after, sector[:, 1, :].to(DT)), "tg=1 feeds sector[1]"
    print(f"[b] coarse_sector switched sector[0]->sector[1] on handoff")


# ================================================================================================
# (c) COARSE MAP: sector[tg] fed matches; auto-fill buckets a known geometry; overridable.
# ================================================================================================
def test_c_coarse_map_autofill_and_override():
    N = 3
    # known geometry: g0 straight; at g1 the course turns LEFT (+y) and DOWN; at g2 turns RIGHT + UP.
    gate_pos = torch.tensor([[
        [10.0, 0.0, 0.0],     # g0 (ahead)
        [20.0, 6.0, -3.0],    # g1: leg g0->g1 goes +y (left) & -z (down)
        [30.0, 6.0, 2.0],     # g2: leg g1->g2 goes +x only (no turn) & +z (up)
    ]], dtype=DT).expand(N, 3, 3).contiguous()
    gate_yaw = torch.zeros(N, 3, dtype=DT)
    spawn = torch.zeros(N, 3, dtype=DT)
    sector = C.build_coarse_map(gate_pos, spawn)                          # (N,3,2) in {-1,0,1}
    assert sector.shape == (N, 3, 2)
    assert set(sector.unique().tolist()) <= {-1, 0, 1}

    # g0: incoming spawn->g0 = (10,0,0); outgoing g0->g1 = (10,6,-3): turn LEFT (+) , outgoing elev DOWN.
    assert sector[0, 0, 0].item() == 1, sector[0, 0].tolist()            # horiz LEFT
    assert sector[0, 0, 1].item() == -1, sector[0, 0].tolist()          # vert DOWN
    # g1: incoming (10,6,-3); outgoing g1->g2 = (10,0,5): turn back RIGHT (-), outgoing elev UP.
    assert sector[0, 1, 0].item() == -1, sector[0, 1].tolist()          # horiz RIGHT
    assert sector[0, 1, 1].item() == 1, sector[0, 1].tolist()           # vert UP
    # g2 (last): outgoing reuses incoming (10,0,5) -> no turn (horiz 0), elev UP.
    assert sector[0, 2, 0].item() == 0, sector[0, 2].tolist()
    assert sector[0, 2, 1].item() == 1, sector[0, 2].tolist()
    print(f"\n[c] auto-fill sectors gate0={sector[0,0].tolist()} gate1={sector[0,1].tolist()} "
          f"gate2={sector[0,2].tolist()}")

    # sector[tg] is what the obs feeds -- verify via ego_actor_obs for a couple of targets.
    est = _make_est(N, gate_pos, gate_yaw, seed=5)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    detect = torch.ones(N, 3, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1 / 30, detectable=detect, prev_quat=q)
    for tg_i in range(3):
        tg = torch.full((N,), tg_i, dtype=torch.long)
        obs = C.ego_actor_obs(out, detect, tg, torch.zeros(N, dtype=DT), sector, 3)
        assert torch.allclose(obs[:, 9:11], sector[:, tg_i, :].to(DT)), tg_i

    # OVERRIDE: a human-set sector is fed instead of the auto-fill.
    override = sector.clone()
    override[:, 1, :] = torch.tensor([1, -1])                            # force g1 sector
    tg = torch.ones(N, dtype=torch.long)
    obs = C.ego_actor_obs(out, detect, tg, torch.zeros(N, dtype=DT), override, 3)
    assert torch.allclose(obs[:, 9:11], torch.tensor([1.0, -1.0], dtype=DT).expand(N, 2))
    print(f"[c] override honored: tg=1 sector -> {obs[0,9:11].tolist()}")


# ================================================================================================
# (d) VISIBLE_AREA in obs: in [0,1], head-on ~1, masked 0 undetectable (channel is obs slot [.,4]).
# ================================================================================================
def test_d_visible_area_in_obs():
    N, G = 2000, 1
    gate_pos = torch.tensor([[[15.0, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)                                # normal +x, head-on
    cfg = EE.EgoEstimatorConfig(miss_prob=0.0, teleport_prob=0.0, visible_area_sigma=0.05)
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=7)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    detect = torch.ones(N, 1, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1 / 30, detectable=detect, prev_quat=q)
    sector = C.build_coarse_map(gate_pos, torch.zeros(N, 3, dtype=DT))
    tg = torch.zeros(N, dtype=torch.long)
    obs = C.ego_actor_obs(out, detect, tg, torch.zeros(N, dtype=DT), sector, 1)
    # slot0 visible_area is the 5th element of slot0 -> index 11 + 4 = 15
    va = obs[:, 15]
    assert (va >= 0.0).all() and (va <= 1.0).all(), "visible_area in obs must be in [0,1]"
    assert va.mean().item() > 0.9, va.mean().item()                      # head-on ~1
    # undetectable -> masked 0 in obs
    obs_masked = C.ego_actor_obs(out, torch.zeros(N, 1, dtype=torch.bool), tg,
                                 torch.zeros(N, dtype=DT), sector, 1)
    assert obs_masked[:, 15].abs().max().item() == 0.0, "undetectable -> visible_area 0 in obs"
    print(f"\n[d] obs visible_area head-on mean={va.mean().item():.3f}, undetectable=0")


# ================================================================================================
# (e) KILL ON CONTACT: a contact step terminates (done) with a large negative reward.
# ================================================================================================
def test_e_kill_on_contact_terminates_with_large_negative():
    """Reuse the env's REAL contact classification (crossing_events on the true frame geometry) + the
    env's REAL kill helper (apply_contact_kill). A trajectory that threads the FRAME MATERIAL (in-frame
    band, not the aperture) -> gate_collision -> terminated AND a large negative reward that step."""
    HALF_IN, HALF_OUT = 0.75, 1.36
    penalty = C.EGO_CONTACT_PENALTY_DEFAULT
    # single gate at origin (gate frame). CONTACT: cross the plane at |y| in (0.75, 1.36] = frame band.
    N = 3
    rel_prev = torch.tensor([[[-0.3, 1.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()  # frame band
    rel_curr = torch.tensor([[[0.3, 1.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    ev = crossing_events(rel_prev, rel_curr, HALF_IN, HALF_OUT)
    # env's target-gate collision derivation (single gate -> the target IS gate 0)
    tg = torch.zeros(N, dtype=torch.long)
    ar = torch.arange(N)
    frame_target = (ev["fwd"] | ev["bwd"])[ar, tg] & ev["in_frame"][ar, tg]
    gate_collision = frame_target                                        # (N,) bool
    assert gate_collision.all(), "threading the frame band must be a contact"

    # termination (env: terminated |= gate_collision) + reward penalty (env: apply_contact_kill)
    base_reward = torch.ones(N, dtype=DT) * 2.0                          # some positive base reward
    reward = C.apply_contact_kill(base_reward, gate_collision, penalty)
    terminated = gate_collision                                          # env ORs this into `terminated`
    assert terminated.all(), "contact -> done"
    assert (reward < -0.5 * penalty).all(), reward.tolist()             # LARGE negative that step
    print(f"\n[e] contact -> done={terminated.tolist()}, reward={reward[0].item():.1f} "
          f"(base 2.0 - penalty {penalty})")

    # CONTRAST: a clean center pass is NOT a contact and keeps its positive reward.
    rp = torch.tensor([[[-0.3, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    rc = torch.tensor([[[0.3, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    ev2 = crossing_events(rp, rc, HALF_IN, HALF_OUT)
    coll2 = ((ev2["fwd"] | ev2["bwd"])[ar, tg] & ev2["in_frame"][ar, tg])
    assert not coll2.any(), "a center pass is not a contact"
    rew2 = C.apply_contact_kill(base_reward, coll2, penalty)
    assert torch.allclose(rew2, base_reward), "no contact -> reward unchanged (no soft penalty)"
    print(f"[e] clean pass: contact={coll2.tolist()}, reward unchanged={rew2[0].item():.1f}")


# ================================================================================================
# (f) NO WORLD POSITION: translate the whole scene (drone + all gates) + global yaw -> obs UNCHANGED.
# ================================================================================================
def test_f_no_world_position_translation_and_yaw_invariance():
    N, G = 32, 4
    torch.manual_seed(0)
    base_gate = torch.randn(N, G, 3, dtype=DT) * 4.0 + torch.tensor([14.0, 0.0, 1.0], dtype=DT)
    base_yaw = torch.rand(N, G, dtype=DT) * 0.4
    base_pos = torch.randn(N, 3, dtype=DT) * 2.0
    base_vel = torch.randn(N, 3, dtype=DT)
    rates = torch.randn(N, 3, dtype=DT) * 0.1
    detect = torch.ones(N, G, dtype=torch.bool)
    tg = torch.randint(0, G, (N,))
    last_coll = torch.rand(N, dtype=DT)

    def Rz(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DT)

    def run(offset, yaw_off, rotate):
        gp = base_gate.clone()
        gy = base_yaw + yaw_off
        pos = base_pos.clone()
        vel = base_vel.clone()
        q = _yaw_pitch_roll_quat(yaw=0.2 + yaw_off, pitch=-0.12, roll=0.08, n=N)
        if rotate:
            R = Rz(yaw_off)
            gp = torch.einsum("ij,ngj->ngi", R, gp)
            pos = torch.einsum("ij,nj->ni", R, pos)
            vel = torch.einsum("ij,nj->ni", R, vel)
        gp = gp + offset
        pos = pos + offset
        spawn = torch.zeros(N, 3, dtype=DT)
        est = _make_est(N, gp, gy, EE.EgoEstimatorConfig(), seed=123)
        est.reset_idx(torch.arange(N), pos, vel, q)
        out = est.step(pos, vel, q, rates, dt=1 / 30, detectable=detect, prev_quat=q)
        sector = C.build_coarse_map(gp, spawn + offset)
        return C.ego_actor_obs(out, detect, tg, last_coll, sector, G)

    zero = torch.zeros(3, dtype=DT)
    obs_a = run(zero, 0.0, rotate=False)
    obs_b = run(torch.tensor([111.0, -47.0, 6.0], dtype=DT), 0.0, rotate=False)     # pure translation
    d_tr = (obs_a - obs_b).abs().max().item()
    print(f"\n[f] pure translation: obs max|delta| = {d_tr:.2e}")
    assert d_tr < 1e-9, d_tr

    obs_c = run(torch.tensor([80.0, 20.0, -5.0], dtype=DT), 0.6, rotate=True)        # translation + yaw
    d_ty = (obs_a - obs_c).abs().max().item()
    print(f"[f] translation + global yaw: obs max|delta| = {d_ty:.2e}")
    assert d_ty < 1e-8, d_ty


# ================================================================================================
# (g) MASKING: out-of-window / undetectable slot -> zeros + confidence 0 + visible_area 0.
# ================================================================================================
def test_g_masking_out_of_window_and_undetectable():
    N, G = 5, 2                                                          # WINDOW=2: slot1 = tg+1
    gate_pos, gate_yaw, spawn = _straight_course(N, G)
    est = _make_est(N, gate_pos, gate_yaw, seed=9)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    detect = torch.ones(N, G, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1 / 30, detectable=detect, prev_quat=q)
    sector = C.build_coarse_map(gate_pos, spawn)

    def slot(obs, k):
        base = 11 + 5 * k
        return obs[:, base:base + 5]                                    # rel_pos(3)+conf(1)+area(1)

    # OUT-OF-WINDOW: target the LAST gate (tg=1) -> slot1 (tg+1 = 2 >= G=2) is past the course -> masked.
    tg_last = torch.ones(N, dtype=torch.long)
    obs_last = C.ego_actor_obs(out, detect, tg_last, torch.zeros(N, dtype=DT), sector, G)
    s_oob = slot(obs_last, 1)
    assert s_oob.abs().max().item() == 0.0, f"OOB slot must be all zeros, got {s_oob[0].tolist()}"
    print(f"\n[g] OOB slot (tg+1 past last gate) fully masked: {s_oob[0].tolist()}")

    # UNDETECTABLE gate 1 -> slot1 masked (zeros), while detectable slot0 stays non-zero.
    tg = torch.zeros(N, dtype=torch.long)
    det2 = detect.clone()
    det2[:, 1] = False
    obs2 = C.ego_actor_obs(out, det2, tg, torch.zeros(N, dtype=DT), sector, G)
    s1 = slot(obs2, 1)
    assert s1.abs().max().item() == 0.0, "undetectable slot -> zeros + conf 0 + area 0"
    assert slot(obs2, 0).abs().max().item() > 0.0, "detectable current slot stays non-zero"
    # confidence (index 3) and visible_area (index 4) of the masked slot are exactly 0
    assert s1[:, 3].abs().max().item() == 0.0 and s1[:, 4].abs().max().item() == 0.0
    print(f"[g] undetectable slot1 masked (conf 0, area 0); slot0 kept")
