"""Tests for the OBS-V2 APPEND-ONLY windowed coarse-map extension (2026-07-15, ``+env.ego_obs_v2``).

The v2 obs adds a SECOND sliding coarse-direction hint (the NEXT gate's sector) so the policy can
anticipate a gate before it is visible. It is knob-gated + APPEND-ONLY:
  * OFF (default) -> EGO_OBS_DIM==21, obs BYTE-IDENTICAL to today (obs_v2=False == the no-arg call).
  * ON -> EGO_OBS_DIM_V2==23; obs[0:21] BYTE-IDENTICAL to the 21-dim obs (incl. obs[9:11]=sector[tg],
    the CURRENT gate hint) + obs[21:23] APPENDED = sector[clamp(tg+1)], masked [0,0] past the last gate
    (the SAME valid[:,1] mask the rel-pos slot1 uses -> masks in lockstep, slides with no teleport).

Warm-transfer safety is verified separately: a 21-dim actor whose first layer is zero-padded to 23
(pad_actor_input_cols_zero) produces an IDENTICAL action to the unpadded 21-dim actor on the
zero-appended obs -- so the champion vpeffs0 warm-loads bit-for-bit at step 0.

The PeregrineRacingEgo class needs diffaero (cluster-only); every decision it delegates to is a PURE
torch function tested here (the test_ego_obs_env.py convention). Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_obs_v2.py -q
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn                                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import ego_estimator as EE                                                 # noqa: E402
import peregrine_racing_ego as C                                          # noqa: E402

DT = torch.float64


# ---- builders (mirror test_ego_obs_env.py) ------------------------------------------------------
def _identity_quat(n=1):
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DT)
    return q.unsqueeze(0).expand(n, 4).contiguous()


def _make_est(n, gate_pos, gate_yaw, cfg=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return EE.BatchedEgoEstimator(n, gate_pos, gate_yaw, config=cfg or EE.EgoEstimatorConfig(),
                                  device=torch.device("cpu"), dtype=DT, generator=gen)


def _turning_course(n, G):
    """A course that genuinely turns/climbs at every gate so sector[g] is non-trivial (mixes -1/0/1)."""
    pts = [[10.0, 0.0, 0.0]]
    steps = [(10.0, 6.0, -3.0), (10.0, -6.0, 2.0), (10.0, 5.0, 4.0), (10.0, -4.0, -2.0),
             (10.0, 3.0, 1.0), (10.0, -3.0, -1.0), (10.0, 2.0, 2.0)]
    for k in range(1, G):
        px, py, pz = pts[-1]
        dx, dy, dz = steps[(k - 1) % len(steps)]
        pts.append([px + dx, py + dy, pz + dz])
    gate_pos = torch.tensor(pts[:G], dtype=DT).unsqueeze(0).expand(n, G, 3).contiguous()
    gate_yaw = torch.zeros(n, G, dtype=DT)
    spawn = torch.zeros(n, 3, dtype=DT)
    return gate_pos, gate_yaw, spawn


def _stepped_est(N, gate_pos, gate_yaw, seed=0):
    est = _make_est(N, gate_pos, gate_yaw, seed=seed)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.randn(N, 3, generator=torch.Generator().manual_seed(seed + 1), dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    G = gate_pos.shape[1]
    detect = torch.ones(N, G, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1 / 30, detectable=detect, prev_quat=q)
    return out, detect


# ================================================================================================
# (1) OFF (default) -> dim 21, byte-identical; ON -> dim 23.
# ================================================================================================
def test_off_is_byte_identical_and_dims():
    assert C.EGO_OBS_DIM == 21
    assert C.EGO_OBS_DIM_V2 == 23
    assert C.EGO_OBS_V2_EXTRA == 2

    N, G = 8, 4
    gate_pos, gate_yaw, spawn = _turning_course(N, G)
    out, detect = _stepped_est(N, gate_pos, gate_yaw, seed=2)
    sector = C.build_coarse_map(gate_pos, spawn)
    tg = torch.full((N,), 1, dtype=torch.long)
    last_coll = torch.zeros(N, dtype=DT)

    obs_default = C.ego_actor_obs(out, detect, tg, last_coll, sector, G)              # no obs_v2 arg
    obs_off = C.ego_actor_obs(out, detect, tg, last_coll, sector, G, obs_v2=False)
    assert obs_default.shape == (N, 21)
    assert obs_off.shape == (N, 21)
    # default call == explicit OFF, bit-for-bit (the knob defaults False -> baseline HEAD behaviour).
    assert torch.equal(obs_default, obs_off), "obs_v2=False must be byte-identical to the no-arg call"

    obs_on = C.ego_actor_obs(out, detect, tg, last_coll, sector, G, obs_v2=True)
    assert obs_on.shape == (N, 23)
    # obs[0:21] BYTE-IDENTICAL between ON and OFF (append-only; obs[9:11] still sector[tg]).
    assert torch.equal(obs_on[:, :21], obs_off), "obs_v2 ON obs[0:21] must equal the 21-dim obs exactly"
    print(f"\n[1] OFF dim={obs_off.shape[1]} == default; ON dim={obs_on.shape[1]}; obs[0:21] identical")


# ================================================================================================
# (2) ON -> obs[21:23] == sector[clamp(tg+1)], [0,0] at the last gate.
# ================================================================================================
def test_on_next_sector_and_last_gate_mask():
    N, G = 6, 5
    gate_pos, gate_yaw, spawn = _turning_course(N, G)
    out, detect = _stepped_est(N, gate_pos, gate_yaw, seed=4)
    sector = C.build_coarse_map(gate_pos, spawn)
    last_coll = torch.zeros(N, dtype=DT)

    for tg_i in range(G):
        tg = torch.full((N,), tg_i, dtype=torch.long)
        obs = C.ego_actor_obs(out, detect, tg, last_coll, sector, G, obs_v2=True)
        nxt = obs[:, 21:23]
        if tg_i < G - 1:
            expected = sector[:, tg_i + 1, :].to(DT)
            assert torch.equal(nxt, expected), (tg_i, nxt[0].tolist(), expected[0].tolist())
        else:
            assert nxt.abs().max().item() == 0.0, f"last gate: next-sector must be [0,0], got {nxt[0].tolist()}"
    print(f"\n[2] obs[21:23]==sector[tg+1] for tg<{G-1}; [0,0] masked at the last gate")

    # current-gate hint obs[9:11] is UNCHANGED by obs_v2 (still sector[tg]) -- the append does not reindex.
    tg = torch.full((N,), 2, dtype=torch.long)
    obs = C.ego_actor_obs(out, detect, tg, last_coll, sector, G, obs_v2=True)
    assert torch.equal(obs[:, 9:11], sector[:, 2, :].to(DT)), "obs[9:11] must stay the CURRENT gate sector"


# ================================================================================================
# (3) SLIDE: advancing the target moves the next-gate hint with NO teleport; masks past the last gate.
# ================================================================================================
def test_slide_next_hint_promotes_no_teleport():
    N, G = 4, 5
    gate_pos, gate_yaw, spawn = _turning_course(N, G)
    out, detect = _stepped_est(N, gate_pos, gate_yaw, seed=6)
    sector = C.build_coarse_map(gate_pos, spawn)
    last_coll = torch.zeros(N, dtype=DT)

    def obs_at(tg_i):
        tg = torch.full((N,), tg_i, dtype=torch.long)
        return C.ego_actor_obs(out, detect, tg, last_coll, sector, G, obs_v2=True)

    # NO TELEPORT / PROMOTION: the NEXT-gate hint at tg becomes the CURRENT-gate hint at tg+1 unchanged
    # (both read sector[tg+1]) -- exactly the rel-pos window's promotion, for the coarse sector.
    for tg_i in range(G - 1):
        nxt_before = obs_at(tg_i)[:, 21:23]        # next-gate hint while targeting tg_i
        cur_after = obs_at(tg_i + 1)[:, 9:11]      # current-gate hint after advancing to tg_i+1
        d = (nxt_before - cur_after).abs().max().item()
        assert d == 0.0, f"tg={tg_i}: next-hint should promote to current-hint unchanged, |delta|={d}"
    # and the last gate's next-hint stays masked.
    assert obs_at(G - 1)[:, 21:23].abs().max().item() == 0.0
    print(f"\n[3] next-gate hint promotes to the current-gate hint with no teleport across all {G-1} advances")


# ================================================================================================
# (4) WARM-PAD: a 21-dim actor zero-padded to 23 gives an IDENTICAL action on the zero-appended obs.
# ================================================================================================
class _Blk(nn.Module):
    """Mirrors the diffaero MLP block key path so the first-layer weight is ``head.0.linear.weight``."""
    def __init__(self, a, b):
        super().__init__()
        self.linear = nn.Linear(a, b)

    def forward(self, x):
        return torch.relu(self.linear(x))


class _TinyActor(nn.Module):
    def __init__(self, in_dim, H=16, out=4):
        super().__init__()
        self.head = nn.Sequential(_Blk(in_dim, H), _Blk(H, H), nn.Linear(H, out))

    def forward(self, x):
        return torch.tanh(self.head(x))     # test-mode action = tanh(actor_mean(obs)) (fly_rl.policy_step)


def test_warm_pad_zero_cols_gives_identical_action():
    torch.manual_seed(0)
    N = 32
    actor21 = _TinyActor(C.EGO_OBS_DIM).to(DT)
    sd21 = actor21.state_dict()
    assert "head.0.linear.weight" in sd21 and sd21["head.0.linear.weight"].shape[1] == 21

    # pad the first layer with EGO_OBS_V2_EXTRA ZERO input columns (the pure warm-pad helper).
    sd23 = C.pad_actor_input_cols_zero(sd21, C.EGO_OBS_V2_EXTRA)
    # input dict NOT mutated; padded weight is a fresh tensor of the right shape with zero new columns.
    assert sd21["head.0.linear.weight"].shape[1] == 21, "input state_dict must not be mutated"
    w23 = sd23["head.0.linear.weight"]
    assert w23.shape == (16, 23)
    assert torch.equal(w23[:, :21], sd21["head.0.linear.weight"]), "first 21 cols preserved"
    assert w23[:, 21:].abs().max().item() == 0.0, "appended cols must be ZERO"
    # every other tensor carried through unchanged.
    for k in sd21:
        if k != "head.0.linear.weight":
            assert torch.equal(sd23[k], sd21[k]), k

    # load the padded state_dict STRICTLY into a genuine 23-dim actor (proves key compatibility).
    actor23 = _TinyActor(C.EGO_OBS_DIM_V2).to(DT)
    missing = actor23.load_state_dict(sd23, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys, missing

    # IDENTICAL action: obs' = [obs(21) | 0 0] through the 23-dim padded net == obs through the 21-dim net.
    # W' @ [obs|0 0] == W @ obs in EXACT arithmetic (the appended columns are exactly 0.0 -> contribute
    # nothing); the only residual is float REDUCTION-ORDER noise (BLAS blocks a width-23 dot product
    # differently than width-21, so the two exactly-zero extra terms shift rounding by ~machine-eps). At
    # float64 that is <1e-15; the pad is bit-for-bit correct by construction, not merely close.
    obs21 = torch.randn(N, 21, dtype=DT)
    obs23 = torch.cat([obs21, torch.zeros(N, C.EGO_OBS_V2_EXTRA, dtype=DT)], dim=1)
    with torch.no_grad():
        a21 = actor21(obs21)
        a23 = actor23(obs23)
    d = (a21 - a23).abs().max().item()
    assert d < 1e-15, f"padded 23-dim actor must reproduce the 21-dim action (reduction-order eps), max|delta|={d}"
    print(f"\n[4] warm-pad: 23-dim padded action == 21-dim action, max|delta|={d:.1e} (machine-eps)")


def test_pad_helper_missing_key_raises():
    with pytest.raises(KeyError):
        C.pad_actor_input_cols_zero({"not.the.key": torch.zeros(4, 21)}, 2)
    # n_new_cols <= 0 -> unchanged shallow copy (no-op pad).
    sd = {"head.0.linear.weight": torch.randn(8, 21)}
    out = C.pad_actor_input_cols_zero(sd, 0)
    assert torch.equal(out["head.0.linear.weight"], sd["head.0.linear.weight"])
