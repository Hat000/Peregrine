"""Tests for the OBS BLACKOUT COAST knob (``+env.ego_obs_coast`` / ego_actor_obs(..., obs_coast=)).

Motivation: handoff/audit-ego-inc9-2026-07-09/read_estimator-kf-audit.md STALE-CLIFF -- the obs builder
hard-masks each gate slot on INSTANTANEOUS detectability (keep = valid & det & conf>0), so the
estimator's coasted prior is ZEROED the moment the gate goes non-detectable and the crossing endgame is
flown on zeros. ``obs_coast`` drops the instantaneous ``det`` term so the coasted rel_pos + linearly-
decaying confidence feed the obs THROUGH a blackout, masking only past the estimator's stale horizon
(conf==0) -- the DESIGN.md §5.A propagate-then-mask intent.

Coverage (prompt task):
  (a) coast OFF  -> obs BIT-IDENTICAL to the legacy hard-mask behavior on a seeded synthetic rollout
      (driven directly through the estimator + obs builder; compared to an INDEPENDENT reference that
      re-implements the OLD keep = valid & det & (conf>0) masking).
  (b) coast ON   -> during an induced blackout: rel_pos NONZERO + confidence DECAYING while in-horizon,
      then ZERO (masked) past the stale horizon; coast OFF zeros the same slot immediately.

The PeregrineRacingEgo env needs diffaero (cluster-only); every masking decision is the PURE torch
function ego_actor_obs, driven here directly with a real BatchedEgoEstimator (same pattern as
tests/test_ego_obs_env.py). Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_ego_obs_coast.py -q
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import ego_estimator as EE                                                 # noqa: E402
import peregrine_racing_ego as C                                          # noqa: E402

DT = torch.float64
CPU = torch.device("cpu")


# ---- builders (mirror tests/test_ego_obs_env.py) ------------------------------------------------
def _identity_quat(n=1):
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DT)
    return q.unsqueeze(0).expand(n, 4).contiguous()


def _make_est(n, gate_pos, gate_yaw, cfg=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return EE.BatchedEgoEstimator(n, gate_pos, gate_yaw, config=cfg or EE.EgoEstimatorConfig(),
                                  device=CPU, dtype=DT, generator=gen)


def _straight_course(n, G, spacing=12.0):
    xs = spacing * (1.0 + torch.arange(G, dtype=DT))
    gate_pos = torch.stack([xs, torch.zeros(G, dtype=DT), torch.zeros(G, dtype=DT)], dim=-1)
    gate_pos = gate_pos.unsqueeze(0).expand(n, G, 3).contiguous()
    gate_yaw = torch.zeros(n, G, dtype=DT)
    spawn = torch.zeros(n, 3, dtype=DT)
    return gate_pos, gate_yaw, spawn


def _reference_hardmask_obs(est_out, detectable, target_gates, last_collective, sector, n_gates):
    """INDEPENDENT re-implementation of the LEGACY (pre-coast) obs masking: keep = valid & det & conf>0.
    A golden reference to prove ego_actor_obs(..., obs_coast=False) is byte-identical to the old code."""
    N = est_out.rel_pos.shape[0]
    dt_ = est_out.rel_pos.dtype
    ar = torch.arange(N)
    tg = target_gates.long()
    lc = last_collective.unsqueeze(-1) if last_collective.dim() == 1 else last_collective
    gidx, valid = C.ego_window_indices(tg, n_gates)
    slots = []
    for k in range(C.WINDOW):
        g = gidx[:, k]
        rel = est_out.rel_pos[ar, g]
        conf = est_out.confidence[ar, g]
        area = est_out.visible_area[ar, g]
        det = detectable[ar, g]
        keep = (valid[:, k] & det & (conf > 0.0)).to(dt_)
        slots.append(torch.cat([rel * keep.unsqueeze(-1), (conf * keep).unsqueeze(-1),
                                (area * keep).unsqueeze(-1)], dim=-1))
    coarse = sector[ar, tg].to(dt_)
    return torch.cat([est_out.velocity, est_out.roll_pitch, est_out.body_rates, lc, coarse, *slots], dim=-1)


# ================================================================================================
# (a) coast OFF == byte-identical legacy hard-mask (over a rollout exercising every mask branch).
# ================================================================================================
def test_a_coast_off_bit_identical_to_legacy_hardmask():
    """Drive a seeded rollout that toggles detectability (some gates blackout, some go STALE past the
    horizon) and assert ego_actor_obs(obs_coast=False) == the independent legacy-masking reference,
    bit for bit. Also assert the DEFAULT (obs_coast omitted) equals the explicit False."""
    N, G = 24, 3
    gate_pos, gate_yaw, spawn = _straight_course(N, G, spacing=11.0)
    est = _make_est(N, gate_pos, gate_yaw, seed=17)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.full((N, 3), 0.0, dtype=DT)
    vel[:, 0] = 3.0                                                        # move toward the gates
    rates = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    sector = C.build_coarse_map(gate_pos, spawn)
    last_coll = torch.rand(N, dtype=DT)

    torch.manual_seed(3)
    n_mismatch = 0
    for step_i in range(40):
        # a per-step, per-gate detectability mask that blacks some gates out (drives coast vs hard-mask
        # apart) and leaves others stale long enough to cross the 0.5 s horizon.
        detect = torch.rand(N, G) > (0.35 + 0.01 * step_i)
        out = est.step(pos, vel, q, rates, dt=1 / 30, detectable=detect, prev_quat=q)
        tg = torch.randint(0, G, (N,))
        ref = _reference_hardmask_obs(out, detect, tg, last_coll, sector, G)
        got_default = C.ego_actor_obs(out, detect, tg, last_coll, sector, G)              # default False
        got_explicit = C.ego_actor_obs(out, detect, tg, last_coll, sector, G, obs_coast=False)
        assert torch.equal(got_default, ref), f"default != legacy reference at step {step_i}"
        assert torch.equal(got_explicit, got_default), f"explicit False != default at step {step_i}"
        n_mismatch += int((got_default != ref).sum().item())
    print(f"\n[a] coast OFF bit-identical to legacy hard-mask over 40 seeded steps "
          f"(total elementwise mismatches={n_mismatch})")
    assert n_mismatch == 0


def test_a_coast_changes_something_when_blackout_with_live_prior():
    """Sanity: coast is NOT a no-op. Construct a state with a gate NON-detectable but conf>0 (a coast
    window) and show coast=True DIFFERS from coast=False (else test_a would be vacuous)."""
    N, G = 8, 1
    gate_pos, gate_yaw, spawn = _straight_course(N, G)
    cfg = EE.EgoEstimatorConfig(miss_prob=0.0, teleport_prob=0.0)
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=1)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    rates = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    det_on = torch.ones(N, G, dtype=torch.bool)
    est.step(pos, vel, q, rates, dt=1 / 30, detectable=det_on, prev_quat=q)   # a fix -> conf ~1
    det_off = torch.zeros(N, G, dtype=torch.bool)
    out = est.step(pos, vel, q, rates, dt=1 / 30, detectable=det_off, prev_quat=q)  # blackout, conf still >0
    tg = torch.zeros(N, dtype=torch.long)
    lc = torch.zeros(N, dtype=DT)
    off = C.ego_actor_obs(out, det_off, tg, lc, sector := C.build_coarse_map(gate_pos, spawn), G,
                          obs_coast=False)
    on = C.ego_actor_obs(out, det_off, tg, lc, sector, G, obs_coast=True)
    assert off[:, 11:14].abs().max().item() == 0.0, "coast OFF zeros the blacked-out slot"
    assert on[:, 11:14].abs().max().item() > 0.0, "coast ON keeps the coasted rel_pos"
    print(f"\n[a-sanity] blackout w/ live prior: coast OFF rel|max|={off[:,11:14].abs().max():.3e}, "
          f"coast ON rel|max|={on[:,11:14].abs().max():.3f} (coast is not a no-op)")


# ================================================================================================
# (b) coast ON: rel_pos nonzero + conf decaying through a blackout; zero past horizon.
# ================================================================================================
def test_b_coast_on_decays_through_blackout_then_masks():
    """Warm up with fixes (conf->1), then a sustained blackout. With coast ON: while in-horizon the
    current slot's rel_pos stays NONZERO and confidence DECAYS ~linearly toward 0; past the stale horizon
    (conf hits 0) the slot is fully MASKED. Coast OFF zeros the slot for the WHOLE blackout."""
    N, G = 4, 1
    gate_pos, gate_yaw, spawn = _straight_course(N, G, spacing=15.0)
    # noise-free, drift-free so rel_pos stays constant during the coast (v==0, no bias/teleport/miss);
    # isolates the confidence-decay + masking behavior.
    cfg = EE.EgoEstimatorConfig(miss_prob=0.0, teleport_prob=0.0, normal_flip_prob=0.0,
                                inject_bias=False, dr_accel_bias=False, stale_horizon_s=0.5)
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=2)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    rates = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    sector = C.build_coarse_map(gate_pos, spawn)
    tg = torch.zeros(N, dtype=torch.long)
    lc = torch.zeros(N, dtype=DT)
    dt = 1.0 / 30.0

    det_on = torch.ones(N, G, dtype=torch.bool)
    det_off = torch.zeros(N, G, dtype=torch.bool)
    for _ in range(5):                                                    # warm up: accept fixes
        est.step(pos, vel, q, rates, dt=dt, detectable=det_on, prev_quat=q)

    confs, relnorms, hard_relnorms = [], [], []
    n_black = 25                                                          # 25 * (1/30) s ~ 0.83 s > 0.5 horizon
    for _ in range(n_black):
        out = est.step(pos, vel, q, rates, dt=dt, detectable=det_off, prev_quat=q)
        obs_on = C.ego_actor_obs(out, det_off, tg, lc, sector, G, obs_coast=True)
        obs_off = C.ego_actor_obs(out, det_off, tg, lc, sector, G, obs_coast=False)
        confs.append(obs_on[0, 14].item())                               # slot0 confidence
        relnorms.append(obs_on[0, 11:14].norm().item())                  # slot0 rel_pos norm
        hard_relnorms.append(obs_off[0, 11:14].norm().item())
        # coast OFF: the slot is masked (zero) for the ENTIRE blackout, every step.
        assert obs_off[:, 11:15].abs().max().item() == 0.0

    # first blackout step: still in-horizon -> nonzero rel_pos + conf in (0,1).
    assert relnorms[0] > 1.0, relnorms[0]                                 # gate ~15 m away, coasted
    assert 0.0 < confs[0] < 1.0, confs[0]
    # confidence DECAYS monotonically while positive, at ~dt/horizon per step (linear ramp).
    per_step = dt / cfg.stale_horizon_s                                   # ~0.0667
    for i in range(1, len(confs)):
        if confs[i - 1] > 0.0 and confs[i] > 0.0:
            assert confs[i] < confs[i - 1] + 1e-12, (i, confs[i - 1], confs[i])
            assert abs((confs[i - 1] - confs[i]) - per_step) < 1e-6, (i, confs[i - 1], confs[i])
    # in-horizon steps keep rel_pos nonzero; the LAST blackout step is past the horizon -> fully masked.
    first_masked = next((i for i, c in enumerate(confs) if c == 0.0), None)
    assert first_masked is not None, "confidence never reached 0 within the blackout window"
    assert relnorms[first_masked] == 0.0, "past-horizon slot rel_pos must be masked to 0 (coast ON)"
    assert confs[-1] == 0.0 and relnorms[-1] == 0.0, "end of blackout is masked"
    # every in-horizon coast step held rel_pos NONZERO (the coast actually fed the policy).
    for i in range(first_masked):
        assert relnorms[i] > 0.0, (i, relnorms[i])
    print(f"\n[b] coast ON blackout: conf {confs[0]:.3f} -> ... -> 0 (masked at step {first_masked}); "
          f"rel_pos coasted nonzero while in-horizon (norm~{relnorms[0]:.2f}); "
          f"coast OFF rel_pos==0 for ALL {n_black} blackout steps (max {max(hard_relnorms):.1e})")


# ================================================================================================
# (c) STALE-HORIZON env plumbing (+env.ego_stale_horizon_s): default 0.5 preserved; override reaches
#     the estimator config and stretches the coast window (the 0.7-2.2 m terminal-blind-onset fix).
# ================================================================================================
def test_c_stale_horizon_default_and_env_plumbing():
    """(1) The default stale_horizon_s stays 0.5 (byte-identical). (2) The env __init__ plumbs the
    ``ego_stale_horizon_s`` cfg key into EgoEstimatorConfig(stale_horizon_s=...) -- PeregrineRacingEgo
    cannot be CONSTRUCTED without diffaero, so the wiring is asserted on the method source (the same
    pattern the latch-wiring test uses). (3) Behaviorally, a 1.2 s horizon keeps the coasted obs alive
    ~2.4x longer than 0.5 s under an identical blackout (the slow-lap crossing window the override buys)."""
    import inspect
    # (1) default preserved
    assert EE.EgoEstimatorConfig().stale_horizon_s == 0.5, "default must stay 0.5 (byte-identical)"
    # (2) env plumbing present: cfg key read + passed as the stale_horizon_s kwarg
    src = inspect.getsource(C.PeregrineRacingEgo.__init__)
    assert '"ego_stale_horizon_s"' in src, "__init__ must read the ego_stale_horizon_s cfg key"
    assert "stale_horizon_s=float(getattr(cfg, \"ego_stale_horizon_s\"" in src, \
        "the cfg key must be passed as EgoEstimatorConfig(stale_horizon_s=...)"

    # (3) behavioral: identical blackout, horizons 0.5 vs 1.2 -> coast survives ~2.4x longer.
    N, G = 4, 1
    gate_pos, gate_yaw, spawn = _straight_course(N, G, spacing=15.0)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    rates = torch.zeros(N, 3, dtype=DT)
    sector = C.build_coarse_map(gate_pos, spawn)
    tg = torch.zeros(N, dtype=torch.long)
    lc = torch.zeros(N, dtype=DT)
    dt = 1.0 / 30.0
    det_on = torch.ones(N, G, dtype=torch.bool)
    det_off = torch.zeros(N, G, dtype=torch.bool)

    def first_masked_step(horizon_s, n_black=60):
        cfg = EE.EgoEstimatorConfig(miss_prob=0.0, teleport_prob=0.0, normal_flip_prob=0.0,
                                    inject_bias=False, dr_accel_bias=False,
                                    stale_horizon_s=horizon_s)
        est = _make_est(N, gate_pos, gate_yaw, cfg, seed=4)
        est.reset_idx(torch.arange(N), pos, vel, q)
        for _ in range(5):
            est.step(pos, vel, q, rates, dt=dt, detectable=det_on, prev_quat=q)
        for i in range(n_black):
            out = est.step(pos, vel, q, rates, dt=dt, detectable=det_off, prev_quat=q)
            obs = C.ego_actor_obs(out, det_off, tg, lc, sector, G, obs_coast=True)
            if obs[0, 11:15].abs().max().item() == 0.0:                  # slot0 fully masked
                return i
        return n_black

    m05 = first_masked_step(0.5)
    m12 = first_masked_step(1.2)
    assert m05 < m12, (m05, m12)
    # linear staleness ramp -> the mask step scales ~ horizon/dt (15 vs 36 at 30 Hz).
    assert abs(m05 - round(0.5 / dt)) <= 1, m05
    assert abs(m12 - round(1.2 / dt)) <= 1, m12
    print(f"\n[c] stale-horizon plumbing OK: default 0.5 kept; coast masks at step {m05} (0.5 s) vs "
          f"{m12} (1.2 s) under the same blackout -- the override stretches the crossing window")


# ================================================================================================
# (d) MAX-RANGE SLOT-FILL CAP (pefcap 2026-07-12, ego_actor_obs slot_range_cap_m). +inf == OFF ==
#     byte-identical; a finite cap masks a slot whose ESTIMATED range exceeds it (far gate -> empty slot).
# ================================================================================================
def test_d_slot_range_cap_masks_far_gate_and_inf_is_byte_identical():
    N, G = 8, 3
    gate_pos, gate_yaw, spawn = _straight_course(N, G, spacing=12.0)       # gates at 12, 24, 36 m
    est = _make_est(N, gate_pos, gate_yaw, cfg=EE.EgoEstimatorConfig(noise_scale=0.0), seed=5)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT); vel = torch.zeros(N, 3, dtype=DT); rates = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    sector = C.build_coarse_map(gate_pos, spawn)
    lc = torch.zeros(N, dtype=DT)
    detect = torch.ones(N, G, dtype=torch.bool)                            # all in view
    out = est.step(pos, vel, q, rates, dt=1 / 30, detectable=detect, prev_quat=q)
    tg = torch.zeros(N, dtype=torch.long)                                  # slot0=gate0(12m), slot1=gate1(24m)
    d_default = C.ego_actor_obs(out, detect, tg, lc, sector, G)
    d_inf = C.ego_actor_obs(out, detect, tg, lc, sector, G, slot_range_cap_m=float("inf"))
    assert torch.equal(d_default, d_inf)                                   # +inf == OFF == byte-identical
    capped = C.ego_actor_obs(out, detect, tg, lc, sector, G, slot_range_cap_m=20.0)
    slot0, slot1 = capped[:, 11:16], capped[:, 16:21]                      # obs: ...sector[9:11], slot0[11:16], slot1[16:21]
    assert torch.all(slot1 == 0.0)                                         # gate1 (~24 m > 20) does NOT fill slot1
    assert torch.any(slot0 != 0.0)                                         # gate0 (~12 m < 20) still fills slot0
    assert torch.equal(slot0, d_default[:, 11:16])                         # only the FAR slot is dropped
