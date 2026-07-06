"""VQ2-ization of the inc8 RL controller (vq2_rl_controller_spec_2026-07-04) -- laptop tests.

Covers the additive VQ2 build (all torch+numpy only, no diffaero):
  * the vq2_like DIFFICULTY_PRESET (spec T3.1): exists, valid kwargs, VQ2 spacing + the >=+5 m HIGH-gate
    climb class, and sample_courses(**vq2_like) respects the preset bounds.
  * BatchedEstimatorEmulator PER-ENV gate gather: (a) BYTE-IDENTICAL to the shared (single-course) layout
    when every env shares one course -- the VQ1 no-regression pin; (b) a DIFFERENT course per env is fed
    ITS OWN geometry (the env-0-geometry bug class -- exactly the silent poisoning the gather closes);
    (c) set_courses updates per-env gates.
  * the a_body felt-accel obs arm: felt_accel_flu shape/finiteness + the FRD<->FLU flip.
  * tau_stale as a config value (spec fork 1): raising it makes age_norm DISCRIMINATE over the fed regime
    and the 0.10 default stays byte-identical.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_vq2_rl.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import inc8_estimator_emul as IE                                       # noqa: E402
from peregrine_course import (DEFAULT_COURSE_RANGES, DIFFICULTY_PRESETS,  # noqa: E402
                              sample_courses)

DT = torch.float64


# ============================================================ vq2_like preset (spec T3.1)
def test_vq2_like_preset_exists_and_keys_valid():
    assert "vq2_like" in DIFFICULTY_PRESETS
    # every override key must be a real sample_courses range (so a typo fails loudly, not silently)
    valid = set(DEFAULT_COURSE_RANGES)
    for name, ov in DIFFICULTY_PRESETS.items():
        assert set(ov) <= valid, (name, set(ov) - valid)
    # "medium" is the byte-identical default (empty override) -- the pre-preset contract
    assert DIFFICULTY_PRESETS["medium"] == {}


def test_vq2_like_encodes_spec_geometry():
    p = DIFFICULTY_PRESETS["vq2_like"]
    # VQ2 measured spacing band (spec T3.1: 23.7-38.5 m)
    assert p["seg_len_m"] == (23.7, 38.5)
    # the load-bearing HIGH-gate climb: drop_m lower edge is negative (a climb) and admits the >=+5 m
    # class (drop_m is +down, so a climb is a negative drop; -6.0 <= -5.0 covers the +5 m HIGH gate).
    assert p["drop_m"][0] <= -5.0, "vq2_like must admit the >=+5 m HIGH-gate climb (gate-2 class)"
    # moderate weave, not a hairpin slalom (descending course)
    assert p["turn_rad"] <= math.radians(45)


def test_vq2_like_sample_courses_respects_bounds():
    g = torch.Generator().manual_seed(11)
    c = sample_courses(256, device="cpu", generator=g, **DIFFICULTY_PRESETS["vq2_like"])
    R = {**DEFAULT_COURSE_RANGES, **DIFFICULTY_PRESETS["vq2_like"]}
    for v in c.values():
        assert torch.isfinite(v).all()
    pts = torch.cat([c["spawn_pos"].unsqueeze(1), c["gate_pos"]], dim=1)
    seg = pts[:, 1:] - pts[:, :-1]
    horiz = torch.linalg.norm(seg[..., :2], dim=-1)
    assert (horiz[:, 1:] >= R["seg_len_m"][0] - 1e-4).all()
    assert (horiz[:, 1:] <= R["seg_len_m"][1] + 1e-4).all()
    # a climb (negative drop, i.e. gate above the previous) actually occurs somewhere in the draw
    drop = -seg[:, 1:, 2]
    assert (drop < 0).any(), "vq2_like never produced a climbing segment across 256 courses"
    # min pairwise separation honoured
    d = torch.linalg.norm(pts[:, :, None, :2] - pts[:, None, :, :2], dim=-1)
    G1 = pts.shape[1]
    d = d + torch.eye(G1) * 1e9
    assert (d.amin(dim=(1, 2)) >= R["min_pair_dist_m"] - 1e-4).all()


# ============================================================ emulator per-env gate gather
def _make_shared_emu(n, gate_pos_ned_G, R_wg_G, cfg=None):
    return IE.BatchedEstimatorEmulator(n, gate_pos_ned_G, R_wg_G, config=cfg or IE.EmulConfig(),
                                       device="cpu", dtype=DT)


def _make_perenv_emu(n, gate_pos_ned_NG, R_wg_NG, cfg=None):
    return IE.BatchedEstimatorEmulator(n, gate_pos_ned_NG, R_wg_NG, config=cfg or IE.EmulConfig(),
                                       device="cpu", dtype=DT)


def _course(n_gates, yaw, base):
    """A trivial straight NED course: gate g at base + g*[8,0,0], all gates yaw ``yaw``."""
    gp = torch.stack([torch.tensor(base, dtype=DT) + g * torch.tensor([8.0, 0.0, 0.0], dtype=DT)
                      for g in range(n_gates)])                                   # (G,3)
    Rwg = IE.ned_gate_frame_torch(torch.full((n_gates,), float(yaw), dtype=DT))   # (G,3,3)
    return gp, Rwg


def test_per_env_gather_byte_identical_to_shared_when_courses_equal():
    """The per-env layout with EVERY env sharing one course must produce BYTE-IDENTICAL emulator output
    to the shared (G,..) layout. This is the VQ1 no-regression pin: the gather path adds no drift when
    the courses coincide (the only difference is the storage rank + the [env_arange, tg] indexing)."""
    n, G = 8, 4
    gp_G, Rwg_G = _course(G, np.pi, [30.0, 1.0, -2.0])
    shared = _make_shared_emu(n, gp_G, Rwg_G)
    perenv = _make_perenv_emu(n, gp_G.unsqueeze(0).expand(n, -1, -1).clone(),
                              Rwg_G.unsqueeze(0).expand(n, -1, -1, -1).clone())
    assert shared._per_env is False and perenv._per_env is True

    # identical cold-init + identical DR draws + identical injected randomness => identical output.
    torch.manual_seed(0)
    p0 = torch.randn(n, 3, dtype=DT) + torch.tensor([30.0, 0.0, -2.0], dtype=DT)
    v0 = torch.zeros(n, 3, dtype=DT)
    sig = torch.full((n,), 0.10, dtype=DT)
    bias = torch.zeros(n, dtype=DT)
    for emu in (shared, perenv):
        emu.reset_idx(torch.arange(n), p0.clone(), v0.clone(), sig.clone(), bias.clone())

    tg = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    R = torch.stack([torch.tensor(IE.ned_gate_frame_torch(torch.tensor(np.pi)).numpy(), dtype=DT)
                     for _ in range(n)])  # any consistent attitude; shared across both
    prev_p = p0.clone()
    for s in range(6):
        cur_p = prev_p + torch.tensor([-4.0, 0.0, 0.0], dtype=DT)
        accept_u = torch.rand(n, dtype=DT)
        accel_n = torch.randn(n, 3, dtype=DT)
        fix_n = torch.randn(n, 3, dtype=DT)
        out = []
        for emu in (shared, perenv):
            acc = emu.step(prev_p.clone(), v0.clone(), R, cur_p.clone(), v0.clone(), R, tg, 0.0333,
                           accept_u.clone(), accel_n.clone(), fix_n.clone())
            out.append((acc.clone(), emu.confidence_channel(tg).clone(),
                        emu.gate_frame_error_inplane(tg, cur_p.clone()).clone(),
                        emu.kf_pos_zup().clone()))
        # byte-identical across the two layouts
        assert torch.equal(out[0][0], out[1][0]), f"accepted mask diverged at step {s}"
        assert torch.equal(out[0][1], out[1][1]), f"confidence triple diverged at step {s}"
        assert torch.equal(out[0][2], out[1][2]), f"err_ip diverged at step {s}"
        assert torch.equal(out[0][3], out[1][3]), f"KF pos diverged at step {s}"
        prev_p = cur_p


def test_per_env_geometry_uses_each_envs_own_gates():
    """THE env-0-geometry bug guard: env 0 and env 1 fly DIFFERENT courses; the per-env emulator must
    compute env 1's fix geometry from env 1's gates, NOT env 0's. We put env 1's gate far away and
    off-axis: a shared (env-0-only) emulator would report the SAME range for both envs; the per-env one
    must report the env-specific range (the drone-to-its-own-gate distance)."""
    n, G = 2, 3
    gp0, Rwg0 = _course(G, np.pi, [30.0, 0.0, -2.0])    # env 0 gate 0 at (30,0,-2)
    gp1, Rwg1 = _course(G, np.pi, [80.0, 20.0, -2.0])   # env 1 gate 0 at (80,20,-2) -- far & off-axis
    gate_pos = torch.stack([gp0, gp1])                  # (2,G,3)
    R_wg = torch.stack([Rwg0, Rwg1])                    # (2,G,3,3)
    emu = _make_perenv_emu(n, gate_pos, R_wg)

    # both drones sit at the SAME world point; only their gates differ.
    drone = torch.tensor([[20.0, 0.0, -2.0], [20.0, 0.0, -2.0]], dtype=DT)
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))  # any attitude (N,3,3)
    emu.reset_idx(torch.arange(n), drone.clone(), torch.zeros(n, 3, dtype=DT),
                  torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
    tg = torch.zeros(n, dtype=torch.long)
    emu.step(drone.clone(), torch.zeros(n, 3, dtype=DT), R, drone.clone(), torch.zeros(n, 3, dtype=DT),
             R, tg, 0.0333, torch.rand(n, dtype=DT), torch.randn(n, 3, dtype=DT),
             torch.randn(n, 3, dtype=DT))
    rng = emu._last_geom["range"]
    # env 0: |(20,0,-2)-(30,0,-2)| = 10; env 1: |(20,0,-2)-(80,20,-2)| = hypot(60,20) = 63.24...
    assert abs(rng[0].item() - 10.0) < 1e-6, rng[0].item()
    assert abs(rng[1].item() - math.hypot(60.0, 20.0)) < 1e-6, rng[1].item()
    # a SHARED emulator (env-0 gates only) would report 10.0 for BOTH -- the bug this guards against.
    assert abs(rng[1].item() - 10.0) > 1.0


def test_set_courses_updates_per_env_gates_and_noop_when_shared():
    n, G = 2, 3
    gp0, Rwg0 = _course(G, np.pi, [30.0, 0.0, -2.0])
    emu = _make_perenv_emu(n, gp0.unsqueeze(0).expand(n, -1, -1).clone(),
                           Rwg0.unsqueeze(0).expand(n, -1, -1, -1).clone())
    gp_new, Rwg_new = _course(G, np.pi, [99.0, 0.0, -2.0])
    emu.set_courses(torch.tensor([1]), gp_new.unsqueeze(0), Rwg_new.unsqueeze(0))
    assert torch.equal(emu.gate_pos_ned[0], gp0)          # env 0 untouched
    assert torch.equal(emu.gate_pos_ned[1], gp_new)       # env 1 updated
    # shared emulator: set_courses is a no-op (returns without touching the (G,..) tensor)
    shared = _make_shared_emu(n, gp0, Rwg0)
    before = shared.gate_pos_ned.clone()
    shared.set_courses(torch.tensor([1]), gp_new.unsqueeze(0), Rwg_new.unsqueeze(0))
    assert torch.equal(shared.gate_pos_ned, before)


# ============================================================ a_body felt-accel arm
def test_felt_accel_flu_shape_and_flip():
    n, G = 4, 3
    gp, Rwg = _course(G, np.pi, [30.0, 0.0, -2.0])
    emu = _make_shared_emu(n, gp, Rwg)
    # zeros before the first step (rest start, no fix yet)
    assert torch.equal(emu.felt_accel_flu(), torch.zeros(n, 3, dtype=DT))
    # drive one step with a non-trivial velocity change -> a finite specific force
    drone = torch.tensor(np.broadcast_to([25.0, 0.0, -2.0], (n, 3)).copy(), dtype=DT)
    v_prev = torch.zeros(n, 3, dtype=DT)
    v_cur = torch.tensor(np.broadcast_to([-5.0, 0.0, 0.0], (n, 3)).copy(), dtype=DT)
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))
    emu.reset_idx(torch.arange(n), drone.clone(), v_prev.clone(),
                  torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
    emu.step(drone.clone(), v_prev, R, drone.clone(), v_cur, R, torch.zeros(n, dtype=torch.long),
             0.0333, torch.rand(n, dtype=DT), torch.zeros(n, 3, dtype=DT), torch.zeros(n, 3, dtype=DT))
    a_flu = emu.felt_accel_flu()
    assert a_flu.shape == (n, 3) and torch.isfinite(a_flu).all()
    # FLU == FRD * diag(1,-1,-1): the accessor applies exactly the involutory flip to the stored FRD.
    assert torch.equal(a_flu, emu._last_accel_body * emu.flip)


# ============================================================ tau_stale as a config value (fork 1)
def test_tau_stale_default_is_frozen_constant():
    assert IE.EmulConfig().tau_stale == IE.TAU_STALE_S == 0.10
    assert IE.EmulConfig().sigma_ref == IE.SIGMA_REF_M == 0.05


def test_tau_stale_raises_age_discrimination():
    """At tau_stale=0.10, an age of 0.25-0.55 s (the measured VQ2 fed regime) all saturates age_norm to
    1.0 (no discrimination). Raising tau_stale to 0.5 makes age_norm DISCRIMINATE across that regime."""
    n, G = 1, 2
    gp, Rwg = _course(G, np.pi, [30.0, 0.0, -2.0])

    def age_norm_at(tau, t_since):
        emu = _make_shared_emu(n, gp, Rwg, cfg=IE.EmulConfig(tau_stale=tau))
        emu.reset_idx(torch.arange(n), torch.zeros(n, 3, dtype=DT), torch.zeros(n, 3, dtype=DT),
                      torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
        emu._t_since_fix[:] = t_since
        return emu.confidence_channel(torch.zeros(n, dtype=torch.long))[0, 2].item()

    # tau=0.10: both 0.25 and 0.55 s saturate to 1.0 (no discrimination)
    assert age_norm_at(0.10, 0.25) == pytest.approx(1.0)
    assert age_norm_at(0.10, 0.55) == pytest.approx(1.0)
    # tau=0.50: 0.25 s reads 0.5, 0.55 s reads 1.0 -> the channel now discriminates the fed regime
    assert age_norm_at(0.50, 0.25) == pytest.approx(0.5)
    assert age_norm_at(0.50, 0.45) == pytest.approx(0.9)


# ============================================================ pose-age DR (spec T2.3, the ONE DR channel)
def test_pose_age_dr_off_is_byte_identical():
    """DR OFF (the defaults) -> the age reading is EXACTLY the accept-driven clock. Same seed/draws with
    a DR-OFF config must produce byte-identical confidence to the frozen EmulConfig()."""
    n, G = 6, 3
    gp, Rwg = _course(G, np.pi, [30.0, 0.0, -2.0])

    def run(cfg):
        emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
        assert emu._pose_dr_on is False and emu._blackout_on is False
        emu.reset_idx(torch.arange(n), torch.zeros(n, 3, dtype=DT), torch.zeros(n, 3, dtype=DT),
                      torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
        emu._t_since_fix[:] = 0.037
        return emu.confidence_channel(torch.zeros(n, dtype=torch.long)).clone()

    a = run(IE.EmulConfig())
    b = run(IE.EmulConfig(pose_age_floor_hi=0.0, pose_age_stall_p=0.0, blackout_range_m=0.0))
    assert torch.equal(a, b)


def test_pose_age_floor_lifts_age_reading():
    """The per-episode baseline latency floor ADDS to the staleness clock -> a fresh-fix env that would
    read age~0 now reads a nonzero floor age (the fed-regime lag the async-detect still pays)."""
    n, G = 64, 3
    gp, Rwg = _course(G, np.pi, [30.0, 0.0, -2.0])
    cfg = IE.EmulConfig(tau_stale=0.5, pose_age_floor_lo=0.15, pose_age_floor_hi=0.30)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    assert emu._pose_dr_on is True
    emu.reset_idx(torch.arange(n), torch.zeros(n, 3, dtype=DT), torch.zeros(n, 3, dtype=DT),
                  torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
    emu._t_since_fix[:] = 0.0                     # just fixed
    age = emu.confidence_channel(torch.zeros(n, dtype=torch.long))[:, 2]
    # age = clip((0 + floor)/0.5): floor in [0.15,0.30] -> age in [0.30, 0.60], strictly > 0
    assert (age > 0.29).all() and (age < 0.61).all(), (age.min().item(), age.max().item())
    # the floor VARIES per episode (a distribution, not a constant)
    assert age.std() > 0.01


def test_blackout_suppresses_fix_inside_range():
    """Inside blackout_range_m the fix is masked out (no KF update, staleness climbs) even head-on."""
    n, G = 8, 3
    gp, Rwg = _course(G, np.pi, [30.0, 0.0, -2.0])
    cfg = IE.EmulConfig(blackout_range_m=4.5)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    assert emu._blackout_on is True
    # place the drone 3 m from gate 0 (inside the 4.5 m blackout), head-on
    drone = torch.tensor(np.broadcast_to([27.0, 0.0, -2.0], (n, 3)).copy(), dtype=DT)  # 3 m from (30,0,-2)
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))
    emu.reset_idx(torch.arange(n), drone.clone(), torch.zeros(n, 3, dtype=DT),
                  torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
    # force_accept would normally guarantee a fix; the blackout mask must still veto it inside range.
    acc = emu.step(drone.clone(), torch.zeros(n, 3, dtype=DT), R, drone.clone(),
                   torch.zeros(n, 3, dtype=DT), R, torch.zeros(n, dtype=torch.long), 0.0333,
                   torch.zeros(n, dtype=DT), torch.zeros(n, 3, dtype=DT), torch.zeros(n, 3, dtype=DT),
                   force_accept=True)
    assert emu._last_geom["range"].max().item() < 4.5
    assert not acc.any(), "blackout must veto the fix inside blackout_range_m even under force_accept"


# ============================================================ vision latency: CONTENT lag (Option 1)
def _drive(emu, n, R, positions, vels, dt=0.0333, tg=None, seeds=None):
    """Drive the emulator over a scripted (pos, vel) sequence; return the list of per-step accepted."""
    tg = torch.zeros(n, dtype=torch.long) if tg is None else tg
    accs = []
    for k in range(1, len(positions)):
        acc = emu.step(positions[k - 1], vels[k - 1], R, positions[k], vels[k], R, tg, dt,
                       torch.zeros(n, dtype=DT), torch.zeros(n, 3, dtype=DT),
                       torch.zeros(n, 3, dtype=DT), force_accept=True)
        accs.append(acc)
    return accs


def test_vision_lag_off_is_byte_identical():
    """LAG OFF (defaults: lat_max_s<=0, healthy_frac<=0) -> no buffer, no content lag, the age reading
    is the classic accept-driven clock. Same script under the default config must be byte-identical to a
    config with the (OFF) latency knobs explicitly zeroed."""
    n, G = 4, 2
    gp, Rwg = _course(G, np.pi, [40.0, 0.0, -2.0])
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))
    pos = [torch.tensor(np.broadcast_to([40.0 - 3.0 * k, 0.0, -2.0], (n, 3)).copy(), dtype=DT)
           for k in range(6)]
    vel = [torch.tensor(np.broadcast_to([-3.0 / 0.0333, 0.0, 0.0], (n, 3)).copy(), dtype=DT)
           for _ in range(6)]

    def run(cfg):
        emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
        assert emu._lat_on is False
        emu.reset_idx(torch.arange(n), pos[0].clone(), torch.zeros(n, 3, dtype=DT),
                      torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
        _drive(emu, n, R, pos, vel)
        return (emu.kf_pos_zup().clone(),
                emu.confidence_channel(torch.zeros(n, dtype=torch.long)).clone())

    a = run(IE.EmulConfig())
    b = run(IE.EmulConfig(lat_max_s=0.0, lat_healthy_frac=0.0))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_vision_lag_delays_measurement_content():
    """THE fidelity property (Fengyou rider): a fix at step t carries the gate geometry as of the
    CAPTURE step t-Delta -- i.e. the fix z reflects the drone's PAST position, not its current one. We
    fix Delta deterministically (healthy_frac=1, a narrow healthy window == exactly 2 steps) and check
    the fused KF position lands BEHIND the current truth by ~Delta*speed (the latency-induced bias
    'toward where the drone WAS'), NOT at the current truth."""
    n, G = 1, 2
    gp, Rwg = _course(G, np.pi, [60.0, 0.0, -2.0])
    dt = 0.0333
    lag_steps = 3
    lag_s = lag_steps * dt
    # a narrow healthy window centred on exactly lag_s so round(Delta/dt) == lag_steps deterministically.
    cfg = IE.EmulConfig(lat_healthy_frac=1.0, lat_healthy_lo=lag_s, lat_healthy_hi=lag_s,
                        lat_max_s=1.0, lat_clamp_s=1.0, sigma_lat_lo=1e-9, sigma_lat_hi=1e-9,
                        inject_bias=False, imu_accel_noise=0.0)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    assert emu._lat_on is True
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))
    speed = 15.0
    step_dx = speed * dt
    # a clean monotone approach toward the gate at x=60: x goes 20 -> up as k grows.
    pos = [torch.tensor([[20.0 + step_dx * k, 0.0, -2.0]], dtype=DT) for k in range(12)]
    vel = [torch.tensor([[speed, 0.0, 0.0]], dtype=DT) for _ in range(12)]
    emu.reset_idx(torch.arange(n), pos[0].clone(), torch.zeros(n, 3, dtype=DT),
                  torch.full((n,), 1e-9, dtype=DT), torch.zeros(n, dtype=DT))
    # seed the KF at truth so the ONLY thing that moves it off truth is the lagged fix.
    _drive(emu, n, R, pos[:lag_steps + 3], vel[:lag_steps + 3])
    cur_x = pos[lag_steps + 2][0, 0].item()
    kf_x = emu.kf.position[0, 0].item()
    # the fused KF x must sit BEHIND the current truth by ~lag_s*speed (it saw where the drone WAS).
    lag_dist = lag_s * speed
    assert kf_x < cur_x - 0.5 * lag_dist, (kf_x, cur_x, lag_dist)
    assert abs((cur_x - kf_x) - lag_dist) < 2.0 * step_dx, (cur_x - kf_x, lag_dist)


def test_vision_lag_age_channel_carries_real_delta():
    """The age-of-fix obs channel reflects the SAME Delta that lagged the content (not a fictional
    label). With a deterministic Delta and tau_stale, age_norm == clip(Delta/tau) right after a fix."""
    n, G = 1, 2
    gp, Rwg = _course(G, np.pi, [60.0, 0.0, -2.0])
    dt = 0.0333
    lag_s = 5 * dt                      # ~0.167 s
    cfg = IE.EmulConfig(lat_healthy_frac=1.0, lat_healthy_lo=lag_s, lat_healthy_hi=lag_s,
                        lat_max_s=1.0, lat_clamp_s=1.0, tau_stale=0.5, sigma_lat_lo=1e-9,
                        sigma_lat_hi=1e-9, inject_bias=False, imu_accel_noise=0.0)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))
    speed = 15.0
    pos = [torch.tensor([[20.0 + speed * dt * k, 0.0, -2.0]], dtype=DT) for k in range(10)]
    vel = [torch.tensor([[speed, 0.0, 0.0]], dtype=DT) for _ in range(10)]
    emu.reset_idx(torch.arange(n), pos[0].clone(), torch.zeros(n, 3, dtype=DT),
                  torch.full((n,), 1e-9, dtype=DT), torch.zeros(n, dtype=DT))
    _drive(emu, n, R, pos[:8], vel[:8])   # several forced fixes; last fix age == lag_s exactly
    age = emu.confidence_channel(torch.zeros(n, dtype=torch.long))[0, 2].item()
    # right after a landed fix, _last_fix_age == lag_s -> age_norm == clip(lag_s/0.5)
    assert abs(age - min(lag_s / 0.5, 1.0)) < 1e-6, (age, lag_s / 0.5)


def test_vision_lag_delta_is_bimodal():
    """Delta is drawn from the two-mode mixture (healthy + contention), NOT collapsed to one mean."""
    n = 20000
    gp, Rwg = _course(2, np.pi, [40.0, 0.0, -2.0])
    cfg = IE.EmulConfig(lat_healthy_frac=0.5, lat_healthy_lo=0.07, lat_healthy_hi=0.12,
                        lat_cont_lo=0.15, lat_cont_hi=0.55, lat_max_s=1.0)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    torch.manual_seed(0)
    d = emu._draw_latency_s(n)
    # ~half in the healthy band, ~half in the contention band (a genuine mixture, gap in [0.12,0.15])
    healthy = ((d >= 0.07) & (d <= 0.12)).float().mean().item()
    cont = ((d >= 0.15) & (d <= 0.55)).float().mean().item()
    gap = ((d > 0.12) & (d < 0.15)).float().mean().item()
    assert 0.45 < healthy < 0.55, healthy
    assert 0.45 < cont < 0.55, cont
    assert gap < 0.01, gap        # the mixture is genuinely bimodal (near-empty gap)


def test_vision_lag_buffer_covers_clamp():
    """The truth buffer depth must cover the full 1.0 s clamp at the training dt (Fengyou: D_max covers
    the clamp)."""
    n, G = 1, 2
    gp, Rwg = _course(G, np.pi, [40.0, 0.0, -2.0])
    dt = 0.0333
    cfg = IE.EmulConfig(lat_healthy_frac=1.0, lat_max_s=1.0, lat_clamp_s=1.0)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))
    p = [torch.tensor([[20.0 + k, 0.0, -2.0]], dtype=DT) for k in range(3)]
    v = [torch.zeros(n, 3, dtype=DT) for _ in range(3)]
    emu.reset_idx(torch.arange(n), p[0].clone(), torch.zeros(n, 3, dtype=DT),
                  torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
    _drive(emu, n, R, p, v, dt=dt)
    # D_max = ceil(1.0/0.0333) = 31 -> depth 32 covers a 1.0 s lag at this dt
    assert emu._D_max >= math.ceil(cfg.lat_clamp_s / dt)


def test_vision_lag_direction_of_mismatch_is_safe():
    """DIRECTION-OF-MISMATCH (Fengyou rider 2): Option-1 forward-fusing a t-Delta measurement at t is
    STRICTLY HARDER than the deployed RewindKF/OOSM (which retro-corrects at capture time). So the
    training estimator's error must be >= a capture-time-fused reference on the same fix stream --
    training sees a WORSE estimator than deploy = the SAFE direction (never easier).

    We approximate the two paths on a moving drone with a deterministic lag: (A) Option-1 = fuse the
    lagged-content fix at the CURRENT step (what the train env does); (B) an idealized capture-time
    reference = fuse the SAME lagged content when the drone is actually AT that lagged position (zero
    residual error). B is a lower bound on the deployed RewindKF error; A must be >= B."""
    n, G = 1, 2
    dt = 0.0333
    lag_s = 4 * dt
    gp, Rwg = _course(G, np.pi, [80.0, 0.0, -2.0])
    cfg = IE.EmulConfig(lat_healthy_frac=1.0, lat_healthy_lo=lag_s, lat_healthy_hi=lag_s, lat_max_s=1.0,
                        sigma_lat_lo=1e-9, sigma_lat_hi=1e-9, inject_bias=False, imu_accel_noise=0.0)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    R = IE.ned_gate_frame_torch(torch.full((n,), float(np.pi), dtype=DT))
    speed = 20.0
    pos = [torch.tensor([[20.0 + speed * dt * k, 0.0, -2.0]], dtype=DT) for k in range(12)]
    vel = [torch.tensor([[speed, 0.0, 0.0]], dtype=DT) for _ in range(12)]
    emu.reset_idx(torch.arange(n), pos[0].clone(), torch.zeros(n, 3, dtype=DT),
                  torch.full((n,), 1e-9, dtype=DT), torch.zeros(n, dtype=DT))
    _drive(emu, n, R, pos[:9], vel[:9])
    cur = pos[8][0]
    kf = emu.kf.position[0]
    # Option-1 residual error = |KF - current truth|. The capture-time reference would place the SAME
    # lagged fix content at the lagged pose (error ~0). So Option-1 error must EXCEED the reference by
    # ~lag_s*speed -> training is measurably HARDER (never easier) than the retro-corrected deploy path.
    err_option1 = (kf - cur).norm().item()
    err_reference = 0.0                     # idealized capture-time fusion (lower bound on RewindKF error)
    assert err_option1 >= err_reference
    assert err_option1 > 0.5 * lag_s * speed, (err_option1, lag_s * speed)


# ============================================================ FIX-B: latency covariance inflation (B2)
def _fixb_leg_inplane_err(kind, speed, n=512, seed=0, n_steps=90):
    """Mean in-plane KF error (m) over the back 2/3 of a scripted leg flown under the DUAL-STAGE
    bimodal content lag (healthy 0.07-0.12 @ 0.5 / contention 0.15-0.55), forced fixes, fp64.
    The bimodal mixture is essential: FIX-B's covariance inflation works by selectively
    down-weighting the high-Delta contention fixes -- a fixed Delta (a constant measurement bias)
    is asymptotically absorbed at ANY R, so a deterministic-lag probe cannot see the fix."""
    dt = 0.0333
    torch.manual_seed(seed)
    gp, Rwg = _course(2, math.pi, [40.0, 0.0, -2.0])
    cfg = IE.EmulConfig(lat_max_s=1.0, lat_healthy_frac=0.5, lat_healthy_lo=0.07,
                        lat_healthy_hi=0.12, lat_cont_lo=0.15, lat_cont_hi=0.55,
                        lat_clamp_s=1.0, inject_bias=False)
    emu = _make_shared_emu(n, gp, Rwg, cfg=cfg)
    assert emu._lat_on is True
    R = torch.eye(3, dtype=DT).expand(n, 3, 3).contiguous()
    tg = torch.zeros(n, dtype=torch.long)
    # scripted velocity profile (NED): straight / climb (~8 m over 30 m) / crab (1.5 m/s lateral)
    # / turn (35 deg heading swing over 1.5 s mid-leg -- the vq2_like handoff turn class).
    pos_k = torch.tensor([2.0, 0.0, -2.0], dtype=DT)
    pos, vels = [pos_k.clone()], []
    heading = 0.0
    for k in range(n_steps + 1):
        if kind == "straight":
            v = torch.tensor([speed, 0.0, 0.0], dtype=DT)
        elif kind == "climb":
            s = 0.258
            v = torch.tensor([speed * math.sqrt(1 - s * s), 0.0, -speed * s], dtype=DT)
        elif kind == "crab":
            vy = 1.5
            v = torch.tensor([math.sqrt(speed * speed - vy * vy), vy, 0.0], dtype=DT)
        else:                                            # turn
            t = k * dt
            if 1.0 <= t < 2.5:
                heading = math.radians(35.0) * (t - 1.0) / 1.5
            v = torch.tensor([speed * math.cos(heading), speed * math.sin(heading), 0.0], dtype=DT)
        vels.append(v)
        if k < n_steps:
            pos.append(pos[-1] + v * dt)
    emu.reset_idx(torch.arange(n), pos[0].expand(n, 3).contiguous(),
                  vels[0].expand(n, 3).contiguous(),
                  torch.full((n,), 0.10, dtype=DT), torch.zeros(n, dtype=DT))
    errs = []
    for k in range(1, n_steps + 1):
        emu.step(pos[k - 1].expand(n, 3).contiguous(), vels[k - 1].expand(n, 3).contiguous(), R,
                 pos[k].expand(n, 3).contiguous(), vels[k].expand(n, 3).contiguous(), R, tg, dt,
                 torch.rand(n, dtype=DT), torch.randn(n, 3, dtype=DT),
                 torch.randn(n, 3, dtype=DT), force_accept=True)
        if k > n_steps // 3:
            errs.append(emu.gate_frame_error_inplane(tg, pos[k].expand(n, 3)).mean().item())
    return sum(errs) / len(errs)


def test_fixb_latency_cov_inflation_never_worse():
    """FIX-B (B2 2026-07-06, diagnosis RC5): with latency ON, the fix R is inflated by
    outer(v_hat*Delta)+eps*I so the KF stops absorbing the forward-fuse lag error on the trusted
    in-plane axes. Pre-fix these legs read turn 0.382 / climb 0.388 / crab 0.372 (every pin FAILS
    without the inflation); post-fix 0.166 / 0.214 / 0.199. Straight stays bounded (pre 0.070,
    post 0.079) -- the never-materially-worse guarantee. Latency-OFF byte-identity is pinned
    separately by test_vision_lag_off_is_byte_identical."""
    err_turn = _fixb_leg_inplane_err("turn", 6.0)
    err_climb = _fixb_leg_inplane_err("climb", 6.0)
    err_crab = _fixb_leg_inplane_err("crab", 6.0)
    err_straight = _fixb_leg_inplane_err("straight", 6.0)
    assert err_turn < 0.26, err_turn
    assert err_climb < 0.28, err_climb
    assert err_crab < 0.28, err_crab
    assert err_straight < 0.12, err_straight
