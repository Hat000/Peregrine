"""2026-07-06 B2 fixes -- laptop tests, torch+numpy only, no diffaero.

The I3 integration pin file for the B2 package (docs/vq2-handoff-diagnosis-2026-07-06):

  M1 SPAWN FIX: near-spawn resets must NEVER target the last gate when the course has >= 2 gates
    (the RC1 'spawn lottery' -- 70% of resets spawned 1 m before a uniformly-random gate INCLUDING
    the last, banking trivial finishes). The draw lives in
    peregrine_racing.sample_reset_target_gates, the pure helper reset_idx now calls, so these
    tests exercise the ACTUAL shipped draw logic without diffaero.
  M4+M6 LADDER: handoff_drill rung (drop clamped [-2,+4] m via the new course_drop_lo/hi sampler
    keys) + dual_gate_full + STAGE_ORDER rewire + fix economy (rw_estimerr 1.0, rw_fix_bonus 0.75)
    + the as-flown dual_gate FROZEN byte-identically (job 3295856 reproducibility).
  M3 NOISE ANNEAL: every FLOWN stage renders the '+algo.noise_*' ceiling schedule (no floors).
  M5 FIX-B: latency covariance inflation R += outer(v_hat*Delta)+eps*I -- never-worse on the
    turn leg, bounded on the straight leg, and byte-identical (KF state AND covariance) with
    latency OFF.
  A1: lookat_rate_clamp byte-id at the default 0.0 (identity OBJECT, zero float ops).
  M2a: spawn-class metric safe division (empty class -> exactly 0.0, never NaN).

Sibling B2 pin files (deeper coverage, non-overlapping test names):
tests/test_b2_spawn_class_metrics.py (M2a/b/c) · tests/test_vq2_b2_arms.py (A1/A2 full) ·
tests/test_vq2_rl.py FIX-B section (all four legs) · tests/test_inc8_noise_anneal.py (M3 schedule).

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_vq2_b2_fixes.py -q
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import inc8_estimator_emul as IE                                       # noqa: E402
import inc8_reward as R8                                               # noqa: E402
import inc8_spawn_metrics as SM                                        # noqa: E402
from peregrine_racing import sample_reset_target_gates                 # noqa: E402
from vq2_curriculum import (STAGES, STAGE_ORDER, COURSE_SAMPLER_KEYS,  # noqa: E402
                            render_overrides)

DT = torch.float64


# ============================================================ M1: the spawn-lottery fix
def test_near_spawn_never_targets_last_gate_G_ge_2():
    """THE M1 pin: for every multi-gate course size the near-spawn target draw excludes the last
    gate entirely and still reaches every non-last gate ~uniformly (no trivial 1 m finishes)."""
    torch.manual_seed(0)
    for G in (2, 3, 5, 7):
        standing = torch.zeros(4096, dtype=torch.bool)
        tg = sample_reset_target_gates(4096, G, standing)
        assert tg.dtype == torch.int32, "reset_idx contract: int32 target indices"
        assert int(tg.min()) >= 0
        assert int(tg.max()) <= G - 2, f"G={G}: near-spawn drew the LAST gate (spawn lottery)"
        counts = torch.bincount(tg.long(), minlength=G)
        assert int(counts[G - 1]) == 0
        assert bool((counts[: G - 1] > 0).all()), f"G={G}: some non-last gate never drawn"
        frac = counts[: G - 1].float() / 4096
        assert bool((frac - 1.0 / (G - 1)).abs().max() < 0.05), f"G={G}: non-uniform {frac}"


def test_single_gate_draw_unchanged_G1():
    """G=1 stages (single_gate, blackout_pass) are UNCHANGED: the draw equals the legacy
    randint(0, 1) bit-for-bit under an identical generator state (all zeros, same dtype)."""
    standing = torch.zeros(512, dtype=torch.bool)
    tg = sample_reset_target_gates(512, 1, standing)
    assert torch.equal(tg, torch.zeros(512, dtype=torch.int32))


def test_standing_split_untouched_and_forced_to_gate0():
    """The fix changes only the TARGET draw: standing entries still land on gate 0, the caller's
    standing mask is never mutated, and the standing/near split (drawn by reset_idx BEFORE this
    helper from torch.rand < standing_start_frac) is independent of the fix."""
    torch.manual_seed(1)
    standing = torch.rand(2048) < 0.3
    before = standing.clone()
    tg = sample_reset_target_gates(2048, 5, standing)
    assert bool((tg[standing] == 0).all())
    assert int(tg[~standing].max()) <= 3, "non-standing draws must still exclude the last gate"
    assert torch.equal(standing, before), "helper must not mutate the caller's standing mask"


def test_reset_idx_wired_to_helper_SOURCE_PIN():
    """Wiring pin (reset_idx itself needs diffaero, so pin the source): reset_idx must call the
    helper and the legacy inline full-range randint must be gone."""
    src = (ROOT / "rl" / "peregrine_racing.py").read_text(encoding="utf-8")
    assert "sample_reset_target_gates(m, self.n_gates, standing" in src
    assert "torch.randint(0, self.n_gates, (m,)" not in src


# ============================================================ M6: the B2 ladder + course_drop keys
# The as-flown dual_gate render at B1 HEAD f63b4d9 (job 3295856 curr_a) -- byte-identical pin.
# If this fails, someone let _COMMON (or the renderer) drift into the FROZEN reproducibility stage.
_DUAL_GATE_AS_FLOWN = (
    "+env.course_mode=random +env.track_difficulty=vq2_like +env.emul_camera_flip=True "
    "+env.emul_tau_stale=0.5 +env.rw_gate_progress=10.0 +env.rw_estimerr_clamp=0.5 "
    "+env.rw_through_centering=10.0 +env.lookat_g_yaw=3.0 +env.lookat_g_pitch=3.0 "
    "+env.lookat_warmup_updates=200 +env.course_n_gates=2 +env.course_seg_len_lo=23.7 "
    "+env.course_seg_len_hi=28.0 +env.emul_blackout_range_m=4.3 +env.emul_lat_max_s=1.0 "
    "+env.emul_lat_healthy_frac=0.5 +env.emul_lat_healthy_lo=0.07 +env.emul_lat_healthy_hi=0.12 "
    "+env.emul_lat_cont_lo=0.15 +env.emul_lat_cont_hi=0.55 "
    "env.max_time=60 algo.gamma=0.995 algo.clip_value_loss=False"
)

_LAT_KEYS = ("emul_lat_max_s", "emul_lat_healthy_frac", "emul_lat_healthy_lo",
             "emul_lat_healthy_hi", "emul_lat_cont_lo", "emul_lat_cont_hi")


def test_stage_order_is_the_b2_ladder():
    assert STAGE_ORDER == ("single_gate", "blackout_pass", "handoff_drill", "dual_gate_full",
                           "multi_gate")
    assert "dual_gate" in STAGES and "dual_gate" not in STAGE_ORDER   # frozen, not flown


def test_b2b_reverted_m3_noise_anneal_and_m4_economy():
    # B2b (2026-07-06): M3 noise_anneal + M4 fix economy REVERTED after the curr_b probe verdict --
    # sgpA (anneal OFF) flew (raw 0.79 / standing 0.39, ~= curr_a), sgpB (econ OFF) also recovered
    # (0.20 @ upd 1120 vs curr_b ~0.001). The 0.35 ceiling starved discovery-phase exploration and
    # the economy was misapplied outside the post-handoff regime; both are deferred to measured
    # increments. STRUCTURAL wins (spawn fix, honest metrics, FIX-B, handoff_drill) are KEPT.
    for s in STAGE_ORDER:
        toks = render_overrides(s)
        assert not any("noise_anneal" in t or "noise_std" in t for t in toks), s
        assert not any(t.startswith("+env.rw_estimerr=") for t in toks), s
        assert not any("rw_fix_bonus" in t for t in toks), s
        # the audit-A2 anchor CLAMP stays (structural; never part of the reverted economy)
        assert "+env.rw_estimerr_clamp=0.5" in toks, s


def test_handoff_drill_stage_kept_after_b2b_revert():
    # M6 (the handoff_drill drop clamp) is a STRUCTURAL win, unaffected by the M3/M4 revert.
    toks = render_overrides("handoff_drill")
    assert "+env.course_drop_lo=-2.0" in toks and "+env.course_drop_hi=4.0" in toks
    assert "+env.course_n_gates=2" in toks
    assert not any("noise_anneal" in t for t in toks)   # no anneal even on the hard stages now


def test_handoff_drill_stage():
    d = STAGES["handoff_drill"]
    assert d["course_n_gates"] == 2
    assert (d["course_seg_len_lo"], d["course_seg_len_hi"]) == (23.7, 28.0)
    assert (d["course_drop_lo"], d["course_drop_hi"]) == (-2.0, 4.0)
    assert d["emul_blackout_range_m"] == 4.3
    for k in _LAT_KEYS:                 # latency block VERBATIM from the as-flown dual_gate
        assert d[k] == STAGES["dual_gate"][k], k
    assert d["_raw"]["env.max_time"] == 60
    assert d["_raw"]["algo.gamma"] == 0.995
    assert d["_raw"]["algo.clip_value_loss"] is False


def test_dual_gate_full_has_no_drop_narrowing():
    d = STAGES["dual_gate_full"]
    assert not any(k.startswith("course_drop") for k in d)
    assert not any("drop" in t for t in render_overrides("dual_gate_full"))
    # otherwise the flown dual_gate content: 2 gates, narrowed seg band, blackout+latency, same _raw core
    assert d["course_n_gates"] == 2
    assert (d["course_seg_len_lo"], d["course_seg_len_hi"]) == (23.7, 28.0)
    for k in _LAT_KEYS:
        assert d[k] == STAGES["dual_gate"][k], k
    assert d["_raw"]["env.max_time"] == 60 and d["_raw"]["algo.gamma"] == 0.995


def test_frozen_dual_gate_renders_byte_identical_to_as_flown():
    assert " ".join(render_overrides("dual_gate")) == _DUAL_GATE_AS_FLOWN


def test_course_sampler_keys_include_drop():
    assert COURSE_SAMPLER_KEYS == ("course_n_gates", "course_seg_len_lo", "course_seg_len_hi",
                                   "course_drop_lo", "course_drop_hi")


def test_sampler_honours_drop_band():
    from peregrine_course import sample_courses, DIFFICULTY_PRESETS
    ov = dict(DIFFICULTY_PRESETS["vq2_like"])
    ov.update(n_gates=2, seg_len_m=(23.7, 28.0), drop_m=(-2.0, 4.0))
    g = torch.Generator().manual_seed(0)
    c = sample_courses(2048, device="cpu", generator=g, **ov)
    dz_up = c["gate_pos"][:, 1, 2] - c["gate_pos"][:, 0, 2]     # drop is +DOWN => dz_up = -drop
    assert dz_up.min() >= -4.0 - 1e-5 and dz_up.max() <= 2.0 + 1e-5
    # max_grade 0.55 * seg_len_lo 23.7 = 13.0 m >> 4 -> the grade clamp never binds; band is exact
    assert dz_up.min() < -3.5 and dz_up.max() > 1.5             # both edges actually reached


def test_env_wires_course_drop_SOURCE_PIN():
    """Wiring pin (PeregrineRacing.__init__ needs diffaero): the env forwards course_drop_lo/hi to
    the sampler's drop_m EXACTLY like the proven course_seg_len wiring; unset leaves the overrides
    dict untouched (byte-identical)."""
    src = (ROOT / "rl" / "peregrine_racing.py").read_text(encoding="utf-8")
    assert 'getattr(cfg, "course_drop_lo", None)' in src
    assert 'getattr(cfg, "course_drop_hi", None)' in src
    assert 'overrides["drop_m"]' in src


# ============================================================ M5 FIX-B: latency covariance inflation
def _fixb_leg_inplane_err(kind, speed, n=512, seed=0, n_steps=90):
    """Mean in-plane KF error (m) over the back 2/3 of a scripted leg flown under the DUAL-STAGE
    bimodal content lag (healthy 0.07-0.12 @ 0.5 / contention 0.15-0.55), forced fixes, fp64.
    The bimodal mixture is essential: FIX-B's covariance inflation works by selectively
    down-weighting the high-Delta contention fixes -- a fixed Delta (a constant measurement bias)
    is asymptotically absorbed at ANY R, so a deterministic-lag probe cannot see the fix.
    (Mirrors tests/test_vq2_rl.py; here only the turn/straight ends are pinned -- benefit +
    never-worse -- climb/crab live in test_vq2_rl.py's four-leg version.)"""
    dt = 0.0333
    torch.manual_seed(seed)
    gp = torch.stack([torch.tensor([40.0, 0.0, -2.0], dtype=DT) + g * torch.tensor(
        [8.0, 0.0, 0.0], dtype=DT) for g in range(2)])
    Rwg = IE.ned_gate_frame_torch(torch.full((2,), math.pi, dtype=DT))
    cfg = IE.EmulConfig(lat_max_s=1.0, lat_healthy_frac=0.5, lat_healthy_lo=0.07,
                        lat_healthy_hi=0.12, lat_cont_lo=0.15, lat_cont_hi=0.55,
                        lat_clamp_s=1.0, inject_bias=False)
    emu = IE.BatchedEstimatorEmulator(n, gp, Rwg, config=cfg, device="cpu", dtype=DT)
    assert emu._lat_on is True
    R = torch.eye(3, dtype=DT).expand(n, 3, 3).contiguous()
    tg = torch.zeros(n, dtype=torch.long)
    pos_k = torch.tensor([2.0, 0.0, -2.0], dtype=DT)
    pos, vels = [pos_k.clone()], []
    heading = 0.0
    for k in range(n_steps + 1):
        if kind == "straight":
            v = torch.tensor([speed, 0.0, 0.0], dtype=DT)
        else:                                            # turn (35 deg swing over 1.5 s mid-leg)
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


def test_fixb_turn_benefit_and_straight_never_worse():
    """FIX-B (B2, diagnosis RC5): the inflated fix R stops the KF absorbing the forward-fuse lag
    error under the bimodal mixture. Pre-fix the turn leg reads 0.382 (FAILS the 0.26 pin); post-fix
    0.166. The straight leg stays bounded (pre 0.070, post 0.079, far below the 0.12 ceiling) --
    the never-materially-worse guarantee."""
    assert _fixb_leg_inplane_err("turn", 6.0) < 0.26
    assert _fixb_leg_inplane_err("straight", 6.0) < 0.12


def test_fixb_latency_off_is_byte_identical():
    """With latency OFF (delta_s is None on every step) the FIX-B inflation path must be
    structurally untouched: same scripted leg under the default config vs explicitly-zeroed latency
    knobs gives bit-equal KF position, velocity AND full covariance P (the covariance is the very
    thing FIX-B touches, so P equality is the sharp pin)."""
    n = 8
    dt = 0.0333
    gp = torch.stack([torch.tensor([40.0, 0.0, -2.0], dtype=DT) + g * torch.tensor(
        [8.0, 0.0, 0.0], dtype=DT) for g in range(2)])
    Rwg = IE.ned_gate_frame_torch(torch.full((2,), math.pi, dtype=DT))
    R = torch.eye(3, dtype=DT).expand(n, 3, 3).contiguous()
    tg = torch.zeros(n, dtype=torch.long)

    def run(cfg):
        torch.manual_seed(3)
        emu = IE.BatchedEstimatorEmulator(n, gp, Rwg, config=cfg, device="cpu", dtype=DT)
        assert emu._lat_on is False
        emu.reset_idx(torch.arange(n), torch.full((n, 3), 2.0, dtype=DT),
                      torch.zeros(n, 3, dtype=DT), torch.full((n,), 0.10, dtype=DT),
                      torch.zeros(n, dtype=DT))
        p = torch.full((n, 3), 2.0, dtype=DT)
        v = torch.tensor([6.0, 0.0, 0.0], dtype=DT).expand(n, 3).contiguous()
        for k in range(30):
            p_next = p + v * dt
            emu.step(p, v, R, p_next, v, R, tg, dt, torch.rand(n, dtype=DT),
                     torch.randn(n, 3, dtype=DT), torch.randn(n, 3, dtype=DT), force_accept=True)
            p = p_next
        return emu.kf_pos_zup().clone(), emu.kf.velocity.clone(), emu.kf.P.clone()

    a = run(IE.EmulConfig())
    b = run(IE.EmulConfig(lat_max_s=0.0, lat_healthy_frac=0.0))
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2]), "KF covariance drifted with latency OFF -- FIX-B gate leaked"


# ============================================================ A1: lookat_max_rate byte-id at default
def test_lookat_max_rate_default_is_byte_identical():
    """The A1 knob at its default (0.0, and any <=0) must return the INPUT TENSOR ITSELF -- no
    copy, no float op == the unclamped legacy path exactly. Active cap: symmetric per-axis clamp,
    in-bound components bit-exact. (Full geometry-driven coverage: tests/test_vq2_b2_arms.py.)"""
    dlook = torch.tensor([[0.494, -0.103, 1.358], [-2.0, 0.25, -0.75]], dtype=DT)
    assert R8.lookat_rate_clamp(dlook, 0.0) is dlook          # identity OBJECT == byte-identical
    assert R8.lookat_rate_clamp(dlook, -1.0) is dlook
    out = R8.lookat_rate_clamp(dlook, 1.0)
    assert out.abs().max().item() <= 1.0
    assert torch.equal(torch.sign(out), torch.sign(dlook))
    inb = dlook.abs() <= 1.0
    assert torch.equal(out[inb], dlook[inb])
    # env wiring pin: knob read default-0.0 and applied through the helper (source pin -- the env
    # class itself needs diffaero to construct).
    src = (ROOT / "rl" / "peregrine_racing_inc8.py").read_text(encoding="utf-8")
    assert 'getattr(cfg, "lookat_max_rate", 0.0)' in src
    assert "R8.lookat_rate_clamp(dlook, self._lookat_max_rate)" in src


# ============================================================ M2a: spawn-class metric safe division
def test_spawn_class_rates_safe_division_never_nan():
    """Empty classes must read EXACTLY 0.0, never NaN: at cold start (all-zero accumulators) and
    after updates that leave some class empty (class 2 goes to ~0 mass post-M1 for G>=2 -- the
    metric must stay finite there forever)."""
    ep = torch.zeros(SM.N_CLASSES)
    su = torch.zeros(SM.N_CLASSES)
    np_ = torch.zeros(SM.N_CLASSES)
    for t in SM.class_rates(ep, su, np_):
        assert torch.isfinite(t).all() and (t == 0.0).all()
    # populate ONLY classes 0 and 1; class 2 (nearlast) stays empty -> 0.0 exactly, not NaN
    cls = torch.tensor([0, 1, 1])
    SM.update_class_accumulators(ep, su, np_, cls, torch.tensor([True, True, True]),
                                 torch.tensor([True, False, True]),
                                 torch.tensor([1.0, 0.0, 1.0]), decay=1.0)
    r_s, r_n, r_f = SM.class_rates(ep, su, np_)
    assert torch.isfinite(r_s).all() and torch.isfinite(r_n).all() and torch.isfinite(r_f).all()
    assert r_s[SM.CLS_NEARLAST].item() == 0.0
    assert r_n[SM.CLS_NEARLAST].item() == 0.0
    assert r_f[SM.CLS_NEARLAST].item() == 0.0
    assert r_s[0].item() == pytest.approx(1.0) and r_s[1].item() == pytest.approx(0.5)
