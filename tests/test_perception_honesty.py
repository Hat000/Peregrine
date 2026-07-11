"""Tests for the PERCEPTION-HONESTY / HARD NO-SPIN package (2026-07-10, DESIGN.md §P).

Covers the PURE, laptop-testable core (the env class body is diffaero-gated cluster-only, so all new
step() logic deliberately lives in module-level functions -- these tests + the sbatch PRECHECK=1 run
are the coverage story):

  (1) DEFAULTS-REPRODUCE: gate_detectable is untouched by the package; the blur hard gate at hi=inf
      is the identity; BatchedEgoEstimator.step with blur_extra_miss=None is trajectory-identical to
      an explicit zeros tensor (the fold-in is neutral at zero) through propagate/mask/reacquire.
  (2) HARD-GATE semantics: a geometrically visible gate becomes undetectable when the LOS-perp rate
      exceeds the cutoff (the AND-composition the env applies at BOTH detectable call sites).
  (3) SOFT BAND: extra miss probability raises the miss count monotonically, and -- LOAD-BEARING for
      the noise-0 calibration boot -- blur is NOT scaled by noise_scale (camera physics survives a
      perfect-estimator stage; a blur-free boot would re-learn spin-scan).
  (4) DETERMINISM: the blur-gated detectable composition is a pure function of current truth (the
      two-call-site consistency contract at peregrine_racing_ego.py 'detectable is a pure stateless
      function of the current truth').
  (5) FATAL SPIN ABORT: the sustained-rate clock (reused inc8 bsr3_update) + the leaky accumulated-
      rotation trigger, including the brief-aggressive-bank LEGALITY case (a ~130-deg bank in 0.3 s
      aborts on NEITHER trigger) and the slow-continuous-scan FATALITY case (3.0 rad/s sustained
      aborts via the rev trigger even though it never trips the 3.5 rad/s rate clock).
  (6) YAW-COMMAND clamp: clone semantics (input never mutated), only channel 3 touched, 0 == no-op.
  (7) ENV WIRING (the load-bearing composition itself, not just the pure parts): the class methods
      _step_estimator / _current_detectable are EXECUTED against a stub self (both call sites apply
      the blur hard-cut, agree with each other, and feed the soft band to the estimator), and the
      step() source is PINNED (inspect.getsource, the repo convention from
      test_ego_parabola_latch_wiring.py) for the yaw clamp + the lethal-mask plumbing into
      floor_contact=/forfeit_mask= -- the exact spin-as-free-exit trap a silent 'simplification'
      would reintroduce while every pure-helper test stayed green.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_perception_honesty.py -q
"""
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import gate_visibility as GV                                             # noqa: E402
import peregrine_racing_ego as PRE                                       # noqa: E402
from ego_estimator import BatchedEgoEstimator, EgoEstimatorConfig        # noqa: E402
from peregrine_racing_ego import (leaky_rotation_update, clamp_yaw_command,   # noqa: E402
                                  sustained_spin_update)

DT = torch.float64


# ================================================================================================
# (1) DEFAULTS-REPRODUCE.
# ================================================================================================
def test_gate_detectable_signature_untouched_and_hard_gate_identity_at_inf():
    """The package adds NO argument to gate_detectable (the estimator's standalone fallback and every
    existing caller stay byte-identical), and the env's hard-gate composition with hi=inf is the
    identity on a RANDOMIZED pose battery."""
    import inspect
    params = list(inspect.signature(GV.gate_detectable).parameters)
    assert params == ["drone_pos", "drone_quat_or_R", "gate_pos", "gate_yaw", "far_cap_m", "is_quat"]

    g = torch.Generator().manual_seed(1234)
    N, G = 64, 3
    drone_pos = torch.randn(N, 3, generator=g, dtype=DT) * 5.0
    gate_pos = torch.randn(N, G, 3, generator=g, dtype=DT) * 10.0
    gate_pos[..., 0] += 12.0                                    # bias ahead so some are detectable
    gate_yaw = torch.randn(N, G, generator=g, dtype=DT)
    q = torch.randn(N, 4, generator=g, dtype=DT)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    det, n = GV.gate_detectable(drone_pos, q, gate_pos, gate_yaw, is_quat=True)
    assert bool(det.any()) and not bool(det.all())              # a mixed battery (non-degenerate)

    w = torch.randn(N, 3, generator=g, dtype=DT) * 3.0
    rate = GV.gate_los_perp_rate(drone_pos, q, gate_pos, w, is_quat=True)
    det_composed = det & (rate < math.inf)
    assert bool((det_composed == det).all())                    # hi=inf -> bit-identical


def _mk_estimator(n, gate_pos, gate_yaw, seed, **cfg_kw):
    gen = torch.Generator().manual_seed(seed)
    cfg = EgoEstimatorConfig(**cfg_kw) if cfg_kw else EgoEstimatorConfig()
    est = BatchedEgoEstimator(n, gate_pos, gate_yaw, config=cfg, dtype=DT, generator=gen)
    return est


def _run_seq(est, blur_extra_miss_fn):
    """Drive a fixed propagate/mask/reacquire sequence; return stacked outputs. detectable toggles
    OFF for steps 6-24 (long enough to cross the 0.5 s stale horizon at dt=1/30 -> MASK) then back ON
    (re-acquisition snap)."""
    n, G = est.n, est.G
    dt = 1.0 / 30.0
    idx = torch.arange(n)
    pos = torch.zeros(n, 3, dtype=DT)
    vel = torch.zeros(n, 3, dtype=DT)
    vel[:, 0] = 2.0
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DT).expand(n, 4).contiguous()
    rates = torch.zeros(n, 3, dtype=DT)
    est.reset_idx(idx, pos, vel, quat)
    outs = []
    for t in range(30):
        pos = pos + vel * dt
        det_on = not (6 <= t < 25)
        det = torch.full((n, G), det_on, dtype=torch.bool)
        bem = blur_extra_miss_fn(n, G)
        e = est.step(pos, vel, quat, rates, dt, detectable=det, blur_extra_miss=bem)
        outs.append(torch.cat([e.rel_pos.reshape(n, -1), e.confidence, e.visible_area,
                               e.velocity, e.rel_normal.reshape(n, -1)], dim=-1))
    return torch.stack(outs)


def test_estimator_blur_none_equals_explicit_zeros_golden_trajectory():
    """blur_extra_miss=None (the default every existing caller hits) is TRAJECTORY-IDENTICAL to an
    explicit zeros tensor through a propagate -> mask -> reacquire sequence: the fold-in is neutral
    at zero and consumes NO extra RNG draws (the seeded streams stay aligned)."""
    gp = torch.tensor([[10.0, 0.0, 0.0], [15.0, 5.0, 0.0]], dtype=DT)
    gy = torch.zeros(2, dtype=DT)
    a = _run_seq(_mk_estimator(8, gp, gy, seed=7), lambda n, G: None)
    b = _run_seq(_mk_estimator(8, gp, gy, seed=7), lambda n, G: torch.zeros(n, G, dtype=DT))
    assert torch.equal(a, b)


# ================================================================================================
# (2) HARD-GATE semantics + (4) determinism.
# ================================================================================================
def _level_R(n=1):
    return torch.eye(3, dtype=DT).unsqueeze(0).expand(n, 3, 3)


def test_hard_gate_kills_a_spinning_but_geometrically_visible_pose():
    """A gate that gate_detectable happily reports (perfect shutter) becomes UNDETECTABLE under the
    blur hard cut when the LOS sweeps faster than hi -- the exact spin-scan honesty fix."""
    p = torch.zeros(1, 3, dtype=DT)
    gp = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT)
    gy = torch.zeros(1, 1, dtype=DT)
    det, _ = GV.gate_detectable(p, _level_R(), gp, gy, is_quat=False)
    assert bool(det[0, 0])                                       # geometrically visible
    w_spin = torch.tensor([[0.0, 0.0, 9.3]], dtype=DT)           # the vn16 corkscrew |w_z|
    rate = GV.gate_los_perp_rate(p, _level_R(), gp, w_spin, is_quat=False)
    assert rate[0, 0].item() > 4.0                               # far beyond the hard cutoff
    assert not bool((det & (rate < 4.0))[0, 0])                  # blur-gated: NOT detectable
    # holding steady: unchanged.
    w_hold = torch.zeros(1, 3, dtype=DT)
    rate2 = GV.gate_los_perp_rate(p, _level_R(), gp, w_hold, is_quat=False)
    assert bool((det & (rate2 < 4.0))[0, 0])


def test_blur_gated_detectable_is_deterministic_across_invocations():
    """The env computes the blur-gated detectable TWICE per step (_step_estimator + the stateless
    _current_detectable). Both compose pure functions of the SAME current truth, so a double
    invocation must be bit-identical -- the PURE-FUNCTION half of the two-call-site consistency
    contract (the env methods themselves are executed in section (7) below)."""
    g = torch.Generator().manual_seed(99)
    N, G = 32, 2
    p = torch.randn(N, 3, generator=g, dtype=DT) * 3.0
    gp = torch.randn(N, G, 3, generator=g, dtype=DT) * 8.0
    gp[..., 0] += 10.0
    gy = torch.randn(N, G, generator=g, dtype=DT) * 0.3
    q = torch.randn(N, 4, generator=g, dtype=DT)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w = torch.randn(N, 3, generator=g, dtype=DT) * 3.0

    def gated():
        det, _ = GV.gate_detectable(p, q, gp, gy, is_quat=True)
        rate = GV.gate_los_perp_rate(p, q, gp, w, is_quat=True)
        return det & (rate < 4.0)

    a, b = gated(), gated()
    assert torch.equal(a, b)
    assert bool(a.any()) and not bool(a.all())                   # non-degenerate battery


# ================================================================================================
# (3) SOFT BAND: monotone extra miss + the noise-0 survival property (LOAD-BEARING for the boot).
# ================================================================================================
def _fresh_fix_fraction(extra_miss_val, noise_scale=1.0, seed=3):
    n = 4000
    gp = torch.tensor([[10.0, 0.0, 0.0]], dtype=DT)
    gy = torch.zeros(1, dtype=DT)
    est = _mk_estimator(n, gp, gy, seed=seed, noise_scale=noise_scale)
    idx = torch.arange(n)
    pos = torch.zeros(n, 3, dtype=DT)
    vel = torch.zeros(n, 3, dtype=DT)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DT).expand(n, 4).contiguous()
    est.reset_idx(idx, pos, vel, quat)
    det = torch.ones(n, 1, dtype=torch.bool)
    bem = None if extra_miss_val is None else torch.full((n, 1), float(extra_miss_val), dtype=DT)
    e = est.step(pos, vel, quat, torch.zeros(n, 3, dtype=DT), 1.0 / 30.0,
                 detectable=det, blur_extra_miss=bem)
    # a FRESH fix this step resets the staleness clock -> confidence == 1 exactly.
    return float((e.confidence[:, 0] == 1.0).float().mean())


def test_soft_band_raises_miss_monotonically():
    f0 = _fresh_fix_fraction(0.0)
    f3 = _fresh_fix_fraction(0.3)
    f7 = _fresh_fix_fraction(0.7)
    f10 = _fresh_fix_fraction(1.0)
    # baseline ~ 1 - miss_prob (0.10); then strictly monotone-decreasing with the added band, 0 at 1.0.
    assert f0 == pytest.approx(0.9, abs=0.03)
    assert f0 > f3 > f7 > f10
    assert f10 == 0.0                                            # miss prob saturates at 1


def test_blur_survives_noise_scale_zero_the_boot_stage_property():
    """noise_scale=0 (the calibration boot) zeroes EVERY estimator corruption -- but NOT the blur
    extra miss (camera physics). A blur-saturated gate gets NO fix even on the perfect-estimator
    boot; with no blur the boot sees every fix. If this pin ever breaks, the noise-0 boot re-learns
    the spin-scan gait and the fullstack inherits it."""
    assert _fresh_fix_fraction(None, noise_scale=0.0) == 1.0     # perfect estimator: every fix lands
    assert _fresh_fix_fraction(1.0, noise_scale=0.0) == 0.0      # blur still bites at noise 0
    # partial band also survives noise-0 (not just the saturated end).
    mid = _fresh_fix_fraction(0.5, noise_scale=0.0)
    assert 0.35 < mid < 0.65


# ================================================================================================
# (5) FATAL SPIN ABORT: sustained clock (reused inc8 bsr3) + leaky rotation trigger.
# ================================================================================================
def test_sustained_spin_clock_boundaries():
    dt = 0.05
    clock = torch.zeros(1, dtype=DT)
    w = torch.tensor([[0.0, 0.0, 3.6]], dtype=DT)               # above the 3.5 abort
    aborts = []
    for _ in range(6):                                           # 0.30 s sustained
        clock, ab = sustained_spin_update(clock, w, dt, 3.5, 0.4)
        aborts.append(bool(ab[0]))
    assert not any(aborts)                                       # 0.30 s <= 0.4 -> legal
    for _ in range(3):                                           # -> 0.45 s sustained
        clock, ab = sustained_spin_update(clock, w, dt, 3.5, 0.4)
    assert bool(ab[0])                                           # sustained past 0.4 s -> FATAL

    # dipping below the rate resets the clock.
    clock = torch.tensor([0.35], dtype=DT)
    clock, ab = sustained_spin_update(clock, torch.tensor([[0.0, 0.0, 1.0]], dtype=DT), dt, 3.5, 0.4)
    assert clock[0].item() == 0.0 and not bool(ab[0])

    # rate_abort <= 0 -> permanently disarmed (the default-off contract).
    clock = torch.zeros(1, dtype=DT)
    for _ in range(100):
        clock, ab = sustained_spin_update(clock, torch.tensor([[0.0, 0.0, 50.0]], dtype=DT),
                                          dt, 0.0, 0.4)
        assert not bool(ab[0])


def test_sustained_spin_clock_exact_time_abort_boundary_is_strict():
    """The abort fires on new_clock > time_abort STRICTLY (inc8 bsr3 semantics): a clock landing
    EXACTLY on the threshold is still legal; the next over-rate tick fires. Binary-exact dt/threshold
    (0.125 / 0.375) so float accumulation cannot smear the boundary."""
    dt, time_abort = 0.125, 0.375
    clock = torch.zeros(1, dtype=DT)
    w = torch.tensor([[0.0, 0.0, 3.6]], dtype=DT)
    for _ in range(3):                                           # clock -> exactly 0.375
        clock, ab = sustained_spin_update(clock, w, dt, 3.5, time_abort)
    assert clock[0].item() == time_abort and not bool(ab[0])     # == threshold: NOT fatal (strict >)
    clock, ab = sustained_spin_update(clock, w, dt, 3.5, time_abort)
    assert bool(ab[0])                                           # first tick past it: FATAL


def test_leaky_rotation_slow_continuous_scan_is_fatal_but_2rads_is_not():
    """Steady-state accum ~ ||omega|| * window: at window 4.0 s / 1.5 rev (9.42 rad) a constant
    2.0 rad/s tops out ~8.0 (legal) while 3.0 rad/s crosses (FATAL) -- the slow-but-continuous
    scan the sustained-rate clock (3.5) would never catch."""
    dt = 1.0 / 30.0
    for w_mag, should_abort in ((2.0, False), (3.0, True)):
        accum = torch.zeros(1, dtype=DT)
        w = torch.tensor([[0.0, 0.0, w_mag]], dtype=DT)
        fired = False
        for _ in range(int(30.0 / dt)):                          # 30 s -- way past settling
            accum, ab = leaky_rotation_update(accum, w, dt, 4.0, 1.5)
            fired = fired or bool(ab[0])
        assert fired == should_abort, (w_mag, accum.item())
        if not should_abort:
            assert accum[0].item() < 2.0 * math.pi * 1.5         # settled strictly under the bar


def test_brief_aggressive_bank_is_legal_on_both_triggers():
    """The A2-observed class: a single ~130-deg (2.27 rad) bank flown in 0.3 s (~7.56 rad/s). Rate
    is way above 3.5 but NOT sustained past 0.4 s; the rotation impulse decays in the leaky window.
    BOTH triggers must stay silent -- aggressive banking through a gate remains legal."""
    dt = 0.05
    clock = torch.zeros(1, dtype=DT)
    accum = torch.zeros(1, dtype=DT)
    w_bank = torch.tensor([[7.56, 0.0, 0.0]], dtype=DT)          # roll impulse (all-axis rule)
    w_calm = torch.zeros(1, 3, dtype=DT)
    for t in range(200):                                          # 10 s: 0.3 s bank then calm flight
        w = w_bank if t < 6 else w_calm
        clock, ab1 = sustained_spin_update(clock, w, dt, 3.5, 0.4)
        accum, ab2 = leaky_rotation_update(accum, w, dt, 4.0, 1.5)
        assert not bool(ab1[0]) and not bool(ab2[0]), t
    assert accum[0].item() < 0.5                                  # the impulse decayed away


def test_leaky_rotation_disarmed_at_zero_rev_abort():
    dt = 1.0 / 30.0
    accum = torch.zeros(1, dtype=DT)
    w = torch.tensor([[0.0, 0.0, 50.0]], dtype=DT)
    for _ in range(1000):
        accum, ab = leaky_rotation_update(accum, w, dt, 4.0, 0.0)
        assert not bool(ab[0])                                   # rev_abort=0 -> never (default-off)


def test_spin_abort_is_all_axis_not_yaw_only():
    """A roll/pitch tumble-scan (the yaw-clamp evasion gait) trips the SAME gates: the triggers read
    ||omega||, never w_z alone."""
    dt = 0.05
    clock = torch.zeros(1, dtype=DT)
    w_tumble = torch.tensor([[4.0, 2.0, 0.0]], dtype=DT)         # zero yaw rate, |w| ~ 4.47
    fired = False
    for _ in range(12):                                          # 0.6 s sustained tumble
        clock, ab = sustained_spin_update(clock, w_tumble, dt, 3.5, 0.4)
        fired = fired or bool(ab[0])
    assert fired


# ================================================================================================
# (6) YAW-COMMAND clamp (invariant 3: action space untouched; applied-command clamp only).
# ================================================================================================
def test_clamp_yaw_command_clone_semantics_and_channel_isolation():
    act = torch.tensor([[1.0, 2.0, -2.5, 3.14], [0.2, -0.1, 0.4, -3.0]], dtype=DT)
    orig = act.clone()
    out = clamp_yaw_command(act, 0.35)
    assert torch.equal(act, orig)                                # INPUT NEVER MUTATED (caller holds refs)
    assert out is not act
    assert torch.equal(out[:, :3], act[:, :3])                   # thrust/roll/pitch untouched
    assert out[0, 3].item() == pytest.approx(0.35)
    assert out[1, 3].item() == pytest.approx(-0.35)
    # inside the clamp: passes through unchanged.
    act2 = torch.tensor([[1.0, 0.0, 0.0, 0.2]], dtype=DT)
    assert torch.equal(clamp_yaw_command(act2, 0.35), act2)


def test_clamp_yaw_command_zero_is_identity_no_copy():
    act = torch.tensor([[1.0, 2.0, -2.5, 3.14]], dtype=DT)
    out = clamp_yaw_command(act, 0.0)
    assert out is act                                            # 0 == OFF == the exact input tensor


# ================================================================================================
# (7) ENV WIRING -- the composition inside PeregrineRacingEgo itself. The class body imports fine on
# the laptop (PeregrineRacing base + inc8_estimator_emul); only construction/step need diffaero's
# dynamics, so _step_estimator / _current_detectable are EXECUTED here against a stub self
# (object.__new__, no __init__), and step()'s un-executable plumbing is pinned by source
# (the repo convention: test_ego_parabola_latch_wiring.py / test_ego_obs_coast.py).
# ================================================================================================
class _RecordingEstimator:
    """Stands in for BatchedEgoEstimator inside _step_estimator; records exactly what the env feeds
    the estimator (the detectable mask AFTER the blur cut + the soft-band extra miss)."""
    def __init__(self):
        self.calls = []

    def step(self, p, v, q, w, dt, detectable=None, prev_quat=None, apparent_area=None,
             blur_extra_miss=None, sf_body=None, gyro_sample=None):
        self.calls.append({
            "detectable": detectable.clone(),
            "blur_extra_miss": None if blur_extra_miss is None else blur_extra_miss.clone(),
            "sf_body": None if sf_body is None else sf_body.clone(),
        })
        return "EST-SENTINEL"


def _mk_env_wiring_stub(w, blur_gate=True):
    """A minimal stub self for the two detectable call sites: env0 = the vn16 spinner (|w_z|=9.3),
    env1 = holding steady; ONE gate dead ahead (the geometry the pure-function battery already
    proves detectable). _cam_flip False isolates the WIRING under test (the flip has its own pins)."""
    if not PRE._HAVE_DIFFAERO:
        pytest.skip("peregrine_racing base / inc8_estimator_emul not importable here")
    n = w.shape[0]
    stub = object.__new__(PRE.PeregrineRacingEgo)
    stub._p = torch.zeros(n, 3, dtype=DT)
    stub._q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DT).expand(n, 4).contiguous()
    stub._v = torch.zeros(n, 3, dtype=DT)
    stub._w = w
    stub.gate_pos = torch.tensor([[10.0, 0.0, 0.0]], dtype=DT).expand(n, 3).reshape(n, 1, 3).contiguous()
    stub.gate_yaw = torch.zeros(n, 1, dtype=DT)
    stub.dt = 1.0 / 30.0
    stub._cam_flip = False
    stub._ego_cfg = SimpleNamespace(far_cap_m=30.0)
    stub._blur_gate = blur_gate
    stub._blur_rate_lo = 2.0
    stub._blur_rate_hi = 4.0
    stub._blur_miss_max = 1.0
    # estimator-faithful package (2026-07-11): default-off in these wiring tests (the faithful
    # wiring has its own executed stub tests in test_estimator_faithful.py)
    stub._est_needs_sf = False
    stub._estimator = _RecordingEstimator()
    return stub


def test_env_call_sites_execute_the_blur_cut_and_agree():
    """EXECUTES the real _step_estimator + _current_detectable (not a re-implementation): the
    spinner's geometrically-visible gate is blur-cut at BOTH call sites, the steady drone's is not,
    the two call sites agree bit-for-bit on the same truth state, and the estimator receives the
    blur-cut mask + the saturated soft-band extra miss. This is the coverage the pure-helper tests
    cannot give: dropping the blur AND from either call site fails HERE while everything else stays
    green."""
    w = torch.tensor([[0.0, 0.0, 9.3],                           # env0: the vn16 corkscrew
                      [0.0, 0.0, 0.0]], dtype=DT)                # env1: holding steady
    stub = _mk_env_wiring_stub(w)
    prev_q = stub._q.clone()

    est, det_step = PRE.PeregrineRacingEgo._step_estimator(stub, prev_q)
    assert est == "EST-SENTINEL"                                 # the estimator's return is passed through
    assert det_step.tolist() == [[False], [True]]                # blur kills the spinner's fix stream
    assert stub._stepped is True
    assert torch.equal(stub._last_detectable, det_step)          # the duty diagnostic reads this
    rec = stub._estimator.calls[-1]
    assert torch.equal(rec["detectable"], det_step)              # estimator got the BLUR-CUT mask
    assert rec["blur_extra_miss"] is not None
    assert rec["blur_extra_miss"][0, 0].item() == pytest.approx(1.0)   # 9.3 >> hi: saturated
    assert rec["blur_extra_miss"][1, 0].item() == pytest.approx(0.0)   # steady: blur-free band

    det_obs = PRE.PeregrineRacingEgo._current_detectable(stub)
    assert torch.equal(det_obs, det_step)                        # the two-call-site agreement, EXECUTED


def test_env_call_sites_blur_off_is_geometry_only_and_no_extra_miss():
    """The default-off contract, executed through the same wiring: _blur_gate False -> both call
    sites return pure gate_detectable geometry (the spinner IS detectable again -- the perfect
    shutter) and the estimator receives blur_extra_miss=None (byte-identical existing behavior)."""
    w = torch.tensor([[0.0, 0.0, 9.3], [0.0, 0.0, 0.0]], dtype=DT)
    stub = _mk_env_wiring_stub(w, blur_gate=False)
    _, det_step = PRE.PeregrineRacingEgo._step_estimator(stub, stub._q.clone())
    assert det_step.tolist() == [[True], [True]]                 # perfect shutter: spin is free
    assert stub._estimator.calls[-1]["blur_extra_miss"] is None
    assert torch.equal(PRE.PeregrineRacingEgo._current_detectable(stub), det_step)


def test_step_source_pins_yaw_clamp_and_lethal_mask_plumbing():
    """SOURCE-PIN (getsource, the parabola-latch-wiring convention) for the step() plumbing that is
    laptop-unexecutable (needs diffaero dynamics) but load-bearing:
      * the yaw clamp is applied to the ACTION at the top of step(),
      * lethal composes below_floor | spin_abort,
      * lethal (NOT bare below_floor) rides floor_contact= AND forfeit_mask= into the reward -- the
        spin-as-free-exit trap: reverting either to below_floor makes a spin abort a penalty-free
        episode exit under the champion parabola regime while the full pure-helper suite stays green,
      * the gate_collision fold and the oob exclusion stay consistent with the lethal mask,
    plus the blur AND at BOTH detectable call sites (dropping either is the silent honesty rollback)."""
    src_step = inspect.getsource(PRE.PeregrineRacingEgo.step)
    assert "clamp_yaw_command(action" in src_step
    assert "lethal = below_floor | spin_abort" in src_step
    assert "floor_contact=lethal.to(" in src_step
    assert "forfeit_mask=(lethal.to(" in src_step
    assert "gate_collision = gate_collision | lethal" in src_step
    assert "oob = oob_full & ~lethal" in src_step

    src_se = inspect.getsource(PRE.PeregrineRacingEgo._step_estimator)
    src_cd = inspect.getsource(PRE.PeregrineRacingEgo._current_detectable)
    for src in (src_se, src_cd):
        assert "detectable & (los_rate < self._blur_rate_hi)" in src
    assert "self._last_detectable = detectable" in src_se        # the duty diagnostic's source
    assert "blur_extra_miss=extra_miss" in src_se                # the soft band reaches the estimator
