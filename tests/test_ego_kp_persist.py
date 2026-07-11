"""Tests for the KEYPOINT-PERSISTENCE debounce (Fengyou 2026-07-11): a gate transmits (estimator
fix + obs slot feed) only after the detectability condition held N CONSECUTIVE detection
opportunities; a single miss resets the streak. Train knob ``ego_kp_persist_frames`` (<=1 == OFF ==
byte-identical: "1 frame suffices" == today); deploy twin ``fly_rl.py --ego-kp-persist``.

Coverage (test_perception_honesty.py conventions -- the env class is cluster-only, so the pure
helper gets direct unit tests, the wiring is EXECUTED against a stub self, and the
laptop-unexecutable step()/reset_idx plumbing is source-pinned in
test_perception_honesty.py::test_step_source_pins_yaw_clamp_and_lethal_mask_plumbing):

  (1) PURE FN ``kp_persist_update``: streak build (n=2: first hit suppressed, transmits ON the
      2nd), single-miss reset + re-streak, saturation at n (counter never grows unbounded, stays
      transmitting), the transmit <= detectable invariant on random masks, dtype/shape
      preservation, purity (inputs never mutated; no generator arg -- RNG-NEUTRAL by construction).
  (2) ENV WIRING, EXECUTED (recording estimator): armed n=2 the estimator receives False on the
      first detectable tick and True on the second; a miss (gate moved behind) resets and the
      re-streak suppresses exactly one tick again; the debounce input is the POST-BLUR-AND mask
      (a blur-cut tick is a miss -- no double interaction); _current_detectable agrees with the
      mask the estimator received (two-site agreement) and is READ-ONLY (0..N calls per tick never
      advance the counter); OFF (n=0/1) never suppresses and never touches the counter.
  (3) GOLDEN TRAJECTORY through the REAL BatchedEgoEstimator driven by the real _step_estimator:
      n=0 vs n=1 bit-identical (the OFF contract; knob-ABSENT == 0 is pinned at the getattr
      default in the source-pin test -- the env cannot be constructed on the laptop); armed n=2
      with never-detectable geometry bit-identical to OFF (RNG neutrality: the debounce adds no
      draws and shifts none); armed n=2 with a visible gate delays the first fresh fix by exactly
      one tick (the seeded miss-draw streams stay aligned).
  (4) RESET CLEARS: the exact reset_idx / gate-advance clear lines simulated on a counter (the
      test_ego_parabola_latch_wiring.py convention -- the source pins guarantee the lines exist in
      the env; this documents their ROW-clear semantics: gate advance restarts the streak for the
      WHOLE advancing env, matching deploy's single-track seeker.reset()).

Matched-pair semantic gaps (DESIGN doc S3, abbreviated): train opportunity = 33 ms estimator tick,
deploy = fresh camera frame (near-matched at 30 Hz, deploy stricter in wall-time under frame
starvation); train debounces the raycast+blur gate-level condition, deploy debounces the accepted
PnP pose (keypoint counts unobservable on the wire); BOTH sides reset on gate advance; deploy
resets on ANY fresh-frame None incl. continuity_reject == train's any-non-detectable-tick reset;
ego_estimator.py's standalone detectable=None fallback bypasses the debounce (env-only knob).

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_kp_persist.py -q
"""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import peregrine_racing_ego as PRE                                        # noqa: E402
from peregrine_racing_ego import kp_persist_update                        # noqa: E402
from ego_estimator import BatchedEgoEstimator, EgoEstimatorConfig         # noqa: E402

DT = torch.float64


# ================================================================================================
# (1) PURE FN kp_persist_update.
# ================================================================================================
def test_streak_build_transmits_on_the_nth_hit_and_saturates():
    count = torch.zeros(2, 1, dtype=torch.long)
    det = torch.ones(2, 1, dtype=torch.bool)
    count, tx = kp_persist_update(count, det, 2)
    assert count.tolist() == [[1], [1]] and tx.tolist() == [[False], [False]]   # 1st hit suppressed
    count, tx = kp_persist_update(count, det, 2)
    assert count.tolist() == [[2], [2]] and tx.tolist() == [[True], [True]]     # transmits ON the 2nd
    for _ in range(5):
        count, tx = kp_persist_update(count, det, 2)
    assert int(count.max()) == 2 and bool(tx.all())          # saturates at n, stays transmitting


def test_single_miss_resets_and_restreak_pays_the_full_debounce_again():
    count = torch.zeros(1, 1, dtype=torch.long)
    seq = [True, True, False, True, True]
    want_tx = [False, True, False, False, True]
    want_count = [1, 2, 0, 1, 2]
    for d, w_tx, w_c in zip(seq, want_tx, want_count):
        det = torch.tensor([[d]])
        count, tx = kp_persist_update(count, det, 2)
        assert bool(tx[0, 0]) is w_tx and int(count[0, 0]) == w_c


def test_n3_needs_three_consecutive_hits():
    count = torch.zeros(1, 1, dtype=torch.long)
    det = torch.ones(1, 1, dtype=torch.bool)
    seen = []
    for _ in range(4):
        count, tx = kp_persist_update(count, det, 3)
        seen.append(bool(tx[0, 0]))
    assert seen == [False, False, True, True]


def test_transmit_le_detectable_dtype_shape_and_saturation_on_random_masks():
    g = torch.Generator().manual_seed(7)
    for n_required in (2, 3, 5):
        c = torch.zeros(16, 3, dtype=torch.long)
        for _ in range(40):
            det = torch.rand(16, 3, generator=g) < 0.6
            c, tx = kp_persist_update(c, det, n_required)
            assert c.dtype == torch.long and tx.dtype == torch.bool
            assert c.shape == (16, 3) and tx.shape == (16, 3)
            assert bool((tx <= det).all())                   # transmit <= detectable ALWAYS
            assert int(c.max()) <= n_required                # never grows unbounded


def test_pure_deterministic_no_rng_and_inputs_never_mutated():
    # RNG-NEUTRAL by construction: no generator parameter, no draws.
    assert "generator" not in inspect.signature(kp_persist_update).parameters
    g = torch.Generator().manual_seed(3)
    count = torch.randint(0, 3, (8, 2), generator=g)
    det = torch.rand(8, 2, generator=g) < 0.5
    c_in, d_in = count.clone(), det.clone()
    out1 = kp_persist_update(count, det, 2)
    out2 = kp_persist_update(count, det, 2)
    assert torch.equal(out1[0], out2[0]) and torch.equal(out1[1], out2[1])   # deterministic
    assert torch.equal(count, c_in) and torch.equal(det, d_in)               # inputs untouched


def test_n_below_two_is_a_caller_bug_and_asserts():
    # HARDENING (audit 2026-07-11): with n <= 0, ``new_count >= n`` would emit transmit=True
    # on NON-detectable gates (0 >= 0), violating transmit <= detectable. OFF (<=1) lives at
    # the call sites (self._kp_persist_n >= 2 guard); the pure fn refuses n < 2 outright.
    count = torch.zeros(1, 2, dtype=torch.long)
    det = torch.tensor([[True, False]])
    for bad_n in (-1, 0, 1):
        with pytest.raises(AssertionError):
            kp_persist_update(count, det, bad_n)


# ================================================================================================
# (2) ENV WIRING, EXECUTED (the test_perception_honesty stub-self convention).
# ================================================================================================
class _RecordingEstimator:
    """Records exactly the detectable mask the env feeds the estimator each tick."""
    def __init__(self):
        self.calls = []

    def step(self, p, v, q, w, dt, detectable=None, prev_quat=None, apparent_area=None,
             blur_extra_miss=None, sf_body=None, gyro_sample=None):
        self.calls.append({"detectable": detectable.clone()})
        return "EST-SENTINEL"


def _mk_kp_stub(kp_n, n=1, estimator=None, blur_gate=False, w=None):
    """Stub self for _step_estimator/_current_detectable with the kp-persist debounce armed at
    ``kp_n``. One gate dead ahead at x=+10 (the geometry the perception-honesty battery proves
    detectable); a MISS tick is injected by moving stub.gate_pos behind the drone (x=-10)."""
    if not PRE._HAVE_DIFFAERO:
        pytest.skip("peregrine_racing base / inc8_estimator_emul not importable here")
    stub = object.__new__(PRE.PeregrineRacingEgo)
    stub._p = torch.zeros(n, 3, dtype=DT)
    stub._q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=DT).expand(n, 4).contiguous()
    stub._v = torch.zeros(n, 3, dtype=DT)
    stub._w = w if w is not None else torch.zeros(n, 3, dtype=DT)
    stub.gate_pos = torch.tensor([[10.0, 0.0, 0.0]], dtype=DT).expand(n, 3).reshape(n, 1, 3).contiguous()
    stub.gate_yaw = torch.zeros(n, 1, dtype=DT)
    stub.dt = 1.0 / 30.0
    stub._cam_flip = False
    stub._ego_cfg = SimpleNamespace(far_cap_m=30.0)
    stub._blur_gate = blur_gate
    stub._blur_rate_lo = 2.0
    stub._blur_rate_hi = 4.0
    stub._blur_miss_max = 1.0
    stub._est_needs_sf = False
    stub._kp_persist_n = kp_n
    stub._kp_persist_count = torch.zeros(n, 1, dtype=torch.long)
    stub._estimator = estimator if estimator is not None else _RecordingEstimator()
    return stub


def _tick(stub, behind=False):
    stub.gate_pos[..., 0] = -10.0 if behind else 10.0
    est, det = PRE.PeregrineRacingEgo._step_estimator(stub, stub._q.clone())
    return est, det


def test_armed_streak_miss_reset_and_two_site_agreement_executed():
    """EXECUTES the real _step_estimator/_current_detectable armed at n=2: first detectable tick ->
    the estimator receives False (suppressed), second -> True; an injected miss resets; the
    re-streak suppresses exactly one tick again; and after EVERY tick _current_detectable returns
    bit-for-bit the mask the estimator received (the two-site agreement, debounced)."""
    stub = _mk_kp_stub(kp_n=2)
    script = [(False, False), (False, True), (False, True),   # streak: suppress, transmit, saturated
              (True, False),                                  # miss (gate behind): reset
              (False, False), (False, True)]                  # re-streak: suppress once, transmit
    for i, (behind, want) in enumerate(script):
        _, det = _tick(stub, behind=behind)
        assert det.tolist() == [[want]], (i, det.tolist())
        rec = stub._estimator.calls[-1]["detectable"]
        assert torch.equal(rec, det)                          # the estimator got the debounced mask
        assert torch.equal(PRE.PeregrineRacingEgo._current_detectable(stub), det), i
    assert stub._last_detectable.tolist() == [[True]]         # duty diag reads the TRANSMITTED mask


def test_current_detectable_is_read_only_executed():
    """_current_detectable may run 0..N times per tick (obs + lazy first-obs paths) and must NEVER
    advance the streak: three back-to-back calls return identical masks and leave the counter
    untouched, both mid-streak (count 1 -> still suppressed) and post-streak (count 2)."""
    stub = _mk_kp_stub(kp_n=2)
    _tick(stub)                                               # count -> 1 (suppressed tick)
    for _ in range(3):
        assert PRE.PeregrineRacingEgo._current_detectable(stub).tolist() == [[False]]
    assert stub._kp_persist_count.tolist() == [[1]]           # READ-ONLY: no advance
    _tick(stub)                                               # count -> 2 (transmitting)
    for _ in range(3):
        assert PRE.PeregrineRacingEgo._current_detectable(stub).tolist() == [[True]]
    assert stub._kp_persist_count.tolist() == [[2]]


def test_off_path_never_suppresses_and_never_touches_the_counter():
    """<=1 == OFF: n=0 and n=1 transmit on the very first detectable tick (today's behavior) and
    the counter buffer is never advanced (the guard short-circuits before touching it)."""
    for kp_n in (0, 1):
        stub = _mk_kp_stub(kp_n=kp_n)
        _, det = _tick(stub)
        assert det.tolist() == [[True]], kp_n                 # one frame suffices
        assert torch.equal(PRE.PeregrineRacingEgo._current_detectable(stub), det)
        assert stub._kp_persist_count.tolist() == [[0]], kp_n # buffer untouched


def test_debounce_input_is_the_post_blur_and_mask_executed():
    """The debounce sits AFTER the blur hard-cut (never double-interacts): with blur armed, the
    vn16 spinner's blur-cut ticks are MISSES (its streak never starts) while the steady drone
    debounces normally (suppress, then transmit)."""
    w = torch.tensor([[0.0, 0.0, 9.3],                        # env0: the vn16 corkscrew (blur-cut)
                      [0.0, 0.0, 0.0]], dtype=DT)             # env1: holding steady
    stub = _mk_kp_stub(kp_n=2, n=2, blur_gate=True, w=w)
    _, det = _tick(stub)
    assert det.tolist() == [[False], [False]]                 # env0 blur; env1 debounce-suppressed
    assert stub._kp_persist_count.tolist() == [[0], [1]]      # blur-cut tick == a MISS (count 0)
    _, det = _tick(stub)
    assert det.tolist() == [[False], [True]]                  # env1 transmits on its 2nd hit
    assert stub._kp_persist_count.tolist() == [[0], [2]]


# ================================================================================================
# (3) GOLDEN TRAJECTORY through the REAL estimator (OFF byte-identity + RNG neutrality).
# ================================================================================================
def _drive_real(kp_n, behind_ticks, n=6, seed=11, ticks=24):
    """Drive the REAL BatchedEgoEstimator through the real _step_estimator for ``ticks`` control
    ticks; detectability toggles via gate-behind on ``behind_ticks``. Returns stacked outputs."""
    gp = torch.tensor([[10.0, 0.0, 0.0]], dtype=DT)
    gy = torch.zeros(1, dtype=DT)
    gen = torch.Generator().manual_seed(seed)
    est = BatchedEgoEstimator(n, gp, gy, config=EgoEstimatorConfig(), dtype=DT, generator=gen)
    stub = _mk_kp_stub(kp_n=kp_n, n=n, estimator=est)
    est.reset_idx(torch.arange(n), stub._p, stub._v, stub._q)
    outs = []
    for t in range(ticks):
        e, _ = _tick(stub, behind=(t in behind_ticks))
        outs.append(torch.cat([e.rel_pos.reshape(n, -1), e.confidence, e.visible_area,
                               e.velocity, e.rel_normal.reshape(n, -1)], dim=-1))
    return torch.stack(outs)


def test_off_byte_identity_n0_vs_n1_golden_trajectory():
    """n=0 and n=1 are BOTH the OFF contract: trajectories bit-identical through a mixed
    detectable/miss sequence (knob-ABSENT == 0 is pinned at the getattr default in
    test_perception_honesty.py's source-pin test -- the env is not constructible on the laptop)."""
    behind = {6, 7, 12}
    a = _drive_real(0, behind)
    b = _drive_real(1, behind)
    assert torch.equal(a, b)


def test_rng_neutrality_armed_but_never_detectable_equals_off():
    """Armed n=2 with NEVER-detectable geometry is bit-identical to OFF: the debounce of an
    all-False mask is all-False, adds no draws and shifts none -- the seeded miss-draw streams
    stay aligned (the design's RNG-neutrality acceptance)."""
    behind = set(range(24))
    a = _drive_real(0, behind)
    b = _drive_real(2, behind)
    assert torch.equal(a, b)


def test_armed_delays_the_first_fresh_fix_by_exactly_one_tick():
    """Armed n=2 with an always-visible gate: tick 0's fix opportunity is suppressed (no env can
    show a FRESH fix, confidence == 1.0 exactly), while OFF lands fresh fixes at tick 0; at tick 1
    the armed arm's (stream-aligned) draws land -- the first fix is delayed by exactly the
    debounce, never dropped. (Output col 3 == confidence for the single gate.)"""
    a = _drive_real(0, set())                                 # OFF
    b = _drive_real(2, set())                                 # armed n=2
    conf_a0, conf_b0, conf_b1 = a[0][:, 3], b[0][:, 3], b[1][:, 3]
    assert bool((conf_a0 == 1.0).any())                       # OFF: fresh fixes land at tick 0
    assert not bool((conf_b0 == 1.0).any())                   # armed: tick 0 suppressed for ALL envs
    assert bool((conf_b1 == 1.0).any())                       # armed: fixes land on the 2nd hit
    assert not torch.equal(a, b)                              # the delay is real, not a no-op


# ================================================================================================
# (4) RESET CLEARS -- exact-line simulation (test_ego_parabola_latch_wiring.py convention; the
# lines' PRESENCE in reset_idx/step is source-pinned in test_perception_honesty.py).
# ================================================================================================
def test_reset_idx_and_gate_advance_exact_clear_lines_row_semantics():
    """reset_idx clears by env_idx (terminated AND truncated funnel through it); the gate-advance
    clear zeroes the WHOLE advancing env row (all gate slots) -- the deploy-matching rule: fly_rl
    drops its single seeker track on advance, so a co-visible next gate must NOT arrive
    pre-debounced in train when it never would on the wire."""
    count = torch.tensor([[2, 1], [1, 2], [2, 2]], dtype=torch.long)
    env_idx = torch.tensor([0, 2])
    count[env_idx] = 0                                        # reset_idx's exact clear
    assert count.tolist() == [[0, 0], [1, 2], [0, 0]]
    count = torch.tensor([[2, 1], [1, 2], [2, 2]], dtype=torch.long)
    advance = torch.tensor([True, False, True])
    count[advance] = 0                                        # step()'s exact gate-advance clear
    assert count.tolist() == [[0, 0], [1, 2], [0, 0]]         # ROW clear, not a single-slot clear
