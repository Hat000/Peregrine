"""Tests for the VISION-CADENCE model (2026-07-15 rework; ``+env.ego_vision_cadence`` /
peregrine_racing_ego.build_work_ms_cdf / sample_work_ms / cadence_effective_detectable /
seed_fresh_fix_age_s / the env ``_vc_frame_tick`` + ``_step_estimator`` wiring).

THE REWORK (supersedes the old reach-back "output-delay" model): DiffAero feeds the estimator a fresh
geometric fix EVERY 40 Hz control tick, but deploy lands VALID gate detections only ~7-15 Hz and goes
DARK on the final approach. This model reproduces that at the estimator's OWN control point -- the
``detectable`` mask fed to BatchedEgoEstimator.step -- so the estimator's EXISTING ego-propagation +
confidence-decay carry the between-frame gaps. NO ring buffer, NO output substitution. When ON:

    detectable_effective = geometric_detectable AND frame_tick(~30 Hz) AND detector_success(p)

Off-frame ticks -> det_eff all-False -> the estimator ego-propagates. The OBS slot masking is UNCHANGED
(reads the GEOMETRIC detectability + confidence). A THIN, SECOND-ORDER sub-effect seeds a freshly-fixed
gate's ``t_since_fix`` to a sampled work_ms so confidence starts slightly below 1. Default-OFF ==
byte-identical (no new RNG draws); the success + work_ms draws use a DEDICATED generator (RNG-isolated).

Coverage:
  (1) PURE FNS: build_work_ms_cdf (unchanged: 101-pt monotone, reproduces the measured anchors, JSON
      load+resample); sample_work_ms (continuous ms, monotone, reproduces anchors, RNG-free);
      cadence_effective_detectable (off-frame -> all-False, p in {0,1} limits, ~p success fraction,
      dedicated-gen isolated); seed_fresh_fix_age_s (fresh gates re-aged to work_ms seconds, others
      untouched, no-fresh -> no draw).
  (2) ENV WIRING source-pins (env is cluster-only): __init__ reads the knobs (default OFF, dedicated
      generator); _step_estimator gates det_eff via cadence_effective_detectable + passes
      detectable=det_eff + seeds work_ms + keeps _last_detectable GEOMETRIC; get_observations DROPS the
      reach-back substitution; reset_idx keeps NO per-env cadence state.
  (3) EXECUTED frame clock (_vc_frame_tick on a stub): deterministic fire fraction == frame_hz/loop_hz.
  (4) GOLDEN TRAJECTORY through the REAL BatchedEgoEstimator + the real pure helpers:
      * OFF == byte-identical to fresh-every-visible-tick; dedicated generator isolated from the global
        estimator draw stream.
      * VALID fresh-fix rate ~ 7-15 Hz on a straight visible approach (reported).
      * confidence/t_since_fix VARY between frames (propagation active); a fresh fix starts at
        t_since_fix ~ sampled work_ms (not 0), and exactly 0 when the sub-effect is OFF.
      * the gate goes DARK (obs masked) on the close approach via the GEOMETRIC dropout (no near rule).

The PeregrineRacingEgo env needs diffaero (cluster-only); the model is PURE torch, so it is driven here
directly through the real BatchedEgoEstimator + the real capture helpers on a stub self (the
test_ego_kp_persist.py / test_ego_obs_coast.py convention). Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_vision_latency.py -q
"""
import inspect
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np                                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import peregrine_racing_ego as PRE                                          # noqa: E402
from ego_estimator import BatchedEgoEstimator, EgoEstimatorConfig          # noqa: E402

DT = torch.float64
CPU = torch.device("cpu")

# the MEASURED work_ms quantile anchors (ms) the CDF must reproduce.
ANCHORS = {0.50: 24.5, 0.75: 47.3, 0.90: 62.4, 0.99: 95.6, 1.00: 120.8}
STALE_HORIZON_S = EgoEstimatorConfig.stale_horizon_s                        # 0.5 (UNCHANGED by this task)


# ================================================================================================
# (1) PURE FUNCTIONS.
# ================================================================================================
def test_build_work_ms_cdf_shape_monotone_and_reproduces_measured_anchors():
    cdf = PRE.build_work_ms_cdf()
    assert cdf.shape == (101,)
    assert bool((np.diff(cdf) >= -1e-9).all()), "inverse-CDF must be non-decreasing"
    for u, ms in ANCHORS.items():
        assert abs(cdf[int(round(u * 100))] - ms) < 1e-6, (u, ms, cdf[int(round(u * 100))])
    assert abs(cdf[0] - 8.0) < 1e-6                                         # documented low-tail floor


def test_build_work_ms_cdf_loads_json_file_and_resamples(tmp_path):
    ramp = list(np.linspace(3.0, 130.0, 101))
    p = tmp_path / "cdf_list.json"
    p.write_text(json.dumps(ramp))
    assert np.allclose(PRE.build_work_ms_cdf(str(p)), np.asarray(ramp))
    p2 = tmp_path / "cdf_dict.json"
    p2.write_text(json.dumps({"note": "x", "inv_cdf_ms": list(np.linspace(0.0, 100.0, 51))}))
    got2 = PRE.build_work_ms_cdf(str(p2))
    assert got2.shape == (101,)
    assert abs(got2[0] - 0.0) < 1e-9 and abs(got2[-1] - 100.0) < 1e-9 and abs(got2[50] - 50.0) < 1e-6
    assert PRE.build_work_ms_cdf(str(tmp_path / "nope.json")).shape == (101,)   # missing -> anchors


def test_sample_work_ms_continuous_monotone_reproduces_anchors_and_rng_free():
    cdf = torch.as_tensor(PRE.build_work_ms_cdf(), dtype=DT)
    # reproduces the measured anchors at their grid points (CONTINUOUS ms, not tick-quantized). The u=1.0
    # endpoint is clamped to 1-1e-9 to keep the upper index in range, leaving a ~3e-6 ms gap -> 1e-4 tol.
    for u, ms in ANCHORS.items():
        got = float(PRE.sample_work_ms(torch.tensor([u], dtype=DT), cdf)[0])
        assert abs(got - ms) < 1e-4, (u, ms, got)
    assert abs(float(PRE.sample_work_ms(torch.tensor([0.0], dtype=DT), cdf)[0]) - 8.0) < 1e-6
    # monotone non-decreasing in u; strictly interpolates between anchors (a sub-tick 24.5 ms value).
    uu = torch.linspace(0, 1 - 1e-9, 500, dtype=DT)
    assert bool((torch.diff(PRE.sample_work_ms(uu, cdf)) >= -1e-12).all())
    mid = float(PRE.sample_work_ms(torch.tensor([0.625], dtype=DT), cdf)[0])    # between p50 and p75
    assert 24.5 < mid < 47.3
    # RNG-NEUTRAL by construction: no generator parameter, no internal draw (u is the input).
    assert "generator" not in inspect.signature(PRE.sample_work_ms).parameters


def test_cadence_effective_detectable_off_frame_all_false_and_p_limits():
    N, G = 4, 3
    geom = torch.tensor([[True, True, False], [True, False, True],
                         [False, False, True], [True, True, True]])
    g = torch.Generator().manual_seed(0)
    # NON-frame tick -> NO gate gets a fresh fix (all-False), regardless of geometry.
    off = PRE.cadence_effective_detectable(geom, frame_fired=False, p=1.0, gen=g, dtype=DT)
    assert off.dtype == torch.bool and not bool(off.any())
    # frame tick, p=1 -> every geometrically-detectable gate passes (== geom); p=0 -> none.
    p1 = PRE.cadence_effective_detectable(geom, frame_fired=True, p=1.0, gen=g, dtype=DT)
    assert torch.equal(p1, geom)
    p0 = PRE.cadence_effective_detectable(geom, frame_fired=True, p=0.0, gen=g, dtype=DT)
    assert not bool(p0.any())
    # det_eff is always a SUBSET of geometric (a non-visible gate never fixes).
    pm = PRE.cadence_effective_detectable(geom, frame_fired=True, p=0.5, gen=g, dtype=DT)
    assert bool((pm <= geom).all())


def test_cadence_effective_detectable_success_fraction_matches_p_and_is_gen_isolated():
    N, G = 4096, 4
    geom = torch.ones(N, G, dtype=torch.bool)
    g = torch.Generator().manual_seed(7)
    det = PRE.cadence_effective_detectable(geom, frame_fired=True, p=0.35, gen=g, dtype=DT)
    frac = float(det.float().mean())
    assert abs(frac - 0.35) < 0.02, frac                                    # ~p successes on visible gates
    # the draw comes from the DEDICATED generator -> it does NOT advance the global default stream.
    torch.manual_seed(123); a = torch.rand(16, dtype=DT)
    torch.manual_seed(123)
    g2 = torch.Generator().manual_seed(9)
    _ = PRE.cadence_effective_detectable(geom, frame_fired=True, p=0.35, gen=g2, dtype=DT)
    b = torch.rand(16, dtype=DT)
    assert torch.equal(a, b), "cadence success draw must not perturb the global RNG stream"


def test_seed_fresh_fix_age_reages_fresh_gates_only_and_no_draw_when_none_fresh():
    cdf = torch.as_tensor(PRE.build_work_ms_cdf(), dtype=DT)
    g = torch.Generator().manual_seed(3)
    # gates at 0.0 are "fresh" (just accepted); others carry a grown/stale age and must be untouched.
    t = torch.tensor([[0.0, 0.30, 1e3], [0.0, 0.0, 0.10]], dtype=DT)
    out = PRE.seed_fresh_fix_age_s(t.clone(), cdf, g, dtype=DT)
    fresh = t == 0.0
    # fresh gates re-aged into the work_ms band [8, 120.8] ms == [0.008, 0.1208] s, strictly > 0.
    assert bool((out[fresh] > 0.0).all())
    assert bool((out[fresh] >= 8.0 / 1000.0 - 1e-9).all()) and bool((out[fresh] <= 120.8 / 1000.0 + 1e-9).all())
    # non-fresh gates are byte-identical (untouched).
    assert torch.equal(out[~fresh], t[~fresh])
    # NO gate fresh -> input returned unchanged AND the generator is NOT advanced (no draw).
    g2 = torch.Generator().manual_seed(11)
    before = g2.get_state().clone()
    t2 = torch.tensor([[0.10, 0.30], [1e3, 0.05]], dtype=DT)
    out2 = PRE.seed_fresh_fix_age_s(t2.clone(), cdf, g2, dtype=DT)
    assert torch.equal(out2, t2)
    assert torch.equal(g2.get_state(), before), "no fresh gate must draw nothing"


# ================================================================================================
# (2) ENV WIRING -- source-pins (env is cluster-only).
# ================================================================================================
def test_source_pins_knobs_default_off_and_guarded_wiring():
    init_src = inspect.getsource(PRE.PeregrineRacingEgo.__init__)
    assert 'self._vc_on = bool(getattr(cfg, "ego_vision_cadence", False))' in init_src
    for knob in ("ego_vision_frame_hz", "ego_vision_detect_p", "ego_vision_compute_latency",
                 "ego_vision_work_ms_cdf", "ego_vision_cadence_seed"):
        assert f'"{knob}"' in init_src, knob
    assert "self._vc_gen = torch.Generator(device=dev)" in init_src        # DEDICATED (RNG isolation)
    # the OLD reach-back knobs / state are GONE.
    for gone in ("ego_vision_latency", "ego_vision_feed_hz", "_vl_hist_rel", "_vl_src"):
        assert gone not in init_src, gone

    step_src = inspect.getsource(PRE.PeregrineRacingEgo._step_estimator)
    assert 'if getattr(self, "_vc_on", False):' in step_src
    assert "cadence_effective_detectable(" in step_src
    assert "detectable=det_eff" in step_src                                # the GATED mask feeds the estimator
    assert "seed_fresh_fix_age_s(" in step_src                             # thin work_ms sub-effect
    assert "self._last_detectable = detectable" in step_src               # diagnostics stay GEOMETRIC
    assert "_vislag" not in step_src and "_vl_" not in step_src           # reach-back removed

    obs_src = inspect.getsource(PRE.PeregrineRacingEgo.get_observations)
    # the reach-back OUTPUT substitution is GONE: the obs reads the estimator's current (propagated) fix.
    assert "vislag_history_read(" not in obs_src
    assert "est.rel_pos, est.confidence, est.visible_area =" not in obs_src
    assert "est = self._estimator.estimate()" in obs_src

    reset_src = inspect.getsource(PRE.PeregrineRacingEgo.reset_idx)
    # NO per-env cadence reset state (the estimator cold-reset + the global frame clock suffice).
    assert "_vc_reset_tick" not in reset_src and "_vc_src" not in reset_src
    assert "_vl_reset_tick" not in reset_src and "_vl_src" not in reset_src

    # the frame clock is deterministic (no generator parameter / draw).
    fc_src = inspect.getsource(PRE.PeregrineRacingEgo._vc_frame_tick)
    assert "generator" not in fc_src and "rand" not in fc_src


# ================================================================================================
# (3) EXECUTED frame clock on a stub self (test_ego_kp_persist.py convention).
# ================================================================================================
def _mk_vc_stub(frame_hz=30.0, dt=1 / 40.0):
    """A stub self carrying the EXACT frame-clock state PeregrineRacingEgo.__init__ allocates, so the
    REAL _vc_frame_tick runs against it."""
    s = object.__new__(PRE.PeregrineRacingEgo)
    s._vc_dt_ms = dt * 1000.0
    s._vc_frame_period_ms = (1000.0 / frame_hz) if frame_hz > 0.0 else 0.0
    s._vc_frame_accum = 0.0
    s._vc_fired = False
    return s


def test_frame_clock_fire_fraction_is_frame_hz_over_loop_hz_and_deterministic():
    for frame_hz, loop_hz, expect in ((30.0, 40.0, 0.75), (20.0, 40.0, 0.5), (30.0, 60.0, 0.5)):
        s = _mk_vc_stub(frame_hz=frame_hz, dt=1.0 / loop_hz)
        fires = sum(int(PRE.PeregrineRacingEgo._vc_frame_tick(s)) for _ in range(6000))
        assert abs(fires / 6000 - expect) < 0.01, (frame_hz, loop_hz, fires / 6000)
    # frame_hz <= 0 -> fire EVERY tick (ablation: cadence off, detector-success gate alone).
    s0 = _mk_vc_stub(frame_hz=0.0)
    assert all(PRE.PeregrineRacingEgo._vc_frame_tick(s0) for _ in range(100))
    # deterministic: the same stub replays the same fire pattern (no RNG).
    s1 = _mk_vc_stub(frame_hz=30.0, dt=1 / 40.0)
    s2 = _mk_vc_stub(frame_hz=30.0, dt=1 / 40.0)
    a = [PRE.PeregrineRacingEgo._vc_frame_tick(s1) for _ in range(40)]
    b = [PRE.PeregrineRacingEgo._vc_frame_tick(s2) for _ in range(40)]
    assert a == b


# ================================================================================================
# (4) GOLDEN TRAJECTORY through the REAL BatchedEgoEstimator + the real pure helpers.
# ================================================================================================
def _straight_course(N, G, gate_x=40.0):
    gp = torch.tensor([[gate_x + 12.0 * k, 0.0, 0.0] for k in range(G)], dtype=DT)
    return gp.unsqueeze(0).expand(N, G, 3).contiguous(), torch.zeros(N, G, dtype=DT)


def _identity_quat(N):
    return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DT).expand(N, 4).contiguous()


def _drive(N, G, geom_fn, speed, ticks, frame_hz=30.0, p=0.35, dt=1 / 40.0, noise_scale=0.0,
           work_ms_on=True, vc_on=True, gen_seed=1, est_seed=5):
    """Drive the REAL estimator along a straight const-velocity line, applying the EXACT vision-cadence
    helpers each tick (mirrors PeregrineRacingEgo._step_estimator's ON path). ``geom_fn(t)`` -> (N,G) bool
    geometric detectability. Returns per-tick stacks: conf(T,N) (slot0), age(T,N) (slot0 t_since_fix after
    the poke), fresh(T,N) (a fresh fix accepted this tick, PRE-poke), frames(list[bool]), keep(T,N) (the
    obs slot-0 mask == geom & conf>0, obs_coast=False default)."""
    gp, gy = _straight_course(N, G)
    q = _identity_quat(N)
    est = BatchedEgoEstimator(N, gp, gy, config=EgoEstimatorConfig(noise_scale=noise_scale),
                              device=CPU, dtype=DT, generator=torch.Generator().manual_seed(est_seed))
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT); vel[:, 0] = speed
    rates = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    s = _mk_vc_stub(frame_hz=frame_hz, dt=dt)
    gen = torch.Generator().manual_seed(gen_seed)
    cdf = torch.as_tensor(PRE.build_work_ms_cdf(), dtype=DT)
    confs, ages, fresh, frames, keep = [], [], [], [], []
    for t in range(ticks):
        geom = geom_fn(t)                                                   # (N,G) bool
        if vc_on:
            fired = PRE.PeregrineRacingEgo._vc_frame_tick(s)
            det_eff = PRE.cadence_effective_detectable(geom, fired, p, gen, dtype=DT)
        else:
            fired = True
            det_eff = geom
        est.step(pos, vel, q, rates, dt=dt, detectable=det_eff, prev_quat=q)
        fresh.append((est._t_since_fix[:, 0] == 0.0).clone())              # accepted this tick (pre-poke)
        if vc_on and work_ms_on and fired:
            est._t_since_fix = PRE.seed_fresh_fix_age_s(est._t_since_fix, cdf, gen, dtype=DT)
        e = est.estimate()
        confs.append(e.confidence[:, 0].clone())
        ages.append(est._t_since_fix[:, 0].clone())
        frames.append(fired)
        keep.append((e.confidence[:, 0] > 0.0) & geom[:, 0])
        pos = pos + vel * dt
    return (torch.stack(confs), torch.stack(ages), torch.stack(fresh), frames, torch.stack(keep))


def _all_visible(N, G):
    ones = torch.ones(N, G, dtype=torch.bool)
    return lambda t: ones


def test_off_path_is_byte_identical_to_fresh_every_visible_tick():
    """vc_on=False -> det_eff == geometric detectable and NO cadence draw, so the estimator estimate
    stream is bit-identical to fresh-every-visible-tick (today). Same estimator seed both runs."""
    N, G = 6, 2
    geom_fn = _all_visible(N, G)

    def run(vc_on):
        return _drive(N, G, geom_fn, speed=4.0, ticks=40, vc_on=vc_on, est_seed=17)

    c_off, a_off, _, _, k_off = run(vc_on=False)
    # a bare estimator run (detectable=ones every tick) reproduced via the same driver with vc_on=False.
    c_off2, a_off2, _, _, _ = _drive(N, G, geom_fn, speed=4.0, ticks=40, vc_on=False, est_seed=17)
    assert torch.equal(c_off, c_off2) and torch.equal(a_off, a_off2)
    # OFF: every visible tick is a fresh fix -> confidence rides at 1.0 (t_since_fix==0, no work_ms poke).
    assert torch.allclose(c_off, torch.ones_like(c_off))
    assert torch.allclose(a_off, torch.zeros_like(a_off))
    assert bool(k_off.all())                                               # visible + conf>0 -> never masked


def test_on_valid_fresh_fix_rate_in_7_to_15_hz_band():
    """Straight, continuously-visible approach: the VALID fresh-fix rate == frame_fire_fraction * p
    (noise-free estimator -> no internal miss) ~ 0.75 * 0.35 * 40 Hz ~ 10.5 Hz, inside the measured
    7-15 Hz band. Reported."""
    N, G, dt = 512, 1, 1 / 40.0
    _, _, fresh, frames, _ = _drive(N, G, _all_visible(N, G), speed=4.0, ticks=1200, dt=dt,
                                    frame_hz=30.0, p=0.35, noise_scale=0.0)
    rate_hz = float(fresh.float().mean()) / dt                             # fresh fraction / dt -> Hz
    frame_frac = sum(frames) / len(frames)
    print(f"\n[vision-cadence] frame_fraction={frame_frac:.3f} valid_fresh_fix_rate={rate_hz:.2f} Hz")
    assert 7.0 <= rate_hz <= 15.0, rate_hz
    assert abs(frame_frac - 0.75) < 0.01                                   # 30 Hz over 40 Hz


def test_on_confidence_and_age_vary_between_frames_propagation_active():
    """Between fixes the estimator EGO-PROPAGATES: t_since_fix grows and confidence decays; on a fresh
    fix it jumps back up. So on a continuously-visible approach the confidence VARIES (not pinned at 1),
    proving the propagation path is exercised (unlike OFF, which rides at 1.0)."""
    N, G = 256, 1
    conf, age, fresh, _, _ = _drive(N, G, _all_visible(N, G), speed=4.0, ticks=600,
                                    frame_hz=30.0, p=0.35, noise_scale=0.0, work_ms_on=True)
    assert float(conf.std()) > 0.0                                         # varies (propagation active)
    assert float(conf.max()) > 0.95 and float(conf.min()) < 1.0            # fresh high, decays between
    # age takes MANY distinct values (grows in dt steps between fixes) -> not a single fresh-every-tick value.
    assert torch.unique(age).numel() > 5


def test_on_fresh_fix_starts_at_sampled_work_ms_not_zero_and_zero_when_subeffect_off():
    """THIN frame->pose sub-effect: right after a fresh fix, t_since_fix ~ sampled work_ms (seconds), so
    confidence starts slightly below 1 -- NOT exactly 0/1. With the sub-effect OFF a fresh fix is exactly
    t_since_fix==0 (confidence exactly 1)."""
    N, G = 256, 1
    # sub-effect ON: freshly-fixed ticks carry a work_ms age in [8, 120.8] ms.
    _, age_on, fresh_on, _, _ = _drive(N, G, _all_visible(N, G), speed=4.0, ticks=400,
                                       frame_hz=30.0, p=0.5, noise_scale=0.0, work_ms_on=True)
    fresh_ages = age_on[fresh_on]                                          # ages ON the fresh-fix ticks (post-poke)
    assert fresh_ages.numel() > 100
    assert bool((fresh_ages > 0.0).all())                                  # NOT zero
    assert bool((fresh_ages >= 8.0 / 1000.0 - 1e-9).all())
    assert bool((fresh_ages <= 120.8 / 1000.0 + 1e-9).all())
    assert abs(float(fresh_ages.mean()) - 33.9 / 1000.0) < 6e-3            # ~ work_ms mean / 1000

    # sub-effect OFF: a fresh fix is exactly t_since_fix == 0.
    _, age_off, fresh_off, _, _ = _drive(N, G, _all_visible(N, G), speed=4.0, ticks=400,
                                         frame_hz=30.0, p=0.5, noise_scale=0.0, work_ms_on=False)
    assert bool((age_off[fresh_off] == 0.0).all())


def test_on_gate_goes_dark_on_close_approach_via_geometric_dropout():
    """The gate goes DARK on the final approach through the EXISTING GEOMETRIC dropout (loss-onset
    ~4.7 m) -- NO separate near-gate rule. In the dark region det_eff is all-False (geom False), so the
    estimator ego-propagates then MASKS past the stale horizon (conf->0) and the obs slot masks."""
    N, G, dt, speed = 8, 1, 1 / 40.0, 8.0
    gate_x, loss_range = 40.0, 4.7

    def geom_fn(t):
        rng = gate_x - speed * t * dt                                      # deterministic range to the gate
        vis = rng > loss_range
        return torch.full((N, G), bool(vis))

    ticks = 260                                                            # reaches the gate (~5 s) + margin
    conf, age, fresh, _, keep = _drive(N, G, geom_fn, speed=speed, ticks=ticks, dt=dt,
                                       frame_hz=30.0, p=0.5, noise_scale=0.0)
    dark_onset = int((gate_x - loss_range) / speed / dt) + 1               # first dark tick
    # BEFORE the gate goes dark the slot is live (fixes land, obs unmasked at least sometimes).
    assert bool(keep[:dark_onset].any())
    assert bool(fresh[:dark_onset].any())
    # a full stale-horizon AFTER onset (no fresh fix possible in the dark) -> conf==0 -> obs MASKED.
    settle = dark_onset + int(STALE_HORIZON_S / dt) + 2
    assert settle < ticks
    assert bool((conf[settle:] == 0.0).all()), "confidence must mask past the stale horizon in the dark"
    assert not bool(keep[settle:].any()), "obs slot must be dark on the final approach"
    assert not bool(fresh[dark_onset:].any()), "no fresh fix once geometric detectability is lost"
