"""Tests for the VQ2 EGOCENTRIC course-variation wiring (BUG-2 fix, component D / DESIGN.md §D).

THE GAP (why this test exists): the ego curriculum (rl/vq2_ego_curriculum.py) emits +env.course_n_gates
(1 / 2 / 6) + course_seg_len_lo/hi=10/20, but the BASE env's _assign_courses hard-codes n_gates=6 (the
VQ1 course JSON) and calls peregrine_course.sample_courses with NO overrides -- so on the inc7/inc8 path
those keys are UNCONSUMED (inc8 additionally PINS course_mode=vq1). The ego env is the FIRST consumer:
rl/peregrine_racing_ego.resolve_course_overrides maps the curriculum keys onto the sampler's kwargs
(n_gates / seg_len_m / drop_m) and the env forwards them so single_gate is REALLY 1 gate, dual 2, multi
6, at the VQ2 10-20 m spacing.

The PeregrineRacingEgo class itself needs diffaero (cluster-only), so -- like tests/test_ego_obs_env.py
-- this pins the PURE, laptop-testable pieces end to end:
  * resolve_course_overrides: the curriculum-key -> sampler-kwarg mapping (+ the half-pair guard).
  * resolve_course_overrides -> sample_courses: the mapped overrides ACTUALLY change gate count + spacing
    (the real sampler, the same call the env makes), for every ego curriculum stage.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_ego_course_wiring.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

# resolve_course_overrides is pure (no torch); import it directly.
import peregrine_racing_ego as C                                          # noqa: E402
# the curriculum ladder (pure data) supplies the per-stage course knobs.
import vq2_ego_curriculum as CUR                                          # noqa: E402

torch = pytest.importorskip("torch")                                     # sample_courses needs torch
from peregrine_course import sample_courses                              # noqa: E402


class _Cfg:
    """A minimal attribute bag standing in for the resolved hydra cfg (getattr access only)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stage_cfg(stage: str) -> _Cfg:
    """Build a cfg bag carrying exactly the course_* keys the curriculum sets for ``stage`` (mirrors
    what hydra materializes from the +env.course_* tokens render_overrides emits)."""
    d = CUR.STAGES[stage]
    return _Cfg(**{k: d[k] for k in CUR.COURSE_SAMPLER_KEYS if k in d})


# ================================================================================================
# (1) the curriculum-key -> sampler-kwarg mapping.
# ================================================================================================
def test_resolve_maps_curriculum_keys_to_sampler_names():
    ov = C.resolve_course_overrides(_Cfg(course_n_gates=1))
    assert ov == {"n_gates": 1}
    ov = C.resolve_course_overrides(_Cfg(course_n_gates=2,
                                         course_seg_len_lo=10.0, course_seg_len_hi=20.0))
    assert ov == {"n_gates": 2, "seg_len_m": (10.0, 20.0)}
    ov = C.resolve_course_overrides(_Cfg(course_n_gates=6, course_seg_len_lo=10.0,
                                         course_seg_len_hi=20.0, course_drop_lo=-2.0,
                                         course_drop_hi=4.0))
    assert ov == {"n_gates": 6, "seg_len_m": (10.0, 20.0), "drop_m": (-2.0, 4.0)}
    # the closer-first-gate pair maps onto the sampler's spawn_dist_m (Fengyou 2026-07-07); the fixed
    # spawn-heading scalar maps onto spawn_heading.
    ov = C.resolve_course_overrides(_Cfg(course_n_gates=1, course_spawn_dist_lo=10.0,
                                         course_spawn_dist_hi=20.0, course_spawn_heading=0.0))
    assert ov == {"n_gates": 1, "spawn_dist_m": (10.0, 20.0), "spawn_heading": 0.0}
    # the gate-0 HEIGHT band maps onto spawn_below_g0_m (Fengyou 2026-07-08 varied-gate stage).
    ov = C.resolve_course_overrides(_Cfg(course_n_gates=1, course_spawn_below_g0_lo=-6.0,
                                         course_spawn_below_g0_hi=6.0))
    assert ov == {"n_gates": 1, "spawn_below_g0_m": (-6.0, 6.0)}
    # a half-specified spawn-distance pair raises (same guard as seg_len/drop).
    with pytest.raises(ValueError):
        C.resolve_course_overrides(_Cfg(course_spawn_dist_lo=10.0))
    # nothing set -> empty (the sampler keeps its own 6-gate / VQ1-range defaults).
    assert C.resolve_course_overrides(_Cfg()) == {}


def test_resolve_rejects_half_specified_pair_and_bad_values():
    # a half-specified seg_len (only lo) is a silent-wrong-band footgun -> must raise.
    with pytest.raises(ValueError):
        C.resolve_course_overrides(_Cfg(course_seg_len_lo=10.0))
    with pytest.raises(ValueError):
        C.resolve_course_overrides(_Cfg(course_seg_len_hi=20.0))
    # lo > hi is nonsense.
    with pytest.raises(ValueError):
        C.resolve_course_overrides(_Cfg(course_seg_len_lo=20.0, course_seg_len_hi=10.0))
    # n_gates < 1 is invalid.
    with pytest.raises(ValueError):
        C.resolve_course_overrides(_Cfg(course_n_gates=0))


def test_only_recognized_overrides_are_sampler_valid():
    """Every key resolve_course_overrides emits must be a name sample_courses accepts (else the env's
    sample_courses(**overrides) call would TypeError). Guards against a rename drift on either side."""
    from peregrine_course import DEFAULT_COURSE_RANGES
    ov = C.resolve_course_overrides(_Cfg(course_n_gates=2, course_seg_len_lo=10.0,
                                         course_seg_len_hi=20.0, course_drop_lo=-2.0,
                                         course_drop_hi=4.0))
    assert set(ov).issubset(set(DEFAULT_COURSE_RANGES)), (set(ov), set(DEFAULT_COURSE_RANGES))


# ================================================================================================
# (2) resolve -> sample_courses: the overrides ACTUALLY vary gate count + spacing (the real sampler).
# ================================================================================================
def _sample_stage(stage: str, n: int, seed: int = 0):
    """The exact call the ego env makes: sample_courses(n, **resolve_course_overrides(stage cfg))."""
    ov = C.resolve_course_overrides(_stage_cfg(stage))
    gen = torch.Generator().manual_seed(seed)
    return sample_courses(n, generator=gen, **ov), ov


@pytest.mark.parametrize("stage,expected_gates", [
    ("single_gate", 1),
    ("handoff_drill", 2),
    ("dual_gate_full", 2),
    ("multi_gate", 6),
])
def test_stage_course_has_the_design_gate_count(stage, expected_gates):
    """DESIGN.md §D: single_gate=1, handoff/dual=2, multi=6. The base env would give 6 for ALL of these;
    the wiring makes each stage's course actually have course_n_gates gates."""
    c, ov = _sample_stage(stage, 256)
    G = c["gate_pos"].shape[1]
    assert G == expected_gates, f"{stage}: expected {expected_gates} gates, sampler gave {G} (ov={ov})"
    assert G == CUR.STAGES[stage]["course_n_gates"], stage


@pytest.mark.parametrize("stage", ["handoff_drill", "dual_gate_full", "multi_gate"])
def test_multi_gate_stages_spacing_in_10_20_band(stage):
    """The VQ2 co-visibility regime: every gate->gate horizontal segment lands in [10, 20] m (DESIGN.md
    §D). single_gate has no gate->gate segment, so it is excluded."""
    c, _ = _sample_stage(stage, 4096)
    gp = c["gate_pos"]                                                    # (n, G, 3)
    seg = gp[:, 1:, :2] - gp[:, :-1, :2]                                  # (n, G-1, 2) horizontal
    seg_len = torch.linalg.norm(seg, dim=-1)
    lo, hi = seg_len.min().item(), seg_len.max().item()
    assert 10.0 - 1e-3 <= lo, f"{stage}: min gate-gate spacing {lo:.3f} < 10 m"
    assert hi <= 20.0 + 1e-3, f"{stage}: max gate-gate spacing {hi:.3f} > 20 m"


@pytest.mark.parametrize("stage", ["single_gate", "handoff_drill", "dual_gate_full", "multi_gate"])
def test_first_gate_distance_in_10_20_band(stage):
    """Fengyou 2026-07-07: the standing-start pad -> gate-0 horizontal distance must be 10-20 m (was the
    sampler default 18-28 m). This is the closer first gate -- easier discovery + less altitude to bleed
    before the gate. Applies to EVERY stage (course_spawn_dist_* lives in _COMMON). The pad is at the
    origin, so the distance is ||gate_pos[:, 0, :2]||."""
    c, ov = _sample_stage(stage, 4096)
    assert ov.get("spawn_dist_m") == (10.0, 20.0), (stage, ov)
    gp = c["gate_pos"]                                                    # (n, G, 3)
    spawn = c["spawn_pos"]                                                # (n, 3)
    d0 = torch.linalg.norm(gp[:, 0, :2] - spawn[:, :2], dim=-1)           # horizontal pad->gate0
    lo, hi = d0.min().item(), d0.max().item()
    assert 10.0 - 1e-3 <= lo, f"{stage}: min first-gate distance {lo:.3f} < 10 m"
    assert hi <= 20.0 + 1e-3, f"{stage}: max first-gate distance {hi:.3f} > 20 m"


@pytest.mark.parametrize("stage", ["single_gate", "handoff_drill", "dual_gate_full", "multi_gate"])
def test_fixed_spawn_heading_pins_courses_not_a_circle(stage):
    """Fengyou 2026-07-07: with course_spawn_heading pinned (0.0), EVERY course starts along +x -- gate 0
    is dead ahead (y ~ 0, x > 0), NOT fanned into a circle around the pad. This is the redundant-global-
    heading removal (the egocentric obs is heading-invariant). Contrast: the sampler's random heading
    would scatter gate-0 y across [-dist, +dist]."""
    ov = C.resolve_course_overrides(_stage_cfg(stage))
    assert ov.get("spawn_heading") == 0.0, (stage, ov)
    gen = torch.Generator().manual_seed(0)
    c = sample_courses(4096, generator=gen, **ov)
    g0 = c["gate_pos"][:, 0, :]                                           # (n, 3)
    assert g0[:, 0].min().item() > 5.0, f"{stage}: gate-0 x should be ahead (+x), got min {g0[:,0].min()}"
    assert g0[:, 1].abs().max().item() < 1e-3, (
        f"{stage}: pinned heading -> gate-0 y ~ 0, got max |y| {g0[:,1].abs().max().item()}")


def test_single_gate_differs_from_multi_gate():
    """The headline regression the wiring prevents: without it EVERY stage would be a 6-gate course. Here
    single_gate is a 1-gate course and multi_gate is a 6-gate course -- the stages genuinely differ."""
    c1, _ = _sample_stage("single_gate", 64)
    c6, _ = _sample_stage("multi_gate", 64)
    assert c1["gate_pos"].shape[1] == 1
    assert c6["gate_pos"].shape[1] == 6
    assert c1["gate_pos"].shape[1] != c6["gate_pos"].shape[1]
