"""Tests for the COURSE VERTICAL-STRUCTURE fix (2026-07-25).

THE MEASURED DEFECT this code exists to close: over 51 deploy flights obs[10] (the coarse VERTICAL
sector, {-1,0,+1}) is fed +1 on 37.6% of ticks, but the training course sampler realises +1 on only
~1.6% of sampled gates -- the policy has effectively never been trained to climb, and the course's two
consecutive UP gates pass at 47%/22% vs 74-83% for level/down gates.

ROOT CAUSE (pinned below, because the obvious suspect is the WRONG one): it is NOT the
``gates_above_spawn_m`` floor -- that clamp only ever SHRINKS a descent, so it converts -1 legs into 0
legs and can never create a climb (test_floor_cannot_create_a_climb). It is the sampler's
DESCENT-BIASED default band ``drop_m=(-3, 12)``: dz = -drop lands in [-12, +3], while a 10-20 m leg
needs dz > tan(0.20)*L ~ 2.0-4.1 m to clear the vertical sector deadband.

WHAT IS PINNED HERE:
  (1) the NEW ``gates_ceiling_m`` sampler knob -- BYTE-IDENTICAL when unset, really clamps when set,
      composes with the floor, fails loud when inverted;
  (2) the ``course_gates_ceiling`` curriculum -> sampler bridge in resolve_course_overrides;
  (3) ``coarse_sector_stats`` -- the diagnostic that reports the realised bucket distribution;
  (4) the END-TO-END measured claim: the LIVE stage config realises +1 on ~1.6% of gates, and the
      shipped v2.0 setting (drop -10/+10 + ceiling 16) lands it in the 25-40% target band. This is the
      regression pin -- if a future sampler change silently flattens the courses again, this fails.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_course_vertical_structure.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

torch = pytest.importorskip("torch")                                      # sample_courses needs torch

import peregrine_racing_ego as EGO                                        # noqa: E402
import vq2_ego_curriculum as CUR                                          # noqa: E402
from peregrine_course import DEFAULT_COURSE_RANGES, sample_courses        # noqa: E402


class _Cfg:
    """A minimal attribute bag standing in for the resolved hydra cfg (getattr access only)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# The LIVE v1.9 training course configuration: the dual_gate_fullstack_floor_pef16 stage's course keys
# with the launcher's course_n_gates=8 (VPEF8NC NGATES) applied -- i.e. exactly what v1.9 trained on.
_LIVE_STAGE = "dual_gate_fullstack_floor_pef16"


def _live_overrides(**extra):
    d = {k: v for k, v in CUR.STAGES[_LIVE_STAGE].items() if k != "_raw"}
    d["course_n_gates"] = 8                                               # the launcher's NGATES
    d.update(extra)
    return EGO.resolve_course_overrides(_Cfg(**d))


def _vert_up_frac(n=8000, seed=20260725, **extra):
    """Realised +1 (UP) fraction of the vertical coarse-sector bucket over ``n`` sampled courses,
    through the EXACT training path (resolve_course_overrides -> sample_courses -> build_coarse_map)."""
    ov = _live_overrides(**extra)
    g = torch.Generator().manual_seed(seed)
    c = sample_courses(n, device="cpu", generator=g, **ov)
    sec = EGO.build_coarse_map(c["gate_pos"], c["spawn_pos"])
    return EGO.coarse_sector_stats(sec), c["gate_pos"][..., 2]


# ================================================================================================
# (1) gates_ceiling_m -- the NEW sampler knob.
# ================================================================================================
def test_ceiling_default_is_off():
    assert "gates_ceiling_m" in DEFAULT_COURSE_RANGES
    assert DEFAULT_COURSE_RANGES["gates_ceiling_m"] is None                # OFF == byte-identical legacy


def test_ceiling_off_is_byte_identical():
    """Passing gates_ceiling_m=None explicitly must reproduce the legacy draw BIT-FOR-BIT (same RNG
    stream, same clamp path). This is the byte-identity guarantee the whole knob rests on."""
    for floor in (None, 0.5):
        a = sample_courses(64, generator=torch.Generator().manual_seed(7), n_gates=8,
                           gates_above_spawn_m=floor)
        b = sample_courses(64, generator=torch.Generator().manual_seed(7), n_gates=8,
                           gates_above_spawn_m=floor, gates_ceiling_m=None)
        for k in a:
            assert torch.equal(a[k], b[k]), (floor, k)


def test_ceiling_actually_clamps():
    c = sample_courses(512, generator=torch.Generator().manual_seed(11), n_gates=8,
                       drop_m=(-10.0, 10.0), gates_ceiling_m=6.0)
    z = c["gate_pos"][..., 2]
    assert float(z.max()) <= 6.0 + 1e-5
    assert float(z.max()) > 5.9                                           # and it BINDS (not vacuous)
    # without it the same band climbs well past the ceiling -> the knob is load-bearing, not cosmetic
    c2 = sample_courses(512, generator=torch.Generator().manual_seed(11), n_gates=8,
                        drop_m=(-10.0, 10.0))
    assert float(c2["gate_pos"][..., 2].max()) > 6.0


def test_ceiling_composes_with_floor():
    c = sample_courses(512, generator=torch.Generator().manual_seed(13), n_gates=8,
                       drop_m=(-10.0, 10.0), gates_above_spawn_m=0.5, gates_ceiling_m=8.0)
    z = c["gate_pos"][..., 2]
    assert float(z.min()) >= 0.5 - 1e-5 and float(z.max()) <= 8.0 + 1e-5


def test_ceiling_below_floor_raises():
    with pytest.raises(ValueError):
        sample_courses(8, n_gates=4, gates_above_spawn_m=5.0, gates_ceiling_m=1.0)


def test_ceiling_alone_without_floor_still_walks_down():
    """Ceiling-only must NOT accidentally install a floor: gates may still sink below the pad."""
    c = sample_courses(512, generator=torch.Generator().manual_seed(17), n_gates=8,
                       gates_ceiling_m=4.0)
    assert float(c["gate_pos"][..., 2].min()) < 0.0


# ================================================================================================
# (2) course_gates_ceiling -- the curriculum -> sampler bridge.
# ================================================================================================
def test_resolve_maps_gates_ceiling():
    assert EGO.resolve_course_overrides(_Cfg(course_gates_ceiling=16.0)) == {"gates_ceiling_m": 16.0}
    assert "gates_ceiling_m" not in EGO.resolve_course_overrides(_Cfg())   # unset -> absent (legacy)
    both = EGO.resolve_course_overrides(_Cfg(course_gates_above_spawn=0.5, course_gates_ceiling=16.0))
    assert both == {"gates_above_spawn_m": 0.5, "gates_ceiling_m": 16.0}


def test_resolve_gates_ceiling_guards():
    with pytest.raises(ValueError):
        EGO.resolve_course_overrides(_Cfg(course_gates_ceiling=0.0))
    with pytest.raises(ValueError):
        EGO.resolve_course_overrides(_Cfg(course_gates_ceiling=-3.0))
    with pytest.raises(ValueError):                                       # ceiling BELOW the floor
        EGO.resolve_course_overrides(_Cfg(course_gates_above_spawn=8.0, course_gates_ceiling=2.0))


def test_resolved_overrides_are_accepted_by_the_sampler():
    """The bridge output must be directly consumable by sample_courses (no unknown-key TypeError) --
    the exact failure class the wiring exists to prevent."""
    ov = _live_overrides(course_drop_lo=-10.0, course_drop_hi=10.0, course_gates_ceiling=16.0)
    assert ov["drop_m"] == (-10.0, 10.0) and ov["gates_ceiling_m"] == 16.0
    c = sample_courses(32, generator=torch.Generator().manual_seed(3), **ov)
    assert c["gate_pos"].shape == (32, 8, 3)


# ================================================================================================
# (3) coarse_sector_stats -- the diagnostic.
# ================================================================================================
def test_coarse_sector_stats_counts():
    # 1 course, 4 gates: vert buckets [+1, +1, 0, -1]; horiz [-1, 0, 0, 0]
    sec = torch.tensor([[[-1, 1], [0, 1], [0, 0], [0, -1]]])
    st = EGO.coarse_sector_stats(sec)
    assert st["vert_up"] == pytest.approx(0.5)
    assert st["vert_level"] == pytest.approx(0.25)
    assert st["vert_down"] == pytest.approx(0.25)
    assert st["horiz_left"] == pytest.approx(0.25)
    assert st["horiz_level"] == pytest.approx(0.75)
    assert st["vert_up"] + st["vert_level"] + st["vert_down"] == pytest.approx(1.0)
    assert st["horiz_right"] + st["horiz_level"] + st["horiz_left"] == pytest.approx(1.0)


def test_coarse_sector_stats_prefix_and_purity():
    sec = torch.tensor([[[0, 1], [0, 0]]])
    before = sec.clone()
    st = EGO.coarse_sector_stats(sec, prefix="course_")
    assert set(st) == {"course_vert_up", "course_vert_level", "course_vert_down",
                       "course_horiz_right", "course_horiz_level", "course_horiz_left"}
    assert torch.equal(sec, before)                                       # PURE: input unmutated


# ================================================================================================
# (4) END-TO-END: the measured distribution claim (the regression pin).
# ================================================================================================
def test_live_default_starves_the_up_bucket():
    """The DEFECT, pinned: the v1.9 course distribution realises +1 on ~1.6% of gates (vs 37.6% of
    deploy ticks). If a future change fixes this by accident, this test tells us."""
    st, _ = _vert_up_frac()
    assert st["vert_up"] < 0.05, st                                       # measured 0.0157
    assert st["vert_level"] > 0.85, st                                    # measured 0.911 -- courses are FLAT


def test_floor_cannot_create_a_climb():
    """THE PREMISE CORRECTION: course_gates_above_spawn is NOT why +1 is rare. The floor clamp only
    ever SHRINKS a descent, so turning it off changes the DOWN bucket dramatically and leaves the UP
    bucket EXACTLY unchanged."""
    on, _ = _vert_up_frac()
    off, _ = _vert_up_frac(course_gates_above_spawn=None)
    assert off["vert_up"] == pytest.approx(on["vert_up"], abs=1e-12)       # UP: bit-identical
    assert off["vert_down"] > 5 * on["vert_down"]                          # DOWN: 7.3% -> 59.5%


def test_v20_setting_lands_in_the_target_band():
    """The SHIPPED v2.0 setting (rl/launch_v20.sh): drop(-10,+10) + ceiling 16 must put the realised
    +1 fraction in the 25-40% band that matches the deploy stream, KEEP a healthy DOWN bucket (this
    trains vertical CONTROL, not a climb bias), and stay inside the warehouse-scale envelope."""
    st, z = _vert_up_frac(course_drop_lo=-10.0, course_drop_hi=10.0, course_gates_ceiling=16.0)
    assert 0.25 <= st["vert_up"] <= 0.40, st                              # measured 0.301
    assert st["vert_down"] > 0.15, st                                     # measured 0.205
    assert float(z.max()) <= 16.0 + 1e-5                                  # bounded by the ceiling
    assert float(z.min()) >= 0.5 - 1e-5                                   # and by the floor


def test_v20_setting_removes_the_all_flat_courses():
    """45.9% of DEFAULT courses have ZERO vertical structure (every gate's vert bucket == 0). The v2.0
    setting must all but eliminate them -- that population is what "never trained to climb" means."""
    def all_flat(**extra):
        ov = _live_overrides(**extra)
        c = sample_courses(4000, device="cpu", generator=torch.Generator().manual_seed(5), **ov)
        v = EGO.build_coarse_map(c["gate_pos"], c["spawn_pos"])[..., 1]
        return float((v == 0).all(dim=1).float().mean())

    assert all_flat() > 0.35                                              # measured 0.459
    assert all_flat(course_drop_lo=-10.0, course_drop_hi=10.0,
                    course_gates_ceiling=16.0) < 0.10                     # measured 0.022


# ================================================================================================
# (5) the deadband is NOT the lever (label semantics are shared with the hand-authored deploy map).
# ================================================================================================
def test_build_coarse_map_vert_deadband_unchanged():
    """build_coarse_map's vertical deadband must stay 0.20 rad by default: the DEPLOY side reads a
    hand-authored coarse-map JSON written to that convention, so moving the training deadband would
    desync the LABEL semantics from the wire. Move the geometry, never the label."""
    import inspect
    sig = inspect.signature(EGO.build_coarse_map)
    assert sig.parameters["vert_thresh_rad"].default == 0.20
    assert sig.parameters["horiz_thresh_rad"].default == 0.20
    # and the bucket really is sign-past-deadband on the OUTGOING leg elevation
    spawn = torch.zeros(1, 3)
    #      g0 at 10 m out level, g1 +3 m up  -> atan2(3,10)=0.291 > 0.20 -> g0 vert = +1
    gp = torch.tensor([[[10.0, 0.0, 0.0], [20.0, 0.0, 3.0]]])
    assert int(EGO.build_coarse_map(gp, spawn)[0, 0, 1]) == 1
    #      g1 only +1.5 m up over 10 m -> atan2(1.5,10)=0.149 < 0.20 -> g0 vert = 0 (inside the band)
    gp = torch.tensor([[[10.0, 0.0, 0.0], [20.0, 0.0, 1.5]]])
    assert int(EGO.build_coarse_map(gp, spawn)[0, 0, 1]) == 0
