"""Tests for the FLOOR-AT-SPAWN training floor (A1 root-cause fix, 2026-07-10).

A1: the deployed ego policy (vn16) never left the VQ2 start pad -- its trained OPENER is a gravity
dive (0.25 g + hard tilt), learned because training's lethal floor is the OOB bbox bottom at
spawn_z - 12 m (peregrine_racing._update_boxes z margin; the below-floor => collision fold in
peregrine_racing_ego.step fires only there), while the REAL warehouse floor is AT spawn height.
Fix = a training floor at the pad + gates above it + a warm opener-relearn. Pieces pinned here
(laptop-testable, the PeregrineRacingEgo class itself is cluster-only):

  (1) resolve_floor_at_spawn -- the pure +env.floor_at_spawn knob resolver (default OFF; the
      standing-start guard REFUSES standing_start_frac < 1.0 rather than being silently wrong).
  (2) _update_boxes floor -- flag ON: box_min z raised to spawn_z - 0.25 so a position 0.5 m below
      spawn z registers as collision-class (the ego fold predicate `z < box_min_z`), while RESTING
      at spawn z does not; flag OFF: box_min byte-identical to the legacy formula.
  (3) sample_courses gates_above_spawn_m -- ON: every gate centre z >= pad + clearance (sequential
      clamp); None/omitted: byte-identical legacy sampling.
  (4) resolve_course_overrides -- course_gates_above_spawn -> gates_above_spawn_m mapping.
  (5) the three floor stages render the intended hydra tokens (floor on, gate-0 band 0.5..6.0,
      vn16 warm init on the single-gate stage, REAL noise where intended, no chain-path collision).

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_floor_at_spawn.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import vq2_ego_curriculum as CUR                                           # noqa: E402
from peregrine_racing import PeregrineRacing, resolve_floor_at_spawn      # noqa: E402
from peregrine_racing_ego import resolve_course_overrides                 # noqa: E402

torch = pytest.importorskip("torch")
from peregrine_course import sample_courses                               # noqa: E402


class _Cfg:
    """Minimal attribute bag standing in for the resolved hydra cfg (getattr access only)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ================================================================================================
# (1) resolve_floor_at_spawn -- pure knob resolver + the standing-start guard.
# ================================================================================================
def test_resolver_default_off_and_defaults():
    on, clr = resolve_floor_at_spawn(_Cfg())
    assert on is False and clr == 0.25
    # OFF ignores standing_start_frac entirely (no guard when the floor is off).
    on, _ = resolve_floor_at_spawn(_Cfg(floor_at_spawn=False, standing_start_frac=0.0))
    assert on is False


def test_resolver_on_requires_standing_start():
    on, clr = resolve_floor_at_spawn(_Cfg(floor_at_spawn=True, standing_start_frac=1.0))
    assert on is True and clr == 0.25
    # custom clearance honored
    on, clr = resolve_floor_at_spawn(_Cfg(floor_at_spawn=True, standing_start_frac=1.0,
                                          floor_clearance_m=0.4))
    assert on is True and clr == 0.4
    # GUARD: floor + mixed/near-gate spawns = refused loudly (never silently wrong/inert -- L16).
    with pytest.raises(ValueError):
        resolve_floor_at_spawn(_Cfg(floor_at_spawn=True, standing_start_frac=0.5))
    with pytest.raises(ValueError):
        resolve_floor_at_spawn(_Cfg(floor_at_spawn=True))                 # frac defaults to 0.0
    with pytest.raises(ValueError):
        resolve_floor_at_spawn(_Cfg(floor_at_spawn=True, standing_start_frac=1.0,
                                    floor_clearance_m=-0.1))


# ================================================================================================
# (2) _update_boxes floor -- exercised on a skeleton env (the real method, no diffaero needed).
# ================================================================================================
def _skeleton_env(n, gate_pos, spawn_pos, floor_on, clearance=0.25):
    env = PeregrineRacing.__new__(PeregrineRacing)                        # no __init__ (cluster-only)
    env.n_envs = n
    env.device = torch.device("cpu")
    env.gate_pos = gate_pos                                               # (N, G, 3)
    env.spawn_pos = spawn_pos                                             # (N, 3)
    env._floor_at_spawn = floor_on
    env._floor_clearance_m = clearance
    return env


def _course(n, spawn_z=-0.02):
    """One gate 12 m out, 1.5 m above the pad; pad at z=spawn_z (the VQ1 pad height)."""
    gate_pos = torch.tensor([[[12.0, 0.0, spawn_z + 1.5]]]).expand(n, 1, 3).contiguous()
    spawn_pos = torch.tensor([[0.0, 0.0, spawn_z]]).expand(n, 3).contiguous()
    return gate_pos, spawn_pos


def test_floor_on_box_min_z_is_spawn_minus_clearance():
    n = 4
    gate_pos, spawn_pos = _course(n)
    env = _skeleton_env(n, gate_pos, spawn_pos, floor_on=True)
    env._update_boxes(torch.arange(n))
    spawn_z = spawn_pos[:, 2]
    assert torch.allclose(env.box_min[:, 2], spawn_z - 0.25), env.box_min[:, 2]
    # xy margins + box_max untouched by the floor
    assert torch.allclose(env.box_min[:, 0], torch.full((n,), 0.0 - 15.0))
    assert torch.allclose(env.box_max[:, 2], gate_pos[:, :, 2].amax(dim=1) + 12.0)

    # the ego fold predicate (peregrine_racing_ego.step: below_floor = z < box_min_z):
    below_floor = lambda z: z < env.box_min[:, 2]                         # noqa: E731
    assert below_floor(spawn_z - 0.5).all(), "0.5 m below spawn must be collision-class"
    assert not below_floor(spawn_z).any(), "RESTING at spawn (pad) z must be legal"
    assert not below_floor(spawn_z - 0.2).any(), "within the 0.25 m clearance is legal"


def test_floor_off_box_min_byte_identical_legacy():
    n = 4
    gate_pos, spawn_pos = _course(n)
    env = _skeleton_env(n, gate_pos, spawn_pos, floor_on=False)
    env._update_boxes(torch.arange(n))
    # legacy: min over (gates + pad) - the 12 m z margin
    pts = torch.cat([gate_pos, spawn_pos.unsqueeze(1)], dim=1)
    legacy_min = pts.amin(dim=1) - torch.tensor([15.0, 15.0, 12.0])
    legacy_max = pts.amax(dim=1) + torch.tensor([15.0, 15.0, 12.0])
    assert torch.equal(env.box_min, legacy_min), "flag OFF must be byte-identical"
    assert torch.equal(env.box_max, legacy_max)
    # and 0.5 m below spawn is NOT below the legacy floor (12 m of free fall -- the A1 bug)
    assert not (spawn_pos[:, 2] - 0.5 < env.box_min[:, 2]).any()


def test_floor_applies_per_env_indexing():
    """The floor is applied through the same env_idx fancy indexing as the box itself."""
    n = 6
    gate_pos, spawn_pos = _course(n)
    env = _skeleton_env(n, gate_pos, spawn_pos, floor_on=True)
    env._update_boxes(torch.arange(n))                                    # all envs first
    before = env.box_min.clone()
    idx = torch.tensor([1, 4])
    env.spawn_pos[idx, 2] = 3.0                                           # move two pads UP
    env.gate_pos[idx, :, 2] = 4.5
    env._update_boxes(idx)
    assert torch.allclose(env.box_min[idx, 2], torch.full((2,), 3.0 - 0.25))
    untouched = torch.tensor([0, 2, 3, 5])
    assert torch.equal(env.box_min[untouched], before[untouched])


# ================================================================================================
# (3) sample_courses gates_above_spawn_m.
# ================================================================================================
def test_sampler_gates_above_spawn_enforces_min_z():
    gen = torch.Generator().manual_seed(0)
    c = sample_courses(256, generator=gen, n_gates=4, seg_len_m=(10.0, 20.0),
                       spawn_dist_m=(8.0, 15.0), spawn_below_g0_m=(0.5, 6.0),
                       drop_m=(6.0, 12.0),                # strong descent: would sink below the pad
                       spawn_heading=0.0, gates_above_spawn_m=0.5)
    z = c["gate_pos"][..., 2]
    assert (z >= 0.5 - 1e-6).all(), f"min gate z {z.min().item()} < 0.5"


def test_sampler_off_would_sink_and_none_is_byte_identical():
    kw = dict(n_gates=4, seg_len_m=(10.0, 20.0), spawn_dist_m=(8.0, 15.0),
              spawn_below_g0_m=(0.5, 6.0), drop_m=(6.0, 12.0), spawn_heading=0.0)
    # OFF: the same strong-descent distribution sinks gates below the pad (the undivable case).
    c_off = sample_courses(256, generator=torch.Generator().manual_seed(0), **kw)
    assert (c_off["gate_pos"][..., 2] < 0.0).any(), "descent band must sink gates when OFF"
    # explicit None == omitted, byte-identical (same seed).
    c_none = sample_courses(256, generator=torch.Generator().manual_seed(0),
                            gates_above_spawn_m=None, **kw)
    assert torch.equal(c_off["gate_pos"], c_none["gate_pos"])
    assert torch.equal(c_off["gate_yaw"], c_none["gate_yaw"])


def test_sampler_clamp_is_sequential_floor_skim():
    """Forced hard descent every segment -> every post-clamp gate sits ON the floor (z == min_z),
    NOT min_z - k*drop (the naive cumulative would) -- the walk resumes FROM the floor."""
    gen = torch.Generator().manual_seed(1)
    c = sample_courses(64, generator=gen, n_gates=5, seg_len_m=(50.0, 50.0),
                       spawn_dist_m=(10.0, 10.0), spawn_below_g0_m=(2.0, 2.0),
                       drop_m=(20.0, 20.0),               # 20 m descent/leg (within max_grade 0.45*50)
                       spawn_heading=0.0, gates_above_spawn_m=0.5)
    z = c["gate_pos"][..., 2]
    assert torch.allclose(z[:, 0], torch.full((64,), 2.0))                # gate 0: 2 m above pad
    assert torch.allclose(z[:, 1:], torch.full((64, 4), 0.5)), "clamped gates sit ON the floor"


def test_sampler_rejects_unknown_key_still():
    with pytest.raises(TypeError):
        sample_courses(4, gates_above_spawn=0.5)                          # the env-side key, not sampler's


# ================================================================================================
# (4) resolve_course_overrides mapping.
# ================================================================================================
def test_resolve_maps_gates_above_spawn():
    ov = resolve_course_overrides(_Cfg(course_gates_above_spawn=0.5))
    assert ov == {"gates_above_spawn_m": 0.5}
    assert resolve_course_overrides(_Cfg()) == {}                          # unset -> absent (legacy)
    with pytest.raises(ValueError):
        resolve_course_overrides(_Cfg(course_gates_above_spawn=-0.5))


# ================================================================================================
# (5) the three floor stages render the intended tokens.
# ================================================================================================
def test_single_gate_floor_stage_tokens():
    s = CUR.STAGES["single_gate_varied_gvf_lpara_floor"]
    assert "single_gate_varied_gvf_lpara_floor" not in CUR.STAGE_ORDER    # off-ladder
    # == the champion stage EXACTLY except the floor deltas (floor flag, gate-0 band, vn16 init).
    champ = CUR.STAGES["single_gate_varied_gvf_lpara_anneal"]
    assert s["course_spawn_below_g0_lo"] == 0.5 and s["course_spawn_below_g0_hi"] == 6.0
    assert s["floor_at_spawn"] is True
    delta = {k for k in set(s) | set(champ)
             if k != "_raw" and s.get(k, None) != champ.get(k, None)}
    assert delta == {"floor_at_spawn", "course_spawn_below_g0_lo"}, delta
    raw_delta = {k for k in set(s["_raw"]) | set(champ["_raw"])
                 if s["_raw"].get(k) != champ["_raw"].get(k)}
    assert raw_delta == {"+init_from"}, raw_delta

    toks = CUR.render_overrides("single_gate_varied_gvf_lpara_floor")
    assert "+env.floor_at_spawn=true" in toks
    assert "+env.course_spawn_below_g0_lo=0.5" in toks
    assert "+env.course_spawn_below_g0_hi=6.0" in toks
    assert ("+init_from=/scratch/network/fl3689/diffaero/outputs/train/"
            "ego_single_gate_varied_gvf_lpara_anneal_seed0_vn16/checkpoints") in toks
    # REAL noise: NO ego_noise_scale override anywhere in the render.
    assert not any("ego_noise_scale" in t for t in toks), toks
    # warm champion trio
    assert "++algo.noise_std_hold=0.12" in toks and "++algo.noise_std_floor=0.03" in toks


def test_dual_gate_floor_chain_tokens():
    for name in ("dual_gate_boot_floor", "dual_gate_fullstack_floor"):
        assert name not in CUR.STAGE_ORDER                                 # off-ladder chain
        s = CUR.STAGES[name]
        assert s["course_n_gates"] == 2
        assert s["floor_at_spawn"] is True
        assert s["course_spawn_below_g0_lo"] == 0.5 and s["course_spawn_below_g0_hi"] == 6.0
        assert s["course_gates_above_spawn"] == 0.5                       # gate 1 above the pad too
        assert s["course_seg_len_lo"] == 10.0 and s["course_seg_len_hi"] == 20.0
        # NEITHER stage hardcodes +init_from: boot is FRESH (H6), fullstack is chained by the
        # sbatch ladder's auto +init_from (a hardcoded path would collide -> hydra append error).
        assert "+init_from" not in s["_raw"], name
        toks = CUR.render_overrides(name)
        assert "+env.floor_at_spawn=true" in toks
        assert "+env.course_gates_above_spawn=0.5" in toks
        assert not any(t.startswith("+init_from") for t in toks), name

    boot = CUR.STAGES["dual_gate_boot_floor"]
    full = CUR.STAGES["dual_gate_fullstack_floor"]
    # boot = noise-0 calibration + FRESH exploration trio; fullstack = REAL noise + warm trio.
    assert boot["ego_noise_scale"] == 0.0
    assert boot["_raw"]["++algo.noise_std_hold"] == 0.30
    assert "ego_noise_scale" not in full                                   # REAL noise (deploy regime)
    assert full["_raw"]["++algo.noise_std_hold"] == 0.12
    boot_toks = CUR.render_overrides("dual_gate_boot_floor")
    full_toks = CUR.render_overrides("dual_gate_fullstack_floor")
    assert "+env.ego_noise_scale=0.0" in boot_toks
    assert not any("ego_noise_scale" in t for t in full_toks)
