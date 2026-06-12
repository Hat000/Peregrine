"""Tests for the INC7 contact-true geometry + structured force-bias DR (training doctrine
2026-06-12, docs/training_doctrine.md Sec 2/3; LAPTOP-INC7-ENV session).

Covers the three doctrine additions at the pure-helper level (the env/dynamics classes need
diffaero -- cluster-only; their wiring is gated by rl/check_diffaero_gate.py + the precheck):
  * ``slab_frame_hits`` -- exact segment-vs-volumetric-frame classification (brute-force
    arbiter + the numpy twin-side mirror + targeted live-failure-class cases);
  * body-radius band inflation through ``crossing_events`` (per-env tensor bands);
  * ``sample_force_bias`` / ``force_bias_active`` -- regime-binned world-bias DR;
  * NEGATIVE CONTROLS: defaults-off paths are the legacy classification bit-for-bit.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_RL = Path(__file__).resolve().parents[1] / "rl"
sys.path.insert(0, str(_RL))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peregrine_racing import crossing_events, slab_frame_hits          # noqa: E402


def _stub_base_dynamics():
    """Install the value-faithful BaseDynamics stub BEFORE the first diffaero_dynamics import
    (the test_measured_aero._import_adapter pattern): the module caches its guarded import, so
    whichever test file imports first must leave a functional base class for the others."""
    import types
    if "diffaero.dynamics.base_dynamics" in sys.modules:
        return
    base_mod = types.ModuleType("diffaero.dynamics.base_dynamics")

    class BaseDynamics:
        def __init__(self, cfg, device):
            self.n_agents = int(getattr(cfg, "n_agents", 1))
            self.n_envs = int(getattr(cfg, "n_envs", 1))
            self.dt = float(cfg.dt)
            self.alpha = float(getattr(cfg, "alpha", 1.0))
            self.device = device

        def grad_decay(self, x):              # value-preserving (real one scales grads)
            return x

        def detach(self):
            self._state = self._state.detach()

    base_mod.BaseDynamics = BaseDynamics
    pkg = types.ModuleType("diffaero")
    dyn_pkg = types.ModuleType("diffaero.dynamics")
    pkg.dynamics = dyn_pkg
    dyn_pkg.base_dynamics = base_mod
    sys.modules.setdefault("diffaero", pkg)
    sys.modules.setdefault("diffaero.dynamics", dyn_pkg)
    sys.modules["diffaero.dynamics.base_dynamics"] = base_mod


_stub_base_dynamics()
from diffaero_dynamics import (FORCE_BIAS_SPEED_BINS, FORCE_BIAS_TILT_BINS_DEG,  # noqa: E402
                               force_bias_active, sample_force_bias)

HALF_IN, HALF_OUT, DEPTH = 0.75, 1.36, 0.30
R = 0.33                                    # a mid-band body radius
HALF_IN_R, HALF_OUT_R = HALF_IN - R, HALF_OUT + R


def _seg(prev, curr):
    return (torch.tensor([prev], dtype=torch.float64),
            torch.tensor([curr], dtype=torch.float64))


def _slab(prev, curr, hi=HALF_IN, ho=HALF_OUT, depth=DEPTH):
    return bool(slab_frame_hits(*_seg(prev, curr), hi, ho, depth).item())


# ---------------------------------------------------------------- slab: targeted cases
def test_slab_preplane_strike_invisible_to_crossing_test():
    """The live standing failure class: frame contact BEFORE the plane, no plane crossing."""
    prev, curr = [-0.5, 0.9, 0.0], [-0.1, 0.9, 0.0]
    assert _slab(prev, curr)
    ev = crossing_events(*_seg(prev, curr), HALF_IN, HALF_OUT)
    assert not (ev["fwd"].item() or ev["bwd"].item() or ev["in_frame"].item())


def test_slab_steep_approach_clips_before_clean_plane_crossing():
    """Crossing point threads the pass band, but the steep approach touches the band inside
    the slab -- plane test says pass, volumetric truth says strike."""
    prev, curr = [-0.35, 1.0, 0.0], [0.05, 0.0, 0.0]
    ev = crossing_events(*_seg(prev, curr), HALF_IN, HALF_OUT)
    assert ev["pass_ok"].item()                       # the fiction
    assert _slab(prev, curr)                          # the truth


def test_slab_shallow_center_pass_clean():
    assert not _slab([-0.4, 0.1, 0.0], [0.4, 0.12, 0.05])


def test_slab_entirely_outside_band_clean():
    assert not _slab([-0.4, 2.0, 0.0], [0.4, 2.1, 0.0])         # beyond outer everywhere
    assert not _slab([0.5, 1.0, 0.0], [1.5, 1.0, 0.0])          # never enters the slab


def test_slab_through_band_inside_slab():
    """From far outside the outer edge to the center, entirely within |x| <= depth: must
    sweep through the material band even though both endpoints are off-band."""
    assert _slab([-0.2, 2.0, 0.0], [0.1, 0.0, 0.0])


def test_slab_plane_parallel_segment():
    assert _slab([0.0, 0.2, 0.0], [0.0, 1.0, 0.0])              # dx=0 inside, sweeps band
    assert not _slab([0.0, 0.2, 0.0], [0.0, 0.5, 0.0])          # dx=0 inside, stays in pass
    assert not _slab([0.6, 0.2, 0.0], [0.6, 1.0, 0.0])          # dx=0 OUTSIDE the slab


def test_slab_z_axis_band():
    assert _slab([-0.25, 0.0, 1.0], [0.05, 0.0, 1.1])           # band entered via |z|


def test_slab_depth_zero_degenerates_to_plane_band():
    """At depth=0 the slab is the plane itself: agreement with the crossing in_frame class
    on crossing segments (callers skip the slab then, but the math must stay coherent)."""
    rng = np.random.default_rng(3)
    prev = torch.tensor(rng.normal(scale=1.2, size=(256, 3)))
    curr = prev + torch.tensor(rng.normal(scale=1.5, size=(256, 3)))
    ev = crossing_events(prev, curr, HALF_IN, HALF_OUT)
    hit = slab_frame_hits(prev, curr, HALF_IN, HALF_OUT, 0.0)
    crossed = ev["fwd"] | ev["bwd"]
    assert torch.equal(hit[crossed], ev["in_frame"][crossed])


# ---------------------------------------------------------------- slab: brute-force arbiter
def _brute(prev, curr, hi, ho, depth, n=20001):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p = prev[None, :] * (1.0 - t) + curr[None, :] * t
    m = np.abs(p[:, 0]) <= depth
    if not m.any():
        return False
    linf = np.max(np.abs(p[m, 1:3]), axis=1)
    return bool(np.any((linf >= hi) & (linf <= ho)))


def test_slab_matches_brute_force_on_random_segments():
    rng = np.random.default_rng(11)
    prev = rng.normal(scale=1.2, size=(1000, 3))
    curr = prev + rng.normal(scale=1.5, size=(1000, 3))
    got = slab_frame_hits(torch.tensor(prev), torch.tensor(curr),
                          HALF_IN_R, HALF_OUT_R, DEPTH).numpy()
    want = np.array([_brute(prev[i], curr[i], HALF_IN_R, HALF_OUT_R, DEPTH)
                     for i in range(len(prev))])
    assert (got == want).all(), f"{int((got != want).sum())} mismatches vs brute force"


def test_slab_numpy_mirror_agrees_with_torch():
    from offline_rollout import slab_frame_hit_np
    rng = np.random.default_rng(12)
    prev = rng.normal(scale=1.2, size=(500, 3))
    curr = prev + rng.normal(scale=1.5, size=(500, 3))
    got_t = slab_frame_hits(torch.tensor(prev), torch.tensor(curr),
                            HALF_IN_R, HALF_OUT_R, DEPTH).numpy()
    got_n = np.array([slab_frame_hit_np(prev[i], curr[i], HALF_IN_R, HALF_OUT_R, DEPTH)
                      for i in range(len(prev))])
    assert (got_t == got_n).all()


def test_slab_per_env_tensor_bands_broadcast():
    """(N, G) batch with per-env (N, 1) inflated bands == per-element scalar calls."""
    rng = np.random.default_rng(13)
    N, G = 8, 6
    prev = torch.tensor(rng.normal(scale=1.2, size=(N, G, 3)))
    curr = prev + torch.tensor(rng.normal(scale=1.5, size=(N, G, 3)))
    r = torch.tensor(rng.uniform(0.28, 0.38, size=(N, 1)))
    got = slab_frame_hits(prev, curr, HALF_IN - r, HALF_OUT + r, DEPTH)
    assert got.shape == (N, G)
    for i in range(N):
        for j in range(G):
            want = _slab(prev[i, j].tolist(), curr[i, j].tolist(),
                         HALF_IN - float(r[i, 0]), HALF_OUT + float(r[i, 0]))
            assert bool(got[i, j].item()) == want


# ---------------------------------------------------------------- body-radius inflation
def test_inflated_bands_reclassify_the_live_crash_offsets():
    """Live standing crashes terminated at Linf 0.37-0.49: passes under the 0.75 point model,
    frame strikes under the inflated bands at every doctrine radius."""
    for linf in (0.37, 0.44, 0.49):
        prev, curr = [-0.3, linf, 0.0], [0.3, linf, 0.0]
        ev0 = crossing_events(*_seg(prev, curr), HALF_IN, HALF_OUT)
        assert ev0["pass_ok"].item()                            # the fiction
        for r in (0.28, 0.33, 0.38):
            ev = crossing_events(*_seg(prev, curr), HALF_IN - r, HALF_OUT + r)
            assert ev["in_frame"].item() == (linf >= HALF_IN - r)
            assert not ev["pass_ok"].item() or linf < HALF_IN - r


def test_inflated_outer_band_converts_near_miss_to_strike():
    prev, curr = [-0.3, 1.5, 0.0], [0.3, 1.5, 0.0]              # 1.36 < 1.5 <= 1.36 + 0.33
    ev0 = crossing_events(*_seg(prev, curr), HALF_IN, HALF_OUT)
    assert not ev0["in_frame"].item() and not ev0["pass_ok"].item()      # legacy: clean miss
    ev = crossing_events(*_seg(prev, curr), HALF_IN_R, HALF_OUT_R)
    assert ev["in_frame"].item()                                          # inflated: strike


def test_crossing_events_per_env_tensor_bands_broadcast():
    rng = np.random.default_rng(14)
    N, G = 8, 6
    prev = torch.tensor(rng.normal(scale=1.2, size=(N, G, 3)))
    curr = prev + torch.tensor(rng.normal(scale=1.5, size=(N, G, 3)))
    r = torch.tensor(rng.uniform(0.28, 0.38, size=(N, 1)))
    ev = crossing_events(prev, curr, HALF_IN - r, HALF_OUT + r)
    for i in range(N):
        for j in range(G):
            want = crossing_events(prev[i:i + 1, j], curr[i:i + 1, j],
                                   HALF_IN - float(r[i, 0]), HALF_OUT + float(r[i, 0]))
            for k in ("fwd", "bwd", "pass_ok", "in_frame"):
                assert bool(ev[k][i, j].item()) == bool(want[k][0].item()), (k, i, j)


# ---------------------------------------------------------------- negative controls (OFF = legacy)
def _legacy_crossing_events(prev_rel, curr_rel, half_inner, half_outer):
    """Verbatim copy of the pre-INC7 crossing_events (float bands) -- the bit-identity pin."""
    px, cx = prev_rel[..., 0], curr_rel[..., 0]
    fwd = (px < 0) & (cx >= 0)
    bwd = (px > 0) & (cx <= 0)
    crossed = fwd | bwd
    denom = (cx - px)
    f = torch.where(crossed, -px / torch.where(denom.abs() < 1e-9,
                                               torch.full_like(denom, 1e-9), denom),
                    torch.zeros_like(denom))
    y = prev_rel[..., 1] + f * (curr_rel[..., 1] - prev_rel[..., 1])
    z = prev_rel[..., 2] + f * (curr_rel[..., 2] - prev_rel[..., 2])
    linf = torch.maximum(y.abs(), z.abs())
    pass_ok = crossed & (linf < half_inner)
    in_frame = crossed & (linf >= half_inner) & (linf <= half_outer)
    return {"fwd": fwd, "bwd": bwd, "pass_ok": pass_ok, "in_frame": in_frame, "linf": linf}


def test_negative_control_crossing_events_unchanged_with_float_bands():
    rng = np.random.default_rng(15)
    prev = torch.tensor(rng.normal(scale=1.2, size=(64, 6, 3)), dtype=torch.float32)
    curr = prev + torch.tensor(rng.normal(scale=1.5, size=(64, 6, 3)), dtype=torch.float32)
    got = crossing_events(prev, curr, HALF_IN, HALF_OUT)
    want = _legacy_crossing_events(prev, curr, HALF_IN, HALF_OUT)
    for k in ("fwd", "bwd", "pass_ok", "in_frame"):
        assert torch.equal(got[k], want[k]), k
    assert torch.equal(got["linf"], want["linf"])


def test_negative_control_gate_event_numpy_defaults_are_legacy():
    """offline_rollout.gate_event with default args must reproduce the legacy classes on the
    documented S1.4 cases (the twin-side OFF path)."""
    from offline_rollout import gate_event, _GATE_POS_ZUP, _FLIP, _R_W2G
    g0_ned = (_GATE_POS_ZUP[0] * _FLIP)
    R_g2w = _R_W2G.T

    def ned(rel):
        return g0_ned + (R_g2w @ np.asarray(rel, float)) * _FLIP

    assert gate_event(ned([-0.2, 0.0, 0.0]), ned([0.3, 0.0, 0.0]), 0) == "pass"
    assert gate_event(ned([-0.25, 1.6, 0.0]), ned([0.75, 0.4, 0.0]), 0) == "collision"
    assert gate_event(ned([-0.5, 2.0, 0.0]), ned([0.5, 2.0, 0.0]), 0) == "miss"
    assert gate_event(ned([0.5, 0.0, 0.0]), ned([1.5, 0.0, 0.0]), 0) is None
    # the pre-plane strike: invisible legacy, collision under contact-true scoring
    assert gate_event(ned([-0.5, 0.9, 0.0]), ned([-0.1, 0.9, 0.0]), 0) is None
    assert gate_event(ned([-0.5, 0.9, 0.0]), ned([-0.1, 0.9, 0.0]), 0,
                      body_radius=0.33, frame_depth=0.30) == "collision"
    # the live crash offset: legacy pass, contact-true strike
    assert gate_event(ned([-0.3, 0.45, 0.0]), ned([0.3, 0.45, 0.0]), 0) == "pass"
    assert gate_event(ned([-0.3, 0.45, 0.0]), ned([0.3, 0.45, 0.0]), 0,
                      body_radius=0.33, frame_depth=0.30) == "collision"


# ---------------------------------------------------------------- force-bias DR helpers
def test_sample_force_bias_respects_band_and_bins():
    torch.manual_seed(0)
    bias, s_lo, s_hi, c_lo, c_hi = sample_force_bias(4096, 3.0, "cpu", torch.float64)
    assert bias.shape == (4096, 3)
    norms = torch.linalg.norm(bias, dim=-1)
    assert float(norms.max()) <= 3.0 + 1e-9
    assert float(norms.min()) >= 0.0
    # bins come from the declared (certified frame_residual_report) sets
    sbins = {(float(a), float(b)) for a, b in zip(s_lo.tolist(), s_hi.tolist())}
    assert sbins <= {(a, b) for a, b in FORCE_BIAS_SPEED_BINS}
    assert len(sbins) == len(FORCE_BIAS_SPEED_BINS)            # all bins get drawn
    cbins = {(round(float(a), 9), round(float(b), 9))
             for a, b in zip(c_lo.tolist(), c_hi.tolist())}
    want = {(round(math.cos(math.radians(t_hi)), 9), round(math.cos(math.radians(t_lo)), 9))
            for t_lo, t_hi in FORCE_BIAS_TILT_BINS_DEG}
    assert cbins == want
    # direction is not degenerate: mean should be near zero on every axis
    assert float(bias.mean(dim=0).abs().max()) < 0.15


def test_force_bias_active_edges():
    s_lo, s_hi = torch.tensor(12.0), torch.tensor(18.0)
    c_lo, c_hi = (torch.tensor(math.cos(math.radians(90.0))),
                  torch.tensor(math.cos(math.radians(35.0))))

    def act(speed, tilt_deg):
        return bool(force_bias_active(torch.tensor(speed),
                                      torch.tensor(math.cos(math.radians(tilt_deg))),
                                      s_lo, s_hi, c_lo, c_hi).item())

    assert act(12.0, 50.0)            # speed lower edge INCLUSIVE
    assert not act(18.0, 50.0)        # speed upper edge EXCLUSIVE
    assert not act(11.99, 50.0)
    assert act(15.0, 35.0)            # tilt lower edge INCLUSIVE
    assert not act(15.0, 90.0)        # tilt upper edge EXCLUSIVE
    assert not act(15.0, 20.0)
    # the climb-bin instance that displaced the live standing approach is representable
    assert act(15.0, 55.0)


def test_force_bias_zero_bin_is_inert():
    """The init state (zeros everywhere): no speed satisfies [0, 0), so the hook is inactive
    -- the parity-gate negative control at the helper level."""
    z = torch.zeros(8)
    speeds = torch.tensor([0.0, 1.0, 10.0, 17.0, 30.0, 0.5, 2.0, 25.0])
    cos_t = torch.linspace(-1.0, 1.0, 8)
    assert not force_bias_active(speeds, cos_t, z, z, z, z).any()
