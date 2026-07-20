"""Unit tests for the v1.7 config-gated mechanisms (ALL default-OFF == byte-identical to v1.6):

  M1 HANDOFF SPAWN REALISM (handoff_spawn_geometry + PeregrineRacingEgo._apply_handoff_spawn): the wire's
     from-rest acquisition geometry as a SIBLING spawn mode. Covers the sampled slant-range band, the
     realised-vs-requested vertical offset under the AGL floor, staying ON the incoming approach leg, the
     selected FRACTION, and -- the load-bearing one -- that a LEVEL handoff pose ACQUIRES gate 0 at t=0
     across the whole sampled band (driven through the REAL gate_detectable projection). The env wiring is
     EXECUTED against a stub self (the test_perception_honesty / test_ego_kp_persist convention), including
     the OFF path's RNG-neutrality (frac 0 draws NOTHING -> the reset stream is bit-identical).
  M2 BLIND-FLIGHT ABORT (blind_abort_update): had-then-lost clocking, and the SCOPING that is the whole
     design -- a start DIVE fires it; the legitimate post-pass acquisition gap CANNOT (the caller clears
     ``acquired`` on advance). Plus the range / closing / OFF gates and the optional advance grace.
  M3 PROGRESS FRAMING MULTIPLIER (progress_frame_factor + compute_ego_reward): f in [floor, 1], the
     detectability hard-mask, POSITIVE-credit-only, the CAN-ONLY-SHRINK invariant that closes the reward
     pump, and FARM RESISTANCE (stationary staring earns ~nothing -- it multiplies zero progress).
  M4 APERTURE TIGHTENING (pass_margin_final_m 1.0 -> 0.75): the anneal END-HOLD endpoint, start > end (the
     train-loop guard), and the resulting weighted-miss envelope (a pure LATERAL tightening 0.5 -> 0.375 m;
     the vertical bound stays inactive at the 0.75 m aperture).
  OFF  DEFAULT-OFF EQUIVALENCE: the v1.7 reward path with every new key unset is BITWISE identical to the
     v1.6 call, and every new weight/knob defaults to its OFF value.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_v17_mechanisms.py -q
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import peregrine_racing_ego as EGO                                       # noqa: E402
import ego_reward as ER                                                  # noqa: E402
import gate_visibility as GV                                             # noqa: E402
from peregrine_racing import quat_xyzw_from_yaw_pitch                    # noqa: E402
from inc8_estimator_emul import _RZ_PI_BODY_NP, quat_xyzw_to_matrix_torch  # noqa: E402

DT = torch.float64
DT_S = 1.0 / 30.0


def _t(x):
    return torch.tensor(x, dtype=DT)


def _lift_pure_fn(name, path):
    """Exec ONE top-level pure function out of a module that is NOT laptop-importable
    (rl/peregrine_train_ego.py has unguarded top-level ``import diffaero.*`` -- it is the Adroit-only
    launcher). ``ast`` lifts the REAL SHIPPED source, so this tests the actual scheduler rather than a
    reimplementation, with zero import side effects. Deliberately NOT the diffaero-stub route: that stub
    is a module-level singleton and tests/test_unwrap_env.py documents how an incomplete one can poison
    other test modules for the whole pytest session."""
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in {path}")


_pass_margin_schedule = _lift_pure_fn("_pass_margin_schedule", ROOT / "rl" / "peregrine_train_ego.py")


# ================================================================================================
# M1 -- HANDOFF SPAWN REALISM.
# ================================================================================================
_HANDOFF_BANDS = dict(range_lo=9.5, range_hi=11.5, gate_dz_lo=-1.5, gate_dz_hi=1.5, min_agl_m=0.5)


def _handoff(n=64, frac=1.0, gate=(12.0, 0.0, 3.0), pad=(0.0, 0.0, 0.0), seed=0, **over):
    """Drive handoff_spawn_geometry over n envs with reproducible draws. gate/pad are broadcast."""
    g = torch.tensor([gate], dtype=DT).expand(n, 3).contiguous()
    a = torch.tensor([pad], dtype=DT).expand(n, 3).contiguous()
    gen = torch.Generator().manual_seed(seed)
    draws = [torch.rand(n, generator=gen, dtype=DT) for _ in range(3)]
    kw = dict(_HANDOFF_BANDS)
    kw.update(over)
    pos, bearing, sel = EGO.handoff_spawn_geometry(g, a, a[:, 2], *draws, frac=frac, **kw)
    return g, a, pos, bearing, sel


def test_handoff_frac_zero_selects_nothing():
    """frac 0 -> the selection mask is all-False (the caller then leaves every pose untouched)."""
    _, _, _, _, sel = _handoff(frac=0.0)
    assert not bool(sel.any())


def test_handoff_fraction_is_respected():
    """The selected share tracks handoff_spawn_frac (the intended arm value ~0.33)."""
    for frac in (0.25, 0.33, 0.5, 1.0):
        _, _, _, _, sel = _handoff(n=20000, frac=frac, seed=7)
        assert sel.float().mean().item() == pytest.approx(frac, abs=0.02), frac


def test_handoff_slant_range_lands_in_the_sampled_band():
    """The 3-D range to the gate -- the quantity the deploy first-lock logs report (10.3-11.2 m) -- lands
    inside [range_lo, range_hi]. It is the SLANT range that is sampled: the horizontal leg is solved from
    it and the realised vertical offset (r_h = sqrt(r^2 - dz^2)), not the other way round."""
    g, _, pos, _, _ = _handoff(n=4096, seed=3)
    r = torch.linalg.norm(g - pos, dim=-1)
    assert float(r.min()) >= 9.5 - 1e-6 and float(r.max()) <= 11.5 + 1e-6, (r.min(), r.max())
    # the band is actually EXERCISED, not collapsed onto a point (the spec's "sample a band")
    assert float(r.max() - r.min()) > 1.5


def test_handoff_gate_sits_near_the_drone_altitude():
    """The gate centre lands within the requested +-1.5 m vertical band of the drone -> elevation ~+-9 deg,
    i.e. ~10-29 deg BELOW the level pose's +19.5 deg optical axis == the wire's in-frame position."""
    g, _, pos, _, _ = _handoff(n=4096, gate=(12.0, 0.0, 5.0), seed=11)
    dz = g[:, 2] - pos[:, 2]
    assert float(dz.min()) >= -1.5 - 1e-6 and float(dz.max()) <= 1.5 + 1e-6
    el = torch.rad2deg(torch.atan2(dz, torch.linalg.norm(g[:, :2] - pos[:, :2], dim=-1)))
    assert float(el.abs().max()) < 9.2, float(el.abs().max())         # atan(1.5 / sqrt(9.5^2-1.5^2))


def test_handoff_agl_floor_clamps_and_realises_the_offset():
    """With a LOW gate the requested "gate below the drone" offset would put the drone under the pad-relative
    floor; the AGL clamp bites, the drone never goes below pad + min_agl, and the REALISED dz (not the
    requested one) is what the horizontal leg is solved from -> the slant range is still exact."""
    g, a, pos, _, _ = _handoff(n=4096, gate=(12.0, 0.0, 0.5), pad=(0.0, 0.0, 0.0), seed=5)
    assert float(pos[:, 2].min()) >= 0.5 - 1e-9                       # pad(0) + min_agl(0.5)
    assert bool((pos[:, 2] <= 0.5 + 1e-9).any())                      # the clamp genuinely bit
    r = torch.linalg.norm(g - pos, dim=-1)
    assert float(r.min()) >= 9.5 - 1e-6 and float(r.max()) <= 11.5 + 1e-6   # range still exact post-clamp


def test_handoff_stays_on_the_incoming_approach_leg():
    """The drone is backed up-course ALONG the incoming leg (pad->gate), so it starts ON the progress
    segment / racing line and the contouring terms see NO spurious spawn offset: the horizontal cross
    product of (gate - pad) with (gate - drone) is ~0 and the drone is on the up-course side."""
    for gate in ((12.0, 0.0, 3.0), (8.0, 9.0, 4.0), (-6.0, 11.0, 2.0)):
        g, a, pos, _, _ = _handoff(n=256, gate=gate, seed=1)
        leg = (g - a)[:, :2]
        back = (g - pos)[:, :2]
        cross = leg[:, 0] * back[:, 1] - leg[:, 1] * back[:, 0]
        assert float(cross.abs().max()) < 1e-6, gate                  # collinear in XY
        assert float((leg * back).sum(dim=-1).min()) > 0.0, gate      # up-course, not past the gate


def test_handoff_degenerate_leg_falls_back_to_plus_x():
    """A degenerate incoming leg (gate directly above the approach start) must not divide by ~0: the
    direction falls back to +x and the geometry is still finite and in-band."""
    _, _, pos, bearing, _ = _handoff(n=32, gate=(0.0, 0.0, 3.0), pad=(0.0, 0.0, 0.0))
    assert bool(torch.isfinite(pos).all()) and bool(torch.isfinite(bearing).all())
    assert float(bearing.abs().max()) == pytest.approx(0.0)           # atan2(0, 1) == 0


def _cam_R_level(yaw):
    """The env's _cam_R_wb composition at a LEVEL attitude: cam = R_wb @ Rz_cam (mount +20 already baked)."""
    q = quat_xyzw_from_yaw_pitch(yaw, torch.zeros_like(yaw))
    R = quat_xyzw_to_matrix_torch(q).to(DT)
    return R @ torch.as_tensor(np.array(_RZ_PI_BODY_NP), dtype=DT)


def test_handoff_level_pose_acquires_the_gate_at_t0():
    """THE LOAD-BEARING M1 INVARIANT, driven through the REAL gate_detectable projection: across the whole
    sampled band -- every range, every vertical offset, every gate height, plus the full +-yaw jitter -- a
    LEVEL handoff pose has gate 0 DETECTABLE at t=0. That is what the deploy logs show ("first slot0 lock at
    10.3-11.2 m on EVERY flight"), and it is what makes the spawn an ACQUISITION geometry rather than a
    search problem. Uses the tail-first body yaw = bearing + pi the env builds."""
    for gate in ((12.0, 0.0, 3.0), (12.0, 0.0, 0.5), (12.0, 0.0, 6.0), (7.0, 8.0, 4.0)):
        g, _, pos, bearing, _ = _handoff(n=192, gate=gate, seed=21)
        d_xy = g[:, :2] - pos[:, :2]
        gyaw = torch.atan2(d_xy[:, 1], d_xy[:, 0]).unsqueeze(-1)      # the gate faces back down the approach
        for jit in (-0.25, 0.0, 0.25):                                # the +-handoff_yaw_jitter_rad corners
            yaw = bearing + math.pi + jit
            det, _ = GV.gate_detectable(pos, _cam_R_level(yaw), g.unsqueeze(1), gyaw,
                                        far_cap_m=30.0, is_quat=False)
            assert bool(det.all()), (gate, jit, float(det.float().mean()))


def test_handoff_tilted_pad_pose_aims_17_deg_lower_than_level():
    """WHY M1 EXISTS (the measured OOD, re-derived here so it cannot silently drift): the VQ1 tilted pad
    (-17.8 deg) and the wire's LEVEL handoff aim the camera ~17.5 deg apart. At the pad the gate sits AT OR
    ABOVE the optical axis; at the wire handoff it sits ~20 deg BELOW it -- an input the policy never saw
    from rest, which is what it extrapolates into the -53 deg start dive."""
    yaw = torch.tensor([math.pi], dtype=DT)
    bands = {}
    for name, pitch in (("pad", -0.31), ("level", 0.0)):
        q = quat_xyzw_from_yaw_pitch(yaw, torch.tensor([pitch], dtype=DT))
        cam = quat_xyzw_to_matrix_torch(q).to(DT) @ torch.as_tensor(np.array(_RZ_PI_BODY_NP), dtype=DT)
        ok = []
        for el_deg in np.arange(-60.0, 60.1, 0.5):
            e = math.radians(el_deg)
            gp = _t([[[10.5 * math.cos(e), 0.0, 10.5 * math.sin(e)]]])
            det, _ = GV.gate_detectable(torch.zeros(1, 3, dtype=DT), cam, gp, _t([[math.pi]]),
                                        far_cap_m=30.0, is_quat=False)
            if bool(det[0, 0]):
                ok.append(el_deg)
        bands[name] = (min(ok), max(ok))
    centre = {k: 0.5 * (v[0] + v[1]) for k, v in bands.items()}
    assert centre["level"] == pytest.approx(19.5, abs=1.0), bands     # == the baked +20 deg mount
    assert centre["pad"] == pytest.approx(2.0, abs=1.0), bands        # the pad tilt cancels ~all of it
    assert centre["level"] - centre["pad"] > 15.0, bands
    # a gate at the WIRE's ~0 deg elevation is comfortably in frame LEVEL but sits BELOW the axis
    assert bands["level"][0] < 0.0 < bands["level"][1]


# ---- M1 env wiring, EXECUTED against a stub self ------------------------------------------------
def _mk_handoff_stub(n=8, frac=1.0, gate=(12.0, 0.0, 3.0)):
    stub = object.__new__(EGO.PeregrineRacingEgo)
    stub.device = torch.device("cpu")
    stub.spawn_pos = torch.zeros(n, 3, dtype=DT)
    stub.gate_pos = torch.tensor([[list(gate), [24.0, 4.0, 3.0]]], dtype=DT).expand(n, 2, 3).contiguous()
    stub.target_gates = torch.zeros(n, dtype=torch.long)
    stub._arange = torch.arange(n)
    stub.init_pos = torch.zeros(n, 3, dtype=DT)
    state = torch.zeros(n, 13, dtype=DT)
    state[:, 6] = 1.0                                                 # identity quat (xyzw)
    state[:, 7:10] = 3.0                                              # a NON-zero velocity to be cleared
    stub.dynamics = SimpleNamespace(_state=state)
    stub._handoff_frac = frac
    stub._handoff_range_lo, stub._handoff_range_hi = 9.5, 11.5
    stub._handoff_dz_lo, stub._handoff_dz_hi = -1.5, 1.5
    stub._handoff_min_agl = 0.5
    stub._handoff_yaw_jitter, stub._handoff_att_jitter = 0.25, 0.3
    return stub


def test_handoff_wiring_off_is_a_bitwise_noop_and_draws_no_rng():
    """OFF (frac 0): the dynamics state is BITWISE unchanged AND the global RNG state is untouched -- the
    helper returns BEFORE drawing, so an unarmed run's reset stream stays bit-identical to v1.6."""
    stub = _mk_handoff_stub(frac=0.0)
    before, rng_before = stub.dynamics._state.clone(), torch.random.get_rng_state()
    EGO.PeregrineRacingEgo._apply_handoff_spawn(stub, torch.arange(8))
    assert torch.equal(stub.dynamics._state, before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)      # NOTHING drawn
    assert torch.equal(stub.init_pos, torch.zeros(8, 3, dtype=DT))


def test_handoff_wiring_repose_is_level_at_rest_and_in_band():
    """ARMED (frac 1): every selected env is re-posed at the handoff geometry -- slant range in band,
    velocity ZEROED (the standing hover, even though the stub started at 3 m/s), body rates untouched, a
    unit quaternion, and a LEVEL attitude (|pitch| only the +-0.15 rad jitter, NOT the -17.8 deg pad)."""
    torch.manual_seed(0)
    stub = _mk_handoff_stub(n=256, frac=1.0)
    EGO.PeregrineRacingEgo._apply_handoff_spawn(stub, torch.arange(256))
    st = stub.dynamics._state
    pos, q, vel, rates = st[:, 0:3], st[:, 3:7], st[:, 7:10], st[:, 10:13]
    r = torch.linalg.norm(stub.gate_pos[:, 0] - pos, dim=-1)
    # 1e-4 (not 1e-6): the env draws float32 uniforms against float64 geometry, so the band edges carry
    # single-precision slop. The PURE-function tests above pin the band in exact float64.
    assert float(r.min()) >= 9.5 - 1e-4 and float(r.max()) <= 11.5 + 1e-4
    assert torch.equal(vel, torch.zeros_like(vel))                    # AT REST
    assert torch.equal(rates, torch.zeros_like(rates))                # untouched (already 0)
    assert torch.allclose(torch.linalg.norm(q, dim=-1), torch.ones(256, dtype=DT), atol=1e-9)
    R = quat_xyzw_to_matrix_torch(q).to(DT)
    pitch = torch.asin((-R[:, 2, 0]).clamp(-1.0, 1.0))                # leveled pitch from body->world
    # LEVEL + the base env's own +-0.15 rad attitude jitter -- NOT the VQ1 pad's fixed -0.31 rad tilt
    assert float(pitch.abs().mean()) < 0.10, float(pitch.abs().mean())
    assert float(pitch.abs().max()) < 0.20, float(pitch.abs().max())
    assert float(pitch.abs().mean()) < 0.31 / 2.0                     # nowhere near the -17.8 deg pad
    assert torch.equal(stub.init_pos, pos)                            # init_pos follows the re-pose


def test_handoff_wiring_leaves_unselected_envs_untouched():
    """A PARTIAL arm (the intended ~0.33) re-poses ONLY the selected envs; the rest keep the base env's
    standing-start pose bit-for-bit -- it is a SIBLING mode, not a replacement."""
    torch.manual_seed(3)
    stub = _mk_handoff_stub(n=512, frac=0.33)
    before = stub.dynamics._state.clone()
    EGO.PeregrineRacingEgo._apply_handoff_spawn(stub, torch.arange(512))
    moved = (stub.dynamics._state[:, 0:3] - before[:, 0:3]).abs().sum(dim=-1) > 0
    assert 0.25 < moved.float().mean().item() < 0.42, moved.float().mean().item()
    untouched = ~moved
    assert torch.equal(stub.dynamics._state[untouched], before[untouched])


# ================================================================================================
# M2 -- BLIND-FLIGHT ABORT.
# ================================================================================================
def _blind(n=1):
    """Fresh (clock, acquired) blind-abort state for n envs."""
    return torch.zeros(n, dtype=DT), torch.zeros(n, dtype=torch.bool)


def _tick(clock, acq, det, *, dist=8.0, closing=True, dt=0.1, abort_s=1.2, range_m=15.0,
          since=None, grace=0.0):
    n = clock.shape[0]
    return EGO.blind_abort_update(
        clock, acq, torch.tensor([det] * n), torch.full((n,), dist, dtype=DT),
        torch.tensor([closing] * n), dt, abort_s, range_m,
        since_advance_s=since, grace_s=grace)


def test_blind_clock_needs_acquisition_first():
    """A gate that has NEVER been in frame banks NO blind time: the had-then-lost precondition is the
    whole scoping mechanism, so an un-acquired gate can never abort no matter how long it stays dark."""
    clock, acq = _blind()
    for _ in range(100):                                              # 10 s of never-acquired darkness
        clock, acq, ab = _tick(clock, acq, False)
        assert not bool(ab.any())
    assert float(clock[0]) == 0.0 and not bool(acq[0])


def test_blind_clock_accumulates_after_acquisition_and_resets_on_sight():
    """Once acquired, blind ticks accumulate dt; ANY re-sight resets the clock to 0 (it measures the
    CURRENT continuous blackout, never a lifetime total)."""
    clock, acq = _blind()
    clock, acq, _ = _tick(clock, acq, True)                           # acquire
    assert bool(acq[0]) and float(clock[0]) == 0.0
    for k in range(1, 6):
        clock, acq, _ = _tick(clock, acq, False)
        assert float(clock[0]) == pytest.approx(0.1 * k)
    clock, acq, _ = _tick(clock, acq, True)                           # re-sight
    assert float(clock[0]) == 0.0 and bool(acq[0])                    # acquired LATCHES, clock resets


def test_blind_abort_fires_on_the_start_dive():
    """THE TARGET FAILURE: gate acquired at t=0 (the handoff spawn guarantees it), then the dive puts it
    out of frame and keeps it there while the drone still closes inside 15 m -> abort at blind_abort_s."""
    clock, acq = _blind()
    clock, acq, _ = _tick(clock, acq, True)                           # acquired on tick 0
    fired_at = None
    for k in range(1, 30):
        clock, acq, ab = _tick(clock, acq, False, dist=10.5, closing=True)
        if bool(ab[0]):
            fired_at = 0.1 * k
            break
    assert fired_at is not None and fired_at == pytest.approx(1.3, abs=1e-9)   # first tick PAST 1.2 s
    assert fired_at > 1.2


def test_blind_abort_does_not_fire_on_the_post_pass_acquisition_gap():
    """THE SCOPING TEST (the explicit requirement). The measured 0.7-5.8 s post-pass blackout on a descend
    leg must NOT abort. The caller clears ``acquired`` on every target ADVANCE, so the NEW gate is
    un-acquired: 6 s of darkness banks no clock and fires nothing -- an EXACT, event-driven grace rather
    than a tuned timeout. Once the new gate IS framed and then lost, the abort arms again (as it should)."""
    clock, acq = _blind()
    clock, acq, _ = _tick(clock, acq, True)                           # gate k framed
    for _ in range(5):
        clock, acq, _ = _tick(clock, acq, False)                      # some blind time banked on gate k
    assert float(clock[0]) > 0.0
    # --- the pass: the env's advance clear (peregrine_racing_ego.step) ---
    clock, acq = torch.zeros_like(clock), torch.zeros_like(acq)
    for _ in range(60):                                               # 6.0 s -- the WORST measured gap
        clock, acq, ab = _tick(clock, acq, False, dist=12.0, closing=True)
        assert not bool(ab.any()), "post-pass acquisition gap must NEVER abort"
    # the new gate finally comes into frame -> only NOW can a subsequent framing loss abort
    clock, acq, _ = _tick(clock, acq, True, dist=12.0)
    fired = False
    for _ in range(20):
        clock, acq, ab = _tick(clock, acq, False, dist=12.0, closing=True)
        fired = fired or bool(ab[0])
    assert fired, "after a genuine acquire-then-lose the abort must arm again"


def test_blind_abort_range_and_closing_gates():
    """A gate BEYOND blind_abort_range_m (not yet a framing failure) and a RECEDING drone (flying away, not
    at it) both suppress the abort even with a long banked blind clock."""
    for kw in (dict(dist=25.0, closing=True), dict(dist=8.0, closing=False)):
        clock, acq = _blind()
        clock, acq, _ = _tick(clock, acq, True, **kw)
        for _ in range(60):
            clock, acq, ab = _tick(clock, acq, False, **kw)
            assert not bool(ab.any()), kw
        assert float(clock[0]) > 1.2                                  # the clock DID run; only the gate held


def test_blind_abort_off_is_byte_identical():
    """blind_abort_s <= 0 -> the abort is all-False for ANY state (the env additionally skips the whole
    block, so nothing is even computed)."""
    clock, acq = _blind()
    clock, acq, _ = _tick(clock, acq, True)
    for _ in range(100):
        clock, acq, ab = _tick(clock, acq, False, abort_s=0.0)
        assert not bool(ab.any())
    assert float(clock[0]) > 5.0                                      # state still tracked, just inert


def test_blind_abort_optional_advance_grace():
    """The belt-and-braces ``blind_abort_grace_s`` suppresses aborts for a while after an advance, on TOP
    of the acquired-reset. grace 0 (the default) == the acquired-reset alone."""
    def _first_fire(grace):
        clock, acq = _blind()
        clock, acq, _ = _tick(clock, acq, True)                       # acquire, then go blind for good
        since = torch.zeros(1, dtype=DT)
        for k in range(1, 60):
            since = since + 0.1
            clock, acq, ab = _tick(clock, acq, False, since=since, grace=grace)
            if bool(ab[0]):
                return 0.1 * k
        return None

    plain = _first_fire(0.0)
    assert plain == pytest.approx(1.3, abs=1e-9)                      # default == the acquired-reset alone
    held = _first_fire(2.5)
    assert held is not None and held >= 2.5 and held > plain          # held off until past the grace


def test_blind_abort_env_wiring_is_present():
    """WIRING WATCHDOG for the parts of M2 that live in step()/reset_idx (cluster-only -- diffaero), so a
    refactor cannot silently drop them. Asserted on the SOURCE, the tests/test_v15_terms.py convention:
      * the ADVANCE CLEAR (``_blind_acquired[advance] = False``) -- the post-pass scoping guarantee;
      * the THREADED-GATE mask (``blind_abort & ~gate_passed``) -- a clean crossing always wins;
      * the OOB-CLASS fold (``oob_full | blind_abort``) -- terminal_oob, banked progress KEPT;
      * the RESET clear + the exit class + the two TB keys."""
    src = (ROOT / "rl" / "peregrine_racing_ego.py").read_text(encoding="utf-8")
    for frag in ("self._blind_acquired[advance] = False",
                 "blind_abort = blind_abort & ~gate_passed",
                 "oob_full = oob_full | blind_abort",
                 "self._blind_acquired[env_idx] = False",
                 "c_blind = _take(blind_abort)",
                 '"exit_blind"',
                 '"blind_abort_rate"',
                 '"blind_clock_mean"'):
        assert frag in src, frag


# ================================================================================================
# M3 -- PROGRESS FRAMING MULTIPLIER.
# ================================================================================================
def _frame(cos_view, det=None, mult=1.0, floor=0.5, scale=0.5, exp=2.0):
    return ER.progress_frame_factor(_t(cos_view), det, mult, floor, scale, exp)


def test_frame_factor_is_one_dead_centre_and_decays_to_the_floor():
    """f == 1 with the gate on the optical axis, decaying MONOTONICALLY toward the floor as it drifts to
    the frame edge -- and never below the floor (the anti-freeze guarantee)."""
    delta = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.2])               # rad off-axis
    f = _frame(np.cos(delta).tolist())
    assert f[0].item() == pytest.approx(1.0, abs=1e-12)               # dead centre -> FULL credit
    assert torch.all(f[1:] - f[:-1] <= 1e-12)                         # monotone decreasing
    assert float(f.min()) >= 0.5 - 1e-12 and float(f.max()) <= 1.0 + 1e-12   # never below the floor
    # the UNCLAMPED shape (floor 0): at delta == scale the framing weight is exactly 1/e
    f0 = _frame(np.cos(delta).tolist(), floor=0.0)
    assert f0[3].item() == pytest.approx(math.exp(-1.0), abs=1e-9)
    assert f[3].item() == pytest.approx(0.5, abs=1e-12)               # ... but the floor holds it up


def test_frame_factor_undetected_gate_pays_the_floor():
    """An OUT-OF-FRAME gate is hard-masked to the floor even if its centre angle happens to be small
    (occluded / beyond the far cap) -- 'in frame' means the SAME geometric mask the M2 abort clocks."""
    det = torch.tensor([True, False])
    f = _frame([1.0, 1.0], det=det)
    assert f[0].item() == pytest.approx(1.0)
    assert f[1].item() == pytest.approx(0.5)                          # floor


def test_frame_factor_blend_strength_and_floor_knobs():
    """progress_frame_mult is a BLEND STRENGTH: 0 -> f == 1 everywhere, 1 -> the full [floor, 1] swing,
    and intermediate values interpolate. The floor knob raises the worst case."""
    cos_edge = [math.cos(1.5)]                                        # far off-axis -> g ~ 0
    assert _frame(cos_edge, mult=0.0)[0].item() == pytest.approx(1.0, abs=1e-12)
    assert _frame(cos_edge, mult=1.0)[0].item() == pytest.approx(0.5, abs=1e-9)      # -> floor
    assert _frame(cos_edge, mult=0.5)[0].item() == pytest.approx(0.5, abs=1e-3)      # 1-0.5 == the floor
    assert _frame(cos_edge, mult=0.5, floor=0.2)[0].item() == pytest.approx(0.5, abs=1e-3)
    assert _frame(cos_edge, mult=1.0, floor=0.2)[0].item() == pytest.approx(0.2, abs=1e-9)
    # a MID strength genuinely interpolates at a MID angle (not just at the rails)
    mid = _frame([math.cos(0.5)], mult=0.5, floor=0.0)[0].item()
    assert 0.5 < mid < 1.0


def _kw(n, *, s_curr=None, s_prev=None, cos_view=None, frame_detectable=None):
    """Minimal compute_ego_reward kwargs; progress driven by s_curr - s_prev, other terms inert."""
    return dict(
        s_curr=(torch.zeros(n, dtype=DT) if s_curr is None else s_curr),
        s_prev=(torch.zeros(n, dtype=DT) if s_prev is None else s_prev),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=torch.zeros(n, dtype=DT), w_g_half=0.75,
        gate_collision=torch.zeros(n, dtype=torch.bool), gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool), banked_progress_return=torch.zeros(n, dtype=DT),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=torch.zeros(n, dtype=DT),
        tilt_cos_r33=torch.ones(n, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT),
        curr_center=torch.zeros(n, 3, dtype=DT), next_center=torch.zeros(n, 3, dtype=DT), dt=DT_S,
        cos_view=cos_view, frame_detectable=frame_detectable,
    )


def test_frame_multiplier_wired_scales_positive_progress_only():
    """Wired into compute_ego_reward: a BLIND leg banks only ``floor`` of a framed leg's progress, while
    NEGATIVE progress (backing up) is credited in FULL -- scaling it would make retreating CHEAPER while
    blind, an incentive to lose the gate on the way backwards."""
    sc, sp = _t([0.2, 0.2, -0.2, -0.2]), torch.zeros(4, dtype=DT)
    cos_view = _t([1.0, 1.0, 1.0, 1.0])
    det = torch.tensor([True, False, True, False])                    # framed / blind / framed / blind
    w = ER.EgoRewardWeights(progress=2.0, progress_frame_mult=1.0, progress_frame_floor=0.5)
    _, comp, r_prog = ER.compute_ego_reward(w, **_kw(4, s_curr=sc, s_prev=sp, cos_view=cos_view,
                                                    frame_detectable=det))
    assert r_prog[0].item() == pytest.approx(0.4)                     # framed  -> full credit
    assert r_prog[1].item() == pytest.approx(0.2)                     # blind   -> floor * credit
    assert r_prog[2].item() == pytest.approx(-0.4)                    # retreat -> FULL, framed
    assert r_prog[3].item() == pytest.approx(-0.4)                    # retreat -> FULL, blind too
    assert comp["frame_factor"] == pytest.approx((1.0 + 0.5 + 1.0 + 0.5) / 4.0)


def test_frame_multiplier_can_only_shrink_credit_never_inflate_it():
    """THE PUMP CLOSURE: f is bounded above by 1, so for ANY framing the multiplied credit is <= the
    unmultiplied one. The multiplier can only ever REDUCE income -> no cycle can pump reward out of the
    (deliberately) broken telescoping."""
    sc = _t(np.linspace(-0.5, 0.5, 21).tolist())
    kw_off = _kw(21, s_curr=sc, s_prev=torch.zeros(21, dtype=DT))
    w_off = ER.EgoRewardWeights(progress=2.0)
    _, _, rp_off = ER.compute_ego_reward(w_off, **kw_off)
    w_on = ER.EgoRewardWeights(progress=2.0, progress_frame_mult=1.0)
    for cos in (1.0, 0.9, 0.5, 0.0, -1.0):
        kw_on = _kw(21, s_curr=sc, s_prev=torch.zeros(21, dtype=DT),
                    cos_view=torch.full((21,), cos, dtype=DT))
        _, _, rp_on = ER.compute_ego_reward(w_on, **kw_on)
        assert torch.all(rp_on <= rp_off + 1e-12), cos


def test_frame_multiplier_is_farm_resistant_stationary_staring_earns_nothing():
    """FARM RESISTANCE BY CONSTRUCTION: the term MULTIPLIES progress, so a policy that hovers and stares at
    the gate (zero progress, perfect framing) earns f * 0 == 0 -- IDENTICAL to the multiplier being off,
    and still a NET LOSS per tick once rw_time is paid. Unlike the additive perception carrot this needs no
    farm-neutrality budget at all."""
    kw_stare = dict(s_curr=torch.zeros(2, dtype=DT), s_prev=torch.zeros(2, dtype=DT),
                    cos_view=torch.ones(2, dtype=DT), frame_detectable=torch.ones(2, dtype=torch.bool))
    w_on = ER.EgoRewardWeights(progress=2.0, progress_frame_mult=1.0, perception=0.02, time=0.02)
    w_off = ER.EgoRewardWeights(progress=2.0, perception=0.02, time=0.02)
    r_on, _, rp_on = ER.compute_ego_reward(w_on, **_kw(2, **kw_stare))
    r_off, _, rp_off = ER.compute_ego_reward(w_off, **_kw(2, **kw_stare))
    assert torch.equal(rp_on, torch.zeros_like(rp_on))                # zero progress -> zero credit
    assert torch.equal(r_on, r_off)                                   # arming it changes NOTHING here
    assert float(r_on.max()) <= 1e-12                                 # staring still nets <= 0 per tick


def test_frame_multiplier_guards_reject_a_pump_or_a_freeze():
    """__post_init__ brackets: strength in [0,1], floor in [0,1] (a floor > 1 would INFLATE credit -> a
    pump), and a positive angular scale."""
    with pytest.raises(AssertionError, match="rw_progress_frame_mult"):
        ER.EgoRewardWeights(progress_frame_mult=1.5)
    with pytest.raises(AssertionError, match="progress_frame_floor"):
        ER.EgoRewardWeights(progress_frame_mult=1.0, progress_frame_floor=1.2)
    with pytest.raises(AssertionError, match="progress_frame_scale_rad"):
        ER.EgoRewardWeights(progress_frame_mult=1.0, progress_frame_scale_rad=0.0)
    ER.EgoRewardWeights(progress_frame_mult=0.0, progress_frame_floor=9.0)   # inert when OFF


def test_frame_weights_resolve_from_cfg():
    """from_cfg's generic loop picks up rw_progress_frame_mult + the bare shape keys (the launcher's
    ++env. convention: weights get rw_, shape params do not)."""
    class _Cfg:
        rw_progress_frame_mult = 1.0
        progress_frame_floor = 0.4
        progress_frame_scale_rad = 0.6
        progress_frame_exponent = 2.0
    w = ER.EgoRewardWeights.from_cfg(_Cfg())
    assert w.progress_frame_mult == 1.0 and w.progress_frame_floor == 0.4
    assert w.progress_frame_scale_rad == 0.6 and w.progress_frame_exponent == 2.0


# ================================================================================================
# M4 -- APERTURE TIGHTENING (pass_margin_final_m 1.0 -> 0.75).
# ================================================================================================
HALF, LATW = 0.75, 2.0                                                # gate half-opening / lateral weight
MARGIN_START = HALF * math.sqrt(LATW ** 2 + 1.0)                       # 1.6771 m


def test_margin_anneal_ends_exactly_at_075_and_holds():
    """The END-HOLD lands EXACTLY on pass_margin_final_m = 0.75 and stays there for the whole hold tail --
    the tight envelope is the guarantee; only the PATH to it is annealed."""
    N, hold = 18000, 0.25
    span = (1.0 - hold) * N
    assert _pass_margin_schedule(0, N, MARGIN_START, 0.75, hold) == pytest.approx(MARGIN_START)
    assert _pass_margin_schedule(int(span), N, MARGIN_START, 0.75, hold) == pytest.approx(0.75)
    for i in (int(span), int(span) + 1, N - 1, N, 2 * N):
        assert _pass_margin_schedule(i, N, MARGIN_START, 0.75, hold) == pytest.approx(0.75), i
    mid = _pass_margin_schedule(int(span // 2), N, MARGIN_START, 0.75, hold)
    assert 0.75 < mid < MARGIN_START                                   # monotone shrink through the front


def test_margin_start_still_exceeds_the_tighter_end():
    """The train-loop wiring RAISES when start <= end (a silently-inert anneal). At the v1.7 value the
    start (full weighted aperture) is still far above 0.75, so the guard passes."""
    assert MARGIN_START > 0.75
    assert MARGIN_START == pytest.approx(1.6771, abs=1e-4)


def test_margin_075_is_a_pure_lateral_tightening():
    """The envelope at 0.75 (lat_w 2): |lat| <= 0.375 m -- HALF the 0.75 m aperture and a 25% tightening of
    v1.6's 0.5 m -- while the vertical bound (0.75 m) still coincides with the aperture itself, so it stays
    INACTIVE. Exactly the right axis: the miner put lateral risk at ~2x vertical and the 7-gate record died
    with a 0.6 m LATERAL offset (weighted miss 1.2 -- already outside even the v1.6 margin)."""
    gp = torch.ones(6, dtype=torch.bool)
    lat = _t([0.370, 0.380, 0.000, 0.000, 0.300, 0.500])
    vert = _t([0.000, 0.000, 0.700, 0.740, 0.300, 0.000])
    _, down = EGO.aperture_margin_reclassify(gp, lat, vert, LATW, 0.75)
    assert down.tolist() == [False, True, False, False, False, True]
    # v1.6 vs v1.7 on the SAME crossings: 0.380 and 0.300/0.300 survive at 1.0, 0.380 does not at 0.75
    _, down_v16 = EGO.aperture_margin_reclassify(gp, lat, vert, LATW, 1.0)
    assert down_v16.tolist() == [False, False, False, False, False, False]
    # the exact envelope corners
    assert 0.75 / LATW == pytest.approx(0.375)                         # lateral half-width
    assert 0.75 >= HALF                                                # vertical bound >= aperture -> inactive


def test_margin_075_reclassification_rate_does_not_explode():
    """OFFLINE SANITY (the spec's ask): fit the reported v1.6 crossing distribution (radial 0.28 m median /
    0.57 m p90 -> an isotropic Rayleigh with sigma ~ 0.24-0.27 m; the two quantiles agree, so the fit holds)
    and price the tightening. 1.0 -> 0.75 moves the downgrade rate from ~4-7% to ~14-19% of geometric
    passes: a ~3x tightening, NOT a collapse, and it is annealed in over the front 75% of training."""
    rng = np.random.default_rng(0)
    for sigma, lo, hi in ((0.2378, 0.10, 0.18), (0.2656, 0.14, 0.24)):
        y, z = rng.normal(0, sigma, 400_000), rng.normal(0, sigma, 400_000)
        geo = (np.abs(y) < HALF) & (np.abs(z) < HALF)                  # a GEOMETRIC pass
        m = np.sqrt((LATW * np.abs(y)) ** 2 + z ** 2)
        rate_075 = (geo & (m > 0.75)).sum() / geo.sum()
        rate_100 = (geo & (m > 1.00)).sum() / geo.sum()
        assert lo < rate_075 < hi, (sigma, rate_075)
        assert rate_100 < rate_075 < 3.6 * rate_100                    # tighter, but same order
        assert (geo & (m > MARGIN_START)).sum() == 0                   # the anneal START is a strict no-op


# ================================================================================================
# DEFAULT-OFF EQUIVALENCE -- the v1.7 code with every new key unset behaves as v1.6.
# ================================================================================================
def test_all_new_reward_weights_default_off():
    """A cfg mentioning NONE of the v1.7 keys resolves every new weight to its OFF value (and the shape
    defaults documented in the launcher header)."""
    w = ER.EgoRewardWeights.from_cfg(type("C", (), {})())
    assert w.progress_frame_mult == 0.0                                # OFF
    assert w.progress_frame_floor == 0.5
    assert w.progress_frame_scale_rad == 0.5
    assert w.progress_frame_exponent == 2.0


def test_all_new_env_knobs_default_off():
    """The env-side v1.7 knobs default to inert values: no handoff spawn, no blind abort (the range/grace
    defaults are SHAPE only and never arm anything by themselves)."""
    src = (ROOT / "rl" / "peregrine_racing_ego.py").read_text(encoding="utf-8")
    for key, default in (('"handoff_spawn_frac", 0.0', "handoff_spawn_frac"),
                         ('"blind_abort_s", 0.0', "blind_abort_s")):
        assert key in src, default
    *_, sel = _handoff(frac=0.0)
    assert not bool(sel.any())
    clock, acq = _blind()
    clock, acq, _ = _tick(clock, acq, True)
    _, _, ab = _tick(clock, acq, False, abort_s=0.0, dist=1.0)
    assert not bool(ab.any())


def test_v17_reward_path_is_bitwise_identical_to_v16_when_unarmed():
    """THE EQUIVALENCE PROOF: with progress_frame_mult unset, compute_ego_reward returns BITWISE the same
    reward / r_prog whether or not the new cos_view + frame_detectable kwargs are supplied -- so an
    unarmed v1.7 run reproduces v1.6 exactly (the frame block is SKIPPED, never multiplied by 1.0)."""
    n = 32
    sc = _t(np.linspace(-0.4, 0.4, n).tolist())
    sp = torch.zeros(n, dtype=DT)
    w = ER.EgoRewardWeights(progress=2.0, progress_vcap_mps=7.5, perception=0.02, time=0.02,
                            yaw_duty=0.15, yaw_jerk=0.05, pitch_duty=0.1, pitch_jerk=0.03)
    kw_v16 = _kw(n, s_curr=sc, s_prev=sp, cos_view=torch.full((n,), 0.7, dtype=DT))
    kw_v16.pop("frame_detectable")                                     # the v1.6 call signature
    kw_v17 = _kw(n, s_curr=sc, s_prev=sp, cos_view=torch.full((n,), 0.7, dtype=DT),
                 frame_detectable=torch.zeros(n, dtype=torch.bool))    # even a fully-BLIND mask
    r16, c16, p16 = ER.compute_ego_reward(w, **kw_v16)
    r17, c17, p17 = ER.compute_ego_reward(w, **kw_v17)
    assert torch.equal(r16, r17), (r16 - r17).abs().max()
    assert torch.equal(p16, p17)
    assert c16["frame_factor"] == 1.0 and c17["frame_factor"] == 1.0   # reported as inert
    assert c16["prog_reward"] == c17["prog_reward"]
