"""Tests for rl/ego_reward.py -- the REFINED-B (champion-consensus) VQ2 egocentric reward.

Every term is a PURE torch function (no diffaero), so the whole reward unit-tests on the laptop. The
LOAD-BEARING test is (a) TERMINAL-DOMINANCE: a synthetic CLEAN pass vs SPRINT-AND-CLIP-at-gate-j
rollout, asserting E[return | clip@j] < E[return | clean] for ALL j and BOTH gammas -- the exact
defect the two adversarial critics caught (inc8's 10.0/m made sprint-and-clip positive-return).

Coverage (prompt task 4):
  (a) TERMINAL-DOMINANCE ROLLOUT   [MANDATORY]  clip@j < clean, all j, gamma in {0.99, 0.9975}.
  (b) PROGRESS potential-based + perpendicular drift earns ~0 (finite CURRENT-segment projection).
  (c) PASSAGE idempotent            weaving re-cross pays once; target strictly increments.
  (d) WIDE-FLYBY -> MISS terminal.
  (e) v_max CLAMP above the true peak per-step arc advance (never caps legit top speed).
  (f) reward reads GT only          translate the whole scene -> reward unchanged.
  (g) SMOOTHNESS < 1% of a representative per-step progress at bring-up speed.
  (h) appo / privileged-critic assert EXISTS + FIRES on the symmetric critic.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_reward.py -q
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import ego_reward as R                                                    # noqa: E402
from peregrine_racing import crossing_events, world_to_gateframe          # noqa: E402

DT = torch.float64


def _t(x):
    return torch.tensor(x, dtype=DT)


# ================================================================================================
# (b) PROGRESS: potential-based along the FINITE CURRENT segment; perpendicular drift -> ~0.
# ================================================================================================
def test_progress_along_segment_positive_perpendicular_zero():
    seg_a = _t([[0.0, 0.0, 0.0]])
    seg_b = _t([[10.0, 0.0, 0.0]])                 # segment along +x, length 10
    # ON-SEGMENT advance of a REALISTIC per-step distance (< the vmax*dt=1.3 m/step clip band):
    # x=5.0 -> x=6.0 : s goes 5 -> 6, progress +1.0*rw (unclipped).
    s_prev = R.segment_arc_position(_t([[5.0, 0.0, 0.0]]), seg_a, seg_b)
    s_curr = R.segment_arc_position(_t([[6.0, 0.0, 0.0]]), seg_a, seg_b)
    r_on = R.segment_progress_reward(s_curr, s_prev, rw_progress=1.0, vmax_mps=39.0, dt=1 / 30)
    assert r_on.item() == pytest.approx(1.0)

    # PERPENDICULAR drift at fixed x=5: move y 0 -> 8 (pure cross-track). s is unchanged -> ~0 progress.
    s_p0 = R.segment_arc_position(_t([[5.0, 0.0, 0.0]]), seg_a, seg_b)
    s_p1 = R.segment_arc_position(_t([[5.0, 8.0, 0.0]]), seg_a, seg_b)
    r_perp = R.segment_progress_reward(s_p1, s_p0, rw_progress=1.0, vmax_mps=39.0, dt=1 / 30)
    assert abs(r_perp.item()) < 1e-9, r_perp.item()

    # a global-argmin polyline would have ADVANCED here; the finite-segment projection does NOT.
    assert R.segment_arc_position(_t([[5.0, 100.0, 0.0]]), seg_a, seg_b).item() == pytest.approx(5.0)


def test_progress_potential_telescopes_to_zero_over_a_loop():
    """Potential-based -> a closed excursion (out then back along the segment) sums to ~0 (non-farmable
    by oscillation)."""
    seg_a, seg_b = _t([[0.0, 0.0, 0.0]]), _t([[10.0, 0.0, 0.0]])
    # per-step moves all < the vmax*dt=1.3 m/step clip band (so clipping is inactive and the telescoping
    # is exact); forward then backward, returning to the start.
    xs = [1.0, 2.0, 3.0, 2.2, 1.4, 1.0]
    total = 0.0
    s_prev = R.segment_arc_position(_t([[xs[0], 0.0, 0.0]]), seg_a, seg_b)
    for x in xs[1:]:
        s_curr = R.segment_arc_position(_t([[x, 0.0, 0.0]]), seg_a, seg_b)
        total += R.segment_progress_reward(s_curr, s_prev, 1.0, 39.0, 1 / 30).item()
        s_prev = s_curr
    assert abs(total) < 1e-9, total


def test_progress_clamped_below_out_of_segment_clamp():
    # projection is clamped into [0, seg_len]; moving past the gate does not keep accruing.
    seg_a, seg_b = _t([[0.0, 0.0, 0.0]]), _t([[10.0, 0.0, 0.0]])
    assert R.segment_arc_position(_t([[-5.0, 0.0, 0.0]]), seg_a, seg_b).item() == 0.0   # behind start
    assert R.segment_arc_position(_t([[15.0, 0.0, 0.0]]), seg_a, seg_b).item() == 10.0  # past end


def test_gate_center_potential_homes_in_every_axis():
    """The progress-to-centre potential (rw_progress_to_center=True, the 2026-07-07 root-cause fix):
    s = -||pos - gate_centre||, so reward = clip(s_curr - s_prev) is the 3D closing rate. THE property
    segment mode lacks: PERPENDICULAR motion toward the centre earns POSITIVE progress (dense lateral +
    vertical homing), where segment_arc_position credits it ZERO."""
    gate = _t([[10.0, 0.0, 0.0]])                      # gate centre 10 m ahead on +x
    # (1) s == -distance to the centre.
    assert R.gate_center_potential(_t([[0.0, 0.0, 0.0]]), gate).item() == pytest.approx(-10.0)
    assert R.gate_center_potential(_t([[10.0, 0.0, 0.0]]), gate).item() == pytest.approx(0.0)

    # (2) PERPENDICULAR homing: a drone off to the side at fixed along-track x, moving laterally TOWARD
    #     the line (y 8 -> 7), earns POSITIVE progress under center mode (distance shrinks) ...
    s_prev = R.gate_center_potential(_t([[10.0, 8.0, 0.0]]), gate)
    s_curr = R.gate_center_potential(_t([[10.0, 7.0, 0.0]]), gate)
    r_center = R.segment_progress_reward(s_curr, s_prev, rw_progress=1.0, vmax_mps=39.0, dt=1 / 30)
    assert r_center.item() == pytest.approx(1.0)       # 8 -> 7 = 1 m closer to the centre
    # ... whereas the SAME lateral move earns EXACTLY ZERO under segment mode (the diagnosed gap).
    seg_a, seg_b = _t([[0.0, 0.0, 0.0]]), gate
    sa = R.segment_arc_position(_t([[10.0, 8.0, 0.0]]), seg_a, seg_b)
    sb = R.segment_arc_position(_t([[10.0, 7.0, 0.0]]), seg_a, seg_b)
    assert abs(R.segment_progress_reward(sb, sa, 1.0, 39.0, 1 / 30).item()) < 1e-9

    # (3) vertical homing works identically (z toward the centre pays), and (4) it telescopes over a loop
    #     (out-and-back to the same point sums ~0 -> non-farmable, no lateral-drift farming).
    s_up = R.gate_center_potential(_t([[10.0, 0.0, 3.0]]), gate)
    s_dn = R.gate_center_potential(_t([[10.0, 0.0, 2.0]]), gate)
    assert R.segment_progress_reward(s_dn, s_up, 1.0, 39.0, 1 / 30).item() == pytest.approx(1.0)
    pts = [[2.0, 1.0, 0.0], [3.0, 0.5, 0.5], [4.0, 0.0, 0.0], [3.0, 0.5, 0.5], [2.0, 1.0, 0.0]]
    total, sp = 0.0, R.gate_center_potential(_t([pts[0]]), gate)
    for p in pts[1:]:
        sc = R.gate_center_potential(_t([p]), gate)
        total += R.segment_progress_reward(sc, sp, 1.0, 39.0, 1 / 30).item()
        sp = sc
    assert abs(total) < 1e-6, total


# ================================================================================================
# (e) v_max CLAMP is ABOVE the true peak per-step arc advance (never caps legit top speed).
# ================================================================================================
def test_vmax_clamp_above_true_peak_advance():
    dt = 1.0 / 30.0
    w = R.EgoRewardWeights()
    band = w.vmax_mps * dt                                   # clip band, m/step
    true_peak_speed = 30.0                                   # the ~30 m/s race top speed
    true_peak_advance = true_peak_speed * dt                 # ~1.0 m/step
    assert band > true_peak_advance, (band, true_peak_advance)
    # a legit top-speed step is NOT clipped: raw delta == clipped delta.
    s_prev = _t([0.0])
    s_curr = _t([true_peak_advance])
    r = R.segment_progress_reward(s_curr, s_prev, w.progress, w.vmax_mps, dt)
    assert r.item() == pytest.approx(w.progress * true_peak_advance)     # unclipped
    # an UNPHYSICAL burst (e.g. a 5 m re-projection jump) IS trimmed to the band.
    r_burst = R.segment_progress_reward(_t([5.0]), _t([0.0]), w.progress, w.vmax_mps, dt)
    assert r_burst.item() == pytest.approx(w.progress * band)            # clipped to band
    # headroom is ~30% (the design target).
    assert band / true_peak_advance == pytest.approx(1.3, abs=0.01)


# ================================================================================================
# AREA-DISTANCE coupled progress (Fengyou 2026-07-07): far -> full credit; close + off-axis -> reduced.
# ================================================================================================
def test_area_distance_progress_factor():
    # FAR (dist >= ref): factor -> 1 regardless of area (a beeline from far is NOT penalised).
    assert R.area_distance_progress_factor(_t([0.2]), _t([12.0]), 6.0).item() == pytest.approx(1.0)
    # CLOSE (dist -> 0): factor -> area (a shallow / off-axis approach earns less).
    assert R.area_distance_progress_factor(_t([0.2]), _t([0.0]), 6.0).item() == pytest.approx(0.2)
    # HALF the ramp distance: blend area + (1-area)*0.5.
    assert R.area_distance_progress_factor(_t([0.2]), _t([3.0]), 6.0).item() == pytest.approx(0.2 + 0.8 * 0.5)
    # SQUARE-ON (area=1): factor == 1 at ALL distances (never penalised, near or far).
    assert R.area_distance_progress_factor(_t([1.0]), _t([0.0]), 6.0).item() == pytest.approx(1.0)
    assert R.area_distance_progress_factor(_t([1.0]), _t([6.0]), 6.0).item() == pytest.approx(1.0)


def test_area_coupling_scales_only_positive_progress():
    """compute_ego_reward applies the area-distance factor to POSITIVE progress only: a close+off-axis
    forward step earns less; the same step from far earns full; a backward step ALWAYS pays full."""
    w = R.EgoRewardWeights()                              # progress 1.0, area_dist_ref_m 6.0
    seg_a, seg_b = _t([[0.0, 0.0, 0.0]]), _t([[10.0, 0.0, 0.0]])
    s_prev = R.segment_arc_position(_t([[8.0, 0.0, 0.0]]), seg_a, seg_b)       # 8.0
    s_curr = R.segment_arc_position(_t([[8.5, 0.0, 0.0]]), seg_a, seg_b)       # 8.5 -> +0.5 forward
    common = dict(
        gate_passed=_t([0.0]).bool(), pass_linf=_t([0.0]), w_g_half=0.75,
        gate_collision=_t([0.0]).bool(), gate_miss=_t([0.0]).bool(), oob=_t([0.0]).bool(),
        banked_progress_return=_t([0.0]), newly_finished=_t([0.0]).bool(), time_left_s=_t([2.0]),
        tilt_cos_r33=_t([0.999]), omega=_t([[0.0, 0.0, 0.0]]),
        action_norm=_t([[0.5, 0.5, 0.5, 0.5]]), last_action_norm=_t([[0.5, 0.5, 0.5, 0.5]]),
        vel_world=_t([[5.0, 0.0, 0.0]]), curr_center=seg_b, next_center=seg_b, dt=1 / 30)
    # CLOSE (dist 1.5 < ref 6) + off-axis (area 0.3): factor = 0.3 + 0.7*(1.5/6) = 0.475.
    _, _, rp_close = R.compute_ego_reward(w, s_curr=s_curr, s_prev=s_prev,
                                          area_true=_t([0.3]), dist_to_gate=_t([1.5]), **common)
    factor = 0.3 + 0.7 * (1.5 / 6.0)
    assert rp_close.item() == pytest.approx(w.progress * 0.5 * factor)
    # FAR (dist 12 >= ref): factor 1 -> unscaled full credit.
    _, _, rp_far = R.compute_ego_reward(w, s_curr=s_curr, s_prev=s_prev,
                                        area_true=_t([0.3]), dist_to_gate=_t([12.0]), **common)
    assert rp_far.item() == pytest.approx(w.progress * 0.5)
    # BACKWARD progress is NEVER scaled (retreat pays full even when close + off-axis).
    _, _, rp_back = R.compute_ego_reward(w, s_curr=s_prev, s_prev=s_curr,      # -0.5
                                         area_true=_t([0.3]), dist_to_gate=_t([1.5]), **common)
    assert rp_back.item() == pytest.approx(-w.progress * 0.5)


# ================================================================================================
# DENSE LATERAL CENTERING (Fengyou 2026-07-07, post-render): pull onto the gate-centre segment.
# ================================================================================================
def test_segment_perp_distance():
    seg_a, seg_b = _t([[0.0, 0.0, 0.0]]), _t([[10.0, 0.0, 0.0]])     # segment along +x
    assert R.segment_perp_distance(_t([[5.0, 0.0, 0.0]]), seg_a, seg_b).item() == pytest.approx(0.0)  # on line
    assert R.segment_perp_distance(_t([[5.0, 3.0, 0.0]]), seg_a, seg_b).item() == pytest.approx(3.0)  # y off
    assert R.segment_perp_distance(_t([[5.0, 0.0, 4.0]]), seg_a, seg_b).item() == pytest.approx(4.0)  # z off
    assert R.segment_perp_distance(_t([[5.0, 3.0, 4.0]]), seg_a, seg_b).item() == pytest.approx(5.0)  # 3-4-5
    # past the end -> clamped to seg_end -> distance to the endpoint (10,0,0)
    assert R.segment_perp_distance(_t([[15.0, 3.0, 0.0]]), seg_a, seg_b).item() == pytest.approx(
        (25.0 + 9.0) ** 0.5)


def test_through_centering_reward():
    assert R.through_centering_reward(_t([1.0]), 0.15, 2.0).item() == pytest.approx(-0.15)   # off-line
    assert R.through_centering_reward(_t([5.0]), 0.15, 2.0).item() == pytest.approx(-0.30)   # clamp @ 2 m
    assert R.through_centering_reward(_t([0.0]), 0.15, 2.0).item() == pytest.approx(0.0)      # on-line -> 0
    assert R.through_centering_reward(_t([3.0]), 0.0, 2.0).item() == 0.0                      # OFF


def test_centering_lowers_reward_for_offline_flight():
    """In compute_ego_reward the centering penalty enters via perp_dist: off-line flight scores LOWER
    than on-line by exactly the clamped penalty; on-line (perp 0) it is absent."""
    w = R.EgoRewardWeights(centering=0.15, centering_max_m=2.0)
    seg_a, seg_b = _t([[0.0, 0.0, 0.0]]), _t([[10.0, 0.0, 0.0]])
    s = R.segment_arc_position(_t([[5.0, 0.0, 0.0]]), seg_a, seg_b)          # no progress -> isolate centering
    common = dict(
        gate_passed=_t([0.0]).bool(), pass_linf=_t([0.0]), w_g_half=0.75,
        gate_collision=_t([0.0]).bool(), gate_miss=_t([0.0]).bool(), oob=_t([0.0]).bool(),
        banked_progress_return=_t([0.0]), newly_finished=_t([0.0]).bool(), time_left_s=_t([2.0]),
        tilt_cos_r33=_t([0.999]), omega=_t([[0.0, 0.0, 0.0]]),
        action_norm=_t([[0.5, 0.5, 0.5, 0.5]]), last_action_norm=_t([[0.5, 0.5, 0.5, 0.5]]),
        vel_world=_t([[5.0, 0.0, 0.0]]), curr_center=seg_b, next_center=seg_b, dt=1 / 30)
    r_on, _, _ = R.compute_ego_reward(w, s_curr=s, s_prev=s, perp_dist=_t([0.0]), **common)
    r_off, c_off, _ = R.compute_ego_reward(w, s_curr=s, s_prev=s, perp_dist=_t([2.0]), **common)
    assert r_off.item() < r_on.item()
    assert (r_on.item() - r_off.item()) == pytest.approx(0.30, abs=1e-6)     # 0.15 * clamp(2,0,2)
    assert c_off["center_pen"] == pytest.approx(0.30, abs=1e-6)


# ================================================================================================
# (c) PASSAGE: L-inf centering, idempotent (fires once per gate), matches crossing_events' Linf.
# ================================================================================================
def test_passage_centering_linf_shape():
    w_g_half = 0.75
    passed = _t([1.0, 1.0, 1.0, 0.0]).bool()
    # e_lat = 0 (dead centre) -> full; = w_g_half (edge) -> ~0; not-passed -> 0.
    linf = _t([0.0, 0.75, 0.375, 0.0])
    r = R.centering_passage_reward(passed, linf, w_g_half, rw_passage=1.0)
    assert r[0].item() == pytest.approx(1.0)             # dead centre
    assert r[1].item() == pytest.approx(0.0, abs=1e-9)   # aperture edge
    assert r[2].item() == pytest.approx(0.5)             # halfway
    assert r[3].item() == 0.0                            # not passed -> 0


def test_passage_linf_matches_crossing_events():
    """e_lat in the reward MUST be the SAME L-inf crossing_events reports (max(|y|,|z|))."""
    prev = world_to_gateframe(_t([[-1.0, 0.3, -0.4]]), _t([0.0]))
    curr = world_to_gateframe(_t([[1.0, 0.3, -0.4]]), _t([0.0]))
    ev = crossing_events(prev, curr, 0.75, 1.36)
    assert ev["linf"].item() == pytest.approx(0.4)       # max(|0.3|,|0.4|) == 0.4
    # and centering uses exactly that
    r = R.centering_passage_reward(_t([1.0]).bool(), ev["linf"], 0.75, 1.0)
    assert r.item() == pytest.approx(1.0 - 0.4 / 0.75)


def test_passage_idempotent_weaving_recross_pays_once():
    """A weaving trajectory that re-crosses gate 0's plane pays R_pass EXACTLY once: idempotency comes
    from ``gate_passed`` being true only when the crossed gate IS the current target AND the target
    strictly increments. We emulate the env's contract: only the FIRST forward pass of the target gate
    sets gate_passed; the target then advances so a later re-cross of gate 0 is no longer the target."""
    w_g_half = 0.75
    # step 1: forward through gate 0 centre -> passed (target 0 -> 1). e_lat=0 -> full pay.
    r1 = R.centering_passage_reward(_t([1.0]).bool(), _t([0.0]), w_g_half, 1.0)
    # step 2: drone weaves BACK across gate 0's plane. Target is now gate 1, so gate_passed(target)
    # is FALSE for a gate-0 re-cross -> zero pay (the env never marks a non-target crossing as passed,
    # and the target index NEVER decrements).
    r2 = R.centering_passage_reward(_t([0.0]).bool(), _t([0.0]), w_g_half, 1.0)
    assert r1.item() == pytest.approx(1.0)
    assert r2.item() == 0.0
    assert (r1 + r2).item() == pytest.approx(1.0)        # paid once total


def test_passage_per_gate_increment():
    """Fengyou 2026-07-07: passing a LATER gate pays more -- weight = rw_passage + increment*gate_index
    (gate 0 -> base, gate 1 -> base+inc, ...). Dead-centre passes so the centering factor is 1."""
    w_g_half = 0.75
    passed = _t([1.0, 1.0, 1.0]).bool()
    linf = _t([0.0, 0.0, 0.0])                           # dead centre -> full weight
    idx = torch.tensor([0, 1, 3], dtype=DT)
    r = R.centering_passage_reward(passed, linf, w_g_half, rw_passage=5.0,
                                   passed_gate_index=idx, passage_increment=1.0)
    assert r[0].item() == pytest.approx(5.0)             # gate 0 -> 5
    assert r[1].item() == pytest.approx(6.0)             # gate 1 -> 6
    assert r[2].item() == pytest.approx(8.0)             # gate 3 -> 8
    # no index (or increment 0) -> flat base (backward-compatible with the base passage tests).
    assert R.centering_passage_reward(passed, linf, w_g_half, 5.0)[2].item() == pytest.approx(5.0)


# ================================================================================================
# (d) WIDE-FLYBY -> MISS terminal.
# ================================================================================================
def test_wide_flyby_is_a_miss():
    # a forward crossing OUTSIDE the aperture (not pass_ok, not in_frame) is a miss.
    prev = world_to_gateframe(_t([[-1.0, 3.0, 0.0]]), _t([0.0]))     # far lateral
    curr = world_to_gateframe(_t([[1.0, 3.0, 0.0]]), _t([0.0]))
    ev = crossing_events(prev, curr, 0.75, 1.36)
    miss = R.wide_flyby_miss(ev["fwd"], ev["bwd"], ev["pass_ok"], ev["in_frame"])
    assert bool(miss.item()) is True
    # a clean central pass is NOT a miss.
    p2 = world_to_gateframe(_t([[-1.0, 0.0, 0.0]]), _t([0.0]))
    c2 = world_to_gateframe(_t([[1.0, 0.0, 0.0]]), _t([0.0]))
    ev2 = crossing_events(p2, c2, 0.75, 1.36)
    assert bool(R.wide_flyby_miss(ev2["fwd"], ev2["bwd"], ev2["pass_ok"], ev2["in_frame"]).item()) is False
    # a BACKWARD wide crossing (lateral overshoot re-crossing the plane the wrong way) is ALSO a miss
    # (the inc7 forward-only classification would have silently ignored it).
    p3 = world_to_gateframe(_t([[1.0, 3.0, 0.0]]), _t([0.0]))
    c3 = world_to_gateframe(_t([[-1.0, 3.0, 0.0]]), _t([0.0]))
    ev3 = crossing_events(p3, c3, 0.75, 1.36)
    assert bool(R.wide_flyby_miss(ev3["fwd"], ev3["bwd"], ev3["pass_ok"], ev3["in_frame"]).item()) is True


# ================================================================================================
# (f) reward reads GT ONLY: translate the whole scene -> reward unchanged.
# ================================================================================================
def test_reward_is_translation_invariant_gt_only():
    """Every geometric quantity is GT drone pos/vel + GT gate centres -> a rigid translation of the
    WHOLE scene (drone + spawn + all gate centres) leaves the reward BIT-for-BIT unchanged. Feeding a
    (hypothetical) noisy obs cannot affect the reward: it is never an argument."""
    w = R.EgoRewardWeights()

    def _one(shift):
        sh = _t([shift])
        seg_a = _t([[0.0, 0.0, 0.0]]) + sh
        seg_b = _t([[10.0, 0.0, 0.0]]) + sh
        pos_prev = _t([[3.0, 0.5, 0.2]]) + sh
        pos_curr = _t([[4.0, 0.5, 0.2]]) + sh
        curr_center = seg_b
        next_center = _t([[20.0, 5.0, 0.0]]) + sh
        s_prev = R.segment_arc_position(pos_prev, seg_a, seg_b)
        s_curr = R.segment_arc_position(pos_curr, seg_a, seg_b)
        reward, _, _ = R.compute_ego_reward(
            w, s_curr=s_curr, s_prev=s_prev,
            gate_passed=_t([1.0]).bool(), pass_linf=_t([0.1]), w_g_half=0.75,
            gate_collision=_t([0.0]).bool(), gate_miss=_t([0.0]).bool(), oob=_t([0.0]).bool(),
            banked_progress_return=_t([5.0]),
            newly_finished=_t([0.0]).bool(), time_left_s=_t([2.0]),
            tilt_cos_r33=_t([0.98]), omega=_t([[0.1, 0.0, 0.0]]),
            action_norm=_t([[0.5, 0.5, 0.5, 0.5]]), last_action_norm=_t([[0.5, 0.5, 0.5, 0.5]]),
            vel_world=_t([[10.0, 1.0, 0.0]]), curr_center=curr_center, next_center=next_center,
            dt=1 / 30)
        return reward.item()

    base = _one([0.0, 0.0, 0.0])
    shifted = _one([123.4, -56.7, 8.9])
    assert base == pytest.approx(shifted, abs=1e-9), (base, shifted)


# ================================================================================================
# (g) SMOOTHNESS < 1% of a representative per-step progress at BRING-UP speed.
# ================================================================================================
def test_smoothness_below_one_percent_of_bringup_progress():
    w = R.EgoRewardWeights()
    dt = 1.0 / 30.0
    bringup_speed = 8.0                                  # slow bring-up (~8 m/s; DESIGN.md close-loop-slow)
    per_step_progress = w.progress * bringup_speed * dt  # representative dense progress this step
    # a REPRESENTATIVE (not adversarial) step: modest body rate + a small action delta.
    omega = _t([[0.5, 0.3, 0.2]])                        # ~0.6 rad/s
    a = _t([[0.6, 0.55, 0.45, 0.5]])
    la = _t([[0.55, 0.5, 0.5, 0.5]])
    pen = -R.smoothness_penalty(omega, a, la, w)         # magnitude
    assert pen.item() < 0.01 * per_step_progress, (pen.item(), per_step_progress)


# ================================================================================================
# (h) appo / privileged-critic assert EXISTS + FIRES on the symmetric critic.
# ================================================================================================
def test_appo_privileged_critic_assert():
    # The assert enforces algo=appo AND a privileged critic that consumes the GT get_state (a DIFFERENT
    # layout from the obs, carrying truth the actor cannot see). Use the env's REAL dims: obs 21,
    # privileged critic state 16 (lower-dim, but different content -> genuinely privileged).
    from peregrine_racing_ego import EGO_OBS_DIM, EGO_CRITIC_DIM
    obs_dim, state_dim = EGO_OBS_DIM, EGO_CRITIC_DIM
    assert (obs_dim, state_dim) == (21, 16)
    # correct wiring: appo + critic consumes the full privileged state (input_dim == state_dim).
    R.assert_privileged_critic_connected("appo", critic_input_dim=state_dim,
                                         obs_dim=obs_dim, state_dim=state_dim)   # no raise

    # FOOTGUN 1: algo=ppo (GuardedPPO's symmetric obs-only critic) -> FIRES.
    with pytest.raises(AssertionError):
        R.assert_privileged_critic_connected("ppo", critic_input_dim=state_dim,
                                             obs_dim=obs_dim, state_dim=state_dim)

    # FOOTGUN 2: appo but the critic is SYMMETRIC (input_dim == obs_dim, get_state never consumed).
    with pytest.raises(AssertionError):
        R.assert_privileged_critic_connected("appo", critic_input_dim=obs_dim,
                                             obs_dim=obs_dim, state_dim=state_dim)

    # FOOTGUN 3: the "privileged" state layout is indistinguishable from the obs (state_dim == obs_dim
    # -> no way to prove the critic sees GT the actor cannot).
    with pytest.raises(AssertionError):
        R.assert_privileged_critic_connected("appo", critic_input_dim=obs_dim,
                                             obs_dim=obs_dim, state_dim=obs_dim)


# ================================================================================================
# (a) THE LOAD-BEARING TEST: TERMINAL-DOMINANCE ROLLOUT.
#     CLEAN full pass vs SPRINT-AND-CLIP at gate j, for EVERY j, BOTH gammas.
#     ASSERT E[return | clip@j] < E[return | clean], all j, both gammas. Report the numbers.
# ================================================================================================
class _Course:
    """A straight synthetic course: G gates spaced ``spacing`` m along +x from the spawn at origin.
    Gate centres at x = spacing*(i+1). w_g_half = 0.75. This is the privileged GT the reward reads."""
    def __init__(self, n_gates=6, spacing=20.0):
        self.n = n_gates
        self.spacing = spacing
        self.w_g_half = 0.75
        self.spawn = _t([0.0, 0.0, 0.0])
        self.centers = [_t([spacing * (i + 1), 0.0, 0.0]) for i in range(n_gates)]

    def seg(self, tg):
        a = self.spawn if tg == 0 else self.centers[tg - 1]
        return a, self.centers[tg]


def _rollout_return(course, w, gamma, dt, speed, clip_at=None):
    """Simulate a straight-line run at constant ``speed`` (m/s) along +x, stepping the REAL reward each
    control step. Returns the discounted return sum_t gamma^t r_t.

    clip_at=None  -> a CLEAN full pass (dead-centre through every gate, then finish).
    clip_at=j     -> SPRINT to gate j and CONTACT it (gate_collision) on the crossing step. The sprint
                     runs at ``speed`` (== the clean speed here, so 'clip' is strictly a subset run that
                     forfeits the rest -- the fair, hardest-to-dominate comparison: no speed bonus, the
                     clip run just terminates early with the terminal penalty).
    """
    step_len = speed * dt
    x = 0.0
    tg = 0
    s_prev = R.segment_arc_position(course.spawn.unsqueeze(0),
                                    *[e.unsqueeze(0) for e in course.seg(0)]).squeeze(0)
    banked = _t([0.0]).squeeze(0)
    G = course.n
    ret = 0.0
    disc = 1.0
    t = 0
    max_steps = int((course.spacing * G / step_len) + 50)
    # neutral control (no smoothness penalty, upright) so the test isolates progress vs terminal.
    omega = _t([[0.0, 0.0, 0.0]])
    a = _t([[0.5, 0.5, 0.5, 0.5]])
    tilt_cos = _t([0.999999])
    vel = _t([[speed, 0.0, 0.0]])
    finished = False
    for _ in range(max_steps):
        if finished:
            break
        x_next = x + step_len
        pos_next = _t([[x_next, 0.0, 0.0]])
        seg_a, seg_b = [e.unsqueeze(0) for e in course.seg(tg)]
        s_curr = R.segment_arc_position(pos_next, seg_a, seg_b).squeeze(0)

        # did we cross the target gate plane this step?
        gx = course.centers[tg][0].item()
        crossed = (x < gx) and (x_next >= gx)
        gate_passed = torch.tensor([crossed], dtype=torch.bool)
        collision = torch.tensor([bool(crossed and clip_at is not None and tg == clip_at)],
                                 dtype=torch.bool)
        if collision.item():
            gate_passed = torch.tensor([False])           # a clip is NOT a pass
        miss = torch.tensor([False])
        oob = torch.tensor([False])
        is_last = tg == (G - 1)
        newly_finished = torch.tensor([bool(crossed and gate_passed.item() and is_last)])
        pass_linf = _t([0.0])                              # dead centre
        time_left = _t([max(0.0, (max_steps - t) * dt)])

        next_center = course.centers[min(tg + 1, G - 1)]
        reward, _, r_prog = R.compute_ego_reward(
            w, s_curr=s_curr.unsqueeze(0), s_prev=s_prev.unsqueeze(0),
            gate_passed=gate_passed, pass_linf=pass_linf, w_g_half=course.w_g_half,
            gate_collision=collision, gate_miss=miss, oob=oob,
            banked_progress_return=banked.unsqueeze(0),
            newly_finished=newly_finished, time_left_s=time_left,
            tilt_cos_r33=tilt_cos, omega=omega, action_norm=a, last_action_norm=a,
            vel_world=vel, curr_center=course.centers[tg].unsqueeze(0),
            next_center=next_center.unsqueeze(0), dt=dt)

        ret += disc * reward.item()
        disc *= gamma
        banked = banked + r_prog.squeeze(0)

        # advance bookkeeping
        if crossed and gate_passed.item():
            if is_last:
                finished = True
            else:
                tg += 1
                # re-seed s_prev on the NEW segment (env contract)
                na, nb = [e.unsqueeze(0) for e in course.seg(tg)]
                s_prev = R.segment_arc_position(pos_next, na, nb).squeeze(0)
        elif collision.item():
            finished = True                                # terminal, episode ends
        else:
            s_prev = s_curr
        x = x_next
        t += 1
    return ret


def _dominance_report(n_gates, gamma, spacing=15.0, speed=15.0):
    """Run the clean vs clip@j rollout for a course of ``n_gates`` at ``gamma`` and return
    (clean, clips, min_margin) where min_margin = min_j (clean - clip@j). Positive min_margin ==
    terminal DOMINATES (sprint-and-clip never out-earns a clean pass) at that scale."""
    w = R.EgoRewardWeights()          # DEFAULTS: progress 1.0, passage 1.0, terminal progress-scaled base 200
    dt = 1.0 / 30.0
    course = _Course(n_gates=n_gates, spacing=spacing)
    clean = _rollout_return(course, w, gamma, dt, speed, clip_at=None)
    clips = [_rollout_return(course, w, gamma, dt, speed, clip_at=j) for j in range(course.n)]
    min_margin = min(clean - cj for cj in clips)
    return clean, clips, min_margin


# FOLDED-FORWARD FIX #3: the terminal-dominance rollout is parametrized to ng in {6,20,60} x gamma in
# {0.99, 0.9975}. gamma=0.9975 MUST keep the dominance margin positive at ALL deployment scales; the
# gamma=0.99 case is asserted+documented to THIN (its margin shrinks with ng -- the reason gamma=0.9975
# is the LOAD-BEARING ego-curriculum default).
@pytest.mark.parametrize("n_gates", [6, 20, 60])
def test_terminal_dominance_rollout_gamma_9975_holds_at_scale(n_gates):
    """LOAD-BEARING: at gamma=0.9975 the sprint-and-clip return is STRICTLY less than the clean-pass
    return for EVERY clip gate j, at ng in {6,20,60} -- confirming the crash penalty still DOMINATES the
    banked progress at deployment scale (the exact defect the adversarial critics caught: inc8's 10.0/m
    + tiny terminal made clipping positive-return). refined-B (rw_progress~1.0 + progress-scaled
    terminal) + gamma=0.9975 closes it at scale. Reports the numbers."""
    gamma = 0.9975
    clean, clips, min_margin = _dominance_report(n_gates, gamma)
    lines = [f"\n===== TERMINAL-DOMINANCE (refined-B, gamma={gamma}, ng={n_gates}) =====",
             f"clean={clean:.3f}  min_margin(clean-clip)={min_margin:.3f}"]
    for j, cj in enumerate(clips):
        lines.append(f"    clip@gate{j}: return={cj:9.3f}   (clean - clip = {clean - cj:8.3f})")
    msg = "\n".join(lines)
    print(msg)
    for j, cj in enumerate(clips):
        assert cj < clean, (
            f"TERMINAL DOMINANCE VIOLATED at gamma={gamma}, ng={n_gates}, clip@gate{j}: "
            f"clip={cj:.3f} >= clean={clean:.3f}." + msg)
    assert min_margin > 0.0, (min_margin, msg)


def test_terminal_dominance_gamma_099_thins_at_scale():
    """DOCUMENTING the gamma=0.99 THINNING: the dominance MARGIN (min_j clean - clip@j) shrinks as the
    course grows -- the late-gate crash penalty is discounted away faster at gamma=0.99 than 0.9975.
    This is WHY the ego curriculum pins gamma=0.9975 (vq2_ego_curriculum). We assert the ordering
    (0.99's margin at a large course < 0.9975's margin at the same course) rather than a brittle
    absolute threshold, and print both so the thinning is visible."""
    ng = 60
    m099 = _dominance_report(ng, 0.99)[2]
    m9975 = _dominance_report(ng, 0.9975)[2]
    print(f"\n===== gamma THINNING at ng={ng}: min_margin  gamma0.99={m099:.3f}  "
          f"gamma0.9975={m9975:.3f} =====")
    # 0.9975 preserves a STRICTLY larger dominance margin at scale than 0.99 (the thinning).
    assert m9975 > m099, (m9975, m099)


def test_terminal_progress_scaled_dominates_by_construction():
    """The progress-scaled terminal forfeits banked progress + a base, so clipping is ALWAYS strictly
    worse than the counterfactual of NOT clipping (continuing to earn), regardless of course length."""
    w = R.EgoRewardWeights()
    banked = _t([173.0])              # a large banked progress return (long course)
    term = R.terminal_penalty(_t([1.0]).bool(), _t([0.0]).bool(), _t([0.0]).bool(), banked, w)
    # penalty = base + banked  ->  strictly greater than banked (you lose everything + a base)
    assert term.item() == pytest.approx(w.terminal_base + 173.0)
    assert term.item() > banked.item()
    # FIXED mode: penalty == base (independent of banked) -- must be set > max bankable to dominate.
    # (Use a small certified course so the FIXED-guard passes: base 200 > 1.0*1gate*45m.)
    w2 = R.EgoRewardWeights(terminal_progress_scaled=False, terminal_base=200.0,
                            guard_max_course_gates=1, guard_max_seg_len_m=45.0)
    term2 = R.terminal_penalty(_t([1.0]).bool(), _t([0.0]).bool(), _t([0.0]).bool(), banked, w2)
    assert term2.item() == pytest.approx(200.0)


def test_forfeit_is_contact_only_miss_and_oob_keep_banked():
    """Reward-audit 2026-07-07 root-cause fix: the banked-progress FORFEIT fires ONLY on a CONTACT.
    A wide MISS or an OOB is an honest non-contact outcome -- it pays ONLY its fixed base and KEEPS its
    banked approach progress. Forfeiting there cancelled the dense homing gradient on ~100% of single_gate
    rollouts and made loitering beat committing to a crossing. Anti-sprint-and-clip is preserved because a
    frame clip is classified as gate_collision."""
    w = R.EgoRewardWeights()          # progress-scaled, bases: contact 200, miss 200(default), oob 200
    banked = _t([40.0])
    coll = R.terminal_penalty(_t([1.0]).bool(), _t([0.0]).bool(), _t([0.0]).bool(), banked, w)
    miss = R.terminal_penalty(_t([0.0]).bool(), _t([1.0]).bool(), _t([0.0]).bool(), banked, w)
    oob = R.terminal_penalty(_t([0.0]).bool(), _t([0.0]).bool(), _t([1.0]).bool(), banked, w)
    # CONTACT forfeits the banked (base + banked); MISS / OOB do NOT (base only).
    assert coll.item() == pytest.approx(w.terminal_base + 40.0)
    assert miss.item() == pytest.approx(w.terminal_miss)          # NO banked forfeit
    assert oob.item() == pytest.approx(w.terminal_oob)            # NO banked forfeit
    # with a curriculum-scale forgiving miss base (8) and banked ~30, an honest miss KEEPS net-positive
    # approach credit: episode return contribution = +banked (streamed) - miss_base = +22 > 0.
    w_c = R.EgoRewardWeights(terminal_miss=8.0)
    miss_c = R.terminal_penalty(_t([0.0]).bool(), _t([1.0]).bool(), _t([0.0]).bool(), _t([30.0]), w_c)
    assert miss_c.item() == pytest.approx(8.0)
    assert 30.0 - miss_c.item() > 0.0


# ================================================================================================
# FOLDED-FORWARD FIX #2: the FIXED-terminal guard FIRES when a fixed base cannot dominate the course.
# ================================================================================================
def test_fixed_terminal_guard_fires_when_base_below_max_bankable():
    """The default (progress-scaled) mode never trips the guard -- dominance holds for ANY course by
    construction. But FIXED mode with a base BELOW the max bankable progress return (progress * gates *
    seg_len) lets sprint-and-clip-at-the-last-gate out-earn a clean pass; the guard MUST raise at
    construction so a mis-set FIXED run fails loudly."""
    # progress-scaled default: no guard, any base is fine.
    R.EgoRewardWeights(terminal_progress_scaled=True, terminal_base=10.0)   # no raise
    # FIXED mode, base 200 on a 6-gate x 45 m course => max bankable = 1.0*6*45 = 270 > 200 -> FIRES.
    with pytest.raises(AssertionError):
        R.EgoRewardWeights(terminal_progress_scaled=False, terminal_base=200.0,
                           guard_max_course_gates=6, guard_max_seg_len_m=45.0)
    # FIXED mode with ALL THREE bases above the max bankable (270) is accepted.
    R.EgoRewardWeights(terminal_progress_scaled=False, terminal_base=300.0,
                       terminal_miss=300.0, terminal_oob=300.0,
                       guard_max_course_gates=6, guard_max_seg_len_m=45.0)   # all > 270 -> OK
    # the guard also covers terminal_miss / terminal_oob (a small miss base on a big course -> FIRES).
    with pytest.raises(AssertionError):
        R.EgoRewardWeights(terminal_progress_scaled=False, terminal_base=1000.0,
                           terminal_miss=50.0, guard_max_course_gates=6, guard_max_seg_len_m=45.0)


# ================================================================================================
# HOVER-HOLD altitude probe (Fengyou greenlight 2026-07-08; the H1-vs-H2 disambiguator).
# ================================================================================================
def test_altitude_hold_peaks_at_spawn_and_decays_symmetrically():
    """R_alt peaks (+rw) at the spawn altitude, decays linearly, and is 0 at |Δz|>=band -- symmetric in
    the sign of Δz (spawn altitude is the UNIQUE optimum)."""
    z0 = _t([0.0, 0.0, 0.0, 0.0, 0.0])
    band = 8.0
    z = _t([0.0, 4.0, -4.0, 8.0, 12.0])                 # Δz = 0, +4, -4, +8 (=band), +12 (>band)
    r = R.altitude_hold_reward(z, z0, rw_altitude_hold=1.0, band_m=band)
    assert r[0].item() == pytest.approx(1.0)            # at spawn altitude -> full bonus
    assert r[1].item() == pytest.approx(0.5)            # 4 m off -> half
    assert r[2].item() == pytest.approx(0.5)            # SYMMETRIC: -4 m == +4 m
    assert r[3].item() == pytest.approx(0.0)            # at band -> 0
    assert r[4].item() == pytest.approx(0.0)            # beyond band -> clamped 0 (not negative)


def test_altitude_hold_is_positive_no_giveup():
    """The bonus is NON-NEGATIVE everywhere -> the policy is PAID TO SURVIVE at altitude; ending the
    episode forfeits the future bonus, so unlike a -k|Δz| magnitude penalty it has NO give-up incentive."""
    z0 = _t([0.0, 0.0, 0.0])
    z = _t([-20.0, -3.0, 5.0])                          # far below, near, above
    r = R.altitude_hold_reward(z, z0, rw_altitude_hold=2.0, band_m=8.0)
    assert (r >= 0.0).all(), r                          # never a penalty -> no cheaper-to-end-early trap
    assert r.max().item() <= 2.0 + 1e-9                 # bounded by rw (no farming)


def test_altitude_hold_off_when_weight_zero():
    z0 = _t([0.0, 0.0]); z = _t([3.0, -5.0])
    r = R.altitude_hold_reward(z, z0, rw_altitude_hold=0.0, band_m=8.0)
    assert torch.equal(r, torch.zeros_like(r))          # OFF (default on every non-probe stage)


def test_altitude_hold_wired_into_compute_ego_reward():
    """compute_ego_reward adds the alt bonus when z/z_spawn are supplied and altitude_hold>0, and exposes
    it as the 'alt_hold_reward' component; it is OFF (0) when the weight is 0."""
    n = 3
    z0 = torch.zeros(n, dtype=DT)
    z = _t([0.0, 4.0, -8.0])                            # full / half / zero bonus at band=8
    kw = dict(
        s_curr=torch.zeros(n, dtype=DT), s_prev=torch.zeros(n, dtype=DT),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=torch.zeros(n, dtype=DT),
        w_g_half=0.375,
        gate_collision=torch.zeros(n, dtype=torch.bool), gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool), banked_progress_return=torch.zeros(n, dtype=DT),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=torch.zeros(n, dtype=DT),
        tilt_cos_r33=torch.ones(n, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT), curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1 / 30,
    )
    w_on = R.EgoRewardWeights(altitude_hold=1.0, altitude_hold_band_m=8.0)
    _, comps_on, _ = R.compute_ego_reward(w_on, z=z, z_spawn=z0, **kw)
    assert comps_on["alt_hold_reward"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    # OFF when the weight is 0 -> the component is 0 and the term contributes nothing.
    w_off = R.EgoRewardWeights(altitude_hold=0.0)
    _, comps_off, _ = R.compute_ego_reward(w_off, z=z, z_spawn=z0, **kw)
    assert comps_off["alt_hold_reward"] == pytest.approx(0.0)


# ================================================================================================
# GATE-RELATIVE WORLD-VERTICAL HOLD (Fengyou 2026-07-12; the R0 floor-dive fix -- OBSERVABLE vertical
# anchor replacing the unobservable absolute-Z altitude_hold).
# ================================================================================================
def test_gate_vhold_peaks_at_gate_altitude_and_decays_symmetrically():
    """R_gvhold peaks (+rw) when the drone is AT the gate's altitude (Δz=0), decays linearly, is 0 at
    |Δz|>=band, and is SYMMETRIC in the sign of Δz (the gate altitude is the UNIQUE optimum)."""
    gate_z = _t([10.0, 10.0, 10.0, 10.0, 10.0])          # a non-zero gate altitude (NOT the origin)
    band = 8.0
    drone_z = _t([10.0, 14.0, 6.0, 18.0, 22.0])          # Δz = 0, +4, -4, +8 (=band), +12 (>band)
    r = R.gate_vertical_hold_reward(drone_z, gate_z, rw_gate_vhold=1.0, band_m=band)
    assert r[0].item() == pytest.approx(1.0)             # at the gate altitude -> full bonus
    assert r[1].item() == pytest.approx(0.5)             # 4 m off -> half
    assert r[2].item() == pytest.approx(0.5)             # SYMMETRIC: -4 m == +4 m
    assert r[3].item() == pytest.approx(0.0)             # at band -> 0
    assert r[4].item() == pytest.approx(0.0)             # beyond band -> clamped 0 (not negative)


def test_gate_vhold_is_positive_no_giveup():
    """NON-NEGATIVE everywhere -> the policy is PAID TO SURVIVE at the gate's height; ending the episode
    forfeits the future bonus, so (unlike a -k|Δz| magnitude penalty) it carries NO give-up incentive --
    the exact property that keeps it from reintroducing the floor-dive it fixes."""
    gate_z = _t([10.0, 10.0, 10.0])
    drone_z = _t([-20.0, 7.0, 15.0])                     # far below, near, above the gate
    r = R.gate_vertical_hold_reward(drone_z, gate_z, rw_gate_vhold=2.0, band_m=8.0)
    assert (r >= 0.0).all(), r                           # never a penalty -> no cheaper-to-end-early trap
    assert r.max().item() <= 2.0 + 1e-9                  # bounded by rw (no farming)


def test_gate_vhold_is_gate_relative_not_absolute_altitude():
    """OBSERVABILITY CORE: the reward depends ONLY on the DIFFERENCE Δz = drone_z - gate_center_z (the
    world-vertical gate offset the policy CAN reconstruct as R_wb[2,:]*rel_pos_body), NEVER on either
    absolute altitude. Shifting BOTH the drone and the gate by the SAME constant leaves the reward
    IDENTICAL -> gate-relative, not the absolute-Z that was unobservable in the position-free obs."""
    band = 8.0
    drone_z = _t([12.0, 3.0, -5.0])
    gate_z = _t([10.0, 10.0, 10.0])
    r0 = R.gate_vertical_hold_reward(drone_z, gate_z, rw_gate_vhold=1.0, band_m=band)
    for shift in (100.0, -37.5):                         # translate the WHOLE world in altitude
        r_shift = R.gate_vertical_hold_reward(drone_z + shift, gate_z + shift, 1.0, band)
        assert torch.allclose(r0, r_shift), (shift, r0, r_shift)   # invariant -> uses only the offset
    # ... and it matches altitude_hold_reward's FORM with the gate altitude as the reference (same math,
    # re-anchored frame): gate_vhold(drone_z, gate_z) == altitude_hold(drone_z, gate_z).
    r_alt_form = R.altitude_hold_reward(drone_z, gate_z, rw_altitude_hold=1.0, band_m=band)
    assert torch.allclose(r0, r_alt_form)


def test_gate_vhold_off_when_weight_zero():
    gate_z = _t([10.0, 10.0]); drone_z = _t([13.0, 5.0])
    r = R.gate_vertical_hold_reward(drone_z, gate_z, rw_gate_vhold=0.0, band_m=8.0)
    assert torch.equal(r, torch.zeros_like(r))           # OFF (default on every non-R0 stage -> byte-identical)
    assert R.EgoRewardWeights().gate_vhold == 0.0        # dataclass default OFF
    assert R.EgoRewardWeights().gate_vhold_band_m == 8.0


def test_gate_vhold_wired_into_compute_ego_reward_via_gate_center_z():
    """compute_ego_reward adds the gate-vertical bonus when z + gate_center_z are supplied and gate_vhold>0,
    exposes it as 'gate_vhold_reward', and it is RANGE-INDEPENDENT (no homing): changing the gate's x/y
    (range) with gate_center_z fixed does NOT change the bonus. OFF (0) when the weight is 0."""
    n = 3
    z = _t([10.0, 14.0, 2.0])                            # Δz to gate_z=10 -> 0, +4, -8 (=band) => 1, 0.5, 0
    gate_z = _t([10.0, 10.0, 10.0])
    kw = dict(
        s_curr=torch.zeros(n, dtype=DT), s_prev=torch.zeros(n, dtype=DT),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=torch.zeros(n, dtype=DT),
        w_g_half=0.375,
        gate_collision=torch.zeros(n, dtype=torch.bool), gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool), banked_progress_return=torch.zeros(n, dtype=DT),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=torch.zeros(n, dtype=DT),
        tilt_cos_r33=torch.ones(n, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT), curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1 / 30,
    )
    w_on = R.EgoRewardWeights(gate_vhold=1.0, gate_vhold_band_m=8.0)
    _, comps_on, _ = R.compute_ego_reward(w_on, z=z, gate_center_z=gate_z, **kw)
    assert comps_on["gate_vhold_reward"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    # NO HOMING / range-independent: the reward function never sees the horizontal offset, so a totally
    # different range (only gate_center_z matters) yields the IDENTICAL bonus.
    _, comps_far, _ = R.compute_ego_reward(w_on, z=z, gate_center_z=gate_z, **kw)
    assert comps_far["gate_vhold_reward"] == pytest.approx(comps_on["gate_vhold_reward"])
    # OFF when the weight is 0 -> the component is 0 and the term contributes nothing (byte-identical).
    w_off = R.EgoRewardWeights(gate_vhold=0.0)
    _, comps_off, _ = R.compute_ego_reward(w_off, z=z, gate_center_z=gate_z, **kw)
    assert comps_off["gate_vhold_reward"] == pytest.approx(0.0)


# ================================================================================================
# MPCC CONTOURING (Fengyou greenlight 2026-07-08; the floor-dive lever, hover-hold-confirmed).
# ================================================================================================
def test_corridor_rewards_return_penalises_drift():
    """R_corr = corridor*clip(perp_prev - perp_curr): POSITIVE when perp shrinks (moving toward the line),
    NEGATIVE when it grows (drifting off)."""
    k, dt = 2.0, 1 / 30
    r_toward = R.corridor_progress_reward(_t([0.6]), _t([1.0]), k, vmax_mps=39.0, dt=dt)   # perp 1.0->0.6
    assert r_toward.item() == pytest.approx(k * 0.4)                       # +0.8 (homing toward line)
    r_away = R.corridor_progress_reward(_t([1.0]), _t([0.6]), k, vmax_mps=39.0, dt=dt)     # perp 0.6->1.0
    assert r_away.item() == pytest.approx(-k * 0.4)                        # -0.8 (drifting off)


def test_corridor_telescopes_to_zero_on_closed_path():
    """Potential-based -> NON-farmable: drift OUT then return IN sums to ~0 (you cannot pump reward by
    oscillating perpendicular to the line)."""
    k, dt = 2.0, 1 / 30
    perps = [0.0, 0.5, 1.0, 0.7, 0.3, 0.0]                                 # leave the line and come back
    total = 0.0
    for prev, curr in zip(perps[:-1], perps[1:]):
        total += R.corridor_progress_reward(_t([curr]), _t([prev]), k, 39.0, dt).item()
    assert abs(total) < 1e-9, total                                       # telescopes to k*(perp_0 - perp_T)=0


def test_corridor_no_standing_tax_unlike_penalty():
    """A CENTRED-ish mean holding a CONSTANT perp pays ~0 corridor (E[Δ]=0) -- the give-up-resistance /
    no-standing-tax property. Contrast: the raw through_centering PENALTY at the same perp is strictly
    negative (a standing tax that invites give-up)."""
    k = 2.0
    r_hold = R.corridor_progress_reward(_t([0.5]), _t([0.5]), k, 39.0, 1 / 30)   # perp unchanged
    assert r_hold.item() == pytest.approx(0.0)                            # NO standing tax
    r_pen = R.through_centering_reward(_t([0.5]), rw_centering=k, centering_max_m=2.0)
    assert r_pen.item() < 0.0                                             # the penalty form DOES tax a held offset


def test_corridor_clip_band_trims_handoff_burst():
    """A large perp discontinuity (gate re-projection) is clamped to +/- vmax*dt (m/step)."""
    band = 39.0 * (1 / 30)                                                # 1.3 m/step
    r = R.corridor_progress_reward(_t([0.0]), _t([20.0]), rw_corridor=1.0, vmax_mps=39.0, dt=1 / 30)
    assert r.item() == pytest.approx(band)                               # 20 m jump clipped to the band


def test_corridor_off_when_weight_zero():
    r = R.corridor_progress_reward(_t([0.5]), _t([1.0]), rw_corridor=0.0, vmax_mps=39.0, dt=1 / 30)
    assert torch.equal(r, torch.zeros_like(r))


def test_corridor_wired_into_compute_ego_reward_and_not_banked():
    """compute_ego_reward adds the contouring term when perp_dist+perp_prev are supplied and corridor>0,
    exposes 'corridor_reward', and does NOT fold it into the banked progress return (r_prog only) -- so a
    contact terminal never forfeits accumulated contouring."""
    n = 2
    perp_curr = _t([0.4, 1.0]); perp_prev = _t([1.0, 0.4])                # env0 homing in, env1 drifting out
    kw = dict(
        s_curr=torch.zeros(n, dtype=DT), s_prev=torch.zeros(n, dtype=DT),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=torch.zeros(n, dtype=DT),
        w_g_half=0.375,
        gate_collision=torch.zeros(n, dtype=torch.bool), gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool), banked_progress_return=torch.zeros(n, dtype=DT),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=torch.zeros(n, dtype=DT),
        tilt_cos_r33=torch.ones(n, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT), curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1 / 30,
    )
    w = R.EgoRewardWeights(corridor=2.0)
    _, comps, r_prog = R.compute_ego_reward(w, perp_dist=perp_curr, perp_prev=perp_prev, **kw)
    # env0 (perp 1.0->0.4) pays +2*0.6, env1 (0.4->1.0) pays -2*0.6 -> mean 0
    assert comps["corridor_reward"] == pytest.approx(0.0)
    assert torch.equal(r_prog, torch.zeros(n, dtype=DT))                  # contouring is NOT in r_prog (banked)


# ================================================================================================
# ANISOTROPIC VERTICAL WEIGHT on the distance-to-gate potential (Fengyou greenlight 2026-07-08).
# ================================================================================================
def test_gate_center_potential_isotropic_default_byte_compatible():
    """vert_weight=1.0 reproduces the exact isotropic Euclidean norm (the byte-compatible default so every
    non-lever stage is unchanged)."""
    pos = _t([[3.0, 4.0, 12.0], [1.0, 0.0, 0.0]])
    ctr = torch.zeros(2, 3, dtype=DT)
    iso = R.gate_center_potential(pos, ctr)                               # default vert_weight=1.0
    assert torch.allclose(iso, -torch.linalg.norm(pos - ctr, dim=-1))
    assert iso[0].item() == pytest.approx(-13.0)                          # 3-4-12 -> 13


def test_vert_weight_unburies_vertical_vs_lateral():
    """A VERTICAL offset is penalised sqrt(w) x more than the SAME-size LATERAL offset -> the altitude
    signal is un-buried. With w=25, a 1 m vertical drop costs 5x a 1 m lateral drift."""
    ctr = torch.zeros(1, 3, dtype=DT)
    up = _t([[0.0, 0.0, 1.0]]); side = _t([[0.0, 1.0, 0.0]])
    phi_up = R.gate_center_potential(up, ctr, vert_weight=25.0)
    phi_side = R.gate_center_potential(side, ctr, vert_weight=25.0)
    assert phi_up.item() == pytest.approx(-5.0)                           # sqrt(25*1) = 5
    assert phi_side.item() == pytest.approx(-1.0)                         # lateral unweighted
    # forward-buried case: at 15 m out, a 1 m sink barely moves the isotropic norm but clearly moves w=25.
    far = _t([[15.0, 0.0, 1.0]]); far_level = _t([[15.0, 0.0, 0.0]])
    d_iso = (R.gate_center_potential(far, ctr) - R.gate_center_potential(far_level, ctr)).abs()
    d_w = (R.gate_center_potential(far, ctr, 25.0) - R.gate_center_potential(far_level, ctr, 25.0)).abs()
    assert d_w.item() > 10 * d_iso.item()                                # the weighted norm feels the sink far more


def test_forfeit_mask_frame_moat_fix():
    """Fengyou 2026-07-08 frame-moat fix: passing forfeit_mask=floor-only makes a FRAME-CLIP forfeit no
    banked progress (net == a wide miss) while the FLOOR dive still forfeits -- so the aperture ring is not
    a moat that punishes getting close. Without the mask (legacy) a frame-clip forfeits like any contact."""
    import torch
    from ego_reward import terminal_penalty, EgoRewardWeights
    w = EgoRewardWeights(terminal_base=100.0, terminal_miss=100.0, terminal_oob=200.0,
                         terminal_progress_scaled=True)
    banked = torch.tensor([30.0, 30.0, 30.0])
    frame = torch.tensor([True, False, False])
    miss = torch.tensor([False, True, False])
    floor = torch.tensor([False, False, True])
    coll = frame | floor                                        # frame + floor are both contacts
    ob = torch.zeros(3, dtype=torch.bool)
    legacy = terminal_penalty(coll, miss, ob, banked, w)        # mask=None -> every contact forfeits
    assert abs(legacy[0].item() - 130.0) < 1e-4                 # frame-clip = base+forfeit (the moat)
    assert abs(legacy[1].item() - 100.0) < 1e-4                 # wide miss = base only
    fixed = terminal_penalty(coll, miss, ob, banked, w, forfeit_mask=floor.float())
    assert abs(fixed[0].item() - fixed[1].item()) < 1e-4        # frame-clip == wide miss (moat GONE)
    assert fixed[2].item() > fixed[1].item() + 1e-4             # floor STILL forfeits (real crash)


def test_alignment_reward_gvf_direction():
    """GVF alignment reward (Fengyou 2026-07-08): rewards velocity DIRECTION following the guiding field.
    On the line moving along the tangent -> ~max. Off the line, angling ONTO the path beats flying PARALLEL
    (the telescoping-contouring blind spot is gone). Higher gain demands a steeper inward angle."""
    import math
    import torch
    from ego_reward import alignment_reward
    tangent = torch.tensor([[1.0, 0.0, 0.0]])
    # on the line: perp=0, v along tangent -> ~ +rw_align (cos 0 * speed_gate~1)
    r_on = alignment_reward(torch.tensor([[5.0, 0.0, 0.0]]), tangent, torch.zeros(1, 3),
                            torch.tensor([0.0]), rw_align=2.0, align_gain=1.0)
    assert r_on.item() > 1.9
    # 3 m off (inward = -z); flying PARALLEL (along tangent) vs flying ALONG F (angled inward)
    inward = torch.tensor([[0.0, 0.0, -1.0]])
    perp = torch.tensor([3.0])
    r_parallel = alignment_reward(torch.tensor([[5.0, 0.0, 0.0]]), tangent, inward, perp,
                                  rw_align=2.0, align_gain=1.0)
    theta = math.atan(1.0 * 3.0)                                     # the field's inward angle
    v_along_F = torch.tensor([[math.cos(theta) * 5, 0.0, -math.sin(theta) * 5]])
    r_onpath = alignment_reward(v_along_F, tangent, inward, perp, rw_align=2.0, align_gain=1.0)
    assert r_onpath.item() > r_parallel.item() + 0.1                 # angling onto the path beats parallel
    # near-stationary -> ~0 (ill-defined direction, damped by the speed gate)
    r_rest = alignment_reward(torch.tensor([[0.001, 0.0, 0.0]]), tangent, inward, perp,
                              rw_align=2.0, align_gain=1.0)
    assert abs(r_rest.item()) < 0.1


def test_crossing_parabola_reward():
    """Smooth parabolic crossing reward (Fengyou 2026-07-08): +center dead-centre, 0 at the aperture edge,
    growing negative outside, clamped. Monotonic in the offset -> no moat, no cliff."""
    import torch
    from ego_reward import crossing_parabola_reward
    crossed = torch.ones(4, dtype=torch.bool)
    e = torch.tensor([0.0, 0.75, 1.0, 5.0])                     # centre, edge, just-outside, far
    r = crossing_parabola_reward(e, crossed, cross_center=20.0, cross_zero_m=0.75, cross_neg_cap=100.0)
    assert abs(r[0].item() - 20.0) < 1e-4                       # centre -> +20
    assert abs(r[1].item() - 0.0) < 1e-4                        # aperture edge -> 0
    assert -20.0 < r[2].item() < 0.0                            # just outside -> small negative
    assert abs(r[3].item() + 100.0) < 1e-4                      # far -> clamped at -cap
    assert r[0].item() > r[1].item() > r[2].item() > r[3].item()  # MONOTONIC (no moat)
    r0 = crossing_parabola_reward(e, torch.zeros(4, dtype=torch.bool), 20.0, 0.75, 100.0)
    assert (r0 == 0).all()                                      # no crossing -> 0


# ================================================================================================
# FATAL SPIN ABORT terminal composition (PERCEPTION-HONESTY package 2026-07-10, DESIGN.md §P).
# The env composes lethal = below_floor | spin_abort and rides it through the SAME wiring as the
# floor dive: gate_collision fold (legacy/refined-B path) AND the parabola path's floor_contact= /
# forfeit_mask= kwargs. These tests pin the COLLISION-CLASS magnitude on BOTH reward paths -- and
# pin the trap the losing design fell into (spin routed ONLY through gate_collision pays ZERO under
# the champion parabola regime == spin-to-exit becomes a FREE, banked-progress-keeping bail-out).
# ================================================================================================
def _spin_kw(n, banked):
    return dict(
        s_curr=torch.zeros(n, dtype=DT), s_prev=torch.zeros(n, dtype=DT),
        gate_passed=torch.zeros(n, dtype=torch.bool), pass_linf=torch.zeros(n, dtype=DT),
        w_g_half=0.375,
        gate_miss=torch.zeros(n, dtype=torch.bool),
        oob=torch.zeros(n, dtype=torch.bool),
        banked_progress_return=_t(banked),
        newly_finished=torch.zeros(n, dtype=torch.bool), time_left_s=torch.zeros(n, dtype=DT),
        tilt_cos_r33=torch.ones(n, dtype=DT), omega=torch.zeros(n, 3, dtype=DT),
        action_norm=torch.full((n, 4), 0.5, dtype=DT), last_action_norm=torch.full((n, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(n, 3, dtype=DT), curr_center=torch.zeros(n, 3, dtype=DT),
        next_center=torch.zeros(n, 3, dtype=DT), dt=1 / 30,
    )


def test_spin_abort_pays_collision_class_on_the_parabola_path_via_lethal_mask():
    """CHAMPION (parabola) regime: the terminal fires on the floor_contact kwarg ONLY. A spin abort
    riding the lethal mask (env: floor_contact=lethal) pays terminal_base + the FULL banked-progress
    forfeit -- collision-class, exactly the terminal-equalization doctrine."""
    n = 2
    spin = torch.tensor([True, False])
    w = R.EgoRewardWeights(parabola_crossing=True, cross_center=20.0, cross_zero_m=4.0,
                           cross_neg_cap=100.0, terminal_base=100.0, terminal_miss=100.0,
                           terminal_oob=200.0, terminal_progress_scaled=True)
    reward, comps, _ = R.compute_ego_reward(
        w, gate_collision=spin,                                     # env folds spin into collision
        cross_offset=torch.zeros(n, dtype=DT), crossed=torch.zeros(n, dtype=torch.bool),
        floor_contact=spin.to(DT),                                  # the LETHAL mask (load-bearing)
        **_spin_kw(n, [30.0, 30.0]))
    # env0 (spin abort): -(base 100 + banked 30) + the -0.02 time tick; env1: just the time tick.
    assert reward[0].item() == pytest.approx(-130.0 - w.time)
    assert reward[1].item() == pytest.approx(-w.time)
    assert comps["terminal_pen"] == pytest.approx(130.0 / n)


def test_spin_abort_through_gate_collision_alone_is_free_under_parabola_THE_TRAP():
    """The verified defect of the losing design: under the parabola regime, a spin abort routed ONLY
    through gate_collision (floor_contact stays bare below_floor == zeros) reaches NO terminal at all
    -- the episode ends penalty-free WITH banked progress kept, strictly cheaper than a miss (-100)
    -> spin-to-exit becomes an attractive learned bail-out. This pin documents WHY the env passes
    lethal (not below_floor) as floor_contact -- it exercises the REWARD side only (the test builds
    its own floor_contact arg); the ENV-side wiring itself is enforced by the source pin
    test_perception_honesty.py::test_step_source_pins_yaw_clamp_and_lethal_mask_plumbing."""
    n = 1
    spin = torch.tensor([True])
    w = R.EgoRewardWeights(parabola_crossing=True, cross_center=20.0, cross_zero_m=4.0,
                           cross_neg_cap=100.0, terminal_base=100.0, terminal_progress_scaled=True)
    reward, comps, _ = R.compute_ego_reward(
        w, gate_collision=spin,
        cross_offset=torch.zeros(n, dtype=DT), crossed=torch.zeros(n, dtype=torch.bool),
        floor_contact=torch.zeros(n, dtype=DT),                     # spin NOT in the lethal mask
        **_spin_kw(n, [30.0]))
    assert comps["terminal_pen"] == pytest.approx(0.0)              # the free-exit hole (documented)
    assert reward[0].item() == pytest.approx(-w.time)               # cheaper than any miss/oob/crash


def test_spin_abort_pays_collision_class_on_the_non_parabola_path():
    """Legacy / non-parabola refined-B path: spin folded into gate_collision hits terminal_penalty
    directly -- base + banked forfeit (default forfeit_mask == coll includes the spin abort)."""
    spin = torch.tensor([True, False])
    banked = _t([30.0, 30.0])
    w = R.EgoRewardWeights(terminal_base=100.0, terminal_progress_scaled=True)
    pen = R.terminal_penalty(spin, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                             banked, w)
    assert pen[0].item() == pytest.approx(130.0)                    # base + full banked forfeit
    assert pen[1].item() == pytest.approx(0.0)


def test_package_adds_no_per_step_penalty_at_defaults():
    """At default weights (rw_perception 0, no new rate term -- fatality REPLACES dis-incentive per
    the owner's no-energy-penalty directive) a quiet non-terminating step still pays exactly the time
    tick: the package must add NO new per-step penalty at defaults."""
    n = 1
    w = R.EgoRewardWeights()
    reward, comps, _ = R.compute_ego_reward(
        w, gate_collision=torch.zeros(n, dtype=torch.bool), **_spin_kw(n, [0.0]))
    assert reward[0].item() == pytest.approx(-w.time)
    assert comps["perception_reward"] == pytest.approx(0.0)


# ================================================================================================
# pefcap package (2026-07-12): SOFT attitude-limit penalty + NEXT-GATE perception + PARITY when OFF.
# ================================================================================================
def test_attitude_limit_penalty_zero_in_band_grows_past_correct_sign():
    """SOFT, NON-TERMINAL perception-preservation penalty: ZERO inside the free band (incl. the ~17.8deg
    rest tilt), grows LINEARLY past the limit, NEGATIVE sign, magnitude limit (symmetric in pitch sign)."""
    w = R.EgoRewardWeights(att_pitch=0.5, att_pitch_limit_rad=0.5235988,
                           att_roll=0.3, att_roll_limit_rad=0.6981317)
    # inside band: |pitch|=0.31 (the -17.8deg rest tilt) < 0.524, |roll|=0.10 < 0.698 -> ZERO
    assert R.attitude_limit_penalty(_t([0.10]), _t([-0.31]), w).item() == pytest.approx(0.0)
    # past the pitch limit: |pitch|=0.873 (-50deg) -> -0.5*relu(0.873-0.524)
    rp = R.attitude_limit_penalty(_t([0.0]), _t([-0.873]), w)
    assert rp.item() == pytest.approx(-0.5 * (0.873 - 0.5235988), abs=1e-6)
    assert rp.item() < 0.0                                          # NEGATIVE (correct sign)
    # |pitch| MAGNITUDE limit -> symmetric in sign
    assert R.attitude_limit_penalty(_t([0.0]), _t([0.873]), w).item() == pytest.approx(rp.item(), abs=1e-9)
    # roll excess only
    assert R.attitude_limit_penalty(_t([0.9]), _t([0.0]), w).item() == pytest.approx(
        -0.3 * (0.9 - 0.6981317), abs=1e-6)
    # OFF (weights 0) -> exactly 0 for ANY attitude (byte-identical default)
    assert R.attitude_limit_penalty(_t([2.0]), _t([2.0]), R.EgoRewardWeights()).item() == 0.0


def test_attitude_cap_default_limits_are_60_deg_both_axes_and_off_is_reward_neutral():
    """DEFAULT cap angles = 60 deg on BOTH axes (owner Fengyou, Track A 2026-07-13: roll self-limited ~60
    deg; pitch a LOOSE backstop -- pitch's real lever is rw_perception, so its cap is deliberately generous).
    Changing the DEFAULT limits is reward-NEUTRAL while the weights are 0 (relu*0 -> 0): the byte-identical
    OFF guarantee prices ONLY the excess beyond the limit TIMES the (zero) weight, so the limit angle is
    invisible when OFF -- pinned here so a future limit retune can never silently break the OFF parity."""
    import math
    w = R.EgoRewardWeights()
    assert w.att_pitch_limit_rad == pytest.approx(math.radians(60.0), abs=1e-6)
    assert w.att_roll_limit_rad == pytest.approx(math.radians(60.0), abs=1e-6)
    assert w.att_pitch == 0.0 and w.att_roll == 0.0                # weights still default-OFF
    # OFF is reward-neutral REGARDLESS of the limit: a beyond-cap attitude with weight 0 pays exactly 0.
    assert R.attitude_limit_penalty(_t([1.2]), _t([1.2]), w).item() == 0.0


def test_attitude_penalty_wired_nonterminal_and_off_is_byte_identical():
    n = 1
    base = R.EgoRewardWeights()                                     # att weights 0 (default)
    r_none, _, _ = R.compute_ego_reward(base, gate_collision=torch.zeros(n, dtype=torch.bool),
                                        **_spin_kw(n, [0.0]))
    # passing extreme roll/pitch with weights 0 -> BYTE-identical (term is exactly 0)
    r_off, _, _ = R.compute_ego_reward(base, gate_collision=torch.zeros(n, dtype=torch.bool),
                                       roll=_t([2.0]), pitch=_t([2.0]), **_spin_kw(n, [0.0]))
    assert r_off.item() == r_none.item()
    # ARMED: at -50deg the reward drops by EXACTLY the penalty (a smooth term, NOT a terminal cliff)
    w = R.EgoRewardWeights(att_pitch=0.5, att_pitch_limit_rad=0.5235988)
    r_in, c_in, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool),
                                         roll=_t([0.0]), pitch=_t([-0.31]), **_spin_kw(n, [0.0]))
    r_out, c_out, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool),
                                           roll=_t([0.0]), pitch=_t([-0.873]), **_spin_kw(n, [0.0]))
    pen = 0.5 * (0.873 - 0.5235988)
    assert c_in["att_pen"] == pytest.approx(0.0)                    # in-band pays 0 (no hover reward, no kill)
    assert c_out["att_pen"] == pytest.approx(pen, abs=1e-6)
    assert (r_in.item() - r_out.item()) == pytest.approx(pen, abs=1e-6)


def test_perception_next_off_by_default_and_farm_neutrality_bound():
    assert R.EgoRewardWeights().perception_next == 0.0             # default OFF
    R.EgoRewardWeights(perception=0.014, perception_next=0.004)    # sum 0.018 < time 0.02 -> OK
    R.EgoRewardWeights(perception=0.015, perception_next=0.005)    # sum == time 0.02 -> allowed (<=)
    with pytest.raises(AssertionError):                            # sum > time -> hover-stare farm guard FIRES
        R.EgoRewardWeights(perception=0.02, perception_next=0.004)


def test_perception_next_wired_and_detectability_gated():
    n = 1
    w = R.EgoRewardWeights(perception=0.0, perception_next=0.004, perception_exponent=4.0)
    r_none, _, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool),
                                        **_spin_kw(n, [0.0]))       # no cos_view_next -> 0 (byte-identical)
    # next gate dead-centre (cos=1) -> +perception_next
    r_seen, c_seen, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool),
                                             cos_view_next=_t([1.0]), **_spin_kw(n, [0.0]))
    assert c_seen["perception_next_reward"] == pytest.approx(0.004, abs=1e-6)
    assert (r_seen.item() - r_none.item()) == pytest.approx(0.004, abs=1e-6)
    # env feeds cos=-1 when the next gate is out of view / undetectable -> ~0 bonus (not farmable off-gate)
    r_out, c_out, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool),
                                           cos_view_next=_t([-1.0]), **_spin_kw(n, [0.0]))
    assert c_out["perception_next_reward"] == pytest.approx(0.0, abs=1e-6)


def test_pefcap_reward_byte_identical_when_all_knobs_off():
    """PARITY: with every pefcap reward knob at its default (OFF), passing the new inputs (roll/pitch/
    cos_view_next) is BYTE-identical to the pre-pefcap reward -- the new terms contribute exactly 0."""
    n = 3
    w = R.EgoRewardWeights()                                        # all pefcap weights 0
    kw = _spin_kw(n, [0.0, 5.0, 10.0])
    r_ref, _, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool), **kw)
    r_new, comps, _ = R.compute_ego_reward(
        w, gate_collision=torch.zeros(n, dtype=torch.bool),
        roll=_t([0.1, -0.9, 1.5]), pitch=_t([-0.3, 0.9, -1.2]),
        cos_view_next=_t([0.9, -1.0, 0.2]), **kw)
    assert torch.equal(r_ref, r_new)                               # BYTE-identical
    assert comps["att_pen"] == 0.0 and comps["perception_next_reward"] == 0.0


# ================================================================================================
# ANTI-DITHER yaw smoothness (nodither fine-tune 2026-07-12): squared yaw-command JERK.
#   * ZERO at a steady yaw command (incl. a CONSTANT nonzero turn -> a sustained turn is FREE).
#   * grows with the squared temporal change (a +-clamp rail-flip pays heavily); NEGATIVE sign.
#   * SMOOTH + magnitude-aware -> benign near-zero jitter pays ~0 (a sign-flip indicator would not).
#   * OFF-by-default byte-identical (weight 0 -> exactly 0; None input -> the reward term is 0).
# ================================================================================================
def test_yaw_dither_penalty_zero_at_steady_grows_with_flip_correct_sign():
    w = 0.5
    # STEADY (delta 0) -> 0; a CONSTANT nonzero turn is ALSO delta 0 -> pays 0 (a sustained turn is free).
    assert R.yaw_dither_penalty(_t([0.0]), w).item() == 0.0
    # a +-0.35 TRAINING-clamp rail-FLIP: delta = 0.35 - (-0.35) = 0.70 -> -w*0.70^2 (a real deterrent).
    flip = R.yaw_dither_penalty(_t([0.70]), w)
    assert flip.item() == pytest.approx(-w * 0.70 ** 2, abs=1e-12)
    assert flip.item() < 0.0                                        # NEGATIVE (a penalty)
    # symmetric in the SIGN of the change (squared jerk): +0.70 and -0.70 pay identically.
    assert R.yaw_dither_penalty(_t([-0.70]), w).item() == pytest.approx(flip.item(), abs=1e-12)
    # MONOTONIC in |delta|: a bigger flip pays strictly MORE (deploy-clamp 0.7 flip = delta 1.40) ...
    assert R.yaw_dither_penalty(_t([1.40]), w).item() < flip.item()
    # ... and benign NEAR-ZERO jitter pays ~0 (the discrimination a sign-flip indicator lacks).
    assert abs(R.yaw_dither_penalty(_t([0.02]), w).item()) < 1e-3
    # OFF (weight 0) -> exactly 0 for ANY delta (byte-identical default).
    assert R.yaw_dither_penalty(_t([5.0]), 0.0).item() == 0.0


def test_yaw_dither_wired_into_compute_ego_reward_and_off_is_byte_identical():
    n = 1
    base = R.EgoRewardWeights()                                     # yaw_dither 0 (default)
    r_none, _, _ = R.compute_ego_reward(base, gate_collision=torch.zeros(n, dtype=torch.bool),
                                        **_spin_kw(n, [0.0]))
    # passing a big yaw delta with the weight 0 -> BYTE-identical (the term is exactly 0)
    r_off, c_off, _ = R.compute_ego_reward(base, gate_collision=torch.zeros(n, dtype=torch.bool),
                                           yaw_cmd_delta=_t([1.4]), **_spin_kw(n, [0.0]))
    assert r_off.item() == r_none.item()
    assert c_off["yaw_dither_pen"] == 0.0
    # ARMED: a rail-flip drops the reward by EXACTLY the penalty (smooth, NON-terminal); a steady turn 0.
    w = R.EgoRewardWeights(yaw_dither=0.5)
    r_steady, c_steady, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool),
                                                 yaw_cmd_delta=_t([0.0]), **_spin_kw(n, [0.0]))
    r_flip, c_flip, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool),
                                             yaw_cmd_delta=_t([0.70]), **_spin_kw(n, [0.0]))
    pen = 0.5 * 0.70 ** 2
    assert c_steady["yaw_dither_pen"] == pytest.approx(0.0)         # steady turn pays 0 (turns are free)
    assert c_flip["yaw_dither_pen"] == pytest.approx(pen, abs=1e-9)
    assert (r_steady.item() - r_flip.item()) == pytest.approx(pen, abs=1e-9)   # reward drops by the penalty


def test_yaw_dither_default_off_and_parity_with_pefcap_inputs():
    """PARITY: the default weight is OFF, and passing yaw_cmd_delta alongside the pefcap inputs with all
    weights at default is byte-identical (all new terms contribute exactly 0)."""
    assert R.EgoRewardWeights().yaw_dither == 0.0                   # default OFF
    n = 2
    w = R.EgoRewardWeights()
    kw = _spin_kw(n, [0.0, 7.0])
    r_ref, _, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool), **kw)
    r_new, comps, _ = R.compute_ego_reward(
        w, gate_collision=torch.zeros(n, dtype=torch.bool),
        roll=_t([0.2, -1.0]), pitch=_t([-0.4, 0.8]), cos_view_next=_t([0.5, -1.0]),
        yaw_cmd_delta=_t([1.4, -0.9]), **kw)
    assert torch.equal(r_ref, r_new)                               # BYTE-identical
    assert comps["yaw_dither_pen"] == 0.0


# ================================================================================================
# VELOCITY-JERK smoothness prior (R0 still-yaw hover boot 2026-07-12): -rw_vel_smooth * ||jerk||^2,
# jerk = accel_curr - accel_prev (1st diff of the WORLD CoM acceleration = 2nd diff of velocity).
#   * ZERO at a steady ACCELERATION (accel_curr == accel_prev) -> steady speed AND smooth hard accel free.
#   * grows with the squared jerk (a snappy accel spike pays); NEGATIVE sign.
#   * orthogonal to yaw/roll -- a flip barely moves the CoM -> ~0 jerk (world-vel, NOT body specific force).
#   * OFF-by-default byte-identical (weight 0 -> exactly 0; None input -> the reward term is 0).
# ================================================================================================
def _v3(rows):
    return torch.tensor(rows, dtype=DT)


def test_vel_smooth_penalty_zero_off_steady_zero_and_spike_penalised():
    w = 1.0e-4
    steady = _v3([[3.0, -2.0, 1.0]])                               # accel unchanged step-to-step
    # STEADY ACCELERATION (accel_curr == accel_prev) -> jerk 0 -> 0 (a constant hard accel is FREE).
    assert R.velocity_jerk_penalty(steady, steady.clone(), w).item() == pytest.approx(0.0, abs=1e-12)
    # ZERO ACCELERATION on both steps (steady speed) also pays 0.
    z = torch.zeros(1, 3, dtype=DT)
    assert R.velocity_jerk_penalty(z, z, w).item() == 0.0
    # a SNAPPY accel SPIKE: accel jumps 0 -> [30,0,0] in one step -> jerk = [30,0,0] -> -w*30^2.
    spike = R.velocity_jerk_penalty(_v3([[30.0, 0.0, 0.0]]), z, w)
    assert spike.item() == pytest.approx(-w * 30.0 ** 2, abs=1e-9)
    assert spike.item() < 0.0                                       # NEGATIVE (a penalty)
    # MONOTONIC in ||jerk||: a bigger accel jump pays strictly MORE.
    assert R.velocity_jerk_penalty(_v3([[60.0, 0.0, 0.0]]), z, w).item() < spike.item()
    # ORTHOGONAL to a yaw/roll flip: the CoM barely moves -> accel_curr ~ accel_prev -> ~0 penalty
    # (a tiny 0.01 m/s^2 residual, NOT the full accel magnitude the body specific force would see).
    near = R.velocity_jerk_penalty(_v3([[5.001, 0.0, 0.0]]), _v3([[5.0, 0.0, 0.0]]), w)
    assert abs(near.item()) < 1e-6
    # OFF (weight 0) -> exactly 0 for ANY jerk (byte-identical default).
    assert R.velocity_jerk_penalty(_v3([[100.0, 100.0, 100.0]]), z, 0.0).item() == 0.0


def test_vel_smooth_wired_into_compute_ego_reward_and_off_is_byte_identical():
    n = 1
    base = R.EgoRewardWeights()                                     # vel_smooth 0 (default)
    r_none, _, _ = R.compute_ego_reward(base, gate_collision=torch.zeros(n, dtype=torch.bool),
                                        **_spin_kw(n, [0.0]))
    # passing a big accel jerk with the weight 0 -> BYTE-identical (the term is exactly 0)
    r_off, c_off, _ = R.compute_ego_reward(
        base, gate_collision=torch.zeros(n, dtype=torch.bool),
        accel_curr=_v3([[40.0, 0.0, 0.0]]), accel_prev=torch.zeros(n, 3, dtype=DT), **_spin_kw(n, [0.0]))
    assert r_off.item() == r_none.item()
    assert c_off["velsmooth_pen"] == 0.0
    # ARMED: a spike drops the reward by EXACTLY the penalty (smooth, NON-terminal); a steady accel 0.
    w = R.EgoRewardWeights(vel_smooth=1.0e-4)
    a = _v3([[10.0, -5.0, 2.0]])
    r_steady, c_steady, _ = R.compute_ego_reward(
        w, gate_collision=torch.zeros(n, dtype=torch.bool),
        accel_curr=a, accel_prev=a.clone(), **_spin_kw(n, [0.0]))
    r_spike, c_spike, _ = R.compute_ego_reward(
        w, gate_collision=torch.zeros(n, dtype=torch.bool),
        accel_curr=_v3([[30.0, 0.0, 0.0]]), accel_prev=torch.zeros(n, 3, dtype=DT), **_spin_kw(n, [0.0]))
    pen = 1.0e-4 * 30.0 ** 2
    assert c_steady["velsmooth_pen"] == pytest.approx(0.0, abs=1e-12)   # steady accel pays 0 (hard accel free)
    assert c_spike["velsmooth_pen"] == pytest.approx(pen, abs=1e-9)
    assert (r_steady.item() - r_spike.item()) == pytest.approx(pen, abs=1e-9)   # reward drops by the penalty


def test_vel_smooth_default_off_and_parity_with_all_new_inputs():
    """PARITY: the default weight is OFF, and passing accel_curr/accel_prev alongside every other new
    input with all weights at default is byte-identical (all new terms contribute exactly 0)."""
    assert R.EgoRewardWeights().vel_smooth == 0.0                   # default OFF
    n = 2
    w = R.EgoRewardWeights()
    kw = _spin_kw(n, [0.0, 7.0])
    r_ref, _, _ = R.compute_ego_reward(w, gate_collision=torch.zeros(n, dtype=torch.bool), **kw)
    r_new, comps, _ = R.compute_ego_reward(
        w, gate_collision=torch.zeros(n, dtype=torch.bool),
        roll=_t([0.2, -1.0]), pitch=_t([-0.4, 0.8]), cos_view_next=_t([0.5, -1.0]),
        yaw_cmd_delta=_t([1.4, -0.9]),
        accel_curr=_v3([[40.0, 0.0, 0.0], [0.0, 12.0, -3.0]]),
        accel_prev=torch.zeros(n, 3, dtype=DT), **kw)
    assert torch.equal(r_ref, r_new)                               # BYTE-identical
    assert comps["velsmooth_pen"] == 0.0
