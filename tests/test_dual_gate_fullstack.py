"""Tests for the DUAL_GATE_FULLSTACK port (multi-gate stage designer, 2026-07-09 audit wave-4).

The stage ``dual_gate_fullstack0`` ports the PROVEN single-gate champion stack
(single_gate_varied_gvf_lpara_anneal: online head-on racing line + PBRS corridor 4 + magnitude
centering 0.4 + the per-crossing parabola) to 2 gates. The audit finding is that the SAME machinery is
ALREADY 2-gate-correct with NO code change -- the racing line is GLOBAL (spans spawn->g0->g1, head-on
at each, one build, no per-target re-plan), the parabola/corridor/centering are target-indexed, the obs
2nd slot carries the next gate, and the terminal fires the finish on the last gate. These tests PROVE
each of those, plus single-gate byte-identity of the shared racing-line + reward code the port reuses.

Coverage (prompt task 3):
  (1) single-gate BYTE-IDENTITY: the ladder is untouched; the champion single-gate stage config is
      unperturbed; a seeded single-gate racing-line + parabola/corridor reward yields the exact golden
      values the champion validated (a regression guard on the shared code the dual-gate stage reuses).
  (2) 2-gate RACING LINE: one global build spans both gates head-on, monotone+continuous arc-length
      across the gate-0 handoff (=> progress needs NO re-plan on advance), and cross-track perp is
      measured against the correct segment on BOTH legs (=> corridor follows the line after advance).
  (3) 2-gate REWARD: the parabola pays PER GATE, ONCE EACH (target-advance idempotency, no latch needed
      under miss_terminates=True), by the crossing offset; the corridor telescopes across the handoff.

Everything here is PURE torch (racing_line + ego_reward); PeregrineRacingEgo needs diffaero (cluster-
only), so -- like tests/test_ego_reward.py / test_racing_line.py -- we drive the pure functions with the
exact quantities the env feeds them. Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_dual_gate_fullstack.py -q
"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import vq2_ego_curriculum as CUR                                            # noqa: E402  (pure data)

torch = pytest.importorskip("torch")

import ego_reward as R                                                      # noqa: E402
from racing_line import build_racing_line, gate_normals_from_yaw           # noqa: E402

DT = torch.float64
STAGE = "dual_gate_fullstack0"


class _Cfg:
    """Attribute bag standing in for the resolved hydra cfg (getattr access only)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stage_weights(stage: str) -> R.EgoRewardWeights:
    """Build the EXACT EgoRewardWeights the env would construct for ``stage`` (the same from_cfg path),
    from the stage's rw_* keys (the _raw block carries no reward knobs)."""
    d = {k: v for k, v in CUR.STAGES[stage].items() if k != "_raw"}
    return R.EgoRewardWeights.from_cfg(_Cfg(**d))


# ================================================================================================
# (1) CONFIG + SINGLE-GATE BYTE-IDENTITY.
# ================================================================================================
def test_dual_gate_fullstack0_is_the_champion_stack_at_2_gates_offladder():
    """The port: OFF-LADDER (does not perturb the 4-stage ladder), 2 gates at 10-20 m spacing, the champion
    lpara reward with a STATIC zero=4 (NO cross-zero anneal), noise 0, warm from vglp4 + the 0.12/0.03/0.5
    std trio. The first-leg geometry matches vglp4 (dist 8-15 m, height +-6 m, yaw jitter 0.25) for a clean
    warm transfer."""
    assert STAGE not in CUR.STAGE_ORDER, "must be off-ladder (run standalone), not in the 4-stage ladder"
    s = CUR.STAGES[STAGE]
    # 2 gates, 10-20 m gate->gate spacing (the VQ2 co-visibility band).
    assert s["course_n_gates"] == 2
    assert s["course_seg_len_lo"] == 10.0 and s["course_seg_len_hi"] == 20.0
    # first leg == vglp4's distribution (clean warm-start transfer of the gate-0 skill).
    assert s["course_spawn_dist_lo"] == 8.0 and s["course_spawn_dist_hi"] == 15.0
    assert s["course_spawn_below_g0_lo"] == -6.0 and s["course_spawn_below_g0_hi"] == 6.0
    assert s["course_spawn_yaw_jitter"] == 0.25
    # the champion lpara reward, VERBATIM (racing line + corridor 4 + magnitude centering 0.4 + parabola).
    assert s["use_racing_line"] is True
    assert s["rw_progress_to_center"] is False        # progress from the global line arc-length (pure GVF)
    assert s["rw_corridor"] == 4.0
    assert s["rw_centering"] == 0.4 and s["rw_centering_max_m"] == 6.0
    assert s["rw_parabola_crossing"] is True
    assert s["rw_cross_center"] == 20.0 and s["rw_cross_zero_m"] == 4.0 and s["rw_cross_neg_cap"] == 100.0
    # STATIC zero=4: NO cross-zero anneal in v1 (one lever at a time; the anneal never beat fixed-4).
    assert "cross_zero_anneal" not in s
    # the latch is NOT enabled -- the port relies on target-advance idempotency (miss_terminates=True),
    # exactly as the champion validated (rw_parabola_latch would only matter with miss_terminates=false).
    assert "rw_parabola_latch" not in s
    # calibration regime: perfect estimator, so the 2-gate CONTROL question is isolated from corruption.
    assert s["ego_noise_scale"] == 0.0
    # warm from vglp4 + the champion 0.12/0.03/0.5 noise-std trio, 2-gate budget (max_time 60).
    raw = s["_raw"]
    assert raw["algo.gamma"] == CUR._GAMMA and raw["env.max_time"] == 60
    assert raw["+init_from"].endswith("ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints")
    assert (raw["++algo.noise_std_hold"], raw["++algo.noise_std_floor"], raw["++algo.noise_hold_frac"]) \
        == (0.12, 0.03, 0.5)


def test_reward_knobs_match_the_champion_single_gate_stage_exactly():
    """The port must reuse the champion's reward VERBATIM -- every rw_* knob in dual_gate_fullstack0 that the
    champion single_gate_varied_gvf_lpara_anneal also sets must have the IDENTICAL value (the deltas are ONLY
    the anneal removal, gate count / spacing, and ego_noise_scale). This pins the 'port, don't re-tune' rule."""
    champ = CUR.STAGES["single_gate_varied_gvf_lpara_anneal"]
    port = CUR.STAGES[STAGE]
    champ_rw = {k: v for k, v in champ.items() if k.startswith("rw_")}
    for k, v in champ_rw.items():
        assert port.get(k) == v, f"reward knob {k} drifted from the champion: {port.get(k)} != {v}"
    # the resolved weights are therefore the champion's, minus the (live-anneal-only) cross-zero schedule.
    w = _stage_weights(STAGE)
    assert w.parabola_crossing is True and w.cross_center == 20.0 and w.cross_zero_m == 4.0
    assert w.corridor == 4.0 and w.centering == 0.4 and w.centering_max_m == 6.0
    assert w.parabola_latch_once is False


def test_single_gate_ladder_and_champion_stage_unperturbed():
    """BYTE-IDENTITY guard (config side): adding the off-ladder port must NOT touch the 4-stage ladder nor
    any single-gate stage's rendered tokens. The champion single-gate stage renders its exact known token
    set (a frozen golden -- a perturbation of _COMMON or the champion dict would break this)."""
    assert CUR.STAGE_ORDER == ("single_gate", "handoff_drill", "dual_gate_full", "multi_gate")
    # the pure-_COMMON discovery stage: EXACTLY _COMMON + course_n_gates + _raw (no reward drift leaked in).
    extra = set(CUR.STAGES["single_gate"]) - set(CUR._COMMON) - {"course_n_gates", "_raw"}
    assert extra == set(), extra
    # the champion single-gate stage still renders the champion tokens (spot the load-bearing ones).
    toks = CUR.render_overrides("single_gate_varied_gvf_lpara_anneal")
    for t in ("+env.use_racing_line=true", "+env.rw_corridor=4.0", "+env.rw_centering=0.4",
              "+env.rw_parabola_crossing=true", "+env.rw_cross_zero_m=4.0", "+env.course_n_gates=1",
              "+env.cross_zero_anneal=true", "algo.gamma=0.9975"):
        assert t in toks, t


def test_single_gate_racingline_reward_golden_shared_path_stable():
    """BYTE-IDENTITY guard (shared-code side): the dual-gate stage reuses racing_line + ego_reward
    UNCHANGED, so the single-gate path must still produce its exact golden values. A dead-ahead single gate
    at 12 m: (i) the line is the straight spawn->gate, on-line perp==0 / s==distance; (ii) the parabola pays
    EXACTLY the champion invariants -- +cross_center dead-centre, 0 at the aperture edge (== cross_zero_m),
    the smooth quadratic in between; (iii) the corridor telescopes by 4*(perp_prev-perp_curr)."""
    spawn = torch.tensor([[0.0, 0.0, 0.0]], dtype=DT)
    gate = torch.tensor([[[12.0, 0.0, 0.0]]], dtype=DT)          # 1 gate, dead ahead
    yaw = torch.tensor([[0.0]], dtype=DT)
    line = build_racing_line(spawn, gate, yaw, samples_per_seg=48)
    # (i) straight line: a point at x=5 on the axis -> perp 0, s 5.
    s, perp, tang, _ = line.query(torch.tensor([[5.0, 0.0, 0.0]], dtype=DT))
    assert abs(float(perp)) < 1e-9 and abs(float(s) - 5.0) < 1e-9
    # (ii) parabola invariants (cross_center=20, cross_zero_m=4): exact math goldens.
    crossed = torch.ones(1, dtype=torch.bool)
    for off, want in [(0.0, 20.0), (4.0, 0.0), (2.0, 15.0), (1.0, 18.75)]:
        r = R.crossing_parabola_reward(torch.tensor([off], dtype=DT), crossed,
                                       cross_center=20.0, cross_zero_m=4.0, cross_neg_cap=100.0)
        assert float(r) == pytest.approx(want, abs=1e-12), (off, float(r), want)
    # (iii) corridor telescopes: moving perp 2.0 -> 1.0 (onto the line) pays 4*(2-1)=4.0 (band inactive).
    r_corr = R.corridor_progress_reward(torch.tensor([1.0], dtype=DT), torch.tensor([2.0], dtype=DT),
                                        rw_corridor=4.0, vmax_mps=39.0, dt=1 / 30)
    assert float(r_corr) == pytest.approx(4.0, abs=1e-12)


# ================================================================================================
# (2) 2-GATE RACING LINE: one global build, head-on at both, continuous arc-length, correct-leg perp.
# ================================================================================================
def _two_gate_line(spp=48):
    """A representative 2-gate course: g0 dead-ahead 12 m, g1 at (24, 6, 3) with a right/up turn."""
    spawn = torch.tensor([[0.0, 0.0, 0.0]], dtype=DT)
    gate = torch.tensor([[[12.0, 0.0, 0.0], [24.0, 6.0, 3.0]]], dtype=DT)
    yaw = torch.tensor([[0.0, 0.4]], dtype=DT)
    return spawn, gate, yaw, build_racing_line(spawn, gate, yaw, samples_per_seg=spp)


def test_global_line_spans_both_gates_head_on():
    """ONE build spans spawn->g0->g1: both gate centres lie on the line, and the tangent crosses EACH gate
    head-on (== the gate's through-normal), G1-continuous (no cusp) -- so a 'gate-2-aware exit tangent' is
    unnecessary AND would break the head-on crossing the parabola/corridor target."""
    spawn, gate, yaw, line = _two_gate_line()
    normals = gate_normals_from_yaw(yaw)[0]                      # (2,3)
    for g in range(2):
        c = gate[0, g]
        j = int(torch.linalg.norm(line.samples[0] - c, dim=-1).argmin())
        assert float(torch.linalg.norm(line.samples[0, j] - c)) < 0.05, f"gate {g} centre off the line"
        # tangent at the nearest sample aligns with that gate's normal (head-on crossing).
        tang = line.tangent[0, j]
        cos = float(torch.dot(tang, normals[g]) / (tang.norm() * normals[g].norm()))
        assert cos > 0.999, f"gate {g} not head-on: cos={cos:.4f}"


def test_arclength_monotone_and_continuous_across_gate0_handoff():
    """The GVF progress potential s is the GLOBAL arc length: monotone non-decreasing along the whole line
    AND continuous through the gate-0 knot (no jump). => progress telescopes across the g0->g1 handoff with
    NO per-target re-plan; the env's post-advance re-query is a position-based no-op (peregrine_racing_ego
    line ~970). Queried at points before g0, at g0, between, and at g1: s strictly increases."""
    spawn, gate, yaw, line = _two_gate_line()
    # sample-level monotonicity (the built line).
    assert (line.arclen[0, 1:] - line.arclen[0, :-1] >= -1e-9).all()
    # query-level monotone + continuity along the flight path (query() is 1-point-per-env, so query each).
    pts = [[3.0, 0.0, 0.0],      # before g0 (on the straight 1st leg)
           [12.0, 0.0, 0.0],     # at g0
           [18.0, 3.0, 1.5],     # roughly between the gates
           [24.0, 6.0, 3.0]]     # at g1
    svals = [float(line.query(torch.tensor([p], dtype=DT))[0]) for p in pts]
    assert svals[0] == pytest.approx(3.0, abs=1e-6)            # straight leg: s == distance
    assert svals[1] == pytest.approx(12.0, abs=1e-6)          # arc length at g0 == first-leg length
    assert svals[1] < svals[2] < svals[3], f"arc length not monotone across the handoff: {svals}"
    # continuity: the jump across the g0 knot is bounded by the local sample spacing (no discontinuity).
    d0 = torch.linalg.norm(line.samples[0] - gate[0, 0], dim=-1)
    j0 = int(d0.argmin())
    step = float((line.arclen[0, j0 + 1] - line.arclen[0, j0 - 1]))
    assert step < 1.0, f"arc length discontinuous at the g0 knot (local step {step:.3f})"


def test_perp_measured_against_the_correct_segment_on_both_legs():
    """Cross-track perp (the corridor/centering error) is read from the GLOBAL polyline, so it tracks the
    NEAREST segment on BOTH legs. On the straight 1st leg an on-axis point has perp 0; a laterally offset
    point returns that offset. On the 2nd leg a point ON a line sample has perp ~0, and a point offset from
    it returns ~the offset -- proving the corridor follows the line AFTER the gate-0 advance (no re-plan)."""
    spawn, gate, yaw, line = _two_gate_line()
    # 1st leg: on-axis -> perp 0; 2 m lateral -> perp 2.
    _, perp_on, _, _ = line.query(torch.tensor([[6.0, 0.0, 0.0]], dtype=DT))
    _, perp_off, _, _ = line.query(torch.tensor([[6.0, 2.0, 0.0]], dtype=DT))
    assert float(perp_on) < 1e-9 and abs(float(perp_off) - 2.0) < 0.05
    # 2nd leg: pick an actual line sample PAST g0, then offset it perpendicular to the local tangent.
    d0 = torch.linalg.norm(line.samples[0] - gate[0, 0], dim=-1)
    j0 = int(d0.argmin())
    j2 = (j0 + line.samples.shape[1]) // 2                      # a sample on the 2nd leg
    p_on = line.samples[0, j2]
    tang = line.tangent[0, j2]
    # a vector perpendicular to the local tangent (cross with world-up, else world-x).
    up = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    perp_dir = torch.linalg.cross(tang, up)
    if float(perp_dir.norm()) < 1e-6:
        perp_dir = torch.linalg.cross(tang, torch.tensor([1.0, 0.0, 0.0], dtype=DT))
    perp_dir = perp_dir / perp_dir.norm()
    s_on, perp_on2, _, _ = line.query(p_on.unsqueeze(0))
    s_off, perp_off2, _, _ = line.query((p_on + 1.5 * perp_dir).unsqueeze(0))
    assert float(perp_on2) < 1e-6, f"on-2nd-leg-sample perp not ~0: {float(perp_on2)}"
    assert abs(float(perp_off2) - 1.5) < 0.05, f"2nd-leg offset perp wrong: {float(perp_off2)}"
    assert float(s_on) > 12.0, "the 2nd-leg sample must be PAST the g0 arc length (12 m)"


# ================================================================================================
# (3) 2-GATE REWARD: parabola pays per gate ONCE EACH; corridor telescopes across the handoff.
# ================================================================================================
def _reward(w, **over):
    """Drive compute_ego_reward for N=1 with benign, non-terminating defaults, mirroring how the env calls
    it under the parabola (floor_contact supplied so the parabola terminal branch is taken). Overrides set
    only the crossing/segment quantities each step needs. Returns (reward, components)."""
    z = torch.zeros(1, dtype=DT)
    zb = torch.zeros(1, dtype=torch.bool)
    kw = dict(
        s_curr=z.clone(), s_prev=z.clone(),
        gate_passed=zb.clone(), pass_linf=z.clone(), w_g_half=0.375,
        gate_collision=zb.clone(), gate_miss=zb.clone(), oob=zb.clone(),
        banked_progress_return=z.clone(),
        newly_finished=zb.clone(), time_left_s=torch.full((1,), 30.0, dtype=DT),
        tilt_cos_r33=torch.ones(1, dtype=DT), omega=torch.zeros(1, 3, dtype=DT),
        action_norm=torch.full((1, 4), 0.5, dtype=DT), last_action_norm=torch.full((1, 4), 0.5, dtype=DT),
        vel_world=torch.zeros(1, 3, dtype=DT),
        curr_center=torch.zeros(1, 3, dtype=DT), next_center=torch.zeros(1, 3, dtype=DT),
        dt=1 / 30,
        perp_dist=z.clone(), perp_prev=z.clone(),
        cross_offset=z.clone(), crossed=zb.clone(),
        floor_contact=z.clone(),
    )
    kw.update(over)
    reward, components, _ = R.compute_ego_reward(w, **kw)
    return reward, components


def test_parabola_pays_per_gate_once_each_over_a_2gate_advance_sequence():
    """The env fires the parabola on ``crossed`` = fwd_t[tg] (the CURRENT target's forward plane crossing).
    A realistic 2-gate episode -- approach g0, cross g0 (advance to tg=1), coast, cross g1 -- pays the
    parabola EXACTLY on the two crossing steps, ONCE per gate, valued by the crossing offset; and 0 on every
    non-crossing step. This is the once-per-gate idempotency the target-advance gives WITHOUT the latch."""
    w = _stage_weights(STAGE)
    no = torch.zeros(1, dtype=torch.bool)
    yes = torch.ones(1, dtype=torch.bool)

    # step A: approaching gate 0, no crossing -> parabola 0.
    _, cA = _reward(w, crossed=no.clone(), cross_offset=torch.tensor([5.0], dtype=DT))
    # step B: cross gate 0 DEAD CENTRE (offset 0) -> parabola +20 (peak).
    _, cB = _reward(w, crossed=yes.clone(), cross_offset=torch.tensor([0.0], dtype=DT),
                    gate_passed=yes.clone())
    # step C: between the gates (target already advanced to g1), no crossing -> 0.
    _, cC = _reward(w, crossed=no.clone(), cross_offset=torch.tensor([3.0], dtype=DT))
    # step D: cross gate 1 at 2.0 m offset -> parabola 20*(1-(2/4)^2) = 15.0.
    _, cD = _reward(w, crossed=yes.clone(), cross_offset=torch.tensor([2.0], dtype=DT),
                    gate_passed=yes.clone())

    assert cA["cross_parabola_reward"] == pytest.approx(0.0, abs=1e-12), "no pay off a crossing (approach)"
    assert cB["cross_parabola_reward"] == pytest.approx(20.0, abs=1e-9), "gate 0 centred pays the peak"
    assert cC["cross_parabola_reward"] == pytest.approx(0.0, abs=1e-12), "no pay between gates"
    assert cD["cross_parabola_reward"] == pytest.approx(15.0, abs=1e-9), "gate 1 pays by its offset"
    # the passage reward is REPLACED by the parabola (both crossing steps), so it never double-credits.
    assert cB["pass_reward"] == pytest.approx(0.0, abs=1e-12)
    assert cD["pass_reward"] == pytest.approx(0.0, abs=1e-12)
    total_parabola = sum(c["cross_parabola_reward"] for c in (cA, cB, cC, cD))
    assert total_parabola == pytest.approx(35.0, abs=1e-9), "exactly gate0(+20) + gate1(+15), once each"


def test_target_advance_is_the_once_per_gate_guard_not_the_latch():
    """WHY no latch is needed: the parabola's once-per-gate idempotency comes from the ENV, not a latch. In
    the env a forward crossing of the target ALWAYS advances the target (pass) or terminates (frame strike /
    wide miss, since miss_terminates=True), so ``crossed``=fwd_t[tg] can fire at most once per gate. This
    test documents the failure mode the env structurally prevents: WITHOUT a latch, two crossed=True steps
    on the SAME offset would double-pay -- which is exactly why the env terminates a non-advancing crossing
    (and why rw_parabola_latch stays off for this stage)."""
    off = torch.tensor([1.0], dtype=DT)
    crossed = torch.ones(1, dtype=torch.bool)
    r1 = R.crossing_parabola_reward(off, crossed, 20.0, 4.0, 100.0)          # no latch
    r2 = R.crossing_parabola_reward(off, crossed, 20.0, 4.0, 100.0)          # would re-pay if env let it re-cross
    assert float(r1) == float(r2) == pytest.approx(18.75, abs=1e-9)          # both pay -> env must NOT re-cross
    # with the latch, the 2nd identical crossing pays 0 (the defensive knob, unused here because the env's
    # miss_terminates=True already guarantees no re-cross of a live target).
    paid = torch.zeros(1, dtype=torch.bool)
    r_latch_1 = R.crossing_parabola_reward(off, crossed, 20.0, 4.0, 100.0, latch_once=True, paid_latch=paid)
    r_latch_2 = R.crossing_parabola_reward(off, crossed, 20.0, 4.0, 100.0, latch_once=True, paid_latch=paid)
    assert float(r_latch_1) == pytest.approx(18.75, abs=1e-9) and float(r_latch_2) == 0.0


def test_corridor_follows_the_line_perp_across_the_handoff():
    """The corridor term (rw_corridor=4.0) telescopes on the GLOBAL-line perp, so it keeps supplying the
    cross-track homing on the 2nd leg after the gate-0 advance. Driven with real perp values from a 2-gate
    line query at two consecutive positions straddling the handoff: moving toward the line (perp shrinks)
    pays positive; drifting off pays negative -- exactly the PBRS contouring, on the post-g0 segment."""
    spawn, gate, yaw, line = _two_gate_line()
    w = _stage_weights(STAGE)
    # two positions on the 2nd leg: one 1.2 m off the line, the next 0.4 m off (moving onto the line).
    d0 = torch.linalg.norm(line.samples[0] - gate[0, 0], dim=-1)
    j0 = int(d0.argmin())
    j2 = (j0 + line.samples.shape[1]) // 2
    p_on = line.samples[0, j2]
    tang = line.tangent[0, j2]
    up = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    perp_dir = torch.linalg.cross(tang, up)
    perp_dir = perp_dir / perp_dir.norm()
    _, perp_prev, _, _ = line.query((p_on + 1.2 * perp_dir).unsqueeze(0))
    _, perp_curr, _, _ = line.query((p_on + 0.4 * perp_dir).unsqueeze(0))
    assert float(perp_prev) > float(perp_curr)                              # moved toward the line
    _, comp = _reward(w, perp_dist=perp_curr, perp_prev=perp_prev)
    # corridor reward = rw_corridor * (perp_prev - perp_curr); the env exposes it as "corridor_reward".
    assert comp["corridor_reward"] == pytest.approx(4.0 * float(perp_prev - perp_curr), abs=1e-9)
    assert comp["corridor_reward"] > 0.0, "moving onto the 2nd-leg line must pay positive contouring"
