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
