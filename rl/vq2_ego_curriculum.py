"""rl/vq2_ego_curriculum.py -- the VQ2 EGOCENTRIC (inc9 / peregrine_racing_ego) curriculum ladder
(DESIGN.md `docs/vq2-egocentric-gen/DESIGN.md` §D/§E). NEW module -- it does NOT touch the inc8
rl/vq2_curriculum.py (that is FROZEN for the inc8 reproducibility pins) nor the diffaero clone.

Like the inc8 curriculum this is a LADDER of config-override dicts: each stage is a set of hydra
`+env.`/bare overrides layered on the base ego recipe, rendered by ``render_overrides`` into the CLI
tokens the sbatch consumes. Pure DATA + a tiny renderer -- laptop-testable, imports nothing heavy.

======================================================================================================
WHY THIS IS A NEW LADDER (not the inc8 one) -- the egocentric generation's structural differences:
======================================================================================================
  * env=peregrine_racing_ego + ``+env.ego=true`` (the 21-dim position-free obs + 16-dim privileged
    critic + refined-B reward). The inc8 ladder trains the gate-frame single-gate obs; incompatible.
  * standing_start_frac=1.0 EVERYWHERE (DESIGN.md §D): the trivial 1 m near-spawn dash is REMOVED --
    every episode is a genuine standing start from the pad. (inc8 used 0.3; the ego generation trains
    the whole approach.) Spawn-attitude jitter is ALREADY always-on in the base env (0.3 rad axis-angle
    + 0.5 m xy at reset), so the gate starts off-centre in the FOV with no extra knob.
  * blackout_pass DELETED (DESIGN.md §5.B/§D): blackout is EMERGENT from the keypoint-visibility model
    (component B computes the exact close-range corner-exit range under the 20 deg tilt) -- there is NO
    dedicated blackout stage and NO emul_blackout_range_m / emul_lat_* knobs (those were inc8
    estimator-emulator knobs; the ego estimator + gate_visibility own that behaviour intrinsically).
  * gamma=0.9975 EVERYWHERE (LOAD-BEARING, DESIGN.md §8 portfolio arm promoted to default): at the
    deployment course scale the crash penalty must still DOMINATE the banked progress return at the
    LATE gates. At gamma=0.99 the terminal is discounted away by gate ~6-20 (credit half-life ~2.3 s at
    30 Hz) and sprint-and-clip creeps back to positive-return; gamma=0.9975 keeps the dominance margin
    positive at ng in {6,20,60} (pinned by tests/test_ego_reward.py::test_terminal_dominance_rollout,
    parametrized over ng x gamma -- gamma=0.99 THINS at scale, documented there).
  * algo=appo EVERYWHERE: the privileged asymmetric critic (get_state 16-dim GT). algo=ppo builds a
    SYMMETRIC obs-only critic (get_state never consumed) = the inc8 seed-collapse root cause; the
    launcher's post-agent-build assert (env.assert_appo_critic) FIRES on ppo/symmetric so a mis-set
    run raises AssertionError instead of silently collapsing.

======================================================================================================
REWARD SCOPING -- the B2b CATASTROPHIC-FORGETTING LESSON, encoded structurally:
======================================================================================================
B2b (2026-07-06): over-applying HARD-STAGE reward knobs (M3 noise_anneal, M4 economy) to the EASY
stages via a shared _COMMON REGRESSED single_gate 0.82 -> 0.035 (even the 1 m near-spawn dash to 4.3%).
The isolation probes proved the easy discovery phase needs its OWN dynamics; a hard-stage knob in
_COMMON is a footgun. So here:

  * _COMMON (EASY-SAFE -- applies to ALL stages INCLUDING single_gate): ONLY the refined-B champion
    defaults that are safe at the discovery phase: rw_progress=1.0, rw_passage=1.0, terminal_base=200,
    terminal_progress_scaled=True, exit_align=0.0 (OFF), the minuscule smoothness (~1e-3), the FIXED
    60 deg free cone, gamma=0.9975. These are the EgoRewardWeights defaults -- single_gate trains on
    the clean champion reward with NO turning-stage pressure.
  * STAGE-SPECIFIC (HARD / turning stages ONLY -- dual_gate_full, multi_gate; NEVER single_gate,
    NEVER handoff_drill's easy discovery): rw_passage bumped 2-4x (the centering margin is thin at race
    speed through a SHARP turn -- a wider passage basin only where turns are hard) and exit_align > 0
    small (the next-gate exit-line anticipation; it is GATE-GATED in ego_reward -- fires once per pass,
    non-farmable -- so it is safe to add). These do NOT go in _COMMON: single_gate must stay on the
    clean champion reward it validated at ~0.8 before any hard-stage knob is layered.

VALIDATE-ON-single_gate discipline (DESIGN.md §6.7): single_gate must reach ~0.8 before promoting; any
NEW knob is validated on single_gate BEFORE it is allowed into _COMMON.
"""
from __future__ import annotations

# ================================================================================================
# _COMMON -- EASY-SAFE knobs applied to EVERY stage (incl. single_gate). ONLY the refined-B champion
# defaults that are safe at the discovery phase. NO hard-stage reward pressure lives here (B2b lesson).
# ================================================================================================
_COMMON = {
    # --- the egocentric env + privileged-critic training path ---
    "ego": True,                     # +env.ego=true -> the 21-dim position-free obs + refined-B reward
    "standing_start_frac": 1.0,      # trivial 1 m dash REMOVED -- genuine standing start EVERY episode
    "course_mode": "random",         # per-env procedural courses (spacing/geometry per stage below)
    # --- CLOSER FIRST GATE (Fengyou 2026-07-07): the standing-start pad -> gate-0 distance. The sampler
    #     default is 18-28 m, but the spec / VQ2 co-visibility regime wants the first gate 10-20 m out.
    #     A shorter first approach is (a) easier to DISCOVER (less distance for aim + altitude error to
    #     compound) and (b) leaves less altitude budget to bleed before the gate (the diagnosed dive).
    #     Applies to EVERY stage so the first leg matches the 10-20 m gate->gate spacing (no odd long
    #     first leg). Consumed by peregrine_racing_ego.resolve_course_overrides -> spawn_dist_m. ---
    "course_spawn_dist_lo": 10.0, "course_spawn_dist_hi": 20.0,
    # --- FIXED SPAWN HEADING (Fengyou 2026-07-07): pin the segment-0 world heading so the courses do NOT
    #     fan into a redundant circle. The egocentric position-free obs (and the body-frame privileged
    #     critic) are INVARIANT to the global course heading, so the sampler's random heading adds zero
    #     training signal -- it only scattered the world-frame render. Fixing it leaves the egocentric
    #     training distribution IDENTICAL while placing every course ahead of the pad (cleaner + clearer).
    #     Relative gate placement still varies via the spawn attitude jitter (0.3 rad) + the turn/drop walk. ---
    "course_spawn_heading": 0.0,
    # --- refined-B reward EASY-SAFE defaults (safe/HELPFUL on single_gate) ---
    # REWARD REDESIGN (2026-07-07, Fengyou "you get the final call"): the render diagnosed a GENUINE
    # wide/angled miss + a near-spawn dive, NOT a scoring bug. The fixes, all DISCOVERY-FRIENDLY (they
    # HELP single_gate -> _COMMON is correct; NOT the B2b hard-stage-leak footgun):
    #   * rw_progress 6.0 -> 2.0: 6.0 over-rewarded rushing forward (the drone pitched hard + dived at the
    #     gate). 2.0 is a moderate dense pull -- still 2x the original 1.0 that was "too weak", but no
    #     longer a sprint incentive. The AREA-DISTANCE coupling (below) further conditions it on a
    #     square-on close-in approach.
    #   * rw_passage 1.0 -> 5.0 + rw_passage_increment 1.0: passing CENTERED is now clearly worth more
    #     than approaching (gate 0 -> +5, gate 1 -> +6, ...); later gates pay more (get deeper = better).
    #   * rw_area_dist_ref_m 6.0: DISTANCE-GATED area coupling of the positive progress -- a shallow /
    #     off-axis approach in the final ~6 m earns LESS (it threads at an angle and exits wide), a
    #     beeline from far earns FULL. Directly attacks the diagnosed angled wide miss; ref sets the
    #     close-in distance over which it engages (see ego_reward.area_distance_progress_factor).
    #   * miss/oob terminals: MISS stays forgiving (~hover) so an imperfect gate ATTEMPT is not
    #     catastrophic; OOB (leaving the arena) stays discouraged; CONTACT stays catastrophic (200).
    # PROGRESS-TO-CENTRE (2026-07-07, THE root-cause fix). The render+geometry diagnosis: on a DEAD-AHEAD
    # gate (spawn azimuth 0) the drone still diffused ~8 m laterally / ~6 m vertically and missed
    # (single_gate 0% across ~5 reward variants). Cause: segment-projected progress credits only
    # ALONG-TRACK advance -- perpendicular drift earns ZERO -- so a diagonal flight banks near-full
    # progress while sliding off the line; there was NO dense lateral/vertical homing gradient. The
    # centering PENALTY (below) tried to add one but as a magnitude penalty it triggered a "give up and
    # leave the box" pathology (the render's 72% early OOB). FIX: switch the progress POTENTIAL to
    # phi = -||pos - gate_centre|| (the inc7/Swift distance-to-gate progress) -> a dense homing gradient
    # in EVERY axis. A true Euclidean potential telescopes -> NON-farmable (does not farm lateral drift,
    # unlike the polyline argmin that motivated segment-only). This is the champion (Swift) progress; the
    # refined-B rework had dropped it for segment-only and THAT is the single_gate regression.
    "rw_progress_to_center": True,   # dense 3D homing to the gate CENTRE (was segment-only along-track)
    "rw_progress": 2.0,              # dense homing progress weight (per metre closed toward the centre)
    "rw_passage": 5.0,               # L-inf centering passage BASE (was 1.0); passing >> approaching now
    "rw_passage_increment": 1.0,     # per-gate passage bump: gate g pays (5 + 1*g) -> deeper = better
    # AREA-DISTANCE coupling now OFF (reward-audit 2026-07-07). With rw_progress_to_center=True the
    # progress scalar BUNDLES the lateral/vertical homing correction; the area multiplier (factor in
    # [area,1], area->0 when off-axis) attenuated that homing by ~35-58% in the final 1-3 m -- worst for
    # the most off-center drones, exactly where centering must happen -- and on a DEAD-AHEAD single_gate
    # its "square-up before an oblique exit" rationale does not even apply. Re-enable ONLY on turning
    # stages if wanted, and only on the ALONG-TRACK component, never as a tax on the homing scalar.
    "rw_area_dist_ref_m": 0.0,       # distance-gated area coupling ramp (m); 0 == coupling OFF
    # DENSE LATERAL CENTERING -- now OFF. It was the WRONG FORM: a per-step magnitude penalty
    # (-rw_centering*perp) the policy could not yet avoid, so it learned to END the episode early (leave
    # the box) to cap the accruing loss -> the render's 72% early-OOB blow-out (worse than the 93% wide
    # miss without it). The progress-to-centre potential above supplies the lateral/vertical homing
    # CONSTRUCTIVELY (reward for getting closer, no give-up incentive), so this redundant + harmful
    # penalty is disabled. Kept as a 0-knob for A/B, not deleted.
    "rw_centering": 0.0,             # OFF (superseded by rw_progress_to_center; the penalty form back-fired)
    "rw_centering_max_m": 2.0,       # (unused while rw_centering==0) clamp (m) on the perpendicular offset
    "rw_terminal_base": 200.0,       # kill-on-contact base magnitude (CONTACT stays catastrophic)
    # MISS base lowered 30 -> 8 (reward-audit 2026-07-07). Now that a miss KEEPS its banked approach
    # progress (the forfeit is contact-only), the miss base must be <= the banked approach (~20 at the
    # 10 m spawn end, ~30 at 20 m) so an HONEST close-but-missed attempt reliably nets POSITIVE vs
    # hovering -- restoring the documented "~hover, forgiving" intent (30 was ~4x the true discounted
    # hover cost, and at the short-spawn end an attempt still lost to hover). Pairs with the forfeit fix.
    "rw_terminal_miss": 8.0,         # fly-by MISS (wide gate attempt, stays in-arena): ~hover -> forgiving
    # OOB (leaving the ARENA) STAYS discouraged at 200 -- distinct from MISS. Measured 2026-07-07: at
    # oob=30 the drone had NO pressure to stay in bounds and 96% of episodes OOB'd at ~2 s (l_episode
    # 2 s vs the 33 s in-bounds hover at oob=200), never reaching the gate. The fix forgives the GATE
    # ATTEMPT (miss), NOT leaving the arena. NOTE: a below-FLOOR dive is NOT oob -- the ego env now
    # reclassifies it as a CONTACT (terminal_base=200, a crash/DQ), so oob here = lateral/ceiling only.
    "rw_terminal_oob": 200.0,        # out-of-bounds = leaving the arena -> stays discouraged (NOT ~hover)
    "rw_terminal_progress_scaled": True,  # forfeit banked progress + base -> sprint-and-clip never wins
    "rw_exit_align": 0.0,           # next-gate exit-line OFF on easy stages (ON only hard stages below)
    "rw_rate": 1.0e-3, "rw_dact": 1.0e-3,  # MINUSCULE smoothness (3-4 orders below progress)
    "rw_tilt_free_rad": 1.0471976,   # FIXED 60 deg free cone
    # HOVER-HOLD probe bonus OFF on every real stage (turned ON only by the hover_hold diagnostic stage).
    "rw_altitude_hold": 0.0,         # give-up-resistant spawn-altitude bonus; 0 == OFF
    "rw_altitude_hold_band_m": 8.0,  # (m) decay half-width (unused while rw_altitude_hold==0)
    # MPCC CONTOURING OFF by default (turned ON only by the single_gate_static_mpcc lever stage).
    "rw_corridor": 0.0,              # PBRS perpendicular-contouring weight; 0 == OFF
    "rw_corridor_clip_mps": 39.0,    # contouring clip band (m/step = mps*dt); trims gate-handoff bursts
}

# The gamma value shared by every stage (LOAD-BEARING for terminal dominance at deployment scale). It
# is an EXISTING base-config key (algo.gamma), so it is emitted VERBATIM (no `+`) via the "_raw" block.
_GAMMA = 0.9975


# ================================================================================================
# The ordered stage ladder. Each value is a hydra key->value rendered `+env.<k>=<v>` EXCEPT the "_raw"
# sub-dict whose entries are emitted VERBATIM (existing non-env keys: algo.gamma / env.max_time -- these
# already exist in the base config, so they take NO `+`). Course-sampler keys (course_n_gates /
# course_seg_len_{lo,hi} / course_drop_{lo,hi}) live under the same `+env.` namespace (COURSE_SAMPLER_KEYS).
# ================================================================================================
STAGES: dict[str, dict] = {
    # 1. SINGLE GATE, standing start, gate off-centre in FOV. The FRESH-START stage: ONE gate, the clean
    #    champion reward (pure _COMMON, no hard-stage knob), standing_start_frac=1.0. MUST reach ~0.8
    #    before promoting (DESIGN.md §6.7). ~1500 updates.
    "single_gate": {
        **_COMMON,
        "course_n_gates": 1,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # DIAGNOSTIC STAGE (Fengyou 2026-07-07): a STATIC single gate -- FIXED 15 m dead-ahead, LEVEL (drop 0),
    # heading pinned -- so the gate does NOT move between episodes. Removes ALL course variance to isolate
    # the gross control failure ("turn down the scope to single gate, gate doesn't move"). NOT in the ladder
    # (STAGE_ORDER); run standalone via STAGES=single_gate_static. Reads out metrics/exit_* (box-exit
    # classification) for reliable spatial geometry from the STOCHASTIC rollouts (renders are untrustworthy).
    "single_gate_static": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 15.0, "course_spawn_dist_hi": 15.0,   # FIXED 15 m (override _COMMON 10-20)
        "course_drop_lo": 0.0, "course_drop_hi": 0.0,                 # LEVEL gate (no climb) -- isolate
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # MPCC-CLEAN CONTOURING LEVER (Fengyou greenlight 2026-07-08, hover-hold-confirmed): the fix for the
    # single_gate_static 100% floor-dive. SAME fixed 15 m level dead-ahead gate as single_gate_static (so
    # the box-exit read is apples-to-apples vs the 100%-floor baseline), but the reward is decomposed MPCC-
    # style: (1) ALONG-TRACK LAG -- rw_progress_to_center=False so progress = segment_arc_position (forward
    # advance along the spawn->gate line ONLY; perpendicular drift earns 0 progress); (2) PBRS CONTOURING --
    # rw_corridor supplies the vertical+lateral homing onto the line that the lag omits, REPLACING the
    # isotropic gate_center_potential's vertical component (no double-count -> no fighting k-sweep). The
    # hover-hold probe (job 3297613: 91% hover, alt_err 1.2m) proved altitude control is LEARNABLE from a
    # clean vertical gradient; this delivers that gradient during forward transit. SUCCESS (box-exit
    # classifier): exit_floor collapses 100%->~0, cross_offset_m 9.7->0, exit_thread rises (leading
    # indicator exit_plane_miss up first = crosses the plane in-bounds before centring). WATCH exit_ceiling
    # (vertical over-correction) -> if high, drop rw_corridor. k=2.0 = parity with the lag rw_progress; needs
    # a sweep. OFF-LADDER; run standalone via STAGES=single_gate_static_mpcc.
    "single_gate_static_mpcc": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 15.0, "course_spawn_dist_hi": 15.0,   # fixed 15 m (== the sgs baseline)
        "course_drop_lo": 0.0, "course_drop_hi": 0.0,                 # level gate
        "rw_progress_to_center": False,   # ALONG-TRACK LAG (segment-arc), NOT the isotropic 3D norm
        "rw_corridor": 2.0,               # PBRS CONTOURING ON (perpendicular homing; parity w/ lag rw_progress)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # HOVER-HOLD PROBE (Fengyou greenlight 2026-07-08): the H1-vs-H2 disambiguator for the single_gate_static
    # 100% floor-dive. NO gate homing/passage/centering -- the reward is a give-up-RESISTANT positive
    # altitude-hold bonus (ego_reward.altitude_hold_reward) ONLY, so holding the spawn altitude is the UNIQUE
    # optimum with NO competing forward objective. A fixed 15 m LEVEL gate is present (course_n_gates=1) but
    # IGNORED by the reward; standing start; terminals = floor(=contact)/oob/timeout unchanged. READ from the
    # box-exit classifier: OUTCOME A -> exit_floor collapses to ~0 + exit_timeout dominant + alt_err_m
    # sub-metre = altitude-hold IS learnable from a clean vertical gradient (H1 / missing-early-gradient;
    # GREEN-LIGHT the contouring/reference lever). OUTCOME B -> exit_floor stays high = a CONTROL-LEARNING
    # basin the policy will not climb out of regardless of gradient (H2; PIVOT to a hover warm-start /
    # action-bias-toward-hover, do NOT add another reward term). OFF-LADDER (not in STAGE_ORDER); run
    # standalone via STAGES=hover_hold.
    "hover_hold": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 15.0, "course_spawn_dist_hi": 15.0,   # fixed 15 m level gate (ignored)
        "course_drop_lo": 0.0, "course_drop_hi": 0.0,
        # ZERO every gate-homing term -> pure altitude-hold, no forward pull, no passage/centering/exit.
        "rw_progress": 0.0, "rw_passage": 0.0, "rw_passage_increment": 0.0,
        "rw_area_dist_ref_m": 0.0, "rw_centering": 0.0, "rw_exit_align": 0.0,
        # the give-up-resistant POSITIVE hold bonus ON: spawn altitude is the unique optimum.
        "rw_altitude_hold": 1.0, "rw_altitude_hold_band_m": 8.0,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # 2. HANDOFF_DRILL (DESIGN.md §D): 2 gates, focus the FIRST gate handoff. Spacing 10-20 m (the VQ2
    #    co-visibility regime where the next gate is trackable through the current one). Still the EASY
    #    reward (no rw_passage bump, exit OFF) -- drill the handoff MECHANICS (window promotion, no
    #    teleport) before adding turning pressure. ~1500 updates.
    "handoff_drill": {
        **_COMMON,
        "course_n_gates": 2,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA},
    },
    # 3. DUAL_GATE_FULL (HARD/turning): 2 gates, full drop band, spacing 10-20 m. STAGE-SPECIFIC hard
    #    knob turns ON here (NOT in _COMMON): a small exit_align (next-gate exit-line, gate-gated once/
    #    pass -> non-farmable -> safe). The passage centering basin is now the _COMMON base-5 + per-gate
    #    increment (the old x3 flat bump is superseded; it would have UNDERCUT the new base 5). ~5000-6000.
    "dual_gate_full": {
        **_COMMON,
        "course_n_gates": 2,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "rw_exit_align": 0.1,        # STAGE-SPECIFIC: next-gate exit-line (gate-gated once/pass -> safe)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA},
    },
    # 4. MULTI_GATE (HARD/turning): N gates, spacing 10-20 m, the full lap. Same hard-stage exit-line as
    #    dual_gate_full. n_envs 4096 (set in the sbatch for throughput). ~5000-6000 updates.
    "multi_gate": {
        **_COMMON,
        "course_n_gates": 6,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "rw_exit_align": 0.1,        # STAGE-SPECIFIC (hard turning)
        "_raw": {"env.max_time": 100, "algo.gamma": _GAMMA},
    },
}

# The ordered ladder (DESIGN.md §D/§7): single_gate -> handoff_drill -> dual_gate_full -> multi_gate.
# blackout_pass is DELETED (emergent from the keypoint model).
STAGE_ORDER = ("single_gate", "handoff_drill", "dual_gate_full", "multi_gate")

# Keys forwarded to the course sampler (documented here so the sbatch/renderer + any future env wiring
# agree on the contract; they are additive `+env.` cfg keys -- unset => the sampler's defaults). This
# mirrors the inc8 curriculum's COURSE_SAMPLER_KEYS convention.
COURSE_SAMPLER_KEYS = ("course_n_gates", "course_seg_len_lo", "course_seg_len_hi",
                       "course_drop_lo", "course_drop_hi",
                       "course_spawn_dist_lo", "course_spawn_dist_hi",
                       "course_spawn_heading")

# The reward knobs that are HARD-STAGE-ONLY (must NEVER appear in _COMMON / never hit single_gate or
# handoff_drill). Named so the test can assert the B2b scoping discipline structurally.
_HARD_ONLY_REWARD_KEYS = ("rw_passage_bump", "rw_exit_align_on")   # semantic sentinels; see tests

# The stages that are allowed to carry hard-stage reward pressure (turning stages).
_HARD_STAGES = ("dual_gate_full", "multi_gate")
# The stages that MUST stay on the clean champion reward (discovery / easy).
_EASY_STAGES = ("single_gate", "handoff_drill")


def render_overrides(stage: str, prefix: str = "+env.") -> list[str]:
    """Render a stage's overrides as hydra CLI tokens (e.g. '+env.course_n_gates=2'). ``prefix`` is
    applied to every key EXCEPT the "_raw" sub-dict, whose entries are emitted verbatim (existing
    non-env keys like algo.gamma / env.max_time take no '+'). Booleans render lowercase (hydra-true).
    Raises on an unknown stage so a typo in the sbatch fails loudly."""
    if stage not in STAGES:
        raise ValueError(f"unknown ego curriculum stage {stage!r}; valid: {sorted(STAGES)}")
    toks = []
    for k, v in STAGES[stage].items():
        if k == "_raw":
            for rk, rv in v.items():
                toks.append(f"{rk}={_fmt(rv)}")
        else:
            toks.append(f"{prefix}{k}={_fmt(v)}")
    return toks


def _fmt(v) -> str:
    """Render a scalar as a hydra token (bool -> lowercase true/false; everything else -> str)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "multi_gate"
    print(" ".join(render_overrides(stage)))
