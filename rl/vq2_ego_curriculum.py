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
    # --- refined-B reward EASY-SAFE defaults (safe/HELPFUL on single_gate) ---
    # RC3 DISCOVERY REBALANCE (2026-07-07): rw_progress 1.0->6.0 (a STRONG dense pull toward the now-
    # visible gate -- with the camera-flip RC1 fix the gate is finally in the obs) and miss/oob terminal
    # base 200->30 (progress-scaled: approach-then-miss nets -base = ~-30 ~= hover, so imperfect approach
    # attempts are no longer catastrophic; passing still nets ~+158). CONTACT base stays 200 (zero-contact
    # rule; sprint-and-clip stays defeated by construction: contact forfeits banked + 200). These are
    # DISCOVERY-FRIENDLY -> _COMMON is correct (they HELP single_gate; NOT the B2b hard-stage-leak footgun).
    "rw_progress": 6.0,              # dense segment-projected progress (was 1.0 -- too weak to pull to gate)
    "rw_passage": 1.0,              # L-inf centering passage, base scale (bumped ONLY on hard stages)
    "rw_terminal_base": 200.0,       # kill-on-contact base magnitude (CONTACT stays catastrophic)
    "rw_terminal_miss": 30.0,        # fly-by MISS (wide gate attempt, stays in-arena): ~hover -> forgiving
    # OOB (leaving the ARENA) STAYS discouraged at 200 -- distinct from MISS. Measured 2026-07-07: at
    # oob=30 the drone had NO pressure to stay in bounds and 96% of episodes OOB'd at ~2 s (l_episode
    # 2 s vs the 33 s in-bounds hover at oob=200), never reaching the gate. The RC3 fix forgives the
    # GATE ATTEMPT (miss), NOT leaving the arena. Approaching the gate stays in-bounds -> not punished.
    "rw_terminal_oob": 200.0,        # out-of-bounds = leaving the arena -> stays discouraged (NOT ~hover)
    "rw_terminal_progress_scaled": True,  # forfeit banked progress + base -> sprint-and-clip never wins
    "rw_exit_align": 0.0,           # next-gate exit-line OFF on easy stages (ON only hard stages below)
    "rw_rate": 1.0e-3, "rw_dact": 1.0e-3,  # MINUSCULE smoothness (3-4 orders below progress)
    "rw_tilt_free_rad": 1.0471976,   # FIXED 60 deg free cone
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
    #    knobs turn ON here (NOT in _COMMON): rw_passage x3 (thin centering margin through a sharp turn
    #    at race speed) + a small exit_align (next-gate exit-line, gate-gated -> safe). ~5000-6000 updates.
    "dual_gate_full": {
        **_COMMON,
        "course_n_gates": 2,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "rw_passage": 3.0,           # STAGE-SPECIFIC: wider centering basin for the sharp turn (NOT _COMMON)
        "rw_exit_align": 0.1,        # STAGE-SPECIFIC: next-gate exit-line (gate-gated once/pass -> safe)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA},
    },
    # 4. MULTI_GATE (HARD/turning): N gates, spacing 10-20 m, the full lap. Same hard-stage reward
    #    knobs as dual_gate_full. n_envs 4096 (set in the sbatch for throughput). ~5000-6000 updates.
    "multi_gate": {
        **_COMMON,
        "course_n_gates": 6,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "rw_passage": 3.0,           # STAGE-SPECIFIC (hard turning)
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
                       "course_drop_lo", "course_drop_hi")

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
