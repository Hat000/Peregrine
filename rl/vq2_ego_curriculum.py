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
    # TERMINAL EQUALIZATION (Fengyou 2026-07-08): if a wide MISS is punished FAR less than a gate CONTACT,
    # bailing wide becomes a strictly-safer play than committing to thread -> the policy learns to AVOID the
    # gate (the risk). Fix: bring the miss and contact BASES to parity (both catastrophic DQ-scale) so
    # bailing is no longer a cheap escape. CONTACT lowered 200 -> 100 (Fengyou-authorized: -100 is still
    # unmistakably a DQ-level penalty, and it buys headroom to raise the miss toward it). MISS raised 8 ->
    # 100 to MATCH. The residual advantage of a miss over a contact is now ONLY the contact's banked-
    # progress FORFEIT (~30 max on a single 8-15 m gate) -- the irreducible sprint-and-clip defence, which
    # must stay contact-only -- shrinking the old bail-incentive gap (~252) to ~banked (~30). Ordering
    # holds: clean pass > miss > contact.
    "rw_terminal_base": 100.0,       # kill-on-contact base (DQ-scale; lowered 200->100 for miss-parity)
    "rw_terminal_miss": 100.0,       # fly-by MISS raised 8->100 == contact base (no cheap bail)
    # OOB (leaving the ARENA) STAYS discouraged at 200 -- distinct from MISS. Measured 2026-07-07: at
    # oob=30 the drone had NO pressure to stay in bounds and 96% of episodes OOB'd at ~2 s (l_episode
    # 2 s vs the 33 s in-bounds hover at oob=200), never reaching the gate. The fix forgives the GATE
    # ATTEMPT (miss), NOT leaving the arena. NOTE: a below-FLOOR dive is NOT oob -- the ego env now
    # reclassifies it as a CONTACT (terminal_base=100, a crash/DQ), so oob here = lateral/ceiling only.
    # OOB kept at 200 (NOT lowered to the new 100 miss/contact parity): it is the anti-arena-exit wall
    # (at oob=30, 96% of episodes OOB'd at ~2 s), a separate failure mode from the gate-avoidance one the
    # miss/contact parity addresses; the GVF line lives inside the arena so this rarely binds anyway.
    "rw_terminal_oob": 200.0,        # out-of-bounds = leaving the arena -> stays discouraged (NOT ~hover)
    "rw_terminal_progress_scaled": True,  # forfeit banked progress + base -> sprint-and-clip never wins
    "rw_exit_align": 0.0,           # next-gate exit-line OFF on easy stages (ON only hard stages below)
    "rw_rate": 1.0e-3, "rw_dact": 1.0e-3,  # MINUSCULE smoothness (3-4 orders below progress)
    "rw_tilt_free_rad": 1.0471976,   # FIXED 60 deg free cone
    # HOVER-HOLD probe bonus OFF on every real stage (turned ON only by the hover_hold diagnostic stage).
    "rw_altitude_hold": 0.0,         # give-up-resistant spawn-altitude bonus; 0 == OFF
    "rw_altitude_hold_band_m": 8.0,  # (m) decay half-width (unused while rw_altitude_hold==0)
    # MPCC CONTOURING OFF by default (superseded by the anisotropic vertical weight below; kept as a
    # 0-knob, not deleted -- the PBRS-rate form was too weak/policy-invariant to escape the sink basin).
    "rw_corridor": 0.0,              # PBRS perpendicular-contouring weight; 0 == OFF
    "rw_corridor_clip_mps": 39.0,    # contouring clip band (m/step = mps*dt); trims gate-handoff bursts
    # ANISOTROPIC VERTICAL WEIGHT isotropic (1.0) by default -> byte-compatible with the isotropic norm on
    # every real stage; the floor-dive lever (single_gate_static_aniso) cranks it to un-bury the vertical.
    "rw_progress_vert_weight": 1.0,  # 1.0 == isotropic Euclidean; >1 up-weights the vertical (Z) axis
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
    # VARIED-GATE STAGE (Fengyou 2026-07-08): the gate spawns at VARIABLE distance (10-20 m) AND VARIABLE
    # HEIGHT (+-6 m via spawn_below_g0) so it appears at different positions in the FOV every episode. Two
    # purposes: (1) forces the policy to USE its inputs (the gate rel_pos) -- it can't memorise one static
    # open-loop trajectory the way a fixed gate allows; (2) UN-BURIES the vertical GEOMETRICALLY -- a gate
    # at varying heights makes the distance-to-gate's vertical component non-trivial from the start (the
    # flat same-height gate is the degenerate worst case for the floor-dive). ISOTROPIC reward inherited
    # from _COMMON (rw_progress_vert_weight=1) so this is a CLEAN test of whether geometry variation ALONE
    # fixes the dive; override +env.rw_progress_vert_weight=25 to test the geometry+anisotropic combo.
    # OFF-LADDER; run standalone via STAGES=single_gate_varied.
    "single_gate_varied": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,         # varied range (Fengyou 2026-07-08: 8-15 m)
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,  # gate +-6 m in HEIGHT (the un-burier)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # VARIED + MODERATE-ANISO COMBO (Fengyou 2026-07-08): the synthesis of the two partial successes. The
    # varied-gate (geometry) run un-buried the vertical GEOMETRICALLY (floor 100%->20% still dropping, alt_err
    # 13->7m) and shifted the failure floor-dive -> WIDE-MISS (reaches the plane, crosses ~7m low), but
    # ISOTROPIC left the vertical only partially un-buried. Flat + aniso w=25 fully un-buried but OVER-
    # corrected (80% backward drift). This combines both at LOW w: geometry does most of the un-burying, a
    # MILD vertical weight (w=6, vs the flat case's 25) sharpens the residual 7m without starving forward.
    # Same varied geometry (dist 8-15 m, height +-6 m). Longer budget (varied-isotropic was still improving
    # at 1500). OFF-LADDER; run standalone via STAGES=single_gate_varied_aniso.
    "single_gate_varied_aniso": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "rw_progress_to_center": True,
        "rw_progress_vert_weight": 6.0,   # MODERATE (geometry already un-buries; low w avoids w=25 over-correction)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # DECOUPLED MPCC lag+corridor on the VARIED gate (adversarial 2026-07-08): raced HEAD-TO-HEAD vs the
    # COUPLED aniso above. The aniso couples forward+vertical in one potential (raising w steals forward ->
    # the w=25 exit_BACK over-correction); the MPCC DECOMPOSITION splits them -- forward = the along-track
    # LAG (rw_progress on segment_arc, progress_to_center=False, undiminished off-altitude) + vertical/lateral
    # homing = the SEPARATE PBRS corridor (rw_corridor). So raising corridor-k to kill the floor does NOT
    # touch forward -> dissolves the aniso knife-edge. The prior corridor FAILURE (single_gate_static_mpcc,
    # k=2, 100% floor) was a MAGNITUDE undershoot (per-step sink penalty 0.27 < forward 0.4), a monotonic
    # raise-k fix; PBRS telescoping means higher k adds NO give-up/standing-tax. k=4 (vs the failed 2). Same
    # varied geometry as the aniso combo. OFF-LADDER; run via STAGES=single_gate_varied_mpcc.
    "single_gate_varied_mpcc": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "rw_progress_to_center": False,   # ALONG-TRACK LAG (decoupled forward drive)
        "rw_corridor": 4.0,               # SEPARATE PBRS vertical/lateral homing (raised from the failed k=2)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # VECTOR-FIELD (GVF) RACING LINE on the VARIED gate (Fengyou greenlight 2026-07-08 -- branch 3 of the
    # 3-branch split; aniso=fallback, mpcc=hard-line, THIS=vector field). Builds an ONLINE non-optimal
    # HERMITE line spawn->gate that arrives HEAD-ON (tangent == gate normal), then measures BOTH reward
    # potentials against that CURVED line instead of the straight segment: progress = arc length of the
    # nearest point (rw_progress_to_center=False routes _progress_scalar to the line's s), contouring =
    # cross-track distance to the line (rw_corridor). This IS Fengyou's dot(v, F) field -- F = line tangent
    # + k*(cross-track pull) -- expressed as the two telescoping potentials (farm-proof; no lag ref to
    # outrun). DIFFERS from single_gate_varied_mpcc ONLY in use_racing_line: mpcc's straight spawn->gate
    # segment arrives at the gate at an ANGLE when the gate is off-axis (the +-6 m height => a vertical
    # angle), whereas the GVF line CURVES to arrive level/head-on -- so the vertical homing target is the
    # floor-dive-correct geometry. k=4 parity with mpcc for a clean A/B. OFF-LADDER; run via
    # STAGES=single_gate_varied_gvf.
    "single_gate_varied_gvf": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,          # progress s + contouring perp read from the CURVED head-on line
        "rw_progress_to_center": False,   # (ignored under use_racing_line; s comes from the line arc length)
        "rw_corridor": 4.0,               # cross-track contouring pull onto the racing line (parity w/ mpcc)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # HOMING + LINE-CONTOURING (Fengyou 2026-07-08 race read -- the data-indicated winner). The vaniso/vmpcc
    # race proved ISOTROPIC 3D HOMING (progress_to_center, w=1) is the ONLY form that reaches the gate plane
    # (varied-isotropic 70% plane-reach, floor 50->20%, STILL improving at cutoff), while BOTH "fixes" that
    # touched the PROGRESS term regressed it: aniso (couples the vertical -> rotates the forward gradient ->
    # 97% side) and mpcc (drops homing for segment-lag -> 60% floor). The residual gap is a pure CENTERING
    # deficit (crosses ~7 m off). So KEEP the isotropic homing that reaches the plane and ADD ONLY the line's
    # DECOUPLED cross-track contouring (perpendicular pull, does NOT touch the forward homing direction) to
    # center the 7 m. racing_line_progress=False routes progress back to the isotropic gate_center_potential
    # while the line supplies perp for rw_corridor. This is the untested cell: homing + decoupled contouring
    # (== Fengyou's "line tracking + distance from gate"). OFF-LADDER; run via STAGES=single_gate_varied_gvfh.
    "single_gate_varied_gvfh": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,          # line supplies the cross-track perp (contouring) ...
        "racing_line_progress": False,    # ... but progress STAYS on isotropic homing (the plane-reacher)
        "rw_progress_to_center": True,    # isotropic 3D distance-to-centre homing (w=1, the un-buried driver)
        "rw_progress_vert_weight": 1.0,   # ISOTROPIC (varied geometry already supplies the vertical gradient)
        "rw_corridor": 4.0,               # DECOUPLED cross-track centering onto the line (vertical + lateral)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA},
    },
    # GVF + NOISE ANNEAL (Fengyou 2026-07-08, the THREAD-RATE endgame): vgvf (pure GVF) is the winning
    # trajectory (plane 83%, floor 3%, xoff 10->5) but PLATEAUED at ~3.9m vertical + ~3.1m lateral off --
    # too wide to thread a 0.75m aperture. The held std-0.30 exploration noise is a STRUCTURAL cap: the
    # DEPLOYED (deterministic mean) policy can't sharpen below the noise smear. This holds 0.30 for the
    # first HALF (learn the field like vgvf did), then ANNEALS to 0.05 over the back half to SHARPEN the
    # mean toward a threading crossing. `_raw` ++overrides the sbatch BASE's flat-0.30 noise knobs. Isolates
    # the anneal (k=4, same as vgvf) so a win attributes to sharpening, not stronger centering. Run via
    # STAGES=single_gate_varied_gvf_anneal, UPD_single_gate_varied_gvf_anneal=4000.
    "single_gate_varied_gvf_anneal": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,   # progress from the line arc-length (the winning GVF form)
        "rw_corridor": 4.0,               # SAME k as vgvf (isolate the anneal)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    # GVF + STRONGER CONTOURING (k=10) + NOISE ANNEAL (Fengyou 2026-07-08, the go-for-90% run): combines
    # the two thread-rate levers -- stronger cross-track centering (k 4->10 to pull the ~3-4m/axis residual
    # tighter) AND the endgame anneal (sharpen the mean). If this threads and the anneal-alone run does not,
    # the extra centering was needed; if both thread, anneal was sufficient. Run via
    # STAGES=single_gate_varied_gvf_k10a, UPD_single_gate_varied_gvf_k10a=4000.
    "single_gate_varied_gvf_k10a": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 10.0,              # STRONGER cross-track centering (up from the plateaued k=4)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    # GVF + SQUARE-ON coupling (Fengyou night-lever 2026-07-08 backup): the winning GVF field + k=10 + anneal
    # PLUS the distance-gated AREA coupling (rw_area_dist_ref_m>0) turned back ON. It scales the POSITIVE
    # progress by how square-on the gate is when CLOSE (factor = area + (1-area)*clip(dist/ref)): a shallow
    # off-axis final approach earns less -> the policy squares up before the gate = a HEAD-ON centred crossing
    # (complements the contouring's position-centering with an ANGLE signal). Was off (reward-audit taxed the
    # buried homing); safe to revisit now the homing works. Run via STAGES=single_gate_varied_gvf_squareon.
    "single_gate_varied_gvf_squareon": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 10.0,
        "rw_area_dist_ref_m": 6.0,        # SQUARE-ON coupling ON (head-on-arrival angle signal)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    # GVF + HARDER PUSH (Fengyou night-lever 2026-07-08 backup): k=15 (stronger centering than k10) + a SHARPER
    # anneal floor (0.02 vs 0.05) to squeeze the deterministic mean tighter toward the 0.75m aperture, if k10+
    # 0.05 plateaus above 90%. Run via STAGES=single_gate_varied_gvf_k15.
    "single_gate_varied_gvf_k15": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 15.0,             # even stronger cross-track centering
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.02, "++algo.noise_hold_frac": 0.5},
    },
    # GVF + MAGNITUDE CENTERING (Fengyou 2026-07-08, the standing-offset lever). vganl (GVF+anneal, k=4)
    # solved the floor (0.7%) + reaches the plane (81%) but the crossing offset PLATEAUED at ~4.4m (3.4m
    # vert + 3.2m lat) and the ANNEAL did NOT tighten it -> the offset is a MEAN-POLICY limit, not noise. ROOT:
    # the PBRS contouring TELESCOPES (rewards REDUCING perp), so a SUSTAINED 4.4m offset earns ZERO gradient
    # -- the policy parks there. FIX: add the DENSE MAGNITUDE centering penalty (through_centering_reward =
    # -rw_centering*clamp(perp,0,max)) on the line perp -- it NAGS a standing offset continuously (the gradient
    # PBRS lacks). Clamp WIDENED to 6m so the gradient spans the whole 4.4m (default 2m would be flat past 2m).
    # Moderate weight (0.4) so it nags without dominating the progress (~2/step) or triggering the give-up/OOB
    # back-fire (now countered by the miss=100 terminal). Keeps k=4 PBRS (guides the approach) + anneal. FRESH
    # 4000 upd. Run via STAGES=single_gate_varied_gvf_ctr.
    "single_gate_varied_gvf_ctr": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,               # PBRS contouring (fast approach guide) -- keep k=4 (k10 collapsed)
        "rw_centering": 0.4,              # NEW: dense MAGNITUDE centering (nags the standing offset)
        "rw_centering_max_m": 6.0,        # clamp widened past the 4.4m offset so the gradient spans it
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    # WARM-START centering (Fengyou 2026-07-08, efficient): CONTINUE from vganl's checkpoint (floor solved,
    # parked at 4.4m) and apply the magnitude centering to sharpen the standing offset -- applying the penalty
    # AFTER the field is learned avoids the early-training give-up risk a fresh magnitude penalty carries.
    # Starts at LOW noise (warm base) and anneals lower; 2500 upd (from a good base). +init_from loads the
    # vganl actor+critic .pth. Run via STAGES=single_gate_varied_gvf_ctrw.
    "single_gate_varied_gvf_ctrw": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4,
        "rw_centering_max_m": 6.0,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_anneal_seed0_vganl/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.35},
    },
    # BOTH-FIXES centering (Fengyou 2026-07-08): warm-start from vganl + MAGNITUDE centering (the integral/
    # standing-offset gradient the telescoping PBRS lacks) + FRAME-MOAT fix (frame_clip_is_miss -> a frame-clip
    # nets == a wide miss, so the ring around the aperture stops punishing getting-close and the CENTRE becomes
    # attractive). Isolation control = ctrw (centering only, no frame fix). Run via
    # STAGES=single_gate_varied_gvf_ctrf.
    "single_gate_varied_gvf_ctrf": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4,
        "rw_centering_max_m": 6.0,
        "frame_clip_is_miss": True,       # FRAME-MOAT fix: frame-clip nets == wide miss (centre attractive)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_anneal_seed0_vganl/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.35},
    },
    # APERTURE CURRICULUM stage 1 (Fengyou 2026-07-08 structural lever for the 3.4m plateau). Warm-start from
    # vgctrw (best centering, parked at 3.4m) and WIDEN the training aperture to 8m (half 4m) so the passage
    # reward gives a gradient across the current offset -- the real 0.75m aperture leaves a DEAD ZONE beyond it
    # that the telescoping field can't bridge. WATCH cross_offset_m (aperture-independent), NOT thread (which is
    # inflated at a wide aperture). If the wide target pulls xoff below 3.4m -> the shrink ladder (ap8->ap4->
    # ap2->ap1.5-real, each warm-started) converts it to real threads; if even a wide aperture can't centre, the
    # limit is CONTROL not reward. Run via STAGES=single_gate_varied_gvf_ap8.
    "single_gate_varied_gvf_ap8": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4,
        "rw_centering_max_m": 6.0,
        "gate_inner_opening_m": 8.0,      # WIDE training aperture (half 4m) -> passage gradient at 3.4m
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_ctrw_seed0_vgctrw/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.35},
    },
    # GVF DIRECTION-ALIGNMENT reward (Fengyou 2026-07-08 -- the TRUE vector field). Progress (along the line) +
    # dot(v_hat, F_hat) alignment REPLACING the telescoping contouring: rewards the velocity DIRECTION following
    # the guiding field everywhere, so a parallel-flying standing offset (the contouring blind spot) is still
    # pressured onto the path. rw_align_gain = the CONVERGENCE TIGHTNESS (Fengyou's "how sharp a turn onto the
    # line"). Two-value SWEEP: SMOOTH (gain 0.5, wide asymptotic curve) vs SHARP (gain 3.0, whip onto the line).
    # FRESH (the align reward is a direction signal, not a magnitude penalty, so it does not back-fire fresh the
    # way the magnitude centering did; and it penalises diving early -> more stable than fresh centering). No
    # corridor/centering (align is the cross-track mechanism now). Run via STAGES=single_gate_varied_gvf_align_smooth
    # / single_gate_varied_gvf_align_sharp.
    "single_gate_varied_gvf_align_smooth": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,   # along-track = line arc-length progress (speed incentive)
        "rw_corridor": 0.0,               # align REPLACES the telescoping contouring
        "rw_centering": 0.0,
        "rw_align": 2.0,                  # GVF direction-alignment weight
        "rw_align_gain": 0.5,             # SMOOTH convergence (wide asymptotic curve)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    "single_gate_varied_gvf_align_sharp": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 0.0,
        "rw_centering": 0.0,
        "rw_align": 2.0,
        "rw_align_gain": 3.0,             # SHARP corner onto the line (steep inward angle)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    # PARABOLA + LATERAL-FOV + ALIGN (Fengyou 2026-07-08 -- the combined upgrade): GVF field + dot(v_hat,F_hat)
    # alignment (cross-track) + the SMOOTH PARABOLIC crossing reward (replaces the thread/clip/miss cliff -> no
    # moat, smooth centering gradient) + LATERAL FOV variation (spawn_yaw_jitter so the gate appears left/right
    # across the view, not just dead-ahead -- realistic). Two align tightnesses swept. Fresh. Run via
    # STAGES=single_gate_varied_gvf_para_smooth / single_gate_varied_gvf_para_sharp.
    "single_gate_varied_gvf_para_smooth": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.35,  # +-20deg lateral FOV variation (gate appears left/right of centre)
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 0.0, "rw_centering": 0.0,
        "rw_align": 2.0, "rw_align_gain": 0.5,   # SMOOTH convergence
        "rw_parabola_crossing": True,     # smooth parabola replaces passage + frame/miss terminals
        "rw_cross_center": 20.0, "rw_cross_zero_m": 0.75, "rw_cross_neg_cap": 100.0,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    "single_gate_varied_gvf_para_sharp": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.35,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 0.0, "rw_centering": 0.0,
        "rw_align": 2.0, "rw_align_gain": 3.0,   # SHARP convergence
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 0.75, "rw_cross_neg_cap": 100.0,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    # CHAMPION + PARABOLA (Fengyou 2026-07-08, isolated): warm-start the CHAMPION recipe (field + contouring
    # k4 + magnitude centering 0.4 -- the best centering to date, xoff 3.4m) from vgctrw's checkpoint and add
    # ONLY the smooth PARABOLIC crossing (replaces the passage + frame/miss cliff -> no moat, smooth final-
    # centering gradient). NO align, NO lateral (the vector field underperformed; lateral needs its own warm-
    # start). critic_warmup 400 so the warm-started critic re-adapts to the parabola crossing reward before the
    # actor moves (avoids the frame-fix-style value-mismatch collapse). Run via STAGES=single_gate_varied_gvf_cpara.
    "single_gate_varied_gvf_cpara": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,               # champion contouring (approach)
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,   # champion magnitude centering (approach)
        "rw_parabola_crossing": True,     # NEW: smooth parabola crossing (no moat, sharpens the crossing)
        "rw_cross_center": 20.0, "rw_cross_zero_m": 0.75, "rw_cross_neg_cap": 100.0,
        # NOTE: the sbatch BOUNDARY_OV always appends `+critic_warmup_updates=100`; setting it AGAIN here
        # (as `++...=400`) is a Hydra APPEND-COLLISION ("item already at critic_warmup_updates") that killed
        # the 2026-07-08 vgcp launch (RC=1, no tfevents). Inherit the 100 default instead of re-adding it.
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_ctrw_seed0_vgctrw/checkpoints",
                 "++algo.noise_std_hold": 0.10, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.35},
    },
    # ================================================================================================
    # LATERAL-FOV AXIS (Fengyou 2026-07-08): the champion lineage (vgvf->vganl->vgctrw) varied only gate
    # DEPTH (8-15m) and HEIGHT (+-6m) with spawn_yaw_jitter=0 -- so the gate ALWAYS appeared dead-centre
    # horizontally and the body-frame obs channel rel_pos[1] (lateral) was ~0 in 100% of training. The
    # policy can SEE an off-axis gate (rel_pos is a full metric FLU vector) but was never rewarded for
    # centring one. This is the untrained axis + a likely contributor to the 3.4m crossing plateau, and
    # it is the honest prerequisite before multigate (after a turn the next gate appears off-axis). Two
    # runs isolate the SAME new axis (course_spawn_yaw_jitter, +-14deg, within HFOV so the gate stays
    # visible): _lat = FRESH on the known-good vganl recipe (GVF + anneal, k=4, NO magnitude centering --
    # fresh magnitude centering back-fires, vgctr 100% floor); _latw = WARM-START the champion vgctrw
    # (its 3.4m centring skill intact) and only WIDEN the distribution. Distribution-widening is a gentler
    # change than the reward-SEMANTIC changes that collapsed warm-starts (frame-moat/wide-aperture), so
    # _latw may hold where those floored. WATCH cross_offset_m + exit_side (lateral misses).
    # Run via STAGES=single_gate_varied_gvf_lat / single_gate_varied_gvf_latw.
    # ================================================================================================
    "single_gate_varied_gvf_lat": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,  # +-14deg lateral FOV variation (the UNTRAINED axis)
        "use_racing_line": True,
        "rw_progress_to_center": False,   # GVF line-arc progress (vganl recipe -- reaches the plane fresh)
        "rw_corridor": 4.0,               # k=4 contouring (floor solved, no fresh magnitude-centering risk)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "++algo.noise_std_floor": 0.05, "++algo.noise_hold_frac": 0.5},
    },
    "single_gate_varied_gvf_latw": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,  # SAME new axis, warm-started onto the champion's centring skill
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,   # champion magnitude centering (now nags LATERAL too)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_ctrw_seed0_vgctrw/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.35},
    },
    # ================================================================================================
    # LATERAL + PARABOLA-CENTERING (Fengyou 2026-07-08, the unified attack). vglat/vglatw proved the DISEASE:
    # with lateral jitter the drone REACHES the plane (75%, unchanged from champion) but crosses ~6m off --
    # it flies forward and lets the whole lateral offset pass UNCORRECTED (ignores rel_pos[1]). Same root as
    # the champion's 3.4m no-jitter plateau: the per-step centering is too WEAK to beat the progress "fly
    # forward fast" incentive. FIX = Fengyou's smooth PARABOLA crossing reward (+cross_center at centre ->
    # 0 at the zero radius -> negative outside) as a STRONG terminal centering gradient the per-step terms
    # lack. Key: the zero radius must be WIDE (~6m) so there is gradient across the current 6m offset (at the
    # real 0.75m aperture it is a DEAD ZONE out there). Warm from champion, lateral ON. Success = xoff DROPS
    # below the 6m plateau (proving the parabola centres); then SHRINK cross_zero_m (6->4->3->1.5->0.75) over
    # warm-started rungs to strengthen the near-centre gradient toward real threads. Sweep the zero radius via
    # EXTRA=++env.rw_cross_zero_m=4.0 (default 6.0). Run via STAGES=single_gate_varied_gvf_lpara.
    # ================================================================================================
    "single_gate_varied_gvf_lpara": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,  # lateral FOV on (the axis the champion ignores)
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,                              # PBRS contouring approach guide (kept)
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,  # per-step magnitude nag (kept)
        "rw_parabola_crossing": True,                    # NEW strong terminal centering gradient
        "rw_cross_center": 20.0, "rw_cross_zero_m": 6.0, "rw_cross_neg_cap": 100.0,  # WIDE zero spans the ~6m offset
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_ctrw_seed0_vgctrw/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.35},
    },
    # ZERO-SHRINK LADDER rung 2 (Fengyou 2026-07-08): vglp4 (lpara zero=4.0) BROKE the plateau -- xoff
    # 6.0->2.2m WITH lateral jitter on, thread 0->4.3% and still improving (the wide zero=6.0 vglp6 FAILED:
    # too flat near centre, drifted to 10m -- so the zero must be TIGHT enough to make a centring gradient).
    # This rung tightens the parabola zero 4.0->3.0 and WARM-STARTS from vglp4's final checkpoint so the
    # centring COMPOUNDS. Principle: keep the zero ~1.3-1.5x the current offset so the drone sits in the
    # POSITIVE region with a live gradient (zero=3.0 at 2.2m offset -> para=+9.2, safe; too-tight punishes
    # early -> give-up/floor). Continue 3.0->2.0->1.5->0.75 (real aperture) as xoff falls. Lateral stays on.
    # Run via STAGES=single_gate_varied_gvf_lpara3.
    "single_gate_varied_gvf_lpara3": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 3.0, "rw_cross_neg_cap": 100.0,  # tightened 4.0->3.0
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.35},
    },
    # IN-RUN ZERO ANNEAL (Fengyou 2026-07-08, the cure for the discrete-shrink collapse). vglp3/vglp3b PROVED
    # a discrete 4->3 zero STEP on warm-start detonates the policy (jumps to 8m, recovers only to ~6m -- the
    # reward for the current crossings drops/goes negative before the critic recalibrates). FIX = shrink the
    # parabola zero CONTINUOUSLY within ONE run (peregrine_train_ego._cross_zero_schedule mutates
    # env._egorw.cross_zero_m per update): NO discontinuity, the near-centre gradient sharpens smoothly as the
    # policy centres. Warm from vglp4 (already ~2m at zero=4) and anneal the zero 4.0->0.75 (real aperture) over
    # the back 90% of a LONGER 4000-upd run. Lateral stays on. Success = xoff -> sub-1m + thread climbing.
    # Run via STAGES=single_gate_varied_gvf_lpara_anneal, UPD_single_gate_varied_gvf_lpara_anneal=4000.
    "single_gate_varied_gvf_lpara_anneal": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,  # initial (anneal overrides live)
        "cross_zero_anneal": True, "cross_zero_start": 4.0, "cross_zero_end": 0.75, "cross_zero_hold_frac": 0.1,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    # ENDGAME FINE-TUNE (Fengyou 2026-07-08): vglpan/vglpan6 (the 4->0.75 anneal) BOTH plateau at xoff ~0.9m /
    # thread ~20% (the 6000-upd run == the 4000 -> NOT convergence-limited). The last 0.9->~0.4m to reach the
    # 0.75m aperture is an ENDGAME precision gap. Two levers, each warm-started from vglpan (the 20% policy),
    # tighter endgame noise floor (0.03->0.02) for a sharper deterministic mean:
    #   _ft_sub = continue the zero anneal BELOW the aperture (0.75->0.5) to pull the crossing MEAN tighter
    #             (marginal edge-threads get punished, but the miss=100 terminal blocks give-up; the bet is
    #             the mean shifts inside 0.75). _ft_c40 = DOUBLE the centre bonus (20->40) at a FIXED 0.75 zero
    #             = stronger pull with NO sub-aperture risk (valid threads stay positive). Whichever lifts
    #             thread more wins the next rung. Run via STAGES=single_gate_varied_gvf_lpara_ft_sub / _ft_c40.
    "single_gate_varied_gvf_lpara_ft_sub": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 0.75, "rw_cross_neg_cap": 100.0,
        "cross_zero_anneal": True, "cross_zero_start": 0.75, "cross_zero_end": 0.5, "cross_zero_hold_frac": 0.1,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_anneal_seed0_vglpan/checkpoints",
                 "++algo.noise_std_hold": 0.06, "++algo.noise_std_floor": 0.02, "++algo.noise_hold_frac": 0.3},
    },
    "single_gate_varied_gvf_lpara_ft_c40": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 40.0, "rw_cross_zero_m": 0.75, "rw_cross_neg_cap": 100.0,  # DOUBLE pull, fixed 0.75 zero
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_anneal_seed0_vglpan/checkpoints",
                 "++algo.noise_std_hold": 0.06, "++algo.noise_std_floor": 0.02, "++algo.noise_hold_frac": 0.3},
    },
    # PERCEPTION-REWARD FINE-TUNE (Fengyou 2026-07-09; the r_perc lever). The vglpns0 ablation PROVED the
    # ~0.88 m centring floor is PERCEPTION-limited (a PERFECT estimator -> 0.52 m and falling / thread 0.20->
    # 0.32), NOT control-limited -- ~0.28 m of measurement noise is amplified by the closed loop into ~0.4 m+
    # of crossing offset. The field's proven fix for exactly this (Swift + Geles both carry it; Geles gate-
    # passing error 0.5 m -> 0.15 m) is the PERCEPTION reward r_perc = perception*exp(-delta_cam^4): pay the
    # policy to keep the camera axis on the gate CENTRE so the gate stays DETECTABLE near the plane (in OUR
    # sim gate_detectable is FOV-geometry-dependent -> a gate that drifts out of frame on a fast offset
    # approach gets MASKED and rel_pos drifts exactly at the crossing; r_perc prevents that loss-of-lock) +
    # an attention pressure to fly at the gate. Warm from vglpan (the champion -- the L1-survivable base for a
    # NEW structural term) with a FIXED 0.75 zero (no re-anneal) so the ONLY change vs vglpan is r_perc. Set
    # the weight via EXTRA=++env.rw_perception (dose-response, e.g. 0.05 / 0.15). ~2000 upd.
    # Run via STAGES=single_gate_varied_gvf_lpara_perc, UPD_single_gate_varied_gvf_lpara_perc=2000.
    "single_gate_varied_gvf_lpara_perc": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 0.75, "rw_cross_neg_cap": 100.0,   # champion settings, fixed 0.75
        "rw_perception": 0.05, "rw_perception_exponent": 4.0,     # r_perc ON (override weight via EXTRA)
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_anneal_seed0_vglpan/checkpoints",
                 "++algo.noise_std_hold": 0.06, "++algo.noise_std_floor": 0.02, "++algo.noise_hold_frac": 0.3},
    },
    # ANISOTROPIC VERTICAL-WEIGHT LEVER (Fengyou greenlight 2026-07-08): the floor-dive fix that SUPERSEDES
    # the MPCC-clean contouring below (the PBRS-rate contouring was too weak -- policy-invariant, couldn't
    # escape the sink basin, and a pure additive line-bonus HOVER-FARMS: strong-enough-to-hold == strong-
    # enough-to-hover-at-spawn). This instead UN-BURIES the vertical INSIDE the single distance-to-gate
    # potential: phi = -sqrt(dx^2 + dy^2 + w*dz^2), w>>1 (rw_progress_vert_weight). SAME fixed 15 m level
    # gate as single_gate_static (apples-to-apples box-exit vs the 100%-floor baseline). isotropic progress
    # ON (progress_to_center=True) with the vertical weight cranked so sinking costs progress ~w x more. NO
    # hover-farm (progress TOWARD the gate; sitting earns 0) and NO ceiling on w. SUCCESS: exit_floor 100%
    # ->~0, cross_offset_m 9.7->0, exit_thread rises. WATCH exit_ceiling (over-correction -> lower w). w=25
    # (vertical ~1.7x forward pull at 1 m sink from 15 m) is the start; sweep. OFF-LADDER; run standalone via
    # STAGES=single_gate_static_aniso.
    "single_gate_static_aniso": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 15.0, "course_spawn_dist_hi": 15.0,   # fixed 15 m (== the sgs baseline)
        "course_drop_lo": 0.0, "course_drop_hi": 0.0,                 # level gate
        "rw_progress_to_center": True,    # isotropic base potential ON (the un-buried homing driver)
        "rw_progress_vert_weight": 25.0,  # crank the VERTICAL weight -> sinking costs progress ~25x more
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
    # NOISE CURRICULUM (Fengyou 2026-07-09; the data-motivated response to the vglpns0 finding). The
    # ablation PROVED the ~0.88 m centring floor is PERCEPTION-NOISE-limited (perfect estimator -> 64%
    # thread / 0.32 m) and that a FIXED intermediate noise DETONATES a warm-start (vglpns5 0.5 -> collapse),
    # while r_perc (a new positive dense reward) also detonated (farmable fly-away). This lever changes NO
    # reward and uses the PROVEN continuous-anneal mechanism: warm from the CHAMPION vglpan (fixed 0.75
    # zero) and ANNEAL ego_noise_scale 0 -> 1 within the run -> the policy centres on a clean signal first,
    # then adapts that skill to be noise-ROBUST as the measured vision noise ramps in. Isolates the noise
    # curriculum as the SINGLE change vs vglpan (control = vglpan at fixed noise 1.0 = 0.20/0.88 m). Start
    # overridable via EXTRA=++env.noise_scale_start (dose-response 0.0 / 0.3). ~2000 upd.
    # Run via STAGES=single_gate_varied_gvf_lpara_nscale, UPD_single_gate_varied_gvf_lpara_nscale=2000.
    "single_gate_varied_gvf_lpara_nscale": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,   # FIXED zero=4 (== real vglpan; the anneal was inert) -> isolates the noise lever
        "noise_scale_anneal": True, "noise_scale_start": 0.0, "noise_scale_end": 1.0, "noise_scale_hold_frac": 0.1,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints",
                 "++algo.noise_std_hold": 0.06, "++algo.noise_std_floor": 0.02, "++algo.noise_hold_frac": 0.3},
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
                       "course_spawn_below_g0_lo", "course_spawn_below_g0_hi",
                       "course_spawn_heading", "course_spawn_yaw_jitter")

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
