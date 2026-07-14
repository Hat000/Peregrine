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

OFF-LADDER STAGE INDEX (standalone via STAGES=<name>): single_gate_static / _varied / _varied_* levers,
hover_hold probe, dual_gate_fullstack0/0f, dual_gate_boot_floor -> dual_gate_fullstack_floor (the live
floor chain), and dual_gate_boot_floor_percept -> dual_gate_fullstack_floor_percept (the 2026-07-10
PERCEPTION-HONESTY / HARD NO-SPIN chain: fatal spin abort + yaw clamp + blur gate + r_perc, DESIGN.md §P).
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
    # FLOOR-MATCHED OPENER RELEARN (A1 root cause, 2026-07-10). The deployed vn16 policy never left the
    # VQ2 start pad: its trained OPENER is a gravity dive (0.25 g + hard tilt) because training's lethal
    # floor is the OOB bbox bottom at spawn_z - 12 m (peregrine_racing._update_boxes z margin; the
    # below-floor => collision fold only fires THERE), while the REAL warehouse floor is AT spawn (pad)
    # height. Adapter + obs contract verified CORRECT -- the fix is training-side. This stage = the
    # champion single_gate_varied_gvf_lpara_anneal copied EXACTLY, plus:
    #   * floor_at_spawn=true: per-env floor z = spawn_z - 0.25 m (resting ON the pad stays legal; the
    #     existing below-floor => gate_collision fold makes any dive below it a terminal crash).
    #   * course_spawn_below_g0 0.5..6.0: ALL gates ABOVE the pad (the old +-6 band puts half the gates
    #     below the floor = undivable-to). VQ2 floor-matched standing start.
    #   * warm +init_from = the vn16 run (the REAL-noise champion whose opener we are relearning), with
    #     the champion warm noise trio 0.12/0.03/0.5. REAL noise (NO ego_noise_scale override) -- same
    #     regime as vn16 so the ONLY change is the floor + gate-height geometry.
    # ~6000 upd (opener relearn on a warm base, not a fresh train). OFF-LADDER; run via
    # STAGES=single_gate_varied_gvf_lpara_floor, UPD_single_gate_varied_gvf_lpara_floor=6000.
    "single_gate_varied_gvf_lpara_floor": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,  # ALL gates ABOVE the pad
        "course_spawn_yaw_jitter": 0.25,
        "floor_at_spawn": True,                       # lethal floor at spawn_z - 0.25 m (A1 fix)
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,  # initial (anneal overrides live)
        "cross_zero_anneal": True, "cross_zero_start": 4.0, "cross_zero_end": 0.75, "cross_zero_hold_frac": 0.1,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_anneal_seed0_vn16/checkpoints",
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
    # BLACKOUT-COAST CALIBRATION (2026-07-09 audit wave-2). The vczns0 pair (noise 0 + REAL 4->0.75 anneal,
    # seeds 0/1, the first runs with the anneal verifiably ON) landed DET thread 0.574/0.403 with collision
    # 0.42/0.57 -- a genuine non-reward residual under PERFECT perception. The offline characterization
    # (handoff/audit-ego-inc9-2026-07-09/estimator-characterization/) showed why: the terminal 0.7-2.2 m is
    # geometrically BLIND under the +20deg mount in ALL geometries, the obs builder zeroes rel_pos the
    # instant detectability drops, and the 0.5 s stale horizon expires before the crossing at slow-lap
    # speeds -- so every run (incl. noise-0) flies the endgame on ZEROS, while the estimator's coasted
    # estimate is nearly free (~0.01 m error over 1.5 s). This stage = the vczns0 recipe + the COAST
    # PACKAGE (ego_obs_coast + horizon 1.2 s), ONE package vs vczns0 as the exact control: if the
    # 0.42-0.57 collision residual converts to threads, endgame blindness was the layer under perception.
    # Run via STAGES=single_gate_varied_gvf_lpara_coast0, UPD_single_gate_varied_gvf_lpara_coast0=4000.
    "single_gate_varied_gvf_lpara_coast0": {
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
        "ego_noise_scale": 0.0,       # perfect estimator == the vczns0 calibration regime
        "ego_obs_coast": True,        # feed coasted rel_pos + decaying conf through the blackout
        "ego_stale_horizon_s": 1.2,   # blind onset 0.7-2.2 m: the 0.5 s horizon zeroes conf pre-crossing at slow-lap
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    # CLIP-TERMINAL CALIBRATION (2026-07-09 audit wave-3). Across THREE noise-0 configs (vglpns0 fixed-4
    # 0.641 · vczns0 real-anneal 0.574/0.403 · vcoast0 anneal+coast 0.522) the DET ceiling sits at
    # ~0.5-0.65 with a stubborn 0.4-0.5 FRAME-CLIP residual that neither reward-zero tightening nor
    # perfect endgame observability (the coast package) touches. The remaining reward suspect: with the
    # parabola on, a clip still pays +19.2/20 (terminals bypassed) -- the policy is near-indifferent
    # between threading and clipping. This stage = the vczns0 recipe + the clip-penalty ANNEAL 0->20
    # (rl/ego_reward.clip_terminal_w via the peregrine_train_ego clip_pen hook; hold 0.3 lets the warm
    # policy settle before the tail-penalty ramps). ONE lever vs the vczns0 control: if the clip residual
    # converts to threads, the residual was reward-indifference; if it persists (or converts to misses),
    # it is control precision. Run via STAGES=single_gate_varied_gvf_lpara_clip0, UPD_..._clip0=4000.
    "single_gate_varied_gvf_lpara_clip0": {
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
        "ego_noise_scale": 0.0,       # perfect estimator == the vczns0 calibration regime
        "rw_clip_terminal": 0.0,      # initial (the clip_pen anneal drives clip_terminal_w live)
        "clip_pen_anneal": True, "clip_pen_start": 0.0, "clip_pen_end": 20.0, "clip_pen_hold_frac": 0.3,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
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
    # MULTI-GATE BASELINE PAIR (2026-07-09 audit wave-4). The noise-0 single-gate calibration EXHAUSTED the
    # reward/obs lever set (fixed-4 0.641 · anneal 0.574/0.403 · +coast 0.522 · +clip-pen 0.478/0.322 — every
    # config in the 0.32-0.64 band, collisions ~0.5 immune to a -20 clip penalty): the binding layer at
    # ~13 m/s is CONTROL PRECISION, and the audit's pivot trigger (plateau) fired. Per CPC/TOGT the crossing
    # point is defined by the gate SEQUENCE (approach geometry changes), per Song N=2 obs slashes crashes —
    # so the next calibration regime is 2-gate. This pair = handoff_drill at noise 0, FRESH vs WARM-from-vglp4:
    # the warm arm fills the previously-always-empty second obs slot (the audit's H6 OOD liability) — transfer
    # vs detonation is itself a needed datum. EASY reward per the drill's design (mechanics, not thread-rate).
    # Run via STAGES=handoff_drill0 / handoff_drill0w, UPD_<stage>=2000.
    "handoff_drill0": {
        **_COMMON,
        "course_n_gates": 2,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "ego_noise_scale": 0.0,       # isolate handoff mechanics from estimator corruption (calibration regime)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA},
    },
    "handoff_drill0w": {
        **_COMMON,
        "course_n_gates": 2,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "ego_noise_scale": 0.0,
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    # ================================================================================================
    # DUAL_GATE_FULLSTACK (2026-07-09 audit wave-4 -> multi-gate stage designer). The PROVEN single-gate
    # champion stack (single_gate_varied_gvf_lpara_anneal: use_racing_line + PBRS corridor 4 + magnitude
    # centering 0.4 + the per-crossing PARABOLA) ported VERBATIM to 2 gates. The ONLY deltas vs the champion
    # are the gate count (1->2), the gate0->gate1 spacing (course_seg_len 10-20 m = the VQ2 co-visibility
    # band), and the parabola zero held STATIC at 4.0 (NO cross-zero anneal in v1: one lever at a time, and
    # the anneal never beat fixed-4 per the wave data). Everything else is byte-identical to the champion and
    # is ALREADY 2-gate-correct with NO code change (audit finding, stage-designer 2026-07-09):
    #   * racing_line.build_racing_line already builds the GLOBAL head-on Hermite line spanning
    #     spawn->g0->g1 (G segments; tangent == each gate's through-normal at its centre; G1-continuous ->
    #     no cusp). The GVF query projects onto the WHOLE polyline, so progress s (arc length) AND contouring
    #     perp (cross-track) are CONTINUOUS across the gate-0 handoff -- there is NO per-target re-plan (the
    #     env's advance re-query is a position-based no-op on the global line, peregrine_racing_ego.py:970).
    #     A "gate-2-aware exit tangent" was CONSIDERED and REJECTED: blending gate-0's crossing tangent toward
    #     gate 1 would BREAK the head-on crossing the parabola + corridor target (the reward wants a CENTRED
    #     head-on crossing at EACH gate; the inter-gate curve already supplies the turn between them).
    #   * the parabola (crossing_parabola_reward on pass_linf[tg] / fwd_t[tg]) is TARGET-INDEXED -> pays per
    #     gate; with miss_terminates=True (default) ANY forward crossing of the target either ADVANCES the
    #     target or TERMINATES, so fwd_t fires at most once per gate -> once-each payment WITHOUT the latch
    #     (rw_parabola_latch stays off, exactly as the champion validated; the latch only matters if a future
    #     variant sets miss_terminates=false).
    #   * the obs 2nd window slot (WINDOW=2) already carries gate 1 during the gate-0 approach (n_gates=2,
    #     tg=0 -> slot1=gate1 valid, ego_window_indices); the coarse map + privileged critic are per-gate;
    #     the terminal fires the finish on the LAST gate (tg==G-1) and advances on the non-last pass.
    # Warm from vglp4 (the non-fragile base) with the champion 0.12/0.03/0.5 noise-std trio, at
    # ego_noise_scale=0 (the CALIBRATION regime -- isolate the 2-gate control question from estimator
    # corruption; the warm arm also FILLS the previously-always-empty 2nd obs slot, the audit's H6 OOD
    # liability). first-leg geometry (dist 8-15 m, height +-6 m, yaw jitter 0.25) == vglp4's distribution so
    # the warm-started gate-0 skill transfers cleanly; the 10-20 m gate0->gate1 leg is the new 2-gate content.
    # OFF-LADDER; run standalone via STAGES=dual_gate_fullstack0, UPD_dual_gate_fullstack0=4000.
    "dual_gate_fullstack0": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # first leg == vglp4 (clean warm transfer)
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,  # gate-0 height band == vglp4
        "course_spawn_yaw_jitter": 0.25,                                    # lateral FOV == vglp4
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,              # gate0->gate1 spacing (VQ2 co-visibility)
        "use_racing_line": True,
        "rw_progress_to_center": False,                  # progress from the GLOBAL line arc-length (pure GVF)
        "rw_corridor": 4.0,                              # PBRS cross-track contouring onto the line (champion)
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,  # per-step magnitude centering nag (champion)
        "rw_parabola_crossing": True,                    # per-crossing smooth parabola (champion)
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,  # STATIC zero=4 (NO anneal in v1)
        "ego_noise_scale": 0.0,       # calibration regime (perfect estimator -- isolate the 2-gate control question)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 "+init_from": "/scratch/network/fl3689/diffaero/outputs/train/ego_single_gate_varied_gvf_lpara_seed0_vglp4/checkpoints",
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    # dual_gate_fullstack0f: the FRESH long-budget arm = the 2026-07-09 audit's H6 FIX. The warm pair
    # dgfs0/dgfs0s1 (dual_gate_fullstack0, warm from vglp4) came back DET 0.000 2/2 -- warm transfer into
    # the 2-gate stage is DEAD (the always-empty 2nd obs slot the single-gate base never saw poisons the
    # transfer; slot-fill kills warm transfer, H6 CONFIRMED 2/2). This arm trains the SAME 2-gate content
    # FROM SCRATCH on a fresh long (16k) budget. IDENTICAL to dual_gate_fullstack0 EXCEPT: (a) NO +init_from
    # (fresh random init, not warm from vglp4); (b) the fresh-init exploration trio 0.30/0.03/0.5 -- start
    # at the BASE 0.30 noise-std ceiling (do NOT copy the warm pair's 0.12 ceiling, which UNDER-explores a
    # fresh init) and anneal to the champion 0.03 floor over the back half (hold_frac 0.5). ONE lever vs
    # vcz16: fresh init + 2 gates (vcz16 = warm single-gate at noise 0). ego_noise_scale=0.0 (calibration
    # regime), cross-zero held STATIC at 4.0. OFF-LADDER; run standalone via STAGES=dual_gate_fullstack0f,
    # UPD_dual_gate_fullstack0f=16000.
    # LAUNCHER NOTE (peregrine_train_ego.py, verified 2026-07-09): the sbatch BOUNDARY_OV
    # +warmstart_reset_logstd=true is a HARMLESS NO-OP on a fresh run -- the logstd reset is gated on
    # `warmstart_from is not None` (train ego L457-466), so with NO +init_from maybe_warmstart returns None
    # and the initial std stays at the 0.30 ceiling (the fresh-exploration intent holds; no override off
    # needed). +critic_warmup_updates=100 DOES apply on fresh (documented, train ego L468-472): the actor
    # is frozen for the first 100 updates while the random critic calibrates to the return scale -- intended
    # and cheap (100/16000).
    "dual_gate_fullstack0f": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # first leg == vglp4 (same geometry)
        "course_spawn_below_g0_lo": -6.0, "course_spawn_below_g0_hi": 6.0,  # gate-0 height band == vglp4
        "course_spawn_yaw_jitter": 0.25,                                    # lateral FOV == vglp4
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,              # gate0->gate1 spacing (VQ2 co-visibility)
        "use_racing_line": True,
        "rw_progress_to_center": False,                  # progress from the GLOBAL line arc-length (pure GVF)
        "rw_corridor": 4.0,                              # PBRS cross-track contouring onto the line (champion)
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,  # per-step magnitude centering nag (champion)
        "rw_parabola_crossing": True,                    # per-crossing smooth parabola (champion)
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,  # STATIC zero=4 (NO anneal)
        "ego_noise_scale": 0.0,       # calibration regime (perfect estimator -- isolate the 2-gate control question)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from: fresh random init (H6 fix -- warm transfer into 2 gates is dead).
                 "++algo.noise_std_hold": 0.30, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    # ================================================================================================
    # DUAL-GATE FLOOR CHAIN (A1 floor fix, 2026-07-10) -- the 2-gate arm of the floor-matched relearn,
    # run as a TWO-STAGE LADDER in ONE sbatch job (STAGES="dual_gate_boot_floor dual_gate_fullstack_floor"):
    # the sbatch loop auto-wires stage 2's +init_from to stage 1's checkpoints, so NEITHER stage hardcodes
    # a chain path. Both stages carry the FULL floor package (vs the single-gate floor stage, dual courses
    # additionally need course_gates_above_spawn: spawn_below_g0 only constrains gate 0 -- the default
    # drop band can sink gate 1 below the pad = undivable-to with the floor on):
    #   floor_at_spawn=true + course_spawn_below_g0 0.5..6.0 + course_gates_above_spawn=0.5.
    # Env content = dual_gate_fullstack0f's champion stack VERBATIM (racing line + corridor 4 +
    # centering 0.4 + parabola, STATIC zero=4). Per H6 (slot-fill kills single->dual warm transfer,
    # CONFIRMED 2/2) the chain trains 2-gate FROM SCRATCH:
    #   * dual_gate_boot_floor  = FRESH boot at ego_noise_scale=0 (calibration regime: discover the
    #     floor-constrained 2-gate behaviour on a clean signal), fresh exploration trio 0.30/0.03/0.5.
    #     ~4000 upd.
    #   * dual_gate_fullstack_floor = REAL noise (NO ego_noise_scale override -- the deploy regime),
    #     warm from the boot via the sbatch chain, warm trio 0.12/0.03/0.5. ~12000 upd (the 16k finals
    #     proved budget was the binder).
    # Run via: STAGES="dual_gate_boot_floor dual_gate_fullstack_floor",
    #          UPD_dual_gate_boot_floor=4000, UPD_dual_gate_fullstack_floor=12000.
    "dual_gate_boot_floor": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # first leg == the sg champion
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,   # gate 0 ABOVE the pad (floor)
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,              # gate0->gate1 (VQ2 co-visibility)
        "course_gates_above_spawn": 0.5,              # gate 1 too: every gate >= pad + 0.5 m (sampler clamp)
        "floor_at_spawn": True,                       # lethal floor at spawn_z - 0.25 m (A1 fix)
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,  # STATIC zero=4
        "ego_noise_scale": 0.0,       # calibration boot: clean signal for the fresh discovery phase
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from: fresh init (H6 -- single->dual warm transfer is dead); the fullstack
                 # stage warm-starts from THIS stage via the sbatch ladder's auto +init_from.
                 "++algo.noise_std_hold": 0.30, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    "dual_gate_fullstack_floor": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        # REAL noise (NO ego_noise_scale override) -- the deploy regime, same as vn16.
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here: the sbatch ladder wires +init_from=<boot checkpoints> when this
                 # runs as stage 2 of STAGES="dual_gate_boot_floor dual_gate_fullstack_floor" (a
                 # hardcoded path here would COLLIDE with the ladder's append -> hydra error).
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    # ================================================================================================
    # PERCEPTION-HONESTY / HARD NO-SPIN chain (2026-07-10, DESIGN.md §P) -- the dual_gate floor chain
    # VERBATIM plus the anti-spin-scan package, OFF-LADDER, run as the two-stage sbatch chain:
    #   sbatch --export=ALL,SEED=0,RUNTAG=vperc0,\
    #     STAGES="dual_gate_boot_floor_percept dual_gate_fullstack_floor_percept",\
    #     UPD_dual_gate_boot_floor_percept=4000,UPD_dual_gate_fullstack_floor_percept=12000,PRECHECK=1 \
    #     rl/peregrine_vq2_ego.sbatch
    # PRECHECK=1 is MANDATORY on the first launch (512 envs x 3 updates catches the new env wiring
    # before budget), AND the precheck log must show the new loss_components keys actually emitting
    # (spin_abort_rate / spin_rot_accum_mean / target_detectable_duty) -- the L16 silently-inert-hook
    # guard: absent keys == the package did not arm.
    # THE PACKAGE (all knobs are NEW +env. appends; the existing floor stages above are byte-untouched
    # -- live vdflr0/vdff1 safe; knob names deliberately ego_spin_*, NOT the inc8 spin_rate_abort
    # names, so a stale sbatch passing the old keys can never silently arm this gate):
    #   * FATAL SPIN ABORT (the owner's non-spin GUARANTEE): ego_spin_rate_abort=3.5 rad/s sustained
    #     0.4 s, OR'd with the leaky accumulated-rotation trigger 1.5 rev over a 4.0 s window; fires
    #     COLLISION-CLASS (terminal_base + banked forfeit) on the parabola path via the lethal mask.
    #   * YAW-COMMAND CLAMP 0.35 rad/s at the point of application (realized ~3.5x -> ~1.2 rad/s,
    #     inside the owner's 1-1.5 target). Action space stays +/-3.14 (deploy contract untouched).
    #     🚩 DEPLOY-SIDE: any FLIGHT of a _percept checkpoint requires a YAW-ONLY clamp ADDED to
    #     fly_rl.policy_step (a CODE CHANGE in the deploy repo at flight time: clip(rate_flu[2],
    #     +/-0.35) after the rescale) -- a training-side clamp does NOT bind the wire, and NO existing
    #     fly_rl argument implements it: max_rate clips ALL THREE axes (roll/pitch trained at full
    #     +/-3.14 -> ~9x authority cut, catastrophically OOD) and yaw_scale is MULTIPLICATIVE (mis-
    #     scales the yaw transfer function: railed 3.14 -> ~3.8 rad/s realized; legal 0.30 -> ~3x under).
    #   * MOTION-BLUR gate: ego_blur_rate_lo/hi = 2.0/4.0 rad/s -- PLACEHOLDERS pending the measured
    #     A2 detect-vs-angular-rate curve (recalibrate before any flight ckpt; the slot-in point is
    #     gate_visibility.blur_extra_miss_prob ONLY). Blur is ON in the noise-0 boot too: it is
    #     camera physics, INDEPENDENT of ego_noise_scale -- a blur-free boot would re-discover
    #     spin-scan and the fullstack would inherit it. (hi=4.0 sits ABOVE the 3.5 rate abort:
    #     harmless, but NOT because [3.5,4.0) is "fatal anyway" -- the blur cut is instantaneous
    #     per-gate LOS-PERP rate while the abort is all-axis ||omega|| SUSTAINED >0.4 s, so a brief
    #     3.8 transient is blur-cut-not-fatal and a sustained 3.7 roll-about-LOS is fatal-not-blur-
    #     cut. Kept at 4.0: blinding brief transients is desirable honesty, sustained band rotation
    #     is priced by the abort, and the A2 calibration structure is preserved.)
    #   * r_perc (framing preference) rw_perception=0.02 = rw_time -- the FARM-NEUTRALITY BOUND
    #     (rw_perception <= rw_time, pinned by tests): hover-and-stare nets <= 0/tick, so the one
    #     prior detonation mode (0.05 warm-start farmable fly-away) is priced out; introduced at the
    #     FRESH boot (perception + progress learned jointly). Escalation path if it destabilizes:
    #     the clip_pen_anneal plain-float pattern (ego_reward.py:231-240) + the L16 '[hook] ON' print.
    #   * THRESHOLD ORDERING (blind-policy defense, pinned by tests): clamped realized yaw ~1.2 <
    #     blur-free lo 2.0 < rate abort 3.5 -- a full-authority pointing sweep is never blur-punished
    #     and never fatal, so 'point the camera at the gate' stays the constructive strategy.
    # SUCCESS READS (training metrics, never renders): DET completion vs the vdflr0 twin;
    # spin_abort_rate/exit_spin -> ~0 by convergence (early nonzero = the gate is teaching);
    # target_detectable_duty >> vn16's 40.8% spin-scan duty; realized |w_z| from the trace <= ~1.5;
    # SUB-ABORT CONSTANT-ROTATION check (the honest guarantee ceiling is ~2.36 rad/s SUSTAINED =
    # 2*pi*rev/window; see DESIGN.md P.8): the trace's sustained ||omega|| distribution must NOT
    # plateau at ~1.8-2.35 rad/s -- a slow roll/pitch corkscrew there is a spinner while
    # spin_abort_rate reads 0 (corroborate: spin_rot_accum_mean near its fixed point ~7-9.4 rad
    # instead of decaying between banks); exit_timeout/exit_front flat (r_perc not being farmed).
    # NOTE ego_collision_rate INCLUDES spin aborts here -- subtract spin_abort_rate before comparing
    # collision vs non-percept twins.
    # OPTIONAL CHEAP RUNG (B2b validate-small-first; Fengyou's call given the July clock): a ~2-4k-upd
    # single-gate smoke first = STAGES="dual_gate_boot_floor_percept" alone at UPD=2000 with
    # EXTRA="++env.course_n_gates=1" -- skippable; the boot stage itself is already the cheap arm.
    # ================================================================================================
    "dual_gate_boot_floor_percept": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_boot_floor
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,  # STATIC zero=4
        "ego_noise_scale": 0.0,       # calibration boot -- but blur stays ON (independent of noise_scale)
        # ---- the perception-honesty package (see the block comment above) ----
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_blur_rate_hi_rad_s": 4.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_spin_rate_abort": 3.5,        # rad/s sustained (fatal, collision-class)
        "ego_spin_time_abort": 0.4,        # s
        "ego_spin_rev_abort": 1.5,         # revolutions (leaky window trigger)
        "ego_spin_rev_window_s": 4.0,      # s
        "ego_yaw_cmd_clamp_rad_s": 0.35,   # commanded; realized ~3.5x -> ~1.2 rad/s
        "rw_perception": 0.02,             # == rw_time (farm-neutrality bound; NOT the detonated 0.05)
        "rw_perception_exponent": 4.0,     # Geles/Swift
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from: fresh boot (H6); the fullstack _percept stage chains from THIS stage.
                 "++algo.noise_std_hold": 0.30, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    "dual_gate_fullstack_floor_percept": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_fullstack_floor
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        # REAL noise (NO ego_noise_scale override) -- the deploy regime; the package rides along.
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_blur_rate_hi_rad_s": 4.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_spin_rate_abort": 3.5,
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,
        "rw_perception": 0.02,
        "rw_perception_exponent": 4.0,
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here (LOAD-BEARING): the sbatch ladder auto-appends
                 # +init_from=<boot _percept checkpoints> when run as stage 2 of the chain; a
                 # hardcoded path would COLLIDE with that append -> hydra error.
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5},
    },
    # ================================================================================================
    # ESTIMATOR-FAITHFUL chain (_pef, 2026-07-11) -- the _percept chain VERBATIM plus the
    # ESTIMATOR-FAITHFUL ACTOR OBS package (owner directive: THE ACTOR NEVER SEES GROUND TRUTH),
    # OFF-LADDER, run as the two-stage sbatch chain:
    #   sbatch --export=ALL,SEED=<s>,RUNTAG=vpef<s>,\
    #     STAGES="dual_gate_boot_floor_pef dual_gate_fullstack_floor_pef",\
    #     UPD_dual_gate_boot_floor_pef=4000,UPD_dual_gate_fullstack_floor_pef=12000,PRECHECK=1 \
    #     rl/peregrine_vq2_ego.sbatch
    # PRECHECK=1 MANDATORY + the L16 guard: the precheck log must show the NEW loss_components keys
    # emitting (eskf_tilt_err_deg_mean / eskf_tilt_err_deg_p90 / eskf_accel_update_duty /
    # kf_vel_err_mean / sf_mag_g_mean) alongside the _percept keys (spin_abort_rate etc.) -- absent
    # keys == the package did not arm. THE PACKAGE (all knobs NEW appends; existing stages
    # byte-untouched -- live vperc0/vdflr0 safe):
    #   * +env.ego_faithful=true: training obs[0:8] comes from the translated deploy vq2_ego_lean
    #     chain (rl/ego_ins_emul.py): roll/pitch = the emulated ESKF leveler (full gate stack incl.
    #     A8 motion-reject, stepped once per tick on the latest IMU sample -- the verified deploy
    #     rate contract); velocity = the emulated LinearKF strapdown through the SAME lying attitude
    #     + position-fix-only corrections, projected world->body through it; rates = the raw latest
    #     gyro sample (measured-zero sensor noise). Per-channel ablation: +env.ego_att_model /
    #     ego_vel_model / ego_rate_model.
    #   * ++dynamics.n_substeps=5 (~150 Hz plant substeps ~ the measured 143.3 Hz wire IMU): the
    #     load-bearing piece -- at n_substeps=1 plant truth is itself ZOH-at-tick-rate, a tick-rate
    #     leveler would track truth EXACTLY, and the measured-fatal aliasing channel (a5: median
    #     ~5 deg / p90 ~27 deg per ~57 ms tick at 3 g; 33 ms training band 1.5/4.6/8.4 deg,
    #     rl/tools/leveler_bench.py) could not exist in-sim. n_substeps changes plant trajectories,
    #     so it rides ONLY these new stages. 🚩 PRE-LAUNCH CHECK: params.transport_delay_steps
    #     counts SUBSTEPS (diffaero_dynamics.py:88-93) -- if the resolved plant params ever set it
    #     nonzero it must be scaled x5 here or the transport latency silently shrinks 5x (expected
    #     0; the tick-level DR latency 1-3 steps ring buffer is separate and unaffected). Also read
    #     resolved env.dt from a completed stage's .hydra (0.0333 s corroborated by two in-repo
    #     fallbacks) before hardcoding any further substep-ratio claims.
    #   * ++dynamics.capture_specific_force=true: the plant's last-substep body specific force =
    #     the emulated IMU sample (observation-only, scalar parity branch untouched).
    #   * +env.ego_est_dt_ticks_hi=4 (CHOKED-LOOP dt emulation; reviewer-caught 2026-07-11): the
    #     wire nav loop has NEVER run at the 30 Hz training tick (measured ego_obs tick gaps: a5
    #     median 41.6 ms / max 139; a7 median 34.8 / max 83 -- leveler_bench tick_gap_report), and
    #     per-tick leveler divergence SCALES WITH dt, so a fixed-33 ms emulation is ~3.5x CLEANER
    #     than every measured wire operating point. The emulated ESKF/KF/rate channels advance on a
    #     RENEWAL schedule over the MEASURED pooled tick-gap pmf (ego_ins_emul.MEASURED_TICK_GAP_PMF,
    #     k in {1..4} -> dt in {33..133} ms), holding (frozen obs) in between -- the training-dt
    #     mixture covers every wire operating point measured to date.
    # BOOT SEMANTICS (deliberate, the blur precedent): ego_noise_scale=0.0 zeroes the VISION noise
    # but the leveler/KF are ALGORITHM-STRUCTURAL and noise_scale-INDEPENDENT -- the noise-0 boot
    # already flies the lying attitude (else the fullstack would inherit a boot trained on truth
    # attitude = the exact fatal gap). If the boot fails to learn AT ALL, diagnose against the
    # vperc0 twin's boot curve BEFORE touching the emulation (estimator-corruption-tax-at-boot vs
    # package bug fork).
    # ACCEPTANCE READ (training metrics): eskf_tilt_err_deg_* must sit AT/ABOVE the leveler_bench
    # MODE B(iii) renewal band on high-|f| rollouts (REAL a5 IMU, k_hi=4, seeds 0-2, run
    # 2026-07-11: median 2.4-3.0 / p90 19.6-33.4 / max 35.8-64.2 deg per advance; a7 corroborates
    # 2.1-2.9 / 6.9-30.3 / 39.5-48.8. The in-env key is an ABSOLUTE error incl. held-tick
    # staleness, so it reads >= the per-advance band). The B(iii) median sits ~0.5x the recorded-
    # grid band (5.2/26.8) from 33 ms grid quantization -- documented; the p90/max TAIL (the fatal
    # 27-45 deg steering-on-lies regime that crashed a5) matches/exceeds the wire, and the tail is
    # the crash mechanism. |f|-matching still required (sf_mag_g_mean >= ~2.5 -- hover rollouts
    # trivially read ~0 and prove nothing); ~0 deg under matched |f| at n_substeps=5 is a STOP-SHIP
    # signal (escalate n_substeps 5->8 once, then intra-tick rate-loop sysid realism -- NEVER an
    # invented noise constant). eskf_accel_update_duty ~0 in aggressive flight / nonzero at
    # spawn+coast (the A8 re-level moments) = the gate stack is alive.
    # 🚩 dt LAUNCH GATE (pre-flight, MANDATORY): before ANY _pef ckpt flies, run leveler_bench's
    # tick-gap report on the incoming flight's ego_obs.jsonl -- the deploy loop must sit INSIDE the
    # trained k<=4 band (sustained gaps <= ~133 ms, i.e. loop >= ~7.5 Hz; a5/a7 both inside). A
    # choked loop beyond that band = the ckpt is OOD on dt: fix the loop rate or re-measure the pmf
    # + raise ego_est_dt_ticks_hi and retrain -- do NOT fly through it.
    # ================================================================================================
    "dual_gate_boot_floor_pef": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_boot_floor
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,  # STATIC zero=4
        "ego_noise_scale": 0.0,       # calibration boot -- vision noise 0, but the LEVELER STILL LIES
        # ---- the perception-honesty package (== _percept verbatim) ----
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_blur_rate_hi_rad_s": 4.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_spin_rate_abort": 3.5,
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,
        "rw_perception": 0.02,
        "rw_perception_exponent": 4.0,
        # ---- the estimator-faithful package (see the block comment above) ----
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,          # measured choked-loop dt band (renewal over the pmf)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from: fresh boot (H6); the fullstack _pef stage chains from THIS stage.
                 "++algo.noise_std_hold": 0.30, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 # estimator-faithful plant knobs (++ = add-or-override; not sbatch-appended keys)
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    "dual_gate_fullstack_floor_pef": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_fullstack_floor
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        # REAL noise (NO ego_noise_scale override) -- the deploy regime; the packages ride along.
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_blur_rate_hi_rad_s": 4.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_spin_rate_abort": 3.5,
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,
        "rw_perception": 0.02,
        "rw_perception_exponent": 4.0,
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,          # measured choked-loop dt band (renewal over the pmf)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here (LOAD-BEARING): the sbatch ladder auto-appends it on stage 2.
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    # ================================================================================================
    # NODITHER fine-tune (2026-07-12) -- dual_gate_fullstack_floor_pef VERBATIM (course + reward + the
    # HARD no-spin package + the estimator-faithful _raw ALL byte-identical, so the flight skill is
    # preserved) PLUS exactly ONE new armed knob: rw_yaw_dither. A SHORT WARM-STARTED fine-tune whose
    # ONLY job is to produce a "DITHER-FREE ANCESTOR": warm from the best low-dither flyer (vpeffs0) and
    # add the anti-dither yaw-jerk penalty until the yaw command is STILL and it STILL flies, so future
    # lineages warm from a clean base. OFF-LADDER, run as a SINGLE standalone stage warm-started from
    # vpeffs0 (the launcher wires +init_from at run time via EXTRA -- NOT hardcoded in the stage):
    #   sbatch --export=ALL,SEED=0,RUNTAG=vpefnd0,\
    #     STAGES="dual_gate_fullstack_floor_pef_nodither",\
    #     UPD_dual_gate_fullstack_floor_pef_nodither=3000,\
    #     EXTRA="+init_from=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_vpeffs0/checkpoints" \
    #     rl/peregrine_vq2_ego.sbatch
    # (confirm the vpeffs0 checkpoints dir before launch -- RUNTAG that produced vpeffs0). A single-stage
    # run leaves PREV_CKPT empty so the loop appends NO +init_from -> the EXTRA one does not collide.
    # WHY the anti-dither term is authorized + rides ALONGSIDE the no-spin guarantee (Fengyou 2026-07-12):
    # it is a pure SMOOTHNESS penalty on the yaw-command temporal CHANGE (a rail-FLIP pays; a sustained
    # smooth TURN pays ~0 -- turns are needed downstream, speed is NOT penalised), NOT a magnitude/|omega|/
    # energy penalty. The HARD no-spin stays BY CONSTRUCTION: the fatal spin abort (ego_spin_*) + yaw clamp
    # are UNTOUCHED here. The jerk penalty alone has an escape -- a slow CONSTANT yaw drift has ~0 jerk, so
    # it would dodge the penalty and become a slow spin; the RETAINED accumulated-rotation spin abort
    # (ego_spin_rev_abort=1.5 rev / 4 s window) is what closes that hole. Jerk kills the dither, the abort
    # kills the constant-spin escape -- KEEP BOTH.
    # WEIGHT CALIBRATION (rw_yaw_dither=0.5, conservative; SWEEPABLE via EXTRA=++env.rw_yaw_dither=...):
    # the penalty is -0.5 * (delta yaw_cmd)^2 on the APPLIED (post-clamp) yaw rate. At the TRAINING clamp
    # ego_yaw_cmd_clamp_rad_s=0.35 a full rail-FLIP is delta = 0.35-(-0.35) = 0.70 rad/s -> penalty
    # 0.5*0.70^2 = 0.245/step -- ~41% of a bring-up per-step progress (rw_progress 2.0 * ~0.30 m/step ~
    # 0.60), a REAL deterrent that CANNOT dominate a productive step; and it only bites the ~30% of ticks
    # that flip, so the average tax is ~0.07/step and vanishes to 0 at convergence (steady yaw -> delta 0).
    # A half-flip (rail->0, delta 0.35) pays 0.061; benign near-zero jitter (delta 0.05) pays 0.00125 (~0).
    # (At the DEPLOY clamp 0.7 the same flip is delta 1.4 -> 0.98; that is the deploy regime, not trained
    # here.) SWEEP UP (0.5->1.0->2.0) if dither persists; DOWN if gate acquisition / flight degrades.
    # BUDGET: SHORT -- ~2000-4000 updates (a warm reshape of ONE behaviour, not fresh discovery); 3000
    # recommended, early-stoppable once exit_spin/spin_abort_rate ~0 AND DET thread holds vs vpeffs0.
    # ================================================================================================
    "dual_gate_fullstack_floor_pef_nodither": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_fullstack_floor_pef
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        # REAL noise (NO ego_noise_scale override) -- the deploy regime; the packages ride along.
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,
        "ego_blur_rate_hi_rad_s": 4.0,
        "ego_spin_rate_abort": 3.5,        # HARD no-spin package == _pef VERBATIM (UNTOUCHED)
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,         # the accumulated-rotation trigger = the constant-drift closer
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,   # yaw clamp UNTOUCHED (the anti-dither term rides ALONGSIDE)
        "rw_perception": 0.02,
        "rw_perception_exponent": 4.0,
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,
        # ---- nodither ADDITION (the ONE new armed knob; default-OFF on every other stage) ----
        "rw_yaw_dither": 0.5,              # anti-dither yaw-jerk penalty (conservative; see the block comment)
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here: run STANDALONE, the launcher wires vpeffs0 via EXTRA (single-stage
                 # run -> PREV_CKPT empty -> no loop-appended +init_from to collide with).
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    # ================================================================================================
    # TRACK B: ANNEALED ATTITUDE-CAP fine-tune (attcap, 2026-07-12) -- dual_gate_fullstack_floor_pef
    # VERBATIM (course + reward + the HARD no-spin package + the estimator-faithful _raw ALL byte-identical,
    # so the flight skill is preserved) PLUS exactly ONE added arm: the SOFT attitude caps
    # (attitude_limit_penalty, ego_reward.py). A SHORT WARM-STARTED fine-tune of the working ego champion
    # (vpeffs0) whose ONLY job is a GENTLER, deployable champion that flies NATIVELY inside a pitch/roll
    # band (so it needs no deploy-side control clamp). OFF-LADDER, run as a SINGLE standalone stage
    # warm-started from vpeffs0 (the launcher wires +init_from at run time via EXTRA -- NOT hardcoded):
    #   sbatch --export=ALL,SEED=0,RUNTAG=vattcap0,\
    #     STAGES="dual_gate_fullstack_floor_pef_attcap",\
    #     UPD_dual_gate_fullstack_floor_pef_attcap=3000,PRECHECK=1,\
    #     EXTRA="+init_from=/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_vpeffs0/checkpoints" \
    #     rl/peregrine_vq2_ego.sbatch
    # (confirm the vpeffs0 checkpoints dir before launch -- the RUNTAG that produced vpeffs0. A single-stage
    # run leaves PREV_CKPT empty so the loop appends NO +init_from -> the EXTRA one does not collide.)
    # THE ONE ADDED ARM -- SOFT, NON-TERMINAL attitude caps (perception-preservation, NOT energy):
    #   R_att = -rw_att_pitch*relu(|pitch|-att_pitch_limit_rad) - rw_att_roll*relu(|roll|-att_roll_limit_rad)
    # on the TRUE leveled body attitude (GT legal in reward), ZERO inside the band, linear ramp past it,
    # NEVER a termination. TARGETS (Fengyou deploy findings 2026-07-12):
    #   * PITCH band 12 deg (0.2094 rad), weight 0.3. DELIBERATELY tight and -- UNLIKE the _pefcap 30 deg
    #     band -- BELOW the airframe's ~17.8 deg (0.31 rad) tilted-pad REST pitch and normal forward-cruise
    #     pitch: the diagnosed deploy killer is the head-down DIVE (body pitch ~ -50 deg -> the +20 deg
    #     camera points -30 deg -> LOSES the gate), and a HARD 10 deg pitch clamp is what threaded past 4
    #     gates in deploy. A tight band taxes forward-cruise pitch too (it WILL slow the racer) -- ACCEPTED
    #     under BANK-FIRST (complete gates at any speed); the SOFT annealed hinge NUDGES the nose up rather
    #     than WALLING it, so it reshapes the flyer instead of breaking it. (Sweep 12 -> 10 deg via
    #     EXTRA=++env.att_pitch_limit_rad=0.1745329 if the head-down behaviour persists.)
    #   * ROLL band 60 deg (1.0472 rad), weight 0.2. GENEROUS: roll is LOAD-BEARING for turns, so the cap
    #     only clips extreme high-g banks that swing the camera off the gate; normal turning banks pay 0.
    # ANNEAL (the nodither lesson: a full-strength penalty HOT-APPLIED to a competent policy detonates it):
    # the penalty WEIGHTS ramp IN from 0 -> target over the FRONT of the run, END-HOLD at the full caps for
    # the last 30% (att_cap_anneal, peregrine_train_ego._resolve_att_cap_anneal, reusing the progress_ramp /
    # spin-abort END-HOLD schedule). At update 0 the caps are INERT (scale 0) = the vpeffs0 behaviour
    # UNTOUCHED; they grow gradually; the saved ckpt's converged regime IS the full cap. The LIMITS are
    # FIXED from the start -- ONLY the weights ramp. NO mini-ladder needed (single stage, single ckpt pull).
    # HARD no-spin stays BY CONSTRUCTION: the fatal spin abort (ego_spin_*) + yaw clamp are UNTOUCHED here;
    # the caps are a pure reward term riding alongside. Everything else is BYTE-IDENTICAL to _pef (NO
    # out-of-view spawn, NO 8 m spacing / range cap, NO perception-next, NO anti-dither -- those live only
    # in _pefcap / _nodither). BUDGET: SHORT ~3000 updates (a warm RESHAPE of one behaviour). SUCCESS
    # (training metrics; renders untrustworthy): DET box-exit thread must NOT collapse vs vpeffs0 ~0.969
    # (caps are non-load-bearing so expected safe; a tight pitch cap SLOWS it -- fine, BANK-FIRST -- as long
    # as it still threads), att_pitch/att_roll ramp visible in the [att-cap-anneal] logs, exit_spin ~0.
    # ================================================================================================
    "dual_gate_fullstack_floor_pef_attcap": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_fullstack_floor_pef
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        # REAL noise (NO ego_noise_scale override) -- the deploy regime; the packages ride along.
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_blur_rate_hi_rad_s": 4.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_spin_rate_abort": 3.5,        # HARD no-spin package == _pef VERBATIM (UNTOUCHED)
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,         # the accumulated-rotation trigger (constant-drift closer) preserved
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,   # yaw clamp UNTOUCHED (the caps ride ALONGSIDE)
        "rw_perception": 0.02,
        "rw_perception_exponent": 4.0,
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,          # measured choked-loop dt band (renewal over the pmf)
        # ---- attcap ADDITION (the ONE added arm; SOFT annealed attitude caps; default-OFF elsewhere) ----
        "rw_att_pitch": 0.3, "att_pitch_limit_rad": 0.2094395,   # 12 deg pitch band (below the 17.8deg pad-rest
        #                                                          tilt on purpose: the head-down dive is the killer)
        "rw_att_roll": 0.2, "att_roll_limit_rad": 1.0471976,     # 60 deg roll band (roll load-bearing for turns)
        "att_cap_anneal": True, "att_cap_start": 0.0, "att_cap_hold_frac": 0.3,  # weight 0->target, END-HOLD last 30%
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here: run STANDALONE, the launcher wires vpeffs0 via EXTRA (single-stage
                 # run -> PREV_CKPT empty -> no loop-appended +init_from to collide with).
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    # ================================================================================================
    # TRACK A: FULL-LINEAGE RETRAIN with BOTH behaviours annealed IN (2026-07-13) -- dual_gate_fullstack_
    # floor_pef VERBATIM (course + reward + the HARD no-spin package + the estimator-faithful _raw all byte-
    # identical) PLUS exactly TWO default-OFF-elsewhere arms, BOTH ANNEALED from 0: (A) the SOFT attitude
    # cap (rw_att_pitch/rw_att_roll + att_cap_anneal) and (B) the anti-dither yaw-jerk penalty (rw_yaw_dither
    # + yaw_dither_anneal). GOAL: bake the two deploy-side walls -- the roll/pitch control clamp and the yaw
    # rail-flip clamp -- INTO the trained policy so the deployed ckpt needs NO external clamps.
    #
    # WHY A FULL-LINEAGE RETRAIN, NOT A WARM FINE-TUNE (the load-bearing lesson): hot-applying EITHER term at
    # full strength to the finished champion (vpeffs0, 96.9% DET) DETONATED it -- the 'nodither' hot yaw-jerk
    # fine-tune collapsed it 96.9% -> 0.16%, and a hot attitude cap collapsed the Track-B fine-tune 96.9% ->
    # 9.6%. A short warm-start fine-tune is still fundamentally re-shaping a CONVERGED policy. The fix is to
    # run the WHOLE _pef lineage (fresh boot -> fullstack) with the two terms ANNEALED IN from 0 over the
    # fullstack's updates, so the policy LEARNS to fly inside the band / with a still yaw from the start rather
    # than being slammed with a strong penalty after convergence. Run as the TWO-STAGE _pef chain (fresh boot,
    # then THIS fullstack -- the sbatch auto-appends +init_from on stage 2, exactly like plain _pef/_pefcap):
    #     sbatch --export=ALL,SEED=0,RUNTAG=vtrackA0,\
    #       STAGES="dual_gate_boot_floor_pef dual_gate_fullstack_floor_pef_trackA",\
    #       UPD_dual_gate_boot_floor_pef=4000,UPD_dual_gate_fullstack_floor_pef_trackA=12000,PRECHECK=1 \
    #       rl/peregrine_vq2_ego.sbatch
    #
    # THE TWO ARMS (each a NEW default-OFF append; the _pef stages + every existing stage stay byte-identical):
    #   (A) SOFT ATTITUDE CAP (perception-preservation, NOT energy): -rw_att_pitch*relu(|pitch|-att_pitch_limit)
    #       -rw_att_roll*relu(|roll|-att_roll_limit) on the TRUE leveled attitude (GT legal in reward; PRICES
    #       only obs[3:5] roll/pitch). ROLL band 60 deg (owner Fengyou: roll self-limited ~60 deg -> REPLACES
    #       the deploy roll clamp). PITCH band 60 deg = a LOOSE BACKSTOP: pitch's real failure lever is NOT the
    #       cap -- the gate-losing head-down dive is ~-50deg (INSIDE 60 deg) and is a PERCEPTION loss (camera
    #       sees floor), owned by rw_perception=0.02 (already in _pef), NOT a tighter pitch cap. So we LEAN ON
    #       PERCEPTION for pitch and keep the pitch cap only as a >60deg safety limit (a tighter pitch band is a
    #       one-line EXTRA=++env.att_pitch_limit_rad=... if head-down persists, the _attcap 12deg precedent).
    #       Weights MILD (0.2/0.2). ANNEALED: att_cap_anneal, weight 0->target over the front, END-HOLD last 30%.
    #   (B) ANTI-DITHER yaw-jerk penalty: -rw_yaw_dither*(yaw_cmd_t - yaw_cmd_{t-1})^2 on the APPLIED (post-
    #       clamp) yaw-rate action (channel 3) -> prices YAW-COMMAND JERK, which the policy OBSERVES (yaw rate =
    #       obs[7]) and CONTROLS (the yaw action). A steady turn pays ~0 (turns are FREE -- speed is NOT
    #       penalised); only the +-clamp rail-flip oscillation pays. rw_yaw_dither=0.5 is CALIBRATED FOR THE
    #       CLAMPED regime: the yaw clamp stays ARMED at 0.35 (== _pef VERBATIM) so a rail-FLIP delta =
    #       0.35-(-0.35) = 0.70 rad/s -> -0.5*0.70^2 = -0.245/step (arming the clamp rather than retuning the
    #       weight for the +-3.14 rail keeps the still-yaw skill calibrated to the SAME clamp the lineage flies).
    #       The slow CONSTANT-drift escape (steady yaw = ~0 jerk) is closed BY CONSTRUCTION by the RETAINED
    #       fatal spin abort (ego_spin_rev_abort=1.5 rev accumulator) + the yaw clamp. ANNEALED: yaw_dither_
    #       anneal, weight 0->target over the front, END-HOLD last 30%.
    # NO-SPIN stays HARD / BY CONSTRUCTION (the fatal abort package + realized-yaw clamp EXACTLY the _pef values,
    # UNTOUCHED); the two arms are pure reward terms riding alongside. NO GT added to the actor obs (roll/pitch =
    # obs[3:5], yaw rate = obs[7] already observable; the reward reads GT internally but prices only observable
    # state). algo=appo, gamma=0.9975 (both MANDATORY). BUDGET: full 4000 boot + 12000 fullstack. SUCCESS
    # (training metrics; renders untrustworthy): DET-thread ~ the _pef champion (>=~0.9) with att_pen ->
    # bounded + yaw_dither_pen -> ~0 by convergence, roll |leveled| <= ~60 deg, exit_spin ~0 (NON-spinning),
    # and the [att-cap-anneal] + [yaw-dither-anneal] ramp visible in the precheck log.
    # ================================================================================================
    "dual_gate_fullstack_floor_pef_trackA": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_fullstack_floor_pef
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        # REAL noise (NO ego_noise_scale override) -- the deploy regime; the arms ride along.
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_blur_rate_hi_rad_s": 4.0,     # PLACEHOLDER pending the A2 detect-vs-rate curve
        "ego_spin_rate_abort": 3.5,        # HARD no-spin package == _pef VERBATIM (UNTOUCHED)
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,         # the accumulated-rotation trigger (constant-drift closer) preserved
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,   # yaw clamp UNTOUCHED (both arms ride ALONGSIDE; yaw-dither calibrated for it)
        "rw_perception": 0.02,
        "rw_perception_exponent": 4.0,
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,          # measured choked-loop dt band (renewal over the pmf)
        # ---- ARM (A): SOFT ANNEALED ATTITUDE CAP (roll 60deg owner cap; pitch 60deg LOOSE backstop, leans on perception) ----
        "rw_att_pitch": 0.2, "att_pitch_limit_rad": 1.0471976,   # 60 deg pitch band (LOOSE backstop; rw_perception owns head-down)
        "rw_att_roll": 0.2, "att_roll_limit_rad": 1.0471976,     # 60 deg roll band (owner Fengyou: roll capped ~60 deg)
        "att_cap_anneal": True, "att_cap_start": 0.0, "att_cap_hold_frac": 0.3,     # weight 0->target, END-HOLD last 30%
        # ---- ARM (B): ANTI-DITHER ANNEALED YAW-JERK penalty (calibrated for the 0.35 clamp above) ----
        "rw_yaw_dither": 0.5,
        "yaw_dither_anneal": True, "yaw_dither_start": 0.0, "yaw_dither_hold_frac": 0.3,  # weight 0->target, END-HOLD last 30%
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here (LOAD-BEARING, == plain _pef fullstack): the sbatch ladder auto-appends it
                 # on stage 2 of the boot->trackA chain (a full-lineage retrain, NOT a warm-from-vpeffs0 fine-tune).
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    # ================================================================================================
    # RECOVERY / DAMPING chain (_recover, 2026-07-14) -- dual_gate_fullstack_floor_pef_trackA VERBATIM
    # (course + reward + HARD no-spin package + estimator-faithful _raw all byte-identical, INCLUDING the
    # trackA att-cap + anti-dither arms) PLUS one MORE default-OFF arm: the RECOVERY / DAMPING roll term,
    # ANNEALED from 0. The 386-flight forensic diagnosed the live failure as a RECOVERY failure, NOT a
    # speed/tuning one: bank/speed/altitude ACCUMULATE gate-over-gate (peak roll ~22deg@g1 -> ~70deg@g5,
    # never damping) until the trajectory diverges into a wall (100% eventual collide). The course GENUINELY
    # needs aggressive banks (the g2->g3 turn needs ~61deg, p90 71deg), so a roll CAP or deploy FENCE
    # backfires (a 20-25deg cap makes the turn impossible; flights die exactly at that turn -- the
    # instability lives in the SAME regime as the maneuver). The lever is therefore a TRAINING reward that
    # teaches bank-hard-then-LEVEL: RECOVER to neutral BETWEEN maneuvers WITHOUT penalizing the turn-bank
    # itself. OFF-LADDER; run as the two-stage _pef chain (fresh boot, then THIS fullstack -- the sbatch
    # auto-appends +init_from on stage 2, exactly like _pef / trackA):
    #     sbatch --export=ALL,SEED=0,RUNTAG=vtrackR0,\
    #       STAGES="dual_gate_boot_floor_pef dual_gate_fullstack_floor_pef_recover",\
    #       UPD_dual_gate_boot_floor_pef=4000,UPD_dual_gate_fullstack_floor_pef_recover=12000,PRECHECK=1 \
    #       rl/peregrine_vq2_ego.sbatch
    #
    # THE RECOVERY ARM (a NEW default-OFF append; the trackA/_pef stages + every existing stage stay
    # byte-identical). Form (A) is ARMED here (the recommended primary -- it damps the accumulation on the
    # STRAIGHT legs where it builds, and its bearing weight FREES the turn); form (B) is a one-line
    # alternative (add "rw_cross_level": <w>, the recovery_anneal ramps BOTH weights):
    #   (A) BEARING-WEIGHTED ROLL PENALTY (dense; ego_reward.recovery_roll_penalty): -rw_roll_recover *
    #       roll^2 * w(theta), theta = the leveled HORIZONTAL bearing of the current target gate relative to
    #       the drone's horizontal TRAVEL direction (0 == dead-ahead of travel -> LINED UP -> should be
    #       LEVEL), w = exp(-(theta/rw_roll_recover_theta0_rad)^2). It penalizes a SUSTAINED bank ONLY while
    #       the gate is centred ahead (the accumulation case) and w -> 0 as the gate moves off-axis (banking
    #       toward it is a legitimate TURN -> freed). HORIZONTAL-only -> an up-leg with the gate straight
    #       ahead keeps w~1 and prices ONLY roll, never the climb-pitch. WHY TRAVEL-relative and NOT
    #       nose-relative: the perception-yaw coupling keeps the gate CENTRED IN VIEW (nose tracks the gate)
    #       so a nose-relative bearing reads ~0 even mid-turn and would wrongly punish the ~61deg turn --
    #       travel-relative is large precisely WHILE redirecting velocity onto the new leg (banking through
    #       the turn) and shrinks to 0 once flying straight at the gate (see leveled_horizontal_bearing).
    #       GT-legal (TRUE leveled roll + GT gate geometry; NOT the actor obs). Weight MILD (0.05); theta0
    #       30deg (lined-up zone |theta|<~30deg, turn freed beyond). ANNEALED: recovery_anneal, weight
    #       0->target over the front, END-HOLD last 30% (hot-applying a strong roll penalty to a competent
    #       flyer risks the att-cap/nodither DETONATION; ramp it in so the level-out skill grows gradually).
    # NO-SPIN stays HARD / BY CONSTRUCTION (the fatal abort package + realized-yaw clamp == trackA/_pef
    # VERBATIM, UNTOUCHED); the recovery arm is a pure reward term riding alongside. NO deploy-side roll
    # clamp/governor is added (the whole point: fence the moment, not the axis). algo=appo, gamma=0.9975.
    # SUCCESS (training metrics; renders untrustworthy): DET-thread ~ the trackA champion with peak/mean
    # leveled |roll| DAMPING gate-over-gate (the 22->70deg accumulation flattened) + roll_recover_pen ->
    # bounded and shrinking by convergence (the drone flies legs level and banks only through the turn),
    # exit_spin ~0, and the [recovery-anneal] ramp visible in the precheck log.
    # ================================================================================================
    "dual_gate_fullstack_floor_pef_recover": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == trackA / _pef fullstack
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,
        "ego_blur_rate_hi_rad_s": 4.0,
        "ego_spin_rate_abort": 3.5,        # HARD no-spin package == trackA / _pef VERBATIM (UNTOUCHED)
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,
        "rw_perception": 0.02,
        "rw_perception_exponent": 4.0,
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,
        # ---- trackA arms preserved VERBATIM (att cap + anti-dither, both annealed) ----
        "rw_att_pitch": 0.2, "att_pitch_limit_rad": 1.0471976,
        "rw_att_roll": 0.2, "att_roll_limit_rad": 1.0471976,
        "att_cap_anneal": True, "att_cap_start": 0.0, "att_cap_hold_frac": 0.3,
        "rw_yaw_dither": 0.5,
        "yaw_dither_anneal": True, "yaw_dither_start": 0.0, "yaw_dither_hold_frac": 0.3,
        # ---- NEW ARM: RECOVERY / DAMPING form (A), bearing-weighted roll^2, ANNEALED from 0 ----
        # (form (B) alternative: add "rw_cross_level": <w>; recovery_anneal ramps BOTH weights together)
        "rw_roll_recover": 0.05,           # (A) dense bearing-weighted roll^2 penalty (lined-up bank only)
        "rw_roll_recover_theta0_rad": 0.5235988,  # 30 deg free-turn half-width (turn freed beyond ~30 deg)
        "rw_cross_level": 0.0,             # (B) sparse per-crossing roll^2 (OFF here; one-line to A/B it)
        "recovery_anneal": True, "recovery_start": 0.0, "recovery_hold_frac": 0.3,  # weight 0->target, END-HOLD last 30%
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here (== trackA / _pef fullstack): the sbatch ladder auto-appends it on stage 2.
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    # ================================================================================================
    # BEHAVIORAL-CAP chain (_pefcap, 2026-07-12) -- the _pef chain VERBATIM plus a DEFAULT-OFF package
    # that pushes the deployed behavioral limits INTO the trained policy so it self-limits WITHOUT any
    # deploy-side control clamp. Targets the two live-flight failures: (a) the coarse-map horiz turn prior
    # (obs[9:11]) is UNDER-TRAINED -- gate 1 is co-visible at 10-20 m so the policy ignored the map; (b) the
    # policy flies over-aggressively head-down (body pitch ~ -50deg -> the +20deg camera points -30deg ->
    # LOSES the gate) and high-g. OFF-LADDER, run as the two-stage chain (mirrors _pef):
    #   sbatch --export=ALL,SEED=<s>,RUNTAG=vpefcap<s>,\
    #     STAGES="dual_gate_boot_floor_pefcap dual_gate_fullstack_floor_pefcap",\
    #     UPD_dual_gate_boot_floor_pefcap=4000,UPD_dual_gate_fullstack_floor_pefcap=12000,PRECHECK=1 \
    #     rl/peregrine_vq2_ego.sbatch
    # THE CONCEPTUAL CORE it teaches (Fengyou 2026-07-12): "when you pass a gate and the next-gate slot is
    # EMPTY, do NOT freelance -- TRUST THE COARSE MAP to turn toward the general direction until a gate comes
    # into range, then fill the slot and continue." The levers below make slot1 GENUINELY EMPTY between gates
    # (gate 1 spawned out of view + far gates range-gated out) so the coarse sector is the ONLY next-gate
    # signal, and reward following it. No-spin stays HARD: the coarse map gives a BOUNDED 3x3 turn bucket,
    # never license to scan/spin (no scan behavior is added anywhere).
    # THE PACKAGE (every knob a NEW default-OFF append; the _pef stages + all existing stages stay
    # byte-untouched -- live vpef* safe). It ARMS:
    #   (A) SOFT ATTITUDE-LIMIT penalties (perception-preservation, NOT energy): -rw_att_pitch*relu(|pitch|
    #       -att_pitch_limit_rad) -rw_att_roll*relu(|roll|-att_roll_limit_rad) on the TRUE leveled attitude,
    #       per step, NON-TERMINAL. ZERO inside the free band (incl. the ~17.8deg nose-down REST tilt, below
    #       the 30deg pitch limit) so it never rewards hovering and never kills a flight; grows past the
    #       limit. Conservative weights (pitch 0.5, roll 0.3): at -50deg the pitch penalty is ~0.5*relu(0.873
    #       -0.524)=0.177/step ~ 30% of a bring-up per-step progress (rw_progress 2.0 * ~0.30 m/step ~ 0.60)
    #       and ~13% at race speed -- a real deterrent that does NOT dominate progress. SWEEP UP (0.5->1.0->
    #       2.0) via EXTRA if the head-down behavior persists; RAISE the pitch limit (e.g. 0.6-0.7) if it
    #       over-taxes normal cruise (the 30deg default leaves only ~12deg beyond the 18deg rest tilt).
    #   (B) GATE-1 OUT OF FOV: the sampler REPLACES the segment-0->1 turn with the GEOMETRY-ADAPTIVE turn
    #       (computed PER-ENV from the ACTUAL sampled spawn dist L0 + spacing L1 via the parallax solve, NOT a
    #       fixed angle -- see peregrine_course) that lands gate 1 at a target bearing just past the camera FOV
    #       edge on the gate-0 approach -> acquiring it REQUIRES the coarse-map turn prior (build_coarse_map is
    #       UNCHANGED; it becomes load-bearing because vision no longer covers gate 1 pre-pass). RAMP: boot
    #       target bearing 0.82-0.95 rad (47-54deg, "just past" the 45deg half-HFOV -> gentle) -> fullstack
    #       0.95-1.10 rad (54-63deg, "sharper" -> gate 1 hidden earlier). VERIFIED (rl/gate_visibility + the
    #       REAL sampler): with the knob ON gate-1 detectable-fraction in the pre-pass window collapses vs the
    #       OFF baseline's ~1.0 co-visible; gate 1 is re-acquirable after the turn. horiz sector -> +-1.
    #   (B') MAX-RANGE SLOT-FILL CAP (ego_obs_slot_range_cap_m=30 m): a gate beyond ~30 m estimated range does
    #       NOT fill its slot (BOTH slots). PINS train/deploy parity (deploy range-gates slot fills at ~30 m to
    #       reject far downstream gates -- a 50 m gate-4 seen through the openings must NEVER land in slot1 and
    #       make the drone skip gates 1-3 to dive at gate 4) AND keeps slot1 empty between gates so the coarse
    #       sector is the only next-gate signal. Default +inf == OFF (byte-identical).
    #   (C) SEG-LENGTH RAMP: boot 10-20 m (GENTLE, protect the fragile warm-boot); fullstack 8-20 m (TIGHTER
    #       min -- kills the long-coast+small-tweak shortcut, reduces co-visibility so the map is far more
    #       load-bearing; 8 m is a deliberate hardening margin below the VQ2 real 10-20 m). It is the existing
    #       course_seg_len_lo knob (default 10-20 -> existing stages byte-identical).
    #   (D) NEXT-GATE perception bonus (rw_perception_next) + a budget SPLIT of the current perception:
    #       rw_perception 0.02 -> 0.014, rw_perception_next 0.004 (sum 0.018 < rw_time 0.02 -> farm-neutral,
    #       hover-and-stare nets <= 0/tick). The env GATES the next term on next-gate DETECTABILITY (within
    #       ~30 m + >=4 corners), so it pays 0 while gate 1 is out of view -- it rewards TURNING the coarse-map
    #       direction until the real next gate comes into range and fills slot1 (no camera pull off gate 0
    #       during the approach; no off-gate farm). DEVIATION from _pef-verbatim: this re-allocates the field-
    #       proven current-gate lever -- the split is SWEEPABLE via EXTRA (e.g. 0.012/0.006) keeping sum < 0.02.
    # BOOT/FULLSTACK semantics identical to _pef (boot ego_noise_scale=0.0 calibration; fullstack real
    # noise, +init_from auto-appended). The _pef estimator-faithful + hard-no-spin package rides along
    # VERBATIM. Study before flight: the same _pef dt LAUNCH GATE + acceptance reads apply.
    # ================================================================================================
    "dual_gate_boot_floor_pefcap": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_boot_floor_pef
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,   # (RAMP) GENTLE 10-20 m in the boot (protect the
        #                                                        fragile warm-boot learnability); fullstack -> 8 m
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        "ego_noise_scale": 0.0,       # calibration boot (== _pef): vision noise 0, leveler STILL LIES
        # ---- perception-honesty package (== _percept/_pef verbatim) ----
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,
        "ego_blur_rate_hi_rad_s": 4.0,
        "ego_spin_rate_abort": 3.5,
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,
        # ---- estimator-faithful package (== _pef verbatim) ----
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,
        # ---- pefcap ADDITIONS (default-OFF knobs, ARMED here) ----
        "rw_att_pitch": 0.5, "att_pitch_limit_rad": 0.5235988,   # (A) 30 deg free band (> 17.8 deg rest)
        "rw_att_roll": 0.3, "att_roll_limit_rad": 0.6981317,     # (A) 40 deg free band
        "course_g1_out_of_fov_lo": 0.82, "course_g1_out_of_fov_hi": 0.95,   # (B) target gate1 bearing past the
        #                                              FOV edge (47-54 deg), "just past edge"; the sampler solves
        #                                              the actual turn per-env from the sampled spawn dist + spacing
        "ego_obs_slot_range_cap_m": 30.0,                        # (B') far gates (>30 m est range) don't fill a slot
        "rw_perception": 0.014, "rw_perception_next": 0.004,     # (C) split, sum 0.018 < rw_time 0.02
        "rw_perception_exponent": 4.0,
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from: fresh boot (H6); the fullstack _pefcap stage chains from THIS stage.
                 "++algo.noise_std_hold": 0.30, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    "dual_gate_fullstack_floor_pefcap": {
        **_COMMON,
        "course_n_gates": 2,
        "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,          # == dual_gate_fullstack_floor_pef
        "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
        "course_spawn_yaw_jitter": 0.25,
        "course_seg_len_lo": 8.0, "course_seg_len_hi": 20.0,    # (RAMP) TIGHTER min 10->8 m (kills the long-coast
        #                                                        shortcut + reduces co-visibility; a deliberate
        #                                                        hardening margin below the VQ2 real 10-20 m)
        "course_min_pair_dist_m": 7.0,   # (RAMP) lower the sampler REJECTION floor 10->7 so the 8 m spacing is
        #                                  ACTUALLY realized (at the default 10 m every <10 m course is redrawn ->
        #                                  seg_len_lo=8 silently truncates to 10). Also un-truncates spawn_dist to
        #                                  8 m (harder, intended). Still rejects genuine <7 m overlaps.
        "course_gates_above_spawn": 0.5,
        "floor_at_spawn": True,
        "use_racing_line": True,
        "rw_progress_to_center": False,
        "rw_corridor": 4.0,
        "rw_centering": 0.4, "rw_centering_max_m": 6.0,
        "rw_parabola_crossing": True,
        "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
        # REAL noise (NO ego_noise_scale override) -- the deploy regime; the packages ride along.
        "ego_blur_gate": True,
        "ego_blur_rate_lo_rad_s": 2.0,
        "ego_blur_rate_hi_rad_s": 4.0,
        "ego_spin_rate_abort": 3.5,
        "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 1.5,
        "ego_spin_rev_window_s": 4.0,
        "ego_yaw_cmd_clamp_rad_s": 0.35,
        "ego_faithful": True,
        "ego_est_dt_ticks_hi": 4,
        # ---- pefcap ADDITIONS (default-OFF knobs, ARMED here) ----
        "rw_att_pitch": 0.5, "att_pitch_limit_rad": 0.5235988,   # (A) 30 deg free band (> 17.8 deg rest)
        "rw_att_roll": 0.3, "att_roll_limit_rad": 0.6981317,     # (A) 40 deg free band
        "course_g1_out_of_fov_lo": 0.95, "course_g1_out_of_fov_hi": 1.10,   # (B) SHARPER ramp: target gate1
        #                                              bearing 54-63 deg past the FOV edge (hidden earlier on approach)
        "ego_obs_slot_range_cap_m": 30.0,                        # (B') far gates (>30 m est range) don't fill a slot
        "rw_perception": 0.014, "rw_perception_next": 0.004,     # (C) split, sum 0.018 < rw_time 0.02
        "rw_perception_exponent": 4.0,
        "_raw": {"env.max_time": 60, "algo.gamma": _GAMMA,
                 # NO +init_from here (LOAD-BEARING): the sbatch ladder auto-appends it on stage 2.
                 "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03, "++algo.noise_hold_frac": 0.5,
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
    },
    # ================================================================================================
    # R0 STILL-YAW HOVER BOOT (hover_still_boot, 2026-07-12) -- the FIRST rung (R0) of a FRESH curriculum
    # lineage. GOAL: a fresh-trained HOVER that (a) holds position/altitude and (b) does NOT yaw-dither, so
    # every later flying rung WARM-STARTS from a "still-yaw" base and no downstream stage can co-opt yaw-
    # dither as a tracking crutch -- plus a VERY GENTLE velocity-jerk smoothness prior from birth. OFF-LADDER
    # (STAGES=hover_still_boot); FRESH-START (no init_from). Run:
    #   sbatch --export=ALL,SEED=0,RUNTAG=vr0h0,STAGES="hover_still_boot",\
    #     UPD_hover_still_boot=4000,PRECHECK=1 rl/peregrine_vq2_ego.sbatch
    # PRECHECK=1 MANDATORY on the first launch + the L16 guard: the precheck log must show the NEW per-step
    # loss_components keys emitting -- gate_vhold_reward + velsmooth_pen + yaw_dither_pen + perception_reward +
    # spin_abort_rate + the estimator-faithful keys (eskf_tilt_err_deg_mean / kf_vel_err_mean / sf_mag_g_mean);
    # absent keys == a knob did not arm (in particular gate_vhold_reward absent == the vertical anchor is OFF).
    # BASE = a fixed 15 m LEVEL gate whose FORWARD-HOMING terms are IGNORED by the reward (progress/passage/
    # increment/area/centering/exit -- ZEROED, no coarse map, no parabola crossing), so there is NO forward
    # objective. But the gate is NOT invisible to the reward: it is the STATION-KEEPING ANCHOR -- the drone
    # holds position by keeping the front gate CENTRED (rw_perception, lateral/heading) and at its own altitude
    # (rw_gate_vhold, the world-vertical offset). Both anchors are OBSERVABLE in the position-free obs (the
    # unique optimum is the gate-relative hold, NOT an absolute spawn pose). ARMED FROM UPDATE 0 (the four R0 arms):
    #   (i)   rw_yaw_dither=0.5 -- the anti-dither yaw-jerk penalty (yaw-stillness; ego_reward.yaw_dither_penalty).
    #   (ii)  the FATAL SPIN ABORT package (ego_spin_rate/time/rev aborts) -- REQUIRED to close the slow
    #         CONSTANT-yaw-drift escape the jerk penalty alone leaves (a steady drift has ~0 jerk); the
    #         accumulated-rotation trigger (ego_spin_rev_abort=1.5 rev / 4 s window) is the drift closer. NO-SPIN
    #         stays HARD / BY CONSTRUCTION (fatal abort + yaw clamp fold COLLISION-CLASS), never by reward shaping.
    #   (iii) rw_vel_smooth=1e-4 -- the NEW velocity-jerk prior (VERY gentle; clips only extreme snappy CoM-accel
    #         spikes; steady speed AND smooth hard accel both pay ~0; orthogonal to yaw/roll -- a flip barely
    #         moves the CoM). SWEEPABLE via EXTRA=++env.rw_vel_smooth=...
    #   (iv)  rw_gate_vhold=1.0 -- the give-up-resistant POSITIVE GATE-RELATIVE world-vertical hold (the
    #         OBSERVABLE vertical anchor; REPLACES altitude_hold, whose absolute-Z reference was unobservable ->
    #         R0-v3 sank). Δz = gate_center_z - drone_z = R_wb[2,:]*rel_pos_body -> reconstructable from the obs;
    #         range-independent -> NO homing. Paired with rw_perception=0.02 (the lateral/heading anchor).
    # YAW-CLAMP CALIBRATION FOOTGUN: the yaw-dither penalty prices the APPLIED (post-clamp) yaw-rate delta. So
    # the yaw clamp is ARMED AT THE BASE too -- ego_yaw_cmd_clamp_rad_s=0.35 (== the _pef/nodither lineage) --
    # and rw_yaw_dither stays calibrated FOR THE CLAMPED regime: a rail-FLIP delta = 0.35-(-0.35) = 0.70 rad/s
    # -> -0.5*0.70^2 = -0.245/step. Arming the clamp here (the PREFERRED option) rather than retuning
    # rw_yaw_dither ~20x DOWN for the unclamped +-3.14 rail keeps the still-yaw skill calibrated to the SAME
    # clamp the lineage flies, so it transfers on warm-start. (Deploy note carried from _percept: any flight
    # of a clamped ckpt also needs the fly_rl yaw-only clamp -- N/A for this hover ancestor, which is not flown.)
    # ESTIMATOR-FAITHFUL, LOW noise (R0; hardening ramps in LATER rungs): ego_faithful=true with
    # ego_noise_scale=0.0 (clean VISION noise -- the ESKF/KF are algorithm-structural and noise_scale-
    # INDEPENDENT, so the base already flies the LYING estimator obs, the owner NO-GT directive) and
    # ego_est_dt_ticks_hi=1 (CLEAN 30 Hz leveler ~2 deg err; the dt-curriculum 1->4 hardens in later rungs).
    # The plant knobs (++dynamics.n_substeps=5 + capture_specific_force) MATCH the _pef lineage so the warm-
    # start transfer is a clean SAME-PLANT continuation (n_substeps changes plant trajectories -> a mismatch
    # would strand the transfer). 🚩 same _pef PRE-LAUNCH CHECK: params.transport_delay_steps counts SUBSTEPS.
    # GROUND: floor_at_spawn=true (lethal floor at spawn_z-0.25, the A1 ground-contact-deadlock fix) +
    # standing_start (inherited, standing_start_frac=1.0 -- REQUIRED by floor_at_spawn) so the hover boot can
    # leave the pad; a floor / ceiling / spin exit folds COLLISION-CLASS (gate_collision -> terminal_base +
    # banked forfeit) -> NEVER a free reward exit (rw_parabola_crossing stays OFF so the terminal fires on the
    # non-parabola path where the lethal mask is already inside gate_collision). algo=appo, gamma=0.9975 (both
    # MANDATORY -- ppo leaves the privileged critic disconnected = seed collapse). BUDGET ~4000 upd (fresh
    # discovery of a hover). SUCCESS (training metrics; renders untrustworthy): the GATE-RELATIVE vertical
    # offset |drone_z - gate_center_z| sub-metre (gate_vhold_reward -> ~its 1.0 ceiling), exit_floor ~0,
    # exit_spin/spin_abort_rate ~0 (NON-spinning), exit_timeout dominant, yaw_dither_pen + velsmooth_pen -> ~0
    # by convergence.
    # ================================================================================================
    "hover_still_boot": {
        **_COMMON,
        "course_n_gates": 1,
        "course_spawn_dist_lo": 15.0, "course_spawn_dist_hi": 15.0,   # fixed 15 m level gate (IGNORED by reward)
        "course_drop_lo": 0.0, "course_drop_hi": 0.0,
        "floor_at_spawn": True,                       # lethal floor at spawn_z - 0.25 m (A1 fix; leave the pad)
        # ZERO the gate-homing PULL (progress/passage/centering) -> no forward pull; rw_progress=0 makes
        # progress_to_center INERT (it only swaps the s-formula, still x rw_progress). The 15 m front gate
        # therefore stays a VISUAL ANCHOR only (rw_perception below), NOT a target to fly at.
        "rw_progress": 0.0, "rw_passage": 0.0, "rw_passage_increment": 0.0,
        "rw_area_dist_ref_m": 0.0, "rw_centering": 0.0, "rw_exit_align": 0.0,
        # (iv) VERTICAL ANCHOR = GATE-RELATIVE WORLD-VERTICAL HOLD (Fengyou 2026-07-12; the R0 floor-dive fix).
        # altitude_hold DROPPED (-> 0.0 via _COMMON): its |z - z_spawn| targets ABSOLUTE world-Z, which is
        # NOWHERE in the 21-dim position-free obs -> unlearnable -> R0-v3 sank into the floor (exit_floor 0.42,
        # alt_err 2.2) WITH perception=0.02 already active (perception alone does NOT pin altitude -- it is
        # attitude-coupled: a sinking, pitched-up drone keeps the camera on the gate at ANY altitude). REPLACED
        # by the give-up-resistant POSITIVE bonus on the world-vertical offset to the front gate: r = w * (1 -
        # clip(|drone_z - gate_center_z|/band)). WHY OBSERVABLE: Δz = gate_center_z - drone_z = R_wb[2,:] *
        # rel_pos_body; the third row of R_wb (onto world-Z) is YAW-INVARIANT so it depends only on roll & pitch,
        # and roll_pitch + the body-frame slot0 rel_pos are BOTH in the obs -> the policy CAN reconstruct Δz.
        # RANGE-INDEPENDENT (vertical component only) -> NO homing (zero pull along range; it can never fly the
        # drone into the gate). weight 1.0 mirrors the old altitude_hold magnitude so give-up-resistance holds.
        "rw_gate_vhold": 1.0, "rw_gate_vhold_band_m": 8.0,
        # LATERAL/HEADING ANCHOR (Fengyou 2026-07-12): rw_perception keeps the front gate CENTRED in view ->
        # an OBSERVABLE lateral/heading reference (GT view-angle legal in reward). It is NOT the vertical anchor
        # (it is attitude-coupled and under-determines altitude -- gate_vhold owns vertical); world-lateral
        # needs YAW, which is NOT in the obs, so perception + the yaw clamp are the right lateral tools. Dense
        # (centred > off-centre); no farm risk at a hover. Becomes the YAW anchor at R0.5 when the clamp opens.
        "rw_perception": 0.02, "rw_perception_exponent": 4.0,
        # (i) anti-dither yaw-jerk penalty, calibrated for the CLAMPED regime (see the block comment).
        "rw_yaw_dither": 0.5,
        # (iii) NEW velocity-jerk smoothness prior -- VERY gentle (clips only extreme snappy spikes).
        "rw_vel_smooth": 1.0e-4,
        # (ii) NO-SPIN BY CONSTRUCTION: yaw CLAMPED TO 0 at the base -> drone physically cannot spin. Keep the
        # instantaneous-RATE abort (on obs rates -> OBSERVABLE -> fair; catches genuine tumbles); DROP the
        # rev-ACCUMULATOR abort (Fengyou 2026-07-12: accumulated rotation is NOT in the obs -> unobservable ->
        # an unfair/unlearnable termination; the yaw clamp makes it unnecessary anyway).
        "ego_spin_rate_abort": 3.5, "ego_spin_time_abort": 0.4,
        "ego_spin_rev_abort": 0.0, "ego_spin_rev_window_s": 4.0,   # rev-accumulator DISABLED (unobservable state)
        "ego_yaw_cmd_clamp_rad_s": 0.05,              # yaw PINNED near 0 (tiny ACTIVE clamp). NOT 0.0 -- clamp<=0
                                                      # DISABLES the clamp (full +-3.14 authority); R0-v2 spun 100%.
        # ESTIMATOR-FAITHFUL, LOW noise (R0): clean vision (0.0) + clean 30 Hz leveler (ticks_hi=1).
        "ego_noise_scale": 0.0,
        "ego_faithful": True, "ego_est_dt_ticks_hi": 1,
        "_raw": {"env.max_time": 40, "algo.gamma": _GAMMA,
                 # estimator-faithful plant knobs (== the _pef lineage plant, for a clean same-plant warm-start).
                 "++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True},
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
                       "course_spawn_heading", "course_spawn_yaw_jitter",
                       "course_gates_above_spawn",
                       "course_g1_out_of_fov_lo", "course_g1_out_of_fov_hi",
                       "course_min_pair_dist_m")

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
