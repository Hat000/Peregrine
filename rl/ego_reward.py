"""rl/ego_reward.py -- the REFINED-B (champion-consensus) reward for the VQ2 egocentric generation
(DESIGN.md `docs/vq2-egocentric-gen/DESIGN.md`; inc9 / `peregrine_racing_ego`).

This is the CORRECTED reward the adversarial-critique workflow converged on -- Swift / SkyDreamer /
Geles / MonoRace scale, with EVERY critic fix baked in. It REPLACES the inc7 option-B reward that the
ego env currently reuses via `peregrine_racing.compute_reward_terms` (a to-centre progress + big
passage + tiny terminals). All terms are pure functions of state/event tensors so they unit-test on
the laptop without diffaero; the env calls them under `no_grad` and every geometric quantity is read
from PRIVILEGED GROUND TRUTH (GT drone pos/vel + GT gate centres), NEVER the position-free obs.

============================================================================================
WHY refined-B (the two adversarial critics' verdicts, encoded here as executable choices):
============================================================================================
 * CATASTROPHIC DEFECT (critics' #1): inc8's rw_progress = 10.0/m makes a SPRINT-AND-CLIP a
   positive-return strategy -- the terminal (-25/-50) is 8-40x too small vs the bankable
   +10/m * O(path_len) progress return, so clipping a gate early can still out-earn a clean pass.
   FIX: Geles-scale rw_progress ~= 1.0 AND a terminal that DOMINATES the banked progress (either a
   large fixed magnitude or a PROGRESS-SCALED forfeit). `terminal_penalty` + the rollout check in
   tests/test_ego_reward.py::test_terminal_dominance_rollout tune it until
   E[return | clip at gate j] < E[return | clean pass] for ALL j and BOTH gammas.
 * REFUTED arc-length-LINE progress (the global `ReferenceLine.progress` polyline argmin): the
   global nearest-point advances even for a PERPENDICULAR offset -> it FARMS lateral drift. FIX:
   project onto the CURRENT gate-centre SEGMENT ONLY (a FINITE segment), so perpendicular drift earns
   ZERO progress. `segment_progress_arc`.
 * The passage term carries the anti-corner-cut signal via L-INF centering (MATCHING crossing_events'
   Linf), fires ONCE per gate (idempotent), and the target index STRICTLY increments (bwd crossings
   never pay). `centering_passage_reward` + the env's strictly-incrementing target_gates.

============================================================================================
THE TERMS (each with its weight + sign):
============================================================================================
  R_prog  PROGRESS   dense, PRIMARY, potential-based, PATH-PROJECTED to the CURRENT gate segment:
                     rw_progress * clip(s_curr - s_prev, -vmax*dt, +vmax*dt), s = along-segment arc
                     position on seg [prev_center -> curr_center] clamped to [0, seg_len].
                     rw_progress ~= 1.0 (Geles). vmax*dt from the TRUE peak per-step arc advance.
  R_pass  PASSAGE+CENTERING   sparse, on the GT plane crossing: rw_passage * (1 - e_lat/w_g_half),
                     e_lat = L-INF (max(|y|,|z|)) in-plane offset at the crossing. Fires ONCE per gate
                     index (idempotent), only on a VALID fwd pass, target STRICTLY increments.
                     rw_passage ~= 1.0 (SB), knob to ~4x for the vision-noise regime (Geles).
  TERMINAL  KILL-ON-CONTACT (hard terminal): tuned so a crash is ALWAYS worse than any partial
                     progress. Either a large FIXED magnitude OR PROGRESS-SCALED (forfeit the banked
                     progress return + a base). Whichever PASSES the rollout check.
  T4  finish-time    (rw_finish + rw_finish_time * t_left_s) * 1[finished].  (KEEP)
  R3  time           -0.02/step.  (KEEP)
  R4  free-cone      -rw_tilt * relu(cos(tilt_free) - R33)^2, cone 60 deg (CAP 70).  (KEEP)
  R6/R5  smoothness  MINUSCULE (Geles: 3-4 orders below progress). rate ~1e-3, dact tiny, R7 corner.
                     MUST be <1% of a per-step progress at SLOW bring-up speed.  (KEEP, re-scaled)
  R_exit  NEXT-GATE anticipation (dual_gate stage): reward alignment of the EXIT velocity with the
                     current->next gate-centre bearing. Small weight; on by curriculum stage.
  MISS  WIDE-FLYBY -> MISS terminal: a lateral flyby past a gate's along-track plane without a valid
                     crossing is force-classified as a miss (never silently ignored).
"""
from __future__ import annotations

from dataclasses import dataclass, fields


# ================================================================================================
# THE LOAD-BEARING FOOTGUN: enforce algo=appo + a CONNECTED privileged critic.
# ================================================================================================
def assert_privileged_critic_connected(algo: str, critic_input_dim: int, obs_dim: int,
                                       state_dim: int) -> None:
    """FIRE if the anti-damping guarantee is broken. The "caution emerges from the truth-seeing
    critic" property is FALSE under algo=ppo: GuardedPPO builds a SYMMETRIC obs-only critic, get_state
    is NEVER consumed -> the inc8 seed-collapse root cause. This runtime assert enforces BOTH:
      1. algo == 'appo' (the asymmetric / privileged-critic path), and
      2. the critic actually CONSUMES the GT privileged state (get_state), i.e. critic_input_dim ==
         state_dim AND the critic is NOT the symmetric obs-only critic (critic_input_dim != obs_dim).

    NOTE the privileged-ness is about CONTENT, not dimensionality: the ego critic state (get_state) is
    GT truth the actor never sees (true rel_pos of the window gates + true body velocity/attitude/
    rates), even though it happens to be LOWER-dimensional than the actor obs (the actor also carries
    noisy-obs-only channels -- visible_area, the coarse sector, last_collective -- that the god-view
    critic omits). So the discriminating check is state_dim != obs_dim (the layouts differ) with the
    critic wired to state_dim, NOT state_dim > obs_dim. A symmetric critic (critic_input_dim ==
    obs_dim) is exactly the ppo footgun and FAILS. Called by the env / trainer right after the agent is
    built; raises AssertionError with an actionable message.
    """
    assert str(algo).lower() == "appo", (
        f"[ego-reward] algo={algo!r} but the refined-B anti-damping guarantee REQUIRES algo=appo "
        "(the privileged asymmetric critic). Under algo=ppo GuardedPPO builds a SYMMETRIC obs-only "
        "critic, get_state is never consumed -> the inc8 seed-collapse root cause. Set algo=appo.")
    assert state_dim != obs_dim, (
        f"[ego-reward] the privileged critic state_dim ({state_dim}) EQUALS the actor obs_dim "
        f"({obs_dim}) -- the two layouts are indistinguishable, so a critic built on either is not "
        "provably consuming GT. The get_state layout must differ from the obs (it carries truth the "
        "actor cannot see).")
    assert critic_input_dim == state_dim, (
        f"[ego-reward] the critic input_dim ({critic_input_dim}) does NOT match the privileged "
        f"state_dim ({state_dim}). If it equals obs_dim ({obs_dim}) the critic is SYMMETRIC (the ppo "
        "footgun: get_state never consumed). The critic must consume the full GT privileged state.")

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None
    Tensor = "Tensor"                   # type: ignore


# ================================================================================================
# Weights (cfg-overridable as ``+env.rw_<field>`` / ``+env.<field>``).
# ================================================================================================
@dataclass
class EgoRewardWeights:
    """Refined-B reward weights. Geles/Swift/SB scale -- deliberately NOT the inc8 10.0/m progress
    (the critics proved 10.0 makes sprint-and-clip a positive-return strategy)."""
    # --- PRIMARY dense progress (Geles scale ~1.0) ---
    progress: float = 1.0            # R_prog, per meter of progress-potential advance
    # PROGRESS POTENTIAL MODE (Fengyou 2026-07-07). False (default) -> segment-projected arc position
    # (s = along-track advance on the current gate SEGMENT; perpendicular drift earns ZERO -- the
    # refined-B champion default). True -> 3D distance-to-current-gate-CENTRE potential
    # (phi = -||pos - gate_centre||) == the inc7 / Swift distance-to-gate progress: a dense homing
    # gradient in ALL axes (lateral + vertical + along-track). WHY the switch exists: the 2026-07-07
    # render PROVED segment-only progress STARVES lateral/vertical homing -- on a DEAD-AHEAD gate
    # (azimuth 0) the drone diffused ~8 m laterally / ~6 m vertically and missed (single_gate stuck 0%),
    # because a diagonal flight banks near-full along-track progress while sliding off the line. A true
    # Euclidean potential telescopes over any closed path (sum ~0) so it does NOT "farm lateral drift"
    # (the objection that motivated segment-only was about the NON-potential global polyline argmin, not
    # distance to a POINT). The env swaps ONLY the s computation on this flag; the clip band, banked-
    # progress forfeit, area-distance coupling and passage centering are all unchanged.
    progress_to_center: bool = False
    # ANISOTROPIC VERTICAL WEIGHT for the progress_to_center potential (Fengyou greenlight 2026-07-08, the
    # floor-dive fix). 1.0 == isotropic Euclidean (default, byte-compatible). >1 up-weights the vertical
    # (Z, gravity-loaded) axis in phi = -sqrt(dx^2 + dy^2 + w*dz^2) so a same-height approach does not bury
    # the altitude signal -> sinking costs progress ~w x more. No hover-farm, no ceiling on w (see
    # gate_center_potential). Only consulted when progress_to_center is True.
    progress_vert_weight: float = 1.0
    # v_max clamp: the per-step arc-advance clip band = vmax_mps * dt (m/step). vmax_mps derived from
    # the TRUE peak speed (~30 m/s) + ~30% headroom (-> 39 m/s), so the clamp trims only UNPHYSICAL
    # bursts, never legit top speed. dt is passed at call time (env control dt, ~1/30 s).
    vmax_mps: float = 39.0

    # --- PASSAGE + centering (SB ~1.0; knob to ~4x for the vision-noise regime) ---
    passage: float = 1.0             # R_pass BASE, on (1 - e_lat / w_g_half) at the crossing
    # PER-GATE passage increment (Fengyou 2026-07-07): the passage weight for gate g is
    # ``passage + passage_increment * g`` so passing a LATER gate pays more (gate 0 -> base, gate 1
    # -> base+inc, ...). Rewards getting DEEPER into the course (a partial run that reaches gate 3 beats
    # one that only reaches gate 1). 0 == flat passage (no increment). Sparse (fires once per gate) so
    # it is non-farmable, like the base passage.
    passage_increment: float = 1.0

    # --- AREA-DISTANCE coupled progress (Fengyou 2026-07-07) ---
    # Scale the (positive) progress reward by how SQUARE-ON the gate is -- but ONLY when close. Far out a
    # beeline at an angle is fine (you straighten out); in the final metres a shallow/off-axis approach
    # leads to a wide, clipped exit, so it should earn LESS -> the policy learns to square up before the
    # gate. factor = area + (1-area)*clip(dist/ref, 0, 1): dist>=ref -> 1 (no angle penalty far out),
    # dist->0 -> area (|cos(view_ray, gate_normal)| in [0,1]). ``area_dist_ref_m`` is the ramp distance
    # (m) over which the coupling engages; <=0 == coupling OFF (factor == 1 everywhere).
    area_dist_ref_m: float = 6.0

    # --- DENSE LATERAL CENTERING (Fengyou 2026-07-07) ---
    # A per-step pull onto the CURRENT gate-centre segment so the drone crosses CENTERED, not wide. The
    # area coupling shapes the approach ANGLE; THIS shapes the lateral POSITION. The render (2026-07-07)
    # showed 93% WIDE miss with the dive SOLVED -- the drone reliably reaches the plane and crosses
    # off-aperture, a pure centering-GRADIENT gap (the reward already pays a centred crossing +25 vs a
    # wide miss -30, so it WANTS to centre; it just has no dense signal for HOW). Penalty =
    # -rw_centering * clamp(perp_dist, 0, max). At convergence (on-line, perp~0) it is ZERO, so it only
    # makes OFF-line flight costly -- it cannot be farmed and adds nothing once the policy flies centred.
    centering: float = 0.0           # rw_centering; 0 == OFF (turned ON by the curriculum)
    centering_max_m: float = 2.0     # clamp (m) on the perpendicular offset so a big early drift is bounded

    # --- HOVER-HOLD altitude probe (Fengyou greenlight 2026-07-08; the H1-vs-H2 disambiguator) ---
    # A GIVE-UP-RESISTANT POSITIVE altitude-hold bonus for the `hover_hold` diagnostic stage ONLY (no gate
    # homing/passage): r = altitude_hold * (1 - clip(|z - z_spawn| / band, 0, 1)). It PEAKS (+altitude_hold)
    # at the spawn altitude and decays linearly to 0 at |Δz| >= band -> spawn altitude is the UNIQUE optimum,
    # AND the reward is POSITIVE everywhere, so ending the episode FORFEITS the future bonus == the policy is
    # paid to SURVIVE at altitude. That is why this form carries NO give-up incentive, unlike a -k|Δz|
    # MAGNITUDE penalty (the rw_centering / through_centering back-fire class, which can drive a FAILING
    # policy to floor sooner to stop the accruing bleed and thus CONFOUND a control-learning read). With no
    # competing forward objective, altitude-hold is strictly optimal here, so a FLOOR exit is an unambiguous
    # CONTROL-LEARNING (H2) signal rather than reward give-up. 0 == OFF (default; off on every non-probe stage).
    altitude_hold: float = 0.0       # rw_altitude_hold; 0 == OFF
    altitude_hold_band_m: float = 8.0  # (m) half-width over which the hold bonus decays to 0 (restoring
    #                                    gradient present over the whole reachable descent range)

    # --- GATE-RELATIVE WORLD-VERTICAL HOLD (Fengyou 2026-07-12; the R0 floor-dive fix -- the OBSERVABLE
    # replacement for altitude_hold's UNOBSERVABLE absolute-Z reference). SAME give-up-resistant POSITIVE
    # form as altitude_hold, but the reference is the CURRENT target gate's world-Z, not the spawn altitude:
    #   r = gate_vhold * (1 - clip(|drone_z - gate_center_z| / band, 0, 1))
    # WHY OBSERVABLE (the whole point): the rewarded quantity is the WORLD-vertical gate offset
    # Δz = gate_center_z - drone_z = R_wb[2,:] · rel_pos_body. The third row of R_wb (the map onto world-Z)
    # is YAW-INVARIANT -> it depends only on roll & pitch, and BOTH the roll_pitch channel AND the body-frame
    # slot0 rel_pos are in the 21-dim ego obs, so a sufficient policy CAN reconstruct Δz from its obs -- unlike
    # absolute Z (altitude_hold's z/z_spawn), which is NOWHERE in the obs (the unlearnable-state bug that sank
    # R0-v3, exit_floor 0.42). Gate-RELATIVE (a difference -> no world coordinate) and RANGE-INDEPENDENT (the
    # pure vertical component) -> it pins altitude to the gate WITHOUT any homing gradient (zero pull along
    # range -> it can NEVER fly the drone into the gate; the fly-into-gate crash the owner forbade cannot arise
    # here). WHY a POSITIVE bonus, NOT a -k|Δz| penalty: the floor-dive was a GIVE-UP-and-end-episode failure;
    # a magnitude penalty is minimized by TERMINATING (floor sooner to stop the bleed) -> it reintroduces
    # exactly that. The positive "paid to survive" form (ending forfeits the future bonus) is give-up-RESISTANT
    # -- the property that made altitude_hold's FORM right; only its reference FRAME was wrong. Lateral/heading
    # is left to rw_perception + the yaw clamp (world-lateral needs YAW, which is NOT in the obs -> not
    # reconstructable; vertical is the yaw-invariant axis that IS observable AND the axis that was failing).
    # 0 == OFF (byte-identical). Tune via +env.rw_gate_vhold / +env.rw_gate_vhold_band_m.
    gate_vhold: float = 0.0          # rw_gate_vhold; gate-relative world-vertical hold bonus; 0 == OFF
    gate_vhold_band_m: float = 8.0   # (m) half-width over which the vertical-offset bonus decays to 0

    # --- MPCC CONTOURING (Fengyou greenlight 2026-07-08; the floor-dive lever, hover-hold-confirmed) ---
    # A give-up-RESISTANT PBRS (telescoping potential) term on the PERPENDICULAR deviation from the current
    # gate-centre segment: phi_corr = -perp, reward = corridor * clip(perp_prev - perp_curr, band). Positive
    # when the drone moves TOWARD the line (perp shrinks), negative when it drifts off. Pairs with the
    # ALONG-TRACK LAG progress (progress_to_center=False / segment_arc_position) to form the MPCC lag+
    # contouring decomposition: LAG drives forward, CONTOURING supplies the vertical+lateral homing that
    # segment-arc lag omits -- REPLACING the isotropic gate_center_potential's vertical component (no
    # double-count). WHY PBRS not the raw through_centering PENALTY: the penalty accrues a standing tax on a
    # centred mean under held noise + invites give-up (end early to stop the bleed = the ego_ctr 72%-OOB
    # class); a telescoping potential has E[Δ]~0 at a centred mean (zero standing tax) and its episode sum
    # telescopes to a boundary term (ending early yields no escape). The hover-hold probe (job 3297613:
    # 91% hover, alt_err 1.2m) PROVED altitude control is learnable from a clean vertical gradient; this
    # supplies exactly that gradient during forward transit. 0 == OFF (default). γ=1 differencing (matches
    # the along-track progress term's convention); the γ<1 residual is a negligible off-line penalty.
    corridor: float = 0.0            # rw_corridor (PBRS contouring weight); 0 == OFF
    corridor_clip_mps: float = 39.0  # clip band (m/step = mps*dt) trimming the gate-handoff re-projection burst

    # --- GVF DIRECTION-ALIGNMENT (Fengyou 2026-07-08 -- the TRUE vector-field reward). Unlike the telescoping
    # contouring (which rewards MOVING toward the line and pays 0 for a parallel-flying standing offset), this
    # rewards the velocity DIRECTION following the guiding field F everywhere: r = align * dot(v_hat, F_hat),
    # F = cos(theta)*tangent + sin(theta)*inward, theta = atan(align_gain*perp). On the line -> follow the
    # tangent; off it -> angle inward, so a parallel drone is MIS-aligned and pressured to turn onto the path
    # (no blind spot). NON-telescoping (direction, not displacement). Speed-blind ([-1,1]); pair with progress
    # for the speed incentive. align_gain = the CONVERGENCE TIGHTNESS: HIGH -> sharp corner onto the line, LOW
    # -> smooth wide asymptotic curve (Fengyou's "how tight a turn"). 0 == OFF.
    align: float = 0.0               # rw_align (GVF alignment weight); 0 == OFF
    align_gain: float = 1.0          # convergence tightness (theta = atan(align_gain*perp)); higher = sharper

    # --- PERCEPTION reward (Fengyou 2026-07-09; the Swift/Geles field-proven CENTERING lever). A dense,
    # POSITIVE per-step bonus for keeping the camera optical axis pointed at the gate CENTRE:
    #   r_perc = perception · exp(−δ_cam^perception_exponent),  δ_cam = angle(optical axis, drone→gate-centre)
    # (radians; cos δ_cam from gate_visibility.gate_center_view_cos). Peaks (+perception) with the gate dead-
    # centre in view, decaying as it drifts to the frame edge. WHY: Swift AND Geles both carry this term and
    # NOTHING else in their reward, yet it takes gate-passing error from ~0.5 m (no r_perc) to ~0.12–0.22 m
    # (Geles Table I) -- our ~0.88 m L-inf floor is ~the field's no-r_perc baseline. Mechanism = keeping the
    # gate centred in the FOV yields a better estimate on approach (the concrete form of "reduce the noise
    # arriving to the policy") AND an attention pressure to fly at the gate. Positive + dense (like Swift's
    # r_perc, subtracted-crash convention aside); it cannot be farmed off-gate (points AT the gate). 0 == OFF
    # (default -> byte-identical). Geles uses λ₂=0.025, exponent 4; Swift ≈ same. Tune via +env.rw_perception.
    perception: float = 0.0          # rw_perception; 0 == OFF
    perception_exponent: float = 4.0 # δ_cam power inside the exp (Geles/Swift = 4)
    # --- NEXT-GATE perception bonus (pefcap package, 2026-07-12; the coarse-map-load-bearing lever). The
    # SAME Swift/Geles exp(−δ^exponent) form applied to the NEXT target gate's optical-axis angle, rewarding
    # the policy for pointing at / acquiring the UPCOMING gate. The env GATES this on the next gate being
    # DETECTABLE (it passes cos ≈ −1 -> term ~0 while the next gate is out of view / absent / behind), so it
    # cannot pull the camera off the current gate during the approach and cannot be farmed by pointing at an
    # unseen gate. FARM-NEUTRALITY: perception + perception_next must stay <= rw_time (a hover-and-stare nets
    # <= 0/tick) -- enforced in __post_init__. 0 == OFF (byte-identical). Tune via +env.rw_perception_next.
    perception_next: float = 0.0     # rw_perception_next; 0 == OFF
    # --- ATTITUDE-LIMIT penalties (pefcap package, 2026-07-12; PERCEPTION-PRESERVATION, NOT energy). SOFT,
    # NON-TERMINAL per-step penalties on EXCESS leveled body attitude beyond a free band, on the TRUE leveled
    # attitude (GT legal in reward): −att_pitch·relu(|pitch|−att_pitch_limit_rad) − att_roll·relu(|roll|−
    # att_roll_limit_rad). They target the diagnosed failure (the policy pitches head-down to ~−50deg -> the
    # +20deg camera points at −30deg -> LOSES the gate; high-g banks break perception the same way) by pushing
    # a self-limit INTO the trained policy so it does not need a deploy-side control clamp. Deliberately NOT an
    # |omega|/thrust/jerk/effort penalty (fast flight is fine -- speed is NOT penalized); the penalty is ZERO
    # inside the band, INCLUDING the airframe's ~17.8deg (0.31 rad) nose-down REST tilt which sits far below the
    # 60deg default pitch limit, so it never rewards hovering (hover pays 0 here, same as flying within the
    # band) and never kills a flight (purely a reward term, never a termination). Grows linearly past the
    # limit. 0 == OFF (byte-identical). Tune via +env.rw_att_pitch / +env.rw_att_roll.
    #
    # DEFAULT CAP ANGLES = 60 deg for BOTH axes (owner Fengyou, 2026-07-13 Track A): the deploy roll/pitch
    # clamps this term REPLACES were ~60 deg, and the owner wants ROLL self-limited at ~60 deg. The PITCH
    # default is deliberately LOOSE (also 60 deg, up from the old 30 deg) because pitch's real failure lever is
    # NOT the attitude cap: the head-down dive that lost gates was ~-50deg -- INSIDE any 60deg cone -- and is a
    # PERCEPTION loss (the +20deg camera points at the floor), addressed by rw_perception, NOT by a tighter
    # pitch cap. So a stage that wants head-down pressure should lean on rw_perception and leave this pitch cap
    # as a loose >60deg BACKSTOP (a tighter pitch band is still available per-stage via +env.att_pitch_limit_rad,
    # e.g. the _pefcap 30deg / _attcap 12deg experiments, which set it explicitly). Both limits are per-axis and
    # independently tunable. Changing these DEFAULTS is reward-neutral while the weights are 0 (relu*0 -> 0), so
    # the byte-identical OFF guarantee is untouched.
    att_pitch: float = 0.0           # rw_att_pitch; weight on the |leveled pitch| excess; 0 == OFF
    att_pitch_limit_rad: float = 1.0471976   # 60 deg free band on |pitch| (LOOSE backstop; pitch's lever is rw_perception)
    att_roll: float = 0.0            # rw_att_roll; weight on the |leveled roll| excess; 0 == OFF
    att_roll_limit_rad: float = 1.0471976    # 60 deg free band on |roll| (owner Fengyou: roll capped ~60 deg)

    # --- SMOOTH PARABOLIC CROSSING reward (Fengyou 2026-07-08 -- "policy reacts better to smooth things").
    # Replaces the DISCONTINUOUS {thread=+passage, clip=-100, miss=-100} cliff with one smooth downward
    # parabola of the crossing offset e (L-inf): r = clamp(cross_center * (1 - (e/cross_zero_m)^2), -cross_neg_
    # cap, cross_center). +cross_center dead-centre, 0 at the aperture edge (cross_zero_m = half-opening),
    # growing NEGATIVE outside and capped. SMOOTH + MONOTONIC -> no cliff, and NO MOAT (closer is ALWAYS
    # better) -- the property frame_clip_is_miss failed to give (it removed the penalty and floor-dived; this
    # keeps a growing penalty). When ON: the passage reward is REPLACED and the frame-clip/miss TERMINAL
    # penalties are dropped (floor + oob keep theirs). parabola_crossing==False -> legacy.
    parabola_crossing: bool = False
    cross_center: float = 20.0       # peak (dead-centre) crossing reward
    cross_zero_m: float = 0.75       # offset where the parabola crosses 0 (== the gate half-opening / aperture edge)
    cross_neg_cap: float = 100.0     # floor on the (negative) wide-crossing penalty (avoids a giant terminal)
    # --- GRADED ANTI-CLIP terminal ON TOP of the parabola (audit-C5 fix, 2026-07-09; from_cfg key
    # ``rw_clip_terminal``). When parabola_crossing is ON the frame-clip/miss TERMINAL penalties are dropped
    # (only floor+oob keep theirs), so at the de-facto fixed cross_zero_m=4.0 a sub-aperture FRAME STRIKE -- a
    # competition DQ -- earns ~+19 of the +20 peak, ~indistinguishable from a clean thread, and the perfect-
    # estimator run kept a 35% clip rate with NO reward pressure against it. clip_terminal_w>0 ADDS a flat
    # -clip_terminal_w on a frame strike (the collision the parabola dropped; the floor dive already pays its
    # terminal), restoring graded anti-clip pressure while the parabola still pays. Plain mutable python-float
    # attribute read FRESH each step, so the train-loop anneal machinery (peregrine_train_ego._unwrap_env_with,
    # mutating _egorw attributes per update) can anneal it 0->W with no extra plumbing. 0 == OFF (byte-identical).
    clip_terminal_w: float = 0.0     # rw_clip_terminal; flat extra frame-strike terminal when parabola on; 0 == OFF
    # --- ONCE-PER-GATE parabola latch (audit red-flag re-payment farm; from_cfg key ``rw_parabola_latch``).
    # The parabola fires on ``crossed``==fwd_t, which -- UNLIKE the idempotent passage -- is NOT once-per-gate: if
    # miss_terminates=false is ever combined with parabola_crossing, a wide forward crossing neither advances the
    # target nor terminates, so an oscillating drone re-crosses the SAME target plane and FARMS the parabola every
    # step. True gates crossing_parabola_reward on a caller-maintained per-env ``parabola_paid`` latch so each gate
    # pays at most once per episode (the caller clears it on a target ADVANCE and on episode RESET, mirroring the
    # passage's strictly-incrementing-target idempotency). False (default) == byte-identical (no current stage sets
    # miss_terminates=false + parabola, so this is a defensive knob).
    parabola_latch_once: bool = False  # rw_parabola_latch; once-per-gate parabola payment; False == OFF (legacy)

    # --- TERMINAL (kill-on-contact). Two modes; ``terminal_progress_scaled`` picks. ---
    # FIXED mode: penalty = terminal_base (large, dominates the banked progress return).
    # PROGRESS-SCALED mode: penalty = terminal_base + accumulated_progress_return (clipping forfeits
    # ALL banked progress + a base -> clipping can never out-earn continuing). The rollout check in
    # tests/test_ego_reward.py tunes terminal_base until dominance holds for both gammas.
    terminal_base: float = 200.0     # base contact/miss/oob penalty magnitude (>0; subtracted)
    terminal_progress_scaled: bool = True   # True -> forfeit banked progress + base (RECOMMENDED)
    # Separate knobs so miss/oob can differ from a frame strike if ever wanted (default: all == base).
    terminal_miss: float = 200.0
    terminal_oob: float = 200.0

    # --- T4 finish (KEEP) ---
    finish: float = 20.0
    finish_time: float = 1.0

    # --- R3 time (KEEP) ---
    time: float = 0.02

    # --- R4 free-cone (KEEP; 60 deg, CAP 70) ---
    tilt: float = 4.0
    tilt_free_rad: float = 1.0471976        # 60 deg
    tilt_free_cap_rad: float = 1.2217305    # 70 deg hard cap on the relaxed cone

    # --- SMOOTHNESS (MINUSCULE: 3-4 orders below progress; <1% of a bring-up-speed per-step prog) ---
    rate: float = 1.0e-3             # R6 on ||omega||
    dact: float = 1.0e-3             # R5 on ||Delta action||^2
    corner: float = 0.0              # R7 joint corner tax (PREFERRED smoothness); 0 == OFF by default

    # --- ANTI-DITHER YAW SMOOTHNESS (nodither fine-tune, 2026-07-12; AUTHORIZED anti-dither, NOT energy).
    # A SMOOTHNESS penalty on the temporal CHANGE of the yaw-rate COMMAND: R_yawdith = -yaw_dither *
    # (yaw_cmd_t - yaw_cmd_{t-1})^2, on the APPLIED (post-clamp) yaw action channel (channel 3, rad/s). It
    # targets the DITHER the position-free egocentric obs leaves UNPRICED (no yaw channel -> yaw is reward-
    # INDIFFERENT -> the command rails the +-clamp and SIGN-FLIPS ~30% of ticks on flights). WHY a squared
    # JERK (temporal change) and NOT a magnitude/|omega|/energy penalty (owner directive): a SUSTAINED,
    # steady yaw command (a needed downstream TURN) has Delta~0 -> pays ~0, so a smooth turn / speed is
    # NEVER penalised; ONLY the oscillation transient pays (a +-clamp rail-flip Delta=2*clamp -> heavy). WHY
    # squared-jerk over a SIGN-FLIP INDICATOR (the considered alternative): the indicator is DISCONTINUOUS
    # (bad gradient; "policy reacts better to smooth things") AND magnitude-BLIND -- it taxes benign tiny
    # jitter around zero-yaw straight cruise EXACTLY as much as a full rail-flip (a standing tax on straight
    # flight), whereas the squared form pays ~0 for near-zero jitter and the MOST for a rail-flip -- the
    # exact dither discrimination wanted -- and vanishes at convergence (steady command -> Delta 0 -> 0, so
    # it is NON-farmable, min 0 at steady yaw). The CONSTANT-DRIFT escape (a slow STEADY yaw = 0 jerk = a
    # slow spin that dodges THIS term) is closed BY CONSTRUCTION by the RETAINED fatal spin abort (the
    # ego_spin_rev_* accumulated-rotation trigger + the yaw clamp) -- this term kills the dither, the abort
    # kills the constant-spin escape; keep BOTH. 0 == OFF (byte-identical). Tune via +env.rw_yaw_dither.
    yaw_dither: float = 0.0          # rw_yaw_dither; weight on (delta yaw_cmd)^2 [per (rad/s)^2]; 0 == OFF

    # --- VELOCITY-JERK smoothness prior (R0 still-yaw hover boot, 2026-07-12; AUTHORIZED smoothness, NOT
    # energy/speed). A SMOOTHNESS penalty on the temporal CHANGE of the WORLD-frame CoM acceleration:
    # R_velsmooth = -vel_smooth * ||jerk||^2, jerk = accel_curr - accel_prev (the 1st difference of
    # acceleration == the 2nd time-difference of velocity; accel = (v_t - v_{t-1})/dt, world Z-up). It
    # penalises snappy JERK SPIKES, NOT velocity or acceleration magnitude: a STEADY SPEED (accel 0) AND a
    # SMOOTH HARD acceleration (accel constant, jerk 0) both pay ~0 -- ONLY a SUDDEN change of acceleration
    # bites, so it is NOT a speed/energy penalty (fast, hard, smooth flight is free). ORTHOGONAL to the
    # yaw/roll control mechanism BY CONSTRUCTION: a yaw (or roll) flip barely moves a quad's centre of mass,
    # so the WORLD-velocity jerk is ~0 and pays ~0 -- deliberately the world CoM jerk and NOT the body
    # specific force, whose ~1 g gravity component ROTATES with attitude and would spuriously TAX every
    # roll/pitch as if it were a CoM jerk. Squared (SMOOTH + magnitude-aware): benign near-zero jitter pays
    # ~0, a big spike pays the most, and it vanishes at convergence (steady accel -> jerk 0 -> 0, non-
    # farmable). VERY gentle when armed (Fengyou: "just clip extreme snappy spikes"). Sign NEGATIVE. The env
    # threads the previous world velocity + acceleration and passes accel_curr/accel_prev (None -> the term
    # is 0). 0 == OFF (byte-identical). Tune via +env.rw_vel_smooth.
    vel_smooth: float = 0.0          # rw_vel_smooth; weight on ||jerk||^2 [per (m/s^2)^2]; 0 == OFF

    # --- VELOCITY-CAP soft-hinge (anti-velocity-runaway package, 2026-07-13; NOT a speed reward, NOT
    # energy). The champion (_pef / vpeffs0) has a velocity RUNAWAY: its optimal speed is ~8-9 m/s but it
    # accelerates to ~12 m/s where control fails. A ONE-SIDED SOFT-HINGE penalty on TOTAL SPEED ||v|| that
    # is EXACTLY ZERO at/below a soft threshold and ramps up QUADRATICALLY above it, prohibitive near a hard
    # threshold:
    #   R_vcap = -v_cap * (relu(||v|| - v_cap_soft) / (v_cap_hard - v_cap_soft))^2
    # so ``v_cap`` is the per-step penalty magnitude AT the hard cap (||v||==v_cap_hard -> normalised
    # excess 1 -> -v_cap), growing > v_cap super-linearly BEYOND it (the runaway tail is prohibitive). This
    # is DELIBERATELY NOT a speed reward and MUST NOT change the reward below v_cap_soft (owner Fengyou,
    # explicit: "no speed term, don't lose rewards elsewhere") -- below the soft knee the relu is 0 so the
    # term is EXACTLY 0 (byte-identical to the proven _pef reward across the whole useful speed band), and
    # the quadratic makes it C^1 at the knee (value AND slope 0 -> no cliff; "the policy reacts better to
    # smooth things"), MONOTONE increasing in magnitude above it. ONLY-OBSERVABLE-STATE (owner directive,
    # non-negotiable): the penalised quantity is TOTAL SPEED ||v||, which the actor CAN reconstruct from its
    # obs -- the 21-dim ego obs carries body velocity (FLU) at obs[0:3], and ||v_body|| == ||v_world|| (a
    # rotation preserves the norm), so the cap prices ONLY a quantity the policy sees. The env computes ||v||
    # from GT world velocity (GT is legal in the reward; the observability constraint is about what is
    # PRICED, not what it is computed from). NOT an energy penalty -- it is 0 across the entire useful speed
    # band and bites only the unphysical runaway tail. 0 == OFF (byte-identical). Tune via +env.rw_v_cap /
    # +env.rw_v_cap_soft / +env.rw_v_cap_hard.
    v_cap: float = 0.0               # rw_v_cap; per-step penalty magnitude AT the hard cap; 0 == OFF
    v_cap_soft: float = 9.0          # (m/s) soft threshold; reward UNCHANGED at/below this (the relu knee)
    v_cap_hard: float = 12.0         # (m/s) hard threshold; normalised excess==1 here (penalty == v_cap)

    # --- NEXT-GATE exit-line anticipation (dual_gate+ stage; small; on by curriculum) ---
    exit_align: float = 0.0          # R_exit weight; 0 == OFF (single_gate/handoff stages)

    # --- FIXED-terminal guard bounds (folded-forward fix #2). Only consulted when terminal_progress_
    # scaled is False: the FIXED terminal_base must exceed the MAX bankable progress return for the
    # course, else sprint-and-clip-at-the-last-gate out-earns a clean pass (the exact critics' defect).
    # Max bankable progress return (undiscounted upper bound) = progress * (gates * seg_len) [the total
    # along-track arc]. These two knobs describe the LARGEST course the FIXED base is certified for; on a
    # course bigger than this the FIXED mode is FORBIDDEN (use the progress-scaled default instead).
    guard_max_course_gates: int = 6         # certify FIXED base up to this many gates ...
    guard_max_seg_len_m: float = 45.0       # ... at this per-segment length (VQ1-derived worst case)

    def __post_init__(self):
        """FOLDED-FORWARD FIX #2 -- guard the NON-DEFAULT FIXED-terminal footgun. The default
        (terminal_progress_scaled=True) forfeits banked progress + base by construction, so dominance
        holds for ANY course length -- no guard needed. But if a run FORCES the FIXED mode
        (terminal_progress_scaled=False), a fixed terminal_base that is SMALLER than the max bankable
        progress return lets sprint-and-clip-at-the-last-gate out-earn a clean pass (the critics'
        catastrophic defect). Assert terminal_base (and terminal_miss/oob) EXCEED the max bankable
        progress return for the certified course; otherwise raise so a mis-set FIXED run fails loudly at
        construction instead of silently training a gate-clipper. progress-scaled -> no-op.

        FARM-NEUTRALITY GUARD (Stage-1 vtrackAr5): additionally, UNCONDITIONALLY assert rw_perception <=
        rw_time. r_perc (the pointing carrot) must never out-earn the per-step time cost, else
        hover-and-stare at the gate nets a POSITIVE per-tick return and the policy farms the carrot
        instead of racing (the exact class of mis-set positive weight rs0 shipped). Checked FIRST, before
        the perception_next / v_cap guards and the terminal_progress_scaled early-return, so it fires for
        every stage. Byte-identical at the vetted trackA weights (rw_perception 0.02 == rw_time 0.02);
        only a DETONATED rw_perception (e.g. 0.05) raises. Mirrors the curriculum's stage-level rule
        (tests/test_vq2_ego_curriculum.py); a deliberate perception-dose-response study must lower
        rw_perception to <= rw_time or is blocked."""
        assert self.perception <= self.time + 1e-9, (
            f"[ego-reward] FARM-NEUTRALITY VIOLATION: rw_perception ({self.perception}) exceeds rw_time "
            f"({self.time}). The pointing carrot must be <= the per-step time cost, else hover-and-stare "
            f"nets a positive per-tick return (a reward FARM). Lower rw_perception to <= rw_time (0.02 is "
            f"the vetted value) or, for a deliberate dose-response study, run it knowing this guard fires.")
        # FARM-NEUTRALITY GUARD (perception_next, pefcap 2026-07-12): a POSITIVE next-gate perception bonus
        # must keep perception + perception_next <= rw_time so a hover-and-stare (point at both gates, make
        # no progress) nets <= 0/tick and is never a positive-return strategy (the same bound the existing
        # rw_perception respects: rw_perception <= rw_time). Checked FIRST so it runs under the default
        # terminal_progress_scaled=True (which returns early below). perception_next==0 -> no-op (byte-id).
        if self.perception_next > 0.0:
            assert self.perception + self.perception_next <= self.time + 1e-9, (
                f"[ego-reward] rw_perception ({self.perception}) + rw_perception_next "
                f"({self.perception_next}) = {self.perception + self.perception_next:.4g} EXCEEDS the farm-"
                f"neutrality ceiling rw_time ({self.time}); a hover-and-stare would net > 0/tick. Split the "
                "perception budget so perception + perception_next <= rw_time.")
        # VELOCITY-CAP sanity (anti-velocity-runaway 2026-07-13): the hard cap must EXCEED the soft cap (the
        # hinge span v_cap_hard - v_cap_soft is the quadratic normaliser). Checked BEFORE the early return so
        # it runs under the default terminal_progress_scaled=True. v_cap==0 -> no-op (byte-identical).
        if self.v_cap > 0.0:
            assert self.v_cap_hard > self.v_cap_soft, (
                f"[ego-reward] rw_v_cap_hard ({self.v_cap_hard}) must EXCEED rw_v_cap_soft "
                f"({self.v_cap_soft}) -- the soft-hinge ramps quadratically over (soft, hard]. Set "
                "rw_v_cap_hard > rw_v_cap_soft (defaults 12 > 9).")
        if self.terminal_progress_scaled:
            return
        max_bankable = self.progress * self.guard_max_course_gates * self.guard_max_seg_len_m
        for name, base in (("terminal_base", self.terminal_base),
                           ("terminal_miss", self.terminal_miss),
                           ("terminal_oob", self.terminal_oob)):
            assert base > max_bankable, (
                f"[ego-reward] FIXED terminal ({name}={base}) does NOT exceed the max bankable progress "
                f"return ({max_bankable:.1f} = rw_progress {self.progress} * {self.guard_max_course_gates} "
                f"gates * {self.guard_max_seg_len_m} m). A FIXED terminal below the banked progress lets "
                "sprint-and-clip-at-the-last-gate out-earn a clean pass (the critics' catastrophic "
                "defect). Either raise the terminal base above the bankable progress, SHRINK the "
                "certified course (guard_max_course_gates/guard_max_seg_len_m), or -- RECOMMENDED -- use "
                "the default terminal_progress_scaled=True (dominates for ANY course length).")

    @classmethod
    def from_cfg(cls, cfg) -> "EgoRewardWeights":
        kw = {}
        for f in fields(cls):
            if f.type == "bool" or isinstance(f.default, bool):
                kw[f.name] = bool(getattr(cfg, f"rw_{f.name}",
                                          getattr(cfg, f.name, f.default)))
            else:
                kw[f.name] = float(getattr(cfg, f"rw_{f.name}",
                                           getattr(cfg, f.name, f.default)))
        # SHORT-KEY ALIASES: these two knobs use a terser cfg key than the generic ``rw_<field>`` (so the CLI
        # override string stays short): ``rw_clip_terminal`` -> clip_terminal_w, ``rw_parabola_latch`` ->
        # parabola_latch_once. Fall back to the value the generic loop already resolved (== the default when the
        # key is absent) so a cfg mentioning NEITHER key is byte-identical to the pre-audit behaviour.
        kw["clip_terminal_w"] = float(getattr(cfg, "rw_clip_terminal", kw["clip_terminal_w"]))
        kw["parabola_latch_once"] = bool(getattr(cfg, "rw_parabola_latch", kw["parabola_latch_once"]))
        return cls(**kw)


# ================================================================================================
# R_prog: PATH-PROJECTED progress onto the CURRENT gate-centre SEGMENT (perpendicular drift -> 0).
# ================================================================================================
def segment_arc_position(pos: Tensor, seg_start: Tensor, seg_end: Tensor) -> Tensor:
    """Along-segment arc position s = clamp(dot(pos - seg_start, seg_dir), 0, seg_len) for the FINITE
    segment [seg_start -> seg_end]. This is the CORE of the corrected progress: projecting onto a
    finite segment (NOT the global polyline argmin) makes a PERPENDICULAR offset advance s by ZERO --
    only motion ALONG the segment (toward the current gate centre) earns progress.

    pos/seg_start/seg_end: (N,3) GT world (Z-up). Returns s (N,) in [0, seg_len]. A degenerate
    zero-length segment (prev_center == curr_center) returns 0 (no along-track axis; nothing to farm).
    """
    assert torch is not None
    d = seg_end - seg_start                                   # (N,3) segment vector
    seg_len = torch.linalg.norm(d, dim=-1)                    # (N,)
    safe = seg_len.clamp(min=1e-9)
    u = d / safe.unsqueeze(-1)                                # unit along-track (junk where degenerate)
    s = ((pos - seg_start) * u).sum(dim=-1)                   # signed projection
    s = s.clamp(min=torch.zeros_like(seg_len), max=seg_len)   # clamp to the FINITE segment
    # degenerate segment -> no along-track axis; return 0 so nothing accrues
    s = torch.where(seg_len < 1e-9, torch.zeros_like(s), s)
    return s


def gate_center_potential(pos: Tensor, gate_center: Tensor, vert_weight: float = 1.0) -> Tensor:
    """Progress potential s = -sqrt(dx^2 + dy^2 + vert_weight*dz^2) (N,), Z-up GT. Fed to
    ``segment_progress_reward``, reward = clip(s_curr - s_prev) becomes the CLOSING rate toward the current
    gate CENTRE -- a dense homing gradient in EVERY axis (the inc7 / Swift distance-to-gate progress).
    Contrast segment_arc_position, which credits only along-track advance (perpendicular drift -> 0).

    ANISOTROPIC VERTICAL WEIGHT (Fengyou greenlight 2026-07-08 -- the floor-dive fix). ``vert_weight`` (=1
    == isotropic Euclidean, the default) up-weights the VERTICAL (Z, gravity-loaded) axis so a same-height
    approach does not BURY the altitude signal under forward progress. On a level dead-ahead gate the drone
    and gate are co-altitude, so the unit-to-gate points purely FORWARD (dz=0) and the vertical restoring
    gradient of the isotropic norm is ~0 -- the 100%-floor-dive root cause. Weighting dz by w makes the
    vertical/forward gradient ratio ~ w*dz/dx (vs dz/dx isotropic), i.e. w x louder: sinking now COSTS
    progress in proportion to w. No hover-farm (it is progress TOWARD the gate; sitting still earns 0) and
    NO ceiling on w -- crank it until the vertical pull beats the gravity+thrust-vectoring down-push. w==1
    reproduces the exact isotropic norm (byte-compatible default for every non-lever stage).

    A true (weighted-Euclidean) potential: telescopes over a closed path -> NON-farmable. ``gate_center`` =
    the CURRENT target gate centre (N,3) -- the same seg_end the segment mode projects onto."""
    assert torch is not None
    if vert_weight == 1.0:
        return -torch.linalg.norm(pos - gate_center, dim=-1)
    d = pos - gate_center
    return -torch.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + vert_weight * d[..., 2] ** 2 + 1e-12)


def segment_progress_reward(s_curr: Tensor, s_prev: Tensor, rw_progress: float,
                            vmax_mps: float, dt: float) -> Tensor:
    """R_prog = rw_progress * clip(s_curr - s_prev, -vmax*dt, +vmax*dt).

    Potential-based (telescoping in s) -> NON-farmable (the sum over a closed loop is ~0; you cannot
    pump reward by oscillating). The clip band ``vmax_mps * dt`` (m/step) trims only UNPHYSICAL bursts
    (e.g. a discontinuous segment re-projection at a gate handoff); at the true peak per-step advance
    (~30 m/s * dt) it is INACTIVE (vmax = 39 m/s = 30 * 1.3 leaves ~30% headroom). Can be negative
    (the drone backed up along the segment)."""
    assert torch is not None
    band = vmax_mps * dt
    delta = (s_curr - s_prev).clamp(min=-band, max=band)
    return rw_progress * delta


def area_distance_progress_factor(area_true: Tensor, dist_to_gate: Tensor,
                                  area_dist_ref_m: float) -> Tensor:
    """The DISTANCE-GATED area coupling multiplier (Fengyou 2026-07-07 "area-distance coupled progress").

        factor = area + (1 - area) * clip(dist / ref, 0, 1)   in [area, 1]

      * FAR (dist >= ref): factor -> 1 -- a beeline approach is NOT penalised for its angle (the drone
        will straighten out on the way in); progress earns full credit.
      * CLOSE (dist -> 0): factor -> area (= |cos(view_ray, gate_normal)| in [0,1], 1 == perfectly
        square-on) -- a shallow / off-axis approach in the final metres earns LESS, because that geometry
        threads the aperture at an angle and exits WIDE. The policy learns to square up before the gate.

    ``area_true`` (N,) in [0,1] is the PRIVILEGED GT foreshortening (the same |cos| the estimator's
    visible_area channel exposes, but noiseless); ``dist_to_gate`` (N,) is the GT metres to the current
    target gate centre. Both are translation- and yaw-invariant (a normalised difference / a norm), so
    the coupled reward stays GT-only / position-free-safe. Returns (N,)."""
    assert torch is not None
    ramp = (dist_to_gate / max(area_dist_ref_m, 1e-9)).clamp(0.0, 1.0)
    return area_true + (1.0 - area_true) * ramp


def segment_perp_distance(pos: Tensor, seg_start: Tensor, seg_end: Tensor) -> Tensor:
    """Perpendicular distance (N,) from ``pos`` to the FINITE segment [seg_start -> seg_end] -- the lateral
    offset from the current gate-centre line. Reuses the segment projection: closest = seg_start + s*u
    (s clamped to [0, seg_len]), perp = ||pos - closest||. A degenerate zero-length segment -> distance to
    seg_start. (N,3) inputs, Z-up GT."""
    assert torch is not None
    d = seg_end - seg_start                                   # (N,3)
    seg_len = torch.linalg.norm(d, dim=-1)                    # (N,)
    safe = seg_len.clamp(min=1e-9)
    u = d / safe.unsqueeze(-1)                                # unit along-track
    s = ((pos - seg_start) * u).sum(dim=-1)                   # signed projection
    s = s.clamp(min=torch.zeros_like(seg_len), max=seg_len)   # clamp to the FINITE segment
    closest = seg_start + s.unsqueeze(-1) * u                 # nearest point on the segment
    return torch.linalg.norm(pos - closest, dim=-1)           # (N,) lateral offset


def through_centering_reward(perp_dist: Tensor, rw_centering: float, centering_max_m: float) -> Tensor:
    """R_center = -rw_centering * clamp(perp_dist, 0, centering_max_m). A DENSE per-step pull onto the
    gate-centre segment (Fengyou 2026-07-07): off-line flight is penalised (clamped so a big early drift
    does not dominate), on-line flight (perp~0) pays ~0 -> non-farmable, vanishes at convergence. Shapes
    the lateral POSITION toward a CENTRED crossing (complement to the area coupling's ANGLE shaping).
    rw_centering==0 -> OFF. Returns the (negative) penalty (N,)."""
    assert torch is not None
    if rw_centering == 0.0:
        return torch.zeros_like(perp_dist)
    return -rw_centering * perp_dist.clamp(0.0, centering_max_m)


def altitude_hold_reward(z: Tensor, z_spawn: Tensor, rw_altitude_hold: float,
                         band_m: float) -> Tensor:
    """R_alt = rw_altitude_hold * (1 - clip(|z - z_spawn| / band, 0, 1)). A POSITIVE, bounded per-step
    bonus peaking (+rw_altitude_hold) at the spawn altitude and decaying linearly to 0 at |Δz| >= band.
    Used ONLY by the `hover_hold` diagnostic probe stage (Fengyou 2026-07-08): with NO gate homing/passage
    it makes holding spawn altitude the UNIQUE optimum, and because it PAYS THE POLICY TO SURVIVE (positive
    reward -> ending the episode forfeits the future bonus) it carries NO give-up incentive -- distinct from
    a -k|Δz| MAGNITUDE penalty (the rw_centering back-fire class) that can drive a failing policy to floor
    sooner to stop the bleed and CONFOUND the H1-vs-H2 read. rw_altitude_hold==0 -> OFF (zeros). z / z_spawn
    are (N,) GT world Z-up altitudes. Returns (N,)."""
    assert torch is not None
    if rw_altitude_hold == 0.0:
        return torch.zeros_like(z)
    err = (z - z_spawn).abs() / max(band_m, 1e-9)
    return rw_altitude_hold * (1.0 - err.clamp(0.0, 1.0))


def gate_vertical_hold_reward(drone_z: Tensor, gate_center_z: Tensor, rw_gate_vhold: float,
                              band_m: float) -> Tensor:
    """R_gvhold = rw_gate_vhold * (1 - clip(|drone_z - gate_center_z| / band, 0, 1)). The GATE-RELATIVE
    WORLD-VERTICAL hold (Fengyou 2026-07-12; the R0 floor-dive fix) -- the OBSERVABLE replacement for
    altitude_hold_reward. IDENTICAL give-up-resistant POSITIVE form, but the reference is the CURRENT target
    gate's world-Z (``gate_center_z``) instead of the spawn altitude, so the rewarded quantity is the world-
    vertical gate offset Δz = gate_center_z - drone_z, which a sufficient policy CAN reconstruct from its obs
    (Δz = R_wb[2,:] · rel_pos_body; the third row of R_wb is YAW-INVARIANT, and roll_pitch + the body-frame
    slot0 rel_pos are BOTH in the 21-dim ego obs). Peaks (+rw_gate_vhold) with the drone at the gate's
    altitude and decays linearly to 0 at |Δz| >= band. POSITIVE + bounded -> ending the episode FORFEITS the
    future bonus == paid to survive at the gate's height (give-up-RESISTANT, unlike a -k|Δz| magnitude
    penalty, which is minimized by TERMINATING sooner and would REINTRODUCE the floor-dive). RANGE-INDEPENDENT
    (the vertical component only) -> NO homing: zero gradient along range, so it can never pull the drone into
    the gate. Only the DIFFERENCE (drone_z - gate_center_z) enters the reward -> no absolute coordinate is
    used (both are GT world Z-up (N,) altitudes; GT is legal in the reward, and the difference is observable).
    rw_gate_vhold==0 -> OFF (zeros -> byte-identical). Returns (N,)."""
    assert torch is not None
    if rw_gate_vhold == 0.0:
        return torch.zeros_like(drone_z)
    err = (drone_z - gate_center_z).abs() / max(band_m, 1e-9)
    return rw_gate_vhold * (1.0 - err.clamp(0.0, 1.0))


def corridor_progress_reward(perp_curr: Tensor, perp_prev: Tensor, rw_corridor: float,
                             vmax_mps: float, dt: float) -> Tensor:
    """R_corr = rw_corridor * clip(perp_prev - perp_curr, -band, +band). The MPCC CONTOURING term: a
    telescoping potential (phi_corr = -perp) differenced at gamma=1, so it is POSITIVE when the drone moves
    TOWARD the current gate-centre segment (perp shrinks) and negative when it drifts off. NON-farmable
    (the episode sum telescopes to perp_0 - perp_T, a boundary term -> no accrual to pump, no give-up
    escape) and it levies NO standing tax on a centred mean (E[Δ]~0), unlike the raw through_centering
    magnitude penalty. The band = vmax_mps*dt (m/step) trims only the discontinuous perp re-projection at a
    gate handoff (same rationale as segment_progress_reward); at real flight speed it is INACTIVE. Pairs
    with the ALONG-TRACK LAG progress (progress_to_center=False) to supply the vertical+lateral homing the
    lag omits. rw_corridor==0 -> OFF (zeros). Returns (N,)."""
    assert torch is not None
    if rw_corridor == 0.0:
        return torch.zeros_like(perp_curr)
    band = vmax_mps * dt
    return rw_corridor * (perp_prev - perp_curr).clamp(min=-band, max=band)


def alignment_reward(vel_world: Tensor, tangent: Tensor, inward_unit: Tensor, perp: Tensor,
                     rw_align: float, align_gain: float) -> Tensor:
    """GVF DIRECTION-ALIGNMENT reward (Fengyou 2026-07-08): r = rw_align * dot(v_hat, F_hat), the cosine
    alignment of the drone's velocity DIRECTION with the guiding vector field F. F angles from the line
    tangent toward the line by ``theta = atan(align_gain * perp)``:

        F = cos(theta) * tangent + sin(theta) * inward_unit          (already unit: tangent ⟂ inward_unit)

      * ON the line (perp=0 -> theta=0): F = tangent -> reward following the path down-course.
      * OFF the line: F angles INWARD; a drone flying PARALLEL (v ⟂ inward) is MIS-aligned (dot < 1), so it
        is pressured to turn its velocity toward the path -- the standing-offset blind spot the telescoping
        contouring has (dot(v, F_cross)=0 for parallel motion) is GONE, because this scores DIRECTION not
        displacement.

    ``align_gain`` is the CONVERGENCE TIGHTNESS (Fengyou's "how sharp a turn onto the path"): high gain ->
    theta reaches ~90deg close to the line -> a SHARP corner; low gain -> a gentle, wide, asymptotic curve.
    Speed-blind (unit vectors, dot in [-1,1]); pair with the progress term for the speed/racing incentive.
    A near-stationary drone (|v|~0) has an ill-defined direction -> its alignment is damped toward 0 by the
    velocity-norm guard, so it earns ~0 (neither rewarded nor punished) rather than a spurious value.
    rw_align==0 -> OFF (zeros). vel_world/tangent/inward_unit (N,3) world Z-up; perp (N,). Returns (N,)."""
    assert torch is not None
    if rw_align == 0.0:
        return torch.zeros(vel_world.shape[0], device=vel_world.device, dtype=vel_world.dtype)
    theta = torch.atan(align_gain * perp)                              # (N,) inward angle
    F = (torch.cos(theta).unsqueeze(-1) * tangent
         + torch.sin(theta).unsqueeze(-1) * inward_unit)               # (N,3) unit guiding field
    v_norm = torch.linalg.norm(vel_world, dim=-1, keepdim=True)
    v_hat = vel_world / v_norm.clamp(min=1e-6)                         # (N,3); ~0 when stationary
    align = (v_hat * F).sum(dim=-1)                                    # (N,) dot(v_hat, F_hat) in [-1,1]
    # damp the alignment for a near-stationary drone (ill-defined direction) so it earns ~0, not a spurious
    # value: scale by v/(v+eps_speed) which -> 1 at speed, -> 0 at rest.
    speed_gate = v_norm.squeeze(-1) / (v_norm.squeeze(-1) + 0.2)
    return rw_align * align * speed_gate


def crossing_parabola_reward(cross_offset: Tensor, crossed: Tensor, cross_center: float,
                             cross_zero_m: float, cross_neg_cap: float,
                             latch_once: bool = False,
                             paid_latch: "Tensor | None" = None) -> Tensor:
    """Smooth PARABOLIC crossing reward (Fengyou 2026-07-08): fires ONCE on a forward gate-plane crossing.
        r = clamp(cross_center * (1 - (e/R)^2), -cross_neg_cap, cross_center)
    where e = ``cross_offset`` (L-inf in-plane offset at the crossing) and R = ``cross_zero_m`` (the aperture
    half-opening). +cross_center dead-centre -> 0 at the aperture edge -> a downward parabola growing NEGATIVE
    outside, floored at -cross_neg_cap. SMOOTH + MONOTONIC in the offset: no thread/clip/miss cliff and NO
    MOAT (getting closer is ALWAYS better). REPLACES the passage reward + the frame-clip/miss terminal
    penalties (the env drops those when parabola_crossing is on; floor + oob keep theirs). ``crossed`` (N,)
    bool = the forward target-plane crossing this step. cross_center==0 -> OFF (zeros). Returns (N,).

    ONCE-PER-GATE LATCH (audit red-flag; ``latch_once`` + ``paid_latch``): ``crossed``==fwd_t is NOT idempotent
    per gate the way the passage's gate_passed is -- with miss_terminates=false a wide forward crossing neither
    advances the target nor terminates, so an oscillating drone re-crosses the SAME target plane and the parabola
    is paid every step (a re-payment FARM). When ``latch_once`` and a per-env bool ``paid_latch`` (N,) are given,
    a gate already marked in ``paid_latch`` pays 0 and the latch is set IN PLACE for every env that crossed (so
    the caller retains the state); the caller clears ``paid_latch`` on a target ADVANCE and on episode RESET.
    latch_once=False (default) -> byte-identical (no latch, no mutation)."""
    assert torch is not None
    if cross_center == 0.0:
        return torch.zeros_like(cross_offset)
    R = max(cross_zero_m, 1e-6)
    para = cross_center * (1.0 - (cross_offset / R) ** 2)
    para = para.clamp(min=-cross_neg_cap, max=cross_center)
    fire = crossed.to(torch.bool)
    if latch_once and paid_latch is not None:
        newly = fire & ~paid_latch                # suppress a gate already paid this episode
        paid_latch |= fire                        # in-place mark so the caller keeps the latch
        fire = newly
    return para * fire.to(cross_offset.dtype)


# ================================================================================================
# R_pass: PASSAGE + L-INF centering (matches crossing_events' Linf), idempotent per gate.
# ================================================================================================
def centering_passage_reward(gate_passed: Tensor, pass_linf: Tensor, w_g_half: float,
                             rw_passage: float, passed_gate_index: Tensor | None = None,
                             passage_increment: float = 0.0) -> Tensor:
    """R_pass = (rw_passage + passage_increment * gate_index) * (1 - e_lat / w_g_half) on a VALID fwd
    pass, else 0.

    e_lat == ``pass_linf`` == the L-INF (max(|y|,|z|)) in-plane offset at the interpolated crossing
    point (the SAME quantity crossing_events reports and the SAME metric pass_ok thresholds against
    w_g_half). A dead-centre pass (e_lat=0) pays the full weight; a pass at the aperture edge
    (e_lat=w_g_half) pays ~0 -> the anti-corner-cut / centering signal. Clamped >= 0 so an edge pass
    never pays NEGATIVE (a valid pass is always non-negative; the miss/contact terminals carry the
    penalty side).

    PER-GATE INCREMENT (Fengyou 2026-07-07): when ``passed_gate_index`` (N,) is supplied, the weight for
    each env is ``rw_passage + passage_increment * passed_gate_index`` so a LATER gate pays more (gate 0
    -> base, gate 1 -> base+inc, ...). ``passed_gate_index`` is the CURRENT-target index at the crossing
    (before the target advances). Omit it (or passage_increment=0) for a flat passage.

    IDEMPOTENCY is enforced by the CALLER: ``gate_passed`` is true for exactly the ONE step the target
    gate is crossed forward AND the target index strictly increments, so a weaving re-crossing of an
    already-passed gate is NOT the current target -> gate_passed is False -> pays nothing again."""
    assert torch is not None
    centered = (1.0 - pass_linf / max(w_g_half, 1e-9)).clamp(min=0.0)
    weight = rw_passage
    if passed_gate_index is not None and passage_increment != 0.0:
        weight = rw_passage + passage_increment * passed_gate_index.to(pass_linf.dtype)
    return weight * centered * gate_passed.to(pass_linf.dtype)


# ================================================================================================
# TERMINAL: kill-on-contact / miss / oob, tuned to DOMINATE the banked progress return.
# ================================================================================================
def terminal_penalty(gate_collision: Tensor, gate_miss: Tensor, oob: Tensor,
                     banked_progress_return: Tensor, w: EgoRewardWeights,
                     forfeit_mask: "Tensor | None" = None) -> Tensor:
    """The hard terminal penalty (subtracted from the reward on the terminating step).

    Two modes (``w.terminal_progress_scaled``):
      * PROGRESS-SCALED (recommended): penalty = base + max(banked_progress_return, 0) ON A CONTACT ONLY.
        Clipping a gate (a frame contact) FORFEITS all banked progress return PLUS a base, so a
        sprint-and-clip can NEVER out-earn continuing -- the defect the critics caught (10.0/m made
        clipping positive-return) is closed by construction regardless of course length. A wide MISS or
        an OOB is an HONEST non-contact outcome: it pays ONLY its fixed base and KEEPS its banked
        approach progress (forfeiting there was the 2026-07-07 single_gate root cause -- it cancelled the
        dense homing gradient and made loitering beat committing). ``banked_progress_return`` = the
        (undiscounted) progress reward accumulated so far this episode (the env tracks it).
      * FIXED: penalty = base (a large fixed magnitude). Requires base > max bankable progress return;
        the rollout check picks base.

    Applied to ANY of contact / miss / oob (each terminal). Returns a NON-NEGATIVE magnitude (N,) to
    SUBTRACT; the caller does reward - terminal_penalty(...). Contact uses ``terminal_base`` (+ forfeit),
    miss uses ``terminal_miss``, oob uses ``terminal_oob`` (default bases all equal)."""
    assert torch is not None
    dt = banked_progress_return.dtype
    coll = gate_collision.to(dt)
    miss = gate_miss.to(dt)
    ob = oob.to(dt)
    # per-terminal base magnitude: the LARGEST single base that fired (a single terminal event pays ONE
    # penalty; if several fire on a step we take the max, never the sum).
    base = torch.maximum(torch.maximum(w.terminal_base * coll, w.terminal_miss * miss),
                         w.terminal_oob * ob)
    fired = (coll + miss + ob) > 0
    if w.terminal_progress_scaled:
        # FORFEIT the banked progress ONLY on a CONTACT (reward-audit 2026-07-07 -- THE single_gate
        # root cause). ``banked`` is the undiscounted sum of the SAME r_prog already streamed into the
        # per-step return, so forfeiting it EXACTLY CANCELS that dense progress. Doing so on a wide MISS
        # or an OOB -- honest NON-contact outcomes -- cancelled the homing gradient on ~100% of rollouts
        # AND made "approach then loiter/miss" out-earn committing to a crossing (an unescapable
        # exploration trap). The sprint-and-clip defence is fully preserved: a frame clip is classified
        # as gate_collision (env: in_frame -> contact), so clipping STILL forfeits. Miss/oob keep their
        # banked approach progress and pay ONLY their fixed base.
        #
        # FORFEIT_MASK (Fengyou 2026-07-08, the frame-moat fix): normally the forfeit fires on EVERY contact
        # (mask == coll). But a FRAME-CLIP costing MORE than a WIDE-MISS makes the ring around the aperture a
        # MOAT -- getting closer (wide -> frame band) is punished more, so the CENTRE is not attractive and
        # the policy parks wide (the 4.4m plateau). ``forfeit_mask`` = FLOOR-contact-only makes a frame-clip
        # forfeit NOTHING (net == a wide miss: both -base, both keep banked) while the floor-dive (a real
        # crash) still forfeits -> the centre is the strictly-best crossing, no moat.
        fmask = forfeit_mask.to(dt) if forfeit_mask is not None else coll
        forfeit = banked_progress_return.clamp(min=0.0) * fmask
        return (base + forfeit) * fired.to(dt)
    return base * fired.to(dt)


# ================================================================================================
# WIDE-FLYBY -> MISS: force-classify a lateral flyby past the along-track plane as a miss terminal.
# ================================================================================================
def wide_flyby_miss(fwd: Tensor, bwd: Tensor, pass_ok: Tensor, in_frame: Tensor) -> Tensor:
    """A lateral flyby that CROSSES the target gate's along-track plane (fwd or bwd) without a VALID
    forward pass and without a frame strike (in_frame) is a MISS -- it must terminate as T2, never be
    silently ignored. Returns bool (N,).

    A crossing is classified into exactly one of {pass_ok (valid pass), in_frame (frame strike ->
    contact), miss (the wide/off-aperture flyby)}. The env's inc7 classification already computes
    ``gate_miss = fwd & ~pass_ok & ~in_frame`` for the FORWARD case; this ALSO catches a BACKWARD wide
    crossing of the target plane (a drone that overshoots laterally and re-crosses the plane going the
    wrong way past the aperture) so no lateral flyby escapes classification."""
    assert torch is not None
    crossed = fwd | bwd
    return crossed & (~pass_ok) & (~in_frame)


# ================================================================================================
# T4 finish-time (KEEP).
# ================================================================================================
def finish_reward(newly_finished: Tensor, time_left_s: Tensor, w: EgoRewardWeights) -> Tensor:
    """T4 = (rw_finish + rw_finish_time * t_left_s) * 1[finished]. Rewards finishing AND finishing
    fast (more time left on the clock -> more reward). Fires once, on the last-gate pass step."""
    assert torch is not None
    return (w.finish + w.finish_time * time_left_s) * newly_finished.to(time_left_s.dtype)


# ================================================================================================
# R4 confidence-gated free-cone (FIXED relaxed cone, zero inside; 60 deg, CAP 70).
# ================================================================================================
def free_cone_penalty(tilt_cos_r33: Tensor, w: EgoRewardWeights) -> Tensor:
    """R4 = -rw_tilt * relu(cos(tilt_free) - R33)^2 -- a FIXED relaxed cone (60 deg), ZERO inside the
    cone (nothing to chatter against when upright), quadratic outside. ``tilt_cos_r33`` = R33 =
    cos(total tilt) = world-z . body-z. The free half-angle is capped at 70 deg (``tilt_free_cap_rad``)
    so a mis-set cfg can never relax it past the safety cap. Returns the (negative) penalty (N,)."""
    assert torch is not None
    import math as _m
    free = min(w.tilt_free_rad, w.tilt_free_cap_rad)
    cos_free = _m.cos(free)
    pen = torch.relu(cos_free - tilt_cos_r33) ** 2
    return -w.tilt * pen


# ================================================================================================
# Smoothness (MINUSCULE): R6 rate + R5 dact + optional R7 joint corner tax.
# ================================================================================================
def smoothness_penalty(omega: Tensor, action_norm: Tensor, last_action_norm: Tensor,
                       w: EgoRewardWeights) -> Tensor:
    """The combined MINUSCULE smoothness penalty (Geles: 3-4 orders below progress). Returns the
    (negative) penalty (N,).
      R6 rate  = -rw_rate * ||omega||           (rad/s)
      R5 dact  = -rw_dact * ||Delta action||^2  (span-normalised)
      R7 corner= -rw_corner * |thr-0.5| * ||2(rate-0.5)||   (joint corner tax; PREFERRED; 0 == off)
    At the SLOW bring-up speed this MUST be <1% of a representative per-step progress reward -- pinned
    by tests/test_ego_reward.py::test_smoothness_below_one_percent."""
    assert torch is not None
    rate_mag = torch.linalg.norm(omega, dim=-1)
    dact = ((action_norm - last_action_norm) ** 2).sum(dim=-1)
    corner = (action_norm[..., 0] - 0.5).abs() * torch.linalg.norm(
        2.0 * (action_norm[..., 1:4] - 0.5), dim=-1)
    return -(w.rate * rate_mag + w.dact * dact + w.corner * corner)


def yaw_dither_penalty(yaw_cmd_delta: Tensor, rw_yaw_dither: float) -> Tensor:
    """ANTI-DITHER yaw SMOOTHNESS penalty (nodither fine-tune 2026-07-12): R_yawdith = -rw_yaw_dither *
    yaw_cmd_delta^2, where ``yaw_cmd_delta`` = yaw_cmd_t - yaw_cmd_{t-1} is the temporal change of the
    APPLIED (post-clamp) yaw-rate command (action channel 3, rad/s). Penalises OSCILLATION of the yaw
    command (a +-clamp rail-flip -> large |delta| -> heavy penalty) while a SUSTAINED / steady yaw command
    (a needed turn -> delta ~0) pays ~0 -- so a smooth turn / speed is NEVER penalised and only the dither
    transient pays. Squared (SMOOTH + magnitude-aware): near-zero jitter pays ~0, a rail-flip pays the most
    -- the dither discrimination a sign-flip indicator LACKS (the indicator is discontinuous AND over-taxes
    benign near-zero jitter around zero-yaw straight flight). NON-farmable: a steady command earns 0, so it
    vanishes at convergence (min 0 at steady yaw). The constant-DRIFT escape (a slow steady spin, 0 jerk) is
    NOT closed here -- the retained fatal spin abort (ego_spin_rev_*) closes it BY CONSTRUCTION. Sign:
    NEGATIVE (a penalty). rw_yaw_dither==0 -> OFF (zeros -> byte-identical). Returns the (<=0) penalty (N,)."""
    assert torch is not None
    if rw_yaw_dither == 0.0:
        return torch.zeros_like(yaw_cmd_delta)
    return -rw_yaw_dither * yaw_cmd_delta ** 2


def velocity_jerk_penalty(accel_curr: Tensor, accel_prev: Tensor, rw_vel_smooth: float) -> Tensor:
    """VELOCITY-JERK smoothness prior (R0 still-yaw hover boot 2026-07-12): R_velsmooth = -rw_vel_smooth *
    ||jerk||^2, where jerk = ``accel_curr`` - ``accel_prev`` (the 1st difference of acceleration == the 2nd
    time-difference of velocity). ``accel_curr``/``accel_prev`` are the CURRENT and PREVIOUS-step WORLD-frame
    CoM accelerations (N,3), accel = (v_t - v_{t-1})/dt (the env threads the prev velocity + acceleration).

    Penalises a snappy JERK SPIKE, NOT velocity or acceleration magnitude: a STEADY SPEED (accel 0 -> jerk 0)
    AND a SMOOTH HARD acceleration (accel constant -> jerk 0) BOTH pay ~0 -- only a SUDDEN change of
    acceleration bites, so this is NOT a speed/energy penalty (fast, hard, smooth flight is free). ORTHOGONAL
    to the yaw/roll mechanism BY CONSTRUCTION: a yaw (or roll) flip barely moves the centre of mass, so the
    WORLD-velocity jerk is ~0 -> ~0 penalty (this is WHY it is the world CoM jerk and NOT the body specific
    force, whose ~1 g gravity term rotates with attitude and would spuriously tax roll/pitch). Squared
    (smooth + magnitude-aware) so benign near-zero jitter pays ~0 and it vanishes at convergence (steady
    accel -> jerk 0 -> 0, non-farmable). Sign NEGATIVE. rw_vel_smooth==0 -> zeros (byte-identical). (N,)."""
    assert torch is not None
    if rw_vel_smooth == 0.0:
        return torch.zeros(accel_curr.shape[0], device=accel_curr.device, dtype=accel_curr.dtype)
    jerk = accel_curr - accel_prev
    return -rw_vel_smooth * (jerk ** 2).sum(dim=-1)


def velocity_cap_penalty(speed: Tensor, rw_v_cap: float, v_cap_soft: float,
                         v_cap_hard: float) -> Tensor:
    """ONE-SIDED SOFT-HINGE velocity-cap penalty (anti-velocity-runaway 2026-07-13):
        R_vcap = -rw_v_cap * (relu(speed - v_cap_soft) / (v_cap_hard - v_cap_soft))^2
    on TOTAL SPEED ``speed`` = ||v|| (m/s). EXACTLY ZERO at/below ``v_cap_soft`` (relu -> 0, so the reward is
    UNCHANGED across the whole useful speed band -- no speed term, no reward lost elsewhere), then ramps up
    QUADRATICALLY: at speed == ``v_cap_hard`` the normalised excess is 1 so the penalty is -rw_v_cap
    (rw_v_cap == the per-step penalty magnitude AT the hard cap), and it grows > rw_v_cap super-linearly
    beyond it (prohibitive in the runaway tail). C^1 at the soft knee (value AND slope 0 -> smooth, no
    cliff), MONOTONE increasing in magnitude above it.

    OBSERVABLE-STATE ONLY (owner directive): ||v|| is reconstructable from the actor obs (body velocity at
    obs[0:3]; ||v_body|| == ||v_world|| since a rotation preserves the norm), so the cap prices only what the
    policy sees. NOT an energy penalty (0 below the soft cap). ``v_cap_hard`` > ``v_cap_soft`` is enforced by
    EgoRewardWeights.__post_init__; the span is clamped to 1e-6 here as a defensive backstop. rw_v_cap==0 ->
    OFF (zeros -> byte-identical). Returns the (<=0) penalty (N,)."""
    assert torch is not None
    if rw_v_cap == 0.0:
        return torch.zeros_like(speed)
    span = max(float(v_cap_hard) - float(v_cap_soft), 1e-6)
    excess = torch.relu(speed - float(v_cap_soft)) / span
    return -rw_v_cap * excess ** 2


# ================================================================================================
# R_exit: NEXT-GATE exit-line anticipation (dual_gate+ stage; small; on by curriculum).
# ================================================================================================
def exit_line_reward(vel_world: Tensor, curr_center: Tensor, next_center: Tensor,
                     gate_passed: Tensor, w: EgoRewardWeights) -> Tensor:
    """R_exit = rw_exit * cos(angle between the drone's EXIT velocity and the current->next gate-centre
    bearing), fired ONLY on the gate-pass step. Rewards LEAVING gate-k on a heading that SETS UP gate-k+1
    (the 2-gate slider otherwise shapes only the current approach, so a policy can thread gate-k on a
    heading that dooms gate-k+1). Small weight; ON only by curriculum stage (``exit_align`` > 0 on
    dual_gate+).

    GATED ON ``gate_passed`` (fires ONCE per gate, at the crossing) -- NOT a dense per-step term. This is
    the critic-mandated fix: a dense every-step version is (a) FARMABLE (hold velocity toward the next
    bearing for the whole approach) and (b) a CORNER-CUT bias (pulls the whole approach heading toward
    the next gate). Firing only at the exit step shapes the exit heading without either pathology, and is
    naturally bounded (once per gate).

    vel_world (N,3) GT velocity; curr_center/next_center (N,3) GT gate centres (Z-up); gate_passed (N,)
    bool = true only on the pass step. The bearing is curr->next (a GT-centre DIFFERENCE -> only a
    direction, no world coordinate). Zero at the last gate / near-stationary / non-pass steps."""
    assert torch is not None
    if w.exit_align == 0.0:
        return torch.zeros(vel_world.shape[0], device=vel_world.device, dtype=vel_world.dtype)
    bearing = next_center - curr_center
    b_norm = torch.linalg.norm(bearing, dim=-1)
    v_norm = torch.linalg.norm(vel_world, dim=-1)
    denom = (b_norm * v_norm).clamp(min=1e-6)
    cos_align = (bearing * vel_world).sum(dim=-1) / denom
    # fire ONLY at the pass step, and only with a distinct next gate + a moving drone
    valid = (b_norm > 1e-6) & (v_norm > 1e-6) & gate_passed.to(torch.bool)
    return w.exit_align * cos_align * valid.to(vel_world.dtype)


# ================================================================================================
# The full step reward assembly (pure; the env calls this under no_grad).
# ================================================================================================
def perception_reward(cos_view: Tensor, rw_perception: float, exponent: float = 4.0) -> Tensor:
    """Swift/Geles PERCEPTION reward: r = rw_perception · exp(−δ_cam^exponent), δ_cam = angle between the
    camera optical axis and the drone→gate-centre vector (rad). ``cos_view`` = cos δ_cam in [−1,1] (from
    gate_visibility.gate_center_view_cos). Peaks at +rw_perception with the gate centre on the optical
    axis (δ=0) and decays smoothly as the gate drifts to the frame edge -> the policy is paid, every
    step, to keep the gate centred in view (better approach estimate + attention at the gate; the field's
    ~0.5 m -> ~0.15 m gate-passing-error lever). Dense + positive; cannot be farmed off-gate (it points
    AT the gate). 0 == OFF (byte-identical)."""
    assert torch is not None
    if rw_perception == 0.0:
        return torch.zeros_like(cos_view)
    delta = torch.acos(cos_view.clamp(-1.0, 1.0))
    return rw_perception * torch.exp(-(delta ** exponent))


def attitude_limit_penalty(roll: Tensor, pitch: Tensor, w: EgoRewardWeights) -> Tensor:
    """SOFT, NON-TERMINAL attitude-limit penalty (pefcap 2026-07-12; PERCEPTION-preservation, not energy):
        R_att = −rw_att_pitch·relu(|pitch| − att_pitch_limit_rad) − rw_att_roll·relu(|roll| − att_roll_limit_rad)
    on the TRUE gravity-leveled body attitude (GT is legal in reward). ZERO inside the free band -- and the
    band sits ABOVE the airframe's ~17.8deg (0.31 rad) nose-down REST tilt (default pitch limit 60deg), so a
    hovering/resting drone pays ZERO here (this term never makes hovering better than flying-within-band) --
    then grows LINEARLY once the excursion passes the limit. It caps only the OVER-AGGRESSIVE head-down /
    high-bank attitudes that swing the +20deg-mounted camera off the gate and break the estimate; it is NOT
    an |omega|/thrust/jerk penalty and does NOT penalise speed (a fast drone within the band pays 0). It is
    a pure reward term -- it NEVER terminates a flight. rw_att_pitch==rw_att_roll==0 -> zeros (byte-identical).
    roll/pitch (N,) leveled rad. Returns the (negative) penalty (N,)."""
    assert torch is not None
    pen = torch.zeros_like(pitch)
    if w.att_pitch != 0.0:
        pen = pen + w.att_pitch * torch.relu(pitch.abs() - w.att_pitch_limit_rad)
    if w.att_roll != 0.0:
        pen = pen + w.att_roll * torch.relu(roll.abs() - w.att_roll_limit_rad)
    return -pen


def compute_ego_reward(
    w: EgoRewardWeights, *,
    s_curr: Tensor, s_prev: Tensor,
    gate_passed: Tensor, pass_linf: Tensor, w_g_half: float,
    gate_collision: Tensor, gate_miss: Tensor, oob: Tensor,
    banked_progress_return: Tensor,
    newly_finished: Tensor, time_left_s: Tensor,
    tilt_cos_r33: Tensor, omega: Tensor, action_norm: Tensor, last_action_norm: Tensor,
    vel_world: Tensor, curr_center: Tensor, next_center: Tensor,
    dt: float,
    area_true: "Tensor | None" = None,
    dist_to_gate: "Tensor | None" = None,
    passed_gate_index: "Tensor | None" = None,
    perp_dist: "Tensor | None" = None,
    z: "Tensor | None" = None,
    z_spawn: "Tensor | None" = None,
    gate_center_z: "Tensor | None" = None,
    perp_prev: "Tensor | None" = None,
    forfeit_mask: "Tensor | None" = None,
    line_tangent: "Tensor | None" = None,
    line_inward: "Tensor | None" = None,
    cross_offset: "Tensor | None" = None,
    crossed: "Tensor | None" = None,
    parabola_paid: "Tensor | None" = None,
    floor_contact: "Tensor | None" = None,
    cos_view: "Tensor | None" = None,
    cos_view_next: "Tensor | None" = None,
    roll: "Tensor | None" = None,
    pitch: "Tensor | None" = None,
    yaw_cmd_delta: "Tensor | None" = None,
    accel_curr: "Tensor | None" = None,
    accel_prev: "Tensor | None" = None,
):
    """Assemble the refined-B step reward from GT event/state tensors (all (N,) or (N,k)). Returns
    (reward (N,), components dict, r_prog (N,)) -- r_prog is returned so the env can ACCUMULATE the
    banked progress return for the progress-scaled terminal (r_prog is the AREA-DISTANCE-scaled progress
    actually credited, so the banked-forfeit reflects what was earned).

    The terminal is applied LAST and SUBTRACTS a magnitude that dominates the banked progress; on a
    terminating step the shaping terms still apply but the terminal swamps them (verified by the
    rollout check). All quantities are PRIVILEGED GT (never the position-free obs).

    ``area_true``/``dist_to_gate`` (N,) enable the DISTANCE-GATED area coupling of the POSITIVE progress
    (Fengyou 2026-07-07): far -> full credit, close + off-axis -> reduced (see area_distance_progress_
    factor). Backward progress is never scaled (you always pay full for retreating). Omit them (or set
    w.area_dist_ref_m<=0) to disable. ``passed_gate_index`` (N,) enables the per-gate passage increment."""
    assert torch is not None
    r_prog = segment_progress_reward(s_curr, s_prev, w.progress, w.vmax_mps, dt)
    # DISTANCE-GATED area coupling: scale ONLY the positive (forward) progress -- a shallow close-in
    # approach earns less; a beeline from far earns full; retreat always pays full (never discounted).
    area_factor = None
    if area_true is not None and dist_to_gate is not None and w.area_dist_ref_m > 0.0:
        area_factor = area_distance_progress_factor(area_true, dist_to_gate, w.area_dist_ref_m)
        r_prog = torch.where(r_prog > 0, r_prog * area_factor, r_prog)
    # SMOOTH PARABOLIC CROSSING (Fengyou 2026-07-08): when ON it REPLACES the passage reward + drops the
    # frame-clip/miss terminal penalties (floor+oob keep theirs, below). One smooth downward parabola of the
    # crossing offset -> no cliff, no moat.
    parabola_on = (w.parabola_crossing and cross_offset is not None and crossed is not None)
    r_cross = (crossing_parabola_reward(cross_offset, crossed, w.cross_center, w.cross_zero_m, w.cross_neg_cap,
                                        latch_once=w.parabola_latch_once, paid_latch=parabola_paid)
               if parabola_on else torch.zeros_like(r_prog))
    r_pass = (torch.zeros_like(r_prog) if parabola_on else
              centering_passage_reward(gate_passed, pass_linf, w_g_half, w.passage,
                                       passed_gate_index=passed_gate_index,
                                       passage_increment=w.passage_increment))
    # DENSE lateral centering: pull onto the current gate-centre segment (off-line -> penalised, on-line
    # -> ~0). Complements the area coupling (angle); this shapes lateral POSITION toward a centred cross.
    r_center = (through_centering_reward(perp_dist, w.centering, w.centering_max_m)
                if perp_dist is not None else torch.zeros_like(r_prog))
    # HOVER-HOLD probe bonus (OFF unless w.altitude_hold>0, i.e. only the hover_hold diagnostic stage): a
    # give-up-resistant positive pull to the spawn altitude with no competing forward objective.
    r_alt = (altitude_hold_reward(z, z_spawn, w.altitude_hold, w.altitude_hold_band_m)
             if (z is not None and z_spawn is not None) else torch.zeros_like(r_prog))
    # GATE-RELATIVE WORLD-VERTICAL hold (Fengyou 2026-07-12; OFF unless w.gate_vhold>0): the OBSERVABLE
    # replacement for altitude_hold -- a give-up-resistant positive bonus on the world-vertical offset to the
    # CURRENT target gate (Δz = gate_center_z - drone_z), reconstructable from the obs (yaw-invariant
    # R_wb[2,:]·rel_pos_body). Range-independent -> NO homing. ``z`` is the drone world-Z already passed above.
    r_gvhold = (gate_vertical_hold_reward(z, gate_center_z, w.gate_vhold, w.gate_vhold_band_m)
                if (z is not None and gate_center_z is not None) else torch.zeros_like(r_prog))
    # MPCC CONTOURING (OFF unless w.corridor>0): PBRS potential on the perpendicular offset from the gate-
    # centre segment -- the vertical+lateral homing that pairs with the along-track lag progress. Like
    # r_corr it is NOT banked (only r_prog is), so a contact terminal does not forfeit accumulated contouring.
    r_corr = (corridor_progress_reward(perp_dist, perp_prev, w.corridor, w.corridor_clip_mps, dt)
              if (perp_dist is not None and perp_prev is not None) else torch.zeros_like(r_prog))
    # GVF DIRECTION-ALIGNMENT (Fengyou 2026-07-08; OFF unless w.align>0): reward the velocity DIRECTION
    # following the guiding field (angles onto the line by atan(align_gain*perp)) -- no parallel-flying blind
    # spot, unlike the telescoping contouring. Needs the line tangent + inward-unit (from racing_line.query).
    r_align = (alignment_reward(vel_world, line_tangent, line_inward, perp_dist, w.align, w.align_gain)
               if (line_tangent is not None and line_inward is not None and perp_dist is not None)
               else torch.zeros_like(r_prog))
    # PERCEPTION reward (Fengyou 2026-07-09; OFF unless w.perception>0): Swift/Geles r_perc -- keep the
    # camera axis on the gate centre (cos_view from gate_center_view_cos) -> the field's centering lever.
    r_perc = (perception_reward(cos_view, w.perception, w.perception_exponent)
              if cos_view is not None else torch.zeros_like(r_prog))
    # NEXT-GATE perception (pefcap 2026-07-12; OFF unless w.perception_next>0): the same exp(-δ^exp) bonus on
    # the NEXT gate's optical-axis angle. The caller passes cos_view_next ALREADY GATED on next-gate
    # detectability (cos ≈ -1 -> term ~0 while the next gate is out of view / absent), so it rewards
    # ACQUIRING/centering the upcoming gate only once it is genuinely visible -- it cannot be farmed by
    # pointing at an unseen gate, and current+next is bounded <= rw_time (EgoRewardWeights.__post_init__).
    r_perc_next = (perception_reward(cos_view_next, w.perception_next, w.perception_exponent)
                   if cos_view_next is not None else torch.zeros_like(r_prog))
    # ATTITUDE-LIMIT penalty (pefcap 2026-07-12; OFF unless w.att_pitch/att_roll>0): soft, non-terminal
    # perception-preservation penalty on the leveled attitude beyond the free band (see attitude_limit_penalty).
    r_att = (attitude_limit_penalty(roll, pitch, w)
             if (roll is not None and pitch is not None) else torch.zeros_like(r_prog))
    # ANTI-DITHER yaw smoothness (nodither fine-tune 2026-07-12; OFF unless w.yaw_dither>0 -> byte-identical):
    # penalise the temporal CHANGE of the applied yaw-rate command (a +-clamp rail-flip pays heavy; a steady
    # turn pays ~0). The env passes yaw_cmd_delta = applied_yaw_t - applied_yaw_{t-1} (channel 3, post-clamp)
    # or None. See yaw_dither_penalty: the retained fatal spin abort closes the constant-drift (slow-spin) hole.
    r_yawdith = (yaw_dither_penalty(yaw_cmd_delta, w.yaw_dither)
                 if yaw_cmd_delta is not None else torch.zeros_like(r_prog))
    # VELOCITY-JERK smoothness prior (R0 still-yaw hover boot 2026-07-12; OFF unless w.vel_smooth>0 ->
    # byte-identical): penalise the temporal CHANGE of the world-frame CoM acceleration (a snappy spike pays;
    # steady speed AND smooth hard accel both pay ~0). The env passes accel_curr = (v_t - v_{t-1})/dt +
    # accel_prev (the threaded prev accel), or None. See velocity_jerk_penalty: orthogonal to yaw/roll (a
    # flip barely moves the CoM -> ~0 jerk), NOT a speed/energy penalty.
    r_velsmooth = (velocity_jerk_penalty(accel_curr, accel_prev, w.vel_smooth)
                   if (accel_curr is not None and accel_prev is not None) else torch.zeros_like(r_prog))
    # VELOCITY-CAP soft-hinge (anti-velocity-runaway 2026-07-13; OFF unless w.v_cap>0 -> byte-identical):
    # penalise TOTAL SPEED ||v|| above the soft cap (quadratic, prohibitive near the hard cap; EXACTLY 0
    # at/below the soft cap so the useful speed band is unchanged). ||v|| is OBSERVABLE (body velocity at
    # obs[0:3]; the norm is frame-invariant) -- computed here from vel_world (GT is legal in the reward;
    # observability is about what is PRICED, not what it is computed from). vel_world is always supplied.
    r_vcap = velocity_cap_penalty(torch.linalg.norm(vel_world, dim=-1), w.v_cap, w.v_cap_soft, w.v_cap_hard)
    r_fin = finish_reward(newly_finished, time_left_s, w)
    r_cone = free_cone_penalty(tilt_cos_r33, w)
    r_smooth = smoothness_penalty(omega, action_norm, last_action_norm, w)
    r_exit = exit_line_reward(vel_world, curr_center, next_center, gate_passed, w)
    r_time = -w.time
    if parabola_on and floor_contact is not None:
        # the parabola IS the frame-clip/miss outcome -> the terminal penalty fires on FLOOR + OOB only
        # (floor forfeits banked; oob keeps the strong wall). Frame-clip/miss pay ONLY the parabola.
        term = terminal_penalty(floor_contact, torch.zeros_like(gate_miss), oob,
                                banked_progress_return, w, forfeit_mask=floor_contact)
        # AUDIT-C5 GRADED ANTI-CLIP RESTORE: with the parabola on, a sub-aperture FRAME STRIKE (a DQ) pays
        # only the ~+19 parabola, ~indistinguishable from a clean thread -> no pressure against the 35% clip
        # rate. clip_terminal_w>0 ADDS a flat -clip_terminal_w on the frame strike the parabola DROPPED, i.e.
        # the collision that is NOT the floor dive (floor already pays its terminal above; miss stays a pure
        # parabola outcome). Read fresh from the plain-float attr each step so the train-loop 0->W anneal needs
        # no extra plumbing. w.clip_terminal_w==0 -> byte-identical (no extra penalty).
        if w.clip_terminal_w > 0.0:
            frame_strike = gate_collision.to(torch.bool) & ~floor_contact.to(torch.bool)
            term = term + w.clip_terminal_w * frame_strike.to(term.dtype)
    else:
        term = terminal_penalty(gate_collision, gate_miss, oob, banked_progress_return, w,
                                forfeit_mask=forfeit_mask)

    reward = (r_prog + r_pass + r_cross + r_center + r_alt + r_gvhold + r_corr + r_align + r_perc
              + r_perc_next + r_att + r_yawdith + r_velsmooth + r_vcap + r_fin + r_cone + r_smooth + r_exit
              + r_time - term)

    components = {
        "prog_reward": float(r_prog.mean()),
        "pass_reward": float(r_pass.mean()),
        "cross_parabola_reward": float(r_cross.mean()),
        "center_pen": float((-r_center).mean()),
        "alt_hold_reward": float(r_alt.mean()),
        "gate_vhold_reward": float(r_gvhold.mean()),
        "corridor_reward": float(r_corr.mean()),
        "align_reward": float(r_align.mean()),
        "perception_reward": float(r_perc.mean()),
        "perception_next_reward": float(r_perc_next.mean()),
        "att_pen": float((-r_att).mean()),
        "yaw_dither_pen": float((-r_yawdith).mean()),
        "velsmooth_pen": float((-r_velsmooth).mean()),
        "vcap_pen": float((-r_vcap).mean()),
        "finish_reward": float(r_fin.mean()),
        "cone_pen": float((-r_cone).mean()),
        "smooth_pen": float((-r_smooth).mean()),
        "exit_reward": float(r_exit.mean()),
        "terminal_pen": float(term.mean()),
        "collision_rate": float(gate_collision.float().mean()),
        "miss_rate": float(gate_miss.float().mean()),
        "oob_rate": float(oob.float().mean()),
        "total_reward": float(reward.mean()),
        # mean area-distance coupling multiplier applied to progress this step (1.0 == coupling off /
        # all far-or-square-on); < 1 == some envs are close AND off-axis (being nudged to square up).
        "area_factor": float(area_factor.mean()) if area_factor is not None else 1.0,
    }
    return reward, components, r_prog
