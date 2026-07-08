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
        construction instead of silently training a gate-clipper. progress-scaled -> no-op."""
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


def gate_center_potential(pos: Tensor, gate_center: Tensor) -> Tensor:
    """Progress potential s = -||pos - gate_centre|| (N,), Z-up GT. Fed to ``segment_progress_reward``,
    reward = clip(s_curr - s_prev) becomes the 3D CLOSING rate toward the current gate CENTRE -- a dense
    homing gradient in EVERY axis (the inc7 / Swift distance-to-gate progress). Contrast
    segment_arc_position, which credits only along-track advance (perpendicular drift -> 0) and so gives
    NO lateral/vertical homing -- the 2026-07-07 render's diffuse-and-miss failure on a dead-ahead gate.

    A true Euclidean potential: over any closed path the telescoping sum is ~0, so it is NON-farmable
    and cannot farm lateral drift (moving perpendicular TOWARD the centre reduces the distance = genuine
    homing = exactly what we want; moving away is penalised). ``gate_center`` = the CURRENT target gate
    centre (N,3) -- the same seg_end the segment mode projects onto."""
    assert torch is not None
    return -torch.linalg.norm(pos - gate_center, dim=-1)


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
                     banked_progress_return: Tensor, w: EgoRewardWeights) -> Tensor:
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
        forfeit = banked_progress_return.clamp(min=0.0) * coll
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
    r_pass = centering_passage_reward(gate_passed, pass_linf, w_g_half, w.passage,
                                      passed_gate_index=passed_gate_index,
                                      passage_increment=w.passage_increment)
    # DENSE lateral centering: pull onto the current gate-centre segment (off-line -> penalised, on-line
    # -> ~0). Complements the area coupling (angle); this shapes lateral POSITION toward a centred cross.
    r_center = (through_centering_reward(perp_dist, w.centering, w.centering_max_m)
                if perp_dist is not None else torch.zeros_like(r_prog))
    r_fin = finish_reward(newly_finished, time_left_s, w)
    r_cone = free_cone_penalty(tilt_cos_r33, w)
    r_smooth = smoothness_penalty(omega, action_norm, last_action_norm, w)
    r_exit = exit_line_reward(vel_world, curr_center, next_center, gate_passed, w)
    r_time = -w.time
    term = terminal_penalty(gate_collision, gate_miss, oob, banked_progress_return, w)

    reward = r_prog + r_pass + r_center + r_fin + r_cone + r_smooth + r_exit + r_time - term

    components = {
        "prog_reward": float(r_prog.mean()),
        "pass_reward": float(r_pass.mean()),
        "center_pen": float((-r_center).mean()),
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
