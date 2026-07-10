"""PeregrineRacingEgo -- the VQ2 EGOCENTRIC generation env (component C, DESIGN.md
`docs/vq2-egocentric-gen/DESIGN.md` §5.C).

A SUBCLASS of PeregrineRacing (rl/peregrine_racing.py). The pristine diffaero clone is NEVER edited;
the inc7/inc8 envs are NEVER destructively changed (this file imports them, never mutates them).
Registered via ENV_ALIAS["peregrine_racing_ego"] by a launcher (the ENV_ALIAS / monkeypatch pattern,
exactly like rl/peregrine_train_inc8.py) -- no edits to the diffaero clone.

WHAT IT PRODUCES (all gated behind ``+env.ego=true``; default OFF == byte-identical inc7):

  * A 21-DIM EGOCENTRIC, POSITION-FREE actor observation, body-frame, multi-gate, driven per-step by a
    ``BatchedEgoEstimator`` (component A, rl/ego_estimator.py) + ``gate_detectable`` (component B,
    rl/gate_visibility.py):

        velocity(3, body) + roll_pitch(2) + body_rates(3) + last_collective(1)          =  9
        + coarse_sector(2)   [the CURRENT target gate's prebuilt (horiz, vert) in {-1,0,1}]
        + 2 forward window slots [current, next] (WINDOW=2, Fengyou 2026-07-07), each:
              rel_pos(3, body) + confidence(1) + visible_area(1)   = 5  -> 2*5           = 10
        = 21

    NO world position, NO world heading anywhere in the actor obs (the DEPLOY-obtainable set). The
    window is TARGET-RELATIVE (slot 0 = current target gate tg, slot 1 = tg+1), clamped
    at the last gate, and a slot past the last gate (or an undetectable / stale gate) is MASKED: its
    rel_pos, confidence and visible_area are zeros. On a gate pass the target index advances and the
    window PROMOTES (next -> current) with NO teleport -- the next gate was already tracked in its slot.

  * A STATIC PREBUILT per-gate COARSE-MAP sector array ``sector[G] = (horiz, vert)``, each in {-1,0,1}
    (a 3x3 bucket in the drone's gravity-leveled heading frame), AUTO-FILLED from the course geometry
    (the nominal leveled turn/climb the course takes at each gate) and settable/overridable. The obs
    feeds ``sector[tg]`` (the current target's sector, 2 dims) -- an acquisition/anticipation prior
    (where to look before the target is visible), NOT a competing position signal.

  * The PRIVILEGED (asymmetric) critic state: GROUND-TRUTH body-frame rel_pos for the 2 window gates +
    true body velocity + true roll/pitch/body-rates + the per-gate confidence (the actor sees the
    NOISED A outputs; the critic sees truth). Documented layout below (``ego_critic_state``).

  * HARD KILL-ON-CONTACT (absolute): ANY gate contact -> immediate episode termination (done=True) + a
    LARGE NEGATIVE reward that step. Zero contact is THE validity rule (never a soft penalty). This
    REUSES the base env's collision-based termination on the TRUE frame geometry (``gate_collision``
    from the inc7 crossing/slab classification) and adds the large negative on top.

PPO-ONLY / NO_GRAD: the estimator emulation (Bernoulli fix / teleport outlier / colored gyro) is
non-differentiable and runs under ``torch.no_grad()``; reward is detached (mirrors the parent).

The pure obs/critic/coarse-map ASSEMBLY logic lives in module-level functions (``build_coarse_map``,
``ego_actor_obs``, ``ego_critic_state``, ``ego_window_indices``) so it is unit-testable WITHOUT the
diffaero dynamics (which is a cluster-only dependency). The env class just wires them to the truth
tensors + the estimator each step.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None
    Tensor = "Tensor"                   # type: ignore

# Component A (estimator) + B (visibility). Pure-torch, no diffaero.
from ego_estimator import BatchedEgoEstimator, EgoEstimatorConfig, EgoEstimate
from gate_visibility import gate_detectable, gate_apparent_area, gate_center_view_cos

# The REFINED-B (champion-consensus) reward -- pure functions, laptop-testable. This is the reward
# THIS GENERATION TRAINS ON (default ON when +env.ego=true); it REPLACES the inc7 option-B reward the
# env previously reused via peregrine_racing.compute_reward_terms. See rl/ego_reward.py + DESIGN.md.
from ego_reward import (EgoRewardWeights, segment_arc_position, segment_perp_distance,
                        gate_center_potential, compute_ego_reward, wide_flyby_miss)
# Vector-field (GVF) racing line: online, batched, per-episode NON-OPTIMAL line through the gate
# centres, head-on at each gate. When ``+env.use_racing_line=true`` the progress potential s and the
# contouring perp are measured against THIS curved line (arc length / cross-track) instead of the
# straight current-gate segment. See rl/racing_line.py.
from racing_line import build_racing_line, RacingLine

# The obs contract dimensions (FIXED). 2-gate slider [current, next] (Fengyou 2026-07-07).
WINDOW = 2                              # [current, next]
PER_SLOT = 5                            # rel_pos(3) + confidence(1) + visible_area(1)
EGO_OBS_DIM = 9 + 2 + WINDOW * PER_SLOT # 9 (vel/rp/rates/coll) + 2 (sector) + 10 = 21
# critic: v(3) + roll_pitch(2) + body_rates(3) + 2*(rel_pos 3 + conf 1) = 8 + 8 = 16
EGO_CRITIC_DIM = 3 + 2 + 3 + WINDOW * (3 + 1)

# Large negative applied on ANY gate contact (kill-on-contact). A config knob (``+env.ego_contact_penalty``).
EGO_CONTACT_PENALTY_DEFAULT = 50.0


# ================================================================================================
# Course-sampler override resolution -- the CURRICULUM->SAMPLER key bridge (component D / DESIGN.md §D).
# ================================================================================================
# The ego curriculum (rl/vq2_ego_curriculum.py) varies the course PER STAGE via +env. keys:
#   course_n_gates  (single_gate=1, dual/handoff=2, multi=6)
#   course_seg_len_lo / course_seg_len_hi  (10-20 m VQ2 spacing)
#   course_drop_lo  / course_drop_hi       (optional per-segment descent band)
# peregrine_course.sample_courses accepts these as **overrides under DIFFERENT names -- n_gates (int),
# seg_len_m (lo,hi), drop_m (lo,hi). The BASE env's _assign_courses calls sample_courses with NO
# overrides (and hard-codes n_gates=6 from the VQ1 JSON), so those curriculum keys are UNCONSUMED on the
# inc7/inc8 path -- inc8 additionally PINS course_mode=vq1 (fixed 6-gate reference line), so it never
# varies gate count at all. The ego generation is the FIRST consumer: this pure function maps the
# curriculum keys onto the sampler's kwargs so single_gate is genuinely 1 gate, dual 2, multi 6, at
# 10-20 m spacing. Pure (no torch, no diffaero) -> laptop-testable; the env applies the result.
_COURSE_SAMPLER_DEFAULTS = {                    # sampler names the ego curriculum keys map ONTO
    "course_n_gates": "n_gates",               # int
    # seg_len / drop are (lo, hi) PAIRS assembled below (each half is its own +env. key).
}


def resolve_course_overrides(cfg) -> dict:
    """Map the ego curriculum's ``course_*`` cfg keys onto peregrine_course.sample_courses ``**overrides``
    (n_gates / seg_len_m / drop_m). Returns a dict of ONLY the keys that were set (unset -> the sampler's
    own defaults, i.e. 6 gates / VQ1-derived ranges). Pure: reads via getattr, no torch/diffaero, so a
    unit test can pin the mapping without the cluster env.

    Recognized cfg keys:
      course_n_gates                              -> n_gates      (int)
      course_seg_len_lo,   course_seg_len_hi      -> seg_len_m    (lo, hi)   [both must be set together]
      course_drop_lo,      course_drop_hi         -> drop_m       (lo, hi)   [both must be set together]
      course_spawn_dist_lo,course_spawn_dist_hi   -> spawn_dist_m (lo, hi)   [both must be set together]
      course_spawn_below_g0_lo, course_spawn_below_g0_hi -> spawn_below_g0_m (lo, hi) [gate-0 HEIGHT band;
                                                    +ve == gate ABOVE the pad. Vary it to un-bury the
                                                    vertical signal geometrically -- a gate at varying
                                                    heights makes the distance-to-gate's vertical component
                                                    non-trivial from the start (the flat same-height gate is
                                                    the degenerate worst case for the floor-dive).]
      course_spawn_heading                        -> spawn_heading (scalar)  [pins segment-0 heading]
      course_gates_above_spawn                    -> gates_above_spawn_m (scalar) [A1 floor fix
                                                    2026-07-10: keep EVERY gate centre z >= pad z +
                                                    this clearance. spawn_below_g0 only constrains
                                                    gate 0; with the default drop band a LATER gate
                                                    can sink below the pad -- undivable-to once
                                                    floor_at_spawn is on.]

    ``spawn_dist_m`` is the standing-start pad -> gate-0 horizontal distance (Fengyou 2026-07-07: keep
    the FIRST gate 10-20 m out, not the sampler's default 18-28 m -- a shorter first approach is easier
    to discover and leaves less altitude to bleed before the gate). ``spawn_heading`` pins the segment-0
    world heading (Fengyou 2026-07-07: the egocentric obs is heading-invariant, so the sampler's random
    heading is a redundant global DOF that just fans the world-frame layout into a confusing circle; the
    ego sets 0.0 so every course starts ahead of the pad, with the egocentric distribution unchanged).

    A half-specified pair (only lo OR only hi) raises ValueError -- a silent half-override would sample
    the wrong band and waste compute (the exact class of bug this wiring exists to prevent)."""
    out: dict = {}
    n_gates = getattr(cfg, "course_n_gates", None)
    if n_gates is not None:
        ng = int(n_gates)
        if ng < 1:
            raise ValueError(f"course_n_gates must be >= 1, got {ng}")
        out["n_gates"] = ng
    out.update(_resolve_pair(cfg, "course_seg_len_lo", "course_seg_len_hi", "seg_len_m"))
    out.update(_resolve_pair(cfg, "course_drop_lo", "course_drop_hi", "drop_m"))
    out.update(_resolve_pair(cfg, "course_spawn_dist_lo", "course_spawn_dist_hi", "spawn_dist_m"))
    out.update(_resolve_pair(cfg, "course_spawn_below_g0_lo", "course_spawn_below_g0_hi",
                             "spawn_below_g0_m"))
    # course_spawn_heading (scalar): pin the segment-0 world heading so the egocentric courses do not fan
    # into a redundant circle (the obs is heading-invariant). Unset -> the sampler's random heading.
    spawn_heading = getattr(cfg, "course_spawn_heading", None)
    if spawn_heading is not None:
        out["spawn_heading"] = float(spawn_heading)
    # course_spawn_yaw_jitter (scalar, Fengyou 2026-07-08): jitter the drone's spawn yaw off the gate bearing
    # so the gate lands across the FOV (realistic left/right variation). Unset -> 0 (dead-ahead, legacy).
    spawn_yaw_jitter = getattr(cfg, "course_spawn_yaw_jitter", None)
    if spawn_yaw_jitter is not None:
        out["spawn_yaw_jitter_rad"] = float(spawn_yaw_jitter)
    # course_gates_above_spawn (scalar, A1 floor fix 2026-07-10): sampler-side floor -- every gate
    # centre z >= pad z + clearance (see peregrine_course.gates_above_spawn_m). Unset -> OFF (legacy).
    gates_above = getattr(cfg, "course_gates_above_spawn", None)
    if gates_above is not None:
        ga = float(gates_above)
        if ga < 0.0:
            raise ValueError(f"course_gates_above_spawn must be >= 0, got {ga}")
        out["gates_above_spawn_m"] = ga
    return out


def _resolve_pair(cfg, lo_key: str, hi_key: str, sampler_key: str) -> dict:
    """Assemble a (lo, hi) sampler override from two cfg half-keys; raise if exactly one is set."""
    lo = getattr(cfg, lo_key, None)
    hi = getattr(cfg, hi_key, None)
    if lo is None and hi is None:
        return {}
    if lo is None or hi is None:
        raise ValueError(f"{lo_key} and {hi_key} must BOTH be set (or neither); got "
                         f"{lo_key}={lo!r}, {hi_key}={hi!r}")
    lo_f, hi_f = float(lo), float(hi)
    if lo_f > hi_f:
        raise ValueError(f"{lo_key}={lo_f} must be <= {hi_key}={hi_f}")
    return {sampler_key: (lo_f, hi_f)}


# ================================================================================================
# Coarse map -- STATIC PREBUILT per-gate 3x3 sector, AUTO-FILLED from course geometry.
# ================================================================================================
def build_coarse_map(gate_pos_zup: Tensor, spawn_pos_zup: Tensor,
                     horiz_thresh_rad: float = 0.20, vert_thresh_rad: float = 0.20) -> Tensor:
    """Auto-fill the per-gate coarse-map sector ``sector[N,G,2]`` (horiz, vert) each in {-1,0,1} from
    the course geometry, in the drone's GRAVITY-LEVELED heading frame.

    The nominal APPROACH direction into gate g is ``a_g = normalize(gate[g] - prev[g])`` where prev is
    gate[g-1] (or the spawn for g=0). The sector answers "relative to how you arrive AT gate g, where
    does the NEXT leg (toward gate g+1) bend?" -- the acquisition/anticipation prior the policy uses to
    know roughly where to look for the upcoming target before it is visible:
      * horiz bucket = sign of the LEVELED azimuth turn from the incoming leg a_g to the outgoing leg
        a_{g+1} (left/centre/right = {-1,0,1}); the last gate has no outgoing leg -> centre (0).
      * vert bucket  = sign of the ELEVATION of the outgoing leg (down/level/up = {-1,0,1}); last gate
        -> the elevation of the incoming leg (there is nowhere further to bend, so anticipate holding).

    Buckets are the SIGN past a deadband (``horiz_thresh_rad`` / ``vert_thresh_rad``): within the band
    -> 0 (centre). Gravity-leveled == azimuth measured in the world XY plane (Z-up), elevation from the
    world +Z, so a pure global-yaw of the whole scene rotates BOTH legs equally and leaves the turn
    sign invariant (the sector is heading-relative, position-free).

    gate_pos_zup (N,G,3) Z-up; spawn_pos_zup (N,3) Z-up. Returns LongTensor (N,G,2) in {-1,0,1}.
    """
    assert torch is not None, "build_coarse_map requires torch"
    N, G, _ = gate_pos_zup.shape
    dev, dt = gate_pos_zup.device, gate_pos_zup.dtype
    # incoming leg direction a_g = gate[g] - prev[g] (prev[0] = spawn)
    prev = torch.empty_like(gate_pos_zup)
    prev[:, 0, :] = spawn_pos_zup
    if G > 1:
        prev[:, 1:, :] = gate_pos_zup[:, :-1, :]
    incoming = gate_pos_zup - prev                                       # (N,G,3)
    # outgoing leg direction b_g = gate[g+1] - gate[g] (last gate: reuse the incoming leg)
    outgoing = torch.empty_like(gate_pos_zup)
    if G > 1:
        outgoing[:, :-1, :] = gate_pos_zup[:, 1:, :] - gate_pos_zup[:, :-1, :]
    outgoing[:, -1, :] = incoming[:, -1, :]

    def _azimuth(v):                    # leveled heading angle in world XY (Z-up)
        return torch.atan2(v[..., 1], v[..., 0])

    def _elevation(v):                  # angle above the world XY plane
        horiz_mag = torch.linalg.norm(v[..., :2], dim=-1).clamp(min=1e-9)
        return torch.atan2(v[..., 2], horiz_mag)

    # HORIZ: signed leveled turn from the incoming to the outgoing leg, wrapped to (-pi, pi].
    turn = _azimuth(outgoing) - _azimuth(incoming)                       # (N,G)
    turn = torch.atan2(torch.sin(turn), torch.cos(turn))                 # wrap
    # VERT: elevation of the outgoing leg (last gate -> incoming, handled by the reuse above).
    elev = _elevation(outgoing)                                         # (N,G)

    horiz = torch.zeros(N, G, device=dev, dtype=torch.long)
    horiz[turn > horiz_thresh_rad] = 1
    horiz[turn < -horiz_thresh_rad] = -1
    vert = torch.zeros(N, G, device=dev, dtype=torch.long)
    vert[elev > vert_thresh_rad] = 1
    vert[elev < -vert_thresh_rad] = -1
    return torch.stack([horiz, vert], dim=-1)                            # (N,G,2) long in {-1,0,1}


# ================================================================================================
# Target-relative window indexing (current, next, next-next) with course-end clamping + mask.
# ================================================================================================
def ego_window_indices(target_gates: Tensor, n_gates: int):
    """Return ``(gidx (N,WINDOW) long, valid (N,WINDOW) bool)``.

    gidx[:, k] = clamp(tg + k, max=n_gates-1) -- the clamped gate index feeding slot k (so a slot past
    the last gate reuses the last gate's index but is MASKED). valid[:, k] = (tg + k) < n_gates -- a
    slot past the last real gate is INVALID (-> masked to zeros in the obs). This is the ONLY window
    bookkeeping: as ``target_gates`` advances on a pass, slot k reads gate tg+k, so the window promotes
    (the old slot 1 gate becomes slot 0) with no teleport."""
    ar = torch.arange(WINDOW, device=target_gates.device)
    raw = target_gates.long().unsqueeze(-1) + ar.unsqueeze(0)           # (N,WINDOW)
    valid = raw < n_gates
    gidx = raw.clamp(max=n_gates - 1)
    return gidx, valid


# ================================================================================================
# Actor obs assembly (26-dim, position-free) -- PURE, testable without diffaero.
# ================================================================================================
def ego_actor_obs(est: EgoEstimate, detectable: Tensor, target_gates: Tensor,
                  last_collective: Tensor, sector: Tensor, n_gates: int,
                  obs_coast: bool = False) -> Tensor:
    """Assemble the 26-dim egocentric actor observation from the estimator outputs + visibility + the
    coarse map. POSITION-FREE (only body-frame velocity / attitude / rates / relative geometry + the
    heading-relative sector).

    Inputs:
      est               EgoEstimate (component A): velocity(N,3) body, roll_pitch(N,2), body_rates(N,3),
                        rel_pos(N,G,3) body, confidence(N,G), visible_area(N,G).
      detectable        (N,G) bool -- component B visibility.
      target_gates      (N,) current target gate index.
      last_collective   (N,) or (N,1) last rescaled collective (thrust) command.
      sector            (N,G,2) coarse-map sector (from build_coarse_map or an override).
      n_gates           int.
      obs_coast         OBS BLACKOUT COAST (Fengyou 2026-07-09, ``+env.ego_obs_coast``). Default False ==
                        BYTE-IDENTICAL legacy: a slot is masked whenever the gate is NOT detectable THIS
                        step (keep = valid & det & conf>0), so the estimator's coasted prior is ZEROED the
                        instant the gate goes non-detectable and the crossing endgame is flown on zeros
                        (audit read_estimator-kf-audit.md STALE-CLIFF). When True the instantaneous ``det``
                        term is DROPPED (keep = valid & conf>0): the coasted rel_pos + LINEARLY-DECAYING
                        confidence feed the obs THROUGH a blackout, hard-masking only past the estimator's
                        stale horizon (conf==0) -- the DESIGN.md §5.A "brief ego-motion propagation through
                        gaps; mask past ~0.5-1 s stale" intent. Confidence already encodes staleness
                        (EgoEstimate masks it to 0 past the horizon), so conf>0 alone is the in-horizon
                        test; det is AND'd in ONLY for the legacy hard-mask.

    Slot k (k=0 current, 1 next) reads gate g=clamp(tg+k). A slot is MASKED (rel_pos=0, confidence=0,
    visible_area=0) when the slot is past the last gate OR confidence==0 (stale past horizon) OR --
    UNLESS obs_coast -- the gate is not detectable this step. The coarse_sector fed is sector[tg].

    Returns (N, 26)."""
    assert torch is not None
    N = est.rel_pos.shape[0]
    dev, dt = est.rel_pos.device, est.rel_pos.dtype
    ar = torch.arange(N, device=dev)
    tg = target_gates.long()

    if last_collective.dim() == 1:
        last_collective = last_collective.unsqueeze(-1)                 # (N,1)

    gidx, valid = ego_window_indices(tg, n_gates)                       # (N,WINDOW)

    # per-slot gather of rel_pos / confidence / visible_area, then mask
    slots = []
    for k in range(WINDOW):
        g = gidx[:, k]                                                  # (N,)
        rel = est.rel_pos[ar, g]                                        # (N,3) body-frame
        conf = est.confidence[ar, g]                                    # (N,)
        area = est.visible_area[ar, g]                                  # (N,)
        det = detectable[ar, g]                                        # (N,)
        # OBS BLACKOUT COAST: conf>0 is the in-horizon test (confidence is already masked to 0 past the
        # stale horizon in EgoEstimate). Legacy (obs_coast=False) ALSO requires instantaneous detectability,
        # zeroing the coasted prior during a blackout; coast keeps the coasted estimate flowing until the
        # horizon. AND is commutative -> coast=False is bit-identical to the prior `valid & det & (conf>0)`.
        keep = valid[:, k] & (conf > 0.0)                             # (N,) in-horizon test
        if not obs_coast:
            keep = keep & det                                          # legacy hard-mask on this-step visibility
        keep_f = keep.to(dt)
        rel = rel * keep_f.unsqueeze(-1)
        conf = conf * keep_f
        area = area * keep_f
        slots.append(torch.cat([rel, conf.unsqueeze(-1), area.unsqueeze(-1)], dim=-1))  # (N,5)

    coarse_sector = sector[ar, tg].to(dt)                              # (N,2) current target's sector

    obs = torch.cat([
        est.velocity,                                                  # (N,3) body velocity
        est.roll_pitch,                                               # (N,2)
        est.body_rates,                                              # (N,3)
        last_collective,                                            # (N,1)
        coarse_sector,                                              # (N,2)
        *slots,                                                     # 3 x (N,5) = (N,15)
    ], dim=-1)
    return obs                                                        # (N,26)


# ================================================================================================
# Privileged critic state assembly -- GROUND TRUTH (no noise), documented layout.
# ================================================================================================
def ego_critic_state(rel_pos_true_body: Tensor, vel_body_true: Tensor, roll_pitch_true: Tensor,
                     body_rates_true: Tensor, confidence: Tensor, target_gates: Tensor,
                     n_gates: int) -> Tensor:
    """Assemble the PRIVILEGED critic state (GROUND TRUTH god-view).

    LAYOUT (EGO_CRITIC_DIM = 20), in order:
      [0:3]   true body-frame velocity
      [3:5]   true roll, pitch
      [5:8]   true body rates
      then per window slot k in {0,1,2} (current, next, next-next), clamped at the last gate:
        [8+4k : 8+4k+3]   TRUE body-frame rel_pos of gate clamp(tg+k)      (3)
        [8+4k+3]          per-gate confidence of gate clamp(tg+k)          (1)
    -> 8 + 3*4 = 20.

    The critic sees the TRUE relative geometry (unmasked, un-noised) for all window gates + the true
    velocity/attitude/rates, plus the (noised-pipeline) per-gate confidence so it can attribute the
    actor's uncertainty. A slot past the last gate reuses the last gate's TRUE rel_pos (clamped) -- the
    critic is privileged, so no masking is needed; the confidence still reflects visibility.

    rel_pos_true_body (N,G,3); vel_body_true (N,3); roll_pitch_true (N,2); body_rates_true (N,3);
    confidence (N,G); target_gates (N,). Returns (N, 20)."""
    assert torch is not None
    N = rel_pos_true_body.shape[0]
    dev, dt = rel_pos_true_body.device, rel_pos_true_body.dtype
    ar = torch.arange(N, device=dev)
    gidx, _ = ego_window_indices(target_gates, n_gates)                 # (N,WINDOW)
    parts = [vel_body_true, roll_pitch_true, body_rates_true]
    for k in range(WINDOW):
        g = gidx[:, k]
        parts.append(rel_pos_true_body[ar, g])                         # (N,3) TRUE
        parts.append(confidence[ar, g].unsqueeze(-1))                  # (N,1)
    return torch.cat(parts, dim=-1)                                    # (N,20)


def apply_contact_kill(reward: Tensor, gate_collision: Tensor, penalty: float) -> Tensor:
    """The ABSOLUTE kill-on-contact reward: subtract a LARGE NEGATIVE (``penalty``) on ANY gate contact
    (``gate_collision``), never a soft shaping term. Pure, so the env and the test share ONE code path.
    Returns the penalised reward; the caller marks the same envs terminated (done=True)."""
    return reward - penalty * gate_collision.to(reward.dtype)


def true_rel_pos_body(gate_pos_zup: Tensor, drone_pos_zup: Tensor, R_wb_zup: Tensor) -> Tensor:
    """GT per-gate drone->gate vector in the drone BODY frame (N,G,3) = R_wb^T @ (gate_pos - drone_pos).
    A pure difference rotated into the body frame -> translation-invariant, holds no world coordinate."""
    lever = gate_pos_zup - drone_pos_zup.unsqueeze(1)                   # (N,G,3)
    return torch.einsum("nji,ngj->ngi", R_wb_zup, lever)


# ================================================================================================
# The environment (requires diffaero -- training/eval cluster only).
# ================================================================================================
# The base import is deferred to class-body time so that the PURE functions above are importable on a
# machine without diffaero (the unit tests import this module and use the functions only).
try:                                    # pragma: no cover - exercised on the cluster
    from peregrine_racing import (PeregrineRacing, world_to_gateframe, crossing_events,
                                  slab_frame_hits, tilt_cos_from_quat_xyzw, roll_from_quat_xyzw,
                                  compute_reward_terms, rel_tables)
    from inc8_estimator_emul import quat_xyzw_to_matrix_torch, _RZ_PI_BODY_NP
    _HAVE_DIFFAERO = True
except Exception:                       # pragma: no cover
    _HAVE_DIFFAERO = False
    PeregrineRacing = object            # type: ignore


class PeregrineRacingEgo(PeregrineRacing):          # pragma: no cover - cluster-only (needs diffaero)
    """The VQ2 egocentric env. OFF (``+env.ego`` unset/false) -> byte-identical inc7."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)               # builds the inc7 env (gate tensors, contact geom)
        self._ego_on = bool(getattr(cfg, "ego", False))
        if not self._ego_on:
            return                                  # pure inc7 fallback
        if torch is None:                           # pragma: no cover
            raise RuntimeError("ego env requires torch")

        dev = self.device
        self._ego_dtype = self.gate_pos.dtype
        # EMULATED-CAMERA VIRTUAL FLIP (RC1 fix, 2026-07-07): the drone flies TAIL-FIRST (the VQ1/CTBR
        # control alias -- DO NOT touch the control sign config), but the REAL deploy camera is
        # NOSE-FIRST, so the emulated camera must apply fly_rl's pi-about-body-z flip (_RZ_PI_BODY) to
        # look along the TRAVEL direction and SEE the gates ahead. Without it the camera points ~180 deg
        # away from travel -> the gate is NEVER detectable (proven locally: 0% at spawn/approach/5m,
        # flipped=100%) -> rel_pos stays masked -> single_gate is unlearnable (the observed collapse).
        # Only the CAMERA is rotated; the control frame, rel_pos, velocity and rates keep the unflipped
        # tail-first body frame (consistent). ON by default (unflipped was a bug);
        # +env.ego_camera_virtual_flip=false restores the raw camera for A/B.
        self._cam_flip = bool(getattr(cfg, "ego_camera_virtual_flip", True))
        self._Rz_cam = torch.as_tensor(_RZ_PI_BODY_NP, device=dev, dtype=self._ego_dtype)
        self._ego_contact_penalty = float(getattr(cfg, "ego_contact_penalty",
                                                   EGO_CONTACT_PENALTY_DEFAULT))
        # MISS TERMINATION (Fengyou A/B 2026-07-07): default True -> a wide flyby TERMINATES (miss, the
        # inc7/refined-B behaviour). +env.miss_terminates=false -> a wide flyby does NOT terminate and
        # carries NO terminal penalty: the drone flies past and must stay in-bounds + re-approach to
        # score (tests whether removing the clean 'safe-miss' exit helps vs the centering fix alone).
        # CONTACT and OOB always terminate regardless.
        self._miss_terminates = bool(getattr(cfg, "miss_terminates", True))

        # ===== COURSE VARIATION (component D / DESIGN.md §D): make single_gate REALLY 1 gate, dual 2,
        # multi 6, at the VQ2 10-20 m spacing. The base env hard-codes n_gates=6 (VQ1 JSON) and calls
        # sample_courses with NO overrides, so the curriculum's course_n_gates / course_seg_len_lo/hi /
        # course_drop_lo/hi are UNCONSUMED on the inc7/inc8 path. The ego generation IS the consumer: we
        # forward them to the sampler. Requires course_mode=random (the ego curriculum sets it); with the
        # default course_mode=vq1 there is nothing to vary, so leave the fixed VQ1 course untouched.
        self._course_overrides = resolve_course_overrides(cfg) if self.course_mode == "random" else {}
        if self._course_overrides:
            self._apply_course_overrides(self._arange, first_build=True)

        # ===== REFINED-B reward (this generation trains on it; DEFAULT ON under +env.ego=true) =====
        # ``+env.reward_refined_b=false`` -> fall back to the legacy inc7 option-B path (the old
        # compute_reward_terms spine) for an A/B, but the DEFAULT for the ego generation is refined-B.
        self._use_refined_b = bool(getattr(cfg, "reward_refined_b", True))
        self._egorw = EgoRewardWeights.from_cfg(cfg)
        # per-env progress-potential state (segment arc position s_prev) + the accumulated
        # (undiscounted) progress return, for the PROGRESS-SCALED terminal (forfeit banked progress).
        self._seg_s_prev = torch.zeros(self.n_envs, device=dev, dtype=self._ego_dtype)
        self._banked_prog = torch.zeros(self.n_envs, device=dev, dtype=self._ego_dtype)
        # per-env PBRS contouring potential state: the previous-step perpendicular offset from the current
        # gate-centre segment (for the MPCC contouring term corridor*(perp_prev - perp_curr)).
        self._corr_perp_prev = torch.zeros(self.n_envs, device=dev, dtype=self._ego_dtype)
        # ONCE-PER-GATE parabola latch buffer (ego_reward.crossing_parabola_reward contract; the audit
        # re-payment-farm fix): per-env bool, passed into compute_ego_reward EVERY step and mutated IN
        # PLACE there (marked on a forward target-plane crossing) ONLY when rw_parabola_latch is on.
        # The env owns the CLEARS: (1) on a target ADVANCE, AFTER the step's reward is computed (new
        # gate -> new payment window), and (2) in reset_idx on EVERY episode reset path -- terminated
        # AND truncated/timeout both funnel through step()'s reset_idx call, so a stale latch can never
        # suppress the next episode's first crossing. With parabola_latch_once False (the default) the
        # reward fn neither reads nor mutates the buffer -> byte-identical.
        self._parabola_paid = torch.zeros(self.n_envs, dtype=torch.bool, device=dev)
        # VECTOR-FIELD racing line (Fengyou greenlight 2026-07-08). When ON, the progress potential s and
        # the contouring perp are read from the CURVED head-on racing line (arc length / cross-track)
        # instead of the straight current-gate segment. Built per-env at reset from GT spawn + gate
        # centres/normals; consumed only by the reward (never the position-free obs).
        self._use_racing_line = bool(getattr(cfg, "use_racing_line", False))
        self._racing_samples_per_seg = int(getattr(cfg, "racing_samples_per_seg", 24))
        # DECOUPLE knob (Fengyou 2026-07-08 race read): when use_racing_line is on, does the PROGRESS
        # potential come from the line arc-length too (True, the pure-GVF form), or does progress stay on
        # the isotropic gate_center_potential HOMING while the line supplies ONLY the cross-track
        # contouring (False = "homing + line-centering")? The race proved isotropic homing is the ONLY
        # form that reaches the gate plane (70% vs the along-line/segment-lag forms' side/floor divergence),
        # so the winning recipe KEEPS homing and only ADDS the line's perpendicular centering. Default True
        # (pure GVF); set +env.racing_line_progress=false for the homing+contouring variant.
        self._racing_line_progress = bool(getattr(cfg, "racing_line_progress", True))
        self._racing_line: RacingLine | None = None
        # FRAME-MOAT fix (Fengyou 2026-07-08): when True, a FRAME-CLIP forfeits no banked progress (net ==
        # a wide miss) so the ring around the aperture is not a moat that punishes getting close; the FLOOR
        # dive (a real crash) still forfeits. Default False = legacy (every contact forfeits).
        self._frame_clip_is_miss = bool(getattr(cfg, "frame_clip_is_miss", False))
        # APERTURE CURRICULUM (Fengyou 2026-07-08): override the gate half-opening (the pass/thread radius +
        # the passage-centering scale w_g_half). The real 0.75m aperture has a DEAD ZONE -- a crossing at the
        # 3.4m plateau earns ZERO passage reward (linf >> 0.75) so nothing sparse pulls it in. A WIDER training
        # aperture gives the passage reward a gradient across the current offset; shrink it toward 0.75m across
        # warm-started stages so the policy centres progressively. Only the INNER scales (base untouched); when
        # the aperture exceeds the outer frame the frame simply vanishes (a soft target) until it shrinks back.
        # NOTE: the `thread`/success metric is APERTURE-RELATIVE, so during a wide stage watch cross_offset_m
        # (aperture-independent), not thread. None -> the real spec aperture. Set via +env.gate_inner_opening_m.
        _ap = getattr(cfg, "gate_inner_opening_m", None)
        if _ap is not None:
            self.gate_half_opening_m = float(_ap) / 2.0
        # OBS BLACKOUT COAST (Fengyou 2026-07-09; audit read_estimator-kf-audit.md STALE-CLIFF). Default
        # False == byte-identical (the obs builder hard-masks each gate slot on INSTANTANEOUS detectability,
        # so the estimator's coast-through-gaps never reaches the policy and the crossing endgame is flown
        # on zeros -- even with a perfect estimator). +env.ego_obs_coast=true drops the instantaneous `det`
        # term in ego_actor_obs so the coasted rel_pos + linearly-decaying confidence feed the obs during a
        # blackout, masking only past the estimator's stale horizon (conf==0) -- the DESIGN.md §5.A intent.
        self._ego_obs_coast = bool(getattr(cfg, "ego_obs_coast", False))
        # estimator config (all knobs Fengyou-pinnable via +env.*)
        ecfg = EgoEstimatorConfig(
            visible_area_sigma=float(getattr(cfg, "ego_visible_area_sigma",
                                             EgoEstimatorConfig.visible_area_sigma)),
            far_cap_m=float(getattr(cfg, "ego_far_cap_m", EgoEstimatorConfig.far_cap_m)),
            noise_scale=float(getattr(cfg, "ego_noise_scale", EgoEstimatorConfig.noise_scale)),
            # STALE HORIZON (Fengyou 2026-07-09, +env.ego_stale_horizon_s; default 0.5 = byte-identical).
            # Measured terminal blind onset is 0.7-2.2 m in ALL geometries, so at slow-lap speeds the
            # 0.5 s horizon zeroes confidence BEFORE the crossing even with ego_obs_coast on (the coast
            # can only feed the obs while conf>0). The upcoming flight sets ~1.2 s so the coasted prior
            # survives the endgame blackout to the gate plane.
            stale_horizon_s=float(getattr(cfg, "ego_stale_horizon_s",
                                          EgoEstimatorConfig.stale_horizon_s)),
        )
        self._ego_cfg = ecfg
        # the estimator holds per-env course geometry (Z-up, matching gate_visibility/ego_estimator)
        self._estimator = BatchedEgoEstimator(
            self.n_envs, self.gate_pos, self.gate_yaw, config=ecfg,
            device=dev, dtype=self._ego_dtype)
        # STATIC coarse map, auto-filled from the course geometry; settable/overridable via set_coarse_map
        self._coarse_map = build_coarse_map(self.gate_pos, self.spawn_pos)   # (N,G,2) long
        self._prev_q = self._q.clone()
        # GT apparent opening area (normalized, square-on==1) for ALL gates, refreshed every _step_estimator
        # (the reward's area-distance coupling reads the current target's). Zeros until the first step.
        self._apparent_area_gt = torch.zeros(self.n_envs, self.n_gates, device=dev, dtype=self._ego_dtype)
        # The estimator is STEPPED exactly ONCE per env-step (inside step()). get_observations() and
        # get_state() only READ estimator STATE via .estimate() (never re-step) -- a double step would
        # corrupt the per-gate staleness clocks / colored-gyro AR(1) state. This mirrors inc8, where
        # step() advances the KF once and get_observations reads kf_pos_zup(). ``detectable`` is a pure
        # stateless function of the current truth, so it is recomputed where needed (cheap, no state).
        self._stepped = False               # has the estimator been stepped since the last reset?

        # obs/critic dims (deploy/load gate on the sidecar obs-dim)
        self.obs_dim = EGO_OBS_DIM                  # 21
        # privileged critic state (GT god-view). Differs from obs_dim (a DIFFERENT layout, carrying GT
        # truth the actor cannot see); it is LOWER-dim only because the actor also carries noisy-obs-
        # only channels (visible_area, coarse sector, last_collective) the critic omits.
        self.state_dim = EGO_CRITIC_DIM             # 16
        self._reset_estimator(self._arange)
        self._rebuild_racing_line(self._arange)
        if self._use_refined_b:
            seg_a, seg_b = self._current_segment(self._arange)
            self._seg_s_prev.copy_(self._progress_scalar(self._p, seg_a, seg_b))
            self._banked_prog.zero_()
            self._corr_perp_prev.copy_(self._perp_scalar(self._p, seg_a, seg_b))

    # ---- course-variation wiring (component D): forward the curriculum course_* keys to the sampler --
    def _sample_ego_courses(self, m: int):
        """Sample ``m`` fresh courses at the STAGE's gate count / spacing (self._course_overrides fed to
        peregrine_course.sample_courses as **overrides: n_gates / seg_len_m / drop_m). Returns the raw
        sample_courses dict."""
        return self._sample_courses(m, device=self.device, **self._course_overrides)

    def _apply_course_overrides(self, env_idx, first_build: bool) -> None:
        """Apply the stage's course overrides.

        first_build=True (once, in __init__): the requested n_gates may DIFFER from the base env's
        VQ1-derived n_gates=6, so REALLOCATE every G-shaped course tensor to the new gate count, sample a
        fresh course for ALL envs, rebuild the rel-tables + OOB boxes, and reset the target index to a
        valid range. After this the env's n_gates / gate tensors reflect the stage (single_gate -> 1
        gate, dual -> 2, multi -> 6), so the estimator + coarse map (built right after) get the right
        geometry.

        first_build=False (per-reset, via the _assign_courses override): n_gates is already correct;
        just resample the given envs in place (the base _assign_courses path, but WITH the overrides)."""
        m = int(env_idx.numel())
        if m == 0:
            return
        if not first_build:
            c = self._sample_ego_courses(m)
            self.gate_pos[env_idx] = c["gate_pos"].to(self.gate_pos.dtype)
            self.gate_yaw[env_idx] = c["gate_yaw"].to(self.gate_yaw.dtype)
            self.spawn_pos[env_idx] = c["spawn_pos"].to(self.spawn_pos.dtype)
            self.spawn_yaw[env_idx] = c["spawn_yaw"].to(self.spawn_yaw.dtype)
            return

        # ---- FIRST BUILD: (re)allocate to the stage's gate count ----
        n = self.n_envs
        c = self._sample_ego_courses(n)                       # ALL envs, at the stage n_gates
        new_G = int(c["gate_pos"].shape[1])
        self.n_gates = new_G
        self.gate_pos = c["gate_pos"].to(self.gate_pos.dtype).contiguous()         # (N, new_G, 3)
        self.gate_yaw = c["gate_yaw"].to(self.gate_yaw.dtype).contiguous()         # (N, new_G)
        self.spawn_pos = c["spawn_pos"].to(self.spawn_pos.dtype).contiguous()      # (N, 3)
        self.spawn_yaw = c["spawn_yaw"].to(self.spawn_yaw.dtype).contiguous()      # (N,)
        # rel-tables + OOB boxes for the NEW geometry
        self.gate_rel_pos, self.gate_yaw_rel = rel_tables(self.gate_pos, self.gate_yaw)
        self._update_boxes(self._arange)
        # the base init set target_gates via the first reset with n_gates=6; clamp into the new range
        # (a later reset re-samples 0..new_G-1). target_pos follows the (clamped) target.
        self.target_gates = self.target_gates.clamp(max=new_G - 1)
        self.target_pos.copy_(self.gate_pos[self._arange, self.target_gates.long()])

    # ---- per-reset course resampling: override to forward the stage overrides (ego path only) --------
    def _assign_courses(self, env_idx) -> None:
        """Ego override of the base course resampler: when a stage carries course overrides, resample the
        given envs at the stage's gate count / spacing (the base version calls sample_courses with NO
        overrides -> always the sampler-default 6 gates / VQ1 ranges). Falls back to the base behaviour
        when there are no overrides (or on the pure inc7 OFF path, where _course_overrides is absent)."""
        if getattr(self, "_ego_on", False) and getattr(self, "_course_overrides", None):
            self._apply_course_overrides(env_idx, first_build=False)
        else:
            super()._assign_courses(env_idx)

    # ---- LOAD-BEARING appo / privileged-critic assert (call after the agent is built) -----------
    def assert_appo_critic(self, algo: str, critic_input_dim: int) -> None:
        """Enforce algo=appo AND that the built critic consumes the GT privileged state (input_dim ==
        state_dim > obs_dim). Delegates to ego_reward.assert_privileged_critic_connected. The trainer
        (peregrine_train_ego) calls this right after the agent is constructed; a symmetric obs-only
        critic (the ppo footgun) raises AssertionError."""
        from ego_reward import assert_privileged_critic_connected
        assert_privileged_critic_connected(algo, critic_input_dim, self.obs_dim, self.state_dim)

    # ---- coarse-map override (human-overridable / deploy designation) -------------------------
    def set_coarse_map(self, sector: Tensor) -> None:
        """Override the prebuilt per-gate sector (N,G,2) or a single (G,2) broadcast across envs.
        Each entry must be in {-1,0,1}. This is the human-overridable / deploy-designation hook."""
        s = sector.to(device=self.device)
        if s.dim() == 2:
            s = s.unsqueeze(0).expand(self.n_envs, -1, -1)
        assert s.shape == self._coarse_map.shape, (s.shape, self._coarse_map.shape)
        self._coarse_map = s.long().clone()

    # ---- estimator reset (cold-init at truth spawn) -------------------------------------------
    def _reset_estimator(self, env_idx) -> None:
        if int(env_idx.numel()) == 0:
            return
        self._estimator.reset_idx(env_idx, self._p[env_idx], self._v[env_idx], self._q[env_idx])

    def reset_idx(self, env_idx):
        super().reset_idx(env_idx)
        if getattr(self, "_ego_on", False) and hasattr(self, "_estimator"):
            self._reset_estimator(env_idx)
            self._prev_q[env_idx] = self._q[env_idx]
            # clear the ONCE-PER-GATE parabola latch for EVERY reset path: step() funnels BOTH
            # terminated and truncated (timeout) envs through reset_idx, so this is the single choke
            # point -- a latch surviving a truncation would silently suppress the NEXT episode's first
            # crossing payment (the stale-latch bug the latch contract calls out).
            self._parabola_paid[env_idx] = False
            if getattr(self, "_use_refined_b", False):
                # rebuild the racing line for the reset envs on their FRESH geometry, THEN re-seed the
                # progress potential on the (post-reset) current segment / line so the first step's
                # (s_curr - s_prev) starts from the true spawn projection (no spurious first-step burst),
                # and clear the banked-progress accumulator.
                self._rebuild_racing_line(env_idx)
                seg_a, seg_b = self._current_segment(env_idx)
                self._seg_s_prev[env_idx] = self._progress_scalar(self._p[env_idx], seg_a, seg_b,
                                                                  env_idx=env_idx)
                self._banked_prog[env_idx] = 0.0
                self._corr_perp_prev[env_idx] = self._perp_scalar(self._p[env_idx], seg_a, seg_b,
                                                                  env_idx=env_idx)

    # ---- current-target gate-centre SEGMENT [prev_center -> target_center] (GT, Z-up) ----------
    def _current_segment(self, env_idx=None):
        """Return (seg_start, seg_end) for the CURRENT target gate of the selected envs (all if None).
        seg_end = current target gate centre; seg_start = the PREVIOUS gate centre, or the SPAWN for
        target gate 0. Pure GT (gate_pos + spawn_pos), Z-up -- the finite segment the progress projects
        onto (perpendicular drift off this segment earns ZERO progress)."""
        if env_idx is None:
            env_idx = self._arange
        tg = self.target_gates.long()[env_idx]
        seg_end = self.gate_pos[env_idx, tg]                              # (M,3) target centre
        prev_idx = (tg - 1)
        seg_start = torch.where(
            (tg == 0).unsqueeze(-1),
            self.spawn_pos[env_idx],                                      # gate 0 -> spawn
            self.gate_pos[env_idx, prev_idx.clamp(min=0)])               # else previous gate centre
        return seg_start, seg_end

    def _rebuild_racing_line(self, env_idx=None) -> None:
        """(Re)build the vector-field racing line from the CURRENT GT geometry (self.spawn_pos/gate_pos/
        gate_yaw). No-op unless ``use_racing_line``. Rebuilds the WHOLE batch (the geometry tensors already
        hold each env's fresh course after the reset's course resample; the query gathers per env), so the
        ``env_idx`` arg is advisory -- it only gates the no-empty-work guard. Cheap (N*G*spp small)."""
        if not self._use_racing_line:
            return
        if env_idx is not None and int(env_idx.numel()) == 0:
            return
        self._racing_line = build_racing_line(self.spawn_pos, self.gate_pos, self.gate_yaw,
                                              samples_per_seg=self._racing_samples_per_seg)

    def _progress_scalar(self, pos, seg_start, seg_end, env_idx=None):
        """The progress POTENTIAL s for these envs.
          * use_racing_line -> arc length of the nearest point on the CURVED head-on racing line (the GVF
            along-track potential; monotone down-course, telescoping -> non-farmable, bounded by the line
            length; NO lag reference to outrun). ``env_idx`` selects the per-env lines.
          * else rw_progress_to_center=False -> segment_arc_position: along-track advance on the current
            gate SEGMENT (perpendicular drift earns ZERO -- the refined-B champion default).
          * else rw_progress_to_center=True -> gate_center_potential: s = -||pos - gate_centre|| (aniso-
            weighted) == the inc7/Swift distance-to-gate homing gradient in every axis.
        Downstream (clip, banked forfeit, area coupling) is identical -- only the potential differs."""
        if self._use_racing_line and self._racing_line_progress and self._racing_line is not None:
            s, _, _, _ = self._racing_line.query(pos, env_idx=env_idx)
            return s
        if self._egorw.progress_to_center:
            return gate_center_potential(pos, seg_end, self._egorw.progress_vert_weight)
        return segment_arc_position(pos, seg_start, seg_end)

    def _perp_scalar(self, pos, seg_start, seg_end, env_idx=None):
        """Cross-track offset for the contouring / centering terms: the CURVED racing line's perpendicular
        distance (the GVF contouring error) when use_racing_line, else the straight current-gate segment
        perp. Orthogonal to _progress_scalar's along-track s (the two decouple -- raising the contouring
        weight strengthens line-pull without stealing forward progress)."""
        if self._use_racing_line and self._racing_line is not None:
            _, perp, _, _ = self._racing_line.query(pos, env_idx=env_idx)
            return perp
        return segment_perp_distance(pos, seg_start, seg_end)

    # ---- EMULATED-CAMERA body->world (nose-first virtual flip; see __init__) --------------------
    def _cam_R_wb(self):
        """Body->world matrix for the emulated CAMERA visibility test. Applies the virtual
        pi-about-body-z flip so the tail-first-flying drone's camera looks along the TRAVEL direction
        (nose-first deploy convention) and sees the gates AHEAD. Only the camera is rotated -- the
        control frame / rel_pos / velocity / rates all use the unflipped self._q."""
        R_wb = quat_xyzw_to_matrix_torch(self._q)
        return R_wb @ self._Rz_cam if self._cam_flip else R_wb

    # ---- STEP the estimator one control step at the CURRENT truth (mutates estimator state) ----
    def _step_estimator(self, prev_q):
        with torch.no_grad():
            cam_R = self._cam_R_wb()
            detectable, _ = gate_detectable(self._p, cam_R, self.gate_pos, self.gate_yaw,
                                            far_cap_m=self._ego_cfg.far_cap_m, is_quat=False)
            # APPARENT projected opening area (normalized, square-on==1), computed with the EMULATED
            # (flipped) camera so the obs matches what the detector sees. Stored (GT, noiseless) for the
            # reward's area-distance coupling; passed to the estimator which noises it for the obs.
            apparent_area = gate_apparent_area(self._p, cam_R, self.gate_pos, self.gate_yaw,
                                               is_quat=False)
            self._apparent_area_gt = apparent_area
            est = self._estimator.step(self._p, self._v, self._q, self._w, float(self.dt),
                                       detectable=detectable, prev_quat=prev_q,
                                       apparent_area=apparent_area)
        self._stepped = True
        return est, detectable

    def _current_detectable(self):
        with torch.no_grad():
            detectable, _ = gate_detectable(self._p, self._cam_R_wb(), self.gate_pos, self.gate_yaw,
                                            far_cap_m=self._ego_cfg.far_cap_m, is_quat=False)
        return detectable

    # ---- observation (ego 26-dim; OFF -> byte-identical inc7) ----------------------------------
    def get_observations(self, with_grad=False):
        if not self._ego_on:
            return super().get_observations(with_grad)
        # get_observations only READS estimator state. On the FIRST obs after a reset (before any step()
        # this episode) the estimator has been cold-reset but not stepped -- step it once here at the
        # spawn truth so the visible gates get their first fix (mirrors the runner's post-reset obs).
        if not self._stepped:
            self._step_estimator(self._prev_q)
            self._prev_q = self._q.clone()
        detectable = self._current_detectable()
        est = self._estimator.estimate()
        obs = ego_actor_obs(est, detectable, self.target_gates, self.last_action[..., 0],
                            self._coarse_map, self.n_gates, obs_coast=self._ego_obs_coast)
        finite = torch.isfinite(obs)
        if not bool(finite.all()):
            obs = torch.where(finite, obs, torch.zeros_like(obs))
        return obs if with_grad else obs.detach()

    # ---- privileged critic state (GT geometry + true vel/att/rates + confidence) ---------------
    def get_state(self, with_grad=False):
        if not self._ego_on:
            return super().get_state(with_grad)
        with torch.no_grad():
            R_wb = quat_xyzw_to_matrix_torch(self._q)
            rel_true = true_rel_pos_body(self.gate_pos, self._p, R_wb)      # (N,G,3) body TRUTH
            vel_body_true = torch.einsum("nji,nj->ni", R_wb, self._v)       # (N,3) body TRUTH
            est = self._estimator.estimate()
            state = ego_critic_state(rel_true, vel_body_true, est.roll_pitch, self._w,
                                     est.confidence, self.target_gates, self.n_gates)
        return state if with_grad else state.detach()

    # ---- step (ego estimator + HARD kill-on-contact; OFF -> byte-identical inc7) ----------------
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        if not self._ego_on:
            return super().step(action, next_obs_before_reset, next_state_before_reset)
        ar, G = self._arange, self.n_gates
        prev_pos = self._p.clone()
        prev_q = self._q.clone()
        self.dynamics.step(action)
        curr_pos = self._p

        # advance the estimator ONCE to the new truth (prev_q is the pre-step attitude for the
        # ego-propagation rotation). The final get_observations() below reads this stepped state.
        self._step_estimator(prev_q)
        self._prev_q = self._q.clone()

        # ===== crossings / terminations / advance: VERBATIM inc7 (frozen contact contract) =====
        rel_prev = world_to_gateframe(prev_pos[:, None, :] - self.gate_pos, self.gate_yaw)
        rel_curr = world_to_gateframe(curr_pos[:, None, :] - self.gate_pos, self.gate_yaw)
        half_in_eff, half_out_eff = self._contact_bands()
        ev = crossing_events(rel_prev, rel_curr, half_in_eff, half_out_eff)
        slab_hit = (slab_frame_hits(rel_prev, rel_curr, half_in_eff, half_out_eff,
                                    self.frame_depth_m) if self.frame_depth_m > 0.0 else None)
        tg = self.target_gates.long()
        fwd_t = ev["fwd"][ar, tg]
        gate_passed = fwd_t & ev["pass_ok"][ar, tg]
        # WIDE-FLYBY -> MISS (refined-B): a lateral flyby that crosses the target gate's along-track
        # plane (fwd OR bwd) without a valid pass and without a frame strike is a MISS -- never silently
        # ignored. The inc7 path only classified the FORWARD off-aperture crossing; wide_flyby_miss ALSO
        # catches a BACKWARD wide crossing (a lateral overshoot re-crossing the plane the wrong way).
        if self._use_refined_b:
            gate_miss = wide_flyby_miss(ev["fwd"][ar, tg], ev["bwd"][ar, tg],
                                        ev["pass_ok"][ar, tg], ev["in_frame"][ar, tg])
        else:
            gate_miss = fwd_t & ~ev["pass_ok"][ar, tg] & ~ev["in_frame"][ar, tg]
        frame_target = (ev["fwd"] | ev["bwd"])[ar, tg] & ev["in_frame"][ar, tg]
        strike_any = (ev["fwd"] | ev["bwd"]) & ev["in_frame"]
        strike_any[ar, tg] = False
        gate_collision = frame_target | strike_any.any(dim=1)
        if slab_hit is not None:
            slab_t = slab_hit[ar, tg]
            gate_passed = gate_passed & ~slab_t
            gate_miss = gate_miss & ~slab_t
            gate_collision = gate_collision | slab_hit.any(dim=1)
        pass_linf = ev["linf"][ar, tg]

        # PRE-ADVANCE current-target segment [prev_center -> target_center] (GT, Z-up). Progress this
        # step is credited along the segment that was CURRENT at the START of the step (the drone moved
        # toward THIS gate). seg_end = gate_pos[tg]; seg_start = spawn (tg==0) else gate_pos[tg-1].
        if self._use_refined_b:
            seg_end_pre = self.gate_pos[ar, tg]                              # (N,3)
            seg_start_pre = torch.where(
                (tg == 0).unsqueeze(-1), self.spawn_pos,
                self.gate_pos[ar, (tg - 1).clamp(min=0)])
            s_curr = self._progress_scalar(curr_pos, seg_start_pre, seg_end_pre)   # (N,)

        is_last = tg == (G - 1)
        newly_finished = gate_passed & is_last & ~self.finished
        advance = gate_passed & ~is_last
        self.target_gates[advance] = self.target_gates[advance] + 1
        self.n_passed_gates[gate_passed] += 1
        self.finished |= newly_finished
        tg_new = self.target_gates.long()
        self.target_pos.copy_(self.gate_pos[ar, tg_new])

        oob_full = ((curr_pos < self.box_min) | (curr_pos > self.box_max)).any(dim=-1)
        # FLOOR CONTACT = a CRASH / DISQUALIFICATION, as expensive as a gate strike (Fengyou 2026-07-07).
        # Diving below the arena floor (Z-up z < box_min_z) is physically a GROUND CONTACT, not merely
        # "off course": in the competition hitting the floor and hitting a gate are BOTH DQs. So fold a
        # below-floor exit into gate_collision -> it costs terminal_base (== gate contact) and is counted
        # in collision_rate (honest: a floor dive IS a crash). ``oob`` then means a LATERAL/CEILING arena
        # exit only (kept distinct so it can later be tuned independently of the lethal floor).
        below_floor = curr_pos[:, 2] < self.box_min[:, 2]
        gate_collision = gate_collision | below_floor
        oob = oob_full & ~below_floor

        # ===== HARD KILL-ON-CONTACT: any gate contact -> done + large negative reward =====
        # A wide flyby (gate_miss) terminates ONLY when miss_terminates (default). When OFF it does not
        # terminate and carries no terminal penalty (miss_term zeroed) -- the drone flies past + must
        # re-approach; CONTACT and OOB always terminate. gate_miss (real) is still used for stats below.
        miss_term = gate_miss if self._miss_terminates else torch.zeros_like(gate_miss)
        terminated = gate_collision | miss_term | oob | self.finished
        truncated = self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        truncated = truncated & ~terminated
        success = self.finished.clone()

        tilt = torch.arccos(tilt_cos_from_quat_xyzw(self._q).clamp(-1.0, 1.0))
        self._peak_tilt = torch.maximum(self._peak_tilt, tilt)
        self._peak_roll = torch.maximum(self._peak_roll, roll_from_quat_xyzw(self._q).abs())
        speed = torch.linalg.norm(self._v, dim=-1)
        self._speed_sum += speed

        time_left_s = (self.max_steps - self.progress).clamp(min=0).float() * self.dt
        a_norm = (action - self._act_lo) / self._act_span
        last_norm = (self.last_action - self._act_lo) / self._act_span

        if self._use_refined_b:
            # ===== REFINED-B (champion-consensus) reward: this generation trains on it =====
            # PROGRESS = clip(s_curr - s_prev) along the PRE-ADVANCE current segment (perpendicular
            # drift earns 0). PASSAGE = (1 - e_lat/w_g_half) L-inf centering, idempotent. TERMINAL =
            # progress-scaled forfeit (dominates the banked progress -> sprint-and-clip is never a
            # positive-return strategy). EXIT-LINE anticipation is on by curriculum (rw_exit>0).
            tilt_cos = tilt_cos_from_quat_xyzw(self._q)
            # exit-line bearing: current->next gate CENTRE (Z-up GT); clamp next at the last gate.
            next_idx = (tg + 1).clamp(max=G - 1)
            curr_center = self.gate_pos[ar, tg]
            next_center = self.gate_pos[ar, next_idx]
            # AREA-DISTANCE coupled progress (Fengyou 2026-07-07): the PRIVILEGED GT APPARENT opening
            # area of the current target gate (normalized projected inner-opening area, [0,1], 1 ==
            # square-on) + GT metres to it. area_true is the SAME quantity the estimator's visible_area obs
            # carries (recalibrated 2026-07-07 from the old |cos| proxy to the true projected area a
            # detector reports), noiseless -- computed with the emulated camera in _step_estimator this
            # step. Progress is credited less for a shallow CLOSE-in approach (exits wide), full for a
            # beeline from far -- distance gates the coupling (see ego_reward.area_distance_*).
            los = curr_center - curr_pos                                     # (N,3) Z-up drone->gate
            dist_to_gate = torch.linalg.norm(los, dim=-1)                    # (N,)
            area_true = self._apparent_area_gt[ar, tg]                       # (N,) in [0,1], square-on==1
            # DENSE lateral centering: perpendicular offset from the PRE-ADVANCE current-target segment
            # (spawn->gate0 or gate[k-1]->gate[k]); the reward pulls this toward 0 -> a CENTRED crossing.
            perp_dist = self._perp_scalar(curr_pos, seg_start_pre, seg_end_pre)   # (N,) seg OR line perp
            # GVF DIRECTION-ALIGNMENT field (tangent + inward-unit at curr_pos) -- only when the alignment
            # reward is ON (rw_align != 0) and the racing line exists; else None (term zeros out).
            line_tangent = line_inward = None
            if (self._use_racing_line and self._racing_line is not None
                    and self._egorw.align != 0.0):
                _, _, line_tangent, line_inward = self._racing_line.query(curr_pos)
            # PERCEPTION reward cue (Fengyou 2026-07-09; None unless rw_perception>0 -> byte-identical off):
            # cos of the angle between the EMULATED camera's optical axis and the drone->current-gate-centre
            # vector, for the Swift/Geles r_perc = perception*exp(-acos(cos)^exp). Uses the SAME flipped
            # camera (self._cam_R_wb) as the detector/visible_area so "point at the gate" matches the FOV.
            cos_view = None
            if self._egorw.perception != 0.0:
                cos_view = gate_center_view_cos(self._p, self._cam_R_wb(), self.gate_pos,
                                                self.gate_yaw, is_quat=False)[ar, tg]   # (N,)
            reward, loss_components, r_prog = compute_ego_reward(
                self._egorw,
                s_curr=s_curr, s_prev=self._seg_s_prev,
                gate_passed=gate_passed, pass_linf=pass_linf,
                w_g_half=self.gate_half_opening_m,
                gate_collision=gate_collision, gate_miss=miss_term, oob=oob,
                banked_progress_return=self._banked_prog,
                newly_finished=newly_finished, time_left_s=time_left_s,
                tilt_cos_r33=tilt_cos, omega=self._w, action_norm=a_norm, last_action_norm=last_norm,
                vel_world=self._v, curr_center=curr_center, next_center=next_center,
                dt=float(self.dt),
                area_true=area_true, dist_to_gate=dist_to_gate, passed_gate_index=tg,
                perp_dist=perp_dist,
                # HOVER-HOLD probe: GT altitude + spawn altitude for the give-up-resistant altitude-hold
                # bonus (OFF unless rw_altitude_hold>0, i.e. only the hover_hold diagnostic stage).
                z=curr_pos[:, 2], z_spawn=self.spawn_pos[:, 2],
                # MPCC CONTOURING: previous-step perp offset for the PBRS contouring potential (OFF unless
                # rw_corridor>0). perp_dist above is the current-step offset from the same segment.
                perp_prev=self._corr_perp_prev,
                # FRAME-MOAT fix: when frame_clip_is_miss, ONLY the floor dive forfeits banked progress
                # (a frame-clip then nets == a wide miss -> no moat around the aperture). Else None ->
                # every contact forfeits (legacy). below_floor is the GT floor-contact mask this step.
                forfeit_mask=(below_floor.to(self._ego_dtype) if self._frame_clip_is_miss else None),
                # GVF direction-alignment field (None unless rw_align>0): reward velocity-direction following
                # the guiding field so a parallel-flying standing offset is still pressured onto the line.
                line_tangent=line_tangent, line_inward=line_inward,
                # SMOOTH PARABOLIC CROSSING (None-safe; active only when rw_parabola_crossing): the L-inf
                # crossing offset + the forward target-plane crossing mask + the floor mask (so the terminal
                # penalty fires on floor+oob only, frame-clip/miss paying the smooth parabola instead).
                cross_offset=pass_linf, crossed=fwd_t, floor_contact=below_floor.to(self._ego_dtype),
                # ONCE-PER-GATE parabola latch: the env-owned per-env bool buffer, mutated IN PLACE by
                # crossing_parabola_reward (marked where crossed) ONLY when rw_parabola_latch is on --
                # with the flag off (default) the reward fn neither reads nor writes it (byte-identical).
                # Cleared below on a target ADVANCE and in reset_idx on every episode reset/truncation.
                parabola_paid=self._parabola_paid,
                # PERCEPTION reward (None unless rw_perception>0): cos(optical-axis, drone->gate-centre).
                cos_view=cos_view)
            # accumulate the (undiscounted) banked progress return for the progress-scaled terminal,
            # then roll the progress potential forward: on an ADVANCE (gate pass) re-seed s_prev onto
            # the NEW current segment (the drone's projection there) so the handoff adds no spurious
            # burst; otherwise carry s_curr as the next step's s_prev (telescoping potential).
            self._banked_prog = self._banked_prog + r_prog.detach()
            # ADVANCE-clear of the parabola latch, AFTER the reward computed on the PRE-advance target:
            # the crossing that advanced the target was latched inside compute_ego_reward this step; the
            # NEW target gate opens a fresh payment window. A wide miss (no advance, miss_terminates=false)
            # deliberately KEEPS the latch -- that is the re-payment-farm defense. Terminating envs
            # (contact/miss/oob/finish + truncation) are cleared in reset_idx instead.
            self._parabola_paid[advance] = False
            new_s_prev = s_curr.detach().clone()
            adv_idx = advance.nonzero().view(-1)
            new_perp_prev = perp_dist.detach().clone()                     # roll the contouring potential
            if adv_idx.numel() > 0:
                seg_a, seg_b = self._current_segment(adv_idx)               # NEW (post-advance) segment
                new_s_prev[adv_idx] = self._progress_scalar(curr_pos[adv_idx], seg_a, seg_b,
                                                            env_idx=adv_idx)
                # re-seed perp_prev onto the NEW segment too (mirror _seg_s_prev) so the handoff adds no
                # spurious contouring burst (the clip band is the backstop if the re-projection jumps).
                # With the GLOBAL racing line s/perp are position-based (continuous across gates), so this
                # re-query returns the same value -- a harmless no-op that keeps the segment path correct.
                new_perp_prev[adv_idx] = self._perp_scalar(curr_pos[adv_idx], seg_a, seg_b,
                                                           env_idx=adv_idx)
            self._seg_s_prev = new_s_prev
            self._corr_perp_prev = new_perp_prev
            loss_components["ego_collision_rate"] = float(gate_collision.float().mean())
            loss_components["banked_prog_mean"] = float(self._banked_prog.mean())
        else:
            # ===== LEGACY inc7 option-B path (A/B fallback; +env.reward_refined_b=false) =====
            gate_pos_t = self.gate_pos[ar, tg_new]
            prev_d2g = torch.linalg.norm(prev_pos - gate_pos_t, dim=-1)
            curr_d2g = torch.linalg.norm(curr_pos - gate_pos_t, dim=-1)
            reward, loss_components = compute_reward_terms(
                self.rw, prev_d2g=prev_d2g, curr_d2g=curr_d2g, gate_passed=gate_passed,
                gate_collision=gate_collision, gate_miss=gate_miss, oob=oob,
                newly_finished=newly_finished, time_left_s=time_left_s, quat_xyzw=self._q,
                omega=self._w, action_norm=a_norm, last_action_norm=last_norm)
            reward = apply_contact_kill(reward, gate_collision, self._ego_contact_penalty)
            loss_components["ego_collision_rate"] = float(gate_collision.float().mean())

        # diffaero's runner.py progress bar reads env_info['loss_components']['total_loss'] EVERY step
        # (utils/runner.py:136). The base compute_reward_terms emits it (peregrine_racing.py:396) but the
        # refined-B compute_ego_reward does not, so set it here for BOTH paths (base convention: the
        # negative mean reward). Idempotent on the legacy path (overwrites the same value).
        loss_components["total_loss"] = float(-reward.mean().item())

        loss = (-reward).detach()
        reward = reward.detach()
        self.last_action.copy_(action.detach())

        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)

        # ===================== BOX-EXIT CLASSIFICATION (Fengyou 2026-07-07 diagnostic) =====================
        # Partition every terminating/truncating episode by WHERE its path leaves the arena box (self.box_min/
        # max is the diagnostic bounding box) or HOW it ends -- logged as metrics/exit_* from the STOCHASTIC
        # training rollouts (the deterministic render proved untrustworthy). Mutually EXCLUSIVE by priority so
        # the classes sum to ~1. PURE DIAGNOSTIC: reads the already-final terminal flags + curr_pos; touches
        # NEITHER reward nor termination. Also logs cross_offset_m = the L-inf off-centre distance for EVERY
        # target-plane crossing (pass + frame + wide miss), so we see HOW FAR OFF the crossings actually land.
        with torch.no_grad():
            cp = curr_pos
            bx_lo, bx_hi = self.box_min, self.box_max
            _asg = torch.zeros(self.n_envs, dtype=torch.bool, device=cp.device)

            def _take(mask):                                                # first-come priority, exclusive
                m = mask & ~_asg
                _asg[m] = True
                return m
            c_thread = _take(success)                                       # threaded the gate (WIN)
            c_floor = _take(below_floor)                                    # dived below the floor (contact)
            c_frame = _take(gate_collision)                                 # hit the gate FRAME (contact, non-floor)
            c_pmiss = _take(gate_miss)                                      # crossed the gate PLANE wide (in-bounds)
            c_ceil = _take(oob & (cp[:, 2] > bx_hi[:, 2]))                 # climbed out the CEILING
            c_side = _take(oob & ((cp[:, 1] < bx_lo[:, 1]) | (cp[:, 1] > bx_hi[:, 1])))   # ran out a SIDE wall
            c_back = _take(oob & (cp[:, 0] < bx_lo[:, 0]))                 # flew BACKWARD out the back wall
            c_front = _take(oob & (cp[:, 0] > bx_hi[:, 0]))               # overshot out the FRONT wall
            c_time = _take(truncated)                                       # hovered in-box to TIMEOUT
            crossed_tg = (ev["fwd"] | ev["bwd"])[ar, tg]                    # any target-plane crossing this step

        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "loss_components": loss_components,
            "stats_raw": {
                "success_rate": success[reset].float(),
                "survive_rate": truncated[reset].float(),
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "n_passed_gates": self.n_passed_gates[reset].float(),
                "collision_rate": gate_collision[reset].float(),
                "miss_rate": gate_miss[reset].float(),
                "oob_rate": oob[reset].float(),
                # box-exit breakdown (mutually exclusive, sum ~1) -> metrics/exit_*
                "exit_thread": c_thread[reset].float(),
                "exit_floor": c_floor[reset].float(),
                "exit_frame": c_frame[reset].float(),
                "exit_plane_miss": c_pmiss[reset].float(),
                "exit_ceiling": c_ceil[reset].float(),
                "exit_side": c_side[reset].float(),
                "exit_back": c_back[reset].float(),
                "exit_front": c_front[reset].float(),
                "exit_timeout": c_time[reset].float(),
                # how far off-centre every target-plane crossing lands (pass + frame + wide miss), L-inf m
                "cross_offset_m": pass_linf[crossed_tg],
                # HOVER-HOLD / altitude read: |z - z_spawn| (m) at episode end -> "sub-metre hold?" for the
                # hover_hold probe, and the vertical error of every terminating episode generally.
                "alt_err_m": (cp[:, 2] - self.spawn_pos[:, 2]).abs()[reset],
                "peak_tilt_deg": torch.rad2deg(self._peak_tilt)[reset],
                "peak_roll_deg": torch.rad2deg(self._peak_roll)[reset],
                "mean_speed": (self._speed_sum / self.progress.clamp(min=1).float())[reset],
                "finish_time_s": ((self.progress.clone() - 1).float() * self.dt)[newly_finished],
                "pass_offset_m": pass_linf[gate_passed],
            },
        }
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (loss, reward), terminated, extra
