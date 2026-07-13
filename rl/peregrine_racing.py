"""PeregrineRacing -- DiffAero racing env on OUR courses (given pose, no vision). S1.4 redesign.

Registered into ``diffaero.env.ENV_ALIAS["peregrine_racing"]`` by the launcher
(``peregrine_train_racing``) so the pristine diffaero clone is never edited.

================================================================================================
S1.4 ENV-COHERENCE REDESIGN (session S15, 2026-06-10) -- the audit and the design, in full.
The reward and the termination are designed TOGETHER; every term documents its formula, weight,
units, what behavior it buys, and what breaks if it is removed. This docstring is a deliverable.
================================================================================================

AUDITED INCOHERENCES in the S1.3 env (parent ``Racing`` + the old subclass), each now fixed:

C1. OOB ESCAPE WAS FREE, COLLISION WAS PUNISHED. The parent classifies out-of-bounds as
    TRUNCATION; diffaero's PPO bootstraps V(s') on truncation (PPO.py: ``next_done = terminated``
    only, GAE uses V(next_obs_before_reset)) and the quadrotor reward has NO oob term (the
    ``oob_loss`` computed in racing.py is applied only in the pointmass branch). The OOB box is
    also INVISIBLE in the gate-relative obs. Net effect: near a risky gate, "fly past the gate and
    out of the box" returned bootstrapped continuation value with zero penalty, while attempting
    the aperture risked -10 and a zeroed future -- the env was TEACHING escape over threading.
    FIX: OOB is now a TERMINATION with a penalty equal to a frame strike (both are a lost run in
    the live sim), logged separately (``oob_rate``).

C2. ONE COLLISION CLASS FOR THREE PHYSICALLY DIFFERENT EVENTS. The old ``is_passed`` called ANY
    off-aperture plane crossing "collision": clipping the 2.72 m physical frame (live: tumble --
    the S1.2 crash was a gate-post strike at -63 rad/s), missing wide through free air (live: the
    run is invalid but flight continues), and crossing 5 m above the gate (live: nothing there).
    Worse, the aperture test used the POST-crossing position, so at racing speed (~0.5 m/step at
    30 Hz) a clean center pass could read as off-aperture and vice versa. And lateral/backward
    post strikes never registered at all (plane-crossing-only detection).
    FIX: crossings are classified at the INTERPOLATED plane-crossing point against the twin gate
    geometry: |y|,|z| L-inf < 0.75 m (1.5 m inner opening) = PASS; in (0.75, 1.36] m (2.72 m outer
    frame) = FRAME COLLISION (terminal, big penalty); > 1.36 m = CLEAN MISS (terminal, smaller
    penalty -- deliberately conservative: live a wide miss is recoverable by circling back, but
    training must learn clean lines, so we terminate; documented trade-off). Frame strikes are
    detected on ALL gates (both crossing directions), not just the target -- gates are physical
    everywhere in the live sim. SCOPE: detection stays crossing-based -- a pure lateral graze
    (touching a frame while moving parallel to its plane, no x sign change) remains unmodeled;
    that needs a capsule-vs-frame contact model and is out of scope here. The dominant
    racing-line failure modes (off-aperture crossings, the S1.2 through-the-post line) are
    covered.

C3. ATTITUDE PENALTY TAXED ALL TILT. S1.3's ``attitude=2.0`` penalizes roll^2+pitch^2 -- including
    the 30-60 deg of tilt a racing quad NEEDS to accelerate and corner. It bought peak roll 65 deg
    but fights speed everywhere.
    FIX: a HINGE tilt penalty, zero inside a 60 deg free cone, quadratic beyond it (see term R4).
    Explicitly a SMOOTHNESS/VQ2-style + stay-in-the-well-modeled-envelope term; the characterize
    sweep (510da24) proved there is NO tilt anomaly in the sim, so this is NOT anomaly avoidance
    and there is deliberately NO tilt-threshold termination.

C4. "JERK" WAS A BODY-RATE TAX. The parent's quadrotor ``jerk_loss`` is ||omega|| -- it taxes
    cornering itself, not roughness. With the super-rate plant the policy legitimately commands
    ~11 rad/s transients.
    FIX: a true command-smoothness term on the ACTION RATE (R5) + a small residual ||omega||
    style term (R6) at half the old weight.

C5. NO TIME PRESSURE BEYOND THE DISCOUNT. Nothing penalized loitering; a policy hovering 1 cm
    before the finish plane lost nothing (timeout truncation bootstraps, see C1).
    FIX: an explicit small per-step time penalty (R3) + a finish-time bonus (T4) -- the moderate
    speed shaping VQ2's ranking needs, sized so transfer/validity still dominate (the aggressive
    speed iteration comes later, against a live-validated baseline).

C6. STATS HID THE FAILURE MODES. ``survive_rate`` counted OOB escapes as survival; collisions,
    misses, OOB and timeouts were indistinguishable.
    FIX: per-episode outcome stats: finish / frame-collision / miss / oob / timeout rates, plus
    peak tilt, mean speed, pass offset (L-inf at the crossing point), and finish time.

Also fixed while in here: ``get_state`` (used by asymmetric critics + check_dims) inherited the
parent's looping ``% n_gates`` lookahead -- physically meaningless teleports past the finish on a
point-to-point course, and shape-incompatible with per-env courses. Now clamped + gathered.

------------------------------------------------------------------------------------------------
THE REWARD/TERMINATION CONTRACT (all weights cfg-overridable as ``+env.rw_<name>=...``; defaults
in ``RewardWeights``). Per-step reward r_t =

    R1  + rw_progress   * (d2g_prev - d2g_curr)      [10 / m]
    R2  + rw_passage    * 1[gate passed]             [+10]
    T4  + (rw_finish + rw_finish_time * t_left_s) * 1[finished]   [+20 + 1.0/s]
    T1  - rw_collision  * 1[frame strike, any gate]  [-25, TERMINAL]
    T2  - rw_miss       * 1[clean miss, target]      [-15, TERMINAL]
    T3  - rw_oob        * 1[out of bounds]           [-25, TERMINAL]
    R3  - rw_time       * 1                          [-0.02 / step  (~0.6/s @ 30 Hz)]
    R4  - rw_tilt       * relu(cos(rw_tilt_free) - R33)^2          [4.0; free cone 60 deg]
    R5  - rw_dact       * ||a_t - a_{t-1}||^2        [0.25; normalized action units^2]
    R6  - rw_rate       * ||omega||                  [0.05 / (rad/s)]
    R7  - rw_corner     * |a_thr - 0.5| * ||2(a_rate - 0.5)||   [0.0 = OFF; S17 mixer-corner tax]

R1 PROGRESS (dense): Euclidean distance-to-gate-CENTER delta, measured against the (possibly just
   advanced) target gate for BOTH prev and curr (parent convention -- no spike at passage).
   Units: reward per meter of approach. Buys: the only dense gradient toward the task; PPO never
   discovers the first passage from sparse terms alone at this horizon. Removed => no learning.
   Known bias (accepted, documented): radial-to-center shaping fights wide racing lines; the
   TOGT-reference upgrade (stack-review #2) replaces it in a later session.
R2 PASSAGE (+10, at the interpolated crossing): makes the gate plane itself worth crossing.
   Removed => near a gate the plane offers risk (T1/T2 nearby) with no differential payoff;
   policies hover-stall short of hard gates.
T4 FINISH (+20) + FINISH-TIME (+1.0 per second left on the clock): the finish is terminal, so
   without a bonus the last plane is just where reward STOPS (value cliff to 0) -- a mild
   anti-finish gradient. The time term is the explicit "faster lap" pressure (VQ2 ranks on time).
   Removed => laps drift slow; only the discount factor pushes speed, weakly and implicitly.
T1 FRAME COLLISION (-25, terminal, any gate, either crossing direction, at the interpolated
   crossing point): the live sim's real failure mode (S1.2: gate-post strike, -63 rad/s tumble).
   Sized 2.5x the passage bonus and ~3.5x the largest plausible single-step progress (+7 at
   0.7 m/step) so "clip the frame to grab the bonus sooner" is never positive-EV, and so early
   in training (noisy values) the margin is wide. Removed/undersized => progress + passage can
   bribe through posts; termination alone is NOT a penalty under PPO (V(crash)=0 can beat a
   locally-bad continuation, which is exactly incoherence C1).
T2 CLEAN MISS (-15, terminal, target gate only): crossing the target plane beyond the 1.36 m
   physical frame. Less than T1 (live: no tumble, just an invalid run) but MORE than the +10
   passage so deliberately skipping a hard gate never pays. Removed => missing terminates at 0,
   so for a risky gate the policy rationally prefers a free miss over a -25-risk attempt --
   institutionalized gate-skipping.
T3 OOB (-25, terminal): closes C1. The box is generous (course bbox + spawn corridor + 15/12 m
   margins) so only genuinely lost drones hit it. Equal to T1: both are a lost run live.
   Removed => bootstrapped free-escape returns (C1).
R3 TIME (-0.02/step): the explicit cost of existing. SIZING COHERENCE: total over a full 40 s
   episode = -24, comparable to ONE terminal penalty -- but every failure mode is ALREADY
   terminal-with-penalty <= -15 and timeout TRUNCATES (bootstraps V, no penalty), so ending early
   never beats staying alive productively, and hovering forever is strictly dominated. Removed =>
   loitering near hard gates is free; episodes fill to timeout and dilute the batch with idle
   states.
R4 TILT HINGE (zero below 60 deg of total tilt; R33 = world-z component of body-z = 1-2(qx^2+qy^2)
   from the XYZW quat): at 80 deg costs 0.43/step, at the S1.2 backflip's 104 deg costs 2.2/step,
   at inversion 9/step. Buys: kills the backflip-diver STYLE (in-twin-optimal, live-divergent)
   without taxing the racing tilt envelope the task needs. Removed => the S1.2 style returns.
   NOT a termination, NOT anomaly avoidance (no anomaly exists -- sweep 510da24).
R5 ACTION-RATE (||Delta action||^2 in per-axis span-normalized units, full-scale flip = 1/axis):
   command smoothness. Buys: keeps the command stream inside the measured static-map regime
   (large per-step command steps excite the slew-limit corner where the twin is least faithful),
   and live actuator-friendliness (VQ2 style). Removed => bang-bang CTBR chatter, the regime
   where transfer is weakest.
R6 BODY-RATE (0.05*||omega||): residual style term at HALF the parent's old 0.1 -- the super-rate
   plant legitimately uses ~11 rad/s transients and yaw is the worst-modeled axis, so we tax
   sustained tumbling-style rates lightly without fighting cornering. Removed => mostly fine;
   kept as a cheap regularizer against rate-riding solutions.
R7 MIXER-CORNER (S17, default 0 = OFF): |a_thr - 0.5| (span units: 0.5 at EITHER thrust rail,
   ~0.23 at hover) x the span-normalized commanded-rate magnitude. Taxes exactly the two motor-
   mixer rails -- (thr~0 x high rate) = parasitic lift, (thr~1 x rate) = authority/thrust sag,
   the corners that killed the inc4/inc5 live transfers -- WITHOUT suppressing mid-range thrust
   corrections (the blunt R5 alternative taxes every fast correction equally). A/B'd against
   raised R5 in the inc6 round-1 sweep; on COMMANDED actions (like R5), so it shapes style even
   where the mixer-ON plant already prices the realized physics.

TERMINATION SET (terminal, PPO sees done=1, no bootstrap): frame collision (T1) | clean miss (T2)
| OOB (T3) | finished (T4). TRUNCATION SET (PPO bootstraps V(s')): timeout, GUI reset. This is
the coherence core: every BAD absorbing event carries an explicit penalty AND kills the future;
the only neutral exits are non-events (timeout) that bootstrap honestly.

LOSS OUTPUT: this env is PPO-ONLY. ``loss`` is returned as ``-reward`` (detached) to keep the
diffaero runner's logging alive; do NOT train a BPTT algorithm (SHAC/APG) against it.

------------------------------------------------------------------------------------------------
INC7 CONTACT-TRUE GEOMETRY (training doctrine 2026-06-12 Sec 2; defaults OFF = bit-identical
legacy). The S1.4 contact model collides a POINT with a PLANE band; the live sim collides the
drone's BODY (rotor halo ~0.3 m) with a VOLUMETRIC frame -- live standing crashes terminated at
L-inf 0.37-0.49 m, positions the 0.75 point-mass model scores as comfortable passes, and strikes
occur up to 0.5 m BEFORE the plane on slope-0.45 approaches. Two opt-in env features close the
fiction (reward UNTOUCHED -- margin comes from the world model, never from reward shaping):

  ``+env.body_radius_lo`` / ``+env.body_radius_hi`` -- per-env body radius r ~ U[lo, hi],
     resampled at every reset (the halo is not precisely known and is unobservable, so the
     policy trains to the sampled worst case). Pass band L-inf < 0.75 - r; frame band
     [0.75 - r, 1.36 + r]. Doctrine band [0.28, 0.38] m brackets the measured live contact
     offsets (corner-pass probe contact at 0.60; live crashes 0.37-0.49 on steep descents).
  ``+env.frame_depth_m`` -- the frame band becomes MATERIAL over gate-frame |x| <= depth
     (``slab_frame_hits``: exact segment-vs-volumetric-frame classification). Catches the
     "hit the frame before the plane" class invisible to any plane-crossing test. A crossing
     that threads the (inflated) pass band but touches material inside the slab is a
     COLLISION, not a pass. Doctrine value 0.30 m.

With both at 0 (default) the event set, reward stream and RNG draw sequence are bit-identical
to the pre-inc7 env (slab test skipped entirely; bands stay python floats; no extra RNG).
New diagnostics: ``slab_collision_rate`` (volumetric strikes) and ``pass_margin_m``
(contact-true crossing margin (0.75 - r) - L-inf -- the gauntlet tail metric).

------------------------------------------------------------------------------------------------
PROCEDURAL COURSES (S1.4, stack-review meta-gap #1): ``+env.course_mode=random`` trains on
per-env sampled 6-gate courses (``peregrine_course.sample_courses``, VQ1-derived ranges, resampled
per env at every reset); ``course_mode=vq1`` (default) broadcasts the fixed VQ1 course (the
HELD-OUT eval). Obs are already gate-relative/translation-invariant, so only the training
distribution changes. Course tensors are per-env throughout: gate_pos (N,G,3), gate_yaw (N,G).

SPAWNS (per env, at reset): with prob ``standing_start_frac`` a STANDING START (the course's pad:
23.3 m-class up-course of gate 0, at rest, tilted -17.8 deg pitch, TAIL-FIRST body yaw =
gate0_yaw + pi -- the VQ1 deployment convention behind fly_rl.py's virtual flip; on VQ1 this
reproduces the measured spawn pose); otherwise 1 m up-course of a random target gate, at rest,
tail-first w.r.t. that gate (on VQ1 with gate yaws = pi this equals the parent's identity-attitude
spawn EXACTLY, so the S1.3 obs distribution is preserved). Both get small pose jitter.

WHAT IS DELIBERATELY UNCHANGED (frozen deployment contract, fly_rl.py): obs layout (17 dims,
same order/frames), action semantics (4-dim [normed_thrust, rates FLU], tanh+rescale), dt, the
[-1]-wrap convention in gate_rel_pos (index 0 is never consumed: next_gate_idx is clamped >= 1),
gate-frame conventions. A policy trained here deploys through the existing fly_rl.py unchanged
(only the rescale bound for thrust follows the training cfg: 3.765 for S1.3+).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields
from typing import Dict, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None
    Tensor = "Tensor"                   # type: ignore

try:                                    # diffaero + pytorch3d only exist on the training cluster;
    import pytorch3d.transforms as T    # the pure helpers below stay importable/testable anywhere.
    from diffaero.env.racing import Racing, get_gate_rotmat_w2g
    from diffaero.utils.math import mvp
    _HAVE_DIFFAERO = True
except Exception:                       # pragma: no cover
    Racing = object
    _HAVE_DIFFAERO = False

_COURSE_JSON = os.path.join(os.path.dirname(__file__), "peregrine_course_diffaero.json")


# ================================================================================================
# Pure helpers (torch-only, no diffaero) -- unit-tested on the laptop (tests/test_peregrine_racing_core.py)
# ================================================================================================

def world_to_gateframe(d: Tensor, yaw: Tensor) -> Tensor:
    """Rotate world-frame deltas ``d`` (..., 3) into gate frames given gate ``yaw`` (...).
    Gate frame: +x = exit/down-course direction = world [cos(yaw), sin(yaw), 0]; z stays world z.
    Identical math to diffaero's ``get_gate_rotmat_w2g`` (rows [c,s,0; -s,c,0; 0,0,1])."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    return torch.stack([x, y, d[..., 2]], dim=-1)


def crossing_events(prev_rel: Tensor, curr_rel: Tensor, half_inner: float, half_outer: float):
    """Classify gate-plane crossings at the INTERPOLATED crossing point.

    ``prev_rel``/``curr_rel``: positions in the gate frame, shape (..., 3); the gate plane is x=0.
    Returns dict of bool tensors shaped (...):
      fwd      -- crossed the plane in the exit direction (prev x<0 -> curr x>=0)
      bwd      -- crossed backward
      pass_ok  -- crossing point inside the 1.5 m inner opening (L-inf < half_inner)
      in_frame -- crossing point in the physical frame band (half_inner <= L-inf <= half_outer)
      linf     -- float tensor: L-inf off-axis distance at the crossing point (junk where no cross)
    The interpolation matters at racing speed: ~0.5 m/step at 30 Hz is most of the aperture.
    ``half_inner``/``half_outer`` may be floats (legacy) or tensors broadcastable against the
    crossing dims -- the INC7 per-env body-radius-inflated bands."""
    px, cx = prev_rel[..., 0], curr_rel[..., 0]
    fwd = (px < 0) & (cx >= 0)
    bwd = (px > 0) & (cx <= 0)
    crossed = fwd | bwd
    denom = (cx - px)
    f = torch.where(crossed, -px / torch.where(denom.abs() < 1e-9,
                                               torch.full_like(denom, 1e-9), denom),
                    torch.zeros_like(denom))
    y = prev_rel[..., 1] + f * (curr_rel[..., 1] - prev_rel[..., 1])
    z = prev_rel[..., 2] + f * (curr_rel[..., 2] - prev_rel[..., 2])
    linf = torch.maximum(y.abs(), z.abs())
    pass_ok = crossed & (linf < half_inner)
    in_frame = crossed & (linf >= half_inner) & (linf <= half_outer)
    return {"fwd": fwd, "bwd": bwd, "pass_ok": pass_ok, "in_frame": in_frame, "linf": linf}


def slab_frame_hits(prev_rel: Tensor, curr_rel: Tensor, half_inner, half_outer,
                    depth: float) -> Tensor:
    """Segment-vs-volumetric-frame test (INC7 contact-true geometry, doctrine Sec 2).

    The gate frame is MATERIAL over the slab |x| <= depth wherever the in-plane L-inf radius
    lies in the frame band [half_inner, half_outer] (gate frame; the plane is x=0). Classifies
    the straight motion segment ``prev_rel`` -> ``curr_rel`` ((..., 3)) EXACTLY:

      * in-slab parameter interval [t0, t1] = {t in [0, 1] : |x(t)| <= depth};
      * on it linf(t) = max(|y(t)|, |z(t)|) is CONVEX piecewise-linear, so its range over the
        interval is [min over the <= 6 derivative breakpoints/endpoints, max of the endpoint
        values] (breakpoints: y=0, z=0, y=+-z; all closed-form, clamped into the interval);
      * the segment touches material iff that range intersects [half_inner, half_outer].

    ``half_inner``/``half_outer`` may be floats or tensors broadcastable against the leading
    dims (the per-env inflated bands). At depth=0 this degenerates to the plane-band test
    (callers skip it then -- the legacy crossing classification already covers x=0). Returns
    bool (...)."""
    px, py, pz = prev_rel[..., 0], prev_rel[..., 1], prev_rel[..., 2]
    dx = curr_rel[..., 0] - px
    dy = curr_rel[..., 1] - py
    dz = curr_rel[..., 2] - pz
    zero, one = torch.zeros_like(px), torch.ones_like(px)

    def _safe_div(num, den):
        return num / torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)

    # in-slab interval [t0, t1] intersected with [0, 1]; empty <=> t1 < t0. The x-parallel
    # branch (dx ~ 0) is inside-for-all-t or never (marker t1 = -1 < t0 = 0).
    ta = _safe_div(-depth - px, dx)
    tb = _safe_div(depth - px, dx)
    parallel = dx.abs() < 1e-12
    t0 = torch.where(parallel, zero, torch.maximum(torch.minimum(ta, tb), zero))
    t1 = torch.where(parallel, torch.where(px.abs() <= depth, one, -one),
                     torch.minimum(torch.maximum(ta, tb), one))
    valid = t1 >= t0

    def _linf(t):
        return torch.maximum((py + t * dy).abs(), (pz + t * dz).abs())

    lmin = torch.minimum(_linf(t0), _linf(t1))
    for cand in (_safe_div(-py, dy), _safe_div(-pz, dz),
                 _safe_div(pz - py, dy - dz), _safe_div(-(py + pz), dy + dz)):
        lmin = torch.minimum(lmin, _linf(torch.clamp(cand, min=t0, max=t1)))
    lmax = torch.maximum(_linf(t0), _linf(t1))
    return valid & (lmin <= half_outer) & (lmax >= half_inner)


def tilt_cos_from_quat_xyzw(q: Tensor) -> Tensor:
    """cos(total tilt) = world-z component of body-z = R33 = 1 - 2(qx^2 + qy^2). XYZW quats."""
    return 1.0 - 2.0 * (q[..., 0] ** 2 + q[..., 1] ** 2)


def roll_from_quat_xyzw(q: Tensor) -> Tensor:
    """ZYX Euler roll = atan2(R32, R33) = atan2(2(qy*qz + qw*qx), 1 - 2(qx^2+qy^2)). XYZW quats.
    Matches pytorch3d's matrix_to_euler_angles(..., "ZYX") last angle -- the S1.3 peak-roll
    acceptance metric, tracked HERE (pre-reset) so the terminal/crash pose is included."""
    return torch.atan2(2.0 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0]),
                       tilt_cos_from_quat_xyzw(q))


@dataclass
class RewardWeights:
    """All S1.4 reward/termination weights (see module docstring for the per-term contract).
    Override any of them from hydra with ``+env.rw_<field>=<value>``."""
    progress: float = 10.0       # R1, per meter of approach to the target gate center
    passage: float = 10.0        # R2, per gate passed
    finish: float = 20.0         # T4, at the finish crossing
    finish_time: float = 1.0     # T4, per second left on the episode clock at finish
    time: float = 0.02           # R3, per step
    collision: float = 25.0      # T1, frame strike (terminal)
    miss: float = 15.0           # T2, clean miss of the target plane (terminal)
    oob: float = 25.0            # T3, out of bounds (terminal)
    tilt: float = 4.0            # R4, on relu(cos(tilt_free) - R33)^2
    tilt_free_rad: float = 1.0471976   # R4 free cone half-angle: 60 deg
    dact: float = 0.25           # R5, on ||Delta action||^2 (span-normalized units)
    rate: float = 0.05           # R6, on ||omega|| (rad/s)
    corner: float = 0.0          # R7 (S17), on |a_thr - 0.5| * ||2(a_rate - 0.5)||; 0 = OFF

    @classmethod
    def from_cfg(cls, cfg) -> "RewardWeights":
        kw = {}
        for f in fields(cls):
            kw[f.name] = float(getattr(cfg, f"rw_{f.name}", f.default))
        # honor the legacy S1.3 cfg names if present (back-compat with old sbatch files)
        kw["passage"] = float(getattr(cfg, "passage_bonus", kw["passage"]))
        kw["finish"] = float(getattr(cfg, "finish_bonus", kw["finish"]))
        return cls(**kw)


def compute_reward_terms(w: RewardWeights, *, prev_d2g, curr_d2g, gate_passed, gate_collision,
                         gate_miss, oob, newly_finished, time_left_s, quat_xyzw, omega,
                         action_norm, last_action_norm):
    """The S1.4 reward, as a pure function of event/state tensors (all shaped (N,) or (N,k)).
    ``action_norm``/``last_action_norm`` are span-normalized to [0,1] per axis. Returns
    (reward (N,), components dict of floats for logging)."""
    progress = prev_d2g - curr_d2g
    tilt_pen = torch.relu(math.cos(w.tilt_free_rad) - tilt_cos_from_quat_xyzw(quat_xyzw)) ** 2
    dact = ((action_norm - last_action_norm) ** 2).sum(dim=-1)
    rate_mag = torch.linalg.norm(omega, dim=-1)
    # R7 (S17): mixer-corner tax on the COMMANDED action -- |thr - mid| (span units; 0.5 at
    # either thrust rail) x commanded-rate magnitude (span units; 1 per railed axis).
    corner = (action_norm[..., 0] - 0.5).abs() * torch.linalg.norm(
        2.0 * (action_norm[..., 1:4] - 0.5), dim=-1)
    fin = newly_finished.float()
    reward = (
        w.progress * progress
        + w.passage * gate_passed.float()
        + (w.finish + w.finish_time * time_left_s) * fin
        - w.collision * gate_collision.float()
        - w.miss * gate_miss.float()
        - w.oob * oob.float()
        - w.time
        - w.tilt * tilt_pen
        - w.dact * dact
        - w.rate * rate_mag
        - w.corner * corner
    )
    components = {
        "progress_loss": -progress.mean().item(),
        "tilt_pen": tilt_pen.mean().item(),
        "dact_pen": dact.mean().item(),
        "corner_pen": corner.mean().item(),
        "rate_pen": rate_mag.mean().item(),
        "collision_loss": gate_collision.float().mean().item(),
        "miss_loss": gate_miss.float().mean().item(),
        "oob_loss": oob.float().mean().item(),
        "total_reward": reward.mean().item(),
        "total_loss": -reward.mean().item(),
    }
    return reward, components


def rel_tables(gate_pos: Tensor, gate_yaw: Tensor) -> Tuple[Tensor, Tensor]:
    """Per-course next-gate lookahead tables, (N,G,3)/(N,G) -> (N,G,3), (N,G).
    ``gate_rel_pos[:, i]`` = gate i's position in gate (i-1)'s frame; index 0 wraps to the last
    gate (the parent's Python ``[i-1]`` convention, mirrored by fly_rl.py's precomputed tables;
    index 0 is never consumed by the obs because next_gate_idx is clamped >= 1)."""
    prev_pos = torch.roll(gate_pos, shifts=1, dims=1)
    prev_yaw = torch.roll(gate_yaw, shifts=1, dims=1)
    rel = world_to_gateframe(gate_pos - prev_pos, prev_yaw)
    dy = gate_yaw - prev_yaw
    yaw_rel = torch.atan2(torch.sin(dy), torch.cos(dy))
    return rel, yaw_rel


def quat_xyzw_from_yaw_pitch(yaw: Tensor, pitch: Tensor) -> Tensor:
    """Body-to-world quaternion (XYZW, Hamilton, matches pytorch3d after the roll) for
    R = Rz(yaw) @ Ry(pitch), roll = 0. Built by hand so the helpers stay diffaero-free."""
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    # q = qz (x) qy  (wxyz): [cy*cp, -sy*sp, cy*sp, sy*cp]
    w = cy * cp
    x = -sy * sp
    y = cy * sp
    z = sy * cp
    return torch.stack([x, y, z, w], dim=-1)


def quat_xyzw_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product on XYZW quaternions (a (x) b)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_xyzw_from_axis_angle(rotvec: Tensor) -> Tensor:
    """Axis-angle vector (..., 3) -> XYZW quaternion (small-angle-safe)."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * theta
    w = torch.cos(half)
    scale = 0.5 * torch.sinc(half / math.pi)       # sin(half)/theta, robust at 0
    return torch.cat([scale * rotvec, w], dim=-1)


def spawn_velocity_toward_gate(to_gate: Tensor, sel_draw: Tensor, mag_draw: Tensor,
                               spawn_vel_frac: float, spawn_vel_max: float) -> Tensor:
    """SPAWN-VELOCITY RANDOMIZATION (anti-velocity-runaway 2026-07-13; default OFF -> all-zeros ==
    byte-identical at-rest spawn).

    Give a FRACTION of the reset envs an initial WORLD-frame velocity directed at their TARGET gate,
    magnitude U[0, spawn_vel_max). The proven _pef path spawns every episode AT REST (spawn_vel_frac=0),
    and an episode is too short for the policy to accelerate into the ~12 m/s runaway regime on its own,
    so it never experiences (and never learns to respect) that regime; a fast spawn EXPOSES it -- pairs
    with the velocity-cap reward.

        moving = sel_draw < spawn_vel_frac                     # which envs get a moving start
        mag    = clip(mag_draw, 0, 1) * spawn_vel_max          # U[0, spawn_vel_max)
        dir    = normalize(to_gate)                            # spawn -> target gate centre (world)
        v0     = dir * mag  on moving envs, ZERO elsewhere

    ``to_gate`` (N,3) = the world spawn->target-gate-centre vector (== gate0 - spawn for the standing-start
    _pef path, so the velocity points at the FIRST gate); ``sel_draw`` / ``mag_draw`` (N,) are U[0,1) draws.
    A degenerate zero-length ``to_gate`` (spawn ON the gate) yields ZERO velocity for that env (safe
    normalize). ``spawn_vel_frac`` <= 0 -> the WHOLE tensor is zeros (no env moves) so the caller leaves the
    spawn velocity at 0 -- byte-identical. Returns v0 (N,3) world-frame velocity."""
    assert torch is not None
    if spawn_vel_frac <= 0.0:
        return torch.zeros_like(to_gate)
    moving = (sel_draw < float(spawn_vel_frac)).to(to_gate.dtype)            # (N,)
    norm = torch.linalg.norm(to_gate, dim=-1, keepdim=True)                  # (N,1)
    dirn = torch.where(norm > 1e-6, to_gate / norm.clamp(min=1e-6),
                       torch.zeros_like(to_gate))                            # unit (0 where degenerate)
    mag = mag_draw.clamp(0.0, 1.0) * float(spawn_vel_max) * moving           # (N,) 0 where not moving
    return dirn * mag.unsqueeze(-1)                                          # (N,3)


# ================================================================================================
# FLOOR-AT-SPAWN knob resolution (A1 fix 2026-07-10) -- PURE (getattr-only), laptop-testable.
# ================================================================================================
def resolve_floor_at_spawn(cfg):
    """Resolve ``+env.floor_at_spawn`` / ``+env.floor_clearance_m`` -> (on: bool, clearance_m: float).

    A1 ROOT CAUSE (2026-07-10): the deployed ego policy's trained OPENER is a gravity dive because
    training's lethal floor is the OOB bbox bottom at spawn_z - 12 m (the _update_boxes z margin) --
    12 m of free fall is reward-cheap altitude to bleed, but the REAL VQ2 warehouse floor is AT spawn
    (pad) height. When ON, _update_boxes raises box_min z to spawn_z - clearance (default 0.25 m, so
    RESTING on the pad stays legal -- the drone spawns AT pad z), and the ego env's existing
    below-floor => gate_collision fold (peregrine_racing_ego.step) makes diving below it a CRASH.

    GUARD: the floor is defined relative to the STANDING-START pad. With standing_start_frac < 1.0
    some episodes spawn near a random gate instead -- the pad-based floor is then only meaningful if
    every gate is above the pad, which this resolver cannot verify (courses are sampled per reset).
    Rather than be silently wrong (or silently inert -- the L16 lesson), REFUSE the combination: the
    floor stages all run standing_start_frac=1.0. Default OFF == byte-identical legacy boxes."""
    on = bool(getattr(cfg, "floor_at_spawn", False))
    clearance = float(getattr(cfg, "floor_clearance_m", 0.25))
    if on:
        if clearance < 0.0:
            raise ValueError(f"floor_clearance_m must be >= 0, got {clearance}")
        ssf = float(getattr(cfg, "standing_start_frac", 0.0))
        if ssf < 1.0:
            raise ValueError(
                f"floor_at_spawn=true requires standing_start_frac=1.0 (the floor is defined "
                f"relative to the standing-start pad); got standing_start_frac={ssf}")
    return on, clearance


# ================================================================================================
# The environment (requires diffaero -- training/eval cluster only).
# ================================================================================================
class PeregrineRacing(Racing):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)  # builds the figure-8; everything below overwrites it

        n = self.n_envs
        self._arange = torch.arange(n, device=device)

        # --- FLOOR-AT-SPAWN (A1 fix 2026-07-10; default OFF = byte-identical). Parsed BEFORE the
        # course section because _update_boxes (end of that section) consumes the flag. ---
        self._floor_at_spawn, self._floor_clearance_m = resolve_floor_at_spawn(cfg)

        # --- course source: fixed VQ1 (held-out eval / legacy) or per-env procedural sampling ---
        self.course_mode = str(getattr(cfg, "course_mode", "vq1"))
        course_path = getattr(cfg, "course_json", None) or _COURSE_JSON
        course = json.loads(open(course_path).read())
        vq1_pos = torch.tensor([g["pos_zup"] for g in course["gates"]], device=device,
                               dtype=torch.float32)
        vq1_yaw = torch.tensor([g["yaw"] for g in course["gates"]], device=device,
                               dtype=torch.float32)
        self.n_gates = vq1_pos.shape[0]
        self.gate_half_opening_m = float(course.get("inner_opening_m", 1.5)) / 2.0   # 0.75 m
        self.gate_half_outer_m = float(getattr(cfg, "gate_outer_m", 2.72)) / 2.0     # 1.36 m

        # per-env course tensors (broadcast VQ1 or sampled); + standing-start pads
        from peregrine_course import (sample_courses, VQ1_SPAWN_POS_ZUP, VQ1_SPAWN_YAW,
                                      VQ1_SPAWN_PITCH_RAD)
        self._sample_courses = sample_courses
        self._spawn_pitch = float(VQ1_SPAWN_PITCH_RAD)
        self._vq1 = {
            "gate_pos": vq1_pos, "gate_yaw": vq1_yaw,
            "spawn_pos": torch.tensor(VQ1_SPAWN_POS_ZUP, device=device, dtype=torch.float32),
            "spawn_yaw": float(VQ1_SPAWN_YAW),
        }
        self.gate_pos = vq1_pos.unsqueeze(0).expand(n, -1, -1).clone()     # (N, G, 3)
        self.gate_yaw = vq1_yaw.unsqueeze(0).expand(n, -1).clone()         # (N, G)
        self.spawn_pos = self._vq1["spawn_pos"].unsqueeze(0).expand(n, -1).clone()
        self.spawn_yaw = torch.full((n,), self._vq1["spawn_yaw"], device=device)
        if self.course_mode == "random":
            self._assign_courses(self._arange)
        self.gate_rel_pos, self.gate_yaw_rel = rel_tables(self.gate_pos, self.gate_yaw)
        self._update_boxes(self._arange)

        # --- INC7 contact-true geometry (training doctrine 2026-06-12 Sec 2; defaults OFF) ---
        self.body_radius_lo = float(getattr(cfg, "body_radius_lo", 0.0))
        self.body_radius_hi = float(getattr(cfg, "body_radius_hi", self.body_radius_lo))
        if not 0.0 <= self.body_radius_lo <= self.body_radius_hi < self.gate_half_opening_m:
            raise ValueError(f"need 0 <= body_radius_lo <= body_radius_hi < "
                             f"{self.gate_half_opening_m}; got "
                             f"[{self.body_radius_lo}, {self.body_radius_hi}]")
        self.frame_depth_m = float(getattr(cfg, "frame_depth_m", 0.0))
        if self.frame_depth_m < 0.0:
            raise ValueError(f"frame_depth_m must be >= 0; got {self.frame_depth_m}")
        self._body_radius_on = self.body_radius_hi > 0.0
        self.body_radius = torch.zeros(n, device=device)     # per-env r (m); stays 0 when OFF
        if self._body_radius_on:
            self._sample_body_radius(self._arange)

        # --- episode bookkeeping ---
        self.finished = torch.zeros(n, dtype=torch.bool, device=device)
        self.rw = RewardWeights.from_cfg(cfg)
        self.standing_start_frac = float(getattr(cfg, "standing_start_frac", 0.0))
        # SPAWN-VELOCITY RANDOMIZATION (anti-velocity-runaway 2026-07-13; default OFF == byte-identical
        # at-rest spawn). spawn_vel_frac>0 gives that FRACTION of reset envs an initial velocity directed at
        # their target gate, magnitude U[0, spawn_vel_max). Episodes are too short for the policy to
        # accelerate into the ~12 m/s runaway regime on its own, so a fast spawn EXPOSES it (pairs with the
        # velocity-cap reward term). Consumed in reset_idx; spawn_vel_frac<=0 skips that block ENTIRELY (no
        # RNG draw, velocity stays 0) so the RNG stream + spawn state are bit-identical to the legacy spawn.
        self.spawn_vel_frac = float(getattr(cfg, "spawn_vel_frac", 0.0))
        self.spawn_vel_max = float(getattr(cfg, "spawn_vel_max", 15.0))
        # per-episode diagnostics (running; pre-reset, so terminal poses are included)
        self._peak_tilt = torch.zeros(n, device=device)          # rad
        self._peak_roll = torch.zeros(n, device=device)          # rad (ZYX Euler |roll|)
        self._speed_sum = torch.zeros(n, device=device)          # sum of |v| per step
        self._nonfinite_obs = 0                                  # lifetime count (see get_observations)
        # action span for the R5 normalization (set lazily: dynamics bounds exist after init)
        span = (self.dynamics.max_action - self.dynamics.min_action).clamp(min=1e-6)
        self._act_lo, self._act_span = self.dynamics.min_action, span

        # obs = parent's 13 + body rates (3) + collective (1) -- FROZEN deployment contract
        self.obs_dim = 17

    # ---- INC7 contact-true geometry plumbing ---------------------------------------------------
    def _sample_body_radius(self, env_idx: Tensor) -> None:
        """Resample the per-env body radius r ~ U[lo, hi] (doctrine Sec 2: the halo is
        unobservable, so per-episode sampling makes the policy carry the worst case)."""
        m = int(env_idx.numel())
        if m == 0:
            return
        self.body_radius[env_idx] = (self.body_radius_lo
                                     + (self.body_radius_hi - self.body_radius_lo)
                                     * torch.rand(m, device=self.device))

    def _contact_bands(self):
        """(half_inner_eff, half_outer_eff): the body-radius-inflated frame band, per env
        (N, 1) when the feature is ON; the legacy python floats when OFF (bit-identity)."""
        if self._body_radius_on:
            r = self.body_radius.unsqueeze(-1)
            return self.gate_half_opening_m - r, self.gate_half_outer_m + r
        return self.gate_half_opening_m, self.gate_half_outer_m

    # ---- course plumbing ---------------------------------------------------------------------
    def _assign_courses(self, env_idx: Tensor) -> None:
        """Sample fresh courses for ``env_idx`` (random mode) and update the per-env tensors."""
        m = int(env_idx.numel())
        if m == 0:
            return
        c = self._sample_courses(m, device=self.device)
        self.gate_pos[env_idx] = c["gate_pos"].to(self.gate_pos.dtype)
        self.gate_yaw[env_idx] = c["gate_yaw"].to(self.gate_yaw.dtype)
        self.spawn_pos[env_idx] = c["spawn_pos"].to(self.spawn_pos.dtype)
        self.spawn_yaw[env_idx] = c["spawn_yaw"].to(self.spawn_yaw.dtype)

    def _update_boxes(self, env_idx: Tensor) -> None:
        """Per-env OOB box: course bbox INCLUDING the spawn pad, +-15 m xy / +-12 m z margins
        (the S1.3 lesson: exclude the spawn corridor and every standing start truncates on step 1).

        FLOOR-AT-SPAWN (A1 fix 2026-07-10, flag-gated, default OFF): raise box_min z to
        spawn_z - clearance (0.25 m) so the arena floor sits AT the pad instead of 12 m below it --
        the 12 m z margin below the course is exactly the free-fall budget the dive opener exploited,
        and the real VQ2 warehouse floor is at spawn height. The ego env's below-floor =>
        gate_collision fold (peregrine_racing_ego.step reads box_min[:, 2] LIVE every step) then makes
        a dive below the pad a terminal crash for free. Requires every gate ABOVE the pad (the floor
        stages pin course_spawn_below_g0 >= 0.5 and course_gates_above_spawn) or the course is
        undivable-to. box_max / the xy margins are untouched."""
        if not hasattr(self, "box_min"):
            self.box_min = torch.zeros(self.n_envs, 3, device=self.device)
            self.box_max = torch.zeros(self.n_envs, 3, device=self.device)
        margin = torch.tensor([15.0, 15.0, 12.0], device=self.device)
        pts = torch.cat([self.gate_pos[env_idx], self.spawn_pos[env_idx].unsqueeze(1)], dim=1)
        self.box_min[env_idx] = pts.amin(dim=1) - margin
        self.box_max[env_idx] = pts.amax(dim=1) + margin
        if self._floor_at_spawn:
            floor_z = self.spawn_pos[env_idx, 2] - self._floor_clearance_m
            self.box_min[env_idx, 2] = torch.maximum(self.box_min[env_idx, 2], floor_z)

    # ---- observation: parent's gate-relative obs + body rates + last collective (FROZEN) ------
    def get_observations(self, with_grad=False):
        ar, tg = self._arange, self.target_gates.long()
        gate_pos = self.gate_pos[ar, tg]
        gate_yaw = self.gate_yaw[ar, tg]
        rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)

        pos_g = mvp(rotmat_w2g, gate_pos - self._p)             # target gate rel pos, gate frame (3)
        vel_g = mvp(rotmat_w2g, self._v)                        # velocity, gate frame (3)
        rotmat_b2w = T.quaternion_to_matrix(self.q.roll(1, dims=-1))
        # CLAMP before the Euler extraction: pytorch3d's matrix_to_euler_angles takes an asin of
        # matrix entries with NO clamping -- float32 products can land at |entry| = 1+1e-7, and
        # asin(1+eps) = NaN. One poisoned env then NaNs the actor mean and kills the whole run at
        # Normal(loc) validation (S1.4 seed-0 job 3267229 died at update 848 exactly here; the
        # S1.3 5000-update NaN had the same signature). The parent clamps in its LOSS path
        # (racing.py:331) but not in get_observations -- same fix, same epsilon.
        rotmat_b2g = torch.matmul(rotmat_w2g, rotmat_b2w).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        rpy_g = T.matrix_to_euler_angles(rotmat_b2g, "ZYX")[..., [2, 1, 0]]   # attitude vs gate (3)

        nxt = torch.clamp(tg + 1, max=self.n_gates - 1)
        collective = self.last_action[..., 0:1]                 # last RESCALED normed thrust (1)

        obs = torch.cat([
            pos_g, vel_g, rpy_g,
            self._w,                                            # body rates (3)
            collective,                                         # collective (1)
            self.gate_rel_pos[ar, nxt],                         # next gate rel pos (3)
            self.gate_yaw_rel[ar, nxt].unsqueeze(-1),           # next gate rel yaw (1)
        ], dim=-1)
        # LAST-RESORT lifeline, observable not silent: any residual non-finite entry is counted
        # (exported via loss_components as obs_nonfinite) and zeroed, so a one-in-30M numerical
        # edge costs one weird-but-finite obs instead of the training run. The clamp above removes
        # the only KNOWN source; this guard exists for the unknown ones.
        finite = torch.isfinite(obs)
        if not bool(finite.all()):
            self._nonfinite_obs += int((~finite).sum())
            obs = torch.where(finite, obs, torch.zeros_like(obs))
        return obs if with_grad else obs.detach()

    # ---- state (asymmetric-critic input; also keeps check_dims honest): clamped, not wrapped --
    def get_state(self, with_grad=False):
        ar = self._arange
        states = [self._v, self.q]
        for i in range(3):
            gi = torch.clamp(self.target_gates.long() + i, max=self.n_gates - 1)
            gate_pos = self.gate_pos[ar, gi]
            gate_yaw = self.gate_yaw[ar, gi]
            rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)
            pos_g = mvp(rotmat_w2g, gate_pos - self._p)
            vel_g = mvp(rotmat_w2g, self._v)
            rotmat_b2w = T.quaternion_to_matrix(self.q.roll(1, dims=-1))
            rotmat_b2g = torch.matmul(rotmat_w2g, rotmat_b2w).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            rpy_g = T.matrix_to_euler_angles(rotmat_b2g, "ZYX")[..., [2, 1, 0]]
            states += [pos_g, vel_g, rpy_g]
        states = torch.cat(states, dim=-1)
        return states if with_grad else states.detach()

    # ---- step: event classification -> termination/truncation -> reward -----------------------
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        # type: (Tensor, bool, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        prev_pos = self._p.clone()
        self.dynamics.step(action)
        curr_pos = self._p
        ar, G = self._arange, self.n_gates

        # crossing events against ALL gates (gates are physical everywhere); INC7: the bands
        # are the per-env body-radius-inflated ones (legacy floats when the feature is OFF)
        rel_prev = world_to_gateframe(prev_pos[:, None, :] - self.gate_pos, self.gate_yaw)
        rel_curr = world_to_gateframe(curr_pos[:, None, :] - self.gate_pos, self.gate_yaw)
        half_in_eff, half_out_eff = self._contact_bands()
        ev = crossing_events(rel_prev, rel_curr, half_in_eff, half_out_eff)     # all (N, G)
        # INC7 volumetric frame: a segment touching frame material anywhere in the slab
        # |x| <= depth collides -- including plane crossings whose interpolated point threads
        # the pass band but clip material on the way (the live standing failure class).
        slab_hit = (slab_frame_hits(rel_prev, rel_curr, half_in_eff, half_out_eff,
                                    self.frame_depth_m)                         # (N, G)
                    if self.frame_depth_m > 0.0 else None)

        tg = self.target_gates.long()
        fwd_t = ev["fwd"][ar, tg]
        gate_passed = fwd_t & ev["pass_ok"][ar, tg]
        gate_miss = fwd_t & ~ev["pass_ok"][ar, tg] & ~ev["in_frame"][ar, tg]   # beyond the frame
        frame_target = (ev["fwd"] | ev["bwd"])[ar, tg] & ev["in_frame"][ar, tg]
        strike_any = (ev["fwd"] | ev["bwd"]) & ev["in_frame"]
        strike_any[ar, tg] = False
        gate_collision = frame_target | strike_any.any(dim=1)
        if slab_hit is not None:
            slab_t = slab_hit[ar, tg]
            gate_passed = gate_passed & ~slab_t      # a striking crossing is NOT a pass
            gate_miss = gate_miss & ~slab_t          # collision dominates on the target gate
            gate_collision = gate_collision | slab_hit.any(dim=1)
        pass_linf = ev["linf"][ar, tg]                                          # diag: pass offset

        # target advance / finish (non-looping point-to-point course)
        is_last = tg == (G - 1)
        newly_finished = gate_passed & is_last & ~self.finished
        advance = gate_passed & ~is_last
        self.target_gates[advance] = self.target_gates[advance] + 1
        self.n_passed_gates[gate_passed] += 1
        self.finished |= newly_finished
        tg_new = self.target_gates.long()
        self.target_pos.copy_(self.gate_pos[ar, tg_new])

        # out of bounds (per-env box) -- TERMINAL with penalty (audit C1)
        oob = ((curr_pos < self.box_min) | (curr_pos > self.box_max)).any(dim=-1)

        terminated = gate_collision | gate_miss | oob | self.finished
        truncated = self.truncated()             # parent timing: evaluated pre-increment
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        truncated = truncated & ~terminated      # a terminal step is not also a truncation
        success = self.finished.clone()

        # per-episode diagnostics
        tilt = torch.arccos(tilt_cos_from_quat_xyzw(self._q).clamp(-1.0, 1.0))
        self._peak_tilt = torch.maximum(self._peak_tilt, tilt)
        self._peak_roll = torch.maximum(self._peak_roll, roll_from_quat_xyzw(self._q).abs())
        speed = torch.linalg.norm(self._v, dim=-1)
        self._speed_sum += speed

        # reward (see module docstring; d2g measured against the POST-advance target for both
        # endpoints -- the parent convention, no spike at passage)
        gate_pos_t = self.gate_pos[ar, tg_new]
        prev_d2g = torch.linalg.norm(prev_pos - gate_pos_t, dim=-1)
        curr_d2g = torch.linalg.norm(curr_pos - gate_pos_t, dim=-1)
        time_left_s = (self.max_steps - self.progress).clamp(min=0).float() * self.dt
        a_norm = (action - self._act_lo) / self._act_span
        last_norm = (self.last_action - self._act_lo) / self._act_span
        reward, loss_components = compute_reward_terms(
            self.rw, prev_d2g=prev_d2g, curr_d2g=curr_d2g, gate_passed=gate_passed,
            gate_collision=gate_collision, gate_miss=gate_miss, oob=oob,
            newly_finished=newly_finished, time_left_s=time_left_s, quat_xyzw=self._q,
            omega=self._w, action_norm=a_norm, last_action_norm=last_norm)
        loss = (-reward).detach()        # PPO-only env: loss kept for runner logging, NOT for BPTT
        reward = reward.detach()
        loss_components["obs_nonfinite"] = float(self._nonfinite_obs)
        self.last_action.copy_(action.detach())

        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "loss_components": loss_components,
            "stats_raw": {
                "success_rate": success[reset].float(),
                "survive_rate": truncated[reset].float(),          # now == timeout rate (audit C6)
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "n_passed_gates": self.n_passed_gates[reset].float(),
                "collision_rate": gate_collision[reset].float(),
                "miss_rate": gate_miss[reset].float(),
                "oob_rate": oob[reset].float(),
                "peak_tilt_deg": torch.rad2deg(self._peak_tilt)[reset],
                "peak_roll_deg": torch.rad2deg(self._peak_roll)[reset],
                "mean_speed": (self._speed_sum / self.progress.clamp(min=1).float())[reset],
                "finish_time_s": ((self.progress.clone() - 1).float() * self.dt)[newly_finished],
                "pass_offset_m": pass_linf[gate_passed],
                # INC7 gauntlet diagnostics: volumetric strikes + contact-true crossing margin
                "slab_collision_rate": (slab_hit.any(dim=1) if slab_hit is not None
                                        else torch.zeros_like(gate_collision))[reset].float(),
                "pass_margin_m": ((half_in_eff.squeeze(-1) if torch.is_tensor(half_in_eff)
                                   else torch.full_like(pass_linf, half_in_eff))
                                  - pass_linf)[gate_passed],
            },
        }
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (loss, reward), terminated, extra

    # ---- legacy aperture API (kept for any external callers; target gate only) -----------------
    def is_passed(self, prev_pos):
        ar, tg = self._arange, self.target_gates.long()
        gate_pos = self.gate_pos[ar, tg]
        gate_yaw = self.gate_yaw[ar, tg]
        rel_prev = world_to_gateframe(prev_pos - gate_pos, gate_yaw)
        rel_curr = world_to_gateframe(self.p - gate_pos, gate_yaw)
        half_in_eff, half_out_eff = self._contact_bands()
        ev = crossing_events(rel_prev, rel_curr,
                             (half_in_eff.squeeze(-1) if torch.is_tensor(half_in_eff)
                              else half_in_eff),
                             (half_out_eff.squeeze(-1) if torch.is_tensor(half_out_eff)
                              else half_out_eff))
        return ev["fwd"] & ev["pass_ok"], ev["fwd"] & ~ev["pass_ok"]

    # ---- truncation: timeout only (OOB is a termination now; see step()) -----------------------
    def truncated(self):
        return self.progress >= self.max_steps

    # ---- reset: full override (per-env courses; parent's reset indexes shared-course tensors) --
    def reset_idx(self, env_idx: Tensor):
        self.randomizer.refresh(env_idx)
        m = int(env_idx.numel())
        if m == 0:
            return
        dev = self.device

        if self.course_mode == "random":
            self._assign_courses(env_idx)
            self.gate_rel_pos[env_idx], self.gate_yaw_rel[env_idx] = \
                (t[env_idx] for t in rel_tables(self.gate_pos, self.gate_yaw))
            self._update_boxes(env_idx)
        if self._body_radius_on:                      # INC7: fresh halo per episode
            self._sample_body_radius(env_idx)

        # spawn selection: standing start (the pad) vs 1 m up-course of a random target gate
        standing = torch.rand(m, device=dev) < self.standing_start_frac
        tg_new = torch.randint(0, self.n_gates, (m,), device=dev, dtype=torch.int32)
        tg_new[standing] = 0

        gp = self.gate_pos[env_idx, tg_new.long()]
        gy = self.gate_yaw[env_idx, tg_new.long()]
        near_pos = gp - torch.stack([torch.cos(gy), torch.sin(gy), torch.zeros_like(gy)], dim=-1)
        near_yaw = gy + math.pi                       # tail-first w.r.t. the target gate
        sp = self.spawn_pos[env_idx]
        sy = self.spawn_yaw[env_idx]

        pos = torch.where(standing.unsqueeze(-1), sp, near_pos)
        yaw = torch.where(standing, sy, near_yaw)
        pitch = torch.where(standing, torch.full_like(yaw, self._spawn_pitch),
                            torch.zeros_like(yaw))
        # pose jitter: +-0.25 m xy + +-0.15 rad axis-angle (robustness around the nominal pose)
        pos = pos + torch.cat([0.5 * (torch.rand(m, 2, device=dev) - 0.5),
                               torch.zeros(m, 1, device=dev)], dim=-1)
        q = quat_xyzw_from_yaw_pitch(yaw, pitch)
        q = quat_xyzw_mul(q, quat_xyzw_from_axis_angle(0.3 * (torch.rand(m, 3, device=dev) - 0.5)))

        state = torch.zeros(m, self.dynamics.state_dim, device=dev)
        state[:, 0:3] = pos
        state[:, 3:7] = q                              # XYZW; v = w = 0 (at rest)
        # SPAWN-VELOCITY RANDOMIZATION (anti-velocity-runaway 2026-07-13; default OFF -> v stays 0, byte-
        # identical). A FRACTION (spawn_vel_frac) of the reset envs get a WORLD-frame initial velocity
        # directed at their target gate (gp, == gate 0 for the standing-start _pef path -> toward the FIRST
        # gate), magnitude U[0, spawn_vel_max). spawn_vel_frac<=0 skips this block ENTIRELY (no RNG draw,
        # v==0), so the RNG stream + spawn state stay bit-identical to the at-rest legacy spawn. On the ego
        # path the estimator cold-init (ego reset_idx, after super()) reads self._v -> it sees this spawn.
        if self.spawn_vel_frac > 0.0:
            state[:, 7:10] = spawn_velocity_toward_gate(
                gp - pos, torch.rand(m, device=dev), torch.rand(m, device=dev),
                self.spawn_vel_frac, self.spawn_vel_max)
        mask = torch.zeros_like(self.dynamics._state, dtype=torch.bool)
        mask[env_idx] = True
        full = torch.zeros_like(self.dynamics._state)
        full[env_idx] = state
        self.dynamics._state = torch.where(mask, full, self.dynamics._state)
        self.dynamics.reset_idx(env_idx)               # DR resample + aux state (thrust, buffers)

        self.target_gates[env_idx] = tg_new
        self.n_passed_gates[env_idx] = 0
        self.finished[env_idx] = False
        self.init_pos[env_idx] = pos
        self.progress[env_idx] = 0
        self.arrive_time[env_idx] = 0
        self.last_action[env_idx] = 0.0                # collective obs starts at 0 (contract)
        self._peak_tilt[env_idx] = 0.0
        self._peak_roll[env_idx] = 0.0
        self._speed_sum[env_idx] = 0.0
        self.max_vel[env_idx] = (torch.rand(m, device=dev)
                                 * (self.max_target_vel - self.min_target_vel)
                                 + self.min_target_vel)
        self.target_pos.copy_(self.gate_pos[self._arange, self.target_gates.long()])
