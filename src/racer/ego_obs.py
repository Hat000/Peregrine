"""Egocentric 21-dim RL observation builder for the VQ2 deploy wire (the fly_ego adapter).

Reproduces, on the live MAVLink wire, the observation contract of the egocentric training env
(chaum worktree ``rl/peregrine_racing_ego.py::ego_actor_obs`` + ``rl/ego_estimator.py``):

  idx    | name              | units / frame
  0:3    | velocity          | m/s, BODY FLU (R_wb^T v_world), virtual-flipped (tail-first)
  3:5    | roll, pitch       | rad, gravity-leveled, NO yaw, of the virtual-flipped body
  5:8    | body_rates        | rad/s, BODY FLU, virtual-flipped
  8      | last_collective   | previous tick's RESCALED normed_thrust, g-units [0, 3.765]
  9:11   | coarse_sector     | (horiz, vert) each in {-1, 0, 1}, gravity-leveled heading frame
  11:14  | slot0 rel_pos     | m, drone->ACTIVE-gate (tg), BODY FLU, virtual-flipped, MASKED to 0
  14     | slot0 confidence  | [0,1] staleness scalar (1 fresh, linear -> 0 at the horizon)
  15     | slot0 visible_area| [0,1] normalized apparent inner-opening area (1 == square-on)
  16:19  | slot1 rel_pos     | m, drone->NEXT-gate (tg+1), same frame/mask as slot0 -- ZEROS unless
         |                   |   ``slot1_enabled`` AND a fresh next-gate pose is supplied (else H6-safe)
  19     | slot1 confidence  | [0,1] staleness scalar for the NEXT gate (0 when slot1 masked)
  20     | slot1 visible_area| [0,1] apparent area for the NEXT gate (0 when slot1 masked)

WINDOW=2 (``[current, next]``): slot0 is the ACTIVE gate (tg = RACE_STATUS active_gate_index),
slot1 is the NEXT gate in COURSE ORDER (tg+1) -- mirroring the training ``ego_window_indices``
(gate index clamp(tg+k, max=n_gates-1)) + per-slot INDEPENDENT masking of ``ego_actor_obs``. The
single-gate champions trained with slot1 PINNED ZERO (H6); the multi-gate ``_pef`` generation
trained with slot1 frequently populated (n_gates=2, 10-20 m spacing -> the next gate co-visible on
the current approach), so for THOSE checkpoints feeding zeros is OOD. ``slot1_enabled`` (default
OFF) selects between the two: OFF -> the single-slot n_gates=1 path (slot1 zeros, byte-identical to
the single-gate deploy); ON -> a real 2-slot window fed by a caller-supplied ``next_pose``. The
next-gate SOURCE (surfacing + associating tg+1 as its own PnP pose) is VISION-STACK-owned; this
module is only the SINK. Training NEVER fed a WRONG gate into slot1 -- only the correct tg+1 or
zero -- so an absent/uncertain ``next_pose`` MUST mask to zero here (never fabricated).

FRAME CONVENTIONS (the part that silently kills policies):
  * Training flies TAIL-FIRST: all body-frame obs quantities (velocity, rates, rel_pos, roll/
    pitch) are expressed in the UNFLIPPED tail-first control body frame (Z-up FLU). The deployed
    drone flies NOSE-first, so the SAME virtual pi-about-body-z flip fly_rl uses for the inc7 obs
    (``_RZ_PI_BODY = diag(-1,-1,1)``) is applied here to velocity, body rates AND rel_pos, and the
    attitude is flipped BEFORE the roll/pitch extraction (exactly ``obs_from_zup``'s convention).
    ``policy_step`` un-flips the action rates -- the flip must NOT be applied twice.
  * rel_pos comes from the PnP GatePose ``t_cam_gate`` (camera OPTICAL frame) rotated into body
    FRD through ``frames.R_camera_from_body()`` (inheriting the 20 deg mount + any ANGULAR
    boresight), plus the METRIC boresight camera-vs-body vertical offset ``frames.BORESIGHT.
    vert_offset_m`` (-0.25 m, FRD +Z down) -- the SAME dual-form bake the localization +L lever
    applies (localization._apply_camera_vert_offset), read LIVE from ``frames.BORESIGHT``.
  * roll/pitch use the TRAINING extraction on the Z-up body->world matrix
    (ego_estimator.py::_euler_roll_pitch_from_R):
        pitch = atan2(-R20, hypot(R21, R22));  roll = atan2(R21, R22).

MASKING PARITY (champion ``single_gate_varied_gvf_lpara_anneal`` family = obs_coast OFF,
stale_horizon 0.5 s -- EgoEstimatorConfig defaults; the stage overrides neither):
  * confidence = clamp(1 - age/stale_horizon, 0, 1); the slot is masked to ZEROS when the
    confidence hits 0 (training hard-mask past the horizon).
  * training's ``det`` is GEOMETRIC detectability recomputed every 33 ms tick; on the wire
    detections arrive ~7-15 Hz, so literal this-tick masking would flicker at a duty cycle
    training never saw. The deploy analog is ``det_proxy = (age < det_hold_s)`` (default 0.2 s):
    continuous while tracked, masks ~det_hold_s after loss-of-lock -- reproducing the training
    blackout cliff slightly delayed. A coasted non-zero rel_pos is NEVER fed past that (the
    champion trained on zeros at the crossing -- coast-OFF).
    🛑 THAT PROXY DOES NOT REPRODUCE THE ONE BLACKOUT THAT MATTERS. Every fresh fix resets ``age``,
    so the cliff mostly never fires before the gate plane: measured over 899 seam-guarded confirmed
    passes, slot0 stays NON-ZERO through the whole blind run-in on 65.7% of approaches (mean filled
    fraction 0.872), while TRAINING masks slot0 on 100% of ticks inside 1.0 m. The final ~1.8 m is
    flown blind by construction (VFOV 58.7 deg, axis +20 deg up -> the gate overflows vertically),
    and that is the interval in which every gate outcome is decided. ``det_geometric`` (default OFF)
    restores the missing test by ANDing training's own 8-keypoint rule into ``det`` -- see
    ``gate_detectable_geometric`` for the algorithm, the measured training-parity calibration and
    the two documented divergences.
  * between accepted fixes (age < det_hold) the held rel_pos is EGO-PROPAGATED exactly like the
    training estimator (rotate by -body_rate*dt, translate by -v_body*dt;
    ego_estimator.py:447-459) so the intra-gap obs stays coherent.
  * an ACCEPTED fix is blended into that propagated belief with a scalar gain ``fix_gain``:
    rel_new = (1-K)*propagated_held + K*fix. Deploy default K=1.0 == the historical SNAP (the
    held belief is discarded on every fix), which is why measurement noise -- 96% of fixes inside
    1-2 m of a gate carry <4 corners -- reaches the policy unfiltered. Training used the
    steady-state low-pass K = 1/N_eff with N_eff ~ U[4,9] (ego_estimator.py:722-734, K~0.154), so
    the pilot can dial the deploy gain onto the training smoothing. K is FORCED to 1.0 on
    RE-ACQUISITION (no held belief, or a held belief older than ``stale_horizon_s``) -- exactly
    the training reacquisition branch, which discards a prior that has drifted through a blackout.

COARSE SECTOR (deploy analog of ``build_coarse_map``, peregrine_racing_ego.py:180-235):
  Training's sector is a STATIC per-gate (horiz, vert) bucket from COURSE geometry: horiz = the
  sign of the leveled azimuth TURN from the incoming leg to the outgoing leg; vert = the sign of
  the outgoing leg's elevation (last gate reuses the incoming leg). For a SINGLE-gate course
  (the champion's entire training life) outgoing == incoming => turn == 0 => horiz IDENTICALLY 0,
  and vert = the elevation bucket of the spawn->gate leg (constant per episode).
  The wire has no course map, so the analog computed here is:
    * horiz = 0 (matches the training definition exactly when no next-gate leg is known, which
      is also the ONLY value the single-gate champion ever saw);
    * vert  = the elevation bucket (deadband 0.20 rad, the training default) of the gravity-
      leveled drone->gate vector at the FIRST accepted fix for the current gate index -- the
      closest wire-observable stand-in for the incoming-leg elevation (at acquisition, right
      after the previous pass / spawn, drone->gate IS approximately the leg).
  The sector is then HELD STATIC until the active gate index changes (training's sector is
  static and fed regardless of visibility/masking). Before the first fix of a gate it is (0,0)
  (neutral prior; training would have fed the true static bucket -- documented divergence).
  ``sector_mode='zero'`` pins (0,0) permanently (the flat-course / diagnostic fallback).
  ``sector_mode='map'`` feeds a hand-authored STATIC per-gate ``coarse_map[active_gate_index]``
  (horiz,vert) bucket, latched on each active-gate change BEFORE the first fix -- the deploy
  restoration of the horizontal TURN prior the ``_pef`` champions trained on (``auto`` can only
  ever emit horiz=0: the wire has no next-gate geometry). This is what lets the policy anticipate
  an off-axis next gate instead of flying as if every gate is dead ahead.

The final 21-dim assembly + masking goes through ``ego_actor_obs_np`` -- a faithful single-env
numpy port of the training ``ego_actor_obs`` (window indexing, keep-logic, mask multiply, concat
order, NaN guard) -- so the deploy masking semantics are the TRAINING code path by construction
(pinned element-exact against the verbatim reference in tests/test_ego_deploy_obs.py).

PER-GATE AIM OFFSET (2026-07-27, default OFF): ``aim_offsets`` shifts the slot-0 lever the policy
consumes by a per-gate-index 2-vector in BODY FLU, released by range before the gate. This is the
repo form of the pilot's ShadowPC-only ``aim_off`` dodge for the two invisible obstacles ~14.5 m
short of gates 4 and 5; ``parse_aim_offsets`` carries the measured sign convention and why the
``GateSeeker._valid_poses`` camera-frame bias is NOT the same lever.

ADDITIVE + OPT-IN: nothing imports this module on the default fly_rl paths; VQ1/inc7/gate-seeker
stay byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from racer import frames
from racer.contracts import GatePose
from racer.ego_velocity import LateralVelocityFuser, VelocityFusionConfig
from racer.vision.gate_pose import GATE_INNER_SIZE_M, gate_object_points

# ---------------------------------------------------------------------------------------------
# Contract constants (chaum rl/peregrine_racing_ego.py:78-81 -- FIXED by the trained checkpoint)
# ---------------------------------------------------------------------------------------------
EGO_WINDOW = 2                                   # [current, next]
EGO_PER_SLOT = 5                                 # rel_pos(3) + confidence(1) + visible_area(1)
EGO_OBS_DIM = 9 + 2 + EGO_WINDOW * EGO_PER_SLOT  # 21

# Champion estimator defaults (chaum rl/ego_estimator.py:124; the deployed stage overrides none).
EGO_STALE_HORIZON_S = 0.5    # confidence 1 -> 0 over this after loss-of-lock, then hard mask
EGO_DET_HOLD_S = 0.2         # deploy analog of the per-tick geometric ``det`` (see module doc)
SECTOR_DEADBAND_RAD = 0.20   # build_coarse_map horiz/vert deadband (training default)

# --- TRAINING VISIBILITY MODEL constants, mirrored 1:1 from chaum rl/gate_visibility.py -----------
# (GATE_OUTER_M 2.7 / GATE_INNER_M 1.5 / GATE_DEPTH_M 0.26 / MIN_VISIBLE_CORNERS / FAR_CAP_M_DEFAULT).
# These are the spec-3.7 gate, so they are NOT the pass-test aperture: training threads a gate with
# ``0.75 - body_radius`` (r ~ U[0.28,0.38] => 0.37-0.47 m). For VISIBILITY the PHYSICAL half-width is
# what projects into the frame, so 0.75/1.35 are correct here and must not be shrunk.
GATE_VIS_INNER_HALF_M = 0.75          # gate_visibility._HALF_INNER  (inner opening 1.5 m)
GATE_VIS_OUTER_HALF_M = 1.35          # gate_visibility._HALF_OUTER  (outer frame 2.7 m)
GATE_VIS_FRONT_FACE_M = -0.13         # gate_visibility._FRONT_FACE_DOWNRANGE (= -GATE_DEPTH_M/2):
                                      # the keypoints sit on the face NEARER the incoming drone.
GATE_VIS_MIN_CORNERS = 4              # gate_visibility.MIN_VISIBLE_CORNERS
GATE_VIS_FAR_CAP_M = 30.0             # gate_visibility.FAR_CAP_M_DEFAULT (cap on the CENTRE range)
# The 8 keypoints as (right, up) coefficients, inner ring then outer ring, in the corner order
# gate_visibility.corners_gate_frame() emits ((-,-), (+,-), (+,+), (-,+)). Built once: the armed
# path runs every control tick on a COMPUTE-BOUND loop, so per-tick allocation is not free.
_GATE_VIS_RU = np.array(
    [[-GATE_VIS_INNER_HALF_M, -GATE_VIS_INNER_HALF_M],
     [+GATE_VIS_INNER_HALF_M, -GATE_VIS_INNER_HALF_M],
     [+GATE_VIS_INNER_HALF_M, +GATE_VIS_INNER_HALF_M],
     [-GATE_VIS_INNER_HALF_M, +GATE_VIS_INNER_HALF_M],
     [-GATE_VIS_OUTER_HALF_M, -GATE_VIS_OUTER_HALF_M],
     [+GATE_VIS_OUTER_HALF_M, -GATE_VIS_OUTER_HALF_M],
     [+GATE_VIS_OUTER_HALF_M, +GATE_VIS_OUTER_HALF_M],
     [-GATE_VIS_OUTER_HALF_M, +GATE_VIS_OUTER_HALF_M]], dtype=np.float64)

# Body-frame FRD <-> FLU flip (R_x(pi) diagonal) and the virtual pi-about-body-z flip -- SAME
# constants as rl/fly_rl.py (_FLIP body part / _RZ_PI_BODY). Redefined here so src/ never imports
# rl/ (keeps the dependency direction clean); pinned equal in tests.
_FLIP_FRD_FLU = np.array([1.0, -1.0, -1.0], dtype=np.float64)
_RZ_PI_BODY = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


# ---------------------------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------------------------
def roll_pitch_zup(R_b2w_zup: np.ndarray) -> tuple[float, float]:
    """Gravity-leveled (roll, pitch) from a body->world Z-up rotation -- the TRAINING extraction
    (chaum rl/ego_estimator.py::_euler_roll_pitch_from_R:223-236). NO yaw (forbidden)."""
    r20 = float(R_b2w_zup[2, 0])
    r21 = float(R_b2w_zup[2, 1])
    r22 = float(R_b2w_zup[2, 2])
    pitch = float(np.arctan2(-r20, np.hypot(r21, r22)))
    roll = float(np.arctan2(r21, r22))
    return roll, pitch


def leveled_from_body(v_body_flu: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """Rotate a body-FLU vector into the gravity-leveled heading frame (yaw removed):
    for R_wb = Rz(yaw) Ry(pitch) Rx(roll) (the Z-up ZYX convention the extraction above inverts),
    Rz(-yaw) v_world = Ry(pitch) Rx(roll) v_body. Used only for the coarse-sector buckets, whose
    azimuth-DIFFERENCE / elevation outputs are invariant to the residual global-yaw ambiguity."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    return Ry @ Rx @ np.asarray(v_body_flu, dtype=np.float64)


def rel_pos_body_frd_from_gatepose(t_cam_gate: np.ndarray) -> np.ndarray:
    """Drone->gate-centre vector in BODY FRD from the PnP camera-optical lever.

    Goes through the SAME transform path the localization +L lever uses so the calibrated
    extrinsics are inherited, not re-derived:
      * ``frames.R_camera_from_body()`` -- the +20 deg mount plus any ANGULAR boresight;
      * the METRIC boresight ``frames.BORESIGHT.vert_offset_m`` (-0.25 m, FRD +Z down): the PnP
        lever lands at the camera OPTICAL CENTRE; the body-origin-relative vector is
        rel_body = R_cb^T t_cam + [0, 0, vert_offset_m] (the exact body-frame form of
        localization._apply_camera_vert_offset's world-frame correction). Read LIVE from
        ``frames.BORESIGHT`` (module attr) so a calibration edit propagates here too."""
    t = np.asarray(t_cam_gate, dtype=np.float64)
    rel = frames.R_camera_from_body().T @ t
    voff = frames.BORESIGHT.vert_offset_m
    if voff != 0.0:
        rel = rel + np.array([0.0, 0.0, voff])
    return rel


def visible_area_from_gatepose(R_cam_gate: np.ndarray, t_cam_gate: np.ndarray) -> float:
    """Normalized APPARENT projected inner-opening area in [0,1] (1 == square-on) -- the deploy
    analog of training's ``gate_visibility.gate_apparent_area`` (chaum gate_visibility.py:291-338).

    Training: image-space shoelace area of the 4 projected INNER corners / the square-on area a
    frontal opening of the same size projects at the same range (fx*fy*L^2/r^2) -- range-invariant.
    Deploy: project the 4 gate-model inner corners through the ESTIMATED PnP pose (works for both
    4-corner IPPE and 3-corner P3P fits -- the pose defines the full quad) with the real camera
    intrinsics, then the identical shoelace/reference ratio with r = |t_cam_gate|. Any corner at
    or behind the image plane -> 0 (matches training's ``all_front`` guard)."""
    R = np.asarray(R_cam_gate, dtype=np.float64)
    t = np.asarray(t_cam_gate, dtype=np.float64)
    corners_cam = (R @ gate_object_points().T).T + t          # (4,3) camera optical
    z = corners_cam[:, 2]
    if not np.all(z > 1e-6):
        return 0.0
    K = frames.CAMERA_INTRINSICS_K
    fx, fy = float(K[0, 0]), float(K[1, 1])
    u = fx * corners_cam[:, 0] / z + float(K[0, 2])
    v = fy * corners_cam[:, 1] / z + float(K[1, 2])
    # shoelace over the canonical corner ring (LL, LR, UR, UL -- a convex quad; abs -> unsigned)
    area_px = 0.5 * abs(
        u[0] * v[1] - u[1] * v[0]
        + u[1] * v[2] - u[2] * v[1]
        + u[2] * v[3] - u[3] * v[2]
        + u[3] * v[0] - u[0] * v[3])
    rng = float(np.linalg.norm(t))
    if rng < 1e-3:
        return 0.0
    ref = fx * fy * (GATE_INNER_SIZE_M ** 2) / (rng * rng)
    return float(np.clip(area_px / max(ref, 1e-9), 0.0, 1.0))


# ---------------------------------------------------------------------------------------------
# GEOMETRIC DETECTABILITY (2026-07-27, default OFF) -- the deploy mirror of training's ``det``
# ---------------------------------------------------------------------------------------------
def gate_detectable_geometric(rel_flu: "np.ndarray | None", R_b2w_zup: np.ndarray, *,
                              min_corners: int = GATE_VIS_MIN_CORNERS,
                              far_cap_m: float = GATE_VIS_FAR_CAP_M,
                              R_cam_from_body: "np.ndarray | None" = None) -> tuple[bool, int]:
    """Is the gate GEOMETRICALLY detectable from the held belief? -- the deploy mirror of chaum
    ``rl/gate_visibility.py::gate_detectable`` (8 keypoints, >= 4 in frame, 30 m centre cap).

    WHY THIS EXISTS. Training recomputes ``det`` from GEOMETRY every 33 ms and hard-masks
    ``obs[11:16]`` to zeros the moment the gate stops fitting in frame. Deploy substitutes a TIME
    proxy, ``det_proxy = (age < det_hold_s)`` -- and that proxy mostly NEVER FIRES before the gate
    plane, because every fresh fix resets ``age``. Measured over 899 seam-guarded confirmed passes:
      * NOT ONE pass ever received a fix inside 1.19 m; last-sighted range p10 1.48 / med 1.78 / p90
        2.80 m -- the final ~1.8 m is flown blind BY CONSTRUCTION (camera HFOV 90 deg but VFOV only
        58.7 deg with the axis pitched +20 deg UP, so the gate overflows VERTICALLY first);
      * TRAINING masks slot0 to zeros on 100% of ticks inside 1.0 m;
      * THE WIRE keeps feeding a filled, coasted lever -- 65.7% of approaches stay non-zero through
        the whole blind run-in, mean filled fraction 0.872, 94% filled at the last tick before the
        gate advance.
    So the policy is handed a lever in exactly the interval where every gate outcome is decided, in
    a state it never trained on. This function is the geometry the proxy was standing in for.

    THE ALGORITHM IS TRAINING'S, NOT A NEW ONE. 8 front-face keypoints (4 inner at +-0.75 m, 4 outer
    at +-1.35 m, all offset -0.13 m downrange onto the face nearer the drone), projected through the
    SAME intrinsics and the SAME +20 deg mount, counted IN-FRAME iff camera-depth tz > 0 and
    u in [0, 640) and v in [0, 360); detectable iff >= 4 of 8 in frame AND the centre range <= 30 m.

    NO GROUND TRUTH. Inputs are the builder's own HELD belief (``rel_flu``, TRUE body FLU, from PnP
    + ego-propagation) and the AHRS attitude. There is no world position, no gate map, no truth.

    WHAT DIFFERS FROM TRAINING -- both measured, not asserted:

    1. GATE ORIENTATION. Training knows each gate's world yaw. The wire does not (no map), and the
       one wire quantity that could supply it -- the PnP ``R_cam_gate`` -- is documented on
       ``GatePose.visible_area_meas`` as "the one genuinely ambiguous part of the pose fit (IPPE's
       two solutions differ in tilt SIGN)" and measured a near-constant 0.99 apparent area in
       flight, i.e. it reports "square-on" always and carries no information. So this uses the
       SQUARE-ON model the closed form already validated: the gate is a WORLD-VERTICAL square whose
       normal is the HORIZONTAL component of the line of sight. It is exact whenever the approach
       azimuth matches the gate yaw (which a drone about to fly THROUGH the opening approximately
       satisfies) and degrades smoothly otherwise.
       COST, measured against training's own ``gate_detectable`` (torch) over 398 random realistic
       approaches (roll ~N(0,15) deg, pitch ~N(24,10) deg, gate yaw vs approach ~N(0,12) deg,
       lateral/vertical offsets ~N(0,0.4) m): cutoff range median TRAINING 1.861 m vs DEPLOY 1.861 m,
       per-approach delta median +0.009 m, p10 -0.256 / p90 +0.217 m. UNBIASED, scatter ~+-0.25 m.
       On a straight-in approach it is EXACT: this function reproduces training's published horizon
       table to the millimetre (head-on level 1.290, 0.3 m high 1.185, 0.3 m low 1.550, pitch +10
       1.780, pitch +24 1.715 -- all delta 0.000, pinned in tests/test_ego_det_geometric.py).

    2. OCCLUSION BY OTHER GATES. Training runs an image-space projected-annulus test against every
       OTHER gate. The wire has no other-gate geometry, so it is omitted. This can only make the
       deploy test MORE permissive (it never masks for an occluder), i.e. it never fires spuriously.

    3. THE METRIC BORESIGHT. ``frames.BORESIGHT.vert_offset_m`` (-0.25 m) is undone here, because
       ``rel_pos_body_frd_from_gatepose`` ADDED it to move the PnP lever from the camera optical
       centre to the body origin. Undoing it reconstructs ``t_cam_gate`` EXACTLY at a fresh fix, so
       this test evaluates precisely the geometry the detector itself was working with -- which is
       the point, and is true whatever the boresight physically represents. Training has no such
       term, so it costs a measured +0.123 m median cutoff vs training (masks that much earlier);
       that is smaller than the p10/p90 scatter above, and it is a REAL property of the deploy
       chain, not an approximation.

    FRAMES -- the part that silently inverts. Everything here is the TRUE, UNFLIPPED physical body:
    ``rel_flu`` is TRUE body FLU [fwd, LEFT, UP] and ``R_b2w_zup`` is the TRUE body-FLU -> world-Z-up
    rotation. The virtual pi-about-body-z flip (``_RZ_PI_BODY``) is NEVER applied: it exists because
    training flies TAIL-FIRST and flips only its EMULATED camera so it looks along travel
    (peregrine_racing_ego.py::_cam_R_wb), while the deployed drone flies NOSE-first with a real
    camera already pointing that way. Same physical situation, so the deploy side uses the unflipped
    frame. Do NOT pass ``obs[3:5]`` / ``obs[11:14]`` here -- those are virtual-flipped.
    World UP in body FLU is taken as ``R_b2w_zup[2, :]`` (the third ROW == R^T @ e_z), which removes
    the drone's own attitude EXACTLY and is yaw-free by construction.

    VALIDATED AGAINST THE REAL DETECTOR, not only against training. Replayed over 564 recorded
    flights / 36 967 ticks that carried a genuine fix (``pose_seen``): if the detector produced a
    fix the gate demonstrably WAS in frame, so every disagreement is a tick this test would mask
    wrongly. Disagreement is 3.07% overall and is 0.0-0.2% in EVERY range bin from 3 m to 23 m --
    it is confined to 1-3 m (83.2% of the 1-2 m bin), with the median disagreeing tick at 1.82 m
    carrying 3 of 8 corners, i.e. sitting on the 4-of-8 threshold. That is not model error: it is
    the deploy<->training gap itself, and it lands on training's own 50% crossing of 1.83 m. Effect
    on the state that matters: inside 1.0 m, where training feeds ZEROS on 100% of ticks, the wire
    has 571 filled slot0 ticks today and 0 when this is armed.
    🚩 The FIRST replay metric I wrote -- "closest still-detectable range per approach" -- gave a
    p90 of 18 m and was WITHDRAWN: it read the range off the last-in-time detectable tick, where
    ``rel_flu`` is a coasted (sometimes mis-locked) belief whose range jumps, so drifted levers
    dominated the tail. Compare this test against a MEASUREMENT (``pose_seen``), never against
    another propagated belief.

    ``R_cam_from_body`` is an optional pre-built ``frames.R_camera_from_body()`` -- purely a cost
    hoist for the two-slot caller (that call rebuilds two scipy Rotations, ~165 us, and the deploy
    loop is COMPUTE-BOUND at p50 29.5 ms/tick where what pays is vision freshness). None (the
    default) reads it LIVE, so a calibration edit still propagates. It is NOT a place to inject a
    different camera.

    Returns ``(detectable, n_corners_in_frame)``. A None / non-finite / zero-length lever, or a
    degenerate attitude, returns ``(False, 0)`` -- "I cannot see it", the safe reading for a mask.
    """
    if rel_flu is None:
        return False, 0
    rel = np.asarray(rel_flu, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(rel)):
        return False, 0
    rng = float(np.linalg.norm(rel))
    if rng < 1e-6 or rng > float(far_cap_m):
        return False, 0

    R = np.asarray(R_b2w_zup, dtype=np.float64).reshape(3, 3)
    up_b = R[2, :].astype(np.float64).copy()       # world UP expressed in body FLU (yaw-free)
    n_up = float(np.linalg.norm(up_b))
    if not np.isfinite(n_up) or n_up < 1e-9:
        return False, 0
    up_b /= n_up

    los = rel / rng
    normal = los - float(los @ up_b) * up_b        # HORIZONTAL component of the LOS == gate normal
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-6:
        # Gate (almost) straight overhead or underfoot: the horizontal LOS vanishes and the
        # world-vertical gate model is undefined. Fall back to a plane perpendicular to the LOS with
        # an arbitrary but orthonormal in-plane basis -- the +-h corner SET is symmetric, so the
        # choice only permutes the corner list. Never reached on a racing approach; here so the
        # function cannot emit NaN.
        normal = los
        alt = np.array([1.0, 0.0, 0.0]) if abs(float(los[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        up_b = np.cross(normal, alt)
        up_b /= max(float(np.linalg.norm(up_b)), 1e-12)
    else:
        normal = normal / n_norm
    right_b = np.cross(up_b, normal)               # sign irrelevant: the corner set is +- symmetric

    # 8 front-face keypoints, inner ring then outer ring (gate_visibility.corners_gate_frame order).
    corners_flu = (rel + GATE_VIS_FRONT_FACE_M * normal)[None, :] + (
        _GATE_VIS_RU @ np.stack([right_b, up_b]))                                   # (8,3)

    # body FLU -> body FRD -> camera optical, undoing the metric boresight the held lever carries
    # (exact inverse of rel_pos_body_frd_from_gatepose, so at a fresh fix t == pose.t_cam_gate).
    c_frd = corners_flu * _FLIP_FRD_FLU[None, :]
    voff = frames.BORESIGHT.vert_offset_m
    if voff != 0.0:
        c_frd[:, 2] -= voff
    R_cb = frames.R_camera_from_body() if R_cam_from_body is None else R_cam_from_body
    t = c_frd @ np.asarray(R_cb, dtype=np.float64).T                                # (8,3)

    K = frames.CAMERA_INTRINSICS_K
    z = t[:, 2]
    front = z > 0.0
    z_safe = np.where(front, z, 1.0)
    u = K[0, 0] * t[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * t[:, 1] / z_safe + K[1, 2]
    in_frame = (front & (u >= 0.0) & (u < frames.IMAGE_WIDTH)
                & (v >= 0.0) & (v < frames.IMAGE_HEIGHT))
    n_vis = int(np.count_nonzero(in_frame))
    return bool(n_vis >= int(min_corners)), n_vis


# ---------------------------------------------------------------------------------------------
# Single-env numpy port of the training obs assembly (masking parity by construction)
# ---------------------------------------------------------------------------------------------
def ego_actor_obs_np(velocity: np.ndarray, roll_pitch: np.ndarray, body_rates: np.ndarray,
                     last_collective: float, sector: np.ndarray, rel_pos: np.ndarray,
                     confidence: np.ndarray, visible_area: np.ndarray, detectable: np.ndarray,
                     target_gate: int, n_gates: int, obs_coast: bool = False) -> np.ndarray:
    """Faithful SINGLE-ENV numpy port of chaum ``rl/peregrine_racing_ego.py::ego_actor_obs``
    (window indexing lines 241-253 + masking/concat lines 291-333) plus the ``get_observations``
    NaN guard (lines 759-761). Pinned element-exact against the verbatim torch reference in
    tests/test_ego_deploy_obs.py.

    velocity (3,), roll_pitch (2,), body_rates (3,), sector (G,2), rel_pos (G,3),
    confidence (G,), visible_area (G,), detectable (G,) bool. Returns float32 (21,)."""
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    roll_pitch = np.asarray(roll_pitch, dtype=np.float64).reshape(2)
    body_rates = np.asarray(body_rates, dtype=np.float64).reshape(3)
    rel_pos = np.asarray(rel_pos, dtype=np.float64).reshape(-1, 3)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    visible_area = np.asarray(visible_area, dtype=np.float64).reshape(-1)
    detectable = np.asarray(detectable, dtype=bool).reshape(-1)
    sector = np.asarray(sector, dtype=np.float64).reshape(-1, 2)
    tg = int(target_gate)

    slots = []
    for k in range(EGO_WINDOW):
        raw = tg + k                                   # ego_window_indices: raw = tg + k
        valid = raw < n_gates                          # valid[:,k] = raw < n_gates
        g = min(raw, n_gates - 1)                      # gidx = clamp(raw, max=n_gates-1)
        conf = float(confidence[g])
        area = float(visible_area[g])
        det = bool(detectable[g])
        keep = valid and (conf > 0.0)                  # in-horizon test
        if not obs_coast:
            keep = keep and det                        # legacy hard-mask on this-step visibility
        kf = 1.0 if keep else 0.0
        rel = rel_pos[g] * kf
        slots.append(np.concatenate([rel, [conf * kf, area * kf]]))

    obs = np.concatenate([
        velocity,                    # (3,)
        roll_pitch,                  # (2,)
        body_rates,                  # (3,)
        [float(last_collective)],    # (1,)
        sector[tg],                  # (2,)  the CURRENT target's sector
        *slots,                      # 2 x (5,)
    ])
    obs = np.where(np.isfinite(obs), obs, 0.0)         # NaN guard (peregrine_racing_ego.py:759-761)
    return obs.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# PER-GATE AIM OFFSET (2026-07-27) -- the repo form of the pilot's ShadowPC-only ``aim_off`` dodge
# ---------------------------------------------------------------------------------------------
def parse_aim_offsets(spec: str) -> dict[int, tuple[float, float]]:
    """Parse ``"4:0,10;5:0,10"`` -> ``{4: (0.0, 10.0), 5: (0.0, 10.0)}``.

    Grammar: ``gate_index:lateral_m,vertical_m`` groups separated by ``;`` (or whitespace).
    Blank/whitespace-only -> ``{}`` (OFF). Malformed input RAISES -- a mistyped dodge silently
    doing nothing on a flight is worse than a pad abort.

    SIGN CONVENTION (body FLU, measured -- not assumed):
      * ``lateral_m``  +RIGHT  -- the perceived gate moves RIGHT, so the policy aims/passes RIGHT.
        Applied as ``rel_flu[1] (LEFT) -= lateral_m``.
      * ``vertical_m`` +UP     -- the perceived gate moves UP, so the policy aims/passes HIGHER.
        Applied as ``rel_flu[2] (UP) += vertical_m``.

    This is the convention the pilot flew, recovered from the eight ``aim_off`` sessions in
    data/runs by differencing the logged body-FLU lever across the arm/release ticks (attitude is
    constant tick-to-tick, so the jump isolates the injection):

        aim_off        D_fwd    D_left   D_up      (release transition, aim -> None)
        [0.0, -3.0]   +0.202   +0.035   +2.916
        [0.0, +3.0]   -0.144   +0.036   -3.039
        [-3.0, 0.0]   +0.168   -3.026   -0.500
        [0.0, +10.0]  -0.078   -0.213   -9.797
        [0.0, +10.0]  -0.112   -0.177   -9.679
        [0.0, +10.0]  -0.276   -0.058  -10.162
        [0.0, +10.0]  +0.161   -0.039  -10.260

    Two consequences, both load-bearing:

    1. THE VERTICAL SIGN IS OPPOSITE TO ``--ego-gate-z-bias``. That knob adds its value to the
       camera-frame +Y, which is DOWN ("+ lowers the gate"); this one is +UP. ``vertical_m=+10``
       here is roughly ``ego_gate_z_bias = -10.6``, NOT ``+10``. They do not share a sign.
    2. THE FRAME IS BODY FLU, NOT THE CAMERA FRAME. ``frames.R_camera_from_body()`` puts a camera
       +Y offset ``b`` at body FLU ``(+0.342b, 0, -0.940b)`` -- i.e. a 10 m vertical injected at
       ``GateSeeker._valid_poses`` would ALSO move the perceived RANGE by 3.42 m. The flown deltas
       put 0.00 there (|D_fwd| <= 0.28 m, all of it one tick of ownship motion) and the full
       magnitude on UP. So the pilot's dodge is NOT the ``_valid_poses`` injection point and
       cannot be reproduced there.
    """
    out: dict[int, tuple[float, float]] = {}
    if spec is None:
        return out
    for group in str(spec).replace(";", " ").split():
        if not group:
            continue
        head, sep, tail = group.partition(":")
        if not sep:
            raise ValueError(f"aim-offset group {group!r} is missing ':' "
                             "(expected 'gate_index:lateral_m,vertical_m')")
        parts = tail.split(",")
        if len(parts) != 2:
            raise ValueError(f"aim-offset group {group!r} needs exactly two offsets "
                             "'lateral_m,vertical_m'")
        try:
            gate = int(head.strip())
            lat, vert = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(f"aim-offset group {group!r} is not numeric: {exc}") from exc
        if gate < 0:
            raise ValueError(f"aim-offset gate index must be >= 0 (0-based), got {gate}")
        if not (np.isfinite(lat) and np.isfinite(vert)):
            raise ValueError(f"aim-offset group {group!r} has a non-finite offset")
        if gate in out:
            raise ValueError(f"aim-offset gate {gate} listed twice")
        out[gate] = (lat, vert)
    return out


# ---------------------------------------------------------------------------------------------
# The stateful deploy-side builder
# ---------------------------------------------------------------------------------------------
@dataclass
class EgoObsBuilderConfig:
    stale_horizon_s: float = EGO_STALE_HORIZON_S   # confidence decay horizon (champion default 0.5)
    det_hold_s: float = EGO_DET_HOLD_S             # deploy det-proxy hold after loss-of-lock
    obs_coast: bool = False                        # champion = coast OFF (blackout cliff)
    det_geometric: bool = False                    # GEOMETRIC DETECTABILITY (2026-07-27). False
                                                   # (default) == OFF == byte-identical: the geometry
                                                   # is not evaluated at all and no new diag key is
                                                   # emitted. True ANDs training's 8-keypoint
                                                   # ``gate_detectable`` mirror into the det used for
                                                   # obs masking (see gate_detectable_geometric, and
                                                   # the AND-vs-REPLACE rationale in `update`).
    virtual_flip: bool = True                      # tail-first virtual body flip (fly_rl convention)
    slot1_enabled: bool = False                    # WINDOW=2 next-gate slot. OFF (default) = single-slot
                                                   # n_gates=1 path (slot1 zeros; single-gate champions /
                                                   # H6). ON = real 2-slot window fed by update(next_pose=)
                                                   # for the multi-gate _pef generation; a None/uncertain
                                                   # next_pose still masks slot1 to zero (never fabricated).
    sector_mode: str = "auto"                      # 'auto' (first-fix elevation bucket) | 'zero' | 'map'
    coarse_map: "np.ndarray | None" = None         # (G,2) per-gate [horiz,vert] in {-1,0,1}; REQUIRED for
                                                   # sector_mode='map'. horiz -1=next gate RIGHT / +1=LEFT /
                                                   # 0=straight; vert +1=UP / -1=DOWN / 0=level. sector[g] is
                                                   # fed while active_gate_index==g (the multi-gate turn prior
                                                   # the wire cannot compute -- the deploy analog of the STATIC
                                                   # build_coarse_map bucket the _pef champions trained on).
    sector_deadband_rad: float = SECTOR_DEADBAND_RAD
    propagate_gaps: bool = True                    # ego-propagate held rel_pos between fixes
    fix_gain: float = 1.0                          # K in rel_new = (1-K)*propagated_held + K*fix, applied
                                                   # to BOTH slots. 1.0 (default) == the historical SNAP ==
                                                   # byte-identical. Training's estimator low-passed with
                                                   # K = 1/N_eff, N_eff ~ U[4,9] (ego_estimator.py:722-734)
                                                   # -> K ~ 0.154 at the mean; the deploy default is NOT
                                                   # moved there (the pilot passes the value). Clamped to
                                                   # [0,1] like training; K=1 is FORCED on re-acquisition.
    vel_fuse_gain: float = 0.0                     # D1 VISION-REFERENCED LATERAL VELOCITY (2026-07-27).
                                                   # 0.0 (default) == OFF == byte-identical: the fuser is
                                                   # not even constructed and obs[0:3] stays the raw KF
                                                   # velocity. > 0 arms racer.ego_velocity.
                                                   # LateralVelocityFuser, which estimates the DEAD-
                                                   # RECKONING error of obs[0:3] against the tracked gate
                                                   # (a world-fixed landmark) and adds it back, in the
                                                   # LOS-PERPENDICULAR subspace only. See that module's
                                                   # docstring for the observability limits.
    vel_fuse: "VelocityFusionConfig | None" = None  # optional full knob set for the fuser; ``gain`` is
                                                   # always taken from ``vel_fuse_gain`` above.
    # --- PER-GATE AIM OFFSET (2026-07-27) --------------------------------------------------------
    # {gate_index: (lateral_m, vertical_m)} added to the slot-0 lever the policy consumes, in BODY
    # FLU, while ``gate_index`` is active. None/empty (default) == OFF == byte-identical (no dict
    # lookup, no arithmetic, no new key in last_diag). See ``parse_aim_offsets`` for the sign
    # convention and the measurement that fixed it.
    aim_offsets: "dict[int, tuple[float, float]] | None" = None
    # Range (m, TRUE held-belief range EXCLUDING the offset) at/below which the offset is released.
    # 12.0 = the median of the pilot's five hand-timed releases on gate 5 (10.58/11.36/12.63/14.67/
    # 17.20 m); it sits BELOW the 11-16 m obstacle band so the offset is still up while the drone
    # crosses it, and clear of the gate so the last ~12 m are threaded on the honest lever.
    aim_release_m: float = 12.0
    # Width (m) of a linear fade above ``aim_release_m``: the offset scales 1 -> 0 over
    # [release, release+fade]. 0.0 (default) == a HARD STEP == what the pilot actually flew.
    aim_fade_m: float = 0.0


class EgoObsBuilder:
    """Per-flight stateful builder: feed it one ``update(...)`` per control tick, get the 21-dim
    obs. Holds the PER-SLOT track state (held rel_flu, area, last fix + propagation clocks) for the
    WINDOW=2 window [slot0=active gate, slot1=next gate] plus the shared static coarse sector, and
    resets ALL of it on an active-gate-index change (the training window promotion carries no state:
    the new slot0 -- and, when enabled, the new slot1 -- start masked until first acquisition).

    slot0 is always driven by ``update(pose=...)`` (the active gate). When ``slot1_enabled`` the
    next-gate slot is driven by ``update(next_pose=...)``; both slots ego-propagate through fix gaps
    and mask by the SAME staleness/det-hold/conf>0 rule, INDEPENDENTLY. With slot1 disabled (or no
    next_pose ever supplied) the emitted obs is byte-identical to the single-gate deploy."""

    def __init__(self, config: EgoObsBuilderConfig | None = None):
        self.cfg = config or EgoObsBuilderConfig()
        if self.cfg.sector_mode == "map":
            if self.cfg.coarse_map is None:
                raise ValueError(
                    "sector_mode='map' requires coarse_map: an (G,2) per-gate [horiz,vert] bucket "
                    "array in {-1,0,1} (the hand-authored VQ2 turn prior).")
            self._coarse_map = np.asarray(self.cfg.coarse_map, dtype=np.float64).reshape(-1, 2)
            if not np.isin(self._coarse_map, (-1.0, 0.0, 1.0)).all():
                raise ValueError("coarse_map buckets must each be in {-1,0,1}.")
        else:
            self._coarse_map = None
        # D1: the vision-referenced lateral-velocity fuser is only CONSTRUCTED when armed, so the
        # default path runs exactly the code it always ran (byte-identical, no new arithmetic).
        self._vfuse: LateralVelocityFuser | None = None
        if float(self.cfg.vel_fuse_gain) > 0.0:
            vcfg = self.cfg.vel_fuse or VelocityFusionConfig()
            self._vfuse = LateralVelocityFuser(
                VelocityFusionConfig(**{**vcfg.__dict__, "gain": float(self.cfg.vel_fuse_gain)}))
        self._last_fuse_ns: int | None = None
        # PER-GATE AIM OFFSET: normalised once here so the hot path is a single ``is not None``
        # test. Empty/absent -> None -> the tick does exactly the arithmetic it always did.
        self._aim_offsets: dict[int, tuple[float, float]] | None = None
        if self.cfg.aim_offsets:
            self._aim_offsets = {int(g): (float(v[0]), float(v[1]))
                                 for g, v in dict(self.cfg.aim_offsets).items()}
        # GEOMETRIC DETECTABILITY: normalised once so the hot path is a single ``if`` that is False
        # on the default path -- gate_detectable_geometric is then never called and no arithmetic
        # beyond the existing det-proxy comparison runs.
        self._det_geometric = bool(self.cfg.det_geometric)
        self.last_diag: dict = {}
        self._reset_slot()
        self._gate_index: int | None = None

    # -- state management -------------------------------------------------------------------
    def _reset_slot(self) -> None:
        # PER-SLOT held state for the WINDOW=2 window [slot0=active gate, slot1=next gate]. slot1
        # is only maintained/consumed when cfg.slot1_enabled; kept allocated regardless so the state
        # shape is constant. The coarse ``_sector`` is SHARED (the active gate's static turn bucket).
        self._rel_flu: list[np.ndarray | None] = [None, None]   # held drone->gate, body FLU, UNflipped
        self._area: list[float] = [0.0, 0.0]                    # held apparent-area (refreshed on a fix)
        self._last_fix_sim_ns: list[int | None] = [None, None]  # staleness clock anchor, per slot
        self._last_prop_sim_ns: list[int | None] = [None, None] # ego-propagation clock, per slot
        self._sector: tuple[float, float] | None = None         # static per-gate (horiz, vert), shared

    def reset(self) -> None:
        """Full reset (sim epoch restart / new flight)."""
        self._reset_slot()
        self._gate_index = None
        if self._vfuse is not None:
            self._vfuse.reset()
            self._last_fuse_ns = None

    def slot0_hint_frd(self) -> np.ndarray | None:
        """Patch-1 WP1b: the held ACTIVE-gate (slot0) lever as a body-FRD 3-vector [forward, right, down]
        (m), ego-propagated through the current detection gap, for the seeker's re-acquire HINT.

        Returns None when there is no held slot0 position for the current gate (``_rel_flu[0]`` is
        cleared on a gate change / reset) OR the held estimate has decayed past the confidence horizon
        (conf<=0, i.e. age>=stale_horizon_s) -- the SAME liveness the obs masking uses, so the hint is
        trusted only while the builder still treats the held position as live. NEVER fabricates: it only
        surfaces what the builder already holds (the seeker uses it to disambiguate a re-acquisition, not
        to synthesise a fix). Read at the TOP of a control tick (before ``update``), so it reflects the
        previous tick's ego-propagated hold -- one control tick stale, negligible for a bearing prior."""
        rel_flu = self._rel_flu[0]
        if rel_flu is None:
            return None
        # liveness: reuse the confidence computed at the last update (conf>0 == age<stale_horizon_s).
        if float(self.last_diag.get("conf", 0.0) or 0.0) <= 0.0:
            return None
        return _FLIP_FRD_FLU * np.asarray(rel_flu, dtype=np.float64)   # FLU [f,l,u] -> FRD [f,r,d]

    # -- per-tick ---------------------------------------------------------------------------
    def update(self, *, sim_time_ns: int, gate_index: int, R_frd2ned: np.ndarray,
               vel_ned: np.ndarray | None, gyro_frd: np.ndarray | None,
               pose: GatePose | None, last_normed_thrust: float,
               next_pose: GatePose | None = None) -> np.ndarray:
        """One control tick -> the 21-dim float32 obs.

        sim_time_ns   IMU master clock (drives staleness + propagation dt).
        gate_index    RACE_STATUS active_gate_index (slot0 = this gate; slot1 = gate_index+1).
        R_frd2ned     TRUE physical body FRD -> world NED rotation (AHRS).
        vel_ned       KF world velocity (NED); None -> zeros.
        gyro_frd      TRUE FRD body rates (DroneState.gyro_body, wire-sign-corrected); None -> zeros.
        pose          a FRESH quality-gated GatePose for the ACTIVE gate (caller passes it only
                      once per new frame_id), or None. Drives slot0.
        last_normed_thrust  the previous tick's RESCALED normed_thrust in g-units [0, act_max]
                      (policy_step's third return; 0.0 at flight start) -- NOT the wire collective.
        next_pose     a FRESH quality-gated GatePose for the NEXT gate (tg+1), or None. Drives slot1
                      and is consumed ONLY when cfg.slot1_enabled; None (or stale) -> slot1 masks to
                      ZERO. The caller (vision stack) MUST have ASSOCIATED this pose to the tg+1 gate
                      -- training only ever fed the correct next gate or zero, never a wrong gate, so
                      pass None whenever the association is unavailable or uncertain (do NOT fabricate).
        """
        cfg = self.cfg
        t_ns = int(sim_time_ns)

        # -- active-gate change: the window "promotes"; deploy has no next-gate state to carry,
        # so the new slot0 starts cold (masked) until its first accepted fix.
        if self._gate_index is None or int(gate_index) != self._gate_index:
            self._reset_slot()
            self._gate_index = int(gate_index)
            # 'map' mode: latch the hand-authored per-gate turn bucket NOW -- before the first fix,
            # so the anticipation prior is live through the blind approach (training feeds sector[tg]
            # statically regardless of visibility). 'auto' still waits for the first fix (below);
            # 'zero' leaves it None -> (0,0).
            #
            # PAST THE LAST MAPPED GATE -> (0, 0), "I do not know", NOT the last row (2026-07-27).
            # The map is hand-authored from FLOWN gates, so it necessarily ends where our deepest
            # flight ended -- 9 rows today against a 20-gate course. The old behaviour CLAMPED the
            # index, so every gate from the last mapped one onward was fed that row's bucket as if it
            # were a survey result. Today that is harmless only by luck: the last row happens to be
            # [0, 0]. The moment anyone extends the map and its final row carries a real turn, all
            # eleven unmapped gates would silently inherit a CONFIDENT WRONG turn prior -- and this
            # channel is the "where to look before the gate is visible" cue, so a wrong prior delays
            # acquisition on exactly the gates we have never reached. (0, 0) is the honest value: the
            # same thing sector_mode='zero' feeds, and what an unmapped gate means.
            if cfg.sector_mode == "map":
                gi = int(self._gate_index)
                if 0 <= gi < self._coarse_map.shape[0]:
                    self._sector = (float(self._coarse_map[gi, 0]), float(self._coarse_map[gi, 1]))
                else:
                    self._sector = (0.0, 0.0)
            # D1: the landmark changed, so the open measurement window is void. The estimated IMU
            # bias is KEPT (it belongs to the IMU, not to the gate).
            if self._vfuse is not None:
                self._vfuse.on_gate_change()

        # -- frames --------------------------------------------------------------------------
        R = np.asarray(R_frd2ned, dtype=np.float64)
        R_b2w_zup = (_FLIP_FRD_FLU[:, None] * R) * _FLIP_FRD_FLU[None, :]   # FLU -> Z-up world
        v_ned = (np.zeros(3) if vel_ned is None
                 else np.asarray(vel_ned, dtype=np.float64).reshape(3))
        w_frd = (np.zeros(3) if gyro_frd is None
                 else np.asarray(gyro_frd, dtype=np.float64).reshape(3))
        # a transient non-finite KF velocity/rate must not latch NaN into the held rel_pos
        # via the gap propagation below (it would stay poisoned until the next accepted fix)
        v_ned = np.where(np.isfinite(v_ned), v_ned, 0.0)
        w_frd = np.where(np.isfinite(w_frd), w_frd, 0.0)
        v_flu = _FLIP_FRD_FLU * (R.T @ v_ned)      # true body FLU velocity (= R_zup^T v_zup)
        w_flu = _FLIP_FRD_FLU * w_frd              # true body FLU rates
        roll_true, pitch_true = roll_pitch_zup(R_b2w_zup)

        # -- D1 (armed only): roll the vision-referenced velocity-error tracker forward one control
        # tick. It runs on the RAW KF velocity (open loop -- the estimate must not depend on the
        # correction it produces) and on the SAME gyro/dt the lever propagation below uses.
        if self._vfuse is not None:
            _fdt = (0.0 if self._last_fuse_ns is None else (t_ns - self._last_fuse_ns) / 1e9)
            self._last_fuse_ns = t_ns
            self._vfuse.propagate(w_flu, v_flu, _fdt)

        # virtual-flipped (tail-first) quantities the policy consumes
        if cfg.virtual_flip:
            R_obs = R_b2w_zup @ _RZ_PI_BODY
            v_obs = _RZ_PI_BODY @ v_flu
            w_obs = _RZ_PI_BODY @ w_flu
        else:
            R_obs, v_obs, w_obs = R_b2w_zup, v_flu, w_flu
        roll_obs, pitch_obs = roll_pitch_zup(R_obs)

        # -- PER-SLOT ego-propagation + fresh fix. slot0 <- pose (active gate tg); slot1 <- next_pose
        # (tg+1), consumed ONLY when slot1_enabled (n_slots stays 1 otherwise, so slot1 state is never
        # touched and the single-slot path below is byte-identical to the pre-slot1 builder). Both
        # slots use the SAME propagation/fix math; the coarse sector is computed from slot0 ONLY. -----
        n_slots = EGO_WINDOW if cfg.slot1_enabled else 1
        slot_poses = (pose, next_pose)
        pose_seen = [False, False]
        for slot in range(n_slots):
            sp = slot_poses[slot]
            # ego-propagate the held rel_pos through the fix gap (training estimator parity:
            # rel_new = exp(-[w]x dt) @ rel_old - v_body*dt; chaum ego_estimator.py:447-459)
            if (self._rel_flu[slot] is not None and cfg.propagate_gaps
                    and self._last_prop_sim_ns[slot] is not None):
                dt = (t_ns - self._last_prop_sim_ns[slot]) / 1e9
                if dt > 0.0:
                    dR = Rotation.from_rotvec(-w_flu * dt).as_matrix()
                    self._rel_flu[slot] = dR @ self._rel_flu[slot] - v_flu * dt
            self._last_prop_sim_ns[slot] = t_ns
            # fresh fix: BLEND into the (just ego-propagated) held belief + refresh the held area +
            # reset the staleness clock
            seen = sp is not None and np.isfinite(np.asarray(sp.t_cam_gate)).all()
            pose_seen[slot] = seen
            if seen:
                rel_frd = rel_pos_body_frd_from_gatepose(sp.t_cam_gate)
                fix_flu = _FLIP_FRD_FLU * rel_frd
                # D1 (armed only): feed the RAW measurement, never the blended/propagated hold --
                # a hold carries the dead-reckoned velocity inside it and would make the estimate
                # circular. slot0 only (the ACTIVE gate is the landmark the window is anchored to).
                if slot == 0 and self._vfuse is not None:
                    self._vfuse.on_fix(fix_flu)
                # SCALAR-GAIN BLEND (training parity, chaum rl/ego_estimator.py:722-734):
                #   rel_new = rel_held + K * (fix - rel_held) == (1-K)*rel_held + K*fix
                # with rel_held the belief the loop above just ego-propagated to THIS tick. K=1
                # (cfg.fix_gain default) is the historical SNAP and takes the ORIGINAL assignment
                # below verbatim -- no arithmetic runs, so the default path is byte-identical.
                #
                # RE-ACQUISITION SNAP: training forces K=1 when the gate was MASKED before this fix
                # (``reacq = t_since_fix > stale_horizon_s``, ego_estimator.py:730-732) because a
                # prior that coasted past the horizon has drifted and must be discarded -- otherwise
                # confidence would read fresh (1.0) while rel_pos still carried stale-propagation
                # error. Same test here, on the SAME clock: ``_last_fix_sim_ns`` still holds the
                # PREVIOUS fix time at this point (it is rewritten below), so the age computed here
                # is training's ``_t_since_fix`` at the identical point of the update -- after the
                # gap propagation, before the measurement. No held belief at all (cold slot, or the
                # slot was reset by a gate advance) is likewise a snap.
                k = float(np.clip(cfg.fix_gain, 0.0, 1.0))
                held = self._rel_flu[slot]
                age_prior = (float("inf") if self._last_fix_sim_ns[slot] is None
                             else (t_ns - self._last_fix_sim_ns[slot]) / 1e9)
                reacq = (held is None or age_prior > cfg.stale_horizon_s
                         or not bool(np.all(np.isfinite(held))))
                if k >= 1.0 or reacq:
                    self._rel_flu[slot] = fix_flu
                else:
                    self._rel_flu[slot] = (1.0 - k) * held + k * fix_flu
                # visible_area is derived by PROJECTING the gate model through R_cam_gate, so it is
                # only meaningful when the pose carries a real ORIENTATION. An M+1 centre-emit pose
                # with <3 corners has none -- it is synthesised with identity rotation because the
                # regressed centre gives position without orientation -- and identity reads as a
                # perfectly square-on gate, i.e. a FABRICATED ~1.0 area on exactly the cropped gates
                # the centre path recovers. Mask it instead (0.0), which is what this contract has
                # always done for a cropped gate: the partial-rescue path reports n_corners=3 and
                # GateObservation.visible_area_ratio already returns None below 4 corners. Position
                # is still delivered -- only the foreshortening cue is withheld, and withholding it
                # is what training saw. n_corners < 3 uniquely marks the orientation-free poses
                # (real IPPE fits are 4, P3P 3).
                # PREFER the MEASURED corner foreshortening when the seeker carried one
                # (--ego-area-src corners). It comes straight off the observed quad -- no PnP and no
                # range coupling -- whereas the projection below re-derives area from R_cam_gate,
                # the one genuinely ambiguous term of the fit. Measured in flight, the projected
                # form sat at median 0.990 / max 1.000 across 184 ticks while the drone banked
                # 50-61 deg: it reported "head-on" essentially always and told the policy nothing
                # about approach angle, and it correlated 0.44 with distance despite being defined
                # range-free. The two normalisations agree to <=0.073 over 0-60 deg tilt, so this
                # swaps the SOURCE without moving the policy onto a different scale.
                _meas = getattr(sp, "visible_area_meas", None)
                if _meas is not None:
                    self._area[slot] = float(_meas)
                elif int(getattr(sp, "n_corners", 4)) >= 3:
                    self._area[slot] = visible_area_from_gatepose(sp.R_cam_gate, sp.t_cam_gate)
                else:
                    # ORIENTATION-FREE POSE (M+1 centre emit, <3 corners): HOLD the last measured
                    # area -- do NOT write 0.0, and do NOT project through the synthesised identity
                    # rotation (that fabricates ~1.0).
                    #
                    # Writing 0.0 here CAUSED A FLIGHT REGRESSION (2026-07-24). Measured: 0.0% of
                    # close-range ticks fed area==0 on the pnp path vs 24-73% on centre, and the
                    # crashes matched exactly -- area fell 1.0 -> 0.0 at ~2.9 m, never recovered,
                    # and the commanded rates diverged within a few ticks. record9 config scored
                    # 5,5 gates on pnp and 0,0,0,1,0,1,0 on centre. The cause is a TRAINING
                    # MISMATCH, flagged in advance by the vision session: training's shoelace ratio
                    # is UNCLIPPED, so a close square-on gate whose corners have cropped reads ~1.0
                    # there -- never 0.0. Feeding 0.0 tells the policy "edge-on / no opening" at the
                    # exact moment it is committing to the gate, which is out of distribution.
                    #
                    # HOLDING is what the rest of this contract already does through a blackout, and
                    # the held value was MEASURED a few ticks earlier when the corners were still in
                    # frame. Foreshortening changes slowly next to the crop transition, so the stale
                    # value is close to true -- and far closer than either 0.0 or a fabricated 1.0.
                    # (Deliberately no decay: this is a geometry cue, not a freshness one, and
                    # ``confidence`` already carries staleness.)
                    pass
                self._last_fix_sim_ns[slot] = t_ns
                # coarse sector (auto) is the ACTIVE gate's first-fix elevation bucket -- slot0 ONLY.
                if slot == 0 and self._sector is None and cfg.sector_mode == "auto":
                    self._sector = self._compute_sector(self._rel_flu[0], roll_true, pitch_true)

        # -- PER-GATE AIM OFFSET (armed only): the effective (lateral, vertical) for THIS tick.
        # Computed here, AFTER the fuse/propagate and AFTER _compute_sector, so it touches neither
        # the held belief nor the latched sector -- exactly what the pilot's flights show (the
        # logged sector is identical either side of a release, and dropping the offset removes it
        # from a COASTED lever in one tick, which a belief-baked offset could not do).
        aim = self._aim_offset_now(int(gate_index))
        aimed: list[np.ndarray | None] = [None]   # slot0's OFFSET lever, for last_diag (see below)
        geo: list[tuple[bool, int] | None] = [None, None]   # per-slot geometric det, for last_diag
        # built ONCE per tick when armed (it rebuilds two scipy Rotations), still read LIVE so a
        # frames.BORESIGHT calibration edit propagates; None on the default path == not built.
        R_cb_tick = frames.R_camera_from_body() if self._det_geometric else None

        # -- per-slot staleness -> confidence + det proxy + the virtual-flipped rel_pos obs -------
        def _channels(slot: int) -> tuple[np.ndarray, float, bool, float]:
            rel_flu = self._rel_flu[slot]
            if self._last_fix_sim_ns[slot] is None or rel_flu is None:
                age = float("inf")
            else:
                age = max(0.0, (t_ns - self._last_fix_sim_ns[slot]) / 1e9)
            c = float(np.clip(1.0 - age / max(cfg.stale_horizon_s, 1e-9), 0.0, 1.0))
            d = age < cfg.det_hold_s
            # GEOMETRIC DETECTABILITY (armed only) -- ANDed into the det, never substituted for it.
            #
            # WHY *AND* AND NOT *REPLACE*. The age proxy is not merely a bad stand-in for geometry;
            # it also carries the ONE masking event geometry cannot see -- LOSS OF LOCK. Training's
            # ``det`` sits next to an estimator that always has truth available, so geometry alone
            # is a complete answer there. The wire's detector genuinely fails (motion blur, dropped
            # frames, a mis-lock): the gate can be perfectly in frame while nothing has been
            # measured for a second. REPLACING the proxy would delete the blackout cliff at
            # loss-of-lock and feed an indefinitely coasted lever whenever the geometry is happy --
            # strictly worse than today. ANDing can only ever mask MORE, never less, so it adds the
            # missing blind-run-in blackout without opening any new unmasked window. Every masking
            # event that fires today still fires.
            #
            # Evaluated on the HONEST held belief -- BEFORE the aim offset is applied below -- for
            # the same reason ``_aim_offset_now`` measures its release range offset-EXCLUDED: a
            # +3 m vertical dodge would otherwise push the perceived gate out of frame and mask the
            # slot exactly while the dodge is armed, i.e. the knob would silently disable the other.
            if self._det_geometric:
                g = gate_detectable_geometric(rel_flu, R_b2w_zup, R_cam_from_body=R_cb_tick)
                geo[slot] = g
                d = d and g[0]
            # slot0 ONLY -- the pilot's dodge left slot1 untouched (measured: rel_flu1 is continuous
            # across every release), and a knob keyed on the ACTIVE gate index has no business
            # shifting the NEXT gate. Applied BEFORE the virtual flip and BEFORE the confidence
            # mask (a masked slot stays exactly zero).
            if slot == 0 and aim is not None and rel_flu is not None:
                rel_flu = rel_flu + np.array([0.0, -aim[0], aim[1]], dtype=np.float64)
                aimed[0] = rel_flu
            r = (np.zeros(3) if rel_flu is None
                 else (_RZ_PI_BODY @ rel_flu if cfg.virtual_flip else rel_flu))
            return r, c, d, age

        rel0, conf0, det0, age0 = _channels(0)
        sector_row = self._sector if self._sector is not None else (0.0, 0.0)

        # -- D1 (armed only): obs[0:3] = the KF velocity + the vision-referenced LOS-perpendicular
        # correction. Re-derives v_obs from the CORRECTED body-FLU velocity through the identical
        # flip; nothing else in the obs moves (the held lever keeps propagating on the raw KF
        # velocity, exactly as trained).
        if self._vfuse is not None:
            _v_corr = self._vfuse.correct(v_flu)
            v_obs = (_RZ_PI_BODY @ _v_corr) if cfg.virtual_flip else _v_corr

        # -- assemble through the training masking/concat logic. slot1_enabled => a real WINDOW=2
        # window (n_gates=2, both slots masked independently); OFF => the single-gate n_gates=1 path
        # (slot1 window-invalid => zeros), byte-identical to the pre-slot1 builder. -----------------
        if cfg.slot1_enabled:
            rel1, conf1, det1, age1 = _channels(1)
            sector = np.array([sector_row, sector_row], dtype=np.float64)       # (2,2); only [tg=0] used
            obs = ego_actor_obs_np(
                velocity=v_obs, roll_pitch=np.array([roll_obs, pitch_obs]), body_rates=w_obs,
                last_collective=float(last_normed_thrust), sector=sector,
                rel_pos=np.stack([rel0, rel1]), confidence=np.array([conf0, conf1]),
                visible_area=np.array([self._area[0], self._area[1]]),
                detectable=np.array([det0, det1]),
                target_gate=0, n_gates=2, obs_coast=cfg.obs_coast)
        else:
            conf1 = det1 = age1 = None
            sector = np.array([sector_row], dtype=np.float64)                    # (1,2)
            obs = ego_actor_obs_np(
                velocity=v_obs, roll_pitch=np.array([roll_obs, pitch_obs]), body_rates=w_obs,
                last_collective=float(last_normed_thrust), sector=sector,
                rel_pos=rel0.reshape(1, 3), confidence=np.array([conf0]),
                visible_area=np.array([self._area[0]]), detectable=np.array([det0]),
                target_gate=0, n_gates=1,          # n_gates=1 => slot1 window-invalid => zeros,
                obs_coast=cfg.obs_coast)           # EXACTLY the single-gate champion's training state

        # ``rel_flu`` reports the lever the POLICY was fed this tick -- the aim offset included,
        # exactly as the pilot's own build logged it, so his eight aim_off sessions and every
        # flight on this knob stay one comparable corpus (and render_yolo keeps drawing where we
        # actually aimed). ``aim_off`` carries the offset so any consumer can subtract it back out.
        _rel_diag = self._rel_flu[0] if aimed[0] is None else aimed[0]
        self.last_diag = {
            "age_s": age0 if np.isfinite(age0) else None,
            "conf": conf0, "det_proxy": bool(det0), "area": self._area[0],
            "pose_seen": bool(pose_seen[0]),
            "rel_flu": None if _rel_diag is None else _rel_diag.tolist(),
            "sector": list(sector[0]),
            "roll_obs": roll_obs, "pitch_obs": pitch_obs,
            "slot1_enabled": bool(cfg.slot1_enabled),
        }
        if self._aim_offsets is not None:
            # present on EVERY tick once the knob is configured (None while not applying), so the
            # field's presence marks "this flight could have been offset" -- the shape
            # scripts/d1_velocity_replay.py and scripts/failure_profile/* already consume.
            self.last_diag["aim_off"] = None if aim is None else [aim[0], aim[1]]
        if self._det_geometric:
            # Emitted ONLY when the knob is armed, so a default flight's log stays byte-identical
            # and the KEY'S PRESENCE marks "this flight could have been geometrically masked".
            # ``det_geom`` is the geometry alone; the det the obs actually used is
            # ``det_proxy AND det_geom`` (``det_proxy`` deliberately keeps meaning the AGE proxy so
            # the arrestor's det_fresh and every existing log analysis read what they always read).
            # ``det_corners`` is the 0..8 in-frame keypoint count -- the calibration number: it
            # should decay smoothly through the run-in and cross 4 at ~1.8 m.
            self.last_diag["det_geom"] = bool(geo[0][0]) if geo[0] is not None else False
            self.last_diag["det_corners"] = int(geo[0][1]) if geo[0] is not None else 0
        if self._vfuse is not None:
            self.last_diag["vfuse"] = self._vfuse.diag()
        if cfg.slot1_enabled:
            self.last_diag.update({
                "conf1": conf1, "det_proxy1": bool(det1), "area1": self._area[1],
                "age_s1": age1 if np.isfinite(age1) else None,
                "pose_seen1": bool(pose_seen[1]),
                "rel_flu1": None if self._rel_flu[1] is None else self._rel_flu[1].tolist(),
            })
            if self._det_geometric:
                self.last_diag["det_geom1"] = bool(geo[1][0]) if geo[1] is not None else False
                self.last_diag["det_corners1"] = int(geo[1][1]) if geo[1] is not None else 0
        return obs

    # -- per-gate aim offset ------------------------------------------------------------------
    def _aim_offset_now(self, gate_index: int) -> tuple[float, float] | None:
        """The EFFECTIVE (lateral_m, vertical_m) to apply to slot0 this tick, or None.

        Active when (a) the knob is configured, (b) ``gate_index`` has an entry, (c) a slot-0
        belief exists at all -- an offset must SHIFT a real lever, never conjure one out of a
        blank slot -- and (d) the range gate passes.

        RANGE GATE. The obstacle sits ~14.5 m short of the gate but the gate still has to be
        threaded at 0 m, so the offset must come off. What the pilot actually flew (measured over
        all eight sessions) was a ~2.0 s hold from the gate-index change, released BY HAND:
        five of eight ran 1.97-2.00 s and three were cut short (0.47/0.70/0.94 s), so the release
        RANGE scattered over 10.58-17.20 m with no rule behind it. A wall-clock hold does not
        transfer across speeds, so this reproduces the same envelope with the physically
        meaningful quantity instead: release at ``aim_release_m`` (default 12.0 = the median of
        those five releases), hard step by default because a hard step is what flew. Set
        ``aim_fade_m`` > 0 for a linear fade over [release, release+fade] -- gentler, and NEVER
        FLOWN, so treat the first flights on it as a new arm.

        Range is measured on the HELD belief, offset EXCLUDED, so the gate cannot chase its own
        output (a +10 m vertical inflates |rel| by ~4 m at 12 m range -- enough to hold itself on
        for another ~0.4 s of closing)."""
        if self._aim_offsets is None:
            return None
        off = self._aim_offsets.get(int(gate_index))
        if off is None:
            return None
        rel = self._rel_flu[0]
        if rel is None:
            return None
        rng = float(np.linalg.norm(rel))
        if not np.isfinite(rng):
            return None
        rel_m = max(float(self.cfg.aim_release_m), 0.0)
        fade_m = max(float(self.cfg.aim_fade_m), 0.0)
        if rng <= rel_m:
            return None
        scale = 1.0 if fade_m <= 0.0 else min(1.0, (rng - rel_m) / fade_m)
        if scale <= 0.0:
            return None
        return (off[0] * scale, off[1] * scale)

    # -- coarse sector (see module docstring) -------------------------------------------------
    def _compute_sector(self, rel_flu: np.ndarray, roll_true: float,
                        pitch_true: float) -> tuple[float, float]:
        """Static per-gate sector at FIRST acquisition: horiz = 0 (no next-gate leg on the wire;
        identically the single-gate training value), vert = elevation bucket of the leveled
        drone->gate vector (the incoming-leg stand-in), deadband ``sector_deadband_rad``."""
        lv = leveled_from_body(rel_flu, roll_true, pitch_true)
        horiz_mag = float(np.hypot(lv[0], lv[1]))
        elev = float(np.arctan2(lv[2], max(horiz_mag, 1e-9)))
        vert = 0.0
        if elev > self.cfg.sector_deadband_rad:
            vert = 1.0
        elif elev < -self.cfg.sector_deadband_rad:
            vert = -1.0
        return (0.0, vert)
