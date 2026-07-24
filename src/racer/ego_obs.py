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
  * between accepted fixes (age < det_hold) the held rel_pos is EGO-PROPAGATED exactly like the
    training estimator (rotate by -body_rate*dt, translate by -v_body*dt;
    ego_estimator.py:447-459) so the intra-gap obs stays coherent.

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

ADDITIVE + OPT-IN: nothing imports this module on the default fly_rl paths; VQ1/inc7/gate-seeker
stay byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from racer import frames
from racer.contracts import GatePose
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
# The stateful deploy-side builder
# ---------------------------------------------------------------------------------------------
@dataclass
class EgoObsBuilderConfig:
    stale_horizon_s: float = EGO_STALE_HORIZON_S   # confidence decay horizon (champion default 0.5)
    det_hold_s: float = EGO_DET_HOLD_S             # deploy det-proxy hold after loss-of-lock
    obs_coast: bool = False                        # champion = coast OFF (blackout cliff)
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
            # 'zero' leaves it None -> (0,0). Past the last mapped gate -> clamp to the last row.
            if cfg.sector_mode == "map":
                gi = int(np.clip(self._gate_index, 0, self._coarse_map.shape[0] - 1))
                self._sector = (float(self._coarse_map[gi, 0]), float(self._coarse_map[gi, 1]))

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
            # fresh fix: snap (K=1) + refresh the held area + reset the staleness clock
            seen = sp is not None and np.isfinite(np.asarray(sp.t_cam_gate)).all()
            pose_seen[slot] = seen
            if seen:
                rel_frd = rel_pos_body_frd_from_gatepose(sp.t_cam_gate)
                self._rel_flu[slot] = _FLIP_FRD_FLU * rel_frd
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

        # -- per-slot staleness -> confidence + det proxy + the virtual-flipped rel_pos obs -------
        def _channels(slot: int) -> tuple[np.ndarray, float, bool, float]:
            rel_flu = self._rel_flu[slot]
            if self._last_fix_sim_ns[slot] is None or rel_flu is None:
                age = float("inf")
            else:
                age = max(0.0, (t_ns - self._last_fix_sim_ns[slot]) / 1e9)
            c = float(np.clip(1.0 - age / max(cfg.stale_horizon_s, 1e-9), 0.0, 1.0))
            d = age < cfg.det_hold_s
            r = (np.zeros(3) if rel_flu is None
                 else (_RZ_PI_BODY @ rel_flu if cfg.virtual_flip else rel_flu))
            return r, c, d, age

        rel0, conf0, det0, age0 = _channels(0)
        sector_row = self._sector if self._sector is not None else (0.0, 0.0)

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

        self.last_diag = {
            "age_s": age0 if np.isfinite(age0) else None,
            "conf": conf0, "det_proxy": bool(det0), "area": self._area[0],
            "pose_seen": bool(pose_seen[0]),
            "rel_flu": None if self._rel_flu[0] is None else self._rel_flu[0].tolist(),
            "sector": list(sector[0]),
            "roll_obs": roll_obs, "pitch_obs": pitch_obs,
            "slot1_enabled": bool(cfg.slot1_enabled),
        }
        if cfg.slot1_enabled:
            self.last_diag.update({
                "conf1": conf1, "det_proxy1": bool(det1), "area1": self._area[1],
                "age_s1": age1 if np.isfinite(age1) else None,
                "pose_seen1": bool(pose_seen[1]),
                "rel_flu1": None if self._rel_flu[1] is None else self._rel_flu[1].tolist(),
            })
        return obs

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
