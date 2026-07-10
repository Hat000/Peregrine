"""Egocentric 21-dim RL observation builder for the VQ2 deploy wire (the fly_ego adapter).

Reproduces, on the live MAVLink wire, the observation contract of the egocentric training env
(chaum worktree ``rl/peregrine_racing_ego.py::ego_actor_obs`` + ``rl/ego_estimator.py``):

  idx    | name              | units / frame
  0:3    | velocity          | m/s, BODY FLU (R_wb^T v_world), virtual-flipped (tail-first)
  3:5    | roll, pitch       | rad, gravity-leveled, NO yaw, of the virtual-flipped body
  5:8    | body_rates        | rad/s, BODY FLU, virtual-flipped
  8      | last_collective   | previous tick's RESCALED normed_thrust, g-units [0, 3.765]
  9:11   | coarse_sector     | (horiz, vert) each in {-1, 0, 1}, gravity-leveled heading frame
  11:14  | slot0 rel_pos     | m, drone->ACTIVE-gate, BODY FLU, virtual-flipped, MASKED to 0
  14     | slot0 confidence  | [0,1] staleness scalar (1 fresh, linear -> 0 at the horizon)
  15     | slot0 visible_area| [0,1] normalized apparent inner-opening area (1 == square-on)
  16:19  | slot1 rel_pos     | PINNED ZEROS (single-gate-trained champion; H6)
  19     | slot1 confidence  | 0
  20     | slot1 visible_area| 0

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
    slot1_enabled: bool = False                    # STUB: future multi-gate policies (H6: keep off)
    sector_mode: str = "auto"                      # 'auto' (first-fix elevation bucket) | 'zero'
    sector_deadband_rad: float = SECTOR_DEADBAND_RAD
    propagate_gaps: bool = True                    # ego-propagate held rel_pos between fixes


class EgoObsBuilder:
    """Per-flight stateful builder: feed it one ``update(...)`` per control tick, get the 21-dim
    obs. Holds the per-gate slot state (last fix, staleness clock, held visible_area, static
    sector) and resets it on an active-gate-index change (the training window promotion has no
    state to carry: the new slot0 starts masked until first acquisition)."""

    def __init__(self, config: EgoObsBuilderConfig | None = None):
        self.cfg = config or EgoObsBuilderConfig()
        if self.cfg.slot1_enabled:
            raise NotImplementedError(
                "slot1 filling is a STUB for future multi-gate policies; the deployed champions "
                "are single-gate-trained (slot1 was zero their whole training life -- H6).")
        self.last_diag: dict = {}
        self._reset_slot()
        self._gate_index: int | None = None

    # -- state management -------------------------------------------------------------------
    def _reset_slot(self) -> None:
        self._rel_flu: np.ndarray | None = None    # held drone->gate, body FLU, UNflipped
        self._area: float = 0.0                    # held apparent-area (refreshed on a fix)
        self._last_fix_sim_ns: int | None = None   # staleness clock anchor
        self._last_prop_sim_ns: int | None = None  # ego-propagation clock
        self._sector: tuple[float, float] | None = None   # static per-gate (horiz, vert)

    def reset(self) -> None:
        """Full reset (sim epoch restart / new flight)."""
        self._reset_slot()
        self._gate_index = None

    # -- per-tick ---------------------------------------------------------------------------
    def update(self, *, sim_time_ns: int, gate_index: int, R_frd2ned: np.ndarray,
               vel_ned: np.ndarray | None, gyro_frd: np.ndarray | None,
               pose: GatePose | None, last_normed_thrust: float) -> np.ndarray:
        """One control tick -> the 21-dim float32 obs.

        sim_time_ns   IMU master clock (drives staleness + propagation dt).
        gate_index    RACE_STATUS active_gate_index (slot0 = this gate ONLY).
        R_frd2ned     TRUE physical body FRD -> world NED rotation (AHRS).
        vel_ned       KF world velocity (NED); None -> zeros.
        gyro_frd      TRUE FRD body rates (DroneState.gyro_body, wire-sign-corrected); None -> zeros.
        pose          a FRESH quality-gated GatePose for the active gate (caller passes it only
                      once per new frame_id), or None.
        last_normed_thrust  the previous tick's RESCALED normed_thrust in g-units [0, act_max]
                      (policy_step's third return; 0.0 at flight start) -- NOT the wire collective.
        """
        cfg = self.cfg
        t_ns = int(sim_time_ns)

        # -- active-gate change: the window "promotes"; deploy has no next-gate state to carry,
        # so the new slot0 starts cold (masked) until its first accepted fix.
        if self._gate_index is None or int(gate_index) != self._gate_index:
            self._reset_slot()
            self._gate_index = int(gate_index)

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

        # -- ego-propagate the held rel_pos through the fix gap (training estimator parity:
        # rel_new = exp(-[w]x dt) @ rel_old - v_body*dt; chaum ego_estimator.py:447-459) --------
        if self._rel_flu is not None and cfg.propagate_gaps and self._last_prop_sim_ns is not None:
            dt = (t_ns - self._last_prop_sim_ns) / 1e9
            if dt > 0.0:
                dR = Rotation.from_rotvec(-w_flu * dt).as_matrix()
                self._rel_flu = dR @ self._rel_flu - v_flu * dt
        self._last_prop_sim_ns = t_ns

        # -- fresh fix: snap (K=1) + refresh the held area + reset the staleness clock ---------
        pose_seen = pose is not None and np.isfinite(np.asarray(pose.t_cam_gate)).all()
        if pose_seen:
            rel_frd = rel_pos_body_frd_from_gatepose(pose.t_cam_gate)
            self._rel_flu = _FLIP_FRD_FLU * rel_frd
            self._area = visible_area_from_gatepose(pose.R_cam_gate, pose.t_cam_gate)
            self._last_fix_sim_ns = t_ns
            if self._sector is None and cfg.sector_mode == "auto":
                self._sector = self._compute_sector(self._rel_flu, roll_true, pitch_true)

        # -- staleness -> confidence + det proxy ----------------------------------------------
        if self._last_fix_sim_ns is None or self._rel_flu is None:
            age_s = float("inf")
        else:
            age_s = max(0.0, (t_ns - self._last_fix_sim_ns) / 1e9)
        conf = float(np.clip(1.0 - age_s / max(cfg.stale_horizon_s, 1e-9), 0.0, 1.0))
        det_proxy = age_s < cfg.det_hold_s

        # -- assemble through the training masking/concat logic --------------------------------
        rel_obs = (np.zeros(3) if self._rel_flu is None
                   else (_RZ_PI_BODY @ self._rel_flu if cfg.virtual_flip else self._rel_flu))
        sector = np.array([self._sector if self._sector is not None else (0.0, 0.0)],
                          dtype=np.float64)                                    # (1,2)
        obs = ego_actor_obs_np(
            velocity=v_obs, roll_pitch=np.array([roll_obs, pitch_obs]), body_rates=w_obs,
            last_collective=float(last_normed_thrust), sector=sector,
            rel_pos=rel_obs.reshape(1, 3), confidence=np.array([conf]),
            visible_area=np.array([self._area]), detectable=np.array([det_proxy]),
            target_gate=0, n_gates=1,          # n_gates=1 => slot1 window-invalid => zeros,
            obs_coast=cfg.obs_coast)           # EXACTLY the single-gate champion's training state

        self.last_diag = {
            "age_s": age_s if np.isfinite(age_s) else None,
            "conf": conf, "det_proxy": bool(det_proxy), "area": self._area,
            "pose_seen": bool(pose_seen),
            "rel_flu": None if self._rel_flu is None else self._rel_flu.tolist(),
            "sector": list(sector[0]),
            "roll_obs": roll_obs, "pitch_obs": pitch_obs,
        }
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
