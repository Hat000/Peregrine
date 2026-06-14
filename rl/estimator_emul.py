"""rl/estimator_emul.py -- NO-RENDER eval-side estimator emulation for inc8 policy SELECTION.

WHY THIS EXISTS. inc8 must SELECT the fastest contact-valid policy on the obs an ESTIMATOR would
actually deliver -- not on perfect pose. Today ``rl/contact_true_eval.py`` flies on
``obs_from_truth`` (zero-noise ground-truth pose), so a camera-pointing policy and a blind one score
identically and the selection crowns a fiction. This module closes that gap WITHOUT a renderer: it
runs an ACTUAL ``racer.state_estimator.LinearKF`` driven by the calibrated ``rl/fix_surrogate.py``
fix model, so camera-pointing -> fix density -> KF accuracy -> the obs the policy is scored on. The
obs seam mirrors the DEPLOY seam (``racer.estimator_obs.estimator_state_for_obs``) EXACTLY: replace
ONLY pos/vel with the KF estimate; keep the trusted given attitude/rates.

WHAT IT IS / IS NOT. A numpy EVAL tool. It is NOT the train env -- it does not touch
``peregrine_racing.py`` or the DiffAero clone. The torch train-env port (the held Deliverable 2) is
gated on the ESCAPE-HATCH PROBE this module enables.

DESIGN CHOICES (each load-bearing; see handoff/p2-inc8-rl-2026-06-14/REPORT.md):
  * COLD prior. The KF inits at the true start pos (``pos_std=1.0``) with a COLD velocity prior
    (``vel_std=5.0``): case-C velocity is observable only through fix-differencing + IMU dead-
    reckoning. At a real episode start the drone is at REST (vel==0==truth), so the cold-ness is in
    the COVARIANCE, not the value.
  * IMU = trusted-with-noise. The predict step integrates a specific force SYNTHESIZED from truth
    (a_world = (v_cur-v_prev)/dt) plus isotropic N(0, sigma_imu). The case-C BINDING error is the
    VISION bias, NOT the IMU -- so the emulator KF runs with ``attitude_noise_std=0`` and
    ``accel_noise_std=sigma_imu``, making P an HONEST match to the injected IMU+fix noise (the
    NEES~3 calibration the confidence channel needs).
  * Per-fix LATERAL sigma DR ~ U[0.05,0.15] m (measured 0.10 supersedes the modeled 0.265): applied
    by ``dataclasses.replace``-ing fix_surrogate's ``sigma_lateral_floor`` per episode, KEEPING its
    range-collapse a1 (and its fitted vertical 0.28 / depth 0.85 floors).
  * ONE-SIGNED in-plane PnP/extrinsic bias, magnitude ~ U[0,0.19] m, per-episode CONSTANT RANDOM
    SIGN, applied to the lateral AND vertical gate-frame axes. This SUPERSEDES d5's zero-mean +-0.10
    -- a one-sided 0.19 m bias is the BINDING case (the conservative case-(b)/OPEN assumption: the
    perception bias SURVIVES the +L gate-relative fix). A later P3 boresight head-on arbiter may
    resolve case-(a)/CLOSE and relax it; this probe is valid under either case (it tests the POINTING
    signal, which the bias does not gate).
  * MAP bias is NOT injected -- it cancels in gate-relative (c1: db drops out, E_bias -0.000);
    injecting it is the forbidden anti-pattern. Here gate_pos IS truth (delta_map~0, discriminator
    d7c592e), so pos_g = R_w2g @ (gate_pos - p_KF) yields GT_lever - kf_residual, the correct
    "GT +L + residual" pattern. The perception bias enters ONLY through fix_surrogate, never a
    separate biased gate_map.

The frame wiring is INHERITED from the canonical ``fly_rl.obs_from_zup`` (substituting KF pos/vel),
so the 17-dim obs is bit-identical to inc7's when the KF tracks truth -- pinned by the FRAME-SEAM
IDENTITY test (tests/test_estimator_emul.py GATE#1). The 20-dim confidence channel [17:20] follows
the FROZEN d5 contract (sigma_ref/sigma_hat ratios + staleness clock).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys

import numpy as np

_RL = Path(__file__).resolve().parent
_SRC = _RL.parent / "src"
for _p in (str(_SRC), str(_RL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fix_surrogate as FS                                              # noqa: E402
from racer.contracts import Gate                                       # noqa: E402
from racer.rl_plant import quat_rotate                                 # noqa: E402
from racer.state_estimator import GRAVITY_NED, LinearKF               # noqa: E402
# Canonical obs builder + frame constants (THE source of truth -- never re-derive the frame math).
from fly_rl import (                                                   # noqa: E402
    N_GATES, _FLIP, _GATE_POS_ZUP, obs_from_zup,
)

# Obs[17:20] FROZEN contract (d5 §1.2; MEMORY OBS CONTRACT 2026-06-13).
SIGMA_REF_M = 0.05          # confidence normalizer: c=1 <=> at-or-better-than the 1-sigma bar
TAU_STALE_S = 0.10          # staleness horizon (~3 ticks @ 30 Hz)

# Gate-4 instrumentation bands (the inc8 binding-gate windows).
GATE4_BAND_M = 28.0         # the fix-offered approach window to gate-4
TERMINAL_LOCK_M = 5.0       # the final approach over which terminal gate-lock is measured
GATE4_IDX = 4


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    """R_world_body (NED) from a wxyz quaternion -- columns are the body axes in world (rl_plant
    convention, torch-free). Identical to offline_rollout._R_from_quat."""
    return np.stack([quat_rotate(q, e) for e in np.eye(3)], axis=-1)


def ned_gate_frame(yaw: float) -> np.ndarray:
    """The contracts.Gate NED gate frame R_world_gate (gate->world NED) for a Z-up through-yaw.

    Columns are the gate axes in world NED: [right (in-plane lateral), down (in-plane vertical),
    downrange (along-track, the through-direction)]:

        R = [[ s, 0,  c],
             [ c, 0, -s],
             [ 0, 1,  0]]   (c=cos yaw, s=sin yaw)

    THIS IS NOT the Z-up obs frame ``_R_W2G = diag(-1,-1,1)`` (footgun): the obs frame has downrange
    as its FIRST axis in Z-up world; the NED Gate convention (contracts.py) has downrange as the
    THIRD axis in world NED. The two are consistent on the OPENING PLANE -- pinned to machine
    precision by the C1 identity (a pure in-plane NED-gate displacement produces ZERO obs along-track
    change, and a pure downrange displacement ZERO obs in-plane change; test_estimator_emul GATE#1).
    The frame's only roles here are (a) shaping the fix covariance / one-signed bias and (b)
    projecting the KF cov into the gate plane for obs[17:20] -- both sign-robust, both using THIS
    same frame, so they are self-consistent."""
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[s, 0.0, c],
                     [c, 0.0, -s],
                     [0.0, 1.0, 0.0]], dtype=np.float64)


def make_ned_gate(gate_id: int, gate_pos_zup: np.ndarray, yaw: float) -> Gate:
    """Build the NED ``racer.contracts.Gate`` for one course gate from its Z-up centre + through-yaw.
    position_ned = gate_pos_zup * _FLIP (Z-up<->NED is involutory); R_world_gate = ned_gate_frame(yaw)."""
    return Gate(gate_id=int(gate_id),
                position_ned=np.asarray(gate_pos_zup, dtype=np.float64) * _FLIP,
                R_world_gate=ned_gate_frame(float(yaw)))


# ============================================================================ config
@dataclass(frozen=True)
class EmulConfig:
    """The case-C estimator-emulation DR + encoding contract (supersedes d5's older values per the
    P2 INC8-RL prompt / MEMORY NOW). All distributions are sampled PER EPISODE in ``reset``."""

    # -- per-fix LATERAL sigma DR (gate-frame, m/axis). Overrides fix_surrogate.sigma_lateral_floor
    #    per episode; KEEPS its range-collapse a1 and its fitted vertical (0.28) / depth (0.85). The
    #    measured 0.10 supersedes the modeled 0.265.
    sigma_lat_lo: float = 0.05
    sigma_lat_hi: float = 0.15

    # -- one-signed PnP/extrinsic bias (the BINDING case-(b) assumption: survives the +L fix).
    #    magnitude ~ U[bias_mag_lo, bias_mag_hi]; a per-episode CONSTANT RANDOM SIGN; the SAME signed
    #    value is applied to the lateral AND vertical in-plane gate-frame axes (depth bias = 0, the
    #    along-track is IMU-owned). Each per-axis bias stays in [0, 0.19] exactly as specified;
    #    inject_bias=False zeroes it (frame-seam / NEES-calibration tests).
    bias_mag_lo: float = 0.0
    bias_mag_hi: float = 0.19
    inject_bias: bool = True

    # -- obs[17:20] encoding (FROZEN d5 §1.2).
    sigma_ref: float = SIGMA_REF_M
    tau_stale: float = TAU_STALE_S

    # -- COLD KF init. pos known at start, velocity cold (fix-differencing-observable).
    pos_std_init: float = 1.0
    vel_std_init: float = 5.0

    # -- IMU predict noise. Trusted-with-noise: small isotropic accel noise, NO attitude-error Q
    #    (case-C binding error is the VISION bias, not the IMU -> honest P for the confidence channel).
    imu_accel_noise: float = 0.3

    def sample_episode_sigma_lat(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.sigma_lat_lo, self.sigma_lat_hi))

    def sample_episode_bias(self, rng: np.random.Generator) -> float:
        """One signed in-plane bias value (applied to lateral AND vertical). 0 if inject_bias is off."""
        if not self.inject_bias:
            return 0.0
        mag = float(rng.uniform(self.bias_mag_lo, self.bias_mag_hi))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return sign * mag


# ============================================================================ instrumentation
@dataclass
class _Trace:
    """Per-step instrumentation (the inc8 selection + escape-hatch-probe signals)."""
    target_gate: list = field(default_factory=list)
    range_target: list = field(default_factory=list)     # range to the CURRENT target gate
    range_gate4: list = field(default_factory=list)       # range to gate-4 (binding gate)
    in_image_target: list = field(default_factory=list)   # gate centre projects in-frame (pointing)
    fix_accepted: list = field(default_factory=list)       # a fix was sampled+applied this step
    speed: list = field(default_factory=list)             # true |v_NED|
    inplane_err: list = field(default_factory=list)        # |e_g in-plane| KF vs truth, gate frame
    along_err: list = field(default_factory=list)          # |e_g along-track| KF vs truth
    c_inplane: list = field(default_factory=list)
    c_along: list = field(default_factory=list)
    age_norm: list = field(default_factory=list)


# ============================================================================ the emulator
class EstimatorEmulator:
    """Holds a LinearKF + a FixSurrogate + per-episode DR draws + a fix-staleness clock.

    Usage (mirrors the contact_true_eval loop):
        emu = EstimatorEmulator(EmulConfig())
        emu.reset(start_state, target_gate, rng)
        # per step, AFTER plant_step advances st_prev -> st_cur:
        emu.step(st_prev, st_cur, target_gate, dt)
        obs = emu.obs(st_cur, target_gate, last_normed, virtual_flip, gate_map, obs_dim)
    """

    def __init__(self, config: EmulConfig | None = None,
                 surrogate: FS.FixSurrogate | None = None,
                 gate_pos_zup: np.ndarray | None = None,
                 gate_yaw: np.ndarray | None = None):
        self.config = config or EmulConfig()
        # DEFAULT surrogate = the baked Track-3 calibration (single swappable checkpoint -- do NOT
        # hardcode sigma anywhere the checkpoint should own; from_checkpoints() recalibrates it).
        self._base_surrogate = surrogate if surrogate is not None else FS.DEFAULT
        self._gate_pos_zup = (np.asarray(gate_pos_zup, dtype=np.float64)
                              if gate_pos_zup is not None else _GATE_POS_ZUP.copy())
        n = len(self._gate_pos_zup)
        # VQ1 course is all-pi (fly_rl: yaw = 3.141592569 ~= pi). Default to pi for every gate.
        self._gate_yaw = (np.asarray(gate_yaw, dtype=np.float64)
                          if gate_yaw is not None else np.full(n, np.pi, dtype=np.float64))
        self.gates: list[Gate] = [make_ned_gate(g, self._gate_pos_zup[g], self._gate_yaw[g])
                                  for g in range(n)]
        # episode state (set in reset)
        self.kf: LinearKF | None = None
        self._surrogate_ep: FS.FixSurrogate | None = None
        self._t_since_fix: float = 0.0
        self._sigma_lat_ep: float = float("nan")
        self._bias_ep: float = float("nan")
        self.trace = _Trace()

    # ---- episode lifecycle --------------------------------------------------
    def reset(self, st0, target_gate: int, rng: np.random.Generator) -> None:
        """Init the KF COLD at truth (pos seeded, velocity cold) and draw the per-episode DR.

        Seeds the KF at (st0.pos, st0.vel): at a real episode start st0.vel==0 (rest), so this is the
        cold prior (uncertainty in the COVARIANCE). For the frame-seam identity test (random st0) the
        seeding makes kf == truth, isolating the wiring."""
        cfg = self.config
        self._sigma_lat_ep = cfg.sample_episode_sigma_lat(rng)
        self._bias_ep = cfg.sample_episode_bias(rng)
        # Per-episode surrogate: override the lateral floor (DR) + the in-plane biases (one-signed),
        # keep the range-collapse a1 + the fitted vertical/depth floors. Depth bias = 0 (IMU-owned).
        self._surrogate_ep = replace(
            self._base_surrogate,
            sigma_lateral_floor=self._sigma_lat_ep,
            sigma_lateral_bias=self._bias_ep,
            sigma_vertical_bias=self._bias_ep,
            sigma_depth_bias=0.0,
        )
        self.kf = LinearKF.initialize(
            np.asarray(st0.pos, dtype=np.float64),
            np.asarray(st0.vel, dtype=np.float64),
            pos_std=cfg.pos_std_init, vel_std=cfg.vel_std_init,
            accel_noise_std=cfg.imu_accel_noise,
            attitude_noise_std=0.0,          # IMU+attitude trusted-with-noise (case-C)
        )
        self._t_since_fix = 1e3              # no fix yet -> age_norm == 1 (cold/stale)
        self.trace = _Trace()

    def seed_truth(self, st) -> None:
        """Force the KF state to truth (pos+vel). The operationalization of a PERFECT zero-noise
        force-accepted estimator -- used by the frame-seam identity test (GATE#1) to pin the wiring
        independent of KF convergence."""
        assert self.kf is not None, "reset() first"
        self.kf.x[:3] = np.asarray(st.pos, dtype=np.float64)
        self.kf.x[3:] = np.asarray(st.vel, dtype=np.float64)

    # ---- the per-step update ------------------------------------------------
    def step(self, st_prev, st_cur, target_gate: int, dt: float,
             rng: np.random.Generator) -> bool:
        """Advance the KF one control step: IMU predict (st_prev->st_cur) then a fix to ``target_gate``.

        Returns whether a fix was accepted. Records per-step instrumentation. ``rng`` drives the
        Bernoulli accept + the noise draw (pass the SAME rng as reset for a reproducible episode)."""
        assert self.kf is not None and self._surrogate_ep is not None, "reset() first"
        dt = float(dt)
        # -- IMU predict: synthesize body specific force from truth, add isotropic accel noise.
        R_prev = _R_from_quat(np.asarray(st_prev.quat, dtype=np.float64))
        if dt > 0.0:
            a_world = (np.asarray(st_cur.vel, dtype=np.float64)
                       - np.asarray(st_prev.vel, dtype=np.float64)) / dt
        else:
            a_world = np.zeros(3)
        accel_body = R_prev.T @ (a_world - GRAVITY_NED)
        accel_body = accel_body + self.config.imu_accel_noise * rng.standard_normal(3)
        self.kf.predict(accel_body, R_prev, dt)
        self._t_since_fix += max(dt, 0.0)

        # -- vision fix to the current target gate (camera-pointing gates acceptance).
        gate = self.gates[target_gate]
        R_cur = _R_from_quat(np.asarray(st_cur.quat, dtype=np.float64))
        geom = FS.geometry(np.asarray(st_cur.pos, dtype=np.float64), R_cur, gate)
        fix = self._surrogate_ep.sample_fix(geom, rng, include_bias=True)
        accepted = fix is not None
        if accepted:
            z, cov = fix
            self.kf.update_position(z, cov)
            self._t_since_fix = 0.0

        self._record(st_cur, target_gate, geom, accepted)
        return accepted

    # ---- obs builders -------------------------------------------------------
    def obs(self, st_cur, target_gate: int, last_normed: float,
            virtual_flip: bool, gate_map, obs_dim: int) -> np.ndarray:
        """The policy obs sourced from the KF estimate (pos/vel) + TRUTH attitude/rates -- the EXACT
        deploy seam (estimator_obs.estimator_state_for_obs). obs_dim 17 -> the inc7 contract verbatim
        (policy ignores [17:20]); obs_dim 20 -> append the d5 confidence triple."""
        assert self.kf is not None, "reset() first"
        R_ned = _R_from_quat(np.asarray(st_cur.quat, dtype=np.float64))
        R_zup = (_FLIP[:, None] * R_ned) * _FLIP[None, :]
        obs17 = obs_from_zup(self.kf.position * _FLIP, self.kf.velocity * _FLIP, R_zup,
                             np.asarray(st_cur.omega, dtype=np.float64) * _FLIP,
                             target_gate, last_normed, virtual_flip=virtual_flip, gate_map=gate_map)
        if obs_dim <= 17:
            return obs17
        triple = self.confidence_channel(target_gate).astype(np.float32)
        return np.concatenate([obs17, triple]).astype(np.float32)

    def confidence_channel(self, target_gate: int) -> np.ndarray:
        """[c_inplane, c_along, age_norm] (d5 §1.2/§2.2) from the CALIBRATED KF gate-frame covariance
        + the staleness clock. sigma_inplane_hat = sqrt((P_E+P_D)/2); sigma_along_hat = sqrt(P_along);
        c = clip(sigma_ref / sigma_hat, 0, 1); age_norm = clip(t_since_fix / TAU_STALE, 0, 1)."""
        assert self.kf is not None
        cfg = self.config
        sig_ip, sig_al = self._gate_frame_sigmas(target_gate)
        c_inplane = float(np.clip(cfg.sigma_ref / sig_ip, 0.0, 1.0)) if sig_ip > 0 else 1.0
        c_along = float(np.clip(cfg.sigma_ref / sig_al, 0.0, 1.0)) if sig_al > 0 else 1.0
        age = float(np.clip(self._t_since_fix / cfg.tau_stale, 0.0, 1.0))
        return np.array([c_inplane, c_along, age], dtype=np.float64)

    # ---- internals ----------------------------------------------------------
    def _gate_frame_sigmas(self, target_gate: int) -> tuple[float, float]:
        """(sigma_inplane_hat, sigma_along_hat): KF position cov projected into the NED gate frame.
        in-plane = RMS of the two opening-plane axis stds; along-track = the through-axis std."""
        Rwg = self.gates[target_gate].R_world_gate
        P_pos = self.kf.P[:3, :3]
        P_gate = Rwg.T @ P_pos @ Rwg
        var_ip = 0.5 * (max(P_gate[0, 0], 0.0) + max(P_gate[1, 1], 0.0))   # (P_E + P_D)/2
        var_al = max(P_gate[2, 2], 0.0)
        return float(np.sqrt(var_ip)), float(np.sqrt(var_al))

    def gate_frame_error(self, st_cur, target_gate: int) -> np.ndarray:
        """KF-vs-truth position error expressed in the NED gate frame: [e_right, e_down, e_along].
        e_inplane = (e_right, e_down) is the binding (miss) error; e_along is phase/timing."""
        Rwg = self.gates[target_gate].R_world_gate
        e_world = self.kf.position - np.asarray(st_cur.pos, dtype=np.float64)
        return Rwg.T @ e_world

    def nees_inplane_along(self, st_cur, target_gate: int) -> tuple[float, float, float]:
        """Per-axis NEES = e_g**2 / diag(P_gate) for (right, down, along). Mean over a rollout batch
        must land in [0.8, 1.3] (NEES~3 over 3 DOF) for the confidence channel to be honest (d5 §2.2;
        validated bias-OFF -- the one-signed bias is an unobservable systematic, not a variance term)."""
        Rwg = self.gates[target_gate].R_world_gate
        e_g = self.gate_frame_error(st_cur, target_gate)
        P_gate = Rwg.T @ self.kf.P[:3, :3] @ Rwg
        d = np.array([P_gate[0, 0], P_gate[1, 1], P_gate[2, 2]], dtype=np.float64)
        d = np.maximum(d, 1e-12)
        nees = (e_g ** 2) / d
        return float(nees[0]), float(nees[1]), float(nees[2])

    def _record(self, st_cur, target_gate: int, geom, accepted: bool) -> None:
        tr = self.trace
        e_g = self.gate_frame_error(st_cur, target_gate)
        conf = self.confidence_channel(target_gate)
        g4_pos = self.gates[GATE4_IDX].position_ned
        tr.target_gate.append(int(target_gate))
        tr.range_target.append(float(geom.range_m))
        tr.range_gate4.append(float(np.linalg.norm(g4_pos - np.asarray(st_cur.pos, dtype=np.float64))))
        tr.in_image_target.append(bool(geom.in_image))
        tr.fix_accepted.append(bool(accepted))
        tr.speed.append(float(np.linalg.norm(np.asarray(st_cur.vel, dtype=np.float64))))
        tr.inplane_err.append(float(np.hypot(e_g[0], e_g[1])))
        tr.along_err.append(float(abs(e_g[2])))
        tr.c_inplane.append(float(conf[0]))
        tr.c_along.append(float(conf[1]))
        tr.age_norm.append(float(conf[2]))

    # ---- instrumentation accessors (the inc8 selection numbers) -------------
    def v_star(self) -> float:
        """|v_NED| at the LAST pre-gate-4 step (the gate-4 approach speed): the speed at the final
        recorded step whose target gate is gate-4. NaN if gate-4 was never the target."""
        tg = np.asarray(self.trace.target_gate)
        sp = np.asarray(self.trace.speed)
        mask = tg == GATE4_IDX
        if not mask.any():
            return float("nan")
        return float(sp[mask][-1])

    def gate4_band_fix_rate(self) -> float:
        """Fraction of steps that landed an accepted fix, over steps targeting gate-4 within the
        GATE4_BAND_M approach window (the gate-4 fix density the policy's pointing earns). NaN if no
        such step occurred."""
        tg = np.asarray(self.trace.target_gate)
        rg = np.asarray(self.trace.range_gate4)
        fx = np.asarray(self.trace.fix_accepted, dtype=bool)
        mask = (tg == GATE4_IDX) & (rg <= GATE4_BAND_M)
        if not mask.any():
            return float("nan")
        return float(fx[mask].mean())

    def terminal_gate_lock_frac(self) -> float:
        """Fraction of the final TERMINAL_LOCK_M m of the gate-4 approach with the gate in-image
        (the TERMINAL gate-lock the margin-closure envelope found load-bearing). NaN if no terminal
        step occurred."""
        tg = np.asarray(self.trace.target_gate)
        rg = np.asarray(self.trace.range_gate4)
        im = np.asarray(self.trace.in_image_target, dtype=bool)
        mask = (tg == GATE4_IDX) & (rg <= TERMINAL_LOCK_M)
        if not mask.any():
            return float("nan")
        return float(im[mask].mean())

    def gate4_inplane_error_series(self) -> np.ndarray:
        """Per-step in-plane (gate-frame) KF error over the gate-4 approach window -- the raw series
        the selection p90/p99 are computed from."""
        tg = np.asarray(self.trace.target_gate)
        rg = np.asarray(self.trace.range_gate4)
        er = np.asarray(self.trace.inplane_err)
        mask = (tg == GATE4_IDX) & (rg <= GATE4_BAND_M)
        return er[mask]

    @property
    def episode_sigma_lat(self) -> float:
        return self._sigma_lat_ep

    @property
    def episode_bias(self) -> float:
        return self._bias_ep
