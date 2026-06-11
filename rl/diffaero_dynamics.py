"""DRAFT -- DiffAero dynamics adapter that injects Peregrine's system-ID'd plant.

================================================================================================
DRAFT -- written against GitHub HEAD of flyingbitac/diffaero (fetched 2026-06-08):
    dynamics/base_dynamics.py, dynamics/quadrotor.py, dynamics/controller.py, cfg/dynamics/quad.yaml.
RECONCILE against the Adroit clone /scratch/network/fl3689/diffaero_repo before training -- the
upstream interface may have moved. Every assumption is tagged ``# RECONCILE`` inline.
This file CANNOT run on the dev laptop (no torch / pytorch3d / diffaero); it targets the Adroit
GPU env. It is parse-checkable (py_compile) and the physics is validated against the parity-tested
numpy reference ``racer.rl_plant`` via :meth:`PeregrinePlantDynamics.check_against_rl_plant`.
================================================================================================

WHAT THIS DOES
    Replaces DiffAero's QuadrotorModel rigid-body+RateController dynamics with our offline twin's
    plant (``racer.rl_plant``, itself parity-tested cell-for-cell against ``racer.twin``). A policy
    trained on these dynamics sees the SAME plant the model-based VQ1 stack was tuned against, so it
    transfers to the sim. We keep DiffAero's *substrate* (vectorised GPU envs, differentiable
    rollout, randomizers, logging) and swap only the dynamics map.

    DiffAero's QuadrotorModel models a full rigid body (mass m, inertia J, drag D, arm length, torque
    constant) with an inner RateController emitting torque. OUR plant abstracts the inner loop as a
    measured first-order rate lag (``rate_gain``, ``rate_tau_s``) + a one-parameter thrust map
    (``hover_thrust``) + world-frame linear drag (``linear_drag``). So we DROP DiffAero's m/J/D/arm_l/
    c_tau and its RateController and integrate ``rl_plant`` instead.

--------------------------------------------------------------------------------------------------
INTERFACE TARGETED (flyingbitac/diffaero @ HEAD 2026-06-08)
--------------------------------------------------------------------------------------------------
  BaseDynamics(cfg: DictConfig, device): sets .n_agents, .n_envs, .dt, .alpha, ._G, ._G_vec=[0,0,-g].
    abstract step(U)->None ; abstract props min_action/max_action, _p,_v,_a,_w,_q ; concrete
    p/v/a/w/q = detached props ; R = quaternion_to_matrix(q.roll(1,-1)) [=> q is XYZW, real-LAST] ;
    body2world=quat_rotate(q,.) ; grad_decay(state) decays gradient by exp(-alpha*dt) per step.
  QuadrotorModel(BaseDynamics): state_dim=13, action_dim=4,
    _state[..., :3]=p, [3:7]=q (XYZW), [7:10]=v, [10:13]=w(body rate) ; _acc separate -> _a ;
    step(U): new=solver(self.dynamics,_state,U,dt,M=n_substeps); renorm quat; grad_decay.
    batch shape: (n_envs, n_agents, ...) but SQUEEZED to (n_envs, ...) when n_agents==1.
  RateController(q_xyzw, w, action)->(thrust,torque): action = [normed_thrust, roll_rate, pitch_rate,
    yaw_rate]; thrust = action[:,0]*thrust_ratio*g*m  (=> normed_thrust == 1.0 is HOVER, since
    compensate_gravity:False in quad.yaml); rates in rad/s, body frame (cfg.action_frame:"body").
  cfg/dynamics/quad.yaml: g=9.81, dt=${env.dt}, n_substeps=1, solver_type=rk4, controller rate
    bounds +-3.14 rad/s, normed_thrust [0,5], thrust_ratio=torque_ratio=1, compensate_gravity:False.

--------------------------------------------------------------------------------------------------
INTERFACE MISMATCHES / ASSUMPTIONS (each tagged ``# RECONCILE`` at its use site)
--------------------------------------------------------------------------------------------------
 1. NUMPY<->TORCH + AUTOGRAD (the big one). ``rl_plant`` is numpy. DiffAero is torch on GPU and its
    edge over Isaac is being DIFFERENTIABLE. Two backends here:
      * backend="rl_plant_numpy" (DEFAULT in this DRAFT): literally delegates to ``rl_plant.step``
        on the CPU. Bit-faithful to the parity-tested plant, but NON-differentiable and slow (host
        round-trip). Use it to (a) smoke-test wiring and (b) VALIDATE the torch port.
      * backend="torch": ``_integrate_torch`` -- a line-for-line torch MIRROR of ``rl_plant.step``
        (GPU + autograd). This is the production path; it must be verified equal to rl_plant on
        Adroit (``check_against_rl_plant``) before trusting it. Until then, treat it as UNVERIFIED.
 2. WORLD FRAME: DiffAero is Z-UP (``_G_vec=[0,0,-g]``); our plant is NED (Z-DOWN, g=+z, thrust=-z
    body). We convert with R_x(pi)=diag(1,-1,-1) on BOTH world and body (NED/FRD <-> NWU/FLU). This
    is a proper rotation that preserves the gravity-down/thrust-up invariants, so the dynamics are
    physically correct in DiffAero's frame regardless of which horizontal axis is "north". # RECONCILE
    the ABSOLUTE horizontal labels ONLY matter if you import a world gate-map into the env -- then
    match the env's actual world axes.
 3. QUATERNION ORDER: DiffAero ``_state[3:7]`` is XYZW (real-last); our plant is WXYZ (real-first).
    Converted at the boundary (roll + the diag(1,-1,-1) similarity for the frame flip).
 4. ACTION SPACE: DiffAero policy action = [normed_thrust, roll_rate, pitch_rate, yaw_rate] (FLU body,
    normed_thrust 1==hover). Mapped to our CTBR [wx,wy,wz (FRD), collective(0..1)] by
    collective = normed_thrust * hover_thrust (=> a_up = g*normed_thrust, == DiffAero accel) and
    body-rate FLU->FRD = diag(1,-1,-1). # RECONCILE the policy's action SCALING/bounds with the
    training cfg (we expose min_action/max_action from cfg.controller, unchanged by the sign-flip).
 5. DOMAIN RANDOMIZATION: DiffAero randomizes m/J/D/arm_l/c_tau via build_randomizer -- we use NONE
    of those (different plant). Randomize OUR params instead -- since the characterize-sweep
    (2026-06-10): super_rate_s, rate_tau, alpha_max (absolute measured bands) + hover/drag
    (fractional), NOT rate_gain (the old asymmetric "+30% band" was the flat-gain model's shadow of
    the unmodeled static map -- disproven; G0 stays fixed at the shipped nominal). # RECONCILE
 6. INTEGRATOR: DiffAero applies solver(euler|rk4) x n_substeps to X_dot. Our plant integrates
    INTERNALLY (semi-implicit Euler) at its own dt; we bypass DiffAero's solver and call rl_plant per
    substep with dt/n_substeps. (rate/thrust lags use exp(-dt/tau), so substepping is consistent.)
 7. ACCELERATION ``_a``: rl_plant doesn't emit accel; we expose world accel by finite-difference
    (v_new - v_old)/dt, converted to DiffAero frame. Good enough for observations; not a true
    accelerometer reading. # RECONCILE if the policy/reward needs specific force.
 8. TRANSPORT DELAY -- two INDEPENDENT mechanisms (do not enable both):
    (a) per-episode latency DR: a per-env action ring buffer applied in ``step()`` (item 4a,
        ``_apply_latency``), randomized delay in {0..max} control steps, active only under DR.
        The integrators never see it, so the parity gate is unaffected.
    (b) ``params.transport_delay_steps`` (rl_plant's internal ring buffer): now mirrored in BOTH
        backends (S12 handoff item 4a) via the persistent ``self._plant_act_buf`` carried across
        ``step()`` calls -- ``_step_numpy`` threads it through rl_plant's own ``PlantState.act_buf``;
        ``_step_torch`` replicates the push/pop per SUBSTEP exactly. Default OFF (0) -- the
        validated config; used for eval-time fixed latency and for the parity gate's
        ``transport_delay_steps>0`` configs. Delay counts SUBSTEPS (rl_step is called per substep).
 9. n_agents: handled via rl_plant's leading-batch convention; we mirror DiffAero's squeeze at
    n_agents==1. State stored in DiffAero layout/frame so _p/_v/_q/_w/R/body2world all work unchanged.
"""
from __future__ import annotations

import numpy as np

# --- guarded imports so this DRAFT is at least importable/parse-checkable off-Adroit ----------------
try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent on the dev laptop
    torch = None
    Tensor = "Tensor"                   # type: ignore  # annotations are strings via __future__

try:
    from diffaero.dynamics.base_dynamics import BaseDynamics   # the class we subclass
except Exception:                       # pragma: no cover - diffaero absent off-Adroit
    BaseDynamics = object               # RECONCILE: real base on Adroit; object lets this file load

# Our parity-tested numpy plant (the physics source of truth). On Adroit, make it importable by
# `pip install -e` of the Peregrine repo, or copy rl_plant.py next to this file. # RECONCILE path.
try:
    from racer.rl_plant import (PlantParams, PlantState, step as rl_step,
                                SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED)
except Exception:                       # pragma: no cover
    PlantParams = PlantState = None     # RECONCILE: ensure racer.rl_plant is on PYTHONPATH on Adroit
    SUPER_RATE_S_MEASURED = 0.30                                  # characterize-sweep nominals
    ALPHA_MAX_RPS2_MEASURED = np.array([260.0, 260.0, 80.0])


# ================================================================================================
# Frame / convention bridges.  R_x(pi) = diag(1,-1,-1) maps NED<->(Z-up) for world AND FRD<->FLU
# for body.  Quaternion (wxyz) under the same flip on both sides: (w,x,y,z) -> (w, x, -y, -z)
# (a similarity by q_flip=(0,1,0,0); it rotates the vector part by diag(1,-1,-1), leaving w).
# ================================================================================================
_FLIP = np.array([1.0, -1.0, -1.0])         # world & body axis flip (NED<->Zup, FRD<->FLU)
_QFLIP = np.array([1.0, 1.0, -1.0, -1.0])   # wxyz under that flip: w,x kept; y,z negated


def _ned_from_diffaero_np(p_d, q_xyzw_d, v_d, w_d):
    """DiffAero state slices (Z-up, XYZW, FLU body) -> our NED/FRD/WXYZ numpy arrays."""
    p = p_d * _FLIP
    v = v_d * _FLIP
    w = w_d * _FLIP                          # body rate FLU -> FRD
    q_wxyz_d = np.concatenate([q_xyzw_d[..., 3:4], q_xyzw_d[..., 0:3]], axis=-1)  # xyzw -> wxyz
    q = q_wxyz_d * _QFLIP                    # frame flip on the orientation
    return p, v, q, w


def _diffaero_from_ned_np(p, v, q_wxyz, w):
    """Our NED/FRD/WXYZ numpy arrays -> DiffAero state slices (Z-up, XYZW, FLU body)."""
    p_d = p * _FLIP
    v_d = v * _FLIP
    w_d = w * _FLIP
    q_wxyz_d = q_wxyz * _QFLIP               # inverse of the flip == the flip (involutory)
    q_xyzw_d = np.concatenate([q_wxyz_d[..., 1:4], q_wxyz_d[..., 0:1]], axis=-1)  # wxyz -> xyzw
    return p_d, v_d, q_xyzw_d, w_d


def _action_diffaero_to_ctbr_np(U):
    """DiffAero action [normed_thrust, roll_rate, pitch_rate, yaw_rate] (FLU) -> our CTBR
    [wx, wy, wz (FRD), collective(0..1)].  collective filled later (needs hover_thrust)."""
    rate_frd = U[..., 1:4] * _FLIP          # FLU body rates -> FRD
    return rate_frd, U[..., 0]              # (rate_frd, normed_thrust)


# ================================================================================================
# Torch quaternion helpers -- a 1:1 MIRROR of racer.rl_plant's numpy helpers (same math, torch ops).
# Kept in sync manually; rl_plant is the parity-tested reference. (DiffAero ships pytorch3d, but we
# port our own to guarantee identical conventions to the verified plant.)
# ================================================================================================
def _t_quat_multiply(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _t_rotvec_to_quat(rotvec):
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * theta
    w = torch.cos(half)
    scale = 0.5 * torch.sinc(half / np.pi)        # == sin(half)/theta, robust at 0
    return torch.cat([w, scale * rotvec], dim=-1)


def _t_quat_rotate(q, v):
    if v.shape != q[..., 1:4].shape:
        v = torch.broadcast_to(v, q[..., 1:4].shape)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = torch.linalg.cross(u, v, dim=-1)
    return v + 2.0 * (w * uv + torch.linalg.cross(u, uv, dim=-1))


def _t_quat_normalize(q, eps=1e-12):
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)


def _t_clip_to_norm(v, max_norm):
    if max_norm <= 0.0:
        return v
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v * torch.where(n > max_norm, max_norm / torch.clamp(n, min=1e-12), torch.ones_like(n))


# ================================================================================================
# The adapter.
# ================================================================================================
class PeregrinePlantDynamics(BaseDynamics):
    """DiffAero dynamics whose map is Peregrine's system-ID'd plant (``racer.rl_plant``).

    Drop-in for ``QuadrotorModel``: same 13-dim ``_state`` layout [p, q(xyzw), v, w] in DiffAero's
    Z-up frame, same action_dim=4 [normed_thrust, roll, pitch, yaw]. Internally converts to NED,
    integrates our plant, converts back. ``backend`` selects rl_plant-numpy (default, verified, slow,
    non-diff) vs torch (production, GPU/diff, MUST be validated via :meth:`check_against_rl_plant`)."""

    def __init__(self, cfg, device, *, backend: str = "rl_plant_numpy",
                 params: "PlantParams | None" = None):
        super().__init__(cfg, device)                       # sets n_agents, n_envs, dt, alpha, _G, _G_vec
        # RECONCILED (vs Adroit clone): report "quadrotor", NOT "peregrine_plant". env/racing.py keys
        # its reset identity-quaternion branch on ``dynamic_type == "quadrotor"`` (racing.py:206) and
        # selects the quadrotor reward/loss via ``isinstance(.., PointMassModelBase)`` (racing.py:289).
        # We ARE a quadrotor dynamics -- just with a system-ID'd inner plant -- so "quadrotor" is correct.
        # build_dynamics still selects THIS class by cfg.dynamics.name == "peregrine_plant" (independent).
        self.type = "quadrotor"
        self.plant_name = "peregrine_plant"
        self.state_dim = 13
        self.action_dim = 4
        self.backend = backend
        self.n_substeps = int(getattr(cfg, "n_substeps", 1))

        # --- OUR plant params (telemetry-free PHYSICS). Default = validated faithful config. ---
        # RECONCILE: optionally read overrides from cfg.peregrine_plant.* ; align g with cfg.g.
        self.params = params or PlantParams()
        self.params.g = float(getattr(cfg, "g", self.params.g))
        # DR (mismatch #5) randomizes OUR params per-env (super_rate_s/rate_tau/alpha_max/hover/
        # drag), NOT DiffAero's m/J/D -- see the DR block below. rate_gain (G0) stays fixed.

        # cached torch params for the torch backend
        if torch is not None:
            self._rate_gain = torch.tensor(self.params.rate_gain, device=device, dtype=torch.float32)
            self._rate_sign = torch.tensor(self.params.rate_sign, device=device, dtype=torch.float32)
            self._BODY_UP = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
            self._g_vec_ned = torch.tensor([0.0, 0.0, self.params.g], device=device, dtype=torch.float32)
            # super-rate map / slew (None -> legacy flat gain / unlimited; see rl_plant docstring)
            self._super_s = (None if self.params.super_rate_s is None else
                             torch.tensor(np.broadcast_to(self.params.super_rate_s, (3,)).copy(),
                                          device=device, dtype=torch.float32))
            self._alpha_max = (None if self.params.alpha_max_rps2 is None else
                               torch.tensor(np.broadcast_to(self.params.alpha_max_rps2, (3,)).copy(),
                                            device=device, dtype=torch.float32))
        # params-level transport-delay ring buffer (mismatch #8b): the NED CTBR action queue carried
        # across step() calls, SHARED by both backends (numpy threads it through rl_plant's
        # PlantState.act_buf; torch mirrors the push/pop). None = cold; rl_plant cold-seeds with the
        # first action, so None here reproduces rl_plant's exact warm-up. Only used when
        # params.transport_delay_steps > 0 (default OFF).
        self._plant_act_buf = None

        # --- state in DiffAero layout/frame; hover init (identity attitude, normed_thrust=1) ---
        bshape = (self.n_envs,) if self.n_agents == 1 else (self.n_envs, self.n_agents)
        self._state = torch.zeros(*bshape, 13, device=device) if torch is not None else None
        if self._state is not None:
            self._state[..., 3] = 0.0       # qx
            self._state[..., 6] = 1.0        # qw (xyzw real-last) -> identity
            # p=0, v=0, w=0 already; thrust state is implicit (collective applied per-step)
        self._acc = torch.zeros(*bshape, 3, device=device) if torch is not None else None
        # realized collective carried for the thrust_tau_s lag (NED-frame scalar per env)
        self._thrust = (torch.full(bshape, float(self.params.hover_thrust), device=device)
                        if torch is not None else None)

        # --- action bounds: mirror RateController (sign-flip leaves symmetric bounds unchanged) ---
        cc = getattr(cfg, "controller", None)
        if torch is not None and cc is not None:
            self._min_action = torch.tensor(
                [cc.min_normed_thrust, cc.min_roll_rate, cc.min_pitch_rate, cc.min_yaw_rate],
                device=device)
            self._max_action = torch.tensor(
                [cc.max_normed_thrust, cc.max_roll_rate, cc.max_pitch_rate, cc.max_yaw_rate],
                device=device)
        elif torch is not None:             # RECONCILE: fall back to our command envelope
            self._min_action = torch.tensor([0.0, -8.0, -8.0, -8.0], device=device)
            self._max_action = torch.tensor([2.0, 8.0, 8.0, 8.0], device=device)

        # --- domain randomization of OUR plant params (mismatch #5), per-env, resampled at reset ----
        # Enabled with `+dynamics.dr=true` (hydra adds the key). Since the characterize-sweep
        # (2026-06-10): the static super-rate map is ALWAYS ON under DR (it IS the measured plant);
        # s / rate_tau / alpha_max take ABSOLUTE measured bands; hover/drag stay fractional;
        # rate_gain (G0) is FIXED at the shipped nominal -- the old asymmetric "+30% rate_gain band"
        # (S1.2 overshoot proxy) was the flat-gain model's shadow of the unmodeled static map and is
        # REPLACED by it. g and thrust_tau_s are held fixed. When DISABLED the validated scalar path
        # in _step_torch runs unchanged -> the check_against_rl_plant gate stays bit-for-bit.
        self._dr_enabled = bool(getattr(cfg, "dr", False))
        self._dr_bands = {
            "s_lo":        float(getattr(cfg, "dr_s_lo", 0.25)),           # super-rate s (absolute;
            "s_hi":        float(getattr(cfg, "dr_s_hi", 0.35)),           #  measured 0.30 +- ~0.02)
            "rate_tau_lo": float(getattr(cfg, "dr_rate_tau_lo", 0.015)),   # seconds (absolute;
            "rate_tau_hi": float(getattr(cfg, "dr_rate_tau_hi", 0.030)),   #  tau_eq 25-33 ms sat.)
            "alpha_lo":    float(getattr(cfg, "dr_alpha_max_lo", 200.0)),  # rad/s^2 roll/pitch
            "alpha_hi":    float(getattr(cfg, "dr_alpha_max_hi", 320.0)),  #  (yaw scales by 80/260)
            "hover":       float(getattr(cfg, "dr_hover_frac", 0.05)),     # +- 5%
            "drag":        float(getattr(cfg, "dr_drag_frac", 0.30)),      # +-30%
        }
        # DR nominal for alpha_max (the yaw column scales with it): the params' value if map-ON
        # params were passed, else the measured nominal [260, 260, 80].
        self._alpha_nom = np.broadcast_to(
            ALPHA_MAX_RPS2_MEASURED if self.params.alpha_max_rps2 is None
            else self.params.alpha_max_rps2, (3,)).astype(np.float64)
        if torch is not None and self._dr_enabled:
            n = self.n_envs
            self._dr_s = torch.full((n, 3), float(SUPER_RATE_S_MEASURED), device=device)
            self._dr_alpha_max = (torch.tensor(self._alpha_nom, device=device,
                                               dtype=torch.float32).unsqueeze(0).expand(n, 3).clone())
            self._dr_hover = torch.full((n,), float(self.params.hover_thrust), device=device)
            self._dr_drag = torch.full((n,), float(self.params.linear_drag), device=device)
            self._dr_rate_tau = torch.full((n,), float(self.params.rate_tau_s), device=device)

        # --- control-latency DR (item 4a): a per-env action RING BUFFER. The policy's command is
        # delayed by a per-episode integer number of control steps (resampled at reset) to model the
        # measured ~40 ms transport lag and randomize over the unmeasurable VQ1->VQ2 latency delta.
        # CRITICAL (parity): the delay is applied in `step()` (a transport wrapper) -- NOT inside
        # `_step_torch`/`_step_numpy`. `check_against_rl_plant` calls those backends DIRECTLY, so it
        # never sees the buffer => the 4.4e-16 parity gate is preserved bit-for-bit. The buffer is a
        # pure pass-through when latency is OFF or a given env's delay == 0. Active only when DR is on
        # AND max>0. At env.dt=0.02 s, {0,1,2} steps span 0/20/40 ms (~the measured 40 ms lag).
        # Characterize-sweep 2026-06-10: the measured INPUT delay is 5-15 ms -- one control step
        # (33 ms @ 30 Hz, 20 ms @ 50 Hz) already covers it, so keep delay HERE (transport mechanism)
        # and do NOT double-count it into rate_tau (whose DR band is the loop dynamics, not delay).
        # NOTE: actions are detached into the buffer -- correct for PPO (no pathwise grad through the
        # env); a BPTT algo (SHAC/APG) with latency-DR would need a differentiable buffer instead.
        self._latency_max = int(getattr(cfg, "dr_latency_max_steps", 2))   # delay in {0..max} steps
        self._latency_enabled = self._dr_enabled and self._latency_max > 0
        if torch is not None and self._latency_enabled:
            n, K = self.n_envs, self._latency_max
            buf = torch.zeros(n, K + 1, self.action_dim, device=device)    # [:, 0] = newest
            buf[..., 0] = 1.0                                              # hover (normed_thrust=1)
            self._act_buf = buf
            self._latency_steps = torch.randint(0, K + 1, (n,), device=device)  # per-env delay

    # ---- abstract API ----------------------------------------------------------------------------
    @property
    def min_action(self) -> Tensor: return self._min_action
    @property
    def max_action(self) -> Tensor: return self._max_action

    @property
    def _p(self) -> Tensor: return self._state[..., 0:3]
    @property
    def _q(self) -> Tensor: return self._state[..., 3:7]     # XYZW (DiffAero convention)
    @property
    def _v(self) -> Tensor: return self._state[..., 7:10]
    @property
    def _w(self) -> Tensor: return self._state[..., 10:13]
    @property
    def _a(self) -> Tensor: return self._acc

    # ---- step --------------------------------------------------------------------------------------
    def step(self, U: Tensor) -> None:
        """Advance all envs one control dt under action ``U`` (..., 4). Updates ``self._state``."""
        if self._latency_enabled:
            U = self._apply_latency(U)          # item 4a: per-env transport delay (parity-safe wrapper)
        if self.backend == "rl_plant_numpy":
            self._step_numpy(U)
        elif self.backend == "torch":
            self._step_torch(U)
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

    def _apply_latency(self, U: Tensor) -> Tensor:
        """Push the newest action to the front of the ring buffer, drop the oldest, and return each
        env's action delayed by its per-episode ``_latency_steps`` count. delay==0 -> returns U as-is
        (the front of the buffer). Detached: the delayed command feeds the plant but carries no
        policy gradient (PPO-appropriate; see __init__ note)."""
        self._act_buf = torch.cat([U.detach().unsqueeze(1), self._act_buf[:, :-1, :]], dim=1)
        idx = self._latency_steps.view(-1, 1, 1).expand(-1, 1, self.action_dim)
        return torch.gather(self._act_buf, 1, idx).squeeze(1)

    # ---- backend A: literal delegation to racer.rl_plant (verified, CPU, NON-differentiable) ------
    def _step_numpy(self, U: Tensor) -> None:
        """Bit-faithful to the parity-tested plant; detaches (mismatch #1). For wiring + validation."""
        dev = self._state.device
        st = self._state.detach().cpu().numpy()
        p_d, q_d, v_d, w_d = st[..., 0:3], st[..., 3:7], st[..., 7:10], st[..., 10:13]
        p, v, q, w = _ned_from_diffaero_np(p_d, q_d, v_d, w_d)
        rate_frd, normed_thrust = _action_diffaero_to_ctbr_np(U.detach().cpu().numpy())
        collective = normed_thrust * self.params.hover_thrust          # normed 1 -> hover (mismatch #4)
        action = np.concatenate([rate_frd, collective[..., None]], axis=-1)

        thrust0 = self._thrust.detach().cpu().numpy()
        # params-level transport delay (mismatch #8b): thread the persistent ring buffer through
        # rl_plant's own PlantState.act_buf (None = cold; rl_step seeds it from the first action).
        buf0 = (None if self._plant_act_buf is None
                else self._plant_act_buf.detach().cpu().numpy().astype(np.float64))
        ps = PlantState(pos=p.copy(), vel=v.copy(), quat=q.copy(), omega=w.copy(),
                        thrust=thrust0.copy(), act_buf=buf0)
        v_prev = v.copy()
        sub_dt = self.dt / self.n_substeps
        for _ in range(self.n_substeps):                              # mismatch #6
            ps = rl_step(ps, action, sub_dt, self.params)
        acc_ned = (ps.vel - v_prev) / self.dt                         # mismatch #7 (finite-diff accel)

        p_d2, v_d2, q_d2, w_d2 = _diffaero_from_ned_np(ps.pos, ps.vel, ps.quat, ps.omega)
        acc_d = acc_ned * _FLIP
        new = np.concatenate([p_d2, q_d2, v_d2, w_d2], axis=-1)
        self._state = torch.from_numpy(new).to(dev).to(self._state.dtype)
        self._acc = torch.from_numpy(acc_d).to(dev).to(self._acc.dtype)
        self._thrust = torch.from_numpy(np.asarray(ps.thrust)).to(dev).to(self._thrust.dtype)
        self._plant_act_buf = (None if ps.act_buf is None else
                               torch.from_numpy(np.ascontiguousarray(ps.act_buf)).to(dev)
                               .to(self._state.dtype))
        # NOTE: grad_decay intentionally skipped -- this path is non-differentiable. # RECONCILE

    # ---- backend B: torch-native MIRROR of rl_plant.step (GPU + autograd; UNVERIFIED until checked)-
    def _step_torch(self, U: Tensor) -> None:
        p_d, q_d, v_d, w_d = (self._state[..., 0:3], self._state[..., 3:7],
                              self._state[..., 7:10], self._state[..., 10:13])
        # diffaero -> NED (torch mirror of _ned_from_diffaero_np)
        flip = torch.tensor(_FLIP, device=self._state.device, dtype=self._state.dtype)
        qflip = torch.tensor(_QFLIP, device=self._state.device, dtype=self._state.dtype)
        p = p_d * flip
        v = v_d * flip
        w = w_d * flip
        q = torch.cat([q_d[..., 3:4], q_d[..., 0:3]], dim=-1) * qflip   # xyzw->wxyz, frame flip
        rate_frd = U[..., 1:4] * flip
        sub_dt = self.dt / self.n_substeps

        # --- param sources: per-env DR tensors (training) OR the validated scalars (gate/eval). The
        # scalar branch is identical to rl_plant op-for-op (incl. map-ON params), so
        # check_against_rl_plant stays bit-for-bit. Under DR the super-rate map + slew are ALWAYS on
        # (they ARE the measured plant; s/alpha_max are the randomized quantities, G0 is fixed). ---
        if self._dr_enabled:
            hover = self._dr_hover                                               # (n_envs,)
            drag = self._dr_drag.unsqueeze(-1)                                   # (n_envs, 1)
            alpha = (1.0 - torch.exp(-sub_dt / self._dr_rate_tau)).unsqueeze(-1) # (n_envs, 1)
            super_s = self._dr_s                                                 # (n_envs, 3)
            alpha_max = self._dr_alpha_max                                       # (n_envs, 3)
        else:
            hover = self.params.hover_thrust                                     # float
            drag = self.params.linear_drag                                      # float
            alpha = 1.0 - np.exp(-sub_dt / max(self.params.rate_tau_s, 1e-9))    # float (const over substeps)
            super_s = self._super_s                                              # (3,) or None (legacy)
            alpha_max = self._alpha_max                                          # (3,) or None (legacy)
        collective = U[..., 0] * hover                                           # (n_envs,)

        # params-level transport delay (mismatch #8b): mirror rl_plant's ring buffer EXACTLY --
        # the post-mapping NED CTBR action is pushed per SUBSTEP; the oldest queued one applies.
        k = int(self.params.transport_delay_steps)
        action_ned = (torch.cat([rate_frd, collective.unsqueeze(-1)], dim=-1) if k > 0 else None)

        thrust = self._thrust
        v_prev = v
        for _ in range(self.n_substeps):
            if k > 0:
                buf = self._plant_act_buf
                if buf is None:                                  # cold buffer -> seed with this action
                    buf = action_ned[..., None, :].expand(
                        *action_ned.shape[:-1], k, 4).clone()
                applied = buf[..., 0, :]                         # oldest queued command
                self._plant_act_buf = torch.cat([buf[..., 1:, :], action_ned[..., None, :]], dim=-2)
                cmd_rate, cmd_coll = applied[..., 0:3], applied[..., 3]
            else:
                cmd_rate, cmd_coll = rate_frd, collective
            # inner rate loop: flat gain (legacy) or the static super-rate map; optional slew clamp.
            if super_s is not None:
                gain = self._rate_gain / (1.0 - super_s * torch.clamp(cmd_rate.abs(), max=np.pi) / np.pi)
            else:
                gain = self._rate_gain
            target = gain * self._rate_sign * cmd_rate
            domega = alpha * (target - w)
            if alpha_max is not None:
                lim = alpha_max * sub_dt
                domega = torch.clamp(domega, -lim, lim)
            w = _t_clip_to_norm(w + domega, self.params.max_omega_rps)
            q = _t_quat_normalize(_t_quat_multiply(q, _t_rotvec_to_quat(w * sub_dt)))
            if self.params.thrust_tau_s > 0.0:
                beta = 1.0 - np.exp(-sub_dt / self.params.thrust_tau_s)
                thrust = thrust + beta * (cmd_coll - thrust)
            else:
                thrust = torch.broadcast_to(cmd_coll, thrust.shape)
            a_up = self.params.g * thrust / hover
            f_world = a_up.unsqueeze(-1) * _t_quat_rotate(q, self._BODY_UP)
            f_world = f_world - drag * v
            accel = f_world + self._g_vec_ned
            v = v + accel * sub_dt
            p = p + v * sub_dt
        acc_ned = (v - v_prev) / self.dt

        # NED -> diffaero (torch mirror of _diffaero_from_ned_np)
        q_wxyz_d = q * qflip
        q_xyzw_d = torch.cat([q_wxyz_d[..., 1:4], q_wxyz_d[..., 0:1]], dim=-1)
        new = torch.cat([p * flip, q_xyzw_d, v * flip, w * flip], dim=-1)
        self._thrust = thrust
        self._acc = acc_ned * flip
        self._state = self.grad_decay(new)                  # differentiable: keep DiffAero's grad decay

    # ---- reset / validation ----------------------------------------------------------------------
    def detach(self) -> None:
        """Detach carried state from the autograd graph between rollouts. BaseDynamics.detach only
        detaches _state; mirror QuadrotorModel and also detach our extra _acc / _thrust state (and
        the params-level delay buffer, whose entries are pushed grad-carrying for BPTT)."""
        super().detach()
        self._acc = self._acc.detach()
        self._thrust = self._thrust.detach()
        if self._plant_act_buf is not None:
            self._plant_act_buf = self._plant_act_buf.detach()

    def reset_idx(self, env_idx) -> None:
        """RECONCILED (vs Adroit clone): reset ONLY our carried aux state (_acc, _thrust), OUT-OF-PLACE.
        env/racing.py BaseEnv.reset_idx has ALREADY written self._state[env_idx] with the reset pose +
        identity quaternion (racing.py:203-208, via torch.where -- enabled by our type=="quadrotor"),
        so we must NOT clobber _state here. We use torch.where (not in-place) because mid-rollout these
        tensors are non-leaf/grad-bearing -- exactly why QuadrotorModel.reset_idx does the same."""
        amask = torch.zeros_like(self._acc, dtype=torch.bool)
        amask[env_idx] = True
        self._acc = torch.where(amask, 0.0, self._acc)
        if self._dr_enabled:
            self._resample_dr(env_idx)        # new per-env plant params for the reset envs
            hover = self._dr_hover            # (n_envs,) -- new per-env hover collective
        else:
            hover = float(self.params.hover_thrust)
        tmask = torch.zeros_like(self._thrust, dtype=torch.bool)
        tmask[env_idx] = True
        self._thrust = torch.where(tmask, hover, self._thrust)
        # params-level transport-delay buffer (mismatch #8b): re-seed the reset envs' queue with the
        # hover command [0,0,0,hover] -- the exact analog of PlantState.hover's act_buf seeding.
        # torch.where (out-of-place) because the buffer can carry grads mid-rollout.
        if self._plant_act_buf is not None and env_idx.numel() > 0:
            hov = (hover if torch.is_tensor(hover) else
                   torch.full((self.n_envs,), float(hover), device=self._plant_act_buf.device))
            hov_ned = torch.zeros_like(self._plant_act_buf[:, :1, :])          # (n, 1, 4)
            hov_ned[..., 3] = hov.to(hov_ned.dtype).view(-1, 1)
            bmask = torch.zeros(self.n_envs, 1, 1, dtype=torch.bool,
                                device=self._plant_act_buf.device)
            bmask[env_idx] = True
            self._plant_act_buf = torch.where(bmask, hov_ned, self._plant_act_buf)
        if self._latency_enabled and env_idx.numel() > 0:
            # new per-episode delay + flush the buffer to hover for the reset envs (no stale commands)
            self._latency_steps[env_idx] = torch.randint(
                0, self._latency_max + 1, (env_idx.numel(),), device=self._latency_steps.device)
            self._act_buf[env_idx] = 0.0
            self._act_buf[env_idx, :, 0] = 1.0

    def _resample_dr(self, env_idx) -> None:
        """Resample per-env plant params for the reset envs: super-rate s / rate_tau / alpha_max
        uniform within their ABSOLUTE measured bands, hover/drag within the fractional ones.
        rate_gain (G0) is deliberately NOT resampled -- fixed at the shipped nominal (the old
        "+30% band" is superseded by the static map; characterize-sweep 2026-06-10)."""
        m = int(env_idx.numel())
        if m == 0:
            return
        dev = self._dr_s.device
        b = self._dr_bands

        def u(shape, frac):                   # symmetric uniform multiplier in [1-frac, 1+frac]
            return 1.0 + frac * (2.0 * torch.rand(*shape, device=dev) - 1.0)

        def u_abs(shape, lo, hi):             # absolute uniform in [lo, hi]
            return lo + (hi - lo) * torch.rand(*shape, device=dev)

        self._dr_s[env_idx] = u_abs((m, 3), b["s_lo"], b["s_hi"])
        # alpha_max: roll/pitch sampled in the absolute [alpha_lo, alpha_hi] rad/s^2 band; the yaw
        # column scales by its nominal ratio (80/260 by default) so every axis randomizes with the
        # same relative width around its own measured nominal.
        amax = u_abs((m, 3), b["alpha_lo"], b["alpha_hi"])
        amax[:, 2] = amax[:, 2] * float(self._alpha_nom[2] / self._alpha_nom[0])
        self._dr_alpha_max[env_idx] = amax
        self._dr_hover[env_idx] = float(self.params.hover_thrust) * u((m,), b["hover"])
        self._dr_drag[env_idx] = float(self.params.linear_drag) * u((m,), b["drag"])
        self._dr_rate_tau[env_idx] = u_abs((m,), b["rate_tau_lo"], b["rate_tau_hi"])

    def check_against_rl_plant(self, U: Tensor, atol: float = 1e-5) -> float:
        """Validate the torch backend against the parity-tested numpy plant from the CURRENT state.
        ``U`` is one action ``(..., 4)`` or a stack ``(T, ..., 4)``: both backends are stepped
        through the whole stack from an identical start -- including an identical copy of the
        params-level transport-delay buffer, so ``transport_delay_steps > 0`` configs are genuinely
        exercised (S12 handoff item 4a) -- and EVERY intermediate state is compared. Returns the max
        abs state divergence over the trajectory (also asserts < atol). Run on Adroit before
        trusting backend='torch'. (Delegates to ``rl_plant`` -- the source of truth.)"""
        Us = U if U.dim() == self._state.dim() + 1 else U.unsqueeze(0)
        snap = self._state.clone(); thr = self._thrust.clone()
        abuf = None if self._plant_act_buf is None else self._plant_act_buf.clone()
        # numpy reference trajectory
        self.backend = "rl_plant_numpy"
        refs = []
        for u in Us:
            self._step_numpy(u)
            refs.append(self._state.detach().cpu().numpy().copy())
        # torch candidate from the identical start
        self._state = snap.clone(); self._thrust = thr.clone()
        self._plant_act_buf = None if abuf is None else abuf.clone()
        self.backend = "torch"
        div = 0.0
        for u, ref in zip(Us, refs):
            self._step_torch(u)
            div = max(div, float(np.max(np.abs(self._state.detach().cpu().numpy() - ref))))
        assert div < atol, f"torch backend diverges from rl_plant: {div:.2e} >= {atol:.0e}"
        return div


# RECONCILE: register with DiffAero's dynamics factory (e.g. add to dynamics/__init__.py's name->class
# map, or pass --dynamics=peregrine_plant) and add a cfg/dynamics/peregrine_plant.yaml carrying
# n_envs/n_agents/dt/alpha/g/n_substeps + a controller block for the action bounds. See quad.yaml.
