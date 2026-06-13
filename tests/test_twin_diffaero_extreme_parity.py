"""REGRESSION: twin/rl_plant <-> DiffAero adapter step parity at EXTREME states, anchored to an
EXTERNAL world-NED invariant measured from REAL live flight telemetry.

================================================================================================
BUG CLASS THIS GUARDS (one line):
  A frame-convention / sign-alias error in the DiffAero<->NED bridge of ``rl.diffaero_dynamics``
  (the R_y(pi) ODOMETRY-quat conjugation, a body<->world velocity-frame mix, or a FLU<->FRD /
  Z-up<->Z-down axis-flip sign) that silently rotates the trained plant relative to the deployed
  one -- a train<->eval plant divergence that nominal-state, internal-consistency parity tests miss.
================================================================================================

WHY INTERNAL-CONSISTENCY CHECKS MISS IT (the banked methodology, this project bitten 4x):
  A proper-rotation conjugation or a self-inverse axis-flip is INVISIBLE to internal-consistency
  checks. Concretely for this adapter:
    * ``_ned_from_diffaero_np`` and ``_diffaero_from_ned_np`` use the SAME involutory ``_QFLIP`` /
      ``_FLIP``, so a pure round-trip (NED -> diffaero -> NED) returns the input EXACTLY for ANY
      self-inverse flip -- including a WRONG one. A round-trip identity test proves nothing.
    * The torch backend (``_step_torch``) and the numpy backend (``_step_numpy``) share the same
      bridge constants, so torch<->numpy parity ALSO stays bit-identical under a wrong-but-shared
      convention. The 4.4e-16 ``check_against_rl_plant`` gate is necessary but NOT sufficient.
    * quat-vs-rate finite-difference, level-flight correlation, twist round-trip -- all PASS while
      the convention is wrong (a level hover looks identical to its conjugate; the body z-axis is
      aligned with world-down so the tilt-sensitive East term vanishes).

  ONLY an EXTERNAL invariant evaluated at a TILTED phase discriminates. The canonical anchor
  (handoff/laptop-frame-audit-2026-06-12/scripts/audit_candidates.py, factored into
  scratch/_audit_io.py) is: central finite-difference of the PRISTINE ``state.velocity_ned``
  (world NED, the trusted external truth) vs the attitude-derived specific force
  ``a_up * R(q) @ [0,0,-1] + quad_drag(R, v) + [0,0,g]``, on samples with tilt > 35 deg.
  DISCRIMINATING AXIS = EAST: a roll-mirror leaves North/Down nearly invariant (never judge the
  convention on N/D), but flips the East-axis correlation from ~ +0.99 (TRUE [1,-1,1,-1]) to
  ~ -0.85 (AS-IS [1,1,1,1]).

WHAT THIS TEST DOES (two independent guards):
  TEST 1 -- EXTERNAL INVARIANT through the DEPLOYED plant (rl_plant, the numpy backend's literal
    delegate; the torch backend mirrors it op-for-op and is gated bit-identical in TEST 2). For
    every tilt>35deg tick of the real recordings, seed the plant at the live world-NED state with
    the TRUE attitude, step one tick at the realized collective, and compare the integrated
    world-NED acceleration (v_new - v)/dt to the live FD acceleration. ASSERT East corr is strongly
    positive AND median |residual| is small. NEGATIVE CONTROL: repeat with the AS-IS [1,1,1,1] quat
    convention (the bug) and ASSERT East corr goes NEGATIVE -- proving the test would catch a
    re-introduced conjugation. (If the East discriminator ever stopped flipping, the test itself
    would have lost its teeth; the negative control guards the guard.)

  TEST 2 -- EXTREME-STATE adapter parity. Drive the ACTUAL ``rl.diffaero_dynamics`` backends
    (_step_torch and _step_numpy, through the real NED<->diffaero bridges) from a battery of
    EXTREME states -- full-stick thrust (normed up to 5), |body-rate command| > pi, INVERTED
    attitudes (random unit quats incl. upside-down), HIGH SPEED (|v| up to ~43 m/s), large body
    rates, AND the transport-delay ring buffer engaged (transport_delay_steps > 0) -- and ASSERT
    both backends reproduce the parity-tested numpy ``rl_plant`` to a tight tolerance. Nominal-state
    parity tests (hover, gentle rates) never exercise the super-rate map's full-stick branch, the
    quad-drag sign split at high speed, or the inverted-attitude thrust projection where a bridge
    sign error actually bites.

COVERAGE NOTE -- what each guard does and does NOT catch (verified, do not "fix" by adding a
  confounded check):
    * TEST 1 catches a re-introduced bug in the deployed plant's PHYSICS attitude convention --
      the actual 4x bug class: a quat-rotate / rate_sign / _BODY_UP / drag-sign error that rotates
      thrust+drag relative to the trusted world-NED velocity anchor. This is the load-bearing guard
      (it runs through ``rl_plant``, which the numpy backend delegates to literally and the torch
      backend mirrors bit-identically per TEST 2).
    * TEST 2 catches a torch<->numpy backend divergence + a physics-block divergence at extremes.
    * NEITHER catches a mutation of the adapter's own ``_QFLIP`` / ``_FLIP`` NED<->diffaero bridge
      constants -- and that is CORRECT, not a gap: those constants define a self-consistent FRAME
      CHOICE (diag(1,-1,-1) on world AND body is a proper rotation that preserves gravity-down /
      thrust-up), the policy is trained and deployed entirely WITHIN that frame, and they cancel in
      every round-trip. The absolute E/D world labels only matter when importing a world gate-map
      (adapter mismatch #2), which is a separate concern. Verified empirically: an adapter-routed
      external test on ``_acc`` does NOT discriminate (East corr is invariant to a _QFLIP flip and
      sits at -0.99 by frame choice), so this file deliberately routes TEST 1 through the
      bridge-free physics block instead. Do not add an ``_acc``-vs-true-NED check -- it is confounded
      by the symmetric _FLIP and would assert a false expectation.

RUN MODES:
  * direct:  ``.venv/Scripts/python.exe handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_twin_diffaero_extreme_parity.py``
             prints PASS/FAIL + the key numbers.
  * pytest:  ``test_*`` functions with assertions (collectable into tests/).

SCOPE NOTE: does NOT import or audit ``rl/contact_true_eval.py`` (out of scope; edited elsewhere).
Requires torch (present in the repo .venv); skips the torch half cleanly if torch is unavailable.
The DiffAero package itself is NOT required -- the adapter falls back to ``BaseDynamics=object`` and
we drive its real ``_step_torch`` / ``_step_numpy`` through a thin construction shim (``_AdapterShim``)
that sets ONLY the attributes the backends read, with all domain-randomization OFF (the scalar path,
which is bit-identical to rl_plant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pytest

# --- shared substrate (tests/_audit_io.py) lives beside this file; + repo rl/src -----------------
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import _audit_io as A                             # shared loader + force model + canary constants  # noqa: E402

_ROOT = A.ROOT                                    # repo root (depth-independent, from _audit_io)
for _p in (str(_ROOT / "rl"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)

# torch is optional; the external-invariant TEST 1 runs without it (numpy plant), TEST 2 needs it.
try:
    import torch
    _HAVE_TORCH = True
except Exception:                                  # pragma: no cover
    torch = None
    _HAVE_TORCH = False

# the deployed faithful plant constants (map-ON) + the numpy reference step -----------------------
from racer.rl_plant import (  # noqa: E402
    PlantParams, PlantState, step as rl_step,
    SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
    QUAD_DRAG_C2_MEASURED, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
    MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED,
)

# the adapter under test + its frame bridges + action map (the production train<->eval seam) -------
import diffaero_dynamics as DA  # noqa: E402

# FLU->FRD body-rate map the adapter applies inside _action_diffaero_to_ctbr_np (== _FLIP[1,-1,-1])
_FLU_TO_FRD = np.array([1.0, -1.0, -1.0])

# ============================ thresholds (documented, not magic) =================================
TILT_MIN_DEG = 35.0           # external invariant is admissible only above this tilt (banked)
EAST_CORR_MIN_TRUE = 0.90     # TRUE attitude: East corr is reliably ~ +0.97..+0.99 (pooled +0.99)
EAST_CORR_MAX_BUG = 0.50      # AS-IS (bug) attitude: East corr goes NEGATIVE; require clearly < this
EAST_RES_MAX_TRUE = 4.0       # TRUE attitude: median |residual| on East stays small (m/s^2)
PARITY_ATOL = 1e-6            # torch/numpy backend vs rl_plant at extreme states (float64 path)
MIN_TILTED = 200              # need a real tilted sample population to trust the correlation


# =================================================================================================
# Faithful (deployed) map-ON plant: super-rate + slew + convex collective + quad drag + S17 mixer.
# linear_drag = 0 (the quad term replaces it; matches the live deploy config).
# =================================================================================================
def _faithful_params() -> PlantParams:
    return PlantParams(
        super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED,
        quad_drag_c2=QUAD_DRAG_C2_MEASURED, coll_map_thr=COLL_MAP_THR_MEASURED,
        coll_map_accel=COLL_MAP_ACCEL_MEASURED, linear_drag=0.0,
        mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
    )


# For the EXTERNAL-invariant test we use the SAME force model the canary uses (no mixer, so the
# realized collective feeds the thrust map directly and the one-step accel == Test-A's force model).
def _aero_params() -> PlantParams:
    return PlantParams(
        super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED,
        quad_drag_c2=QUAD_DRAG_C2_MEASURED, coll_map_thr=COLL_MAP_THR_MEASURED,
        coll_map_accel=COLL_MAP_ACCEL_MEASURED, linear_drag=0.0,
    )


# =================================================================================================
# TEST 1 -- EXTERNAL INVARIANT: deployed plant one-step accel vs FD of pristine vel_ned, tilt>35.
#   The plant's translational block IS the audit force model; driving it one step at the realized
#   collective (omega=0) yields exactly K*L-free a_up*R@[0,0,-1] + quad_drag + g.  We pin the
#   2-tick collective delay (coll[k-2]) the reference Test A uses.  use_lapse is irrelevant -- the
#   deployed config has NO lapse (S18 VOIDED), so we do not enable it.
# =================================================================================================
def _plant_accel_vs_fd(run: dict, quat_sign: np.ndarray, params: PlantParams):
    """Per-axis (median|residual|, corr) of (deployed plant one-step accel) vs (FD of vel_ned) in the
    tilt>TILT_MIN_DEG bin, with the candidate ``quat_sign`` applied to the raw ODOMETRY quat."""
    t, vel, coll = run["t"], run["vel"], run["coll"]
    q = run["q_raw"] * quat_sign[None, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    tilt_raw = A.tilt_deg(A.Rmats(run["q_raw"], sign=A.CAND_ASIS))  # tilt invariant across candidates
    n = len(t)
    model, meas, tl = [], [], []
    for k in range(2, n - 2):
        dt = t[k + 1] - t[k - 1]
        if not (0.05 < dt < 0.09):
            continue
        a_meas = (vel[k + 1] - vel[k - 1]) / dt
        if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
            continue
        coll_d = float(coll[k - 2])                                # 2-tick delay (reference Test A)
        ps = PlantState(pos=np.zeros(3), vel=vel[k].copy(), quat=q[k].copy(),
                        omega=np.zeros(3), thrust=np.array(coll_d))
        ps2 = rl_step(ps, np.array([0.0, 0.0, 0.0, coll_d]), dt, params)
        model.append((ps2.vel - vel[k]) / dt)                      # integrated world-NED accel
        meas.append(a_meas)
        tl.append(tilt_raw[k])
    model = np.array(model); meas = np.array(meas); tl = np.array(tl)
    m = tl > TILT_MIN_DEG
    AX = ["N", "E", "D"]
    out = {"n_tilted": int(m.sum()), "axes": {}}
    for ax in range(3):
        if m.sum() < 5:
            out["axes"][AX[ax]] = (np.nan, np.nan)
            continue
        md = float(np.median(np.abs(model[m, ax] - meas[m, ax])))
        cor = (float(np.corrcoef(model[m, ax], meas[m, ax])[0, 1])
               if model[m, ax].std() > 1e-6 else np.nan)
        out["axes"][AX[ax]] = (md, cor)
    return out


def _pool_runs(dataset: str):
    """Yield loaded runs for a dataset, tolerating header-only / empty runs (banked caveat:
    20260612_184029_..._std_f2 is header-only -> load_run raises ValueError; skip it)."""
    for rd in A.list_runs(dataset):
        try:
            yield A.load_run(rd)
        except ValueError:
            continue


def _external_invariant_pooled(dataset: str = "refit"):
    """Pool the external-invariant accel-vs-FD over all (loadable) runs of a dataset, for the TRUE
    and the AS-IS (bug) quat conventions. Returns the pooled per-axis (med|res|, corr) for each."""
    params = _aero_params()
    pooled = {"TRUE": {"model": [], "meas": [], "tilt": []},
              "ASIS": {"model": [], "meas": [], "tilt": []}}
    for run in _pool_runs(dataset):
        t, vel, coll = run["t"], run["vel"], run["coll"]
        tilt_raw = A.tilt_deg(A.Rmats(run["q_raw"], sign=A.CAND_ASIS))
        n = len(t)
        qs = {"TRUE": run["q_raw"] * A.CAND_TRUE[None, :],
              "ASIS": run["q_raw"] * A.CAND_ASIS[None, :]}
        for key in qs:
            qs[key] = qs[key] / np.linalg.norm(qs[key], axis=-1, keepdims=True)
        for k in range(2, n - 2):
            dt = t[k + 1] - t[k - 1]
            if not (0.05 < dt < 0.09):
                continue
            a_meas = (vel[k + 1] - vel[k - 1]) / dt
            if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
                continue
            coll_d = float(coll[k - 2])
            for key in ("TRUE", "ASIS"):
                ps = PlantState(pos=np.zeros(3), vel=vel[k].copy(), quat=qs[key][k].copy(),
                                omega=np.zeros(3), thrust=np.array(coll_d))
                ps2 = rl_step(ps, np.array([0.0, 0.0, 0.0, coll_d]), dt, params)
                pooled[key]["model"].append((ps2.vel - vel[k]) / dt)
                pooled[key]["meas"].append(a_meas)
                pooled[key]["tilt"].append(tilt_raw[k])
    res = {}
    AX = ["N", "E", "D"]
    for key in ("TRUE", "ASIS"):
        model = np.array(pooled[key]["model"]); meas = np.array(pooled[key]["meas"])
        tl = np.array(pooled[key]["tilt"]); m = tl > TILT_MIN_DEG
        axes = {}
        for ax in range(3):
            md = float(np.median(np.abs(model[m, ax] - meas[m, ax])))
            cor = float(np.corrcoef(model[m, ax], meas[m, ax])[0, 1])
            axes[AX[ax]] = (md, cor)
        res[key] = {"n_tilted": int(m.sum()), "axes": axes}
    return res


# =================================================================================================
# TEST 2 -- EXTREME-STATE adapter parity.  We drive the REAL rl.diffaero_dynamics backends through
# their REAL NED<->diffaero bridges, with all DR OFF (the scalar path == rl_plant op-for-op).
# =================================================================================================
class _AdapterShim(DA.PeregrinePlantDynamics):
    """Construct the adapter WITHOUT DiffAero's BaseDynamics / hydra cfg: set ONLY the attributes
    ``_step_torch`` / ``_step_numpy`` / ``check_against_rl_plant`` read, all DR OFF (scalar path).
    grad_decay is identity (PPO path; no BPTT here).  This exercises the adapter's ACTUAL backend
    code + frame bridges -- it does not re-implement them."""

    def __init__(self, params: PlantParams, n_envs: int, dt: float):
        if not _HAVE_TORCH:                       # pragma: no cover
            raise RuntimeError("torch required for the adapter shim")
        self.params = params
        self.params.g = float(params.g)
        self.dt = float(dt)
        self.n_substeps = 1
        self.n_envs = int(n_envs)
        self.n_agents = 1
        self.backend = "torch"
        self.type = "quadrotor"
        self.state_dim = 13
        self.action_dim = 4
        for f in ("_dr_enabled", "_dr_aero", "_dr_mixer", "_dr_lapse",
                  "_dr_force_bias", "_latency_enabled"):
            setattr(self, f, False)
        self._plant_act_buf = None
        dt64 = torch.float64
        self._rate_gain = torch.tensor(params.rate_gain, dtype=dt64)
        self._rate_sign = torch.tensor(params.rate_sign, dtype=dt64)
        self._BODY_UP = torch.tensor([0.0, 0.0, -1.0], dtype=dt64)
        self._g_vec_ned = torch.tensor([0.0, 0.0, params.g], dtype=dt64)
        self._super_s = (None if params.super_rate_s is None else
                         torch.tensor(np.broadcast_to(params.super_rate_s, (3,)).copy(), dtype=dt64))
        self._alpha_max = (None if params.alpha_max_rps2 is None else
                           torch.tensor(np.broadcast_to(params.alpha_max_rps2, (3,)).copy(), dtype=dt64))
        self._quad_c2 = (None if params.quad_drag_c2 is None else
                         torch.tensor(params.quad_drag_c2, dtype=dt64))
        self._coll_knots = (None if params.coll_map_thr is None else
                            torch.tensor(params.coll_map_thr, dtype=dt64))
        self._coll_kvals = (None if params.coll_map_accel is None else
                            torch.tensor(params.coll_map_accel, dtype=dt64))
        self._lapse_knots = (None if params.lapse_speed is None else
                             torch.tensor(params.lapse_speed, dtype=dt64))
        self._lapse_vals = (None if params.lapse_factor is None else
                            torch.tensor(params.lapse_factor, dtype=dt64))
        self._mix_rfit = (None if params.mixer_idle is None else
                          torch.tensor(params._mixer_r_fit, dtype=dt64))
        bshape = (self.n_envs,)
        self._state = torch.zeros(*bshape, 13, dtype=dt64)
        self._state[..., 6] = 1.0                 # identity quat (xyzw real-last)
        self._acc = torch.zeros(*bshape, 3, dtype=dt64)
        self._thrust = torch.full(bshape, float(params.hover_thrust), dtype=dt64)

    def grad_decay(self, x):                       # PPO/eval path: no gradient decay
        return x

    # convenience: seed the adapter at a batch of live NED states ------------------------------
    def seed_ned(self, pos, vel, quat_wxyz, omega, thrust):
        pd, vd, qd, wd = DA._diffaero_from_ned_np(pos, vel, quat_wxyz, omega)
        self._state = torch.tensor(np.concatenate([pd, qd, vd, wd], axis=-1), dtype=torch.float64)
        self._thrust = torch.tensor(np.asarray(thrust, dtype=float), dtype=torch.float64)

    def read_ned(self):
        st = self._state.detach().cpu().numpy()
        p, v, q, w = DA._ned_from_diffaero_np(st[..., 0:3], st[..., 3:7], st[..., 7:10], st[..., 10:13])
        return p, v, q, w


def _rl_reference_step(params, pos, vel, quat, omega, thrust, U, dt):
    """One rl_plant step per env, applying the adapter's OWN action map (FLU->FRD rates,
    collective = normed_thrust * hover) -- the ground-truth the adapter must reproduce."""
    n = pos.shape[0]
    out = np.zeros((n, 13))
    for i in range(n):
        rate_frd = U[i, 1:4] * _FLU_TO_FRD
        collective = U[i, 0] * params.hover_thrust
        ps = PlantState(pos=pos[i].copy(), vel=vel[i].copy(), quat=quat[i].copy(),
                        omega=omega[i].copy(), thrust=np.array(float(thrust[i])))
        ps2 = rl_step(ps, np.concatenate([rate_frd, [collective]]), dt, params)
        out[i] = np.concatenate([ps2.pos, ps2.quat, ps2.vel, ps2.omega])
    return out


def _align_quat_sign(got, ref):
    """q and -q are the same rotation; align signs before comparing the wxyz block (cols 3:7)."""
    got = got.copy()
    for i in range(got.shape[0]):
        if np.dot(got[i, 3:7], ref[i, 3:7]) < 0:
            got[i, 3:7] *= -1.0
    return got


def _extreme_battery(seed=0):
    """A battery of EXTREME states + actions that nominal-state parity tests never reach."""
    rng = np.random.default_rng(seed)
    n = 96
    q = rng.standard_normal((n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)  # incl. inverted
    pos = rng.uniform(-50, 50, (n, 3))
    vel = rng.uniform(-25, 25, (n, 3))            # |v| up to ~43 m/s (high speed -> quad-drag sign split)
    omega = rng.uniform(-12, 12, (n, 3))          # large body rates
    thrust = rng.uniform(0.05, 1.0, n)
    U = np.zeros((n, 4))
    U[:, 0] = rng.uniform(0.0, 5.0, n)            # full-stick thrust (normed up to 5)
    U[:, 1:] = rng.uniform(-6.0, 6.0, (n, 3))     # |cmd rate| > pi (super-rate full-stick branch)
    return pos, vel, q, omega, thrust, U


def _backend_divergence(params, dt, transport_delay_steps=0):
    """Max abs state divergence of BOTH adapter backends vs rl_plant over the extreme battery,
    plus the torch<->numpy divergence. transport_delay_steps>0 exercises the ring buffer (the
    train<->eval latency seam) via a multi-step rollout from the same start."""
    pos, vel, q, omega, thrust, U = _extreme_battery()
    p = params
    if transport_delay_steps > 0:
        p = _faithful_params()
        p.transport_delay_steps = int(transport_delay_steps)
        p.super_rate_s = params.super_rate_s
    n = pos.shape[0]
    T = 1 if transport_delay_steps == 0 else (transport_delay_steps + 2)

    # action stack: same start, T steps (delay only matters with T>1)
    rng = np.random.default_rng(1)
    Us = [U]
    for _ in range(T - 1):
        Uk = np.zeros((n, 4)); Uk[:, 0] = rng.uniform(0.0, 5.0, n); Uk[:, 1:] = rng.uniform(-6.0, 6.0, (n, 3))
        Us.append(Uk)

    # --- numpy reference: rl_plant carried through the stack (its OWN act_buf for the delay) ---
    ref_final = None
    ref_pos, ref_vel, ref_q, ref_w, ref_thr = pos.copy(), vel.copy(), q.copy(), omega.copy(), thrust.copy()
    ref_buf = None
    for Uk in Us:
        # build the rl_plant state with the carried act_buf and step once per env
        outs = np.zeros((n, 13)); new_thr = np.zeros(n)
        new_buf = None if p.transport_delay_steps == 0 else np.zeros((n, p.transport_delay_steps, 4))
        for i in range(n):
            rate_frd = Uk[i, 1:4] * _FLU_TO_FRD
            collective = Uk[i, 0] * p.hover_thrust
            buf_i = None if ref_buf is None else ref_buf[i].copy()
            ps = PlantState(pos=ref_pos[i].copy(), vel=ref_vel[i].copy(), quat=ref_q[i].copy(),
                            omega=ref_w[i].copy(), thrust=np.array(float(ref_thr[i])), act_buf=buf_i)
            ps2 = rl_step(ps, np.concatenate([rate_frd, [collective]]), dt, p)
            outs[i] = np.concatenate([ps2.pos, ps2.quat, ps2.vel, ps2.omega]); new_thr[i] = float(ps2.thrust)
            if new_buf is not None:
                new_buf[i] = ps2.act_buf
        ref_pos, ref_q, ref_vel, ref_w = outs[:, 0:3], outs[:, 3:7], outs[:, 7:10], outs[:, 10:13]
        ref_thr = new_thr; ref_buf = new_buf
        ref_final = outs.copy()

    # --- adapter torch + numpy backends from the identical start ---
    def run(backend):
        s = _AdapterShim(p, n, dt)
        s.backend = backend
        s.seed_ned(pos, vel, q, omega, thrust)
        for Uk in Us:
            if backend == "torch":
                s._step_torch(torch.tensor(Uk, dtype=torch.float64))
            else:
                s._step_numpy(torch.tensor(Uk, dtype=torch.float64))
        pp, vv, qq, ww = s.read_ned()
        return np.concatenate([pp, qq, vv, ww], axis=-1)

    got_t = _align_quat_sign(run("torch"), ref_final)
    got_n = _align_quat_sign(run("rl_plant_numpy"), ref_final)
    ref_a = _align_quat_sign(ref_final.copy(), ref_final)
    return {
        "torch_vs_rl": float(np.max(np.abs(got_t - ref_a))),
        "numpy_vs_rl": float(np.max(np.abs(got_n - ref_a))),
        "torch_vs_numpy": float(np.max(np.abs(got_t - got_n))),
        "n": n, "T": T, "delay": p.transport_delay_steps,
    }


# =================================================================================================
# pytest entry points
# =================================================================================================
def test_external_invariant_true_attitude_wins_on_east():
    """TRUE [1,-1,1,-1] attitude -> deployed-plant accel correlates ~+0.99 with FD of vel_ned on the
    EAST axis at tilt>35deg, with small residual. The discriminating external invariant."""
    res = _external_invariant_pooled("refit")
    nt = res["TRUE"]["n_tilted"]
    e_res, e_cor = res["TRUE"]["axes"]["E"]
    assert nt >= MIN_TILTED, f"too few tilted samples ({nt} < {MIN_TILTED}); test would be untrustworthy"
    assert e_cor >= EAST_CORR_MIN_TRUE, f"TRUE East corr {e_cor:+.3f} < {EAST_CORR_MIN_TRUE} (FRAME BROKEN?)"
    assert e_res <= EAST_RES_MAX_TRUE, f"TRUE East median|res| {e_res:.2f} > {EAST_RES_MAX_TRUE} m/s^2"


def test_external_invariant_negative_control_asis_breaks_east():
    """NEGATIVE CONTROL: the AS-IS [1,1,1,1] convention (the R_y(pi)-conjugation bug, un-undone)
    must ANTI-correlate on East -- proving the external invariant has discriminating power. If this
    ever stopped failing, the guard above would be toothless."""
    res = _external_invariant_pooled("refit")
    _, e_cor_bug = res["ASIS"]["axes"]["E"]
    _, e_cor_true = res["TRUE"]["axes"]["E"]
    assert e_cor_bug < EAST_CORR_MAX_BUG, (
        f"AS-IS East corr {e_cor_bug:+.3f} NOT clearly broken (>= {EAST_CORR_MAX_BUG}); "
        f"the East discriminator has lost its teeth")
    assert e_cor_bug < e_cor_true - 0.5, (
        f"AS-IS East corr {e_cor_bug:+.3f} not separated from TRUE {e_cor_true:+.3f} by > 0.5")


def test_external_invariant_holds_on_postfix_dataset():
    """The mirror is in the SIM TELEMETRY (present in BOTH datasets) -> the same TRUE-wins /
    AS-IS-breaks split must hold on the postfix (frame-clean navigator) dataset too. Tolerant of
    the header-only run."""
    res = _external_invariant_pooled("postfix")
    _, e_true = res["TRUE"]["axes"]["E"]
    _, e_bug = res["ASIS"]["axes"]["E"]
    assert e_true >= EAST_CORR_MIN_TRUE, f"postfix TRUE East corr {e_true:+.3f} < {EAST_CORR_MIN_TRUE}"
    assert e_bug < EAST_CORR_MAX_BUG, f"postfix AS-IS East corr {e_bug:+.3f} not broken"


def test_adapter_torch_backend_parity_extreme_states():
    """rl.diffaero_dynamics torch backend reproduces rl_plant at EXTREME states
    (full-stick, |cmd|>pi, inverted, high-speed) through its real NED<->diffaero bridges."""
    if not _HAVE_TORCH:                            # pragma: no cover
        import pytest
        pytest.skip("torch unavailable")
    d = _backend_divergence(_faithful_params(), dt=0.0333)
    assert d["torch_vs_rl"] < PARITY_ATOL, f"torch backend diverges from rl_plant: {d['torch_vs_rl']:.2e}"
    assert d["numpy_vs_rl"] < 1e-12, f"numpy backend (literal rl_plant) diverges: {d['numpy_vs_rl']:.2e}"
    assert d["torch_vs_numpy"] < PARITY_ATOL, f"torch<->numpy backend divergence: {d['torch_vs_numpy']:.2e}"


def test_adapter_parity_with_transport_delay_buffer():
    """The transport-delay ring buffer (the train<->eval latency seam) stays bit-faithful across the
    adapter backends at extreme states -- a stale/mis-indexed buffer would be a silent plant divergence."""
    if not _HAVE_TORCH:                            # pragma: no cover
        import pytest
        pytest.skip("torch unavailable")
    d = _backend_divergence(_faithful_params(), dt=0.0333, transport_delay_steps=2)
    assert d["torch_vs_rl"] < PARITY_ATOL, f"torch backend w/ delay diverges: {d['torch_vs_rl']:.2e}"
    assert d["numpy_vs_rl"] < 1e-12, f"numpy backend w/ delay diverges: {d['numpy_vs_rl']:.2e}"
    assert d["torch_vs_numpy"] < PARITY_ATOL, f"torch<->numpy w/ delay: {d['torch_vs_numpy']:.2e}"


# =================================================================================================
# direct-run reporter
# =================================================================================================
def _fmt_axes(axes):
    return "  ".join(f"{ax}: med|r|={axes[ax][0]:6.2f} corr={axes[ax][1]:+5.2f}" for ax in ("N", "E", "D"))


def main() -> int:
    print("=" * 92)
    print("twin/rl_plant <-> DiffAero adapter EXTREME-state parity + EXTERNAL-invariant regression")
    print(f"  torch available: {_HAVE_TORCH}   |   diffaero pkg: {DA.BaseDynamics is not object}")
    print("=" * 92)
    failures = []

    # ---- TEST 1 + negative control ----
    print("\n[TEST 1] EXTERNAL INVARIANT -- deployed plant one-step accel vs FD(vel_ned), tilt>35deg")
    for ds in ("refit", "postfix"):
        try:
            res = _external_invariant_pooled(ds)
        except Exception as e:                     # pragma: no cover
            print(f"  dataset {ds!r}: ERROR {e}")
            failures.append(f"external/{ds}: {e}")
            continue
        for key in ("TRUE", "ASIS"):
            label = "TRUE  [1,-1,1,-1]" if key == "TRUE" else "AS-IS [1, 1,1, 1] (bug/neg-ctrl)"
            print(f"  [{ds:7s}] {label}  (n_tilt={res[key]['n_tilted']:5d})  {_fmt_axes(res[key]['axes'])}")
        e_true = res["TRUE"]["axes"]["E"][1]
        e_bug = res["ASIS"]["axes"]["E"][1]
        ok = (e_true >= EAST_CORR_MIN_TRUE) and (e_bug < EAST_CORR_MAX_BUG) and (e_bug < e_true - 0.5)
        print(f"           -> East discriminates: TRUE {e_true:+.3f} (>= {EAST_CORR_MIN_TRUE}) "
              f"vs AS-IS {e_bug:+.3f} (< {EAST_CORR_MAX_BUG})  [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            failures.append(f"external/{ds}: East TRUE {e_true:+.3f} / AS-IS {e_bug:+.3f}")
        if res["TRUE"]["n_tilted"] < MIN_TILTED:
            failures.append(f"external/{ds}: only {res['TRUE']['n_tilted']} tilted samples")

    # ---- TEST 2 ----
    print("\n[TEST 2] EXTREME-STATE adapter parity -- torch / numpy backends vs rl_plant")
    if not _HAVE_TORCH:
        print("  torch unavailable -> SKIPPED (TEST 1 external invariant still ran on the numpy plant)")
    else:
        for label, delay in (("no transport delay", 0), ("transport_delay_steps=2", 2)):
            d = _backend_divergence(_faithful_params(), dt=0.0333, transport_delay_steps=delay)
            ok = (d["torch_vs_rl"] < PARITY_ATOL and d["numpy_vs_rl"] < 1e-12
                  and d["torch_vs_numpy"] < PARITY_ATOL)
            print(f"  [{label:24s}] n={d['n']} T={d['T']} delay={d['delay']}  "
                  f"torch-vs-rl={d['torch_vs_rl']:.2e}  numpy-vs-rl={d['numpy_vs_rl']:.2e}  "
                  f"torch-vs-numpy={d['torch_vs_numpy']:.2e}  [{'PASS' if ok else 'FAIL'}]")
            if not ok:
                failures.append(f"parity/{label}: {d}")

    print("\n" + "=" * 92)
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  -", f)
        print("=" * 92)
        return 1
    print("RESULT: PASS  (external invariant discriminates on East; extreme-state backend parity holds)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
