"""rl/fly_rl.py  --  Stage-1 inc-1 CTBR RL policy deployment in the VQ1 sim.

Loads the trained actor checkpoint, connects via MAVLink, waits for a fresh race
start (auto-requesting one via MAV_CMD 31000 when possible), arms, then flies at
the TRAINING control rate (30 Hz = racing.yaml dt 0.0333): builds the 17-dim
observation (exactly matching peregrine_racing.get_observations), replicates the
training action pipeline (tanh -> rescale to [min,max] action bounds), sends
SET_ATTITUDE_TARGET (BODY_RATE).  Records each flight for race_outcome, and can
chain --flights N attempts back-to-back via the sim-reset command.

VQ2 CONTROL RECIPE (live-confirmed 2026-06-29, sim build 1.0.3379) -- this deploy path
ALREADY matches it byte-for-byte (no change needed beyond the two opt-in deltas below):
  * ARM = MAV_CMD_COMPONENT_ARM_DISARM (=400), param1=1  -> client.arm() (mavlink_client.py).
  * Control = SET_ATTITUDE_TARGET in BODY-RATE mode, type_mask=0b10000000
    (_ATT_MASK_BODY_RATE in mavlink_client.py), FRD body rates + normalized collective [0,1],
    streamed at the training cadence (30 Hz here; sim default mode ACRO == ControlMode.BODY_RATE).
    The collective is clipped to [0,1] in policy_step; the rates are FRD. This is the SAME uplink
    fly_rl already drove on VQ1 -- verified, not re-plumbed.
  * DELTA 1 (cmd_rate_scale): the VQ2 sim realizes a commanded body rate ~2.5x. --cmd-rate-scale
    (MavlinkClient.cmd_rate_scale, default 1.0 == byte-identical) feed-forward-compensates at the
    uplink; pass ~0.4 (=1/2.5). Default 1.0 lets the CLOSED-LOOP policy absorb the gain instead
    (the correct/safe default) -- the scale is OPT-IN.
  * DELTA 2 (gyro_body deploy source): VQ2 BLOCKS ODOMETRY, so the obs body rate (obs[9:12] w_flu)
    + attitude (obs[6:9] rpy_g) must come from the AHRS fed by the RAW HIGHRES_IMU gyro
    (DroneState.gyro_body, body FRD), NOT the ODOMETRY-derived angular_rate_body. That routing is
    OWNED by the navigator's ``use_ahrs`` seam (NavigatorConfig.use_ahrs=True; AHRSAttitudeSource
    consumes ds.gyro_body -- ahrs_adapter.py / test_use_ahrs_wiring.py). The RL deploy obs path
    consumes the AHRS-sourced rate read-only via estimator_obs20(nav.obs_drone_state(ds), ...).
    DEPLOY TOGGLE: build the Navigator with NavigatorConfig(use_ahrs=True) for the case-C VQ2 run;
    use_ahrs=False (default) keeps the VQ1 ODOMETRY-rate path byte-identical. build_obs(state, ...)
    reads state.angular_rate_body, so the case-C wiring supplies the AHRS rate through that field
    (Fix-A convention: ODOMETRY-sign-compatible) -- fly_rl's raw build_obs default path is unchanged.

DEPLOYMENT MATH (verified against the diffaero source the checkpoint trained on,
S1.2 session 2026-06-10 -- see the constants below for the per-item derivations):
  * action = tanh(actor_mean(obs)); env action = min + (max-min)*(action+1)/2
    (StochasticActor.forward test branch + BaseEnv.rescale_action; the raw mean
    is NEVER sent directly).
  * obs[12] = the previous action's RESCALED normed_thrust, in [0, act_max] g-units
    (act_max = the sidecar's act_max_thrust, e.g. 3.765 for inc7; the LEGACY 5.0 otherwise)
    -- NOT the [0,1] wire collective. 0.0 at episode start (BaseEnv.last_action zeroed in
    reset_idx). The [0,1] collective sent to the wire is normed_thrust * hover_thrust (clipped).
  * policy rates are FLU body rates; the training adapter maps them with the
    FLU->FRD flip [1,-1,-1] onto a plant with rate_sign [+1,+1,-1], so the
    TRAINED semantics is realized_frd = g * [+1,-1,+1] * rate_flu. The live
    sim's TRUE command->rate map has NO inversion on any axis (FRAME-AUDIT
    2026-06-12: open-loop replay of recorded wire commands reproduces the true
    attitude trajectory with rate_sign [+1,+1,+1]) -> wire = rate_flu*[+1,-1,+1].
  * the raw ODOMETRY quat is the attitude in an R_y(pi)-CONJUGATED frame pair
    (roll AND yaw Euler negated, pitch intact): true quat = q_raw*[1,-1,1,-1];
    true FRD rate = -angular_rate_raw (all axes). Pinned against FD of pristine
    vel_ned over 17 live runs (LAPTOP-FRAME-AUDIT). The 2026-06-12 bcc93f9
    reading (quat as-is, rates [+1,-1,+1], wire [-1,-1,-1]) was a SECOND
    self-consistent mirror: rotationally coherent, so rates/attitude verified
    "end-to-end", but its thrust->world LATERAL projection is East-mirrored --
    invisible level, fatal in the banked gate-0 flare (S18 smoking gun).

Usage:
  .venv\\Scripts\\python.exe rl\\fly_rl.py --label rl_s1_v2
  .venv\\Scripts\\python.exe rl\\fly_rl.py --flights 4 --label rl_s1_batch
  .venv\\Scripts\\python.exe rl\\fly_rl.py --max-rate 1.5 --label rl_s1_capped
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from racer.contracts import ControlCommand, ControlMode
from racer.finish_hold import drain_until_finish, sim_finish_confirmed
from racer.firstcontact import telemetry_summary
from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp
from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver

# ---------------------------------------------------------------------------
# Frame & sign constants — must match peregrine_racing.py + diffaero_dynamics.py
# ---------------------------------------------------------------------------
# R_x(π) = diag(1,-1,-1) maps NED↔Z-up (world) and FRD↔FLU (body).
_FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float64)  # world & body frame flip

# ODOMETRY quat -> TRUE attitude: R_y(pi) conjugation, i.e. negate the x and z
# quat components (roll AND yaw Euler negated, pitch intact). FRAME-AUDIT
# 2026-06-12: pinned per-axis against the EXTERNAL invariant (FD of pristine
# vel_ned, 17 runs, 28k banked ticks) -- force projection corr +0.97..+0.99 on
# all axes; the as-is reading anti-correlates on East at bank (-0.84). A proper
# conjugation passes every INTERNAL consistency test (quat-FD vs rates, twist
# round-trip), which is how it survived bcc93f9's tilted-phase validation --
# only force/course direction against vel_ned discriminates.
_ODO_QUAT_TRUE_CONJ = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)

# Raw ODOMETRY angular_rate -> TRUE FRD body rate: ALL THREE axes negated
# (quat-FD of the conjugated attitude == -w_raw, gain 0.999/0.999/0.996, all 17
# runs). Supersedes bcc93f9's [+1,-1,+1], which was the same measurement read
# against the unconjugated quat (self-consistent mirror).
_ODO_RATE_SIGN = np.array([-1.0, -1.0, -1.0], dtype=np.float64)

# RL action rates (FLU, diffaero Z-up world) -> FRD wire command. The training
# adapter does rate_frd = U[1:4] * [1,-1,-1] and the TRAINING plant applies
# rate_sign [+1,+1,-1], so the trained semantics is realized = g*[+1,-1,+1]*
# rate_flu. The live sim's TRUE command->rate map is [+1,+1,+1] -- NO inversion
# on any axis (FRAME-AUDIT: rate_sign [+1,+1,+1] open-loop-reproduces the true
# attitude trajectory from recorded wire commands; every historical "inversion"
# was the telemetry conjugation read back as physics). Preserving the trained
# semantics therefore needs wire = rate_flu * [+1,-1,+1].
_ACT_FLU_TO_FRD = np.array([1.0, -1.0, 1.0], dtype=np.float64)

_HOVER_THRUST = 0.2656   # PlantParams.hover_thrust; collective at zero net vertical accel

# Training action bounds (cfg/dynamics/quad.yaml controller block, verified):
# normed_thrust [0, 5], body rates ±3.14 rad/s (FLU). env.rescale_action maps
# tanh(mean) ∈ [-1,1] linearly onto these.
_ACT_MIN = np.array([0.0, -3.14, -3.14, -3.14], dtype=np.float64)
_ACT_MAX = np.array([5.0,  3.14,  3.14,  3.14], dtype=np.float64)

# Training control interval (cfg/env/racing.yaml dt: 0.0333 — no sbatch override).
_TRAIN_DT = 0.0333

# ---------------------------------------------------------------------------
# Course: 6 VQ1 gates in DiffAero Z-up frame (from rl/peregrine_course_diffaero.json)
# Z-up frame = NED * [1,-1,-1]; all gate yaws = 3.141592569 (=π to 8e-8).
# ---------------------------------------------------------------------------
N_GATES = 6
_GATE_POS_ZUP = np.array([
    [-23.2979679107666,    0.39990234375,       1.3919580206274986 ],
    [-46.89374923706055,   2.499990224838257,  -3.708041787147522  ],
    [-74.59375,           -1.2000097036361694, -12.308041214942932 ],
    [-111.49374389648438,  5.099989891052246,  -23.208040833473206 ],
    [-135.49374389648438,  0.7999902367591858, -23.995653748512268 ],
    [-159.19374084472656,  4.399990081787109,  -24.60804045200348  ],
], dtype=np.float64)

# World-to-gate rotation for yaw=π: R_z(π) = diag(-1,-1,1).
_R_W2G = np.diag([-1.0, -1.0, 1.0])

# Pre-computed gate_rel_pos lookup (peregrine_racing.py §__init__):
#   gate_rel_pos[i] = R_w2g(yaw[i-1]) @ (gate_pos[i] - gate_pos[i-1])
# Python [-1] wraps to the last gate (gate 5) for i=0.  All yaws identical -> same R_w2g.
_GATE_REL_POS = np.array(
    [_R_W2G @ (_GATE_POS_ZUP[i] - _GATE_POS_ZUP[i - 1]) for i in range(N_GATES)],
    dtype=np.float64,
)
_GATE_YAW_REL = np.zeros(N_GATES, dtype=np.float64)   # 0 everywhere (uniform yaw)

# Obs dim labels (debug dumps + _deprecated/replay_obs.py). Matches obs_from_zup's layout.
OBS_LABELS = (
    [f"pos_g{i}" for i in "xyz"] + [f"vel_g{i}" for i in "xyz"]
    + ["rpy_g_r", "rpy_g_p", "rpy_g_y"] + [f"w_flu{i}" for i in "xyz"]
    + ["prev_normed_thrust"] + [f"nxt_rel{i}" for i in "xyz"] + ["nxt_relyaw"]
)

# VIRTUAL BODY FLIP (π about body z). Training resets at identity Z-up attitude
# with all gates at yaw π => the policy learned to fly the course TAIL-FIRST
# (gate-frame yaw ≈ π throughout). The real sim spawns the drone NOSE-first
# (NED yaw ≈ ±180° => gate-frame yaw ≈ 0) — 180° attitude-OOD. Rigid-body
# dynamics are exactly symmetric under a π body-z rotation (plant rate_gain
# x/y differ 0.1%), so presenting the policy a virtual body frame rotated π
# about z (obs: R_b2w·Rz(π), rates·[-1,-1,1]; action rates·[-1,-1,1]) makes the
# nose-first spawn look exactly like the trained tail-first regime.
_RZ_PI_BODY = np.diag([-1.0, -1.0, 1.0])

# VQ1 design assumption: every gate yaw is π (the deployed course). _R_W2G / _GATE_REL_POS /
# _GATE_YAW_REL above are the EXACT yaw=π specialization (diag(-1,-1,1), sin(π) dropped). This
# constant lets the import-time guard catch a half-migrated edit (e.g. bumping _GATE_POS_ZUP to a
# non-π course while leaving the hardcoded gate frame), and seeds the deploy-time all-π check.
_GATE_YAW_ZUP = np.full(N_GATES, np.pi, dtype=np.float64)
# Tolerance for "this course is the all-π VQ1 course": the real course JSON stores yaw=3.141592569
# (π to ~8.4e-8), so the gate-relative obs the hardcoded path produces is correct to <1e-5 m; any
# deviation beyond this is a genuinely different (VQ2 / random / bent) course that the hardcoded
# yaw=π gate frame would silently corrupt by up to ~4.2 m (CONFIRMED bug P4-C05).
_GATE_YAW_TOL = 1e-4


# ---------------------------------------------------------------------------
# Per-gate gate-frame plumbing (P4-C05 fix) — the gate frame must follow the
# RUNTIME gate yaw, not a hardcoded yaw=π. The legacy _R_W2G/_GATE_REL_POS path
# is correct ONLY on VQ1 (all gates yaw=π); on any non-π / VQ2 / course_mode=random
# course it silently corrupts the consumed gate-relative obs (pos_g/vel_g/rpy_g_y/
# next-gate lookahead) by up to ~4.2 m. These helpers reproduce the TRAIN-side
# definition (peregrine_racing.get_observations / world_to_gateframe / rel_tables)
# EXACTLY so a yaw-aware deploy obs == the obs the policy was trained on, on any course.
# ---------------------------------------------------------------------------
class GateMap(NamedTuple):
    """A course's gate geometry for the obs builder (DiffAero Z-up frame), mirroring the
    runtime per-env course tensors in peregrine_racing (gate_pos / gate_yaw) plus the
    precomputed next-gate lookahead (rel_tables). Pass one to obs_from_zup / build_obs to
    fly a non-π course; the default (None) uses the bit-exact hardcoded VQ1 constants."""
    gate_pos: np.ndarray      # (N,3)  opening centres, Z-up
    gate_yaw: np.ndarray      # (N,)   through-yaw, rad
    gate_rel_pos: np.ndarray  # (N,3)  next-gate lookahead position (gate (i-1)'s frame)
    gate_yaw_rel: np.ndarray  # (N,)   next-gate lookahead rel yaw, wrapped


def _gate_rotmat_w2g(yaw: float) -> np.ndarray:
    """numpy world->gate rotation, rows [c,s,0; -s,c,0; 0,0,1] — EXACTLY diffaero's
    get_gate_rotmat_w2g and peregrine_racing.world_to_gateframe. For yaw=π this is
    [[-1, sin(π)≈1.2e-16, 0], [-1.2e-16, -1, 0], [0,0,1]] — i.e. _R_W2G to the sin(π)
    epsilon (the legacy path uses the EXACT diag, so the default obs is bit-identical)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rel_tables_np(gate_pos: np.ndarray, gate_yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """numpy mirror of peregrine_racing.rel_tables: next-gate lookahead, per gate.
    gate_rel_pos[i] = R_w2g(yaw[i-1]) @ (pos[i] - pos[i-1]); index 0 wraps to the last gate
    (the parent's Python [i-1] convention; index 0 is never consumed — next_gate_idx is
    clamped >= 1). gate_yaw_rel[i] = wrap(yaw[i] - yaw[i-1])."""
    n = len(gate_pos)
    rel = np.empty((n, 3), dtype=np.float64)
    yaw_rel = np.empty(n, dtype=np.float64)
    for i in range(n):
        rel[i] = _gate_rotmat_w2g(float(gate_yaw[i - 1])) @ (gate_pos[i] - gate_pos[i - 1])
        dy = float(gate_yaw[i]) - float(gate_yaw[i - 1])
        yaw_rel[i] = np.arctan2(np.sin(dy), np.cos(dy))
    return rel, yaw_rel


def make_gate_map(gate_pos_zup: np.ndarray, gate_yaw: np.ndarray) -> GateMap:
    """Build a GateMap (positions + yaws + precomputed lookahead) for an arbitrary course,
    matching peregrine_racing's per-env course tensors. Use this to fly a non-π / VQ2 /
    procedural course through the existing obs builder."""
    gate_pos_zup = np.asarray(gate_pos_zup, dtype=np.float64)
    gate_yaw = np.asarray(gate_yaw, dtype=np.float64)
    rel, yaw_rel = _rel_tables_np(gate_pos_zup, gate_yaw)
    return GateMap(gate_pos_zup, gate_yaw, rel, yaw_rel)


def assert_gate_map_allpi(gate_yaw: np.ndarray, *, where: str = "deploy",
                          tol: float = _GATE_YAW_TOL) -> None:
    """LOUD guard for the historically-hardcoded yaw=π obs path (CONFIRMED bug P4-C05).
    The default obs_from_zup gate frame (_R_W2G = diag(-1,-1,1)) and lookahead tables are
    correct ONLY when every gate yaw is π. On any non-π course they silently corrupt the
    gate-relative obs (pos_g/vel_g/rpy_g_y/next-gate lookahead) by up to ~4.2 m. Call this
    before trusting the default (gate_map=None) builder on a runtime course; to fly a non-π
    course, pass an explicit gate_map=make_gate_map(...) (the yaw-aware path) instead."""
    gy = np.asarray(gate_yaw, dtype=np.float64)
    dev = np.abs(np.arctan2(np.sin(gy - np.pi), np.cos(gy - np.pi)))  # wrapped distance to π
    if not np.all(dev <= tol):
        bad = np.where(dev > tol)[0].tolist()
        raise AssertionError(
            f"[{where}] P4-C05 GUARD: the hardcoded yaw=π gate-frame obs path is being used on "
            f"a NON-π course (gates {bad} deviate from π by up to {float(dev.max()):.4f} rad = "
            f"{np.degrees(float(dev.max())):.2f}°). This silently corrupts the gate-relative obs "
            f"by metres. Fly a non-π course via build_obs(..., gate_map=make_gate_map(pos, yaw)).")


def _assert_vq1_constants_consistent() -> None:
    """Import-time self-check: the hardcoded VQ1 obs constants (_R_W2G via _GATE_REL_POS/
    _GATE_YAW_REL) must equal the general per-gate construction on the all-π VQ1 course.
    Guards a HALF-MIGRATED edit — e.g. bumping _GATE_POS_ZUP to a non-π course while leaving
    the hardcoded gate frame — which would silently corrupt the deployed obs (P4-C05)."""
    assert_gate_map_allpi(_GATE_YAW_ZUP, where="fly_rl import")
    rel, yaw_rel = _rel_tables_np(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    # _R_W2G is the EXACT yaw=π diag; the general rel tables match to the sin(π) epsilon (~1e-14).
    if not np.allclose(rel, _GATE_REL_POS, atol=1e-9):
        raise AssertionError(
            f"_GATE_REL_POS diverges from the per-gate rel_tables on VQ1 "
            f"(max|diff|={float(np.abs(rel - _GATE_REL_POS).max()):.2e}); the hardcoded gate-frame "
            f"constants are no longer consistent with _GATE_POS_ZUP / _GATE_YAW_ZUP.")
    if not np.allclose(yaw_rel, _GATE_YAW_REL, atol=1e-9):
        raise AssertionError(
            f"_GATE_YAW_REL diverges from the per-gate rel_tables on VQ1 "
            f"(max|diff|={float(np.abs(yaw_rel - _GATE_YAW_REL).max()):.2e}).")


def _assert_live_course_is_vq1(track_gates, *, where: str = "fly_rl deploy",
                               pos_tol_m: float = 2.0) -> None:
    """Deploy-time LOUD guard (P4-C05): the live RL loop builds obs through the hardcoded
    yaw=π VQ1 gate frame (gate_map=None). Before trusting it, verify the sim is actually
    broadcasting the VQ1 course — same gate count, and each gate within ``pos_tol_m`` of the
    hardcoded _GATE_POS_ZUP (gates are ~24 m apart, so 2 m never false-fires on VQ1 but catches
    a different/VQ2 course). On a genuinely different course the hardcoded yaw=π frame would
    silently corrupt the obs by metres; the fix is to thread a live gate_map (make_gate_map).
    No-op when no track map has arrived yet (warn only — the legacy behaviour)."""
    if not track_gates:
        print(f"  [P4-C05 guard] no TRACK_INFO yet -> cannot verify the live course is VQ1; "
              f"proceeding on the hardcoded yaw=π gate frame (valid only if VQ1).",
              file=sys.stderr)
        return
    if len(track_gates) != N_GATES:
        raise AssertionError(
            f"[{where}] P4-C05 GUARD: live course has {len(track_gates)} gates, hardcoded VQ1 "
            f"obs expects {N_GATES}. The yaw=π gate frame is invalid here — thread a live "
            f"gate_map (make_gate_map) into build_obs.")
    by_id = sorted(track_gates, key=lambda g: int(g["gate_id"]))
    live_zup = np.asarray([g["position_ned"] for g in by_id], dtype=np.float64) * _FLIP
    dpos = float(np.abs(live_zup - _GATE_POS_ZUP).max())
    if dpos > pos_tol_m:
        raise AssertionError(
            f"[{where}] P4-C05 GUARD: live gate positions deviate from the hardcoded VQ1 course "
            f"by up to {dpos:.2f} m (> {pos_tol_m} m) — this is NOT the all-π VQ1 course the "
            f"hardcoded yaw=π obs frame is valid for. Thread a live gate_map (make_gate_map) "
            f"into build_obs; the hardcoded path would silently corrupt the gate-relative obs.")


_assert_vq1_constants_consistent()


# ---------------------------------------------------------------------------
# Math helpers (no pytorch3d dependency on ShadowPC)
# ---------------------------------------------------------------------------
def _euler_zyx(R: np.ndarray) -> np.ndarray:
    """ZYX Euler [roll, pitch, yaw] from rotation matrix.
    Matches pytorch3d matrix_to_euler_angles(R, 'ZYX')[..., [2,1,0]] (verified
    against the pytorch3d _angle_from_tan source: yaw=atan2(R10,R00),
    pitch=asin(-R20), roll=atan2(R21,R22))."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    yaw   = float(np.arctan2(R[1, 0], R[0, 0]))
    roll  = float(np.arctan2(R[2, 1], R[2, 2]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


# ---------------------------------------------------------------------------
# Observation builder — EXACT match to peregrine_racing.get_observations()
# ---------------------------------------------------------------------------
def obs_from_zup(pos_zup: np.ndarray, vel_zup: np.ndarray, R_b2w_zup: np.ndarray,
                 w_flu: np.ndarray, target_gate: int,
                 last_normed_thrust: float, virtual_flip: bool = False,
                 gate_map: GateMap | None = None) -> np.ndarray:
    """17-dim obs (float32) from TRUE state already in the DiffAero Z-up/FLU frame.

    Layout (matching peregrine_racing.py):
      [0:3]   pos_g        R_w2g @ (gate_pos - pos)          gate-relative position
      [3:6]   vel_g        R_w2g @ vel                        velocity, gate frame
      [6:9]   rpy_g        ZYX euler of (R_w2g @ R_b2w)       attitude vs gate
      [9:12]  body_rates   FLU body rates
      [12]    prev_normed_thrust  previous RESCALED normed_thrust in [0, act_max] g-units
                                  (0.0 at episode start) -- NOT the [0,1] wire collective
      [13:16] next_relpos  pre-computed next-gate relative position
      [16]    next_relyaw  inter-gate yaw delta (0 on VQ1; nonzero on a bent course)

    The gate frame follows ``gate_map`` per-gate yaw (P4-C05 fix). ``gate_map=None`` uses the
    hardcoded VQ1 constants (_R_W2G/_GATE_REL_POS/_GATE_YAW_REL) — bit-identical to the legacy
    builder and correct ONLY because every VQ1 gate yaw is π. Pass an explicit gate_map (e.g.
    make_gate_map(pos, yaw)) for any non-π / VQ2 / course_mode=random course; the yaw-aware path
    reproduces peregrine_racing.get_observations EXACTLY (the obs the policy was trained on).
    """
    if virtual_flip:
        R_b2w_zup = R_b2w_zup @ _RZ_PI_BODY     # body axes rotated π about body z
        w_flu = _RZ_PI_BODY @ w_flu
    nxt = min(target_gate + 1, N_GATES - 1)
    if gate_map is None:
        # LEGACY VQ1 path — EXACT yaw=π specialization (no cos/sin epsilon). Bit-identical to the
        # pre-P4-C05 builder; valid only on the all-π course (guarded at import + via the deploy
        # all-π check). Same constant objects + arithmetic as before -> old==new is exactly 0.
        R_w2g      = _R_W2G
        gp         = _GATE_POS_ZUP[target_gate]
        nxt_rel    = _GATE_REL_POS[nxt]
        nxt_relyaw = _GATE_YAW_REL[nxt]
    else:
        # YAW-AWARE path — per-gate gate frame from the runtime course (matches
        # peregrine_racing.get_observations: get_gate_rotmat_w2g(gate_yaw[tg]) + rel_tables).
        R_w2g      = _gate_rotmat_w2g(float(gate_map.gate_yaw[target_gate]))
        gp         = gate_map.gate_pos[target_gate]
        nxt_rel    = gate_map.gate_rel_pos[nxt]
        nxt_relyaw = gate_map.gate_yaw_rel[nxt]
    pos_g = R_w2g @ (gp - pos_zup)
    vel_g = R_w2g @ vel_zup
    rpy_g = _euler_zyx(R_w2g @ R_b2w_zup)
    obs = np.concatenate([
        pos_g, vel_g, rpy_g, w_flu,
        [last_normed_thrust],
        nxt_rel,
        [nxt_relyaw],
    ])
    return obs.astype(np.float32)


def build_obs(state, target_gate: int, last_normed_thrust: float,
              virtual_flip: bool = False, gate_map: GateMap | None = None) -> np.ndarray:
    """Telemetry -> 17-dim obs: conjugate the ODOMETRY quat to the TRUE attitude
    (R_y(pi) telemetry frame -- see _ODO_QUAT_TRUE_CONJ), negate the raw rates to
    TRUE FRD, convert NED/FRD -> Z-up/FLU, then obs_from_zup.

    ``gate_map`` threads the runtime per-gate yaw through to obs_from_zup (P4-C05). None =
    the bit-exact hardcoded VQ1 path; pass make_gate_map(...) for a non-π course."""
    from scipy.spatial.transform import Rotation as _Rot

    pos_ned = np.asarray(state.position_ned,         dtype=np.float64)
    vel_ned = np.asarray(state.velocity_ned,         dtype=np.float64)
    q_raw   = np.asarray(state.orientation_ned_wxyz, dtype=np.float64)
    w_raw   = np.asarray(state.angular_rate_body,    dtype=np.float64)

    # Attitude: telemetry conjugation -> true quat (wxyz scalar-first -> scipy xyzw).
    q_true = q_raw * _ODO_QUAT_TRUE_CONJ
    R_frd2ned = _Rot.from_quat([q_true[1], q_true[2], q_true[3], q_true[0]]).as_matrix()
    R_b2w_zup = (_FLIP[:, None] * R_frd2ned) * _FLIP[None, :]   # FLU -> Z-up world

    w_frd = w_raw * _ODO_RATE_SIGN              # raw -> true FRD body rates
    return obs_from_zup(pos_ned * _FLIP, vel_ned * _FLIP, R_b2w_zup,
                        w_frd * _FLIP, target_gate, last_normed_thrust,
                        virtual_flip=virtual_flip, gate_map=gate_map)


def telemetry_health(state, now_ns: int, stale_s: float) -> tuple[str, float]:
    """Classify a DroneState snapshot for the deploy loop's F-C gate (audit D1/D2/R4/R5).

    Returns ``(status, odo_age_s)`` where status is one of:
      "no_fix"     -- a flight field is still None (pre-first-ODOMETRY warmup; skip the tick)
      "stale"      -- ODOMETRY's per-field arrival age exceeds ``stale_s`` (selective drop;
                      attitude/rate are frozen -> flying on them is open-loop divergence)
      "non_finite" -- a NaN/inf component, or a zero-norm quat (would raise in build_obs or
                      push NaN to the wire)
      "ok"         -- fresh + finite; safe to build_obs + command.

    Pure function of the snapshot so the gate is unit-testable WITHOUT a live socket. The
    None check is FIRST (pre-fix warmup is normal, must not arm the recovery timer); the
    freshness check uses ``odo_recv_ns`` (NOT the shared ``recv_monotonic_ns``, which LPN/IMU
    also bump); finiteness covers pos/vel/quat/rate and rejects a zero-norm quat (q.q>1e-12)."""
    if (state.position_ned is None or state.velocity_ned is None
            or state.orientation_ned_wxyz is None or state.angular_rate_body is None):
        return "no_fix", float("inf")
    odo_recv = int(state.odo_recv_ns)
    age_s = (now_ns - odo_recv) / 1e9 if odo_recv > 0 else float("inf")
    if age_s > stale_s:
        return "stale", age_s
    q = np.asarray(state.orientation_ned_wxyz, dtype=np.float64)
    finite = bool(
        np.all(np.isfinite(np.asarray(state.position_ned, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(state.velocity_ned, dtype=np.float64)))
        and np.all(np.isfinite(q)) and float(q @ q) > 1e-12
        and np.all(np.isfinite(np.asarray(state.angular_rate_body, dtype=np.float64))))
    if not finite:
        return "non_finite", age_s
    return "ok", age_s


# ---------------------------------------------------------------------------
# Policy inference
# ---------------------------------------------------------------------------

class _LNBlock(nn.Module):
    """Linear -> LayerNorm -> ELU  (diffaero utils.nn.NormedLinear)."""
    def __init__(self, a: int, b: int):
        super().__init__()
        self.linear = nn.Linear(a, b)
        self.ln     = nn.LayerNorm(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.ln(self.linear(x)))


class _ActorMean(nn.Module):
    """obs_dim -> 256 -> 128 -> 4 MLP with LN blocks (cfg/network/mlp.yaml hidden_dim).

    obs_dim defaults to 17 (the inc7 contract); inc8 checkpoints carry obs_dim=20
    (the d5 confidence triple [17:20]). load_actor infers the width from the
    checkpoint's first layer so a 20-dim inc8 actor loads without a hand-set width."""
    def __init__(self, obs_dim: int = 17):
        super().__init__()
        self.head = nn.Sequential(
            _LNBlock(obs_dim, 256),
            _LNBlock(256, 128),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _apply_checkpoint_sidecar(path: str) -> None:
    """Per-checkpoint training constants (S1.4+): a JSON sidecar next to the .pth, e.g.
    ``stage1_inc4_actor.json`` beside ``stage1_inc4_actor.pth``, carrying the action bounds the
    checkpoint was TRAINED with. S1.3+ train with ``max_normed_thrust=3.765`` (the live collective
    ceiling) while the original constant here was quad.yaml's 5.0 -- deploying a 3.765-trained
    actor through a [0,5] rescale overdrives every thrust command by up to 33%. The sidecar makes
    the rescale follow the checkpoint instead of trusting a hand-synced constant.
    Mutates _ACT_MAX/_ACT_MIN IN PLACE so policy_step and every importer see it."""
    import json as _json
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        print(f"[load_actor] WARNING: no sidecar {sidecar.name} -- assuming the LEGACY action "
              f"bounds thrust [0,{_ACT_MAX[0]:.3f}] / rates +-{_ACT_MAX[1]:.2f}. Correct ONLY "
              f"for checkpoints trained with quad.yaml defaults (stage1_inc1). S1.3+ trained "
              f"with max_normed_thrust=3.765 -- deploying one without its sidecar overdrives "
              f"every thrust command up to 33% AND corrupts the obs[12] collective feedback.")
        return
    meta = _json.loads(sidecar.read_text())
    if "act_max_thrust" in meta:
        _ACT_MAX[0] = float(meta["act_max_thrust"])
    if "act_max_rate" in meta:
        _ACT_MAX[1:4] = float(meta["act_max_rate"])
        _ACT_MIN[1:4] = -float(meta["act_max_rate"])
    print(f"[load_actor] sidecar {sidecar.name}: action bounds -> "
          f"thrust [{_ACT_MIN[0]:.3f},{_ACT_MAX[0]:.3f}] rates +-{_ACT_MAX[1]:.2f} rad/s")


def load_actor(path: str) -> nn.Module:
    """Load actor.pth.  Handles both a full nn.Module save and the dict format
    {'actor_mean': state_dict, 'actor_logstd': tensor} produced by DiffAero PPO.
    Applies the checkpoint's JSON sidecar (training action bounds) if present."""
    _apply_checkpoint_sidecar(path)
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "actor_mean" in d:
        # Infer the obs width from the checkpoint's first layer (17 inc7 / 20 inc8): the
        # head.0.linear.weight columns ARE the obs dim, so a 20-dim inc8 actor loads
        # without a hand-set width and without trusting the sidecar.
        _w0 = d["actor_mean"].get("head.0.linear.weight")
        _obs_dim = int(_w0.shape[1]) if _w0 is not None else 17
        actor = _ActorMean(_obs_dim)
        miss  = actor.load_state_dict(d["actor_mean"], strict=True)
        if miss.missing_keys or miss.unexpected_keys:
            raise RuntimeError(f"state_dict mismatch: missing={miss.missing_keys} "
                               f"unexpected={miss.unexpected_keys}")
        actor.eval()
        return actor
    # fallback: checkpoint is already an nn.Module
    if hasattr(d, "eval"):
        d.eval()
    return d


# ---------------------------------------------------------------------------
# EGO (21-dim egocentric) actor loading — HARDCODED action bounds.
# ---------------------------------------------------------------------------
# The ego generation trains with the sbatch override dynamics.controller.max_normed_thrust=3.765
# (peregrine_vq2_ego.sbatch:80 on the cluster; the quad.yaml default 5.0 is OVERRIDDEN) and the
# quad.yaml body-rate bound ±3.14 rad/s. The ego launcher (peregrine_train_ego.py) calls plain
# agent.save — NO inc8-style JSON sidecar carrying these — so the generic sidecar path would fall
# through to the LEGACY [0,5] thrust bounds and OVERDRIVE every thrust command by ~33% AND corrupt
# the obs[8] collective feedback. The ego path therefore HARDCODES the trained bounds and never
# consults a sidecar. [ego-deploy 2026-07-09; verified against the cluster config + launch echoes]
_EGO_ACT_MAX_THRUST = 3.765   # g-units; sbatch dynamics.controller.max_normed_thrust
_EGO_ACT_MAX_RATE = 3.14      # rad/s per axis, FLU; cluster cfg/dynamics/quad.yaml
_EGO_OBS_DIM = 21             # racer.ego_obs.EGO_OBS_DIM (WINDOW=2 egocentric contract)


def _apply_ego_action_bounds() -> None:
    """Mutate the module action bounds to the EGO trained values (in place, like the sidecar
    path, so policy_step and every importer see them). Factored out for the unit tests."""
    _ACT_MIN[0] = 0.0
    _ACT_MAX[0] = _EGO_ACT_MAX_THRUST
    _ACT_MAX[1:4] = _EGO_ACT_MAX_RATE
    _ACT_MIN[1:4] = -_EGO_ACT_MAX_RATE
    print(f"[load_ego_actor] EGO action bounds HARDCODED: thrust [0,{_EGO_ACT_MAX_THRUST}] "
          f"rates +-{_EGO_ACT_MAX_RATE} rad/s (no sidecar consulted; the ego launcher writes "
          f"none and the legacy [0,5] fallback would overdrive thrust ~33%).")


def load_ego_actor(path: str) -> nn.Module:
    """Load a 21-dim egocentric actor.pth ({'actor_mean': state_dict, ...} DiffAero format).

    Unlike ``load_actor`` this NEVER reads a JSON sidecar and NEVER falls back to the legacy
    [0,5] thrust bounds — the ego bounds are hardcoded (see _EGO_ACT_MAX_THRUST above). The obs
    width is asserted == 21 (the WINDOW=2 egocentric contract this deploy adapter builds); a
    mismatched checkpoint fails LOUD at startup instead of flying a garbled obs."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(d, dict) and "actor_mean" in d):
        raise SystemExit(f"--ego-ckpt {path!r} is not a DiffAero actor checkpoint "
                         f"({{'actor_mean': ...}} dict expected).")
    w0 = d["actor_mean"].get("head.0.linear.weight")
    obs_dim = int(w0.shape[1]) if w0 is not None else -1
    if obs_dim != _EGO_OBS_DIM:
        raise SystemExit(
            f"--ego-ckpt obs width {obs_dim} != the {_EGO_OBS_DIM}-dim egocentric contract this "
            f"adapter builds (racer.ego_obs). Wrong checkpoint generation? (inc7=17, inc8=20.)")
    actor = _ActorMean(obs_dim)
    miss = actor.load_state_dict(d["actor_mean"], strict=True)
    if miss.missing_keys or miss.unexpected_keys:
        raise RuntimeError(f"ego state_dict mismatch: missing={miss.missing_keys} "
                           f"unexpected={miss.unexpected_keys}")
    actor.eval()
    _apply_ego_action_bounds()
    return actor


@torch.no_grad()
def policy_step(actor: nn.Module, obs_np: np.ndarray, max_rate: float = 0.0,
                virtual_flip: bool = False, max_thrust: float = 0.0,
                debug: dict | None = None, yaw_scale: float = 1.0
                ) -> tuple[np.ndarray, float, float]:
    """One forward pass, replicating the TRAINING action pipeline exactly:
    test-mode action = tanh(actor_mean(obs)), then env.rescale_action onto
    [_ACT_MIN, _ACT_MAX].

    Returns:
      body_rate_frd  (3,)  rad/s, FRD  — ready for ControlCommand.body_rate
      collective     float [0,1]       — ready for ControlCommand.thrust (the wire collective)
      normed_thrust  float [0,act_max] — RESCALED action[0] in g-units; becomes the next
                                         obs[12] (NOT the [0,1] collective above)

    ``debug``: pass a dict to receive the pipeline internals (raw mean, tanh,
    rescaled action) for the --debug-obs per-step dump.

    ``yaw_scale``: scale on the policy's yaw-rate command (S17 mixer mitigation).
    The policy dithers yaw at the action rail every tick — net yaw ≈ 0 in-twin, but
    live the never-converging yaw demand is the largest motor-differential source,
    and at low collective the mixer's idle-floor clipping turns it into ~0.2-0.4 of
    PARASITIC collective (measured: mixer_probe 2026-06-11). 0 = drop yaw entirely.
    """
    obs_t = torch.as_tensor(obs_np[None], dtype=torch.float32)   # (1, 17)
    mean  = actor(obs_t)[0].cpu().numpy().astype(np.float64)     # (4,) raw mean
    a     = np.tanh(mean)                                        # training squash
    act   = _ACT_MIN + (_ACT_MAX - _ACT_MIN) * (a + 1.0) / 2.0   # rescale_action

    normed_thrust = float(act[0])
    if max_thrust > 0.0:
        normed_thrust = min(normed_thrust, max_thrust)           # OOD start cap
    rate_flu = act[1:4]
    if max_rate > 0.0:
        rate_flu = np.clip(rate_flu, -max_rate, max_rate)        # OOD diagnostic cap
    if yaw_scale != 1.0:
        rate_flu = rate_flu.copy()
        rate_flu[2] *= yaw_scale       # body z is body z under both flip and FLU->FRD
    if virtual_flip:
        rate_flu = _RZ_PI_BODY @ rate_flu       # virtual body frame -> real body
    rate_frd = rate_flu * _ACT_FLU_TO_FRD
    collective = float(np.clip(normed_thrust * _HOVER_THRUST, 0.0, 1.0))
    if debug is not None:
        debug["actor_mean"] = mean.tolist()
        debug["tanh"] = a.tolist()
        debug["act_rescaled"] = act.tolist()
    return rate_frd, collective, normed_thrust


# ---------------------------------------------------------------------------
# PATH B: CTBR bridge — the flight-proven model-based launcher (fly_vq1 faithful
# stack: Navigator + ReactivePlanner + Mission + launch ramp) flies takeoff ->
# gate-0 approach; the RL policy takes over at the offline-validated seam
# (<3 m along-track of gate 0, >4 m/s; validated envelope 4-12 m/s, 3-5 m).
# ---------------------------------------------------------------------------
def build_bridge(map_path: str):
    """The exact canonical-6/6 stack (data/runs/20260607_194615_course_60s meta):
    make_controller(_FAITHFUL_SIGNS, **FAITHFUL_TUNED_GAINS) + faithful planner
    (course yaw) + saved map corner->center + launch_ramp_s 0.6."""
    from racer.mission import Mission, MissionConfig
    from racer.navigator import Navigator, NavigatorConfig, load_track_map
    from racer.planner import ReactivePlanner
    from twin_fly_course import _FAITHFUL_SIGNS, make_controller
    from twin_tune import FAITHFUL_TUNED_GAINS, FAITHFUL_TUNED_PLANNER

    gates = load_track_map(map_path, corner_to_center=True)
    nav = Navigator(gates=gates, detector=None,
                    config=NavigatorConfig(use_vision=False, use_given_position=True))
    mission = Mission(
        gates=gates,
        planner=ReactivePlanner(yaw_mode="course", **FAITHFUL_TUNED_PLANNER),
        controller=make_controller(signs=_FAITHFUL_SIGNS, **FAITHFUL_TUNED_GAINS),
        config=MissionConfig(takeoff_altitude_m=1.5, takeoff_tol_m=0.3,
                             gate_pass_radius_m=0.75),
    )
    return nav, mission


def run_bridge(client, args, gate0_x_ned: float) -> str:
    """Fly the CTBR bridge until the handoff seam.  Returns 'HANDOFF' on success,
    else a terminal state string.  Runs at the CTBR-validated 100 Hz."""
    from dataclasses import replace as _dc_replace

    from racer.mission import MissionState

    nav, mission = build_bridge(args.map)
    mission.start()
    print(f"[bridge] CTBR faithful stack -> handoff at along-track<{args.handoff_dist:g} m "
          f"& speed>{args.handoff_speed_min:g} m/s (gate0 x_ned={gate0_x_ned:.1f}) ...")

    tick      = 1.0 / 100.0
    deadline  = time.monotonic() + args.bridge_max_s
    next_t    = time.monotonic()
    last_sim  = int(client.state.sim_time_ns)
    last_adv  = time.monotonic()
    last_p    = 0.0
    n_coll0   = len(client.collisions)

    while time.monotonic() < deadline:
        while time.monotonic() < next_t:
            client.pump()
            time.sleep(0.001)
        client.pump()
        next_t = time.monotonic() + tick
        now = time.monotonic()

        st = int(client.state.sim_time_ns)
        if st > last_sim:
            last_sim = st
            last_adv = now
        elif now - last_adv > 1.5:
            print("\n  [bridge] sim_time stalled -> stopping.")
            return "STALLED"
        if any(c["threat_level"] >= 2 for c in client.collisions[n_coll0:]):
            print("\n  [bridge] HARD COLLISION -> abort.")
            return "CRASH"

        gs = client.state
        if gs.position_ned is None or gs.velocity_ned is None:
            continue
        ns = nav.update(gs, None)
        # control on the RAW given pos+vel (the VQ1-proven choice; KF vz lags)
        ns = _dc_replace(ns,
                         position_ned=np.asarray(gs.position_ned, dtype=np.float64),
                         velocity_ned=np.asarray(gs.velocity_ned, dtype=np.float64))

        # handoff seam check (along-track distance to the gate-0 plane, +x_ned side)
        along = float(ns.position_ned[0]) - gate0_x_ned
        speed = float(np.linalg.norm(ns.velocity_ned))
        if (mission.state == MissionState.RUN
                and along < args.handoff_dist and speed > args.handoff_speed_min):
            print(f"\n  [bridge] HANDOFF  along={along:+.2f} m  speed={speed:.1f} m/s  "
                  f"pos=({ns.position_ned[0]:+.1f},{ns.position_ned[1]:+.1f},"
                  f"{ns.position_ned[2]:+.1f})")
            return "HANDOFF"

        client.send_command(mission.step(ns))

        if now - last_p >= 1.0:
            print(f"  [bridge] {mission.state.name:<7} along={along:+6.1f} m "
                  f"speed={speed:4.1f} m/s alt={-ns.position_ned[2]:+.1f} m   ",
                  end="\r", flush=True)
            last_p = now

    print("\n  [bridge] no handoff within bridge-max-s -> abort.")
    return "BRIDGE_TIMEOUT"


# ---------------------------------------------------------------------------
# Sim window control (unattended full reset, S17 harness hardening)
# ---------------------------------------------------------------------------
_VK_RETURN, _VK_ESCAPE, _VK_DOWN, _KEYUP = 0x0D, 0x1B, 0x28, 0x0002


def _find_sim_window(title_substr: str = "AI-GP") -> int | None:
    """HWND of the first visible window whose title contains ``title_substr``
    (falls back to the FlightSim binary name)."""
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _lp):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                t = buf.value.lower()
                if title_substr.lower() in t or "flightsim" in t or "ai grand prix" in t:
                    found.append(hwnd)
        return True

    user32.EnumWindows(_cb, None)
    return found[0] if found else None


def _send_key(vk: int, hold_s: float = 0.06, settle_s: float = 0.25) -> None:
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(hold_s)
    user32.keybd_event(vk, 0, _KEYUP, 0)
    time.sleep(settle_s)


def _force_foreground(hwnd, attempts: int = 3) -> bool:
    """Restore-if-minimized + VERIFIED foreground. The fullscreen sim AUTO-MINIMIZES
    when it loses focus (observed mid-batch 2026-06-11: every subsequent keybd_event
    landed in whatever was focused instead -> NO_GO chain), and SetForegroundWindow
    from a background process is refused unless wrapped in a synthetic ALT press."""
    user32 = ctypes.windll.user32
    for _ in range(attempts):
        user32.ShowWindow(hwnd, 9)               # SW_RESTORE (no-op if not minimized)
        time.sleep(0.3)
        user32.keybd_event(0x12, 0, 0, 0)        # ALT down: unlock SetForegroundWindow
        user32.SetForegroundWindow(hwnd)
        user32.keybd_event(0x12, 0, _KEYUP, 0)
        time.sleep(0.4)
        if user32.GetForegroundWindow() == hwnd:
            return True
    return False


def full_sim_reset() -> bool:
    """The BETWEEN-FLIGHTS full reset (S17 mandate): ESC + Down*3 + Enter exits the
    race to HOME, then Enter*2 starts a fresh waiting room -> race countdown. Unlike
    MAV_CMD 31000 (in-race restart), this clears all race residue, so every flight
    starts from an identical fresh countdown. Needs the sim window (Win32 focus)."""
    hwnd = _find_sim_window()
    if hwnd is None:
        print("  full-reset: sim window not found -> falling back to MAV_CMD 31000",
              file=sys.stderr)
        return False
    if not _force_foreground(hwnd):
        print("  full-reset: could not verify sim foreground -> skipping key sends",
              file=sys.stderr)
        return False
    _send_key(_VK_ESCAPE, settle_s=0.6)          # pause menu
    for _ in range(3):
        _send_key(_VK_DOWN)                      # highlight "exit to home"
    _send_key(_VK_RETURN, settle_s=2.5)          # confirm -> HOME
    for _ in range(2):
        _send_key(_VK_RETURN, settle_s=1.5)      # HOME -> waiting room -> race
    return True


def kick_sim_from_home(n_enter: int = 2, settle_s: float = 1.5) -> bool:
    """HOME/parked-page recovery (rate_sysid S1.2 mechanics): MAV_CMD 31000 is a no-op
    on the home/off-race screens, so focus the window and send Enter*n."""
    hwnd = _find_sim_window()
    if hwnd is None:
        print("  home-kick: sim window not found", file=sys.stderr)
        return False
    if not _force_foreground(hwnd):
        print("  home-kick: could not verify sim foreground", file=sys.stderr)
        return False
    for _ in range(n_enter):
        _send_key(_VK_RETURN, settle_s=settle_s)
    return True


# ---------------------------------------------------------------------------
# Race lifecycle helpers
# ---------------------------------------------------------------------------
def wait_fresh_go(client, args, auto_reset: bool) -> bool:
    """Pump until a FRESH race GO (race_start within the last 2 s and past the
    start margin, drone near origin).  If auto_reset, fire send_sim_reset()
    (MAV_CMD 31000) whenever no fresh GO shows up for --reset-after seconds.

    F-D/N1: when NOT auto_reset (the submission-safe passive posture) there is no reset
    to fall back on, so a LATE-JOINED race (countdown elapsed > 2 s ago) would otherwise
    spin to the deadline as a silent NO_GO. In that config we also accept the first-seen
    STARTED race that is still at the start (gate 0, not finished, drone at origin)."""
    deadline   = time.monotonic() + args.wait_seconds
    margin_ms  = args.start_margin_s * 1000.0
    next_reset = time.monotonic() + (args.reset_after if auto_reset else 1e18)
    last_p     = 0.0
    stale_strikes = 0     # consecutive ineffective 31000s (sim parked off-race / at HOME)
    while time.monotonic() < deadline:
        client.pump()
        s  = client.state
        rs = client.race_status
        # F-D / VQ2 (live-confirmed 2026-06-29): a self-localizing wire DENIES raw position_ned (and
        # attitude). Liveness must NOT gate on wire position -- the sim is "live" whenever its master
        # clock is advancing (sim_time_ns > 0). RACE_STATUS (started + active_gate_index) is the
        # start-line authority; the drone-at-origin distance guards only apply when a position exists.
        have_pos = s.position_ned is not None
        live = s.sim_time_ns > 0
        now  = time.monotonic()
        if rs and rs["started"] and live:
            to_go = rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]
            fresh = rs["race_start_boot_time_ms"] >= 0 and to_go > -2000.0
            if fresh and to_go <= -margin_ms:
                # Stale-GO-by-distance is only knowable WITH a position; on a position-denied wire fall
                # back to the RACE_STATUS gate-0 check below (a fresh countdown that just elapsed at
                # gate 0 is a genuine GO). pos_off=0.0 when position is denied -> never spuriously stale.
                pos_off = float(np.linalg.norm(s.position_ned)) if have_pos else 0.0
                gi0 = rs.get("active_gate_index")
                at_start = gi0 is None or int(gi0) == 0
                if have_pos and pos_off > 5.0:
                    print(f"\n  stale GO: drone {pos_off:.0f} m from origin — waiting on.",
                          file=sys.stderr)
                elif not have_pos and not at_start:
                    print(f"\n  stale GO: position-denied wire already past gate 0 (gi={gi0}) "
                          "— waiting on.", file=sys.stderr)
                else:
                    print(f"\n  GO!  pos_off={pos_off:.2f} m.  {telemetry_summary(client)}")
                    return True
            elif fresh:
                # a fresh countdown is ticking — never fire a reset into it
                next_reset = max(next_reset, now + args.reset_after)
                stale_strikes = 0
                if now - last_p >= 0.25:
                    print(f"  countdown {to_go/1000:+.2f}s   ", end="\r", flush=True)
                    last_p = now
            elif (not auto_reset and rs["race_start_boot_time_ms"] >= 0
                    and to_go <= -2000.0):
                # F-D/N1: passive late-join. The countdown elapsed > 2 s ago (we attached into
                # a running race, or GO fired before our first RACE_STATUS). With no reset to
                # fall back on, accept it iff the race is still at the start line.
                gi = rs.get("active_gate_index")
                at_gate0 = gi is None or int(gi) == 0
                not_finished = not rs.get("finished")
                # position-denied wire (VQ2): pos_off unknown -> 0.0, and the gate-0 + not-finished
                # RACE_STATUS check is the start-line authority on its own.
                pos_off = float(np.linalg.norm(s.position_ned)) if have_pos else 0.0
                if at_gate0 and not_finished and pos_off <= 5.0:
                    print(f"\n  LATE-JOIN GO!  to_go={to_go/1000:+.2f}s  pos_off={pos_off:.2f} m"
                          f"  gi={gi}.  {telemetry_summary(client)}")
                    return True
                elif now - last_p >= 1.0:
                    print(f"\n  late race not joinable (to_go={to_go/1000:+.1f}s gi={gi} "
                          f"finished={rs.get('finished')} pos_off={pos_off:.0f} m) — waiting.",
                          file=sys.stderr)
                    last_p = now
        if now >= next_reset:
            stale_strikes += 1
            if stale_strikes >= 3:
                # 31000 is a no-op at HOME / on the parked off-race screen (measured
                # 2026-06-10) -> escalate to the Win32 focus + Enter*2 recovery.
                print("\n  no fresh GO after repeated resets -> home-kick (Enter*2) ...")
                kick_sim_from_home()
                stale_strikes = 0
            else:
                print("\n  no fresh GO -> requesting sim reset (MAV_CMD 31000) ...")
                try:
                    client.send_sim_reset()
                except Exception as exc:
                    print(f"  sim reset send failed: {exc}", file=sys.stderr)
            next_reset = now + args.reset_after
        elif now - last_p >= 2.0:
            print(f"  waiting: started={bool(rs and rs['started'])} "
                  f"pos={'y' if live else 'n'} map={len(client.track_gates or [])}   ",
                  end="\r", flush=True)
            last_p = now
        time.sleep(0.005)
    # F-D: log WHY we gave up (to_go + started visibility for the live operator).
    rs = client.race_status
    if rs:
        to_go = rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]
        print(f"\n  wait_fresh_go: {args.wait_seconds:g}s deadline, no acceptable GO "
              f"(started={rs.get('started')} finished={rs.get('finished')} "
              f"to_go={to_go/1000:+.2f}s gi={rs.get('active_gate_index')} "
              f"auto_reset={auto_reset}).", file=sys.stderr)
    else:
        print(f"\n  wait_fresh_go: {args.wait_seconds:g}s deadline, no RACE_STATUS seen.",
              file=sys.stderr)
    return False


def fly_once(client, actor, args, flight_idx: int,
             session_dir: Path | None = None) -> dict:
    """One full attempt: wait fresh GO -> arm -> RL control loop -> finish hold ->
    disarm.  Returns a per-flight result dict.  Caller owns recorder lifecycle."""
    from pymavlink import mavutil

    result = {"flight": flight_idx, "final_state": "NO_GO", "gate_index": 0,
              "collisions_at_start": len(client.collisions)}

    # ------------------------------------------------------------------
    # Fresh race GO. F-A/R1: the submitted path emits NO sim-control command --
    # auto_reset is opt-in via --dev-auto-reset (dev rig only) and a hard
    # --no-auto-reset always wins. Default = passive wait for the organizer's GO.
    # ------------------------------------------------------------------
    auto_reset = bool(args.dev_auto_reset) and not args.no_auto_reset
    if flight_idx == 1:
        if auto_reset:
            print(">>> [DEV] auto-reset ON: will request a fresh race via MAV_CMD 31000.")
        else:
            print(">>> Waiting PASSIVELY for the race GO (no sim-control command will be "
                  "sent; this is the submission-safe posture).")
    if not wait_fresh_go(client, args, auto_reset=auto_reset):
        print("no GO -> abort flight.", file=sys.stderr)
        return result

    # ------------------------------------------------------------------
    # Arm (F-D / AR1: bounded re-send + force-arm last resort). The original single
    # arm() declared ARM_REFUSED on any transient MAV_RESULT_TEMPORARILY_REJECTED or
    # a slow SAFETY_ARMED bit -- a recoverable rejection ended the one-shot run. We
    # re-send up to --arm-attempts times (branching on the ACK), escalating the final
    # attempt to arm(force=True) (the 21196 pre-arm bypass), all well inside the cap.
    # ------------------------------------------------------------------
    print("\n[arm] ...")
    armed = False
    for attempt in range(1, args.arm_attempts + 1):
        force = attempt == args.arm_attempts and args.arm_attempts > 1   # last try escalates
        client.last_command_ack = None
        client.arm(force=force)
        ack   = client.wait_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout_s=3.0)
        armed = client.wait_armed(True, timeout_s=5.0)
        rn = ack["result_name"] if ack else "none"
        print(f"  attempt {attempt}/{args.arm_attempts}"
              f"{' (force)' if force else ''}: ACK={rn}  armed={armed}")
        if armed:
            break
        if attempt < args.arm_attempts:
            print(f"  arm not confirmed (ack={rn}) -> retrying in {args.arm_backoff_s:g}s ...",
                  file=sys.stderr)
            t_back = time.monotonic() + args.arm_backoff_s
            while time.monotonic() < t_back:
                client.pump()          # keep heartbeats/acks flowing during backoff
                time.sleep(0.01)
    if not armed:
        print(f"  arming refused after {args.arm_attempts} attempts -> abort flight.",
              file=sys.stderr)
        result["final_state"] = "ARM_REFUSED"
        return result

    # ------------------------------------------------------------------
    # ARMED: every exit from here MUST disarm (F-B). A finally wrapping the whole
    # bridge + RL loop + finish-hold converts EVERY crash path (missing map,
    # degenerate-quat ValueError, recorder OSError, ...) into a DISARMED exit
    # instead of leaving the vehicle armed with the last command latched. Mirrors
    # fly_vq1.py's finally-disarm. (autonomy-readiness audit F-B / R2,R3,R4,n3.)
    # ------------------------------------------------------------------
    try:
        return _fly_armed(client, actor, args, flight_idx, session_dir, result)
    finally:
        print("\n[safety] disarming ...")
        try:
            client.disarm(force=True)
            client.wait_armed(False, timeout_s=3.0)
        except Exception as exc:
            print(f"  disarm error: {exc}", file=sys.stderr)


def _resolve_seeker_weights(args) -> str | None:
    """The weights spec the yolo gate-seeker detector will load: ``--seeker-weights`` if set, else the
    legacy fallback to ``--checkpoint``. Single source of truth for the loaders + the startup guard."""
    return args.seeker_weights or args.checkpoint


def _looks_like_detector_weights(spec) -> bool:
    """True when ``spec`` looks like an ultralytics YOLO weights file (``.pt``, or an ``a.pt++b.pt``
    ensemble spec) rather than the RL-actor ``.pth`` checkpoint. Every member of an ensemble spec must
    look like a detector weight. The RL actor default is ``stage1_inc7_actor.pth`` -> False (the footgun)."""
    if not spec:
        return False
    members = [s.strip().lower() for s in str(spec).split("++") if s.strip()]
    return bool(members) and all(m.endswith(".pt") for m in members)


def _validate_seeker_detector(args) -> None:
    """FAIL LOUD at startup if the gate-seeker yolo path has no usable detector weights (2026-07-01
    footgun fix). With ``--seeker-detector`` now defaulting to ``yolo``, a bare ``--gate-seeker`` run
    would otherwise silently fall back to ``--checkpoint`` (the RL-actor ``.pth``) and hand a POLICY
    net to ``YOLO(...)`` as if it were detector weights. Guard: on the gate-seeker + yolo path, require
    a spec that looks like detector weights (``--seeker-weights <model.pt>``, or an ``a.pt++b.pt``
    ensemble); raise ``SystemExit`` with a clear message otherwise. No-op for non-gate-seeker runs and
    for the explicit ``red_glow`` / ``none`` opt-ins (which need no weights).

    The EGO path (--ego-ckpt) flies on the SAME case-C perception stack (detector + PnP + temporal
    track) so it is validated identically — its actor .pth lives in --ego-ckpt, never here."""
    if (not (getattr(args, "gate_seeker", False) or getattr(args, "ego_ckpt", None))
            or args.seeker_detector != "yolo"):
        return
    spec = _resolve_seeker_weights(args)
    if not _looks_like_detector_weights(spec):
        raise SystemExit(
            "gate-seeker/ego YOLO needs --seeker-weights <model.pt> (a trained gate detector). "
            f"Got seeker_weights={args.seeker_weights!r}, and the --checkpoint fallback "
            f"({args.checkpoint!r}) is the RL-actor .pth, NOT detector weights -- loading it as "
            "YOLO weights would silently fly a broken detector. Pass --seeker-weights explicitly, "
            "or use --seeker-detector red_glow for the classical (no-GPU) detector."
        )


def _prewarm_detector(args) -> None:
    """PRE-WARM the YOLO gate detector BEFORE arm (the A15 launch-window-freeze fix).

    A15: the FIRST live ``detector.detect`` (tick 0 of the flight loop) cost ~2873 ms -- the
    one-time Ultralytics/CUDA warmup (context init + cuDNN autotune + kernel compile + the model's
    own first-predict predictor setup). That stall froze the onboard camera / recorder over the exact
    launch window we need to OBSERVE, and starved the video thread (~19% frame drops under GPU load),
    so we could not even tell whether the drone lifted off. THE FIX: build the detector + run ONE
    dummy inference on a black frame HERE, at startup -- before GO / arm / the recorder attaching --
    so the expensive first-predict happens off the flight critical path and the loop's tick 0 is fast.
    The warmed instance is stashed on ``args`` and REUSED by ``_build_casec_seeker`` (no double-load).

    Only the ``yolo`` path warms (red_glow is pure OpenCV -> no warmup cost). Any failure (no
    ultralytics / no GPU / bad weights) is logged and swallowed: the real load in _build_casec_seeker
    surfaces a hard error later, and a warmup miss must never abort the run. Idempotent + additive:
    non-yolo / non-gate-seeker runs are byte-identical (nothing is built, nothing stashed).
    The EGO path (--ego-ckpt) shares the case-C perception stack, so it pre-warms identically."""
    if (not (getattr(args, "gate_seeker", False) or getattr(args, "ego_ckpt", None))
            or args.seeker_detector != "yolo"):
        return
    try:
        import numpy as np

        from racer.contracts import Frame
        from racer.vision.detector import GateDetector
        t0 = time.monotonic()
        detector = GateDetector.load(_resolve_seeker_weights(args))
        # A black (360, 640, 3) frame matching the live camera resolution (contracts.Frame): the
        # SHAPE is what drives cuDNN autotune + kernel compile, so warming on the true resolution
        # warms the exact kernels the flight will use. A black frame yields no detections (fine).
        dummy = Frame(frame_id=-1, sim_time_ns=0,
                      image_bgr=np.zeros((360, 640, 3), dtype=np.uint8))
        detector.detect(dummy)   # the expensive first-predict -> now off the flight critical path
        args._prewarmed_detector = detector
        print(f"  [prewarm] YOLO detector warmed in {(time.monotonic() - t0):.2f}s "
              f"(first-predict off the launch window; tick-0 stall eliminated).")
    except Exception as exc:
        print(f"  [prewarm] WARNING: detector pre-warm skipped ({type(exc).__name__}: {exc}); "
              f"the real load happens in _build_casec_seeker (tick-0 may stall as before).",
              file=sys.stderr)


def _build_casec_seeker(args, gates):
    """Construct the case-C Navigator + the slow gate-seeker from the deploy profile (--gate-seeker).

    Returns ``(navigator, seeker, profile)``. The Navigator runs the SELF-LOCALIZING estimator
    (use_ahrs + vision yaw/z + gate-relative chain, NO given position) with an opt-in detector;
    the seeker is the transparent slow pursuit controller. No torch / no RL checkpoint needed."""
    from racer.deploy_profile import get_profile
    from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller
    from racer.navigator import Navigator

    profile = get_profile(args.deploy_profile)
    detector = None
    if args.seeker_detector == "red_glow":
        from racer.vision.red_glow_detector import RedGlowGateDetector
        detector = RedGlowGateDetector()
    elif args.seeker_detector == "yolo":
        from racer.vision.detector import GateDetector
        # Detector weights come from --seeker-weights (SEPARATE from --checkpoint, the RL actor).
        # A single .pt path -> one model; an 'a.pt++b.pt' spec -> EnsembleGateDetector (union+dedup).
        # Falls back to --checkpoint only if --seeker-weights is unset (legacy convenience).
        # REUSE the PRE-WARMED instance (_prewarm_detector, the A15 launch-freeze fix) when present,
        # so the expensive first-predict already ran at startup off the flight critical path; else
        # build it here (byte-identical to the legacy path when no pre-warm ran).
        detector = (getattr(args, "_prewarmed_detector", None)
                    or GateDetector.load(_resolve_seeker_weights(args)))  # weights (artifact-pipe)
    nav = Navigator(gates=gates, detector=detector, config=profile.nav_config)
    # The seeker shares the SAME detector instance: it runs its OWN detect+PnP each tick to recover
    # the SEEN gate's relative bearing (the MAP-FREE visual servo, command_visual) -- it does NOT
    # steer to the absolute map position (the 2026-06-29 blind-launch fix).
    # cruise_speed: the slow cap. settle_s: the post-arm cold-AHRS settle hold (2026-06-29 attempt-2
    # tumble fix) -- hold conservative level + bounded hover thrust + clamped rates while the mag-free
    # AHRS gravity-aligns before any lean. anchor_release_detections is the map-free anchor-release
    # streak (BUG A fix): release the launch-hold on the seeker's OWN consecutive detections, since on
    # the live VQ2 wire the navigator runs map-free and nav.time_since_vision_update_s never goes finite.
    # Thread the profile's GateSeekerConfig overrides (e.g. vq2_case_c -> egress_freeze_attitude=True,
    # the A10 acquisition-trap fix) through the seeker construction seam. None == no overrides ==
    # byte-identical seeker config (VQ1 / case-A). The explicit hardcoded kwargs above are the CLI-
    # driven knobs; there is no key collision with the profile overrides today.
    # Thread the profile's CONTROLLER overrides the EXACT parallel way (vq2_case_c -> kp_att=4.0 +
    # body_rate_slew_max_rps2=8.0, the A11 control-softening fix that de-saturates the egress->pursuit
    # handoff). None == no overrides == today's controller gains (VQ1 / case-A byte-identical).
    seeker = GateSeeker(
        config=GateSeekerConfig(
            cruise_speed=args.seeker_speed,
            settle_s=args.seeker_settle,
            anchor_release_detections=args.seeker_anchor_dets,
            **(profile.seeker_overrides or {}),
        ),
        controller=make_seeker_controller(**(profile.controller_overrides or {})),
        detector=detector,
    )
    return nav, seeker, profile


# ---------------------------------------------------------------------------
# Nav-estimate logging helpers (per-flight nav_estimate.jsonl).
# NOTE: on the VQ2 wire position_ned is DEAD-RECKONED (no GPS/NED truth) so
# the logged position is the KF integrated estimate, not ground truth.
# ---------------------------------------------------------------------------

def _nav_estimate_record(nav_state, nav, s, cmd, gate_index: int, tick_index: int,
                         seeker=None) -> dict:
    """Build one JSONL record from the current tick's navigator output + command.

    All numpy arrays are converted to plain Python lists (.tolist()) so
    ``json.dumps`` can serialise them directly. Fields that are absent or not
    yet available on the nav/nav_state object are logged as null rather than
    raising (a logging bug must NEVER crash a flight).
    """
    rec: dict = {}
    try:
        rec["sim_time_ns"] = int(s.sim_time_ns)
    except Exception:
        rec["sim_time_ns"] = None
    rec["tick_index"] = tick_index
    rec["gate_index"] = gate_index

    # --- estimator attitude ---
    # Try the AHRS true quaternion first (available when use_ahrs is ON), then
    # fall back to the ODOMETRY-convention quat cached on the navigator.  If
    # neither is present log null; the NavState roll/pitch/yaw are always there.
    try:
        ahrs = getattr(nav, "_ahrs", None)
        if ahrs is not None:
            q_true = np.asarray(ahrs.q_wxyz, dtype=np.float64)
            rec["ahrs_quat_wxyz"] = q_true.tolist()
        else:
            rec["ahrs_quat_wxyz"] = None
    except Exception:
        rec["ahrs_quat_wxyz"] = None

    try:
        rec["roll_rad"]  = float(nav_state.roll)
        rec["pitch_rad"] = float(nav_state.pitch)
        rec["yaw_rad"]   = float(nav_state.yaw)
    except Exception:
        rec["roll_rad"] = rec["pitch_rad"] = rec["yaw_rad"] = None

    # --- estimator position (dead-reckoned on VQ2) ---
    try:
        rec["position_ned"] = np.asarray(nav_state.position_ned, dtype=np.float64).tolist()
    except Exception:
        rec["position_ned"] = None

    # --- time since last vision fix ---
    try:
        tsv = float(nav_state.time_since_vision_update_s)
        rec["time_since_vision_s"] = None if (tsv != tsv or tsv == float("inf")) else tsv
    except Exception:
        rec["time_since_vision_s"] = None

    # --- commanded control ---
    try:
        br = cmd.body_rate
        rec["body_rate"] = np.asarray(br, dtype=np.float64).tolist() if br is not None else None
    except Exception:
        rec["body_rate"] = None
    try:
        rec["thrust"] = float(cmd.thrust) if cmd.thrust is not None else None
    except Exception:
        rec["thrust"] = None

    # --- A14 yaw-steer-sign probe (instrumentation only; never feeds control) ---
    # yaw_des_rad: the seeker's PRE-SLEW desired yaw toward the seen gate (atan2(los[1],los[0])),
    # stashed on the seeker each pursuit tick. On a no-pursuit tick the last stashed value (or None)
    # is logged -- we do NOT fabricate a fresh one (matches the "last" semantics of the other fields).
    try:
        yld = getattr(seeker, "_last_yaw_des", None) if seeker is not None else None
        rec["yaw_des_rad"] = float(yld) if yld is not None else None
    except Exception:
        rec["yaw_des_rad"] = None

    # raw_gyro_yaw: the RAW HIGHRES_IMU zgyro (z/yaw axis) BEFORE the gyro_sign correction. Read from
    # the pre-sign stash on the drone state; None when the gyro is unpopulated. INDEPENDENT of gyro_sign.
    try:
        graw = getattr(s, "gyro_body_raw", None)
        rec["raw_gyro_yaw"] = (
            float(np.asarray(graw, dtype=np.float64)[2]) if graw is not None else None
        )
    except Exception:
        rec["raw_gyro_yaw"] = None

    return rec


def _write_nav_estimate_jsonl(records: list, session_dir: "Path | None") -> None:
    """Write the in-memory record list to ``<session_dir>/nav_estimate.jsonl``.

    One JSON object per line, written in a single pass at loop exit.  Errors
    are printed but not re-raised (a write failure must NEVER crash the flight
    result bookkeeping that follows).
    """
    if session_dir is None or not records:
        return
    out = Path(session_dir) / "nav_estimate.jsonl"
    try:
        lines = "\n".join(json.dumps(r) for r in records) + "\n"
        out.write_text(lines, encoding="utf-8")
        print(f"  [nav-log] wrote {len(records)} ticks -> {out}")
    except Exception as exc:
        print(f"  [nav-log] WARNING: failed to write nav_estimate.jsonl: {exc}")


def _fly_gate_seeker(client, args, flight_idx: int,
                     session_dir: Path | None, result: dict) -> dict:
    """SLOW GATE-SEEKER deploy loop on the case-C self-localizing stack (--gate-seeker).

    The transparent alternative to the RL policy: build the case-C Navigator (deploy profile,
    self-localizing) + the slow pursuit gate-seeker, then each tick run
    ``navigator.update(state, frame) -> NavState``, map ``RACE_STATUS.active_gate_index`` to the
    ordered gate, and command ``seeker.command(nav, gate, idx) -> CTBR``. Same safety scaffolding
    as the RL loop (collision / sim-reset / finish / sim-stall guards). The deploy profile's
    ``cmd_rate_scale`` was applied to ``client`` at construction (main()).

    The MAP-FREE visual servo (``GateSeeker.command_visual``) steers from the SEEN gate's relative
    bearing, so STEERING needs no absolute map. The case-C Navigator still consumes a gate map for
    its world-frame SUPPORT fixes (gate-bearing yaw, gate-relative +L), but a STALE / wrong map there
    is dangerous (it injects bad yaw/position fixes), so a self-localizing profile NEVER silently
    falls back to the default ``--map``: it uses the live TRACK_INFO when present, else flies MAP-FREE
    (empty gate list -> the navigator's map-dependent fixes no-op; vision yaw/z + visual servo carry
    the lap). A NON-self-localizing (case-A / VQ1) profile keeps the explicit ``--map`` fallback."""
    from racer.contracts import Frame
    from racer.deploy_profile import get_profile
    from racer.navigator import gates_from_track_records, load_track_map

    profile = get_profile(args.deploy_profile)

    # --- gate map: prefer the live TRACK_INFO; otherwise map handling depends on the profile ---
    # GUARD (2026-06-29 blind-launch fix): the stale-VQ1-map fallback that drove the blind launch
    # U-turn is REMOVED for self-localizing (VQ2 case-C) profiles. The seeker is map-free; the
    # navigator runs map-free (its gate-map fixes simply don't fire without gates).
    if client.track_gates:
        gates = gates_from_track_records(client.track_gates, corner_to_center=True)
        print(f"  [gate-seeker] gate map from live TRACK_INFO ({len(gates)} gates).")
    elif profile.self_localizing:
        gates = []
        print("  [gate-seeker] no live TRACK_INFO + self-localizing profile -> MAP-FREE flight "
              "(visual servo + vision yaw/z; NO absolute map). The stale --map is IGNORED.",
              file=sys.stderr)
    elif args.map and Path(args.map).exists():
        gates = load_track_map(args.map, corner_to_center=True)
        print(f"  [gate-seeker] gate map from --map {args.map} ({len(gates)} gates).")
    else:
        print("  [gate-seeker] no gate map (no TRACK_INFO, no --map) -> cannot fly. abort.",
              file=sys.stderr)
        result["final_state"] = "NO_MAP"
        return result
    n_gates = len(gates)

    nav, seeker, _ = _build_casec_seeker(args, gates)
    print(f"\n[gate-seeker] profile={profile.name} self_localizing={profile.self_localizing} "
          f"cmd_rate_scale={client.cmd_rate_scale:g} detector={args.seeker_detector} "
          f"cruise={args.seeker_speed:g} m/s  gates={n_gates}  max={args.max_seconds:g}s ...")

    tick        = 1.0 / args.rate
    deadline    = time.monotonic() + args.max_seconds
    next_t      = time.monotonic()
    last_sim_t  = int(client.state.sim_time_ns)
    last_adv_w  = time.monotonic()
    last_p      = 0.0
    n_coll0     = result["collisions_at_start"]
    gate_index  = 0
    final_state = "IDLE"

    # --- loop-rate self-report (the 2.3Hz->30Hz confirmation) ---------------------
    # A1-A6 were judged on a loop choked to ~2.3 Hz by the pre-vectorization VP RANSAC
    # (281ms/tick) -> thrust oscillation -> floor slam -> free-fall -> AHRS inversion.
    # We time the WORK per tick (pump->command, excluding the rate-limiter sleep) so a
    # saturated loop is unambiguous: work_dt > tick means we cannot keep up at --rate.
    loop_t0      = time.monotonic()
    n_ticks      = 0
    worst_work_ms = 0.0
    n_over_budget = 0

    reset_counter0 = int(client.state.reset_counter)
    prev_pos = (np.asarray(client.state.position_ned, dtype=np.float64).copy()
                if client.state.position_ned is not None else None)

    # --- nav-estimate log (in-memory buffer; written ONCE at loop exit) -------
    # No per-tick I/O: we collect small dicts here and flush them in a single
    # write after the loop exits.  At <=30 Hz for <=30 s that is <=~900 rows.
    # Guard on session_dir so runs without a recording dir are byte-identical.
    _nav_log: list = []   # populated only when session_dir is not None
    _nav_log_errors: int = 0

    while time.monotonic() < deadline:
        while time.monotonic() < next_t:
            client.pump()
            time.sleep(0.001)
        client.pump()
        next_t = time.monotonic() + tick
        now = time.monotonic()
        s  = client.state
        rs = client.race_status

        # --- stop conditions (mirror the RL loop) ---
        st = int(s.sim_time_ns)
        if st > last_sim_t:
            last_sim_t, last_adv_w = st, now
        elif now - last_adv_w > 1.5:
            print("\n  [gate-seeker] sim_time stalled (race ended) -> stopping.")
            break
        if rs and rs.get("finished"):
            print("\n  [gate-seeker] RACE_STATUS finished -> stopping.")
            final_state = "FINISHED"
            break
        if any(c["threat_level"] >= 2 for c in client.collisions[n_coll0:]):
            print("\n  [gate-seeker] HARD COLLISION -> abort.")
            final_state = "CRASH"
            break

        # --- sim-reset guard (epoch discontinuity -> cut commands) ---
        jump = (float(np.linalg.norm(np.asarray(s.position_ned) - prev_pos))
                if (s.position_ned is not None and prev_pos is not None) else 0.0)
        if int(s.reset_counter) != reset_counter0 or jump > 10.0:
            print("\n  [gate-seeker] SIM RESET DETECTED -> cutting commands.")
            final_state = "SIM_RESET"
            break
        if s.position_ned is not None:
            prev_pos = np.asarray(s.position_ned, dtype=np.float64).copy()

        # --- active gate from RACE_STATUS (authoritative ordering signal) ---
        gi = (int(rs["active_gate_index"])
              if rs and rs.get("active_gate_index") is not None else gate_index)
        if gi > gate_index:
            print(f"\n  [gate-seeker] gate {gate_index} PASSED -> targeting {gi}", flush=True)
        # the wire index drives the navigator's active-gate yaw lock + the advance bookkeeping; the
        # MAP-FREE seeker steers off the SEEN gate, so it does NOT need the gate to exist in any map.
        gate_index = max(gi, 0)
        is_final = n_gates > 0 and gate_index >= n_gates - 1

        # --- perception: read the freshest frame the video thread published (non-blocking) ---
        # The video thread owns the single UDP receiver and stashes the latest reassembled Frame
        # in client._latest_frame; we consume it here, once per new frame_id (the navigator's
        # _maybe_run_vision is itself frame_id-idempotent, so re-feeding the same frame is a no-op).
        frame: Frame | None = getattr(client, "_latest_frame", None)

        # --- estimate (case-C self-localizing) then command the MAP-FREE visual servo ---
        # The seeker chases the gate the CAMERA SEES (command_visual): it runs its own detect+PnP on
        # the live frame, steers to center + fly through the SEEN opening, and HOLDS (no blind slew)
        # until the estimator records its first vision fix (launch anchor) or when no gate is detected.
        # NO absolute map / NO absolute self-position drives steering (the 2026-06-29 blind-launch fix).
        nav_state = nav.update(s, frame)
        cmd = seeker.command_visual(nav_state, frame, gate_index, is_final_gate=is_final)
        client.send_command(cmd)

        # --- nav-estimate log: append one dict to the in-memory buffer (no I/O) ---
        if session_dir is not None:
            try:
                _nav_log.append(_nav_estimate_record(
                    nav_state, nav, s, cmd, gate_index, n_ticks, seeker=seeker))
            except Exception:
                _nav_log_errors += 1

        # work time = everything from the post-wait `now` through command send (no sleep)
        work_ms = (time.monotonic() - now) * 1e3
        n_ticks += 1
        if work_ms > worst_work_ms:
            worst_work_ms = work_ms
        if work_ms > tick * 1e3:
            n_over_budget += 1

        if now - last_p >= 1.0:
            p = nav_state.position_ned
            br = cmd.body_rate if cmd.body_rate is not None else np.zeros(3)
            print(f"  t={s.sim_time_ns/1e9:7.2f}s gi={gate_index} "
                  f"pos=({p[0]:+6.1f},{p[1]:+6.1f},{p[2]:+6.1f}) "
                  f"thr={cmd.thrust:.3f} rate=[{br[0]:+.2f},{br[1]:+.2f},{br[2]:+.2f}] "
                  f"tsv={nav_state.time_since_vision_update_s:.2f}s   ",
                  end="\r", flush=True)
            last_p = now

    if final_state == "IDLE":
        final_state = "TIMEOUT" if time.monotonic() >= deadline else "STALLED"
        print(f"\n  ({final_state.lower()})")

    # --- nav-estimate log: single write after ALL exit paths -------------------
    # Written here (not inside each break) so every break/timeout/crash path is
    # covered in one place.  _write_nav_estimate_jsonl swallows errors internally.
    _write_nav_estimate_jsonl(_nav_log, session_dir)
    if _nav_log_errors:
        print(f"  [nav-log] WARNING: {_nav_log_errors} per-tick record errors (logging bug, not flight bug)")

    # --- loop-rate verdict: did we actually realize the 30 Hz loop? ---------------
    elapsed = max(time.monotonic() - loop_t0, 1e-6)
    achieved_hz = n_ticks / elapsed
    over_pct = 100.0 * n_over_budget / max(n_ticks, 1)
    rate_ok = achieved_hz >= 0.9 * args.rate and over_pct < 5.0
    print(f"  [loop-rate] {achieved_hz:5.1f} Hz over {n_ticks} ticks "
          f"(target {args.rate:g}); worst work {worst_work_ms:.0f} ms; "
          f"{over_pct:.1f}% ticks over budget -> {'OK' if rate_ok else 'CHOKED'}")
    result["achieved_hz"]       = round(achieved_hz, 2)
    result["worst_work_ms"]     = round(worst_work_ms, 1)
    result["loop_over_budget_pct"] = round(over_pct, 1)

    # --- vision-timing: per-STEP breakdown of the per-frame vision pipeline (logging only) --------
    # Pins WHICH _maybe_run_vision sub-step (detect / vp_yaw [VP RANSAC + Manhattan lines] /
    # floor_height / pnp) is driving worst_work_ms above, so the next tuning pass targets the right
    # step instead of guessing. Reads the Navigator's in-memory vision_step_ms dict (same
    # accumulate-in-memory / print-once-at-exit pattern as [seeker-diag]); swallowed if absent/empty
    # so an older Navigator (or a run with no vision) doesn't break this print.
    step_ms = getattr(nav, "vision_step_ms", None)
    if isinstance(step_ms, dict) and any(v.get("count", 0) for v in step_ms.values()):
        parts = []
        for name, v in step_ms.items():
            n = int(v.get("count", 0))
            if n == 0:
                continue
            mean_ms = v.get("total_ms", 0.0) / n
            parts.append(f"{name}: mean={mean_ms:.1f}ms max={v.get('max_ms', 0.0):.1f}ms n={n}")
        worst = getattr(nav, "_vision_worst_tick_ms", None) or {}
        worst_total = sum(worst.values())
        worst_str = ", ".join(f"{k}={v:.1f}ms" for k, v in sorted(worst.items(),
                                                                    key=lambda kv: -kv[1]))
        print(f"  [vision-timing] " + "  ".join(parts))
        print(f"  [vision-timing] worst tick sum={worst_total:.1f}ms breakdown: {worst_str}")
        result["vision_step_ms"] = {k: dict(v) for k, v in step_ms.items()}
        result["vision_worst_tick_ms"] = dict(worst)

    # --- A13 seeker diagnostics: pose-None breakdown + hold-last-demand bridge coverage -----------
    # Logging only (no behaviour change): the seeker tallies per-tick command regimes in memory
    # (no per-tick I/O, mirrors the buffered nav_estimate.jsonl pattern); we emit a single summary
    # line here at loop exit + stash the counts in result. This tells the next fly WHETHER/WHICH gate
    # is the dominant pose-gap source (track-continuity vs valid-poses-empty) so A14 can relax it with
    # data, and whether the hold-last-demand bridge actually made the per-tick command continuous.
    dc = getattr(seeker, "diag_counts", None)
    if isinstance(dc, dict) and dc:
        none_tot = max(int(dc.get("none_total", 0)), 0)
        cmd_tot = int(dc.get("pursuit", 0)) + none_tot
        bridged_pct = 100.0 * int(dc.get("bridged", 0)) / max(none_tot, 1)
        print(f"  [seeker-diag] cmds={cmd_tot} pursuit={dc.get('pursuit', 0)} "
              f"none={none_tot} (valid_empty={dc.get('none_valid_poses_empty', 0)} "
              f"continuity={dc.get('none_continuity_reject', 0)} "
              f"first_acq={dc.get('none_first_acq_reject', 0)} other={dc.get('none_other', 0)}) "
              f"-> bridged={dc.get('bridged', 0)} ({bridged_pct:.0f}% of none) "
              f"held_legacy={dc.get('held_legacy', 0)}")
        result["seeker_diag"] = dict(dc)

    result["final_state"] = final_state
    result["gate_index"]  = gate_index
    result["collisions"]  = len(client.collisions) - n_coll0
    return result


def _fly_ego(client, actor, args, flight_idx: int,
             session_dir: Path | None, result: dict) -> dict:
    """EGO (21-dim egocentric) RL deploy loop on the case-C self-localizing stack (--ego-ckpt).

    A HYBRID of the two existing loops [ego-deploy 2026-07-09]:
      * the gate-seeker's SCAFFOLDING: case-C Navigator (deploy profile, AHRS + vision yaw/z,
        map-free), the shared pre-warmed detector + the seeker's temporal-tracked
        ``detect_gate_lever`` (used ONLY as the perception source -- the seeker's controller is
        never called), non-blocking ``client._latest_frame``, the epoch/reset/collision guards,
        and IMU/AHRS-liveness gating (NOT the RL loop's ODOMETRY-staleness gate -- ODOMETRY is
        blocked on the VQ2 wire, so telemetry_health would read no_fix forever);
      * the RL loop's ACTION pipeline: ``policy_step`` (tanh -> rescale -> virtual flip ->
        FLU->FRD -> hover-scale) reused UNCHANGED, with the obs[8] normed-thrust feedback.

    The 21-dim obs itself is built by ``racer.ego_obs.EgoObsBuilder`` (see its module docstring
    for the full frame/masking contract). Per-tick products are buffered in memory and written
    ONCE at loop exit (<session>/ego_obs.jsonl) -- no per-tick blocking I/O in the control loop.
    ADDITIVE + OPT-IN: nothing on the default RL / gate-seeker paths changes."""
    from scipy.spatial.transform import Rotation as _Rot

    from racer.contracts import Frame
    from racer.deploy_profile import get_profile
    from racer.ego_obs import EgoObsBuilder, EgoObsBuilderConfig
    from racer.navigator import gates_from_track_records, load_track_map

    profile = get_profile(args.deploy_profile)

    # --- gate map: mirrors _fly_gate_seeker (live TRACK_INFO > map-free for self-localizing) ---
    if client.track_gates:
        gates = gates_from_track_records(client.track_gates, corner_to_center=True)
        print(f"  [ego] gate map from live TRACK_INFO ({len(gates)} gates).")
    elif profile.self_localizing:
        gates = []
        print("  [ego] no live TRACK_INFO + self-localizing profile -> MAP-FREE flight "
              "(vision yaw/z + the tracked gate lever; NO absolute map).", file=sys.stderr)
    elif args.map and Path(args.map).exists():
        gates = load_track_map(args.map, corner_to_center=True)
        print(f"  [ego] gate map from --map {args.map} ({len(gates)} gates).")
    else:
        print("  [ego] no gate map (no TRACK_INFO, no --map) -> cannot fly. abort.",
              file=sys.stderr)
        result["final_state"] = "NO_MAP"
        return result

    # The seeker is built ONLY for its perception half (detect_gate_lever: detector + PnP +
    # quality gates + the temporal track); its pursuit controller is never invoked.
    nav, seeker, _ = _build_casec_seeker(args, gates)

    builder = EgoObsBuilder(EgoObsBuilderConfig(
        stale_horizon_s=args.ego_stale_horizon,
        det_hold_s=args.ego_det_hold,
        obs_coast=args.ego_obs_coast,
        virtual_flip=args.virtual_flip,
        slot1_enabled=False,               # --ego-slot1 is rejected at startup (stub)
        sector_mode=args.ego_sector_mode,
    ))

    print(f"\n[ego] ckpt={args.ego_ckpt}  profile={profile.name} "
          f"self_localizing={profile.self_localizing}")
    print(f"[ego] act bounds thrust [{_ACT_MIN[0]:g},{_ACT_MAX[0]:g}] rates +-{_ACT_MAX[1]:g} "
          f"| cmd_rate_scale={client.cmd_rate_scale:g} (ego FORCES wire scale via "
          f"--ego-rate-scale; the seeker profile's 0.4 is NOT applied -- the RL plant was "
          f"sysid'd at wire scale 1.0)")
    print(f"[ego] det_hold={args.ego_det_hold:g}s stale_horizon={args.ego_stale_horizon:g}s "
          f"obs_coast={args.ego_obs_coast} sector_mode={args.ego_sector_mode} "
          f"virtual_flip={args.virtual_flip} rate={args.rate:g}Hz max={args.max_seconds:g}s ...")

    tick        = 1.0 / args.rate
    deadline    = time.monotonic() + args.max_seconds
    next_t      = time.monotonic()
    last_sim_t  = int(client.state.sim_time_ns)
    last_adv_w  = time.monotonic()
    last_p      = 0.0
    last_normed = 0.0          # training: last_action zeroed at reset -> obs[8]=0
    n_coll0     = result["collisions_at_start"]
    gate_index  = 0
    final_state = "IDLE"

    loop_t0       = time.monotonic()
    n_ticks       = 0
    worst_work_ms = 0.0
    n_over_budget = 0
    n_pose_ticks  = 0          # ticks with a fresh accepted gate lever
    n_masked      = 0          # ticks flown with slot0 masked (the blackout regime)

    reset_counter0 = int(client.state.reset_counter)
    race_start0    = (int(client.race_status["race_start_boot_time_ms"])
                      if client.race_status else None)
    prev_pos = (np.asarray(client.state.position_ned, dtype=np.float64).copy()
                if client.state.position_ned is not None else None)
    last_lever_fid: int | None = None   # feed each frame_id to the lever ONCE (fresh-fix gating)

    _ego_log: list = []        # in-memory; single write at exit (no per-tick I/O)
    _ego_log_errors = 0

    while time.monotonic() < deadline:
        while time.monotonic() < next_t:
            client.pump()
            time.sleep(0.001)
        client.pump()
        next_t = time.monotonic() + tick
        now = time.monotonic()
        s  = client.state
        rs = client.race_status

        # --- stop conditions (mirror the RL + seeker loops) ---
        st = int(s.sim_time_ns)
        if st > last_sim_t:
            last_sim_t, last_adv_w = st, now
        elif now - last_adv_w > 1.5:
            print("\n  [ego] sim_time stalled (race ended) -> stopping.")
            break
        if rs and rs.get("finished"):
            print("\n  [ego] RACE_STATUS finished -> stopping.")
            final_state = "FINISHED"
            break
        if any(c["threat_level"] >= 2 for c in client.collisions[n_coll0:]):
            print("\n  [ego] HARD COLLISION -> abort.")
            final_state = "CRASH"
            break

        # --- sim-reset guard: epoch discontinuity -> CUT commands NOW (RL-loop parity) ---
        gi_now = (int(rs["active_gate_index"])
                  if rs and rs.get("active_gate_index") is not None else None)
        jump = (float(np.linalg.norm(np.asarray(s.position_ned) - prev_pos))
                if (s.position_ned is not None and prev_pos is not None) else 0.0)
        reset_why = None
        if int(s.reset_counter) != reset_counter0:
            reset_why = f"reset_counter {reset_counter0}->{s.reset_counter}"
        elif (rs and race_start0 is not None
                and int(rs["race_start_boot_time_ms"]) != race_start0):
            reset_why = "RACE_STATUS race_start changed (new race)"
        elif gi_now is not None and gi_now < gate_index:
            reset_why = f"active_gate_index dropped {gate_index}->{gi_now}"
        elif jump > 10.0:
            reset_why = f"position teleport ({jump:.1f} m in one tick)"
        if reset_why is not None:
            print(f"\n  [ego] SIM RESET DETECTED ({reset_why}) -> cutting commands.")
            final_state = "SIM_RESET"
            break
        if s.position_ned is not None:
            prev_pos = np.asarray(s.position_ned, dtype=np.float64).copy()

        # --- active gate from RACE_STATUS; on an advance drop the perception track + slot state
        # (the new slot0 starts cold/masked until first acquisition -- window-promotion analog) ---
        if gi_now is not None and gi_now > gate_index:
            print(f"\n  [ego] gate {gate_index} PASSED -> targeting {gi_now}", flush=True)
            gate_index = gi_now
            seeker.reset()            # drop the temporal track -> re-acquire the NEW gate
            last_lever_fid = None
        elif gi_now is not None:
            gate_index = max(gi_now, 0)

        # --- perception + estimation (case-C, all self-localized) ---
        frame: Frame | None = getattr(client, "_latest_frame", None)
        nav_state = nav.update(s, frame)
        ods = nav.obs_drone_state(s)

        # --- IMU/AHRS-liveness gate (NOT the RL loop's odo-staleness gate: ODOMETRY is blocked
        # on this wire, so odo_recv_ns never fires; the AHRS is alive once HIGHRES_IMU flows) ---
        if ods.orientation_ned_wxyz is None:
            continue                   # pre-first-IMU warmup: no attitude -> no command
        q_raw = np.asarray(ods.orientation_ned_wxyz, dtype=np.float64)
        if not (np.all(np.isfinite(q_raw)) and float(q_raw @ q_raw) > 1e-12):
            continue                   # degenerate quat -> skip the tick (never NaN the wire)

        # TRUE physical attitude: the AHRS cache is re-encoded in the ODOMETRY-wire convention,
        # so RE-apply the same conjugation build_obs uses (fly_rl frame contract).
        q_true = q_raw * _ODO_QUAT_TRUE_CONJ
        R_frd2ned = _Rot.from_quat([q_true[1], q_true[2], q_true[3], q_true[0]]).as_matrix()

        # --- gate lever: feed each frame_id ONCE (a repeated pose is NOT a fresh fix; the
        # builder ego-propagates through the gap between real detections) ---
        pose = None
        if frame is not None and frame.frame_id != last_lever_fid:
            pose = seeker.detect_gate_lever(frame)   # detect_cached: shared + frame_id-idempotent
            last_lever_fid = frame.frame_id
        if pose is not None:
            n_pose_ticks += 1

        # --- 21-dim obs + policy + command ---
        obs = builder.update(
            sim_time_ns=st, gate_index=gate_index, R_frd2ned=R_frd2ned,
            vel_ned=nav_state.velocity_ned, gyro_frd=s.gyro_body,
            pose=pose, last_normed_thrust=last_normed)
        if not builder.last_diag.get("det_proxy", False):
            n_masked += 1
        rate_frd, collective, last_normed = policy_step(
            actor, obs, args.max_rate, virtual_flip=args.virtual_flip,
            yaw_scale=args.yaw_scale)
        client.send_command(ControlCommand(
            mode=ControlMode.BODY_RATE,
            sim_time_ns=st,
            body_rate=rate_frd,
            thrust=collective,
        ))

        # --- in-memory forensics record (single write at exit) ---
        if session_dir is not None:
            try:
                d = builder.last_diag
                _ego_log.append({
                    "k": n_ticks, "sim_time_ns": st, "gate_index": gate_index,
                    "obs": np.asarray(obs, dtype=np.float64).round(5).tolist(),
                    "pose_seen": d.get("pose_seen"), "age_s": d.get("age_s"),
                    "conf": round(float(d.get("conf", 0.0)), 4),
                    "area": round(float(d.get("area", 0.0)), 4),
                    "sector": d.get("sector"),
                    "rate_frd": rate_frd.round(4).tolist(),
                    "collective": round(collective, 5),
                    "normed_thrust": round(last_normed, 5),
                    "kf_pos_ned": np.asarray(nav_state.position_ned).round(3).tolist(),
                    "tsv": (None if not np.isfinite(nav_state.time_since_vision_update_s)
                            else round(nav_state.time_since_vision_update_s, 3)),
                })
            except Exception:
                _ego_log_errors += 1

        work_ms = (time.monotonic() - now) * 1e3
        n_ticks += 1
        if work_ms > worst_work_ms:
            worst_work_ms = work_ms
        if work_ms > tick * 1e3:
            n_over_budget += 1

        if now - last_p >= 1.0:
            d = builder.last_diag
            print(f"  t={st/1e9:7.2f}s gi={gate_index} "
                  f"conf={d.get('conf', 0.0):.2f} area={d.get('area', 0.0):.2f} "
                  f"thr={collective:.3f} rate=[{rate_frd[0]:+.2f},{rate_frd[1]:+.2f},"
                  f"{rate_frd[2]:+.2f}]   ", end="\r", flush=True)
            last_p = now

    if final_state == "IDLE":
        final_state = "TIMEOUT" if time.monotonic() >= deadline else "STALLED"
        print(f"\n  ({final_state.lower()})")

    # --- ego forensics log: single write covering every exit path ---
    if session_dir is not None and _ego_log:
        out = Path(session_dir) / "ego_obs.jsonl"
        try:
            out.write_text("\n".join(json.dumps(r) for r in _ego_log) + "\n", encoding="utf-8")
            print(f"  [ego-log] wrote {len(_ego_log)} ticks -> {out}")
        except Exception as exc:
            print(f"  [ego-log] WARNING: failed to write ego_obs.jsonl: {exc}")
    if _ego_log_errors:
        print(f"  [ego-log] WARNING: {_ego_log_errors} per-tick record errors "
              f"(logging bug, not flight bug)")

    # --- loop-rate + perception-supply verdicts (A19 health-check parity) ---
    elapsed = max(time.monotonic() - loop_t0, 1e-6)
    achieved_hz = n_ticks / elapsed
    over_pct = 100.0 * n_over_budget / max(n_ticks, 1)
    rate_ok = achieved_hz >= 0.9 * args.rate and over_pct < 5.0
    print(f"  [loop-rate] {achieved_hz:5.1f} Hz over {n_ticks} ticks (target {args.rate:g}); "
          f"worst work {worst_work_ms:.0f} ms; {over_pct:.1f}% ticks over budget "
          f"-> {'OK' if rate_ok else 'CHOKED'}")
    print(f"  [ego-diag] fresh gate levers={n_pose_ticks}  masked-slot0 ticks={n_masked}/{n_ticks} "
          f"({100.0 * n_masked / max(n_ticks, 1):.0f}% blackout duty)")
    dc = getattr(seeker, "diag_counts", None)
    if isinstance(dc, dict) and dc:
        result["seeker_diag"] = dict(dc)
    step_ms = getattr(nav, "vision_step_ms", None)
    if isinstance(step_ms, dict) and any(v.get("count", 0) for v in step_ms.values()):
        result["vision_step_ms"] = {k: dict(v) for k, v in step_ms.items()}
    result["achieved_hz"] = round(achieved_hz, 2)
    result["worst_work_ms"] = round(worst_work_ms, 1)
    result["loop_over_budget_pct"] = round(over_pct, 1)
    result["ego_fresh_levers"] = n_pose_ticks
    result["ego_masked_ticks"] = n_masked
    result["final_state"] = final_state
    result["gate_index"] = gate_index
    result["collisions"] = len(client.collisions) - n_coll0
    return result


def _fly_armed(client, actor, args, flight_idx: int,
               session_dir: Path | None, result: dict) -> dict:
    """The armed-flight body of ``fly_once`` (PATH B bridge -> RL control loop ->
    finish hold), factored out so ``fly_once`` can wrap it in a try/finally that
    force-disarms on EVERY exit path (F-B). Mutates + returns ``result``; the caller
    owns arming and the guaranteed disarm, so this function NEVER disarms itself."""
    # ------------------------------------------------------------------
    # OPT-IN: SLOW GATE-SEEKER on the case-C self-localizing stack (VQ2 "slow is
    # smooth" first lap). --gate-seeker swaps the RL policy for the transparent
    # pursuit controller (gate_seeker.py) flying on the case-C Navigator (deploy
    # profile). DEFAULT OFF: the RL policy path below is untouched. [VQ2 slow-lap]
    # ------------------------------------------------------------------
    if getattr(args, "gate_seeker", False):
        return _fly_gate_seeker(client, args, flight_idx, session_dir, result)

    # ------------------------------------------------------------------
    # OPT-IN: EGO (21-dim egocentric) RL policy on the case-C self-localizing
    # stack. --ego-ckpt swaps the inc7 given-pose RL loop for _fly_ego (the
    # gate-seeker scaffolding + the RL action pipeline). DEFAULT OFF: without
    # --ego-ckpt every existing path below is untouched. [ego-deploy 2026-07-09]
    # ------------------------------------------------------------------
    if getattr(args, "ego_ckpt", None):
        return _fly_ego(client, actor, args, flight_idx, session_dir, result)

    # ------------------------------------------------------------------
    # PATH B bridge: CTBR launcher to the handoff seam
    # ------------------------------------------------------------------
    if args.bridge:
        bres = run_bridge(client, args, gate0_x_ned=float(_GATE_POS_ZUP[0, 0]))
        if bres != "HANDOFF":
            result["final_state"] = bres
            return result   # fly_once's finally disarms

    # ------------------------------------------------------------------
    # P4-C05 deploy guard: the RL loop below builds obs through the hardcoded yaw=π VQ1 gate
    # frame (build_obs gate_map=None). Verify the sim is broadcasting the VQ1 course before
    # trusting it — a non-π / different course would silently corrupt the obs by metres.
    # ------------------------------------------------------------------
    _assert_live_course_is_vq1(client.track_gates)

    # ------------------------------------------------------------------
    # RL control loop at --rate Hz (training cadence)
    # ------------------------------------------------------------------
    print(f"\n[fly] RL policy  rate={args.rate:g} Hz  max_rate="
          f"{args.max_rate if args.max_rate > 0 else 'off'}  "
          f"yaw_scale={args.yaw_scale:g}  "
          f"virtual_flip={args.virtual_flip}  max={args.max_seconds:g}s ...")
    final_state  = "IDLE"
    tick         = 1.0 / args.rate
    deadline     = time.monotonic() + args.max_seconds
    next_t       = time.monotonic()
    last_sim_t   = int(client.state.sim_time_ns)
    last_adv_w   = time.monotonic()
    last_p       = 0.0
    last_normed  = 0.0        # training: last_action zeroed at reset -> obs[12]=0
    n_coll0      = result["collisions_at_start"]
    gate_index   = 0

    # --- S17 live-reset guard baselines: the sim AUTORESETS the race on sustained
    # gate contact; commanding through one latches throttle into the fresh race
    # (2026-06-11: minutes of uncontrolled spinning). Detect the epoch break and
    # CUT commands immediately.
    reset_counter0 = int(client.state.reset_counter)
    race_start0    = (int(client.race_status["race_start_boot_time_ms"])
                      if client.race_status else None)
    prev_pos       = (np.asarray(client.state.position_ned, dtype=np.float64).copy()
                      if client.state.position_ned is not None else None)
    spin_t0: float | None = None   # S17 spin guard: onset of sustained high |rate|
    spin_gate      = gate_index
    bad_t0: float | None = None    # F-C: onset of a stale/non-finite ODOMETRY recovery window

    # --- --debug-obs per-step dump (S17): everything the policy saw and emitted.
    dbg_f = None
    dbg_k = 0
    if args.debug_obs and session_dir is not None:
        dbg_f = open(session_dir / "debug_obs.jsonl", "w", encoding="utf-8")
        dbg_f.write(json.dumps({
            "type": "header", "flight": flight_idx, "checkpoint": str(args.checkpoint),
            "act_min": _ACT_MIN.tolist(), "act_max": _ACT_MAX.tolist(),
            "hover_thrust": _HOVER_THRUST, "rate_hz": args.rate,
            "max_rate": args.max_rate, "yaw_scale": args.yaw_scale,
            "virtual_flip": args.virtual_flip,
            "obs_labels": OBS_LABELS,
        }) + "\n")

    try:
        while time.monotonic() < deadline:
            # pace to target rate while draining telemetry
            while time.monotonic() < next_t:
                client.pump()
                time.sleep(0.001)
            client.pump()
            next_t = time.monotonic() + tick

            s  = client.state
            rs = client.race_status
            now = time.monotonic()

            # --- stop conditions ---
            st = int(s.sim_time_ns)
            if st > last_sim_t:
                last_sim_t = st
                last_adv_w = now
            elif now - last_adv_w > 1.5:
                print("\n  sim_time stalled (race ended) -> stopping.")
                break

            if rs and rs["finished"]:
                print("\n  RACE_STATUS finished -> stopping.")
                final_state = "FINISHED"
                break

            if any(c["threat_level"] >= 2 for c in client.collisions[n_coll0:]):
                print("\n  HARD COLLISION -> abort.")
                final_state = "CRASH"
                break

            # --- S17 live-reset guard: epoch discontinuity -> CUT commands NOW ---
            gi_now = (int(rs["active_gate_index"])
                      if rs and rs.get("active_gate_index") is not None else None)
            jump = (float(np.linalg.norm(np.asarray(s.position_ned) - prev_pos))
                    if (s.position_ned is not None and prev_pos is not None) else 0.0)
            reset_why = None
            if int(s.reset_counter) != reset_counter0:
                reset_why = f"ODOMETRY reset_counter {reset_counter0}->{s.reset_counter}"
            elif (rs and race_start0 is not None
                    and int(rs["race_start_boot_time_ms"]) != race_start0):
                reset_why = "RACE_STATUS race_start changed (new race)"
            elif gi_now is not None and gi_now < gate_index:
                reset_why = f"active_gate_index dropped {gate_index}->{gi_now}"
            elif jump > 10.0:   # respawns teleport 20+ m; max real movement ~1 m/tick
                reset_why = f"position teleport ({jump:.1f} m in one tick)"
            if reset_why is not None:
                print(f"\n  SIM RESET DETECTED ({reset_why}) -> cutting RL commands.")
                final_state = "SIM_RESET"
                break
            if s.position_ned is not None:
                prev_pos = np.asarray(s.position_ned, dtype=np.float64).copy()

            # --- gate tracking via RACE_STATUS.active_gate_index ---
            if gi_now is not None:
                if gi_now > gate_index:
                    print(f"\n  gate {gate_index} PASSED -> targeting {gi_now}", flush=True)
                gate_index = min(gi_now, N_GATES - 1)

            # --- F-C: ODOMETRY freshness + finite gate (audit D1/D2/R4/R5) ---------------
            # MUST sit ABOVE the spin guard + build_obs. ODOMETRY is the SOLE source of the
            # attitude quat + body rate; the shared recv_monotonic_ns is also bumped by LPN/IMU,
            # so the old None-guard alone is blind to a selective ODOMETRY drop (the quat/rate
            # freeze verbatim, non-None). Flying on a frozen attitude is open-loop divergence
            # [D1]; a frozen high |w| would false-fire the spin guard [D2]; a NaN/inf/zero-norm
            # field would raise inside build_obs or push NaN to the wire [R4/R5]. We gate command
            # emission on the per-field odo_recv_ns age + finiteness (telemetry_health), holding a
            # SAFE HOVER for a bounded recovery window then aborting cleanly (mirrors fly_vq1.py).
            health, odo_age_s = telemetry_health(s, time.monotonic_ns(), args.odo_stale_s)
            if health == "no_fix":
                continue   # ODOMETRY not arrived yet (pre-first-fix warmup; do not arm timer)
            if health != "ok":
                why = ("non-finite telemetry" if health == "non_finite"
                       else f"ODOMETRY stale {odo_age_s * 1e3:.0f} ms")
                if bad_t0 is None:
                    bad_t0 = now
                    print(f"\n  [odo-guard] {why} -> SAFE HOVER "
                          f"(recovery <= {args.odo_recovery_s:g}s) ...")
                # Neutral hold: stop rotating + hover collective. NEVER re-latch the stale RL
                # command, and never feed a NaN/zero-norm quat into build_obs.
                client.send_command(ControlCommand(
                    mode=ControlMode.BODY_RATE,
                    sim_time_ns=int(s.sim_time_ns),
                    body_rate=np.zeros(3),
                    thrust=_HOVER_THRUST,
                ))
                if now - bad_t0 > args.odo_recovery_s:
                    print(f"\n  ODOMETRY did not recover within {args.odo_recovery_s:g}s "
                          f"({why}) -> abort.")
                    final_state = "ODO_STALE" if health == "stale" else "NON_FINITE"
                    break
                continue
            bad_t0 = None   # fresh + finite ODOMETRY -> clear the recovery timer

            # --- S17 spin guard: sustained high body rate with zero gate progress ---
            # (Reached only on a FRESH, finite ODOMETRY tick -> the rate is never frozen here,
            #  closing the D2 false-SPIN_ABORT on a stale |w|.)
            w_mag = float(np.linalg.norm(np.asarray(s.angular_rate_body, dtype=np.float64)))
            if w_mag < args.spin_rate_abort or gate_index != spin_gate:
                spin_t0, spin_gate = None, gate_index
            elif spin_t0 is None:
                spin_t0 = now
            elif now - spin_t0 > args.spin_time_abort:
                print(f"\n  UNRECOVERED SPIN (|w|={w_mag:.1f} rad/s > "
                      f"{args.spin_rate_abort:g} for {args.spin_time_abort:g}s, "
                      f"no gate progress) -> abort.")
                final_state = "SPIN_ABORT"
                break

            # --- build obs & run policy (telemetry already None/finite/freshness-gated) ---
            obs = build_obs(s, gate_index, last_normed, virtual_flip=args.virtual_flip)
            dbg: dict | None = {} if dbg_f is not None else None
            rate_frd, collective, last_normed = policy_step(
                actor, obs, args.max_rate, virtual_flip=args.virtual_flip, debug=dbg,
                yaw_scale=args.yaw_scale)

            client.send_command(ControlCommand(
                mode=ControlMode.BODY_RATE,
                sim_time_ns=int(s.sim_time_ns),
                body_rate=rate_frd,
                thrust=collective,
            ))

            if dbg_f is not None:
                dbg.update({
                    "k": dbg_k, "t_mono": now, "sim_time_ns": st,
                    "gate_index": gate_index,
                    "odo_age_ms": round((time.monotonic_ns() - int(s.odo_recv_ns)) / 1e6, 1)
                    if int(s.odo_recv_ns) > 0 else None,
                    "pos_ned": np.asarray(s.position_ned).round(4).tolist(),
                    "vel_ned": np.asarray(s.velocity_ned).round(4).tolist(),
                    "q_raw_wxyz": np.asarray(s.orientation_ned_wxyz).round(6).tolist(),
                    "w_raw": np.asarray(s.angular_rate_body).round(4).tolist(),
                    "reset_counter": int(s.reset_counter),
                    "n_coll": len(client.collisions) - n_coll0,
                    "obs": np.asarray(obs, dtype=np.float64).round(5).tolist(),
                    "rate_frd": rate_frd.round(4).tolist(),
                    "collective": round(collective, 5),
                    "normed_thrust": round(last_normed, 5),
                })
                dbg_f.write(json.dumps(dbg) + "\n")
                dbg_k += 1
                if dbg_k % 30 == 0:
                    dbg_f.flush()

            if now - last_p >= 1.0 and s.position_ned is not None:
                p = s.position_ned
                print(f"  t={s.sim_time_ns/1e9:7.2f}s  gi={gate_index}  "
                      f"pos=({p[0]:+6.1f},{p[1]:+6.1f},{p[2]:+6.1f})  "
                      f"thr={collective:.3f}  "
                      f"rate=[{rate_frd[0]:+.2f},{rate_frd[1]:+.2f},{rate_frd[2]:+.2f}]   ",
                      end="\r", flush=True)
                last_p = now
    finally:
        if dbg_f is not None:
            dbg_f.close()

    if final_state == "IDLE":
        final_state = "TIMEOUT" if time.monotonic() >= deadline else "STALLED"
        print(f"\n  ({final_state.lower()})")

    # ------------------------------------------------------------------
    # Finish hold (same drain logic as fly_vq1.py)
    # ------------------------------------------------------------------
    finished_run = (final_state == "FINISHED"
                    or sim_finish_confirmed(client.race_status, N_GATES))
    if finished_run and args.finish_hold_s > 0:
        print(f"\n[finish] gate {gate_index}/{N_GATES} cleared — "
              f"holding up to {args.finish_hold_s:g}s for terminal RACE_STATUS ...")
        hold_next_t = time.monotonic()

        def _hold_tick():
            nonlocal hold_next_t
            while time.monotonic() < hold_next_t:
                client.pump()
                time.sleep(0.001)
            client.pump()
            hold_next_t = time.monotonic() + tick
            client.send_command(ControlCommand(
                mode=ControlMode.BODY_RATE,
                sim_time_ns=int(client.state.sim_time_ns),
                body_rate=np.zeros(3),
                thrust=_HOVER_THRUST,
            ))

        try:
            res = drain_until_finish(
                _hold_tick, lambda: client.race_status,
                n_gates=N_GATES, max_hold_s=args.finish_hold_s,
                post_confirm_hold_s=min(0.4, args.finish_hold_s),
            )
            rs2 = client.race_status or {}
            ft  = rs2.get("race_finish_time_ns", -1)
            if res["confirmed"]:
                tstr = (f", time={ft/1e9:.2f}s"
                        if ft is not None and ft >= 0 else "")
                print(f"  terminal RACE_STATUS at +{res['confirmed_at_s']:.2f}s"
                      f" (finished={rs2.get('finished')}{tstr}).")
            else:
                print(f"  !! no terminal RACE_STATUS within {args.finish_hold_s:g}s "
                      "— race_outcome may under-certify.")
        except Exception as exc:
            print(f"  finish-hold error: {exc}", file=sys.stderr)

    # NB: disarm is owned by fly_once's finally (F-B) -- do NOT disarm here.
    result["final_state"] = final_state
    result["gate_index"]  = gate_index
    result["collisions"]  = len(client.collisions) - n_coll0
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Factored out of main() so the (autonomy-critical) defaults are
    unit-testable without connecting a socket (F-A verify)."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default=str(Path(__file__).resolve().parent / "checkpoints"
                                / "stage1_inc7_actor.pth"),
                    help="path to actor .pth (its .json sidecar, if present, sets the "
                         "trained action bounds). DEFAULT = inc7 (LIVE-CONFIRMED current best; "
                         "F-A. The retired inc4 default was a footgun -- it ships an aero-blind "
                         "actor unless inc7 is passed explicitly).")
    ap.add_argument("--endpoint",     default="udp:127.0.0.1:14550")
    ap.add_argument("--video-port",   type=int, default=VIDEO_PORT)
    ap.add_argument("--label",        default="rl_s1")
    ap.add_argument("--rate",         type=float, default=30.0,
                    help="control loop Hz; default 30 = the TRAINING dt 0.0333 "
                         "(racing.yaml). The policy was trained on 33 ms action holds.")
    ap.add_argument("--max-rate",     type=float, default=0.0,
                    help="cap |body-rate| (rad/s, per-axis, FLU) before sending; "
                         "0 = off. OOD-start diagnostic (PATH A).")
    ap.add_argument("--cmd-rate-scale", type=float, default=1.0,
                    help="command->realized body-rate calibration applied at the MAVLink uplink "
                         "(MavlinkClient.cmd_rate_scale). DEFAULT 1.0 == byte-identical (no scaling; "
                         "the VQ1 path). The VQ2 sim (build 1.0.3379) realizes a commanded body rate "
                         "~2.5x, so pass ~0.4 (=1/2.5) to feed-forward-compensate. The alternative is "
                         "to let the closed-loop policy absorb the 2.5x via obs[9:12], so 1.0 is the "
                         "safe default and the scale is OPT-IN. Scales BODY_RATE rates only; collective "
                         "thrust is untouched. [VQ2-CONTROL-HANDSHAKE 2026-06-29]")
    # --- SLOW GATE-SEEKER on the case-C self-localizing stack (VQ2 first lap; opt-in) ---
    ap.add_argument("--gate-seeker", action="store_true",
                    help="OPT-IN: fly the TRANSPARENT slow gate-seeker (gate_seeker.py) on the "
                         "case-C self-localizing Navigator (deploy profile) INSTEAD of the RL "
                         "policy. The 'slow is smooth' first VQ2 lap: a pursuit law that flies "
                         "slowly through the active gate's centre, all self-localized (vision "
                         "yaw/z, no mag/baro). DEFAULT OFF (the RL path is unchanged).")
    ap.add_argument("--deploy-profile", default="vq2_case_c",
                    help="named estimator+control preset for --gate-seeker (racer.deploy_profile): "
                         "'vq2_case_c' (self-localizing, default) or 'vq1_case_a' (legacy/given pose). "
                         "Sets the NavigatorConfig flags + the uplink cmd_rate_scale as one bundle.")
    ap.add_argument("--seeker-detector", default="yolo",
                    choices=["red_glow", "yolo", "none"],
                    help="perception detector for --gate-seeker: 'yolo' (DEFAULT; the trained model, "
                         "weights from --seeker-weights) -- changed from 'red_glow' (2026-07-01) so a "
                         "bare --gate-seeker run no longer silently flies the CLASSICAL detector; "
                         "'red_glow' (classical, no GPU) stays available as an explicit opt-in; 'none' "
                         "(estimator coasts on IMU/AHRS with no vision fix -- diagnostic only). The yolo "
                         "path REQUIRES real detector weights via --seeker-weights (fails loud at startup "
                         "if only the RL-actor --checkpoint default is present).")
    ap.add_argument("--seeker-weights", default=None,
                    help="YOLO detector weights for --seeker-detector yolo, SEPARATE from --checkpoint "
                         "(the RL actor). A single .pt path, or an 'a.pt++b.pt' spec to load the "
                         "EnsembleGateDetector (union+dedup before the KF). Falls back to --checkpoint "
                         "if unset. This split is what makes the yolo path deployable: --checkpoint is "
                         "consumed by the RL actor loader, so a detector spec passed there died in "
                         "load_actor (why the trained detector was never in the live gate-seeker loop).")
    ap.add_argument("--seeker-speed", type=float, default=3.0,
                    help="gate-seeker cruise speed cap (m/s). SLOW first (default 3.0): more frames "
                         "per metre, no motion blur, vision yaw/z self-loc works. Ramp later.")
    ap.add_argument("--seeker-settle", type=float, default=0.75,
                    help="post-arm SETTLE hold (s) for --gate-seeker (default 0.75): hold a "
                         "conservative LEVEL attitude + bounded hover thrust + clamped roll/pitch/yaw "
                         "rates while the cold mag-free AHRS gravity-aligns + the gyro bias converges, "
                         "BEFORE any estimator-driven lean. The 2026-06-29 attempt-2 cold-AHRS tumble "
                         "fix. 0 => no settle.")
    ap.add_argument("--seeker-anchor-dets", type=int, default=3,
                    help="map-free ANCHOR-RELEASE streak for --gate-seeker (default 3): leave the "
                         "launch-hold after this many CONSECUTIVE own quality-gated gate detections. "
                         "On the live VQ2 wire the navigator is map-free so nav.time_since_vision_"
                         "update_s never goes finite -- the seeker releases on its OWN detections "
                         "instead (the 2026-06-29 attempt-2 BUG A fix).")
    # --- EGO (21-dim egocentric) RL policy on the case-C stack (opt-in; ego-deploy 2026-07-09) ---
    ap.add_argument("--ego-ckpt", default=None,
                    help="OPT-IN: fly a 21-dim EGOCENTRIC RL actor (.pth, DiffAero format) on the "
                         "case-C self-localizing perception stack (_fly_ego). Action bounds are "
                         "HARDCODED to the ego training values (thrust [0,3.765], rates +-3.14 -- "
                         "the ego launcher writes no sidecar and the legacy [0,5] fallback would "
                         "overdrive thrust ~33%). Needs --seeker-weights (the gate detector). "
                         "DEFAULT OFF: every existing path is byte-identical without it.")
    ap.add_argument("--ego-det-hold", type=float, default=0.2,
                    help="EGO det-proxy hold (s): the slot0 gate is treated as 'detected' while "
                         "the last accepted PnP fix is younger than this, then MASKED TO ZEROS "
                         "(the champion trained obs_coast=OFF -- it EXPECTS the blackout cliff at "
                         "loss-of-lock). Deploy analog of training's per-33ms geometric det: "
                         "detections arrive ~7-15 Hz on the wire, so literal this-tick masking "
                         "would flicker at a duty cycle training never saw. Default 0.2.")
    ap.add_argument("--ego-stale-horizon", type=float, default=0.5,
                    help="EGO confidence staleness horizon (s): conf = clamp(1-age/horizon,0,1), "
                         "hard-masked at 0. Default 0.5 = the champion's EgoEstimatorConfig "
                         "default (the deployed stage overrides nothing). Match the TRAINED "
                         "value of the checkpoint being flown.")
    ap.add_argument("--ego-obs-coast", action=argparse.BooleanOptionalAction, default=False,
                    help="EGO obs blackout-coast (matches training +env.ego_obs_coast): when ON "
                         "the det-proxy hard-mask is dropped and the coasted rel_pos + decaying "
                         "confidence feed the obs until the stale horizon. DEFAULT OFF -- the "
                         "champion candidates trained coast-OFF; only flip for a coast-trained "
                         "checkpoint.")
    ap.add_argument("--ego-rate-scale", type=float, default=1.0,
                    help="EGO uplink cmd_rate_scale. The ego path FORCES this (default 1.0) "
                         "instead of the vq2_case_c profile's 0.4: the RL plant was sysid'd at "
                         "wire scale 1.0, so the policy expects the raw command->realized gain. "
                         "Override only for a deliberate gain experiment.")
    ap.add_argument("--ego-sector-mode", default="auto", choices=["auto", "zero"],
                    help="EGO coarse-sector (obs[9:11]) source: 'auto' = static per-gate bucket "
                         "computed at first acquisition (horiz=0, vert=leveled elevation bucket "
                         "-- the wire analog of training's build_coarse_map); 'zero' = pin (0,0) "
                         "(flat-course / diagnostic fallback).")
    ap.add_argument("--ego-slot1", action="store_true",
                    help="STUB (rejected at startup): fill obs slot1 with the NEXT gate for a "
                         "future multi-gate-trained policy. The current champions are single-"
                         "gate-trained -- slot1 was zero their whole training life; filling it "
                         "is OOD (audit H6). Kept as the CLI seam for the multi-gate generation.")
    ap.add_argument("--yaw-scale",    type=float, default=1.0,
                    help="scale the policy's yaw-rate command (0 = drop yaw). S17 "
                         "mixer mitigation: the policy's per-tick yaw rail dither is "
                         "net-zero motion but forces large motor differentials, which "
                         "the mixer's idle floor turns into ~0.2-0.4 PARASITIC "
                         "collective at low commanded thrust (mixer_probe 2026-06-11).")
    ap.add_argument("--virtual-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="run the policy in a body frame rotated π about body z "
                         "(training flies the course tail-first; the sim spawns "
                         "nose-first — this maps the spawn into distribution). "
                         "DEFAULT ON (offline: required for gate passes).")
    ap.add_argument("--bridge", action=argparse.BooleanOptionalAction, default=False,
                    help="PATH B: fly the proven CTBR stack to ~3 m before gate 0 "
                         "then hand off to the policy. DEFAULT OFF (F-A): the standing start "
                         "(--no-bridge) is the deployment target, deletes the non-deterministic "
                         "bridge seam, AND is map-free -- the bridge loads a gitignored "
                         "data/runs map absent from a clean checkout (uncaught FileNotFoundError "
                         "after arm). inc5+ pass 6/6 in-twin from the raw standing start "
                         "(INC5-LIVE Appendix B; the old '0/6 standing start' was inc4-era).")
    ap.add_argument("--map", default="data/runs/track_map_20260602_114630.json",
                    help="saved deterministic gate map for the bridge navigator")
    ap.add_argument("--handoff-dist", type=float, default=3.0,
                    help="bridge->policy seam: along-track metres before the gate-0 plane")
    ap.add_argument("--handoff-speed-min", type=float, default=4.0,
                    help="bridge->policy seam: minimum speed m/s (validated 4-12)")
    ap.add_argument("--bridge-max-s", type=float, default=25.0,
                    help="abort if the bridge has not reached the seam by then")
    ap.add_argument("--flights",      type=int, default=1,
                    help="number of back-to-back attempts (dev: --dev-auto-reset to sim-reset "
                         "between). DEFAULT 1 = the submission shape.")
    ap.add_argument("--dev-auto-reset", action="store_true",
                    help="DEV-RIG ONLY: permit emitting sim-control commands (MAV_CMD 31000 "
                         "SIM_RESET + Win32 home-kick) to auto-request a fresh race when no GO "
                         "shows up. OFF BY DEFAULT (F-A / R1): on the JUDGED link a client "
                         "sim-control command during a timed run is a §7 DQ, so the submitted "
                         "path NEVER emits one -- it waits passively for the organizer's GO.")
    ap.add_argument("--no-auto-reset", action="store_true",
                    help="hard override: never send MAV_CMD 31000 even with --dev-auto-reset "
                         "(redundant in the default safe config; kept for explicit wrappers).")
    ap.add_argument("--reset-after",  type=float, default=8.0,
                    help="dev only: seconds without a fresh GO before firing a sim reset "
                         "(used only under --dev-auto-reset).")
    ap.add_argument("--max-seconds",  type=float, default=120.0)
    ap.add_argument("--wait-seconds", type=float, default=180.0)
    ap.add_argument("--start-margin-s", type=float, default=0.3)
    ap.add_argument("--finish-hold-s",  type=float, default=1.0)
    ap.add_argument("--connect-timeout",type=float, default=15.0)
    ap.add_argument("--debug-obs", action=argparse.BooleanOptionalAction, default=False,
                    help="write <session>/debug_obs.jsonl: per-step telemetry, the "
                         "labeled 17-dim obs, actor mean/tanh/rescale, and the wire "
                         "command (S17 forensics; ~1 KB/step). DEFAULT OFF (F-A): it is the "
                         "only per-tick blocking file I/O in the control loop -- enable with "
                         "--debug-obs for a forensic dev run.")
    ap.add_argument("--full-reset", action=argparse.BooleanOptionalAction, default=True,
                    help="between flights, exit the race to HOME (ESC+Down*3+Enter) "
                         "and re-enter (Enter*2) so every flight starts from a fresh "
                         "countdown with zero race residue (S17). Falls back to "
                         "MAV_CMD 31000 when the sim window is not found.")
    ap.add_argument("--spin-rate-abort", type=float, default=6.0,
                    help="S17 spin guard: abort when |body rate| exceeds this (rad/s) "
                         "with no gate progress for --spin-time-abort seconds")
    ap.add_argument("--spin-time-abort", type=float, default=2.0)
    ap.add_argument("--arm-attempts", type=int, default=3,
                    help="F-D/AR1: bounded arm re-send attempts before ARM_REFUSED. The final "
                         "attempt escalates to arm(force=True) (21196 pre-arm bypass). 1 = the "
                         "old single-shot behaviour. Total budget stays well inside the 8-min cap.")
    ap.add_argument("--arm-backoff-s", type=float, default=0.5,
                    help="F-D/AR1: pump-while-waiting backoff between arm attempts (s).")
    ap.add_argument("--odo-stale-s", type=float, default=0.15,
                    help="F-C ODOMETRY freshness gate: treat the attitude/rate as STALE when "
                         "its per-field arrival age exceeds this (s). 0.15 s = ~11 missed 75 Hz "
                         "ODOMETRY frames -> ~10x the benign inter-arrival jitter, so it never "
                         "trips a healthy run but catches a selective ODOMETRY drop (audit D1/D2).")
    ap.add_argument("--odo-recovery-s", type=float, default=0.5,
                    help="F-C: SAFE-HOVER for up to this long waiting for a stale/non-finite "
                         "ODOMETRY stream to recover, then abort+disarm (ODO_STALE/NON_FINITE). "
                         "0.5 s < the 1.5 s sim-stall and 2.0 s spin windows, so the odo gate "
                         "acts first on a selective drop while bounding open-loop time.")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    # FAIL LOUD before any connection/load if the gate-seeker yolo path has no usable detector weights
    # (2026-07-01: --seeker-detector now defaults to yolo, so a bare --gate-seeker must NOT silently
    # fall back to the RL-actor --checkpoint .pth as if it were YOLO weights).
    _validate_seeker_detector(args)

    # --ego-slot1 is a STUB seam for future multi-gate policies: reject LOUD at startup rather
    # than silently flying a single-gate champion with an OOD-filled slot1 (audit H6).
    if getattr(args, "ego_slot1", False):
        raise SystemExit("--ego-slot1 is a STUB: slot1 filling is not implemented (the deployed "
                         "champions are single-gate-trained; filling slot1 is OOD -- audit H6). "
                         "Remove the flag.")

    # -- load checkpoint (RL actor) --
    # Under --gate-seeker the RL actor is UNUSED: fly_once passes it straight through to the
    # gate-seeker early-return in _fly_armed without ever calling it. --checkpoint's default is the
    # inc7 actor, and the gate-seeker's YOLO detector takes its weights from --seeker-weights. So
    # SKIP the actor load on the gate-seeker path -- (a) a run needs no valid actor .pth, and (b) a
    # detector-weights spec can never be mis-fed to load_actor (the overload that kept yolo undeployed).
    if getattr(args, "gate_seeker", False):
        actor = None
        print("gate-seeker: skipping RL actor load (unused on this path)")
    elif getattr(args, "ego_ckpt", None):
        # EGO path: hardcoded action bounds (no sidecar exists for ego checkpoints; the legacy
        # [0,5] fallback would overdrive thrust ~33%), obs width pinned to 21 at load.
        print(f"loading EGO actor: {args.ego_ckpt}")
        actor = load_ego_actor(args.ego_ckpt)
        print(f"  type: {type(actor).__name__}")
        _out = actor(torch.zeros(1, _EGO_OBS_DIM))
        _act = (_out[0] if isinstance(_out, (tuple, list)) else _out)[0]
        print(f"  obs_dim={_EGO_OBS_DIM} -> action shape={tuple(_act.shape)}  (expected (4,))")
        if _act.shape != (4,):
            print(f"  WARNING: unexpected action shape {tuple(_act.shape)}; "
                  "check DiffAero actor architecture.", file=sys.stderr)
    else:
        print(f"loading actor: {args.checkpoint}")
        actor = load_actor(args.checkpoint)
        print(f"  type: {type(actor).__name__}")
        _out = actor(torch.zeros(1, 17))
        _act = (_out[0] if isinstance(_out, (tuple, list)) else _out)[0]
        print(f"  obs_dim=17 -> action shape={tuple(_act.shape)}  (expected (4,))")
        if _act.shape != (4,):
            print(f"  WARNING: unexpected action shape {tuple(_act.shape)}; "
                  "check DiffAero actor architecture.", file=sys.stderr)

    # -- MAVLink --
    # cmd_rate_scale (default 1.0 == identity / byte-identical VQ1 path). ~0.4 compensates the
    # VQ2 ~2.5x command->realized body-rate gain at the uplink (see build_parser --cmd-rate-scale).
    # Under --gate-seeker the deploy profile's cmd_rate_scale wins UNLESS the user passed an explicit
    # --cmd-rate-scale (!= the 1.0 default) -- so the case-C profile is one flag, but still overridable.
    cmd_rate_scale = args.cmd_rate_scale
    # LIVE-WIRE gyro-sign correction: identity (1,1,1) == byte-identical default; the deploy profile
    # supplies the VQ2 HIGHRES_IMU pitch flip (vq2_case_c -> (1,-1,1)) at the SAME seam as cmd_rate_scale.
    gyro_sign = (1.0, 1.0, 1.0)
    if getattr(args, "gate_seeker", False) and args.cmd_rate_scale == 1.0:
        from racer.deploy_profile import get_profile
        _profile = get_profile(args.deploy_profile)
        cmd_rate_scale = _profile.cmd_rate_scale
        gyro_sign = _profile.gyro_sign
        print(f"  [gate-seeker] deploy profile {args.deploy_profile!r} -> "
              f"cmd_rate_scale={cmd_rate_scale:g} gyro_sign={tuple(gyro_sign)}")
    elif getattr(args, "ego_ckpt", None):
        # EGO path: the profile supplies ONLY the gyro sign (the wire convention the AHRS needs);
        # cmd_rate_scale is FORCED to --ego-rate-scale (default 1.0 -- the RL plant was sysid'd
        # at wire scale 1.0; the seeker profile's 0.4 belongs to the classical pursuit gains, NOT
        # to the trained policy). LOUD by design: this is the knob that silently detunes a policy.
        from racer.deploy_profile import get_profile
        _profile = get_profile(args.deploy_profile)
        gyro_sign = _profile.gyro_sign
        cmd_rate_scale = float(args.ego_rate_scale)
        print(f"  [ego] FORCING cmd_rate_scale={cmd_rate_scale:g} (--ego-rate-scale; profile "
              f"{args.deploy_profile!r} value {_profile.cmd_rate_scale:g} NOT applied) "
              f"gyro_sign={tuple(gyro_sign)}")
    client = MavlinkClient(args.endpoint, cmd_rate_scale=cmd_rate_scale, gyro_sign=gyro_sign)
    if args.cmd_rate_scale != 1.0:
        print(f"  [vq2] cmd_rate_scale={args.cmd_rate_scale:g} -> BODY_RATE commands scaled at the "
              f"uplink (command->realized ~{1.0 / args.cmd_rate_scale:.2f}x compensation).")
    print(f"connecting {args.endpoint} ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)

    # recorder holder: the video/mavlink taps write into the CURRENT flight's
    # recorder (swapped per flight; None between flights)
    holder: dict = {"rec": None}
    stop = threading.Event()

    client._latest_frame = None   # gate-seeker perception source (freshest reassembled Frame)

    # --- video-thread diagnostics (A17 freeze hunt) ------------------------------------
    # ADDITIVE / logging-only: pins WHERE a multi-second onboard-video freeze lives (wire
    # starvation vs. publish-side stall vs. a swallowed reconnect) by timing the SAME two
    # things the control loop already times ([loop-rate]/[vision-timing] above) but for the
    # video thread's own two phases: (a) the gap between successive PUBLISHED frames, which
    # is dominated by socket wait when the sim stops emitting UDP (the leading hypothesis),
    # and (b) the cost of the publish+record step itself, which would instead pin a stall on
    # OUR side (e.g. rec.record_frame). Mirrors navigator._time_step's defensive style: every
    # timing/bookkeeping op is try/except-wrapped so an instrumentation bug can never raise
    # into the video loop and starve client._latest_frame for real. Counters are plain ints
    # and a bounded deque -- GIL-atomic reads/writes, no new locks, no change to the publish/
    # record semantics or the reconnect behaviour (still sleep 0.5 + loop).
    vstats: dict = {
        "n_frames": 0,
        "max_gap_ms": 0.0,
        "max_gap_fid": None,
        "gaps_over_200": 0,
        "gaps_over_1000": 0,
        "gap_sum_ms": 0.0,
        "max_pub_ms": 0.0,
        "reconnects_idle": 0,
        "reconnects_exc": 0,
        "last_session_s": 0.0,   # duration of the last `with JpegUdpReceiver(...)` session
        "wire": None,   # last rx.metrics snapshot (ReceiverMetrics), or None if never seen
        "ring": deque(maxlen=64),   # bounded: (ts, frame_id, gap_ms, pub_ms, exc_repr|None)
    }

    def _video():
        t_prev = None   # monotonic timestamp of the last PUBLISHED frame (persists across
                         # reconnects on purpose: a gap spanning a reconnect IS the freeze)
        while not stop.is_set():
            # --- receiver-wait / wire-health bracket: monotonic before/after each `with
            # JpegUdpReceiver(...)` session, so a hang INSIDE the `with` (vs. between
            # sessions, e.g. the 0.5s reconnect sleep) is distinguishable if ever needed. Not
            # printed mid-flight (see the exit-only rule); kept on vstats for post-hoc reading.
            t_rx_open = time.monotonic()
            try:
                with JpegUdpReceiver(port=args.video_port) as rx:
                    for fr in rx.frames(max_wait_s=5.0):
                        # --- inter-published-frame gap + publish-cost split (additive) ---
                        try:
                            now_m = time.monotonic()
                            if t_prev is not None:
                                gap_ms = (now_m - t_prev) * 1e3
                            else:
                                gap_ms = 0.0
                            t_prev = now_m
                        except Exception:
                            gap_ms = 0.0

                        # publish the freshest frame for the gate-seeker's case-C perception
                        # (the navigator is frame_id-idempotent, so re-reads are harmless).
                        _pub_t0 = time.perf_counter()
                        client._latest_frame = fr
                        rec = holder["rec"]
                        if rec is not None:
                            rec.record_frame(fr)
                        try:
                            pub_ms = (time.perf_counter() - _pub_t0) * 1e3
                        except Exception:
                            pub_ms = 0.0

                        try:
                            vstats["n_frames"] += 1
                            vstats["gap_sum_ms"] += gap_ms
                            if gap_ms > vstats["max_gap_ms"]:
                                vstats["max_gap_ms"] = gap_ms
                                vstats["max_gap_fid"] = fr.frame_id
                            if pub_ms > vstats["max_pub_ms"]:
                                vstats["max_pub_ms"] = pub_ms
                            if gap_ms > 200.0:
                                vstats["gaps_over_200"] += 1
                                vstats["ring"].append(
                                    (now_m, fr.frame_id, gap_ms, pub_ms, None))
                            if gap_ms > 1000.0:
                                vstats["gaps_over_1000"] += 1
                        except Exception:
                            pass

                        if stop.is_set():
                            break
                    # --- for-loop exit with NO exception: either the idle timeout (`rx.frames`'s
                    # own max_wait_s=5.0 -- no datagram for 5s) fired, or `break` above on stop
                    # (clean shutdown). A17 fix: snapshot rx.metrics UNCONDITIONALLY on EVERY normal
                    # for-loop exit (both paths), so the exit [video-thread] line ALWAYS shows
                    # datagrams/completed/evicted/decode_failed -- the ONLY way to split "sim stopped
                    # emitting" (datagrams flat) from "packet loss" (evicted/decode_failed climbing).
                    # In A17 the idle-timeout never fired (the loop broke on stop), so the OLD
                    # stop-gated snapshot left us with no wire counters at all. ``reconnects_idle``
                    # stays gated on ``not stop`` (its meaning = "actually going to reconnect").
                    try:
                        vstats["last_session_s"] = time.monotonic() - t_rx_open
                        m = getattr(rx, "metrics", None)
                        if m is not None:
                            vstats["wire"] = m
                        if not stop.is_set():
                            vstats["reconnects_idle"] += 1
                    except Exception:
                        pass
            except Exception as exc:
                # UN-BLIND: this used to be a silent `except Exception: pass`, a dangerous
                # blind spot over the exact failure mode we're hunting. Same reconnect
                # behaviour (sleep 0.5 + loop below) -- just make it visible.
                try:
                    vstats["reconnects_exc"] += 1
                    vstats["ring"].append(
                        (time.monotonic(), None, 0.0, 0.0, repr(exc)))
                    # A17 fix: snapshot rx.metrics on the exception exit too (rx is bound once the
                    # `with` __enter__ succeeded, i.e. the exception came from .frames()/publish), so
                    # even a receiver-error exit still leaves the last wire counters on vstats.
                    m = getattr(locals().get("rx", None), "metrics", None)
                    if m is not None:
                        vstats["wire"] = m
                except Exception:
                    pass
            if not stop.is_set():
                time.sleep(0.5)
    vthread = threading.Thread(target=_video, name="video", daemon=True)
    vthread.start()

    # PRE-WARM the YOLO detector NOW -- before GO / arm / the recorder attaching (the A15 launch-
    # window-freeze fix). Moves the one-time ~2.9 s Ultralytics/CUDA first-predict off the flight
    # critical path so tick 0 is fast and the launch is OBSERVABLE (no camera/recorder freeze, no
    # GPU-contention frame drops over the launch). No-op unless --gate-seeker + --seeker-detector yolo.
    _prewarm_detector(args)

    def _on_msg(msg):
        if msg.get_type() == "BAD_DATA":
            return
        rec = holder["rec"]
        if rec is None:
            return
        buf = msg.get_msgbuf()
        if buf:
            rec.record_mavlink(bytes(buf))
    client.on_message = _on_msg

    results = []
    rc = 0
    try:
        for flight in range(1, args.flights + 1):
            print(f"\n{'='*22} FLIGHT {flight}/{args.flights} {'='*22}")
            session  = Path("data/runs") / f"{session_stamp()}_{args.label}_f{flight}"
            recorder = Recorder(session)
            recorder.start()
            recorder.add_meta(
                endpoint=args.endpoint, label=args.label, flight=flight,
                rate_hz=args.rate, max_rate=args.max_rate, yaw_scale=args.yaw_scale,
                cmd_rate_scale=args.cmd_rate_scale,
                virtual_flip=args.virtual_flip, bridge=args.bridge,
                handoff_dist=args.handoff_dist,
                handoff_speed_min=args.handoff_speed_min,
                checkpoint=str(args.checkpoint), max_seconds=args.max_seconds,
                # ego meta ONLY on the ego path (VQ1/gate-seeker meta.json byte-identical)
                **({"ego_ckpt": str(args.ego_ckpt),
                    "ego_det_hold": args.ego_det_hold,
                    "ego_stale_horizon": args.ego_stale_horizon,
                    "ego_obs_coast": args.ego_obs_coast,
                    "ego_rate_scale": args.ego_rate_scale,
                    "ego_sector_mode": args.ego_sector_mode}
                   if getattr(args, "ego_ckpt", None) else {}),
            )
            holder["rec"] = recorder
            print(f"recording -> {session}")

            res = {"flight": flight, "final_state": "ERROR", "gate_index": 0}
            try:
                res = fly_once(client, actor, args, flight, session_dir=session)
            except KeyboardInterrupt:
                print("\nCtrl-C -> stopping.")
                try:
                    client.disarm(force=True)
                except Exception:
                    pass
                res["final_state"] = "INTERRUPTED"
                results.append({**res, "session": str(session)})
                raise
            except Exception as exc:
                # F-B: any crash in fly_once is already disarmed by its finally; this is a
                # backstop + LOUD log + re-raise so the failure is never silently swallowed.
                print(f"\n[safety] flight {flight} crashed "
                      f"({type(exc).__name__}: {exc}) -> backstop disarm; re-raising.",
                      file=sys.stderr)
                try:
                    client.disarm(force=True)
                except Exception:
                    pass
                res["final_state"] = "EXCEPTION"
                results.append({**res, "session": str(session)})
                raise
            finally:
                holder["rec"] = None
                recorder.add_meta(
                    final_state=res.get("final_state"),
                    gate_index=res.get("gate_index", 0),
                    collisions=len(client.collisions),
                    race_status=client.race_status,
                )
                recorder.close()

            res["session"] = str(session)
            results.append(res)
            print(f"\n  flight {flight}: {res['final_state']}  "
                  f"gates={res['gate_index']}  session={session}")
            if res["final_state"] in ("NO_GO", "ARM_REFUSED"):
                print("  cannot start races -> stopping the batch.", file=sys.stderr)
                break
            if args.full_reset and flight < args.flights:
                # S17: full ESC->HOME->Enter*2 reset so the next flight starts from a
                # fresh countdown with no race residue (31000 restarts can carry the
                # old race's RACE_STATUS + collision state).
                print("  [full-reset] exiting race to HOME and re-entering ...")
                full_sim_reset()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # F-B: do NOT silently swallow. The vehicle is already disarmed (fly_once finally +
        # per-flight backstop); log loudly, disarm once more defensively, exit non-zero.
        traceback.print_exc()
        print(f"\n[safety] run crashed ({type(exc).__name__}: {exc}); vehicle disarmed "
              f"(fly_once finally + handler backstop). Stopping.", file=sys.stderr)
        try:
            client.disarm(force=True)
        except Exception:
            pass
        rc = 1
    finally:
        stop.set()
        vthread.join(timeout=6.0)

    # --- [video-thread] exit summary: mirrors [loop-rate]/[vision-timing] above, but for the
    # video thread (spans ALL flights, not one -- the thread outlives fly_once). Printed ONCE
    # here (never mid-flight, so it can't contend with the tick loop's \r status line). Pins
    # WHERE a multi-second freeze lives: a wire-side starvation shows up as a big max_gap_ms
    # with a wire snapshot showing few/no new datagrams; a publish-side stall shows up as a
    # big max_pub_ms instead; a swallowed reconnect exception now shows up as reconnects>0
    # (exc) with the repr() in the ring dump, instead of vanishing into `except: pass`.
    try:
        vs = vstats
        wire = vs.get("wire")
        if wire is not None:
            wire_str = (f" | wire: datagrams={wire.datagrams} completed={wire.frames_completed} "
                        f"evicted={wire.partials_evicted} dup={wire.duplicate_datagrams} "
                        f"decode_failed={wire.frames_decode_failed} "
                        f"size_mismatch={wire.frames_size_mismatch} "
                        f"bad_chunkmap={wire.frames_bad_chunkmap} "
                        f"short={wire.short_datagrams}")
        else:
            wire_str = " | wire: (no rx.metrics snapshot -- receiver never opened a session)"
        print(f"  [video-thread] frames={vs['n_frames']} "
              f"max_gap={vs['max_gap_ms']:.0f}ms@fid={vs['max_gap_fid']} "
              f"gaps>200ms={vs['gaps_over_200']} gaps>1s={vs['gaps_over_1000']} "
              f"gap_sum={vs['gap_sum_ms']:.0f}ms max_pub={vs['max_pub_ms']:.1f}ms "
              f"reconnects={vs['reconnects_idle'] + vs['reconnects_exc']}"
              f"(idle={vs['reconnects_idle']},exc={vs['reconnects_exc']})" + wire_str)
        if vs["gaps_over_1000"] > 0 and vs["ring"]:
            print("  [video-thread] ring buffer (gap/exception events, ts/frame_id/gap_ms/pub_ms/exc):")
            for ts, fid, gap_ms, pub_ms, exc in vs["ring"]:
                print(f"    ts={ts:.3f} fid={fid} gap={gap_ms:.0f}ms pub={pub_ms:.1f}ms exc={exc}")
    except Exception as exc:
        print(f"  [video-thread] WARNING: summary print failed ({type(exc).__name__}: {exc}); "
              "raw counters not shown (logging bug, not flight bug).")

    # -- summary --
    print("\n==== fly_rl summary ====")
    for r in results:
        print(f"  flight {r['flight']}: {r['final_state']:<10} gates={r['gate_index']}"
              f"  {r.get('session','')}")
    if client.collisions:
        ids = sorted({c["id"] for c in client.collisions})
        print(f"  collisions (all flights): {len(client.collisions)} "
              f"(ids {ids}; 1001=gate 1002=env)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
