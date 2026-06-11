"""rl/fly_rl.py  --  Stage-1 inc-1 CTBR RL policy deployment in the VQ1 sim.

Loads the trained actor checkpoint, connects via MAVLink, waits for a fresh race
start (auto-requesting one via MAV_CMD 31000 when possible), arms, then flies at
the TRAINING control rate (30 Hz = racing.yaml dt 0.0333): builds the 17-dim
observation (exactly matching peregrine_racing.get_observations), replicates the
training action pipeline (tanh -> rescale to [min,max] action bounds), sends
SET_ATTITUDE_TARGET (BODY_RATE).  Records each flight for race_outcome, and can
chain --flights N attempts back-to-back via the sim-reset command.

DEPLOYMENT MATH (verified against the diffaero source the checkpoint trained on,
S1.2 session 2026-06-10 -- see the constants below for the per-item derivations):
  * action = tanh(actor_mean(obs)); env action = min + (max-min)*(action+1)/2
    (StochasticActor.forward test branch + BaseEnv.rescale_action; the raw mean
    is NEVER sent directly).
  * obs[12] (collective) = the RESCALED normed_thrust of the previous action;
    0.0 at episode start (BaseEnv.last_action zeroed in reset_idx).
  * policy rates are FLU body rates; the training dynamics adapter maps them to
    the plant with the plain FLU->FRD flip [1,-1,-1] and the PLANT applies
    rate_gain*rate_sign -- identically in the live sim (that is the system-ID).
    So the wire command is rate_flu * [1,-1,-1]; no ff_gain, no rate_gain algebra.
  * raw ODOMETRY angular_rate -> true FRD rate = *[-1,-1,1] (flight-proven VQ1
    odo_rate_sign; re-verified offline 2026-06-10 by correlating raw rates vs
    quat-finite-difference rates on a course recording: corr [+,-,+] vs the
    REPORTED-attitude derivative, whose roll is itself inverted).

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from racer.contracts import ControlCommand, ControlMode
from racer.finish_hold import drain_until_finish, sim_finish_confirmed
from racer.firstcontact import telemetry_summary
from racer.frames import R_world_from_body as _R_world_from_body
from racer.frames import euler_from_quat_wxyz as _euler_from_q
from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp
from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver

# ---------------------------------------------------------------------------
# Frame & sign constants — must match peregrine_racing.py + diffaero_dynamics.py
# ---------------------------------------------------------------------------
# R_x(π) = diag(1,-1,-1) maps NED↔Z-up (world) and FRD↔FLU (body).
_FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float64)  # world & body frame flip

# Raw ODOMETRY angular_rate -> TRUE FRD body rate (reporting artifact undo).
# Flight-proven VQ1 value (fly_vq1 --odo-rate-sign=-1,-1,1); re-verified offline
# 2026-06-10: corr(raw, quat-derived) = [+0.998, -0.998, +0.999] where the
# quat-derived rate is the REPORTED-attitude derivative (roll inverted) =>
# raw->true = [+, -, +] * [-1, 1, 1] = [-1, -1, +1].
_ODO_RATE_SIGN = np.array([-1.0, -1.0, 1.0], dtype=np.float64)

# RL action rates (FLU, diffaero Z-up world) -> FRD wire command. The training
# adapter (diffaero_dynamics._action_diffaero_to_ctbr_np) does rate_frd =
# U[1:4] * [1,-1,-1] and the plant then applies rate_gain*rate_sign — the SAME
# amplification the live sim applies (system-ID'd plant). Identical realized
# physics therefore needs only the same FLU->FRD flip on the wire command.
_ACT_FLU_TO_FRD = _FLIP

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

# Obs dim labels (debug dumps + replay_obs.py). Matches obs_from_zup's layout.
OBS_LABELS = (
    [f"pos_g{i}" for i in "xyz"] + [f"vel_g{i}" for i in "xyz"]
    + ["rpy_g_r", "rpy_g_p", "rpy_g_y"] + [f"w_flu{i}" for i in "xyz"]
    + ["collective_prev"] + [f"nxt_rel{i}" for i in "xyz"] + ["nxt_relyaw"]
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
                 last_normed_thrust: float, virtual_flip: bool = False) -> np.ndarray:
    """17-dim obs (float32) from TRUE state already in the DiffAero Z-up/FLU frame.

    Layout (matching peregrine_racing.py):
      [0:3]   pos_g        R_w2g @ (gate_pos - pos)          gate-relative position
      [3:6]   vel_g        R_w2g @ vel                        velocity, gate frame
      [6:9]   rpy_g        ZYX euler of (R_w2g @ R_b2w)       attitude vs gate
      [9:12]  body_rates   FLU body rates
      [12]    collective   previous RESCALED normed_thrust (0.0 at episode start)
      [13:16] next_relpos  pre-computed next-gate relative position
      [16]    next_relyaw  0 (uniform gate yaw)
    """
    if virtual_flip:
        R_b2w_zup = R_b2w_zup @ _RZ_PI_BODY     # body axes rotated π about body z
        w_flu = _RZ_PI_BODY @ w_flu
    gp    = _GATE_POS_ZUP[target_gate]
    pos_g = _R_W2G @ (gp - pos_zup)
    vel_g = _R_W2G @ vel_zup
    rpy_g = _euler_zyx(_R_W2G @ R_b2w_zup)
    nxt   = min(target_gate + 1, N_GATES - 1)
    obs = np.concatenate([
        pos_g, vel_g, rpy_g, w_flu,
        [last_normed_thrust],
        _GATE_REL_POS[nxt],
        [_GATE_YAW_REL[nxt]],
    ])
    return obs.astype(np.float32)


def build_obs(state, target_gate: int, last_normed_thrust: float,
              virtual_flip: bool = False) -> np.ndarray:
    """Telemetry -> 17-dim obs: undo the ODOMETRY reporting artifacts (roll-inverted
    quat, [-1,-1,1] raw rate), convert NED/FRD -> Z-up/FLU, then obs_from_zup."""
    pos_ned = np.asarray(state.position_ned,         dtype=np.float64)
    vel_ned = np.asarray(state.velocity_ned,         dtype=np.float64)
    q_raw   = np.asarray(state.orientation_ned_wxyz, dtype=np.float64)
    w_raw   = np.asarray(state.angular_rate_body,    dtype=np.float64)

    # Attitude: ODOMETRY quat reports roll INVERTED (odo_att_sign=[-1,1,1]).
    # Extract ZYX euler, negate roll, rebuild R_FRD2NED, sandwich with the flip.
    roll, pitch, yaw = _euler_from_q(q_raw)
    R_frd2ned = _R_world_from_body(-roll, pitch, yaw)
    R_b2w_zup = (_FLIP[:, None] * R_frd2ned) * _FLIP[None, :]   # FLU -> Z-up world

    w_frd = w_raw * _ODO_RATE_SIGN              # raw -> true FRD body rates
    return obs_from_zup(pos_ned * _FLIP, vel_ned * _FLIP, R_b2w_zup,
                        w_frd * _FLIP, target_gate, last_normed_thrust,
                        virtual_flip=virtual_flip)


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
    """17 -> 256 -> 128 -> 4 MLP with LN blocks (cfg/network/mlp.yaml hidden_dim)."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            _LNBlock(17, 256),
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
        actor = _ActorMean()
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


@torch.no_grad()
def policy_step(actor: nn.Module, obs_np: np.ndarray, max_rate: float = 0.0,
                virtual_flip: bool = False, max_thrust: float = 0.0,
                debug: dict | None = None) -> tuple[np.ndarray, float, float]:
    """One forward pass, replicating the TRAINING action pipeline exactly:
    test-mode action = tanh(actor_mean(obs)), then env.rescale_action onto
    [_ACT_MIN, _ACT_MAX].

    Returns:
      body_rate_frd  (3,)  rad/s, FRD  — ready for ControlCommand.body_rate
      collective     float [0,1]       — ready for ControlCommand.thrust
      normed_thrust  float [0,5]       — RESCALED action[0]; next obs[12]

    ``debug``: pass a dict to receive the pipeline internals (raw mean, tanh,
    rescaled action) for the --debug-obs per-step dump.
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
    user32 = ctypes.windll.user32
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)
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
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)
    for _ in range(n_enter):
        _send_key(_VK_RETURN, settle_s=settle_s)
    return True


# ---------------------------------------------------------------------------
# Race lifecycle helpers
# ---------------------------------------------------------------------------
def wait_fresh_go(client, args, auto_reset: bool) -> bool:
    """Pump until a FRESH race GO (race_start within the last 2 s and past the
    start margin, drone near origin).  If auto_reset, fire send_sim_reset()
    (MAV_CMD 31000) whenever no fresh GO shows up for --reset-after seconds."""
    deadline   = time.monotonic() + args.wait_seconds
    margin_ms  = args.start_margin_s * 1000.0
    next_reset = time.monotonic() + (args.reset_after if auto_reset else 1e18)
    last_p     = 0.0
    stale_strikes = 0     # consecutive ineffective 31000s (sim parked off-race / at HOME)
    while time.monotonic() < deadline:
        client.pump()
        s  = client.state
        rs = client.race_status
        live = s.position_ned is not None and s.sim_time_ns > 0
        now  = time.monotonic()
        if rs and rs["started"] and live:
            to_go = rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]
            fresh = rs["race_start_boot_time_ms"] >= 0 and to_go > -2000.0
            if fresh and to_go <= -margin_ms:
                pos_off = float(np.linalg.norm(s.position_ned))
                if pos_off > 5.0:
                    print(f"\n  stale GO: drone {pos_off:.0f} m from origin — waiting on.",
                          file=sys.stderr)
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
    return False


def fly_once(client, actor, args, flight_idx: int,
             session_dir: Path | None = None) -> dict:
    """One full attempt: wait fresh GO -> arm -> RL control loop -> finish hold ->
    disarm.  Returns a per-flight result dict.  Caller owns recorder lifecycle."""
    from pymavlink import mavutil

    result = {"flight": flight_idx, "final_state": "NO_GO", "gate_index": 0,
              "collisions_at_start": len(client.collisions)}

    # ------------------------------------------------------------------
    # Fresh race GO (auto-reset between flights; manual fallback hint)
    # ------------------------------------------------------------------
    if flight_idx == 1:
        print(">>> Need a FRESH race (auto-reset will be tried; manual: home -> "
              "waiting room -> Race, ~3 s countdown).")
    if not wait_fresh_go(client, args, auto_reset=not args.no_auto_reset):
        print("no GO -> abort flight.", file=sys.stderr)
        return result

    # ------------------------------------------------------------------
    # Arm
    # ------------------------------------------------------------------
    print("\n[arm] ...")
    client.last_command_ack = None
    client.arm()
    ack   = client.wait_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout_s=3.0)
    armed = client.wait_armed(True, timeout_s=5.0)
    print(f"  ACK={ack['result_name'] if ack else 'none'}  armed={armed}")
    if not armed:
        print("  arming refused -> abort flight.", file=sys.stderr)
        result["final_state"] = "ARM_REFUSED"
        return result

    # ------------------------------------------------------------------
    # PATH B bridge: CTBR launcher to the handoff seam
    # ------------------------------------------------------------------
    if args.bridge:
        bres = run_bridge(client, args, gate0_x_ned=float(_GATE_POS_ZUP[0, 0]))
        if bres != "HANDOFF":
            result["final_state"] = bres
            try:
                client.disarm(force=True)
            except Exception:
                pass
            return result

    # ------------------------------------------------------------------
    # RL control loop at --rate Hz (training cadence)
    # ------------------------------------------------------------------
    print(f"\n[fly] RL policy  rate={args.rate:g} Hz  max_rate="
          f"{args.max_rate if args.max_rate > 0 else 'off'}  "
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

    # --- --debug-obs per-step dump (S17): everything the policy saw and emitted.
    dbg_f = None
    dbg_k = 0
    if args.debug_obs and session_dir is not None:
        dbg_f = open(session_dir / "debug_obs.jsonl", "w", encoding="utf-8")
        dbg_f.write(json.dumps({
            "type": "header", "flight": flight_idx, "checkpoint": str(args.checkpoint),
            "act_min": _ACT_MIN.tolist(), "act_max": _ACT_MAX.tolist(),
            "hover_thrust": _HOVER_THRUST, "rate_hz": args.rate,
            "max_rate": args.max_rate, "virtual_flip": args.virtual_flip,
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

            # --- S17 spin guard: sustained high body rate with zero gate progress ---
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

            # --- build obs & run policy ---
            if (s.position_ned is None or s.velocity_ned is None
                    or s.orientation_ned_wxyz is None
                    or s.angular_rate_body is None):
                continue   # ODOMETRY not arrived yet

            obs = build_obs(s, gate_index, last_normed, virtual_flip=args.virtual_flip)
            dbg: dict | None = {} if dbg_f is not None else None
            rate_frd, collective, last_normed = policy_step(
                actor, obs, args.max_rate, virtual_flip=args.virtual_flip, debug=dbg)

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
                    "odo_age_ms": round((time.monotonic_ns() - s.recv_monotonic_ns) / 1e6, 1),
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

    print("\n[safety] disarming ...")
    try:
        client.disarm(force=True)
        client.wait_armed(False, timeout_s=3.0)
    except Exception as exc:
        print(f"  disarm error: {exc}", file=sys.stderr)

    result["final_state"] = final_state
    result["gate_index"]  = gate_index
    result["collisions"]  = len(client.collisions) - n_coll0
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default=str(Path(__file__).resolve().parent / "checkpoints"
                                / "stage1_inc4_actor.pth"),
                    help="path to actor .pth (its .json sidecar, if present, sets the "
                         "trained action bounds)")
    ap.add_argument("--endpoint",     default="udp:127.0.0.1:14550")
    ap.add_argument("--video-port",   type=int, default=VIDEO_PORT)
    ap.add_argument("--label",        default="rl_s1")
    ap.add_argument("--rate",         type=float, default=30.0,
                    help="control loop Hz; default 30 = the TRAINING dt 0.0333 "
                         "(racing.yaml). The policy was trained on 33 ms action holds.")
    ap.add_argument("--max-rate",     type=float, default=0.0,
                    help="cap |body-rate| (rad/s, per-axis, FLU) before sending; "
                         "0 = off. OOD-start diagnostic (PATH A).")
    ap.add_argument("--virtual-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="run the policy in a body frame rotated π about body z "
                         "(training flies the course tail-first; the sim spawns "
                         "nose-first — this maps the spawn into distribution). "
                         "DEFAULT ON (offline: required for gate passes).")
    ap.add_argument("--bridge", action=argparse.BooleanOptionalAction, default=True,
                    help="PATH B: fly the proven CTBR stack to ~3 m before gate 0 "
                         "then hand off to the policy. DEFAULT ON (offline: 4/6 "
                         "gates vs 0/6 from the raw standing start).")
    ap.add_argument("--map", default="data/runs/track_map_20260602_114630.json",
                    help="saved deterministic gate map for the bridge navigator")
    ap.add_argument("--handoff-dist", type=float, default=3.0,
                    help="bridge->policy seam: along-track metres before the gate-0 plane")
    ap.add_argument("--handoff-speed-min", type=float, default=4.0,
                    help="bridge->policy seam: minimum speed m/s (validated 4-12)")
    ap.add_argument("--bridge-max-s", type=float, default=25.0,
                    help="abort if the bridge has not reached the seam by then")
    ap.add_argument("--flights",      type=int, default=1,
                    help="number of back-to-back attempts (auto sim-reset between)")
    ap.add_argument("--no-auto-reset", action="store_true",
                    help="never send MAV_CMD 31000; wait for a manual race start")
    ap.add_argument("--reset-after",  type=float, default=8.0,
                    help="seconds without a fresh GO before firing a sim reset")
    ap.add_argument("--max-seconds",  type=float, default=120.0)
    ap.add_argument("--wait-seconds", type=float, default=180.0)
    ap.add_argument("--start-margin-s", type=float, default=0.3)
    ap.add_argument("--finish-hold-s",  type=float, default=1.0)
    ap.add_argument("--connect-timeout",type=float, default=15.0)
    ap.add_argument("--debug-obs", action=argparse.BooleanOptionalAction, default=True,
                    help="write <session>/debug_obs.jsonl: per-step telemetry, the "
                         "labeled 17-dim obs, actor mean/tanh/rescale, and the wire "
                         "command (S17 forensics; ~1 KB/step). DEFAULT ON.")
    ap.add_argument("--full-reset", action=argparse.BooleanOptionalAction, default=True,
                    help="between flights, exit the race to HOME (ESC+Down*3+Enter) "
                         "and re-enter (Enter*2) so every flight starts from a fresh "
                         "countdown with zero race residue (S17). Falls back to "
                         "MAV_CMD 31000 when the sim window is not found.")
    ap.add_argument("--spin-rate-abort", type=float, default=6.0,
                    help="S17 spin guard: abort when |body rate| exceeds this (rad/s) "
                         "with no gate progress for --spin-time-abort seconds")
    ap.add_argument("--spin-time-abort", type=float, default=2.0)
    args = ap.parse_args()

    # -- load checkpoint --
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
    client = MavlinkClient(args.endpoint)
    print(f"connecting {args.endpoint} ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)

    # recorder holder: the video/mavlink taps write into the CURRENT flight's
    # recorder (swapped per flight; None between flights)
    holder: dict = {"rec": None}
    stop = threading.Event()

    def _video():
        while not stop.is_set():
            try:
                with JpegUdpReceiver(port=args.video_port) as rx:
                    for fr in rx.frames(max_wait_s=5.0):
                        rec = holder["rec"]
                        if rec is not None:
                            rec.record_frame(fr)
                        if stop.is_set():
                            break
            except Exception:
                pass
            if not stop.is_set():
                time.sleep(0.5)
    vthread = threading.Thread(target=_video, name="video", daemon=True)
    vthread.start()

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
    try:
        for flight in range(1, args.flights + 1):
            print(f"\n{'='*22} FLIGHT {flight}/{args.flights} {'='*22}")
            session  = Path("data/runs") / f"{session_stamp()}_{args.label}_f{flight}"
            recorder = Recorder(session)
            recorder.start()
            recorder.add_meta(
                endpoint=args.endpoint, label=args.label, flight=flight,
                rate_hz=args.rate, max_rate=args.max_rate,
                virtual_flip=args.virtual_flip, bridge=args.bridge,
                handoff_dist=args.handoff_dist,
                handoff_speed_min=args.handoff_speed_min,
                checkpoint=str(args.checkpoint), max_seconds=args.max_seconds,
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
    finally:
        stop.set()
        vthread.join(timeout=6.0)

    # -- summary --
    print("\n==== fly_rl summary ====")
    for r in results:
        print(f"  flight {r['flight']}: {r['final_state']:<10} gates={r['gate_index']}"
              f"  {r.get('session','')}")
    if client.collisions:
        ids = sorted({c["id"] for c in client.collisions})
        print(f"  collisions (all flights): {len(client.collisions)} "
              f"(ids {ids}; 1001=gate 1002=env)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
