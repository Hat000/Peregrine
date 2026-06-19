"""rl/contact_true_eval.py — inc8 Phase-0(b) metric instrument.

Replaces the legacy point-mass L-inf<0.75 pass proxy with CONTACT-TRUE scoring
(body-radius DR + 0.30 m frame extrusion) and provides:
  * per-gate margin distributions with gate-3 isolated
  * seed-stability score S_stable (fraction of start seeds achieving sr >= 0.90)
  * gate-3 D-offset sensitivity probe (±1.5 m in NED Down)

Geometry: reuses slab_frame_hit_np and _HALF_OPEN/_HALF_OUTER constants imported
from offline_rollout verbatim -- no re-derivation of the geometry. The episode
rollout logic mirrors offline_rollout.main()'s loop.

ALL rollouts run with --plant mixer (default) matching inc6+ checkpoints and the
FRAME-AUDIT 2026-06-12 fully measured plant. Pass --plant map for legacy evals.

Cross-validation: _score_gate(body_radius=0, frame_depth=0) reproduces the legacy
gate_event() output exactly (negative-control test in tests/test_contact_true_eval.py).

Usage (from repo root, .venv active):
  .venv\\Scripts\\python.exe rl/contact_true_eval.py ^
      --ckpt rl/checkpoints/stage1_inc7_actor.pth
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_RL = Path(__file__).resolve().parent
_SRC = _RL.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_RL))

from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, PlantParams, PlantState,
                            SUPER_RATE_S_MEASURED, QUAD_DRAG_C2_MEASURED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED,
                            step as plant_step)
from fly_rl import (GateMap, N_GATES, _FLIP, _ACT_FLU_TO_FRD, _GATE_POS_ZUP, _R_W2G, _HOVER_THRUST,
                    _TRAIN_DT, _gate_rotmat_w2g, load_actor, make_gate_map, policy_step,
                    _ACT_MIN, _ACT_MAX)
from offline_rollout import (slab_frame_hit_np, _HALF_OPEN, _HALF_OUTER,
                              _RATE_SIGN_LIVE, obs_from_truth, _quat_from_rpy)
from estimator_emul import actor_obs_dim

# ---- inc8 active-perception LOOK-AT primitive (eval-faithfulness port, 2026-06-18) -----------
# Training (peregrine_racing_inc8.step) composes a camera->gate body-rate correction onto the CTBR
# action BEFORE the dynamics, gated to the approach band. The eval must apply the SAME composition or
# the 20-dim inc8 policy's EXECUTED control is wrong -> it dies at gate-0 (worker-disambiguated: inc7,
# no look-at, flies this harness 3/3; the omission is inc8-specific). We REUSE the pinned, tested
# primitive (rl/inc8_reward.lookat_correction + the baked r_body_from_camera 20deg matrix -- NOT a
# reimpl; pinned by tests/test_inc8_lookat). The correction is built in the FLU ACTION convention
# (exactly as training adds it to action[...,1:4]); the eval carries rates as FRD (policy_step output).
# 🚩 BODY-Y/YAW FIX (2026-06-18): policy_step encodes the policy's FLU rates via the LIVE wire map
# _ACT_FLU_TO_FRD=[1,-1,1] (the eval plant runs the live rate_sign _RATE_SIGN_LIVE=[1,1,1], NOT the
# trained-world [1,1,-1]). To recover the FLU action we INVERT THE SAME WIRE MAP, add band*dlook, clamp,
# then re-apply it -- NOT the training FLU<->FRD adapter _FLIP=[1,-1,-1]. _FLIP and _ACT_FLU_TO_FRD agree
# on roll/pitch but differ on YAW, so the old _FLIP reconstruction landed the look-at's yaw realized rate
# with a FLIPPED sign vs training (matched-state trace handoff/inc8-eval-pitch-2026-06-18/: pitch was
# already bit-identical; yaw diverged ~0.06-0.10 rad/s; _ACT_FLU_TO_FRD zeroes ALL axes). Both maps are
# involutory, so with dlook==0 (look-at OFF / out-of-band) rate_frd is unchanged -> inc7 byte-identical.
import torch as _torch  # noqa: E402
from inc8_reward import (lookat_correction as _lookat_correction,   # noqa: E402
                         r_body_from_camera as _r_body_from_camera, _FLIP_FRD_FLU as _FLIP_FRD_FLU)
_LOOKAT_R_BC = _r_body_from_camera(None, _torch.float64)             # baked R_body_from_camera (20deg)
_LOOKAT_FLIP = _torch.tensor(_FLIP_FRD_FLU, dtype=_torch.float64)    # FRD<->FLU body-rate flip (1,-1,-1)
# Trained look-at config for the rc1 recenter seeds (peregrine_inc8_recenter.sbatch COMMON/RC_OVERRIDES):
LOOKAT_G_YAW_DEFAULT = -3.0     # empirical S0 yaw sign
LOOKAT_G_PITCH_DEFAULT = 3.0    # empirical S2 pitch sign (opposite)
LOOKAT_R_LO_DEFAULT = 8.0
LOOKAT_R_HI_DEFAULT = 30.0


def _lookat_dlook_flu(t_cam_np: np.ndarray, g_yaw: float, g_pitch: float) -> np.ndarray:
    """FLU action-convention body-rate correction (rad/s) for one (3,) camera-frame gate vector.
    Thin numpy wrapper over the pinned torch lookat_correction (warmup factor wf == 1 post-training)."""
    t = _torch.as_tensor(np.asarray(t_cam_np, dtype=np.float64), dtype=_torch.float64)
    d = _lookat_correction(t, float(g_yaw), float(g_pitch), _LOOKAT_R_BC, _LOOKAT_FLIP)
    return d.cpu().numpy()

# ---------------------------------------------------------------------------
# Contact-true geometry constants (verbatim from test_contact_geometry.py inc7 doctrine)
# ---------------------------------------------------------------------------
BODY_RADIUS_NOM = 0.33       # nominal mid-point of [0.28, 0.38] DR range
FRAME_DEPTH_NOM = 0.30       # inc7 contact-true frame extrusion
PASS_BAND_NOM = _HALF_OPEN - BODY_RADIUS_NOM   # 0.42 m

# D-axis sensitivity probe (matches live finding A: gate-3 D-offset ~1.46 m)
G3_D_PROBE_M = 1.5           # probe at ±1.5 m in NED Down = ∓1.5 in Z-up z

# S_stable VQ1 threshold
SR_STABLE_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateCrossing:
    gate: int
    verdict: str   # 'pass' | 'collision' | 'miss'
    linf: float    # in-plane L-inf at plane crossing; NaN for slab-only collisions
    t: float       # sim time at event

    @property
    def margin(self) -> float:
        """Contact-true margin = PASS_BAND_NOM - linf; positive = safe."""
        return PASS_BAND_NOM - self.linf


@dataclass
class EpisodeResult:
    outcome: str   # 'FINISHED' | 'COLLISION' | 'MISS' | 'OOB' | 'TIMEOUT'
    crossings: list[GateCrossing] = field(default_factory=list)
    finish_t: float = float('nan')
    start_label: str = ''
    emulator: object | None = None   # the EstimatorEmulator (estim-emul runs only); carries the
                                      # per-step instrumentation (v*, gate-4 fix-rate, terminal lock)

    @property
    def success(self) -> bool:
        return self.outcome == 'FINISHED'

    def passed_linfs(self, gate: int) -> list[float]:
        return [c.linf for c in self.crossings
                if c.gate == gate and c.verdict == 'pass']


# ---------------------------------------------------------------------------
# Core geometry: _score_gate
# ---------------------------------------------------------------------------

def _score_gate(
    prev_ned: np.ndarray,
    cur_ned: np.ndarray,
    gate: int,
    body_radius: float,
    frame_depth: float,
    gate_pos_zup: np.ndarray,
    gate_yaw: np.ndarray | None = None,
) -> tuple[str | None, float]:
    """Returns (verdict, linf) using the contact-true geometry.

    Reuses slab_frame_hit_np (imported from offline_rollout) and _HALF_OPEN /
    _HALF_OUTER constants verbatim. The gate_pos_zup argument enables overriding
    the gate position for the map-offset sensitivity probe (score only; trajectory
    is generated with the nominal map).

    With body_radius=0 and frame_depth=0 this reproduces the legacy gate_event()
    output exactly (verified by test_contact_true_eval.test_legacy_parity).

    ``gate_yaw`` (P4-C05): None = the hardcoded yaw=π gate frame (_R_W2G, bit-exact VQ1,
    correct only because every VQ1 gate yaw is π); pass per-gate yaws to score a non-π /
    VQ2 course correctly (so the selection metric does not silently mis-score bent gates)."""
    w2g = _R_W2G if gate_yaw is None else _gate_rotmat_w2g(float(gate_yaw[gate]))
    half_in = _HALF_OPEN - body_radius
    half_out = _HALF_OUTER + body_radius
    prev_rel = w2g @ (prev_ned * _FLIP - gate_pos_zup[gate])
    cur_rel = w2g @ (cur_ned * _FLIP - gate_pos_zup[gate])

    # Volumetric slab contact (pre-plane and steep-approach strikes, inc7 doctrine)
    if frame_depth > 0.0 and slab_frame_hit_np(prev_rel, cur_rel, half_in, half_out,
                                               frame_depth):
        return "collision", max(abs(cur_rel[1]), abs(cur_rel[2]))

    # Plane crossing
    fwd = prev_rel[0] < 0.0 and cur_rel[0] >= 0.0
    bwd = prev_rel[0] > 0.0 and cur_rel[0] <= 0.0
    if not (fwd or bwd):
        return None, float('nan')

    f = -prev_rel[0] / ((cur_rel[0] - prev_rel[0]) or 1e-9)
    y = prev_rel[1] + f * (cur_rel[1] - prev_rel[1])
    z = prev_rel[2] + f * (cur_rel[2] - prev_rel[2])
    linf = max(abs(y), abs(z))

    if fwd and linf < half_in:
        return "pass", linf
    if half_in <= linf <= half_out:
        return "collision", linf
    return ("miss" if fwd else None), linf


# ---------------------------------------------------------------------------
# Plant construction
# ---------------------------------------------------------------------------

def _build_plant_params(plant: str) -> PlantParams:
    _aero = dict(rate_sign=_RATE_SIGN_LIVE.copy(),
                 super_rate_s=SUPER_RATE_S_MEASURED,
                 alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                 linear_drag=0.0,
                 quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                 coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                 coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
    if plant == "mixer":
        return PlantParams(**_aero,
                           mixer_idle=MIXER_IDLE_MEASURED,
                           mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                           mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
                           mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)
    if plant == "aero":
        return PlantParams(**_aero)
    if plant == "map":
        return PlantParams(rate_sign=_RATE_SIGN_LIVE.copy(),
                           super_rate_s=SUPER_RATE_S_MEASURED,
                           alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)
    # flat legacy
    return PlantParams(rate_sign=_RATE_SIGN_LIVE.copy())


# ---------------------------------------------------------------------------
# Start state construction
# ---------------------------------------------------------------------------

def _build_start(kind: str, gate: int = 0) -> tuple[PlantState, int, bool]:
    """Returns (start_state, target_gate, virtual_flip).

    virtual_flip mirrors offline_rollout conventions:
      simstart / racestart: True  (nose-first spawn needs the π body-z flip)
      trainreset: False           (already in the training-native tail-first frame)
    """
    if kind == "simstart":
        pos_ned = np.array([0.0, 0.0, 0.02])
        quat = _quat_from_rpy(0.0, np.radians(-17.8), np.radians(-179.9))
        return PlantState(pos=pos_ned, vel=np.zeros(3), quat=quat,
                          omega=np.zeros(3), thrust=np.float64(_HOVER_THRUST)), 0, True
    if kind == "racestart":
        pos_ned = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        return PlantState(pos=pos_ned, vel=np.zeros(3), quat=quat,
                          omega=np.zeros(3), thrust=np.float64(_HOVER_THRUST)), 0, True
    if kind == "trainreset":
        pos_ned = (_GATE_POS_ZUP[gate] + np.array([1.0, 0.0, 0.0])) * _FLIP
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return PlantState(pos=pos_ned, vel=np.zeros(3), quat=quat,
                          omega=np.zeros(3), thrust=np.float64(_HOVER_THRUST)), gate, False
    raise ValueError(f"unknown start kind: {kind!r}")


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    actor,
    start_state: PlantState,
    target_gate: int,
    virtual_flip: bool,
    params: PlantParams,
    body_radius: float = BODY_RADIUS_NOM,
    frame_depth: float = FRAME_DEPTH_NOM,
    gate_pos_zup: np.ndarray | None = None,
    gate_yaw: np.ndarray | None = None,
    max_time: float = 40.0,
    dt: float = _TRAIN_DT,
    start_label: str = '',
    record_pos: bool = False,
    estim_emul: bool = False,
    emul_config=None,
    obs_dim: int = 17,
    emul_seed: int = 0,
    emul_passive: bool = False,
    lookat: bool | None = None,
    lookat_g_yaw: float = LOOKAT_G_YAW_DEFAULT,
    lookat_g_pitch: float = LOOKAT_G_PITCH_DEFAULT,
    lookat_r_lo: float = LOOKAT_R_LO_DEFAULT,
    lookat_r_hi: float = LOOKAT_R_HI_DEFAULT,
) -> tuple[EpisodeResult, np.ndarray | None]:
    """Run one closed-loop episode and score with contact-true geometry.

    Returns (EpisodeResult, pos_trajectory_ned) where pos_trajectory is (T+1, 3)
    NED if record_pos=True (includes the start position), else None.

    Physics and observation building replicate offline_rollout.main()'s loop.
    gate_pos_zup overrides gate positions for scoring; when None uses _GATE_POS_ZUP.

    ``estim_emul`` (default OFF = the OPTIMISTIC obs_from_truth control): when ON, the policy obs is
    sourced from an ACTUAL LinearKF driven by the calibrated fix_surrogate (rl/estimator_emul.py),
    so camera-pointing -> fix density -> KF accuracy -> the obs the policy is scored on. The KF
    replaces ONLY pos/vel (truth attitude/rates kept) -- the deploy seam. ``obs_dim`` (17 inc7 / 20
    inc8) selects the obs width. The emulator (with its per-step instrumentation: v*, gate-4 fix-rate,
    terminal gate-lock, gate-frame error series) is attached to the returned EpisodeResult.emulator.

    ``emul_passive`` (PASSIVE-OBSERVER instrument, inc8): when True AND estim_emul is ON, the
    estimator still runs alongside (recording fix-rate / KF error / gate-lock on every step) but the
    POLICY continues to fly on obs_from_truth. This measures the estimator machinery under a
    course-completing policy (how L3 measured inc7's fix-rate), without coupling the noisy obs back
    into the control loop. result.emulator holds the full instrumentation after the episode.

    ``gate_yaw`` (P4-C05): None = the hardcoded all-π VQ1 course (obs gate_map=None +
    yaw=π scoring, bit-exact). Pass per-gate yaws for a non-π / VQ2 course: the obs is then
    built through a yaw-aware gate_map AND the geometry is scored per-gate, so the selection
    metric and the policy both see the correct gate frame."""
    if gate_pos_zup is None:
        gate_pos_zup = _GATE_POS_ZUP

    # Obs gate map: None on VQ1 (bit-exact); yaw-aware (positions + yaws) on a non-π course so
    # the policy sees the same gate frame the metric scores against.
    gate_map = None if gate_yaw is None else make_gate_map(gate_pos_zup, gate_yaw)

    # OOB box (exact training box from peregrine_racing._update_boxes)
    pts = np.vstack([gate_pos_zup, [0.0, 0.0, -0.02]])
    oob_lo = pts.min(0) - np.array([15, 15, 12])
    oob_hi = pts.max(0) + np.array([15, 15, 12])

    st = start_state
    gate = target_gate
    last_normed = 0.0
    n_steps = int(round(max_time / dt))
    crossings: list[GateCrossing] = []
    outcome = "TIMEOUT"
    finish_t = float('nan')
    pos_log = [start_state.pos.copy()] if record_pos else None

    # ESTIMATOR-EMULATION obs path (opt-in). Run an actual LinearKF off the calibrated fix_surrogate
    # so the policy is scored on the obs the estimator delivers, not perfect pose. The emulator owns
    # the gate frames (Z-up positions threaded through), the per-episode DR, and the staleness clock.
    emulator = None
    st_prev_obs = start_state
    if estim_emul:
        from estimator_emul import EmulConfig, EstimatorEmulator
        emul_rng = np.random.default_rng(emul_seed)
        emulator = EstimatorEmulator(emul_config or EmulConfig(),
                                     gate_pos_zup=gate_pos_zup, gate_yaw=gate_yaw)
        emulator.reset(start_state, gate, emul_rng)

    # inc8-only look-at gate: compose the camera-pointing correction iff flying the ACTIVE estimator-
    # emul obs at the 20-dim inc8 width. The passive-observer (truth-obs) path and the 17-dim inc7
    # contract get NO look-at -> inc7 stays byte-identical (it already flies this harness).
    apply_lookat = (lookat if lookat is not None
                    else (estim_emul and (not emul_passive) and obs_dim >= 20))

    for k in range(n_steps):
        if estim_emul:
            if k > 0:
                emulator.step(st_prev_obs, st, gate, dt, emul_rng)
            if emul_passive:
                # PASSIVE OBSERVER: policy flies on truth obs (completes the validated course);
                # the emulator records the fix stream + KF error WITHOUT closing the loop.
                # Measures estimator machinery under a course-completing policy (how L3
                # measured inc7's fix-rate). result.emulator holds the instrumentation.
                obs = obs_from_truth(st, gate, last_normed, virtual_flip, gate_map=gate_map)
            else:
                obs = emulator.obs(st, gate, last_normed, virtual_flip, gate_map, obs_dim)
            st_prev_obs = st
        else:
            obs = obs_from_truth(st, gate, last_normed, virtual_flip, gate_map=gate_map)
        rate_frd, _coll, last_normed = policy_step(actor, obs, 0.0, virtual_flip)
        # ---- inc8 LOOK-AT composition (faithful to peregrine_racing_inc8.step:233-246) ----
        # Add the camera->gate body-rate correction to the realised CTBR command, gated to the
        # approach band (range in [r_lo,r_hi] & gate in front, t_cam_z>0). Composed in the FLU action
        # convention training uses (action[...,1:4]); rate_frd is real FRD, so map FRD->FLU, add, clamp
        # to the action bounds, map back. wf==1 post-training (warmup complete). No geom (k==0 / no
        # prior) -> pass through unchanged, exactly as training.
        if apply_lookat and emulator is not None and emulator._last_geom is not None:
            g0 = emulator._last_geom
            if (lookat_r_lo <= g0.range_m <= lookat_r_hi) and (float(g0.t_cam[2]) > 0.0):
                dlook = _lookat_dlook_flu(g0.t_cam, lookat_g_yaw, lookat_g_pitch)   # FLU action rate
                flu = rate_frd * _ACT_FLU_TO_FRD                    # FRD -> FLU (invert policy_step wire map)
                flu = np.clip(flu + dlook, _ACT_MIN[1:4], _ACT_MAX[1:4])
                rate_frd = flu * _ACT_FLU_TO_FRD                    # FLU -> FRD (re-apply wire map)
        collective = last_normed * _HOVER_THRUST   # un-clipped, exactly training
        action = np.concatenate([rate_frd, [collective]])

        prev_pos = st.pos.copy()
        st = plant_step(st, action, dt, params)
        t = (k + 1) * dt

        if pos_log is not None:
            pos_log.append(st.pos.copy())

        # Non-target gate frame-strikes take precedence (mirrors offline_rollout)
        collision_gate = None
        for g in range(N_GATES):
            if g == gate:
                continue
            v, linf = _score_gate(prev_pos, st.pos, g, body_radius, frame_depth, gate_pos_zup,
                                  gate_yaw)
            if v == "collision":
                crossings.append(GateCrossing(gate=g, verdict="collision", linf=linf, t=t))
                collision_gate = g
                break

        if collision_gate is not None:
            outcome = "COLLISION"
            break

        # Target gate
        v, linf = _score_gate(prev_pos, st.pos, gate, body_radius, frame_depth, gate_pos_zup,
                              gate_yaw)
        if v == "pass":
            crossings.append(GateCrossing(gate=gate, verdict="pass", linf=linf, t=t))
            if gate == N_GATES - 1:
                outcome = "FINISHED"
                finish_t = t
                break
            gate += 1
            continue
        if v == "collision":
            crossings.append(GateCrossing(gate=gate, verdict="collision", linf=linf, t=t))
            outcome = "COLLISION"
            break
        if v == "miss":
            crossings.append(GateCrossing(gate=gate, verdict="miss", linf=linf, t=t))
            outcome = "MISS"
            break

        # OOB check (Z-up frame)
        if np.any(st.pos * _FLIP < oob_lo) or np.any(st.pos * _FLIP > oob_hi):
            outcome = "OOB"
            break

    result = EpisodeResult(
        outcome=outcome, crossings=crossings, finish_t=finish_t, start_label=start_label,
        emulator=emulator,
    )
    pos_arr = np.array(pos_log) if pos_log else None
    return result, pos_arr


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def per_gate_margin_stats(results: list[EpisodeResult],
                          start_filter: str | None = None,
                          pass_band: float = PASS_BAND_NOM) -> dict:
    """Compute per-gate contact-true margin distributions across a list of episodes.

    Returns a dict gate_id -> {'linfs': list, 'margins': list, 'n_pass': int,
    'n_collision': int, 'p10': float, 'median': float, 'min': float} for each gate.
    Gates with no passes report NaN statistics.

    start_filter: if set, only episodes whose start_label STARTS WITH this prefix are
    included (e.g. 'simstart' for the competition-representative full-course rollout,
    'trainreset' for the synthetic at-rest 1 m-back resets). None pools everything.

    pass_band: contact-true pass band = _HALF_OPEN - body_radius. MUST be passed to
    match the radius the trajectories were scored at, otherwise margins are computed
    against the nominal r=0.33 band regardless of --body-radius (the recorded linf is
    radius-independent geometry; only the pass band shifts with radius).
    """
    from collections import defaultdict
    if start_filter is not None:
        results = [r for r in results if r.start_label.startswith(start_filter)]
    linfs_by_gate: dict[int, list[float]] = defaultdict(list)
    coll_by_gate: dict[int, int] = defaultdict(int)
    for r in results:
        for c in r.crossings:
            if c.verdict == "pass" and not np.isnan(c.linf):
                linfs_by_gate[c.gate].append(c.linf)
            elif c.verdict == "collision":
                coll_by_gate[c.gate] += 1

    stats: dict[int, dict] = {}
    for g in range(N_GATES):
        linfs = linfs_by_gate[g]
        if linfs:
            arr = np.array(linfs)
            margins = pass_band - arr
            stats[g] = {
                "linfs": linfs,
                "margins": margins.tolist(),
                "n_pass": len(linfs),
                "n_collision": coll_by_gate[g],
                "linf_p10": float(np.percentile(arr, 10)),
                "linf_median": float(np.median(arr)),
                "linf_max": float(arr.max()),
                "margin_p10": float(np.percentile(margins, 10)),
                "margin_median": float(np.median(margins)),
                "margin_min": float(margins.min()),
            }
        else:
            stats[g] = {
                "linfs": [], "margins": [],
                "n_pass": 0, "n_collision": coll_by_gate[g],
                "linf_p10": float('nan'), "linf_median": float('nan'), "linf_max": float('nan'),
                "margin_p10": float('nan'), "margin_median": float('nan'), "margin_min": float('nan'),
            }
    return stats


def compute_s_stable(
    results_by_seed: dict[str, list[EpisodeResult]],
    threshold: float = SR_STABLE_THRESHOLD,
) -> tuple[float, dict[str, float]]:
    """Compute S_stable = fraction of seeds achieving sr >= threshold.

    results_by_seed: {seed_label -> [EpisodeResult, ...]}. Each list may contain
    multiple episodes for that seed (sr = fraction successful). Returns
    (s_stable, {seed_label: sr}).
    """
    per_seed_sr: dict[str, float] = {}
    for label, eps in results_by_seed.items():
        if eps:
            per_seed_sr[label] = float(np.mean([e.success for e in eps]))
        else:
            per_seed_sr[label] = 0.0
    n_stable = sum(1 for sr in per_seed_sr.values() if sr >= threshold)
    s_stable = n_stable / max(len(per_seed_sr), 1)
    return s_stable, per_seed_sr


# ---------------------------------------------------------------------------
# Map-offset sensitivity probe
# ---------------------------------------------------------------------------

def gate_d_offset_probe(
    pos_traj: np.ndarray,
    delta_d_m: float,
    gate_id: int = 3,
    body_radius: float = BODY_RADIUS_NOM,
    frame_depth: float = FRAME_DEPTH_NOM,
    gate_yaw: np.ndarray | None = None,
) -> dict:
    """Re-score a pre-computed trajectory with gate `gate_id` shifted by delta_d_m in NED Down.

    Parametric generalization of the original gate-3-only probe: tells us whether ANY
    gate's contact-true margin is verdict-FRAGILE (flips under a plausible map mis-cal),
    not just gate-3. Use it on gates 4 and 5 to test the post-gate-3 binding hypothesis.

    In NED: D (down) = +z_ned. In Z-up: z_zup = -z_ned. So shifting gate `gate_id` down by
    delta_d_m in NED = shifting _GATE_POS_ZUP[gate_id][2] by -delta_d_m.

    pos_traj: (T+1, 3) NED positions from a completed rollout.
    delta_d_m: positive = gate physically lower (drone passes above), negative = higher.

    Returns dict with the probed gate's pass/collision counts and margin stats under the
    offset (keys: delta_d_m, gate_id, outcome, n_gate_pass, n_gate_collision, gate_linf,
    gate_margin, gates_passed).
    """
    gate_pos = _GATE_POS_ZUP.copy()
    gate_pos[gate_id, 2] -= delta_d_m   # D down = Z-up z decreases

    crossings: list[GateCrossing] = []
    gate = 0
    outcome = "TIMEOUT"

    for k in range(1, len(pos_traj)):
        prev_ned = pos_traj[k - 1]
        cur_ned = pos_traj[k]
        t = k * _TRAIN_DT

        collision_g = None
        for g in range(N_GATES):
            if g == gate:
                continue
            v, linf = _score_gate(prev_ned, cur_ned, g, body_radius, frame_depth, gate_pos,
                                  gate_yaw)
            if v == "collision":
                crossings.append(GateCrossing(gate=g, verdict="collision", linf=linf, t=t))
                collision_g = g
                break

        if collision_g is not None:
            outcome = "COLLISION"
            break

        v, linf = _score_gate(prev_ned, cur_ned, gate, body_radius, frame_depth, gate_pos,
                              gate_yaw)
        if v == "pass":
            crossings.append(GateCrossing(gate=gate, verdict="pass", linf=linf, t=t))
            if gate == N_GATES - 1:
                outcome = "FINISHED"
                break
            gate += 1
        elif v == "collision":
            crossings.append(GateCrossing(gate=gate, verdict="collision", linf=linf, t=t))
            outcome = "COLLISION"
            break
        elif v == "miss":
            crossings.append(GateCrossing(gate=gate, verdict="miss", linf=linf, t=t))
            outcome = "MISS"
            break

    passes = [c for c in crossings if c.gate == gate_id and c.verdict == "pass"]
    colls = [c for c in crossings if c.gate == gate_id and c.verdict == "collision"]
    g_linf = passes[0].linf if passes else float('nan')
    g_margin = passes[0].margin if passes else float('nan')

    return {
        "delta_d_m": delta_d_m,
        "gate_id": gate_id,
        "outcome": outcome,
        "n_gate_pass": len(passes),
        "n_gate_collision": len(colls),
        "gate_linf": g_linf,
        "gate_margin": g_margin,
        "gates_passed": sum(1 for c in crossings if c.verdict == "pass"),
    }


def gate3_d_offset_probe(
    pos_traj: np.ndarray,
    delta_d_m: float,
    body_radius: float = BODY_RADIUS_NOM,
    frame_depth: float = FRAME_DEPTH_NOM,
) -> dict:
    """Gate-3-specific backward-compatible wrapper over gate_d_offset_probe.

    Preserves the original g3-prefixed key schema relied upon by existing callers
    and tests. New code should call gate_d_offset_probe(..., gate_id=...) directly.
    """
    r = gate_d_offset_probe(pos_traj, delta_d_m, gate_id=3,
                            body_radius=body_radius, frame_depth=frame_depth)
    return {
        "delta_d_m": r["delta_d_m"],
        "outcome": r["outcome"],
        "n_g3_pass": r["n_gate_pass"],
        "n_g3_collision": r["n_gate_collision"],
        "g3_linf": r["gate_linf"],
        "g3_margin": r["gate_margin"],
        "gates_passed": r["gates_passed"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="path to actor.pth (inc7: stage1_inc7_actor.pth)")
    ap.add_argument("--plant", default="mixer",
                    choices=["map", "flat", "aero", "mixer"],
                    help="plant model (mixer = fully measured, default for inc6+)")
    ap.add_argument("--body-radius", type=float, default=BODY_RADIUS_NOM,
                    help=f"contact-true body radius (default {BODY_RADIUS_NOM} m)")
    ap.add_argument("--frame-depth", type=float, default=FRAME_DEPTH_NOM,
                    help=f"frame slab depth (default {FRAME_DEPTH_NOM} m)")
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--sr-threshold", type=float, default=SR_STABLE_THRESHOLD)
    ap.add_argument("--g3-probe-range", type=float, default=G3_D_PROBE_M,
                    help="gate-3 D-offset probe range in meters")
    ap.add_argument("--estim-emul", action="store_true",
                    help="score on ESTIMATOR-EMULATED obs (LinearKF + fix_surrogate) instead of "
                         "perfect pose -- the inc8 selection path; default OFF = optimistic control")
    ap.add_argument("--obs-dim", type=int, default=0,
                    help="policy obs width (17 inc7 / 20 inc8); 0 = infer from the actor")
    ap.add_argument("--emul-seed", type=int, default=0,
                    help="base RNG seed for the per-episode estimator-emulation DR draws")
    args = ap.parse_args()

    actor = load_actor(args.ckpt)
    params = _build_plant_params(args.plant)

    # Infer the policy obs width (inc7=17, inc8=20): --obs-dim pin overrides, else actor_obs_dim.
    obs_dim = args.obs_dim if args.obs_dim > 0 else actor_obs_dim(actor)
    if args.estim_emul:
        print(f"[ESTIM-EMUL ON] policy obs from emulated KF pose (no-render fix_surrogate); "
              f"obs_dim={obs_dim} (17 inc7 / 20 inc8). Scoring is unchanged (TRUE trajectory).")
        if obs_dim not in (17, 20):
            print(f"  !! unexpected obs_dim={obs_dim}; emulator emits 17 or 20 -- check the checkpoint.")

    print(f"\n{'='*80}")
    print(f"CONTACT-TRUE EVAL  ckpt={Path(args.ckpt).name}  plant={args.plant}")
    print(f"body_radius={args.body_radius}  frame_depth={args.frame_depth}")
    print(f"pass_band={_HALF_OPEN - args.body_radius:.3f} m  "
          f"(legacy L-inf<{_HALF_OPEN} -> contact-true L-inf<{_HALF_OPEN - args.body_radius:.3f})")
    print(f"{'='*80}\n")

    # --- Seed set: simstart + trainreset gates 0-5 ---
    seeds = [("simstart", "simstart", 0)]
    for g in range(N_GATES):
        seeds.append((f"trainreset_g{g}", "trainreset", g))

    results_by_seed: dict[str, list[EpisodeResult]] = {}
    all_results: list[EpisodeResult] = []
    simstart_pos_traj: np.ndarray | None = None
    emul_metrics: dict[str, dict] = {}   # per-label estimator-emulation selection numbers (inc8)

    print("Running seed episodes...")
    for si, (label, kind, gate_idx) in enumerate(seeds):
        st, tgt, vflip = _build_start(kind, gate_idx)
        record = (kind == "simstart")
        result, pos_traj = run_episode(
            actor, st, tgt, vflip, params,
            body_radius=args.body_radius, frame_depth=args.frame_depth,
            max_time=args.max_time, start_label=label, record_pos=record,
            estim_emul=args.estim_emul, obs_dim=obs_dim, emul_seed=args.emul_seed + si,
        )
        results_by_seed[label] = [result]
        all_results.append(result)
        if record:
            simstart_pos_traj = pos_traj
        if result.emulator is not None:
            emu = result.emulator
            ip = emu.gate4_inplane_error_series()
            emul_metrics[label] = {
                "v_star": emu.v_star(),
                "g4_fix_rate": emu.gate4_band_fix_rate(),
                "g4_lock": emu.terminal_gate_lock_frac(),
                "g4_ip_p90": float(np.percentile(ip, 90)) if ip.size else float("nan"),
                "g4_ip_p99": float(np.percentile(ip, 99)) if ip.size else float("nan"),
                "n_fix": int(sum(emu.trace.fix_accepted)),
                "n_g4_band": int(ip.size),
            }
        status = "FINISHED" if result.success else result.outcome
        g3_linfs = result.passed_linfs(3)
        g3_str = f"  g3_linf={g3_linfs[0]:.3f}" if g3_linfs else ""
        emul_str = ""
        if label in emul_metrics:
            m = emul_metrics[label]
            emul_str = (f"  [emul v*={m['v_star']:.1f} fixrate_g4={m['g4_fix_rate']:.3f} "
                        f"lock_g4={m['g4_lock']:.2f} ip_p90={m['g4_ip_p90']:.3f}]")
        print(f"  {label:20s}  {status}{g3_str}{emul_str}")

    # --- S_stable ---
    s_stable, per_seed_sr = compute_s_stable(results_by_seed, threshold=args.sr_threshold)
    n_seeds = len(results_by_seed)
    n_stable = sum(1 for sr in per_seed_sr.values() if sr >= args.sr_threshold)
    print(f"\n[S_STABLE] {s_stable:.3f}  ({n_stable}/{n_seeds} seeds sr>={args.sr_threshold})")
    if s_stable < 2.0 / 3.0:
        print(f"  !! NARROW BASIN: S_stable < 2/3  (inc7 training was 1/3-viable; "
              f"see memory inc7 convergence)")

    # --- Per-gate margin distributions (pooled + disaggregated by start type) ---
    def _print_margin_table(stats: dict, header: str) -> None:
        print(f"\n[PER-GATE MARGINS — {header}]  pass_band={_HALF_OPEN - args.body_radius:.3f} m  "
              f"(L-inf < this = contact-true PASS)")
        print(f"  {'gate':>4}  {'n_pass':>6}  {'n_coll':>6}  "
              f"{'linf_p10':>8}  {'linf_med':>8}  {'linf_max':>8}  "
              f"{'marg_p10':>8}  {'marg_med':>8}  {'marg_min':>8}")
        for g in range(N_GATES):
            s = stats[g]
            flag = "GATE-3 (historically binding)" if g == 3 else ""
            if s["n_pass"] > 0:
                print(f"  {g:>4}  {s['n_pass']:>6}  {s['n_collision']:>6}  "
                      f"{s['linf_p10']:>8.3f}  {s['linf_median']:>8.3f}  {s['linf_max']:>8.3f}  "
                      f"{s['margin_p10']:>8.3f}  {s['margin_median']:>8.3f}  "
                      f"{s['margin_min']:>8.3f}  {flag.strip()}")
            else:
                print(f"  {g:>4}  {s['n_pass']:>6}  {s['n_collision']:>6}  "
                      f"{'N/A':>8}  {'N/A':>8}  {'N/A':>8}  "
                      f"{'N/A':>8}  {'N/A':>8}  {'N/A':>8}{flag}")

    _band = _HALF_OPEN - args.body_radius
    _print_margin_table(per_gate_margin_stats(all_results, pass_band=_band),
                        "POOLED (all starts)")
    _print_margin_table(per_gate_margin_stats(all_results, start_filter="simstart",
                                              pass_band=_band),
                        "SIMSTART ONLY (competition-representative full-course)")
    _print_margin_table(per_gate_margin_stats(all_results, start_filter="trainreset",
                                              pass_band=_band),
                        "TRAINRESET ONLY (synthetic at-rest 1 m-back)")
    print("\n  NOTE: only SIMSTART carries a realistic high-speed post-gate-3 approach; "
          "it alone answers the gate-4/5 binding question.")

    # --- Multi-gate D-offset sensitivity probe (gates 3, 4, 5) ---
    if simstart_pos_traj is not None and len(simstart_pos_traj) > 1:
        print(f"\n[MULTI-GATE D-OFFSET SENSITIVITY]  "
              f"simstart trajectory re-scored with each gate shifted +/-{args.g3_probe_range} m D")
        print(f"  (D+ = gate physically LOWER in NED; tests whether a gate's margin is "
              f"verdict-FRAGILE under a plausible map mis-cal — not that the map IS off)")
        for gate_id in (3, 4, 5):
            print(f"  -- gate {gate_id} --")
            for delta in (0.0, -args.g3_probe_range, +args.g3_probe_range):
                res = gate_d_offset_probe(
                    simstart_pos_traj, delta, gate_id=gate_id,
                    body_radius=args.body_radius, frame_depth=args.frame_depth
                )
                gm = f"{res['gate_margin']:+.3f}" if not np.isnan(res['gate_margin']) else "N/A"
                gl = f"{res['gate_linf']:.3f}" if not np.isnan(res['gate_linf']) else "N/A"
                label = "nominal" if delta == 0.0 else f"D{delta:+.1f}m"
                print(f"    delta_D={delta:+.1f} m  ({label:9s})  outcome={res['outcome']:10s}  "
                      f"g{gate_id}_pass={res['n_gate_pass']}  g{gate_id}_coll={res['n_gate_collision']}  "
                      f"g{gate_id}_linf={gl}  g{gate_id}_margin={gm}")
        print(f"  Interpretation: a gate whose verdict FLIPS pass->collision under +/-1.5 m is "
              f"map-confounded (offline metric not yet trustworthy for it); a gate that "
              f"stays PASS is robust to map mis-cal at this radius.")
    else:
        print("\n[MULTI-GATE D-OFFSET SENSITIVITY] skipped (simstart pos_traj unavailable)")

    # --- Estimator-emulation selection numbers (inc8) ---
    if emul_metrics:
        print(f"\n[ESTIMATOR-EMULATION — gate-4 selection numbers]  obs_dim={obs_dim}  "
              f"(central report r=0.30, select r=0.38; per MEMORY reporting discipline)")
        print(f"  {'seed':20s}  {'v*':>5}  {'fixrate_g4':>10}  {'lock_g4':>7}  "
              f"{'ip_p90':>7}  {'ip_p99':>7}  {'n_fix':>5}  {'n_band':>6}")
        for label, _k, _g in seeds:
            m = emul_metrics.get(label)
            if m is None:
                continue
            print(f"  {label:20s}  {m['v_star']:>5.1f}  {m['g4_fix_rate']:>10.3f}  "
                  f"{m['g4_lock']:>7.2f}  {m['g4_ip_p90']:>7.3f}  {m['g4_ip_p99']:>7.3f}  "
                  f"{m['n_fix']:>5d}  {m['n_g4_band']:>6d}")
        print("  NOTE: SIMSTART carries the realistic high-speed gate-4 approach; it alone answers "
              "the fix-rate / gate-lock / in-plane-error question. fixrate_g4/lock_g4 = TERMINAL "
              "camera pointing (the load-bearing inc8 lever).")

    # --- Machine-readable summary ---
    sim_res = results_by_seed["simstart"][0]
    sim_g3 = sim_res.passed_linfs(3)
    summary = (f"\nCONTACT_TRUE_SUMMARY "
               f"ckpt={Path(args.ckpt).name} plant={args.plant} "
               f"body_r={args.body_radius} frame_d={args.frame_depth} "
               f"s_stable={s_stable:.3f} n_stable={n_stable}/{n_seeds} "
               f"simstart={sim_res.outcome} "
               f"g3_linf_simstart={sim_g3[0] if sim_g3 else 'N/A'}")
    if args.estim_emul:
        sm = emul_metrics.get("simstart", {})
        summary += (f" estim_emul=ON obs_dim={obs_dim} "
                    f"v_star={sm.get('v_star', float('nan')):.2f} "
                    f"g4_fix_rate={sm.get('g4_fix_rate', float('nan')):.3f} "
                    f"g4_terminal_lock={sm.get('g4_lock', float('nan')):.3f} "
                    f"g4_inplane_p90={sm.get('g4_ip_p90', float('nan')):.3f} "
                    f"g4_inplane_p99={sm.get('g4_ip_p99', float('nan')):.3f}")
    else:
        summary += " estim_emul=OFF"
    print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
