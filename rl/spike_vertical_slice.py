"""Vertical-slice spike: close the full deploy loop on ONE synthetic-exact frame and FREEZE it.

frame(pixels) -> detector slice -> PnP GatePose -> case-C Navigator (RewindKF + gate-relative +L)
-> NavState -> obs[0:20] (build_obs / obs_from_zup ++ the d5 confidence triple) -> inc8 actor -> action.

This module builds ONLY the two seams that have no production home yet:
  (1) `slice_8kp_to_4`        -- the 8-keypoint -> inner-4 detector slice (detector.py asserts (4,2));
  (2) `deploy_confidence_triple` -- obs[17:20], reproducing rl/estimator_emul.confidence_channel EXACTLY
      (the ONLY builder the inc8 actor trained on) but driven by a REAL KF P-matrix + fix clock. The
      emul's sigma_inplane_hat = sqrt((P_E+P_D)/2) (the 0.5 AVERAGE, NOT NavState.nav_inplane_sigma's
      sqrt(P_E+P_D) sum -- a 1/sqrt(2) train/deploy gap if you used NavState's field).

Everything else (PnP, the +L lever, RewindKF, the relinnov chi2 gate, the R_y(pi) odo conjugation,
BORESIGHT -0.25, obs_from_zup) is the REAL stack, driven not reimplemented.

`run_slice_once(...)` returns the frozen 4-tuple (GatePose, NavState, obs20, action). `python -m
spike_vertical_slice` (from rl/) prints the goldens for the test to bake in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_RL = Path(__file__).resolve().parent
_SRC = _RL.parent / "src"
for _p in (str(_SRC), str(_RL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from racer.contracts import DroneState, Frame, Gate, GateObservation
from racer.frames import R_camera_from_body
from racer.kf_rewind import RewindKF
from racer.navigator import Navigator, NavigatorConfig
from racer.vision.detector import N_CORNERS, observations_from_keypoints
from racer.vision.gate_pose import estimate_gate_pose, project_gate_corners

# --- emul-faithful constants (rl/estimator_emul.py:70-71, FROZEN d5 contract) -------------
SIGMA_REF_M = 0.05
TAU_STALE_S = 0.10

_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])   # FRD specific force at hover (reaction = -g on +Z down)
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])        # level, scalar-first wxyz


# =====================================================================================
# SEAM (1): 8-keypoint -> inner-4 slice shim  (blocker#3: best.pt emits (n,8,2); PnP wants (4,2))
# =====================================================================================
def slice_8kp_to_4(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray):
    """YOLO-8kpt (n,8,2)+(n,8) -> inner-4 (n,4,2)+(n,4) for the PnP path.

    The 8-kpt VQ2 contract is inner-4 (0..3 = LL,LR,UR,UL, the PnP keypoints) THEN outer-4 (4..7);
    the inner-4 are already in gate_pose IPPE_SQUARE order (blender_gen/contract.py:89-106), so we
    SUBSET, never reorder. A 4-kpt model is already (n,4,*) and passes through unchanged.
    """
    xy = np.asarray(keypoints_xy, dtype=np.float64)
    conf = np.asarray(keypoints_conf, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1] < N_CORNERS or xy.shape[2] != 2:
        raise ValueError(f"keypoints_xy must be (n,>={N_CORNERS},2), got {xy.shape}")
    return xy[:, :N_CORNERS, :], conf[:, :N_CORNERS]


def observations_from_8kp_results(frame: Frame, keypoints_xy, keypoints_conf, det_scores, **kw):
    """Slice an 8-kpt detector's raw arrays to the inner-4, then run the canonical adapter."""
    xy4, conf4 = slice_8kp_to_4(keypoints_xy, keypoints_conf)
    return observations_from_keypoints(frame, xy4, conf4, det_scores, **kw)


# =====================================================================================
# SEAM (2): obs[17:20] deploy builder -- reproduce estimator_emul.confidence_channel EXACTLY,
# but from a REAL KF P-matrix + fix clock (the deploy path has NO producer for [17:20]).
# =====================================================================================
def deploy_confidence_triple(P_pos_3x3, R_world_gate, t_since_fix,
                             sigma_ref=SIGMA_REF_M, tau_stale=TAU_STALE_S):
    """[c_inplane, c_along, age_norm] from a 3x3 world-NED position covariance + a gate frame.

    BYTE-faithful to rl/estimator_emul.EstimatorEmulator.confidence_channel/_gate_frame_sigmas:
      P_gate = R_world_gate.T @ P_pos @ R_world_gate
      sigma_inplane = sqrt(0.5*(P_gate[0,0]+P_gate[1,1]))   # <-- the 0.5 AVERAGE is load-bearing
      sigma_along   = sqrt(P_gate[2,2])
      c = clip(sigma_ref/sigma, 0, 1)  (1.0 when sigma<=0 -- degenerate cov reads MAX confidence)
      age_norm = clip(t_since_fix/tau_stale, 0, 1)
    Returns float32 (3,) -- the dtype the actor saw (obs() casts the triple to float32).
    """
    Rwg = np.asarray(R_world_gate, dtype=np.float64)
    P = np.asarray(P_pos_3x3, dtype=np.float64)
    P_gate = Rwg.T @ P @ Rwg
    var_ip = 0.5 * (max(P_gate[0, 0], 0.0) + max(P_gate[1, 1], 0.0))
    var_al = max(P_gate[2, 2], 0.0)
    sig_ip, sig_al = float(np.sqrt(var_ip)), float(np.sqrt(var_al))
    c_ip = float(np.clip(sigma_ref / sig_ip, 0.0, 1.0)) if sig_ip > 0 else 1.0
    c_al = float(np.clip(sigma_ref / sig_al, 0.0, 1.0)) if sig_al > 0 else 1.0
    age = float(np.clip(t_since_fix / tau_stale, 0.0, 1.0))
    return np.array([c_ip, c_al, age], dtype=np.float32), (sig_ip, sig_al)


# =====================================================================================
# Synthetic-exact scenario -> sliced detector -> real Navigator -> NavState
# =====================================================================================
def _gate_facing_north(position_ned, gate_id=0, inner=1.5) -> Gate:
    """A gate whose through-axis (+Z) points world +N; the proven in-frame level-drone geometry
    (clone of tests/test_navigator_gate_relative.py:_gate_facing_north)."""
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position_ned, float),
                R_world_gate=R, inner_size_m=inner)


class _SlicedFakeDetector:
    """Projects the true gate from the true (level) drone pose, runs the 8->4 SLICE, and emits a
    GateObservation -- so the closed loop exercises slice_8kp_to_4 + the real PnP, not a bypass.

    The synthetic 8-kpt array is inner-4 (the real opening corners) ++ outer-4 (gate frame outer
    square); the slice must recover the inner-4 bit-exact. R_camera_world == R_camera_from_body()
    for a level drone (matches the test's _FakeDetector)."""

    def __init__(self, gate: Gate, drone_pos, inner=1.5, outer=2.72):
        self.gate = gate
        self.drone_pos = np.asarray(drone_pos, float)
        self.inner, self.outer = inner, outer

    def detect(self, frame: Frame):
        R_camera_world = R_camera_from_body()                       # level drone
        t_cam_gate = R_camera_world @ (self.gate.position_ned - self.drone_pos)
        R_cam_gate = R_camera_world @ self.gate.R_world_gate
        inner_px = project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m=self.inner)   # (4,2)
        outer_px = project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m=self.outer)   # (4,2)
        xy8 = np.concatenate([inner_px, outer_px], axis=0)[None, ...]     # (1,8,2)
        conf8 = np.ones((1, 8))
        scores = np.array([0.9])
        return observations_from_8kp_results(frame, xy8, conf8, scores)


def _ds(sim_time_ns):
    return DroneState(sim_time_ns=int(sim_time_ns), recv_monotonic_ns=0,
                      orientation_ned_wxyz=_LEVEL_Q.copy(), accel_body=_HOVER_ACCEL.copy(),
                      position_ned=None, velocity_ned=None, reset_counter=0)


def _frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8), recv_monotonic_ns=0)


def drive_navigator(gate_pos_ned=(9.0, 0.0, -2.5), drone_true_ned=(1.0, -0.5, -2.0),
                    n_ticks=80, tick_ns=10_000_000):
    """Set up the proven facing-north case-C scenario and drive the REAL Navigator to convergence.

    Returns (nav, gate, GatePose_at_convergence, final_NavState). GatePose is the PnP output on the
    sliced synthetic-exact pixels (what the navigator fuses)."""
    gate = _gate_facing_north(gate_pos_ned, gate_id=0)
    detector = _SlicedFakeDetector(gate, drone_true_ned)
    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                          use_rewind_kf=True, use_gate_relative=True)
    nav = Navigator(gates=[gate], detector=detector, config=cfg)

    nav.update(_ds(0), _frame(0, 0))                       # tick 0 seeds only
    ns = None
    for k in range(1, n_ticks + 1):
        ns = nav.update(_ds(k * tick_ns), _frame(k, k * tick_ns))

    # The GatePose the slice+PnP produced (recompute on the same corners for the golden):
    obs_list = detector.detect(_frame(n_ticks + 1, (n_ticks + 1) * tick_ns))
    gp = estimate_gate_pose(obs_list[0], compute_covariance=True, weighted_refine=True)
    return nav, gate, gp, ns


# =====================================================================================
# obs[0:20] build (deploy path) + actor step
# =====================================================================================
DEFAULT_ACTOR = str(_RL / "checkpoints" / "inc8_ws1_seed0_BEST_actor.pth")


def _gate_map_for(gate_pos_ned, target_gate=0):
    """A VQ1-shaped 6-gate map with target_gate's position set to the scenario gate (Z-up), all
    yaw=pi (the convention the inc8 policy trained on). Returns (gate_map, gate_zup)."""
    from fly_rl import _FLIP, _GATE_POS_ZUP, _GATE_YAW_ZUP, make_gate_map
    pos_zup = _GATE_POS_ZUP.copy()
    gate_zup = np.asarray(gate_pos_ned, float) * _FLIP                      # NED -> Z-up
    pos_zup[target_gate] = gate_zup
    return make_gate_map(pos_zup, _GATE_YAW_ZUP.copy()), gate_zup


def build_obs20(nav, gate, ns, *, target_gate=0, virtual_flip=True, last_normed=0.0):
    """Build the 20-dim deploy obs and ALSO the obs_from_zup (emul-builder) variant for parity.

    obs[0:17] via the DEPLOY seam estimator_state_for_obs(ds, nav)->build_obs (the actual deploy path).
    obs[17:20] via deploy_confidence_triple (emul-faithful, from the REAL KF P, floor ON).
    Returns (obs20, info) where info carries obs17_buildobs, obs17_fromzup, triple, sigmas."""
    from fly_rl import _FLIP, build_obs, obs_from_zup
    from racer.estimator_obs import estimator_state_for_obs

    gm, _ = _gate_map_for(gate.position_ned, target_gate)
    ds = _ds(ns.sim_time_ns)                                                # level identity wire state
    ds_est = estimator_state_for_obs(ds, ns)                               # pos/vel <- estimator
    obs17_buildobs = build_obs(ds_est, target_gate, last_normed,
                               virtual_flip=virtual_flip, gate_map=gm)      # (17,) float32

    # emul-builder variant (obs_from_zup directly, level identity -> R_zup = I):
    R_zup = np.eye(3)
    obs17_fromzup = obs_from_zup(ns.position_ned * _FLIP, ns.velocity_ned * _FLIP, R_zup,
                                 np.zeros(3), target_gate, last_normed,
                                 virtual_flip=virtual_flip, gate_map=gm)

    triple, sigmas = deploy_confidence_triple(nav.kf.P[:3, :3], gate.R_world_gate,
                                              ns.time_since_vision_update_s)
    obs20 = np.concatenate([obs17_buildobs, triple]).astype(np.float32)
    return obs20, dict(obs17_buildobs=obs17_buildobs, obs17_fromzup=obs17_fromzup,
                       triple=triple, sigmas=sigmas, gate_map=gm)


def actor_step(obs20, actor_path=DEFAULT_ACTOR, *, virtual_flip=True):
    """Load the inc8 actor (sidecar action bounds auto-applied) and take one policy step.
    Returns (rate_frd (3,), collective, normed_thrust, finite_inbounds: bool, actor)."""
    from fly_rl import load_actor, policy_step
    actor = load_actor(actor_path)
    rate_frd, collective, normed_thrust = policy_step(actor, obs20, virtual_flip=virtual_flip)
    # in-bounds: rates within +-act_max_rate, collective in [0,1], normed_thrust in [0, act_max_thrust]
    from fly_rl import _ACT_MAX, _ACT_MIN
    ok = (np.all(np.isfinite(rate_frd)) and np.isfinite(collective) and np.isfinite(normed_thrust)
          and np.all(rate_frd >= _ACT_MIN[1:4] - 1e-6) and np.all(rate_frd <= _ACT_MAX[1:4] + 1e-6)
          and 0.0 <= collective <= 1.0 and _ACT_MIN[0] - 1e-6 <= normed_thrust <= _ACT_MAX[0] + 1e-6)
    return rate_frd, float(collective), float(normed_thrust), bool(ok), actor


def run_slice_once(actor_path=DEFAULT_ACTOR, **scenario):
    """The full vertical slice: frame->slice->PnP->Navigator->NavState->obs[0:20]->actor->action.
    Returns a dict of the frozen 4-tuple + diagnostics."""
    nav, gate, gp, ns = drive_navigator(**scenario)
    obs20, info = build_obs20(nav, gate, ns)
    rate_frd, collective, normed_thrust, ok, _ = actor_step(obs20, actor_path)
    return dict(nav=nav, gate=gate, gate_pose=gp, nav_state=ns, obs20=obs20,
                rate_frd=rate_frd, collective=collective, normed_thrust=normed_thrust,
                action_ok=ok, info=info)


# =====================================================================================
# emul cross-check (#37 "emul fiction"): is the deploy obs the SAME the inc8 actor trained on?
# =====================================================================================
def emul_confidence_fidelity(P_pos_3x3, gate_pos_ned, t_since_fix, target_gate=0):
    """(a) BUILDER FIDELITY: deploy_confidence_triple vs the REAL EstimatorEmulator.confidence_channel
    for the SAME P / gate frame / fix clock. Delta must be ~0 (proves the deploy [17:20] builder is
    byte-faithful to what the inc8 actor trained on)."""
    from estimator_emul import EstimatorEmulator, EmulConfig
    emu = EstimatorEmulator(EmulConfig())
    rng = np.random.default_rng(0)
    # seed a KF, then OVERWRITE its covariance + gate frame + clock with the controlled values:
    from racer.state_estimator import LinearKF

    class _S:  # minimal start-state (reset reads .pos/.vel)
        pos = np.zeros(3); vel = np.zeros(3)
    emu.reset(_S(), target_gate, rng)
    emu.kf.P[:3, :3] = np.asarray(P_pos_3x3, dtype=np.float64)
    emu._t_since_fix = float(t_since_fix)
    Rwg_emul = emu.gates[target_gate].R_world_gate                          # emul's NED gate frame (yaw=pi)
    emul_triple = emu.confidence_channel(target_gate)                       # the REAL emul code
    deploy_triple, _ = deploy_confidence_triple(emu.kf.P[:3, :3], Rwg_emul, t_since_fix)
    return emul_triple, deploy_triple


def regime_delta_floor_on_vs_off():
    """(c) #37 DATUM: deploy KF (inplane floor ON, 0.05) vs the emul training KF regime (floor OFF,
    over-converges). Same real Navigator code + geometry, single flag. Returns the two triples."""
    # floor ON (deploy)
    nav_on, gate, _, ns_on = drive_navigator()
    t_on, _ = deploy_confidence_triple(nav_on.kf.P[:3, :3], gate.R_world_gate, ns_on.time_since_vision_update_s)
    # floor OFF (emul training regime)
    gate2 = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    det2 = _SlicedFakeDetector(gate2, [1.0, -0.5, -2.0])
    cfg2 = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                           use_rewind_kf=True, use_gate_relative=True, use_inplane_pos_floor=False)
    nav_off = Navigator(gates=[gate2], detector=det2, config=cfg2)
    nav_off.update(_ds(0), _frame(0, 0))
    ns_off = None
    for k in range(1, 81):
        ns_off = nav_off.update(_ds(k * 10_000_000), _frame(k, k * 10_000_000))
    t_off, _ = deploy_confidence_triple(nav_off.kf.P[:3, :3], gate2.R_world_gate, ns_off.time_since_vision_update_s)
    return dict(floor_on=t_on, floor_off=t_off,
                P_on=np.diag(nav_on.kf.P[:3, :3]).copy(), P_off=np.diag(nav_off.kf.P[:3, :3]).copy())


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=False, floatmode="maxprec")
    r = run_slice_once()
    gp, ns = r["gate_pose"], r["nav_state"]
    print("=== GatePose (slice->PnP) ===")
    print("  t_cam_gate =", repr(gp.t_cam_gate.tolist()), "range =", gp.range_m,
          "reproj =", gp.reproj_error_px, "n_corners =", gp.n_corners)
    print("=== NavState (converged) ===")
    print("  position_ned =", repr(ns.position_ned.tolist()))
    print("  nav_inplane_sigma =", ns.nav_inplane_sigma, " nav_along_sigma =", ns.nav_along_sigma)
    print("  time_since_vision_update_s =", ns.time_since_vision_update_s)
    print("  isinstance RewindKF =", isinstance(r["nav"].kf, RewindKF),
          " n_rel_applied =", r["nav"].vision_diag.n_rel_applied)
    info = r["info"]
    sig_ip, sig_al = info["sigmas"]
    print("=== obs[17:20] deploy (emul-convention) ===")
    print("  sigma_inplane(emul 0.5avg) =", sig_ip, " sigma_along =", sig_al)
    print("  NavState.nav_inplane_sigma (sqrt-SUM) =", ns.nav_inplane_sigma,
          " ratio sum/avg =", ns.nav_inplane_sigma / sig_ip if sig_ip > 0 else float("nan"))
    print("  triple [c_ip, c_al, age] =", repr(info["triple"].tolist()))
    print("  obs17 build_obs vs obs_from_zup max|delta| =",
          float(np.max(np.abs(info["obs17_buildobs"].astype(np.float64)
                              - info["obs17_fromzup"].astype(np.float64)))))
    print("=== obs20 (frozen golden) ===")
    print("  obs20 =", repr(r["obs20"].tolist()))
    print("=== action ===")
    print("  rate_frd =", repr(r["rate_frd"].tolist()), " collective =", r["collective"],
          " normed_thrust =", r["normed_thrust"], " finite_inbounds =", r["action_ok"])
    print("=== emul cross-check (a) BUILDER FIDELITY ===")
    P = r["nav"].kf.P[:3, :3]
    emul_t, deploy_t = emul_confidence_fidelity(P, r["gate"].position_ned, ns.time_since_vision_update_s)
    print("  emul.confidence_channel =", repr(np.asarray(emul_t).tolist()))
    print("  deploy_confidence_triple =", repr(np.asarray(deploy_t).tolist()))
    print("  max|delta| =", float(np.max(np.abs(np.asarray(emul_t) - np.asarray(deploy_t)))))
    print("=== emul cross-check (c) FLOOR ON (deploy) vs OFF (emul-train regime) ===")
    rd = regime_delta_floor_on_vs_off()
    print("  P_pos diag floor ON  =", repr(rd["P_on"].tolist()))
    print("  P_pos diag floor OFF =", repr(rd["P_off"].tolist()))
    print("  triple floor ON  =", repr(rd["floor_on"].tolist()))
    print("  triple floor OFF =", repr(rd["floor_off"].tolist()))
