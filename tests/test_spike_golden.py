"""FROZEN golden + per-seam regression for the vertical-slice spike (handoff/spike-vertical-2026-06-18).

Proves the full deploy loop CLOSES on one synthetic-exact frame and pins every seam so a future
refactor can't silently break the wiring:

  SENSE   frame(pixels) -> 8->4 slice -> PnP GatePose            (test_ippe_roundtrip, test_slice_8kp_to_4)
  +L      GatePose -> localization world fix (gate-relative +L)  (test_plus_L_sign_localization)
  KF      case-C Navigator (RewindKF + relinnov gate) -> NavState(test_full_slice_navstate_obs_golden)
  OBS     NavState -> obs[0:17] (build_obs==obs_from_zup) ++ [17:20] confidence triple
  ACTOR   obs[0:20] -> inc8 actor -> finite in-bounds action     (test_full_slice_action_golden)
  #37     deploy obs[17:20] builder == the emul the actor trained on (test_deploy_confidence_matches_emul)
  FOOTGUN NavState.nav_inplane_sigma = sqrt(2) x the emul sigma  (test_navstate_inplane_sigma_is_sqrt2_emul)

The frozen vectors are deterministic (verified bit-identical run-to-run). The full-loop golden uses the
DEPLOY path (frames.BORESIGHT = -0.25 live); the +L SIGN identity zeroes BORESIGHT (the -0.25 metric bake
breaks the raw +L identity by ~0.25 m). The action test SKIPS if the gitignored inc8 .pth is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RL = _ROOT / "rl"
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_RL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from racer.contracts import Gate, GateObservation, GatePose
from racer.vision.gate_pose import estimate_gate_pose, project_gate_corners

# ---- FROZEN GOLDENS (deploy path, BORESIGHT=-0.25 live; gate [9,0,-2.5], drone [1,-0.5,-2.0]) -----
G_T_CAM_GATE = np.array([0.5000000000000001, 2.266314836212397, 7.688551037950105])
G_NAV_POS = np.array([1.0001335396363824, -0.5000255216380949, -1.7500885221718736])
G_NAV_INPLANE_SIGMA = 0.0712970386841009          # NavState field = sqrt(P_E+P_D) (sum)
G_NAV_ALONG_SIGMA = 0.08230937118302234
G_OBS20 = np.array([
    -7.999866485595703, 0.5000255107879639, 0.7499114871025085,
    -0.0005160819855518639, -9.774198406375945e-05, 0.0003392087819520384,
    0.0, -0.0, 1.2246468525851679e-16, 0.0, 0.0, 0.0, 0.0,
    55.89374923706055, -2.499990224838257, -6.208041667938232, 0.0,
    0.9917758107185364, 0.6074642539024353, 0.0,
])
G_RATE_FRD = np.array([0.5338851595908487, 2.346648858910744, 0.07093029956333252])
G_COLLECTIVE = 1.072211705592352e-09
G_NORMED_THRUST = 4.036941662621807e-09

GATE_NED = (9.0, 0.0, -2.5)
DRONE_TRUE = np.array([1.0, -0.5, -2.0])
_ACTOR_PTH = _RL / "checkpoints" / "inc8_ws1_seed0_BEST_actor.pth"


# =====================================================================================
# SENSE seam (torch-free)
# =====================================================================================
def test_ippe_roundtrip_recovers_pose_exactly():
    """project a KNOWN gate pose -> 4 px (canonical LL,LR,UR,UL) -> PnP -> recover, empirically
    pinning the IPPE object-point order matches the projector."""
    import racer.frames as _F
    for R, t in [(np.eye(3), np.array([0.0, 0.0, 8.0])),
                 (np.eye(3), np.array([0.3, -0.2, 8.0]))]:
        px = project_gate_corners(R, t)
        obs = GateObservation(frame_id=1, sim_time_ns=1, corners_px=px,
                              corner_confidence=np.ones(4), score=0.95)
        gp = estimate_gate_pose(obs, compute_covariance=True, weighted_refine=True)
        assert gp is not None
        assert np.max(np.abs(gp.t_cam_gate - t)) < 1e-9
        assert gp.t_cam_gate[2] > 0.0          # cheirality: gate in front
        assert gp.reproj_error_px < 1e-6


def test_slice_8kp_to_4_is_inner_subset():
    """8-kpt (n,8,2) -> inner-4 slice is bit-identical to feeding the inner-4 directly (blocker#3)."""
    from spike_vertical_slice import slice_8kp_to_4, observations_from_8kp_results
    from racer.contracts import Frame
    inner = project_gate_corners(np.eye(3), np.array([0.2, -0.1, 7.0]))
    outer = project_gate_corners(np.eye(3), np.array([0.2, -0.1, 7.0]), inner_size_m=2.72)
    xy8 = np.concatenate([inner, outer], axis=0)[None, ...]
    conf8 = np.ones((1, 8))
    xy4, conf4 = slice_8kp_to_4(xy8, conf8)
    assert np.array_equal(xy4[0], inner) and xy4.shape == (1, 4, 2)
    frame = Frame(frame_id=1, sim_time_ns=1, image_bgr=np.zeros((360, 640, 3), np.uint8))
    obs = observations_from_8kp_results(frame, xy8, conf8, np.array([0.9]))
    assert len(obs) == 1 and np.allclose(obs[0].corners_px, inner, atol=1e-12)


# =====================================================================================
# +L sign seam (torch-free): MUST be +L, and -L must break by >=10 m. BORESIGHT zeroed for the identity.
# =====================================================================================
def test_plus_L_sign_localization(monkeypatch):
    import racer.frames as _F
    monkeypatch.setattr(_F, "BORESIGHT", _F.BoresightCorrection())     # zero the -0.25 bake for the identity
    from racer.frames import R_camera_from_body, R_world_from_body
    from racer.localization import gate_pose_to_world_position
    gate_ned = np.array(GATE_NED)
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    R_wc = R_wb @ R_camera_from_body().T
    L = gate_ned - DRONE_TRUE
    gate = Gate(gate_id=0, position_ned=gate_ned, R_world_gate=np.eye(3))
    gp_plus = GatePose(0, 0, np.eye(3), R_wc.T @ L, 0.0, gate_id=0, n_corners=4)
    gp_minus = GatePose(0, 0, np.eye(3), R_wc.T @ (-L), 0.0, gate_id=0, n_corners=4)
    p_plus, _ = gate_pose_to_world_position(gp_plus, gate, R_wb, attitude_noise_std=0.0, fix_cov_floor_std=0.0)
    p_minus, _ = gate_pose_to_world_position(gp_minus, gate, R_wb, attitude_noise_std=0.0, fix_cov_floor_std=0.0)
    assert np.max(np.abs(p_plus - DRONE_TRUE)) <= 1e-6        # +L recovers the true drone pose
    assert np.linalg.norm(p_minus - DRONE_TRUE) >= 10.0      # -L control breaks (proves discrimination)


# =====================================================================================
# #37 deploy obs[17:20] builder == the EMUL the inc8 actor trained on (byte-faithful)
# =====================================================================================
def test_deploy_confidence_matches_emul():
    pytest.importorskip("torch")           # estimator_emul -> fly_rl pulls torch
    from spike_vertical_slice import deploy_confidence_triple
    from estimator_emul import EstimatorEmulator, EmulConfig
    rng = np.random.default_rng(0)
    emu = EstimatorEmulator(EmulConfig())

    class _S:
        pos = np.zeros(3); vel = np.zeros(3)
    emu.reset(_S(), 0, rng)
    P = np.diag([0.0067, 0.0025, 0.0025]) + 0.0003 * np.ones((3, 3))   # a representative SPD position cov
    emu.kf.P[:3, :3] = P
    emu._t_since_fix = 0.03
    Rwg = emu.gates[0].R_world_gate
    emul_triple = np.asarray(emu.confidence_channel(0), dtype=np.float64)
    deploy_triple, _ = deploy_confidence_triple(emu.kf.P[:3, :3], Rwg, 0.03)
    assert np.max(np.abs(emul_triple - deploy_triple.astype(np.float64))) <= 1e-6


def test_navstate_inplane_sigma_is_sqrt2_emul():
    """FOOTGUN GUARD: NavState.nav_inplane_sigma = sqrt(P_E+P_D) (sum); the emul confidence channel uses
    sqrt((P_E+P_D)/2) (average). The deploy obs[17:20] builder MUST use the emul convention, else
    c_inplane is wrong by 1/sqrt(2). Pin the exact sqrt(2) relationship so a 'fix' to either can't drift."""
    pytest.importorskip("torch")
    from spike_vertical_slice import drive_navigator, deploy_confidence_triple
    nav, gate, _, ns = drive_navigator()
    _, (sig_ip_emul, _) = deploy_confidence_triple(nav.kf.P[:3, :3], gate.R_world_gate,
                                                   ns.time_since_vision_update_s)
    assert abs(ns.nav_inplane_sigma / sig_ip_emul - np.sqrt(2.0)) < 1e-9


# =====================================================================================
# FULL SLICE: NavState + obs[0:20] golden (deploy path, BORESIGHT live). torch, no .pth needed.
# =====================================================================================
def test_full_slice_navstate_obs_golden():
    pytest.importorskip("torch")
    from spike_vertical_slice import drive_navigator, build_obs20
    from racer.kf_rewind import RewindKF
    nav, gate, gp, ns = drive_navigator(gate_pos_ned=GATE_NED, drone_true_ned=DRONE_TRUE)

    # GatePose (8->4 slice -> PnP), exact:
    assert gp.n_corners == 4 and gp.reproj_error_px < 1e-6
    assert np.max(np.abs(gp.t_cam_gate - G_T_CAM_GATE)) <= 1e-9

    # NavState (case-C RewindKF, gate-relative +L; z carries the -0.25 boresight):
    assert isinstance(nav.kf, RewindKF) and nav.vision_diag.n_rel_applied >= 1
    assert np.max(np.abs(ns.position_ned - G_NAV_POS)) <= 1e-9
    assert abs(ns.nav_inplane_sigma - G_NAV_INPLANE_SIGMA) <= 1e-9
    assert abs(ns.nav_along_sigma - G_NAV_ALONG_SIGMA) <= 1e-9

    # obs[0:20] (build_obs deploy path == obs_from_zup emul builder; [17:20] confidence triple):
    obs20, info = build_obs20(nav, gate, ns)
    assert np.max(np.abs(info["obs17_buildobs"].astype(np.float64)
                         - info["obs17_fromzup"].astype(np.float64))) <= 1e-6
    assert obs20.shape == (20,)
    assert np.max(np.abs(obs20.astype(np.float64) - G_OBS20)) <= 1e-6


# =====================================================================================
# FULL SLICE: action (LOOP CLOSES). Needs the gitignored inc8 .pth -> skip if absent.
# =====================================================================================
def test_full_slice_action_golden():
    pytest.importorskip("torch")
    if not _ACTOR_PTH.exists():
        pytest.skip(f"inc8 actor checkpoint absent (gitignored): {_ACTOR_PTH}")
    from spike_vertical_slice import run_slice_once
    r = run_slice_once(gate_pos_ned=GATE_NED, drone_true_ned=DRONE_TRUE)
    # THE headline: the loop closes with a FINITE, IN-BOUNDS action.
    assert r["action_ok"] is True
    # frozen action (torch float32; tiny platform drift tolerated):
    assert np.max(np.abs(r["rate_frd"] - G_RATE_FRD)) <= 1e-5
    assert abs(r["collective"] - G_COLLECTIVE) <= 1e-6
    assert abs(r["normed_thrust"] - G_NORMED_THRUST) <= 1e-6
