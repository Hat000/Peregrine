"""2026-07-05 training-audit fixes (A0-A3) -- laptop tests, torch+numpy only, no diffaero.

THE headline pin: test_legacy_camera_points_backward_TAIL_FIRST_PROOF proves, through the actual env
geometry code, that under the documented tail-first spawn convention (peregrine_course.py: spawn yaw =
gate_yaw + pi) the LEGACY emulated camera mount points BACKWARD relative to flight -- the gate sits
behind the camera on the ideal approach, so every pre-fix curriculum stage trained perception-blind.
The rest pin the fix (EmulConfig.camera_flip + the tail-mount look-at frame + analytic gain signs) and
the reward/exploration repairs (GT-anchor clamp, signed gate-progress, logstd reset, curriculum knobs).

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_vq2_audit_fixes.py -q
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import inc8_estimator_emul as IE                                       # noqa: E402
import inc8_reward as R8                                               # noqa: E402

DT = torch.float64


def _tail_first_pose_ned(gate_dist_m: float = 25.0):
    """The canonical training approach pose, built EXACTLY as the env does: gate at NED origin with
    gate yaw 0 (gate frame == world), drone gate_dist_m up-course (gate-frame -x), spawn yaw =
    gate_yaw + pi (TAIL-FIRST, peregrine_course.py) => the drone flies toward the gate along body -x.
    Returns (drone_pos_ned (1,3), R_wb_ned (1,3,3), gate_pos_ned (1,3), R_wg (1,3,3))."""
    gate_pos_ned = torch.zeros(1, 3, dtype=DT)
    R_wg = IE.ned_gate_frame_torch(torch.zeros(1, dtype=DT))
    drone_pos_ned = torch.tensor([[-gate_dist_m, 0.0, 0.0]], dtype=DT)
    yaw = math.pi
    q = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]], dtype=DT)
    R_zup = IE.quat_xyzw_to_matrix_torch(q)
    f = torch.tensor([1.0, -1.0, -1.0], dtype=DT)
    R_ned = R_zup * f.view(1, 3, 1) * f.view(1, 1, 3)     # the env's _flipconj
    return drone_pos_ned, R_ned, gate_pos_ned, R_wg


def _rz_pi():
    return torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=DT))


# ============================================================ A0: the camera-mount inversion
def test_legacy_camera_points_backward_TAIL_FIRST_PROOF():
    """THE audit-A0 proof: under the tail-first spawn convention the LEGACY mount puts the gate BEHIND
    the emulated camera (t_cam z < 0, not in image) for a drone dead-centered on the ideal approach.
    This is the structural reason every prior stage trained blind."""
    pos, R_ned, gp, Rwg = _tail_first_pose_ned()
    R_cb, K, _, _ = IE._const("cpu", DT)
    geom = IE.batched_geometry(pos, R_ned, gp, Rwg, R_cb, K)
    assert geom["t_cam"][0, 2].item() < 0, "legacy mount: gate must be BEHIND the camera (the bug)"
    assert not bool(geom["in_image"][0]), "legacy mount: gate must NOT be in image (the bug)"


def test_camera_flip_faces_flight_direction():
    """The fix: camera_flip remounts the emulated camera pi about body z -> the same tail-first pose
    has the gate IN FRONT (t_cam z > 0) and IN IMAGE, dead-center in azimuth."""
    pos, R_ned, gp, Rwg = _tail_first_pose_ned()
    R_cb, K, _, _ = IE._const("cpu", DT)
    geom = IE.batched_geometry(pos, R_ned, gp, Rwg, R_cb @ _rz_pi(), K)
    assert geom["t_cam"][0, 2].item() > 0, "flipped mount: gate must be IN FRONT of the camera"
    assert bool(geom["in_image"][0]), "flipped mount: gate must be in image on the ideal approach"
    assert abs(geom["az_deg"][0].item()) < 1e-6


def test_emulator_camera_flip_config_wiring():
    """EmulConfig.camera_flip=True must produce the flipped R_cb on the emulator instance; default
    False must stay byte-identical to the legacy constant."""
    gp = torch.zeros(1, 3, dtype=DT)
    Rwg = IE.ned_gate_frame_torch(torch.zeros(1, dtype=DT))
    legacy = IE.BatchedEstimatorEmulator(1, gp, Rwg, config=IE.EmulConfig(), dtype=DT)
    fixed = IE.BatchedEstimatorEmulator(1, gp, Rwg, config=IE.EmulConfig(camera_flip=True), dtype=DT)
    R_cb, _, _, _ = IE._const(legacy.device, DT)
    assert torch.equal(legacy.R_cb, R_cb), "default must stay byte-identical"
    assert torch.allclose(fixed.R_cb, R_cb @ _rz_pi().to(fixed.R_cb.device))


def test_r_bc_tail_mount_matches_flipped_camera():
    """r_body_from_camera(tail_mount=True) must equal (R_cb @ Rz_pi)^T -- the look-at correction and
    the fix geometry MUST live in the same virtual camera frame -- and stay a proper rotation."""
    R_cb, _, _, _ = IE._const("cpu", DT)
    r_bc_tail = R8.r_body_from_camera(dtype=DT, tail_mount=True)
    assert torch.allclose(r_bc_tail, (R_cb @ _rz_pi()).T, atol=1e-12)
    assert torch.allclose(r_bc_tail @ r_bc_tail.T, torch.eye(3, dtype=DT), atol=1e-12)
    assert abs(torch.linalg.det(r_bc_tail).item() - 1.0) < 1e-9
    assert torch.allclose(R8.r_body_from_camera(dtype=DT), R_cb.T, atol=1e-12), "default byte-id"


def test_lookat_analytic_signs_reduce_azimuth_in_fixed_frame():
    """END-TO-END sign pin for the fixed frame: gate ~10 deg off the FLIPPED camera axis; the look-at
    correction with the ANALYTIC POSITIVE gains, mapped through the tail-mount r_bc + the FRD->FLU
    action flip and integrated one small step on the body attitude, must REDUCE |azimuth|. (The
    historical -3 yaw gain was compensation for the mirrored legacy frame.)"""
    pos, R_ned, gp, Rwg = _tail_first_pose_ned()
    gp = gp.clone()
    gp[0, 1] = -25.0 * math.tan(math.radians(10.0))
    R_cb, K, _, _ = IE._const("cpu", DT)
    R_cb_f = R_cb @ _rz_pi()
    geom0 = IE.batched_geometry(pos, R_ned, gp, Rwg, R_cb_f, K)
    az0 = abs(geom0["az_deg"][0].item())
    assert az0 > 5.0, "test setup: gate must start meaningfully off-axis"
    r_bc = R8.r_body_from_camera(dtype=DT, tail_mount=True)
    flip = torch.tensor(R8._FLIP_FRD_FLU, dtype=DT)
    w_flu = R8.lookat_correction(geom0["t_cam"], 3.0, 3.0, r_bc, flip)   # ANALYTIC positive gains
    dt_s = 0.02
    w = w_flu[0] * dt_s
    wx = torch.tensor([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=DT)
    R_exp = torch.matrix_exp(wx)
    f = torch.tensor([1.0, -1.0, -1.0], dtype=DT)
    R_zup0 = R_ned * f.view(1, 3, 1) * f.view(1, 1, 3)                    # NED -> Z-up (involutory)
    R_zup1 = R_zup0[0] @ R_exp
    R_ned1 = (R_zup1 * f.view(3, 1) * f.view(1, 3)).unsqueeze(0)
    geom1 = IE.batched_geometry(pos, R_ned1, gp, Rwg, R_cb_f, K)
    az1 = abs(geom1["az_deg"][0].item())
    assert az1 < az0, f"analytic +gains must REDUCE |az| in the fixed frame ({az0:.3f} -> {az1:.3f})"


# ============================================================ A2: reward economics
def test_gt_anchor_clamp():
    """estimerr_clamp caps the per-step drain; 0 stays byte-identical unclamped."""
    err = torch.tensor([0.1, 0.5, 3.0, 10.0], dtype=DT)
    un = R8.gt_estimerr_anchor(err, 2.0)
    assert torch.allclose(un, -2.0 * err)
    cl = R8.gt_estimerr_anchor(err, 2.0, clamp_m=0.5)
    assert torch.allclose(cl, -2.0 * torch.clamp(err, max=0.5))
    assert cl.min().item() == -1.0, "0.5 m clamp at estimerr=2 must cap the drain at -1/step"
    assert torch.allclose(R8.gt_estimerr_anchor(err, 2.0, clamp_m=0.0), un), "clamp 0 == byte-id"


def test_gate_progress_reward_signed_and_gated():
    """Signed (retreat costs -- no free oscillation), telescoping, exact-zero at weight 0."""
    prev = torch.tensor([25.0, 25.0, 25.0], dtype=DT)
    curr = torch.tensor([24.8, 25.3, 25.0], dtype=DT)
    r = R8.gate_progress_reward(prev, curr, 10.0)
    assert r[0].item() == pytest.approx(2.0)
    assert r[1].item() == pytest.approx(-3.0)
    assert r[2].item() == 0.0
    z = R8.gate_progress_reward(prev, curr, 0.0)
    assert torch.equal(z, torch.zeros_like(prev)), "weight 0 must be the exact zero tensor (byte-id)"


def test_inc8_weights_new_fields_default_off():
    """The new Inc8RewardWeights fields default OFF (byte-identical legacy) and are cfg-mappable."""
    w = R8.Inc8RewardWeights()
    assert w.estimerr_clamp == 0.0
    assert w.gate_progress == 0.0


# ============================================================ A3: exploration reset
def test_warmstart_logstd_reset_helper():
    """reset_actor_logstd zeroes every actor_logstd on the module tree (fresh-init exploration on
    transferred means), leaves the critic untouched, and reports how many it touched."""
    from inc8_warmstart import reset_actor_logstd

    class _Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_logstd = torch.nn.Parameter(torch.full((1, 4), -4.9))

    class _AC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = _Actor()
            self.critic = torch.nn.Linear(3, 1)

    class _AgentShim:
        def __init__(self):
            self.agent = _AC()

    shim = _AgentShim()
    n = reset_actor_logstd(shim)
    assert n == 1
    assert torch.equal(shim.agent.actor.actor_logstd.data, torch.zeros(1, 4))
    assert any(p.abs().sum() > 0 for p in shim.agent.critic.parameters())


# ============================================================ curriculum carries the fixes
def test_curriculum_stages_carry_audit_fixes():
    """Every DEFAULT-ladder stage must carry the audit-A0/A2 knobs (a stage silently dropping one is
    exactly the regression class this pins); dual_gate sits between blackout_pass and multi_gate with
    obs[13:17] live (2 gates); the _raw renderer emits existing keys verbatim."""
    from vq2_curriculum import STAGES, STAGE_ORDER, render_overrides
    assert STAGE_ORDER == ("single_gate", "blackout_pass", "dual_gate", "multi_gate")
    for s in STAGE_ORDER:
        d = STAGES[s]
        assert d["emul_camera_flip"] is True, (s, "camera_flip")
        assert d["emul_tau_stale"] == 0.5, (s, "tau_stale")
        assert d["rw_gate_progress"] == 10.0, (s, "gate_progress")
        assert d["rw_estimerr_clamp"] == 0.5, (s, "estimerr_clamp")
        assert d["lookat_g_yaw"] == 3.0 and d["lookat_g_pitch"] == 3.0, (s, "analytic lookat signs")
    assert STAGES["dual_gate"]["course_n_gates"] == 2
    assert STAGES["multi_gate"]["_raw"]["env.max_time"] == 100
    assert STAGES["multi_gate"]["_raw"]["algo.gamma"] == 0.995
    toks = render_overrides("multi_gate")
    assert "env.max_time=100" in toks and "algo.gamma=0.995" in toks
    assert "+env.emul_camera_flip=True" in toks
    assert not any(t.startswith("+env._raw") for t in toks)
