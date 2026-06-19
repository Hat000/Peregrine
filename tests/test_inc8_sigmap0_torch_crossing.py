"""Pin the diffaero-free MATH that rl/inc8_sigmap0_torch_eval.py relies on to compute the GT
gate-4 crossing (the sigma_p0 ensemble). The torch env itself needs diffaero (Adroit-only), so
this test does NOT import it; it pins the two pieces the evaluator's correctness hinges on:

  1. FRAME EQUIVALENCE. The evaluator records crossings via peregrine_racing.world_to_gateframe with
     the RUNTIME gate yaw (== pi on VQ1). The numpy tool rl/inc8_sigmap0_eval.py uses the hardcoded
     yaw=pi specialization _R_W2G = diag(-1,-1,1). For the two sigma_p0 numbers to measure the SAME
     physical quantity, world_to_gateframe(d, pi) must equal _R_W2G @ d. (A divergence here would be
     a silent frame bug -> a meaningless cross-check.)
  2. PLANE-CROSSING INTERPOLATION. The evaluator interpolates rel_prev -> rel_curr to the gate plane
     x=0 with the SAME formula as inc8_sigmap0_eval.gate4_true_crossing_yz (and crossing_events).

These are replicated (not imported) deliberately: importing the real functions would drag in diffaero.
The replicas are byte-copies of peregrine_racing.world_to_gateframe / inc8_sigmap0_eval's interpolation;
if those source formulas change, update this pin.
"""
import numpy as np
import torch


def _world_to_gateframe(d, yaw):
    """Byte-copy of peregrine_racing.world_to_gateframe (rows [c,s,0; -s,c,0; 0,0,1])."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    return torch.stack([x, y, d[..., 2]], dim=-1)


def _interp_yz(rel_prev, rel_curr):
    """Byte-copy of the evaluator's / inc8_sigmap0_eval.gate4_true_crossing_yz x=0 interpolation."""
    denom = rel_curr[..., 0] - rel_prev[..., 0]
    denom = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    f = -rel_prev[..., 0] / denom
    y = rel_prev[..., 1] + f * (rel_curr[..., 1] - rel_prev[..., 1])
    z = rel_prev[..., 2] + f * (rel_curr[..., 2] - rel_prev[..., 2])
    return y, z


def test_world_to_gateframe_pi_equals_R_W2G():
    """world_to_gateframe(d, pi) == diag(-1,-1,1) @ d to the sin(pi) epsilon (the frame the two
    sigma_p0 tools share)."""
    torch.manual_seed(0)
    d = torch.randn(2000, 3, dtype=torch.float64)
    yaw = torch.full((2000,), float(np.pi), dtype=torch.float64)
    got = _world_to_gateframe(d, yaw).numpy()
    R_W2G = np.diag([-1.0, -1.0, 1.0])
    ref = d.numpy() @ R_W2G.T          # (R @ d) per row
    assert np.abs(got - ref).max() < 1e-12, np.abs(got - ref).max()


def test_interp_midpoint_and_known_crossing():
    """A prev/curr pair straddling x=0 interpolates to the geometric crossing point."""
    # symmetric straddle -> f = 0.5 -> midpoint in y,z
    rel_prev = torch.tensor([[-1.0, 0.2, -0.4]], dtype=torch.float64)
    rel_curr = torch.tensor([[+1.0, 0.6, 0.0]], dtype=torch.float64)
    y, z = _interp_yz(rel_prev, rel_curr)
    assert abs(float(y) - 0.4) < 1e-12 and abs(float(z) - (-0.2)) < 1e-12
    # asymmetric straddle: prev x=-0.25, curr x=+0.75 -> f = 0.25
    rel_prev = torch.tensor([[-0.25, 1.0, 2.0]], dtype=torch.float64)
    rel_curr = torch.tensor([[+0.75, 5.0, 6.0]], dtype=torch.float64)
    y, z = _interp_yz(rel_prev, rel_curr)
    assert abs(float(y) - (1.0 + 0.25 * 4.0)) < 1e-12   # 2.0
    assert abs(float(z) - (2.0 + 0.25 * 4.0)) < 1e-12   # 3.0


def test_full_crossing_matches_numpy_tool_frame():
    """End-to-end on shared Z-up points: the evaluator path (world_to_gateframe at yaw=pi then
    interpolate) equals the numpy tool path (_R_W2G @ delta then the same interpolate)."""
    torch.manual_seed(1)
    g = torch.tensor([-135.49, 0.80, -24.00], dtype=torch.float64)   # gate-4 Z-up (approx)
    yaw = torch.tensor(float(np.pi), dtype=torch.float64)
    R_W2G = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    # build a forward-straddling pair in gate frame, map back to Z-up world, feed both paths
    for _ in range(500):
        rp = torch.tensor([-torch.rand(1).item() * 0.5 - 1e-3,
                           torch.randn(1).item() * 0.2, torch.randn(1).item() * 0.2],
                          dtype=torch.float64)
        rc = torch.tensor([torch.rand(1).item() * 0.5 + 1e-3,
                           torch.randn(1).item() * 0.2, torch.randn(1).item() * 0.2],
                          dtype=torch.float64)
        # gate-frame rel -> world delta -> world pos (R is involutory so R^-1 == R)
        wp = R_W2G @ rp + g
        wc = R_W2G @ rc + g
        # evaluator path
        rel_p_eval = _world_to_gateframe((wp - g).unsqueeze(0), yaw.unsqueeze(0))
        rel_c_eval = _world_to_gateframe((wc - g).unsqueeze(0), yaw.unsqueeze(0))
        y_e, z_e = _interp_yz(rel_p_eval, rel_c_eval)
        # numpy-tool path (hardcoded _R_W2G)
        rel_p_np = (R_W2G @ (wp - g)).unsqueeze(0)
        rel_c_np = (R_W2G @ (wc - g)).unsqueeze(0)
        y_n, z_n = _interp_yz(rel_p_np, rel_c_np)
        assert abs(float(y_e) - float(y_n)) < 1e-9
        assert abs(float(z_e) - float(z_n)) < 1e-9
