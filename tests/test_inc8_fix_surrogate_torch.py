"""S1 PARITY GATE -- batched torch fix-surrogate camera geometry == numpy fix_surrogate (<=1e-4).

The torch core (rl/inc8_estimator_emul.py) is the production twin of the numpy reference; if its
geometry / accept / sigma / covariance / sample diverge, the verified GREEN selection metric is void.
This suite pins every deterministic surrogate primitive over a batch of random poses, plus the baked
geometry constants against their numpy source (frames / fix_surrogate). Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_inc8_fix_surrogate_torch.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import fix_surrogate as FS                                              # noqa: E402
from racer import frames as FR                                         # noqa: E402
from racer.contracts import Gate                                       # noqa: E402
from racer.rl_plant import quat_rotate                                 # noqa: E402
from racer.state_estimator import GRAVITY_NED                          # noqa: E402
import inc8_estimator_emul as IE                                       # noqa: E402
from estimator_emul import ned_gate_frame                             # noqa: E402

DT64 = torch.float64


def _R_from_quat(q):
    return np.stack([quat_rotate(np.asarray(q, np.float64), e) for e in np.eye(3)], axis=-1)


def _quat_from_rpy(roll, pitch, yaw):
    from scipy.spatial.transform import Rotation
    qx = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([qx[3], qx[0], qx[1], qx[2]])


# ============================================================ constants
def test_inc8_constants_match_numpy():
    """The baked (scipy-free) geometry constants equal their numpy source EXACTLY (so the train-env
    core needs no scipy on Adroit, yet is frame-faithful)."""
    assert np.allclose(IE.R_CAMERA_FROM_BODY_NP, FR.R_camera_from_body(), atol=1e-12)
    assert np.allclose(IE.CAMERA_INTRINSICS_K_NP, FR.CAMERA_INTRINSICS_K, atol=0.0)
    assert IE.IMAGE_WIDTH == FR.IMAGE_WIDTH and IE.IMAGE_HEIGHT == FR.IMAGE_HEIGHT
    assert np.allclose(IE.GRAVITY_NED_NP, GRAVITY_NED, atol=0.0)


def test_inc8_surrogate_params_match_fix_surrogate():
    """TorchSurrogateParams defaults == fix_surrogate.FixSurrogate() defaults (the calibration the
    torch core consumes is the baked Track-3 checkpoint, byte-for-byte)."""
    tp = IE.TorchSurrogateParams()
    fs = FS.FixSurrogate()
    for f in ("accept_pmax", "accept_rlo", "accept_wlo", "accept_rhi", "accept_whi",
              "accept_p_out_of_image", "accept_range_guard_lo", "accept_range_guard_hi",
              "sigma_lateral_floor", "sigma_lateral_a1", "sigma_vertical_floor", "sigma_vertical_a1",
              "sigma_depth_floor", "sigma_depth_a1", "sigma_growth_max_range_m", "cov_spd_floor"):
        assert getattr(tp, f) == getattr(fs, f), f
    # and from_fix_surrogate round-trips a recalibrated surrogate
    assert IE.TorchSurrogateParams.from_fix_surrogate(fs) == tp


# ============================================================ helpers to build a shared batch
def _build_batch(seed=0, n=128):
    """Random (drone pose, gate) batch in NED; gates use the canonical ned_gate_frame(yaw)."""
    rng = np.random.default_rng(seed)
    gates_zup = np.array([
        [-23.30, 0.40, 1.39], [-46.89, 2.50, -3.71], [-74.59, -1.20, -12.31],
        [-111.49, 5.10, -23.21], [-135.49, 0.80, -24.00], [-159.19, 4.40, -24.61]])
    flip = np.array([1.0, -1.0, -1.0])
    drones, R_wbs, gate_pos, R_wgs, yaws = [], [], [], [], []
    for _ in range(n):
        gi = rng.integers(0, 6)
        gp = gates_zup[gi] * flip                                          # gate centre NED
        # place the drone somewhere in front (range 8..34 m) with a random attitude
        drone = gp + rng.uniform(-30, 30, 3)
        q = _quat_from_rpy(*rng.uniform([-0.8, -0.6, -np.pi], [0.8, 0.6, np.pi]))
        yaw = np.pi
        drones.append(drone); R_wbs.append(_R_from_quat(q))
        gate_pos.append(gp); R_wgs.append(ned_gate_frame(yaw)); yaws.append(yaw)
    return (np.asarray(drones), np.asarray(R_wbs), np.asarray(gate_pos), np.asarray(R_wgs))


def _numpy_geoms(drones, R_wbs, gate_pos, R_wgs):
    out = []
    for d, Rwb, gp, Rwg in zip(drones, R_wbs, gate_pos, R_wgs):
        g = Gate(gate_id=0, position_ned=gp, R_world_gate=Rwg)
        out.append(FS.geometry(d, Rwb, g))
    return out


def _torch_geom(drones, R_wbs, gate_pos, R_wgs):
    R_cb, K, _, _ = IE._const("cpu", DT64)
    return IE.batched_geometry(
        torch.tensor(drones, dtype=DT64), torch.tensor(R_wbs, dtype=DT64),
        torch.tensor(gate_pos, dtype=DT64), torch.tensor(R_wgs, dtype=DT64), R_cb, K)


# ============================================================ S1.a geometry
def test_geometry_parity():
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=1)
    ngeoms = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    tg = _torch_geom(drones, R_wbs, gate_pos, R_wgs)
    for key, attr in [("range", "range_m"), ("az_deg", "azimuth_deg"), ("el_deg", "elevation_deg"),
                      ("bearing_deg", "bearing_deg"), ("view_deg", "view_angle_deg")]:
        a = tg[key].numpy()
        b = np.array([getattr(g, attr) for g in ngeoms])
        assert np.nanmax(np.abs(a - b)) <= 1e-4, (key, np.nanmax(np.abs(a - b)))
    # in_image: exact boolean agreement
    a_img = tg["in_image"].numpy()
    b_img = np.array([g.in_image for g in ngeoms])
    assert np.array_equal(a_img, b_img), f"in_image mismatch on {int((a_img != b_img).sum())} envs"
    # t_cam / lever vectors
    a_tcam = tg["t_cam"].numpy(); b_tcam = np.array([g.t_cam for g in ngeoms])
    assert np.max(np.abs(a_tcam - b_tcam)) <= 1e-4
    a_lev = tg["lever"].numpy(); b_lev = np.array([g.lever_world for g in ngeoms])
    assert np.max(np.abs(a_lev - b_lev)) <= 1e-4


# ============================================================ S1.b accept band-pass
def test_p_accept_parity():
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=2)
    ngeoms = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    tg = _torch_geom(drones, R_wbs, gate_pos, R_wgs)
    sur = IE.BatchedFixSurrogate(IE.TorchSurrogateParams(), "cpu", DT64)
    a = sur.p_accept(tg["range"], tg["in_image"]).numpy()
    b = np.array([FS.DEFAULT.p_accept(g) for g in ngeoms])
    assert np.max(np.abs(a - b)) <= 1e-4, np.max(np.abs(a - b))
    # also pin accept_prob_in_image across a dense range sweep (the band-pass shape)
    r = torch.linspace(0.0, 45.0, 400, dtype=DT64)
    at = sur.accept_prob_in_image(r).numpy()
    bn = FS.DEFAULT.accept_prob_in_image(r.numpy())
    assert np.max(np.abs(at - bn)) <= 1e-4


# ============================================================ S1.c sigma + covariance
def test_sigma_and_covariance_parity():
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=3)
    ngeoms = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    tg = _torch_geom(drones, R_wbs, gate_pos, R_wgs)
    n = len(ngeoms)
    rng = np.random.default_rng(30)
    # per-env EmulConfig-style episode draws (lateral floor + one-signed bias on lat+vert)
    sig_lat = rng.uniform(0.05, 0.15, n)
    bias = rng.choice([-1.0, 1.0], n) * rng.uniform(0.0, 0.19, n)
    sur = IE.BatchedFixSurrogate(IE.TorchSurrogateParams(), "cpu", DT64)
    sigma_t = sur.fix_sigma(tg["range"], torch.tensor(sig_lat, dtype=DT64)).numpy()
    cov_t = sur.fix_covariance(torch.tensor(sigma_t, dtype=DT64),
                               torch.tensor(R_wgs, dtype=DT64)).numpy()
    for i, g in enumerate(ngeoms):
        ep = FS.replace(FS.DEFAULT, sigma_lateral_floor=float(sig_lat[i]),
                        sigma_lateral_bias=float(bias[i]), sigma_vertical_bias=float(bias[i]),
                        sigma_depth_bias=0.0)
        sl, sv, sd = ep.fix_sigma(g)
        assert abs(sigma_t[i, 0] - sl) <= 1e-4 and abs(sigma_t[i, 1] - sv) <= 1e-4 \
            and abs(sigma_t[i, 2] - sd) <= 1e-4, (i, sigma_t[i], (sl, sv, sd))
        cov_np = ep.fix_covariance(g)
        assert np.max(np.abs(cov_t[i] - cov_np)) <= 1e-4, (i, np.max(np.abs(cov_t[i] - cov_np)))
        # SPD
        assert np.all(np.linalg.eigvalsh(cov_t[i]) > 0)


# ============================================================ S1.d sample (injected randomness)
def test_sample_fix_parity_injected_noise():
    """sample_fix z/cov/accept are EXACT under a shared injected draw: z reconstructs the numpy
    formula (drone_true + Rwg @ (bias + sigma*noise)); accept mirrors u < p_accept."""
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=4)
    ngeoms = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    tg = _torch_geom(drones, R_wbs, gate_pos, R_wgs)
    n = len(ngeoms)
    rng = np.random.default_rng(40)
    sig_lat = rng.uniform(0.05, 0.15, n)
    bias = rng.choice([-1.0, 1.0], n) * rng.uniform(0.0, 0.19, n)
    noise = rng.standard_normal((n, 3))
    accept_u = rng.random(n)
    sur = IE.BatchedFixSurrogate(IE.TorchSurrogateParams(), "cpu", DT64)
    z_t, cov_t, acc_t = sur.sample_fix(
        tg, torch.tensor(drones, dtype=DT64), torch.tensor(R_wgs, dtype=DT64),
        torch.tensor(sig_lat, dtype=DT64), torch.tensor(bias, dtype=DT64),
        torch.tensor(accept_u, dtype=DT64), torch.tensor(noise, dtype=DT64))
    z_t, acc_t = z_t.numpy(), acc_t.numpy()
    for i, g in enumerate(ngeoms):
        ep = FS.replace(FS.DEFAULT, sigma_lateral_floor=float(sig_lat[i]),
                        sigma_lateral_bias=float(bias[i]), sigma_vertical_bias=float(bias[i]),
                        sigma_depth_bias=0.0)
        sl, sv, sd = ep.fix_sigma(g)
        sig = np.array([sl, sv, sd])
        bias_vec = np.array([bias[i], bias[i], 0.0])
        noise_gate = bias_vec + sig * noise[i]
        drone_true = g.gate_position_ned - g.lever_world
        z_np = drone_true + g.R_world_gate @ noise_gate
        assert np.max(np.abs(z_t[i] - z_np)) <= 1e-4, (i, np.max(np.abs(z_t[i] - z_np)))
        # accept mirrors numpy semantics: numpy rejects iff u >= p  -> accept iff u < p
        p = ep.p_accept(g)
        assert bool(acc_t[i]) == bool(accept_u[i] < p), (i, accept_u[i], p)


# ============================================================ float32 deployment dtype
def test_geometry_parity_float32_within_tol():
    """The DEPLOYMENT dtype (float32, the GPU env) stays within the 1e-4 parity band."""
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=5)
    ngeoms = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    R_cb, K, _, _ = IE._const("cpu", torch.float32)
    tg = IE.batched_geometry(
        torch.tensor(drones, dtype=torch.float32), torch.tensor(R_wbs, dtype=torch.float32),
        torch.tensor(gate_pos, dtype=torch.float32), torch.tensor(R_wgs, dtype=torch.float32), R_cb, K)
    a = tg["range"].numpy(); b = np.array([g.range_m for g in ngeoms])
    assert np.max(np.abs(a - b)) <= 1e-4
    assert np.array_equal(tg["in_image"].numpy(), np.array([g.in_image for g in ngeoms]))
