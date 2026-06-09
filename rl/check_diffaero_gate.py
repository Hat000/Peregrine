"""Peregrine T4 GATE -- validate the DiffAero torch dynamics mirror vs the numpy rl_plant.

Constructs :class:`PeregrinePlantDynamics` under the REAL diffaero ``BaseDynamics``, sets a
randomized NON-TRIVIAL state (random pose/vel/rate -- NOT hover, so every sign/rotation path
is exercised), and steps both backends from an identical start:

  * float64 run -> ALGEBRAIC correctness of the hand-written torch mirror. With params rebuilt
    from the float64 source, an exact mirror matches the parity-tested numpy plant to ~1e-12.
    THIS IS THE GATE (numpy rl_plant is ground truth, itself bit-identical to twin.py).
  * float32 run -> realistic training-precision divergence (single step). Informational.

The adapter caches rate_gain / g_vec / BODY_UP as float32; we rebuild them from the float64
``params`` at the target dtype so the float64 gate measures MATH error, not float32 param rounding.

Prints DIV_FLOAT64 / DIV_FLOAT32 (overall + per-axis) and an explicit GATE_PASS / GATE_FAIL line.
"""
import numpy as np
import torch
from omegaconf import OmegaConf

from diffaero_dynamics import PeregrinePlantDynamics

GATE_TOL = 1e-9     # float64 algebraic-equivalence bound (parity test hit ~1.6e-12)
F32_TOL = 1e-4      # float32 single-step advisory bound
N_ENVS = 16
SEEDS = range(6)


def build_cfg(n_envs):
    return OmegaConf.create({
        "name": "peregrine_plant",
        "n_envs": int(n_envs),
        "n_agents": 1,
        "dt": 0.02,
        "alpha": 1.0,
        "g": 9.80665,
        "n_substeps": 1,
        "controller": {
            "min_normed_thrust": 0.0, "max_normed_thrust": 5.0,
            "min_roll_rate": -3.14, "max_roll_rate": 3.14,
            "min_pitch_rate": -3.14, "max_pitch_rate": 3.14,
            "min_yaw_rate": -3.14, "max_yaw_rate": 3.14,
        },
    })


def rebuild_params(dyn, device, dtype):
    """Rebuild the cached torch params from the float64 source at ``dtype`` (so float64 == numpy)."""
    dyn._rate_gain = torch.tensor(dyn.params.rate_gain, device=device, dtype=dtype)
    dyn._rate_sign = torch.tensor(dyn.params.rate_sign, device=device, dtype=dtype)
    dyn._BODY_UP = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
    dyn._g_vec_ned = torch.tensor([0.0, 0.0, float(dyn.params.g)], device=device, dtype=dtype)
    dyn._acc = dyn._acc.to(dtype)


def random_state(n_envs, device, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.randn(n_envs, 3, generator=g) * 3.0
    q = torch.randn(n_envs, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)              # valid unit quaternions, full SO(3)
    v = torch.randn(n_envs, 3, generator=g) * 2.0
    w = torch.randn(n_envs, 3, generator=g) * 0.6     # rad/s body rate
    state = torch.cat([p, q, v, w], dim=-1).to(device=device, dtype=dtype)
    u = torch.empty(n_envs, 4)
    u[:, 0] = 1.0 + 0.3 * torch.randn(n_envs, generator=g)   # normed thrust around hover
    u[:, 1:] = 0.8 * torch.randn(n_envs, 3, generator=g)     # body-rate setpoints rad/s
    return state, u.to(device=device, dtype=dtype)


def per_axis(cand, ref):
    d = np.abs(cand - ref)
    return (float(d[..., 0:3].max()), float(d[..., 3:7].max()),
            float(d[..., 7:10].max()), float(d[..., 10:13].max()))


def gate_once(dtype, device, seed):
    cfg = build_cfg(N_ENVS)
    dyn = PeregrinePlantDynamics(cfg, device, backend="torch")
    rebuild_params(dyn, device, dtype)
    state, u = random_state(N_ENVS, device, dtype, seed)
    thr0 = torch.full((N_ENVS,), float(dyn.params.hover_thrust), device=device, dtype=dtype)

    # --- the named gate: scalar max-abs divergence (huge atol so it measures, we judge below) ---
    dyn._state = state.clone()
    dyn._thrust = thr0.clone()
    div = dyn.check_against_rl_plant(u, atol=1e30)

    # --- per-axis diagnostics: re-run both backends from the identical start ---
    dyn._state = state.clone(); dyn._thrust = thr0.clone(); dyn.backend = "rl_plant_numpy"
    dyn._step_numpy(u); ref = dyn._state.detach().cpu().numpy().copy()
    dyn._state = state.clone(); dyn._thrust = thr0.clone(); dyn.backend = "torch"
    dyn._step_torch(u); cand = dyn._state.detach().cpu().numpy().copy()
    return float(div), per_axis(cand, ref)


def summarize(dtype, device):
    worst = 0.0
    worst_axes = (0.0, 0.0, 0.0, 0.0)
    for s in SEEDS:
        div, axes = gate_once(dtype, device, s)
        if div > worst:
            worst, worst_axes = div, axes
    return worst, worst_axes


def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print("DEVICE", device, (torch.cuda.get_device_name(0) if has_cuda else "cpu"),
          "torch", torch.__version__)

    d64, ax64 = summarize(torch.float64, device)
    d32, ax32 = summarize(torch.float32, device)

    print("DIV_FLOAT64 %.3e  [pos %.2e quat %.2e vel %.2e omega %.2e]" % (d64, *ax64))
    print("DIV_FLOAT32 %.3e  [pos %.2e quat %.2e vel %.2e omega %.2e]" % (d32, *ax32))

    if d64 < GATE_TOL:
        print("GATE_PASS  float64 max-divergence %.3e < %.0e  "
              "(torch mirror is algebraically faithful to numpy rl_plant)" % (d64, GATE_TOL))
    else:
        print("GATE_FAIL  float64 max-divergence %.3e >= %.0e" % (d64, GATE_TOL))
    print("FLOAT32_%s %.3e  (single-step training precision; advisory tol %.0e)"
          % ("OK" if d32 < F32_TOL else "HIGH", d32, F32_TOL))


if __name__ == "__main__":
    main()
