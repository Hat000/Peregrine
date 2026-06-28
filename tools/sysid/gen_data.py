"""Generate plant-identification data by rolling out diffaero's QuadrotorModel.

Instantiates ``QuadrotorModel`` with randomizers DISABLED so every env carries the
nominal ``quad.yaml`` params (the KNOWN recovery target). Applies informative excitation
(chirp + random walk across the flight envelope), and logs ``(X, U, Xdot)`` where ``Xdot``
is the EXACT analytic derivative from ``model.dynamics()`` -- no finite-difference error, so
the deterministic-sim target noise floor is float32 eps.

Output: ``$SYSID_WS/iddata.npz`` with X (N,13), U (N,4), Xdot (N,13), plus the true params.

Run inside the diffaero conda env with ``PYTHONPATH`` pointing at the diffaero repo parent
(so ``import diffaero.dynamics.quadrotor`` resolves). See README.md.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from omegaconf import OmegaConf

torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WS = os.environ.get("SYSID_WS", "/scratch/network/fl3689/sysid_ws")

from diffaero.dynamics.quadrotor import QuadrotorModel  # noqa: E402


def rz(default):
    """A diffaero randomizer spec pinned to a constant (enabled=False -> uses default)."""
    return {"default": default, "enabled": False, "min": default, "max": default}


N_ENVS = 4096
DT = 0.02
CFG = OmegaConf.create({
    "name": "quadrotor", "n_envs": N_ENVS, "n_agents": 1, "dt": DT,
    "action_frame": "body", "alpha": 0.0,
    "m": rz(1.0), "arm_l": rz(0.15), "c_tau": rz(0.0133),
    "g": 9.81,
    "J": {"xy": rz(0.01), "z": rz(0.02)},
    "D": {"xy": rz(0.6), "z": rz(0.6)},
    "max_w_xy": 5.0, "max_w_z": 1.0, "max_T": 4.179, "min_T": 0.0, "lmbda": 0.1,
    "solver_type": "rk4", "n_substeps": 1,
    "controller": {
        "compensate_gravity": False,
        "min_normed_thrust": 0.0, "max_normed_thrust": 5.0,
        "min_pitch_rate": -3.14, "max_pitch_rate": 3.14,
        "min_roll_rate": -3.14, "max_roll_rate": 3.14,
        "min_yaw_rate": -3.14, "max_yaw_rate": 3.14,
        "thrust_ratio": 1.0, "torque_ratio": 1.0,
        "min_normed_torque": [-15., -15., -15.], "max_normed_torque": [15., 15., 15.],
        "K_angvel": [1., 1., 1.],
    },
})

TRUE = {  # the recovery target (from quad.yaml defaults)
    "m": 1.0, "g": 9.81, "J_xy": 0.01, "J_z": 0.02, "D_xy": 0.6, "D_z": 0.6,
    "arm_l": 0.15, "c_tau": 0.0133, "K_angvel": [1., 1., 1.],
}


def make_excitation(n_envs, n_steps, seed=0):
    """Per-env informative control: chirped body rates + a thrust band that keeps the drone
    manoeuvring across the envelope (not hover). Returns (T, N, 4)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.arange(n_steps).float() * DT
    f0 = 0.2 + 1.8 * torch.rand(n_envs, 3, generator=g)        # start freq Hz
    f1 = 1.0 + 4.0 * torch.rand(n_envs, 3, generator=g)        # end freq Hz
    amp = 0.5 + 2.0 * torch.rand(n_envs, 3, generator=g)       # rate amplitude rad/s
    phase = 2 * np.pi * torch.rand(n_envs, 3, generator=g)
    T_total = n_steps * DT
    k = (f1 - f0) / T_total
    ph = 2 * np.pi * (f0[None] * t[:, None, None] + 0.5 * k[None] * (t[:, None, None] ** 2)) + phase[None]
    rates = amp[None] * torch.sin(ph)                          # (T,N,3)
    rw = torch.cumsum(0.15 * torch.randn(n_steps, n_envs, 3, generator=g), dim=0)
    rw = rw - rw.mean(0, keepdim=True)
    rates = torch.clamp(rates + 0.5 * rw, -3.14, 3.14)
    tf0 = 0.3 + 1.0 * torch.rand(n_envs, generator=g)
    tamp = 0.3 + 0.8 * torch.rand(n_envs, generator=g)
    toff = 0.9 + 0.6 * torch.rand(n_envs, generator=g)        # center 0.9..1.5
    thrust = toff[None] + tamp[None] * torch.sin(2 * np.pi * tf0[None] * t[:, None] + phase[:, 0][None])
    thrust = torch.clamp(thrust, 0.0, 5.0)                     # (T,N)
    return torch.cat([thrust[..., None], rates], dim=-1)       # (T,N,4)


def main():
    n_steps = int(os.environ.get("N_STEPS", 400))
    warmup = 20
    model = QuadrotorModel(CFG, torch.device(DEVICE))
    # random initial attitude/velocity so we cover the SO(3) x velocity envelope
    g = torch.Generator(device="cpu").manual_seed(123)
    q = torch.randn(N_ENVS, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    model._state[..., 3:7] = q.to(DEVICE)
    model._state[..., 7:10] = (2.0 * torch.randn(N_ENVS, 3, generator=g)).to(DEVICE)   # vel
    model._state[..., 10:13] = (1.0 * torch.randn(N_ENVS, 3, generator=g)).to(DEVICE)  # body rate

    U_all = make_excitation(N_ENVS, n_steps, seed=7).to(DEVICE)

    Xs, Us, Xdots = [], [], []
    with torch.no_grad():
        for i in range(n_steps):
            U = U_all[i]
            X = model._state.clone()
            Xdot = model.dynamics(X, U)          # EXACT analytic derivative
            if i >= warmup:                      # let transients spread coverage
                Xs.append(X.cpu().numpy())
                Us.append(U.cpu().numpy())
                Xdots.append(Xdot.cpu().numpy())
            model.step(U)
            bad = ~torch.isfinite(model._state).all(dim=-1)
            if bad.any():                        # respawn diverged (float32 blowup) envs
                model._state[bad] = 0.0
                model._state[bad, 6] = 1.0

    X = np.concatenate(Xs, 0).astype(np.float64)
    U = np.concatenate(Us, 0).astype(np.float64)
    Xdot = np.concatenate(Xdots, 0).astype(np.float64)
    os.makedirs(WS, exist_ok=True)
    out = os.path.join(WS, "iddata.npz")
    np.savez_compressed(out, X=X, U=U, Xdot=Xdot, true=np.array([str(TRUE)]))
    print(f"saved {out}  X{X.shape} U{U.shape} Xdot{Xdot.shape}")
    print("frac envs |v|>1: %.2f  |w|>0.5: %.2f" % (
        (np.linalg.norm(X[:, 7:10], axis=1) > 1).mean(),
        (np.linalg.norm(X[:, 10:13], axis=1) > 0.5).mean()))


if __name__ == "__main__":
    main()
