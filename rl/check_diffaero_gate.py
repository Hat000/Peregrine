"""Peregrine T4 GATE -- validate the DiffAero torch dynamics mirror vs the numpy rl_plant.

Constructs :class:`PeregrinePlantDynamics` under the REAL diffaero ``BaseDynamics`` and drives BOTH
backends through randomized NON-TRIVIAL multi-step trajectories (random pose/vel/rate starts --
NOT hover -- and rate commands with tails beyond pi, so every sign/rotation path, the super-rate
map's min(|c|,pi) boundary, the slew clamp, and the transport-delay ring buffer all get exercised)
across a CONFIG MATRIX:

  * legacy      -- default PlantParams (the exact pre-map plant; historical passes 2.2e-16..4.4e-16)
  * super_rate  -- measured static gain map + slew (characterize-sweep 2026-06-10, s=0.30,
                   alpha_max=[260,260,80])
  * delay2      -- params.transport_delay_steps=2 (rl_plant's internal ring buffer, mirrored in the
                   torch backend per the S12 handoff item 4a)
  * map_delay   -- both at once
  * aero        -- measured aero (twin-falsify 2026-06-11): body-frame sign-split quadratic drag +
                   convex collective knot table (linear_drag=0); thrust commands include both
                   knot-table end clamps
  * aero_full   -- aero + super-rate map + slew + transport delay, everything ON at once

Per config: float64 run -> ALGEBRAIC correctness of the hand-written torch mirror (THIS IS THE
GATE; numpy rl_plant is ground truth, itself bit-identical to twin.py); float32 run -> realistic
training-precision divergence over the trajectory (informational). Every intermediate state along
the trajectory is compared (``check_against_rl_plant`` with an action stack).

The adapter caches rate_gain / g_vec / BODY_UP / super_s / alpha_max as float32; we rebuild them
from the float64 ``params`` at the target dtype so the float64 gate measures MATH error, not
float32 param rounding.

Prints per-config DIV_FLOAT64 / DIV_FLOAT32 and an explicit GATE_PASS / GATE_FAIL line.
"""
import numpy as np
import torch
from omegaconf import OmegaConf

from racer.rl_plant import (PlantParams, QUAD_DRAG_C2_MEASURED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
from diffaero_dynamics import PeregrinePlantDynamics

GATE_TOL = 1e-9     # float64 algebraic-equivalence bound (acceptance <= ~1e-6; history ~2e-16)
F32_TOL = 1e-3      # float32 multi-step advisory bound (8 chaotic steps compound rounding)
N_ENVS = 16
T_STEPS = 8         # steps per trajectory (> transport_delay_steps so the delay buffer cycles)
SEEDS = range(6)

_AERO = dict(linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
             coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
             coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
_MAP = dict(super_rate_s=0.30, alpha_max_rps2=np.array([260.0, 260.0, 80.0]))

CONFIGS = {
    "legacy":     dict(),
    "super_rate": dict(_MAP),
    "delay2":     dict(transport_delay_steps=2),
    "map_delay":  dict(_MAP, transport_delay_steps=2),
    # measured aero (twin-falsify 2026-06-11): quad body drag + convex collective knot table.
    # The random velocities exercise the per-axis sign split; the thrust commands span the knot
    # interior + both end clamps (see random_traj).
    "aero":       dict(_AERO),
    # everything ON at once: aero + super-rate map + slew + transport delay
    "aero_full":  dict(_AERO, **_MAP, transport_delay_steps=2),
}


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
    dyn._super_s = (None if dyn.params.super_rate_s is None else
                    torch.tensor(np.broadcast_to(dyn.params.super_rate_s, (3,)).copy(),
                                 device=device, dtype=dtype))
    dyn._alpha_max = (None if dyn.params.alpha_max_rps2 is None else
                      torch.tensor(np.broadcast_to(dyn.params.alpha_max_rps2, (3,)).copy(),
                                   device=device, dtype=dtype))
    dyn._quad_c2 = (None if dyn.params.quad_drag_c2 is None else
                    torch.tensor(dyn.params.quad_drag_c2, device=device, dtype=dtype))
    dyn._coll_knots = (None if dyn.params.coll_map_thr is None else
                       torch.tensor(dyn.params.coll_map_thr, device=device, dtype=dtype))
    dyn._coll_kvals = (None if dyn.params.coll_map_accel is None else
                       torch.tensor(dyn.params.coll_map_accel, device=device, dtype=dtype))
    dyn._plant_act_buf = None        # cold delay buffer; both backends seed it identically
    dyn._acc = dyn._acc.to(dtype)


def random_traj(n_envs, device, dtype, seed):
    """A random non-trivial start state + a (T_STEPS, n_envs, 4) action stack."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.randn(n_envs, 3, generator=g) * 3.0
    q = torch.randn(n_envs, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)              # valid unit quaternions, full SO(3)
    v = torch.randn(n_envs, 3, generator=g) * 2.0
    w = torch.randn(n_envs, 3, generator=g) * 0.6     # rad/s body rate
    state = torch.cat([p, q, v, w], dim=-1).to(device=device, dtype=dtype)
    u = torch.empty(T_STEPS, n_envs, 4)
    u[..., 0] = 1.0 + 0.3 * torch.randn(T_STEPS, n_envs, generator=g)   # normed thrust ~ hover
    # knot-table edge coverage (aero configs): one step above the last knot (collective > 1.0 ->
    # upper end clamp), one in the bottom region (floored 0.0/0.10 knots; lower edge). Harmless
    # for the linear-map configs (their thrust map has no knots).
    u[T_STEPS - 2, :, 0] = 3.8 + 1.2 * torch.rand(n_envs, generator=g)  # collective ~ [1.01, 1.33]
    u[T_STEPS - 1, :, 0] = 0.4 * torch.rand(n_envs, generator=g)        # collective ~ [0, 0.106]
    # body-rate setpoints with tails beyond pi: exercises the map's min(|c|,pi) clamp + the slew
    u[..., 1:] = 1.5 * torch.randn(T_STEPS, n_envs, 3, generator=g)
    return state, u.to(device=device, dtype=dtype)


def gate_once(kwargs, dtype, device, seed):
    cfg = build_cfg(N_ENVS)
    dyn = PeregrinePlantDynamics(cfg, device, backend="torch", params=PlantParams(**kwargs))
    rebuild_params(dyn, device, dtype)
    state, u = random_traj(N_ENVS, device, dtype, seed)
    dyn._state = state.clone()
    dyn._thrust = torch.full((N_ENVS,), float(dyn.params.hover_thrust), device=device, dtype=dtype)
    # huge atol: the method measures, we judge below
    return float(dyn.check_against_rl_plant(u, atol=1e30))


def main():
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print("DEVICE", device, (torch.cuda.get_device_name(0) if has_cuda else "cpu"),
          "torch", torch.__version__)

    all_pass = True
    worst64 = 0.0
    for name, kw in CONFIGS.items():
        d64 = max(gate_once(kw, torch.float64, device, s) for s in SEEDS)
        d32 = max(gate_once(kw, torch.float32, device, s) for s in SEEDS)
        ok = d64 < GATE_TOL
        all_pass &= ok
        worst64 = max(worst64, d64)
        print("CONFIG %-10s DIV_FLOAT64 %.3e (%s)   DIV_FLOAT32 %.3e (%s; advisory)"
              % (name, d64, "ok" if ok else "FAIL", d32, "ok" if d32 < F32_TOL else "HIGH"))

    if all_pass:
        print("GATE_PASS  float64 max-divergence %.3e < %.0e over %d configs x %d seeds x %d steps"
              "  (torch mirror is algebraically faithful to numpy rl_plant, incl. the super-rate"
              " map + slew + transport delay)"
              % (worst64, GATE_TOL, len(CONFIGS), len(SEEDS), T_STEPS))
    else:
        print("GATE_FAIL  float64 max-divergence %.3e >= %.0e" % (worst64, GATE_TOL))


if __name__ == "__main__":
    main()
