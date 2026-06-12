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

INC7 additions (training doctrine 2026-06-12):
  * dr_nominal  -- the FULL DR codepath (dr + dr_aero + dr_mixer + dr_force_bias) with every
    per-env DR tensor pinned at its float64 nominal and the force-bias at zero/empty-bin: the
    DR branch of _step_torch must be ALGEBRAICALLY the scalar plant (the inert-hook negative
    control -- this is the branch training actually runs, and the path the new dr_force_bias
    hook lives on). Gate vs the numpy rl_plant reference like every other config.
  * FORCE_BIAS behavioral check (torch-only self-consistency): with a known bias + bin set
    directly, one step in-bin shifts v by exactly bias*dt and out-of-bin by exactly 0.

Prints per-config DIV_FLOAT64 / DIV_FLOAT32 and an explicit GATE_PASS / GATE_FAIL line.
"""
import numpy as np
import torch
from omegaconf import OmegaConf

from racer.rl_plant import (PlantParams, QUAD_DRAG_C2_MEASURED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            LAPSE_SPEED_MEASURED, LAPSE_FACTOR_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED)
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
_MIXER = dict(mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
              mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
              mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)
_LAPSE = dict(lapse_speed=LAPSE_SPEED_MEASURED.copy(), lapse_factor=LAPSE_FACTOR_MEASURED.copy())

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
    # S17 motor mixer (live-deploy diag 2026-06-11): per-motor clip of collective +- rate
    # differentials -> parasitic-lift mean + Q-scaled slew authority. The big rate commands at
    # the collective end clamps (random_traj's last two steps) drive both rails hard.
    "mixer":      dict(_AERO, **_MAP, **_MIXER),
    "mixer_full": dict(_AERO, **_MAP, **_MIXER, transport_delay_steps=2),
    # S18 airspeed thrust lapse (refit 2026-06-12): a_up *= interp(|vel|, lapse_speed, lapse_factor).
    # random_traj's |v| ~ 3-8 m/s lands in the lapse-active band (L < 1); aero alone + the fully
    # measured plant + lapse, with and without transport delay.
    "lapse":      dict(_AERO, **_LAPSE),
    "lapse_full": dict(_AERO, **_MAP, **_MIXER, **_LAPSE, transport_delay_steps=2),
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
    dyn._lapse_knots = (None if dyn.params.lapse_speed is None else
                        torch.tensor(dyn.params.lapse_speed, device=device, dtype=dtype))
    dyn._lapse_vals = (None if dyn.params.lapse_factor is None else
                       torch.tensor(dyn.params.lapse_factor, device=device, dtype=dtype))
    dyn._mix_rfit = (None if dyn.params.mixer_idle is None else
                     torch.tensor(dyn.params._mixer_r_fit, device=device, dtype=dtype))
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
    # knot-table end-clamp + deep-rail coverage on the LAST FOUR steps: under the delayed
    # configs (transport_delay_steps=2) the applied action at step t is a_{t-2}, so crafting
    # only T-2/T-1 would push the rail actions into the ring buffer and never apply them
    # (S17 review finding) -- T-4/T-3 drain through the buffer, T-2/T-1 cover the k=0 configs.
    u[T_STEPS - 4, :, 0] = 3.8 + 1.2 * torch.rand(n_envs, generator=g)  # collective ~ [1.01, 1.33]
    u[T_STEPS - 3, :, 0] = 0.4 * torch.rand(n_envs, generator=g)        # collective ~ [0, 0.106]
    u[T_STEPS - 2, :, 0] = 3.8 + 1.2 * torch.rand(n_envs, generator=g)
    u[T_STEPS - 1, :, 0] = 0.4 * torch.rand(n_envs, generator=g)
    # body-rate setpoints with tails beyond pi: exercises the map's min(|c|,pi) clamp + the slew
    u[..., 1:] = 1.5 * torch.randn(T_STEPS, n_envs, 3, generator=g)
    # deep mixer-rail coverage (S17): large rate demands AT the collective end clamps -- the
    # (top x rate) headroom collapse and the (bottom x rate) parasitic-lift clip both engage.
    # Changes the gate trajectories for ALL configs (the gate is self-comparative; re-verified).
    u[T_STEPS - 4:, :, 1:] = u[T_STEPS - 4:, :, 1:] * 2.5
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


# ---------------------------------------------------------------------------- INC7 DR-path gate
def build_cfg_dr(n_envs):
    cfg = build_cfg(n_envs)
    cfg.dr = True
    cfg.dr_aero = True
    cfg.dr_mixer = True
    cfg.dr_force_bias = True
    return cfg


def rebuild_dr_nominal(dyn, device, dtype):
    """Pin every per-env DR tensor at its float64 nominal (and the force-bias at zero / empty
    bin) so the DR branch of _step_torch is ALGEBRAICALLY the scalar plant. _dr_mix_rfit is
    taken from params._mixer_r_fit (the cached float64 the scalar path uses), NOT recomputed
    via _t_mixer_r_fit, so the comparison measures the step math only."""
    n, p = dyn.n_envs, dyn.params
    full = lambda val: torch.full((n,), float(val), device=device, dtype=dtype)
    rep = lambda arr: (torch.tensor(np.asarray(arr, dtype=np.float64), device=device,
                                    dtype=dtype).unsqueeze(0).expand(n, *np.shape(arr)).clone())
    dyn._dr_s = rep(np.broadcast_to(p.super_rate_s, (3,)).copy())
    dyn._dr_alpha_max = rep(dyn._alpha_nom)
    dyn._dr_rate_tau = full(p.rate_tau_s)
    dyn._dr_hover = full(p.hover_thrust)
    dyn._dr_drag = full(p.linear_drag if p.quad_drag_c2 is not None else 0.0)
    dyn._dr_c2 = rep(dyn._c2_nom)
    dyn._coll_knots_dr = torch.tensor(dyn._coll_thr_nom, device=device, dtype=dtype)
    dyn._dr_coll_kvals = rep(dyn._coll_kvals_nom)
    dyn._dr_mix_idle = full(dyn._mix_idle_nom)
    dyn._dr_mix_kerr = full(dyn._mix_kerr_nom)
    dyn._dr_mix_khold = full(dyn._mix_khold_nom)
    dyn._dr_mix_zeta = full(dyn._mix_zeta_nom)
    dyn._dr_mix_rfit = rep(p._mixer_r_fit)
    for attr in ("_fb_bias", "_fb_s_lo", "_fb_s_hi", "_fb_c_lo", "_fb_c_hi"):
        setattr(dyn, attr, getattr(dyn, attr).to(dtype))     # zeros: bias off, bin empty


def gate_once_dr(kwargs, dtype, device, seed):
    cfg = build_cfg_dr(N_ENVS)
    dyn = PeregrinePlantDynamics(cfg, device, backend="torch", params=PlantParams(**kwargs))
    rebuild_params(dyn, device, dtype)
    rebuild_dr_nominal(dyn, device, dtype)
    state, u = random_traj(N_ENVS, device, dtype, seed)
    dyn._state = state.clone()
    dyn._thrust = torch.full((N_ENVS,), float(dyn.params.hover_thrust), device=device, dtype=dtype)
    return float(dyn.check_against_rl_plant(u, atol=1e30))


def force_bias_behavior(device) -> float:
    """Torch-only self-consistency of the INC7 force-bias hook: from an identical mid-flight
    state, one step with a known (bias, bin) minus one step with zero bias must shift the NED
    velocity by exactly bias*dt when the state is IN the bin, and by exactly 0 when OUT.
    Returns the max abs error of both checks (float64)."""
    dtype = torch.float64
    kwargs = CONFIGS["mixer"]
    bias_ned = np.array([0.5, -1.0, 2.0])

    def one_step(bias_on, s_lo, s_hi):
        cfg = build_cfg_dr(N_ENVS)
        dyn = PeregrinePlantDynamics(cfg, device, backend="torch", params=PlantParams(**kwargs))
        rebuild_params(dyn, device, dtype)
        rebuild_dr_nominal(dyn, device, dtype)
        state = torch.zeros(N_ENVS, 13, device=device, dtype=dtype)
        state[:, 6] = 1.0                                  # identity attitude (tilt 0 deg)
        state[:, 7] = 2.0                                  # |v| = 2 m/s (z-up x == NED x)
        dyn._state = state
        dyn._thrust = torch.full((N_ENVS,), float(dyn.params.hover_thrust),
                                 device=device, dtype=dtype)
        if bias_on:
            dyn._fb_bias = torch.tensor(bias_ned, device=device,
                                        dtype=dtype).unsqueeze(0).expand(N_ENVS, 3).clone()
            dyn._fb_s_lo = torch.full((N_ENVS,), s_lo, device=device, dtype=dtype)
            dyn._fb_s_hi = torch.full((N_ENVS,), s_hi, device=device, dtype=dtype)
            dyn._fb_c_lo = torch.full((N_ENVS,), float(np.cos(np.radians(15.0))),
                                      device=device, dtype=dtype)
            dyn._fb_c_hi = torch.full((N_ENVS,), 1.0, device=device, dtype=dtype)
        u = torch.zeros(N_ENVS, 4, device=device, dtype=dtype)
        u[:, 0] = 1.0                                      # hover command
        dyn._step_torch(u)
        return dyn._state[:, 7:10].cpu().numpy()           # v in diffaero frame

    v_ref = one_step(False, 0.0, 0.0)
    flip = np.array([1.0, -1.0, -1.0])
    dv_in = (one_step(True, 0.0, 4.0) - v_ref) * flip      # back to NED
    dv_out = (one_step(True, 12.0, 18.0) - v_ref) * flip   # speed 2 not in [12, 18)
    dt = build_cfg(N_ENVS).dt
    err = max(float(np.abs(dv_in - bias_ned * dt).max()), float(np.abs(dv_out).max()))
    return err


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

    # INC7: the DR codepath itself, pinned at nominals (incl. the zeroed force-bias hook) --
    # this is the branch training actually runs; it must be the scalar plant algebraically.
    d64 = max(gate_once_dr(CONFIGS["mixer"], torch.float64, device, s) for s in SEEDS)
    d32 = max(gate_once_dr(CONFIGS["mixer"], torch.float32, device, s) for s in SEEDS)
    ok = d64 < GATE_TOL
    all_pass &= ok
    worst64 = max(worst64, d64)
    print("CONFIG %-10s DIV_FLOAT64 %.3e (%s)   DIV_FLOAT32 %.3e (%s; advisory)"
          % ("dr_nominal", d64, "ok" if ok else "FAIL", d32, "ok" if d32 < F32_TOL else "HIGH"))
    fb_err = force_bias_behavior(device)
    fb_ok = fb_err < 1e-12
    all_pass &= fb_ok
    print("FORCE_BIAS behavioral check: max|err| %.3e (%s; in-bin dv==bias*dt, out-of-bin dv==0)"
          % (fb_err, "ok" if fb_ok else "FAIL"))

    if all_pass:
        print("GATE_PASS  float64 max-divergence %.3e < %.0e over %d configs x %d seeds x %d steps"
              "  (torch mirror is algebraically faithful to numpy rl_plant, incl. the super-rate"
              " map + slew + transport delay)"
              % (worst64, GATE_TOL, len(CONFIGS) + 1, len(SEEDS), T_STEPS))
    else:
        print("GATE_FAIL  float64 max-divergence %.3e >= %.0e" % (worst64, GATE_TOL))


if __name__ == "__main__":
    main()
