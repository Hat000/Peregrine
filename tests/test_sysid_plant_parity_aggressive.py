"""SYS-ID registration (#4 plant-parity, AGGRESSIVE regime): ``racer.rl_plant`` (numpy) vs
``racer.twin.CtbrPlant`` (scipy, GROUND TRUTH) integrate IDENTICAL physics in the corners the
nominal/random battery in ``tests/test_rl_plant_parity.py`` does not deterministically pin:

  * RATE RAILS -- step commands large enough to pin ``|omega|`` at the 25 rad/s norm clamp and the
    per-axis slew limit (the ``_clip_to_norm`` vs ``_clip_norm`` and slew-clip paths).
  * FULL FLIP -- a sustained single-axis rate that rotates the body through 90 deg into a FULLY
    INVERTED attitude (>135 deg tilt), where the thrust projection ``R(q)@[0,0,-a_up]`` flips sign.
  * PAST-1.0 COLLECTIVE -- collective ramped to 1.4, saturating the convex knot-table END CLAMP
    (``_interp1d`` vs ``np.interp`` beyond the last knot) and the legacy ``g*thr/hover`` map.

Driven open-loop with IDENTICAL deterministic CTBR sequences from a fixed initial state across the
faithful + aero_full (super-rate map + slew + convex collective + quad drag) + mixer_full (+ S17
motor mixer) configs, at the live (50 Hz), course (100 Hz), and on-target (~30 Hz) dt.

REGISTRATION (measured 2026-06-18, handoff/system-id-2026-06-18/scratch/plant-parity/, 45000 steps,
max tilt 179.6 deg, |omega| pinned at 25.000, collective to 1.400):
  omega and thrust are BIT-IDENTICAL (exactly 0.0) in every tilt/rate/collective bin -- the rate
  loop, norm clamp, slew limit, mixer, and thrust lag run identical arithmetic. pos/vel/attitude
  diverge only by float-rounding in the quaternion path (rl_plant's hand-rolled helpers vs scipy
  ``Rotation`` in twin.py): pos <= 1.02e-12 m, vel <= 1.30e-13 m/s, attitude <= 7.9e-15 rad --
  INVARIANT to tilt, rate, and collective. twin.py is ground truth; any divergence above tolerance
  is a bug in rl_plant.py.

Run: ``.venv\\Scripts\\python.exe -m pytest tests/test_sysid_plant_parity_aggressive.py -q``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# mirror the sys.path bootstrap other tests/ use: ensure src/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant, CtbrPlantConfig
from racer.twin_fit import faithful_config
from racer import rl_plant as rp

# TIGHT tolerances. omega/thrust are bit-identical; pos/vel/att are float-rounding only. The bounds
# below sit ~10x above the measured aggressive-battery maxima (pos 1.0e-12, vel 1.3e-13, att 7.9e-15).
ATOL_POS = 1e-10      # m       (observed aggressive max ~1.0e-12)
ATOL_VEL = 1e-11      # m/s     (~1.3e-13)
ATOL_OMEGA = 0.0      # rad/s   (BIT-IDENTICAL -- exactly 0.0, same arithmetic incl. the norm clamp)
ATOL_ATT = 1e-13      # rad     (~7.9e-15, machine precision)
ATOL_THRUST = 0.0     # collective (BIT-IDENTICAL -- exactly 0.0, incl. the past-1.0 end clamp)


# --------------------------------------------------------------------------- config / state bridges
def _params_from_cfg(cfg: CtbrPlantConfig, dt: float) -> rp.PlantParams:
    """Telemetry-free :class:`rp.PlantParams` from a twin config (drop the odo report-signs)."""
    nlag = int(round(cfg.cmd_latency_s / dt)) if cfg.cmd_latency_s > 0.0 else 0
    return rp.PlantParams(
        hover_thrust=cfg.hover_thrust, g=cfg.g, rate_tau_s=cfg.rate_tau_s,
        rate_gain=np.asarray(cfg.rate_gain, float).copy(),
        rate_sign=np.asarray(cfg.rate_sign, float).copy(),
        super_rate_s=None if cfg.super_rate_s is None else np.asarray(cfg.super_rate_s, float).copy(),
        alpha_max_rps2=None if cfg.alpha_max_rps2 is None else np.asarray(cfg.alpha_max_rps2, float).copy(),
        linear_drag=cfg.linear_drag,
        quad_drag_c2=None if cfg.quad_drag_c2 is None else np.asarray(cfg.quad_drag_c2, float).copy(),
        coll_map_thr=None if cfg.coll_map_thr is None else np.asarray(cfg.coll_map_thr, float).copy(),
        coll_map_accel=None if cfg.coll_map_accel is None else np.asarray(cfg.coll_map_accel, float).copy(),
        mixer_idle=cfg.mixer_idle, mixer_kappa_err=cfg.mixer_kappa_err,
        mixer_kappa_hold=cfg.mixer_kappa_hold, mixer_zeta_yaw=cfg.mixer_zeta_yaw,
        thrust_tau_s=cfg.thrust_tau_s, transport_delay_steps=nlag, max_omega_rps=cfg.max_omega_rps,
    )


def _seed_pair(cfg, pos, vel, q, omega, thrust):
    """A twin + an rl_plant state seeded to the SAME initial physics (twin normalises q on init)."""
    q = np.asarray(q, float)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    plant = CtbrPlant(cfg, position_ned=pos.copy(), velocity_ned=vel.copy(), q_wxyz=q.copy())
    plant.omega = omega.astype(float).copy()
    plant._thrust = float(thrust)
    st = rp.PlantState(pos=pos.astype(float).copy(), vel=vel.astype(float).copy(), quat=q.copy(),
                       omega=omega.astype(float).copy(), thrust=np.asarray(float(thrust)), act_buf=None)
    return plant, st


def _quat_angle(qa, qb) -> float:
    """Geodesic angle (rad) between two wxyz quaternions (sign-agnostic), arctan2 form (machine-
    precision for tiny angles)."""
    qa = np.asarray(qa, float) / max(float(np.linalg.norm(qa)), 1e-12)
    qb = np.asarray(qb, float) / max(float(np.linalg.norm(qb)), 1e-12)
    rel = rp.quat_multiply(rp.quat_conjugate(qa), qb)
    return float(2.0 * np.arctan2(float(np.linalg.norm(rel[1:4])), abs(float(rel[0]))))


def _tilt_deg(q) -> float:
    """Tilt of body up (-Z) away from world up, deg (world up is NED -Z)."""
    up_world = rp.quat_rotate(np.asarray(q, float), rp._BODY_UP)
    cos = np.clip(-up_world[2] / max(np.linalg.norm(up_world), 1e-12), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# --------------------------------------------------------------------------- deterministic aggressive battery
_N = 250
_HOV = 0.2656


def _rate_rail_slam() -> np.ndarray:
    """Alternating full +-10 rad/s step commands on all 3 axes (target ~25/axis -> |omega| pinned at
    the 25 norm clamp), collective swung to the rails."""
    s = (-1.0) ** np.arange(_N)
    seq = np.zeros((_N, 4))
    seq[:, 0] = 10.0 * s
    seq[:, 1] = 10.0 * np.roll(s, 1)
    seq[:, 2] = 10.0 * np.roll(s, 2)
    seq[:, 3] = np.where(np.arange(_N) % 2 == 0, 0.05, 0.95)
    return seq


def _sustained_flip() -> np.ndarray:
    """A big single-axis pitch rate that sweeps through 90 deg into fully inverted and keeps rotating."""
    seq = np.zeros((_N, 4)); seq[:, 3] = _HOV; seq[:, 1] = -6.0
    return seq


def _inverted_high_collective() -> np.ndarray:
    """Roll to invert while the collective ramps PAST 1.0 (saturates the knot-table / legacy map)."""
    seq = np.zeros((_N, 4)); seq[:, 0] = 4.0; seq[:, 3] = np.linspace(0.0, 1.4, _N)
    return seq


def _slew_pump() -> np.ndarray:
    """Fast square-wave rate reversals that exercise the per-axis slew clip."""
    seq = np.zeros((_N, 4)); seq[:, 3] = _HOV
    seq[:, 2] = 9.0 * ((np.arange(_N) // 6) % 2 * 2 - 1)
    seq[:, 0] = 9.0 * ((np.arange(_N) // 7) % 2 * 2 - 1)
    return seq


_SEQS = {
    "rate_rail_slam": _rate_rail_slam(),
    "sustained_flip": _sustained_flip(),
    "inverted_high_collective": _inverted_high_collective(),
    "slew_pump": _slew_pump(),
}

# faithful (legacy) + aero_full (map+slew+convex+quad) + mixer_full (+S17 mixer); all pure numpy/scipy
_CFGS = {
    "faithful": lambda: faithful_config(),
    "aero_full": lambda: faithful_config(super_rate=True, measured_aero=True),
    "mixer_full": lambda: faithful_config(super_rate=True, measured_aero=True, mixer=True),
}

# fixed non-degenerate initial state (deterministic -- no RNG, so the pin is reproducible)
_POS0 = np.array([1.5, -2.0, 0.5])
_VEL0 = np.array([6.0, -4.0, 3.0])
_Q0 = np.array([0.8, 0.2, -0.3, 0.1])
_OMEGA0 = np.array([1.0, -1.5, 2.0])
_THRUST0 = 0.30

_CASES = [(c, dt, s) for c in _CFGS for dt in (0.02, 0.01, 0.0333) for s in _SEQS]


@pytest.mark.parametrize("cfg_name,dt,seq_name", _CASES)
def test_aggressive_parity(cfg_name, dt, seq_name):
    cfg = _CFGS[cfg_name]()
    params = _params_from_cfg(cfg, dt)
    seq = _SEQS[seq_name]
    plant, st = _seed_pair(cfg, _POS0, _VEL0, _Q0, _OMEGA0, _THRUST0)

    m_pos = m_vel = m_omega = m_att = m_thrust = 0.0
    max_tilt = max_rate = max_coll = 0.0
    for k in range(seq.shape[0]):
        a = seq[k]
        plant.step(ControlCommand(mode=ControlMode.BODY_RATE, body_rate=a[:3].copy(),
                                  thrust=float(a[3])), dt)
        st = rp.step(st, a, dt, params)
        m_pos = max(m_pos, float(np.max(np.abs(st.pos - plant.pos))))
        m_vel = max(m_vel, float(np.max(np.abs(st.vel - plant.vel))))
        m_omega = max(m_omega, float(np.max(np.abs(st.omega - plant.omega))))
        m_att = max(m_att, _quat_angle(st.quat, plant.q))
        m_thrust = max(m_thrust, abs(float(st.thrust) - float(plant._thrust)))
        max_tilt = max(max_tilt, _tilt_deg(plant.q))
        max_rate = max(max_rate, float(np.linalg.norm(plant.omega)))
        max_coll = max(max_coll, float(plant._thrust))

    msg = (f"[{cfg_name} dt={dt} {seq_name}] tilt<={max_tilt:.0f}deg |w|<={max_rate:.2f} "
           f"coll<={max_coll:.2f} | div pos={m_pos:.2e} vel={m_vel:.2e} omega={m_omega:.2e} "
           f"att={m_att:.2e} thrust={m_thrust:.2e}")
    assert m_omega <= ATOL_OMEGA, "omega NOT bit-identical: " + msg
    assert m_thrust <= ATOL_THRUST, "thrust NOT bit-identical: " + msg
    assert m_pos < ATOL_POS, msg
    assert m_vel < ATOL_VEL, msg
    assert m_att < ATOL_ATT, msg


def test_rate_rail_actually_pins_the_norm_clamp():
    """Guard the guard: the rate_rail_slam sequence must actually drive |omega| onto the 25 rad/s
    norm clamp (otherwise the bit-identity of the clamp path would be vacuously asserted)."""
    cfg = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    plant, _ = _seed_pair(cfg, _POS0, _VEL0, _Q0, _OMEGA0, _THRUST0)
    seq = _SEQS["rate_rail_slam"]
    hits = 0
    for k in range(seq.shape[0]):
        a = seq[k]
        plant.step(ControlCommand(mode=ControlMode.BODY_RATE, body_rate=a[:3].copy(),
                                  thrust=float(a[3])), 0.02)
        if abs(float(np.linalg.norm(plant.omega)) - cfg.max_omega_rps) < 1e-6:
            hits += 1
    # mixer_full throttles rate growth via the slew limit, so the clamp is hit fewer times than the
    # legacy config; >50 of 250 still proves the norm-clamp path is genuinely exercised here.
    assert hits > 50, f"rate_rail_slam pinned the norm clamp only {hits}/{seq.shape[0]} steps"


def test_flip_actually_inverts():
    """Guard the guard: sustained_flip must actually carry the body past 135 deg tilt (fully
    inverted), else the inverted-thrust-projection corner is not exercised."""
    cfg = faithful_config()
    plant, _ = _seed_pair(cfg, _POS0, _VEL0, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), _HOV)
    seq = _SEQS["sustained_flip"]
    max_tilt = 0.0
    for k in range(seq.shape[0]):
        a = seq[k]
        plant.step(ControlCommand(mode=ControlMode.BODY_RATE, body_rate=a[:3].copy(),
                                  thrust=float(a[3])), 0.02)
        max_tilt = max(max_tilt, _tilt_deg(plant.q))
    assert max_tilt > 135.0, f"sustained_flip only reached {max_tilt:.0f} deg tilt (not inverted)"
