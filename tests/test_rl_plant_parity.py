"""Parity: ``racer.rl_plant`` (clean, telemetry-free, batch numpy plant) vs ``racer.twin.CtbrPlant``
(the system-ID'd offline twin, GROUND TRUTH) -- they must integrate IDENTICAL PHYSICS.

Both are driven from the same initial physical state over a battery of random CTBR sequences (hover,
aggressive rates into the norm clamp, swept collective, sustained forward) across the faithful + a
canonical + an actuator-lag + a transport-delay + a super-rate (static amplitude-dependent gain map
+ slew limit) + a super-rate-with-delay + a measured-aero (body-frame sign-split quadratic drag +
convex collective knot table, twin-falsify 2026-06-11) + an everything-ON (aero + map + collective
lag + delay) config, at the live (50 Hz) and twin-course (100 Hz) dt. We compare the TRUE physical state -- twin's internals ``plant.pos/vel/q/omega/_thrust`` (NOT
``plant.state()``, which re-applies the sim's telemetry report-signs that rl_plant deliberately omits).

Also asserts rl_plant's hand-rolled quaternion helpers match ``scipy ... Rotation`` (the math twin.py
delegates to scipy). If anything diverges, the fix goes in ``rl_plant.py`` -- twin.py wins.

Run: ``.venv\\Scripts\\python.exe -m pytest tests/test_rl_plant_parity.py -q``
(add ``-s`` to see the per-case max-divergence report table).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant, CtbrPlantConfig
from racer.twin_fit import faithful_config
from racer import rl_plant as rp

# ----- TIGHT tolerances. The two integrators run identical arithmetic except the quaternion path
# (rl_plant's own helpers vs scipy Rotation in twin.py), so divergence is pure float-rounding that
# accumulates over a rollout. Observed maxima sit far below these (see the printed report).
ATOL_POS = 1e-9       # m      (observed global max over the battery ~1.6e-12)
ATOL_VEL = 1e-9       # m/s    (~3.9e-13)
ATOL_OMEGA = 1e-12    # rad/s  (bit-identical: exactly 0.0 -- same arithmetic)
ATOL_ATT = 1e-11      # rad    (geodesic angle; ~8e-15, i.e. machine precision)
ATOL_THRUST = 1e-12   # normalised collective (bit-identical: exactly 0.0)

_AXES = ("x", "y", "z")


# --------------------------------------------------------------------------- config / state bridges
def _params_from_cfg(cfg: CtbrPlantConfig, dt: float) -> rp.PlantParams:
    """Telemetry-free :class:`rp.PlantParams` from a twin :class:`CtbrPlantConfig` (drop the odo
    report-signs; convert ``cmd_latency_s`` to integer steps at this ``dt``)."""
    nlag = int(round(cfg.cmd_latency_s / dt)) if cfg.cmd_latency_s > 0.0 else 0
    return rp.PlantParams(
        hover_thrust=cfg.hover_thrust,
        g=cfg.g,
        rate_tau_s=cfg.rate_tau_s,
        rate_gain=np.asarray(cfg.rate_gain, float).copy(),
        rate_sign=np.asarray(cfg.rate_sign, float).copy(),
        super_rate_s=None if cfg.super_rate_s is None else np.asarray(cfg.super_rate_s, float).copy(),
        alpha_max_rps2=None if cfg.alpha_max_rps2 is None else np.asarray(cfg.alpha_max_rps2, float).copy(),
        linear_drag=cfg.linear_drag,
        quad_drag_c2=None if cfg.quad_drag_c2 is None else np.asarray(cfg.quad_drag_c2, float).copy(),
        coll_map_thr=None if cfg.coll_map_thr is None else np.asarray(cfg.coll_map_thr, float).copy(),
        coll_map_accel=None if cfg.coll_map_accel is None else np.asarray(cfg.coll_map_accel, float).copy(),
        lapse_speed=None if cfg.lapse_speed is None else np.asarray(cfg.lapse_speed, float).copy(),
        lapse_factor=None if cfg.lapse_factor is None else np.asarray(cfg.lapse_factor, float).copy(),
        mixer_idle=cfg.mixer_idle,
        mixer_kappa_err=cfg.mixer_kappa_err,
        mixer_kappa_hold=cfg.mixer_kappa_hold,
        mixer_zeta_yaw=cfg.mixer_zeta_yaw,
        thrust_tau_s=cfg.thrust_tau_s,
        transport_delay_steps=nlag,
        max_omega_rps=cfg.max_omega_rps,
    )


def _seed_pair(cfg, params, pos, vel, q, omega, thrust):
    """A twin + an rl_plant state seeded to the SAME initial physics. (twin normalises q on init;
    match it. rl_plant's transport buffer is left cold (None) so its first-command warm-up matches
    twin's ``_cmd_buf`` seeding exactly.)"""
    q = np.asarray(q, float)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    plant = CtbrPlant(cfg, position_ned=pos.copy(), velocity_ned=vel.copy(), q_wxyz=q.copy())
    plant.omega = omega.astype(float).copy()
    plant._thrust = float(thrust)
    st = rp.PlantState(
        pos=pos.astype(float).copy(),
        vel=vel.astype(float).copy(),
        quat=q.copy(),
        omega=omega.astype(float).copy(),
        thrust=np.asarray(float(thrust)),
        act_buf=None,
    )
    return plant, st


def _quat_angle(qa, qb) -> float:
    """Geodesic angle (rad) between two wxyz quaternions (sign-agnostic), via the relative
    quaternion + ``arctan2`` -- numerically stable for TINY angles (``2*arccos(dot)`` floors at
    ~3e-8 rad because ``arccos`` near 1.0 amplifies a 1-ULP dot into ``sqrt(2*eps)``; arctan2
    resolves down to machine precision)."""
    qa = np.asarray(qa, float) / max(float(np.linalg.norm(qa)), 1e-12)
    qb = np.asarray(qb, float) / max(float(np.linalg.norm(qb)), 1e-12)
    rel = rp.quat_multiply(rp.quat_conjugate(qa), qb)            # qa^-1 (x) qb
    return float(2.0 * np.arctan2(float(np.linalg.norm(rel[1:4])), abs(float(rel[0]))))


# --------------------------------------------------------------------------- battery
def _configs():
    base = faithful_config()
    canon = CtbrPlantConfig()                                             # unity gain, no drag, tau 0.05
    lag = faithful_config(); lag.thrust_tau_s = 0.05                      # actuator collective lag ON
    delay = faithful_config(); delay.cmd_latency_s = 0.0                  # set per-dt below (3 steps)
    smap = faithful_config(super_rate=True)                               # static gain map + slew ON
    smap_delay = faithful_config(super_rate=True)                         # map + transport delay
    aero = faithful_config(measured_aero=True)                            # quad drag + knot collective
    aero_full = faithful_config(super_rate=True, measured_aero=True)      # everything measured ON
    aero_full.thrust_tau_s = 0.05                                         # + collective lag + delay
    mixer = faithful_config(super_rate=True, measured_aero=True, mixer=True)  # + motor mixer (S17)
    mixer_full = faithful_config(super_rate=True, measured_aero=True, mixer=True)
    mixer_full.thrust_tau_s = 0.05                                        # + collective lag + delay
    lapse = faithful_config(super_rate=True, measured_aero=True, mixer=True, lapse=True)  # + S18 lapse
    lapse_full = faithful_config(super_rate=True, measured_aero=True, mixer=True, lapse=True)
    lapse_full.thrust_tau_s = 0.05                                        # + collective lag + delay
    return {
        "faithful": base,
        "canonical": canon,
        "thrust_lag": lag,
        "transport_delay": delay,                                        # cmd_latency_s patched in the loop
        "super_rate": smap,
        "super_rate_delay": smap_delay,                                  # cmd_latency_s patched in the loop
        "measured_aero": aero,
        "aero_full": aero_full,                                          # cmd_latency_s patched in the loop
        "mixer": mixer,
        "mixer_full": mixer_full,                                        # cmd_latency_s patched in the loop
        "lapse": lapse,
        "lapse_full": lapse_full,                                        # cmd_latency_s patched in the loop
    }


def _sequences(rng: np.random.Generator, n: int = 400) -> dict:
    """A battery of (N,4) CTBR command streams ``[wx, wy, wz, collective]`` covering the flight envelope."""
    hov = 0.2656
    seqs = {}
    # hover wobble: tiny rates, collective jittering around hover
    seqs["hover"] = np.column_stack([rng.normal(0, 0.05, (n, 3)), hov + rng.normal(0, 0.02, n)])
    # aggressive: full body-rate range (controller clamps ~8 rad/s; gain ~2.5 -> target ~20/axis ->
    # |omega| pushes the 25 rad/s norm clamp), collective all over [0.1, 0.6], resampled every step
    seqs["aggressive"] = np.column_stack([rng.uniform(-8, 8, (n, 3)), rng.uniform(0.1, 0.6, n)])
    # smooth-aggressive: large sinusoidal rates (sustained big attitudes), collective swelling
    t = np.arange(n) * 0.01
    seqs["smooth_aggressive"] = np.column_stack([
        6.0 * np.sin(2 * np.pi * np.array([0.7, 1.1, 0.4]) * t[:, None] + np.array([0, 1, 2])),
        hov + 0.25 * np.sin(2 * np.pi * 0.5 * t),
    ])
    # swept collective: small rates, collective ramped 0 -> 0.8 (exercises the thrust map + lag)
    seqs["collective_sweep"] = np.column_stack([
        rng.normal(0, 0.2, (n, 3)), np.linspace(0.0, 0.8, n),
    ])
    # sustained forward: hold a nose-down pitch rate then steady (large attitude + drag terminal vel)
    fwd = np.zeros((n, 4)); fwd[:, 3] = hov
    fwd[:, 1] = np.concatenate([np.full(40, -1.5), np.zeros(n - 40)])     # pitch nose-down then hold
    seqs["sustained_forward"] = fwd
    return seqs


def _initial(rng: np.random.Generator):
    """A randomised but non-degenerate initial physical state."""
    pos = rng.uniform(-5, 5, 3)
    vel = rng.uniform(-3, 3, 3)
    q = rng.normal(0, 1, 4)
    omega = rng.uniform(-1.5, 1.5, 3)
    thrust = float(rng.uniform(0.2, 0.5))
    return pos, vel, q, omega, thrust


_CASES = [(c, dt, s) for c in _configs() for dt in (0.02, 0.01)
          for s in ("hover", "aggressive", "smooth_aggressive", "collective_sweep", "sustained_forward")]

# module-level accumulator so the (optional) report shows global headroom
_REPORT: list[tuple] = []


@pytest.mark.parametrize("cfg_name,dt,seq_name", _CASES)
def test_parity(cfg_name, dt, seq_name):
    cfg = _configs()[cfg_name]
    if cfg_name in ("transport_delay", "super_rate_delay", "aero_full", "mixer_full", "lapse_full"):
        cfg.cmd_latency_s = 3 * dt                                        # exactly 3 steps of delay
    params = _params_from_cfg(cfg, dt)

    rng = np.random.default_rng(hash((cfg_name, seq_name)) % (2**32))
    pos, vel, q, omega, thrust = _initial(rng)
    seq = _sequences(rng)[seq_name]
    plant, st = _seed_pair(cfg, params, pos, vel, q, omega, thrust)

    m_pos = m_vel = m_omega = m_att = m_thrust = 0.0
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

    _REPORT.append((cfg_name, dt, seq_name, m_pos, m_vel, m_omega, m_att, m_thrust))
    msg = (f"[{cfg_name} dt={dt} {seq_name}] max div: pos={m_pos:.2e} vel={m_vel:.2e} "
           f"omega={m_omega:.2e} att={m_att:.2e} thrust={m_thrust:.2e}")
    print(msg)
    assert m_pos < ATOL_POS, msg
    assert m_vel < ATOL_VEL, msg
    assert m_omega < ATOL_OMEGA, msg
    assert m_att < ATOL_ATT, msg
    assert m_thrust < ATOL_THRUST, msg


def test_zzz_report():
    """Print the global max-divergence table (named to sort last). Run with ``-s`` to see it."""
    if not _REPORT:
        pytest.skip("parity cases did not run in this selection")
    gp = max(r[3] for r in _REPORT); gv = max(r[4] for r in _REPORT)
    go = max(r[5] for r in _REPORT); ga = max(r[6] for r in _REPORT); gt = max(r[7] for r in _REPORT)
    print("\n=== rl_plant vs twin -- GLOBAL max divergence over the battery ===")
    print(f"  pos    {gp:.3e} m     (tol {ATOL_POS:.0e})")
    print(f"  vel    {gv:.3e} m/s   (tol {ATOL_VEL:.0e})")
    print(f"  omega  {go:.3e} rad/s (tol {ATOL_OMEGA:.0e})")
    print(f"  att    {ga:.3e} rad   (tol {ATOL_ATT:.0e})")
    print(f"  thrust {gt:.3e}       (tol {ATOL_THRUST:.0e})")
    assert gp < ATOL_POS and gv < ATOL_VEL and go < ATOL_OMEGA and ga < ATOL_ATT and gt < ATOL_THRUST


# --------------------------------------------------------------------------- quaternion helper parity
def _wxyz(xyzw: np.ndarray) -> np.ndarray:
    """scipy as_quat (xyzw) -> wxyz."""
    xyzw = np.asarray(xyzw, float)
    return np.concatenate([xyzw[..., 3:4], xyzw[..., 0:3]], axis=-1)


def _assert_quat_eq(a_wxyz, b_wxyz, atol, what):
    """Compare wxyz quaternions up to global sign (q and -q are the same rotation)."""
    a = a_wxyz / np.linalg.norm(a_wxyz, axis=-1, keepdims=True)
    b = b_wxyz / np.linalg.norm(b_wxyz, axis=-1, keepdims=True)
    sgn = np.sign(np.sum(a * b, axis=-1, keepdims=True))
    sgn = np.where(sgn == 0, 1.0, sgn)
    err = float(np.max(np.abs(a - sgn * b)))
    assert err < atol, f"{what}: max |dq| = {err:.2e} >= {atol:.0e}"


def test_rotvec_to_quat_matches_scipy():
    rng = np.random.default_rng(0)
    rotvecs = rng.normal(0, 1.5, (200, 3))
    rotvecs[:5] = 0.0                                    # exact zero
    rotvecs[5:10] *= 1e-9                                # tiny-angle (sinc path)
    rotvecs[10:15] *= 4.0                                # large angle (> pi)
    mine = rp.rotvec_to_quat(rotvecs)                    # (200,4) batched
    scip = _wxyz(Rotation.from_rotvec(rotvecs).as_quat())
    _assert_quat_eq(mine, scip, 1e-12, "rotvec_to_quat")
    # unit norm
    assert np.allclose(np.linalg.norm(mine, axis=-1), 1.0, atol=1e-12)


def test_quat_multiply_matches_scipy():
    rng = np.random.default_rng(1)
    ra = Rotation.from_rotvec(rng.normal(0, 1.5, (200, 3)))
    rb = Rotation.from_rotvec(rng.normal(0, 1.5, (200, 3)))
    a = _wxyz(ra.as_quat()); b = _wxyz(rb.as_quat())
    mine = rp.quat_multiply(a, b)
    scip = _wxyz((ra * rb).as_quat())                   # rotation composition R(a)@R(b)
    _assert_quat_eq(mine, scip, 1e-12, "quat_multiply")


def test_quat_rotate_matches_scipy():
    rng = np.random.default_rng(2)
    r = Rotation.from_rotvec(rng.normal(0, 1.5, (200, 3)))
    q = _wxyz(r.as_quat())
    v = rng.normal(0, 2.0, (200, 3))
    # forward: R_world_body @ v
    mine = rp.quat_rotate(q, v)
    scip = r.apply(v)
    assert float(np.max(np.abs(mine - scip))) < 1e-12
    # inverse: R.T @ v
    mine_inv = rp.quat_rotate_inverse(q, v)
    scip_inv = r.apply(v, inverse=True)
    assert float(np.max(np.abs(mine_inv - scip_inv))) < 1e-12
    # rotating the body-up axis matches as_matrix() @ [0,0,-1] (the exact thrust-direction call in twin)
    up = rp.quat_rotate(q, rp._BODY_UP)
    mat_up = np.einsum("nij,j->ni", r.as_matrix(), rp._BODY_UP)
    assert float(np.max(np.abs(up - mat_up))) < 1e-12


def test_batch_matches_loop():
    """rl_plant vectorised over a batch == looping it per element (the property DiffAero relies on)."""
    params = rp.PlantParams()
    rng = np.random.default_rng(3)
    B = 16
    pos = rng.uniform(-2, 2, (B, 3)); vel = rng.uniform(-2, 2, (B, 3))
    q = rp.quat_normalize(rng.normal(0, 1, (B, 4)))
    omega = rng.uniform(-1, 1, (B, 3)); thrust = rng.uniform(0.2, 0.5, B)
    batch = rp.PlantState(pos=pos, vel=vel, quat=q, omega=omega, thrust=thrust)
    acts = np.column_stack([rng.uniform(-5, 5, (B, 3)), rng.uniform(0.1, 0.6, B)])
    out = rp.step(batch, acts, 0.02, params)
    for i in range(B):
        si = rp.PlantState(pos=pos[i], vel=vel[i], quat=q[i], omega=omega[i], thrust=np.asarray(thrust[i]))
        oi = rp.step(si, acts[i], 0.02, params)
        assert np.allclose(out.pos[i], oi.pos, atol=1e-12)
        assert np.allclose(out.vel[i], oi.vel, atol=1e-12)
        assert np.allclose(out.quat[i], oi.quat, atol=1e-12)
        assert np.allclose(out.omega[i], oi.omega, atol=1e-12)
