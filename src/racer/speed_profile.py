"""Time-optimal speed profiling — the RACE *speed* stage (a TOPP-RA spike).

Geometry (where the line goes) and timing (how fast to fly it) are separable. The planner
ladder's geometry rung is the min-snap line through the ordered gates; THIS module is the
timing rung: given a geometric path + the drone's velocity/acceleration limits, it computes
the *time-optimal* speed profile along that path and emits :class:`Setpoint`s. It therefore
slots behind the very same Setpoint seam the controller already consumes — RACE-mode output,
identical type to :class:`ReactivePlanner`'s.

Why pure numpy instead of the ``toppra`` library
-------------------------------------------------
``toppra`` (the reachability-analysis TOPP) is a Cython/C++ extension that does NOT build on
this Windows + py3.13 box (missing Windows SDK -> ``io.h``; the same build-pain class as
acados — verified 2026-05-31). The forward/backward *numerical-integration* method here gives
the same profile for our smooth gate paths; ``toppra``'s reachability analysis is a robustness
/ convergence upgrade we can drop in on the Linux sim box (the ``[planning]`` extra) behind
this exact API — ``time_optimal_profile`` -> ``Trajectory`` stays the contract.

Model
-----
Isotropic speed cap ``v_max`` and acceleration-magnitude cap ``a_max`` (feed it the
*system-ID'd achievable* limits, not theoretical — the black-box stabilizer's authority is
what bounds us; that is R2 / ``innerloop_step``). At path curvature ``kappa`` the lateral
(centripetal) accel is ``kappa * v**2``, so the tangential budget is
``a_tan <= sqrt(a_max**2 - (kappa v**2)**2)`` — the standard time-optimal coupling (slow into
corners, fast on straights). A forward pass accelerates within that budget up to the ceiling;
a backward pass guarantees we can still brake into every corner and hit the end speed.

Limits of the spike: ``a_max`` is treated as a single magnitude (no separate vertical/thrust
envelope), accel feedforward is finite-differenced (not analytic jerk-limited), and yaw is
interpolated raw (no +-pi unwrap mid-segment). All fine for a baseline + an RL reference line;
revisit when wiring the real RACE follower (post-VQ2 per the plan).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racer.contracts import Setpoint


@dataclass(frozen=True, eq=False)
class Trajectory:
    """A time-parameterised reference path. Sampled arrays + a time query.

    ``setpoint_at(t)`` is the seam: it returns the controller's :class:`Setpoint` (world-NED
    position + velocity + accel feedforward + heading) at any time, clamped to ``[0, duration]``.
    """

    t: np.ndarray        # (N,) strictly increasing seconds, t[0] = 0
    pos: np.ndarray      # (N, 3) world NED
    vel: np.ndarray      # (N, 3) world NED, m/s
    accel: np.ndarray    # (N, 3) world NED, m/s^2 (feedforward)
    yaw: np.ndarray      # (N,) NED heading (north->east), radians

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.pos, axis=0), axis=1).sum())

    def setpoint_at(self, t_query: float) -> Setpoint:
        t = self.t
        tq = float(np.clip(t_query, 0.0, t[-1]))
        i = int(np.searchsorted(t, tq))
        i = min(max(i, 1), len(t) - 1)
        frac = (tq - t[i - 1]) / max(t[i] - t[i - 1], 1e-9)

        def lerp(a: np.ndarray) -> np.ndarray:
            return a[i - 1] + frac * (a[i] - a[i - 1])

        return Setpoint(
            sim_time_ns=0,
            position_ned=lerp(self.pos),
            velocity_ned=lerp(self.vel),
            accel_ned=lerp(self.accel),
            yaw=float(self.yaw[i - 1] + frac * (self.yaw[i] - self.yaw[i - 1])),
        )


def time_optimal_profile(
    waypoints: np.ndarray,
    v_max: float,
    a_max: float,
    v_start: float = 0.0,
    v_end: float = 0.0,
    n_samples: int | None = None,
) -> Trajectory:
    """Time-optimal speed profile through ``waypoints`` (>=2 distinct (x,y,z) NED points).

    Returns a :class:`Trajectory`. ``v_start`` / ``v_end`` are the boundary speeds (m/s).
    """
    from scipy.interpolate import CubicSpline

    wp = np.asarray(waypoints, dtype=np.float64)
    assert wp.ndim == 2 and wp.shape[1] == 3, f"waypoints must be (M,3), got {wp.shape}"
    # Drop consecutive duplicates (zero chord -> spline blows up).
    keep = [0]
    for j in range(1, len(wp)):
        if np.linalg.norm(wp[j] - wp[keep[-1]]) > 1e-6:
            keep.append(j)
    wp = wp[keep]
    if len(wp) < 2:
        raise ValueError("need at least 2 distinct waypoints")

    # Parameterise the spline by cumulative chord length, then sample densely.
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    u_wp = np.concatenate([[0.0], np.cumsum(seg)])
    cs = CubicSpline(u_wp, wp, bc_type="natural")
    if n_samples is None:
        n_samples = max(80, 40 * (len(wp) - 1))
    u = np.linspace(0.0, u_wp[-1], n_samples)

    r = cs(u)                       # (N,3) positions
    d1 = cs(u, 1)                   # dr/du
    d2 = cs(u, 2)                   # d2r/du2
    ds_du = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    tang = d1 / ds_du[:, None]      # unit tangents
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / ds_du**3   # curvature

    # Arc length between consecutive samples (trapezoid in u).
    s = np.concatenate([[0.0], np.cumsum((ds_du[:-1] + ds_du[1:]) / 2 * np.diff(u))])
    seglen = np.diff(s)             # (N-1,)
    N = n_samples

    # Speed ceiling: hard cap + the cornering cap (so the a_tan sqrt stays real).
    v_corner = np.where(kappa > 1e-6, np.sqrt(a_max / np.maximum(kappa, 1e-12)), np.inf)
    v = np.minimum(v_max, v_corner)

    def a_tan(i: int, vi: float) -> float:
        lat = kappa[i] * vi * vi
        return float(np.sqrt(max(0.0, a_max * a_max - lat * lat)))

    # Forward pass: accelerate within budget, capped by the ceiling.
    v[0] = min(v[0], v_start)
    for i in range(N - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * a_tan(i, v[i]) * seglen[i]))
    # Backward pass: guarantee we can brake into each point + hit v_end.
    v[-1] = min(v[-1], v_end)
    for i in range(N - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_tan(i + 1, v[i + 1]) * seglen[i]))

    # Integrate time from the speed profile.
    t = np.zeros(N)
    for i in range(N - 1):
        t[i + 1] = t[i] + seglen[i] / max((v[i] + v[i + 1]) / 2.0, 1e-6)

    vel = v[:, None] * tang
    accel = np.zeros((N, 3))
    accel[1:-1] = (vel[2:] - vel[:-2]) / np.maximum((t[2:] - t[:-2])[:, None], 1e-9)
    accel[0] = (vel[1] - vel[0]) / max(t[1] - t[0], 1e-9)
    accel[-1] = (vel[-1] - vel[-2]) / max(t[-1] - t[-2], 1e-9)
    yaw = np.arctan2(tang[:, 1], tang[:, 0])
    return Trajectory(t=t, pos=r, vel=vel, accel=accel, yaw=yaw)


def waypoints_from_gates(gates) -> np.ndarray:
    """Gate centres -> an (M,3) NED waypoint list, in the given order. The smallest bridge
    from the :class:`~racer.contracts.Gate` map to a path; the min-snap rung will later shape
    the through-gate geometry (entry/exit along each gate normal) — this just stacks centres."""
    return np.asarray([g.position_ned for g in gates], dtype=np.float64)
