"""Attitude error metrics for AHRS filter evaluation.

All functions use quaternion (w,x,y,z) scalar-FIRST layout, matching frames.py.

Geodesic error
--------------
The geodesic (minimal-arc) angle between two orientations on SO(3):
    theta = 2 * arccos(|q1 . q2|)
This is the only rotation-invariant, parameter-free attitude error metric.
It avoids gimbal-lock singularities and Euler-angle ambiguities.
Range: [0, pi] radians.

Note: |q1 . q2| is used (not q1.q2) because q and -q represent the same rotation.

Usage
-----
    from racer.ahrs.metrics import geodesic_error_rad, score_filter
    errors = geodesic_error_rad(q_est, q_gt)   # per-timestep errors (N,)
    stats = score_filter(errors)               # dict with mean/p90/max/rms
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def geodesic_error_rad(
    q_est: np.ndarray,
    q_gt: np.ndarray,
) -> np.ndarray:
    """Geodesic (angle-axis) attitude error between estimated and GT quaternions.

    Parameters
    ----------
    q_est : (N, 4) estimated quaternions (w,x,y,z).
    q_gt : (N, 4) ground-truth quaternions (w,x,y,z).

    Returns
    -------
    errors : (N,) geodesic error in radians per timestep.
    """
    q_est = np.asarray(q_est, dtype=np.float64)
    q_gt = np.asarray(q_gt, dtype=np.float64)

    if q_est.ndim == 1:
        q_est = q_est[np.newaxis]
    if q_gt.ndim == 1:
        q_gt = q_gt[np.newaxis]

    # Dot product between each pair; take absolute value (q and -q same rotation)
    dots = np.abs(np.sum(q_est * q_gt, axis=1))
    # Clamp to [0, 1] to avoid arccos domain error from floating-point noise
    dots = np.clip(dots, 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def score_filter(
    errors_rad: np.ndarray,
    skip_init_steps: int = 100,
) -> Dict[str, float]:
    """Summarise per-timestep geodesic errors into scalar statistics.

    Parameters
    ----------
    errors_rad : (N,) geodesic errors in radians.
    skip_init_steps : ignore the first N steps (filter convergence transient).
                      At 200 Hz, 100 steps = 0.5 s warmup.

    Returns
    -------
    dict with keys: mean_deg, p90_deg, max_deg, rms_deg, mean_rad, p90_rad.
    """
    e = errors_rad[skip_init_steps:] if len(errors_rad) > skip_init_steps else errors_rad
    if len(e) == 0:
        return {k: float("nan") for k in ["mean_deg", "p90_deg", "max_deg", "rms_deg", "mean_rad", "p90_rad"]}
    return {
        "mean_deg": float(np.mean(e) * 180 / np.pi),
        "p90_deg": float(np.percentile(e, 90) * 180 / np.pi),
        "max_deg": float(np.max(e) * 180 / np.pi),
        "rms_deg": float(np.sqrt(np.mean(e**2)) * 180 / np.pi),
        "mean_rad": float(np.mean(e)),
        "p90_rad": float(np.percentile(e, 90)),
    }


def run_filter_on_sequence(
    filt,
    seq,
    warmup_q: np.ndarray | None = None,
) -> np.ndarray:
    """Run a filter (ESKF, Madgwick, or Mahony) over an IMUSequence.

    Parameters
    ----------
    filt : filter instance with .reset() and .step(gyro, accel, dt) -> q_wxyz.
    seq : IMUSequence (from imu_gen).
    warmup_q : optional initial quaternion override (w,x,y,z).

    Returns
    -------
    q_est : (N, 4) estimated quaternions (w,x,y,z).
    """
    filt.reset(warmup_q)
    dt = seq.dt
    q_est = np.zeros((seq.N, 4))
    for i in range(seq.N):
        q_est[i] = filt.step(seq.gyro[i], seq.accel[i], dt)
    return q_est
