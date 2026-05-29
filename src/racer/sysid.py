"""System-identification fitters for the inner-loop / airframe probe.

Pure functions over numpy arrays (no MAVLink, no sim) so they unit-test on synthetic
responses and are reusable offline against recordings. ``innerloop_step.py`` feeds them the
measured step responses; the outputs calibrate the controller's placeholder thrust model
(``Controller.hover_thrust``) and bound how aggressively the planner may demand attitude.
"""
from __future__ import annotations

import numpy as np

# 10-90% rise time of a first-order step response is ln(0.9/0.1) = 2.197 time constants.
_RISE_TAU_RATIO = 2.1972


def fit_hover_thrust(thrusts, up_accels) -> tuple[float, float]:
    """Linear-fit net UPWARD acceleration vs normalized thrust across a thrust sweep.

    Returns ``(hover_thrust, slope)`` where ``hover_thrust`` is the thrust at zero net
    vertical acceleration (the value to seed ``Controller.hover_thrust``) and ``slope`` is
    d(up-accel)/d(thrust) in m/s^2 per unit throttle (its scale + linearity is the throttle
    calibration the controller needs). ``hover_thrust`` is NaN if the sweep is degenerate.
    """
    t = np.asarray(thrusts, dtype=np.float64)
    a = np.asarray(up_accels, dtype=np.float64)
    if t.size < 2 or np.ptp(t) < 1e-9:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(t, a, 1)
    hover = -intercept / slope if abs(slope) > 1e-9 else float("nan")
    return float(hover), float(slope)


def step_response_metrics(t, y, t_step: float, *, settle_frac: float = 0.2) -> dict:
    """Characterise a (possibly noisy) step response ``y(t)`` stepped at ``t_step``.

    Uses the mean of the pre-step samples as the baseline and the mean of the last
    ``settle_frac`` of post-step samples as the steady state, then reports (all seconds
    relative to ``t_step``, NaN where indeterminate):
      delay_s     time for the response to reach 10% of the step (apparent dead time)
      rise_time_s 10% -> 90% rise
      tau_s       first-order time constant estimate (= rise_time / 2.197)
      overshoot   peak overshoot as a fraction of the step (0 if none)
      steady_state, y0, step
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nan = float("nan")
    empty = {"y0": nan, "steady_state": nan, "step": nan, "delay_s": nan,
             "rise_time_s": nan, "tau_s": nan, "overshoot": nan}
    pre = y[t < t_step]
    post = y[t >= t_step]
    tpost = t[t >= t_step]
    if post.size < 3 or pre.size == 0:
        return empty
    y0 = float(np.mean(pre))
    k = max(1, int(post.size * settle_frac))
    yf = float(np.mean(post[-k:]))
    step = yf - y0
    if abs(step) < 1e-9:
        return {**empty, "y0": y0, "steady_state": yf, "step": step}
    norm = (post - y0) / step  # 0 at baseline, 1 at steady state

    def _cross(frac: float) -> float:
        hits = np.flatnonzero(norm >= frac)
        return float(tpost[hits[0]] - t_step) if hits.size else nan

    t10, t90 = _cross(0.1), _cross(0.9)
    rise = (t90 - t10) if (t10 == t10 and t90 == t90) else nan
    tau = rise / _RISE_TAU_RATIO if rise == rise else nan
    overshoot = float(max(0.0, np.max(norm) - 1.0))
    return {
        "y0": y0, "steady_state": yf, "step": step,
        "delay_s": t10, "rise_time_s": rise, "tau_s": tau, "overshoot": overshoot,
    }
