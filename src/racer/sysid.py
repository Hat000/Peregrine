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


def fit_rate_gain(commanded, measured) -> dict:
    """Linear-fit the inner rate loop: measured body rate vs COMMANDED body rate, on one axis.

    For an open-loop body-rate STEP probe (a fixed rate held until the response settles), feed
    the per-step (commanded constant, STEADY measured rate) pairs across both signs / several
    magnitudes. Fits ``measured = gain * commanded + offset`` and returns
    ``{gain, offset, r2, n}``:

      gain    steady-state rate scaling. ``gain ~ 1`` means the sim tracks the commanded rate;
              ``gain > 1`` means it rotates FASTER than commanded (feedforward 1/gain to match);
              ``gain < 0`` means the sim INVERTS this axis (the measured roll+yaw sign flip).
      offset  rate at zero command (a trim/bias; should be ~0 for a clean plant).
      r2      goodness of fit (1 = perfectly linear); low r2 => the steady rate is not a clean
              linear function of the command (saturation / a bad measurement window).

    This is the STEADY-state companion to ``step_response_metrics`` (which reports the transient
    overshoot/tau): together they answer whether an observed command-vs-actual ratio is a steady
    GAIN (shows up here) or a transient OVERSHOOT (shows up there with gain ~ 1). ``gain``/``offset``
    are NaN if the command has no spread (a single magnitude+sign can't separate gain from offset).
    """
    c = np.asarray(commanded, dtype=np.float64)
    m = np.asarray(measured, dtype=np.float64)
    nan = float("nan")
    if c.size < 2 or np.ptp(c) < 1e-9:
        return {"gain": nan, "offset": nan, "r2": nan, "n": int(c.size)}
    gain, offset = (float(v) for v in np.polyfit(c, m, 1))
    resid = m - (gain * c + offset)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((m - np.mean(m)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else nan
    return {"gain": gain, "offset": offset, "r2": r2, "n": int(c.size)}


def fit_thrust_curve(thrusts, up_accels) -> dict:
    """Fit net upward acceleration vs normalized thrust with a QUADRATIC, and report how
    non-linear the plant is.

    The controller's placeholder maps required force to throttle linearly (``hover_thrust *
    f/g``), but a real rotor plant (PWM->RPM->force) is roughly quadratic, so a 2 g demand can
    badly undershoot if the sim does not linearise it -- the drone sags into the bottom of a
    gate on a climb. This characterises the curve so we know whether the linear map is good
    enough or the controller needs a curve / sqrt mapping. Returns ``coeffs`` (a2, a1, a0 for
    ``a = a2 t^2 + a1 t + a0``), ``hover_thrust`` (smallest root in [0,1] where net accel = 0),
    and ``nonlinearity`` = RMS(quadratic residual) / RMS(linear residual): ~1 means linear,
    <<1 means real curvature the linear fit misses. NaNs if the sweep is degenerate.
    """
    t = np.asarray(thrusts, dtype=np.float64)
    a = np.asarray(up_accels, dtype=np.float64)
    nan = float("nan")
    if t.size < 3 or np.ptp(t) < 1e-9:
        return {"coeffs": (nan, nan, nan), "hover_thrust": nan, "nonlinearity": nan}
    a2, a1, a0 = (float(c) for c in np.polyfit(t, a, 2))
    roots = np.roots([a2, a1, a0]) if abs(a2) > 1e-12 else np.array([-a0 / a1] if abs(a1) > 1e-12 else [])
    real = [float(r.real) for r in np.atleast_1d(roots) if abs(r.imag) < 1e-9 and -0.05 <= r.real <= 1.05]
    hover = min(real, key=lambda r: abs(r - 0.5)) if real else nan
    lin_res = a - np.polyval(np.polyfit(t, a, 1), t)
    quad_res = a - np.polyval([a2, a1, a0], t)
    rms = lambda e: float(np.sqrt(np.mean(e**2)))
    lin_rms = rms(lin_res)
    nonlin = (rms(quad_res) / lin_rms) if lin_rms > 1e-9 else nan
    return {"coeffs": (a2, a1, a0), "hover_thrust": hover, "nonlinearity": nonlin}


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
