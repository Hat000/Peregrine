"""Tests for the system-ID fitters (pure; synthetic responses)."""
from __future__ import annotations

import numpy as np
import pytest

from racer.sysid import (
    fit_hover_thrust,
    fit_rate_gain,
    fit_thrust_curve,
    step_response_metrics,
)


def test_fit_hover_thrust_recovers_known_line():
    hover, slope = 0.55, 30.0
    thrusts = [0.35, 0.45, 0.55, 0.65, 0.75]
    accels = [slope * (t - hover) for t in thrusts]   # zero net accel at hover
    h, s = fit_hover_thrust(thrusts, accels)
    assert abs(h - hover) < 1e-6
    assert abs(s - slope) < 1e-6


def test_fit_hover_thrust_degenerate_is_nan():
    h, s = fit_hover_thrust([0.5, 0.5, 0.5], [1.0, 2.0, 3.0])   # no thrust spread
    assert np.isnan(h) and np.isnan(s)
    h2, _ = fit_hover_thrust([0.5], [1.0])                      # single point
    assert np.isnan(h2)


def test_fit_thrust_curve_recovers_quadratic_and_flags_nonlinearity():
    # [red-team 2026-05-30] A real rotor plant is quadratic (PWM->RPM->force). The fitter must
    # recover the curve, find hover, and flag that a linear fit misses real curvature.
    a2, a1, a0 = 40.0, 5.0, -6.0                      # a(t) = 40 t^2 + 5 t - 6
    thrusts = np.linspace(0.2, 0.9, 8)
    accels = a2 * thrusts**2 + a1 * thrusts + a0
    out = fit_thrust_curve(thrusts, accels)
    np.testing.assert_allclose(out["coeffs"], [a2, a1, a0], atol=1e-6)
    root = (-a1 + np.sqrt(a1**2 - 4 * a2 * a0)) / (2 * a2)  # positive root in [0,1]
    assert out["hover_thrust"] == pytest.approx(root, abs=1e-6)
    assert out["nonlinearity"] < 0.2                  # quadratic residual << linear residual


def test_fit_thrust_curve_linear_plant_reports_linear():
    thrusts = np.linspace(0.3, 0.8, 6)
    accels = 30.0 * (thrusts - 0.55)                  # purely linear
    out = fit_thrust_curve(thrusts, accels)
    assert out["hover_thrust"] == pytest.approx(0.55, abs=1e-6)
    # On linear data the quadratic can't beat the linear fit -> ratio ~1 (or NaN if both ~0).
    assert np.isnan(out["nonlinearity"]) or out["nonlinearity"] > 0.5


def test_fit_thrust_curve_degenerate_is_nan():
    out = fit_thrust_curve([0.5, 0.5], [1.0, 2.0])    # <3 points / no spread
    assert np.isnan(out["hover_thrust"])


def test_fit_rate_gain_recovers_steady_gain():
    # Open-loop rate STEP probe: command +/-c, the plant settles to gain*c. The fitter must
    # recover the steady scaling that distinguishes a real gain from a transient overshoot.
    gain = 2.5
    commanded = [0.0, 0.3, -0.3, 0.5, -0.5]
    measured = [gain * c for c in commanded]
    out = fit_rate_gain(commanded, measured)
    assert out["gain"] == pytest.approx(gain, abs=1e-6)
    assert out["offset"] == pytest.approx(0.0, abs=1e-6)
    assert out["r2"] == pytest.approx(1.0, abs=1e-9)


def test_fit_rate_gain_detects_sign_inversion():
    # The sim INVERTS roll + yaw rate commands: a +command yields a NEGATIVE measured rate.
    # The fitter must report a negative gain (the [-1, ...] in body_rate_sign), with the
    # measured-from-command magnitude preserved.
    commanded = [0.0, 0.2, 0.4, -0.2, -0.4]
    measured = [-2.7 * c for c in commanded]
    out = fit_rate_gain(commanded, measured)
    assert out["gain"] == pytest.approx(-2.7, abs=1e-6)
    assert out["gain"] < 0.0


def test_fit_rate_gain_with_offset_and_noise():
    rng = np.random.default_rng(0)
    gain, offset = 1.8, 0.05
    commanded = np.repeat([0.0, 0.25, -0.25, 0.5, -0.5], 4)
    measured = gain * commanded + offset + rng.normal(0, 0.01, commanded.size)
    out = fit_rate_gain(commanded, measured)
    assert out["gain"] == pytest.approx(gain, abs=0.05)
    assert out["offset"] == pytest.approx(offset, abs=0.02)
    assert out["r2"] > 0.99
    assert out["n"] == commanded.size


def test_fit_rate_gain_degenerate_is_nan():
    out = fit_rate_gain([0.3, 0.3, 0.3], [0.8, 0.8, 0.8])   # one magnitude, no spread
    assert np.isnan(out["gain"]) and np.isnan(out["offset"])
    out2 = fit_rate_gain([0.3], [0.8])                       # single point
    assert np.isnan(out2["gain"])


def test_step_metrics_first_order():
    tau, step, t_step = 0.2, 0.5, 1.0
    t = np.arange(0.0, 3.0, 0.01)
    y = np.where(t < t_step, 0.0, step * (1.0 - np.exp(-(t - t_step) / tau)))
    m = step_response_metrics(t, y, t_step=t_step)
    assert abs(m["steady_state"] - step) < 0.02
    assert abs(m["tau_s"] - tau) < 0.05
    assert m["delay_s"] < 0.1          # ~0.105*tau for a clean first-order
    assert m["overshoot"] < 0.05       # first-order does not overshoot


def test_step_metrics_detects_overshoot():
    t = np.array([0.0, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0])
    y = np.array([0.0, 0.0, 0.0, 0.6, 1.2, 1.1, 1.05, 1.0, 1.0, 1.0, 1.0])  # peaks at 1.2, settles 1.0
    m = step_response_metrics(t, y, t_step=1.0)
    assert m["overshoot"] > 0.1
    assert abs(m["steady_state"] - 1.0) < 0.05


def test_step_metrics_flat_input_has_zero_step():
    t = np.arange(0.0, 2.0, 0.01)
    y = np.full_like(t, 5.0)
    m = step_response_metrics(t, y, t_step=1.0)
    assert abs(m["step"]) < 1e-9
    assert np.isnan(m["tau_s"])
